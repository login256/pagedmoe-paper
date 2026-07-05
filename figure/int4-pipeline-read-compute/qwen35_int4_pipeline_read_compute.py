#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# This benchmark is for SSD read plus pure CPU expert compute. Hide CUDA before
# importing torch so accidental CUDA dispatch is not possible in the timed path.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import torch.nn.functional as F
from torchao.quantization.utils import unpack_tinygemm_scales_and_zeros


DEFAULT_EXPERT_ROOT = Path(
    "/mnt/test/model/experts/Qwen3.5-35B-A3B/int4/model_layers_0_mlp"
)


@dataclass
class DenseExpert:
    expert_id: int
    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor


@dataclass
class LoadResult:
    expert_id: int
    expert: DenseExpert
    start_ns: int
    end_ns: int
    read_start_ns: int
    read_end_ns: int
    decode_start_ns: int
    decode_end_ns: int
    materialize_start_ns: int
    materialize_end_ns: int
    bytes_read: int


def ns() -> int:
    return time.perf_counter_ns()


def cpulist_to_set(text: str) -> set[int]:
    cpus: set[int] = set()
    for part in text.strip().split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            cpus.update(range(int(lo), int(hi) + 1))
        else:
            cpus.add(int(part))
    return cpus


def bind_numa_node(node: int | None) -> int | None:
    if node is None:
        return None
    cpulist = Path(f"/sys/devices/system/node/node{node}/cpulist")
    if not cpulist.exists():
        raise SystemExit(f"NUMA node {node} does not exist: {cpulist}")
    cpus = cpulist_to_set(cpulist.read_text())
    os.sched_setaffinity(0, cpus)
    return len(cpus)


def drop_file_cache(path: Path) -> None:
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    except OSError:
        pass
    finally:
        os.close(fd)


def expert_path(root: Path, expert_id: int) -> Path:
    return root / f"expert_{expert_id:03d}" / "int4.pt"


def tile_unpack_indices() -> tuple[torch.Tensor, torch.Tensor]:
    rows = torch.arange(8, dtype=torch.long)[:, None]
    cols = torch.arange(128, dtype=torch.long)[None, :]
    word = rows * 16 + (cols // 32) + ((cols % 8) // 2) * 4
    shift = 4 * (((cols % 32) // 8) + 4 * (cols % 2))
    return word.reshape(-1), shift.expand(8, -1).reshape(-1)


_TILE_WORD: torch.Tensor | None = None
_TILE_SHIFT: torch.Tensor | None = None


def dequantize_tile_packed_int4(weight: torch.Tensor) -> torch.Tensor:
    global _TILE_WORD, _TILE_SHIFT
    if _TILE_WORD is None or _TILE_SHIFT is None:
        _TILE_WORD, _TILE_SHIFT = tile_unpack_indices()

    qdata = weight.qdata.contiguous()
    n_blocks, k_blocks, _, _ = qdata.shape
    qflat = qdata.reshape(n_blocks, k_blocks, 128).to(torch.int64)
    qflat = torch.bitwise_and(qflat, 0xFFFFFFFF)
    selected = torch.index_select(qflat, 2, _TILE_WORD)
    shifts = _TILE_SHIFT.view(1, 1, -1)
    quant = torch.bitwise_and(torch.bitwise_right_shift(selected, shifts), 0xF)
    quant = quant.to(torch.float32)
    quant = (
        quant.reshape(n_blocks, k_blocks, 8, 128)
        .permute(0, 2, 1, 3)
        .reshape(n_blocks * 8, k_blocks * 128)
    )

    scale, zero = unpack_tinygemm_scales_and_zeros(weight.scale_and_zero)
    scale = scale.squeeze(-1).to(torch.float32)
    zero = zero.squeeze(-1).to(torch.float32)
    group_size = int(weight.block_size[-1])

    dense = (quant.reshape(quant.shape[0], -1, group_size) - 8.0)
    dense = dense * scale[:, :, None] + zero[:, :, None]
    dense = dense.reshape(quant.shape)
    dense = dense[: weight.shape[0], : weight.shape[1]].contiguous()
    return dense.to(torch.bfloat16)


def materialize_expert(raw: dict[str, torch.Tensor], expert_id: int) -> DenseExpert:
    return DenseExpert(
        expert_id=expert_id,
        gate=dequantize_tile_packed_int4(raw["gate_proj.weight"]),
        up=dequantize_tile_packed_int4(raw["up_proj.weight"]),
        down=dequantize_tile_packed_int4(raw["down_proj.weight"]),
    )


def assert_cpu_experts(experts: dict[int, DenseExpert], x: torch.Tensor) -> None:
    if x.device.type != "cpu":
        raise SystemExit(f"input tensor is not on CPU: {x.device}")
    for expert_id, expert in experts.items():
        for name, tensor in [
            ("gate", expert.gate),
            ("up", expert.up),
            ("down", expert.down),
        ]:
            if tensor.device.type != "cpu":
                raise SystemExit(f"expert {expert_id} {name} tensor is not on CPU: {tensor.device}")


def load_expert(root: Path, expert_id: int, drop_cache: bool) -> LoadResult:
    path = expert_path(root, expert_id)
    if drop_cache:
        drop_file_cache(path)

    start = ns()
    read_start = ns()
    data = path.read_bytes()
    read_end = ns()

    decode_start = ns()
    raw = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
    decode_end = ns()

    materialize_start = ns()
    expert = materialize_expert(raw, expert_id)
    materialize_end = ns()
    end = ns()
    return LoadResult(
        expert_id=expert_id,
        expert=expert,
        start_ns=start,
        end_ns=end,
        read_start_ns=read_start,
        read_end_ns=read_end,
        decode_start_ns=decode_start,
        decode_end_ns=decode_end,
        materialize_start_ns=materialize_start,
        materialize_end_ns=materialize_end,
        bytes_read=len(data),
    )


def compute_expert(expert: DenseExpert, x: torch.Tensor) -> torch.Tensor:
    gate = F.linear(x, expert.gate)
    up = F.linear(x, expert.up)
    return F.linear(F.silu(gate) * up, expert.down)


def merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted((s, e) for s, e in intervals if e > s)
    if not ordered:
        return []
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def interval_duration(intervals: Iterable[tuple[int, int]]) -> int:
    return sum(e - s for s, e in merge_intervals(intervals))


def interval_overlap(a: Iterable[tuple[int, int]], b: Iterable[tuple[int, int]]) -> int:
    aa = merge_intervals(a)
    bb = merge_intervals(b)
    i = j = 0
    total = 0
    while i < len(aa) and j < len(bb):
        start = max(aa[i][0], bb[j][0])
        end = min(aa[i][1], bb[j][1])
        if end > start:
            total += end - start
        if aa[i][1] < bb[j][1]:
            i += 1
        else:
            j += 1
    return total


def ms(delta_ns: int) -> float:
    return delta_ns / 1_000_000.0


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def run_compute_only(
    experts: dict[int, DenseExpert],
    x: torch.Tensor,
    expert_ids: list[int],
    repeats: int,
    warmup: int,
) -> tuple[list[float], float]:
    checksum = 0.0
    with torch.no_grad():
        for _ in range(warmup):
            for expert_id in expert_ids:
                out = compute_expert(experts[expert_id], x)
                checksum += float(out[0, 0])

        samples: list[float] = []
        for _ in range(repeats):
            start = ns()
            for expert_id in expert_ids:
                out = compute_expert(experts[expert_id], x)
                checksum += float(out[0, 0])
            end = ns()
            samples.append(ms(end - start))
    return samples, checksum


def run_pipeline_once(
    root: Path,
    hit_cache: dict[int, DenseExpert],
    hit_ids: list[int],
    miss_ids: list[int],
    x: torch.Tensor,
    loader_workers: int,
    drop_cache: bool,
) -> dict[str, float | int | str]:
    load_results: list[LoadResult] = []
    compute_intervals: list[tuple[int, int]] = []
    checksum = 0.0

    wall_start = ns()
    with ThreadPoolExecutor(max_workers=loader_workers) as pool:
        futures = {
            pool.submit(load_expert, root, expert_id, drop_cache): expert_id
            for expert_id in miss_ids
        }

        with torch.no_grad():
            for expert_id in hit_ids:
                start = ns()
                out = compute_expert(hit_cache[expert_id], x)
                end = ns()
                compute_intervals.append((start, end))
                checksum += float(out[0, 0])

            for future in as_completed(futures):
                result = future.result()
                load_results.append(result)
                start = ns()
                out = compute_expert(result.expert, x)
                end = ns()
                compute_intervals.append((start, end))
                checksum += float(out[0, 0])
    wall_end = ns()

    load_intervals = [(r.start_ns, r.end_ns) for r in load_results]
    read_intervals = [(r.read_start_ns, r.read_end_ns) for r in load_results]
    decode_intervals = [(r.decode_start_ns, r.decode_end_ns) for r in load_results]
    materialize_intervals = [
        (r.materialize_start_ns, r.materialize_end_ns) for r in load_results
    ]

    load_union_ns = interval_duration(load_intervals)
    read_union_ns = interval_duration(read_intervals)
    compute_union_ns = interval_duration(compute_intervals)
    overlap_ns = interval_overlap(load_intervals, compute_intervals)
    read_overlap_ns = interval_overlap(read_intervals, compute_intervals)
    exposed_load_ns = max(0, load_union_ns - overlap_ns)
    exposed_read_ns = max(0, read_union_ns - read_overlap_ns)

    return {
        "wall_ms": ms(wall_end - wall_start),
        "compute_sum_ms": ms(sum(e - s for s, e in compute_intervals)),
        "compute_union_ms": ms(compute_union_ns),
        "load_sum_ms": ms(sum(r.end_ns - r.start_ns for r in load_results)),
        "load_union_ms": ms(load_union_ns),
        "load_read_sum_ms": ms(
            sum(r.read_end_ns - r.read_start_ns for r in load_results)
        ),
        "load_read_union_ms": ms(read_union_ns),
        "load_read_overlap_ms": ms(read_overlap_ns),
        "load_read_exposed_ms": ms(exposed_read_ns),
        "load_decode_sum_ms": ms(
            sum(r.decode_end_ns - r.decode_start_ns for r in load_results)
        ),
        "load_decode_union_ms": ms(interval_duration(decode_intervals)),
        "load_materialize_sum_ms": ms(
            sum(r.materialize_end_ns - r.materialize_start_ns for r in load_results)
        ),
        "load_materialize_union_ms": ms(interval_duration(materialize_intervals)),
        "load_overlap_ms": ms(overlap_ns),
        "load_exposed_ms": ms(exposed_load_ns),
        "bytes_read": sum(r.bytes_read for r in load_results),
        "checksum": checksum,
    }


def pct(value: float, baseline: float) -> float:
    return 100.0 * value / baseline if baseline > 0 else 0.0


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_svg_plot(path: Path, summary_rows: list[dict[str, object]]) -> None:
    x_labels = [str(int(r["hit_experts"])) for r in summary_rows]
    load_union = [float(r["load_union_pct_of_compute"]) for r in summary_rows]
    exposed = [float(r["load_exposed_pct_of_compute"]) for r in summary_rows]
    read = [float(r["ssd_read_pct_of_compute"]) for r in summary_rows]
    rel_mfu = [float(r["relative_mfu_pct"]) for r in summary_rows]
    series = [load_union, exposed, read, rel_mfu]
    ymax = max([100.0, *[v for values in series for v in values]]) * 1.12

    width, height = 760, 460
    left, right, top, bottom = 70, 24, 28, 64
    plot_w = width - left - right
    plot_h = height - top - bottom
    group_w = plot_w / max(1, len(x_labels))
    bar_w = min(34.0, group_w / 5.0)

    def y(value: float) -> float:
        return top + plot_h - (value / ymax) * plot_h

    colors = {
        "load": "#4C78A8",
        "exposed": "#F58518",
        "read": "#54A24B",
        "mfu": "#111111",
        "grid": "#D8DEE9",
    }
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="460" viewBox="0 0 760 460">',
        '<rect width="760" height="460" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;font-size:13px;fill:#222}.small{font-size:12px}.axis{stroke:#333;stroke-width:1}.grid{stroke:#D8DEE9;stroke-width:1}</style>',
    ]
    for tick in range(0, int(ymax) + 1, 50):
        yy = y(float(tick))
        parts.append(f'<line class="grid" x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}"/>')
        parts.append(f'<text class="small" x="{left-10}" y="{yy+4:.1f}" text-anchor="end">{tick}</text>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}"/>')
    parts.append(f'<line class="axis" x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}"/>')

    points: list[tuple[float, float]] = []
    for i, label in enumerate(x_labels):
        cx = left + group_w * (i + 0.5)
        for offset, value, color in [
            (-bar_w, load_union[i], colors["load"]),
            (0.0, exposed[i], colors["exposed"]),
            (bar_w, read[i], colors["read"]),
        ]:
            x0 = cx + offset - bar_w / 2
            yy = y(value)
            parts.append(
                f'<rect x="{x0:.1f}" y="{yy:.1f}" width="{bar_w:.1f}" height="{height-bottom-yy:.1f}" fill="{color}"/>'
            )
        points.append((cx, y(rel_mfu[i])))
        parts.append(f'<text x="{cx:.1f}" y="{height-bottom+24}" text-anchor="middle">{label}</text>')

    if points:
        polyline = " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
        parts.append(f'<polyline points="{polyline}" fill="none" stroke="{colors["mfu"]}" stroke-width="2"/>')
        for px, py in points:
            parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{colors["mfu"]}"/>')

    parts.append(f'<text x="{left + plot_w/2:.1f}" y="{height-18}" text-anchor="middle">hit experts out of 10</text>')
    parts.append(
        f'<text x="18" y="{top + plot_h/2:.1f}" transform="rotate(-90 18 {top + plot_h/2:.1f})" text-anchor="middle">time / compute-only time (%)</text>'
    )
    legend_x, legend_y = left + 12, top + 10
    legend = [
        ("load ready total", colors["load"]),
        ("exposed load", colors["exposed"]),
        ("ssd read", colors["read"]),
        ("relative MFU", colors["mfu"]),
    ]
    for i, (name, color) in enumerate(legend):
        yy = legend_y + i * 20
        if name == "relative MFU":
            parts.append(f'<line x1="{legend_x}" y1="{yy}" x2="{legend_x+22}" y2="{yy}" stroke="{color}" stroke-width="2"/>')
            parts.append(f'<circle cx="{legend_x+11}" cy="{yy}" r="3" fill="{color}"/>')
        else:
            parts.append(f'<rect x="{legend_x}" y="{yy-8}" width="16" height="12" fill="{color}"/>')
        parts.append(f'<text class="small" x="{legend_x+28}" y="{yy+4}">{name}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def maybe_plot(path: Path, summary_rows: list[dict[str, object]]) -> str:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        svg_path = path.with_suffix(".svg")
        write_svg_plot(svg_path, summary_rows)
        convert = shutil.which("convert")
        if convert:
            subprocess.run([convert, str(svg_path), str(path)], check=True)
            return f"{path} and {svg_path} (svg fallback: {exc})"
        return f"{svg_path} (svg fallback: {exc})"

    x = [int(r["hit_experts"]) for r in summary_rows]
    load_union = [float(r["load_union_pct_of_compute"]) for r in summary_rows]
    exposed = [float(r["load_exposed_pct_of_compute"]) for r in summary_rows]
    read = [float(r["ssd_read_pct_of_compute"]) for r in summary_rows]
    rel_mfu = [float(r["relative_mfu_pct"]) for r in summary_rows]

    fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
    width = 0.24
    xs = list(range(len(x)))
    ax1.bar([v - width for v in xs], load_union, width, label="load ready total")
    ax1.bar(xs, exposed, width, label="exposed load")
    ax1.bar([v + width for v in xs], read, width, label="ssd read")
    ax1.set_xticks(xs, [str(v) for v in x])
    ax1.set_xlabel("hit experts out of 10")
    ax1.set_ylabel("time / compute-only time (%)")
    ax1.grid(axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(xs, rel_mfu, marker="o", color="black", label="relative MFU")
    ax2.set_ylabel("relative MFU (%)")
    ax2.set_ylim(bottom=0)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline SSD read and CPU compute benchmark for 10 Qwen3.5 int4 experts."
    )
    parser.add_argument("--expert-root", type=Path, default=DEFAULT_EXPERT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--num-experts", type=int, default=10)
    parser.add_argument("--hit-experts", type=int, nargs="+", default=[2, 4, 6, 8])
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--baseline-repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--loader-workers", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=16)
    parser.add_argument("--numa-node", type=int, default=1)
    parser.add_argument("--no-drop-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bound_cpus = bind_numa_node(args.numa_node)
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)

    root = args.expert_root
    missing = [str(expert_path(root, i)) for i in range(args.num_experts) if not expert_path(root, i).exists()]
    if missing:
        raise SystemExit("missing expert files:\n" + "\n".join(missing))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    expert_ids = list(range(args.num_experts))

    print(f"python={sys.executable}")
    print(f"torch={torch.__version__}")
    print(f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '')!r}")
    print(f"torch_cuda_available={torch.cuda.is_available()}")
    print(f"expert_root={root}")
    print(f"numa_node={args.numa_node} affinity_cpus={bound_cpus}")
    print(f"torch_threads={torch.get_num_threads()} loader_workers={args.loader_workers}")
    print("preloading and materializing 10 int4 experts...")

    load_samples: list[LoadResult] = []
    experts: dict[int, DenseExpert] = {}
    for expert_id in expert_ids:
        result = load_expert(root, expert_id, drop_cache=False)
        load_samples.append(result)
        experts[expert_id] = result.expert

    hidden = experts[0].gate.shape[1]
    x = torch.randn(args.tokens, hidden, dtype=torch.bfloat16)
    assert_cpu_experts(experts, x)

    print("running compute-only baseline...")
    baseline_samples, checksum = run_compute_only(
        experts, x, expert_ids, args.baseline_repeats, args.warmup
    )
    baseline_compute_ms = median(baseline_samples)

    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    print(f"compute-only median for 10 experts = {baseline_compute_ms:.3f} ms (100%)")
    print(
        "hit miss wall% compute% load_ready% exposed_load% "
        "ssd_read% ssd_exposed% rel_mfu%"
    )

    for hit_count in args.hit_experts:
        if hit_count < 0 or hit_count > args.num_experts:
            raise SystemExit(f"invalid hit expert count: {hit_count}")
        hit_ids = expert_ids[:hit_count]
        miss_ids = expert_ids[hit_count:]
        per_repeat: list[dict[str, float | int | str]] = []

        for repeat in range(args.repeats):
            row = run_pipeline_once(
                root=root,
                hit_cache=experts,
                hit_ids=hit_ids,
                miss_ids=miss_ids,
                x=x,
                loader_workers=args.loader_workers,
                drop_cache=not args.no_drop_cache,
            )
            row["hit_experts"] = hit_count
            row["miss_experts"] = len(miss_ids)
            row["repeat"] = repeat
            row["hit_ids"] = " ".join(str(v) for v in hit_ids)
            row["miss_ids"] = " ".join(str(v) for v in miss_ids)
            per_repeat.append(row)
            detail_rows.append(row)
            checksum += float(row["checksum"])

        def med(key: str) -> float:
            return median([float(r[key]) for r in per_repeat])

        summary = {
            "hit_experts": hit_count,
            "miss_experts": len(miss_ids),
            "compute_only_ms": baseline_compute_ms,
            "wall_ms": med("wall_ms"),
            "compute_sum_ms": med("compute_sum_ms"),
            "load_sum_ms": med("load_sum_ms"),
            "load_union_ms": med("load_union_ms"),
            "load_read_sum_ms": med("load_read_sum_ms"),
            "load_read_union_ms": med("load_read_union_ms"),
            "load_read_overlap_ms": med("load_read_overlap_ms"),
            "load_read_exposed_ms": med("load_read_exposed_ms"),
            "load_decode_sum_ms": med("load_decode_sum_ms"),
            "load_materialize_sum_ms": med("load_materialize_sum_ms"),
            "load_overlap_ms": med("load_overlap_ms"),
            "load_exposed_ms": med("load_exposed_ms"),
            "bytes_read": int(median([float(r["bytes_read"]) for r in per_repeat])),
        }
        summary.update(
            {
                "wall_pct_of_compute": pct(float(summary["wall_ms"]), baseline_compute_ms),
                "compute_sum_pct_of_compute": pct(
                    float(summary["compute_sum_ms"]), baseline_compute_ms
                ),
                "load_sum_pct_of_compute": pct(
                    float(summary["load_sum_ms"]), baseline_compute_ms
                ),
                "load_union_pct_of_compute": pct(
                    float(summary["load_union_ms"]), baseline_compute_ms
                ),
                "load_exposed_pct_of_compute": pct(
                    float(summary["load_exposed_ms"]), baseline_compute_ms
                ),
                "ssd_read_pct_of_compute": pct(
                    float(summary["load_read_sum_ms"]), baseline_compute_ms
                ),
                "ssd_read_union_pct_of_compute": pct(
                    float(summary["load_read_union_ms"]), baseline_compute_ms
                ),
                "ssd_read_exposed_pct_of_compute": pct(
                    float(summary["load_read_exposed_ms"]), baseline_compute_ms
                ),
                "ssd_read_overlap_pct_of_compute": pct(
                    float(summary["load_read_overlap_ms"]), baseline_compute_ms
                ),
                "load_overlap_pct_of_compute": pct(
                    float(summary["load_overlap_ms"]), baseline_compute_ms
                ),
                "relative_mfu_pct": pct(
                    baseline_compute_ms, float(summary["wall_ms"])
                ),
            }
        )
        summary_rows.append(summary)
        print(
            f"{hit_count:>3} {len(miss_ids):>4} "
            f"{summary['wall_pct_of_compute']:>6.1f} "
            f"{summary['compute_sum_pct_of_compute']:>8.1f} "
            f"{summary['load_union_pct_of_compute']:>11.1f} "
            f"{summary['load_exposed_pct_of_compute']:>13.1f} "
            f"{summary['ssd_read_pct_of_compute']:>9.1f} "
            f"{summary['ssd_read_exposed_pct_of_compute']:>12.1f} "
            f"{summary['relative_mfu_pct']:>8.1f}"
        )

    detail_csv = args.output_dir / "qwen35_int4_pipeline_read_compute_detail.csv"
    summary_csv = args.output_dir / "qwen35_int4_pipeline_read_compute_summary.csv"
    summary_json = args.output_dir / "qwen35_int4_pipeline_read_compute_summary.json"
    plot_path = args.output_dir / "qwen35_int4_pipeline_read_compute.png"

    write_csv(detail_csv, detail_rows)
    write_csv(summary_csv, summary_rows)
    payload = {
        "config": {
            "expert_root": str(root),
            "num_experts": args.num_experts,
            "hit_experts": args.hit_experts,
            "tokens": args.tokens,
            "repeats": args.repeats,
            "baseline_repeats": args.baseline_repeats,
            "warmup": args.warmup,
            "loader_workers": args.loader_workers,
            "torch_threads": args.torch_threads,
            "numa_node": args.numa_node,
            "drop_cache": not args.no_drop_cache,
            "python": sys.executable,
            "torch": torch.__version__,
        },
        "preload_load_ms_mean": mean([ms(r.end_ns - r.start_ns) for r in load_samples]),
        "preload_read_ms_mean": mean(
            [ms(r.read_end_ns - r.read_start_ns) for r in load_samples]
        ),
        "baseline_compute_ms_samples": baseline_samples,
        "summary": summary_rows,
        "checksum": checksum,
    }
    summary_json.write_text(json.dumps(payload, indent=2) + "\n")
    plot_result = maybe_plot(plot_path, summary_rows)

    print(f"detail_csv={detail_csv}")
    print(f"summary_csv={summary_csv}")
    print(f"summary_json={summary_json}")
    print(f"plot={plot_result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
