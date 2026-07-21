#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


TRACE_RE = re.compile(
    r"\[pagedmoe-precision-trace\].*?layer=Some\((\d+)\).*?experts=\[([^]]+)\]"
)
LAYERS = 60
TOP_K = 10
PLANES = 4
PLANE_BYTES = 3 * 32 * 16 * 1024
GIB = 1024**3


@dataclass(frozen=True)
class LayerAccess:
    token: int
    layer: int
    experts: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Replay a 500-token expert trace through a 4-bit LRU and pipeline model."
    )
    parser.add_argument("--trace-log", type=Path, default=here / "decode500_server.log")
    parser.add_argument(
        "--calibration-csv",
        type=Path,
        default=here.parent
        / "int4-pipeline-read-compute"
        / "qwen35_397b_int4_pipeline_read_compute_summary.csv",
    )
    parser.add_argument("--decode-tokens", type=int, default=500)
    parser.add_argument("--cache-gib", type=int, nargs="+", default=list(range(4, 65, 4)))
    parser.add_argument("--output-dir", type=Path, default=here)
    return parser.parse_args()


def read_decode_trace(path: Path, decode_tokens: int) -> tuple[list[LayerAccess], int]:
    records: list[tuple[int, tuple[int, ...]]] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = TRACE_RE.search(line)
            if not match:
                continue
            experts = tuple(int(value.strip()) for value in match.group(2).split(","))
            if len(experts) != TOP_K:
                raise ValueError(f"expected {TOP_K} experts, got {len(experts)}")
            records.append((int(match.group(1)), experts))

    # Autoregressive generation emits the first output token from the final
    # prefill position. SGLang logs prefill routes layer-major, then logs the
    # remaining decode steps as ordinary 0..59 layer cycles.
    decode_steps = decode_tokens - 1
    cyclic_records = decode_steps * LAYERS
    if len(records) < cyclic_records + LAYERS:
        raise ValueError(f"trace has {len(records)} records; need decode plus prefill")
    decode_tail = records[-cyclic_records:] if cyclic_records else []
    if [layer for layer, _ in decode_tail] != list(range(LAYERS)) * decode_steps:
        raise ValueError("trace does not end in the expected decode layer cycles")
    prefill = records[:-cyclic_records] if cyclic_records else records
    first_token: list[tuple[int, tuple[int, ...]]] = []
    for layer in range(LAYERS):
        candidates = [record for record in prefill if record[0] == layer]
        if not candidates:
            raise ValueError(f"prefill contains no record for layer {layer}")
        first_token.append(candidates[-1])
    records = first_token + decode_tail
    expected_layers = list(range(LAYERS))
    accesses: list[LayerAccess] = []
    for token in range(decode_tokens):
        group = records[token * LAYERS : (token + 1) * LAYERS]
        layers = [layer for layer, _ in group]
        if layers != expected_layers:
            raise ValueError(f"token {token} has invalid layer sequence: {layers[:5]}...{layers[-5:]}")
        accesses.extend(
            LayerAccess(token, layer, experts) for layer, experts in group
        )
    return accesses, len(records)


def read_calibration(path: Path) -> dict[int, dict[str, float]]:
    measured: dict[int, dict[str, float]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            misses = int(row["miss_experts"])
            wall = float(row["wall_ms"])
            exposed = float(row["read_exposed_ms"])
            overlap = float(row["read_overlap_ms"])
            measured[misses] = {
                "wall_ms": wall,
                "read_exposed_ms": exposed,
                "read_overlap_ms": overlap,
                "tail_compute_ms": max(0.0, wall - exposed - overlap),
                "read_sum_ms": float(row["read_sum_ms"]),
            }
    baseline = next(iter(measured.values()))
    compute_only = float(
        next(csv.DictReader(path.open(newline="")))["compute_only_ms"]
    )
    measured[0] = {
        "wall_ms": compute_only,
        "read_exposed_ms": 0.0,
        "read_overlap_ms": 0.0,
        "tail_compute_ms": compute_only,
        "read_sum_ms": 0.0,
    }
    return measured


def interpolate(measured: dict[int, dict[str, float]], misses: int) -> dict[str, float]:
    if misses in measured:
        return measured[misses]
    lower = max(value for value in measured if value < misses)
    upper = min(value for value in measured if value > misses)
    fraction = (misses - lower) / (upper - lower)
    return {
        key: measured[lower][key] + fraction * (measured[upper][key] - measured[lower][key])
        for key in measured[lower]
    }


def simulate(
    accesses: list[LayerAccess], cache_gib: int, calibration: dict[int, dict[str, float]]
) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    capacity_planes = cache_gib * GIB // PLANE_BYTES
    cache: OrderedDict[tuple[int, int, int], None] = OrderedDict()
    hits = requests = 0
    totals = {key: 0.0 for key in next(iter(calibration.values()))}
    per_token: list[dict[str, float | int]] = []
    token_hits = token_requests = 0
    token_totals = {key: 0.0 for key in totals}

    for index, access in enumerate(accesses):
        missed_experts = 0
        for expert in access.experts:
            expert_missed = False
            for plane in range(PLANES):
                requests += 1
                token_requests += 1
                key = (access.layer, expert, plane)
                if key in cache:
                    hits += 1
                    token_hits += 1
                    cache.move_to_end(key)
                else:
                    expert_missed = True
                    cache[key] = None
                    if len(cache) > capacity_planes:
                        cache.popitem(last=False)
            missed_experts += int(expert_missed)

        costs = interpolate(calibration, missed_experts)
        for key, value in costs.items():
            totals[key] += value
            token_totals[key] += value

        if (index + 1) % LAYERS == 0:
            per_token.append(
                {
                    "cache_gib": cache_gib,
                    "token": access.token,
                    "hit_rate": token_hits / token_requests,
                    **token_totals,
                }
            )
            token_hits = token_requests = 0
            token_totals = {key: 0.0 for key in totals}

    decode_tokens = len(per_token)
    wall_ms = totals["wall_ms"]
    result: dict[str, float | int] = {
        "cache_gib": cache_gib,
        "capacity_planes": capacity_planes,
        "capacity_expert_equivalents": capacity_planes // PLANES,
        "requests": requests,
        "hits": hits,
        "misses": requests - hits,
        "hit_rate": hits / requests,
        **totals,
        "moe_pipeline_token_s": 1000.0 * decode_tokens / wall_ms,
        "wait_read_share": totals["read_exposed_ms"] / wall_ms,
        "overlap_share": totals["read_overlap_ms"] / wall_ms,
        "tail_compute_share": totals["tail_compute_ms"] / wall_ms,
        "ssd_gib_read": (requests - hits) * PLANE_BYTES / GIB,
    }
    return result, per_token


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_trace(path: Path, accesses: list[LayerAccess]) -> None:
    rows = [
        {"token": item.token, "layer": item.layer, "experts": " ".join(map(str, item.experts))}
        for item in accesses
    ]
    write_csv(path, rows)


def plot_results(path: Path, results: list[dict[str, float | int]]) -> None:
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"], "font.size": 11})
    x = [int(row["cache_gib"]) for row in results]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.65))
    axes[0].plot(x, [100 * float(row["hit_rate"]) for row in results], marker="o", color="#D55E00")
    axes[0].set_ylabel("4-bit cache hit rate (%)")
    axes[1].plot(x, [float(row["moe_pipeline_token_s"]) for row in results], marker="o", color="#4C78A8")
    axes[1].set_ylabel("Estimated MoE pipeline (token/s)")
    bottom = [0.0] * len(x)
    for key, label, color in (
        ("wait_read_share", "Wait for SSD", "#9D755D"),
        ("overlap_share", "Read/compute overlap", "#F58518"),
        ("tail_compute_share", "Tail compute", "#4C78A8"),
    ):
        values = [100 * float(row[key]) for row in results]
        axes[2].bar(x, values, bottom=bottom, width=2.8, label=label, color=color)
        bottom = [a + b for a, b in zip(bottom, values)]
    axes[2].set_ylabel("Share of pipeline wall time (%)")
    axes[2].legend(frameon=False, fontsize=7.5, loc="lower right")
    for ax in axes:
        ax.set_xlabel("4-bit cache capacity (GiB)")
        ax.set_xticks([4, 16, 32, 48, 64])
        ax.grid(axis="y", linestyle=(0, (2, 2)), linewidth=0.5, alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.6, w_pad=1.0)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    accesses, records = read_decode_trace(args.trace_log, args.decode_tokens)
    calibration = read_calibration(args.calibration_csv)
    results: list[dict[str, float | int]] = []
    token_rows: list[dict[str, float | int]] = []
    for cache_gib in args.cache_gib:
        result, per_token = simulate(accesses, cache_gib, calibration)
        results.append(result)
        token_rows.extend(per_token)
    write_trace(args.output_dir / "decode500_expert_trace.csv", accesses)
    write_csv(args.output_dir / "simulation_summary.csv", results)
    write_csv(args.output_dir / "simulation_per_token.csv", token_rows)
    plot_results(args.output_dir / "int4_trace_pipeline_motivation", results)
    metadata = {
        "trace_log": str(args.trace_log),
        "calibration_csv": str(args.calibration_csv),
        "decode_tokens": args.decode_tokens,
        "layers": LAYERS,
        "top_k": TOP_K,
        "precision_bits": PLANES,
        "bitplane_bytes": PLANE_BYTES,
        "decode_trace_records": records,
        "expert_accesses": len(accesses) * TOP_K,
        "cache_policy": "global bitplane LRU; codebooks excluded from bitplane capacity",
        "pipeline_model": "per-layer miss-count interpolation from Qwen3.5-397B INT4 O_DIRECT calibration",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for row in results:
        print(
            f"cache={row['cache_gib']:>2} GiB hit={100*float(row['hit_rate']):5.1f}% "
            f"moe_pipeline={float(row['moe_pipeline_token_s']):.2f} token/s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
