#!/usr/bin/env python3
"""Sweep average bit-widths on a fixed layer-30 real-decode hidden-state trace."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch

import run_qwen35_layer30_pagedmoe_dynamic_q4 as q4


DEFAULT_HIDDEN_STATES_PATH = (
    q4.SCRIPT_DIR
    / "qwen35_layer30_real_decode_hidden_states_nonzero_combined_n100.pt"
)
DEFAULT_OUTPUT_STEM = (
    "qwen35_layer30_pagedmoe_dynamic_bitwidth_sweep_real_decode_nonzero_combined_n100"
)
DEFAULT_BITWIDTHS = (3.5, 3.0, 2.5, 2.0)
METHOD_ORDER = ("all", "candidate_oracle", "routing_weight_aware")
METHOD_LABELS = {
    "all": "uniform",
    "candidate_oracle": "Candidate Oracle",
    "routing_weight_aware": "Routing-Weight Aware",
}
METHOD_COLORS = {
    "all": q4.PALETTE["int4"],
    "candidate_oracle": q4.PALETTE["oracle"],
    "routing_weight_aware": q4.PALETTE["ours"],
}
METHOD_HATCHES = {
    "all": "//",
    "candidate_oracle": "\\\\",
    "routing_weight_aware": "",
}


@dataclass(frozen=True)
class BitwidthSpec:
    bitwidth: float
    budget_bits: int
    oracle_min_level: int
    all_levels: tuple[int, ...]
    oracle_candidates: tuple[tuple[int, ...], ...]
    oracle_candidate_tensor: torch.Tensor

    @property
    def key(self) -> str:
        return format_bitwidth(self.bitwidth)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=q4.DEFAULT_MODEL_ROOT)
    parser.add_argument("--storage-root", type=Path, default=q4.DEFAULT_STORAGE_ROOT)
    parser.add_argument("--ktensor-lib-dir", type=Path, default=q4.DEFAULT_KTENSOR_LIB_DIR)
    parser.add_argument("--gain-path", type=Path, default=q4.DEFAULT_GAIN_PATH)
    parser.add_argument("--hidden-states-path", type=Path, default=DEFAULT_HIDDEN_STATES_PATH)
    parser.add_argument("--output-dir", type=Path, default=q4.SCRIPT_DIR)
    parser.add_argument("--output-stem", default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--layer-index", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument(
        "--bitwidths",
        default=",".join(format_bitwidth(value) for value in DEFAULT_BITWIDTHS),
        help="comma-separated average bit-width budgets, e.g. 3.5,3,2.5,2",
    )
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--max-load-bits", type=int, default=10)
    parser.add_argument(
        "--oracle-min-level",
        type=int,
        default=0,
        help="minimum per-rank precision allowed in Candidate Oracle sequences",
    )
    parser.add_argument(
        "--dynamic-cache-mode",
        choices=("retained", "empty-each-sample"),
        default="retained",
    )
    parser.add_argument("--oracle-chunk-size", type=int, default=256)
    return parser.parse_args()


def parse_bitwidths(value: str) -> tuple[float, ...]:
    bitwidths = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not bitwidths:
        raise ValueError("--bitwidths must contain at least one value")
    for bitwidth in bitwidths:
        budget = bitwidth * q4.TOP_K
        if not math.isclose(budget, round(budget), abs_tol=1e-9):
            raise ValueError(f"bitwidth {bitwidth} does not produce an integer top-k budget")
        if bitwidth < 0.0 or bitwidth > q4.MAX_LOGK:
            raise ValueError(f"bitwidth {bitwidth} must be in [0, {q4.MAX_LOGK}]")
    return bitwidths


def format_bitwidth(bitwidth: float) -> str:
    if float(bitwidth).is_integer():
        return str(int(bitwidth))
    return f"{bitwidth:g}"


def bitwidth_label(bitwidth: float) -> str:
    return f"{format_bitwidth(bitwidth)}-bit"


def precision_string(levels: tuple[int, ...] | list[int]) -> str:
    return "/".join(str(level) for level in levels)


def flat_budget_levels(bitwidth: float) -> tuple[int, ...]:
    total = int(round(bitwidth * q4.TOP_K))
    low = total // q4.TOP_K
    high_count = total - low * q4.TOP_K
    return tuple([low + 1] * high_count + [low] * (q4.TOP_K - high_count))


def enumerate_monotone_levels(
    total: int,
    *,
    min_level: int = 0,
) -> tuple[tuple[int, ...], ...]:
    levels: list[tuple[int, ...]] = []

    def rec(prefix: list[int], max_next: int, remaining: int) -> None:
        pos = len(prefix)
        slots_left = q4.TOP_K - pos
        if slots_left == 0:
            if remaining == 0:
                levels.append(tuple(prefix))
            return
        upper = min(max_next, q4.MAX_LOGK, remaining)
        for level in range(upper, min_level - 1, -1):
            new_remaining = remaining - level
            min_remaining = (slots_left - 1) * min_level
            if new_remaining < min_remaining:
                continue
            if new_remaining > (slots_left - 1) * level:
                continue
            prefix.append(level)
            rec(prefix, level, new_remaining)
            prefix.pop()

    rec([], q4.MAX_LOGK, total)
    return tuple(levels)


def make_bitwidth_spec(bitwidth: float, *, oracle_min_level: int = 0) -> BitwidthSpec:
    budget_bits = int(round(bitwidth * q4.TOP_K))
    if budget_bits < oracle_min_level * q4.TOP_K:
        raise ValueError(
            f"bitwidth {bitwidth} cannot satisfy oracle_min_level={oracle_min_level}"
        )
    all_levels = flat_budget_levels(bitwidth)
    if min(all_levels) < oracle_min_level:
        raise ValueError(
            f"all-level baseline {all_levels} is below oracle_min_level={oracle_min_level}"
        )
    candidates = enumerate_monotone_levels(budget_bits, min_level=oracle_min_level)
    if all_levels not in candidates:
        raise AssertionError(f"all-level baseline {all_levels} missing from oracle candidates")
    candidate_tensor = torch.tensor(candidates, dtype=torch.long)
    return BitwidthSpec(
        bitwidth=bitwidth,
        budget_bits=budget_bits,
        oracle_min_level=oracle_min_level,
        all_levels=all_levels,
        oracle_candidates=candidates,
        oracle_candidate_tensor=candidate_tensor,
    )


def combine_rank_outputs(
    weighted_rank_level_outputs: torch.Tensor,
    levels: tuple[int, ...] | list[int],
) -> torch.Tensor:
    rank_indices = torch.arange(q4.TOP_K, dtype=torch.long)
    level_tensor = torch.tensor(levels, dtype=torch.long)
    return weighted_rank_level_outputs[rank_indices, level_tensor].sum(dim=0)


def select_oracle_candidate(
    *,
    reference: torch.Tensor,
    weighted_rank_level_outputs: torch.Tensor,
    candidate_tensor: torch.Tensor,
    chunk_size: int,
) -> tuple[tuple[int, ...], dict[str, float]]:
    rank_indices = torch.arange(q4.TOP_K, dtype=torch.long).unsqueeze(0)
    best_levels: tuple[int, ...] | None = None
    best_metrics: dict[str, float] | None = None
    best_error = float("inf")

    for start in range(0, candidate_tensor.shape[0], chunk_size):
        chunk = candidate_tensor[start : start + chunk_size]
        selected = weighted_rank_level_outputs[rank_indices, chunk]
        outputs = selected.sum(dim=1)
        diffs = outputs.float() - reference.float().unsqueeze(0)
        ref_norm = max(torch.linalg.vector_norm(reference.float()).item(), 1e-12)
        rel_errors = torch.linalg.vector_norm(diffs, dim=1) / ref_norm
        local_index = int(torch.argmin(rel_errors).item())
        local_error = float(rel_errors[local_index].item())
        if local_error < best_error:
            best_error = local_error
            levels = tuple(int(value) for value in chunk[local_index].tolist())
            metrics = q4.relative_metrics(reference, outputs[local_index])
            best_levels = levels
            best_metrics = metrics
        del selected, outputs, diffs, rel_errors

    if best_levels is None or best_metrics is None:
        raise RuntimeError("oracle candidate list is empty")
    return best_levels, best_metrics


def summarize_rows(sample_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary_rows: list[dict[str, object]] = []
    bitwidths = sorted({float(row["bitwidth"]) for row in sample_rows}, reverse=True)
    for bitwidth in bitwidths:
        for strategy in METHOD_ORDER:
            rows = [
                row
                for row in sample_rows
                if float(row["bitwidth"]) == bitwidth and row["strategy"] == strategy
            ]
            rel = [float(row["relative_l2_error"]) for row in rows]
            cos = [float(row["cosine_similarity"]) for row in rows]
            summary_rows.append(
                {
                    "bitwidth": format_bitwidth(bitwidth),
                    "bitwidth_label": bitwidth_label(bitwidth),
                    "strategy": strategy,
                    "label": METHOD_LABELS[strategy],
                    "sample_count": len(rows),
                    "mean_relative_l2_error": statistics.mean(rel),
                    "std_relative_l2_error": statistics.stdev(rel) if len(rel) > 1 else 0.0,
                    "median_relative_l2_error": statistics.median(rel),
                    "min_relative_l2_error": min(rel),
                    "max_relative_l2_error": max(rel),
                    "mean_cosine_similarity": statistics.mean(cos),
                    "mean_target_bits": statistics.mean(
                        float(row["target_bits"]) for row in rows
                    ),
                    "mean_target_average_bits": statistics.mean(
                        float(row["target_average_bits"]) for row in rows
                    ),
                    "all_precision_by_rank": rows[0]["all_precision_by_rank"],
                    "oracle_min_level": rows[0]["oracle_min_level"],
                    "oracle_candidate_count": rows[0]["oracle_candidate_count"],
                }
            )
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows to write to {path}")
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def configure_matplotlib() -> None:
    q4.configure_matplotlib()
    plt.rcParams.update(
        {
            "legend.fontsize": 9,
            "hatch.linewidth": 0.45,
        }
    )


def plot_grouped_summary(path_stem: Path, summary_rows: list[dict[str, object]]) -> None:
    configure_matplotlib()
    bitwidths = []
    for row in summary_rows:
        label = str(row["bitwidth_label"])
        if label not in bitwidths:
            bitwidths.append(label)

    group_count = len(bitwidths)
    x_values = torch.arange(group_count, dtype=torch.float32).numpy()
    bar_width = 0.22
    offsets = {
        "all": -bar_width,
        "candidate_oracle": 0.0,
        "routing_weight_aware": bar_width,
    }

    fig_width = max(5.4, 1.25 * group_count + 1.7)
    fig, ax = plt.subplots(figsize=(fig_width, 2.85))

    row_by_key = {
        (str(row["bitwidth_label"]), str(row["strategy"])): row for row in summary_rows
    }
    y_max = 0.0
    for strategy in METHOD_ORDER:
        y_values = [
            float(row_by_key[(bitwidth, strategy)]["mean_relative_l2_error"]) * 100
            for bitwidth in bitwidths
        ]
        y_errors = [
            float(row_by_key[(bitwidth, strategy)]["std_relative_l2_error"]) * 100
            for bitwidth in bitwidths
        ]
        y_max = max(y_max, max(value + error for value, error in zip(y_values, y_errors)))
        bars = ax.bar(
            x_values + offsets[strategy],
            y_values,
            width=bar_width,
            label=METHOD_LABELS[strategy],
            color=METHOD_COLORS[strategy],
            yerr=y_errors,
            error_kw={
                "ecolor": "#333333",
                "elinewidth": 0.8,
                "capsize": 2.0,
                "capthick": 0.8,
            },
            **q4.BAR_KWARGS,
        )
        hatch = METHOD_HATCHES[strategy]
        for bar in bars:
            bar.set_hatch(hatch)
        for bar, value, error in zip(bars, y_values, y_errors):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + error + max(y_max, 1.0) * 0.025,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.3,
                color="#333333",
            )

    ax.set_ylim(0, max(1.0, y_max * 1.23))
    ax.set_xticks(x_values)
    ax.set_xticklabels(bitwidths)
    ax.set_ylabel("Mean relative output error (%)")
    ax.set_xlabel("Average bit-width budget")
    ax.set_axisbelow(True)
    ax.grid(
        axis="y",
        linestyle=(0, (2, 2)),
        color="#B8C2CC",
        linewidth=0.5,
        alpha=0.6,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.13),
        ncol=3,
        frameon=False,
        handlelength=1.4,
        columnspacing=1.4,
    )
    fig.tight_layout(pad=0.45)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.num_samples < 1:
        raise ValueError("--num-samples must be positive")
    if args.max_load_bits < 0:
        raise ValueError("--max-load-bits must be non-negative")
    if args.oracle_min_level < 0 or args.oracle_min_level > q4.MAX_LOGK:
        raise ValueError(f"--oracle-min-level must be in [0, {q4.MAX_LOGK}]")
    if args.oracle_chunk_size < 1:
        raise ValueError("--oracle-chunk-size must be positive")

    bitwidths = parse_bitwidths(args.bitwidths)
    specs = [
        make_bitwidth_spec(bitwidth, oracle_min_level=args.oracle_min_level)
        for bitwidth in bitwidths
    ]

    torch.set_num_threads(max(1, args.torch_threads))
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    kt = q4.import_ktensor(str(args.ktensor_lib_dir))
    codebook_holder, codebook_view = q4.make_codebook_holder(kt)
    router, router_key, router_path = q4.load_router(args.model_root, args.layer_index)
    all_gains = q4.load_gain_table(args.gain_path)
    layer_gains = all_gains[args.layer_index]
    hidden_states, hidden_state_metadata = q4.load_hidden_states(args.hidden_states_path)
    if args.num_samples > hidden_states.shape[0]:
        raise ValueError(
            f"--num-samples={args.num_samples} exceeds hidden-state count "
            f"{hidden_states.shape[0]} in {args.hidden_states_path}"
        )

    seeds = [args.seed + offset for offset in range(args.num_samples)]
    cached_by_bitwidth = {spec.key: [0] * q4.NUM_EXPERTS for spec in specs}
    sample_rows: list[dict[str, object]] = []
    sample_metadata: list[dict[str, object]] = []

    print(
        f"model={args.model_root} storage={args.storage_root} layer={args.layer_index} "
        f"samples={args.num_samples} bitwidths={[spec.key for spec in specs]} "
        f"hidden_states_path={args.hidden_states_path} cache_mode={args.dynamic_cache_mode}",
        flush=True,
    )
    for spec in specs:
        print(
            f"bitwidth={spec.key} all={precision_string(spec.all_levels)} "
            f"oracle_candidates={len(spec.oracle_candidates)}",
            flush=True,
        )

    with q4.PagedMoeExpertReader(args.storage_root) as reader, q4.ExpertWeightLoader(
        args.model_root, args.layer_index
    ) as expert_loader:
        for sample_index, seed in enumerate(seeds):
            sample_started = time.time()
            hidden_state_bf16 = hidden_states[sample_index].to(
                dtype=torch.bfloat16, device="cpu"
            ).contiguous()
            router_logits, routed_weights, expert_ids_tensor = q4.select_experts(
                router, hidden_state_bf16
            )
            expert_ids = [int(expert_id.item()) for expert_id in expert_ids_tensor]
            route_weights = [float(weight.item()) for weight in routed_weights]

            reference = torch.zeros(q4.HIDDEN_SIZE, dtype=torch.float32)
            weighted_rank_level_outputs = torch.zeros(
                (q4.TOP_K, q4.MAX_LOGK + 1, q4.HIDDEN_SIZE),
                dtype=torch.float32,
            )
            routed_experts: list[dict[str, object]] = []
            for rank, (expert_id, route_weight) in enumerate(
                zip(expert_ids, route_weights), start=1
            ):
                expert_started = time.time()
                gate, up, down = expert_loader.load(expert_id)
                reference_expert = q4.unquantized_expert(
                    gate,
                    up,
                    down,
                    hidden_state_bf16,
                    round_intermediate_to_bf16=True,
                )
                reference.add_(reference_expert, alpha=route_weight)
                for level in range(1, q4.MAX_LOGK + 1):
                    output = q4.quantized_expert(
                        reader=reader,
                        kt=kt,
                        codebook_holder=codebook_holder,
                        codebook_view=codebook_view,
                        layer_index=args.layer_index,
                        expert_id=expert_id,
                        x_bf16=hidden_state_bf16,
                        level=level,
                    )
                    weighted_rank_level_outputs[rank - 1, level].copy_(
                        output.float() * route_weight
                    )
                routed_experts.append(
                    {
                        "rank": rank,
                        "expert_id": expert_id,
                        "routed_weight": route_weight,
                        "elapsed_s": time.time() - expert_started,
                    }
                )
                del gate, up, down, reference_expert, output
                gc.collect()

            print(
                f"sample={sample_index} seed={seed} experts={expert_ids} "
                f"weights={[round(weight, 6) for weight in route_weights]}",
                flush=True,
            )

            sample_bitwidths: list[dict[str, object]] = []
            for spec in specs:
                all_output = combine_rank_outputs(
                    weighted_rank_level_outputs, spec.all_levels
                )
                all_metrics = q4.relative_metrics(reference, all_output)

                oracle_levels, oracle_metrics = select_oracle_candidate(
                    reference=reference,
                    weighted_rank_level_outputs=weighted_rank_level_outputs,
                    candidate_tensor=spec.oracle_candidate_tensor,
                    chunk_size=args.oracle_chunk_size,
                )

                cached_precisions = (
                    [0] * q4.TOP_K
                    if args.dynamic_cache_mode == "empty-each-sample"
                    else [cached_by_bitwidth[spec.key][expert_id] for expert_id in expert_ids]
                )
                dynamic_levels = q4.choose_dynamic_precisions(
                    layer_gains=layer_gains,
                    expert_ids=expert_ids,
                    route_weights=route_weights,
                    cached_precisions=cached_precisions,
                    compute_avg_bits=spec.bitwidth,
                    max_load_bits=args.max_load_bits,
                )
                dynamic_output = combine_rank_outputs(
                    weighted_rank_level_outputs, dynamic_levels
                )
                dynamic_metrics = q4.relative_metrics(reference, dynamic_output)
                if args.dynamic_cache_mode == "retained":
                    for expert_id, target_precision in zip(expert_ids, dynamic_levels):
                        cached_by_bitwidth[spec.key][expert_id] = max(
                            cached_by_bitwidth[spec.key][expert_id], target_precision
                        )

                base_fields = {
                    "sample_index": sample_index,
                    "seed": seed,
                    "bitwidth": spec.key,
                    "bitwidth_label": bitwidth_label(spec.bitwidth),
                    "target_bits": spec.budget_bits,
                    "target_average_bits": spec.bitwidth,
                    "all_precision_by_rank": precision_string(spec.all_levels),
                    "oracle_min_level": spec.oracle_min_level,
                    "oracle_candidate_count": len(spec.oracle_candidates),
                }
                rows = [
                    {
                        **base_fields,
                        "strategy": "all",
                        "label": METHOD_LABELS["all"],
                        "precision_by_rank": precision_string(spec.all_levels),
                        "selected_strategy": "",
                        **all_metrics,
                    },
                    {
                        **base_fields,
                        "strategy": "candidate_oracle",
                        "label": METHOD_LABELS["candidate_oracle"],
                        "precision_by_rank": "per-sample best",
                        "selected_strategy": precision_string(oracle_levels),
                        **oracle_metrics,
                    },
                    {
                        **base_fields,
                        "strategy": "routing_weight_aware",
                        "label": METHOD_LABELS["routing_weight_aware"],
                        "precision_by_rank": precision_string(dynamic_levels),
                        "selected_strategy": "",
                        **dynamic_metrics,
                    },
                ]
                sample_rows.extend(rows)
                sample_bitwidths.append(
                    {
                        "bitwidth": spec.key,
                        "all_precision_by_rank": precision_string(spec.all_levels),
                        "oracle_selected_precision_by_rank": precision_string(oracle_levels),
                        "dynamic_precision_by_rank": precision_string(dynamic_levels),
                        "cached_precisions_before": cached_precisions,
                        "relative_l2_error": {
                            row["strategy"]: row["relative_l2_error"] for row in rows
                        },
                    }
                )
                print(
                    "  "
                    f"{spec.key}-bit "
                    f"all={all_metrics['relative_l2_error'] * 100:.4f}% "
                    f"candidate_oracle={oracle_metrics['relative_l2_error'] * 100:.4f}% "
                    f"routing_weight_aware={dynamic_metrics['relative_l2_error'] * 100:.4f}% "
                    f"oracle_selected={precision_string(oracle_levels)} "
                    f"dynamic={precision_string(dynamic_levels)}",
                    flush=True,
                )
                del all_output, dynamic_output

            sample_metadata.append(
                {
                    "sample_index": sample_index,
                    "seed": seed,
                    "router_logits_l2_norm": torch.linalg.vector_norm(router_logits).item(),
                    "top10_routed_weight_sum": routed_weights.sum().item(),
                    "routed_experts": routed_experts,
                    "bitwidths": sample_bitwidths,
                    "reference_output_l2_norm": torch.linalg.vector_norm(reference).item(),
                    "elapsed_s": time.time() - sample_started,
                }
            )
            del hidden_state_bf16, router_logits, routed_weights, expert_ids_tensor
            del reference, weighted_rank_level_outputs
            gc.collect()

    summary_rows = summarize_rows(sample_rows)
    output_stem = args.output_dir / args.output_stem
    csv_path = output_stem.with_suffix(".csv")
    samples_csv_path = output_stem.with_name(output_stem.name + "_samples.csv")
    metadata_path = output_stem.with_suffix(".json")
    write_csv(csv_path, summary_rows)
    write_csv(samples_csv_path, sample_rows)
    plot_grouped_summary(output_stem, summary_rows)

    metadata = {
        "model_root": str(args.model_root.resolve()),
        "storage_root": str(args.storage_root.resolve()),
        "ktensor_lib_dir": str(args.ktensor_lib_dir.resolve()),
        "gain_path": str(args.gain_path.resolve()),
        "layer_index": args.layer_index,
        "hidden_states": {
            "path": str(args.hidden_states_path.resolve()),
            "sample_count": args.num_samples,
            "source_metadata": hidden_state_metadata,
        },
        "router": {
            "tensor": router_key,
            "shard": str(router_path),
        },
        "bitwidths": [
            {
                "bitwidth": spec.key,
                "budget_bits": spec.budget_bits,
                "oracle_min_level": spec.oracle_min_level,
                "all_precision_by_rank": precision_string(spec.all_levels),
                "oracle_candidate_count": len(spec.oracle_candidates),
                "oracle_candidates": [
                    precision_string(levels) for levels in spec.oracle_candidates
                ],
            }
            for spec in specs
        ],
        "dynamic_precision": {
            "decider": "PagedMoE marginal-gain load/giveup algorithm mirrored from src/pagedmoe-core/src/precision.rs",
            "max_load_bits": args.max_load_bits,
            "cache_mode": args.dynamic_cache_mode,
            "gain": "route_weight^2 * expert_bit_gain[layer][expert][from_precision]",
        },
        "reported_methods": list(METHOD_ORDER),
        "summary_csv": str(csv_path.resolve()),
        "samples_csv": str(samples_csv_path.resolve()),
        "figure_pdf": str(output_stem.with_suffix(".pdf").resolve()),
        "figure_png": str(output_stem.with_suffix(".png").resolve()),
        "samples": sample_metadata,
        "elapsed_s": time.time() - started,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"summary_csv={csv_path}", flush=True)
    print(f"samples_csv={samples_csv_path}", flush=True)
    print(f"metadata_json={metadata_path}", flush=True)
    print(f"figure_pdf={output_stem.with_suffix('.pdf')}", flush=True)
    print(f"figure_png={output_stem.with_suffix('.png')}", flush=True)
    print(f"elapsed_s={time.time() - started:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
