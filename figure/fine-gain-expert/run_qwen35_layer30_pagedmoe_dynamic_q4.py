#!/usr/bin/env python3
"""Compare layer-30 routed output error for uniform Q4, Q4 oracle, and Routing-Weight Aware dynamic precision.

This is the ktensor-lite/PagedMoE-storage counterpart of the older llama.cpp
rank-precision microbenchmark. It uses one Qwen3.5-397B router, the routed
top-10 experts, and deterministic synthetic BF16 hidden states.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from functools import cmp_to_key
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
SUBEXP_DIR = REPO_ROOT / "subexp"
if str(SUBEXP_DIR) not in sys.path:
    sys.path.insert(0, str(SUBEXP_DIR))

from compare_qwen35_expert_ktensor_lite import (  # noqa: E402
    import_ktensor,
    make_codebook_holder,
    quantized_expert,
    unquantized_expert,
)
from pagedmoe_ktensor_lite_io import (  # noqa: E402
    HIDDEN_SIZE,
    MOE_INTERMEDIATE_SIZE,
    PagedMoeExpertReader,
)


NUM_EXPERTS = 512
TOP_K = 10
MAX_LOGK = 8
GAIN_ORDER_EPS = 1e-20
DEFAULT_MODEL_ROOT = Path("/mnt/data/Qwen3.5-397B-A17B")
DEFAULT_STORAGE_ROOT = Path("/mnt/data/qwen35_397b_runtime_full_rust_fixed")
DEFAULT_KTENSOR_LIB_DIR = REPO_ROOT / "src" / "target" / "release"
DEFAULT_GAIN_PATH = (
    REPO_ROOT
    / "subexp/results/"
    "qwen35_bit_gains_relact_l2sq_inf1_layers_000_060_experts_000_512_n10_seed_20260513"
    "_reprocessed_rel_l2_sq_pairavg_inf1/expert_bit_gains_f64.bin"
)
DEFAULT_OUTPUT_STEM = "qwen35_layer30_pagedmoe_dynamic_q4_relative_error"

PALETTE = {
    "ours": "#E45756",
    "strong_baseline": "#4C78A8",
    "oracle": "#54A24B",
    "int4": "#4C78A8",
    "dark_gray": "#555555",
    "light_gray": "#DDDDDD",
    "white": "#FFFFFF",
}
BAR_KWARGS = {"edgecolor": "#333333", "linewidth": 0.8}


@dataclass(frozen=True)
class PrecisionStrategy:
    name: str
    label: str
    levels_by_rank: tuple[int, ...]


ORACLE_Q4_CANDIDATES = (
    PrecisionStrategy(
        "rank_aware_8665432222",
        "Rank-aware 8665432222",
        (8, 6, 6, 5, 4, 3, 2, 2, 2, 2),
    ),
    PrecisionStrategy(
        "rank_aware_6655443322",
        "Rank-aware 6655443322",
        (6, 6, 5, 5, 4, 4, 3, 3, 2, 2),
    ),
    PrecisionStrategy(
        "rank_aware_6555444322",
        "Rank-aware 6555444322",
        (6, 5, 5, 5, 4, 4, 4, 3, 2, 2),
    ),
    PrecisionStrategy(
        "rank_aware_5555444422",
        "Rank-aware 5555444422",
        (5, 5, 5, 5, 4, 4, 4, 4, 2, 2),
    ),
    PrecisionStrategy(
        "rank_aware_5554444333",
        "Rank-aware 5554444333",
        (5, 5, 5, 4, 4, 4, 4, 3, 3, 3),
    ),
    PrecisionStrategy(
        "uniform_q4",
        "uniform",
        (4, 4, 4, 4, 4, 4, 4, 4, 4, 4),
    ),
)


@dataclass(frozen=True)
class GainCandidate:
    weighted_gain: float
    expert_index: int
    from_precision: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--storage-root", type=Path, default=DEFAULT_STORAGE_ROOT)
    parser.add_argument("--ktensor-lib-dir", type=Path, default=DEFAULT_KTENSOR_LIB_DIR)
    parser.add_argument("--gain-path", type=Path, default=DEFAULT_GAIN_PATH)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--output-stem", default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--layer-index", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument(
        "--hidden-states-path",
        type=Path,
        help="optional torch-saved tensor or payload with hidden_states [N, 4096]",
    )
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--compute-avg-bits", type=float, default=4.0)
    parser.add_argument("--max-load-bits", type=int, default=10)
    parser.add_argument(
        "--dynamic-cache-mode",
        choices=("retained", "empty-each-sample"),
        default="retained",
        help=(
            "retained keeps loaded expert precision across samples without eviction; "
            "empty-each-sample passes zero cached precision for every sample"
        ),
    )
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Linux Libertine O", "Libertinus Serif", "STIXGeneral"],
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 0,
            "ytick.major.size": 3,
            "ytick.major.width": 0.7,
            "hatch.linewidth": 0.45,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "figure.facecolor": PALETTE["white"],
            "axes.facecolor": PALETTE["white"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_weight_map(model_root: Path) -> dict[str, str]:
    with (model_root / "model.safetensors.index.json").open("rb") as index_file:
        weight_map = json.load(index_file)["weight_map"]
    return {str(key): str(value) for key, value in weight_map.items()}


def load_router(model_root: Path, layer_index: int) -> tuple[torch.Tensor, str, Path]:
    router_key = f"model.language_model.layers.{layer_index}.mlp.gate.weight"
    weight_map = load_weight_map(model_root)
    router_path = model_root / weight_map[router_key]
    with safe_open(router_path, framework="pt", device="cpu") as weights:
        router = weights.get_tensor(router_key).contiguous()
    if router.dtype != torch.bfloat16:
        raise TypeError(f"expected BF16 router weights, got {router.dtype}")
    if tuple(router.shape) != (NUM_EXPERTS, HIDDEN_SIZE):
        raise ValueError(f"unexpected router shape {tuple(router.shape)}")
    return router, router_key, router_path


class ExpertWeightLoader:
    def __init__(self, model_root: Path, layer_index: int):
        self.model_root = model_root
        self.layer_index = layer_index
        self._gate_up_file = None
        self._down_file = None
        self._gate_up_key = (
            f"model.language_model.layers.{layer_index}.mlp.experts.gate_up_proj"
        )
        self._down_key = (
            f"model.language_model.layers.{layer_index}.mlp.experts.down_proj"
        )

    def __enter__(self) -> "ExpertWeightLoader":
        weight_map = load_weight_map(self.model_root)
        self._gate_up_file = safe_open(
            self.model_root / weight_map[self._gate_up_key],
            framework="pt",
            device="cpu",
        )
        self._down_file = safe_open(
            self.model_root / weight_map[self._down_key],
            framework="pt",
            device="cpu",
        )
        return self

    def __exit__(self, *_: object) -> None:
        if self._gate_up_file is not None:
            self._gate_up_file.__exit__(None, None, None)
        if self._down_file is not None:
            self._down_file.__exit__(None, None, None)

    def load(self, expert_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._gate_up_file is None or self._down_file is None:
            raise RuntimeError("ExpertWeightLoader must be used as a context manager")
        gate_up = self._gate_up_file.get_slice(self._gate_up_key)[expert_id].contiguous()
        down = self._down_file.get_slice(self._down_key)[expert_id].contiguous()
        if tuple(gate_up.shape) != (MOE_INTERMEDIATE_SIZE * 2, HIDDEN_SIZE):
            raise ValueError(f"unexpected gate_up shape {tuple(gate_up.shape)}")
        if tuple(down.shape) != (HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE):
            raise ValueError(f"unexpected down shape {tuple(down.shape)}")
        gate = gate_up[:MOE_INTERMEDIATE_SIZE].contiguous()
        up = gate_up[MOE_INTERMEDIATE_SIZE:].contiguous()
        return gate, up, down


def select_experts(
    router: torch.Tensor, hidden_state_bf16: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    router_logits = torch.mv(router.float(), hidden_state_bf16.float())
    all_routing_weights = F.softmax(router_logits, dim=0, dtype=torch.float32)
    routed_weights, expert_ids = torch.topk(all_routing_weights, TOP_K)
    routed_weights = routed_weights / routed_weights.sum()
    return router_logits, routed_weights, expert_ids


def load_gain_table(path: Path) -> np.ndarray:
    gains = np.fromfile(path, dtype="<f8")
    expected = 60 * NUM_EXPERTS * MAX_LOGK
    if gains.size != expected:
        raise ValueError(f"{path} has {gains.size} f64 entries, expected {expected}")
    return gains.reshape(60, NUM_EXPERTS, MAX_LOGK)


def load_hidden_states(path: Path) -> tuple[torch.Tensor, dict[str, object]]:
    payload = torch.load(path, map_location="cpu")
    metadata: dict[str, object] = {}
    if isinstance(payload, dict) and "hidden_states" in payload:
        hidden_states = payload["hidden_states"]
        raw_metadata = payload.get("metadata", {})
        if isinstance(raw_metadata, dict):
            metadata = dict(raw_metadata)
    elif isinstance(payload, torch.Tensor):
        hidden_states = payload
    else:
        raise TypeError(
            f"{path} must contain a tensor or a dict with a 'hidden_states' tensor"
        )
    if not isinstance(hidden_states, torch.Tensor):
        raise TypeError(f"hidden_states in {path} is not a tensor")
    if hidden_states.ndim != 2 or hidden_states.shape[1] != HIDDEN_SIZE:
        raise ValueError(
            f"hidden_states must have shape [N, {HIDDEN_SIZE}], got {tuple(hidden_states.shape)}"
        )
    return hidden_states.to(dtype=torch.bfloat16, device="cpu").contiguous(), metadata


def compare_candidate_gain(lhs: GainCandidate, rhs: GainCandidate) -> int:
    if lhs.weighted_gain < rhs.weighted_gain - GAIN_ORDER_EPS:
        return -1
    if lhs.weighted_gain > rhs.weighted_gain + GAIN_ORDER_EPS:
        return 1
    if rhs.from_precision < lhs.from_precision:
        return -1
    if rhs.from_precision > lhs.from_precision:
        return 1
    return 0


def compare_load_candidate(lhs: GainCandidate, rhs: GainCandidate) -> int:
    ordering = compare_candidate_gain(lhs, rhs)
    if ordering > 0:
        return -1
    if ordering < 0:
        return 1
    return (lhs.expert_index > rhs.expert_index) - (lhs.expert_index < rhs.expert_index)


def compare_giveup_candidate(lhs: GainCandidate, rhs: GainCandidate) -> int:
    ordering = compare_candidate_gain(lhs, rhs)
    if ordering > 0:
        return 1
    if ordering < 0:
        return -1
    return (lhs.expert_index > rhs.expert_index) - (lhs.expert_index < rhs.expert_index)


def weighted_gain(
    layer_gains: np.ndarray,
    expert_id: int,
    route_weight: float,
    precision: int,
) -> float | None:
    if precision >= MAX_LOGK:
        return None
    gain = float(layer_gains[expert_id, precision])
    weight = float(route_weight) * float(route_weight)
    if math.isnan(gain):
        return 0.0
    if math.isinf(gain) and gain > 0.0:
        return math.inf if math.isfinite(weight) and weight > 0.0 else 0.0
    weighted = weight * gain
    return 0.0 if math.isnan(weighted) else weighted


def choose_dynamic_precisions(
    *,
    layer_gains: np.ndarray,
    expert_ids: list[int],
    route_weights: list[float],
    cached_precisions: list[int],
    compute_avg_bits: float,
    max_load_bits: int,
) -> list[int]:
    target_precisions = list(cached_precisions)
    count = len(expert_ids)
    if count == 0:
        return target_precisions

    compute_bit_budget = int(
        math.floor(compute_avg_bits * count)
    )
    compute_bit_budget = max(0, min(compute_bit_budget, count * MAX_LOGK))
    cached_bits = sum(cached_precisions)
    max_compute_bits = count * MAX_LOGK
    available_load_bits = max_compute_bits - cached_bits

    load_candidates: list[GainCandidate] = []
    giveup_candidates: list[GainCandidate] = []
    for expert_index, (expert_id, route_weight, cached_precision) in enumerate(
        zip(expert_ids, route_weights, cached_precisions)
    ):
        for from_precision in range(cached_precision, MAX_LOGK):
            gain = weighted_gain(layer_gains, expert_id, route_weight, from_precision)
            if gain is not None:
                load_candidates.append(GainCandidate(gain, expert_index, from_precision))
        for from_precision in range(cached_precision):
            gain = weighted_gain(layer_gains, expert_id, route_weight, from_precision)
            if gain is not None:
                giveup_candidates.append(GainCandidate(gain, expert_index, from_precision))

    load_candidates.sort(key=cmp_to_key(compare_load_candidate))
    giveup_candidates.sort(key=cmp_to_key(compare_giveup_candidate))

    start_load_bits = max(compute_bit_budget - cached_bits, 0)
    initial_giveup_bits = cached_bits + start_load_bits - compute_bit_budget
    load_limit = min(
        max(max_load_bits, start_load_bits),
        available_load_bits,
        compute_bit_budget,
    )

    expert_precisions = list(cached_precisions)
    load_cursor = 0
    giveup_cursor = 0

    for _ in range(start_load_bits):
        candidate = load_candidates[load_cursor]
        load_cursor += 1
        if expert_precisions[candidate.expert_index] != candidate.from_precision:
            raise AssertionError("load queue is not precision-prefix ordered")
        expert_precisions[candidate.expert_index] += 1

    for _ in range(initial_giveup_bits):
        candidate = giveup_candidates[giveup_cursor]
        giveup_cursor += 1
        if expert_precisions[candidate.expert_index] != candidate.from_precision + 1:
            raise AssertionError("giveup queue is not precision-suffix ordered")
        expert_precisions[candidate.expert_index] -= 1

    for _load_bits in range(start_load_bits + 1, load_limit + 1):
        if load_cursor >= len(load_candidates) or giveup_cursor >= len(giveup_candidates):
            break
        load_candidate = load_candidates[load_cursor]
        giveup_candidate = giveup_candidates[giveup_cursor]
        if compare_candidate_gain(load_candidate, giveup_candidate) < 0:
            break

        if expert_precisions[load_candidate.expert_index] != load_candidate.from_precision:
            raise AssertionError("load candidate is not the next precision bit")
        giveup_precision_after_load = expert_precisions[
            giveup_candidate.expert_index
        ] + int(giveup_candidate.expert_index == load_candidate.expert_index)
        if giveup_precision_after_load != giveup_candidate.from_precision + 1:
            raise AssertionError("giveup candidate is not the current top precision bit")

        load_cursor += 1
        giveup_cursor += 1
        expert_precisions[load_candidate.expert_index] += 1
        expert_precisions[giveup_candidate.expert_index] -= 1

    return expert_precisions


def relative_metrics(reference: torch.Tensor, output: torch.Tensor) -> dict[str, float]:
    diff = output.float() - reference.float()
    ref_norm = torch.linalg.vector_norm(reference.float()).item()
    out_norm = torch.linalg.vector_norm(output.float()).item()
    diff_norm = torch.linalg.vector_norm(diff).item()
    denom = max(ref_norm, 1e-12)
    return {
        "relative_l2_error": diff_norm / denom,
        "l2_error": diff_norm,
        "max_absolute_error": diff.abs().max().item(),
        "mean_absolute_error": diff.abs().mean().item(),
        "cosine_similarity": F.cosine_similarity(reference.float(), output.float(), dim=0).item(),
        "reference_l2_norm": ref_norm,
        "output_l2_norm": out_norm,
    }


def precision_string(levels: tuple[int, ...] | list[int]) -> str:
    return "/".join(str(level) for level in levels)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows to write to {path}")
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(sample_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    report_order = ["uniform_q4", "oracle_q4", "pagedmoe_dynamic"]
    summary_rows: list[dict[str, object]] = []
    for strategy_name in report_order:
        rows = [row for row in sample_rows if row["strategy"] == strategy_name]
        rel = [float(row["relative_l2_error"]) for row in rows]
        cos = [float(row["cosine_similarity"]) for row in rows]
        summary_rows.append(
            {
                "strategy": strategy_name,
                "label": rows[0]["label"],
                "allocation_level": rows[0]["allocation_level"],
                "precision_by_rank": (
                    "per-sample marginal-gain"
                    if strategy_name == "pagedmoe_dynamic"
                    else rows[0]["precision_by_rank"]
                ),
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
            }
        )
    return summary_rows


def plot_summary(path_stem: Path, summary_rows: list[dict[str, object]]) -> None:
    configure_matplotlib()
    x_values = list(range(len(summary_rows)))
    y_values = [float(row["mean_relative_l2_error"]) * 100 for row in summary_rows]
    y_errors = [float(row["std_relative_l2_error"]) * 100 for row in summary_rows]
    colors = [PALETTE["int4"], PALETTE["oracle"], PALETTE["ours"]]
    hatches = ["//", "\\\\", ""]

    fig, ax = plt.subplots(figsize=(4.6, 2.55))
    bars = ax.bar(
        x_values,
        y_values,
        width=0.58,
        color=colors,
        yerr=y_errors,
        error_kw={
            "ecolor": "#333333",
            "elinewidth": 0.9,
            "capsize": 2.5,
            "capthick": 0.9,
        },
        **BAR_KWARGS,
    )
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    labels = ["uniform", "Candidate\nOracle", "Routing-Weight\nAware"]
    y_max = max(value + err for value, err in zip(y_values, y_errors))
    ax.set_ylim(0, max(1.0, y_max * 1.25))
    ax.set_xticks(x_values)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean relative output error (%)")
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
    for bar, value, error in zip(bars, y_values, y_errors):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + error + y_max * 0.035,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10.5,
            color="#333333",
        )

    fig.tight_layout(pad=0.45)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_sample(
    *,
    sample_index: int,
    seed: int,
    args: argparse.Namespace,
    router: torch.Tensor,
    layer_gains: np.ndarray,
    cached_by_expert: list[int],
    expert_loader: ExpertWeightLoader,
    reader: PagedMoeExpertReader,
    kt,
    codebook_holder,
    codebook_view,
    hidden_state_override: torch.Tensor | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    sample_started = time.time()
    if hidden_state_override is None:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        hidden_state_bf16 = torch.randn(
            HIDDEN_SIZE, generator=generator, dtype=torch.float32
        ).to(torch.bfloat16).contiguous()
    else:
        hidden_state_bf16 = hidden_state_override.to(
            dtype=torch.bfloat16, device="cpu"
        ).contiguous()

    router_logits, routed_weights, expert_ids_tensor = select_experts(router, hidden_state_bf16)
    expert_ids = [int(expert_id.item()) for expert_id in expert_ids_tensor]
    route_weights = [float(weight.item()) for weight in routed_weights]

    if args.dynamic_cache_mode == "empty-each-sample":
        cached_precisions = [0] * TOP_K
    else:
        cached_precisions = [cached_by_expert[expert_id] for expert_id in expert_ids]

    dynamic_levels = choose_dynamic_precisions(
        layer_gains=layer_gains,
        expert_ids=expert_ids,
        route_weights=route_weights,
        cached_precisions=cached_precisions,
        compute_avg_bits=args.compute_avg_bits,
        max_load_bits=args.max_load_bits,
    )

    strategy_levels = {
        strategy.name: list(strategy.levels_by_rank)
        for strategy in ORACLE_Q4_CANDIDATES
    }
    strategy_levels["pagedmoe_dynamic"] = dynamic_levels

    reference = torch.zeros(HIDDEN_SIZE, dtype=torch.float32)
    outputs = {
        strategy_name: torch.zeros(HIDDEN_SIZE, dtype=torch.float32)
        for strategy_name in strategy_levels
    }
    candidate_rows: list[dict[str, object]] = []
    routed_experts: list[dict[str, object]] = []

    print(
        f"sample={sample_index} seed={seed} experts={expert_ids} "
        f"weights={[round(weight, 6) for weight in route_weights]} "
        f"cached={cached_precisions} dynamic={dynamic_levels}",
        flush=True,
    )

    for rank, (expert_id, routed_weight) in enumerate(
        zip(expert_ids, route_weights), start=1
    ):
        expert_started = time.time()
        gate, up, down = expert_loader.load(expert_id)
        reference_expert = unquantized_expert(
            gate, up, down, hidden_state_bf16, round_intermediate_to_bf16=True
        )
        reference.add_(reference_expert, alpha=routed_weight)

        required_levels = sorted(
            {
                levels[rank - 1]
                for levels in strategy_levels.values()
                if levels[rank - 1] > 0
            }
        )
        outputs_by_level: dict[int, torch.Tensor] = {}
        for level in required_levels:
            outputs_by_level[level] = quantized_expert(
                reader=reader,
                kt=kt,
                codebook_holder=codebook_holder,
                codebook_view=codebook_view,
                layer_index=args.layer_index,
                expert_id=expert_id,
                x_bf16=hidden_state_bf16,
                level=level,
            )

        for strategy_name, levels in strategy_levels.items():
            level = levels[rank - 1]
            if level > 0:
                outputs[strategy_name].add_(outputs_by_level[level], alpha=routed_weight)

        routed_experts.append(
            {
                "rank": rank,
                "expert_id": expert_id,
                "routed_weight": routed_weight,
                "required_levels": required_levels,
                "elapsed_s": time.time() - expert_started,
            }
        )
        del gate, up, down, reference_expert, outputs_by_level
        gc.collect()

    for strategy in ORACLE_Q4_CANDIDATES:
        row = {
            "sample_index": sample_index,
            "seed": seed,
            "strategy": strategy.name,
            "label": strategy.label,
            "allocation_level": "routed_rank",
            "precision_by_rank": precision_string(strategy.levels_by_rank),
            "selected_strategy": "",
            "target_bits": sum(strategy.levels_by_rank),
            "target_average_bits": sum(strategy.levels_by_rank) / TOP_K,
        }
        row.update(relative_metrics(reference, outputs[strategy.name]))
        candidate_rows.append(row)

    winner = min(candidate_rows, key=lambda row: float(row["relative_l2_error"]))
    report_rows: list[dict[str, object]] = []
    uniform_row = next(row for row in candidate_rows if row["strategy"] == "uniform_q4")
    report_rows.append(dict(uniform_row))
    oracle_row = dict(winner)
    oracle_row.update(
        {
            "strategy": "oracle_q4",
            "label": "Candidate Oracle",
            "allocation_level": "per_sample_oracle",
            "precision_by_rank": "per-sample best",
            "selected_strategy": winner["strategy"],
        }
    )
    report_rows.append(oracle_row)
    dynamic_row = {
        "sample_index": sample_index,
        "seed": seed,
        "strategy": "pagedmoe_dynamic",
        "label": "Routing-Weight Aware",
        "allocation_level": "runtime_marginal_gain",
        "precision_by_rank": precision_string(dynamic_levels),
        "selected_strategy": "",
        "target_bits": sum(dynamic_levels),
        "target_average_bits": sum(dynamic_levels) / TOP_K,
    }
    dynamic_row.update(relative_metrics(reference, outputs["pagedmoe_dynamic"]))
    report_rows.append(dynamic_row)

    if args.dynamic_cache_mode == "retained":
        for expert_id, target_precision in zip(expert_ids, dynamic_levels):
            cached_by_expert[expert_id] = max(cached_by_expert[expert_id], target_precision)

    print(
        "  "
        + "  ".join(
            f"{row['strategy']}={float(row['relative_l2_error']) * 100:.4f}%"
            for row in report_rows
        )
        + f"  oracle_selected={winner['strategy']}",
        flush=True,
    )

    sample_metadata = {
        "sample_index": sample_index,
        "seed": seed,
        "router_logits_l2_norm": torch.linalg.vector_norm(router_logits).item(),
        "top10_routed_weight_sum": routed_weights.sum().item(),
        "routed_experts": routed_experts,
        "cached_precisions_before": cached_precisions,
        "dynamic_precisions": dynamic_levels,
        "oracle_candidate_rows": candidate_rows,
        "oracle_selected_strategy": winner["strategy"],
        "reference_output_l2_norm": torch.linalg.vector_norm(reference).item(),
        "elapsed_s": time.time() - sample_started,
    }

    del hidden_state_bf16, router_logits, routed_weights, expert_ids_tensor
    del reference, outputs
    gc.collect()
    return report_rows, sample_metadata


def main() -> int:
    args = parse_args()
    if args.num_samples < 1:
        raise ValueError("--num-samples must be positive")
    if not 0 <= args.compute_avg_bits <= MAX_LOGK:
        raise ValueError(f"--compute-avg-bits must be in [0, {MAX_LOGK}]")
    if args.max_load_bits < 0:
        raise ValueError("--max-load-bits must be non-negative")

    torch.set_num_threads(max(1, args.torch_threads))
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    kt = import_ktensor(str(args.ktensor_lib_dir))
    codebook_holder, codebook_view = make_codebook_holder(kt)
    router, router_key, router_path = load_router(args.model_root, args.layer_index)
    all_gains = load_gain_table(args.gain_path)
    layer_gains = all_gains[args.layer_index]
    hidden_states: torch.Tensor | None = None
    hidden_state_metadata: dict[str, object] = {}
    if args.hidden_states_path is not None:
        hidden_states, hidden_state_metadata = load_hidden_states(args.hidden_states_path)
        if args.num_samples > hidden_states.shape[0]:
            raise ValueError(
                f"--num-samples={args.num_samples} exceeds hidden-state count "
                f"{hidden_states.shape[0]} in {args.hidden_states_path}"
            )
    seeds = [args.seed + offset for offset in range(args.num_samples)]
    cached_by_expert = [0] * NUM_EXPERTS

    sample_rows: list[dict[str, object]] = []
    samples: list[dict[str, object]] = []
    print(
        f"model={args.model_root} storage={args.storage_root} layer={args.layer_index} "
        f"samples={args.num_samples} seeds={seeds[0]}..{seeds[-1]} "
        f"dynamic_avg={args.compute_avg_bits} max_load={args.max_load_bits} "
        f"cache_mode={args.dynamic_cache_mode} "
        f"hidden_states_path={args.hidden_states_path}",
        flush=True,
    )

    with PagedMoeExpertReader(args.storage_root) as reader, ExpertWeightLoader(
        args.model_root, args.layer_index
    ) as expert_loader:
        for sample_index, seed in enumerate(seeds):
            rows, sample_metadata = run_sample(
                sample_index=sample_index,
                seed=seed,
                args=args,
                router=router,
                layer_gains=layer_gains,
                cached_by_expert=cached_by_expert,
                expert_loader=expert_loader,
                reader=reader,
                kt=kt,
                codebook_holder=codebook_holder,
                codebook_view=codebook_view,
                hidden_state_override=(
                    None if hidden_states is None else hidden_states[sample_index]
                ),
            )
            sample_rows.extend(rows)
            samples.append(sample_metadata)

    summary_rows = summarize_rows(sample_rows)
    output_stem = args.output_dir / args.output_stem
    csv_path = output_stem.with_suffix(".csv")
    samples_csv_path = output_stem.with_name(output_stem.name + "_samples.csv")
    metadata_path = output_stem.with_suffix(".json")
    write_csv(csv_path, summary_rows)
    write_csv(samples_csv_path, sample_rows)
    plot_summary(output_stem, summary_rows)

    metadata = {
        "model_root": str(args.model_root.resolve()),
        "storage_root": str(args.storage_root.resolve()),
        "ktensor_lib_dir": str(args.ktensor_lib_dir.resolve()),
        "gain_path": str(args.gain_path.resolve()),
        "layer_index": args.layer_index,
        "hidden_states": {
            "description": (
                "loaded external BF16 hidden states"
                if args.hidden_states_path is not None
                else "deterministic synthetic BF16 hidden states sampled from N(0, 1)"
            ),
            "base_seed": args.seed,
            "seeds": seeds,
            "sample_count": args.num_samples,
            "size": HIDDEN_SIZE,
            "path": (
                str(args.hidden_states_path.resolve())
                if args.hidden_states_path is not None
                else None
            ),
            "source_metadata": hidden_state_metadata,
        },
        "router": {
            "tensor": router_key,
            "shard": str(router_path),
            "routing": (
                "FP32 softmax over all 512 logits, top-10 selection, then "
                "renormalization of selected route weights to sum to one"
            ),
        },
        "reference": (
            "FP32 weighted sum of top-10 outputs from original BF16 routed experts; "
            "each expert rounds the gated intermediate activation to BF16 before down_proj"
        ),
        "reported_methods": [
            "uniform_q4",
            "oracle_q4",
            "pagedmoe_dynamic",
        ],
        "oracle_q4_candidates": [
            {
                "strategy": strategy.name,
                "label": strategy.label,
                "precision_by_rank": precision_string(strategy.levels_by_rank),
            }
            for strategy in ORACLE_Q4_CANDIDATES
        ],
        "dynamic_precision": {
            "decider": "PagedMoE marginal-gain load/giveup algorithm mirrored from src/pagedmoe-core/src/precision.rs",
            "compute_avg_bits": args.compute_avg_bits,
            "max_load_bits": args.max_load_bits,
            "cache_mode": args.dynamic_cache_mode,
            "cache_note": (
                "retained mode starts from zero precision and preserves loaded "
                "per-expert precision across these samples without modeling cache eviction"
            ),
            "gain": "route_weight^2 * expert_bit_gain[layer][expert][from_precision]",
        },
        "summary_csv": str(csv_path.resolve()),
        "samples_csv": str(samples_csv_path.resolve()),
        "figure_pdf": str(output_stem.with_suffix(".pdf").resolve()),
        "figure_png": str(output_stem.with_suffix(".png").resolve()),
        "samples": samples,
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
