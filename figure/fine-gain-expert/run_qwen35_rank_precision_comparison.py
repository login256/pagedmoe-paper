#!/usr/bin/env python3
"""Compare rank-based mixed-precision allocations for routed Qwen3.5 MoE tokens.

The script loads one router and one selected expert at a time. It computes the
top-10 routed-expert weighted sum for deterministic synthetic hidden states
with original BF16 weights as the reference, then compares equal-average
nominal-bit allocations. The 1-bit slots use llama.cpp IQ1_M; the other slots
use llama.cpp Qx_K formats.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import resource
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import torch
import torch.nn.functional as F
from safetensors import safe_open


SCRIPT_DIR = Path(__file__).resolve().parent
BIT_ERROR_DIR = SCRIPT_DIR.parent / "bit-error"
if str(BIT_ERROR_DIR) not in sys.path:
    sys.path.insert(0, str(BIT_ERROR_DIR))

from run_qwen35_expert_qk_sweep import (  # noqa: E402
    DEFAULT_LLAMA_CPP_LIB,
    DEFAULT_MODEL_ROOT,
    HIDDEN_SIZE,
    LlamaCppQK,
    QUANT_TYPES,
    load_one_expert,
    metrics,
    quantized_expert as quantized_qk_expert,
    unquantized_expert,
)
from run_qwen35_expert_iq_sweep import (  # noqa: E402
    IQ_TYPES,
    LlamaCppIQ,
    quantized_expert as quantized_iq_expert,
)


NUM_EXPERTS = 512
TOP_K = 10
ORIGINAL_EXPERT_BYTES = 3 * HIDDEN_SIZE * 1024 * torch.bfloat16.itemsize


@dataclass(frozen=True)
class Strategy:
    name: str
    label: str
    formats_by_rank: tuple[str, ...] = ()
    static_formats_by_tier: tuple[str, str] | None = None


IQ1M_STRATEGIES = (
    Strategy(
        "rank_aware_iq1m",
        "Rank-aware",
        ("Q8_K", "Q6_K", "Q4_K", "Q3_K", "Q2_K", "Q2_K", "Q2_K", "IQ1_M", "IQ1_M", "IQ1_M"),
    ),
    Strategy(
        "q4_q2_split",
        "Q4/Q2 split",
        ("Q4_K", "Q4_K", "Q4_K", "Q4_K", "Q4_K", "Q2_K", "Q2_K", "Q2_K", "Q2_K", "Q2_K"),
    ),
    Strategy(
        "uniform_q3",
        "Uniform Q3",
        ("Q3_K", "Q3_K", "Q3_K", "Q3_K", "Q3_K", "Q3_K", "Q3_K", "Q3_K", "Q3_K", "Q3_K"),
    ),
)
QK654_STRATEGIES = (
    Strategy(
        "rank_aware_qk654",
        "Rank-aware",
        ("Q6_K", "Q5_K", "Q4_K", "Q3_K", "Q2_K", "Q2_K", "Q2_K", "Q2_K", "Q2_K", "Q2_K"),
    ),
    Strategy(
        "q4_q2_split",
        "Q4/Q2 split",
        ("Q4_K", "Q4_K", "Q4_K", "Q4_K", "Q4_K", "Q2_K", "Q2_K", "Q2_K", "Q2_K", "Q2_K"),
    ),
    Strategy(
        "uniform_q3",
        "Uniform Q3",
        ("Q3_K", "Q3_K", "Q3_K", "Q3_K", "Q3_K", "Q3_K", "Q3_K", "Q3_K", "Q3_K", "Q3_K"),
    ),
)
FIVEWAY_STRATEGIES = (
    Strategy(
        "rank_aware_iq1m",
        "Rank-aware IQ1_M",
        ("Q8_K", "Q6_K", "Q4_K", "Q3_K", "Q2_K", "Q2_K", "Q2_K", "IQ1_M", "IQ1_M", "IQ1_M"),
    ),
    Strategy(
        "rank_aware_qk654",
        "Rank-aware 654",
        ("Q6_K", "Q5_K", "Q4_K", "Q3_K", "Q2_K", "Q2_K", "Q2_K", "Q2_K", "Q2_K", "Q2_K"),
    ),
    Strategy(
        "rank_aware_qk55433",
        "Rank-aware 55433",
        ("Q5_K", "Q5_K", "Q4_K", "Q3_K", "Q3_K", "Q2_K", "Q2_K", "Q2_K", "Q2_K", "Q2_K"),
    ),
    Strategy(
        "q4_q2_split",
        "Q4/Q2 split",
        ("Q4_K", "Q4_K", "Q4_K", "Q4_K", "Q4_K", "Q2_K", "Q2_K", "Q2_K", "Q2_K", "Q2_K"),
    ),
    Strategy(
        "uniform_q3",
        "Uniform Q3",
        ("Q3_K", "Q3_K", "Q3_K", "Q3_K", "Q3_K", "Q3_K", "Q3_K", "Q3_K", "Q3_K", "Q3_K"),
    ),
)
ORACLE_DYNAMIC_CANDIDATES = (
    Strategy(
        "rank_aware_qk862",
        "Rank-aware 862",
        ("Q8_K", "Q6_K", "Q2_K", "Q2_K", "Q2_K", "Q2_K", "Q2_K", "Q2_K", "Q2_K", "Q2_K"),
    ),
    Strategy(
        "rank_aware_qk654",
        "Rank-aware 654",
        ("Q6_K", "Q5_K", "Q4_K", "Q3_K", "Q2_K", "Q2_K", "Q2_K", "Q2_K", "Q2_K", "Q2_K"),
    ),
    Strategy(
        "rank_aware_qk55433",
        "Rank-aware 55433",
        ("Q5_K", "Q5_K", "Q4_K", "Q3_K", "Q3_K", "Q2_K", "Q2_K", "Q2_K", "Q2_K", "Q2_K"),
    ),
    Strategy(
        "rank_aware_qk4443333",
        "Rank-aware 4443333",
        ("Q4_K", "Q4_K", "Q4_K", "Q3_K", "Q3_K", "Q3_K", "Q3_K", "Q2_K", "Q2_K", "Q2_K"),
    ),
    Strategy(
        "q4_q2_split",
        "Q4/Q2 split",
        ("Q4_K", "Q4_K", "Q4_K", "Q4_K", "Q4_K", "Q2_K", "Q2_K", "Q2_K", "Q2_K", "Q2_K"),
    ),
    Strategy(
        "uniform_q3",
        "Uniform Q3",
        ("Q3_K", "Q3_K", "Q3_K", "Q3_K", "Q3_K", "Q3_K", "Q3_K", "Q3_K", "Q3_K", "Q3_K"),
    ),
)
AVG3_EXTRA_CANDIDATES = (
    Strategy(
        "rank_aware_qk83333",
        "Avg3 83333",
        ("Q8_K", "Q3_K", "Q3_K", "Q3_K", "Q3_K", "Q2_K", "Q2_K", "Q2_K", "Q2_K", "Q2_K"),
    ),
    Strategy(
        "rank_aware_qk6444",
        "Avg3 6444",
        ("Q6_K", "Q4_K", "Q4_K", "Q4_K", "Q2_K", "Q2_K", "Q2_K", "Q2_K", "Q2_K", "Q2_K"),
    ),
    Strategy(
        "rank_aware_qk633333",
        "Avg3 633333",
        ("Q6_K", "Q3_K", "Q3_K", "Q3_K", "Q3_K", "Q3_K", "Q3_K", "Q2_K", "Q2_K", "Q2_K"),
    ),
    Strategy(
        "rank_aware_qk5553",
        "Avg3 5553",
        ("Q5_K", "Q5_K", "Q5_K", "Q3_K", "Q2_K", "Q2_K", "Q2_K", "Q2_K", "Q2_K", "Q2_K"),
    ),
    Strategy(
        "rank_aware_qk544333",
        "Avg3 544333",
        ("Q5_K", "Q4_K", "Q4_K", "Q3_K", "Q3_K", "Q3_K", "Q2_K", "Q2_K", "Q2_K", "Q2_K"),
    ),
    Strategy(
        "rank_aware_qk444433",
        "Avg3 444433",
        ("Q4_K", "Q4_K", "Q4_K", "Q4_K", "Q3_K", "Q3_K", "Q2_K", "Q2_K", "Q2_K", "Q2_K"),
    ),
    Strategy(
        "uniform_q3",
        "Uniform Q3",
        ("Q3_K", "Q3_K", "Q3_K", "Q3_K", "Q3_K", "Q3_K", "Q3_K", "Q3_K", "Q3_K", "Q3_K"),
    ),
)
AVG3_COMBINED_CANDIDATES = ORACLE_DYNAMIC_CANDIDATES[:-1] + AVG3_EXTRA_CANDIDATES
STATIC_EXPERT_Q4_Q2 = Strategy(
    "static_expert_q4_q2",
    "Static expert-level",
    static_formats_by_tier=("Q4_K", "Q2_K"),
)
ORACLE_DYNAMIC_Q4_CANDIDATES = (
    Strategy(
        "rank_aware_qk8665432222",
        "Rank-aware 8665432222",
        ("Q8_K", "Q6_K", "Q6_K", "Q5_K", "Q4_K", "Q3_K", "Q2_K", "Q2_K", "Q2_K", "Q2_K"),
    ),
    Strategy(
        "rank_aware_qk6655443322",
        "Rank-aware 6655443322",
        ("Q6_K", "Q6_K", "Q5_K", "Q5_K", "Q4_K", "Q4_K", "Q3_K", "Q3_K", "Q2_K", "Q2_K"),
    ),
    Strategy(
        "rank_aware_qk6555444322",
        "Rank-aware 6555444322",
        ("Q6_K", "Q5_K", "Q5_K", "Q5_K", "Q4_K", "Q4_K", "Q4_K", "Q3_K", "Q2_K", "Q2_K"),
    ),
    Strategy(
        "rank_aware_qk5555444422",
        "Rank-aware 5555444422",
        ("Q5_K", "Q5_K", "Q5_K", "Q5_K", "Q4_K", "Q4_K", "Q4_K", "Q4_K", "Q2_K", "Q2_K"),
    ),
    Strategy(
        "rank_aware_qk5554444333",
        "Rank-aware 5554444333",
        ("Q5_K", "Q5_K", "Q5_K", "Q4_K", "Q4_K", "Q4_K", "Q4_K", "Q3_K", "Q3_K", "Q3_K"),
    ),
    Strategy(
        "uniform_q4",
        "Uniform Q4",
        ("Q4_K", "Q4_K", "Q4_K", "Q4_K", "Q4_K", "Q4_K", "Q4_K", "Q4_K", "Q4_K", "Q4_K"),
    ),
)
STATIC_EXPERT_Q6_Q2 = Strategy(
    "static_expert_q6_q2",
    "Static expert-level",
    static_formats_by_tier=("Q6_K", "Q2_K"),
)
STRATEGY_SETS = {
    "avg3_combined": AVG3_COMBINED_CANDIDATES,
    "avg3_extra": AVG3_EXTRA_CANDIDATES,
    "fiveway": FIVEWAY_STRATEGIES,
    "iq1m": IQ1M_STRATEGIES,
    "oracle_dynamic": ORACLE_DYNAMIC_CANDIDATES + (STATIC_EXPERT_Q4_Q2,),
    "oracle_dynamic_q4": ORACLE_DYNAMIC_Q4_CANDIDATES + (STATIC_EXPERT_Q6_Q2,),
    "qk654": QK654_STRATEGIES,
}
DEFAULT_OUTPUT_STEMS = {
    "avg3_combined": "qwen35_rank_precision_avg3_combined_relative_error",
    "avg3_extra": "qwen35_rank_precision_avg3_extra_relative_error",
    "fiveway": "qwen35_rank_precision_fiveway_relative_error",
    "iq1m": "qwen35_rank_precision_relative_error",
    "oracle_dynamic": "qwen35_rank_precision_oracle_dynamic_relative_error",
    "oracle_dynamic_q4": "qwen35_rank_precision_oracle_dynamic_q4_relative_error",
    "qk654": "qwen35_rank_precision_qk654_relative_error",
}
ORACLE_STRATEGY_NAMES = {
    "avg3_combined": "oracle_avg3_combined",
    "avg3_extra": "oracle_avg3_extra",
    "oracle_dynamic": "oracle_dynamic",
    "oracle_dynamic_q4": "oracle_dynamic_q4",
}
ORACLE_BASELINE_STRATEGY_NAMES = {
    "avg3_combined": "uniform_q3",
    "avg3_extra": "uniform_q3",
    "oracle_dynamic": "uniform_q3",
    "oracle_dynamic_q4": "uniform_q4",
}
ORACLE_CANDIDATE_STRATEGIES = {
    "avg3_combined": AVG3_COMBINED_CANDIDATES,
    "avg3_extra": AVG3_EXTRA_CANDIDATES,
    "oracle_dynamic": ORACLE_DYNAMIC_CANDIDATES,
    "oracle_dynamic_q4": ORACLE_DYNAMIC_Q4_CANDIDATES,
}
STATIC_EXPERT_STRATEGY_NAMES = {
    "oracle_dynamic": STATIC_EXPERT_Q4_Q2.name,
    "oracle_dynamic_q4": STATIC_EXPERT_Q6_Q2.name,
}
STRATEGIES = IQ1M_STRATEGIES
IQ1_M = IQ_TYPES["IQ1_M"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--llama-cpp-lib", type=Path, default=DEFAULT_LLAMA_CPP_LIB)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--strategy-set",
        choices=sorted(STRATEGY_SETS),
        default="iq1m",
    )
    parser.add_argument(
        "--output-stem",
        help="output filename stem; defaults to a strategy-set-specific name",
    )
    parser.add_argument(
        "--iq-workers",
        type=int,
        default=1,
        help="number of matrix-row workers used for IQ1_M quantization",
    )
    parser.add_argument(
        "--iq-tasks-per-worker",
        type=int,
        default=4,
        help="split each IQ1_M matrix into approximately workers * this many tasks",
    )
    parser.add_argument(
        "--static-calibration-samples",
        type=int,
        default=100,
        help="number of synthetic hidden states used to rank experts for static allocation",
    )
    parser.add_argument(
        "--static-calibration-seed",
        type=int,
        help="base seed for static-allocation calibration; defaults to --seed + 1000000",
    )
    parser.add_argument(
        "--static-assignment-stem",
        default="qwen35_static_expert_precision_assignment",
        help="output filename stem for the static expert-level assignment",
    )
    return parser.parse_args()


def load_weight_map(model_root: Path) -> dict[str, str]:
    with (model_root / "model.safetensors.index.json").open("rb") as index_file:
        weight_map = json.load(index_file)["weight_map"]
    return {str(key): str(value) for key, value in weight_map.items()}


def load_router(
    model_root: Path, layer_index: int
) -> tuple[torch.Tensor, str, Path]:
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


def select_experts(
    router: torch.Tensor, hidden_state_bf16: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    router_logits = torch.mv(router.float(), hidden_state_bf16.float())
    all_routing_weights = F.softmax(router_logits, dim=0, dtype=torch.float32)
    routed_weights, expert_ids = torch.topk(all_routing_weights, TOP_K)
    routed_weights = routed_weights / routed_weights.sum()
    return router_logits, routed_weights, expert_ids


def nominal_bits(format_name: str) -> int:
    if format_name == IQ1_M.name:
        return IQ1_M.nominal_bits
    return QUANT_TYPES[format_name].nominal_bits


def bits_by_rank(strategy: Strategy) -> tuple[int, ...]:
    return tuple(nominal_bits(format_name) for format_name in strategy.formats_by_rank)


def precision_string(strategy: Strategy) -> str:
    if strategy.static_formats_by_tier is not None:
        bits = tuple(nominal_bits(format_name) for format_name in strategy.static_formats_by_tier)
        return f"expert-level fixed {bits[0]}/{bits[1]}"
    return "/".join(str(bit) for bit in bits_by_rank(strategy))


def format_string(strategy: Strategy) -> str:
    if strategy.static_formats_by_tier is not None:
        return f"expert-level fixed {'/'.join(strategy.static_formats_by_tier)}"
    return "/".join(strategy.formats_by_rank)


def configured_formats(strategy: Strategy) -> tuple[str, ...]:
    if strategy.static_formats_by_tier is not None:
        return strategy.static_formats_by_tier
    return strategy.formats_by_rank


def nominal_average_bits(strategy: Strategy) -> float:
    if strategy.static_formats_by_tier is not None:
        return statistics.mean(
            nominal_bits(format_name) for format_name in strategy.static_formats_by_tier
        )
    return sum(bits_by_rank(strategy)) / TOP_K


def static_storage_budget(strategy: Strategy) -> dict[str, object]:
    if strategy.static_formats_by_tier is None:
        raise ValueError(f"{strategy.name} is not a static expert-level strategy")
    return {
        "formats_by_tier": strategy.static_formats_by_tier,
        "nominal_average_bits_per_weight": nominal_average_bits(strategy),
        "effective_average_bits_per_weight": statistics.mean(
            QUANT_TYPES[format_name].effective_bpw
            for format_name in strategy.static_formats_by_tier
        ),
    }


def format_for_expert(
    strategy: Strategy,
    rank: int,
    expert_id: int,
    high_precision_expert_ids: frozenset[int],
) -> str:
    if strategy.static_formats_by_tier is not None:
        tier_index = 0 if expert_id in high_precision_expert_ids else 1
        return strategy.static_formats_by_tier[tier_index]
    return strategy.formats_by_rank[rank - 1]


def calibrate_static_expert_assignment(
    *,
    router: torch.Tensor,
    base_seed: int,
    num_samples: int,
) -> tuple[frozenset[int], list[dict[str, object]]]:
    selected_counts = [0] * NUM_EXPERTS
    routed_weight_sums = [0.0] * NUM_EXPERTS
    for offset in range(num_samples):
        generator = torch.Generator(device="cpu").manual_seed(base_seed + offset)
        hidden_state_bf16 = torch.randn(
            HIDDEN_SIZE, generator=generator, dtype=torch.float32
        ).to(torch.bfloat16)
        _, routed_weights, expert_ids = select_experts(router, hidden_state_bf16)
        for expert_id_tensor, routed_weight_tensor in zip(expert_ids, routed_weights):
            expert_id = int(expert_id_tensor.item())
            selected_counts[expert_id] += 1
            routed_weight_sums[expert_id] += float(routed_weight_tensor.item())
        del hidden_state_bf16, routed_weights, expert_ids

    ranked_expert_ids = sorted(
        range(NUM_EXPERTS),
        key=lambda expert_id: (-selected_counts[expert_id], expert_id),
    )
    high_precision_expert_ids = frozenset(ranked_expert_ids[: NUM_EXPERTS // 2])
    assignment_rows: list[dict[str, object]] = []
    for popularity_rank, expert_id in enumerate(ranked_expert_ids, start=1):
        selected_count = selected_counts[expert_id]
        is_high_precision = expert_id in high_precision_expert_ids
        assignment_rows.append(
            {
                "popularity_rank": popularity_rank,
                "expert_id": expert_id,
                "selected_count": selected_count,
                "selection_probability": selected_count / num_samples,
                "routed_weight_sum": routed_weight_sums[expert_id],
                "mean_selected_routed_weight": (
                    routed_weight_sums[expert_id] / selected_count
                    if selected_count
                    else 0.0
                ),
                "tier": "high" if is_high_precision else "low",
                "q3_format": (
                    STATIC_EXPERT_Q4_Q2.static_formats_by_tier[0]
                    if is_high_precision
                    else STATIC_EXPERT_Q4_Q2.static_formats_by_tier[1]
                ),
                "q4_format": (
                    STATIC_EXPERT_Q6_Q2.static_formats_by_tier[0]
                    if is_high_precision
                    else STATIC_EXPERT_Q6_Q2.static_formats_by_tier[1]
                ),
            }
        )
    return high_precision_expert_ids, assignment_rows


def write_static_assignment_csv(
    path: Path, rows: list[dict[str, object]]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_sample_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "sample_index",
        "seed",
        "strategy",
        "label",
        "allocation_level",
        "precision_by_rank",
        "format_by_rank",
        "selected_strategy",
        "nominal_average_bits",
        "effective_average_bits_per_weight",
        "quantized_top10_bytes",
        "relative_l2_error",
        "l2_error",
        "max_absolute_error",
        "mean_absolute_error",
        "cosine_similarity",
        "reference_l2_norm",
        "output_l2_norm",
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "strategy",
        "label",
        "allocation_level",
        "precision_by_rank",
        "format_by_rank",
        "nominal_average_bits",
        "effective_average_bits_per_weight",
        "quantized_top10_bytes",
        "sample_count",
        "win_count",
        "mean_relative_l2_error",
        "std_relative_l2_error",
        "median_relative_l2_error",
        "min_relative_l2_error",
        "max_relative_l2_error",
        "mean_cosine_similarity",
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_sample(
    *,
    sample_index: int,
    seed: int,
    args: argparse.Namespace,
    router: torch.Tensor,
    qk: LlamaCppQK,
    iq: LlamaCppIQ | None,
    high_precision_expert_ids: frozenset[int],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    sample_started = time.time()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    hidden_state_bf16 = torch.randn(
        HIDDEN_SIZE, generator=generator, dtype=torch.float32
    ).to(torch.bfloat16)
    router_logits, routed_weights, expert_ids = select_experts(
        router, hidden_state_bf16
    )
    reference = torch.zeros(HIDDEN_SIZE, dtype=torch.float32)
    strategy_outputs = {
        strategy.name: torch.zeros(HIDDEN_SIZE, dtype=torch.float32)
        for strategy in STRATEGIES
    }
    strategy_bytes = {strategy.name: 0 for strategy in STRATEGIES}
    routed_experts: list[dict[str, object]] = []

    print(
        f"sample={sample_index} seed={seed} "
        f"top10_routed_weight_sum={routed_weights.sum().item():.8f}",
        flush=True,
    )
    for rank, (expert_id_tensor, routed_weight_tensor) in enumerate(
        zip(expert_ids, routed_weights), start=1
    ):
        expert_started = time.time()
        expert_id = int(expert_id_tensor.item())
        routed_weight = float(routed_weight_tensor.item())
        required_formats = sorted(
            {
                format_for_expert(
                    strategy, rank, expert_id, high_precision_expert_ids
                )
                for strategy in STRATEGIES
            }
        )

        gate, up, down, _ = load_one_expert(
            args.model_root, args.layer_index, expert_id
        )
        reference_expert = unquantized_expert(
            gate, up, down, hidden_state_bf16
        )
        reference.add_(reference_expert, alpha=routed_weight)

        outputs_by_format: dict[str, torch.Tensor] = {}
        bytes_by_format: dict[str, int] = {}
        for format_name in required_formats:
            if format_name == IQ1_M.name:
                if iq is None:
                    raise RuntimeError("IQ1_M requested without an initialized IQ binding")
                try:
                    output, quantized_bytes = quantized_iq_expert(
                        iq,
                        IQ1_M,
                        gate,
                        up,
                        down,
                        hidden_state_bf16,
                        workers=args.iq_workers,
                        tasks_per_worker=args.iq_tasks_per_worker,
                    )
                finally:
                    iq.free_tables()
            else:
                output, quantized_bytes = quantized_qk_expert(
                    qk,
                    QUANT_TYPES[format_name],
                    gate,
                    up,
                    down,
                    hidden_state_bf16,
                )
            outputs_by_format[format_name] = output
            bytes_by_format[format_name] = quantized_bytes

        for strategy in STRATEGIES:
            format_name = format_for_expert(
                strategy, rank, expert_id, high_precision_expert_ids
            )
            strategy_outputs[strategy.name].add_(
                outputs_by_format[format_name], alpha=routed_weight
            )
            strategy_bytes[strategy.name] += bytes_by_format[format_name]

        routed_experts.append(
            {
                "rank": rank,
                "expert_id": expert_id,
                "routed_weight": routed_weight,
                "required_formats": required_formats,
                "elapsed_seconds": time.time() - expert_started,
            }
        )
        del gate, up, down, reference_expert, outputs_by_format, bytes_by_format
        gc.collect()

    rows: list[dict[str, object]] = []
    for strategy in STRATEGIES:
        output = strategy_outputs[strategy.name]
        row: dict[str, object] = {
            "sample_index": sample_index,
            "seed": seed,
            "strategy": strategy.name,
            "label": strategy.label,
            "allocation_level": (
                "expert_id"
                if strategy.static_formats_by_tier is not None
                else "routed_rank"
            ),
            "precision_by_rank": precision_string(strategy),
            "format_by_rank": format_string(strategy),
            "selected_strategy": "",
            "nominal_average_bits": nominal_average_bits(strategy),
            "effective_average_bits_per_weight": (
                strategy_bytes[strategy.name] * 8
                / (TOP_K * ORIGINAL_EXPERT_BYTES / torch.bfloat16.itemsize)
            ),
            "quantized_top10_bytes": strategy_bytes[strategy.name],
        }
        row.update(metrics(reference, output))
        rows.append(row)
    oracle_selection: dict[str, object] | None = None
    oracle_strategy_name = ORACLE_STRATEGY_NAMES.get(args.strategy_set)
    if oracle_strategy_name is not None:
        candidate_names = {
            strategy.name for strategy in ORACLE_CANDIDATE_STRATEGIES[args.strategy_set]
        }
        winner = min(
            (row for row in rows if row["strategy"] in candidate_names),
            key=lambda row: float(row["relative_l2_error"]),
        )
        oracle_row = dict(winner)
        oracle_row.update(
            {
                "strategy": oracle_strategy_name,
                "label": "Oracle dynamic",
                "allocation_level": "per_sample_oracle",
                "precision_by_rank": "per-sample best",
                "format_by_rank": "per-sample best",
                "selected_strategy": winner["strategy"],
            }
        )
        rows.append(oracle_row)
        oracle_selection = {
            "strategy": winner["strategy"],
            "label": winner["label"],
            "precision_by_rank": winner["precision_by_rank"],
            "format_by_rank": winner["format_by_rank"],
            "relative_l2_error": winner["relative_l2_error"],
            "effective_average_bits_per_weight": winner[
                "effective_average_bits_per_weight"
            ],
        }
    print(
        "  "
        + "  ".join(
            f"{row['strategy']}={float(row['relative_l2_error']) * 100:.4f}%"
            for row in rows
        ),
        flush=True,
    )
    sample_metadata = {
        "sample_index": sample_index,
        "seed": seed,
        "router": {
            "top10_routed_weight_sum": routed_weights.sum().item(),
            "router_logits_l2_norm": torch.linalg.vector_norm(router_logits).item(),
        },
        "routed_experts": routed_experts,
        "reference_output_l2_norm": torch.linalg.vector_norm(reference).item(),
        "oracle_selection": oracle_selection,
        "elapsed_seconds": time.time() - sample_started,
    }
    del hidden_state_bf16, router_logits, routed_weights, expert_ids
    del reference, strategy_outputs
    gc.collect()
    return rows, sample_metadata


def summarize_rows(
    sample_rows: list[dict[str, object]],
    strategy_names: list[str] | None = None,
) -> list[dict[str, object]]:
    if strategy_names is None:
        strategy_names = [strategy.name for strategy in STRATEGIES]
    rows_by_strategy = {
        strategy_name: [
            row for row in sample_rows if row["strategy"] == strategy_name
        ]
        for strategy_name in strategy_names
    }
    wins = {strategy_name: 0 for strategy_name in strategy_names}
    for sample_index in sorted({int(row["sample_index"]) for row in sample_rows}):
        rows = [
            row
            for row in sample_rows
            if int(row["sample_index"]) == sample_index
            and str(row["strategy"]) in strategy_names
        ]
        winner = min(rows, key=lambda row: float(row["relative_l2_error"]))
        wins[str(winner["strategy"])] += 1

    summary_rows: list[dict[str, object]] = []
    for strategy_name in strategy_names:
        rows = rows_by_strategy[strategy_name]
        relative_errors = [float(row["relative_l2_error"]) for row in rows]
        cosine_similarities = [float(row["cosine_similarity"]) for row in rows]
        summary_rows.append(
            {
                "strategy": strategy_name,
                "label": rows[0]["label"],
                "allocation_level": rows[0]["allocation_level"],
                "precision_by_rank": rows[0]["precision_by_rank"],
                "format_by_rank": rows[0]["format_by_rank"],
                "nominal_average_bits": statistics.mean(
                    float(row["nominal_average_bits"]) for row in rows
                ),
                "effective_average_bits_per_weight": statistics.mean(
                    float(row["effective_average_bits_per_weight"]) for row in rows
                ),
                "quantized_top10_bytes": statistics.mean(
                    int(row["quantized_top10_bytes"]) for row in rows
                ),
                "sample_count": len(rows),
                "win_count": wins[strategy_name],
                "mean_relative_l2_error": statistics.mean(relative_errors),
                "std_relative_l2_error": (
                    statistics.stdev(relative_errors) if len(rows) > 1 else 0.0
                ),
                "median_relative_l2_error": statistics.median(relative_errors),
                "min_relative_l2_error": min(relative_errors),
                "max_relative_l2_error": max(relative_errors),
                "mean_cosine_similarity": statistics.mean(cosine_similarities),
            }
        )
    return summary_rows


def main() -> int:
    global STRATEGIES
    args = parse_args()
    if args.num_samples < 1:
        raise ValueError(f"--num-samples must be positive, got {args.num_samples}")
    if args.iq_workers < 1:
        raise ValueError(f"--iq-workers must be positive, got {args.iq_workers}")
    if args.iq_tasks_per_worker < 1:
        raise ValueError(
            f"--iq-tasks-per-worker must be positive, got {args.iq_tasks_per_worker}"
        )
    if args.static_calibration_samples < 1:
        raise ValueError(
            "--static-calibration-samples must be positive, got "
            f"{args.static_calibration_samples}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, args.torch_threads))
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)

    started = time.time()
    STRATEGIES = STRATEGY_SETS[args.strategy_set]
    output_stem = args.output_stem or DEFAULT_OUTPUT_STEMS[args.strategy_set]
    uses_iq1_m = any(
        IQ1_M.name in configured_formats(strategy) for strategy in STRATEGIES
    )
    router, router_key, router_path = load_router(args.model_root, args.layer_index)
    qk = LlamaCppQK(args.llama_cpp_lib)
    iq = LlamaCppIQ(args.llama_cpp_lib) if uses_iq1_m else None
    seeds = [args.seed + offset for offset in range(args.num_samples)]
    calibration_seed = (
        args.static_calibration_seed
        if args.static_calibration_seed is not None
        else args.seed + 1_000_000
    )
    uses_static_expert_assignment = any(
        strategy.static_formats_by_tier is not None for strategy in STRATEGIES
    )
    high_precision_expert_ids: frozenset[int] = frozenset()
    static_assignment_rows: list[dict[str, object]] = []
    static_assignment_csv_path: Path | None = None
    static_assignment_json_path: Path | None = None
    if uses_static_expert_assignment:
        print(
            "calibrating_static_expert_assignment "
            f"samples={args.static_calibration_samples} "
            f"seeds={calibration_seed}.."
            f"{calibration_seed + args.static_calibration_samples - 1}",
            flush=True,
        )
        high_precision_expert_ids, static_assignment_rows = (
            calibrate_static_expert_assignment(
                router=router,
                base_seed=calibration_seed,
                num_samples=args.static_calibration_samples,
            )
        )
        static_assignment_csv_path = (
            args.output_dir / f"{args.static_assignment_stem}.csv"
        )
        static_assignment_json_path = (
            args.output_dir / f"{args.static_assignment_stem}.json"
        )
        write_static_assignment_csv(static_assignment_csv_path, static_assignment_rows)
        static_assignment_metadata = {
            "description": (
                "Static expert-level mixed-precision assignment. Experts are ranked "
                "by empirical top-10 selection probability over calibration hidden "
                "states; ties are resolved by ascending expert ID. The first half "
                "is permanently assigned to the high-precision tier and the second "
                "half to the low-precision tier."
            ),
            "layer_index": args.layer_index,
            "calibration_hidden_states": {
                "description": (
                    "deterministic synthetic BF16 hidden states sampled from N(0, 1)"
                ),
                "base_seed": calibration_seed,
                "seeds": [
                    calibration_seed + offset
                    for offset in range(args.static_calibration_samples)
                ],
                "sample_count": args.static_calibration_samples,
                "size": HIDDEN_SIZE,
            },
            "router_top_k": TOP_K,
            "expert_count": NUM_EXPERTS,
            "high_precision_expert_count": len(high_precision_expert_ids),
            "low_precision_expert_count": NUM_EXPERTS - len(high_precision_expert_ids),
            "q3_tiers": {
                "high": STATIC_EXPERT_Q4_Q2.static_formats_by_tier[0],
                "low": STATIC_EXPERT_Q4_Q2.static_formats_by_tier[1],
            },
            "q3_storage_budget": static_storage_budget(STATIC_EXPERT_Q4_Q2),
            "q4_tiers": {
                "high": STATIC_EXPERT_Q6_Q2.static_formats_by_tier[0],
                "low": STATIC_EXPERT_Q6_Q2.static_formats_by_tier[1],
            },
            "q4_storage_budget": static_storage_budget(STATIC_EXPERT_Q6_Q2),
            "ranking": static_assignment_rows,
            "csv": str(static_assignment_csv_path.resolve()),
        }
        static_assignment_json_path.write_text(
            json.dumps(static_assignment_metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"static_assignment_csv={static_assignment_csv_path}", flush=True)
        print(f"static_assignment_json={static_assignment_json_path}", flush=True)
    sample_rows: list[dict[str, object]] = []
    samples: list[dict[str, object]] = []
    print(
        f"model={args.model_root} layer={args.layer_index} top_k={TOP_K} "
        f"samples={args.num_samples} seeds={seeds[0]}..{seeds[-1]}",
        flush=True,
    )
    try:
        for sample_index, seed in enumerate(seeds):
            rows, sample_metadata = run_sample(
                sample_index=sample_index,
                seed=seed,
                args=args,
                router=router,
                qk=qk,
                iq=iq,
                high_precision_expert_ids=high_precision_expert_ids,
            )
            sample_rows.extend(rows)
            samples.append(sample_metadata)
    finally:
        if iq is not None:
            iq.free_tables()
    oracle_strategy_name = ORACLE_STRATEGY_NAMES.get(args.strategy_set)
    if oracle_strategy_name is None:
        reporting_strategy_names = None
    elif args.strategy_set in STATIC_EXPERT_STRATEGY_NAMES:
        reporting_strategy_names = [
            oracle_strategy_name,
            STATIC_EXPERT_STRATEGY_NAMES[args.strategy_set],
            ORACLE_BASELINE_STRATEGY_NAMES[args.strategy_set],
        ]
    else:
        reporting_strategy_names = [
            oracle_strategy_name,
            *[
                strategy.name
                for strategy in ORACLE_CANDIDATE_STRATEGIES[args.strategy_set]
            ],
        ]
    summary_rows = summarize_rows(sample_rows, reporting_strategy_names)
    oracle_selection_counts = {
        strategy.name: sum(
            sample["oracle_selection"] is not None
            and sample["oracle_selection"]["strategy"] == strategy.name
            for sample in samples
        )
        for strategy in ORACLE_CANDIDATE_STRATEGIES.get(args.strategy_set, STRATEGIES)
    }

    csv_path = args.output_dir / f"{output_stem}.csv"
    samples_csv_path = args.output_dir / f"{output_stem}_samples.csv"
    metadata_path = args.output_dir / f"{output_stem}.json"
    write_summary_csv(csv_path, summary_rows)
    write_sample_csv(samples_csv_path, sample_rows)
    metadata = {
        "model_root": str(args.model_root.resolve()),
        "llama_cpp_library": str(qk.library_path),
        "layer_index": args.layer_index,
        "strategy_set": args.strategy_set,
        "output_stem": output_stem,
        "hidden_states": {
            "description": "deterministic synthetic BF16 hidden states sampled from N(0, 1)",
            "base_seed": args.seed,
            "seeds": seeds,
            "sample_count": args.num_samples,
            "size": HIDDEN_SIZE,
        },
        "router": {
            "tensor": router_key,
            "shard": str(router_path),
            "shape": [NUM_EXPERTS, HIDDEN_SIZE],
            "routing": (
                "FP32 softmax over all router logits, followed by top-10 selection; "
                "selected routed weights are renormalized to sum to one"
            ),
        },
        "reference": {
            "description": (
                "Routed-branch-only FP32 weighted sum of top-10 outputs using the "
                "original stored BF16 expert weights; shared_expert and "
                "shared_expert_gate are intentionally excluded. Matrix products use "
                "FP32 accumulation and each expert rounds the gated intermediate "
                "activation to BF16."
            ),
        },
        "strategies": [
            {
                **asdict(strategy),
                "bits_by_rank": bits_by_rank(strategy),
                "precision_by_rank": precision_string(strategy),
                "format_by_rank": format_string(strategy),
                "nominal_average_bits": nominal_average_bits(strategy),
            }
            for strategy in STRATEGIES
        ],
        "reporting_strategies": [
            str(row["strategy"]) for row in summary_rows
        ],
        "oracle_dynamic": (
            {
                "description": (
                    "retrospectively select the candidate with the lowest relative "
                    "L2 error against the BF16 reference for each hidden state"
                ),
                "selection_counts": oracle_selection_counts,
            }
            if oracle_strategy_name is not None
            else None
        ),
        "static_expert_assignment": (
            {
                "description": (
                    "fixed expert-level tiers calibrated from empirical router "
                    "top-10 selection probability; see the separate assignment file"
                ),
                "calibration_base_seed": calibration_seed,
                "calibration_sample_count": args.static_calibration_samples,
                "high_precision_expert_count": len(high_precision_expert_ids),
                "low_precision_expert_count": (
                    NUM_EXPERTS - len(high_precision_expert_ids)
                ),
                "storage_budget": static_storage_budget(
                    next(
                        strategy
                        for strategy in STRATEGIES
                        if strategy.static_formats_by_tier is not None
                    )
                ),
                "csv": str(static_assignment_csv_path.resolve()),
                "json": str(static_assignment_json_path.resolve()),
            }
            if uses_static_expert_assignment
            else None
        ),
        "quantization": {
            "description": (
                "Q2_K through Q8_K slots use llama.cpp quantize_row_qx_K_ref "
                "followed by dequantize_row_qx_K. If present, one-bit slots use "
                "llama.cpp IQ1_M via ggml_quantize_chunk followed by "
                "dequantize_row_iq1_m. Dequantized matrices are multiplied in FP32 "
                "to measure weight-representation error rather than end-to-end "
                "llama.cpp kernel error. One selected expert and one matrix are "
                "processed at a time."
            ),
            "uses_iq1_m": uses_iq1_m,
            "iq1_m_importance": (
                "per-input weights derived from the measured activation: hidden_state^2 "
                "for gate/up and quantized-path intermediate^2 for down"
            ),
            "iq_workers": args.iq_workers,
            "iq_tasks_per_worker": args.iq_tasks_per_worker,
            "q8_note": QUANT_TYPES["Q8_K"].note,
        },
        "samples": samples,
        "summary": summary_rows,
        "elapsed_seconds": time.time() - started,
        "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "csv": str(csv_path.resolve()),
        "samples_csv": str(samples_csv_path.resolve()),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"csv={csv_path}")
    print(f"samples_csv={samples_csv_path}")
    print(f"metadata={metadata_path}")
    print(f"max_rss_mib={metadata['max_rss_mib']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
