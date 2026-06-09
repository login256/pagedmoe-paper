#!/usr/bin/env python3
"""Plot per-hidden-state errors for oracle-dynamic precision candidates."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = SCRIPT_DIR / "qwen35_rank_precision_oracle_dynamic_relative_error_samples.csv"
DEFAULT_OUTPUT_STEM = SCRIPT_DIR / "qwen35_rank_precision_oracle_dynamic_trace"
CANDIDATE_SETS = {
    "avg3_combined": (
        ("rank_aware_qk862", "8/6/2/2/2/2/2/2/2/2"),
        ("rank_aware_qk654", "6/5/4/3/2/2/2/2/2/2"),
        ("rank_aware_qk55433", "5/5/4/3/3/2/2/2/2/2"),
        ("rank_aware_qk4443333", "4/4/4/3/3/3/3/2/2/2"),
        ("q4_q2_split", "4/4/4/4/4/2/2/2/2/2"),
        ("rank_aware_qk83333", "8/3/3/3/3/2/2/2/2/2"),
        ("rank_aware_qk6444", "6/4/4/4/2/2/2/2/2/2"),
        ("rank_aware_qk633333", "6/3/3/3/3/3/3/2/2/2"),
        ("rank_aware_qk5553", "5/5/5/3/2/2/2/2/2/2"),
        ("rank_aware_qk544333", "5/4/4/3/3/3/2/2/2/2"),
        ("rank_aware_qk444433", "4/4/4/4/3/3/2/2/2/2"),
        ("uniform_q3", "Uniform Q3"),
    ),
    "avg3_extra": (
        ("rank_aware_qk83333", "8/3/3/3/3/2/2/2/2/2"),
        ("rank_aware_qk6444", "6/4/4/4/2/2/2/2/2/2"),
        ("rank_aware_qk633333", "6/3/3/3/3/3/3/2/2/2"),
        ("rank_aware_qk5553", "5/5/5/3/2/2/2/2/2/2"),
        ("rank_aware_qk544333", "5/4/4/3/3/3/2/2/2/2"),
        ("rank_aware_qk444433", "4/4/4/4/3/3/2/2/2/2"),
        ("uniform_q3", "Uniform Q3"),
    ),
    "q3": (
        ("rank_aware_qk862", "8/6/2/2/2/2/2/2/2/2"),
        ("rank_aware_qk654", "6/5/4/3/2/2/2/2/2/2"),
        ("rank_aware_qk55433", "5/5/4/3/3/2/2/2/2/2"),
        ("rank_aware_qk4443333", "4/4/4/3/3/3/3/2/2/2"),
        ("q4_q2_split", "4/4/4/4/4/2/2/2/2/2"),
        ("uniform_q3", "Uniform Q3"),
    ),
    "q4": (
        ("rank_aware_qk8665432222", "8/6/6/5/4/3/2/2/2/2"),
        ("rank_aware_qk6655443322", "6/6/5/5/4/4/3/3/2/2"),
        ("rank_aware_qk6555444322", "6/5/5/5/4/4/4/3/2/2"),
        ("rank_aware_qk5555444422", "5/5/5/5/4/4/4/4/2/2"),
        ("rank_aware_qk5554444333", "5/5/5/4/4/4/4/3/3/3"),
        ("uniform_q4", "Uniform Q4"),
    ),
}
ORACLE_STRATEGY_NAMES = {
    "avg3_combined": "oracle_avg3_combined",
    "avg3_extra": "oracle_avg3_extra",
    "q3": "oracle_dynamic",
    "q4": "oracle_dynamic_q4",
}
STATIC_COMPARISONS = {
    "avg3_combined": None,
    "avg3_extra": None,
    "q3": ("static_expert_q4_q2", "Static expert-level Q4/Q2"),
    "q4": ("static_expert_q6_q2", "Static expert-level Q6/Q2"),
}
COLORS = (
    "#0072B2",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#9467BD",
    "#7F7F7F",
    "#E69F00",
    "#332288",
    "#88CCEE",
    "#44AA99",
    "#117733",
    "#AA4499",
)
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "*", "+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--candidate-set", choices=sorted(CANDIDATE_SETS), default="q3")
    return parser.parse_args()


def load_rows(
    path: Path,
    candidates: tuple[tuple[str, str], ...],
    static_comparison: tuple[str, str] | None,
    oracle_strategy_name: str,
) -> tuple[list[int], dict[str, list[float]]]:
    by_sample: dict[int, dict[str, float]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            by_sample[int(row["sample_index"])][row["strategy"]] = (
                float(row["relative_l2_error"]) * 100
            )

    sample_indices = sorted(by_sample)
    if sample_indices != list(range(len(sample_indices))):
        raise ValueError(f"expected contiguous sample indices in {path}")

    candidate_names = [name for name, _ in candidates]
    expected = set(candidate_names + [oracle_strategy_name])
    if static_comparison is not None:
        expected.add(static_comparison[0])
    series: dict[str, list[float]] = defaultdict(list)
    for sample_index in sample_indices:
        sample = by_sample[sample_index]
        if set(sample) != expected:
            raise ValueError(
                f"sample {sample_index} has strategies {sorted(sample)}, "
                f"expected {sorted(expected)}"
            )
        candidate_min = min(sample[name] for name in candidate_names)
        if abs(sample[oracle_strategy_name] - candidate_min) > 1e-10:
            raise ValueError(f"sample {sample_index} oracle value is not the candidate minimum")
        for strategy in expected:
            series[strategy].append(sample[strategy])
    return sample_indices, series


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
            "legend.fontsize": 9.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot(
    output_stem: Path,
    sample_indices: list[int],
    series: dict[str, list[float]],
    candidates: tuple[tuple[str, str], ...],
    static_comparison: tuple[str, str] | None,
    oracle_strategy_name: str,
) -> None:
    configure_matplotlib()
    x_values = [sample_index + 1 for sample_index in sample_indices]
    fig, ax = plt.subplots(figsize=(6.8, 3.0))

    for (strategy, label), color, marker in zip(candidates, COLORS, MARKERS):
        ax.plot(
            x_values,
            series[strategy],
            label=label,
            color=color,
            linewidth=0.85,
            marker=marker,
            markersize=2.5,
            markeredgewidth=0,
            alpha=0.72,
            zorder=2,
        )
    if static_comparison is not None:
        ax.plot(
            x_values,
            series[static_comparison[0]],
            label=static_comparison[1],
            color="#333333",
            linewidth=1.2,
            linestyle=(0, (3, 2)),
            marker="x",
            markersize=3.2,
            markeredgewidth=0.7,
            zorder=3,
        )
    ax.plot(
        x_values,
        series[oracle_strategy_name],
        label="Oracle dynamic: per-sample best",
        color="#D55E00",
        linewidth=2.0,
        marker="o",
        markersize=3.4,
        markeredgecolor="white",
        markeredgewidth=0.35,
        zorder=4,
    )

    ax.set_xlim(0.5, len(x_values) + 0.5)
    ax.set_ylim(bottom=0)
    ax.set_xticks(x_values)
    ax.set_ylabel("Relative output error (%)")
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
        bbox_to_anchor=(0.5, 1.32),
        ncol=4,
        frameon=False,
        fontsize=9.5,
        handlelength=2.1,
        columnspacing=1.0,
        handletextpad=0.4,
    )

    fig.tight_layout(pad=0.45)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    candidates = CANDIDATE_SETS[args.candidate_set]
    static_comparison = STATIC_COMPARISONS[args.candidate_set]
    oracle_strategy_name = ORACLE_STRATEGY_NAMES[args.candidate_set]
    sample_indices, series = load_rows(
        args.csv, candidates, static_comparison, oracle_strategy_name
    )
    plot(
        args.output_stem,
        sample_indices,
        series,
        candidates,
        static_comparison,
        oracle_strategy_name,
    )
    print(f"samples={len(sample_indices)}")
    print(f"pdf={args.output_stem.with_suffix('.pdf')}")
    print(f"png={args.output_stem.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
