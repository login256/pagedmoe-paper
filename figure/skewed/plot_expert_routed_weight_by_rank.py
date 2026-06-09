#!/usr/bin/env python3
"""Plot routed-weight distributions for experts ranked by importance."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_TRACE = (
    REPO_ROOT
    / "src/target/bench/precision_trace_c4l10_20260518_122725"
    / "precision_trace.log"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR
EXPECTED_RANK_COUNT = 10
TRACE_PATTERN = re.compile(
    r"\[pagedmoe-precision-trace\].*?"
    r"\bexperts=\[(?P<experts>[^\]]*)\].*?"
    r"\bweights=\[(?P<weights>[^\]]*)\]"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot expert routed-weight distributions by rank."
    )
    parser.add_argument(
        "--trace",
        type=Path,
        default=DEFAULT_TRACE,
        help=f"precision trace to parse (default: {DEFAULT_TRACE})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"directory for generated files (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def parse_trace(trace_path: Path) -> list[list[float]]:
    """Return routed weights grouped by one-indexed expert rank."""
    routed_weights_by_rank: list[list[float]] = [
        [] for _ in range(EXPECTED_RANK_COUNT)
    ]
    record_count = 0

    with trace_path.open(encoding="utf-8", errors="replace") as trace_file:
        for line_number, line in enumerate(trace_file, start=1):
            match = TRACE_PATTERN.search(line)
            if match is None:
                continue

            experts = [
                int(value.strip())
                for value in match.group("experts").split(",")
                if value.strip()
            ]
            weights = [
                float(value.strip())
                for value in match.group("weights").split(",")
                if value.strip()
            ]
            if len(experts) != EXPECTED_RANK_COUNT:
                raise ValueError(
                    f"line {line_number} has {len(experts)} experts; "
                    f"expected {EXPECTED_RANK_COUNT}"
                )
            if len(weights) != len(experts):
                raise ValueError(
                    f"line {line_number} has {len(weights)} weights for "
                    f"{len(experts)} experts"
                )
            if any(weight < 0.0 or weight > 1.0 for weight in weights):
                raise ValueError(f"line {line_number} has a weight outside [0, 1]")
            if any(left < right for left, right in zip(weights, weights[1:])):
                raise ValueError(
                    f"line {line_number} weights are not sorted in descending order"
                )

            for rank, weight in enumerate(weights):
                routed_weights_by_rank[rank].append(weight)
            record_count += 1

    if record_count == 0:
        raise ValueError(f"no precision-trace records found in {trace_path}")

    return routed_weights_by_rank


def write_statistics_csv(
    output_path: Path, routed_weights_by_rank: list[list[float]]
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file, lineterminator="\n")
        writer.writerow(
            [
                "rank",
                "sample_count",
                "min",
                "p25",
                "median",
                "p75",
                "max",
                "mean",
            ]
        )
        for rank, weights in enumerate(routed_weights_by_rank, start=1):
            minimum, p25, median, p75, maximum = np.quantile(
                weights, [0.0, 0.25, 0.5, 0.75, 1.0]
            )
            writer.writerow(
                [
                    rank,
                    len(weights),
                    f"{minimum:.8f}",
                    f"{p25:.8f}",
                    f"{median:.8f}",
                    f"{p75:.8f}",
                    f"{maximum:.8f}",
                    f"{np.mean(weights):.8f}",
                ]
            )


def plot_routed_weights(
    output_dir: Path,
    routed_weights_by_rank: list[list[float]],
    log_scale: bool = False,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Linux Libertine O", "Libertinus Serif", "STIXGeneral"],
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(3.35, 2.25))
    positions = list(range(1, EXPECTED_RANK_COUNT + 1))
    violin_fill = "#9ECAE1"
    violin_edge = "#08306B"
    box_edge = "#08306B"
    median_color = "#D55E00"
    mean_color = "#E69F00"

    violins = ax.violinplot(
        routed_weights_by_rank,
        positions=positions,
        widths=0.74,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body in violins["bodies"]:
        body.set_facecolor(violin_fill)
        body.set_edgecolor(violin_edge)
        body.set_alpha(0.92)
        body.set_linewidth(0.8)

    boxplot = ax.boxplot(
        routed_weights_by_rank,
        positions=positions,
        widths=0.15,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
        medianprops={"color": median_color, "linewidth": 1.2},
        boxprops={"facecolor": "white", "edgecolor": box_edge, "linewidth": 0.75},
        whiskerprops={"color": box_edge, "linewidth": 0.75},
        capprops={"visible": False},
    )
    for box in boxplot["boxes"]:
        box.set_alpha(0.94)

    means = [float(np.mean(weights)) for weights in routed_weights_by_rank]
    ax.plot(
        positions,
        means,
        "D",
        color=mean_color,
        markersize=2.4,
        markeredgewidth=0,
        zorder=4,
    )

    ax.set_xlim(0.45, EXPECTED_RANK_COUNT + 0.55)
    ax.set_xticks(positions)
    if log_scale:
        ax.set_yscale("log")
        ax.set_ylim(0.002, 1.0)
        ax.yaxis.set_major_locator(
            FixedLocator([0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
        )
        ax.yaxis.set_major_formatter(
            FuncFormatter(
                lambda value, _: (
                    f"{value:.1%}" if value < 0.01 else f"{value:.0%}"
                )
            )
        )
    else:
        ax.set_ylim(0.0, 1.0)
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle=(0, (2, 2)), color="#B8C2CC", linewidth=0.5, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Expert rank")
    ax.set_ylabel("Routed weight")
    legend_handles = [
        Line2D([0], [0], color=median_color, linewidth=1.2, label="Median"),
        Line2D(
            [0],
            [0],
            color=mean_color,
            marker="D",
            linestyle="None",
            markersize=3,
            markeredgewidth=0,
            label="Mean",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        ncol=2,
        frameon=False,
        handlelength=1.0,
        handletextpad=0.35,
        columnspacing=0.9,
        borderaxespad=0.2,
    )

    fig.tight_layout(pad=0.45)
    output_stem = "expert_routed_weight_by_rank"
    if log_scale:
        output_stem += "_log"
    fig.savefig(output_dir / f"{output_stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(
        output_dir / f"{output_stem}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    trace_path = args.trace.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    routed_weights_by_rank = parse_trace(trace_path)
    write_statistics_csv(
        output_dir / "expert_routed_weight_by_rank.csv",
        routed_weights_by_rank,
    )
    plot_routed_weights(output_dir, routed_weights_by_rank)
    plot_routed_weights(output_dir, routed_weights_by_rank, log_scale=True)

    print(f"Trace records: {len(routed_weights_by_rank[0])}")
    for rank, weights in enumerate(routed_weights_by_rank, start=1):
        print(
            f"Rank {rank:2d}: samples={len(weights)}, "
            f"median={np.median(weights):.2%}, mean={np.mean(weights):.2%}"
        )


if __name__ == "__main__":
    main()
