#!/usr/bin/env python3
"""Plot decode throughput versus cache size for static and dynamic caches."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_SUMMARY = (
    REPO_ROOT
    / "src/target/bench/kt_pagedmoe_dynamic_static_cache8-64_perf_hitrate_summary.csv"
)

PALETTE = {
    "ours": "#D55E00",
    "strong_baseline": "#009E73",
    "static": "#0072B2",
    "black": "#000000",
    "light_gray": "#DDDDDD",
    "white": "#FFFFFF",
}

SERIES = [
    {
        "case": "ik_llama.cpp-mmap-Q4_0",
        "label": "ik_llama.cpp Q4_0",
        "color": PALETTE["black"],
        "linestyle": "-.",
        "marker": "^",
        "manual_values": [(16.0, 1.21), (32.0, 1.50), (48.0, 1.99), (64.0, 2.32)],
    },
    {
        "case": "all4",
        "label": "Uniform 4bit",
        "color": PALETTE["static"],
        "linestyle": "-",
        "marker": "s",
    },
    {
        "case": "c4l40",
        "label": "Dynamic Avg 4bit",
        "color": PALETTE["strong_baseline"],
        "linestyle": "-",
        "marker": "D",
    },
    {
        "case": "c2.5l25",
        "label": "Dynamic Avg 2.5bit",
        "color": PALETTE["ours"],
        "linestyle": "-",
        "marker": "o",
    },
]


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
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.fontsize": 7.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": PALETTE["white"],
            "axes.facecolor": PALETTE["white"],
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def read_summary(summary_csv: Path) -> dict[str, list[tuple[float, float]]]:
    values_by_case: dict[str, defaultdict[float, list[float]]] = {
        spec["case"]: defaultdict(list)
        for spec in SERIES
        if "manual_values" not in spec
    }

    with summary_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            case = row["case"]
            if case not in values_by_case or row["status"] != "ok":
                continue
            values_by_case[case][float(row["cache_size_gib"])].append(
                float(row["decode_token_per_s"])
            )

    series: dict[str, list[tuple[float, float]]] = {}
    for spec in SERIES:
        if "manual_values" in spec:
            series[spec["case"]] = sorted(spec["manual_values"])

    for case, values_by_cache in values_by_case.items():
        if not values_by_cache:
            raise ValueError(f"missing case in {summary_csv}: {case}")
        series[case] = [
            (cache_gib, sum(tokens_per_s) / len(tokens_per_s))
            for cache_gib, tokens_per_s in sorted(values_by_cache.items())
        ]
    return series


def plot(summary_csv: Path, output_stem: Path) -> None:
    configure_matplotlib()
    series = read_summary(summary_csv)

    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    y_values_all: list[float] = []

    for spec in SERIES:
        values = series[spec["case"]]
        x_values = [cache_gib for cache_gib, _ in values]
        y_values = [tokens_per_s for _, tokens_per_s in values]
        y_values_all.extend(y_values)

        ax.plot(
            x_values,
            y_values,
            color=spec["color"],
            linestyle=spec["linestyle"],
            linewidth=1.45,
            marker=spec["marker"],
            markersize=3.0,
            markerfacecolor=PALETTE["white"],
            markeredgecolor=spec["color"],
            markeredgewidth=0.75,
            label=spec["label"],
            zorder=3,
        )

    y_min = max(0.0, min(y_values_all) - 0.4)
    y_max = max(y_values_all) + 0.5
    ax.set_xlim(6, 66)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([8, 16, 24, 32, 40, 48, 56, 64])
    ax.set_yticks([1, 2, 4, 6, 8, 10])
    ax.grid(
        axis="y",
        which="major",
        linestyle=(0, (2, 2)),
        color=PALETTE["light_gray"],
        linewidth=0.5,
        alpha=0.8,
    )
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Cache size (GiB)")
    ax.set_ylabel("Decode throughput (token/s)")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        handlelength=1.5,
        columnspacing=0.65,
        handletextpad=0.35,
        borderaxespad=0.0,
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.45)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=DEFAULT_SUMMARY,
        help=f"benchmark summary CSV (default: {DEFAULT_SUMMARY})",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=SCRIPT_DIR / "fig_cache_size_token_s",
        help="output figure stem without extension",
    )
    args = parser.parse_args()
    plot(args.summary_csv, args.output_stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
