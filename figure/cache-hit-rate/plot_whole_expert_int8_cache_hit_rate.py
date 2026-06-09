#!/usr/bin/env python3
"""Plot whole-expert uniform INT8 cache hit rate versus cache size."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


PALETTE = {
    "line_blue": "#0072B2",
    "dark_gray": "#555555",
    "light_gray": "#DDDDDD",
    "white": "#FFFFFF",
}

MIN_CACHE_GIB = 8
MAX_CACHE_GIB = 64


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
            "legend.fontsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": PALETTE["white"],
            "axes.facecolor": PALETTE["white"],
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def read_rows(csv_path: Path) -> tuple[list[int], list[float], list[int]]:
    cache_gib: list[int] = []
    hit_rates: list[float] = []
    capacity_experts: list[int] = []
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            cache_gib.append(int(row["cache_gib"]))
            capacity_experts.append(int(row["capacity_experts"]))
            hit_rates.append(float(row["hit_rate"]))
    return cache_gib, hit_rates, capacity_experts


def plot(csv_path: Path, output_stem: Path) -> None:
    configure_matplotlib()
    cache_gib, hit_rates, capacity_experts = read_rows(csv_path)
    filtered_rows = [
        (cache, hit_rate, capacity)
        for cache, hit_rate, capacity in zip(cache_gib, hit_rates, capacity_experts)
        if MIN_CACHE_GIB <= cache <= MAX_CACHE_GIB
    ]
    cache_gib = [row[0] for row in filtered_rows]
    hit_rates = [row[1] for row in filtered_rows]
    capacity_experts = [row[2] for row in filtered_rows]
    y_values = [value * 100.0 for value in hit_rates]

    fig, ax = plt.subplots(figsize=(3.35, 2.25))
    ax.plot(
        cache_gib,
        y_values,
        color=PALETTE["line_blue"],
        linewidth=1.45,
        marker="o",
        markersize=3.2,
        markerfacecolor=PALETTE["white"],
        markeredgecolor=PALETTE["line_blue"],
        markeredgewidth=0.8,
        zorder=3,
    )
    ax.fill_between(
        cache_gib,
        y_values,
        [40.0] * len(y_values),
        color=PALETTE["line_blue"],
        alpha=0.08,
        linewidth=0,
        zorder=1,
    )

    ax.set_xlim(6, 66)
    ax.set_ylim(34, 71)
    ax.set_xticks([8, 16, 24, 32, 40, 48, 56, 64])
    ax.set_yticks([40, 50, 60, 70])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}%"))
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
    ax.set_ylabel("Expert cache hit rate")

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.45)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).with_name("whole_expert_int8_cache_hit_rate.csv"),
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=Path(__file__).with_name("fig_whole_expert_int8_cache_hit_rate"),
    )
    args = parser.parse_args()
    plot(args.csv, args.output_stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
