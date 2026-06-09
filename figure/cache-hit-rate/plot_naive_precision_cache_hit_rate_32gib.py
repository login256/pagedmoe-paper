#!/usr/bin/env python3
"""Plot naive exact-precision object cache hit rate at 32 GiB."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


PALETTE = {
    "avg3": "#9ECAE1",
    "avg4": "#6BAED6",
    "edge": "#08306B",
    "dark_gray": "#555555",
    "light_gray": "#DDDDDD",
    "white": "#FFFFFF",
}

METHOD_ORDER = ["Uniform3", "Avg3", "Uniform4", "Avg4"]
LABELS = {
    "Uniform3": "Uniform\n3-bit",
    "Avg3": "Avg3\nranked",
    "Uniform4": "Uniform\n4-bit",
    "Avg4": "Avg4\nranked",
}
BAR_COLORS = {
    "Uniform3": PALETTE["avg3"],
    "Avg3": PALETTE["avg3"],
    "Uniform4": PALETTE["avg4"],
    "Avg4": PALETTE["avg4"],
}
HATCHES = {
    "Uniform3": "",
    "Avg3": "//",
    "Uniform4": "",
    "Avg4": "//",
}


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


def read_rows(csv_path: Path) -> dict[str, dict[str, str]]:
    with csv_path.open(newline="") as handle:
        rows = {row["method"]: row for row in csv.DictReader(handle)}
    missing = [method for method in METHOD_ORDER if method not in rows]
    if missing:
        raise ValueError(f"missing methods in {csv_path}: {missing}")
    return rows


def plot(csv_path: Path, output_stem: Path) -> None:
    configure_matplotlib()
    rows = read_rows(csv_path)
    x_positions = [0.0, 1.0, 2.45, 3.45]
    hit_rates = [float(rows[method]["request_hit_rate"]) * 100.0 for method in METHOD_ORDER]

    fig, ax = plt.subplots(figsize=(3.35, 2.25))
    for x_value, method, hit_rate in zip(x_positions, METHOD_ORDER, hit_rates):
        ax.bar(
            x_value,
            hit_rate,
            width=0.74,
            color=BAR_COLORS[method],
            edgecolor=PALETTE["edge"],
            linewidth=0.8,
            hatch=HATCHES[method],
            zorder=3,
        )

    ax.set_xlim(-0.65, 4.1)
    ax.set_ylim(0, 80)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([LABELS[method] for method in METHOD_ORDER], linespacing=0.9)
    ax.set_yticks([0, 20, 40, 60, 80])
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
    ax.set_ylabel("Exact-object hit rate")

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
        default=Path(__file__).with_name("naive_precision_cache_hit_rate_32gib.csv"),
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=Path(__file__).with_name("fig_naive_precision_cache_hit_rate_32gib"),
    )
    args = parser.parse_args()
    plot(args.csv, args.output_stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
