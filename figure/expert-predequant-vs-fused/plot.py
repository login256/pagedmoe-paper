#!/usr/bin/env python3
"""Plot the expert-predequant ablation using the paper's cache-size style."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data.csv"
OUTPUT_STEM = ROOT / "fig_expert_predequant_vs_fused"
EXPECTED_GRID = tuple(float(value) for value in range(4, 49, 4))

PALETTE = {
    "fused": "#D55E00",
    "expert": "#0072B2",
    "grid": "#DDDDDD",
    "white": "#FFFFFF",
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Linux Libertine O", "Libertinus Serif", "STIXGeneral"],
            "font.size": 12,
            "axes.labelsize": 10,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": PALETTE["white"],
            "axes.facecolor": PALETTE["white"],
            "savefig.dpi": 300,
            "savefig.bbox": None,
            "savefig.pad_inches": 0.04,
        }
    )


def load_data() -> tuple[list[float], list[float], list[float]]:
    with DATA.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    caches = [float(row["cache_size_gib"]) for row in rows]
    if tuple(caches) != EXPECTED_GRID:
        raise ValueError(f"expected cache grid {EXPECTED_GRID}, got {tuple(caches)}")
    fused = [float(row["fused_decode_token_per_s"]) for row in rows]
    expert = [float(row["expert_predequant_decode_token_per_s"]) for row in rows]
    return caches, fused, expert


def draw_series(
    ax: plt.Axes,
    x_values: list[float],
    y_values: list[float],
    *,
    color: str,
    linestyle: str,
    marker: str,
    label: str,
) -> None:
    ax.plot(
        x_values,
        y_values,
        color=color,
        linestyle=linestyle,
        linewidth=1.45,
        marker=marker,
        markersize=3.0,
        markerfacecolor=PALETTE["white"],
        markeredgecolor=color,
        markeredgewidth=0.75,
        label=label,
        zorder=3,
    )


def main() -> None:
    configure_matplotlib()
    caches, fused, expert = load_data()
    fig, ax = plt.subplots(figsize=(3.35, 2.10))

    draw_series(
        ax,
        caches,
        fused,
        color=PALETTE["fused"],
        linestyle="-",
        marker="o",
        label="Direct",
    )
    draw_series(
        ax,
        caches,
        expert,
        color=PALETTE["expert"],
        linestyle="--",
        marker="s",
        label="Materialize",
    )

    ax.set_xlim(0, 50)
    ax.set_ylim(0, 12.5)
    ax.set_xticks(list(range(0, 49, 8)))
    ax.set_yticks([0, 2, 4, 6, 8, 10, 12])
    ax.grid(
        axis="y",
        linestyle=(0, (2, 2)),
        color=PALETTE["grid"],
        linewidth=0.5,
        alpha=0.8,
    )
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Expert cache capacity (GiB)")
    ax.set_ylabel("Decode throughput (tokens/s)")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        ncol=2,
        frameon=False,
        handlelength=2.2,
        columnspacing=0.55,
        handletextpad=0.3,
        borderaxespad=0.0,
    )
    fig.tight_layout(pad=0.45)
    fig.subplots_adjust(bottom=fig.subplotpars.bottom - 0.02)
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"))
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
