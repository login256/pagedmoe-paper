#!/usr/bin/env python3
"""Plot compact Qwen3.5 fixed4 timing breakdown with cache hit rate."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from plot_qwen35_cache_size_absolute_time_breakdown import (
    BAR_KWARGS,
    COMPACT_CACHE_GIB,
    COMPONENTS,
    PALETTE,
    SCRIPT_DIR,
    configure_matplotlib,
    normalize_svg,
    read_data,
)


HIT_RATE_COLOR = "#68616F"


def plot(input_csv: Path, output_stem: Path) -> None:
    configure_matplotlib()
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.labelsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 9.5,
        }
    )
    rows_by_cache = {row.cache_size_gib: row for row in read_data(input_csv)}
    rows = [rows_by_cache[cache_size] for cache_size in COMPACT_CACHE_GIB]
    x_values = [row.cache_size_gib for row in rows]

    fig, ax = plt.subplots(figsize=(7.0, 2.45))
    bottoms = [0.0] * len(rows)
    for field, label, color, hatch in COMPONENTS:
        values = [getattr(row, field) for row in rows]
        ax.bar(
            x_values,
            values,
            bottom=bottoms,
            width=3.4,
            color=color,
            hatch=hatch,
            label=label,
            zorder=3,
            **BAR_KWARGS,
        )
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    ax_hit = ax.twinx()
    ax_hit.plot(
        x_values,
        [row.window_hit_rate for row in rows],
        color=HIT_RATE_COLOR,
        linestyle="-",
        linewidth=1.5,
        marker="o",
        markersize=4.5,
        markerfacecolor=PALETTE["white"],
        markeredgecolor=HIT_RATE_COLOR,
        markeredgewidth=0.85,
        label="Hit rate",
        zorder=5,
    )

    ax.set_xlim(4, 52)
    ax.set_ylim(0, 300)
    ax_hit.set_ylim(0.0, 1.0)
    ax.set_xticks(COMPACT_CACHE_GIB)
    ax.set_yticks(list(range(0, 301, 50)))
    ax_hit.set_yticks([index / 5 for index in range(6)])
    ax_hit.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{100.0 * value:.0f}%")
    )
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
    ax_hit.spines["top"].set_visible(False)
    ax.set_xlabel("Expert cache capacity (GiB)")
    ax.set_ylabel("Decode latency (ms/token)")
    ax_hit.tick_params(axis="y", colors=HIT_RATE_COLOR)
    ax_hit.spines["right"].set_color(HIT_RATE_COLOR)

    bar_handles, _ = ax.get_legend_handles_labels()
    hit_handles, _ = ax_hit.get_legend_handles_labels()
    legend_labels = [
        "Exposed\nSSD I/O",
        "I/O--compute\noverlap",
        "Non-overlapped\ncompute",
        "Other model\nand runtime",
        "Cache hit rate",
    ]
    fig.legend(
        bar_handles + hit_handles,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(0.84, 0.57),
        ncol=1,
        frameon=False,
        handlelength=1.1,
        handletextpad=0.4,
        borderaxespad=0.0,
        labelspacing=0.65,
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.24, right=0.76, bottom=0.23, top=0.96)
    with plt.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0}):
        fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches=None)
        fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches=None)
        svg_path = output_stem.with_suffix(".svg")
        fig.savefig(svg_path, bbox_inches=None)
    normalize_svg(svg_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=SCRIPT_DIR / "qwen35_fixed4_cache_size_absolute_time_breakdown.csv",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=SCRIPT_DIR / "fig_qwen35_cache_size_time_and_hit_rate_compact",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plot(args.input_csv, args.output_stem)
    for suffix in (".pdf", ".png", ".svg"):
        print(f"wrote {args.output_stem.with_suffix(suffix)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
