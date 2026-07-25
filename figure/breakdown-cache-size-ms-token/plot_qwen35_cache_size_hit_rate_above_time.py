#!/usr/bin/env python3
"""Plot cache hit rate above the fixed4 timing breakdown with a shared x-axis."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
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


def configure_axis(axis: Axes) -> None:
    axis.grid(
        axis="y",
        which="major",
        linestyle=(0, (2, 2)),
        color=PALETTE["light_gray"],
        linewidth=0.5,
        alpha=0.8,
    )
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def plot(input_csv: Path, output_stem: Path) -> None:
    configure_matplotlib()
    rows_by_cache = {row.cache_size_gib: row for row in read_data(input_csv)}
    rows = [rows_by_cache[cache_size] for cache_size in COMPACT_CACHE_GIB]
    x_values = [row.cache_size_gib for row in rows]

    fig, (ax_hit, ax_time) = plt.subplots(
        2,
        1,
        figsize=(6.0, 4.2),
        sharex=True,
        gridspec_kw={"height_ratios": (1, 2)},
    )

    ax_hit.plot(
        x_values,
        [row.window_hit_rate for row in rows],
        color=PALETTE["dark_gray"],
        linestyle="-",
        linewidth=1.3,
        marker="o",
        markersize=4.0,
        markerfacecolor=PALETTE["white"],
        markeredgecolor=PALETTE["dark_gray"],
        markeredgewidth=0.85,
        zorder=3,
    )
    ax_hit.set_xlim(4, 52)
    ax_hit.set_ylim(0.0, 1.0)
    ax_hit.set_yticks([index / 5 for index in range(6)])
    ax_hit.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{100.0 * value:.0f}%")
    )
    ax_hit.set_ylabel("Expert cache hit rate")
    ax_hit.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_hit.text(0.01, 0.88, "(a)", transform=ax_hit.transAxes)
    configure_axis(ax_hit)

    bottoms = [0.0] * len(rows)
    for field, label, color, hatch in COMPONENTS:
        values = [getattr(row, field) for row in rows]
        ax_time.bar(
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

    ax_time.set_ylim(0, 300)
    ax_time.set_xticks(COMPACT_CACHE_GIB)
    ax_time.set_yticks(list(range(0, 301, 50)))
    ax_time.set_xlabel("Expert cache capacity (GiB)")
    ax_time.set_ylabel("Decode latency (ms/token)")
    ax_time.text(0.01, 0.93, "(b)", transform=ax_time.transAxes)
    configure_axis(ax_time)

    bar_handles, bar_labels = ax_time.get_legend_handles_labels()
    fig.legend(
        bar_handles,
        bar_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
        handlelength=1.45,
        columnspacing=1.0,
        handletextpad=0.4,
        borderaxespad=0.0,
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92), pad=0.45, h_pad=0.2)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    svg_path = output_stem.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight")
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
        default=SCRIPT_DIR / "fig_qwen35_cache_size_hit_rate_above_time_compact",
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
