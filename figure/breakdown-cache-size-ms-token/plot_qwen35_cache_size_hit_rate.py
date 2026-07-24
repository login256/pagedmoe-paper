#!/usr/bin/env python3
"""Plot compact Qwen3.5 fixed4 whole-expert cache hit rate."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from plot_qwen35_cache_size_absolute_time_breakdown import (
    COMPACT_CACHE_GIB,
    PALETTE,
    SCRIPT_DIR,
    configure_matplotlib,
    normalize_svg,
    read_data,
)


def plot(input_csv: Path, output_stem: Path) -> None:
    configure_matplotlib()
    rows_by_cache = {row.cache_size_gib: row for row in read_data(input_csv)}
    rows = [rows_by_cache[cache_size] for cache_size in COMPACT_CACHE_GIB]
    x_values = [row.cache_size_gib for row in rows]
    hit_rates = [100.0 * row.window_hit_rate for row in rows]

    fig, ax = plt.subplots(figsize=(4.0, 2.3))
    ax.plot(
        x_values,
        hit_rates,
        color=PALETTE["cpu"],
        linestyle="-",
        linewidth=1.45,
        marker="o",
        markersize=4.0,
        markerfacecolor=PALETTE["white"],
        markeredgecolor=PALETTE["cpu"],
        markeredgewidth=0.85,
        zorder=3,
    )

    ax.set_xlim(4, 52)
    ax.set_ylim(30, 80)
    ax.set_xticks(COMPACT_CACHE_GIB)
    ax.set_yticks(list(range(30, 81, 10)))
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
    ax.set_xlabel("Expert cache capacity (GiB)")
    ax.set_ylabel("Expert cache hit rate")

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.45)
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
        default=SCRIPT_DIR / "fig_qwen35_cache_size_hit_rate_compact",
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
