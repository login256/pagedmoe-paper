#!/usr/bin/env python3
"""Plot naive exact-precision object cache hit rate versus cache size."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


PALETTE = {
    "avg4": "#D55E00",
    "avg3": "#0072B2",
    "uniform": "#999999",
    "light_gray": "#DDDDDD",
    "white": "#FFFFFF",
}

METHOD_ORDER = ["Avg4", "Avg3", "Uniform4", "Uniform3"]
STYLE_BY_METHOD = {
    "Uniform3": {
        "label": "Uniform3",
        "color": PALETTE["uniform"],
        "linestyle": ":",
        "marker": "o",
    },
    "Uniform4": {
        "label": "Uniform4",
        "color": PALETTE["uniform"],
        "linestyle": ":",
        "marker": "s",
    },
    "Avg3": {
        "label": "Avg3",
        "color": PALETTE["avg3"],
        "linestyle": "--",
        "marker": "o",
    },
    "Avg4": {
        "label": "Avg4",
        "color": PALETTE["avg4"],
        "linestyle": "-",
        "marker": "s",
    },
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
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": PALETTE["white"],
            "axes.facecolor": PALETTE["white"],
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def read_rows(csv_path: Path) -> dict[str, list[tuple[int, float]]]:
    series: dict[str, list[tuple[int, float]]] = {
        method: [] for method in METHOD_ORDER
    }
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            method = row["method"]
            if method not in series:
                continue
            series[method].append(
                (int(row["cache_gib"]), float(row["request_hit_rate"]) * 100.0)
            )
    for method, values in series.items():
        if not values:
            raise ValueError(f"missing method in {csv_path}: {method}")
        values.sort()
    return series


def plot(csv_path: Path, output_stem: Path) -> None:
    configure_matplotlib()
    series = read_rows(csv_path)

    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    for method in METHOD_ORDER:
        values = series[method]
        x_values = [cache_gib for cache_gib, _ in values]
        y_values = [hit_rate for _, hit_rate in values]
        style = STYLE_BY_METHOD[method]
        ax.plot(
            x_values,
            y_values,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.45,
            marker=style["marker"],
            markersize=3.0,
            markerfacecolor=PALETTE["white"],
            markeredgecolor=style["color"],
            markeredgewidth=0.75,
            label=style["label"],
            zorder=3,
        )

    ax.set_xlim(6, 66)
    ax.set_ylim(10, 92)
    ax.set_xticks([8, 16, 24, 32, 40, 48, 56, 64])
    ax.set_yticks([20, 40, 60, 80])
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
    ax.set_ylabel("Exact-object hit rate")
    ax.legend(
        loc="upper left",
        ncol=2,
        frameon=False,
        handlelength=1.6,
        columnspacing=0.8,
        handletextpad=0.4,
        borderaxespad=0.2,
    )

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
        default=Path(__file__).with_name("naive_precision_cache_hit_rate_sweep.csv"),
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=Path(__file__).with_name("fig_naive_precision_cache_hit_rate_sweep"),
    )
    args = parser.parse_args()
    plot(args.csv, args.output_stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
