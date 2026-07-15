#!/usr/bin/env python3
"""Plot five cache hit-rate series for static and dynamic precision policies."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
AVG4_SERIES_CSV = SCRIPT_DIR / "precision_object_cache_hit_rate_plot_series.csv"
BREAKDOWN_SUMMARY = REPO_ROOT / "logs/qwen35_breakdown_combined_20260701/summary.md"
CACHE_SIZES = [8, 16, 24, 32, 40, 48, 56, 64]

PALETTE = {
    "uniform": "#0072B2",
    "avg4": "#009E73",
    "avg4_naive": "#009E73",
    "avg2p5": "#D55E00",
    "avg2p5_naive": "#D55E00",
    "light_gray": "#DDDDDD",
    "white": "#FFFFFF",
}

SERIES_SPECS = [
    {
        "key": "uniform4",
        "label": "Uniform 4-bit",
        "color": PALETTE["uniform"],
        "linestyle": "-",
        "marker": "s",
    },
    {
        "key": "dynamic4",
        "label": "Dynamic Avg 4-bit",
        "color": PALETTE["avg4"],
        "linestyle": "-",
        "marker": "D",
    },
    {
        "key": "dynamic4_no_prefix",
        "label": "Dynamic Avg 4-bit w/o Prefix Reuse",
        "color": PALETTE["avg4_naive"],
        "linestyle": "--",
        "marker": "D",
    },
    {
        "key": "dynamic2p5",
        "label": "Dynamic Avg 2.5-bit",
        "color": PALETTE["avg2p5"],
        "linestyle": "-",
        "marker": "o",
    },
    {
        "key": "dynamic2p5_no_prefix",
        "label": "Dynamic Avg 2.5-bit w/o Prefix Reuse",
        "color": PALETTE["avg2p5_naive"],
        "linestyle": "--",
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
            "legend.fontsize": 6.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": PALETTE["white"],
            "axes.facecolor": PALETTE["white"],
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def split_markdown_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def read_markdown_table(path: Path, heading: str) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    heading_index = lines.index(heading)
    table_lines: list[str] = []
    for line in lines[heading_index + 1 :]:
        if line.startswith("|"):
            table_lines.append(line)
        elif table_lines:
            break
    if len(table_lines) < 3:
        raise ValueError(f"no table after {heading} in {path}")
    header = split_markdown_row(table_lines[0])
    return [
        dict(zip(header, split_markdown_row(line), strict=True))
        for line in table_lines[2:]
    ]


def read_avg4_series() -> dict[str, list[tuple[int, float]]]:
    label_to_key = {
        "Avg4": "dynamic4",
        "Avg4 Naive Cache": "dynamic4_no_prefix",
    }
    result = {key: [] for key in label_to_key.values()}
    with AVG4_SERIES_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = label_to_key.get(row["series"])
            cache_gib = int(row["cache_gib"])
            if key is None or cache_gib not in CACHE_SIZES:
                continue
            result[key].append((cache_gib, float(row["hit_rate"])))
    return result


def read_breakdown_series() -> dict[str, list[tuple[int, float]]]:
    c2p5_rows = read_markdown_table(BREAKDOWN_SUMMARY, "## c2.5l25 Cache Metrics")
    static4_rows = read_markdown_table(BREAKDOWN_SUMMARY, "## c4/static4 Cache Metrics")
    return {
        "uniform4": [
            (int(row["cache GiB"]), float(row["bitplane static4 hit"]))
            for row in static4_rows
        ],
        "dynamic2p5": [
            (int(row["cache GiB"]), float(row["bitplane hit"]))
            for row in c2p5_rows
        ],
        "dynamic2p5_no_prefix": [
            (int(row["cache GiB"]), float(row["naive hit"]))
            for row in c2p5_rows
        ],
    }


def load_series() -> dict[str, list[tuple[int, float]]]:
    series = {**read_avg4_series(), **read_breakdown_series()}
    for spec in SERIES_SPECS:
        values = sorted(series.get(spec["key"], []))
        if [cache_gib for cache_gib, _ in values] != CACHE_SIZES:
            raise ValueError(f"incomplete cache sweep for {spec['key']}: {values}")
        series[spec["key"]] = values
    return series


def write_data_csv(
    output_path: Path, series: dict[str, list[tuple[int, float]]]
) -> None:
    sources = {
        "uniform4": BREAKDOWN_SUMMARY.relative_to(REPO_ROOT),
        "dynamic4": AVG4_SERIES_CSV.relative_to(REPO_ROOT),
        "dynamic4_no_prefix": AVG4_SERIES_CSV.relative_to(REPO_ROOT),
        "dynamic2p5": BREAKDOWN_SUMMARY.relative_to(REPO_ROOT),
        "dynamic2p5_no_prefix": BREAKDOWN_SUMMARY.relative_to(REPO_ROOT),
    }
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["series", "cache_gib", "hit_rate", "source"])
        for spec in SERIES_SPECS:
            for cache_gib, hit_rate in series[spec["key"]]:
                writer.writerow(
                    [
                        spec["label"],
                        cache_gib,
                        f"{hit_rate:.8f}",
                        sources[spec["key"]],
                    ]
                )


def plot(output_stem: Path, series: dict[str, list[tuple[int, float]]]) -> None:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(3.35, 2.35))

    for spec in SERIES_SPECS:
        values = series[spec["key"]]
        ax.plot(
            [cache_gib for cache_gib, _ in values],
            [hit_rate * 100.0 for _, hit_rate in values],
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

    y_values = [hit_rate * 100.0 for values in series.values() for _, hit_rate in values]
    y_min = max(0, math.floor((min(y_values) - 4.0) / 10.0) * 10)
    y_max = min(100, math.ceil((max(y_values) + 4.0) / 10.0) * 10)
    ax.set_xlim(6, 66)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(CACHE_SIZES)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}%"))
    ax.grid(
        axis="y",
        linestyle=(0, (2, 2)),
        color=PALETTE["light_gray"],
        linewidth=0.5,
        alpha=0.8,
    )
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Cache size (GiB)")
    ax.set_ylabel("Cache hit rate")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        handlelength=1.45,
        columnspacing=0.55,
        handletextpad=0.3,
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
        "--output-stem",
        type=Path,
        default=SCRIPT_DIR / "fig_precision_object_cache_hit_rate_five_series",
    )
    args = parser.parse_args()

    series = load_series()
    write_data_csv(
        args.output_stem.parent / "precision_object_cache_hit_rate_five_series.csv",
        series,
    )
    plot(args.output_stem, series)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
