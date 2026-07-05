#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


PALETTE = {
    "ours": "#D55E00",
    "cpu": "#4C78A8",
    "ssd": "#9D755D",
    "data_movement": "#F58518",
    "black": "#000000",
    "dark_gray": "#555555",
    "gray": "#999999",
    "light_gray": "#DDDDDD",
    "white": "#FFFFFF",
}


BAR_KWARGS = {
    "edgecolor": "#333333",
    "linewidth": 0.8,
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
            "legend.fontsize": 8.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": PALETTE["white"],
            "axes.facecolor": PALETTE["white"],
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def read_summary(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            relative_throughput = row.get(
                "relative_expert_throughput_pct",
                row.get("relative_expert_compute_util_pct", row["relative_mfu_pct"]),
            )
            rows.append(
                {
                    "hit": float(row["hit_experts"]),
                    "miss": float(row["miss_experts"]),
                    "hit_rate": 100.0
                    * float(row["hit_experts"])
                    / (float(row["hit_experts"]) + float(row["miss_experts"])),
                    "read_overlap": float(row["read_overlap_pct_of_compute"]),
                    "read_exposed": float(row["read_exposed_pct_of_compute"]),
                    "relative_throughput": float(relative_throughput),
                    "wall": float(row["wall_pct_of_compute"]),
                }
            )
    return rows


def pct_formatter(value: float, _pos: int) -> str:
    if abs(value) >= 100:
        return f"{value:.0f}"
    return f"{value:.0f}"


def plot(summary_csv: Path, output_stem: Path) -> None:
    configure_matplotlib()
    rows = read_summary(summary_csv)
    x = [row["hit_rate"] for row in rows]
    wait_read_raw = [row["read_exposed"] for row in rows]
    overlap_raw = [row["read_overlap"] for row in rows]
    tail_compute_raw = [
        max(0.0, row["wall"] - row["read_exposed"] - row["read_overlap"])
        for row in rows
    ]
    wall = [row["wall"] for row in rows]
    wait_read = [
        100.0 * value / total if total > 0 else 0.0
        for value, total in zip(wait_read_raw, wall)
    ]
    overlap = [
        100.0 * value / total if total > 0 else 0.0
        for value, total in zip(overlap_raw, wall)
    ]
    tail_compute = [
        100.0 * value / total if total > 0 else 0.0
        for value, total in zip(tail_compute_raw, wall)
    ]
    relative_throughput = [row["relative_throughput"] for row in rows]

    fig, ax = plt.subplots(figsize=(3.75, 2.35))
    bar_width = 7.0

    ax.bar(
        x,
        wait_read,
        width=bar_width,
        color=PALETTE["ssd"],
        hatch="//",
        label="Wait for SSD",
        zorder=3,
        **BAR_KWARGS,
    )
    ax.bar(
        x,
        overlap,
        bottom=wait_read,
        width=bar_width,
        color=PALETTE["data_movement"],
        hatch="\\\\",
        label="Read/compute overlap",
        zorder=3,
        **BAR_KWARGS,
    )
    ax.bar(
        x,
        tail_compute,
        bottom=[wait + ov for wait, ov in zip(wait_read, overlap)],
        width=bar_width,
        color=PALETTE["cpu"],
        hatch="",
        label="Tail compute",
        zorder=3,
        **BAR_KWARGS,
    )

    ax2 = ax.twinx()
    ax2.plot(
        x,
        relative_throughput,
        color=PALETTE["cpu"],
        linestyle="-",
        linewidth=1.45,
        marker="o",
        markersize=3.2,
        markerfacecolor=PALETTE["white"],
        markeredgecolor=PALETTE["cpu"],
        markeredgewidth=0.85,
        label="Relative expert throughput",
        zorder=4,
    )

    ax.set_xlim(min(x) - 10, max(x) + 10)
    right_ymax = max(100.0, max(relative_throughput))
    right_ymax = max(100.0, math.ceil(right_ymax / 25.0) * 25.0)
    ax.set_ylim(0, 100)
    ax2.set_ylim(0, right_ymax)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{value:.0f}%" for value in x])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax2.set_yticks(list(range(0, int(right_ymax) + 1, 25)))
    ax.yaxis.set_major_formatter(FuncFormatter(pct_formatter))
    ax2.yaxis.set_major_formatter(FuncFormatter(pct_formatter))
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
    ax2.spines["top"].set_visible(False)
    ax.set_xlabel("Expert hit rate")
    ax.set_ylabel("Share of wall time (%)")
    ax2.set_ylabel("Rel. expert throughput (%)")

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        handlelength=1.25,
        columnspacing=0.65,
        handletextpad=0.3,
        borderaxespad=0.0,
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.45)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Plot Qwen3.5-397B low-bit read/compute overlap using the paper figure style."
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=base / "qwen35_397b_int8_pipeline_read_compute_summary.csv",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=base / "qwen35_397b_int8_pipeline_read_compute",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plot(args.summary_csv, args.output_stem)
    print(f"wrote {args.output_stem.with_suffix('.pdf')}")
    print(f"wrote {args.output_stem.with_suffix('.png')}")
    print(f"wrote {args.output_stem.with_suffix('.svg')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
