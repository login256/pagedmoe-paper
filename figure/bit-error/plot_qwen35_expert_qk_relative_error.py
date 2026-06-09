#!/usr/bin/env python3
"""Plot llama.cpp Qx_K relative output error for one Qwen3.5 routed expert."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = SCRIPT_DIR / "qwen35_expert_qk_relative_error.csv"
DEFAULT_OUTPUT_STEM = SCRIPT_DIR / "qwen35_expert_qk_relative_error"
LINE_COLOR = "#0072B2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    return parser.parse_args()


def load_rows(path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    available = [row for row in rows if row["status"] == "available"]
    unavailable = [row for row in rows if row["status"] != "available"]
    if not available:
        raise ValueError(f"no available quantization results in {path}")
    return available, unavailable


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
            "legend.fontsize": 11,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot(
    output_stem: Path,
    available: list[dict[str, str]],
    unavailable: list[dict[str, str]],
) -> None:
    configure_matplotlib()
    x_values = [int(row["nominal_bits"]) - 2 for row in available]
    y_values = [float(row["relative_l2_error"]) * 100 for row in available]

    fig, ax = plt.subplots(figsize=(4.35, 2.7))
    ax.plot(
        x_values,
        y_values,
        color=LINE_COLOR,
        linewidth=1.3,
        marker="o",
        markersize=0,
        zorder=2,
    )
    ax.scatter(
        x_values,
        y_values,
        color=LINE_COLOR,
        s=17,
        edgecolors="#08306B",
        linewidths=0.6,
        zorder=3,
    )
    ax.set_xlim(-0.2, 6.2)
    ax.set_ylim(0, 40)
    ax.set_xticks(range(7))
    available_by_bit = {
        int(row["nominal_bits"]): row["quant_type"] for row in available
    }
    ax.set_xticklabels(
        [
            f"{bit} bit\n({available_by_bit.get(bit, 'none')})"
            for bit in range(2, 9)
        ],
    )
    ax.set_ylabel("Relative output error (%)")
    ax.set_axisbelow(True)
    ax.grid(
        axis="y",
        which="major",
        linestyle=(0, (2, 2)),
        color="#B8C2CC",
        linewidth=0.5,
        alpha=0.6,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=0.35)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    available, unavailable = load_rows(args.csv)
    plot(args.output_stem, available, unavailable)
    print(f"pdf={args.output_stem.with_suffix('.pdf')}")
    print(f"png={args.output_stem.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
