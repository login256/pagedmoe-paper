#!/usr/bin/env python3
"""Plot mean rank-based expert-precision allocation errors over routed tokens."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = SCRIPT_DIR / "qwen35_rank_precision_relative_error.csv"
DEFAULT_OUTPUT_STEM = SCRIPT_DIR / "qwen35_rank_precision_relative_error"
COLORS = (
    "#D55E00",
    "#0072B2",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#9467BD",
    "#7F7F7F",
    "#E69F00",
    "#332288",
    "#88CCEE",
    "#44AA99",
    "#117733",
    "#AA4499",
)
HATCHES = ("", "//", "\\\\", "--", "xx", "..", "++", "oo", "||", "**", "OO", "\\\\//", "...")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not 2 <= len(rows) <= len(COLORS):
        raise ValueError(
            f"expected 2 to {len(COLORS)} strategy rows in {path}, got {len(rows)}"
        )
    return rows


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
            "xtick.major.size": 0,
            "ytick.major.size": 3,
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


def wrapped_tick_label(row: dict[str, str]) -> str:
    if row["precision_by_rank"] == "per-sample best":
        return f"{row['label']}\nper-sample best"
    if row.get("allocation_level") == "expert_id":
        bits = row["precision_by_rank"].rsplit(" ", 1)[-1]
        return f"{row['label']}\nQ{bits.replace('/', '/Q')} fixed"
    if row["strategy"].startswith("uniform_q"):
        return row["label"]
    bits = row["precision_by_rank"].split("/")
    iq1_note = "\n(1 = IQ1_M)" if "IQ1_M" in row["format_by_rank"] else ""
    return f"{row['label']}\n{'/'.join(bits[:5])}\n{'/'.join(bits[5:])}{iq1_note}"


def plot(output_stem: Path, rows: list[dict[str, str]]) -> None:
    configure_matplotlib()
    x_values = list(range(len(rows)))
    y_values = [float(row["mean_relative_l2_error"]) * 100 for row in rows]
    y_errors = [float(row["std_relative_l2_error"]) * 100 for row in rows]

    fig_width = 4.6 if len(rows) <= 3 else max(4.6, min(12.0, 0.95 * len(rows)))
    fig, ax = plt.subplots(
        figsize=(fig_width, 2.75 if len(rows) > 8 else 2.6 if len(rows) > 5 else 2.5)
    )
    bars = ax.bar(
        x_values,
        y_values,
        width=0.62,
        color=COLORS[: len(rows)],
        edgecolor="#333333",
        linewidth=0.8,
        yerr=y_errors,
        error_kw={
            "ecolor": "#333333",
            "elinewidth": 0.9,
            "capsize": 2.5,
            "capthick": 0.9,
        },
    )
    for bar, hatch in zip(bars, HATCHES[: len(rows)]):
        bar.set_hatch(hatch)

    y_max = max(value + error for value, error in zip(y_values, y_errors))
    ax.set_ylim(0, max(1.0, y_max * 1.22))
    ax.set_xticks(x_values)
    ax.set_xticklabels([wrapped_tick_label(row) for row in rows])
    ax.set_ylabel("Mean relative output error (%)")
    ax.set_axisbelow(True)
    ax.grid(
        axis="y",
        linestyle=(0, (2, 2)),
        color="#B8C2CC",
        linewidth=0.5,
        alpha=0.6,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, value, error in zip(bars, y_values, y_errors):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + error + y_max * 0.035,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10.5,
            color="#333333",
        )

    fig.tight_layout(pad=0.45)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    rows = load_rows(args.csv)
    plot(args.output_stem, rows)
    print(f"pdf={args.output_stem.with_suffix('.pdf')}")
    print(f"png={args.output_stem.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
