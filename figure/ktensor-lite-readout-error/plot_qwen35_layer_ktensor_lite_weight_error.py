#!/usr/bin/env python3
"""Plot ktensor-lite direct-vs-readout weight error."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = SCRIPT_DIR / "qwen35_layer30_ktensor_lite_weight_error_summary.csv"
DEFAULT_OUTPUT_STEM = SCRIPT_DIR / "qwen35_layer30_ktensor_lite_weight_error"
METHOD_LABELS = {
    "direct": "Direct b-bit",
    "readout_from_pagedmoe_8bit": "8-bit readout",
    "readout_from_reencoded_8bit": "8-bit readout",
    "gptq_int4": "GPTQ INT4",
}
METHOD_STYLES = {
    "direct": {"color": "#0072B2", "marker": "o", "linestyle": "-"},
    "readout_from_pagedmoe_8bit": {
        "color": "#D55E00",
        "marker": "s",
        "linestyle": "--",
    },
    "readout_from_reencoded_8bit": {
        "color": "#D55E00",
        "marker": "s",
        "linestyle": "--",
    },
    "gptq_int4": {
        "color": "#009E73",
        "marker": "D",
        "linestyle": "None",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--tensor", choices=("all", "gate", "up", "down"), default="all")
    parser.add_argument("--linear-y", action="store_true")
    return parser.parse_args()


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
            "legend.fontsize": 9,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "savefig.dpi": 300,
            "savefig.bbox": None,
            "savefig.pad_inches": 0.02,
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_rows(path: Path, tensor: str) -> dict[str, list[tuple[int, float]]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        rows = [row for row in csv.DictReader(csv_file) if row["tensor"] == tensor]
    if not rows:
        raise ValueError(f"no rows with tensor={tensor!r} in {path}")
    series: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        method = row["method"]
        if method not in METHOD_LABELS:
            continue
        series.setdefault(method, []).append(
            (int(row["bit"]), float(row["relative_l2_error"]) * 100.0)
        )
    for values in series.values():
        values.sort()
    if "direct" not in series or not any(method.startswith("readout_from_") for method in series):
        raise ValueError(f"summary must contain direct and readout methods: {path}")
    return series


def marker_edge(method: str) -> str:
    if method == "direct":
        return "#08306B"
    if method == "gptq_int4":
        return "#005A32"
    return "#8C2D04"


def plot(
    series: dict[str, list[tuple[int, float]]],
    output_stem: Path,
    *,
    tensor: str,
    linear_y: bool,
) -> None:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(3.35, 1.95))
    method_order = ["direct"] + sorted(
        method for method in series if method.startswith("readout_from_")
    )
    if "gptq_int4" in series:
        method_order.append("gptq_int4")
    for method in method_order:
        values = series[method]
        x_values = [bit for bit, _ in values]
        y_values = [error for _, error in values]
        style = METHOD_STYLES[method]
        ax.plot(
            x_values,
            y_values,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.35,
            markersize=4.0,
            markeredgewidth=0.6,
            markeredgecolor=marker_edge(method),
            label=METHOD_LABELS[method],
        )

    ax.set_xlim(0.8, 8.2)
    ax.set_xticks(range(1, 9))
    ax.set_xlabel("Dequantized bit width")
    ax.set_ylabel("Relative weight error (%)")
    if not linear_y:
        ax.set_yscale("log")
    if tensor != "all":
        ax.set_title(tensor, pad=3, fontsize=12)
    ax.grid(
        axis="y",
        which="major",
        linestyle=(0, (2, 2)),
        color="#B8C2CC",
        linewidth=0.5,
        alpha=0.65,
    )
    ax.grid(
        axis="y",
        which="minor",
        linestyle=(0, (1, 3)),
        color="#D7DEE5",
        linewidth=0.4,
        alpha=0.45,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    legend_order = ["GPTQ INT4", "Direct b-bit", "8-bit readout"]
    ordered = []
    for label in legend_order:
        ordered.extend(
            (handle, found_label)
            for handle, found_label in zip(handles, labels)
            if found_label == label
        )
    ordered_labels = {label for _, label in ordered}
    ordered.extend(
        (handle, label)
        for handle, label in zip(handles, labels)
        if label not in ordered_labels
    )
    ax.legend(
        [handle for handle, _ in ordered],
        [label for _, label in ordered],
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        borderaxespad=0.0,
        handlelength=1.9,
        handletextpad=0.5,
        labelspacing=0.35,
        markerscale=0.9,
    )

    fig.tight_layout(pad=0.45)
    fig.subplots_adjust(bottom=fig.subplotpars.bottom - 0.02)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"))
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    series = load_rows(args.csv, args.tensor)
    output_stem = args.output_stem
    if args.tensor != "all":
        output_stem = output_stem.with_name(f"{output_stem.name}_{args.tensor}")
    plot(series, output_stem, tensor=args.tensor, linear_y=args.linear_y)
    print(f"pdf={output_stem.with_suffix('.pdf')}")
    print(f"png={output_stem.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
