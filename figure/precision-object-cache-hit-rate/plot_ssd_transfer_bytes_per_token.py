#!/usr/bin/env python3
"""Derive SSD bitplane transfer per token from the five cache-hit series."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "precision_object_cache_hit_rate_five_series.csv"
DEFAULT_OUTPUT_STEM = SCRIPT_DIR / "fig_ssd_transfer_bytes_per_token_five_series"

GIB = 1024**3
MOE_LAYERS = 60
EXPERTS_PER_TOKEN = 10
BITPLANE_BYTES = 3 * 32 * 16 * 1024  # One plane across all expert tensors/blocks.

PALETTE = {
    "uniform4": "#0072B2",
    "dynamic4": "#009E73",
    "dynamic2p5": "#D55E00",
    "light_gray": "#DDDDDD",
    "white": "#FFFFFF",
}

SERIES_SPECS = [
    {
        "label": "Uniform 4-bit",
        "avg_bits": 4.0,
        "color": PALETTE["uniform4"],
        "linestyle": "-",
        "marker": "s",
    },
    {
        "label": "Dynamic Avg 4-bit",
        "avg_bits": 4.0,
        "color": PALETTE["dynamic4"],
        "linestyle": "-",
        "marker": "D",
    },
    {
        "label": "Dynamic Avg 4-bit w/o Prefix Reuse",
        "avg_bits": 4.0,
        "color": PALETTE["dynamic4"],
        "linestyle": "--",
        "marker": "D",
    },
    {
        "label": "Dynamic Avg 2.5-bit",
        "avg_bits": 2.5,
        "color": PALETTE["dynamic2p5"],
        "linestyle": "-",
        "marker": "o",
    },
    {
        "label": "Dynamic Avg 2.5-bit w/o Prefix Reuse",
        "avg_bits": 2.5,
        "color": PALETTE["dynamic2p5"],
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


def load_hit_rates(input_csv: Path) -> dict[str, list[tuple[int, float]]]:
    series = {spec["label"]: [] for spec in SERIES_SPECS}
    with input_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label = row["series"]
            if label in series:
                series[label].append((int(row["cache_gib"]), float(row["hit_rate"])))

    expected_cache_sizes = [8, 16, 24, 32, 40, 48, 56, 64]
    for label, values in series.items():
        values.sort()
        if [cache_gib for cache_gib, _ in values] != expected_cache_sizes:
            raise ValueError(f"incomplete cache sweep for {label}: {values}")
    return series


def transfer_bytes_per_token(hit_rate: float, avg_bits: float) -> float:
    bitplanes_per_token = MOE_LAYERS * EXPERTS_PER_TOKEN * avg_bits
    return (1.0 - hit_rate) * bitplanes_per_token * BITPLANE_BYTES


def derive_transfer_series(
    hit_rates: dict[str, list[tuple[int, float]]],
) -> dict[str, list[tuple[int, float, float]]]:
    result: dict[str, list[tuple[int, float, float]]] = {}
    for spec in SERIES_SPECS:
        result[spec["label"]] = [
            (
                cache_gib,
                hit_rate,
                transfer_bytes_per_token(hit_rate, spec["avg_bits"]),
            )
            for cache_gib, hit_rate in hit_rates[spec["label"]]
        ]
    return result


def write_data_csv(
    output_path: Path,
    transfer_series: dict[str, list[tuple[int, float, float]]],
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "series",
                "cache_gib",
                "hit_rate",
                "miss_rate",
                "avg_bits",
                "moe_layers",
                "experts_per_token",
                "bitplanes_per_token",
                "bitplane_bytes",
                "ssd_transfer_bytes_per_token",
                "ssd_transfer_gib_per_token",
                "source",
            ]
        )
        for spec in SERIES_SPECS:
            bitplanes_per_token = MOE_LAYERS * EXPERTS_PER_TOKEN * spec["avg_bits"]
            for cache_gib, hit_rate, transfer_bytes in transfer_series[spec["label"]]:
                writer.writerow(
                    [
                        spec["label"],
                        cache_gib,
                        f"{hit_rate:.8f}",
                        f"{1.0 - hit_rate:.8f}",
                        f"{spec['avg_bits']:g}",
                        MOE_LAYERS,
                        EXPERTS_PER_TOKEN,
                        f"{bitplanes_per_token:g}",
                        BITPLANE_BYTES,
                        f"{transfer_bytes:.3f}",
                        f"{transfer_bytes / GIB:.8f}",
                        DEFAULT_INPUT.name,
                    ]
                )


def plot(
    output_stem: Path,
    transfer_series: dict[str, list[tuple[int, float, float]]],
) -> None:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(3.35, 2.35))

    for spec in SERIES_SPECS:
        values = transfer_series[spec["label"]]
        ax.plot(
            [cache_gib for cache_gib, _, _ in values],
            [transfer_bytes / GIB for _, _, transfer_bytes in values],
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

    ax.set_xlim(6, 66)
    ax.set_ylim(bottom=0)
    ax.set_xticks([8, 16, 24, 32, 40, 48, 56, 64])
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
    ax.set_ylabel("SSD transfer (GiB/token)")
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
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    args = parser.parse_args()

    transfer_series = derive_transfer_series(load_hit_rates(args.input_csv))
    write_data_csv(
        args.output_stem.parent / "ssd_transfer_bytes_per_token_five_series.csv",
        transfer_series,
    )
    plot(args.output_stem, transfer_series)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
