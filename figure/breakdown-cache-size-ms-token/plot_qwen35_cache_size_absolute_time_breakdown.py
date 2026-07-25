#!/usr/bin/env python3
"""Plot the Qwen3.5 fixed4 absolute per-token activity breakdown."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
EXPECTED_CACHE_GIB = tuple(range(4, 49, 4))
COMPACT_CACHE_GIB = tuple(range(8, 49, 8))

PALETTE = {
    "cpu": "#4C78A8",
    "ssd": "#9D755D",
    "data_movement": "#F58518",
    "other": "#BAB0AC",
    "dark_gray": "#555555",
    "light_gray": "#DDDDDD",
    "white": "#FFFFFF",
}

BAR_KWARGS = {
    "edgecolor": "#333333",
    "linewidth": 0.8,
}

COMPONENTS = (
    ("ssd_only_ms", "Exposed SSD I/O", PALETTE["ssd"], "//"),
    (
        "ssd_compute_overlap_ms",
        "I/O-computation overlap",
        PALETTE["data_movement"],
        "\\\\",
    ),
    (
        "compute_only_ms",
        "Non-overlapped expert computation",
        PALETTE["cpu"],
        "",
    ),
    ("other_ms", "Other model and runtime work", PALETTE["other"], ".."),
)


@dataclass(frozen=True)
class TimingRow:
    cache_size_gib: int
    window_tps: float
    window_hit_rate: float
    window_tokens: int
    total_ms: float
    ssd_only_ms: float
    compute_only_ms: float
    ssd_compute_overlap_ms: float
    other_ms: float


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
            "legend.fontsize": 9.0,
            "hatch.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": PALETTE["white"],
            "axes.facecolor": PALETTE["white"],
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def read_data(path: Path) -> list[TimingRow]:
    rows: list[TimingRow] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for line_number, raw in enumerate(csv.DictReader(handle), start=2):
            try:
                row = TimingRow(
                    cache_size_gib=int(raw["cache_size_gib"]),
                    window_tps=float(raw["window_tps"]),
                    window_hit_rate=float(raw["window_hit_rate"]),
                    window_tokens=int(raw["window_tokens"]),
                    total_ms=float(raw["total_ms_per_token"]),
                    ssd_only_ms=float(raw["ssd_only_ms_per_token"]),
                    compute_only_ms=float(raw["compute_only_ms_per_token"]),
                    ssd_compute_overlap_ms=float(
                        raw["ssd_compute_overlap_ms_per_token"]
                    ),
                    other_ms=float(raw["other_ms_per_token"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid timing row at {path}:{line_number}") from exc
            rows.append(row)

    rows.sort(key=lambda row: row.cache_size_gib)
    cache_grid = tuple(row.cache_size_gib for row in rows)
    if cache_grid != EXPECTED_CACHE_GIB:
        raise ValueError(
            f"expected cache grid {EXPECTED_CACHE_GIB}, found {cache_grid} in {path}"
        )

    for row in rows:
        if (
            not math.isfinite(row.window_hit_rate)
            or not 0.0 <= row.window_hit_rate <= 1.0
        ):
            raise ValueError(f"invalid hit rate at {row.cache_size_gib} GiB")
        if row.window_tokens != 500:
            raise ValueError(f"expected 500 window tokens at {row.cache_size_gib} GiB")
        if not math.isfinite(row.window_tps) or row.window_tps <= 0.0:
            raise ValueError(f"invalid window TPS at {row.cache_size_gib} GiB")
        components = (
            row.ssd_only_ms,
            row.compute_only_ms,
            row.ssd_compute_overlap_ms,
            row.other_ms,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in components):
            raise ValueError(f"invalid timing component at {row.cache_size_gib} GiB")
        if not math.isclose(sum(components), row.total_ms, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                f"timing components do not sum to total at {row.cache_size_gib} GiB"
            )
        if not math.isclose(
            row.total_ms, 1000.0 / row.window_tps, rel_tol=0.0, abs_tol=1e-5
        ):
            raise ValueError(
                f"total time and window TPS disagree at {row.cache_size_gib} GiB"
            )
    return rows


def normalize_svg(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def plot(input_csv: Path, output_stem: Path, *, compact: bool = False) -> None:
    configure_matplotlib()
    rows = read_data(input_csv)
    if compact:
        rows_by_cache = {row.cache_size_gib: row for row in rows}
        rows = [rows_by_cache[cache_size] for cache_size in COMPACT_CACHE_GIB]
    x_values = [row.cache_size_gib for row in rows]

    figure_size = (4.0, 2.3) if compact else (6.0, 3.45)
    bar_width = 3.4 if compact else 1.7
    x_limits = (4, 52) if compact else (0, 50)
    x_ticks = list(COMPACT_CACHE_GIB if compact else range(0, 49, 4))

    fig, ax = plt.subplots(figsize=figure_size)
    bottoms = [0.0] * len(rows)
    for field, label, color, hatch in COMPONENTS:
        values = [getattr(row, field) for row in rows]
        ax.bar(
            x_values,
            values,
            bottom=bottoms,
            width=bar_width,
            color=color,
            hatch=hatch,
            label=label,
            zorder=3,
            **BAR_KWARGS,
        )
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    ax.set_xlim(*x_limits)
    ax.set_ylim(0, 300)
    ax.set_xticks(x_ticks)
    ax.set_yticks(list(range(0, 301, 50)))
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
    ax.set_ylabel("Decode latency (ms/token)")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=2,
        frameon=False,
        handlelength=1.45,
        columnspacing=1.0,
        handletextpad=0.4,
        borderaxespad=0.0,
    )

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
        default=None,
        help="Output path without a suffix (default depends on --compact)",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Plot only the 8, 16, 24, 32, 40, and 48 GiB points",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    default_name = "fig_qwen35_cache_size_absolute_time_breakdown"
    if args.compact:
        default_name += "_compact"
    output_stem = args.output_stem or SCRIPT_DIR / default_name
    plot(args.input_csv, output_stem, compact=args.compact)
    for suffix in (".pdf", ".png", ".svg"):
        print(f"wrote {output_stem.with_suffix(suffix)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
