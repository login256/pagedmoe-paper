#!/usr/bin/env python3
"""Plot the same-layer expert reuse-distance CDF from a precision trace.

For an expert access at token i, the reuse distance is i - j, where j is the
previous token that accessed the same expert in the same layer. Thus, distance
N means that the expert appeared within the previous N tokens. First accesses
have no finite reuse distance and are excluded from the CDF.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_TRACE = (
    REPO_ROOT
    / "src/target/bench/precision_trace_c4l10_20260518_122725"
    / "precision_trace.log"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR
TRACE_PATTERN = re.compile(
    r"\[pagedmoe-precision-trace\].*?"
    r"\blayer=Some\((?P<layer>\d+)\).*?"
    r"\bexperts=\[(?P<experts>[^\]]*)\]"
)
GUIDE_PERCENTILES = (0.60, 0.80)
REPORTED_PERCENTILES = (0.60, 0.70, 0.80)
MAX_PLOTTED_DISTANCE = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the same-layer expert reuse-distance CDF."
    )
    parser.add_argument(
        "--trace",
        type=Path,
        default=DEFAULT_TRACE,
        help=f"precision trace to parse (default: {DEFAULT_TRACE})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"directory for generated files (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def parse_trace(trace_path: Path) -> tuple[list[int], int, int, set[int]]:
    """Return reuse distances, record count, access count, and observed layers."""
    last_positions: dict[tuple[int, int], int] = {}
    layer_token_positions: defaultdict[int, int] = defaultdict(int)
    reuse_distances: list[int] = []
    record_count = 0
    access_count = 0

    with trace_path.open(encoding="utf-8", errors="replace") as trace_file:
        for line in trace_file:
            match = TRACE_PATTERN.search(line)
            if match is None:
                continue

            layer = int(match.group("layer"))
            experts = [
                int(value.strip())
                for value in match.group("experts").split(",")
                if value.strip()
            ]
            if not experts:
                raise ValueError(f"record {record_count + 1} has no experts")

            record_count += 1
            for expert in experts:
                position = layer_token_positions[layer]
                key = (layer, expert)
                if key in last_positions:
                    reuse_distances.append(position - last_positions[key])
                last_positions[key] = position
                access_count += 1
            layer_token_positions[layer] += 1

    if record_count == 0:
        raise ValueError(f"no precision-trace records found in {trace_path}")
    if not reuse_distances:
        raise ValueError(f"no repeated same-layer expert accesses found in {trace_path}")

    return reuse_distances, record_count, access_count, set(layer_token_positions)


def build_cdf(reuse_distances: list[int]) -> tuple[list[int], list[float], list[int]]:
    counts = Counter(reuse_distances)
    distances = sorted(counts)
    cumulative_counts: list[int] = []
    cdf: list[float] = []
    cumulative = 0

    for distance in distances:
        cumulative += counts[distance]
        cumulative_counts.append(cumulative)
        cdf.append(cumulative / len(reuse_distances))

    return distances, cdf, cumulative_counts


def percentile_distances(
    distances: list[int], cdf: list[float]
) -> list[tuple[float, int]]:
    thresholds: list[tuple[float, int]] = []
    for percentile in REPORTED_PERCENTILES:
        for distance, probability in zip(distances, cdf):
            if probability >= percentile:
                thresholds.append((percentile, distance))
                break
    return thresholds


def write_cdf_csv(
    output_path: Path,
    distances: list[int],
    cdf: list[float],
    cumulative_counts: list[int],
    access_count: int,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file, lineterminator="\n")
        writer.writerow(
            [
                "reuse_distance_tokens",
                "cdf_of_all_accesses",
                "cdf_of_recurring_accesses",
                "cumulative_recurring_accesses",
            ]
        )
        for distance, probability, cumulative in zip(
            distances, cdf, cumulative_counts
        ):
            writer.writerow(
                [
                    distance,
                    f"{cumulative / access_count:.8f}",
                    f"{probability:.8f}",
                    cumulative,
                ]
            )


def plot_cdf(
    output_dir: Path,
    distances: list[int],
    recurring_access_cdf: list[float],
    guides: list[tuple[int, float]],
    output_stem: str,
    linear_scale: bool = False,
) -> None:
    max_tick = MAX_PLOTTED_DISTANCE
    probability_by_distance = dict(zip(distances, recurring_access_cdf))
    x_values = list(range(max_tick + 1))
    y_values: list[float] = []
    probability = 0.0
    for distance in x_values:
        probability = probability_by_distance.get(distance, probability)
        y_values.append(probability)
    if linear_scale:
        ticks = sorted(
            set(range(0, max_tick + 1, 32))
            | {distance for distance, _ in guides}
        )
    else:
        guide_distances = {distance for distance, _ in guides}
        power_of_two_ticks = {
            1 << exponent for exponent in range(int(math.log2(max_tick)) + 1)
        }
        ticks = sorted(
            {
                tick
                for tick in power_of_two_ticks
                if all(abs(tick - distance) > 1 for distance in guide_distances)
            }
            | {0}
            | guide_distances
        )

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
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    color = "#0072B2"

    ax.plot(
        x_values,
        y_values,
        color=color,
        linewidth=1.5,
    )
    for distance, probability in guides:
        ax.plot(
            distance,
            probability,
            "o",
            color=color,
            markersize=3.5,
            zorder=3,
        )
        ax.vlines(
            distance,
            ymin=0,
            ymax=probability,
            colors=color,
            linestyles=":",
            linewidth=0.8,
            alpha=0.8,
        )

    if not linear_scale:
        ax.set_xscale("symlog", base=2, linthresh=1)
    ax.set_xlim(0, max_tick)
    ax.set_ylim(0, 1.03)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value)}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.grid(axis="y", which="major", linestyle="--", linewidth=0.45, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Reuse distance (tokens)")
    ax.set_ylabel("Cumulative fraction")

    fig.tight_layout()
    fig.savefig(output_dir / f"{output_stem}.pdf", bbox_inches="tight")
    fig.savefig(
        output_dir / f"{output_stem}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    trace_path = args.trace.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reuse_distances, record_count, access_count, layers = parse_trace(trace_path)
    distances, recurring_access_cdf, cumulative_counts = build_cdf(reuse_distances)
    recurring_thresholds = percentile_distances(distances, recurring_access_cdf)
    guide_distances = [
        distance
        for percentile, distance in recurring_thresholds
        if percentile in GUIDE_PERCENTILES
    ]
    probability_by_distance = dict(zip(distances, recurring_access_cdf))
    guides = [
        (distance, probability_by_distance[distance])
        for distance in guide_distances
    ]
    cold_accesses = access_count - len(reuse_distances)

    write_cdf_csv(
        output_dir / "expert_reuse_distance_cdf.csv",
        distances,
        recurring_access_cdf,
        cumulative_counts,
        access_count,
    )
    plot_cdf(
        output_dir,
        distances,
        recurring_access_cdf,
        guides,
        "expert_reuse_distance_cdf",
    )
    plot_cdf(
        output_dir,
        distances,
        recurring_access_cdf,
        guides,
        "expert_reuse_distance_cdf_linear",
        linear_scale=True,
    )

    print(f"Trace records: {record_count}")
    print(f"Layers: {len(layers)}")
    print(f"Expert accesses: {access_count}")
    print(
        f"Recurring expert accesses: {len(reuse_distances)} "
        f"({len(reuse_distances) / access_count:.1%})"
    )
    print(
        f"First accesses without reuse distance: {cold_accesses} "
        f"({cold_accesses / access_count:.1%})"
    )
    for percentile, distance in recurring_thresholds:
        print(
            f"{percentile:.0%} of recurring expert accesses reuse an expert "
            f"from the previous {distance} tokens."
        )
    print("Guide points plotted on the CDF:")
    for distance, probability in guides:
        print(f"  distance={distance} tokens, CDF={probability:.2%}")


if __name__ == "__main__":
    main()
