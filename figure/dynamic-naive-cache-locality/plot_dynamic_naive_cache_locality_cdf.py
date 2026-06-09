#!/usr/bin/env python3
"""Plot expert-precision object reuse-distance CDF from a precision trace.

This figure uses the expert choices from the precision trace, but does not use
the precision choices recorded in the trace. Each strategy assigns precision by
expert rank, and the cache object is (layer, expert, precision). Therefore, the
same expert at different precisions is counted as a different object.

Strategies:

    avg4:    (8, 7, 6, 5, 4, 3, 3, 2, 1, 1)
    avg3:    (6, 6, 5, 4, 3, 2, 1, 1, 1, 1)
    uniform: all ranks use the same precision, so the object reduces to
             (layer, expert) for locality purposes.

Reuse distance is measured in tokens within the same layer. First accesses have
no finite reuse distance and are excluded from the CDF.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
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
STRATEGIES: tuple[tuple[str, str, tuple[int, ...] | None], ...] = (
    ("avg4", "Avg4", (8, 7, 6, 5, 4, 3, 3, 2, 1, 1)),
    ("avg3", "Avg3", (6, 6, 5, 4, 3, 2, 1, 1, 1, 1)),
    ("uniform", "Uniform", None),
)
STYLE_BY_STRATEGY = {
    "avg4": {"color": "#D55E00", "linestyle": "-", "marker": "o"},
    "avg3": {"color": "#0072B2", "linestyle": "--", "marker": "s"},
    "uniform": {"color": "#999999", "linestyle": ":", "marker": "x"},
}
GUIDE_PERCENTILES = (0.60, 0.80)
REPORTED_PERCENTILES = (0.60, 0.70, 0.80)
MAX_PLOTTED_DISTANCE = 48


@dataclass(frozen=True)
class CdfSeries:
    name: str
    label: str
    precision_by_rank: tuple[int, ...] | None
    distances: list[int]
    cdf: list[float]
    cumulative_counts: list[int]
    record_count: int
    access_count: int
    recurring_count: int
    layer_count: int
    thresholds: list[tuple[float, int]]

    @property
    def cold_accesses(self) -> int:
        return self.access_count - self.recurring_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot same-layer reuse-distance CDF for dynamic-naive "
            "expert-precision objects."
        )
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


def parse_trace(
    trace_path: Path,
    precision_by_rank: tuple[int, ...] | None,
) -> tuple[list[int], int, int, set[int]]:
    """Return reuse distances, record count, access count, and observed layers."""
    last_positions: dict[tuple[int, int] | tuple[int, int, int], int] = {}
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
            if precision_by_rank is not None and len(experts) != len(precision_by_rank):
                raise ValueError(
                    f"record {record_count + 1} has {len(experts)} experts, "
                    f"expected {len(precision_by_rank)}"
                )

            record_count += 1
            position = layer_token_positions[layer]
            for rank, expert in enumerate(experts):
                if precision_by_rank is None:
                    key = (layer, expert)
                else:
                    key = (layer, expert, precision_by_rank[rank])
                if key in last_positions:
                    reuse_distances.append(position - last_positions[key])
                last_positions[key] = position
                access_count += 1
            layer_token_positions[layer] += 1

    if record_count == 0:
        raise ValueError(f"no precision-trace records found in {trace_path}")
    if not reuse_distances:
        raise ValueError(f"no repeated expert-precision objects found in {trace_path}")

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


def build_series(
    trace_path: Path,
    name: str,
    label: str,
    precision_by_rank: tuple[int, ...] | None,
) -> CdfSeries:
    reuse_distances, record_count, access_count, layers = parse_trace(
        trace_path, precision_by_rank
    )
    distances, cdf, cumulative_counts = build_cdf(reuse_distances)
    thresholds = percentile_distances(distances, cdf)
    return CdfSeries(
        name=name,
        label=label,
        precision_by_rank=precision_by_rank,
        distances=distances,
        cdf=cdf,
        cumulative_counts=cumulative_counts,
        record_count=record_count,
        access_count=access_count,
        recurring_count=len(reuse_distances),
        layer_count=len(layers),
        thresholds=thresholds,
    )


def write_cdf_csv(output_path: Path, series_list: list[CdfSeries]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file, lineterminator="\n")
        writer.writerow(
            [
                "strategy",
                "precision_by_rank",
                "reuse_distance_tokens",
                "cdf_of_all_accesses",
                "cdf_of_recurring_accesses",
                "cumulative_recurring_accesses",
            ]
        )
        for series in series_list:
            precision_label = precision_to_text(series.precision_by_rank)
            for distance, probability, cumulative in zip(
                series.distances,
                series.cdf,
                series.cumulative_counts,
            ):
                writer.writerow(
                    [
                        series.name,
                        precision_label,
                        distance,
                        f"{cumulative / series.access_count:.8f}",
                        f"{probability:.8f}",
                        cumulative,
                    ]
                )


def configure_style() -> None:
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
            "legend.fontsize": 11,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def cdf_grid(series: CdfSeries) -> tuple[list[int], list[float]]:
    probability_by_distance = dict(zip(series.distances, series.cdf))
    x_values = list(range(MAX_PLOTTED_DISTANCE + 1))
    y_values: list[float] = []
    probability = 0.0
    for distance in x_values:
        probability = probability_by_distance.get(distance, probability)
        y_values.append(probability)
    return x_values, y_values


def plot_cdf(output_dir: Path, series_list: list[CdfSeries], output_stem: str) -> None:
    ticks = [0, 12, 24, 36, 48]
    configure_style()
    fig, ax = plt.subplots(figsize=(3.3, 2.2))

    for series in series_list:
        x_values, y_values = cdf_grid(series)
        style = STYLE_BY_STRATEGY[series.name]
        ax.plot(
            x_values,
            y_values,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.5,
            label=series.label,
        )
        for percentile, distance in series.thresholds:
            if percentile not in GUIDE_PERCENTILES:
                continue
            probability = dict(zip(series.distances, series.cdf))[distance]
            ax.plot(
                distance,
                probability,
                marker=style["marker"],
                color=style["color"],
                markersize=3.0,
                linestyle="None",
                zorder=3,
            )

    ax.set_xlim(0, MAX_PLOTTED_DISTANCE)
    ax.set_ylim(0, 1.03)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value)}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.grid(axis="y", which="major", linestyle="--", linewidth=0.45, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Reuse distance (tokens)")
    ax.set_ylabel("Cumulative fraction")
    ax.legend(loc="lower right", frameon=False, handlelength=2.2)

    fig.tight_layout()
    fig.savefig(output_dir / f"{output_stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{output_stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def precision_to_text(precision_by_rank: tuple[int, ...] | None) -> str:
    if precision_by_rank is None:
        return "uniform"
    return "/".join(str(value) for value in precision_by_rank)


def main() -> None:
    args = parse_args()
    trace_path = args.trace.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    series_list = [
        build_series(trace_path, name, label, precision_by_rank)
        for name, label, precision_by_rank in STRATEGIES
    ]
    reference = series_list[0]

    write_cdf_csv(
        output_dir / "dynamic_naive_cache_reuse_distance_cdf.csv",
        series_list,
    )
    plot_cdf(
        output_dir,
        series_list,
        "dynamic_naive_cache_reuse_distance_cdf",
    )

    print(f"Trace records: {reference.record_count}")
    print(f"Layers: {reference.layer_count}")
    print(f"Object accesses per strategy: {reference.access_count}")
    for series in series_list:
        print()
        print(f"{series.label}: precision_by_rank={precision_to_text(series.precision_by_rank)}")
        print(
            f"  Recurring object accesses: {series.recurring_count} "
            f"({series.recurring_count / series.access_count:.1%})"
        )
        print(
            f"  First object accesses without reuse distance: {series.cold_accesses} "
            f"({series.cold_accesses / series.access_count:.1%})"
        )
        for percentile, distance in series.thresholds:
            print(
                f"  {percentile:.0%} of recurring object accesses reuse an object "
                f"from the previous {distance} tokens."
            )
        print("  Guide points plotted on the CDF:")
        probability_by_distance = dict(zip(series.distances, series.cdf))
        for percentile, distance in series.thresholds:
            if percentile in GUIDE_PERCENTILES:
                print(
                    f"    {percentile:.0%}: distance={distance} tokens, "
                    f"CDF={probability_by_distance[distance]:.2%}"
                )


if __name__ == "__main__":
    main()
