#!/usr/bin/env python3
"""Simulate exact precision-object cache hit rate from a precision trace.

The simulated cache object is exactly (layer, expert_id, precision). A p-bit
expert object is loaded as a whole object, so (layer, expert_id, 4) and
(layer, expert_id, 5) are different objects and do not share capacity.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_TRACE = (
    REPO_ROOT
    / "src/target/bench/precision_trace_c4l40_mmlu-abstract_algebra-idx53_thinking_cache48_compute16_decode500_seed1926817_cudaord2_20260607_152724"
    / "precision_trace.log"
)
DEFAULT_SUMMARY = (
    REPO_ROOT
    / "src/target/bench/kt_pagedmoe_dynamic_static_cache8-64_perf_hitrate_summary.csv"
)
DEFAULT_MULTI_TRACE_ROOT = (
    REPO_ROOT
    / "src/target/bench/precision_trace_c3l30_c2l20_mmlu-abstract_algebra-idx53_thinking_cache48_compute16_decode500_seed1926817_cudaord2_20260608_052337"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR

TRACE_PATTERN = re.compile(
    r"\[pagedmoe-precision-trace\].*?"
    r"\blayer=Some\((?P<layer>\d+)\).*?"
    r"\bexperts=\[(?P<experts>[^\]]*)\].*?"
    r"\btarget=\[(?P<target>[^\]]*)\]"
)

MIN_CACHE_GIB = 8
MAX_CACHE_GIB = 64
CACHE_STEP_GIB = 4
GIB = 1024**3

TENSORS_PER_EXPERT = 3
NUM_BLOCKS = 32
BITPLANE_BYTES = 16 * 1024
CODEBOOK_BYTES = 1024
MAX_PRECISION_BITS = 8

BITPLANE_BYTES_PER_EXPERT_BIT = TENSORS_PER_EXPERT * NUM_BLOCKS * BITPLANE_BYTES
CODEBOOK_BYTES_PER_EXPERT = TENSORS_PER_EXPERT * NUM_BLOCKS * CODEBOOK_BYTES

PALETTE = {
    "trace": "#D55E00",
    "c4l40": "#0072B2",
    "all4": "#999999",
    "light_gray": "#DDDDDD",
    "white": "#FFFFFF",
}
STYLE_BY_SERIES = {
    "naive": {
        "color": PALETTE["trace"],
        "linestyle": "-",
        "marker": "o",
    },
    "dynamic": {
        "color": PALETTE["c4l40"],
        "linestyle": "--",
        "marker": "s",
    },
    "static": {
        "color": PALETTE["all4"],
        "linestyle": ":",
        "marker": "^",
    },
}


@dataclass(frozen=True)
class Access:
    layer: int
    expert_id: int
    precision: int

    @property
    def key(self) -> tuple[int, int, int]:
        return (self.layer, self.expert_id, self.precision)


@dataclass(frozen=True)
class TraceStats:
    record_count: int
    access_count: int
    layer_count: int
    unique_object_count: int


@dataclass(frozen=True)
class SimResult:
    cache_gib: int
    request_hit_rate: float
    byte_hit_rate: float
    hits: int
    requests: int
    byte_hits: int
    request_bytes: int
    objects_final: int
    used_bytes_final: int


@dataclass(frozen=True)
class FigureSpec:
    name: str
    trace_path: Path
    dynamic_case: str
    static_case: str
    naive_label: str
    dynamic_label: str
    static_label: str
    output_stem: str
    simulated_csv: str
    plot_series_csv: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace",
        type=Path,
        default=DEFAULT_TRACE,
        help=f"precision trace to parse (default: {DEFAULT_TRACE})",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=DEFAULT_SUMMARY,
        help=f"benchmark summary CSV for c4l40/all4 (default: {DEFAULT_SUMMARY})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"directory for generated files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--output-stem",
        default="fig_precision_object_cache_hit_rate",
        help="output plot stem without extension for single-spec mode",
    )
    parser.add_argument(
        "--only",
        choices=("avg4", "avg3", "avg2", "all"),
        default="all",
        help="which figure set to generate (default: all)",
    )
    parser.add_argument("--min-cache-gib", type=int, default=MIN_CACHE_GIB)
    parser.add_argument("--max-cache-gib", type=int, default=MAX_CACHE_GIB)
    parser.add_argument("--cache-step-gib", type=int, default=CACHE_STEP_GIB)
    parser.add_argument(
        "--expected-repeats",
        type=int,
        help="Require this many runtime measurements per case/cache point.",
    )
    return parser.parse_args()


def parse_int_list(text: str) -> list[int]:
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def object_size_bytes(precision: int) -> int:
    if precision < 1 or precision > MAX_PRECISION_BITS:
        raise ValueError(
            f"precision must be in [1, {MAX_PRECISION_BITS}], got {precision}"
        )
    return CODEBOOK_BYTES_PER_EXPERT + precision * BITPLANE_BYTES_PER_EXPERT_BIT


def parse_trace(trace_path: Path) -> tuple[list[Access], TraceStats]:
    accesses: list[Access] = []
    layers: set[int] = set()
    record_count = 0

    with trace_path.open(encoding="utf-8", errors="replace") as trace_file:
        for line in trace_file:
            match = TRACE_PATTERN.search(line)
            if match is None:
                continue

            layer = int(match.group("layer"))
            experts = parse_int_list(match.group("experts"))
            targets = parse_int_list(match.group("target"))
            if len(experts) != len(targets):
                raise ValueError(
                    f"record {record_count + 1} has {len(experts)} experts "
                    f"but {len(targets)} target precisions"
                )

            record_count += 1
            layers.add(layer)
            for expert_id, precision in zip(experts, targets):
                object_size_bytes(precision)
                accesses.append(Access(layer, expert_id, precision))

    if record_count == 0:
        raise ValueError(f"no precision-trace records found in {trace_path}")
    if not accesses:
        raise ValueError(f"no expert accesses found in {trace_path}")

    stats = TraceStats(
        record_count=record_count,
        access_count=len(accesses),
        layer_count=len(layers),
        unique_object_count=len({access.key for access in accesses}),
    )
    return accesses, stats


def simulate_lru(accesses: list[Access], cache_gib: int) -> SimResult:
    capacity_bytes = cache_gib * GIB
    cache: OrderedDict[tuple[int, int, int], int] = OrderedDict()
    used_bytes = 0
    hits = 0
    byte_hits = 0
    request_bytes = 0

    for access in accesses:
        key = access.key
        size_bytes = object_size_bytes(access.precision)
        request_bytes += size_bytes

        if key in cache:
            hits += 1
            byte_hits += size_bytes
            cache.move_to_end(key)
            continue

        if size_bytes > capacity_bytes:
            continue

        while used_bytes + size_bytes > capacity_bytes and cache:
            _, evicted_size = cache.popitem(last=False)
            used_bytes -= evicted_size

        cache[key] = size_bytes
        used_bytes += size_bytes

    requests = len(accesses)
    return SimResult(
        cache_gib=cache_gib,
        request_hit_rate=hits / requests,
        byte_hit_rate=byte_hits / request_bytes,
        hits=hits,
        requests=requests,
        byte_hits=byte_hits,
        request_bytes=request_bytes,
        objects_final=len(cache),
        used_bytes_final=used_bytes,
    )


def simulate_cache_sweep(
    accesses: list[Access],
    min_cache_gib: int,
    max_cache_gib: int,
    cache_step_gib: int,
) -> list[SimResult]:
    if min_cache_gib <= 0 or max_cache_gib < min_cache_gib or cache_step_gib <= 0:
        raise ValueError("invalid cache sweep bounds")
    cache_sizes = range(min_cache_gib, max_cache_gib + 1, cache_step_gib)
    return [simulate_lru(accesses, cache_gib) for cache_gib in cache_sizes]


def read_summary_series(
    summary_csv: Path,
    cases: tuple[str, ...],
    expected_cache_gibs: tuple[int, ...] | None = None,
    expected_repeats: int | None = None,
) -> dict[str, list[tuple[int, float]]]:
    samples: dict[
        str, defaultdict[int, list[tuple[float, int | None, int | None]]]
    ] = {
        case: defaultdict(list) for case in cases
    }
    with summary_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            case = row["case"]
            if case not in samples or row.get("status") != "ok":
                continue
            cache_gib = int(float(row["cache_size_gib"]))
            hit_rate = float(row["read_job_cache_hit_rate"])
            repeat_index_text = row.get("repeat_index", "").strip()
            repeat_count_text = row.get("repeat_count", "").strip()
            samples[case][cache_gib].append(
                (
                    hit_rate,
                    int(repeat_index_text) if repeat_index_text else None,
                    int(repeat_count_text) if repeat_count_text else None,
                )
            )

    series: dict[str, list[tuple[int, float]]] = {}
    observed_sample_counts: set[int] = set()
    for case, values_by_cache in samples.items():
        if not values_by_cache:
            raise ValueError(f"missing case={case} in {summary_csv}")
        if expected_cache_gibs is not None and set(values_by_cache) != set(
            expected_cache_gibs
        ):
            raise ValueError(
                f"case={case} cache grid mismatch: expected={expected_cache_gibs}, "
                f"actual={tuple(sorted(values_by_cache))}"
            )
        points = []
        for cache_gib, values in sorted(values_by_cache.items()):
            observed_sample_counts.add(len(values))
            repeat_indices = [value[1] for value in values]
            repeat_counts = [value[2] for value in values]
            if len(values) > 1 and all(index is None for index in repeat_indices):
                raise ValueError(
                    f"case={case} cache={cache_gib}: missing repeat indices"
                )
            if any(index is not None for index in repeat_indices):
                if any(index is None for index in repeat_indices) or {
                    int(index) for index in repeat_indices if index is not None
                } != set(range(1, len(values) + 1)):
                    raise ValueError(
                        f"case={case} cache={cache_gib}: invalid repeat indices"
                    )
            if any(count is not None for count in repeat_counts):
                if any(count is None for count in repeat_counts) or {
                    int(count) for count in repeat_counts if count is not None
                } != {len(values)}:
                    raise ValueError(
                        f"case={case} cache={cache_gib}: invalid repeat count"
                    )
            points.append(
                (cache_gib, statistics.fmean(value[0] for value in values))
            )
        series[case] = points
    if len(observed_sample_counts) != 1:
        raise ValueError(
            f"inconsistent repeat counts in {summary_csv}: "
            f"{sorted(observed_sample_counts)}"
        )
    if expected_repeats is not None and observed_sample_counts != {expected_repeats}:
        raise ValueError(
            f"expected {expected_repeats} repeats in {summary_csv}, "
            f"got {sorted(observed_sample_counts)}"
        )
    return series


def write_simulation_csv(output_path: Path, spec: FigureSpec, results: list[SimResult]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file, lineterminator="\n")
        writer.writerow(
            [
                "series",
                "trace",
                "cache_gib",
                "request_hit_rate",
                "byte_hit_rate",
                "hits",
                "requests",
                "byte_hits",
                "request_bytes",
                "objects_final",
                "used_gib_final",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    spec.naive_label,
                    spec.trace_path,
                    result.cache_gib,
                    f"{result.request_hit_rate:.8f}",
                    f"{result.byte_hit_rate:.8f}",
                    result.hits,
                    result.requests,
                    result.byte_hits,
                    result.request_bytes,
                    result.objects_final,
                    f"{result.used_bytes_final / GIB:.6f}",
                ]
            )


def write_plot_series_csv(
    output_path: Path,
    spec: FigureSpec,
    results: list[SimResult],
    summary_series: dict[str, list[tuple[int, float]]],
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file, lineterminator="\n")
        writer.writerow(["series", "cache_gib", "hit_rate"])
        for result in results:
            writer.writerow([spec.naive_label, result.cache_gib, f"{result.byte_hit_rate:.8f}"])
        for case, label in (
            (spec.dynamic_case, spec.dynamic_label),
            (spec.static_case, spec.static_label),
        ):
            for cache_gib, hit_rate in summary_series[case]:
                writer.writerow([label, cache_gib, f"{hit_rate:.8f}"])


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


def plot(
    output_stem: Path,
    spec: FigureSpec,
    results: list[SimResult],
    summary_series: dict[str, list[tuple[int, float]]],
) -> None:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(3.35, 2.35))

    trace_values = [
        (result.cache_gib, result.byte_hit_rate * 100.0) for result in results
    ]
    all_series = {
        "naive": (spec.naive_label, trace_values),
        "dynamic": (
            spec.dynamic_label,
            [
                (cache_gib, hit_rate * 100.0)
                for cache_gib, hit_rate in summary_series[spec.dynamic_case]
            ],
        ),
        "static": (
            spec.static_label,
            [
                (cache_gib, hit_rate * 100.0)
                for cache_gib, hit_rate in summary_series[spec.static_case]
            ],
        ),
    }

    for name in ("naive", "dynamic", "static"):
        label, values = all_series[name]
        style = STYLE_BY_SERIES[name]
        ax.plot(
            [cache_gib for cache_gib, _ in values],
            [hit_rate for _, hit_rate in values],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.45,
            marker=style["marker"],
            markersize=3.0,
            markerfacecolor=PALETTE["white"],
            markeredgecolor=style["color"],
            markeredgewidth=0.75,
            label=label,
            zorder=3,
        )

    y_values = [
        hit_rate
        for _, values in all_series.values()
        for _, hit_rate in values
    ]
    y_min = max(0, math.floor((min(y_values) - 4.0) / 10.0) * 10)
    y_max = min(100, math.ceil((max(y_values) + 4.0) / 10.0) * 10)
    if y_max - y_min < 20:
        y_max = min(100, y_min + 20)

    ax.set_xlim(6, 66)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([8, 16, 24, 32, 40, 48, 56, 64])
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
    ax.set_ylabel("Cache hit rate")
    ax.legend(
        loc="lower right",
        frameon=False,
        handlelength=1.8,
        handletextpad=0.45,
        borderaxespad=0.2,
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.45)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_specs(args: argparse.Namespace) -> dict[str, FigureSpec]:
    return {
        "avg4": FigureSpec(
            name="avg4",
            trace_path=args.trace.resolve(),
            dynamic_case="c4l40",
            static_case="all4",
            naive_label="Avg4 Naive Cache",
            dynamic_label="Avg4",
            static_label="All4",
            output_stem=args.output_stem,
            simulated_csv="precision_object_cache_hit_rate_simulated.csv",
            plot_series_csv="precision_object_cache_hit_rate_plot_series.csv",
        ),
        "avg3": FigureSpec(
            name="avg3",
            trace_path=(DEFAULT_MULTI_TRACE_ROOT / "c3l30/precision_trace.log").resolve(),
            dynamic_case="c3l30",
            static_case="all3",
            naive_label="Avg3 Naive Cache",
            dynamic_label="Avg3",
            static_label="All3",
            output_stem="fig_precision_object_cache_hit_rate_avg3",
            simulated_csv="precision_object_cache_hit_rate_avg3_simulated.csv",
            plot_series_csv="precision_object_cache_hit_rate_avg3_plot_series.csv",
        ),
        "avg2": FigureSpec(
            name="avg2",
            trace_path=(DEFAULT_MULTI_TRACE_ROOT / "c2l20/precision_trace.log").resolve(),
            dynamic_case="c2l20",
            static_case="all2",
            naive_label="Avg2 Naive Cache",
            dynamic_label="Avg2",
            static_label="All2",
            output_stem="fig_precision_object_cache_hit_rate_avg2",
            simulated_csv="precision_object_cache_hit_rate_avg2_simulated.csv",
            plot_series_csv="precision_object_cache_hit_rate_avg2_plot_series.csv",
        ),
    }


def run_spec(
    spec: FigureSpec,
    summary_csv: Path,
    output_dir: Path,
    min_cache_gib: int,
    max_cache_gib: int,
    cache_step_gib: int,
    expected_repeats: int | None,
) -> None:
    accesses, trace_stats = parse_trace(spec.trace_path)
    sim_results = simulate_cache_sweep(
        accesses,
        min_cache_gib,
        max_cache_gib,
        cache_step_gib,
    )
    summary_series = read_summary_series(
        summary_csv,
        (spec.dynamic_case, spec.static_case),
        tuple(range(min_cache_gib, max_cache_gib + 1, cache_step_gib)),
        expected_repeats,
    )

    write_simulation_csv(output_dir / spec.simulated_csv, spec, sim_results)
    write_plot_series_csv(
        output_dir / spec.plot_series_csv,
        spec,
        sim_results,
        summary_series,
    )
    plot(output_dir / spec.output_stem, spec, sim_results, summary_series)

    print(f"\n{spec.naive_label}:")
    print(f"  Trace: {spec.trace_path}")
    print(f"  Trace records: {trace_stats.record_count}")
    print(f"  Layers: {trace_stats.layer_count}")
    print(f"  Object accesses: {trace_stats.access_count}")
    print(f"  Unique exact objects: {trace_stats.unique_object_count}")
    print(
        "  Object bytes: "
        f"codebook={CODEBOOK_BYTES_PER_EXPERT} "
        f"per_bitplane={BITPLANE_BYTES_PER_EXPERT_BIT}"
    )
    for result in sim_results:
        print(
            f"  {result.cache_gib:>2} GiB: "
            f"request_hit_rate={result.request_hit_rate:.4f} "
            f"byte_hit_rate={result.byte_hit_rate:.4f} "
            f"objects_final={result.objects_final} "
            f"used_gib={result.used_bytes_final / GIB:.3f}"
        )


def main() -> int:
    args = parse_args()
    if args.expected_repeats is not None and args.expected_repeats < 1:
        raise ValueError("--expected-repeats must be positive")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.summary_csv.resolve()
    specs = build_specs(args)
    selected = specs.keys() if args.only == "all" else (args.only,)
    for name in selected:
        run_spec(
            specs[name],
            summary_csv,
            output_dir,
            args.min_cache_gib,
            args.max_cache_gib,
            args.cache_step_gib,
            args.expected_repeats,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
