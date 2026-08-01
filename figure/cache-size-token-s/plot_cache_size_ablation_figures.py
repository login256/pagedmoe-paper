#!/usr/bin/env python3
"""Plot the three Qwen3.5 cache-size figures from the audited July 14 runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_ROOT = SCRIPT_DIR.parents[1]
DATA_DIR = SCRIPT_DIR / "data"
EXPECTED_GRID = tuple(float(value) for value in range(4, 53, 4))
MAX_PLOT_MEMORY_GIB = 48.0
PLOT_XTICKS = list(range(0, 49, 8))
TOTAL_MEMORY_XTICKS = list(range(0, 57, 8))
GIB = 1024**3

OVERHEAD_SOURCE = DATA_DIR / "shared_non_bitplane_memory_overhead_20260714_165151.csv"
OVERHEAD_SOURCE_SHA256 = "74d3ded11bbf994414d1bf91641925b2c08393eb2c3acfd94ef80fbb8c867ab2"
OVERHEAD_FIELD = "final_non_pool_pss_exact"
SHARED_MEMORY_OVERHEAD_GIB = 5.2

IK_LLAMA_SOURCE = DATA_DIR / "ik_llama_ud_iq4_xs_expert_20260716_190707.csv"
IK_LLAMA_SOURCE_SHA256 = "1a9ae021643bd0fe2c57941a759a1457b23b736148b05dd8c629ffed46ed57ca"
IK_LLAMA_RUN_ID = "20260716_190707"
IK_LLAMA_EXPECTED_GRID = tuple(float(value) for value in range(8, 49, 8))
IK_LLAMA_HOST_OVERHEAD_BYTES = 1_364_856_832
IK_LLAMA_MESSAGE_SHA256 = (
    "1f22083f046b74831775c19b1e27e7e903543131f79d66e8076b35bc842821b5"
)

PALETTE = {
    "dynamic2p5": "#D55E00",
    "dynamic4": "#009E73",
    "uniform4": "#0072B2",
    "black": "#000000",
    "light_gray": "#DDDDDD",
    "white": "#FFFFFF",
}


@dataclass(frozen=True)
class SourceSpec:
    path: Path
    expected_sha256: str
    run_id: str
    budget_kind: str
    precision_mode: str
    precision: str
    average_bits: float
    max_load_bits: int | None = None


@dataclass(frozen=True)
class Point:
    memory_gib: float
    value: float | None
    status: str
    n: int
    std: float | None


@dataclass(frozen=True)
class SeriesData:
    points: tuple[Point, ...]
    source: str
    source_sha256: str
    run_id: str
    budget_kind: str


@dataclass(frozen=True)
class MemoryOverhead:
    value_gib: float
    source: str
    source_sha256: str
    basis: str


SHARED_SOURCES = {
    "uniform4": SourceSpec(
        path=DATA_DIR / "shared_static4_20260714_104754.csv",
        expected_sha256="d897953bf07e61758eea29bab3a252fbd80d62cf53ed9acb0ab8ea093dd988f6",
        run_id="20260714_104754",
        budget_kind="user_space_shared_bitplane_pool",
        precision_mode="schedule",
        precision="4,4,4,4,4,4,4,4,4,4",
        average_bits=4.0,
    ),
    "dynamic4": SourceSpec(
        path=DATA_DIR / "shared_c4l40_20260714_111629.csv",
        expected_sha256="35e922400e11b65ffea97ac4e8506afab748b965aec42ffc4cd8d2c1ce2f454b",
        run_id="20260714_111629",
        budget_kind="user_space_shared_bitplane_pool",
        precision_mode="marginal_gain",
        precision="c4l40",
        average_bits=4.0,
        max_load_bits=40,
    ),
    "dynamic2p5": SourceSpec(
        path=DATA_DIR / "shared_c2p5l25_20260714_114526.csv",
        expected_sha256="6f34fbe2d3830e7c1d6720fedd7cf63e90ce26cf1faa653b21227f4ade02e646",
        run_id="20260714_114526",
        budget_kind="user_space_shared_bitplane_pool",
        precision_mode="marginal_gain",
        precision="c2p5l25",
        average_bits=2.5,
        max_load_bits=25,
    ),
    "naive2p5": SourceSpec(
        path=DATA_DIR / "naive_c2p5l25_20260714_120810.csv",
        expected_sha256="210b1d932d1dfd47935062d2f755f9c611460d95b558878184dda76ca3b9c983",
        run_id="20260714_120810",
        budget_kind="user_space_independent_precision_pool",
        precision_mode="marginal_gain",
        precision="c2p5l25",
        average_bits=2.5,
        max_load_bits=25,
    ),
}

MMAP_SOURCE = SourceSpec(
    path=DATA_DIR / "mmap_static4_c2p5l25_20260714_125412.csv",
    expected_sha256="7a13c60f17a2146dde6c00135c8407482f9b8d0833b2ab0b4fa6491bb11664af",
    run_id="20260714_125412",
    budget_kind="total_cgroup_memory_max",
    precision_mode="mixed",
    precision="mixed",
    average_bits=0.0,
)

CURVE_STYLE = {
    "ik_llama": {
        "label": "ik_llama IQ4",
        "color": PALETTE["black"],
        "linestyle": "-.",
        "marker": "^",
    },
    "uniform4": {
        "label": "Uniform4",
        "color": PALETTE["uniform4"],
        "linestyle": "-",
        "marker": "s",
    },
    "dynamic4": {
        "label": "Dyn Avg4",
        "color": PALETTE["dynamic4"],
        "linestyle": "-",
        "marker": "D",
    },
    "dynamic2p5": {
        "label": "Dyn Avg2.5",
        "color": PALETTE["dynamic2p5"],
        "linestyle": "-",
        "marker": "o",
    },
    "naive2p5": {
        "label": "Dyn Avg2.5 (Independent)",
        "color": PALETTE["dynamic2p5"],
        "linestyle": "--",
        "marker": "o",
    },
    "mmap_uniform4": {
        "label": "Uniform4",
        "color": PALETTE["uniform4"],
        "linestyle": ":",
        "marker": "s",
    },
    "mmap_dynamic2p5": {
        "label": "Dyn Avg2.5",
        "color": PALETTE["dynamic2p5"],
        "linestyle": ":",
        "marker": "o",
    },
}

FIGURE_SERIES = {
    "end-to-end": ("ik_llama", "uniform4", "dynamic4", "dynamic2p5"),
    "independent-precision": ("uniform4", "naive2p5", "dynamic2p5"),
    "mmap-backend": (
        "uniform4",
        "dynamic2p5",
        "mmap_uniform4",
        "mmap_dynamic2p5",
    ),
}

OUTPUT_NAMES = {
    "end-to-end": (
        "fig_cache_size_end_to_end_throughput",
        "fig_cache_size_token_s",
    ),
    "independent-precision": (
        "fig_cache_size_independent_precision_comparison",
        # Compatibility alias for the filename introduced in paper commit 588afbf.
        "fig_cache_size_prefix_reuse_ablation",
    ),
    "mmap-backend": (
        "fig_cache_size_mmap_backend_comparison",
        # Compatibility alias for the filename introduced in paper commit 588afbf.
        "fig_cache_size_mmap_fallback_ablation",
    ),
}


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
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": PALETTE["white"],
            "axes.facecolor": PALETTE["white"],
            "savefig.dpi": 300,
            "savefig.bbox": None,
            "savefig.pad_inches": 0.04,
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_source(path: Path) -> str:
    return str(path.resolve().relative_to(PAPER_ROOT.resolve()))


def require_columns(path: Path, fieldnames: list[str] | None, required: set[str]) -> None:
    missing = required - set(fieldnames or ())
    if missing:
        raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")


def aggregate_ok_points(values: dict[float, list[float]]) -> tuple[Point, ...]:
    points = []
    for memory_gib, samples in sorted(values.items()):
        std = statistics.stdev(samples) if len(samples) > 1 else None
        points.append(
            Point(
                memory_gib=memory_gib,
                value=statistics.fmean(samples),
                status="ok",
                n=len(samples),
                std=std,
            )
        )
    return tuple(points)


def validate_grid(path: Path, points: tuple[Point, ...]) -> None:
    observed = tuple(point.memory_gib for point in points)
    if observed != EXPECTED_GRID:
        raise ValueError(f"{path}: expected grid {EXPECTED_GRID}, got {observed}")


def read_shared_series(spec: SourceSpec) -> SeriesData:
    required = {
        "status",
        "returncode",
        "run_id",
        "precision_mode",
        "precision",
        "average_precision",
        "compute_avg_bits",
        "max_load_bits",
        "cache_size_gib",
        "decode_steps",
        "decode_token_per_s",
        "decode_log_tokens_used",
        "submitted_reads",
        "completed_reads",
        "submitted_compute_tasks",
        "completed_compute_tasks",
    }
    grouped: dict[float, list[float]] = defaultdict(list)
    repeat_metadata: dict[
        float, list[tuple[int | None, int | None]]
    ] = defaultdict(list)
    actual_sha256 = sha256(spec.path)
    if actual_sha256 != spec.expected_sha256:
        raise ValueError(
            f"{spec.path}: SHA256 changed; expected {spec.expected_sha256}, "
            f"got {actual_sha256}"
        )
    with spec.path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        require_columns(spec.path, reader.fieldnames, required)
        rows = list(reader)

    for row_number, row in enumerate(rows, start=2):
        prefix = f"{spec.path}:{row_number}"
        if row["status"] != "ok" or row["returncode"] != "0":
            raise ValueError(f"{prefix}: unsuccessful shared/naive case")
        if row["run_id"] != spec.run_id:
            raise ValueError(f"{prefix}: expected run {spec.run_id}, got {row['run_id']}")
        if row["precision_mode"] != spec.precision_mode:
            raise ValueError(f"{prefix}: unexpected precision mode")
        if row["precision"] != spec.precision:
            raise ValueError(f"{prefix}: unexpected precision label/schedule")
        if not math.isclose(float(row["average_precision"]), spec.average_bits):
            raise ValueError(f"{prefix}: unexpected average precision")
        if spec.max_load_bits is None:
            if row["compute_avg_bits"] or row["max_load_bits"]:
                raise ValueError(f"{prefix}: static schedule has a dynamic budget")
        else:
            if not math.isclose(float(row["compute_avg_bits"]), spec.average_bits):
                raise ValueError(f"{prefix}: unexpected compute-average budget")
            if int(row["max_load_bits"]) != spec.max_load_bits:
                raise ValueError(f"{prefix}: unexpected max-load budget")
        if row["decode_steps"] != "500" or row["decode_log_tokens_used"] != "480":
            raise ValueError(f"{prefix}: unexpected decode workload")
        if row["submitted_reads"] != row["completed_reads"]:
            raise ValueError(f"{prefix}: incomplete reads")
        if row["submitted_compute_tasks"] != row["completed_compute_tasks"]:
            raise ValueError(f"{prefix}: incomplete compute tasks")

        memory_gib = float(row["cache_size_gib"])
        value = float(row["decode_token_per_s"])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{prefix}: invalid decode throughput {value}")
        grouped[memory_gib].append(value)
        repeat_index_text = row.get("repeat_index", "").strip()
        repeat_count_text = row.get("repeat_count", "").strip()
        repeat_metadata[memory_gib].append(
            (
                int(repeat_index_text) if repeat_index_text else None,
                int(repeat_count_text) if repeat_count_text else None,
            )
        )

    points = aggregate_ok_points(grouped)
    validate_grid(spec.path, points)
    repeat_counts = {point.n for point in points}
    if len(repeat_counts) != 1:
        raise ValueError(
            f"{spec.path}: inconsistent repeat counts: {sorted(repeat_counts)}"
        )
    for memory_gib, metadata in repeat_metadata.items():
        repeat_indices = [item[0] for item in metadata]
        declared_counts = [item[1] for item in metadata]
        if len(metadata) > 1 and all(index is None for index in repeat_indices):
            raise ValueError(
                f"{spec.path}: cache={memory_gib:g} missing repeat indices"
            )
        if any(index is not None for index in repeat_indices):
            if any(index is None for index in repeat_indices) or {
                int(index) for index in repeat_indices if index is not None
            } != set(range(1, len(metadata) + 1)):
                raise ValueError(
                    f"{spec.path}: cache={memory_gib:g} invalid repeat indices"
                )
        if any(count is not None for count in declared_counts):
            if any(count is None for count in declared_counts) or {
                int(count) for count in declared_counts if count is not None
            } != {len(metadata)}:
                raise ValueError(
                    f"{spec.path}: cache={memory_gib:g} invalid repeat count"
                )
    return SeriesData(
        points=points,
        source=relative_source(spec.path),
        source_sha256=actual_sha256,
        run_id=spec.run_id,
        budget_kind=spec.budget_kind,
    )


def read_mmap_series(precision_label: str, expected_bits: float) -> SeriesData:
    path = MMAP_SOURCE.path
    required = {
        "run_id",
        "precision_label",
        "precision_mode",
        "precision_value",
        "max_load_bits",
        "memory_max_gib",
        "bench_cache_size_gib",
        "decode_steps",
        "exit_code",
        "completion_tokens",
        "decode_log_tokens_used",
        "decode_token_per_s_from_log",
        "avg_target_bits",
        "submitted_compute_tasks",
        "completed_compute_tasks",
    }
    actual_sha256 = sha256(path)
    if actual_sha256 != MMAP_SOURCE.expected_sha256:
        raise ValueError(
            f"{path}: SHA256 changed; expected {MMAP_SOURCE.expected_sha256}, "
            f"got {actual_sha256}"
        )
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        require_columns(path, reader.fieldnames, required)
        all_rows = list(reader)

    observed_labels = {row["precision_label"] for row in all_rows}
    if observed_labels != {"static4", "c2p5l25"}:
        raise ValueError(f"{path}: unexpected precision labels: {sorted(observed_labels)}")
    rows = [
        (row_number, row)
        for row_number, row in enumerate(all_rows, start=2)
        if row["precision_label"] == precision_label
    ]

    grouped: dict[float, list[float]] = defaultdict(list)
    failures: dict[float, list[str]] = defaultdict(list)
    for row_number, row in rows:
        prefix = f"{path}:{row_number} ({precision_label})"
        if row["run_id"] != MMAP_SOURCE.run_id:
            raise ValueError(f"{prefix}: unexpected run id")
        memory_gib = float(row["memory_max_gib"])
        if float(row["bench_cache_size_gib"]) != memory_gib:
            raise ValueError(f"{prefix}: benchmark label differs from MemoryMax")
        if row["decode_steps"] != "500":
            raise ValueError(f"{prefix}: unexpected decode length")
        if precision_label == "static4":
            if row["precision_mode"] != "target":
                raise ValueError(f"{prefix}: static4 is not target mode")
            if row["precision_value"] != "4,4,4,4,4,4,4,4,4,4":
                raise ValueError(f"{prefix}: static4 schedule is not fixed 4-bit")
            if row["max_load_bits"]:
                raise ValueError(f"{prefix}: static4 unexpectedly has max-load bits")
        else:
            if row["precision_mode"] != "dynamic":
                raise ValueError(f"{prefix}: c2p5l25 is not dynamic mode")
            if not math.isclose(float(row["precision_value"]), 2.5):
                raise ValueError(f"{prefix}: c2p5l25 average is not 2.5")
            if row["max_load_bits"] != "25":
                raise ValueError(f"{prefix}: c2p5l25 max-load budget is not 25")

        if row["exit_code"] == "0":
            if row["completion_tokens"] != "500" or row["decode_log_tokens_used"] != "480":
                raise ValueError(f"{prefix}: incomplete successful workload")
            if not math.isclose(float(row["avg_target_bits"]), expected_bits):
                raise ValueError(f"{prefix}: unexpected average target bits")
            if row["submitted_compute_tasks"] != row["completed_compute_tasks"]:
                raise ValueError(f"{prefix}: incomplete compute tasks")
            value = float(row["decode_token_per_s_from_log"])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{prefix}: invalid decode throughput {value}")
            grouped[memory_gib].append(value)
        else:
            failures[memory_gib].append(row["exit_code"])

    observed_grid = tuple(sorted(set(grouped) | set(failures)))
    if observed_grid != EXPECTED_GRID:
        raise ValueError(f"{path}: expected {EXPECTED_GRID}, got {observed_grid}")
    if set(failures) != {4.0} or not failures[4.0] or any(
        code != "137" for code in failures[4.0]
    ):
        raise ValueError(f"{path}: unexpected {precision_label} failures: {dict(failures)}")
    if 4.0 in grouped:
        raise ValueError(
            f"{path}: {precision_label} has both successful and OOM 4 GiB rows"
        )
    repeat_count = len(failures[4.0])
    invalid_repeat_counts = {
        memory_gib: len(values)
        for memory_gib, values in grouped.items()
        if len(values) != repeat_count
    }
    if invalid_repeat_counts:
        raise ValueError(
            f"{path}: inconsistent {precision_label} repeat counts: "
            f"expected {repeat_count}, got {invalid_repeat_counts}"
        )

    points = list(aggregate_ok_points(grouped))
    points.append(Point(4.0, None, "oom", repeat_count, None))
    points.sort(key=lambda point: point.memory_gib)
    return SeriesData(
        points=tuple(points),
        source=relative_source(path),
        source_sha256=actual_sha256,
        run_id=MMAP_SOURCE.run_id,
        budget_kind=MMAP_SOURCE.budget_kind,
    )


def read_shared_memory_overhead() -> MemoryOverhead:
    actual_sha256 = sha256(OVERHEAD_SOURCE)
    if actual_sha256 != OVERHEAD_SOURCE_SHA256:
        raise ValueError(
            f"{OVERHEAD_SOURCE}: SHA256 changed; expected {OVERHEAD_SOURCE_SHA256}, "
            f"got {actual_sha256}"
        )
    with OVERHEAD_SOURCE.open(newline="") as handle:
        reader = csv.DictReader(handle)
        require_columns(
            OVERHEAD_SOURCE,
            reader.fieldnames,
            {"cache_size_gib", OVERHEAD_FIELD},
        )
        rows = list(reader)
    cache_sizes = [float(row["cache_size_gib"]) for row in rows]
    if cache_sizes != [4.0, 8.0, 12.0]:
        raise ValueError(f"{OVERHEAD_SOURCE}: unexpected cache cases {cache_sizes}")
    samples_gib = [int(row[OVERHEAD_FIELD]) / GIB for row in rows]
    if not all(5.0 <= value <= 5.5 for value in samples_gib):
        raise ValueError(f"{OVERHEAD_SOURCE}: unexpected overhead samples {samples_gib}")
    return MemoryOverhead(
        value_gib=SHARED_MEMORY_OVERHEAD_GIB,
        source=relative_source(OVERHEAD_SOURCE),
        source_sha256=actual_sha256,
        basis=(
            "fixed 5.2 GiB post-decode process-PSS overhead, rounded from "
            "the formal 4/8/12-GiB measurements"
        ),
    )


def read_ik_llama_series() -> SeriesData:
    path = IK_LLAMA_SOURCE
    required = {
        "phase",
        "case_id",
        "nominal_gib",
        "memory_max_bytes",
        "memory_max_gib",
        "status",
        "error",
        "prompt_tokens",
        "completion_tokens",
        "predicted_ms",
        "predicted_per_second",
        "reasoning_present",
        "message_sha256",
        "routed_expert_resident",
        "non_routed_overhead_current",
        "oom",
        "oom_kill_events",
        "cache_clear_after_stop_ok",
    }
    actual_sha256 = sha256(path)
    if actual_sha256 != IK_LLAMA_SOURCE_SHA256:
        raise ValueError(
            f"{path}: SHA256 changed; expected {IK_LLAMA_SOURCE_SHA256}, "
            f"got {actual_sha256}"
        )
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        require_columns(path, reader.fieldnames, required)
        rows = list(reader)

    grouped: dict[float, list[float]] = defaultdict(list)
    repeat_ids: dict[float, list[int | None]] = defaultdict(list)
    for row_number, row in enumerate(rows, start=2):
        prefix = f"{path}:{row_number}"
        nominal_gib = float(row["nominal_gib"])
        if row["phase"] != "expert":
            raise ValueError(f"{prefix}: expected expert-budget phase")
        expected_case_prefix = f"expert_{int(nominal_gib):02d}g"
        repeat_match = re.fullmatch(
            rf"{re.escape(expected_case_prefix)}_rep(\d+)", row["case_id"]
        )
        if row["case_id"] != expected_case_prefix and repeat_match is None:
            raise ValueError(f"{prefix}: case id does not match nominal budget")
        if row["status"] != "success" or row["error"]:
            raise ValueError(f"{prefix}: unsuccessful ik_llama.cpp case")

        expected_memory_max = int(nominal_gib * GIB) + IK_LLAMA_HOST_OVERHEAD_BYTES
        memory_max_bytes = int(row["memory_max_bytes"])
        if memory_max_bytes != expected_memory_max:
            raise ValueError(f"{prefix}: unexpected total MemoryMax")
        if not math.isclose(float(row["memory_max_gib"]), memory_max_bytes / GIB):
            raise ValueError(f"{prefix}: inconsistent MemoryMax GiB")
        if row["prompt_tokens"] != "118" or row["completion_tokens"] != "500":
            raise ValueError(f"{prefix}: unexpected decode workload")
        if row["reasoning_present"] != "True":
            raise ValueError(f"{prefix}: thinking output is absent")
        if row["message_sha256"] != IK_LLAMA_MESSAGE_SHA256:
            raise ValueError(f"{prefix}: output differs from the audited workload")
        if row["oom"] != "False" or row["oom_kill_events"] != "0":
            raise ValueError(f"{prefix}: OOM in ik_llama.cpp case")
        if row["cache_clear_after_stop_ok"] != "True":
            raise ValueError(f"{prefix}: post-case cache clear was not verified")

        predicted_ms = float(row["predicted_ms"])
        value = float(row["predicted_per_second"])
        calculated_value = 500 / (predicted_ms / 1000)
        if not math.isclose(value, calculated_value, rel_tol=1e-6):
            raise ValueError(f"{prefix}: inconsistent decode timing")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{prefix}: invalid decode throughput {value}")
        grouped[nominal_gib].append(value)
        repeat_ids[nominal_gib].append(
            int(repeat_match.group(1)) if repeat_match is not None else None
        )

    points = aggregate_ok_points(grouped)
    observed_grid = tuple(point.memory_gib for point in points)
    if observed_grid != IK_LLAMA_EXPECTED_GRID:
        raise ValueError(
            f"{path}: expected grid {IK_LLAMA_EXPECTED_GRID}, got {observed_grid}"
        )
    repeat_counts = {point.n for point in points}
    if len(repeat_counts) != 1:
        raise ValueError(f"{path}: inconsistent repeat counts: {sorted(repeat_counts)}")
    for nominal_gib, identifiers in repeat_ids.items():
        if len(identifiers) > 1 and all(
            identifier is None for identifier in identifiers
        ):
            raise ValueError(
                f"{path}: cache={nominal_gib:g} missing repeat identifiers"
            )
        if any(identifier is not None for identifier in identifiers):
            if any(identifier is None for identifier in identifiers) or {
                int(identifier)
                for identifier in identifiers
                if identifier is not None
            } != set(range(1, len(identifiers) + 1)):
                raise ValueError(
                    f"{path}: cache={nominal_gib:g} invalid repeat identifiers"
                )
    return SeriesData(
        points=points,
        source=relative_source(path),
        source_sha256=actual_sha256,
        run_id=IK_LLAMA_RUN_ID,
        budget_kind="nominal_routed_expert_mmap_budget",
    )


def load_all_series() -> dict[str, SeriesData]:
    series = {key: read_shared_series(spec) for key, spec in SHARED_SOURCES.items()}
    series["mmap_uniform4"] = read_mmap_series("static4", 4.0)
    series["mmap_dynamic2p5"] = read_mmap_series("c2p5l25", 2.5)
    series["ik_llama"] = read_ik_llama_series()
    return series


def validate_expected_repeats(
    all_series: dict[str, SeriesData], expected_repeats: int | None
) -> None:
    if expected_repeats is None:
        return
    for key, data in all_series.items():
        mismatches = [
            (point.memory_gib, point.n)
            for point in data.points
            if point.n != expected_repeats
        ]
        if mismatches:
            raise ValueError(
                f"{key}: expected {expected_repeats} repeats at every point, "
                f"got {mismatches}"
            )


def draw_series(
    ax: plt.Axes,
    key: str,
    data: SeriesData,
    label: str | None = None,
    x_offset_gib: float = 0.0,
) -> None:
    style = CURVE_STYLE[key]
    points = [
        point
        for point in data.points
        if point.status != "oom" and point.memory_gib <= MAX_PLOT_MEMORY_GIB
    ]
    x_values = [point.memory_gib + x_offset_gib for point in points]
    y_values = [point.value for point in points]
    ax.plot(
        x_values,
        y_values,
        color=style["color"],
        linestyle=style["linestyle"],
        linewidth=1.45,
        marker=style["marker"],
        markersize=3.0,
        markerfacecolor=PALETTE["white"],
        markeredgecolor=style["color"],
        markeredgewidth=0.75,
        label=label or style["label"],
        zorder=3,
    )


def configure_axes(
    ax: plt.Axes,
    *,
    xlim: tuple[float, float],
    xticks: list[int],
    xlabel: str,
    ylabel: bool,
) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(0, 12.5)
    ax.set_xticks(xticks)
    ax.set_yticks([0, 2, 4, 6, 8, 10, 12])
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
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel("Decode throughput (tokens/s)")


def save_figure(fig: plt.Figure, output_dir: Path, output_names: tuple[str, ...]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for output_name in output_names:
        output_stem = output_dir / output_name
        fig.savefig(output_stem.with_suffix(".pdf"))
        fig.savefig(output_stem.with_suffix(".png"), dpi=300)


def plot_single_panel(
    figure_name: str,
    all_series: dict[str, SeriesData],
    output_dir: Path,
    output_names: tuple[str, ...] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(3.35, 2.10))
    for key in FIGURE_SERIES[figure_name]:
        draw_series(ax, key, all_series[key])

    if figure_name == "end-to-end":
        configure_axes(
            ax,
            xlim=(0, 50),
            xticks=PLOT_XTICKS,
            xlabel="Expert cache capacity (GiB)",
            ylabel=True,
        )
    else:
        configure_axes(
            ax,
            xlim=(0, 50),
            xticks=PLOT_XTICKS,
            xlabel="Cache capacity (GiB)",
            ylabel=True,
        )

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        ncol=2,
        frameon=False,
        handlelength=2.2,
        columnspacing=0.55,
        handletextpad=0.3,
        borderaxespad=0.0,
    )
    fig.tight_layout(pad=0.45)
    fig.subplots_adjust(bottom=fig.subplotpars.bottom - 0.02)
    save_figure(fig, output_dir, output_names or OUTPUT_NAMES[figure_name])
    plt.close(fig)


def plot_mmap_backend(
    all_series: dict[str, SeriesData],
    output_dir: Path,
    output_names: tuple[str, ...] | None = None,
) -> None:
    overhead = read_shared_memory_overhead()
    fig, ax = plt.subplots(figsize=(3.35, 2.10))

    draw_series(
        ax,
        "uniform4",
        all_series["uniform4"],
        label="Uniform4",
        x_offset_gib=overhead.value_gib,
    )
    draw_series(
        ax,
        "mmap_uniform4",
        all_series["mmap_uniform4"],
        label="Uniform4 (mmap)",
    )
    draw_series(
        ax,
        "dynamic2p5",
        all_series["dynamic2p5"],
        label="Dyn Avg2.5",
        x_offset_gib=overhead.value_gib,
    )
    draw_series(
        ax,
        "mmap_dynamic2p5",
        all_series["mmap_dynamic2p5"],
        label="Dyn Avg2.5 (mmap)",
    )
    configure_axes(
        ax,
        xlim=(0, 56),
        xticks=TOTAL_MEMORY_XTICKS,
        xlabel="Total Memory (GiB)",
        ylabel=True,
    )
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        handlelength=2.2,
        columnspacing=0.55,
        handletextpad=0.3,
        labelspacing=0.2,
        borderaxespad=0.0,
        fontsize=7.5,
    )
    fig.tight_layout(pad=0.45)
    fig.subplots_adjust(bottom=fig.subplotpars.bottom - 0.02)
    save_figure(fig, output_dir, output_names or OUTPUT_NAMES["mmap-backend"])
    plt.close(fig)


def plot_figure(
    figure_name: str,
    all_series: dict[str, SeriesData],
    output_dir: Path,
    output_names: tuple[str, ...] | None = None,
) -> None:
    if figure_name == "mmap-backend":
        plot_mmap_backend(all_series, output_dir, output_names)
    else:
        plot_single_panel(figure_name, all_series, output_dir, output_names)


def display_label(figure_name: str, key: str) -> str:
    if figure_name == "mmap-backend":
        policy_key = key.removeprefix("mmap_")
        label = CURVE_STYLE[policy_key]["label"]
        return f"{label} (mmap)" if key.startswith("mmap_") else label
    return str(CURVE_STYLE[key]["label"])


def write_plot_data(
    selected_figures: list[str],
    all_series: dict[str, SeriesData],
    output_dir: Path,
) -> None:
    overhead = read_shared_memory_overhead()
    rows = []
    for figure_name in selected_figures:
        for key in FIGURE_SERIES[figure_name]:
            data = all_series[key]
            for point in data.points:
                if point.memory_gib > MAX_PLOT_MEMORY_GIB:
                    continue
                plotted_memory_gib = point.memory_gib
                plotted_budget_kind = data.budget_kind
                memory_overhead_gib = 0.0
                overhead_source = ""
                overhead_source_sha256 = ""
                overhead_basis = ""
                if figure_name == "mmap-backend" and not key.startswith("mmap_"):
                    plotted_memory_gib += overhead.value_gib
                    plotted_budget_kind = "estimated_total_memory_capacity_plus_pss_overhead"
                    memory_overhead_gib = overhead.value_gib
                    overhead_source = overhead.source
                    overhead_source_sha256 = overhead.source_sha256
                    overhead_basis = overhead.basis
                rows.append(
                    {
                        "figure": figure_name,
                        "curve_key": key,
                        "series": display_label(figure_name, key),
                        "memory_gib": f"{plotted_memory_gib:.6f}",
                        "source_memory_gib": f"{point.memory_gib:g}",
                        "memory_overhead_gib": f"{memory_overhead_gib:.6f}",
                        "decode_token_per_s": (
                            "" if point.value is None else f"{point.value:.4f}"
                        ),
                        "status": point.status,
                        "budget_kind": plotted_budget_kind,
                        "source_budget_kind": data.budget_kind,
                        "n": str(point.n),
                        "std": "" if point.std is None else f"{point.std:.6f}",
                        "run_id": data.run_id,
                        "source": data.source,
                        "source_sha256": data.source_sha256,
                        "overhead_source": overhead_source,
                        "overhead_source_sha256": overhead_source_sha256,
                        "overhead_basis": overhead_basis,
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "cache_size_ablation_plot_data.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            lineterminator="\n",
            fieldnames=[
                "figure",
                "curve_key",
                "series",
                "memory_gib",
                "source_memory_gib",
                "memory_overhead_gib",
                "decode_token_per_s",
                "status",
                "budget_kind",
                "source_budget_kind",
                "n",
                "std",
                "run_id",
                "source",
                "source_sha256",
                "overhead_source",
                "overhead_source_sha256",
                "overhead_basis",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure",
        choices=["all", *FIGURE_SERIES],
        default="all",
        help="figure to generate",
    )
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument(
        "--expected-repeats",
        type=int,
        help="Require this many independent runs for every plotted point.",
    )
    args = parser.parse_args(argv)
    if args.expected_repeats is not None and args.expected_repeats < 1:
        parser.error("--expected-repeats must be positive")

    configure_matplotlib()
    all_series = load_all_series()
    validate_expected_repeats(all_series, args.expected_repeats)
    selected = list(FIGURE_SERIES) if args.figure == "all" else [args.figure]
    for figure_name in selected:
        plot_figure(figure_name, all_series, args.output_dir)
    # Keep the provenance snapshot complete even when only one image is requested.
    write_plot_data(list(FIGURE_SERIES), all_series, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
