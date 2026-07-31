#!/usr/bin/env python3
"""Replay real PagedMoE precision traces through an exact-object LRU cache.

The dynamic series use the target precision emitted by PagedMoE's
``marginal_gain`` allocator. The fixed series reuse the expert routes from the
corresponding dynamic run, but replace every target with a uniform precision.
The comparison is pairwise matched: Fixed4/DynAvg4 use the c4l40 routes and
Fixed3/DynAvg3 use the c3l30 routes. It does not assume that the two live runs
produce the same routes.

An exact object is keyed by ``(layer, expert_id, precision)`` and is inserted
atomically. Objects at different precisions do not share bytes or cache hits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_STEM = "fig_actual_pagedmoe_exact_object_cache_hit_rate"
DEFAULT_CSV_NAME = "actual_pagedmoe_exact_object_cache_hit_rate.csv"
DEFAULT_PRIMARY_CSV_NAME = "actual_pagedmoe_exact_object_cache_hit_rate_24gib.csv"
DEFAULT_METADATA_NAME = "actual_pagedmoe_exact_object_cache_hit_rate_metadata.json"

TRACE_TAG = "[pagedmoe-precision-trace]"
MARGINAL_MARKER_RE = re.compile(r"\[pagedmoe-precision-trace\]\s+mode=marginal_gain\b")
TRACE_RE = re.compile(
    r"\[pagedmoe-precision-trace\]\s+mode=marginal_gain\s+"
    r"layer=Some\((?P<layer>\d+)\).*?"
    r"\bexperts=\[(?P<experts>[^\]]*)\].*?"
    r"\btarget=\[(?P<target>[^\]]*)\].*?"
    r"\btarget_bits=(?P<target_bits>\d+)\s+"
    r"compute_bit_budget=(?P<compute_budget>Some\(\d+\)|None)\s+"
    r"max_load_bits=(?P<max_load_bits>Some\(\d+\)|None)"
)

LAYERS = 60
TOP_K = 10
DECODE_TOKENS = 500
MIN_PRECISION = 1
MAX_PRECISION = 8
GIB = 1024**3

TENSORS_PER_EXPERT = 3
NUM_BLOCKS = 32
CODEBOOK_BYTES_PER_BLOCK = 1024
BITPLANE_BYTES_PER_BLOCK = 16 * 1024
CODEBOOK_BYTES_PER_OBJECT = TENSORS_PER_EXPERT * NUM_BLOCKS * CODEBOOK_BYTES_PER_BLOCK
BITPLANE_BYTES_PER_OBJECT_BIT = (
    TENSORS_PER_EXPERT * NUM_BLOCKS * BITPLANE_BYTES_PER_BLOCK
)

METHOD_ORDER = ("Fixed4", "Fixed3", "DynAvg4", "DynAvg3")
PAIR_BY_METHOD = {
    "Fixed4": "c4l40:Fixed4-vs-DynAvg4",
    "DynAvg4": "c4l40:Fixed4-vs-DynAvg4",
    "Fixed3": "c3l30:Fixed3-vs-DynAvg3",
    "DynAvg3": "c3l30:Fixed3-vs-DynAvg3",
}
METHOD_STYLE = {
    "Fixed4": {
        "label": "Fixed 4",
        "color": "#777777",
        "linestyle": "-",
        "marker": "s",
    },
    "Fixed3": {
        "label": "Fixed 3",
        "color": "#999999",
        "linestyle": "--",
        "marker": "o",
    },
    "DynAvg4": {
        "label": "Dynamic avg. 4",
        "color": "#D55E00",
        "linestyle": "-",
        "marker": "s",
    },
    "DynAvg3": {
        "label": "Dynamic avg. 3",
        "color": "#0072B2",
        "linestyle": "--",
        "marker": "o",
    },
}


@dataclass(frozen=True)
class PrecisionRecord:
    layer: int
    experts: tuple[int, ...]
    targets: tuple[int, ...]
    target_bits: int
    compute_bit_budget: int | None
    max_load_bits: int | None
    source_line: int


@dataclass(frozen=True)
class ParseStats:
    trace_marker_count: int
    marginal_gain_marker_count: int
    marginal_gain_record_count: int


@dataclass(frozen=True)
class DecodeTrace:
    records: tuple[PrecisionRecord, ...]
    source_record_count: int
    prefill_record_count: int
    decode_tokens: int
    layers: int
    top_k: int


@dataclass(frozen=True)
class Access:
    layer: int
    expert_id: int
    precision: int

    @property
    def key(self) -> tuple[int, int, int]:
        return (self.layer, self.expert_id, self.precision)


@dataclass(frozen=True)
class SimResult:
    capacity_bytes: int
    hits: int
    requests: int
    byte_hits: int
    request_bytes: int
    objects_final: int
    used_bytes_final: int

    @property
    def request_hit_rate(self) -> float:
        return self.hits / self.requests

    @property
    def byte_hit_rate(self) -> float:
        return self.byte_hits / self.request_bytes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--c4-trace",
        type=Path,
        help="server log from a marginal-gain c4l40 PagedMoE run",
    )
    parser.add_argument(
        "--c3-trace",
        type=Path,
        help="server log from a marginal-gain c3l30 PagedMoE run",
    )
    parser.add_argument("--decode-tokens", type=int, default=DECODE_TOKENS)
    parser.add_argument("--layers", type=int, default=LAYERS)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--expected-c4-bits", type=int, default=40)
    parser.add_argument("--expected-c3-bits", type=int, default=30)
    parser.add_argument(
        "--cache-gib",
        type=int,
        nargs="+",
        default=list(range(8, 65, 4)),
        help="cache capacities to simulate (default: 8, 12, ..., 64)",
    )
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--output-stem", default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--csv-name", default=DEFAULT_CSV_NAME)
    parser.add_argument("--primary-cache-gib", type=int, default=24)
    parser.add_argument("--primary-csv-name", default=DEFAULT_PRIMARY_CSV_NAME)
    parser.add_argument("--metadata-name", default=DEFAULT_METADATA_NAME)
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="run lightweight parser/selection/LRU checks without trace files",
    )
    return parser.parse_args()


def parse_int_list(text: str) -> tuple[int, ...]:
    return tuple(int(value.strip()) for value in text.split(",") if value.strip())


def parse_option_usize(text: str) -> int | None:
    if text == "None":
        return None
    if not text.startswith("Some(") or not text.endswith(")"):
        raise ValueError(f"invalid Rust Option<usize>: {text}")
    return int(text[5:-1])


def parse_trace_lines(
    lines: Iterable[str], source: str, top_k: int
) -> tuple[list[PrecisionRecord], ParseStats]:
    records: list[PrecisionRecord] = []
    trace_markers = 0
    marginal_markers = 0

    for line_number, line in enumerate(lines, start=1):
        trace_markers += line.count(TRACE_TAG)
        expected_matches = len(MARGINAL_MARKER_RE.findall(line))
        marginal_markers += expected_matches
        matches = list(TRACE_RE.finditer(line))
        if len(matches) != expected_matches:
            raise ValueError(
                f"{source}:{line_number}: found {expected_matches} marginal-gain "
                f"markers but parsed {len(matches)} complete records"
            )

        for match in matches:
            experts = parse_int_list(match.group("experts"))
            targets = parse_int_list(match.group("target"))
            if len(experts) != top_k:
                raise ValueError(
                    f"{source}:{line_number}: expected top-{top_k}, "
                    f"got {len(experts)} experts"
                )
            if len(targets) != len(experts):
                raise ValueError(
                    f"{source}:{line_number}: {len(experts)} experts but "
                    f"{len(targets)} target precisions"
                )
            if len(set(experts)) != len(experts):
                raise ValueError(f"{source}:{line_number}: duplicate routed expert")
            invalid_targets = [
                precision
                for precision in targets
                if precision < MIN_PRECISION or precision > MAX_PRECISION
            ]
            if invalid_targets:
                raise ValueError(
                    f"{source}:{line_number}: target precision outside "
                    f"[{MIN_PRECISION}, {MAX_PRECISION}]: {invalid_targets}"
                )

            target_bits = int(match.group("target_bits"))
            if sum(targets) != target_bits:
                raise ValueError(
                    f"{source}:{line_number}: sum(target)={sum(targets)} but "
                    f"target_bits={target_bits}"
                )
            records.append(
                PrecisionRecord(
                    layer=int(match.group("layer")),
                    experts=experts,
                    targets=targets,
                    target_bits=target_bits,
                    compute_bit_budget=parse_option_usize(
                        match.group("compute_budget")
                    ),
                    max_load_bits=parse_option_usize(match.group("max_load_bits")),
                    source_line=line_number,
                )
            )

    if not records:
        raise ValueError(f"no marginal-gain precision records found in {source}")
    return records, ParseStats(trace_markers, marginal_markers, len(records))


def parse_trace(path: Path, top_k: int) -> tuple[list[PrecisionRecord], ParseStats]:
    with path.open(encoding="utf-8", errors="replace") as handle:
        return parse_trace_lines(handle, str(path), top_k)


def validate_trace_config(
    records: Sequence[PrecisionRecord], case: str, expected_bits: int
) -> None:
    invalid_layers = [record.layer for record in records if record.layer < 0]
    if invalid_layers:
        raise ValueError(f"{case}: negative layer ids: {invalid_layers[:5]}")

    fields = (
        ("target_bits", {record.target_bits for record in records}),
        (
            "compute_bit_budget",
            {record.compute_bit_budget for record in records},
        ),
        ("max_load_bits", {record.max_load_bits for record in records}),
    )
    for field, observed in fields:
        if observed != {expected_bits}:
            raise ValueError(
                f"{case}: expected every {field} to be {expected_bits}, "
                f"observed {sorted(observed, key=lambda value: -1 if value is None else value)}"
            )


def select_decode_trace(
    records: Sequence[PrecisionRecord],
    decode_tokens: int,
    layers: int,
    top_k: int,
) -> DecodeTrace:
    if decode_tokens < 1:
        raise ValueError("decode_tokens must be positive")
    if layers < 1 or top_k < 1:
        raise ValueError("layers and top_k must be positive")

    decode_tail_count = (decode_tokens - 1) * layers
    minimum_records = decode_tail_count + layers
    if len(records) < minimum_records:
        raise ValueError(
            f"trace has {len(records)} records; need at least {minimum_records} "
            f"for {decode_tokens} tokens and {layers} layers"
        )

    if decode_tail_count:
        prefill = records[:-decode_tail_count]
        decode_tail = records[-decode_tail_count:]
    else:
        prefill = records
        decode_tail = ()

    expected_tail_layers = list(range(layers)) * (decode_tokens - 1)
    observed_tail_layers = [record.layer for record in decode_tail]
    if observed_tail_layers != expected_tail_layers:
        mismatch = next(
            (
                index
                for index, (observed, expected) in enumerate(
                    zip(observed_tail_layers, expected_tail_layers)
                )
                if observed != expected
            ),
            min(len(observed_tail_layers), len(expected_tail_layers)),
        )
        raise ValueError(
            "trace does not end in complete 0.."
            f"{layers - 1} decode cycles; first mismatch at tail record {mismatch}"
        )

    last_prefill_index: dict[int, int] = {}
    for index, record in enumerate(prefill):
        if 0 <= record.layer < layers:
            last_prefill_index[record.layer] = index
    missing_layers = [
        layer for layer in range(layers) if layer not in last_prefill_index
    ]
    if missing_layers:
        raise ValueError(f"prefill contains no record for layers {missing_layers}")

    first_token_indices = [last_prefill_index[layer] for layer in range(layers)]
    if first_token_indices != sorted(first_token_indices):
        raise ValueError(
            "the final prefill access for each layer is not ordered 0.."
            f"{layers - 1}; cannot form the first generated token"
        )

    selected = tuple(prefill[index] for index in first_token_indices) + tuple(
        decode_tail
    )
    expected_selected_layers = list(range(layers)) * decode_tokens
    observed_selected_layers = [record.layer for record in selected]
    if observed_selected_layers != expected_selected_layers:
        raise AssertionError(
            "internal error: selected trace has invalid layer sequence"
        )
    if any(len(record.experts) != top_k for record in selected):
        raise AssertionError("internal error: selected trace has inconsistent top-k")

    return DecodeTrace(
        records=selected,
        source_record_count=len(records),
        prefill_record_count=len(prefill),
        decode_tokens=decode_tokens,
        layers=layers,
        top_k=top_k,
    )


def object_size_bytes(precision: int) -> int:
    if precision < MIN_PRECISION or precision > MAX_PRECISION:
        raise ValueError(
            f"precision must be in [{MIN_PRECISION}, {MAX_PRECISION}], got {precision}"
        )
    return CODEBOOK_BYTES_PER_OBJECT + precision * BITPLANE_BYTES_PER_OBJECT_BIT


def build_accesses(trace: DecodeTrace, fixed_precision: int | None) -> list[Access]:
    accesses: list[Access] = []
    if fixed_precision is not None:
        object_size_bytes(fixed_precision)
    for record in trace.records:
        targets = (
            (fixed_precision,) * trace.top_k
            if fixed_precision is not None
            else record.targets
        )
        accesses.extend(
            Access(record.layer, expert, precision)
            for expert, precision in zip(record.experts, targets)
        )
    return accesses


def simulate_lru(accesses: Sequence[Access], capacity_bytes: int) -> SimResult:
    if capacity_bytes < 0:
        raise ValueError("cache capacity must be non-negative")
    if not accesses:
        raise ValueError("cannot simulate an empty access sequence")

    cache: OrderedDict[tuple[int, int, int], int] = OrderedDict()
    used_bytes = 0
    hits = 0
    byte_hits = 0
    request_bytes = 0

    for access in accesses:
        key = access.key
        size = object_size_bytes(access.precision)
        request_bytes += size
        if key in cache:
            hits += 1
            byte_hits += size
            cache.move_to_end(key)
            continue

        if size > capacity_bytes:
            continue
        while cache and used_bytes + size > capacity_bytes:
            _, evicted_size = cache.popitem(last=False)
            used_bytes -= evicted_size
        cache[key] = size
        used_bytes += size

    return SimResult(
        capacity_bytes=capacity_bytes,
        hits=hits,
        requests=len(accesses),
        byte_hits=byte_hits,
        request_bytes=request_bytes,
        objects_final=len(cache),
        used_bytes_final=used_bytes,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def route_sha256(trace: DecodeTrace) -> str:
    digest = hashlib.sha256()
    for token in range(trace.decode_tokens):
        for record in trace.records[token * trace.layers : (token + 1) * trace.layers]:
            digest.update(
                f"{token}:{record.layer}:{','.join(map(str, record.experts))}\n".encode()
            )
    return digest.hexdigest()


def compare_routes(c4: DecodeTrace, c3: DecodeTrace) -> dict[str, object]:
    if (
        c4.decode_tokens,
        c4.layers,
        c4.top_k,
    ) != (
        c3.decode_tokens,
        c3.layers,
        c3.top_k,
    ):
        raise ValueError("cannot compare traces with different decode dimensions")

    matching_records = 0
    matching_expert_sets = 0
    matching_positions = 0
    expert_set_overlap = 0
    first_divergence: dict[str, object] | None = None
    for index, (record4, record3) in enumerate(zip(c4.records, c3.records)):
        if record4.layer != record3.layer:
            raise AssertionError("aligned decode traces have different layers")
        if record4.experts == record3.experts:
            matching_records += 1
        elif first_divergence is None:
            first_divergence = {
                "token": index // c4.layers,
                "layer": record4.layer,
                "c4_experts": list(record4.experts),
                "c3_experts": list(record3.experts),
            }
        if set(record4.experts) == set(record3.experts):
            matching_expert_sets += 1
        matching_positions += sum(
            expert4 == expert3
            for expert4, expert3 in zip(record4.experts, record3.experts)
        )
        expert_set_overlap += len(set(record4.experts) & set(record3.experts))

    record_count = len(c4.records)
    expert_position_count = record_count * c4.top_k
    return {
        "identical": matching_records == record_count,
        "record_count": record_count,
        "matching_records": matching_records,
        "divergent_records": record_count - matching_records,
        "matching_record_rate": matching_records / record_count,
        "matching_expert_sets": matching_expert_sets,
        "matching_expert_set_rate": matching_expert_sets / record_count,
        "expert_position_count": expert_position_count,
        "matching_expert_positions": matching_positions,
        "divergent_expert_positions": expert_position_count - matching_positions,
        "matching_expert_position_rate": matching_positions / expert_position_count,
        "mean_expert_set_overlap": expert_set_overlap / record_count,
        "first_divergence": first_divergence,
    }


def access_stats(accesses: Sequence[Access]) -> dict[str, object]:
    histogram = Counter(access.precision for access in accesses)
    return {
        "requests": len(accesses),
        "mean_precision_bits": sum(access.precision for access in accesses)
        / len(accesses),
        "precision_histogram": {
            str(precision): histogram[precision] for precision in sorted(histogram)
        },
        "unique_exact_objects": len({access.key for access in accesses}),
        "unique_layer_experts": len(
            {(access.layer, access.expert_id) for access in accesses}
        ),
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
            "legend.fontsize": 8.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def plot_results(
    output_stem: Path,
    rows: Sequence[dict[str, object]],
    cache_gibs: Sequence[int],
) -> None:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(3.65, 2.45))
    all_y: list[float] = []
    for method in METHOD_ORDER:
        method_rows = [row for row in rows if row["method"] == method]
        method_rows.sort(key=lambda row: int(row["cache_gib"]))
        x_values = [int(row["cache_gib"]) for row in method_rows]
        if x_values != sorted(cache_gibs):
            raise ValueError(f"plot data for {method} has the wrong cache grid")
        y_values = [100.0 * float(row["request_hit_rate"]) for row in method_rows]
        all_y.extend(y_values)
        style = METHOD_STYLE[method]
        ax.plot(
            x_values,
            y_values,
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=1.45,
            markersize=3.0,
            markerfacecolor="#FFFFFF",
            markeredgecolor=style["color"],
            markeredgewidth=0.75,
            zorder=3,
        )
    y_min = max(0, math.floor((min(all_y) - 4.0) / 10.0) * 10)
    y_max = min(100, math.ceil((max(all_y) + 4.0) / 10.0) * 10)
    if y_max - y_min < 20:
        y_max = min(100, y_min + 20)
        y_min = max(0, y_max - 20)

    cache_min = min(cache_gibs)
    cache_max = max(cache_gibs)
    ax.set_xlim(cache_min - 2, cache_max + 2)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(
        [value for value in cache_gibs if value % 8 == 0] or list(sorted(cache_gibs))
    )
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}%"))
    ax.grid(
        axis="y",
        linestyle=(0, (2, 2)),
        color="#DDDDDD",
        linewidth=0.5,
        alpha=0.8,
    )
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Cache size (GiB)")
    ax.set_ylabel("Exact-object request hit rate")
    ax.legend(
        loc="best",
        ncol=2,
        frameon=False,
        handlelength=1.7,
        columnspacing=0.8,
        handletextpad=0.4,
        borderaxespad=0.2,
    )
    fig.tight_layout(pad=0.45)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty result CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def trace_metadata(
    case: str,
    path: Path,
    file_sha256: str,
    parse_stats: ParseStats,
    trace: DecodeTrace,
) -> dict[str, object]:
    selected_target_bits = [record.target_bits for record in trace.records]
    return {
        "case": case,
        "path": str(path),
        "file_size_bytes": path.stat().st_size,
        "sha256": file_sha256,
        "trace_marker_count": parse_stats.trace_marker_count,
        "marginal_gain_marker_count": parse_stats.marginal_gain_marker_count,
        "marginal_gain_record_count": parse_stats.marginal_gain_record_count,
        "source_record_count": trace.source_record_count,
        "prefill_record_count": trace.prefill_record_count,
        "selected_record_count": len(trace.records),
        "selected_expert_access_count": len(trace.records) * trace.top_k,
        "selected_target_bits_min": min(selected_target_bits),
        "selected_target_bits_max": max(selected_target_bits),
        "selected_compute_bit_budgets": sorted(
            {record.compute_bit_budget for record in trace.records}
        ),
        "selected_max_load_bits": sorted(
            {record.max_load_bits for record in trace.records}
        ),
        "route_sha256": route_sha256(trace),
    }


def result_rows(
    methods: dict[str, tuple[str, list[Access]]],
    traces: dict[str, DecodeTrace],
    trace_meta: dict[str, dict[str, object]],
    cache_gibs: Sequence[int],
    script_sha256: str,
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    rows: list[dict[str, object]] = []
    method_metadata: dict[str, dict[str, object]] = {}
    for method in METHOD_ORDER:
        case, accesses = methods[method]
        stats = access_stats(accesses)
        method_metadata[method] = {"trace_case": case, **stats}
        meta = trace_meta[case]
        trace = traces[case]
        for cache_gib in sorted(cache_gibs):
            result = simulate_lru(accesses, cache_gib * GIB)
            rows.append(
                {
                    "method": method,
                    "comparison_design": "pairwise matched within each live-run route",
                    "matched_pair": PAIR_BY_METHOD[method],
                    "trace_case": case,
                    "trace_path": meta["path"],
                    "trace_sha256": meta["sha256"],
                    "route_sha256": meta["route_sha256"],
                    "script_sha256": script_sha256,
                    "decode_selection": "last prefill access per layer + final N-1 decode cycles",
                    "decode_tokens": trace.decode_tokens,
                    "layers": trace.layers,
                    "top_k": trace.top_k,
                    "source_records": trace.source_record_count,
                    "prefill_records": trace.prefill_record_count,
                    "selected_records": len(trace.records),
                    "mean_precision_bits": f"{float(stats['mean_precision_bits']):.8f}",
                    "precision_histogram": json.dumps(
                        stats["precision_histogram"],
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "unique_exact_objects": stats["unique_exact_objects"],
                    "unique_layer_experts": stats["unique_layer_experts"],
                    "target_bits_min": meta["selected_target_bits_min"],
                    "target_bits_max": meta["selected_target_bits_max"],
                    "compute_bit_budgets": json.dumps(
                        meta["selected_compute_bit_budgets"], separators=(",", ":")
                    ),
                    "max_load_bits": json.dumps(
                        meta["selected_max_load_bits"], separators=(",", ":")
                    ),
                    "cache_policy": "global atomic byte-capacity LRU",
                    "object_key": "(layer,expert_id,precision)",
                    "codebook_bytes_per_object": CODEBOOK_BYTES_PER_OBJECT,
                    "bitplane_bytes_per_object_bit": BITPLANE_BYTES_PER_OBJECT_BIT,
                    "cache_gib": cache_gib,
                    "capacity_bytes": result.capacity_bytes,
                    "hits": result.hits,
                    "misses": result.requests - result.hits,
                    "requests": result.requests,
                    "request_hit_rate": f"{result.request_hit_rate:.8f}",
                    "byte_hits": result.byte_hits,
                    "byte_misses": result.request_bytes - result.byte_hits,
                    "request_bytes": result.request_bytes,
                    "byte_hit_rate": f"{result.byte_hit_rate:.8f}",
                    "objects_final": result.objects_final,
                    "used_bytes_final": result.used_bytes_final,
                    "used_gib_final": f"{result.used_bytes_final / GIB:.8f}",
                }
            )
    return rows, method_metadata


def make_record(
    layer: int,
    expert_base: int,
    precision: int = 4,
    source_line: int = 1,
) -> PrecisionRecord:
    experts = tuple(range(expert_base, expert_base + TOP_K))
    targets = (precision,) * TOP_K
    bits = precision * TOP_K
    return PrecisionRecord(
        layer=layer,
        experts=experts,
        targets=targets,
        target_bits=bits,
        compute_bit_budget=bits,
        max_load_bits=bits,
        source_line=source_line,
    )


def run_self_check() -> None:
    sample = (
        "prefix [pagedmoe-precision-trace] mode=marginal_gain layer=Some(7) "
        "experts=[0,1,2,3,4,5,6,7,8,9] weights=[] cached=[] "
        "target=[4,4,4,4,4,4,4,4,4,4] delta=[] cached_bits=0 "
        "target_bits=40 compute_bit_budget=Some(40) max_load_bits=Some(40)\n"
    )
    parsed, stats = parse_trace_lines([sample], "self-check", TOP_K)
    assert len(parsed) == 1 and stats.marginal_gain_record_count == 1
    validate_trace_config(parsed, "self-check", 40)

    prefill: list[PrecisionRecord] = []
    for layer in range(LAYERS):
        prefill.append(make_record(layer, 0))
        prefill.append(make_record(layer, 10))
    tail = [make_record(layer, 20) for layer in range(LAYERS)]
    selected = select_decode_trace(prefill + tail, 2, LAYERS, TOP_K)
    assert len(selected.records) == 2 * LAYERS
    assert selected.records[0].experts[0] == 10
    assert selected.records[LAYERS].experts[0] == 20

    accesses = [
        Access(0, 0, 4),
        Access(0, 0, 4),
        Access(0, 0, 3),
        Access(0, 0, 4),
    ]
    both_objects = simulate_lru(accesses, object_size_bytes(4) + object_size_bytes(3))
    one_object = simulate_lru(accesses, object_size_bytes(4))
    assert both_objects.hits == 2
    assert one_object.hits == 1
    assert object_size_bytes(4) == 3 * 32 * 1024 + 4 * 3 * 32 * 16 * 1024
    print("self-check passed: parser, decode selection, and exact-object LRU")


def main() -> int:
    args = parse_args()
    if args.self_check:
        run_self_check()
        return 0
    if args.c4_trace is None or args.c3_trace is None:
        raise ValueError("--c4-trace and --c3-trace are required")
    if not args.cache_gib or any(value <= 0 for value in args.cache_gib):
        raise ValueError("--cache-gib values must be positive")
    if len(set(args.cache_gib)) != len(args.cache_gib):
        raise ValueError("--cache-gib values must be unique")
    if args.primary_cache_gib not in args.cache_gib:
        raise ValueError("--primary-cache-gib must be present in --cache-gib")

    c4_path = args.c4_trace.resolve(strict=True)
    c3_path = args.c3_trace.resolve(strict=True)
    c4_records, c4_parse_stats = parse_trace(c4_path, args.top_k)
    c3_records, c3_parse_stats = parse_trace(c3_path, args.top_k)
    validate_trace_config(c4_records, "c4l40", args.expected_c4_bits)
    validate_trace_config(c3_records, "c3l30", args.expected_c3_bits)
    c4_trace = select_decode_trace(
        c4_records, args.decode_tokens, args.layers, args.top_k
    )
    c3_trace = select_decode_trace(
        c3_records, args.decode_tokens, args.layers, args.top_k
    )

    traces = {"c4l40": c4_trace, "c3l30": c3_trace}
    trace_meta = {
        "c4l40": trace_metadata(
            "c4l40", c4_path, sha256_file(c4_path), c4_parse_stats, c4_trace
        ),
        "c3l30": trace_metadata(
            "c3l30", c3_path, sha256_file(c3_path), c3_parse_stats, c3_trace
        ),
    }
    methods = {
        "Fixed4": ("c4l40", build_accesses(c4_trace, fixed_precision=4)),
        "Fixed3": ("c3l30", build_accesses(c3_trace, fixed_precision=3)),
        "DynAvg4": ("c4l40", build_accesses(c4_trace, fixed_precision=None)),
        "DynAvg3": ("c3l30", build_accesses(c3_trace, fixed_precision=None)),
    }
    script_sha256 = sha256_file(Path(__file__).resolve())
    rows, method_meta = result_rows(
        methods, traces, trace_meta, args.cache_gib, script_sha256
    )
    route_comparison = compare_routes(c4_trace, c3_trace)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / args.csv_name
    primary_csv_path = output_dir / args.primary_csv_name
    output_stem = output_dir / args.output_stem
    metadata_path = output_dir / args.metadata_name
    write_csv(csv_path, rows)
    primary_rows = [
        row for row in rows if int(row["cache_gib"]) == args.primary_cache_gib
    ]
    write_csv(primary_csv_path, primary_rows)
    plot_results(output_stem, rows, args.cache_gib)
    primary_results = {
        str(row["method"]): {
            "request_hit_rate": float(row["request_hit_rate"]),
            "byte_hit_rate": float(row["byte_hit_rate"]),
            "hits": int(row["hits"]),
            "requests": int(row["requests"]),
        }
        for row in primary_rows
    }
    metadata = {
        "script": str(Path(__file__).resolve()),
        "script_sha256": script_sha256,
        "comparison_design": {
            "type": "pairwise matched within each live-run route",
            "pairs": {
                "c4l40": ["Fixed4", "DynAvg4"],
                "c3l30": ["Fixed3", "DynAvg3"],
            },
            "warning": (
                "The c4l40 and c3l30 generations may have different routes; "
                "do not interpret all four methods as replaying one shared trace."
            ),
        },
        "cache_policy": "global atomic byte-capacity LRU",
        "exact_object_key": ["layer", "expert_id", "precision"],
        "object_size_bytes": {
            "formula": "3*32*1024 + precision*3*32*16384",
            "codebook": CODEBOOK_BYTES_PER_OBJECT,
            "per_bitplane": BITPLANE_BYTES_PER_OBJECT_BIT,
        },
        "decode_selection": (
            "first output token uses the last prefill record for each layer; "
            "the remaining tokens use the final complete 0..layers-1 cycles"
        ),
        "cache_gib": sorted(args.cache_gib),
        "primary_cache_gib": args.primary_cache_gib,
        "primary_results": primary_results,
        "traces": trace_meta,
        "methods": method_meta,
        "route_comparison": route_comparison,
        "outputs": {
            "csv": str(csv_path),
            "primary_csv": str(primary_csv_path),
            "png": str(output_stem.with_suffix(".png")),
            "pdf": str(output_stem.with_suffix(".pdf")),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {csv_path}")
    print(f"wrote {primary_csv_path}")
    print(f"wrote {output_stem.with_suffix('.png')}")
    print(f"wrote {output_stem.with_suffix('.pdf')}")
    print(f"wrote {metadata_path}")
    print(
        "routes: "
        f"identical={route_comparison['identical']} "
        f"matching_records={route_comparison['matching_records']}/"
        f"{route_comparison['record_count']} "
        f"matching_expert_positions={route_comparison['matching_expert_positions']}/"
        f"{route_comparison['expert_position_count']}"
    )
    if route_comparison["first_divergence"] is not None:
        first = route_comparison["first_divergence"]
        print(f"first route divergence: token={first['token']} layer={first['layer']}")
    for method in METHOD_ORDER:
        stats = method_meta[method]
        print(
            f"{method}: requests={stats['requests']} "
            f"mean_precision={float(stats['mean_precision_bits']):.4f} "
            f"unique_exact_objects={stats['unique_exact_objects']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
