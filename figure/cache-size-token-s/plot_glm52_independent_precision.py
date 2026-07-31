#!/usr/bin/env python3
"""Plot the GLM-5.2 independent-precision cache ablation (Figure 2)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


SCRIPT_DIR = Path(__file__).resolve().parent
EXPECTED_CACHE_SIZES = tuple(float(value) for value in range(4, 49, 4))
METRIC = "response_decode_phase_token_per_s"
EXPECTED_PROMPT_SHA256 = (
    "b80afd3b60770ee77ab03e892065ebda05c006ebeb981a529f2ee5efb6f60131"
)
EXPECTED_SHARED_SHA256 = (
    "b66c7561203158a1907c448e4e75f68405bc13b25c32ded20fab94097ce149ee"
)
EXPECTED_NAIVE_SHA256 = (
    "17e3159b6f0c5d9b4b6952d803cad80c54117d41d2f139d7a938bd45ef14f79a"
)
EXPECTED_NAIVE_RUN_ID = "glm52_fig2_naive_formal_20260730_0808"

PALETTE = {
    "uniform4": "#0072B2",
    "dynamic2p5": "#D55E00",
    "light_gray": "#DDDDDD",
    "white": "#FFFFFF",
}


@dataclass(frozen=True)
class SeriesSpec:
    key: str
    label: str
    backend: str
    source_kind: str
    source_case: str
    color: str
    linestyle: str
    marker: str


SERIES = (
    SeriesSpec(
        key="uniform4",
        label="Uniform 4-bit",
        backend="shared-bitplane",
        source_kind="shared",
        source_case="static4",
        color=PALETTE["uniform4"],
        linestyle="-",
        marker="s",
    ),
    SeriesSpec(
        key="naive2p5",
        label="Dynamic Avg 2.5-bit (independent)",
        backend="independent-precision",
        source_kind="naive",
        source_case="c2p5l20",
        color=PALETTE["dynamic2p5"],
        linestyle="--",
        marker="o",
    ),
    SeriesSpec(
        key="shared2p5",
        label="Dynamic Avg 2.5-bit",
        backend="shared-bitplane",
        source_kind="shared",
        source_case="c2p5l20",
        color=PALETTE["dynamic2p5"],
        linestyle="-",
        marker="o",
    ),
)


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
            "legend.fontsize": 6.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": PALETTE["white"],
            "axes.facecolor": PALETTE["white"],
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_columns(path: Path, fieldnames: list[str] | None) -> None:
    required = {
        "status",
        "run_id",
        "precision_mode",
        "precision",
        "average_precision",
        "compute_avg_bits",
        "max_load_bits",
        "cache_size_gib",
        "decode_steps",
        "prompt_sha256",
        "prompt_reasoning_effort_max",
        "prompt_endswith_think",
        "response_prompt_tokens",
        "completion_tokens",
        "stream",
        "stream_interval",
        "response_first_chunk_completion_tokens",
        "response_decode_phase_tokens",
        METRIC,
    }
    missing = required.difference(fieldnames or ())
    if missing:
        raise ValueError(f"{path}: missing columns: {sorted(missing)}")


def validate_protocol(path: Path, row: dict[str, str]) -> None:
    checks = {
        "status": row["status"] == "ok",
        "decode_steps": row["decode_steps"] == "500",
        "prompt_sha256": row["prompt_sha256"] == EXPECTED_PROMPT_SHA256,
        "reasoning_effort": row["prompt_reasoning_effort_max"] == "1",
        "think_suffix": row["prompt_endswith_think"] == "1",
        "prompt_tokens": row["response_prompt_tokens"] == "118",
        "completion_tokens": row["completion_tokens"] == "500",
        "stream": row["stream"] == "1",
        "stream_interval": row["stream_interval"] == "50",
        "first_chunk": row["response_first_chunk_completion_tokens"] == "1",
        "decode_phase_tokens": row["response_decode_phase_tokens"] == "499",
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            f"{path}: cache={row['cache_size_gib']} failed protocol checks: "
            + ", ".join(failed)
        )


def validate_precision(path: Path, row: dict[str, str], spec: SeriesSpec) -> None:
    if spec.key == "uniform4":
        expected = {
            "precision_mode": "schedule",
            "precision": "4,4,4,4,4,4,4,4",
            "average_precision": "4.0000",
            "compute_avg_bits": "",
            "max_load_bits": "",
        }
    else:
        expected = {
            "precision_mode": "marginal_gain",
            "precision": "c2p5l20",
            "average_precision": "2.5000",
            "compute_avg_bits": "2.5",
            "max_load_bits": "20",
        }
    mismatches = [
        f"{key}={row[key]!r} (expected {value!r})"
        for key, value in expected.items()
        if row[key] != value
    ]
    if mismatches:
        raise ValueError(f"{path}: " + ", ".join(mismatches))


def read_series(
    shared_csv: Path, naive_csv: Path
) -> tuple[dict[str, list[tuple[float, float, str]]], dict[str, str]]:
    source_rows: dict[str, list[dict[str, str]]] = {}
    for kind, path in (("shared", shared_csv), ("naive", naive_csv)):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            require_columns(path, reader.fieldnames)
            if kind == "shared" and "case" not in (reader.fieldnames or []):
                raise ValueError(f"{path}: missing column: case")
            source_rows[kind] = list(reader)

    result: dict[str, list[tuple[float, float, str]]] = {}
    for spec in SERIES:
        path = shared_csv if spec.source_kind == "shared" else naive_csv
        rows = source_rows[spec.source_kind]
        if spec.source_kind == "shared":
            rows = [row for row in rows if row["case"] == spec.source_case]
        else:
            rows = [row for row in rows if row["precision"] == spec.source_case]

        values: dict[float, tuple[float, str]] = {}
        for row in rows:
            validate_protocol(path, row)
            validate_precision(path, row, spec)
            cache_size = float(row["cache_size_gib"])
            if cache_size > 48:
                continue
            if cache_size in values:
                raise ValueError(f"{path}: duplicate cache size {cache_size:g}")
            metric = float(row[METRIC])
            if not math.isfinite(metric) or metric <= 0:
                raise ValueError(
                    f"{path}: cache={cache_size:g} has invalid metric {metric}"
                )
            if (
                spec.source_kind == "naive"
                and row["run_id"] != EXPECTED_NAIVE_RUN_ID
            ):
                raise ValueError(
                    f"{path}: unexpected naive run_id={row['run_id']!r}"
                )
            values[cache_size] = (metric, row["run_id"])

        actual = set(values)
        expected = set(EXPECTED_CACHE_SIZES)
        if actual != expected:
            raise ValueError(
                f"{spec.key} cache grid mismatch: "
                f"missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            )
        result[spec.key] = [
            (cache_size, metric, run_id)
            for cache_size, (metric, run_id) in sorted(values.items())
        ]

    source_hashes = {"shared": sha256(shared_csv), "naive": sha256(naive_csv)}
    expected_hashes = {
        "shared": EXPECTED_SHARED_SHA256,
        "naive": EXPECTED_NAIVE_SHA256,
    }
    for kind, expected_hash in expected_hashes.items():
        if source_hashes[kind] != expected_hash:
            raise ValueError(
                f"{kind} source hash mismatch: expected={expected_hash} "
                f"actual={source_hashes[kind]}"
            )
    return result, source_hashes


def write_plot_data(
    output_path: Path,
    values: dict[str, list[tuple[float, float, str]]],
    source_hashes: dict[str, str],
    shared_csv: Path,
    naive_csv: Path,
) -> None:
    rows = []
    for spec in SERIES:
        source = shared_csv if spec.source_kind == "shared" else naive_csv
        for cache_size, metric, run_id in values[spec.key]:
            rows.append(
                {
                    "curve_key": spec.key,
                    "series": spec.label,
                    "backend": spec.backend,
                    "cache_size_gib": f"{cache_size:g}",
                    METRIC: f"{metric:.8f}",
                    "run_id": run_id,
                    # Keep the derived artifact stable across checkout locations.
                    "source": source.name,
                    "source_sha256": source_hashes[spec.source_kind],
                }
            )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot(
    shared_csv: Path, naive_csv: Path, output_dir: Path, plot_data_csv: Path
) -> None:
    configure_matplotlib()
    values, source_hashes = read_series(shared_csv, naive_csv)

    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    y_max = 0.0
    for spec in SERIES:
        x_values = [point[0] for point in values[spec.key]]
        y_values = [point[1] for point in values[spec.key]]
        y_max = max(y_max, *y_values)
        ax.plot(
            x_values,
            y_values,
            color=spec.color,
            linestyle=spec.linestyle,
            linewidth=1.45,
            marker=spec.marker,
            markersize=3.0,
            markerfacecolor=PALETTE["white"],
            markeredgecolor=spec.color,
            markeredgewidth=0.75,
            label=spec.label,
            zorder=3,
        )

    ax.set_xlim(0, 50)
    ax.set_ylim(0, y_max * 1.08)
    ax.set_xticks(range(0, 49, 8))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4))
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
    ax.set_xlabel("Cache capacity (GiB)")
    ax.set_ylabel("Decode throughput (tokens/s)")
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

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.45)
    for name in (
        "fig_cache_size_independent_precision_comparison_glm52",
        "fig_cache_size_prefix_reuse_ablation_glm52",
    ):
        stem = output_dir / name
        fig.savefig(
            stem.with_suffix(".pdf"),
            bbox_inches="tight",
            metadata={"CreationDate": None, "ModDate": None},
        )
        fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    write_plot_data(
        plot_data_csv, values, source_hashes, shared_csv, naive_csv
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shared-summary-csv",
        type=Path,
        default=SCRIPT_DIR / "glm52_cache_size_token_s.csv",
    )
    parser.add_argument(
        "--naive-summary-csv",
        type=Path,
        default=SCRIPT_DIR / "glm52_naive_c2p5l20.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument(
        "--plot-data-csv",
        type=Path,
        default=SCRIPT_DIR / "glm52_figure2_cache_size_token_s.csv",
    )
    args = parser.parse_args()
    plot(
        args.shared_summary_csv,
        args.naive_summary_csv,
        args.output_dir,
        args.plot_data_csv,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
