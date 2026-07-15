#!/usr/bin/env python3
"""Plot cache-size throughput and cache-policy ablations for Qwen3.5."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
OLD_SUMMARY = (
    REPO_ROOT
    / "src/target/bench/kt_pagedmoe_dynamic_static_cache8-64_perf_hitrate_summary.csv"
)
BREAKDOWN_SUMMARY = REPO_ROOT / "logs/qwen35_breakdown_combined_20260701/summary.md"
MMAP_STATIC4_RAW_SUMMARY = (
    REPO_ROOT
    / "logs/qwen35_mmap_static4_c2p5l25_8to64_20260630_182406/summary.raw.csv"
)
IK_LLAMA_MMAP_Q4_0 = [(16.0, 1.21), (32.0, 1.50), (48.0, 1.99), (64.0, 2.32)]

PALETTE = {
    "ours": "#D55E00",
    "strong_baseline": "#009E73",
    "secondary": "#009E73",
    "gray": "#0072B2",
    "black": "#000000",
    "light_gray": "#DDDDDD",
    "white": "#FFFFFF",
}

FIGURES = {
    "end-to-end": {
        "output_name": "fig_cache_size_end_to_end_throughput",
        "series": [
            {
                "key": "ik_llama_mmap_q4_0",
                "label": "ik_llama.cpp Q4_0",
                "color": PALETTE["black"],
                "linestyle": "-.",
                "marker": "^",
            },
            {
                "key": "uniform4",
                "label": "Uniform 4-bit",
                "color": PALETTE["gray"],
                "linestyle": "-",
                "marker": "s",
            },
            {
                "key": "dynamic4",
                "label": "Dynamic Avg 4-bit",
                "color": PALETTE["strong_baseline"],
                "linestyle": "-",
                "marker": "D",
            },
            {
                "key": "dynamic2p5",
                "label": "Dynamic Avg 2.5-bit",
                "color": PALETTE["ours"],
                "linestyle": "-",
                "marker": "o",
            },
        ],
    },
    "prefix-reuse": {
        "output_name": "fig_cache_size_prefix_reuse_ablation",
        "series": [
            {
                "key": "uniform4",
                "label": "Uniform 4-bit",
                "color": PALETTE["gray"],
                "linestyle": "-",
                "marker": "s",
            },
            {
                "key": "naive2p5",
                "label": "Dynamic Avg 2.5-bit w/o Prefix Reuse",
                "color": PALETTE["ours"],
                "linestyle": "--",
                "marker": "o",
            },
            {
                "key": "dynamic2p5",
                "label": "Dynamic Avg 2.5-bit",
                "color": PALETTE["ours"],
                "linestyle": "-",
                "marker": "o",
            },
        ],
    },
    "mmap-fallback": {
        "output_name": "fig_cache_size_mmap_fallback_ablation",
        "series": [
            {
                "key": "uniform4",
                "label": "Uniform 4-bit",
                "color": PALETTE["gray"],
                "linestyle": "-",
                "marker": "s",
            },
            {
                "key": "mmap_uniform4_static",
                "label": "Uniform 4-bit w/ mmap fallback",
                "color": PALETTE["gray"],
                "linestyle": ":",
                "marker": "s",
            },
            {
                "key": "dynamic2p5",
                "label": "Dynamic Avg 2.5-bit",
                "color": PALETTE["ours"],
                "linestyle": "-",
                "marker": "o",
            },
            {
                "key": "mmap_dynamic2p5_breakdown",
                "label": "Dynamic Avg 2.5-bit w/ mmap fallback",
                "color": PALETTE["ours"],
                "linestyle": ":",
                "marker": "o",
            },
        ],
    },
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
            "legend.fontsize": 6.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": PALETTE["white"],
            "axes.facecolor": PALETTE["white"],
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def read_old_series() -> dict[str, list[tuple[float, float]]]:
    case_to_key = {"c4l40": "dynamic4"}
    grouped: dict[str, defaultdict[float, list[float]]] = {
        key: defaultdict(list) for key in case_to_key.values()
    }
    with OLD_SUMMARY.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = case_to_key.get(row["case"])
            if key is None or row["status"] != "ok":
                continue
            grouped[key][float(row["cache_size_gib"])].append(
                float(row["decode_token_per_s"])
            )
    return {
        key: [
            (cache_gib, sum(values) / len(values))
            for cache_gib, values in sorted(by_cache.items())
        ]
        for key, by_cache in grouped.items()
    }


def split_markdown_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def read_breakdown_series() -> dict[str, list[tuple[float, float]]]:
    lines = BREAKDOWN_SUMMARY.read_text(encoding="utf-8").splitlines()
    heading_index = lines.index("## Decode TPS")
    table_lines: list[str] = []
    for line in lines[heading_index + 1 :]:
        if line.startswith("|"):
            table_lines.append(line)
        elif table_lines:
            break

    header = split_markdown_row(table_lines[0])
    rows = [
        dict(zip(header, split_markdown_row(line), strict=True))
        for line in table_lines[2:]
    ]
    column_to_key = {
        "bitplane static4": "uniform4",
        "mmap c2.5l25": "mmap_dynamic2p5_breakdown",
        "naive c2.5l25": "naive2p5",
        "bitplane c2.5l25": "dynamic2p5",
    }
    return {
        key: [
            (float(row["cache GiB"]), float(row[column]))
            for row in rows
            if row[column] not in {"", "-", "failed"}
        ]
        for column, key in column_to_key.items()
    }


def extract_metric(log_path: Path, name: str) -> float:
    text = log_path.read_text(encoding="utf-8")
    match = re.search(rf"\b{re.escape(name)}=([0-9.]+)", text)
    if match is None:
        raise ValueError(f"missing {name} in {log_path}")
    return float(match.group(1))


def read_mmap_static4_series() -> dict[str, list[tuple[float, float]]]:
    values: list[tuple[float, float]] = []
    with MMAP_STATIC4_RAW_SUMMARY.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["precision"] != "static4" or row["exit_code"] != "0":
                continue
            log_path = Path(row["stdout_log"])
            if not log_path.is_absolute():
                log_path = REPO_ROOT / log_path
            if extract_metric(log_path, "prompt_tokens") != 118:
                raise ValueError(f"unexpected prompt length in {log_path}")
            if extract_metric(log_path, "decode_steps") != 500:
                raise ValueError(f"unexpected decode length in {log_path}")
            values.append(
                (
                    float(row["cache_gib"]),
                    extract_metric(log_path, "decode_token_per_s_from_log"),
                )
            )
    values.sort()
    return {"mmap_uniform4_static": values}


def load_all_series() -> dict[str, list[tuple[float, float]]]:
    series = {
        **read_old_series(),
        **read_breakdown_series(),
        **read_mmap_static4_series(),
        "ik_llama_mmap_q4_0": IK_LLAMA_MMAP_Q4_0,
    }
    required = {spec["key"] for config in FIGURES.values() for spec in config["series"]}
    missing = sorted(required - series.keys())
    if missing:
        raise ValueError(f"missing series: {missing}")
    return series


def plot_figure(
    figure_name: str,
    all_series: dict[str, list[tuple[float, float]]],
    output_dir: Path,
) -> None:
    config = FIGURES[figure_name]
    fig, ax = plt.subplots(figsize=(3.35, 2.35))

    for spec in config["series"]:
        values = all_series[spec["key"]]
        ax.plot(
            [cache_gib for cache_gib, _ in values],
            [tokens_per_s for _, tokens_per_s in values],
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
    ax.set_ylim(0, 10.5)
    ax.set_xticks([8, 16, 24, 32, 40, 48, 56, 64])
    ax.set_yticks([0, 2, 4, 6, 8, 10])
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
    output_stem = output_dir / config["output_name"]
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_plot_data(
    selected_figures: list[str],
    all_series: dict[str, list[tuple[float, float]]],
    output_dir: Path,
) -> None:
    sources = {
        "dynamic4": str(OLD_SUMMARY.relative_to(REPO_ROOT)),
        "uniform4": str(BREAKDOWN_SUMMARY.relative_to(REPO_ROOT)),
        "naive2p5": str(BREAKDOWN_SUMMARY.relative_to(REPO_ROOT)),
        "dynamic2p5": str(BREAKDOWN_SUMMARY.relative_to(REPO_ROOT)),
        "ik_llama_mmap_q4_0": (
            "paper/figure/cache-size-token-s/plot_cache_size_token_s.py:manual_values"
        ),
        "mmap_uniform4_static": str(
            MMAP_STATIC4_RAW_SUMMARY.relative_to(REPO_ROOT)
        ),
        "mmap_dynamic2p5_breakdown": str(BREAKDOWN_SUMMARY.relative_to(REPO_ROOT)),
    }
    rows = []
    for figure_name in selected_figures:
        for spec in FIGURES[figure_name]["series"]:
            for cache_gib, tokens_per_s in all_series[spec["key"]]:
                rows.append(
                    {
                        "figure": figure_name,
                        "series": spec["label"],
                        "cache_size_gib": f"{cache_gib:g}",
                        "decode_token_per_s": f"{tokens_per_s:.4f}",
                        "source": sources[spec["key"]],
                    }
                )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "cache_size_ablation_plot_data.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            lineterminator="\n",
            fieldnames=[
                "figure",
                "series",
                "cache_size_gib",
                "decode_token_per_s",
                "source",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure",
        choices=["all", *FIGURES],
        default="all",
        help="figure to generate",
    )
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    args = parser.parse_args()

    configure_matplotlib()
    all_series = load_all_series()
    selected = list(FIGURES) if args.figure == "all" else [args.figure]
    for figure_name in selected:
        plot_figure(figure_name, all_series, args.output_dir)
    write_plot_data(selected, all_series, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
