#!/usr/bin/env python3
"""Plot breakdown decode throughput versus cache size from summary.md."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_SUMMARY = REPO_ROOT / "logs/qwen35_breakdown_combined_20260701/summary.md"

PALETTE = {
    "ours": "#D55E00",
    "ours_light": "#E69F00",
    "strong_baseline": "#009E73",
    "baseline_1": "#009E73",
    "baseline_2": "#CC79A7",
    "black": "#000000",
    "gray": "#0072B2",
    "light_gray": "#DDDDDD",
    "white": "#FFFFFF",
}

MODEL_CONFIGS = {
    "qwen35": {
        "heading": "## Decode TPS",
        "output_name": "fig_qwen35_breakdown_cache_size_token_s",
        "yticks": [0, 2, 4, 6, 8, 10],
        "ylim": (0, 10),
        "series": [
            {
                "column": "mmap c4l40",
                "label": "mmap c4 L40",
                "color": PALETTE["strong_baseline"],
                "linestyle": ":",
                "marker": "D",
            },
            {
                "column": "bitplane static4",
                "label": "bitplane static4",
                "color": PALETTE["gray"],
                "linestyle": "-",
                "marker": "s",
            },
            {
                "column": "mmap c2.5l25",
                "label": "mmap c2.5 L25",
                "color": PALETTE["ours"],
                "linestyle": ":",
                "marker": "o",
            },
            {
                "column": "naive c2.5l25",
                "label": "naive c2.5 L25",
                "color": PALETTE["ours"],
                "linestyle": "--",
                "marker": "o",
            },
            {
                "column": "bitplane c2.5l25",
                "label": "bitplane c2.5 L25",
                "color": PALETTE["ours"],
                "linestyle": "-",
                "marker": "o",
            },
        ],
    },
    "glm52": {
        "heading": "## GLM5.2 Decode TPS",
        "output_name": "fig_glm52_breakdown_cache_size_token_s",
        "yticks": [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        "ylim": (0, 3.0),
        "series": [
            {
                "column": "mmap 4",
                "label": "mmap 4",
                "color": PALETTE["strong_baseline"],
                "linestyle": ":",
                "marker": "D",
            },
            {
                "column": "bitplane 4",
                "label": "bitplane 4",
                "color": PALETTE["gray"],
                "linestyle": "-",
                "marker": "s",
            },
            {
                "column": "mmap 2.5*",
                "label": "mmap 2.5",
                "color": PALETTE["ours"],
                "linestyle": ":",
                "marker": "o",
            },
            {
                "column": "naive 2.5",
                "label": "naive 2.5",
                "color": PALETTE["ours"],
                "linestyle": "--",
                "marker": "o",
            },
            {
                "column": "bitplane 2.5",
                "label": "bitplane 2.5",
                "color": PALETTE["ours"],
                "linestyle": "-",
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
            "legend.fontsize": 6.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": PALETTE["white"],
            "axes.facecolor": PALETTE["white"],
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def split_markdown_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def parse_number(value: str) -> float | None:
    normalized = value.strip().lower()
    if normalized in {"", "-", "failed"}:
        return None
    return float(normalized.rstrip("x"))


def read_markdown_table(summary_md: Path, heading: str) -> list[dict[str, str]]:
    lines = summary_md.read_text(encoding="utf-8").splitlines()
    try:
        heading_index = next(
            index for index, line in enumerate(lines) if line.strip() == heading
        )
    except StopIteration as exc:
        raise ValueError(f"heading not found in {summary_md}: {heading}") from exc

    table_lines: list[str] = []
    for line in lines[heading_index + 1 :]:
        if line.startswith("|"):
            table_lines.append(line)
        elif table_lines:
            break

    if len(table_lines) < 3:
        raise ValueError(f"no markdown table found after heading: {heading}")

    header = split_markdown_row(table_lines[0])
    rows = []
    for line in table_lines[2:]:
        cells = split_markdown_row(line)
        if len(cells) != len(header):
            raise ValueError(f"malformed markdown table row: {line}")
        rows.append(dict(zip(header, cells, strict=True)))
    return rows


def read_series(
    summary_md: Path, heading: str, series_specs: list[dict[str, str]]
) -> dict[str, list[tuple[float, float]]]:
    rows = read_markdown_table(summary_md, heading)
    series: dict[str, list[tuple[float, float]]] = {
        spec["column"]: [] for spec in series_specs
    }

    for row in rows:
        cache_gib = parse_number(row["cache GiB"])
        if cache_gib is None:
            continue

        for spec in series_specs:
            value = parse_number(row[spec["column"]])
            if value is not None:
                series[spec["column"]].append((cache_gib, value))

    missing = [column for column, values in series.items() if not values]
    if missing:
        raise ValueError(f"missing series values in {summary_md}: {missing}")
    return series


def plot(summary_md: Path, output_stem: Path, model: str) -> None:
    configure_matplotlib()
    config = MODEL_CONFIGS[model]
    data = read_series(summary_md, config["heading"], config["series"])

    fig, ax = plt.subplots(figsize=(3.35, 2.35))

    for spec in config["series"]:
        values = data[spec["column"]]
        x_values = [cache_gib for cache_gib, _ in values]
        y_values = [tokens_per_s for _, tokens_per_s in values]

        marker = spec["marker"]
        markerfacecolor = PALETTE["white"] if marker != "x" else spec["color"]
        ax.plot(
            x_values,
            y_values,
            color=spec["color"],
            linestyle=spec["linestyle"],
            linewidth=1.45,
            marker=marker,
            markersize=3.0,
            markerfacecolor=markerfacecolor,
            markeredgecolor=spec["color"],
            markeredgewidth=0.75,
            label=spec["label"],
            zorder=3,
        )

    ax.set_xlim(6, 66)
    ax.set_ylim(*config["ylim"])
    ax.set_xticks([8, 16, 24, 32, 40, 48, 56, 64])
    ax.set_yticks(config["yticks"])
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

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.45)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def default_output_stem(model: str) -> Path:
    return SCRIPT_DIR / MODEL_CONFIGS[model]["output_name"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-md",
        type=Path,
        default=DEFAULT_SUMMARY,
        help=f"breakdown summary markdown (default: {DEFAULT_SUMMARY})",
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_CONFIGS),
        default="qwen35",
        help="which Decode TPS table to plot",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=None,
        help="output figure stem without extension",
    )
    args = parser.parse_args()

    output_stem = args.output_stem or default_output_stem(args.model)
    plot(args.summary_md, output_stem, args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
