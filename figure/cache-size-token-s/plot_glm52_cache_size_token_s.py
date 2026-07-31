#!/usr/bin/env python3
"""Plot the GLM-5.2 Figure 1 cache-size throughput experiment."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


SCRIPT_DIR = Path(__file__).resolve().parent
EXPECTED_CACHE_SIZES = tuple(float(value) for value in range(4, 49, 4))
EXPECTED_IK_CACHE_SIZES = tuple(float(value) for value in range(8, 49, 8))
IK_CASE = "ik_llama_ud_iq4_xs"
EXPECTED_IK_OUTPUT_SHA256 = (
    "580f38d8ae85667c3005688eaf6ba9f5fa02e7ca7da9607fb17f72d3790ec8e3"
)

PALETTE = {
    "ours": "#D55E00",
    "strong_baseline": "#0072B2",
    "static": "#999999",
    "black": "#000000",
    "light_gray": "#DDDDDD",
    "white": "#FFFFFF",
}

SERIES = (
    {
        "case": IK_CASE,
        "label": "ik_llama.cpp UD-IQ4_XS",
        "color": PALETTE["black"],
        "linestyle": "-.",
        "marker": "^",
        "source": "ik_llama",
    },
    {
        "case": "static4",
        "label": "Uniform 4-bit",
        "color": PALETTE["static"],
        "linestyle": ":",
        "marker": "s",
        "source": "pagedmoe",
    },
    {
        "case": "c4l32",
        "label": "Dynamic Avg 4-bit",
        "color": PALETTE["strong_baseline"],
        "linestyle": "--",
        "marker": "o",
        "source": "pagedmoe",
    },
    {
        "case": "c2p5l20",
        "label": "Dynamic Avg 2.5-bit",
        "color": PALETTE["ours"],
        "linestyle": "-",
        "marker": "D",
        "source": "pagedmoe",
    },
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
            "legend.fontsize": 7.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": PALETTE["white"],
            "axes.facecolor": PALETTE["white"],
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def read_summary(
    summary_csv: Path, ik_summary_csv: Path, metric: str
) -> dict[str, list[tuple[float, float]]]:
    values_by_case: dict[str, defaultdict[float, list[float]]] = {
        spec["case"]: defaultdict(list)
        for spec in SERIES
        if spec["source"] == "pagedmoe"
    }
    with summary_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"case", "status", "cache_size_gib", metric}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"missing columns in {summary_csv}: {sorted(missing)}")
        for row in reader:
            case = row["case"]
            if case not in values_by_case or row["status"] != "ok":
                continue
            cache_size = float(row["cache_size_gib"])
            if cache_size <= 48:
                values_by_case[case][cache_size].append(float(row[metric]))

    expected = set(EXPECTED_CACHE_SIZES)
    result: dict[str, list[tuple[float, float]]] = {}
    for case, values_by_cache in values_by_case.items():
        actual = set(values_by_cache)
        if actual != expected:
            raise ValueError(
                f"{case} cache grid mismatch: missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            )
        result[case] = [
            (cache_size, sum(values) / len(values))
            for cache_size, values in sorted(values_by_cache.items())
        ]

    ik_values: dict[float, float] = {}
    with ik_summary_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "case",
            "status",
            "cache_size_gib",
            "actual_routed_resident_gib",
            "decode_phase_tokens",
            "decode_token_per_s",
            "output_sha256",
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"missing columns in {ik_summary_csv}: {sorted(missing)}"
            )
        for row in reader:
            if row["case"] != IK_CASE or row["status"] != "ok":
                continue
            cache_size = float(row["cache_size_gib"])
            if cache_size in ik_values:
                raise ValueError(f"duplicate ik_llama cache size: {cache_size}")
            if int(row["decode_phase_tokens"]) != 499:
                raise ValueError(f"invalid ik_llama decode window at {cache_size} GiB")
            if row["output_sha256"] != EXPECTED_IK_OUTPUT_SHA256:
                raise ValueError(f"ik_llama output hash mismatch at {cache_size} GiB")
            actual_routed = float(row["actual_routed_resident_gib"])
            if abs(actual_routed - cache_size) > 0.5:
                raise ValueError(
                    f"ik_llama routed residency differs from {cache_size} GiB by "
                    f"more than 0.5 GiB: {actual_routed}"
                )
            ik_values[cache_size] = float(row["decode_token_per_s"])

    expected_ik = set(EXPECTED_IK_CACHE_SIZES)
    actual_ik = set(ik_values)
    if actual_ik != expected_ik:
        raise ValueError(
            "ik_llama cache grid mismatch: "
            f"missing={sorted(expected_ik - actual_ik)}, "
            f"unexpected={sorted(actual_ik - expected_ik)}"
        )
    result[IK_CASE] = sorted(ik_values.items())
    return result


def plot(
    summary_csv: Path, ik_summary_csv: Path, output_stem: Path, metric: str
) -> None:
    configure_matplotlib()
    series = read_summary(summary_csv, ik_summary_csv, metric)

    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    y_max = 0.0
    for spec in SERIES:
        values = series[spec["case"]]
        x_values = [cache_size for cache_size, _ in values]
        y_values = [tokens_per_s for _, tokens_per_s in values]
        y_max = max(y_max, *y_values)
        ax.plot(
            x_values,
            y_values,
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

    ax.set_xlim(0, 50)
    ax.set_ylim(0, y_max * 1.08)
    ax.set_xticks(range(0, 49, 8))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4))
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
    ax.set_ylabel("Decode throughput (token/s)")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        handlelength=1.5,
        columnspacing=0.65,
        handletextpad=0.35,
        borderaxespad=0.0,
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.45)
    fig.savefig(
        output_stem.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=SCRIPT_DIR / "glm52_cache_size_token_s.csv",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=SCRIPT_DIR / "fig_cache_size_token_s_glm52",
    )
    parser.add_argument(
        "--ik-summary-csv",
        type=Path,
        default=SCRIPT_DIR / "glm52_ik_llama_ud_iq4_xs.csv",
    )
    parser.add_argument(
        "--metric",
        choices=(
            "decode_token_per_s",
            "response_decode_token_per_s",
            "response_decode_phase_token_per_s",
        ),
        default="response_decode_phase_token_per_s",
        help=(
            "Use the exact 499-token decode phase between the first and 500th "
            "completion-token events. The raw response and legacy log metrics are "
            "retained only for audits."
        ),
    )
    args = parser.parse_args()
    plot(args.summary_csv, args.ik_summary_csv, args.output_stem, args.metric)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
