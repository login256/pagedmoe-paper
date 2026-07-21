#!/usr/bin/env python3
"""Compatibility entry point for the end-to-end cache-size figure."""

from __future__ import annotations

import argparse
from pathlib import Path

from plot_cache_size_ablation_figures import (
    SCRIPT_DIR,
    configure_matplotlib,
    load_all_series,
    plot_figure,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=SCRIPT_DIR / "fig_cache_size_token_s",
        help="output figure stem without extension",
    )
    args = parser.parse_args()

    configure_matplotlib()
    plot_figure(
        "end-to-end",
        load_all_series(),
        args.output_stem.parent,
        output_names=(args.output_stem.name,),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
