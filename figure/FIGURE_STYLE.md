# Figure Style Guide

This file defines the unified visual style for all figures in this paper.
All plotting scripts should follow this guide unless a specific figure explicitly requires otherwise.

The goal is to make all figures visually consistent, colorblind-friendly, printer-friendly, and suitable for conference paper submission.

---

## 1. General Principles

1. Use only the colors defined in this file.
2. The same semantic object must use the same color across all figures.
3. Do not use `rainbow`, `jet`, or other non-uniform colormaps.
4. Do not rely on color alone. Use line styles, markers, hatches, or annotations when needed.
5. Keep figures clean and low-saturation. Avoid overly bright or decorative colors.
6. Background should be white.
7. Grid lines should be light gray and unobtrusive.
8. Axis labels, legends, and tick labels should remain readable after resizing to paper column width.
9. Prefer vector output: PDF/SVG for paper figures.
10. Avoid changing colors locally inside individual plotting scripts unless necessary.

---

## 2. Global Color Palette

Use the following color constants.

```python
PALETTE = {
    # Main method and baselines
    "ours": "#D55E00",
    "ours_light": "#E69F00",
    "strong_baseline": "#0072B2",
    "baseline_1": "#009E73",
    "baseline_2": "#CC79A7",
    "baseline_3": "#56B4E9",
    "baseline_4": "#F0E442",

    # Reference methods
    "oracle": "#000000",
    "upper_bound": "#000000",
    "naive": "#999999",
    "static": "#999999",
    "disabled": "#CCCCCC",

    # System components
    "gpu": "#E45756",
    "cpu": "#4C78A8",
    "dram": "#4C78A8",
    "ssd": "#9D755D",
    "pcie": "#F58518",
    "data_movement": "#F58518",
    "cache": "#54A24B",
    "cache_hit": "#009E73",
    "cache_miss": "#D55E00",

    # Precision / quantization
    "fp16": "#08306B",
    "bf16": "#08519C",
    "fp8": "#2171B5",
    "int8": "#4292C6",
    "int4": "#6BAED6",
    "int2": "#9ECAE1",

    # Neutral colors
    "black": "#000000",
    "dark_gray": "#555555",
    "gray": "#999999",
    "light_gray": "#DDDDDD",
    "very_light_gray": "#F2F2F2",
    "white": "#FFFFFF",
}
```

---

## 3. Semantic Color Mapping

Use semantic colors consistently.

### Main algorithm comparison

| Semantic meaning | Color key | Hex |
|---|---:|---:|
| Our method | `ours` | `#D55E00` |
| Our ablation / variant | `ours_light` | `#E69F00` |
| Strongest baseline | `strong_baseline` | `#0072B2` |
| Other baseline 1 | `baseline_1` | `#009E73` |
| Other baseline 2 | `baseline_2` | `#CC79A7` |
| Other baseline 3 | `baseline_3` | `#56B4E9` |
| Naive / static method | `naive` / `static` | `#999999` |
| Oracle / upper bound | `oracle` / `upper_bound` | `#000000` |

Rules:

- The proposed method should almost always use `ours`.
- If there are multiple variants of our method, the main variant uses `ours`; secondary variants use `ours_light` or the same color with different line styles.
- Oracle / upper bound should use black dashed lines.
- Naive / static baselines should use gray.

---

## 4. Hardware / System Component Colors

Use these mappings when visualizing system architecture, memory hierarchy, data movement, or execution breakdown.

| Component | Color key | Hex |
|---|---:|---:|
| GPU / HBM | `gpu` | `#E45756` |
| CPU / DRAM | `cpu` / `dram` | `#4C78A8` |
| SSD / NVMe | `ssd` | `#9D755D` |
| PCIe / data movement | `pcie` / `data_movement` | `#F58518` |
| Cache | `cache` | `#54A24B` |
| Cache hit | `cache_hit` | `#009E73` |
| Cache miss | `cache_miss` | `#D55E00` |

Rules:

- Use the same hardware colors in all pipeline diagrams, stacked bars, and breakdown figures.
- Do not recolor GPU/CPU/SSD differently across figures.
- Data movement should be orange, not red, to distinguish it from GPU computation.

---

## 5. Precision / Bit-width Colors

For precision-related figures, use a sequential blue palette.

```python
BITWIDTH_COLORS = {
    1: "#EFF3FF",
    2: "#C6DBEF",
    3: "#9ECAE1",
    4: "#6BAED6",
    5: "#4292C6",
    6: "#2171B5",
    7: "#08519C",
    8: "#08306B",
}
```

Rules:

- Higher bit-width should use darker color.
- Lower bit-width should use lighter color.
- Do not use categorical colors for bit-width unless bit-width is not the main semantic dimension.
- For line plots with many bit-widths, use both color and marker.

Recommended mapping:

| Precision | Color |
|---|---:|
| 8-bit | `#08306B` |
| 6-bit | `#2171B5` |
| 4-bit | `#6BAED6` |
| 2-bit | `#9ECAE1` |
| 1-bit | `#EFF3FF` |

---

## 6. Expert Rank / Routed Weight Distribution

For figures showing expert routed weight, expert rank, routing skewness, or importance distribution:

Recommended default:

```python
EXPERT_DIST_STYLE = {
    "violin_fill": "#9ECAE1",
    "violin_edge": "#08306B",
    "box_edge": "#08306B",
    "median": "#D55E00",
    "mean": "#E69F00",
    "outlier": "#555555",
}
```

Rules:

- Do not assign a completely different categorical color to every expert rank.
- Prefer one consistent blue family for distributions.
- Use orange/red only for key statistics such as median, mean, or highlighted skewness.
- For rank 1 to rank 10, it is acceptable to use the bit-width-like sequential blue palette, but the default should be a single-color violin/box style.

---

## 7. Line Styles and Markers

When multiple methods appear in the same line plot, use both color and line style.

Recommended line styles:

```python
LINE_STYLES = {
    "ours": "-",
    "ours_variant": "--",
    "strong_baseline": "-.",
    "baseline": ":",
    "oracle": "--",
    "naive": ":",
}
```

Recommended markers:

```python
MARKERS = {
    "ours": "o",
    "ours_variant": "s",
    "strong_baseline": "^",
    "baseline_1": "D",
    "baseline_2": "v",
    "baseline_3": "P",
    "oracle": "*",
    "naive": "x",
}
```

Rules:

- Our method should use a solid line unless the plot contains multiple variants of our method.
- Oracle should use black dashed line.
- Naive/static methods should use gray dotted line.
- Use markers for small figures where lines may be hard to distinguish.

### PagedMoE precision/cache line plots

The following visual language is mandatory for PagedMoE throughput, cache-size,
cache-hit-rate, independent-precision, and mmap-backend line plots. It overrides the
generic method-oriented line and marker mappings above.

Color and marker encode the precision policy:

| Precision policy | Color | Marker |
|---|---:|---:|
| External mmap Q4_0 baseline | `#000000` | `^` triangle |
| Uniform 4-bit | `#0072B2` | `s` square |
| Dynamic Avg 4-bit | `#009E73` | `D` diamond |
| Dynamic Avg 2.5-bit | `#D55E00` | `o` circle |

Line style encodes the execution or cache path:

| Execution path | Line style |
|---|---:|
| Complete PagedMoE | `-` solid |
| Without prefix reuse | `--` dashed |
| mmap backend | `:` dotted |
| External mmap Q4_0 baseline | `-.` dash-dot |

Required combined styles:

| Series | Required style |
|---|---|
| mmap Q4_0 | black `-.` with hollow `^` |
| Uniform 4-bit | deep blue `-` with hollow `s` |
| Dynamic Avg 4-bit | teal `-` with hollow `D` |
| Dynamic Avg 4-bit w/o Prefix Reuse | teal `--` with hollow `D` |
| Dynamic Avg 2.5-bit | orange `-` with hollow `o` |
| Dynamic Avg 2.5-bit w/o Prefix Reuse | orange `--` with hollow `o` |
| Uniform 4-bit on mmap backend | deep blue `:` with hollow `s` |
| Dynamic Avg 2.5-bit on mmap backend | orange `:` with hollow `o` |

Rules:

- Keep a policy's color and marker unchanged across ablations. Only the line
  style changes with the execution or cache path.
- All markers must be hollow: use `markerfacecolor="#FFFFFF"` and set
  `markeredgecolor` to the curve color.
- Reserve black dash-dot triangles for the external mmap Q4_0 baseline. The
  internal mmap-backend comparison keeps its policy color and marker.
- Do not use an `x` marker for w/o-prefix curves; use the base policy marker
  with a dashed line.

Reference constants:

```python
PAGEDMOE_POLICY_STYLE = {
    "mmap_q4_0": {"color": "#000000", "marker": "^"},
    "uniform_4bit": {"color": "#0072B2", "marker": "s"},
    "dynamic_avg4": {"color": "#009E73", "marker": "D"},
    "dynamic_avg2p5": {"color": "#D55E00", "marker": "o"},
}

PAGEDMOE_PATH_LINESTYLE = {
    "complete": "-",
    "without_prefix_reuse": "--",
    "mmap_backend": ":",
    "external_mmap_q4_0": "-.",
}

HOLLOW_MARKER_STYLE = {
    "markerfacecolor": "#FFFFFF",
    "markeredgewidth": 0.75,
}
```

---

## 8. Bar Charts

For bar charts:

1. Use semantic colors rather than arbitrary categorical colors.
2. Use hatches when the figure may be printed in grayscale.
3. Keep bar edge color as dark gray or black.
4. Avoid very thin bars.

Recommended hatches:

```python
HATCHES = {
    "ours": "",
    "strong_baseline": "//",
    "baseline_1": "\\\\",
    "baseline_2": "xx",
    "baseline_3": "..",
    "naive": "--",
}
```

Recommended bar style:

```python
bar_kwargs = {
    "edgecolor": "#333333",
    "linewidth": 0.8,
}
```

---

## 9. Breakdown / Stacked Bar Figures

For latency, memory, bandwidth, or execution-time breakdowns, use component colors.

Recommended mapping:

```python
BREAKDOWN_COLORS = {
    "GPU compute": "#E45756",
    "CPU compute": "#4C78A8",
    "SSD read": "#9D755D",
    "PCIe transfer": "#F58518",
    "Cache lookup": "#54A24B",
    "Other": "#BAB0AC",
}
```

Rules:

- Stack order should generally follow the execution path:
  1. SSD read
  2. CPU compute
  3. PCIe transfer
  4. GPU compute
  5. Other
- Keep the same stack order across all figures.
- Do not sort stack components differently across figures unless explicitly justified.

---

## 10. Matplotlib Default Style

All plotting scripts should import and call a shared style function if available.

For two-column papers where each plot is a single-column figure and figures are
often placed two per row, use a larger text setting so labels remain readable
after LaTeX scaling:

```python
def apply_single_column_large_text_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Linux Libertine O", "Libertinus Serif", "STIXGeneral"],
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,

        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,

        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
```

Use this setting for dense single-column plots such as cache-hit curves, latency
curves, and small ablation plots. For plots with long tick labels, rotate or
thin ticks before reducing font size.

Recommended default:

```python
import matplotlib.pyplot as plt

def apply_paper_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,

        "axes.linewidth": 0.8,
        "lines.linewidth": 1.8,
        "lines.markersize": 4.5,

        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,

        "axes.grid": True,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.7,

        "axes.facecolor": "#FFFFFF",
        "figure.facecolor": "#FFFFFF",

        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
```

Rules:

- Do not use seaborn styles unless explicitly required.
- Do not use dark background.
- Do not use thick grid lines.
- Save paper figures as `.pdf` whenever possible.

---

## 11. Figure Size

Use consistent sizes.

Recommended sizes:

```python
FIG_SIZE = {
    "single_column": (3.3, 2.2),
    "single_column_tall": (3.3, 2.6),
    "double_column": (6.8, 2.4),
    "double_column_tall": (6.8, 3.2),
}
```

Rules:

- Single-column figures should usually be around 3.3 inches wide.
- Double-column figures should usually be around 6.8 inches wide.
- Avoid excessive vertical whitespace.
- Legends should not dominate the figure.

---

## 12. Export Rules

All scripts should save figures as:

```text
.pdf
```

Optional additional exports:

```text
.png
.svg
```

Recommended:

```python
plt.savefig("figure_name.pdf")
plt.savefig("figure_name.png", dpi=300)
```

Rules:

- PDF is required for paper submission.
- PNG is useful for quick preview.
- Use transparent background only when needed for slides, not for paper figures.
- Ensure all text remains selectable in PDF output when possible.

---

## 13. Naming Rules

Use clear and stable output filenames.

Examples:

```text
fig_motivation_expert_skew.pdf
fig_latency_breakdown.pdf
fig_cache_hit_rate.pdf
fig_precision_tradeoff.pdf
fig_end_to_end_speedup.pdf
fig_ablation_prefetch.pdf
fig_ablation_precision.pdf
```

Rules:

- Use lowercase filenames.
- Use underscores instead of spaces.
- Prefix paper figures with `fig_`.
- Keep filenames stable so LaTeX references do not break.

---

## 14. Special Rules for This Paper

This paper studies local MoE inference with SSD-resident experts, expert caching, prefetching, and adaptive precision.

Therefore, the following semantic mappings are mandatory:

| Concept | Required color |
|---|---:|
| Proposed system | `#D55E00` |
| Best baseline | `#0072B2` |
| SSD-resident expert loading | `#9D755D` |
| CPU-side execution / DRAM | `#4C78A8` |
| GPU-side execution / HBM | `#E45756` |
| PCIe transfer / data movement | `#F58518` |
| Expert cache hit | `#009E73` |
| Expert cache miss | `#D55E00` |
| Static precision | `#999999` |
| Adaptive precision | `#D55E00` |
| Oracle expert cache / upper bound | `#000000` |

---

## 15. Do Not Do

Do not use:

```python
cmap="jet"
cmap="rainbow"
plt.style.use("dark_background")
plt.style.use("ggplot")
plt.style.use("seaborn")
```

Do not:

- assign random colors based on plotting order;
- use red/green as the only distinction;
- use more than 6 unrelated categorical colors in one figure;
- change the color of "ours" across figures;
- use saturated pure red `#FF0000` or pure green `#00FF00`;
- use low-contrast yellow on white background for important curves;
- make legends larger than the plotted data.

---

## 16. Minimal Example

```python
import matplotlib.pyplot as plt

from plot_style import apply_paper_style, PALETTE

apply_paper_style()

x = [1, 2, 4, 8]
ours = [1.0, 1.8, 2.7, 3.2]
baseline = [1.0, 1.3, 1.6, 1.7]

plt.figure(figsize=(3.3, 2.2))
plt.plot(
    x,
    ours,
    label="Ours",
    color=PALETTE["ours"],
    marker="o",
    linewidth=1.8,
)
plt.plot(
    x,
    baseline,
    label="Baseline",
    color=PALETTE["strong_baseline"],
    marker="^",
    linewidth=1.8,
)

plt.xlabel("Batch size")
plt.ylabel("Speedup")
plt.legend()
plt.savefig("fig_example.pdf")
plt.savefig("fig_example.png", dpi=300)
```

---

## 17. Optional: Shared `plot_style.py`

For stronger consistency, place the following file in the project root as `plot_style.py` and import it from all plotting scripts.

```python
import matplotlib.pyplot as plt

PALETTE = {
    "ours": "#D55E00",
    "ours_light": "#E69F00",
    "strong_baseline": "#0072B2",
    "baseline_1": "#009E73",
    "baseline_2": "#CC79A7",
    "baseline_3": "#56B4E9",
    "baseline_4": "#F0E442",

    "oracle": "#000000",
    "upper_bound": "#000000",
    "naive": "#999999",
    "static": "#999999",
    "disabled": "#CCCCCC",

    "gpu": "#E45756",
    "cpu": "#4C78A8",
    "dram": "#4C78A8",
    "ssd": "#9D755D",
    "pcie": "#F58518",
    "data_movement": "#F58518",
    "cache": "#54A24B",
    "cache_hit": "#009E73",
    "cache_miss": "#D55E00",

    "fp16": "#08306B",
    "bf16": "#08519C",
    "fp8": "#2171B5",
    "int8": "#4292C6",
    "int4": "#6BAED6",
    "int2": "#9ECAE1",

    "black": "#000000",
    "dark_gray": "#555555",
    "gray": "#999999",
    "light_gray": "#DDDDDD",
    "very_light_gray": "#F2F2F2",
    "white": "#FFFFFF",
}

BITWIDTH_COLORS = {
    1: "#EFF3FF",
    2: "#C6DBEF",
    3: "#9ECAE1",
    4: "#6BAED6",
    5: "#4292C6",
    6: "#2171B5",
    7: "#08519C",
    8: "#08306B",
}

BREAKDOWN_COLORS = {
    "GPU compute": "#E45756",
    "CPU compute": "#4C78A8",
    "SSD read": "#9D755D",
    "PCIe transfer": "#F58518",
    "Cache lookup": "#54A24B",
    "Other": "#BAB0AC",
}

EXPERT_DIST_STYLE = {
    "violin_fill": "#9ECAE1",
    "violin_edge": "#08306B",
    "box_edge": "#08306B",
    "median": "#D55E00",
    "mean": "#E69F00",
    "outlier": "#555555",
}

LINE_STYLES = {
    "ours": "-",
    "ours_variant": "--",
    "strong_baseline": "-.",
    "baseline": ":",
    "oracle": "--",
    "naive": ":",
}

MARKERS = {
    "ours": "o",
    "ours_variant": "s",
    "strong_baseline": "^",
    "baseline_1": "D",
    "baseline_2": "v",
    "baseline_3": "P",
    "oracle": "*",
    "naive": "x",
}

# Mandatory for PagedMoE precision/cache line plots.
PAGEDMOE_POLICY_STYLE = {
    "mmap_q4_0": {"color": "#000000", "marker": "^"},
    "uniform_4bit": {"color": "#0072B2", "marker": "s"},
    "dynamic_avg4": {"color": "#009E73", "marker": "D"},
    "dynamic_avg2p5": {"color": "#D55E00", "marker": "o"},
}

PAGEDMOE_PATH_LINESTYLE = {
    "complete": "-",
    "without_prefix_reuse": "--",
    "mmap_backend": ":",
    "external_mmap_q4_0": "-.",
}

HOLLOW_MARKER_STYLE = {
    "markerfacecolor": "#FFFFFF",
    "markeredgewidth": 0.75,
}

HATCHES = {
    "ours": "",
    "strong_baseline": "//",
    "baseline_1": "\\\\",
    "baseline_2": "xx",
    "baseline_3": "..",
    "naive": "--",
}

FIG_SIZE = {
    "single_column": (3.3, 2.2),
    "single_column_tall": (3.3, 2.6),
    "double_column": (6.8, 2.4),
    "double_column_tall": (6.8, 3.2),
}


def apply_paper_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,

        "axes.linewidth": 0.8,
        "lines.linewidth": 1.8,
        "lines.markersize": 4.5,

        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,

        "axes.grid": True,
        "grid.color": PALETTE["light_gray"],
        "grid.linewidth": 0.5,
        "grid.alpha": 0.7,

        "axes.facecolor": PALETTE["white"],
        "figure.facecolor": PALETTE["white"],

        "legend.frameon": False,

        # Keep text editable/selectable in vector outputs.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_figure(path_without_suffix):
    plt.savefig(f"{path_without_suffix}.pdf")
    plt.savefig(f"{path_without_suffix}.png", dpi=300)
```

---

## 18. Prompt for Codex / Agent

When asking Codex or another coding agent to draw figures, include:

```text
Please follow FIGURE_STYLE.md and import colors/styles from plot_style.py.
Do not introduce new colors unless necessary.
Use semantic colors consistently across all figures.
For PagedMoE precision/cache line plots, color and marker encode precision
policy, line style encodes execution path, and every marker is hollow.
```
