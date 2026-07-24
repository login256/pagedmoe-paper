# Qwen3.5 Fixed4 Absolute Time Breakdown

This figure uses the fixed-4-bit whole-expert Qwen3.5-397B-A17B cache sweep from
run `qwen35_fixed4_whole_expert_activity_runtime500_20260721_171957`.

Each plotted window contains runtime decode tokens 0 through 499. Runtime token
500 provides the layer-0 right boundary, and untraced token 501 flushes each
worker's buffered trace. The boundary and flush tokens are excluded from every
reported metric. Every stacked bar is measured in `ms/token` over the same 500
tokens and contains four mutually exclusive worker-activity buckets:

- `Unhidden SSD I/O`: read tasks active without an expert compute task.
- `Overlapped I/O and computation`: read and expert compute tasks active
  concurrently.
- `Expert computation only`: expert compute tasks active without a read task.
- `Other model and runtime work`: neither a read task nor an expert compute task
  active.

The components sum to the token-cycle wall time. These are host worker-activity
categories, not pure device service time or a strict model MoE/non-MoE split.

The source CSV also records `window_tps` and `window_hit_rate` from exactly the
same 500-token window as the bars. Thus `total_ms_per_token = 1000 / window_tps`.
Hit rate is available both as a separate figure and on the right axis of an
optional combined figure. Raw summary, logs, and traces are under:

```text
src/target/bench/qwen35_fixed4_whole_expert_activity_runtime500_20260721_171957/
```

Regenerate all figure formats with:

```bash
python paper/figure/breakdown-cache-size-ms-token/plot_qwen35_cache_size_absolute_time_breakdown.py
```

Generate the compact 4-inch version containing only the 8, 16, 24, 32, 40,
and 48 GiB points with:

```bash
python paper/figure/breakdown-cache-size-ms-token/plot_qwen35_cache_size_absolute_time_breakdown.py --compact
```

Generate the separate compact hit-rate figure for the same six cache capacities
with:

```bash
python paper/figure/breakdown-cache-size-ms-token/plot_qwen35_cache_size_hit_rate.py
```

Generate a combined dual-axis figure with the timing breakdown on the left axis
and hit rate on the right axis with:

```bash
python paper/figure/breakdown-cache-size-ms-token/plot_qwen35_cache_size_time_and_hit_rate.py
```

Outputs:

```text
fig_qwen35_cache_size_absolute_time_breakdown.pdf
fig_qwen35_cache_size_absolute_time_breakdown.png
fig_qwen35_cache_size_absolute_time_breakdown.svg
fig_qwen35_cache_size_absolute_time_breakdown_compact.pdf
fig_qwen35_cache_size_absolute_time_breakdown_compact.png
fig_qwen35_cache_size_absolute_time_breakdown_compact.svg
fig_qwen35_cache_size_hit_rate_compact.pdf
fig_qwen35_cache_size_hit_rate_compact.png
fig_qwen35_cache_size_hit_rate_compact.svg
fig_qwen35_cache_size_time_and_hit_rate_compact.pdf
fig_qwen35_cache_size_time_and_hit_rate_compact.png
fig_qwen35_cache_size_time_and_hit_rate_compact.svg
```
