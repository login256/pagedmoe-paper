# Qwen3.5 Fixed4 Absolute Time Breakdown

This figure uses the fixed-4-bit whole-expert Qwen3.5-397B-A17B cache sweep from
run `qwen35_fixed4_whole_expert_activity_decode500_rerun_20260721_161635`.

The full run contains 499 runtime decode tokens. The plotted absolute wall times
are measured over decode model tokens 120 through 129, with token 130 providing
the right boundary. Every stacked bar is measured in `ms/token` and contains four
mutually exclusive worker-activity buckets:

- `Unhidden SSD I/O`: read tasks active without an expert compute task.
- `Overlapped I/O and computation`: read and expert compute tasks active
  concurrently.
- `Expert computation only`: expert compute tasks active without a read task.
- `Other model and runtime work`: neither a read task nor an expert compute task
  active.

The components sum to the token-cycle wall time. These are host worker-activity
categories, not pure device service time or a strict model MoE/non-MoE split.

The source CSV also records `full_epoch_hit_rate` from all 499 runtime decode
tokens and `sample_hit_rate` from the same 10-token window as the bars. Hit rate
is retained as data rather than plotted on the absolute-time axis. Raw summary,
logs, and traces are under:

```text
src/target/bench/qwen35_fixed4_whole_expert_activity_decode500_rerun_20260721_161635/
```

Regenerate all figure formats with:

```bash
python paper/figure/breakdown-cache-size-ms-token/plot_qwen35_cache_size_absolute_time_breakdown.py
```

Outputs:

```text
fig_qwen35_cache_size_absolute_time_breakdown.pdf
fig_qwen35_cache_size_absolute_time_breakdown.png
fig_qwen35_cache_size_absolute_time_breakdown.svg
```
