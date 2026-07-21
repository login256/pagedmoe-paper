# INT4 Trace Pipeline Motivation

This experiment connects a real 500-output-token Qwen3.5-397B route trace to a
static 4-bit cache simulation and the measured single-reader SSD/CPU pipeline.

## Trace run

- Model: Qwen3.5-397B-A17B, all 60 MoE layers, top-10 routing.
- Prompt: MMLU abstract algebra question used by the existing benchmarks.
- Output: 500 tokens with EOS ignored, seed 1926817.
- Precision: static 4 bits for every selected expert.
- Runtime cache: 48 GiB; compute threads: 16; bitplane readers: 8.
- Request wall time: 83.7785 s; request throughput: 5.9681 token/s.
- Runtime bitplane hit rate (including request prefill): 76.29%.

The first generated token is sampled from the final prefill position. Its 60
routes are reconstructed by taking the last position of each layer-major
prefill group. The remaining 499 tokens appear as 499 complete layer 0..59
cycles. The extracted trace therefore contains exactly 30,000 layer records
and 300,000 top-10 expert selections.

## Cache model

The simulator uses the runtime hit-rate unit: `(layer, expert, bitplane)`.
Every 4-bit route accesses four 1.5 MiB planes, and a global exact LRU is shared
by all layers. Codebooks are not charged to bitplane cache capacity, matching
the runtime statistic `read_job_cache_hit_rate`.

At 48 GiB the trace simulation gives 75.23%, close to the real runtime's
76.29%. The small difference is expected because the real statistic also
contains prefill and health-check accesses and the runtime has reservation and
in-flight-read behavior that a serial LRU does not model.

## Pipeline model

For every traced layer, the simulator counts experts with at least one missed
plane. It interpolates the measured 0/2/4/6/8/10-miss results in
`../int4-pipeline-read-compute/qwen35_397b_int4_pipeline_read_compute_summary.csv`
and accumulates wall time, exposed SSD time, read/compute overlap, tail compute,
and total read time.

The resulting `moe_pipeline_token_s` is a motivation-model estimate for the 60
MoE layers only. It is calibrated from the single-reader `O_DIRECT` experiment
and must not be compared as an end-to-end prediction against the real server,
which uses eight bitplane readers and also has attention/runtime overhead.

## Outputs

- `decode500_server.log`: complete real server and precision trace log.
- `decode500_expert_trace.csv`: normalized 500-token expert trace.
- `simulation_summary.csv`: cache sweep and aggregate pipeline metrics.
- `simulation_per_token.csv`: per-token hit rate and timing estimates.
- `int4_trace_pipeline_motivation.{png,pdf}`: summary figure.
- `metadata.json`: dimensions and simulation assumptions.

Re-run the deterministic trace simulation with:

```bash
python paper/figure/int4-trace-pipeline-motivation/run_trace_simulation.py
```
