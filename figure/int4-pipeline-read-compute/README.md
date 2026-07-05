# Qwen3.5-397B Expert Read/Compute Pipeline Experiment

## Purpose

This experiment measures how SSD expert loading overlaps with CPU expert compute for one Qwen3.5-397B MoE layer.  The plotted metric is **Relative expert throughput**, not hardware MFU.

```text
Relative expert throughput = compute-only time for 10 experts / pipeline wall time
```

Since every run computes the same 10 experts, this is equivalent to the expert throughput of the SSD pipeline relative to a compute-only all-hit baseline.  It is not `actual FLOPs / hardware peak FLOPs`.

## Setup

- Model: `/mnt/data/Qwen3.5-397B-A17B`
- Layer: `43`
- Experts: `0..9`
- Input tokens: `1`
- CPU binding: NUMA node 1
- PyTorch threads: `16`
- GPU disabled via `CUDA_VISIBLE_DEVICES=""`
- Repeats: `10`
- Warmup: `2`
- Hit rates plotted: `0%, 20%, 40%, 60%, 80%`

The benchmark script is:

```text
paper/figure/int4-pipeline-read-compute/qwen35_397b_int4_pipeline_read_compute.py
```

The plot script is:

```text
paper/figure/int4-pipeline-read-compute/plot_qwen35_397b_int4_pipeline_read_compute.py
```

## Storage Formats

The current final experiments use contiguous per-expert pack files opened once and read with `O_DIRECT`.

### BF16

Pack file:

```text
/mnt/test/qwen35_397b_10expert_int_pack/qwen35_397b_layer_043_experts_000_009_bf16.bin
```

Format:

- Raw BF16 weights.
- Per expert: `gate`, `up`, `down`.
- Each tensor shape is 4M elements.
- Bytes per expert: `25,165,824`.
- 10 experts total: `251,658,240` bytes.

Compute path:

```text
PyTorch F.linear, BF16 input, BF16 weight
```

On this machine, oneDNN verbose showed this dispatching to AMX BF16 matmul:

```text
brg_matmul:avx10_1_512_amx
```

### W8A16

Pack file:

```text
/mnt/test/qwen35_397b_10expert_int_pack/qwen35_397b_layer_043_experts_000_009_w8a16.bin
```

Format:

- torchao `Int8WeightOnlyConfig` payload.
- Per linear: int8 data, BF16 scale, int64 zero point.
- Per expert: `gate`, `up`, `down`.
- Bytes per expert: `12,644,352`.
- 10 experts total: `126,443,520` bytes.

Compute path:

```text
torchao quantized nn.Linear, BF16 activation, int8 weight payload
```

The script does not Python-side dequantize the full weight back to BF16 after quantization.  torchao may still unpack or transform internally inside its own kernel implementation.

## Pipeline

For each hit rate, the first `hit_experts` are treated as resident and computed immediately.  The remaining experts are treated as SSD misses.

Miss experts are read one expert at a time from the pre-opened contiguous pack file:

```text
read expert i -> compute expert i -> consume next read
```

The loader starts before hit compute begins, so even `0%` hit rate can overlap the read of the next expert with compute of the current loaded expert.

The reader uses:

- `os.open(..., O_DIRECT)`
- `os.preadv`
- a page-aligned anonymous `mmap` buffer for direct I/O alignment

No system page cache is intentionally used for measured reads.

## Timing Definitions

Each bar in the figure is normalized to pipeline wall time, so the stacked bar sums to 100%.

- **Wait for SSD**: read time that is not overlapped with compute.
- **Read/compute overlap**: intersection of read intervals and compute intervals.
- **Tail compute**: remaining wall time after exposed read and overlap.
- **Relative expert throughput**: compute-only baseline divided by pipeline wall time.

The right-axis throughput line is shown as a relative throughput percentage.  The 100% hit-rate endpoint is not plotted, because by definition it is just the normalization point.

## Current Results

### BF16

Compute-only median for 10 experts:

```text
4.382 ms
```

Effective compute-side weight bandwidth:

```text
251,658,240 bytes / 4.381542 ms = 57.43 GB/s
```

| hit rate | wall ms | compute ms | read ms | read BW | relative expert throughput |
|---:|---:|---:|---:|---:|---:|
| 0% | 20.799 | 5.352 | 20.181 | 12.47 GB/s | 21.1% |
| 20% | 16.434 | 5.331 | 15.848 | 12.70 GB/s | 26.7% |
| 40% | 12.473 | 5.306 | 11.926 | 12.66 GB/s | 35.1% |
| 60% | 8.536 | 5.456 | 8.009 | 12.57 GB/s | 51.3% |
| 80% | 5.297 | 5.255 | 4.053 | 12.42 GB/s | 82.7% |

### W8A16

Compute-only median for 10 experts:

```text
5.489 ms
```

Effective compute-side weight-payload bandwidth:

```text
126,443,520 bytes / 5.489254 ms = 23.03 GB/s
```

| hit rate | wall ms | compute ms | read ms | read BW | relative expert throughput |
|---:|---:|---:|---:|---:|---:|
| 0% | 11.265 | 5.827 | 10.521 | 12.02 GB/s | 48.7% |
| 20% | 9.127 | 5.715 | 8.387 | 12.06 GB/s | 60.1% |
| 40% | 7.135 | 5.715 | 6.318 | 12.01 GB/s | 76.9% |
| 60% | 6.158 | 5.671 | 4.260 | 11.87 GB/s | 89.1% |
| 80% | 5.652 | 5.612 | 2.269 | 11.15 GB/s | 97.1% |

## Outputs

BF16:

```text
qwen35_397b_bf16_pipeline_read_compute_summary.csv
qwen35_397b_bf16_pipeline_read_compute_detail.csv
qwen35_397b_bf16_pipeline_read_compute_summary.json
qwen35_397b_bf16_pipeline_read_compute.{pdf,png,svg}
```

W8A16:

```text
qwen35_397b_w8a16_pipeline_read_compute_summary.csv
qwen35_397b_w8a16_pipeline_read_compute_detail.csv
qwen35_397b_w8a16_pipeline_read_compute_summary.json
qwen35_397b_w8a16_pipeline_read_compute.{pdf,png,svg}
```

Older INT4/INT8 output files remain in this directory from earlier exploratory runs, but the current comparison of interest is BF16 raw-weight storage versus W8A16 torchao int8-weight storage.

## Reproduce

Run BF16:

```bash
numactl --cpunodebind=1 --membind=1 \
  /home/lyt/anaconda3/envs/smoe/bin/python \
  paper/figure/int4-pipeline-read-compute/qwen35_397b_int4_pipeline_read_compute.py \
  --compute-backend bf16 \
  --repeats 10 \
  --baseline-repeats 10 \
  --warmup 2 \
  --loader-workers 1
```

Run W8A16:

```bash
numactl --cpunodebind=1 --membind=1 \
  /home/lyt/anaconda3/envs/smoe/bin/python \
  paper/figure/int4-pipeline-read-compute/qwen35_397b_int4_pipeline_read_compute.py \
  --compute-backend torchao-w8a16 \
  --repeats 10 \
  --baseline-repeats 10 \
  --warmup 2 \
  --loader-workers 1
```

Regenerate a figure from an existing summary CSV:

```bash
python paper/figure/int4-pipeline-read-compute/plot_qwen35_397b_int4_pipeline_read_compute.py \
  --summary-csv paper/figure/int4-pipeline-read-compute/qwen35_397b_w8a16_pipeline_read_compute_summary.csv \
  --output-stem paper/figure/int4-pipeline-read-compute/qwen35_397b_w8a16_pipeline_read_compute
```
