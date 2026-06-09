# Relative Error and Oracle Computation Method

This document describes how the current figure data computes relative error
and how the oracle rows are constructed. It covers the single-expert
error-vs-bits figures under `paper/figure/bit-error/` and the routed top-10
mixed-precision figures under `paper/figure/fine-gain-expert/`.

## 1. Scope

The current measurements are microbenchmarks on Qwen3.5-397B-A17B. They are
weight-representation error measurements, not end-to-end task accuracy
measurements.

For the fine-grained expert tests:

- Model layer: layer 0.
- Hidden states: deterministic synthetic BF16 vectors sampled from `N(0, 1)`.
- Evaluation seeds: `20260602` through `20260621`, 20 hidden states total.
- Router: the layer router is recomputed for every hidden state.
- Routed experts: top-10 experts are selected from the router output.
- Shared path: `shared_expert` and `shared_expert_gate` are excluded.
- Reference: original stored BF16 expert weights.

For the single-expert error-vs-bits tests:

- Model layer: layer 0.
- Expert: expert 0.
- Hidden state: one deterministic synthetic BF16 vector with seed `20260602`.
- Reference: original stored BF16 output of that one expert.

## 2. Expert Output Definition

Each expert has three matrices: `gate_proj`, `up_proj`, and `down_proj`. The
code loads the fused `gate_up_proj`, splits the first half as `gate_proj` and
the second half as `up_proj`, then uses `down_proj`.

For a hidden state `x`, the expert output is:

```text
gate_out = gate_proj(x)
up_out   = up_proj(x)
h        = BF16(SiLU(gate_out) * up_out)
y        = down_proj(h)
```

The matrix products use FP32 accumulation. The gated intermediate activation
`SiLU(gate_out) * up_out` is rounded to BF16 before applying `down_proj`.

The quantized path uses the same formula, but each weight matrix is first
quantized with llama.cpp and then dequantized back to FP32 before the matrix
multiply. The benchmark therefore measures weight-representation error. It
does not execute llama.cpp's runtime matmul kernels end to end.

## 3. Relative Error

The relative error is always computed on the final output vector being tested.
For a BF16-reference output `y_ref` and a quantized output `y_quant`, the
reported value is:

```text
relative_l2_error = ||y_quant - y_ref||_2 / max(||y_ref||_2, 1e-12)
```

The CSV files store this as a fraction. The plotting scripts multiply it by
100 and display it as a percentage.

The same metrics function also records:

- `l2_error`: absolute L2 error.
- `max_absolute_error`: max elementwise absolute error.
- `mean_absolute_error`: mean elementwise absolute error.
- `cosine_similarity`: cosine similarity between `y_ref` and `y_quant`.
- `reference_l2_norm`: `||y_ref||_2`.
- `output_l2_norm`: `||y_quant||_2`.

## 4. Single-Expert Error-vs-Bits Figures

Files:

- `paper/figure/bit-error/qwen35_expert_qk_relative_error.csv`
- `paper/figure/bit-error/qwen35_expert_iq_relative_error.csv`

For Qx_K, one expert is evaluated independently for each available llama.cpp
format:

```text
Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K
```

llama.cpp does not define `Q1_K` or `Q7_K`. The current Qx_K plot starts at
2 bit and still shows `7 bit (none)` as an unavailable intermediate tick. No
error value is fabricated for 7 bit.

For each Qx_K format:

1. Quantize `gate_proj`, `up_proj`, and `down_proj`.
2. Dequantize each matrix.
3. Run the expert forward pass with the dequantized matrices.
4. Compare the output to the original BF16 expert output with the relative L2
   formula above.

For IQ formats, the same output-error formula is used. The IQ runner evaluates
the IQ primitives available in llama.cpp, such as `IQ1_S`, `IQ1_M`, `IQ2_XXS`,
`IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ4_XS`, and `IQ4_NL`.

IQ measurements use importance weights derived from the current activation:

- `hidden_state^2` for `gate_proj` and `up_proj`.
- quantized-path intermediate activation squared for `down_proj`.

This makes IQ results, especially very low-bit IQ results, token-aware
estimates rather than fixed offline expert quantizations.

## 5. Routed Top-10 Relative Error

Files under `paper/figure/fine-gain-expert/` use the router and top-10 experts.

For each hidden state `x`, the router computes logits for all 512 experts:

```text
router_logits = router(x)
routing_probs = softmax(router_logits)
```

The benchmark selects the top-10 experts:

```text
raw_top10_weights, expert_ids = topk(routing_probs, k=10)
routed_weights = raw_top10_weights / sum(raw_top10_weights)
```

The renormalization makes the selected top-10 routed weights sum to one,
matching the Qwen3.5 Transformers router behavior.

The BF16 reference for one hidden state is the weighted sum of the selected
original expert outputs:

```text
y_ref = sum_i routed_weights[i] * expert_bf16_i(x)
```

A mixed-precision strategy `s` assigns one quantization format to each selected
expert. The quantized output is:

```text
y_s = sum_i routed_weights[i] * expert_quant_i,s(x)
```

The reported per-sample error is then:

```text
relative_l2_error_s = ||y_s - y_ref||_2 / max(||y_ref||_2, 1e-12)
```

This means the relative error is measured after summing the weighted top-10
expert outputs, not separately per expert.

## 6. Rank-Based Strategy Assignment

For rank-based strategies, the quantization format is assigned by routed rank.
Rank 1 is the highest routed-weight expert for the current hidden state, rank
10 is the lowest among the selected top-10.

Example:

```text
6/5/4/3/2/2/2/2/2/2
```

means:

- rank 1 uses `Q6_K`
- rank 2 uses `Q5_K`
- rank 3 uses `Q4_K`
- rank 4 uses `Q3_K`
- ranks 5 through 10 use `Q2_K`

The same selected expert IDs and routed weights are used for every strategy
on a given hidden state. Only the quantization format changes.

## 7. Static Expert-Level Strategy

Static expert-level mixed precision is not a rank-based candidate. It assigns
formats by expert ID, independent of the current token's routed rank.

The current static assignment is calibrated with 100 extra hidden states:

- Calibration seeds: `21260602` through `21260701`.
- For each calibration hidden state, the router selects top-10 experts.
- Each expert's empirical top-10 selection count is accumulated.
- Experts are sorted by selection count, with expert ID as the tie-breaker.
- The first 256 experts are the high-precision tier.
- The remaining 256 experts are the low-precision tier.

For the 3-bit static baseline:

```text
high tier: Q4_K
low tier:  Q2_K
```

For the 4-bit static baseline:

```text
high tier: Q6_K
low tier:  Q2_K
```

Static expert-level rows are shown as comparison baselines, but they are not
included in the rank-based oracle candidate set.

## 8. Oracle Computation

The oracle is an offline, retrospective lower envelope over a fixed candidate
set. It is not an online policy.

For each hidden state, the runner first computes `relative_l2_error` for every
candidate strategy in the chosen candidate set. Then it selects the candidate
with the smallest relative L2 error:

```text
winner(x) = argmin_s relative_l2_error_s(x)
```

The oracle row for that hidden state is a copy of the winning candidate row,
with these fields changed:

```text
strategy          = oracle_*
label             = Oracle dynamic
allocation_level  = per_sample_oracle
precision_by_rank = per-sample best
format_by_rank    = per-sample best
selected_strategy = winner strategy name
```

The oracle output value is therefore:

```text
relative_l2_error_oracle(x)
    = min_s relative_l2_error_s(x)
```

The summary CSV reports the mean, standard deviation, median, min, max, and
mean cosine similarity over the 20 evaluation hidden states.

## 9. Current Combined 3-Bit Oracle

The current paper-facing 3-bit figure uses:

- Oracle dynamic from the combined rank-based candidate set.
- Static expert-level `Q4_K/Q2_K` as a baseline.
- Uniform `Q3_K` as a baseline.

The figure data source is:

```text
paper/figure/fine-gain-expert/
  qwen35_rank_precision_avg3_combined_oracle_static_uniform_relative_error.csv
```

The complete candidate data is still stored in:

```text
paper/figure/fine-gain-expert/
  qwen35_rank_precision_avg3_combined_relative_error.csv
  qwen35_rank_precision_avg3_combined_relative_error_samples.csv
```

The combined 3-bit oracle candidate set contains 12 rank-based candidates:

| Candidate | Precision by routed rank 1 to 10 |
|---|---|
| `rank_aware_qk862` | `8/6/2/2/2/2/2/2/2/2` |
| `rank_aware_qk654` | `6/5/4/3/2/2/2/2/2/2` |
| `rank_aware_qk55433` | `5/5/4/3/3/2/2/2/2/2` |
| `rank_aware_qk4443333` | `4/4/4/3/3/3/3/2/2/2` |
| `q4_q2_split` | `4/4/4/4/4/2/2/2/2/2` |
| `rank_aware_qk83333` | `8/3/3/3/3/2/2/2/2/2` |
| `rank_aware_qk6444` | `6/4/4/4/2/2/2/2/2/2` |
| `rank_aware_qk633333` | `6/3/3/3/3/3/3/2/2/2` |
| `rank_aware_qk5553` | `5/5/5/3/2/2/2/2/2/2` |
| `rank_aware_qk544333` | `5/4/4/3/3/3/2/2/2/2` |
| `rank_aware_qk444433` | `4/4/4/4/3/3/2/2/2/2` |
| `uniform_q3` | `3/3/3/3/3/3/3/3/3/3` |

Static expert-level `Q4_K/Q2_K` is shown in the paper-facing bar plot but is
not part of this oracle candidate set.

## 10. Important Interpretation Notes

- The oracle is an upper-bound analysis tool. It assumes the benchmark already
  knows the BF16 reference error for every candidate on the current hidden
  state.
- A real system would need a separate online predictor or policy to choose a
  candidate before seeing the quantization error.
- Uniform baselines are ordinary fixed rank-based strategies and can be oracle
  candidates.
- Static expert-level mixed precision is calibrated once from router
  popularity and then fixed by expert ID. It is intentionally reported
  separately from the oracle.
- `effective_average_bits_per_weight` in top-10 routed summaries is averaged
  over the experts selected by the evaluation hidden states. For static
  expert-level strategies, this routed average can be higher than the global
  full-expert storage budget because popular experts are assigned high
  precision and are selected more often.
- The plotted percentages are `relative_l2_error * 100`.
