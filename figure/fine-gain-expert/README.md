# Rank-Based Expert Precision Allocation Error

This microbenchmark compares three rank-based llama.cpp mixed-precision
allocations for routed MoE tokens from Qwen3.5-397B-A17B. It uses layer 0,
deterministic synthetic BF16 hidden states sampled from `N(0, 1)`, and the
layer router to select the top-10 experts. By default it evaluates 20 hidden
states with seeds `20260602` through `20260621`.

The router weights are computed by applying FP32 softmax over all 512 router
logits and then selecting the top-10 values. The selected routed weights are
renormalized to sum to one, matching the Qwen3.5 Transformers router. The
reference output is the routed-weighted sum of the top-10 original BF16-weight
expert outputs. Each quantized output uses the same selected experts and routed
weights.

This is a routed-branch-only weight-representation microbenchmark. It
intentionally excludes `shared_expert` and `shared_expert_gate`, so it is not a
full MoE-block output-error measurement. The stored BF16 weights and the
dequantized weights are multiplied with FP32 accumulation, with the gated
intermediate activation rounded to BF16 before `down_proj`. The quantized path
does not execute llama.cpp's runtime matrix-multiplication kernels.

All three strategies have a nominal average precision of 3 bits per selected
expert. Slots marked as 1 bit use llama.cpp `IQ1_M`; all other slots use the
matching llama.cpp `Qx_K` format:

`IQ1_M` is measured with importance weights derived from the current hidden
state for `gate_proj` and `up_proj`, and from the current quantized-path
intermediate activation for `down_proj`. Those IQ1_M measurements are
token-aware optimistic estimates, not fixed offline expert quantizations.

| Strategy | Nominal precision by routed-expert rank 1 to 10 |
|---|---|
| Rank-aware | `8/6/4/3/2/2/2/1/1/1` |
| Q4/Q2 split | `4/4/4/4/4/2/2/2/2/2` |
| Uniform Q3 | `3/3/3/3/3/3/3/3/3/3` |

The script loads one router and only one selected expert at a time. The main
CSV reports cross-seed summary statistics. The `_samples.csv` file preserves
the individual measurements. The figure reports the mean relative L2 output
error, with sample standard deviation shown as error bars.

```bash
python paper/figure/fine-gain-expert/run_qwen35_rank_precision_comparison.py
python paper/figure/fine-gain-expert/plot_qwen35_rank_precision_relative_error.py
```

The separate pure-Qx_K comparison uses:

| Strategy | Nominal precision by routed-expert rank 1 to 10 |
|---|---|
| Rank-aware | `6/5/4/3/2/2/2/2/2/2` |
| Q4/Q2 split | `4/4/4/4/4/2/2/2/2/2` |
| Uniform Q3 | `3/3/3/3/3/3/3/3/3/3` |

```bash
python paper/figure/fine-gain-expert/run_qwen35_rank_precision_comparison.py \
  --strategy-set qk654
python paper/figure/fine-gain-expert/plot_qwen35_rank_precision_relative_error.py \
  --csv paper/figure/fine-gain-expert/qwen35_rank_precision_qk654_relative_error.csv \
  --output-stem paper/figure/fine-gain-expert/qwen35_rank_precision_qk654_relative_error
```

The five-way comparison combines both sets and adds `5/5/4/3/3/2/2/2/2/2`:

```bash
python paper/figure/fine-gain-expert/run_qwen35_rank_precision_comparison.py \
  --strategy-set fiveway \
  --iq-workers 56
python paper/figure/fine-gain-expert/plot_qwen35_rank_precision_relative_error.py \
  --csv paper/figure/fine-gain-expert/qwen35_rank_precision_fiveway_relative_error.csv \
  --output-stem paper/figure/fine-gain-expert/qwen35_rank_precision_fiveway_relative_error
```

The oracle-dynamic comparison evaluates six pure-Qx_K candidates for every
hidden state and retrospectively selects the candidate with the lowest
relative L2 output error. This is an upper-bound microbenchmark, not an online
selection policy. The figure also includes a static expert-level mixed
precision baseline. It samples 100 separate hidden states, ranks all 512
experts by empirical top-10 selection probability, permanently assigns the
first half to `Q4_K`, and permanently assigns the second half to `Q2_K`.
Ties are resolved by ascending expert ID. The resulting 3-bit assignment is
written to `qwen35_static_expert_precision_assignment.csv` and
`qwen35_static_expert_precision_assignment.json`:

The static assignment file distinguishes the fixed full-expert storage budget
from the average encoded precision of the experts actually routed by the
evaluation tokens. The former is `3.5625` effective bits per weight for the
`Q4_K/Q2_K` split and `4.59375` effective bits per weight for the `Q6_K/Q2_K`
split after llama.cpp block overhead. The latter can be higher because the
popular high-precision experts are selected more frequently.

```bash
python paper/figure/fine-gain-expert/run_qwen35_rank_precision_comparison.py \
  --strategy-set oracle_dynamic
python paper/figure/fine-gain-expert/plot_qwen35_rank_precision_relative_error.py \
  --csv paper/figure/fine-gain-expert/qwen35_rank_precision_oracle_dynamic_relative_error.csv \
  --output-stem paper/figure/fine-gain-expert/qwen35_rank_precision_oracle_dynamic_relative_error
```

To show the per-hidden-state variability of the six candidates and the
retrospective oracle minimum, plot the detailed sample CSV:

```bash
python paper/figure/fine-gain-expert/plot_qwen35_oracle_dynamic_trace.py
```

An additional average-3-bit sweep evaluates seven pure-Qx_K candidates. These
sequences were chosen to cover different ways of spending the same top-10
nominal budget: one very high-rank-heavy sequence, several top-3/top-4-heavy
sequences, smoother mid-rank allocations, and Uniform Q3.

| Strategy | Nominal precision by routed-expert rank 1 to 10 |
|---|---|
| Avg3 83333 | `8/3/3/3/3/2/2/2/2/2` |
| Avg3 6444 | `6/4/4/4/2/2/2/2/2/2` |
| Avg3 633333 | `6/3/3/3/3/3/3/2/2/2` |
| Avg3 5553 | `5/5/5/3/2/2/2/2/2/2` |
| Avg3 544333 | `5/4/4/3/3/3/2/2/2/2` |
| Avg3 444433 | `4/4/4/4/3/3/2/2/2/2` |
| Uniform Q3 | `3/3/3/3/3/3/3/3/3/3` |

```bash
python paper/figure/fine-gain-expert/run_qwen35_rank_precision_comparison.py \
  --strategy-set avg3_extra
python paper/figure/fine-gain-expert/plot_qwen35_rank_precision_relative_error.py \
  --csv paper/figure/fine-gain-expert/qwen35_rank_precision_avg3_extra_relative_error.csv \
  --output-stem paper/figure/fine-gain-expert/qwen35_rank_precision_avg3_extra_relative_error
python paper/figure/fine-gain-expert/plot_qwen35_oracle_dynamic_trace.py \
  --candidate-set avg3_extra \
  --csv paper/figure/fine-gain-expert/qwen35_rank_precision_avg3_extra_relative_error_samples.csv \
  --output-stem paper/figure/fine-gain-expert/qwen35_rank_precision_avg3_extra_trace
```

The combined average-3-bit oracle uses both groups as candidates, deduplicating
Uniform Q3. It therefore selects the per-hidden-state best result from 12
rank-based pure-Qx_K candidates:

```bash
python paper/figure/fine-gain-expert/run_qwen35_rank_precision_comparison.py \
  --strategy-set avg3_combined
python paper/figure/fine-gain-expert/plot_qwen35_rank_precision_relative_error.py \
  --csv paper/figure/fine-gain-expert/qwen35_rank_precision_avg3_combined_oracle_static_uniform_relative_error.csv \
  --output-stem paper/figure/fine-gain-expert/qwen35_rank_precision_avg3_combined_relative_error
python paper/figure/fine-gain-expert/plot_qwen35_oracle_dynamic_trace.py \
  --candidate-set avg3_combined \
  --csv paper/figure/fine-gain-expert/qwen35_rank_precision_avg3_combined_relative_error_samples.csv \
  --output-stem paper/figure/fine-gain-expert/qwen35_rank_precision_avg3_combined_trace
```

The separate Q4-budget oracle comparison evaluates six pure-Qx_K candidates
whose nominal precision sums to 40 bits across the top-10 routed experts. Its
static expert-level baseline uses the same calibrated popularity ranking, with
the first half permanently assigned to `Q6_K` and the second half permanently
assigned to `Q2_K`:

| Strategy | Nominal precision by routed-expert rank 1 to 10 |
|---|---|
| Rank-aware | `8/6/6/5/4/3/2/2/2/2` |
| Rank-aware | `6/6/5/5/4/4/3/3/2/2` |
| Rank-aware | `6/5/5/5/4/4/4/3/2/2` |
| Rank-aware | `5/5/5/5/4/4/4/4/2/2` |
| Rank-aware | `5/5/5/4/4/4/4/3/3/3` |
| Uniform Q4 | `4/4/4/4/4/4/4/4/4/4` |

```bash
python paper/figure/fine-gain-expert/run_qwen35_rank_precision_comparison.py \
  --strategy-set oracle_dynamic_q4
python paper/figure/fine-gain-expert/plot_qwen35_rank_precision_relative_error.py \
  --csv paper/figure/fine-gain-expert/qwen35_rank_precision_oracle_dynamic_q4_relative_error.csv \
  --output-stem paper/figure/fine-gain-expert/qwen35_rank_precision_oracle_dynamic_q4_relative_error
python paper/figure/fine-gain-expert/plot_qwen35_oracle_dynamic_trace.py \
  --candidate-set q4 \
  --csv paper/figure/fine-gain-expert/qwen35_rank_precision_oracle_dynamic_q4_relative_error_samples.csv \
  --output-stem paper/figure/fine-gain-expert/qwen35_rank_precision_oracle_dynamic_q4_trace
```
