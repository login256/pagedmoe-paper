# Qwen3.5 Expert Qx_K Output Error

This figure measures the relative L2 output error of one routed expert from
Qwen3.5-397B-A17B after quantizing all three expert matrices with llama.cpp
Qx_K formats.

The checked-in result uses layer 0, expert 0, and one deterministic synthetic
BF16 hidden state sampled from `N(0, 1)` with seed `20260602`. The reference is
the original BF16 expert output. Both the reference and quantized paths round
the gated intermediate activation to BF16 before the down projection.

`Q1_K` and `Q7_K` do not exist in llama.cpp and are recorded as unavailable.
`Q8_K` is included as the high-precision endpoint requested for the sweep, but
llama.cpp defines it as an internal intermediate/dot-product format. The CSV
records both nominal bit widths and the actual storage cost including Qx_K
block metadata.

The runner uses `safe_open(...).get_slice()[expert_id]` and loads only one
expert from the original model. It quantizes, dequantizes, and multiplies one
matrix at a time so that multiple quantized expert copies are not resident
together.

```bash
python paper/figure/bit-error/run_qwen35_expert_qk_sweep.py
python paper/figure/bit-error/plot_qwen35_expert_qk_relative_error.py
```

The IQ sweep records every IQ primitive defined by llama.cpp:
`IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`,
`IQ4_XS`, and `IQ4_NL`. llama.cpp does not define IQ primitives from IQ5
through IQ8. The IQ formats use per-input importance weights derived from the
measured activation. Because the IQ1 and IQ2 reference quantizers perform an
expensive grid search, the IQ runner supports row-level parallelism:

```bash
python paper/figure/bit-error/run_qwen35_expert_iq_sweep.py --workers 1
python paper/figure/bit-error/plot_qwen35_expert_iq_relative_error.py
```
