# KTensor-Lite Direct Quantization vs 8-Bit Readout

This experiment measures weight-space relative L2 error for one full
Qwen3.5-397B routed-MoE layer.

Default layer: `30`

Default expert range: all `512` experts

For each expert and each expert tensor (`gate`, `up`, `down`), the runner
processes the same 128K-element blocks used by the PagedMoE exporter.

It compares:

- `direct`: `ktensor-lite` encodes each block directly with `logk=b`, then
  decodes at `b` bits.
- `readout_from_pagedmoe_8bit`: the runner reads the already-exported PagedMoE
  8-bit ktensor-lite codebook and bitplanes, then decodes with only the first
  `b` bitplanes and the corresponding prefix codebook.
- `gptq_int4`: a GPTQ-Int4 reference from
  `/mnt/data/Qwen3.5-397B-A17B-GPTQ-Int4`, plotted at 4-bit.

The reported metric is:

```text
sqrt(sum((decoded_weight - original_weight)^2)) /
sqrt(sum(original_weight^2))
```

over the selected layer weights. The summary CSV includes both the whole-layer
aggregate (`tensor=all`) and per-tensor aggregates.

## Commands

Full layer, using the PagedMoE runtime storage path:

```bash
python paper/figure/ktensor-lite-readout-error/run_qwen35_layer_ktensor_lite_weight_error.py \
  --workers 8 \
  --force
```

Resume an interrupted run:

```bash
python paper/figure/ktensor-lite-readout-error/run_qwen35_layer_ktensor_lite_weight_error.py \
  --workers 8 \
  --resume
```

Quick smoke test on one expert:

```bash
python paper/figure/ktensor-lite-readout-error/run_qwen35_layer_ktensor_lite_weight_error.py \
  --expert-limit 1 \
  --output-stem smoke_qwen35_layer30_ktensor_lite_weight_error \
  --force
```

Pure ktensor-lite comparison without reading PagedMoE storage:

```bash
python paper/figure/ktensor-lite-readout-error/run_qwen35_layer_ktensor_lite_weight_error.py \
  --readout-source reencode \
  --workers 8 \
  --force
```

Plot the whole-layer summary after the full run:

```bash
python paper/figure/ktensor-lite-readout-error/plot_qwen35_layer_ktensor_lite_weight_error.py
```

Add the GPTQ-Int4 reference rows to the existing CSV files before plotting:

```bash
python paper/figure/ktensor-lite-readout-error/add_qwen35_layer_gptq_int4_weight_error.py \
  --workers 8
python paper/figure/ktensor-lite-readout-error/plot_qwen35_layer_ktensor_lite_weight_error.py
```

Plot one tensor:

```bash
python paper/figure/ktensor-lite-readout-error/plot_qwen35_layer_ktensor_lite_weight_error.py \
  --tensor gate
```

## Outputs

- `qwen35_layer30_ktensor_lite_weight_error_experts.csv`: per-expert aggregate
  rows for `direct` and `readout`.
- `qwen35_layer30_ktensor_lite_weight_error_tensors.csv`: per-expert,
  per-tensor rows.
- `qwen35_layer30_ktensor_lite_weight_error_summary.csv`: whole-layer and
  per-tensor aggregates.
- `qwen35_layer30_ktensor_lite_weight_error.json`: run metadata.
- `qwen35_layer30_ktensor_lite_weight_error_gptq_int4.json`: GPTQ reference
  metadata.
- `qwen35_layer30_ktensor_lite_weight_error.pdf/png`: plotted figure.
