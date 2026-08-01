# Direct vs Materialize

该目录比较 Qwen3.5-397B 在 shared PagedMoE backend 下的 Direct（fused）与
Materialize（完整 expert 先解量化）路径。

共同配置：

```text
precision: marginal-gain c2.5l25
prompt tokens: 118
decode tokens: 500
cache grid: 4,8,...,48 GiB
compute threads: 16
codebook / bitplane workers: 1 / 8
CUDA graph: enabled
```

Direct 数据复用论文 Figure 的本机实测快照：

```text
paper/figure/cache-size-token-s/data/shared_c2p5l25_20260714_114526.csv
SHA256: 6f34fbe2d3830e7c1d6720fedd7cf63e90ce26cf1faa653b21227f4ade02e646
```

Materialize 数据来自：

```text
src/target/bench/expert_predequant_c2p5l25_decode500_matrix_20260720/summary.csv
SHA256: c761ce995a2f567909266ce0d6b9811f56b34d88ceaeddecef600b669198e201
```

主图使用两份 summary 的 `decode_token_per_s`。`data.csv` 同时保留 client
高精度计时的 `completion_token_per_s`、逐点 slowdown 和 cache hit rate，便于后续审计。

绘图风格与 `paper/figure/cache-size-token-s/fig_cache_size_end_to_end_throughput.png`
保持一致。复现命令：

```bash
python paper/figure/expert-predequant-vs-fused/plot.py
```

输出：

```text
fig_expert_predequant_vs_fused.png
fig_expert_predequant_vs_fused.pdf
```

注意：每个点只有一次冷进程运行。Materialize 曲线存在 allocator、内存带宽和
高 cache 容量压力造成的非单调波动；图中保留原始结果，没有平滑。
