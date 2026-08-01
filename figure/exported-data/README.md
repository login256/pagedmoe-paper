# 论文图片数据导出

本目录冻结并整理了 2026-07-30 当前使用的 7 组图片数据：

1. Qwen3.5 Figure 1：cache capacity 与 decode throughput；
2. Qwen3.5 Figure 2：shared-bitplane 与 independent-precision；
3. Qwen3.5 Figure 3：shared-bitplane 与 mmap backend；
4. Qwen3.5 cache hit rate；
5. Qwen3.5 SSD traffic；
6. GLM-5.2 Figure 1；
7. GLM-5.2 Figure 2。

`all_figure_data.csv` 和 `all_figure_data.xlsx` 共包含 **282 个实际绘制点**。
逐图 CSV 不包含空值、失败点或未画出的 52 GiB 诊断点。完整输入快照位于
`raw/`，来源及 SHA-256 位于 `sources.csv`。

## 文件索引

| 文件 | 内容 | 点数 |
|---|---|---:|
| `qwen35_figure1.csv` | Qwen Figure 1 | 42 |
| `qwen35_figure2.csv` | Qwen Figure 2 | 36 |
| `qwen35_figure3.csv` | Qwen Figure 3 | 46 |
| `qwen35_cache_hit_rate.csv` | 五条 cache hit-rate 曲线 | 40 |
| `qwen35_ssd_traffic.csv` | 五条推导 SSD traffic 曲线 | 40 |
| `glm52_figure1.csv` | GLM Figure 1 | 42 |
| `glm52_figure2.csv` | GLM Figure 2 | 36 |
| `all_figure_data.csv` | 上述 7 组数据的统一长表 | 282 |
| `all_figure_data.xlsx` | 同一数据的多 sheet 工作簿 | 282 |
| `manifest.csv` | 图片名、别名、坐标轴、点数及主来源 | 7 |
| `sources.csv` | 输入快照、revision 与 SHA-256 | 17 |
| `metadata.json` | 导出版本和逐图行数 | - |
| `SHA256SUMS` | 本目录所有导出物的校验和 | - |

CSV 是规范数据；XLSX 仅用于查看和交付。`raw/` 保留绘图器读取的完整输入，
逐图 CSV 则只保留图上真正可见的数据点，并增加统一 provenance 字段。
规范 CSV 的 `series` 使用当前统一名称；`raw/` 中的历史名称按来源快照原样保留。

## 版本身份

Qwen 三张主图的最终审计数据固定取自：

```text
paper/main@fcfecb966bace07c85f3893533d51b9fb89a754b
```

`figure/cache-size-token-s` 从 `c11306c` 到上述 revision 没有变化。导出脚本
默认使用完整 revision 通过 `git show <revision>:<path>` 读取对象，并硬校验
SHA-256，因此不会因 `main` 后续移动而改变，也不会误读带手工 Q4_0 点的早期
Figure 1。

GLM 两张图和四个输入文件与本导出包一同版本化。由于包含这些文件的提交不能
自引用自身 commit，GLM 的 revision 写成 `content-addressed experiment snapshot
2026-07-30`；准确内容身份由 `sources.csv` 和 `SHA256SUMS` 中的 SHA-256 固定。

## Qwen3.5 Figure 1

规范图片：

```text
figure/cache-size-token-s/fig_cache_size_end_to_end_throughput.{pdf,png}
```

兼容别名为 `fig_cache_size_token_s.{pdf,png}`。x 轴为
`Expert cache capacity (GiB)`，y 轴为 `Decode throughput (token/s)`。

| 曲线 | 实际实验 | x 网格 | 点数 |
|---|---|---|---:|
| ik_llama IQ4 | routed-expert mmap budget | 8..48，步进 8 | 6 |
| Uniform4 | shared-bitplane `static4`，top-10 固定 4 bit | 4..48，步进 4 | 12 |
| Dyn Avg4 | shared-bitplane gain-based `c4l40` | 4..48，步进 4 | 12 |
| Dyn Avg2.5 | shared-bitplane gain-based `c2p5l25` | 4..48，步进 4 | 12 |

`Uniform4` 是固定 4-bit `static4`，不是 `c4l40`。当前 ik_llama 曲线是
实测 `UD-IQ4_XS`，不是旧图中的手工 Q4_0 数据。

PagedMoE 三条曲线取 server log 中 480-token decode window；ik_llama 取
`predicted_per_second = 500 / predicted_ms`。两类 backend 的吞吐时间窗并非完全
同源。ik_llama 的图上 x 是 nominal routed-expert mmap budget，实际 routed
residency、host peak 和 `MemoryMax` 已分别导出到 `actual_cache_gib`、
`total_host_memory_gib` 和 `memory_max_gib`。

## Qwen3.5 Figure 2

规范图片：

```text
figure/cache-size-token-s/fig_cache_size_independent_precision_comparison.{pdf,png}
```

兼容别名为 `fig_cache_size_prefix_reuse_ablation.{pdf,png}`。三条曲线均使用
4..48 GiB、步进 4 的 user-space cache capacity：

| 曲线 | backend/policy | 点数 |
|---|---|---:|
| Uniform4 | shared-bitplane `static4` | 12 |
| Dyn Avg2.5 (Independent) | independent-precision `c2p5l25` | 12 |
| Dyn Avg2.5 | shared-bitplane `c2p5l25` | 12 |

`(Independent)` 表示 independent-precision backend。该实验不仅让 cache key
包含 precision，也改变 selector 可见的当前 cache hint，因此不能把全部差异严格
解释为只关闭 bitplane prefix reuse。

## Qwen3.5 Figure 3

规范图片：

```text
figure/cache-size-token-s/fig_cache_size_mmap_backend_comparison.{pdf,png}
```

兼容别名为 `fig_cache_size_mmap_fallback_ablation.{pdf,png}`。当前是单面板，
x 轴为 `Total Memory (GiB)`。

| 曲线 | x 口径 | 点数 |
|---|---|---:|
| Uniform4 | cache capacity + 5.2 GiB | 12 |
| Dyn Avg2.5 | cache capacity + 5.2 GiB | 12 |
| Uniform4 (mmap) | cgroup `MemoryMax` | 11 |
| Dyn Avg2.5 (mmap) | cgroup `MemoryMax` | 11 |

shared x 为 9.2, 13.2, ..., 53.2 GiB。5.2 GiB 是论文指定的固定近似值，
来自 `final_non_pool_pss_exact` 三个样本 5.2104/5.2796/5.3646 GiB；它不是
实际 `memory.current`，也不是三个样本按一位小数得到的均值。mmap x 为实际配置的
total cgroup limit 8, 12, ..., 48 GiB，不再增加额外开销。

两条 4-bit 曲线都是固定 `static4`；Figure 3 没有 `c4l40`。mmap 是独立 backend，
不是在完整 shared-bitplane PagedMoE 中打开 fallback。原始 4 GiB mmap 两个 case
没有吞吐值，绘图器将其过滤且图片不标注；它们只保留在 raw snapshot 中。

## Cache Hit Rate

规范图片：

```text
figure/precision-object-cache-hit-rate/
fig_precision_object_cache_hit_rate_five_series.{pdf,png}
```

五条曲线均为 8..64 GiB、步进 8，共 40 点。`y_value` 和 `hit_rate` 使用 0..1
fraction，绘图时乘 100 显示百分比。

| 曲线 | 数据性质 |
|---|---|
| Uniform4 | runtime shared-bitplane counter |
| Dyn Avg4 | runtime shared-bitplane counter |
| Dyn Avg4 (Independent) | **precision trace 上的 exact-object LRU simulation** |
| Dyn Avg2.5 | runtime shared-bitplane counter |
| Dyn Avg2.5 (Independent) | runtime independent-precision counter |

因此这张图并非五条曲线全部来自同一个 runtime counter。特别是 Dyn Avg
Avg4 (Independent) 以 `(layer, expert, precision)` 为独立对象进行 LRU 模拟；
其 `data_kind` 明确为 `simulated_lru`。原模拟的 4-GiB 步进完整数据也保存在
`raw/qwen35_precision_object_cache_hit_rate_simulated.csv`。

uniform4、dynamic2.5 shared 和 dynamic2.5 independent 的上游 markdown summary
当前 checkout 已不存在；已提交的 five-series CSV 和本导出 raw snapshot 仍冻结了
最终数值。`sources.csv` 将此缺口标为 `not present in current checkout`。

## SSD Traffic

规范图片：

```text
figure/precision-object-cache-hit-rate/
fig_ssd_transfer_bytes_per_token_five_series.{pdf,png}
```

这 40 个点不是 SSD/NVMe 设备计数器实测流量，而是由 hit rate 解析推导：

```text
SSD bytes/token
  = (1 - hit_rate)
    * 60 MoE layers
    * 10 experts/token
    * average bits
    * 1,572,864 bytes/bitplane
```

图中的 GiB/token 再除以 `2^30`。每行保留 `hit_rate`、`miss_rate`、`avg_bits`、
`moe_layers`、`experts_per_token`、`bitplanes_per_token`、`bitplane_bytes` 和未换算的
`ssd_transfer_bytes_per_token`。所有行的 `data_kind` 都是
`analytically_derived`。Dyn Avg4 (Independent) 的输入 hit rate 还是
上述模拟值，其余四条使用 runtime hit-rate 输入。

## GLM-5.2 Figure 1

规范图片：

```text
paper/figure/cache-size-token-s/fig_cache_size_token_s_glm52.{pdf,png}
```

| 曲线 | 实际实验 | x 网格 | 点数 |
|---|---|---|---:|
| ik_llama IQ4 | routed-expert mmap budget | 8..48，步进 8 | 6 |
| Uniform4 | shared-bitplane `static4` | 4..48，步进 4 | 12 |
| Dyn Avg4 | shared-bitplane `c4l32` | 4..48，步进 4 | 12 |
| Dyn Avg2.5 | shared-bitplane `c2p5l20` | 4..48，步进 4 | 12 |

GLM 每 token 路由 top-8，所以平均 4-bit/2.5-bit 的总 bit budget 分别是 32/20。
PagedMoE y 使用 streamed response 中第 1 到第 500 completion event 之间的 499
token decode-phase throughput；ik_llama 同样使用 499-token decode window。

每个点 `n=1`。PagedMoE 的 `read_job_cache_hit_rate` 作为辅助字段导出，但 server
只周期性输出统计快照，不能将该字段描述为完整 500-token 请求的最终累计值。

## GLM-5.2 Figure 2

规范图片：

```text
paper/figure/cache-size-token-s/
fig_cache_size_independent_precision_comparison_glm52.{pdf,png}
```

兼容别名为 `fig_cache_size_prefix_reuse_ablation_glm52.{pdf,png}`。三条曲线为：

| 曲线 | backend/policy | 点数 |
|---|---|---:|
| Uniform4 | shared-bitplane `static4` | 12 |
| Dyn Avg2.5 (Independent) | independent-precision `c2p5l20` | 12 |
| Dyn Avg2.5 | shared-bitplane `c2p5l20` | 12 |

shared 两条曲线复用 2026-07-25 Figure 1 数据，independent 曲线测试于
2026-07-30。已有同日 4/24/48 GiB control 表明 shared 公共路径的两个 C ABI
差异不超过 0.345%，但历史 shared 曲线比 7 月 30 日同日 control 快 2.7% 到
4.2%。因此该图的大趋势成立，精确百分比仍包含跨日机器状态漂移；若用于严格
单变量论文结论，应同日交错重跑完整 shared sweep。

## 统一字段

最常用字段如下：

| 字段 | 含义 |
|---|---|
| `figure_id` | 稳定的图标识 |
| `curve_key` / `series` | 程序 key 与图例文字 |
| `x_value`, `x_name`, `x_unit` | 实际画出的 x |
| `source_x_value` | 加 overhead 或名义预算转换前的 x |
| `x_value_kind` / `budget_kind` | cache capacity、MemoryMax 或估算 total memory |
| `y_value`, `y_name`, `y_unit` | 实际画出的 y |
| `data_kind` | runtime measured、simulated 或 analytically derived |
| `metric_detail` | 吞吐时间窗或 hit-rate 定义 |
| `precision_case` | `static4`、`c4l40`、`c2p5l25` 等 |
| `n`, `std` | 重复次数及标准差；当前吞吐点均为 `n=1` |
| `source`, `source_sha256` | 直接数据源和内容身份 |
| `source_revision` | git revision 或明确的 working-tree snapshot |
| `raw_*` | 可用时指向原 stdout/server/command |
| `formula` | 模拟或解析推导公式 |
| `notes` | 该点适用的口径 caveat |

## 重新导出与验证

在仓库根目录运行：

```bash
python script/export_paper_figure_data.py
sha256sum -c paper/figure/exported-data/SHA256SUMS
```

脚本会验证 Qwen git 对象和 GLM 当前输入的预期 SHA-256、逐图完整网格、所有
绘制点 `status=ok` 且 y 非空，以及固定行数 42/36/46/40/40/42/36。若论文数据
被有意替换，必须同时更新输入快照、预期 hash、导出 README 和来源说明，不能
让脚本静默接受另一组数据。
