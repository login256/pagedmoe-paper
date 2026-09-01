# PagedMoE 全文论证与实验审计

审计对象：`pagedmoe_VLDB(3).pdf`（13 页，其中正文至第 12 页，第 13 页为参考文献）。

## 0. 总判断

这篇论文不是“实验很少”，而是**不同论点的证据强度不均匀**：

- 最核心、最有价值的系统结论——“动态精度只有在不同精度共享同一缓存状态时，才能同时保留低请求字节数和跨 token 局部性”——已经有相当强的因果证据。`Independent` 与 `PagedMoE` 重放完全相同的 expert 请求和位宽分配，只改变缓存身份与缺失后缀复用，随后同时报告 byte hit、SSD bytes 和端到端吞吐；这是全文最 solid 的部分。
- “2.5-bit 动态分配仍保持足够质量”已有两个大模型和多任务端到端结果，广度尚可；但为什么 allocator 有效、routing weight 与 expert profile 两个信号是否都必要、随机 Gaussian profile 是否稳定，目前证据不够。
- “prefix compatibility 几乎没有额外量化损失”仅用 Qwen 的第 30 层权重误差支持，证据范围明显小于全文表述。
- “部署上 practical”目前缺少 checkpoint 构建时间、profile 构建时间、峰值内存、实际磁盘占用和在线 allocator 开销。
- “端到端优于 ik_llama.cpp 3.58×”是有用的 supporting comparison，但当前没有同时报告所用 UD-IQ4 checkpoint 的质量，不能把它写成严格的 matched-quality superiority。
- 性能实验未充分说明 prompt/trace 来源、prompt 数量、prefill 长度、重复次数和方差。若当前曲线主要来自单个 500-token decode trace，这是最容易被系统审稿人攻击的实验方法问题。

因此，当前稿件可概括为：**核心 insight 和核心因果实验已经成立；allocator、representation generality、workload robustness 和 practical overhead 还没有达到同样强度。**

一个直观评分如下（不是录用概率，只表示证据完整度）：

| 论证 | 当前强度 | 判断 |
|---|---:|---|
| SSD I/O 在 practical cache 下仍在关键路径 | B+ | 有完整 latency breakdown，但只覆盖一平台/未充分定义工作负载 |
| independent precision objects 破坏复用 | A- | 逻辑、trace simulation、真实 runtime 三层证据齐全 |
| cross-precision reuse 降低实际 SSD traffic | A | identical-request controlled experiment 很强 |
| traffic reduction 转化为吞吐提升 | A- | 两个模型、cache sweep；缺新的 latency breakdown 和跨 prompt 方差 |
| 2.5-bit 保持任务质量 | B+ | 两模型、多任务；但只展示少数预算点 |
| allocator 优于简单策略且两个信号均必要 | B- | 有 Uniform、restricted candidate set、rank baseline；缺 routing-only/profile-only 与校准稳健性 |
| prefix compatibility 代价很小 | B- | 仅 Qwen layer 30 的 weight error；缺全层、第二模型和 output error |
| direct compact execution 是必要且高效的 | B+ | 与 BF16 materialize 的严格控制比较很强；但 materialize 是较弱对手，缺 packed low-bit kernel overhead |
| explicit management 优于 mmap | B | 同 checkpoint/assignment 且 total-memory accounting 合理；但多个 runtime 因素共同变化，论文已承认不能单独归因 |
| offline/online 成本 practical | C | 基本未测 |
| 外部系统 matched-quality/matched-resource 优势 | C+ | 有 ik_llama throughput，但缺同 checkpoint 质量与更完整运行配置 |
| 对工作负载和硬件平衡稳健 | C | 两模型、cache sweep 有帮助；prompt 与硬件敏感性不足 |

## 1. 全文真正要证明的逻辑链

全文所有内容最终都服务于下面五个命题：

1. 在 batch=1、SSD-resident MoE 的本地推理中，即使已有 DRAM expert cache 和 I/O–compute overlap，实际 cache 容量下仍有显著 SSD stall。
2. 减少每次请求的 expert bits 可以降低 miss payload 和 resident footprint；token-level 非均匀分配比 uniform precision 有更好的质量—字节折中。
3. 但若每个位宽是独立对象，动态位宽会把一个 expert 的 cache identity 分裂成 `(expert, bitwidth)`，丢掉跨精度复用，抵消一部分收益。
4. 因此精度应表示一个共享 expert 状态的 refinement depth；表示、I/O、缓存和执行都必须保留这个嵌套关系。
5. PagedMoE 的 tree codebook、bitplane、pre-I/O allocator、missing-suffix loading 和 direct execution 共同实现这条链，并在质量可接受时提高吞吐。

其中第 3、4 个命题是论文真正的新意。第 1 个命题建立问题，第 2 个命题建立机会，第 5 个命题证明实现可行。

## 2. 逐章逐段论证审计

下面按“语义单元”覆盖全文所有实质性段落。重复出现的背景句和结果句合并审计，但没有略过任何独立观点。

### Abstract（第 1 页）

1. 第一组句子定义场景：MoE 稀疏路由使完整 expert pool 可以驻留 SSD、DRAM 缓存近期 expert，从而支持本地推理。作用是限定论文不是通用 MoE serving，而是 batch=1、SSD–DRAM–CPU/GPU 分层执行。
2. 第二组句子给出问题：practical DRAM cache 下，即使 I/O 与计算重叠，SSD I/O 仍在关键路径。它需要端到端 latency breakdown，而不只是 SSD bandwidth 或 cache hit；Figure 1 正在承担这项证明。
3. 第三组句子给出机会：依据 token importance 与 expert sensitivity 动态分配精度可以减少数据需求。它需要质量—bit budget 曲线和 policy 对比；当前主要由 Figure 2、Figure 10(b)、Table 2 和 Table 1 支撑，但证据较分散。
4. 第四组句子给出障碍：独立位宽对象不能跨精度复用并导致重复读取。这是系统核心命题；Figure 2(d)、Figure 8、Figure 9 共同支持。
5. 第五组句子提出 insight：precision 是一个共享 expert state 的 refinement depth。该句是 abstraction，不直接依赖实验；但其系统价值必须由 identical-request 的 Shared/Independent 比较验证。
6. 第六组句子概述设计：prefix-compatible tree codebook、bitplane、pre-I/O allocation、1–8-bit direct execution。这里实际包含三个需要分别验证的必要条件：prefix quality、allocator quality、compact execution cost。
7. 最后两句给出质量和吞吐 headline。当前数值都有对应实验，但“close”是主观措辞，应继续保留 model/task-dependent caveat，并最好给出 quality–throughput Pareto，而不是只给 2.5-bit 单点。

### Section 1 Introduction（第 1–2 页）

1. 第 1 段解释 sparse compute 如何使超大 MoE 本地推理成为可能，建立 GPU 执行 dense、CPU 执行 selected experts、SSD 容纳 pool、DRAM 做 cache 的系统模型。此段是 scope，不需额外实验，但后文 setup 必须严格与之对应。
2. 第 2 段把“sparse compute 不等于 sparse data movement”具体化，并用 Qwen、24 GiB、57.08% hit、1.53 GiB/token、87.16 ms exposed I/O、49.28% latency、5.65 token/s 证明瓶颈。作用是把论文从“低比特通常更快”提升为“缓存和 overlap 后仍必须减少请求字节”。风险是这些数字依赖 trace；当前没有在此处说明 trace 来源和方差。
3. 第 3 段解释 uniform precision 忽略 routing weights 与 expert sensitivity 两种变化。它同时提出 allocator 的两个输入信号；因此后文必须分别 ablate 两个信号，当前没有完全做到。
4. 第 4 段定义 independent precision object，并用 Dynamic Avg4 exact-object hit 从 57.08% 降到 35.15% 说明 cache identity fragmentation。逻辑清楚，但与第 8.3 节 byte-hit 实验重复，且 exact-object hit 与 byte hit 两套指标容易让读者误读。
5. 第 5 段提出 nested precision，可向上只补 suffix、向下直接使用长 prefix。它定义了双向复用：`b <= r` 的 downward reuse 与 `b > r` 的 incremental upward reuse。当前只报告总体 byte hit/traffic，未分别报告两类事件的频率和节省字节；这不是必做，但会让机制更直观。
6. 第 6 段给出 design principle，并说明一个 expert 只存一次、一个 cache identity、precision change 不产生新对象。这是论文最应反复保持一致的术语。
7. 第 7 段区分相关工作：多精度 checkpoint 和动态精度已有，PagedMoE 的贡献是把 nested state 贯穿 SSD、DRAM cache 和 direct execution。由此决定论文不能把“动态精度”或“nested quantization”本身作为主要 novelty，而必须围绕 persistent shared state 和 end-to-end data path。
8. 第 8 段列出三个必要条件：prefix-valid and independently addressable；after-routing-before-I/O allocation；direct compact execution。Section 5、6、7 与 Figure 10 应一一对应。这一组织非常好，应在实验结构中保留。
9. 第 9 段把三个条件映射到具体实现。它隐含声称 offline solver practical、allocator overhead small、generic 1–8-bit kernel practical；后两项目前缺直接测量。
10. 第 10 段用 Uniform4 → Independent Avg2.5 → Shared Avg2.5 的 5.71 → 6.67 → 8.89 分解总体收益，并补充 GLM 与 mmap/ik_llama。这是最好的结果叙述框架，应该成为实验章节的主轴。
11. 三个 contribution bullet 分别对应 abstraction、end-to-end state-preserving runtime、pre-I/O allocator/evaluation。第一、第二项证据较强；第三项中 allocator 本身的证据最弱。

### Section 2 Background and Motivation（第 2–3 页）

#### 2.1 SSD-Resident Local MoE Inference

1. Equation (1) 定义 routed output，并强调 expert identity 是 layer-specific。后面所有 cache key 都必须包含 layer；正文有时简写 `(e,b)`，应明确这只是省略 layer。
2. Qwen 的 10/512、7.55B active、386.55B total 用来证明 sparse activation 不能消除完整 pool 的容量需求。这是模型事实，不需实验。
3. GPU/CPU/SSD/DRAM 的执行组织定义性能边界：expert bytes、resident fraction、overlapped 后 exposed I/O、DRAM traffic。这实际上给出了一个很好的实验指标链：`requested bytes → hit → SSD bytes → exposed I/O → total latency`。当前论文只在不同图里报告其中一部分，最适合合并成一个机制表。

#### 2.2 Fixed-Precision Caching Leaves SSD I/O on the Critical Path

1. Fixed4 被描述为完整 4-bit expert object、fused low-bit kernel、asynchronous reads 与 ready-expert computation overlap。该实验是“已经优化常规手段后仍有问题”的反事实基线。
2. Figure 1 同时给 cache hit 与 latency breakdown，24 GiB 点支持“SSD I/O 是 first-order bottleneck”。这是必要且应保留的图。
3. 结论是减少 expert data volume 同时改善 cache footprint 和 miss payload。该结论由系统关系和后续实验支持。

风险：motivation 的 `Fixed4=5.65` 与 evaluation 的 `Uniform4=5.71` 不是同一缓存粒度；正文解释了 Uniform4 使用 prefix-preserving cache，但读者仍可能认为数字不一致。建议统一命名并在首次出现时用一句话明确：Fixed4 是 whole-object baseline，Uniform4 是 PagedMoE runtime 上的 fixed-request configuration。

### Section 3 Dynamic Expert Precision and Cross-Precision Reuse（第 3–5 页）

#### 3.1 Token-Level Precision Reduces Expert Data Demand

1. 低精度有两种系统收益：miss payload 小、同一 cache 容量容纳更多状态。此命题无需单独证明，但要在机制表中同时展示 SSD traffic 与 byte hit。
2. Figure 2(a) 说明 routing rank 只给顺序，不给当前 token 的实际幅值和 gap；它用于反驳纯 rank policy。
3. Figure 2(b) 说明不同精度的 expert-output error 有长尾；它用于反驳 uniform 降位宽。
4. Figure 2(c) 的 restricted per-token candidate-set best 在 avg 4 下把 routed-output error 从 14.56% 降到 10.59%，用于证明 token-level headroom，而不是证明 PagedMoE allocator 最优。正文对这一边界说明得正确。
5. 本小节只看 Qwen layer 30，而且使用 independently quantized group-wise uniform quantizer；后面的 progressive representation 和 Figure 10(b) 使用另一实验口径，导致 14.56% 与 15.3% 等相近但不同的数字。建议把 motivation headroom 与 allocator evaluation 合并成一个一致的实验，或显式说明 quantizer/trace 差异。

#### 3.2 Precision-Specific Objects Break Reuse

1. 独立 1–8-bit variants 需要 36 bits/weight，是单个 4-bit 表示的 9×。这是解析结论；还需要实际 checkpoint size 和 metadata 来说明部署代价。
2. cache key 从 `(layer, expert)` 变为 `(layer, expert, bitwidth)`，相邻 token 只有位宽也相同时才能共享。这是严格的身份逻辑。
3. “高精度可满足低精度”只解决单向复用；低到高仍不能复用，且 cache 中可能同时保留重叠数据。该分析很重要，应保留。
4. Figure 2(d) 显示独立对象使 Avg4/Avg3 hit 明显低于 Fixed4/Fixed3，证明 obstacle。由于 Figure 8 后面以真实 runtime、byte hit 和 SSD bytes 更完整地重做了这个问题，Figure 2(d) 是优先可删除或迁移的重复图。

#### 3.3 Precision as Refinement Depth

1. Equation (2) 把 expert 表示定义为 refinement sequence，Equation (3) 定义请求只读 missing suffix。这是 abstraction 的形式化。
2. `b <= r` 可直接用长 prefix，`b > r` 可复用短 prefix，是双向 cross-precision reuse。
3. prefix-valid 与 independently addressable 是 representation 必要条件；pre-I/O 与 direct execution 是端到端必要条件。
4. 这些是结构性推论，不需要每条做独立实验；但需要分别证明 representation penalty 小、SSD 确实只读 suffix、执行没有重新膨胀。当前三类证据都有，但第一类范围偏窄。

#### 3.4 Assign Precision before Expert I/O

1. Equation (4) 是理想 current-token routed-output deviation；若用它决策，必须先载入并执行候选位宽，产生循环依赖。
2. 8^10 搜索空间说明 online exact selection 不实际。
3. rank-only 忽略当前 routing magnitude，routing-only 忽略 expert/bit sensitivity，因此提出 `routing weight + offline profile`。
4. allocator 与 cache state 解耦：allocator 决定质量—requested-bits，cache 决定所需 prefix 中哪些真正 miss。这个设计使质量策略与缓存策略可以被独立比较，是优点。
5. 由于正文明确批评 rank-only、routing-only，所以至少应该实验比较 Uniform、Rank、Routing-only、Profile-only、Full；当前只有 Uniform、少量 rank policy 和 Full，论证没有闭合。

#### 3.5 Execute the Selected Prefix Directly

1. Materialize-BF16 会把低比特读取变成完整 BF16 写回与再次读取，可能超过 SSD 边界节省的数据。Figure 10(c) 强力支持这一点。
2. CPU expert execution 与 GPU staging 的 0.095–0.096 ms 对 0.114–0.121/0.31–0.32 ms 用来说明 batch=1 下无需逐 expert 搬到 GPU。该微基准缺少位宽、payload、PCIe/传输设置说明；若保留强数字，应在 setup/表中补齐，否则将表述弱化为平台限定观察。
3. generic direct kernel 必须支持非连续 bitplanes 与 1–8-bit selected prefix。这不仅要比 Materialize 快，还应测它相对 conventional packed fixed-bit kernel 的额外成本。

### Section 4 Overview（第 5–6 页）

本节没有新结论，而是把 offline progressive checkpoint/error profiles 与 online routing/allocation/cache/direct execution 串起来。Figure 3 很有效，应保留。需要注意：overview 多次重复“same compact state”“no separate object”，可压缩 1–2 段，为实验让页数。

### Section 5 Representation（第 6–7 页）

#### 5.1–5.2 Tree Codebook

1. 最大深度 K-means 定义 8-bit leaf codebook；排序后自底向上合并，internal node 使用后代频率加权平均。这保证固定 leaf assignment/tree 下的平方误差最小。
2. 每个 index prefix 都对应一个 internal/leaf reconstruction，因此 1–8-bit 均有效。这是构造性质。
3. 低精度 codebook 不是独立最优，而是从最大深度 assignment 派生，因此必须测 prefix-compatibility penalty。当前 Figure 10(a) 正是对应实验，但只测 layer 30 的 weight error，不能充分支持“nearly overlap across bit-widths”作为一般结论。

#### 5.3 Bitplane Layout

1. element-major index 在增量读一位时会带来 read amplification；bitplane transpose 使每个 refinement 独立可寻址。
2. 当前正文把 `P_e,j` 描述为逻辑上的 expert-level bitplane，但没有给实际 physical I/O unit、对齐和 request size。实现中若以 16 KiB block-bitplane 为最小 I/O 单位，应明确区分“logical expert plane”和“physical cache/I/O page”；否则 reviewer 无法复现实测 SSD traffic。

#### 5.4 Storage and Metadata

1. 8-bit index payload 是 Fixed4 的 2×，但比独立 1–8-bit variants 的 36 bits/weight 小 4.5×；这是重要 trade-off，不能只强调优点。
2. 每 `N >= 2^17` block 的 tree codebook metadata 小于 8-bit payload 的 0.8%。但对 1-bit 或 2.5-bit 当前请求而言，metadata 占 active bytes 的比例会更高；实际 SSD bytes 已包含这些影响时应明确说明。
3. “optimized offline solver makes construction practical”没有实验支持。需要实际 checkpoint size、build time、profile time 和 peak memory。

### Section 6 Pre-I/O Allocation（第 7–8 页）

#### 6.1 Offline Profiles

1. 用 10 个 dataset-independent Gaussian BF16 inputs 构建每个 expert/bit 的 normalized output-error profile，优点是不需要任务数据。
2. 这里隐含三个尚未验证的命题：10 个样本足够；Gaussian input 能代表真实 hidden-state sensitivity；不同 seed/profile 不会显著改变分配与质量。

#### 6.2 Surrogate

1. 从 exact squared routed error 出发，省略 cross-expert inner products，得到 separable `sum g_i^2 epsilon_{e_i,b_i}`。
2. `g_i^2` 表示当前 token importance，`epsilon` 表示 expert/precision sensitivity。
3. 数学动机合理，但当前没有测 surrogate 与真实 marginal quality gain 的相关性或 assignment regret；Figure 10(b) 与 Table 2 只给间接结果。

#### 6.3 Greedy Allocation

1. 从每 expert 1 bit 开始，对所有 refinement 的 `Delta` 全局排序，选择 `(c-1)K` 个。
2. 99.45% Qwen profiles 自然满足 nonnegative diminishing returns，其余 clip+PAVA；修正后是 separable discrete-concave allocation，因此 greedy 对修正 surrogate 最优。
3. 数学最优性不需要再用实验“证明”；真正需要测的是 profile 修正是否改变质量、allocator wall-clock latency，以及 surrogate 是否对应真实质量。
4. `O(K bmax log(K bmax))` 与 70 candidates 只说明渐进复杂度小，不等于实际开销可忽略。

### Section 7 Runtime（第 8–9 页）

#### 7.1 Prefix-Preserving Cache

1. bitplane 是 allocation/I/O/replacement 单位，codebook 与第一 plane 共同生存；因此 executable prefix 一定有 codebook。
2. reverse refresh 使低位更新得更近、高位优先淘汰，维持 prefix-residency invariant。
3. bitplanes 可物理不连续，执行时 pin 住请求 prefix。这些主要是正确性设计，不一定需要昂贵 ablation。
4. explicit management 的理由不是虚拟地址能否连续，而是 scheduler 需要知道 residency、missing suffix、in-flight I/O 和 readiness。Figure 10(d) 只能证明 complete runtime 差异，不能把全部差异单独归因给 replacement policy；正文已正确承认。

#### 7.2 Missing-Suffix and Scheduling

1. 相同请求下只读 `r+1...b`，是核心机制；Figure 8 的 SSD bytes 是直接证据。
2. ready experts 与剩余 I/O pipeline，不改变请求字节，只隐藏 residual latency。它是常规但必要的实现。除非把 scheduler 本身列为贡献，否则不必优先做 barrier-vs-pipeline ablation。

#### 7.3 Direct Execution

1. kernel 从每个 plane 取 32-bit slice，在寄存器内组装 32 个 8-bit indices，再查当前深度 codebook 并立即 BF16 dot-product、FP32 accumulate。
2. `b <= 5` 单寄存器 lookup、`b > 5` 多寄存器合并，暗示不同 b 的 kernel cost 非线性；需要 per-bit microbenchmark 才能解释 Dynamic Avg4 虽 SSD traffic 更低却略慢于 Uniform4 的现象。
3. 四行 unroll、block parallel、never write expert-scale intermediate 支持 direct compact execution。Figure 10(c) 证明避免 materialize 很重要，但未量化 bitplane-native 相对 packed fixed-bit kernel 的开销。

### Section 8 Evaluation（第 9–12 页）

#### 8.1 Setup

优点：硬件、两模型、量化 scope、主要 configurations、cache/total-memory accounting、direct-I/O metric、500 decode tokens 都已说明。

缺口：

- prompt/trace 来自什么数据集、prompt 数量、prefill token 数或分布没有说明；
- “每次从空 expert cache 开始，prefill 后测 500 token”实际是 empty-before-prefill、prefill-warmed decode，不是 decode cold-cache；应准确命名；
- 性能运行次数、均值/中位数、方差或置信区间未写。若已经做 3 次平均，只需补一句并在图上说明 error bars 小于 marker；
- 没有说明生成参数/teacher forcing、相同 token sequence 如何保证，以及不同系统是否使用同一 routing trace；
- 没有给 OS/kernel、SGLang/ik_llama commit、线程绑定、direct-I/O alignment/queue depth、physical I/O unit；这些可放 artifact，但正文至少需给关键项；
- allocator latency、checkpoint construction/profile cost 未纳入。

#### 8.2 Quality and End-to-End

1. Table 1 在 Qwen 上比较 BF16、GPTQ、Uniform4、Dyn2.5，在 GLM 上比较 official FP8、MXFP4、Uniform4、Dyn2.5。两模型和多任务覆盖足够，不需要再堆更多 benchmark。
2. Uniform4 接近 GPTQ/MXFP4 支持 4-bit prefix-compatible representation 的 end-to-end quality。
3. Dyn2.5 在 Qwen 各任务接近 GPTQ，在 GLM 多数任务接近或超过 MXFP4，但 GSM8K 明显下降；正文诚实保留 model/task-dependent caveat。
4. Figure 7 的 cache-capacity sweep 证明性能不是单一 24 GiB 点，且两模型趋势一致。
5. Dynamic Avg4 与 Uniform4 吞吐接近，说明在线动态精度没有巨大总体开销，但没有 latency breakdown，无法区分 allocator、不同 b kernel cost 与 I/O 的贡献。
6. ik_llama 的速度比较使用了不同 checkpoint/quantizer/runtime。它可以证明“完整可运行系统的现实参考”，但若不测 IQ4 质量，不能证明在相同质量下快 3.58×。

#### 8.3 Fewer Bits vs Cross-Precision Reuse

这是全文最强的一节。

1. Independent 与 Shared 使用完全相同 expert requests 和 bit assignments，只改变 `(e,b)` 独立对象还是共享 resident bitplanes；这是正确的 controlled baseline。
2. Qwen 24 GiB：Dyn Avg2.5 byte hit 47.94% → 68.50%，SSD 1.14 → 0.69 GiB/token，吞吐 6.67 → 8.89 token/s。数据从机制到结果闭环。
3. Uniform4 5.71 → Independent 6.67 是 fewer requested bits 的 1.17×；Independent 6.67 → Shared 8.89 是 reuse 的额外 1.33×；合计 1.56×。因果分解清楚。
4. GLM 24 GiB 的 1.58 → 2.12 → 2.64 重复同一趋势，排除了只在 Qwen 上成立。
5. Figure 8（hit/traffic）和 Figure 9（throughput）都做 cache sweep，但与 Figure 2(d)、Figure 7 有重复，可重新组合以节省版面。
6. 最值得新增的是 24 GiB 的 `exposed SSD stall` 和 `total latency` 列，使 `hit → traffic → critical-path stall → throughput` 完全闭环。

#### 8.4 Component Validation

1. Prefix compatibility：分析 storage/metadata 很清楚，但实验仅为 Qwen layer 30 weight error。应升级为全层/多层、两模型、expert-output error；GPTQ 的 weight-error 横向参考意义有限，可删。
2. Allocation：Figure 10(b) 的 Uniform/Restricted candidate best/Full 和 Table 2 的 rank baselines说明 Full 有效，但没有 routing-only/profile-only，也没有 random profile 稳健性；这是最需要补的 component experiment。
3. Direct execution：same requests/assignments/cache/I/O 下 Direct 比 Materialize 快 3.94–8.79×，控制非常好。但 Materialize 本来就会产生巨大 BF16 traffic；最好补 `packed fixed-b fused` 或至少 per-bit kernel latency，证明 generic bitplane kernel 的实际代价。
4. mmap：same progressive checkpoint/assignment、total-memory accounting 是合理对照；论文也承认 residency/replacement/I/O/scheduling 同时变化。这个实验应作为 supporting comparison，不应占据比 allocator validation 更大的版面。

### Sections 9–10（第 12 页）

1. Related Work 正确地把 PagedMoE 与 fixed-object offloading、nested quantization、dynamic-precision MoE、compact kernels 分开。
2. 当前实验没有直接运行 HOBBIT、D2MoE、SliceMoE、DyMoE、MxMoE/GEMQ。若代码/模型/硬件不可比，不必强行复现全部；更可行的是在同一 runtime 中实现其核心 policy 等价基线：miss-aware low bit、rank/gate-only、profile-only、independent precision、mmap progressive。
3. Conclusion 只重述核心结果，边界较克制。若不补硬件敏感性，应继续限定为“on our local-PC platform / evaluated models”，不要扩大到一般 serving。

## 3. 当前实验清单：保留、合并与替代建议

| 当前项目 | 它证明什么 | 当前问题 | 建议 |
|---|---|---|---|
| Figure 1 Fixed4 hit + latency breakdown | practical cache 后仍 I/O-bound | workload 未充分定义 | 保留；补 prompt/repeat 信息 |
| Figure 2(a) routing distributions | rank 不等于当前 importance | 仅 Qwen/layer 30 | 可保留但缩小 |
| Figure 2(b) output-error distributions | sensitivity 随 expert/bit 变化 | 与后面 allocator/profile 实验重复 | 与新的 allocator validation 合并 |
| Figure 2(c) restricted headroom | token-level allocation 有机会 | 与 Figure 10(b) 数字/quantizer 口径不同 | 移到 §8.4 或删除，只在 motivation 引用后文 |
| Figure 2(d) exact-object hit | independent objects 破坏 reuse | 与 Figure 8 重复，指标又不同 | 优先删除 |
| Table 1 quality | chosen operating point 的任务质量 | 无 IQ4 质量；无 budget frontier | 保留；补 IQ4 或弱化 3.58× 比较 |
| Figure 7 throughput vs cache | 两模型端到端性能和 external runtime | 与 Figure 9 重复 | 与 Figure 9 合并 |
| Figure 8 byte hit + SSD traffic | reuse 的直接 I/O 机制 | 两条 capacity 曲线占空间；无 exposed stall | 改为 24 GiB 机制表，或加第三列 stall |
| Figure 9 U4→Independent→Shared | fewer bits 与 reuse 的吞吐分解 | 与 Figure 7 重复 | 合并为同一 2-model cache-sweep 图 |
| Figure 10(a) prefix error | prefix compatibility cost | 仅 layer 30 weight error | 用全层/两模型聚合结果替代 |
| Figure 10(b) allocator | pre-I/O policy 有效 | 缺两个输入信号 ablation、校准稳定性 | 用 Full/Routing-only/Profile-only/Rank/Uniform 替代 |
| Figure 10(c) Direct vs Materialize | 保持 compactness 必要 | Materialize 较弱；capacity sweep 非必要 | 压缩为一个点或 per-bit microbenchmark |
| Figure 10(d) mmap | explicit complete runtime 更快 | 多因素共同变化 | 留一个代表点/小表，给 allocator 腾空间 |
| Table 2 rank PPL | Full 优于简单 rank | 与 Figure 10(b) 重复且仅 WikiText | 改为 allocator ablation/预算 Pareto 表 |

最明显的三组重复是：

1. Figure 2(d) 与 Figure 8；
2. Figure 7 与 Figure 9；
3. Figure 2(c)、Figure 10(b) 与 Table 2。

因此新增 solid 实验不必增加总页数，关键是**用更强实验替换重复实验**。

## 4. 需要补充的实验：具体做法

### P0-1：工作负载、重复性与方差

**要证明：**结果不是单个 prompt/单条 routing trace 的偶然局部性。

**做法：**

- 选 16–32 个 prompt，至少覆盖 3–4 类文本（对话、代码、推理、普通文本）；固定报告采样规则，不手挑 prompt。
- 明确 prompt token length；可用固定 128/512 两档，或报告真实分布。
- 每个 prompt 在 empty-before-prefill 条件下执行 prefill，然后测 500 decode tokens；明确这是 prefill-warmed decode。
- 关键 24 GiB 配置至少运行 3 次；报告跨 prompt mean/median、95% CI 或 min–max。完整 cache sweep 可只在代表 trace 上运行，但 24 GiB 主结论必须跨 prompt。
- 为严格比较 Independent 与 Shared，重放相同 expert IDs 和 bit assignments；说明 trace 来自哪个配置。另做一次真实 free-running end-to-end，确认 replay 结论与真实生成一致。
- 图上若误差线小于 marker，直接写明。

**指标：**token/s、byte hit、SSD GiB/token、exposed SSD ms/token；同时报告 prompt 间 CV 或 p5/p95。

**预期：**绝对吞吐随 prompt 变化，但 `Uniform < Independent < Shared` 的次序稳定；若少数 prompt 上 reuse 收益很小，应报告它们的 reuse distance/working-set 特征，而不是隐藏。

**版面：**主要只增加 §8.1 的 3–4 句和图中 error bar，不需要新图。

### P0-2：质量—吞吐 Pareto / bit-budget sweep

**要证明：**2.5 bit 不是事后挑出的单点，allocator 在一段预算范围内都给出有效折中。

**做法：**

- 预算 `c ∈ {2.0, 2.5, 3.0, 3.5, 4.0}`；若资源允许再加 4.5。
- Qwen 与 GLM 在 24 GiB 测 token/s、SSD bytes 和 WikiText perplexity；完整任务表仍只保留 2.5 与 4 两个 operating points。
- 同一图加入 Uniform2/3/4、简单 Rank policies 和 Full allocator。横轴可以是 PPL/质量损失，纵轴为 token/s；每点标注平均 bits。
- 也可用两个小图：`PPL vs avg bits` 与 `throughput vs avg bits`，比跨任务聚合分数更容易解释。

**预期：**Full allocator 在 2–4 bit 区间形成比 Uniform/Rank 更好的 Pareto frontier，2.5 是明显 knee；如果只在 2.5 附近有效，应缩小论文本身的普适表述。

**版面：**替换 Figure 2(c) 或 Table 2，不应新增整块空间。

### P0-3：allocator 的两个信号和校准稳健性

**要证明：**`routing weight × expert-specific marginal error` 不是无必要的复杂度；10 个 Gaussian inputs 确实足够。

**Policy ablation：**

1. Uniform；
2. Rank-only（现有 A/B/C 可精简为最强一个）；
3. Routing-only：保留当前 `g_i^2`，但所有 expert 使用全局平均 marginal-error curve；
4. Profile-only：使用 expert-specific `epsilon_{e,b}`，但令所有 selected experts importance 相同；
5. Full：`g_i^2 × Delta epsilon_{e,b}`；
6. Restricted/sample oracle，仅作为 upper-bound reference，并明确不是 global oracle。

**测量范围：**

- held-out tokens，至少覆盖 Qwen/GLM 的 early/middle/late layers；最好对全层聚合，报告 median 和 p90 actual routed-output error。
- 在 WikiText PPL 上对最关键的 Uniform、Routing-only、Profile-only、Full 做端到端确认。
- 计算 surrogate marginal gain 与真实 held-out marginal output-error reduction 的 Spearman correlation；或从大量随机可行 assignments 中报告 Full 的 regret/percentile。

**Calibration ablation：**

- `|Z| ∈ {1, 5, 10, 20/50}`；至少 3 个 random seeds；
- Gaussian inputs 与少量真实 hidden-state calibration inputs 对比；
- 指标为 profile rank correlation、assignment 一致率、routed-output error 和一项端到端 PPL。

**PAVA ablation（可选一行）：**raw profiles、clip-only、clip+PAVA 的质量差异；若 99.45% 已满足，预期差异极小，可只报告“改变了多少 assignment”。

**预期：**Full 优于两个 single-signal variant；10-sample Gaussian 接近 20/50-sample，并在 seed 间稳定。若 Gaussian 明显差于真实 hidden states，应改用真实 calibration 或把 dataset-independent 降为 engineering choice。

**版面：**完全替换现有 Figure 10(b)+Table 2；这是最高价值的替换。

### P0-4：representation 的全层/两模型 prefix penalty

**要证明：**从 8-bit leaf assignment 读取短 prefix 的代价在整个模型上小，而不是只在 Qwen layer 30 恰好小。

**做法：**

- 对 `b=1...8` 比较 independently trained b-bit codebook 与 progressive b-bit prefix。
- 覆盖两个模型的所有 routed-expert layers；若全量代价过高，至少固定 early/middle/late 五层，并随机采样 expert/block。
- 同时测 relative weight error 与 held-out expert-output error。output error 比 weight error 更贴近 allocator 与模型质量。
- 报告每个 b 的 median、p90/p95 和最坏层；再保留 Uniform4 end-to-end quality 作为系统级验证。

**预期：**3–8 bit penalty 很小，1–2 bit 可能更明显。若低位 penalty 大，不要声称“all bit-widths nearly overlap”，改成“the operating range used by the allocator has small penalty”。

**版面：**直接替换 Figure 10(a)，不加页。

### P0-5：从 SSD bytes 到关键路径 latency 的完整因果表

**要证明：**shared reuse 不仅减少日志中的读字节，也确实减少 exposed I/O 并带来吞吐提升。

**做法：**在 Qwen/24 GiB 用同一 trace 报告：

| Configuration | Avg bits | Byte hit | SSD GiB/token | Exposed SSD ms | Expert compute ms | Other ms | Total ms | token/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Uniform4 | 4.0 | 已有/补测 | 已有/补测 | 补测 | 补测 | 补测 | 补测 | 5.71 |
| Dynamic2.5-Independent | 2.5 | 47.94% | 1.14 | 补测 | 补测 | 补测 | 补测 | 6.67 |
| PagedMoE2.5 | 2.5 | 68.50% | 0.69 | 补测 | 补测 | 补测 | 补测 | 8.89 |

最好再加 Dynamic4 或在正文解释为什么它虽然 traffic 较低，吞吐仍略低于 Uniform4：高位 expert 的 kernel cost、allocator overhead 或 overlap 形态是哪一项。

**预期：**Independent 的 fewer bits 降低 miss payload；Shared 进一步提高 byte hit/只读 suffix；两者逐步降低 exposed SSD stall。compute/other 不应发生无法解释的大变化。

**版面：**用这张紧凑表替代 Figure 8 的两张曲线；capacity robustness 由合并后的 Figure 7/9 保留。

### P0-6：实际部署成本和在线开销

**要证明：**“single-copy/practical/pre-I/O”不仅在理论上成立。

**做法：**做一个很小的两模型表：

- BF16 routed-expert pool size；Fixed4 checkpoint size；Paged 8-bit checkpoint size；independent 1–8-bit analytical/actual size；
- tree-codebook metadata absolute size；
- checkpoint construction wall time、peak host memory；
- 10-input profile construction wall time、profile storage size；
- allocator latency（µs/layer/token 与 ms/token）、cache bookkeeping latency；
- 若实际 I/O/cache page 是 16 KiB，写出 unit、alignment 与 direct-I/O request coalescing。

**预期：**Paged disk footprint约为 Fixed4 的 2×、明显小于 independent 36-bit variants；metadata 很小；offline 时间可能是数小时但只需一次；allocator 相对 112–177 ms/token 的 decode latency应接近可忽略。

**版面：**一个 6–8 行小表或 §8.4 最后一段，不超过 0.15–0.2 页。

### P1-1：direct kernel 的更强对照

**做法：**对 `b=1...8` 测单 expert invocation latency/有效带宽：

- bitplane-native Direct；
- conventional packed fixed-b fused kernel（若有）；
- Materialize BF16 + BF16 GEMV。

保持相同线程数、expert shape、activation，报告 DRAM bytes。Direct vs Materialize 仍证明“不能膨胀”；Direct vs packed 证明“支持 progressive/noncontiguous prefix 的额外成本可接受”。

**预期：**Direct 大幅优于 Materialize；相对 packed kernel 有一定 overhead，且 `b>5` 因多寄存器 lookup 出现台阶。这个结果还能解释 Dynamic Avg4 的性能。

**版面：**把当前 Figure 10(c) 的 cache-capacity sweep 改成 per-bit 小图；无须在每个 cache 容量重复 Materialize 比较。

### P1-2：外部 baseline 的 matched-quality 说明

**最低成本做法：**用同一 harness 测 Qwen/GLM 的 UD-IQ4 checkpoint 质量，并把结果加入 Table 1 或 artifact；正文写清 ik_llama 是 different-format complete-runtime reference。

**更强做法：**若能运行 GPTQ-INT4 或另一固定 4-bit runtime，按相同 prompt、host-memory、GPU usage 做 performance；或选 PagedMoE budget 使其 WikiText PPL 与 IQ4 匹配，再比较 token/s。

**预期：**即使不能完全 matched，读者能看到速度与质量的相对位置，不会怀疑 3.58× 来自质量更差的 checkpoint。

### P1-3：SSD/CPU 平衡敏感性

**做法：**在 24 GiB 下用 I/O throttling 模拟 2/4/8/12 GB/s，或补一个 PCIe 4.0 SSD；另测 8/16/32 CPU threads。只需 Qwen 和三种核心配置。

**要看：**当 SSD 更慢时 shared reuse 增益应更大；当 CPU 成为瓶颈时相对增益应收缩但不反转。这样能界定 PagedMoE 适用区间。

**优先级：**低于 workload robustness 和 allocator validation。两模型+cache sweep 已提供一部分泛化证据，页数紧时可只放 artifact。

### P2：可以不优先做的实验

- 第三个超大模型：当前已有 Qwen 与 GLM，边际价值低于 allocator/workload 补强。
- 更多下游 benchmark：Table 1 的任务广度已足够。
- batch>1：明确限定 local batch=1 即可。
- 每种 cache replacement policy 大规模对比：prefix invariant 是正确性需求，核心 Independent/Shared 对照已经更重要。
- exhaustive 证明 greedy 对 surrogate 最优：已有数学条件与引用；应验证 surrogate，而不是重复验证 greedy 算法。
- 单独 barrier-vs-pipeline ablation：除非把 scheduler 写成主要贡献，否则可省。

## 5. 方案一：从头设计一套更 solid 的 Evaluation

建议把实验写成四个研究问题，而不是按“总结果—分解—零散组件”堆图。

### 8.1 Questions, Methodology, and Baselines（约 0.45 页）

先用四句话明确 RQ：

- RQ1：PagedMoE 在什么质量代价下减少 requested expert data 并提高端到端吞吐？
- RQ2：收益中多少来自 fewer requested bits，多少来自 cross-precision reuse？
- RQ3：prefix representation、pre-I/O allocator 和 direct execution 是否分别满足必要条件？
- RQ4：结果对 cache capacity、prompt 和 runtime choice 是否稳健，部署成本是多少？

随后说明硬件、模型、量化 scope、prompt 集、prefill/decode、3 次重复、metrics、memory accounting。把 baseline 分三类：

1. quality：BF16/FP8、GPTQ/MXFP4、UD-IQ4；
2. causal system：Uniform、Dynamic-Independent、PagedMoE；
3. complete runtime：ik_llama、mmap-progressive。

### 8.2 Quality–Efficiency Frontier（约 0.65 页）

**Table 1：**保留当前多任务质量表，在 Qwen/GLM 各加入 IQ4（或明确放 artifact）；保留 4-bit 与 2.5-bit operating points。

**Figure 7(a,b)：**两个模型的 budget sweep：PPL/quality 对 token/s，budgets 2–4；加入 Uniform/Rank/Full allocator。用它证明 2.5-bit 是 Pareto knee，而非 cherry-picked point。

叙述顺序：

1. Prefix Uniform4 与 GPTQ/MXFP4 的质量接近；
2. Full allocator 在多个预算上优于 Uniform/Rank；
3. 选择 2.5-bit 作为代表 operating point；
4. 任务表显示其模型/任务依赖边界。

### 8.3 End-to-End Performance and Causal Decomposition（约 0.75 页）

**Figure 7(c,d)：**把当前 Figure 7 与 Figure 9 合并：Qwen/GLM 的 throughput-vs-cache，只画 Uniform4、Dynamic2.5-Independent、PagedMoE2.5，外部 ik_llama 可用虚线/少量点。Dynamic Avg4 只在文字或机制表报告。

**Table 2：**使用前述 24 GiB 因果表：avg bits、byte hit、SSD bytes、exposed stall、total latency、tps。

叙述必须严格按：

1. Uniform → Independent：只证明 fewer requested bits；
2. Independent → Shared：相同 requests/assignments，只证明 cross-precision cache reuse；
3. hit/bytes → exposed stall → throughput：证明机制进入关键路径；
4. 第二模型、cache sweep、跨 prompt error bars：证明不是单点。

### 8.4 Necessary Conditions, Costs, and Robustness（约 0.75 页）

**Figure 8，2×2 panels：**

- (a) 全层/两模型的 progressive-vs-independent prefix penalty；
- (b) Uniform/Rank/Routing-only/Profile-only/Full allocator 的 actual output error 或 PPL；
- (c) per-bit Direct/packed/Materialize kernel latency；
- (d) mmap vs explicit 的代表点，或 SSD-bandwidth sensitivity（二选一；mmap 数值可移到小表）。

**Table 3（很小）：**checkpoint size、build/profile time、allocator latency、metadata。

结尾明确哪些是必要条件、哪些只是 supporting comparison：

- prefix penalty 小 → shared state 不以质量为代价；
- Full allocator 胜出 → before-I/O decision 有效；
- Direct 胜 Materialize且接近 packed → compactness 能延续到 compute；
- overhead 可接受 → end-to-end deployable；
- mmap/SSD sensitivity → runtime 与适用范围。

### 方案一的最终图表布局

| 位置 | 内容 | 替代当前内容 |
|---|---|---|
| Table 1 | chosen points 的多任务质量 + IQ4 | 当前 Table 1 扩充 |
| Figure 7(a,b) | 两模型 quality–throughput Pareto | 当前 Figure 2(c)/Table 2 |
| Figure 7(c,d) | 两模型 U4/Independent/Shared cache sweep | 合并当前 Figure 7 与 Figure 9 |
| Table 2 | 24 GiB causal metrics chain | 替代当前 Figure 8 |
| Figure 8(a) | 全层 prefix penalty | 替代 Figure 10(a) |
| Figure 8(b) | allocator 两信号 ablation | 替代 Figure 10(b)+Table 2 |
| Figure 8(c) | per-bit direct kernel | 替代 Figure 10(c) |
| Figure 8(d) | mmap 或 SSD sensitivity | 压缩 Figure 10(d) |
| Table 3 | offline/online overhead | 新增，但很小 |

这个结构比现稿更 solid，而且图表总面积可以不增加。

## 6. 方案二：在现有结构上最小增补

若不想重写实验章节，可按下面顺序修改。

### 6.1 §8.1 Setup：只增加必要可复现信息

增加一个紧凑段落，包含：prompt 数据集/数量/长度、empty-before-prefill、500 decode、生成参数、3 次重复、error bars、trace replay、I/O unit/queue、commit。无需新图。

### 6.2 §8.2：保留 Table 1 和 Figure 7

- Table 1 尽量加入 UD-IQ4 quality；若太宽，放一行 WikiText PPL 或把完整 IQ4 表移 artifact，正文明确引用。
- Figure 7 保持两模型 cache sweep，但曲线改成 `Uniform4 / Independent2.5 / Shared2.5 / ik_llama`。Dynamic Avg4 的 24 GiB 数值写在正文，用来说明动态 kernel/allocator overhead 小。
- 在 Figure 7 中用小 inset 或 marker 颜色加入 2/2.5/3/3.5/4 budget 的 PPL–throughput frontier；若太挤，则替换 Table 2。

### 6.3 §8.3：合并 Figure 8 与 Figure 9

- 保留 Figure 9 的两模型 throughput decomposition/capacity sweep；
- 把 Figure 8 的曲线改成一个 24 GiB 表，加入 exposed SSD stall；
- 删除 Figure 2(d)，motivation 处改成“Section 8.3 quantifies the resulting loss under identical requests”。

这样既增强因果链，又节省空间。

### 6.4 §8.4.1：替换而非新增 representation 实验

- 用 all-layer/two-model progressive penalty 替换现有 layer-30 Figure 10(a)；
- 删除 GPTQ weight-error 曲线，只保留 Table 1 的 end-to-end GPTQ 对照。不同 format 的 weight error 没有直接可比性，释放 legend 和解释文字。

### 6.5 §8.4.2：完全重做 allocator panel 和 Table 2

- Figure 10(b) 改成 Uniform/Rank/Routing-only/Profile-only/Full；budgets 至少 2/3/4；
- Table 2 改成 WikiText PPL + avg bits，保留 Full 和三个最强基线；
- 在正文加一句 calibration sample-size/seed 结果。若完整曲线放不下，正文只报 “10 vs 50 inputs 的 PPL/assignment difference”。

### 6.6 §8.4.3：保留核心结论但压缩图

- current Direct-vs-Materialize capacity curve 可以缩成 24 GiB 单点或小表；
- 腾出的 panel 改为 b=1–8 的 Direct kernel latency，若能加 packed baseline 最好；
- allocator latency 加在同一小表中。

### 6.7 §8.4.4：mmap 保留一个代表点

当前整条 total-memory curve 可缩成 29.2/32/33.27 GiB 三个代表点的表述；正文已有谨慎限定。省下的图面积用于 allocator 或 overhead，价值更高。

### 6.8 页数收支

粗略估计：

- 删除 Figure 2(d) 及重复解释：约 0.15–0.25 页；
- 合并 Figure 7 与 Figure 9：约 0.25–0.35 页；
- Figure 8 曲线改紧凑机制表：约 0.10–0.20 页；
- Direct/mmap 全曲线改代表点：约 0.10–0.20 页；
- 压缩 Section 4 重复 overview 与 §8.2 重复数字：约 0.10–0.15 页。

总计可回收约 0.7–1.1 页。新增 Pareto/allocator/overhead 即使占 0.4–0.6 页，也可以做到正文不增长，甚至略缩短。

## 7. 推荐的实际执行顺序

1. **先补 allocator ablation 与 calibration sensitivity。**这项结果可能反过来改变方法表述或 profile 选择，必须最先知道。
2. **补多 prompt/重复运行。**先在 24 GiB、三种核心配置上做，确认主结论稳定，再决定是否重跑全部 cache sweep。
3. **做 2–4 bit Pareto sweep。**它决定 2.5 是否真是最合适 headline operating point。
4. **补 all-layer/two-model prefix penalty。**主要是 offline evaluation，通常不必完整跑端到端。
5. **记录 offline/online overhead。**不复杂，但必须从真实 pipeline 计时，不能估算。
6. **补 24 GiB latency decomposition。**复用已有 instrumented runtime，把 SSD traffic 与 throughput 闭环。
7. **如果还有资源，再做 packed-kernel、IQ4 quality 和 SSD/CPU sensitivity。**

## 8. 如果只能新增很少实验

### 只能补两个实验

1. Full/Routing-only/Profile-only/Rank/Uniform + calibration-size/seed 的 allocator validation；
2. 多 prompt 的 Uniform/Independent/Shared 24 GiB 因果表，包含 byte hit、SSD bytes、exposed stall、tps 和方差。

这两项分别补齐当前最弱的算法论证和最危险的方法学缺口。

### 能补四个实验

在上面两项之外，再补：

3. 2–4 bit quality–throughput Pareto；
4. all-layer/two-model prefix penalty + offline/online overhead 小表。

完成这四项后，论文会从“核心结果强、组件证据不均匀”变为一套相当完整的系统论文 evaluation。

## 9. 最终建议

不要再把主要精力放在增加第三个模型、更多任务或更多大而全 baseline 上。当前最该做的是让每个核心句子都有同粒度的证据：

- “both routing and sensitivity matter” → 两信号 ablation；
- “random offline profiles suffice” → calibration stability；
- “prefix penalty is small” → 全层/两模型；
- “practical” → offline/online cost；
- “SSD bytes become speed” → exposed-stall causal table；
- “2.5 bit is a good operating point” → Pareto sweep；
- “end-to-end result is robust” → 多 prompt、重复和方差。

最重要的是保住并突出当前已有的 identical-request `Independent → Shared` 实验。它比额外堆一个不完全可比的外部系统 baseline 更能证明 PagedMoE 的核心创新。
