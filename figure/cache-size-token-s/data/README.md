# Audited Qwen3.5 cache-size data

These CSV files are immutable snapshots of the 2026-07-14 Qwen3.5-397B-A17B
benchmark summaries used by `../plot_cache_size_ablation_figures.py`. They are
tracked here because the original `src/target/bench` results are build artifacts
and are not available in a fresh checkout.

| Snapshot | Original result | SHA256 |
|---|---|---|
| `shared_static4_20260714_104754.csv` | `src/target/bench/figure_shared_static4_20260714_104754/summary.csv` | `d897953bf07e61758eea29bab3a252fbd80d62cf53ed9acb0ab8ea093dd988f6` |
| `shared_c4l40_20260714_111629.csv` | `src/target/bench/figure_shared_c4l40_20260714_111629/summary.csv` | `35e922400e11b65ffea97ac4e8506afab748b965aec42ffc4cd8d2c1ce2f454b` |
| `shared_c2p5l25_20260714_114526.csv` | `src/target/bench/figure_shared_c2p5l25_20260714_114526/summary.csv` | `6f34fbe2d3830e7c1d6720fedd7cf63e90ce26cf1faa653b21227f4ade02e646` |
| `naive_c2p5l25_20260714_120810.csv` | `src/target/bench/figure_naive_c2p5l25_20260714_120810/summary.csv` | `210b1d932d1dfd47935062d2f755f9c611460d95b558878184dda76ca3b9c983` |
| `mmap_static4_c2p5l25_20260714_125412.csv` | `src/target/bench/figure_mmap_static4_c2p5l25_20260714_125412/summary.csv` | `7a13c60f17a2146dde6c00135c8407482f9b8d0833b2ab0b4fa6491bb11664af` |
| `shared_non_bitplane_memory_overhead_20260714_165151.csv` | `src/target/bench/memory_non_bitplane_c2p5l25_formal_20260714_165151/summary.csv` | `74d3ded11bbf994414d1bf91641925b2c08393eb2c3acfd94ef80fbb8c867ab2` |
| `ik_llama_ud_iq4_xs_expert_20260716_190707.csv` | expert-budget rows from `logs/ik_llama_iq4_xs_gpu_nonrouted_full_20260716_190707/summary.csv` | `1a9ae021643bd0fe2c57941a759a1457b23b736148b05dd8c629ffed46ed57ca` |

The plotting script verifies these hashes before reading any values. To replace
a snapshot after an intentional rerun, update the CSV, the hash in this table,
and the matching expected-hash constant in the plotting script together.

The ik_llama.cpp snapshot uses commit
`1fddd12ba861c4815a8633f14d9c5670692099cc` (build 4726) and the
Qwen3.5-397B-A17B `UD-IQ4_XS` GGUF. Its source `summary.csv` SHA256 is
`fec29841fe95228afe9aee1a0a967ee25450c14f1cc13510f60558f0239f3a95`.
Figure 1 plots `nominal_gib`, the configured routed-expert mmap budget. Each
case sets total host `MemoryMax` to this budget plus the measured fixed
1,364,856,832-byte non-routed host overhead.

For Figure 3, shared-bitplane x values are estimated as configured bitplane-pool
capacity plus a fixed 5.2 GiB. This is the paper's rounded extra-memory value,
grounded in `final_non_pool_pss_exact` from the 4/8/12-GiB formal memory runs.
mmap x values remain the measured cgroup `MemoryMax` settings.
