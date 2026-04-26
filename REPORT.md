# STA5012 实验报告（简要）

## 实验日志

### 2026-04-17

**做了什么**

1. 检查既有 `RelErr_ker` 实现，确认指标公式与 `STA5012_Final_Proj_Intro.md` 一致，无 bug。
2. 诊断 `RelErr_ker` 曲线随 m 非单调的原因 = 重尾 + 小 trials + mean 聚合 + 各 m 独立采 ω；给出结构性结论。
3. 在 `_eval_loop` 里改用「嵌套 ω 前缀」：每个 (map, trial) 采一份 `omega_full[max_m, d]`，各 m 取 `omega_full[:m]`，消除 m 间 ω 采样噪声。
4. 跑 1000 / 5000 trials 的 `RelErr_ker` 高精度基线。

**最好结果**

| Setting | 指标 | trials | 数据规模 | 图 |
|---|---:|---:|---|---|
| Gaussian | `RelErr_ker` mean | 1000 | 10000 pairs, seed 42 | `outputs/gaussian_performer_m-16-24-32-40-48-56-64-72-80_10000pairs_1000trials_seed42/relerr_kernel_vs_dim_mean.png` |
| GPT-2 + WikiText | `RelErr_ker` mean | 5000 | 1000 docs, layers {2,4,6,8,10}, heads {2,4,6,8,10} | `outputs/gpt2_performer_m-16-24-32-40-48-56-64-72-80_1000docs_5000trials_seed42_layers-2-4-6-8-10_heads-2-4-6-8-10_pos-2/error_vs_dim.png` |

**最好数值**

- Gaussian `RelErr_ker`: **1.489e6 (m=16) → 6.388e4 (m=80)**，1000 trials 后 mean 曲线严格下降。
- GPT-2 `RelErr_ker` 5000 trials: **1.0273 (m=16) → 1.0017 (m=80)**，整体接近 1。

## 2026-04-24

**做了什么**

1. 补齐 intro 里要求的 `RelErr_out = ||AV - AhatV||_F / ||AV||_F`。
2. 新增 `sample_qkv_gaussian`、`iter_qkv_gpt2_wikitext`，支持序列级 Q/K/V 评测。
3. `run_eval.py` 新增 `--metric ker|out|both`。
4. `plot_error_vs_dim.py` 新增 `--agg mean|median|geomean` 与 `--layer/--head` 真过滤。

## 2026-04-25

**做了什么**

1. 用本地环境补跑 1000 trials 的 `RelErr_out`。
2. 为两种 setting、两种指标统一保存 mean 曲线图片。
3. 整理 `outputs/`：删除小样本旧结果，只保留 1000-trial 主结果和一个 5000-trial 高质量参考结果。
4. 优化 `RelErr_out` 运行：exact `O=AV` 对同一批 Q/K/V 只计算一次，避免 1000 trials 重复算 exact softmax attention。

**1000 trials 主结果**

| Setting | 指标 | 数据规模 | m=16 | m=80 | 图 |
|---|---:|---|---:|---:|---|
| Gaussian | `RelErr_ker` mean | 10000 pairs | 1.489e6 | 6.388e4 | `outputs/gaussian_performer_m-16-24-32-40-48-56-64-72-80_10000pairs_1000trials_seed42/relerr_kernel_vs_dim_mean.png` |
| Gaussian | `RelErr_out` mean | 8 seq × 64 len | 2.2550 | 2.0321 | `outputs/gaussian_performer_m-16-24-32-40-48-56-64-72-80_8seq-64len_1000trials_seed42_metric-out/relerr_output_vs_dim_mean.png` |
| GPT-2 + WikiText | `RelErr_ker` mean | 400 docs, layers {2,4,6,8,10}, heads {2,4,6,8,10} | 1.0026 | 1.0078 | `outputs/gpt2_performer_m-16-24-32-40-48-56-64-72-80_400docs_1000trials_seed42_layers-2-4-6-8-10_heads-2-4-6-8-10_pos-2/relerr_kernel_vs_dim_mean.png` |
| GPT-2 + WikiText | `RelErr_out` mean | 8 docs, layer 10 head 6 | 1.1968 | 1.0739 | `outputs/gpt2_performer_m-16-24-32-40-48-56-64-72-80_8docs_1000trials_seed42_layers-10_heads-6_metric-out/relerr_output_vs_dim_mean.png` |

**观察**

- `RelErr_out` 在 Gaussian 与 GPT-2 两种 setting 下都随 m 稳定下降，比 `RelErr_ker` 更适合展示 m 增大带来的改善。
- Gaussian `RelErr_ker` 即使用 1000 trials，数值仍远大于 `RelErr_out`，说明 pair-level kernel MSE 被极端样本主导。
- GPT-2 `RelErr_ker` 在 1000 trials 下接近 1，但 mean 曲线仍有轻微波动；5000 trials 版本更平滑，建议作为最终报告参考图。

## 2026-04-26

### 本次新增 feature maps

- `scaled_performer`：对输入做逐维经验标准化，再进入 Performer feature。
- `power_performer (alpha=0.5)`：对输入做幂压缩 `sign(x) * |x|^alpha` 再进入 Performer。
- `rala_performer (mix=0.5)`：RALA-inspired：`phi_performer(x)` 乘上下文权重 `gamma(x)`，再做轻量通道混合。

### 四组实验汇总（m=16 → 80）

> 说明：下表的 `median`/`mean` 都是直接对 `results.json` 中同一 `(feature_map, dim_m)` 的所有记录聚合。
> `GPT-2 / out` 的记录包含 25 个 `(layer,head)` 组合与 10 个 trials，总计每个 m 每个 map 为 250 条记录。

#### GPT-2 + WikiText：`RelErr_ker`（1000 docs × 25 heads/layers × 5000 trials，pos=-2）

目录：`outputs/gpt2_performer-scaled_performer-power_performer-rala_performer_m-16-...-80_1000docs_5000trials_seed0_layers-2-4-6-8-10_heads-2-4-6-8-10_alpha0.5_rala0.5_metric-ker_pos-2/`

- **median**：四种 map 在 m=16 与 m=80 上都几乎是 **1 → 1**（到小数点后 6 位仍接近 1）。
- **mean（m=16 → 80）**：
  - `scaled_performer`: **1.00003 → 1.00001**（最好）
  - `rala_performer`: **1.00006 → 1.00005**
  - `power_performer`: **1.00212 → 1.00025**
  - `performer`: **1.00090 → 1.00034**

结论：在这个 GPT-2 ker 设定下，所有方法都非常接近 1；`scaled_performer` 的 mean 最低但优势极小（~1e-4 量级）。

#### GPT-2 + WikiText：`RelErr_out`（100 docs × 25 heads/layers × 10 trials）

目录：`outputs/gpt2_performer-scaled_performer-power_performer-rala_performer_m-16-...-80_100docs_10trials_seed42_layers-2-4-6-8-10_heads-2-4-6-8-10_alpha0.5_rala0.5_metric-out/`

- **median（m=16 → 80）**：
  - `power_performer`: **0.96480 → 0.87103**（最好）
  - `performer`: **1.00775 → 0.96870**
  - `scaled_performer`: **1.20869 → 1.11465**
  - `rala_performer`: **1.24231 → 1.23288**

结论：在 GPT-2 out 上，`power_performer(alpha=0.5)` 明显优于其它 map，并随 m 增大持续下降。

#### Gaussian：`RelErr_ker`（10000 pairs × 5 trials）

目录：`outputs/gaussian_performer-scaled_performer-power_performer-rala_performer_m-16-...-80_10000pairs_5trials_seed0_alpha0.5_rala0.5_metric-ker/`

- **median（m=16 → 80）**：
  - `rala_performer`: **31.53 → 22.05**（最好）
  - `power_performer`: **25.20 → 24.78**
  - `scaled_performer`: **20.78 → 63.04**
  - `performer`: **56.56 → 344.34**

结论：Gaussian ker 依旧重尾，median 更可靠；在这组实验中 `rala_performer` 的 median 最小且随 m 下降。

#### Gaussian：`RelErr_out`（32 seq × 128 len × 5 trials）

目录：`outputs/gaussian_performer-scaled_performer-power_performer-rala_performer_m-16-...-80_32seq-128len_5trials_seed0_alpha0.5_rala0.5_metric-out/`

- **median（m=16 → 80）**：
  - `power_performer`: **2.314 → 2.013**（最好）
  - `rala_performer`: **2.408 → 2.081**
  - `scaled_performer`: **2.694 → 2.522**
  - `performer`: **2.861 → 2.529**

结论：Gaussian out 上 `power_performer(alpha=0.5)` 最好，且随 m 单调下降最稳定；`rala_performer` 次之。
