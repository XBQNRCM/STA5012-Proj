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

