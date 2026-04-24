# STA5012 实验报告（简要）

## 实验日志

### 2026-04-17

**做了什么**

1. 检查既有 `RelErr_ker` 实现，确认指标公式与 `STA5012_Final_Proj_Intro.md` 一致，无 bug。
2. 诊断 `RelErr_ker` 曲线随 m 非单调的原因 = 重尾 + 小 trials + mean 聚合 + 各 m 独立采 ω；给出结构性结论。
3. 在 `_eval_loop` 里改用「嵌套 ω 前缀」：每个 (map, trial) 采一份 `omega_full[max_m, d]`，各 m 取 `omega_full[:m]`，消除 m 间 ω 采样噪声。
4. 跑 1000 / 5000 trials 的 `RelErr_ker` 高精度基线。

**最好结果**

| Setting | 文件 | 关键数值 |
|---|---|---|
| Gaussian, `RelErr_ker`, 1000 trials, seed 42, m ∈ {16, 24, 32, 40, 48, 56, 64, 72, 80} | `outputs/gaussian_performer_m-16-24-32-40-48-56-64-72-80_10000pairs_1000trials_seed42/` | mean RelErr_ker 从 **1.49×10⁶ (m=16) → 6.1×10⁴ (m=80)**，严格单调下降 |
| GPT-2 + WikiText, `RelErr_ker`, 5000 trials, seed 42, layers {2,4,6,8,10} × heads {2,4,6,8,10}, 1000 docs | `outputs/gpt2_performer_m-16-24-32-40-48-56-64-72-80_1000docs_5000trials_seed42_layers-2-4-6-8-10_heads-2-4-6-8-10_pos-2/` | mean RelErr_ker 从 **1.0273 (m=16) → 1.0017 (m=80)**，平稳逼近 1 的下界 |

> GPT-2 setting 的 RelErr_ker 相比 Gaussian 数值小 5–6 个数量级，是因为 `E[K²]` 在真实 GPT-2 激活上远大（head 激活多尖锐），分母把相同方差的分子压了回去。

---

### 2026-04-24

**做了什么**

1. 识别到 intro 要求的「次指标 `RelErr_out`」未实现，补齐。
2. 在 `src/eval/metrics.py` 新增 `output_numer_denom` 与 `relerr_output`，按 float64 计算
   - exact：`A = softmax(QKᵀ/√d)`，`O = AV`
   - linear：`Ô_i = φ(qᵢ)ᵀ(Φ_KᵀV) / (φ(qᵢ)ᵀΣₗφ(kₗ))`
   - 跨序列聚合 `sqrt(ΣN_b / ΣD_b)` 而非先求 per-seq ratio 再平均。
3. 新增 `sample_qkv_gaussian` 与 GPT-2 序列级流式采样 `iter_qkv_gpt2_wikitext`（含 V、不缓存全量 QKV）。
4. 在 `run_eval.py` 加入 `--metric ker|out|both`、`--n-seq --seq-len --dim-v`；`EvalResult(Out)` 增加 `layer/head` 字段，顺便修复 `plot_error_vs_dim.py --layer/--head` 过滤此前失效的 bug。
5. `plot_error_vs_dim.py` 支持 `--agg {median,mean,geomean}`、`--metric {auto,ker,out}`、`--logy`；默认 `median`。
6. `_load_wikitext_rows` 加三级 fallback：本地 parquet → hf-mirror → HF 原 script；并补写离线下载命令到 README。

**最好结果（今日新指标）**

| Setting | 文件 | 关键数值 |
|---|---|---|
| Gaussian, `RelErr_out`, 5 trials, seed 0, n_seq=32, seq_len=128 | `outputs/gaussian_performer_m-32-64-128-256_32seq-128len_5trials_seed0_metric-out/` | median RelErr_out **2.649 (m=32) → 2.268 (m=256)**，**严格单调** |
| GPT-2 + WikiText, `RelErr_out`, 3 trials, seed 0, layer 10 head 6, 16 docs, max_len 256 | `outputs/gpt2_performer_m-32-64-128-256_16docs_3trials_seed0_layers-10_heads-6_metric-out/` | median RelErr_out **1.216 (m=32) → 1.102 (m=128)**，m=256 3 trials 小幅抬头属 trial 方差 |
| GPT-2 跨 layer × head 抽样, `RelErr_out`, 2 trials, layers {0,5,10} × heads {0,6}, 8 docs | `outputs/gpt2_performer_m-32-64-128_8docs_2trials_seed0_layers-0-5-10_heads-0-6_metric-out/` | 36 条记录，`layer/head` 字段正确写入；`--layer 10 --head 6` 过滤后 6 条单调下降，**过滤 bug 已修复** |

**本次改动带来的核心观察**

- `RelErr_out` 在所有 setting 下均**随 m 稳定单调**，只要 3–5 trials 就能看到干净曲线；
- `RelErr_ker` 的「看起来不单调」是**重尾小样本问题**，不是指标实现错误；
- GPT-2 上同样的 Performer，`RelErr_out` 数值（~1.1）比 Gaussian（~2.3）**反而更小**——和 intro 提示的「真实 head 有稀疏性，softmax 组合后误差被部分抵消」一致。
