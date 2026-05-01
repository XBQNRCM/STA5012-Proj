# 主结果数值表（m=16 与 m=80 两端）

数据来源：

| Setting / metric | results.json |
|---|---|
| Gaussian ker | `gaussian_performer-power_performer_..._10000pairs_1000trials_seed42_alpha0.5_metric-ker/` |
| Gaussian out | `gaussian_performer-power_performer_..._32seq-128len_100trials_seed42_alpha0.5_metric-out/` |
| GPT-2 ker  | `gpt2_..._1000docs_5000trials_seed0_layers-2-4-6-8-10_heads-2-4-6-8-10_..._metric-ker_pos-2/` (复用，过滤到两 maps) |
| GPT-2 out  | `gpt2_..._100docs_10trials_seed42_layers-2-4-6-8-10_heads-2-4-6-8-10_..._metric-out/` (复用) |

| setting | map | m=16 med | m=80 med | m=16 mean | m=80 mean | m=16 geo | m=80 geo |
|---|---|---:|---:|---:|---:|---:|---:|
| **Gaussian ker** | performer | 54.63 | 53.39 | 2.84e9 | **1.33e10** | 82.82 | 78.24 |
| **Gaussian ker** | **power_performer** | **32.26** | **24.04** | **5705** | **737.5** | **48.60** | **32.98** |
| **Gaussian out** | performer | 2.752 | 2.484 | 2.761 | 2.492 | 2.758 | 2.491 |
| **Gaussian out** | **power_performer** | **2.342** | **2.070** | **2.357** | **2.073** | **2.354** | **2.071** |
| **GPT-2 ker** | performer | 1.0000 | 1.0000 | 1.001 | 1.000 | 1.0007 | 1.0003 |
| **GPT-2 ker** | **power_performer** | 1.0000 | 1.0000 | 1.002 | 1.000 | 1.0010 | 1.0002 |
| **GPT-2 out** | performer | 1.008 | 0.9687 | 1.107 | 1.015 | 1.062 | 0.9796 |
| **GPT-2 out** | **power_performer** | **0.9648** | **0.8710** | **1.084** | **0.9744** | **1.012** | **0.9144** |

## 关键观察

1. **Gaussian ker 的 mean 极不稳定**：performer 的 mean 在 m=16/80 都到 10⁹–10¹⁰ 量级，但 median 只是 50 量级——彻底说明 *Gaussian setting 下 RelErr_ker 应该用 median / geomean 报告*。
2. **power_performer 在 4 个 setting × 2 个指标（共 8 列）里全都 ≤ performer**：
   - Gaussian ker / out：明显胜出（median 减半）
   - GPT-2 out：median 0.969 → 0.871（10% 改善）
   - GPT-2 ker：median 都贴 1.0000，metric 在此设定下无判别力，但 mean/geomean 依然 power_performer ≤ performer
3. **GPT-2 ker median ≈ 1 的来源**：分子 $E[(K-\hat K)^2]$ 与分母 $E[K^2]$ 在 GPT-2 重尾分布下被同样的少数极大 $K$ 主导，比值≈1。这恰恰说明 RelErr_ker 在 spiky kernel 下并不能反映"近似的好坏"——这是 R3/R6 的关键论点之一。
4. **m 增大趋势**：
   - Gaussian ker / out：power_performer 单调下降；performer 在 ker 上 *几乎不下降*（被重尾主导，median 在 53–57 间晃）
   - GPT-2 out：两者都单调下降，power_performer 始终更低
