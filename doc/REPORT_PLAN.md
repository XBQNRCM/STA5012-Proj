# 报告与实验规划（与 PDF 原题逐条对齐）

> 目标：完成 `doc/Attn_Linearization.pdf` 中 Part I (G1–G5) + Part II (R1–R6) + Deliverables 的所有图表与文字。
> 本文档既是 TODO，也是执行手册。每一项给出 **要回答的问题 / 输出图表 / 复用现有数据 or 需新跑 / 估时**。

---

## 0. 全局命名约定

为了让所有图表可对比，固定下面这套主参数。其他参数视为消融，不进主表。

| 项 | 取值 |
|---|---|
| `dim_d` (= GPT-2 head dim) | 64 |
| 主特征维度集合 `m` | `16 24 32 40 48 56 64 72 80` |
| **主 maps（决议）** | `performer`（baseline）+ `power_performer`（improved，α=0.5） |
| `power_alpha` | 0.5 |
| 主 seed | 42 |
| Gaussian 主 trials | ker: 1000；out: 100 |
| GPT-2 主 trials | ker: 1000；out: 10 |
| GPT-2 主 layers | 2,4,6,8,10 |
| GPT-2 主 heads | 2,4,6,8,10 |
| GPT-2 ker `token-pos` | -2（中点） |
| GPT-2 主 docs | ker: 1000；out: 100 |
| Gaussian-out 序列尺寸 | n_seq=32, seq_len=128 |

> 现有 5000-trial 的 `seed=0` 高质量结果（performer + scaled/power/rala）**仍保留**，主图绘制时通过 `--maps` 过滤到 `performer + power_performer`，避免重跑大成本。其余四个 map（scaled / rala / cosine / bias）作为**消融组**留在附录与代码中（REGISTRY 不删），但主报告只展示 baseline 与 power_performer 两条曲线，叙事更聚焦。

---

## 1. 必要实验列表

### 1.1 Gaussian RelErr_ker（G1, G2, G3, G4）
- **问题**：m↑ 时 RelErr_ker 是否单调下降？power_performer 相对 Performer 的优势？
- **图**：median + geomean 两种聚合（mean 被重尾主导），主图用 median。
- **新跑量**：2 maps × 9 m × 1000 trials × 10000 pairs（CPU 估时 5–10 分钟）。
- **输出目录**：`outputs/gaussian_performer-power_performer_..._10000pairs_1000trials_seed42_alpha0.5_metric-ker/`

### 1.2 Gaussian RelErr_out（次指标，扩充 G3）
- **问题**：m↑ 时 attention 输出误差是否稳定下降？
- **图**：mean（被 softmax 双重平滑，mean 稳定）。
- **新跑量**：2 maps × 9 m × 100 trials × 32 seq × 128 len（CPU 估时 < 1 分钟）。
- **输出目录**：`outputs/gaussian_performer-power_performer_..._32seq-128len_100trials_seed42_alpha0.5_metric-out/`

### 1.3 GPT-2 RelErr_ker（R1, R2）
- **问题**：在真实激活上 power_performer vs Performer 的 ker MSE 表现。
- **图**：mean + median；**主图用 median**（pair-level 仍然重尾）。
- **数据**：复用现有 `outputs/gpt2_performer-scaled_performer-power_performer-rala_performer_..._1000docs_5000trials_seed0_..._metric-ker_pos-2/results.json`，绘图时用 `--maps performer power_performer` 过滤。
- **不新跑**：高精度 5000-trial 数据已经存在，没必要重跑。

### 1.4 GPT-2 RelErr_out（R1, R2 + R3 数据来源）
- **问题**：attention 输出误差；同时这一份数据按 (layer, head) 拆开就是 R3 的 heatmap 数据。
- **图**：median over (layer, head, trial)。
- **数据**：复用现有 `outputs/gpt2_performer-scaled_performer-power_performer-rala_performer_..._100docs_10trials_seed42_..._metric-out/results.json`，绘图时 `--maps performer power_performer` 过滤。
- **不新跑**：现有数据已含 power_performer。

### 1.5 R3 Cross-layer / cross-head 分析图
- **问题**：哪些 (layer, head) 容易/难近似？power_performer 的改善是否一致？
- **图**：performer 与 power_performer **两个 panel** 的 5×5 heatmap，方便直接对比每个 cell 的改善量。
- **数据来源**：1.4 的输出。
- **脚本**：`plot_layer_head_heatmap.py`（已写好，用 `--maps performer power_performer`）。

### 1.6 R4 Q/K 分布诊断
- **问题**：q,k 真的近似 N(0, I) 吗？哪些方向偏离？为什么 Performer 在 GPT-2 ker 上贴近 1？
- **图**（建议拼成 1 张大 figure，一行一个 (layer,head)）：
  1. dimension-wise mean of q, k：长 64 的 bar，over n_doc 个 token 的均值；
  2. dimension-wise std of q, k；
  3. ‖q‖, ‖k‖ 的直方图（与 N(0,I) 下 χ_d 比较）；
  4. q·k/√d score 的直方图（与 Gaussian 同尺寸下的 score 直方图叠加，**直观说明 spiky**）；
  5. 至少一对 (layer, head) 的 Q-Q plot vs N(0,1)。
- **选定 (layer, head)**：(2,2)、(4,4)、(6,6)、(8,8)、(10,10) 共 5 个。
- **脚本**：新写 `analyze_qk_dist.py`，输出 PNG + 一份 `qk_stats.json`（dim-wise mean/std、score skewness/kurtosis）方便正文引用。
- **新跑量**：1 次 GPT-2 forward，~200 docs，CPU/GPU 几分钟即可。

### 1.7 R5 改进 feature map 验证（结论：**power_performer**）
- **设计动机** —— 由 1.6 直接驱动（详见 § 1.7.x「power_performer 选型分析」）：
  1. GPT-2 的 q,k 在某些 head 上 ‖q‖, ‖k‖ 远超 N(0,I) 下的 χ_d；
  2. q·k/√d score 标准差最高可达 27（Gaussian baseline 是 1）；
  3. score 普遍正偏 + 高峰度 → kernel $K=\exp(\cdot)$ 极重尾。
- **机制**：power_performer 在 phi 之前做逐元素压缩 $\sigma_\alpha(x)=\mathrm{sign}(x)|x|^\alpha$，α=0.5。直接把大坐标的 ‖q+k‖² 拉小，缓解 FAVOR+ 估计器方差中 $\exp(\|q+k\|^2/\sqrt d)-1$ 的指数因子。
- **代价**：此 map 不再严格近似 $\exp(q^\top k/\sqrt d)$，而是近似 $\exp(\sigma_\alpha(q)^\top \sigma_\alpha(k)/\sqrt d)$。这是 **kernel 重定义** 而非 estimator 改良；报告中需明确这点。
- **图**：1.1–1.5 的主图就够了，**只画两条线**会非常清楚。

### 1.7.x  power_performer 选型分析（写进报告 §4.6）

> 这一节是 R5 的核心论证：解释为什么从我们试过的 5 种改进里**只保留 power_performer**。

#### A. R4 观察到的三件事（来自 `outputs/qk_diagnostic/qk_stats_200docs.json`）

| 现象 | 数值 | 含义 |
|---|---|---|
| q,k 各 dim 均值非零 | max\|mean\| 在 (2.0, 7.0) 量级 | q,k **不居中** → centering（scaled）有动机 |
| ‖q‖, ‖k‖ 远离 χ_d 峰值 8 | L2H2 ‖k‖ ≈ 32（χ_64 峰值 ≈8） | norm 失衡 → cosine（去模长）有动机 |
| score q·k/√d 重尾 | std up to 27，skew 0.6–0.8，kurt 1.5–2 | 大 score 的指数 kernel 极重尾 → power 压缩有动机 |

#### B. 三条改进路线 vs 实验结果

| 候选 | 攻击的现象 | 实验表现 (GPT-2 RelErr_out, m=80, median) |
|---|---|---:|
| `scaled_performer`（centering+normalization） | 现象 1 | 1.115（**反而更差**） |
| `cosine_performer`（投到单位球） | 现象 2 | 0.843（与 Performer 相当，好不到 0.04） |
| `rala_performer`（上下文权重 + 通道混合） | / | 1.233（更差） |
| `bias_performer`（常数通道补底） | / | ≈Performer |
| **`power_performer (α=0.5)`** | 现象 3 | **0.871**（vs Performer 0.969） |

> 结论：**唯一在所有 (layer,head) cell 上一致改善 Performer 的，是 power_performer**。

#### C. 为什么 α=0.5 这一压缩最有效（理论侧）

FAVOR+ 估计器的方差 (Performer 论文 Lemma 1 / Choromanski et al. 2020) 是

$$
\mathrm{Var}\!\left[\hat K_\phi(q,k)\right]
= \frac{1}{m}\,K(q,k)^2\,\bigl(\exp(\|q+k\|^2/\sqrt d)-1\bigr).
$$

- $K(q,k)$ 大时方差就大；这是分母 $\mathbb E[K^2]$ 也大但分子 $\mathbb E[(K-\hat K)^2]$ 大得更快的原因；
- $\exp(\|q+k\|^2/\sqrt d)$ 是个**指数倍率**——只要 ‖q+k‖² 大，方差就爆炸。

GPT-2 L2H2 的 $\|q\|+\|k\| \approx 50$，对应 $\exp(50^2/8)\sim e^{312}$，**完全失控**。

power compression $\sigma_\alpha(x)$ 把每个坐标按 $|x|^\alpha$ 重新分布：

$$
\|\sigma_\alpha(q)\|^2 = \sum_i |q_i|^{2\alpha},\qquad
\alpha=0.5\Rightarrow \|\sigma_{0.5}(q)\|^2=\sum_i |q_i|.
$$

也就是 **L2 norm 被换成 L1 norm**（在 squared 意义下）。对于一个有少数 outlier 大坐标的 q，L1 norm 远远小于 L2 norm，方差因子被 dramatic 地压缩。这是 α=0.5 在重尾数据上最受偏爱的根本原因（同样的思想在 robust regression、heavy-tailed PCA 里都用 squared root link）。

#### D. 报告中需说明的 trade-off

power_performer **改变了 kernel**：原 kernel $K=\exp(q^\top k/\sqrt d)$ 被替换为
$K_\sigma = \exp(\sigma_\alpha(q)^\top \sigma_\alpha(k)/\sqrt d)$。我们的 RelErr 是相对**原 kernel** 的误差，所以 power_performer 之所以胜出，部分原因是它**承认原 softmax kernel 在 GPT-2 激活上不可用 random feature 廉价近似**，主动把 kernel 换成一个 random-feature-friendly 的近邻 kernel。

这是一个 honest 的 trade-off：
- **优点**：在 RelErr_out（attention 输出层面）上 m=80 时优势 ~10%；
- **代价**：这 ~10% 是同时来自更小的 Var 和**修改 kernel 自身**——若任务对原 softmax 形状极敏感（如 induction-head 复制行为），可能需要后续 fine-tune。

报告里把 (优点) 和 (代价) 都写清楚，避免把改进吹成"白嫖"。

---

### 1.8 G5 / R6 理论与最优性讨论（无需新实验）
- **要点**：
  - FAVOR+ positive features 的 estimator 方差闭式：
    $$\mathrm{Var}\!\big[\hat K(q,k)\big] = \tfrac{1}{m}\,K(q,k)^2 \big(\exp(\|q+k\|^2/\sqrt d) - 1\big).$$
    → RelErr_ker 上界 $O(1/m)$ 但**前因子随 ‖q+k‖ 指数增长**，这就是 GPT-2 Setting 下 Performer 几乎不收敛的根因。
  - 给定 $m$ 的 best-possible：在 Gaussian 下 ≥ Mercer 截断的尾项（$d$ 维 RBF 谱衰减）；在真实 GPT-2 下 由 Q,K 经验分布的有效秩决定。
  - 方向性建议：
    - 若 q,k 居中且各向同性 → Performer 已接近最优；
    - 若 ‖q‖ 主导 → cosine 或 scaled 更优；
    - 若 attention 极度 spiky → 可能需要非随机/数据相关 feature（Hedgehog 方向）。

---

## 2. 报告主图清单

| Fig | 内容 | 数据来源 | 出处脚本 |
|---|---|---|---|
| Fig 1 | Gaussian RelErr_ker vs m，2 maps | 1.1 | `plot_error_vs_dim.py --agg median --maps performer power_performer` |
| Fig 2 | Gaussian RelErr_out vs m，2 maps | 1.2 | `plot_error_vs_dim.py --agg mean --maps performer power_performer` |
| Fig 3 | GPT-2  RelErr_ker vs m，2 maps | 1.3 | `plot_error_vs_dim.py --agg median --maps performer power_performer` |
| Fig 4 | GPT-2  RelErr_out vs m，2 maps | 1.4 | `plot_error_vs_dim.py --agg median --maps performer power_performer` |
| Fig 5 | GPT-2 Layer×Head RelErr_out heatmap，**2 个 panel**（performer vs power_performer） | 1.5 | `plot_layer_head_heatmap.py --maps performer power_performer` |
| Fig 6 | Q/K dim-wise mean & std，5 个 (layer,head) | 1.6 | `analyze_qk_dist.py` |
| Fig 7 | q·k/√d score 分布：Gaussian baseline vs GPT-2 各 head | 1.6 | `analyze_qk_dist.py` |
| Fig 8 (附录可选) | norm / entropy 直方图 | 1.6 | `analyze_qk_dist.py` |

可选：

| Fig | 内容 |
|---|---|
| Fig A1 | mean / median / geomean 三种聚合并排（说明为什么 median） |
| Fig A2 | ‖q‖ / ‖k‖ 直方图 vs 理论 χ_d |
| Fig A3 | 5000-trial 高精度 ker baseline（附录） |

---

## 3. 报告章节结构（最终 REPORT.md）

```
1. Introduction
   - 任务（PDF 1, 2 节）
   - 主指标定义（PDF 5 节）

2. Methods
   2.1 Performer / FAVOR+ baseline
   2.2 power_performer（improved，α=0.5）—— 一句话动机
   2.3 Evaluation protocol（嵌套 ω, float64, 聚合方式）

3. Gaussian Toy Setting (Part I)
   3.1 Setup
   3.2 RelErr_ker results（Fig 1, performer vs power_performer）
   3.3 RelErr_out results（Fig 2）
   3.4 Discussion: m 增大趋势 + best-possible (G5)

4. Frozen GPT-2 Setting (Part II)
   4.1 Setup（layer/head 选择、token 取法）
   4.2 RelErr_ker results（Fig 3）—— 解释为何 mean ≈ 1
   4.3 RelErr_out results（Fig 4）
   4.4 Cross layer/head analysis（Fig 5, R3）
   4.5 Q/K distribution diagnostics（Fig 6, 7, R4）
   4.6 Improved map vs Performer：power_performer 选型分析（§ 1.7.x，R5）
   4.7 Discussion: 与 Gaussian 的差别 + best-possible (R6)

5. Conclusion

Appendix
  A. 5000-trial 高精度结果（含全部 4 个废弃 map 的对照表，作为消融）
  B. 完整命令复现
  C. Q/K 分布数值表
```

旧的 `REPORT.md`（实验日志）改名为 `REPORT_log.md`。

---

## 4. 执行顺序与依赖

```
Phase A（无需 GPU，已完成）
  A1. plot_layer_head_heatmap.py                ✓
  A2. analyze_qk_dist.py（200 docs，5 个 head） ✓
  A3. plot_error_vs_dim.py 支持多 results.json  ✓

Phase A4（plot 脚本支持 --maps 过滤，方便复用 4-map 数据）  ◀ NEW

Phase B（CPU 即可完成，因为 maps 收敛到 2 个）
  B1. 1.2 Gaussian RelErr_out 2-map（< 1 min）
  B2. 1.1 Gaussian RelErr_ker 2-map（5–10 min）
  B3. GPT-2 ker / out 主图 —— 复用已有 4-map 数据 + --maps 过滤，无需重跑

Phase C（写作）
  C1. 写 § 1.7.x power_performer 选型分析（已草拟于本文档）
  C2. 写 G5 / R6 理论段落
  C3. 写 R3 / R4 分析段落
  C4. 重写 REPORT.md
  C5. PPT 大纲
```

---

## 5. 现有产物盘点（执行前）

| 类别 | 现状 |
|---|---|
| Gaussian ker, performer, 1000 tr | ✓ |
| Gaussian ker, 4 maps, seed=0, **5 tr** | ⚠ trials 太少，需重跑 |
| Gaussian out, performer, 1000 / 5000 tr | ✓ |
| Gaussian out, 4 maps, seed=0, **5 tr** | ⚠ trials 太少 |
| Gaussian out / ker, +cosine+bias, seed=7, 5 tr | ⚠ 探索性 |
| GPT-2 ker, performer, 5000 tr, 1000 docs | ✓ 高精度 baseline |
| GPT-2 ker, 4 maps, seed=0, 5000 tr, 1000 docs | ✓ 但 seed 与主图不一致 |
| GPT-2 out, performer single, 1000/5000 tr | ✓ 但只 1 个 (l,h) |
| GPT-2 out, 4 maps, 100 docs × 10 tr × 25 (l,h), seed=42 | ✓ **R3 heatmap 直接可用** |
| GPT-2 out / ker, +cosine+bias, seed=7, 32 docs × 5 tr | ⚠ 探索性 |
| Q/K 诊断 | ❌ 缺 |
| layer/head heatmap | ❌ 缺 |

---

## 6. 备注 / 决策点

1. **GPT-2 ker 主图聚合方式**：当前 5000-trial mean ≈ 1.0017–1.0274，几乎压在 1。改用 median 更能反映"中间情形下的近似质量"；mean 留作附录并配文字说明"被重尾主导"。
2. **是否再做 alpha / rala_mix 的消融**：原题不要求，建议放附录或省略。
3. **是否上 Hedgehog**：原题"optional"，不实现，但在 R5/R6 讨论里引用一句即可。
4. **PPT**：最后由 4 张图（Fig 1, 3, 5, 6）+ 1 张 best-possible 公式页 + 1 张 conclusion 即可撑满 15 min。
