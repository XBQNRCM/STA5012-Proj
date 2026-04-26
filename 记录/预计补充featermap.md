2.2 指数多项式核特征图（Exponential Polynomial Feature Map）
思想：使用泰勒展开式，近似$$\exp\left( \frac{q^\top k}{\sqrt{d}} \right)$$，不需要随机投影

步骤：我们有泰勒展开式：$$\exp(z) = \sum_{t=0}^T \frac{z^t}{t!} + R_T$$，令其中$$z = \frac{q^\top k}{\sqrt{d}}$$，则映射关系为：
$$\phi_{\text{Poly}}(q) =
\left[
1,\ \frac{q}{\sqrt{d}},\ \frac{q^{\otimes 2}}{\sqrt{2!}d},\ \dots,\ \frac{q^{\otimes T}}{\sqrt{T!}d^{T/2}}
\right]$$
可证明有恒等式（由泰勒展开定义可知）：$$\phi(q)^\top \phi(k) = \sum_{t=0}^T \frac{(q^\top k / \sqrt{d})^t}{t!} \approx \exp\left( \frac{q^\top k}{\sqrt{d}} \right)$$

2.4 余弦归一化特征图（Cosine Normalization Feature Map）
思想：把 q/k 先做单位长度归一化，再进 Performer，消除模长干扰，让注意力只看方向相似度
$$\bar{q} = \frac{q}{\|q\|+\epsilon},\quad \bar{k} = \frac{k}{\|k\|+\epsilon}$$
结果：消除向量长度影响，只关注角度相似度（GPT-2 里 q/k 长度波动大）

2.6 带偏置的线性化特征图（Bias-Augmented Feature Map）
思想：在特征图里加一个常数偏置维度，专门拟合 exp (0) 附近的核值，大幅提升低维度近似精度
在 Performer 特征末尾拼接常数 1
$$\phi_{\text{Bias}}(q)
=
\left[
\phi_{\text{Performer}}(q),\ 1
\right]$$，有：$$\hat{K}(q,k) = \phi(q)^\top \phi(k) + 1$$

结果：专门修正小点积区域的近似误差

2.10 Learnable Kernel Feature Map（可学习核特征图）
论文：Linear Transformers with Learnable Kernel Functions (ACL 2024)
思想：Performer 用固定核 exp；改用数据驱动可学习核，自动适配高斯 / GPT-2 分布

用参数化核代替固定的指数exp核
$$K_{\theta}(q,k) = \sigma\left( \frac{q^\top k}{\sqrt{d}} \cdot \alpha + \beta \right)$$
其中，激活函数$$\sigma$$可选Swish / GELU
$$\phi_{\text{Learn}}(q) = \sigma\left( W_q \cdot q + b_q \right)$$

# 随机投影矩阵 W_q（同Performer ）
W_q = torch.randn(d, m)  # 正态分布 N(0,1)

for 层 in [1, 7]:  
# 参考project文档Part II 的 Choose any real data to your liking and extract query/key/value vectors 
# from several layers (e.g., layer 1 and 7) and heads.
    for 头 in 每个注意力头: 
        1. 抽取该层该头的 Q, K 
        2. 采样 (q,k) 对，生成 x =qᵀk/√d，y = exp(x) 
        3. 在这组数据上拟合 → 得到专属的 α_layer_head, beta_layer_head 
        4. 用这组参数构造特征图 φ(q)
        5. 计算 RelErr_ker

# 1. 抽取 GPT-2 的 Q, K
model = freeze(GPT2)
Q, K = extract_QK(model, text)

# 2. 构建拟合数据
x = (q*k).sum()/√d
y_true = exp(x)

# 3. 拟合 alpha, beta
alpha, beta = curve_fit(swish(ax+b), x, y_true) # 或者 GELU(ax+b)

# 4. 构建特征图
def φ(q):
    return swish(alpha*q/√d + beta)

def feature_map_learnable(q, W_q, alpha, b): 
    # q: [N, d] 
    proj = q @ W_q  # 线性投影 [N, m] 
    proj = proj /torch.sqrt(torch.tensor(d)) 
    z = alpha * proj + b  # 用拟合的 α, b 
    phi = z * torch.sigmoid(z)  # Swish 
    return phi  # [N, m]

# 5. 近似核
K_approx = φ(Q) @ φ(K).T

# 6. 计算误差
err = mean((K_true-K_approx)^2) / mean(K_true^2)

---

## 2026-04-26 实现与快速实验记录

### 本轮实际实现

本轮先实现两个与现有 `run_eval.py --m ...` 兼容、无需训练的 feature map：

1. `cosine_performer`
   - 对输入做 L2 归一化：
     $$x_{\text{norm}} = x / (\|x\| + \epsilon)$$
   - 再进入 Performer FAVOR+。
   - 目的：验证“只看方向相似度，削弱 GPT-2 Q/K 模长波动”的假设。

2. `bias_performer`
   - 用 `m-1` 维 Performer 随机特征 + 1 维常数通道：
     $$\phi_{\text{bias}}(x)=[\phi_{\text{performer}}(x), 1]$$
   - 目的：测试常数项是否能修正小点积区域的核值。

暂缓：

- `poly`：完整二阶/高阶张量特征会让维度从 $m$ 变成 $1+d+d^2+\cdots$，不适合当前固定 `--m` 的评测框架；若要做，应单独设计 sketch 版本。
- `learnable kernel`：需要训练 / 拟合过程，不再是纯 feature map forward，需要额外的数据拟合脚本，后续单独实现。

### 实验配置

统一比较：

- `performer`
- `power_performer(alpha=0.5)`
- `cosine_performer`
- `bias_performer`

使用：

- `m = 16,24,32,40,48,56,64,72,80`
- `n_trials = 5`
- `seed = 7`

四组快速筛选实验：

| Setting | metric | 数据规模 | 输出目录 |
|---|---|---|---|
| Gaussian | `RelErr_ker` | 10000 pairs | `outputs/gaussian_performer-power_performer-cosine_performer-bias_performer_m-16-24-32-40-48-56-64-72-80_10000pairs_5trials_seed7_alpha0.5_metric-ker/` |
| Gaussian | `RelErr_out` | 32 seq × 128 len | `outputs/gaussian_performer-power_performer-cosine_performer-bias_performer_m-16-24-32-40-48-56-64-72-80_32seq-128len_5trials_seed7_alpha0.5_metric-out/` |
| GPT-2 layer10 head6 | `RelErr_ker` | 32 docs, max_len=128 | `outputs/gpt2_performer-power_performer-cosine_performer-bias_performer_m-16-24-32-40-48-56-64-72-80_32docs_5trials_seed7_layers-10_heads-6_alpha0.5_metric-ker_pos-2/` |
| GPT-2 layer10 head6 | `RelErr_out` | 8 docs, max_len=128 | `outputs/gpt2_performer-power_performer-cosine_performer-bias_performer_m-16-24-32-40-48-56-64-72-80_8docs_5trials_seed7_layers-10_heads-6_alpha0.5_metric-out/` |

### 结果摘要（median, m=16 → m=80）

#### Gaussian `RelErr_ker`

| map | m=16 | m=80 | 观察 |
|---|---:|---:|---|
| `cosine_performer` | **0.722** | **0.720** | 最好，几乎不受 m 影响 |
| `power_performer` | 21.17 | 42.11 | 中等，仍有波动 |
| `bias_performer` | 18.40 | 256.80 | 低 m 有效，高 m 退化 |
| `performer` | 310.35 | 54.97 | 重尾波动很大 |

#### Gaussian `RelErr_out`

| map | m=16 | m=80 | 观察 |
|---|---:|---:|---|
| `cosine_performer` | **0.781** | **0.778** | 最好且稳定 |
| `bias_performer` | 1.399 | 1.575 | 低 m 较好，但随 m 变差 |
| `power_performer` | 2.422 | 2.027 | 随 m 下降 |
| `performer` | 2.655 | 2.421 | baseline |

#### GPT-2 layer10 head6 `RelErr_ker`

| map | m=16 | m=80 | 观察 |
|---|---:|---:|---|
| `cosine_performer` | **0.829** | **0.800** | 最好 |
| `bias_performer` | 0.825 | 2.104 | 高 m 明显退化 |
| `power_performer` | 0.834 | 0.884 | 低 m 好，高 m 变差 |
| `performer` | 0.999 | 0.992 | baseline 接近 1 |

#### GPT-2 layer10 head6 `RelErr_out`

| map | m=16 | m=80 | 观察 |
|---|---:|---:|---|
| `bias_performer` | **0.833** | 0.838 | 低 m 最好，但不随 m 改善 |
| `cosine_performer` | 0.841 | **0.843** | 与 bias 接近，m=80 略优于 Performer |
| `performer` | 1.203 | 0.851 | baseline，随 m 明显改善 |
| `power_performer` | 1.118 | 0.994 | 改善但不如 baseline 高 m |

### 结论

1. `cosine_performer` 是目前最值得保留的新增 feature map：
   - Gaussian `ker/out` 都显著优于 Performer；
   - GPT-2 `ker` 明显优于 Performer；
   - GPT-2 `out` 在 m=80 略优于 Performer，且远好于 power。
2. `bias_performer` 只在低 m 的 `out` 指标上有优势，高 m 经常退化；暂时不作为主方法。
3. `power_performer(alpha=0.5)` 在之前多层多头 GPT-2 out 实验中表现很好，但这次单 head 快速筛选不如 `cosine_performer` 稳定，后续需要更大规模复验。
4. 下一步建议：用 `cosine_performer` 加入完整 5000 trials / 多 layer-head 的正式对比。

### `cosine_performer` 方法简析

**动机**：原始 Performer 直接近似
$$K(q,k)=\exp(q^\top k/\sqrt d)$$
对 Q/K 模长非常敏感；当 `||q||` 或 `||k||` 偏大时，随机特征中的指数项容易产生 heavy-tail，导致 `RelErr_ker` 被少数极端样本支配。

**方法**：先做 L2 归一化，再进入 Performer：
$$\tilde x = x / (\|x\|+\epsilon),\quad \phi_{\cos}(x)=\phi_{\text{Performer}}(\tilde x)$$
这样保留方向相似度，削弱模长带来的方差爆炸。

**优势**：
- 显著降低 Gaussian 下的重尾问题；
- 对 GPT-2 Q/K 的 norm 波动更鲁棒；
- 当前快速实验中，Gaussian `ker/out` 与 GPT-2 `ker` 都明显优于 baseline。

**缺陷**：
- 不再严格近似原始 softmax kernel，而是近似角度相似度版本；
- 丢失 Q/K 模长信息，可能削弱真实 attention 中的 sharpness；
- m 增大收益有限，曲线很平，说明可能接近一个“归一化 baseline”；
- GPT-2 `out` 上优势较小，需要多 layer/head、大 trials 复验。

**一句话结论**：`cosine_performer` 是稳定性改进，不是无偏核近似改进；适合作为 improved baseline，但报告中要说明它牺牲了 norm 信息。

