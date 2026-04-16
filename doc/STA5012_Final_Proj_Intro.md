# STA5012 Final Project

**概要**
> **我们想用一个低维特征映射（feature map）来近似 softmax attention 里的 kernel，从而把 attention 的计算从二次复杂度降到线性复杂度，并比较不同近似方法在 toy Gaussian 数据和真实 GPT-2 激活上的效果。**

---

## 1. self-attention 的瓶颈

### 1.1 Self-attention 的基本形式

在 transformer 里，每个位置 $i$ 都会有三个向量：

- query: $q_i \in \mathbb{R}^d$
- key: $k_i \in \mathbb{R}^d$
- value: $v_i \in \mathbb{R}^{d_v}$

对于第 $i$ 个 token，attention 会把它和前面所有位置 $j \le i$ 的 key 做相似度比较，然后对对应的 value 做加权平均。

在 **causal attention**（自回归）下，第 $i$ 个位置的输出是：


$$
o_i=
\frac{\sum_{j\le i}\exp(q_i^\top k_j/\sqrt d)\, v_j}
{\sum_{j\le i}\exp(q_i^\top k_j/\sqrt d)}.
$$

这里最关键的相似度函数是

$$
K(q,k)=\exp(q^\top k/\sqrt d),
$$

它就是 **softmax attention kernel**。



### 1.2 为什么复杂度是二次的？

如果序列长度是 $n$，那么每个 query $q_i$ 都要和所有可见的 key $k_j$ 计算一次内积：

- 第 1 个位置算 1 次
- 第 2 个位置算 2 次
- ...
- 第 $n$ 个位置算 $n$ 次

这些一共有大约 $\frac{n(n+1)}{2} = O(n^2)$ 个 query-key interaction。

如果写成矩阵形式，就是构造一个 $n\times n$ 的 attention score matrix：

$$
S_{ij} = \frac{q_i^\top k_j}{\sqrt d},
\qquad
A_{ij} = \mathrm{softmax}(S_{ij}).
$$

所以：

- **时间复杂度**：通常是 $O(n^2 d)$
- **内存复杂度**：通常也是 $O(n^2)$，因为要存 attention matrix 或其中间结果



### 1.3 瓶颈在哪里？

你通常需要形成或者至少隐式处理这个 $n\times n$ 的矩阵 $A$，所以时间和内存都会随着 $n^2$ 增长。当 $n$ 很长时，这个 $n^2$ 会变得非常昂贵。例如：

- 短文本时还能接受
- 长上下文（几千到几万 token）时，attention 的计算和显存就会成为主要瓶颈

所以，大家一直在寻找办法：

> **能不能不显式算每一对 query-key 的相互作用，却还能得到一个足够接近 softmax attention 的结果？**

这就是 **attention linearization** 的出发点。

---

## 2. Performer 的改进



### 2.1 核心思想：用 feature map 近似 kernel

经典 attention 里最难的部分是这个 kernel：

$$
K(q,k)=\exp(q^\top k/\sqrt d).
$$

如果我们能找到一个映射（feature map）

$$
\phi:\mathbb R^d \to \mathbb R^m
$$

使得

$$
K(q,k)\approx \phi(q)^\top \phi(k),
$$

也就是把原来的指数核，近似成两个 $m$ 维向量的内积。那么 attention 就变成：

$$
\hat o_i=
\frac{\sum_{j\le i}\phi(q_i)^\top\phi(k_j)\,v_j}
{\sum_{j\le i}\phi(q_i)^\top\phi(k_j)}.
$$

由于 $\phi(q_i)$ 不依赖于求和指标 $j$，可以把它提出来：

$$
\hat o_i=
\frac{\phi(q_i)^\top\left(\sum_{j\le i}\phi(k_j)v_j^\top\right)}
{\phi(q_i)^\top\left(\sum_{j\le i}\phi(k_j)\right)}.
$$



### 2.2 为什么这个形式会把复杂度降成线性？

定义两个前缀和：

$$
S_i=\sum_{j\le i}\phi(k_j)v_j^\top \in \mathbb{R}^{m\times d_v},
$$

$$
z_i=\sum_{j\le i}\phi(k_j)\in\mathbb{R}^m.
$$

那么输出可以写成

$$
\hat o_i=\frac{\phi(q_i)^\top S_i}{\phi(q_i)^\top z_i}.
$$

现在看计算过程：

- 每来一个新的 token，只需要更新一次
  $$
  S_i=S_{i-1}+\phi(k_i)v_i^\top
  $$
- 同时更新
  $$
  z_i=z_{i-1}+\phi(k_i)
  $$

于是第 $i$ 步不需要再和所有历史 token 一个个重新算 kernel 了，只需要：

1. 算一次 $\phi(q_i)$；
2. 和累计好的 $S_i,z_i$ 做内积。

如果 $m$ 是固定的，那么每一步代价大约是 $O(m d_v)$ 或 $O(m)$ 量级，所以总复杂度变成对序列长度 $n$ 线性增长，也就是 $O(n)$。

- **时间复杂度约线性于 $n$**
- **内存复杂度约线性于 $n$**（甚至流式实现时更低）

这就是“linear attention”的来源。


### 2.3 Performer 做了什么？

Performer 是这个方向中的一个经典工作。它的主要贡献是：

> 为 softmax kernel 设计了一个适合随机特征近似（random features）的正值 feature map。

普通 random Fourier features 更适合平移不变 kernel（比如 RBF），但 softmax kernel

$$
\exp(q^\top k/\sqrt d)
$$

不是这种形式，所以不能直接套用最基础的方法。

Performer 提出的 FAVOR+ 方法，专门为 softmax attention 构造了一个 **正的随机特征映射**，使得：

1. 它能近似 softmax kernel；
2. 近似值保持非负，这对 attention 的数值稳定性和归一化很重要；
3. 可以直接用于线性化 attention 的计算。

你们这个项目里，**Performer 是 baseline**。意思是：

- 你们需要先实现或调用 Performer 的特征映射；
- 然后再拿它和至少一个其他方法比较；
- 最终目标不是重新发明完整模型，而是在相同评测指标下找到一个**比 Performer 更好的 kernel approximation 方法**。

---

## 3. 这次项目的核心工作


### 3.1 设计/选择一个 feature map 来近似 softmax kernel

整个项目围绕下面这个目标展开：

$$
K(q,k)=\exp(q^\top k/\sqrt d)
\quad\text{被}\quad
\hat K_\phi(q,k)=\phi(q)^\top\phi(k)
\quad\text{近似。}
$$

你们要做的是：

- 以 Performer 为 baseline；
- 再选一个你们自己的方法（自己设计，或来自文献）；
- 比较它们在两个 setting 下近似 softmax kernel 的能力。

这两个 setting 是：

1. **Gaussian toy setting**
2. **Frozen GPT-2 real-data setting**

重点不是训练一个新的 transformer，而是**只研究 kernel approximation 本身**。


### 3.2 在 Gaussian & GPT-2 两个 Setting 下评测

#### (1) Gaussian toy setting

这里假设：

$$
q,k \overset{i.i.d.}{\sim} \mathcal N(0,I_d).
$$

这个设置的优点是：

- 简单、可控
- 易于大量采样
- 能看出方法在“理想化分布”下的理论/数值表现

它像是一个“实验室环境”。

#### (2) GPT-2 real-data setting

这里你们不再自己假设 $q,k$ 的分布，而是：

- 用 Hugging Face 上预训练好的 **GPT-2 small**
- 输入真实文本
- 从若干层、若干头中提取真实的 query/key/value 向量

这个设置的意义在于：

- 检查方法是否在真实 transformer 激活上仍然有效
- 对比 toy setting 与真实 setting 的差异
- 理解为什么一些在 Gaussian 情况下有效的方法，在真实语言模型里可能失败


#### (3) 大致的评测流程


> *Step 1：准备测试数据*

**在 Gaussian setting 中准备数据**

直接按题目要求采样：

$$
q,k \sim \mathcal N(0,I_d).
$$

通常做法是：

- 固定维度 $d$
- 采样很多组 $(q^{(t)}, k^{(t)})$, $t=1,\dots,T$ 作为测试样本



**在 GPT-2 setting 中准备数据**

这里是从真实模型中抽取 activations。基本流程如下：

1. 选择一些真实文本数据  
   - WikiText
   - 新闻文本
   - 书籍片段

2. 把文本输入 frozen GPT-2 small  
   注意：**模型保持冻结**，不训练、不微调。

3. 从指定层和指定 head 中提取 query/key/value  
   例如：
   - 第 1 层和第 7 层
   - 每层选几个 attention heads

4. 形成大量 $(q,k)$ 样本对作为测试数据。


> *Step 2：应用你们的 kernel approximation 方法*

对每一种 feature map 方法（比如 Performer、你们提出的改进方法），以及对每个 feature dimension $m$，都要做下面这件事：

1. 对每个 query $q$，计算 $\phi(q)\in \mathbb R^m$
2. 对每个 key $k$，计算 $\phi(k)\in \mathbb R^m$
3. 用内积形成近似 kernel：

$$
\hat K_\phi(q,k)=\phi(q)^\top \phi(k).
$$

这里还要研究：当 feature dimension $m$ 增加时，这个近似会不会稳定变好？

所以你们需要对多个 $m$ 反复评测，比如：

- $m=16, 32, 64, 128, 256, \dots$


> *Step 3：与 ground truth 比较，计算评测指标*

项目要求的主指标是 **relative kernel approximation error**：

$$
\mathrm{RelErr}_{\mathrm{ker}}(\phi)
=\frac{
\mathbb E\left[(K(q,k)-\hat K_\phi(q,k))^2\right]
}{
\mathbb E\left[K(q,k)^2\right]
}.
$$

其中：

- ground truth kernel 是

$$
K(q,k)=\exp(q^\top k/\sqrt d)
$$

- 近似 kernel 是

$$
\hat K_\phi(q,k)=\phi(q)^\top\phi(k)
$$


**这个指标在 Gaussian 和 GPT-2 两个 setting 中分别怎么算？**

**Gaussian setting**
用 Monte Carlo 平均来估计：

$$
\mathrm{RelErr}_{\mathrm{ker}}(\phi)
\approx
\frac{
\frac1T\sum_{t=1}^T
\left(
K(q^{(t)},k^{(t)})-\hat K_\phi(q^{(t)},k^{(t)})
\right)^2
}{
\frac1T\sum_{t=1}^T
K(q^{(t)},k^{(t)})^2
}.
$$

**GPT-2 setting**
用提取到的真实 $(q,k)$ 样本做经验平均：

$$
\mathrm{RelErr}_{\mathrm{ker}}(\phi)
\approx
\frac{
\frac1N\sum_{s=1}^N
\left(
K(q^{(s)},k^{(s)})-\hat K_\phi(q^{(s)},k^{(s)})
\right)^2
}{
\frac1N\sum_{s=1}^N
K(q^{(s)},k^{(s)})^2
}.
$$



> *Step 4：可选地评测 attention output error*

项目还建议一个次要指标，更贴近真正 attention 输出的误差，例如：

$$
\mathrm{RelErr}_{\mathrm{out}}
=\frac{\|AV-\hat AV\|_F}{\|AV\|_F}.
$$

这里：

- $A$ 是 exact softmax attention matrix
- $\hat A$ 是近似得到的 attention matrix
- $V$ 是 value matrix

这个指标不是必须，但很有帮助，因为：

- kernel 近似好，不一定代表最终 attention 输出一定好；
- attention-output error 更接近模型真正使用 attention 的方式。

所以理想报告中最好同时给出：

1. **主指标**：kernel relative error
2. **次指标**：attention output error

---

## 5. 你们最终要回答什么问题？


### 5.1 Feature dimension $m$ 增大时，误差是否下降？

这是最基本的问题。  
如果一个 feature map 真的是合理的近似方法，那么随着 $m$ 增加，理论上应该越来越接近真实 kernel。

所以主图一般应该是：

- 横轴：feature dimension $m$
- 纵轴：$\mathrm{RelErr}_{\mathrm{ker}}$

观察：

- 是否单调下降？
- 是否下降得足够稳定？
- 是否有平台期？


### 5.2 找到一个比 Performer 更好的 "improved feature map

这是项目明确要求的：

> 把 Performer 当 baseline，找到至少一个在同一指标下表现更好的方法。

这里的“更好”可以体现在：

- 对同样的 $m$，误差更小
- 对不同层/头更稳定
- 在 Gaussian 和 GPT-2 两个 setting 下都更强
- 尤其在 GPT-2 setting 里更 robust


### 5.3 Gaussian setting 和 GPT-2 setting 的差别是什么？

这是项目非常强调的一点。

很多方法在 Gaussian setting 下会表现不错，因为：

- 数据分布简单
- 各维独立、均值为 0、方差一致
- 内积分布相对规整

但真实 GPT-2 的 query/key 可能：

- 不居中
- 各维方差不同
- 有明显相关性
- 不像高斯
- 某些 head 会产生非常尖锐（spiky）的 attention

这会让 softmax kernel 更难近似。

所以你们需要分析：

- 哪些层/头比较容易近似？
- 哪些层/头比较困难？
- 为什么 Gaussian 实验里的成功不能自动推广到真实模型？


### 5.4 GPT-2 Setting 下，feature map 应该如何改进？

在 GPT-2 Setting 下找到一个比 Performer更 好的 feature map。如果Gaussian Setting 下的 improved feature map在 GPT-2 Setting 下依然比Performer 更好，也可以沿用。

例如，你们可以从数据观察出：

- query/key 均值偏移明显  
  → 也许要先做 centering / normalization

- 某些方向特别重要  
  → 也许要考虑 data-dependent projection

- dot product 分布非常偏，attention 很尖  
  → 也许要让 feature map 更关注大内积区域的近似精度

总之，改进方法不一定要很复杂，但要有**清晰动机**：

> 观察数据分布 → 发现 Performer 的弱点 → 提出更合适的近似方式 → 用同样 metric 验证确实更好。

---

## TODO

1. 实现 Performer's feature map
2. 调研其他 attention linearization 的方法，实现一个 improved feature map
3. 搭建一个评测框架
   - 构造两份测试集（分别在 Gaussian Setting & GPT-2 Setting 下抽取样本）
   - 撰写测试脚本
      - 实现两个metrics: RelErr_ker & RelErr_out
      - 画 error-vs-feature-dimension 曲线
4. 解释性的实验
   - 做跨 setting、跨 layer、跨 head 的比较分析（作业要求中的R3）
   - 做 Q/K 分布诊断与解释性分析（作业要求中的R4）
5. （看实验结果，不一定要做）在GPT-2 Setting 下，改进 feature map
6. 写报告
7. pre