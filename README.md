# STA5012：Attention Linearization 实验

用 feature map $\phi:\mathbb{R}^d\to\mathbb{R}^m$ 近似 softmax kernel $K(q,k)=\exp(q^\top k/\sqrt{d})$，在 **Gaussian 合成数据**与**冻结 GPT-2 + WikiText** 两种设定下评测两类相对误差：

- **主指标 `RelErr_ker`**：$\mathbb E[(K-\hat K)^2]/\mathbb E[K^2]$，kernel pair-level 误差
- **次指标 `RelErr_out`**：$\|AV-\hat AV\|_F/\|AV\|_F$，attention 输出 Frobenius 相对误差

---

## 目录结构

```
run_eval.py                     # 采样 + 评测 CLI，结果写入 outputs/
plot_error_vs_dim.py            # 绘制 RelErr vs m 曲线（支持 median/mean/geomean）
download_gpt2.sh                # 下载 gpt2 模型权重（modelscope）
src/
  constants.py                  # D_HEAD = 64（与 GPT-2 small head 维一致）
  feature_maps/
    base.py                     # FeatureMap 抽象基类
    performer.py                # PerformerFeatureMap / ScaledPerformerFeatureMap
    __init__.py                 # REGISTRY + build_feature_map
  sampling/
    gaussian.py                 # sample_qk_gaussian / sample_qkv_gaussian
    gpt2_wikitext.py            # collect_qk_gpt2_wikitext / iter_qkv_gpt2_wikitext
  eval/
    metrics.py                  # relerr_kernel_pairs / relerr_output / output_numer_denom
    runner.py                   # run_gaussian(_output) / run_gpt2(_output) / EvalResult(Out)
outputs/                        # 实验结果
```

---

## 快速上手

```bash
# 安装依赖
pip install -r requirements.txt

# 下载 gpt2（任选其一）
bash download_gpt2.sh                                                    # modelscope
python -m modelscope.cli.cli download --model openai-community/gpt2 \    # 或精选
  --include "pytorch_model.bin" "config.json" "tokenizer*" "vocab.json" "merges.txt" \
  --local_dir gpt2

# Gaussian setting：同时跑 ker+out
python run_eval.py --mode gaussian --metric both \
  --maps performer scaled_performer power_performer rala_performer \
  --m 32 64 128 256 --n-pairs 10000 --n-trials 5 \
  --n-seq 32 --seq-len 128

# GPT-2 setting（首次运行需联网下载 WikiText-2 parquet 到 ./wikitext-2-raw-v1/）
python run_eval.py --mode gpt2 --metric both --model-path ./gpt2 \
  --m 32 64 128 256 --n-docs 128 --max-length 512 \
  --layers 10 --heads 6 --n-trials 3

# 绘图（支持 median / geomean / mean 聚合）
python plot_error_vs_dim.py outputs/gaussian_.../results.json --agg median
python plot_error_vs_dim.py outputs/gpt2_..._metric-out/results.json --agg median --layer 10 --head 6
```

结果写入 `outputs/<run_name>/results.json`，图片默认保存到同目录。`--metric both` 会分别写两个子目录（一个 `metric-ker`，一个 `metric-out`）。

当前已注册的 feature map：

- `performer`：标准 Performer / FAVOR+ 正随机特征。
- `scaled_performer`：先对输入 batch 做逐维经验标准化，再套 Performer。
- `power_performer`：先做逐元素幂压缩 `sign(x) * |x|^alpha`，再套 Performer；`alpha` 由 `--power-alpha` 控制。
- `rala_performer`：RALA-inspired 版本，`phi_performer(x)` 乘以上下文权重 `gamma(x)`，再做轻量通道混合；`--rala-mix` 控制混合强度。

---

## `run_eval.py` 参数

### 通用

| 参数             | 默认              | 说明                                                 |
| -------------- | --------------- | -------------------------------------------------- |
| `--mode`       | —               | `gaussian` 或 `gpt2`（必填）                            |
| `--metric`     | `ker`           | `ker` / `out` / `both`，选择要算哪个指标                    |
| `--maps`       | `performer`     | feature map 名，空格分隔，须在 `REGISTRY` 中注册                 |
| `--m`          | `32 64 128 256` | 特征维度 m，可传多个                                          |
| `--dim-d`      | `64`            | 输入维度 d（= GPT-2 head dim）                              |
| `--n-trials`   | `1`             | 每个 (map) 下随机特征矩阵重采样次数；同一 trial 内各 m 共用嵌套 ω 前缀          |
| `--seed`       | `0`             | 随机种子                                                 |
| `--device`     | `cpu`           | `cpu` 或 `cuda`                                       |
| `--output-dir` | `outputs`       | 结果根目录                                                |
| `--power-alpha`| `0.5`           | `power_performer` 的幂指数，需在 `(0,1)`                         |
| `--rala-mix`   | `0.5`           | `rala_performer` 的通道混合强度                                  |

### Gaussian 专用

| 参数          | 默认     | 说明                                                  | 生效指标 |
| ----------- | ------ | --------------------------------------------------- | ---- |
| `--n-pairs` | `10000` | 独立采样 (q,k) 对数                                         | ker  |
| `--n-seq`   | `32`   | 序列条数                                                   | out  |
| `--seq-len` | `256`  | 每条序列长度                                                 | out  |
| `--dim-v`   | `dim-d`| V 维度                                                  | out  |

### GPT-2 专用

| 参数             | 默认     | 说明                                           | 生效指标 |
| -------------- | ------ | -------------------------------------------- | ---- |
| `--model-path` | `gpt2` | 本地权重目录或 HuggingFace model id                   | 二者   |
| `--n-docs`     | `128`  | WikiText 文档条数                                  | 二者   |
| `--max-length` | `512`  | tokenizer 截断长度                                 | 二者   |
| `--layers`     | 全部层    | 逗号分隔，如 `0,1,11`                                 | 二者   |
| `--heads`      | 全部头    | 逗号分隔，如 `0,3`                                    | 二者   |
| `--token-pos`  | `-2`   | 取 q,k 的 token 位置：`-2`=序列中点，`-1`=末尾                | ker  |

GPT-2 + `out` 会对所选 `(layer, head)` 每个组合独立累加 $\|\cdot\|_F^2$ 后聚合为单一 RelErr；同一条序列的所有 token 同时参与 exact softmax 与 linearized 两份 attention 计算。

### 输出目录命名规则

```
# Gaussian
outputs/gaussian_<maps>_m-<m...>_<size>_<n-trials>trials_seed<seed>_metric-<ker|out>/results.json
#   ker: <size>  = <n-pairs>pairs
#   out: <size>  = <n-seq>seq-<seq-len>len

# GPT-2
outputs/gpt2_<maps>_m-<m...>_<n-docs>docs_<n-trials>trials_seed<seed>_layers-<...>_heads-<...>_metric-<ker|out>[_pos<token-pos>]/results.json
```

> 老结果目录（不带 `metric-*` 段）仍可被 `plot_error_vs_dim.py` 自动识别，向前兼容。

---

## `plot_error_vs_dim.py` 参数

```bash
python plot_error_vs_dim.py <results.json> \
  [--agg {median,mean,geomean}] [--metric {auto,ker,out}] \
  [--layer N ...] [--head N ...] [--logy] [--save-path path.png]
```

| 参数              | 默认                                        | 说明                                          |
| --------------- | ----------------------------------------- | ------------------------------------------- |
| `results`（位置参数） | —                                         | `results.json` 路径                           |
| `--agg`         | `median`                                  | 聚合 trials 的方式；重尾场景下推荐 `median` / `geomean`     |
| `--metric`      | `auto`                                    | 从 records 自动识别 `relerr_kernel` / `relerr_output` |
| `--layer`       | 全部                                        | 只绘制指定 layer（GPT-2 结果含 `layer` 字段时生效）           |
| `--head`        | 全部                                        | 只绘制指定 head                                    |
| `--logy`        | off                                       | y 轴取对数                                         |
| `--save-path`   | 同目录 `relerr_<ker\|output>_vs_dim_<agg>.png` | 图片保存路径                                         |

横轴 m，纵轴按 `--agg` 聚合后的 RelErr，每种 feature map 一条线。

---

## 如何扩展 Feature Map

1. 在 `src/feature_maps/` 新建类，继承 `FeatureMap`，实现 `forward(self, x) -> Tensor`（`x: [..., d]` → `[..., m]`）。`__init__` 建议写 `(self, dim_d, dim_m, **kwargs)`，吞掉 runner 可能传入的 `omega` 等；`PerformerFeatureMap` 会读取 `omega`、`generator`。
2. 在 `src/feature_maps/__init__.py` 的 `REGISTRY` 中注册：

```python
REGISTRY: dict[str, type[FeatureMap]] = {
    "performer": PerformerFeatureMap,
    "power_performer": PowerPerformerFeatureMap,
    "rala_performer": RALAPerformerFeatureMap,
    "my_map": MyFeatureMap,
}
```

3. 命令行直接使用：`python run_eval.py --mode gaussian --maps performer my_map --m 64 128`

---

## 数据集

### WikiText-2-raw-v1

`src/sampling/gpt2_wikitext.py` 内 `_load_wikitext_rows` 有三级 fallback：

1. 本地 `./wikitext-2-raw-v1/train.parquet`（推荐离线常备）
2. `https://hf-mirror.com/.../train-00000-of-00001.parquet`
3. `datasets.load_dataset("wikitext", "wikitext-2-raw-v1")`（新版 `datasets>=4.0` 可能不支持 script 格式）

离线一次性准备：

```bash
mkdir -p wikitext-2-raw-v1
curl -L -o wikitext-2-raw-v1/train.parquet \
  https://hf-mirror.com/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/train-00000-of-00001.parquet
```

---

## 备注

- **float64 评测**：`relerr_kernel_pairs` / `relerr_output` 内部将 q、k、V、phi 一并转 float64，规避 `exp(q·k/√d)` 在 float32 下的溢出。
- **嵌套 ω**：每个 `(map, trial)` 在 CPU 上采样一次 `omega_full[max(m), d]`，各 `m` 使用 `omega_full[:m]`，保证同一 trial 内比较 m 效应时不掺入 ω 采样噪声；`phi` 再 `.to(device)` 与 q、k 对齐。
- **聚合约定**：`RelErr_out` 跨序列时先对分子/分母 $\|\cdot\|_F^2$ 求和，再 `sqrt` 相除，等价于把所有 (seq, i, j) 元素展平作单一 Frobenius 比值，避免 per-seq ratio 平均的重尾偏差。
- **重尾与 trials**：`RelErr_ker` 的分子 $(K-\hat K)^2$ 在 Gaussian 下极重尾，单 trial 波动可达 3 个数量级；经验上需要 `n_trials≥500` 才能用 mean 得到稳定曲线，或退而用 `--agg median`。`RelErr_out` 被 softmax 与 linear 组合双重平滑，通常 3–10 trials 就稳定单调。

---

## 实验摘要

详见 [`REPORT.md`](REPORT.md)。关键两张图：

- `outputs/gaussian_performer_m-16..80_10000pairs_1000trials_seed42/error_vs_dim.png`  
  Gaussian, RelErr_ker（1000 trials mean），m: 16 → 80，~1.5×10⁶ → 6×10⁴，严格单调下降。
- `outputs/gpt2_performer_m-16..80_1000docs_5000trials_seed42_layers-2-4-6-8-10_heads-2-4-6-8-10_pos-2/error_vs_dim.png`  
  GPT-2, RelErr_ker（5 layers × 5 heads, 5000 trials），1.027 → 1.002，分子恰好被分母压住。
