# STA5012：Attention Linearization 实验

用 feature map $\phi:\mathbb{R}^d\to\mathbb{R}^m$ 近似 softmax kernel $K(q,k)=\exp(q^\top k/\sqrt{d})$，在 **Gaussian 合成数据**与**冻结 GPT-2 + WikiText** 两种设定下评测 kernel 相对误差。

---

## 目录结构

```
run_eval.py                     # 采样 + 评测 CLI，结果写入 outputs/
plot_error_vs_dim.py            # 绘制 RelErr vs m 曲线
download_gpt2.sh                # 下载 gpt2 模型权重
src/
  constants.py                  # D_HEAD = 64（与 GPT-2 small head 维一致）
  feature_maps/
    base.py                     # FeatureMap 抽象基类
    performer.py                # PerformerFeatureMap
    __init__.py                 # REGISTRY + build_feature_map
  sampling/
    gaussian.py                 # q, k ~ N(0, I_d)
    gpt2_wikitext.py            # WikiText-2 + frozen GPT-2 提取 q, k
  eval/
    metrics.py                  # relerr_kernel_pairs
    runner.py                   # run_gaussian / run_gpt2 / EvalResult
outputs/                        # 实验结果
```

---

## 快速上手

```bash
# 安装依赖
pip install -r requirements.txt

# 下载 gpt2
bash download_gpt2.sh

# Gaussian setting
python run_eval.py --mode gaussian --maps performer --m 32 64 128 256 --n-pairs 10000 --n-trials 5

# GPT-2 setting（首次运行需联网下载模型和数据）
python run_eval.py --mode gpt2 --model-path gpt2 --maps performer --m 64 128 --n-docs 64

# 绘图
python plot_error_vs_dim.py outputs/gaussian_performer_.../results.json
```

结果保存到 `outputs/<run_name>/results.json`，图片默认保存到同目录的 `error_vs_dim.png`。

---

## `run_eval.py` 参数

### 通用


| 参数             | 默认              | 说明                                   |
| -------------- | --------------- | ------------------------------------ |
| `--mode`       | —               | `gaussian` 或 `gpt2`（必填）              |
| `--maps`       | `performer`     | feature map 名，空格分隔，须在 `REGISTRY` 中注册 |
| `--m`          | `32 64 128 256` | 特征维度 m，可传多个                          |
| `--dim-d`      | `64`            | 输入维度 d（= GPT-2 head dim）             |
| `--n-trials`   | `1`             | 每个 (map) 下随机特征矩阵重采样次数；同一 trial 内各 m 共用嵌套 ω 前缀 |
| `--seed`       | `0`             | 随机种子                                 |
| `--device`     | `cpu`           | `cpu` 或 `cuda`                       |
| `--output-dir` | `outputs`       | 结果根目录                                |


### Gaussian 专用


| 参数          | 默认      | 说明            |
| ----------- | ------- | ------------- |
| `--n-pairs` | `10000` | 独立采样 (q,k) 对数 |


### GPT-2 专用


| 参数             | 默认     | 说明                                 |
| -------------- | ------ | ---------------------------------- |
| `--model-path` | `gpt2` | 本地权重目录或 HuggingFace model id       |
| `--n-docs`     | `128`  | WikiText 文档条数（每条跨层×头各取一对 q,k）      |
| `--max-length` | `512`  | tokenizer 截断长度                     |
| `--layers`     | 全部层    | 逗号分隔，如 `0,1,11`                    |
| `--heads`      | 全部头    | 逗号分隔，如 `0,3`                       |
| `--token-pos`  | `-2`   | 取 q,k 的 token 位置：`-2`=序列中点，`-1`=末尾 |


### 输出目录命名规则

```
# Gaussian
outputs/gaussian_<maps>_m-<m...>_<n-pairs>pairs_<n-trials>trials_seed<seed>/results.json

# GPT-2
outputs/gpt2_<maps>_m-<m...>_<n-docs>docs_<n-trials>trials_seed<seed>_layers-<...>_heads-<...>_pos<token-pos>/results.json
```

---

## `plot_error_vs_dim.py` 参数

```bash
python plot_error_vs_dim.py <results.json> [--layer N ...] [--head N ...] [--save-path path.png]
```


| 参数              | 默认                     | 说明                                 |
| --------------- | ---------------------- | ---------------------------------- |
| `results`（位置参数） | —                      | `results.json` 路径                  |
| `--layer`       | 全部                     | 只绘制指定层（results 含 `layer` 字段时生效）    |
| `--head`        | 全部                     | 只绘制指定 head（results 含 `head` 字段时生效） |
| `--save-path`   | 同目录 `error_vs_dim.png` | 图片保存路径                             |


横轴为 m，纵轴为各 trial 的 RelErr 均值，每种 feature map 一条线。

---

## 如何扩展 Feature Map

1. 在 `src/feature_maps/` 新建类，继承 `FeatureMap`，实现 `forward(self, x) -> Tensor`（`x: [..., d]` → `[..., m]`）。`__init__` 建议写 `(self, dim_d, dim_m, **kwargs)`，吞掉 runner 可能传入的 `omega` 等；`PerformerFeatureMap` 会读取 `omega`、`generator`。
2. 在 `src/feature_maps/__init__.py` 的 `REGISTRY` 中注册：

```python
REGISTRY: dict[str, type[FeatureMap]] = {
    "performer": PerformerFeatureMap,
    "my_map": MyFeatureMap,
}
```

3. 命令行直接使用：`python run_eval.py --mode gaussian --maps performer my_map --m 64 128`

---

## 备注

- **float64 评测**：`relerr_kernel_pairs` 内部将 q, k 与 phi 转为 float64 后计算，避免 `exp(q·k/√d)` 在 float32 下溢出。
- **嵌套 ω**：每个 `(map, trial)` 在 CPU 上采样一次 `omega_full[max(m), d]`，各 `m` 使用 `omega_full[:m]`，便于比较「增大 m」时的误差曲线；`phi` 再 `.to(device)` 与 q、k 对齐。

