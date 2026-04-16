#!/usr/bin/env python3
"""绘制 RelErr_ker 均值 vs feature dimension m 曲线。

用法示例：
  # 指定 results
  python plot_error_vs_dim.py outputs/gaussian_performer_.../results.json
  # 指定 layer 和 head
  python plot_error_vs_dim.py outputs/gpt2_.../results.json --layer 0 1 --head 0
  # 指定保存路径
  python plot_error_vs_dim.py outputs/gaussian_performer_.../results.json --save-path outputs/gaussian_performer_.../plot.png
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def filter_records(records: list[dict], layers: list[int] | None, heads: list[int] | None) -> list[dict]:
    has_layer = any("layer" in r for r in records)
    has_head = any("head" in r for r in records)

    if (layers or heads) and not (has_layer or has_head):
        print(
            "warning: results 中无 layer/head 字段，--layer/--head 过滤被忽略。",
            file=sys.stderr,
        )
        return records

    out = []
    for r in records:
        if layers is not None and has_layer and r.get("layer") not in layers:
            continue
        if heads is not None and has_head and r.get("head") not in heads:
            continue
        out.append(r)
    return out


def compute_mean_error(records: list[dict]) -> dict[str, dict[int, float]]:
    """返回 {feature_map: {dim_m: mean_relerr}}。"""
    # {feature_map: {dim_m: [relerr, ...]}}
    bucket: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        bucket[r["feature_map"]][r["dim_m"]].append(r["relerr_kernel"])

    return {
        fm: {m: float(np.mean(errs)) for m, errs in sorted(m_dict.items())}
        for fm, m_dict in bucket.items()
    }


def plot(mean_errs: dict[str, dict[int, float]], title: str, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))

    for fm, m_to_err in mean_errs.items():
        ms = list(m_to_err.keys())
        errs = list(m_to_err.values())
        ax.plot(ms, errs, marker="o", label=fm)

    ax.set_xlabel("Feature dimension $m$")
    ax.set_ylabel(r"$\mathrm{RelErr}_{\mathrm{ker}}$ (mean over trials)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"plot saved → {save_path}")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("results", type=Path, help="results.json 路径")
    p.add_argument("--layer", type=int, nargs="+", default=None, help="只绘制指定层（GPT-2 结果含 layer 字段时生效）")
    p.add_argument("--head", type=int, nargs="+", default=None, help="只绘制指定 head（GPT-2 结果含 head 字段时生效）")
    p.add_argument("--save-path", type=Path, default=None, help="图片保存路径，默认与 results.json 同目录下的 error_vs_dim.png")
    args = p.parse_args()

    results_path: Path = args.results.resolve()
    if not results_path.exists():
        sys.exit(f"error: {results_path} not found")

    records = load_results(results_path)
    records = filter_records(records, args.layer, args.head)
    if not records:
        sys.exit("error: 过滤后无记录，请检查 --layer/--head 参数。")

    mean_errs = compute_mean_error(records)

    # 标题：从 results 中读取 setting
    settings = {r["setting"] for r in records}
    title = "RelErr$_{ker}$ vs $m$  [" + ", ".join(sorted(settings)) + "]"

    save_path = args.save_path or (results_path.parent / "error_vs_dim.png")
    plot(mean_errs, title, save_path)


if __name__ == "__main__":
    main()
