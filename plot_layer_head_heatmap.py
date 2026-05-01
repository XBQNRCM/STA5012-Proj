#!/usr/bin/env python3
"""绘制 GPT-2 setting 下每个 feature map 的 layer × head heatmap（R3）。

输入一个或多个 results.json（带 ``layer`` / ``head`` 字段、含 ``relerr_output`` 或
``relerr_kernel``）。脚本会：

1. 把多份 json 直接合并；
2. 选取指定 m（默认取最大 m，即近似最稳的那一档）；
3. 对每个 feature_map，把 (layer, head) 上跨 trials 聚合（默认 median），画一张 5×5 heatmap；
4. 多个 feature_map 拼成一行/多行 panels。

用法示例：
  python plot_layer_head_heatmap.py \
      outputs/gpt2_..._metric-out/results.json \
      outputs/gpt2_cosine-bias_..._metric-out/results.json \
      --m 80 --agg median --metric out --save-path outputs/heatmap_out_m80.png
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRIC_LABEL = {
    "relerr_kernel": r"$\mathrm{RelErr}_{\mathrm{ker}}$",
    "relerr_output": r"$\mathrm{RelErr}_{\mathrm{out}}$",
}


def _load(paths: list[Path]) -> list[dict]:
    out: list[dict] = []
    for p in paths:
        out.extend(json.loads(p.read_text(encoding="utf-8")))
    return out


def _detect_metric_key(records: list[dict], requested: str) -> str:
    if requested == "ker":
        return "relerr_kernel"
    if requested == "out":
        return "relerr_output"
    for k in ("relerr_output", "relerr_kernel"):
        if any(k in r for r in records):
            return k
    raise ValueError("no known metric field in records")


def _aggregate(values: list[float], agg: str) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    if agg == "mean":
        return float(arr.mean())
    if agg == "median":
        return float(np.median(arr))
    if agg == "geomean":
        pos = arr[arr > 0]
        if pos.size == 0:
            return float("nan")
        return float(math.exp(float(np.log(pos).mean())))
    raise ValueError(f"unknown agg: {agg}")


def build_heatmaps(
    records: list[dict],
    metric_key: str,
    target_m: int,
    agg: str,
) -> tuple[dict[str, np.ndarray], list[int], list[int]]:
    """返回 {feature_map: (n_layers, n_heads)} ndarray，以及排序后的 layer / head 轴。"""
    layers = sorted({r["layer"] for r in records if r.get("layer") is not None})
    heads = sorted({r["head"] for r in records if r.get("head") is not None})
    if not layers or not heads:
        sys.exit("error: records missing layer/head fields; can't draw heatmap.")
    layer_idx = {l: i for i, l in enumerate(layers)}
    head_idx = {h: i for i, h in enumerate(heads)}

    bucket: dict[str, dict[tuple[int, int], list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in records:
        if r.get("dim_m") != target_m:
            continue
        if metric_key not in r:
            continue
        if r.get("layer") is None or r.get("head") is None:
            continue
        bucket[r["feature_map"]][(r["layer"], r["head"])].append(float(r[metric_key]))

    grids: dict[str, np.ndarray] = {}
    for fm, cell_map in bucket.items():
        g = np.full((len(layers), len(heads)), np.nan, dtype=float)
        for (l, h), vals in cell_map.items():
            g[layer_idx[l], head_idx[h]] = _aggregate(vals, agg)
        grids[fm] = g
    return grids, layers, heads


def _grid_layout(n: int) -> tuple[int, int]:
    if n <= 3:
        return 1, n
    if n <= 6:
        return 2, math.ceil(n / 2)
    return 3, math.ceil(n / 3)


def plot(
    grids: dict[str, np.ndarray],
    layers: list[int],
    heads: list[int],
    metric_key: str,
    target_m: int,
    agg: str,
    save_path: Path,
    map_order: list[str] | None = None,
) -> None:
    if map_order is None:
        map_order = list(grids.keys())
    map_order = [fm for fm in map_order if fm in grids]
    if not map_order:
        sys.exit("error: no maps to plot")

    nrows, ncols = _grid_layout(len(map_order))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.4 * ncols + 1.0, 2.8 * nrows + 0.6), squeeze=False
    )

    # 用全局共享色阶，便于跨 panel 比较
    all_vals = np.concatenate([g[~np.isnan(g)].ravel() for g in grids.values()])
    vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))

    last_im = None
    for k, fm in enumerate(map_order):
        ax = axes[k // ncols][k % ncols]
        g = grids[fm]
        im = ax.imshow(g, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(heads)))
        ax.set_xticklabels(heads)
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layers)
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
        ax.set_title(fm)
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                v = g[i, j]
                if not math.isnan(v):
                    ax.text(
                        j, i, f"{v:.2f}",
                        ha="center", va="center", fontsize=7,
                        color="white" if v < (vmin + vmax) / 2 else "black",
                    )
        last_im = im

    for k in range(len(map_order), nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")

    label = METRIC_LABEL[metric_key]
    fig.suptitle(
        f"{label} by (layer, head) at $m={target_m}$  [agg={agg}]",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 0.93, 0.96))
    if last_im is not None:
        cax = fig.add_axes([0.945, 0.12, 0.015, 0.76])
        fig.colorbar(last_im, cax=cax, label=label)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"saved -> {save_path}")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("results", type=Path, nargs="+",
                   help="一个或多个 results.json，会被合并")
    p.add_argument("--metric", choices=("auto", "ker", "out"), default="out")
    p.add_argument("--m", type=int, default=None,
                   help="选取的 dim_m，默认取数据中最大的 m")
    p.add_argument("--agg", choices=("mean", "median", "geomean"), default="median")
    p.add_argument("--maps", nargs="+", default=None,
                   help="若指定，只画这些 feature map（也决定顺序）")
    p.add_argument("--save-path", type=Path, default=None)
    args = p.parse_args()

    records = _load(args.results)
    if not records:
        sys.exit("error: no records loaded")

    metric_key = _detect_metric_key(records, args.metric)
    available_m = sorted({r["dim_m"] for r in records})
    target_m = args.m if args.m is not None else available_m[-1]
    if target_m not in available_m:
        sys.exit(f"error: m={target_m} not in data; available={available_m}")

    grids, layers, heads = build_heatmaps(records, metric_key, target_m, args.agg)
    if not grids:
        sys.exit("error: no per-(layer,head) data for chosen m")

    save_path = args.save_path or (
        args.results[0].parent / f"heatmap_{metric_key}_m{target_m}_{args.agg}.png"
    )
    plot(grids, layers, heads, metric_key, target_m, args.agg, save_path, args.maps)


if __name__ == "__main__":
    main()
