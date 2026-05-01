#!/usr/bin/env python3
"""绘制 RelErr（ker / out）聚合值 vs feature dimension m 曲线。

用法示例：
  # 单个文件
  python plot_error_vs_dim.py outputs/gaussian_.../results.json
  # 多个文件合并（不同 maps 的同 setting 实验拼成一张图）
  python plot_error_vs_dim.py results_a.json results_b.json --save-path merged.png
  # 按 layer / head 过滤（GPT-2 out 结果带 layer/head 字段时生效）
  python plot_error_vs_dim.py outputs/gpt2_out_.../results.json --layer 10 --head 6
  # 指定聚合方式
  python plot_error_vs_dim.py outputs/.../results.json --agg median
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


METRIC_KEYS = ("relerr_kernel", "relerr_output")
METRIC_LABEL = {
    "relerr_kernel": r"$\mathrm{RelErr}_{\mathrm{ker}}$",
    "relerr_output": r"$\mathrm{RelErr}_{\mathrm{out}}$",
}


def load_results(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_many(paths: list[Path]) -> list[dict]:
    out: list[dict] = []
    for p in paths:
        out.extend(load_results(p))
    return out


def filter_records(
    records: list[dict],
    layers: list[int] | None,
    heads: list[int] | None,
    maps: list[str] | None = None,
) -> list[dict]:
    has_layer = any(r.get("layer") is not None for r in records)
    has_head = any(r.get("head") is not None for r in records)

    if (layers or heads) and not (has_layer or has_head):
        print(
            "warning: records have no layer/head fields; --layer/--head ignored.",
            file=sys.stderr,
        )
        layers = heads = None

    out = []
    for r in records:
        if layers is not None and has_layer and r.get("layer") not in layers:
            continue
        if heads is not None and has_head and r.get("head") not in heads:
            continue
        if maps is not None and r.get("feature_map") not in maps:
            continue
        out.append(r)
    return out


def _aggregate(values: list[float], agg: str) -> float:
    arr = np.asarray(values, dtype=float)
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


def detect_metric_key(records: list[dict]) -> str:
    for key in METRIC_KEYS:
        if any(key in r for r in records):
            return key
    raise ValueError(f"records have no known metric field ({METRIC_KEYS}).")


def compute_agg_error(
    records: list[dict], metric_key: str, agg: str
) -> dict[str, dict[int, float]]:
    """返回 {feature_map: {dim_m: agg_val}}。"""
    bucket: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if metric_key not in r:
            continue
        bucket[r["feature_map"]][r["dim_m"]].append(float(r[metric_key]))
    return {
        fm: {m: _aggregate(errs, agg) for m, errs in sorted(m_dict.items())}
        for fm, m_dict in bucket.items()
    }


def plot(
    agg_errs: dict[str, dict[int, float]],
    title: str,
    ylabel: str,
    save_path: Path,
    logy: bool = False,
    map_order: list[str] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    if map_order:
        ordered = [fm for fm in map_order if fm in agg_errs]
        ordered += [fm for fm in agg_errs if fm not in map_order]
    else:
        ordered = list(agg_errs.keys())
    for fm in ordered:
        m_to_err = agg_errs[fm]
        ms = list(m_to_err.keys())
        errs = list(m_to_err.values())
        ax.plot(ms, errs, marker="o", label=fm)
    ax.set_xlabel("Feature dimension $m$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"plot saved -> {save_path}")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("results", type=Path, nargs="+",
                   help="一个或多个 results.json，会被合并")
    p.add_argument("--layer", type=int, nargs="+", default=None)
    p.add_argument("--head", type=int, nargs="+", default=None)
    p.add_argument("--maps", nargs="+", default=None,
                   help="只画指定的 feature_map（也决定绘图顺序）")
    p.add_argument("--agg", choices=("mean", "median", "geomean"), default="median",
                   help="跨 trials/(layer,head) 的聚合方式，默认 median（对重尾更稳）")
    p.add_argument("--metric", choices=("auto", "ker", "out"), default="auto",
                   help="强制选择绘图指标；默认自动从 results 里选")
    p.add_argument("--logy", action="store_true", help="纵轴取 log")
    p.add_argument("--save-path", type=Path, default=None)
    args = p.parse_args()

    results_paths = [r.resolve() for r in args.results]
    for rp in results_paths:
        if not rp.exists():
            sys.exit(f"error: {rp} not found")

    records = load_many(results_paths)
    records = filter_records(records, args.layer, args.head, args.maps)
    if not records:
        sys.exit("error: no records after filtering; check --layer/--head/--maps.")

    if args.metric == "auto":
        metric_key = detect_metric_key(records)
    else:
        metric_key = "relerr_kernel" if args.metric == "ker" else "relerr_output"

    agg_errs = compute_agg_error(records, metric_key, args.agg)

    settings = {r["setting"] for r in records}
    label_core = METRIC_LABEL[metric_key]
    title = f"{label_core} vs $m$  [{', '.join(sorted(settings))}]  agg={args.agg}"
    ylabel = f"{label_core} ({args.agg} over trials)"

    if args.save_path is not None:
        save_path = args.save_path
    elif len(results_paths) == 1:
        save_path = results_paths[0].parent / f"{metric_key}_vs_dim_{args.agg}.png"
    else:
        save_path = results_paths[0].parent / f"merged_{metric_key}_vs_dim_{args.agg}.png"
    plot(agg_errs, title, ylabel, save_path, logy=args.logy, map_order=args.maps)


if __name__ == "__main__":
    main()
