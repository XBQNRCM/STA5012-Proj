#!/usr/bin/env python3
"""绘制 RelErr（ker / out）聚合值 vs feature dimension m 曲线。

用法示例：
  # ker
  python plot_error_vs_dim.py outputs/gaussian_.../results.json
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


def filter_records(records: list[dict], layers: list[int] | None, heads: list[int] | None) -> list[dict]:
    has_layer = any(r.get("layer") is not None for r in records)
    has_head = any(r.get("head") is not None for r in records)

    if (layers or heads) and not (has_layer or has_head):
        print(
            "warning: records have no layer/head fields; --layer/--head ignored.",
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
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for fm, m_to_err in agg_errs.items():
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
    fig.savefig(save_path, dpi=150)
    print(f"plot saved -> {save_path}")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("results", type=Path)
    p.add_argument("--layer", type=int, nargs="+", default=None)
    p.add_argument("--head", type=int, nargs="+", default=None)
    p.add_argument("--agg", choices=("mean", "median", "geomean"), default="median",
                   help="跨 trials/(layer,head) 的聚合方式，默认 median（对重尾更稳）")
    p.add_argument("--metric", choices=("auto", "ker", "out"), default="auto",
                   help="强制选择绘图指标；默认自动从 results 里选")
    p.add_argument("--logy", action="store_true", help="纵轴取 log")
    p.add_argument("--save-path", type=Path, default=None)
    args = p.parse_args()

    results_path: Path = args.results.resolve()
    if not results_path.exists():
        sys.exit(f"error: {results_path} not found")

    records = load_results(results_path)
    records = filter_records(records, args.layer, args.head)
    if not records:
        sys.exit("error: no records after filtering; check --layer/--head.")

    if args.metric == "auto":
        metric_key = detect_metric_key(records)
    else:
        metric_key = "relerr_kernel" if args.metric == "ker" else "relerr_output"

    agg_errs = compute_agg_error(records, metric_key, args.agg)

    settings = {r["setting"] for r in records}
    label_core = METRIC_LABEL[metric_key]
    title = f"{label_core} vs $m$  [{', '.join(sorted(settings))}]  agg={args.agg}"
    ylabel = f"{label_core} ({args.agg} over trials)"

    save_path = args.save_path or (results_path.parent / f"{metric_key}_vs_dim_{args.agg}.png")
    plot(agg_errs, title, ylabel, save_path, logy=args.logy)


if __name__ == "__main__":
    main()
