#!/usr/bin/env python3
"""绘制 RelErr vs feature dim：横轴按 2^n 等间距展示（适合 m 为 2 的幂）。

与 ``plot_error_vs_dim.py`` 的数据格式、过滤与聚合一致；差别是横轴用
``n = log2(m)`` 取点，刻度标签为 ``2^n``（或近似），相邻幂次在图上等距。

用法示例：
  python plot_error_vs_dim_pow2.py \\
    outputs/gaussian_.../results.json
  python plot_error_vs_dim_pow2.py results_a.json results_b.json --save-path merged_pow2.png
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from plot_error_vs_dim import (
    METRIC_LABEL,
    compute_agg_error,
    detect_metric_key,
    filter_records,
    load_many,
)


def _format_xtick(n: float) -> str:
    """刻度位置为 n=log2(m)；整数 n 时标成 2^n，否则标近似 m。"""
    k = round(n)
    if abs(n - k) < 1e-6:
        return rf"$2^{{{k}}}$"
    m = 2.0**n
    return rf"${m:g}$"


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

    all_ns: set[float] = set()
    for fm in ordered:
        m_to_err = agg_errs[fm]
        ms = sorted(m_to_err.keys())
        ns = [math.log2(m) for m in ms]
        errs = [m_to_err[m] for m in ms]
        all_ns.update(ns)
        ax.plot(ns, errs, marker="o", label=fm)

    tick_ns = sorted(all_ns)
    ax.set_xticks(tick_ns)
    ax.set_xticklabels([_format_xtick(n) for n in tick_ns])
    ax.set_xlabel(r"Feature dimension $m=2^n$ (equal spacing in $n$)")
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
    p = argparse.ArgumentParser(
        description="RelErr vs m，横轴为 n=log2(m)，刻度显示为 2^n（幂次等间距）。"
    )
    p.add_argument("results", type=Path, nargs="+",
                   help="一个或多个 results.json，会被合并")
    p.add_argument("--layer", type=int, nargs="+", default=None)
    p.add_argument("--head", type=int, nargs="+", default=None)
    p.add_argument("--maps", nargs="+", default=None,
                   help="只画指定的 feature_map（也决定绘图顺序）")
    p.add_argument("--agg", choices=("mean", "median", "geomean"), default="median")
    p.add_argument("--metric", choices=("auto", "ker", "out"), default="auto")
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
    title = (
        f"{label_core} vs $m=2^n$  [{', '.join(sorted(settings))}]  agg={args.agg}"
    )
    ylabel = f"{label_core} ({args.agg} over trials)"

    if args.save_path is not None:
        save_path = args.save_path
    elif len(results_paths) == 1:
        save_path = (
            results_paths[0].parent / f"{metric_key}_vs_dim_pow2_{args.agg}.png"
        )
    else:
        save_path = (
            results_paths[0].parent
            / f"merged_{metric_key}_vs_dim_pow2_{args.agg}.png"
        )
    plot(agg_errs, title, ylabel, save_path, logy=args.logy, map_order=args.maps)


if __name__ == "__main__":
    main()
