#!/usr/bin/env python3
"""GPT-2 + WikiText Q/K 分布诊断（R4 主战场）。

对若干 (layer, head)：
  1. 收集所有 token 的 q, k，得到 [N, d_head]；
  2. 画 dim-wise mean / std（与 N(0, I_d) 对比）；
  3. 画 ||q||, ||k|| 直方图，叠加 chi_d 理论密度；
  4. 画 q·k / sqrt(d) score 直方图，叠加 Gaussian baseline；
  5. 画每条序列 softmax(score) 的 normalized entropy 分布，反映 attention spikiness；
  6. 把数值摘要写到 qk_stats.json，供报告引用。

用法：
  python analyze_qk_dist.py \
      --model-path ./gpt2 --n-docs 200 --max-length 512 \
      --pairs 2,2 4,4 6,6 8,8 10,10 \
      --output-dir outputs/qk_diagnostic \
      --device cpu
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats as scistats

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constants import D_HEAD
from src.sampling.gpt2_wikitext import iter_qkv_gpt2_wikitext


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


def collect(
    model_path: str,
    n_docs: int,
    max_length: int,
    pairs: list[tuple[int, int]],
    device: str,
) -> dict[tuple[int, int], dict]:
    """逐 (layer, head) 累加所有 token 的 q, k 与每条序列的 normalized entropy。"""
    layer_set = sorted({l for l, _ in pairs})
    head_set = sorted({h for _, h in pairs})
    pair_set = set(pairs)

    out: dict[tuple[int, int], dict] = {
        (l, h): {"q": [], "k": [], "entropy": [], "seq_len": []}
        for (l, h) in pairs
    }

    for doc_idx, li, hi, q_seq, k_seq, _v in iter_qkv_gpt2_wikitext(
        model_path=model_path,
        n_docs=n_docs,
        max_length=max_length,
        layers=layer_set,
        heads=head_set,
        device=device,
    ):
        if (li, hi) not in pair_set:
            continue
        # q_seq, k_seq: [1, T, d_head]
        q_flat = q_seq[0].detach().to(torch.float64).cpu()
        k_flat = k_seq[0].detach().to(torch.float64).cpu()
        T, d = q_flat.shape

        out[(li, hi)]["q"].append(q_flat)
        out[(li, hi)]["k"].append(k_flat)

        S = (q_flat @ k_flat.T) / math.sqrt(d)
        A = torch.softmax(S, dim=-1)
        H = -(A * (A.clamp_min(1e-12).log())).sum(dim=-1)
        H_norm = (H / math.log(T)).numpy() if T > 1 else np.array([0.0])
        out[(li, hi)]["entropy"].append(H_norm)
        out[(li, hi)]["seq_len"].append(T)

    for (l, h), d in out.items():
        if not d["q"]:
            continue
        d["q"] = torch.cat(d["q"], dim=0).numpy()
        d["k"] = torch.cat(d["k"], dim=0).numpy()
        d["entropy"] = np.concatenate(d["entropy"])
        d["seq_len"] = np.array(d["seq_len"])
    return out


def gaussian_reference(n: int, d: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((n, d))
    k = rng.standard_normal((n, d))
    score = (q * k).sum(axis=-1) / math.sqrt(d)
    return {"q": q, "k": k, "score": score}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_dim_wise(
    data: dict[tuple[int, int], dict],
    pairs: list[tuple[int, int]],
    save_path: Path,
) -> None:
    n = len(pairs)
    fig, axes = plt.subplots(n, 2, figsize=(12, 2.6 * n), squeeze=False)
    for r, (l, h) in enumerate(pairs):
        d = data[(l, h)]
        q, k = d["q"], d["k"]
        d_dim = q.shape[1]
        xs = np.arange(d_dim)
        for col, (X, name) in enumerate([(q, "q"), (k, "k")]):
            ax = axes[r][col]
            mean = X.mean(axis=0)
            std = X.std(axis=0)
            ax.bar(xs, mean, color="tab:blue", alpha=0.55, label=f"per-dim mean of {name}")
            ax.plot(xs, std, color="tab:red", lw=1.2, label=f"per-dim std of {name}")
            ax.axhline(0.0, color="black", lw=0.5, ls="--", alpha=0.6)
            ax.axhline(1.0, color="tab:red", lw=0.5, ls="--", alpha=0.4,
                       label=r"$\mathcal{N}(0,I)$ std=1")
            ax.set_title(f"layer {l}, head {h}  -  {name}  (N={X.shape[0]})", fontsize=9)
            ax.set_xlabel("dim index")
            ax.set_ylabel("value")
            if r == 0 and col == 1:
                ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("Dimension-wise mean (bar) and std (line) of q, k", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(save_path, dpi=150)
    print(f"saved -> {save_path}")
    plt.close(fig)


def plot_norms(
    data: dict[tuple[int, int], dict],
    pairs: list[tuple[int, int]],
    save_path: Path,
) -> None:
    n = len(pairs)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.0), squeeze=False)
    for c, (l, h) in enumerate(pairs):
        d = data[(l, h)]
        nq = np.linalg.norm(d["q"], axis=-1)
        nk = np.linalg.norm(d["k"], axis=-1)
        ax = axes[0][c]
        ax.hist(nq, bins=60, alpha=0.55, color="tab:blue", density=True, label="‖q‖")
        ax.hist(nk, bins=60, alpha=0.55, color="tab:orange", density=True, label="‖k‖")

        d_dim = d["q"].shape[1]
        xs = np.linspace(0, max(nq.max(), nk.max()) * 1.05, 400)
        chi_pdf = scistats.chi.pdf(xs, df=d_dim)
        ax.plot(xs, chi_pdf, "k--", lw=1, label=fr"$\chi_{{{d_dim}}}$")

        ax.set_title(f"L{l} H{h}", fontsize=9)
        ax.set_xlabel("norm")
        if c == 0:
            ax.set_ylabel("density")
            ax.legend(fontsize=7)
    fig.suptitle(r"Histograms of $\|q\|$, $\|k\|$  (vs $\chi_{d}$ baseline)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(save_path, dpi=150)
    print(f"saved -> {save_path}")
    plt.close(fig)


def plot_scores(
    data: dict[tuple[int, int], dict],
    pairs: list[tuple[int, int]],
    gauss_score: np.ndarray,
    save_path: Path,
    n_pairs_per_lh: int = 50_000,
) -> None:
    rng = np.random.default_rng(0)
    n = len(pairs)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.0), squeeze=False)
    for c, (l, h) in enumerate(pairs):
        d = data[(l, h)]
        q, k = d["q"], d["k"]
        d_dim = q.shape[1]
        N = q.shape[0]
        idx_q = rng.integers(0, N, size=n_pairs_per_lh)
        idx_k = rng.integers(0, N, size=n_pairs_per_lh)
        s = (q[idx_q] * k[idx_k]).sum(axis=-1) / math.sqrt(d_dim)

        ax = axes[0][c]
        bins = np.linspace(
            min(s.min(), gauss_score.min()),
            max(s.max(), gauss_score.max()),
            120,
        )
        ax.hist(s, bins=bins, alpha=0.6, density=True, color="tab:blue", label="GPT-2")
        ax.hist(gauss_score, bins=bins, alpha=0.4, density=True, color="tab:orange",
                label=r"$\mathcal{N}(0,I)$")
        ax.set_title(f"L{l} H{h}", fontsize=9)
        ax.set_xlabel(r"$q\cdot k/\sqrt{d}$")
        if c == 0:
            ax.set_ylabel("density")
            ax.legend(fontsize=7)
    fig.suptitle(r"Score $q\cdot k/\sqrt{d}$ histogram (GPT-2 vs Gaussian baseline)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(save_path, dpi=150)
    print(f"saved -> {save_path}")
    plt.close(fig)


def plot_entropy(
    data: dict[tuple[int, int], dict],
    pairs: list[tuple[int, int]],
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bins = np.linspace(0, 1, 60)
    for (l, h) in pairs:
        d = data[(l, h)]
        if "entropy" not in d or len(d["entropy"]) == 0:
            continue
        ax.hist(d["entropy"], bins=bins, alpha=0.45, density=True,
                label=f"L{l}H{h}")
    ax.axvline(1.0, color="black", lw=0.7, ls="--", label="uniform attention")
    ax.set_xlabel("normalized entropy of softmax row")
    ax.set_ylabel("density")
    ax.set_title("Attention spikiness (lower = spikier)")
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"saved -> {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Stats summary
# ---------------------------------------------------------------------------


def summarize(
    data: dict[tuple[int, int], dict],
    gauss: dict,
) -> dict:
    out: dict[str, dict] = {}
    for (l, h), d in data.items():
        q, k = d["q"], d["k"]
        d_dim = q.shape[1]
        rng = np.random.default_rng(42)
        N = q.shape[0]
        idx_q = rng.integers(0, N, size=min(50_000, N * N))
        idx_k = rng.integers(0, N, size=min(50_000, N * N))
        s = (q[idx_q] * k[idx_k]).sum(axis=-1) / math.sqrt(d_dim)

        out[f"L{l}H{h}"] = {
            "n_tokens": int(N),
            "q_dim_mean_max_abs": float(np.abs(q.mean(axis=0)).max()),
            "q_dim_mean_l2": float(np.linalg.norm(q.mean(axis=0))),
            "k_dim_mean_max_abs": float(np.abs(k.mean(axis=0)).max()),
            "k_dim_mean_l2": float(np.linalg.norm(k.mean(axis=0))),
            "q_dim_std_mean": float(q.std(axis=0).mean()),
            "k_dim_std_mean": float(k.std(axis=0).mean()),
            "q_norm_mean": float(np.linalg.norm(q, axis=-1).mean()),
            "q_norm_std": float(np.linalg.norm(q, axis=-1).std()),
            "k_norm_mean": float(np.linalg.norm(k, axis=-1).mean()),
            "k_norm_std": float(np.linalg.norm(k, axis=-1).std()),
            "score_mean": float(s.mean()),
            "score_std": float(s.std()),
            "score_skew": float(scistats.skew(s)),
            "score_kurtosis": float(scistats.kurtosis(s)),
            "entropy_mean": float(d["entropy"].mean()) if len(d["entropy"]) else None,
            "entropy_median": float(np.median(d["entropy"])) if len(d["entropy"]) else None,
        }

    gs = gauss["score"]
    out["__gaussian_baseline__"] = {
        "n_pairs": int(len(gs)),
        "score_mean": float(gs.mean()),
        "score_std": float(gs.std()),
        "score_skew": float(scistats.skew(gs)),
        "score_kurtosis": float(scistats.kurtosis(gs)),
    }
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_pairs(items: list[str]) -> list[tuple[int, int]]:
    out = []
    for s in items:
        a, b = s.split(",")
        out.append((int(a), int(b)))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="gpt2")
    p.add_argument("--n-docs", type=int, default=200)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--pairs", nargs="+", default=["2,2", "4,4", "6,6", "8,8", "10,10"],
                   help="若干 'layer,head' 对，空格分隔")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/qk_diagnostic"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs = parse_pairs(args.pairs)

    print(f"collecting q,k from {len(pairs)} (layer,head) pairs over {args.n_docs} docs...")
    data = collect(args.model_path, args.n_docs, args.max_length, pairs, args.device)

    n_token_total = sum(d["q"].shape[0] for d in data.values() if isinstance(d["q"], np.ndarray))
    print(f"collected total {n_token_total} q,k tokens (sum across pairs)")

    gauss = gaussian_reference(50_000, D_HEAD, seed=args.seed)

    plot_dim_wise(data, pairs, args.output_dir / f"qk_dim_wise_{args.n_docs}docs.png")
    plot_norms(data, pairs, args.output_dir / f"qk_norm_hist_{args.n_docs}docs.png")
    plot_scores(data, pairs, gauss["score"],
                args.output_dir / f"qk_score_hist_{args.n_docs}docs.png")
    plot_entropy(data, pairs,
                 args.output_dir / f"attn_entropy_{args.n_docs}docs.png")

    stats = summarize(data, gauss)
    stats_path = args.output_dir / f"qk_stats_{args.n_docs}docs.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved -> {stats_path}")


if __name__ == "__main__":
    main()
