#!/usr/bin/env python3
"""统一入口：Gaussian / GPT-2 采样 + kernel 相对误差评测。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.constants import D_HEAD
from src.eval.runner import results_to_json, run_gaussian, run_gpt2


def _parse_int_list(s: str | None) -> list[int] | None:
    if not s or not s.strip():
        return None
    return [int(x) for x in s.split(",") if x.strip()]


def _build_filename(args) -> str:
    maps_str = "-".join(args.maps)
    m_str = "m-" + "-".join(str(m) for m in args.dims_m)
    trials_str = f"{args.n_trials}trials"
    seed_str = f"seed{args.seed}"

    if args.mode == "gaussian":
        parts = [args.mode, maps_str, m_str, f"{args.n_pairs}pairs", trials_str, seed_str]
    else:
        layers = _parse_int_list(args.layers)
        heads = _parse_int_list(args.heads)
        layers_str = "layers-" + ("-".join(str(l) for l in layers) if layers else "all")
        heads_str = "heads-" + ("-".join(str(h) for h in heads) if heads else "all")
        pos_str = f"pos{args.token_pos}"
        parts = [args.mode, maps_str, m_str, f"{args.n_docs}docs", trials_str, seed_str, layers_str, heads_str, pos_str]

    return "_".join(parts)


def save_results(results, base_dir: Path, run_name: str) -> Path:
    out_dir = base_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "results.json"
    path.write_text(results_to_json(results), encoding="utf-8")
    print(f"saved → {path}")
    return path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("gaussian", "gpt2"), required=True)
    p.add_argument("--maps", nargs="+", default=["performer"])
    p.add_argument("--m", dest="dims_m", type=int, nargs="+", default=[32, 64, 128, 256])
    p.add_argument("--dim-d", type=int, default=D_HEAD)
    p.add_argument("--n-pairs", type=int, default=10_000, help="Gaussian: (q,k) 对数")
    p.add_argument("--n-trials", type=int, default=1, help="omega 重采样次数")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    # GPT-2 专用
    p.add_argument("--model-path", default="gpt2")
    p.add_argument("--n-docs", type=int, default=128)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--layers", type=str, default=None, help="逗号分隔，如 0,1,11")
    p.add_argument("--heads", type=str, default=None, help="逗号分隔")
    p.add_argument("--token-pos", type=int, default=-2, help="-2=中点, -1=末尾")
    args = p.parse_args()

    device = torch.device(args.device)

    if args.mode == "gaussian":
        results = run_gaussian(
            dim_d=args.dim_d,
            n_pairs=args.n_pairs,
            map_names=args.maps,
            dims_m=args.dims_m,
            n_trials=args.n_trials,
            seed=args.seed,
            device=device,
        )
    else:
        results = run_gpt2(
            dim_d=args.dim_d,
            map_names=args.maps,
            dims_m=args.dims_m,
            n_trials=args.n_trials,
            seed=args.seed,
            device=device,
            model_path=args.model_path,
            n_docs=args.n_docs,
            max_length=args.max_length,
            layers=_parse_int_list(args.layers),
            heads=_parse_int_list(args.heads),
            token_pos=args.token_pos,
        )

    save_results(results, args.output_dir, _build_filename(args))



if __name__ == "__main__":
    main()
