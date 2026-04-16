from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import torch
from tqdm import tqdm

from src.eval.metrics import relerr_kernel_pairs
from src.feature_maps import build_feature_map
from src.sampling import sample_qk_gaussian, collect_qk_gpt2_wikitext


@dataclass
class EvalResult:
    setting: str
    feature_map: str
    dim_m: int
    trial: int
    relerr_kernel: float
    n_pairs: int


def _eval_loop(
    setting: str,
    q: torch.Tensor,
    k: torch.Tensor,
    dim_d: int,
    map_names: list[str],
    dims_m: list[int],
    n_trials: int,
    seed: int,
    device: torch.device,
) -> list[EvalResult]:
    """对给定的 (q, k) 遍历实验组合：map × m × trial。"""
    n_pairs = q.shape[0]
    results: list[EvalResult] = []

    for imap, map_name in enumerate(map_names):
        for trial in range(n_trials):
            for dim_m in dims_m:
                g = torch.Generator(device=device)
                g.manual_seed(seed + 10007 * trial + imap * 1009 + dim_m)
                phi = build_feature_map(map_name, dim_d, dim_m, g)
                err = relerr_kernel_pairs(q, k, phi, dim_d)
                results.append(EvalResult(setting, map_name, dim_m, trial, err, n_pairs))

    return results


def run_gaussian(
    *,
    dim_d: int,
    n_pairs: int,
    map_names: list[str],
    dims_m: list[int],
    n_trials: int,
    seed: int,
    device: torch.device,
) -> list[EvalResult]:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    q, k = sample_qk_gaussian(n_pairs, dim_d, generator=gen, device=device)
    return _eval_loop("gaussian", q, k, dim_d, map_names, dims_m, n_trials, seed, device)


def run_gpt2(
    *,
    dim_d: int,
    map_names: list[str],
    dims_m: list[int],
    n_trials: int,
    seed: int,
    device: torch.device,
    model_path: str,
    n_docs: int,
    max_length: int,
    layers: list[int] | None,
    heads: list[int] | None,
    token_pos: int,
) -> list[EvalResult]:
    q, k = collect_qk_gpt2_wikitext(
        model_path=model_path,
        n_docs=n_docs,
        max_length=max_length,
        layers=layers,
        heads=heads,
        token_pos=token_pos,
        device=device,
    )
    q = q.to(device=device, dtype=torch.float32)
    k = k.to(device=device, dtype=torch.float32)
    return _eval_loop("gpt2_wikitext", q, k, dim_d, map_names, dims_m, n_trials, seed, device)


def results_to_json(results: list[EvalResult]) -> str:
    return json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False)
