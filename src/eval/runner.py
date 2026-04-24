from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass

import torch

from src.eval.metrics import output_numer_denom, relerr_kernel_pairs
from src.feature_maps import build_feature_map
from src.feature_maps.base import FeatureMap
from src.sampling import (
    collect_qk_gpt2_wikitext,
    sample_qk_gaussian,
    sample_qkv_gaussian,
)
from src.sampling.gpt2_wikitext import iter_qkv_gpt2_wikitext


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    setting: str
    feature_map: str
    dim_m: int
    trial: int
    relerr_kernel: float
    n_pairs: int
    layer: int | None = None
    head: int | None = None


@dataclass
class EvalResultOut:
    setting: str
    feature_map: str
    dim_m: int
    trial: int
    relerr_output: float
    n_seqs: int
    seq_len: int
    layer: int | None = None
    head: int | None = None


# ---------------------------------------------------------------------------
# omega / phi 构建：嵌套截断（一次采 max_m 份，各 m 共用前缀）
# ---------------------------------------------------------------------------


def _build_phis_prefix(
    map_names: list[str],
    dim_d: int,
    dims_m: list[int],
    n_trials: int,
    seed: int,
    device: torch.device,
) -> dict[tuple[int, int, int], FeatureMap]:
    """返回 {(imap, trial, dim_m): phi}。各 (imap, trial) 共用 omega_full[max_m, d]。"""
    max_m = max(dims_m)
    phis: dict[tuple[int, int, int], FeatureMap] = {}
    for imap, map_name in enumerate(map_names):
        for trial in range(n_trials):
            g = torch.Generator(device="cpu")
            g.manual_seed(seed + 10007 * trial + imap * 1009)
            omega_full = torch.randn(max_m, dim_d, generator=g, device=torch.device("cpu"))
            for dim_m in dims_m:
                phi = build_feature_map(
                    map_name, dim_d, dim_m, omega=omega_full[:dim_m]
                ).to(device)
                phis[(imap, trial, dim_m)] = phi
    return phis


# ---------------------------------------------------------------------------
# RelErr_ker：pair-level
# ---------------------------------------------------------------------------


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
    n_pairs = q.shape[0]
    phis = _build_phis_prefix(map_names, dim_d, dims_m, n_trials, seed, device)
    results: list[EvalResult] = []
    for (imap, trial, dim_m), phi in phis.items():
        err = relerr_kernel_pairs(q, k, phi, dim_d)
        results.append(
            EvalResult(setting, map_names[imap], dim_m, trial, err, n_pairs)
        )
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


# ---------------------------------------------------------------------------
# RelErr_out：序列级 attention output
# ---------------------------------------------------------------------------


def run_gaussian_output(
    *,
    dim_d: int,
    dim_v: int,
    n_seq: int,
    seq_len: int,
    map_names: list[str],
    dims_m: list[int],
    n_trials: int,
    seed: int,
    device: torch.device,
) -> list[EvalResultOut]:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    Q, K, V = sample_qkv_gaussian(
        n_seq, seq_len, dim_d, dim_v, generator=gen, device=device
    )
    phis = _build_phis_prefix(map_names, dim_d, dims_m, n_trials, seed, device)

    results: list[EvalResultOut] = []
    for (imap, trial, dim_m), phi in phis.items():
        num_sq, den_sq = output_numer_denom(Q, K, V, phi, dim_d)
        err = float(math.sqrt(num_sq / max(den_sq, 1e-12)))
        results.append(
            EvalResultOut(
                setting="gaussian",
                feature_map=map_names[imap],
                dim_m=dim_m,
                trial=trial,
                relerr_output=err,
                n_seqs=n_seq,
                seq_len=seq_len,
            )
        )
    return results


def run_gpt2_output(
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
) -> list[EvalResultOut]:
    """对每个 (layer, head) 按 doc 流式累加 ||·||_F^2，避免同时缓存所有 QKV。"""
    phis = _build_phis_prefix(map_names, dim_d, dims_m, n_trials, seed, device)

    # 累加：(imap, trial, dim_m, layer, head) -> [num_sum, den_sum, n_seqs, tot_len]
    acc: dict[tuple[int, int, int, int, int], list[float]] = defaultdict(
        lambda: [0.0, 0.0, 0, 0]
    )

    for doc_idx, li, hi, q_seq, k_seq, v_seq in iter_qkv_gpt2_wikitext(
        model_path=model_path,
        n_docs=n_docs,
        max_length=max_length,
        layers=layers,
        heads=heads,
        device=device,
    ):
        q_seq = q_seq.to(device=device, dtype=torch.float32)
        k_seq = k_seq.to(device=device, dtype=torch.float32)
        v_seq = v_seq.to(device=device, dtype=torch.float32)
        seq_len = q_seq.shape[-2]
        for (imap, trial, dim_m), phi in phis.items():
            num_sq, den_sq = output_numer_denom(q_seq, k_seq, v_seq, phi, dim_d)
            entry = acc[(imap, trial, dim_m, li, hi)]
            entry[0] += num_sq
            entry[1] += den_sq
            entry[2] += 1
            entry[3] += seq_len

    results: list[EvalResultOut] = []
    for (imap, trial, dim_m, li, hi), (num_sq, den_sq, n_seqs, tot_len) in acc.items():
        err = float(math.sqrt(num_sq / max(den_sq, 1e-12)))
        avg_len = int(round(tot_len / max(n_seqs, 1)))
        results.append(
            EvalResultOut(
                setting="gpt2_wikitext",
                feature_map=map_names[imap],
                dim_m=dim_m,
                trial=trial,
                relerr_output=err,
                n_seqs=n_seqs,
                seq_len=avg_len,
                layer=li,
                head=hi,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def results_to_json(results: list) -> str:
    return json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False)
