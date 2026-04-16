from __future__ import annotations

import math

import torch

from src.feature_maps.base import FeatureMap


def softmax_kernel(q: torch.Tensor, k: torch.Tensor, dim_d: int) -> torch.Tensor:
    """K(q,k) = exp(q·k / sqrt(d))，q/k 形状 [..., d]，返回 [...]。"""
    return torch.exp((q * k).sum(dim=-1) / math.sqrt(dim_d))


def relerr_kernel_pairs(
    q: torch.Tensor,
    k: torch.Tensor,
    phi: FeatureMap,
    dim_d: int,
) -> float:
    """RelErr_ker = E[(K - Khat)^2] / E[K^2]，用样本均值估计。q, k: [N, d]。"""
    # 用 float64 避免 exp 溢出
    q64, k64 = q.detach().double(), k.detach().double()
    phi64 = phi.to(dtype=torch.float64)

    k_true = softmax_kernel(q64, k64, dim_d)
    k_hat = (phi64(q64) * phi64(k64)).sum(dim=-1)

    mse = ((k_true - k_hat) ** 2).mean()
    denom = (k_true ** 2).mean().clamp_min(1e-12)
    return float((mse / denom).item())
