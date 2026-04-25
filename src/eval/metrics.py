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
    dev = q.device
    q64, k64 = q.detach().double(), k.detach().double()
    phi64 = phi.to(device=dev, dtype=torch.float64)

    k_true = softmax_kernel(q64, k64, dim_d)
    k_hat = (phi64(q64) * phi64(k64)).sum(dim=-1)

    mse = ((k_true - k_hat) ** 2).mean()
    denom = (k_true ** 2).mean().clamp_min(1e-12)
    return float((mse / denom).item())


@torch.no_grad()
def exact_attention_output(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    dim_d: int,
) -> torch.Tensor:
    """Exact softmax attention output O = softmax(QK^T/sqrt(d))V。"""
    Q64 = Q.detach().double()
    K64 = K.detach().double()
    V64 = V.detach().double()
    S = torch.matmul(Q64, K64.transpose(-1, -2)) / math.sqrt(dim_d)
    A = torch.softmax(S, dim=-1)
    return torch.matmul(A, V64)


@torch.no_grad()
def output_numer_denom_from_exact(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    O: torch.Tensor,
    phi: FeatureMap,
    *,
    eps: float = 1e-12,
) -> tuple[float, float]:
    """给定 exact O，返回 (||O - O_hat||_F^2, ||O||_F^2)。"""
    dev = Q.device
    Q64 = Q.detach().double()
    K64 = K.detach().double()
    V64 = V.detach().double()
    O64 = O.detach().to(device=dev, dtype=torch.float64)
    phi64 = phi.to(device=dev, dtype=torch.float64)

    # linearized: O_hat_i = phi(q_i)^T (Phi_K^T V) / (phi(q_i)^T sum_l phi(k_l))
    PhiQ = phi64(Q64)
    PhiK = phi64(K64)
    KV = torch.matmul(PhiK.transpose(-1, -2), V64)
    numer_mat = torch.matmul(PhiQ, KV)
    phi_k_sum = PhiK.sum(dim=-2, keepdim=True)
    denom_vec = (PhiQ * phi_k_sum).sum(dim=-1, keepdim=True).clamp_min(eps)
    O_hat = numer_mat / denom_vec

    num_sq = float(((O64 - O_hat) ** 2).sum().item())
    den_sq = float((O64 ** 2).sum().item())
    return num_sq, den_sq


@torch.no_grad()
def output_numer_denom(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    phi: FeatureMap,
    dim_d: int,
    *,
    eps: float = 1e-12,
) -> tuple[float, float]:
    """返回 (||O - O_hat||_F^2, ||O||_F^2)。Q/K/V 形状 [..., n, d] / [..., n, d_v]。"""
    O = exact_attention_output(Q, K, V, dim_d)
    return output_numer_denom_from_exact(Q, K, V, O, phi, eps=eps)


def relerr_output(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    phi: FeatureMap,
    dim_d: int,
    *,
    eps: float = 1e-12,
) -> float:
    """RelErr_out = ||AV - AhatV||_F / ||AV||_F，跨 batch 聚合为 sqrt(sum_num / sum_den)。"""
    num_sq, den_sq = output_numer_denom(Q, K, V, phi, dim_d, eps=eps)
    return float(math.sqrt(num_sq / max(den_sq, eps)))
