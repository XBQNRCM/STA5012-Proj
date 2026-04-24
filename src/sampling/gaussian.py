from __future__ import annotations

import torch


def sample_qk_gaussian(
    n: int,
    dim_d: int,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """q, k ~ N(0, I_d)，返回 [n, d], [n, d]。"""
    dev = device or torch.device("cpu")
    q = torch.randn(n, dim_d, generator=generator, device=dev)
    k = torch.randn(n, dim_d, generator=generator, device=dev)
    return q, k


def sample_qkv_gaussian(
    n_seq: int,
    seq_len: int,
    dim_d: int,
    dim_v: int | None = None,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Q, K ~ N(0, I_d)；V ~ N(0, I_{d_v})。返回 [n_seq, seq_len, d], [..], [n_seq, seq_len, d_v]。"""
    dev = device or torch.device("cpu")
    dv = dim_v if dim_v is not None else dim_d
    Q = torch.randn(n_seq, seq_len, dim_d, generator=generator, device=dev)
    K = torch.randn(n_seq, seq_len, dim_d, generator=generator, device=dev)
    V = torch.randn(n_seq, seq_len, dv, generator=generator, device=dev)
    return Q, K, V
