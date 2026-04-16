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
