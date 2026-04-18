from __future__ import annotations

import math

import torch

from .base import FeatureMap


class PerformerFeatureMap(FeatureMap):
    """
    Performer / FAVOR+ 正随机特征。

    令 z = x / d^{1/4}，phi_i(z) = exp(w_i^T z - ||z||^2/2) / sqrt(m)，
    w_i ~ N(0, I)，使得 E[phi(z_q)^T phi(z_k)] = exp(z_q^T z_k)
    = exp(x_q^T x_k / sqrt(d))。

    若传入 ``omega``（形状 [m, d]），则使用该矩阵；否则用 ``generator`` 在 CPU 上采样。
    嵌套截断：对同一 trial 先采 ``omega_full[max_m, d]``，再取 ``omega_full[:m]`` 传入即可。
    """

    def __init__(
        self,
        dim_d: int,
        dim_m: int,
        *,
        omega: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        self.dim_d = dim_d
        self.dim_m = dim_m
        if omega is not None:
            if omega.shape != (dim_m, dim_d):
                raise ValueError(f"omega shape must be [{dim_m}, {dim_d}], got {list(omega.shape)}")
            w = omega
        else:
            w = torch.randn(dim_m, dim_d, generator=generator)
        self.register_buffer("omega", w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x * (self.dim_d ** -0.25)
        log_h = z @ self.omega.T - (z * z).sum(dim=-1, keepdim=True) / 2.0
        return torch.exp(log_h) / math.sqrt(self.dim_m)
