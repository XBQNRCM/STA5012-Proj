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
    """

    def __init__(self, dim_d: int, dim_m: int, generator: torch.Generator | None = None) -> None:
        super().__init__()
        self.dim_d = dim_d
        self.dim_m = dim_m
        omega = torch.randn(dim_m, dim_d, generator=generator)
        self.register_buffer("omega", omega)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x * (self.dim_d ** -0.25)
        log_h = z @ self.omega.T - (z * z).sum(dim=-1, keepdim=True) / 2.0
        return torch.exp(log_h) / math.sqrt(self.dim_m)
