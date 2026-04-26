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
        **kwargs,
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


class ScaledPerformerFeatureMap(PerformerFeatureMap):
    """
    先对输入做逐维经验标准化，再套 Performer。

    forward(q) 和 forward(k) 会分别基于各自输入 batch 的均值/标准差做标准化：
        x_tilde = (x - mean(x)) / (std(x) + eps)

    这对应实验中的 data-adaptive scaled FAVOR+ 思路。
    """

    def __init__(
        self,
        dim_d: int,
        dim_m: int,
        *,
        omega: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        eps: float = 1e-5,
        **kwargs,
    ) -> None:
        super().__init__(dim_d, dim_m, omega=omega, generator=generator)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, x.shape[-1])
        mean = flat.mean(dim=0)
        std = flat.std(dim=0, unbiased=flat.shape[0] > 1)
        x_scaled = (x - mean) / (std + self.eps)
        return super().forward(x_scaled)


class PowerPerformerFeatureMap(PerformerFeatureMap):
    """
    先做逐元素幂归一化，再套 Performer。

        x_tilde = sign(x) * |x|^alpha, 0 < alpha < 1

    alpha 越小，大坐标被压得越强，通常能缓解 exp 随机特征的重尾爆炸。
    """

    def __init__(
        self,
        dim_d: int,
        dim_m: int,
        *,
        omega: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        power_alpha: float = 0.5,
        **kwargs,
    ) -> None:
        if not 0.0 < power_alpha < 1.0:
            raise ValueError(f"power_alpha must be in (0, 1), got {power_alpha}")
        super().__init__(dim_d, dim_m, omega=omega, generator=generator)
        self.power_alpha = power_alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_power = torch.sign(x) * torch.abs(x).pow(self.power_alpha)
        return super().forward(x_power)


class RALAPerformerFeatureMap(PerformerFeatureMap):
    """
    RALA-inspired Performer feature map.

    原 RALA 是 attention block 设计，不是严格的 pairwise kernel feature map。这里实现一个适配
    本项目评测框架的最小版本：

        phi_rala(x) = W_phi( phi_performer(x) * gamma(x) )

    其中 gamma(x) 用当前 batch/sequence 的全局均值向量产生上下文感知权重；W_phi 用相邻
    feature channel 的轻量混合近似实现，用于提高 feature 维之间的交互。
    """

    def __init__(
        self,
        dim_d: int,
        dim_m: int,
        *,
        omega: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        rala_mix: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(dim_d, dim_m, omega=omega, generator=generator)
        self.rala_mix = rala_mix

    def _gamma(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, x.shape[-1])
        if flat.shape[0] == 1:
            return torch.ones(*x.shape[:-1], 1, device=x.device, dtype=x.dtype)

        global_query = flat.mean(dim=0, keepdim=True)
        scores = (flat * global_query).sum(dim=-1) / math.sqrt(self.dim_d)
        weights = flat.shape[0] * torch.softmax(scores, dim=0)
        return weights.reshape(*x.shape[:-1], 1)

    def _mix_channels(self, h: torch.Tensor) -> torch.Tensor:
        if self.rala_mix == 0.0:
            return h
        neighbor = 0.5 * (torch.roll(h, shifts=1, dims=-1) + torch.roll(h, shifts=-1, dims=-1))
        return (h + self.rala_mix * neighbor) / (1.0 + self.rala_mix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = super().forward(x)
        h = h * self._gamma(x).to(dtype=h.dtype)
        return self._mix_channels(h)
