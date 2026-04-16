from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class FeatureMap(nn.Module, ABC):
    """phi: R^d -> R^m，子类实现 forward 即可。"""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., d] -> [..., m]"""
