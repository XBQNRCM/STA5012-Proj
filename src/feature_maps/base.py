from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class FeatureMap(nn.Module, ABC):
    """phi: R^d -> R^m，子类实现 forward 即可。

    ``build_feature_map(..., **kwargs)`` 会把多余关键字传给子类；不需要的参数请在
    ``__init__(self, dim_d, dim_m, **kwargs)`` 中忽略，以免 runner 传入 ``omega`` 等时报错。
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., d] -> [..., m]"""
