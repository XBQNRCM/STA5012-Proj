import torch

from .base import FeatureMap
from .performer import PerformerFeatureMap

REGISTRY: dict[str, type[FeatureMap]] = {
    "performer": PerformerFeatureMap,
}


def build_feature_map(name: str, dim_d: int, dim_m: int, generator: torch.Generator | None = None) -> FeatureMap:
    return REGISTRY[name](dim_d, dim_m, generator)
