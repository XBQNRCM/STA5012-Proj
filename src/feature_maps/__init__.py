from .base import FeatureMap
from .performer import PerformerFeatureMap

REGISTRY: dict[str, type[FeatureMap]] = {
    "performer": PerformerFeatureMap,
}


def build_feature_map(name: str, dim_d: int, dim_m: int, **kwargs) -> FeatureMap:
    """各 map 的 ``__init__`` 自行从 kwargs 取所需参数；不需要的应写 ``**kwargs`` 吞掉。"""
    return REGISTRY[name](dim_d, dim_m, **kwargs)
