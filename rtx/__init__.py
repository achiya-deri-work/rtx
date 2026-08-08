"""RTX Blackwell low-precision linear layers."""

from ._version import __version__

from .fp8 import (
    DEFAULT_MXFP8_PREQUANT_CONFIG,
    MXFP8Linear,
    MXFP8PrequantConfig,
    mxfp8_linear,
)
from .fp8_bwd import (
    DEFAULT_MXFP8_BWD_CONFIG,
    MXFP8BwdConfig,
    MXFP8BwdMatmulConfig,
    mxfp8_linear_backward,
)
from .fp4 import NVFP4Linear, nvfp4_linear

__all__ = [
    "__version__",
    "CoordinateDescentPolicy",
    "BWD_COORDINATE_ORDER",
    "BWD_SEARCH_SPACE",
    "BwdTuningResult",
    "DEFAULT_MXFP8_PREQUANT_CONFIG",
    "DEFAULT_MXFP8_BWD_CONFIG",
    "MXFP8Linear",
    "NVFP4Linear",
    "MXFP8BwdConfig",
    "MXFP8BwdMatmulConfig",
    "MXFP8PrequantConfig",
    "PREQUANT_COORDINATE_ORDER",
    "PREQUANT_SEARCH_SPACE",
    "PrequantTuningResult",
    "load_cached_mxfp8_fwd_config",
    "load_cached_mxfp8_bwd_config",
    "load_mxfp8_bwd_config",
    "load_cached_mxfp8_prequant_config",
    "mxfp8_linear",
    "mxfp8_linear_backward",
    "nvfp4_linear",
    "tune_mxfp8_fwd",
    "tune_mxfp8_backward",
    "tune_mxfp8_prequant",
]


def __getattr__(name: str):
    if name in {
        "BWD_COORDINATE_ORDER",
        "BWD_SEARCH_SPACE",
        "BwdTuningResult",
        "load_cached_mxfp8_bwd_config",
        "load_mxfp8_bwd_config",
        "tune_mxfp8_backward",
    }:
        from . import bwd_autotune

        return getattr(bwd_autotune, name)
    if name in {
        "CoordinateDescentPolicy",
        "load_cached_mxfp8_fwd_config",
        "tune_mxfp8_fwd",
    }:
        from . import autotune

        return getattr(autotune, name)
    if name in {
        "PREQUANT_COORDINATE_ORDER",
        "PREQUANT_SEARCH_SPACE",
        "PrequantTuningResult",
        "load_cached_mxfp8_prequant_config",
        "tune_mxfp8_prequant",
    }:
        from . import prequant_autotune

        return getattr(prequant_autotune, name)
    raise AttributeError(name)
