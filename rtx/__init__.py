"""RTX Blackwell low-precision linear layers."""

from .fp8 import (
    DEFAULT_MXFP8_PREQUANT_CONFIG,
    MXFP8Linear,
    MXFP8PrequantConfig,
    mxfp8_linear,
)

__all__ = [
    "CoordinateDescentPolicy",
    "DEFAULT_MXFP8_PREQUANT_CONFIG",
    "MXFP8Linear",
    "MXFP8PrequantConfig",
    "PREQUANT_COORDINATE_ORDER",
    "PREQUANT_SEARCH_SPACE",
    "PrequantTuningResult",
    "load_cached_mxfp8_fwd_config",
    "load_cached_mxfp8_prequant_config",
    "mxfp8_linear",
    "tune_mxfp8_fwd",
    "tune_mxfp8_prequant",
]


def __getattr__(name: str):
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
