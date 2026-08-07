"""RTX Blackwell low-precision linear layers."""

from .fp8 import MXFP8Linear, mxfp8_linear

__all__ = [
    "CoordinateDescentPolicy",
    "MXFP8Linear",
    "load_cached_mxfp8_fwd_config",
    "mxfp8_linear",
    "tune_mxfp8_fwd",
]


def __getattr__(name: str):
    if name in {
        "CoordinateDescentPolicy",
        "load_cached_mxfp8_fwd_config",
        "tune_mxfp8_fwd",
    }:
        from . import autotune

        return getattr(autotune, name)
    raise AttributeError(name)
