"""Stable user-facing policy types shared by both linear frontends."""

from typing import Literal


AutotuneMode = Literal["off", "cache", "balanced", "online", "coordinate"]
CanonicalAutotuneMode = Literal["off", "cache", "balanced", "coordinate"]
LinearBackend = Literal["auto", "fused", "materialized"]
MXFP8Backend = Literal["auto", "fused", "materialized", "prequant"]
NVFP4Backend = Literal["auto", "materialized"]
NVFP4ScalingMode = Literal[
    "delayed", "current", "jit_row_region", "block"
]


__all__ = [
    "AutotuneMode",
    "CanonicalAutotuneMode",
    "LinearBackend",
    "MXFP8Backend",
    "NVFP4Backend",
    "NVFP4ScalingMode",
]
