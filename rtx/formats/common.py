"""Shared metadata and validation for packed low-precision operands."""

from __future__ import annotations

from math import prod
from typing import Literal

import torch


ScaleLayout = Literal["row_major", "mma64", "mma128"]
Orientation = Literal["row_major", "transpose"]
LinearOperandState = Literal[
    "dynamic", "weight_prequantized", "fully_prequantized"
]
PACKED_OPERAND_SCHEMA_VERSION = 1
SCALE_LAYOUT_CODES = {"row_major": 0, "mma64": 1, "mma128": 2}
SCALE_LAYOUT_NAMES = {value: key for key, value in SCALE_LAYOUT_CODES.items()}


def flattened_matrix_shape(shape: tuple[int, ...]) -> tuple[int, int]:
    if not shape:
        raise ValueError("a packed operand must have at least one dimension")
    return prod(shape[:-1]) if len(shape) > 1 else 1, shape[-1]


def reject_packed_dtype_conversion(
    args: tuple[object, ...],
    kwargs: dict[str, object],
    *,
    format_name: str,
) -> None:
    """Reject ``Module.to`` calls that would cast heterogeneous packed data."""

    _device, dtype, _non_blocking, _memory_format = torch._C._nn._parse_to(
        *args, **kwargs
    )
    if dtype is not None:
        raise TypeError(
            f"a prequantized {format_name} module cannot be cast to {dtype}; "
            "move it with .to(device=...) and requantize from the BF16 master "
            "weight to change numeric formats"
        )


__all__ = [
    "Orientation",
    "LinearOperandState",
    "PACKED_OPERAND_SCHEMA_VERSION",
    "SCALE_LAYOUT_CODES",
    "SCALE_LAYOUT_NAMES",
    "ScaleLayout",
    "flattened_matrix_shape",
    "reject_packed_dtype_conversion",
]
