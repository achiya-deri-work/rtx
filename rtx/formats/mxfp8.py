"""Portable logical container for E4M3 values with E8M0 block scales."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils import _pytree

from .common import (
    Orientation,
    PACKED_OPERAND_SCHEMA_VERSION,
    ScaleLayout,
    flattened_matrix_shape,
    move_tensor,
)


@dataclass(frozen=True, slots=True)
class MXFP8Tensor:
    """A prequantized MXFP8 operand independent of any linear module.

    ``data`` is logically shaped like ``shape`` but stored as a flattened 2D
    E4M3 matrix. ``scales`` contains one E8M0 scale per 32 values in either a
    portable row-major or tensor-core-native physical layout.
    """

    data: torch.Tensor
    scales: torch.Tensor
    shape: tuple[int, ...]
    scale_layout: ScaleLayout = "row_major"
    orientation: Orientation = "row_major"
    schema_version: int = PACKED_OPERAND_SCHEMA_VERSION

    def __post_init__(self) -> None:
        rows, k = flattened_matrix_shape(self.shape)
        if self.schema_version != PACKED_OPERAND_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported MXFP8 packed schema {self.schema_version}"
            )
        if k % 32:
            raise ValueError("MXFP8 packed K must be divisible by 32")
        if self.data.dtype is not torch.float8_e4m3fn:
            raise TypeError(f"MXFP8 data must be float8_e4m3fn, got {self.data.dtype}")
        if self.scales.dtype is not torch.float8_e8m0fnu:
            raise TypeError(
                f"MXFP8 scales must be float8_e8m0fnu, got {self.scales.dtype}"
            )
        if self.data.device != self.scales.device:
            raise ValueError("MXFP8 data and scales must share one device")
        if tuple(self.data.shape) != (rows, k):
            raise ValueError(
                f"MXFP8 data shape must be flattened {(rows, k)}, got {self.data.shape}"
            )
        if self.scale_layout == "row_major":
            expected = (rows, k // 32)
        elif self.scale_layout in ("mma64", "mma128"):
            tile_rows = 64 if self.scale_layout == "mma64" else 128
            if rows % tile_rows or k % 128:
                raise ValueError(
                    f"{self.scale_layout} scales require rows divisible by "
                    f"{tile_rows} and K divisible by 128"
                )
            expected = (rows // tile_rows, k // 128, 512)
        else:
            raise ValueError(f"unknown MXFP8 scale layout {self.scale_layout!r}")
        if tuple(self.scales.shape) != expected:
            raise ValueError(
                f"MXFP8 {self.scale_layout} scales must have shape {expected}, "
                f"got {tuple(self.scales.shape)}"
            )
        if self.orientation not in ("row_major", "transpose"):
            raise ValueError(f"unknown MXFP8 orientation {self.orientation!r}")

    @property
    def device(self) -> torch.device:
        return self.data.device

    @property
    def dtype(self) -> torch.dtype:
        return self.data.dtype

    @property
    def matrix_shape(self) -> tuple[int, int]:
        return flattened_matrix_shape(self.shape)

    def to(
        self,
        device: torch.device | str | None = None,
        *,
        non_blocking: bool = False,
    ) -> "MXFP8Tensor":
        """Move packed storage without ever casting its low-precision dtypes."""

        return MXFP8Tensor(
            move_tensor(self.data, device, non_blocking),
            move_tensor(self.scales, device, non_blocking),
            self.shape,
            self.scale_layout,
            self.orientation,
            self.schema_version,
        )

    def detach(self) -> "MXFP8Tensor":
        return MXFP8Tensor(
            self.data.detach(),
            self.scales.detach(),
            self.shape,
            self.scale_layout,
            self.orientation,
            self.schema_version,
        )


def _flatten(value: MXFP8Tensor):
    return [value.data, value.scales], (
        value.shape,
        value.scale_layout,
        value.orientation,
        value.schema_version,
    )


def _unflatten(values, context):
    shape, scale_layout, orientation, schema_version = context
    return MXFP8Tensor(
        values[0], values[1], shape, scale_layout, orientation, schema_version
    )


_pytree.register_pytree_node(MXFP8Tensor, _flatten, _unflatten)


__all__ = ["MXFP8Tensor"]
