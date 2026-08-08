"""Portable logical container for two-level-scaled NVFP4 operands."""

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
class NVFP4Tensor:
    """Packed E2M1 values, E4M3 scales per 16 values, and one FP32 scale.

    ``shape`` is the logical BF16/FP4 shape. PyTorch's
    ``float4_e2m1fn_x2`` tensor shape counts packed bytes, so ``data`` has
    half as many columns as the logical matrix.
    """

    data: torch.Tensor
    block_scales: torch.Tensor
    tensor_scale: torch.Tensor
    shape: tuple[int, ...]
    scale_layout: ScaleLayout = "row_major"
    orientation: Orientation = "row_major"
    schema_version: int = PACKED_OPERAND_SCHEMA_VERSION

    def __post_init__(self) -> None:
        rows, k = flattened_matrix_shape(self.shape)
        fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
        if self.schema_version != PACKED_OPERAND_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported NVFP4 packed schema {self.schema_version}"
            )
        if k % 16:
            raise ValueError("NVFP4 packed K must be divisible by 16")
        if fp4_dtype is None or self.data.dtype is not fp4_dtype:
            raise TypeError("NVFP4 data must use torch.float4_e2m1fn_x2")
        if self.block_scales.dtype is not torch.float8_e4m3fn:
            raise TypeError("NVFP4 block scales must use float8_e4m3fn")
        if self.tensor_scale.dtype is not torch.float32 or self.tensor_scale.numel() != 1:
            raise TypeError("NVFP4 tensor_scale must be one FP32 value")
        if not (
            self.data.device == self.block_scales.device == self.tensor_scale.device
        ):
            raise ValueError("NVFP4 data and both scale levels must share one device")
        expected_data = (rows, k // 2)
        if tuple(self.data.shape) != expected_data:
            raise ValueError(
                f"NVFP4 packed data shape must be {expected_data} for logical "
                f"shape {(rows, k)}, got {tuple(self.data.shape)}"
            )
        if self.scale_layout == "row_major":
            expected = (rows, k // 16)
            if tuple(self.block_scales.shape) != expected:
                raise ValueError(
                    f"NVFP4 row-major scales must have shape {expected}, "
                    f"got {tuple(self.block_scales.shape)}"
                )
        elif self.scale_layout in ("mma64", "mma128"):
            raise ValueError(
                "tensor-core-native NVFP4 scale layouts are reserved but not "
                "defined until the NVFP4 kernel is implemented"
            )
        else:
            raise ValueError(f"unknown NVFP4 scale layout {self.scale_layout!r}")
        if self.orientation not in ("row_major", "transpose"):
            raise ValueError(f"unknown NVFP4 orientation {self.orientation!r}")

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
    ) -> "NVFP4Tensor":
        return NVFP4Tensor(
            move_tensor(self.data, device, non_blocking),
            move_tensor(self.block_scales, device, non_blocking),
            move_tensor(self.tensor_scale, device, non_blocking),
            self.shape,
            self.scale_layout,
            self.orientation,
            self.schema_version,
        )

    def detach(self) -> "NVFP4Tensor":
        return NVFP4Tensor(
            self.data.detach(),
            self.block_scales.detach(),
            self.tensor_scale.detach(),
            self.shape,
            self.scale_layout,
            self.orientation,
            self.schema_version,
        )


def _flatten(value: NVFP4Tensor):
    return [value.data, value.block_scales, value.tensor_scale], (
        value.shape,
        value.scale_layout,
        value.orientation,
        value.schema_version,
    )


def _unflatten(values, context):
    shape, scale_layout, orientation, schema_version = context
    return NVFP4Tensor(
        values[0],
        values[1],
        values[2],
        shape,
        scale_layout,
        orientation,
        schema_version,
    )


_pytree.register_pytree_node(NVFP4Tensor, _flatten, _unflatten)


__all__ = ["NVFP4Tensor"]
