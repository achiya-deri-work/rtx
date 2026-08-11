"""TorchAO-backed NVFP4 operand helpers for native RTX kernels."""

from __future__ import annotations

from math import ceil

import torch
from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

from .common import Orientation, ScaleLayout, flattened_matrix_shape


def make_nvfp4_tensor(
    qdata: torch.Tensor,
    scales: torch.Tensor,
    tensor_scale: torch.Tensor,
    shape: tuple[int, ...],
    scale_layout: ScaleLayout = "row_major",
) -> NVFP4Tensor:
    rows, k = flattened_matrix_shape(shape)
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    if qdata.dtype not in (torch.uint8, fp4_dtype):
        raise TypeError(
            "NVFP4 qdata must use TorchAO's uint8 container or "
            "torch.float4_e2m1fn_x2"
        )
    if (qdata.numel() * 2) % rows:
        raise ValueError("NVFP4 qdata rows do not divide its packed storage")
    storage_k = qdata.numel() * 2 // rows
    expected_storage_k = (k + 15) // 16 * 16
    if storage_k != expected_storage_k:
        raise ValueError(
            f"NVFP4 qdata storage K={storage_k} cannot represent logical K={k}; "
            f"expected the minimal block-aligned K={expected_storage_k}"
        )
    if scales.dtype is not torch.float8_e4m3fn:
        raise TypeError("NVFP4 block scales must use float8_e4m3fn")
    if tensor_scale.dtype is not torch.float32 or tensor_scale.numel() != 1:
        raise TypeError("NVFP4 per-tensor scale must be one FP32 value")
    if not (qdata.device == scales.device == tensor_scale.device):
        raise ValueError("NVFP4 qdata and both scale levels must share one device")
    if scale_layout == "row_major":
        expected_scales = rows * (storage_k // 16)
        scale_shape = (*shape[:-1], storage_k // 16)
    elif scale_layout == "mma128":
        if storage_k != k:
            raise ValueError("blocked NVFP4 scales require aligned logical K")
        expected_scales = ceil(rows / 128) * ceil(k / 64) * 512
        scale_shape = (ceil(rows / 128) * 32, ceil(k / 64) * 16)
    else:
        raise ValueError(
            "TorchAO NVFP4 supports row-major or standard blocked scales"
        )
    if scales.numel() != expected_scales:
        raise ValueError(
            f"NVFP4 {scale_layout} scales have {scales.numel()} values, "
            f"expected {expected_scales}"
        )
    value = NVFP4Tensor(
        qdata.view(*shape[:-1], storage_k // 2),
        scales.view(scale_shape),
        16,
        torch.bfloat16,
        tensor_scale.reshape(()),
        is_swizzled_scales=scale_layout != "row_major",
    )
    value._rtx_scale_layout = scale_layout
    value._rtx_logical_shape = tuple(int(dim) for dim in shape)
    return value


def validate_nvfp4_tensor(value: NVFP4Tensor) -> None:
    if value.block_size != 16:
        raise ValueError(f"RTX NVFP4 requires block_size=16, got {value.block_size}")
    if value.orig_dtype is not torch.bfloat16:
        raise TypeError(
            f"RTX NVFP4 requires BF16 logical dtype, got {value.orig_dtype}"
        )
    rows, k = nvfp4_matrix_shape(value)
    storage_k = int(value.qdata.shape[-1]) * 2
    if storage_k < k or storage_k % 16:
        raise ValueError("RTX NVFP4 storage K must cover logical K in blocks of 16")
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    if value.qdata.dtype not in (torch.uint8, fp4_dtype):
        raise TypeError("RTX NVFP4 requires uint8 or float4_e2m1fn_x2 qdata")
    if value.qdata.numel() != rows * (storage_k // 2):
        raise ValueError("TorchAO NVFP4 qdata does not match its logical shape")
    if value.scale.dtype is not torch.float8_e4m3fn:
        raise TypeError("RTX NVFP4 requires E4M3 block scales")
    if value.per_tensor_scale is not None:
        if (
            value.per_tensor_scale.dtype is not torch.float32
            or value.per_tensor_scale.numel() != 1
        ):
            raise TypeError("RTX NVFP4 per_tensor_scale must be one FP32 value")
        if value.per_tensor_scale.device != value.qdata.device:
            raise ValueError("NVFP4 qdata and per-tensor scale must share a device")
    if value.qdata.device != value.scale.device:
        raise ValueError("NVFP4 qdata and block scales must share one device")
    layout = nvfp4_scale_layout(value)
    expected_scales = (
        rows * (storage_k // 16)
        if layout == "row_major"
        else ceil(rows / 128) * ceil(k / 64) * 512
    )
    if value.scale.numel() != expected_scales:
        raise ValueError(
            f"NVFP4 {layout} scale storage has {value.scale.numel()} values, "
            f"expected {expected_scales}"
        )


def nvfp4_orientation(value: NVFP4Tensor) -> Orientation:
    if value.qdata.is_contiguous():
        return "row_major"
    if value.qdata.ndim == 2 and value.qdata.t().is_contiguous():
        return "transpose"
    raise ValueError("RTX kernels require row-major or logical-transpose qdata")


def nvfp4_matrix_shape(value: NVFP4Tensor) -> tuple[int, int]:
    logical_shape = getattr(value, "_rtx_logical_shape", None)
    return flattened_matrix_shape(
        tuple(int(dim) for dim in (logical_shape or value.shape))
    )


def nvfp4_scale_layout(value: NVFP4Tensor) -> ScaleLayout:
    marker = getattr(value, "_rtx_scale_layout", None)
    if marker == "row_major" and not value.is_swizzled_scales:
        return marker
    if marker == "mma128" and value.is_swizzled_scales:
        return marker
    return "mma128" if value.is_swizzled_scales else "row_major"


def nvfp4_tensor_scale(value: NVFP4Tensor) -> torch.Tensor:
    """Return TorchAO's optional global decode scale as an explicit scalar."""

    validate_nvfp4_tensor(value)
    if value.per_tensor_scale is None:
        return torch.ones((), dtype=torch.float32, device=value.qdata.device)
    return value.per_tensor_scale.reshape(())


__all__ = [
    "NVFP4Tensor",
    "make_nvfp4_tensor",
    "nvfp4_matrix_shape",
    "nvfp4_orientation",
    "nvfp4_scale_layout",
    "nvfp4_tensor_scale",
    "validate_nvfp4_tensor",
]
