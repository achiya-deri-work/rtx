"""TorchAO-backed MXFP8 operand helpers for RTX kernels."""

from __future__ import annotations

from math import ceil

import torch
from torchao.prototype.mx_formats.mx_tensor import MXTensor
from torchao.quantization.quantize_.common.kernel_preference import (
    KernelPreference,
)

from .common import Orientation, ScaleLayout, flattened_matrix_shape


# Keep the descriptive RTX spelling as a public alias while using TorchAO's
# canonical tensor subclass as the actual runtime type.
MXFP8Tensor = MXTensor


def _logical_qdata_view(
    qdata: torch.Tensor, shape: tuple[int, ...]
) -> torch.Tensor:
    rows, k = flattened_matrix_shape(shape)
    if qdata.dtype is not torch.float8_e4m3fn:
        raise TypeError(f"MXFP8 qdata must be float8_e4m3fn, got {qdata.dtype}")
    if qdata.numel() % rows:
        raise ValueError("MXFP8 qdata rows do not divide its storage")
    storage_k = qdata.numel() // rows
    expected_storage_k = (k + 31) // 32 * 32
    if storage_k != expected_storage_k:
        raise ValueError(
            f"MXFP8 qdata storage K={storage_k} cannot represent logical "
            f"K={k}; expected the minimal block-aligned K={expected_storage_k}"
        )
    return qdata.view(*shape[:-1], storage_k)


def _canonical_scale_view(
    scales: torch.Tensor,
    shape: tuple[int, ...],
    storage_k: int,
    scale_layout: ScaleLayout,
) -> torch.Tensor:
    rows, k = flattened_matrix_shape(shape)
    if scales.dtype is not torch.float8_e8m0fnu:
        raise TypeError(
            f"MXFP8 scales must be float8_e8m0fnu, got {scales.dtype}"
        )
    if storage_k < k or storage_k % 32:
        raise ValueError("MXFP8 storage K must cover logical K in 32-value blocks")
    if scale_layout == "row_major":
        expected = rows * (storage_k // 32)
        canonical_shape = (*shape[:-1], storage_k // 32)
    else:
        if scale_layout not in ("mma64", "mma128"):
            raise ValueError(f"unknown MXFP8 scale layout {scale_layout!r}")
        if storage_k != k or k % 128:
            raise ValueError("tensor-core-native MXFP8 scales require K % 128 == 0")
        tile_rows = 64 if scale_layout == "mma64" else 128
        if rows % tile_rows:
            raise ValueError(
                f"{scale_layout} scales require flattened rows divisible by "
                f"{tile_rows}"
            )
        expected = (rows // tile_rows) * (k // 128) * 512
        if scale_layout == "mma128":
            # This is TorchAO's public blocked-scale shape. It is a zero-copy
            # view of the [M/128, K/128, 512] CuTe kernel representation.
            canonical_shape = (
                ceil(rows / 128) * 32,
                ceil(k / 128) * 16,
            )
        else:
            # SM120's 64-row schedule stores independently padded half tiles.
            canonical_shape = (rows // 64, k // 128, 512)
    if scales.numel() != expected:
        raise ValueError(
            f"MXFP8 {scale_layout} scales have {scales.numel()} values, "
            f"expected {expected}"
        )
    return scales.view(canonical_shape)


def make_mxfp8_tensor(
    qdata: torch.Tensor,
    scales: torch.Tensor,
    shape: tuple[int, ...],
    scale_layout: ScaleLayout = "row_major",
) -> MXTensor:
    """Wrap RTX-produced storage in TorchAO's canonical MX tensor subclass."""

    if qdata.device != scales.device:
        raise ValueError("MXFP8 qdata and scales must share one device")
    qdata = _logical_qdata_view(qdata, shape)
    storage_k = int(qdata.shape[-1])
    scales = _canonical_scale_view(scales, shape, storage_k, scale_layout)
    value = MXTensor.from_qdata_and_scales(
        qdata,
        scales,
        torch.bfloat16,
        block_size=32,
        kernel_preference=KernelPreference.AUTO,
        is_swizzled_scales=scale_layout != "row_major",
    )
    # TorchAO's bool identifies its standard blocked layout. The extra marker
    # distinguishes our independently padded 64-row schedule; storage size is
    # also sufficient to recover it after TorchAO device movement.
    value._rtx_scale_layout = scale_layout
    value._rtx_logical_shape = tuple(int(dim) for dim in shape)
    return value


def validate_mxfp8_tensor(value: MXTensor) -> None:
    if value.elem_dtype is not torch.float8_e4m3fn:
        raise TypeError("RTX MXFP8 requires TorchAO MXTensor with E4M3 qdata")
    if value.block_size != 32:
        raise ValueError(f"RTX MXFP8 requires block_size=32, got {value.block_size}")
    if value.orig_dtype is not torch.bfloat16:
        raise TypeError(
            f"RTX MXFP8 requires BF16 logical dtype, got {value.orig_dtype}"
        )
    if value.qdata.device != value.scale.device:
        raise ValueError("MXFP8 qdata and scales must share one device")
    rows, k = mxfp8_matrix_shape(value)
    storage_k = int(value.qdata.shape[-1])
    if storage_k < k or storage_k % 32:
        raise ValueError("RTX MXFP8 storage K must cover logical K in blocks of 32")
    if value.qdata.numel() != rows * storage_k:
        raise ValueError("TorchAO MXTensor qdata does not match its logical shape")
    if value.scale.dtype is not torch.float8_e8m0fnu:
        raise TypeError("RTX MXFP8 requires E8M0 scales")
    _ = mxfp8_scale_layout(value)


def mxfp8_orientation(value: MXTensor) -> Orientation:
    validate_rank = value.qdata.ndim >= 1
    if not validate_rank:
        raise ValueError("MXFP8 operand must have at least one dimension")
    if value.qdata.is_contiguous():
        return "row_major"
    if value.qdata.ndim == 2 and value.qdata.t().is_contiguous():
        return "transpose"
    raise ValueError("RTX kernels require row-major or logical-transpose qdata")


def mxfp8_matrix_shape(value: MXTensor) -> tuple[int, int]:
    logical_shape = getattr(value, "_rtx_logical_shape", None)
    return flattened_matrix_shape(
        tuple(int(dim) for dim in (logical_shape or value.shape))
    )


def mxfp8_scale_layout(value: MXTensor) -> ScaleLayout:
    rows, k = mxfp8_matrix_shape(value)
    storage_k = int(value.qdata.shape[-1])
    if not value.is_swizzled_scales:
        expected = rows * (storage_k // 32)
        if value.scale.numel() != expected:
            raise ValueError(
                f"row-major MXFP8 scales have {value.scale.numel()} values, "
                f"expected {expected}"
            )
        return "row_major"
    if storage_k != k or k % 128:
        raise ValueError("blocked MXFP8 scales require K divisible by 128")
    mma128_values = ceil(rows / 128) * (k // 128) * 512
    mma64_values = (
        (rows // 64) * (k // 128) * 512 if rows % 64 == 0 else -1
    )
    marker = getattr(value, "_rtx_scale_layout", None)
    if marker == "mma64":
        if value.scale.numel() != mma64_values:
            raise ValueError("RTX mma64 marker does not match scale storage")
        return "mma64"
    if marker == "mma128":
        if value.scale.numel() != mma128_values or rows % 128:
            raise ValueError("RTX mma128 marker does not match scale storage")
        return "mma128"
    if value.scale.numel() == mma64_values and rows % 128:
        return "mma64"
    if value.scale.numel() == mma128_values:
        return "mma128"
    if value.scale.numel() == mma64_values:
        return "mma64"
    raise ValueError(
        "blocked MXFP8 scale storage is neither TorchAO mma128 nor RTX mma64"
    )


def mxfp8_qdata_2d(value: MXTensor) -> torch.Tensor:
    validate_mxfp8_tensor(value)
    rows, _k = mxfp8_matrix_shape(value)
    storage_k = int(value.qdata.shape[-1])
    if mxfp8_orientation(value) != "row_major":
        raise ValueError("packed RTX linear operands must be row-major")
    return value.qdata.view(rows, storage_k)


def mxfp8_scales_for_kernel(value: MXTensor) -> torch.Tensor:
    validate_mxfp8_tensor(value)
    rows, k = mxfp8_matrix_shape(value)
    storage_k = int(value.qdata.shape[-1])
    layout = mxfp8_scale_layout(value)
    if layout == "row_major":
        return value.scale.view(rows, storage_k // 32)
    tile_rows = 64 if layout == "mma64" else 128
    if rows % tile_rows:
        raise ValueError(
            f"RTX {layout} kernel transport requires rows divisible by "
            f"{tile_rows}; TorchAO's padded storage alone is insufficient"
        )
    return value.scale.view(rows // tile_rows, k // 128, 512)


__all__ = [
    "MXFP8Tensor",
    "MXTensor",
    "make_mxfp8_tensor",
    "mxfp8_matrix_shape",
    "mxfp8_orientation",
    "mxfp8_qdata_2d",
    "mxfp8_scale_layout",
    "mxfp8_scales_for_kernel",
    "validate_mxfp8_tensor",
]
