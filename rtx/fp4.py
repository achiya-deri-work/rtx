"""Torch frontend contract for NVFP4-forward, MXFP8-backward linear layers.

The NVFP4 forward kernel is deliberately not implemented here.  Keeping its
dispatcher and module boundary independent lets that kernel evolve without
forking the already registered and autotunable MXFP8 backward implementation.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import torch
from torch import nn

from .formats import NVFP4Tensor
from .formats.common import SCALE_LAYOUT_CODES, reject_packed_dtype_conversion

if TYPE_CHECKING:
    from .kernels.mxfp8_bwd import MXFP8BwdConfig

WeightMode = Literal["dynamic", "prequantized"]


def _check_nvfp4_inputs(x: torch.Tensor, weight: torch.Tensor) -> None:
    if x.device.type != "cuda" or weight.device.type != "cuda":
        raise ValueError("NVFP4Linear only accepts CUDA tensors")
    if x.dtype is not torch.bfloat16 or weight.dtype is not torch.bfloat16:
        raise TypeError(
            "NVFP4Linear quantizes BF16 activations and weights in-kernel; "
            f"got x={x.dtype}, weight={weight.dtype}"
        )
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError(
            f"internal NVFP4 op expects 2D tensors, got {x.shape=} and "
            f"{weight.shape=}"
        )
    if x.shape[1] != weight.shape[1]:
        raise ValueError(
            f"in_features mismatch: activation K={x.shape[1]}, "
            f"weight K={weight.shape[1]}"
        )


def _launch_nvfp4_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    forward_config_key: str,
) -> torch.Tensor:
    _check_nvfp4_inputs(x, weight)
    raise NotImplementedError(
        "NVFP4 forward is registered but its RTX Blackwell kernel has not "
        "been implemented yet"
    )


def quantize_nvfp4(tensor: torch.Tensor) -> NVFP4Tensor:
    """Prequantize one NVFP4 operand when the RTX kernel becomes available."""

    if tensor.ndim < 1 or tensor.dtype is not torch.bfloat16:
        raise TypeError("NVFP4 quantization requires a BF16 tensor")
    if tensor.device.type != "cuda":
        raise ValueError("NVFP4 quantization requires a CUDA tensor")
    raise NotImplementedError(
        "the NVFP4 packed-operand contract is available, but its RTX "
        "quantization kernel has not been implemented yet"
    )


@torch.library.custom_op(
    "rtx::nvfp4_linear_fwd",
    mutates_args=(),
    device_types="cuda",
)
def _nvfp4_linear_fwd_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    forward_config_key: str,
) -> torch.Tensor:
    return _launch_nvfp4_forward(x, weight, forward_config_key)


@_nvfp4_linear_fwd_op.register_fake
def _nvfp4_linear_fwd_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    forward_config_key: str,
) -> torch.Tensor:
    return torch.empty(
        (x.shape[0], weight.shape[0]), dtype=torch.bfloat16, device=x.device
    )


@torch.library.custom_op(
    "rtx::nvfp4_linear_dynamic_x_prequant_w",
    mutates_args=(),
    device_types="cuda",
)
def _nvfp4_linear_dynamic_x_prequant_w_op(
    x: torch.Tensor,
    weight_data: torch.Tensor,
    weight_block_scales: torch.Tensor,
    weight_tensor_scale: torch.Tensor,
    n: int,
    k: int,
    weight_scale_layout: str,
    forward_config_key: str,
) -> torch.Tensor:
    raise NotImplementedError(
        "NVFP4 dynamic-X/prequant-W execution awaits the RTX kernel"
    )


@_nvfp4_linear_dynamic_x_prequant_w_op.register_fake
def _nvfp4_linear_dynamic_x_prequant_w_fake(
    x: torch.Tensor,
    weight_data: torch.Tensor,
    weight_block_scales: torch.Tensor,
    weight_tensor_scale: torch.Tensor,
    n: int,
    k: int,
    weight_scale_layout: str,
    forward_config_key: str,
) -> torch.Tensor:
    return torch.empty((x.shape[0], n), dtype=torch.bfloat16, device=x.device)


@torch.library.custom_op(
    "rtx::nvfp4_linear_prequantized",
    mutates_args=(),
    device_types="cuda",
)
def _nvfp4_linear_prequantized_op(
    x_data: torch.Tensor,
    weight_data: torch.Tensor,
    x_block_scales: torch.Tensor,
    weight_block_scales: torch.Tensor,
    x_tensor_scale: torch.Tensor,
    weight_tensor_scale: torch.Tensor,
    m: int,
    n: int,
    k: int,
    x_scale_layout: str,
    weight_scale_layout: str,
    forward_config_key: str,
) -> torch.Tensor:
    raise NotImplementedError(
        "NVFP4 prequantized GEMM execution awaits the RTX kernel"
    )


@_nvfp4_linear_prequantized_op.register_fake
def _nvfp4_linear_prequantized_fake(
    x_data: torch.Tensor,
    weight_data: torch.Tensor,
    x_block_scales: torch.Tensor,
    weight_block_scales: torch.Tensor,
    x_tensor_scale: torch.Tensor,
    weight_tensor_scale: torch.Tensor,
    m: int,
    n: int,
    k: int,
    x_scale_layout: str,
    weight_scale_layout: str,
    forward_config_key: str,
) -> torch.Tensor:
    return torch.empty((m, n), dtype=torch.bfloat16, device=x_data.device)


@torch.library.custom_op(
    "rtx::nvfp4_linear_train",
    mutates_args=(),
    device_types="cuda",
)
def _nvfp4_linear_train_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    forward_config_key: str,
    backward_config_key: str,
) -> torch.Tensor:
    return _launch_nvfp4_forward(x, weight, forward_config_key)


@_nvfp4_linear_train_op.register_fake
def _nvfp4_linear_train_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    forward_config_key: str,
    backward_config_key: str,
) -> torch.Tensor:
    return torch.empty(
        (x.shape[0], weight.shape[0]), dtype=torch.bfloat16, device=x.device
    )


def _setup_nvfp4_context(ctx, inputs, output) -> None:
    x, weight, _forward_config_key, backward_config_key = inputs
    ctx.save_for_backward(x, weight)
    ctx.backward_config_key = backward_config_key


def _nvfp4_backward(ctx, grad_output: torch.Tensor):
    from .fp8_bwd import (
        _mxfp8_linear_bwd_op,
        _mxfp8_linear_dw_op,
        _mxfp8_linear_dx_op,
    )

    x, weight = ctx.saved_tensors
    need_x, need_weight = ctx.needs_input_grad[:2]
    grad_x = grad_weight = None
    if need_x and need_weight:
        grad_x, grad_weight = _mxfp8_linear_bwd_op(
            grad_output, x, weight, ctx.backward_config_key
        )
    elif need_x:
        grad_x = _mxfp8_linear_dx_op(
            grad_output, x, weight, ctx.backward_config_key
        )
    elif need_weight:
        grad_weight = _mxfp8_linear_dw_op(
            grad_output, x, weight, ctx.backward_config_key
        )
    return grad_x, grad_weight, None, None


torch.library.register_autograd(
    "rtx::nvfp4_linear_train",
    _nvfp4_backward,
    setup_context=_setup_nvfp4_context,
)


def nvfp4_linear(
    x: torch.Tensor | NVFP4Tensor,
    weight: torch.Tensor | NVFP4Tensor,
    *,
    backward_config: "MXFP8BwdConfig | None" = None,
) -> torch.Tensor:
    """Apply NVFP4 forward and MXFP8 backward to BF16 operands.

    The public contract is available now, but execution intentionally raises
    until the separate NVFP4 forward kernel is implemented.
    """

    if isinstance(weight, NVFP4Tensor):
        n, k = weight.matrix_shape
        if isinstance(x, NVFP4Tensor):
            m, x_k = x.matrix_shape
            if x_k != k or x.device != weight.device:
                raise ValueError("packed NVFP4 X/W shape or device mismatch")
            out = _nvfp4_linear_prequantized_op(
                x.data,
                weight.data,
                x.block_scales,
                weight.block_scales,
                x.tensor_scale,
                weight.tensor_scale,
                m,
                n,
                k,
                x.scale_layout,
                weight.scale_layout,
                "unimplemented",
            )
            return out.reshape(*x.shape[:-1], n)
        if x.ndim < 1 or x.shape[-1] != k:
            raise ValueError(f"expected activation [..., {k}], got {x.shape}")
        if x.device.type != "cuda" or weight.device.type != "cuda":
            raise ValueError("dynamic-X/prequant-W NVFP4 execution requires CUDA")
        if x.device != weight.device:
            raise ValueError("dynamic X and packed W must be on one CUDA device")
        if x.dtype is not torch.bfloat16:
            raise TypeError(f"dynamic NVFP4 activation must be BF16, got {x.dtype}")
        if torch.is_grad_enabled() and x.requires_grad:
            raise RuntimeError("prequantized NVFP4 weights are inference-only")
        leading = x.shape[:-1]
        x_2d = x.reshape(-1, k)
        out = _nvfp4_linear_dynamic_x_prequant_w_op(
            x_2d,
            weight.data,
            weight.block_scales,
            weight.tensor_scale,
            n,
            k,
            weight.scale_layout,
            "unimplemented",
        )
        return out.reshape(*leading, n)
    if isinstance(x, NVFP4Tensor):
        raise TypeError("a prequantized NVFP4 activation requires a prequantized weight")
    if x.ndim < 1:
        raise ValueError("activation must have at least one dimension")
    if weight.ndim != 2:
        raise ValueError(f"weight must be [out_features, in_features], got {weight.shape}")
    if x.shape[-1] != weight.shape[1]:
        raise ValueError(
            f"in_features mismatch: activation K={x.shape[-1]}, "
            f"weight K={weight.shape[1]}"
        )
    leading_shape = x.shape[:-1]
    x_2d = x.reshape(-1, x.shape[-1])
    _check_nvfp4_inputs(x_2d, weight)
    forward_key = "unimplemented"
    if torch.is_grad_enabled() and (x_2d.requires_grad or weight.requires_grad):
        from .fp8_bwd import DEFAULT_MXFP8_BWD_CONFIG, _intern_bwd_config

        backward_key = _intern_bwd_config(
            backward_config or DEFAULT_MXFP8_BWD_CONFIG
        )
        out = _nvfp4_linear_train_op(
            x_2d, weight, forward_key, backward_key
        )
    else:
        out = _nvfp4_linear_fwd_op(x_2d, weight, forward_key)
    return out.reshape(*leading_shape, weight.shape[0])


class NVFP4Linear(nn.Module):
    """No-bias BF16 linear with NVFP4 forward and MXFP8 backward."""

    __constants__ = ["in_features", "out_features", "weight_mode"]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        backward_config: "MXFP8BwdConfig | None" = None,
        packed_weight: NVFP4Tensor | None = None,
    ) -> None:
        super().__init__()
        if bias:
            raise NotImplementedError(
                "NVFP4Linear is a no-bias linear layer; pass bias=False"
            )
        if dtype is not torch.bfloat16:
            raise TypeError(f"NVFP4Linear parameters must be BF16, got {dtype}")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.backward_config = backward_config
        self.weight_mode: WeightMode = (
            "prequantized" if packed_weight is not None else "dynamic"
        )
        if packed_weight is None:
            self.weight = nn.Parameter(
                torch.empty(
                    (out_features, in_features), device=device, dtype=torch.bfloat16
                )
            )
            self.reset_parameters()
        else:
            if packed_weight.shape != (out_features, in_features):
                raise ValueError(
                    f"packed NVFP4 weight must have shape {(out_features, in_features)}"
                )
            if packed_weight.orientation != "row_major":
                raise ValueError("packed linear weights must be row-major")
            if device is not None:
                packed_weight = packed_weight.to(device)
            self.register_parameter("weight", None)
            self.register_buffer("weight_data", packed_weight.data)
            self.register_buffer("weight_block_scales", packed_weight.block_scales)
            self.register_buffer("weight_tensor_scale", packed_weight.tensor_scale)
            self.register_buffer(
                "weight_packing_meta",
                torch.tensor(
                    [
                        packed_weight.schema_version,
                        SCALE_LAYOUT_CODES[packed_weight.scale_layout],
                    ],
                    dtype=torch.int64,
                    device=packed_weight.device,
                ),
            )
            self._weight_scale_layout = packed_weight.scale_layout
            self._weight_packing_schema = packed_weight.schema_version
            self.training = False

    @property
    def bias(self) -> None:
        return None

    def to(self, *args, **kwargs):
        if self.weight_mode == "prequantized":
            reject_packed_dtype_conversion(args, kwargs, format_name="NVFP4")
        return super().to(*args, **kwargs)

    def train(self, mode: bool = True):
        if self.weight_mode == "prequantized" and mode:
            raise RuntimeError(
                "a prequantized NVFP4Linear is inference-only; keep a dynamic "
                "BF16-master module for training"
            )
        return super().train(mode)

    def half(self):
        if self.weight_mode == "prequantized":
            raise TypeError("a prequantized NVFP4 module cannot be dtype-cast")
        return super().half()

    def float(self):
        if self.weight_mode == "prequantized":
            raise TypeError("a prequantized NVFP4 module cannot be dtype-cast")
        return super().float()

    def bfloat16(self):
        if self.weight_mode == "prequantized":
            raise TypeError("a prequantized NVFP4 module cannot be dtype-cast")
        return super().bfloat16()

    def type(self, dst_type=None):
        if self.weight_mode == "prequantized" and dst_type is not None:
            raise TypeError("a prequantized NVFP4 module cannot be dtype-cast")
        return super().type(dst_type)

    def reset_parameters(self) -> None:
        if self.weight is not None:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    @property
    def packed_weight(self) -> NVFP4Tensor | None:
        if self.weight_mode != "prequantized":
            return None
        return NVFP4Tensor(
            self.weight_data,
            self.weight_block_scales,
            self.weight_tensor_scale,
            (self.out_features, self.in_features),
            self._weight_scale_layout,
            schema_version=self._weight_packing_schema,
        )

    @classmethod
    def from_float(cls, module: nn.Linear) -> "NVFP4Linear":
        if module.bias is not None:
            raise NotImplementedError("NVFP4Linear.from_float requires bias=False")
        packed = quantize_nvfp4(module.weight.detach())
        return cls(
            module.in_features,
            module.out_features,
            bias=False,
            device=module.weight.device,
            packed_weight=packed,
        )

    def to_quantized_weight(self) -> "NVFP4Linear":
        if self.weight_mode == "prequantized":
            return self
        assert self.weight is not None
        packed = quantize_nvfp4(self.weight.detach())
        return type(self)(
            self.in_features,
            self.out_features,
            bias=False,
            device=self.weight.device,
            backward_config=self.backward_config,
            packed_weight=packed,
        )

    def forward(self, x: torch.Tensor | NVFP4Tensor) -> torch.Tensor:
        if self.weight_mode == "prequantized":
            packed_weight = self.packed_weight
            assert packed_weight is not None
            return nvfp4_linear(x, packed_weight)
        if isinstance(x, NVFP4Tensor):
            raise TypeError(
                "a prequantized activation requires a prequantized module weight"
            )
        assert self.weight is not None
        return nvfp4_linear(
            x,
            self.weight,
            backward_config=self.backward_config,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias=False, forward=NVFP4, backward=MXFP8, weight_mode={self.weight_mode}"
        )


__all__ = ["NVFP4Linear", "nvfp4_linear", "quantize_nvfp4"]
