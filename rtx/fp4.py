"""Torch frontend contract for NVFP4-forward, MXFP8-backward linear layers.

The NVFP4 forward kernel is deliberately not implemented here.  Keeping its
dispatcher and module boundary independent lets that kernel evolve without
forking the already registered and autotunable MXFP8 backward implementation.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from .kernels.mxfp8_bwd import MXFP8BwdConfig


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
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    backward_config: "MXFP8BwdConfig | None" = None,
) -> torch.Tensor:
    """Apply NVFP4 forward and MXFP8 backward to BF16 operands.

    The public contract is available now, but execution intentionally raises
    until the separate NVFP4 forward kernel is implemented.
    """

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

    __constants__ = ["in_features", "out_features"]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        backward_config: "MXFP8BwdConfig | None" = None,
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
        self.weight = nn.Parameter(
            torch.empty(
                (out_features, in_features), device=device, dtype=torch.bfloat16
            )
        )
        self.reset_parameters()

    @property
    def bias(self) -> None:
        return None

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nvfp4_linear(
            x,
            self.weight,
            backward_config=self.backward_config,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            "bias=False, forward=NVFP4, backward=MXFP8"
        )


__all__ = ["NVFP4Linear", "nvfp4_linear"]
