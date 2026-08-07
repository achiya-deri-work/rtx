"""Torch frontend for fused MXFP8 linear layers on RTX Blackwell GPUs."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from dataclasses import asdict
from threading import RLock
from typing import TYPE_CHECKING, Literal
import torch
from torch import nn

from .kernels.mxfp8 import DEFAULT_MXFP8_FWD_CONFIG, MXFP8FwdConfig, MXFP8Problem
from .kernels.mxfp8_fwd import compile_mxfp8_fwd

if TYPE_CHECKING:
    from .autotune import CoordinateDescentPolicy

AutotuneMode = Literal["off", "cache", "coordinate"]


_CONFIGS: dict[str, MXFP8FwdConfig] = {}
_CONFIG_LOCK = RLock()


def _intern_config(config: MXFP8FwdConfig) -> str:
    key = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    with _CONFIG_LOCK:
        _CONFIGS[key] = config
    return key


def _check_inputs(x: torch.Tensor, weight: torch.Tensor) -> None:
    if x.device.type != "cuda" or weight.device.type != "cuda":
        raise ValueError("MXFP8Linear only accepts CUDA tensors")
    if x.dtype is not torch.bfloat16 or weight.dtype is not torch.bfloat16:
        raise TypeError(
            "MXFP8Linear quantizes BF16 activations and weights in-kernel; "
            f"got x={x.dtype}, weight={weight.dtype}"
        )
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError(
            f"internal MXFP8 op expects 2D tensors, got {x.shape=} and {weight.shape=}"
        )
    if x.shape[1] != weight.shape[1]:
        raise ValueError(
            f"in_features mismatch: activation K={x.shape[1]}, weight K={weight.shape[1]}"
        )
    if x.shape[1] % 32:
        raise ValueError(f"in_features must be divisible by 32, got {x.shape[1]}")


@torch.library.custom_op(
    "rtx::mxfp8_linear_fwd",
    mutates_args=(),
    device_types="cuda",
)
def _mxfp8_linear_fwd_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    config_key: str,
) -> torch.Tensor:
    _check_inputs(x, weight)
    x_c = x if x.is_contiguous() else x.contiguous()
    weight_c = weight if weight.is_contiguous() else weight.contiguous()
    problem = MXFP8Problem(
        m=int(x_c.shape[0]),
        n=int(weight_c.shape[0]),
        k=int(x_c.shape[1]),
    )
    try:
        config = _CONFIGS[config_key]
    except KeyError as exc:
        raise RuntimeError("unknown MXFP8 configuration key") from exc
    rejection = config.implementation_rejection(problem)
    if rejection is not None:
        raise RuntimeError(f"MXFP8 configuration cannot run this problem: {rejection}")

    major, _minor = torch.cuda.get_device_capability(x.device)
    if major != 12:
        raise RuntimeError(
            "native RTX MXFP8 kernel requires an SM120/SM121 GPU; "
            f"got compute capability {torch.cuda.get_device_capability(x.device)}"
        )

    out = torch.empty(
        (problem.m, problem.n), dtype=torch.bfloat16, device=x.device
    )
    launcher = compile_mxfp8_fwd(problem, config)
    launcher(x_c, weight_c, out)
    # TVM-FFI launches asynchronously.  Keep all inputs alive until the result
    # tensor is released, matching the existing project kernel convention.
    out._base_inputs = (x_c, weight_c)
    return out


@_mxfp8_linear_fwd_op.register_fake
def _mxfp8_linear_fwd_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    config_key: str,
) -> torch.Tensor:
    return torch.empty(
        (x.shape[0], weight.shape[0]), dtype=torch.bfloat16, device=x.device
    )


def mxfp8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    config: MXFP8FwdConfig | None = None,
    autotune: AutotuneMode | bool | None = None,
    tuning_policy: "CoordinateDescentPolicy | None" = None,
    autotune_cache_dir: Path | str | None = None,
) -> torch.Tensor:
    """Apply a no-bias linear transform with fused row-wise MXFP8 operands.

    Leading activation dimensions are flattened for the kernel and restored in
    the BF16 result.  ``weight`` has the normal PyTorch ``[out, in]`` layout.
    """

    if x.ndim < 1:
        raise ValueError("activation must have at least one dimension")
    if weight.ndim != 2:
        raise ValueError(f"weight must be [out_features, in_features], got {weight.shape}")
    if x.shape[-1] != weight.shape[1]:
        raise ValueError(
            f"in_features mismatch: activation K={x.shape[-1]}, weight K={weight.shape[1]}"
        )
    leading_shape = x.shape[:-1]
    x_2d = x.reshape(-1, x.shape[-1])
    _check_inputs(x_2d, weight)
    selected_config = _resolve_fwd_config(
        x_2d,
        weight,
        config=config,
        autotune=autotune,
        tuning_policy=tuning_policy,
        cache_dir=autotune_cache_dir,
    )
    key = _intern_config(selected_config)
    out = _mxfp8_linear_fwd_op(x_2d, weight, key)
    return out.reshape(*leading_shape, weight.shape[0])


def _autotune_mode(value: AutotuneMode | bool | None) -> AutotuneMode:
    if isinstance(value, bool):
        return "coordinate" if value else "off"
    selected = os.getenv("RTX_MXFP8_AUTOTUNE", "cache") if value is None else value
    if selected not in ("off", "cache", "coordinate"):
        raise ValueError(
            "autotune must be off, cache, or coordinate; "
            f"got {selected!r}"
        )
    return selected


def _resolve_fwd_config(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    config: MXFP8FwdConfig | None,
    autotune: AutotuneMode | bool | None,
    tuning_policy: "CoordinateDescentPolicy | None",
    cache_dir: Path | str | None,
) -> MXFP8FwdConfig:
    if config is not None:
        return config
    mode = _autotune_mode(autotune)
    if mode == "off":
        return DEFAULT_MXFP8_FWD_CONFIG
    # Tuning launches and synchronizes kernels; never start it from a compiler
    # trace. Cache-only selection is also skipped to keep graph capture pure.
    if torch.compiler.is_compiling():
        return DEFAULT_MXFP8_FWD_CONFIG

    from .autotune import (
        CoordinateDescentPolicy,
        load_cached_mxfp8_fwd_config,
        tune_mxfp8_fwd,
    )

    problem = MXFP8Problem(x.shape[0], weight.shape[0], x.shape[1])
    cached = load_cached_mxfp8_fwd_config(
        problem, device=x.device, cache_dir=cache_dir
    )
    if cached is not None or mode == "cache":
        return cached or DEFAULT_MXFP8_FWD_CONFIG

    policy = tuning_policy
    if policy is None:
        policy = CoordinateDescentPolicy(
            time_budget_s=float(os.getenv("RTX_MXFP8_AUTOTUNE_SECONDS", "1800")),
            max_passes=int(os.getenv("RTX_MXFP8_AUTOTUNE_PASSES", "4")),
        )
    return tune_mxfp8_fwd(
        x, weight, policy=policy, cache_dir=cache_dir
    ).config


class MXFP8Linear(nn.Module):
    """No-bias BF16 linear module whose two operands are MXFP8 in the kernel."""

    __constants__ = ["in_features", "out_features"]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        config: MXFP8FwdConfig | None = None,
        autotune: AutotuneMode | bool | None = None,
        tuning_policy: "CoordinateDescentPolicy | None" = None,
        autotune_cache_dir: Path | str | None = None,
    ) -> None:
        super().__init__()
        if in_features % 32:
            raise ValueError(
                f"MXFP8 in_features must be divisible by scale-vector size 32, "
                f"got {in_features}"
            )
        if dtype is not torch.bfloat16:
            raise TypeError(f"MXFP8Linear parameters must be BF16, got {dtype}")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.config = config
        self.autotune = autotune
        self.tuning_policy = tuning_policy
        self.autotune_cache_dir = autotune_cache_dir
        self.weight = nn.Parameter(
            torch.empty(
                (out_features, in_features), device=device, dtype=torch.bfloat16
            )
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return mxfp8_linear(
            x,
            self.weight,
            config=self.config,
            autotune=self.autotune,
            tuning_policy=self.tuning_policy,
            autotune_cache_dir=self.autotune_cache_dir,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            "bias=False, format=E4M3xE8M0"
        )


__all__ = ["MXFP8Linear", "mxfp8_linear"]
