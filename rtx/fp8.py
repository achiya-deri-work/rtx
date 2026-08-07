"""Torch frontend for fused MXFP8 linear layers on RTX Blackwell GPUs."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from dataclasses import asdict, dataclass, replace
from threading import RLock
from typing import TYPE_CHECKING, Literal
import torch
from torch import nn
from torch._library._out_variant import register_out_variant

from .kernels.mxfp8 import DEFAULT_MXFP8_FWD_CONFIG, MXFP8FwdConfig, MXFP8Problem
from .kernels.mxfp8_fwd import compile_mxfp8_fwd
from .kernels.mxfp8_gemm import MXFP8GemmConfig, compile_mxfp8_gemm
from .kernels.mxfp8_quant import (
    MXFP8QuantConfig,
    compile_mxfp8_dual_quant,
)

if TYPE_CHECKING:
    from .autotune import CoordinateDescentPolicy

AutotuneMode = Literal["off", "cache", "coordinate"]
MXFP8Backend = Literal["auto", "fused", "prequant"]


@dataclass(frozen=True, slots=True)
class MXFP8PrequantConfig:
    """Schedules for materialize-once quantization plus native-scale GEMM."""

    quant: MXFP8QuantConfig = MXFP8QuantConfig(
        load_bits=32,
        maxrregcount=96,
        persistent_waves=6,
        quant_amax="fp32",
        quant_math="fp32",
        scale_layout="mma128",
    )
    gemm: MXFP8GemmConfig = MXFP8GemmConfig(
        atom_layout_m=4,
        b_swizzle="128b",
        consumer_registers=232,
        producer_registers=64,
        scale_role="tma",
        scale_layout="mma128",
    )
    weight_scale_layout: str | None = None

    def rejection(self, problem: MXFP8Problem) -> str | None:
        x_rejection = self.quant.rejection(problem.m, problem.k)
        if x_rejection is not None:
            return f"activation quantizer: {x_rejection}"
        weight_config = replace(
            self.quant,
            scale_layout=self.weight_scale_layout or self.quant.scale_layout,
        )
        weight_rejection = weight_config.rejection(problem.n, problem.k)
        if weight_rejection is not None:
            return f"weight quantizer: {weight_rejection}"
        expected_layouts = {
            "row_major": ("row_major", "row_major"),
            "mma128": ("mma128", "mma128"),
            "mma64x128": ("mma64", "mma128"),
        }
        expected = expected_layouts.get(self.gemm.scale_layout)
        actual = (self.quant.scale_layout, weight_config.scale_layout)
        if expected is None or actual != expected:
            return (
                f"GEMM scale layout {self.gemm.scale_layout} requires "
                f"quantizer layouts {expected}, got {actual}"
            )
        return self.gemm.rejection(problem)


DEFAULT_MXFP8_PREQUANT_CONFIG = MXFP8PrequantConfig()


_CONFIGS: dict[str, MXFP8FwdConfig] = {}
_PREQUANT_CONFIGS: dict[str, MXFP8PrequantConfig] = {}
_PREQUANT_RUNNERS: dict[
    tuple[object, ...],
    tuple[object, object, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
] = {}
_INDUCTOR_PREQUANT_LAUNCHERS: dict[str, object] = {}
_INDUCTOR_PREQUANT_LAUNCHER_IDS: dict[str, str] = {}
_CONFIG_LOCK = RLock()


def _intern_config(config: MXFP8FwdConfig) -> str:
    key = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    with _CONFIG_LOCK:
        _CONFIGS[key] = config
    return key


def _intern_prequant_config(config: MXFP8PrequantConfig) -> str:
    key = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    with _CONFIG_LOCK:
        _PREQUANT_CONFIGS[key] = config
    return key


_DEFAULT_MXFP8_PREQUANT_KEY = _intern_prequant_config(
    DEFAULT_MXFP8_PREQUANT_CONFIG
)


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


def _allocate_scales(
    rows: int,
    k: int,
    scale_layout: str,
    device: torch.device,
) -> torch.Tensor:
    if scale_layout == "row_major":
        return torch.empty(
            rows, k // 32, dtype=torch.float8_e8m0fnu, device=device
        )
    tile_rows = 64 if scale_layout == "mma64" else 128
    return torch.empty(
        rows // tile_rows,
        k // 128,
        512,
        dtype=torch.float8_e8m0fnu,
        device=device,
    )


def _build_prequant_runner(
    x: torch.Tensor,
    weight: torch.Tensor,
    config_key: str,
):
    problem = MXFP8Problem(
        m=int(x.shape[0]), n=int(weight.shape[0]), k=int(x.shape[1])
    )
    try:
        config = _PREQUANT_CONFIGS[config_key]
    except KeyError as exc:
        raise RuntimeError("unknown MXFP8 prequant configuration key") from exc
    rejection = config.rejection(problem)
    if rejection is not None:
        raise RuntimeError(f"prequant MXFP8 cannot run this problem: {rejection}")
    major, _minor = torch.cuda.get_device_capability(x.device)
    if major != 12:
        raise RuntimeError(
            "native RTX MXFP8 kernel requires an SM120/SM121 GPU; "
            f"got compute capability {torch.cuda.get_device_capability(x.device)}"
        )
    qx = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    qw = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
    weight_scale_layout = (
        config.weight_scale_layout or config.quant.scale_layout
    )
    sx = _allocate_scales(
        x.shape[0], x.shape[1], config.quant.scale_layout, x.device
    )
    sw = _allocate_scales(
        weight.shape[0], weight.shape[1], weight_scale_layout, x.device
    )
    quant = compile_mxfp8_dual_quant(
        problem.m,
        problem.n,
        problem.k,
        config.quant,
        weight_scale_layout=config.weight_scale_layout,
    )
    gemm = compile_mxfp8_gemm(problem, config.gemm)
    return quant, gemm, qx, qw, sx, sw


def _launch_prequant_out(
    x: torch.Tensor,
    weight: torch.Tensor,
    config_key: str,
    out: torch.Tensor,
) -> None:
    x_c = x if x.is_contiguous() else x.contiguous()
    weight_c = weight if weight.is_contiguous() else weight.contiguous()
    stream = torch.cuda.current_stream(x.device)
    runner_key = (
        x.device.index,
        int(stream.cuda_stream),
        int(x_c.shape[0]),
        int(x_c.shape[1]),
        int(weight_c.shape[0]),
        config_key,
    )
    runner = _PREQUANT_RUNNERS.get(runner_key)
    if runner is None:
        with _CONFIG_LOCK:
            runner = _PREQUANT_RUNNERS.get(runner_key)
            if runner is None:
                runner = _build_prequant_runner(x_c, weight_c, config_key)
                _PREQUANT_RUNNERS[runner_key] = runner
    quant, gemm, qx, qw, sx, sw = runner
    quant(x_c, weight_c, qx, qw, sx, sw)
    gemm(qx, qw, sx, sw, out)


class _InductorPrequantLauncher:
    """Shape/config-bound hot path used by generated Inductor wrappers."""

    def __init__(self, config_key: str) -> None:
        self.config_key = config_key
        self.runners: dict[tuple[int | None, int, int, int, int], tuple] = {}

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        *,
        out: torch.Tensor,
    ) -> None:
        stream_id = int(torch._C._cuda_getCurrentRawStream(x.device.index))
        key = (
            x.device.index,
            stream_id,
            int(x.shape[0]),
            int(x.shape[1]),
            int(weight.shape[0]),
        )
        runner = self.runners.get(key)
        if runner is None:
            with _CONFIG_LOCK:
                runner = self.runners.get(key)
                if runner is None:
                    runner = _build_prequant_runner(
                        x, weight, self.config_key
                    )
                    self.runners[key] = runner
        quant, gemm, qx, qw, sx, sw = runner
        quant(x, weight, qx, qw, sx, sw)
        gemm(qx, qw, sx, sw, out)


def _inductor_prequant_launcher_name(config_key: str) -> str:
    with _CONFIG_LOCK:
        launcher_id = _INDUCTOR_PREQUANT_LAUNCHER_IDS.get(config_key)
        if launcher_id is None:
            launcher_id = f"launcher_{len(_INDUCTOR_PREQUANT_LAUNCHERS)}"
            _INDUCTOR_PREQUANT_LAUNCHER_IDS[config_key] = launcher_id
            _INDUCTOR_PREQUANT_LAUNCHERS[launcher_id] = (
                _InductorPrequantLauncher(config_key)
            )
    return f"torch._rtx_mxfp8_prequant_launchers[{launcher_id!r}]"


@torch.library.custom_op(
    "rtx::mxfp8_linear_prequant",
    mutates_args=(),
    device_types="cuda",
)
def _mxfp8_linear_prequant_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    config_key: str,
) -> torch.Tensor:
    out = torch.empty(
        (x.shape[0], weight.shape[0]), dtype=torch.bfloat16, device=x.device
    )
    _launch_prequant_out(x, weight, config_key, out)
    return out


@_mxfp8_linear_prequant_op.register_fake
def _mxfp8_linear_prequant_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    config_key: str,
) -> torch.Tensor:
    return torch.empty(
        (x.shape[0], weight.shape[0]), dtype=torch.bfloat16, device=x.device
    )


_PREQUANT_LIBRARY = torch.library.Library("rtx", "FRAGMENT")
_PREQUANT_LIBRARY.define(
    "mxfp8_linear_prequant_out("
    "Tensor x, Tensor weight, str config_key, Tensor(a!) out) -> ()"
)


@torch.library.impl("rtx::mxfp8_linear_prequant_out", "cuda")
def _mxfp8_linear_prequant_out_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    config_key: str,
    out: torch.Tensor,
) -> None:
    _launch_prequant_out(x, weight, config_key, out)


@torch.library.register_fake("rtx::mxfp8_linear_prequant_out")
def _mxfp8_linear_prequant_out_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    config_key: str,
    out: torch.Tensor,
) -> None:
    return None


register_out_variant(
    torch.ops.rtx.mxfp8_linear_prequant.default,
    torch.ops.rtx.mxfp8_linear_prequant_out.default,
)


def _register_prequant_inductor_lowering() -> None:
    from torch._inductor import ir
    from torch._inductor.lowering import register_lowering

    # Generated Python wrappers always import ``torch``.  Publishing this
    # narrow launcher there avoids a dispatcher round trip on every call.
    torch._rtx_mxfp8_prequant_launchers = _INDUCTOR_PREQUANT_LAUNCHERS

    @register_lowering(
        torch.ops.rtx.mxfp8_linear_prequant.default,
        type_promotion_kind=None,
    )
    def lower_prequant(x, weight, config_key):
        x_input = ir.ExternKernel.require_contiguous(
            ir.ExternKernel.realize_input(x)
        )
        weight_input = ir.ExternKernel.require_contiguous(
            ir.ExternKernel.realize_input(weight)
        )
        m = x.get_size()[0]
        n = weight.get_size()[0]
        result = ir.ExternKernelOut(
            layout=ir.FixedLayout(
                device=x.get_device(),
                dtype=torch.bfloat16,
                size=[m, n],
                stride=[n, 1],
            ),
            inputs=[x_input, weight_input],
            python_kernel_name=_inductor_prequant_launcher_name(config_key),
        )
        return ir.TensorBox.create(result)


_register_prequant_inductor_lowering()


def _run_prequant(
    x: torch.Tensor,
    weight: torch.Tensor,
    config_key: str,
) -> torch.Tensor:
    return _mxfp8_linear_prequant_op(x, weight, config_key)


def mxfp8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    config: MXFP8FwdConfig | None = None,
    autotune: AutotuneMode | bool | None = None,
    tuning_policy: "CoordinateDescentPolicy | None" = None,
    autotune_cache_dir: Path | str | None = None,
    backend: MXFP8Backend = "auto",
    prequant_config: MXFP8PrequantConfig | None = None,
) -> torch.Tensor:
    """Apply a no-bias linear transform with dynamic block-scaled MXFP8 operands.

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
    if backend not in ("auto", "fused", "prequant"):
        raise ValueError(
            f"backend must be auto, fused, or prequant; got {backend!r}"
        )
    problem = MXFP8Problem(x_2d.shape[0], weight.shape[0], x_2d.shape[1])
    selected_prequant = prequant_config or DEFAULT_MXFP8_PREQUANT_CONFIG
    use_prequant = backend == "prequant" or (
        backend == "auto"
        and config is None
        and autotune is not True
        and autotune != "coordinate"
        and selected_prequant.rejection(problem) is None
    )
    if use_prequant:
        rejection = selected_prequant.rejection(problem)
        if rejection is not None:
            raise RuntimeError(f"prequant MXFP8 backend is unavailable: {rejection}")
        key = (
            _DEFAULT_MXFP8_PREQUANT_KEY
            if prequant_config is None
            else _intern_prequant_config(selected_prequant)
        )
        out = _run_prequant(x_2d, weight, key)
        return out.reshape(*leading_shape, weight.shape[0])
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

    __constants__ = ["in_features", "out_features", "backend"]

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
        backend: MXFP8Backend = "auto",
        prequant_config: MXFP8PrequantConfig | None = None,
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
        self.backend = backend
        self.prequant_config = prequant_config
        self._prequant_config_key = _intern_prequant_config(
            prequant_config or DEFAULT_MXFP8_PREQUANT_CONFIG
        )
        self.weight = nn.Parameter(
            torch.empty(
                (out_features, in_features), device=device, dtype=torch.bfloat16
            )
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backend == "prequant":
            if x.ndim < 1 or x.shape[-1] != self.in_features:
                raise ValueError(
                    f"expected activation [..., {self.in_features}], got {x.shape}"
                )
            leading_shape = x.shape[:-1]
            x_2d = x.reshape(-1, self.in_features)
            _check_inputs(x_2d, self.weight)
            out = _run_prequant(
                x_2d,
                self.weight,
                self._prequant_config_key,
            )
            return out.reshape(*leading_shape, self.out_features)
        return mxfp8_linear(
            x,
            self.weight,
            config=self.config,
            autotune=self.autotune,
            tuning_policy=self.tuning_policy,
            autotune_cache_dir=self.autotune_cache_dir,
            backend=self.backend,
            prequant_config=self.prequant_config,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias=False, format=E4M3xE8M0, backend={self.backend}"
        )


__all__ = [
    "DEFAULT_MXFP8_PREQUANT_CONFIG",
    "MXFP8Linear",
    "MXFP8PrequantConfig",
    "mxfp8_linear",
]
