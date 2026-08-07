"""Torch frontend for fused MXFP8 linear layers on RTX Blackwell GPUs."""

from __future__ import annotations

import ctypes
import hashlib
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
    compile_mxfp8_quant,
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
    # A dual launch has less launch overhead, while independent launches can
    # use different schedules for M and N and may overlap better on a stream.
    quant_launches: str = "dual"
    weight_quant: MXFP8QuantConfig | None = None
    weight_scale_layout: str | None = None
    # cudaLimitMaxL2FetchGranularity is process-global. It is represented so
    # experiments and selected winners are reproducible; the tuner restores
    # the previous value between candidates.
    l2_fetch_granularity: int | None = None

    def resolved_weight_quant(self) -> MXFP8QuantConfig:
        layout = self.weight_scale_layout or self.quant.scale_layout
        if self.weight_quant is None:
            return replace(self.quant, scale_layout=layout)
        if self.weight_scale_layout is not None and (
            self.weight_quant.scale_layout != self.weight_scale_layout
        ):
            return replace(self.weight_quant, scale_layout=self.weight_scale_layout)
        return self.weight_quant

    def normalized(self) -> "MXFP8PrequantConfig":
        """Collapse equivalent inherited/explicit W schedules to one key."""

        weight = self.resolved_weight_quant()
        if replace(weight, scale_layout=self.quant.scale_layout) == self.quant:
            weight_layout = (
                None
                if weight.scale_layout == self.quant.scale_layout
                else weight.scale_layout
            )
            return replace(
                self,
                weight_quant=None,
                weight_scale_layout=weight_layout,
            )
        return replace(self, weight_quant=weight, weight_scale_layout=None)

    def rejection(self, problem: MXFP8Problem) -> str | None:
        x_rejection = self.quant.rejection(problem.m, problem.k)
        if x_rejection is not None:
            return f"activation quantizer: {x_rejection}"
        if self.quant_launches not in ("dual", "separate"):
            return "quant_launches must be dual or separate"
        if self.l2_fetch_granularity not in (None, 0, 32, 64, 128):
            return "L2 fetch granularity must be None, 0, 32, 64, or 128"
        weight_config = self.resolved_weight_quant()
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
        if self.quant_launches == "dual":
            # The combined kernel shares one instruction/launch schedule and
            # permits only the physical scale layout to differ per operand.
            if replace(weight_config, scale_layout=self.quant.scale_layout) != self.quant:
                return (
                    "dual quantization requires identical X/W schedules except "
                    "for scale_layout"
                )
        return self.gemm.rejection(problem)


DEFAULT_MXFP8_PREQUANT_CONFIG = MXFP8PrequantConfig()


_CONFIGS: dict[str, MXFP8FwdConfig] = {}
_PREQUANT_CONFIGS: dict[str, MXFP8PrequantConfig] = {}
_PREQUANT_AUTOTUNE_REQUESTS: dict[str, "_PrequantAutotuneRequest"] = {}
_PREQUANT_AUTOTUNE_SELECTIONS: dict[tuple[object, ...], str] = {}
_PREQUANT_RUNNERS: dict[
    tuple[object, ...],
    "_PrequantRunner",
] = {}


class _InductorPrequantLauncherRegistry(dict[str, object]):
    def __missing__(self, config_key: str) -> object:
        with _CONFIG_LOCK:
            launcher = self.get(config_key)
            if launcher is None:
                launcher = _InductorPrequantLauncher(config_key)
                self[config_key] = launcher
        return launcher


_INDUCTOR_PREQUANT_LAUNCHERS = _InductorPrequantLauncherRegistry()
_CONFIG_LOCK = RLock()
_CUDA_RUNTIME: object | None = None
_CURRENT_L2_FETCH_GRANULARITY: int | None = None


def _set_l2_fetch_granularity(value: int) -> int:
    """Set CUDA's process-global L2 fetch limit and return its old value."""

    global _CUDA_RUNTIME, _CURRENT_L2_FETCH_GRANULARITY
    if _CUDA_RUNTIME is None:
        site_packages = Path(torch.__file__).resolve().parent.parent
        candidates = tuple((site_packages / "nvidia").glob("cu*/lib/libcudart.so.*"))
        if not candidates:
            raise RuntimeError("could not locate libcudart for the L2 fetch limit")
        runtime = ctypes.CDLL(str(candidates[0]))
        runtime.cudaDeviceGetLimit.argtypes = [
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_int,
        ]
        runtime.cudaDeviceSetLimit.argtypes = [ctypes.c_int, ctypes.c_size_t]
        _CUDA_RUNTIME = runtime
    runtime = _CUDA_RUNTIME
    previous = ctypes.c_size_t()
    if runtime.cudaDeviceGetLimit(ctypes.byref(previous), 5) != 0:
        raise RuntimeError("cudaDeviceGetLimit(MaxL2FetchGranularity) failed")
    if runtime.cudaDeviceSetLimit(5, value) != 0:
        raise RuntimeError("cudaDeviceSetLimit(MaxL2FetchGranularity) failed")
    _CURRENT_L2_FETCH_GRANULARITY = value
    return int(previous.value)


def _ensure_l2_fetch_granularity(value: int | None) -> None:
    if value is not None and _CURRENT_L2_FETCH_GRANULARITY != value:
        _set_l2_fetch_granularity(value)


@dataclass(slots=True)
class _PrequantRunner:
    quant_launches: str
    quant_x: object
    quant_w: object | None
    gemm: object
    qx: torch.Tensor
    qw: torch.Tensor
    sx: torch.Tensor
    sw: torch.Tensor
    l2_fetch_granularity: int | None

    def __call__(self, x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor) -> None:
        _ensure_l2_fetch_granularity(self.l2_fetch_granularity)
        if self.quant_launches == "dual":
            self.quant_x(x, weight, self.qx, self.qw, self.sx, self.sw)
        else:
            self.quant_x(x, self.qx, self.sx)
            assert self.quant_w is not None
            self.quant_w(weight, self.qw, self.sw)
        self.gemm(self.qx, self.qw, self.sx, self.sw, out)


@dataclass(frozen=True, slots=True)
class _PrequantAutotuneRequest:
    mode: AutotuneMode
    policy: object | None
    cache_dir: str | None
    initial: MXFP8PrequantConfig


def _intern_config(config: MXFP8FwdConfig) -> str:
    key = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    with _CONFIG_LOCK:
        _CONFIGS[key] = config
    return key


@torch.compiler.assume_constant_result
def _intern_prequant_config(config: MXFP8PrequantConfig) -> str:
    config = config.normalized()
    key = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    with _CONFIG_LOCK:
        _PREQUANT_CONFIGS[key] = config
    return key


@torch.compiler.assume_constant_result
def _intern_prequant_autotune_request(
    mode: AutotuneMode,
    policy: object | None,
    cache_dir: Path | str | None,
    initial: MXFP8PrequantConfig,
) -> str:
    policy_value = None if policy is None else asdict(policy)
    payload = {
        "mode": mode,
        "policy": policy_value,
        "cache_dir": None if cache_dir is None else str(Path(cache_dir).expanduser()),
        "initial": asdict(initial),
    }
    digest = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    key = "autotune:" + hashlib.sha256(digest.encode()).hexdigest()[:24]
    with _CONFIG_LOCK:
        _PREQUANT_AUTOTUNE_REQUESTS[key] = _PrequantAutotuneRequest(
            mode=mode,
            policy=policy,
            cache_dir=payload["cache_dir"],
            initial=initial,
        )
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
    request = _PREQUANT_AUTOTUNE_REQUESTS.get(config_key)
    if request is not None:
        selection_key = (
            config_key,
            x.device.index,
            problem.m,
            problem.n,
            problem.k,
        )
        selected_key = _PREQUANT_AUTOTUNE_SELECTIONS.get(selection_key)
        if selected_key is None:
            from .prequant_autotune import (
                load_cached_mxfp8_prequant_config,
                tune_mxfp8_prequant,
            )

            if request.mode == "coordinate":
                result = tune_mxfp8_prequant(
                    x,
                    weight,
                    policy=request.policy,
                    initial=request.initial,
                    cache_dir=request.cache_dir,
                    progress=print,
                )
                selected = result.config
            else:
                selected = load_cached_mxfp8_prequant_config(
                    problem,
                    device=x.device,
                    cache_dir=request.cache_dir,
                )
            selected_key = _intern_prequant_config(selected or request.initial)
            _PREQUANT_AUTOTUNE_SELECTIONS[selection_key] = selected_key
        return _build_prequant_runner(x, weight, selected_key)
    try:
        config = _PREQUANT_CONFIGS[config_key]
    except KeyError as exc:
        raise RuntimeError("unknown MXFP8 prequant configuration key") from exc
    rejection = config.rejection(problem)
    if rejection is not None:
        raise RuntimeError(f"prequant MXFP8 cannot run this problem: {rejection}")
    if config.l2_fetch_granularity is not None:
        _set_l2_fetch_granularity(config.l2_fetch_granularity)
    major, _minor = torch.cuda.get_device_capability(x.device)
    if major != 12:
        raise RuntimeError(
            "native RTX MXFP8 kernel requires an SM120/SM121 GPU; "
            f"got compute capability {torch.cuda.get_device_capability(x.device)}"
        )
    qx = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    qw = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
    weight_config = config.resolved_weight_quant()
    weight_scale_layout = weight_config.scale_layout
    sx = _allocate_scales(
        x.shape[0], x.shape[1], config.quant.scale_layout, x.device
    )
    sw = _allocate_scales(
        weight.shape[0], weight.shape[1], weight_scale_layout, x.device
    )
    if config.quant_launches == "dual":
        quant_x = compile_mxfp8_dual_quant(
            problem.m,
            problem.n,
            problem.k,
            config.quant,
            weight_scale_layout=weight_scale_layout,
        )
        quant_w = None
    else:
        quant_x = compile_mxfp8_quant(problem.m, problem.k, config.quant)
        quant_w = compile_mxfp8_quant(problem.n, problem.k, weight_config)
    gemm = compile_mxfp8_gemm(problem, config.gemm)
    return _PrequantRunner(
        config.quant_launches,
        quant_x,
        quant_w,
        gemm,
        qx,
        qw,
        sx,
        sw,
        config.l2_fetch_granularity,
    )


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
    runner(x_c, weight_c, out)


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
        runner(x, weight, out)


def _inductor_prequant_launcher_name(config_key: str) -> str:
    # Generated wrappers may be loaded from Inductor's on-disk cache without
    # executing this lowering again. The registry's __missing__ constructs the
    # shape/config launcher lazily in that process.
    return f"torch._rtx_mxfp8_prequant_launchers[{config_key!r}]"


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
        and selected_prequant.rejection(problem) is None
    )
    if use_prequant:
        rejection = selected_prequant.rejection(problem)
        if rejection is not None:
            raise RuntimeError(f"prequant MXFP8 backend is unavailable: {rejection}")
        mode = _autotune_mode(autotune)
        if prequant_config is not None or mode == "off":
            key = (
                _DEFAULT_MXFP8_PREQUANT_KEY
                if prequant_config is None
                else _intern_prequant_config(selected_prequant)
            )
        else:
            # This request token survives torch.compile tracing. The generated
            # launcher resolves/tunes it against real tensors on first use.
            key = _intern_prequant_autotune_request(
                mode,
                tuning_policy,
                autotune_cache_dir,
                selected_prequant,
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
        selected_prequant = prequant_config or DEFAULT_MXFP8_PREQUANT_CONFIG
        mode = _autotune_mode(autotune)
        if prequant_config is not None or mode == "off":
            self._prequant_config_key = _intern_prequant_config(selected_prequant)
        else:
            self._prequant_config_key = _intern_prequant_autotune_request(
                mode,
                tuning_policy,
                autotune_cache_dir,
                selected_prequant,
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
        use_prequant = self.backend == "prequant"
        if (
            self.backend == "auto"
            and self.config is None
            and x.ndim >= 1
            and x.shape[-1] == self.in_features
        ):
            candidate = self.prequant_config or DEFAULT_MXFP8_PREQUANT_CONFIG
            problem = MXFP8Problem(
                x.reshape(-1, self.in_features).shape[0],
                self.out_features,
                self.in_features,
            )
            use_prequant = candidate.rejection(problem) is None
        if use_prequant:
            if x.ndim < 1 or x.shape[-1] != self.in_features:
                raise ValueError(
                    f"expected activation [..., {self.in_features}], got {x.shape}"
                )
            leading_shape = x.shape[:-1]
            x_2d = x.reshape(-1, self.in_features)
            _check_inputs(x_2d, self.weight)
            out = _run_prequant(x_2d, self.weight, self._prequant_config_key)
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
