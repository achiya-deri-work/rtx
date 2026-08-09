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

from .formats import MXFP8Tensor, make_mxfp8_tensor
from .formats.common import (
    PACKED_OPERAND_SCHEMA_VERSION,
    SCALE_LAYOUT_CODES,
    reject_packed_dtype_conversion,
)
from .formats.mxfp8 import (
    mxfp8_matrix_shape,
    mxfp8_orientation,
    mxfp8_qdata_2d,
    mxfp8_scale_layout,
    mxfp8_scales_for_kernel,
    validate_mxfp8_tensor,
)
from .configs import MXFP8GemmConfig, MXFP8QuantConfig
from .kernels.mxfp8 import (
    DEFAULT_MXFP8_FWD_CONFIG,
    MXFP8FwdConfig,
    MXFP8Problem,
    fwd_config_from_dict,
)
from .runtime import BoundedCache, load_kernel_symbol, runner_cache_limit

if TYPE_CHECKING:
    from .autotune import CoordinateDescentPolicy
    from .kernels.mxfp8_bwd import MXFP8BwdConfig

AutotuneMode = Literal["off", "cache", "coordinate"]
MXFP8Backend = Literal["auto", "fused", "prequant"]
WeightMode = Literal["dynamic", "prequantized"]


def compile_mxfp8_fwd(*args, **kwargs):
    return load_kernel_symbol("mxfp8_fwd", "compile_mxfp8_fwd")(*args, **kwargs)


def compile_mxfp8_gemm(*args, **kwargs):
    return load_kernel_symbol("mxfp8_gemm", "compile_mxfp8_gemm")(*args, **kwargs)


def compile_mxfp8_quant(*args, **kwargs):
    return load_kernel_symbol("mxfp8_quant", "compile_mxfp8_quant")(*args, **kwargs)


def compile_mxfp8_dual_quant(*args, **kwargs):
    return load_kernel_symbol("mxfp8_quant", "compile_mxfp8_dual_quant")(
        *args, **kwargs
    )


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
DEFAULT_MXFP8_INFERENCE_CONFIG = MXFP8PrequantConfig(
    quant=MXFP8QuantConfig(),
    gemm=MXFP8GemmConfig(epilogue="direct", store_vec=1),
)


_CONFIGS: dict[str, MXFP8FwdConfig] = {}
_FWD_AUTOTUNE_REQUESTS: dict[str, "_FwdAutotuneRequest"] = {}
_FWD_AUTOTUNE_SELECTIONS: dict[tuple[object, ...], str] = {}
_PREQUANT_CONFIGS: dict[str, MXFP8PrequantConfig] = {}
_PREQUANT_AUTOTUNE_REQUESTS: dict[str, "_PrequantAutotuneRequest"] = {}
_PACKED_INFERENCE_AUTOTUNE_REQUESTS: dict[
    str, "_PackedInferenceAutotuneRequest"
] = {}
_PREQUANT_AUTOTUNE_SELECTIONS: dict[tuple[object, ...], str] = {}
_PREQUANT_RUNNERS: BoundedCache[
    tuple[object, ...],
    "_PrequantRunner",
] = BoundedCache(runner_cache_limit("prequant", 8))
_WEIGHT_PREQUANT_RUNNERS: BoundedCache[
    tuple[object, ...], "_WeightPrequantRunner"
] = BoundedCache(runner_cache_limit("weight_prequant", 8))
_PACKED_INFERENCE_SELECTIONS: dict[tuple[object, ...], MXFP8PrequantConfig] = {}


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


@dataclass(slots=True)
class _WeightPrequantRunner:
    quant_x: object
    gemm: object
    qx: torch.Tensor
    sx: torch.Tensor
    l2_fetch_granularity: int | None

    def __call__(
        self,
        x: torch.Tensor,
        weight: MXFP8Tensor,
        out: torch.Tensor,
    ) -> None:
        _ensure_l2_fetch_granularity(self.l2_fetch_granularity)
        self.quant_x(x, self.qx, self.sx)
        self.gemm(
            self.qx,
            mxfp8_qdata_2d(weight),
            self.sx,
            mxfp8_scales_for_kernel(weight),
            out,
        )


@dataclass(frozen=True, slots=True)
class _PrequantAutotuneRequest:
    mode: AutotuneMode
    policy: object | None
    cache_dir: str | None
    initial: MXFP8PrequantConfig


@dataclass(frozen=True, slots=True)
class _FwdAutotuneRequest:
    mode: AutotuneMode
    policy: object | None
    cache_dir: str | None


@dataclass(frozen=True, slots=True)
class _PackedInferenceAutotuneRequest:
    mode: AutotuneMode
    policy: object | None
    cache_dir: str | None


def _intern_config(config: MXFP8FwdConfig) -> str:
    key = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    with _CONFIG_LOCK:
        _CONFIGS[key] = config
    return key


@torch.compiler.assume_constant_result
def _intern_fwd_autotune_request(
    mode: AutotuneMode,
    policy: object | None,
    cache_dir: Path | str | None,
) -> str:
    payload = {
        "mode": mode,
        "policy": None if policy is None else asdict(policy),
        "cache_dir": None if cache_dir is None else str(Path(cache_dir).expanduser()),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    key = "fwd-autotune:" + hashlib.sha256(encoded.encode()).hexdigest()[:24]
    with _CONFIG_LOCK:
        _FWD_AUTOTUNE_REQUESTS[key] = _FwdAutotuneRequest(
            mode, policy, payload["cache_dir"]
        )
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


@torch.compiler.assume_constant_result
def _intern_packed_inference_autotune_request(
    mode: AutotuneMode,
    policy: object | None,
    cache_dir: Path | str | None,
) -> str:
    policy_value = None if policy is None else asdict(policy)
    payload = {
        "mode": mode,
        "policy": policy_value,
        "cache_dir": None if cache_dir is None else str(Path(cache_dir).expanduser()),
    }
    digest = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    key = "packed-autotune:" + hashlib.sha256(digest.encode()).hexdigest()[:24]
    with _CONFIG_LOCK:
        _PACKED_INFERENCE_AUTOTUNE_REQUESTS[key] = (
            _PackedInferenceAutotuneRequest(
                mode=mode,
                policy=policy,
                cache_dir=payload["cache_dir"],
            )
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
    return _launch_fused(x, weight, config_key)


def _launch_fused(
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
    request = _FWD_AUTOTUNE_REQUESTS.get(config_key)
    if request is not None:
        selection_key = (
            config_key,
            x.device.index,
            problem.m,
            problem.n,
            problem.k,
        )
        selected_key = _FWD_AUTOTUNE_SELECTIONS.get(selection_key)
        if selected_key is None:
            selected = _resolve_fwd_config(
                x_c,
                weight_c,
                config=None,
                autotune=request.mode,
                tuning_policy=request.policy,
                cache_dir=request.cache_dir,
                defer_compiler_trace=False,
            )
            selected_key = _intern_config(selected)
            _FWD_AUTOTUNE_SELECTIONS[selection_key] = selected_key
        config_key = selected_key
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


def quantize_mxfp8(
    tensor: torch.Tensor,
    *,
    config: MXFP8QuantConfig | None = None,
) -> MXFP8Tensor:
    """Prequantize BF16 once and return TorchAO's canonical ``MXTensor``."""

    if tensor.ndim < 1:
        raise ValueError("MXFP8 quantization requires at least one dimension")
    if tensor.device.type != "cuda":
        raise ValueError("MXFP8 quantization requires a CUDA tensor")
    if tensor.dtype is not torch.bfloat16:
        raise TypeError(f"MXFP8 quantization requires BF16, got {tensor.dtype}")
    leading_shape = tuple(int(value) for value in tensor.shape)
    source = tensor.reshape(-1, tensor.shape[-1])
    source = source if source.is_contiguous() else source.contiguous()
    selected = config or MXFP8QuantConfig()
    rejection = selected.rejection(int(source.shape[0]), int(source.shape[1]))
    if rejection is not None:
        raise RuntimeError(f"MXFP8 operand cannot be quantized: {rejection}")
    major, minor = torch.cuda.get_device_capability(tensor.device)
    if major != 12:
        raise RuntimeError(
            "native RTX MXFP8 quantization requires SM120/SM121; "
            f"got {(major, minor)}"
        )
    data = torch.empty_like(source, dtype=torch.float8_e4m3fn)
    scales = _allocate_scales(
        int(source.shape[0]),
        int(source.shape[1]),
        selected.scale_layout,
        source.device,
    )
    launcher = compile_mxfp8_quant(
        int(source.shape[0]), int(source.shape[1]), selected
    )
    launcher(source, data, scales)
    # CuTe/TVM-FFI launches asynchronously. Preserve the BF16 source until
    # both materialized outputs have become ready on the launch stream.
    data._base_inputs = (source,)
    scales._base_inputs = (source,)
    return make_mxfp8_tensor(
        data, scales, leading_shape, selected.scale_layout
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
            from .autotune.winners import load_runtime_winner, runtime_winner_key
            from .prequant_autotune import (
                load_cached_mxfp8_prequant_config,
                prequant_config_from_dict,
                tune_mxfp8_prequant,
            )

            selected = load_runtime_winner(
                runtime_winner_key(
                    "mxfp8_prequant_fwd", problem, device=x.device
                ),
                prequant_config_from_dict,
                root=request.cache_dir,
                rejection=lambda candidate: candidate.rejection(problem),
            )
            if selected is not None:
                pass
            elif request.mode == "coordinate":
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
        self.runners: BoundedCache[
            tuple[int | None, int, int, int, int], tuple
        ] = BoundedCache(runner_cache_limit("inductor_prequant", 8))

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


def _packed_weight_from_tensors(
    data: torch.Tensor,
    scales: torch.Tensor,
    n: int,
    k: int,
    scale_layout: str,
) -> MXFP8Tensor:
    return make_mxfp8_tensor(
        data, scales, (n, k), scale_layout  # type: ignore[arg-type]
    )


def _validate_packed_linear_operands(
    x: MXFP8Tensor,
    weight: MXFP8Tensor,
) -> MXFP8Problem:
    validate_mxfp8_tensor(x)
    validate_mxfp8_tensor(weight)
    if (
        mxfp8_orientation(x) != "row_major"
        or mxfp8_orientation(weight) != "row_major"
    ):
        raise ValueError("public packed linear operands must be row-major logical views")
    m, k = mxfp8_matrix_shape(x)
    n, weight_k = mxfp8_matrix_shape(weight)
    if k != weight_k:
        raise ValueError(f"packed linear K mismatch: X={k}, W={weight_k}")
    if x.device != weight.device:
        raise ValueError("packed X and W must be on one CUDA device")
    if x.device.type != "cuda":
        raise ValueError("packed MXFP8 execution requires CUDA operands")
    return MXFP8Problem(m, n, k)


def _packed_gemm_config(
    problem: MXFP8Problem,
    x_layout: str,
    weight_layout: str,
    config_key: str,
) -> MXFP8PrequantConfig:
    try:
        config = _PREQUANT_CONFIGS[config_key]
    except KeyError as exc:
        raise RuntimeError("unknown MXFP8 inference configuration key") from exc
    expected = {
        "row_major": ("row_major", "row_major"),
        "mma128": ("mma128", "mma128"),
        "mma64x128": ("mma64", "mma128"),
    }.get(config.gemm.scale_layout)
    if expected != (x_layout, weight_layout):
        raise RuntimeError(
            f"packed layouts {(x_layout, weight_layout)} are incompatible with "
            f"GEMM layout {config.gemm.scale_layout}"
        )
    rejection = config.gemm.rejection(problem)
    if rejection is not None:
        raise RuntimeError(f"packed MXFP8 GEMM cannot run: {rejection}")
    return config


def _launch_weight_prequant_out(
    x: torch.Tensor,
    weight: MXFP8Tensor,
    config_key: str,
    out: torch.Tensor,
) -> None:
    if x.ndim != 2:
        raise ValueError("internal dynamic-X/prequant-W op expects 2D X")
    if x.device.type != "cuda" or x.dtype is not torch.bfloat16:
        raise TypeError("dynamic-X/prequant-W MXFP8 expects CUDA BF16 X")
    n, k = mxfp8_matrix_shape(weight)
    weight_layout = mxfp8_scale_layout(weight)
    if x.shape[1] != k or x.device != weight.device:
        raise ValueError("dynamic X and packed W have incompatible shape/device")
    x_c = x if x.is_contiguous() else x.contiguous()
    problem = MXFP8Problem(int(x_c.shape[0]), n, k)
    config_key = _resolve_packed_inference_request(
        problem, weight, x=None, config_key=config_key
    )
    try:
        requested = _PREQUANT_CONFIGS[config_key]
    except KeyError as exc:
        raise RuntimeError("unknown MXFP8 inference configuration key") from exc
    config = _packed_gemm_config(
        problem,
        requested.quant.scale_layout,
        weight_layout,
        config_key,
    )
    rejection = config.quant.rejection(problem.m, problem.k)
    if rejection is not None:
        raise RuntimeError(f"activation quantizer cannot run: {rejection}")
    stream = torch.cuda.current_stream(x.device)
    runner_key = (
        x.device.index,
        int(stream.cuda_stream),
        problem.m,
        problem.n,
        problem.k,
        weight_layout,
        config_key,
    )
    runner = _WEIGHT_PREQUANT_RUNNERS.get(runner_key)
    if runner is None:
        with _CONFIG_LOCK:
            runner = _WEIGHT_PREQUANT_RUNNERS.get(runner_key)
            if runner is None:
                if config.l2_fetch_granularity is not None:
                    _set_l2_fetch_granularity(config.l2_fetch_granularity)
                qx = torch.empty_like(x_c, dtype=torch.float8_e4m3fn)
                sx = _allocate_scales(
                    problem.m,
                    problem.k,
                    config.quant.scale_layout,
                    x.device,
                )
                runner = _WeightPrequantRunner(
                    quant_x=compile_mxfp8_quant(
                        problem.m, problem.k, config.quant
                    ),
                    gemm=compile_mxfp8_gemm(problem, config.gemm),
                    qx=qx,
                    sx=sx,
                    l2_fetch_granularity=config.l2_fetch_granularity,
                )
                _WEIGHT_PREQUANT_RUNNERS[runner_key] = runner
    runner(x_c, weight, out)


@torch.library.custom_op(
    "rtx::mxfp8_linear_dynamic_x_prequant_w",
    mutates_args=(),
    device_types="cuda",
)
def _mxfp8_linear_dynamic_x_prequant_w_op(
    x: torch.Tensor,
    weight_data: torch.Tensor,
    weight_scales: torch.Tensor,
    n: int,
    k: int,
    weight_scale_layout: str,
    config_key: str,
) -> torch.Tensor:
    weight = _packed_weight_from_tensors(
        weight_data, weight_scales, n, k, weight_scale_layout
    )
    out = torch.empty((x.shape[0], n), dtype=torch.bfloat16, device=x.device)
    _launch_weight_prequant_out(x, weight, config_key, out)
    out._base_inputs = (x, weight_data, weight_scales)
    return out


@_mxfp8_linear_dynamic_x_prequant_w_op.register_fake
def _mxfp8_linear_dynamic_x_prequant_w_fake(
    x: torch.Tensor,
    weight_data: torch.Tensor,
    weight_scales: torch.Tensor,
    n: int,
    k: int,
    weight_scale_layout: str,
    config_key: str,
) -> torch.Tensor:
    return torch.empty((x.shape[0], n), dtype=torch.bfloat16, device=x.device)


@torch.library.custom_op(
    "rtx::mxfp8_linear_prequantized",
    mutates_args=(),
    device_types="cuda",
)
def _mxfp8_linear_prequantized_op(
    x_data: torch.Tensor,
    weight_data: torch.Tensor,
    x_scales: torch.Tensor,
    weight_scales: torch.Tensor,
    m: int,
    n: int,
    k: int,
    x_scale_layout: str,
    weight_scale_layout: str,
    config_key: str,
) -> torch.Tensor:
    x = _packed_weight_from_tensors(x_data, x_scales, m, k, x_scale_layout)
    weight = _packed_weight_from_tensors(
        weight_data, weight_scales, n, k, weight_scale_layout
    )
    problem = _validate_packed_linear_operands(x, weight)
    config_key = _resolve_packed_inference_request(
        problem, weight, x=x, config_key=config_key
    )
    config = _packed_gemm_config(
        problem,
        mxfp8_scale_layout(x),
        mxfp8_scale_layout(weight),
        config_key,
    )
    _ensure_l2_fetch_granularity(config.l2_fetch_granularity)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    launcher = compile_mxfp8_gemm(problem, config.gemm)
    launcher(
        mxfp8_qdata_2d(x),
        mxfp8_qdata_2d(weight),
        mxfp8_scales_for_kernel(x),
        mxfp8_scales_for_kernel(weight),
        out,
    )
    out._base_inputs = (x_data, weight_data, x_scales, weight_scales)
    return out


@_mxfp8_linear_prequantized_op.register_fake
def _mxfp8_linear_prequantized_fake(
    x_data: torch.Tensor,
    weight_data: torch.Tensor,
    x_scales: torch.Tensor,
    weight_scales: torch.Tensor,
    m: int,
    n: int,
    k: int,
    x_scale_layout: str,
    weight_scale_layout: str,
    config_key: str,
) -> torch.Tensor:
    return torch.empty((m, n), dtype=torch.bfloat16, device=x_data.device)


def _run_weight_prequantized(
    x: torch.Tensor,
    weight: MXFP8Tensor,
    config_key: str,
) -> torch.Tensor:
    n, k = mxfp8_matrix_shape(weight)
    return _mxfp8_linear_dynamic_x_prequant_w_op(
        x,
        mxfp8_qdata_2d(weight),
        mxfp8_scales_for_kernel(weight),
        n,
        k,
        mxfp8_scale_layout(weight),
        config_key,
    )


def _run_fully_prequantized(
    x: MXFP8Tensor,
    weight: MXFP8Tensor,
    config_key: str,
) -> torch.Tensor:
    problem = _validate_packed_linear_operands(x, weight)
    return _mxfp8_linear_prequantized_op(
        mxfp8_qdata_2d(x),
        mxfp8_qdata_2d(weight),
        mxfp8_scales_for_kernel(x),
        mxfp8_scales_for_kernel(weight),
        problem.m,
        problem.n,
        problem.k,
        mxfp8_scale_layout(x),
        mxfp8_scale_layout(weight),
        config_key,
    )


def _default_packed_inference_config(
    problem: MXFP8Problem,
    weight: MXFP8Tensor,
    x: MXFP8Tensor | None,
) -> MXFP8PrequantConfig:
    weight_layout = mxfp8_scale_layout(weight)
    if x is None:
        if weight_layout == "row_major":
            layouts = ("row_major", "row_major")
        elif problem.m % 128 == 0:
            layouts = ("mma128", "mma128")
        elif problem.m % 64 == 0:
            layouts = ("mma64", "mma128")
        else:
            raise RuntimeError(
                "native packed W requires M divisible by 64 for a compatible "
                "activation scale layout"
            )
    else:
        layouts = (mxfp8_scale_layout(x), weight_layout)
    if layouts == ("row_major", "row_major"):
        return DEFAULT_MXFP8_INFERENCE_CONFIG
    if layouts == ("mma128", "mma128"):
        gemm = MXFP8GemmConfig(
            atom_layout_m=4,
            scale_role="tma",
            scale_layout="mma128",
            epilogue="direct",
            store_vec=1,
        )
    elif layouts == ("mma64", "mma128"):
        gemm = MXFP8GemmConfig(
            tile_m=64,
            atom_layout_m=2,
            stages=1,
            scale_role="tma",
            scale_layout="mma64x128",
            epilogue="direct",
            store_vec=1,
        )
    else:
        raise RuntimeError(
            f"no MXFP8 GEMM transport accepts packed layouts {layouts}"
        )
    return MXFP8PrequantConfig(
        quant=replace(MXFP8QuantConfig(), scale_layout=layouts[0]),
        weight_quant=replace(MXFP8QuantConfig(), scale_layout=layouts[1]),
        gemm=gemm,
        quant_launches="separate",
    )


def _resolve_packed_inference_config(
    problem: MXFP8Problem,
    weight: MXFP8Tensor,
    *,
    x: MXFP8Tensor | None,
    explicit: MXFP8PrequantConfig | None,
    autotune: AutotuneMode | bool | None,
    tuning_policy: object | None,
    cache_dir: Path | str | None,
) -> MXFP8PrequantConfig:
    if explicit is not None:
        return explicit
    fallback = _default_packed_inference_config(problem, weight, x)
    weight_layout = mxfp8_scale_layout(weight)
    x_layout = None if x is None else mxfp8_scale_layout(x)
    mode = _autotune_mode(autotune)
    if (
        mode == "off"
        or torch.compiler.is_compiling()
        or not torch.cuda.is_available()
    ):
        return fallback
    selection_key = (
        "fully_prequantized" if x is not None else "weight_prequantized",
        weight.device.index,
        problem.m,
        problem.n,
        problem.k,
        x_layout,
        weight_layout,
        mode,
        None if cache_dir is None else str(Path(cache_dir).expanduser()),
    )
    cached_selection = _PACKED_INFERENCE_SELECTIONS.get(selection_key)
    if cached_selection is not None:
        return cached_selection
    from .autotune.winners import load_runtime_winner, runtime_winner_key
    from .inference_autotune import (
        fully_prequant_config_from_dict,
        tune_mxfp8_inference_state,
        weight_prequant_config_from_dict,
    )

    if x is None:
        from .configs import MXFP8WeightPrequantConfig

        family = "mxfp8_weight_prequant_fwd"
        variant = f"w-{weight_layout}"
        key = runtime_winner_key(
            family, problem, device=weight.device, variant=variant
        )
        selected = load_runtime_winner(
            key,
            weight_prequant_config_from_dict,
            root=cache_dir,
            rejection=lambda config: (
                config.rejection(problem)
                or (
                    "cached weight scale layout does not match packed W"
                    if config.operand_scale_layouts[1] != weight_layout
                    else None
                )
            ),
        )
        if selected is None:
            if mode != "coordinate":
                _PACKED_INFERENCE_SELECTIONS[selection_key] = fallback
                return fallback
            selected = tune_mxfp8_inference_state(
                problem,
                state="weight_prequantized",
                weight_layout=weight_layout,
                device=weight.device,
                cache_dir=cache_dir,
                policy=tuning_policy,
            )
        assert isinstance(selected, MXFP8WeightPrequantConfig)
        resolved = MXFP8PrequantConfig(
            quant=selected.quant_x,
            weight_quant=selected.weight_packing_quant(),
            gemm=selected.gemm,
            quant_launches="separate",
            l2_fetch_granularity=selected.l2_fetch_granularity,
        )
        _PACKED_INFERENCE_SELECTIONS[selection_key] = resolved
        return resolved

    from .configs import MXFP8FullyPrequantConfig

    family = "mxfp8_fully_prequant_fwd"
    assert x_layout is not None
    variant = f"x-{x_layout}_w-{weight_layout}"
    key = runtime_winner_key(
        family, problem, device=weight.device, variant=variant
    )
    selected = load_runtime_winner(
        key,
        fully_prequant_config_from_dict,
        root=cache_dir,
        rejection=lambda config: (
            config.rejection(problem)
            or (
                "cached operand layouts do not match packed X/W"
                if config.operand_scale_layouts
                != (x_layout, weight_layout)
                else None
            )
        ),
    )
    if selected is None:
        if mode != "coordinate":
            _PACKED_INFERENCE_SELECTIONS[selection_key] = fallback
            return fallback
        selected = tune_mxfp8_inference_state(
            problem,
            state="fully_prequantized",
            activation_layout=x_layout,
            weight_layout=weight_layout,
            device=weight.device,
            cache_dir=cache_dir,
            policy=tuning_policy,
        )
    assert isinstance(selected, MXFP8FullyPrequantConfig)
    resolved = MXFP8PrequantConfig(
        quant=selected.activation_packing_quant(),
        weight_quant=selected.weight_packing_quant(),
        gemm=selected.gemm,
        quant_launches="separate",
        l2_fetch_granularity=selected.l2_fetch_granularity,
    )
    _PACKED_INFERENCE_SELECTIONS[selection_key] = resolved
    return resolved


def _resolve_packed_inference_request(
    problem: MXFP8Problem,
    weight: MXFP8Tensor,
    *,
    x: MXFP8Tensor | None,
    config_key: str,
) -> str:
    request = _PACKED_INFERENCE_AUTOTUNE_REQUESTS.get(config_key)
    if request is None:
        return config_key
    selected = _resolve_packed_inference_config(
        problem,
        weight,
        x=x,
        explicit=None,
        autotune=request.mode,
        tuning_policy=request.policy,
        cache_dir=request.cache_dir,
    )
    return _intern_prequant_config(selected)


def _packed_inference_config_key(
    problem: MXFP8Problem,
    weight: MXFP8Tensor,
    *,
    x: MXFP8Tensor | None,
    explicit: MXFP8PrequantConfig | None,
    autotune: AutotuneMode | bool | None,
    tuning_policy: object | None,
    cache_dir: Path | str | None,
) -> str:
    mode = _autotune_mode(autotune)
    if explicit is not None or mode == "off":
        selected = _resolve_packed_inference_config(
            problem,
            weight,
            x=x,
            explicit=explicit,
            autotune=mode,
            tuning_policy=tuning_policy,
            cache_dir=cache_dir,
        )
        return _intern_prequant_config(selected)
    # The request token remains opaque to Dynamo/Inductor. The custom-op
    # implementation resolves it against the real device, shape, and physical
    # operand layouts on first execution.
    return _intern_packed_inference_autotune_request(
        mode, tuning_policy, cache_dir
    )


def _launch_training_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    config_key: str,
) -> torch.Tensor:
    """Launch either registered MXFP8 forward family for the autograd op."""

    if config_key in _CONFIGS or config_key in _FWD_AUTOTUNE_REQUESTS:
        return _launch_fused(x, weight, config_key)
    out = torch.empty(
        (x.shape[0], weight.shape[0]), dtype=torch.bfloat16, device=x.device
    )
    _launch_prequant_out(x, weight, config_key, out)
    return out


def mxfp8_linear(
    x: torch.Tensor | MXFP8Tensor,
    weight: torch.Tensor | MXFP8Tensor,
    *,
    config: MXFP8FwdConfig | None = None,
    autotune: AutotuneMode | bool | None = None,
    tuning_policy: "CoordinateDescentPolicy | None" = None,
    autotune_cache_dir: Path | str | None = None,
    backend: MXFP8Backend = "auto",
    prequant_config: MXFP8PrequantConfig | None = None,
    backward_config: "MXFP8BwdConfig | None" = None,
) -> torch.Tensor:
    """Apply MXFP8 linear in dynamic or prequantized inference states.

    Leading activation dimensions are flattened for the kernel and restored in
    the BF16 result. A packed weight makes the operation inference-only; a
    packed activation additionally skips all quantization in this invocation.
    """

    if isinstance(weight, MXFP8Tensor):
        validate_mxfp8_tensor(weight)
        if isinstance(x, MXFP8Tensor):
            if x.shape[-1] != weight.shape[-1]:
                raise ValueError("packed activation and weight K must match")
            problem = _validate_packed_linear_operands(x, weight)
            key = _packed_inference_config_key(
                problem,
                weight,
                x=x,
                explicit=prequant_config,
                autotune=autotune,
                tuning_policy=tuning_policy,
                cache_dir=autotune_cache_dir,
            )
            out = _run_fully_prequantized(x, weight, key)
            return out.reshape(*x.shape[:-1], mxfp8_matrix_shape(weight)[0])
        if x.ndim < 1 or x.shape[-1] != weight.shape[-1]:
            raise ValueError(
                f"expected activation [..., {weight.shape[-1]}], got {x.shape}"
            )
        if x.device.type != "cuda" or weight.device.type != "cuda":
            raise ValueError("dynamic-X/prequant-W MXFP8 execution requires CUDA")
        if x.device != weight.device:
            raise ValueError("dynamic X and packed W must be on one CUDA device")
        if x.dtype is not torch.bfloat16:
            raise TypeError(f"dynamic MXFP8 activation must be BF16, got {x.dtype}")
        if torch.is_grad_enabled() and x.requires_grad:
            raise RuntimeError(
                "prequantized MXFP8 weights are inference-only and cannot "
                "participate in autograd"
            )
        leading_shape = x.shape[:-1]
        x_2d = x.reshape(-1, x.shape[-1])
        weight_rows, weight_k = mxfp8_matrix_shape(weight)
        problem = MXFP8Problem(int(x_2d.shape[0]), weight_rows, weight_k)
        key = _packed_inference_config_key(
            problem,
            weight,
            x=None,
            explicit=prequant_config,
            autotune=autotune,
            tuning_policy=tuning_policy,
            cache_dir=autotune_cache_dir,
        )
        out = _run_weight_prequantized(x_2d, weight, key)
        return out.reshape(*leading_shape, weight_rows)
    if isinstance(x, MXFP8Tensor):
        raise TypeError("a prequantized MXFP8 activation requires a prequantized weight")

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
        if torch.is_grad_enabled() and (
            x_2d.requires_grad or weight.requires_grad
        ):
            from .fp8_bwd import (
                _intern_bwd_config,
                _mxfp8_linear_train_op,
            )

            bwd_key = (
                _intern_bwd_config(backward_config)
                if backward_config is not None
                else _backward_config_key(
                    mode, tuning_policy, autotune_cache_dir
                )
            )
            out = _mxfp8_linear_train_op(x_2d, weight, key, bwd_key)
        else:
            out = _run_prequant(x_2d, weight, key)
        return out.reshape(*leading_shape, weight.shape[0])
    mode = _autotune_mode(autotune)
    if config is None and mode != "off" and torch.compiler.is_compiling():
        # Resolve the device/shape-specific runtime winner in the custom-op
        # implementation, outside Dynamo's graph-capture side effects.
        key = _intern_fwd_autotune_request(
            mode, tuning_policy, autotune_cache_dir
        )
    else:
        selected_config = _resolve_fwd_config(
            x_2d,
            weight,
            config=config,
            autotune=mode,
            tuning_policy=tuning_policy,
            cache_dir=autotune_cache_dir,
        )
        key = _intern_config(selected_config)
    if torch.is_grad_enabled() and (x_2d.requires_grad or weight.requires_grad):
        from .fp8_bwd import (
            _intern_bwd_config,
            _mxfp8_linear_train_op,
        )

        bwd_key = (
            _intern_bwd_config(backward_config)
            if backward_config is not None
            else _backward_config_key(mode, tuning_policy, autotune_cache_dir)
        )
        out = _mxfp8_linear_train_op(x_2d, weight, key, bwd_key)
    else:
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


def _backward_config_key(
    mode: AutotuneMode,
    policy: object | None,
    cache_dir: Path | str | None,
) -> str:
    from .fp8_bwd import (
        _DEFAULT_BWD_KEY,
        _intern_bwd_autotune_request,
    )

    return (
        _DEFAULT_BWD_KEY
        if mode == "off"
        else _intern_bwd_autotune_request(mode, policy, cache_dir)
    )


def _clear_runtime_caches() -> dict[str, object]:
    """Internal half of :func:`rtx.clear_runtime_caches`."""

    before = {
        "prequant": _PREQUANT_RUNNERS.stats(),
        "weight_prequant": _WEIGHT_PREQUANT_RUNNERS.stats(),
        "inductor_launchers": len(_INDUCTOR_PREQUANT_LAUNCHERS),
        "inductor_runners": sum(
            len(launcher.runners)
            for launcher in _INDUCTOR_PREQUANT_LAUNCHERS.values()
            if isinstance(launcher, _InductorPrequantLauncher)
        ),
    }
    _PREQUANT_RUNNERS.clear()
    _WEIGHT_PREQUANT_RUNNERS.clear()
    for launcher in _INDUCTOR_PREQUANT_LAUNCHERS.values():
        if isinstance(launcher, _InductorPrequantLauncher):
            launcher.runners.clear()
    _PACKED_INFERENCE_SELECTIONS.clear()
    _PREQUANT_AUTOTUNE_SELECTIONS.clear()
    _FWD_AUTOTUNE_SELECTIONS.clear()
    return before


def _resolve_fwd_config(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    config: MXFP8FwdConfig | None,
    autotune: AutotuneMode | bool | None,
    tuning_policy: "CoordinateDescentPolicy | None",
    cache_dir: Path | str | None,
    defer_compiler_trace: bool = True,
) -> MXFP8FwdConfig:
    if config is not None:
        return config
    mode = _autotune_mode(autotune)
    if mode == "off":
        return DEFAULT_MXFP8_FWD_CONFIG
    # Tuning launches and synchronizes kernels; never start it from a compiler
    # trace. Cache-only selection is also skipped to keep graph capture pure.
    if defer_compiler_trace and torch.compiler.is_compiling():
        return DEFAULT_MXFP8_FWD_CONFIG

    from .autotune import (
        CoordinateDescentPolicy,
        load_cached_mxfp8_fwd_config,
        tune_mxfp8_fwd,
    )
    from .autotune.winners import load_runtime_winner, runtime_winner_key

    problem = MXFP8Problem(x.shape[0], weight.shape[0], x.shape[1])
    cached = load_runtime_winner(
        runtime_winner_key("mxfp8_fused_fwd", problem, device=x.device),
        lambda value: fwd_config_from_dict(dict(value)),
        root=cache_dir,
        rejection=lambda candidate: candidate.implementation_rejection(problem),
    )
    if cached is None:
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
    """No-bias MXFP8 linear supporting dynamic and packed inference operands."""

    __constants__ = ["in_features", "out_features", "backend", "weight_mode"]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        config: MXFP8FwdConfig | None = None,
        autotune: AutotuneMode | bool | None = None,
        tuning_policy: "CoordinateDescentPolicy | None" = None,
        autotune_cache_dir: Path | str | None = None,
        backend: MXFP8Backend = "auto",
        prequant_config: MXFP8PrequantConfig | None = None,
        backward_config: "MXFP8BwdConfig | None" = None,
        packed_weight: MXFP8Tensor | None = None,
    ) -> None:
        super().__init__()
        if bias:
            raise NotImplementedError(
                "MXFP8Linear is a no-bias linear layer; pass bias=False"
            )
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
        self.backward_config = backward_config
        self.weight_mode: WeightMode = (
            "prequantized" if packed_weight is not None else "dynamic"
        )
        selected_prequant = prequant_config or (
            DEFAULT_MXFP8_INFERENCE_CONFIG
            if packed_weight is not None
            else DEFAULT_MXFP8_PREQUANT_CONFIG
        )
        mode = _autotune_mode(autotune)
        if prequant_config is not None or mode == "off":
            self._prequant_config_key = _intern_prequant_config(
                selected_prequant
            )
        else:
            self._prequant_config_key = _intern_prequant_autotune_request(
                mode,
                tuning_policy,
                autotune_cache_dir,
                selected_prequant,
            )
        from .fp8_bwd import _intern_bwd_config

        self._backward_config_key = (
            _intern_bwd_config(backward_config)
            if backward_config is not None
            else _backward_config_key(
                mode, tuning_policy, autotune_cache_dir
            )
        )
        if packed_weight is None:
            self.weight = nn.Parameter(
                torch.empty(
                    (out_features, in_features), device=device, dtype=torch.bfloat16
                )
            )
            self.reset_parameters()
        else:
            validate_mxfp8_tensor(packed_weight)
            if packed_weight.shape != (out_features, in_features):
                raise ValueError(
                    "packed MXFP8 weight shape must equal "
                    f"{(out_features, in_features)}, got {packed_weight.shape}"
                )
            if mxfp8_orientation(packed_weight) != "row_major":
                raise ValueError("packed linear weights must be row-major")
            if device is not None:
                packed_weight = packed_weight.to(device)
            packed_layout = mxfp8_scale_layout(packed_weight)
            self.register_parameter("weight", None)
            self.register_buffer("weight_data", mxfp8_qdata_2d(packed_weight))
            self.register_buffer("weight_scales", packed_weight.scale)
            self.register_buffer(
                "weight_packing_meta",
                torch.tensor(
                    [
                        PACKED_OPERAND_SCHEMA_VERSION,
                        SCALE_LAYOUT_CODES[packed_layout],
                    ],
                    dtype=torch.int64,
                    device=packed_weight.device,
                ),
            )
            self._weight_scale_layout = packed_layout
            self._weight_packing_schema = PACKED_OPERAND_SCHEMA_VERSION
            self.training = False

    @property
    def bias(self) -> None:
        return None

    def to(self, *args, **kwargs):
        if self.weight_mode == "prequantized":
            reject_packed_dtype_conversion(args, kwargs, format_name="MXFP8")
        return super().to(*args, **kwargs)

    def train(self, mode: bool = True):
        if self.weight_mode == "prequantized" and mode:
            raise RuntimeError(
                "a prequantized MXFP8Linear is inference-only; keep a dynamic "
                "BF16-master module for training"
            )
        return super().train(mode)

    def half(self):
        if self.weight_mode == "prequantized":
            raise TypeError("a prequantized MXFP8 module cannot be dtype-cast")
        return super().half()

    def float(self):
        if self.weight_mode == "prequantized":
            raise TypeError("a prequantized MXFP8 module cannot be dtype-cast")
        return super().float()

    def bfloat16(self):
        if self.weight_mode == "prequantized":
            raise TypeError("a prequantized MXFP8 module cannot be dtype-cast")
        return super().bfloat16()

    def type(self, dst_type=None):
        if self.weight_mode == "prequantized" and dst_type is not None:
            raise TypeError("a prequantized MXFP8 module cannot be dtype-cast")
        return super().type(dst_type)

    def reset_parameters(self) -> None:
        if self.weight is not None:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    @property
    def packed_weight(self) -> MXFP8Tensor | None:
        if self.weight_mode != "prequantized":
            return None
        return make_mxfp8_tensor(
            self.weight_data,
            self.weight_scales,
            (self.out_features, self.in_features),
            self._weight_scale_layout,
        )

    @classmethod
    def from_float(
        cls,
        module: nn.Linear,
        *,
        prequant_config: MXFP8PrequantConfig | None = None,
    ) -> "MXFP8Linear":
        """Create an inference module with its weight quantized exactly once."""

        if module.bias is not None:
            raise NotImplementedError("MXFP8Linear.from_float requires bias=False")
        if module.weight.dtype is not torch.bfloat16:
            raise TypeError("MXFP8Linear.from_float requires a BF16 weight")
        selected = prequant_config or DEFAULT_MXFP8_INFERENCE_CONFIG
        packed = quantize_mxfp8(
            module.weight.detach(), config=selected.resolved_weight_quant()
        )
        return cls(
            module.in_features,
            module.out_features,
            bias=False,
            device=module.weight.device,
            prequant_config=prequant_config,
            packed_weight=packed,
        )

    def to_quantized_weight(
        self,
        *,
        prequant_config: MXFP8PrequantConfig | None = None,
    ) -> "MXFP8Linear":
        """Return an inference-only copy with a persistent packed weight."""

        if self.weight_mode == "prequantized":
            return self
        assert self.weight is not None
        selected = prequant_config or self.prequant_config or DEFAULT_MXFP8_INFERENCE_CONFIG
        packed = quantize_mxfp8(
            self.weight.detach(), config=selected.resolved_weight_quant()
        )
        return type(self)(
            self.in_features,
            self.out_features,
            bias=False,
            device=self.weight.device,
            config=self.config,
            autotune=self.autotune,
            tuning_policy=self.tuning_policy,
            autotune_cache_dir=self.autotune_cache_dir,
            backend=self.backend,
            prequant_config=(
                prequant_config
                if prequant_config is not None
                else self.prequant_config
            ),
            backward_config=self.backward_config,
            packed_weight=packed,
        )

    def forward(self, x: torch.Tensor | MXFP8Tensor) -> torch.Tensor:
        if self.weight_mode == "prequantized":
            packed_weight = self.packed_weight
            assert packed_weight is not None
            return mxfp8_linear(
                x,
                packed_weight,
                autotune=self.autotune,
                tuning_policy=self.tuning_policy,
                autotune_cache_dir=self.autotune_cache_dir,
                prequant_config=self.prequant_config,
            )
        if isinstance(x, MXFP8Tensor):
            raise TypeError(
                "a prequantized activation requires a prequantized module weight"
            )
        assert self.weight is not None
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
            if torch.is_grad_enabled() and (
                x_2d.requires_grad or self.weight.requires_grad
            ):
                from .fp8_bwd import _mxfp8_linear_train_op

                out = _mxfp8_linear_train_op(
                    x_2d,
                    self.weight,
                    self._prequant_config_key,
                    self._backward_config_key,
                )
            else:
                out = _run_prequant(
                    x_2d, self.weight, self._prequant_config_key
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
            backward_config=self.backward_config,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias=False, format=E4M3xE8M0, backend={self.backend}, "
            f"weight_mode={self.weight_mode}"
        )


__all__ = [
    "DEFAULT_MXFP8_PREQUANT_CONFIG",
    "DEFAULT_MXFP8_INFERENCE_CONFIG",
    "MXFP8Linear",
    "MXFP8PrequantConfig",
    "mxfp8_linear",
    "quantize_mxfp8",
]
