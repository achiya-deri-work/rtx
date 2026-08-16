"""Torch frontend for NVFP4-forward, MXFP8-backward linear layers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Literal

import torch
from torch import nn

from .configs import (
    DEFAULT_NVFP4_GEMM_CONFIG,
    DEFAULT_NVFP4_SCALE_CONFIG,
    DEFAULT_NVFP4_DYNAMIC_CONFIG,
    DEFAULT_NVFP4_QUANT_CONFIG,
    NVFP4ScaleConfig,
    NVFP4FullyPrequantConfig,
    NVFP4DynamicConfig,
    NVFP4GemmConfig,
    NVFP4Problem,
    NVFP4QuantConfig,
    NVFP4WeightPrequantConfig,
)
from .formats import NVFP4Tensor, make_nvfp4_tensor
from .formats.common import (
    PACKED_OPERAND_SCHEMA_VERSION,
    SCALE_LAYOUT_CODES,
    reject_packed_dtype_conversion,
)
from .formats.nvfp4 import (
    nvfp4_matrix_shape,
    nvfp4_orientation,
    nvfp4_scale_layout,
    nvfp4_tensor_scale,
    validate_nvfp4_tensor,
)
from .runtime import BoundedCache, load_kernel_symbol, runner_cache_limit
from .types import (
    AutotuneMode,
    CanonicalAutotuneMode,
    NVFP4Backend as BackendMode,
    NVFP4ScalingMode as ScalingMode,
)

if TYPE_CHECKING:
    from .kernels.mxfp8_bwd import MXFP8BwdConfig

WeightMode = Literal["dynamic", "prequantized"]
NVFP4_FRONTEND_REVISION = 6
DEFAULT_NVFP4_X_SCALE_REGION_ROWS = 5
DEFAULT_NVFP4_WEIGHT_SCALE_REGION_ROWS = 4


def _resolve_scale_region_rows(
    shared: int | None,
    x_rows: int | None,
    weight_rows: int | None,
) -> tuple[int, int]:
    """Resolve the public shared override or the asymmetric portable seed."""

    if shared is not None and shared <= 0:
        raise ValueError("scale_region_rows must be positive")
    resolved_x = (
        x_rows
        if x_rows is not None
        else shared
        if shared is not None
        else DEFAULT_NVFP4_X_SCALE_REGION_ROWS
    )
    resolved_weight = (
        weight_rows
        if weight_rows is not None
        else shared
        if shared is not None
        else DEFAULT_NVFP4_WEIGHT_SCALE_REGION_ROWS
    )
    if resolved_x <= 0 or resolved_weight <= 0:
        raise ValueError("X and weight scale-region rows must be positive")
    return int(resolved_x), int(resolved_weight)


def _effective_scale_region_rows(rows: int, requested: int) -> int:
    """Resolve a row-region cap, falling back to one region for ragged shapes."""

    if requested <= 0 or rows <= requested or rows % requested:
        return 0
    return requested


def compile_nvfp4_quant(*args, **kwargs):
    return load_kernel_symbol("nvfp4_quant", "compile_nvfp4_quant")(
        *args, **kwargs
    )


def compile_nvfp4_dual_quant(*args, **kwargs):
    return load_kernel_symbol("nvfp4_quant", "compile_nvfp4_dual_quant")(
        *args, **kwargs
    )


def compile_nvfp4_delayed_dual_quant(*args, **kwargs):
    return load_kernel_symbol(
        "nvfp4_quant", "compile_nvfp4_delayed_dual_quant"
    )(*args, **kwargs)


def compile_nvfp4_jit_region_dual_quant(*args, **kwargs):
    return load_kernel_symbol(
        "nvfp4_quant", "compile_nvfp4_jit_region_dual_quant"
    )(*args, **kwargs)


def compile_nvfp4_region_rescale(*args, **kwargs):
    return load_kernel_symbol(
        "nvfp4_gemm", "compile_nvfp4_region_rescale"
    )(*args, **kwargs)


def compile_nvfp4_region_expand(*args, **kwargs):
    return load_kernel_symbol(
        "nvfp4_gemm", "compile_nvfp4_region_expand"
    )(*args, **kwargs)


def compile_nvfp4_block_quant(*args, **kwargs):
    return load_kernel_symbol("nvfp4_quant", "compile_nvfp4_block_quant")(
        *args, **kwargs
    )


def compile_nvfp4_block_dual_quant(*args, **kwargs):
    return load_kernel_symbol("nvfp4_quant", "compile_nvfp4_block_dual_quant")(
        *args, **kwargs
    )


def compile_nvfp4_gemm(*args, **kwargs):
    return load_kernel_symbol("nvfp4_gemm", "compile_nvfp4_gemm")(
        *args, **kwargs
    )


def compile_nvfp4_block_gemm(*args, **kwargs):
    return load_kernel_symbol("nvfp4_gemm", "compile_nvfp4_block_gemm")(
        *args, **kwargs
    )


def compile_nvfp4_jit_region_gemm(*args, **kwargs):
    return load_kernel_symbol(
        "nvfp4_gemm", "compile_nvfp4_jit_region_gemm"
    )(*args, **kwargs)


@dataclass(frozen=True, slots=True)
class NVFP4ForwardConfig:
    """Dynamic quantization and GEMM schedules plus delayed-scale policy."""

    quant: NVFP4QuantConfig = DEFAULT_NVFP4_QUANT_CONFIG
    gemm: NVFP4GemmConfig = DEFAULT_NVFP4_GEMM_CONFIG
    quant_launches: str = "dual"
    policy: NVFP4ScaleConfig | None = None
    l2_fetch_granularity: int | None = None
    x_scale_region_rows: int = 0
    weight_scale_region_rows: int = 0
    tensor_scale_mode: str = "power2"
    region_amax_load_bits: int = 128
    region_amax_unroll: int = 1
    region_waves: int = 4
    region_order: str = "x_first"
    region_ownership: str = "warp"
    programmatic_dependent_launch: bool = False

    def rejection(self, problem: NVFP4Problem) -> str | None:
        return self.materialized_rejection(problem)

    def materialized_rejection(self, problem: NVFP4Problem) -> str | None:
        return NVFP4DynamicConfig(
            quant=self.quant,
            gemm=self.gemm,
            quant_launches=self.quant_launches,
            l2_fetch_granularity=self.l2_fetch_granularity,
            x_scale_region_rows=self.x_scale_region_rows,
            weight_scale_region_rows=self.weight_scale_region_rows,
            tensor_scale_mode=self.tensor_scale_mode,
            region_amax_load_bits=self.region_amax_load_bits,
            region_amax_unroll=self.region_amax_unroll,
            region_waves=self.region_waves,
            region_order=self.region_order,
            region_ownership=self.region_ownership,
            programmatic_dependent_launch=self.programmatic_dependent_launch,
        ).rejection(problem)

    @classmethod
    def from_materialized(cls, config: NVFP4DynamicConfig) -> "NVFP4ForwardConfig":
        return cls(
            quant=config.quant,
            gemm=config.gemm,
            quant_launches=config.quant_launches,
            l2_fetch_granularity=config.l2_fetch_granularity,
            x_scale_region_rows=config.x_scale_region_rows,
            weight_scale_region_rows=config.weight_scale_region_rows,
            tensor_scale_mode=config.tensor_scale_mode,
            region_amax_load_bits=config.region_amax_load_bits,
            region_amax_unroll=config.region_amax_unroll,
            region_waves=config.region_waves,
            region_order=config.region_order,
            region_ownership=config.region_ownership,
            programmatic_dependent_launch=config.programmatic_dependent_launch,
        )


DEFAULT_NVFP4_FORWARD_CONFIG = NVFP4ForwardConfig()
_FORWARD_CONFIGS: dict[str, NVFP4ForwardConfig] = {}
_CONFIG_LOCK = RLock()
_INFERENCE_CONFIG_SELECTIONS: dict[tuple[object, ...], str] = {}
_DYNAMIC_CONFIG_SELECTIONS: dict[tuple[object, ...], str] = {}
_AUTOTUNE_REQUESTS: dict[str, "_NVFP4AutotuneRequest"] = {}


@dataclass(frozen=True, slots=True)
class _NVFP4AutotuneRequest:
    mode: AutotuneMode
    policy: object | None
    cache_dir: str | None
    dynamic: NVFP4DynamicConfig | None = None
    weight_prequantized: NVFP4WeightPrequantConfig | None = None
    fully_prequantized: NVFP4FullyPrequantConfig | None = None


@torch.compiler.assume_constant_result
def _intern_autotune_request(
    mode: AutotuneMode,
    policy: object | None,
    cache_dir: Path | str | None,
    *,
    dynamic: NVFP4DynamicConfig | None = None,
    weight_prequantized: NVFP4WeightPrequantConfig | None = None,
    fully_prequantized: NVFP4FullyPrequantConfig | None = None,
) -> str:
    root = None if cache_dir is None else str(Path(cache_dir).expanduser())
    payload = {
        "mode": mode,
        "policy": None if policy is None else asdict(policy),
        "cache_dir": root,
        "dynamic": None if dynamic is None else asdict(dynamic),
        "weight_prequantized": (
            None if weight_prequantized is None else asdict(weight_prequantized)
        ),
        "fully_prequantized": (
            None if fully_prequantized is None else asdict(fully_prequantized)
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    key = (
        f"nvfp4-autotune:v{NVFP4_FRONTEND_REVISION}:"
        + hashlib.sha256(encoded.encode()).hexdigest()[:24]
    )
    with _CONFIG_LOCK:
        _AUTOTUNE_REQUESTS[key] = _NVFP4AutotuneRequest(
            mode=mode,
            policy=policy,
            cache_dir=root,
            dynamic=dynamic,
            weight_prequantized=weight_prequantized,
            fully_prequantized=fully_prequantized,
        )
    return key


def _nvfp4_autotune_mode(
    value: AutotuneMode | bool | None,
) -> CanonicalAutotuneMode:
    if isinstance(value, bool):
        return "coordinate" if value else "off"
    selected = (
        os.getenv("RTX_NVFP4_AUTOTUNE", os.getenv("RTX_AUTOTUNE", "cache"))
        if value is None
        else value
    )
    if selected == "online":
        selected = "coordinate"
    if selected not in ("off", "cache", "coordinate"):
        raise ValueError(
            "autotune must be off, cache, online, or coordinate; "
            f"got {selected!r}"
        )
    return selected


_DEFAULT_NVFP4_AUTOTUNE_REQUEST_KEY = _intern_autotune_request(
    "cache", None, None
)


@torch.compiler.assume_constant_result
def _intern_forward_config(config: NVFP4ForwardConfig) -> str:
    key = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    with _CONFIG_LOCK:
        _FORWARD_CONFIGS[key] = config
    return key


_DEFAULT_NVFP4_FORWARD_KEY = _intern_forward_config(DEFAULT_NVFP4_FORWARD_CONFIG)


@torch.compiler.assume_constant_result
def _functional_forward_config_key(
    policy: NVFP4ScaleConfig | None,
    x_region_rows: int,
    weight_region_rows: int,
) -> str:
    """Intern functional policy without constructing dataclasses in FX."""

    if policy is None and x_region_rows == 0 and weight_region_rows == 0:
        return _DEFAULT_NVFP4_FORWARD_KEY
    return _intern_forward_config(
        NVFP4ForwardConfig(
            policy=policy,
            x_scale_region_rows=x_region_rows,
            weight_scale_region_rows=weight_region_rows,
        )
    )


def _current_tensor_scale(tensor: torch.Tensor) -> torch.Tensor:
    """Return the exact TorchAO two-level decode scale without a host sync."""

    amax = torch.amax(torch.abs(tensor.detach().float()))
    scale = amax / (448.0 * 6.0)
    # A zero tensor has no useful dynamic range. One is a benign decode scale
    # and avoids 0/0 while every quantized value remains exactly zero.
    return torch.where(amax > 0.0, scale, torch.ones_like(scale)).reshape(1)



def _outer_tensor_scale(tensor: torch.Tensor, mode: str) -> torch.Tensor:
    """Return the scalar outer decode scale consumed by materialized quant."""

    if mode == "exact":
        return _current_tensor_scale(tensor)
    if mode == "power2":
        amax = torch.amax(torch.abs(tensor.detach().float()))
        target = amax / 2688.0
        safe_target = torch.clamp_min(target, torch.finfo(torch.float32).tiny)
        scale = torch.exp2(torch.ceil(torch.log2(safe_target)))
        return torch.where(amax > 0.0, scale, torch.ones_like(scale)).reshape(1)
    raise ValueError(f"unknown NVFP4 tensor scale mode {mode!r}")


def _delayed_amax_state(tensor: torch.Tensor, values: int) -> torch.Tensor:
    """Bootstrap delayed telemetry/history without a host synchronization."""

    amax = torch.amax(torch.abs(tensor.detach().float())).reshape(1)
    return amax.expand(values).clone()


@torch.compiler.assume_constant_result
def _delayed_history_values_from_key(forward_config_key: str) -> int:
    """Return the compact scalar history used by materialized delayed quant."""

    public = _resolve_forward_config(forward_config_key)
    policy = public.policy or DEFAULT_NVFP4_SCALE_CONFIG
    return int(policy.amax_history_len)


def _packed_fp4_view(tensor: torch.Tensor) -> torch.Tensor:
    dtype = getattr(torch, "float4_e2m1fn_x2", None)
    if dtype is None:
        raise RuntimeError("PyTorch must expose torch.float4_e2m1fn_x2")
    return tensor if tensor.dtype is dtype else tensor.view(dtype)


def _empty_nvfp4_scales(
    rows: int,
    k: int,
    config: NVFP4QuantConfig,
    device: torch.device,
) -> torch.Tensor:
    if config.scale_layout == "row_major":
        shape = (rows, ((k + 15) // 16))
    else:
        shape = (rows // 128, k // 128, 1024)
    return torch.empty(shape, dtype=torch.float8_e4m3fn, device=device)


@dataclass(slots=True)
class _DynamicRunner:
    quant_launches: str
    quant_x: object
    quant_w: object | None
    gemm: object
    qx: torch.Tensor
    qw: torch.Tensor
    sx: torch.Tensor
    sw: torch.Tensor
    qx_packed: torch.Tensor
    qw_packed: torch.Tensor
    l2_fetch_granularity: int | None = None
    quant_stream: torch.cuda.Stream | None = None

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        x_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        output_scale: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        from .fp8 import _ensure_l2_fetch_granularity

        _ensure_l2_fetch_granularity(self.l2_fetch_granularity)
        if self.quant_launches == "dual":
            self.quant_x(
                x, weight, self.qx, self.qw, self.sx, self.sw, x_scale, weight_scale
            )
        elif self.quant_launches == "concurrent":
            assert self.quant_w is not None and self.quant_stream is not None
            caller = torch.cuda.current_stream(x.device)
            self.quant_stream.wait_stream(caller)
            self.quant_x(x, self.qx, self.sx, x_scale)
            with torch.cuda.stream(self.quant_stream):
                self.quant_w(weight, self.qw, self.sw, weight_scale)
            caller.wait_stream(self.quant_stream)
        else:
            self.quant_x(x, self.qx, self.sx, x_scale)
            assert self.quant_w is not None
            self.quant_w(weight, self.qw, self.sw, weight_scale)
        self.gemm(
            self.qx_packed,
            self.qw_packed,
            self.sx,
            self.sw,
            out,
            output_scale,
        )


_DYNAMIC_RUNNERS: BoundedCache[tuple[object, ...], _DynamicRunner] = BoundedCache(
    runner_cache_limit("dynamic", 8, namespace="NVFP4")
)


@dataclass(slots=True)
class _DelayedDynamicRunner:
    """One-time delayed dual quantization followed by native NVFP4 GEMM."""

    quant: object
    gemm: object
    qx: torch.Tensor
    qw: torch.Tensor
    sx: torch.Tensor
    sw: torch.Tensor
    qx_packed: torch.Tensor
    qw_packed: torch.Tensor
    output_scale: torch.Tensor
    history_len: int
    l2_fetch_granularity: int | None = None

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        x_history: torch.Tensor,
        weight_history: torch.Tensor,
        next_x_history: torch.Tensor,
        next_weight_history: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        from .fp8 import _ensure_l2_fetch_granularity
        from .fp8_bwd import _zero_tensor_async

        _ensure_l2_fetch_granularity(self.l2_fetch_granularity)
        _zero_tensor_async(next_x_history)
        _zero_tensor_async(next_weight_history)
        self.quant(
            x,
            weight,
            self.qx,
            self.qw,
            self.sx,
            self.sw,
            x_history,
            weight_history,
            next_x_history,
            next_weight_history,
            self.output_scale,
        )
        self.gemm(
            self.qx_packed,
            self.qw_packed,
            self.sx,
            self.sw,
            out,
            self.output_scale,
        )


_DELAYED_DYNAMIC_RUNNERS: BoundedCache[
    tuple[object, ...], _DelayedDynamicRunner
] = BoundedCache(runner_cache_limit("delayed_dynamic", 8, namespace="NVFP4"))


@dataclass(slots=True)
class _JITRegionDynamicRunner:
    """Current row-region observer/quantizer followed by native NVFP4 GEMM."""

    quant: object
    gemm: object
    qx: torch.Tensor
    qw: torch.Tensor
    sx: torch.Tensor
    sw: torch.Tensor
    qx_packed: torch.Tensor
    qw_packed: torch.Tensor
    region_scales: torch.Tensor
    l2_fetch_granularity: int | None = None

    def __call__(
        self, x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor
    ) -> None:
        from .fp8 import _ensure_l2_fetch_granularity

        _ensure_l2_fetch_granularity(self.l2_fetch_granularity)
        self.quant(
            x,
            weight,
            self.qx,
            self.qw,
            self.sx,
            self.sw,
            self.region_scales,
        )
        self.gemm(
            self.qx_packed,
            self.qw_packed,
            self.sx,
            self.sw,
            out,
            self.region_scales,
        )


@dataclass(slots=True)
class _ExpandedFactorRegionGemm:
    expand: object
    gemm: object
    expanded: torch.Tensor

    def __call__(self, qx, qw, sx, sw, out, region_factors) -> None:
        self.expand(region_factors, self.expanded)
        self.gemm(qx, qw, sx, sw, out, self.expanded)


_JIT_REGION_DYNAMIC_RUNNERS: BoundedCache[
    tuple[object, ...], _JITRegionDynamicRunner
] = BoundedCache(runner_cache_limit("jit_region_dynamic", 8, namespace="NVFP4"))


@dataclass(slots=True)
class _BlockDynamicRunner:
    quant_launches: str
    quant_x: object
    quant_w: object | None
    gemm: object
    qx: torch.Tensor
    qw: torch.Tensor
    sx: torch.Tensor
    sw: torch.Tensor
    qx_packed: torch.Tensor
    qw_packed: torch.Tensor
    l2_fetch_granularity: int | None = None
    quant_stream: torch.cuda.Stream | None = None

    def __call__(
        self, x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor
    ) -> None:
        from .fp8 import _ensure_l2_fetch_granularity

        _ensure_l2_fetch_granularity(self.l2_fetch_granularity)
        if self.quant_launches == "dual":
            self.quant_x(x, weight, self.qx, self.qw, self.sx, self.sw)
        elif self.quant_launches == "concurrent":
            assert self.quant_w is not None and self.quant_stream is not None
            caller = torch.cuda.current_stream(x.device)
            self.quant_stream.wait_stream(caller)
            self.quant_x(x, self.qx, self.sx)
            with torch.cuda.stream(self.quant_stream):
                self.quant_w(weight, self.qw, self.sw)
            caller.wait_stream(self.quant_stream)
        else:
            self.quant_x(x, self.qx, self.sx)
            assert self.quant_w is not None
            self.quant_w(weight, self.qw, self.sw)
        self.gemm(
            self.qx_packed,
            self.qw_packed,
            self.sx,
            self.sw,
            out,
        )


_BLOCK_DYNAMIC_RUNNERS: BoundedCache[
    tuple[object, ...], _BlockDynamicRunner
] = BoundedCache(runner_cache_limit("block_dynamic", 8, namespace="NVFP4"))


@dataclass(slots=True)
class _DynamicXRunner:
    quant_x: object
    gemm: object
    qx: torch.Tensor
    sx: torch.Tensor
    qx_packed: torch.Tensor
    l2_fetch_granularity: int | None = None

    def __call__(
        self,
        x: torch.Tensor,
        weight_data: torch.Tensor,
        weight_scales: torch.Tensor,
        x_scale: torch.Tensor,
        output_scale: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        from .fp8 import _ensure_l2_fetch_granularity

        _ensure_l2_fetch_granularity(self.l2_fetch_granularity)
        self.quant_x(x, self.qx, self.sx, x_scale)
        self.gemm(
            self.qx_packed,
            weight_data,
            self.sx,
            weight_scales,
            out,
            output_scale,
        )


_DYNAMIC_X_RUNNERS: BoundedCache[tuple[object, ...], _DynamicXRunner] = BoundedCache(
    runner_cache_limit("dynamic_x", 8, namespace="NVFP4")
)


def _make_dynamic_runner(
    problem: NVFP4Problem,
    config: NVFP4ForwardConfig,
    device: torch.device,
) -> _DynamicRunner:
    storage_problem = NVFP4Problem(problem.m, problem.n, problem.storage_k)
    qx = torch.empty(
        (problem.m, problem.storage_k // 2), dtype=torch.uint8, device=device
    )
    qw = torch.empty(
        (problem.n, problem.storage_k // 2), dtype=torch.uint8, device=device
    )
    sx = _empty_nvfp4_scales(problem.m, problem.k, config.quant, device)
    sw = _empty_nvfp4_scales(problem.n, problem.k, config.quant, device)
    if config.quant_launches == "dual":
        quant_x = compile_nvfp4_dual_quant(
            problem.m, problem.n, problem.k, config.quant
        )
        quant_w = None
    else:
        quant_x = compile_nvfp4_quant(problem.m, problem.k, config.quant)
        quant_w = compile_nvfp4_quant(problem.n, problem.k, config.quant)
    quant_stream = (
        torch.cuda.Stream(device=device)
        if config.quant_launches == "concurrent"
        else None
    )
    return _DynamicRunner(
        config.quant_launches,
        quant_x,
        quant_w,
        compile_nvfp4_gemm(storage_problem, config.gemm),
        qx,
        qw,
        sx,
        sw,
        _packed_fp4_view(qx),
        _packed_fp4_view(qw),
        config.l2_fetch_granularity,
        quant_stream,
    )


def _make_delayed_dynamic_runner(
    problem: NVFP4Problem,
    config: NVFP4ForwardConfig,
    policy: NVFP4ScaleConfig,
    device: torch.device,
) -> _DelayedDynamicRunner:
    quant_config = config.quant
    storage_problem = NVFP4Problem(problem.m, problem.n, problem.storage_k)
    qx = torch.empty(
        (problem.m, problem.storage_k // 2), dtype=torch.uint8, device=device
    )
    qw = torch.empty(
        (problem.n, problem.storage_k // 2), dtype=torch.uint8, device=device
    )
    sx = _empty_nvfp4_scales(problem.m, problem.k, quant_config, device)
    sw = _empty_nvfp4_scales(problem.n, problem.k, quant_config, device)
    return _DelayedDynamicRunner(
        compile_nvfp4_delayed_dual_quant(
            problem.m,
            problem.n,
            problem.k,
            quant_config,
            policy.amax_history_len,
            policy.amax_history_algo,
            policy.tensor_scale_mode,
        ),
        compile_nvfp4_gemm(storage_problem, config.gemm),
        qx,
        qw,
        sx,
        sw,
        _packed_fp4_view(qx),
        _packed_fp4_view(qw),
        torch.empty(1, dtype=torch.float32, device=device),
        policy.amax_history_len,
        config.l2_fetch_granularity,
    )


def _make_jit_region_dynamic_runner(
    problem: NVFP4Problem,
    config: NVFP4ForwardConfig,
    device: torch.device,
) -> _JITRegionDynamicRunner:
    if config.x_scale_region_rows < 1 or config.weight_scale_region_rows < 1:
        raise ValueError("JIT row-region runner requires positive region sizes")
    storage_problem = NVFP4Problem(problem.m, problem.n, problem.storage_k)
    qx = torch.empty(
        (problem.m, problem.storage_k // 2), dtype=torch.uint8, device=device
    )
    qw = torch.empty(
        (problem.n, problem.storage_k // 2), dtype=torch.uint8, device=device
    )
    sx = _empty_nvfp4_scales(problem.m, problem.k, config.quant, device)
    sw = _empty_nvfp4_scales(problem.n, problem.k, config.quant, device)
    region_scale_count = (
        (problem.m + config.x_scale_region_rows - 1)
        // config.x_scale_region_rows
        + (problem.n + config.weight_scale_region_rows - 1)
        // config.weight_scale_region_rows
    )
    gemm = compile_nvfp4_jit_region_gemm(
        storage_problem,
        config.gemm,
        config.x_scale_region_rows,
        config.weight_scale_region_rows,
        config.programmatic_dependent_launch,
    )
    if config.gemm.regional_scale_epilogue == "expanded_factors":
        gemm = _ExpandedFactorRegionGemm(
            compile_nvfp4_region_expand(
                problem.m,
                problem.n,
                config.x_scale_region_rows,
                config.weight_scale_region_rows,
            ),
            gemm,
            torch.empty(
                problem.m + problem.n,
                dtype=torch.float32,
                device=device,
            ),
        )
    return _JITRegionDynamicRunner(
        compile_nvfp4_jit_region_dual_quant(
            problem.m,
            problem.n,
            problem.k,
            config.quant,
            config.x_scale_region_rows,
            config.weight_scale_region_rows,
            config.tensor_scale_mode,
            config.region_amax_load_bits,
            config.region_amax_unroll,
            config.region_waves,
            config.region_order,
            config.region_ownership,
            config.programmatic_dependent_launch,
        ),
        gemm,
        qx,
        qw,
        sx,
        sw,
        _packed_fp4_view(qx),
        _packed_fp4_view(qw),
        torch.empty(region_scale_count, dtype=torch.float32, device=device),
        config.l2_fetch_granularity,
    )


def _make_block_dynamic_runner(
    problem: NVFP4Problem,
    config: NVFP4ForwardConfig,
    device: torch.device,
) -> _BlockDynamicRunner:
    storage_problem = NVFP4Problem(problem.m, problem.n, problem.storage_k)
    qx = torch.empty(
        (problem.m, problem.storage_k // 2), dtype=torch.uint8, device=device
    )
    qw = torch.empty(
        (problem.n, problem.storage_k // 2), dtype=torch.uint8, device=device
    )
    sx = _empty_nvfp4_scales(problem.m, problem.k, config.quant, device)
    sw = _empty_nvfp4_scales(problem.n, problem.k, config.quant, device)
    if config.quant_launches == "dual":
        quant_x = compile_nvfp4_block_dual_quant(
            problem.m, problem.n, problem.k, config.quant
        )
        quant_w = None
    else:
        quant_x = compile_nvfp4_block_quant(problem.m, problem.k, config.quant)
        quant_w = compile_nvfp4_block_quant(problem.n, problem.k, config.quant)
    quant_stream = (
        torch.cuda.Stream(device=device)
        if config.quant_launches == "concurrent"
        else None
    )
    return _BlockDynamicRunner(
        config.quant_launches,
        quant_x,
        quant_w,
        compile_nvfp4_block_gemm(storage_problem, config.gemm),
        qx,
        qw,
        sx,
        sw,
        _packed_fp4_view(qx),
        _packed_fp4_view(qw),
        config.l2_fetch_granularity,
        quant_stream,
    )


def _make_dynamic_x_runner(
    problem: NVFP4Problem,
    config: NVFP4ForwardConfig,
    device: torch.device,
) -> _DynamicXRunner:
    storage_problem = NVFP4Problem(problem.m, problem.n, problem.storage_k)
    qx = torch.empty(
        (problem.m, problem.storage_k // 2), dtype=torch.uint8, device=device
    )
    sx = torch.empty(
        (problem.m, problem.storage_k // 16), dtype=torch.float8_e4m3fn, device=device
    )
    return _DynamicXRunner(
        compile_nvfp4_quant(problem.m, problem.k, config.quant),
        compile_nvfp4_gemm(storage_problem, config.gemm),
        qx,
        sx,
        _packed_fp4_view(qx),
        config.l2_fetch_granularity,
    )


def _resolve_forward_config(key: str) -> NVFP4ForwardConfig:
    try:
        return _FORWARD_CONFIGS[key]
    except KeyError as exc:
        raise RuntimeError("unknown NVFP4 forward configuration key") from exc


def _packed_inference_config_key(
    problem: NVFP4Problem,
    device: torch.device,
    *,
    fully_prequantized: bool,
    request_key: str,
) -> str:
    """Resolve an explicit, cached, or tuned state-specific configuration."""

    state = "fully_prequantized" if fully_prequantized else "weight_prequantized"
    request = _AUTOTUNE_REQUESTS[request_key]
    selection_key = (
        state,
        device.index,
        problem.m,
        problem.n,
        problem.k,
        request_key,
    )
    selected_key = _INFERENCE_CONFIG_SELECTIONS.get(selection_key)
    if selected_key is not None:
        return selected_key
    from .autotune.winners import load_runtime_winner, runtime_winner_key
    from .nvfp4_inference_autotune import (
        fully_prequant_config_from_dict,
        weight_prequant_config_from_dict,
    )

    explicit = (
        request.fully_prequantized
        if fully_prequantized
        else request.weight_prequantized
    )
    if fully_prequantized:
        family = "nvfp4_fully_prequant_fwd"
        variant = "x-row_major_w-row_major"
        selected = explicit
        if selected is None and request.mode != "off":
            selected = load_runtime_winner(
                runtime_winner_key(family, problem, device=device, variant=variant),
                fully_prequant_config_from_dict,
                root=request.cache_dir,
                rejection=lambda value: value.rejection(problem),
            )
        if selected is None and request.mode == "coordinate":
            from .nvfp4_inference_autotune import tune_nvfp4_inference_state

            selected = tune_nvfp4_inference_state(
                problem,
                state=state,
                device=device,
                cache_dir=request.cache_dir,
                policy=request.policy,
            )
        selected = selected or NVFP4FullyPrequantConfig()
        config = NVFP4ForwardConfig(
            gemm=selected.gemm,
            l2_fetch_granularity=selected.l2_fetch_granularity,
        )
    else:
        family = "nvfp4_weight_prequant_fwd"
        variant = "w-row_major"
        selected = explicit
        if selected is None and request.mode != "off":
            selected = load_runtime_winner(
                runtime_winner_key(family, problem, device=device, variant=variant),
                weight_prequant_config_from_dict,
                root=request.cache_dir,
                rejection=lambda value: value.rejection(problem),
            )
        if selected is None and request.mode == "coordinate":
            from .nvfp4_inference_autotune import tune_nvfp4_inference_state

            selected = tune_nvfp4_inference_state(
                problem,
                state=state,
                device=device,
                cache_dir=request.cache_dir,
                policy=request.policy,
            )
        selected = selected or NVFP4WeightPrequantConfig()
        config = NVFP4ForwardConfig(
            quant=selected.quant_x,
            gemm=selected.gemm,
            quant_launches="independent",
            l2_fetch_granularity=selected.l2_fetch_granularity,
        )
    selected_key = _intern_forward_config(config)
    _INFERENCE_CONFIG_SELECTIONS[selection_key] = selected_key
    return selected_key


@torch.compiler.assume_constant_result
def _packed_inference_config_key_from_dims(
    m: int,
    n: int,
    k: int,
    device: torch.device,
    fully_prequantized: bool,
    request_key: str,
) -> str:
    """Compiler-safe shape facade for the inference winner lookup."""

    return _packed_inference_config_key(
        NVFP4Problem(int(m), int(n), int(k)),
        device,
        fully_prequantized=fully_prequantized,
        request_key=request_key,
    )


def _materialized_dynamic_config_key(
    problem: NVFP4Problem,
    device: torch.device,
    request_key: str,
    family: str = "nvfp4_dynamic_fwd",
    x_scale_region_rows: int = 0,
    weight_scale_region_rows: int = 0,
) -> str:
    """Resolve a verified joint quantize+GEMM winner for dynamic operands."""

    request = _AUTOTUNE_REQUESTS[request_key]
    selection_key = (
        family,
        device.index,
        problem.m,
        problem.n,
        problem.k,
        request_key,
        x_scale_region_rows,
        weight_scale_region_rows,
    )
    selected_key = _DYNAMIC_CONFIG_SELECTIONS.get(selection_key)
    if selected_key is not None:
        return selected_key
    from .autotune.winners import load_runtime_winner, runtime_winner_key
    from .nvfp4_inference_autotune import (
        dynamic_config_from_dict,
        preferred_dynamic_config,
    )

    selected = request.dynamic
    if selected is None and request.mode != "off":
        def reject_runtime_dynamic(value: NVFP4DynamicConfig) -> str | None:
            if (
                family == "nvfp4_jit_row_region_fwd"
                and not value.jit_row_region
            ):
                return "JIT row-region winner has no region geometry"
            return value.rejection(problem)

        selected = load_runtime_winner(
            runtime_winner_key(family, problem, device=device),
            dynamic_config_from_dict,
            root=request.cache_dir,
            rejection=reject_runtime_dynamic,
        )
        if selected is None and family == "nvfp4_delayed_fwd":
            selected = load_runtime_winner(
                runtime_winner_key(
                    "nvfp4_dynamic_fwd", problem, device=device
                ),
                dynamic_config_from_dict,
                root=request.cache_dir,
                rejection=lambda value: value.rejection(problem),
            )
    if (
        selected is None
        and request.mode == "coordinate"
        and family in ("nvfp4_dynamic_fwd", "nvfp4_jit_row_region_fwd")
    ):
        from .nvfp4_inference_autotune import tune_nvfp4_inference_state

        selected = tune_nvfp4_inference_state(
            problem,
            state=(
                "jit_row_region"
                if family == "nvfp4_jit_row_region_fwd"
                else "dynamic"
            ),
            device=device,
            cache_dir=request.cache_dir,
            policy=request.policy,
        )
    if selected is None and family == "nvfp4_jit_row_region_fwd":
        from .nvfp4_inference_autotune import preferred_jit_row_region_config

        preferred = preferred_jit_row_region_config(problem)
        selected = replace(
            preferred,
            x_scale_region_rows=(
                x_scale_region_rows or preferred.x_scale_region_rows
            ),
            weight_scale_region_rows=(
                weight_scale_region_rows
                or preferred.weight_scale_region_rows
            ),
        )
    if family == "nvfp4_jit_row_region_fwd" and selected is not None:
        selected = replace(
            selected,
            x_scale_region_rows=(
                x_scale_region_rows or selected.x_scale_region_rows or 1
            ),
            weight_scale_region_rows=(
                weight_scale_region_rows
                or selected.weight_scale_region_rows
                or 1
            ),
            quant_launches="dual",
            gemm=replace(
                selected.gemm,
                epilogue="direct",
                epilogue_stages=1,
                store_vec=1,
            ),
        )
    config = NVFP4ForwardConfig.from_materialized(
        preferred_dynamic_config(problem) if selected is None else selected
    )
    selected_key = _intern_forward_config(config)
    _DYNAMIC_CONFIG_SELECTIONS[selection_key] = selected_key
    return selected_key


@torch.compiler.assume_constant_result
def _materialized_dynamic_config_key_from_dims(
    m: int,
    n: int,
    k: int,
    device: torch.device,
    request_key: str,
    family: str = "nvfp4_dynamic_fwd",
    x_scale_region_rows: int = 0,
    weight_scale_region_rows: int = 0,
) -> str:
    return _materialized_dynamic_config_key(
        NVFP4Problem(int(m), int(n), int(k)),
        device,
        request_key,
        family,
        x_scale_region_rows,
        weight_scale_region_rows,
    )


def _check_sm12x(device: torch.device) -> None:
    capability = torch.cuda.get_device_capability(device)
    if capability[0] != 12:
        raise RuntimeError(
            "native RTX NVFP4 kernels require an SM120/SM121 GPU; "
            f"got compute capability {capability}"
        )


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


def _launch_nvfp4_forward_materialized(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    output_scale: torch.Tensor,
    forward_config_key: str,
) -> torch.Tensor:
    """Quantize both operands, then launch the native prequantized GEMM."""

    _check_nvfp4_inputs(x, weight)
    _check_sm12x(x.device)
    x_c = x if x.is_contiguous() else x.contiguous()
    weight_c = weight if weight.is_contiguous() else weight.contiguous()
    for name, value in (
        ("x_scale", x_scale),
        ("weight_scale", weight_scale),
        ("output_scale", output_scale),
    ):
        if value.dtype is not torch.float32 or value.numel() != 1:
            raise TypeError(f"materialized NVFP4 {name} must be one FP32 value")
        if value.device != x.device:
            raise ValueError(f"materialized NVFP4 {name} must share the input device")
    problem = NVFP4Problem(
        int(x_c.shape[0]), int(weight_c.shape[0]), int(x_c.shape[1])
    )
    config = _resolve_forward_config(forward_config_key)
    rejection = config.materialized_rejection(problem)
    if rejection is not None:
        raise RuntimeError(f"materialized NVFP4 cannot run this problem: {rejection}")
    stream = torch.cuda.current_stream(x.device)
    cache_key = (
        x.device.index,
        stream.cuda_stream,
        problem.m,
        problem.n,
        problem.k,
        config,
    )
    runner = _DYNAMIC_RUNNERS.get(cache_key)
    if runner is None:
        runner = _make_dynamic_runner(problem, config, x.device)
        _DYNAMIC_RUNNERS[cache_key] = runner
    out = torch.empty((problem.m, problem.n), dtype=torch.bfloat16, device=x.device)
    runner(x_c, weight_c, x_scale, weight_scale, output_scale, out)
    out._base_inputs = (
        x_c,
        weight_c,
        x_scale,
        weight_scale,
        output_scale,
        runner,
    )
    return out


def _launch_nvfp4_forward_block(
    x: torch.Tensor,
    weight: torch.Tensor,
    forward_config_key: str,
) -> torch.Tensor:
    _check_nvfp4_inputs(x, weight)
    _check_sm12x(x.device)
    x_c = x if x.is_contiguous() else x.contiguous()
    weight_c = weight if weight.is_contiguous() else weight.contiguous()
    problem = NVFP4Problem(
        int(x_c.shape[0]), int(weight_c.shape[0]), int(x_c.shape[1])
    )
    config = _resolve_forward_config(forward_config_key)
    rejection = config.materialized_rejection(problem)
    if rejection is not None:
        raise RuntimeError(f"block-only NVFP4 cannot run this problem: {rejection}")
    stream = torch.cuda.current_stream(x.device)
    cache_key = (
        x.device.index,
        stream.cuda_stream,
        problem.m,
        problem.n,
        problem.k,
        config,
    )
    runner = _BLOCK_DYNAMIC_RUNNERS.get(cache_key)
    if runner is None:
        runner = _make_block_dynamic_runner(problem, config, x.device)
        _BLOCK_DYNAMIC_RUNNERS[cache_key] = runner
    out = torch.empty((problem.m, problem.n), dtype=torch.bfloat16, device=x.device)
    runner(x_c, weight_c, out)
    out._base_inputs = (x_c, weight_c, runner)
    return out


def _launch_nvfp4_forward_jit_row_region(
    x: torch.Tensor,
    weight: torch.Tensor,
    forward_config_key: str,
) -> torch.Tensor:
    """Compute current local scales, quantize, and GEMM without eager ops."""

    _check_nvfp4_inputs(x, weight)
    _check_sm12x(x.device)
    x_c = x if x.is_contiguous() else x.contiguous()
    weight_c = weight if weight.is_contiguous() else weight.contiguous()
    problem = NVFP4Problem(
        int(x_c.shape[0]), int(weight_c.shape[0]), int(x_c.shape[1])
    )
    config = _resolve_forward_config(forward_config_key)
    rejection = config.materialized_rejection(problem)
    if rejection is not None:
        raise RuntimeError(
            f"JIT row-region NVFP4 cannot run this problem: {rejection}"
        )
    if config.x_scale_region_rows < 1 or config.weight_scale_region_rows < 1:
        raise RuntimeError("JIT row-region NVFP4 requires positive region sizes")
    stream = torch.cuda.current_stream(x.device)
    cache_key = (
        x.device.index,
        stream.cuda_stream,
        problem.m,
        problem.n,
        problem.k,
        config,
    )
    runner = _JIT_REGION_DYNAMIC_RUNNERS.get(cache_key)
    if runner is None:
        runner = _make_jit_region_dynamic_runner(problem, config, x.device)
        _JIT_REGION_DYNAMIC_RUNNERS[cache_key] = runner
    out = torch.empty((problem.m, problem.n), dtype=torch.bfloat16, device=x.device)
    runner(x_c, weight_c, out)
    out._base_inputs = (x_c, weight_c, runner)
    return out


def _launch_nvfp4_forward_delayed(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_amax_state: torch.Tensor,
    weight_amax_state: torch.Tensor,
    forward_config_key: str,
    materialized_config_key: str = "",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    next_x_amax_state = torch.empty_like(x_amax_state)
    next_weight_amax_state = torch.empty_like(weight_amax_state)
    out = _launch_nvfp4_forward_delayed_out(
        x,
        weight,
        x_amax_state,
        weight_amax_state,
        next_x_amax_state,
        next_weight_amax_state,
        forward_config_key,
        materialized_config_key,
    )
    return out, next_x_amax_state, next_weight_amax_state


def _launch_nvfp4_forward_delayed_out(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_amax_state: torch.Tensor,
    weight_amax_state: torch.Tensor,
    next_x_amax_state: torch.Tensor,
    next_weight_amax_state: torch.Tensor,
    forward_config_key: str,
    materialized_config_key: str = "",
) -> torch.Tensor:
    _check_nvfp4_inputs(x, weight)
    _check_sm12x(x.device)
    for name, value in (
        ("x_amax_state", x_amax_state),
        ("weight_amax_state", weight_amax_state),
        ("next_x_amax_state", next_x_amax_state),
        ("next_weight_amax_state", next_weight_amax_state),
    ):
        if value.dtype is not torch.float32:
            raise TypeError(f"{name} must use FP32")
        if value.device != x.device:
            raise ValueError(f"{name} must share the input CUDA device")
    x_c = x if x.is_contiguous() else x.contiguous()
    weight_c = weight if weight.is_contiguous() else weight.contiguous()
    problem = NVFP4Problem(int(x_c.shape[0]), int(weight_c.shape[0]), int(x_c.shape[1]))
    if not materialized_config_key:
        raise ValueError("delayed NVFP4 requires a materialized configuration")
    config = _resolve_forward_config(materialized_config_key)
    rejection = config.materialized_rejection(problem)
    if rejection is not None:
        raise RuntimeError(f"materialized delayed NVFP4 cannot run: {rejection}")
    public = _resolve_forward_config(forward_config_key)
    policy = public.policy or DEFAULT_NVFP4_SCALE_CONFIG
    expected_values = int(policy.amax_history_len)
    for name, value in (
        ("x_amax_state", x_amax_state),
        ("weight_amax_state", weight_amax_state),
        ("next_x_amax_state", next_x_amax_state),
        ("next_weight_amax_state", next_weight_amax_state),
    ):
        if value.numel() != expected_values:
            raise ValueError(
                f"{name} has {value.numel()} values, expected {expected_values}"
            )
    if next_x_amax_state.data_ptr() == x_amax_state.data_ptr():
        raise ValueError("delayed x amax generations must not alias")
    if next_weight_amax_state.data_ptr() == weight_amax_state.data_ptr():
        raise ValueError("delayed weight amax generations must not alias")
    stream = torch.cuda.current_stream(x.device)
    cache_key = (
        x.device.index,
        stream.cuda_stream,
        problem.m,
        problem.n,
        problem.k,
        config,
        policy.amax_history_len,
        policy.amax_history_algo,
        policy.tensor_scale_mode,
    )
    runner = _DELAYED_DYNAMIC_RUNNERS.get(cache_key)
    if runner is None:
        runner = _make_delayed_dynamic_runner(problem, config, policy, x.device)
        _DELAYED_DYNAMIC_RUNNERS[cache_key] = runner
    out = torch.empty((problem.m, problem.n), dtype=torch.bfloat16, device=x.device)
    runner(
        x_c,
        weight_c,
        x_amax_state,
        weight_amax_state,
        next_x_amax_state,
        next_weight_amax_state,
        out,
    )
    out._base_inputs = (
        x_c,
        weight_c,
        x_amax_state,
        weight_amax_state,
        next_x_amax_state,
        next_weight_amax_state,
        runner,
    )
    return out


def quantize_nvfp4(
    tensor: torch.Tensor,
    *,
    tensor_scale: torch.Tensor | None = None,
    config: NVFP4QuantConfig | None = None,
) -> NVFP4Tensor:
    """Prequantize BF16 into TorchAO's canonical ``NVFP4Tensor``."""

    if tensor.ndim < 1 or tensor.dtype is not torch.bfloat16:
        raise TypeError("NVFP4 quantization requires a BF16 tensor")
    if tensor.device.type != "cuda":
        raise ValueError("NVFP4 quantization requires a CUDA tensor")
    source = tensor.reshape(-1, tensor.shape[-1])
    source = source if source.is_contiguous() else source.contiguous()
    rows, k = int(source.shape[0]), int(source.shape[1])
    selected = config or DEFAULT_NVFP4_QUANT_CONFIG
    rejection = selected.rejection(rows, k)
    if rejection is not None:
        raise RuntimeError(f"NVFP4 operand cannot be quantized: {rejection}")
    _check_sm12x(source.device)
    scale = _current_tensor_scale(source) if tensor_scale is None else tensor_scale
    if scale.dtype is not torch.float32 or scale.numel() != 1:
        raise TypeError("NVFP4 tensor_scale must be one FP32 value")
    if scale.device != source.device:
        raise ValueError("NVFP4 tensor_scale and source must share one device")
    storage_k = (k + 15) // 16 * 16
    qdata = torch.empty((rows, storage_k // 2), dtype=torch.uint8, device=source.device)
    scales = torch.empty(
        (rows, storage_k // 16), dtype=torch.float8_e4m3fn, device=source.device
    )
    scale_1d = scale.reshape(1)
    compile_nvfp4_quant(rows, k, selected)(source, qdata, scales, scale_1d)
    qdata._base_inputs = (source, scale)
    scales._base_inputs = (source, scale)
    return make_nvfp4_tensor(
        _packed_fp4_view(qdata),
        scales,
        scale.reshape(()),
        tuple(int(v) for v in tensor.shape),
    )


@torch.library.custom_op(
    "rtx::nvfp4_linear_materialized_fwd",
    mutates_args=(),
    device_types="cuda",
)
def _nvfp4_linear_materialized_fwd_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    output_scale: torch.Tensor,
    forward_config_key: str,
) -> torch.Tensor:
    return _launch_nvfp4_forward_materialized(
        x,
        weight,
        x_scale,
        weight_scale,
        output_scale,
        forward_config_key,
    )


@_nvfp4_linear_materialized_fwd_op.register_fake
def _nvfp4_linear_materialized_fwd_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    output_scale: torch.Tensor,
    forward_config_key: str,
) -> torch.Tensor:
    return torch.empty(
        (x.shape[0], weight.shape[0]), dtype=torch.bfloat16, device=x.device
    )


@torch.library.custom_op(
    "rtx::nvfp4_linear_block_fwd",
    mutates_args=(),
    device_types="cuda",
)
def _nvfp4_linear_block_fwd_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    forward_config_key: str,
) -> torch.Tensor:
    return _launch_nvfp4_forward_block(x, weight, forward_config_key)


@_nvfp4_linear_block_fwd_op.register_fake
def _nvfp4_linear_block_fwd_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    forward_config_key: str,
) -> torch.Tensor:
    return torch.empty(
        (x.shape[0], weight.shape[0]), dtype=torch.bfloat16, device=x.device
    )


@torch.library.custom_op(
    "rtx::nvfp4_linear_jit_row_region_fwd",
    mutates_args=(),
    device_types="cuda",
)
def _nvfp4_linear_jit_row_region_fwd_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    forward_config_key: str,
) -> torch.Tensor:
    return _launch_nvfp4_forward_jit_row_region(x, weight, forward_config_key)


@_nvfp4_linear_jit_row_region_fwd_op.register_fake
def _nvfp4_linear_jit_row_region_fwd_fake(
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
    x_scale: torch.Tensor,
    output_scale: torch.Tensor,
    n: int,
    k: int,
    weight_scale_layout: str,
    forward_config_key: str,
) -> torch.Tensor:
    if weight_scale_layout != "row_major":
        raise RuntimeError("native NVFP4 GEMM currently requires row-major scales")
    storage_k = (k + 15) // 16 * 16
    if x.ndim != 2 or tuple(weight_data.shape) != (n, storage_k // 2):
        raise ValueError("dynamic-X/prequant-W NVFP4 operand shape mismatch")
    _check_sm12x(x.device)
    x_c = x if x.is_contiguous() else x.contiguous()
    problem = NVFP4Problem(int(x_c.shape[0]), int(n), int(k))
    config = _resolve_forward_config(forward_config_key)
    rejection = config.materialized_rejection(problem)
    if rejection is not None:
        raise RuntimeError(f"NVFP4 configuration cannot run this problem: {rejection}")
    stream = torch.cuda.current_stream(x.device)
    cache_key = (
        x.device.index,
        stream.cuda_stream,
        problem.m,
        problem.n,
        problem.k,
        config,
    )
    runner = _DYNAMIC_X_RUNNERS.get(cache_key)
    if runner is None:
        runner = _make_dynamic_x_runner(problem, config, x.device)
        _DYNAMIC_X_RUNNERS[cache_key] = runner
    out = torch.empty((problem.m, problem.n), dtype=torch.bfloat16, device=x.device)
    runner(
        x_c,
        weight_data,
        weight_block_scales,
        x_scale,
        output_scale,
        out,
    )
    out._base_inputs = (
        x_c,
        weight_data,
        weight_block_scales,
        x_scale,
        output_scale,
        runner,
    )
    return out


@_nvfp4_linear_dynamic_x_prequant_w_op.register_fake
def _nvfp4_linear_dynamic_x_prequant_w_fake(
    x: torch.Tensor,
    weight_data: torch.Tensor,
    weight_block_scales: torch.Tensor,
    x_scale: torch.Tensor,
    output_scale: torch.Tensor,
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
    output_scale: torch.Tensor,
    m: int,
    n: int,
    k: int,
    x_scale_layout: str,
    weight_scale_layout: str,
    forward_config_key: str,
) -> torch.Tensor:
    if x_scale_layout != "row_major" or weight_scale_layout != "row_major":
        raise RuntimeError("native NVFP4 GEMM currently requires row-major scales")
    problem = NVFP4Problem(int(m), int(n), int(k))
    config = _resolve_forward_config(forward_config_key)
    rejection = config.gemm.rejection(problem)
    if rejection is not None:
        raise RuntimeError(f"NVFP4 GEMM cannot run this problem: {rejection}")
    _check_sm12x(x_data.device)
    from .fp8 import _ensure_l2_fetch_granularity

    _ensure_l2_fetch_granularity(config.l2_fetch_granularity)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x_data.device)
    compile_nvfp4_gemm(
        NVFP4Problem(problem.m, problem.n, int(x_data.shape[-1]) * 2),
        config.gemm,
    )(
        x_data,
        weight_data,
        x_block_scales,
        weight_block_scales,
        out,
        output_scale,
    )
    out._base_inputs = (
        x_data,
        weight_data,
        x_block_scales,
        weight_block_scales,
        output_scale,
    )
    return out


@_nvfp4_linear_prequantized_op.register_fake
def _nvfp4_linear_prequantized_fake(
    x_data: torch.Tensor,
    weight_data: torch.Tensor,
    x_block_scales: torch.Tensor,
    weight_block_scales: torch.Tensor,
    output_scale: torch.Tensor,
    m: int,
    n: int,
    k: int,
    x_scale_layout: str,
    weight_scale_layout: str,
    forward_config_key: str,
) -> torch.Tensor:
    return torch.empty((m, n), dtype=torch.bfloat16, device=x_data.device)


def _nvfp4_backward(ctx, grad_output: torch.Tensor):
    from .fp8_bwd import (
        _mxfp8_bwd_compiler_visible,
        _mxfp8_dw_compiler_visible,
        _mxfp8_dx_compiler_visible,
    )

    x, weight = ctx.saved_tensors
    need_x, need_weight = ctx.needs_input_grad[:2]
    grad_x = grad_weight = None
    if need_x and need_weight:
        grad_x, grad_weight = _mxfp8_bwd_compiler_visible(
            grad_output, x, weight, ctx.backward_config_key
        )
    elif need_x:
        grad_x = _mxfp8_dx_compiler_visible(
            grad_output, x, weight, ctx.backward_config_key
        )
    elif need_weight:
        grad_weight = _mxfp8_dw_compiler_visible(
            grad_output, x, weight, ctx.backward_config_key
        )
    return grad_x, grad_weight, None, None, None, None


@torch.library.custom_op(
    "rtx::nvfp4_linear_materialized_train",
    mutates_args=(),
    device_types="cuda",
)
def _nvfp4_linear_materialized_train_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    output_scale: torch.Tensor,
    forward_config_key: str,
    backward_config_key: str,
) -> torch.Tensor:
    return _launch_nvfp4_forward_materialized(
        x,
        weight,
        x_scale,
        weight_scale,
        output_scale,
        forward_config_key,
    )


@_nvfp4_linear_materialized_train_op.register_fake
def _nvfp4_linear_materialized_train_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    output_scale: torch.Tensor,
    forward_config_key: str,
    backward_config_key: str,
) -> torch.Tensor:
    return torch.empty(
        (x.shape[0], weight.shape[0]), dtype=torch.bfloat16, device=x.device
    )


def _setup_nvfp4_materialized_context(ctx, inputs, output) -> None:
    x, weight, *_scales, _forward_config_key, backward_config_key = inputs
    ctx.save_for_backward(x, weight)
    ctx.backward_config_key = backward_config_key


def _nvfp4_materialized_backward(ctx, grad_output: torch.Tensor):
    gradients = _nvfp4_backward(ctx, grad_output)
    return gradients[:2] + (None, None, None, None, None)


torch.library.register_autograd(
    "rtx::nvfp4_linear_materialized_train",
    _nvfp4_materialized_backward,
    setup_context=_setup_nvfp4_materialized_context,
)


@torch.library.custom_op(
    "rtx::nvfp4_linear_block_train",
    mutates_args=(),
    device_types="cuda",
)
def _nvfp4_linear_block_train_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    forward_config_key: str,
    backward_config_key: str,
) -> torch.Tensor:
    return _launch_nvfp4_forward_block(x, weight, forward_config_key)


@_nvfp4_linear_block_train_op.register_fake
def _nvfp4_linear_block_train_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    forward_config_key: str,
    backward_config_key: str,
) -> torch.Tensor:
    return torch.empty(
        (x.shape[0], weight.shape[0]), dtype=torch.bfloat16, device=x.device
    )


def _setup_nvfp4_block_context(ctx, inputs, output) -> None:
    x, weight, _forward_config_key, backward_config_key = inputs
    ctx.save_for_backward(x, weight)
    ctx.backward_config_key = backward_config_key


def _nvfp4_block_backward(ctx, grad_output: torch.Tensor):
    gradients = _nvfp4_backward(ctx, grad_output)
    return gradients[:2] + (None, None)


torch.library.register_autograd(
    "rtx::nvfp4_linear_block_train",
    _nvfp4_block_backward,
    setup_context=_setup_nvfp4_block_context,
)


@torch.library.custom_op(
    "rtx::nvfp4_linear_jit_row_region_train",
    mutates_args=(),
    device_types="cuda",
)
def _nvfp4_linear_jit_row_region_train_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    forward_config_key: str,
    backward_config_key: str,
) -> torch.Tensor:
    return _launch_nvfp4_forward_jit_row_region(x, weight, forward_config_key)


@_nvfp4_linear_jit_row_region_train_op.register_fake
def _nvfp4_linear_jit_row_region_train_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    forward_config_key: str,
    backward_config_key: str,
) -> torch.Tensor:
    return torch.empty(
        (x.shape[0], weight.shape[0]), dtype=torch.bfloat16, device=x.device
    )


torch.library.register_autograd(
    "rtx::nvfp4_linear_jit_row_region_train",
    _nvfp4_block_backward,
    setup_context=_setup_nvfp4_block_context,
)


@torch.library.custom_op(
    "rtx::nvfp4_linear_train_delayed",
    mutates_args=("next_x_amax_state", "next_weight_amax_state"),
    device_types="cuda",
)
def _nvfp4_linear_train_delayed_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_amax_state: torch.Tensor,
    weight_amax_state: torch.Tensor,
    next_x_amax_state: torch.Tensor,
    next_weight_amax_state: torch.Tensor,
    forward_config_key: str,
    materialized_config_key: str,
    backward_config_key: str,
) -> torch.Tensor:
    return _launch_nvfp4_forward_delayed_out(
        x,
        weight,
        x_amax_state,
        weight_amax_state,
        next_x_amax_state,
        next_weight_amax_state,
        forward_config_key,
        materialized_config_key,
    )


@_nvfp4_linear_train_delayed_op.register_fake
def _nvfp4_linear_train_delayed_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_amax_state: torch.Tensor,
    weight_amax_state: torch.Tensor,
    next_x_amax_state: torch.Tensor,
    next_weight_amax_state: torch.Tensor,
    forward_config_key: str,
    materialized_config_key: str,
    backward_config_key: str,
) -> torch.Tensor:
    return torch.empty(
        (x.shape[0], weight.shape[0]), dtype=torch.bfloat16, device=x.device
    )


class _InductorNVFP4DelayedLauncher:
    """Shape-bound delayed launcher used directly by generated wrappers."""

    def __init__(
        self, forward_config_key: str, materialized_config_key: str
    ) -> None:
        self.forward_config_key = forward_config_key
        self.materialized_config_key = materialized_config_key
        self.runners: BoundedCache[tuple[object, ...], object] = (
            BoundedCache(runner_cache_limit("inductor_delayed", 8, namespace="NVFP4"))
        )

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        x_amax_state: torch.Tensor,
        weight_amax_state: torch.Tensor,
        next_x_amax_state: torch.Tensor,
        next_weight_amax_state: torch.Tensor,
        *,
        out: torch.Tensor,
    ) -> None:
        problem = NVFP4Problem(
            int(x.shape[0]), int(weight.shape[0]), int(x.shape[1])
        )
        stream_id = int(torch._C._cuda_getCurrentRawStream(x.device.index))
        if not self.materialized_config_key:
            raise ValueError("delayed NVFP4 requires a materialized configuration")
        config = _resolve_forward_config(self.materialized_config_key)
        public = _resolve_forward_config(self.forward_config_key)
        policy = public.policy or DEFAULT_NVFP4_SCALE_CONFIG
        key = (
            "materialized",
            x.device.index,
            stream_id,
            problem.m,
            problem.n,
            problem.k,
            config,
            policy.amax_history_len,
            policy.amax_history_algo,
            policy.tensor_scale_mode,
        )
        runner = self.runners.get(key)
        if runner is None:
            runner = _make_delayed_dynamic_runner(
                problem, config, policy, x.device
            )
            self.runners[key] = runner
        runner(
            x,
            weight,
            x_amax_state,
            weight_amax_state,
            next_x_amax_state,
            next_weight_amax_state,
            out,
        )


class _InductorNVFP4DelayedRegistry(dict[tuple[str, str], object]):
    def __missing__(self, config_key: tuple[str, str]) -> object:
        with _CONFIG_LOCK:
            launcher = self.get(config_key)
            if launcher is None:
                launcher = _InductorNVFP4DelayedLauncher(*config_key)
                self[config_key] = launcher
        return launcher


_INDUCTOR_DELAYED_LAUNCHERS = _InductorNVFP4DelayedRegistry()


class _InductorNVFP4MaterializedLauncher:
    """Direct quantize-both plus GEMM launcher used by generated wrappers."""

    def __init__(self, forward_config_key: str) -> None:
        self.forward_config_key = forward_config_key
        self.runners: BoundedCache[tuple[object, ...], _DynamicRunner] = (
            BoundedCache(
                runner_cache_limit("inductor_materialized", 8, namespace="NVFP4")
            )
        )

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        x_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        output_scale: torch.Tensor,
        *,
        out: torch.Tensor,
    ) -> None:
        problem = NVFP4Problem(
            int(x.shape[0]), int(weight.shape[0]), int(x.shape[1])
        )
        stream_id = int(torch._C._cuda_getCurrentRawStream(x.device.index))
        config = _resolve_forward_config(self.forward_config_key)
        rejection = config.materialized_rejection(problem)
        if rejection is not None:
            raise RuntimeError(f"materialized NVFP4 cannot run: {rejection}")
        key = (
            x.device.index,
            stream_id,
            problem.m,
            problem.n,
            problem.k,
            config,
        )
        runner = self.runners.get(key)
        if runner is None:
            runner = _make_dynamic_runner(problem, config, x.device)
            self.runners[key] = runner
        runner(x, weight, x_scale, weight_scale, output_scale, out)


class _InductorNVFP4MaterializedRegistry(dict[str, object]):
    def __missing__(self, config_key: str) -> object:
        with _CONFIG_LOCK:
            launcher = self.get(config_key)
            if launcher is None:
                launcher = _InductorNVFP4MaterializedLauncher(config_key)
                self[config_key] = launcher
        return launcher


_INDUCTOR_MATERIALIZED_LAUNCHERS = _InductorNVFP4MaterializedRegistry()


class _InductorNVFP4BlockLauncher:
    """Two-input block-only launcher matching the MXFP8 prequant hot ABI."""

    def __init__(self, forward_config_key: str) -> None:
        self.forward_config_key = forward_config_key
        self.runners: BoundedCache[tuple[object, ...], _BlockDynamicRunner] = (
            BoundedCache(runner_cache_limit("inductor_block", 8, namespace="NVFP4"))
        )

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
            int(weight.shape[0]),
            int(x.shape[1]),
        )
        runner = self.runners.get(key)
        if runner is None:
            problem = NVFP4Problem(key[2], key[3], key[4])
            config = _resolve_forward_config(self.forward_config_key)
            rejection = config.materialized_rejection(problem)
            if rejection is not None:
                raise RuntimeError(f"block-only NVFP4 cannot run: {rejection}")
            runner = _make_block_dynamic_runner(problem, config, x.device)
            self.runners[key] = runner
        runner(x, weight, out)


class _InductorNVFP4BlockRegistry(dict[str, object]):
    def __missing__(self, config_key: str) -> object:
        with _CONFIG_LOCK:
            launcher = self.get(config_key)
            if launcher is None:
                launcher = _InductorNVFP4BlockLauncher(config_key)
                self[config_key] = launcher
        return launcher


_INDUCTOR_BLOCK_LAUNCHERS = _InductorNVFP4BlockRegistry()


class _InductorNVFP4JITRegionLauncher:
    """Two-input current-region launcher used directly by generated wrappers."""

    def __init__(self, forward_config_key: str) -> None:
        self.forward_config_key = forward_config_key
        self.runners: BoundedCache[
            tuple[object, ...], _JITRegionDynamicRunner
        ] = BoundedCache(
            runner_cache_limit("inductor_jit_region", 8, namespace="NVFP4")
        )

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        *,
        out: torch.Tensor,
    ) -> None:
        problem = NVFP4Problem(
            int(x.shape[0]), int(weight.shape[0]), int(x.shape[1])
        )
        stream_id = int(torch._C._cuda_getCurrentRawStream(x.device.index))
        config = _resolve_forward_config(self.forward_config_key)
        rejection = config.materialized_rejection(problem)
        if rejection is not None:
            raise RuntimeError(f"JIT row-region NVFP4 cannot run: {rejection}")
        key = (
            x.device.index,
            stream_id,
            problem.m,
            problem.n,
            problem.k,
            config,
        )
        runner = self.runners.get(key)
        if runner is None:
            runner = _make_jit_region_dynamic_runner(problem, config, x.device)
            self.runners[key] = runner
        runner(x, weight, out)


class _InductorNVFP4JITRegionRegistry(dict[str, object]):
    def __missing__(self, config_key: str) -> object:
        with _CONFIG_LOCK:
            launcher = self.get(config_key)
            if launcher is None:
                launcher = _InductorNVFP4JITRegionLauncher(config_key)
                self[config_key] = launcher
        return launcher


_INDUCTOR_JIT_REGION_LAUNCHERS = _InductorNVFP4JITRegionRegistry()


class _InductorNVFP4DynamicXLauncher:
    """Launch-only dynamic-X/prequant-W path used by generated wrappers."""

    def __init__(self, forward_config_key: str) -> None:
        self.forward_config_key = forward_config_key
        self.runners: BoundedCache[tuple[object, ...], _DynamicXRunner] = (
            BoundedCache(runner_cache_limit("inductor_dynamic_x", 8, namespace="NVFP4"))
        )

    def __call__(
        self,
        x: torch.Tensor,
        weight_data: torch.Tensor,
        weight_block_scales: torch.Tensor,
        x_scale: torch.Tensor,
        output_scale: torch.Tensor,
        *,
        out: torch.Tensor,
    ) -> None:
        problem = NVFP4Problem(
            int(x.shape[0]), int(weight_data.shape[0]), int(x.shape[1])
        )
        stream_id = int(torch._C._cuda_getCurrentRawStream(x.device.index))
        config = _resolve_forward_config(self.forward_config_key)
        rejection = config.materialized_rejection(problem)
        if rejection is not None:
            raise RuntimeError(
                f"NVFP4 configuration cannot run this problem: {rejection}"
            )
        key = (
            x.device.index,
            stream_id,
            problem.m,
            problem.n,
            problem.k,
            config,
        )
        runner = self.runners.get(key)
        if runner is None:
            runner = _make_dynamic_x_runner(problem, config, x.device)
            self.runners[key] = runner
        runner(
            x,
            weight_data,
            weight_block_scales,
            x_scale,
            output_scale,
            out,
        )


class _InductorNVFP4DynamicXRegistry(dict[str, object]):
    def __missing__(self, config_key: str) -> object:
        with _CONFIG_LOCK:
            launcher = self.get(config_key)
            if launcher is None:
                launcher = _InductorNVFP4DynamicXLauncher(config_key)
                self[config_key] = launcher
        return launcher


_INDUCTOR_DYNAMIC_X_LAUNCHERS = _InductorNVFP4DynamicXRegistry()


class _InductorNVFP4PrequantLauncher:
    """Launch-only fully prequantized path used by generated wrappers."""

    def __init__(self, forward_config_key: str) -> None:
        self.forward_config_key = forward_config_key
        self.runners: BoundedCache[tuple[object, ...], object] = BoundedCache(
            runner_cache_limit("inductor_prequant", 8, namespace="NVFP4")
        )

    def __call__(
        self,
        x_data: torch.Tensor,
        weight_data: torch.Tensor,
        x_block_scales: torch.Tensor,
        weight_block_scales: torch.Tensor,
        output_scale: torch.Tensor,
        *,
        out: torch.Tensor,
    ) -> None:
        problem = NVFP4Problem(
            int(x_data.shape[0]),
            int(weight_data.shape[0]),
            int(x_data.shape[1]) * 2,
        )
        stream_id = int(torch._C._cuda_getCurrentRawStream(x_data.device.index))
        config = _resolve_forward_config(self.forward_config_key)
        rejection = config.gemm.rejection(problem)
        if rejection is not None:
            raise RuntimeError(f"NVFP4 GEMM cannot run this problem: {rejection}")
        key = (
            x_data.device.index,
            stream_id,
            problem.m,
            problem.n,
            problem.k,
            config.gemm,
        )
        runner = self.runners.get(key)
        if runner is None:
            runner = compile_nvfp4_gemm(problem, config.gemm)
            self.runners[key] = runner
        from .fp8 import _ensure_l2_fetch_granularity

        _ensure_l2_fetch_granularity(config.l2_fetch_granularity)
        runner(
            x_data,
            weight_data,
            x_block_scales,
            weight_block_scales,
            out,
            output_scale,
        )


class _InductorNVFP4PrequantRegistry(dict[str, object]):
    def __missing__(self, config_key: str) -> object:
        with _CONFIG_LOCK:
            launcher = self.get(config_key)
            if launcher is None:
                launcher = _InductorNVFP4PrequantLauncher(config_key)
                self[config_key] = launcher
        return launcher


_INDUCTOR_PREQUANT_LAUNCHERS = _InductorNVFP4PrequantRegistry()


def _register_delayed_inductor_lowering() -> None:
    from torch._inductor import ir
    from torch._inductor.lowering import register_lowering

    torch._rtx_nvfp4_delayed_launchers = _INDUCTOR_DELAYED_LAUNCHERS
    torch._rtx_nvfp4_materialized_launchers = _INDUCTOR_MATERIALIZED_LAUNCHERS
    torch._rtx_nvfp4_block_launchers = _INDUCTOR_BLOCK_LAUNCHERS
    torch._rtx_nvfp4_jit_region_launchers = _INDUCTOR_JIT_REGION_LAUNCHERS
    torch._rtx_nvfp4_dynamic_x_launchers = _INDUCTOR_DYNAMIC_X_LAUNCHERS
    torch._rtx_nvfp4_prequant_launchers = _INDUCTOR_PREQUANT_LAUNCHERS

    def lower_materialized_common(
        x, weight, x_scale, weight_scale, output_scale, config_key
    ):
        inputs = [
            ir.ExternKernel.require_contiguous(ir.ExternKernel.realize_input(value))
            for value in (x, weight, x_scale, weight_scale, output_scale)
        ]
        m = x.get_size()[0]
        n = weight.get_size()[0]
        name = f"torch._rtx_nvfp4_materialized_launchers[{config_key!r}]"
        return ir.TensorBox.create(
            ir.ExternKernelOut(
                layout=ir.FixedLayout(
                    device=x.get_device(),
                    dtype=torch.bfloat16,
                    size=[m, n],
                    stride=[n, 1],
                ),
                inputs=inputs,
                python_kernel_name=name,
            )
        )

    @register_lowering(
        torch.ops.rtx.nvfp4_linear_materialized_fwd.default,
        type_promotion_kind=None,
    )
    def lower_materialized_fwd(
        x, weight, x_scale, weight_scale, output_scale, config_key
    ):
        return lower_materialized_common(
            x, weight, x_scale, weight_scale, output_scale, config_key
        )

    @register_lowering(
        torch.ops.rtx.nvfp4_linear_materialized_train.default,
        type_promotion_kind=None,
    )
    def lower_materialized_train(
        x,
        weight,
        x_scale,
        weight_scale,
        output_scale,
        config_key,
        backward_config_key,
    ):
        return lower_materialized_common(
            x, weight, x_scale, weight_scale, output_scale, config_key
        )

    def lower_block_common(x, weight, config_key):
        inputs = [
            ir.ExternKernel.require_contiguous(ir.ExternKernel.realize_input(value))
            for value in (x, weight)
        ]
        m = x.get_size()[0]
        n = weight.get_size()[0]
        name = f"torch._rtx_nvfp4_block_launchers[{config_key!r}]"
        return ir.TensorBox.create(
            ir.ExternKernelOut(
                layout=ir.FixedLayout(
                    device=x.get_device(),
                    dtype=torch.bfloat16,
                    size=[m, n],
                    stride=[n, 1],
                ),
                inputs=inputs,
                python_kernel_name=name,
            )
        )

    @register_lowering(
        torch.ops.rtx.nvfp4_linear_block_fwd.default,
        type_promotion_kind=None,
    )
    def lower_block_fwd(x, weight, config_key):
        return lower_block_common(x, weight, config_key)

    @register_lowering(
        torch.ops.rtx.nvfp4_linear_block_train.default,
        type_promotion_kind=None,
    )
    def lower_block_train(x, weight, config_key, backward_config_key):
        return lower_block_common(x, weight, config_key)

    def lower_jit_region_common(x, weight, config_key):
        inputs = [
            ir.ExternKernel.require_contiguous(
                ir.ExternKernel.realize_input(value)
            )
            for value in (x, weight)
        ]
        m = x.get_size()[0]
        n = weight.get_size()[0]
        name = f"torch._rtx_nvfp4_jit_region_launchers[{config_key!r}]"
        return ir.TensorBox.create(
            ir.ExternKernelOut(
                layout=ir.FixedLayout(
                    device=x.get_device(),
                    dtype=torch.bfloat16,
                    size=[m, n],
                    stride=[n, 1],
                ),
                inputs=inputs,
                python_kernel_name=name,
            )
        )

    @register_lowering(
        torch.ops.rtx.nvfp4_linear_jit_row_region_fwd.default,
        type_promotion_kind=None,
    )
    def lower_jit_region_fwd(x, weight, config_key):
        return lower_jit_region_common(x, weight, config_key)

    @register_lowering(
        torch.ops.rtx.nvfp4_linear_jit_row_region_train.default,
        type_promotion_kind=None,
    )
    def lower_jit_region_train(
        x, weight, config_key, backward_config_key
    ):
        return lower_jit_region_common(x, weight, config_key)

    @register_lowering(
        torch.ops.rtx.nvfp4_linear_dynamic_x_prequant_w.default,
        type_promotion_kind=None,
    )
    def lower_dynamic_x(
        x,
        weight_data,
        weight_block_scales,
        x_scale,
        output_scale,
        n,
        k,
        weight_scale_layout,
        config_key,
    ):
        inputs = [
            ir.ExternKernel.require_contiguous(ir.ExternKernel.realize_input(value))
            for value in (
                x,
                weight_data,
                weight_block_scales,
                x_scale,
                output_scale,
            )
        ]
        m = x.get_size()[0]
        name = f"torch._rtx_nvfp4_dynamic_x_launchers[{config_key!r}]"
        return ir.TensorBox.create(
            ir.ExternKernelOut(
                layout=ir.FixedLayout(
                    device=x.get_device(),
                    dtype=torch.bfloat16,
                    size=[m, n],
                    stride=[n, 1],
                ),
                inputs=inputs,
                python_kernel_name=name,
            )
        )

    @register_lowering(
        torch.ops.rtx.nvfp4_linear_prequantized.default,
        type_promotion_kind=None,
    )
    def lower_prequantized(
        x_data,
        weight_data,
        x_block_scales,
        weight_block_scales,
        output_scale,
        m,
        n,
        k,
        x_scale_layout,
        weight_scale_layout,
        config_key,
    ):
        inputs = [
            ir.ExternKernel.require_contiguous(ir.ExternKernel.realize_input(value))
            for value in (
                x_data,
                weight_data,
                x_block_scales,
                weight_block_scales,
                output_scale,
            )
        ]
        name = f"torch._rtx_nvfp4_prequant_launchers[{config_key!r}]"
        return ir.TensorBox.create(
            ir.ExternKernelOut(
                layout=ir.FixedLayout(
                    device=x_data.get_device(),
                    dtype=torch.bfloat16,
                    size=[m, n],
                    stride=[n, 1],
                ),
                inputs=inputs,
                python_kernel_name=name,
            )
        )

    @register_lowering(
        torch.ops.rtx.nvfp4_linear_train_delayed.default,
        type_promotion_kind=None,
    )
    def lower_delayed(
        x,
        weight,
        x_amax_state,
        weight_amax_state,
        next_x_amax_state,
        next_weight_amax_state,
        forward_config_key,
        materialized_config_key,
        backward_config_key,
    ):
        inputs = [
            ir.ExternKernel.require_contiguous(ir.ExternKernel.realize_input(value))
            for value in (
                x,
                weight,
                x_amax_state,
                weight_amax_state,
                next_x_amax_state,
                next_weight_amax_state,
            )
        ]
        m = x.get_size()[0]
        n = weight.get_size()[0]
        name = (
            "torch._rtx_nvfp4_delayed_launchers"
            f"[{(forward_config_key, materialized_config_key)!r}]"
        )
        return ir.TensorBox.create(
            ir.ExternKernelOut(
                layout=ir.FixedLayout(
                    device=x.get_device(),
                    dtype=torch.bfloat16,
                    size=[m, n],
                    stride=[n, 1],
                ),
                inputs=inputs,
                python_kernel_name=name,
            )
        )


_register_delayed_inductor_lowering()


def _nvfp4_delayed_backward(
    ctx,
    grad_output: torch.Tensor,
):
    from .fp8_bwd import (
        _mxfp8_bwd_compiler_visible,
        _mxfp8_dw_compiler_visible,
        _mxfp8_dx_compiler_visible,
    )

    x, weight = ctx.saved_tensors
    need_x, need_weight = ctx.needs_input_grad[:2]
    grad_x = grad_weight = None
    if need_x and need_weight:
        grad_x, grad_weight = _mxfp8_bwd_compiler_visible(
            grad_output, x, weight, ctx.backward_config_key
        )
    elif need_x:
        grad_x = _mxfp8_dx_compiler_visible(
            grad_output, x, weight, ctx.backward_config_key
        )
    elif need_weight:
        grad_weight = _mxfp8_dw_compiler_visible(
            grad_output, x, weight, ctx.backward_config_key
        )
    return grad_x, grad_weight, None, None, None, None, None, None, None


class _NVFP4DelayedAutograd(torch.autograd.Function):
    """Autograd shell around the state-mutating delayed forward operator.

    ``torch.library`` intentionally rejects registered backward formulas on
    mutating schemas.  The shell keeps the device operation explicit for
    Dynamo/functionalization while AOTAutograd can still capture the MXFP8
    backward as part of the compiled training graph.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        x_amax_state: torch.Tensor,
        weight_amax_state: torch.Tensor,
        next_x_amax_state: torch.Tensor,
        next_weight_amax_state: torch.Tensor,
        forward_config_key: str,
        materialized_config_key: str,
        backward_config_key: str,
    ) -> torch.Tensor:
        ctx.save_for_backward(x, weight)
        ctx.backward_config_key = backward_config_key
        return _nvfp4_linear_train_delayed_op(
            x,
            weight,
            x_amax_state,
            weight_amax_state,
            next_x_amax_state,
            next_weight_amax_state,
            forward_config_key,
            materialized_config_key,
            backward_config_key,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return _nvfp4_delayed_backward(ctx, grad_output)


def nvfp4_linear(
    x: torch.Tensor | NVFP4Tensor,
    weight: torch.Tensor | NVFP4Tensor,
    *,
    scale_config: NVFP4ScaleConfig | None = None,
    backward_config: "MXFP8BwdConfig | None" = None,
    scaling: Literal["current", "jit_row_region", "block"] | None = None,
    scale_region_rows: int | None = None,
    x_scale_region_rows: int | None = None,
    weight_scale_region_rows: int | None = None,
    backend: BackendMode = "auto",
    autotune: AutotuneMode | bool | None = None,
    tuning_policy: object | None = None,
    autotune_cache_dir: Path | str | None = None,
    dynamic_config: NVFP4DynamicConfig | None = None,
    weight_prequant_config: NVFP4WeightPrequantConfig | None = None,
    fully_prequant_config: NVFP4FullyPrequantConfig | None = None,
) -> torch.Tensor:
    """Apply NVFP4 forward and MXFP8 backward to BF16 or packed operands."""

    if scaling is None:
        scaling = "current" if isinstance(weight, NVFP4Tensor) else "jit_row_region"
    if scaling not in ("current", "jit_row_region", "block"):
        raise ValueError(
            "functional NVFP4 scaling must be current, jit_row_region, or block"
        )
    x_region_rows, weight_region_rows = _resolve_scale_region_rows(
        scale_region_rows, x_scale_region_rows, weight_scale_region_rows
    )
    region_geometry_explicit = any(
        value is not None
        for value in (
            scale_region_rows,
            x_scale_region_rows,
            weight_scale_region_rows,
        )
    )
    if backend not in ("auto", "materialized"):
        raise ValueError("NVFP4 backend must be auto or materialized")
    mode = _nvfp4_autotune_mode(autotune)
    request_key = _intern_autotune_request(
        mode,
        tuning_policy,
        autotune_cache_dir,
        dynamic=dynamic_config,
        weight_prequantized=weight_prequant_config,
        fully_prequantized=fully_prequant_config,
    )

    if isinstance(weight, NVFP4Tensor):
        validate_nvfp4_tensor(weight)
        n, k = nvfp4_matrix_shape(weight)
        weight_layout = nvfp4_scale_layout(weight)
        if isinstance(x, NVFP4Tensor):
            validate_nvfp4_tensor(x)
            m, x_k = nvfp4_matrix_shape(x)
            if x_k != k or x.device != weight.device:
                raise ValueError("packed NVFP4 X/W shape or device mismatch")
            x_tensor_scale = nvfp4_tensor_scale(x).reshape(1)
            weight_tensor_scale = nvfp4_tensor_scale(weight).reshape(1)
            output_scale = x_tensor_scale * weight_tensor_scale
            out = _nvfp4_linear_prequantized_op(
                _packed_fp4_view(x.qdata),
                _packed_fp4_view(weight.qdata),
                x.scale,
                weight.scale,
                output_scale,
                m,
                n,
                k,
                nvfp4_scale_layout(x),
                weight_layout,
                _packed_inference_config_key_from_dims(
                    m,
                    n,
                    k,
                    weight.device,
                    True,
                    request_key,
                ),
            )
            logical_shape = getattr(x, "_rtx_logical_shape", tuple(x.shape))
            return out.reshape(*logical_shape[:-1], n)
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
        if scaling == "jit_row_region":
            raise NotImplementedError(
                "JIT row-region scaling currently requires dynamic BF16 X and W; "
                "the prequantized-W epilogue still consumes one output scale"
            )
        x_scale = (
            x_2d.new_ones(1, dtype=torch.float32)
            if scaling == "block"
            else _current_tensor_scale(x_2d)
        )
        weight_scale = nvfp4_tensor_scale(weight).reshape(1)
        output_scale = x_scale * weight_scale
        out = _nvfp4_linear_dynamic_x_prequant_w_op(
            x_2d,
            _packed_fp4_view(weight.qdata),
            weight.scale,
            x_scale,
            output_scale,
            n,
            k,
            weight_layout,
            _packed_inference_config_key_from_dims(
                int(x_2d.shape[0]),
                n,
                k,
                weight.device,
                False,
                request_key,
            ),
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
    forward_key = _functional_forward_config_key(
        scale_config,
        x_region_rows if scaling == "jit_row_region" else 0,
        weight_region_rows if scaling == "jit_row_region" else 0,
    )
    tensor_scale_mode = (
        "power2" if scale_config is None else scale_config.tensor_scale_mode
    )
    from .fp8 import _autotune_mode, _backward_config_key
    from .fp8_bwd import _intern_bwd_config

    backward_key = (
        _intern_bwd_config(backward_config)
        if backward_config is not None
        else _backward_config_key(_autotune_mode(None), None, None)
    )
    return _nvfp4_dynamic_linear_with_keys(
        x,
        weight,
        forward_key=forward_key,
        backward_key=backward_key,
        tensor_scale_mode=tensor_scale_mode,
        x_scale_region_rows=(x_region_rows if scaling == "jit_row_region" else 0),
        weight_scale_region_rows=(
            weight_region_rows if scaling == "jit_row_region" else 0
        ),
        region_geometry_explicit=region_geometry_explicit,
        jit_row_region=scaling == "jit_row_region",
        block_only=scaling == "block",
        backend=backend,
        materialized_request_key=request_key,
    )


def _nvfp4_dynamic_linear_with_keys(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    forward_key: str,
    backward_key: str,
    tensor_scale_mode: str,
    x_scale_region_rows: int = 0,
    weight_scale_region_rows: int = 0,
    region_geometry_explicit: bool = True,
    jit_row_region: bool = False,
    block_only: bool = False,
    backend: BackendMode = "auto",
    materialized_request_key: str = _DEFAULT_NVFP4_AUTOTUNE_REQUEST_KEY,
) -> torch.Tensor:
    """Compiler-visible current scaling around launch-only NVFP4 operators."""

    leading_shape = x.shape[:-1]
    x_2d = x.reshape(-1, x.shape[-1])
    _check_nvfp4_inputs(x_2d, weight)
    selected_backend = "materialized" if backend == "auto" else backend
    # These reductions and pointwise expressions intentionally remain in FX.
    # Inductor owns their fusion and schedules only the final CuTe launch as an
    # external kernel; none of this eager tensor work is hidden in registration.
    if selected_backend == "materialized":
        if jit_row_region:
            materialized_key = _materialized_dynamic_config_key_from_dims(
                int(x_2d.shape[0]),
                int(weight.shape[0]),
                int(x_2d.shape[1]),
                x_2d.device,
                materialized_request_key,
                "nvfp4_jit_row_region_fwd",
                x_scale_region_rows if region_geometry_explicit else 0,
                weight_scale_region_rows if region_geometry_explicit else 0,
            )
            out = (
                _nvfp4_linear_jit_row_region_train_op(
                    x_2d,
                    weight,
                    materialized_key,
                    backward_key,
                )
                if torch.is_grad_enabled()
                and (x_2d.requires_grad or weight.requires_grad)
                else _nvfp4_linear_jit_row_region_fwd_op(
                    x_2d, weight, materialized_key
                )
            )
            return out.reshape(*leading_shape, weight.shape[0])
        materialized_key = _materialized_dynamic_config_key_from_dims(
            int(x_2d.shape[0]),
            int(weight.shape[0]),
            int(x_2d.shape[1]),
            x_2d.device,
            materialized_request_key,
        )
        if block_only:
            out = (
                _nvfp4_linear_block_train_op(
                    x_2d,
                    weight,
                    materialized_key,
                    backward_key,
                )
                if torch.is_grad_enabled()
                and (x_2d.requires_grad or weight.requires_grad)
                else _nvfp4_linear_block_fwd_op(
                    x_2d, weight, materialized_key
                )
            )
            return out.reshape(*leading_shape, weight.shape[0])
        x_scale = _outer_tensor_scale(x_2d, tensor_scale_mode)
        weight_scale = _outer_tensor_scale(weight, tensor_scale_mode)
        output_scale = x_scale * weight_scale
        if torch.is_grad_enabled() and (x_2d.requires_grad or weight.requires_grad):
            out = _nvfp4_linear_materialized_train_op(
                x_2d,
                weight,
                x_scale,
                weight_scale,
                output_scale,
                materialized_key,
                backward_key,
            )
        else:
            out = _nvfp4_linear_materialized_fwd_op(
                x_2d,
                weight,
                x_scale,
                weight_scale,
                output_scale,
                materialized_key,
            )
        return out.reshape(*leading_shape, weight.shape[0])
    raise AssertionError(f"unreachable NVFP4 backend {selected_backend!r}")


class NVFP4Linear(nn.Module):
    """Drop-in no-bias linear with NVFP4 forward and MXFP8 backward.

    Dynamic BF16 operands use current JIT row-region scaling by default.  The
    portable geometry groups five activation rows independently from four
    weight rows; installed autotuning winners may select another geometry for
    a particular device and problem shape.

    Args:
        in_features: Size of each input sample's last dimension.
        out_features: Size of each output sample's last dimension.
        bias: Must be ``False``; NVFP4Linear intentionally has no bias path.
        device: Initial parameter device.
        dtype: Master-weight dtype. Only ``torch.bfloat16`` is supported.
        scale_config: Optional outer-scale and delayed-history policy. Leave unset for the
            portable default.
        backward_config: Optional MXFP8 backward configuration shared by dX
            and dW. Leave unset to use the runtime cache/default.
        scaling: Outer-scale policy. ``None`` selects ``"jit_row_region"`` for
            dynamic BF16 X/W and ``"current"`` for a packed weight. Explicit
            choices are ``"jit_row_region"`` (current
            local amax), ``"delayed"`` (history-based tensor scale),
            ``"current"`` (current tensorwide amax), and ``"block"`` (no FP32
            outer scaling).
        scale_region_rows: Optional symmetric JIT-region override. Setting N
            applies N rows to both operands unless an operand-specific value is
            supplied. ``None`` permits the asymmetric 5-by-4 default.
        x_scale_region_rows: Optional activation-region row count. Overrides
            ``scale_region_rows`` for X only.
        weight_scale_region_rows: Optional weight-region row count. Overrides
            ``scale_region_rows`` for W only.
        backend: ``"auto"`` or ``"materialized"`` selects the production
            materialized quantize/GEMM pipeline.
        packed_weight: Optional prequantized :class:`NVFP4Tensor`. Providing it
            creates an inference-only module without a BF16 Parameter.
        autotune: ``None`` follows the environment default; ``"off"`` uses
            explicit/portable configs, ``"cache"`` loads installed winners,
            and ``"coordinate"`` may tune a missing shape. Booleans retain
            their documented compatibility mapping.
        tuning_policy: Optional autotuner policy object used for online tuning.
        autotune_cache_dir: Optional runtime-winner/cache directory.
        dynamic_config: Optional fixed schedule for dynamic BF16 X/W.
        weight_prequant_config: Optional fixed schedule for dynamic-X,
            prequantized-W inference.
        fully_prequant_config: Optional fixed schedule when both X and W are
            prequantized.

    The FP32 outer-scale policy affects only the NVFP4 forward. Training keeps
    BF16 master weights and uses MXFP8 backward kernels. Delayed scaling owns
    mutable amax history; JIT-regional, current, and block modes are stateless.
    See https://github.com/achiya-deri-work/rtx/blob/main/docs/nvfp4_linear.md
    for policy selection and examples.
    """

    __constants__ = [
        "in_features",
        "out_features",
        "weight_mode",
        "scaling",
        "_forward_config_key",
        "_backward_config_key",
        "_tensor_scale_mode",
        "scale_region_rows",
        "x_scale_region_rows",
        "weight_scale_region_rows",
        "_region_geometry_explicit",
        "backend",
        "_materialized_request_key",
    ]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        scale_config: NVFP4ScaleConfig | None = None,
        backward_config: "MXFP8BwdConfig | None" = None,
        scaling: ScalingMode | None = None,
        scale_region_rows: int | None = None,
        x_scale_region_rows: int | None = None,
        weight_scale_region_rows: int | None = None,
        backend: BackendMode = "auto",
        packed_weight: NVFP4Tensor | None = None,
        autotune: AutotuneMode | bool | None = None,
        tuning_policy: object | None = None,
        autotune_cache_dir: Path | str | None = None,
        dynamic_config: NVFP4DynamicConfig | None = None,
        weight_prequant_config: NVFP4WeightPrequantConfig | None = None,
        fully_prequant_config: NVFP4FullyPrequantConfig | None = None,
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
        if scaling is None:
            scaling = "current" if packed_weight is not None else "jit_row_region"
        if scaling not in (
            "delayed", "current", "jit_row_region", "block"
        ):
            raise ValueError(
                "NVFP4 scaling must be delayed, current, jit_row_region, or block"
            )
        if packed_weight is not None and scaling not in ("current", "block"):
            raise ValueError(
                "a prequantized NVFP4 weight supports current or block scaling; "
                f"got {scaling!r}"
            )
        if backend not in ("auto", "materialized"):
            raise ValueError("NVFP4 backend must be auto or materialized")
        self.scale_config = scale_config
        self.backward_config = backward_config
        self.scaling = scaling
        self.scale_region_rows = scale_region_rows
        self._region_geometry_explicit = any(
            value is not None
            for value in (
                scale_region_rows,
                x_scale_region_rows,
                weight_scale_region_rows,
            )
        )
        (
            self.x_scale_region_rows,
            self.weight_scale_region_rows,
        ) = _resolve_scale_region_rows(
            scale_region_rows, x_scale_region_rows, weight_scale_region_rows
        )
        self.backend = backend
        self.autotune = autotune
        self.tuning_policy = tuning_policy
        self.autotune_cache_dir = autotune_cache_dir
        self.dynamic_config = dynamic_config
        self.weight_prequant_config = weight_prequant_config
        self.fully_prequant_config = fully_prequant_config
        mode = _nvfp4_autotune_mode(autotune)
        self._materialized_request_key = _intern_autotune_request(
            mode,
            tuning_policy,
            autotune_cache_dir,
            dynamic=dynamic_config,
            weight_prequantized=weight_prequant_config,
            fully_prequantized=fully_prequant_config,
        )
        self._forward_config_key = _intern_forward_config(
            NVFP4ForwardConfig(
                policy=scale_config,
                x_scale_region_rows=(
                    self.x_scale_region_rows if scaling == "jit_row_region" else 0
                ),
                weight_scale_region_rows=(
                    self.weight_scale_region_rows
                    if scaling == "jit_row_region"
                    else 0
                ),
            )
        )
        self._tensor_scale_mode = (
            "power2" if scale_config is None else scale_config.tensor_scale_mode
        )
        self.register_buffer(
            "_block_scale_pack",
            torch.tensor(
                (1.0, 1.0, 1.0 / 6.0),
                dtype=torch.float32,
                device=device,
            ),
            persistent=False,
        )
        from .fp8 import _backward_config_key
        from .fp8_bwd import _intern_bwd_config

        self._backward_config_key = (
            _intern_bwd_config(backward_config)
            if backward_config is not None
            else _backward_config_key(mode, tuning_policy, autotune_cache_dir)
        )
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
            self.register_buffer(
                "_x_amax_state",
                torch.empty(0, dtype=torch.float32, device=device),
                persistent=False,
            )
            self.register_buffer(
                "_weight_amax_state",
                torch.empty(0, dtype=torch.float32, device=device),
                persistent=False,
            )
            self.register_buffer(
                "_next_x_amax_state",
                torch.empty(0, dtype=torch.float32, device=device),
                persistent=False,
            )
            self.register_buffer(
                "_next_weight_amax_state",
                torch.empty(0, dtype=torch.float32, device=device),
                persistent=False,
            )
            self._delayed_initialized = False
            self._delayed_problem: tuple[object, ...] | None = None
        else:
            validate_nvfp4_tensor(packed_weight)
            if nvfp4_matrix_shape(packed_weight) != (
                out_features,
                in_features,
            ):
                raise ValueError(
                    f"packed NVFP4 weight must have shape "
                    f"{(out_features, in_features)}"
                )
            if nvfp4_orientation(packed_weight) != "row_major":
                raise ValueError("packed linear weights must be row-major")
            if device is not None:
                packed_weight = packed_weight.to(device)
            packed_layout = nvfp4_scale_layout(packed_weight)
            self.register_parameter("weight", None)
            self.register_buffer("weight_data", packed_weight.qdata)
            self.register_buffer("weight_block_scales", packed_weight.scale)
            self.register_buffer(
                "weight_tensor_scale", nvfp4_tensor_scale(packed_weight)
            )
            self.register_buffer(
                "weight_packing_meta",
                torch.tensor(
                    [
                        PACKED_OPERAND_SCHEMA_VERSION,
                        SCALE_LAYOUT_CODES[packed_layout],
                        self.in_features,
                        int(packed_weight.qdata.shape[-1]) * 2,
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
            reject_packed_dtype_conversion(args, kwargs, format_name="NVFP4")
        return super().to(*args, **kwargs)

    def train(self, mode: bool = True):
        if self.weight_mode == "prequantized" and mode:
            raise RuntimeError(
                "a prequantized NVFP4Linear is inference-only; keep a dynamic "
                "BF16-master module for training"
            )
        entering_training = mode and not self.training
        result = super().train(mode)
        if self.weight_mode == "dynamic" and entering_training:
            self.reset_delayed_scales()
        return result

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        if self.weight_mode == "dynamic":
            # Delayed telemetry is deliberately absent from state_dict. A
            # restored master weight must bootstrap fresh amax state.
            self.reset_delayed_scales()

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

    @torch.no_grad()
    def reset_delayed_scales(self) -> None:
        """Force the next training call to bootstrap from current amax."""

        if self.weight_mode != "dynamic":
            return
        self._x_amax_state = torch.empty(
            0,
            dtype=torch.float32,
            device=self._x_amax_state.device,
        )
        self._weight_amax_state = torch.empty_like(self._x_amax_state)
        self._next_x_amax_state = torch.empty_like(self._x_amax_state)
        self._next_weight_amax_state = torch.empty_like(self._x_amax_state)
        self._delayed_initialized = False
        self._delayed_problem = None

    @property
    def packed_weight(self) -> NVFP4Tensor | None:
        if self.weight_mode != "prequantized":
            return None
        return make_nvfp4_tensor(
            self.weight_data,
            self.weight_block_scales,
            self.weight_tensor_scale,
            (self.out_features, self.in_features),
            self._weight_scale_layout,
        )

    @classmethod
    def from_float(cls, module: nn.Linear, **kwargs) -> "NVFP4Linear":
        if module.bias is not None:
            raise NotImplementedError("NVFP4Linear.from_float requires bias=False")
        if module.weight.dtype is not torch.bfloat16:
            raise TypeError("NVFP4Linear.from_float requires a BF16 weight")
        requested_scaling = kwargs.get("scaling")
        packed = quantize_nvfp4(
            module.weight.detach(),
            tensor_scale=(
                torch.ones((), dtype=torch.float32, device=module.weight.device)
                if requested_scaling == "block"
                else None
            ),
        )
        return cls(
            module.in_features,
            module.out_features,
            bias=False,
            device=module.weight.device,
            packed_weight=packed,
            **kwargs,
        )

    def to_quantized_weight(
        self,
        *,
        scaling: Literal["current", "block"] | None = None,
    ) -> "NVFP4Linear":
        """Return an inference-only copy with a persistent packed weight.

        JIT-regional and delayed scaling require a dynamic BF16 weight. Their
        packed-weight conversion therefore defaults to current activation
        scaling. Block-only modules preserve block scaling. Pass ``scaling``
        explicitly to select either supported packed-weight policy.
        """

        if self.weight_mode == "prequantized":
            return self
        assert self.weight is not None
        packed_scaling = (
            scaling
            if scaling is not None
            else "block"
            if self.scaling == "block"
            else "current"
        )
        packed = quantize_nvfp4(
            self.weight.detach(),
            tensor_scale=(
                torch.ones((), dtype=torch.float32, device=self.weight.device)
                if packed_scaling == "block"
                else None
            ),
        )
        return type(self)(
            self.in_features,
            self.out_features,
            bias=False,
            device=self.weight.device,
            backward_config=self.backward_config,
            scale_config=self.scale_config,
            scaling=packed_scaling,
            backend=self.backend,
            packed_weight=packed,
            autotune=self.autotune,
            tuning_policy=self.tuning_policy,
            autotune_cache_dir=self.autotune_cache_dir,
            dynamic_config=self.dynamic_config,
            weight_prequant_config=self.weight_prequant_config,
            fully_prequant_config=self.fully_prequant_config,
        )

    def forward(self, x: torch.Tensor | NVFP4Tensor) -> torch.Tensor:
        if self.weight_mode == "prequantized":
            if self.scaling == "jit_row_region":
                raise NotImplementedError(
                    "JIT row-region scaling currently requires a dynamic BF16 weight"
                )
            if isinstance(x, NVFP4Tensor):
                validate_nvfp4_tensor(x)
                m, k = nvfp4_matrix_shape(x)
                if k != self.in_features or x.device != self.weight_data.device:
                    raise ValueError("packed NVFP4 X/W shape or device mismatch")
                x_scale = nvfp4_tensor_scale(x).reshape(1)
                weight_scale = self.weight_tensor_scale.reshape(1)
                output_scale = x_scale * weight_scale
                out = _nvfp4_linear_prequantized_op(
                    _packed_fp4_view(x.qdata),
                    _packed_fp4_view(self.weight_data),
                    x.scale,
                    self.weight_block_scales,
                    output_scale,
                    m,
                    self.out_features,
                    self.in_features,
                    nvfp4_scale_layout(x),
                    self._weight_scale_layout,
                    _packed_inference_config_key_from_dims(
                        m,
                        self.out_features,
                        self.in_features,
                        self.weight_data.device,
                        True,
                        self._materialized_request_key,
                    ),
                )
                logical_shape = getattr(
                    x, "_rtx_logical_shape", tuple(x.shape)
                )
                return out.reshape(*logical_shape[:-1], self.out_features)
            if x.ndim < 1 or x.shape[-1] != self.in_features:
                raise ValueError(
                    f"expected activation [..., {self.in_features}], got {x.shape}"
                )
            if x.device != self.weight_data.device:
                raise ValueError("dynamic X and packed W must share a CUDA device")
            if x.dtype is not torch.bfloat16:
                raise TypeError(
                    f"dynamic NVFP4 activation must be BF16, got {x.dtype}"
                )
            if torch.is_grad_enabled() and x.requires_grad:
                raise RuntimeError("prequantized NVFP4 weights are inference-only")
            leading_shape = x.shape[:-1]
            x_2d = x.reshape(-1, self.in_features)
            if self.scaling == "block":
                x_scale = self._block_scale_pack[:1]
                output_scale = self.weight_tensor_scale.reshape(1)
            else:
                x_scale = _current_tensor_scale(x_2d)
                output_scale = x_scale * self.weight_tensor_scale.reshape(1)
            out = _nvfp4_linear_dynamic_x_prequant_w_op(
                x_2d,
                _packed_fp4_view(self.weight_data),
                self.weight_block_scales,
                x_scale,
                output_scale,
                self.out_features,
                self.in_features,
                self._weight_scale_layout,
                _packed_inference_config_key_from_dims(
                    int(x_2d.shape[0]),
                    self.out_features,
                    self.in_features,
                    self.weight_data.device,
                    False,
                    self._materialized_request_key,
                ),
            )
            return out.reshape(*leading_shape, self.out_features)
        if isinstance(x, NVFP4Tensor):
            raise TypeError(
                "a prequantized activation requires a prequantized module weight"
            )
        assert self.weight is not None
        if self.scaling == "delayed":
            leading_shape = x.shape[:-1]
            x_2d = x.reshape(-1, x.shape[-1])
            _check_nvfp4_inputs(x_2d, self.weight)
            materialized_key = _materialized_dynamic_config_key_from_dims(
                int(x_2d.shape[0]),
                int(self.weight.shape[0]),
                int(x_2d.shape[1]),
                x_2d.device,
                self._materialized_request_key,
                "nvfp4_delayed_fwd",
            )
            delayed_problem = (
                x_2d.device.type,
                x_2d.device.index,
                torch.cuda.current_stream(x_2d.device).cuda_stream,
                int(x_2d.shape[0]),
                int(self.weight.shape[0]),
                int(x_2d.shape[1]),
                materialized_key,
            )
            if (
                not self._delayed_initialized
                or self._delayed_problem != delayed_problem
            ):
                with torch.no_grad():
                    state_values = _delayed_history_values_from_key(
                        self._forward_config_key
                    )
                    self._x_amax_state = _delayed_amax_state(
                        x_2d, state_values
                    )
                    self._weight_amax_state = _delayed_amax_state(
                        self.weight, state_values
                    )
                    self._next_x_amax_state = torch.empty_like(
                        self._x_amax_state
                    )
                    self._next_weight_amax_state = torch.empty_like(
                        self._weight_amax_state
                    )
                self._delayed_initialized = True
                self._delayed_problem = delayed_problem
            needs_grad = torch.is_grad_enabled() and (
                x_2d.requires_grad or self.weight.requires_grad
            )
            if needs_grad:
                out = _NVFP4DelayedAutograd.apply(
                    x_2d,
                    self.weight,
                    self._x_amax_state,
                    self._weight_amax_state,
                    self._next_x_amax_state,
                    self._next_weight_amax_state,
                    self._forward_config_key,
                    materialized_key,
                    self._backward_config_key,
                )
            else:
                # Delayed scaling is a forward numerical/performance policy,
                # not an autograd policy.  Evaluation, no_grad, and inference
                # must consume the same prior history and publish the same
                # successor state as training; only the autograd shell is
                # conditional on whether a backward can be requested.
                out = _nvfp4_linear_train_delayed_op(
                    x_2d,
                    self.weight,
                    self._x_amax_state,
                    self._weight_amax_state,
                    self._next_x_amax_state,
                    self._next_weight_amax_state,
                    self._forward_config_key,
                    materialized_key,
                    self._backward_config_key,
                )
            self._x_amax_state, self._next_x_amax_state = (
                self._next_x_amax_state,
                self._x_amax_state,
            )
            self._weight_amax_state, self._next_weight_amax_state = (
                self._next_weight_amax_state,
                self._weight_amax_state,
            )
            return out.reshape(*leading_shape, self.out_features)
        return _nvfp4_dynamic_linear_with_keys(
            x,
            self.weight,
            forward_key=self._forward_config_key,
            backward_key=self._backward_config_key,
            tensor_scale_mode=self._tensor_scale_mode,
            x_scale_region_rows=(
                self.x_scale_region_rows
                if self.scaling == "jit_row_region"
                else 0
            ),
            weight_scale_region_rows=(
                self.weight_scale_region_rows
                if self.scaling == "jit_row_region"
                else 0
            ),
            region_geometry_explicit=self._region_geometry_explicit,
            jit_row_region=self.scaling == "jit_row_region",
            block_only=self.scaling == "block",
            fixed_scale_pack=(
                self._block_scale_pack
                if self.scaling in ("block", "jit_row_region")
                else None
            ),
            backend=self.backend,
            materialized_request_key=self._materialized_request_key,
        )

    def explain(self, x: torch.Tensor | NVFP4Tensor) -> "LinearExecutionDecision":
        """Describe routing for ``x`` without compiling, benchmarking, or tuning."""

        from .selection import LinearExecutionDecision

        if isinstance(x, NVFP4Tensor):
            if self.weight_mode != "prequantized":
                raise TypeError(
                    "a prequantized activation requires a prequantized module weight"
                )
            m, k = nvfp4_matrix_shape(x)
            state = "fully_prequantized"
            family = "nvfp4_fully_prequant_fwd"
            config = self.fully_prequant_config
        else:
            if x.ndim < 1 or x.shape[-1] != self.in_features:
                raise ValueError(
                    f"expected activation [..., {self.in_features}], got {x.shape}"
                )
            m, k = int(x.numel() // x.shape[-1]), int(x.shape[-1])
            if self.weight_mode == "prequantized":
                state = "weight_prequantized"
                family = "nvfp4_weight_prequant_fwd"
                config = self.weight_prequant_config
            else:
                state = "dynamic"
                family = (
                    "nvfp4_jit_row_region_fwd"
                    if self.scaling == "jit_row_region"
                    else "nvfp4_delayed_fwd"
                    if self.scaling == "delayed"
                    else "nvfp4_dynamic_fwd"
                )
                config = self.dynamic_config
        backend = "materialized"
        mode = _nvfp4_autotune_mode(self.autotune)
        explicit = config is not None or self.scale_config is not None
        source = (
            "explicit"
            if explicit
            else "portable"
            if mode == "off"
            else "deferred_runtime"
        )
        return LinearExecutionDecision(
            format="nvfp4",
            operand_state=state,
            problem=(m, self.out_features, k),
            backend=backend,
            family=family,
            selection_source=source,
            autotune=mode,
            scaling=self.scaling,
            x_scale_region_rows=(
                self.x_scale_region_rows
                if self.scaling == "jit_row_region"
                else None
            ),
            weight_scale_region_rows=(
                self.weight_scale_region_rows
                if self.scaling == "jit_row_region"
                else None
            ),
            config={} if config is None else asdict(config),
            notes=(
                "Exact winner resolution is deferred to the registered runtime launcher."
                if source == "deferred_runtime"
                else "No compile or benchmark was performed."
            ,),
        )

    def extra_repr(self) -> str:
        geometry = (
            f", scale_regions=X:{self.x_scale_region_rows}/W:"
            f"{self.weight_scale_region_rows}"
            if self.scaling == "jit_row_region"
            else ""
        )
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias=False, forward=NVFP4, backward=MXFP8, "
            f"weight_mode={self.weight_mode}, scaling={self.scaling}"
            f"{geometry}, backend={self.backend}"
        )


def _clear_runtime_caches() -> dict[str, object]:
    before = {
        "materialized_dynamic": _DYNAMIC_RUNNERS.stats(),
        "delayed_dynamic": _DELAYED_DYNAMIC_RUNNERS.stats(),
        "jit_region_dynamic": _JIT_REGION_DYNAMIC_RUNNERS.stats(),
        "block_dynamic": _BLOCK_DYNAMIC_RUNNERS.stats(),
        "dynamic_x": _DYNAMIC_X_RUNNERS.stats(),
    }
    _DYNAMIC_RUNNERS.clear()
    _DELAYED_DYNAMIC_RUNNERS.clear()
    _JIT_REGION_DYNAMIC_RUNNERS.clear()
    _BLOCK_DYNAMIC_RUNNERS.clear()
    _DYNAMIC_X_RUNNERS.clear()
    _INFERENCE_CONFIG_SELECTIONS.clear()
    _DYNAMIC_CONFIG_SELECTIONS.clear()
    return before


__all__ = [
    "DEFAULT_NVFP4_WEIGHT_SCALE_REGION_ROWS",
    "DEFAULT_NVFP4_X_SCALE_REGION_ROWS",
    "NVFP4Linear",
    "nvfp4_linear",
    "quantize_nvfp4",
]
