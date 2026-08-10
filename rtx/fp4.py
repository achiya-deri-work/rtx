"""Torch frontend for NVFP4-forward, MXFP8-backward linear layers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
from threading import RLock
from typing import TYPE_CHECKING, Literal

import torch
from torch import nn

from .configs import (
    DEFAULT_NVFP4_GEMM_CONFIG,
    DEFAULT_NVFP4_FWD_CONFIG,
    DEFAULT_NVFP4_QUANT_CONFIG,
    NVFP4FwdConfig,
    NVFP4GemmConfig,
    NVFP4Problem,
    NVFP4QuantConfig,
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

if TYPE_CHECKING:
    from .kernels.mxfp8_bwd import MXFP8BwdConfig

WeightMode = Literal["dynamic", "prequantized"]
ScalingMode = Literal["delayed", "current", "regional", "block"]


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


def compile_nvfp4_gemm(*args, **kwargs):
    return load_kernel_symbol("nvfp4_gemm", "compile_nvfp4_gemm")(
        *args, **kwargs
    )


def compile_nvfp4_fwd(*args, **kwargs):
    return load_kernel_symbol("nvfp4_fwd", "compile_nvfp4_fwd")(
        *args, **kwargs
    )


def nvfp4_grid_ctas(*args, **kwargs):
    return load_kernel_symbol("nvfp4_fwd", "nvfp4_grid_ctas")(
        *args, **kwargs
    )


def nvfp4_telemetry_values(*args, **kwargs):
    return load_kernel_symbol("nvfp4_fwd", "nvfp4_telemetry_values")(
        *args, **kwargs
    )


@dataclass(frozen=True, slots=True)
class NVFP4ForwardConfig:
    """Dynamic fused schedule plus inference materialization schedules."""

    quant: NVFP4QuantConfig = DEFAULT_NVFP4_QUANT_CONFIG
    gemm: NVFP4GemmConfig = DEFAULT_NVFP4_GEMM_CONFIG
    quant_launches: str = "dual"
    fused: NVFP4FwdConfig | None = None
    l2_fetch_granularity: int | None = None
    x_scale_region_rows: int = 0
    weight_scale_region_rows: int = 0

    def rejection(self, problem: NVFP4Problem) -> str | None:
        fused = self.fused or _fallback_fused_config(problem)
        fused = replace(
            fused,
            x_scale_region_rows=_effective_scale_region_rows(
                problem.m, self.x_scale_region_rows
            ),
            weight_scale_region_rows=_effective_scale_region_rows(
                problem.n, self.weight_scale_region_rows
            ),
        )
        return fused.implementation_rejection(problem)

    def materialized_rejection(self, problem: NVFP4Problem) -> str | None:
        if self.quant_launches not in ("dual", "independent"):
            return "quant_launches must be dual or independent"
        for rows in (problem.m, problem.n):
            reason = self.quant.rejection(rows, problem.k)
            if reason is not None:
                return reason
        if self.l2_fetch_granularity not in (None, 0, 32, 64, 128):
            return "L2 fetch granularity must be None, 0, 32, 64, or 128"
        return self.gemm.rejection(problem)


DEFAULT_NVFP4_FORWARD_CONFIG = NVFP4ForwardConfig()
_FORWARD_CONFIGS: dict[str, NVFP4ForwardConfig] = {}
_CONFIG_LOCK = RLock()
_INFERENCE_CONFIG_SELECTIONS: dict[tuple[object, ...], str] = {}
_FUSED_CONFIG_SELECTIONS: dict[tuple[object, ...], NVFP4FwdConfig] = {}


@torch.compiler.assume_constant_result
def _intern_forward_config(config: NVFP4ForwardConfig) -> str:
    key = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    with _CONFIG_LOCK:
        _FORWARD_CONFIGS[key] = config
    return key


_DEFAULT_NVFP4_FORWARD_KEY = _intern_forward_config(DEFAULT_NVFP4_FORWARD_CONFIG)


@torch.compiler.assume_constant_result
def _functional_forward_config_key(
    fused: NVFP4FwdConfig | None,
    region_rows: int,
) -> str:
    """Intern functional policy without constructing dataclasses in FX."""

    if fused is None and region_rows == 0:
        return _DEFAULT_NVFP4_FORWARD_KEY
    return _intern_forward_config(
        NVFP4ForwardConfig(
            fused=fused,
            x_scale_region_rows=region_rows,
            weight_scale_region_rows=region_rows,
        )
    )


def _current_tensor_scale(tensor: torch.Tensor) -> torch.Tensor:
    """Return the exact TorchAO two-level decode scale without a host sync."""

    amax = torch.amax(torch.abs(tensor.detach().float()))
    scale = amax / (448.0 * 6.0)
    # A zero tensor has no useful dynamic range. One is a benign decode scale
    # and avoids 0/0 while every quantized value remains exactly zero.
    return torch.where(amax > 0.0, scale, torch.ones_like(scale)).reshape(1)


def _power2_scale_pack_from_amax(amax: torch.Tensor) -> torch.Tensor:
    target = amax / 2688.0
    safe_target = torch.clamp_min(target, torch.finfo(torch.float32).tiny)
    scale = torch.exp2(torch.ceil(torch.log2(safe_target)))
    scale = torch.where(amax > 0.0, scale, torch.ones_like(scale))
    inverse = torch.reciprocal(scale)
    return torch.stack((scale, inverse, inverse / 6.0), dim=-1).reshape(-1)


def _power2_tensor_scale_pack(tensor: torch.Tensor) -> torch.Tensor:
    """Current amax scale and exact reciprocals consumed by fused NVFP4."""

    amax = torch.amax(torch.abs(tensor.detach().float()))
    return _power2_scale_pack_from_amax(amax)


def _exact_tensor_scale_pack(tensor: torch.Tensor) -> torch.Tensor:
    """Exact current TorchAO tensor scale and its supplied reciprocals."""

    scale = _current_tensor_scale(tensor)
    inverse = torch.reciprocal(scale)
    return torch.cat((scale, inverse, inverse / 6.0))


def _regional_tensor_scale_pack(
    tensor: torch.Tensor,
    region_rows: int,
    mode: str,
) -> torch.Tensor:
    """Return adjacent ``(scale, inverse, inverse/6)`` packs per row region.

    The reduction and scale math deliberately remain ordinary Torch tensor
    expressions so ``torch.compile`` can generate the producer kernel.  The
    registered CuTe op only launches the fused quantize/GEMM consumer.
    """

    rows, k = tensor.shape
    if region_rows <= 0 or rows % region_rows:
        raise ValueError(
            f"regional NVFP4 scaling requires rows divisible by region_rows; "
            f"got rows={rows}, region_rows={region_rows}"
        )
    amax = torch.amax(
        torch.abs(tensor.detach().float()).reshape(
            rows // region_rows, region_rows * k
        ),
        dim=1,
    )
    if mode == "power2":
        return _power2_scale_pack_from_amax(amax)
    if mode == "exact":
        scale = amax / 2688.0
        scale = torch.where(amax > 0.0, scale, torch.ones_like(scale))
        inverse = torch.reciprocal(scale)
        return torch.stack((scale, inverse, inverse / 6.0), dim=-1).reshape(-1)
    raise ValueError(f"unknown NVFP4 tensor scale mode {mode!r}")


def _tensor_scale_pack(tensor: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "power2":
        return _power2_tensor_scale_pack(tensor)
    if mode == "exact":
        return _exact_tensor_scale_pack(tensor)
    raise ValueError(f"unknown NVFP4 tensor scale mode {mode!r}")


def _delayed_amax_state(tensor: torch.Tensor, values: int) -> torch.Tensor:
    """Bootstrap delayed telemetry/history without a host synchronization."""

    amax = torch.amax(torch.abs(tensor.detach().float())).reshape(1)
    return amax.expand(values).clone()


def _fallback_fused_config(
    problem: NVFP4Problem,
    *,
    collect_amax: bool = False,
) -> NVFP4FwdConfig:
    """Measured-safe fallback; runtime winner caches may replace it."""

    config = replace(DEFAULT_NVFP4_FWD_CONFIG, collect_amax=collect_amax)
    if problem.k % 256:
        config = replace(config, tile_k=128, bf16_tile_k=128)
    if (
        problem.m % config.tile_m
        or problem.n % config.tile_n
        or problem.k % config.bf16_tile_k
    ):
        # Vector loads require full tiles. Scalar BF16 loads retain predicates
        # for ragged M/N and for the native K=64 tail of a 128-wide CTA tile.
        config = replace(config, quant_load_bits=16)
    if problem.m <= 64:
        config = replace(
            config,
            tile_m=64,
            atom_layout_m=2,
            num_mma_warps=4,
            quantizer_warps=4,
            num_threads=128,
        )
    if problem.m >= 512 and problem.m % 256 == 0:
        config = replace(
            config,
            tile_m=256,
            persistent_waves=1,
            raster="m",
            grid_swizzle=1,
        )
        if problem.m <= 1024:
            # A focused 5070 Ti search found this one-wave basin consistently
            # faster than carrying the persistent work-loop machinery when the
            # natural grid already fits the device.
            config = replace(
                config,
                persistent=False,
                a_ldmatrix_matrices=4,
                b_ldmatrix_matrices=2,
                b_swizzle="none",
                sfa_s2r_bits=8,
                sfb_s2r_bits=8,
                maxrregcount=192,
            )
        else:
            config = replace(
                config,
                persistent=True,
                grid_swizzle=8,
                a_ldmatrix_matrices=1,
            )
    return config


def _runtime_fused_config(
    problem: NVFP4Problem,
    device: torch.device,
    *,
    collect_amax: bool,
) -> NVFP4FwdConfig:
    from .autotune.winners import load_runtime_winner, runtime_winner_key
    from .configs.nvfp4 import normalize_nvfp4_fwd_config

    cached = load_runtime_winner(
        runtime_winner_key("nvfp4_fused_fwd", problem, device=device),
        lambda value: normalize_nvfp4_fwd_config(**dict(value)),
        rejection=lambda candidate: candidate.implementation_rejection(problem),
    )
    selected = cached or _fallback_fused_config(problem)
    return replace(selected, collect_amax=collect_amax)


def _resolved_fused_config(
    forward_config_key: str,
    problem: NVFP4Problem,
    device: torch.device,
    *,
    collect_amax: bool,
) -> NVFP4FwdConfig:
    """Resolve explicit/runtime schedules once per device and static shape."""

    key = (
        forward_config_key,
        device.index,
        problem.m,
        problem.n,
        problem.k,
        collect_amax,
    )
    selected = _FUSED_CONFIG_SELECTIONS.get(key)
    if selected is not None:
        return selected
    with _CONFIG_LOCK:
        selected = _FUSED_CONFIG_SELECTIONS.get(key)
        if selected is None:
            public_config = _resolve_forward_config(forward_config_key)
            selected = (
                replace(public_config.fused, collect_amax=collect_amax)
                if public_config.fused is not None
                else _runtime_fused_config(
                    problem, device, collect_amax=collect_amax
                )
            )
            selected = replace(
                selected,
                x_scale_region_rows=_effective_scale_region_rows(
                    problem.m, public_config.x_scale_region_rows
                ),
                weight_scale_region_rows=_effective_scale_region_rows(
                    problem.n, public_config.weight_scale_region_rows
                ),
            )
            _FUSED_CONFIG_SELECTIONS[key] = selected
    return selected


def _packed_fp4_view(tensor: torch.Tensor) -> torch.Tensor:
    dtype = getattr(torch, "float4_e2m1fn_x2", None)
    if dtype is None:
        raise RuntimeError("PyTorch must expose torch.float4_e2m1fn_x2")
    return tensor if tensor.dtype is dtype else tensor.view(dtype)


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
class _FusedDynamicRunner:
    launcher: object
    grid_ctas: int
    telemetry_values: int
    scalar_telemetry: bool
    dummy_telemetry: torch.Tensor | None = None

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        x_scale_pack: torch.Tensor,
        weight_scale_pack: torch.Tensor,
        out: torch.Tensor,
        x_amax: torch.Tensor | None,
        weight_amax: torch.Tensor | None,
    ) -> None:
        if x_amax is None or weight_amax is None:
            if self.dummy_telemetry is None:
                raise RuntimeError("NVFP4 current-scale runner has no telemetry sink")
            x_amax = weight_amax = self.dummy_telemetry
        elif self.scalar_telemetry:
            # Stream-ordered raw CUDA operations keep mutable state preparation
            # out of the eager/FX graph while avoiding a separate CuTe kernel.
            from .fp8_bwd import _zero_tensor_async

            _zero_tensor_async(x_amax)
            _zero_tensor_async(weight_amax)
        self.launcher(
            x,
            weight,
            out,
            x_scale_pack,
            weight_scale_pack,
            x_amax,
            weight_amax,
        )


_FUSED_DYNAMIC_RUNNERS: BoundedCache[
    tuple[object, ...], _FusedDynamicRunner
] = BoundedCache(runner_cache_limit("fused_dynamic", 8, namespace="NVFP4"))


def _make_fused_dynamic_runner(
    problem: NVFP4Problem,
    config: NVFP4FwdConfig,
    device: torch.device | None = None,
) -> _FusedDynamicRunner:
    dummy_telemetry = None
    if not config.collect_amax:
        if device is None:
            raise ValueError("current-scale NVFP4 runner requires a CUDA device")
        # The ABI retains one ignored telemetry pointer even when collection
        # is compiled out. Allocate its required one-element shape once, when
        # the runner is built, rather than slicing a scale pack on every call.
        dummy_telemetry = torch.empty(1, dtype=torch.float32, device=device)
    return _FusedDynamicRunner(
        compile_nvfp4_fwd(problem, config),
        nvfp4_grid_ctas(problem, config),
        nvfp4_telemetry_values(problem, config) if config.collect_amax else 0,
        bool(config.collect_amax and config.telemetry_layout == "scalar_atomic"),
        dummy_telemetry,
    )


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
    qx = torch.empty((problem.m, problem.k // 2), dtype=torch.uint8, device=device)
    qw = torch.empty((problem.n, problem.k // 2), dtype=torch.uint8, device=device)
    sx = torch.empty(
        (problem.m, problem.k // 16), dtype=torch.float8_e4m3fn, device=device
    )
    sw = torch.empty(
        (problem.n, problem.k // 16), dtype=torch.float8_e4m3fn, device=device
    )
    if config.quant_launches == "dual":
        quant_x = compile_nvfp4_dual_quant(
            problem.m, problem.n, problem.k, config.quant
        )
        quant_w = None
    else:
        quant_x = compile_nvfp4_quant(problem.m, problem.k, config.quant)
        quant_w = compile_nvfp4_quant(problem.n, problem.k, config.quant)
    return _DynamicRunner(
        config.quant_launches,
        quant_x,
        quant_w,
        compile_nvfp4_gemm(problem, config.gemm),
        qx,
        qw,
        sx,
        sw,
        _packed_fp4_view(qx),
        _packed_fp4_view(qw),
        config.l2_fetch_granularity,
    )


def _make_dynamic_x_runner(
    problem: NVFP4Problem,
    config: NVFP4ForwardConfig,
    device: torch.device,
) -> _DynamicXRunner:
    qx = torch.empty((problem.m, problem.k // 2), dtype=torch.uint8, device=device)
    sx = torch.empty(
        (problem.m, problem.k // 16), dtype=torch.float8_e4m3fn, device=device
    )
    return _DynamicXRunner(
        compile_nvfp4_quant(problem.m, problem.k, config.quant),
        compile_nvfp4_gemm(problem, config.gemm),
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
) -> str:
    """Resolve an installed state-specific winner once per device and shape."""

    state = "fully_prequantized" if fully_prequantized else "weight_prequantized"
    selection_key = (
        state,
        device.index,
        problem.m,
        problem.n,
        problem.k,
    )
    selected_key = _INFERENCE_CONFIG_SELECTIONS.get(selection_key)
    if selected_key is not None:
        return selected_key
    from .autotune.winners import load_runtime_winner, runtime_winner_key
    from .nvfp4_inference_autotune import (
        fully_prequant_config_from_dict,
        weight_prequant_config_from_dict,
    )

    if fully_prequantized:
        family = "nvfp4_fully_prequant_fwd"
        variant = "x-row_major_w-row_major"
        selected = load_runtime_winner(
            runtime_winner_key(family, problem, device=device, variant=variant),
            fully_prequant_config_from_dict,
            rejection=lambda value: value.rejection(problem),
        )
        config = (
            DEFAULT_NVFP4_FORWARD_CONFIG
            if selected is None
            else NVFP4ForwardConfig(
                gemm=selected.gemm,
                l2_fetch_granularity=selected.l2_fetch_granularity,
            )
        )
    else:
        family = "nvfp4_weight_prequant_fwd"
        variant = "w-row_major"
        selected = load_runtime_winner(
            runtime_winner_key(family, problem, device=device, variant=variant),
            weight_prequant_config_from_dict,
            rejection=lambda value: value.rejection(problem),
        )
        config = (
            DEFAULT_NVFP4_FORWARD_CONFIG
            if selected is None
            else NVFP4ForwardConfig(
                quant=selected.quant_x,
                gemm=selected.gemm,
                quant_launches="independent",
                l2_fetch_granularity=selected.l2_fetch_granularity,
            )
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
) -> str:
    """Compiler-safe shape facade for the inference winner lookup."""

    return _packed_inference_config_key(
        NVFP4Problem(int(m), int(n), int(k)),
        device,
        fully_prequantized=fully_prequantized,
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


def _launch_nvfp4_forward_scaled(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    forward_config_key: str,
) -> torch.Tensor:
    _check_nvfp4_inputs(x, weight)
    _check_sm12x(x.device)
    x_c = x if x.is_contiguous() else x.contiguous()
    weight_c = weight if weight.is_contiguous() else weight.contiguous()
    problem = NVFP4Problem(int(x_c.shape[0]), int(weight_c.shape[0]), int(x_c.shape[1]))
    config = _resolve_forward_config(forward_config_key)
    rejection = config.rejection(problem)
    if rejection is not None:
        raise RuntimeError(f"NVFP4 configuration cannot run this problem: {rejection}")
    stream = torch.cuda.current_stream(x.device)
    fused_config = _resolved_fused_config(
        forward_config_key,
        problem,
        x.device,
        collect_amax=False,
    )
    expected_x_scales = (
        3 * (problem.m // fused_config.x_scale_region_rows)
        if fused_config.x_scale_region_rows
        else 3
    )
    expected_weight_scales = (
        3 * (problem.n // fused_config.weight_scale_region_rows)
        if fused_config.weight_scale_region_rows
        else 3
    )
    if x_scale.dtype is not torch.float32 or weight_scale.dtype is not torch.float32:
        raise TypeError("fused NVFP4 scale packs must use FP32")
    if x_scale.numel() != expected_x_scales:
        raise ValueError(
            f"fused NVFP4 X scale pack has {x_scale.numel()} values, "
            f"expected {expected_x_scales}"
        )
    if weight_scale.numel() != expected_weight_scales:
        raise ValueError(
            f"fused NVFP4 weight scale pack has {weight_scale.numel()} values, "
            f"expected {expected_weight_scales}"
        )
    cache_key = (
        x.device.index,
        stream.cuda_stream,
        problem.m,
        problem.n,
        problem.k,
        fused_config,
    )
    runner = _FUSED_DYNAMIC_RUNNERS.get(cache_key)
    if runner is None:
        runner = _make_fused_dynamic_runner(problem, fused_config, x.device)
        _FUSED_DYNAMIC_RUNNERS[cache_key] = runner
    out = torch.empty((problem.m, problem.n), dtype=torch.bfloat16, device=x.device)
    runner(x_c, weight_c, x_scale, weight_scale, out, None, None)
    out._base_inputs = (x_c, weight_c, x_scale, weight_scale, runner)
    return out


def _launch_nvfp4_forward_delayed(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_amax_state: torch.Tensor,
    weight_amax_state: torch.Tensor,
    forward_config_key: str,
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
    fused_config = _resolved_fused_config(
        forward_config_key,
        problem,
        x.device,
        collect_amax=True,
    )
    rejection = fused_config.implementation_rejection(problem)
    if rejection is not None:
        raise RuntimeError(f"delayed NVFP4 configuration cannot run: {rejection}")
    stream = torch.cuda.current_stream(x.device)
    cache_key = (
        x.device.index,
        stream.cuda_stream,
        problem.m,
        problem.n,
        problem.k,
        fused_config,
    )
    runner = _FUSED_DYNAMIC_RUNNERS.get(cache_key)
    if runner is None:
        runner = _make_fused_dynamic_runner(problem, fused_config)
        _FUSED_DYNAMIC_RUNNERS[cache_key] = runner
    for name, value in (
        ("x_amax_state", x_amax_state),
        ("weight_amax_state", weight_amax_state),
        ("next_x_amax_state", next_x_amax_state),
        ("next_weight_amax_state", next_weight_amax_state),
    ):
        if value.numel() != runner.telemetry_values:
            raise ValueError(
                f"{name} has {value.numel()} values, expected "
                f"{runner.telemetry_values}"
            )
    out = torch.empty((problem.m, problem.n), dtype=torch.bfloat16, device=x.device)
    if next_x_amax_state.data_ptr() == x_amax_state.data_ptr():
        raise ValueError("delayed x amax generations must not alias")
    if next_weight_amax_state.data_ptr() == weight_amax_state.data_ptr():
        raise ValueError("delayed weight amax generations must not alias")
    runner(
        x_c,
        weight_c,
        x_amax_state,
        weight_amax_state,
        out,
        next_x_amax_state,
        next_weight_amax_state,
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
    qdata = torch.empty((rows, k // 2), dtype=torch.uint8, device=source.device)
    scales = torch.empty(
        (rows, k // 16), dtype=torch.float8_e4m3fn, device=source.device
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
    "rtx::nvfp4_linear_fwd",
    mutates_args=(),
    device_types="cuda",
)
def _nvfp4_linear_fwd_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    forward_config_key: str,
) -> torch.Tensor:
    return _launch_nvfp4_forward_scaled(
        x, weight, x_scale, weight_scale, forward_config_key
    )


@_nvfp4_linear_fwd_op.register_fake
def _nvfp4_linear_fwd_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
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
    if x.ndim != 2 or tuple(weight_data.shape) != (n, k // 2):
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
    compile_nvfp4_gemm(problem, config.gemm)(
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


@torch.library.custom_op(
    "rtx::nvfp4_linear_train",
    mutates_args=(),
    device_types="cuda",
)
def _nvfp4_linear_train_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    forward_config_key: str,
    backward_config_key: str,
) -> torch.Tensor:
    return _launch_nvfp4_forward_scaled(
        x, weight, x_scale, weight_scale, forward_config_key
    )


@_nvfp4_linear_train_op.register_fake
def _nvfp4_linear_train_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    forward_config_key: str,
    backward_config_key: str,
) -> torch.Tensor:
    return torch.empty(
        (x.shape[0], weight.shape[0]), dtype=torch.bfloat16, device=x.device
    )


def _setup_nvfp4_context(ctx, inputs, output) -> None:
    (
        x,
        weight,
        _x_scale,
        _weight_scale,
        _forward_config_key,
        backward_config_key,
    ) = inputs
    ctx.save_for_backward(x, weight)
    ctx.backward_config_key = backward_config_key


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


torch.library.register_autograd(
    "rtx::nvfp4_linear_train",
    _nvfp4_backward,
    setup_context=_setup_nvfp4_context,
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
    backward_config_key: str,
) -> torch.Tensor:
    return torch.empty(
        (x.shape[0], weight.shape[0]), dtype=torch.bfloat16, device=x.device
    )


class _InductorNVFP4DelayedLauncher:
    """Shape-bound delayed launcher used directly by generated wrappers."""

    def __init__(self, forward_config_key: str) -> None:
        self.forward_config_key = forward_config_key
        self.runners: BoundedCache[tuple[object, ...], _FusedDynamicRunner] = (
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
        fused_config = _resolved_fused_config(
            self.forward_config_key,
            problem,
            x.device,
            collect_amax=True,
        )
        key = (
            x.device.index,
            stream_id,
            problem.m,
            problem.n,
            problem.k,
            fused_config,
        )
        runner = self.runners.get(key)
        if runner is None:
            runner = _make_fused_dynamic_runner(problem, fused_config, x.device)
            self.runners[key] = runner
        runner(
            x,
            weight,
            x_amax_state,
            weight_amax_state,
            out,
            next_x_amax_state,
            next_weight_amax_state,
        )


class _InductorNVFP4DelayedRegistry(dict[str, object]):
    def __missing__(self, config_key: str) -> object:
        with _CONFIG_LOCK:
            launcher = self.get(config_key)
            if launcher is None:
                launcher = _InductorNVFP4DelayedLauncher(config_key)
                self[config_key] = launcher
        return launcher


_INDUCTOR_DELAYED_LAUNCHERS = _InductorNVFP4DelayedRegistry()


class _InductorNVFP4CurrentLauncher:
    """Direct fused launch after compiler-visible current-scale preparation."""

    def __init__(self, forward_config_key: str) -> None:
        self.forward_config_key = forward_config_key
        self.runners: BoundedCache[tuple[object, ...], _FusedDynamicRunner] = (
            BoundedCache(runner_cache_limit("inductor_current", 8, namespace="NVFP4"))
        )

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        x_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        *,
        out: torch.Tensor,
    ) -> None:
        problem = NVFP4Problem(
            int(x.shape[0]), int(weight.shape[0]), int(x.shape[1])
        )
        stream_id = int(torch._C._cuda_getCurrentRawStream(x.device.index))
        fused_config = _resolved_fused_config(
            self.forward_config_key,
            problem,
            x.device,
            collect_amax=False,
        )
        rejection = fused_config.implementation_rejection(problem)
        if rejection is not None:
            raise RuntimeError(f"NVFP4 configuration cannot run: {rejection}")
        expected_x_scales = (
            3 * (problem.m // fused_config.x_scale_region_rows)
            if fused_config.x_scale_region_rows
            else 3
        )
        expected_weight_scales = (
            3 * (problem.n // fused_config.weight_scale_region_rows)
            if fused_config.weight_scale_region_rows
            else 3
        )
        if x_scale.numel() != expected_x_scales:
            raise RuntimeError(
                f"compiled NVFP4 X scale ABI expected {expected_x_scales} "
                f"values, got {x_scale.numel()}"
            )
        if weight_scale.numel() != expected_weight_scales:
            raise RuntimeError(
                f"compiled NVFP4 weight scale ABI expected "
                f"{expected_weight_scales} values, got {weight_scale.numel()}"
            )
        key = (
            x.device.index,
            stream_id,
            problem.m,
            problem.n,
            problem.k,
            fused_config,
        )
        runner = self.runners.get(key)
        if runner is None:
            runner = _make_fused_dynamic_runner(problem, fused_config, x.device)
            self.runners[key] = runner
        runner(x, weight, x_scale, weight_scale, out, None, None)


class _InductorNVFP4CurrentRegistry(dict[str, object]):
    def __missing__(self, config_key: str) -> object:
        with _CONFIG_LOCK:
            launcher = self.get(config_key)
            if launcher is None:
                launcher = _InductorNVFP4CurrentLauncher(config_key)
                self[config_key] = launcher
        return launcher


_INDUCTOR_CURRENT_LAUNCHERS = _InductorNVFP4CurrentRegistry()


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
    torch._rtx_nvfp4_current_launchers = _INDUCTOR_CURRENT_LAUNCHERS
    torch._rtx_nvfp4_dynamic_x_launchers = _INDUCTOR_DYNAMIC_X_LAUNCHERS
    torch._rtx_nvfp4_prequant_launchers = _INDUCTOR_PREQUANT_LAUNCHERS

    def lower_current_common(x, weight, x_scale, weight_scale, config_key):
        inputs = [
            ir.ExternKernel.require_contiguous(ir.ExternKernel.realize_input(value))
            for value in (x, weight, x_scale, weight_scale)
        ]
        m = x.get_size()[0]
        n = weight.get_size()[0]
        name = f"torch._rtx_nvfp4_current_launchers[{config_key!r}]"
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
        torch.ops.rtx.nvfp4_linear_fwd.default,
        type_promotion_kind=None,
    )
    def lower_current_fwd(x, weight, x_scale, weight_scale, config_key):
        return lower_current_common(
            x, weight, x_scale, weight_scale, config_key
        )

    @register_lowering(
        torch.ops.rtx.nvfp4_linear_train.default,
        type_promotion_kind=None,
    )
    def lower_current_train(
        x,
        weight,
        x_scale,
        weight_scale,
        config_key,
        backward_config_key,
    ):
        return lower_current_common(
            x, weight, x_scale, weight_scale, config_key
        )

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
            f"[{forward_config_key!r}]"
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
    return grad_x, grad_weight, None, None, None, None, None, None


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
            backward_config_key,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return _nvfp4_delayed_backward(ctx, grad_output)


def nvfp4_linear(
    x: torch.Tensor | NVFP4Tensor,
    weight: torch.Tensor | NVFP4Tensor,
    *,
    forward_config: NVFP4FwdConfig | None = None,
    backward_config: "MXFP8BwdConfig | None" = None,
    scaling: Literal["current", "regional", "block"] = "current",
    scale_region_rows: int = 1,
) -> torch.Tensor:
    """Apply NVFP4 forward and MXFP8 backward to BF16 or packed operands."""

    if scaling not in ("current", "regional", "block"):
        raise ValueError(
            "functional NVFP4 scaling must be current, regional, or block"
        )
    if scale_region_rows <= 0:
        raise ValueError("scale_region_rows must be positive")

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
                ),
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
        if scaling == "regional":
            raise NotImplementedError(
                "regional outer scaling currently requires dynamic BF16 X and W; "
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
    region_rows = scale_region_rows if scaling == "regional" else 0
    forward_key = _functional_forward_config_key(
        forward_config,
        region_rows,
    )
    tensor_scale_mode = (
        "power2" if forward_config is None else forward_config.tensor_scale_mode
    )
    from .fp8 import _autotune_mode, _backward_config_key
    from .fp8_bwd import _intern_bwd_config

    backward_key = (
        _intern_bwd_config(backward_config)
        if backward_config is not None
        else _backward_config_key(_autotune_mode(None), None, None)
    )
    fixed_scale_pack = None
    if scaling == "block":
        # Compiler-visible constant construction is folded/hoisted by
        # Inductor. Stateful delayed scaling remains an nn.Module policy.
        fixed_scale_pack = x.new_tensor(
            (1.0, 1.0, 1.0 / 6.0), dtype=torch.float32
        )
    return _nvfp4_dynamic_linear_with_keys(
        x,
        weight,
        forward_key=forward_key,
        backward_key=backward_key,
        tensor_scale_mode=tensor_scale_mode,
        x_scale_region_rows=region_rows,
        weight_scale_region_rows=region_rows,
        fixed_scale_pack=fixed_scale_pack,
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
    fixed_scale_pack: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compiler-visible current scaling around launch-only NVFP4 operators."""

    leading_shape = x.shape[:-1]
    x_2d = x.reshape(-1, x.shape[-1])
    _check_nvfp4_inputs(x_2d, weight)
    # These reductions and pointwise expressions intentionally remain in FX.
    # Inductor owns their fusion and schedules only the final CuTe launch as an
    # external kernel; none of this eager tensor work is hidden in registration.
    if fixed_scale_pack is None:
        effective_x_rows = _effective_scale_region_rows(
            x_2d.shape[0], x_scale_region_rows
        )
        effective_weight_rows = _effective_scale_region_rows(
            weight.shape[0], weight_scale_region_rows
        )
        x_scale = (
            _regional_tensor_scale_pack(
                x_2d, effective_x_rows, tensor_scale_mode
            )
            if effective_x_rows
            else _tensor_scale_pack(x_2d, tensor_scale_mode)
        )
        weight_scale = (
            _regional_tensor_scale_pack(
                weight, effective_weight_rows, tensor_scale_mode
            )
            if effective_weight_rows
            else _tensor_scale_pack(weight, tensor_scale_mode)
        )
    else:
        if fixed_scale_pack.dtype is not torch.float32:
            raise TypeError("block-only NVFP4 scale pack must use FP32")
        x_scale = weight_scale = fixed_scale_pack
    if torch.is_grad_enabled() and (x_2d.requires_grad or weight.requires_grad):
        out = _nvfp4_linear_train_op(
            x_2d,
            weight,
            x_scale,
            weight_scale,
            forward_key,
            backward_key,
        )
    else:
        out = _nvfp4_linear_fwd_op(
            x_2d, weight, x_scale, weight_scale, forward_key
        )
    return out.reshape(*leading_shape, weight.shape[0])


class NVFP4Linear(nn.Module):
    """No-bias BF16 linear with NVFP4 forward and MXFP8 backward."""

    __constants__ = [
        "in_features",
        "out_features",
        "weight_mode",
        "scaling",
        "_forward_config_key",
        "_backward_config_key",
        "_tensor_scale_mode",
        "scale_region_rows",
    ]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        forward_config: NVFP4FwdConfig | None = None,
        backward_config: "MXFP8BwdConfig | None" = None,
        scaling: ScalingMode = "delayed",
        scale_region_rows: int = 1,
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
        if scaling not in ("delayed", "current", "regional", "block"):
            raise ValueError(
                "NVFP4 scaling must be delayed, current, regional, or block"
            )
        if scale_region_rows <= 0:
            raise ValueError("scale_region_rows must be positive")
        self.forward_config = forward_config
        self.backward_config = backward_config
        self.scaling = scaling
        self.scale_region_rows = int(scale_region_rows)
        self._forward_config_key = _intern_forward_config(
            NVFP4ForwardConfig(
                fused=forward_config,
                x_scale_region_rows=(
                    self.scale_region_rows if scaling == "regional" else 0
                ),
                weight_scale_region_rows=(
                    self.scale_region_rows if scaling == "regional" else 0
                ),
            )
        )
        self._tensor_scale_mode = (
            "power2" if forward_config is None else forward_config.tensor_scale_mode
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
        from .fp8 import _autotune_mode, _backward_config_key
        from .fp8_bwd import _intern_bwd_config

        self._backward_config_key = (
            _intern_bwd_config(backward_config)
            if backward_config is not None
            else _backward_config_key(_autotune_mode(None), None, None)
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
            if packed_weight.shape != (out_features, in_features):
                raise ValueError(
                    f"packed NVFP4 weight must have shape {(out_features, in_features)}"
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
            forward_config=self.forward_config,
            scaling=self.scaling,
            scale_region_rows=self.scale_region_rows,
            packed_weight=packed,
        )

    def forward(self, x: torch.Tensor | NVFP4Tensor) -> torch.Tensor:
        if self.weight_mode == "prequantized":
            if self.scaling == "regional":
                raise NotImplementedError(
                    "regional outer scaling currently requires a dynamic BF16 weight"
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
                    ),
                )
                return out.reshape(*x.shape[:-1], self.out_features)
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
                ),
            )
            return out.reshape(*leading_shape, self.out_features)
        if isinstance(x, NVFP4Tensor):
            raise TypeError(
                "a prequantized activation requires a prequantized module weight"
            )
        assert self.weight is not None
        if (
            self.training
            and torch.is_grad_enabled()
            and (x.requires_grad or self.weight.requires_grad)
            and self.scaling == "delayed"
        ):
            leading_shape = x.shape[:-1]
            x_2d = x.reshape(-1, x.shape[-1])
            _check_nvfp4_inputs(x_2d, self.weight)
            delayed_problem = (
                x_2d.device.type,
                x_2d.device.index,
                torch.cuda.current_stream(x_2d.device).cuda_stream,
                int(x_2d.shape[0]),
                int(self.weight.shape[0]),
                int(x_2d.shape[1]),
            )
            if (
                not self._delayed_initialized
                or self._delayed_problem != delayed_problem
            ):
                with torch.no_grad():
                    problem = NVFP4Problem(
                        int(x_2d.shape[0]),
                        int(self.weight.shape[0]),
                        int(x_2d.shape[1]),
                    )
                    fused_config = (
                        replace(self.forward_config, collect_amax=True)
                        if self.forward_config is not None
                        else _runtime_fused_config(
                            problem,
                            x_2d.device,
                            collect_amax=True,
                        )
                    )
                    state_values = nvfp4_telemetry_values(problem, fused_config)
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
            out = _NVFP4DelayedAutograd.apply(
                x_2d,
                self.weight,
                self._x_amax_state,
                self._weight_amax_state,
                self._next_x_amax_state,
                self._next_weight_amax_state,
                self._forward_config_key,
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
                self.scale_region_rows if self.scaling == "regional" else 0
            ),
            weight_scale_region_rows=(
                self.scale_region_rows if self.scaling == "regional" else 0
            ),
            fixed_scale_pack=(
                self._block_scale_pack if self.scaling == "block" else None
            ),
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias=False, forward=NVFP4, backward=MXFP8, "
            f"weight_mode={self.weight_mode}, scaling={self.scaling}"
        )


def _clear_runtime_caches() -> dict[str, object]:
    before = {
        "fused_dynamic": _FUSED_DYNAMIC_RUNNERS.stats(),
        "materialized_dynamic": _DYNAMIC_RUNNERS.stats(),
        "dynamic_x": _DYNAMIC_X_RUNNERS.stats(),
    }
    _FUSED_DYNAMIC_RUNNERS.clear()
    _DYNAMIC_RUNNERS.clear()
    _DYNAMIC_X_RUNNERS.clear()
    _INFERENCE_CONFIG_SELECTIONS.clear()
    _FUSED_CONFIG_SELECTIONS.clear()
    return before


__all__ = ["NVFP4Linear", "nvfp4_linear", "quantize_nvfp4"]
