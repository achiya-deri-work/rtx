"""Torch-facing MXFP8 linear backward composition and launcher cache."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Literal

import torch

from .kernels.mxfp8 import MXFP8Problem
from .kernels.mxfp8_bwd import (
    DEFAULT_MXFP8_BWD_CONFIG,
    MXFP8BwdConfig,
    MXFP8BwdMatmulConfig,
)
from .kernels.mxfp8_gemm import compile_mxfp8_gemm
from .kernels.mxfp8_quant import (
    compile_mxfp8_oriented_dual_quant,
    compile_mxfp8_quant,
    compile_mxfp8_transposed_quant,
)

if TYPE_CHECKING:
    from .autotune import CoordinateDescentPolicy

AutotuneMode = Literal["off", "cache", "coordinate"]


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


@dataclass(slots=True)
class _MatmulRunner:
    quant_launches: str
    quant_a: object
    quant_b: object | None
    gemm: object
    quantized_a: torch.Tensor
    quantized_b: torch.Tensor
    scales_a: torch.Tensor
    scales_b: torch.Tensor

    def quantize(self, source_a: torch.Tensor, source_b: torch.Tensor) -> None:
        if self.quant_launches == "dual":
            self.quant_a(
                source_a,
                source_b,
                self.quantized_a,
                self.quantized_b,
                self.scales_a,
                self.scales_b,
            )
        else:
            self.quant_a(source_a, self.quantized_a, self.scales_a)
            assert self.quant_b is not None
            self.quant_b(source_b, self.quantized_b, self.scales_b)

    def matmul(self, out: torch.Tensor) -> None:
        self.gemm(
            self.quantized_a,
            self.quantized_b,
            self.scales_a,
            self.scales_b,
            out,
        )

    def __call__(
        self,
        source_a: torch.Tensor,
        source_b: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        self.quantize(source_a, source_b)
        self.matmul(out)


@dataclass(slots=True)
class _BwdRunner:
    execution_order: str
    dx: _MatmulRunner
    dw: _MatmulRunner

    def __call__(
        self,
        grad_output: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        grad_x: torch.Tensor,
        grad_weight: torch.Tensor,
    ) -> None:
        if self.execution_order == "dx_first":
            self.dx(grad_output, weight.T, grad_x)
            self.dw(grad_output.T, x.T, grad_weight)
        else:
            self.dw(grad_output.T, x.T, grad_weight)
            self.dx(grad_output, weight.T, grad_x)


def _build_matmul_runner(
    *,
    problem: MXFP8Problem,
    config: MXFP8BwdMatmulConfig,
    device: torch.device,
) -> _MatmulRunner:
    a_shape = (problem.m, problem.k)
    b_shape = (problem.n, problem.k)
    quantized_a = torch.empty(a_shape, dtype=torch.float8_e4m3fn, device=device)
    quantized_b = torch.empty(b_shape, dtype=torch.float8_e4m3fn, device=device)
    quant_b_config = config.resolved_quant_b()
    scales_a = _allocate_scales(
        problem.m, problem.k, config.quant_a.scale_layout, device
    )
    scales_b = _allocate_scales(
        problem.n, problem.k, quant_b_config.scale_layout, device
    )
    if config.quant_launches == "dual":
        quant_a_launcher = compile_mxfp8_oriented_dual_quant(
            problem.m,
            problem.n,
            problem.k,
            config.quant_a,
            b_config=quant_b_config,
            a_orientation=config.a_orientation,
            b_orientation=config.b_orientation,
        )
        quant_b_launcher = None
    else:
        compile_a = (
            compile_mxfp8_transposed_quant
            if config.a_orientation == "transpose"
            else compile_mxfp8_quant
        )
        compile_b = (
            compile_mxfp8_transposed_quant
            if config.b_orientation == "transpose"
            else compile_mxfp8_quant
        )
        quant_a_launcher = compile_a(problem.m, problem.k, config.quant_a)
        quant_b_launcher = compile_b(problem.n, problem.k, quant_b_config)
    return _MatmulRunner(
        quant_launches=config.quant_launches,
        quant_a=quant_a_launcher,
        quant_b=quant_b_launcher,
        gemm=compile_mxfp8_gemm(problem, config.gemm),
        quantized_a=quantized_a,
        quantized_b=quantized_b,
        scales_a=scales_a,
        scales_b=scales_b,
    )


def _build_bwd_runner(
    problem: MXFP8Problem,
    config: MXFP8BwdConfig,
    device: torch.device,
) -> _BwdRunner:
    reason = config.implementation_rejection(problem)
    if reason is not None:
        raise RuntimeError(f"MXFP8 backward cannot run this configuration: {reason}")
    dx = _build_matmul_runner(
        problem=MXFP8Problem(problem.m, problem.k, problem.n),
        config=config.dx,
        device=device,
    )
    dw = _build_matmul_runner(
        problem=MXFP8Problem(problem.n, problem.k, problem.m),
        config=config.dw,
        device=device,
    )
    return _BwdRunner(config.execution_order, dx, dw)


_CONFIGS: dict[str, MXFP8BwdConfig] = {}
_RUNNERS: dict[tuple[object, ...], _BwdRunner] = {}
_LOCK = RLock()


@torch.compiler.assume_constant_result
def _intern_bwd_config(config: MXFP8BwdConfig) -> str:
    config = config.normalized()
    key = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    with _LOCK:
        _CONFIGS[key] = config
    return key


_DEFAULT_BWD_KEY = _intern_bwd_config(DEFAULT_MXFP8_BWD_CONFIG)


def _check_bwd_inputs(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
) -> MXFP8Problem:
    if grad_output.device.type != "cuda" or x.device.type != "cuda" or weight.device.type != "cuda":
        raise ValueError("MXFP8 backward only accepts CUDA tensors")
    if not (grad_output.device == x.device == weight.device):
        raise ValueError("grad_output, x, and weight must be on one CUDA device")
    if not (
        grad_output.dtype is torch.bfloat16
        and x.dtype is torch.bfloat16
        and weight.dtype is torch.bfloat16
    ):
        raise TypeError("MXFP8 backward expects BF16 grad_output, x, and weight")
    if grad_output.ndim != 2 or x.ndim != 2 or weight.ndim != 2:
        raise ValueError("internal MXFP8 backward expects three 2D tensors")
    problem = MXFP8Problem(int(x.shape[0]), int(weight.shape[0]), int(x.shape[1]))
    if weight.shape[1] != problem.k or grad_output.shape != (problem.m, problem.n):
        raise ValueError(
            f"backward shape mismatch: {grad_output.shape=} {x.shape=} {weight.shape=}"
        )
    return problem


def _launch_bwd(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    config_key: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    problem, runner, grad_output_c, x_c, weight_c = _resolve_bwd_runner(
        grad_output, x, weight, config_key
    )
    grad_x = torch.empty_like(x_c)
    grad_weight = torch.empty_like(weight_c)
    runner(grad_output_c, x_c, weight_c, grad_x, grad_weight)
    grad_x._base_inputs = (grad_output_c, x_c, weight_c)
    grad_weight._base_inputs = (grad_output_c, x_c, weight_c, grad_x)
    return grad_x, grad_weight


def _resolve_bwd_runner(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    config_key: str,
) -> tuple[
    MXFP8Problem,
    _BwdRunner,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    problem = _check_bwd_inputs(grad_output, x, weight)
    config = _CONFIGS.get(config_key)
    if config is None:
        raise RuntimeError("unknown MXFP8 backward configuration key")
    reason = config.implementation_rejection(problem)
    if reason is not None:
        raise RuntimeError(f"MXFP8 backward configuration cannot run: {reason}")
    major, _minor = torch.cuda.get_device_capability(x.device)
    if major != 12:
        raise RuntimeError(
            "native RTX MXFP8 backward requires SM120/SM121; "
            f"got {torch.cuda.get_device_capability(x.device)}"
        )
    grad_output_c = grad_output if grad_output.is_contiguous() else grad_output.contiguous()
    x_c = x if x.is_contiguous() else x.contiguous()
    weight_c = weight if weight.is_contiguous() else weight.contiguous()
    stream = torch.cuda.current_stream(x.device)
    runner_key = (
        x.device.index,
        int(stream.cuda_stream),
        problem.m,
        problem.n,
        problem.k,
        config_key,
    )
    runner = _RUNNERS.get(runner_key)
    if runner is None:
        with _LOCK:
            runner = _RUNNERS.get(runner_key)
            if runner is None:
                runner = _build_bwd_runner(problem, config, x.device)
                _RUNNERS[runner_key] = runner
    return problem, runner, grad_output_c, x_c, weight_c


def _launch_dx(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    config_key: str,
) -> torch.Tensor:
    _problem, runner, grad_output_c, x_c, weight_c = _resolve_bwd_runner(
        grad_output, x, weight, config_key
    )
    grad_x = torch.empty_like(x_c)
    runner.dx(grad_output_c, weight_c.T, grad_x)
    grad_x._base_inputs = (grad_output_c, x_c, weight_c)
    return grad_x


def _launch_dw(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    config_key: str,
) -> torch.Tensor:
    _problem, runner, grad_output_c, x_c, weight_c = _resolve_bwd_runner(
        grad_output, x, weight, config_key
    )
    grad_weight = torch.empty_like(weight_c)
    runner.dw(grad_output_c.T, x_c.T, grad_weight)
    grad_weight._base_inputs = (grad_output_c, x_c, weight_c)
    return grad_weight


@torch.library.custom_op(
    "rtx::mxfp8_linear_bwd",
    mutates_args=(),
    device_types="cuda",
)
def _mxfp8_linear_bwd_op(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    config_key: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _launch_bwd(grad_output, x, weight, config_key)


@_mxfp8_linear_bwd_op.register_fake
def _mxfp8_linear_bwd_fake(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    config_key: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(x), torch.empty_like(weight)


@torch.library.custom_op(
    "rtx::mxfp8_linear_dx",
    mutates_args=(),
    device_types="cuda",
)
def _mxfp8_linear_dx_op(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    config_key: str,
) -> torch.Tensor:
    return _launch_dx(grad_output, x, weight, config_key)


@_mxfp8_linear_dx_op.register_fake
def _mxfp8_linear_dx_fake(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    config_key: str,
) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op(
    "rtx::mxfp8_linear_dw",
    mutates_args=(),
    device_types="cuda",
)
def _mxfp8_linear_dw_op(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    config_key: str,
) -> torch.Tensor:
    return _launch_dw(grad_output, x, weight, config_key)


@_mxfp8_linear_dw_op.register_fake
def _mxfp8_linear_dw_fake(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    config_key: str,
) -> torch.Tensor:
    return torch.empty_like(weight)


@torch.library.custom_op(
    "rtx::mxfp8_linear_train",
    mutates_args=(),
    device_types="cuda",
)
def _mxfp8_linear_train_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    forward_config_key: str,
    backward_config_key: str,
) -> torch.Tensor:
    from .fp8 import _launch_training_forward

    return _launch_training_forward(x, weight, forward_config_key)


@_mxfp8_linear_train_op.register_fake
def _mxfp8_linear_train_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    forward_config_key: str,
    backward_config_key: str,
) -> torch.Tensor:
    return torch.empty(
        (x.shape[0], weight.shape[0]), dtype=torch.bfloat16, device=x.device
    )


def _setup_train_context(ctx, inputs, output) -> None:
    x, weight, _forward_config_key, backward_config_key = inputs
    ctx.save_for_backward(x, weight)
    ctx.backward_config_key = backward_config_key


def _train_backward(ctx, grad_output: torch.Tensor):
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
    "rtx::mxfp8_linear_train",
    _train_backward,
    setup_context=_setup_train_context,
)


def mxfp8_linear_backward(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    config: MXFP8BwdConfig | None = None,
    autotune: AutotuneMode | bool | None = None,
    tuning_policy: "CoordinateDescentPolicy | None" = None,
    autotune_cache_dir: Path | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute BF16 dX and dW using independently quantized MXFP8 GEMMs.

    Leading dimensions of ``x`` and ``grad_output`` are flattened into the
    token/reduction dimension.  The initial executable family requires both
    that dimension and ``out_features`` to be divisible by 128 when using the
    default native-scale/TMA GEMM configuration.
    """

    if x.ndim < 1 or grad_output.ndim < 1 or weight.ndim != 2:
        raise ValueError("expected x/grad_output with leading dims and a 2D weight")
    if x.shape[:-1] != grad_output.shape[:-1]:
        raise ValueError("x and grad_output leading dimensions must match")
    if x.shape[-1] != weight.shape[1] or grad_output.shape[-1] != weight.shape[0]:
        raise ValueError("x, grad_output, and weight feature dimensions do not match")
    leading = x.shape[:-1]
    x_2d = x.reshape(-1, x.shape[-1])
    grad_2d = grad_output.reshape(-1, grad_output.shape[-1])
    selected = config
    if selected is None:
        mode: AutotuneMode
        if isinstance(autotune, bool):
            mode = "coordinate" if autotune else "off"
        else:
            mode = (
                os.getenv("RTX_MXFP8_BWD_AUTOTUNE", "cache")
                if autotune is None
                else autotune
            )
        if mode not in ("off", "cache", "coordinate"):
            raise ValueError(
                "backward autotune must be off, cache, or coordinate; "
                f"got {mode!r}"
            )
        if mode == "off" or torch.compiler.is_compiling():
            selected = DEFAULT_MXFP8_BWD_CONFIG
        else:
            from .bwd_autotune import (
                load_cached_mxfp8_bwd_config,
                tune_mxfp8_backward,
            )

            problem = MXFP8Problem(
                int(x_2d.shape[0]), int(weight.shape[0]), int(x_2d.shape[1])
            )
            cached = load_cached_mxfp8_bwd_config(
                problem,
                device=x.device,
                cache_dir=autotune_cache_dir,
            )
            if cached is not None:
                selected = cached
            elif mode == "coordinate":
                selected = tune_mxfp8_backward(
                    grad_2d,
                    x_2d,
                    weight,
                    policy=tuning_policy,
                    cache_dir=autotune_cache_dir,
                ).config
            else:
                selected = DEFAULT_MXFP8_BWD_CONFIG
    key = (
        _DEFAULT_BWD_KEY
        if selected == DEFAULT_MXFP8_BWD_CONFIG
        else _intern_bwd_config(selected)
    )
    grad_x, grad_weight = _mxfp8_linear_bwd_op(grad_2d, x_2d, weight, key)
    return grad_x.reshape(*leading, x.shape[-1]), grad_weight


__all__ = [
    "DEFAULT_MXFP8_BWD_CONFIG",
    "MXFP8BwdConfig",
    "MXFP8BwdMatmulConfig",
    "mxfp8_linear_backward",
]
