"""Fused BF16 -> NVFP4 -> BF16 linear forward for RTX Blackwell."""

from __future__ import annotations

from functools import lru_cache
import os

os.environ.setdefault("CUTE_DSL_ARCH", "sm_120a")
os.environ.setdefault("QUACK_ARCH", "sm_120a")

import cutlass.cute as cute
import cuda.bindings.driver as cuda
import cutlass
from cutlass import BFloat16, Float32

from ..configs.nvfp4 import NVFP4FwdConfig, NVFP4Problem
from .mxfp8_fwd import MXFP8LinearFwdKernel


class NVFP4LinearFwdKernel(MXFP8LinearFwdKernel):
    """Native fused NVFP4 specialization of the cooperative SM120 pipeline."""

    def __init__(self, problem: NVFP4Problem, config: NVFP4FwdConfig):
        super().__init__(problem, config, nvfp4=True)  # type: ignore[arg-type]


@lru_cache(maxsize=None)
def compile_nvfp4_fwd(
    problem: NVFP4Problem,
    config: NVFP4FwdConfig,
):
    problem.validate()
    kernel = NVFP4LinearFwdKernel(problem, config)
    x = cute.runtime.make_fake_tensor(
        BFloat16,
        (problem.m, problem.k),
        stride=(problem.k, 1),
        assumed_align=16,
    )
    weight = cute.runtime.make_fake_tensor(
        BFloat16,
        (problem.n, problem.k),
        stride=(problem.k, 1),
        assumed_align=16,
    )
    out = cute.runtime.make_fake_tensor(
        BFloat16,
        (problem.m, problem.n),
        stride=(problem.n, 1),
        assumed_align=16,
    )
    telemetry_slots = (
        1 if config.telemetry_layout == "scalar_atomic" else kernel.grid_ctas
    )
    state_values = telemetry_slots * config.amax_history_len
    scale_values = state_values if config.collect_amax else 3
    tensor_scale = cute.runtime.make_fake_tensor(
        Float32, (scale_values,), stride=(1,), assumed_align=4
    )
    telemetry_values = state_values if config.collect_amax else 1
    amax = cute.runtime.make_fake_tensor(
        Float32, (telemetry_values,), stride=(1,), assumed_align=4
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel,
        x,
        weight,
        out,
        tensor_scale,
        tensor_scale,
        amax,
        amax,
        stream,
        options=(
            "--enable-tvm-ffi --opt-level 3 "
            f"--ptxas-options '-O3 -v --maxrregcount={config.maxrregcount}'"
        ),
    )
def nvfp4_grid_ctas(problem: NVFP4Problem, config: NVFP4FwdConfig) -> int:
    """Return the exact telemetry length used by one compiled schedule."""

    return NVFP4LinearFwdKernel(problem, config).grid_ctas


def nvfp4_telemetry_values(
    problem: NVFP4Problem, config: NVFP4FwdConfig
) -> int:
    """Return FP32 values in one X or W delayed-history buffer."""

    slots = 1 if config.telemetry_layout == "scalar_atomic" else nvfp4_grid_ctas(
        problem, config
    )
    return slots * config.amax_history_len


__all__ = [
    "NVFP4FwdConfig",
    "NVFP4LinearFwdKernel",
    "compile_nvfp4_fwd",
    "nvfp4_grid_ctas",
    "nvfp4_telemetry_values",
]
