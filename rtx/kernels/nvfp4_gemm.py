"""Pure prequantized NVFP4 GEMM for RTX Blackwell SM120/SM121."""

from __future__ import annotations

from functools import lru_cache
import os

os.environ.setdefault("CUTE_DSL_ARCH", "sm_120a")
os.environ.setdefault("QUACK_ARCH", "sm_120a")

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float4E2M1FN, Float8E4M3FN
from cutlass.cute.nvgpu import warp

from ..configs.nvfp4 import NVFP4GemmConfig, NVFP4Problem, NVFP4_SF_VEC_SIZE
from .mxfp8_gemm import MXFP8GemmKernel


class NVFP4GemmKernel(MXFP8GemmKernel):
    """SM120 warp-MMA GEMM with packed E2M1 and E4M3 block scales."""

    def __init__(self, problem: NVFP4Problem, config: NVFP4GemmConfig):
        super().__init__(problem, config)  # type: ignore[arg-type]
        self.ab_dtype = Float4E2M1FN
        self.sf_dtype = Float8E4M3FN
        self.sf_vec_size = NVFP4_SF_VEC_SIZE
        self.use_mxf8f6f4 = False
        self.apply_output_scale = True

    def _make_mma_op(self):
        return warp.MmaMXF4NVF4Op(
            Float4E2M1FN,
            cutlass.Float32,
            Float8E4M3FN,
        )


class NVFP4BlockGemmKernel(NVFP4GemmKernel):
    """Block-only GEMM whose unit output scale is folded out of device IR."""

    def __init__(self, problem: NVFP4Problem, config: NVFP4GemmConfig):
        super().__init__(problem, config)
        self.apply_output_scale = False


class _UnitOutputScaleLauncher:
    def __init__(self, compiled):
        self.compiled = compiled

    def __call__(self, qx, qw, sx, sw, out):
        return self.compiled(qx, qw, sx, sw, out, out)


@lru_cache(maxsize=None)
def compile_nvfp4_gemm(
    problem: NVFP4Problem,
    config: NVFP4GemmConfig = NVFP4GemmConfig(),
):
    """Compile one shape/config-specific native SM120 NVFP4 GEMM."""

    kernel = NVFP4GemmKernel(problem, config)
    qx = cute.runtime.make_fake_tensor(
        Float4E2M1FN,
        (problem.m, problem.k),
        stride=(problem.k, 1),
        assumed_align=16,
    )
    qw = cute.runtime.make_fake_tensor(
        Float4E2M1FN,
        (problem.n, problem.k),
        stride=(problem.k, 1),
        assumed_align=16,
    )
    sx = cute.runtime.make_fake_tensor(
        Float8E4M3FN,
        (problem.m, problem.k // NVFP4_SF_VEC_SIZE),
        stride=(problem.k // NVFP4_SF_VEC_SIZE, 1),
        assumed_align=16,
    )
    sw = cute.runtime.make_fake_tensor(
        Float8E4M3FN,
        (problem.n, problem.k // NVFP4_SF_VEC_SIZE),
        stride=(problem.k // NVFP4_SF_VEC_SIZE, 1),
        assumed_align=16,
    )
    out = cute.runtime.make_fake_tensor(
        BFloat16,
        (problem.m, problem.n),
        stride=(problem.n, 1),
        assumed_align=16,
    )
    output_scale = cute.runtime.make_fake_tensor(
        cutlass.Float32,
        (1,),
        stride=(1,),
        assumed_align=4,
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel,
        qx,
        qw,
        sx,
        sw,
        out,
        output_scale,
        stream,
        options=(
            "--enable-tvm-ffi --opt-level 3 "
            f"--ptxas-options '-O3 -v --maxrregcount={config.maxrregcount}'"
        ),
    )


@lru_cache(maxsize=None)
def compile_nvfp4_block_gemm(
    problem: NVFP4Problem,
    config: NVFP4GemmConfig = NVFP4GemmConfig(),
):
    """Compile block-only NVFP4 GEMM with a five-tensor host ABI."""

    kernel = NVFP4BlockGemmKernel(problem, config)
    qx = cute.runtime.make_fake_tensor(
        Float4E2M1FN,
        (problem.m, problem.k),
        stride=(problem.k, 1),
        assumed_align=16,
    )
    qw = cute.runtime.make_fake_tensor(
        Float4E2M1FN,
        (problem.n, problem.k),
        stride=(problem.k, 1),
        assumed_align=16,
    )
    sx = cute.runtime.make_fake_tensor(
        Float8E4M3FN,
        (problem.m, problem.k // NVFP4_SF_VEC_SIZE),
        stride=(problem.k // NVFP4_SF_VEC_SIZE, 1),
        assumed_align=16,
    )
    sw = cute.runtime.make_fake_tensor(
        Float8E4M3FN,
        (problem.n, problem.k // NVFP4_SF_VEC_SIZE),
        stride=(problem.k // NVFP4_SF_VEC_SIZE, 1),
        assumed_align=16,
    )
    out = cute.runtime.make_fake_tensor(
        BFloat16,
        (problem.m, problem.n),
        stride=(problem.n, 1),
        assumed_align=16,
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return _UnitOutputScaleLauncher(
        cute.compile(
            kernel,
            qx,
            qw,
            sx,
            sw,
            out,
            out,
            stream,
            options=(
                "--enable-tvm-ffi --opt-level 3 "
                f"--ptxas-options '-O3 -v --maxrregcount={config.maxrregcount}'"
            ),
        )
    )


__all__ = [
    "NVFP4GemmConfig",
    "NVFP4GemmKernel",
    "compile_nvfp4_block_gemm",
    "compile_nvfp4_gemm",
]
