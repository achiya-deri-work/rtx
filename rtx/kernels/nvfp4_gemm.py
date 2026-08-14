"""Pure prequantized NVFP4 GEMM for RTX Blackwell SM120/SM121."""

from __future__ import annotations

from functools import lru_cache
import os

import cuda.bindings.driver as cuda

os.environ.setdefault("CUTE_DSL_ARCH", "sm_120a")
os.environ.setdefault("QUACK_ARCH", "sm_120a")

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float4E2M1FN, Float8E4M3FN, Float32, Int32
from cutlass.cute.nvgpu import warp

from ..configs.nvfp4 import NVFP4GemmConfig, NVFP4Problem, NVFP4_SF_VEC_SIZE
from .mxfp8_gemm import MXFP8GemmKernel


class NVFP4GemmKernel(MXFP8GemmKernel):
    """SM120 warp-MMA GEMM with packed E2M1 and E4M3 block scales."""

    def __init__(
        self,
        problem: NVFP4Problem,
        config: NVFP4GemmConfig,
        *,
        use_pdl: bool = False,
    ):
        super().__init__(problem, config, use_pdl=use_pdl)  # type: ignore[arg-type]
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


class NVFP4JITRegionGemmKernel(NVFP4GemmKernel):
    """NVFP4 GEMM with fused per-row-region FP32 outer-scale epilogue."""

    def __init__(
        self,
        problem: NVFP4Problem,
        config: NVFP4GemmConfig,
        x_region_rows: int,
        weight_region_rows: int,
        use_pdl: bool = False,
    ):
        if x_region_rows < 1 or weight_region_rows < 1:
            raise ValueError("JIT NVFP4 region rows must be positive")
        self.cache_regional_output_scales = True
        self.output_x_scale_cache_elems = (
            config.tile_m + x_region_rows - 1
        ) // x_region_rows
        self.output_w_scale_cache_elems = (
            config.tile_n + weight_region_rows - 1
        ) // weight_region_rows
        product_count = (
            self.output_x_scale_cache_elems
            * self.output_w_scale_cache_elems
        )
        # A complete 2-D product cache removes one SMEM load and one multiply
        # from every output element, but fine row geometries can require a
        # 16-KiB table and exceed SM120's launch-time SMEM limit.  Keep the
        # compact fast path and retain factorized caches above this bound.
        # Revision 3 uses either one tile-uniform register scale or direct
        # read-only scale loads. A CTA-wide product table reintroduced the
        # synchronization bottleneck and is deliberately disabled.
        self.cache_output_scale_products = False
        self.output_scale_product_cache_elems = 0
        self.tile_uniform_regional_output_scale = (
            x_region_rows >= config.tile_m
            and weight_region_rows >= config.tile_n
        )
        self.direct_regional_output_scales = (
            not self.tile_uniform_regional_output_scale
        )
        super().__init__(problem, config, use_pdl=use_pdl)
        self.apply_output_scale = False
        self.x_region_rows = x_region_rows
        self.weight_region_rows = weight_region_rows
        self.x_regions = (problem.m + x_region_rows - 1) // x_region_rows

    @cute.jit
    def _epilogue_output_scale(
        self,
        output_scale: cute.Tensor,
        row: Int32,
        column: Int32,
        global_output_scale: Float32,
        cached_x_scales: cute.Tensor,
        cached_weight_scales: cute.Tensor,
        cached_scale_products: cute.Tensor,
        tile_row: Int32,
        tile_column: Int32,
    ):
        x_index = (row - tile_row) // self.x_region_rows
        weight_index = (column - tile_column) // self.weight_region_rows
        if cutlass.const_expr(self.tile_uniform_regional_output_scale):
            return (
                self._load_x_output_scale(output_scale, row)
                * self._load_weight_output_scale(output_scale, column)
            )
        if cutlass.const_expr(self.direct_regional_output_scales):
            return (
                self._load_x_output_scale(output_scale, row)
                * self._load_weight_output_scale(output_scale, column)
            )
        if cutlass.const_expr(self.cache_output_scale_products):
            return Float32(
                cached_scale_products[
                    x_index * self.output_w_scale_cache_elems + weight_index
                ]
            )
        return (
            Float32(cached_x_scales[x_index])
            * Float32(cached_weight_scales[weight_index])
        )

    @cute.jit
    def _load_x_output_scale(self, output_scale: cute.Tensor, row: Int32):
        return Float32(output_scale[row // self.x_region_rows])

    @cute.jit
    def _load_weight_output_scale(
        self, output_scale: cute.Tensor, column: Int32
    ):
        return Float32(
            output_scale[
                self.x_regions + column // self.weight_region_rows
            ]
        )

    @cute.jit
    def _x_output_scale_cache_offset(self, cache_index: Int32):
        return cache_index * self.x_region_rows

    @cute.jit
    def _w_output_scale_cache_offset(self, cache_index: Int32):
        return cache_index * self.weight_region_rows

    @cute.jit
    def _scale_tma_accumulator(
        self,
        value: Float32,
        cached_x_scales: cute.Tensor,
        cached_weight_scales: cute.Tensor,
        cached_scale_products: cute.Tensor,
        tile_output_scale: Float32,
        output_scale: cute.Tensor,
        global_row: Int32,
        global_column: Int32,
        local_row: Int32,
        local_column: Int32,
    ):
        if cutlass.const_expr(self.tile_uniform_regional_output_scale):
            return value * tile_output_scale
        if cutlass.const_expr(self.direct_regional_output_scales):
            return (
                value
                * self._load_x_output_scale(output_scale, global_row)
                * self._load_weight_output_scale(output_scale, global_column)
            )
        x_index = local_row // self.x_region_rows
        weight_index = local_column // self.weight_region_rows
        if cutlass.const_expr(self.cache_output_scale_products):
            product_index = (
                x_index * self.output_w_scale_cache_elems + weight_index
            )
            return value * Float32(cached_scale_products[product_index])
        return (
            value
            * Float32(cached_x_scales[x_index])
            * Float32(cached_weight_scales[weight_index])
        )


class _UnitOutputScaleLauncher:
    def __init__(self, compiled):
        self.compiled = compiled

    def __call__(self, qx, qw, sx, sw, out):
        return self.compiled(qx, qw, sx, sw, out, out)


class NVFP4RegionRescaleKernel:
    """Vectorized in-place application of local FP32 outer scales."""

    def __init__(
        self,
        m: int,
        n: int,
        x_region_rows: int,
        weight_region_rows: int,
        values_per_thread: int,
        num_warps: int,
        waves: int,
    ):
        self.m = m
        self.n = n
        self.x_region_rows = x_region_rows
        self.weight_region_rows = weight_region_rows
        self.x_regions = cute.ceil_div(m, x_region_rows)
        self.values_per_thread = values_per_thread
        self.threads = num_warps * 32
        natural = cute.ceil_div(m * n, self.threads * values_per_thread)
        sm_count = cutlass.utils.HardwareInfo().get_device_multiprocessor_count()
        self.grid = min(natural, sm_count * waves)

    @cute.jit
    def __call__(
        self,
        out: cute.Tensor,
        region_scales: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(out, region_scales).launch(
            grid=(self.grid, 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, out: cute.Tensor, region_scales: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        linear_thread = block_idx * self.threads + tidx
        thread_stride = self.grid * self.threads
        total = self.m * self.n
        for base in cutlass.range(
            linear_thread * self.values_per_thread,
            total,
            thread_stride * self.values_per_thread,
            unroll=1,
        ):
            row = base // self.n
            x_scale = Float32(region_scales[row // self.x_region_rows])
            for vec in cutlass.range_constexpr(self.values_per_thread):
                linear = base + vec
                if linear < total:
                    column = linear % self.n
                    weight_scale = Float32(
                        region_scales[
                            self.x_regions
                            + column // self.weight_region_rows
                        ]
                    )
                    out[row, column] = BFloat16(
                        Float32(out[row, column]) * x_scale * weight_scale
                    )


def _fake_nvfp4_scales(
    rows: int,
    k: int,
    config: NVFP4GemmConfig,
):
    if config.scale_layout == "row_major":
        return cute.runtime.make_fake_tensor(
            Float8E4M3FN,
            (rows, k // NVFP4_SF_VEC_SIZE),
            stride=(k // NVFP4_SF_VEC_SIZE, 1),
            assumed_align=16,
        )
    return cute.runtime.make_fake_tensor(
        Float8E4M3FN,
        (rows // 128, k // 128, 1024),
        stride=(k // 128 * 1024, 1024, 1),
        assumed_align=16,
    )


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
    sx = _fake_nvfp4_scales(problem.m, problem.k, config)
    sw = _fake_nvfp4_scales(problem.n, problem.k, config)
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
    sx = _fake_nvfp4_scales(problem.m, problem.k, config)
    sw = _fake_nvfp4_scales(problem.n, problem.k, config)
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


@lru_cache(maxsize=None)
def compile_nvfp4_jit_region_gemm(
    problem: NVFP4Problem,
    config: NVFP4GemmConfig,
    x_region_rows: int,
    weight_region_rows: int,
    use_pdl: bool = False,
):
    """Compile native NVFP4 GEMM with local-current outer-scale fusion."""

    kernel = NVFP4JITRegionGemmKernel(
        problem, config, x_region_rows, weight_region_rows, use_pdl
    )
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
    sx = _fake_nvfp4_scales(problem.m, problem.k, config)
    sw = _fake_nvfp4_scales(problem.n, problem.k, config)
    out = cute.runtime.make_fake_tensor(
        BFloat16,
        (problem.m, problem.n),
        stride=(problem.n, 1),
        assumed_align=16,
    )
    region_scales = cute.runtime.make_fake_tensor(
        cutlass.Float32,
        (
            (problem.m + x_region_rows - 1) // x_region_rows
            + (problem.n + weight_region_rows - 1) // weight_region_rows,
        ),
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
        region_scales,
        stream,
        options=(
            "--enable-tvm-ffi --opt-level 3 "
            f"--ptxas-options '-O3 -v --maxrregcount={config.maxrregcount}'"
        ),
    )


@lru_cache(maxsize=None)
def compile_nvfp4_region_rescale(
    m: int,
    n: int,
    x_region_rows: int,
    weight_region_rows: int,
    values_per_thread: int = 8,
    num_warps: int = 8,
    waves: int = 4,
):
    if values_per_thread not in (1, 2, 4, 8, 16):
        raise ValueError("regional rescale vector must be 1, 2, 4, 8, or 16")
    if num_warps not in (4, 8, 16):
        raise ValueError("regional rescale requires 4, 8, or 16 warps")
    kernel = NVFP4RegionRescaleKernel(
        m,
        n,
        x_region_rows,
        weight_region_rows,
        values_per_thread,
        num_warps,
        waves,
    )
    out = cute.runtime.make_fake_tensor(
        BFloat16, (m, n), stride=(n, 1), assumed_align=16
    )
    scales = cute.runtime.make_fake_tensor(
        Float32,
        (
            cute.ceil_div(m, x_region_rows)
            + cute.ceil_div(n, weight_region_rows),
        ),
        stride=(1,),
        assumed_align=4,
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel,
        out,
        scales,
        stream,
        options="--enable-tvm-ffi --opt-level 3",
    )


__all__ = [
    "NVFP4GemmConfig",
    "NVFP4GemmKernel",
    "NVFP4JITRegionGemmKernel",
    "NVFP4RegionRescaleKernel",
    "compile_nvfp4_block_gemm",
    "compile_nvfp4_gemm",
    "compile_nvfp4_jit_region_gemm",
    "compile_nvfp4_region_rescale",
]
