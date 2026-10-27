"""Pure prequantized NVFP4 GEMM for RTX Blackwell SM120/SM121."""

from __future__ import annotations

from functools import lru_cache
import math
import os

import cuda.bindings.driver as cuda

os.environ.setdefault("CUTE_DSL_ARCH", "sm_120a")
os.environ.setdefault("QUACK_ARCH", "sm_120a")

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import BFloat16, Float4E2M1FN, Float8E4M3FN, Float32, Int32
from cutlass.cute.nvgpu import warp
from cutlass.experimental.primitives import nvvm_wrapper as nvvm

from ..configs.nvfp4 import NVFP4GemmConfig, NVFP4Problem, NVFP4_SF_VEC_SIZE
from .mxfp8_gemm import MXFP8GemmKernel
from .mxfp8_reduce import MXFP8WorkspaceReduceKernel
from ._layout_contracts import preserve_tma_singletons


class NVFP4GemmKernel(MXFP8GemmKernel):
    """SM120 warp-MMA GEMM with packed E2M1 and E4M3 block scales."""

    def __init__(
        self,
        problem: NVFP4Problem,
        config: NVFP4GemmConfig,
        *,
        use_pdl: bool = False,
        split_reduction: int = 1,
        reduction_tile: int = 0,
        outer_factor_mode: str = "product_kernel",
    ):
        if outer_factor_mode not in ("product_kernel", "pair"):
            raise ValueError("invalid outer factor mode")
        self.outer_factor_mode=outer_factor_mode
        self.warp_specialized_epilogue = (
            config.regional_epilogue_schedule == "warp_specialized"
        )
        self.num_epilogue_warps = config.num_epilogue_warps
        self.regional_scale_cache_stages = (
            2 if self.warp_specialized_epilogue else 1
        )
        self.epilogue_accumulator_elements = (
            config.tile_m * config.tile_n
            if self.warp_specialized_epilogue
            else 0
        )
        super().__init__(problem, config, use_pdl=use_pdl,
                         split_reduction=split_reduction,
                         reduction_tile=reduction_tile)  # type: ignore[arg-type]
        self.ab_dtype = Float4E2M1FN
        self.sf_dtype = Float8E4M3FN
        self.sf_vec_size = NVFP4_SF_VEC_SIZE
        self.use_mxf8f6f4 = False
        self.apply_output_scale = True

    @cute.jit
    def _load_global_output_scale(self, output_scale: cute.Tensor):
        factor=Float32(output_scale[0])
        if cutlass.const_expr(self.outer_factor_mode=="pair"):
            # Preserve the separately rounded FP32 product before accumulator
            # scaling; never reassociate this as (accumulator * A) * B.
            factor=nvvm.inline_ptx_hl("mul.rn.f32 {$w0}, {$r0}, {$r1};",
                write_only_types=[Float32],read_only_args=[factor,Float32(output_scale[1])])
        return factor

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


class NVFP4SplitGemmKernel(NVFP4GemmKernel):
    """Research split-K: raw FP32 partials, no outer factor or BF16 cast.

    The base split traversal owns disjoint [split,M,N] workspace slices. Its
    direct split epilogue writes FP32 accumulators, unlike the unsplit BF16
    epilogue. A separate reduction applies the caller's outer factor once.
    """

    def __init__(self, problem, config, *, split_reduction, reduction_tile):
        if type(split_reduction) is not int or split_reduction not in (2, 4, 8, 16, 32):
            raise ValueError("NVFP4 workspace split count must be 2, 4, 8, 16, or 32")
        if type(reduction_tile) is not int:
            raise ValueError("NVFP4 reduction tile must be an integer")
        super().__init__(problem, config, split_reduction=split_reduction,
                         reduction_tile=reduction_tile)
        # Also folds out the common loader's otherwise-unused output_scale[0]
        # read. The partial compiler has no alpha tensor in its launcher ABI.
        self.apply_output_scale = False
        self.workspace_elements = split_reduction * problem.m * problem.n
        self.workspace_bytes = self.workspace_elements * 4


class NVFP4SplitWorkspaceReduceKernel(MXFP8WorkspaceReduceKernel):
    """Existing FP32 workspace-reduction policy with one scaled final cast.

    Do not use the unmodified MXFP8 reducer and rescale afterward: that would
    introduce a BF16 intermediate. Only this final epilogue consumes alpha.
    """

    def __init__(self,*args,outer_factor_mode="product_kernel",**kwargs):
        if outer_factor_mode not in ("product_kernel", "pair"):
            raise ValueError("invalid outer factor mode")
        self.outer_factor_mode=outer_factor_mode
        super().__init__(*args,**kwargs)

    @cute.jit
    def __call__(self, workspace: cute.Tensor, out: cute.Tensor,
                 output_scale: cute.Tensor, stream: cuda.CUstream):
        elements = self.m * self.n
        grid_ctas = cute.ceil_div(elements, self.threads * self.vector)
        if cutlass.const_expr(self.algorithm == "persistent_tree"):
            sm_count = utils.HardwareInfo().get_device_multiprocessor_count()
            grid_ctas = cutlass.min(grid_ctas, sm_count * self.persistent_waves)
        self.kernel(workspace, out, output_scale).launch(
            grid=(grid_ctas, 1, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, workspace: cute.Tensor, out: cute.Tensor, output_scale: cute.Tensor):
        thread, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        grid, _, _ = cute.arch.grid_dim()
        elements = self.m * self.n
        out_flat = cute.make_tensor(out.iterator, cute.make_layout((elements,), stride=(1,)))
        factor = Float32(output_scale[0])
        if cutlass.const_expr(self.outer_factor_mode=="pair"):
            factor=nvvm.inline_ptx_hl("mul.rn.f32 {$w0}, {$r0}, {$r1};",
                write_only_types=[Float32],read_only_args=[factor,Float32(output_scale[1])])
        first = (block * self.threads + thread) * self.vector
        step = grid * self.threads * self.vector
        for base in cutlass.range(first, elements, step, unroll=1):
            for lane in cutlass.range_constexpr(self.vector):
                index = base + lane
                if index < elements:
                    values = [Float32(0.)] * self.split
                    for part in cutlass.range_constexpr(self.split):
                        values[part] = workspace[part * elements + index]
                    if cutlass.const_expr(self.algorithm == "serial"):
                        total = Float32(0.)
                        for part in cutlass.range_constexpr(self.split):
                            total += values[part]
                    else:
                        width = self.split
                        for _level in cutlass.range_constexpr(int(math.log2(self.split))):
                            for pair in cutlass.range_constexpr(width // 2):
                                values[pair] = values[2 * pair] + values[2 * pair + 1]
                            width //= 2
                        total = values[0]
                    scaled = nvvm.inline_ptx_hl("mul.rn.f32 {$w0}, {$r0}, {$r1};",
                        write_only_types=[Float32], read_only_args=[total, factor])
                    out_flat[index] = BFloat16(scaled)


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
        # A tile may start in the middle of a region. Account for the leading
        # partial region as well as the trailing one; ceil(tile/region) alone
        # under-allocates for non-divisor geometries such as 128/5.
        self.output_x_scale_cache_elems = (
            config.tile_m + x_region_rows - 2
        ) // x_region_rows + 1
        self.output_w_scale_cache_elems = (
            config.tile_n + weight_region_rows - 2
        ) // weight_region_rows + 1
        product_count = (
            self.output_x_scale_cache_elems
            * self.output_w_scale_cache_elems
        )
        strategy = config.regional_scale_epilogue
        self.expanded_factor_output_scales = strategy == "expanded_factors"
        self.fragment_register_output_scales = (
            strategy == "fragment_registers"
        )
        self.cache_output_scale_products = strategy == "product"
        self.output_scale_product_cache_elems = (
            product_count if self.cache_output_scale_products else 0
        )
        self.tile_uniform_regional_output_scale = (
            not self.expanded_factor_output_scales
            and x_region_rows % config.tile_m == 0
            and weight_region_rows % config.tile_n == 0
        )
        self.direct_regional_output_scales = (
            strategy in ("direct", "expanded_factors", "fragment_registers")
            and not self.tile_uniform_regional_output_scale
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
        if cutlass.const_expr(self.expanded_factor_output_scales):
            return Float32(output_scale[row]) * Float32(
                output_scale[self.problem.m + column]
            )
        x_index = (
            row // self.x_region_rows - tile_row // self.x_region_rows
        )
        weight_index = (
            column // self.weight_region_rows
            - tile_column // self.weight_region_rows
        )
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
        # The common loader adds this to the tile origin and then resolves the
        # global region. Offset zero loads the leading partial region; adding
        # one full region advances exactly one region even for misaligned CTAs.
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
        if cutlass.const_expr(self.expanded_factor_output_scales):
            return (
                value
                * Float32(output_scale[global_row])
                * Float32(output_scale[self.problem.m + global_column])
            )
        if cutlass.const_expr(self.tile_uniform_regional_output_scale):
            return value * tile_output_scale
        if cutlass.const_expr(self.direct_regional_output_scales):
            return (
                value
                * self._load_x_output_scale(output_scale, global_row)
                * self._load_weight_output_scale(output_scale, global_column)
            )
        x_index = (
            global_row // self.x_region_rows
            - (global_row - local_row) // self.x_region_rows
        )
        weight_index = (
            global_column // self.weight_region_rows
            - (global_column - local_column) // self.weight_region_rows
        )
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
            linear_thread,
            total,
            thread_stride * self.values_per_thread,
            unroll=1,
        ):
            for vec in cutlass.range_constexpr(self.values_per_thread):
                # Strip-mine by the complete launch grid.  For a fixed
                # ``vec`` adjacent lanes now touch adjacent BF16 values;
                # the old thread-contiguous mapping made scalar instructions
                # from neighboring lanes 16 values apart and turned a simple
                # bandwidth pass into heavily scattered memory traffic.
                linear = base + vec * thread_stride
                if linear < total:
                    row = linear // self.n
                    column = linear % self.n
                    x_scale = Float32(
                        region_scales[row // self.x_region_rows]
                    )
                    weight_scale = Float32(
                        region_scales[
                            self.x_regions
                            + column // self.weight_region_rows
                        ]
                    )
                    out[row, column] = BFloat16(
                        Float32(out[row, column]) * x_scale * weight_scale
                    )


class NVFP4RegionExpandKernel:
    """Expand compact regional factors to division-free row/column arrays."""

    def __init__(
        self,
        m: int,
        n: int,
        x_region_rows: int,
        weight_region_rows: int,
    ):
        self.m = m
        self.n = n
        self.x_region_rows = x_region_rows
        self.weight_region_rows = weight_region_rows
        x_regions = cute.ceil_div(m, x_region_rows)
        self.x_regions = x_regions
        self.count = m + n
        self.threads = 256

    @cute.jit
    def __call__(
        self,
        factors: cute.Tensor,
        expanded: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(factors, expanded).launch(
            grid=(cute.ceil_div(self.count, self.threads), 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, factors: cute.Tensor, expanded: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        index = block_idx * self.threads + tidx
        if index < self.count:
            if index < self.m:
                expanded[index] = Float32(
                    factors[index // self.x_region_rows]
                )
            else:
                column = index - self.m
                expanded[index] = Float32(
                    factors[
                        self.x_regions
                        + column // self.weight_region_rows
                    ]
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
    *,
    outer_factor_mode: str = "product_kernel",
):
    """Compile one shape/config-specific native SM120 NVFP4 GEMM."""

    kernel = NVFP4GemmKernel(problem, config,outer_factor_mode=outer_factor_mode)
    qx = cute.runtime.make_fake_tensor(
        Float4E2M1FN,
        preserve_tma_singletons((problem.m, problem.k)),
        stride=(problem.k, 1),
        assumed_align=16,
    )
    qw = cute.runtime.make_fake_tensor(
        Float4E2M1FN,
        preserve_tma_singletons((problem.n, problem.k)),
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
        (2 if outer_factor_mode=="pair" else 1,),
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
def compile_nvfp4_split_gemm(
    problem: NVFP4Problem,
    config: NVFP4GemmConfig = NVFP4GemmConfig(epilogue="direct", store_vec=1),
    *,
    split_reduction: int,
    reduction_tile: int,
):
    """Research-only ``(qx,qw,sx,sw,fp32_workspace)`` split-K compiler.

    Caller allocates ``split_reduction*M*N`` contiguous FP32 elements. There
    are no atomics, zero-fill requirements, outer factors, or BF16 partials.
    Explicitly select a direct, nonpersistent one-output-tile configuration;
    unsupported schedules are rejected rather than silently rewritten.
    """
    kernel = NVFP4SplitGemmKernel(problem, config, split_reduction=split_reduction,
                                 reduction_tile=reduction_tile)
    qx = cute.runtime.make_fake_tensor(Float4E2M1FN,
        preserve_tma_singletons((problem.m, problem.k)), stride=(problem.k, 1), assumed_align=16)
    qw = cute.runtime.make_fake_tensor(Float4E2M1FN,
        preserve_tma_singletons((problem.n, problem.k)), stride=(problem.k, 1), assumed_align=16)
    sx = _fake_nvfp4_scales(problem.m, problem.k, config)
    sw = _fake_nvfp4_scales(problem.n, problem.k, config)
    workspace = cute.runtime.make_fake_tensor(Float32, (kernel.workspace_elements,),
                                               stride=(1,), assumed_align=16)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled = cute.compile(kernel, qx, qw, sx, sw, workspace, workspace, stream,
        options=("--enable-tvm-ffi --opt-level 3 "
                 f"--ptxas-options '-O3 -v --maxrregcount={config.maxrregcount}'"))
    return _UnitOutputScaleLauncher(compiled)


@lru_cache(maxsize=None)
def compile_nvfp4_split_workspace_reduce(
    m: int,
    n: int,
    split: int,
    *,
    algorithm: str = "tree",
    threads: int = 256,
    vector: int = 4,
    persistent_waves: int = 4,
    outer_factor_mode: str = "product_kernel",
):
    """Research ``(fp32_workspace,bf16_out,fp32_alpha[1])`` final reduction.

    Sum partials in FP32, multiply by alpha once in FP32, then cast once.
    Split-K changes FP32 summation order, not the quantizer/source contract.
    """
    kernel = NVFP4SplitWorkspaceReduceKernel(m, n, split, algorithm=algorithm,
        threads=threads, vector=vector, persistent_waves=persistent_waves,outer_factor_mode=outer_factor_mode)
    workspace = cute.runtime.make_fake_tensor(Float32, (split * m * n,),
        stride=(1,), assumed_align=16)
    out = cute.runtime.make_fake_tensor(BFloat16, (m, n), stride=(n, 1), assumed_align=16)
    output_scale = cute.runtime.make_fake_tensor(Float32, (2 if outer_factor_mode=="pair" else 1,), stride=(1,), assumed_align=4)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(kernel, workspace, out, output_scale, stream,
        options="--enable-tvm-ffi --opt-level 3 --ptxas-options '-O3 -v'")


@lru_cache(maxsize=None)
def compile_nvfp4_block_gemm(
    problem: NVFP4Problem,
    config: NVFP4GemmConfig = NVFP4GemmConfig(),
):
    """Compile block-only NVFP4 GEMM with a five-tensor host ABI."""

    kernel = NVFP4BlockGemmKernel(problem, config)
    qx = cute.runtime.make_fake_tensor(
        Float4E2M1FN,
        preserve_tma_singletons((problem.m, problem.k)),
        stride=(problem.k, 1),
        assumed_align=16,
    )
    qw = cute.runtime.make_fake_tensor(
        Float4E2M1FN,
        preserve_tma_singletons((problem.n, problem.k)),
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
        preserve_tma_singletons((problem.m, problem.k)),
        stride=(problem.k, 1),
        assumed_align=16,
    )
    qw = cute.runtime.make_fake_tensor(
        Float4E2M1FN,
        preserve_tma_singletons((problem.n, problem.k)),
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
    x_regions = (problem.m + x_region_rows - 1) // x_region_rows
    weight_regions = (
        problem.n + weight_region_rows - 1
    ) // weight_region_rows
    scale_values = (
        problem.m + problem.n
        if config.regional_scale_epilogue == "expanded_factors"
        else x_regions + weight_regions
    )
    region_scales = cute.runtime.make_fake_tensor(
        cutlass.Float32,
        (scale_values,),
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
def compile_nvfp4_region_expand(
    m: int,
    n: int,
    x_region_rows: int,
    weight_region_rows: int,
):
    x_regions = cute.ceil_div(m, x_region_rows)
    weight_regions = cute.ceil_div(n, weight_region_rows)
    kernel = NVFP4RegionExpandKernel(
        m, n, x_region_rows, weight_region_rows
    )
    factors = cute.runtime.make_fake_tensor(
        Float32,
        (x_regions + weight_regions,),
        stride=(1,),
        assumed_align=4,
    )
    expanded = cute.runtime.make_fake_tensor(
        Float32,
        (m + n,),
        stride=(1,),
        assumed_align=4,
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel,
        factors,
        expanded,
        stream,
        options="--enable-tvm-ffi --opt-level 3",
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
    "NVFP4SplitGemmKernel",
    "NVFP4SplitWorkspaceReduceKernel",
    "NVFP4JITRegionGemmKernel",
    "NVFP4RegionRescaleKernel",
    "NVFP4RegionExpandKernel",
    "compile_nvfp4_block_gemm",
    "compile_nvfp4_gemm",
    "compile_nvfp4_split_gemm",
    "compile_nvfp4_split_workspace_reduce",
    "compile_nvfp4_jit_region_gemm",
    "compile_nvfp4_region_expand",
    "compile_nvfp4_region_rescale",
]
