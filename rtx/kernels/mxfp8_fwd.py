"""Fused BF16 -> MXFP8 -> BF16 linear forward kernel.

This revision implements a configurable cooperative SM120 schedule.  Both
operands are quantized in CTA shared memory; no quantized tensor or scale is
materialized in global memory.  The matrix multiply is the native Blackwell
``mma.sync.kind::mxf8`` operation with FP32 accumulation.

The performance schedules (TMA/cp.async producer warps, deeper pipelines and
persistent reuse) are represented by :class:`MXFP8FwdConfig` and are added as
separate implementations, rather than changing this numerical baseline.
"""

from __future__ import annotations

import os
import math
from functools import lru_cache

os.environ.setdefault("CUTE_DSL_ARCH", "sm_120a")
os.environ.setdefault("QUACK_ARCH", "sm_120a")

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm120_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass import (
    BFloat16,
    Float8E4M3FN,
    Float8E8M0FNU,
    Float32,
    Int16,
    Int32,
    Uint16,
    Uint8,
)
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.experimental.primitives import nvvm_wrapper as nvvm

from .mxfp8 import MXFP8FwdConfig, MXFP8Problem


SF_VEC_SIZE = 32
F8_MAX = 448.0
F8_MAX_POW2 = 8
F32_MANTISSA_BITS = 23
F32_EXPONENT_BIAS = 127
E8M0_MIN_UNBIASED = -127
E8M0_MAX_UNBIASED = 128


def _make_ldmatrix_atom(
    dtype: type[cutlass.Numeric],
    transpose: bool,
    num_matrices: int,
):
    return cute.make_copy_atom(
        warp.LdMatrix8x8x16bOp(
            transpose=transpose,
            num_matrices=num_matrices,
        ),
        dtype,
    )


def _make_sm120_sfa_layout_64(
    tiled_mma: cute.TiledMma,
    tile_k: int,
    num_stages: int,
):
    """SM120 SFA layout for a logical 64-row tile in a padded 128-row block.

    NVIDIA's public helper materializes ``(32, 4)`` row groups and therefore
    requires M to be divisible by 128.  The first 64 logical rows are exactly
    the first two of those four interleaved groups: ``(32, 2):(16, 4)``.  The
    unchanged strides preserve the hardware's 128-row address encoding while
    the logical extent seen by fragment partitioning is genuinely 64.
    """

    mma_nsf = tiled_mma.shape_mnk[2] // SF_VEC_SIZE
    mn_shape = ((32, 2), 1)
    mn_stride = ((16, 4), 512)
    k_shape = (
        (SF_VEC_SIZE, mma_nsf),
        4 // mma_nsf,
        tile_k // SF_VEC_SIZE // 4,
    )
    k_stride = ((0, 1), mma_nsf, 512)
    layout = cute.make_layout(
        (mn_shape, k_shape),
        stride=(mn_stride, k_stride),
    )
    return cute.append(
        layout,
        cute.make_layout(
            num_stages,
            stride=cute.cosize(cute.filter_zeros(layout)),
        ),
    )


def _make_scale_s2r_atom(dtype: type[cutlass.Numeric], bits: int):
    if bits:
        return cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            dtype,
            num_bits_per_copy=bits,
        )
    return cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), dtype)


class MXFP8LinearFwdKernel:
    """One CTA computes one output tile using cooperative quantization/MMA."""

    def __init__(
        self,
        problem: MXFP8Problem,
        config: MXFP8FwdConfig,
        *,
        a_orientation: str = "row",
        b_orientation: str = "row",
        split_reduction: int = 1,
        reduction_tile: int = 0,
        atomic_output: bool = False,
    ):
        rejection = config.oriented_implementation_rejection(
            problem, a_orientation, b_orientation
        )
        if rejection is not None:
            raise ValueError(f"illegal MXFP8 forward configuration: {rejection}")
        self.problem = problem
        self.config = config
        self.a_orientation = a_orientation
        self.b_orientation = b_orientation
        self.split_reduction = split_reduction
        self.reduction_tile = reduction_tile
        self.atomic_output = atomic_output
        if split_reduction < 1:
            raise ValueError("split_reduction must be positive")
        if split_reduction == 1 and (reduction_tile != 0 or atomic_output):
            raise ValueError("an unsplit kernel must use reduction_tile=0")
        if split_reduction > 1:
            if config.epilogue != "direct":
                raise ValueError("split reduction requires the FP32 direct epilogue")
            if config.persistent:
                raise ValueError("split reduction persistence is tuned in its epilogue")
            if reduction_tile <= 0 or reduction_tile % config.bf16_tile_k:
                raise ValueError(
                    "split reduction tile must be a positive BF16-tile multiple"
                )
            if not (
                (split_reduction - 1) * reduction_tile < problem.k
                <= split_reduction * reduction_tile
            ):
                raise ValueError(
                    "split reduction count/tile must cover K without an empty slice"
                )
        self.tile_shape_mnk = (config.tile_m, config.tile_n, config.tile_k)
        self.num_mma_warps = config.num_mma_warps
        self.threads_per_cta = config.num_threads
        total_tiles = split_reduction * (
            (problem.m + config.tile_m - 1) // config.tile_m
        ) * ((problem.n + config.tile_n - 1) // config.tile_n)
        self.grid_ctas = total_tiles
        if config.persistent:
            sm_count = utils.HardwareInfo().get_device_multiprocessor_count()
            self.grid_ctas = min(
                total_tiles, sm_count * config.persistent_waves
            )
            while total_tiles % self.grid_ctas:
                self.grid_ctas -= 1
        self.work_tiles_per_cta = total_tiles // self.grid_ctas
        self.mma_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.num_mma_warps * 32,
        )
        self.epilogue_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.num_mma_warps * 32,
        )

        self.a_dtype = Float8E4M3FN
        self.b_dtype = Float8E4M3FN
        self.sf_dtype = Float8E8M0FNU
        self.acc_dtype = Float32
        self.c_dtype = Float32 if split_reduction > 1 else BFloat16
        self.a_layout = utils.LayoutEnum.ROW_MAJOR
        self.b_layout = utils.LayoutEnum.ROW_MAJOR

    def _setup_static_layouts(self) -> None:
        cfg = self.config
        mma_op = warp.MmaMXF8Op(self.a_dtype, self.acc_dtype, self.sf_dtype)
        atom_layout = cute.make_layout(
            (cfg.atom_layout_m, cfg.atom_layout_n, 1)
        )
        permutation_mnk = sm120_utils.get_permutation_mnk(
            self.tile_shape_mnk, SF_VEC_SIZE, True
        )
        self.tiled_mma = cute.make_tiled_mma(
            mma_op, atom_layout, permutation_mnk=permutation_mnk
        )

        swizzle_kinds = {
            "none": cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_INTER,
            "32b": cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW32,
            "64b": cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW64,
            "128b": cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW128,
        }
        mn_swizzle_kinds = {
            "none": cute.nvgpu.warpgroup.SmemLayoutAtomKind.MN_INTER,
            "32b": cute.nvgpu.warpgroup.SmemLayoutAtomKind.MN_SW32,
            "64b": cute.nvgpu.warpgroup.SmemLayoutAtomKind.MN_SW64,
            "128b": cute.nvgpu.warpgroup.SmemLayoutAtomKind.MN_SW128,
        }
        # Every choice is a legal K-major ldmatrix atom; unlike the previous
        # heuristic-only path, the bank-conflict swizzle is part of generated
        # shared-memory addressing and therefore a real tuning coordinate.
        a_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            swizzle_kinds[cfg.a_swizzle],
            self.a_dtype,
        )
        b_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            swizzle_kinds[cfg.b_swizzle],
            self.b_dtype,
        )
        self.a_smem_layout = cute.tile_to_shape(
            a_atom,
            (cfg.tile_m, cfg.tile_k, cfg.mxfp8_stages),
            order=(0, 1, 2),
        )
        self.b_smem_layout = cute.tile_to_shape(
            b_atom,
            (cfg.tile_n, cfg.tile_k, cfg.mxfp8_stages),
            order=(0, 1, 2),
        )
        # The SM120 instruction addresses scales in indivisible 128-row/column
        # physical blocks.  A 64-row logical CTA therefore uses a padded SFA
        # allocation while the tiled MMA permutation and copy partitions retain
        # the real 64-row extent.
        scale_layout_tile = (
            ((cfg.tile_m + 127) // 128) * 128,
            ((cfg.tile_n + 127) // 128) * 128,
            cfg.tile_k,
        )
        if cfg.tile_m == 64:
            self.sfa_smem_layout = _make_sm120_sfa_layout_64(
                self.tiled_mma,
                cfg.tile_k,
                cfg.mxfp8_stages,
            )
        else:
            self.sfa_smem_layout = blockscaled_utils.sm120_make_smem_layout_sfa(
                self.tiled_mma,
                scale_layout_tile,
                SF_VEC_SIZE,
                cfg.mxfp8_stages,
            )
        self.sfb_smem_layout = blockscaled_utils.sm120_make_smem_layout_sfb(
            self.tiled_mma,
            scale_layout_tile,
            SF_VEC_SIZE,
            cfg.mxfp8_stages,
        )
        # TMA follows the contiguous basis of each GMEM tensor.  A metadata-only
        # transpose has stride (1, rows), so its staging tile must be MN-major;
        # using a K-major staging atom would reinterpret rather than transpose
        # the tile.  The quantizer indexes either layout logically and still
        # emits the final FP8 MMA operands in their required K-major layouts.
        bf16_a_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            (
                mn_swizzle_kinds[cfg.bf16_swizzle]
                if self.a_orientation == "transpose"
                else swizzle_kinds[cfg.bf16_swizzle]
            ),
            BFloat16,
        )
        bf16_b_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            (
                mn_swizzle_kinds[cfg.bf16_swizzle]
                if self.b_orientation == "transpose"
                else swizzle_kinds[cfg.bf16_swizzle]
            ),
            BFloat16,
        )
        self.bf16_a_smem_layout = cute.tile_to_shape(
            bf16_a_atom,
            (cfg.tile_m, cfg.bf16_tile_k, cfg.bf16_stages),
            order=(0, 1, 2),
        )
        self.bf16_b_smem_layout = cute.tile_to_shape(
            bf16_b_atom,
            (cfg.tile_n, cfg.bf16_tile_k, cfg.bf16_stages),
            order=(0, 1, 2),
        )
        out_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            swizzle_kinds["128b"],
            self.c_dtype,
        )
        self.out_smem_layout = cute.tile_to_shape(
            out_atom,
            (cfg.tile_m, cfg.tile_n, cfg.epilogue_stages),
            order=(0, 1, 2),
        )

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        out: cute.Tensor,
        stream: cuda.CUstream,
    ):
        # Layout/MMA objects are IR values and must be constructed while CuTe is
        # tracing this JIT entry point (not in the plain-Python constructor).
        self._setup_static_layouts()
        cfg = self.config

        bf16_a_stage_layout = cute.slice_(
            self.bf16_a_smem_layout, (None, None, 0)
        )
        bf16_b_stage_layout = cute.slice_(
            self.bf16_b_smem_layout, (None, None, 0)
        )
        tma_atom_x, tma_tensor_x = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            x,
            bf16_a_stage_layout,
            (cfg.tile_m, cfg.bf16_tile_k),
        )
        tma_atom_weight, tma_tensor_weight = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            weight,
            bf16_b_stage_layout,
            (cfg.tile_n, cfg.bf16_tile_k),
        )
        out_stage_layout = cute.slice_(
            self.out_smem_layout, (None, None, 0)
        )
        out_tma_view = cute.make_tensor(
            out.iterator,
            cute.make_layout(
                (self.problem.m, self.problem.n, 1),
                stride=(self.problem.n, 1, self.problem.m * self.problem.n),
            ),
        )
        tma_atom_out, tma_tensor_out = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            out_tma_view,
            out_stage_layout,
            (cfg.tile_m, cfg.tile_n),
        )
        cp_async_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL),
            BFloat16,
            num_bits_per_copy=128,
        )
        cp_async_values = 128 // BFloat16.width
        cp_async_k_threads = cfg.bf16_tile_k // cp_async_values
        cp_async_thread_layout = cute.make_layout(
            (cfg.num_threads // cp_async_k_threads, cp_async_k_threads),
            stride=(cp_async_k_threads, 1),
        )
        cp_async_value_layout = cute.make_layout((1, cp_async_values))
        cp_async_tiled_copy = cute.make_tiled_copy_tv(
            cp_async_atom,
            cp_async_thread_layout,
            cp_async_value_layout,
        )

        @cute.struct
        class SharedStorageStaged:
                bf16_a: cute.struct.Align[
                    cute.struct.MemRange[
                        BFloat16, cute.cosize(self.bf16_a_smem_layout)
                    ],
                    1024,
                ]
                bf16_b: cute.struct.Align[
                    cute.struct.MemRange[
                        BFloat16, cute.cosize(self.bf16_b_smem_layout)
                    ],
                    1024,
                ]
                q_a: cute.struct.Align[
                    cute.struct.MemRange[
                        self.a_dtype, cute.cosize(self.a_smem_layout)
                    ],
                    1024,
                ]
                q_b: cute.struct.Align[
                    cute.struct.MemRange[
                        self.b_dtype, cute.cosize(self.b_smem_layout)
                    ],
                    1024,
                ]
                scale_a: cute.struct.Align[
                    cute.struct.MemRange[
                        self.sf_dtype, cute.cosize(self.sfa_smem_layout)
                    ],
                    128,
                ]
                scale_b: cute.struct.Align[
                    cute.struct.MemRange[
                        self.sf_dtype, cute.cosize(self.sfb_smem_layout)
                    ],
                    128,
                ]
                tma_pipeline: cute.struct.Align[
                    cute.struct.MemRange[
                        cutlass.Int64, cfg.bf16_stages * 2
                    ],
                    8,
                ]
                quant_pipeline: cute.struct.Align[
                    cute.struct.MemRange[
                        cutlass.Int64, cfg.mxfp8_stages * 2
                    ],
                    8,
                ]
                out: cute.struct.Align[
                    cute.struct.MemRange[
                        self.c_dtype,
                        (
                            cute.cosize(self.out_smem_layout)
                            if cfg.epilogue == "tma"
                            else 0
                        ),
                    ],
                    1024,
                ]
        @cute.struct
        class SharedStorageScalar:
                q_a: cute.struct.Align[
                    cute.struct.MemRange[
                        self.a_dtype, cute.cosize(self.a_smem_layout)
                    ],
                    1024,
                ]
                q_b: cute.struct.Align[
                    cute.struct.MemRange[
                        self.b_dtype, cute.cosize(self.b_smem_layout)
                    ],
                    1024,
                ]
                scale_a: cute.struct.Align[
                    cute.struct.MemRange[
                        self.sf_dtype, cute.cosize(self.sfa_smem_layout)
                    ],
                    128,
                ]
                scale_b: cute.struct.Align[
                    cute.struct.MemRange[
                        self.sf_dtype, cute.cosize(self.sfb_smem_layout)
                    ],
                    128,
                ]
                out: cute.struct.Align[
                    cute.struct.MemRange[
                        self.c_dtype,
                        (
                            cute.cosize(self.out_smem_layout)
                            if cfg.epilogue == "tma"
                            else 0
                        ),
                    ],
                    1024,
                ]

        self.shared_storage = SharedStorageScalar
        if cutlass.const_expr(cfg.load_engine != "scalar"):
            self.shared_storage = SharedStorageStaged
        # A linear grid lets raster order and grouped CTA swizzling be genuine
        # compile-time schedule choices without padding the edge group.
        m_tiles = (self.problem.m + cfg.tile_m - 1) // cfg.tile_m
        n_tiles = (self.problem.n + cfg.tile_n - 1) // cfg.tile_n
        grid = (self.grid_ctas, 1, 1)
        self.kernel(
            x,
            weight,
            out,
            tma_atom_x,
            tma_tensor_x,
            tma_atom_weight,
            tma_tensor_weight,
            tma_atom_out,
            tma_tensor_out,
            self.tiled_mma,
            self.a_smem_layout,
            self.b_smem_layout,
            self.sfa_smem_layout,
            self.sfb_smem_layout,
            self.bf16_a_smem_layout,
            self.bf16_b_smem_layout,
            self.out_smem_layout,
            cp_async_tiled_copy,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            stream=stream,
        )

    @cute.jit
    def _scale_from_amax(self, amax: Float32):
        """Return FLOOR-mode E8M0 and its exact FP32 power-of-two inverse."""

        exponent = ((amax.bitcast(Int32) >> F32_MANTISSA_BITS) & 0xFF) - (
            F32_EXPONENT_BIAS + F8_MAX_POW2
        )
        exponent = cutlass.max(exponent, E8M0_MIN_UNBIASED)
        exponent = cutlass.min(exponent, E8M0_MAX_UNBIASED)
        biased = exponent + F32_EXPONENT_BIAS
        if cute.isnan(amax):
            biased = Int32(255)

        scale_e8m0 = Uint8(biased).bitcast(Float8E8M0FNU)

        # Quantization clamps code 0 to the smallest normal FP32 value, exactly
        # like the reference.  The stored E8M0 code itself remains zero.
        fp32_exponent = Int32(1) if biased == 0 else biased
        # Division by an E8M0 scale is multiplication by an exact power of two.
        # Build the reciprocal directly.  E=254 needs the exactly representable
        # subnormal 2^-127; E=255 (NaN scale code / +inf decoded scale) maps to
        # zero.  All other reciprocals are normal powers of two.
        reciprocal_bits = (Int32(254) - fp32_exponent) << F32_MANTISSA_BITS
        if fp32_exponent == 254:
            reciprocal_bits = Int32(1 << (F32_MANTISSA_BITS - 1))
        if fp32_exponent == 255:
            reciprocal_bits = Int32(0)
        inv_scale_fp32 = reciprocal_bits.bitcast(Float32)
        return scale_e8m0, inv_scale_fp32

    @cute.jit
    def _warp_amax(
        self,
        value: Float32,
        threads_in_group: cutlass.Constexpr,
        lane_idx: Int32,
    ):
        """NaN-propagating absolute maximum over a warp or subwarp.

        SM120 does not support the FP32 ``redux.sync.max.abs`` form exposed for
        datacenter Blackwell.  Magnitude FP32 bit patterns are monotonically
        ordered as unsigned integers, however, so ``redux.sync.max.u32`` gives
        the identical amax (including selecting NaN over finite values).  The
        runtime member mask preserves quant_vec's independent subwarp groups.
        """

        maximum_bits = value.bitcast(Int32) & Int32(0x7FFFFFFF)
        if cutlass.const_expr(self.config.reduction == "redux"):
            group_base = lane_idx & Int32(~(threads_in_group - 1))
            member_mask = Int32((1 << threads_in_group) - 1) << group_base
            return nvvm.redux_sync(
                maximum_bits,
                nvvm.ReductionKind.UMAX,
                member_mask,
            ).bitcast(Float32)

        maximum = maximum_bits.bitcast(Float32)
        # PTX shfl's packed clamp selects independent power-of-two segments.
        # This lets one warp quantize 1/2/4/8 independent MX blocks while every
        # lane still owns 1/2/4/8 contiguous BF16 values respectively.
        mask_and_clamp = ((32 - threads_in_group) << 8) | (threads_in_group - 1)
        for level in cutlass.range_constexpr(int(math.log2(threads_in_group))):
            other = cute.arch.shuffle_sync_bfly(
                maximum,
                (threads_in_group // 2) >> level,
                mask_and_clamp=mask_and_clamp,
            )
            maximum = cute.arch.fmax(maximum, other, nan=True)
        return maximum

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        out: cute.Tensor,
        tma_atom_x: cute.CopyAtom,
        tma_tensor_x: cute.Tensor,
        tma_atom_weight: cute.CopyAtom,
        tma_tensor_weight: cute.Tensor,
        tma_atom_out: cute.CopyAtom,
        tma_tensor_out: cute.Tensor,
        tiled_mma: cute.TiledMma,
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        sfa_smem_layout: cute.Layout,
        sfb_smem_layout: cute.Layout,
        bf16_a_smem_layout: cute.ComposedLayout,
        bf16_b_smem_layout: cute.ComposedLayout,
        out_smem_layout: cute.ComposedLayout,
        cp_async_tiled_copy: cute.TiledCopy,
    ):
        cfg = self.config
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = cute.arch.lane_idx()
        linear_tile, _, _ = cute.arch.block_idx()
        m_tiles = cute.ceil_div(self.problem.m, cfg.tile_m)
        n_tiles = cute.ceil_div(self.problem.n, cfg.tile_n)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        s_a = storage.q_a.get_tensor(
            a_smem_layout.outer, swizzle=a_smem_layout.inner
        )
        s_b = storage.q_b.get_tensor(
            b_smem_layout.outer, swizzle=b_smem_layout.inner
        )
        s_sfa = storage.scale_a.get_tensor(sfa_smem_layout)
        s_sfb = storage.scale_b.get_tensor(sfb_smem_layout)
        if cutlass.const_expr(cfg.epilogue == "tma"):
            s_out = storage.out.get_tensor(
                out_smem_layout.outer,
                swizzle=out_smem_layout.inner,
            )

        if cutlass.const_expr(cfg.load_engine != "scalar"):
            s_bf16_a = storage.bf16_a.get_tensor(
                bf16_a_smem_layout.outer, swizzle=bf16_a_smem_layout.inner
            )
            s_bf16_b = storage.bf16_b.get_tensor(
                bf16_b_smem_layout.outer, swizzle=bf16_b_smem_layout.inner
            )
        if cutlass.const_expr(cfg.load_engine == "tma"):
            pipeline_storage = storage.tma_pipeline.data_ptr()
            g_x_tiles = cute.local_tile(
                tma_tensor_x,
                (cfg.tile_m, cfg.bf16_tile_k),
                (None, None),
            )
            g_weight_tiles = cute.local_tile(
                tma_tensor_weight,
                (cfg.tile_n, cfg.bf16_tile_k),
                (None, None),
            )
            t_xs, t_xg = cpasync.tma_partition(
                tma_atom_x,
                0,
                cute.make_layout(1),
                cute.group_modes(s_bf16_a, 0, 2),
                cute.group_modes(g_x_tiles, 0, 2),
            )
            t_ws, t_wg = cpasync.tma_partition(
                tma_atom_weight,
                0,
                cute.make_layout(1),
                cute.group_modes(s_bf16_b, 0, 2),
                cute.group_modes(g_weight_tiles, 0, 2),
            )
            tma_bytes = cute.size_in_bytes(
                BFloat16, cute.slice_(bf16_a_smem_layout, (None, None, 0))
            ) + cute.size_in_bytes(
                BFloat16, cute.slice_(bf16_b_smem_layout, (None, None, 0))
            )
            tma_pipeline = pipeline.PipelineTmaAsync.create(
                num_stages=cfg.bf16_stages,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    (
                        cfg.quantizer_warps
                        if cfg.schedule == "three_role"
                        else self.num_mma_warps
                    ),
                ),
                tx_count=tma_bytes,
                barrier_storage=pipeline_storage,
                cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
            )
            tma_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, cfg.bf16_stages
            )
            tma_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, cfg.bf16_stages
            )
        if cutlass.const_expr(cfg.schedule == "three_role"):
            quant_pipeline = pipeline.PipelineAsync.create(
                num_stages=cfg.mxfp8_stages,
                producer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, cfg.quantizer_warps * 32
                ),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, self.num_mma_warps * 32
                ),
                barrier_storage=storage.quant_pipeline.data_ptr(),
            )
            quant_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, cfg.mxfp8_stages
            )
            quant_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, cfg.mxfp8_stages
            )

        # MMA/register views are invariant across K tiles because shared memory
        # is reused after a CTA barrier.
        thr_mma = tiled_mma.get_slice(tidx)
        t_cs_a = thr_mma.partition_A(s_a)
        t_cs_b = thr_mma.partition_B(s_b)
        t_cr_a = tiled_mma.make_fragment_A(t_cs_a[None, None, None, 0])
        t_cr_b = tiled_mma.make_fragment_B(t_cs_b[None, None, None, 0])
        t_cr_sfa = sm120_utils.partition_fragment_SFA(
            s_sfa[None, None, 0], thr_mma, tidx
        )
        t_cr_sfb = sm120_utils.partition_fragment_SFB(
            s_sfb[None, None, 0], thr_mma, tidx
        )
        t_cr_sfa = cute.group_modes(t_cr_sfa, 2, cute.rank(t_cr_sfa))
        t_cr_sfb = cute.group_modes(t_cr_sfb, 2, cute.rank(t_cr_sfb))

        atom_a = _make_ldmatrix_atom(
            self.a_dtype,
            transpose=False,
            num_matrices=cfg.a_ldmatrix_matrices,
        )
        atom_b = _make_ldmatrix_atom(
            self.b_dtype,
            transpose=False,
            num_matrices=cfg.b_ldmatrix_matrices,
        )
        copy_a = cute.make_tiled_copy_A(atom_a, tiled_mma)
        copy_b = cute.make_tiled_copy_B(atom_b, tiled_mma)
        thr_copy_a = copy_a.get_slice(tidx)
        thr_copy_b = copy_b.get_slice(tidx)
        t_cs_a_copy = thr_copy_a.partition_S(s_a)
        t_cs_b_copy = thr_copy_b.partition_S(s_b)
        t_cr_a_copy = thr_copy_a.retile(t_cr_a)
        t_cr_b_copy = thr_copy_b.retile(t_cr_b)

        sfa_atom = _make_scale_s2r_atom(self.sf_dtype, cfg.sfa_s2r_bits)
        sfb_atom = _make_scale_s2r_atom(self.sf_dtype, cfg.sfb_s2r_bits)
        copy_sfa = cute.make_tiled_copy(
            sfa_atom,
            sm120_utils.get_layoutSFA_TV(tiled_mma),
            (
                cute.size(tiled_mma.permutation_mnk[0]),
                cute.size(tiled_mma.permutation_mnk[2]),
            ),
        )
        copy_sfb = cute.make_tiled_copy(
            sfb_atom,
            sm120_utils.get_layoutSFB_TV(tiled_mma),
            (
                cute.size(tiled_mma.permutation_mnk[1]),
                cute.size(tiled_mma.permutation_mnk[2]),
            ),
        )
        thr_copy_sfa = copy_sfa.get_slice(tidx)
        thr_copy_sfb = copy_sfb.get_slice(tidx)
        t_cs_sfa_copy = thr_copy_sfa.partition_S(s_sfa)
        t_cs_sfb_copy = thr_copy_sfb.partition_S(s_sfb)
        t_cr_sfa_copy = thr_copy_sfa.retile(t_cr_sfa)
        t_cr_sfb_copy = thr_copy_sfb.retile(t_cr_sfb)

        total_tiles = m_tiles * n_tiles
        for work_slot in cutlass.range_constexpr(self.work_tiles_per_cta):
            work_linear = linear_tile + work_slot * self.grid_ctas
            if cutlass.const_expr(cfg.persistent and cfg.reuse != "none"):
                # Contiguous chunks preserve an operand coordinate across
                # adjacent work: N-varying for X, M-varying for weight.
                work_linear = (
                    linear_tile * self.work_tiles_per_cta + work_slot
                )
            split_id = Int32(0)
            if cutlass.const_expr(self.split_reduction > 1):
                split_id = work_linear // total_tiles
                work_linear = work_linear % total_tiles
            # Grouped rasterization is bijective even for a partial final
            # group.  Reuse policies select the matching contiguous order.
            raster_n = cfg.raster == "n"
            if cutlass.const_expr(cfg.reuse == "x"):
                raster_n = True
            elif cutlass.const_expr(cfg.reuse == "weight"):
                raster_n = False
            if cutlass.const_expr(raster_n):
                full_group_size = n_tiles * cfg.grid_swizzle
                group = work_linear // full_group_size
                offset = work_linear % full_group_size
                rows_in_group = cutlass.min(
                    cfg.grid_swizzle,
                    m_tiles - group * cfg.grid_swizzle,
                )
                block_n = offset // rows_in_group
                block_m = (
                    group * cfg.grid_swizzle + offset % rows_in_group
                )
            else:
                full_group_size = m_tiles * cfg.grid_swizzle
                group = work_linear // full_group_size
                offset = work_linear % full_group_size
                cols_in_group = cutlass.min(
                    cfg.grid_swizzle,
                    n_tiles - group * cfg.grid_swizzle,
                )
                block_m = offset // cols_in_group
                block_n = (
                    group * cfg.grid_swizzle + offset % cols_in_group
                )

            # partition_C gives an accumulator and a direct global view with the
            # same per-thread coordinate order.  The identity tensor supplies tail
            # predicates for non-multiple M/N dimensions.
            out_matrix = out
            if cutlass.const_expr(
                self.split_reduction > 1 and not self.atomic_output
            ):
                out_matrix = cute.make_tensor(
                    out.iterator
                    + split_id * self.problem.m * self.problem.n,
                    cute.make_layout(
                        (self.problem.m, self.problem.n),
                        stride=(self.problem.n, 1),
                    ),
                )
            g_out_tile = cute.local_tile(
                out_matrix,
                (cfg.tile_m, cfg.tile_n),
                (block_m, block_n),
            )
            c_out_tile = cute.local_tile(
                cute.make_identity_tensor((self.problem.m, self.problem.n)),
                (cfg.tile_m, cfg.tile_n),
                (block_m, block_n),
            )
            t_cg_out = thr_mma.partition_C(g_out_tile)
            t_cc_out = thr_mma.partition_C(c_out_tile)
            accumulators = cute.make_rmem_tensor(t_cg_out.shape, self.acc_dtype)
            accumulators.fill(0.0)
    
            if cutlass.const_expr(cfg.epilogue == "tma"):
                r2s_atom = cute.make_copy_atom(
                    warp.StMatrix8x8x16bOp(
                        transpose=False,
                        num_matrices=cfg.store_vec,
                    ),
                    self.c_dtype,
                )
                c_layout_atom = cute.make_copy_atom(
                    warp.StMatrix8x8x16bOp(
                        transpose=False,
                        num_matrices=cfg.store_vec,
                    ),
                    self.c_dtype,
                )
                c_fragment_copy = cute.make_tiled_copy_C_atom(
                    c_layout_atom, tiled_mma
                )
                copy_r2s = cute.make_tiled_copy_S(r2s_atom, c_fragment_copy)
                thr_copy_r2s = copy_r2s.get_slice(tidx)
                t_rs_s_out = thr_copy_r2s.partition_D(s_out)
                t_rs_r_acc = copy_r2s.retile(accumulators)
                r_out_shape = cute.shape(thr_copy_r2s.partition_S(s_out))
                r_out_layout = cute.make_layout(r_out_shape[:3])
                t_rs_r_out = cute.make_rmem_tensor(
                    r_out_layout.shape, self.c_dtype
                )
    
                g_out_tiles_tma = cute.local_tile(
                    tma_tensor_out,
                    (cfg.tile_m, cfg.tile_n),
                    (None, None, None),
                )
                g_out_tile_tma = g_out_tiles_tma[
                    (None, None, block_m, block_n, 0)
                ]
                g_out_tile_tma = cute.zipped_divide(
                    g_out_tile_tma, (cfg.tile_m, cfg.tile_n)
                )
                b_sg_s_out, b_sg_g_out = cpasync.tma_partition(
                    tma_atom_out,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(s_out, 0, 2),
                    g_out_tile_tma,
                )
                tma_store_pipeline = pipeline.PipelineTmaStore.create(
                    num_stages=cfg.epilogue_stages,
                    producer_group=pipeline.CooperativeGroup(
                        pipeline.Agent.Thread, self.num_mma_warps * 32
                    ),
                )
    
            blocks_per_load = cfg.bf16_tile_k // SF_VEC_SIZE
            loads_per_mma_tile = cfg.tile_k // cfg.bf16_tile_k
            a_scale_blocks = cfg.tile_m * blocks_per_load
            b_scale_blocks = cfg.tile_n * blocks_per_load
            num_k_tiles = cute.ceil_div(self.problem.k, cfg.bf16_tile_k)
    
            if cutlass.const_expr(
                cfg.schedule in ("warp_specialized", "three_role")
            ):
                producer_warp = self.num_mma_warps
                if cutlass.const_expr(cfg.schedule == "three_role"):
                    producer_warp += cfg.quantizer_warps
                if warp_idx == producer_warp:
                    cute.arch.setmaxregister_decrease(cfg.producer_registers)
                elif (
                    cutlass.const_expr(cfg.schedule == "three_role")
                    and warp_idx >= self.num_mma_warps
                ):
                    cute.arch.setmaxregister_decrease(cfg.quantizer_registers)
                else:
                    cute.arch.setmaxregister_increase(cfg.consumer_registers)
    
            first_k_tile = Int32(0)
            final_k_tile = Int32(num_k_tiles)
            if cutlass.const_expr(self.split_reduction > 1):
                reduction_tiles = self.reduction_tile // cfg.bf16_tile_k
                first_k_tile = split_id * reduction_tiles
                final_k_tile = cutlass.min(
                    first_k_tile + reduction_tiles, num_k_tiles
                )
            for k_tile in cutlass.range(
                first_k_tile,
                final_k_tile,
                unroll=cfg.k_unroll,
            ):
                local_k_tile = k_tile - first_k_tile
                mma_tile_k = local_k_tile // loads_per_mma_tile
                k_block_base = (k_tile % loads_per_mma_tile) * blocks_per_load
                stage = Int32(mma_tile_k % cfg.mxfp8_stages)
                bf16_stage = Int32(0)
                if cutlass.const_expr(cfg.load_engine == "tma"):
                    issue_warp = Int32(0)
                    if cutlass.const_expr(cfg.schedule == "warp_specialized"):
                        issue_warp = Int32(self.num_mma_warps)
                    elif cutlass.const_expr(cfg.schedule == "three_role"):
                        issue_warp = Int32(
                            self.num_mma_warps + cfg.quantizer_warps
                        )
                    if warp_idx == issue_warp:
                        tma_pipeline.producer_acquire(tma_producer_state)
                        cute.copy(
                            tma_atom_x,
                            t_xg[(None, block_m, k_tile)],
                            t_xs[(None, tma_producer_state.index)],
                            tma_bar_ptr=tma_pipeline.producer_get_barrier(
                                tma_producer_state
                            ),
                        )
                        cute.copy(
                            tma_atom_weight,
                            t_wg[(None, block_n, k_tile)],
                            t_ws[(None, tma_producer_state.index)],
                            tma_bar_ptr=tma_pipeline.producer_get_barrier(
                                tma_producer_state
                            ),
                        )
                        tma_pipeline.producer_commit(tma_producer_state)
                        tma_producer_state.advance()
                    is_tma_consumer = warp_idx < self.num_mma_warps
                    if cutlass.const_expr(cfg.schedule == "three_role"):
                        is_tma_consumer = (
                            warp_idx >= self.num_mma_warps
                            and warp_idx
                            < self.num_mma_warps + cfg.quantizer_warps
                        )
                    if is_tma_consumer:
                        tma_full = tma_pipeline.consumer_try_wait(tma_consumer_state)
                        tma_pipeline.consumer_wait(tma_consumer_state, tma_full)
                        bf16_stage = tma_consumer_state.index
                elif cutlass.const_expr(cfg.load_engine == "cpasync"):
                    bf16_stage = Int32(k_tile % cfg.bf16_stages)
                    g_x_tile = cute.local_tile(
                        x,
                        (cfg.tile_m, cfg.bf16_tile_k),
                        (block_m, k_tile),
                    )
                    g_weight_tile = cute.local_tile(
                        weight,
                        (cfg.tile_n, cfg.bf16_tile_k),
                        (block_n, k_tile),
                    )
                    cp_async_thread = cp_async_tiled_copy.get_slice(tidx)
                    cute.copy(
                        cp_async_tiled_copy,
                        cp_async_thread.partition_S(g_x_tile),
                        cp_async_thread.partition_D(
                            s_bf16_a[None, None, bf16_stage]
                        ),
                    )
                    cute.copy(
                        cp_async_tiled_copy,
                        cp_async_thread.partition_S(g_weight_tile),
                        cp_async_thread.partition_D(
                            s_bf16_b[None, None, bf16_stage]
                        ),
                    )
                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.sync_threads()
                # A warp owns ``quant_vec`` independent 32-value MX blocks. Each
                # subwarp owns one block and each lane loads ``quant_vec`` adjacent
                # BF16 values, providing real vector/ILP variants while retaining
                # one E8M0 scale per 32 values.
                threads_per_scale = 32 // cfg.quant_vec
                scale_in_warp = lane_idx // threads_per_scale
                lane_in_scale = lane_idx % threads_per_scale
                scale_groups = (a_scale_blocks + b_scale_blocks) // cfg.quant_vec
                quant_warp_idx = warp_idx
                quant_warp_count = self.num_mma_warps
                if cutlass.const_expr(cfg.schedule == "warp_specialized"):
                    if warp_idx == self.num_mma_warps:
                        quant_warp_idx = Int32(scale_groups)
                elif cutlass.const_expr(cfg.schedule == "three_role"):
                    is_quantizer = (
                        warp_idx >= self.num_mma_warps
                        and warp_idx < self.num_mma_warps + cfg.quantizer_warps
                    )
                    if is_quantizer:
                        quant_pipeline.producer_acquire(quant_producer_state)
                        stage = quant_producer_state.index
                        quant_warp_idx = warp_idx - self.num_mma_warps
                    else:
                        quant_warp_idx = Int32(scale_groups)
                    quant_warp_count = cfg.quantizer_warps
                for task_group in cutlass.range(
                    quant_warp_idx,
                    scale_groups,
                    quant_warp_count,
                    unroll=1,
                ):
                    task = task_group * cfg.quant_vec + scale_in_warp
                    is_a = task < a_scale_blocks
                    local_task = task if is_a else task - a_scale_blocks
                    row = local_task // blocks_per_load
                    scale_block = local_task % blocks_per_load
                    global_row = (
                        block_m * cfg.tile_m + row
                        if is_a
                        else block_n * cfg.tile_n + row
                    )
                    row_limit = self.problem.m if is_a else self.problem.n
    
                    bf16_values = [BFloat16(0.0)] * cfg.quant_vec
                    values = [Float32(0.0)] * cfg.quant_vec
                    local_ks = [Int32(0)] * cfg.quant_vec
                    bf16_ks = [Int32(0)] * cfg.quant_vec
                    local_maximum = Float32(0.0)
                    local_maximum_bits = Int32(0)
                    if cutlass.const_expr(cfg.quant_load_bits > 16):
                        values_per_load = cfg.quant_load_bits // BFloat16.width
                        load_count = cfg.quant_vec // values_per_load
                        for load_idx in cutlass.range_constexpr(load_count):
                            vec_base = load_idx * values_per_load
                            bf16_k_base = (
                                scale_block * SF_VEC_SIZE
                                + lane_in_scale * cfg.quant_vec
                                + vec_base
                            )
                            if cutlass.const_expr(cfg.load_engine != "scalar"):
                                if is_a:
                                    src_row_a = s_bf16_a[
                                        row, None, bf16_stage
                                    ]
                                    loaded = nvvm.load_ext(
                                        src_row_a.iterator
                                        + src_row_a.layout(bf16_k_base),
                                        dtype=Uint16,
                                        count=values_per_load,
                                    ).bitcast(BFloat16)
                                    for load_vec in cutlass.range_constexpr(
                                        values_per_load
                                    ):
                                        bf16_values[vec_base + load_vec] = loaded[
                                            load_vec
                                        ]
                                else:
                                    src_row_b = s_bf16_b[
                                        row, None, bf16_stage
                                    ]
                                    loaded = nvvm.load_ext(
                                        src_row_b.iterator
                                        + src_row_b.layout(bf16_k_base),
                                        dtype=Uint16,
                                        count=values_per_load,
                                    ).bitcast(BFloat16)
                                    for load_vec in cutlass.range_constexpr(
                                        values_per_load
                                    ):
                                        bf16_values[vec_base + load_vec] = loaded[
                                            load_vec
                                        ]
                            else:
                                global_k_base = (
                                    k_tile * cfg.bf16_tile_k + bf16_k_base
                                )
                                if is_a:
                                    src_row_a = x[global_row, None]
                                    loaded = nvvm.load_ext(
                                        src_row_a.iterator
                                        + src_row_a.layout(global_k_base),
                                        dtype=Uint16,
                                        count=values_per_load,
                                    ).bitcast(BFloat16)
                                    for load_vec in cutlass.range_constexpr(
                                        values_per_load
                                    ):
                                        bf16_values[vec_base + load_vec] = loaded[
                                            load_vec
                                        ]
                                else:
                                    src_row_b = weight[global_row, None]
                                    loaded = nvvm.load_ext(
                                        src_row_b.iterator
                                        + src_row_b.layout(global_k_base),
                                        dtype=Uint16,
                                        count=values_per_load,
                                    ).bitcast(BFloat16)
                                    for load_vec in cutlass.range_constexpr(
                                        values_per_load
                                    ):
                                        bf16_values[vec_base + load_vec] = loaded[
                                            load_vec
                                        ]
                    for vec in cutlass.range_constexpr(cfg.quant_vec):
                        bf16_k = (
                            scale_block * SF_VEC_SIZE
                            + lane_in_scale * cfg.quant_vec
                            + vec
                        )
                        local_k = k_block_base * SF_VEC_SIZE + bf16_k
                        bf16_ks[vec] = bf16_k
                        local_ks[vec] = local_k
                        global_k = k_tile * cfg.bf16_tile_k + bf16_k
                        value_bf16 = bf16_values[vec]
                        if cutlass.const_expr(cfg.quant_load_bits == 16):
                            if global_row < row_limit and global_k < self.problem.k:
                                if cutlass.const_expr(cfg.load_engine != "scalar"):
                                    value_bf16 = (
                                        s_bf16_a[row, bf16_ks[vec], bf16_stage]
                                        if is_a
                                        else s_bf16_b[
                                            row, bf16_ks[vec], bf16_stage
                                        ]
                                    )
                                else:
                                    value_bf16 = (
                                        x[global_row, global_k]
                                        if is_a
                                        else weight[global_row, global_k]
                                    )
                        bf16_values[vec] = value_bf16
                        value = Float32(value_bf16)
                        values[vec] = value
                        if cutlass.const_expr(cfg.quant_amax == "fp32"):
                            magnitude = (
                                value.bitcast(Int32) & Int32(0x7FFFFFFF)
                            ).bitcast(Float32)
                            local_maximum = cute.arch.fmax(
                                local_maximum, magnitude, nan=True
                            )
                        else:
                            # BF16 magnitudes have the same unsigned ordering
                            # as their numerical values, with NaNs above all
                            # finite encodings. Reduce those exact source bits
                            # and widen only the final per-thread maximum.
                            magnitude_bits = Int32(
                                value_bf16.bitcast(Uint16)
                            ) & Int32(0x7FFF)
                            local_maximum_bits = cutlass.max(
                                local_maximum_bits, magnitude_bits
                            )

                    if cutlass.const_expr(cfg.quant_amax == "bf16_bits"):
                        local_maximum = Float32(
                            Uint16(local_maximum_bits).bitcast(BFloat16)
                        )
    
                    amax = self._warp_amax(
                        local_maximum,
                        threads_per_scale,
                        lane_idx,
                    )
                    scale_e8m0, inv_scale_fp32 = self._scale_from_amax(amax)
                    # The converter consumes two distinct FP32 values per issue.
                    # PTX places operand 0 in the high byte and operand 1 in the
                    # low byte, so store the bytes back in that order.  quant_vec=1
                    # retains a duplicated tail but every wider tuning choice uses
                    # the full x2 converter throughput.
                    pair_count = (cfg.quant_vec + 1) // 2
                    for pair in cutlass.range_constexpr(pair_count):
                        vec0 = pair * 2
                        vec1 = cutlass.min(vec0 + 1, cfg.quant_vec - 1)
                        if cutlass.const_expr(cfg.quant_math == "bf16x2"):
                            # Both the reciprocal and source values are exact
                            # BF16 powers/values. Pack two independent lanes,
                            # scale them with one native mul.bf16x2, then feed
                            # the result directly to the SM120 packed E4M3
                            # converter. Saturation is part of the conversion.
                            bits0 = bf16_values[vec0].bitcast(Uint16)
                            bits1 = bf16_values[vec1].bitcast(Uint16)
                            packed_values = Int32(bits0) | (Int32(bits1) << 16)
                            inv_bits = BFloat16(inv_scale_fp32).bitcast(Uint16)
                            packed_inv = Int32(inv_bits) | (Int32(inv_bits) << 16)
                            scaled_bf16x2 = nvvm.mul_bf16x2(
                                packed_values, packed_inv
                            )
                            packed = nvvm.inline_ptx_hl(
                                "cvt.rn.satfinite.e4m3x2.bf16x2 {$w0}, {$r0};",
                                write_only_types=[Int16],
                                read_only_args=[scaled_bf16x2],
                            )
                            quantized0 = Uint8(
                                packed & Int16(0xFF)
                            ).bitcast(Float8E4M3FN)
                            quantized1 = Uint8(
                                (packed >> Int16(8)) & Int16(0xFF)
                            ).bitcast(Float8E4M3FN)
                        else:
                            scaled0 = values[vec0] * inv_scale_fp32
                            scaled1 = values[vec1] * inv_scale_fp32
                            scaled0 = cute.arch.fmax(scaled0, -F8_MAX, nan=True)
                            scaled0 = cute.arch.fmin(scaled0, F8_MAX, nan=True)
                            scaled1 = cute.arch.fmax(scaled1, -F8_MAX, nan=True)
                            scaled1 = cute.arch.fmin(scaled1, F8_MAX, nan=True)
                            packed = nvvm.inline_ptx_hl(
                                "cvt.rn.satfinite.e4m3x2.f32 {$w0}, {$r0}, {$r1};",
                                write_only_types=[Int16],
                                read_only_args=[scaled0, scaled1],
                            )
                            quantized0 = Uint8(
                                (packed >> Int16(8)) & Int16(0xFF)
                            ).bitcast(Float8E4M3FN)
                            quantized1 = Uint8(packed & Int16(0xFF)).bitcast(
                                Float8E4M3FN
                            )
    
                        if is_a:
                            s_a[row, local_ks[vec0], stage] = quantized0
                            if vec1 != vec0:
                                s_a[row, local_ks[vec1], stage] = quantized1
                        else:
                            s_b[row, local_ks[vec0], stage] = quantized0
                            if vec1 != vec0:
                                s_b[row, local_ks[vec1], stage] = quantized1
    
                    if lane_in_scale == 0:
                        scale_k = (
                            k_block_base + scale_block
                        ) * SF_VEC_SIZE
                        if is_a:
                            s_sfa[row, scale_k, stage] = scale_e8m0
                        else:
                            s_sfb[row, scale_k, stage] = scale_e8m0
    
                if cutlass.const_expr(cfg.load_engine == "tma"):
                    is_tma_consumer = warp_idx < self.num_mma_warps
                    if cutlass.const_expr(cfg.schedule == "three_role"):
                        is_tma_consumer = (
                            warp_idx >= self.num_mma_warps
                            and warp_idx
                            < self.num_mma_warps + cfg.quantizer_warps
                        )
                    if is_tma_consumer:
                        tma_pipeline.consumer_release(tma_consumer_state)
                        tma_consumer_state.advance()
    
                if cutlass.const_expr(cfg.schedule == "three_role"):
                    if (
                        warp_idx >= self.num_mma_warps
                        and warp_idx < self.num_mma_warps + cfg.quantizer_warps
                    ):
                        quant_pipeline.producer_commit(quant_producer_state)
                        quant_producer_state.advance()
                    if warp_idx < self.num_mma_warps:
                        quant_full = quant_pipeline.consumer_try_wait(
                            quant_consumer_state
                        )
                        quant_pipeline.consumer_wait(
                            quant_consumer_state, quant_full
                        )
                        stage = quant_consumer_state.index
                elif cutlass.const_expr(cfg.schedule == "warp_specialized"):
                    if warp_idx < self.num_mma_warps:
                        self.mma_sync_barrier.arrive_and_wait()
                else:
                    cute.arch.sync_threads()
    
                num_k_blocks = cute.size(t_cr_a, mode=[2])
                if warp_idx < self.num_mma_warps:
                    for k_block in cutlass.range_constexpr(num_k_blocks):
                        if (
                            k_block >= k_block_base
                            and k_block < k_block_base + blocks_per_load
                        ):
                            cute.copy(
                                copy_a,
                                t_cs_a_copy[None, None, k_block, stage],
                                t_cr_a_copy[None, None, k_block],
                            )
                            cute.copy(
                                copy_b,
                                t_cs_b_copy[None, None, k_block, stage],
                                t_cr_b_copy[None, None, k_block],
                            )
                            cute.copy(
                                copy_sfa,
                                cute.filter_zeros(t_cs_sfa_copy)[
                                    None, None, k_block, stage
                                ],
                                cute.filter_zeros(t_cr_sfa_copy)[None, None, k_block],
                            )
                            cute.copy(
                                copy_sfb,
                                cute.filter_zeros(t_cs_sfb_copy)[
                                    None, None, k_block, stage
                                ],
                                cute.filter_zeros(t_cr_sfb_copy)[None, None, k_block],
                            )
                            cute.gemm(
                                tiled_mma,
                                accumulators,
                                [
                                    t_cr_a[None, None, k_block],
                                    t_cr_sfa[None, None, k_block],
                                ],
                                [
                                    t_cr_b[None, None, k_block],
                                    t_cr_sfb[None, None, k_block],
                                ],
                                accumulators,
                            )
    
                if cutlass.const_expr(cfg.schedule == "three_role"):
                    if warp_idx < self.num_mma_warps:
                        quant_pipeline.consumer_release(quant_consumer_state)
                        quant_consumer_state.advance()
                elif cutlass.const_expr(cfg.schedule == "warp_specialized"):
                    if warp_idx < self.num_mma_warps:
                        self.mma_sync_barrier.arrive_and_wait()
                else:
                    cute.arch.sync_threads()
    
            if cutlass.const_expr(
                cfg.schedule in ("warp_specialized", "three_role")
            ):
                producer_warp = self.num_mma_warps
                if cutlass.const_expr(cfg.schedule == "three_role"):
                    producer_warp += cfg.quantizer_warps
                if warp_idx == producer_warp:
                    tma_pipeline.producer_tail(tma_producer_state)
            if cutlass.const_expr(cfg.schedule == "three_role"):
                if (
                    warp_idx >= self.num_mma_warps
                    and warp_idx < self.num_mma_warps + cfg.quantizer_warps
                ):
                    quant_pipeline.producer_tail(quant_producer_state)
    
            if cutlass.const_expr(cfg.epilogue == "direct"):
                if warp_idx < self.num_mma_warps:
                    for elem in cutlass.range(
                        cute.size(accumulators), unroll_full=True
                    ):
                        coord = t_cc_out[elem]
                        if coord[0] < self.problem.m and coord[1] < self.problem.n:
                            if cutlass.const_expr(self.atomic_output):
                                cute.arch.atomic_add(
                                    t_cg_out.iterator + t_cg_out.layout(elem),
                                    accumulators[elem],
                                    sem="relaxed",
                                    scope="gpu",
                                )
                            else:
                                t_cg_out[elem] = self.c_dtype(accumulators[elem])
            elif cutlass.const_expr(cfg.epilogue == "tma"):
                full_output_tile = (
                    (block_m + 1) * cfg.tile_m <= self.problem.m
                    and (block_n + 1) * cfg.tile_n <= self.problem.n
                )
                if full_output_tile:
                    if warp_idx < self.num_mma_warps:
                        for elem in cutlass.range(
                            cute.size(t_rs_r_out), unroll_full=True
                        ):
                            t_rs_r_out[elem] = BFloat16(t_rs_r_acc[elem])
                        cute.copy(
                            copy_r2s,
                            t_rs_r_out,
                            t_rs_s_out[(None, None, None, 0)],
                        )
                        cute.arch.fence_proxy("async.shared", space="cta")
                        self.epilogue_sync_barrier.arrive_and_wait()
                        if warp_idx == 0:
                            cute.copy(
                                tma_atom_out,
                                b_sg_s_out[(None, 0)],
                                b_sg_g_out[(None, 0)],
                            )
                            tma_store_pipeline.producer_commit()
                            tma_store_pipeline.producer_acquire()
                            # A TMA store may overlap all work after its issue, but
                            # the final group must drain before the CTA exits.
                            tma_store_pipeline.producer_tail()
                else:
                    # Tensor-map stores do not predicate a partial output tile on
                    # this SM120 stack.  Preserve exact boundary semantics with the
                    # already-partitioned direct store rather than allowing an OOB
                    # TMA transaction to overwrite neighboring tiles.
                    if warp_idx < self.num_mma_warps:
                        for elem in cutlass.range(
                            cute.size(accumulators), unroll_full=True
                        ):
                            coord = t_cc_out[elem]
                            if (
                                coord[0] < self.problem.m
                                and coord[1] < self.problem.n
                            ):
                                t_cg_out[elem] = BFloat16(accumulators[elem])


@lru_cache(maxsize=None)
def compile_mxfp8_fwd(
    problem: MXFP8Problem,
    config: MXFP8FwdConfig,
    *,
    a_orientation: str = "row",
    b_orientation: str = "row",
):
    """Compile a fused GEMM over row-major or metadata-only transpose views."""

    problem.validate()
    kernel = MXFP8LinearFwdKernel(
        problem,
        config,
        a_orientation=a_orientation,
        b_orientation=b_orientation,
    )
    a_stride = (
        (problem.k, 1)
        if a_orientation == "row"
        else (1, problem.m)
    )
    b_stride = (
        (problem.k, 1)
        if b_orientation == "row"
        else (1, problem.n)
    )
    x = cute.runtime.make_fake_tensor(
        BFloat16,
        (problem.m, problem.k),
        stride=a_stride,
        assumed_align=16,
    )
    weight = cute.runtime.make_fake_tensor(
        BFloat16,
        (problem.n, problem.k),
        stride=b_stride,
        assumed_align=16,
    )
    out = cute.runtime.make_fake_tensor(
        BFloat16,
        (problem.m, problem.n),
        stride=(problem.n, 1),
        assumed_align=16,
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel,
        x,
        weight,
        out,
        stream,
        options=(
            "--enable-tvm-ffi --opt-level 3 "
            f"--ptxas-options '-O3 -v --maxrregcount={config.maxrregcount}'"
        ),
    )


@lru_cache(maxsize=None)
def compile_mxfp8_split_fwd(
    problem: MXFP8Problem,
    config: MXFP8FwdConfig,
    *,
    a_orientation: str,
    b_orientation: str,
    split_reduction: int,
    reduction_tile: int,
):
    """Compile fused dynamic quantization into FP32 split-K partials.

    The launcher writes a flat ``[split, M, N]`` workspace. Every slice owns a
    disjoint, block-aligned interval of the original reduction dimension, and
    accumulation remains FP32 from tensor-core issue through the workspace.
    """

    problem.validate()
    kernel = MXFP8LinearFwdKernel(
        problem,
        config,
        a_orientation=a_orientation,
        b_orientation=b_orientation,
        split_reduction=split_reduction,
        reduction_tile=reduction_tile,
    )
    a_stride = (problem.k, 1) if a_orientation == "row" else (1, problem.m)
    b_stride = (problem.k, 1) if b_orientation == "row" else (1, problem.n)
    x = cute.runtime.make_fake_tensor(
        BFloat16,
        (problem.m, problem.k),
        stride=a_stride,
        assumed_align=16,
    )
    weight = cute.runtime.make_fake_tensor(
        BFloat16,
        (problem.n, problem.k),
        stride=b_stride,
        assumed_align=16,
    )
    workspace = cute.runtime.make_fake_tensor(
        Float32,
        (split_reduction * problem.m * problem.n,),
        stride=(1,),
        assumed_align=16,
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel,
        x,
        weight,
        workspace,
        stream,
        options=(
            "--enable-tvm-ffi --opt-level 3 "
            f"--ptxas-options '-O3 -v --maxrregcount={config.maxrregcount}'"
        ),
    )


@lru_cache(maxsize=None)
def compile_mxfp8_atomic_split_fwd(
    problem: MXFP8Problem,
    config: MXFP8FwdConfig,
    *,
    a_orientation: str,
    b_orientation: str,
    split_reduction: int,
    reduction_tile: int,
):
    """Compile fused split-K partials which atomically accumulate in FP32."""

    problem.validate()
    kernel = MXFP8LinearFwdKernel(
        problem,
        config,
        a_orientation=a_orientation,
        b_orientation=b_orientation,
        split_reduction=split_reduction,
        reduction_tile=reduction_tile,
        atomic_output=True,
    )
    a_stride = (problem.k, 1) if a_orientation == "row" else (1, problem.m)
    b_stride = (problem.k, 1) if b_orientation == "row" else (1, problem.n)
    x = cute.runtime.make_fake_tensor(
        BFloat16,
        (problem.m, problem.k),
        stride=a_stride,
        assumed_align=16,
    )
    weight = cute.runtime.make_fake_tensor(
        BFloat16,
        (problem.n, problem.k),
        stride=b_stride,
        assumed_align=16,
    )
    accumulator = cute.runtime.make_fake_tensor(
        Float32,
        (problem.m, problem.n),
        stride=(problem.n, 1),
        assumed_align=16,
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel,
        x,
        weight,
        accumulator,
        stream,
        options=(
            "--enable-tvm-ffi --opt-level 3 "
            f"--ptxas-options '-O3 -v --maxrregcount={config.maxrregcount}'"
        ),
    )


__all__ = [
    "MXFP8LinearFwdKernel",
    "compile_mxfp8_fwd",
    "compile_mxfp8_atomic_split_fwd",
    "compile_mxfp8_split_fwd",
]
