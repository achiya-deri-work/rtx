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

import torch

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
    Float4E2M1FN,
    Float4E2M1FNx2,
    Float8E4M3FN,
    Float8E8M0FNU,
    Float32,
    Int16,
    Int32,
    Uint16,
    Uint32,
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
    sf_vec_size: int,
    num_stages: int,
):
    """SM120 SFA layout for a logical 64-row tile in a padded 128-row block.

    NVIDIA's public helper materializes ``(32, 4)`` row groups and therefore
    requires M to be divisible by 128.  The first 64 logical rows are exactly
    the first two of those four interleaved groups: ``(32, 2):(16, 4)``.  The
    unchanged strides preserve the hardware's 128-row address encoding while
    the logical extent seen by fragment partitioning is genuinely 64.
    """

    mma_nsf = tiled_mma.shape_mnk[2] // sf_vec_size
    mn_shape = ((32, 2), 1)
    mn_stride = ((16, 4), 512)
    k_shape = (
        (sf_vec_size, mma_nsf),
        4 // mma_nsf,
        tile_k // sf_vec_size // 4,
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
        cluster_output: bool = False,
        nvfp4: bool = False,
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
        self.cluster_output = cluster_output
        self.nvfp4 = nvfp4
        if split_reduction < 1:
            raise ValueError("split_reduction must be positive")
        if split_reduction == 1 and (
            reduction_tile != 0 or atomic_output or cluster_output
        ):
            raise ValueError("an unsplit kernel must use reduction_tile=0")
        if atomic_output and cluster_output:
            raise ValueError("atomic and cluster outputs are mutually exclusive")
        if split_reduction > 1:
            if config.epilogue != "direct":
                raise ValueError("split reduction requires the FP32 direct epilogue")
            if cluster_output and split_reduction not in (2, 4, 8):
                raise ValueError("SM120 CTA clusters support split counts 2, 4, or 8")
            if cluster_output and config.persistent:
                raise ValueError("cluster split reduction has a fixed CTA topology")
            if cluster_output and config.cluster_reuse != "none":
                raise ValueError(
                    "split reduction and operand reuse require separate CTA clusters"
                )
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
        output_tiles = (
            (problem.m + config.tile_m - 1) // config.tile_m
        ) * ((problem.n + config.tile_n - 1) // config.tile_n)
        total_tiles = split_reduction * output_tiles
        self.split_grid_ctas = output_tiles
        self.grid_ctas = total_tiles
        if config.persistent:
            sm_count = utils.HardwareInfo().get_device_multiprocessor_count()
            if split_reduction > 1:
                # Keep every CTA on one reduction slice so its TMA and
                # quantizer pipelines can remain live across output tiles.
                # Divide the device budget among slices, then choose an exact
                # divisor so every CTA owns the same constexpr work count.
                per_split_budget = max(
                    1,
                    sm_count * config.persistent_waves // split_reduction,
                )
                self.split_grid_ctas = min(output_tiles, per_split_budget)
                while output_tiles % self.split_grid_ctas:
                    self.split_grid_ctas -= 1
                self.grid_ctas = split_reduction * self.split_grid_ctas
            else:
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

        self.a_dtype = Float4E2M1FN if nvfp4 else Float8E4M3FN
        self.b_dtype = Float4E2M1FN if nvfp4 else Float8E4M3FN
        self.sf_dtype = Float8E4M3FN if nvfp4 else Float8E8M0FNU
        self.sf_vec_size = 16 if nvfp4 else SF_VEC_SIZE
        self.apply_output_scale = nvfp4
        self.acc_dtype = Float32
        self.c_dtype = (
            Float32 if split_reduction > 1 and not cluster_output else BFloat16
        )
        self.a_layout = utils.LayoutEnum.ROW_MAJOR
        self.b_layout = utils.LayoutEnum.ROW_MAJOR

    @cute.jit
    def _cluster_broadcast(
        self,
        barriers: cute.Tensor,
        barrier_index: cutlass.Constexpr,
        phase: Int32,
        cluster_rank: Int32,
    ) -> None:
        """Rank 0 publishes one generation to every CTA in the cluster."""
        cute.arch.sync_threads()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            with cute.arch.elect_one():
                if cluster_rank == 0:
                    for peer in cutlass.range_constexpr(self.split_reduction):
                        peer_barrier = cute.arch.map_dsmem_ptr(
                            barriers.iterator + barrier_index, Int32(peer)
                        )
                        nvvm.mbarrier_arrive(
                            peer_barrier,
                            scope=nvvm.MemScope.CLUSTER,
                        )
                cute.arch.mbarrier_wait(
                    barriers.iterator + barrier_index, phase
                )
        cute.arch.sync_threads()

    @cute.jit
    def _cluster_gather(
        self,
        barriers: cute.Tensor,
        barrier_index: cutlass.Constexpr,
        phase: Int32,
        cluster_rank: Int32,
    ) -> None:
        """Every CTA publishes a contribution; rank 0 waits for all peers."""
        cute.arch.sync_threads()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            with cute.arch.elect_one():
                leader_barrier = cute.arch.map_dsmem_ptr(
                    barriers.iterator + barrier_index, Int32(0)
                )
                nvvm.mbarrier_arrive(
                    leader_barrier,
                    scope=nvvm.MemScope.CLUSTER,
                )
                if cluster_rank == 0:
                    cute.arch.mbarrier_wait(
                        barriers.iterator + barrier_index, phase
                    )
        cute.arch.sync_threads()

    def _setup_static_layouts(self) -> None:
        cfg = self.config
        if cutlass.const_expr(self.nvfp4):
            mma_op = warp.MmaMXF4NVF4Op(
                self.a_dtype, self.acc_dtype, self.sf_dtype
            )
        else:
            mma_op = warp.MmaMXF8Op(
                self.a_dtype, self.acc_dtype, self.sf_dtype
            )
        atom_layout = cute.make_layout(
            (cfg.atom_layout_m, cfg.atom_layout_n, 1)
        )
        permutation_mnk = sm120_utils.get_permutation_mnk(
            self.tile_shape_mnk, self.sf_vec_size, not self.nvfp4
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
        if cutlass.const_expr(self.nvfp4):
            a_packed_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
                swizzle_kinds[cfg.a_swizzle], Float4E2M1FNx2
            )
            b_packed_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
                swizzle_kinds[cfg.b_swizzle], Float4E2M1FNx2
            )
            self.a_packed_layout = cute.tile_to_shape(
                a_packed_atom,
                (cfg.tile_m, cfg.tile_k // 2, cfg.mxfp8_stages),
                order=(0, 1, 2),
            )
            self.b_packed_layout = cute.tile_to_shape(
                b_packed_atom,
                (cfg.tile_n, cfg.tile_k // 2, cfg.mxfp8_stages),
                order=(0, 1, 2),
            )
            self.a_smem_layout = cute.recast_layout(
                4, 8, self.a_packed_layout
            )
            self.b_smem_layout = cute.recast_layout(
                4, 8, self.b_packed_layout
            )
            self.q_storage_dtype = Float4E2M1FNx2
        else:
            a_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
                swizzle_kinds[cfg.a_swizzle], self.a_dtype
            )
            b_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
                swizzle_kinds[cfg.b_swizzle], self.b_dtype
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
            self.a_packed_layout = self.a_smem_layout
            self.b_packed_layout = self.b_smem_layout
            self.q_storage_dtype = self.a_dtype
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
                self.sf_vec_size,
                cfg.mxfp8_stages,
            )
        else:
            self.sfa_smem_layout = blockscaled_utils.sm120_make_smem_layout_sfa(
                self.tiled_mma,
                scale_layout_tile,
                self.sf_vec_size,
                cfg.mxfp8_stages,
            )
        self.sfb_smem_layout = blockscaled_utils.sm120_make_smem_layout_sfb(
            self.tiled_mma,
            scale_layout_tile,
            self.sf_vec_size,
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
        if self.a_orientation == "transpose" and cfg.bf16_swizzle == "none":
            self.bf16_a_smem_layout = cute.make_composed_layout(
                cute.make_swizzle(0, 0, 0),
                0,
                cute.make_ordered_layout(
                    (cfg.tile_m, cfg.bf16_tile_k, cfg.bf16_stages),
                    order=(0, 1, 2),
                ),
            )
        else:
            self.bf16_a_smem_layout = cute.tile_to_shape(
                bf16_a_atom,
                (cfg.tile_m, cfg.bf16_tile_k, cfg.bf16_stages),
                order=(0, 1, 2),
            )
        if self.b_orientation == "transpose" and cfg.bf16_swizzle == "none":
            self.bf16_b_smem_layout = cute.make_composed_layout(
                cute.make_swizzle(0, 0, 0),
                0,
                cute.make_ordered_layout(
                    (cfg.tile_n, cfg.bf16_tile_k, cfg.bf16_stages),
                    order=(0, 1, 2),
                ),
            )
        else:
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
        x_tensor_scale: cute.Tensor,
        weight_tensor_scale: cute.Tensor,
        x_amax_out: cute.Tensor,
        weight_amax_out: cute.Tensor,
        stream: cuda.CUstream,
    ):
        # Runtime receives original contiguous operands. Logical transpose is a
        # CuTe layout over the same pointer, created before any TMA/cp.async
        # descriptor; Python never constructs a torch.Tensor .T view.
        if cutlass.const_expr(self.a_orientation == "transpose"):
            x = cute.make_tensor(
                x.iterator,
                cute.make_layout(
                    (self.problem.m, self.problem.k),
                    stride=(1, self.problem.m),
                ),
            )
        if cutlass.const_expr(self.b_orientation == "transpose"):
            weight = cute.make_tensor(
                weight.iterator,
                cute.make_layout(
                    (self.problem.n, self.problem.k),
                    stride=(1, self.problem.n),
                ),
            )
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

        def make_oriented_cpasync_copy(
            rows: cutlass.Constexpr,
            orientation: cutlass.Constexpr,
        ):
            if cutlass.const_expr(orientation == "transpose"):
                # Logical-K is physically strided, but logical rows are the
                # contiguous basis. Vectorize that M/N basis and land it in
                # the matching MN-major SMEM layout; no transpose copy exists.
                row_threads = rows // cp_async_values
                thread_layout = cute.make_layout(
                    (row_threads, cfg.num_threads // row_threads),
                    stride=(1, row_threads),
                )
                value_layout = cute.make_layout((cp_async_values, 1))
            else:
                k_threads = cfg.bf16_tile_k // cp_async_values
                thread_layout = cute.make_layout(
                    (cfg.num_threads // k_threads, k_threads),
                    stride=(k_threads, 1),
                )
                value_layout = cute.make_layout((1, cp_async_values))
            return cute.make_tiled_copy_tv(
                cp_async_atom,
                thread_layout,
                value_layout,
            )

        cp_async_tiled_copy_a = make_oriented_cpasync_copy(
            cfg.tile_m, self.a_orientation
        )
        cp_async_tiled_copy_b = make_oriented_cpasync_copy(
            cfg.tile_n, self.b_orientation
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
                        self.q_storage_dtype,
                        cute.cosize(self.a_packed_layout),
                    ],
                    1024,
                ]
                q_b: cute.struct.Align[
                    cute.struct.MemRange[
                        self.q_storage_dtype,
                        cute.cosize(self.b_packed_layout),
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
                cluster_barrier: cute.struct.Align[
                    cute.struct.MemRange[
                        cutlass.Int64, 2 if self.cluster_output else 0
                    ],
                    8,
                ]
                delayed_scale: cute.struct.Align[
                    cute.struct.MemRange[
                        Float32,
                        6 if self.nvfp4 and cfg.collect_amax else 0,
                    ],
                    16,
                ]
        @cute.struct
        class SharedStorageScalar:
                q_a: cute.struct.Align[
                    cute.struct.MemRange[
                        self.q_storage_dtype,
                        cute.cosize(self.a_packed_layout),
                    ],
                    1024,
                ]
                q_b: cute.struct.Align[
                    cute.struct.MemRange[
                        self.q_storage_dtype,
                        cute.cosize(self.b_packed_layout),
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
                cluster_barrier: cute.struct.Align[
                    cute.struct.MemRange[
                        cutlass.Int64, 2 if self.cluster_output else 0
                    ],
                    8,
                ]
                delayed_scale: cute.struct.Align[
                    cute.struct.MemRange[
                        Float32,
                        6 if self.nvfp4 and cfg.collect_amax else 0,
                    ],
                    16,
                ]

        self.shared_storage = SharedStorageScalar
        if cutlass.const_expr(cfg.load_engine != "scalar"):
            self.shared_storage = SharedStorageStaged
        # A linear grid lets raster order and grouped CTA swizzling be genuine
        # compile-time schedule choices without padding the edge group.
        m_tiles = (self.problem.m + cfg.tile_m - 1) // cfg.tile_m
        n_tiles = (self.problem.n + cfg.tile_n - 1) // cfg.tile_n
        grid = (self.grid_ctas, 1, 1)
        kernel = self.kernel(
            x,
            weight,
            out,
            x_tensor_scale,
            weight_tensor_scale,
            x_amax_out,
            weight_amax_out,
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
            cp_async_tiled_copy_a,
            cp_async_tiled_copy_b,
        )
        if cutlass.const_expr(self.cluster_output):
            kernel.launch(
                grid=grid,
                block=[self.threads_per_cta, 1, 1],
                cluster=[self.split_reduction, 1, 1],
                stream=stream,
            )
        elif cutlass.const_expr(cfg.cluster_reuse != "none"):
            kernel.launch(
                grid=grid,
                block=[self.threads_per_cta, 1, 1],
                cluster=[cfg.cluster_size, 1, 1],
                stream=stream,
            )
        else:
            kernel.launch(
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
    def _nvfp4_scale_from_amax(
        self,
        amax: Float32,
        tensor_scale: Float32,
        inv_tensor_scale: Float32,
        quant_multiplier: Float32,
    ):
        """Return TorchAO-compatible E4M3 block scale and FP4 reciprocal."""

        if cutlass.const_expr(
            self.config.scale_reciprocal
            in (
                "supplied_pow2",
                "supplied_pow2_ptx_lut",
                "supplied_pow2_ptx_rcp",
            )
        ):
            raw_scale = nvvm.inline_ptx_hl(
                "mul.rn.f32 {$w0}, {$r0}, {$r1};",
                write_only_types=[Float32],
                read_only_args=[amax, quant_multiplier],
            )
        else:
            raw_scale = amax / (Float32(6.0) * tensor_scale)
        raw_scale = cute.arch.fmax(raw_scale, Float32(0.015625), nan=True)
        raw_scale = cute.arch.fmin(raw_scale, Float32(448.0), nan=True)
        packed = nvvm.inline_ptx_hl(
            "cvt.rn.satfinite.e4m3x2.f32 {$w0}, {$r0}, {$r0};",
            write_only_types=[Int16],
            read_only_args=[raw_scale],
        )
        scale = Uint8(packed & Int16(0xFF)).bitcast(Float8E4M3FN)
        if cutlass.const_expr(
            self.config.scale_reciprocal == "supplied_pow2_ptx_lut"
        ):
            # E4M3 normal values are (1 + mantissa/8) * 2**(exponent-7).
            # Decode their reciprocal without an FP32 divide. Keeping the
            # entire non-uniform lookup in one PTX region also avoids a CuTe
            # staged-SSA issue observed with Python control flow around the
            # selected reciprocal.
            reciprocal = nvvm.inline_ptx_hl(
                "{.reg .u32 code, exp, mant; "
                ".reg .b32 exp_scale, mrec; .reg .f32 tmp; .reg .pred p; "
                "mov.b32 code, {$r0}; "
                "shr.u32 exp, code, 3; and.b32 exp, exp, 15; "
                "sub.u32 exp, 134, exp; shl.b32 exp_scale, exp, 23; "
                "and.b32 mant, code, 7; mov.b32 mrec, 0f3f800000; "
                "setp.eq.u32 p, mant, 1; selp.b32 mrec, 0f3f638e39, mrec, p; "
                "setp.eq.u32 p, mant, 2; selp.b32 mrec, 0f3f4ccccd, mrec, p; "
                "setp.eq.u32 p, mant, 3; selp.b32 mrec, 0f3f3a2e8c, mrec, p; "
                "setp.eq.u32 p, mant, 4; selp.b32 mrec, 0f3f2aaaab, mrec, p; "
                "setp.eq.u32 p, mant, 5; selp.b32 mrec, 0f3f1d89d9, mrec, p; "
                "setp.eq.u32 p, mant, 6; selp.b32 mrec, 0f3f124925, mrec, p; "
                "setp.eq.u32 p, mant, 7; selp.b32 mrec, 0f3f088889, mrec, p; "
                "mul.rn.f32 tmp, {$r1}, exp_scale; "
                "mul.rn.f32 {$w0}, tmp, mrec;}",
                write_only_types=[Float32],
                read_only_args=[
                    Int32(scale.bitcast(Uint8)),
                    inv_tensor_scale,
                ],
            )
        elif cutlass.const_expr(
            self.config.scale_reciprocal == "supplied_pow2_ptx_rcp"
        ):
            reciprocal = nvvm.inline_ptx_hl(
                "{.reg .f32 r; rcp.approx.f32 r, {$r1}; "
                "mul.rn.f32 {$w0}, {$r0}, r;}",
                write_only_types=[Float32],
                read_only_args=[inv_tensor_scale, Float32(scale)],
            )
        elif cutlass.const_expr(self.config.scale_reciprocal != "direct"):
            reciprocal = inv_tensor_scale / Float32(scale)
        else:
            reciprocal = (Float32(1.0) / tensor_scale) / Float32(scale)
        return scale, reciprocal

    @cute.jit
    def _nvfp4_tensor_scale_from_amax(self, amax: Float32):
        """Prepare the configured delayed tensor scale inside each CTA."""

        if cutlass.const_expr(self.config.tensor_scale_mode == "exact"):
            scale = amax * Float32(1.0 / 2688.0)
            if amax == Float32(0.0):
                scale = Float32(1.0)
            inverse = Float32(1.0) / scale
            return scale, inverse, inverse * Float32(1.0 / 6.0)

        target = cute.arch.fmax(
            amax * Float32(1.0 / 2688.0),
            Float32(2.0**-126),
            nan=True,
        )
        bits = target.bitcast(Int32)
        exponent_bits = bits & Int32(0x7F800000)
        if (bits & Int32(0x007FFFFF)) != 0:
            exponent_bits += Int32(1 << 23)
        scale = exponent_bits.bitcast(Float32)
        if amax == Float32(0.0):
            scale = Float32(1.0)
        exponent = (scale.bitcast(Int32) >> Int32(23)) & Int32(0xFF)
        inverse = ((Int32(254) - exponent) << Int32(23)).bitcast(Float32)
        return scale, inverse, inverse * Float32(1.0 / 6.0)

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
        x_tensor_scale: cute.Tensor,
        weight_tensor_scale: cute.Tensor,
        x_amax_out: cute.Tensor,
        weight_amax_out: cute.Tensor,
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
        cp_async_tiled_copy_a: cute.TiledCopy,
        cp_async_tiled_copy_b: cute.TiledCopy,
    ):
        cfg = self.config
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = cute.arch.lane_idx()
        linear_tile, _, _ = cute.arch.block_idx()
        m_tiles = cute.ceil_div(self.problem.m, cfg.tile_m)
        n_tiles = cute.ceil_div(self.problem.n, cfg.tile_n)
        output_scale = Float32(1.0)
        x_tensor_scale_value = Float32(1.0)
        weight_tensor_scale_value = Float32(1.0)
        x_inv_tensor_scale = Float32(1.0)
        weight_inv_tensor_scale = Float32(1.0)
        x_quant_multiplier = Float32(1.0)
        weight_quant_multiplier = Float32(1.0)
        if cutlass.const_expr(
            self.apply_output_scale and not self.config.collect_amax
        ):
            x_tensor_scale_value = Float32(x_tensor_scale[0])
            weight_tensor_scale_value = Float32(weight_tensor_scale[0])
            output_scale = x_tensor_scale_value * weight_tensor_scale_value
            if cutlass.const_expr(
                self.config.scale_reciprocal != "direct"
            ):
                x_inv_tensor_scale = Float32(x_tensor_scale[1])
                weight_inv_tensor_scale = Float32(weight_tensor_scale[1])
            if cutlass.const_expr(
                self.config.scale_reciprocal
                in (
                    "supplied_pow2",
                    "supplied_pow2_ptx_lut",
                    "supplied_pow2_ptx_rcp",
                )
            ):
                x_quant_multiplier = Float32(x_tensor_scale[2])
                weight_quant_multiplier = Float32(weight_tensor_scale[2])

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        if cutlass.const_expr(
            self.nvfp4 and self.config.collect_amax
        ):
            # Generation t consumes the configured history from generation
            # t-1. ``per_cta`` is self-resetting and preserves the original
            # barrier-free ABI. ``scalar_atomic`` reads one global slot per
            # history entry; the launch wrapper clears its fresh slot with an
            # asynchronous stream-ordered memset before entering this kernel.
            prior_x_amax = Float32(0.0)
            prior_weight_amax = Float32(0.0)
            telemetry_slots = self.grid_ctas
            if cutlass.const_expr(cfg.telemetry_layout == "scalar_atomic"):
                telemetry_slots = 1
            history_entries = cfg.amax_history_len
            if cutlass.const_expr(cfg.amax_history_algo == "most_recent"):
                history_entries = 1
            if warp_idx == 0:
                for telemetry_idx in cutlass.range(
                    lane_idx,
                    telemetry_slots * history_entries,
                    32,
                    unroll=1,
                ):
                    prior_x_amax = cute.arch.fmax(
                        prior_x_amax,
                        Float32(x_tensor_scale[telemetry_idx]),
                        nan=True,
                    )
                    prior_weight_amax = cute.arch.fmax(
                        prior_weight_amax,
                        Float32(weight_tensor_scale[telemetry_idx]),
                        nan=True,
                    )
                prior_x_amax = nvvm.redux_sync(
                    prior_x_amax.bitcast(Int32),
                    nvvm.ReductionKind.UMAX,
                    Int32(0xFFFFFFFF),
                ).bitcast(Float32)
                prior_weight_amax = nvvm.redux_sync(
                    prior_weight_amax.bitcast(Int32),
                    nvvm.ReductionKind.UMAX,
                    Int32(0xFFFFFFFF),
                ).bitcast(Float32)
                if lane_idx == 0:
                    x_scale, x_inverse, x_multiplier = (
                        self._nvfp4_tensor_scale_from_amax(prior_x_amax)
                    )
                    weight_scale, weight_inverse, weight_multiplier = (
                        self._nvfp4_tensor_scale_from_amax(
                            prior_weight_amax
                        )
                    )
                    storage.delayed_scale[0] = x_scale
                    storage.delayed_scale[1] = x_inverse
                    storage.delayed_scale[2] = x_multiplier
                    storage.delayed_scale[3] = weight_scale
                    storage.delayed_scale[4] = weight_inverse
                    storage.delayed_scale[5] = weight_multiplier
                    if cutlass.const_expr(cfg.telemetry_layout == "per_cta"):
                        x_amax_out[linear_tile] = Float32(0.0)
                        weight_amax_out[linear_tile] = Float32(0.0)
                        for history_idx in cutlass.range_constexpr(
                            1, cfg.amax_history_len
                        ):
                            next_idx = history_idx * telemetry_slots + linear_tile
                            prior_idx = (
                                (history_idx - 1) * telemetry_slots + linear_tile
                            )
                            x_amax_out[next_idx] = x_tensor_scale[prior_idx]
                            weight_amax_out[next_idx] = weight_tensor_scale[
                                prior_idx
                            ]
                    elif linear_tile == 0:
                        # Slot zero was cleared by the wrapper. CTA zero only
                        # rotates immutable older generations while all CTAs
                        # may atomically contribute to the fresh slot.
                        for history_idx in cutlass.range_constexpr(
                            1, cfg.amax_history_len
                        ):
                            x_amax_out[history_idx] = x_tensor_scale[
                                history_idx - 1
                            ]
                            weight_amax_out[history_idx] = weight_tensor_scale[
                                history_idx - 1
                            ]
            cute.arch.sync_threads()
            x_tensor_scale_value = Float32(storage.delayed_scale[0])
            weight_tensor_scale_value = Float32(storage.delayed_scale[3])
            output_scale = x_tensor_scale_value * weight_tensor_scale_value
            x_inv_tensor_scale = Float32(storage.delayed_scale[1])
            weight_inv_tensor_scale = Float32(storage.delayed_scale[4])
            x_quant_multiplier = Float32(storage.delayed_scale[2])
            weight_quant_multiplier = Float32(storage.delayed_scale[5])
        if cutlass.const_expr(self.nvfp4):
            a_packed_layout = cute.recast_layout(8, 4, a_smem_layout)
            b_packed_layout = cute.recast_layout(8, 4, b_smem_layout)
            s_a_packed = storage.q_a.get_tensor(
                a_packed_layout.outer, swizzle=a_packed_layout.inner
            )
            s_b_packed = storage.q_b.get_tensor(
                b_packed_layout.outer, swizzle=b_packed_layout.inner
            )
            s_a_bytes = cute.recast_tensor(s_a_packed, Uint8)
            s_b_bytes = cute.recast_tensor(s_b_packed, Uint8)
            s_a = cute.recast_tensor(s_a_packed, Float4E2M1FN)
            s_b = cute.recast_tensor(s_b_packed, Float4E2M1FN)
        else:
            s_a = storage.q_a.get_tensor(
                a_smem_layout.outer, swizzle=a_smem_layout.inner
            )
            s_b = storage.q_b.get_tensor(
                b_smem_layout.outer, swizzle=b_smem_layout.inner
            )
        s_sfa = storage.scale_a.get_tensor(sfa_smem_layout)
        s_sfb = storage.scale_b.get_tensor(sfb_smem_layout)
        if cutlass.const_expr(self.cluster_output):
            s_cluster_barrier = storage.cluster_barrier.get_tensor(
                cute.make_layout((2,), stride=(1,))
            )
            if warp_idx == 0:
                with cute.arch.elect_one():
                    nvvm.mbarrier_init(
                        s_cluster_barrier.iterator,
                        1,
                    )
                    nvvm.mbarrier_init(
                        s_cluster_barrier.iterator + 1,
                        self.split_reduction,
                    )
            cute.arch.mbarrier_init_fence()
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
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

        if cutlass.const_expr(cfg.cluster_reuse != "none"):
            if cutlass.const_expr(cfg.cluster_reuse == "a"):
                cluster_rows = cfg.tile_m
            else:
                cluster_rows = cfg.tile_n

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

        if cutlass.const_expr(cfg.cluster_reuse != "none"):
            # Establish every peer's dynamic-SMEM allocation once. Subsequent
            # K stages need only publication and release barriers; repeating
            # this rendezvous before every remote store is pure overhead.
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

        total_tiles = m_tiles * n_tiles
        num_k_tiles = cute.ceil_div(self.problem.k, cfg.bf16_tile_k)
        persistent_split_id = Int32(0)
        persistent_split_k_tiles = Int32(num_k_tiles)
        if cutlass.const_expr(
            self.split_reduction > 1 and cfg.persistent
        ):
            persistent_split_id = linear_tile // self.split_grid_ctas
            reduction_tiles = self.reduction_tile // cfg.bf16_tile_k
            persistent_first_k = persistent_split_id * reduction_tiles
            persistent_final_k = cutlass.min(
                persistent_first_k + reduction_tiles, num_k_tiles
            )
            persistent_split_k_tiles = persistent_final_k - persistent_first_k
        for work_slot in cutlass.range_constexpr(self.work_tiles_per_cta):
            split_id = Int32(0)
            if cutlass.const_expr(self.cluster_output):
                split_id = cute.arch.block_idx_in_cluster()
                work_linear = linear_tile // self.split_reduction
            elif cutlass.const_expr(
                self.split_reduction > 1 and cfg.persistent
            ):
                split_id = persistent_split_id
                local_cta = linear_tile - split_id * self.split_grid_ctas
                work_linear = local_cta + work_slot * self.split_grid_ctas
                if cutlass.const_expr(cfg.reuse != "none"):
                    work_linear = (
                        local_cta * self.work_tiles_per_cta + work_slot
                    )
            else:
                work_linear = linear_tile + work_slot * self.grid_ctas
                if cutlass.const_expr(cfg.persistent and cfg.reuse != "none"):
                    # Contiguous chunks preserve an operand coordinate across
                    # adjacent work: N-varying for X, M-varying for weight.
                    work_linear = (
                        linear_tile * self.work_tiles_per_cta + work_slot
                    )
                if cutlass.const_expr(self.split_reduction > 1):
                    split_id = work_linear // total_tiles
                    work_linear = work_linear % total_tiles

            # Mutable PipelineState objects carried across this constexpr
            # output loop produce non-dominating SSA in CuTe DSL. The exact
            # phase is a pure function of the preceding output tiles, so
            # reconstruct it at each slot just as the prequantized GEMM does.
            pipeline_count = work_slot * cute.ceil_div(
                self.problem.k, cfg.bf16_tile_k
            )
            if cutlass.const_expr(
                self.split_reduction > 1 and cfg.persistent
            ):
                pipeline_count = work_slot * persistent_split_k_tiles
            elif cutlass.const_expr(self.split_reduction > 1):
                pipeline_count = 0
            if cutlass.const_expr(cfg.load_engine == "tma"):
                tma_index = pipeline_count % cfg.bf16_stages
                tma_phase = (pipeline_count // cfg.bf16_stages) & 1
                tma_producer_state = pipeline.PipelineState(
                    cfg.bf16_stages,
                    Int32(pipeline_count),
                    Int32(tma_index),
                    Int32(1 ^ tma_phase),
                )
                tma_consumer_state = pipeline.PipelineState(
                    cfg.bf16_stages,
                    Int32(pipeline_count),
                    Int32(tma_index),
                    Int32(tma_phase),
                )
            if cutlass.const_expr(cfg.schedule == "three_role"):
                quant_index = pipeline_count % cfg.mxfp8_stages
                quant_phase = (pipeline_count // cfg.mxfp8_stages) & 1
                quant_producer_state = pipeline.PipelineState(
                    cfg.mxfp8_stages,
                    Int32(pipeline_count),
                    Int32(quant_index),
                    Int32(1 ^ quant_phase),
                )
                quant_consumer_state = pipeline.PipelineState(
                    cfg.mxfp8_stages,
                    Int32(pipeline_count),
                    Int32(quant_index),
                    Int32(quant_phase),
                )
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
                self.split_reduction > 1
                and not self.atomic_output
                and not self.cluster_output
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
    
            blocks_per_load = cfg.bf16_tile_k // self.sf_vec_size
            loads_per_mma_tile = cfg.tile_k // cfg.bf16_tile_k
            # Scale metadata and MMA fragments have the same K quantum for
            # MXFP8 (32), but NVFP4 has four 16-value scale blocks per K=64
            # tensor-core fragment. Keep these address spaces distinct so a
            # 64-wide BF16 transport stage quantizes and consumes exactly one
            # NVFP4 fragment instead of reading an uninitialized half-tile.
            mma_instruction_k = 64 if self.nvfp4 else 32
            mma_blocks_per_load = cfg.bf16_tile_k // mma_instruction_k
            a_scale_blocks = cfg.tile_m * blocks_per_load
            b_scale_blocks = cfg.tile_n * blocks_per_load
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
                scale_block_base = (
                    k_tile % loads_per_mma_tile
                ) * blocks_per_load
                mma_block_base = (
                    k_tile % loads_per_mma_tile
                ) * mma_blocks_per_load
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
                    cp_async_thread_a = cp_async_tiled_copy_a.get_slice(tidx)
                    cp_async_thread_b = cp_async_tiled_copy_b.get_slice(tidx)
                    cluster_rank = Int32(0)
                    if cutlass.const_expr(cfg.cluster_reuse != "none"):
                        cluster_rank = cute.arch.block_idx_in_cluster()
                    # Rank zero is the native-tile owner.  Peers do not fetch
                    # the shared BF16 operand at all: after owner quantization,
                    # its E4M3/E8M0 stage is pushed through DSMEM below.
                    if cutlass.const_expr(cfg.cluster_reuse != "a") or (
                        cluster_rank == 0
                    ):
                        cute.copy(
                            cp_async_tiled_copy_a,
                            cp_async_thread_a.partition_S(g_x_tile),
                            cp_async_thread_a.partition_D(
                                s_bf16_a[None, None, bf16_stage]
                            ),
                        )
                    if cutlass.const_expr(cfg.cluster_reuse != "b") or (
                        cluster_rank == 0
                    ):
                        cute.copy(
                            cp_async_tiled_copy_b,
                            cp_async_thread_b.partition_S(g_weight_tile),
                            cp_async_thread_b.partition_D(
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
                threads_per_scale = self.sf_vec_size // cfg.quant_vec
                scale_in_warp = lane_idx // threads_per_scale
                lane_in_scale = lane_idx % threads_per_scale
                blocks_per_warp = 32 // threads_per_scale
                scale_groups = (
                    a_scale_blocks + b_scale_blocks
                ) // blocks_per_warp
                warp_amax_a = Float32(0.0)
                warp_amax_b = Float32(0.0)
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
                first_task_group = quant_warp_idx
                final_task_group = Int32(scale_groups)
                if cutlass.const_expr(cfg.cluster_reuse != "none"):
                    cluster_rank = cute.arch.block_idx_in_cluster()
                    if cluster_rank != 0:
                        a_task_groups = Int32(a_scale_blocks // blocks_per_warp)
                        if cutlass.const_expr(cfg.cluster_reuse == "a"):
                            first_task_group = a_task_groups + quant_warp_idx
                        else:
                            final_task_group = a_task_groups
                for task_group in cutlass.range(
                    first_task_group,
                    final_task_group,
                    quant_warp_count,
                    unroll=1,
                ):
                    task = task_group * blocks_per_warp + scale_in_warp
                    # A's task count is a multiple of every represented
                    # quant_vec, making this branch warp-uniform.  That is
                    # required by the transposing ldmatrix collective.
                    is_a = task_group * blocks_per_warp < a_scale_blocks
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
                                scale_block * self.sf_vec_size
                                + lane_in_scale * cfg.quant_vec
                                + vec_base
                            )
                            loaded_values = [BFloat16(0.0)] * values_per_load
                            if cutlass.const_expr(cfg.load_engine != "scalar"):
                                if is_a:
                                    if cutlass.const_expr(
                                        self.a_orientation == "transpose"
                                    ):
                                        row_base = row - scale_in_warp
                                        source_stage = s_bf16_a[
                                            None, None, bf16_stage
                                        ]
                                        source_ptr = (
                                            source_stage.iterator
                                            + source_stage.layout(
                                                (row_base, lane_idx)
                                            )
                                        )
                                        packed_rows = nvvm.inline_ptx_hl(
                                            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
                                            "{{$w0}, {$w1}, {$w2}, {$w3}}, "
                                            "[{$r0}];",
                                            write_only_types=[Int32] * 4,
                                            read_only_args=[
                                                Uint32(source_ptr.toint())
                                            ],
                                        )
                                        for register in cutlass.range_constexpr(4):
                                            packed_pair = packed_rows[register]
                                            loaded_values[register * 2] = Uint16(
                                                packed_pair & Int32(0xFFFF)
                                            ).bitcast(BFloat16)
                                            loaded_values[register * 2 + 1] = Uint16(
                                                (packed_pair >> Int32(16))
                                                & Int32(0xFFFF)
                                            ).bitcast(BFloat16)
                                    else:
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
                                            loaded_values[load_vec] = loaded[load_vec]
                                else:
                                    if cutlass.const_expr(
                                        self.b_orientation == "transpose"
                                    ):
                                        row_base = row - scale_in_warp
                                        source_stage = s_bf16_b[
                                            None, None, bf16_stage
                                        ]
                                        source_ptr = (
                                            source_stage.iterator
                                            + source_stage.layout(
                                                (row_base, lane_idx)
                                            )
                                        )
                                        packed_rows = nvvm.inline_ptx_hl(
                                            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
                                            "{{$w0}, {$w1}, {$w2}, {$w3}}, "
                                            "[{$r0}];",
                                            write_only_types=[Int32] * 4,
                                            read_only_args=[
                                                Uint32(source_ptr.toint())
                                            ],
                                        )
                                        for register in cutlass.range_constexpr(4):
                                            packed_pair = packed_rows[register]
                                            loaded_values[register * 2] = Uint16(
                                                packed_pair & Int32(0xFFFF)
                                            ).bitcast(BFloat16)
                                            loaded_values[register * 2 + 1] = Uint16(
                                                (packed_pair >> Int32(16))
                                                & Int32(0xFFFF)
                                            ).bitcast(BFloat16)
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
                                            loaded_values[load_vec] = loaded[load_vec]
                                for load_vec in cutlass.range_constexpr(
                                    values_per_load
                                ):
                                    bf16_values[vec_base + load_vec] = loaded_values[
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
                            scale_block * self.sf_vec_size
                            + lane_in_scale * cfg.quant_vec
                            + vec
                        )
                        # ldmatrix.x4.trans distributes one two-BF16 pair from
                        # each 8-K matrix to a lane: register r owns
                        # K=8*r+2*lane_in_scale+{0,1}. Preserve that measured
                        # delivery rather than pretending each lane received
                        # eight consecutive K values.
                        ldmatrix_k = (
                            scale_block * self.sf_vec_size
                            + (vec // 2) * 8
                            + lane_in_scale * 2
                            + vec % 2
                        )
                        if cutlass.const_expr(
                            self.a_orientation == "transpose"
                            and self.b_orientation == "transpose"
                            and cfg.quant_load_bits == 128
                        ):
                            bf16_k = ldmatrix_k
                        elif cutlass.const_expr(
                            self.a_orientation == "transpose"
                            and cfg.quant_load_bits == 128
                        ):
                            if is_a:
                                bf16_k = ldmatrix_k
                        elif cutlass.const_expr(
                            self.b_orientation == "transpose"
                            and cfg.quant_load_bits == 128
                        ):
                            if not is_a:
                                bf16_k = ldmatrix_k
                        local_k = scale_block_base * self.sf_vec_size + bf16_k
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
                    if cutlass.const_expr(
                        self.nvfp4 and self.config.collect_amax
                    ):
                        if is_a:
                            if cutlass.const_expr(
                                cfg.telemetry_ownership == "all"
                            ) or block_n == 0:
                                warp_amax_a = cute.arch.fmax(
                                    warp_amax_a, amax, nan=True
                                )
                        else:
                            if cutlass.const_expr(
                                cfg.telemetry_ownership == "all"
                            ) or block_m == 0:
                                warp_amax_b = cute.arch.fmax(
                                    warp_amax_b, amax, nan=True
                                )
                    if cutlass.const_expr(self.nvfp4):
                        tensor_scale = (
                            x_tensor_scale_value
                            if is_a
                            else weight_tensor_scale_value
                        )
                        scale_e8m0 = Float8E4M3FN(1.0)
                        inv_scale_fp32 = Float32(1.0)
                        if is_a:
                            scale_e8m0, inv_scale_fp32 = (
                                self._nvfp4_scale_from_amax(
                                    amax,
                                    tensor_scale,
                                    x_inv_tensor_scale,
                                    x_quant_multiplier,
                                )
                            )
                        else:
                            scale_e8m0, inv_scale_fp32 = (
                                self._nvfp4_scale_from_amax(
                                    amax,
                                    tensor_scale,
                                    weight_inv_tensor_scale,
                                    weight_quant_multiplier,
                                )
                            )
                    else:
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
                        if cutlass.const_expr(self.nvfp4):
                            packed = nvvm.inline_ptx_hl(
                                "{.reg .b8 b; "
                                "cvt.rn.satfinite.e2m1x2.f32 b, {$r1}, {$r0}; "
                                "mov.b16 {$w0}, {b, 0};}",
                                write_only_types=[Int16],
                                read_only_args=[
                                    values[vec0] * inv_scale_fp32,
                                    values[vec1] * inv_scale_fp32,
                                ],
                            )
                            quantized0 = Uint8(packed)
                            quantized1 = Uint8(0)
                        elif cutlass.const_expr(cfg.quant_math == "bf16x2"):
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
    
                        if cutlass.const_expr(self.nvfp4):
                            if is_a:
                                s_a_bytes[
                                    row, local_ks[vec0] // 2, stage
                                ] = quantized0
                            else:
                                s_b_bytes[
                                    row, local_ks[vec0] // 2, stage
                                ] = quantized0
                        else:
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
                            scale_block_base + scale_block
                        ) * self.sf_vec_size
                        if is_a:
                            s_sfa[row, scale_k, stage] = scale_e8m0
                        else:
                            s_sfb[row, scale_k, stage] = scale_e8m0

                if cutlass.const_expr(
                    self.nvfp4 and self.config.collect_amax
                ):
                    warp_amax_a = nvvm.redux_sync(
                        warp_amax_a.bitcast(Int32),
                        nvvm.ReductionKind.UMAX,
                        Int32(0xFFFFFFFF),
                    ).bitcast(Float32)
                    warp_amax_b = nvvm.redux_sync(
                        warp_amax_b.bitcast(Int32),
                        nvvm.ReductionKind.UMAX,
                        Int32(0xFFFFFFFF),
                    ).bitcast(Float32)
                    if lane_idx == 0:
                        telemetry_index = linear_tile
                        if cutlass.const_expr(
                            cfg.telemetry_layout == "scalar_atomic"
                        ):
                            telemetry_index = 0
                        cute.arch.atomic_fmax(
                            x_amax_out.iterator
                            + x_amax_out.layout(telemetry_index),
                            warp_amax_a,
                            sign_bit=False,
                            scope="gpu",
                        )
                        cute.arch.atomic_fmax(
                            weight_amax_out.iterator
                            + weight_amax_out.layout(telemetry_index),
                            warp_amax_b,
                            sign_bit=False,
                            scope="gpu",
                        )
    
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

                if cutlass.const_expr(cfg.cluster_reuse != "none"):
                    # Every role reaches the cluster barrier before CTA 0
                    # publishes the shared native tile. SM120 supports remote
                    # DSMEM stores but remote loads fault, and ldmatrix accepts
                    # local shared addresses only. The owner therefore pushes
                    # E4M3/E8M0 bytes directly into each peer's local stage;
                    # peers perform no BF16 reload, amax, scaling, or convert.
                    # This rendezvous cannot be folded into the previous
                    # release: without it, independent role progress permits
                    # the owner to publish while a peer still owns the stage,
                    # producing sparse but repeatable corruption on SM120.
                    cute.arch.cluster_arrive()
                    cute.arch.cluster_wait()
                    cluster_rank = cute.arch.block_idx_in_cluster()
                    if cluster_rank == 0:
                        for peer in cutlass.range_constexpr(1, cfg.cluster_size):
                            if cutlass.const_expr(cfg.cluster_reuse == "a"):
                                if cutlass.const_expr(self.nvfp4):
                                    peer_q_ptr = cute.arch.map_dsmem_ptr(
                                        s_a_bytes.iterator, Int32(peer)
                                    )
                                else:
                                    peer_q_ptr = cute.arch.map_dsmem_ptr(
                                        s_a.iterator, Int32(peer)
                                    )
                                peer_sf_ptr = cute.arch.map_dsmem_ptr(
                                    s_sfa.iterator, Int32(peer)
                                )
                            else:
                                if cutlass.const_expr(self.nvfp4):
                                    peer_q_ptr = cute.arch.map_dsmem_ptr(
                                        s_b_bytes.iterator, Int32(peer)
                                    )
                                else:
                                    peer_q_ptr = cute.arch.map_dsmem_ptr(
                                        s_b.iterator, Int32(peer)
                                    )
                                peer_sf_ptr = cute.arch.map_dsmem_ptr(
                                    s_sfb.iterator, Int32(peer)
                                )
                            if cutlass.const_expr(self.nvfp4):
                                packed_k = cfg.bf16_tile_k // 2
                                for item in cutlass.range(
                                    tidx,
                                    cluster_rows * packed_k // 4,
                                    cfg.num_threads,
                                    unroll=1,
                                ):
                                    q_byte_base = item * 4
                                    q_row = q_byte_base // packed_k
                                    q_byte_k = (
                                        scale_block_base * self.sf_vec_size // 2
                                        + q_byte_base % packed_k
                                    )
                                    if cutlass.const_expr(
                                        cfg.cluster_reuse == "a"
                                    ):
                                        q_offset = s_a_bytes.layout(
                                            (q_row, q_byte_k, stage)
                                        )
                                        q0 = s_a_bytes[q_row, q_byte_k, stage]
                                        q1 = s_a_bytes[q_row, q_byte_k + 1, stage]
                                        q2 = s_a_bytes[q_row, q_byte_k + 2, stage]
                                        q3 = s_a_bytes[q_row, q_byte_k + 3, stage]
                                    else:
                                        q_offset = s_b_bytes.layout(
                                            (q_row, q_byte_k, stage)
                                        )
                                        q0 = s_b_bytes[q_row, q_byte_k, stage]
                                        q1 = s_b_bytes[q_row, q_byte_k + 1, stage]
                                        q2 = s_b_bytes[q_row, q_byte_k + 2, stage]
                                        q3 = s_b_bytes[q_row, q_byte_k + 3, stage]
                                    q_word = (
                                        Uint32(q0)
                                        | (Uint32(q1) << 8)
                                        | (Uint32(q2) << 16)
                                        | (Uint32(q3) << 24)
                                    )
                                    nvvm.inline_ptx_hl(
                                        "st.shared::cluster.u32 [{$r0}], {$r1};",
                                        write_only_types=[],
                                        read_only_args=[
                                            Uint32(
                                                (peer_q_ptr + q_offset).toint()
                                            ),
                                            q_word,
                                        ],
                                    )
                            else:
                                for item in cutlass.range(
                                    tidx,
                                    cluster_rows * cfg.bf16_tile_k // 4,
                                    cfg.num_threads,
                                    unroll=1,
                                ):
                                    q_base = item * 4
                                    q_row = q_base // cfg.bf16_tile_k
                                    q_k = (
                                        scale_block_base * self.sf_vec_size
                                        + q_base % cfg.bf16_tile_k
                                    )
                                    if cutlass.const_expr(
                                        cfg.cluster_reuse == "a"
                                    ):
                                        q_offset = s_a.layout((q_row, q_k, stage))
                                        q0 = s_a[q_row, q_k, stage]
                                        q1 = s_a[q_row, q_k + 1, stage]
                                        q2 = s_a[q_row, q_k + 2, stage]
                                        q3 = s_a[q_row, q_k + 3, stage]
                                    else:
                                        q_offset = s_b.layout((q_row, q_k, stage))
                                        q0 = s_b[q_row, q_k, stage]
                                        q1 = s_b[q_row, q_k + 1, stage]
                                        q2 = s_b[q_row, q_k + 2, stage]
                                        q3 = s_b[q_row, q_k + 3, stage]
                                    q_word = (
                                        Uint32(q0.bitcast(Uint8))
                                        | (Uint32(q1.bitcast(Uint8)) << 8)
                                        | (Uint32(q2.bitcast(Uint8)) << 16)
                                        | (Uint32(q3.bitcast(Uint8)) << 24)
                                    )
                                    nvvm.inline_ptx_hl(
                                        "st.shared::cluster.u32 [{$r0}], {$r1};",
                                        write_only_types=[],
                                        read_only_args=[
                                            Uint32(
                                                (peer_q_ptr + q_offset).toint()
                                            ),
                                            q_word,
                                        ],
                                    )
                            for item in cutlass.range(
                                tidx,
                                cluster_rows * blocks_per_load,
                                cfg.num_threads,
                                unroll=1,
                            ):
                                sf_row = item // blocks_per_load
                                sf_k = (
                                    scale_block_base + item % blocks_per_load
                                ) * self.sf_vec_size
                                if cutlass.const_expr(cfg.cluster_reuse == "a"):
                                    sf_offset = s_sfa.layout((sf_row, sf_k, stage))
                                    sf_value = s_sfa[sf_row, sf_k, stage]
                                else:
                                    sf_offset = s_sfb.layout((sf_row, sf_k, stage))
                                    sf_value = s_sfb[sf_row, sf_k, stage]
                                nvvm.inline_ptx_hl(
                                    "st.shared::cluster.u8 [{$r0}], {$r1};",
                                    write_only_types=[],
                                    read_only_args=[
                                        Uint32((peer_sf_ptr + sf_offset).toint()),
                                        Uint32(sf_value.bitcast(Uint8)),
                                    ],
                                )
                    cute.arch.cluster_arrive()
                    cute.arch.cluster_wait()

                num_k_blocks = cute.size(t_cr_a, mode=[2])
                if warp_idx < self.num_mma_warps:
                    for k_block in cutlass.range_constexpr(num_k_blocks):
                        if (
                            k_block >= mma_block_base
                            and k_block < mma_block_base + mma_blocks_per_load
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

                if cutlass.const_expr(cfg.cluster_reuse != "none"):
                    # CTA 0 must not recycle the shared stage until every peer
                    # has completed its remote ldmatrix/scale-fragment reads.
                    cute.arch.cluster_arrive()
                    cute.arch.cluster_wait()

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
                self.split_reduction > 1 and not cfg.persistent
            ):
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
                        and warp_idx
                        < self.num_mma_warps + cfg.quantizer_warps
                    ):
                        quant_pipeline.producer_tail(quant_producer_state)
    
            if cutlass.const_expr(cfg.epilogue == "direct"):
                if cutlass.const_expr(self.cluster_output):
                    # The quantized A stages are dead after the mainloop. Reuse
                    # them as a chunked FP32 DSMEM reduction window so fused
                    # dynamic quantization never materializes partials in GMEM.
                    scratch_elements = cute.cosize(a_smem_layout) // 4
                    mma_threads = self.num_mma_warps * 32
                    accum_elements = cute.size(accumulators)
                    chunk_elements = cutlass.min(
                        accum_elements,
                        scratch_elements // mma_threads,
                    )
                    scratch = cute.make_tensor(
                        cute.recast_ptr(s_a.iterator, dtype=Float32),
                        cute.make_layout((scratch_elements,), stride=(1,)),
                    )
                    cluster_rank = cute.arch.block_idx_in_cluster()
                    leader_ptr = cute.arch.map_dsmem_ptr(
                        scratch.iterator, Int32(0)
                    )
                    chunk_count = cute.ceil_div(
                        accum_elements, chunk_elements
                    )
                    # Producer and quantizer tails are warp-local. Publish their
                    # completion CTA-wide before overlaying the operand stages.
                    cute.arch.sync_threads()
                    for chunk in cutlass.range_constexpr(chunk_count):
                        first_elem = chunk * chunk_elements
                        final_elem = cutlass.min(
                            first_elem + chunk_elements, accum_elements
                        )
                        if cluster_rank == 0 and warp_idx < self.num_mma_warps:
                            for elem in cutlass.range(
                                first_elem, final_elem, unroll_full=True
                            ):
                                scratch[
                                    tidx * chunk_elements + elem - first_elem
                                ] = Float32(0.0)
                        self._cluster_broadcast(
                            s_cluster_barrier,
                            0,
                            Int32((chunk * 2) & 1),
                            cluster_rank,
                        )
                        if warp_idx < self.num_mma_warps:
                            for elem in cutlass.range(
                                first_elem, final_elem, unroll_full=True
                            ):
                                nvvm.red(
                                    nvvm.ReductionOp.ADD,
                                    nvvm.ReductionType.F32,
                                    leader_ptr
                                    + tidx * chunk_elements
                                    + elem
                                    - first_elem,
                                    accumulators[elem],
                                    mem_order="relaxed",
                                    mem_scope="cluster",
                                )
                        self._cluster_gather(
                            s_cluster_barrier,
                            1,
                            Int32(chunk & 1),
                            cluster_rank,
                        )
                        if cluster_rank == 0 and warp_idx < self.num_mma_warps:
                            for elem in cutlass.range(
                                first_elem, final_elem, unroll_full=True
                            ):
                                coord = t_cc_out[elem]
                                if (
                                    coord[0] < self.problem.m
                                    and coord[1] < self.problem.n
                                ):
                                    t_cg_out[elem] = BFloat16(
                                        scratch[
                                            tidx * chunk_elements
                                            + elem
                                            - first_elem
                                        ]
                                    )
                        self._cluster_broadcast(
                            s_cluster_barrier,
                            0,
                            Int32((chunk * 2 + 1) & 1),
                            cluster_rank,
                        )
                elif warp_idx < self.num_mma_warps:
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
                                t_cg_out[elem] = self.c_dtype(
                                    accumulators[elem] * output_scale
                                )
            elif cutlass.const_expr(cfg.epilogue == "tma"):
                epilogue_stage = Int32(work_slot % cfg.epilogue_stages)
                full_output_tile = (
                    (block_m + 1) * cfg.tile_m <= self.problem.m
                    and (block_n + 1) * cfg.tile_n <= self.problem.n
                )
                if full_output_tile:
                    if warp_idx < self.num_mma_warps:
                        for elem in cutlass.range(
                            cute.size(t_rs_r_out), unroll_full=True
                        ):
                            t_rs_r_out[elem] = BFloat16(
                                t_rs_r_acc[elem] * output_scale
                            )
                        # Warp 0 reaches this only after its previous TMA wait;
                        # use that arrival to gate stage reuse by all MMA warps.
                        self.epilogue_sync_barrier.arrive_and_wait()
                        cute.copy(
                            copy_r2s,
                            t_rs_r_out,
                            t_rs_s_out[
                                (None, None, None, epilogue_stage)
                            ],
                        )
                        cute.arch.fence_proxy("async.shared", space="cta")
                        self.epilogue_sync_barrier.arrive_and_wait()
                        if warp_idx == 0:
                            cute.copy(
                                tma_atom_out,
                                b_sg_s_out[(None, epilogue_stage)],
                                b_sg_g_out[(None, 0)],
                            )
                            tma_store_pipeline.producer_commit()
                            tma_store_pipeline.producer_acquire()
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
                                t_cg_out[elem] = BFloat16(
                                    accumulators[elem] * output_scale
                                )

        if cutlass.const_expr(
            (self.split_reduction == 1 or cfg.persistent)
            and cfg.load_engine == "tma"
            and cfg.schedule in ("warp_specialized", "three_role")
        ):
            final_k_count = Int32(num_k_tiles)
            if cutlass.const_expr(self.split_reduction > 1):
                final_k_count = persistent_split_k_tiles
            final_count = self.work_tiles_per_cta * final_k_count
            tma_final_index = final_count % cfg.bf16_stages
            tma_final_phase = (final_count // cfg.bf16_stages) & 1
            tma_final_state = pipeline.PipelineState(
                cfg.bf16_stages,
                Int32(final_count),
                Int32(tma_final_index),
                Int32(1 ^ tma_final_phase),
            )
            producer_warp = self.num_mma_warps
            if cutlass.const_expr(cfg.schedule == "three_role"):
                producer_warp += cfg.quantizer_warps
            if warp_idx == producer_warp:
                tma_pipeline.producer_tail(tma_final_state)

        if cutlass.const_expr(
            (self.split_reduction == 1 or cfg.persistent)
            and cfg.schedule == "three_role"
        ):
            final_k_count = Int32(num_k_tiles)
            if cutlass.const_expr(self.split_reduction > 1):
                final_k_count = persistent_split_k_tiles
            final_count = self.work_tiles_per_cta * final_k_count
            quant_final_index = final_count % cfg.mxfp8_stages
            quant_final_phase = (final_count // cfg.mxfp8_stages) & 1
            quant_final_state = pipeline.PipelineState(
                cfg.mxfp8_stages,
                Int32(final_count),
                Int32(quant_final_index),
                Int32(1 ^ quant_final_phase),
            )
            if (
                warp_idx >= self.num_mma_warps
                and warp_idx < self.num_mma_warps + cfg.quantizer_warps
            ):
                quant_pipeline.producer_tail(quant_final_state)

        if cutlass.const_expr(cfg.epilogue == "tma"):
            if warp_idx == 0:
                # Earlier stores remain in flight while later output tiles run;
                # only the final groups must drain before the CTA exits.
                cute.arch.cp_async_bulk_wait_group(0, read=True)


class _UnitScaleLauncher:
    """Preserve the established three-tensor MXFP8 launcher contract."""

    def __init__(self, launcher):
        self.launcher = launcher

    def __call__(self, x, weight, destination):
        # The common generated signature reserves four FP32 pointers for the
        # NVFP4 tensor-scale/amax path.  MXFP8 never dereferences them, so use
        # an allocation-free view of two adjacent BF16 storage elements.  Do
        # not require a logically contiguous innermost dimension: backward
        # intentionally presents transposed CuTe layouts without materializing
        # a physical transpose.
        placeholder = x.as_strided(
            (2,), (1,), storage_offset=x.storage_offset()
        ).view(torch.float32)
        return self.launcher(
            x,
            weight,
            destination,
            placeholder,
            placeholder,
            placeholder,
            placeholder,
        )


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
    a_shape = (
        (problem.m, problem.k)
        if a_orientation == "row"
        else (problem.k, problem.m)
    )
    b_shape = (
        (problem.n, problem.k)
        if b_orientation == "row"
        else (problem.k, problem.n)
    )
    a_stride = (a_shape[1], 1)
    b_stride = (b_shape[1], 1)
    x = cute.runtime.make_fake_tensor(
        BFloat16,
        a_shape,
        stride=a_stride,
        assumed_align=16,
    )
    weight = cute.runtime.make_fake_tensor(
        BFloat16,
        b_shape,
        stride=b_stride,
        assumed_align=16,
    )
    out = cute.runtime.make_fake_tensor(
        BFloat16,
        (problem.m, problem.n),
        stride=(problem.n, 1),
        assumed_align=16,
    )
    tensor_scale = cute.runtime.make_fake_tensor(
        Float32, (1,), stride=(1,), assumed_align=4
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    launcher = cute.compile(
        kernel,
        x,
        weight,
        out,
        tensor_scale,
        tensor_scale,
        tensor_scale,
        tensor_scale,
        stream,
        options=(
            "--enable-tvm-ffi --opt-level 3 "
            f"--ptxas-options '-O3 -v --maxrregcount={config.maxrregcount}'"
        ),
    )
    return _UnitScaleLauncher(launcher)


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
    a_shape = (problem.m, problem.k) if a_orientation == "row" else (problem.k, problem.m)
    b_shape = (problem.n, problem.k) if b_orientation == "row" else (problem.k, problem.n)
    a_stride = (a_shape[1], 1)
    b_stride = (b_shape[1], 1)
    x = cute.runtime.make_fake_tensor(
        BFloat16,
        a_shape,
        stride=a_stride,
        assumed_align=16,
    )
    weight = cute.runtime.make_fake_tensor(
        BFloat16,
        b_shape,
        stride=b_stride,
        assumed_align=16,
    )
    workspace = cute.runtime.make_fake_tensor(
        Float32,
        (split_reduction * problem.m * problem.n,),
        stride=(1,),
        assumed_align=16,
    )
    tensor_scale = cute.runtime.make_fake_tensor(
        Float32, (1,), stride=(1,), assumed_align=4
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    launcher = cute.compile(
        kernel,
        x,
        weight,
        workspace,
        tensor_scale,
        tensor_scale,
        tensor_scale,
        tensor_scale,
        stream,
        options=(
            "--enable-tvm-ffi --opt-level 3 "
            f"--ptxas-options '-O3 -v --maxrregcount={config.maxrregcount}'"
        ),
    )
    return _UnitScaleLauncher(launcher)


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
    a_shape = (problem.m, problem.k) if a_orientation == "row" else (problem.k, problem.m)
    b_shape = (problem.n, problem.k) if b_orientation == "row" else (problem.k, problem.n)
    a_stride = (a_shape[1], 1)
    b_stride = (b_shape[1], 1)
    x = cute.runtime.make_fake_tensor(
        BFloat16,
        a_shape,
        stride=a_stride,
        assumed_align=16,
    )
    weight = cute.runtime.make_fake_tensor(
        BFloat16,
        b_shape,
        stride=b_stride,
        assumed_align=16,
    )
    accumulator = cute.runtime.make_fake_tensor(
        Float32,
        (problem.m, problem.n),
        stride=(problem.n, 1),
        assumed_align=16,
    )
    tensor_scale = cute.runtime.make_fake_tensor(
        Float32, (1,), stride=(1,), assumed_align=4
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    launcher = cute.compile(
        kernel,
        x,
        weight,
        accumulator,
        tensor_scale,
        tensor_scale,
        tensor_scale,
        tensor_scale,
        stream,
        options=(
            "--enable-tvm-ffi --opt-level 3 "
            f"--ptxas-options '-O3 -v --maxrregcount={config.maxrregcount}'"
        ),
    )
    return _UnitScaleLauncher(launcher)


@lru_cache(maxsize=None)
def compile_mxfp8_cluster_split_fwd(
    problem: MXFP8Problem,
    config: MXFP8FwdConfig,
    *,
    a_orientation: str,
    b_orientation: str,
    split_reduction: int,
    reduction_tile: int,
):
    """Compile fused split-K with cluster-local FP32 reduction to BF16."""

    problem.validate()
    kernel = MXFP8LinearFwdKernel(
        problem,
        config,
        a_orientation=a_orientation,
        b_orientation=b_orientation,
        split_reduction=split_reduction,
        reduction_tile=reduction_tile,
        cluster_output=True,
    )
    a_shape = (problem.m, problem.k) if a_orientation == "row" else (problem.k, problem.m)
    b_shape = (problem.n, problem.k) if b_orientation == "row" else (problem.k, problem.n)
    a_stride = (a_shape[1], 1)
    b_stride = (b_shape[1], 1)
    x = cute.runtime.make_fake_tensor(
        BFloat16,
        a_shape,
        stride=a_stride,
        assumed_align=16,
    )
    weight = cute.runtime.make_fake_tensor(
        BFloat16,
        b_shape,
        stride=b_stride,
        assumed_align=16,
    )
    out = cute.runtime.make_fake_tensor(
        BFloat16,
        (problem.m, problem.n),
        stride=(problem.n, 1),
        assumed_align=16,
    )
    tensor_scale = cute.runtime.make_fake_tensor(
        Float32, (1,), stride=(1,), assumed_align=4
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    launcher = cute.compile(
        kernel,
        x,
        weight,
        out,
        tensor_scale,
        tensor_scale,
        tensor_scale,
        tensor_scale,
        stream,
        options=(
            "--enable-tvm-ffi --opt-level 3 "
            f"--ptxas-options '-O3 -v --maxrregcount={config.maxrregcount}'"
        ),
    )
    return _UnitScaleLauncher(launcher)


__all__ = [
    "MXFP8LinearFwdKernel",
    "compile_mxfp8_fwd",
    "compile_mxfp8_atomic_split_fwd",
    "compile_mxfp8_cluster_split_fwd",
    "compile_mxfp8_split_fwd",
]
