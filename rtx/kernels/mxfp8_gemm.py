"""Pure prequantized MXFP8 GEMM for RTX Blackwell SM120."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import os

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
    Float8E4M3FN,
    Float8E8M0FNU,
    Float32,
    Int16,
    Int32,
    Int64,
    Uint8,
)
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.experimental.primitives import nvvm_wrapper as nvvm

from ..configs.mxfp8 import MXFP8GemmConfig
from .mxfp8 import MXFP8Problem
from .mxfp8_fwd import (
    SF_VEC_SIZE,
    _make_ldmatrix_atom,
    _make_scale_s2r_atom,
    _make_sm120_sfa_layout_64,
)


class MXFP8GemmKernel:
    def __init__(
        self,
        problem: MXFP8Problem,
        config: MXFP8GemmConfig,
        *,
        split_reduction: int = 1,
        reduction_tile: int = 0,
        atomic_output: bool = False,
        cluster_output: bool = False,
        use_pdl: bool = False,
    ):
        if problem.k % config.tile_k:
            # Qdata TMA naturally zero-fills a short K tile. Scale vectors are
            # byte-addressed separately, and a wide GMEM load would either
            # cross the row boundary or require alignment based on the padded
            # macro tile. Use predicated scalar scale loads only for this final
            # tile; full-tile configurations retain their autotuned vector path.
            config = replace(
                config,
                scale_load_vec=1,
                scale_smem_store="scalar",
                scale_l2_prefetch="none",
                scale_l1_evict="default",
                scale_cache="default",
            )
        self.ab_dtype = Float8E4M3FN
        self.sf_dtype = Float8E8M0FNU
        self.sf_vec_size = SF_VEC_SIZE
        self.use_mxf8f6f4 = True
        self.apply_output_scale = False
        if not hasattr(self, "cache_regional_output_scales"):
            self.cache_regional_output_scales = False
        if not hasattr(self, "output_x_scale_cache_elems"):
            self.output_x_scale_cache_elems = 0
        if not hasattr(self, "output_w_scale_cache_elems"):
            self.output_w_scale_cache_elems = 0
        if not hasattr(self, "output_scale_product_cache_elems"):
            self.output_scale_product_cache_elems = 0
        if not hasattr(self, "cache_output_scale_products"):
            self.cache_output_scale_products = False
        if not hasattr(self, "tile_uniform_regional_output_scale"):
            self.tile_uniform_regional_output_scale = False
        if not hasattr(self, "direct_regional_output_scales"):
            self.direct_regional_output_scales = False
        if not hasattr(self, "warp_specialized_epilogue"):
            self.warp_specialized_epilogue = False
        if not hasattr(self, "num_epilogue_warps"):
            self.num_epilogue_warps = 0
        if not hasattr(self, "regional_scale_cache_stages"):
            self.regional_scale_cache_stages = 1
        if not hasattr(self, "epilogue_accumulator_elements"):
            self.epilogue_accumulator_elements = 0
        rejection = config.rejection(problem)
        if rejection is not None:
            raise ValueError(f"illegal prequantized MXFP8 GEMM: {rejection}")
        self.problem = problem
        self.config = config
        self.split_reduction = split_reduction
        self.reduction_tile = reduction_tile
        self.atomic_output = atomic_output
        self.cluster_output = cluster_output
        self.use_pdl = use_pdl
        if split_reduction < 1:
            raise ValueError("split_reduction must be positive")
        if split_reduction == 1 and (
            reduction_tile != 0 or atomic_output or cluster_output
        ):
            raise ValueError("an unsplit GEMM must use reduction_tile=0")
        if atomic_output and cluster_output:
            raise ValueError("atomic and cluster outputs are mutually exclusive")
        if config.persistent_waves and split_reduction > 1:
            raise ValueError("balanced persistent grids currently require unsplit GEMM")
        if split_reduction > 1:
            if config.epilogue != "direct":
                raise ValueError("split-K prequant GEMM requires direct FP32 output")
            if config.tiles_per_cta != 1:
                raise ValueError("split-K and multi-output persistence are independent")
            if cluster_output and split_reduction not in (2, 4, 8):
                raise ValueError("SM120 CTA clusters support split counts 2, 4, or 8")
            if reduction_tile <= 0 or reduction_tile % config.tile_k:
                raise ValueError("reduction_tile must be a positive tile-K multiple")
            if not (
                (split_reduction - 1) * reduction_tile < problem.k
                <= split_reduction * reduction_tile
            ):
                raise ValueError("split count/tile must cover K without an empty slice")
        self.tile_shape_mnk = (config.tile_m, config.tile_n, config.tile_k)
        self.m_tiles = (problem.m + config.tile_m - 1) // config.tile_m
        self.n_tiles = (problem.n + config.tile_n - 1) // config.tile_n
        self.total_tiles = self.m_tiles * self.n_tiles
        self.total_work_tiles = self.total_tiles * split_reduction
        capped_grid = (
            self.total_work_tiles + config.tiles_per_cta - 1
        ) // config.tiles_per_cta
        self.grid_ctas = capped_grid
        if config.persistent_waves:
            sm_count = utils.HardwareInfo().get_device_multiprocessor_count()
            wave_grid = min(
                self.total_work_tiles,
                sm_count * config.persistent_waves,
            )
            self.grid_ctas = max(capped_grid, wave_grid)
        self.work_tiles_per_cta = (
            (self.total_work_tiles + self.grid_ctas - 1) // self.grid_ctas
            if config.persistent_waves
            else config.tiles_per_cta
        )
        self.num_k_tiles = (problem.k + config.tile_k - 1) // config.tile_k
        scale_threads = config.num_mma_warps * 32
        if config.scale_role == "producer":
            scale_threads += 32
        self.scale_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=scale_threads
        )
        self.epilogue_barrier = pipeline.NamedBarrier(
            barrier_id=2, num_threads=config.num_mma_warps * 32
        )
        epilogue_threads = (
            config.num_mma_warps + self.num_epilogue_warps
        ) * 32
        self.epilogue_ready_barrier = pipeline.NamedBarrier(
            barrier_id=3, num_threads=epilogue_threads
        )
        self.epilogue_free_barrier = pipeline.NamedBarrier(
            barrier_id=4, num_threads=epilogue_threads
        )
        self.regional_cache_barrier = pipeline.NamedBarrier(
            barrier_id=5, num_threads=config.num_mma_warps * 32
        )

    @cute.jit
    def _cluster_broadcast(
        self,
        barriers: cute.Tensor,
        barrier_index: cutlass.Constexpr,
        phase: Int32,
        cluster_rank: Int32,
    ) -> None:
        """Rank 0 publishes a phase to one local mbarrier in every CTA."""
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
        """Every CTA publishes to rank 0; only rank 0 waits for completion."""
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
        mma_op = self._make_mma_op()
        self.tiled_mma = cute.make_tiled_mma(
            mma_op,
            cute.make_layout((cfg.atom_layout_m, cfg.atom_layout_n, 1)),
            permutation_mnk=sm120_utils.get_permutation_mnk(
                self.tile_shape_mnk, self.sf_vec_size, self.use_mxf8f6f4
            ),
        )
        swizzles = {
            "none": cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_INTER,
            "32b": cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW32,
            "64b": cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW64,
            "128b": cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW128,
        }
        self.a_layout = cute.tile_to_shape(
            cute.nvgpu.warpgroup.make_smem_layout_atom(
                swizzles[cfg.a_swizzle], self.ab_dtype
            ),
            (cfg.tile_m, cfg.tile_k, cfg.stages),
            order=(0, 1, 2),
        )
        self.b_layout = cute.tile_to_shape(
            cute.nvgpu.warpgroup.make_smem_layout_atom(
                swizzles[cfg.b_swizzle], self.ab_dtype
            ),
            (cfg.tile_n, cfg.tile_k, cfg.stages),
            order=(0, 1, 2),
        )
        padded_tile = (
            ((cfg.tile_m + 127) // 128) * 128,
            ((cfg.tile_n + 127) // 128) * 128,
            cfg.tile_k,
        )
        if cfg.tile_m == 64:
            self.sfa_layout = _make_sm120_sfa_layout_64(
                self.tiled_mma,
                cfg.tile_k,
                self.sf_vec_size,
                cfg.stages,
            )
        else:
            self.sfa_layout = blockscaled_utils.sm120_make_smem_layout_sfa(
                self.tiled_mma, padded_tile, self.sf_vec_size, cfg.stages
            )
        self.sfb_layout = blockscaled_utils.sm120_make_smem_layout_sfb(
            self.tiled_mma, padded_tile, self.sf_vec_size, cfg.stages
        )
        self.out_layout = cute.tile_to_shape(
            cute.nvgpu.warpgroup.make_smem_layout_atom(
                cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW128,
                BFloat16,
            ),
            (cfg.tile_m, cfg.tile_n, cfg.epilogue_stages),
            order=(0, 1, 2),
        )

    def _make_mma_op(self):
        return warp.MmaMXF8Op(self.ab_dtype, Float32, self.sf_dtype)

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
        """Resolve an output element's outer scale.

        MXFP8 and ordinary NVFP4 use a scalar.  NVFP4 JIT row-region scaling
        overrides this hook so the native GEMM can apply the product of its X
        and weight region scales without a separate pointwise kernel.
        """

        return global_output_scale

    @cute.jit
    def _load_x_output_scale(self, output_scale: cute.Tensor, row: Int32):
        return Float32(output_scale[0])

    @cute.jit
    def _load_weight_output_scale(
        self, output_scale: cute.Tensor, column: Int32
    ):
        return Float32(output_scale[0])

    @cute.jit
    def _x_output_scale_cache_offset(self, cache_index: Int32):
        return cache_index

    @cute.jit
    def _w_output_scale_cache_offset(self, cache_index: Int32):
        return cache_index

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
        return value

    @cute.jit
    def _stage_scales(
        self,
        sx: cute.Tensor,
        sw: cute.Tensor,
        s_sfa: cute.Tensor,
        s_sfb: cute.Tensor,
        stage: Int32,
        block_m: Int32,
        block_n: Int32,
        k_tile: Int32,
        start_thread: Int32,
        thread_count: cutlass.Constexpr,
    ):
        cfg = self.config
        scale_prefetch = {
            "none": None,
            "64b": nvvm.L2PrefetchSize.SIZE_64B,
            "128b": nvvm.L2PrefetchSize.SIZE_128B,
            "256b": nvvm.L2PrefetchSize.SIZE_256B,
        }[cfg.scale_l2_prefetch]
        scale_evict = {
            "default": None,
            "normal": nvvm.L1EvictKind.NORMAL,
            "first": nvvm.L1EvictKind.FIRST,
            "last": nvvm.L1EvictKind.LAST,
            "noallocate": nvvm.L1EvictKind.NOALLOCATE,
        }[cfg.scale_l1_evict]
        scale_cache = {
            "default": None,
            "ca": nvvm.LoadCacheModifier.CA,
            "cg": nvvm.LoadCacheModifier.CG,
            "cs": nvvm.LoadCacheModifier.CS,
        }[cfg.scale_cache]
        scale_blocks_per_tile = cfg.tile_k // self.sf_vec_size
        scale_chunks_per_row = scale_blocks_per_tile // cfg.scale_load_vec
        a_scale_count = cfg.tile_m * scale_chunks_per_row
        total_scale_count = (
            cfg.tile_m + cfg.tile_n
        ) * scale_chunks_per_row
        for scale_task in cutlass.range(
            start_thread,
            total_scale_count,
            thread_count,
            unroll=1,
        ):
            is_a = scale_task < a_scale_count
            local_task = scale_task if is_a else scale_task - a_scale_count
            row = local_task // scale_chunks_per_row
            k_chunk = local_task % scale_chunks_per_row
            k_block = k_chunk * cfg.scale_load_vec
            global_row = (
                block_m * cfg.tile_m + row
                if is_a
                else block_n * cfg.tile_n + row
            )
            row_limit = self.problem.m if is_a else self.problem.n
            global_k_block = k_tile * scale_blocks_per_tile + k_block
            if cutlass.const_expr(cfg.scale_load_vec == 1):
                scale = Uint8(0).bitcast(self.sf_dtype)
                if (
                    global_row < row_limit
                    and global_k_block
                    < self.problem.k // self.sf_vec_size
                ):
                    if is_a:
                        scale = sx[global_row, global_k_block]
                    else:
                        scale = sw[global_row, global_k_block]
                if is_a:
                    s_sfa[row, k_block * self.sf_vec_size, stage] = scale
                else:
                    s_sfb[row, k_block * self.sf_vec_size, stage] = scale
            else:
                if is_a:
                    if global_row < row_limit:
                        src_row = sx[global_row, None]
                        loaded = nvvm.load_ext(
                            src_row.iterator + src_row.layout(global_k_block),
                            dtype=Uint8,
                            count=cfg.scale_load_vec,
                            prefetch=scale_prefetch,
                            evict=scale_evict,
                            cache_modifier=scale_cache,
                        ).bitcast(self.sf_dtype)
                        for vec in cutlass.range_constexpr(cfg.scale_load_vec):
                            s_sfa[
                                row,
                                (k_block + vec) * self.sf_vec_size,
                                stage,
                            ] = loaded[vec]
                    else:
                        for vec in cutlass.range_constexpr(cfg.scale_load_vec):
                            s_sfa[
                                row,
                                (k_block + vec) * self.sf_vec_size,
                                stage,
                            ] = Uint8(0).bitcast(self.sf_dtype)

                else:
                    if global_row < row_limit:
                        src_row = sw[global_row, None]
                        loaded = nvvm.load_ext(
                            src_row.iterator + src_row.layout(global_k_block),
                            dtype=Uint8,
                            count=cfg.scale_load_vec,
                            prefetch=scale_prefetch,
                            evict=scale_evict,
                            cache_modifier=scale_cache,
                        ).bitcast(self.sf_dtype)
                        for vec in cutlass.range_constexpr(cfg.scale_load_vec):
                            s_sfb[
                                row,
                                (k_block + vec) * self.sf_vec_size,
                                stage,
                            ] = loaded[vec]
                    else:
                        for vec in cutlass.range_constexpr(cfg.scale_load_vec):
                            s_sfb[
                                row,
                                (k_block + vec) * self.sf_vec_size,
                                stage,
                            ] = Uint8(0).bitcast(self.sf_dtype)

    @cute.jit
    def _store_consumer_scale_chunk(
        self,
        destination: cute.Tensor,
        row: Int32,
        k_block: Int32,
        stage: Int32,
        loaded: cute.Tensor,
    ) -> None:
        """Store one contiguous GMEM scale chunk into native SMEM layout."""

        cfg = self.config
        if cutlass.const_expr(cfg.scale_smem_store == "packed"):
            if cutlass.const_expr(cfg.scale_load_vec == 2):
                packed = Int32(loaded[0].bitcast(Uint8)) | (
                    Int32(loaded[1].bitcast(Uint8)) << 8
                )
                nvvm.store_ext(
                    Int16(packed),
                    destination.iterator
                    + destination.layout(
                        (row, k_block * self.sf_vec_size, stage)
                    ),
                )
            else:
                for group in cutlass.range_constexpr(
                    cfg.scale_load_vec // 4
                ):
                    first = group * 4
                    packed = (
                        Int32(loaded[first].bitcast(Uint8))
                        | (Int32(loaded[first + 1].bitcast(Uint8)) << 8)
                        | (Int32(loaded[first + 2].bitcast(Uint8)) << 16)
                        | (Int32(loaded[first + 3].bitcast(Uint8)) << 24)
                    )
                    nvvm.store_ext(
                        packed,
                        destination.iterator
                        + destination.layout(
                            (
                                row,
                                (k_block + first) * self.sf_vec_size,
                                stage,
                            )
                        ),
                    )
        else:
            for vec in cutlass.range_constexpr(cfg.scale_load_vec):
                destination[
                    row,
                    (k_block + vec) * self.sf_vec_size,
                    stage,
                ] = loaded[vec]

    @cute.jit
    def __call__(
        self,
        qx: cute.Tensor,
        qw: cute.Tensor,
        sx: cute.Tensor,
        sw: cute.Tensor,
        out: cute.Tensor,
        output_scale: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self._setup_static_layouts()
        cfg = self.config
        tma_x, tma_x_tensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            qx,
            cute.slice_(self.a_layout, (None, None, 0)),
            (cfg.tile_m, cfg.tile_k),
        )
        tma_w, tma_w_tensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            qw,
            cute.slice_(self.b_layout, (None, None, 0)),
            (cfg.tile_n, cfg.tile_k),
        )
        scale_a_tile_elems = (
            512
            if cfg.scale_layout == "mma64x128"
            else cfg.tile_m * cfg.tile_k // self.sf_vec_size
        )
        scale_b_tile_elems = cfg.tile_n * cfg.tile_k // self.sf_vec_size
        scale_a_flat_layout = cute.make_layout(
            (scale_a_tile_elems, cfg.stages),
            stride=(1, scale_a_tile_elems),
        )
        scale_b_flat_layout = cute.make_layout(
            (scale_b_tile_elems, cfg.stages),
            stride=(1, scale_b_tile_elems),
        )
        if cutlass.const_expr(cfg.epilogue == "tma"):
            out_view = cute.make_tensor(
                out.iterator,
                cute.make_layout(
                    (self.problem.m, self.problem.n, 1),
                    stride=(self.problem.n, 1, self.problem.m * self.problem.n),
                ),
            )
            tma_out, tma_out_tensor = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                out_view,
                cute.slice_(self.out_layout, (None, None, 0)),
                (cfg.tile_m, cfg.tile_n),
            )
        else:
            # Direct and split-K epilogues never instantiate the constexpr TMA
            # store branch. Reuse a typed placeholder instead of attempting to
            # build a BF16 tensor map over an FP32 workspace.
            tma_out, tma_out_tensor = tma_x, tma_x_tensor

        @cute.struct
        class SharedStorage:
            a: cute.struct.Align[
                cute.struct.MemRange[self.ab_dtype, cute.cosize(self.a_layout)],
                1024,
            ]
            b: cute.struct.Align[
                cute.struct.MemRange[self.ab_dtype, cute.cosize(self.b_layout)],
                1024,
            ]
            sfa: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype,
                    scale_a_tile_elems * cfg.stages
                    if cfg.scale_layout == "mma64x128"
                    else cute.cosize(self.sfa_layout),
                ],
                128,
            ]
            sfb: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfb_layout)],
                128,
            ]
            pipeline: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int64, cfg.stages * 2], 8
            ]
            out: cute.struct.Align[
                cute.struct.MemRange[
                    BFloat16,
                    cute.cosize(self.out_layout) if cfg.epilogue == "tma" else 0,
                ],
                1024,
            ]
            cluster_barrier: cute.struct.Align[
                cute.struct.MemRange[Int64, 2 if self.cluster_output else 0],
                8,
            ]
            outer_x: cute.struct.Align[
                cute.struct.MemRange[
                    Float32,
                    self.output_x_scale_cache_elems
                    * self.regional_scale_cache_stages,
                ],
                16,
            ]
            outer_w: cute.struct.Align[
                cute.struct.MemRange[
                    Float32,
                    self.output_w_scale_cache_elems
                    * self.regional_scale_cache_stages,
                ],
                16,
            ]
            outer_product: cute.struct.Align[
                cute.struct.MemRange[
                    Float32,
                    self.output_scale_product_cache_elems
                    * self.regional_scale_cache_stages,
                ],
                16,
            ]
            epilogue_accumulators: cute.struct.Align[
                cute.struct.MemRange[
                    Float32,
                    self.epilogue_accumulator_elements,
                ],
                1024,
            ]

        self.shared_storage = SharedStorage
        m_tiles = cute.ceil_div(self.problem.m, cfg.tile_m)
        n_tiles = cute.ceil_div(self.problem.n, cfg.tile_n)
        if cutlass.const_expr(cfg.scale_role == "tma"):
            sx_flat = cute.make_tensor(
                sx.iterator,
                cute.make_layout((cute.size(sx),), stride=(1,)),
            )
            sw_flat = cute.make_tensor(
                sw.iterator,
                cute.make_layout((cute.size(sw),), stride=(1,)),
            )
            tma_sx, tma_sx_tensor = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                sx_flat,
                cute.slice_(scale_a_flat_layout, (None, 0)),
                (scale_a_tile_elems,),
            )
            tma_sw, tma_sw_tensor = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                sw_flat,
                cute.slice_(scale_b_flat_layout, (None, 0)),
                (scale_b_tile_elems,),
            )
            launch = self.kernel(
                qx, qw, sx, sw, out, output_scale,
                tma_x, tma_x_tensor, tma_w, tma_w_tensor,
                tma_sx, tma_sx_tensor, tma_sw, tma_sw_tensor,
                tma_out, tma_out_tensor, self.tiled_mma,
                self.a_layout, self.b_layout,
                self.sfa_layout, self.sfb_layout,
                scale_a_flat_layout, scale_b_flat_layout, self.out_layout,
            )
            if cutlass.const_expr(self.cluster_output):
                launch.launch(
                    grid=(self.grid_ctas, 1, 1),
                    block=(cfg.num_threads, 1, 1),
                    cluster=(self.split_reduction, 1, 1),
                    stream=stream,
                    use_pdl=self.use_pdl,
                )
            else:
                launch.launch(
                    grid=(self.grid_ctas, 1, 1),
                    block=(cfg.num_threads, 1, 1),
                    stream=stream,
                    use_pdl=self.use_pdl,
                )
        else:
            launch = self.kernel(
                qx, qw, sx, sw, out, output_scale,
                tma_x, tma_x_tensor, tma_w, tma_w_tensor,
                tma_x, tma_x_tensor, tma_w, tma_w_tensor,
                tma_out, tma_out_tensor, self.tiled_mma,
                self.a_layout, self.b_layout,
                self.sfa_layout, self.sfb_layout,
                scale_a_flat_layout, scale_b_flat_layout, self.out_layout,
            )
            if cutlass.const_expr(self.cluster_output):
                launch.launch(
                    grid=(self.grid_ctas, 1, 1),
                    block=(cfg.num_threads, 1, 1),
                    cluster=(self.split_reduction, 1, 1),
                    stream=stream,
                    use_pdl=self.use_pdl,
                )
            else:
                launch.launch(
                    grid=(self.grid_ctas, 1, 1),
                    block=(cfg.num_threads, 1, 1),
                    stream=stream,
                    use_pdl=self.use_pdl,
                )

    @cute.kernel
    def kernel(
        self,
        qx: cute.Tensor,
        qw: cute.Tensor,
        sx: cute.Tensor,
        sw: cute.Tensor,
        out: cute.Tensor,
        output_scale: cute.Tensor,
        tma_x: cute.CopyAtom,
        tma_x_tensor: cute.Tensor,
        tma_w: cute.CopyAtom,
        tma_w_tensor: cute.Tensor,
        tma_sx: cute.CopyAtom,
        tma_sx_tensor: cute.Tensor,
        tma_sw: cute.CopyAtom,
        tma_sw_tensor: cute.Tensor,
        tma_out: cute.CopyAtom,
        tma_out_tensor: cute.Tensor,
        tiled_mma: cute.TiledMma,
        a_layout: cute.ComposedLayout,
        b_layout: cute.ComposedLayout,
        sfa_layout: cute.Layout,
        sfb_layout: cute.Layout,
        scale_a_flat_layout: cute.Layout,
        scale_b_flat_layout: cute.Layout,
        out_layout: cute.ComposedLayout,
    ):
        cfg = self.config
        if cutlass.const_expr(self.use_pdl):
            if nvvm.elect_sync():
                nvvm.griddepcontrol("wait")
        scale_prefetch = {
            "none": None,
            "64b": nvvm.L2PrefetchSize.SIZE_64B,
            "128b": nvvm.L2PrefetchSize.SIZE_128B,
            "256b": nvvm.L2PrefetchSize.SIZE_256B,
        }[cfg.scale_l2_prefetch]
        scale_evict = {
            "default": None,
            "normal": nvvm.L1EvictKind.NORMAL,
            "first": nvvm.L1EvictKind.FIRST,
            "last": nvvm.L1EvictKind.LAST,
            "noallocate": nvvm.L1EvictKind.NOALLOCATE,
        }[cfg.scale_l1_evict]
        scale_cache = {
            "default": None,
            "ca": nvvm.LoadCacheModifier.CA,
            "cg": nvvm.LoadCacheModifier.CG,
            "cs": nvvm.LoadCacheModifier.CS,
        }[cfg.scale_cache]
        global_output_scale = Float32(1.0)
        if cutlass.const_expr(self.apply_output_scale):
            global_output_scale = Float32(output_scale[0])
        sx_row_view = cute.make_tensor(
            sx.iterator,
            cute.make_layout(
                (self.problem.m, self.problem.k // self.sf_vec_size),
                stride=(self.problem.k // self.sf_vec_size, 1),
            ),
        )
        sw_row_view = cute.make_tensor(
            sw.iterator,
            cute.make_layout(
                (self.problem.n, self.problem.k // self.sf_vec_size),
                stride=(self.problem.k // self.sf_vec_size, 1),
            ),
        )
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        cta_idx, _, _ = cute.arch.block_idx()
        m_tiles = cute.ceil_div(self.problem.m, cfg.tile_m)
        n_tiles = cute.ceil_div(self.problem.n, cfg.tile_n)

        storage = cutlass.utils.SmemAllocator().allocate(self.shared_storage)
        s_a = storage.a.get_tensor(a_layout.outer, swizzle=a_layout.inner)
        s_b = storage.b.get_tensor(b_layout.outer, swizzle=b_layout.inner)
        s_sfa = storage.sfa.get_tensor(sfa_layout)
        s_sfb = storage.sfb.get_tensor(sfb_layout)
        s_sfa_flat = cute.make_tensor(s_sfa.iterator, scale_a_flat_layout)
        s_sfb_flat = cute.make_tensor(s_sfb.iterator, scale_b_flat_layout)
        # Typed placeholders for ordinary GEMMs; regional specializations
        # replace them with their non-empty FP32 cache tensors below.
        s_outer_x = s_sfa_flat
        s_outer_w = s_sfa_flat
        if cutlass.const_expr(self.cache_regional_output_scales):
            s_outer_x = storage.outer_x.get_tensor(
                cute.make_layout(
                    (
                        self.output_x_scale_cache_elems
                        * self.regional_scale_cache_stages,
                    ),
                    stride=(1,),
                )
            )
            s_outer_w = storage.outer_w.get_tensor(
                cute.make_layout(
                    (
                        self.output_w_scale_cache_elems
                        * self.regional_scale_cache_stages,
                    ),
                    stride=(1,),
                )
            )
        # Product caching is an optional specialization hook.  Alias the X
        # cache as a typed placeholder so ordinary MXFP8/NVFP4 kernels pay no
        # additional shared-memory allocation (even one byte can round the
        # launch above SM120's allocation quantum).
        s_outer_product = s_outer_x
        if cutlass.const_expr(self.cache_output_scale_products):
            s_outer_product = storage.outer_product.get_tensor(
                cute.make_layout(
                    (
                        self.output_scale_product_cache_elems
                        * self.regional_scale_cache_stages,
                    ),
                    stride=(1,),
                )
            )
        s_epilogue_accumulators = s_outer_x
        if cutlass.const_expr(self.warp_specialized_epilogue):
            s_epilogue_accumulators = storage.epilogue_accumulators.get_tensor(
                cute.make_layout(
                    (self.epilogue_accumulator_elements,), stride=(1,)
                )
            )
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
            # One launch-time rendezvous publishes every CTA's initialized
            # mbarrier. Repeated epilogue phases use the generation-counted
            # barriers below rather than barrier.cluster phase reuse.
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
        if cutlass.const_expr(cfg.epilogue == "tma"):
            s_out = storage.out.get_tensor(
                out_layout.outer, swizzle=out_layout.inner
            )

        g_x = cute.local_tile(
            tma_x_tensor, (cfg.tile_m, cfg.tile_k), (None, None)
        )
        g_w = cute.local_tile(
            tma_w_tensor, (cfg.tile_n, cfg.tile_k), (None, None)
        )
        tx_s, tx_g = cpasync.tma_partition(
            tma_x,
            0,
            cute.make_layout(1),
            cute.group_modes(s_a, 0, 2),
            cute.group_modes(g_x, 0, 2),
        )
        tw_s, tw_g = cpasync.tma_partition(
            tma_w,
            0,
            cute.make_layout(1),
            cute.group_modes(s_b, 0, 2),
            cute.group_modes(g_w, 0, 2),
        )
        scale_a_tile_elems = (
            512
            if cfg.scale_layout == "mma64x128"
            else cfg.tile_m * cfg.tile_k // self.sf_vec_size
        )
        scale_b_tile_elems = cfg.tile_n * cfg.tile_k // self.sf_vec_size
        tsx_s, tsx_g = tx_s, tx_g
        tsw_s, tsw_g = tw_s, tw_g
        if cutlass.const_expr(cfg.scale_role == "tma"):
            g_sx = cute.local_tile(
                tma_sx_tensor, (scale_a_tile_elems,), (None,)
            )
            g_sw = cute.local_tile(
                tma_sw_tensor, (scale_b_tile_elems,), (None,)
            )
            tsx_s, tsx_g = cpasync.tma_partition(
                tma_sx,
                0,
                cute.make_layout(1),
                cute.group_modes(s_sfa_flat, 0, 1),
                cute.group_modes(g_sx, 0, 1),
            )
            tsw_s, tsw_g = cpasync.tma_partition(
                tma_sw,
                0,
                cute.make_layout(1),
                cute.group_modes(s_sfb_flat, 0, 1),
                cute.group_modes(g_sw, 0, 1),
            )
        tma_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=cfg.stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, cfg.num_mma_warps
            ),
            # TMA mbarriers count completed bytes. MXFP8's element count is
            # already a byte count, but packed NVFP4 transfers two logical
            # E2M1 values per byte.
            tx_count=(
                (cfg.tile_m + cfg.tile_n)
                * cfg.tile_k
                * self.ab_dtype.width
                // 8
            )
            + (
                scale_a_tile_elems + scale_b_tile_elems
                if cfg.scale_role == "tma"
                else 0
            ),
            barrier_storage=storage.pipeline.data_ptr(),
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
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

        copy_a = cute.make_tiled_copy_A(
            _make_ldmatrix_atom(
                self.ab_dtype, False, cfg.a_ldmatrix_matrices
            ),
            tiled_mma,
        )
        copy_b = cute.make_tiled_copy_B(
            _make_ldmatrix_atom(
                self.ab_dtype, False, cfg.b_ldmatrix_matrices
            ),
            tiled_mma,
        )
        thr_copy_a = copy_a.get_slice(tidx)
        thr_copy_b = copy_b.get_slice(tidx)
        t_cs_a_copy = thr_copy_a.partition_S(s_a)
        t_cs_b_copy = thr_copy_b.partition_S(s_b)
        t_cr_a_copy = thr_copy_a.retile(t_cr_a)
        t_cr_b_copy = thr_copy_b.retile(t_cr_b)

        copy_sfa = cute.make_tiled_copy(
            _make_scale_s2r_atom(self.sf_dtype, cfg.sfa_s2r_bits),
            sm120_utils.get_layoutSFA_TV(tiled_mma),
            (
                cute.size(tiled_mma.permutation_mnk[0]),
                cute.size(tiled_mma.permutation_mnk[2]),
            ),
        )
        copy_sfb = cute.make_tiled_copy(
            _make_scale_s2r_atom(self.sf_dtype, cfg.sfb_s2r_bits),
            sm120_utils.get_layoutSFB_TV(tiled_mma),
            (
                cute.size(tiled_mma.permutation_mnk[1]),
                cute.size(tiled_mma.permutation_mnk[2]),
            ),
        )
        thr_copy_sfa = copy_sfa.get_slice(tidx)
        thr_copy_sfb = copy_sfb.get_slice(tidx)
        t_cs_sfa = thr_copy_sfa.partition_S(s_sfa)
        t_cs_sfb = thr_copy_sfb.partition_S(s_sfb)
        t_cr_sfa_copy = thr_copy_sfa.retile(t_cr_sfa)
        t_cr_sfb_copy = thr_copy_sfb.retile(t_cr_sfb)

        total_tiles = m_tiles * n_tiles
        total_work_tiles = total_tiles * self.split_reduction
        for work_slot in cutlass.range_constexpr(self.work_tiles_per_cta):
            if cutlass.const_expr(cfg.persistent_waves):
                raw_work_linear = cta_idx + work_slot * self.grid_ctas
            else:
                raw_work_linear = cta_idx * cfg.tiles_per_cta + work_slot
            active_tile = raw_work_linear < total_work_tiles
            # A final partial CTA follows the full pipeline on the last valid
            # tile so every unrolled slot advances identical barrier phases.
            # Only its epilogue is predicated, avoiding duplicate output stores.
            work_linear = cutlass.min(raw_work_linear, total_work_tiles - 1)
            split_id = Int32(0)
            if cutlass.const_expr(self.cluster_output):
                split_id = cute.arch.block_idx_in_cluster()
                work_linear = cta_idx // self.split_reduction
                active_tile = work_linear < total_tiles
            elif cutlass.const_expr(self.split_reduction > 1):
                split_id = work_linear // total_tiles
                work_linear = work_linear % total_tiles
            if cutlass.const_expr(cfg.tile_locality == "same_a"):
                block_m = work_linear // n_tiles
                block_n = work_linear % n_tiles
            elif cutlass.const_expr(cfg.tile_locality == "same_b"):
                block_n = work_linear // m_tiles
                block_m = work_linear % m_tiles
            elif cutlass.const_expr(cfg.tile_locality == "serpentine_a"):
                block_m = work_linear // n_tiles
                n_offset = work_linear % n_tiles
                block_n = (
                    n_offset
                    if block_m % 2 == 0
                    else n_tiles - 1 - n_offset
                )
            elif cutlass.const_expr(cfg.tile_locality == "serpentine_b"):
                block_n = work_linear // m_tiles
                m_offset = work_linear % m_tiles
                block_m = (
                    m_offset
                    if block_n % 2 == 0
                    else m_tiles - 1 - m_offset
                )
            elif cutlass.const_expr(cfg.raster == "n"):
                full_group = n_tiles * cfg.grid_swizzle
                group = work_linear // full_group
                offset = work_linear % full_group
                rows_in_group = cutlass.min(
                    cfg.grid_swizzle,
                    m_tiles - group * cfg.grid_swizzle,
                )
                block_n = offset // rows_in_group
                block_m = (
                    group * cfg.grid_swizzle + offset % rows_in_group
                )
            else:
                full_group = m_tiles * cfg.grid_swizzle
                group = work_linear // full_group
                offset = work_linear % full_group
                cols_in_group = cutlass.min(
                    cfg.grid_swizzle,
                    n_tiles - group * cfg.grid_swizzle,
                )
                block_m = offset // cols_in_group
                block_n = (
                    group * cfg.grid_swizzle + offset % cols_in_group
                )

            tile_output_scale = global_output_scale
            if cutlass.const_expr(self.tile_uniform_regional_output_scale):
                tile_output_scale = (
                    self._load_x_output_scale(
                        output_scale, block_m * cfg.tile_m
                    )
                    * self._load_weight_output_scale(
                        output_scale, block_n * cfg.tile_n
                    )
                )

            tile_s_outer_x = s_outer_x
            tile_s_outer_w = s_outer_w
            tile_s_outer_product = s_outer_product
            if cutlass.const_expr(self.warp_specialized_epilogue):
                cache_stage = Int32(work_slot % self.regional_scale_cache_stages)
                tile_s_outer_x = cute.make_tensor(
                    s_outer_x.iterator
                    + cache_stage * self.output_x_scale_cache_elems,
                    cute.make_layout(
                        (self.output_x_scale_cache_elems,), stride=(1,)
                    ),
                )
                tile_s_outer_w = cute.make_tensor(
                    s_outer_w.iterator
                    + cache_stage * self.output_w_scale_cache_elems,
                    cute.make_layout(
                        (self.output_w_scale_cache_elems,), stride=(1,)
                    ),
                )
                if cutlass.const_expr(self.cache_output_scale_products):
                    tile_s_outer_product = cute.make_tensor(
                        s_outer_product.iterator
                        + cache_stage * self.output_scale_product_cache_elems,
                        cute.make_layout(
                            (self.output_scale_product_cache_elems,), stride=(1,)
                        ),
                    )

            if cutlass.const_expr(
                self.cache_regional_output_scales
                and not self.tile_uniform_regional_output_scale
                and not self.direct_regional_output_scales
            ):
                cache_tidx = tidx
                cache_threads = cfg.num_threads
                if cutlass.const_expr(self.warp_specialized_epilogue):
                    cache_threads = cfg.num_mma_warps * 32
                    if warp_idx >= cfg.num_mma_warps:
                        cache_tidx = Int32(cache_threads)
                for local_row in cutlass.range(
                    cache_tidx,
                    self.output_x_scale_cache_elems,
                    cache_threads,
                    unroll=1,
                ):
                    global_row = cutlass.min(
                        block_m * cfg.tile_m
                        + self._x_output_scale_cache_offset(local_row),
                        self.problem.m - 1,
                    )
                    tile_s_outer_x[local_row] = self._load_x_output_scale(
                        output_scale, global_row
                    )
                for local_column in cutlass.range(
                    cache_tidx,
                    self.output_w_scale_cache_elems,
                    cache_threads,
                    unroll=1,
                ):
                    global_column = cutlass.min(
                        block_n * cfg.tile_n
                        + self._w_output_scale_cache_offset(local_column),
                        self.problem.n - 1,
                    )
                    tile_s_outer_w[local_column] = self._load_weight_output_scale(
                        output_scale, global_column
                    )
                if cutlass.const_expr(self.warp_specialized_epilogue):
                    if warp_idx < cfg.num_mma_warps:
                        self.regional_cache_barrier.arrive_and_wait()
                else:
                    cute.arch.sync_threads()
                if cutlass.const_expr(self.cache_output_scale_products):
                    for product_index in cutlass.range(
                        cache_tidx,
                        self.output_scale_product_cache_elems,
                        cache_threads,
                        unroll=1,
                    ):
                        x_index = (
                            product_index // self.output_w_scale_cache_elems
                        )
                        weight_index = (
                            product_index % self.output_w_scale_cache_elems
                        )
                        tile_s_outer_product[product_index] = (
                            Float32(tile_s_outer_x[x_index])
                            * Float32(tile_s_outer_w[weight_index])
                        )
                    if cutlass.const_expr(self.warp_specialized_epilogue):
                        if warp_idx < cfg.num_mma_warps:
                            self.regional_cache_barrier.arrive_and_wait()
                    else:
                        cute.arch.sync_threads()

            # Reconstruct the exact circular-pipeline phase for this unrolled
            # output tile. Carrying the mutable PipelineState Python object
            # across an outer constexpr loop currently produces non-dominating
            # SSA in CuTe DSL, while the phase itself is fully determined by
            # the number of K stages completed by preceding work slots.
            pipeline_count = work_slot * self.num_k_tiles
            pipeline_index = pipeline_count % cfg.stages
            pipeline_phase = (pipeline_count // cfg.stages) & 1
            producer_state = pipeline.PipelineState(
                cfg.stages,
                Int32(pipeline_count),
                Int32(pipeline_index),
                Int32(1 ^ pipeline_phase),
            )
            consumer_state = pipeline.PipelineState(
                cfg.stages,
                Int32(pipeline_count),
                Int32(pipeline_index),
                Int32(pipeline_phase),
            )

            out_matrix = cute.make_tensor(
                out.iterator
                + (
                    split_id * self.problem.m * self.problem.n
                    if self.split_reduction > 1
                    and not self.atomic_output
                    and not self.cluster_output
                    else 0
                ),
                cute.make_layout(
                    (self.problem.m, self.problem.n),
                    stride=(self.problem.n, 1),
                ),
            )
            g_out = cute.local_tile(
                out_matrix, (cfg.tile_m, cfg.tile_n), (block_m, block_n)
            )
            c_out = cute.local_tile(
                cute.make_identity_tensor((self.problem.m, self.problem.n)),
                (cfg.tile_m, cfg.tile_n),
                (block_m, block_n),
            )
            t_cg_out = thr_mma.partition_C(g_out)
            t_cc_out = thr_mma.partition_C(c_out)
            accumulators = cute.make_rmem_tensor(t_cg_out.shape, Float32)
            accumulators.fill(0.0)

            if cutlass.const_expr(cfg.epilogue == "tma"):
                epilogue_stage = Int32(
                    work_slot % cfg.epilogue_stages
                )
                s_out_stage = s_out[(None, None, epilogue_stage)]
                r2s_atom = cute.make_copy_atom(
                    warp.StMatrix8x8x16bOp(False, cfg.store_vec), BFloat16
                )
                c_atom = cute.make_copy_atom(
                    warp.StMatrix8x8x16bOp(False, cfg.store_vec), BFloat16
                )
                c_copy = cute.make_tiled_copy_C_atom(c_atom, tiled_mma)
                copy_r2s = cute.make_tiled_copy_S(r2s_atom, c_copy)
                thr_r2s = copy_r2s.get_slice(tidx)
                t_rs_s = thr_r2s.partition_D(s_out_stage)
                t_rs_acc = copy_r2s.retile(accumulators)
                r_out_shape = cute.shape(thr_r2s.partition_S(s_out_stage))
                r_out = cute.make_rmem_tensor(
                    cute.make_layout(r_out_shape[:3]).shape, BFloat16
                )
                g_out_tma = cute.local_tile(
                    tma_out_tensor,
                    (cfg.tile_m, cfg.tile_n),
                    (None, None, None),
                )[(None, None, block_m, block_n, 0)]
                g_out_tma = cute.zipped_divide(
                    g_out_tma, (cfg.tile_m, cfg.tile_n)
                )
                out_s, out_g = cpasync.tma_partition(
                    tma_out,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(s_out_stage, 0, 2),
                    g_out_tma,
                )
                out_pipeline = pipeline.PipelineTmaStore.create(
                    num_stages=cfg.epilogue_stages,
                    producer_group=pipeline.CooperativeGroup(
                        pipeline.Agent.Thread, cfg.num_mma_warps * 32
                    ),
                )

            if warp_idx == cfg.num_mma_warps:
                cute.arch.setmaxregister_decrease(cfg.producer_registers)
            elif cutlass.const_expr(self.warp_specialized_epilogue):
                if warp_idx > cfg.num_mma_warps:
                    cute.arch.setmaxregister_decrease(
                        cfg.regional_epilogue_registers
                    )
                else:
                    cute.arch.setmaxregister_increase(cfg.consumer_registers)
            else:
                cute.arch.setmaxregister_increase(cfg.consumer_registers)

            num_k_tiles = cute.ceil_div(self.problem.k, cfg.tile_k)
            scale_blocks_per_tile = cfg.tile_k // self.sf_vec_size
            # One MXFP8 instruction consumes one 32-value scale block, while
            # one NVFP4 instruction consumes four 16-value scale blocks.  The
            # MMA fragment's K mode is the authoritative instruction count;
            # using the number of scale vectors only happened to be correct
            # for MXFP8 and walks beyond NVFP4 fragments.
            mma_k_blocks_per_tile = cute.size(t_cr_a, mode=[2])
            scale_chunks_per_row = scale_blocks_per_tile // cfg.scale_load_vec
            a_scale_count = cfg.tile_m * scale_chunks_per_row
            total_scale_count = (
                cfg.tile_m + cfg.tile_n
            ) * scale_chunks_per_row
            first_k_tile = Int32(0)
            final_k_tile = Int32(num_k_tiles)
            if cutlass.const_expr(self.split_reduction > 1):
                reduction_tiles = self.reduction_tile // cfg.tile_k
                first_k_tile = split_id * reduction_tiles
                final_k_tile = cutlass.min(
                    first_k_tile + reduction_tiles,
                    num_k_tiles,
                )
            for k_tile in cutlass.range(
                first_k_tile,
                final_k_tile,
                unroll=1,
            ):
                if warp_idx == cfg.num_mma_warps:
                    producer_stage = producer_state.index
                    tma_pipeline.producer_acquire(producer_state)
                    cute.copy(
                        tma_x,
                        tx_g[(None, block_m, k_tile)],
                        tx_s[(None, producer_state.index)],
                        tma_bar_ptr=tma_pipeline.producer_get_barrier(producer_state),
                    )
                    cute.copy(
                        tma_w,
                        tw_g[(None, block_n, k_tile)],
                        tw_s[(None, producer_state.index)],
                        tma_bar_ptr=tma_pipeline.producer_get_barrier(producer_state),
                    )
                    if cutlass.const_expr(cfg.scale_role == "tma"):
                        sx_tile = block_m * num_k_tiles + k_tile
                        sw_tile = block_n * num_k_tiles + k_tile
                        cute.copy(
                            tma_sx,
                            tsx_g[(None, sx_tile)],
                            tsx_s[(None, producer_stage)],
                            tma_bar_ptr=tma_pipeline.producer_get_barrier(
                                producer_state
                            ),
                        )
                        cute.copy(
                            tma_sw,
                            tsw_g[(None, sw_tile)],
                            tsw_s[(None, producer_stage)],
                            tma_bar_ptr=tma_pipeline.producer_get_barrier(
                                producer_state
                            ),
                        )
                    tma_pipeline.producer_commit(producer_state)
                    producer_state.advance()
                    if cutlass.const_expr(cfg.scale_role == "producer"):
                        self._stage_scales(
                            sx_row_view,
                            sw_row_view,
                            s_sfa,
                            s_sfb,
                            producer_stage,
                            block_m,
                            block_n,
                            k_tile,
                            tidx - cfg.num_mma_warps * 32,
                            32,
                        )
                        self.scale_barrier.arrive_and_wait()

                if warp_idx < cfg.num_mma_warps:
                    stage = consumer_state.index
                    if cutlass.const_expr(
                        cfg.scale_role == "tma"
                        or cfg.scale_schedule == "after_wait"
                    ):
                        ready = tma_pipeline.consumer_try_wait(consumer_state)
                        tma_pipeline.consumer_wait(consumer_state, ready)
                    scale_start = tidx
                    if cutlass.const_expr(cfg.scale_role in ("producer", "tma")):
                        scale_start = Int32(total_scale_count)
                    for scale_task in cutlass.range(
                        scale_start,
                        total_scale_count,
                        cfg.num_mma_warps * 32,
                        unroll=1,
                    ):
                        is_a = scale_task < a_scale_count
                        local_task = scale_task if is_a else scale_task - a_scale_count
                        row = local_task // scale_chunks_per_row
                        k_chunk = local_task % scale_chunks_per_row
                        k_block = k_chunk * cfg.scale_load_vec
                        global_row = (
                            block_m * cfg.tile_m + row
                            if is_a
                            else block_n * cfg.tile_n + row
                        )
                        row_limit = self.problem.m if is_a else self.problem.n
                        global_k_block = (
                            k_tile * scale_blocks_per_tile + k_block
                        )
                        if cutlass.const_expr(cfg.scale_load_vec == 1):
                            scale = Uint8(0).bitcast(self.sf_dtype)
                            if (
                                global_row < row_limit
                                and global_k_block
                                < self.problem.k // self.sf_vec_size
                            ):
                                if is_a:
                                    scale = sx_row_view[global_row, global_k_block]
                                else:
                                    scale = sw_row_view[global_row, global_k_block]
                            if is_a:
                                s_sfa[row, k_block * self.sf_vec_size, stage] = scale
                            else:
                                s_sfb[row, k_block * self.sf_vec_size, stage] = scale
                        else:
                            if is_a:
                                if global_row < row_limit:
                                    src_row = sx_row_view[global_row, None]
                                    loaded = nvvm.load_ext(
                                        src_row.iterator
                                        + src_row.layout(global_k_block),
                                        dtype=Uint8,
                                        count=cfg.scale_load_vec,
                                        prefetch=scale_prefetch,
                                        evict=scale_evict,
                                        cache_modifier=scale_cache,
                                    ).bitcast(self.sf_dtype)
                                    self._store_consumer_scale_chunk(
                                        s_sfa,
                                        row,
                                        k_block,
                                        stage,
                                        loaded,
                                    )
                                else:
                                    for vec in cutlass.range_constexpr(
                                        cfg.scale_load_vec
                                    ):
                                        s_sfa[
                                            row,
                                            (k_block + vec) * self.sf_vec_size,
                                            stage,
                                        ] = Uint8(0).bitcast(self.sf_dtype)
                            else:
                                if global_row < row_limit:
                                    src_row = sw_row_view[global_row, None]
                                    loaded = nvvm.load_ext(
                                        src_row.iterator
                                        + src_row.layout(global_k_block),
                                        dtype=Uint8,
                                        count=cfg.scale_load_vec,
                                        prefetch=scale_prefetch,
                                        evict=scale_evict,
                                        cache_modifier=scale_cache,
                                    ).bitcast(self.sf_dtype)
                                    self._store_consumer_scale_chunk(
                                        s_sfb,
                                        row,
                                        k_block,
                                        stage,
                                        loaded,
                                    )
                                else:
                                    for vec in cutlass.range_constexpr(
                                        cfg.scale_load_vec
                                    ):
                                        s_sfb[
                                            row,
                                            (k_block + vec) * self.sf_vec_size,
                                            stage,
                                        ] = Uint8(0).bitcast(self.sf_dtype)
                    if cutlass.const_expr(cfg.scale_role != "tma"):
                        self.scale_barrier.arrive_and_wait()
                    if cutlass.const_expr(
                        cfg.scale_role != "tma"
                        and cfg.scale_schedule == "before_wait"
                    ):
                        ready = tma_pipeline.consumer_try_wait(consumer_state)
                        tma_pipeline.consumer_wait(consumer_state, ready)
                    if cutlass.const_expr(cfg.mma_schedule == "preload"):
                        for k_block in cutlass.range_constexpr(
                            mma_k_blocks_per_tile
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
                                cute.filter_zeros(t_cs_sfa)[
                                    None, None, k_block, stage
                                ],
                                cute.filter_zeros(t_cr_sfa_copy)[
                                    None, None, k_block
                                ],
                            )
                            cute.copy(
                                copy_sfb,
                                cute.filter_zeros(t_cs_sfb)[
                                    None, None, k_block, stage
                                ],
                                cute.filter_zeros(t_cr_sfb_copy)[
                                    None, None, k_block
                                ],
                            )
                        for k_block in cutlass.range_constexpr(
                            mma_k_blocks_per_tile
                        ):
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
                    else:
                        for k_block in cutlass.range_constexpr(
                            mma_k_blocks_per_tile
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
                                cute.filter_zeros(t_cs_sfa)[
                                    None, None, k_block, stage
                                ],
                                cute.filter_zeros(t_cr_sfa_copy)[
                                    None, None, k_block
                                ],
                            )
                            cute.copy(
                                copy_sfb,
                                cute.filter_zeros(t_cs_sfb)[
                                    None, None, k_block, stage
                                ],
                                cute.filter_zeros(t_cr_sfb_copy)[
                                    None, None, k_block
                                ],
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
                    if cutlass.const_expr(
                        cfg.scale_role == "consumers"
                        and cfg.scale_recycle == "barrier"
                    ):
                        # Consumer warps write the next K tile's scales without
                        # passing through the TMA pipeline's empty barrier.  Do
                        # not let a fast warp recycle this scale stage while a
                        # slower peer is still loading its final fragments.
                        self.scale_barrier.arrive_and_wait()
                    tma_pipeline.consumer_release(consumer_state)
                    consumer_state.advance()

            if cutlass.const_expr(self.split_reduction > 1):
                if warp_idx == cfg.num_mma_warps:
                    tma_pipeline.producer_tail(producer_state)

            if cutlass.const_expr(
                cfg.epilogue == "tma" and self.cache_regional_output_scales
            ):
                if warp_idx < cfg.num_mma_warps:
                    for elem in cutlass.range(
                        cute.size(accumulators), unroll_full=True
                    ):
                        coord = t_cc_out[elem]
                        if (
                            coord[0] < self.problem.m
                            and coord[1] < self.problem.n
                        ):
                            accumulators[elem] = self._scale_tma_accumulator(
                                accumulators[elem],
                                tile_s_outer_x,
                                tile_s_outer_w,
                                tile_s_outer_product,
                                tile_output_scale,
                                output_scale,
                                coord[0],
                                coord[1],
                                coord[0] - block_m * cfg.tile_m,
                                coord[1] - block_n * cfg.tile_n,
                            )

            if cutlass.const_expr(self.warp_specialized_epilogue):
                first_epilogue_thread = (cfg.num_mma_warps + 1) * 32
                epilogue_threads = self.num_epilogue_warps * 32
                if warp_idx < cfg.num_mma_warps:
                    # The previous tile's epilogue owns the single FP32
                    # handoff slot while this tile is accumulating.  Delay
                    # only the publication, preserving MMA/epilogue overlap.
                    if cutlass.const_expr(work_slot > 0):
                        self.epilogue_free_barrier.arrive_and_wait()
                    for elem in cutlass.range(
                        cute.size(accumulators), unroll_full=True
                    ):
                        coord = t_cc_out[elem]
                        local_row = coord[0] - block_m * cfg.tile_m
                        local_column = coord[1] - block_n * cfg.tile_n
                        s_epilogue_accumulators[
                            local_row * cfg.tile_n + local_column
                        ] = accumulators[elem]
                    self.epilogue_ready_barrier.arrive_and_wait()
                elif warp_idx > cfg.num_mma_warps:
                    if cutlass.const_expr(work_slot > 0):
                        self.epilogue_free_barrier.arrive_and_wait()
                    self.epilogue_ready_barrier.arrive_and_wait()
                    epilogue_tidx = tidx - first_epilogue_thread
                    for output_index in cutlass.range(
                        epilogue_tidx,
                        cfg.tile_m * cfg.tile_n,
                        epilogue_threads,
                        unroll=1,
                    ):
                        local_row = output_index // cfg.tile_n
                        local_column = output_index % cfg.tile_n
                        global_row = block_m * cfg.tile_m + local_row
                        global_column = block_n * cfg.tile_n + local_column
                        if (
                            active_tile
                            and global_row < self.problem.m
                            and global_column < self.problem.n
                        ):
                            element_scale = self._epilogue_output_scale(
                                output_scale,
                                global_row,
                                global_column,
                                global_output_scale,
                                tile_s_outer_x,
                                tile_s_outer_w,
                                tile_s_outer_product,
                                block_m * cfg.tile_m,
                                block_n * cfg.tile_n,
                            )
                            out_matrix[global_row, global_column] = BFloat16(
                                element_scale
                                * s_epilogue_accumulators[output_index]
                            )
            elif cutlass.const_expr(cfg.epilogue == "direct"):
                if cutlass.const_expr(self.cluster_output):
                    # Reuse the now-dead A pipeline storage as a contiguous
                    # FP32 DSMEM reduction window.  The full accumulator tile
                    # need not fit: each CTA contributes register fragments in
                    # chunks, and cluster rank 0 converts each completed chunk
                    # to BF16 exactly once.
                    scratch_elements = cute.cosize(a_layout) // 4
                    mma_threads = cfg.num_mma_warps * 32
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
                    chunk_count = cute.ceil_div(accum_elements, chunk_elements)
                    # The producer warp alone executes producer_tail above.
                    # Do not overlay the A stages until every warp has observed
                    # that tail; otherwise a late TMA completion can overwrite
                    # rank 0's freshly zeroed FP32 reduction window.
                    cute.arch.sync_threads()
                    for chunk in cutlass.range_constexpr(chunk_count):
                        first_elem = chunk * chunk_elements
                        final_elem = cutlass.min(
                            first_elem + chunk_elements, accum_elements
                        )
                        if cluster_rank == 0 and warp_idx < cfg.num_mma_warps:
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
                        if warp_idx < cfg.num_mma_warps:
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
                        if cluster_rank == 0 and warp_idx < cfg.num_mma_warps:
                            for elem in cutlass.range(
                                first_elem, final_elem, unroll_full=True
                            ):
                                coord = t_cc_out[elem]
                                if (
                                    active_tile
                                    and coord[0] < self.problem.m
                                    and coord[1] < self.problem.n
                                ):
                                    element_scale = self._epilogue_output_scale(
                                        output_scale,
                                        coord[0],
                                        coord[1],
                                        global_output_scale,
                                        tile_s_outer_x,
                                        tile_s_outer_w,
                                        tile_s_outer_product,
                                        block_m * cfg.tile_m,
                                        block_n * cfg.tile_n,
                                    )
                                    t_cg_out[elem] = BFloat16(
                                        element_scale * scratch[
                                            tidx * chunk_elements
                                            + elem
                                            - first_elem
                                        ]
                                    )
                        # Rank 0 must finish reading the window before any CTA
                        # starts the next chunk's zero/contribution phase.
                        self._cluster_broadcast(
                            s_cluster_barrier,
                            0,
                            Int32((chunk * 2 + 1) & 1),
                            cluster_rank,
                        )
                elif active_tile:
                    if warp_idx < cfg.num_mma_warps:
                        for elem in cutlass.range(
                            cute.size(accumulators), unroll_full=True
                        ):
                            coord = t_cc_out[elem]
                            if (
                                coord[0] < self.problem.m
                                and coord[1] < self.problem.n
                            ):
                                if cutlass.const_expr(self.atomic_output):
                                    cute.arch.atomic_add(
                                        t_cg_out.iterator + t_cg_out.layout(elem),
                                        accumulators[elem],
                                        sem="relaxed",
                                        scope="gpu",
                                    )
                                elif cutlass.const_expr(self.split_reduction > 1):
                                    t_cg_out[elem] = accumulators[elem]
                                else:
                                    element_scale = self._epilogue_output_scale(
                                        output_scale,
                                        coord[0],
                                        coord[1],
                                        global_output_scale,
                                        tile_s_outer_x,
                                        tile_s_outer_w,
                                        tile_s_outer_product,
                                        block_m * cfg.tile_m,
                                        block_n * cfg.tile_n,
                                    )
                                    t_cg_out[elem] = BFloat16(
                                        element_scale * accumulators[elem]
                                    )
            else:
                if active_tile:
                    if warp_idx < cfg.num_mma_warps:
                        for elem in cutlass.range(
                            cute.size(r_out), unroll_full=True
                        ):
                            r_out[elem] = BFloat16(
                                global_output_scale * t_rs_acc[elem]
                            )
                        # Warp 0 cannot enter this barrier until its preceding
                        # TMA wait has completed, so it also guards reuse of the
                        # selected shared-memory stage by every MMA warp.
                        self.epilogue_barrier.arrive_and_wait()
                        cute.copy(
                            copy_r2s,
                            r_out,
                            t_rs_s,
                        )
                        cute.arch.fence_proxy("async.shared", space="cta")
                        self.epilogue_barrier.arrive_and_wait()
                        if warp_idx == 0:
                            cute.copy(
                                tma_out,
                                out_s,
                                out_g[(None, 0)],
                            )
                            out_pipeline.producer_commit()
                            out_pipeline.producer_acquire()

            if cutlass.const_expr(
                self.cache_regional_output_scales
                and not self.tile_uniform_regional_output_scale
                and not self.direct_regional_output_scales
                and not self.warp_specialized_epilogue
            ):
                # The next persistent tile reuses the scale cache. Ensure all
                # MMA warps have consumed this tile's values first.
                cute.arch.sync_threads()

        if cutlass.const_expr(self.split_reduction == 1):
            # Balanced persistence may raise the launch grid above the
            # tiles-per-CTA cap, reducing the actual constexpr work slots.
            # Drain the phases advanced by this CTA, not the requested cap.
            final_count = self.work_tiles_per_cta * self.num_k_tiles
            final_index = final_count % cfg.stages
            final_phase = (final_count // cfg.stages) & 1
            final_producer_state = pipeline.PipelineState(
                cfg.stages,
                Int32(final_count),
                Int32(final_index),
                Int32(1 ^ final_phase),
            )
            if warp_idx == cfg.num_mma_warps:
                # Drain once after the complete persistent sequence.  Draining
                # every output tile advances the producer state by ``stages``;
                # reconstructing the next tile from only its K-count then uses
                # stale mbarrier phases and intermittently consumes old data.
                tma_pipeline.producer_tail(final_producer_state)

        if cutlass.const_expr(cfg.epilogue == "tma"):
            if warp_idx == 0:
                cute.arch.cp_async_bulk_wait_group(0, read=True)


class _UnitOutputScaleLauncher:
    """Preserve the public five-tensor MXFP8 launcher signature."""

    def __init__(self, compiled):
        self.compiled = compiled

    def __call__(self, qx, qw, sx, sw, out):
        # The format-specialized branch removes this placeholder from MXFP8
        # device IR. Reusing ``out`` avoids allocating a process-global scalar.
        return self.compiled(qx, qw, sx, sw, out, out)


@lru_cache(maxsize=None)
def compile_mxfp8_gemm(
    problem: MXFP8Problem,
    config: MXFP8GemmConfig = MXFP8GemmConfig(),
    *,
    split_reduction: int = 1,
    reduction_tile: int = 0,
    atomic_output: bool = False,
    cluster_output: bool = False,
):
    kernel = MXFP8GemmKernel(
        problem,
        config,
        split_reduction=split_reduction,
        reduction_tile=reduction_tile,
        atomic_output=atomic_output,
        cluster_output=cluster_output,
    )
    qx = cute.runtime.make_fake_tensor(
        Float8E4M3FN,
        (problem.m, problem.k),
        stride=(problem.k, 1),
        assumed_align=16,
    )
    qw = cute.runtime.make_fake_tensor(
        Float8E4M3FN,
        (problem.n, problem.k),
        stride=(problem.k, 1),
        assumed_align=16,
    )
    if config.scale_layout == "row_major":
        sx = cute.runtime.make_fake_tensor(
            Float8E8M0FNU,
            (problem.m, problem.k // SF_VEC_SIZE),
            stride=(problem.k // SF_VEC_SIZE, 1),
            assumed_align=16,
        )
        sw = cute.runtime.make_fake_tensor(
            Float8E8M0FNU,
            (problem.n, problem.k // SF_VEC_SIZE),
            stride=(problem.k // SF_VEC_SIZE, 1),
            assumed_align=16,
        )
    elif config.scale_layout == "mma128":
        sx = cute.runtime.make_fake_tensor(
            Float8E8M0FNU,
            (problem.m // 128, problem.k // 128, 512),
            stride=(problem.k // 128 * 512, 512, 1),
            assumed_align=16,
        )
        sw = cute.runtime.make_fake_tensor(
            Float8E8M0FNU,
            (problem.n // 128, problem.k // 128, 512),
            stride=(problem.k // 128 * 512, 512, 1),
            assumed_align=16,
        )
    else:
        sx = cute.runtime.make_fake_tensor(
            Float8E8M0FNU,
            (problem.m // 64, problem.k // 128, 512),
            stride=(problem.k // 128 * 512, 512, 1),
            assumed_align=16,
        )
        sw = cute.runtime.make_fake_tensor(
            Float8E8M0FNU,
            (problem.n // 128, problem.k // 128, 512),
            stride=(problem.k // 128 * 512, 512, 1),
            assumed_align=16,
        )
    if split_reduction > 1 and not cluster_output:
        out_elements = (
            problem.m * problem.n
            if atomic_output
            else split_reduction * problem.m * problem.n
        )
        out = cute.runtime.make_fake_tensor(
            Float32,
            (out_elements,),
            stride=(1,),
            assumed_align=16,
        )
    else:
        out = cute.runtime.make_fake_tensor(
            BFloat16,
            (problem.m, problem.n),
            stride=(problem.n, 1),
            assumed_align=16,
        )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled = cute.compile(
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
    return _UnitOutputScaleLauncher(compiled)


__all__ = ["MXFP8GemmConfig", "MXFP8GemmKernel", "compile_mxfp8_gemm"]
