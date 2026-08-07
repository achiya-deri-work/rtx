"""Pure prequantized MXFP8 GEMM for RTX Blackwell SM120."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os

os.environ.setdefault("CUTE_DSL_ARCH", "sm_120a")
os.environ.setdefault("QUACK_ARCH", "sm_120a")

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm120_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass import (
    BFloat16,
    Float8E4M3FN,
    Float8E8M0FNU,
    Float32,
    Int32,
    Uint8,
)
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.experimental.primitives import nvvm_wrapper as nvvm

from .mxfp8 import MXFP8Problem, SM120_SMEM_CAPACITY_BYTES
from .mxfp8_fwd import (
    SF_VEC_SIZE,
    _make_ldmatrix_atom,
    _make_scale_s2r_atom,
    _make_sm120_sfa_layout_64,
)


@dataclass(frozen=True, slots=True)
class MXFP8GemmConfig:
    tile_m: int = 128
    tile_n: int = 128
    tile_k: int = 128
    atom_layout_m: int = 8
    atom_layout_n: int = 2
    stages: int = 2
    a_swizzle: str = "64b"
    b_swizzle: str = "64b"
    a_ldmatrix_matrices: int = 4
    b_ldmatrix_matrices: int = 4
    sfa_s2r_bits: int = 8
    sfb_s2r_bits: int = 8
    scale_schedule: str = "before_wait"
    scale_load_vec: int = 4
    scale_l2_prefetch: str = "none"
    scale_l1_evict: str = "default"
    scale_cache: str = "default"
    scale_role: str = "consumers"
    scale_layout: str = "row_major"
    epilogue: str = "tma"
    store_vec: int = 4
    maxrregcount: int = 255
    producer_registers: int = 48
    consumer_registers: int = 192
    raster: str = "n"
    grid_swizzle: int = 2

    @property
    def num_mma_warps(self) -> int:
        return self.atom_layout_m * self.atom_layout_n

    @property
    def num_threads(self) -> int:
        return (self.num_mma_warps + 1) * 32

    def rejection(self, problem: MXFP8Problem) -> str | None:
        try:
            problem.validate()
        except ValueError as exc:
            return str(exc)
        if (self.tile_m != 64 and self.tile_m % 128) or self.tile_n % 128:
            return "SM120 block-scale tiles require M=64 or M/N divisible by 128"
        if self.tile_k % 128:
            return "tile K must be divisible by 128"
        if self.tile_n != 64 * self.atom_layout_n:
            return "tile_n must equal 64 * atom_layout_n"
        if self.tile_m == 64 and self.atom_layout_m != 2:
            return "64-row SFA fragments require atom_layout_m=2"
        if self.tile_m == 256 and self.atom_layout_m != 8:
            return "256-row SFA fragments require atom_layout_m=8"
        if self.num_threads > 1024:
            return "CUDA limits one CTA to 1024 threads"
        if self.stages not in (1, 2, 3, 4):
            return "stages must be one of 1, 2, 3, 4"
        if self.a_swizzle not in ("none", "32b", "64b", "128b"):
            return "invalid A swizzle"
        if self.b_swizzle not in ("none", "32b", "64b", "128b"):
            return "invalid B swizzle"
        if self.a_ldmatrix_matrices not in (1, 2, 4):
            return "A ldmatrix width must be x1, x2, or x4"
        if self.b_ldmatrix_matrices not in (1, 2, 4):
            return "B ldmatrix width must be x1, x2, or x4"
        if self.sfa_s2r_bits not in (0, 8) or self.sfb_s2r_bits not in (0, 8):
            return "scale S2R widths must be auto or 8 bits"
        if self.scale_schedule not in ("after_wait", "before_wait"):
            return "scale_schedule must be after_wait or before_wait"
        if self.scale_load_vec not in (1, 2, 4, 8):
            return "scale_load_vec must be x1, x2, x4, or x8"
        if (self.tile_k // SF_VEC_SIZE) % self.scale_load_vec:
            return "scale_load_vec must divide the K tile scale count"
        if self.scale_l2_prefetch not in ("none", "64b", "128b", "256b"):
            return "invalid scale L2 prefetch size"
        if self.scale_l1_evict not in (
            "default",
            "normal",
            "first",
            "last",
            "noallocate",
        ):
            return "invalid scale L1 eviction priority"
        if self.scale_cache not in ("default", "ca", "cg", "cs"):
            return "invalid scale cache modifier"
        if self.scale_l1_evict != "default" and self.scale_cache != "default":
            return "scale eviction and cache modifiers are mutually exclusive"
        if self.scale_role not in ("consumers", "producer", "tma"):
            return "scale_role must be consumers, producer, or tma"
        if self.scale_role == "tma" and (
            self.scale_schedule != "before_wait"
            or self.scale_load_vec != 4
            or self.scale_l2_prefetch != "none"
            or self.scale_l1_evict != "default"
            or self.scale_cache != "default"
        ):
            return "scalar scale-staging controls are inactive for TMA scales"
        if self.scale_load_vec == 1 and (
            self.scale_l2_prefetch != "none"
            or self.scale_l1_evict != "default"
            or self.scale_cache != "default"
        ):
            return "vector-load cache controls are inactive for scalar scale loads"
        if self.scale_layout not in ("row_major", "mma128", "mma64x128"):
            return "scale_layout must be row_major, mma128, or mma64x128"
        if self.scale_layout == "mma128" and (
            self.scale_role != "tma"
            or self.tile_m != 128
            or self.tile_n != 128
            or self.tile_k != 128
            or problem.m % 128
            or problem.n % 128
            or problem.k % 128
        ):
            return "mma128 scales require TMA and full 128-row operand tiles"
        if self.scale_role == "tma" and self.scale_layout != "mma128":
            if not (
                self.scale_layout == "mma64x128"
                and self.tile_m == 64
                and self.tile_n == 128
                and self.tile_k == 128
                and self.stages == 1
                and problem.m % 64 == 0
                and problem.n % 128 == 0
                and problem.k % 128 == 0
            ):
                return "TMA scale transport requires a compatible native layout"
        if self.epilogue not in ("direct", "tma"):
            return "epilogue must be direct or tma"
        if self.store_vec not in (1, 2, 4):
            return "store_vec must be x1, x2, or x4"
        if self.epilogue == "direct" and self.store_vec != 1:
            return "store_vec only applies to the TMA epilogue"
        if self.epilogue == "tma" and (
            problem.m % self.tile_m or problem.n % self.tile_n
        ):
            return "TMA epilogue requires full M/N tiles"
        q_bytes = self.stages * (self.tile_m + self.tile_n) * self.tile_k
        scale_bytes = self.stages * (
            ((self.tile_m + 127) // 128) * 128
            + ((self.tile_n + 127) // 128) * 128
        ) * (self.tile_k // SF_VEC_SIZE)
        out_bytes = (
            self.tile_m * self.tile_n * BFloat16.width // 8
            if self.epilogue == "tma"
            else 0
        )
        if q_bytes + scale_bytes + out_bytes > SM120_SMEM_CAPACITY_BYTES:
            return "prequantized GEMM exceeds SM120 shared-memory capacity"
        if self.raster not in ("m", "n") or self.grid_swizzle not in (1, 2, 4, 8):
            return "invalid raster/grid swizzle"
        return None


class MXFP8GemmKernel:
    def __init__(self, problem: MXFP8Problem, config: MXFP8GemmConfig):
        rejection = config.rejection(problem)
        if rejection is not None:
            raise ValueError(f"illegal prequantized MXFP8 GEMM: {rejection}")
        self.problem = problem
        self.config = config
        self.tile_shape_mnk = (config.tile_m, config.tile_n, config.tile_k)
        scale_threads = config.num_mma_warps * 32
        if config.scale_role == "producer":
            scale_threads += 32
        self.scale_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=scale_threads
        )
        self.epilogue_barrier = pipeline.NamedBarrier(
            barrier_id=2, num_threads=config.num_mma_warps * 32
        )

    def _setup_static_layouts(self) -> None:
        cfg = self.config
        mma_op = warp.MmaMXF8Op(Float8E4M3FN, Float32, Float8E8M0FNU)
        self.tiled_mma = cute.make_tiled_mma(
            mma_op,
            cute.make_layout((cfg.atom_layout_m, cfg.atom_layout_n, 1)),
            permutation_mnk=sm120_utils.get_permutation_mnk(
                self.tile_shape_mnk, SF_VEC_SIZE, True
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
                swizzles[cfg.a_swizzle], Float8E4M3FN
            ),
            (cfg.tile_m, cfg.tile_k, cfg.stages),
            order=(0, 1, 2),
        )
        self.b_layout = cute.tile_to_shape(
            cute.nvgpu.warpgroup.make_smem_layout_atom(
                swizzles[cfg.b_swizzle], Float8E4M3FN
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
                self.tiled_mma, cfg.tile_k, cfg.stages
            )
        else:
            self.sfa_layout = blockscaled_utils.sm120_make_smem_layout_sfa(
                self.tiled_mma, padded_tile, SF_VEC_SIZE, cfg.stages
            )
        self.sfb_layout = blockscaled_utils.sm120_make_smem_layout_sfb(
            self.tiled_mma, padded_tile, SF_VEC_SIZE, cfg.stages
        )
        self.out_layout = cute.tile_to_shape(
            cute.nvgpu.warpgroup.make_smem_layout_atom(
                cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW128,
                BFloat16,
            ),
            (cfg.tile_m, cfg.tile_n, 1),
            order=(0, 1, 2),
        )

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
        scale_blocks_per_tile = cfg.tile_k // SF_VEC_SIZE
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
                scale = Uint8(0).bitcast(Float8E8M0FNU)
                if global_row < row_limit:
                    if is_a:
                        scale = sx[global_row, global_k_block]
                    else:
                        scale = sw[global_row, global_k_block]
                if is_a:
                    s_sfa[row, k_block * SF_VEC_SIZE, stage] = scale
                else:
                    s_sfb[row, k_block * SF_VEC_SIZE, stage] = scale
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
                        ).bitcast(Float8E8M0FNU)
                        for vec in cutlass.range_constexpr(cfg.scale_load_vec):
                            s_sfa[
                                row,
                                (k_block + vec) * SF_VEC_SIZE,
                                stage,
                            ] = loaded[vec]
                    else:
                        for vec in cutlass.range_constexpr(cfg.scale_load_vec):
                            s_sfa[
                                row,
                                (k_block + vec) * SF_VEC_SIZE,
                                stage,
                            ] = Uint8(0).bitcast(Float8E8M0FNU)
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
                        ).bitcast(Float8E8M0FNU)
                        for vec in cutlass.range_constexpr(cfg.scale_load_vec):
                            s_sfb[
                                row,
                                (k_block + vec) * SF_VEC_SIZE,
                                stage,
                            ] = loaded[vec]
                    else:
                        for vec in cutlass.range_constexpr(cfg.scale_load_vec):
                            s_sfb[
                                row,
                                (k_block + vec) * SF_VEC_SIZE,
                                stage,
                            ] = Uint8(0).bitcast(Float8E8M0FNU)

    @cute.jit
    def __call__(
        self,
        qx: cute.Tensor,
        qw: cute.Tensor,
        sx: cute.Tensor,
        sw: cute.Tensor,
        out: cute.Tensor,
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
            else cfg.tile_m * cfg.tile_k // SF_VEC_SIZE
        )
        scale_b_tile_elems = cfg.tile_n * cfg.tile_k // SF_VEC_SIZE
        scale_a_flat_layout = cute.make_layout(
            (scale_a_tile_elems, cfg.stages),
            stride=(1, scale_a_tile_elems),
        )
        scale_b_flat_layout = cute.make_layout(
            (scale_b_tile_elems, cfg.stages),
            stride=(1, scale_b_tile_elems),
        )
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

        @cute.struct
        class SharedStorage:
            a: cute.struct.Align[
                cute.struct.MemRange[Float8E4M3FN, cute.cosize(self.a_layout)],
                1024,
            ]
            b: cute.struct.Align[
                cute.struct.MemRange[Float8E4M3FN, cute.cosize(self.b_layout)],
                1024,
            ]
            sfa: cute.struct.Align[
                cute.struct.MemRange[
                    Float8E8M0FNU,
                    scale_a_tile_elems * cfg.stages
                    if cfg.scale_layout == "mma64x128"
                    else cute.cosize(self.sfa_layout),
                ],
                128,
            ]
            sfb: cute.struct.Align[
                cute.struct.MemRange[Float8E8M0FNU, cute.cosize(self.sfb_layout)],
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
            self.kernel(
                qx, qw, sx, sw, out,
                tma_x, tma_x_tensor, tma_w, tma_w_tensor,
                tma_sx, tma_sx_tensor, tma_sw, tma_sw_tensor,
                tma_out, tma_out_tensor, self.tiled_mma,
                self.a_layout, self.b_layout,
                self.sfa_layout, self.sfb_layout,
                scale_a_flat_layout, scale_b_flat_layout, self.out_layout,
            ).launch(
                grid=(m_tiles * n_tiles, 1, 1),
                block=(cfg.num_threads, 1, 1),
                stream=stream,
            )
        else:
            self.kernel(
                qx, qw, sx, sw, out,
                tma_x, tma_x_tensor, tma_w, tma_w_tensor,
                tma_x, tma_x_tensor, tma_w, tma_w_tensor,
                tma_out, tma_out_tensor, self.tiled_mma,
                self.a_layout, self.b_layout,
                self.sfa_layout, self.sfb_layout,
                scale_a_flat_layout, scale_b_flat_layout, self.out_layout,
            ).launch(
                grid=(m_tiles * n_tiles, 1, 1),
                block=(cfg.num_threads, 1, 1),
                stream=stream,
            )

    @cute.kernel
    def kernel(
        self,
        qx: cute.Tensor,
        qw: cute.Tensor,
        sx: cute.Tensor,
        sw: cute.Tensor,
        out: cute.Tensor,
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
        sx_row_view = cute.make_tensor(
            sx.iterator,
            cute.make_layout(
                (self.problem.m, self.problem.k // SF_VEC_SIZE),
                stride=(self.problem.k // SF_VEC_SIZE, 1),
            ),
        )
        sw_row_view = cute.make_tensor(
            sw.iterator,
            cute.make_layout(
                (self.problem.n, self.problem.k // SF_VEC_SIZE),
                stride=(self.problem.k // SF_VEC_SIZE, 1),
            ),
        )
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        linear_tile, _, _ = cute.arch.block_idx()
        m_tiles = cute.ceil_div(self.problem.m, cfg.tile_m)
        n_tiles = cute.ceil_div(self.problem.n, cfg.tile_n)

        if cutlass.const_expr(cfg.raster == "n"):
            full_group = n_tiles * cfg.grid_swizzle
            group = linear_tile // full_group
            offset = linear_tile % full_group
            rows_in_group = cutlass.min(
                cfg.grid_swizzle, m_tiles - group * cfg.grid_swizzle
            )
            block_n = offset // rows_in_group
            block_m = group * cfg.grid_swizzle + offset % rows_in_group
        else:
            full_group = m_tiles * cfg.grid_swizzle
            group = linear_tile // full_group
            offset = linear_tile % full_group
            cols_in_group = cutlass.min(
                cfg.grid_swizzle, n_tiles - group * cfg.grid_swizzle
            )
            block_m = offset // cols_in_group
            block_n = group * cfg.grid_swizzle + offset % cols_in_group

        storage = cutlass.utils.SmemAllocator().allocate(self.shared_storage)
        s_a = storage.a.get_tensor(a_layout.outer, swizzle=a_layout.inner)
        s_b = storage.b.get_tensor(b_layout.outer, swizzle=b_layout.inner)
        s_sfa = storage.sfa.get_tensor(sfa_layout)
        s_sfb = storage.sfb.get_tensor(sfb_layout)
        s_sfa_flat = cute.make_tensor(s_sfa.iterator, scale_a_flat_layout)
        s_sfb_flat = cute.make_tensor(s_sfb.iterator, scale_b_flat_layout)
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
            else cfg.tile_m * cfg.tile_k // SF_VEC_SIZE
        )
        scale_b_tile_elems = cfg.tile_n * cfg.tile_k // SF_VEC_SIZE
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
            tx_count=(cfg.tile_m + cfg.tile_n) * cfg.tile_k
            + (
                scale_a_tile_elems + scale_b_tile_elems
                if cfg.scale_role == "tma"
                else 0
            ),
            barrier_storage=storage.pipeline.data_ptr(),
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, cfg.stages
        )
        consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, cfg.stages
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
                Float8E4M3FN, False, cfg.a_ldmatrix_matrices
            ),
            tiled_mma,
        )
        copy_b = cute.make_tiled_copy_B(
            _make_ldmatrix_atom(
                Float8E4M3FN, False, cfg.b_ldmatrix_matrices
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
            _make_scale_s2r_atom(Float8E8M0FNU, cfg.sfa_s2r_bits),
            sm120_utils.get_layoutSFA_TV(tiled_mma),
            (
                cute.size(tiled_mma.permutation_mnk[0]),
                cute.size(tiled_mma.permutation_mnk[2]),
            ),
        )
        copy_sfb = cute.make_tiled_copy(
            _make_scale_s2r_atom(Float8E8M0FNU, cfg.sfb_s2r_bits),
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

        g_out = cute.local_tile(
            out, (cfg.tile_m, cfg.tile_n), (block_m, block_n)
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
            r2s_atom = cute.make_copy_atom(
                warp.StMatrix8x8x16bOp(False, cfg.store_vec), BFloat16
            )
            c_atom = cute.make_copy_atom(
                warp.StMatrix8x8x16bOp(False, cfg.store_vec), BFloat16
            )
            c_copy = cute.make_tiled_copy_C_atom(c_atom, tiled_mma)
            copy_r2s = cute.make_tiled_copy_S(r2s_atom, c_copy)
            thr_r2s = copy_r2s.get_slice(tidx)
            t_rs_s = thr_r2s.partition_D(s_out)
            t_rs_acc = copy_r2s.retile(accumulators)
            r_out_shape = cute.shape(thr_r2s.partition_S(s_out))
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
                cute.group_modes(s_out, 0, 2),
                g_out_tma,
            )
            out_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=1,
                producer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, cfg.num_mma_warps * 32
                ),
            )

        if warp_idx == cfg.num_mma_warps:
            cute.arch.setmaxregister_decrease(cfg.producer_registers)
        else:
            cute.arch.setmaxregister_increase(cfg.consumer_registers)

        num_k_tiles = cute.ceil_div(self.problem.k, cfg.tile_k)
        scale_blocks_per_tile = cfg.tile_k // SF_VEC_SIZE
        scale_chunks_per_row = scale_blocks_per_tile // cfg.scale_load_vec
        a_scale_count = cfg.tile_m * scale_chunks_per_row
        total_scale_count = (
            cfg.tile_m + cfg.tile_n
        ) * scale_chunks_per_row
        for k_tile in cutlass.range(num_k_tiles, unroll=1):
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
                        scale = Uint8(0).bitcast(Float8E8M0FNU)
                        if global_row < row_limit:
                            if is_a:
                                scale = sx_row_view[global_row, global_k_block]
                            else:
                                scale = sw_row_view[global_row, global_k_block]
                        if is_a:
                            s_sfa[row, k_block * SF_VEC_SIZE, stage] = scale
                        else:
                            s_sfb[row, k_block * SF_VEC_SIZE, stage] = scale
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
                                ).bitcast(Float8E8M0FNU)
                                for vec in cutlass.range_constexpr(
                                    cfg.scale_load_vec
                                ):
                                    s_sfa[
                                        row,
                                        (k_block + vec) * SF_VEC_SIZE,
                                        stage,
                                    ] = loaded[vec]
                            else:
                                for vec in cutlass.range_constexpr(
                                    cfg.scale_load_vec
                                ):
                                    s_sfa[
                                        row,
                                        (k_block + vec) * SF_VEC_SIZE,
                                        stage,
                                    ] = Uint8(0).bitcast(Float8E8M0FNU)
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
                                ).bitcast(Float8E8M0FNU)
                                for vec in cutlass.range_constexpr(
                                    cfg.scale_load_vec
                                ):
                                    s_sfb[
                                        row,
                                        (k_block + vec) * SF_VEC_SIZE,
                                        stage,
                                    ] = loaded[vec]
                            else:
                                for vec in cutlass.range_constexpr(
                                    cfg.scale_load_vec
                                ):
                                    s_sfb[
                                        row,
                                        (k_block + vec) * SF_VEC_SIZE,
                                        stage,
                                    ] = Uint8(0).bitcast(Float8E8M0FNU)
                if cutlass.const_expr(cfg.scale_role != "tma"):
                    self.scale_barrier.arrive_and_wait()
                if cutlass.const_expr(
                    cfg.scale_role != "tma"
                    and cfg.scale_schedule == "before_wait"
                ):
                    ready = tma_pipeline.consumer_try_wait(consumer_state)
                    tma_pipeline.consumer_wait(consumer_state, ready)
                for k_block in cutlass.range_constexpr(scale_blocks_per_tile):
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
                        cute.filter_zeros(t_cs_sfa)[None, None, k_block, stage],
                        cute.filter_zeros(t_cr_sfa_copy)[None, None, k_block],
                    )
                    cute.copy(
                        copy_sfb,
                        cute.filter_zeros(t_cs_sfb)[None, None, k_block, stage],
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
                tma_pipeline.consumer_release(consumer_state)
                consumer_state.advance()

        if warp_idx == cfg.num_mma_warps:
            tma_pipeline.producer_tail(producer_state)

        if cutlass.const_expr(cfg.epilogue == "direct"):
            if warp_idx < cfg.num_mma_warps:
                for elem in cutlass.range(cute.size(accumulators), unroll_full=True):
                    coord = t_cc_out[elem]
                    if coord[0] < self.problem.m and coord[1] < self.problem.n:
                        t_cg_out[elem] = BFloat16(accumulators[elem])
        else:
            if warp_idx < cfg.num_mma_warps:
                for elem in cutlass.range(cute.size(r_out), unroll_full=True):
                    r_out[elem] = BFloat16(t_rs_acc[elem])
                cute.copy(copy_r2s, r_out, t_rs_s[(None, None, None, 0)])
                cute.arch.fence_proxy("async.shared", space="cta")
                self.epilogue_barrier.arrive_and_wait()
                if warp_idx == 0:
                    cute.copy(tma_out, out_s[(None, 0)], out_g[(None, 0)])
                    out_pipeline.producer_commit()
                    out_pipeline.producer_acquire()
                    out_pipeline.producer_tail()


@lru_cache(maxsize=None)
def compile_mxfp8_gemm(
    problem: MXFP8Problem,
    config: MXFP8GemmConfig = MXFP8GemmConfig(),
):
    kernel = MXFP8GemmKernel(problem, config)
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
    out = cute.runtime.make_fake_tensor(
        BFloat16,
        (problem.m, problem.n),
        stride=(problem.n, 1),
        assumed_align=16,
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel,
        qx,
        qw,
        sx,
        sw,
        out,
        stream,
        options=(
            "--enable-tvm-ffi --opt-level 3 "
            f"--ptxas-options '-O3 -v --maxrregcount={config.maxrregcount}'"
        ),
    )


__all__ = ["MXFP8GemmConfig", "MXFP8GemmKernel", "compile_mxfp8_gemm"]
