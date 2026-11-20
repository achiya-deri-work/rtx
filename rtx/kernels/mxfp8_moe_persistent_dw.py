"""Unsplit expert dW with a common TMA operand panel shared by MMA groups.

Each CTA owns neighboring 64x64 output tiles along one output axis. Each
consumer group has the ordinary 64x64 register footprint, rather than using
a wider accumulator tile. Across the full K loop the producer issues one
TMA copy for the common operand and one for each varying operand per stage.
"""

from functools import lru_cache

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm120_utils
from cutlass import BFloat16, Float8E4M3FN, Float8E8M0FNU, Float32, Int32, Int64, Uint8
from cutlass.cute.nvgpu import cpasync, warp

from .mxfp8_fwd import _make_ldmatrix_atom, _make_scale_s2r_atom, _make_sm120_sfa_layout_64


class PersistentExpertDW:
    def __init__(self, experts, problem, config, axis, tiles, active_k=False):
        self.experts = experts
        self.problem = problem
        self.config = config
        self.active_k = active_k
        self.tiles = tiles
        self.reuse_a = axis == "n"
        self.tile_count_m = (problem.m + 63) // 64
        self.tile_count_n = (problem.n + 63) // 64
        self.groups_m = (self.tile_count_m + tiles - 1) // tiles
        self.groups_n = (self.tile_count_n + tiles - 1) // tiles
        self.a_panels = 1 if self.reuse_a else tiles
        self.b_panels = tiles if self.reuse_a else 1
        self.grid = self.tile_count_m * self.groups_n if self.reuse_a else self.tile_count_n * self.groups_m
        self.consumer_warps = tiles * config.num_mma_warps
        self.consumer_threads = self.consumer_warps * 32
        self.scale_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=self.consumer_threads)

    @cute.jit
    def __call__(self, qa, qb, sa, sb, out, stream, active_capacity=None):
        cfg = self.config
        m, n, k = self.problem.m, self.problem.n, self.problem.k
        block_q = (64, cfg.tile_k)
        qa = cute.make_tensor(qa.iterator, cute.make_layout((m, k, self.experts), stride=(k, 1, m * k)))
        qb = cute.make_tensor(qb.iterator, cute.make_layout((n, k, self.experts), stride=(k, 1, n * k)))
        tiled_mma = cute.make_tiled_mma(
            warp.MmaMXF8Op(Float8E4M3FN, Float32, Float8E8M0FNU),
            cute.make_layout((2, 2, 1)),
            permutation_mnk=sm120_utils.get_permutation_mnk((64, 64, cfg.tile_k), 32, True),
        )
        swizzles = {
            "none": cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_INTER,
            "32b": cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW32,
            "64b": cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW64,
            "128b": cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW128,
        }
        a_layout = cute.tile_to_shape(
            cute.nvgpu.warpgroup.make_smem_layout_atom(swizzles[cfg.a_swizzle], Float8E4M3FN),
            (64, cfg.tile_k, cfg.stages, self.a_panels),
            order=(0, 1, 2, 3),
        )
        b_layout = cute.tile_to_shape(
            cute.nvgpu.warpgroup.make_smem_layout_atom(swizzles[cfg.b_swizzle], Float8E4M3FN),
            (64, cfg.tile_k, cfg.stages, self.b_panels),
            order=(0, 1, 2, 3),
        )
        s_layout = _make_sm120_sfa_layout_64(tiled_mma, cfg.tile_k, 32, cfg.stages)
        panel_stride = cute.cosize(s_layout)
        sfa_layout = cute.append(s_layout, cute.make_layout(self.a_panels, stride=panel_stride))
        sfb_layout = cute.append(s_layout, cute.make_layout(self.b_panels, stride=panel_stride))
        tma_a, tma_qa = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), qa, cute.slice_(a_layout, (None, None, 0, 0)), block_q,
        )
        tma_b, tma_qb = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), qb, cute.slice_(b_layout, (None, None, 0, 0)), block_q,
        )

        @cute.struct
        class Storage:
            a: cute.struct.Align[cute.struct.MemRange[Float8E4M3FN, cute.cosize(a_layout)], 1024]
            b: cute.struct.Align[cute.struct.MemRange[Float8E4M3FN, cute.cosize(b_layout)], 1024]
            sa: cute.struct.Align[cute.struct.MemRange[Float8E8M0FNU, cute.cosize(sfa_layout)], 128]
            sb: cute.struct.Align[cute.struct.MemRange[Float8E8M0FNU, cute.cosize(sfb_layout)], 128]
            barriers: cute.struct.Align[cute.struct.MemRange[Int64, 2 * cfg.stages], 8]

        self.storage = Storage
        self.kernel(
            tma_a, tma_qa, tma_b, tma_qb, sa, sb, out, tiled_mma, a_layout, b_layout,
            sfa_layout, sfb_layout, active_capacity,
        ).launch(grid=(self.grid, 1, self.experts), block=(self.consumer_threads + 32, 1, 1), stream=stream)

    @cute.jit
    def _stage_scales(self, sa, sb, sfa, sfb, stage, expert, block_m, block_n, k_tile, tidx):
        cfg = self.config
        blocks = cfg.tile_k // 32
        a_count = self.a_panels * 64 * blocks
        total_count = (self.tiles + 1) * 64 * blocks
        m, n, k = self.problem.m, self.problem.n, self.problem.k
        for task in cutlass.range(tidx, total_count, self.consumer_threads, unroll=1):
            is_a = task < a_count
            local = task if is_a else task - a_count
            panel = local // (64 * blocks)
            row = local // blocks % 64
            column = local % blocks
            block = k_tile * blocks + column
            scale = Uint8(0).bitcast(Float8E8M0FNU)
            if is_a:
                source_row = (block_m + panel) * 64 + row
                if source_row < m and block < k // 32:
                    scale = sa[expert * m + source_row, block].bitcast(Float8E8M0FNU)
                sfa[row, column * 32, stage, panel] = scale
            else:
                source_row = (block_n + panel) * 64 + row
                if source_row < n and block < k // 32:
                    scale = sb[expert * n + source_row, block].bitcast(Float8E8M0FNU)
                sfb[row, column * 32, stage, panel] = scale

    @cute.kernel
    def kernel(
        self, tma_a: cute.CopyAtom, tma_qa: cute.Tensor, tma_b: cute.CopyAtom,
        tma_qb: cute.Tensor, sa: cute.Tensor, sb: cute.Tensor, out: cute.Tensor,
        tiled_mma: cute.TiledMma, a_layout: cute.ComposedLayout, b_layout: cute.ComposedLayout,
        sfa_layout: cute.Layout, sfb_layout: cute.Layout, active_capacity,
    ):
        cfg = self.config
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        cta, _, expert = cute.arch.block_idx()
        if cutlass.const_expr(self.reuse_a):
            block_m = cta // self.groups_n
            block_n = cta % self.groups_n * self.tiles
        else:
            block_m = cta % self.groups_m * self.tiles
            block_n = cta // self.groups_m
        k_tiles = cute.ceil_div(self.problem.k, cfg.tile_k)
        if cutlass.const_expr(self.active_k):
            # Read the device bound uniformly, before diverging into the
            # producer and all consumer groups. Tensor-map extents/strides
            # stay physical; even a partial final tile runs the same TMA
            # and MMA schedule with the caller's scrubbed operand tail.
            active = cutlass.min(self.problem.k, cutlass.max(active_capacity[0], 0))
            k_tiles = cute.ceil_div(active, cfg.tile_k)
        storage = utils.SmemAllocator().allocate(self.storage)
        a = storage.a.get_tensor(a_layout.outer, swizzle=a_layout.inner)
        b = storage.b.get_tensor(b_layout.outer, swizzle=b_layout.inner)
        sfa = storage.sa.get_tensor(sfa_layout)
        sfb = storage.sb.get_tensor(sfb_layout)
        pipe = pipeline.PipelineTmaAsync.create(
            num_stages=cfg.stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.consumer_warps),
            # One common tile plus G private tiles, including zero-filled
            # bytes on the physical boundary of a tensor map.
            tx_count=(self.tiles + 1) * 64 * cfg.tile_k,
            barrier_storage=storage.barriers.data_ptr(),
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        if warp_idx == self.consumer_warps:
            g_a = cute.local_tile(tma_qa[None, None, expert], (64, cfg.tile_k), (None, None))
            g_b = cute.local_tile(tma_qb[None, None, expert], (64, cfg.tile_k), (None, None))
            ta_s, ta_g = cpasync.tma_partition(
                tma_a, 0, cute.make_layout(1), cute.group_modes(a, 0, 2), cute.group_modes(g_a, 0, 2),
            )
            tb_s, tb_g = cpasync.tma_partition(
                tma_b, 0, cute.make_layout(1), cute.group_modes(b, 0, 2), cute.group_modes(g_b, 0, 2),
            )
            producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, cfg.stages)
            for k_tile in cutlass.range(k_tiles, unroll=cfg.k_tile_unroll):
                pipe.producer_acquire(producer)
                bar = pipe.producer_get_barrier(producer)
                for i in cutlass.range_constexpr(self.a_panels):
                    # A partial group completes the same barrier phases with
                    # a safe descriptor coordinate for the inactive panel.
                    m_tile = cutlass.min(block_m + i, self.tile_count_m - 1)
                    cute.copy(tma_a, ta_g[None, m_tile, k_tile], ta_s[None, producer.index, i], tma_bar_ptr=bar)
                for i in cutlass.range_constexpr(self.b_panels):
                    n_tile = cutlass.min(block_n + i, self.tile_count_n - 1)
                    cute.copy(tma_b, tb_g[None, n_tile, k_tile], tb_s[None, producer.index, i], tma_bar_ptr=bar)
                producer.advance()
            if cutlass.const_expr(self.active_k):
                # With no stages issued, no consumer arrival is owed. The
                # consumers still store their zero-initialized dW fragments.
                if k_tiles > 0:
                    pipe.producer_tail(producer)
            else:
                pipe.producer_tail(producer)
        else:
            self._consume(pipe, sa, sb, a, b, sfa, sfb, out, tiled_mma, tidx, expert, block_m, block_n, k_tiles)

    @cute.jit
    def _consume(self, pipe, sa, sb, a, b, sfa, sfb, out, tiled_mma, tidx, expert, block_m, block_n, k_tiles):
        cfg = self.config
        group_threads = cfg.num_mma_warps * 32
        tid = tidx % group_threads
        group = cute.arch.make_warp_uniform(tidx // group_threads)
        panel_a = Int32(0) if self.reuse_a else group
        panel_b = group if self.reuse_a else Int32(0)
        ma, nb = block_m + panel_a, block_n + panel_b
        m, n = self.problem.m, self.problem.n
        g_out = cute.make_tensor(out.iterator + expert * m * n, cute.make_layout((m, n), stride=(n, 1)))
        coords = cute.make_identity_tensor((m, n))
        mma = tiled_mma.get_slice(tid)
        t_out = mma.partition_C(cute.local_tile(g_out, (64, 64), (ma, nb)))
        t_coords = mma.partition_C(cute.local_tile(coords, (64, 64), (ma, nb)))
        accum = cute.make_rmem_tensor(t_coords.shape, Float32)
        accum.fill(0.0)
        a_panel = a[None, None, None, panel_a]
        b_panel = b[None, None, None, panel_b]
        sfa_panel = sfa[None, None, None, panel_a]
        sfb_panel = sfb[None, None, None, panel_b]
        ra = tiled_mma.make_fragment_A(mma.partition_A(a_panel)[None, None, None, 0])
        rb = tiled_mma.make_fragment_B(mma.partition_B(b_panel)[None, None, None, 0])
        rsa = sm120_utils.partition_fragment_SFA(sfa_panel[None, None, 0], mma, tid)
        rsb = sm120_utils.partition_fragment_SFB(sfb_panel[None, None, 0], mma, tid)
        rsa = cute.group_modes(rsa, 2, cute.rank(rsa))
        rsb = cute.group_modes(rsb, 2, cute.rank(rsb))
        ca = cute.make_tiled_copy_A(_make_ldmatrix_atom(Float8E4M3FN, False, cfg.a_ldmatrix_matrices), tiled_mma)
        cb = cute.make_tiled_copy_B(_make_ldmatrix_atom(Float8E4M3FN, False, cfg.b_ldmatrix_matrices), tiled_mma)
        csa = cute.make_tiled_copy(
            _make_scale_s2r_atom(Float8E8M0FNU, cfg.sfa_s2r_bits),
            sm120_utils.get_layoutSFA_TV(tiled_mma),
            (cute.size(tiled_mma.permutation_mnk[0]), cute.size(tiled_mma.permutation_mnk[2])),
        )
        csb = cute.make_tiled_copy(
            _make_scale_s2r_atom(Float8E8M0FNU, cfg.sfb_s2r_bits),
            sm120_utils.get_layoutSFB_TV(tiled_mma),
            (cute.size(tiled_mma.permutation_mnk[1]), cute.size(tiled_mma.permutation_mnk[2])),
        )
        ta, tb, tsa, tsb = ca.get_slice(tid), cb.get_slice(tid), csa.get_slice(tid), csb.get_slice(tid)
        la, lb = ta.partition_S(a_panel), tb.partition_S(b_panel)
        lsa = cute.filter_zeros(tsa.partition_S(sfa_panel))
        lsb = cute.filter_zeros(tsb.partition_S(sfb_panel))
        da, db = ta.retile(ra), tb.retile(rb)
        dsa = cute.filter_zeros(tsa.retile(rsa))
        dsb = cute.filter_zeros(tsb.retile(rsb))
        consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, cfg.stages)
        for k_tile in cutlass.range(k_tiles, unroll=cfg.k_tile_unroll):
            stage = consumer.index
            if cutlass.const_expr(cfg.scale_schedule == "after_wait"):
                pipe.consumer_wait(consumer)
            self._stage_scales(sa, sb, sfa, sfb, stage, expert, block_m, block_n, k_tile, tidx)
            self.scale_barrier.arrive_and_wait()
            if cutlass.const_expr(cfg.scale_schedule == "before_wait"):
                pipe.consumer_wait(consumer)
            for kk in cutlass.range_constexpr(cute.size(ra, mode=[2])):
                cute.copy(ca, la[None, None, kk, stage], da[None, None, kk])
                cute.copy(cb, lb[None, None, kk, stage], db[None, None, kk])
                cute.copy(csa, lsa[None, None, kk, stage], dsa[None, None, kk])
                cute.copy(csb, lsb[None, None, kk, stage], dsb[None, None, kk])
                cute.gemm(tiled_mma, accum, [ra[None, None, kk], rsa[None, None, kk]],
                          [rb[None, None, kk], rsb[None, None, kk]], accum)
            # With before-wait staging, readers must meet writers from every
            # group before any consumer can independently recycle scales.
            if cutlass.const_expr(cfg.scale_schedule == "before_wait"):
                self.scale_barrier.arrive_and_wait()
            # The empty stage counts one arrival per consumer warp. The
            # producer cannot overwrite the common operand until all G
            # consumer groups finish their shared reads.
            cute.arch.fence_view_async_shared()
            pipe.consumer_release(consumer)
            consumer.advance()
        for i in cutlass.range(cute.size(accum), unroll_full=True):
            coord = t_coords[i]
            if coord[0] < m and coord[1] < n:
                t_out[i] = BFloat16(accum[i])


@lru_cache(maxsize=128)
def compile_persistent_dw(experts, problem, config, axis="n", tiles=2, active_k=False):
    """Compile a byte-scale launcher, optionally with a final int32[1] K bound.

    The prefix comes from the device group32 scan. As with packed active dW,
    callers must initialize both operands and scales through the last physical
    K tile, with zero contribution for rows beyond the prefix in that tile.
    No suffix tiles are read; no device allocation or stream argument is needed.
    """
    from ..configs.moe_backward import persistent_dw_rejection

    if type(experts) is not int or not 1 <= experts <= 65535:
        raise ValueError("persistent dW requires an expert count in [1, 65535]")
    if type(active_k) is not bool:
        raise ValueError("persistent dW active_k must be a bool")
    if reason := persistent_dw_rejection(problem, config, axis, tiles):
        raise ValueError(f"illegal persistent dW: {reason}")
    m, n, k = problem.m, problem.n, problem.k
    if experts * max(m * k, n * k, m * n) >= 2**31:
        raise ValueError("persistent dW operands exceed the native signed index range")

    def tensor(dtype, rows, cols):
        return cute.runtime.make_fake_tensor(dtype, (experts * rows, cols), stride=(cols, 1), assumed_align=16)

    extra = ()
    if active_k:
        from .mxfp8_moe_active import fake_prefix

        extra = (fake_prefix(),)
    return cute.compile(
        PersistentExpertDW(experts, problem, config, axis, tiles, active_k),
        tensor(Float8E4M3FN, m, k), tensor(Float8E4M3FN, n, k),
        tensor(Uint8, m, k // 32), tensor(Uint8, n, k // 32), tensor(BFloat16, m, n),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True), *extra,
        options="--enable-tvm-ffi --opt-level 3",
    )
