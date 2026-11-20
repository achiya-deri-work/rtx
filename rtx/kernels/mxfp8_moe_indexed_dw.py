"""Gate dW with on-chip quantization of the indexed, transposed X operand.

The packed derivative operand (qP transpose) is unchanged. An output CTA
gathers BF16 X in feature-contiguous strips, reduces one amax per feature / 32
expert slots, and publishes the transposed FP8 words plus scales directly to
the MMA stage. There is no global quantized X, no split-K workspace, and the
only output is the BF16 cast of the full FP32 reduction.
"""
from functools import lru_cache

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm120_utils
from cutlass import BFloat16, Float8E4M3FN, Float8E8M0FNU, Float32, Int16, Int32, Int64, Uint8, Uint16, Uint32, Uint64
from cutlass.experimental.primitives import nvvm_wrapper as nvvm

from ..configs.moe_backward import indexed_gate_dw_rejection, indexed_gate_dw_schedule
from .mxfp8_fwd import _make_ldmatrix_atom, _make_scale_s2r_atom
from .mxfp8_moe_cooperative import CooperativeExpertGemm
from .mxfp8_moe_warp_quant import ExpertWarpTransposeOperand


class _IndexedXStage(ExpertWarpTransposeOperand):
    """The standalone warp transpose recipe with a stage-local sink."""

    def __init__(self, tokens, features, slots, config, token_lanes, gather_pairs):
        super().__init__(features, slots, config, token_lanes)
        self.tokens = tokens
        self.gather_pairs = gather_pairs
        self.gather_features = self.features_per_warp * gather_pairs

    @cute.jit
    def load(self, x, token, row):
        bits = Uint64(0)
        if token >= 0 and token < self.tokens and row < self.rows:
            index = token*self.rows + row
            pointer = x.iterator + index
            if cutlass.const_expr(self.gather_pairs == 2):
                if row + 3 < self.rows and pointer.toint() % 8 == 0:
                    bits = nvvm.inline_ptx_hl('ld.global.u64 {$w0}, [{$r0}];',
                        write_only_types=[Uint64], read_only_args=[pointer])
                else:
                    lo = self.load_pair(x, index, Int32(0), row, Int32(0), None)
                    hi = self.load_pair(x, index+2, Int32(0), row+2, Int32(0), None)
                    bits = Uint64(lo) | (Uint64(hi) << 32)
            else:
                bits = Uint64(self.load_pair(x, index, Int32(0), row, Int32(0), None))
        return bits

    @cute.jit
    def encode_pair(self, bits, inv_a, inv_b):
        if cutlass.const_expr(self.config.quant_math == 'bf16x2'):
            reciprocal = (Int32(BFloat16(inv_a).bitcast(Uint16))
                          | (Int32(BFloat16(inv_b).bitcast(Uint16)) << 16))
            scaled = nvvm.mul_bf16x2(Int32(bits), reciprocal)
            packed = nvvm.inline_ptx_hl('cvt.rn.satfinite.e4m3x2.bf16x2 {$w0}, {$r0};',
                write_only_types=[Int16], read_only_args=[scaled])
            a, b = Uint32(packed & Int16(255)), Uint32((packed >> 8) & Int16(255))
        else:
            a32 = Float32(Uint16(bits & Uint32(65535)).bitcast(BFloat16)) * inv_a
            b32 = Float32(Uint16(bits >> 16).bitcast(BFloat16)) * inv_b
            packed = nvvm.inline_ptx_hl('cvt.rn.satfinite.e4m3x2.f32 {$w0}, {$r0}, {$r1};',
                write_only_types=[Int16], read_only_args=[a32, b32])
            a, b = Uint32((packed >> 8) & Int16(255)), Uint32(packed & Int16(255))
        return a, b

    @cute.jit
    def fill(self, x, mapping, qb, sf, layout, tile_n: cutlass.Constexpr,
             tile_k: cutlass.Constexpr, row_base, k_base, stage, thread,
             num_warps: cutlass.Constexpr):
        physical = cute.make_composed_layout(layout.inner, 0, qb.layout)
        qptr = cute.recast_ptr(qb.iterator, dtype=Uint8)
        lane = cute.arch.lane_idx()
        lane_feature = lane % self.feature_pairs
        lane_token = lane // self.feature_pairs
        feature_groups = tile_n // self.gather_features
        for task in cutlass.range(thread//32, feature_groups * (tile_k//32), num_warps, unroll=1):
            k_block = task // feature_groups
            row = (task % feature_groups)*self.gather_features + 2*self.gather_pairs*lane_feature
            values = cute.make_rmem_tensor((self.gather_pairs, self.rounds), Uint32)
            maximum = cute.make_rmem_tensor((2*self.gather_pairs,), Int32)
            maximum.fill(Int32(0))
            for item in cutlass.range_constexpr(self.rounds):
                slot = k_base + k_block*32 + lane_token + item*self.token_lanes
                token = Int32(-1)
                if lane_feature == 0 and slot < self.k:
                    token = mapping[slot]
                token = cute.arch.shuffle_sync(token, lane_token*self.feature_pairs)
                bits = self.load(x, token, row_base+row)
                for pair in cutlass.range_constexpr(self.gather_pairs):
                    data = Uint32(bits >> (pair*32))
                    values[pair, item] = data
                    maximum[2*pair] = cutlass.max(maximum[2*pair], Int32(data & Uint32(0x7fff)))
                    maximum[2*pair+1] = cutlass.max(maximum[2*pair+1], Int32((data >> 16) & Uint32(0x7fff)))
            pos = lane_token*4
            if cutlass.const_expr(self.token_lanes >= 4):
                order = ((lane_token & 1) << 1) | ((lane_token >> 1) & 1)
                pos = order*self.token_lanes + (lane_token//4)*4
            for pair in cutlass.range_constexpr(self.gather_pairs):
                max_a = self.maximum(maximum[2*pair], lane_feature)
                max_b = self.maximum(maximum[2*pair+1], lane_feature)
                sf_a, inv_a = self._scale_from_amax(Float32(Uint16(max_a).bitcast(BFloat16)))
                sf_b, inv_b = self._scale_from_amax(Float32(Uint16(max_b).bitcast(BFloat16)))
                qa = cute.make_rmem_tensor((self.words,), Uint32)
                qb_words = cute.make_rmem_tensor((self.words,), Uint32)
                qa.fill(Uint32(0)); qb_words.fill(Uint32(0))
                for item in cutlass.range_constexpr(self.rounds):
                    a, b = self.encode_pair(values[pair, item], inv_a, inv_b)
                    qa[item//4] |= a << ((item%4)*8)
                    qb_words[item//4] |= b << ((item%4)*8)
                local_row = row + 2*pair
                for word in cutlass.range_constexpr(self.words):
                    a = self.transpose_word(qa[word], lane_token)
                    b = self.transpose_word(qb_words[word], lane_token)
                    col = k_block*32 + word*4*self.token_lanes + pos
                    pa = cute.recast_ptr(qptr + physical((local_row, col, stage)), dtype=Uint32)
                    pb = cute.recast_ptr(qptr + physical((local_row+1, col, stage)), dtype=Uint32)
                    cute.make_tensor(pa, cute.make_layout(1))[0] = a
                    cute.make_tensor(pb, cute.make_layout(1))[0] = b
                if lane_token == 0:
                    sf[local_row, k_block*32, stage] = sf_a
                    sf[local_row+1, k_block*32, stage] = sf_b


class IndexedGateDW(CooperativeExpertGemm):
    def __init__(self, tokens, experts, problem, gemm_config, quant_config,
                 token_lanes, gather_pairs, producer_warps):
        if reason := indexed_gate_dw_rejection(problem, gemm_config, quant_config,
                                               token_lanes, gather_pairs, producer_warps):
            raise ValueError(reason)
        if tokens <= 0 or experts <= 0 or max(tokens*problem.n, experts*problem.k,
                experts*problem.m*max(problem.k, problem.n)) >= 2**31:
            raise ValueError('Indexed gate dW requires positive shapes within signed indexing range')
        super().__init__(problem, indexed_gate_dw_schedule(gemm_config, producer_warps))
        self.batch_count = experts
        self.x_stage = _IndexedXStage(tokens, problem.n, problem.k, quant_config, token_lanes, gather_pairs)
        self.producer_warps = producer_warps

    def _setup_static_layouts(self):
        cfg = self.config
        if cfg.tile_k == 128:
            super()._setup_static_layouts()
            return
        self.tiled_mma = cute.make_tiled_mma(self._make_mma_op(),
            cute.make_layout((cfg.atom_layout_m, cfg.atom_layout_n, 1)),
            permutation_mnk=sm120_utils.get_permutation_mnk(
                self.tile_shape_mnk, self.sf_vec_size, self.use_mxf8f6f4))
        atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW32, self.ab_dtype)
        self.a_layout = cute.tile_to_shape(atom, (cfg.tile_m, cfg.tile_k, cfg.stages), order=(0,1,2))
        self.b_layout = cute.tile_to_shape(atom, (cfg.tile_n, cfg.tile_k, cfg.stages), order=(0,1,2))
        # The short native-K path patches the layout's stage stride, keeping
        # an independent scale byte for every physical 32-slot block.
        self.sfa_layout = cute.make_layout((cfg.tile_m, (32, cfg.tile_k//32), cfg.stages),
            stride=(1, (0, cfg.tile_m), cfg.tile_m*(cfg.tile_k//32)))
        self.sfb_layout = cute.make_layout((cfg.tile_n, (32, cfg.tile_k//32), cfg.stages),
            stride=(1, (0, cfg.tile_n), cfg.tile_n*(cfg.tile_k//32)))

    @cute.jit
    def __call__(self, a, sa, x, mapping, out, stream):
        scales = cute.make_tensor(cute.recast_ptr(sa.iterator, dtype=Float8E8M0FNU), sa.layout)
        self.kernel(a, scales, x, mapping, out).launch(grid=(self.total_tiles, 1, self.batch_count),
            block=((self.config.num_mma_warps + self.producer_warps)*32, 1, 1), stream=stream)

    @cute.jit
    def _produce(self, a, sa, x, mapping, qa, qb, sfa, sfb, bm, bn, kt, stage, thread,
                 warps: cutlass.Constexpr):
        cfg = self.config
        m, k = self.problem.m, self.problem.k
        qa_layout = cute.make_composed_layout(self.a_layout.inner, 0, qa.layout)
        qptr = cute.recast_ptr(qa.iterator, dtype=Uint8)
        rows = cfg.tile_k//16
        k_base = kt*cfg.tile_k
        for task in cutlass.range(thread, cfg.tile_m*rows, warps*32, unroll=1):
            row, col = task//rows, (task%rows)*16
            gr, gk = bm*cfg.tile_m + row, k_base + col
            valid = Int32(0)
            if gr < m and gk < k:
                valid = Int32(16)
            safe_row, safe_k = cutlass.min(gr, m-1), cutlass.min(gk, k-16)
            nvvm.cp_async_shared_global(qptr + qa_layout((row, col, stage)),
                a.iterator + safe_row*k + safe_k, 16, 'cg', cp_size=valid)
        cute.arch.cp_async_commit_group()
        for task in cutlass.range(thread, cfg.tile_m*(cfg.tile_k//32), warps*32, unroll=1):
            row, kb = task//(cfg.tile_k//32), task%(cfg.tile_k//32)
            gr, gk = bm*cfg.tile_m + row, k_base//32 + kb
            value = Uint8(0).bitcast(Float8E8M0FNU)
            if gr < m and gk < k//32:
                value = sa[gr, gk]
            sfa[row, kb*32, stage] = value
        self.x_stage.fill(x, mapping, qb, sfb, self.b_layout, cfg.tile_n, cfg.tile_k,
                          bn*cfg.tile_n, k_base, stage, thread, warps)
        cute.arch.cp_async_wait_group(0)

    @cute.jit
    def _consume(self, acc, qa, qb, sfa, sfb, stage, thread):
        cfg = self.config
        mma = self.tiled_mma
        thr = mma.get_slice(thread)
        ra = mma.make_fragment_A(thr.partition_A(qa)[None, None, None, 0])
        rb = mma.make_fragment_B(thr.partition_B(qb)[None, None, None, 0])
        rsa = sm120_utils.partition_fragment_SFA(sfa[None, None, 0], thr, thread)
        rsb = sm120_utils.partition_fragment_SFB(sfb[None, None, 0], thr, thread)
        rsa = cute.group_modes(rsa, 2, cute.rank(rsa)); rsb = cute.group_modes(rsb, 2, cute.rank(rsb))
        ca = cute.make_tiled_copy_A(_make_ldmatrix_atom(Float8E4M3FN, False, cfg.a_ldmatrix_matrices), mma)
        cb = cute.make_tiled_copy_B(_make_ldmatrix_atom(Float8E4M3FN, False, cfg.b_ldmatrix_matrices), mma)
        pa, pb = ca.get_slice(thread), cb.get_slice(thread)
        ga, gb = pa.partition_S(qa), pb.partition_S(qb)
        rca, rcb = pa.retile(ra), pb.retile(rb)
        csa = cute.make_tiled_copy(_make_scale_s2r_atom(Float8E8M0FNU, cfg.sfa_s2r_bits),
            sm120_utils.get_layoutSFA_TV(mma), (cute.size(mma.permutation_mnk[0]), cute.size(mma.permutation_mnk[2])))
        csb = cute.make_tiled_copy(_make_scale_s2r_atom(Float8E8M0FNU, cfg.sfb_s2r_bits),
            sm120_utils.get_layoutSFB_TV(mma), (cute.size(mma.permutation_mnk[1]), cute.size(mma.permutation_mnk[2])))
        psa, psb = csa.get_slice(thread), csb.get_slice(thread)
        gsa, gsb = psa.partition_S(sfa), psb.partition_S(sfb)
        rcsa, rcsb = psa.retile(rsa), psb.retile(rsb)
        for kb in cutlass.range_constexpr(cute.size(ra, mode=[2])):
            cute.copy(ca, ga[None, None, kb, stage], rca[None, None, kb])
            cute.copy(cb, gb[None, None, kb, stage], rcb[None, None, kb])
            cute.copy(csa, cute.filter_zeros(gsa)[None, None, kb, stage], cute.filter_zeros(rcsa)[None, None, kb])
            cute.copy(csb, cute.filter_zeros(gsb)[None, None, kb, stage], cute.filter_zeros(rcsb)[None, None, kb])
            cute.gemm(mma, acc, [ra[None, None, kb], rsa[None, None, kb]],
                      [rb[None, None, kb], rsb[None, None, kb]], acc)

    @cute.kernel
    def kernel(self, a, sa, x, mapping, out):
        cfg = self.config
        self._setup_static_layouts()
        thread = cute.arch.thread_idx()[0]
        block, _, expert = cute.arch.block_idx()
        m, n, k = self.problem.m, self.problem.n, self.problem.k
        a = cute.make_tensor(a.iterator + expert*m*k, cute.make_layout((m,k), stride=(k,1)))
        sa = cute.make_tensor(sa.iterator + expert*m*(k//32), cute.make_layout((m,k//32), stride=(k//32,1)))
        x = cute.make_tensor(x.iterator, cute.make_layout((self.x_stage.tokens*n,)))
        mapping = cute.make_tensor(mapping.iterator + expert*k, cute.make_layout((k,)))
        out = cute.make_tensor(out.iterator + expert*m*n, cute.make_layout((m,n), stride=(n,1)))
        if cutlass.const_expr(cfg.raster == 'n'):
            group = block//(self.n_tiles*cfg.grid_swizzle)
            offset = block%(self.n_tiles*cfg.grid_swizzle)
            count = cutlass.min(self.m_tiles-group*cfg.grid_swizzle, cfg.grid_swizzle)
            bm, bn = group*cfg.grid_swizzle+offset%count, offset//count
        else:
            group = block//(self.m_tiles*cfg.grid_swizzle)
            offset = block%(self.m_tiles*cfg.grid_swizzle)
            count = cutlass.min(self.n_tiles-group*cfg.grid_swizzle, cfg.grid_swizzle)
            bm, bn = offset//count, group*cfg.grid_swizzle+offset%count
        allocator = cutlass.utils.SmemAllocator()
        qa = allocator.allocate_tensor(Float8E4M3FN, self.a_layout.outer, byte_alignment=1024, swizzle=self.a_layout.inner)
        qb = allocator.allocate_tensor(Float8E4M3FN, self.b_layout.outer, byte_alignment=1024, swizzle=self.b_layout.inner)
        sfa = allocator.allocate_tensor(Float8E8M0FNU, self.sfa_layout, byte_alignment=128)
        sfb = allocator.allocate_tensor(Float8E8M0FNU, self.sfb_layout, byte_alignment=128)
        mma_threads = cfg.num_mma_warps*32
        coords = self.tiled_mma.get_slice(thread % mma_threads).partition_C(cute.make_identity_tensor((cfg.tile_m, cfg.tile_n)))
        accumulator = cute.make_rmem_tensor(coords.shape, Float32)
        accumulator.fill(Float32(0))
        if cutlass.const_expr(self.producer_warps):
            barriers = allocator.allocate_tensor(Int64, cute.make_layout(2*cfg.stages), byte_alignment=8)
            pipe = pipeline.PipelineAsync.create(num_stages=cfg.stages,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.producer_warps*32),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, mma_threads),
                barrier_storage=barriers.iterator)
            if thread >= mma_threads:
                producer = pipeline.PipelineState(cfg.stages, Int32(0), Int32(0), Int32(1))
                for kt in cutlass.range(self.num_k_tiles, unroll=cfg.k_tile_unroll):
                    pipe.producer_acquire(producer)
                    self._produce(a, sa, x, mapping, qa, qb, sfa, sfb, bm, bn, kt,
                                  producer.index, thread-mma_threads, self.producer_warps)
                    pipe.producer_commit(producer)
                    producer.advance()
                pipe.producer_tail(producer)
            else:
                consumer = pipeline.PipelineState(cfg.stages, Int32(0), Int32(0), Int32(0))
                for kt in cutlass.range(self.num_k_tiles, unroll=cfg.k_tile_unroll):
                    pipe.consumer_wait(consumer, pipe.consumer_try_wait(consumer))
                    self._consume(accumulator, qa, qb, sfa, sfb, consumer.index, thread)
                    pipe.consumer_release(consumer)
                    consumer.advance()
        else:
            for kt in cutlass.range(self.num_k_tiles, unroll=cfg.k_tile_unroll):
                self._produce(a, sa, x, mapping, qa, qb, sfa, sfb, bm, bn, kt,
                              Int32(0), thread, cfg.num_mma_warps)
                cute.arch.sync_threads()
                self._consume(accumulator, qa, qb, sfa, sfb, Int32(0), thread)
                cute.arch.sync_threads()
        if thread < mma_threads:
            for i in cutlass.range_constexpr(cute.size(accumulator)):
                row, col = bm*cfg.tile_m+coords[i][0], bn*cfg.tile_n+coords[i][1]
                if row < m and col < n:
                    out[row, col] = BFloat16(accumulator[i])


@lru_cache(maxsize=128)
def compile_indexed_gate_dw(tokens, experts, problem, gemm_config, quant_config,
                            token_lanes=8, gather_pairs=1, producer_warps=0):
    """Compile an allocation-free launcher; its caller owns all CUDA tensors."""
    def fake(dtype, rows, cols, alignment=16):
        return cute.runtime.make_fake_tensor(dtype, (rows, cols), stride=(cols,1), assumed_align=alignment)

    m, n, k = problem.m, problem.n, problem.k
    return cute.compile(IndexedGateDW(tokens, experts, problem, gemm_config, quant_config,
                                    token_lanes, gather_pairs, producer_warps),
        fake(Float8E4M3FN, experts*m, k), fake(Uint8, experts*m, k//32),
        fake(BFloat16, tokens, n, 2), fake(Int32, experts*k, 1), fake(BFloat16, experts*m, n, 2),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True), options="--enable-tvm-ffi --opt-level 3 --ptxas-options '-O3'")
