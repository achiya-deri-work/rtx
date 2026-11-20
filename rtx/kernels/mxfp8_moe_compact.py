"""Private stable expert-row compaction for the packed forward M axis.

The public route continues to address ``expert * c + physical_row``. Forward
uses a separate dense per-expert stride and original-rank FP32 combine.
None of these maps is saved for backward or used to group transposed scales.
"""

from functools import lru_cache

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Boolean, Float8E4M3FN, Float32, Int32, Int64, Uint8, Uint32


class CompactRoute:
    """One CTA scans an expert's original slots in order without atomics."""

    threads = 256
    items = 4
    warps = threads // 32
    tile = threads * items

    def __init__(self, n, e, c, k, compact_c):
        self.n, self.e, self.c, self.k, self.compact_c = n, e, c, k, compact_c

    @cute.jit
    def __call__(self, ids, inverse, original_to_compact, compact_to_original, counts, stream):
        self.kernel(ids, inverse, original_to_compact, compact_to_original, counts).launch(
            grid=(self.e, 1, 1), block=(self.threads, 1, 1), stream=stream,
        )

    @cute.kernel
    def kernel(self, ids, inverse, original_to_compact, compact_to_original, counts):
        expert = cute.arch.block_idx()[0]
        tid = cute.arch.thread_idx()[0]
        lane = cute.arch.lane_idx()
        warp = tid // 32
        smem = cutlass.utils.SmemAllocator()
        totals = smem.allocate_tensor(Int32, cute.make_layout((self.warps * self.items,)))
        offsets = smem.allocate_tensor(Int32, cute.make_layout((self.warps * self.items,)))
        tile_count = smem.allocate_tensor(Int32, cute.make_layout((1,)))
        total = Int32(0)

        for start in cutlass.range(0, self.c, self.tile, unroll=1):
            local = cute.make_rmem_tensor((self.items,), Int32)
            for item in cutlass.range_constexpr(self.items):
                row = start + item * self.threads + tid
                valid = Boolean(False)
                if row < self.c:
                    token64 = ids[expert, row]
                    if token64 >= 0 and token64 < self.n:
                        token = Int32(token64)
                        logical = expert * self.c + row
                        for rank in cutlass.range_constexpr(self.k):
                            if inverse[token, rank] == logical:
                                valid = Boolean(True)
                ballot = Uint32(cute.arch.vote_ballot_sync(valid))
                if lane == 0:
                    totals[item * self.warps + warp] = Int32(cute.arch.popc(ballot))
                prior_lanes = (Uint32(1) << Uint32(lane)) - Uint32(1)
                local[item] = Int32(cute.arch.popc(ballot & prior_lanes)) if valid else Int32(-1)

            cute.arch.sync_threads()
            if warp == 0:
                value = totals[lane]
                inclusive = value
                for level in cutlass.range_constexpr(5):
                    prior = cute.arch.shuffle_sync_up(inclusive, 1 << level)
                    if lane >= 1 << level:
                        inclusive += prior
                offsets[lane] = inclusive - value
                if lane == 31:
                    tile_count[0] = inclusive
            cute.arch.sync_threads()

            for item in cutlass.range_constexpr(self.items):
                row = start + item * self.threads + tid
                if row < self.c:
                    dense = Int32(-1)
                    if local[item] >= 0:
                        position = total + offsets[item * self.warps + warp] + local[item]
                        # A proven route cannot overflow. Keep all address
                        # arithmetic safe even when private callers violate it.
                        if position < self.compact_c:
                            dense = position
                            compact_to_original[expert, dense] = row
                    original_to_compact[expert, row] = dense
            total += tile_count[0]

        for dense in cutlass.range(total + tid, self.compact_c, self.threads, unroll=1):
            compact_to_original[expert, dense] = Int32(-1)
        if tid == 0:
            counts[expert] = cutlass.min(total, self.compact_c)


class CompactGather:
    def __init__(self, n, d, e, c, compact_c, threads):
        self.n, self.e, self.c, self.compact_c, self.threads = n, e, c, compact_c, threads
        self.width = (d + 31) // 32 * 32
        self.scales = self.width // 32

    @cute.jit
    def __call__(self, global_q, global_sf, ids, compact_to_original, counts, qx, sx, prefix, stream):
        self.kernel(global_q, global_sf, ids, compact_to_original, counts, qx, sx, prefix).launch(
            grid=(cute.ceil_div(self.e * self.compact_c, self.threads // 32), 1, 1),
            block=(self.threads, 1, 1), stream=stream,
        )

    @cute.kernel
    def kernel(self, global_q, global_sf, ids, compact_to_original, counts, qx, sx, prefix):
        tid = cute.arch.thread_idx()[0]
        block = cute.arch.block_idx()[0]
        lane = cute.arch.lane_idx()
        slot = block * (self.threads // 32) + tid // 32

        # Consumer launches follow this launch; no cross-CTA coordination is
        # required in the gather itself. Every row of the compact descriptor,
        # including each consumer's last padded tile, gets defined bytes.
        if block == 0 and tid < 32:
            upper = Int32(0)
            for expert in cutlass.range(lane, self.e, 32, unroll=1):
                upper = cutlass.max(upper, counts[expert])
            upper = cute.arch.warp_reduction_max(upper)
            if lane == 0:
                prefix[0] = upper

        token = Int32(-1)
        if slot < self.e * self.compact_c and lane == 0:
            expert = slot // self.compact_c
            physical = compact_to_original[expert, slot % self.compact_c]
            if physical >= 0 and physical < self.c:
                token64 = ids[expert, physical]
                if token64 >= 0 and token64 < self.n:
                    token = Int32(token64)
        token = cute.arch.shuffle_sync(token, 0)

        if slot < self.e * self.compact_c:
            # Copy four E4M3 bytes at once, without decoding them.
            shape = self.width // 4
            src = cute.make_tensor(cute.recast_ptr(global_q.iterator, dtype=Int32),
                                   cute.make_layout((self.n, shape), stride=(shape, 1)))
            dst = cute.make_tensor(cute.recast_ptr(qx.iterator, dtype=Int32),
                                   cute.make_layout((self.e * self.compact_c, shape), stride=(shape, 1)))
            for word in cutlass.range(lane, shape, 32, unroll=1):
                bits = Int32(0)
                if token >= 0:
                    bits = src[token, word]
                dst[slot, word] = bits
            for scale_col in cutlass.range(lane, self.scales, 32, unroll=1):
                bits = Uint8(0)
                if token >= 0:
                    bits = Uint8(global_sf[token, scale_col])
                sx[slot, scale_col] = bits


class CompactCombine:
    def __init__(self, n, d, e, c, k, compact_c, threads, width, lanes):
        self.n, self.d, self.e, self.c, self.k = n, d, e, c, k
        self.compact_c, self.threads, self.width, self.lanes = compact_c, threads, width, lanes
        self.stripe = lanes * width
        self.stripes = (d + self.stripe - 1) // self.stripe
        self.out_width = min(width, 4)

    @cute.jit
    def __call__(self, source, inverse, weights, original_to_compact, out, stream):
        if cutlass.const_expr(self.width > 1):
            self.vector(source, inverse, weights, original_to_compact, out).launch(
                grid=(cute.ceil_div(self.n * self.stripes, self.threads // self.lanes), 1, 1),
                block=(self.threads, 1, 1), stream=stream,
            )
        else:
            self.scalar(source, inverse, weights, original_to_compact, out).launch(
                grid=(cute.ceil_div(self.n * self.d, self.threads), 1, 1),
                block=(self.threads, 1, 1), stream=stream,
            )

    @cute.jit
    def _translate(self, inverse, weights, original_to_compact, token, rank):
        physical = Int32(-1)
        weight = Float32(0)
        slot64 = inverse[token, rank]
        if slot64 >= 0 and slot64 < self.e * self.c:
            slot = Int32(slot64)
            expert, row = slot // self.c, slot % self.c
            dense = original_to_compact[expert, row]
            if dense >= 0 and dense < self.compact_c:
                physical = expert * self.compact_c + dense
                weight = weights[expert, row]
        return physical, weight

    @cute.kernel
    def scalar(self, source, inverse, weights, original_to_compact, out):
        index = cute.arch.block_idx()[0] * self.threads + cute.arch.thread_idx()[0]
        if index < self.n * self.d:
            token, col = index // self.d, index % self.d
            acc = Float32(0)
            for rank in cutlass.range_constexpr(self.k):
                dense, weight = self._translate(inverse, weights, original_to_compact, token, rank)
                if dense >= 0:
                    acc += Float32(source[dense, col]) * weight
            out[token, col] = acc

    @cute.kernel
    def vector(self, source, inverse, weights, original_to_compact, out):
        tid = cute.arch.thread_idx()[0]
        lane = cute.arch.lane_idx()
        local_lane = lane % self.lanes
        group = cute.arch.block_idx()[0] * (self.threads // self.lanes) + tid // self.lanes
        token = group // self.stripes
        col = group % self.stripes * self.stripe + local_lane * self.width
        physical, local_weight = Int32(-1), Float32(0)
        if token < self.n and local_lane < self.k:
            physical, local_weight = self._translate(inverse, weights, original_to_compact, token, local_lane)

        acc = cute.make_rmem_tensor((self.width,), Float32)
        acc.fill(Float32(0))
        load = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=self.width * 16)
        for rank in cutlass.range_constexpr(self.k):
            leader = lane - local_lane + rank
            dense = cute.arch.shuffle_sync(physical, leader)
            weight = cute.arch.shuffle_sync(local_weight, leader)
            if dense >= 0 and col < self.d:
                values = cute.make_rmem_tensor((self.width,), BFloat16)
                offset = cute.assume(dense * self.d + col, divby=self.width)
                source_tile = cute.make_tensor(source.iterator + offset, cute.make_layout((self.width,)))
                cute.copy(load, source_tile, values)
                for i in cutlass.range_constexpr(self.width):
                    acc[i] += Float32(values[i]) * weight

        if token < self.n and col < self.d:
            store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=self.out_width * 32)
            layout = cute.make_layout((self.out_width,))
            for i in cutlass.range_constexpr(0, self.width, self.out_width):
                offset = cute.assume(token * self.d + col + i, divby=self.out_width)
                src = cute.make_tensor(acc.iterator + i, layout)
                dst = cute.make_tensor(out.iterator + offset, layout)
                cute.copy(store, src, dst)


def _fake(dtype, rows, cols, alignment):
    return cute.runtime.make_fake_tensor(dtype, (rows, cols), stride=(cols, 1), assumed_align=alignment)


def _vector(length):
    return cute.runtime.make_fake_tensor(Int32, (length,), stride=(1,), assumed_align=4)


def _validate(n, e, c, compact_c):
    if any(type(v) is not int or v <= 0 for v in (n, e, c, compact_c)):
        raise ValueError('Compact forward requires positive extents')
    if max(n, e * c, e * compact_c) >= 2**31:
        raise ValueError('Compact row maps require signed int32 addressing')


def _compile(kernel, *operands):
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(kernel, *operands, stream, options='--enable-tvm-ffi --opt-level 3')


@lru_cache(maxsize=128)
def compile_compact_route(n, e, c, k, compact_c):
    """Launch: ids, inverse, original_to_compact, compact_to_original, counts."""
    _validate(n, e, c, compact_c)
    if type(k) is not int or not 0 < k <= e:
        raise ValueError('Invalid route fanout')
    return _compile(CompactRoute(n, e, c, k, compact_c), _fake(Int64, e, c, 8),
                    _fake(Int64, n, k, 8), _fake(Int32, e, c, 4),
                    _fake(Int32, e, compact_c, 4), _vector(e))


@lru_cache(maxsize=128)
def compile_compact_gather(n, d, e, c, compact_c, threads=128):
    """Launch: global_q, global_sf, ids, compact_to_original, counts, qx, sx, prefix."""
    _validate(n, e, c, compact_c)
    if type(d) is not int or d <= 0 or threads not in (64, 128, 256, 512):
        raise ValueError('Invalid compact gather width or movement thread count')
    kernel = CompactGather(n, d, e, c, compact_c, threads)
    return _compile(kernel, _fake(Float8E4M3FN, n, kernel.width, 4), _fake(Uint8, n, kernel.scales, 1),
                    _fake(Int64, e, c, 8), _fake(Int32, e, compact_c, 4), _vector(e),
                    _fake(Float8E4M3FN, e * compact_c, kernel.width, 4),
                    _fake(Uint8, e * compact_c, kernel.scales, 1), _vector(1))


@lru_cache(maxsize=128)
def _compile_combine(n, d, e, c, k, compact_c, threads, width, lanes):
    kernel = CompactCombine(n, d, e, c, k, compact_c, threads, width, lanes)
    return _compile(kernel, _fake(BFloat16, e * compact_c, d, width * 2),
                    _fake(Int64, n, k, 8), _fake(Float32, e, c, 4), _fake(Int32, e, c, 4),
                    _fake(Float32, n, d, kernel.out_width * 4))


@lru_cache(maxsize=128)
def compile_compact_combine(n, d, e, c, k, compact_c, threads=128, engine='scalar',
                            vector_width=4, lanes_per_token=32):
    """Launch: compact BF16 source, original inverse/weights, map, FP32 result."""
    _validate(n, e, c, compact_c)
    if (type(d) is not int or d <= 0 or type(k) is not int or not 0 < k <= e
            or threads not in (64, 128, 256, 512) or engine not in ('scalar', 'token')
            or vector_width not in (2, 4, 8) or lanes_per_token not in (8, 16, 32)):
        raise ValueError('Invalid compact combine geometry or engine')
    if engine == 'scalar' or d % vector_width or k > lanes_per_token:
        return _compile_combine(n, d, e, c, k, compact_c, threads, 1, 1)
    vector = _compile_combine(n, d, e, c, k, compact_c, threads, vector_width, lanes_per_token)

    def launch(source, inverse, weights, original_to_compact, out):
        if source.data_ptr() % (vector_width * 2) or out.data_ptr() % (min(vector_width, 4) * 4):
            scalar = _compile_combine(n, d, e, c, k, compact_c, threads, 1, 1)
            scalar(source, inverse, weights, original_to_compact, out)
        else:
            vector(source, inverse, weights, original_to_compact, out)

    return launch
