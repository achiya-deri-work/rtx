"""Vector token-feature ownership for routed combine and input gradients.

The scalar movement runner is the fallback for geometry or pointers that
cannot support vector access.  This module does not change the buffer ABI.
"""
from functools import lru_cache

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64

from .mxfp8_moe import RoutedMovement


class TokenReduce:
    def __init__(self, n, d, e, c, k, threads, kind, storage_c, vector_width,
                 lanes_per_token):
        self.n, self.d, self.e, self.c, self.k = n, d, e, c, k
        self.threads, self.kind, self.storage_c = threads, kind, storage_c
        self.width, self.lanes = vector_width, lanes_per_token
        self.stripe = self.lanes * self.width
        self.stripes = (d + self.stripe - 1) // self.stripe
        self.out_type = Float32 if kind == 'combine' else BFloat16
        # Universal copy supports at most 128 bits per lane.
        self.out_width = min(self.width, 4 if kind == 'combine' else 8)

    @cute.jit
    def __call__(self, source, ids, inverse, weights, extra, out, auxiliary, stream):
        grid = cute.ceil_div(self.n * self.stripes, self.threads // self.lanes)
        self.kernel(source, inverse, weights, out).launch(
            grid=(grid, 1, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, source, inverse, weights, out):
        thread = cute.arch.thread_idx()[0]
        block = cute.arch.block_idx()[0]
        lane = cute.arch.lane_idx()
        local_lane = lane % self.lanes
        group = block * (self.threads // self.lanes) + thread // self.lanes
        token = group // self.stripes
        col = group % self.stripes * self.stripe + local_lane * self.width

        # One lane per rank loads and translates metadata.  All lanes of
        # each warp (including inactive tail lanes) participate in shuffles.
        physical = Int32(-1)
        route_weight = Float32(0)
        if token < self.n and local_lane < self.k:
            slot = Int32(inverse[token, local_lane])
            if slot >= 0 and slot < self.e * self.c:
                expert, local_slot = slot // self.c, slot % self.c
                physical = expert * self.storage_c + local_slot
                if cutlass.const_expr(self.kind == 'combine'):
                    route_weight = weights[expert, local_slot]

        accum = cute.make_rmem_tensor((self.width,), Float32)
        accum.fill(Float32(0))
        load = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BFloat16,
                                   num_bits_per_copy=self.width * 16)
        for rank in cutlass.range_constexpr(self.k):
            leader = lane - local_lane + rank
            row = cute.arch.shuffle_sync(physical, leader)
            weight = Float32(0)
            if cutlass.const_expr(self.kind == 'combine'):
                weight = cute.arch.shuffle_sync(route_weight, leader)
            if row >= 0 and col < self.d:
                values = cute.make_rmem_tensor((self.width,), BFloat16)
                offset = cute.assume(row * self.d + col, divby=self.width)
                tile = cute.make_tensor(source.iterator + offset, cute.make_layout((self.width,)))
                cute.copy(load, tile, values)
                for i in cutlass.range_constexpr(self.width):
                    contribution = Float32(values[i])
                    if cutlass.const_expr(self.kind == 'combine'):
                        contribution *= weight
                    accum[i] += contribution

        if token < self.n and col < self.d:
            result = cute.make_rmem_tensor((self.width,), self.out_type)
            for i in cutlass.range_constexpr(self.width):
                result[i] = self.out_type(accum[i])
            store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.out_type,
                                       num_bits_per_copy=self.out_width * self.out_type.width)
            layout = cute.make_layout((self.out_width,))
            for i in cutlass.range_constexpr(0, self.width, self.out_width):
                offset = cute.assume(token * self.d + col + i, divby=self.out_width)
                src = cute.make_tensor(result.iterator + i, layout)
                dst = cute.make_tensor(out.iterator + offset, layout)
                cute.copy(store, src, dst)


def _compile(kernel, n, d, e, c, k, storage_c, kind, source_align, out_align):
    def fake(dtype, rows, cols, align):
        return cute.runtime.make_fake_tensor(dtype, (rows, cols), stride=(cols, 1),
                                             assumed_align=align)

    source = fake(BFloat16, e * storage_c, d, source_align)
    out_type = Float32 if kind == 'combine' else BFloat16
    out = fake(out_type, n, d, out_align)
    ids = fake(Int64, e, c, 8)
    inverse = fake(Int64, n, k, 8)
    weights = fake(Float32, e, c, 4)
    # Neither extra nor auxiliary is accessed by either reduction.
    extra = fake(BFloat16, e * storage_c, d, 2)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(kernel, source, ids, inverse, weights, extra, out, weights, stream,
                        options='--enable-tvm-ffi --opt-level 3')


@lru_cache(maxsize=128)
def _compile_scalar(n, d, e, c, k, threads, kind, storage_c):
    # The old kernel itself, compiled with honest alignment for shifted
    # contiguous views (e.g. a BF16 source one element into its allocation).
    scalar = RoutedMovement(n, d, e, c, k, threads, kind, storage_c=storage_c)
    return _compile(scalar, n, d, e, c, k, storage_c, kind, 2,
                    4 if kind == 'combine' else 2)


@lru_cache(maxsize=128)
def compile_token_reduce(n, d, e, c, k, threads, kind, storage_c=0,
                         vector_width=4, lanes_per_token=32):
    if kind not in ('combine', 'dx') or threads not in (64, 128, 256, 512):
        raise ValueError('Invalid token reduction mode or thread count')
    if vector_width not in (2, 4, 8) or lanes_per_token not in (8, 16, 32):
        raise ValueError('Invalid token reduction vector width or ownership')
    cs = storage_c or c
    if min(n, d, e, c, k) <= 0 or cs < c:
        raise ValueError('Invalid token reduction geometry')
    if d % vector_width or k > lanes_per_token:
        return _compile_scalar(n, d, e, c, k, threads, kind, cs)

    kernel = TokenReduce(n, d, e, c, k, threads, kind, cs, vector_width, lanes_per_token)
    source_align = vector_width * 2
    out_align = kernel.out_width * (4 if kind == 'combine' else 2)
    vector = _compile(kernel, n, d, e, c, k, cs, kind, source_align, out_align)

    def launch(source, ids, inverse, weights, extra, out, auxiliary):
        if source.data_ptr() % source_align or out.data_ptr() % out_align:
            scalar = _compile_scalar(n, d, e, c, k, threads, kind, cs)
            scalar(source, ids, inverse, weights, extra, out, auxiliary)
        else:
            vector(source, ids, inverse, weights, extra, out, auxiliary)

    return launch
