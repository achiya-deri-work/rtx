"""Device bounds and reduction-tail initialization for packed experts.

The inverse route names *real* assignments only. The largest expert-local
index, rounded to a quantization-group boundary, bounds every real row
without relying on route weights being nonzero or on any host readback.
"""
from functools import lru_cache

import cutlass
import cutlass.cute as cute
from cutlass import Float8E4M3FN, Int32, Int64, Uint8, Uint32


class ActivePrefix:
    def __init__(self, n, e, c, k):
        self.n, self.e, self.c, self.k = n, e, c, k
        self.threads = 256

    @cute.jit
    def __call__(self, inverse, prefix, stream):
        self.kernel(inverse, prefix).launch(
            grid=(1, 1, 1), block=(self.threads, 1, 1), stream=stream,
        )

    @cute.kernel
    def kernel(self, inverse, prefix):
        thread = cute.arch.thread_idx()[0]
        lane = cute.arch.lane_idx()
        maxima = cutlass.utils.SmemAllocator().allocate_tensor(
            Int32, cute.make_layout(self.threads // 32), byte_alignment=4,
        )
        upper = Int32(0)
        flat = cute.make_tensor(inverse.iterator, cute.make_layout(self.n * self.k))
        for i in cutlass.range(thread, self.n * self.k, self.threads, unroll=1):
            slot = Int32(flat[i])
            if slot >= 0 and slot < self.e * self.c:
                upper = cutlass.max(upper, slot % self.c + 1)
        upper = cute.arch.warp_reduction_max(upper)
        if lane == 0:
            maxima[thread // 32] = upper
        cute.arch.sync_threads()
        if thread < 32:
            upper = Int32(0)
            if lane < self.threads // 32:
                upper = maxima[lane]
            upper = cute.arch.warp_reduction_max(upper)
            if lane == 0:
                prefix[0] = cutlass.min(self.c, cute.ceil_div(upper, 32) * 32)


class InitializeReductionTail:
    """Initialize only the skipped recompute rows a final TMA tile can read."""

    def __init__(self, e, c, h, recompute_alignment, down_tile_k, gate_tile_k):
        self.e, self.c, self.h = e, c, h
        self.recompute_alignment = recompute_alignment
        self.down_tile_k, self.gate_tile_k = down_tile_k, gate_tile_k
        self.threads = 128
        self.rows_per_cta = 64

    @cute.jit
    def __call__(self, prefix, qht, sht, qpt, spt, stream):
        self.kernel(prefix, qht, sht, qpt, spt).launch(
            grid=(cute.ceil_div(self.h, self.rows_per_cta) +
                  cute.ceil_div(2 * self.h, self.rows_per_cta), 1, self.e),
            block=(self.threads, 1, 1), stream=stream,
        )

    @cute.jit
    def _zero(self, data, scales, row, rows, lo, hi, thread):
        # All bounds are multiples of 32. Four adjacent E4M3 positive zeros
        # share a vector store. E8M0 code 0 is the quantizer's all-zero scale.
        data = cute.make_tensor(
            cute.recast_ptr(data.iterator, dtype=Uint32),
            cute.make_layout(cute.size(data) // 4),
        )
        if lo < hi:
            words = (hi - lo) // 4
            for index in cutlass.range(thread, rows * words, self.threads, unroll=1):
                r, col = row + index // words, lo // 4 + index % words
                data[r * (self.c // 4) + col] = Uint32(0)
                if col % 8 == 0:
                    scales[r, col // 8] = Uint8(0)

    @cute.kernel
    def kernel(self, prefix, qht, sht, qpt, spt):
        cta, _, expert = cute.arch.block_idx()
        thread = cute.arch.thread_idx()[0]
        hidden_ctas = cute.ceil_div(self.h, self.rows_per_cta)
        active = prefix[0]
        lo = cute.ceil_div(active, self.recompute_alignment) * self.recompute_alignment
        if cta < hidden_ctas:
            row = cta * self.rows_per_cta
            hi = cutlass.min(self.c, cute.ceil_div(active, self.down_tile_k) * self.down_tile_k)
            self._zero(qht, sht, expert * self.h + row,
                       cutlass.min(self.rows_per_cta, self.h - row), lo, hi, thread)
        else:
            row = (cta - hidden_ctas) * self.rows_per_cta
            hi = cutlass.min(self.c, cute.ceil_div(active, self.gate_tile_k) * self.gate_tile_k)
            self._zero(qpt, spt, expert * 2 * self.h + row,
                       cutlass.min(self.rows_per_cta, 2 * self.h - row), lo, hi, thread)


def fake_prefix():
    return cute.runtime.make_fake_tensor(Int32, (1,), stride=(1,), assumed_align=4)


def _fake_matrix(dtype, rows, cols):
    return cute.runtime.make_fake_tensor(dtype, (rows, cols), stride=(cols, 1), assumed_align=16)


@lru_cache(maxsize=128)
def compile_active_prefix(n, e, c, k):
    inverse = _fake_matrix(Int64, n, k)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        ActivePrefix(n, e, c, k), inverse, fake_prefix(), stream,
        options='--enable-tvm-ffi --opt-level 3',
    )


@lru_cache(maxsize=128)
def compile_initialize_reduction_tail(e, c, h, recompute_alignment, down_tile_k, gate_tile_k):
    if any(x % 32 for x in (c, recompute_alignment, down_tile_k, gate_tile_k)):
        raise ValueError('Active reduction tails require complete MXFP8 scale blocks')
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    operands = tuple(
        v for rows in (e * h, e * 2 * h) for v in (
            _fake_matrix(Float8E4M3FN, rows, c),
            _fake_matrix(Uint8, rows, c // 32),
        )
    )
    return cute.compile(
        InitializeReductionTail(e, c, h, recompute_alignment, down_tile_k, gate_tile_k),
        fake_prefix(), *operands, stream, options='--enable-tvm-ffi --opt-level 3',
    )
