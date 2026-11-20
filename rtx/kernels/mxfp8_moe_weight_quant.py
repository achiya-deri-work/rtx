"""Dynamic MXFP8 backward prep with one BF16 read per weight tile.

The two output axes still compute independent groups of 32. Only the GU
row output is interleaved; its transpose runs over the original two halves.
Original-token X retains the unmodified row quantizer in the same launch.
"""
from dataclasses import replace
from functools import lru_cache

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float8E4M3FN, Float8E8M0FNU, Int32, Uint8

from ..configs.moe_backward import backward_weight_quant_rejection
from .mxfp8_quant import MXFP8TransposedQuantKernel


def _width(k):
    return (k + 31) // 32 * 32


class BackwardWeightQuant(MXFP8TransposedQuantKernel):
    def __init__(self, n, d, e, h, quant, grid_mode, tile_rows, tile_cols,
                 load_values, store_bits, weight_warps, raster):
        # The staged transpose's independent source-grid contract does not
        # apply; it supplies only the mathematical shared-memory emitter.
        from .mxfp8_moe_prepared import ExpertQuantOperand

        self.config = replace(quant, num_warps=weight_warps, quant_store_bits=store_bits)
        self.x_operand = ExpertQuantOperand(n, d, quant, 'row', False)
        self.x_operand.source_alignment = 2
        self.x_warps = quant.num_warps
        self.threads = 32 * max(quant.num_warps, weight_warps)
        self.tile_rows, self.tile_cols, self.load_values = tile_rows, tile_cols, load_values
        self.raster = raster
        self.e, self.d, self.h = e, d, h
        sm = cutlass.utils.HardwareInfo().get_device_multiprocessor_count()
        budget = sm * quant.persistent_waves
        source_elements = n * d + 2 * e * (2 * h * d + d * h)
        if grid_mode == 'portfolio':
            # Keep the exact baseline singleton-X grid; its proportional
            # budget was computed with two orientations per weight matrix.
            self.x_operand.grid_ctas = min(self.x_operand.grid_ctas, cute.ceil_div(budget * n * d, source_elements))
        jobs = []
        for rows, cols in ((2 * h, d), (d, h)):
            tasks = cute.ceil_div(rows, tile_rows) * cute.ceil_div(cols, tile_cols)
            cap = budget
            if grid_mode == 'expert':
                cap = cute.ceil_div(budget, e)
            elif grid_mode == 'portfolio':
                cap = cute.ceil_div(budget * 2 * rows * cols, source_elements)
            jobs.append(min(cap, tasks))
        self.gu_ctas, self.down_ctas = jobs
        self.gu_start = self.x_operand.grid_ctas
        self.down_start = self.gu_start + self.gu_ctas * e
        self.grid = self.down_start + self.down_ctas * e

    @cute.jit
    def __call__(self, x, gu, down, qx, sx, qgu, sgu, qd, sd, qdt, sdt, qgut, sgut, stream):
        self.kernel(x, gu, down, qx, self._scale(sx), qgu, self._scale(sgu),
                    qd, self._scale(sd), qdt, self._scale(sdt), qgut, self._scale(sgut)).launch(
                        grid=(self.grid, 1, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.jit
    def _scale(self, s):
        return cute.make_tensor(cute.recast_ptr(s.iterator, dtype=Float8E8M0FNU), s.layout)

    @cute.jit
    def _mat(self, out, expert, rows: cutlass.Constexpr, cols: cutlass.Constexpr):
        return cute.make_tensor(out.iterator + expert * rows * cols,
                                cute.make_layout((rows, cols), stride=(cols, 1)))

    @cute.jit
    def _copy(self, src, tile, rb, cb, tid, rows: cutlass.Constexpr, cols: cutlass.Constexpr):
        vec = self.load_values
        for task in cutlass.range(tid, self.tile_rows * self.tile_cols // vec,
                                  self.config.num_warps * 32, unroll=1):
            offset = task * vec
            r, c = offset // self.tile_cols, offset % self.tile_cols
            gr, gc = rb + r, cb + c
            if cutlass.const_expr(vec > 1):
                gptr = src.iterator + gr * cols + gc
                valid = gptr.toint() % (2 * vec) == 0
                if cutlass.const_expr(rows % self.tile_rows):
                    valid = valid and gr < rows
                if cutlass.const_expr(cols % self.tile_cols):
                    valid = valid and gc + vec <= cols
                if valid:
                    cute.arch.cp_async_shared_global(tile.iterator + tile.layout((r, c)), gptr, 2 * vec,
                                                    'cg' if cutlass.const_expr(vec == 8) else 'ca')
                else:
                    for i in cutlass.range_constexpr(vec):
                        value = BFloat16(0.0)
                        if gr < rows and gc + i < cols:
                            value = src[gr, gc + i]
                        tile[r, c + i] = value
            else:
                value = BFloat16(0.0)
                if gr < rows and gc < cols:
                    value = src[gr, gc]
                tile[r, c] = value
        if cutlass.const_expr(vec > 1):
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)

    @cute.jit
    def _emit(self, tile, q, s, rb, kb, lane, warp, rows: cutlass.Constexpr,
              width: cutlass.Constexpr, tr: cutlass.Constexpr, tc: cutlass.Constexpr,
              interleave: cutlass.Constexpr):
        vec = self.config.quant_vec
        block_lanes = 32 // vec
        blocks_per_row = tc // 32
        for group in cutlass.range(warp, tr * blocks_per_row // vec, self.config.num_warps, unroll=1):
            local = group * vec + lane // block_lanes
            lr, lb = local // blocks_per_row, local % blocks_per_row
            gr, gb = rb + lr, kb + lb
            row = gr
            if cutlass.const_expr(interleave):
                row = (gr % (rows // 2)) * 2 + gr // (rows // 2)
            self._quantize_smem_row(tile, q, s, lr, row, gb, lb, lane % block_lanes,
                                    0, None, True, gr < rows and gb < width // 32)

    @cute.jit
    def _dual(self, src, q0, s0, q1, s1, tile, transposed, block, expert, tid, lane, warp,
              rows: cutlass.Constexpr, cols: cutlass.Constexpr, ctas: cutlass.Constexpr,
              interleave: cutlass.Constexpr):
        src = self._mat(src, expert, rows, cols)
        w0, w1 = _width(cols), _width(rows)
        qrow, srow = self._mat(q0, expert, rows, w0), self._mat(s0, expert, rows, w0 // 32)
        qcol, scol = self._mat(q1, expert, cols, w1), self._mat(s1, expert, cols, w1 // 32)
        tr, tc = self.tile_rows, self.tile_cols
        rt, ct = cute.ceil_div(rows, tr), cute.ceil_div(cols, tc)
        for task in cutlass.range(block, rt * ct, ctas, unroll=1):
            if cutlass.const_expr(self.raster == 'features'):
                r, c = task // ct, task % ct
            else:
                r, c = task % rt, task // rt
            rb, cb = r * tr, c * tc
            if warp < self.config.num_warps:
                self._copy(src, tile, rb, cb, tid, rows, cols)
            cute.arch.sync_threads()
            if warp < self.config.num_warps:
                self._emit(tile, qrow, srow, rb, cb // 32, lane, warp, rows, w0, tr, tc, interleave)
                self._emit(transposed, qcol, scol, cb, rb // 32, lane, warp, cols, w1, tc, tr, False)
            # No subsequent shared writes can race an emitter on a prior tile.
            if cutlass.const_expr(ctas < rt * ct):
                cute.arch.sync_threads()

    @cute.kernel
    def kernel(self, x, gu, down, qx, sx, qgu, sgu, qd, sd, qdt, sdt, qgut, sgut):
        block, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        lane = cute.arch.lane_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tr, tc = self.tile_rows, self.tile_cols
        # XOR within the tile; the bottom three column bits are unchanged,
        # so every 4/8/16-byte cp.async destination remains aligned.
        log_cols = tc.bit_length() - 1
        swizzle = cute.make_swizzle(min(3, log_cols - 3), 3, log_cols - 3)
        allocator = cutlass.utils.SmemAllocator()
        tile = allocator.allocate_tensor(BFloat16, cute.make_layout((tr * tc,)), byte_alignment=16)
        transposed = cute.make_tensor(tile.iterator, cute.make_composed_layout(
            swizzle, 0, cute.make_layout((tc, tr), stride=(1, tc))))
        tile = cute.make_tensor(tile.iterator, cute.make_composed_layout(
            swizzle, 0, cute.make_layout((tr, tc), stride=(tc, 1))))
        if block < self.gu_start:
            if warp < self.x_warps:
                self.x_operand.run(x, qx, sx, block, Int32(0))
        elif block < self.down_start:
            local = block - self.gu_start
            self._dual(gu, qgu, sgu, qgut, sgut, tile, transposed, local % self.gu_ctas,
                       local // self.gu_ctas, tid, lane, warp, 2 * self.h, self.d, self.gu_ctas, True)
        else:
            local = block - self.down_start
            self._dual(down, qd, sd, qdt, sdt, tile, transposed, local % self.down_ctas,
                       local // self.down_ctas, tid, lane, warp, self.d, self.h, self.down_ctas, False)


@lru_cache(maxsize=128)
def compile_backward_weight_quant(n, d, e, h, quant, grid_mode='operand', tile_rows=32,
                                  tile_cols=64, load_values=4, store_bits=32,
                                  weight_warps=4, raster='features'):
    """ABI: X/GU/down then (FP8, uint8 scale) for X, GU, down, down.T, GU.T."""
    dims = (n, d, e, h, quant, grid_mode, tile_rows, tile_cols, load_values, store_bits, weight_warps, raster)
    reason = backward_weight_quant_rejection(*dims)
    if reason:
        raise ValueError(reason)

    def fake(dtype, rows, cols, align=16):
        return cute.runtime.make_fake_tensor(dtype, (rows, cols), stride=(cols, 1), assumed_align=align)

    outs = tuple(tensor for rows, cols in ((n, d), (e * 2 * h, d), (e * d, h),
                                          (e * h, d), (e * d, 2 * h))
                 for tensor in (fake(Float8E4M3FN, rows, _width(cols)), fake(Uint8, rows, _width(cols) // 32)))
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(BackwardWeightQuant(*dims), fake(BFloat16, n, d, 2),
                        fake(BFloat16, e * 2 * h, d, 2), fake(BFloat16, e * d, h, 2), *outs, stream,
                        options='--enable-tvm-ffi --opt-level 3')
