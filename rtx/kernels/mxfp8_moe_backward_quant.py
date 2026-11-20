"""Experimental single-CTA portfolio for routed dY and indexed X transpose.

One tile's slot metadata is loaded once. The BF16 tile is reused for X after
both weighted-dY orientations are emitted, so no extra BF16 GMEM or second
large SMEM tile is needed. The ordinary fused-dY and warp-XT paths remain
independent; only a whole-path tuner may prefer this shared implementation.
"""
from functools import lru_cache

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float8E4M3FN, Float8E8M0FNU, Float32, Int32, Uint8, Uint16
from cutlass.experimental.primitives import nvvm_wrapper as nvvm

from .mxfp8_moe_dy import RoutedDYQuant
from .mxfp8_moe_input import _fake


class RoutedBackwardQuant(RoutedDYQuant):
    @cute.jit
    def __call__(self, dy, mapping, weights, qdy, sdy, qdyt, sdyt, x, qxt, sxt, stream):
        sdy = cute.make_tensor(cute.recast_ptr(sdy.iterator, dtype=Float8E8M0FNU), sdy.layout)
        sdyt = cute.make_tensor(cute.recast_ptr(sdyt.iterator, dtype=Float8E8M0FNU), sdyt.layout)
        sxt = cute.make_tensor(cute.recast_ptr(sxt.iterator, dtype=Float8E8M0FNU), sxt.layout)
        self.kernel_all(dy, mapping, weights, qdy, sdy, qdyt, sdyt, x, qxt, sxt).launch(
            grid=(self.grid_ctas, 1, self.e), block=(self.config.num_warps * 32, 1, 1), stream=stream,
        )

    @cute.jit
    def _load_operand(self, source, mapping, weights, scratch, smem, expert, slot_base, feature_base,
                      lane, warp, is_dy: cutlass.Constexpr):
        cfg = self.config
        for row in cutlass.range(warp, cfg.transposed_tile_k, cfg.num_warps, unroll=1):
            token = Int32(-1)
            weight = Float32(0.0)
            slot = slot_base + row
            if lane == 0:
                if cutlass.const_expr(is_dy):
                    if slot < self.c:
                        token = mapping[expert * self.storage_c + slot, 0]
                        if token >= 0 and token < self.n:
                            weight = weights[expert, slot]
                    scratch[row] = token
                else:
                    token = scratch[row]
            token = cute.arch.shuffle_sync(token, 0)
            if cutlass.const_expr(is_dy):
                weight = cute.arch.shuffle_sync(weight, 0)
            for col_group in cutlass.range_constexpr(cfg.transposed_tile_rows // (32 * self.load_values)):
                col = (col_group * 32 + lane) * self.load_values
                feature = feature_base + col
                values = [BFloat16(0.0)] * self.load_values
                if token >= 0 and token < self.n:
                    if cutlass.const_expr(self.load_values > 1):
                        pointer = source.iterator + token * self.d + feature
                        if feature + self.load_values <= self.d and pointer.toint() % (self.load_values * (4 if is_dy else 2)) == 0:
                            if cutlass.const_expr(is_dy):
                                loaded = nvvm.load_ext(pointer, dtype=Float32, count=self.load_values)
                            else:
                                loaded = nvvm.load_ext(pointer, dtype=Uint16, count=self.load_values).bitcast(BFloat16)
                            for v in cutlass.range_constexpr(self.load_values):
                                if cutlass.const_expr(is_dy):
                                    values[v] = BFloat16(loaded[v] * weight)
                                else:
                                    values[v] = loaded[v]
                        else:
                            for v in cutlass.range_constexpr(self.load_values):
                                if feature + v < self.d:
                                    if cutlass.const_expr(is_dy):
                                        values[v] = BFloat16(source[token, feature + v] * weight)
                                    else:
                                        values[v] = source[token, feature + v]
                    elif feature < self.d:
                        if cutlass.const_expr(is_dy):
                            values[0] = BFloat16(source[token, feature] * weight)
                        else:
                            values[0] = source[token, feature]
                for v in cutlass.range_constexpr(self.load_values):
                    smem[row, col + v] = values[v]

    @cute.kernel
    def kernel_all(self, dy, mapping, weights, qdy, sdy, qdyt, sdyt, x, qxt, sxt):
        cfg = self.config
        slots, features = cfg.transposed_tile_k, cfg.transposed_tile_rows
        pitch = features + cfg.transposed_smem_padding
        allocator = cutlass.utils.SmemAllocator()
        shared = allocator.allocate_tensor(BFloat16, cute.make_layout((slots, features), stride=(pitch, 1)),
                                           byte_alignment=16)
        slot_tokens = allocator.allocate_tensor(Int32, cute.make_layout(slots), byte_alignment=16)
        transposed = cute.make_tensor(shared.iterator, cute.make_layout((features, slots), stride=(1, pitch)))
        lane = cute.arch.lane_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        block, _, expert = cute.arch.block_idx()
        qrow, srow = self._output(qdy, sdy, expert, self.storage_c, self.storage_k)
        qcol, scol = self._output(qdyt, sdyt, expert, self.d, self.storage_c)
        qxcol, sxcol = self._output(qxt, sxt, expert, self.d, self.storage_c)
        for tile in cutlass.range(block, self.task_groups, self.grid_ctas, unroll=1):
            if cutlass.const_expr(self.raster == 'features'):
                slot_tile, feature_tile = tile // self.feature_tiles, tile % self.feature_tiles
            else:
                slot_tile, feature_tile = tile % self.slot_tiles, tile // self.slot_tiles
            slot_base, feature_base = slot_tile * slots, feature_tile * features
            self._load_operand(dy, mapping, weights, slot_tokens, shared, expert, slot_base, feature_base,
                               lane, warp, True)
            cute.arch.sync_threads()
            self._emit_tile(shared, qrow, srow, slot_base, feature_base // 32, lane, warp,
                            self.storage_c, self.storage_k, slots, features // 32)
            self._emit_tile(transposed, qcol, scol, feature_base, slot_base // 32, lane, warp,
                            self.d, self.storage_c, features, slots // 32)
            cute.arch.sync_threads()
            self._load_operand(x, mapping, weights, slot_tokens, shared, expert, slot_base, feature_base,
                               lane, warp, False)
            cute.arch.sync_threads()
            self._emit_tile(transposed, qxcol, sxcol, feature_base, slot_base // 32, lane, warp,
                            self.d, self.storage_c, features, slots // 32)
            cute.arch.sync_threads()


@lru_cache(maxsize=128)
def compile_routed_backward_quant(n, d, e, c, storage_c, config, token_lanes=8,
                                  raster='features', grid_mode='operand'):
    """ABI: dY, map, weights, qdY, sdY, qdYT, sdYT, X, qXT, sXT."""
    if token_lanes not in (2, 4, 8):
        raise ValueError('Invalid portfolio warp token lane setting')
    kernel = RoutedBackwardQuant(n, d, e, c, storage_c, config, raster, grid_mode)
    width = kernel.storage_k
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(kernel, _fake(Float32, n, d, alignment=4), _fake(Int32, e * storage_c, 1),
                        _fake(Float32, e, c), _fake(Float8E4M3FN, e * storage_c, width),
                        _fake(Uint8, e * storage_c, width // 32), _fake(Float8E4M3FN, e * d, storage_c),
                        _fake(Uint8, e * d, storage_c // 32), _fake(BFloat16, n, d, alignment=2),
                        _fake(Float8E4M3FN, e * d, storage_c), _fake(Uint8, e * d, storage_c // 32), stream,
                        options='--enable-tvm-ffi --opt-level 3 '
                                f"--ptxas-options '-O3 --maxrregcount={config.maxrregcount}'")
