"""Route and quantize both weighted-dY orientations from one BF16 SMEM tile.

No expert BF16 matrix is written to global memory. The shared tile holds the
old FP32 product rounded to BF16; each orientation computes its own amax and
E8M0 scale over exactly 32 elements. The compact slot map is the one already
produced by packed input gather, with -1 for dummy and physical gap slots.
"""
from functools import lru_cache

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float8E4M3FN, Float8E8M0FNU, Float32, Int32, Uint8
from cutlass.experimental.primitives import nvvm_wrapper as nvvm

from .mxfp8_moe_input import _fake, rounded_prefix
from .mxfp8_quant import MXFP8QuantKernel, MXFP8TransposedQuantKernel


class RoutedDYQuant(MXFP8TransposedQuantKernel):
    def __init__(self, n, d, e, c, storage_c, config, raster, grid_mode,
                 active_m_alignment=0, transpose_tile_k=0, row_tile_m=1):
        # The emitter can mask feature/tile tails. This is not a conventional
        # transposed input matrix, so the staged-transpose load restrictions
        # (e.g. complete tiles and cp.async compatibility) do not apply.
        if min(n, d, e, c) <= 0 or storage_c < c or storage_c % 32:
            raise ValueError('Routed dY needs positive extents and complete 32-slot blocks')
        reason = config.rejection(d, storage_c)
        if reason:
            raise ValueError(f'Illegal routed dY transpose quantizer: {reason}')
        if config.scale_layout not in ('row_major', 'mma128') or config.native_scale_store != 'scalar':
            raise ValueError('Routed dY supports row-major or native128 scalar scales')
        if raster not in ('features', 'tokens') or grid_mode not in ('operand', 'expert', 'portfolio'):
            raise ValueError('Illegal routed dY tile raster or grid policy')
        if (type(active_m_alignment) is not int or active_m_alignment < 0
                or type(transpose_tile_k) is not int or type(row_tile_m) is not int):
            raise ValueError('Invalid routed dY consumer alignment')
        if active_m_alignment:
            if (config.scale_layout != 'row_major' or min(transpose_tile_k,row_tile_m) < 32
                    or any(tile % 32 for tile in (active_m_alignment, transpose_tile_k, row_tile_m))):
                raise ValueError('Bounded routed dY requires complete row-major scale/consumer tiles')
        elif transpose_tile_k or row_tile_m != 1:
            raise ValueError('Routed dY consumer tiles require an active M alignment')
        MXFP8QuantKernel.__init__(self, storage_c, d, config)
        self.active_m_alignment = active_m_alignment
        self.transpose_tile_k, self.row_tile_m = transpose_tile_k, row_tile_m
        self.n, self.d, self.e, self.c, self.storage_c = n, d, e, c, storage_c
        self.feature_tiles = cute.ceil_div(d, config.transposed_tile_rows)
        self.slot_tiles = cute.ceil_div(storage_c, config.transposed_tile_k)
        self.task_groups = self.feature_tiles * self.slot_tiles
        self.raster = raster
        self.load_values = min(4, config.load_bits // 16, config.transposed_tile_rows // 32)
        sm = cutlass.utils.HardwareInfo().get_device_multiprocessor_count()
        budget = sm * config.persistent_waves
        if grid_mode != 'operand':
            budget = cute.ceil_div(budget, e)
        self.grid_ctas = min(self.task_groups, budget)

    @cute.jit
    def __call__(self, dy, mapping, weights, qdy, sdy, qdyt, sdyt, stream, active_capacity=None):
        sdy = cute.make_tensor(cute.recast_ptr(sdy.iterator, dtype=Float8E8M0FNU), sdy.layout)
        sdyt = cute.make_tensor(cute.recast_ptr(sdyt.iterator, dtype=Float8E8M0FNU), sdyt.layout)
        self.kernel(dy, mapping, weights, qdy, sdy, qdyt, sdyt, active_capacity).launch(
            grid=(self.grid_ctas, 1, self.e), block=(self.config.num_warps * 32, 1, 1), stream=stream,
        )

    @cute.jit
    def _load_tile(self, dy, mapping, weights, smem, expert, slot_base, feature_base, lane, warp,
                   active):
        cfg = self.config
        for row in cutlass.range(warp, cfg.transposed_tile_k, cfg.num_warps, unroll=1):
            slot = slot_base + row
            token = Int32(-1)
            weight = Float32(0.0)
            if lane == 0 and slot < self.c and slot < active:
                token = mapping[expert * self.storage_c + slot, 0]
                if token >= 0 and token < self.n:
                    weight = weights[expert, slot]
            token = cute.arch.shuffle_sync(token, 0)
            weight = cute.arch.shuffle_sync(weight, 0)
            for col_group in cutlass.range_constexpr(cfg.transposed_tile_rows // (32 * self.load_values)):
                col = (col_group * 32 + lane) * self.load_values
                feature = feature_base + col
                values = [BFloat16(0.0)] * self.load_values
                if token >= 0 and token < self.n:
                    if cutlass.const_expr(self.load_values > 1):
                        pointer = dy.iterator + token * self.d + feature
                        if feature + self.load_values <= self.d and pointer.toint() % (self.load_values * 4) == 0:
                            loaded = nvvm.load_ext(pointer, dtype=Float32, count=self.load_values)
                            for v in cutlass.range_constexpr(self.load_values):
                                values[v] = BFloat16(loaded[v] * weight)
                        else:
                            for v in cutlass.range_constexpr(self.load_values):
                                if feature + v < self.d:
                                    values[v] = BFloat16(dy[token, feature + v] * weight)
                    elif feature < self.d:
                        values[0] = BFloat16(dy[token, feature] * weight)
                for v in cutlass.range_constexpr(self.load_values):
                    smem[row, col + v] = values[v]

    @cute.jit
    def _output(self, q, s, expert, rows: cutlass.Constexpr, width: cutlass.Constexpr):
        q = cute.make_tensor(q.iterator + expert * rows * width,
                             cute.make_layout((rows, width), stride=(width, 1)))
        s = cute.make_tensor(s.iterator + expert * rows * (width // 32),
                             cute.make_layout((rows, width // 32), stride=(width // 32, 1)))
        if cutlass.const_expr(self.scale_tile_rows == 128):
            s = cute.make_tensor(s.iterator, cute.make_layout((rows // 128, width // 128, 512),
                                 stride=(width // 128 * 512, 512, 1)))
        return q, s

    @cute.jit
    def _emit_tile(self, smem, q, s, row_base, block_base, lane, warp,
                   rows: cutlass.Constexpr, width: cutlass.Constexpr,
                   tile_rows: cutlass.Constexpr, tile_blocks: cutlass.Constexpr,
                   active_rows=None, active_blocks=None):
        cfg = self.config
        lanes_per_block = 32 // cfg.quant_vec
        block_in_warp, lane_in_block = lane // lanes_per_block, lane % lanes_per_block
        row_extent, block_extent = rows, width // 32
        if cutlass.const_expr(active_rows is not None):
            row_extent = active_rows
        if cutlass.const_expr(active_blocks is not None):
            block_extent = active_blocks
        for group in cutlass.range(warp, cute.ceil_div(tile_rows * tile_blocks, cfg.quant_vec),
                                   cfg.num_warps, unroll=1):
            local = group * cfg.quant_vec + block_in_warp
            local_row, local_block = local // tile_blocks, local % tile_blocks
            row, block = row_base + local_row, block_base + local_block
            # Complete shared tiles are safe to quantize even when some output
            # subwarps are padded. Keep all lanes in the shuffle, mask stores.
            self._quantize_smem_row(smem, q, s, local_row, row, block, local_block,
                                    lane_in_block, self.scale_tile_rows, None, True,
                                    row < row_extent and block < block_extent)

    @cute.kernel
    def kernel(self, dy, mapping, weights, qdy, sdy, qdyt, sdyt, active_capacity):
        cfg = self.config
        slots, features = cfg.transposed_tile_k, cfg.transposed_tile_rows
        pitch = features + cfg.transposed_smem_padding
        allocator = cutlass.utils.SmemAllocator()
        shared = allocator.allocate_tensor(BFloat16, cute.make_layout((slots, features), stride=(pitch, 1)),
                                           byte_alignment=16)
        transposed = cute.make_tensor(shared.iterator, cute.make_layout((features, slots), stride=(1, pitch)))
        lane = cute.arch.lane_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        block, _, expert = cute.arch.block_idx()
        qrow, srow = self._output(qdy, sdy, expert, self.storage_c, self.storage_k)
        qcol, scol = self._output(qdyt, sdyt, expert, self.d, self.storage_c)
        if cutlass.const_expr(self.active_m_alignment):
            active = active_capacity[0]
            row_extent = rounded_prefix(rounded_prefix(active, self.storage_c, self.active_m_alignment),
                                        self.storage_c, self.row_tile_m)
            column_extent = rounded_prefix(active, self.storage_c, self.transpose_tile_k)
            slot_tiles = cute.ceil_div(cutlass.max(row_extent, column_extent), slots)
        else:
            active = self.c
            row_extent = self.storage_c
            column_extent = self.storage_c
            slot_tiles = self.slot_tiles
        # A tile and its barriers are owned by the entire CTA. Both
        # rasters enumerate the same compact work interval, including when
        # it is zero, while output tensors retain their original pitches.
        for tile in cutlass.range(block, slot_tiles * self.feature_tiles, self.grid_ctas, unroll=1):
            if cutlass.const_expr(self.raster == 'features'):
                slot_tile, feature_tile = tile // self.feature_tiles, tile % self.feature_tiles
            else:
                slot_tile, feature_tile = tile % slot_tiles, tile // slot_tiles
            slot_base, feature_base = slot_tile * slots, feature_tile * features
            self._load_tile(dy, mapping, weights, shared, expert, slot_base, feature_base, lane, warp,
                            active)
            cute.arch.sync_threads()
            if cutlass.const_expr(self.active_m_alignment):
                if slot_base < row_extent:
                    self._emit_tile(shared, qrow, srow, slot_base, feature_base // 32, lane, warp,
                                    self.storage_c, self.storage_k, slots, features // 32,
                                    active_rows=row_extent)
                if slot_base < column_extent:
                    self._emit_tile(transposed, qcol, scol, feature_base, slot_base // 32, lane, warp,
                                    self.d, self.storage_c, features, slots // 32,
                                    active_blocks=column_extent // 32)
            else:
                self._emit_tile(shared, qrow, srow, slot_base, feature_base // 32, lane, warp,
                                self.storage_c, self.storage_k, slots, features // 32)
                self._emit_tile(transposed, qcol, scol, feature_base, slot_base // 32, lane, warp,
                                self.d, self.storage_c, features, slots // 32)
            cute.arch.sync_threads()


@lru_cache(maxsize=128)
def compile_routed_dy_quant(n, d, e, c, storage_c, config, token_lanes=8,
                            raster='features', grid_mode='operand', active_m_alignment=0,
                            transpose_tile_k=0, row_tile_m=1):
    """Caller-owned FP32 dY/weights, i32 map, and FP8/byte-scale output pairs.

    The shared-tile schedule uses quantizer tile dimensions, not warp token
    lanes; that portfolio setting is used independently by the indexed X path.
    The bounded ABI appends the device i32 prefix to the existing tensors.
    """
    if token_lanes not in (2, 4, 8):
        raise ValueError('Invalid portfolio warp token lane setting')
    kernel = RoutedDYQuant(n, d, e, c, storage_c, config, raster, grid_mode,
                           active_m_alignment, transpose_tile_k, row_tile_m)
    width = kernel.storage_k
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    extra = ()
    if active_m_alignment:
        from .mxfp8_moe_active import fake_prefix
        extra = (fake_prefix(),)
    return cute.compile(kernel, _fake(Float32, n, d, alignment=4), _fake(Int32, e * storage_c, 1),
                        _fake(Float32, e, c), _fake(Float8E4M3FN, e * storage_c, width),
                        _fake(Uint8, e * storage_c, width // 32), _fake(Float8E4M3FN, e * d, storage_c),
                        _fake(Uint8, e * d, storage_c // 32), stream, *extra,
                        options='--enable-tvm-ffi --opt-level 3 '
                                f"--ptxas-options '-O3 --maxrregcount={config.maxrregcount}'")
