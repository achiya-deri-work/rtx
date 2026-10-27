"""Research-only compact Rademacher sign cache; finite BF16 inputs, signs ±1.

Not a replacement for the general external-sign producer. Only the sign cache
representation changes; Hadamard, scale, correction and output math are inherited.
"""
from functools import lru_cache
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Uint8
from .ms_eden_bank_layout import BankLayoutPreparationKernel
from .nvfp4_posthoc import PosthocScaleConfig


class PackedSignPreparationKernel(BankLayoutPreparationKernel):
    def __init__(self, *args, min_blocks_hint=0, **kwargs):
        super().__init__(*args, **kwargs)
        if min_blocks_hint not in (0, 7, 8):
            raise ValueError('unsupported research occupancy hint')
        if min_blocks_hint and self.stages != 1:
            raise ValueError('occupancy-hint probe is restricted to one SMEM stage')
        self.min_blocks_hint = min_blocks_hint
        if self.signs_mode != 'register':
            raise ValueError('packed Rademacher signs require register sign caching')

    @cute.jit
    def __call__(self, src: cute.Tensor, signs: cute.Tensor, packed: cute.Tensor,
                 scales: cute.Tensor, partials: cute.Tensor, stream: cuda.CUstream):
        if cutlass.const_expr(self.min_blocks_hint > 0):
            self.buffered_rotated_kernel(src, signs, packed, scales, partials).launch(
                grid=(self.blocks, 1, 1), block=(128, 1, 1), stream=stream,
                max_number_threads=(128, 1, 1), min_blocks_per_mp=self.min_blocks_hint)
        else:
            self.buffered_rotated_kernel(src, signs, packed, scales, partials).launch(
                grid=(self.blocks, 1, 1), block=(128, 1, 1), stream=stream)

    @cute.jit
    def apply_cached_sign(self, value, signs, local_lane, j: cutlass.Constexpr):
        mask = ((signs[0] >> j) & Int32(1)) << 31
        return (value.bitcast(Int32) ^ mask).bitcast(Float32)

    @cute.kernel
    def buffered_rotated_kernel(self, src: cute.Tensor, signs: cute.Tensor,
                                packed: cute.Tensor, scales: cute.Tensor,
                                partials: cute.Tensor):
        thread, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        largest = Float32(0.)
        cached_reciprocal = Float32(1.)
        if cutlass.const_expr(self.grouped_refinement_lut or self.normalization_math in ('refine_warp_lut', 'guarded_scale_free')):
            cached_reciprocal = self.warp_reciprocal_entry()
        bits = Int32(0)
        for j in cutlass.range_constexpr(16):
            bits |= ((signs[(lane % 8) * 16 + j].bitcast(Int32) >> 31) & Int32(1)) << j
        cached_signs = cute.make_rmem_tensor((1,), Int32)
        cached_signs[0] = bits
        smem = cutlass.utils.SmemAllocator()
        warp_maxima = partials
        if cutlass.const_expr(self.partial_max_scope == 'cta'):
            warp_maxima = smem.allocate_tensor(Float32, cute.make_layout((self.config.threads // 32,)), byte_alignment=4)
        staged = smem.allocate_tensor(BFloat16,
            cute.make_layout((128, self.source_tile_rows, self.stages),
                             stride=(self.async_row_stride, 1, 128 * self.async_row_stride)),
            byte_alignment=16)
        limit = self.tiles // self.source_tile_rows
        for stage in cutlass.range_constexpr(self.stages):
            work_tile = block + stage * self.blocks
            if work_tile < limit:
                self.stage_async(src, staged, work_tile, stage)
            cute.arch.cp_async_commit_group()
        for iteration in cutlass.range(0, (limit + self.blocks - 1 - block) // self.blocks, unroll=1):
            slot = iteration % self.stages
            work_tile = block + iteration * self.blocks
            cute.arch.cp_async_wait_group(self.stages - 1)
            cute.arch.sync_threads()
            row_group = work_tile // (self.cols // 128)
            chunk = work_tile % (self.cols // 128)
            if cutlass.const_expr(self.tile_order == 'reduction_major'):
                row_group = work_tile % (self.rows // self.source_tile_rows)
                chunk = work_tile // (self.rows // self.source_tile_rows)
            tile = (row_group * self.source_tile_rows + thread // 32) * (self.cols // 128) + chunk
            largest = self.prepare_rows(src, cached_signs, packed, scales, signs, signs,
                staged[None, None, slot], tile, largest, cached_reciprocal)
            cute.arch.sync_threads()
            future = work_tile + self.stages * self.blocks
            if future < limit:
                self.stage_async(src, staged, future, slot)
            cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        self.write_partial_maximum(largest, partials, warp_maxima)


@lru_cache(maxsize=64)
def compile_packed_rademacher(rows, cols, config=PosthocScaleConfig(), mantissa_bits=4, transpose=False, *, register_cap=255, **kwargs):
    if register_cap not in (64, 72, 80, 96, 255):
        raise ValueError('unsupported research register ceiling')
    kernel = PackedSignPreparationKernel(rows, cols, config, mantissa_bits, transpose, **kwargs)
    def fake(dtype, size):
        return cute.runtime.make_fake_tensor(dtype, (size,), stride=(1,), assumed_align=4)
    options = '--enable-tvm-ffi --opt-level 3'
    if register_cap != 255:
        options += f" --ptxas-options '-O3 --maxrregcount={register_cap}'"
    run = cute.compile(kernel, fake(BFloat16, rows * cols), fake(Float32, 128),
        fake(Uint8, rows * cols // 2), fake(BFloat16, rows * cols // 16), fake(Float32, kernel.partial_count),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True), options=options)
    return run, kernel.partial_count
