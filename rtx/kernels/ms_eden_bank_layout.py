"""Research-only pair-preserving XOR layout for the VPL16 transpose producer.

Arithmetic and work ownership remain inherited from the retained producer.
No public dispatch or default is changed.
"""
from functools import lru_cache
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Uint8
from cutlass.experimental.primitives import nvvm_wrapper as nvvm
from cutlass.cute.nvgpu import cpasync
from .nvfp4_posthoc import HadamardPreparationKernel, PosthocScaleConfig


def bank_layout_offset(row, column):
    """32-column BF16 tile; permute whole 32-bit pairs, never pair elements."""
    mask = (((row >> 4) & 1) << 1) | (((row >> 5) & 1) << 4)
    return (row ^ ((row >> 6) & 1)) * 32 + (column ^ mask)


class BankLayoutPreparationKernel(HadamardPreparationKernel):
    def __init__(self, *args, address_hoist=False, min_blocks_hint=0, butterfly_fma=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.address_hoist = address_hoist
        if type(butterfly_fma) is not bool:
            raise ValueError('butterfly_fma must be boolean')
        self.butterfly_fma = butterfly_fma
        if not (self.transpose and self.async_copy and self.config.threads == 128
                and self.source_tile_rows == 32 and self.values_per_lane == 16
                and self.producer_copy_bits == 32):
            raise ValueError('bank-layout experiment requires async x8 transpose, 128 threads, VPL16, copy32')
        self.async_row_stride = 32
        if min_blocks_hint not in (0, 7, 8):
            raise ValueError('unsupported research occupancy hint')
        if min_blocks_hint and self.stages != 1:
            raise ValueError('occupancy-hint probe requires one SMEM stage')
        self.min_blocks_hint = min_blocks_hint

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
    def grouped_transform_values(self, values, local_lane):
        if cutlass.const_expr(not self.butterfly_fma):
            HadamardPreparationKernel.grouped_transform_values(self, values, local_lane)
        else:
            # Keep register-local association; only fuse sign selection with
            # the cross-lane add. Multiplication by +/-1 is exact for finite
            # FP32, so FMA introduces no additional rounding or narrowing.
            for level in cutlass.range_constexpr(self.values_per_lane.bit_length()-1):
                distance = 1 << level
                for group in cutlass.range_constexpr(self.values_per_lane//(2*distance)):
                    for j in cutlass.range_constexpr(distance):
                        lo = group*2*distance+j
                        a, b = values[lo], values[lo+distance]
                        values[lo], values[lo+distance] = a+b, a-b
            for level in cutlass.range_constexpr(self.lanes_per_group.bit_length()-1):
                sign = (Int32(0x3f800000) | ((local_lane & (1 << level)) << (31-level))).bitcast(Float32)
                for j in cutlass.range_constexpr(self.values_per_lane):
                    own = values[j]
                    other = cute.arch.shuffle_sync_bfly(own, 1 << level)
                    values[j] = nvvm.inline_ptx_hl(
                        'fma.rn.f32 {$w0}, {$r0}, {$r1}, {$r2};',
                        write_only_types=[Float32], read_only_args=[sign, own, other])

    @cute.jit
    def offset(self, row, column):
        mask = (((row >> 4) & 1) << 1) | (((row >> 5) & 1) << 4)
        return (row ^ ((row >> 6) & 1)) * 32 + (column ^ mask)

    @cute.jit
    def stage_async(self, src, staged, work_tile, stage):
        thread, _, _ = cute.arch.thread_idx()
        row_group = work_tile // (self.cols // 128)
        chunk = work_tile % (self.cols // 128)
        if cutlass.const_expr(self.tile_order == 'reduction_major'):
            row_group = work_tile % (self.rows // self.source_tile_rows)
            chunk = work_tile // (self.rows // self.source_tile_rows)
        atom = cute.make_copy_atom(cpasync.CopyG2SOp(), BFloat16, num_bits_per_copy=32)
        for j in cutlass.range_constexpr(16):
            index = (thread + j * 128) * 2
            row, column = index // 32, index % 32
            source = cute.assume((chunk * 128 + row) * self.source_cols + row_group * 32 + column, divby=2)
            target = cute.assume(self.offset(row, column) + stage * 128 * 32, divby=2)
            cute.copy(atom, cute.make_tensor(src.iterator + source, cute.make_layout((2,))),
                      cute.make_tensor(staged.iterator + target, cute.make_layout((2,))))

    @cute.jit
    def apply_cached_sign(self, value, signs, local_lane, j: cutlass.Constexpr):
        sign = signs[j] if cutlass.const_expr(self.signs_mode == 'register') else signs[local_lane * 16 + j]
        return value * sign

    @cute.jit
    def prepare_transpose_group(self, staged, signs, packed, scales, tile, base, largest, cached_reciprocal):
        thread, _, _ = cute.arch.thread_idx()
        lane = cute.arch.lane_idx()
        local_lane = lane % 8
        subrow = base + lane // 8
        column = thread // 32 + subrow * 4
        local_tile = tile + subrow * 4 * (self.cols // 128)
        flat = cute.make_tensor(staged.iterator, cute.make_layout((128 * 32,)))
        values = cute.make_rmem_tensor((16,), Float32)
        if cutlass.const_expr(self.address_hoist):
            even = self.offset(local_lane * 16, column)
            odd = self.offset(local_lane * 16 + 1, column)
        for j in cutlass.range_constexpr(16):
            if cutlass.const_expr(self.address_hoist):
                address = (even if cutlass.const_expr(j % 2 == 0) else odd) + (j // 2) * 64
            else:
                address = self.offset(local_lane * 16 + j, column)
            values[j] = self.apply_cached_sign(flat[address].to(Float32), signs, local_lane, j)
        return self.grouped_prepare_values(values, packed, scales, local_tile, local_lane, True, largest, cached_reciprocal)


@lru_cache(maxsize=64)
def compile_bank_layout(rows, cols, config=PosthocScaleConfig(), mantissa_bits=4, transpose=False, **kwargs):
    kernel = BankLayoutPreparationKernel(rows, cols, config, mantissa_bits, transpose, **kwargs)
    def fake(dtype, size, align=4):
        return cute.runtime.make_fake_tensor(dtype, (size,), stride=(1,), assumed_align=align)
    run = cute.compile(kernel, fake(BFloat16, rows * cols), fake(Float32, 128),
        fake(Uint8, rows * cols // 2, max(4, kernel.grouped_store_bits // 8)), fake(BFloat16, rows * cols // 16), fake(Float32, kernel.partial_count),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True), options='--enable-tvm-ffi --opt-level 3')
    return run, kernel.partial_count


class RowFmaPreparationKernel(HadamardPreparationKernel):
    """Research row reader with the already-tested exact butterfly instruction form."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.transpose or self.values_per_lane not in (8, 16) or self.packed_source:
            raise ValueError('row FMA requires original BF16 grouped row preparation')
        self.butterfly_fma = True

    @cute.jit
    def grouped_transform_values(self, values, local_lane):
        BankLayoutPreparationKernel.grouped_transform_values(self, values, local_lane)


@lru_cache(maxsize=64)
def compile_row_fma(rows, cols, config=PosthocScaleConfig(), mantissa_bits=4, transpose=False, **kwargs):
    kernel = RowFmaPreparationKernel(rows, cols, config, mantissa_bits, transpose, **kwargs)
    def fake(dtype, size, align=4):
        return cute.runtime.make_fake_tensor(dtype, (size,), stride=(1,), assumed_align=align)
    run = cute.compile(kernel, fake(BFloat16, rows * cols), fake(Float32, 128),
        fake(Uint8, rows * cols // 2, max(4, kernel.grouped_store_bits // 8)),
        fake(BFloat16, rows * cols // 16), fake(Float32, kernel.partial_count),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options='--enable-tvm-ffi --opt-level 3')
    return run, kernel.partial_count
