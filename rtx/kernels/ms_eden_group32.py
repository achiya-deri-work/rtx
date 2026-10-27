"""Research VPL32 transpose preparation with unchanged 16-value scales.

One additional butterfly level is register-local. Each resulting 32-value
fragment is quantized as two independent 16-value groups using the retained
FP32 normalization/correction implementation. No runtime/default dispatch.
"""
from functools import lru_cache
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Uint8
from .ms_eden_bank_layout import BankLayoutPreparationKernel
from .nvfp4_posthoc import HadamardPreparationKernel, PosthocScaleConfig


def group32_offset(row, column):
    return (row ^ ((row >> 6) & 1))*32 + (column ^ (((row >> 5) & 1) << 1))


class _QuantizeTransformed16(HadamardPreparationKernel):
    @cute.jit
    def grouped_transform_values(self, values, local_lane):
        # Parent grouped_prepare_values still applies the H128 normalization
        # factor, exactly once; only the already-completed butterfly is skipped.
        pass


class Group32PreparationKernel(BankLayoutPreparationKernel):
    def __init__(self, *args, combined_stores=False, **kwargs):
        if type(combined_stores) is not bool:
            raise ValueError('combined_stores must be boolean')
        self.combined_stores = combined_stores
        if kwargs.get('butterfly_fma', False):
            raise ValueError('group32 probe preserves ordinary butterfly arithmetic')
        super().__init__(*args, **kwargs)
        self.quantizer = _QuantizeTransformed16(
            128, 128, self.config, self.mantissa_bits, False,
            producer_values_per_lane=16, correction_decode=self.correction_decode,
            producer_scale_math=self.producer_scale_math,
            producer_grouped_store_bits=self.grouped_store_bits)
        self.values_per_lane = 32
        self.groups_per_warp = 8
        self.lanes_per_group = 4

    @cute.jit
    def offset(self, row, column):
        return (row ^ ((row >> 6) & 1))*32 + (column ^ (((row >> 5) & 1) << 1))

    @cute.jit
    def prepare_transpose_group(self, staged, signs, packed, scales, tile, base, largest, cached_reciprocal):
        thread, _, _ = cute.arch.thread_idx()
        lane = cute.arch.lane_idx()
        local_lane = lane % 4
        subrow = base + lane//4
        column = thread//32 + subrow*4
        local_tile = tile + subrow*4*(self.cols//128)
        flat = cute.make_tensor(staged.iterator, cute.make_layout((128*32,)))
        values = cute.make_rmem_tensor((32,), Float32)
        even = self.offset(local_lane*32, column)
        odd = self.offset(local_lane*32+1, column)
        for j in cutlass.range_constexpr(32):
            address = (even if cutlass.const_expr(j%2 == 0) else odd) + (j//2)*64
            sign = signs[j] if cutlass.const_expr(self.signs_mode == 'register') else signs[local_lane*32+j]
            values[j] = flat[address].to(Float32)*sign
        # Same seven bit stages in the same order; bit4 moved from a shuffle
        # butterfly to the register-local butterfly. No MMA reassociation.
        HadamardPreparationKernel.grouped_transform_values(self, values, local_lane)
        if cutlass.const_expr(self.combined_stores):
            local_codes = cute.make_rmem_tensor((16,), Uint8)
            local_scales = cute.make_rmem_tensor((2,), BFloat16)
        for half in cutlass.range_constexpr(2):
            fragment = cute.make_rmem_tensor((16,), Float32)
            for j in cutlass.range_constexpr(16):
                fragment[j] = values[half*16+j]
            if cutlass.const_expr(self.combined_stores):
                codes_half = cute.make_tensor(local_codes.iterator + half*8, cute.make_layout((8,)))
                scales_half = cute.make_tensor(local_scales.iterator + half, cute.make_layout((1,)))
                largest = self.quantizer.grouped_prepare_values(
                    fragment, codes_half, scales_half, 0, 0,
                    True, largest, cached_reciprocal)
            else:
                largest = self.quantizer.grouped_prepare_values(
                    fragment, packed, scales, local_tile, local_lane*2+half,
                    True, largest, cached_reciprocal)
        if cutlass.const_expr(self.combined_stores):
            packed_offset = cute.assume(local_tile*64 + local_lane*16, divby=16)
            scale_offset = cute.assume(local_tile*8 + local_lane*2, divby=2)
            cute.copy(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Uint8, num_bits_per_copy=128),
                      local_codes, cute.make_tensor(packed.iterator + packed_offset, cute.make_layout((16,))))
            cute.copy(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=32),
                      local_scales, cute.make_tensor(scales.iterator + scale_offset, cute.make_layout((2,))))
        return largest


@lru_cache(maxsize=64)
def compile_group32(rows, cols, config=PosthocScaleConfig(), mantissa_bits=4, transpose=False, **kwargs):
    kernel = Group32PreparationKernel(rows, cols, config, mantissa_bits, transpose, **kwargs)
    def fake(dtype, size, align=4):
        return cute.runtime.make_fake_tensor(dtype, (size,), stride=(1,), assumed_align=align)
    run = cute.compile(kernel, fake(BFloat16, rows*cols), fake(Float32, 128),
        fake(Uint8, rows*cols//2, 16 if kernel.combined_stores else 4), fake(BFloat16, rows*cols//16), fake(Float32, kernel.partial_count),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True), options='--enable-tvm-ffi --opt-level 3')
    return run, kernel.partial_count
