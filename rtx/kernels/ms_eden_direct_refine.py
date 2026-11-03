"""Research exact reciprocal sharing without per-value exponent rescaling.

On scale exponents [-30,30], quotient magnitudes [2^-32,16] and five-bit
positive significands, the normalized refinement proof transfers by exact
power-of-two scaling: reciprocals, estimates and nonzero residuals stay normal.
Everything outside that bounded domain uses explicit div.rn.f32.
"""
from functools import lru_cache
import cutlass
import cutlass.cute as cute
from cutlass import Float32,Int32,BFloat16,Uint8
from cutlass.experimental.primitives import nvvm_wrapper as nvvm
from .ms_eden_grouped_refine import GroupedRefinePreparationKernel
from .nvfp4_posthoc import PosthocScaleConfig


class DirectRefinePreparationKernel(GroupedRefinePreparationKernel):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,streamed=True,**kwargs)

    @cute.jit
    def normalization_parameters(self,scale:Float32,need_reciprocal:cutlass.Constexpr=True):
        bits=scale.bitcast(Int32)
        exponent=(bits>>23)&255
        eligible=(bits>0)&(exponent>=97)&(exponent<=157)&((bits&0x7ffff)==0)
        reciprocal=Float32(1.)
        if eligible:
            reciprocal=nvvm.inline_ptx_hl('div.rn.f32 {$w0}, 0f3f800000, {$r0};',
                write_only_types=[Float32],read_only_args=[scale])
        return eligible,scale,Float32(1.),reciprocal

    @cute.jit
    def normalize_refined(self,value:Float32,scale:Float32,eligible,
                          normalized_scale:Float32,inverse_power:Float32,reciprocal:Float32):
        magnitude=(value.bitcast(Int32)&Int32(0x7fffffff)).bitcast(Float32)
        fast=eligible&(magnitude>=scale*Float32(2.**-32))&(magnitude<=scale*Float32(16.))
        result=Float32(0.)
        if eligible&(magnitude==Float32(0.)):
            result=value
        elif fast:
            result=self.grouped_refined_quotient(value,scale,reciprocal)
        else:
            result=nvvm.inline_ptx_hl('div.rn.f32 {$w0}, {$r0}, {$r1};',
                write_only_types=[Float32],read_only_args=[value,scale])
        return result


@lru_cache(maxsize=64)
def compile_direct_refine(rows,cols,config=PosthocScaleConfig(),mantissa_bits=4,transpose=False,**kwargs):
    kernel=DirectRefinePreparationKernel(rows,cols,config,mantissa_bits,transpose,**kwargs)
    def fake(dtype,size,align=4):
        return cute.runtime.make_fake_tensor(dtype,(size,),stride=(1,),assumed_align=align)
    alignment=max(4,kernel.producer_copy_bits//8) if kernel.async_copy else max(4,kernel.producer_row_load_bits//8) if not transpose else 4
    run=cute.compile(kernel,fake(BFloat16,rows*cols,alignment),fake(Float32,128),
        fake(Uint8,rows*cols//2),fake(BFloat16,rows*cols//16),fake(Float32,kernel.partial_count),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),options='--enable-tvm-ffi --opt-level 3')
    return run,kernel.partial_count
