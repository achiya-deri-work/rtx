"""Research-only FP32x2 correction statistics; no precision policy change.

PTX 8.6 f32x2 mul/fma require SM100 or newer. Pair the independent norm and
dot products, preserving the scalar b-product then a-FMA rounding tree.
"""
from functools import lru_cache
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Uint8
from cutlass.experimental.primitives import nvvm_wrapper as nvvm
from rtx.kernels.nvfp4_posthoc import HadamardPreparationKernel, PosthocScaleConfig


class PairedFP32PreparationKernel(HadamardPreparationKernel):
    def __init__(self,*args,paired_transform=False,paired_statistics=True,**kwargs):
        super().__init__(*args,**kwargs)
        if self.values_per_lane not in (8,16) or self.packed_source:
            raise ValueError('paired statistics require original BF16 VPL8/16')
        if type(paired_transform) is not bool or type(paired_statistics) is not bool:
            raise ValueError('paired arithmetic switches must be boolean')
        self.paired_transform=paired_transform
        self.paired_statistics=paired_statistics

    @cute.jit
    def pair_butterfly(self,a0,a1,b0,b1):
        return nvvm.inline_ptx_hl(
            '{.reg .b64 av,bv,sv,dv; '
            'mov.b64 av, {{$r0}, {$r1}}; mov.b64 bv, {{$r2}, {$r3}}; '
            'add.rn.f32x2 sv, av, bv; sub.rn.f32x2 dv, av, bv; '
            'mov.b64 {{$w0}, {$w1}}, sv; mov.b64 {{$w2}, {$w3}}, dv;}',
            write_only_types=[Float32,Float32,Float32,Float32],read_only_args=[a0,a1,b0,b1])

    @cute.jit
    def pair_shuffle_sum(self,a0,a1,b0,b1,negative):
        return nvvm.inline_ptx_hl(
            '{.reg .b64 av,bv,cv; .reg .pred p; '
            'mov.b64 av, {{$r0}, {$r1}}; mov.b64 bv, {{$r2}, {$r3}}; '
            'setp.eq.u32 p, {$r4}, 0; '
            '@p add.rn.f32x2 cv, av, bv; @!p sub.rn.f32x2 cv, bv, av; '
            'mov.b64 {{$w0}, {$w1}}, cv;}',
            write_only_types=[Float32,Float32],read_only_args=[a0,a1,b0,b1,negative])

    @cute.jit
    def grouped_transform_values(self,values,local_lane):
        if cutlass.const_expr(not self.paired_transform):
            HadamardPreparationKernel.grouped_transform_values(self,values,local_lane)
        else:
            # First level has one sum/difference, not two independent pairs.
            for group in cutlass.range_constexpr(self.values_per_lane//2):
                a,b=values[2*group],values[2*group+1]
                values[2*group],values[2*group+1]=a+b,a-b
            for level in cutlass.range_constexpr(1,self.values_per_lane.bit_length()-1):
                distance=1 << level
                for group in cutlass.range_constexpr(self.values_per_lane//(2*distance)):
                    for pair in cutlass.range_constexpr(distance//2):
                        lo=group*2*distance+pair*2
                        s0,s1,d0,d1=self.pair_butterfly(values[lo],values[lo+1],values[lo+distance],values[lo+distance+1])
                        values[lo],values[lo+1]=s0,s1
                        values[lo+distance],values[lo+distance+1]=d0,d1
            for level in cutlass.range_constexpr(self.lanes_per_group.bit_length()-1):
                for pair in cutlass.range_constexpr(self.values_per_lane//2):
                    a,b=values[2*pair],values[2*pair+1]
                    other_a=cute.arch.shuffle_sync_bfly(a,1 << level)
                    other_b=cute.arch.shuffle_sync_bfly(b,1 << level)
                    values[2*pair],values[2*pair+1]=self.pair_shuffle_sum(a,b,other_a,other_b,local_lane & (1 << level))

    @cute.jit
    def grouped_pair_statistics(self,a,b,qa,qb):
        if cutlass.const_expr(not self.paired_statistics):
            return HadamardPreparationKernel.grouped_pair_statistics(self,a,b,qa,qb)
        return nvvm.inline_ptx_hl(
            '{.reg .b64 av,bv,cv,pv; '
            'mov.b64 av, {{$r1}, {$r1}}; mov.b64 bv, {{$r1}, {$r3}}; '
            'mul.rn.f32x2 pv, av, bv; '
            'mov.b64 av, {{$r0}, {$r0}}; mov.b64 bv, {{$r0}, {$r2}}; '
            'fma.rn.f32x2 cv, av, bv, pv; mov.b64 {{$w0}, {$w1}}, cv;}',
            write_only_types=[Float32,Float32],read_only_args=[a,b,qa,qb])


@lru_cache(maxsize=64)
def compile_paired_fp32(rows,cols,config=PosthocScaleConfig(),mantissa_bits=4,transpose=False,*,paired_transform=False,paired_statistics=True,**kwargs):
    kernel=PairedFP32PreparationKernel(rows,cols,config,mantissa_bits,transpose,
        paired_transform=paired_transform,paired_statistics=paired_statistics,**kwargs)
    def fake(dtype,size,alignment=4):
        return cute.runtime.make_fake_tensor(dtype,(size,),stride=(1,),assumed_align=alignment)
    alignment=max(4,kernel.producer_copy_bits//8) if kernel.async_copy else max(4,kernel.producer_row_load_bits//8) if not transpose else 4
    runner=cute.compile(kernel,fake(BFloat16,rows*cols,alignment),fake(Float32,128),
        fake(Uint8,rows*cols//2,max(4,kernel.grouped_store_bits//8)),
        fake(BFloat16,rows*cols//16),fake(Float32,kernel.partial_count),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),options='--enable-tvm-ffi --opt-level 3')
    return runner,kernel.partial_count
