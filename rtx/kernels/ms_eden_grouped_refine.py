"""Research-only quotient and scale-free experiments for VPL8/16 producers.

Same proof/domain as normalize_refined, but one fragment-wide safety decision
and reciprocal amortized across 8/16 quotients. No default dispatch change.
The separate scale_free option preserves FP4 codes but reassociates FP32 EDEN
correction; its BF16 scales are bounded against an independent FP64 oracle,
not claimed byte-identical to the ordinary producer.
"""
from functools import lru_cache
import cutlass.cute as cute
from cutlass import BFloat16,Float32,Uint8
from rtx.kernels.nvfp4_posthoc import HadamardPreparationKernel,PosthocScaleConfig


class GroupedRefinePreparationKernel(HadamardPreparationKernel):
    def __init__(self,*args,streamed=False,reciprocal_lut=False,scale_free=False,scale_free_streamed=False,**kwargs):
        super().__init__(*args,**kwargs)
        if self.values_per_lane not in (8,16) or self.mantissa_bits!=4 or self.packed_source:
            raise ValueError("grouped refinement requires VPL8/16, mantissa4 and original BF16")
        self.grouped_refinement=True
        if type(streamed) is not bool:
            raise ValueError('streamed must be boolean')
        self.grouped_refinement_streamed=streamed
        if type(reciprocal_lut) is not bool:
            raise ValueError('reciprocal_lut must be boolean')
        self.grouped_refinement_lut=reciprocal_lut
        if type(scale_free) is not bool:
            raise ValueError('scale_free must be boolean')
        if scale_free and (streamed or reciprocal_lut):
            raise ValueError('scale-free and reciprocal refinement are separate policies')
        self.grouped_scale_free=scale_free
        self.grouped_refinement=not scale_free
        if type(scale_free_streamed) is not bool or (scale_free_streamed and not scale_free):
            raise ValueError('scale_free_streamed requires scale_free')
        self.grouped_scale_free_streamed=scale_free_streamed


@lru_cache(maxsize=64)
def compile_grouped_refine(rows,cols,config=PosthocScaleConfig(),mantissa_bits=4,transpose=False,*,streamed=False,reciprocal_lut=False,scale_free=False,scale_free_streamed=False,**kwargs):
    kernel=GroupedRefinePreparationKernel(rows,cols,config,mantissa_bits,transpose,streamed=streamed,reciprocal_lut=reciprocal_lut,scale_free=scale_free,scale_free_streamed=scale_free_streamed,**kwargs)
    def fake(dtype,size,alignment=4):
        return cute.runtime.make_fake_tensor(dtype,(size,),stride=(1,),assumed_align=alignment)
    alignment=max(4,kernel.producer_copy_bits//8) if kernel.async_copy else max(4,kernel.producer_row_load_bits//8) if not transpose else 4
    runner=cute.compile(kernel,fake(BFloat16,rows*cols,alignment),fake(Float32,128),
        fake(Uint8,rows*cols//2),fake(BFloat16,rows*cols//16),fake(Float32,kernel.partial_count),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),options="--enable-tvm-ffi --opt-level 3")
    return runner,kernel.partial_count
