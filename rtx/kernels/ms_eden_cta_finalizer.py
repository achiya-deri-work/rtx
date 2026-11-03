"""Research finalizer: cooperate across CTA when reducing producer maxima."""
from functools import lru_cache

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Float8E4M3FN
from rtx.kernels.nvfp4_posthoc import PosthocScaleConfig, PosthocScaleKernel


class CTAScaleKernel(PosthocScaleKernel):
    @cute.jit
    def tensor_scale(self, maximum: cute.Tensor):
        largest=Float32(0.)
        if cutlass.const_expr(self.partial_count==1):
            largest=maximum[0]
        else:
            thread,_,_=cute.arch.thread_idx()
            lane=cute.arch.lane_idx()
            allocator=cutlass.utils.SmemAllocator()
            maxima=allocator.allocate_tensor(Float32,cute.make_layout((self.config.threads//32,)),byte_alignment=4)
            for index in cutlass.range(thread,self.partial_count,self.config.threads,unroll=1):
                largest=cute.arch.fmax(largest,maximum[index])
            for level in cutlass.range_constexpr(5):
                largest=cute.arch.fmax(largest,cute.arch.shuffle_sync_bfly(largest,16 >> level))
            if lane==0:
                maxima[thread//32]=largest
            cute.arch.sync_threads()
            largest=Float32(0.)
            for warp in cutlass.range_constexpr(self.config.threads//32):
                largest=cute.arch.fmax(largest,maxima[warp])
        scale=Float32(1.)
        if largest>Float32(0.):
            scale=largest*Float32(1./448.)
        return scale


@lru_cache(maxsize=64)
def compile_cta_scales(rows,groups,config=PosthocScaleConfig(),partial_count=1):
    kernel=CTAScaleKernel(rows,groups,config,partial_count)
    def fake(dtype,size):
        return cute.runtime.make_fake_tensor(dtype,(size,),stride=(1,),assumed_align=4)
    return cute.compile(kernel,fake(BFloat16,rows*groups),fake(Float32,partial_count),
        fake(Float32,rows*groups),fake(Float8E4M3FN,rows*groups),fake(Float32,1),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options='--enable-tvm-ffi --opt-level 3')
