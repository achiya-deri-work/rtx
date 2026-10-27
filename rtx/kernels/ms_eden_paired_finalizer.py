"""Research-only paired scale conversion; stable finalizer remains unchanged."""
from functools import lru_cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Float8E4M3FN, Int16, Int32, Uint8
from cutlass.experimental.primitives import nvvm_wrapper as nvvm

from rtx.kernels.nvfp4_posthoc import PosthocScaleConfig, PosthocScaleKernel


class PairedScaleKernel(PosthocScaleKernel):
    def __init__(self, rows, groups, config, partial_count=1):
        super().__init__(rows, groups, config, partial_count)
        if groups % 2:
            raise ValueError('paired finalization requires an even group count')
        self.blocks = min(config.blocks, (rows*groups//2+config.threads-1)//config.threads)

    @cute.jit
    def store_code(self, dst: cute.Tensor, i, value, code, uniform):
        nearest = Uint8(code).bitcast(Float8E4M3FN).to(Float32)
        lo, hi = code, code
        if nearest > value:
            lo = code-1
        if nearest < value:
            hi = code+1
        low = Uint8(lo).bitcast(Float8E4M3FN).to(Float32)
        high = Uint8(hi).bitcast(Float8E4M3FN).to(Float32)
        probability = self.rounding_probability(value, low, high)
        selected = lo
        if uniform < probability:
            selected = hi
        offset = i
        if cutlass.const_expr(self.config.layout == 'mma128'):
            row, group = i//self.groups, i%self.groups
            offset = ((row//128)*(self.groups//8)+group//8)*1024
            offset += (group%8//4)*512+(row%32)*16+(row//32%4)*4+group%4
        dst[offset] = Uint8(selected).bitcast(Float8E4M3FN)

    @cute.jit
    def __call__(self, src: cute.Tensor, maximum: cute.Tensor, uniforms: cute.Tensor,
                 dst: cute.Tensor, outer: cute.Tensor, stream: cuda.CUstream):
        self.paired_kernel(src,maximum,uniforms,dst,outer).launch(
            grid=(self.blocks,1,1),block=(self.config.threads,1,1),stream=stream)

    @cute.kernel
    def paired_kernel(self, src: cute.Tensor, maximum: cute.Tensor, uniforms: cute.Tensor,
                      dst: cute.Tensor, outer: cute.Tensor):
        thread,_,_ = cute.arch.thread_idx()
        block,_,_ = cute.arch.block_idx()
        scale = self.tensor_scale(maximum)
        index = (block*self.config.threads+thread)*2
        if index == 0:
            outer[0] = scale
        for i in cutlass.range(index,self.rows*self.groups,self.blocks*self.config.threads*2,unroll=1):
            first = cute.arch.fmin(src[i].to(Float32)/scale,Float32(448.))
            second = cute.arch.fmin(src[i+1].to(Float32)/scale,Float32(448.))
            packed = nvvm.inline_ptx_hl(
                'cvt.rn.satfinite.e4m3x2.f32 {$w0}, {$r0}, {$r1};',
                write_only_types=[Int16],read_only_args=[second,first])
            self.store_code(dst,i,first,Int32(packed & Int16(255)),uniforms[i])
            self.store_code(dst,i+1,second,Int32((packed >> Int16(8)) & Int16(255)),uniforms[i+1])


@lru_cache(maxsize=64)
def compile_paired_scales(rows, groups, config=PosthocScaleConfig(), partial_count=1):
    kernel = PairedScaleKernel(rows,groups,config,partial_count)
    def fake(dtype,size):
        return cute.runtime.make_fake_tensor(dtype,(size,),stride=(1,),assumed_align=4)
    return cute.compile(kernel,fake(BFloat16,rows*groups),fake(Float32,partial_count),
        fake(Float32,rows*groups),fake(Float8E4M3FN,rows*groups),fake(Float32,1),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options='--enable-tvm-ffi --opt-level 3')
