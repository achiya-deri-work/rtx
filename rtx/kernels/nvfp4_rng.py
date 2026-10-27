"""Counter RNG for experimental MS-EDEN scale finalization.

Philox4x32-10; explicit int64[2] key tensor with words in [0, 2**32).
Counter=(logical_scale_index//4, operand_stream, 0, 0), output lane=index%4.
The top 24 bits produce an exactly representable Float32 uniform in [0,1).
No mutable generator state, host scalar extraction, or large RNG buffers.
"""
from functools import lru_cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Float8E4M3FN, Int64, Uint32
from cutlass.experimental.primitives import nvvm_wrapper as nvvm

from .nvfp4_posthoc import PosthocScaleConfig, PosthocScaleKernel


@cute.jit
def _mul_hi(a: Uint32,b: Uint32):
    return nvvm.inline_ptx_hl("mul.hi.u32 {$w0}, {$r0}, {$r1};",
        write_only_types=[Uint32],read_only_args=[a,b])


@cute.jit
def philox4x32(seed: cute.Tensor,packet,operand_stream):
    c0,c1,c2,c3=Uint32(packet),Uint32(operand_stream),Uint32(0),Uint32(0)
    k0,k1=Uint32(seed[0]),Uint32(seed[1])
    for round_index in cutlass.range_constexpr(10):
        hi0,hi1=_mul_hi(Uint32(0xD2511F53),c0),_mul_hi(Uint32(0xCD9E8D57),c2)
        lo0,lo1=Uint32(0xD2511F53)*c0,Uint32(0xCD9E8D57)*c2
        c0,c1,c2,c3=hi1^c1^k0,lo1,hi0^c3^k1,lo0
        if cutlass.const_expr(round_index<9):
            k0,k1=k0+Uint32(0x9E3779B9),k1+Uint32(0xBB67AE85)
    return c0,c1,c2,c3


@cute.jit
def philox_uniform(seed: cute.Tensor,index,operand_stream):
    c0,c1,c2,c3=philox4x32(seed,index//4,operand_stream)
    value=c0
    if index%4==1:
        value=c1
    if index%4==2:
        value=c2
    if index%4==3:
        value=c3
    return (value>>Uint32(8)).to(Float32)*Float32(2.**-24)


class PhiloxPosthocScaleKernel(PosthocScaleKernel):
    def __init__(self,rows,groups,config,partial_count,operand_stream,packet_size=4):
        super().__init__(rows,groups,config,partial_count)
        if type(operand_stream) is not int or not 0<=operand_stream<2**32:
            raise ValueError("Philox operand stream must fit uint32")
        if rows*groups>=2**31:
            raise ValueError("Philox logical scale index exceeds signed kernel index domain")
        self.operand_stream=operand_stream
        if type(packet_size) is not int or packet_size not in (1,4):
            raise ValueError("Philox packet size must be 1 or 4")
        self.packet_size=packet_size
        self.packets=(rows*groups+3)//4
        self.packet_blocks=min(config.blocks,(self.packets+config.threads-1)//config.threads)

    @cute.jit
    def sample_uniform(self,seed: cute.Tensor,index):
        return philox_uniform(seed,index,self.operand_stream)

    @cute.jit
    def __call__(self,src:cute.Tensor,maximum:cute.Tensor,seed:cute.Tensor,
                 dst:cute.Tensor,outer:cute.Tensor,stream:cuda.CUstream):
        if cutlass.const_expr(self.packet_size==1):
            self.kernel(src,maximum,seed,dst,outer).launch(
                grid=(self.blocks,1,1),block=(self.config.threads,1,1),stream=stream)
        else:
            self.packet_kernel(src,maximum,seed,dst,outer).launch(
                grid=(self.packet_blocks,1,1),block=(self.config.threads,1,1),stream=stream)

    @cute.kernel
    def packet_kernel(self,src:cute.Tensor,maximum:cute.Tensor,seed:cute.Tensor,
                      dst:cute.Tensor,outer:cute.Tensor):
        thread,_,_=cute.arch.thread_idx()
        block,_,_=cute.arch.block_idx()
        scale=self.tensor_scale(maximum)
        first=block*self.config.threads+thread
        if first==0:
            outer[0]=scale
        for packet in cutlass.range(first,self.packets,self.packet_blocks*self.config.threads,unroll=1):
            words=philox4x32(seed,packet,self.operand_stream)
            for j in cutlass.range_constexpr(4):
                i=packet*4+j
                if i<self.rows*self.groups:
                    uniform=(words[j]>>Uint32(8)).to(Float32)*Float32(2.**-24)
                    self.finalize_one(src,dst,i,scale,uniform)


@lru_cache(maxsize=64)
def compile_philox_posthoc_scales(rows,groups,config=PosthocScaleConfig(),partial_count=1,operand_stream=0,packet_size=4):
    kernel=PhiloxPosthocScaleKernel(rows,groups,config,partial_count,operand_stream,packet_size)
    def fake(dtype,size):
        return cute.runtime.make_fake_tensor(dtype,(size,),stride=(1,),assumed_align=4)
    return cute.compile(kernel,fake(BFloat16,rows*groups),fake(Float32,partial_count),
        fake(Int64,2),fake(Float8E4M3FN,rows*groups),fake(Float32,1),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi --opt-level 3")


class PhiloxUniformKernel:
    """Diagnostic materialization for independent RNG validation, not runtime."""
    def __init__(self,count,operand_stream):
        if type(count) is not int or not 0<count<2**31:
            raise ValueError("positive Philox count within signed kernel index domain required")
        if type(operand_stream) is not int or not 0<=operand_stream<2**32:
            raise ValueError("Philox operand stream must fit uint32")
        self.count,self.operand_stream=count,operand_stream

    @cute.jit
    def __call__(self,seed:cute.Tensor,out:cute.Tensor,stream:cuda.CUstream):
        self.kernel(seed,out).launch(grid=(min(256,(self.count+127)//128),1,1),
                                     block=(128,1,1),stream=stream)

    @cute.kernel
    def kernel(self,seed:cute.Tensor,out:cute.Tensor):
        thread,_,_=cute.arch.thread_idx()
        block,_,_=cute.arch.block_idx()
        for index in cutlass.range(block*128+thread,self.count,
                min(256,(self.count+127)//128)*128,unroll=1):
            out[index]=philox_uniform(seed,index,self.operand_stream)


@lru_cache(maxsize=32)
def compile_philox_uniforms(count,operand_stream=0):
    kernel=PhiloxUniformKernel(count,operand_stream)
    seed=cute.runtime.make_fake_tensor(Int64,(2,),stride=(1,),assumed_align=8)
    out=cute.runtime.make_fake_tensor(Float32,(count,),stride=(1,),assumed_align=4)
    return cute.compile(kernel,seed,out,cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                        options="--enable-tvm-ffi --opt-level 3")
