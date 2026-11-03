"""Research-only full-K BF16 H16 transform; no quantizer/dispatch integration.

Two MMA instructions transform two independent H128 blocks without padding.
Original BF16 inputs and unit signs are exact MMA operands; accumulation is
FP32 but its association is not claimed bitwise equivalent to scalar FWHT.
"""
from dataclasses import dataclass
from functools import lru_cache
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16,Float32,Int32,Uint16
from cutlass.cute.nvgpu.warp import MmaF16BF16Op
from rtx.kernels.ms_eden_tf32_hadamard import TF32H8Config,TF32H8HadamardKernel


@dataclass(frozen=True)
class H16Config:
    threads:int=128
    blocks:int=128
    force_scalar:bool=False
    output_store_bits:int=32

    def __post_init__(self):
        TF32H8Config(self.threads,self.blocks,self.force_scalar)
        if type(self.output_store_bits) is not int or self.output_store_bits not in (32,64):
            raise ValueError('output_store_bits must be 32 or 64')


class H16HadamardKernel(TF32H8HadamardKernel):
    def __init__(self,rows,cols,config=H16Config()):
        super().__init__(rows,cols,config)
        self.pairs=(self.tiles+1)//2
        self.blocks=min(config.blocks,(self.pairs+config.threads//32-1)//(config.threads//32))
        self.warps=self.blocks*(config.threads//32)

    @cute.kernel
    def kernel(self,source:cute.Tensor,signs:cute.Tensor,out:cute.Tensor,used_tc:cute.Tensor):
        thread,_,_=cute.arch.thread_idx()
        block,_,_=cute.arch.block_idx()
        lane=cute.arch.lane_idx()
        warp=block*(self.config.threads//32)+thread//32
        scalar=cute.make_rmem_tensor((4,),Float32)
        if cutlass.const_expr(not self.config.force_scalar):
            cached_signs=cute.make_rmem_tensor((8,),Float32)
            valid=Int32(1)
            for reg in cutlass.range_constexpr(8):
                index=16*(lane//4)+2*(lane%4)+(reg&1)+8*(reg>>2)
                cached_signs[reg]=signs[index]
                if (cached_signs[reg].bitcast(Int32)&Int32(0x7fffffff))!=Int32(0x3f800000):valid=Int32(0)
            rademacher=cute.arch.vote_all_sync(valid!=0)
            mma=cute.make_mma_atom(MmaF16BF16Op(BFloat16,Float32,(16,8,16)))
            a=cute.make_rmem_tensor((8,1,1),BFloat16)
            coefficients=cute.make_rmem_tensor((4,2),BFloat16)
            for half in cutlass.range_constexpr(2):
                for reg in cutlass.range_constexpr(4):
                    k=2*(lane&3)+(reg&1)+8*(reg>>1)
                    n=(lane>>2)+8*half
                    parity=cute.arch.popc(k&n)&1
                    coefficients[reg,half]=Uint16(0x3f80|(parity<<15)).bitcast(BFloat16)
        for pair in cutlass.range(warp,self.pairs,self.warps,unroll=1):
            fast=False
            if cutlass.const_expr(not self.config.force_scalar):
                valid=Int32(1)
                for reg in cutlass.range_constexpr(8):
                    index=16*(lane//4)+2*(lane%4)+(reg&1)+128*((reg>>1)&1)+8*(reg>>2)
                    value=Float32(0.)
                    if pair*256+index<self.rows*self.cols:
                        value=source[pair*256+index].to(Float32)
                    absolute=value.bitcast(Int32)&Int32(0x7fffffff)
                    if (absolute!=0)&((absolute<Int32(0x0f800000))|(absolute>Int32(0x6f800000))):valid=Int32(0)
                    # Only consumed on the guarded unit-sign/normal-range path.
                    a[reg,0,0]=(value*cached_signs[reg]).to(BFloat16)
                fast=rademacher&cute.arch.vote_all_sync(valid!=0)
                if fast:
                    for half in cutlass.range_constexpr(2):
                        b=cute.make_rmem_tensor((4,1,1),BFloat16)
                        for reg in cutlass.range_constexpr(4):b[reg,0,0]=coefficients[reg,half]
                        acc=cute.make_rmem_tensor((4,1,1),Float32)
                        acc.fill(Float32(0.))
                        cute.gemm(mma,acc,a,b,acc)
                        for level in cutlass.range_constexpr(3):
                            for reg in cutlass.range_constexpr(4):
                                own=acc[reg,0,0]
                                other=cute.arch.shuffle_sync_bfly(own,1<<(level+2))
                                if (lane&(1<<(level+2)))==0:acc[reg,0,0]=own+other
                                else:acc[reg,0,0]=other-own
                        if cutlass.const_expr(self.config.output_store_bits==64):
                            for part in cutlass.range_constexpr(2):
                                index=16*(lane//4)+2*(lane%4)+128*part+8*half
                                if pair*256+index<self.rows*self.cols:
                                    values=cute.make_rmem_tensor((2,),Float32)
                                    for j in cutlass.range_constexpr(2):
                                        values[j]=acc[part*2+j,0,0]*Float32(.08838834764831845)
                                    offset=cute.assume(pair*256+index,divby=2)
                                    destination=cute.make_tensor(out.iterator+offset,cute.make_layout((2,),stride=(1,)))
                                    copy=cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(),Float32,num_bits_per_copy=64)
                                    cute.copy(copy,values,destination)
                        else:
                            for reg in cutlass.range_constexpr(4):
                                index=16*(lane//4)+2*(lane%4)+(reg&1)+128*(reg>>1)+8*half
                                if pair*256+index<self.rows*self.cols:
                                    out[pair*256+index]=acc[reg,0,0]*Float32(.08838834764831845)
            if not fast:
                for half in cutlass.range_constexpr(2):
                    tile=pair*2+half
                    if tile<self.tiles:
                        self.scalar_transform(source,signs,tile,scalar)
                        for reg in cutlass.range_constexpr(4):
                            out[tile*128+lane*4+reg]=scalar[reg]*Float32(.08838834764831845)
            if lane==0:
                used_tc[pair*2]=Int32(fast)
                if pair*2+1<self.tiles:used_tc[pair*2+1]=Int32(fast)


@lru_cache(maxsize=64)
def compile_h16_hadamard(rows,cols,config=H16Config()):
    kernel=H16HadamardKernel(rows,cols,config)
    def fake(dtype,size,alignment=4):return cute.runtime.make_fake_tensor(dtype,(size,),stride=(1,),assumed_align=alignment)
    return cute.compile(kernel,fake(BFloat16,rows*cols),fake(Float32,128),
        fake(Float32,rows*cols,config.output_store_bits//8),fake(Int32,rows*cols//128),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options='--enable-tvm-ffi --opt-level 3')
