"""Experimental guarded TF32-H8 + FP32 butterfly transform, not integrated.

Original BF16 values and Rademacher signs are exactly representable in TF32.
MMA accumulation ordering/rounding is NOT claimed to match scalar FWHT. Unsafe
range or arbitrary Float32 signs use the original scalar butterfly route.
No quantization, EDEN statistics, RNG, autograd, or production dispatch changes.
"""
from dataclasses import dataclass
from functools import lru_cache
import os

os.environ.setdefault("CUTE_DSL_ARCH","sm_120a")
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16,Float32,TFloat32,Int32,Uint16,Uint32
from cutlass.cute.nvgpu.warp import MmaTF32Op,MmaF16BF16Op


@dataclass(frozen=True)
class TF32H8Config:
    threads: int = 128
    blocks: int = 128
    force_scalar: bool = False
    mma_kind: str = "tf32"

    def __post_init__(self):
        if type(self.threads) is not int or self.threads not in (64,128,256):
            raise ValueError("TF32-H8 threads must be 64, 128, or 256")
        if type(self.blocks) is not int or self.blocks<1:
            raise ValueError("TF32-H8 blocks must be positive")
        if type(self.force_scalar) is not bool:
            raise ValueError("force_scalar must be boolean")
        if self.mma_kind not in ("tf32","bf16_padded"):
            raise ValueError("mma_kind must be tf32 or bf16_padded")


class TF32H8HadamardKernel:
    """One warp transforms one original contiguous 128-value block.

    A(m,k) uses logical bit positions M={4,5,6,1}, K={0,2,3}.
    C fragment registers then map directly to original index 4*lane+reg.
    Full-warp guards ensure every lane takes the same MMA/fallback branch.
    """
    def __init__(self,rows,cols,config=TF32H8Config()):
        if (type(rows) is not int or type(cols) is not int or rows<1 or cols<1 or cols%128):
            raise ValueError("positive rows and columns divisible by 128 required")
        self.rows,self.cols,self.config=rows,cols,config
        self.tiles=rows*(cols//128)
        self.blocks=min(config.blocks,(self.tiles+config.threads//32-1)//(config.threads//32))
        self.warps=self.blocks*(config.threads//32)

    @cute.jit
    def __call__(self,source:cute.Tensor,signs:cute.Tensor,out:cute.Tensor,
                 used_tc:cute.Tensor,stream:cuda.CUstream):
        self.kernel(source,signs,out,used_tc).launch(
            grid=(self.blocks,1,1),block=(self.config.threads,1,1),stream=stream)

    @cute.jit
    def scalar_transform(self,source,signs,tile,values,signs_are_cached:cutlass.Constexpr=False):
        """Original sign multiply and seven scalar butterfly bit stages.

        Keep source expressions identical to the original producer; don't
        replace arbitrary signs with sign-bit operations or narrow a product.
        """
        lane=cute.arch.lane_idx()
        for j in cutlass.range_constexpr(4):
            if cutlass.const_expr(signs_are_cached):
                values[j]=source[tile*128+lane*4+j].to(Float32)*signs[j]
            else:
                values[j]=source[tile*128+lane*4+j].to(Float32)*signs[lane*4+j]
        a,b,c,d=values[0],values[1],values[2],values[3]
        values[0],values[1],values[2],values[3]=a+b,a-b,c+d,c-d
        a,b,c,d=values[0],values[1],values[2],values[3]
        values[0],values[1],values[2],values[3]=a+c,b+d,a-c,b-d
        for level in cutlass.range_constexpr(5):
            for j in cutlass.range_constexpr(4):
                other=cute.arch.shuffle_sync_bfly(values[j],1 << level)
                own=values[j]
                if (lane & (1 << level))==0:
                    values[j]=own+other
                else:
                    values[j]=other-own

    @cute.kernel
    def kernel(self,source:cute.Tensor,signs:cute.Tensor,out:cute.Tensor,used_tc:cute.Tensor):
        thread,_,_=cute.arch.thread_idx()
        block,_,_=cute.arch.block_idx()
        lane=cute.arch.lane_idx()
        warp=block*(self.config.threads//32)+thread//32
        values=cute.make_rmem_tensor((4,),Float32)
        if cutlass.const_expr(self.config.force_scalar):
            # Fair persistent-loop control: production scalar producers cache
            # their four contiguous signs once per launch, not once per tile.
            # This constexpr branch adds no sign registers to the TF32 route.
            scalar_signs=cute.make_rmem_tensor((4,),Float32)
            for j in cutlass.range_constexpr(4):
                scalar_signs[j]=signs[lane*4+j]
            for tile in cutlass.range(warp,self.tiles,self.warps,unroll=1):
                self.scalar_transform(source,scalar_signs,tile,values,True)
                for j in cutlass.range_constexpr(4):
                    out[tile*128+lane*4+j]=values[j]*Float32(.08838834764831845)
                if lane==0:used_tc[tile]=Int32(0)
        else:
            # A register coordinates: row=(lane>>2)+8*(reg&1),
            # col=(lane&3)+4*(reg>>1). Their original input indices are
            # base+{0,2,8,10}, not the contiguous output-register coordinates.
            a_base=(lane&1)+((lane&2)<<1)+((lane&28)<<2)
            cached_signs=cute.make_rmem_tensor((4,),Float32)
            sign_valid=Int32(1)
            for j in cutlass.range_constexpr(4):
                if cutlass.const_expr(self.config.mma_kind=="bf16_padded"):
                    index=lane*4+j
                else:
                    index=a_base+((j&1)<<1)+((j>>1)<<3)
                cached_signs[j]=signs[index]
                if (cached_signs[j].bitcast(Int32)&Int32(0x7fffffff))!=Int32(0x3f800000):
                    sign_valid=Int32(0)
            rademacher=cute.arch.vote_all_sync(sign_valid!=0)
            if cutlass.const_expr(self.config.mma_kind=="bf16_padded"):
                mma=cute.make_mma_atom(MmaF16BF16Op(BFloat16,Float32,(16,8,16)))
                b_fragment=cute.make_rmem_tensor((4,1,1),BFloat16)
                b_fragment.fill(BFloat16(0.))
                for j in cutlass.range_constexpr(2):
                    parity=cute.arch.popc((2*(lane&3)+j)&(lane>>2))&1
                    b_fragment[j,0,0]=Uint16(0x3f80|(parity<<15)).bitcast(BFloat16)
                # Two packed sign masks, cached only in this specialization.
                sign_masks=cute.make_rmem_tensor((2,),Uint32)
                for j in cutlass.range_constexpr(2):
                    low=cached_signs[2*j].bitcast(Uint32)&Uint32(0x80000000)
                    high=cached_signs[2*j+1].bitcast(Uint32)&Uint32(0x80000000)
                    sign_masks[j]=(low>>16)|high
                packed_source=cute.make_tensor(cute.recast_ptr(source.iterator,dtype=Uint32),
                    cute.make_layout((self.rows*self.cols//2,),stride=(1,)))
                packed_inputs=cute.make_rmem_tensor((2,),Uint32)
            else:
                mma=cute.make_mma_atom(MmaTF32Op((16,8,8)))
                b_fragment=cute.make_rmem_tensor((2,1,1),TFloat32)
                for j in cutlass.range_constexpr(2):
                    # B is H8(k,n), with n=lane>>2, k=(lane&3)+4*j.
                    parity=cute.arch.popc(((lane&3)+4*j)&(lane>>2))&1
                    coefficient=(Int32(0x3f800000)|(parity<<31)).bitcast(Float32)
                    b_fragment[j,0,0]=coefficient.to(TFloat32)
            input_values=cute.make_rmem_tensor((4,),Float32)
            for tile in cutlass.range(warp,self.tiles,self.warps,unroll=1):
                local_valid=Int32(1)
                if cutlass.const_expr(self.config.mma_kind=="bf16_padded"):
                    for j in cutlass.range_constexpr(2):
                        packed_inputs[j]=packed_source[tile*64+lane*2+j]
                for j in cutlass.range_constexpr(4):
                    if cutlass.const_expr(self.config.mma_kind=="bf16_padded"):
                        word=(packed_inputs[j//2]>>((j&1)*16))&Uint32(0xffff)
                        input_values[j]=(word<<16).bitcast(Float32)
                    else:
                        index=a_base+((j&1)<<1)+((j>>1)<<3)
                        input_values[j]=source[tile*128+index].to(Float32)
                    absolute=input_values[j].bitcast(Int32)&Int32(0x7fffffff)
                    # 2^-96 <= |x| <= 2^96, or zero. Smallest possible
                    # nonzero exact sum has grid 2^-103, safely normal.
                    if (absolute!=0)&((absolute<Int32(0x0f800000))|(absolute>Int32(0x6f800000))):
                        local_valid=Int32(0)
                range_valid=cute.arch.vote_all_sync(local_valid!=0)
                fast=rademacher&range_valid
                if fast:
                    if cutlass.const_expr(self.config.mma_kind=="bf16_padded"):
                        a_fragment=cute.make_rmem_tensor((8,1,1),BFloat16)
                        a_fragment.fill(BFloat16(0.))
                        packed_a=cute.recast_tensor(a_fragment,Uint32)
                        for j in cutlass.range_constexpr(2):
                            packed_a[j,0,0]=packed_inputs[j]^sign_masks[j]
                    else:
                        a_fragment=cute.make_rmem_tensor((4,1,1),TFloat32)
                        for j in cutlass.range_constexpr(4):
                            # Unit sign multiplication is exact for guarded BF16.
                            a_fragment[j,0,0]=(input_values[j]*cached_signs[j]).to(TFloat32)
                    accumulator=cute.make_rmem_tensor((4,1,1),Float32)
                    accumulator.fill(Float32(0.))
                    cute.gemm(mma,accumulator,a_fragment,b_fragment,accumulator)
                    for j in cutlass.range_constexpr(4):values[j]=accumulator[j,0,0]
                    # C layout maps reg bit1 to original bit1; finish that H2.
                    a,b,c,d=values[0],values[1],values[2],values[3]
                    values[0],values[1],values[2],values[3]=a+c,b+d,a-c,b-d
                    # Original bits 4,5,6 are physical lane bits 2,3,4.
                    for level in cutlass.range_constexpr(3):
                        for j in cutlass.range_constexpr(4):
                            other=cute.arch.shuffle_sync_bfly(values[j],1 << (level+2))
                            own=values[j]
                            if (lane & (1 << (level+2)))==0:
                                values[j]=own+other
                            else:
                                values[j]=other-own
                else:
                    self.scalar_transform(source,signs,tile,values)
                for j in cutlass.range_constexpr(4):
                    out[tile*128+lane*4+j]=values[j]*Float32(.08838834764831845)
                if lane==0:used_tc[tile]=Int32(fast)


@lru_cache(maxsize=64)
def compile_tf32_h8_hadamard(rows,cols,config=TF32H8Config()):
    """Prototype launcher(source BF16, signs FP32[128], out FP32, route Int32).

    Output and caller-owned per-128-value route metadata must be allocated
    before capture. No production kernels/dispatch/statistics use this helper.
    """
    kernel=TF32H8HadamardKernel(rows,cols,config)
    fake=lambda dtype,size:cute.runtime.make_fake_tensor(dtype,(size,),stride=(1,),assumed_align=4)
    return cute.compile(kernel,fake(BFloat16,rows*cols),fake(Float32,128),
        fake(Float32,rows*cols),fake(Int32,rows*cols//128),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi --opt-level 3")


__all__=["TF32H8Config","TF32H8HadamardKernel","compile_tf32_h8_hadamard"]
