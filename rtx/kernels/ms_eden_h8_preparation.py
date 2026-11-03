"""Research-only fused BF16-H8 preparation; stable dispatch is unchanged.

Reuses the original producer's loaders, FP32 normalization/correction, packing
and reductions. Only H128 association changes on exactly BF16-representable
signed inputs in a bounded normal range. No transformed GMEM tensor exists.
"""
from functools import lru_cache

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Uint8
from cutlass.cute.nvgpu.warp import MmaF16BF16Op
from rtx.kernels.nvfp4_posthoc import HadamardPreparationKernel, PosthocScaleConfig


class H8PreparationKernel(HadamardPreparationKernel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.values_per_lane != 4:
            raise ValueError("H8 prototype currently requires four values per lane")
        if self.packed_source:
            raise ValueError("H8 prototype requires original BF16 source")

    @cute.jit
    def transform_values(self, values):
        lane = cute.arch.lane_idx()
        valid = Int32(1)
        for j in cutlass.range_constexpr(4):
            absolute = values[j].bitcast(Int32) & Int32(0x7fffffff)
            # This also guards arbitrary signs: no narrowing of the existing
            # FP32 sign product is permitted, even if the source was BF16.
            if ((absolute != 0) & ((absolute < Int32(0x0f800000)) |
                                  (absolute > Int32(0x6f800000)))) | ((absolute & Int32(0xffff)) != 0):
                valid = Int32(0)
        fast = cute.arch.vote_all_sync(valid != 0)
        if fast:
            mma = cute.make_mma_atom(MmaF16BF16Op(BFloat16, Float32, (16,8,16)))
            a = cute.make_rmem_tensor((8,1,1), BFloat16)
            b = cute.make_rmem_tensor((4,1,1), BFloat16)
            c = cute.make_rmem_tensor((4,1,1), Float32)
            a.fill(BFloat16(0.))
            b.fill(BFloat16(0.))
            c.fill(Float32(0.))
            for j in cutlass.range_constexpr(4):
                a[j,0,0] = values[j].to(BFloat16)
            for j in cutlass.range_constexpr(2):
                parity = cute.arch.popc((2*(lane&3)+j)&(lane>>2))&1
                b[j,0,0] = (Int32(0x3f800000)|(parity<<31)).bitcast(Float32).to(BFloat16)
            cute.gemm(mma,c,a,b,c)
            for j in cutlass.range_constexpr(4):
                values[j] = c[j,0,0]
            x,y,z,w = values[0],values[1],values[2],values[3]
            values[0],values[1],values[2],values[3] = x+z,y+w,x-z,y-w
            for level in cutlass.range_constexpr(3):
                for j in cutlass.range_constexpr(4):
                    other = cute.arch.shuffle_sync_bfly(values[j],1 << (level+2))
                    own = values[j]
                    if (lane & (1 << (level+2))) == 0:
                        values[j] = own+other
                    else:
                        values[j] = other-own
        else:
            HadamardPreparationKernel.transform_values(self,values)


@lru_cache(maxsize=64)
def compile_h8_preparation(rows, cols, config=PosthocScaleConfig(), mantissa_bits=4,
                           transpose=False, **kwargs):
    kernel = H8PreparationKernel(rows,cols,config,mantissa_bits,transpose,**kwargs)
    def fake(dtype,size,alignment=4):
        return cute.runtime.make_fake_tensor(dtype,(size,),stride=(1,),assumed_align=alignment)
    alignment = max(4,kernel.producer_copy_bits//8) if kernel.async_copy else max(4,kernel.producer_row_load_bits//8) if not transpose else 4
    compiled = cute.compile(kernel,fake(BFloat16,rows*cols,alignment),fake(Float32,128),
        fake(Uint8,rows*cols//2),fake(BFloat16,rows*cols//16),fake(Float32,kernel.partial_count),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi --opt-level 3")
    return compiled,kernel.partial_count
