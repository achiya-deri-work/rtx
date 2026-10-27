"""Research-only vector stores for unchanged grouped MS-EDEN quantization."""
from functools import lru_cache
import cutlass.cute as cute
from cutlass import BFloat16,Float32,Uint8
from rtx.kernels.nvfp4_posthoc import HadamardPreparationKernel,PosthocScaleConfig


class GroupedStorePreparationKernel(HadamardPreparationKernel):
    def __init__(self,*args,store_bits=32,**kwargs):
        super().__init__(*args,**kwargs)
        if self.values_per_lane not in (8,16) or self.packed_source:
            raise ValueError("grouped stores require VPL8/16 and original BF16")
        if type(store_bits) is not int or store_bits not in (16,32,64) or store_bits>self.values_per_lane*4:
            raise ValueError("store width must be 16/32/64 and fit the thread's packed fragment")
        self.grouped_store_bits=store_bits


@lru_cache(maxsize=64)
def compile_grouped_store(rows,cols,config=PosthocScaleConfig(),mantissa_bits=4,transpose=False,*,store_bits=32,**kwargs):
    kernel=GroupedStorePreparationKernel(rows,cols,config,mantissa_bits,transpose,store_bits=store_bits,**kwargs)
    def fake(dtype,size,alignment=4):
        return cute.runtime.make_fake_tensor(dtype,(size,),stride=(1,),assumed_align=alignment)
    alignment=max(4,kernel.producer_copy_bits//8) if kernel.async_copy else max(4,kernel.producer_row_load_bits//8) if not transpose else 4
    runner=cute.compile(kernel,fake(BFloat16,rows*cols,alignment),fake(Float32,128),
        fake(Uint8,rows*cols//2,max(4,store_bits//8)),fake(BFloat16,rows*cols//16),fake(Float32,kernel.partial_count),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),options="--enable-tvm-ffi --opt-level 3")
    return runner,kernel.partial_count
