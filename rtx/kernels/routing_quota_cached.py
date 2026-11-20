"""Optional register-cached radix thresholds for exact balanced quota repair.

Only the two threshold-selection launches differ from ``compile_quota``.
Small expert segments load their packed keys once; oversized segments use the
same streaming radix algorithm rather than dropping or truncating proposals.
Optional exact early exit recovers the full selected key once the surviving
prefix contains only one proposal, instead of finishing unused radix digits.
The tradeoff is additional registers even on fallback segments, so cache size
and thread count are execution choices, not a universal default.
"""
from functools import lru_cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Boolean,Float32,Int32,Int64,Uint64

from .routing_selection import compile_quota,histogram_add,update_radix


class CachedQuotaSelect:
    def __init__(self,m,e,k,threads=128,items=8,target=False,early_exit=False):
        if (type(m) is not int or not 0<m<2**31 or type(e) is not int or not 1<=e<=1024
                or type(k) is not int or k<1 or m%k):
            raise ValueError('Cached quota selection requires positive M/K, E<=1024 and M<2**31')
        if type(threads) is not int or threads not in (64,128,256):
            raise ValueError('Cached quota selection threads must be 64/128/256')
        if type(items) is not int or items not in (0,4,8,16):
            raise ValueError('Cached quota selection items must be 0/4/8/16')
        if type(target) is not bool:raise TypeError('target must be bool')
        if type(early_exit) is not bool:raise TypeError('early_exit must be bool')
        self.m,self.e,self.k,self.threads=m,e,k,threads
        self.items,self.target=items,target
        self.early_exit=early_exit
        self.capacity=threads*items

    @cute.jit
    def __call__(self,sources,targets,gains,surplus,deficit,slots,loads,phase,
                 counts,offsets,keys,thresholds,source_mask,target_mask,result,new_loads,new_phase,
                 stream:cuda.CUstream):
        self.kernel(sources,targets,gains,surplus,deficit,slots,loads,phase,counts,offsets,keys,
                    thresholds,source_mask,target_mask,result,new_loads,new_phase).launch(
            grid=(self.e,1,1),block=(self.threads,1,1),stream=stream)

    @cute.jit
    def radix_pass(self,keys,cached,offsets,count,expert,hist,prefix,rank,tidx,step):
        if tidx<16:hist[tidx]=Int32(0)
        cute.arch.sync_threads()
        shift=60-step*4
        mask=Uint64(0)
        if step>0:mask=(~Uint64(0))<<Uint64(shift+4)
        wanted=prefix[0]
        if count<=self.capacity:
            for item in cutlass.range_constexpr(self.items):
                element=tidx+item*self.threads
                digit=Int32(16)
                key=cached[item]
                if element<count and (key&mask)==wanted:
                    digit=Int32((key>>Uint64(shift))&Uint64(15))
                histogram_add(hist,digit,tidx)
        else:
            for item in cutlass.range(0,cute.ceil_div(count,self.threads),unroll=1):
                element=tidx+item*self.threads
                digit=Int32(16)
                if element<count:
                    key=keys[offsets[expert]+element]
                    if (key&mask)==wanted:digit=Int32((key>>Uint64(shift))&Uint64(15))
                histogram_add(hist,digit,tidx)
        update_radix(hist,prefix,rank,tidx,shift)

    @cute.jit
    def recover_unique_key(self,keys,cached,offsets,count,expert,prefix,thresholds,tidx,shift):
        # The chosen prefix contains exactly one proposal. Its entire key is
        # copied by that one owner; no atomic or approximate threshold is used.
        mask=(~Uint64(0))<<Uint64(shift)
        wanted=prefix[0]
        if count<=self.capacity:
            for item in cutlass.range_constexpr(self.items):
                element=tidx+item*self.threads
                key=cached[item]
                if element<count and (key&mask)==wanted:thresholds[expert]=key
        else:
            for item in cutlass.range(0,cute.ceil_div(count,self.threads),unroll=1):
                element=tidx+item*self.threads
                if element<count:
                    key=keys[offsets[expert]+element]
                    if (key&mask)==wanted:thresholds[expert]=key

    @cute.kernel
    def kernel(self,sources,targets,gains,surplus,deficit,slots,loads,phase,
               counts,offsets,keys,thresholds,source_mask,target_mask,result,new_loads,new_phase):
        tidx,_,_=cute.arch.thread_idx()
        expert,_,_=cute.arch.block_idx()
        count=offsets[expert+1]-offsets[expert]
        quotas=deficit if self.target else surplus
        take=Int32(quotas[expert])
        if count>take and take>0:
            smem=utils.SmemAllocator()
            hist=smem.allocate_tensor(Int32,cute.make_layout((16,)))
            prefix=smem.allocate_tensor(Uint64,cute.make_layout((1,)))
            rank=smem.allocate_tensor(Int32,cute.make_layout((1,)))
            cached=cute.make_rmem_tensor((max(1,self.items),),Uint64)
            cached.fill(Uint64(0))
            if count<=self.capacity:
                for item in cutlass.range_constexpr(self.items):
                    element=tidx+item*self.threads
                    if element<count:cached[item]=keys[offsets[expert]+element]
            if tidx==0:prefix[0]=Uint64(0);rank[0]=take-1
            if cutlass.const_expr(self.early_exit):
                done=smem.allocate_tensor(Int32,cute.make_layout((1,)))
                if tidx==0:done[0]=Int32(0)
            cute.arch.sync_threads()
            for step in cutlass.range(0,16,unroll=1):
                if cutlass.const_expr(self.early_exit):
                    if done[0]==0:
                        self.radix_pass(keys,cached,offsets,count,expert,hist,prefix,rank,tidx,step)
                        shift=60-step*4
                        if tidx==0:
                            chosen=Int32((prefix[0]>>Uint64(shift))&Uint64(15))
                            if hist[chosen]==1:done[0]=Int32(1)
                        cute.arch.sync_threads()
                        if done[0]!=0:
                            self.recover_unique_key(keys,cached,offsets,count,expert,prefix,
                                                    thresholds,tidx,shift)
                else:
                    self.radix_pass(keys,cached,offsets,count,expert,hist,prefix,rank,tidx,step)
            if cutlass.const_expr(self.early_exit):
                if tidx==0 and done[0]==0:thresholds[expert]=prefix[0]
            else:
                if tidx==0:thresholds[expert]=prefix[0]
        else:
            if tidx==0:thresholds[expert]=Uint64(0)


@lru_cache(maxsize=128)
def compile_quota_cached(m,e,k,threads=128,items=8,early_exit=False):
    """Same 12-stage/17-tensor ABI as segmented ``compile_quota``.

    ``items=0, early_exit=False`` returns the original implementation. All
    variants replace only source/target selection at indices 4 and 9.
    """
    if type(items) is not int or items not in (0,4,8,16):
        raise ValueError('Cached quota selection items must be 0/4/8/16')
    if type(early_exit) is not bool:raise TypeError('early_exit must be bool')
    if items==0 and not early_exit:return compile_quota(m,e,k,threads,True)
    kernels=[CachedQuotaSelect(m,e,k,threads,items,target,early_exit) for target in (False,True)]
    stages=list(compile_quota(m,e,k,threads,True))
    def fake(dtype,size):
        alignment=8 if dtype in (Int64,Uint64) else 1 if dtype==Boolean else 4
        return cute.runtime.make_fake_tensor(dtype,(size,),stride=(1,),assumed_align=alignment)
    tensors=[fake(Int64,m),fake(Int64,m),fake(Float32,m),fake(Int64,e),fake(Int64,e),
             fake(Int64,m),fake(Int64,e),fake(Int64,1),fake(Int32,e),fake(Int32,e+1),
             fake(Uint64,m),fake(Uint64,e),fake(Boolean,m),fake(Boolean,m),fake(Int64,m),
             fake(Int64,e),fake(Int64,1)]
    stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    for index,kernel in zip((4,9),kernels):
        stages[index]=cute.compile(kernel,*tensors,stream,options='--enable-tvm-ffi --opt-level 3')
    return tuple(stages)
