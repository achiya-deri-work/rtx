"""Experimental one-warp emulation of the observed ATen short-row topk path.

Nondifferentiable packed solver only. Not a replacement for torch.topk.
The detector traces use gatherTopK then bitonicSortKVInPlace with 32 slots.
"""
from functools import lru_cache

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.experimental.primitives import nvvm_wrapper as nvvm


class SelectRows:
    def __init__(self,n,e,k,threads):
        if not (0<n and 1<=k<=e<=32 and threads in (64,128,256)):
            raise ValueError('Expected one warp per nonempty short row')
        self.n,self.e,self.k,self.threads=n,e,k,threads

    @cute.jit
    def __call__(self,src,values,ids,stream):
        self.kernel(src,values,ids).launch(grid=(cute.ceil_div(self.n,self.threads//32),1,1),
                                          block=(self.threads,1,1),stream=stream)

    @cute.kernel
    def kernel(self,src,values,ids):
        tid,_,_=cute.arch.thread_idx()
        block,_,_=cute.arch.block_idx()
        lane=tid%32
        row=block*(self.threads//32)+tid//32
        x=Float32(float('-inf'))
        if row<self.n and lane<self.e:x=src[row*self.e+lane]
        bits=nvvm.inline_ptx_hl('mov.b32 {$w0}, {$r0};',write_only_types=[Uint32],read_only_args=[x])
        key=(~bits) if (bits>>Uint32(31))!=0 else bits^Uint32(0x80000000)
        ordered=key if lane<self.e else Uint32(0)
        for level in cutlass.range_constexpr(5):
            for j in cutlass.range_constexpr(level+1):
                offset=1<<(level-j)
                other=cute.arch.shuffle_sync_bfly(ordered,offset)
                high=((lane&(2<<level))==0)==((lane&offset)==0)
                ordered=max(ordered,other) if high else min(ordered,other)
        kth=cute.arch.shuffle_sync(ordered,self.k-1)
        greater=(lane<self.e) & (key>kth)
        equal=(lane<self.e) & (key==kth)
        gt=Uint32(cute.arch.vote_ballot_sync(greater))
        eq=Uint32(cute.arch.vote_ballot_sync(equal))
        prior=(Uint32(1)<<Uint32(lane))-Uint32(1)
        dest=Int32(cute.arch.popc(gt&prior)) if greater else Int32(cute.arch.popc(gt)+cute.arch.popc(eq&prior))
        smem=utils.SmemAllocator()
        packed=smem.allocate_tensor(Int32,cute.make_layout((self.threads,)))
        if (greater or equal) and dest<self.k:packed[tid-lane+dest]=lane
        cute.arch.sync_warp()
        idx=Int32(-1)
        if lane<self.k:idx=packed[tid]
        v=cute.arch.shuffle_sync(x,max(idx,Int32(0)))
        # Match the 32-slot CUDA sort network's equal-key swaps; adding
        # an index tie-breaker or sorting the original row is not equivalent.
        for level in cutlass.range_constexpr(5):
            for j in cutlass.range_constexpr(level+1):
                offset=1<<(level-j)
                pv=cute.arch.shuffle_sync_bfly(v,offset)
                pi=cute.arch.shuffle_sync_bfly(idx,offset)
                lower=(lane&offset)==0
                before=((v>pv and idx>=0) or pi<0) if lower else ((pv>v and pi>=0) or idx<0)
                if before==((lane&(2<<level))!=0):
                    v=pv
                    idx=pi
        if row<self.n and lane<self.k:
            values[row*self.k+lane]=v
            ids[row*self.k+lane]=Int64(idx)


@lru_cache(maxsize=32)
def compile_select(n,e,k,threads=128):
    def tensor(dtype,count):
        return cute.runtime.make_fake_tensor(dtype,(count,),stride=(1,),assumed_align=4 if dtype==Float32 else 8)
    return cute.compile(SelectRows(n,e,k,threads),tensor(Float32,n*e),tensor(Float32,n*k),
                        tensor(Int64,n*k),cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                        options='--enable-tvm-ffi --opt-level 3')
