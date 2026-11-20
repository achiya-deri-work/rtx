"""Stable fixed-capacity expert grouping and selected-log softmax in CuTe.

Integer grouping preserves token-major flattened assignment order exactly.
No atomic slot allocation, global sort, CUB, or materialized tensor transpose.
"""
from functools import lru_cache

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Float32,Int32,Int64,Uint32,Boolean


class StableMetadataKernel:
    def __init__(self,n,e,k,real,threads,mode):
        self.n,self.e,self.k,self.real,self.threads,self.mode=n,e,k,real,threads,mode
        self.m=n*k;self.capacity=self.m//e
        self.chunks=(self.m+threads-1)//threads
        self.warps=threads//32

    @cute.jit
    def __call__(self,logp,assignments,counts,ranks,offsets,ids,weights,inverse,stream):
        grid=(cute.ceil_div(self.n,self.warps),1,1)
        if cutlass.const_expr(self.mode=='count'):grid=(self.chunks,self.e,1)
        if cutlass.const_expr(self.mode=='prefix'):grid=(self.e,1,1)
        self.kernel(logp,assignments,counts,ranks,offsets,ids,weights,inverse).launch(
            grid=grid,block=(self.threads,1,1),stream=stream)

    @cute.kernel
    def kernel(self,logp,assignments,counts,ranks,offsets,ids,weights,inverse):
        tidx,_,_=cute.arch.thread_idx()
        block,expert,_=cute.arch.block_idx()
        lane=tidx%32;warp=tidx//32
        if cutlass.const_expr(self.mode=='count'):
            index=block*self.threads+tidx
            match=Boolean(False)
            if index<self.m:match=assignments[index]==expert
            ballot=Uint32(cute.arch.vote_ballot_sync(match))
            local=Int32(cute.arch.popc(ballot&((Uint32(1)<<Uint32(lane))-Uint32(1))))
            smem=utils.SmemAllocator()
            totals=smem.allocate_tensor(Int32,cute.make_layout((self.warps,)))
            if lane==0:totals[warp]=Int32(cute.arch.popc(ballot))
            cute.arch.sync_threads()
            for prior in cutlass.range_constexpr(self.warps):
                if warp>prior:local+=totals[prior]
            if match:ranks[index]=local
            if tidx==0:
                total=Int32(0)
                for prior in cutlass.range_constexpr(self.warps):total+=totals[prior]
                counts[expert*self.chunks+block]=total
        elif cutlass.const_expr(self.mode=='prefix'):
            smem=utils.SmemAllocator()
            scratch=smem.allocate_tensor(Int32,cute.make_layout((self.warps+1,)))
            if tidx==0:scratch[self.warps]=Int32(0)
            cute.arch.sync_threads()
            for tile in cutlass.range(0,cute.ceil_div(self.chunks,self.threads),unroll=1):
                index=tile*self.threads+tidx
                value=Int32(0)
                if index<self.chunks:value=counts[block*self.chunks+index]
                total=value
                for level in cutlass.range_constexpr(5):
                    previous=cute.arch.shuffle_sync_up(total,1<<level)
                    if lane>=1<<level:total+=previous
                if lane==31:scratch[warp]=total
                cute.arch.sync_threads()
                for prior in cutlass.range_constexpr(self.warps):
                    if warp>prior:total+=scratch[prior]
                total+=scratch[self.warps]
                if index<self.chunks:offsets[block*self.chunks+index]=total-value
                cute.arch.sync_threads()
                if tidx==self.threads-1:scratch[self.warps]=total
                cute.arch.sync_threads()
        else:
            token=block*self.warps+warp
            value=Float32(float('-inf'));slot=Int32(0)
            if token<self.n and lane<self.k:
                index=token*self.k+lane
                selected=Int32(assignments[index])
                slot=selected*self.capacity+offsets[selected*self.chunks+index//self.threads]+ranks[index]
                value=logp[token*self.e+selected]
            maximum=value
            for level in cutlass.range_constexpr(5):
                maximum=cute.arch.fmax(maximum,cute.arch.shuffle_sync_bfly(maximum,1<<level))
            probability=cute.exp(value-maximum)
            total=probability
            for level in cutlass.range_constexpr(5):total+=cute.arch.shuffle_sync_bfly(total,1<<level)
            if token<self.n and lane<self.k:
                ids[slot]=Int64(token if token<self.real else self.real-1)
                weights[slot]=probability/total if token<self.real else Float32(0)
                if token<self.real:inverse[token*self.k+lane]=Int64(slot)


class MetadataBackwardKernel:
    def __init__(self,n,e,k,real,threads):
        self.n,self.e,self.k,self.real,self.threads=n,e,k,real,threads
        self.capacity=n*k//e;self.warps=threads//32

    @cute.jit
    def __call__(self,gradient,weights,inverse,out,stream):
        self.kernel(gradient,weights,inverse,out).launch(
            grid=(cute.ceil_div(self.n,self.warps),1,1),block=(self.threads,1,1),stream=stream)

    @cute.kernel
    def kernel(self,gradient,weights,inverse,out):
        tidx,_,_=cute.arch.thread_idx();block,_,_=cute.arch.block_idx()
        lane=tidx%32;token=block*self.warps+tidx//32
        value=Float32(0);probability=Float32(0);expert=Int32(0)
        if token<self.real and lane<self.k:
            slot=Int32(inverse[token*self.k+lane])
            expert=slot//self.capacity
            probability=weights[slot]
            value=gradient[slot]
        value-=cute.arch.shuffle_sync(value,0)
        total=value*probability
        for level in cutlass.range_constexpr(5):total+=cute.arch.shuffle_sync_bfly(total,1<<level)
        if token<self.n:
            for column in cutlass.range(lane,self.e,32,unroll=1):
                out[token*self.e+column]=Float32(0)
        cute.arch.sync_warp()
        if token<self.real and lane<self.k:
            out[token*self.e+expert]=probability*(value-total)


def _fake(dtype,count):
    return cute.runtime.make_fake_tensor(dtype,(count,),stride=(1,),assumed_align=8 if dtype==Int64 else 4)


@lru_cache(maxsize=128)
def compile_metadata_forward(n,e,k,real,threads):
    m=n*k;chunks=(m+threads-1)//threads
    examples=(_fake(Float32,n*e),_fake(Int64,m),_fake(Int32,e*chunks),_fake(Int32,m),
              _fake(Int32,e*chunks),_fake(Int64,m),_fake(Float32,m),_fake(Int64,real*k))
    stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return tuple(cute.compile(StableMetadataKernel(n,e,k,real,threads,mode),*examples,stream,
                             options='--enable-tvm-ffi --opt-level 3')
                 for mode in ('count','prefix','emit'))


@lru_cache(maxsize=128)
def compile_metadata_backward(n,e,k,real,threads):
    stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(MetadataBackwardKernel(n,e,k,real,threads),_fake(Float32,n*k),
                        _fake(Float32,n*k),_fake(Int64,real*k),_fake(Float32,n*e),stream,
                        options='--enable-tvm-ffi --opt-level 3')
