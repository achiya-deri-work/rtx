"""CuTe radix order statistics and packed, stable per-expert quota selection.

No CUB, NVCC or CUDA C++ dependency. All hard decisions are nondifferentiable.
"""
from functools import lru_cache
import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Float32,Int32,Int64,Uint32,Uint64,Boolean
from cutlass.experimental.primitives import nvvm_wrapper as nvvm


@cute.jit
def ordered_float(value:Float32):
    value=Float32(0) if value==Float32(0) else value
    bits=nvvm.inline_ptx_hl('mov.b32 {$w0}, {$r0};',write_only_types=[Uint32],read_only_args=[value])
    result=bits^Uint32(0x80000000)
    if (bits&Uint32(0x80000000))!=0:result=~bits
    if value!=value:result=Uint32(0xffffffff)
    return result


@cute.jit
def unordered_float(key:Uint32):
    bits=~key
    if (key&Uint32(0x80000000))!=0:bits=key^Uint32(0x80000000)
    return nvvm.inline_ptx_hl('mov.b32 {$w0}, {$r0};',write_only_types=[Float32],read_only_args=[bits])


@cute.jit
def histogram_add(hist,digit,tidx):
    # One shared atomic per equal-digit warp group, including uniform inputs.
    peers=nvvm.inline_ptx_hl('match.any.sync.b32 {$w0}, {$r0}, 0xffffffff;',
                             write_only_types=[Uint32],read_only_args=[Uint32(digit)])
    previous=(Uint32(1)<<Uint32(tidx%32))-Uint32(1)
    if digit<16 and cute.arch.popc(peers&previous)==0:
        cute.arch.atomic_add(hist.iterator+digit,Int32(cute.arch.popc(peers)))


@cute.jit
def update_radix(hist,prefix,rank,tidx,shift):
    cute.arch.sync_threads()
    if tidx==0:
        remaining=rank[0]
        chosen=Int32(0)
        active=Boolean(True)
        for i in cutlass.range_constexpr(16):
            digit=15-i
            if active:
                count=hist[digit]
                if remaining>=count:remaining-=count
                else:
                    chosen=Int32(digit)
                    active=Boolean(False)
        rank[0]=remaining
        prefix[0]=prefix[0]|(Uint64(chosen)<<Uint64(shift))
    cute.arch.sync_threads()


class QuantileKernel:
    def __init__(self,n,e,rank,threads,column):
        self.n,self.e,self.rank,self.threads,self.column=n,e,rank,threads,column
        self.length=n if column else e
        self.vectors=e if column else n
        self.items=math.ceil(self.length/threads)
        self.cached=self.items<=64

    @cute.jit
    def __call__(self,src,dst,stream:cuda.CUstream):
        self.kernel(src,dst).launch(grid=(self.vectors,1,1),block=(self.threads,1,1),stream=stream)

    @cute.jit
    def key(self,src,vector,element):
        key=Uint64(0)
        if element<self.length:
            index=element*self.e+vector if self.column else vector*self.e+element
            key=Uint64(ordered_float(src[index]))
        return key

    @cute.kernel
    def kernel(self,src,dst):
        tidx,_,_=cute.arch.thread_idx()
        vector,_,_=cute.arch.block_idx()
        smem=utils.SmemAllocator()
        hist=smem.allocate_tensor(Int32,cute.make_layout((16,)))
        prefix=smem.allocate_tensor(Uint64,cute.make_layout((1,)))
        rank=smem.allocate_tensor(Int32,cute.make_layout((1,)))
        if tidx==0:prefix[0]=Uint64(0);rank[0]=Int32(self.rank)
        if cutlass.const_expr(self.cached):
            keys=cute.make_rmem_tensor((self.items,),Uint64)
            for j in cutlass.range_constexpr(self.items):keys[j]=self.key(src,vector,tidx+j*self.threads)
        cute.arch.sync_threads()
        for step in cutlass.range(0,8,unroll=1):
            if tidx<16:hist[tidx]=Int32(0)
            cute.arch.sync_threads()
            shift=28-step*4
            mask=Uint64(0xffffffff)^((Uint64(1)<<Uint64(shift+4))-Uint64(1))
            wanted=prefix[0]
            if cutlass.const_expr(self.cached):
                for j in cutlass.range_constexpr(self.items):
                    key=keys[j]
                    digit=Int32(16)
                    if tidx+j*self.threads<self.length and (key&mask)==wanted:
                        digit=Int32((key>>Uint64(shift))&Uint64(15))
                    histogram_add(hist,digit,tidx)
            else:
                for j in cutlass.range(0,self.items,unroll=1):
                    key=self.key(src,vector,tidx+j*self.threads)
                    digit=Int32(16)
                    if tidx+j*self.threads<self.length and (key&mask)==wanted:
                        digit=Int32((key>>Uint64(shift))&Uint64(15))
                    histogram_add(hist,digit,tidx)
            update_radix(hist,prefix,rank,tidx,shift)
        if tidx==0:dst[vector]=unordered_float(Uint32(prefix[0]))


@lru_cache(maxsize=128)
def compile_quantile(n,e,rank,threads,column):
    src=cute.runtime.make_fake_tensor(Float32,(n*e,),stride=(1,),assumed_align=16)
    dst=cute.runtime.make_fake_tensor(Float32,(e if column else n,),stride=(1,),assumed_align=4)
    return cute.compile(QuantileKernel(n,e,rank,threads,column),src,dst,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),options='--enable-tvm-ffi --opt-level 3')


class WarpQuantileKernel:
    """Exact row order statistic, one warp/row and no CTA-wide barriers.

    Sorting operates on the same ordered FP32 keys as radix selection. Padding
    keys are below every valid ordered value; descending rank is unchanged.
    """
    def __init__(self,n,e,rank,threads):
        if n<1 or not 1<=e<=128 or not 0<=rank<e or threads not in (64,128,256):
            raise ValueError('Warp quantile requires N>0, E=1..128, a valid rank and 64/128/256 threads')
        self.n,self.e,self.rank,self.threads=n,e,rank,threads
        self.width=max(32,1<<(e-1).bit_length())
        self.items=self.width//32
        self.levels=self.width.bit_length()-1

    @cute.jit
    def __call__(self,src,dst,stream:cuda.CUstream):
        self.kernel(src,dst).launch(grid=(cute.ceil_div(self.n,self.threads//32),1,1),
                                    block=(self.threads,1,1),stream=stream)

    @cute.kernel
    def kernel(self,src,dst):
        tidx,_,_=cute.arch.thread_idx()
        block,_,_=cute.arch.block_idx()
        lane=tidx%32
        row=block*(self.threads//32)+tidx//32
        keys=cute.make_rmem_tensor((self.items,),Uint32)
        updated=cute.make_rmem_tensor((self.items,),Uint32)
        for item in cutlass.range_constexpr(self.items):
            column=lane+item*32
            key=Uint32(0)
            if row<self.n and column<self.e:key=ordered_float(src[row*self.e+column])
            keys[item]=key
        for level in cutlass.range_constexpr(self.levels):
            width=1<<(level+1)
            for part in cutlass.range_constexpr(level+1):
                distance=1<<(level-part)
                for item in cutlass.range_constexpr(self.items):
                    value=keys[item]
                    if cutlass.const_expr(distance<32):
                        peer=cute.arch.shuffle_sync_bfly(value,distance)
                    else:
                        peer=keys[item^(distance//32)]
                    index=lane+item*32
                    ascending=(index&width)==0
                    lower=(index&distance)==0
                    lo=value if value<peer else peer
                    hi=value if value>peer else peer
                    updated[item]=lo if ascending==lower else hi
                # Register-bank distances must read the old bank throughout
                # each comparator layer, not a partner already overwritten.
                for item in cutlass.range_constexpr(self.items):keys[item]=updated[item]
        target=self.width-1-self.rank
        if row<self.n and lane==target%32:
            dst[row]=unordered_float(keys[target//32])


@lru_cache(maxsize=128)
def compile_warp_quantile(n,e,rank,threads):
    src=cute.runtime.make_fake_tensor(Float32,(n*e,),stride=(1,),assumed_align=4)
    dst=cute.runtime.make_fake_tensor(Float32,(n,),stride=(1,),assumed_align=4)
    return cute.compile(WarpQuantileKernel(n,e,rank,threads),src,dst,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),options='--enable-tvm-ffi --opt-level 3')


class QuotaKernel:
    def __init__(self,m,e,k,threads,mode,target=False):
        self.m,self.e,self.k,self.threads=m,e,k,threads
        self.mode,self.target=mode,target

    @cute.jit
    def __call__(self,sources,targets,gains,surplus,deficit,slots,loads,phase,
                 counts,offsets,keys,thresholds,source_mask,target_mask,result,new_loads,new_phase,
                 stream:cuda.CUstream):
        blocks=cute.ceil_div(self.m,self.threads)
        if cutlass.const_expr(self.mode=='init'):blocks=cute.ceil_div(self.e,self.threads)
        if cutlass.const_expr(self.mode=='prefix'):blocks=1
        if cutlass.const_expr(self.mode=='select'):blocks=self.e
        if cutlass.const_expr(self.mode=='mark' and not self.target):
            blocks=cute.ceil_div(max(self.m,self.e),self.threads)
        self.kernel(sources,targets,gains,surplus,deficit,slots,loads,phase,counts,offsets,keys,
                    thresholds,source_mask,target_mask,result,new_loads,new_phase).launch(
            grid=(blocks,1,1),block=(self.threads,1,1),stream=stream)

    @cute.jit
    def active(self,groups,gains,quotas,source_mask,index):
        active=Boolean(False)
        if index<self.m:
            group=groups[index]
            if group>=0 and group<self.e:
                active=quotas[group]>0 and gains[index]!=Float32(float('-inf'))
                if cutlass.const_expr(self.target):active=active and source_mask[index]
        return active

    @cute.jit
    def key(self,gains,index):
        return (Uint64(ordered_float(gains[index]))<<Uint64(32))|Uint64(Uint32(0xffffffff)-Uint32(index))

    @cute.kernel
    def kernel(self,sources,targets,gains,surplus,deficit,slots,loads,phase,
               counts,offsets,keys,thresholds,source_mask,target_mask,result,new_loads,new_phase):
        tidx,_,_=cute.arch.thread_idx()
        block,_,_=cute.arch.block_idx()
        index=block*self.threads+tidx
        groups=targets if self.target else sources
        quotas=deficit if self.target else surplus
        if cutlass.const_expr(self.mode=='init'):
            if index<self.e:counts[index]=Int32(0);new_loads[index]=loads[index]
            if index==0:new_phase[0]=phase[0]+Int64(1)
        elif cutlass.const_expr(self.mode=='count'):
            if self.active(groups,gains,quotas,source_mask,index):
                cute.arch.atomic_add(counts.iterator+groups[index],Int32(1))
        elif cutlass.const_expr(self.mode=='prefix'):
            smem=utils.SmemAllocator()
            scratch=smem.allocate_tensor(Int32,cute.make_layout((self.threads//32+1,)))
            if tidx==0:scratch[self.threads//32]=Int32(0)
            cute.arch.sync_threads()
            for tile in cutlass.range(0,cute.ceil_div(self.e,self.threads),unroll=1):
                i=tile*self.threads+tidx
                value=Int32(0)
                if i<self.e:value=counts[i]
                total=value
                for level in cutlass.range_constexpr(5):
                    previous=cute.arch.shuffle_sync_up(total,1<<level)
                    if tidx%32>=1<<level:total+=previous
                if tidx%32==31:scratch[tidx//32]=total
                cute.arch.sync_threads()
                for warp in cutlass.range_constexpr(self.threads//32):
                    if tidx//32>warp:total+=scratch[warp]
                total+=scratch[self.threads//32]
                if i<self.e:offsets[i]=total-value;counts[i]=Int32(0)
                cute.arch.sync_threads()
                if tidx==self.threads-1:scratch[self.threads//32]=total
                cute.arch.sync_threads()
            if tidx==0:offsets[self.e]=scratch[self.threads//32]
        elif cutlass.const_expr(self.mode=='scatter'):
            if self.active(groups,gains,quotas,source_mask,index):
                group=groups[index]
                position=cute.arch.atomic_add(counts.iterator+group,Int32(1))
                keys[offsets[group]+position]=self.key(gains,index)
        elif cutlass.const_expr(self.mode=='select'):
            count=offsets[block+1]-offsets[block]
            take=Int32(quotas[block])
            if count>take and take>0:
                smem=utils.SmemAllocator()
                hist=smem.allocate_tensor(Int32,cute.make_layout((16,)))
                prefix=smem.allocate_tensor(Uint64,cute.make_layout((1,)))
                rank=smem.allocate_tensor(Int32,cute.make_layout((1,)))
                if tidx==0:prefix[0]=Uint64(0);rank[0]=take-1
                cute.arch.sync_threads()
                for step in cutlass.range(0,16,unroll=1):
                    if tidx<16:hist[tidx]=Int32(0)
                    cute.arch.sync_threads()
                    shift=60-step*4
                    mask=Uint64(0)
                    if step>0:mask=(~Uint64(0))<<Uint64(shift+4)
                    wanted=prefix[0]
                    for j in cutlass.range(0,cute.ceil_div(count,self.threads),unroll=1):
                        element=tidx+j*self.threads
                        digit=Int32(16)
                        if element<count:
                            key=keys[offsets[block]+element]
                            if (key&mask)==wanted:digit=Int32((key>>Uint64(shift))&Uint64(15))
                        histogram_add(hist,digit,tidx)
                    update_radix(hist,prefix,rank,tidx,shift)
                if tidx==0:thresholds[block]=prefix[0]
            else:
                if tidx==0:thresholds[block]=Uint64(0)
        elif cutlass.const_expr(self.mode=='mark'):
            # Source packing used counts as per-expert cursors. Target
            # counting must start from zero, even when the source and target
            # quota sets are disjoint: stale counts shift target offsets and
            # can exceed the M-key workspace under severe valid imbalance.
            # No mark thread reads counts, so the reset needs no extra launch
            # or cross-CTA synchronization inside this stage.
            if cutlass.const_expr(not self.target):
                if index<self.e:counts[index]=Int32(0)
            if index<self.m:
                accept=Boolean(False)
                if self.active(groups,gains,quotas,source_mask,index):
                    gain=gains[index]
                    accept=(self.key(gains,index)>=thresholds[groups[index]] and gain==gain
                            and gain!=Float32(float('inf')) and gain!=Float32(float('-inf')))
                if cutlass.const_expr(self.target):target_mask[index]=accept
                else:source_mask[index]=accept
        elif cutlass.const_expr(self.mode=='direct'):
            if index<self.m:
                accept=Boolean(False)
                if self.active(groups,gains,quotas,source_mask,index):
                    gain=gains[index]
                    if gain==gain and gain!=Float32(float('inf')) and gain!=Float32(float('-inf')):
                        group=groups[index]
                        key=self.key(gains,index)
                        remaining=Int32(quotas[group])
                        j=Int32(0)
                        while j<self.m and remaining>0:
                            if groups[j]==group:
                                if self.active(groups,gains,quotas,source_mask,j):
                                    if self.key(gains,j)>key:remaining-=1
                            j+=1
                        accept=remaining>0
                if cutlass.const_expr(self.target):target_mask[index]=accept
                else:source_mask[index]=accept
        else:
            if index<self.m:
                accepted=target_mask[index]
                result[(index//self.k)*self.k+slots[index]]=targets[index] if accepted else sources[index]
                if accepted:
                    cute.arch.atomic_add(new_loads.iterator+sources[index],Int64(-1))
                    cute.arch.atomic_add(new_loads.iterator+targets[index],Int64(1))


@lru_cache(maxsize=128)
def compile_quota(m,e,k,threads=128,segmented=True):
    def fake(dtype,size):return cute.runtime.make_fake_tensor(dtype,(size,),stride=(1,),assumed_align=8 if dtype in (Int64,Uint64) else 4 if dtype!=Boolean else 1)
    tensors=[fake(Int64,m),fake(Int64,m),fake(Float32,m),fake(Int64,e),fake(Int64,e),
             fake(Int64,m),fake(Int64,e),fake(Int64,1),fake(Int32,e),fake(Int32,e+1),
             fake(Uint64,m),fake(Uint64,e),fake(Boolean,m),fake(Boolean,m),fake(Int64,m),
             fake(Int64,e),fake(Int64,1)]
    stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    stages=[('init',False)]
    for target in (False,True):
        stages.extend((mode,target) for mode in (('count','prefix','scatter','select','mark') if segmented else ('direct',)))
    stages.append(('apply',True))
    return tuple(cute.compile(QuotaKernel(m,e,k,threads,mode,target),*tensors,stream,
                              options='--enable-tvm-ffi --opt-level 3') for mode,target in stages)


@cute.extern(name='cudaGraphSetConditional',inline=False)
def set_graph_condition(handle:Uint64,value:Uint32)->None:
    ...


class ConditionKernel:
    def __init__(self,e,capacity,rounds):self.e,self.capacity,self.rounds=e,capacity,rounds

    @cute.jit
    def __call__(self,loads,phase,handle:Uint64,stream:cuda.CUstream):
        self.kernel(loads,phase,handle).launch(grid=(1,1,1),block=(128,1,1),stream=stream)

    @cute.kernel
    def kernel(self,loads,phase,handle:Uint64):
        tidx,_,_=cute.arch.thread_idx()
        smem=utils.SmemAllocator()
        scratch=smem.allocate_tensor(Int32,cute.make_layout((4,)))
        unfinished=Int32(0)
        for i in cutlass.range(tidx,self.e,128,unroll=1):
            if loads[i]!=self.capacity:unfinished=Int32(1)
        any_warp=cute.arch.vote_any_sync(unfinished!=0)
        if tidx%32==0:scratch[tidx//32]=Int32(any_warp)
        cute.arch.sync_threads()
        if tidx==0:
            flag=Int32(0)
            for i in cutlass.range_constexpr(4):flag=flag|scratch[i]
            set_graph_condition(handle,Uint32(flag!=0 and phase[0]<self.rounds))


@lru_cache(maxsize=64)
def compile_condition(e,capacity,rounds):
    def fake(size):return cute.runtime.make_fake_tensor(Int64,(size,),stride=(1,),assumed_align=8)
    return cute.compile(ConditionKernel(e,capacity,rounds),fake(e),fake(1),Uint64(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),options='--enable-tvm-ffi --opt-level 3')
