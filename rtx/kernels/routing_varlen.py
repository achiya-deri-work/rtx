"""Packed sequence-local routing primitives; boundaries stay on the device.

No B x max_length activation tensor. Only the routing matrix gets bounded
logical dummy rows. Expert segments are separately padded to 32 slots so
transposed MXFP8 quantization cannot join two sequences' reduction blocks.
"""
from functools import lru_cache
import math

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Float32, Int32, Int64, Uint32, Uint64

from .routing_sinkhorn import SinkhornKernel
from .routing_selection import histogram_add, ordered_float, unordered_float, update_radix, set_graph_condition
from .routing_metadata import StableMetadataKernel


class LayoutKernel:
    def __init__(self,t,b,e,k,n,threads,mode,strict=False):
        self.t,self.b,self.e,self.k,self.n,self.threads,self.mode=t,b,e,k,n,threads,mode
        self.multiple=e//math.gcd(e,k);self.strict=strict

    @cute.jit
    def __call__(self,source,cu,scheduled,physical,sequence,local,rowmap,packed,status,stream):
        blocks=1 if self.mode=='prefix' else cute.ceil_div(self.n*self.e,self.threads)
        self.kernel(source,cu,scheduled,physical,sequence,local,rowmap,packed,status).launch(
            grid=(blocks,1,1),block=(self.threads,1,1),stream=stream)

    @cute.kernel
    def kernel(self,source,cu,scheduled,physical,sequence,local,rowmap,packed,status):
        tidx,_,_=cute.arch.thread_idx();block,_,_=cute.arch.block_idx()
        if cutlass.const_expr(self.mode=='prefix'):
            if tidx==0:
                valid=Int32(1)
                if cu[0]!=0 or cu[self.b]!=self.t:valid=Int32(0)
                for s in cutlass.range(self.b,unroll=1):
                    a=cu[s];z=cu[s+1]
                    if a<0 or z<a or z>self.t:valid=Int32(0)
                    if cutlass.const_expr(self.strict):
                        if (z-a)%self.multiple!=0:valid=Int32(0)
                scheduled[0]=Int32(0);physical[0]=Int32(0)
                total=Int32(0);slots=Int32(0)
                for s in cutlass.range(self.b,unroll=1):
                    length=Int32(0)
                    if valid!=0:length=cu[s+1]-cu[s]
                    ns=cute.ceil_div(length,self.multiple)*self.multiple
                    total+=ns;slots+=cute.ceil_div(ns*self.k//self.e,32)*32
                    scheduled[s+1]=total;physical[s+1]=slots
                status[0]=valid
        else:
            index=block*self.threads+tidx
            if index<self.n*self.e:
                row=index//self.e;expert=index%self.e
                s=Int32(0);position=Int32(0);mapping=Int32(-1);value=Float32(0)
                if row<scheduled[self.b]:
                    lo=Int32(0);hi=Int32(self.b)
                    while lo<hi:
                        mid=(lo+hi)//2
                        if scheduled[mid+1]<=row:lo=mid+1
                        else:hi=mid
                    s=lo;position=row-scheduled[s]
                    length=cu[s+1]-cu[s]
                    value=Float32(-math.log(2.))
                    mapping=-cu[s+1]  # -(safe dummy token + 1)
                    if position<length:
                        mapping=cu[s]+position
                        value=source[mapping*self.e+expert]
                        # Every real score is already visited here. A separate
                        # all-isfinite reduction can make Inductor serialize
                        # the entire affinity epilogue into one CTA. Invalid
                        # inputs alone pay an atomic; the prefix launch resets
                        # status before every invocation/replay.
                        if not (value>Float32(float('-inf')) and value<Float32(float('inf'))):
                            cute.arch.atomic_and(status.iterator,Int32(0))
                packed[index]=value
                if expert==0:
                    sequence[row]=Int64(s);local[row]=Int64(position);rowmap[row]=mapping


class VarlenSinkhorn(SinkhornKernel):
    def __init__(self,n,e,b,k,threads,groups,axis,mode,persistent_rows=0):
        super().__init__(n,e,axis=axis,threads=threads,groups=groups,mode=mode)
        self.b,self.k=b,k
        self.persistent_rows=persistent_rows

    @cute.jit
    def staged_pool(self,scheduled,seq):
        use=True
        if cutlass.const_expr(self.persistent_rows>0):
            use=scheduled[seq+1]-scheduled[seq]>self.persistent_rows
        return use

    @cute.jit
    def __call__(self,source,other,shift,history,out,probabilities,scheduled,sequence,rowmap,iteration:cutlass.Int32,stream):
        vectors=self.n if self.axis==0 else self.b*self.e
        blocks=cute.ceil_div(vectors,self.groups)
        if cutlass.const_expr(self.mode=='init'):blocks=cute.ceil_div(self.b*self.e,self.threads)
        if cutlass.const_expr(self.mode in ('finish','unpack')):blocks=cute.ceil_div(self.n*self.e,self.threads)
        self.kernel(source,other,shift,history,out,probabilities,scheduled,sequence,rowmap,iteration).launch(
            grid=(blocks,1,1),block=(self.threads,1,1),stream=stream)

    @cute.kernel
    def kernel(self,source,other,shift,history,out,probabilities,scheduled,sequence,rowmap,iteration:cutlass.Int32):
        tidx,_,_=cute.arch.thread_idx();block,_,_=cute.arch.block_idx()
        index=block*self.threads+tidx
        if cutlass.const_expr(self.mode=='init'):
            if index<self.b*self.e:other[index]=Float32(0)
        elif cutlass.const_expr(self.mode=='unpack'):
            if index<self.n*self.e:
                row=index//self.e;real=rowmap[row]
                if real>=0:
                    if self.staged_pool(scheduled,Int32(sequence[row])):
                        out[real*self.e+index%self.e]=source[index]
        elif cutlass.const_expr(self.mode=='finish'):
            if index<self.n*self.e:
                row=index//self.e;value=Float32(0)
                write=Int32(1 if self.persistent_rows==0 else 0)
                if row<scheduled[self.b]:
                    s=Int32(sequence[row])
                    if self.staged_pool(scheduled,s):
                        value=source[index]+shift[row]+other[s*self.e+index%self.e]
                        write=Int32(1)
                if write!=0:
                    out[index]=value
                    probabilities[index]=cute.exp(value)
        else:
            smem=utils.SmemAllocator()
            scratch=smem.allocate_tensor(Float32,cute.make_layout((self.groups*(self.threads//32),)))
            vector=block*self.groups+tidx%self.groups
            first=tidx//self.groups;step=self.threads//self.groups
            length=Int32(0);start=Int32(0);s=Int32(0);expert=Int32(0)
            if cutlass.const_expr(self.axis==0):
                if vector<scheduled[self.b]:
                    s=Int32(sequence[vector])
                    if self.staged_pool(scheduled,s):length=Int32(self.e)
            else:
                if vector<self.b*self.e:
                    s=vector//self.e;expert=vector%self.e
                    if self.staged_pool(scheduled,s):
                        start=scheduled[s];length=scheduled[s+1]-start
            base=(2*iteration+self.axis)*self.n*self.e
            if cutlass.const_expr(self.mode=='forward'):
                maximum=Float32(float('-inf'))
                for element in cutlass.range(first,length,step,unroll=1):
                    idx=vector*self.e+element if self.axis==0 else (start+element)*self.e+expert
                    oi=s*self.e+element if self.axis==0 else start+element
                    maximum=cute.arch.fmax(maximum,source[idx]+other[oi])
                maximum=self.reduce(maximum,scratch,tidx,True)
                total=Float32(0)
                for element in cutlass.range(first,length,step,unroll=1):
                    idx=vector*self.e+element if self.axis==0 else (start+element)*self.e+expert
                    oi=s*self.e+element if self.axis==0 else start+element
                    total+=cute.exp(source[idx]+other[oi]-maximum)
                total=self.reduce(total,scratch,tidx,False)
                normalizer=Float32(0)
                if length>0:normalizer=maximum+cute.log(total)
                if first==0:
                    limit=self.n if self.axis==0 else self.b*self.e
                    if vector<limit:
                        mass=Float32(self.k)
                        if cutlass.const_expr(self.axis==1):mass=Float32(length*self.k//self.e)
                        offset=Float32(0)
                        if length>0:offset=cute.log(mass)-normalizer
                        shift[vector]=offset
                for element in cutlass.range(first,length,step,unroll=1):
                    idx=vector*self.e+element if self.axis==0 else (start+element)*self.e+expert
                    oi=s*self.e+element if self.axis==0 else start+element
                    history[base+idx]=cute.exp(source[idx]+other[oi]-normalizer)
                if cutlass.const_expr(self.axis==0 and self.persistent_rows==0):
                    if vector<self.n and length==0:
                        # Both axes leave unused bound rows mathematically
                        # inactive. Initialize their saved state in this
                        # existing row pass, not a full-history memset.
                        for element in cutlass.range(first,self.e,step,unroll=1):
                            idx=vector*self.e+element
                            history[base+idx]=Float32(0)
                            history[base+self.n*self.e+idx]=Float32(0)
            else:
                total=Float32(0)
                for element in cutlass.range(first,length,step,unroll=1):
                    idx=vector*self.e+element if self.axis==0 else (start+element)*self.e+expert
                    total+=source[idx]
                total=self.reduce(total,scratch,tidx,False)
                # Unused upper-bound rows must be overwritten, not left as
                # stale adjoints from a previous graph replay.
                if cutlass.const_expr(self.axis==0 and self.persistent_rows==0):
                    if vector<self.n and length==0:
                        for element in cutlass.range(first,self.e,step,unroll=1):out[vector*self.e+element]=Float32(0)
                for element in cutlass.range(first,length,step,unroll=1):
                    idx=vector*self.e+element if self.axis==0 else (start+element)*self.e+expert
                    out[idx]=source[idx]-history[base+idx]*total


class WarpRowSinkhorn:
    """One contiguous row per warp, keeping its <=128 values in registers.

    Unlike the original multi-warp CTA reduction this needs no shared memory
    or block barriers. Columns retain the independently selected CTA geometry.
    The FP32 normalization/history contract and finite iteration count do not
    change; reduction order can differ just like other reduction geometries.
    """
    def __init__(self,n,e,b,k,threads,mode,persistent_rows=0):
        self.n,self.e,self.b,self.k,self.threads,self.mode=n,e,b,k,threads,mode
        self.items=cute.ceil_div(e,32);self.warps=threads//32
        self.persistent_rows=persistent_rows

    @cute.jit
    def __call__(self,source,other,shift,history,out,probabilities,scheduled,sequence,rowmap,iteration:cutlass.Int32,stream):
        self.kernel(source,other,shift,history,out,scheduled,sequence,iteration).launch(
            grid=(cute.ceil_div(self.n,self.warps),1,1),block=(self.threads,1,1),stream=stream)

    @cute.kernel
    def kernel(self,source,other,shift,history,out,scheduled,sequence,iteration:cutlass.Int32):
        tidx,_,_=cute.arch.thread_idx();block,_,_=cute.arch.block_idx()
        lane=tidx%32;row=block*self.warps+tidx//32
        active=row<scheduled[self.b];s=Int32(0)
        if active:s=Int32(sequence[row])
        owns=row<self.n
        if cutlass.const_expr(self.persistent_rows>0):
            if active:active=scheduled[s+1]-scheduled[s]>self.persistent_rows
            owns=active
        values=cute.make_rmem_tensor((self.items,),Float32)
        values.fill(Float32(float('-inf') if self.mode=='forward' else 0))
        maximum=Float32(float('-inf'));total=Float32(0)
        for item in cutlass.range_constexpr(self.items):
            column=lane+item*32
            if active and column<self.e:
                value=source[row*self.e+column]
                if cutlass.const_expr(self.mode=='forward'):value+=other[s*self.e+column]
                values[item]=value
            if cutlass.const_expr(self.mode=='forward'):maximum=cute.arch.fmax(maximum,values[item])
            else:total+=values[item]
        if cutlass.const_expr(self.mode=='forward'):
            for level in cutlass.range_constexpr(5):
                maximum=cute.arch.fmax(maximum,cute.arch.shuffle_sync_bfly(maximum,1<<level))
            for item in cutlass.range_constexpr(self.items):
                # Inactive warps must not manufacture NaNs from -inf - -inf.
                if active:total+=cute.exp(values[item]-maximum)
        for level in cutlass.range_constexpr(5):total+=cute.arch.shuffle_sync_bfly(total,1<<level)
        base=2*iteration*self.n*self.e
        normalizer=Float32(0)
        if cutlass.const_expr(self.mode=='forward'):
            if active:normalizer=maximum+cute.log(total)
            if owns and lane==0:
                shift[row]=cute.log(Float32(self.k))-normalizer if active else Float32(0)
        for item in cutlass.range_constexpr(self.items):
            column=lane+item*32;index=row*self.e+column
            if owns and column<self.e:
                if cutlass.const_expr(self.mode=='forward'):
                    result=Float32(0)
                    if active:result=cute.exp(values[item]-normalizer)
                    history[base+index]=result
                    if not active:history[base+self.n*self.e+index]=Float32(0)
                else:
                    result=Float32(0)
                    if active:result=values[item]-history[base+index]*total
                    out[index]=result


class PersistentPoolSinkhorn:
    """One CTA owns a whole bounded pool, including all saved FP32 states.

    Odd shared-memory pitch makes both warp-axis traversals bank-friendly.
    The flat stores and adjoint updates traverse global history coalesced.
    No CTA waits for another CTA or reads a different pool's workspaces.
    """
    def __init__(self,n,e,b,k,steps,max_rows,threads,backward,guard_capacity=False):
        self.n,self.e,self.b,self.k=n,e,b,k
        self.steps,self.max_rows,self.threads,self.backward=steps,max_rows,threads,backward
        self.pitch=e|1
        self.warps=threads//32
        self.guard_capacity=guard_capacity

    @cute.jit
    def __call__(self,source,history,out,scheduled,rowmap,probabilities,stream):
        self.kernel(source,history,out,scheduled,rowmap,probabilities).launch(
            grid=(self.b,1,1),block=(self.threads,1,1),stream=stream)

    @cute.jit
    def warp_reduce(self,value,maximum:cutlass.Constexpr):
        for level in cutlass.range_constexpr(5):
            peer=cute.arch.shuffle_sync_bfly(value,1<<level)
            if cutlass.const_expr(maximum):value=cute.arch.fmax(value,peer)
            else:value=value+peer
        return value

    @cute.jit
    def forward_axis(self,matrix,shifts,other,normalizers,length,tid,rows:cutlass.Constexpr):
        vectors=length if rows else self.e
        extent=self.e if rows else length
        mass=Float32(self.k) if rows else Float32(length*self.k//self.e)
        lane=tid%32
        for vector in cutlass.range(tid//32,vectors,self.warps,unroll=1):
            maximum=Float32(float('-inf'))
            for position in cutlass.range(lane,extent,32,unroll=1):
                address=vector*self.pitch+position if rows else position*self.pitch+vector
                maximum=cute.arch.fmax(maximum,matrix[address]+other[position])
            maximum=self.warp_reduce(maximum,True)
            total=Float32(0)
            for position in cutlass.range(lane,extent,32,unroll=1):
                address=vector*self.pitch+position if rows else position*self.pitch+vector
                total+=cute.exp(matrix[address]+other[position]-maximum)
            total=self.warp_reduce(total,False)
            if lane==0:
                normalizer=maximum+cute.log(total)
                shifts[vector]=cute.log(mass)-normalizer
                normalizers[vector]=normalizer

    @cute.jit
    def backward_axis(self,matrix,sums,history,base,start,length,tid,rows:cutlass.Constexpr):
        vectors=length if rows else self.e
        extent=self.e if rows else length
        lane=tid%32
        for vector in cutlass.range(tid//32,vectors,self.warps,unroll=1):
            total=Float32(0)
            for position in cutlass.range(lane,extent,32,unroll=1):
                address=vector*self.pitch+position if rows else position*self.pitch+vector
                total+=matrix[address]
            total=self.warp_reduce(total,False)
            if lane==0:sums[vector]=total
        cute.arch.sync_threads()
        for index in cutlass.range(tid,length*self.e,self.threads,unroll=1):
            row=index//self.e;column=index%self.e
            vector=row if rows else column
            address=row*self.pitch+column
            matrix[address]=matrix[address]-history[base+start*self.e+index]*sums[vector]
        cute.arch.sync_threads()

    @cute.kernel
    def kernel(self,source,history,out,scheduled,rowmap,probabilities):
        tid,_,_=cute.arch.thread_idx();seq,_,_=cute.arch.block_idx()
        start=scheduled[seq];length=scheduled[seq+1]-start
        smem=utils.SmemAllocator()
        matrix=smem.allocate_tensor(Float32,cute.make_layout((self.max_rows*self.pitch,)))
        if cutlass.const_expr(self.backward):
            sums=smem.allocate_tensor(Float32,cute.make_layout((max(self.max_rows,self.e),)))
        else:
            rows=smem.allocate_tensor(Float32,cute.make_layout((self.max_rows,)))
            columns=smem.allocate_tensor(Float32,cute.make_layout((self.e,)))
            normalizers=smem.allocate_tensor(Float32,cute.make_layout((max(self.max_rows,self.e),)))
        active=length>0
        if cutlass.const_expr(self.guard_capacity):
            active=active and length<=self.max_rows
        if active:
            for index in cutlass.range(tid,length*self.e,self.threads,unroll=1):
                matrix[index//self.e*self.pitch+index%self.e]=source[start*self.e+index]
            if cutlass.const_expr(not self.backward):
                for column in cutlass.range(tid,self.e,self.threads,unroll=1):
                    columns[column]=Float32(0)
            cute.arch.sync_threads()
            if cutlass.const_expr(self.backward):
                for step in cutlass.range(self.steps,unroll=1):
                    iteration=self.steps-1-step
                    base=2*iteration*self.n*self.e
                    self.backward_axis(matrix,sums,history,base+self.n*self.e,start,length,tid,False)
                    self.backward_axis(matrix,sums,history,base,start,length,tid,True)
                for index in cutlass.range(tid,length*self.e,self.threads,unroll=1):
                    row=index//self.e;column=index%self.e;real=rowmap[start+row]
                    if real>=0:out[real*self.e+column]=matrix[row*self.pitch+column]
            else:
                for iteration in cutlass.range(self.steps,unroll=1):
                    base=2*iteration*self.n*self.e+start*self.e
                    self.forward_axis(matrix,rows,columns,normalizers,length,tid,True)
                    cute.arch.sync_threads()
                    for index in cutlass.range(tid,length*self.e,self.threads,unroll=1):
                        row=index//self.e;column=index%self.e
                        history[base+index]=cute.exp(matrix[row*self.pitch+column]+columns[column]-normalizers[row])
                    cute.arch.sync_threads()
                    self.forward_axis(matrix,columns,rows,normalizers,length,tid,False)
                    cute.arch.sync_threads()
                    for index in cutlass.range(tid,length*self.e,self.threads,unroll=1):
                        row=index//self.e;column=index%self.e
                        value=matrix[row*self.pitch+column]+rows[row]
                        history[base+self.n*self.e+index]=cute.exp(value-normalizers[column])
                        if iteration==self.steps-1:
                            logp=value+columns[column]
                            out[start*self.e+index]=logp
                            probabilities[start*self.e+index]=cute.exp(logp)
                    cute.arch.sync_threads()
        if cutlass.const_expr(not self.backward):
            # Unused bound rows have no owner; stripe disjoint ranges over
            # all CTAs, even empty ones, so graph replay overwrites old state.
            first=scheduled[self.b]*self.e+seq*self.threads+tid
            for index in cutlass.range(first,self.n*self.e,self.b*self.threads,unroll=1):
                out[index]=Float32(0)
                probabilities[index]=Float32(1)
                for stage in cutlass.range(2*self.steps,unroll=1):
                    history[stage*self.n*self.e+index]=Float32(0)


class ShardedColumnSinkhorn(SinkhornKernel):
    """Two-launch segmented reduction with contiguous columns in each CTA.

    A fixed number of shards per sequence keeps CUDA capture shape-static.
    Each shard covers a runtime-bounded interval of that sequence only. The
    first launch writes one FP32 logsumexp/sum per shard and column. The
    second reduces the small partials locally in each shard's application CTA
    rather than add a third launch and an intermediate normalizer tensor.
    """
    def __init__(self,n,e,b,k,threads,groups,shards,mode,persistent_rows=0):
        super().__init__(n,e,axis=1,threads=threads,groups=groups*4,mode=mode)
        self.b,self.k,self.shards=b,k,shards
        self.tiles=cute.ceil_div(e,self.groups)
        self.row_threads=threads//self.groups
        self.persistent_rows=persistent_rows

    @cute.jit
    def __call__(self,source,row_shift,column_shift,history,out,scheduled,partial,iteration:cutlass.Int32,stream):
        grid=(self.b*self.tiles*self.shards,1,1)
        block=(self.threads,1,1)
        self.partials(source,row_shift,scheduled,partial).launch(grid=grid,block=block,stream=stream)
        self.apply(source,row_shift,column_shift,history,out,scheduled,partial,iteration).launch(
            grid=grid,block=block,stream=stream)

    @cute.jit
    def bounds(self,scheduled,tidx,block):
        s=block//(self.tiles*self.shards)
        shard=block//self.tiles%self.shards
        expert=(block%self.tiles)*self.groups+tidx%self.groups
        start=scheduled[s];length=scheduled[s+1]-start
        if cutlass.const_expr(self.persistent_rows>0):
            if length<=self.persistent_rows:length=Int32(0)
        rows=cute.ceil_div(length,self.shards*self.row_threads)*self.row_threads
        a=start+shard*rows
        z=cutlass.min(start+length,a+rows)
        return s,shard,expert,length,a,z

    @cute.kernel
    def partials(self,source,row_shift,scheduled,partial):
        tidx,_,_=cute.arch.thread_idx();block,_,_=cute.arch.block_idx()
        s,shard,expert,length,a,z=self.bounds(scheduled,tidx,block)
        if a<z:
            smem=utils.SmemAllocator()
            scratch=smem.allocate_tensor(Float32,cute.make_layout((self.groups*(self.threads//32),)))
            first=a+tidx//self.groups
            total=Float32(0)
            if cutlass.const_expr(self.mode=='forward'):
                maximum=Float32(float('-inf'))
                if expert<self.e:
                    for row in cutlass.range(first,z,self.row_threads,unroll=1):
                        maximum=cute.arch.fmax(maximum,source[row*self.e+expert]+row_shift[row])
                maximum=self.reduce(maximum,scratch,tidx,True)
                if expert<self.e:
                    for row in cutlass.range(first,z,self.row_threads,unroll=1):
                        total+=cute.exp(source[row*self.e+expert]+row_shift[row]-maximum)
                total=self.reduce(total,scratch,tidx,False)
                if tidx<self.groups and expert<self.e:
                    partial[(s*self.shards+shard)*self.e+expert]=maximum+cute.log(total)
            else:
                if expert<self.e:
                    for row in cutlass.range(first,z,self.row_threads,unroll=1):
                        total+=source[row*self.e+expert]
                total=self.reduce(total,scratch,tidx,False)
                if tidx<self.groups and expert<self.e:
                    partial[(s*self.shards+shard)*self.e+expert]=total
        elif tidx<self.groups and expert<self.e:
            # Include empty shards in the same fixed-grid reduction without
            # sharing data across sequences or touching an uninitialized slot.
            partial[(s*self.shards+shard)*self.e+expert]=Float32(float('-inf') if self.mode=='forward' else 0)

    @cute.kernel
    def apply(self,source,row_shift,column_shift,history,out,scheduled,partial,iteration:cutlass.Int32):
        tidx,_,_=cute.arch.thread_idx();block,_,_=cute.arch.block_idx()
        s,shard,expert,length,a,z=self.bounds(scheduled,tidx,block)
        if a<z:
            smem=utils.SmemAllocator()
            scratch=smem.allocate_tensor(Float32,cute.make_layout((self.groups*(self.threads//32),)))
            first=tidx//self.groups
            base=(2*iteration+1)*self.n*self.e
            start=(s*self.shards)*self.e+expert
            total=Float32(0)
            if cutlass.const_expr(self.mode=='forward'):
                maximum=Float32(float('-inf'))
                if expert<self.e:
                    for part in cutlass.range(first,self.shards,self.row_threads,unroll=1):
                        maximum=cute.arch.fmax(maximum,partial[start+part*self.e])
                maximum=self.reduce(maximum,scratch,tidx,True)
                if expert<self.e:
                    for part in cutlass.range(first,self.shards,self.row_threads,unroll=1):
                        total+=cute.exp(partial[start+part*self.e]-maximum)
                total=self.reduce(total,scratch,tidx,False)
                if expert<self.e:
                    normalizer=maximum+cute.log(total)
                    if shard==0 and tidx<self.groups:
                        column_shift[s*self.e+expert]=cute.log(Float32(length*self.k//self.e))-normalizer
                    for row in cutlass.range(a+first,z,self.row_threads,unroll=1):
                        index=row*self.e+expert
                        history[base+index]=cute.exp(source[index]+row_shift[row]-normalizer)
            else:
                if expert<self.e:
                    for part in cutlass.range(first,self.shards,self.row_threads,unroll=1):
                        total+=partial[start+part*self.e]
                total=self.reduce(total,scratch,tidx,False)
                if expert<self.e:
                    for row in cutlass.range(a+first,z,self.row_threads,unroll=1):
                        index=row*self.e+expert
                        out[index]=source[index]-history[base+index]*total
        elif cutlass.const_expr(self.mode=='forward'):
            if length==0 and shard==0 and tidx<self.groups and expert<self.e:
                column_shift[s*self.e+expert]=Float32(0)


class ColumnQuantile:
    def __init__(self,n,b,e,k,threads,cache_items=0,expert_keys=False):
        self.n,self.b,self.e,self.k,self.threads=n,b,e,k,threads
        self.cache_items,self.expert_keys=cache_items,expert_keys

    @cute.jit
    def __call__(self,source,scheduled,out,stream):
        self.kernel(source,scheduled,out).launch(grid=(self.b*self.e,1,1),block=(self.threads,1,1),stream=stream)

    @cute.jit
    def key(self,source,row,expert):
        if cutlass.const_expr(self.expert_keys):return source[expert*self.n+row]
        else:return ordered_float(source[row*self.e+expert])

    @cute.kernel
    def kernel(self,source,scheduled,out):
        tidx,_,_=cute.arch.thread_idx();vector,_,_=cute.arch.block_idx()
        s=vector//self.e;expert=vector%self.e
        start=scheduled[s];length=scheduled[s+1]-start
        if length>0:
            smem=utils.SmemAllocator()
            hist=smem.allocate_tensor(Int32,cute.make_layout((16,)))
            prefix=smem.allocate_tensor(Uint64,cute.make_layout((1,)))
            rank=smem.allocate_tensor(Int32,cute.make_layout((1,)))
            cached=cute.make_rmem_tensor((max(1,self.cache_items),),Uint32)
            cached.fill(Uint32(0))
            if cutlass.const_expr(self.cache_items>0):
                if length<=self.threads*self.cache_items:
                    for item in cutlass.range_constexpr(self.cache_items):
                        element=tidx+item*self.threads
                        if element<length:cached[item]=self.key(source,start+element,expert)
            if tidx==0:prefix[0]=Uint64(0);rank[0]=length*self.k//self.e
            cute.arch.sync_threads()
            for step in cutlass.range(8,unroll=1):
                if tidx<16:hist[tidx]=Int32(0)
                cute.arch.sync_threads()
                shift=28-step*4;mask=Uint64(0)
                if step>0:mask=(~Uint64(0))<<Uint64(shift+4)
                wanted=prefix[0]
                if length<=self.threads*self.cache_items:
                    for item in cutlass.range_constexpr(self.cache_items):
                        element=tidx+item*self.threads;digit=Int32(16);key=Uint64(cached[item])
                        if element<length and (key&mask)==wanted:digit=Int32((key>>Uint64(shift))&Uint64(15))
                        histogram_add(hist,digit,tidx)
                else:
                    # The bounded cache never truncates longer segments.
                    for item in cutlass.range(0,cute.ceil_div(length,self.threads),unroll=1):
                        element=tidx+item*self.threads;digit=Int32(16)
                        if element<length:
                            key=Uint64(self.key(source,start+element,expert))
                            if (key&mask)==wanted:digit=Int32((key>>Uint64(shift))&Uint64(15))
                        histogram_add(hist,digit,tidx)
                update_radix(hist,prefix,rank,tidx,shift)
            if tidx==0:out[vector]=unordered_float(Uint32(prefix[0]))
        else:
            if tidx==0:out[vector]=Float32(0)


class ExpertKeysColumnQuantile:
    """Exact radix over contiguous ordered keys rebuilt for every QB round."""
    def __init__(self,n,b,e,k,threads,cache_items):
        self.n,self.e=n,e
        self.expert_tiles=cute.ceil_div(e,32)
        self.select=ColumnQuantile(n,b,e,k,threads,cache_items,expert_keys=True)

    @cute.jit
    def __call__(self,source,scheduled,out,keys,stream):
        self.transpose(source,keys).launch(grid=(cute.ceil_div(self.n,32)*self.expert_tiles,1,1),
            block=(128,1,1),stream=stream)
        self.select(keys,scheduled,out,stream)

    @cute.kernel
    def transpose(self,source,keys):
        tidx,_,_=cute.arch.thread_idx();block,_,_=cute.arch.block_idx()
        row_start=block//self.expert_tiles*32;expert_start=block%self.expert_tiles*32
        row=tidx//32;col=tidx%32
        smem=utils.SmemAllocator()
        tile=smem.allocate_tensor(Uint32,cute.make_layout((32*33,)))
        # Both passes are contiguous within a warp. Padding removes shared
        # bank conflicts in the transposed read. Tokens never cross a pool;
        # the per-pool span is read from scheduled in the radix pass.
        for item in cutlass.range_constexpr(8):
            r=row+item*4
            if row_start+r<self.n and expert_start+col<self.e:
                tile[r*33+col]=ordered_float(source[(row_start+r)*self.e+expert_start+col])
        cute.arch.sync_threads()
        for item in cutlass.range_constexpr(8):
            c=row+item*4
            if row_start+col<self.n and expert_start+c<self.e:
                keys[(expert_start+c)*self.n+row_start+col]=tile[col*33+c]


class FinalizeAssignments:
    def __init__(self,n,b,e,k,threads):self.n,self.b,self.e,self.k,self.threads=n,b,e,k,threads

    @cute.jit
    def __call__(self,current,scheduled,out,stream):
        self.kernel(current,scheduled,out).launch(grid=(cute.ceil_div(self.n*self.k,self.threads),1,1),block=(self.threads,1,1),stream=stream)

    @cute.kernel
    def kernel(self,current,scheduled,out):
        tidx,_,_=cute.arch.thread_idx();block,_,_=cute.arch.block_idx();i=block*self.threads+tidx
        if i<self.n*self.k:
            value=Int64(i%self.e)
            if i//self.k<scheduled[self.b]:value=current[i]%self.e
            out[i]=value


class MetadataEmit:
    def __init__(self,n,b,e,k,c,threads):
        self.n,self.b,self.e,self.k,self.c,self.threads=n,b,e,k,c,threads
        self.chunks=cute.ceil_div(n*k,threads);self.warps=threads//32

    @cute.jit
    def __call__(self,logp,assignments,scheduled,physical,sequence,rowmap,ranks,offsets,ids,weights,inverse,stream):
        self.kernel(logp,assignments,scheduled,physical,sequence,rowmap,ranks,offsets,ids,weights,inverse).launch(
            grid=(cute.ceil_div(self.n,self.warps),1,1),block=(self.threads,1,1),stream=stream)

    @cute.kernel
    def kernel(self,logp,assignments,scheduled,physical,sequence,rowmap,ranks,offsets,ids,weights,inverse):
        tidx,_,_=cute.arch.thread_idx();block,_,_=cute.arch.block_idx()
        lane=tidx%32;row=block*self.warps+tidx//32
        value=Float32(float('-inf'));slot=Int32(0);real=Int32(-1)
        if row<scheduled[self.b] and lane<self.k:
            index=row*self.k+lane;expert=Int32(assignments[index]);s=Int32(sequence[row])
            rank=offsets[expert*self.chunks+index//self.threads]+ranks[index]
            slot=expert*self.c+physical[s]+rank-scheduled[s]*self.k//self.e
            value=logp[row*self.e+expert];real=rowmap[row]
        maximum=value
        for level in cutlass.range_constexpr(5):maximum=cute.arch.fmax(maximum,cute.arch.shuffle_sync_bfly(maximum,1<<level))
        p=cute.exp(value-maximum);total=p
        for level in cutlass.range_constexpr(5):total+=cute.arch.shuffle_sync_bfly(total,1<<level)
        if row<scheduled[self.b] and lane<self.k:
            ids[slot]=Int64(real if real>=0 else -real-1)
            weights[slot]=p/total if real>=0 else Float32(0)
            if real>=0:inverse[real*self.k+lane]=Int64(slot)


class ResetMetadata:
    def __init__(self,size,threads):self.size,self.threads=size,threads
    @cute.jit
    def __call__(self,ids,weights,stream):
        self.kernel(ids,weights).launch(grid=(cute.ceil_div(self.size,self.threads),1,1),block=(self.threads,1,1),stream=stream)
    @cute.kernel
    def kernel(self,ids,weights):
        tidx,_,_=cute.arch.thread_idx();block,_,_=cute.arch.block_idx();i=block*self.threads+tidx
        if i<self.size:ids[i]=Int64(0);weights[i]=Float32(0)


class MetadataAdjoint:
    def __init__(self,n,e,k,c,threads):
        self.n,self.e,self.k,self.c,self.threads=n,e,k,c,threads
        self.warps=threads//32

    @cute.jit
    def __call__(self,weights,inverse,rowmap,gradient,out,stream):
        self.kernel(weights,inverse,rowmap,gradient,out).launch(
            grid=(cute.ceil_div(self.n,self.warps),1,1),block=(self.threads,1,1),stream=stream)

    @cute.kernel
    def kernel(self,weights,inverse,rowmap,gradient,out):
        tidx,_,_=cute.arch.thread_idx();block,_,_=cute.arch.block_idx()
        lane=tidx%32;row=block*self.warps+tidx//32
        p=Float32(0);g=Float32(0);expert=Int32(0);real=Int32(-1)
        if row<self.n:real=rowmap[row]
        if real>=0 and lane<self.k:
            slot=Int32(inverse[real*self.k+lane]);expert=slot//self.c
            p=weights[slot];g=gradient[slot]
        g-=cute.arch.shuffle_sync(g,0)
        total=p*g
        for level in cutlass.range_constexpr(5):total+=cute.arch.shuffle_sync_bfly(total,1<<level)
        if row<self.n:
            for column in cutlass.range(lane,self.e,32,unroll=1):out[row*self.e+column]=Float32(0)
        cute.arch.sync_warp()
        if real>=0 and lane<self.k:out[row*self.e+expert]=p*(g-total)


class VarlenCondition:
    def __init__(self,groups,rounds):self.groups,self.rounds=groups,rounds
    @cute.jit
    def __call__(self,loads,phase,handle:Uint64,stream):
        self.kernel(loads,phase,handle).launch(grid=(1,1,1),block=(128,1,1),stream=stream)
    @cute.kernel
    def kernel(self,loads,phase,handle:Uint64):
        tidx,_,_=cute.arch.thread_idx()
        smem=utils.SmemAllocator();scratch=smem.allocate_tensor(Int32,cute.make_layout((4,)))
        active=Int32(0)
        for i in cutlass.range(tidx,self.groups,128,unroll=1):
            if loads[i]!=loads[self.groups+i]:active=Int32(1)
        any_warp=cute.arch.vote_any_sync(active!=0)
        if tidx%32==0:scratch[tidx//32]=Int32(any_warp)
        cute.arch.sync_threads()
        if tidx==0:
            needed=Int32(0)
            for w in cutlass.range_constexpr(4):needed+=scratch[w]
            set_graph_condition(handle,Uint32(needed>0 and phase[0]<self.rounds))


class LongPoolCondition:
    """Enable the staged child graph iff this invocation has a long pool."""
    def __init__(self,b,max_rows):
        self.b,self.max_rows=b,max_rows

    @cute.jit
    def __call__(self,scheduled,handle:Uint64,stream):
        self.kernel(scheduled,handle).launch(grid=(1,1,1),block=(32,1,1),stream=stream)

    @cute.kernel
    def kernel(self,scheduled,handle:Uint64):
        lane,_,_=cute.arch.thread_idx()
        long=Int32(0)
        for seq in cutlass.range(lane,self.b,32,unroll=1):
            if scheduled[seq+1]-scheduled[seq]>self.max_rows:long=Int32(1)
        any_long=cute.arch.vote_any_sync(long!=0)
        if lane==0:set_graph_condition(handle,Uint32(any_long))


def fake(dtype,count):
    return cute.runtime.make_fake_tensor(dtype,(count,),stride=(1,),assumed_align=8 if dtype==Int64 else 4)


def build(kernel,*args):
    return cute.compile(kernel,*args,cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),options='--enable-tvm-ffi --opt-level 3')


def compile_varlen_quantile(n,b,e,k,threads,cache_items,expert_keys=False):
    args=(fake(Float32,n*e),fake(Int32,b+1),fake(Float32,b*e))
    if expert_keys:
        return build(ExpertKeysColumnQuantile(n,b,e,k,threads,cache_items),*args,fake(Uint32,n*e))
    return build(ColumnQuantile(n,b,e,k,threads,cache_items),*args)


@lru_cache(maxsize=128)
def compile_varlen(t,b,e,k,n,c,iterations,forward,backward,qb_threads,metadata_threads,strict=False,
                   column_cache_items=0,warp_rows=0,column_shards_forward=0,column_shards_backward=0,
                   expert_keys=0,persistent=0,hybrid_rows=0,persistent_threads=0):
    f=lambda size:fake(Float32,size)
    i=lambda size:fake(Int32,size)
    l=lambda size:fake(Int64,size)
    geometry=(f(t*e),i(b+1),i(b+1),i(b+1),l(n),l(n),i(n),f(n*e),i(1))
    prefix=build(LayoutKernel(t,b,e,k,n,metadata_threads,'prefix',strict),*geometry)
    pack=build(LayoutKernel(t,b,e,k,n,metadata_threads,'pack',strict),*geometry)
    examples=(f(n*e),f(max(n,b*e)),f(max(n,b*e)),f(2*iterations*n*e),f(max(n,t)*e),f(n*e),i(b+1),l(n),i(n),cutlass.Int32(0))
    kernels={}
    if persistent:
        multiple=e//math.gcd(e,k)
        max_rows=hybrid_rows or cute.ceil_div(t,multiple)*multiple
        if hybrid_rows:kernels['long_pool_condition',0]=build(LongPoolCondition(b,hybrid_rows),examples[6],Uint64(0))
        for bit,g in ((1,forward),(2,backward)):
            if persistent&bit:
                back=bit==2
                kernels['persistent',int(back)]=build(
                    PersistentPoolSinkhorn(n,e,b,k,iterations,max_rows,persistent_threads or g[0],back,bool(hybrid_rows)),
                    examples[0],examples[3],f((t if back else n)*e),examples[6],examples[8],examples[5])
    for mode,axis,g in (('init',0,forward),('finish',0,forward),('unpack',0,backward),
                        ('forward',0,forward),('forward',1,forward),('backward',0,backward),('backward',1,backward)):
        persistent_direction=bool(persistent&(2 if mode in ('backward','unpack') else 1))
        if persistent_direction and not hybrid_rows:continue
        skip=hybrid_rows if persistent_direction else 0
        threads,rows,columns=g;groups=rows if axis==0 else columns
        shards=column_shards_forward if mode=='forward' else column_shards_backward
        if axis==1 and shards:
            kernels[mode,axis]=build(ShardedColumnSinkhorn(n,e,b,k,threads,groups,shards,mode,skip),
                *examples[:5],examples[6],f(b*shards*e),examples[-1])
            continue
        actual_examples=list(examples)
        if mode=='unpack':actual_examples[4]=f(t*e)
        kernel=(WarpRowSinkhorn(n,e,b,k,threads,mode,skip) if warp_rows and axis==0 and mode in ('forward','backward')
                else VarlenSinkhorn(n,e,b,k,threads,groups,axis,mode,skip))
        kernels[mode,axis]=build(kernel,*actual_examples)
    quantile=compile_varlen_quantile(n,b,e,k,qb_threads,column_cache_items,expert_keys)
    finalize=build(FinalizeAssignments(n,b,e,k,metadata_threads),l(n*k),i(b+1),l(n*k))
    chunks=cute.ceil_div(n*k,metadata_threads)
    meta_examples=(f(n*e),l(n*k),i(e*chunks),i(n*k),i(e*chunks),l(e*c),f(e*c),l(t*k))
    count,pref=(build(StableMetadataKernel(n,e,k,t,metadata_threads,mode),*meta_examples) for mode in ('count','prefix'))
    emit_examples=(f(n*e),l(n*k),i(b+1),i(b+1),l(n),i(n),i(n*k),i(e*chunks),l(e*c),f(e*c),l(t*k))
    emit=build(MetadataEmit(n,b,e,k,c,metadata_threads),*emit_examples)
    adjoint=build(MetadataAdjoint(n,e,k,c,metadata_threads),f(e*c),l(t*k),i(n),f(e*c),f(n*e))
    reset=build(ResetMetadata(e*c,metadata_threads),l(e*c),f(e*c))
    return prefix,pack,kernels,quantile,finalize,(count,pref,emit,adjoint,reset)


@lru_cache(maxsize=128)
def compile_varlen_condition(groups,rounds):
    return build(VarlenCondition(groups,rounds),fake(Int64,2*groups),fake(Int64,1),cutlass.Uint64(0))
