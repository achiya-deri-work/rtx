"""Finite log-Sinkhorn, saving every normalization Jacobian for backward."""
from functools import lru_cache
import math
import os

os.environ.setdefault('CUTE_DSL_ARCH', 'sm_120a')

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Float32


class SinkhornKernel:
    def __init__(self,n,e,*,axis,threads,groups,mode,iteration=0,mass=1):
        self.n,self.e=n,e
        self.axis,self.threads,self.groups=axis,threads,groups
        self.mode,self.iteration=mode,iteration
        self.log_mass=math.log(mass)
        self.vectors=n if axis==0 else e
        self.length=e if axis==0 else n

    @cute.jit
    def __call__(self,source,other,shift,history,out,iteration:cutlass.Int32,stream:cuda.CUstream):
        blocks=cute.ceil_div(self.vectors,self.groups)
        if cutlass.const_expr(self.mode in ('init','finish')):
            blocks=cute.ceil_div(self.e if self.mode=='init' else self.n*self.e,self.threads)
        self.kernel(source,other,shift,history,out,iteration).launch(
            grid=(blocks,1,1),block=(self.threads,1,1),stream=stream)

    @cute.jit
    def reduce(self,value,scratch,tidx,maximum:cutlass.Constexpr):
        lane=tidx%32
        warp=tidx//32
        group=tidx%self.groups
        for level in cutlass.range_constexpr(int(math.log2(32//self.groups))):
            peer=cute.arch.shuffle_sync_bfly(value,self.groups*(1<<level))
            if cutlass.const_expr(maximum):value=cute.arch.fmax(value,peer)
            else:value=value+peer
        if lane<self.groups:scratch[warp*self.groups+group]=value
        cute.arch.sync_threads()
        if warp==0:
            total=Float32(float('-inf') if maximum else 0.)
            for part in cutlass.range_constexpr(cute.ceil_div(self.groups*(self.threads//32),32)):
                index=lane+part*32
                if index<self.groups*(self.threads//32):
                    if cutlass.const_expr(maximum):total=cute.arch.fmax(total,scratch[index])
                    else:total=total+scratch[index]
            for level in cutlass.range_constexpr(int(math.log2(32//self.groups))):
                peer=cute.arch.shuffle_sync_bfly(total,self.groups*(1<<level))
                if cutlass.const_expr(maximum):total=cute.arch.fmax(total,peer)
                else:total=total+peer
            if lane<self.groups:scratch[group]=total
        cute.arch.sync_threads()
        result=scratch[group]
        cute.arch.sync_threads()
        return result

    @cute.kernel
    def kernel(self,source,other,shift,history,out,iteration:cutlass.Int32):
        tidx,_,_=cute.arch.thread_idx()
        block,_,_=cute.arch.block_idx()
        if cutlass.const_expr(self.mode=='init'):
            index=block*self.threads+tidx
            if index<self.e:other[index]=Float32(0)
        elif cutlass.const_expr(self.mode=='finish'):
            index=block*self.threads+tidx
            if index<self.n*self.e:
                out[index]=source[index]+shift[index//self.e]+other[index%self.e]
        else:
            smem=utils.SmemAllocator()
            scratch=smem.allocate_tensor(Float32,cute.make_layout((self.groups*(self.threads//32),)))
            vector=block*self.groups+tidx%self.groups
            first=tidx//self.groups
            step=self.threads//self.groups
            history_base=(2*iteration+self.axis)*self.n*self.e
            if cutlass.const_expr(self.mode=='forward'):
                maximum=Float32(float('-inf'))
                for element in cutlass.range(first,self.length,step,unroll=1):
                    if vector<self.vectors:
                        index=vector*self.e+element if self.axis==0 else element*self.e+vector
                        value=source[index]+other[element]
                        maximum=cute.arch.fmax(maximum,value)
                maximum=self.reduce(maximum,scratch,tidx,True)
                total=Float32(0)
                for element in cutlass.range(first,self.length,step,unroll=1):
                    if vector<self.vectors:
                        index=vector*self.e+element if self.axis==0 else element*self.e+vector
                        total+=cute.exp(source[index]+other[element]-maximum)
                total=self.reduce(total,scratch,tidx,False)
                normalizer=maximum+cute.log(total)
                if vector<self.vectors and first==0:
                    shift[vector]=Float32(self.log_mass)-normalizer
                for element in cutlass.range(first,self.length,step,unroll=1):
                    if vector<self.vectors:
                        index=vector*self.e+element if self.axis==0 else element*self.e+vector
                        history[history_base+index]=cute.exp(source[index]+other[element]-normalizer)
            else:
                total=Float32(0)
                for element in cutlass.range(first,self.length,step,unroll=1):
                    if vector<self.vectors:
                        index=vector*self.e+element if self.axis==0 else element*self.e+vector
                        total+=source[index]
                total=self.reduce(total,scratch,tidx,False)
                for element in cutlass.range(first,self.length,step,unroll=1):
                    if vector<self.vectors:
                        index=vector*self.e+element if self.axis==0 else element*self.e+vector
                        out[index]=source[index]-history[history_base+index]*total


@lru_cache(maxsize=128)
def compile_sinkhorn(n,e,top_k,iterations,threads,row_groups,column_groups):
    def fake(length):
        return cute.runtime.make_fake_tensor(Float32,(length,),stride=(1,),assumed_align=16)
    source,out=fake(n*e),fake(n*e)
    row,column=fake(n),fake(e)
    history=fake(2*iterations*n*e)
    stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    def build(axis,groups,mode,iteration=0,mass=1):
        kernel=SinkhornKernel(n,e,axis=axis,threads=threads,groups=groups,
                              mode=mode,iteration=iteration,mass=mass)
        other,shift=(column,row) if axis==0 else (row,column)
        if mode=='backward':other,shift=fake(1),fake(1)
        return cute.compile(kernel,source,other,shift,history,out,cutlass.Int32(0),stream,
                            options='--enable-tvm-ffi --opt-level 3')
    forward=(build(0,row_groups,'forward',mass=top_k),
             build(1,column_groups,'forward',mass=n*top_k//e))
    backward=(build(1,column_groups,'backward'),build(0,row_groups,'backward'))
    return build(0,1,'init'),forward,build(0,1,'finish'),backward
