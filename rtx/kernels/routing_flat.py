"""Static single-pool row/column Sinkhorn variants, preserving saved FP32 states."""
from functools import lru_cache
import cutlass
import cutlass.cute as cute
import math
from cutlass import Int32, Float32
from .routing_varlen import ShardedColumnSinkhorn


class FlatColumns(ShardedColumnSinkhorn):
    @cute.jit
    def bounds(self,unused,tidx,block):
        shard=block//self.tiles
        expert=(block%self.tiles)*self.groups+tidx%self.groups
        rows=cute.ceil_div(self.n,self.shards*self.row_threads)*self.row_threads
        a=shard*rows
        z=cutlass.min(Int32(self.n),a+rows)
        return Int32(0),shard,expert,Int32(self.n),a,z


class FlatRows:
    """Packed warp-row reduction on one static pool; no length or sequence buffers."""
    def __init__(self,n,e,k,threads,mode):
        self.n,self.e,self.k,self.threads,self.mode=n,e,k,threads,mode
        self.items=cute.ceil_div(e,32);self.warps=threads//32;self.logmass=math.log(k)

    @cute.jit
    def __call__(self,source,other,shift,history,out,iteration:cutlass.Int32,stream):
        self.kernel(source,other,shift,history,out,iteration).launch(
            grid=(cute.ceil_div(self.n,self.warps),1,1),block=(self.threads,1,1),stream=stream)

    @cute.kernel
    def kernel(self,source,other,shift,history,out,iteration:cutlass.Int32):
        tid,_,_=cute.arch.thread_idx();block,_,_=cute.arch.block_idx()
        lane=tid%32;row=block*self.warps+tid//32
        if row<self.n:
            values=cute.make_rmem_tensor((self.items,),Float32)
            values.fill(Float32(float('-inf') if self.mode=='forward' else 0))
            maximum=Float32(float('-inf'));total=Float32(0)
            for item in cutlass.range_constexpr(self.items):
                col=lane+item*32
                if col<self.e:
                    value=source[row*self.e+col]
                    if cutlass.const_expr(self.mode=='forward'):value+=other[col]
                    values[item]=value
                if cutlass.const_expr(self.mode=='forward'):maximum=cute.arch.fmax(maximum,values[item])
                else:total+=values[item]
            if cutlass.const_expr(self.mode=='forward'):
                for level in cutlass.range_constexpr(5):
                    maximum=cute.arch.fmax(maximum,cute.arch.shuffle_sync_bfly(maximum,1<<level))
                for item in cutlass.range_constexpr(self.items):total+=cute.exp(values[item]-maximum)
            for level in cutlass.range_constexpr(5):total+=cute.arch.shuffle_sync_bfly(total,1<<level)
            base=2*iteration*self.n*self.e
            norm=Float32(0)
            if cutlass.const_expr(self.mode=='forward'):
                norm=maximum+cute.log(total)
                if lane==0:shift[row]=Float32(self.logmass)-norm
            for item in cutlass.range_constexpr(self.items):
                col=lane+item*32;idx=row*self.e+col
                if col<self.e:
                    if cutlass.const_expr(self.mode=='forward'):history[base+idx]=cute.exp(values[item]-norm)
                    else:out[idx]=values[item]-history[base+idx]*total


@lru_cache(maxsize=128)
def compile_flat_rows(n,e,k,steps,threads,mode):
    def f(size):return cute.runtime.make_fake_tensor(Float32,(size,),stride=(1,),assumed_align=16)
    stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(FlatRows(n,e,k,threads,mode),f(n*e),f(e if mode=='forward' else 1),
                        f(n if mode=='forward' else 1),f(2*steps*n*e),f(n*e),
                        Int32(0),stream,options='--enable-tvm-ffi --opt-level 3')


@lru_cache(maxsize=128)
def compile_flat_shards(n,e,k,steps,threads,column_groups,shards,mode):
    module=FlatColumns(n,e,1,k,threads,column_groups,shards,mode)
    def f(size):
        return cute.runtime.make_fake_tensor(Float32,(size,),stride=(1,),assumed_align=16)
    # Existing packed compile ABI; scheduled is unused in our specialized bounds.
    stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(module,f(n*e),f(n if mode=='forward' else 1),
                        f(e if mode=='forward' else 1),f(2*steps*n*e),
                        f(n*e),f(n if mode=='forward' else 1),f(shards*e),
                        Int32(0),stream,options='--enable-tvm-ffi --opt-level 3')
