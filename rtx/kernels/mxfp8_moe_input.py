"""Route already-quantized token rows without repeating their quantization.

The caller owns all storage. Row-axis scales are copied with their FP8 values;
this never substitutes them for independently quantized leading-axis scales.
Native-scale destinations are written in their physical layout directly.
"""
from functools import lru_cache

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16,Float8E4M3FN,Float8E8M0FNU,Int32,Int64,Uint8,Uint32
from cutlass.experimental.primitives import nvvm_wrapper as nvvm


def packed_input_gather_rejection(n,d,e,c,k,storage_c,threads,scale_layout,copy_bf16=False):
    if any(type(v) is not int or v<=0 for v in (n,d,e,c,k,storage_c)):
        return 'Positive token/feature/expert/capacity/top-k extents required'
    if storage_c<c:return 'Physical capacity cannot truncate assignments'
    if threads not in (64,128,256,512):return 'Use 64/128/256/512 movement threads'
    if scale_layout not in ('row_major','mma128'):return 'Unsupported destination scale layout'
    if scale_layout=='mma128' and (storage_c%128 or ((d+31)//32*32)%128):
        return 'Native128 destination requires complete 128-row/feature scale tiles'
    if type(copy_bf16) is not bool:return 'copy_bf16 must be boolean'
    return None


class PackedInputGather:
    def __init__(self,n,d,e,c,k,storage_c,threads,scale_layout,copy_bf16=False):
        reason=packed_input_gather_rejection(n,d,e,c,k,storage_c,threads,scale_layout,copy_bf16)
        if reason:raise ValueError(reason)
        self.n,self.d,self.e,self.c,self.k=n,d,e,c,k
        self.storage_c,self.threads=storage_c,threads
        self.width=(d+31)//32*32
        self.scale_cols=self.width//32
        self.scale_layout,self.copy_bf16=scale_layout,copy_bf16

    @cute.jit
    def __call__(self,global_q,global_sf,x,ids,inverse,out,out_sf,gathered,stream):
        grid=cute.ceil_div(self.e*self.storage_c,self.threads//32)
        self.kernel(global_q,global_sf,x,ids,inverse,out,out_sf,gathered).launch(
            grid=(grid,1,1),block=(self.threads,1,1),stream=stream)

    @cute.kernel
    def kernel(self,global_q,global_sf,x,ids,inverse,out,out_sf,gathered):
        flat_sf=cute.make_tensor(out_sf.iterator,cute.make_layout(self.e*self.storage_c*self.scale_cols))
        thread=cute.arch.thread_idx()[0]
        block=cute.arch.block_idx()[0]
        lane=cute.arch.lane_idx()
        slot=block*(self.threads//32)+thread//32
        token,valid=Int32(0),Int32(0)
        if slot<self.e*self.storage_c and slot%self.storage_c<self.c and lane==0:
            expert,row=slot//self.storage_c,slot%self.storage_c
            token=Int32(ids[expert,row])
            if token>=0 and token<self.n:
                logical=expert*self.c+row
                for rank in cutlass.range_constexpr(self.k):
                    if inverse[token,rank]==logical:valid=Int32(1)
        token=cute.arch.shuffle_sync(token,0)
        valid=cute.arch.shuffle_sync(valid,0)
        if slot<self.e*self.storage_c:
            # The physical row width is a multiple of 32, so every u32 copy
            # remains aligned even for an unaligned logical feature tail.
            for col in cutlass.range(lane*4,self.width,128,unroll=1):
                bits=Uint32(0)
                if valid!=0:
                    bits=nvvm.inline_ptx_hl('ld.global.u32 {$w0}, [{$r0}];',
                        write_only_types=[Uint32],
                        read_only_args=[global_q.iterator+token*self.width+col])
                nvvm.inline_ptx_hl('st.global.u32 [{$r0}], {$r1};',
                    write_only_types=[],read_only_args=[out.iterator+slot*self.width+col,bits])
            for scale_col in cutlass.range(lane,self.scale_cols,32,unroll=1):
                scale=Uint8(0).bitcast(Float8E8M0FNU)
                if valid!=0:scale=global_sf[token,scale_col]
                offset=slot*self.scale_cols+scale_col
                if cutlass.const_expr(self.scale_layout=='mma128'):
                    expert,row=slot//self.storage_c,slot%self.storage_c
                    offset=(expert*self.storage_c*self.scale_cols
                            +(row//128)*(self.width//128*512)+(scale_col//4)*512
                            +(row%32)*16+((row//32)%4)*4+scale_col%4)
                flat_sf[offset]=scale
            if cutlass.const_expr(self.copy_bf16):
                for col in cutlass.range(lane,self.d,32,unroll=1):
                    value=BFloat16(0)
                    if valid!=0:value=x[token,col]
                    gathered[slot,col]=value


def _fake(dtype,rows,cols,alignment=16):
    return cute.runtime.make_fake_tensor(dtype,(rows,cols),stride=(cols,1),assumed_align=alignment)


@lru_cache(maxsize=128)
def compile_packed_input_gather(n,d,e,c,k,storage_c,threads=128,scale_layout='row_major',copy_bf16=False):
    """Launch ABI: qX, sX, X, ids, inverse, qGather, sGather, optional BF16 out.

    All tensors are supplied by the caller. With ``copy_bf16=False`` the last
    argument may alias X; it is not read or written. Global-token scales are
    always row-major even when the destination uses native128 physical layout.
    """
    kernel=PackedInputGather(n,d,e,c,k,storage_c,threads,scale_layout,copy_bf16)
    qx=_fake(Float8E4M3FN,n,kernel.width)
    sx=_fake(Float8E8M0FNU,n,kernel.scale_cols)
    x=_fake(BFloat16,n,d,2)
    ids,inverse=_fake(Int64,e,c),_fake(Int64,n,k)
    qout=_fake(Float8E4M3FN,e*storage_c,kernel.width)
    sout=_fake(Float8E8M0FNU,e*storage_c,kernel.scale_cols)
    gathered=_fake(BFloat16,e*storage_c,d) if copy_bf16 else x
    stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(kernel,qx,sx,x,ids,inverse,qout,sout,gathered,stream,
                        options='--enable-tvm-ffi --opt-level 3')
