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


@cute.jit
def rounded_prefix(active, storage: cutlass.Constexpr, tile: cutlass.Constexpr):
    """Full final consumer tile, clamped at the static descriptor boundary."""
    return cutlass.min(storage, cute.ceil_div(cutlass.max(0, active), tile) * tile)


class PackedInputGather:
    def __init__(self,n,d,e,c,k,storage_c,threads,scale_layout,copy_bf16=False,emit_index=False,
                 active_m_alignment=0,row_tile_m=1,map_tile_ks=(),full_map=False):
        reason=packed_input_gather_rejection(n,d,e,c,k,storage_c,threads,scale_layout,copy_bf16)
        if reason:raise ValueError(reason)
        if type(emit_index) is not bool or (emit_index and copy_bf16):
            raise ValueError('Indexed mapping replaces, not accompanies, the BF16 gather')
        if type(full_map) is not bool or (full_map and not (emit_index and active_m_alignment)):
            raise ValueError('Full map override requires a bounded gather with an indexed destination')
        if type(active_m_alignment) is not int or active_m_alignment < 0:
            raise ValueError('Invalid gather active M alignment')
        if active_m_alignment:
            if (scale_layout != 'row_major' or copy_bf16 or storage_c % 32
                    or active_m_alignment % 32 or type(row_tile_m) is not int
                    or row_tile_m < 32 or row_tile_m % 32):
                raise ValueError('Bounded gather requires complete row-major scale/consumer tiles')
            if type(map_tile_ks) is not tuple or any(
                type(k_tile) is not int or k_tile < 32 or k_tile % 32 for k_tile in map_tile_ks
            ) or (map_tile_ks and not emit_index):
                raise ValueError('Bounded gather map K envelopes require an indexed destination')
        elif row_tile_m != 1 or map_tile_ks:
            raise ValueError('Gather consumer tiles require an active M alignment')
        self.n,self.d,self.e,self.c,self.k=n,d,e,c,k
        self.storage_c,self.threads=storage_c,threads
        self.width=(d+31)//32*32
        self.scale_cols=self.width//32
        self.scale_layout,self.copy_bf16=scale_layout,copy_bf16
        self.emit_index=emit_index
        self.active_m_alignment,self.row_tile_m,self.map_tile_ks=active_m_alignment,row_tile_m,map_tile_ks
        self.full_map=full_map

    @cute.jit
    def __call__(self,global_q,global_sf,x,ids,inverse,out,out_sf,gathered,stream,active_capacity=None):
        if cutlass.const_expr(self.active_m_alignment):
            # Each CTA belongs to one expert; the per-warp row never crosses
            # an expert pitch. The grid, like the destination pitch, is static.
            grid=(cute.ceil_div(self.storage_c,self.threads//32),1,self.e)
        else:
            grid=(cute.ceil_div(self.e*self.storage_c,self.threads//32),1,1)
        self.kernel(global_q,global_sf,x,ids,inverse,out,out_sf,gathered,active_capacity).launch(
            grid=grid,block=(self.threads,1,1),stream=stream)

    @cute.kernel
    def kernel(self,global_q,global_sf,x,ids,inverse,out,out_sf,gathered,active_capacity):
        flat_sf=cute.make_tensor(out_sf.iterator,cute.make_layout(self.e*self.storage_c*self.scale_cols))
        thread=cute.arch.thread_idx()[0]
        block,_,z=cute.arch.block_idx()
        lane=cute.arch.lane_idx()
        row=block*(self.threads//32)+thread//32
        if cutlass.const_expr(self.active_m_alignment):
            expert=z
            slot=expert*self.storage_c+row
            active=active_capacity[0]
            rows=rounded_prefix(rounded_prefix(active,self.storage_c,self.active_m_alignment),
                                self.storage_c,self.row_tile_m)
            if cutlass.const_expr(self.full_map):
                # An unbounded downstream producer still reads the original
                # map pitch even at a zero prefix. Initialize its entire map
                # while keeping the separately consumed FP8 rows bounded.
                map_rows=self.storage_c
            else:
                map_rows=rows
                for i in cutlass.range_constexpr(len(self.map_tile_ks)):
                    map_rows=cutlass.max(map_rows,rounded_prefix(active,self.storage_c,self.map_tile_ks[i]))
            visit=row<map_rows
            write_row=row<rows
        else:
            slot=row
            expert,row=slot//self.storage_c,slot%self.storage_c
            active=self.c
            visit=slot<self.e*self.storage_c
            write_row=True
        # visit is uniform within a warp (one warp per row). Every lane
        # participating in these shuffles takes the same branch.
        if visit:
            token,valid=Int32(0),Int32(0)
            if row<self.c and row<active and lane==0:
                token=Int32(ids[expert,row])
                if token>=0 and token<self.n:
                    logical=expert*self.c+row
                    for rank in cutlass.range_constexpr(self.k):
                        if inverse[token,rank]==logical:valid=Int32(1)
            token=cute.arch.shuffle_sync(token,0)
            valid=cute.arch.shuffle_sync(valid,0)
            if cutlass.const_expr(self.emit_index):
                if lane==0:gathered[slot,0]=token if valid!=0 else Int32(-1)
            if write_row:
                self._copy_row(global_q,global_sf,x,out,flat_sf,gathered,slot,expert,row,token,valid,lane)

    @cute.jit
    def _copy_row(self,global_q,global_sf,x,out,flat_sf,gathered,slot,expert,row,token,valid,lane):
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
def compile_packed_input_gather(n,d,e,c,k,storage_c,threads=128,scale_layout='row_major',
                                copy_bf16=False,emit_index=False,active_m_alignment=0,row_tile_m=1,
                                map_tile_ks=(),full_map=False,*,_byte_scales=False):
    """Launch ABI: qX, sX, X, ids, inverse, qGather, sGather, map/BF16 placeholder[, prefix].

    All tensors are supplied by the caller. Without a map or BF16 output its
    placeholder may alias X. The optional prefix is a device i32 tensor; only
    complete readable M/K tiles are defined in that mode unless a full
    dependent producer requests ``full_map``.
    """
    kernel=PackedInputGather(n,d,e,c,k,storage_c,threads,scale_layout,copy_bf16,emit_index,
                             active_m_alignment,row_tile_m,map_tile_ks,full_map)
    sf_type=Uint8 if _byte_scales else Float8E8M0FNU
    qx=_fake(Float8E4M3FN,n,kernel.width)
    sx=_fake(sf_type,n,kernel.scale_cols)
    x=_fake(BFloat16,n,d,2)
    ids,inverse=_fake(Int64,e,c),_fake(Int64,n,k)
    qout=_fake(Float8E4M3FN,e*storage_c,kernel.width)
    sout=_fake(sf_type,e*storage_c,kernel.scale_cols)
    gathered=_fake(Int32,e*storage_c,1) if emit_index else _fake(BFloat16,e*storage_c,d) if copy_bf16 else x
    stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    if _byte_scales:
        from .mxfp8_moe_buffers import BytePackedGather
        kernel=BytePackedGather(kernel)
    extra=()
    if active_m_alignment:
        from .mxfp8_moe_active import fake_prefix
        extra=(fake_prefix(),)
    return cute.compile(kernel,qx,sx,x,ids,inverse,qout,sout,gathered,stream,*extra,
                        options='--enable-tvm-ffi --opt-level 3')
