"""Research four-operand producer with compact interleaved CTA ownership.

Original kernels supply all transform, quantization and correction arithmetic.
Each CTA owns one role; common registers/SMEM can make fusion slower.
"""
from functools import lru_cache
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils
from cutlass import BFloat16,Float32,Int32,Uint8
from .nvfp4_posthoc import HadamardPreparationKernel


def cta_bands(counts):
    if len(counts)!=4 or any(type(n) is not int or n<1 for n in counts):
        raise ValueError('four positive integer CTA counts required')
    bands=[];lower=0;offset=0
    for upper in sorted(set(counts)):
        roles=tuple(i for i,n in enumerate(counts) if n>=upper)
        size=(upper-lower)*len(roles)
        bands.append((offset,offset+size,lower,roles))
        offset+=size;lower=upper
    return tuple(bands)


class PreparationRole(HadamardPreparationKernel):
    @cute.jit
    def write_remapped(self,largest,partials,logical):
        thread,_,_=cute.arch.thread_idx();lane=cute.arch.lane_idx()
        for level in cutlass.range_constexpr(5):
            largest=cute.arch.fmax(largest,cute.arch.shuffle_sync_bfly(largest,16>>level))
        if lane==0:partials[logical*(self.config.threads//32)+thread//32]=largest

    @cute.jit
    def prepare(self,source,signs,packed,scales,partials,logical,shared):
        thread,_,_=cute.arch.thread_idx();lane=cute.arch.lane_idx()
        local_lane=lane%self.lanes_per_group
        cached=cute.make_rmem_tensor((self.values_per_lane,),Float32)
        for j in cutlass.range_constexpr(self.values_per_lane):
            cached[j]=signs[local_lane*self.values_per_lane+j]
        largest=Float32(0.)
        if cutlass.const_expr(not self.transpose):
            warp=logical*(self.config.threads//32)+thread//32
            subgroup=lane//self.lanes_per_group
            for first in cutlass.range(warp,self.tiles,self.work_warp_count*self.groups_per_warp,unroll=1):
                tile=first+subgroup*self.work_warp_count
                active=tile<self.tiles
                values=cute.make_rmem_tensor((self.values_per_lane,),Float32)
                for j in cutlass.range_constexpr(self.values_per_lane):values[j]=Float32(0.)
                if active:
                    offset=tile*128+local_lane*self.values_per_lane
                    for j in cutlass.range_constexpr(self.values_per_lane):
                        values[j]=source[offset+j].to(Float32)*cached[j]
                largest=self.grouped_prepare_values(values,packed,scales,tile,local_lane,active,largest,Float32(1.))
        else:
            staged=cute.make_tensor(shared.iterator,cute.make_layout((128,self.source_tile_rows,1),
                stride=(self.async_row_stride,1,128*self.async_row_stride)))
            limit=self.tiles//self.source_tile_rows
            if logical<limit:self.stage_async(source,staged,logical,0)
            cute.arch.cp_async_commit_group()
            for iteration in cutlass.range(0,(limit+self.blocks-1-logical)//self.blocks,unroll=1):
                tile_index=logical+iteration*self.blocks
                cute.arch.cp_async_wait_group(0);cute.arch.sync_threads()
                row_group=tile_index//(self.cols//128);chunk=tile_index%(self.cols//128)
                if cutlass.const_expr(self.tile_order=='reduction_major'):
                    row_group=tile_index%(self.rows//self.source_tile_rows)
                    chunk=tile_index//(self.rows//self.source_tile_rows)
                tile=(row_group*self.source_tile_rows+thread//32)*(self.cols//128)+chunk
                largest=self.prepare_rows(source,cached,packed,scales,signs,signs,
                    staged[None,None,0],tile,largest,Float32(1.))
                cute.arch.sync_threads()
                future=tile_index+self.blocks
                if future<limit:self.stage_async(source,staged,future,0)
                cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
        self.write_remapped(largest,partials,logical)


class QuadPreparation:
    def __init__(self,roles):
        if len(roles)!=4:raise ValueError('four roles required')
        self.a,self.b,self.c,self.d=roles
        self.threads=roles[0].config.threads
        for i,k in enumerate(roles):
            if type(k) is not PreparationRole:raise ValueError('PreparationRole required')
            if k.config.threads!=self.threads:raise ValueError('common CTA threads required')
            required=dict(packed_source=False,mantissa_bits=4,signs_mode='register',
                correction_decode='native',normalization_math='divide',partial_max_scope='warp',
                producer_scale_math='divide',producer_copy_bits=32,producer_row_load_bits=16,
                producer_row_tile_batch=1,producer_unroll_rows=True)
            for key,value in required.items():
                if getattr(k,key)!=value:raise ValueError(f'unsupported role {i} {key}')
            if k.values_per_lane not in (8,16):raise ValueError('grouped roles required')
            if i==0:
                if k.transpose or k.tile_order!='row_major':raise ValueError('first role must be row-major')
            elif not k.transpose or not k.async_copy or k.stages!=1 or k.rows_per_warp!=8:
                raise ValueError('transpose roles require x8 async stage1')
        self.bands=cta_bands(tuple(k.blocks for k in roles))
        self.blocks=sum(k.blocks for k in roles)
        self.shared_elements=max(128*k.async_row_stride for k in roles[1:])

    @cute.jit
    def __call__(self,sa:cute.Tensor,ha:cute.Tensor,pa:cute.Tensor,ta:cute.Tensor,ma:cute.Tensor,
                 sb:cute.Tensor,hb:cute.Tensor,pb:cute.Tensor,tb:cute.Tensor,mb:cute.Tensor,
                 sc:cute.Tensor,hc:cute.Tensor,pc:cute.Tensor,tc:cute.Tensor,mc:cute.Tensor,
                 sd:cute.Tensor,hd:cute.Tensor,pd:cute.Tensor,td:cute.Tensor,md:cute.Tensor,
                 stream:cuda.CUstream):
        self.kernel(sa,ha,pa,ta,ma,sb,hb,pb,tb,mb,sc,hc,pc,tc,mc,sd,hd,pd,td,md).launch(
            grid=(self.blocks,1,1),block=(self.threads,1,1),stream=stream)

    @cute.kernel
    def kernel(self,sa:cute.Tensor,ha:cute.Tensor,pa:cute.Tensor,ta:cute.Tensor,ma:cute.Tensor,
               sb:cute.Tensor,hb:cute.Tensor,pb:cute.Tensor,tb:cute.Tensor,mb:cute.Tensor,
               sc:cute.Tensor,hc:cute.Tensor,pc:cute.Tensor,tc:cute.Tensor,mc:cute.Tensor,
               sd:cute.Tensor,hd:cute.Tensor,pd:cute.Tensor,td:cute.Tensor,md:cute.Tensor):
        physical,_,_=cute.arch.block_idx()
        role=Int32(0);logical=Int32(0)
        for band in cutlass.range_constexpr(len(self.bands)):
            start,end,lower,roles=self.bands[band]
            if physical>=start and physical<end:
                logical=lower+(physical-start)//len(roles)
                for slot in cutlass.range_constexpr(len(roles)):
                    if (physical-start)%len(roles)==slot:role=Int32(roles[slot])
        allocator=cutlass.utils.SmemAllocator()
        shared=allocator.allocate_tensor(BFloat16,cute.make_layout((self.shared_elements,)),byte_alignment=16)
        if role==0:self.a.prepare(sa,ha,pa,ta,ma,logical,shared)
        elif role==1:self.b.prepare(sb,hb,pb,tb,mb,logical,shared)
        elif role==2:self.c.prepare(sc,hc,pc,tc,mc,logical,shared)
        else:self.d.prepare(sd,hd,pd,td,md,logical,shared)


@lru_cache(maxsize=32)
def compile_quad_preparation(roles):
    kernel=QuadPreparation(roles)
    fake=lambda dtype,n:cute.runtime.make_fake_tensor(dtype,(n,),stride=(1,),assumed_align=4)
    args=[]
    for k in roles:
        n=k.source_rows*k.source_cols
        args.extend((fake(BFloat16,n),fake(Float32,128),fake(Uint8,n//2),
                     fake(BFloat16,n//16),fake(Float32,k.partial_count)))
    return cute.compile(kernel,*args,cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options='--enable-tvm-ffi --opt-level 3')
