"""Research shared BF16 dY tile for independent row/column H128 quantization."""
from functools import lru_cache
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils
from cutlass import BFloat16,Float32
from cutlass.cute.nvgpu import cpasync
from .nvfp4_posthoc import HadamardPreparationKernel,PosthocScaleConfig


class SharedDY:
    def __init__(self,rows,cols,threads=128,blocks=128,vpl=16,row_vector_bits=16,swizzle=0):
        if rows%128 or cols%128 or min(rows,cols)<128:
            raise ValueError('shared dY requires dimensions divisible by128')
        if threads not in (64,128,256) or blocks<1 or vpl not in (8,16):
            raise ValueError('invalid shared dY geometry')
        self.rows,self.cols,self.threads,self.blocks,self.vpl=rows,cols,threads,blocks,vpl
        if row_vector_bits not in (16,128) or swizzle not in (0,1):
            raise ValueError('invalid shared-memory load/layout choice')
        self.row_vector_bits,self.swizzle=row_vector_bits,swizzle
        self.tiles=(rows//128)*(cols//128)
        config=PosthocScaleConfig(threads,blocks)
        common=dict(signs_mode='register',correction_decode='native',
            producer_values_per_lane=vpl,producer_transpose_values_per_lane=vpl,
            producer_grouped_store_bits=32,load_mode='coalesced_x8')
        self.row=HadamardPreparationKernel(rows,cols,config,4,False,**common)
        self.trans=HadamardPreparationKernel(rows,cols,config,4,True,**common)
        self.lanes=128//vpl
        self.groups=32//self.lanes
        self.partial_count=blocks*(threads//32)

    @cute.jit
    def offset(self,row,col):
        # 32-bit word XOR preserves each128-bit copy's contiguous four words.
        mask=(row^(row>>2))&28
        if cutlass.const_expr(self.swizzle==1):mask=((row<<2)^(row>>2))&28
        return row*128+2*((col//2)^mask)+(col&1)

    @cute.jit
    def finish_max(self,largest,partials):
        thread,_,_=cute.arch.thread_idx();block,_,_=cute.arch.block_idx()
        lane=cute.arch.lane_idx()
        for level in cutlass.range_constexpr(5):
            largest=cute.arch.fmax(largest,cute.arch.shuffle_sync_bfly(largest,16>>level))
        if lane==0:partials[block*(self.threads//32)+thread//32]=largest

    @cute.jit
    def __call__(self,source:cute.Tensor,rs:cute.Tensor,rp:cute.Tensor,rt:cute.Tensor,rm:cute.Tensor,
                 cs:cute.Tensor,cp:cute.Tensor,ct:cute.Tensor,cm:cute.Tensor,stream:cuda.CUstream):
        self.kernel(source,rs,rp,rt,rm,cs,cp,ct,cm).launch(grid=(self.blocks,1,1),
            block=(self.threads,1,1),stream=stream)

    @cute.kernel
    def kernel(self,source:cute.Tensor,rs:cute.Tensor,rp:cute.Tensor,rt:cute.Tensor,rm:cute.Tensor,
               cs:cute.Tensor,cp:cute.Tensor,ct:cute.Tensor,cm:cute.Tensor):
        thread,_,_=cute.arch.thread_idx();block,_,_=cute.arch.block_idx()
        lane=cute.arch.lane_idx();local_lane=lane%self.lanes
        subgroup=lane//self.lanes
        group=thread//32*self.groups+subgroup
        allocator=cutlass.utils.SmemAllocator()
        shared=allocator.allocate_tensor(BFloat16,cute.make_layout((128*128,)),byte_alignment=16)
        rsign=cute.make_rmem_tensor((self.vpl,),Float32)
        csign=cute.make_rmem_tensor((self.vpl,),Float32)
        for j in cutlass.range_constexpr(self.vpl):
            rsign[j]=rs[local_lane*self.vpl+j]
            csign[j]=cs[local_lane*self.vpl+j]
        rmax,cmax=Float32(0.),Float32(0.)
        atom=cute.make_copy_atom(cpasync.CopyG2SOp(),BFloat16,num_bits_per_copy=128)
        for tile in cutlass.range(block,self.tiles,self.blocks,unroll=1):
            row_base=(tile//(self.cols//128))*128
            col_base=(tile%(self.cols//128))*128
            for j in cutlass.range_constexpr(2048//self.threads):
                index=(thread+j*self.threads)*8
                row,col=index//128,index%128
                src=cute.assume((row_base+row)*self.cols+col_base+col,divby=8)
                dst=cute.assume(self.offset(row,col),divby=8)
                cute.copy(atom,cute.make_tensor(source.iterator+src,cute.make_layout((8,))),
                    cute.make_tensor(shared.iterator+dst,cute.make_layout((8,))))
            cute.arch.cp_async_commit_group();cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            for index in cutlass.range(0,128//((self.threads//32)*self.groups),unroll=1):
                owned=group+index*(self.threads//32)*self.groups
                rv=cute.make_rmem_tensor((self.vpl,),Float32)
                if cutlass.const_expr(self.row_vector_bits==128):
                    raw=cute.make_rmem_tensor((self.vpl,),BFloat16)
                    load=cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(),BFloat16,num_bits_per_copy=128)
                    for chunk in cutlass.range_constexpr(self.vpl//8):
                        offset=cute.assume(self.offset(owned,local_lane*self.vpl+chunk*8),divby=8)
                        cute.copy(load,cute.make_tensor(shared.iterator+offset,cute.make_layout((8,))),
                            cute.make_tensor(raw.iterator+chunk*8,cute.make_layout((8,))))
                for j in cutlass.range_constexpr(self.vpl):
                    value=(raw[j] if cutlass.const_expr(self.row_vector_bits==128)
                           else shared[self.offset(owned,local_lane*self.vpl+j)])
                    rv[j]=value.to(Float32)*rsign[j]
                row_tile=(row_base+owned)*(self.cols//128)+col_base//128
                rmax=self.row.grouped_prepare_values(rv,rp,rt,row_tile,local_lane,True,rmax,Float32(1.))
                cv=cute.make_rmem_tensor((self.vpl,),Float32)
                for j in cutlass.range_constexpr(self.vpl):
                    cv[j]=shared[self.offset(local_lane*self.vpl+j,owned)].to(Float32)*csign[j]
                col_tile=(col_base+owned)*(self.rows//128)+row_base//128
                cmax=self.trans.grouped_prepare_values(cv,cp,ct,col_tile,local_lane,True,cmax,Float32(1.))
            cute.arch.sync_threads()
        self.finish_max(rmax,rm);self.finish_max(cmax,cm)


class WarpRoleSharedDY(SharedDY):
    """Shared source tile, disjoint warp roles and only one sign vector live."""
    @cute.kernel
    def kernel(self,source:cute.Tensor,rs:cute.Tensor,rp:cute.Tensor,rt:cute.Tensor,rm:cute.Tensor,
               cs:cute.Tensor,cp:cute.Tensor,ct:cute.Tensor,cm:cute.Tensor):
        thread,_,_=cute.arch.thread_idx();block,_,_=cute.arch.block_idx()
        lane=cute.arch.lane_idx();local_lane=lane%self.lanes
        role_warps=self.threads//64
        row_role=thread//32<role_warps
        group=(thread//32%role_warps)*self.groups+lane//self.lanes
        allocator=cutlass.utils.SmemAllocator()
        shared=allocator.allocate_tensor(BFloat16,cute.make_layout((128*128,)),byte_alignment=16)
        signs=cute.make_rmem_tensor((self.vpl,),Float32)
        for j in cutlass.range_constexpr(self.vpl):
            if row_role:signs[j]=rs[local_lane*self.vpl+j]
            else:signs[j]=cs[local_lane*self.vpl+j]
        rmax,cmax=Float32(0.),Float32(0.)
        atom=cute.make_copy_atom(cpasync.CopyG2SOp(),BFloat16,num_bits_per_copy=128)
        for tile in cutlass.range(block,self.tiles,self.blocks,unroll=1):
            row_base=(tile//(self.cols//128))*128
            col_base=(tile%(self.cols//128))*128
            for j in cutlass.range_constexpr(2048//self.threads):
                index=(thread+j*self.threads)*8
                row,col=index//128,index%128
                src=cute.assume((row_base+row)*self.cols+col_base+col,divby=8)
                dst=cute.assume(self.offset(row,col),divby=8)
                cute.copy(atom,cute.make_tensor(source.iterator+src,cute.make_layout((8,))),
                    cute.make_tensor(shared.iterator+dst,cute.make_layout((8,))))
            cute.arch.cp_async_commit_group();cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            for index in cutlass.range(0,128//(role_warps*self.groups),unroll=1):
                owned=group+index*role_warps*self.groups
                values=cute.make_rmem_tensor((self.vpl,),Float32)
                if row_role:
                    if cutlass.const_expr(self.row_vector_bits==128):
                        raw=cute.make_rmem_tensor((self.vpl,),BFloat16)
                        load=cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(),BFloat16,num_bits_per_copy=128)
                        for chunk in cutlass.range_constexpr(self.vpl//8):
                            offset=cute.assume(self.offset(owned,local_lane*self.vpl+chunk*8),divby=8)
                            cute.copy(load,cute.make_tensor(shared.iterator+offset,cute.make_layout((8,))),
                                cute.make_tensor(raw.iterator+chunk*8,cute.make_layout((8,))))
                    for j in cutlass.range_constexpr(self.vpl):
                        value=(raw[j] if cutlass.const_expr(self.row_vector_bits==128)
                               else shared[self.offset(owned,local_lane*self.vpl+j)])
                        values[j]=value.to(Float32)*signs[j]
                    row_tile=(row_base+owned)*(self.cols//128)+col_base//128
                    rmax=self.row.grouped_prepare_values(values,rp,rt,row_tile,local_lane,True,rmax,Float32(1.))
                else:
                    for j in cutlass.range_constexpr(self.vpl):
                        values[j]=shared[self.offset(local_lane*self.vpl+j,owned)].to(Float32)*signs[j]
                    col_tile=(col_base+owned)*(self.rows//128)+row_base//128
                    cmax=self.trans.grouped_prepare_values(values,cp,ct,col_tile,local_lane,True,cmax,Float32(1.))
            cute.arch.sync_threads()
        # Unused-role partials are explicitly zero, so no stale state is read.
        self.finish_max(rmax,rm);self.finish_max(cmax,cm)


@lru_cache(maxsize=32)
def compile_shared_dy(rows,cols,threads=128,blocks=128,vpl=16,row_vector_bits=16,swizzle=0,*,warp_roles=False,register_cap=255):
    from cutlass import Uint8
    if type(warp_roles) is not bool:raise ValueError('warp_roles must be bool')
    if type(register_cap) is not int or register_cap not in (80,96,255):
        raise ValueError('register_cap must be 80,96,or255')
    kernel=(WarpRoleSharedDY if warp_roles else SharedDY)(rows,cols,threads,blocks,vpl,row_vector_bits,swizzle)
    fake=lambda dtype,n:cute.runtime.make_fake_tensor(dtype,(n,),stride=(1,),assumed_align=16)
    args=[fake(BFloat16,rows*cols)]
    for _ in range(2):args.extend((fake(Float32,128),fake(Uint8,rows*cols//2),
                                  fake(BFloat16,rows*cols//16),fake(Float32,kernel.partial_count)))
    options='--enable-tvm-ffi --opt-level 3'
    if register_cap!=255:options+=f" --ptxas-options '-O3 --maxrregcount={register_cap}'"
    return cute.compile(kernel,*args,cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options=options),kernel.partial_count
