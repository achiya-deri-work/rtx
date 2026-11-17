"""Short-reduction packed experts without a dedicated TMA producer warp.

All MMA warps cooperatively issue predicated 16-byte cp.async operand copies.
This single-buffer alternative deliberately trades overlap for less pipeline
setup; ordinary staged TMA remains independently selectable for every GEMM.
"""
import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm120_utils
from cutlass import BFloat16,Float8E4M3FN,Float8E8M0FNU,Float32,Int32,Uint8
from cutlass.experimental.primitives import nvvm_wrapper as nvvm

from .mxfp8_gemm import MXFP8GemmKernel
from .mxfp8_fwd import _make_ldmatrix_atom,_make_scale_s2r_atom


class CooperativeExpertGemm(MXFP8GemmKernel):
    @cute.jit
    def __call__(self,a,b,sa,sb,out,aux,stream):
        cfg=self.config
        tiles=cute.ceil_div(self.problem.m,cfg.tile_m)*cute.ceil_div(self.problem.n,cfg.tile_n)
        self.cooperative_kernel(a,b,sa,sb,out).launch(grid=(tiles,1,self.batch_count),
            block=(cfg.num_mma_warps*32,1,1),stream=stream)

    @cute.jit
    def _load_operand(self,source,scales,smem,sf,layout,rows:cutlass.Constexpr,
                      tile_rows:cutlass.Constexpr,row_base,k_base,thread):
        cfg=self.config
        threads=cfg.num_mma_warps*32
        physical=cute.make_composed_layout(layout.inner,0,smem.layout)
        for task in cutlass.range(thread,tile_rows*cfg.tile_k//16,threads,unroll=1):
            row,col=task//(cfg.tile_k//16),(task%(cfg.tile_k//16))*16
            gr,gk=row_base+row,k_base+col
            valid=Int32(0)
            if gr<rows and gk<self.problem.k:valid=Int32(16)
            safe_row=cutlass.min(gr,rows-1)
            safe_k=cutlass.min(gk,self.problem.k-16)
            destination=cute.recast_ptr(smem.iterator,dtype=Uint8)+physical((row,col,0))
            nvvm.cp_async_shared_global(destination,source.iterator+safe_row*self.problem.k+safe_k,
                                         16,'cg',cp_size=valid)
        for task in cutlass.range(thread,tile_rows*(cfg.tile_k//32),threads,unroll=1):
            row,kb=task//(cfg.tile_k//32),task%(cfg.tile_k//32)
            gr,gkb=row_base+row,k_base//32+kb
            value=Uint8(0).bitcast(Float8E8M0FNU)
            if gr<rows and gkb<self.problem.k//32:value=scales[gr,gkb]
            sf[row,kb*32,0]=value

    @cute.kernel
    def cooperative_kernel(self,a,b,sa,sb,out):
        cfg=self.config
        self._setup_static_layouts()
        thread=cute.arch.thread_idx()[0]
        block,_,expert=cute.arch.block_idx()
        m,n,k=self.problem.m,self.problem.n,self.problem.k
        a=cute.make_tensor(a.iterator+expert*m*k,cute.make_layout((m,k),stride=(k,1)))
        b=cute.make_tensor(b.iterator+expert*n*k,cute.make_layout((n,k),stride=(k,1)))
        sa=cute.make_tensor(sa.iterator+expert*m*(k//32),cute.make_layout((m,k//32),stride=(k//32,1)))
        sb=cute.make_tensor(sb.iterator+expert*n*(k//32),cute.make_layout((n,k//32),stride=(k//32,1)))
        out=cute.make_tensor(out.iterator+expert*m*n,cute.make_layout((m,n),stride=(n,1)))
        mt,nt=cute.ceil_div(m,cfg.tile_m),cute.ceil_div(n,cfg.tile_n)
        if cutlass.const_expr(cfg.raster=='n'):
            group=block//(nt*cfg.grid_swizzle)
            offset=block%(nt*cfg.grid_swizzle)
            count=cutlass.min(mt-group*cfg.grid_swizzle,cfg.grid_swizzle)
            bm=group*cfg.grid_swizzle+offset%count
            bn=offset//count
        else:
            group=block//(mt*cfg.grid_swizzle)
            offset=block%(mt*cfg.grid_swizzle)
            count=cutlass.min(nt-group*cfg.grid_swizzle,cfg.grid_swizzle)
            bn=group*cfg.grid_swizzle+offset%count
            bm=offset//count
        allocator=cutlass.utils.SmemAllocator()
        qa=allocator.allocate_tensor(Float8E4M3FN,self.a_layout.outer,byte_alignment=1024,swizzle=self.a_layout.inner)
        qb=allocator.allocate_tensor(Float8E4M3FN,self.b_layout.outer,byte_alignment=1024,swizzle=self.b_layout.inner)
        sfa=allocator.allocate_tensor(Float8E8M0FNU,self.sfa_layout,byte_alignment=128)
        sfb=allocator.allocate_tensor(Float8E8M0FNU,self.sfb_layout,byte_alignment=128)
        mma=self.tiled_mma;thr=mma.get_slice(thread)
        pa,pb=thr.partition_A(qa),thr.partition_B(qb)
        ra=mma.make_fragment_A(pa[None,None,None,0]);rb=mma.make_fragment_B(pb[None,None,None,0])
        rsa=sm120_utils.partition_fragment_SFA(sfa[None,None,0],thr,thread)
        rsb=sm120_utils.partition_fragment_SFB(sfb[None,None,0],thr,thread)
        rsa=cute.group_modes(rsa,2,cute.rank(rsa));rsb=cute.group_modes(rsb,2,cute.rank(rsb))
        ca=cute.make_tiled_copy_A(_make_ldmatrix_atom(Float8E4M3FN,False,cfg.a_ldmatrix_matrices),mma)
        cb=cute.make_tiled_copy_B(_make_ldmatrix_atom(Float8E4M3FN,False,cfg.b_ldmatrix_matrices),mma)
        ga=ca.get_slice(thread).partition_S(qa);gb=cb.get_slice(thread).partition_S(qb)
        rca=ca.get_slice(thread).retile(ra);rcb=cb.get_slice(thread).retile(rb)
        csa=cute.make_tiled_copy(_make_scale_s2r_atom(Float8E8M0FNU,cfg.sfa_s2r_bits),
            sm120_utils.get_layoutSFA_TV(mma),(cute.size(mma.permutation_mnk[0]),cute.size(mma.permutation_mnk[2])))
        csb=cute.make_tiled_copy(_make_scale_s2r_atom(Float8E8M0FNU,cfg.sfb_s2r_bits),
            sm120_utils.get_layoutSFB_TV(mma),(cute.size(mma.permutation_mnk[1]),cute.size(mma.permutation_mnk[2])))
        gsa=csa.get_slice(thread).partition_S(sfa);gsb=csb.get_slice(thread).partition_S(sfb)
        rcsa=csa.get_slice(thread).retile(rsa);rcsb=csb.get_slice(thread).retile(rsb)
        coords=thr.partition_C(cute.make_identity_tensor((cfg.tile_m,cfg.tile_n)))
        acc=cute.make_rmem_tensor(coords.shape,Float32);acc.fill(Float32(0))
        for kt in cutlass.range(cute.ceil_div(k,cfg.tile_k),unroll=cfg.k_tile_unroll):
            self._load_operand(a,sa,qa,sfa,self.a_layout,m,cfg.tile_m,bm*cfg.tile_m,kt*cfg.tile_k,thread)
            self._load_operand(b,sb,qb,sfb,self.b_layout,n,cfg.tile_n,bn*cfg.tile_n,kt*cfg.tile_k,thread)
            cute.arch.cp_async_commit_group();cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            for kb in cutlass.range_constexpr(cute.size(ra,mode=[2])):
                cute.copy(ca,ga[None,None,kb,0],rca[None,None,kb])
                cute.copy(cb,gb[None,None,kb,0],rcb[None,None,kb])
                cute.copy(csa,cute.filter_zeros(gsa)[None,None,kb,0],cute.filter_zeros(rcsa)[None,None,kb])
                cute.copy(csb,cute.filter_zeros(gsb)[None,None,kb,0],cute.filter_zeros(rcsb)[None,None,kb])
                cute.gemm(mma,acc,[ra[None,None,kb],rsa[None,None,kb]],
                    [rb[None,None,kb],rsb[None,None,kb]],acc)
            cute.arch.sync_threads()
        for i in cutlass.range_constexpr(cute.size(acc)):
            row,col=bm*cfg.tile_m+coords[i][0],bn*cfg.tile_n+coords[i][1]
            if row<m and col<n:out[row,col]=BFloat16(acc[i])
