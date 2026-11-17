"""On-chip H=128 expert FFN; same MXFP8 hidden rounding as staged execution.

The gate's dead operand allocations become the down GEMM's operands. Hidden
bytes/scales never reach GMEM. This is expert-slot fusion, not fused routing or
token combine. Serial and producer/consumer schedules share one numeric recipe.
"""
from functools import lru_cache

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils.blockscaled_layout as blockscaled_utils
import cutlass.utils.blackwell_helpers as sm120_utils
from cutlass import BFloat16,Float8E4M3FN,Float8E8M0FNU,Float32,Int32,Uint8,Uint16,Uint32,Uint64
from cutlass.experimental.primitives import nvvm_wrapper as nvvm

from .mxfp8 import MXFP8Problem
from .mxfp8_fwd import MXFP8LinearFwdKernel,_make_ldmatrix_atom,_make_scale_s2r_atom
from .mxfp8_glu import MXFP8GLUKernel
from .mxfp8_mlp import GLURecomputeDerivative


class OnchipExpertKernel(MXFP8GLUKernel):
    _emit_hidden=MXFP8GLUKernel._glu_epilogue
    _pack_pair=GLURecomputeDerivative._backward_quant_pair

    def __init__(self,experts,problem,config,down_config,output_features,stages=1,schedule='serial'):
        super().__init__(problem,config,'silu','mxfp8')
        if problem.n!=256 or config.tile_n!=256 or config.tile_k!=128:
            raise ValueError('on-chip family currently requires full H=128 gate tiles and K macro=128')
        if config.persistent or config.glu_scratch!='dedicated' or config.mxfp8_stages!=1:
            raise ValueError('on-chip handoff requires dedicated scratch and nonpersistent single FP8 gate staging')
        if down_config.tile_m!=config.tile_m or down_config.tile_k!=128 or down_config.tile_n not in (64,128):
            raise ValueError('on-chip down tiles must share gate M and use N=64/128, K=128')
        if down_config.mxfp8_stages!=1 or down_config.quant_vec<2:
            raise ValueError('on-chip down uses one A stage and paired quantization')
        if stages not in (1,2,3,4) or stages*down_config.tile_n>256:
            raise ValueError('on-chip weight buffers exceed the reused gate allocation')
        if schedule not in ('serial','warp_specialized'):
            raise ValueError('Unknown on-chip down schedule')
        self.output_features=output_features
        self.down_stages=stages
        self.down_schedule=schedule
        self.down_core=MXFP8LinearFwdKernel(MXFP8Problem(problem.m,output_features,128),down_config)
        if self.num_mma_warps<=self.down_core.num_mma_warps:
            raise ValueError('on-chip producer/consumer ownership needs spare gate MMA warps')
        self.batch_count=experts
        self.batch_out_shape=(problem.m,output_features)
        self.batch_aux_shape=(output_features,128)

    def _setup_down_layouts(self):
        # Build these IR values inside the device entry, not the host entry:
        # unlike the original GEMM's explicitly passed layout arguments, these
        # specialization-private objects cannot capture host-region SSA values.
        self.down_core._setup_static_layouts()
        cfg=self.down_core.config
        kinds={'none':cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_INTER,
               '64b':cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW64,
               '128b':cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW128,
               '32b':cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW32}
        self.down_b_layout=cute.tile_to_shape(cute.nvgpu.warpgroup.make_smem_layout_atom(kinds[cfg.b_swizzle],Float8E4M3FN),
            (cfg.tile_n,128,self.down_stages),order=(0,1,2))
        scale_tile=((cfg.tile_m+127)//128*128,(cfg.tile_n+127)//128*128,128)
        if cfg.tile_n==64:
            from .mxfp8_fwd import _make_sm120_sfa_layout_64
            self.down_sfb_layout=_make_sm120_sfa_layout_64(self.down_core.tiled_mma,128,32,self.down_stages)
        else:
            self.down_sfb_layout=blockscaled_utils.sm120_make_smem_layout_sfb(
                self.down_core.tiled_mma,scale_tile,32,self.down_stages)

    @cute.jit
    def _quantize_weight(self,down,s_b,sfb,tile,stage,thread,threads:cutlass.Constexpr):
        cfg=self.down_core.config
        lane=cute.arch.lane_idx()
        group_threads=32//cfg.quant_vec
        blocks_per_warp=cfg.quant_vec
        for task_group in cutlass.range(thread//32,cfg.tile_n*4//blocks_per_warp,threads//32,unroll=1):
            task=task_group*blocks_per_warp+lane//group_threads
            row,kblock=task//4,task%4
            col=kblock*32+(lane%group_threads)*cfg.quant_vec
            global_row=tile*cfg.tile_n+row
            values=cute.make_rmem_tensor((cfg.quant_vec,),Float32)
            values.fill(Float32(0))
            load_count=cfg.quant_load_bits//16
            for part in cutlass.range_constexpr(cfg.quant_vec//load_count):
                if global_row<self.output_features:
                    if cutlass.const_expr(load_count==1):
                        values[part]=Float32(down[global_row,col+part])
                    else:
                        ptr=down.iterator+down.layout((global_row,col+part*load_count))
                        loaded=nvvm.load_ext(ptr,dtype=Uint16,count=load_count).bitcast(BFloat16)
                        for v in cutlass.range_constexpr(load_count):
                            values[part*load_count+v]=Float32(loaded[v])
            maximum=Float32(0)
            for v in cutlass.range_constexpr(cfg.quant_vec):
                maximum=cute.arch.fmax(maximum,cute.abs(values[v]),nan=True)
            maximum=self.down_core._warp_amax(maximum,group_threads,lane)
            scale,inv=self.down_core._scale_from_amax(maximum)
            quantized=cute.make_rmem_tensor((cfg.quant_vec,),Float8E4M3FN)
            for pair in cutlass.range_constexpr(cfg.quant_vec//2):
                packed=self._pack_pair(values[2*pair],values[2*pair+1],inv,cfg.quant_math)
                quantized[2*pair]=Uint8(packed.bitcast(Uint16)&255).bitcast(Float8E4M3FN)
                quantized[2*pair+1]=Uint8(packed.bitcast(Uint16)>>8).bitcast(Float8E4M3FN)
            if cutlass.const_expr(cfg.quant_store_bits==8):
                for v in cutlass.range_constexpr(cfg.quant_vec):
                    s_b[row,col+v,stage]=quantized[v]
            else:
                word_bytes=cfg.quant_store_bits//8
                for word_index in cutlass.range_constexpr(cfg.quant_vec//word_bytes):
                    word=Uint64(0)
                    for byte in cutlass.range_constexpr(word_bytes):
                        word|=Uint64(quantized[word_index*word_bytes+byte].bitcast(Uint8))<<(byte*8)
                    self.down_core._store_quant_word(s_b,row,col+word_index*word_bytes,stage,word,self.down_b_layout.inner)
            if lane%group_threads==0:
                sfb[row,kblock*32,stage]=scale

    @cute.jit
    def _consume(self,a,b,sfa,sfb,out,block_m,tile,stage,thread):
        core=self.down_core
        cfg=core.config
        mma=core.tiled_mma
        thr=mma.get_slice(thread)
        pa,pb=thr.partition_A(a),thr.partition_B(b)
        ra=mma.make_fragment_A(pa[None,None,None,0])
        rb=mma.make_fragment_B(pb[None,None,None,0])
        rsa=sm120_utils.partition_fragment_SFA(sfa[None,None,0],thr,thread)
        rsb=sm120_utils.partition_fragment_SFB(sfb[None,None,0],thr,thread)
        rsa=cute.group_modes(rsa,2,cute.rank(rsa))
        rsb=cute.group_modes(rsb,2,cute.rank(rsb))
        copy_a=cute.make_tiled_copy_A(_make_ldmatrix_atom(Float8E4M3FN,False,cfg.a_ldmatrix_matrices),mma)
        copy_b=cute.make_tiled_copy_B(_make_ldmatrix_atom(Float8E4M3FN,False,cfg.b_ldmatrix_matrices),mma)
        ca,cb=copy_a.get_slice(thread),copy_b.get_slice(thread)
        ga,gb=ca.partition_S(a),cb.partition_S(b)
        rca,rcb=ca.retile(ra),cb.retile(rb)
        copy_sa=cute.make_tiled_copy(_make_scale_s2r_atom(Float8E8M0FNU,cfg.sfa_s2r_bits),
            sm120_utils.get_layoutSFA_TV(mma),(cute.size(mma.permutation_mnk[0]),cute.size(mma.permutation_mnk[2])))
        copy_sb=cute.make_tiled_copy(_make_scale_s2r_atom(Float8E8M0FNU,cfg.sfb_s2r_bits),
            sm120_utils.get_layoutSFB_TV(mma),(cute.size(mma.permutation_mnk[1]),cute.size(mma.permutation_mnk[2])))
        csa,csb=copy_sa.get_slice(thread),copy_sb.get_slice(thread)
        gsa,gsb=csa.partition_S(sfa),csb.partition_S(sfb)
        rcsa,rcsb=csa.retile(rsa),csb.retile(rsb)
        coords=thr.partition_C(cute.make_identity_tensor((cfg.tile_m,cfg.tile_n)))
        acc=cute.make_rmem_tensor(coords.shape,Float32);acc.fill(Float32(0))
        for kb in cutlass.range_constexpr(cute.size(ra,mode=[2])):
            cute.copy(copy_a,ga[None,None,kb,0],rca[None,None,kb])
            cute.copy(copy_b,gb[None,None,kb,stage],rcb[None,None,kb])
            cute.copy(copy_sa,cute.filter_zeros(gsa)[None,None,kb,0],cute.filter_zeros(rcsa)[None,None,kb])
            cute.copy(copy_sb,cute.filter_zeros(gsb)[None,None,kb,stage],cute.filter_zeros(rcsb)[None,None,kb])
            cute.gemm(mma,acc,[ra[None,None,kb],rsa[None,None,kb]],
                      [rb[None,None,kb],rsb[None,None,kb]],acc)
        for i in cutlass.range_constexpr(cute.size(acc)):
            row=block_m*self.config.tile_m+coords[i][0]
            col=tile*cfg.tile_n+coords[i][1]
            if row<self.problem.m and col<self.output_features:
                out[row,col]=BFloat16(acc[i])

    @cute.jit
    def _glu_epilogue(self,storage,acc,coords,out,down,block_m,block_n,thread,
                      tma_out,tma_tensor,out_layout,work_slot):
        self._setup_down_layouts()
        cfg=self.down_core.config
        # All original consumers must finish before dead operand storage is
        # repurposed. The gate scratch remains disjoint during hidden emission.
        cute.arch.sync_threads()
        qh=storage.q_b.get_tensor(cute.make_layout((self.config.tile_m,128),stride=(128,1)))
        rowscale=storage.scale_b.get_tensor(cute.make_layout((self.config.tile_m,4),stride=(4,1)))
        for index in cutlass.range(thread,self.config.tile_m*128,self.config.num_threads,unroll=1):
            qh[index//128,index%128]=Float8E4M3FN(0)
        for index in cutlass.range(thread,self.config.tile_m*4,self.config.num_threads,unroll=1):
            rowscale[index//4,index%4]=Uint8(0).bitcast(Float8E8M0FNU)
        cute.arch.sync_threads()
        qh_global=cute.make_tensor(qh.iterator-block_m*self.config.tile_m*128,
            cute.make_layout((self.problem.m,128),stride=(128,1)))
        scale_global=cute.make_tensor(cute.recast_ptr(rowscale.iterator,dtype=Uint8)-block_m*self.config.tile_m*4,
            cute.make_layout((self.problem.m,4),stride=(4,1)))
        self._emit_hidden(storage,acc,coords,qh_global,scale_global,block_m,block_n,thread,
                          tma_out,tma_tensor,out_layout,work_slot)
        a=storage.q_a.get_tensor(self.down_core.a_smem_layout.outer,swizzle=self.down_core.a_smem_layout.inner)
        sfa=storage.scale_a.get_tensor(self.down_core.sfa_smem_layout)
        for index in cutlass.range(thread,self.config.tile_m*128,self.config.num_threads,unroll=1):
            a[index//128,index%128,0]=qh[index//128,index%128]
        for index in cutlass.range(thread,self.config.tile_m*4,self.config.num_threads,unroll=1):
            sfa[index//4,(index%4)*32,0]=rowscale[index//4,index%4]
        cute.arch.sync_threads()
        b=storage.q_b.get_tensor(self.down_b_layout.outer,swizzle=self.down_b_layout.inner)
        scratch=storage.glu_scratch.get_tensor(cute.make_layout((self.glu_scratch_elems,),stride=(1,)))
        sfb=cute.make_tensor(cute.recast_ptr(scratch.iterator,dtype=Float8E8M0FNU),self.down_sfb_layout)
        tiles=cute.ceil_div(self.output_features,cfg.tile_n)
        if cutlass.const_expr(self.down_schedule=='serial'):
            for tile in cutlass.range(tiles,unroll=1):
                stage=tile%self.down_stages
                if thread<self.config.num_mma_warps*32:
                    self._quantize_weight(down,b,sfb,tile,stage,thread,self.config.num_mma_warps*32)
                cute.arch.sync_threads()
                if thread<cfg.num_mma_warps*32:self._consume(a,b,sfa,sfb,out,block_m,tile,stage,thread)
                cute.arch.sync_threads()
        else:
            consumer_threads=cfg.num_mma_warps*32
            producer_threads=(self.config.num_mma_warps-cfg.num_mma_warps)*32
            barrier_offset=(cute.cosize(self.down_sfb_layout)+15)//16*16
            barriers=cute.recast_ptr(cute.recast_ptr(scratch.iterator,dtype=Uint8)+barrier_offset,dtype=cutlass.Int64)
            pipe=pipeline.PipelineAsync.create(num_stages=self.down_stages,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread,producer_threads),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread,consumer_threads),
                barrier_storage=barriers)
            producer=pipeline.PipelineState(self.down_stages,Int32(0),Int32(0),Int32(1))
            consumer=pipeline.PipelineState(self.down_stages,Int32(0),Int32(0),Int32(0))
            if thread>=consumer_threads and thread<self.config.num_mma_warps*32:
                for tile in cutlass.range(tiles,unroll=1):
                    pipe.producer_acquire(producer)
                    self._quantize_weight(down,b,sfb,tile,producer.index,thread-consumer_threads,producer_threads)
                    pipe.producer_commit(producer);producer.advance()
                pipe.producer_tail(producer)
            elif thread<consumer_threads:
                for tile in cutlass.range(tiles,unroll=1):
                    ready=pipe.consumer_try_wait(consumer)
                    pipe.consumer_wait(consumer,ready)
                    self._consume(a,b,sfa,sfb,out,block_m,tile,consumer.index,thread)
                    pipe.consumer_release(consumer);consumer.advance()
            cute.arch.sync_threads()


@lru_cache(maxsize=128)
def compile_onchip_experts(experts,problem,gate_config,down_config,output_features,stages=1,schedule='serial'):
    kernel=OnchipExpertKernel(experts,problem,gate_config,down_config,output_features,stages,schedule)
    def fake(rows,cols):
        return cute.runtime.make_fake_tensor(BFloat16,(rows,cols),stride=(cols,1),assumed_align=16)
    x,w=fake(experts*problem.m,problem.k),fake(experts*problem.n,problem.k)
    out,down=fake(experts*problem.m,output_features),fake(experts*output_features,128)
    stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(kernel,x,w,out,down,down,down,down,stream,
        options='--enable-tvm-ffi --opt-level 3 '
        f"--ptxas-options '-O3 --maxrregcount={gate_config.maxrregcount}'")
