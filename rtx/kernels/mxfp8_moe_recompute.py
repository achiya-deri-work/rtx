"""Joint expert epilogues over retired TMA operand storage."""
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float8E4M3FN, Float32, Int16, Int32, Uint8, Uint16, Uint32
from cutlass.experimental.primitives import nvvm_wrapper as nvvm

from .mxfp8_moe_prepared import ExpertJointRecompute


class OperandReuseJointRecompute(ExpertJointRecompute):
    @cute.jit
    def _scratch(self, storage, cols):
        cfg=self.config
        layout=cute.make_layout((cfg.tile_m,cols),stride=(cols,1))
        if cutlass.const_expr(cfg.glu_scratch_swizzle=='xor_16b'):
            layout=cute.make_composed_layout(cute.make_swizzle(3,2,3),0,layout)
        return cute.make_tensor(cute.recast_ptr(storage.b.data_ptr(),dtype=Float32),layout)

    @cute.jit
    def _glu_epilogue(self,storage,acc,coords,out,aux,bm,bn,thread,
                      tma_out,tma_out_tensor,out_layout,work_slot):
        # Every MMA warp has waited on the transaction barrier for *each*
        # K tile and released it after its operand reads. The producer has
        # issued its final transfer; these schedules have no next tile.
        # Gather all roles here before assigning a new owner to the bytes.
        cute.arch.sync_threads()
        ExpertJointRecompute._glu_epilogue(
            self,storage,acc,coords,out,aux,bm,bn,thread,
            tma_out,tma_out_tensor,out_layout,work_slot)


class DualOperandJointRecompute(OperandReuseJointRecompute):
    """Retire the gate/up fragments immediately, writing products and dP once."""

    @cute.jit
    def _store_bf16_block(self,values,output,scales,row,col,ld,valid,policy:cutlass.Constexpr):
        maximum=Float32(0)
        bits=Int32(0)
        for i in cutlass.range_constexpr(4):
            v=values[i]
            if cutlass.const_expr(policy[1]=='bf16_bits'):
                bits=cutlass.max(bits,Int32(BFloat16(v).bitcast(Uint16))&Int32(0x7fff))
            else:
                maximum=cute.arch.fmax(maximum,cute.abs(v),nan=True)
        if cutlass.const_expr(policy[1]=='bf16_bits'):
            maximum=Float32(Uint16(bits).bitcast(BFloat16))
        amax=self._warp_amax(maximum,8,cute.arch.lane_idx())
        scale,inv=self._scale_from_amax(amax)
        word=Uint32(0)
        for pair in cutlass.range_constexpr(2):
            packed=self._backward_quant_pair(values[2*pair],values[2*pair+1],inv,policy[0])
            word|=Uint32(packed.bitcast(Uint16))<<(16*pair)
        if valid:
            cute.recast_ptr(output.iterator+row*ld+col,dtype=Uint32)[0]=word
            if cute.arch.lane_idx()%8==0:
                scales[row,col//32]=scale

    @cute.jit
    def _glu_epilogue(self,storage,acc,coords,out,aux,bm,bn,thread,
                      tma_out,tma_out_tensor,out_layout,work_slot):
        cfg=self.config
        m,h=self.problem.m,self.problem.n//2
        width=cfg.tile_n//2
        dh,qh,sh,qht,sht,qp,sp,qpt,spt=self._local_joint_aux(aux)
        product_layout=cute.make_layout((cfg.tile_m,width),stride=(width,1))
        grad_layout=cute.make_layout((cfg.tile_m,cfg.tile_n),stride=(cfg.tile_n,1))
        if cutlass.const_expr(cfg.glu_scratch_swizzle=='xor_16b'):
            product_layout=cute.make_composed_layout(cute.make_swizzle(3,2,3),0,product_layout)
            grad_layout=cute.make_composed_layout(
                cute.make_swizzle(3,3,cfg.tile_n.bit_length()-2),0,grad_layout)
        products=cute.make_tensor(cute.recast_ptr(storage.a.data_ptr(),dtype=Float32),product_layout)
        grad_ptr=cute.recast_ptr(storage.b.data_ptr(),dtype=BFloat16)
        derivatives=cute.make_tensor(grad_ptr,grad_layout)
        # Same proven ownership boundary as OperandReuseJointRecompute.
        cute.arch.sync_threads()
        if thread<cfg.num_mma_warps*32:
            for pair in cutlass.range_constexpr(cute.size(acc)//2):
                r,c=coords[2*pair][0],coords[2*pair][1]//2
                if r<m and c<h:
                    gate,up=acc[2*pair],acc[2*pair+1]
                    sigmoid=self._sigmoid(gate)
                    if cutlass.const_expr(cfg.glu_silu_math=='division'):
                        act=self._activate(gate)
                    else:
                        act=gate*sigmoid
                    derivative=sigmoid*(Float32(1)+gate*(Float32(1)-sigmoid))
                    grad=Float32(dh[r,c])
                    lr,lc=r-bm*cfg.tile_m,c-bn*width
                    derivatives[lr,lc]=BFloat16(grad*up*derivative)
                    derivatives[lr,lc+width]=BFloat16(grad*act)
                    products[lr,lc]=act*up
        cute.arch.sync_threads()
        for group in cutlass.range(cute.ceil_div(cfg.tile_m*cfg.tile_n,4*cfg.num_threads),unroll=1):
            idx=(thread+group*cfg.num_threads)*4
            lr,lc=idx//cfg.tile_n,idx%cfg.tile_n
            gr,gc=bm*cfg.tile_m+lr,bn*width+lc%width
            target=2*gr+lc//width
            valid=lr<cfg.tile_m and gr<m and gc<h
            vals=cute.make_rmem_tensor((4,),Float32)
            vals.fill(Float32(0))
            if valid:
                for v in cutlass.range_constexpr(4):
                    vals[v]=Float32(derivatives[lr,lc+v])
            self._store_bf16_block(vals,qp,sp,target,gc,h,valid,self.gradient_quant_policy)
        self._gradient_transpose_quant_epilogue(derivatives,qpt,spt,bm,bn,thread)
        # All dP consumers must finish before B is overwritten by decoded H.
        cute.arch.sync_threads()
        decoded=cute.make_tensor(grad_ptr,
            cute.make_layout((width,cfg.tile_m+1),stride=(cfg.tile_m+1,1)))
        # FP32 H -> original feature-axis MXFP8 link -> FP32 decode -> BF16.
        # Work in small per-lane vectors; no rounded fragment lives across a
        # CTA barrier or across derivative quantization.
        for group in cutlass.range(cute.ceil_div(cfg.tile_m*width,cfg.num_threads*cfg.glu_quant_vec),unroll=1):
            idx=(thread+group*cfg.num_threads)*cfg.glu_quant_vec
            lr,lc=idx//width,idx%width
            gr,gc=bm*cfg.tile_m+lr,bn*width+lc
            valid=lr<cfg.tile_m and gr<m and gc<h
            vals=cute.make_rmem_tensor((cfg.glu_quant_vec,),Float32)
            maximum=Float32(0)
            for v in cutlass.range_constexpr(cfg.glu_quant_vec):
                value=Float32(0)
                if valid:value=products[lr,lc+v]
                vals[v]=value
                maximum=cute.arch.fmax(maximum,cute.abs(value),nan=True)
            amax=self._warp_amax(maximum,32//cfg.glu_quant_vec,cute.arch.lane_idx())
            scale,inv=self._scale_from_amax(amax)
            if valid:
                if cute.arch.lane_idx()%(32//cfg.glu_quant_vec)==0:sh[gr,gc//32]=scale
                for pair in cutlass.range_constexpr((cfg.glu_quant_vec+1)//2):
                    lo=2*pair
                    hi=cutlass.min(lo+1,cfg.glu_quant_vec-1)
                    packed=nvvm.inline_ptx_hl(
                        "cvt.rn.satfinite.e4m3x2.f32 {$w0}, {$r0}, {$r1};",
                        write_only_types=[Int16],read_only_args=[vals[hi]*inv,vals[lo]*inv])
                    for half in cutlass.range_constexpr(min(2,cfg.glu_quant_vec)):
                        rounded=Uint8(packed>>(8*half)).bitcast(Float8E4M3FN)
                        qh[gr,gc+lo+half]=rounded
                        decoded[lc+lo+half,lr]=BFloat16(Float32(rounded)*Float32(scale))
        cute.arch.sync_threads()
        # The independently rounded token-axis blocks never cross 32 rows
        # or expert boundaries, even on a clipped final M or N tile.
        for group in cutlass.range(cute.ceil_div(cfg.tile_m*width,4*cfg.num_threads),unroll=1):
            idx=(thread+group*cfg.num_threads)*4
            hc,rr=idx//cfg.tile_m,idx%cfg.tile_m
            gh,gm=bn*width+hc,bm*cfg.tile_m+rr
            valid=hc<width and gh<h and gm<m
            vals=cute.make_rmem_tensor((4,),Float32)
            vals.fill(Float32(0))
            if valid:
                for v in cutlass.range_constexpr(4):vals[v]=Float32(decoded[hc,rr+v])
            self._store_bf16_block(vals,qht,sht,gh,gm,m,valid,self.hidden_quant_policy)
