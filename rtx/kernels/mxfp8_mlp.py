"""Strict two-launch MLP building blocks and recomputation helpers."""
from functools import lru_cache

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float8E4M3FN, Float8E8M0FNU, Float32, Int16, Int32, Uint8, Uint16, Uint32
from cutlass.experimental.primitives import nvvm_wrapper as nvvm

from .mxfp8_fwd import MXFP8LinearFwdKernel
from .mxfp8_glu import MXFP8GLUKernel
from .mxfp8_glu_prequant import MXFP8PrequantGLUKernel


@lru_cache(maxsize=128)
def compile_glu_with_down_quant(problem,config,activation,down_rows,quant_config):
    from .mxfp8_quant import MXFP8QuantKernel
    if quant_config.scale_layout != "row_major" or quant_config.scale_rounding != "infinity":
        raise ValueError("producer down quantization requires row-major infinity-rounded scales")
    kernel=MXFP8GLUKernel(problem,config,activation,"mxfp8")
    kernel.quant_down=True
    hidden=problem.n//2
    width=(hidden+31)//32*32
    kernel.down_quantizer=MXFP8QuantKernel(down_rows,hidden,quant_config)
    def fake(dtype,rows,cols):
        return cute.runtime.make_fake_tensor(dtype,(rows,cols),stride=(cols,1),assumed_align=16)
    x=fake(BFloat16,problem.m,problem.k)
    gu=fake(BFloat16,problem.n,problem.k)
    out=fake(Float8E4M3FN,problem.m,width)
    scales=fake(cutlass.Uint8,problem.m,width//32)
    down=fake(BFloat16,down_rows,hidden)
    qw=fake(Float8E4M3FN,down_rows,width)
    sw=fake(Float8E8M0FNU,down_rows,width//32)
    stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(kernel,x,gu,out,scales,down,qw,sw,stream,
                        options="--enable-tvm-ffi --opt-level 3 "
                        f"--ptxas-options '-O3 --maxrregcount={config.maxrregcount}'")


def mixed_down_rejection(problem, config):
    reason = config.implementation_rejection(problem)
    if reason:
        return reason
    if config.load_engine != "scalar":
        return "packed-X/dynamic-W currently requires scalar transport"
    return None


@lru_cache(maxsize=128)
def compile_mixed_down(problem, config):
    reason = mixed_down_rejection(problem,config)
    if reason:
        raise ValueError(reason)
    kernel = MXFP8LinearFwdKernel(problem,config)
    kernel.a_prequantized = True
    width = (problem.k+31)//32*32
    def fake(dtype,rows,cols):
        return cute.runtime.make_fake_tensor(dtype,(rows,cols),stride=(cols,1),assumed_align=16)
    x = fake(Float8E4M3FN,problem.m,width)
    w = fake(BFloat16,problem.n,problem.k)
    out = fake(BFloat16,problem.m,problem.n)
    scales = fake(Float8E8M0FNU,problem.m,width//32)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(kernel,x,w,out,scales,scales,scales,scales,stream,
                        options="--enable-tvm-ffi --opt-level 3 "
                        f"--ptxas-options '-O3 --maxrregcount={config.maxrregcount}'")


class GLURecomputeDerivative(MXFP8GLUKernel):
    """Recompute FP32 projections, differentiate GLU, store only BF16 dGate/dUp."""

    @cute.jit
    def _backward_quant_pair(self,value0,value1,inv,math:cutlass.Constexpr):
        """Pack already-BF16 operands in standalone-quantizer byte order."""
        if cutlass.const_expr(math=='bf16x2'):
            bits0=Int32(BFloat16(value0).bitcast(Uint16))
            bits1=Int32(BFloat16(value1).bitcast(Uint16))
            inv_bits=Int32(BFloat16(inv).bitcast(Uint16))
            scaled=nvvm.mul_bf16x2(bits0|(bits1<<16),inv_bits|(inv_bits<<16))
            return nvvm.inline_ptx_hl(
                "cvt.rn.satfinite.e4m3x2.bf16x2 {$w0}, {$r0};",
                write_only_types=[Int16],read_only_args=[scaled])
        else:
            return nvvm.inline_ptx_hl(
                "cvt.rn.satfinite.e4m3x2.f32 {$w0}, {$r0}, {$r1};",
                write_only_types=[Int16],read_only_args=[value1*inv,value0*inv])

    @cute.jit
    def _hidden_quant_epilogue(self,scratch,quantized,scales,block_m,block_n,tidx):
        """Forward-rounded hidden -> independent leading-axis block quantization.

        Registers bridge two disjoint uses of the same shared allocation. No
        BF16 hidden tensor is written to GMEM by this path.
        """
        cfg=self.config
        width=cfg.tile_n//2
        iterations=cute.ceil_div(cfg.tile_m*width,cfg.num_threads*cfg.glu_quant_vec)
        rounded_values=cute.make_rmem_tensor((iterations*cfg.glu_quant_vec,),BFloat16)
        rounded_values.fill(BFloat16(0))
        for i in cutlass.range_constexpr(iterations):
            index=(tidx+i*cfg.num_threads)*cfg.glu_quant_vec
            row,col=index//width,index%width
            gr,gc=block_m*cfg.tile_m+row,block_n*width+col
            values=cute.make_rmem_tensor((cfg.glu_quant_vec,),Float32)
            maximum=Float32(0)
            for v in cutlass.range_constexpr(cfg.glu_quant_vec):
                value=Float32(0)
                if row<cfg.tile_m and gr<self.problem.m and gc+v<self.problem.n//2:
                    value=scratch[row,col+v]
                values[v]=value
                maximum=cute.arch.fmax(maximum,cute.abs(value),nan=True)
            amax=self._warp_amax(maximum,32//cfg.glu_quant_vec,cute.arch.lane_idx())
            scale,inv=self._scale_from_amax(amax)
            for pair in cutlass.range_constexpr((cfg.glu_quant_vec+1)//2):
                v0=2*pair
                v1=cutlass.min(v0+1,cfg.glu_quant_vec-1)
                packed=nvvm.inline_ptx_hl(
                    "cvt.rn.satfinite.e4m3x2.f32 {$w0}, {$r0}, {$r1};",
                    write_only_types=[Int16],read_only_args=[values[v1]*inv,values[v0]*inv])
                for half in cutlass.range_constexpr(min(2,cfg.glu_quant_vec)):
                    rounded=Uint8(packed>>(8*half)).bitcast(Float8E4M3FN)
                    rounded_values[i*cfg.glu_quant_vec+v0+half]=BFloat16(Float32(rounded)*Float32(scale))
        # All FP32 products have been consumed before reinterpreting storage.
        cute.arch.sync_threads()
        hidden=cute.make_tensor(cute.recast_ptr(scratch.iterator,dtype=BFloat16),
            cute.make_layout((width,cfg.tile_m+1),stride=(cfg.tile_m+1,1)))
        for i in cutlass.range_constexpr(iterations):
            index=(tidx+i*cfg.num_threads)*cfg.glu_quant_vec
            row,col=index//width,index%width
            if row<cfg.tile_m:
                for v in cutlass.range_constexpr(cfg.glu_quant_vec):
                    hidden[col+v,row]=rounded_values[i*cfg.glu_quant_vec+v]
        cute.arch.sync_threads()
        storage_m=((self.problem.m+31)//32)*32
        for group in cutlass.range(cute.ceil_div(width*cfg.tile_m,4*cfg.num_threads),unroll=1):
            idx=(tidx+group*cfg.num_threads)*4
            hr,mr=idx//cfg.tile_m,idx%cfg.tile_m
            gh,gm=block_n*width+hr,block_m*cfg.tile_m+mr
            vals=cute.make_rmem_tensor((4,),Float32)
            maximum=Float32(0)
            maximum_bits=Int32(0)
            for v in cutlass.range_constexpr(4):
                value=Float32(0)
                if hr<width and gh<self.problem.n//2 and gm+v<self.problem.m:
                    value=Float32(hidden[hr,mr+v])
                vals[v]=value
                if cutlass.const_expr(self.hidden_quant_policy[1]=='bf16_bits'):
                    maximum_bits=cutlass.max(maximum_bits,Int32(BFloat16(value).bitcast(Uint16))&Int32(0x7fff))
                else:
                    maximum=cute.arch.fmax(maximum,cute.abs(value),nan=True)
            if cutlass.const_expr(self.hidden_quant_policy[1]=='bf16_bits'):
                maximum=Float32(Uint16(maximum_bits).bitcast(BFloat16))
            amax=self._warp_amax(maximum,8,cute.arch.lane_idx())
            scale,inv=self._scale_from_amax(amax)
            word=Uint32(0)
            for pair in cutlass.range_constexpr(2):
                packed=self._backward_quant_pair(vals[2*pair],vals[2*pair+1],inv,self.hidden_quant_policy[0])
                word|=Uint32(packed.bitcast(Uint16))<<(16*pair)
            if hr<width and gh<self.problem.n//2 and gm<storage_m:
                ptr=cute.recast_ptr(quantized.iterator+gh*storage_m+gm,dtype=Uint32)
                ptr[0]=word
                if cute.arch.lane_idx()%8==0:
                    scales[gh,gm//32]=scale
        cute.arch.sync_threads()

    @cute.jit
    def _glu_epilogue(self,storage,accumulators,coordinates,out,grad_h,
                      block_m,block_n,tidx,tma_out,tma_out_tensor,out_layout,work_slot):
        if cutlass.const_expr(self.emit_hidden_quant):
            hidden_quant,hidden_scales=grad_h[1],grad_h[2]
            grad_h=grad_h[0]
        if cutlass.const_expr(self.emit_gradient_quant):
            gradient_quant,gradient_scales=grad_h[1],grad_h[2]
            grad_h=grad_h[0]
        if cutlass.const_expr(self.emit_hidden):
            if cutlass.const_expr(self.separate_hidden):
                hidden_out=grad_h[1]
                grad_h=grad_h[0]
            else:
                hidden_out=cute.make_tensor(
                    out.iterator+((2*self.problem.m+7)//8)*8*(self.problem.n//2),
                    cute.make_layout((self.problem.m,self.problem.n//2),
                                     stride=(self.problem.n//2,1)))
        if cutlass.const_expr(self.emit_hidden and not self.warp_quant):
            scratch=self._scratch(storage,self.config.tile_n//2)
            if cutlass.const_expr(self.config.glu_grad_load == "shared"):
                # A coalesced cooperative load before accumulator-order reads.
                # Each derivative owner replaces only its own gradient slot
                # with the reconstructed hidden value after the CTA barrier.
                for i in cutlass.range(cute.ceil_div(self.config.tile_m*self.config.tile_n//2,self.config.num_threads),unroll=1):
                    index=tidx+i*self.config.num_threads
                    row,col=index//(self.config.tile_n//2),index%(self.config.tile_n//2)
                    gr,gc=block_m*self.config.tile_m+row,block_n*self.config.tile_n//2+col
                    if row < self.config.tile_m:
                        value=Float32(0)
                        if gr < self.problem.m and gc < self.problem.n//2:
                            value=Float32(grad_h[gr,gc])
                        scratch[row,col]=value
                cute.arch.sync_threads()
        if cutlass.const_expr(self.emit_hidden and (self.warp_quant or self.config.glu_grad_store=="exchange")):
            products=cute.make_rmem_tensor((cute.size(accumulators)//2,),Float32)
            products.fill(Float32(0))
        if cutlass.const_expr(self.emit_hidden and not self.warp_quant and self.config.glu_grad_store=="exchange"):
            # Same bytes as H's FP32 scratch; the lifetimes are separated by
            # CTA barriers. Use a flat BF16 view for a coalesced dP writeout.
            grad_scratch=cute.make_tensor(cute.recast_ptr(scratch.iterator,dtype=BFloat16),
                                         cute.make_layout((self.config.tile_m,self.config.tile_n),stride=(self.config.tile_n,1)))
        if cutlass.const_expr(self.config.glu_grad_load == "prefetch"):
            gradients=cute.make_rmem_tensor((cute.size(accumulators)//2,),Float32)
            gradients.fill(Float32(0))
            if tidx < self.config.num_mma_warps*32:
                for i in cutlass.range_constexpr(cute.size(accumulators)//2):
                    row,col=coordinates[2*i][0],coordinates[2*i][1]//2
                    if row < self.problem.m and col < self.problem.n//2:
                        gradients[i]=Float32(grad_h[row,col])
        if tidx < self.config.num_mma_warps*32:
            for i in cutlass.range_constexpr(cute.size(accumulators)//2):
                coord = coordinates[2*i]
                row, col = coord[0], coord[1]//2
                gate, up = accumulators[2*i], accumulators[2*i+1]
                if cutlass.const_expr(self.activation == "silu"):
                    sigmoid = self._sigmoid(gate)
                    if cutlass.const_expr(self.config.glu_silu_math != "division"):
                        act=gate*sigmoid
                    else:
                        act=self._activate(gate)
                    derivative = sigmoid*(Float32(1)+gate*(Float32(1)-sigmoid))
                elif cutlass.const_expr(self.activation == "gelu"):
                    act=self._activate(gate)
                    derivative = (Float32(.5)*(Float32(1)+cute.erf(gate*Float32(.7071067811865476)))
                                  + gate*Float32(.3989422804014327)*cute.exp(Float32(-.5)*gate*gate))
                else:
                    act=self._activate(gate)
                    derivative = Float32(0)
                    if gate > Float32(0):
                        derivative = Float32(1)
                if row < self.problem.m and col < self.problem.n//2:
                    if cutlass.const_expr(self.config.glu_grad_load == "prefetch"):
                        grad = gradients[i]
                    elif cutlass.const_expr(self.config.glu_grad_load == "shared" and self.emit_hidden and not self.warp_quant):
                        grad = scratch[row-block_m*self.config.tile_m,col-block_n*self.config.tile_n//2]
                    else:
                        grad = Float32(grad_h[row,col])
                    if cutlass.const_expr(self.emit_hidden and not self.warp_quant and self.config.glu_grad_store=="exchange"):
                        lr,lc=row-block_m*self.config.tile_m,col-block_n*self.config.tile_n//2
                        grad_scratch[lr,lc]=BFloat16(grad*up*derivative)
                        grad_scratch[lr,lc+self.config.tile_n//2]=BFloat16(grad*act)
                    elif cutlass.const_expr(self.emit_hidden):
                        out[2*row,col] = BFloat16(grad*up*derivative)
                        out[2*row+1,col] = BFloat16(grad*act)
                    else:
                        out[row,col] = BFloat16(grad*up*derivative)
                        out[row,col+self.problem.n//2] = BFloat16(grad*act)
                if cutlass.const_expr(self.emit_hidden and (self.warp_quant or self.config.glu_grad_store=="exchange")):
                    value=Float32(0)
                    if row < self.problem.m and col < self.problem.n//2:
                        value=act*up
                    products[i]=value
                elif cutlass.const_expr(self.emit_hidden):
                    scratch[row-block_m*self.config.tile_m,col-block_n*(self.config.tile_n//2)] = act*up
        if cutlass.const_expr(self.emit_hidden and not self.warp_quant and self.config.glu_grad_store=="exchange"):
            cute.arch.sync_threads()
            cfg=self.config
            for group in cutlass.range(cute.ceil_div(cfg.tile_m*cfg.tile_n,2*cfg.num_threads),unroll=1):
                idx=(tidx+group*cfg.num_threads)*2
                lr,lc=idx//cfg.tile_n,idx%cfg.tile_n
                gr=block_m*cfg.tile_m+lr
                gc=block_n*cfg.tile_n//2+lc%(cfg.tile_n//2)
                out_row=2*gr+lc//(cfg.tile_n//2)
                if lr<cfg.tile_m and gr<self.problem.m and gc<self.problem.n//2:
                    if gc+1<self.problem.n//2 and (out_row*(self.problem.n//2)+gc)%2==0:
                        word=Uint32(grad_scratch[lr,lc].bitcast(Uint16))|(Uint32(grad_scratch[lr,lc+1].bitcast(Uint16))<<16)
                        ptr=cute.recast_ptr(out.iterator+out_row*(self.problem.n//2)+gc,dtype=Uint32)
                        ptr[0]=word
                    else:
                        out[out_row,gc]=grad_scratch[lr,lc]
                        if gc+1<self.problem.n//2:
                            out[out_row,gc+1]=grad_scratch[lr,lc+1]
            if cutlass.const_expr(self.emit_gradient_quant):
                # Quantize the *rounded BF16* derivatives, before these bytes
                # are reused for hidden reconstruction. Each eight-lane group
                # owns one contiguous 32-value scale block in one branch.
                for group in cutlass.range(cute.ceil_div(cfg.tile_m*cfg.tile_n,4*cfg.num_threads),unroll=1):
                    idx=(tidx+group*cfg.num_threads)*4
                    lr,lc=idx//cfg.tile_n,idx%cfg.tile_n
                    gr=block_m*cfg.tile_m+lr
                    gc=block_n*cfg.tile_n//2+lc%(cfg.tile_n//2)
                    out_row=2*gr+lc//(cfg.tile_n//2)
                    vals=cute.make_rmem_tensor((4,),Float32)
                    maximum=Float32(0)
                    maximum_bits=Int32(0)
                    for v in cutlass.range_constexpr(4):
                        value=Float32(0)
                        if lr<cfg.tile_m and gr<self.problem.m and gc+v<self.problem.n//2:
                            value=Float32(grad_scratch[lr,lc+v])
                        vals[v]=value
                        if cutlass.const_expr(self.gradient_quant_policy[1]=='bf16_bits'):
                            maximum_bits=cutlass.max(maximum_bits,Int32(BFloat16(value).bitcast(Uint16))&Int32(0x7fff))
                        else:
                            magnitude=(value.bitcast(Int32)&Int32(0x7fffffff)).bitcast(Float32)
                            maximum=cute.arch.fmax(maximum,magnitude,nan=True)
                    if cutlass.const_expr(self.gradient_quant_policy[1]=='bf16_bits'):
                        maximum=Float32(Uint16(maximum_bits).bitcast(BFloat16))
                    maximum=self._warp_amax(maximum,8,cute.arch.lane_idx())
                    scale,inv=self._scale_from_amax(maximum)
                    word=Uint32(0)
                    for pair in cutlass.range_constexpr(2):
                        packed=self._backward_quant_pair(vals[2*pair],vals[2*pair+1],inv,self.gradient_quant_policy[0])
                        word|=Uint32(packed.bitcast(Uint16))<<(16*pair)
                    if lr<cfg.tile_m and gr<self.problem.m and gc<self.problem.n//2:
                        ptr=cute.recast_ptr(gradient_quant.iterator+out_row*(self.problem.n//2)+gc,dtype=Uint32)
                        ptr[0]=word
                        if cute.arch.lane_idx()%8==0:
                            gradient_scales[out_row,gc//32]=scale
            cute.arch.sync_threads()
            if tidx<cfg.num_mma_warps*32:
                for i in cutlass.range_constexpr(cute.size(products)):
                    row,col=coordinates[2*i][0],coordinates[2*i][1]//2
                    scratch[row-block_m*cfg.tile_m,col-block_n*cfg.tile_n//2]=products[i]
        if cutlass.const_expr(self.emit_hidden and self.warp_quant):
            quantized,decoded_scales=self._quantize_registers(products,coordinates,storage,tidx,block_m,block_n)
            if tidx < self.config.num_mma_warps*32:
                for i in cutlass.range_constexpr(cute.size(products)):
                    row,col=coordinates[2*i][0],coordinates[2*i][1]//2
                    if row < self.problem.m and col < self.problem.n//2:
                        hidden_out[row,col]=BFloat16(Float32(quantized[i])*decoded_scales[i])
        cute.arch.sync_threads()
        if cutlass.const_expr(self.emit_hidden_quant):
            self._hidden_quant_epilogue(scratch,hidden_quant,hidden_scales,block_m,block_n,tidx)
        if cutlass.const_expr(self.emit_hidden and not self.warp_quant and not self.emit_hidden_quant):
            cfg=self.config
            for i in range(cute.ceil_div(cfg.tile_m*(cfg.tile_n//2),cfg.num_threads*cfg.glu_quant_vec)):
                index=(tidx+i*cfg.num_threads)*cfg.glu_quant_vec
                row,col=index//(cfg.tile_n//2),index%(cfg.tile_n//2)
                gr,gc=block_m*cfg.tile_m+row,block_n*(cfg.tile_n//2)+col
                values=cute.make_rmem_tensor((cfg.glu_quant_vec,),Float32)
                maximum=Float32(0)
                for v in cutlass.range_constexpr(cfg.glu_quant_vec):
                    value=Float32(0)
                    if row<cfg.tile_m and gr<self.problem.m and gc+v<self.problem.n//2:
                        value=scratch[row,col+v]
                    values[v]=value
                    maximum=cute.arch.fmax(maximum,cute.abs(value),nan=True)
                amax=self._warp_amax(maximum,32//cfg.glu_quant_vec,cute.arch.lane_idx())
                scale,inv=self._scale_from_amax(amax)
                # Match the forward's packed converter: two distinct FP32
                # values per issue, without rounding the GLU product to BF16.
                for pair in cutlass.range_constexpr((cfg.glu_quant_vec+1)//2):
                    v0=2*pair
                    v1=cutlass.min(v0+1,cfg.glu_quant_vec-1)
                    packed=nvvm.inline_ptx_hl(
                        "cvt.rn.satfinite.e4m3x2.f32 {$w0}, {$r0}, {$r1};",
                        write_only_types=[Int16],
                        read_only_args=[values[v1]*inv,values[v0]*inv],
                    )
                    if cutlass.const_expr(cfg.glu_hidden_decode in ("fp32_packed","bf16x2") and cfg.glu_quant_vec>1):
                        if cutlass.const_expr(cfg.glu_hidden_decode=="fp32_packed"):
                            # Preserve FP32 decoding/scaling; pack only the final
                            # BF16 conversion/store. Low half is the first value.
                            low=Float32(Uint8(packed).bitcast(Float8E4M3FN))*Float32(scale)
                            high=Float32(Uint8(packed>>8).bitcast(Float8E4M3FN))*Float32(scale)
                            decoded=Uint32(nvvm.inline_ptx_hl(
                                "cvt.rn.bf16x2.f32 {$w0}, {$r0}, {$r1};",
                                write_only_types=[Int32],read_only_args=[high,low]))
                        else:
                            decoded=nvvm.inline_ptx_hl("cvt.rn.bf16x2.e4m3x2 {$w0}, {$r0};",
                                                      write_only_types=[Int32],read_only_args=[packed])
                            scale_bits=Uint32(BFloat16(Float32(scale)).bitcast(Uint16))
                            decoded=Uint32(nvvm.mul_bf16x2(decoded,Int32(scale_bits|(scale_bits<<16))))
                        out_row=gr
                        if row<cfg.tile_m and gr<self.problem.m and gc+v0<self.problem.n//2:
                            if gc+v1<self.problem.n//2 and (out_row*(self.problem.n//2)+gc+v0)%2==0:
                                ptr=cute.recast_ptr(hidden_out.iterator+out_row*(self.problem.n//2)+gc+v0,dtype=Uint32)
                                ptr[0]=decoded
                            else:
                                hidden_out[out_row,gc+v0]=Uint16(decoded).bitcast(BFloat16)
                                if gc+v1<self.problem.n//2:
                                    hidden_out[out_row,gc+v1]=Uint16(decoded>>16).bitcast(BFloat16)
                    else:
                        for half in cutlass.range_constexpr(min(2,cfg.glu_quant_vec)):
                            v=v0+half
                            if row<cfg.tile_m and gr<self.problem.m and gc+v<self.problem.n//2:
                                rounded=Uint8(packed>>(8*half)).bitcast(Float8E4M3FN)
                                hidden_out[gr,gc+v]=BFloat16(Float32(rounded)*Float32(scale))
            cute.arch.sync_threads()


@lru_cache(maxsize=128)
def compile_glu_derivative(problem,config,activation,emit_hidden=False):
    if config.glu_store != "direct":
        raise ValueError("recompute derivative currently requires direct output stores")
    kernel = GLURecomputeDerivative(problem,config,activation,"bf16")
    kernel.emit_hidden=emit_hidden
    kernel.emit_gradient_quant=False
    kernel.emit_hidden_quant=False
    kernel.separate_hidden=False
    kernel.glu_scratch_elems = config.tile_m*config.tile_n//2 if emit_hidden else 0
    if kernel.warp_quant:
        kernel.glu_scratch_elems=config.tile_m*config.tile_n*config.atom_layout_n//64 if emit_hidden else 0
    if kernel.reuse_glu_scratch:
        kernel.glu_alias_bytes=kernel.glu_scratch_elems*4
        kernel.glu_scratch_elems = 0
    def fake(rows,cols):
        return cute.runtime.make_fake_tensor(BFloat16,(rows,cols),stride=(cols,1),assumed_align=16)
    x,w,out,dh = fake(problem.m,problem.k),fake(problem.n,problem.k),fake(problem.m,problem.n),fake(problem.m,problem.n//2)
    if emit_hidden:
        out=fake(((2*problem.m+7)//8)*8+problem.m,problem.n//2)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(kernel,x,w,out,dh,dh,dh,dh,stream,
                        options="--enable-tvm-ffi --opt-level 3 "
                        f"--ptxas-options '-O3 --maxrregcount={config.maxrregcount}'")


class PrequantGLURecomputeDerivative(MXFP8PrequantGLUKernel):
    _backward_quant_pair = GLURecomputeDerivative._backward_quant_pair
    _glu_epilogue = GLURecomputeDerivative._glu_epilogue
    _hidden_quant_epilogue = GLURecomputeDerivative._hidden_quant_epilogue


@lru_cache(maxsize=128)
def compile_prequant_glu_derivative(problem,config,activation,emit_hidden=False,separate_hidden=False,
                                   emit_gradient_quant=False,emit_hidden_quant=False,
                                   gradient_quant_policy=('fp32','fp32'),hidden_quant_policy=('fp32','fp32')):
    for policy in (gradient_quant_policy,hidden_quant_policy):
        if len(policy)!=2 or policy[0] not in ('fp32','bf16x2') or policy[1] not in ('fp32','bf16_bits'):
            raise ValueError('unsupported backward quantization policy')
    if emit_hidden_quant and (not emit_hidden or not separate_hidden or config.tile_m%32 or config.tile_n%64
                              or config.glu_hidden_decode not in ('fp32','fp32_packed')):
        raise ValueError('hidden quantization requires separate joint recomputation, 32-aligned row/branch tiles and FP32 hidden decode')
    if emit_gradient_quant and (not emit_hidden or config.glu_grad_store!='exchange'
                                or problem.n%64 or config.tile_n%64):
        raise ValueError('gradient quantization requires joint exchange recompute and complete 32-value branch blocks')
    if separate_hidden and not emit_hidden:
        raise ValueError('separate hidden storage requires joint recomputation')
    from ..configs.glu import prequant_recompute_rejection
    rejection = prequant_recompute_rejection(config)
    if rejection:
        raise ValueError(rejection)
    if config.glu_epilogue != "registers" or config.glu_store != "direct":
        raise ValueError("prequant derivative requires the direct paired-register epilogue")
    if config.scale_layout != "row_major":
        raise ValueError("prequant derivative currently requires row-major scales")
    kernel=PrequantGLURecomputeDerivative(problem,config,activation,"bf16")
    kernel.single_stage_recompute_handoff=True
    kernel.emit_hidden=emit_hidden
    kernel.emit_gradient_quant=emit_gradient_quant
    kernel.emit_hidden_quant=emit_hidden_quant
    kernel.gradient_quant_policy=gradient_quant_policy
    kernel.hidden_quant_policy=hidden_quant_policy
    kernel.separate_hidden=separate_hidden
    kernel.glu_scratch_elems=config.tile_m*config.tile_n//2 if emit_hidden else 0
    def fake(dtype,rows,cols):
        return cute.runtime.make_fake_tensor(dtype,(rows,cols),stride=(cols,1),assumed_align=16)
    x=fake(Float8E4M3FN,problem.m,problem.k)
    w=fake(Float8E4M3FN,problem.n,problem.k)
    sx=fake(Float8E8M0FNU,problem.m,problem.k//32)
    sw=fake(Float8E8M0FNU,problem.n,problem.k//32)
    h=problem.n//2
    out=fake(BFloat16,((2*problem.m+7)//8)*8+problem.m,h) if emit_hidden else fake(BFloat16,problem.m,problem.n)
    dh=fake(BFloat16,problem.m,h)
    if separate_hidden:
        out=fake(BFloat16,2*problem.m,h)
        dh=(dh,fake(BFloat16,problem.m,h))
    if emit_gradient_quant:
        dh=(dh,fake(Float8E4M3FN,2*problem.m,h),fake(Float8E8M0FNU,2*problem.m,h//32))
    if emit_hidden_quant:
        storage_m=(problem.m+31)//32*32
        dh=(dh,fake(Float8E4M3FN,h,storage_m),fake(Float8E8M0FNU,h,storage_m//32))
    stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(kernel,x,w,sx,sw,out,dh,stream,
                        options="--enable-tvm-ffi --opt-level 3 "
                        f"--ptxas-options '-O3 --maxrregcount={config.maxrregcount}'")


class DequantizeHidden:
    @cute.jit
    def __call__(self,data,scales,out,stream):
        self.kernel(data,scales,out).launch(grid=(cute.ceil_div(cute.size(out),256),1,1),block=(256,1,1),stream=stream)

    @cute.kernel
    def kernel(self,data,scales,out):
        idx = cute.arch.block_idx()[0]*256+cute.arch.thread_idx()[0]
        row,col = idx//cute.size(out,mode=[1]),idx%cute.size(out,mode=[1])
        if idx < cute.size(out):
            out[row,col] = BFloat16(Float32(data[row,col])*Float32(scales[row,col//32]))


@lru_cache(maxsize=128)
def compile_dequant_hidden(m,h):
    width=(h+31)//32*32
    def fake(dtype,cols):
        return cute.runtime.make_fake_tensor(dtype,(m,cols),stride=(cols,1),assumed_align=16)
    stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(DequantizeHidden(),fake(Float8E4M3FN,width),fake(Float8E8M0FNU,width//32),
                        fake(BFloat16,h),stream,options="--enable-tvm-ffi --opt-level 3")
