"""NVFP4 paired GEMM with FP32 SiTU and joint recomputation epilogues.

Independently implemented SiTU formula, following Moonshot's documented
SituAndMul definition. No upstream implementation code is vendored.
"""
from functools import lru_cache

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float4E2M1FN, Float8E4M3FN, Float8E8M0FNU, Float32, Int16, Int32, Uint8
from cutlass.experimental.primitives import nvvm_wrapper as nvvm

from .nvfp4_gemm import NVFP4GemmKernel, _fake_nvfp4_scales
from ._layout_contracts import preserve_tma_singletons
from .nvfp4_quant import NVFP4JITRegionDualQuantKernel


class NVFP4SiTUKernel(NVFP4GemmKernel):
    def __init__(self, problem, config, x_rows, w_rows, beta, linear_beta, derivative=False,
                 epilogue="registers",store_vec=2,swizzle="none",hidden_dtype="bf16",hidden_quant_vec=4):
        super().__init__(problem, config)
        self.glu = True
        self.apply_output_scale = False
        self.x_rows, self.w_rows = x_rows, w_rows
        self.x_regions = (problem.m+x_rows-1)//x_rows
        self.beta, self.linear_beta = beta, linear_beta
        self.derivative = derivative
        self.exchange = epilogue=="exchange"
        self.store_vec,self.scratch_swizzle=store_vec,swizzle
        self.outputs=3 if derivative else 1
        self.packed_hidden=hidden_dtype=='mxfp8'
        self.hidden_quant_vec=hidden_quant_vec
        # Values are rounded only after FP32 SiTU/differentiation, exactly at
        # the public BF16 boundary. Exchange BF16, not oversized FP32 tiles.
        self.glu_scratch_elems=config.tile_m*config.tile_n//4*self.outputs if self.exchange else 0
        if self.packed_hidden:
            self.glu_scratch_elems=config.tile_m*config.tile_n//2

    @cute.jit
    def _tanh(self, value):
        return nvvm.inline_ptx_hl("tanh.approx.f32 {$w0}, {$r0};",
            write_only_types=[Float32], read_only_args=[value])

    @cute.jit
    def _values(self, gate, up):
        t = self._tanh(gate/Float32(self.beta))
        sigmoid = Float32(1)/(Float32(1)+cute.exp(-gate))
        v = self._tanh(up/Float32(self.linear_beta))
        a = Float32(self.beta)*t*sigmoid
        b = Float32(self.linear_beta)*v
        dg = ((Float32(1)-t*t)*sigmoid
              + Float32(self.beta)*t*sigmoid*(Float32(1)-sigmoid))*b
        du = a*(Float32(1)-v*v)
        return a*b, dg, du

    @cute.jit
    def _bf16_glu_epilogue(self, storage, accumulators, coordinates, out, auxiliary,
                      block_m, block_n, tidx, tma_out, tma_out_tensor, out_layout, work_slot):
        outer = auxiliary
        if cutlass.const_expr(self.derivative):
            outer, dh, hidden = auxiliary
        if cutlass.const_expr(self.exchange):
            layout=cute.make_layout((self.config.tile_m,self.config.tile_n//2*self.outputs),
                                   stride=(self.config.tile_n//2*self.outputs,1))
            if cutlass.const_expr(self.scratch_swizzle=='xor_16b'):
                layout=cute.make_composed_layout(cute.make_swizzle(3,3,3),0,layout)
            raw=storage.glu_scratch.get_tensor(cute.make_layout((self.glu_scratch_elems,)))
            scratch=cute.make_tensor(cute.recast_ptr(raw.iterator,dtype=BFloat16),layout)
        if tidx < self.config.num_mma_warps*32:
            for i in cutlass.range_constexpr(cute.size(accumulators)//2):
                row, col = coordinates[2*i][0], coordinates[2*i][1]//2
                if row < self.problem.m and col < self.problem.n//2:
                    xs = Float32(outer[row//self.x_rows])
                    gs = Float32(outer[self.x_regions+col//self.w_rows])
                    us = Float32(outer[self.x_regions+(col+self.problem.n//2)//self.w_rows])
                    gf = nvvm.inline_ptx_hl("mul.rn.f32 {$w0}, {$r0}, {$r1};",
                        write_only_types=[Float32], read_only_args=[xs,gs])
                    uf = nvvm.inline_ptx_hl("mul.rn.f32 {$w0}, {$r0}, {$r1};",
                        write_only_types=[Float32], read_only_args=[xs,us])
                    gate = accumulators[2*i]*gf
                    up = accumulators[2*i+1]*uf
                    value, dg, du = self._values(gate, up)
                    if cutlass.const_expr(self.derivative):
                        gradient = Float32(dh[row,col])
                        if cutlass.const_expr(self.exchange):
                            lr,lc=row-block_m*self.config.tile_m,col-block_n*self.config.tile_n//2
                            scratch[lr,3*lc]=BFloat16(value)
                            scratch[lr,3*lc+1]=BFloat16(gradient*dg)
                            scratch[lr,3*lc+2]=BFloat16(gradient*du)
                        else:
                            out[row,col] = BFloat16(gradient*dg)
                            out[row,col+self.problem.n//2] = BFloat16(gradient*du)
                            hidden[row,col] = BFloat16(value)
                    else:
                        if cutlass.const_expr(self.exchange):
                            scratch[row-block_m*self.config.tile_m,col-block_n*self.config.tile_n//2]=BFloat16(value)
                        else:
                            out[row,col] = BFloat16(value)
        if cutlass.const_expr(self.exchange):
            cute.arch.sync_threads()
            width=self.config.tile_n//2
            for packet in cutlass.range(cute.ceil_div(self.config.tile_m*width,self.config.num_threads*self.store_vec),unroll=1):
                index=(tidx+packet*self.config.num_threads)*self.store_vec
                lr,lc=index//width,index%width
                row,col=block_m*self.config.tile_m+lr,block_n*width+lc
                for v in cutlass.range_constexpr(self.store_vec):
                    if lr<self.config.tile_m and row<self.problem.m and col+v<self.problem.n//2:
                        if cutlass.const_expr(self.derivative):
                            hidden[row,col+v]=BFloat16(scratch[lr,(lc+v)*3])
                            out[row,col+v]=BFloat16(scratch[lr,(lc+v)*3+1])
                            out[row,col+v+self.problem.n//2]=BFloat16(scratch[lr,(lc+v)*3+2])
                        else:
                            out[row,col+v]=BFloat16(scratch[lr,lc+v])
            cute.arch.sync_threads()

    @cute.jit
    def _glu_epilogue(self,storage,accumulators,coordinates,out,auxiliary,
                      block_m,block_n,tidx,tma_out,tma_out_tensor,out_layout,work_slot):
        if cutlass.const_expr(self.packed_hidden):
            self._packed_glu_epilogue(storage,accumulators,coordinates,out,auxiliary,block_m,block_n,tidx)
        else:
            self._bf16_glu_epilogue(storage,accumulators,coordinates,out,auxiliary,
                block_m,block_n,tidx,tma_out,tma_out_tensor,out_layout,work_slot)

    @cute.jit
    def _packed_glu_epilogue(self,storage,accumulators,coordinates,out,auxiliary,block_m,block_n,tidx):
        if cutlass.const_expr(self.derivative):
            outer,dh,hidden,scales=auxiliary
        else:
            outer,scales=auxiliary
            hidden=out
        cfg=self.config
        width=cfg.tile_n//2
        layout=cute.make_layout((cfg.tile_m,width),stride=(width,1))
        if cutlass.const_expr(self.scratch_swizzle=='xor_16b'):
            layout=cute.make_composed_layout(cute.make_swizzle(3,2,3),0,layout)
        scratch=storage.glu_scratch.get_tensor(layout)
        if tidx<cfg.num_mma_warps*32:
            for i in cutlass.range_constexpr(cute.size(accumulators)//2):
                row,col=coordinates[2*i][0],coordinates[2*i][1]//2
                value=Float32(0)
                if row<self.problem.m and col<self.problem.n//2:
                    xs=Float32(outer[row//self.x_rows])
                    gs=Float32(outer[self.x_regions+col//self.w_rows])
                    us=Float32(outer[self.x_regions+(col+self.problem.n//2)//self.w_rows])
                    gf=nvvm.inline_ptx_hl('mul.rn.f32 {$w0}, {$r0}, {$r1};',
                        write_only_types=[Float32],read_only_args=[xs,gs])
                    uf=nvvm.inline_ptx_hl('mul.rn.f32 {$w0}, {$r0}, {$r1};',
                        write_only_types=[Float32],read_only_args=[xs,us])
                    value,dg,du=self._values(accumulators[2*i]*gf,accumulators[2*i+1]*uf)
                    if cutlass.const_expr(self.derivative):
                        grad=Float32(dh[row,col])
                        out[row,col]=BFloat16(grad*dg)
                        out[row,col+self.problem.n//2]=BFloat16(grad*du)
                scratch[row-block_m*cfg.tile_m,col-block_n*width]=value
        cute.arch.sync_threads()
        vec=self.hidden_quant_vec
        lanes=32//vec
        for packet in cutlass.range(cute.ceil_div(cfg.tile_m*width,cfg.num_threads*vec),unroll=1):
            index=(tidx+packet*cfg.num_threads)*vec
            lr,lc=index//width,index%width
            row,col=block_m*cfg.tile_m+lr,block_n*width+lc
            values=cute.make_rmem_tensor((vec,),Float32)
            maximum=Float32(0)
            for v in cutlass.range_constexpr(vec):
                value=Float32(0)
                if lr<cfg.tile_m:
                    value=scratch[lr,lc+v]
                values[v]=value
                maximum=cute.arch.fmax(maximum,cute.abs(value),nan=True)
            lane=cute.arch.lane_idx()
            group=lane&Int32(~(lanes-1))
            mask=Int32((1<<lanes)-1)<<group
            maximum=nvvm.redux_sync(maximum.bitcast(Int32),nvvm.ReductionKind.UMAX,mask).bitcast(Float32)
            bits=maximum.bitcast(Int32)
            exponent=((bits>>23)&255)-135
            if (bits&Int32(0x7fffff))>Int32(0x600000):
                exponent+=1
            exponent=cutlass.min(cutlass.max(exponent,-127),127)
            biased=exponent+127
            normal=cutlass.max(biased,1)
            reciprocal=((Int32(254)-normal)<<23).bitcast(Float32)
            for pair in cutlass.range_constexpr(vec//2):
                packed=nvvm.inline_ptx_hl('cvt.rn.satfinite.e4m3x2.f32 {$w0}, {$r0}, {$r1};',
                    write_only_types=[Int16],read_only_args=[values[2*pair+1]*reciprocal,values[2*pair]*reciprocal])
                if lr<cfg.tile_m and row<self.problem.m and col+2*pair<cute.size(hidden,mode=[1]):
                    hidden[row,col+2*pair]=Uint8(packed&255).bitcast(Float8E4M3FN)
                    hidden[row,col+2*pair+1]=Uint8((packed>>8)&255).bitcast(Float8E4M3FN)
            if lr<cfg.tile_m and row<self.problem.m and col<cute.size(hidden,mode=[1]) and col%32==0:
                scales[row,col//32]=Uint8(biased)
        # The next persistent tile may overwrite this FP32 exchange only
        # after every warp has finished quantizing the current tile.
        cute.arch.sync_threads()


class MXFP8InputNVFP4QuantKernel(NVFP4JITRegionDualQuantKernel):
    """Same quantization, with warp/CTA/register-cached packed-source observers."""
    _bf16_warp_amax=NVFP4JITRegionDualQuantKernel._warp_region_amax
    _bf16_cta_amax=NVFP4JITRegionDualQuantKernel._region_amax

    @cute.jit
    def _mixed_local_amax(self,src,row_begin,row_end,start,threads:cutlass.Constexpr):
        maximum=Float32(0)
        for index in cutlass.range(start,(row_end-row_begin)*self.k,threads,unroll=self.amax_unroll):
            row,col=row_begin+index//self.k,index%self.k
            value=BFloat16(Float32(src[0][row,col])*Float32(src[1][row,col//32]))
            maximum=cute.arch.fmax(maximum,cute.abs(Float32(value)),nan=True)
        return maximum

    @cute.jit
    def _warp_region_amax(self,src,row_begin,row_end):
        if cutlass.const_expr(isinstance(src,tuple)):
            maximum=self._mixed_local_amax(src,row_begin,row_end,cute.arch.lane_idx(),32)
            maximum=nvvm.redux_sync(maximum.bitcast(Int32),nvvm.ReductionKind.UMAX,Int32(-1)).bitcast(Float32)
            return self._tensor_scale_from_amax(maximum,self.tensor_scale_mode)
        else:
            return self._bf16_warp_amax(src,row_begin,row_end)

    @cute.jit
    def _region_amax(self,src,row_begin,row_end,warp_amax,outer_scale):
        if cutlass.const_expr(isinstance(src,tuple)):
            tidx=cute.arch.thread_idx()[0]
            lane=cute.arch.lane_idx()
            warp=tidx//32
            maximum=self._mixed_local_amax(src,row_begin,row_end,tidx,self.config.num_warps*32)
            maximum=nvvm.redux_sync(maximum.bitcast(Int32),nvvm.ReductionKind.UMAX,Int32(-1)).bitcast(Float32)
            if lane==0:
                warp_amax[warp]=maximum
            cute.arch.sync_threads()
            if warp==0:
                maximum=Float32(0)
                if lane<self.config.num_warps:
                    maximum=Float32(warp_amax[lane])
                maximum=nvvm.redux_sync(maximum.bitcast(Int32),nvvm.ReductionKind.UMAX,Int32(-1)).bitcast(Float32)
                if lane==0:
                    outer_scale[0]=self._tensor_scale_from_amax(maximum,self.tensor_scale_mode)
            cute.arch.sync_threads()
        else:
            self._bf16_cta_amax(src,row_begin,row_end,warp_amax,outer_scale)


@lru_cache(maxsize=128)
def compile_nvfp4_situ(problem, config, x_rows, w_rows, beta, linear_beta, derivative=False,
                       epilogue="registers",store_vec=2,swizzle="none",hidden_dtype="bf16",hidden_quant_vec=4):
    kernel = NVFP4SiTUKernel(problem,config,x_rows,w_rows,beta,linear_beta,derivative,epilogue,store_vec,swizzle,hidden_dtype,hidden_quant_vec)
    def fake(dtype, rows, cols, align=16):
        return cute.runtime.make_fake_tensor(dtype,(rows,cols),stride=(cols,1),assumed_align=align)
    def operand(rows):
        return cute.runtime.make_fake_tensor(Float4E2M1FN,
            (cute.sym_int32(),problem.k) if rows==problem.n and problem.n==2
            else preserve_tma_singletons((rows,problem.k)),stride=(problem.k,1),assumed_align=16)
    h = problem.n//2
    width=(h+31)//32*32
    out = fake(BFloat16 if derivative or hidden_dtype=='bf16' else Float8E4M3FN,
               problem.m,problem.n if derivative else h if hidden_dtype=='bf16' else width)
    count = (problem.m+x_rows-1)//x_rows+(problem.n+w_rows-1)//w_rows
    outer = cute.runtime.make_fake_tensor(Float32,(count,),stride=(1,),assumed_align=4)
    auxiliary = (outer,fake(BFloat16,problem.m,h),fake(BFloat16,problem.m,h)) if derivative else outer
    if hidden_dtype=='mxfp8':
        scales=fake(Uint8,problem.m,width//32)
        auxiliary=(outer,fake(BFloat16,problem.m,h),fake(Float8E4M3FN,problem.m,width),scales) if derivative else (outer,scales)
    return cute.compile(kernel,operand(problem.m),operand(problem.n),
        _fake_nvfp4_scales(problem.m,problem.k,config),
        _fake_nvfp4_scales(problem.n,problem.k,config),out,auxiliary,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi --opt-level 3 "
                f"--ptxas-options '-O3 --maxrregcount={config.maxrregcount}'")


@lru_cache(maxsize=128)
def compile_mxfp8_to_nvfp4_prepare(m,n,k,config):
    """Decode hidden into the existing regional register cache, quantize once.

    This shares the independently qualified BF16 regional quantizer's scale
    fitting and FP4 rounding. Only its source load differs; no BF16 hidden
    tensor is allocated or written by this producer.
    """
    from .nvfp4_quant import _fake_bf16,_fake_packed,_fake_scales
    from ..configs.nvfp4 import nvfp4_storage_k
    c=config
    kernel=MXFP8InputNVFP4QuantKernel(m,n,k,c.quant,c.x_scale_region_rows,c.weight_scale_region_rows,
        c.tensor_scale_mode,c.region_amax_load_bits,c.region_amax_unroll,c.region_waves,
        c.region_order,c.region_ownership)
    width=nvfp4_storage_k(k)
    data=cute.runtime.make_fake_tensor(Float8E4M3FN,(m,width),stride=(width,1),assumed_align=16)
    sx8=cute.runtime.make_fake_tensor(Float8E8M0FNU,(m,width//32),stride=(width//32,1),assumed_align=16)
    count=(m+c.x_scale_region_rows-1)//c.x_scale_region_rows+(n+c.weight_scale_region_rows-1)//c.weight_scale_region_rows
    outer=cute.runtime.make_fake_tensor(Float32,(count,),stride=(1,),assumed_align=4)
    return cute.compile(kernel,(data,sx8),_fake_bf16(n,k),_fake_packed(m,k),_fake_packed(n,k),
        _fake_scales(m,k,c.quant.scale_layout),_fake_scales(n,k,c.quant.scale_layout),outer,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi --opt-level 3 "
                f"--ptxas-options '-O3 --maxrregcount={c.quant.maxrregcount}'")
