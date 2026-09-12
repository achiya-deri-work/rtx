"""Experimental quantize-once/GLU cooperative grid.

The caller initializes a private 32-bit counter using a CUDA memset graph
node. This kernel uses exactly one release/acquire rendezvous, then persistent
packed GEMM tiles. Never launch it as an ordinary grid or share the counter
with another invocation. There are two compute kernels plus barrier setup,
not just two device operations.
"""
from functools import lru_cache

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16,Float8E4M3FN,Float8E8M0FNU,Int32,Uint8
from cutlass.experimental.primitives import nvvm_wrapper as nvvm

from .mxfp8_glu_prequant import MXFP8PrequantGLUKernel
from .mxfp8_quant import MXFP8QuantKernel


class CooperativeGLU(MXFP8PrequantGLUKernel):
    def __init__(self,problem,config,activation,quant_config,down_rows,down_quant,grid_ctas):
        from ..configs.glu import cooperative_glu_rejection
        reason=cooperative_glu_rejection(problem,config)
        if reason:
            raise ValueError(reason)
        super().__init__(problem,config,activation,"mxfp8")
        if any(q.scale_layout!="row_major" or q.scale_rounding!="infinity" for q in (quant_config,down_quant)):
            raise ValueError("cooperative quantizers require row-major infinity-rounded scales")
        if grid_ctas<1:
            raise ValueError("cooperative grid must contain at least one CTA")
        if not config.persistent_waves or config.scale_layout != "row_major":
            raise ValueError("cooperative GLU requires persistent row-major GEMM")
        self.cooperative_quant=True
        self.grid_ctas=min(grid_ctas,self.total_tiles)
        self.work_tiles_per_cta=cute.ceil_div(self.total_tiles,self.grid_ctas)
        self.x_quant=MXFP8QuantKernel(problem.m,problem.k,quant_config)
        self.w_quant=MXFP8QuantKernel(problem.n,problem.k,quant_config)
        self.down_quant=MXFP8QuantKernel(down_rows,problem.n//2,down_quant)

    @cute.jit
    def _quant_operand(self,source,out,scales,operand:cutlass.Constexpr):
        tidx=cute.arch.thread_idx()[0]
        cta=cute.arch.block_idx()[0]
        if cutlass.const_expr(operand == 0):
            cfg=self.x_quant.config
            rows,storage_k,task_groups=self.x_quant.rows,self.x_quant.storage_k,self.x_quant.task_groups
        elif cutlass.const_expr(operand == 1):
            cfg=self.w_quant.config
            rows,storage_k,task_groups=self.w_quant.rows,self.w_quant.storage_k,self.w_quant.task_groups
        else:
            cfg=self.down_quant.config
            rows,storage_k,task_groups=self.down_quant.rows,self.down_quant.storage_k,self.down_quant.task_groups
        warps=min(cfg.num_warps,self.config.num_threads//32)
        lane=cute.arch.lane_idx()
        width=32//cfg.quant_vec
        lane_scale=lane%width
        blocks_per_row=storage_k//32
        if tidx < warps*32:
            for group in cutlass.range(cta*warps+tidx//32,task_groups,self.grid_ctas*warps,unroll=1):
                block=group*cfg.quant_vec+lane//width
                if block < rows*blocks_per_row:
                    row,col=block//blocks_per_row,block%blocks_per_row
                    if cutlass.const_expr(operand == 0):
                        self.x_quant._quantize_task(source,out,scales,row,col,lane_scale,col*32+lane_scale*cfg.quant_vec,0)
                    elif cutlass.const_expr(operand == 1):
                        self.w_quant._quantize_task(source,out,scales,row,col,lane_scale,col*32+lane_scale*cfg.quant_vec,0)
                    else:
                        self.down_quant._quantize_task(source,out,scales,row,col,lane_scale,col*32+lane_scale*cfg.quant_vec,0)

    @cute.jit
    def _cooperative_quantize(self,qx,qw,sx,sw,aux):
        scales,x,w,down,qd,sd,counter=aux
        h=self.problem.n//2
        w_interleaved=cute.make_tensor(w.iterator,cute.make_layout(((2,h),self.problem.k),
                                                stride=((h*self.problem.k,self.problem.k),1)))
        self._quant_operand(x,qx,sx,0)
        self._quant_operand(w_interleaved,qw,sw,1)
        self._quant_operand(down,qd,sd,2)
        # Publish generic stores to the asynchronous TMA proxy before any
        # consumer can observe completion of the grid rendezvous.
        nvvm.inline_ptx_hl("fence.proxy.async.global;",write_only_types=[],read_only_args=[])
        cute.arch.sync_threads()
        if cute.arch.thread_idx()[0] == 0:
            nvvm.inline_ptx_hl(
                "{ .reg .u32 count, ignored; .reg .pred done; "
                "atom.add.release.gpu.u32 ignored, [{$r1}], 1; "
                "WAIT: ld.acquire.gpu.u32 count, [{$r1}]; "
                "setp.eq.u32 done, count, {$r0}; @!done bra WAIT; }",
                write_only_types=[],read_only_args=[Int32(self.grid_ctas),counter.iterator])
        cute.arch.sync_threads()


@lru_cache(maxsize=128)
def compile_cooperative_glu(problem,config,activation,quant_config,down_rows,down_quant,grid_ctas):
    if problem.k%32:
        raise ValueError("experimental cooperative GLU currently requires K divisible by 32")
    kernel=CooperativeGLU(problem,config,activation,quant_config,down_rows,down_quant,grid_ctas)
    def fake(dtype,rows,cols):
        return cute.runtime.make_fake_tensor(dtype,(rows,cols),stride=(cols,1),assumed_align=16)
    m,n,k=problem.m,problem.n,problem.k
    h=n//2;hp=(h+31)//32*32
    qx,qw=fake(Float8E4M3FN,m,k),fake(Float8E4M3FN,n,k)
    sx,sw=fake(Float8E8M0FNU,m,k//32),fake(Float8E8M0FNU,n,k//32)
    out,scales=fake(Float8E4M3FN,m,hp),fake(Uint8,m,hp//32)
    aux=(scales,fake(BFloat16,m,k),fake(BFloat16,n,k),fake(BFloat16,down_rows,h),
         fake(Float8E4M3FN,down_rows,hp),fake(Float8E8M0FNU,down_rows,hp//32),fake(Int32,1,1))
    stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(kernel,qx,qw,sx,sw,out,aux,stream,
                        options="--enable-tvm-ffi --opt-level 3 "
                        f"--ptxas-options '-O3 --maxrregcount={config.maxrregcount}'")
