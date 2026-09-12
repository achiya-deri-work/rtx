"""Quantize-once operand reuse with the same fused on-chip GLU epilogue."""
from functools import lru_cache

import cutlass.cute as cute
from cutlass import BFloat16, Float8E4M3FN, Float8E8M0FNU, Uint8

from .mxfp8_gemm import MXFP8GemmKernel
from .mxfp8_glu import MXFP8GLUKernel
from .mxfp8_fwd import MXFP8LinearFwdKernel


class MXFP8PrequantGLUKernel(MXFP8GemmKernel):
    _glu_epilogue = MXFP8GLUKernel._glu_epilogue
    _scratch = MXFP8GLUKernel._scratch
    _activate = MXFP8GLUKernel._activate
    _sigmoid = MXFP8GLUKernel._sigmoid
    _warp_amax = MXFP8LinearFwdKernel._warp_amax
    _scale_from_amax = MXFP8LinearFwdKernel._scale_from_amax

    def __init__(self, problem, config, activation, output_format):
        if config.epilogue != "direct":
            raise ValueError("prequant GLU requires direct output")
        super().__init__(problem,config)
        self.glu = True
        self.reuse_glu_scratch = False
        self.warp_quant = False
        self.glu_tma_store = config.glu_store == "tma"
        self.glu_store_stages = config.glu_store_stages
        self.glu_interleaved_operands = True
        self.activation = activation
        self.output_format = output_format
        self.register_epilogue = config.glu_epilogue == "registers"
        self.glu_scratch_elems = config.tile_m * config.tile_n
        if self.register_epilogue:
            self.glu_scratch_elems = 0 if output_format == "bf16" else self.glu_scratch_elems//2


@lru_cache(maxsize=128)
def compile_prequant_glu(problem, config, activation="silu", output_format="bf16"):
    kernel = MXFP8PrequantGLUKernel(problem,config,activation,output_format)
    def fake(dtype,rows,cols):
        return cute.runtime.make_fake_tensor(dtype,(rows,cols),stride=(cols,1),assumed_align=16)
    qx,qw = fake(Float8E4M3FN,problem.m,problem.k),fake(Float8E4M3FN,problem.n,problem.k)
    def scales(rows,layout):
        if layout == "row_major":
            return fake(Float8E8M0FNU,rows,problem.k//32)
        tile_rows = 64 if layout == "mma64" else 128
        return cute.runtime.make_fake_tensor(Float8E8M0FNU,(rows//tile_rows,problem.k//128,512),
                                             stride=(problem.k//128*512,512,1),assumed_align=16)
    sx = scales(problem.m,"mma64" if config.scale_layout=="mma64x128" else config.scale_layout)
    sw = scales(problem.n,"mma128" if config.scale_layout=="mma64x128" else config.scale_layout)
    h = problem.n//2
    width = h if output_format == "bf16" else (h+31)//32*32
    out = fake(BFloat16 if output_format=="bf16" else Float8E4M3FN,problem.m,width)
    aux = out if output_format=="bf16" else fake(Uint8,problem.m,width//32)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(kernel,qx,qw,sx,sw,out,aux,stream,
                        options="--enable-tvm-ffi --opt-level 3 "
                        f"--ptxas-options '-O3 --maxrregcount={config.maxrregcount}'")


def build_prequant_glu_runner(x,weight,out,scales,config,activation,output_format,*,bind_inputs=True,
                             derivative=False,emit_hidden=False):
    import torch
    from .mxfp8 import MXFP8Problem
    from .mxfp8_quant import compile_mxfp8_quant, compile_mxfp8_dual_quant
    m,k = x.shape
    n = weight.shape[0]
    storage_k = (k+31)//32*32
    rejection = config.rejection(MXFP8Problem(m,n,k),output_format)
    if rejection:
        raise ValueError(rejection)
    qx = torch.empty(m,storage_k,device=x.device,dtype=torch.float8_e4m3fn)
    qw = torch.empty(n,storage_k,device=x.device,dtype=torch.float8_e4m3fn)
    def allocate_scales(rows,layout):
        shape = (rows,storage_k//32) if layout=="row_major" else (rows//(64 if layout=="mma64" else 128),k//128,512)
        return torch.empty(shape,device=x.device,dtype=torch.float8_e8m0fnu)
    wconfig = config.resolved_weight_quant()
    sx = allocate_scales(m,config.quant.scale_layout)
    sw = allocate_scales(n,wconfig.scale_layout)
    if config.quant_launches == "dual":
        quant_dual = compile_mxfp8_dual_quant(m,n,k,config.quant,wconfig.scale_layout,interleave_weight=True)
    else:
        quant_x = compile_mxfp8_quant(m,k,config.quant)
        quant_w = compile_mxfp8_quant(n,k,wconfig,interleave_rows=True)
    if derivative:
        from .mxfp8_mlp import compile_prequant_glu_derivative
        gemm = compile_prequant_glu_derivative(MXFP8Problem(m,n,storage_k),config.gemm,activation,emit_hidden)
    else:
        gemm = compile_prequant_glu(MXFP8Problem(m,n,storage_k),config.gemm,activation,output_format)
    def run(runtime_x,runtime_weight,runtime_out,runtime_scales):
        if config.quant_launches == "dual":
            quant_dual(runtime_x,runtime_weight,qx,qw,sx,sw)
        else:
            quant_x(runtime_x,qx,sx)
            quant_w(runtime_weight,qw,sw)
        aux = runtime_out if output_format=="bf16" and not derivative else runtime_scales
        gemm(qx,qw,sx,sw,runtime_out,aux)
    return (lambda:run(x,weight,out,scales)) if bind_inputs else run
