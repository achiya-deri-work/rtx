"""Canonical byte-scale launch ABI for compiler-owned MoE buffers.

Inductor 2.13 rejects E8M0 tensors while decomposing auto-functionalized
mutations. Torch-visible scale storage therefore stays UINT8 everywhere.
These CuTe host adapters reinterpret pointers, not values, before calling the
same numerical kernels. They add no launch, data copy or Torch dtype alias.
"""
from functools import lru_cache

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float8E4M3FN, Float8E8M0FNU, Int64, Uint8


@cute.jit
def _e8(value):
    return cute.make_tensor(cute.recast_ptr(value.iterator,dtype=Float8E8M0FNU),value.layout)


class ByteQuantPortfolio:
    def __init__(self,inner):self.inner=inner

    @cute.jit
    def __call__(self,anchor,sources,quantized,scales,stream):
        typed=[]
        for i in cutlass.range_constexpr(len(scales)):
            typed.append(_e8(scales[i]))
        self.inner(anchor,sources,quantized,tuple(typed),stream)


class BytePackedGemm:
    def __init__(self,inner):self.inner=inner

    @cute.jit
    def __call__(self,qx,qw,sx,sw,out,aux,stream):
        # GLU's existing auxiliary ABI already stores raw scale bytes.
        self.inner(qx,qw,_e8(sx),_e8(sw),out,aux,stream)


class ByteLegacyLinear:
    """Legacy packed-hidden GEMM: scale bytes stay canonical in Torch."""
    def __init__(self,inner):self.inner=inner

    @cute.jit
    def __call__(self,x,weight,out,aux0,aux1,aux2,aux3,stream):
        self.inner(x,weight,out,_e8(aux0),_e8(aux1),_e8(aux2),_e8(aux3),stream)


class ByteJointRecompute:
    def __init__(self,inner):self.inner=inner

    @cute.jit
    def __call__(self,qx,qw,sx,sw,dh,aux,stream):
        # Only scale slots change their logical pointer type. FP32 activation,
        # BF16 derivative and both independently quantized axes are unchanged.
        typed=(aux[0],aux[1],_e8(aux[2]),aux[3],_e8(aux[4]),
               aux[5],_e8(aux[6]),aux[7],_e8(aux[8]))
        self.inner(qx,qw,_e8(sx),_e8(sw),dh,typed,stream)


class ByteHiddenDequant:
    def __init__(self,inner):self.inner=inner

    @cute.jit
    def __call__(self,data,scales,out,stream):
        self.inner(data,_e8(scales),out,stream)


class BytePackedGather:
    def __init__(self,inner):self.inner=inner

    @cute.jit
    def __call__(self,global_q,global_sf,x,ids,inverse,out,out_sf,gathered,stream):
        self.inner(global_q,_e8(global_sf),x,ids,inverse,out,_e8(out_sf),gathered,stream)


def _fake(dtype,rows,cols,alignment=16):
    return cute.runtime.make_fake_tensor(dtype,(rows,cols),stride=(cols,1),assumed_align=alignment)


@lru_cache(maxsize=128)
def compile_expert_quant(experts,specs,config,transpose_mode='staged',grid_mode='operand'):
    from .mxfp8_moe_prepared import compile_expert_quant as original
    return original(experts,specs,config,transpose_mode,grid_mode,_byte_scales=True)


@lru_cache(maxsize=128)
def compile_expert_packed_gemm(experts,problem,config,kind='linear',transport='tma'):
    from .mxfp8_moe_prepared import compile_expert_packed_gemm as original
    return original(experts,problem,config,kind,transport,_byte_scales=True)


@lru_cache(maxsize=128)
def compile_grouped_linear(experts,problem,config,a_orientation='row',b_orientation='row',prequant=False):
    from .mxfp8_moe import compile_grouped_linear as original
    # Fully dynamic BF16 inputs carry no external scale storage. Preserve
    # their original launch ABI exactly; only packed hidden A needs recasting.
    return original(experts,problem,config,a_orientation,b_orientation,prequant,_byte_scales=prequant)


@lru_cache(maxsize=128)
def compile_expert_joint_recompute(experts,problem,config,quant):
    from .mxfp8_moe_prepared import compile_expert_joint_recompute as original
    return original(experts,problem,config,quant,_byte_scales=True)


@lru_cache(maxsize=128)
def compile_dequant_hidden(m,h):
    from .mxfp8_mlp import DequantizeHidden
    width=(h+31)//32*32
    stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(ByteHiddenDequant(DequantizeHidden()),_fake(Float8E4M3FN,m,width),
                        _fake(Uint8,m,width//32),_fake(BFloat16,m,h),stream,
                        options='--enable-tvm-ffi --opt-level 3')


@lru_cache(maxsize=64)
def compile_native_hidden_scales(e,rows,k,threads=128):
    from .mxfp8_moe_prepared import NativeHiddenScales
    fake=cute.runtime.make_fake_tensor(Uint8,(e*rows*k//32,),stride=(1,),assumed_align=16)
    stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(NativeHiddenScales(e,rows,k,threads),fake,fake,stream,
                        options='--enable-tvm-ffi --opt-level 3')


@lru_cache(maxsize=128)
def compile_packed_input_gather(n,d,e,c,k,storage_c,threads=128,scale_layout='row_major',copy_bf16=False):
    from .mxfp8_moe_input import PackedInputGather
    inner=PackedInputGather(n,d,e,c,k,storage_c,threads,scale_layout,copy_bf16)
    qx=_fake(Float8E4M3FN,n,inner.width)
    sx=_fake(Uint8,n,inner.scale_cols)
    x=_fake(BFloat16,n,d,2)
    ids,inverse=_fake(Int64,e,c),_fake(Int64,n,k)
    qout=_fake(Float8E4M3FN,e*storage_c,inner.width)
    sout=_fake(Uint8,e*storage_c,inner.scale_cols)
    gathered=_fake(BFloat16,e*storage_c,d) if copy_bf16 else x
    stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(BytePackedGather(inner),qx,sx,x,ids,inverse,qout,sout,gathered,stream,
                        options='--enable-tvm-ffi --opt-level 3')
