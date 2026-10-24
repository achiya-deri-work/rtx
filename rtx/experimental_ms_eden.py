"""Experimental original-BF16 MS-EDEN gradient operation.

Not exported by ``rtx`` and not used by NVFP4Linear. Random signs and uniforms
are explicit tensor inputs; the caller owns generation/checkpoint restoration.
Only contiguous BF16 operands with dimensions divisible by 128 are supported.
No higher-order derivative or claim of unbiasedness is provided.
"""
import json
import torch

from .autotune.posthoc import PosthocPreparationSchedule
from .configs.nvfp4 import NVFP4ScaleConfig

_FORWARD_POLICIES={mode:NVFP4ScaleConfig(tensor_scale_mode=mode) for mode in ("power2","exact")}


def _contract(dy, operand, signs, uniform_a, uniform_b, phase, schedule_json, packed=False):
    if phase not in ("dx", "dw"):
        raise ValueError("phase must be dx or dw")
    schedule = PosthocPreparationSchedule(**json.loads(schedule_json))
    if schedule.gemm.scale_layout != "mma128":
        raise ValueError("experimental gradient operator requires native mma128 scales")
    if schedule.reader_mode != "direct" and not packed:
        raise ValueError("this operator accepts original BF16 operands only")
    for index,value in enumerate((dy, operand)):
        expected_dtype=torch.uint8 if packed and index==1 else torch.bfloat16
        if value.ndim != 2 or value.dtype != expected_dtype or not value.is_contiguous():
            raise ValueError(f"contiguous rank-two {expected_dtype} operand required")
        shape=(value.shape[0],value.shape[1]*(2 if packed and index==1 else 1))
        for size in shape:
            torch._check(size > 0)
            torch._check(size % 128 == 0)
    for value in (operand, signs, uniform_a, uniform_b):
        if value.device != dy.device:
            raise ValueError("all inputs must share a device")
    if signs.dtype != torch.float32 or signs.shape != (128,) or not signs.is_contiguous():
        raise ValueError("signs must be contiguous FP32 [128], containing +/-1")
    for uniform, count in ((uniform_a,dy.numel()),(uniform_b,operand.numel()*(2 if packed else 1))):
        if uniform.dtype != torch.float32 or uniform.ndim != 1 or not uniform.is_contiguous():
            raise ValueError("uniforms must be contiguous FP32 vectors in [0,1)")
        torch._check(uniform.numel() == count // 16)
    output_cols=operand.shape[1]*(2 if packed else 1)
    if phase == "dx":
        torch._check(dy.shape[1] == operand.shape[0])
        shape = (dy.shape[0], output_cols, dy.shape[1])
    else:
        torch._check(dy.shape[0] == operand.shape[0])
        shape = (dy.shape[1], output_cols, dy.shape[0])
    return schedule, shape


@torch.library.custom_op("rtx_experimental::ms_eden_gradient", mutates_args=(), device_types="cuda")
def _gradient(dy: torch.Tensor, operand: torch.Tensor, signs: torch.Tensor,
              uniform_a: torch.Tensor, uniform_b: torch.Tensor,
              phase: str, schedule_json: str) -> torch.Tensor:
    schedule, (m,n,k) = _contract(dy,operand,signs,uniform_a,uniform_b,phase,schedule_json)
    return _execute_gradient(dy,operand,signs,uniform_a,uniform_b,phase,schedule,(m,n,k))


def _execute_gradient(dy,operand,signs,uniform_a,uniform_b,phase,schedule,shape,packed_scales=None,packed_outer=None):
    m,n,k=shape
    from .fp4 import compile_nvfp4_gemm, _packed_fp4_view, _check_sm12x
    from .configs.nvfp4 import NVFP4Problem, NVFP4GemmConfig
    from .kernels.nvfp4_posthoc import (compile_hadamard_preparation,
        compile_posthoc_scales, compile_posthoc_scale_product, PosthocScaleConfig)
    with torch.cuda.device(dy.device):
        _check_sm12x(dy.device)
        def allocate(source, transpose, uniforms, source_scales=None, source_outer=None):
            r,c = source.shape
            is_packed=source_scales is not None
            if is_packed:
                c*=2
            rows,cols = (c,r) if transpose else (r,c)
            config=PosthocScaleConfig(schedule.producer_threads,schedule.producer_blocks)
            decode_runner=None
            if is_packed and schedule.reader_mode=="direct":
                from .kernels.nvfp4_posthoc import compile_packed_hadamard_preparation
                producer,count=compile_packed_hadamard_preparation(r,c,config,4,transpose,schedule.producer_order,schedule.producer_load)
            else:
                producer,count = compile_hadamard_preparation(r,c,config,4,transpose,schedule.producer_order,schedule.producer_load)
                if is_packed:
                    from .kernels.nvfp4_posthoc import compile_packed_bf16_decode
                    decode_runner=compile_packed_bf16_decode(r,c,
                        PosthocScaleConfig(schedule.decoder_threads,schedule.decoder_blocks))
            reducer=None
            if schedule.amax_reduction == "staged":
                from .kernels.nvfp4_posthoc import compile_partial_maximum
                reducer=compile_partial_maximum(count)
            finalizer = compile_posthoc_scales(rows,cols//16,
                PosthocScaleConfig(schedule.finalizer_threads,schedule.finalizer_blocks,"mma128"),1 if reducer else count)
            packed = torch.empty(rows*cols//2,device=dy.device,dtype=torch.uint8)
            temporary = torch.empty(rows*cols//16,device=dy.device,dtype=torch.bfloat16)
            partials = torch.empty(count,device=dy.device,dtype=torch.float32)
            maximum = torch.empty(1,device=dy.device,dtype=torch.float32) if reducer else partials
            scales = torch.empty(rows*cols//16,device=dy.device,dtype=torch.float8_e4m3fn)
            outer = torch.empty(1,device=dy.device,dtype=torch.float32)
            src = source.view(-1)
            if is_packed:
                sf=source_scales.view(-1)
                outer_source=source_outer.view(1)
                decoded=torch.empty(r*c,device=dy.device,dtype=torch.bfloat16) if decode_runner else None
            packed_view = _packed_fp4_view(packed.view(rows,cols//2))
            scale_view = scales.view(rows//128,cols//128,1024)
            def run():
                if is_packed:
                    if decode_runner:
                        decode_runner(src,sf,outer_source,decoded)
                        producer(decoded,signs,packed,temporary,partials)
                    else:
                        producer(src,sf,outer_source,signs,packed,temporary,partials)
                else:
                    producer(src,signs,packed,temporary,partials)
                if reducer:
                    reducer(partials,maximum)
                finalizer(temporary,maximum,uniforms,scales,outer)
            return run,packed_view,scale_view,outer
        a, b = allocate(dy,phase=="dw",uniform_a), allocate(operand,True,uniform_b,packed_scales,packed_outer)
        alpha = torch.empty(1,device=dy.device,dtype=torch.float32)
        out = torch.empty(m,n,device=dy.device,dtype=torch.bfloat16)
        product = compile_posthoc_scale_product()
        gemm = compile_nvfp4_gemm(NVFP4Problem(m,n,k),schedule.gemm)
        # All metadata, allocations and compilation precede this launch chain.
        a[0](); b[0]()
        product(a[3],b[3],alpha)
        gemm(a[1],b[1],a[2],b[2],out,alpha)
        return out


@_gradient.register_fake
def _gradient_fake(dy,operand,signs,uniform_a,uniform_b,phase,schedule_json):
    _,(m,n,_) = _contract(dy,operand,signs,uniform_a,uniform_b,phase,schedule_json)
    return dy.new_empty((m,n))


def _no_higher_order(ctx, *grads):
    raise RuntimeError("experimental MS-EDEN gradients do not support higher-order autograd")


_gradient.register_autograd(_no_higher_order)


def _packed_contract(dy,data,scales,outer,signs,ua,ub,phase,schedule_json):
    schedule,shape=_contract(dy,data,signs,ua,ub,phase,schedule_json,packed=True)
    if (scales.dtype != torch.uint8 or not scales.is_contiguous()
            or outer.dtype != torch.float32 or outer.numel()!=1 or not outer.is_contiguous()):
        raise ValueError("packed operand requires contiguous E4M3 scales and one FP32 outer scale")
    if scales.device != dy.device or outer.device != dy.device:
        raise ValueError("packed scales must share the operand device")
    torch._check(scales.numel() == data.numel()//8)
    return schedule,shape


@torch.library.custom_op("rtx_experimental::ms_eden_packed_gradient",mutates_args=(),device_types="cuda")
def _packed_gradient(dy:torch.Tensor,data:torch.Tensor,scales:torch.Tensor,outer:torch.Tensor,
                     signs:torch.Tensor,ua:torch.Tensor,ub:torch.Tensor,
                     phase:str,schedule_json:str)->torch.Tensor:
    schedule,shape=_packed_contract(dy,data,scales,outer,signs,ua,ub,phase,schedule_json)
    return _execute_gradient(dy,data,signs,ua,ub,phase,schedule,shape,scales.view(torch.float8_e4m3fn),outer)


@_packed_gradient.register_fake
def _packed_gradient_fake(dy,data,scales,outer,signs,ua,ub,phase,schedule_json):
    _,(m,n,_)=_packed_contract(dy,data,scales,outer,signs,ua,ub,phase,schedule_json)
    return dy.new_empty((m,n))


_packed_gradient.register_autograd(_no_higher_order)


def _pack_forward_contract(x,w,xs,ws):
    for value in (x,w):
        if value.ndim!=2 or value.dtype!=torch.bfloat16 or not value.is_contiguous():
            raise ValueError("contiguous rank-two BF16 inputs required")
        for size in value.shape:
            torch._check(size>0)
            torch._check(size%128==0)
    torch._check(x.shape[1]==w.shape[1])
    for value in (w,xs,ws):
        if value.device!=x.device:
            raise ValueError("packed forward inputs must share a device")
    for value in (xs,ws):
        if value.dtype!=torch.float32 or value.shape!=(1,) or not value.is_contiguous():
            raise ValueError("outer scales must be contiguous FP32 [1]")


@torch.library.custom_op("rtx_experimental::ms_eden_pack_forward",mutates_args=(),device_types="cuda")
def _pack_forward(x:torch.Tensor,w:torch.Tensor,xs:torch.Tensor,ws:torch.Tensor
                  )->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    _pack_forward_contract(x,w,xs,ws)
    from .fp4 import compile_nvfp4_quant,compile_nvfp4_gemm,_packed_fp4_view,_check_sm12x
    from .configs.nvfp4 import DEFAULT_NVFP4_QUANT_CONFIG,NVFP4GemmConfig,NVFP4Problem
    from .kernels.nvfp4_posthoc import compile_posthoc_scale_product
    m,k=x.shape; n=w.shape[0]
    with torch.cuda.device(x.device):
        _check_sm12x(x.device)
        qx=torch.empty(m,k//2,device=x.device,dtype=torch.uint8)
        qw=torch.empty(n,k//2,device=x.device,dtype=torch.uint8)
        sx=torch.empty(m,k//16,device=x.device,dtype=torch.float8_e4m3fn)
        sw=torch.empty(n,k//16,device=x.device,dtype=torch.float8_e4m3fn)
        alpha=torch.empty(1,device=x.device,dtype=torch.float32)
        out=torch.empty(m,n,device=x.device,dtype=torch.bfloat16)
        quant_x=compile_nvfp4_quant(m,k,DEFAULT_NVFP4_QUANT_CONFIG)
        quant_w=compile_nvfp4_quant(n,k,DEFAULT_NVFP4_QUANT_CONFIG)
        product=compile_posthoc_scale_product()
        gemm=compile_nvfp4_gemm(NVFP4Problem(m,n,k),NVFP4GemmConfig())
        px,pw=_packed_fp4_view(qx),_packed_fp4_view(qw)
        quant_x(x,qx,sx,xs); quant_w(w,qw,sw,ws)
        product(xs,ws,alpha)
        gemm(px,pw,sx,sw,out,alpha)
        return out,qx,sx.view(torch.uint8),qw,sw.view(torch.uint8)


@_pack_forward.register_fake
def _pack_forward_fake(x,w,xs,ws):
    _pack_forward_contract(x,w,xs,ws)
    m,k=x.shape; n=w.shape[0]
    return (x.new_empty((m,n)),x.new_empty((m,k//2),dtype=torch.uint8),
            x.new_empty((m,k//16),dtype=torch.uint8),x.new_empty((n,k//2),dtype=torch.uint8),
            x.new_empty((n,k//16),dtype=torch.uint8))


_pack_forward.register_autograd(_no_higher_order)


def ms_eden_gradient(dy, operand, signs, uniform_a, uniform_b, *, phase, schedule_json="{}"):
    """Explicit experimental dX or dW; output is BF16, accumulation FP32.

    ``phase='dx'`` computes dy @ W; ``phase='dw'`` computes dy.T @ X.
    Original BF16 operands are quantized using shared-sign Hadamard-128,
    E8M4 RTZ temporary scales, EDEN correction and stochastic E4M3 scales.
    Provide one independent uniform per 16 values in each *logical* operand,
    and the same 128 +/-1 signs for both. Contents are caller preconditions;
    no synchronizing host validation or random generation occurs here.

    ``schedule_json`` is a static serialized PosthocPreparationSchedule.
    Warm kernels before CUDA graph capture. This opaque operator permits user
    torch.compile orchestration; it does not capture an internal CUDA graph.
    """
    return _gradient(dy,operand,signs,uniform_a,uniform_b,phase,schedule_json)


def _random_gradient(dy, operand, phase, schedule_json, packed=None):
    # These Torch operations live in the traceable autograd formula, outside
    # the opaque kernel operator. AOTAutograd/Inductor can lower/fuse them.
    signs = torch.where(torch.rand(128,device=dy.device) < .5, -1., 1.)
    ua = torch.rand(dy.numel()//16,device=dy.device,dtype=torch.float32)
    ub = torch.rand(operand.numel()*(2 if packed is not None else 1)//16,device=dy.device,dtype=torch.float32)
    if packed is not None:
        return _packed_gradient(dy,operand,packed[0],packed[1],signs,ua,ub,phase,schedule_json)
    return ms_eden_gradient(dy,operand,signs,ua,ub,phase=phase,schedule_json=schedule_json)


class _ExperimentalLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, schedule_json, source_policy,forward_outer_mode,
                forward_scaling,forward_config,dx_schedule_json,dw_schedule_json):
        from .fp4 import nvfp4_linear,_outer_tensor_scale
        ctx.schedule_json = schedule_json
        ctx.dx_schedule_json = dx_schedule_json
        ctx.dw_schedule_json = dw_schedule_json
        ctx.source_policy = source_policy
        if source_policy == "forward_current_roundtrip_bf16":
            xs,ws=_outer_tensor_scale(x,forward_outer_mode),_outer_tensor_scale(weight,forward_outer_mode)
            out,qx,sx,qw,sw=_pack_forward(x,weight,xs,ws)
            # Detached plain tensors sever Python _base_inputs annotations on
            # quantizer outputs; no original-BF16 operand is saved here.
            ctx.save_for_backward(qx.detach(),sx.detach(),xs.detach(),qw.detach(),sw.detach(),ws.detach())
            return out
        ctx.save_for_backward(x,weight)
        return nvfp4_linear(x,weight,scaling=forward_scaling,autotune="off",
            scale_config=_FORWARD_POLICIES[forward_outer_mode],dynamic_config=forward_config)

    @staticmethod
    def backward(ctx, dy):
        dy = dy.contiguous()
        if ctx.source_policy == "forward_current_roundtrip_bf16":
            x,xs,xo,weight,ws,wo=ctx.saved_tensors
            dx=(_random_gradient(dy,weight,"dx",ctx.dx_schedule_json,(ws,wo))
                if ctx.needs_input_grad[0] else None)
            dw=(_random_gradient(dy,x,"dw",ctx.dw_schedule_json,(xs,xo))
                if ctx.needs_input_grad[1] else None)
            return dx,dw,None,None,None,None,None,None,None
        x,weight = ctx.saved_tensors
        dx = (_random_gradient(dy,weight,"dx",ctx.dx_schedule_json)
              if ctx.needs_input_grad[0] else None)
        dw = (_random_gradient(dy,x,"dw",ctx.dw_schedule_json)
              if ctx.needs_input_grad[1] else None)
        return dx,dw,None,None,None,None,None,None,None


def ms_eden_linear(x, weight, *, schedule_json="{}",source_policy="original_bf16",forward_outer_mode=None,
                   forward_scaling="current",forward_config=None,
                   dx_schedule_json=None,dw_schedule_json=None):
    """Research-only NVFP4 forward with MS-EDEN FP4 backward.

    By default original contiguous rank-two BF16 X/W are saved. The explicit
    forward_current_roundtrip_bf16 policy instead saves FP4 bytes, E4M3 scale
    bytes and FP32 outer scales, and computes gradients from BF16-rounded
    decoded forward operands. Explicit forward_outer_mode='exact' or 'power2'
    permits matched forward-policy comparisons. None preserves earlier defaults:
    power2 for original sources and exact for packed sources.
    Original-source mode supports current, jit_row_region, or block forward
    with an explicit NVFP4DynamicConfig. Packed-source mode retains only the
    fixed tensor-current forward: regional packed scale reuse is not implemented.
    Independent dx_schedule_json/dw_schedule_json override the shared schedule;
    these alter execution, not the gradient numerical policy. Stateful delayed
    forward is not supported by this stateless experimental function.
    All dimensions must be
    positive multiples of 128. Fresh shared signs and independent scale uniforms
    are generated for each requested gradient using PyTorch's RNG, outside the
    opaque gradient operator. RNG generation cost is part of training execution.
    This is not an unbiasedness guarantee: sparse-input conditional error is
    documented in operand_preparation_plan.md. Not a default or stable API.
    """
    if source_policy not in ("original_bf16","forward_current_roundtrip_bf16"):
        raise ValueError("unknown experimental source policy")
    if forward_scaling not in ("current","jit_row_region","block"):
        raise ValueError("experimental forward_scaling must be current, jit_row_region, or block")
    if source_policy == "forward_current_roundtrip_bf16" and (forward_scaling != "current" or forward_config is not None):
        raise ValueError("packed-source backward requires fixed tensor-current forward")
    if forward_outer_mode is None:
        forward_outer_mode="exact" if source_policy=="forward_current_roundtrip_bf16" else "power2"
    if forward_outer_mode not in ("exact","power2"):
        raise ValueError("forward_outer_mode must be exact or power2")
    for value in (x,weight):
        if value.ndim != 2 or value.dtype != torch.bfloat16 or not value.is_contiguous():
            raise ValueError("experimental linear requires contiguous rank-two BF16 inputs")
        for size in value.shape:
            torch._check(size > 0)
            torch._check(size % 128 == 0)
    torch._check(x.shape[1] == weight.shape[1])
    return _ExperimentalLinear.apply(x,weight,schedule_json,source_policy,forward_outer_mode,
        forward_scaling,forward_config,
        schedule_json if dx_schedule_json is None else dx_schedule_json,
        schedule_json if dw_schedule_json is None else dw_schedule_json)
