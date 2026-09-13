"""Recomputed MXFP8 GLU MLP: two device kernels forward, no saved hidden state.

Experimental qualification is tracked in docs/mxfp8_mlp.md. This module owns
first-order autograd; the forward-only GLU API remains unchanged.
"""
from dataclasses import asdict, dataclass, field, fields, replace
from functools import lru_cache
import json

import torch

from .configs.glu import MXFP8GLUConfig, normalize_glu_config, glu_config_from_dict
from .configs.glu import MXFP8GLUPrequantConfig, MXFP8GLUGemmConfig, prequant_glu_config_from_dict
from .kernels.mxfp8 import MXFP8Problem, MXFP8FwdConfig, normalize_fwd_config
from .kernels.mxfp8_bwd import MXFP8BwdConfig
from .configs.mxfp8 import MXFP8QuantConfig,MXFP8GemmConfig


@dataclass(frozen=True)
class MXFP8MLPConfig:
    """Independent forward/backward policies for the recomputed MLP.

    ``None`` backward policies select the existing shape-aware MXFP8 defaults.
    ``down_backend`` chooses on-chip or producer-quantized down weights.
    ``recompute`` independently configures the optional packed backward GEMM.
    ``gate_up.glu_silu_math`` is the SiLU numerical policy for every forward
    and recompute backend; subordinate prequant configurations cannot change it.
    The cooperative forward adds a counter-initialization CUDA memset and is
    experimental. All quantized operands are transient, never cached weights.
    Use ``MXFP8MLP.autotune_`` outside compilation to search legal combinations.
    """
    gate_up: MXFP8GLUConfig = field(default_factory=lambda: normalize_glu_config(
        tile_m=64,quant_vec=4,quant_load_bits=16,glu_epilogue="registers",glu_quant_vec=4))
    down: MXFP8FwdConfig = field(default_factory=lambda: normalize_fwd_config(
        tile_m=64,quant_vec=4,quant_load_bits=16))
    gate_backward: MXFP8BwdConfig | None = None
    down_backward: MXFP8BwdConfig | None = None
    backward_recompute: str = "joint"
    backward_overlap: bool = False
    gate_backend: str = "onchip"
    cooperative_ctas_per_sm: float = 1.0
    gate_prequant: MXFP8GLUPrequantConfig = field(default_factory=lambda: MXFP8GLUPrequantConfig(
        quant=MXFP8QuantConfig(quant_vec=8,load_bits=128,quant_store_bits=32),
        gemm=MXFP8GLUGemmConfig(tile_m=64,atom_layout_m=2,stages=2,persistent_waves=1,
                              glu_epilogue="registers",glu_quant_vec=4)))
    recompute_backend: str = "onchip"
    recompute: MXFP8GLUPrequantConfig = field(default_factory=lambda: MXFP8GLUPrequantConfig(
        quant=MXFP8QuantConfig(quant_vec=8,load_bits=128,quant_store_bits=32),
        gemm=MXFP8GLUGemmConfig(tile_m=64,atom_layout_m=2,stages=2,
                              glu_epilogue="registers",glu_quant_vec=4)))
    down_backend: str = "onchip"
    down_quant: MXFP8QuantConfig = field(default_factory=lambda: MXFP8QuantConfig(
        quant_vec=8,load_bits=128,quant_store_bits=32))
    down_gemm: MXFP8GemmConfig = field(default_factory=lambda: MXFP8GemmConfig(
        epilogue="direct",store_vec=1))

    def __post_init__(self):
        if not isinstance(self.gate_up,MXFP8GLUConfig) or not isinstance(self.down,MXFP8FwdConfig):
            raise TypeError("gate_up/down must be MXFP8GLUConfig/MXFP8FwdConfig")
        if any(c is not None and not isinstance(c,MXFP8BwdConfig)
               for c in (self.gate_backward,self.down_backward)):
            raise TypeError("backward policies must be MXFP8BwdConfig or None")
        if self.backward_recompute not in ("joint","two_pass"):
            raise ValueError("backward_recompute must be joint or two_pass")
        if not isinstance(self.backward_overlap,bool):
            raise TypeError("backward_overlap must be a bool")
        if self.gate_backend not in ("onchip","cooperative_prequant"):
            raise ValueError("gate_backend must be onchip or cooperative_prequant")
        if self.cooperative_ctas_per_sm not in (.25,.5,1.,2.):
            raise ValueError("cooperative_ctas_per_sm must be .25, .5, 1 or 2")
        if not isinstance(self.gate_prequant,MXFP8GLUPrequantConfig):
            raise TypeError("gate_prequant must be MXFP8GLUPrequantConfig")
        if self.recompute_backend not in ("onchip","prequant"):
            raise ValueError("recompute_backend must be onchip or prequant")
        if not isinstance(self.recompute,MXFP8GLUPrequantConfig):
            raise TypeError("recompute must be MXFP8GLUPrequantConfig")
        if self.down_backend not in ("onchip","producer_quantized"):
            raise ValueError("down_backend must be onchip or producer_quantized")
        if not isinstance(self.down_quant,MXFP8QuantConfig) or not isinstance(self.down_gemm,MXFP8GemmConfig):
            raise TypeError("down_quant/down_gemm must be quantizer/GEMM configurations")

    def to_dict(self):
        """Return a revisioned JSON-serializable policy, without tensor state."""
        return {"revision":MLP_CONFIG_REVISION,**asdict(self)}

    @classmethod
    def from_dict(cls,values):
        """Restore an explicitly selected policy; reject stale schema revisions."""
        if not isinstance(values,dict):
            raise TypeError("configuration must be a dictionary")
        return _parse(json.dumps(values,sort_keys=True))

    def to_json(self):
        """Serialize a policy (not weights or a device-portable performance guarantee)."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text):
        """Restore a policy serialized by :meth:`to_json`."""
        return cls.from_dict(json.loads(text))


DEFAULT_MLP_CONFIG = MXFP8MLPConfig()
MLP_CONFIG_REVISION = 8


@torch.compiler.assume_constant_result
def _config_key(config):
    if not isinstance(config,MXFP8MLPConfig):
        raise TypeError("config must be MXFP8MLPConfig")
    return json.dumps({"revision":MLP_CONFIG_REVISION,**asdict(config)},sort_keys=True)


@lru_cache(maxsize=128)
def _parse(key):
    from .bwd_autotune import bwd_config_from_dict
    v=json.loads(key)
    if v.get("revision") != MLP_CONFIG_REVISION:
        raise ValueError("unsupported MXFP8 MLP configuration revision")
    unknown=set(v)-{"revision",*(f.name for f in fields(MXFP8MLPConfig))}
    if unknown:
        raise ValueError(f"unknown MXFP8 MLP configuration fields: {sorted(unknown)}")
    missing={"gate_up","down","gate_backward","down_backward"}-set(v)
    if missing:
        raise ValueError(f"missing MXFP8 MLP configuration fields: {sorted(missing)}")
    return MXFP8MLPConfig(
        gate_up=glu_config_from_dict(v["gate_up"]),down=MXFP8FwdConfig(**v["down"]),
        gate_backward=None if v["gate_backward"] is None else bwd_config_from_dict(v["gate_backward"]),
        down_backward=None if v["down_backward"] is None else bwd_config_from_dict(v["down_backward"]),
        down_backend=v.get("down_backend","onchip"),
        backward_overlap=v.get("backward_overlap",False),
        gate_backend=v.get("gate_backend","onchip"),
        cooperative_ctas_per_sm=v.get("cooperative_ctas_per_sm",1.),
        gate_prequant=prequant_glu_config_from_dict(v["gate_prequant"]) if "gate_prequant" in v else MXFP8MLPConfig().gate_prequant,
        recompute_backend=v.get("recompute_backend","onchip"),
        recompute=prequant_glu_config_from_dict(v["recompute"]) if "recompute" in v else MXFP8MLPConfig().recompute,
        down_quant=MXFP8QuantConfig(**v["down_quant"]) if "down_quant" in v else MXFP8MLPConfig().down_quant,
        down_gemm=MXFP8GemmConfig(**v["down_gemm"]) if "down_gemm" in v else MXFP8MLPConfig().down_gemm,
        backward_recompute=v.get("backward_recompute","two_pass"))


def _validate(x,gate_up,down,activation):
    from .glu import _validate as validate_glu
    validate_glu(x,gate_up,activation,"mxfp8")
    if down.ndim != 2 or down.shape[1] != gate_up.shape[0]//2 or down.shape[0] == 0:
        raise ValueError("down weight must have shape [out_features, hidden_features]")
    if down.device != x.device or down.dtype != torch.bfloat16 or not down.is_contiguous():
        raise ValueError("down weight must be contiguous BF16 on the input CUDA device")


@torch.library.custom_op("rtx::mxfp8_mlp",mutates_args=(),device_types="cuda")
def _mlp_op(x:torch.Tensor,gate_up:torch.Tensor,down:torch.Tensor,
            activation:str,config_key:str)->torch.Tensor:
    _validate(x,gate_up,down,activation)
    y=torch.empty((*x.shape[:-1],down.shape[0]),dtype=torch.bfloat16,device=x.device)
    _run_forward(x,gate_up,down,activation,_parse(config_key),y)
    return y


def _run_forward(x,gate_up,down,activation,c,y):
    _validate(x,gate_up,down,activation)
    from .runtime import validate_runtime_environment
    validate_runtime_environment()
    if torch.cuda.get_device_capability(x.device)[0] != 12:
        raise RuntimeError("MXFP8 MLP requires SM12x")
    from .kernels.mxfp8_glu import compile_mxfp8_glu
    from .kernels.mxfp8_mlp import compile_mixed_down
    m,k=x.numel()//x.shape[-1],x.shape[-1]
    h,o=gate_up.shape[0]//2,down.shape[0]
    width=(h+31)//32*32
    q=torch.empty((m,width),dtype=torch.float8_e4m3fn,device=x.device)
    s=torch.empty((m,width//32),dtype=torch.uint8,device=x.device)
    with torch.cuda.device(x.device):
        sf=s.view(torch.float8_e8m0fnu)
        if c.gate_backend == "cooperative_prequant":
            from .configs.glu import cooperative_glu_rejection
            reason=cooperative_glu_rejection(MXFP8Problem(m,2*h,k),c.gate_prequant.gemm)
            if reason:
                raise ValueError(reason)
            import math
            import cuda.bindings.driver as cuda
            from .kernels.mxfp8_glu_cooperative import compile_cooperative_glu
            from .kernels.mxfp8_gemm import compile_mxfp8_gemm
            if c.down_backend != "producer_quantized":
                raise ValueError("cooperative gate requires producer_quantized down")
            if c.gate_prequant.resolved_weight_quant()!=c.gate_prequant.quant or c.gate_prequant.quant_launches!="dual":
                raise ValueError("cooperative gate currently requires one shared X/weight quantizer policy")
            if c.gate_prequant.quant.scale_layout!="row_major" or c.down_quant.scale_layout!="row_major" or c.down_gemm.scale_layout!="row_major":
                raise ValueError("cooperative forward requires row-major scales")
            reason=c.gate_prequant.rejection(MXFP8Problem(m,2*h,k),"mxfp8")
            if reason:
                raise ValueError(reason)
            def empty(rows,cols,dtype):
                return torch.empty((rows,cols),device=x.device,dtype=dtype)
            qx,qgu=empty(m,k,torch.float8_e4m3fn),empty(2*h,k,torch.float8_e4m3fn)
            sx,sgu=empty(m,k//32,torch.float8_e8m0fnu),empty(2*h,k//32,torch.float8_e8m0fnu)
            qw,sw=empty(o,width,torch.float8_e4m3fn),empty(o,width//32,torch.float8_e8m0fnu)
            counter=empty(1,1,torch.int32)
            grid=math.ceil(torch.cuda.get_device_properties(x.device).multi_processor_count*c.cooperative_ctas_per_sm)
            first=compile_cooperative_glu(MXFP8Problem(m,2*h,k),replace(c.gate_prequant.gemm,glu_silu_math=c.gate_up.glu_silu_math),activation,
                                         c.gate_prequant.quant,o,c.down_quant,grid)
            second=compile_mxfp8_gemm(MXFP8Problem(m,o,width),c.down_gemm)
            status,=cuda.cuMemsetD32Async(counter.data_ptr(),0,1,torch.cuda.current_stream(x.device).cuda_stream)
            if status != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cooperative barrier initialization failed: {status}")
            first(qx,qgu,sx,sgu,q,(s,x.view(m,k),gate_up,down,qw,sw,counter))
            second(q,qw,sf,sw,y.view(m,o))
        elif c.down_backend == "producer_quantized":
            from .kernels.mxfp8_mlp import compile_glu_with_down_quant
            from .kernels.mxfp8_gemm import compile_mxfp8_gemm
            if c.down_gemm.scale_layout != "row_major":
                raise ValueError("producer-quantized down GEMM requires row-major scales")
            qw=torch.empty((o,width),dtype=torch.float8_e4m3fn,device=x.device)
            sw=torch.empty((o,width//32),dtype=torch.float8_e8m0fnu,device=x.device)
            first=compile_glu_with_down_quant(MXFP8Problem(m,2*h,k),c.gate_up,activation,o,c.down_quant)
            second=compile_mxfp8_gemm(MXFP8Problem(m,o,width),c.down_gemm)
            first(x.view(m,k),gate_up,q,s,down,qw,sw)
            second(q,qw,sf,sw,y.view(m,o))
        else:
            first=compile_mxfp8_glu(MXFP8Problem(m,2*h,k),c.gate_up,activation,"mxfp8")
            second=compile_mixed_down(MXFP8Problem(m,o,h),c.down)
            first(x.view(m,k),gate_up,q,s,s,s,s)
            second(q,down,y.view(m,o),sf,sf,sf,sf)


@_mlp_op.register_fake
def _mlp_fake(x,gate_up,down,activation,config_key):
    _validate(x,gate_up,down,activation)
    return x.new_empty((*x.shape[:-1],down.shape[0]))


@torch.library.custom_op("rtx::mxfp8_mlp_dequant",mutates_args=(),device_types="cuda")
def _dequant_op(data:torch.Tensor,scales:torch.Tensor,hidden:int)->torch.Tensor:
    from .kernels.mxfp8_mlp import compile_dequant_hidden
    out=torch.empty((data.shape[0],hidden),dtype=torch.bfloat16,device=data.device)
    with torch.cuda.device(data.device):
        compile_dequant_hidden(data.shape[0],hidden)(data,scales.view(torch.float8_e8m0fnu),out)
    return out


@_dequant_op.register_fake
def _dequant_fake(data,scales,hidden):
    return data.new_empty((data.shape[0],hidden),dtype=torch.bfloat16)


@torch.library.custom_op("rtx::mxfp8_mlp_derivative",mutates_args=(),device_types="cuda")
def _derivative_op(x:torch.Tensor,weight:torch.Tensor,grad_h:torch.Tensor,
                   activation:str,config_key:str)->torch.Tensor:
    policy=_parse(config_key)
    joint=policy.backward_recompute=="joint"
    shape=(((2*x.shape[0]+7)//8)*8+x.shape[0],weight.shape[0]//2) if joint else (x.shape[0],weight.shape[0])
    out=torch.empty(shape,dtype=torch.bfloat16,device=x.device)
    _run_derivative(x,weight,grad_h,activation,policy,out)
    return out


def _run_derivative(x,weight,grad_h,activation,policy,out):
    with torch.cuda.device(x.device):
        if policy.recompute_backend == "prequant":
            from .kernels.mxfp8_glu_prequant import build_prequant_glu_runner
            recompute=replace(policy.recompute,gemm=replace(policy.recompute.gemm,glu_silu_math=policy.gate_up.glu_silu_math))
            runner=build_prequant_glu_runner(x,weight,out,grad_h,recompute,
                        activation,"mxfp8",bind_inputs=False,derivative=True,
                        emit_hidden=policy.backward_recompute=="joint")
            runner(x,weight,out,grad_h)
        else:
            from .kernels.mxfp8_mlp import compile_glu_derivative
            c=replace(policy.gate_up,glu_store="direct",glu_store_stages=1)
            fn=compile_glu_derivative(MXFP8Problem(x.shape[0],weight.shape[0],x.shape[1]),c,
                                     activation,policy.backward_recompute=="joint")
            fn(x,weight,out,grad_h,grad_h,grad_h,grad_h)


@_derivative_op.register_fake
def _derivative_fake(x,weight,grad_h,activation,config_key):
    shape=(((2*x.shape[0]+7)//8)*8+x.shape[0],weight.shape[0]//2) if _parse(config_key).backward_recompute=="joint" else (x.shape[0],weight.shape[0])
    return x.new_empty(shape)


@lru_cache(maxsize=32)
def _overlap_stream(device_index,caller_stream):
    return torch.cuda.Stream(device=device_index)


@torch.library.custom_op("rtx::mxfp8_mlp_parallel_backward",mutates_args=(),device_types="cuda")
def _parallel_backward_op(dp:torch.Tensor,x:torch.Tensor,gu:torch.Tensor,dy:torch.Tensor,
                          hidden:torch.Tensor,down:torch.Tensor,gate_key:str,down_key:str
                          )->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    from .fp8_bwd import _launch_bwd_out,_launch_dw_out
    dx=torch.empty_like(x)
    dg=torch.empty_like(gu)
    dd=torch.empty_like(down)
    with torch.cuda.device(x.device):
        caller=torch.cuda.current_stream(x.device)
        parallel=_overlap_stream(x.device.index,int(caller.cuda_stream))
        parallel.wait_stream(caller)
        with torch.cuda.stream(parallel):
            _launch_dw_out(dy,hidden,down,down_key,dd)
        _launch_bwd_out(dp,x,gu,gate_key,dx,dg)
        caller.wait_stream(parallel)
    return dx,dg,dd


@_parallel_backward_op.register_fake
def _parallel_backward_fake(dp,x,gu,dy,hidden,down,gate_key,down_key):
    return x.new_empty(x.shape),gu.new_empty(gu.shape),down.new_empty(down.shape)


@torch.compiler.assume_constant_result
def _backward_keys(key,m,k,h,o):
    from .fp8_bwd import _default_bwd_config, _intern_bwd_config
    from .glu import _config_key as glu_key
    c=_parse(key)
    gate=c.gate_backward or _default_bwd_config(MXFP8Problem(m,2*h,k))
    down=c.down_backward or _default_bwd_config(MXFP8Problem(m,o,h))
    return glu_key(c.gate_up),_intern_bwd_config(gate),_intern_bwd_config(down)


def _setup_context(ctx,inputs,output):
    x,gate_up,down,activation,key=inputs
    ctx.save_for_backward(x,gate_up,down)
    ctx.activation,ctx.key=activation,key
    ctx.joint=_parse(key).backward_recompute=="joint"
    ctx.overlap=_parse(key).backward_overlap
    ctx.keys=_backward_keys(key,x.numel()//x.shape[-1],x.shape[-1],gate_up.shape[0]//2,down.shape[0])


def _backward(ctx,grad_output):
    from .glu import _glu_op
    from .fp8_bwd import _mxfp8_bwd_compiler_visible, _mxfp8_dx_compiler_visible, _mxfp8_dw_compiler_visible
    x,gate_up,down=ctx.saved_tensors
    need_x,need_gate,need_down=ctx.needs_input_grad[:3]
    m,k=x.numel()//x.shape[-1],x.shape[-1]
    h=gate_up.shape[0]//2
    x2=x.view(m,k)
    dy=grad_output.reshape(m,down.shape[0]).contiguous()
    glu_key,gate_key,down_key=ctx.keys
    joint=ctx.joint and (need_x or need_gate)
    if joint:
        # dH consumes dY/W only; the existing dX ABI needs a shape tensor.
        # This empty tensor is never read by that GEMM and launches no kernel.
        hidden=torch.empty((m,h),dtype=x.dtype,device=x.device)
    else:
        q,s=_glu_op(x2,gate_up,ctx.activation,"mxfp8",glu_key)
        hidden=_dequant_op(q,s,h)
    dh=ddown=None
    if need_x or need_gate:
        if need_down and not joint:
            dh,ddown=_mxfp8_bwd_compiler_visible(dy,hidden,down,down_key)
        else:
            dh=_mxfp8_dx_compiler_visible(dy,hidden,down,down_key)
    elif need_down:
        ddown=_mxfp8_dw_compiler_visible(dy,hidden,down,down_key)
    dx=dgate=None
    if need_x or need_gate:
        dp=_derivative_op(x2,gate_up,dh,ctx.activation,ctx.key)
        if ctx.joint:
            hidden=dp[((2*m+7)//8)*8:]
            dp=dp[:2*m].view(m,2*h)
            if ctx.overlap and need_x and need_gate and need_down:
                dx,dgate,ddown=_parallel_backward_op(dp,x2,gate_up,dy,hidden,down,gate_key,down_key)
                return dx.view_as(x),dgate,ddown,None,None
            if need_down:
                ddown=_mxfp8_dw_compiler_visible(dy,hidden,down,down_key)
        if need_x and need_gate:
            dx,dgate=_mxfp8_bwd_compiler_visible(dp,x2,gate_up,gate_key)
        elif need_x:
            dx=_mxfp8_dx_compiler_visible(dp,x2,gate_up,gate_key)
        else:
            dgate=_mxfp8_dw_compiler_visible(dp,x2,gate_up,gate_key)
    return None if dx is None else dx.view_as(x),dgate,ddown,None,None


torch.library.register_autograd("rtx::mxfp8_mlp",_backward,setup_context=_setup_context)


def mxfp8_mlp(x,gate_up_weight,down_weight,*,activation="silu",config=DEFAULT_MLP_CONFIG,
              precision="mxfp8"):
    """BF16 -> MXFP8 GLU -> MXFP8 down -> BF16; recompute-only first-order backward.

    Saves only X and the two BF16 weight tensors. The transient MXFP8 hidden
    tensor/scales connect the two forward kernels but are not retained by
    autograd. Quantization uses the linear layers' straight-through convention.
    Bias, dropout, residual addition, and higher-order gradients are not part
    of this operator. Configurations are explicit; no tuning during forward.

    ``precision='bf16_gate'`` instead uses BF16 gate forward with MXFP8 up/down
    and combined MXFP8 gate/up backward. ``'bf16'`` explicitly selects BF16
    projections. These traceable compositions retain FP32 activation/product
    and BF16 boundaries, but do not promise a two-kernel forward. Their eager
    training uses reentrant checkpointing and requires ``backward()``; use
    fullgraph compilation for compiled first-order gradients.
    """
    if precision not in ("mxfp8","bf16_gate","bf16"):
        raise ValueError("precision must be mxfp8, bf16_gate, or bf16")
    if precision != "mxfp8":
        _validate(x,gate_up_weight,down_weight,activation)
        if not isinstance(config,MXFP8MLPConfig):
            raise TypeError("config must be MXFP8MLPConfig")
        from ._mlp_precision import recomputed_mlp
        shape=x.shape
        return recomputed_mlp(x.reshape(-1,shape[-1]),gate_up_weight,down_weight,
                              activation,config,precision).reshape(*shape[:-1],down_weight.shape[0])
    return _mlp_op(x,gate_up_weight,down_weight,activation,_config_key(config))


def _clear_runtime_caches():
    """Release compiled helper references without retaining model tensors."""
    import sys
    from ._mlp_inductor import LAUNCHERS
    before={"launchers":len(LAUNCHERS),"configs":_parse.cache_info().currsize}
    LAUNCHERS.clear()
    _parse.cache_clear()
    _overlap_stream.cache_clear()
    kernels=sys.modules.get("rtx.kernels.mxfp8_mlp")
    if kernels is not None:
        for name in ("compile_mixed_down","compile_glu_derivative","compile_dequant_hidden","compile_glu_with_down_quant","compile_prequant_glu_derivative"):
            function=getattr(kernels,name)
            before[name]=function.cache_info().currsize
            function.cache_clear()
    cooperative=sys.modules.get("rtx.kernels.mxfp8_glu_cooperative")
    if cooperative is not None:
        function=cooperative.compile_cooperative_glu
        before["compile_cooperative_glu"]=function.cache_info().currsize
        function.cache_clear()
    return before


class MXFP8MLP(torch.nn.Module):
    """Bias-free GLU MLP with full recomputation and no saved hidden activations.

    Parameters and input/output are BF16. ``precision='mxfp8'`` uses MXFP8
    projections; ``'bf16_gate'`` protects the forward gate GEMM while keeping
    combined MXFP8 backward; ``'bf16'`` is an explicit compiler-managed fallback.
    Selective/BF16 paths use reentrant checkpointing: eager training requires
    ``backward()`` rather than ``autograd.grad()``. Compile with fullgraph=True.
    ``hidden_features`` is the post-GLU width, not the packed 2H weight width.
    ``out_features`` defaults to ``in_features``. ``activation`` accepts silu,
    exact GELU, or ReLU. Backward is first-order STE with full recomputation.
    Call ``autotune_`` explicitly before ``torch.compile(fullgraph=True)``;
    forward never starts a search. See docs/mxfp8_mlp.md for qualification.

    Construct on CPU/meta for checkpoint loading, then move to an SM12x CUDA
    device before execution. Only BF16 execution is supported. ``eval()`` does
    not freeze or prequantize weights. ``from_linears`` copies existing BF16
    gate/up/down projections without changing their source modules. Create a
    new optimizer after conversion: parameters are new objects.
    """
    def __init__(self,in_features,hidden_features,out_features=None,*,activation="silu",
                 config=DEFAULT_MLP_CONFIG,precision="mxfp8",device=None,dtype=torch.bfloat16):
        super().__init__()
        out_features=in_features if out_features is None else out_features
        if any(type(v) is not int or v<=0 for v in (in_features,hidden_features,out_features)):
            raise ValueError("feature dimensions must be positive integers")
        if dtype!=torch.bfloat16:
            raise TypeError("MXFP8MLP parameters must be BF16")
        if activation not in ("silu","gelu","relu"):
            raise ValueError("activation must be silu, gelu, or relu")
        if not isinstance(config,MXFP8MLPConfig):
            raise TypeError("config must be MXFP8MLPConfig")
        if precision not in ("mxfp8","bf16_gate","bf16"):
            raise ValueError("precision must be mxfp8, bf16_gate, or bf16")
        self.in_features,self.hidden_features,self.out_features=in_features,hidden_features,out_features
        self.activation,self.config=activation,config
        self.precision=precision
        self._shape_configs={}
        self.gate_up_weight=torch.nn.Parameter(torch.empty((2*hidden_features,in_features),device=device,dtype=dtype))
        self.down_weight=torch.nn.Parameter(torch.empty((out_features,hidden_features),device=device,dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self):
        for weight in (self.gate_up_weight,self.down_weight):
            torch.nn.init.trunc_normal_(weight,std=.02,a=-.04,b=.04)

    @classmethod
    def from_linears(cls, gate, up, down, *, activation="silu", config=DEFAULT_MLP_CONFIG,
                     precision="mxfp8"):
        """Copy three bias-free BF16 ``nn.Linear`` projections into a fused MLP.

        Gate/up must share shape and ``requires_grad`` because they become one
        parameter. All projections must share device, dtype, and training mode.
        No casts, device moves, source mutations, or optimizer migration occur.
        ``activation`` must match the source model (``gelu`` means exact GELU).
        """
        layers=(gate,up,down)
        if any(not isinstance(layer,torch.nn.Linear) for layer in layers):
            raise TypeError("gate, up and down must be nn.Linear modules")
        if any(layer.bias is not None for layer in layers):
            raise ValueError("MXFP8MLP does not support projection biases")
        if (gate.in_features,gate.out_features)!=(up.in_features,up.out_features):
            raise ValueError("gate and up projection shapes must match")
        if down.in_features!=gate.out_features:
            raise ValueError("down input width must equal gate/up output width")
        if any(layer.weight.dtype!=torch.bfloat16 for layer in layers):
            raise TypeError("source projection weights must be BF16")
        if any(layer.weight.device!=gate.weight.device for layer in layers):
            raise ValueError("source projections must share a device")
        if gate.weight.is_meta:
            raise ValueError("from_linears requires materialized weights, not meta tensors")
        if any(layer.training!=gate.training for layer in layers):
            raise ValueError("source projections must share training mode")
        if gate.weight.requires_grad!=up.weight.requires_grad:
            raise ValueError("gate/up requires_grad must match for the packed parameter")
        # Avoid random initialization and RNG consumption while copying weights.
        result=cls(gate.in_features,gate.out_features,down.out_features,
                   activation=activation,config=config,precision=precision,device="meta")
        result.gate_up_weight=torch.nn.Parameter(
            torch.cat((gate.weight.detach(),up.weight.detach()),dim=0),
            requires_grad=gate.weight.requires_grad)
        result.down_weight=torch.nn.Parameter(down.weight.detach().clone().contiguous(),
                                               requires_grad=down.weight.requires_grad)
        return result.train(gate.training)

    def forward(self,x):
        config,precision=self._shape_configs.get(self._shape_key(x),(self.config,self.precision))
        return mxfp8_mlp(x,self.gate_up_weight,self.down_weight,activation=self.activation,
                         config=config,precision=precision)

    @staticmethod
    def _shape_key(x):
        return (tuple(x.shape),str(x.device),str(x.dtype))

    def configure_shape_(self,example_x,*,config=None,precision=None):
        """Install an explicit shape/device policy before compile/capture.

        At most 128 entries are retained, oldest first. No tensors are cached.
        Precision is never selected by autotuning. Entries are process-local,
        are not state_dict state, and must be reinstalled after device moves.
        Unseen shapes retain constructor settings. Recompile after mutation.
        """
        if torch.compiler.is_compiling():
            raise RuntimeError("configure_shape_ must run before torch.compile")
        if example_x.device.type=="cuda":
            with torch.cuda.device(example_x.device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("configure_shape_ must run before CUDA graph capture")
        config=self.config if config is None else config
        precision=self.precision if precision is None else precision
        if not isinstance(config,MXFP8MLPConfig):
            raise TypeError("config must be MXFP8MLPConfig")
        if precision not in ("mxfp8","bf16_gate","bf16"):
            raise ValueError("precision must be mxfp8, bf16_gate, or bf16")
        if example_x.ndim<2 or example_x.shape[-1]!=self.in_features:
            raise ValueError("example_x must have matching input features and rank >= 2")
        key=self._shape_key(example_x)
        self._shape_configs.pop(key,None)
        self._shape_configs[key]=(config,precision)
        if len(self._shape_configs)>128:
            self._shape_configs.pop(next(iter(self._shape_configs)))
        return self

    def policy_for(self,example_x):
        """Return the effective ``(config, precision)`` without searching or mutating."""
        return self._shape_configs.get(self._shape_key(example_x),(self.config,self.precision))

    def autotune_(self,example_x,**kwargs):
        """Tune before compilation/capture and install the winner on this module.

        Defaults to ``phase='train'`` (all three gradients), even in eval mode;
        pass ``phase='forward'`` for inference. Common bounds are ``max_trials``
        and ``time_budget_s``. ``output_dir`` persists trial evidence. The chosen
        schedule is installed only for the example's shape/device/dtype;
        unseen shapes retain constructor settings. Precision never changes.
        Retune for a new workload/device and recompile/recapture afterward.
        """
        if torch.compiler.is_compiling():
            raise RuntimeError("autotune_ must run before torch.compile")
        _validate(example_x,self.gate_up_weight,self.down_weight,self.activation)
        with torch.cuda.device(example_x.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("autotune_ must run before CUDA graph capture")
        from .autotune.mlp import tune_mxfp8_mlp
        initial,precision=self._shape_configs.get(self._shape_key(example_x),(self.config,self.precision))
        self.tuning_result=tune_mxfp8_mlp(example_x,self.gate_up_weight,self.down_weight,
                                         activation=self.activation,initial=initial,precision=precision,**kwargs)
        self.configure_shape_(example_x,config=self.tuning_result.config,precision=precision)
        return self

    def extra_repr(self):
        return (f"in_features={self.in_features}, hidden_features={self.hidden_features}, "
                f"out_features={self.out_features}, activation={self.activation!r}, precision={self.precision!r}, recompute=True, bias=False")


from ._mlp_inductor import register as _register_inductor
_register_inductor()
