"""Experimental fixed-K balanced MXFP8 SwiGLU mixture of experts.

The initial grouped family materializes expert-major inputs and expert outputs,
but its forward hidden connection is MXFP8 and backward recomputes everything.
No per-expert launch loop, hidden activation save, or eager CUDA tensor math
exists inside the registered launchers. Shared addition remains compiler-visible.
"""
from dataclasses import asdict, dataclass, field
from functools import lru_cache
import json

import torch
from torch import nn

from .kernels.mxfp8 import MXFP8FwdConfig, MXFP8Problem, normalize_fwd_config
from .configs.glu import MXFP8GLUConfig, glu_config_from_dict, normalize_glu_config
from .configs.moe import MXFP8PreparedMoEConfig
from .routing import BalancedRouter, BalancedRouting
from .virtual import VirtualLayer


def _linear_default():
    return normalize_fwd_config(tile_m=32, tile_n=64)


def _gate_default():
    return normalize_glu_config(glu_tile=(32,128,2), glu_quant_vec=4)


@dataclass(frozen=True)
class MXFP8MoEConfig:
    """Independent expert GEMM and movement schedules; not a routing policy."""
    gate_up: MXFP8GLUConfig = field(default_factory=_gate_default)
    down: MXFP8FwdConfig = field(default_factory=_linear_default)
    recompute: MXFP8GLUConfig = field(default_factory=_gate_default)
    down_dx: MXFP8FwdConfig = field(default_factory=_linear_default)
    down_dw: MXFP8FwdConfig = field(default_factory=_linear_default)
    gate_dx: MXFP8FwdConfig = field(default_factory=_linear_default)
    gate_dw: MXFP8FwdConfig = field(default_factory=_linear_default)
    movement_threads: int = 128
    gather_mode: str = 'warp'
    family: str = 'grouped_prepared'
    onchip_down_stages: int = 1
    onchip_pipeline: str = 'serial'
    prepared: MXFP8PreparedMoEConfig = field(default_factory=MXFP8PreparedMoEConfig)

    def __post_init__(self):
        if self.family not in ('grouped_staged','grouped_onchip','grouped_prepared'):
            raise ValueError('Expert family must be grouped_staged/grouped_onchip/grouped_prepared')
        if not isinstance(self.prepared,MXFP8PreparedMoEConfig):raise TypeError('Expected prepared expert schedules')
        if self.onchip_down_stages not in (1,2,3,4) or self.onchip_pipeline not in ('serial','warp_specialized'):
            raise ValueError('Invalid on-chip buffering/pipeline configuration')
        if self.movement_threads not in (64,128,256,512):
            raise ValueError('movement_threads must be 64/128/256/512')
        if self.gather_mode not in ('flat','warp'):raise ValueError('gather_mode must be flat/warp')
        if not all(isinstance(getattr(self,name), MXFP8FwdConfig) for name in
                   ('gate_up','down','recompute','down_dx','down_dw','gate_dx','gate_dw')):
            raise TypeError('Expected MXFP8 GEMM configuration objects')
        if self.down.load_engine != 'scalar':
            raise ValueError('Packed hidden down GEMM currently requires scalar transport')
        for name in ('gate_up','recompute'):
            if not isinstance(getattr(self,name), MXFP8GLUConfig):
                raise TypeError('Gate and recompute configurations must be MXFP8GLUConfig')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        if 'prepared' in values:values['prepared']=MXFP8PreparedMoEConfig.from_dict(values['prepared'])
        for name in ('gate_up','recompute'):
            values[name] = glu_config_from_dict(values[name])
        for name in ('down','down_dx','down_dw','gate_dx','gate_dw'):
            values[name] = MXFP8FwdConfig(**values[name])
        return cls(**values)


DEFAULT_MOE_CONFIG = MXFP8MoEConfig()


@lru_cache(maxsize=128)
@torch.compiler.assume_constant_result
def _parse(key):
    return MXFP8MoEConfig.from_dict(json.loads(key))


@lru_cache(maxsize=128)
@torch.compiler.assume_constant_result
def _config_key(config):
    return json.dumps(config.to_dict(), sort_keys=True)


@torch.compiler.assume_constant_result
def _selection_spec(config,mode,policy,cache_dir):
    if mode not in ('off','cache','balanced'):
        raise ValueError('autotune must be off/cache/balanced')
    from .autotune.effort import AutotunePolicy
    policy = policy or AutotunePolicy.preset('balanced', search_schedule='progressive', candidate_mix='mixed')
    if not isinstance(policy,AutotunePolicy):
        raise TypeError('tuning_policy must be AutotunePolicy')
    return json.dumps(dict(initial=config.to_dict(), mode=mode, policy=asdict(policy),
                           cache_dir=None if cache_dir is None else str(cache_dir)), sort_keys=True)


@lru_cache(maxsize=256)
@torch.compiler.assume_constant_result
def _parse_spec(spec):
    return json.loads(spec)


@lru_cache(maxsize=256)
@torch.compiler.assume_constant_result
def _initial_config(spec):
    return MXFP8MoEConfig.from_dict(_parse_spec(spec)['initial'])


@torch.compiler.assume_constant_result
def _static_policy(spec):
    """Only scalar constants cross Dynamo's metadata boundary, not configs."""
    config=_initial_config(spec)
    return _parse_spec(spec)['mode'],config.family,_config_key(config)


@torch.compiler.assume_constant_result
def _static_family(key):
    return _parse(key).family


def _validate(x, gu, down, ids, weights, inverse):
    if (x.ndim!=2 or gu.ndim!=3 or down.ndim!=3 or min(x.shape)<=0
            or min(gu.shape)<=0 or gu.shape[1]%2):
        raise ValueError('Expected X[N,D], gate/up[E,2H,D], down[E,D,H]')
    n,d = x.shape
    e,h = gu.shape[0],gu.shape[1]//2
    if e>65535:
        raise ValueError('Grouped CUDA expert grids support at most 65535 experts')
    if gu.shape[2]!=d or tuple(down.shape)!=(e,d,h):
        raise ValueError('Expert weight dimensions disagree')
    if (ids.ndim!=2 or ids.shape[0]!=e or min(ids.shape)<=0
            or weights.shape!=ids.shape or inverse.ndim!=2 or inverse.shape[0]!=n
            or not 1<=inverse.shape[1]<=e):
        raise ValueError('Expected expert IDs/weights [E,C] and inverse [N,K]')
    if ids.shape[1]*e < n*inverse.shape[1]:
        raise ValueError('Insufficient scheduled expert capacity')
    if any(t.device!=x.device or not t.is_contiguous() for t in (x,gu,down,ids,weights,inverse)):
        raise ValueError('All expert operands and routing metadata must be contiguous on one device')
    if any(t.dtype!=torch.bfloat16 for t in (x,gu,down)):
        raise TypeError('Expert inputs/parameters must be BF16')
    if ids.dtype!=torch.int64 or inverse.dtype!=torch.int64 or weights.dtype!=torch.float32:
        raise TypeError('Routing IDs/inverse must be int64 and weights FP32')
    if max(n*d,e*ids.shape[1]*max(d,2*h),gu.numel(),down.numel(),inverse.numel())>=2**31:
        raise ValueError('Expert operands exceed native signed index range')


def _movement(source, ids, weights, inverse, out, config, kind, extra=None, auxiliary=None):
    from .runtime import load_kernel_symbol
    compile_movement=load_kernel_symbol('mxfp8_moe','compile_movement')
    n,k = inverse.shape
    e,c = ids.shape
    d = out.shape[1]
    storage_c=(out.shape[0] if kind in ('gather','slot_grad','slot_dy','route_grad') else source.shape[0])//e
    fn = compile_movement(n,d,e,c,k,config.movement_threads,kind,
                          config.gather_mode if kind=='gather' else 'flat',storage_c)
    placeholder=out if kind in ('gather','slot_dy') else source
    fn(source,ids,inverse,weights,placeholder if extra is None else extra,out,
       weights if auxiliary is None else auxiliary)


def _linear(a, b, out, problem, config, a_orientation='row', b_orientation='row', scales=None):
    from .runtime import load_kernel_symbol
    compile_grouped_linear=load_kernel_symbol('mxfp8_moe','compile_grouped_linear')
    fn = compile_grouped_linear(b.shape[0],problem,config,a_orientation,b_orientation,scales is not None)
    aux = out.flatten(0,1) if scales is None else scales.flatten(0,1).view(torch.float8_e8m0fnu)
    fn(a.flatten(0,1),b.flatten(0,1),out.flatten(0,1),aux,aux,aux,aux)


def _projections(xg, gu, down, config):
    from .runtime import load_kernel_symbol
    compile_grouped_glu=load_kernel_symbol('mxfp8_moe','compile_grouped_glu')
    e,c,d = xg.shape
    h = gu.shape[1]//2
    width = (h+31)//32*32
    hidden = torch.empty((e,c,width),device=xg.device,dtype=torch.float8_e4m3fn)
    scales = torch.empty((e,c,width//32),device=xg.device,dtype=torch.uint8)
    fn = compile_grouped_glu(e,MXFP8Problem(c,2*h,d),config.gate_up)
    sf = scales.flatten(0,1)
    fn(xg.flatten(0,1),gu.flatten(0,1),hidden.flatten(0,1),sf,sf,sf,sf)
    outputs = torch.empty((e,c,d),device=xg.device,dtype=torch.bfloat16)
    _linear(hidden,down,outputs,MXFP8Problem(c,d,h),config.down,scales=scales)
    return outputs,hidden,scales


def _run_forward(x,gu,down,ids,weights,inverse,config):
    from .runtime import validate_runtime_environment
    validate_runtime_environment()
    if not x.is_cuda or torch.cuda.get_device_capability(x.device)[0]!=12:
        raise RuntimeError('MXFP8 experts require the SM120/SM121 RTX instruction family')
    from ._moe_execution import run_phase
    return run_phase(x,gu,down,ids,weights,inverse,config)


def _run_backward(x,gu,down,ids,weights,inverse,dy,config):
    from ._moe_execution import run_phase
    return run_phase(x,gu,down,ids,weights,inverse,config,dy)


@torch.library.custom_op('rtx::mxfp8_routed_experts',mutates_args=(),device_types='cuda',
                         tags=(torch.Tag.cudagraph_unsafe,))
def _experts_op(x:torch.Tensor,gu:torch.Tensor,down:torch.Tensor,ids:torch.Tensor,
                weights:torch.Tensor,inverse:torch.Tensor,spec:str,training:bool
                )->tuple[torch.Tensor,torch.Tensor]:
    from ._execution_plan_token import PLAN_TOKEN_BYTES,write_plan_token
    _validate(x,gu,down,ids,weights,inverse)
    base=_parse_spec(spec)
    if base['mode']=='off':
        config=_initial_config(spec)
    else:
        from .autotune.moe import resolve_config
        config=resolve_config((x,gu,down,ids,weights,inverse),spec,training)
    token=torch.empty(PLAN_TOKEN_BYTES,device='cpu',dtype=torch.uint8)
    write_plan_token(token,_config_key(config))
    with torch.cuda.device(x.device):
        output=_run_forward(x,gu,down,ids,weights,inverse,config)
    if base['mode']!='off':
        from ._moe_selection import bind_selected
        shape=(x.shape[0],x.shape[1],gu.shape[0],ids.shape[1],inverse.shape[1],down.shape[2])
        bind_selected(shape,str(x.device),spec,training,_config_key(config))
    return output,token


@_experts_op.register_fake
def _experts_fake(x,gu,down,ids,weights,inverse,spec,training):
    from ._execution_plan_token import PLAN_TOKEN_BYTES
    return torch.empty_like(x,dtype=torch.float32),torch.empty(PLAN_TOKEN_BYTES,device='cpu',dtype=torch.uint8)


@torch.library.custom_op('rtx::mxfp8_routed_experts_backward',mutates_args=(),device_types='cuda',
                         tags=(torch.Tag.cudagraph_unsafe,))
def _experts_backward(x:torch.Tensor,gu:torch.Tensor,down:torch.Tensor,ids:torch.Tensor,
                     weights:torch.Tensor,inverse:torch.Tensor,dy:torch.Tensor,token:torch.Tensor
                     )->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    from ._execution_plan_token import read_plan_token
    with torch.cuda.device(x.device):
        return _run_backward(x,gu,down,ids,weights,inverse,dy,_parse(read_plan_token(token)))


@_experts_backward.register_fake
def _experts_backward_fake(x,gu,down,ids,weights,inverse,dy,token):
    return torch.empty_like(x),torch.empty_like(gu),torch.empty_like(down),torch.empty_like(weights)


def _setup(ctx,inputs,output):
    ctx.save_for_backward(*inputs[:6],output[1])
    ctx.mark_non_differentiable(output[1])


def _backward(ctx,gradient,_unused):
    x,gu,down,ids,weights,inverse,token=ctx.saved_tensors
    dx,dgu,ddown,dweights=_experts_backward(x,gu,down,ids,weights,inverse,gradient.contiguous(),token)
    return dx,dgu,ddown,None,dweights,None,None,None


_experts_op.register_autograd(_backward,setup_context=_setup)


def torch_routed_experts(x,gu,down,route):
    """Compiler-visible batched BF16 baseline, FP32 activation/combine.

    The reference materializes the gather and does no MXFP8 quantization. It
    is a baseline, not an implementation choice hidden under an MXFP8 label.
    """
    ids,weights,inverse=route
    _validate(x,gu,down,ids,weights,inverse)
    n,d=x.shape
    e,c=ids.shape
    h=gu.shape[1]//2
    slots=torch.arange(e*c,device=x.device).reshape(e,c)
    valid=(inverse[ids].eq(slots[...,None])).any(-1)
    gathered=x[ids]*valid[...,None]
    projections=torch.bmm(gathered,gu.transpose(1,2)).float()
    hidden=(torch.nn.functional.silu(projections[...,:h])*projections[...,h:]).bfloat16()
    outputs=torch.bmm(hidden,down.transpose(1,2))
    return (outputs.flatten(0,1)[inverse].float()*weights.flatten()[inverse][...,None]).sum(1)


def mxfp8_routed_experts(x,gate_up_weight,down_weight,route,*,config=DEFAULT_MOE_CONFIG,
                         autotune='off',tuning_policy=None,autotune_cache_dir=None):
    """Experimental routed SwiGLU sum in FP32, ready for shared addition.

    X/parameters are BF16; metadata must satisfy BalancedRouter's contract.
    Hidden is MXFP8, activations/combine FP32, parameter/input gradients BF16.
    Backward saves only X, weights, route metadata and immutable CPU plan data.
    Shared FFNs are not included. This first-order STE uses MXFP8 backward.
    """
    if not x.is_cuda:
        raise ValueError('MXFP8 routed execution requires an RTX CUDA device; use torch_routed_experts for a baseline')
    if not isinstance(config,MXFP8MoEConfig):
        raise TypeError('config must be MXFP8MoEConfig')
    spec=_selection_spec(config,autotune,tuning_policy,autotune_cache_dir)
    training=torch.is_grad_enabled() and any(t.requires_grad for t in (x,gate_up_weight,down_weight,route.route_weights))
    return _dispatch_experts(x,gate_up_weight,down_weight,*route,spec,training)


def _dispatch_experts(x,gu,down,ids,weights,inverse,spec,training):
    _validate(x,gu,down,ids,weights,inverse)
    mode,family,key=_static_policy(spec)
    if mode!='off':
        from ._moe_selection import bound_config
        shape=(x.shape[0],x.shape[1],gu.shape[0],ids.shape[1],inverse.shape[1],down.shape[2])
        key=bound_config(shape,str(x.device),spec,training)
        if key is None:
            # One real-input cold selection. The per-shape cell then changes
            # one guard, allowing the next call to compile exact typed arenas.
            # Its first forward still carries a self-contained backward token.
            return _experts_op(x,gu,down,ids,weights,inverse,spec,training)[0]
        family=_static_family(key)
    if family in ('grouped_prepared','grouped_staged','grouped_onchip'):
        from ._moe_execution import prepared_experts
        return prepared_experts(x,gu,down,ids,weights,inverse,key)
    return _experts_op(x,gu,down,ids,weights,inverse,spec,training)[0]


class BalancedMoE(nn.Module):
    """Experimental balanced MXFP8 SwiGLU experts plus ordinary shared SwiGLU.

    H and shared_hidden_features are post-gating widths. Each projection is
    its own optimizer-visible 2D BF16 Parameter over contiguous VirtualLayer
    storage. No bias, token dropping or implicit residual. Inputs are [N,D].
    ``backend='torch'`` explicitly selects the batched BF16 reference; it is
    never silently substituted for MXFP8. The prepared family is the default;
    staged and H=128 on-chip families remain autotuner choices. Expert scratch
    is automatic and compiler-visible, never a caller-managed workspace.
    This is not yet a speedup guarantee or a production-qualified replacement
    for an existing MoE.
    """
    def __init__(self,d_model,num_experts,expert_hidden_features,*,top_k,
                 shared_hidden_features,num_shared_experts=1,router_dim=64,
                 config=None,backend='mxfp8',autotune=None,shared_config=None,
                 tuning_policy=None,autotune_cache_dir=None,router_options=None,
                 device=None,dtype=torch.bfloat16):
        super().__init__()
        explicit=config is not None
        config=DEFAULT_MOE_CONFIG if config is None else config
        autotune=('off' if explicit else 'balanced') if autotune is None else autotune
        if not isinstance(config,MXFP8MoEConfig):raise TypeError('config must be MXFP8MoEConfig')
        if (any(type(v) is not int or v<=0 for v in (d_model,num_experts,expert_hidden_features,shared_hidden_features))
                or type(num_shared_experts) is not int or num_shared_experts<0):
            raise ValueError('Positive model/expert widths and nonnegative shared expert count required')
        if dtype!=torch.bfloat16 or backend not in ('mxfp8','torch'):
            raise ValueError('BalancedMoE uses BF16 parameters and mxfp8/torch backends')
        self.d_model,self.num_experts,self.expert_hidden_features=d_model,num_experts,expert_hidden_features
        self.top_k,self.config,self.backend=top_k,config,backend
        self.autotune,self.tuning_policy,self.autotune_cache_dir=autotune,tuning_policy,autotune_cache_dir
        self._expert_spec=_selection_spec(config,autotune,tuning_policy,autotune_cache_dir)
        from .mlp import DEFAULT_MLP_CONFIG,MXFP8MLPConfig
        from ._mlp_autotune import deferred_spec
        self.shared_config=DEFAULT_MLP_CONFIG if shared_config is None else shared_config
        if not isinstance(self.shared_config,MXFP8MLPConfig):raise TypeError('shared_config must be MXFP8MLPConfig')
        self._shared_spec=deferred_spec(self.shared_config,'mxfp8',autotune,tuning_policy,autotune_cache_dir)
        options=dict(router_options or {})
        options.setdefault('autotune',autotune)
        options.setdefault('tuning_policy',tuning_policy)
        options.setdefault('autotune_cache_dir',autotune_cache_dir)
        if backend=='torch':options.setdefault('backend','torch')
        self.router=BalancedRouter(d_model,num_experts,top_k=top_k,router_dim=router_dim,
                                   device=device,dtype=dtype,**options)
        def projection(out_features,in_features):
            return nn.Linear(in_features,out_features,bias=False,device=device,dtype=dtype)
        gate_up={f'e{e}_{kind}':projection(expert_hidden_features,d_model)
                 for e in range(num_experts) for kind in ('gate','up')}
        self.gate_up=VirtualLayer(gate_up,packed_shape=(num_experts,2*expert_hidden_features,d_model))
        self.down=VirtualLayer({f'e{e}':projection(d_model,expert_hidden_features) for e in range(num_experts)},
                               packed_shape=(num_experts,d_model,expert_hidden_features))
        self.shared_gate_up=nn.ModuleList(VirtualLayer(
            {kind:projection(shared_hidden_features,d_model) for kind in ('gate','up')},
            packed_shape=(2*shared_hidden_features,d_model)) for _ in range(num_shared_experts))
        self.shared_down=nn.ModuleList(projection(d_model,shared_hidden_features) for _ in range(num_shared_experts))
        self.reset_expert_parameters()

    def reset_expert_parameters(self):
        # Never reset the router's separate zero-projection initialization.
        for module in (self.gate_up,self.down,self.shared_gate_up,self.shared_down):
            for parameter in module.parameters():
                nn.init.trunc_normal_(parameter,std=.02,a=-.04,b=.04)

    def forward(self,x):
        route=self.router(x)
        gu,down=self.gate_up.weight,self.down.weight
        if self.backend=='torch':
            output=torch_routed_experts(x,gu,down,route)
        else:
            training=torch.is_grad_enabled() and (x.requires_grad or gu.requires_grad or down.requires_grad
                                                  or route.route_weights.requires_grad)
            output=_dispatch_experts(x,gu,down,*route,self._expert_spec,training)
        for gate,down in zip(self.shared_gate_up,self.shared_down):
            if self.backend=='torch':
                h=down.weight.shape[1]
                proj=torch.nn.functional.linear(x,gate.weight).float()
                hidden=(torch.nn.functional.silu(proj[:,:h])*proj[:,h:]).bfloat16()
                shared=torch.nn.functional.linear(hidden,down.weight)
            else:
                if self._shared_spec is None:
                    from .mlp import mxfp8_mlp
                    shared=mxfp8_mlp(x,gate.weight,down.weight,activation='silu',config=self.shared_config)
                else:
                    from ._mlp_autotune import selected_mlp
                    shared=selected_mlp(x,gate.weight,down.weight,'silu',self._shared_spec,
                        torch.is_grad_enabled() and (x.requires_grad or gate.weight.requires_grad or down.weight.requires_grad))[0]
            output=output+shared.float()
        return output.to(torch.bfloat16)

    def autotune_experts_(self,x,*,route=None,phase='train',tuning_policy=None,output_dir=None,progress=None):
        """Explicitly tune the complete routed-expert path, not policy/math.

        Router and shared MLP keep their own first-hit contexts. This method
        tunes expert gather, grouped projections/recompute, adjoints and combine
        together. It populates the same persistent cache used by first hit.
        """
        if torch.compiler.is_compiling():raise RuntimeError('Tune before compilation')
        if self.backend!='mxfp8' or not x.is_cuda:raise ValueError('MXFP8 CUDA expert tuning required')
        if phase not in ('forward','train'):raise ValueError('phase must be forward/train')
        if route is None:route=self.router(x)
        if tuning_policy is not None:self.tuning_policy=tuning_policy
        if output_dir is not None:self.autotune_cache_dir=str(output_dir)
        if self.autotune=='off':self.autotune='cache'
        self._expert_spec=_selection_spec(self.config,self.autotune,self.tuning_policy,self.autotune_cache_dir)
        from .autotune.moe import tune_moe,remember_config
        result=tune_moe((x,self.gate_up.weight,self.down.weight,*route),_parse_spec(self._expert_spec),
                        training=phase=='train',progress=progress)
        remember_config((x,self.gate_up.weight,self.down.weight,*route),self._expert_spec,
                        phase=='train',result.config)
        from ._moe_selection import publish_selected
        shape=(x.shape[0],x.shape[1],self.num_experts,route.expert_ids.shape[1],
               self.top_k,self.expert_hidden_features)
        publish_selected(shape,str(x.device),self._expert_spec,phase=='train',_config_key(result.config))
        return result
