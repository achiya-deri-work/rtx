"""Experimental fixed-K balanced MXFP8 SwiGLU mixture of experts.

The initial grouped family materializes expert-major inputs and expert outputs,
but its forward hidden connection is MXFP8 and backward recomputes by default.
Optional prepared training plans retain typed forward tensors for backward.
Shared addition remains compiler-visible.
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
    combine_reduce_engine: str = 'scalar'
    dx_reduce_engine: str = 'scalar'
    combine_vector_width: int = 4
    dx_vector_width: int = 4
    combine_lanes_per_token: int = 32
    dx_lanes_per_token: int = 32
    family: str = 'grouped_prepared'
    onchip_down_stages: int = 1
    onchip_pipeline: str = 'serial'
    prepared: MXFP8PreparedMoEConfig = field(default_factory=MXFP8PreparedMoEConfig)
    training_state: str = 'recompute'

    def __post_init__(self):
        if self.training_state not in ('recompute','saved_mx','saved_bf16_dw'):
            raise ValueError('Unknown MoE training state')
        if self.family not in ('grouped_staged','grouped_onchip','grouped_prepared'):
            raise ValueError('Expert family must be grouped_staged/grouped_onchip/grouped_prepared')
        if not isinstance(self.prepared,MXFP8PreparedMoEConfig):raise TypeError('Expected prepared expert schedules')
        if self.onchip_down_stages not in (1,2,3,4) or self.onchip_pipeline not in ('serial','warp_specialized'):
            raise ValueError('Invalid on-chip buffering/pipeline configuration')
        if self.movement_threads not in (64,128,256,512):
            raise ValueError('movement_threads must be 64/128/256/512')
        if self.gather_mode not in ('flat','warp'):raise ValueError('gather_mode must be flat/warp')
        for kind in ('combine','dx'):
            if getattr(self,kind+'_reduce_engine') not in ('scalar','token'):
                raise ValueError(f'{kind}_reduce_engine must be scalar/token')
            if getattr(self,kind+'_vector_width') not in (2,4,8):
                raise ValueError(f'{kind}_vector_width must be 2/4/8')
            if getattr(self,kind+'_lanes_per_token') not in (8,16,32):
                raise ValueError(f'{kind}_lanes_per_token must be 8/16/32')
        if not all(isinstance(getattr(self,name), MXFP8FwdConfig) for name in
                   ('gate_up','down','recompute','down_dx','down_dw','gate_dx','gate_dw')):
            raise TypeError('Expected MXFP8 GEMM configuration objects')
        if self.down.load_engine != 'scalar':
            raise ValueError('Packed hidden down GEMM currently requires scalar transport')
        for name in ('gate_up','recompute'):
            if not isinstance(getattr(self,name), MXFP8GLUConfig):
                raise TypeError('Gate and recompute configurations must be MXFP8GLUConfig')

    def to_dict(self):
        values = asdict(self)
        # Preserve the historical config IDs used by the proposal priors.
        # Runtime winners are separately scoped by kernel source revision.
        if self.training_state == "recompute":
            values.pop("training_state")
        return values

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


def balanced_moe_checkpoint_policy(fallback=None):
    """Build a PyTorch selective-checkpoint policy that saves packed hard assignments.

    Use with create_selective_checkpoint_contexts and non-reentrant checkpoint.
    Other operations follow the supplied PyTorch policy, or PREFER_RECOMPUTE.
    This does not enable checkpointing or change expert activation precision.
    """
    from torch.utils.checkpoint import CheckpointPolicy
    if fallback is not None and not callable(fallback):
        raise TypeError("fallback must be a callable checkpoint policy")

    def policy(ctx, op, *args, **kwargs):
        if str(op) == "rtx_routing.varlen_hard.default":
            return CheckpointPolicy.MUST_SAVE
        if fallback is None:
            return CheckpointPolicy.PREFER_RECOMPUTE
        return fallback(ctx, op, *args, **kwargs)

    return policy


def _refinement_policy(current, effort):
    from dataclasses import replace
    from .autotune.effort import AutotunePolicy
    if current is None:
        return AutotunePolicy.preset(effort or "balanced", cache_behavior="refine")
    if effort is None:
        return replace(current, cache_behavior="refine")
    budget = AutotunePolicy.preset(effort)
    return replace(current, max_trials=budget.max_trials, max_seconds=budget.max_seconds,
                   forward_trials=None, backward_trials=None, cache_behavior="refine")


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
    initial=config.to_dict()
    # Only the private module route binder may prove a pool bound. A raw
    # functional routing tensor cannot acquire one from a copied config.
    initial['prepared']['compact_pool_count']=0
    return json.dumps(dict(initial=initial, mode=mode, policy=asdict(policy),
                           cache_dir=None if cache_dir is None else str(cache_dir)), sort_keys=True)


@lru_cache(maxsize=256)
@torch.compiler.assume_constant_result
def _parse_spec(spec):
    return json.loads(spec)


@lru_cache(maxsize=256)
@torch.compiler.assume_constant_result
def _packed_expert_spec(spec,sequences):
    values=json.loads(spec)
    values['batch_contract']=dict(layout='independent_varlen',sequences=sequences,segment_alignment=32)
    if values['initial']['prepared'].get('forward_row_layout')=='compact':
        values['initial']['prepared']['compact_pool_count']=sequences
    return json.dumps(values,sort_keys=True)


@lru_cache(maxsize=256)
@torch.compiler.assume_constant_result
def _regular_expert_spec(spec):
    values=json.loads(spec)
    values['batch_contract']=dict(layout='regular',sequences=1,segment_alignment=32)
    if values['initial']['prepared'].get('forward_row_layout')=='compact':
        values['initial']['prepared']['compact_pool_count']=1
    return json.dumps(values,sort_keys=True)


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
    if config.training_state!='recompute':
        # The cold selector has a fixed output ABI. Recreate the selected
        # forward state for its first backward only; bound calls retain it.
        from ._moe_saved import saved_train_call
        return saved_train_call((x,gu,down,ids,weights,inverse),dy,config)[1:]
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
    Default backward saves only X, weights, route metadata and immutable plan
    data. The optional saved-state plans also retain packed operands and BF16
    expert outputs; the cuBLAS variant changes only the two weight gradients
    to BF16 GEMMs. Shared FFNs are not included. Autograd is first order.
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
        if training and _parse(key).training_state!='recompute':
            from ._moe_saved import SavedExperts
            return SavedExperts.apply(x,gu,down,ids,weights,inverse,key,_parse(key).training_state=='saved_bf16_dw')
        from ._moe_execution import prepared_experts
        return prepared_experts(x,gu,down,ids,weights,inverse,key)
    return _experts_op(x,gu,down,ids,weights,inverse,spec,training)[0]


class BalancedMoE(nn.Module):
    """Experimental balanced MXFP8 SwiGLU experts plus ordinary shared SwiGLU.

    H and shared_hidden_features are post-gating widths. Each projection is
    its own optimizer-visible 2D BF16 Parameter over contiguous VirtualLayer
    storage. No bias, token dropping or implicit residual. Inputs are [N,D]
    (one pool), [B,T,D] (independent pools), or packed [N,D] plus cu_seqlens.
    Outputs have the input shape. Tuning, durable caching and scratch ownership
    are automatic; torch.compile remains caller-owned.

    ``autotune='quick'/'balanced'/'thorough'/'extensive'`` requests 8/16/32/64
    trials per component with soft deadlines. No tuning objects are needed.
    ``off`` pins schedules and ``cache`` reuses winners without searching.
    ``tuning_policy`` and schedule/router configs are optional expert controls.
    Set ``tuning_policy=rtx.AutotunePolicy.preset("quick", allow_saved_moe=True)``
    to compare saved and recomputed routed training plans. This never forces a
    save. Set ``allow_bf16_dw=False`` on that policy to keep weight gradients MXFP8.
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
        if autotune in ('quick','thorough','extensive'):
            if tuning_policy is not None:
                raise ValueError('Use an autotune effort name or tuning_policy, not both')
            from .autotune.effort import AutotunePolicy
            tuning_policy=AutotunePolicy.preset(autotune)
            autotune='balanced'
        if not isinstance(config,MXFP8MoEConfig):raise TypeError('config must be MXFP8MoEConfig')
        if (any(type(v) is not int or v<=0 for v in (d_model,num_experts,expert_hidden_features,shared_hidden_features))
                or type(num_shared_experts) is not int or num_shared_experts<0):
            raise ValueError('Positive model/expert widths and nonnegative shared expert count required')
        if type(top_k) is not int or not 1<=top_k<=num_experts:
            raise ValueError('top_k must be an integer in [1,num_experts]')
        if dtype!=torch.bfloat16 or backend not in ('mxfp8','torch'):
            raise ValueError('BalancedMoE uses BF16 parameters and mxfp8/torch backends')
        self.d_model,self.num_experts,self.expert_hidden_features=d_model,num_experts,expert_hidden_features
        self.shared_hidden_features,self.num_shared_experts=shared_hidden_features,num_shared_experts
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

    def reset_parameters(self):
        """Restore the router and std=0.02 truncated-normal expert recipe."""
        self.router.reset_parameters()
        self.reset_expert_parameters()

    def extra_repr(self):
        from .autotune.effort import AutotunePolicy
        trials=(self.tuning_policy or AutotunePolicy.preset('balanced')).max_trials
        return (f'd_model={self.d_model}, num_experts={self.num_experts}, top_k={self.top_k}, '
                f'expert_hidden_features={self.expert_hidden_features}, '
                f'shared_hidden_features={self.shared_hidden_features}, '
                f'num_shared_experts={self.num_shared_experts}, backend={self.backend!r}, '
                f'autotune={self.autotune!r}, max_trials={trials}')

    def __repr__(self):
        # Hundreds of logical optimizer projections are an implementation
        # detail, not a useful default model summary. They remain registered.
        return f'{self.__class__.__name__}({self.extra_repr()})'

    def _input_layout(self,x,cu_seqlens):
        if x.ndim not in (2,3) or x.shape[-1]!=self.d_model or any(s<=0 for s in x.shape):
            raise ValueError('BalancedMoE expects nonempty [N,d_model] or [B,T,d_model] input')
        if x.dtype!=torch.bfloat16 or not x.is_contiguous():
            raise ValueError('BalancedMoE input must be contiguous BF16')
        if x.device!=self.router.projection_weight.device:
            raise ValueError('BalancedMoE input and parameters must be on the same device')
        if x.ndim==3:
            if cu_seqlens is not None:
                raise ValueError('cu_seqlens is only used with packed 2D input; dense batches define their own pools')
            batch,length,width=x.shape
            if batch*length>=2**31:raise ValueError('Dense token count exceeds int32 packed-offset range')
            # Logical flattening, plus compiler-visible regular boundaries.
            # Never copy activations or build one Python graph per sample.
            cu_seqlens=torch.arange(batch+1,device=x.device,dtype=torch.int32)*length
            x=x.view(batch*length,width)
        return x,cu_seqlens

    def forward(self,x,*,cu_seqlens=None):
        """Return BF16 with input shape; dense/packed samples route independently."""
        original_shape=x.shape
        x,cu_seqlens=self._input_layout(x,cu_seqlens)
        route=self.router(x,cu_seqlens=cu_seqlens)
        gu,down=self.gate_up.weight,self.down.weight
        if self.backend=='torch':
            output=torch_routed_experts(x,gu,down,route)
        else:
            training=torch.is_grad_enabled() and (x.requires_grad or gu.requires_grad or down.requires_grad
                                                  or route.route_weights.requires_grad)
            spec=_regular_expert_spec(self._expert_spec) if cu_seqlens is None else _packed_expert_spec(self._expert_spec,cu_seqlens.numel()-1)
            output=_dispatch_experts(x,gu,down,*route,spec,training)
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
                    from ._moe_shared import selected_shared
                    shared=selected_shared(x,gate.weight,down.weight,'silu',self._shared_spec,
                        torch.is_grad_enabled() and (x.requires_grad or gate.weight.requires_grad or down.weight.requires_grad))
            output=output+shared.float()
        return output.to(torch.bfloat16).view(original_shape)

    def autotune_(self,x,*,cu_seqlens=None,effort=None,tuning_policy=None,
                  output_dir=None,phase='train'):
        """Optionally prepare all components before caller compilation/capture.

        Normal forward already tunes automatically. This convenience method
        uses the same first-hit path and caches for router, experts and shared
        MLP, then warms the bound path. It returns self and leaves parameter
        gradients, input requires_grad and train/eval state untouched. Budgets
        are per component, not global. Run outside torch.compile/CUDA capture.
        Choose phase='forward' for no-grad inference; eval() alone is not that.
        """
        if torch.compiler.is_compiling():raise RuntimeError('Call autotune_ before torch.compile')
        if self.backend!='mxfp8' or not x.is_cuda:raise ValueError('autotune_ requires MXFP8 CUDA execution')
        if phase not in ('train','forward'):raise ValueError('phase must be train/forward')
        with torch.cuda.device(x.device):
            if torch.cuda.is_current_stream_capturing():raise RuntimeError('Call autotune_ before CUDA graph capture')
        flat,cu_seqlens=self._input_layout(x,cu_seqlens)
        if effort is not None and tuning_policy is not None:
            raise ValueError('Choose effort or tuning_policy, not both')
        policy=tuning_policy if tuning_policy is not None else _refinement_policy(self.tuning_policy, effort)
        cache_dir=self.autotune_cache_dir if output_dir is None else str(output_dir)
        expert_spec=_selection_spec(self.config,'balanced',policy,cache_dir)
        from ._mlp_autotune import deferred_spec
        shared_spec=deferred_spec(self.shared_config,'mxfp8','balanced',policy,cache_dir)
        self.tuning_policy,self.autotune_cache_dir,self.autotune=policy,cache_dir,'balanced'
        self._expert_spec,self._shared_spec=expert_spec,shared_spec
        self.router.autotune='balanced';self.router.tuning_policy=self.tuning_policy
        self.router.autotune_cache_dir=self.autotune_cache_dir;self.router._refresh_selection_spec()
        with torch.inference_mode(False),torch.set_grad_enabled(phase=='train'):
            value=flat.detach()
            if phase=='train' and value.is_inference():value=value.clone()
            value.requires_grad_(phase=='train')
            leaves=(value,*(p for p in self.parameters() if p.requires_grad))
            for _ in range(2):
                result=self(value,cu_seqlens=cu_seqlens)
                if phase=='train':
                    torch.autograd.grad(result,leaves,torch.ones_like(result))
        return self

    def autotune_experts_(self,x,*,cu_seqlens=None,route=None,phase='train',tuning_policy=None,output_dir=None,progress=None):
        """Explicitly tune the complete routed-expert path, not policy/math.

        Router and shared MLP keep their own first-hit contexts. This method
        tunes expert gather, grouped projections/recompute, adjoints and combine
        together. It populates the same persistent cache used by first hit.
        """
        if torch.compiler.is_compiling():raise RuntimeError('Tune before compilation')
        if self.backend!='mxfp8' or not x.is_cuda:raise ValueError('MXFP8 CUDA expert tuning required')
        if phase not in ('forward','train'):raise ValueError('phase must be forward/train')
        x,cu_seqlens=self._input_layout(x,cu_seqlens)
        if cu_seqlens is not None:
            from ._routing_varlen import validate_cu
            validate_cu(x,cu_seqlens)
        external_route=route is not None
        if route is None:route=self.router(x,cu_seqlens=cu_seqlens)
        if tuning_policy is not None:self.tuning_policy=tuning_policy
        if output_dir is not None:self.autotune_cache_dir=str(output_dir)
        if self.autotune=='off':self.autotune='cache'
        self._expert_spec=_selection_spec(self.config,self.autotune,self.tuning_policy,self.autotune_cache_dir)
        spec=_regular_expert_spec(self._expert_spec) if cu_seqlens is None else _packed_expert_spec(self._expert_spec,cu_seqlens.numel()-1)
        if external_route:
            unproven=json.loads(spec)
            unproven.pop('batch_contract',None)
            unproven['initial']['prepared']['compact_pool_count']=0
            spec=json.dumps(unproven,sort_keys=True)
        from .autotune.moe import tune_moe,remember_config
        result=tune_moe((x,self.gate_up.weight,self.down.weight,*route),_parse_spec(spec),
                        training=phase=='train',progress=progress)
        remember_config((x,self.gate_up.weight,self.down.weight,*route),spec,
                        phase=='train',result.config)
        from ._moe_selection import publish_selected
        shape=(x.shape[0],x.shape[1],self.num_experts,route.expert_ids.shape[1],
               self.top_k,self.expert_hidden_features)
        publish_selected(shape,str(x.device),spec,phase=='train',_config_key(result.config))
        return result
