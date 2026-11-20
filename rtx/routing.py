"""Fixed-K balanced routing, independent of expert execution."""
from dataclasses import dataclass,field,asdict
from functools import lru_cache
import math
from typing import NamedTuple

import torch
from torch import nn

from . import _routing_reference as reference


QB_BACKENDS=('torch','native_column','native','native_warp')
METADATA_BACKENDS=('torch','cute')
PACKED_SINKHORN_BACKENDS=('staged','persistent_forward','persistent_backward','persistent')
ROUTING_TOKEN_SIZE=25


@lru_cache(maxsize=16)
def _validate_native_device(index):
    from .runtime import validate_runtime_environment
    validate_runtime_environment()
    if torch.cuda.get_device_capability(index) not in ((12,0),(12,1)):
        raise RuntimeError('Native BalancedRouter requires RTX SM120/SM121')


@dataclass(frozen=True)
class SinkhornKernelConfig:
    """Independent reduction geometry; never changes the iteration budget."""
    threads: int = 128
    row_groups: int = 1
    column_groups: int = 4

    def __post_init__(self):
        if type(self.threads) is not int or self.threads not in (64,128,256):raise ValueError('threads must be 64/128/256')
        if (type(self.row_groups) is not int or type(self.column_groups) is not int
                or self.row_groups not in (1,2,4,8) or self.column_groups not in (1,2,4,8)):
            raise ValueError('reduction groups must be 1/2/4/8')


@dataclass(frozen=True)
class RoutingKernelConfig:
    """Execution choices only; never changes affinities or solver budgets.

    ``qb_backend='native_warp'`` selects exact warp-local row order statistics
    for at most 128 experts; column order statistics retain radix selection.
    Other backends remain available and the default is unchanged. ``qb_threads``
    controls 2/4/8 row-owning warps per CTA for this optional implementation.

    ``metadata_backend='cute'`` uses exact stable expert-slot construction and
    an FP32 selected-softmax adjoint for E<=128, K<=32; ``metadata_threads``
    selects 128/256/512 threads. ``torch`` retains the compiler-visible
    reference composition. No expert order or dummy-slot policy changes.

    ``quota_cache_items=4/8/16`` optionally caches that many radix keys per
    thread for segmented quota selection; oversized segments stream instead
    of truncating. Zero retains streaming selection. ``quota_early_exit``
    recovers the exact complete key when one candidate remains in a radix
    prefix. ``quota_threads`` selects 64/128/256 threads. All are execution
    choices exposed to the shared autotuner, not routing-policy changes.

    Packed independent pools additionally expose ``packed_row_backend='warp'``
    for register-resident, warp-local Sinkhorn row reductions (FP32, same saved
    normalization states), and ``packed_column_cache_items=4/8/16/32/64`` for
    exact QB column radix selection with bounded keys per thread. Longer
    columns stream without truncation. ``packed_column_shards_forward`` and
    ``packed_column_shards_backward`` select 4/8/16/32-way hierarchical FP32
    Sinkhorn column reductions; zero retains one CTA group over each column.
    In the sharded backend each group maps four adjacent experts to lanes for
    coalesced reads. These options are autotuner choices and are inactive
    without ``cu_seqlens``. Warp rows ignore ``row_groups``; column geometry
    and forward/backward thread counts remain independent.

    ``packed_qb_column_backend='expert_keys'`` optionally builds ordered FP32
    keys in an expert-major layout before exact QB radix selection. The
    default row-major path remains available, and the rank, tie, and repair
    policy do not change.

    ``packed_sinkhorn_backend`` optionally keeps a small pool in one CTA for
    all normalization iterations (forward, backward, or both). It retains
    FP32 normalization history. When the packed-total bound exceeds one CTA,
    a multi-pool invocation uses persistent CTAs for at most 128 scheduled
    rows per pool and the configured staged path for longer pools; boundary
    decisions stay on-device.
    ``packed_persistent_threads=512/1024`` optionally selects a wider CTA
    for each persistent direction; zero retains its ordinary thread count.

    Flat pools independently expose ``flat_column_shards_forward`` and
    ``flat_column_shards_backward`` (0/4/8/16/32/64/128). Nonzero values
    use hierarchical FP32 column reductions and compiler-visible partials;
    zero retains the serial column owner. ``flat_row_backend='warp'`` uses
    register-resident warp rows for E<=128, ignoring row_groups. These
    options are inactive for packed independent pools and never change
    solver budgets, saved-state ownership or precision.
    """
    forward: SinkhornKernelConfig = field(default_factory=SinkhornKernelConfig)
    backward: SinkhornKernelConfig = field(default_factory=SinkhornKernelConfig)
    segmented_quotas: bool = True
    quota_threshold: int = 1024
    qb_backend: str = 'torch'
    qb_threads: int = 128
    metadata_backend: str = 'torch'
    metadata_threads: int = 128
    quota_cache_items: int = 0
    quota_threads: int = 128
    quota_early_exit: bool = False
    packed_column_cache_items: int = 0
    packed_row_backend: str = 'cta'
    packed_column_shards_forward: int = 0
    packed_column_shards_backward: int = 0
    packed_qb_column_backend: str = 'row_major'
    packed_sinkhorn_backend: str = 'staged'
    packed_persistent_threads: int = 0
    flat_column_shards_forward: int = 0
    flat_column_shards_backward: int = 0
    flat_row_backend: str = 'cta'

    def __post_init__(self):
        for name in ('flat_column_shards_forward','flat_column_shards_backward'):
            if type(getattr(self,name)) is not int or getattr(self,name) not in (0,4,8,16,32,64,128):
                raise ValueError(f'{name} must be 0/4/8/16/32/64/128')
        if self.flat_row_backend not in ('cta','warp'):
            raise ValueError('flat_row_backend must be cta/warp')
        if not isinstance(self.forward,SinkhornKernelConfig) or not isinstance(self.backward,SinkhornKernelConfig):
            raise TypeError('forward/backward must be SinkhornKernelConfig')
        if type(self.segmented_quotas) is not bool or type(self.quota_threshold) is not int or self.quota_threshold<0:
            raise ValueError('Invalid quota execution configuration')
        if self.qb_backend not in QB_BACKENDS or type(self.qb_threads) is not int or self.qb_threads not in (64,128,256):
            raise ValueError('Invalid QB execution configuration')
        if self.metadata_backend not in METADATA_BACKENDS or type(self.metadata_threads) is not int or self.metadata_threads not in (128,256,512):
            raise ValueError('metadata_backend must be torch/cute and metadata_threads 128/256/512')
        if type(self.quota_cache_items) is not int or self.quota_cache_items not in (0,4,8,16):
            raise ValueError('quota_cache_items must be 0/4/8/16')
        if type(self.quota_threads) is not int or self.quota_threads not in (64,128,256):
            raise ValueError('quota_threads must be 64/128/256')
        if type(self.quota_early_exit) is not bool:raise TypeError('quota_early_exit must be bool')
        if type(self.packed_column_cache_items) is not int or self.packed_column_cache_items not in (0,4,8,16,32,64):
            raise ValueError('packed_column_cache_items must be 0/4/8/16/32/64')
        if self.packed_row_backend not in ('cta','warp'):raise ValueError('packed_row_backend must be cta/warp')
        for name in ('packed_column_shards_forward','packed_column_shards_backward'):
            value=getattr(self,name)
            if type(value) is not int or value not in (0,4,8,16,32):
                raise ValueError(f'{name} must be 0/4/8/16/32')
        if self.packed_qb_column_backend not in ('row_major','expert_keys'):
            raise ValueError('packed_qb_column_backend must be row_major/expert_keys')
        if self.packed_sinkhorn_backend not in PACKED_SINKHORN_BACKENDS:
            raise ValueError('Invalid packed_sinkhorn_backend')
        if type(self.packed_persistent_threads) is not int or self.packed_persistent_threads not in (0,512,1024):
            raise ValueError('packed_persistent_threads must be 0/512/1024')

    def to_dict(self):return asdict(self)

    @classmethod
    def from_dict(cls,value):
        value=dict(value)
        for name in ('forward','backward'):value[name]=SinkhornKernelConfig(**value[name])
        return cls(**value)

    def to_token(self):
        return [self.forward.threads,self.forward.row_groups,self.forward.column_groups,
                self.backward.threads,self.backward.row_groups,self.backward.column_groups,
                int(self.segmented_quotas),self.quota_threshold,
                QB_BACKENDS.index(self.qb_backend),self.qb_threads,
                METADATA_BACKENDS.index(self.metadata_backend),self.metadata_threads,
                self.quota_cache_items,self.quota_threads,int(self.quota_early_exit),
                self.packed_column_cache_items,int(self.packed_row_backend=='warp'),
                self.packed_column_shards_forward,self.packed_column_shards_backward,
                int(self.packed_qb_column_backend=='expert_keys'),
                PACKED_SINKHORN_BACKENDS.index(self.packed_sinkhorn_backend),self.packed_persistent_threads,
                self.flat_column_shards_forward,self.flat_column_shards_backward,int(self.flat_row_backend=='warp')]

    @classmethod
    def from_token(cls,token):
        a=list(token)
        if len(a)==10:a.extend((0,128,0,128))
        if len(a)==14:a.append(0)
        if len(a)==15:a.extend((0,0))
        if len(a)==17:a.extend((0,0))
        if len(a)==19:a.append(0)
        if len(a)==20:a.extend((0,0))
        if len(a)==22:a.extend((0,0,0))
        if len(a)!=ROUTING_TOKEN_SIZE:raise ValueError('Invalid immutable routing configuration token')
        if (not 0<=a[8]<len(QB_BACKENDS) or not 0<=a[10]<len(METADATA_BACKENDS)
                or not 0<=a[20]<len(PACKED_SINKHORN_BACKENDS)):
            raise ValueError('Invalid routing backend token index')
        return cls(SinkhornKernelConfig(*a[:3]),SinkhornKernelConfig(*a[3:6]),
                   bool(a[6]),a[7],QB_BACKENDS[a[8]],a[9],
                   METADATA_BACKENDS[a[10]],a[11],a[12],a[13],bool(a[14]),a[15],
                   ('cta','warp')[a[16]] if a[16] in (0,1) else 'invalid',a[17],a[18],
                   ('row_major','expert_keys')[a[19]] if a[19] in (0,1) else 'invalid',
                   PACKED_SINKHORN_BACKENDS[a[20]],a[21],a[22],a[23],
                   ('cta','warp')[a[24]] if a[24] in (0,1) else 'invalid')


class BalancedRouting(NamedTuple):
    expert_ids: torch.Tensor
    route_weights: torch.Tensor
    inverse: torch.Tensor


def _validate_sinkhorn(scores,top_k,iterations):
    if scores.ndim!=2 or min(scores.shape)<=0:raise ValueError('Expected nonempty [tokens,experts]')
    if type(top_k) is not int or not 1<=top_k<=scores.shape[1]:raise ValueError('Invalid top_k')
    if scores.shape[0]*top_k%scores.shape[1]:raise ValueError('Scheduled tokens*K must divide over experts')
    if type(iterations) is not int or iterations<1:raise ValueError('iterations must be positive')
    if not scores.is_floating_point():raise ValueError('Floating scores required')


@torch.library.custom_op('rtx_routing::sinkhorn_forward',mutates_args=(),device_types='cuda')
def _sinkhorn_forward(scores:torch.Tensor,top_k:int,iterations:int,threads:int,
                      row_groups:int,column_groups:int)->tuple[torch.Tensor,torch.Tensor]:
    with torch.cuda.device(scores.device):
        return _run_sinkhorn_forward(scores,top_k,iterations,threads,row_groups,column_groups)


def _run_sinkhorn_forward(scores,top_k,iterations,threads,row_groups,column_groups):
    from .runtime import load_kernel_symbol
    _validate_native_device(scores.device.index)
    compile_sinkhorn=load_kernel_symbol('routing_sinkhorn','compile_sinkhorn')
    _validate_sinkhorn(scores,top_k,iterations)
    if scores.dtype!=torch.float32 or not scores.is_contiguous():
        raise ValueError('Native Sinkhorn needs contiguous FP32 scores')
    SinkhornKernelConfig(threads,row_groups,column_groups)
    n,e=scores.shape
    if 2*iterations*n*e>=2**31:raise ValueError('Saved Sinkhorn state exceeds native index range')
    init,stages,finish,_=compile_sinkhorn(n,e,top_k,iterations,threads,row_groups,column_groups)
    out=torch.empty_like(scores)
    history=torch.empty(2*iterations*n*e,device=scores.device,dtype=torch.float32)
    row=torch.empty(n,device=scores.device,dtype=torch.float32)
    column=torch.empty(e,device=scores.device,dtype=torch.float32)
    src,dst=scores.view(-1),out.view(-1)
    init(src,column,row,history,dst,0)
    for i in range(iterations):
        stages[0](src,column,row,history,dst,i)
        stages[1](src,row,column,history,dst,i)
    finish(src,column,row,history,dst,0)
    return out,history


@_sinkhorn_forward.register_fake
def _sinkhorn_fake(scores,top_k,iterations,threads,row_groups,column_groups):
    return torch.empty_like(scores),torch.empty(2*iterations*scores.numel(),device=scores.device,dtype=scores.dtype)


@torch.library.custom_op('rtx_routing::sinkhorn_backward',mutates_args=(),device_types='cuda')
def _sinkhorn_backward(gradient:torch.Tensor,history:torch.Tensor,top_k:int,iterations:int,
                       threads:int,row_groups:int,column_groups:int)->torch.Tensor:
    with torch.cuda.device(gradient.device):
        return _run_sinkhorn_backward(gradient,history,top_k,iterations,threads,row_groups,column_groups)


def _run_sinkhorn_backward(gradient,history,top_k,iterations,threads,row_groups,column_groups):
    from .runtime import load_kernel_symbol
    _validate_native_device(gradient.device.index)
    compile_sinkhorn=load_kernel_symbol('routing_sinkhorn','compile_sinkhorn')
    _validate_sinkhorn(gradient,top_k,iterations)
    SinkhornKernelConfig(threads,row_groups,column_groups)
    n,e=gradient.shape
    if (gradient.dtype!=torch.float32 or not gradient.is_contiguous() or history.dtype!=torch.float32
            or history.device!=gradient.device or history.ndim!=1 or not history.is_contiguous()
            or history.numel()!=2*iterations*n*e):
        raise ValueError('Invalid saved Sinkhorn state or gradient layout')
    stages=compile_sinkhorn(n,e,top_k,iterations,threads,row_groups,column_groups)[3]
    a=torch.empty_like(gradient);b=torch.empty_like(gradient)
    unused=torch.empty(1,device=gradient.device,dtype=torch.float32)
    source=gradient.view(-1)
    for i in reversed(range(iterations)):
        stages[0](source,unused,unused,history,a.view(-1),i)
        stages[1](a.view(-1),unused,unused,history,b.view(-1),i)
        source=b.view(-1)
    return b


@_sinkhorn_backward.register_fake
def _sinkhorn_backward_fake(gradient,history,top_k,iterations,threads,row_groups,column_groups):
    return torch.empty_like(gradient)


def _sinkhorn_setup(ctx,inputs,output):
    _,ctx.top_k,ctx.iterations,ctx.threads,ctx.row_groups,ctx.column_groups=inputs
    logp,history=output
    ctx.mark_non_differentiable(history)
    ctx.save_for_backward(logp,history)


def _sinkhorn_grad(ctx,gradient,_history_gradient):
    if gradient is None:return (None,)*6
    _logp,history=ctx.saved_tensors
    result=_sinkhorn_backward(gradient.contiguous(),history,ctx.top_k,ctx.iterations,
                              ctx.threads,ctx.row_groups,ctx.column_groups)
    return result,None,None,None,None,None


_sinkhorn_forward.register_autograd(_sinkhorn_grad,setup_context=_sinkhorn_setup)


def log_sinkhorn(scores:torch.Tensor,*,top_k:int,iterations:int=5,
                  config:SinkhornKernelConfig|None=None,backend:str='auto')->torch.Tensor:
    """Finite FP32 balanced log-transport, with saved (not recomputed) state.

    CPU/``backend='torch'`` is the transparent policy reference. CUDA auto uses
    CuTe forward/backward. The native backward saves all row/column softmax
    Jacobian probabilities and the final log-transport; it never reruns the
    solver or recomputes its exponentials. Native differentiation is first-order.
    Saved native state costs ``(2*iterations+1)*N*E*4`` bytes, excluding the
    incoming scores and transient forward/backward workspace.
    """
    _validate_sinkhorn(scores,top_k,iterations)
    if backend not in ('auto','torch','cute'):raise ValueError('backend must be auto/torch/cute')
    if backend=='torch' or (backend=='auto' and scores.device.type=='cpu'):
        return reference.log_sinkhorn(scores,top_k=top_k,iterations=iterations)
    if not scores.is_cuda:raise ValueError('CuTe Sinkhorn requires CUDA')
    c=config or SinkhornKernelConfig()
    from ._routing_execution import sinkhorn
    geometry=(c.threads,c.row_groups,c.column_groups)
    return sinkhorn(scores.float().contiguous(),top_k,iterations,
                    geometry+geometry+(1,1024,0,128))[0]


class BalancedRouter(nn.Module):
    """FP32 sigmoid affinities -> Sinkhorn -> detached QB/repair -> soft weights.

    Exactly K distinct experts per real token; exactly equal *scheduled* expert
    capacities, using minimal zero-weight dummies when N*K is nondivisible.
    Integer decisions have no backward. Gradients pass through gathered
    Sinkhorn log-values and their per-token softmax. Shared/expert FFNs are not
    part of this module. Inputs are [N,D]; the balancing pool is explicit.

    ``backend='torch'`` is the portable policy oracle. ``auto`` selects native
    Sinkhorn on CUDA. CuTe order statistics and direct/packed radix quotas use
    cuda-python conditional graphs, without an NVCC build or caller workspace
    API. Compiler-visible proposal arithmetic retains the reference policy.
    No explicit config defaults to bounded balanced first-hit tuning; an
    explicit config defaults to tuning off. ``backend='torch'`` is the reference.
    ``affinity_layout='auto'`` can separately qualify the compiler-visible
    FP32 scalar-affine adjoint for CUDA training. ``'scalar'`` and
    ``'row_broadcast'`` explicitly select the two equivalent FP32 layouts.
    """
    def __init__(self,d_model:int,num_experts:int,*,top_k:int=8,router_dim:int=64,
                 sinkhorn_iterations:int=5,dual_iterations:int=5,repair_rounds:int=64,
                 rms_epsilon:float=1e-6,padding:str='minimal',backend:str='auto',
                 sinkhorn_config:SinkhornKernelConfig|None=None,config:RoutingKernelConfig|None=None,
                 autotune:str|None=None,tuning_policy=None,autotune_cache_dir=None,
                 affinity_layout:str='auto',device=None,dtype=torch.bfloat16):
        super().__init__()
        if min(d_model,num_experts,router_dim)<=0 or not 1<=top_k<=num_experts:
            raise ValueError('Require positive dimensions and 1<=K<=E')
        if min(sinkhorn_iterations,dual_iterations,repair_rounds)<=0 or rms_epsilon<=0:
            raise ValueError('Invalid routing budgets/epsilon')
        if padding not in ('minimal','error'):raise ValueError('padding must be minimal/error')
        if backend not in ('auto','torch','cute'):raise ValueError('Invalid router backend')
        if affinity_layout not in ('auto','scalar','row_broadcast'):raise ValueError('Invalid affinity layout')
        self.d_model,self.num_experts,self.top_k=d_model,num_experts,top_k
        self.router_dim=router_dim
        self.sinkhorn_iterations,self.dual_iterations,self.repair_rounds=sinkhorn_iterations,dual_iterations,repair_rounds
        self.rms_epsilon,self.padding,self.backend=rms_epsilon,padding,backend
        self.affinity_layout=affinity_layout
        if config is not None and sinkhorn_config is not None:raise ValueError('Choose config or sinkhorn_config, not both')
        explicit=config is not None or sinkhorn_config is not None
        seed=sinkhorn_config or SinkhornKernelConfig()
        self.config=config or RoutingKernelConfig(forward=seed,backward=seed)
        self.sinkhorn_config=self.config.forward
        self.autotune=('off' if explicit else 'balanced') if autotune is None else autotune
        self.tuning_policy=tuning_policy
        self.autotune_cache_dir=autotune_cache_dir
        self._refresh_selection_spec()
        self.projection_weight=nn.Parameter(torch.empty(router_dim,d_model,device=device,dtype=dtype))
        self.expert_tokens=nn.Parameter(torch.empty(num_experts,router_dim,device=device,dtype=dtype))
        self.logit_scale=nn.Parameter(torch.empty((),device=device,dtype=torch.float32))
        self.score_bias=nn.Parameter(torch.empty((),device=device,dtype=torch.float32))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        nn.init.zeros_(self.projection_weight)
        nn.init.normal_(self.expert_tokens,std=self.router_dim**-.5)
        self.logit_scale.fill_(-.5*math.log(self.router_dim));self.score_bias.zero_()

    def log_affinities(self,x):
        if x.ndim!=2 or x.shape[1]!=self.d_model or x.shape[0]<=0:
            raise ValueError('Expected positive [N,d_model] input')
        from ._routing_affinity import dispatch
        return dispatch(self._affinity_inputs(x),self.rms_epsilon,self.affinity_layout,
                        'off' if self.backend=='torch' else self.autotune,self._selection_spec)

    def _affinity_inputs(self,x):
        return x,self.projection_weight,self.expert_tokens,self.logit_scale,self.score_bias

    def _refresh_selection_spec(self):
        from ._routing_autotune import selection_spec
        self._selection_spec=selection_spec(self.config,self.autotune,self.tuning_policy,self.autotune_cache_dir,
            dict(d_model=self.d_model,router_dim=self.router_dim,rms_epsilon=self.rms_epsilon,
                 top_k=self.top_k,sinkhorn_iterations=self.sinkhorn_iterations,
                 dual_iterations=self.dual_iterations,repair_rounds=self.repair_rounds))

    def _transport(self,x):
        loga=self.log_affinities(x)
        scheduled=reference.padded_token_count(x.shape[0],self.num_experts,self.top_k)
        if self.padding=='error' and scheduled!=x.shape[0]:
            raise ValueError('Real token count*K is not divisible by E')
        loga=torch.nn.functional.pad(loga,(0,0,0,scheduled-x.shape[0]),value=-math.log(2.))
        if x.is_cuda and self.backend!='torch':
            from ._routing_autotune import dispatch_sinkhorn
            return dispatch_sinkhorn(loga,self.top_k,self.sinkhorn_iterations,self._selection_spec,
                                     torch.is_grad_enabled() and loga.requires_grad,x.shape[0])
        return log_sinkhorn(loga,top_k=self.top_k,iterations=self.sinkhorn_iterations,
                             config=self.sinkhorn_config,backend=self.backend),None

    def transport(self,x):
        """Differentiable log-transport before detached scheduling."""
        return self._transport(x)[0]

    def autotune_(self,x,*,cu_seqlens=None,tuning_policy=None,output_dir=None,phase='train',progress=None):
        """Tune one pool with shared UCB/residual storage before compilation.

        Populates durable winners; it does not silently impose one shape's
        configuration on all future shapes. Returns the shared tuning result.
        """
        from .autotune.routing import tune_routing
        if not x.is_cuda or self.backend=='torch':raise ValueError('Native CUDA routing required for tuning')
        if phase not in ('forward','train'):raise ValueError('phase must be forward/train')
        if x.ndim!=2 or x.shape[1]!=self.d_model or x.shape[0]<=0:
            raise ValueError('Expected positive [N,d_model] input')
        if cu_seqlens is not None:
            from ._routing_varlen import validate_cu
            validate_cu(x,cu_seqlens)
        else:
            ns=reference.padded_token_count(x.shape[0],self.num_experts,self.top_k)
            if self.padding=='error' and ns!=x.shape[0]:raise ValueError('Real token count*K is not divisible by E')
        if tuning_policy is not None:self.tuning_policy=tuning_policy
        if output_dir is not None:self.autotune_cache_dir=str(output_dir)
        if self.autotune=='off':self.autotune='cache'
        self._refresh_selection_spec()
        if phase=='train' and self.affinity_layout=='auto':
            from ._routing_affinity import prepare
            prepare(self._affinity_inputs(x),self._selection_spec,progress)
        with torch.no_grad():scores=self.log_affinities(x)
        if cu_seqlens is not None:
            from .autotune.varlen_routing import tune
            from ._routing_varlen_autotune import publish
            from ._routing_autotune import parse_spec
            result=tune(scores,cu_seqlens,parse_spec(self._selection_spec),training=phase=='train',
                strict=self.padding=='error',policy=self.tuning_policy,progress=progress)
            publish(scores,cu_seqlens,self.top_k,self.sinkhorn_iterations,self._selection_spec,
                    phase=='train',self.padding=='error',result.config)
            return result
        scores=torch.nn.functional.pad(scores,(0,0,0,ns-x.shape[0]),value=-math.log(2))
        from ._routing_autotune import parse_spec,publish_selected_config
        base=parse_spec(self._selection_spec)
        spec={**base,'workload':{**base['workload'],'real_tokens':x.shape[0]}}
        result=tune_routing(scores,spec,phase=phase,policy=self.tuning_policy,progress=progress)
        publish_selected_config(scores,self.top_k,self.sinkhorn_iterations,self._selection_spec,
                                phase=='train',x.shape[0],result.config)
        return result

    def forward(self,x,*,cu_seqlens=None)->BalancedRouting:
        if cu_seqlens is not None:
            from ._routing_varlen import cpu_routing
            if not x.is_cuda:return cpu_routing(self,x,cu_seqlens)
            if self.backend=='torch':raise ValueError('Packed CUDA routing needs backend=auto/cute; torch is a CPU packed reference')
            from ._routing_varlen_autotune import dispatch
            scores=self.log_affinities(x)
            return dispatch(scores,cu_seqlens,self.top_k,self.sinkhorn_iterations,self.dual_iterations,
                self.repair_rounds,self._selection_spec,torch.is_grad_enabled() and scores.requires_grad,
                self.padding=='error',id(self))
        logp,token=self._transport(x)
        torch._assert_async(torch.isfinite(logp).all(),'BalancedRouter requires finite affinities/transport')
        scores=logp.detach().exp()
        if self.backend!='torch' and x.is_cuda:
            if isinstance(token,tuple):
                from ._routing_hard import dispatch_hard
                ids=dispatch_hard(scores,self.top_k,self.dual_iterations,self.repair_rounds,
                                  bool(token[6]),id(self),token[7],
                                  QB_BACKENDS[token[8]],token[9],token[12],token[13],bool(token[14]))
            else:
                from ._routing_autotune import selected_hard
                ids=selected_hard(scores,token,self.top_k,self.dual_iterations,self.repair_rounds,id(self))
            capacity=logp.shape[0]*self.top_k//self.num_experts
            torch._assert_async((reference.assignment_loads(ids,self.num_experts)==capacity).all(),
                                'Balanced repair exhausted its budget')
        else:
            ids=reference.balanced_dual_price(scores,top_k=self.top_k,
                      iterations=self.dual_iterations,repair_rounds=self.repair_rounds)
        if isinstance(token,tuple) and token[10]==1:
            from ._routing_metadata import routing_metadata
            return BalancedRouting(*routing_metadata(logp,ids,num_real_tokens=x.shape[0],threads=token[11]))
        # The single cold-selection invocation retains the original traceable
        # metadata composition. Its subsequent bound plan uses the selected
        # implementation; both preserve stable IDs and the FP32 softmax policy.
        return BalancedRouting(*reference.routing_metadata(logp,ids,num_real_tokens=x.shape[0]))

    def extra_repr(self):
        return f'd_model={self.d_model}, num_experts={self.num_experts}, top_k={self.top_k}, backend={self.backend!r}, autotune={self.autotune!r}'


def _clear_runtime_caches(*,preserve_compiled=False):
    import sys
    result={}
    for name in ('rtx._routing_hard','rtx._routing_autotune','rtx._routing_execution','rtx._routing_metadata',
                 'rtx._routing_varlen','rtx._routing_varlen_autotune','rtx._routing_affinity'):
        module=sys.modules.get(name)
        if module is not None:result.update(module._clear_runtime_caches(preserve_compiled=preserve_compiled))
    return result
