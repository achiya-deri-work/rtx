"""Fixed-K balanced routing, independent of expert execution."""
from dataclasses import dataclass,field,asdict
from functools import lru_cache
import math
from typing import NamedTuple

import torch
from torch import nn

from . import _routing_reference as reference


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
    """Execution choices only; never changes affinities or solver budgets."""
    forward: SinkhornKernelConfig = field(default_factory=SinkhornKernelConfig)
    backward: SinkhornKernelConfig = field(default_factory=SinkhornKernelConfig)
    segmented_quotas: bool = True
    quota_threshold: int = 1024
    qb_backend: str = 'torch'
    qb_threads: int = 128

    def __post_init__(self):
        if not isinstance(self.forward,SinkhornKernelConfig) or not isinstance(self.backward,SinkhornKernelConfig):
            raise TypeError('forward/backward must be SinkhornKernelConfig')
        if type(self.segmented_quotas) is not bool or type(self.quota_threshold) is not int or self.quota_threshold<0:
            raise ValueError('Invalid quota execution configuration')
        if self.qb_backend not in ('torch','native_column','native') or type(self.qb_threads) is not int or self.qb_threads not in (64,128,256):
            raise ValueError('Invalid QB execution configuration')

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
                ('torch','native_column','native').index(self.qb_backend),self.qb_threads]

    @classmethod
    def from_token(cls,token):
        a=list(token)
        if len(a)!=10:raise ValueError('Invalid immutable routing configuration token')
        return cls(SinkhornKernelConfig(*a[:3]),SinkhornKernelConfig(*a[3:6]),
                   bool(a[6]),a[7],('torch','native_column','native')[a[8]],a[9])


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
    return _sinkhorn_forward(scores.float().contiguous(),top_k,iterations,
                              c.threads,c.row_groups,c.column_groups)[0]


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
    """
    def __init__(self,d_model:int,num_experts:int,*,top_k:int=8,router_dim:int=64,
                 sinkhorn_iterations:int=5,dual_iterations:int=5,repair_rounds:int=64,
                 rms_epsilon:float=1e-6,padding:str='minimal',backend:str='auto',
                 sinkhorn_config:SinkhornKernelConfig|None=None,config:RoutingKernelConfig|None=None,
                 autotune:str|None=None,tuning_policy=None,autotune_cache_dir=None,
                 device=None,dtype=torch.bfloat16):
        super().__init__()
        if min(d_model,num_experts,router_dim)<=0 or not 1<=top_k<=num_experts:
            raise ValueError('Require positive dimensions and 1<=K<=E')
        if min(sinkhorn_iterations,dual_iterations,repair_rounds)<=0 or rms_epsilon<=0:
            raise ValueError('Invalid routing budgets/epsilon')
        if padding not in ('minimal','error'):raise ValueError('padding must be minimal/error')
        if backend not in ('auto','torch','cute'):raise ValueError('Invalid router backend')
        self.d_model,self.num_experts,self.top_k=d_model,num_experts,top_k
        self.router_dim=router_dim
        self.sinkhorn_iterations,self.dual_iterations,self.repair_rounds=sinkhorn_iterations,dual_iterations,repair_rounds
        self.rms_epsilon,self.padding,self.backend=rms_epsilon,padding,backend
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
        return reference.sigmoid_router_log_affinities(x,self.projection_weight,self.expert_tokens,
                self.logit_scale,self.score_bias,rms_epsilon=self.rms_epsilon)

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
            from ._routing_autotune import selected_sinkhorn
            out,_,token=selected_sinkhorn(loga,self.top_k,self.sinkhorn_iterations,self._selection_spec,
                                         torch.is_grad_enabled() and loga.requires_grad,x.shape[0])
            return out,token
        return log_sinkhorn(loga,top_k=self.top_k,iterations=self.sinkhorn_iterations,
                             config=self.sinkhorn_config,backend=self.backend),None

    def transport(self,x):
        """Differentiable log-transport before detached scheduling."""
        return self._transport(x)[0]

    def autotune_(self,x,*,tuning_policy=None,output_dir=None,phase='train',progress=None):
        """Tune one pool with shared UCB/residual storage before compilation.

        Populates durable winners; it does not silently impose one shape's
        configuration on all future shapes. Returns the shared tuning result.
        """
        from .autotune.routing import tune_routing
        if not x.is_cuda or self.backend=='torch':raise ValueError('Native CUDA routing required for tuning')
        with torch.no_grad():
            scores=self.log_affinities(x)
            ns=reference.padded_token_count(x.shape[0],self.num_experts,self.top_k)
            if self.padding=='error' and ns!=x.shape[0]:raise ValueError('Real token count*K is not divisible by E')
            scores=torch.nn.functional.pad(scores,(0,0,0,ns-x.shape[0]),value=-math.log(2))
        policy=tuning_policy or self.tuning_policy
        if policy is not None:self.tuning_policy=policy
        if output_dir is not None:self.autotune_cache_dir=str(output_dir)
        if self.autotune=='off':self.autotune='cache'
        self._refresh_selection_spec()
        from ._routing_autotune import parse_spec
        base=parse_spec(self._selection_spec)
        spec={**base,'workload':{**base['workload'],'real_tokens':x.shape[0]}}
        return tune_routing(scores,spec,phase=phase,policy=policy,progress=progress)

    def forward(self,x)->BalancedRouting:
        logp,token=self._transport(x)
        torch._assert_async(torch.isfinite(logp).all(),'BalancedRouter requires finite affinities/transport')
        scores=logp.detach().exp()
        if self.backend!='torch' and x.is_cuda:
            from ._routing_autotune import selected_hard
            ids=selected_hard(scores,token,self.top_k,self.dual_iterations,self.repair_rounds,id(self))
            capacity=logp.shape[0]*self.top_k//self.num_experts
            torch._assert_async((reference.assignment_loads(ids,self.num_experts)==capacity).all(),
                                'Balanced repair exhausted its budget')
        else:
            ids=reference.balanced_dual_price(scores,top_k=self.top_k,
                      iterations=self.dual_iterations,repair_rounds=self.repair_rounds)
        return BalancedRouting(*reference.routing_metadata(logp,ids,num_real_tokens=x.shape[0]))

    def extra_repr(self):
        return f'd_model={self.d_model}, num_experts={self.num_experts}, top_k={self.top_k}, backend={self.backend!r}, autotune={self.autotune!r}'


def _clear_runtime_caches(*,preserve_compiled=False):
    import sys
    result={}
    for name in ('rtx._routing_hard','rtx._routing_autotune'):
        module=sys.modules.get(name)
        if module is not None:result.update(module._clear_runtime_caches(preserve_compiled=preserve_compiled))
    return result
