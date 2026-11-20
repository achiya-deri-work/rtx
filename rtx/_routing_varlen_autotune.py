"""Actual-input packed routing selection, then guarded static hot execution."""
from concurrent.futures import ThreadPoolExecutor
import json
from threading import RLock

import torch

from .runtime import BoundedCache
from ._routing_autotune import _decode_bound_config,_static_policy,execution_config,parse_spec
from . import _routing_varlen as execution


_SELECTIONS=BoundedCache(256)
_LOCK=RLock()


def identity(scores,cu,k,steps,spec,training,strict):
    return (scores.shape[0],cu.numel()-1,scores.shape[1],k,steps,1+int(strict)),str(scores.device),'routing-varlen:'+spec,training


def publish(scores,cu,k,steps,spec,training,strict,config):
    from ._moe_selection import publish_selected
    key=identity(scores,cu,k,steps,spec,training,strict)
    _SELECTIONS[key]=config
    publish_selected(*key,json.dumps(config.to_token()))


def _resolve(scores,cu,k,steps,spec,training,strict):
    from .routing import RoutingKernelConfig
    from .autotune.varlen_routing import contract,tune
    from .autotune.winners import load_runtime_winner,runtime_tuning_lock
    base=parse_spec(spec);mode,initial=_static_policy(spec)
    original=RoutingKernelConfig.from_token(initial)
    if mode=='off':return original
    binding=identity(scores,cu,k,steps,spec,training,strict)
    with _LOCK,torch.cuda.device(scores.device):
        cached=_SELECTIONS.get(binding)
        if cached is not None:return cached
        if torch.cuda.is_current_stream_capturing():raise RuntimeError('Warm packed first-hit selection before manual capture')
        key,root,policy,lengths=contract(scores,cu,base,training,strict)
        def load():
            return load_runtime_winner(key,RoutingKernelConfig.from_dict,root=root,
                                      tuning_policy=policy if mode=='balanced' else None)
        chosen=load()
        if chosen is None and mode=='balanced':
            with runtime_tuning_lock(key,root=root):
                chosen=load()
                if chosen is None:
                    stream=torch.cuda.current_stream(scores.device)
                    def search():
                        from ._managed import tuning_resources
                        @tuning_resources
                        def run():
                            with torch.cuda.device(scores.device),torch.cuda.stream(stream),torch.enable_grad():
                                return tune(scores,cu,base,training=training,strict=strict,
                                            policy=policy,lengths=lengths,runtime_fallback=True)
                        return run()
                    with ThreadPoolExecutor(max_workers=1,thread_name_prefix='rtx-varlen-first-hit') as worker:
                        chosen=worker.submit(search).result().config
        chosen=chosen or original;_SELECTIONS[binding]=chosen
        return chosen


@torch.library.custom_op('rtx_routing::selected_varlen_route',mutates_args=(),device_types='cuda')
def selected_route(scores:torch.Tensor,cu:torch.Tensor,k:int,steps:int,qb:int,rounds:int,
                   spec:str,training:bool,strict:bool,owner:int)->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    from ._moe_selection import bind_selected
    execution.validate_cu(scores,cu)
    c=_resolve(scores,cu,k,steps,spec,training,strict)
    key=execution.layout(scores.shape[0],cu.numel()-1,scores.shape[1],k,steps,tuple(c.to_token()),strict)
    logp,history,scheduled,physical,sequence,local,rowmap,probabilities=execution.transport_forward(scores,cu,key)
    ids,weights,inverse=execution.route_from_transport(logp,scheduled,physical,sequence,local,rowmap,cu,key,c,qb,rounds,owner,probabilities)
    token=torch.tensor(c.to_token(),device='cpu',dtype=torch.int64)
    bind_selected(*identity(scores,cu,k,steps,spec,training,strict),json.dumps(c.to_token()))
    # Floating payload first also lets opcheck's generic output reduction
    # combine heterogeneous metadata without an in-place Long->Float cast.
    return weights,ids,inverse,history,scheduled,sequence,rowmap,token


@selected_route.register_fake
def _selected_fake(scores,cu,k,steps,qb,rounds,spec,training,strict,owner):
    from .routing import ROUTING_TOKEN_SIZE
    # Fake outputs depend only on shape, not execution policy. Do not hash
    # symbolic T/B in the static launch-layout cache during AOT opcheck.
    t=scores.shape[0];b=cu.numel()-1;e=int(scores.shape[1]);d=scores.device
    n,c=execution.workspace_bounds(t,b,e,k)
    return (torch.empty((e,c),device=d,dtype=torch.float32),torch.empty((e,c),device=d,dtype=torch.int64),
        torch.empty((t,k),device=d,dtype=torch.int64),
        torch.empty(2*steps*n*e,device=d,dtype=torch.float32),torch.empty(b+1,device=d,dtype=torch.int32),
        torch.empty(n,device=d,dtype=torch.int64),torch.empty(n,device=d,dtype=torch.int32),
        torch.empty(ROUTING_TOKEN_SIZE,device='cpu',dtype=torch.int64))


@torch.library.custom_op('rtx_routing::selected_varlen_backward',mutates_args=(),device_types='cuda')
def selected_backward(gradient:torch.Tensor,weights:torch.Tensor,inverse:torch.Tensor,history:torch.Tensor,
                      scheduled:torch.Tensor,sequence:torch.Tensor,rowmap:torch.Tensor,token:torch.Tensor,
                      steps:int,strict:bool)->torch.Tensor:
    c=execution_config(token)
    key=execution.layout(inverse.shape[0],scheduled.numel()-1,weights.shape[0],inverse.shape[1],steps,tuple(c.to_token()),strict)
    dense=torch.empty((key[4],key[2]),device=weights.device,dtype=torch.float32)
    execution.metadata_execute([weights.flatten(),inverse.flatten(),rowmap,gradient.contiguous().flatten()],
                               [dense.flatten()],execution._flat(key),True)
    return execution.transport_backward(dense,history,scheduled,sequence,rowmap,key)


@selected_backward.register_fake
def _backward_fake(gradient,weights,inverse,history,scheduled,sequence,rowmap,token,steps,strict):
    return torch.empty((inverse.shape[0],weights.shape[0]),device=weights.device,dtype=torch.float32)


def _setup(ctx,inputs,output):
    _,_,_,ctx.steps,_,_,_,_,ctx.strict,_=inputs
    weights,ids,inverse,history,scheduled,sequence,rowmap,token=output
    ctx.mark_non_differentiable(ids,inverse,history,scheduled,sequence,rowmap,token)
    ctx.save_for_backward(weights,inverse,history,scheduled,sequence,rowmap,token)


def _backward(ctx,gradient,_ids,*_):
    if gradient is None:return (None,)*10
    weights,inverse,history,scheduled,sequence,rowmap,token=ctx.saved_tensors
    value=selected_backward(gradient,weights,inverse,history,scheduled,sequence,rowmap,token,ctx.steps,ctx.strict)
    return value,None,None,None,None,None,None,None,None,None


selected_route.register_autograd(_backward,setup_context=_setup)


def dispatch(scores,cu,k,steps,qb,rounds,spec,training,strict,owner):
    from .routing import BalancedRouting,RoutingKernelConfig
    from ._moe_selection import bound_config
    execution.validate_cu(scores,cu)
    mode,token=_static_policy(spec)
    if mode!='off':
        key=bound_config(*identity(scores,cu,k,steps,spec,training,strict))
        if key is None:
            result=selected_route(scores,cu,k,steps,qb,rounds,spec,training,strict,owner)
            return BalancedRouting(result[1],result[0],result[2])
        token=_decode_bound_config(key)
    return execution.routing(scores,cu,top_k=k,steps=steps,qb=qb,rounds=rounds,
                             config=token,strict=strict,owner=owner)


def _clear_runtime_caches(*,preserve_compiled=False):
    if preserve_compiled:return {}
    from ._moe_selection import clear_selection_cells
    count=len(_SELECTIONS);_SELECTIONS.clear()
    cells=clear_selection_cells(namespace='routing-varlen:')
    return {'varlen_selections':count,'varlen_selection_cells':cells}
