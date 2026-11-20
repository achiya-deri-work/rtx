"""Real-input cold selection and guarded static-policy router dispatch.

Only cold/compatibility calls carry a CPU policy token. Bound Sinkhorn and
hard-routing launches consume immutable scalar policy with visible allocation.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from functools import lru_cache
import json
import logging
from pathlib import Path
from threading import RLock

import torch

from .runtime import BoundedCache

_selections=BoundedCache(256)
_selection_lock=RLock()
_LOG=logging.getLogger('rtx.routing.autotune')


@lru_cache(maxsize=256)
def decode_token(values):
    from .routing import RoutingKernelConfig
    return RoutingKernelConfig.from_token(values)


def execution_config(token):
    from .routing import ROUTING_TOKEN_SIZE
    sizes=(10,14,15,17,19,20,22,ROUTING_TOKEN_SIZE)
    if token.device.type!='cpu' or token.dtype!=torch.int64 or token.ndim!=1 or token.numel() not in sizes:
        raise ValueError(f'Routing execution token must be CPU int64 with length in {sizes}; CUDA readback is forbidden')
    return decode_token(tuple(token.tolist()))


@lru_cache(maxsize=256)
@torch.compiler.assume_constant_result
def initial_config(spec):
    from .routing import RoutingKernelConfig
    return RoutingKernelConfig.from_dict(parse_spec(spec)['initial'])


def selection_spec(config,mode,policy,cache_dir,workload):
    from .autotune.effort import AutotunePolicy
    if mode not in ('off','cache','balanced'):raise ValueError('autotune must be off/cache/balanced')
    if policy is not None and not isinstance(policy,AutotunePolicy):raise TypeError('tuning_policy must be AutotunePolicy')
    policy=policy or AutotunePolicy.preset('balanced',search_schedule='progressive',candidate_mix='mixed')
    return json.dumps(dict(schema=1,initial=config.to_dict(),mode=mode,policy=asdict(policy),
                           cache_dir=None if cache_dir is None else str(Path(cache_dir).expanduser()),
                           workload=workload),sort_keys=True)


@lru_cache(maxsize=256)
@torch.compiler.assume_constant_result
def parse_spec(value):
    return json.loads(value)


@lru_cache(maxsize=256)
@torch.compiler.assume_constant_result
def _static_policy(spec):
    return parse_spec(spec)['mode'],tuple(initial_config(spec).to_token())


@lru_cache(maxsize=256)
@torch.compiler.assume_constant_result
def _decode_bound_config(key):
    return tuple(json.loads(key))


def binding_identity(scores,k,iterations,spec,training,real_tokens=-1):
    """Namespaced scalar guard identity; no input values or CUDA readback."""
    real_tokens=scores.shape[0] if real_tokens<0 else real_tokens
    return (scores.shape[0],scores.shape[1],k,iterations,real_tokens),str(scores.device),'routing:'+spec,training


def bind_selected_config(scores,k,iterations,spec,training,real_tokens,config):
    from ._moe_selection import bind_selected
    bind_selected(*binding_identity(scores,k,iterations,spec,training,real_tokens),
                  json.dumps(config.to_token()))


def publish_selected_config(scores,k,iterations,spec,training,real_tokens,config):
    """Explicit refinement updates future calls, never a saved backward plan."""
    from ._moe_selection import publish_selected
    with _selection_lock:
        _selections[(str(scores.device),tuple(scores.shape),spec,training,real_tokens)]=config
        publish_selected(*binding_identity(scores,k,iterations,spec,training,real_tokens),
                         json.dumps(config.to_token()))


def dispatch_sinkhorn(scores,k,iterations,spec,training,real_tokens=-1):
    """Actual-input cold selection, then static launch-only saved-state execution.

    The returned policy is a CPU tensor only for the cold invocation. Once
    bound it is immutable scalar metadata, consumed directly by hard routing.
    """
    mode,config=_static_policy(spec)
    if mode!='off':
        from ._moe_selection import bound_config
        key=bound_config(*binding_identity(scores,k,iterations,spec,training,real_tokens))
        if key is None:
            out,_,token=selected_sinkhorn(scores,k,iterations,spec,training,real_tokens)
            return out,token
        config=_decode_bound_config(key)
    from ._routing_execution import sinkhorn
    out,_=sinkhorn(scores,k,iterations,config)
    return out,config


def _resolve(scores,spec_key,training,real_tokens=-1):
    from .routing import RoutingKernelConfig
    from .autotune.routing import cache_contract,tune_routing
    from .autotune.winners import load_runtime_winner,runtime_tuning_lock
    base=parse_spec(spec_key)
    spec={**base,'workload':{**base['workload'],'real_tokens':scores.shape[0] if real_tokens<0 else real_tokens}}
    initial=initial_config(spec_key)
    if spec['mode']=='off':return initial
    identity=(str(scores.device),tuple(scores.shape),spec_key,training,real_tokens)
    with _selection_lock,torch.cuda.device(scores.device):
        cached=_selections.get(identity)
        if cached is not None:return cached
        if torch.cuda.is_current_stream_capturing():raise RuntimeError('Warm routing first-hit selection before manual graph capture')
        key,root,policy=cache_contract(scores,spec,'train' if training else 'forward')
        def load():
            return load_runtime_winner(key,RoutingKernelConfig.from_dict,root=root,
                tuning_policy=policy if spec['mode']=='balanced' else None)
        selected=load()
        _LOG.info('cache_decision shape=%s real_tokens=%s training=%s hit=%s mode=%s',
                  tuple(scores.shape),real_tokens,training,selected is not None,spec['mode'])
        if selected is None and spec['mode']=='balanced':
            with runtime_tuning_lock(key,root=root):
                selected=load()
                if selected is None:
                    stream=torch.cuda.current_stream(scores.device)
                    def tune():
                        from ._managed import tuning_resources
                        @tuning_resources
                        def execute():
                            with torch.cuda.device(scores.device),torch.cuda.stream(stream),torch.enable_grad():
                                return tune_routing(scores,spec,phase='train' if training else 'forward',
                                                    runtime_fallback=True,progress=_LOG.info)
                        return execute()
                    with ThreadPoolExecutor(max_workers=1,thread_name_prefix='rtx-routing-first-hit') as worker:
                        result=worker.submit(tune).result()
                    selected=result.config
        selected=selected or initial
        _selections[identity]=selected
        return selected


@torch.library.custom_op('rtx_routing::selected_sinkhorn',mutates_args=(),device_types='cuda')
def selected_sinkhorn(scores:torch.Tensor,k:int,iterations:int,spec:str,
                      training:bool,real_tokens:int=-1)->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    from .routing import _validate_sinkhorn
    from ._routing_execution import run_forward
    _validate_sinkhorn(scores,k,iterations)
    if scores.dtype!=torch.float32 or not scores.is_contiguous():raise ValueError('Contiguous FP32 routing scores required')
    if 2*iterations*scores.numel()>=2**31:raise ValueError('Saved routing state exceeds native index range')
    with torch.cuda.device(scores.device):
        config=_resolve(scores,spec,training,real_tokens)
        out,history=run_forward(scores,k,iterations,tuple(config.to_token()))
        # Immutable CPU metadata is retained by autograd, never read from CUDA
        # and never looked up again in a mutable selection cache in backward.
        token=torch.tensor(config.to_token(),device='cpu',dtype=torch.int64)
        bind_selected_config(scores,k,iterations,spec,training,real_tokens,config)
        return out,history,token


@selected_sinkhorn.register_fake
def _selected_fake(scores,k,iterations,spec,training,real_tokens=-1):
    from .routing import ROUTING_TOKEN_SIZE
    return (torch.empty_like(scores),
            torch.empty(2*iterations*scores.numel(),device=scores.device,dtype=scores.dtype),
            torch.empty(ROUTING_TOKEN_SIZE,device='cpu',dtype=torch.int64))


@torch.library.custom_op('rtx_routing::selected_sinkhorn_backward',mutates_args=(),device_types='cuda')
def selected_sinkhorn_backward(gradient:torch.Tensor,history:torch.Tensor,token:torch.Tensor,
                               k:int,iterations:int)->torch.Tensor:
    from ._routing_execution import run_backward
    c=execution_config(token)
    with torch.cuda.device(gradient.device):
        return run_backward(gradient,history,k,iterations,tuple(c.to_token()))


@selected_sinkhorn_backward.register_fake
def _selected_backward_fake(gradient,history,token,k,iterations):return torch.empty_like(gradient)


def _setup(ctx,inputs,output):
    _,ctx.k,ctx.iterations,_,_,_=inputs
    logp,history,token=output
    ctx.mark_non_differentiable(history,token)
    ctx.save_for_backward(logp,history,token)


def _backward(ctx,gradient,_history_gradient,_token_gradient):
    if gradient is None:return (None,)*6
    _,history,token=ctx.saved_tensors
    return selected_sinkhorn_backward(gradient.contiguous(),history,token,ctx.k,ctx.iterations),None,None,None,None,None


selected_sinkhorn.register_autograd(_backward,setup_context=_setup)


@torch.library.custom_op('rtx_routing::selected_hard',mutates_args=(),device_types='cuda')
def selected_hard(scores:torch.Tensor,token:torch.Tensor,k:int,qb_rounds:int,
                  repair_rounds:int,owner_id:int)->torch.Tensor:
    from .routing import RoutingKernelConfig
    from ._routing_hard import hard_schedule
    c=execution_config(token)
    return hard_schedule(scores,k,qb_rounds,repair_rounds,c.segmented_quotas,owner_id,
                         c.quota_threshold,c.qb_backend,c.qb_threads,c.quota_cache_items,c.quota_threads,c.quota_early_exit)


@selected_hard.register_fake
def _selected_hard_fake(scores,token,k,qb_rounds,repair_rounds,owner_id):
    return torch.empty((scores.shape[0],k),device=scores.device,dtype=torch.int64)


def _clear_runtime_caches(*,preserve_compiled=False):
    # A CPU-only choice is not scratch. Keep it when retiring cold candidate
    # buffers, but explicit full cache clearing can force a fresh disk lookup.
    if preserve_compiled:return {}
    from ._moe_selection import clear_selection_cells
    n=len(_selections);_selections.clear()
    cells=clear_selection_cells(namespace='routing:')
    return {'selections':n,'routing_selection_cells':cells}
