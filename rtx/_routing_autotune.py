"""Opaque first-hit boundary; hot launchers never do eager CUDA tensor math."""
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
    if token.device.type!='cpu' or token.dtype!=torch.int64 or token.shape!=(10,):
        raise ValueError('Routing execution token must be CPU int64[10]; CUDA readback is forbidden')
    return decode_token(tuple(token.tolist()))


@lru_cache(maxsize=256)
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
def parse_spec(value):
    return json.loads(value)


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
    from .routing import _run_sinkhorn_forward,_validate_sinkhorn
    _validate_sinkhorn(scores,k,iterations)
    if scores.dtype!=torch.float32 or not scores.is_contiguous():raise ValueError('Contiguous FP32 routing scores required')
    if 2*iterations*scores.numel()>=2**31:raise ValueError('Saved routing state exceeds native index range')
    with torch.cuda.device(scores.device):
        config=_resolve(scores,spec,training,real_tokens)
        c=config.forward
        out,history=_run_sinkhorn_forward(scores,k,iterations,c.threads,c.row_groups,c.column_groups)
        # Immutable CPU metadata is retained by autograd, never read from CUDA
        # and never looked up again in a mutable selection cache in backward.
        token=torch.tensor(config.to_token(),device='cpu',dtype=torch.int64)
        return out,history,token


@selected_sinkhorn.register_fake
def _selected_fake(scores,k,iterations,spec,training,real_tokens=-1):
    return (torch.empty_like(scores),
            torch.empty(2*iterations*scores.numel(),device=scores.device,dtype=scores.dtype),
            torch.empty(10,device='cpu',dtype=torch.int64))


@torch.library.custom_op('rtx_routing::selected_sinkhorn_backward',mutates_args=(),device_types='cuda')
def selected_sinkhorn_backward(gradient:torch.Tensor,history:torch.Tensor,token:torch.Tensor,
                               k:int,iterations:int)->torch.Tensor:
    from .routing import RoutingKernelConfig,_run_sinkhorn_backward
    c=execution_config(token).backward
    with torch.cuda.device(gradient.device):
        return _run_sinkhorn_backward(gradient,history,k,iterations,c.threads,c.row_groups,c.column_groups)


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
                         c.quota_threshold,c.qb_backend,c.qb_threads)


@selected_hard.register_fake
def _selected_hard_fake(scores,token,k,qb_rounds,repair_rounds,owner_id):
    return torch.empty((scores.shape[0],k),device=scores.device,dtype=torch.int64)


def _clear_runtime_caches(*,preserve_compiled=False):
    # A CPU-only choice is not scratch. Keep it when retiring cold candidate
    # buffers, but explicit full cache clearing can force a fresh disk lookup.
    if preserve_compiled:return {}
    n=len(_selections);_selections.clear()
    return {'selections':n}
