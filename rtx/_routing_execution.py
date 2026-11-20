"""Compiler-visible saved-state Sinkhorn storage and launch-only execution.

This changes ownership/orchestration only. The existing CuTe reductions and
normalization Jacobians are unchanged; all history is retained for backward.
"""
from functools import lru_cache
import json
from threading import RLock

import torch

from .runtime import BoundedCache


_COMPILED=BoundedCache(128)
_COMPILE_LOCK=RLock()


@lru_cache(maxsize=256)
@torch.compiler.assume_constant_result
def layout(shape,k,iterations,config,phase):
    """Immutable scalar metadata; no CUDA work or runtime tensor ownership."""
    from .routing import RoutingKernelConfig
    n,e=shape
    if min(n,e,k,iterations)<=0 or k>e or n*k%e:
        raise ValueError('Invalid scheduled Sinkhorn geometry')
    if 2*iterations*n*e>=2**31:
        raise ValueError('Saved Sinkhorn state exceeds native index range')
    c=RoutingKernelConfig.from_token(config)
    if phase not in ('forward','backward'):raise ValueError('Invalid Sinkhorn phase')
    shards=c.flat_column_shards_forward if phase=='forward' else c.flat_column_shards_backward
    if shards*e>=2**31:raise ValueError('Flat column partials exceed native index range')
    if c.flat_row_backend=='warp' and e>128:
        raise ValueError('Flat warp rows currently require E<=128')
    geometry=list(c.to_token()[:6])
    if c.flat_row_backend=='warp':geometry[1]=geometry[4]=1
    key=json.dumps((shape,k,iterations,geometry,phase,shards,c.flat_row_backend))
    sizes=(n*e,2*iterations*n*e,n,e) if phase=='forward' else (n*e,n*e,1)
    if shards:sizes=(*sizes,shards*e)
    return key,sizes


def allocate(scores,sizes):
    return [torch.empty(size,device=scores.device,dtype=torch.float32) for size in sizes]


def compiled_plan(key,device):
    identity=(key,device)
    result=_COMPILED.get(identity)
    if result is not None:return result
    with _COMPILE_LOCK:
        result=_COMPILED.get(identity)
        if result is None:
            with torch.cuda.device(device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError('Warm the selected Sinkhorn plan before manual CUDA graph capture')
            result=_compile_plan(key,device)
            _COMPILED[identity]=result
    return result


def _compile_plan(key,device):
    from .routing import _validate_native_device
    from .runtime import load_kernel_symbol
    fields=json.loads(key)
    if len(fields)==5:shape,k,iterations,config,phase=fields;shards,row_backend=0,'cta'
    else:shape,k,iterations,config,phase,shards,row_backend=fields
    geometry=config[:3] if phase=='forward' else config[3:6]
    with torch.cuda.device(device):
        _validate_native_device(device)
        compile_sinkhorn=load_kernel_symbol('routing_sinkhorn','compile_sinkhorn')
        init,stages,finish,backward=compile_sinkhorn(*shape,k,iterations,*geometry)
        column=None
        if shards:
            column=load_kernel_symbol('routing_flat','compile_flat_shards')(
                *shape,k,iterations,geometry[0],geometry[2],shards,phase)
        row=stages[0] if phase=='forward' else backward[1]
        if row_backend=='warp':
            row=load_kernel_symbol('routing_flat','compile_flat_rows')(*shape,k,iterations,geometry[0],phase)
    if phase=='forward':
        functions=(init,row,column if shards else stages[1],finish)
        lines=['def launch(inputs,buffers):',
               '    source=inputs[0]',
               '    out,history,row,column'+(',partial' if shards else '')+'=buffers',
               '    kernels[0](source,column,row,history,out,0)']
        for iteration in range(iterations):
            lines.extend((f'    kernels[1](source,column,row,history,out,{iteration})',
                          '    kernels[2](source,row,column,history,out,'+('row,partial,' if shards else '')+f'{iteration})'))
        lines.append('    kernels[3](source,column,row,history,out,0)')
    else:
        functions=(column if shards else backward[0],row)
        lines=['def launch(inputs,buffers):',
               '    source,history=inputs',
               '    a,b,unused'+(',partial' if shards else '')+'=buffers']
        for iteration in reversed(range(iterations)):
            lines.extend(('    kernels[0](source,unused,unused,history,a,'+('unused,partial,' if shards else '')+f'{iteration})',
                          f'    kernels[1](a,unused,unused,history,b,{iteration})',
                          '    source=b'))
    namespace={'kernels':tuple(functions)}
    exec(compile('\n'.join(lines),'<rtx-sinkhorn-launch-plan>','exec'),namespace)
    launch=namespace['launch']
    launch.__rtx_resources__=tuple(functions)
    return launch


@torch.library.custom_op('rtx_routing::sinkhorn_execute',mutates_args=('buffers',),device_types='cuda')
def sinkhorn_execute(inputs:list[torch.Tensor],buffers:list[torch.Tensor],key:str)->None:
    compiled_plan(key,inputs[0].device.index)(inputs,buffers)


@sinkhorn_execute.register_fake
def _execute_fake(inputs,buffers,key):
    return None


def run_forward(scores,k,iterations,config):
    key,sizes=layout(tuple(scores.shape),k,iterations,config,'forward')
    buffers=allocate(scores,sizes)
    sinkhorn_execute([scores.view(-1)],buffers,key)
    return buffers[0].view(scores.shape),buffers[1]


def run_backward(gradient,history,k,iterations,config):
    key,sizes=layout(tuple(gradient.shape),k,iterations,config,'backward')
    buffers=allocate(gradient,sizes)
    sinkhorn_execute([gradient.contiguous().view(-1),history],buffers,key)
    return buffers[1].view(gradient.shape)


class SavedSinkhorn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,scores,k,iterations,config):
        out,history=run_forward(scores,k,iterations,config)
        ctx.k,ctx.iterations,ctx.config=k,iterations,config
        # Preserve the documented saved-state contract, including the final
        # transport. Backward's Jacobian calculation consumes the history.
        ctx.save_for_backward(out,history)
        ctx.mark_non_differentiable(history)
        return out,history

    @staticmethod
    def backward(ctx,gradient,_history_gradient):
        if gradient is None:return (None,)*4
        return run_backward(gradient,ctx.saved_tensors[1],ctx.k,ctx.iterations,ctx.config),None,None,None


def sinkhorn(scores,k,iterations,config):
    """Traceable allocation/autograd wrapper; config is an immutable int tuple."""
    if scores.dtype!=torch.float32 or scores.ndim!=2 or not scores.is_contiguous():
        raise ValueError('Native Sinkhorn needs contiguous FP32 [N,E] scores')
    return SavedSinkhorn.apply(scores,k,iterations,config)


def _clear_runtime_caches(*,preserve_compiled=False):
    if preserve_compiled:return {}
    with _COMPILE_LOCK:
        count=len(_COMPILED);_COMPILED.clear()
    return {'sinkhorn_compiled_plans':count}
