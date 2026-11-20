"""Compiler-owned stable expert metadata with a saved-softmax derivative."""
from functools import lru_cache
from threading import RLock

import torch

from .runtime import BoundedCache


_COMPILED=BoundedCache(128)
_LOCK=RLock()


def rejection(n,e,k,real,threads):
    if min(n,e,k,real)<=0 or real>n or k>e or n*k%e:
        return 'Stable metadata needs positive balanced scheduled dimensions and 1<=real<=N'
    if k>32 or e>128:return 'Stable metadata supports K<=32 and E<=128'
    if threads not in (128,256,512):return 'Metadata threads must be 128/256/512'
    if max(n*e,n*k,e*((n*k+threads-1)//threads))>=2**31:
        return 'Stable metadata exceeds signed native index range'
    return None


@lru_cache(maxsize=256)
@torch.compiler.assume_constant_result
def layout(n,e,k,real,threads):
    reason=rejection(n,e,k,real,threads)
    if reason:raise ValueError(reason)
    chunks=(n*k+threads-1)//threads
    return (n,e,k,real,threads),(e*chunks,n*k,e*chunks)


def compiled(config,device,backward=False):
    identity=(config,device,backward)
    result=_COMPILED.get(identity)
    if result is not None:return result
    with _LOCK:
        result=_COMPILED.get(identity)
        if result is None:
            with torch.cuda.device(device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError('Warm stable metadata before manual CUDA graph capture')
                from .routing import _validate_native_device
                from .runtime import load_kernel_symbol
                _validate_native_device(device)
                factory=load_kernel_symbol('routing_metadata','compile_metadata_backward' if backward else 'compile_metadata_forward')
                result=factory(*config)
                _COMPILED[identity]=result
    return result


@torch.library.custom_op('rtx_routing::metadata_execute',mutates_args=('buffers',),device_types='cuda')
def metadata_execute(logp:torch.Tensor,assignments:torch.Tensor,buffers:list[torch.Tensor],config:list[int])->None:
    for function in compiled(tuple(config),logp.device.index):
        function(logp,assignments,*buffers)


@metadata_execute.register_fake
def _execute_fake(logp,assignments,buffers,config):pass


@torch.library.custom_op('rtx_routing::metadata_backward',mutates_args=('out',),device_types='cuda')
def metadata_backward(gradient:torch.Tensor,weights:torch.Tensor,inverse:torch.Tensor,out:torch.Tensor,
                      config:list[int])->None:
    compiled(tuple(config),out.device.index,True)(gradient,weights,inverse,out)


@metadata_backward.register_fake
def _backward_fake(gradient,weights,inverse,out,config):pass


def run_forward(logp,assignments,real_tokens,threads=128):
    """The same caller-allocated forward objective used by router tuning."""
    n,e=logp.shape;k=assignments.shape[1]
    config,sizes=layout(n,e,k,real_tokens,threads)
    scratch=[torch.empty(size,device=logp.device,dtype=torch.int32) for size in sizes]
    ids=torch.empty((e,n*k//e),device=logp.device,dtype=torch.int64)
    weights=torch.empty((e,n*k//e),device=logp.device,dtype=torch.float32)
    inverse=torch.empty((real_tokens,k),device=logp.device,dtype=torch.int64)
    metadata_execute(logp.view(-1),assignments.view(-1),[*scratch,ids.view(-1),weights.view(-1),inverse.view(-1)],list(config))
    return ids,weights,inverse


def run_backward(dweights,weights,assignments,inverse,*,scheduled_tokens,experts,threads=128):
    """Explicit softmax Jacobian; assignments are recoverable from inverse.

    The assignments argument matches the evaluator's common metadata ABI but
    is not needed or retained by this implementation. Integer slot IDs encode
    their expert through the known fixed expert capacity.
    """
    config,_=layout(scheduled_tokens,experts,inverse.shape[1],inverse.shape[0],threads)
    result=torch.empty((scheduled_tokens,experts),device=weights.device,dtype=torch.float32)
    metadata_backward(dweights.contiguous().view(-1),weights.view(-1),inverse.view(-1),result.view(-1),list(config))
    return result


class Metadata(torch.autograd.Function):
    @staticmethod
    def forward(ctx,logp,assignments,real,threads):
        ids,weights,inverse=run_forward(logp,assignments,real,threads)
        ctx.save_for_backward(weights,inverse);ctx.config=(logp.shape[0],logp.shape[1],threads)
        ctx.mark_non_differentiable(ids,inverse)
        return ids,weights,inverse

    @staticmethod
    def backward(ctx,_ids,gradient,_inverse):
        if gradient is None:return None,None,None,None
        weights,inverse=ctx.saved_tensors;n,e,threads=ctx.config
        result=run_backward(gradient,weights,None,inverse,scheduled_tokens=n,experts=e,threads=threads)
        return result,None,None,None


def routing_metadata(log_transport,expert_assignments,*,num_real_tokens,threads=128):
    if (log_transport.ndim!=2 or log_transport.dtype!=torch.float32 or not log_transport.is_contiguous()
            or expert_assignments.ndim!=2 or expert_assignments.dtype!=torch.int64
            or not expert_assignments.is_contiguous() or expert_assignments.shape[0]!=log_transport.shape[0]
            or expert_assignments.device!=log_transport.device):
        raise ValueError('Stable metadata requires contiguous FP32 [N,E] scores and int64 [N,K] assignments on one device')
    return Metadata.apply(log_transport,expert_assignments,num_real_tokens,threads)


def _clear_runtime_caches(*,preserve_compiled=False):
    if preserve_compiled:return {}
    with _LOCK:
        count=len(_COMPILED);_COMPILED.clear()
    return {'metadata_compiled_plans':count}
