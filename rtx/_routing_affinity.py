"""Compiler-visible alternatives for the router's scalar affine transform."""

from __future__ import annotations

import torch
import torch.nn.functional as F

SCALAR = 'scalar'
ROW = 'row_broadcast'
LAYOUTS = (SCALAR, ROW)
NAMESPACE = 'routing-affinity:'


def log_sigmoid_row_broadcast(
    logits: torch.Tensor,
    logit_scale: torch.Tensor,
    score_bias: torch.Tensor,
) -> torch.Tensor:
    """Apply the same FP32 affine and log-sigmoid with row-wise adjoints.

    Logical [N, 1] views make the two scalar parameter gradients first sum
    over experts per token, then over tokens. This gives the compiler a
    parallel reduction axis even when its flat-scalar heuristic chooses
    one CTA. Neither the complete [N, E] parameter broadcast nor a transpose
    is materialized. Only the association of the scalar sums may differ.
    """
    # A dimensionful FP32 broadcast would upcast an autocast BF16/FP16 dot
    # even though the original FP32 zero-dimensional operands do not.
    if logits.dtype != torch.float32:
        return F.logsigmoid(logits * logit_scale.float().exp() + score_bias.float())
    rows = logits.shape[0]
    scale = logit_scale.float().exp().expand(rows, 1)
    bias = score_bias.float().expand(rows, 1)
    return F.logsigmoid(logits * scale + bias)


def affinity(inputs, epsilon, layout):
    from ._routing_reference import sigmoid_router_log_affinities
    return sigmoid_router_log_affinities(*inputs, rms_epsilon=epsilon,
        _affine=log_sigmoid_row_broadcast if layout==ROW else None)


def grad_mask(inputs):
    return sum((1 << index) for index,value in enumerate(inputs) if value.requires_grad)


def eligible(inputs, mask):
    x=inputs[0]
    return (x.is_cuda and bool(mask & 24)
            and not torch.is_autocast_enabled(x.device.type))


@torch.compiler.assume_constant_result
def _math_context():
    return torch.get_float32_matmul_precision()


@torch.compiler.assume_constant_result
def _policy_key(spec, metadata, precision):
    return NAMESPACE + repr((spec,metadata,precision))


def signature(inputs):
    return tuple((tuple(t.shape),tuple(t.stride()),str(t.dtype)) for t in inputs)


def identity(inputs, spec, mask, precision):
    x,p,e,_,_=inputs
    return ((x.shape[0],x.shape[1],p.shape[0],e.shape[0],1+mask), str(x.device),
            _policy_key(spec,signature(inputs),precision), True)


def _resolve(inputs, spec, mask, precision, *, force=False, progress=None):
    from .autotune.routing_affinity import resolve
    return resolve(inputs,spec,mask,precision,force=force,progress=progress)


@torch.library.custom_op('rtx_routing::prepare_affinity_selection',mutates_args=(),device_types='cuda',
                         tags=(torch.Tag.cudagraph_unsafe,))
def prepare_affinity_selection(x:torch.Tensor, projection:torch.Tensor, experts:torch.Tensor,
                               scale:torch.Tensor, bias:torch.Tensor,
                               spec:str, mask:int, precision:str)->None:
    # Cold orchestration only. The calibrated functions are separately
    # compiled and own no runtime tensors after selection. This marker has
    # no tensor outputs, mutations, numerical work or derivative.
    from ._moe_selection import bind_selected
    inputs=(x,projection,experts,scale,bias)
    selected=_resolve(inputs,spec,mask,precision)
    bind_selected(*identity(inputs,spec,mask,precision),selected)


@prepare_affinity_selection.register_fake
def _prepare_fake(x,projection,experts,scale,bias,spec,mask,precision):
    return None


def _keep_preparation_effect():
    # Torch 2.12/2.13's effect API retains the cold marker across AOT/FX
    # DCE without forcing a copy/data dependency in the numerical affinity.
    from torch._library.effects import EffectType
    prepare_affinity_selection.register_effect(EffectType.ORDERED)


_keep_preparation_effect()


def dispatch(inputs,epsilon,layout,mode,spec):
    if layout!= 'auto':return affinity(inputs,epsilon,layout)
    if mode=='off' or not torch.is_grad_enabled():return affinity(inputs,epsilon,SCALAR)
    mask=grad_mask(inputs)
    if not eligible(inputs,mask):return affinity(inputs,epsilon,SCALAR)
    from ._moe_selection import bind_selected,bound_config
    precision=_math_context()
    key=identity(inputs,spec,mask,precision)
    chosen=bound_config(*key)
    if chosen is not None:return affinity(inputs,epsilon,chosen)
    if not torch.compiler.is_compiling():
        chosen=_resolve(inputs,spec,mask,precision)
        bind_selected(*key,chosen)
        return affinity(inputs,epsilon,chosen)
    # The one cold graph retains the unmodified differentiable scalar
    # computation, not a branch on a CPU token or a scalar copy that can
    # perturb the compiler's normal projection/affine fusion.
    prepare_affinity_selection(*(t.detach() for t in inputs),spec,mask,precision)
    return affinity(inputs,epsilon,SCALAR)


def prepare(inputs,spec,progress=None):
    """Explicitly tune and publish the affinity context, not its transport."""
    from ._moe_selection import publish_selected
    mask=grad_mask(inputs)
    if not eligible(inputs,mask):return SCALAR
    precision=_math_context()
    selected=_resolve(inputs,spec,mask,precision,force=True,progress=progress)
    publish_selected(*identity(inputs,spec,mask,precision),selected)
    return selected


def _clear_runtime_caches(*,preserve_compiled=False):
    if preserve_compiled:return {}
    from ._moe_selection import clear_selection_cells
    return {'affinity_selection_cells':clear_selection_cells(namespace=NAMESPACE)}
