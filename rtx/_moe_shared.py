"""MoE-local first-hit binding for the ordinary shared MXFP8 MLP.

The first call owns a self-contained CPU execution token. Subsequent calls
use the existing static CUDA-only MLP operation, with no private graph replay
or mutable policy lookup in backward. The general MLP API is unchanged.
"""
import torch


def binding_identity(x,gu,down,activation,spec,training):
    shape=tuple(x.shape)+tuple(gu.shape)+tuple(down.shape)
    return shape,str(x.device),'moe_shared:'+activation+':'+spec,training


def bind_shared_config(x,gu,down,activation,spec,training,key):
    from ._moe_selection import bind_selected
    bind_selected(*binding_identity(x,gu,down,activation,spec,training),key)


@torch.library.custom_op('rtx::moe_shared_select',mutates_args=(),device_types='cuda',
                         tags=(torch.Tag.cudagraph_unsafe,))
def _cold_shared(x:torch.Tensor,gu:torch.Tensor,down:torch.Tensor,
                 activation:str,spec:str,training:bool)->tuple[torch.Tensor,torch.Tensor]:
    from ._mlp_autotune import _selected_forward
    from ._execution_plan_token import read_plan_token
    result=_selected_forward(x,gu,down,activation,spec,training,replay=False)
    bind_shared_config(x,gu,down,activation,spec,training,read_plan_token(result[1]))
    return result


@_cold_shared.register_fake
def _cold_shared_fake(x,gu,down,activation,spec,training):
    from ._execution_plan_token import PLAN_TOKEN_BYTES
    return (torch.empty((*x.shape[:-1],down.shape[0]),device=x.device,dtype=x.dtype),
            torch.empty(PLAN_TOKEN_BYTES,device='cpu',dtype=torch.uint8))


@torch.library.custom_op('rtx::moe_shared_selected_backward',mutates_args=(),device_types='cuda',
    tags=(torch.Tag.cudagraph_unsafe,),
    schema='(Tensor x, Tensor gu, Tensor down, Tensor dy, Tensor saved, str activation, bool need_x, bool need_gu, bool need_down) -> (Tensor?, Tensor?, Tensor?)')
def _cold_shared_backward(x:torch.Tensor,gu:torch.Tensor,down:torch.Tensor,dy:torch.Tensor,
                          saved:torch.Tensor,activation:str,need_x:bool,need_gu:bool,need_down:bool
                          )->tuple[torch.Tensor|None,torch.Tensor|None,torch.Tensor|None]:
    from ._mlp_autotune import _selected_backward
    # Do not call selected_backward itself: its global Inductor lowering may
    # request a private replay. MoE uses only its caller's outer graph policy.
    return _selected_backward(x,gu,down,dy,saved,activation,need_x,need_gu,need_down,replay=False)


@_cold_shared_backward.register_fake
def _cold_shared_backward_fake(x,gu,down,dy,saved,activation,need_x,need_gu,need_down):
    return tuple(torch.empty_like(t) if need else None
                 for t,need in zip((x,gu,down),(need_x,need_gu,need_down)))


def _setup(ctx,inputs,output):
    from ._mlp_autotune import _setup as existing_setup
    existing_setup(ctx,inputs,output)


def _backward(ctx,dy,_unused):
    x,gu,down,saved=ctx.saved_tensors
    grads=_cold_shared_backward(x,gu,down,dy,saved,ctx.activation,*ctx.needs_input_grad[:3])
    return (*grads,None,None,None)


_cold_shared.register_autograd(_backward,setup_context=_setup)


def selected_shared(x,gu,down,activation,spec,training):
    from ._moe_selection import bound_config
    key=bound_config(*binding_identity(x,gu,down,activation,spec,training))
    if key is None:return _cold_shared(x,gu,down,activation,spec,training)[0]
    from .mlp import _static_mlp
    return _static_mlp(x,gu,down,activation,key)
