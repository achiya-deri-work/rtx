"""CPU orchestration contracts; these do not validate CUDA numerics."""
import itertools

import pytest
import torch

import rtx.experimental_ms_eden as eden


@pytest.mark.parametrize("dx_policy,dw_policy",list(itertools.product(("ms_eden","mxfp8"),repeat=2)))
@pytest.mark.parametrize("need_x,need_w",[(True,True),(True,False),(False,True)])
def test_independent_policy_dispatch_and_rng(monkeypatch,dx_policy,dw_policy,need_x,need_w):
    import rtx.fp4 as fp4
    import rtx.fp8_bwd as fp8
    calls=[]
    monkeypatch.setattr(fp4,"nvfp4_linear",lambda x,w,**kwargs:x@w.T)
    def fp4_grad(dy,operand,phase,schedule_json):
        calls.append((phase,"ms_eden",schedule_json))
        torch.rand(1)  # Model the explicitly owned stochastic phase.
        return dy@operand if phase=="dx" else dy.T@operand
    monkeypatch.setattr(eden,"_random_gradient",fp4_grad)
    def partial(phase):
        def run(dy,x,w,key):
            calls.append((phase,"mxfp8",None))
            return dy@w if phase=="dx" else dy.T@x
        return run
    monkeypatch.setattr(fp8,"_mxfp8_linear_dx_op",partial("dx"))
    monkeypatch.setattr(fp8,"_mxfp8_linear_dw_op",partial("dw"))
    x=torch.randn(128,128,dtype=torch.bfloat16,requires_grad=need_x)
    w=torch.randn_like(x,requires_grad=need_w)
    # Noncontiguous upstream gradient must be normalized by the wrapper.
    dy=torch.randn(128,256,dtype=torch.bfloat16)[:,::2]
    state=torch.get_rng_state()
    out=eden.ms_eden_linear(x,w,dx_policy=dx_policy,dw_policy=dw_policy,
                           dx_schedule_json="dx",dw_schedule_json="dw")
    out.backward(dy)
    actual_state=torch.get_rng_state()
    torch.set_rng_state(state)
    for needed,policy in ((need_x,dx_policy),(need_w,dw_policy)):
        if needed and policy=="ms_eden":
            torch.rand(1)
    assert torch.equal(actual_state,torch.get_rng_state())
    assert calls==[(phase,policy,phase if policy=="ms_eden" else None)
                   for phase,policy,needed in (("dx",dx_policy,need_x),("dw",dw_policy,need_w)) if needed]
    if need_x:
        torch.testing.assert_close(x.grad,dy@w.detach())
    if need_w:
        torch.testing.assert_close(w.grad,dy.T@x.detach())


@pytest.mark.parametrize("kwargs,match",[
    ({"dx_policy":"fp32"},"dx_policy"),
    ({"dw_policy":"fp32"},"dw_policy"),
    ({"mxfp8_backward_config":object()},"requires an MXFP8"),
    ({"source_policy":"forward_current_roundtrip_bf16","dx_policy":"mxfp8"},"original_bf16"),
    ({"source_policy":"forward_current_roundtrip_bf16","dw_policy":"mxfp8"},"original_bf16"),
])
def test_invalid_policy_is_rejected_before_dispatch(kwargs,match):
    x=torch.empty(128,128,dtype=torch.bfloat16)
    with pytest.raises(ValueError,match=match):
        eden.ms_eden_linear(x,x,**kwargs)


def test_explicit_mxfp8_config_is_forwarded(monkeypatch):
    import rtx.fp4 as fp4
    import rtx.fp8_bwd as fp8
    from rtx.kernels.mxfp8_bwd import DEFAULT_FUSED_MXFP8_BWD_CONFIG
    monkeypatch.setattr(fp4,"nvfp4_linear",lambda x,w,**kwargs:x@w.T)
    seen=[]
    def dx(dy,x,w,key):
        seen.append(key)
        return dy@w
    monkeypatch.setattr(fp8,"_mxfp8_linear_dx_op",dx)
    x=torch.randn(128,128,dtype=torch.bfloat16,requires_grad=True)
    w=torch.randn_like(x)
    eden.ms_eden_linear(x,w,dx_policy="mxfp8",
        mxfp8_backward_config=DEFAULT_FUSED_MXFP8_BWD_CONFIG).sum().backward()
    assert seen==[fp8._intern_bwd_config(DEFAULT_FUSED_MXFP8_BWD_CONFIG)]
