"""Caller RNG ownership/checkpoint contracts for opt-in Philox backward."""
import json

import pytest
import torch

import rtx.experimental_ms_eden as eden


def test_traceable_random_caller_uses_only_small_seed_and_sign_buffers(monkeypatch):
    seen=[]
    def seeded(dy,operand,signs,seed,**kwargs):
        seen.append((signs.clone(),seed.clone(),kwargs))
        return dy@operand
    monkeypatch.setattr(eden,"ms_eden_seeded_gradient",seeded)
    dy=torch.zeros(128,256,dtype=torch.bfloat16)
    weight=torch.zeros(256,128,dtype=torch.bfloat16)
    torch.manual_seed(817)
    state=torch.get_rng_state()
    eden._random_gradient(dy,weight,"dx",'{"rng_mode":"philox"}')
    signs,seed,_=seen.pop()
    assert signs.shape==(128,) and signs.dtype==torch.float32
    assert seed.shape==(2,) and seed.dtype==torch.int64
    assert ((seed>=0)&(seed<2**32)).all()
    after=torch.get_rng_state()
    torch.set_rng_state(state)
    expected_signs=torch.where(torch.rand(128)<.5,-1.,1.)
    expected_seed=torch.randint(0,2**32,(2,),dtype=torch.int64)
    torch.testing.assert_close(signs,expected_signs,rtol=0,atol=0)
    torch.testing.assert_close(seed,expected_seed,rtol=0,atol=0)
    assert torch.equal(after,torch.get_rng_state())


def test_packed_philox_rejected_before_rng_or_forward(monkeypatch):
    x=torch.zeros(128,128,dtype=torch.bfloat16)
    state=torch.get_rng_state()
    with pytest.raises(ValueError,match="original_bf16"):
        eden._random_gradient(x,x,"dx",'{"rng_mode":"philox"}',packed=(x,x))
    with pytest.raises(ValueError,match="original_bf16"):
        eden.ms_eden_linear(x,x,source_policy="forward_current_roundtrip_bf16",
                           schedule_json='{"rng_mode":"philox"}')
    assert torch.equal(state,torch.get_rng_state())


def test_explicit_uniform_contract_does_not_silently_ignore_philox():
    x=torch.zeros(128,128,dtype=torch.bfloat16)
    signs=torch.ones(128)
    u=torch.zeros(x.numel()//16)
    with pytest.raises(ValueError,match="ms_eden_seeded_gradient"):
        eden._contract(x,x,signs,u,u,"dx",'{"rng_mode":"philox"}')
    seed=torch.zeros(2,dtype=torch.int64)
    _,shape=eden._contract(x,x,signs,seed,seed,"dx","{}",seeded=True)
    assert shape==(128,128,128)


cuda=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")


@cuda
@pytest.mark.parametrize("packet_size",[1,4])
def test_philox_linear_checkpoint_rng_restore(packet_size):
    from torch.utils.checkpoint import checkpoint
    torch.manual_seed(817)
    x=(torch.randn(128,128,device="cuda",dtype=torch.bfloat16)*.25).requires_grad_()
    weight=(torch.randn_like(x)*.1).requires_grad_()
    dy=torch.randn(128,256,device="cuda",dtype=torch.bfloat16)[:,::2]
    schedule=json.dumps(dict(rng_mode="philox",philox_packet_size=packet_size,
                             producer_load="coalesced_wide",producer_signs="register"))
    def linear(a,b):
        return eden.ms_eden_linear(a,b,schedule_json=schedule,forward_scaling="block")
    # Warm compilation outside the RNG checkpoint being tested.
    linear(x,weight).backward(dy)
    x.grad=weight.grad=None
    rng=torch.cuda.get_rng_state()
    def step(recompute):
        torch.cuda.set_rng_state(rng)
        y=checkpoint(linear,x,weight,use_reentrant=True) if recompute else linear(x,weight)
        y.backward(dy)
        result=(y.detach().clone(),x.grad.clone(),weight.grad.clone(),torch.cuda.get_rng_state())
        x.grad=weight.grad=None
        return result
    eager,recomputed=step(False),step(True)
    for a,b in zip(eager,recomputed):
        torch.testing.assert_close(a,b,rtol=0,atol=0)


@cuda
def test_philox_compiled_linear_fresh_rng_and_seed_restore():
    torch.manual_seed(818)
    x=(torch.randn(128,128,device="cuda",dtype=torch.bfloat16)*.25).requires_grad_()
    weight=(torch.randn_like(x)*.1).requires_grad_()
    dy=torch.randn_like(x)
    schedule=json.dumps(dict(rng_mode="philox",producer_signs="register"))
    torch.compiler.reset()
    fn=torch.compile(lambda a,b:eden.ms_eden_linear(a,b,schedule_json=schedule,
        forward_scaling="block"),fullgraph=True,dynamic=False,options={"triton.cudagraphs":True})
    def step():
        torch.compiler.cudagraph_mark_step_begin()
        x.grad=weight.grad=None
        y=fn(x,weight)
        y.backward(dy)
        return x.grad.clone(),weight.grad.clone()
    for _ in range(4): step()
    state=torch.cuda.get_rng_state()
    first=step()
    second=step()
    assert any(not torch.equal(a,b) for a,b in zip(first,second))
    torch.cuda.set_rng_state(state)
    restored=step()
    for a,b in zip(first,restored):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert all(t.isfinite().all() for t in (*first,*second,*restored))


@cuda
@pytest.mark.parametrize("need_x,need_w",[(True,False),(False,True)])
def test_philox_partial_gradients_consume_only_requested_rng(need_x,need_w):
    torch.manual_seed(819)
    x=torch.randn(128,128,device="cuda",dtype=torch.bfloat16,requires_grad=need_x)
    w=torch.randn(128,128,device="cuda",dtype=torch.bfloat16,requires_grad=need_w)
    rng=torch.cuda.get_rng_state()
    eden.ms_eden_linear(x,w,schedule_json='{"rng_mode":"philox"}',forward_scaling="block").sum().backward()
    after=torch.cuda.get_rng_state()
    torch.cuda.set_rng_state(rng)
    torch.rand(128,device="cuda")
    torch.randint(0,2**32,(2,),device="cuda",dtype=torch.int64)
    assert torch.equal(after,torch.cuda.get_rng_state())
    assert (x.grad is not None)==need_x and (w.grad is not None)==need_w
