import pytest
import torch

from rtx._moe_execution import layout, plan
from rtx._moe_reference import operational_reference
from rtx._moe_saved import SavedExperts
from rtx.moe import DEFAULT_MOE_CONFIG, _config_key


def test_preexisting_recompute_configs_keep_their_identity():
    from dataclasses import replace
    base = DEFAULT_MOE_CONFIG
    for state in ("recompute", "saved_mx", "saved_bf16_dw"):
        value = replace(base, training_state=state)
        fields = value.to_dict()
        assert ("training_state" in fields) == (state != "recompute")
        assert type(base).from_dict(fields) == value


@pytest.mark.parametrize("vendor", [False, True])
def test_saved_storage_does_not_pin_scratch_and_dw_has_no_expert_loop(vendor):
    shape=(128,128,4,64,2,64)
    config=_config_key(DEFAULT_MOE_CONFIG)
    key,_,_,views,out=layout(shape,config,"saved_forward")
    assert len(out)==8 and all(views[i][2]<0 for i in out)
    assert len(set(out))==len(out)
    b=plan(layout(shape,config,"saved_backward_blas" if vendor else "saved_backward_mx")[0])[0]
    calls=[(name,args) for _,name,_,args in b.steps]
    assert sum(name=="compile_batched_bf16_dw" for name,_ in calls)==2*vendor
    assert sum(name=="compile_dequant_hidden" for name,_ in calls)==1
    assert len(b.outputs)==4


@pytest.mark.parametrize("vendor",[False,True])
def test_saved_family_is_opt_in_and_last_slot_is_vendor_only_when_allowed(vendor):
    from rtx.autotune.effort import AutotunePolicy
    from rtx.autotune.moe import make_moe_adapter, make_moe_tuner
    from rtx.autotune.store import InMemoryTuningStore
    shape=(21504,384,24,4096,4,128)
    policy=AutotunePolicy.preset('quick',allow_saved_moe=True,allow_bf16_dw=vendor)
    tags={'saved_expert_trial':'mx_and_blas' if vendor else 'mx_only'}
    adapter=make_moe_adapter(shape,None,tags=tags)
    tuner=make_moe_tuner(adapter,InMemoryTuningStore(),policy)
    saved=next(p.config for p in tuner.coverage_proposals if p.config.training_state=='saved_mx')
    assert adapter.rejection(saved) is None
    assert make_moe_adapter(shape,None).rejection(saved) is not None
    if vendor:
        last=tuner.final_proposal(saved)
        assert last.config.training_state=='saved_bf16_dw' and adapter.rejection(last.config) is None
    else:
        assert tuner.final_proposal is None


@pytest.mark.parametrize("experts,capacity", [(24,4096),(32,3200)])
def test_packed_b16_budget_actually_dispatches_saved_and_vendor(experts,capacity):
    """Production search policy and plans; synthetic objective, not GPU timing."""
    from rtx.autotune.effort import AutotunePolicy
    from rtx.autotune.moe import make_moe_adapter, make_moe_tuner
    from rtx.autotune.outcomes import TrialOutcome
    from rtx.autotune.store import InMemoryTuningStore
    shape=(21504,384,experts,capacity,4,128)
    policy=AutotunePolicy.preset("quick",max_trials=8,max_seconds=45,
        backward_measurement_mode="cuda_graph",allow_saved_moe=True,allow_bf16_dw=True)
    def evaluate(config):
        state=config.training_state
        if state!="recompute":
            key=_config_key(config)
            layout(shape,key,"saved_forward")
            layout(shape,key,"saved_backward_mx" if state=="saved_mx" else "saved_backward_blas")
        return TrialOutcome("ok",median_ms={"recompute":3.,"saved_mx":2.,"saved_bf16_dw":1.}[state])
    adapter=make_moe_adapter(shape,evaluate,allow_onchip=True,
        device={"multiprocessor_count":170,"shared_memory_per_block_optin":101376},
        tags={"saved_expert_trial":"mx_and_blas","batch_contract":{
            "layout":"independent_varlen","sequences":17,"segment_alignment":32}})
    store=InMemoryTuningStore()
    result=make_moe_tuner(adapter,store,policy).tune()
    records=list(store.records(adapter.context))
    assert len(records)==8
    states=[row["config"].get("training_state","recompute") for row in records[:-1]]
    assert "saved_mx" in states and "saved_bf16_dw" not in states
    assert records[-1]["strategy"]=="final_challenger"
    assert records[-1]["config"]["training_state"]==result.config.training_state=="saved_bf16_dw"


def test_saved_public_forward_fullgraph_fake_cuda():
    from dataclasses import replace
    from torch._subclasses.fake_tensor import FakeTensorMode
    from tests.test_moe_compiler_execution import _fake_operands, _public
    config=replace(DEFAULT_MOE_CONFIG,training_state='saved_bf16_dw',
                   prepared=replace(DEFAULT_MOE_CONFIG.prepared,capacity_multiple=128))
    with FakeTensorMode() as mode:
        x,gu,down,ids,weights,inverse=_fake_operands(mode)
        call=torch.compile(_public(config),backend='eager',fullgraph=True,dynamic=False)
        # Test traceability without launching an autograd engine on a CPU-only host.
        with torch.enable_grad():
            out=call(x,gu.requires_grad_(),down,ids,weights,inverse)
        assert out.shape==x.shape and out.dtype==torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("vendor",[False,True])
def test_saved_gradients_against_mixed_policy_oracle(vendor):
    from tests.test_moe_compiler_execution import _example, _result, _check
    operands,dy=_example((128,128,4,2,64))
    ref=operational_reference(*operands,dy,bf16_dw=vendor)
    key=_config_key(DEFAULT_MOE_CONFIG)
    def call(*args):
        return SavedExperts.apply(*args,key,vendor)
    for fn in (call,torch.compile(call,fullgraph=True,dynamic=False)):
        _check(_result(fn,operands,dy),ref)
