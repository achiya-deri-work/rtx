"""CPU geometry contracts and explicitly opted-in same-draw GPU qualification."""
from dataclasses import FrozenInstanceError, asdict, replace
import json
import os

import pytest
import torch

from rtx.autotune.posthoc import (PosthocOperandGeometry, PosthocPreparationSchedule,
    make_posthoc_preparation_adapter, preparation_schedule_features)


def test_partial_geometry_inheritance_immutability_and_roundtrip():
    shared=PosthocPreparationSchedule(producer_blocks=256,finalizer_blocks=224)
    config=replace(shared,a_geometry={"producer_blocks":64},
        b_geometry=PosthocOperandGeometry(finalizer_threads=256))
    assert config.operand_schedule("a")==replace(shared,producer_blocks=64)
    assert config.operand_schedule("b")==replace(shared,finalizer_threads=256)
    assert PosthocPreparationSchedule(**json.loads(json.dumps(asdict(config))))==config
    assert shared.operand_schedule("a")==shared.operand_schedule("b")==shared
    with pytest.raises(FrozenInstanceError):
        config.a_geometry.producer_blocks=32
    with pytest.raises(ValueError):
        config.operand_schedule("weight")


@pytest.mark.parametrize("fields",[dict(producer_threads=True),dict(finalizer_threads=32),
    dict(producer_blocks=0),dict(finalizer_blocks=False)])
def test_invalid_geometry_is_rejected(fields):
    with pytest.raises(ValueError):
        PosthocPreparationSchedule(a_geometry=fields)


def test_effective_geometry_identity_features_and_axes():
    adapter=make_posthoc_preparation_adapter((4096,512,512),"dx","mma128",lambda _:None)
    shared=adapter.initial_config
    split=replace(shared,b_geometry={"producer_threads":64,"producer_blocks":32,
        "finalizer_threads":256,"finalizer_blocks":16})
    assert adapter.rejection(split) is None
    assert adapter.config_id(shared)!=adapter.config_id(split)
    assert adapter.config_id(shared)==adapter.config_id(replace(shared,a_geometry={}))
    # Different representations of exactly the same effective launches alias.
    equivalent=replace(split,producer_blocks=32,
        a_geometry={"producer_blocks":shared.producer_blocks})
    assert adapter.config_id(split)==adapter.config_id(equivalent)
    assert adapter.deserialize(adapter.serialize(split))==split
    f=preparation_schedule_features((4096,512,512),"dx",split)
    old=preparation_schedule_features((4096,512,512),"dx",shared)
    assert f["a_producer_ctas"]==old["a_producer_ctas"]
    assert f["b_producer_ctas"]==32 and f["b_producer_partial_count"]==64
    assert f["b_finalizer_ctas"]==16 and f["b_finalizer_threads"]==256
    assert f["preparation_scratch_bytes"]<old["preparation_scratch_bytes"]
    for name in ("a_geometry","b_geometry"):
        assert name in adapter.axes
        assert all(adapter.rejection(adapter.update_fn(shared,name,value)) is None
                   for value in adapter.axes[name])


@pytest.mark.skipif(os.environ.get("RTX_RUN_PAIRED_CUDA")!="1",
    reason="explicit assigned-GPU qualification opt-in")
@pytest.mark.parametrize("phase",["dx","dw"])
@pytest.mark.parametrize("amax",["inline","staged"])
def test_same_draw_independent_geometry_payloads_gradients_and_graph(phase,amax):
    """Compare exact payload/scales/factors and complete gradient, not tolerances.

    Same-draw scheduling equivalence is separate from the existing independent
    FP32 algorithm-error oracle. Run only on the explicitly assigned idle GPU.
    """
    from benchmarks.study_ms_eden_backward import prepare
    from rtx.experimental_ms_eden import ms_eden_gradient
    torch.manual_seed(948)
    dy=torch.randn(512,256,device="cuda",dtype=torch.bfloat16)
    operand=torch.randn(256 if phase=="dx" else 512,128,device="cuda",dtype=torch.bfloat16)
    signs=torch.empty(128,device="cuda").random_(0,2).mul_(2).sub_(1)
    base=PosthocPreparationSchedule(producer_blocks=128,producer_load="coalesced_wide",
        producer_order="row_major" if phase=="dx" else "reduction_major",
        producer_signs="register",amax_reduction=amax)
    split=replace(base,a_geometry={"producer_threads":64,"producer_blocks":32,
        "finalizer_threads":256,"finalizer_blocks":16},
        b_geometry={"producer_threads":256,"producer_blocks":64,
        "finalizer_threads":64,"finalizer_blocks":32})
    prepared=[]
    for name,source,transpose in (("a",dy,phase=="dw"),("b",operand,True)):
        expected=prepare(source,transpose,"mma128",base,operand=name)
        actual=prepare(source,transpose,"mma128",split,operand=name)
        actual["uniforms"].copy_(expected["uniforms"])
        prepared.append((expected,actual))
    ua,ub=(pair[0]["uniforms"] for pair in prepared)
    def gradient(config):
        return ms_eden_gradient(dy,operand,signs,ua,ub,phase=phase,
            schedule_json=json.dumps(asdict(config)))
    actual_gradient=gradient(split)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured=gradient(split)
    for distribution in ("gaussian","sparse","zero"):
        dy.normal_(0,.25); operand.normal_(0,.5)
        if distribution=="sparse":
            dy[::2].zero_(); operand[:,::2].zero_()
        elif distribution=="zero":
            dy.zero_(); operand.zero_()
        signs.random_(0,2).mul_(2).sub_(1)
        ua.uniform_(); ub.uniform_()
        for expected,actual in prepared:
            actual["uniforms"].copy_(expected["uniforms"])
            expected["run"](signs); actual["run"](signs)
            for key in ("packed","scales","factor"):
                assert torch.equal(expected[key].view(torch.uint8),actual[key].view(torch.uint8)), key
        expected_gradient=gradient(base)
        actual_gradient=gradient(split)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.isfinite(expected_gradient).all()
        assert torch.equal(expected_gradient,actual_gradient)
        assert torch.equal(expected_gradient,captured)
