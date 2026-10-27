"""Four-operand preparation scheduling; CPU topology is not GPU qualification."""
from contextlib import contextmanager
import json
import os
import threading
from types import SimpleNamespace

import pytest
import torch
import rtx.experimental_ms_eden as eden


@pytest.mark.parametrize("failure",[None,"dx_a","dx_b","dw_a","dw_b","dx_finish","dw_finish","record"])
def test_operand_dependency_topology_lifetimes_and_exception_joins(monkeypatch,failure):
    events=[];active=["caller"]
    class Stream:
        def __init__(self,name):self.name=name
        def wait_stream(self,other):events.append(("wait",self.name,other.name))
    caller=Stream("caller");sides=tuple(Stream(f"side{i}")for i in range(1,4))
    @contextmanager
    def context(stream):
        previous=active[0];active[0]=stream.name
        try:yield
        finally:active[0]=previous
    class Storage:
        def __init__(self,name):self.name=name
        def record_stream(self,stream):
            events.append(("record",self.name,stream.name))
            if failure=="record":raise RuntimeError("record")
    def stage(name):
        def launch():
            events.append((name,active[0]))
            if failure==name:raise RuntimeError(name)
        return launch
    dx=eden._GradientLaunchPlan(stage("dx_a"),stage("dx_b"),stage("dx_finish"))
    dw=eden._GradientLaunchPlan(stage("dw_a"),stage("dw_b"),stage("dw_finish"))
    monkeypatch.setattr(torch.cuda,"current_stream",lambda _:caller)
    monkeypatch.setattr(torch.cuda,"stream",context)
    monkeypatch.setattr(eden,"_operand_streams",lambda *args:(sides,threading.Lock()))
    if failure:
        with pytest.raises(RuntimeError,match=failure):
            eden._launch_operand_paired(dx,dw,[Storage("dx")],[Storage("dw")],None)
        assert events[-3:]==[("wait","caller",side.name)for side in sides]
    else:
        eden._launch_operand_paired(dx,dw,[Storage("dx")],[Storage("dw")],None)
        assert events==[
            ("wait","side1","caller"),("wait","side2","caller"),("wait","side3","caller"),
            ("record","dx","side1"),("record","dw","side2"),("record","dw","side3"),
            ("dx_b","side1"),("dw_a","side2"),("dw_b","side3"),("dx_a","caller"),
            ("wait","caller","side1"),("dx_finish","caller"),("wait","side2","side3"),
            ("dw_finish","side2"),("wait","caller","side2")]


def test_operand_capacity_fallback_and_capture_cache_isolation(monkeypatch):
    monkeypatch.setattr(eden,"_OPERAND_STREAMS",{})
    monkeypatch.setattr(eden,"_OPERAND_STREAMS_PER_DEVICE",1)
    monkeypatch.setattr(torch.cuda,"is_current_stream_capturing",lambda:False)
    created=[]
    def stream(device):created.append(object());return created[-1]
    monkeypatch.setattr(torch.cuda,"Stream",stream)
    device=torch.device("cuda",0)
    first=eden._operand_streams(device,SimpleNamespace(cuda_stream=10))
    assert len(first[0])==3 and len(set(first[0]))==3
    assert eden._operand_streams(device,SimpleNamespace(cuda_stream=10)) is first
    assert eden._operand_streams(device,SimpleNamespace(cuda_stream=11)) is None
    monkeypatch.setattr(torch.cuda,"is_current_stream_capturing",lambda:True)
    assert eden._operand_streams(device,SimpleNamespace(cuda_stream=10)) is first
    with pytest.raises(RuntimeError,match="warm operand_paired"):
        eden._operand_streams(torch.device("cuda",1),SimpleNamespace(cuda_stream=10))
    monkeypatch.setattr(torch.cuda,"current_stream",lambda _:object())
    monkeypatch.setattr(eden,"_operand_streams",lambda *args:None)
    seen=[]
    def plan(phase):return eden._GradientLaunchPlan(*(lambda name=f"{phase}{part}":seen.append(name)for part in "abf"))
    eden._launch_operand_paired(plan("dx"),plan("dw"),[],[],None)
    assert seen==["dxa","dxb","dxf","dwa","dwb","dwf"] and len(created)==3


def test_pipeline_adapter_cli_and_preset_admit_operand_mode():
    from benchmarks import compare_ms_eden_pipeline as compare,tune_ms_eden_pipeline as tuner
    from benchmarks import train_precision20m as training
    action = next(action for action in training.parser()._actions
                  if action.dest == "ms_eden_backward_pipeline")
    assert "operand_paired" in action.choices
    assert action.default == "serial"
    from rtx.autotune.ms_eden_pipeline import make_ms_eden_pipeline_adapter
    adapter=make_ms_eden_pipeline_adapter((512,512,512),lambda _:None)
    config=adapter.update_fn(adapter.initial_config,"backward_pipeline","operand_paired")
    assert adapter.rejection(config) is None
    assert adapter.deserialize(adapter.serialize(config))==config
    assert adapter.config_id(config)!=adapter.config_id(adapter.initial_config)
    assert adapter.extra_features_fn(config)["pipeline_side_streams"]==3
    assert tuner.preset("operand_paired").backward_pipeline=="operand_paired"
    assert compare.parser().parse_args(["--candidate-pipeline","operand_paired","--output","unused"]).candidate_pipeline=="operand_paired"


run_cuda=pytest.mark.skipif(os.environ.get("RTX_RUN_OPERAND_PIPELINE_CUDA")!="1",
    reason="explicit assigned idle GPU qualification opt-in")


@run_cuda
def test_cuda_operand_same_draw_compiled_graph_opcheck():
    from tests.test_ms_eden_paired_pipeline import inputs
    args=list(inputs("cuda"))
    for value in args[:3]:value.normal_()
    for index in (3,6):args[index].fill_(1)
    for index in (4,5,7,8):args[index].uniform_()
    expected=eden.ms_eden_paired_gradient(*args,pipeline="serial")
    actual=eden.ms_eden_paired_gradient(*args,pipeline="operand_paired")
    for a,b in zip(expected,actual):assert torch.equal(a.view(torch.uint8),b.view(torch.uint8))
    compiled=torch.compile(lambda *v:eden.ms_eden_paired_gradient(*v,pipeline="operand_paired"),fullgraph=True)
    compiled(*args)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):compiled(*args)
    stream.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):captured=compiled(*args)
    for value in args[:3]:value.normal_()
    for index in (3,6):args[index].neg_()
    for index in (4,5,7,8):args[index].uniform_()
    expected=eden.ms_eden_paired_gradient(*args,pipeline="serial")
    graph.replay();torch.cuda.synchronize()
    for a,b in zip(expected,captured):assert torch.equal(a.view(torch.uint8),b.view(torch.uint8))
    torch.library.opcheck(eden._paired_gradient,(*args,"{}","{}",False,False,"operand_paired"))


@run_cuda
def test_public_operand_forward_current_rng_memory_guards():
    from benchmarks.compare_ms_eden_pipeline import options,schedule_pair
    from benchmarks.compare_mixed_nvfp4_backward import run_worker
    record,bundle=run_worker(options((512,512,512),schedule_pair(),pipeline="operand_paired"),capture_only=True)
    assert record["guards"]["forward_bitwise_matched"]
    assert record["guards"]["fresh_rng_per_phase"]==[True,True]
    assert record["memory"]["ordinary_compiled_allocation_audit"]["live_growth_bytes"]<=0
    assert record["memory"]["replay_live_growth_bytes"]==0
    assert all(torch.isfinite(value).all()for value in bundle["grads"])
