"""CPU topology tests only; CUDA qualification runs in the research driver."""
from contextlib import contextmanager
import itertools
import json
import os
import threading

import pytest
import torch
import rtx.experimental_ms_eden as eden
from benchmarks import experiment_ms_eden_operand_order as experiment


ORDERS=list(itertools.permutations(experiment.OPERANDS))


def test_order_cannot_silently_replace_dependency_frontier(monkeypatch):
    import sys
    from benchmarks import experiment_ms_eden_dependency as dependency
    monkeypatch.setattr(sys, 'argv', ['probe', '--selection', 'unused.json',
        '--output', 'unused-result.json', '--frontier', 'producer',
        '--operand-order', 'dx_a,dw_a,dx_b,dw_b'])
    with pytest.raises(SystemExit) as error:
        dependency.main()
    assert error.value.code == 2


@pytest.mark.parametrize('flags', [
    ['--scale-finalizer','dual_dx','--finish-admission','dx_a'],
    ['--scale-finalizer','dual_both','--operand-order','dx_a,dx_b,dw_a,dw_b'],
    ['--scale-finalizer','paired','--rng-mode','philox'],
])
def test_finalizer_composition_rejects_incompatible_controls(monkeypatch,flags):
    import sys
    from benchmarks import experiment_ms_eden_dependency as dependency
    monkeypatch.setattr(sys,'argv',['probe','--selection','unused.json',
        '--output','unused-result.json','--frontier','control',*flags])
    with pytest.raises(SystemExit) as error:
        dependency.main()
    assert error.value.code == 2


@pytest.mark.parametrize("order",ORDERS)
@pytest.mark.parametrize("failure",[None,"dx_a","dx_b","dw_a","dw_b","dx_finish","dw_finish","record"])
@pytest.mark.parametrize('finish_admission',['independent','dx_a','dx_ready'])
def test_all_order_topologies_lifetimes_locks_and_failure_joins(monkeypatch,order,failure,finish_admission):
    events=[];active=["caller"];lock=threading.Lock()
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
            assert lock.locked()
            events.append((name,active[0]))
            if failure==name:raise RuntimeError(name)
        return launch
    dx=eden._GradientLaunchPlan(stage("dx_a"),stage("dx_b"),stage("dx_finish"))
    dw=eden._GradientLaunchPlan(stage("dw_a"),stage("dw_b"),stage("dw_finish"))
    monkeypatch.setattr(torch.cuda,"current_stream",lambda _:caller)
    monkeypatch.setattr(torch.cuda,"stream",context)
    monkeypatch.setattr(eden,"_operand_streams",lambda *args:(sides,lock))
    launcher=experiment.ordered_launcher(order,finish_admission)
    if failure:
        with pytest.raises(RuntimeError,match=failure):launcher(dx,dw,[Storage("dx")],[Storage("dw")],None)
        assert events[-3:]==[("wait","caller",side.name)for side in sides]
    else:
        launcher(dx,dw,[Storage("dx")],[Storage("dw")],None)
        streams=dict(zip(experiment.OPERANDS,("caller","side1","side2","side3")))
        assert events==[
            ("wait","side1","caller"),("wait","side2","caller"),("wait","side3","caller"),
            ("record","dx","side1"),("record","dw","side2"),("record","dw","side3"),
            *((name,streams[name])for name in order),
            *([("wait","side2","caller")] if finish_admission=='dx_a' else []),
            ("wait","caller","side1"),
            *([("wait","side2","caller")] if finish_admission=='dx_ready' else []),
            ("dx_finish","caller"),("wait","side2","side3"),
            ("dw_finish","side2"),("wait","caller","side2")]
    assert active[0]=="caller" and not lock.locked()


def test_capacity_serial_fallback_and_patch_restoration(monkeypatch):
    monkeypatch.setattr(torch.cuda,"current_stream",lambda _:None)
    monkeypatch.setattr(eden,"_operand_streams",lambda *a:None)
    seen=[]
    def plan(phase):return eden._GradientLaunchPlan(*(lambda name=phase+part:seen.append(name) for part in "abf"))
    original=eden._launch_operand_paired
    with pytest.raises(RuntimeError,match="probe"):
        with experiment.patched_order(experiment.FIRST_SCREEN[0]):
            assert eden._launch_operand_paired is not original
            eden._launch_operand_paired(plan("dx"),plan("dw"),[],[],None)
            raise RuntimeError("probe")
    assert eden._launch_operand_paired is original
    assert seen==["dxa","dxb","dxf","dwa","dwb","dwf"]


def test_parser_requires_explicit_unique_order_and_balanced_screen():
    parser=experiment.parser()
    for bad in ([],["--order","dx_a,dx_b,dw_a"],["--order","dx_a,dx_a,dw_a,dw_b"]):
        with pytest.raises(SystemExit):parser.parse_args(["--output","unused",*bad])
    args=parser.parse_args(["--output","unused","--order","dx_a,dw_a,dx_b,dw_b"])
    assert args.order==experiment.FIRST_SCREEN[0]
    assert args.candidate_pipeline=="operand_paired" and args.baseline_policy=="stable_mxfp8"
    assert len(set(experiment.FIRST_SCREEN))==8
    assert sum(order[0]=="dx_a" for order in experiment.FIRST_SCREEN)==3
    assert sum(order[0]=="dw_a" for order in experiment.FIRST_SCREEN)==3
    assert all(experiment.order_value(order)==order for order in experiment.FIRST_SCREEN)


def test_main_exact_before_comparison_provenance_config_and_no_env_mutation(monkeypatch,tmp_path):
    seen=[];environment=dict(os.environ);original=eden._launch_operand_paired
    def smoke(shape,schedules,seed):
        assert eden._launch_operand_paired is not original
        seen.append(("smoke",shape,schedules,seed))
        return dict(passed=True)
    def compare(shape,output,candidate,**kwargs):
        assert seen[0][0]=="smoke"
        assert eden._launch_operand_paired is not original
        seen.append(("compare",candidate,kwargs))
        return dict(timings={},source_sha256={})
    monkeypatch.setattr(experiment,"exact_smoke",smoke)
    monkeypatch.setattr(experiment.comparison,"compare",compare)
    path=tmp_path/"result.json"
    args=["--shape","512,256,128","--output",str(path),"--order","dw_a,dx_a,dw_b,dx_b",
        "--candidate-json",'{"producer_stages":2}',"--candidate-dx-json",'{"producer_blocks":52}',
        "--candidate-dw-json",'{"producer_blocks":104}',"--scope","backward","--rounds","4"]
    result=experiment.main(args)
    assert eden._launch_operand_paired is original and dict(os.environ)==environment
    assert json.loads(seen[0][2]["dx"])["producer_blocks"]==52
    assert json.loads(seen[0][2]["dw"])["producer_blocks"]==104
    assert seen[1][2]["baseline_policy"]=="stable_mxfp8" and seen[1][2]["scopes"]==("backward",)
    artifact=json.loads(path.read_text())["operand_order_experiment"]
    assert artifact["preparation_order"]==["dw_a","dx_a","dw_b","dx_b"]
    assert artifact["same_draw_exact"]["passed"]
    assert len(artifact["source_sha256"])==len(artifact["launcher_source_sha256"])==64
    assert result["source_sha256"]["benchmarks/experiment_ms_eden_operand_order.py"]==artifact["source_sha256"]
    with pytest.raises(FileExistsError):experiment.main(args)


def test_failed_exact_gate_never_times_or_leaks_patch(monkeypatch,tmp_path):
    original=eden._launch_operand_paired
    def fail(*a):raise RuntimeError("exact failed")
    monkeypatch.setattr(experiment,"exact_smoke",fail)
    monkeypatch.setattr(experiment.comparison,"compare",lambda *a,**k:pytest.fail("timing before exact gate"))
    output=tmp_path/"failed.json"
    with pytest.raises(RuntimeError,match="exact failed"):
        experiment.main(["--output",str(output),"--order","dx_a,dx_b,dw_a,dw_b"])
    assert eden._launch_operand_paired is original and not output.exists()
