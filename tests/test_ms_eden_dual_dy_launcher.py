"""CPU launch dependency contracts; GPU correctness is separately required."""
from contextlib import contextmanager
from types import SimpleNamespace
import threading
import pytest
import torch
import rtx.experimental_ms_eden as eden
from benchmarks import experiment_ms_eden_dual_dy as experiment


def test_invocation_owned_preparation_descriptor_keeps_order():
    calls=[]
    plan=eden._PreparationLaunchPlan(lambda:calls.append('produce'),lambda:calls.append('finalize'),
        source=object(),signs=object(),packed=object(),temporary=object(),partials=object(),
        source_shape=(128,128),transpose=False,config=None,schedule=None,original_bf16=True)
    plan();assert calls==['produce','finalize']
    plan.finalize();assert calls==['produce','finalize','finalize']


@pytest.mark.parametrize('failure',[None,'fused','dx_b','dw_b','row_finalize','trans_finalize','dx_finish','dw_finish'])
def test_fused_frontier_lifetimes_and_error_joins(monkeypatch,failure):
    events=[];active=['caller']
    class Stream:
        def __init__(self,name):self.name=name
        def wait_stream(self,other):events.append(('wait',self.name,other.name))
    caller=Stream('caller');sides=tuple(Stream(f'side{i}') for i in (1,2,3))
    @contextmanager
    def stream_context(stream):
        previous=active[0];active[0]=stream.name
        try:yield
        finally:active[0]=previous
    def stage(name):
        def run(*args):
            events.append((name,active[0]))
            if failure==name:raise RuntimeError(name)
        return run
    source=SimpleNamespace(data_ptr=lambda:123)
    def operand(transpose):
        return SimpleNamespace(original_bf16=True,transpose=transpose,source=source,
            source_shape=(256,384),config=None,schedule=None,partials=SimpleNamespace(numel=lambda:32),
            signs=None,packed=None,temporary=None,finalize=stage('trans_finalize' if transpose else 'row_finalize'))
    dx=eden._GradientLaunchPlan(operand(False),stage('dx_b'),stage('dx_finish'))
    dw=eden._GradientLaunchPlan(operand(True),stage('dw_b'),stage('dw_finish'))
    monkeypatch.setattr(experiment,'compile_for_schedules',lambda *args:(stage('fused'),(32,32)))
    monkeypatch.setattr(torch.cuda,'current_stream',lambda _:caller)
    monkeypatch.setattr(torch.cuda,'stream',stream_context)
    monkeypatch.setattr(eden,'_operand_streams',lambda *args:(sides,threading.Lock()))
    class Storage:
        def record_stream(self,stream):events.append(('record',stream.name))
    if failure:
        with pytest.raises(RuntimeError,match=failure):experiment.launch_dual_dy(dx,dw,[Storage()],[Storage()],None)
        assert events[-3:]==[('wait','caller',side.name) for side in sides]
    else:
        experiment.launch_dual_dy(dx,dw,[Storage()],[Storage()],None)
        assert events==[
            ('wait','side1','caller'),('wait','side2','caller'),('wait','side3','caller'),
            ('record','side1'),('record','side2'),('record','side3'),('fused','caller'),
            ('wait','side2','caller'),('dx_b','side1'),('dw_b','side3'),
            ('row_finalize','caller'),('trans_finalize','side2'),
            ('wait','caller','side1'),('dx_finish','caller'),('wait','side2','side3'),
            ('dw_finish','side2'),('wait','caller','side2')]
