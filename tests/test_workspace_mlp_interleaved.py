"""Synthetic checkpoint/teacher integration, not an AFRL timing benchmark."""
import json

import pytest
import torch
from torch.utils.checkpoint import checkpoint

import rtx

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('entries', [8, 16])
def test_interleaved_checkpoint_teacher_workspace(tmp_path, monkeypatch, record_property, entries):
    import rtx.fp8_bwd as b
    from rtx.runtime import BoundedCache, tensor_storage_bytes
    from rtx.autotune.mlp import reference
    torch._dynamo.reset()
    torch.manual_seed(871)
    monkeypatch.setattr(b, '_RUNNERS', BoundedCache(8, sizeof=tensor_storage_bytes))
    monkeypatch.setattr(b, '_MATMUL_RUNNERS', BoundedCache(8, sizeof=tensor_storage_bytes))
    rtx.configure_workspace(rtx.WorkspacePolicy(64<<20, 64<<20,
                                               paired_entries=8, partial_entries=entries))
    cases=[]
    for m in (128,256,384,512,640):
        layer=rtx.MXFP8MLP(128,256,128,device='cuda',autotune='cache',
                          autotune_cache_dir=tmp_path/str(m))
        x=(torch.randn(m,128,device='cuda',dtype=torch.bfloat16)*.1).requires_grad_()
        dy=torch.randn_like(x)*.1
        cases.append((layer,x,dy))
    parameters = [p for layer, _, _ in cases for p in layer.parameters()]
    parameter_ids = tuple(map(id, parameters))
    optimizer = torch.optim.AdamW(parameters, lr=1e-3, foreach=False)
    initial_parameters = [p.detach().clone() for p in parameters]

    def workload(*inputs):
        teachers=[]
        students=[]
        for (layer,_,_),x in zip(cases,inputs):
            with torch.no_grad():
                teachers.append(layer(x))
            students.append(checkpoint(layer,x,use_reentrant=True))
        return tuple(teachers),tuple(students)
    fn=torch.compile(workload,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})

    def step(validate=False):
        for layer,x,dy in cases:
            layer.zero_grad(set_to_none=True)
            x.grad=None
        teachers,students=fn(*(case[1] for case in cases))
        torch.autograd.backward(students,tuple(case[2] for case in cases))
        for (layer,x,dy),teacher,y in zip(cases,teachers,students):
            if validate:
                with torch.no_grad():
                    expected=reference(x,layer.gate_up_weight,layer.down_weight,dy,'silu')
                actual=(y,x.grad,layer.gate_up_weight.grad,layer.down_weight.grad)
                torch.testing.assert_close(teacher,y,rtol=0,atol=0)
                for a,ref in zip(actual,expected):
                    assert bool(a.isfinite().all())
                    assert float((a.detach().float()-ref.float()).norm()/ref.float().norm().clamp_min(1e-12))<.025
            del teacher,y
        optimizer.step()
        assert tuple(id(p) for group in optimizer.param_groups for p in group['params']) == parameter_ids
        for p in parameters:
            assert bool(p.isfinite().all())
            for key in ('exp_avg', 'exp_avg_sq'):
                state = optimizer.state[p][key]
                assert state.dtype == torch.bfloat16
                assert bool(state.isfinite().all())
        for layer,x,_ in cases:
            layer.zero_grad(set_to_none=True)
            x.grad=None
    step(validate=True);step()
    assert all(not torch.equal(p, old) for p, old in zip(parameters, initial_parameters))
    del initial_parameters
    torch.cuda.synchronize()
    before=rtx.workspace_report()
    live=[]
    for _ in range(3):
        with torch.no_grad():
            for layer,x,_ in cases:
                x.add_(.0001)
        step()
        torch.cuda.synchronize()
        live.append(torch.cuda.memory_allocated())
    report=rtx.workspace_report(before)
    record_property('workspace_report',json.dumps(report))
    record_property('live_allocated',json.dumps(live))
    assert len(set(live))==1,live
    partial=report['caches']['partial']['interval_counts']
    if entries==8:
        assert partial['entry_limit_evictions']>0 and partial['insertions']>0
    else:
        assert partial['evictions']==partial['insertions']==0
    assert report['caches']['paired']['interval_counts']['insertions']==0
    assert report['caches']['partial']['resident_bytes']<=64<<20
    step(validate=True)
    torch._dynamo.reset()
