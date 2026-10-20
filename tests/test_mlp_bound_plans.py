"""Phase-specific deployment binding; synthetic qualification is not promotion."""
import json
import subprocess
from dataclasses import replace

import pytest
import torch
import rtx


def test_bound_lookup_is_exact_and_detached():
    from rtx._mlp_autotune import _bound_plan
    spec = json.dumps(dict(device='cuda:0', activation='silu', gate_shape=[256,128],
        down_shape=[128,128], plans=[dict(input_shape=[2,64,128], training=True,
                                        config=rtx.MXFP8MLPConfig().to_dict())]))
    args = [spec, (2,64,128), (256,128), (128,128), 'cuda:0', 'torch.bfloat16', 'silu', True, False]
    key = _bound_plan(*args)
    assert json.loads(key) == rtx.MXFP8MLPConfig().to_dict()
    for index, value, message in [(1,(128,128),'undeclared'), (7,False,'undeclared'),
        (4,'cuda:1','device'), (5,'torch.float32','dtype'), (8,True,'selected_direct'),
        (2,(512,128),'parameter shapes')]:
        wrong = args.copy(); wrong[index] = value
        with pytest.raises(RuntimeError, match=message):
            _bound_plan(*wrong)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_real_qualification_phase_binding_compiled_3d(tmp_path, monkeypatch):
    torch.manual_seed(981)
    torch._dynamo.reset()
    requests = []
    policy = rtx.AutotunePolicy(max_trials=1, max_seconds=30, warmup=1, samples=3,
        calls_per_sample=1, confirmation_repeats=0, finalist_rounds=0)
    for phase in ('train','forward'):
        cfg = rtx.MXFP8MLPConfig()
        if phase == 'forward':
            cfg = replace(cfg, joint_output_storage='separate')
        candidate = rtx.MLPCandidate((128,128,128,128), cfg, phase=phase)
        root = tmp_path/phase
        request = candidate.prepare(root, policy=policy)
        result = subprocess.run(request['command'], capture_output=True, text=True, timeout=240)
        assert result.returncode == 0, (result.stdout, result.stderr)
        requests.append((root,(2,64,128)))
    layer = rtx.MXFP8MLP(128,128,device='cuda')
    identities = [id(p) for p in layer.parameters()]
    with pytest.raises(ValueError, match='duplicate'):
        rtx.bind_qualified_mlp_plans_(layer, [requests[0],requests[0]])
    assert layer._first_hit_spec is None
    report = rtx.bind_qualified_mlp_plans_(layer, requests)
    assert report['status'] == 'preflight_plans_pinned'
    assert not report['application_qualified'] and not report['actual_input_layout_qualified']
    assert identities == [id(p) for p in layer.parameters()]
    import rtx.autotune.mlp as tuning
    import rtx.autotune.winners as winners
    monkeypatch.setattr(tuning, 'tune_mxfp8_mlp', lambda *a,**kw: pytest.fail('binding tuned'))
    monkeypatch.setattr(winners, 'load_runtime_winner', lambda *a,**kw: pytest.fail('binding read disk'))
    from rtx._execution_plan_token import read_plan_token
    import rtx._mlp_autotune as selection
    observed = []
    original = selection._selected_forward
    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        observed.append((args[-1], json.loads(read_plan_token(result[1]))))
        return result
    monkeypatch.setattr(selection, '_selected_forward', record)
    compiled = torch.compile(layer, fullgraph=True, dynamic=False,
                             options={'triton.cudagraphs':False})
    for seed in (981,982):
        torch.manual_seed(seed)
        x = (torch.randn(2,64,128,device='cuda',dtype=torch.bfloat16)*.1).requires_grad_()
        dy = torch.randn_like(x)*.1
        y = compiled(x)
        actual = (y,*torch.autograd.grad(y,(x,*layer.parameters()),dy))
        with torch.no_grad():
            expected = tuning.reference(x.reshape(128,128),*layer.parameters(),dy.reshape(128,128),'silu')
        for a,b in zip(actual,expected):
            a=a.reshape_as(b)
            assert bool(a.isfinite().all()) and bool(b.isfinite().all())
            assert float((a.detach().float()-b.float()).norm()/b.float().norm().clamp_min(1e-12)) < .025
        with torch.no_grad():
            torch.testing.assert_close(compiled(x), y, rtol=0,atol=0)
    plans = json.loads(layer._first_hit_spec)['plans']
    for training, config in observed:
        assert config == next(p['config'] for p in plans if p['training'] == training)
    assert {phase for phase,_ in observed} == {True,False}
    with pytest.raises(RuntimeError, match='undeclared'):
        layer(x.reshape(128,128))
    with pytest.raises(ValueError, match='contiguous'):
        layer(x.transpose(0,1))
    with pytest.raises(RuntimeError, match='selected_direct'):
        selection._select(x,*layer.parameters(),'silu',layer._first_hit_spec,True,replay=True)
    with pytest.raises(RuntimeError,match='cannot bypass'):
        layer.configure_shape_(x)
    for phase in (True,False):
        with torch.set_grad_enabled(phase):
            config, precision = layer.policy_for(x)
            assert precision == 'mxfp8'
            assert config.to_dict() == next(p['config'] for p in plans if p['training'] == phase)
    torch._dynamo.reset()
