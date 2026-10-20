import json

import pytest
import torch
import rtx


@pytest.fixture(autouse=True)
def isolated_compiler_cache():
    # This test intentionally compiles valid and failing shape/phase/spec
    # variants. Do not consume the shared nn.Module.forward specialization
    # budget of unrelated test modules.
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def test_public_export_and_strict_mode_preserve_parameters(tmp_path):
    layer = rtx.MXFP8MLP(128, 128, device='cpu', autotune='balanced',
                          autotune_cache_dir=tmp_path/'cache')
    ids = [id(p) for p in layer.parameters()]
    path = tmp_path/'spec.json'
    spec = layer.export_tuning_spec(path)
    assert json.loads(path.read_text()) == spec
    spec['mode'] = 'broken'
    assert layer.export_tuning_spec()['mode'] == 'balanced'
    with pytest.raises(FileExistsError):
        layer.export_tuning_spec(path)
    assert layer.require_cached_() is layer
    assert layer.export_tuning_spec()['cache_miss'] == 'error'
    assert layer.export_tuning_spec()['mode'] == 'cache'
    assert ids == [id(p) for p in layer.parameters()]


def test_export_requires_explicit_cache_and_automatic_policy():
    for mode in ('off', 'balanced'):
        layer = rtx.MXFP8MLP(128, 128, device='cpu', autotune=mode)
        with pytest.raises(ValueError):
            layer.export_tuning_spec()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_compiled_strict_cache_reuses_winner_and_rejects_new_shape(tmp_path, monkeypatch):
    import rtx.autotune.mlp as tuning
    from rtx._mlp_autotune import SELECTIONS
    policy = rtx.AutotunePolicy(max_trials=1, max_seconds=30, warmup=1, samples=3,
        calls_per_sample=1, confirmation_repeats=0, finalist_rounds=0)
    layer = rtx.MXFP8MLP(128, 128, device='cuda', autotune='balanced',
        tuning_policy=policy, autotune_cache_dir=tmp_path)
    x = torch.randn(128, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn_like(x)
    compiled = torch.compile(layer, fullgraph=True, dynamic=False,
                              options={'triton.cudagraphs': False})
    y = compiled(x)
    expected = (y.detach(), *torch.autograd.grad(y, (x, *layer.parameters()), dy))
    layer.require_cached_()
    SELECTIONS.clear()
    def forbid(*args, **kwargs):
        raise AssertionError('strict execution attempted tuning')
    monkeypatch.setattr(tuning, 'tune_mxfp8_mlp', forbid)
    strict = torch.compile(layer, fullgraph=True, dynamic=False,
                           options={'triton.cudagraphs': False})
    y = strict(x)
    actual = (y.detach(), *torch.autograd.grad(y, (x, *layer.parameters()), dy))
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match='requires a cached winner'):
        strict(torch.randn(256, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True))
    with torch.no_grad(), pytest.raises(RuntimeError, match='requires a cached winner'):
        strict(x)
