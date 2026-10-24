"""Offline adapter/fixture contracts; no model download during pytest."""
import torch
from torch import nn
import pytest
from types import SimpleNamespace

import rtx
from benchmarks.ettin_regression import backbone_linear, batches, convert, TEXTS, EttinRecomputedMLP, quality_summary
from benchmarks.train_ettin_qualification import gradient_metrics


class TokenizerFixture:
    mask_token_id = 99

    def __call__(self, texts, **kwargs):
        ids = torch.arange(10).expand(len(texts), -1).clone()
        special = torch.zeros_like(ids)
        special[:, (0, 9)] = 1
        return dict(input_ids=ids, attention_mask=torch.ones_like(ids), special_tokens_mask=special)


def test_mask_fixture_is_reproducible_and_preserves_special_tokens():
    first = list(batches(TokenizerFixture(), 1))
    second = list(batches(TokenizerFixture(), 1))
    assert len(first) == len(TEXTS) // 4
    for (a, labels), (b, repeat) in zip(first, second):
        assert torch.equal(labels, repeat) and torch.equal(a['input_ids'], b['input_ids'])
        assert bool(labels[:, (0, 9)].eq(-100).all())
        selected = labels.ne(-100)
        assert bool(selected.any(-1).all())
        assert bool(a['input_ids'][selected].eq(99).all())
        assert bool(labels[selected].ne(99).all())


@pytest.mark.parametrize('mode,cls', [('mxfp8_dynamic', rtx.MXFP8Linear),
                                    ('nvfp4_jit', rtx.NVFP4Linear),
                                    ('nvfp4_block_four_over_six', rtx.NVFP4Linear)])
def test_conversion_preserves_head_and_parameter_identity(mode, cls):
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([nn.Linear(128, 128, bias=False, dtype=torch.bfloat16)])
    model.decoder = nn.Linear(128, 512, bias=True, dtype=torch.bfloat16)
    original_weight = model.model.layers[0].weight
    head = model.decoder
    assert backbone_linear(model.model.layers[0], 'model.layers.0')
    assert not backbone_linear(head, 'decoder')
    assert convert(model, mode) is model
    assert isinstance(model.model.layers[0], cls)
    assert model.model.layers[0].weight is original_weight
    assert model.decoder is head


def test_recomputed_adapter_preserves_gate_order_and_parameter_names(monkeypatch):
    source = nn.Module()
    source.config = SimpleNamespace(hidden_activation="gelu")
    source.drop = nn.Dropout(0)
    source.Wi = nn.Linear(8, 12, bias=False)
    source.Wo = nn.Linear(6, 8, bias=False)
    def independent(x, wi, wo, activation):
        assert activation == "gelu"
        gate, up = torch.nn.functional.linear(x, wi).chunk(2, -1)
        return torch.nn.functional.linear(torch.nn.functional.gelu(gate)*up, wo)
    monkeypatch.setattr(rtx, 'mxfp8_mlp', independent)
    adapter = EttinRecomputedMLP(source)
    assert adapter.Wi is source.Wi and adapter.Wo is source.Wo
    assert list(dict(adapter.named_parameters())) == ['Wi.weight', 'Wo.weight']
    x = torch.randn(2, 3, 8, requires_grad=True)
    expected = independent(x, source.Wi.weight, source.Wo.weight, "gelu")
    actual = adapter(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    assert x.grad is not None and source.Wi.weight.grad is not None
    source.drop.p = .1
    with pytest.raises(ValueError, match='zero dropout'):
        EttinRecomputedMLP(source)


def test_gradient_audit_requires_complete_finite_parameter_coverage():
    model = nn.Linear(3, 2)
    with pytest.raises(AssertionError, match='missing/nonfinite'):
        gradient_metrics(model)
    model(torch.ones(2, 3)).sum().backward()
    reference, _ = gradient_metrics(model)
    _, metrics = gradient_metrics(model, reference)
    assert len(metrics) == 2
    assert all(m['relative_l2'] == 0 for m in metrics.values())
    model.weight.grad[0, 0] = torch.nan
    with pytest.raises(AssertionError, match='missing/nonfinite'):
        gradient_metrics(model)


def test_quality_guard_rejects_nonfinite_drift_and_insufficient_evidence():
    row = dict(masked_tokens=1001, masked_nll=2.01, reference_nll=2.,
               hidden_mean_cosine=.99, finite=True)
    assert quality_summary([row], 'mxfp8_dynamic')['passed']
    for change in (dict(finite=False), dict(masked_tokens=10), dict(masked_nll=2.2),
                   dict(hidden_mean_cosine=.9)):
        assert not quality_summary([{**row, **change}], 'mxfp8_dynamic')['passed']
