from dataclasses import replace

import pytest
import torch

import rtx
from rtx._bwd_search import bf16_dw_challenger
from rtx.autotune.core import Proposal
from rtx.autotune.outcomes import TrialOutcome
from rtx.autotune.store import InMemoryTuningStore
from rtx.bwd_autotune import bwd_config_from_dict, bwd_config_to_dict
from rtx.kernels.mxfp8 import MXFP8Problem


@pytest.mark.parametrize("terminal_ms", [.25, 10.])
def test_final_trial_is_last_and_counts_towards_the_budget(terminal_ms):
    from rtx.autotune.effort import make_effort_autotuner
    from tests.test_autotune_coverage import Config, adapter

    a = adapter()
    base_evaluate = a.evaluator
    a.evaluator = lambda c: (TrialOutcome("ok", median_ms=terminal_ms)
                              if c.a == 200 else base_evaluate(c))
    store = InMemoryTuningStore()

    def tune(trials):
        tuner = make_effort_autotuner(a, store, rtx.AutotunePolicy(
            max_trials=trials, search_schedule="progressive", confirmation_repeats=0))
        tuner.final_proposal = lambda c: Proposal(replace(c,a=200), "terminal")
        return tuner.tune()

    result = tune(4)
    records = list(store.records(a.context))
    assert len(records) == result.evaluated_trials == 4
    assert [r["strategy"] for r in records].count("final_challenger") == 1
    assert records[-1]["strategy"] == "final_challenger"
    assert records[-1]["parent_config_id"] in {r["config_id"] for r in records[:-1]}
    assert (result.config.a == 200) is (terminal_ms < 1.)
    assert tune(4).evaluated_trials == 0
    assert tune(8).evaluated_trials == 4
    assert len(list(store.records(a.context))) == 8


def test_vendor_config_keeps_dx_and_excludes_joint_quantization():
    from rtx.kernels.mxfp8_bwd import DEFAULT_FUSED_MXFP8_BWD_CONFIG

    mixed = bf16_dw_challenger(DEFAULT_FUSED_MXFP8_BWD_CONFIG)
    assert mixed.dx == DEFAULT_FUSED_MXFP8_BWD_CONFIG.dx
    assert mixed.dw.backend == "cublas_bf16"
    p = MXFP8Problem(217, 29, 33)
    assert mixed.implementation_rejection(p) is None
    assert replace(mixed,execution_order="interleaved").implementation_rejection(p) is not None
    assert replace(mixed,quant_schedule="quad").implementation_rejection(p) is not None
    assert bwd_config_from_dict(bwd_config_to_dict(mixed)) == mixed.normalized()


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="assigned CUDA GPU required")


@cuda
@pytest.mark.parametrize("needs_dx", [True, False])
def test_cublas_dw_linear_fullgraph(needs_dx):
    from rtx.fp8_bwd import _default_bwd_config, _intern_bwd_config, _mxfp8_linear_dw_op

    torch.manual_seed(412)
    m, n, k = 217, 29, 33
    native = _default_bwd_config(MXFP8Problem(m,n,k))
    config = bf16_dw_challenger(native)
    module = rtx.MXFP8Linear(k,n,device="cuda",backend="fused",autotune="off",
                             backward_config=config)
    old = rtx.MXFP8Linear(k,n,device="cuda",backend="fused",autotune="off",
                          backward_config=native)
    with torch.no_grad():
        old.weight.copy_(module.weight)
    x = torch.randn(m,k,device="cuda",dtype=torch.bfloat16,requires_grad=needs_dx)
    dy = torch.randn(n,m,device="cuda",dtype=torch.bfloat16).T
    ref_dw = (dy.float().T @ x.detach().float()).bfloat16()
    if needs_dx:
        torch.library.opcheck(_mxfp8_linear_dw_op,
            (dy,x.detach(),module.weight.detach(),_intern_bwd_config(config)))
    parameters = (x,module.weight) if needs_dx else (module.weight,)
    compiled = torch.compile(module, fullgraph=True, dynamic=False)
    grads = torch.autograd.grad(compiled(x),parameters,dy)
    torch.testing.assert_close(grads[-1],ref_dw,rtol=.006,atol=.02)
    if needs_dx:
        dx = torch.autograd.grad(old(x),x,dy)[0]
        torch.testing.assert_close(grads[0],dx,rtol=0,atol=0)
    repeated = torch.autograd.grad(compiled(x),parameters,-dy)
    torch.testing.assert_close(repeated[-1],-ref_dw,rtol=.006,atol=.02)


@cuda
def test_cublas_dw_mlp_fullgraph_and_independent_recipe():
    from rtx.autotune.mlp import _mlp_bf16_dw_challenger, reference
    from rtx.fp8_bwd import _default_bwd_config
    from rtx.mlp import DEFAULT_MLP_CONFIG

    torch.manual_seed(827)
    m,k,h,o = 257,64,64,64
    initial = replace(DEFAULT_MLP_CONFIG,
        gate_backward=_default_bwd_config(MXFP8Problem(m,2*h,k)),
        down_backward=_default_bwd_config(MXFP8Problem(m,o,h)))
    config = _mlp_bf16_dw_challenger(initial)
    module = rtx.MXFP8MLP(k,h,o,device="cuda",config=config)
    x = (torch.randn(m,k,device="cuda",dtype=torch.bfloat16)*.2).requires_grad_()
    g = torch.randn(o,m,device="cuda",dtype=torch.bfloat16).T
    with torch.no_grad():
        expected = reference(x,module.gate_up_weight,module.down_weight,g,"silu",bf16_dw=(True,True))
    forward = torch.compile(module, fullgraph=True, dynamic=False)
    for _ in range(2):
        y = forward(x)
        grads = torch.autograd.grad(y,(x,module.gate_up_weight,module.down_weight),g)
        for actual, ref in zip((y,*grads),expected):
            assert actual.isfinite().all()
            error = (actual.float()-ref.float()).norm() / ref.float().norm().clamp_min(1e-12)
            assert error < .025
