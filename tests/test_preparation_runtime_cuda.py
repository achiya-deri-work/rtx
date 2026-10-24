"""Explicit CUDA qualification of cold preparation selection and cached replay."""
import json

import pytest
import torch
import rtx
from rtx.autotune.legacy import CoordinateDescentPolicy


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA devices required")
def test_preparation_selectors_use_operand_device_and_restore_current(tmp_path):
    from types import SimpleNamespace
    from rtx.fp8 import DEFAULT_MXFP8_PREQUANT_CONFIG
    from rtx.autotune.preparation_runtime import select_forward_runner, select_backward_config
    with torch.cuda.device(0):
        x = torch.randn(128,128,device="cuda:1",dtype=torch.bfloat16)*.25
        weight = torch.randn_like(x)*.1
        dy = torch.randn_like(x)*.25
        request = SimpleNamespace(initial=DEFAULT_MXFP8_PREQUANT_CONFIG,
            cache_dir=str(tmp_path), policy=rtx.AutotunePolicy(
                max_trials=1,max_seconds=60,confirmation_repeats=0))
        runner = select_forward_runner(x,weight,request)
        assert torch.cuda.current_device() == 0
        output = torch.empty_like(x)
        with torch.cuda.device(1):
            runner(x,weight,output)
        reference = x.float()@weight.float().T
        assert (output.float()-reference).norm()/reference.norm() < .06
        for phase in ("dx","dw"):
            select_backward_config(dy,x,weight,request,phase)
            assert torch.cuda.current_device() == 0
        records = [json.loads(line) for line in
                   (tmp_path/"preparation"/"observations.jsonl").read_text().splitlines()]
        assert len(records) == 3
        assert all(row["outcome"]["status"] == "ok" for row in records)
        expected_l2 = torch.cuda.get_device_properties(1).L2_cache_size
        assert all(row["features"]["context.device.l2_cache_size"] == expected_l2
                   for row in records)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("compiled", [False, True])
def test_hot_preparation_training_does_not_retain_live_allocations(tmp_path, monkeypatch, compiled):
    import gc
    monkeypatch.setenv("RTX_PREPARATION_SELECTION", "1")
    torch.compiler.reset()
    layer = rtx.MXFP8Linear(128,128,device="cuda",backend="auto",autotune="balanced",
        tuning_policy=rtx.AutotunePolicy(max_trials=2,max_seconds=60,confirmation_repeats=0),
        autotune_cache_dir=tmp_path)
    x = torch.randn(128,128,device="cuda",dtype=torch.bfloat16,requires_grad=True)
    dy = torch.randn_like(x)
    run = torch.compile(layer,fullgraph=True,dynamic=False) if compiled else layer
    def step():
        run(x).backward(dy)
        layer.zero_grad(set_to_none=True)
        x.grad = None
    for _ in range(10):
        step()
    gc.collect()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    observations = (tmp_path / "preparation" / "observations.jsonl").read_bytes()
    allocated = []
    for _ in range(3):
        for _ in range(25):
            step()
        gc.collect()
        torch.cuda.synchronize()
        allocated.append(torch.cuda.memory_allocated())
    assert allocated == [before] * 3
    assert (tmp_path / "preparation" / "observations.jsonl").read_bytes() == observations


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("compiled", [False, True])
def test_auto_preparation_first_hit_and_replay(tmp_path, compiled, monkeypatch):
    monkeypatch.setenv("RTX_PREPARATION_SELECTION", "1")
    torch.compiler.reset()
    torch.manual_seed(42)
    module = rtx.MXFP8Linear(128, 128, device="cuda", backend="auto", autotune="balanced",
        tuning_policy=CoordinateDescentPolicy(max_trials=2, time_budget_s=60),
        autotune_cache_dir=tmp_path, backward_config=rtx.DEFAULT_MXFP8_BWD_CONFIG)
    x = (torch.randn(128,128,device="cuda",dtype=torch.bfloat16)*.25).requires_grad_()
    run = torch.compile(module, fullgraph=True, dynamic=False) if compiled else module
    y = run(x)
    reference = x.float() @ module.weight.float().T
    assert torch.isfinite(y).all()
    assert (y.float()-reference).square().mean().sqrt() < .06 * reference.square().mean().sqrt()
    y.float().square().mean().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(module.weight.grad).all()
    observations = tmp_path / "preparation" / "observations.jsonl"
    before = observations.read_bytes()
    records = [json.loads(line) for line in before.splitlines()]
    assert {r["config"]["implementation"] for r in records} == {"materialized", "fused"}
    for _ in range(3):
        run(x)
    assert observations.read_bytes() == before
    # Cold runner on a new stream must reuse persistent measurements.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run(x)
    stream.synchronize()
    assert observations.read_bytes() == before
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = run(x)
    with torch.no_grad():
        x.mul_(2)
    graph.replay()
    torch.cuda.synchronize()
    expected = run(x)
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, expected, rtol=0, atol=0)
    assert observations.read_bytes() == before


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_runtime_refinement_expands_budget(tmp_path, monkeypatch):
    monkeypatch.setenv("RTX_PREPARATION_SELECTION", "1")
    x = torch.randn(128,128,device="cuda",dtype=torch.bfloat16) * .25
    weight = torch.randn(128,128,device="cuda",dtype=torch.bfloat16) * .02
    path = tmp_path / "preparation" / "observations.jsonl"
    for budget in (2, 4):
        policy = rtx.AutotunePolicy(max_trials=budget, max_seconds=60,
                                   cache_behavior="refine", confirmation_repeats=0)
        result = rtx.mxfp8_linear(x, weight, backend="auto", autotune="balanced",
                                 tuning_policy=policy, autotune_cache_dir=tmp_path)
        assert result.isfinite().all()
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == budget
        # Refinement may remeasure a correctness-valid candidate whose earlier
        # latency was nonstationary. Those attempts still consume the budget;
        # a stable observation must never be redundantly explored again.
        previous = {}
        for row in rows:
            cid = row["config_id"]
            for prior in previous.get(cid, []):
                assert prior["outcome"]["status"] == "ok"
                assert prior["outcome"]["metadata"]["sampling"]["collection"]["stationary"] is False
            previous.setdefault(cid, []).append(row)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_nearby_shape_transfer_remeasures_instead_of_reusing_latency(tmp_path, monkeypatch):
    monkeypatch.setenv("RTX_PREPARATION_SELECTION", "1")
    weight = torch.randn(128,128,device="cuda",dtype=torch.bfloat16)*.02
    policy = rtx.AutotunePolicy(max_trials=2,max_seconds=60,confirmation_repeats=0)
    inputs = [torch.randn(m,128,device="cuda",dtype=torch.bfloat16)*.25 for m in (128,256)]
    for x in inputs:
        result = rtx.mxfp8_linear(x,weight,backend="auto",autotune="balanced",
            tuning_policy=policy,autotune_cache_dir=tmp_path)
        reference = x.float() @ weight.float().T
        assert result.isfinite().all()
        assert (result.float()-reference).square().mean().sqrt() < .06 * reference.square().mean().sqrt()
    path = tmp_path / "preparation" / "observations.jsonl"
    before = path.read_bytes()
    rows = [json.loads(line) for line in before.splitlines()]
    contexts = {row["context_id"] for row in rows}
    assert len(contexts) == 2
    assert all(sum(row["context_id"]==context for row in rows)==2 for context in contexts)
    for x in inputs:
        rtx.mxfp8_linear(x,weight,backend="auto",autotune="balanced",
            tuning_policy=policy,autotune_cache_dir=tmp_path)
    assert path.read_bytes() == before


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("compiled", [False, True])
def test_backward_arity_selection(tmp_path, monkeypatch, compiled):
    monkeypatch.delenv("RTX_PREPARATION_SELECTION", raising=False)
    torch.manual_seed(149)
    for x_grad, w_grad in ((True,True), (True,False), (False,True)):
        torch.compiler.reset()
        layer = rtx.MXFP8Linear(128,128,device="cuda",backend="auto",autotune="balanced",
            tuning_policy=rtx.AutotunePolicy(max_trials=2,max_seconds=60,confirmation_repeats=0,
                                           preparation_selection=True),
            autotune_cache_dir=tmp_path)
        layer.weight.requires_grad_(w_grad)
        x = (torch.randn(128,128,device="cuda",dtype=torch.bfloat16)*.25).requires_grad_(x_grad)
        dy = torch.randn(128,128,device="cuda",dtype=torch.bfloat16)*.25
        run = torch.compile(layer,fullgraph=True,dynamic=False) if compiled else layer
        y = run(x)
        y.backward(dy)
        if x_grad:
            ref = dy.float() @ layer.weight.float()
            assert (x.grad.float()-ref).square().mean().sqrt() < .06 * ref.square().mean().sqrt()
        else:
            assert x.grad is None
        if w_grad:
            ref = dy.float().T @ x.float()
            assert (layer.weight.grad.float()-ref).square().mean().sqrt() < .06 * ref.square().mean().sqrt()
        else:
            assert layer.weight.grad is None
        before = (tmp_path / "preparation" / "observations.jsonl").read_bytes()
        run(x).backward(dy)
        assert (tmp_path / "preparation" / "observations.jsonl").read_bytes() == before
    rows = [json.loads(line) for line in before.splitlines()]
    # One forward context and distinct paired/dX-only/dW-only contexts.
    assert len({row["context_id"] for row in rows}) == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_functional_backward_selection_compile(tmp_path, monkeypatch):
    monkeypatch.setenv("RTX_PREPARATION_SELECTION", "1")
    torch.compiler.reset()
    policy = rtx.AutotunePolicy(max_trials=2,max_seconds=60,confirmation_repeats=0)
    x = torch.randn(128,128,device="cuda",dtype=torch.bfloat16)*.25
    w = torch.randn_like(x)*.02
    dy = torch.randn_like(x)*.25
    def call(dy,x,w):
        return rtx.mxfp8_linear_backward(dy,x,w,autotune="balanced",tuning_policy=policy,
                                        autotune_cache_dir=tmp_path)
    compiled = torch.compile(call,fullgraph=True,dynamic=False)
    dx,dw = compiled(dy,x,w)
    torch.testing.assert_close(dx,call(dy,x,w)[0],rtol=0,atol=0)
    assert dx.isfinite().all() and dw.isfinite().all()
    rows = [json.loads(line) for line in (tmp_path / "preparation" / "observations.jsonl").read_text().splitlines()]
    assert len(rows) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_ragged_backward_selection(tmp_path, monkeypatch):
    monkeypatch.setenv("RTX_PREPARATION_SELECTION", "1")
    torch.manual_seed(7)
    x = torch.randn(129,160,device="cuda",dtype=torch.bfloat16)*.25
    w = torch.randn(96,160,device="cuda",dtype=torch.bfloat16)*.02
    dy = torch.randn(129,96,device="cuda",dtype=torch.bfloat16)*.25
    dx,dw = rtx.mxfp8_linear_backward(dy,x,w,autotune="balanced",
        tuning_policy=rtx.AutotunePolicy(max_trials=2,max_seconds=60,confirmation_repeats=0),
        autotune_cache_dir=tmp_path)
    for actual, ref in ((dx,dy.float() @ w.float()), (dw,dy.float().T @ x.float())):
        assert actual.isfinite().all()
        assert (actual.float()-ref).square().mean().sqrt() < .06 * ref.square().mean().sqrt()
