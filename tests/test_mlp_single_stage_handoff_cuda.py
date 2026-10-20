"""Regression for intermittent same-input consumer-scale GLU gradients.

The production quarantine intentionally remains; tests bypass it locally to
exercise the proposed rendezvous repair rather than silently widening support.
"""
from dataclasses import replace
import json

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('schedule', ['before_wait','after_wait'])
@pytest.mark.parametrize('m', [11296,11712,11744])
def test_single_stage_handoff_identical_input_gradients(monkeypatch,schedule,m):
    import rtx.configs.glu as legality
    from rtx._mlp_autotune import selected_mlp
    from rtx.autotune.mlp import make_mlp_adapter, mlp_structural_seeds, reference

    torch._dynamo.reset()
    torch.manual_seed(1729)
    k,h,o = 256,768,256
    values=tuple((torch.randn(*s,device='cuda',dtype=torch.bfloat16)*scale).requires_grad_()
        for s,scale in (((m,k),.2),((2*h,k),.02),((o,h),.02)))
    dy=torch.randn(m,o,device='cuda',dtype=torch.bfloat16)*.1
    adapter=make_mlp_adapter(*values,phase='train',measurement_mode='compiled_direct',
                            warmup=1,samples=3,calls_per_sample=1)
    candidates=[c for c in mlp_structural_seeds(adapter,'mxfp8','train')
        if c.gate_backend=='prequant' and c.down_backend=='prequant'
        and c.recompute_backend=='prequant' and c.recompute.gemm.tile_m==128
        and c.recompute.gemm.tile_n==64 and c.gate_backward.dw.split_reduction==4]
    assert candidates
    base=candidates[0]
    config=replace(base,recompute=replace(base.recompute,
        gemm=replace(base.recompute.gemm,stages=1,scale_role='consumers',scale_schedule=schedule)))
    assert legality.prequant_recompute_rejection(config.recompute.gemm)
    monkeypatch.setattr(legality,'prequant_recompute_rejection',lambda config: None)
    spec=json.dumps(dict(mode='fixed',initial=config.to_dict()),sort_keys=True)
    fn=torch.compile(lambda *v:selected_mlp(*v,'silu',spec,True)[0],
        fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    with torch.no_grad():
        expected=reference(*values,dy,'silu')
    first=None
    for _ in range(32):
        y=fn(*values)
        actual=(y,*torch.autograd.grad(y,values,dy))
        torch.cuda.synchronize()
        if first is None:
            first=tuple(t.detach().clone() for t in actual)
        for a,b,initial in zip(actual,expected,first):
            assert bool(a.isfinite().all())
            assert float((a.detach().float()-b.float()).norm()/b.float().norm().clamp_min(1e-12))<.025
            assert torch.equal(a,initial), 'single-stage repeated gradients changed'

    # Capture on the warmed stream; keep all scratch owners alive with the graph.
    del y,actual,a
    from rtx.runtime import retain_runtime_resources
    def step():
        y=fn(*values)
        return (y,*torch.autograd.grad(y,values,dy))
    capture_stream=torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            step()
    capture_stream.synchronize()
    graph=torch.cuda.CUDAGraph()
    with retain_runtime_resources() as owners:
        with torch.cuda.graph(graph,stream=capture_stream):
            replayed=step()
    assert owners
    for _ in range(8):
        with torch.no_grad():
            for tensor in (*values,dy):
                tensor.add_(.0001)
            expected=reference(*values,dy,'silu')
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            graph.replay()
        capture_stream.synchronize()
        for a,b in zip(replayed,expected):
            assert bool(a.isfinite().all())
            assert float((a.detach().float()-b.float()).norm()/b.float().norm().clamp_min(1e-12))<.025
