"""Ownership equivalence; independent arithmetic gates live in prequant_gate tests."""
from dataclasses import replace
import gc
import json

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('preserve', [False, True])
@pytest.mark.parametrize('selected_overlap', [False, True])
def test_prequant_down_multishape_graph_cleanup(preserve, selected_overlap, record_property):
    import rtx
    from rtx.runtime import clear_runtime_caches, retain_runtime_resources

    torch._dynamo.reset()
    torch.manual_seed(904)
    config = replace(rtx.MXFP8MLPConfig(), gate_backend='prequant', down_backend='prequant')
    if selected_overlap:
        from rtx._mlp_autotune import selected_mlp
        config = replace(config, backward_overlap=True, recompute_backend='prequant')
        spec = json.dumps(dict(mode='fixed', initial=config.to_dict()), sort_keys=True)
        def target(x, gu, down):
            return selected_mlp(x, gu, down, 'silu', spec, True)[0]
    else:
        def target(x, gu, down):
            return rtx.mxfp8_mlp(x, gu, down, config=config)
    fn = torch.compile(target,
                       fullgraph=True, dynamic=False, options={'triton.cudagraphs': False})
    cases = []
    for m, k, h, o in ((128, 128, 256, 128), (17, 35, 97, 41)):
        values = tuple((torch.randn(*s, device='cuda', dtype=torch.bfloat16)*.02).requires_grad_()
                       for s in ((m, k), (2*h, k), (o, h)))
        dy = torch.randn(m, o, device='cuda', dtype=torch.bfloat16)*.1
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        def step(values=values, dy=dy):
            y = fn(*values)
            return (y, *torch.autograd.grad(y, values, dy))
        with torch.cuda.stream(stream):
            for _ in range(3):
                step()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with retain_runtime_resources() as owners:
            with torch.cuda.graph(graph, stream=stream):
                captured = step()
        cases.append((values, stream, graph, owners, captured, step))

    clear_runtime_caches(device='cuda:0', preserve_compiled=preserve)
    live = []
    raw_live = []
    for iteration in range(8):
        expected = []
        for values, stream, graph, owners, captured, step in cases:
            with torch.no_grad():
                for tensor in values:
                    tensor.add_(.0001)
            expected.append(step())
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                graph.replay()
        torch.cuda.synchronize()
        for case, reference in zip(cases, expected):
            for actual, ref in zip(case[4], reference):
                assert bool(actual.isfinite().all())
                torch.testing.assert_close(actual, ref, rtol=.001, atol=1e-6)
        del expected, reference, actual, ref
        torch.cuda.synchronize()
        raw_live.append(torch.cuda.memory_allocated())
        # All captured graphs and owner dictionaries remain reachable. Collect
        # only unreachable Python cycles before checking their stable footprint;
        # arbitrary GC timing elsewhere in a full suite is not an ownership leak.
        gc.collect()
        torch.cuda.synchronize()
        live.append(torch.cuda.memory_allocated())
    record_property('raw_live_allocated_bytes', str(raw_live))
    record_property('live_allocated_bytes', str(live))
    # Ignore the first two iterations where cleared runners are rebuilt.
    assert max(live[2:]) - min(live[2:]) <= 4096, live
    assert all(case[3] for case in cases)
