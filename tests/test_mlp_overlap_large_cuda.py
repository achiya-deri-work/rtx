"""Concurrent recomputed backward under caller-owned graphs at deployment sizes."""
from dataclasses import replace
import json

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('recompute_schedule', ['consumer_two_stage', 'producer_one_stage',
                                               'producer_one_stage_gate128',
                                               'producer_one_stage_gate128_reuse',
                                               'producer_one_stage_gate128_reuse_exchange',
                                               'producer_one_stage_gate128_reuse_exchange_separate',
                                               'producer_one_stage_gate128_reuse_exchange_separate_cpasync',
                                               'producer_one_stage_gate128_reuse_packed'])
@pytest.mark.parametrize('shape', [(20480,256,768,256), (11296,256,768,256),
                                  (11712,128,384,128)])
def test_large_overlap_graph_changing_inputs(shape, recompute_schedule):
    from rtx._mlp_autotune import selected_mlp
    from rtx.autotune.mlp import make_mlp_adapter, mlp_structural_seeds, reference
    from rtx.runtime import clear_runtime_caches, retain_runtime_resources

    torch.manual_seed(1729)
    m,k,h,o = shape
    values = tuple((torch.randn(*s,device='cuda',dtype=torch.bfloat16)*scale).requires_grad_()
                   for s,scale in (((m,k),.2),((2*h,k),.02),((o,h),.02)))
    dy = torch.randn(m,o,device='cuda',dtype=torch.bfloat16)*.1
    adapter = make_mlp_adapter(*values,phase='train',measurement_mode='selected_direct')
    cfg = replace(mlp_structural_seeds(adapter,'mxfp8','train')[1], backward_overlap=True)
    if recompute_schedule.startswith('producer_one_stage'):
        move = {'gemm': {'stages': 1, 'scale_role': 'producer'}}
        assert move in adapter.axes['recompute_gemm.pipeline']
        cfg = adapter.update_fn(cfg,'recompute_gemm.pipeline',move)
    if 'gate128' in recompute_schedule:
        move={'gemm':dict(tile_m=128,tile_n=64,tile_k=128,
                          atom_layout_m=4,atom_layout_n=2)}
        assert move in adapter.axes['cooperative_gemm.gemm_geometry']
        cfg=adapter.update_fn(cfg,'cooperative_gemm.gemm_geometry',move)
    if '_reuse' in recompute_schedule:
        # Aligned seeds can inherit quad/dual-stream policy. This case tests
        # explicit serial reuse, not permission to alias concurrent operands.
        cfg=replace(cfg,gate_backward=replace(cfg.gate_backward,
            execution_order='dx_first',stream_schedule='single',quant_schedule='per_matmul'))
        move={'operand_storage':'serial_reuse'}
        assert move in adapter.axes['gate_backward.operand_storage']
        cfg=adapter.update_fn(cfg,'gate_backward.operand_storage',move)
    if '_exchange' in recompute_schedule:
        cfg=adapter.update_fn(cfg,'recompute_gemm.glu_grad_store','exchange')
    if '_separate' in recompute_schedule:
        assert 'separate' in adapter.axes['joint_output_storage']
        packed=cfg
        cfg=adapter.update_fn(cfg,'joint_output_storage','separate')
        from rtx.autotune.mlp import active_schedule
        assert active_schedule(cfg,'mxfp8','train')!=active_schedule(packed,'mxfp8','train')
        assert active_schedule(cfg,'mxfp8','forward')==active_schedule(packed,'mxfp8','forward')
    if recompute_schedule.endswith('_cpasync'):
        move={'dw': {'quant_a': {'transposed_load_engine':'cp_async',
                                'transposed_smem_padding':0}}}
        assert move in adapter.axes['gate_backward.dw_a_logical_transport']
        cfg=adapter.update_fn(cfg,'gate_backward.dw_a_logical_transport',move)
        assert cfg.gate_backward.dw.quant_a.transposed_load_engine=='cp_async'
    if recompute_schedule.endswith('_packed'):
        assert 'fp32_packed' in adapter.axes['recompute_gemm.glu_hidden_decode']
        cfg=adapter.update_fn(cfg,'recompute_gemm.glu_hidden_decode','fp32_packed')
    assert adapter.rejection(cfg) is None
    spec = json.dumps(dict(mode='fixed',initial=cfg.to_dict()),sort_keys=True)
    def forward(*v):
        return selected_mlp(*v,'silu',spec,True)[0]
    # Each pytest case is an independent deployment, not another specialization
    # of one long-lived caller. Nine cases exceed Dynamo's default limit of 8.
    torch._dynamo.reset_code(forward.__code__)
    fn = torch.compile(forward,
                       fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    def step():
        y = fn(*values)
        return y,*torch.autograd.grad(y,values,dy)
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with retain_runtime_resources() as owners:
        with torch.cuda.graph(graph,stream=stream):
            actual = step()
    clear_runtime_caches(device='cuda:0',preserve_compiled=True)
    assert owners
    for generation in range(3):
        with torch.no_grad():
            if generation:
                for v in values:
                    v.add_(.0002)
                dy.mul_(.99)
            expected = reference(*values,dy,'silu')
        stream.wait_stream(torch.cuda.current_stream())
        first = None
        for _ in range(16):
            with torch.cuda.stream(stream):
                graph.replay()
            stream.synchronize()
            if first is None:
                first = tuple(v.clone() for v in actual)
            for name,v,r,initial in zip(('output','dx','dgate','ddown'),actual,expected,first):
                v = v.detach()
                assert bool(v.isfinite().all()),name
                error = float((v.float()-r.float()).norm()/r.float().norm().clamp_min(1e-12))
                assert error < .0002,(name,error)
                assert torch.equal(v,initial),(name,'non-repeatable graph replay')
        del first,expected
