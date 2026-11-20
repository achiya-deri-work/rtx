"""User-side automatic graphs for a complete fixed-policy MoE training step.

This is a compiler/graph contract, not a throughput or convergence benchmark.
It deliberately uses default compile mode, not reduce-overhead or a private
per-layer replay wrapper. Cold balanced selection is qualified separately.
"""
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(),reason='SM120 CUDA required')
@pytest.mark.parametrize('autotune,metadata_backend',[('off','torch'),('cache','torch'),('cache','cute')])
def test_complete_fixed_moe_automatic_graph_replays_both_partitions(monkeypatch,record_property,tmp_path,autotune,metadata_backend):
    from rtx.moe import BalancedMoE,DEFAULT_MOE_CONFIG
    from rtx.routing import RoutingKernelConfig
    torch.manual_seed(1529)
    model=BalancedMoE(128,8,128,top_k=2,shared_hidden_features=128,router_dim=16,
                      device='cuda',config=DEFAULT_MOE_CONFIG,autotune=autotune,
                      autotune_cache_dir=tmp_path/'cache',
                      router_options={'config':RoutingKernelConfig(qb_backend='native_warp',
                          metadata_backend=metadata_backend,
                          **(dict(quota_cache_items=8,quota_threshold=0,quota_early_exit=True)
                             if metadata_backend=='cute' else {}))})
    with torch.no_grad():model.router.projection_weight.normal_(std=.03)
    x=(torch.randn(64,128,device='cuda',dtype=torch.bfloat16)*.3).requires_grad_()
    dy=torch.randn_like(x)
    parameters=tuple(model.parameters())
    replayed=[]
    original=torch.cuda.CUDAGraph.replay
    def replay(graph):
        replayed.append(id(graph))
        return original(graph)
    monkeypatch.setattr(torch.cuda.CUDAGraph,'replay',replay)
    compiled=torch.compile(model,fullgraph=True,dynamic=False,options={'triton.cudagraphs':True})
    forward_replays=backward_replays=0
    for iteration in range(6):
        torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad():
            x.add_(.001)
            model.router.projection_weight.add_(.0001)
        before=len(replayed)
        output=compiled(x)
        forward_replays+=len(replayed)-before
        saved_output=output.detach().clone()
        before=len(replayed)
        gradients=torch.autograd.grad(output,(x,*parameters),dy)
        backward_replays+=len(replayed)-before
        actual=[saved_output,*[value.detach().clone() for value in gradients]]
        if iteration==5:
            expected_output=model(x)
            expected=[expected_output,*torch.autograd.grad(expected_output,(x,*parameters),dy)]
            for index,(value,reference) in enumerate(zip(actual,expected)):
                assert value.shape==reference.shape and value.dtype==reference.dtype
                assert bool(value.isfinite().all()),index
                delta=value.float()-reference.float()
                nrms=float(delta.norm()/reference.float().norm().clamp_min(1e-12))
                maximum=float(delta.abs().max())
                assert nrms<.01 or maximum<2e-7,(index,nrms,maximum)
        del output,gradients,actual
    record_property('forward_graph_replays',forward_replays)
    record_property('backward_graph_replays',backward_replays)
    assert forward_replays>0,'Inductor skipped automatic forward graphs'
    assert backward_replays>0,'Inductor skipped automatic backward graphs'
