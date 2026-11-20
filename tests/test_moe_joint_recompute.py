"""Low-residency joint expert recomputation: legality, bytes and adjoints."""
from dataclasses import replace

import pytest
import torch

cuda=pytest.mark.skipif(not torch.cuda.is_available(),reason='SM120 CUDA required')


def _schedule(engine,cols=64,stages=1,vec=4,swizzle='xor_16b'):
    from rtx.configs.moe import DEFAULT_PREPARED_MOE_CONFIG
    return replace(DEFAULT_PREPARED_MOE_CONFIG.recompute,tile_m=64,tile_n=cols,
                   tile_k=128,stages=stages,scale_role='producer',
                   glu_grad_store='exchange',glu_grad_load='interleaved',
                   glu_quant_vec=vec,glu_scratch_swizzle=swizzle)


def test_joint_recompute_tuning_seeds_and_rejections():
    from rtx.autotune import AutotunePolicy
    from rtx.autotune.moe import make_moe_adapter,make_moe_tuner,joint_recompute_seeds,rejection
    from rtx.autotune.store import InMemoryTuningStore
    from rtx.moe import DEFAULT_MOE_CONFIG,MXFP8MoEConfig
    shape=(7376,384,32,928,4,128)
    seeds=list(joint_recompute_seeds(shape))
    assert {s.prepared.joint_recompute_engine for s in seeds}=={'operand_reuse','dual_operand'}
    assert all(MXFP8MoEConfig.from_dict(s.to_dict())==s for s in seeds)
    assert all(s.prepared.recompute.maxrregcount==DEFAULT_MOE_CONFIG.prepared.recompute.maxrregcount
               for s in seeds)
    assert DEFAULT_MOE_CONFIG.prepared.joint_recompute_engine=='legacy'
    assert 'prepared.joint_recompute_engine' not in make_moe_adapter(shape,None,phase='forward').axes
    adapter=make_moe_adapter(shape,None)
    tuner=make_moe_tuner(adapter,InMemoryTuningStore(),
                         AutotunePolicy(max_trials=12,candidate_mix='mixed'))
    portfolio=tuner.strategies['structural'].pool
    assert any(c.prepared.joint_recompute_engine!='legacy' for c in portfolio[:6])
    cfg=seeds[0]
    back=adapter.update_fn(cfg,'prepared.backward_preparation','separate')
    assert back.prepared.joint_recompute_engine=='legacy'
    p=cfg.prepared
    assert 'nonpersistent' in rejection(replace(cfg,prepared=replace(
        p,recompute=replace(p.recompute,persistent_waves=1))),shape)
    assert 'retired A' in rejection(replace(cfg,prepared=replace(
        p,recompute=replace(p.recompute,tile_n=128))),shape)
    assert '32-value' in rejection(cfg,(*shape[:-1],65))


@cuda
@pytest.mark.parametrize('engine,cols,stages,vec,swizzle',[
    ('operand_reuse',64,1,4,'xor_16b'),
    ('operand_reuse',128,1,1,'none'),
    ('dual_operand',64,1,8,'none'),
    ('dual_operand',128,2,4,'xor_16b'),
])
def test_joint_recompute_retired_operands_exact(engine,cols,stages,vec,swizzle):
    from rtx.configs.moe import DEFAULT_PREPARED_MOE_CONFIG
    from rtx.kernels.mxfp8 import MXFP8Problem
    from rtx.kernels.mxfp8_moe_buffers import compile_expert_joint_recompute,compile_expert_quant
    # Partial final tiles in both row/hidden axes (for cols=128), and a
    # partial final K stage when using two stages.
    e,m,h,k=3,96,96,160
    g=torch.Generator(device='cuda').manual_seed(7183)
    def normal(rows,cols):
        return torch.randn((e*rows,cols),generator=g,device='cuda',dtype=torch.bfloat16)*.2
    x,w,dh=normal(m,k),normal(2*h,k),normal(m,h)
    x.view(e,m,k)[:,m-9:]=0
    dh.view(e,m,h)[:,m-9:]=0
    base=DEFAULT_PREPARED_MOE_CONFIG
    q=base.quant
    cfg=_schedule(engine,cols,stages,vec,swizzle)
    shape=MXFP8Problem(m,2*h,k)
    def outputs():
        return tuple(torch.empty((e*r,c),device='cuda',dtype=dtype) for r,c,dtype in (
            (m,h,torch.float8_e4m3fn),(m,h//32,torch.uint8),
            (h,m,torch.float8_e4m3fn),(h,m//32,torch.uint8),
            (2*m,h,torch.float8_e4m3fn),(2*m,h//32,torch.uint8),
            (2*h,m,torch.float8_e4m3fn),(2*h,m//32,torch.uint8)))
    qx,qw=(torch.empty_like(v,dtype=torch.float8_e4m3fn) for v in (x,w))
    sx,sw=(torch.empty((v.shape[0],k//32),device='cuda',dtype=torch.uint8) for v in (x,w))
    compile_expert_quant(e,((m,k,'row',False),(2*h,k,'row',True)),q)(x,(x,w),(qx,qw),(sx,sw))
    expected,actual=outputs(),outputs()
    legacy=compile_expert_joint_recompute(e,shape,cfg,q)
    candidate=compile_expert_joint_recompute(e,shape,cfg,q,engine)
    legacy(qx,qw,sx,sw,dh,(dh,*expected))
    for poison in (255,119,231):
        # All output components must overwrite stale bytes, including zero
        # rows emitted by feature- and token-axis quantizers.
        for a in actual:a.view(torch.uint8).fill_(poison)
        candidate(qx,qw,sx,sw,dh,(dh,*actual))
        for i,(a,b) in enumerate(zip(actual,expected)):
            assert torch.equal(a.view(torch.uint8),b.view(torch.uint8)),(engine,i,poison)


@cuda
@pytest.mark.parametrize('engine,shape,compiled',[
    ('operand_reuse',(13,35,5,2,64),False),
    ('dual_operand',(129,160,4,2,96),True),
])
def test_joint_recompute_fp64_fullgraph_replay_all_gradients(engine,shape,compiled):
    from rtx._moe_reference import operational_reference
    from rtx.moe import DEFAULT_MOE_CONFIG,mxfp8_routed_experts
    from rtx.routing import BalancedRouting
    from tests.test_mxfp8_moe import make_route
    n,d,e,k,h=shape
    torch.manual_seed(40541)
    x=(torch.randn(n,d,device='cuda',dtype=torch.bfloat16)*.3).requires_grad_()
    gu=(torch.randn(e,2*h,d,device='cuda',dtype=torch.bfloat16)*.05).requires_grad_()
    down=(torch.randn(e,d,h,device='cuda',dtype=torch.bfloat16)*.05).requires_grad_()
    ids,weights,inverse=make_route(n,e,k,'cuda')
    weights=weights.detach().requires_grad_()
    operands=(x,gu,down,weights)
    config=replace(DEFAULT_MOE_CONFIG,prepared=replace(DEFAULT_MOE_CONFIG.prepared,
        backward_preparation='joint',joint_recompute_engine=engine,
        recompute=_schedule(engine),input_preparation='quantize_once'))
    def model(x,gu,down,weights):
        return mxfp8_routed_experts(x,gu,down,BalancedRouting(ids,weights,inverse),config=config)
    if compiled:model=torch.compile(model,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    dy=torch.randn(d,n,device='cuda').T
    def step():
        y=model(*operands)
        return y,*torch.autograd.grad(y,operands,dy)
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):warm=tuple(value.detach() for value in step())
    stream.synchronize()
    expected=operational_reference(x,gu,down,ids,weights,inverse,dy)
    for name,a,b in zip(('output','dx','dgu','ddown','droute'),warm,expected):
        delta=a.detach().double()-b.double()
        nrms=float(delta.norm()/b.double().norm().clamp_min(1e-12))
        maximum=float(delta.abs().max())
        assert a.shape==b.shape and bool(a.isfinite().all())
        assert nrms<(.003 if name=='output' else .006) or maximum<2e-7,(name,nrms,maximum)
    torch.cuda.synchronize()
    if compiled:
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):replayed=tuple(value.detach() for value in step())
        for _ in range(3):
            graph.replay()
            for a,b in zip(warm,replayed):torch.testing.assert_close(a,b,rtol=0,atol=0)
    torch.cuda.synchronize()
    base=torch.cuda.memory_allocated()
    for _ in range(5):
        check=step(); del check
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated()==base
