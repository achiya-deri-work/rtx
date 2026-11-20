"""K32/K64 weight gradients: storage legality, packed numerics and replay."""
from dataclasses import replace

import pytest
import torch

cuda=pytest.mark.skipif(not torch.cuda.is_available(),reason='SM120 CUDA required')


def test_short_dw_is_a_train_only_tuner_alternative():
    from rtx.autotune.moe import make_moe_adapter,short_dw_seeds,rejection
    from rtx.moe import DEFAULT_MOE_CONFIG,MXFP8MoEConfig
    shape=(512,384,32,64,4,128)
    p=DEFAULT_MOE_CONFIG.prepared
    assert p.gate_dw.tile_k==p.down_dw.tile_k==128
    configs=list(short_dw_seeds(shape))
    assert {c.prepared.gate_dw.tile_k for c in configs}=={32,64,128}
    assert all(MXFP8MoEConfig.from_dict(c.to_dict())==c for c in configs)
    axis='prepared.gate_dw.short_k'
    assert axis not in make_moe_adapter(shape,None,phase='forward').axes
    adapter=make_moe_adapter(shape,None,device={'multiprocessor_count':170})
    assert adapter.axes[axis]==(0,32,64)
    changed=adapter.update_fn(DEFAULT_MOE_CONFIG,axis,32)
    assert adapter.rejection(changed) is None
    original=adapter.features(DEFAULT_MOE_CONFIG)
    current=adapter.features(changed)
    prefix='derived.gate_dw_'
    assert original[prefix+'padded_k']==128 and current[prefix+'padded_k']==64
    assert original[prefix+'padded_reduction_ratio']==2 and current[prefix+'padded_reduction_ratio']==1
    assert original[prefix+'tile_work_amplification']==2 and current[prefix+'tile_work_amplification']==1
    assert original[prefix+'cta_waves_single_residency']==current[prefix+'cta_waves_single_residency']
    back=adapter.update_fn(changed,axis,0)
    assert back.prepared.gate_dw.tile_k==128 and back.prepared.transports[-1]=='tma'
    assert 'one to four' in rejection(changed,(1144,384,32,143,4,128))
    split=replace(changed,prepared=replace(changed.prepared,gate_dw_splits=2))
    assert 'unsplit' in rejection(split,shape)
    assert not list(short_dw_seeds((1144,384,32,143,4,128)))


def test_short_dw_is_evaluated_in_eight_trial_tiny_policy():
    from rtx.autotune.effort import AutotunePolicy
    from rtx.autotune.moe import make_moe_adapter,make_moe_tuner
    from rtx.autotune.outcomes import TrialOutcome
    from rtx.autotune.store import InMemoryTuningStore
    calls=[]
    def measure(config):
        calls.append(config)
        ms=1.0 if config.prepared.down_dw.tile_k>=128 else .97
        return TrialOutcome('ok',median_ms=ms,timings_ms=[ms]*3)
    adapter=make_moe_adapter((512,384,32,64,4,128),measure)
    policy=AutotunePolicy(max_trials=8,max_seconds=120,search_schedule='progressive',
                          candidate_mix='mixed',confirmation_repeats=0,pool_size=64,structural_proposals=64)
    make_moe_tuner(adapter,InMemoryTuningStore(),policy).tune()
    assert calls[0]==adapter.initial_config
    assert len({adapter.config_id(c) for c in calls})==8
    assert any(c.prepared.down_dw.tile_k==c.prepared.gate_dw.tile_k==64 for c in calls)


@cuda
@pytest.mark.parametrize('tile_k,physical_k,shape',[(32,32,(70,73)),(64,64,(64,128)),(64,96,(70,73))])
def test_short_dw_poisoned_grouped_mma32_scale_blocks(tile_k,physical_k,shape):
    from rtx._moe_reference import quantize_decode
    from rtx.configs.moe import DEFAULT_PREPARED_MOE_CONFIG as p
    from rtx.kernels.mxfp8 import MXFP8Problem
    from rtx.kernels.mxfp8_moe_buffers import compile_expert_packed_gemm,compile_expert_quant
    e,m,n=3,*shape
    g=torch.Generator(device='cuda').manual_seed(84122)
    def source(rows):
        x=torch.randn((e,rows,physical_k),generator=g,device='cuda',dtype=torch.bfloat16)
        # Distinct E8M0 values per feature row, expert and K block catch scale
        # selection errors rather than passing accidentally with unit scales.
        x=x*torch.exp2(torch.randint(-6,4,(e,rows,physical_k//32),generator=g,
            device='cuda').float()).repeat_interleave(32,-1).to(x.dtype)
        x[0,:,:]=0
        return x
    a,b=source(m),source(n)
    qx=torch.empty((e*m,physical_k),dtype=torch.float8_e4m3fn,device='cuda')
    qw=torch.empty((e*n,physical_k),dtype=torch.float8_e4m3fn,device='cuda')
    sx=torch.empty((e*m,physical_k//32),dtype=torch.uint8,device='cuda')
    sw=torch.empty((e*n,physical_k//32),dtype=torch.uint8,device='cuda')
    pack=compile_expert_quant(e,((m,physical_k,'row',False),(n,physical_k,'row',False)),p.quant)
    pack(a.flatten(0,1),(a.flatten(0,1),b.flatten(0,1)),(qx,qw),(sx,sw))
    expected=(quantize_decode(a).double() @ quantize_decode(b).double().mT).bfloat16().flatten(0,1)
    out=torch.empty_like(expected)
    cfg=p.with_short_dw('gate_dw',tile_k).gate_dw
    kernel=compile_expert_packed_gemm(e,MXFP8Problem(m,n,physical_k),cfg,'linear','cooperative')
    for poison in (255,47,215):
        out.view(torch.uint8).fill_(poison)
        kernel(qx,qw,sx,sw,out,out)
        assert bool(out.isfinite().all())
        delta=out.float()-expected.float()
        assert float(delta.norm()/expected.float().norm())<.006
        assert not torch.count_nonzero(out[:m])


@cuda
@pytest.mark.parametrize('shape,tile_k,compiled',[
    ((13,35,5,2,64),32,False),((129,160,4,2,96),64,True),
])
def test_short_dw_fp64_and_replay_all_adjoints(shape,tile_k,compiled):
    from rtx._moe_reference import operational_reference
    from rtx.moe import DEFAULT_MOE_CONFIG,mxfp8_routed_experts
    from rtx.routing import BalancedRouting
    from tests.test_mxfp8_moe import make_route
    n,d,e,k,h=shape
    torch.manual_seed(2841)
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x=(torch.randn(n,d,device='cuda',dtype=torch.bfloat16)*.3).requires_grad_()
        gu=(torch.randn(e,2*h,d,device='cuda',dtype=torch.bfloat16)*.05).requires_grad_()
        down=(torch.randn(e,d,h,device='cuda',dtype=torch.bfloat16)*.05).requires_grad_()
        ids,weights,inverse=make_route(n,e,k,'cuda')
        weights=weights.detach().requires_grad_()
        operands=(x,gu,down,weights)
        p=DEFAULT_MOE_CONFIG.prepared
        p=replace(p,backward_preparation='joint',recompute=replace(p.recompute,glu_grad_store='exchange'))
        p=p.with_short_dw('gate_dw',tile_k).with_short_dw('down_dw',tile_k)
        config=replace(DEFAULT_MOE_CONFIG,prepared=p)
        def model(x,gu,down,weights):
            return mxfp8_routed_experts(x,gu,down,BalancedRouting(ids,weights,inverse),config=config)
        if compiled:model=torch.compile(model,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
        dy=torch.randn(d,n,device='cuda').T
        def step():
            y=model(*operands)
            return tuple(v.detach() for v in (y,*torch.autograd.grad(y,operands,dy)))
        warm=step()
        expected=operational_reference(x,gu,down,ids,weights,inverse,dy)
        for name,a,b in zip(('output','dx','dgu','ddown','droute'),warm,expected):
            delta=a.float()-b.float()
            nrms=float(delta.norm()/b.float().norm().clamp_min(1e-12))
            maximum=float(delta.abs().max())
            assert a.shape==b.shape and bool(a.isfinite().all())
            assert nrms<(.003 if name=='output' else .006) or maximum<2e-7,(name,nrms,maximum)
        if compiled:
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):replayed=step()
            for _ in range(3):
                graph.replay()
                for a,b in zip(warm,replayed):torch.testing.assert_close(a,b,rtol=0,atol=0)
        stream.synchronize()
        live=torch.cuda.memory_allocated()
        for _ in range(5):
            result=step();del result
        stream.synchronize()
        assert torch.cuda.memory_allocated()==live
