"""Alignment changes may change schedules, never exclude a complete family."""
import pytest
import torch

import rtx
from rtx.fp8 import DEFAULT_MXFP8_PREQUANT_CONFIG, _default_dynamic_prequant_config
from rtx.kernels.mxfp8 import MXFP8Problem


@pytest.mark.parametrize('m,n,k',[
    (3648,128,256),(3648,512,256),
    (76608,128,256),(76608,256,256),(76608,384,256),(76608,1024,256),
    (310080,256,128),(310080,256,256),
])
def test_native64_family_seed_survives_half_tile_alignment(m,n,k):
    from rtx.prequant_autotune import _legal_default_prequant_config
    problem=MXFP8Problem(m,n,k)
    assert DEFAULT_MXFP8_PREQUANT_CONFIG.rejection(problem) is not None
    config=_default_dynamic_prequant_config(problem)
    assert config.gemm.scale_layout=='mma64x128'
    assert config.quant.scale_layout=='mma64'
    assert config.resolved_weight_quant().scale_layout=='mma128'
    assert config.rejection(problem) is None
    assert _legal_default_prequant_config(problem)==config
    assert config.quant.scale_rounding=='infinity'
    layer=rtx.MXFP8Linear(k,n,device='meta',autotune=False)
    decision=layer.explain(torch.empty((m,k),device='meta',dtype=torch.bfloat16))
    assert decision.backend=='materialized'
    assert decision.config['gemm']['scale_layout']=='mma64x128'


@pytest.mark.parametrize('m,n,k',[(256,128,128),(51072,256,256),(76672,256,256),(310144,256,256)])
def test_existing_aligned_default_is_unchanged(m,n,k):
    assert _default_dynamic_prequant_config(MXFP8Problem(m,n,k)) is DEFAULT_MXFP8_PREQUANT_CONFIG


@pytest.mark.parametrize('m,n,k',[(17,29,33),(624,512,256),(76608,64,256)])
def test_masked_family_seed_preserves_logical_dimensions(m,n,k):
    config=_default_dynamic_prequant_config(MXFP8Problem(m,n,k))
    assert config.gemm.scale_layout=='row_major'
    assert config.gemm.epilogue=='direct'
    assert config.rejection(MXFP8Problem(m,n,k)) is None
    # Native MMA128 legality is not weakened or silently padded.
    assert DEFAULT_MXFP8_PREQUANT_CONFIG.rejection(MXFP8Problem(m,n,k)) is not None


@pytest.mark.parametrize('m,n,k',[(192,128,128),(76608,256,256),(310080,256,128)])
def test_native_dx_is_reachable_in_the_opening_budget(m,n,k):
    from rtx._bwd_search import backward_structural_seeds,backward_search_space
    from rtx.kernels.mxfp8_bwd import DEFAULT_FUSED_MXFP8_BWD_CONFIG
    from rtx.bwd_autotune import BWD_SEARCH_SPACE
    problem=MXFP8Problem(m,n,k)
    seeds=backward_structural_seeds(problem,DEFAULT_FUSED_MXFP8_BWD_CONFIG)
    assert all(c.implementation_rejection(problem) is None for c in seeds)
    assert any(c.dx.gemm.scale_layout=='mma64x128' and c.dx.backend=='decomposed'
               and c.dw.backend=='decomposed' and c.dw.split_reduction>1 for c in seeds[:3])
    axes=backward_search_space(problem,BWD_SEARCH_SPACE)
    assert any(c['dx'].get('gemm',{}).get('scale_layout')=='mma64x128' for c in axes['dx_backend'])


@pytest.mark.skipif(not torch.cuda.is_available(),reason='RTX Blackwell required')
@pytest.mark.parametrize('shape',[(192,128,128),(256,128,128),(3648,512,256),
                                  (76608,256,256),(310080,256,128),(76608,64,256),(17,29,33)])
def test_default_compiled_training_across_alignment_boundary(shape):
    from rtx._bwd_search import backward_structural_seeds
    from rtx.kernels.mxfp8_bwd import DEFAULT_FUSED_MXFP8_BWD_CONFIG
    torch._dynamo.reset();torch.manual_seed(31103)
    torch.set_float32_matmul_precision('highest')
    m,n,k=shape
    seeds=backward_structural_seeds(MXFP8Problem(m,n,k),DEFAULT_FUSED_MXFP8_BWD_CONFIG)
    bwd=next((c for c in seeds if c.dx.backend==c.dw.backend=='decomposed'),DEFAULT_FUSED_MXFP8_BWD_CONFIG)
    layer=rtx.MXFP8Linear(k,n,device='cuda',autotune=False,backward_config=bwd)
    x=(torch.randn(m,k,device='cuda',dtype=torch.bfloat16)*.1).requires_grad_()
    dy=torch.randn(m,n,device='cuda',dtype=torch.bfloat16)*.1
    fn=torch.compile(layer,fullgraph=True,dynamic=False)
    for iteration in range(3):
        if iteration==1:
            with torch.no_grad():x.add_(.001);layer.weight.add_(.0001)
        y=fn(x)
        dx,dw=torch.autograd.grad(y,(x,layer.weight),dy)
        with torch.no_grad():
            refs=(x.float()@layer.weight.float().T,dy.float()@layer.weight.float(),dy.float().T@x.float())
            for actual,reference in zip((y,dx,dw),refs):
                assert actual.isfinite().all()
                nrms=float((actual.float()-reference).norm()/reference.norm().clamp_min(1e-12))
                assert nrms<.08,(shape,nrms)
    if m==192:
        with pytest.raises(RuntimeError,match='activation quantizer'):
            rtx.mxfp8_linear(x,layer.weight,backend='materialized',
                            dynamic_config=DEFAULT_MXFP8_PREQUANT_CONFIG,autotune=False)
        # Forced materialized without an explicit schedule must be shape-aware too.
        with torch.no_grad():
            explicit=rtx.MXFP8Linear(k,n,device='cuda',backend='materialized',autotune=False)
            explicit.load_state_dict(layer.state_dict())
            torch.testing.assert_close(torch.compile(explicit,fullgraph=True,dynamic=False)(x),y,rtol=0,atol=0)
    torch.cuda.synchronize()
    torch._dynamo.reset()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='RTX Blackwell required')
@pytest.mark.parametrize('m',[192,76608])
def test_graph_replay_retains_explicit_native64_preparation(m):
    from rtx._bwd_search import backward_structural_seeds
    from rtx.kernels.mxfp8_bwd import DEFAULT_FUSED_MXFP8_BWD_CONFIG
    torch._dynamo.reset();torch.manual_seed(881)
    config=_default_dynamic_prequant_config(MXFP8Problem(m,128,128))
    bwd=next(c for c in backward_structural_seeds(MXFP8Problem(m,128,128),DEFAULT_FUSED_MXFP8_BWD_CONFIG)
             if c.dx.backend==c.dw.backend=='decomposed')
    layer=rtx.MXFP8Linear(128,128,device='cuda',dynamic_config=config,autotune='cache',
                         backward_config=bwd)
    x=(torch.randn(m,128,device='cuda',dtype=torch.bfloat16)*.1).requires_grad_()
    dy=torch.randn_like(x)*.1
    compiled=torch.compile(layer,fullgraph=True,dynamic=False,options={'triton.cudagraphs':True})
    live=[]
    for iteration in range(5):
        torch.compiler.cudagraph_mark_step_begin()
        y=compiled(x)
        dx,dw=torch.autograd.grad(y,(x,layer.weight),dy)
        with torch.no_grad():
            expected_y=rtx.mxfp8_linear(x,layer.weight,backend='materialized',dynamic_config=config,autotune=False)
            torch.testing.assert_close(y,expected_y,rtol=0,atol=0)
            assert dx.isfinite().all() and dw.isfinite().all()
        del y,dx,dw,expected_y
        torch.cuda.synchronize()
        if iteration>=3:live.append(torch.cuda.memory_allocated())
    assert len(set(live))==1,live
    torch._dynamo.reset()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='RTX Blackwell required')
def test_first_hit_search_reaches_both_families_at_half_alignment(tmp_path):
    import json
    from rtx._bwd_search import backward_structural_seeds
    from rtx.kernels.mxfp8_bwd import DEFAULT_FUSED_MXFP8_BWD_CONFIG
    torch._dynamo.reset();torch.manual_seed(617)
    problem=MXFP8Problem(192,128,128)
    bwd=backward_structural_seeds(problem,DEFAULT_FUSED_MXFP8_BWD_CONFIG)[1]
    policy=rtx.AutotunePolicy(max_trials=2,max_seconds=60,confirmation_repeats=0,
                             preparation_selection=True)
    layer=rtx.MXFP8Linear(128,128,device='cuda',autotune='balanced',tuning_policy=policy,
                         autotune_cache_dir=tmp_path,backward_config=bwd)
    x=(torch.randn(192,128,device='cuda',dtype=torch.bfloat16)*.1).requires_grad_()
    dy=torch.randn_like(x)*.1
    function=torch.compile(layer,fullgraph=True,dynamic=False)
    for _ in range(2):
        y=function(x)
        gradients=torch.autograd.grad(y,(x,layer.weight),dy)
        assert all(g.isfinite().all() for g in gradients)
    path=tmp_path/'preparation'/'observations.jsonl'
    before=path.read_bytes()
    records=[json.loads(line) for line in before.splitlines()]
    assert {row['config']['implementation'] for row in records}=={'fused','materialized'}
    assert any(row['config']['implementation']=='materialized' and
               row['config']['config']['gemm']['scale_layout']=='mma64x128' for row in records)
    function(x)
    assert path.read_bytes()==before
    torch._dynamo.reset()
