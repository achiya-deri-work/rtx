from dataclasses import asdict, replace
import json

import pytest
import torch

from rtx.bwd_autotune import BWD_SEARCH_SPACE, bwd_config_from_dict, bwd_config_to_dict, update_bwd_config
from rtx.kernels.mxfp8 import MXFP8Problem
from rtx.kernels.mxfp8_bwd import DEFAULT_SEPARATE_DECOMPOSED_MXFP8_BWD_CONFIG as BASE


def test_serial_storage_legacy_identity_and_search():
    legacy=asdict(BASE)
    del legacy['operand_storage']
    assert bwd_config_to_dict(BASE)==legacy
    assert bwd_config_from_dict(legacy)==BASE
    move={'operand_storage':'serial_reuse'}
    assert move in BWD_SEARCH_SPACE['operand_storage']
    cfg=update_bwd_config(BASE,move)
    assert bwd_config_from_dict(bwd_config_to_dict(cfg))==cfg
    from rtx.fp8_bwd import _intern_bwd_config
    assert json.loads(_intern_bwd_config(BASE).split(':',1)[1])==legacy
    from rtx.mlp import MXFP8MLPConfig
    old=MXFP8MLPConfig(gate_backward=BASE).to_dict()
    assert 'operand_storage' not in old['gate_backward']
    new=replace(MXFP8MLPConfig(gate_backward=BASE),gate_backward=cfg)
    assert MXFP8MLPConfig.from_dict(new.to_dict())==new


def test_composite_identity_omits_default_and_inactive_partial_reuse():
    from rtx.mlp import MXFP8MLPConfig
    from rtx.autotune.mlp import active_schedule
    c=MXFP8MLPConfig(gate_backward=BASE,down_backward=BASE,backward_recompute='joint')
    original=active_schedule(c,'mxfp8','train')
    assert 'operand_storage' not in original['gate_backward']
    assert 'operand_storage' not in original['down_backward']
    down=replace(c,down_backward=replace(BASE,operand_storage='serial_reuse'))
    assert active_schedule(down,'mxfp8','train')==original
    gate=replace(c,gate_backward=replace(BASE,operand_storage='serial_reuse'))
    assert active_schedule(gate,'mxfp8','train')!=original
    shared=replace(c,down_backward=replace(BASE,operand_storage='serial_reuse_partials'))
    assert active_schedule(shared,'mxfp8','train')!=original
    assert bwd_config_from_dict(bwd_config_to_dict(shared.down_backward))==shared.down_backward
    assert {'operand_storage':'serial_reuse_partials'} in BWD_SEARCH_SPACE['operand_storage']


def test_shared_g_row_major_move_exits_serial_storage_legally():
    initial = replace(BASE, operand_storage='serial_reuse',
                      dx=replace(BASE.dx, gemm=replace(BASE.dx.gemm, epilogue='direct', store_vec=1)))
    problem = MXFP8Problem(11712,1536,256)
    moves = BWD_SEARCH_SPACE['shared_g_row_major']
    assert len(moves) == 12
    assert {move['dx']['gemm']['scale_role'] for move in moves} == {'producer','consumers'}
    for move in moves:
        config = update_bwd_config(initial, move)
        assert config.operand_storage == 'separate'
        assert config.quant_schedule == 'shared_g_quad'
        rejection = config.implementation_rejection(problem)
        if problem.m % config.dx.quant_a.transposed_tile_rows:
            assert 'full SMEM tiles' in rejection
        else:
            assert rejection is None
        for leg in (config.dx, config.dw):
            assert leg.quant_a.scale_layout == leg.gemm.scale_layout == 'row_major'
            assert leg.quant_a.scale_rounding == initial.dx.quant_a.scale_rounding


@pytest.mark.parametrize('change',[
    {'stream_schedule':'dual_stream'}, {'execution_order':'interleaved'},
    {'quant_schedule':'quad'}, {'quant_schedule':'shared_g_quad'},
    {'dx':replace(BASE.dx,backend='fused')}, {'operand_storage':'invalid'},
])
@pytest.mark.parametrize('storage',['serial_reuse','serial_reuse_partials'])
def test_overlapping_lifetimes_rejected(change,storage):
    cfg=replace(BASE,operand_storage=storage)
    assert replace(cfg,**change).implementation_rejection(MXFP8Problem(512,256,128))


def test_shared_views_have_distinct_roles_and_exact_storage_bound():
    from rtx.fp8_bwd import _serial_operand_views
    from rtx.runtime import tensor_storage_bytes
    a,b=_serial_operand_views(MXFP8Problem(512,128,256),MXFP8Problem(256,128,512),
                             BASE.dx,BASE.dw,torch.device('cpu'))
    assert len({v.data_ptr() for v in a})==4
    for left,right in zip(a,b):
        assert left.data_ptr()==right.data_ptr()
        assert left.is_contiguous() and right.is_contiguous()
    assert tensor_storage_bytes((a,b))==sum(max(x.numel(),y.numel()) for x,y in zip(a,b))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('m',[128,512])
@pytest.mark.parametrize('budget',[0,64*1024*1024])
def test_opt_in_partial_pair_cache_accounting(m,budget,monkeypatch):
    import rtx.fp8_bwd as impl
    from rtx.runtime import clear_runtime_caches
    clear_runtime_caches(preserve_compiled=True)
    monkeypatch.setattr(impl._RUNNERS,'max_bytes',budget)
    dy=torch.randn(m,256,device='cuda',dtype=torch.bfloat16)*.1
    x=torch.randn(m,128,device='cuda',dtype=torch.bfloat16)*.1
    w=torch.randn(256,128,device='cuda',dtype=torch.bfloat16)*.02
    base_key=impl._intern_bwd_config(BASE)
    shared_key=impl._intern_bwd_config(replace(BASE,operand_storage='serial_reuse_partials'))
    expected=(torch.empty_like(x),torch.empty_like(w))
    actual=(torch.empty_like(x),torch.empty_like(w))
    for generation in range(3):
        dy.mul_(.99);x.add_(.001);w.add_(.0001)
        impl._launch_dx_out(dy,x,w,base_key,expected[0])
        impl._launch_dw_out(dy,x,w,base_key,expected[1])
        # A single-leg call remains supported, even with zero retention budget.
        impl._launch_dx_out(dy,x,w,shared_key,actual[0])
        impl._launch_dw_out(dy,x,w,shared_key,actual[1])
        for a,b in zip(actual,expected):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert len(impl._RUNNERS)==int(budget>0)
    assert len(impl._MATMUL_RUNNERS)==2  # legacy baseline only
    assert impl._RUNNERS.stats()['resident_bytes']<=budget
    torch.cuda.synchronize()
    clear_runtime_caches(preserve_compiled=True)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('order',['dx_first','dw_first'])
@pytest.mark.parametrize('launches',['separate','dual'])
def test_serial_storage_gradients_graph_cleanup(order,launches):
    from rtx.fp8_bwd import _build_bwd_runner, _intern_bwd_config, _launch_bwd_out
    from rtx.runtime import clear_runtime_caches, retain_runtime_resources, tensor_storage_bytes
    p=MXFP8Problem(512,256,128)
    base=replace(BASE,execution_order=order,
                 dx=replace(BASE.dx,quant_launches=launches),
                 dw=replace(BASE.dw,quant_launches=launches))
    cfg=replace(base,operand_storage='serial_reuse')
    assert cfg.implementation_rejection(p) is None
    ordinary=_build_bwd_runner(p,base,torch.device('cuda:0'))
    reused=_build_bwd_runner(p,cfg,torch.device('cuda:0'))
    expected_saving=sum(min(a.numel(),b.numel()) for a,b in zip(
        (ordinary.dx.quantized_a,ordinary.dx.quantized_b,ordinary.dx.scales_a,ordinary.dx.scales_b),
        (ordinary.dw.quantized_a,ordinary.dw.quantized_b,ordinary.dw.scales_a,ordinary.dw.scales_b)))
    assert tensor_storage_bytes(ordinary)-tensor_storage_bytes(reused)==expected_saving
    torch.manual_seed(701)
    dy=torch.randn(p.m,p.n,device='cuda',dtype=torch.bfloat16)*.1
    x=torch.randn(p.m,p.k,device='cuda',dtype=torch.bfloat16)*.1
    w=torch.randn(p.n,p.k,device='cuda',dtype=torch.bfloat16)*.02
    out=(torch.empty_like(x),torch.empty_like(w))
    baseline=(torch.empty_like(x),torch.empty_like(w))
    key=_intern_bwd_config(cfg)
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _launch_bwd_out(dy,x,w,key,*out)
    stream.synchronize()
    graph=torch.cuda.CUDAGraph()
    with retain_runtime_resources() as owners:
        with torch.cuda.graph(graph,stream=stream):
            _launch_bwd_out(dy,x,w,key,*out)
    assert owners
    clear_runtime_caches(preserve_compiled=True)
    def q(t):
        blocks=t.float().reshape(t.shape[0],-1,32)
        scales=torch.exp2(torch.ceil(torch.log2((blocks.abs().amax(-1,keepdim=True)/448).clamp_min(2.**-126))))
        return ((blocks/scales).clamp(-448,448).to(torch.float8_e4m3fn).float()*scales).reshape(t.shape)
    for generation in range(3):
        if generation:
            dy.mul_(.97);x.add_(.001);w.add_(.0001)
        ordinary(dy,x,w,*baseline)
        expected=((q(dy)@q(w.T.contiguous()).T).bfloat16(),
                  (q(dy.T.contiguous())@q(x.T.contiguous()).T).bfloat16())
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(4): graph.replay()
        stream.synchronize()
        for actual,control,ref in zip(out,baseline,expected):
            assert torch.equal(actual,control)
            assert bool(actual.isfinite().all())
            error=float((actual.float()-ref.float()).norm()/ref.float().norm().clamp_min(1e-12))
            assert error<.0002
