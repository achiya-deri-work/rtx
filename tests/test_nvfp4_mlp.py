"""NVFP4 SiTU semantics, compiler ABI, and recomputation coverage."""
from dataclasses import replace

import pytest
import torch

import rtx
from rtx.configs.nvfp4_mlp import NVFP4MLPConfig


def test_config_roundtrip():
    c=NVFP4MLPConfig(beta=3,linear_beta=20)
    assert NVFP4MLPConfig.from_dict(c.to_dict())==c
    assert NVFP4MLPConfig.from_json(c.to_json())==c
    assert c.rejection(192,128,256,128) is None
    with pytest.raises(ValueError,match="positive"):
        NVFP4MLPConfig(beta=0)
    with pytest.raises(ValueError,match="positive"):
        NVFP4MLPConfig(beta=torch.tensor(4.))


def test_api_rejects_unsupported_activation_and_cpu_tuning():
    with pytest.raises(ValueError,match="situ"):
        rtx.NVFP4MLP(64,128,activation='silu',device='meta')
    layer=rtx.NVFP4MLP(64,128,device='meta')
    with pytest.raises(ValueError,match="real CUDA"):
        layer.autotune_(torch.empty(128,64,device='meta',dtype=torch.bfloat16))
    with pytest.raises(ValueError,match="situ"):
        rtx.nvfp4_mlp(torch.empty(128,64,device='meta',dtype=torch.bfloat16),
            layer.gate_up_weight,layer.down_weight,activation='silu')


def test_half_aligned_backward_has_complete_materialized_openings():
    from rtx.autotune.nvfp4_mlp import resolved,backward_openings
    c=resolved(NVFP4MLPConfig(),8128,768,1536,768)
    for b in (c.gate_backward,c.down_backward):
        assert b.dx.backend==b.dw.backend=='decomposed'
        assert b.dw.gemm.tile_k%128==0
    seeds=backward_openings(c,8128,768,1536,768)
    assert len(seeds)>=6
    assert all(s.rejection(8128,768,1536,768) is None for s in seeds)
    assert seeds[0].gate_backward.dw.split_reduction==seeds[0].down_backward.dw.split_reduction==4


def test_immutable_backward_metadata_is_cached():
    from rtx.fp4_mlp import _key,_backward_keys
    _backward_keys.cache_clear()
    key=_key(NVFP4MLPConfig())
    first=_backward_keys(key,8128,768,1536,768)
    info=_backward_keys.cache_info()
    assert _backward_keys(key,8128,768,1536,768)==first
    assert _backward_keys.cache_info().hits==info.hits+1


def test_search_preserves_numerical_policy():
    from rtx.autotune.nvfp4_mlp import search_space,update,active_schedule
    from rtx.fp8_bwd import _default_bwd_config
    from rtx.kernels.mxfp8 import MXFP8Problem
    c=replace(NVFP4MLPConfig(),gate_backward=_default_bwd_config(MXFP8Problem(192,256,128)),
              down_backward=_default_bwd_config(MXFP8Problem(192,128,128)))
    axes=search_space(192,128,128,128)
    assert all(not any(s in axis for s in ('region_rows','outer_scale_math','fine_region_geometry')) for axis in axes)
    assert {'glu_epilogue','backward_overlap','gate_up.gemm_geometry','down.gemm_geometry'}<=axes.keys()
    for name,values in axes.items():
        result=update(c,name,values[-1])
        assert result.beta==c.beta and result.linear_beta==c.linear_beta
        assert result.gate_up.x_scale_region_rows==c.gate_up.x_scale_region_rows
    assert active_schedule(c)==active_schedule(replace(c,glu_store_vec=8))


def test_meta():
    layer=rtx.NVFP4MLP(128,256,device="meta")
    x=torch.empty((2,96,128),device="meta",dtype=torch.bfloat16,requires_grad=True)
    y=layer(x)
    assert y.shape==(2,96,128)
    dx,dg,dd=torch.autograd.grad(y.sum(),(x,layer.gate_up_weight,layer.down_weight))
    assert dx.shape==x.shape and dg.shape==layer.gate_up_weight.shape and dd.shape==layer.down_weight.shape
    # Exercise Dynamo's configuration ABI without needing CUDA codegen.
    compiled=torch.compile(layer,backend='eager',fullgraph=True,dynamic=False)
    assert compiled(x).shape==y.shape
    torch._dynamo.reset()


def test_autograd_saves_no_hidden_activation():
    layer=rtx.NVFP4MLP(64,128,device='meta')
    x=torch.empty((2,96,64),device='meta',dtype=torch.bfloat16,requires_grad=True)
    saved=[]
    with torch.autograd.graph.saved_tensors_hooks(lambda t:(saved.append(t),t)[1],lambda t:t):
        y=layer(x)
    assert len(saved)==3
    assert all(a is b for a,b in zip(saved,(x,layer.gate_up_weight,layer.down_weight)))
    assert y.shape==(2,96,64)


def test_conversion_and_policy(tmp_path):
    g=torch.nn.Linear(64,128,bias=False,dtype=torch.bfloat16)
    u=torch.nn.Linear(64,128,bias=False,dtype=torch.bfloat16)
    d=torch.nn.Linear(128,64,bias=False,dtype=torch.bfloat16)
    before=torch.random.get_rng_state().clone()
    layer=rtx.NVFP4MLP.from_linears(g,u,d)
    assert torch.equal(before,torch.random.get_rng_state())
    torch.testing.assert_close(layer.gate_up_weight,torch.cat((g.weight,u.weight)))
    assert layer.down_weight.data_ptr()!=d.weight.data_ptr()
    cached=rtx.NVFP4MLP(64,128,device='meta',autotune='cache',autotune_cache_dir=tmp_path)
    assert cached.export_tuning_spec()['mode']=='cache'
    assert cached.require_cached_().export_tuning_spec()['cache_miss']=='error'


def _situ(g,u,c):
    g,u=g.float(),u.float()
    return c.beta*torch.tanh(g/c.beta)*torch.sigmoid(g)*c.linear_beta*torch.tanh(u/c.linear_beta)


@pytest.mark.skipif(not torch.cuda.is_available(),reason="RTX Blackwell required")
@pytest.mark.parametrize("shape",[(128,128,256,128),(192,128,128,64),(257,64,64,128)])
def test_forward_projection_and_recompute(shape):
    from rtx.fp4_mlp import _projection
    from rtx.fp4 import nvfp4_linear
    m,k,h,o=shape
    torch.manual_seed(303)
    c=replace(NVFP4MLPConfig(),hidden_dtype='bf16')
    x=torch.randn(m,k,device="cuda",dtype=torch.bfloat16)*.1
    gu=torch.randn(2*h,k,device="cuda",dtype=torch.bfloat16)*.1
    down=torch.randn(o,h,device="cuda",dtype=torch.bfloat16)*.1
    projection=nvfp4_linear(x,gu,scaling="jit_row_region",dynamic_config=c.gate_up,autotune=False)
    g,u=projection.chunk(2,-1)
    hidden=x.new_empty(m,h)
    _projection(x,gu,c)(x,gu,hidden)
    # The oracle here has rounded GEMM outputs; the fused contract retains FP32.
    torch.testing.assert_close(hidden,_situ(g,u,c).bfloat16(),atol=.0015,rtol=.04)
    dy=torch.randn(m,h,device="cuda",dtype=torch.bfloat16)*.1
    dp=x.new_empty(m,2*h);recomputed=x.new_empty(m,h)
    _projection(x,gu,c,True)(x,gu,dp,dy,recomputed)
    torch.testing.assert_close(recomputed,hidden,atol=0,rtol=0)
    gf=g.float().requires_grad_();uf=u.float().requires_grad_()
    grads=torch.autograd.grad(_situ(gf,uf,c),(gf,uf),dy.float())
    torch.testing.assert_close(dp,torch.cat(grads,-1).bfloat16(),atol=.002,rtol=.05)
    expected=nvfp4_linear(hidden,down,scaling="jit_row_region",dynamic_config=c.down,autotune=False)
    actual=rtx.nvfp4_mlp(x,gu,down,config=c)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(),reason="RTX Blackwell required")
@pytest.mark.parametrize('ownership',['warp','cta','cta_cached'])
@pytest.mark.parametrize('shape',[(192,192,320,192),(17,33,65,29)])
def test_packed_hidden_and_all_down_ownerships(ownership,shape):
    from rtx.fp4_mlp import _projection
    torch.manual_seed(315)
    m,k,h,o=shape
    base=NVFP4MLPConfig()
    c=replace(base,down=replace(base.down,region_ownership=ownership),hidden_quant_vec=8,
              glu_scratch_swizzle='xor_16b')
    x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)*.1
    gu=torch.randn(2*h,k,device='cuda',dtype=torch.bfloat16)*.1
    down=torch.randn(o,h,device='cuda',dtype=torch.bfloat16)*.1
    width=(h+31)//32*32
    q=torch.empty((m,width),device='cuda',dtype=torch.float8_e4m3fn)
    s=torch.empty((m,width//32),device='cuda',dtype=torch.uint8)
    _projection(x,gu,c)(x,gu,q,hidden_scales=s)
    # Verify producer output and the packed-source preparation separately.
    bf16=x.new_empty((m,h))
    _projection(x,gu,replace(c,hidden_dtype='bf16'))(x,gu,bf16)
    decoded=(q.float()*s.view(torch.float8_e8m0fnu).float().repeat_interleave(32,-1))[:,:h]
    assert bool(decoded.isfinite().all())
    assert float((decoded-bf16.float()).norm()/bf16.float().norm().clamp_min(1e-12))<.06
    expected=rtx.nvfp4_linear(decoded.bfloat16(),down,scaling='jit_row_region',dynamic_config=c.down,autotune=False)
    actual=rtx.nvfp4_mlp(x,gu,down,config=c)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(),reason="RTX Blackwell required")
@pytest.mark.parametrize('shape',[(128,128,256,128),(192,128,128,64),(192,192,320,192),(17,33,65,29),(1,1,1,1)])
def test_independent_oracle_and_compiled_gradients(shape):
    from rtx._nvfp4_mlp_reference import reference
    from rtx.fp4_mlp import _key,_mlp_op
    torch._dynamo.reset();torch.manual_seed(931)
    torch.set_float32_matmul_precision('highest')
    m,k,h,o=shape
    layer=rtx.NVFP4MLP(k,h,o,device='cuda')
    x=(torch.randn(m,k,device='cuda',dtype=torch.bfloat16)*.2).requires_grad_()
    dy=torch.randn(m,o,device='cuda',dtype=torch.bfloat16)*.1
    if m==192:
        dy=dy.repeat_interleave(2,-1)[:,::2]  # Noncontiguous upstream gradient.
    fn=torch.compile(layer,fullgraph=True,dynamic=False)
    for i in range(3):
        if i==1:
            with torch.no_grad():
                x.add_(.01);layer.gate_up_weight.add_(.001)
        y=fn(x)
        grads=torch.autograd.grad(y,(x,layer.gate_up_weight,layer.down_weight),dy)
        expected=reference(x,layer.gate_up_weight,layer.down_weight,dy,layer.config)
        for actual,want in zip((y,*grads),expected):
            nrms=float((actual.detach().float()-want.float()).norm()/want.float().norm().clamp_min(1e-12))
            assert bool(actual.isfinite().all()) and nrms<.025,(shape,i,nrms)
    if m==128:
        torch.library.opcheck(_mlp_op,(x,layer.gate_up_weight,layer.down_weight,_key(layer.config)))
    torch._dynamo.reset()


@pytest.mark.skipif(not torch.cuda.is_available(),reason="RTX Blackwell required")
@pytest.mark.parametrize('overlap',[False,True])
def test_exchange_epilogue_and_partial_gradients(overlap):
    from rtx._nvfp4_mlp_reference import reference
    torch._dynamo.reset();torch.manual_seed(49)
    c=replace(NVFP4MLPConfig(),hidden_dtype='bf16',glu_epilogue='exchange',glu_scratch_swizzle='xor_16b',
              glu_store_vec=4,backward_overlap=overlap)
    layer=rtx.NVFP4MLP(64,128,64,config=c,device='cuda')
    x=(torch.randn(192,64,device='cuda',dtype=torch.bfloat16)*.2).requires_grad_()
    dy=torch.randn(192,64,device='cuda',dtype=torch.bfloat16)*.1
    expected=reference(x,layer.gate_up_weight,layer.down_weight,dy,c)
    for mask in ((True,True,True),(True,False,False),(False,True,False),(False,False,True)):
        args=tuple(t.detach().requires_grad_(need) for t,need in zip((x,layer.gate_up_weight,layer.down_weight),mask))
        y=rtx.nvfp4_mlp(*args,config=c)
        grads=torch.autograd.grad(y,tuple(t for t in args if t.requires_grad),dy[:,::1])
        for a,r in zip((y,*grads),(expected[0],*(r for r,need in zip(expected[1:],mask) if need))):
            assert float((a.detach().float()-r.float()).norm()/r.float().norm().clamp_min(1e-12))<.025


@pytest.mark.skipif(not torch.cuda.is_available(),reason="RTX Blackwell required")
@pytest.mark.parametrize('graphs',[False,True])
def test_first_hit_compile_cache_and_outstanding_forwards(tmp_path,graphs):
    import json
    from rtx.configs.nvfp4_mlp import NVFP4MLPConfig
    torch._dynamo.reset();torch.manual_seed(307)
    policy=rtx.AutotunePolicy(max_trials=2,max_seconds=90,warmup=1,samples=5,calls_per_sample=100,
                              confirmation_repeats=0,exploration_trials=0)
    layer=rtx.NVFP4MLP(64,128,device='cuda',autotune='balanced',tuning_policy=policy,autotune_cache_dir=tmp_path)
    x=torch.randn(128,64,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    compiled=torch.compile(layer,fullgraph=True,dynamic=False,options={'triton.cudagraphs':graphs})
    first=compiled(x)
    second=compiled(x+.02)
    refs=list(tmp_path.rglob('runtime_winners/nvfp4_recompute_mlp/**/*.json'))
    assert len(refs)==1
    c=NVFP4MLPConfig.from_dict(json.loads(refs[0].read_text())['config'])
    expected=rtx.nvfp4_mlp(x,layer.gate_up_weight,layer.down_weight,config=c)
    torch.testing.assert_close(first,expected,atol=0,rtol=0)
    for y in (second,first):
        grads=torch.autograd.grad(y,(x,layer.gate_up_weight,layer.down_weight),torch.ones_like(y))
        assert all(bool(g.isfinite().all()) for g in grads)
        direct=rtx.nvfp4_mlp(x+.02 if y is second else x,layer.gate_up_weight,layer.down_weight,config=c)
        expected_gradients=torch.autograd.grad(direct,(x,layer.gate_up_weight,layer.down_weight),torch.ones_like(direct))
        for a,b in zip(grads,expected_gradients):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
    observations={p:p.read_bytes() for p in tmp_path.rglob('observations*.jsonl')}
    compiled(x)
    assert all(p.read_bytes()==before for p,before in observations.items())
    torch._dynamo.reset()


@pytest.mark.skipif(not torch.cuda.is_available(),reason="RTX Blackwell required")
def test_compiled_training_and_bounded_live_memory():
    torch._dynamo.reset();torch.manual_seed(73)
    layer=rtx.NVFP4MLP(64,128,64,device='cuda')
    compiled=torch.compile(layer,fullgraph=True,dynamic=False)
    x=torch.randn(128,64,device='cuda',dtype=torch.bfloat16)
    target=torch.randn(128,64,device='cuda',dtype=torch.bfloat16)*.1
    optimizer=torch.optim.AdamW(layer.parameters(),lr=.01)
    losses=[];live=[]
    for step in range(10):
        optimizer.zero_grad(set_to_none=True)
        y=compiled(x)
        loss=(y.float()-target.float()).square().mean()
        loss.backward();optimizer.step()
        losses.append(float(loss.detach()))
        assert all(bool(p.grad.isfinite().all()) for p in layer.parameters())
        del y,loss
        torch.cuda.synchronize()
        if step>=6:
            live.append(torch.cuda.memory_allocated())
    assert losses[-1]<losses[0],losses
    assert len(set(live))==1,live
    torch._dynamo.reset()


@pytest.mark.skipif(not torch.cuda.is_available(),reason="RTX Blackwell required")
def test_live_graph_survives_runtime_scratch_cleanup():
    from rtx.runtime import clear_runtime_caches
    torch.manual_seed(113)
    layer=rtx.NVFP4MLP(64,128,device='cuda')
    x=torch.randn(128,64,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    dy=torch.randn_like(x)
    def pair():
        y=layer(x)
        gradients=torch.autograd.grad(y,(x,layer.gate_up_weight,layer.down_weight),dy)
        return y.detach(),*(g.detach() for g in gradients)
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        pair();pair()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):
            captured=pair()
    torch.cuda.current_stream().wait_stream(stream)
    clear_runtime_caches(device=x.device,preserve_compiled=True)
    with torch.no_grad():
        x.add_(.02)
    graph.replay()
    expected=pair()
    for a,b in zip(captured,expected):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    graph.reset()


@pytest.mark.skipif(not torch.cuda.is_available(),reason="RTX Blackwell required")
@pytest.mark.parametrize('policy',['amax','four_over_six','mse5','lsq1'])
def test_adaptive_forward_policies(policy):
    from rtx._nvfp4_mlp_reference import reference
    torch.manual_seed(116)
    c=NVFP4MLPConfig()
    c=replace(c,gate_up=replace(c.gate_up,quant=replace(c.gate_up.quant,block_scale_selection=policy)),
              down=replace(c.down,quant=replace(c.down.quant,block_scale_selection=policy)))
    layer=rtx.NVFP4MLP(128,256,128,device='cuda',config=c)
    x=torch.randn(128,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    dy=torch.randn_like(x)*.1
    actual=layer(x)
    gradients=torch.autograd.grad(actual,(x,layer.gate_up_weight,layer.down_weight),dy)
    expected=reference(x,layer.gate_up_weight,layer.down_weight,dy,c)
    for a,r in zip((actual,*gradients),expected):
        assert bool(a.isfinite().all())
        assert float((a.detach().float()-r.float()).norm()/r.float().norm().clamp_min(1e-12))<.025


@pytest.mark.skipif(not torch.cuda.is_available(),reason="RTX Blackwell required")
def test_zero_input():
    layer=rtx.NVFP4MLP(128,256,128,device='cuda')
    x=torch.zeros(128,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    y=layer(x)
    gradients=torch.autograd.grad(y,(x,layer.gate_up_weight,layer.down_weight),torch.ones_like(y))
    assert int(y.count_nonzero())==0
    assert all(int(g.count_nonzero())==0 for g in gradients)
