"""Simple layer usage; kernel machinery stays below the frontend."""
import json
import pytest
import torch
import rtx


def layer(**kwargs):
    return rtx.BalancedMoE(16,8,16,top_k=2,shared_hidden_features=16,router_dim=4,**kwargs)


@pytest.mark.parametrize('effort,trials',[(None,16),('quick',8),('balanced',16),('thorough',32),('extensive',64)])
def test_one_effort_controls_all_components(effort,trials):
    model=layer(autotune=effort)
    for spec in (model._expert_spec,model._shared_spec,model.router._selection_spec):
        assert json.loads(spec)['policy']['max_trials']==trials
    assert model.autotune==model.router.autotune=='balanced'
    assert f'max_trials={trials}' in repr(model) and '\n' not in repr(model)


def test_advanced_and_existing_explicit_config_remain_supported():
    model=layer(config=rtx.MXFP8MoEConfig())
    assert model.autotune==model.router.autotune=='off' and model._shared_spec is None
    policy=rtx.AutotunePolicy.preset('quick',max_trials=3)
    model=layer(tuning_policy=policy)
    assert model.tuning_policy is model.router.tuning_policy is policy
    with pytest.raises(ValueError,match='not both'):layer(autotune='quick',tuning_policy=policy)
    for k in (True,0,9):
        with pytest.raises(ValueError,match='top_k'):
            rtx.BalancedMoE(16,8,16,top_k=k,shared_hidden_features=16)


@pytest.mark.parametrize('shared',[0,1,2])
def test_dense_equals_packed_with_all_cpu_gradients(shared):
    model=layer(backend='torch',num_shared_experts=shared)
    x=torch.randn(2,5,16,dtype=torch.bfloat16,requires_grad=True)
    leaves=(x,*model.parameters());dy=torch.randn_like(x)
    ids=[id(p) for p in model.parameters()];keys=list(model.state_dict())
    y=model(x);dense=(y,*torch.autograd.grad(y,leaves,dy))
    cu=torch.tensor([0,5,10],dtype=torch.int32)
    expected=model(x.view(-1,16),cu_seqlens=cu).view_as(x)
    packed=(expected,*torch.autograd.grad(expected,leaves,dy))
    for a,b in zip(dense,packed,strict=True):torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert y.shape==x.shape and y.dtype==torch.bfloat16
    assert [id(p) for p in model.parameters()]==ids and list(model.state_dict())==keys


def test_dense_samples_independent_and_flat_unchanged():
    model=layer(backend='torch');x=torch.randn(2,5,16,dtype=torch.bfloat16)
    y=model(x);changed=x.clone();changed[1]*=3
    torch.testing.assert_close(model(changed)[0],y[0],rtol=0,atol=0)
    assert model(x.view(-1,16)).shape==(10,16)


def test_reset_and_state_dict_keep_optimizer_parameters():
    model=layer(backend='torch');identities=[id(p) for p in model.parameters()]
    with torch.no_grad():model.router.projection_weight.fill_(.1)
    model.reset_expert_parameters()
    assert bool((model.router.projection_weight==.1).all())
    model.reset_parameters()
    assert not model.router.projection_weight.count_nonzero()
    assert identities==[id(p) for p in model.parameters()]
    for bank in (model.gate_up,model.down,model.shared_gate_up,model.shared_down):
        assert all(p.dtype==torch.bfloat16 and bool((p.abs()<=.04).all()) for p in bank.parameters())
    other=layer(backend='torch');other.load_state_dict(model.state_dict())
    x=torch.randn(2,5,16,dtype=torch.bfloat16)
    torch.testing.assert_close(model(x),other(x),rtol=0,atol=0)


def test_invalid_inputs_have_frontend_errors():
    model=layer(backend='torch')
    for shape in ((16,),(0,16),(2,0,16),(2,4,15),(2,3,4,16)):
        with pytest.raises(ValueError,match='expects nonempty'):model(torch.empty(shape,dtype=torch.bfloat16))
    with pytest.raises(ValueError,match='contiguous BF16'):model(torch.randn(4,16))
    with pytest.raises(ValueError,match='contiguous BF16'):model(torch.empty(16,4,dtype=torch.bfloat16).T)
    with pytest.raises(ValueError,match='packed 2D'):
        model(torch.empty(2,5,16,dtype=torch.bfloat16),cu_seqlens=torch.tensor([0,5,10]))


cuda=pytest.mark.skipif(not torch.cuda.is_available(),reason='RTX CUDA required')


def _close(actual,expected):
    for a,b in zip(actual,expected,strict=True):
        a,b=a.detach(),b.detach()
        delta=a.float()-b.float()
        assert bool(a.isfinite().all())
        assert float(delta.norm()/b.float().norm().clamp_min(1e-12))<.006 or float(delta.abs().max())<2e-7


@cuda
def test_dense_compile_and_changed_input_cuda_graph():
    model=rtx.BalancedMoE(64,8,64,top_k=2,shared_hidden_features=64,device='cuda',autotune='off')
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        x=torch.randn(2,17,64,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        cu=torch.tensor([0,17,34],device='cuda',dtype=torch.int32)
        dy=torch.randn_like(x);leaves=(x,*model.parameters())
        fn=torch.compile(model,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
        packed=torch.compile(lambda t:model(t.view(-1,64),cu_seqlens=cu).view_as(t),
                             fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
        def step(call):
            y=call(x);return y,*torch.autograd.grad(y,leaves,dy)
        for _ in range(3):step(fn);step(packed)
        _close(step(fn),step(packed))
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):actual=step(fn)
        for _ in range(2):
            with torch.no_grad():x.add_(.01*torch.randn_like(x))
            graph.replay();stream.synchronize();_close(actual,step(packed))


@cuda
def test_cold_compiled_dense_layer_needs_no_prepare(tmp_path):
    policy=rtx.AutotunePolicy.preset('quick',max_trials=2,max_seconds=90,
                                   samples=3,warmup=1,confirmation_repeats=0)
    model=rtx.BalancedMoE(64,8,64,top_k=2,shared_hidden_features=64,device='cuda',
        tuning_policy=policy,autotune_cache_dir=tmp_path)
    fn=torch.compile(model,fullgraph=True,dynamic=False)
    x=torch.randn(2,32,64,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    for _ in range(3):
        model.zero_grad(set_to_none=True);x.grad=None
        y=fn(x);assert y.shape==x.shape
        y.float().square().mean().backward()
        assert all(p.grad is not None and bool(p.grad.isfinite().all()) for p in (x,*model.parameters()))
    rows=[json.loads(line) for path in tmp_path.rglob('observations.jsonl') for line in path.read_text().splitlines()]
    assert {'balanced_varlen_routing','mxfp8_balanced_experts','mxfp8_recompute_mlp'}<={r['family'] for r in rows}
    before={str(p):p.read_bytes() for p in tmp_path.rglob('observations.jsonl')}
    fn(x).float().sum().backward()
    assert before=={str(p):p.read_bytes() for p in tmp_path.rglob('observations.jsonl')}


@cuda
@pytest.mark.parametrize('phase',['train','forward'])
def test_optional_all_component_tuning_preserves_grads_and_binds_hot(tmp_path,monkeypatch,phase):
    model=rtx.BalancedMoE(64,8,64,top_k=2,shared_hidden_features=64,
                          device='cuda',autotune='off').eval()
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        x=torch.randn(2,32,64,device='cuda',dtype=torch.bfloat16)
        for p in model.parameters():p.grad=torch.full_like(p,.125)
        previous=[(p.grad,p.grad.clone()) for p in model.parameters()]
        policy=rtx.AutotunePolicy.preset('quick',max_trials=2,max_seconds=90,samples=3,warmup=1,confirmation_repeats=0)
        with torch.inference_mode() if phase=='forward' else torch.no_grad():
            assert model.autotune_(x,tuning_policy=policy,output_dir=tmp_path,phase=phase) is model
        assert not model.training and not x.requires_grad
        for p,(original,values) in zip(model.parameters(),previous,strict=True):
            assert p.grad is original;torch.testing.assert_close(p.grad,values,rtol=0,atol=0)
        rows=[json.loads(line) for path in tmp_path.rglob('observations.jsonl') for line in path.read_text().splitlines()]
        assert {'balanced_varlen_routing','mxfp8_balanced_experts','mxfp8_recompute_mlp'}<={r['family'] for r in rows}
        if phase=='train':
            before={str(p):p.read_bytes() for p in tmp_path.rglob('observations.jsonl')}
            assert model.autotune_(x) is model
            assert model.tuning_policy.cache_behavior=='refine'
            assert before=={str(p):p.read_bytes() for p in tmp_path.rglob('observations.jsonl')}
            for p,(original,values) in zip(model.parameters(),previous,strict=True):
                assert p.grad is original;torch.testing.assert_close(p.grad,values,rtol=0,atol=0)
        from rtx import _routing_varlen_autotune,_mlp_autotune
        from rtx.autotune import moe
        def forbidden(*args,**kw):pytest.fail('Prepared shape selected again')
        monkeypatch.setattr(_routing_varlen_autotune,'_resolve',forbidden)
        monkeypatch.setattr(_mlp_autotune,'_select',forbidden)
        monkeypatch.setattr(moe,'resolve_config',forbidden)
        fn=torch.compile(model,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
        with torch.set_grad_enabled(phase=='train'):
            for _ in range(2):assert fn(x).shape==x.shape
        if phase=='forward':
            with torch.inference_mode():assert fn(x).shape==x.shape
