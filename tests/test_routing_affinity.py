"""The two learned scalar adjoints remain present under row broadcasting."""

import pytest
import torch
import torch.nn.functional as F

from rtx._routing_affinity import ROW, SCALAR, affinity, log_sigmoid_row_broadcast


def _check_affinity(fn, device, shape):
    rows, experts = shape
    generator = torch.Generator(device="cpu").manual_seed(8879)
    logits = (6 * torch.randn(experts, rows, generator=generator).T).to(device).requires_grad_()
    gradient = torch.randn(experts, rows, generator=generator).T.to(device)
    scale = torch.tensor(-0.35, device=device, requires_grad=True)
    bias = torch.tensor(0.12, device=device, requires_grad=True)
    leaves = logits, scale, bias
    output = fn(*leaves)
    gradients = torch.autograd.grad(output, leaves, gradient)

    ordinary = F.logsigmoid(logits * scale.exp() + bias)
    torch.testing.assert_close(output, ordinary, rtol=2e-6, atol=2e-6)
    # Independent FP64 affine/logistic derivatives; use the production
    # upstream and include both scalar grads, not only d(logits).
    exact_logits = logits.detach().double()
    exact_scale = scale.detach().double().exp()
    exact_bias = bias.detach().double()
    dz = gradient.double() * torch.sigmoid(-(exact_logits * exact_scale + exact_bias))
    expected = (dz * exact_scale, (dz * exact_logits).sum() * exact_scale, dz.sum())
    for actual, reference in zip(gradients, expected):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, reference.float(), rtol=2e-5, atol=1e-5)


@pytest.mark.parametrize("shape", [(1, 1), (31, 24), (257, 48)])
def test_row_broadcast_affinity_cpu(shape):
    _check_affinity(log_sigmoid_row_broadcast, "cpu", shape)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("shape", [(7376, 32), (29504, 32)])
def test_row_broadcast_affinity_compiles(shape):
    compute = torch.compile(log_sigmoid_row_broadcast, fullgraph=True, dynamic=False,
                            options={"triton.cudagraphs": False})
    _check_affinity(compute, "cuda", shape)


def _whole_affinity(fn,device,dtype):
    torch.manual_seed(36891)
    x=(torch.randn(384,29,device=device,dtype=dtype).T*.3).requires_grad_()
    p=(torch.randn(16,384,device=device,dtype=dtype)*.05).requires_grad_()
    e=torch.randn(24,16,device=device,dtype=dtype).requires_grad_()
    s=torch.tensor(-.45,device=device,requires_grad=True)
    b=torch.tensor(.11,device=device,requires_grad=True)
    actual=(x,p,e,s,b);oracle=tuple(a.detach().double().requires_grad_() for a in actual)
    dx,dp,de,ds,db=oracle
    latent=dx@dp.T
    dot=(latent*torch.rsqrt(latent.square().mean(1,keepdim=True)+1e-6))@(
        de*torch.rsqrt(de.square().mean(1,keepdim=True)+1e-6)).T
    expected=-F.softplus(-(dot*ds.exp()+db))
    dy=torch.randn(24,29,device=device).T
    result=fn(*actual)
    a_grad=torch.autograd.grad(result,actual,dy)
    refs=torch.autograd.grad(expected,oracle,dy.double())
    for value,ref in zip((result,*a_grad),(expected,*refs)):
        assert bool(value.isfinite().all())
        ref=ref.detach().to(value.dtype).float();delta=value.detach().float()-ref
        assert float(delta.norm()/ref.norm().clamp_min(1e-12))<(4e-3 if dtype==torch.bfloat16 else 6e-6)


@pytest.mark.parametrize('layout',[SCALAR,ROW])
def test_full_affinity_fp64_oracle_cpu(layout):
    _whole_affinity(lambda *x:affinity(x,1e-6,layout),'cpu',torch.float32)


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_full_affinity_fp64_oracle_compiled(dtype):
    compute=torch.compile(lambda *x:affinity(x,1e-6,ROW),fullgraph=True,dynamic=False,
                          options={'triton.cudagraphs':False})
    _whole_affinity(compute,'cuda',dtype)


def test_affinity_autocast_retains_scalar_dtype_and_exact_result():
    torch.manual_seed(18667)
    params=(torch.randn(7,16),torch.randn(8,16),torch.randn(3,8),torch.zeros(()),torch.zeros(()))
    with torch.autocast('cpu',dtype=torch.bfloat16):
        a=affinity(params,1e-6,SCALAR);b=affinity(params,1e-6,ROW)
    assert a.dtype==b.dtype==torch.bfloat16 and torch.equal(a,b)


def test_affinity_fake_first_hit_guard_is_compiler_visible():
    from torch._subclasses.fake_tensor import FakeTensorMode,FakeTensor
    from rtx._moe_selection import publish_selected
    from rtx._routing_affinity import _clear_runtime_caches,grad_mask,identity,_math_context
    from rtx.routing import BalancedRouter
    model=BalancedRouter(16,8,router_dim=8,autotune='cache',dtype=torch.float32)
    graphs=[]
    def backend(graph,args):graphs.append(graph);return graph.forward
    try:
        with FakeTensorMode() as mode:
            def fake(shape,requires_grad):
                return FakeTensor(mode,torch.empty(shape,device='meta',requires_grad=requires_grad),torch.device('cuda:0'))
            for name,param in list(model.named_parameters()):
                setattr(model,name,torch.nn.Parameter(fake(param.shape,True)))
            x=fake((23,16),True)
            fn=torch.compile(model.log_affinities,backend=backend,fullgraph=True,dynamic=False)
            assert fn(x).shape==(23,8)
            tensors=model._affinity_inputs(x)
            publish_selected(*identity(tensors,model._selection_spec,grad_mask(tensors),_math_context()),ROW)
            assert fn(x).shape==(23,8)
            assert fn(x).shape==(23,8)
        assert len(graphs)==2
        code=lambda graph:'\n'.join(m.code for m in graph.modules() if isinstance(m,torch.fx.GraphModule))
        assert 'prepare_affinity_selection' in code(graphs[0])
        assert 'prepare_affinity_selection' not in code(graphs[1]) and '.expand(' in code(graphs[1])
    finally:_clear_runtime_caches()


def test_only_repeat_confirmed_affinity_is_durable():
    from types import SimpleNamespace
    from rtx.autotune.routing_affinity import _confirmed
    def result(layout,reason,confirmation):
        return SimpleNamespace(config=layout,winner_decision=dict(reason=reason,confirmation=confirmation))
    good=dict(status='complete',timing_unreliable=[],invalid=[])
    assert _confirmed(result(ROW,'paired_improvement',good))
    assert _confirmed(result(SCALAR,'incumbent_retained',good))
    assert not _confirmed(result(ROW,'only_candidate',None))
    assert not _confirmed(result(ROW,'incumbent_invalid',dict(good,invalid=['scalar'])))
    assert not _confirmed(result(SCALAR,'incumbent_retained',dict(good,status='incomplete')))
    assert not _confirmed(result(ROW,'paired_improvement',dict(good,timing_unreliable=['scalar'])))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_affinity_selector_opcheck_and_cold_fullgraph_all_grads(monkeypatch):
    import rtx._routing_affinity as module
    from rtx.routing import BalancedRouter
    torch.manual_seed(8801)
    model=BalancedRouter(32,8,router_dim=8,autotune='cache',device='cuda',dtype=torch.float32)
    with torch.no_grad():model.projection_weight.normal_(std=.05)
    x=torch.randn(67,32,device='cuda',requires_grad=True)
    inputs=model._affinity_inputs(x);up=torch.randn(67,8,device='cuda')
    monkeypatch.setattr(module,'_resolve',lambda *args,**kwargs:ROW)
    key=module.identity(inputs,model._selection_spec,module.grad_mask(inputs),module._math_context())
    try:
        values=torch.library.opcheck(module.prepare_affinity_selection,(*(t.detach() for t in inputs),
            model._selection_spec,module.grad_mask(inputs),module._math_context()))
        assert set(values.values())=={'SUCCESS'}
        module._clear_runtime_caches()
        compiled=torch.compile(model.log_affinities,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
        expected=affinity(inputs,model.rms_epsilon,SCALAR)
        expected_grad=torch.autograd.grad(expected,inputs,up)
        for _ in range(3):
            value=compiled(x)
            torch.testing.assert_close(value,expected,rtol=2e-6,atol=2e-6)
            for a,b in zip(torch.autograd.grad(value,inputs,up),expected_grad):
                torch.testing.assert_close(a,b,rtol=2e-4,atol=1e-5)
        from rtx._moe_selection import bound_config
        assert bound_config(*key)==ROW
    finally:module._clear_runtime_caches()
