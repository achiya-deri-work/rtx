"""Fixed-K invariants, saved Sinkhorn state and detached scheduling."""
import copy
import pytest
import torch

from rtx.routing import BalancedRouter,log_sinkhorn,SinkhornKernelConfig


def test_initial_moe_workloads_have_exact_unpadded_capacity():
    import json
    from pathlib import Path
    from rtx._routing_reference import padded_token_count, uniform_capacity
    suite=json.loads((Path(__file__).resolve().parents[1]/'benchmarks/balanced_moe_cases.json').read_text())
    assert suite['schema_version']==1 and suite['token_counts']==[128,512,2000]
    expected=[(384,256,6,128,24,[32,128,500]),
              (384,256,4,128,32,[16,64,250]),
              (512,384,8,128,64,[16,64,250]),
              (768,512,6,256,48,[16,64,250])]
    assert len(suite['cases'])==len(expected)
    for case,(d,hs,k,h,e,capacities) in zip(suite['cases'],expected):
        assert (case['d_model'],case['shared_hidden_dim'],case['top_k'],
                case['expert_hidden_dim'],case['num_experts'])==(d,hs,k,h,e)
        assert case['num_shared_experts']==1
        for n,capacity in zip(suite['token_counts'],capacities):
            assert padded_token_count(n,e,k)==n
            assert uniform_capacity(n,e,k)==capacity


def independent_transport(scores,k,iterations):
    value=scores
    logk=torch.tensor(float(k),dtype=value.dtype,device=value.device).log()
    logc=torch.tensor(float(scores.shape[0]*k/scores.shape[1]),dtype=value.dtype,device=value.device).log()
    for _ in range(iterations):
        value=value-torch.logsumexp(value,1,keepdim=True)+logk
        value=value-torch.logsumexp(value,0,keepdim=True)+logc
    return value


def check_routes(route,n,e,k):
    ids,weights,inverse=route
    c=ids.shape[1]
    assert ids.shape==weights.shape==(e,c) and inverse.shape==(n,k)
    assert ids.dtype==inverse.dtype==torch.int64 and weights.dtype==torch.float32
    assert not ids.requires_grad and not inverse.requires_grad
    assigned_experts=inverse//c
    assert not bool((assigned_experts.sort(1).values.diff(dim=1)==0).any())
    assert torch.equal(ids.flatten()[inverse],torch.arange(n,device=ids.device)[:,None].expand(n,k))
    torch.testing.assert_close(weights.flatten()[inverse].sum(1),torch.ones(n,device=weights.device))
    valid=torch.zeros(e*c,device=ids.device,dtype=torch.bool)
    valid[inverse.flatten()]=True
    assert bool((weights.flatten()[~valid]==0).all()) and bool(weights.isfinite().all())
    return valid


@pytest.mark.parametrize('n,e,k',[(1,8,2),(17,8,2),(65,16,4),(128,16,1),(32,8,8)])
@pytest.mark.parametrize('family',['uniform','normal','sharp'])
def test_cpu_fixed_k_padding_and_finite_gradients(n,e,k,family):
    torch.manual_seed(742)
    r=BalancedRouter(8,e,top_k=k,router_dim=4,dtype=torch.float32)
    if family!='uniform':
        with torch.no_grad():r.projection_weight.normal_(std=.1)
    if family=='sharp':
        with torch.no_grad():r.logit_scale.fill_(3)
    x=torch.randn(n,8,requires_grad=True)
    route=r(x);check_routes(route,n,e,k)
    coefficient=torch.randn_like(route.route_weights)
    (route.route_weights*coefficient).sum().backward()
    assert x.grad is not None and bool(x.grad.isfinite().all())
    for p in r.parameters():assert p.grad is not None and bool(p.grad.isfinite().all())


def test_initial_affinity_and_padding_error():
    r=BalancedRouter(8,16,top_k=4,router_dim=4,dtype=torch.float32)
    x=torch.randn(65,8)
    torch.testing.assert_close(r.log_affinities(x).exp(),torch.full((65,16),.5),rtol=0,atol=0)
    r.padding='error'
    with pytest.raises(ValueError,match='not divisible'):r(x)


def test_router_gradients_follow_gathered_sinkhorn_not_integer_decisions():
    torch.manual_seed(743)
    r=BalancedRouter(5,8,top_k=2,router_dim=4,dtype=torch.float32)
    with torch.no_grad():r.projection_weight.normal_(std=.1)
    x=torch.randn(17,5,requires_grad=True)
    route=r(x);n,e,k=17,8,2;c=route.expert_ids.shape[1]
    selected=route.inverse//c
    coefficient=torch.randn(n,k)
    weights=route.route_weights.flatten()[route.inverse]
    actual=torch.autograd.grad((weights*coefficient).sum(),(x,*r.parameters()))
    xr=x.detach().double().requires_grad_()
    projection,directions,scale,bias=(p.detach().double().requires_grad_() for p in r.parameters())
    latent=torch.nn.functional.linear(xr,projection)
    latent=latent*torch.rsqrt(latent.square().mean(-1,keepdim=True)+r.rms_epsilon)
    unit=directions*torch.rsqrt(directions.square().mean(-1,keepdim=True)+r.rms_epsilon)
    scores=torch.nn.functional.logsigmoid(latent@unit.T*scale.exp()+bias)
    scores=torch.nn.functional.pad(scores,(0,0,0,e*c//k-n),value=-__import__('math').log(2))
    logp=independent_transport(scores,k,5)
    expected_weights=logp[:n].gather(1,selected).softmax(1)
    expected=torch.autograd.grad((expected_weights*coefficient.double()).sum(),(xr,projection,directions,scale,bias))
    torch.testing.assert_close(weights.double(),expected_weights,rtol=2e-5,atol=2e-6)
    for a,b in zip(actual,expected):torch.testing.assert_close(a.double(),b,rtol=2e-4,atol=5e-6)


def test_cpu_fullgraph_aot_and_parameter_updates():
    r=BalancedRouter(8,8,top_k=2,router_dim=4,dtype=torch.float32)
    fn=torch.compile(r,backend='aot_eager',fullgraph=True,dynamic=False)
    opt=torch.optim.SGD(r.parameters(),lr=.001)
    for step in range(3):
        x=torch.randn(17,8,requires_grad=True)
        route=fn(x);check_routes(route,17,8,2)
        opt.zero_grad();(route.route_weights*torch.randn_like(route.route_weights)).sum().backward();opt.step()


@pytest.mark.parametrize('threads,row,column',[(64,1,1),(128,2,4),(256,4,8)])
def test_kernel_config_roundtrip(threads,row,column):
    assert SinkhornKernelConfig(threads,row,column).column_groups==column


cuda=pytest.mark.skipif(not torch.cuda.is_available(),reason='assigned RTX CUDA required')


@cuda
@pytest.mark.parametrize('n,e,k,steps,config',[
    (1,1,1,5,SinkhornKernelConfig(64,8,8)),
    (14,7,3,5,SinkhornKernelConfig(128,2,4)),
    (64,8,1,1,SinkhornKernelConfig(64,1,1)),
    (68,16,4,3,SinkhornKernelConfig(128,2,4)),
    (192,32,8,5,SinkhornKernelConfig(256,4,8)),
    (4096,128,8,5,SinkhornKernelConfig(128,4,4)),
])
def test_native_sinkhorn_forward_backward_no_recomputation(n,e,k,steps,config,monkeypatch,record_property):
    import rtx.kernels.routing_sinkhorn as kernel
    torch.manual_seed(744)
    s=(torch.randn(n,e,device='cuda')*.7-2).requires_grad_()
    reference=s.detach().double().requires_grad_()
    gradient=torch.randn(e,n,device='cuda').T
    saved=[]
    with torch.autograd.graph.saved_tensors_hooks(lambda t:(saved.append(t),t)[1],lambda t:t):
        out=log_sinkhorn(s,top_k=k,iterations=steps,config=config)
    want=independent_transport(reference,k,steps)
    assert any(t.numel()==2*steps*n*e for t in saved)
    assert any(tuple(t.shape)==(n,e) for t in saved)
    original=kernel.compile_sinkhorn
    def only_backward(*a,**kw):
        init,forward,finish,backward=original(*a,**kw)
        def forbidden(*_):raise AssertionError('Sinkhorn forward was recomputed in backward')
        return forbidden,(forbidden,forbidden),forbidden,backward
    monkeypatch.setattr(kernel,'compile_sinkhorn',only_backward)
    actual=torch.autograd.grad(out,s,gradient)[0]
    expected=torch.autograd.grad(want,reference,gradient.double())[0]
    torch.testing.assert_close(out.double(),want,rtol=5e-5,atol=5e-6)
    torch.testing.assert_close(actual.double(),expected,rtol=5e-4,atol=8e-6)
    assert bool(actual.isfinite().all())
    for name,candidate,oracle in [('transport',out,want),('gradient',actual,expected)]:
        error=candidate.double()-oracle
        rms=error.square().mean().sqrt()
        record_property(name+'_max_abs',error.abs().max().item())
        record_property(name+'_rms_error',rms.item())
        record_property(name+'_normalized_rms',(rms/oracle.square().mean().sqrt().clamp_min(1e-12)).item())
    record_property('seed',744)
    record_property('torch_version',torch.__version__)
    record_property('gpu',torch.cuda.get_device_name())


@cuda
@pytest.mark.parametrize('n,e,k,rounds',[(68,16,4,64),(128,32,8,64),(128,32,8,1),(16,4,4,64)])
def test_native_hard_schedule_same_frozen_policy_and_owned_outputs(n,e,k,rounds):
    from rtx._routing_hard import hard_schedule
    from rtx import _routing_reference as ref
    torch.manual_seed(745)
    def fixed(p):
        initial,_=ref.dual_price_assignment(p,top_k=k,iterations=5)
        return ref.repair_balanced_assignment(p,initial,rounds=rounds)
    expected_fn=torch.compile(fixed,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    previous=None
    for uniform in (False,True):
        scores=torch.full((n,e),.1,device='cuda') if uniform else torch.rand(n,e,device='cuda')
        expected=expected_fn(scores)
        actual=hard_schedule(scores,k,5,rounds)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        if previous is not None:torch.testing.assert_close(previous[0],previous[1],rtol=0,atol=0)
        previous=actual,actual.clone()


@cuda
def test_cuda_router_compile_gradients_changed_replay_and_cache_cleanup():
    from rtx import _routing_reference as ref
    from rtx.runtime import clear_runtime_caches
    torch.manual_seed(746)
    router=BalancedRouter(16,16,top_k=4,router_dim=8,device='cuda',dtype=torch.bfloat16,autotune='off')
    with torch.no_grad():router.projection_weight.normal_(std=.1)
    x=torch.randn(65,16,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    coefficient=torch.randn(16,17,device='cuda')
    # Cold preparation must be transparent to torch.compile, not a user API.
    fn=torch.compile(router,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    def run():
        route=fn(x)
        grad=torch.autograd.grad((route.route_weights*coefficient).sum(),(x,*router.parameters()))
        return route,grad
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):run()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):actual=run()
    torch.cuda.current_stream().wait_stream(stream)
    clear_runtime_caches(device=x.device,preserve_compiled=True)
    for _ in range(2):
        with torch.no_grad():x.add_(.002);router.projection_weight.add_(.0001)
        graph.replay();torch.cuda.synchronize()
        route,grads=actual
        check_routes(route,65,16,4)
        # Freeze actual hard assignments; differentiate an independent ordinary
        # Torch Sinkhorn reference on that schedule, never through QB/repair.
        scores=router.log_affinities(x)
        scores=torch.nn.functional.pad(scores,(0,0,0,3),value=-__import__('math').log(2))
        logp=ref.log_sinkhorn(scores,top_k=4,iterations=5)
        selected=route.inverse//17
        weights=logp[:65].gather(1,selected).softmax(1)
        coeff=coefficient.flatten()[route.inverse]
        want=torch.autograd.grad((weights*coeff).sum(),(x,*router.parameters()))
        for a,b in zip(grads,want):torch.testing.assert_close(a,b,rtol=.015,atol=.001)
    graph.reset()


@cuda
def test_cuda_real_operator_contracts():
    from rtx.routing import _sinkhorn_forward
    from rtx._routing_hard import hard_schedule
    scores=torch.randn(64,16,device='cuda',requires_grad=True)
    result=torch.library.opcheck(_sinkhorn_forward,(scores,4,3,128,2,4))
    assert all(v=='SUCCESS' for v in result.values())
    result=torch.library.opcheck(hard_schedule,(scores.detach().exp(),4,5,64))
    assert all(v=='SUCCESS' for v in result.values())


@cuda
def test_large_cuda_router_saved_state_and_live_memory_stability(record_property):
    import gc
    torch.manual_seed(747)
    router=BalancedRouter(32,128,top_k=8,router_dim=16,device='cuda',autotune='off')
    with torch.no_grad():router.projection_weight.normal_(std=.05)
    x=torch.randn(1024,32,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    def step():
        route=router(x)
        check_routes(route,1024,128,8)
        coefficient=torch.linspace(-1,1,route.route_weights.numel(),device='cuda').view_as(route.route_weights)
        gradients=torch.autograd.grad((route.route_weights*coefficient).sum(),(x,*router.parameters()))
        assert all(bool(g.isfinite().all()) for g in gradients)
    for _ in range(3):step()
    gc.collect();torch.cuda.synchronize()
    baseline=torch.cuda.memory_allocated()
    for _ in range(8):step()
    gc.collect();torch.cuda.synchronize()
    final=torch.cuda.memory_allocated()
    record_property('live_allocated_before',baseline)
    record_property('live_allocated_after',final)
    assert final<=baseline+65536


@cuda
def test_native_hard_schedule_stream_and_output_isolation():
    from rtx._routing_hard import hard_schedule
    from rtx import _routing_reference as ref
    torch.manual_seed(748)
    scores=[torch.rand(128,32,device='cuda') for _ in range(2)]
    streams=[torch.cuda.Stream() for _ in range(2)]
    for stream in streams:stream.wait_stream(torch.cuda.current_stream())
    # Cold preparation is serialized; hot submissions may be independent.
    for stream,p in zip(streams,scores):
        with torch.cuda.stream(stream):hard_schedule(p,8,5,64)
    torch.cuda.synchronize()
    results=[]
    for stream,p in zip(streams,scores):
        with torch.cuda.stream(stream):results.append(hard_schedule(p,8,5,64))
    for stream in streams:torch.cuda.current_stream().wait_stream(stream)
    snapshots=[v.clone() for v in results]
    for p in scores:hard_schedule(p,8,5,64)
    for actual,saved in zip(results,snapshots):
        torch.testing.assert_close(actual,saved,rtol=0,atol=0)
        loads=ref.assignment_loads(actual,32)
        torch.testing.assert_close(loads,torch.full_like(loads,32),rtol=0,atol=0)
        assert not bool((actual.sort(1).values.diff(dim=1)==0).any())


@cuda
@pytest.mark.parametrize('threads',[64,128,256])
def test_native_quantile_matches_order_statistic(threads):
    from rtx._routing_hard import quantile
    torch.manual_seed(750)
    matrix=torch.randn(73,31,device='cuda')
    matrix[::3]=0  # Exact ties, including positive/negative zero.
    for column in (False,True):
        length=matrix.shape[0 if column else 1]
        for rank in (0,length//2,length-1):
            expected=matrix.sort(dim=0 if column else 1,descending=True).values
            expected=expected[rank] if column else expected[:,rank]
            torch.testing.assert_close(quantile(matrix,rank,threads,column),expected,rtol=0,atol=0)


@cuda
@pytest.mark.parametrize('backend',['native_column','native'])
@pytest.mark.parametrize('n,e,k',[(68,16,4),(1024,128,8)])
def test_native_qb_preserves_frozen_hard_policy(backend,n,e,k):
    from rtx._routing_hard import hard_schedule
    torch.manual_seed(751)
    for uniform in (False,True):
        scores=torch.full((n,e),.1,device='cuda') if uniform else torch.rand(n,e,device='cuda')
        expected=hard_schedule(scores,k,5,64)
        actual=hard_schedule(scores,k,5,64,True,0,1024,backend,128)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)


@cuda
@pytest.mark.parametrize('n,e',[(8193,7),(40000,2)])
def test_streaming_cute_quantile(n,e):
    from rtx._routing_hard import quantile
    torch.manual_seed(754)
    matrix=torch.randn(n,e,device='cuda')
    rank=n//2
    expected=matrix.sort(0,descending=True).values[rank]
    torch.testing.assert_close(quantile(matrix,rank,128,True),expected,rtol=0,atol=0)
