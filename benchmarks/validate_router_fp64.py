"""Existing strict full-router FP64 thresholds on saved selected arms; never promote a failure."""
from pathlib import Path
import json,sys,math
source=Path(sys.argv[1]).resolve();sys.path.insert(0,str(source))
import torch
from benchmarks.benchmark_moe_varlen import initialize_fixture,compile_options
from benchmarks.benchmark_moe_execution import _tensor_digest
from rtx.moe import BalancedMoE
from rtx.routing import RoutingKernelConfig
from benchmarks.benchmark_moe_execution import _python_source_digest
out=Path(sys.argv[2]);rows=[];overall=True
requalify=sys.argv[3:4]==['--requalify-new-source'];actual_source=_python_source_digest(source)
paths=list(map(Path,sys.argv[4:] if requalify else sys.argv[3:]))
# Preflight every endpoint before any GPU work, including late failed workers.
for path in paths:
    if json.loads(path.read_text()).get('status')!='complete':
        raise RuntimeError(f'Incomplete qualification input: {path}')
for path in paths:
    s=json.loads(path.read_text());assert s['status']=='complete' and (s['source_digest']==actual_source or requalify)
    case=s['case'];c=RoutingKernelConfig.from_dict(s['router_config'])
    def factory(ca):return BalancedMoE(ca['d_model'],ca['num_experts'],ca['expert_hidden_dim'],top_k=ca['top_k'],
        shared_hidden_features=ca['shared_hidden_dim'],device='cuda',autotune='off',
        router_options=dict(backend='cute',autotune='off',config=c,affinity_layout=s['router_affinity_layout']))
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());torch.cuda.set_stream(stream)
    model,x,cu,dy=initialize_fixture(factory,case,s['initialization_case'],s['lengths'],layout='flat',seed=s['seed'],
        stable_inputs=s['stable_inputs'],stable_experts=s['stable_experts'],input_profile=s['input_profile'])
    assert _tensor_digest(x)==s['identities']['input'] and _tensor_digest(dy)==s['identities']['upstream']
    assert all(_tensor_digest(t)==s['identities']['parameters'][n] for n,t in model.named_parameters())
    mod=model.router;n=x.shape[0];e=case['num_experts'];k=case['top_k'];cap=n*k//e
    ups=torch.randn((e,cap),device=x.device,generator=torch.Generator(device=x.device).manual_seed(s['seed']+30593))
    x0=x.detach().clone();u0=ups.clone()
    leaves=(x,*mod.parameters());names=('dx',*dict(mod.named_parameters()))
    fn=torch.compile(mod,**compile_options(True))
    def step():
        route=fn(x);return route,torch.autograd.grad(route.route_weights,leaves,ups)
    for _ in range(3):step()
    stream.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream): outputs=step()
    for changed in (False,True,False):
        with torch.no_grad():
            x.copy_(x0.roll((1,1),(0,1)) if changed else x0)
            ups.copy_(1+u0.roll(1,1)*.0001 if changed else u0)
        graph.replay();route,grads=outputs
        fresh,fresh_grads=step()
        ids,w,inv=route
        for a,b in zip(route,fresh):torch.testing.assert_close(a,b,rtol=0,atol=0)
        if not changed:assert _tensor_digest(ids)==s['identities']['routing']['expert_ids'] and _tensor_digest(inv)==s['identities']['routing']['inverse']
        assert torch.equal(inv.flatten().sort().values,torch.arange(n*k,device=x.device))
        assert torch.equal(ids.flatten()[inv],torch.arange(n,device=x.device)[:,None].expand(n,k))
        refs=[v.detach().double().requires_grad_() for v in leaves]
        inputs,projection,experts,scale,bias=refs
        norm=lambda t:t*(t.square().mean(-1,keepdim=True)+mod.rms_epsilon).rsqrt()
        val=torch.nn.functional.logsigmoid(norm(inputs@projection.T)@norm(experts).T*scale.exp()+bias)
        for _ in range(mod.sinkhorn_iterations):
            val=val-val.logsumexp(1,keepdim=True)+math.log(k)
            val=val-val.logsumexp(0,keepdim=True)+math.log(cap)
        selected=val.gather(1,inv//cap).softmax(1)
        oracle=torch.empty(n*k,device=x.device,dtype=val.dtype).scatter(0,inv.flatten(),selected.flatten()).view(e,cap)
        ref_grads=torch.autograd.grad(oracle,refs,ups.double())
        def compare(a,b):
            delta=(a.double()-b.double());maximum=float(delta.abs().max().detach())
            nrms=float((delta.norm()/b.double().norm().clamp_min(1e-12)).detach())
            finite=bool(a.isfinite().all() and b.isfinite().all())
            if a.numel()==1:
                passed=bool(torch.isclose(a.double(),b.double(),rtol=2e-4,atol=1e-5))
                return dict(passed=finite and passed,finite=finite,actual=a.item(),reference=b.item(),max_abs=maximum,nrms=nrms)
            return dict(passed=finite and (nrms<.006 or maximum<2e-7),finite=finite,max_abs=maximum,nrms=nrms)
        stats={n:compare(a,b) for n,a,b in zip(names,grads,ref_grads)}
        replay={n:compare(a,b) for n,a,b in zip(names,grads,fresh_grads)}
        weights_ok=bool(torch.isclose(w.double(),oracle,rtol=5e-5,atol=5e-7).all())
        good=weights_ok and all(t['passed'] for t in (*stats.values(),*replay.values()))
        rows.append(dict(fixture=str(path),changed=changed,passed=good,weights_close=weights_ok,stats=stats,replay=replay))
        print(path.parent.name,changed,good,{n:v for n,v in stats.items() if not v['passed']},flush=True)
        overall &= good;out.write_text(json.dumps(dict(passed=overall,results=rows,source=str(source),
            source_digest=actual_source,fixture_source_digest=s['source_digest'],
            requalify_on_new_source=requalify),indent=2))
    graph.reset()
sys.exit(0 if overall else 2)
