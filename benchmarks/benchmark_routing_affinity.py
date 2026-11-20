"""Real first-hit affinity selection and fixed-schedule native packed MoE A/B."""
import argparse
import itertools
import json
from pathlib import Path
import statistics
import time

import torch

from rtx._moe_selection import bound_config
from rtx._routing_affinity import LAYOUTS, _clear_runtime_caches, _math_context, grad_mask, identity
from rtx.autotune import AutotunePolicy
from rtx.autotune.measurement import measure_cuda_batches
from rtx.autotune.routing_affinity import source_revision
from rtx.routing import BalancedRouter, RoutingKernelConfig


def compile_static(fn):
    return torch.compile(fn,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})


def first_hit(tokens,experts,root,mode):
    torch.manual_seed(17391)
    policy=AutotunePolicy(max_trials=2,max_seconds=75,samples=7,confirmation_repeats=0)
    model=BalancedRouter(384,experts,top_k=4,autotune=mode,tuning_policy=policy,
        autotune_cache_dir=root,device='cuda')
    with torch.no_grad():model.projection_weight.normal_(std=.05)
    x=(torch.randn(tokens,384,device='cuda',dtype=torch.bfloat16)*.3).requires_grad_()
    inputs=model._affinity_inputs(x);dy=torch.randn(tokens,experts,device='cuda')
    compiled=compile_static(model.log_affinities)
    t0=time.monotonic();score=compiled(x);grads=torch.autograd.grad(score,inputs,dy)
    torch.cuda.synchronize();cold_seconds=time.monotonic()-t0
    layout=bound_config(*identity(inputs,model._selection_spec,grad_mask(inputs),_math_context()))
    hot=compiled(x);hot_grads=torch.autograd.grad(hot,inputs,dy)
    torch.testing.assert_close(score,hot,rtol=0,atol=0)
    for a,b in zip(grads,hot_grads):torch.testing.assert_close(a,b,rtol=2e-4,atol=2e-5)
    assert layout in LAYOUTS
    _clear_runtime_caches()
    return dict(mode=mode,tokens=tokens,experts=experts,selected=layout,cold_seconds=cold_seconds,
                winner_files=[str(p.relative_to(root)) for p in root.glob('runtime_winners/balanced_router_affinity/*/*.json')])


def capture(step):
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):step()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):output=step()
    torch.cuda.current_stream().wait_stream(stream)
    return graph,output


def whole_moe(config_dir,lengths):
    from rtx.mlp import MXFP8MLPConfig
    from rtx.moe import BalancedMoE,MXFP8MoEConfig
    load=lambda name,kind:kind.from_dict(json.loads((config_dir/f'{name}.json').read_text())['config'])
    router,expert,shared=(load('router',RoutingKernelConfig),load('expert',MXFP8MoEConfig),load('shared',MXFP8MLPConfig))
    torch.manual_seed(17391)
    def construct(layout):
        return BalancedMoE(384,32,128,top_k=4,shared_hidden_features=256,
            device='cuda',autotune='off',config=expert,shared_config=shared,
            router_options={'autotune':'off','affinity_layout':layout,'config':router})
    a=construct(LAYOUTS[0]);b=construct(LAYOUTS[1]);b.load_state_dict(a.state_dict())
    with torch.no_grad():
        a.router.projection_weight.normal_(std=.05)
        b.router.projection_weight.copy_(a.router.projection_weight)
    x=(torch.randn(sum(lengths),384,device='cuda',dtype=torch.bfloat16)*.3).requires_grad_()
    dy=torch.randn_like(x)
    cu=torch.tensor([0,*itertools.accumulate(lengths)],device='cuda',dtype=torch.int32)
    graphs={};outputs={};routes={};times={}
    for name,model in zip(LAYOUTS,(a,b)):
        compiled=compile_static(model)
        leaves=(x,*model.parameters())
        def step():
            result=compiled(x,cu_seqlens=cu)
            return result,torch.autograd.grad(result,leaves,dy)
        graphs[name],outputs[name]=capture(step)
        graphs[name].replay();torch.cuda.synchronize();times[name]=[]
        with torch.no_grad():routes[name]=compile_static(model.router)(x,cu_seqlens=cu)
    for actual,expected in zip(routes[LAYOUTS[1]],routes[LAYOUTS[0]]):
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    (y0,g0),(y1,g1)=(outputs[name] for name in LAYOUTS)
    assert torch.equal(y0,y1) and bool(y1.isfinite().all())
    scalar_errors={}
    for name,ref,value in zip(('input',*(n for n,_ in a.named_parameters())),g0,g1):
        assert bool(value.isfinite().all()),name
        if name in ('router.logit_scale','router.score_bias'):
            delta=(value-ref).abs()
            scalar_errors[name]=dict(max_abs=float(delta.max()),nrms=float(delta.norm()/ref.norm().clamp_min(1e-12)))
            torch.testing.assert_close(value,ref,rtol=2e-4,atol=1e-5)
        else:assert torch.equal(ref,value),name
    live=[]
    for index in range(4):
        for name in (LAYOUTS if index%2==0 else LAYOUTS[::-1]):
            samples,meta=measure_cuda_batches(graphs[name].replay,samples=9,calls_per_sample=3)
            times[name].append(dict(samples_ms=samples,median_ms=statistics.median(samples),**meta))
        torch.cuda.synchronize();live.append(torch.cuda.memory_allocated())
    for graph in graphs.values():graph.reset()
    return dict(lengths=lengths,hard_routes_exact=True,output_and_nonscalar_grads_exact=True,
        scalar_errors=scalar_errors,live_growth_bytes=max(live)-min(live),timings=times,
        medians_ms={name:statistics.median(t['median_ms'] for t in values) for name,values in times.items()})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path)
    p.add_argument('--mode',choices=('cache','balanced'),default='balanced')
    p.add_argument('--cases',nargs='+',choices=('tiny','representative'),default=['tiny','representative'])
    p.add_argument('--configs',type=Path,help='Directory containing frozen router, expert, shared .json')
    p.add_argument('--lengths',type=int,nargs='+',default=[512,2000,768,4096])
    args=p.parse_args()
    if args.cache is None and args.configs is None:p.error('Specify --cache and/or --configs')
    print(json.dumps(dict(device=str(torch.cuda.get_device_properties(0)),torch=torch.__version__,
        cuda=torch.version.cuda,source_revision=source_revision())),flush=True)
    if args.cache:
        for case in args.cases:
            tokens,experts=(1,64) if case=='tiny' else (7376,32)
            print(json.dumps(first_hit(tokens,experts,args.cache/case,args.mode)),flush=True)
    if args.configs:print(json.dumps(whole_moe(args.configs,args.lengths)),flush=True)


if __name__=='__main__':main()
