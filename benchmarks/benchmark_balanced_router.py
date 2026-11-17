"""Routing phase baselines and before/after autotuning, with JSON evidence.

Run from the checkout with PYTHONPATH=.; compilation is never timed. The
default CUDA-graph timer measures the stitched GPU path, not Python launch gaps.
"""
import argparse
import inspect
import json
from pathlib import Path
import statistics
import time

import torch

from rtx.routing import BalancedRouter, RoutingKernelConfig, _sinkhorn_forward, _sinkhorn_backward
from rtx import _routing_reference as reference
from rtx.autotune.measurement import measure_cuda_batches, runtime_timing_metadata


def measure(call, *, samples=7, warmup=3, profile=False, label="routing",stream=None,warmup_ms=200):
    stream=torch.cuda.Stream() if stream is None else stream
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):call()
    stream.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):
        result=call()
    torch.cuda.current_stream().wait_stream(stream)
    from rtx.autotune.preparation_runtime import _warm_measurement_graph
    graph_warmup=_warm_measurement_graph(graph)
    if warmup_ms>200:
        extra=min(4096,__import__('math').ceil((warmup_ms-200)/graph_warmup['pilot_graph_ms']))
        for _ in range(extra):graph.replay()
        graph_warmup.update(extra_replays=extra,requested_warmup_ms=warmup_ms)
    timings,metadata=measure_cuda_batches(graph.replay,samples=samples,calls_per_sample=3,
                                         device=torch.cuda.current_device(),min_batch_ms=3.)
    if profile:
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        with torch.cuda.nvtx.range(label):
            for _ in range(3):graph.replay()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    del result
    graph.reset()
    return dict(median_ms=statistics.median(timings),min_ms=min(timings),
                timings_ms=timings,graph_warmup=graph_warmup,**metadata,**runtime_timing_metadata(timings))


def benchmark(n,d,e,k,args):
    from rtx._routing_hard import HardRoutingPlan
    torch.manual_seed(args.seed)
    extra={"autotune":"off"} if "autotune" in inspect.signature(BalancedRouter).parameters else {}
    config=RoutingKernelConfig(qb_backend=args.qb_backend,qb_threads=args.qb_threads,
                               segmented_quotas=not args.no_segmented,quota_threshold=args.quota_threshold)
    router=BalancedRouter(d,e,top_k=k,router_dim=args.router_dim,device=args.device,config=config,**extra)
    with torch.no_grad():router.projection_weight.normal_(std=.02)
    x=torch.randn(n,d,device=args.device,dtype=torch.bfloat16,requires_grad=True)
    compile_options={"fullgraph":True,"dynamic":False,"options":{"triton.cudagraphs":False}}
    fn=torch.compile(router,**compile_options)
    route=fn(x)
    coefficient=torch.randn_like(route.route_weights)
    del route  # Do not retain default-stream autograd nodes into side capture.
    def training():
        r=fn(x)
        return torch.autograd.grad((r.route_weights*coefficient).sum(),(x,*router.parameters()))
    c=router.sinkhorn_config
    affinities=router.log_affinities(x).detach()
    ns=reference.padded_token_count(n,e,k)
    affinities=torch.nn.functional.pad(affinities,(0,0,0,ns-n),value=-__import__('math').log(2))
    logp,history=_sinkhorn_forward(affinities,k,5,c.threads,c.row_groups,c.column_groups)
    gradient=torch.randn_like(logp)
    scores=logp.detach().exp()
    plan=HardRoutingPlan(scores,k,5,64,config.segmented_quotas,config.quota_threshold,
                         config.qb_backend,config.qb_threads)
    ids=plan(scores)
    torch.cuda.synchronize()
    assert bool((reference.assignment_loads(ids,e)==ns*k//e).all())
    assert not bool((ids.sort(1).values.diff(dim=1)==0).any())
    affinity_fn=torch.compile(router.log_affinities,**compile_options)
    metadata_fn=torch.compile(lambda p,i:reference.routing_metadata(p,i,num_real_tokens=n),**compile_options)
    calls={
        "affinities":lambda:affinity_fn(x),
        "sinkhorn_forward":lambda:_sinkhorn_forward(affinities,k,5,c.threads,c.row_groups,c.column_groups),
        "sinkhorn_backward":lambda:_sinkhorn_backward(gradient,history,k,5,c.threads,c.row_groups,c.column_groups),
        "qb":lambda:plan.initialize_op(scores),
        "hard_total":lambda:plan(scores),
        "metadata":lambda:metadata_fn(logp,ids),
        "forward":lambda:fn(x),
        "forward_backward":training,
    }
    # Plans are stream-owned. The phase driver uses the current stream for
    # this private plan; measure() normally prepares on a side capture stream.
    # Reconstruct only the hard benchmark on its intended capture stream.
    if args.phase not in ("all",*calls):raise ValueError("unknown phase")
    phases={}
    for name,call in calls.items():
        if args.phase not in ("all",name):continue
        if name=="hard_total":
            from rtx._routing_hard import hard_schedule
            call=lambda:hard_schedule(scores,k,5,64,config.segmented_quotas,0,config.quota_threshold,
                                      config.qb_backend,config.qb_threads)
        print(f"MEASURE n={n} d={d} e={e} k={k} phase={name}",flush=True)
        phases[name]=measure(call,samples=args.samples,profile=args.profile,label="routing/"+name)
        print(f"RESULT {name} {phases[name]['median_ms']*1000:.3f}us",flush=True)
    torch.cuda.synchronize()
    output=dict(shape=dict(tokens=n,d_model=d,experts=e,top_k=k,router_dim=args.router_dim),
                phases=phases,repair_rounds=int(plan.phase.item()),
                live_allocated=torch.cuda.memory_allocated(),peak_allocated=torch.cuda.max_memory_allocated())
    if args.tune:
        from rtx.autotune import AutotunePolicy
        started=time.monotonic()
        result=router.autotune_(x,tuning_policy=AutotunePolicy.preset(args.tune),
                                output_dir=Path(args.output).parent/"tuning",phase="train",progress=print)
        tuned_fn=torch.compile(router,**compile_options)
        def tuned_training():
            r=tuned_fn(x)
            return torch.autograd.grad((r.route_weights*coefficient).sum(),(x,*router.parameters()))
        output["tuning"]=dict(cold_seconds=time.monotonic()-started,config=result.config.to_dict(),
                              median_ms=result.median_ms)
        output["tuned_forward"]=measure(lambda:tuned_fn(x),samples=args.samples)
        output["tuned_forward_backward"]=measure(tuned_training,samples=args.samples)
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shapes",nargs="+",default=["256:128:32:4","1024:384:128:8","4096:384:128:8"])
    p.add_argument("--router-dim",type=int,default=64)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--samples",type=int,default=7)
    p.add_argument("--seed",type=int,default=749)
    p.add_argument("--phase",default="all")
    p.add_argument("--profile",action="store_true")
    p.add_argument("--tune",choices=["quick","balanced","thorough","extensive"])
    p.add_argument("--qb-backend",choices=["torch","native_column","native"],default="torch")
    p.add_argument("--qb-threads",type=int,default=128)
    p.add_argument("--quota-threshold",type=int,default=1024)
    p.add_argument("--no-segmented",action="store_true")
    p.add_argument("--output",required=True)
    args=p.parse_args()
    with torch.cuda.device(args.device):
        result=dict(torch_version=torch.__version__,cuda_version=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(),seed=args.seed,measurement="cuda_graph",
                    source="benchmark_balanced_router.py",results=[])
        for shape in args.shapes:
            result["results"].append(benchmark(*map(int,shape.split(":")),args))
            path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":main()
