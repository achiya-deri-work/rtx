"""Fresh-process public compiled NVFP4 training graph comparator.

CUDA graph replay latency excludes Python/autograd launch overhead, compile,
capture, and input generation; it is not end-to-end training wall throughput.
"""
import argparse
from dataclasses import asdict,replace
import hashlib
import gc
import json
from pathlib import Path
import statistics
import random
import subprocess
import sys

POLICIES={"fp4_fp4":("ms_eden","ms_eden"),"fp4_mxfp8":("ms_eden","mxfp8"),
          "mxfp8_fp4":("mxfp8","ms_eden"),"mxfp8_mxfp8":("mxfp8","mxfp8"),
          "stable_mxfp8":("mxfp8","mxfp8")}


def source_hashes():
    root=Path(__file__).resolve().parents[1]
    names=("benchmarks/compare_mixed_nvfp4_backward.py","rtx/experimental_ms_eden.py",
           "rtx/kernels/nvfp4_posthoc.py","rtx/fp8_bwd.py","rtx/fp4.py")
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}


def matched_forward_config(m,n,k):
    from rtx.configs.nvfp4 import NVFP4Problem
    from rtx.nvfp4_inference_autotune import preferred_jit_row_region_config
    return replace(preferred_jit_row_region_config(NVFP4Problem(m,n,k)),tensor_scale_mode="exact")


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy",choices=("all",*POLICIES),default="all")
    for name,default in (("m",128),("n",128),("k",128),("samples",9),("calls-per-sample",4),("warmup",5)):
        p.add_argument("--"+name,type=int,default=default)
    p.add_argument("--seed",type=int,default=941)
    p.add_argument("--schedule-json",default="{}")
    p.add_argument("--dx-schedule-json")
    p.add_argument("--dw-schedule-json")
    p.add_argument("--backward-pipeline",choices=("serial","paired","operand_paired"),default="serial")
    p.add_argument("--mxfp8-config-json",help="serialized MXFP8BwdConfig; omitted means repository default")
    p.add_argument("--output",required=True)
    p.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    p.add_argument("--paired",action="store_true",help="within-session paired total-graph comparisons")
    p.add_argument("--rounds",type=int,default=12)
    p.add_argument("--replays",type=int,default=40)
    return p


def validate(args):
    if any(v<=0 or v%128 for v in (args.m,args.n,args.k)):
        raise ValueError("m/n/k must be positive multiples of 128")
    if args.samples<3 or args.calls_per_sample<1 or args.warmup<2:
        raise ValueError("samples>=3, calls-per-sample>=1, warmup>=2 required")
    if args.rounds<4 or args.replays<1:
        raise ValueError("rounds>=4 and replays>=1 required")
    from rtx.autotune.posthoc import PosthocPreparationSchedule
    for value in (args.schedule_json,args.dx_schedule_json,args.dw_schedule_json):
        if value is not None:
            PosthocPreparationSchedule(**json.loads(value))
    if args.mxfp8_config_json is not None:
        from rtx.bwd_autotune import bwd_config_from_dict
        bwd_config_from_dict(json.loads(args.mxfp8_config_json))


def worker_command(args,policy,output):
    command=[sys.executable,"-m","benchmarks.compare_mixed_nvfp4_backward","--worker",
             "--policy",policy,"--output",str(output)]
    for name in ("m","n","k","samples","calls_per_sample","warmup","seed","schedule_json",
                 "dx_schedule_json","dw_schedule_json","mxfp8_config_json","backward_pipeline"):
        value=getattr(args,name,"serial" if name=="backward_pipeline" else None)
        if value is not None:
            command.extend(("--"+name.replace("_","-"),str(value)))
    return command


def timing_summary(values,batching):
    from rtx.autotune.measurement import runtime_timing_metadata
    return dict(median_ms=statistics.median(values),min_ms=min(values),samples_ms=values,
                batching=batching,**runtime_timing_metadata(values))


def independent_shared_leaves(inputs):
    """Alias A/B data without sharing their autograd AccumulateGrad nodes.

    Detach creates tensor metadata only: addresses, layout, contents and version
    tracking stay shared, while each graph bundle owns distinct leaf identities.
    This is capture setup, outside all timing and allocation-audit windows.
    """
    return tuple(value.detach().requires_grad_(value.requires_grad) for value in inputs)


def run_worker(args,*,capture_only=False,shared_inputs=None):
    import torch
    import rtx
    from rtx.experimental_ms_eden import ms_eden_linear
    from rtx.autotune.measurement import measure_cuda_batches
    from rtx.fp8_bwd import _default_bwd_config
    from rtx.bwd_autotune import bwd_config_from_dict
    from rtx.kernels.mxfp8 import MXFP8Problem
    torch.manual_seed(args.seed)
    dx,dw=POLICIES[args.policy]
    config=(bwd_config_from_dict(json.loads(args.mxfp8_config_json)) if args.mxfp8_config_json else
            _default_bwd_config(MXFP8Problem(args.m,args.n,args.k)))
    forward_config=matched_forward_config(args.m,args.n,args.k)
    scale_policy=rtx.NVFP4ScaleConfig(tensor_scale_mode="exact")
    if shared_inputs is None:
        x=(torch.randn(args.m,args.k,device="cuda",dtype=torch.bfloat16)*.25).requires_grad_()
        w=(torch.randn(args.n,args.k,device="cuda",dtype=torch.bfloat16)*.02).requires_grad_()
        dy=torch.randn(args.m,args.n,device="cuda",dtype=torch.bfloat16)
    else:
        x,w,dy=independent_shared_leaves(shared_inputs)
    def forward(a,b):
        if args.policy=="stable_mxfp8":
            return rtx.nvfp4_linear(a,b,scaling="jit_row_region",autotune="off",
                scale_config=scale_policy,
                dynamic_config=forward_config,backward_config=config)
        return ms_eden_linear(a,b,source_policy="original_bf16",forward_scaling="jit_row_region",
            forward_outer_mode="exact",forward_config=forward_config,dx_policy=dx,dw_policy=dw,
            schedule_json=args.schedule_json,dx_schedule_json=args.dx_schedule_json,
            dw_schedule_json=args.dw_schedule_json,
            backward_pipeline=getattr(args,"backward_pipeline","serial"),
            mxfp8_backward_config=config if "mxfp8" in (dx,dw) else None)
    compiled=torch.compile(forward,fullgraph=True,dynamic=False)
    def step():
        out=compiled(x,w)
        return out,torch.autograd.grad(out,(x,w),dy)
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(args.warmup):
            output,grads=step()
    stream.synchronize()
    assert all(torch.isfinite(t).all() for t in (output,*grads)),"nonfinite warmup"
    reference=rtx.nvfp4_linear(x.detach(),w.detach(),scaling="jit_row_region",autotune="off",
        scale_config=scale_policy,dynamic_config=forward_config)
    torch.testing.assert_close(output,reference,atol=0,rtol=0)
    del reference,output,grads
    # Warm the exact default-stream allocation/release lifecycle audited below;
    # side-stream capture warmup alone does not initialize stream-local caches.
    for _ in range(args.warmup):
        ordinary_output,ordinary_grads=step()
        del ordinary_output,ordinary_grads
    torch.cuda.synchronize()
    gc.collect()
    ordinary_before=torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    ordinary_batches=[]
    for _ in range(3):
        for _ in range(10):
            ordinary_output,ordinary_grads=step()
            del ordinary_output,ordinary_grads
        torch.cuda.synchronize()
        gc.collect()
        ordinary_batches.append(torch.cuda.memory_allocated())
    ordinary_after=ordinary_batches[-1]
    from tests.cache_memory_checks import wait_for_cuda_allocation_cleanup
    cleanup_samples=wait_for_cuda_allocation_cleanup(ordinary_before,observed=ordinary_after)
    ordinary_after=cleanup_samples[-1]
    ordinary_memory=dict(iterations=30,batch_size=10,batch_live_allocated_bytes=ordinary_batches,
        warmup_calls_per_stream=args.warmup,idle_cleanup_allocated_samples=cleanup_samples,
        baseline_allocated_bytes=ordinary_before,
        after_release_allocated_bytes=ordinary_after,live_growth_bytes=ordinary_after-ordinary_before,
        peak_allocated_bytes=torch.cuda.max_memory_allocated())
    if ordinary_after>ordinary_before:
        raise RuntimeError(f"ordinary compiled calls retained live allocations: {ordinary_memory}")
    baseline=torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    forward_graph,backward_graph,total_graph=(torch.cuda.CUDAGraph() for _ in range(3))
    with torch.cuda.graph(forward_graph,stream=stream):
        forward_output=compiled(x,w)
    with torch.cuda.graph(backward_graph,stream=stream):
        backward_grads=torch.autograd.grad(forward_output,(x,w),dy,retain_graph=True)
    with torch.cuda.graph(total_graph,stream=stream):
        total_output,total_grads=step()
    torch.cuda.synchronize()
    capture_peak=torch.cuda.max_memory_allocated()
    persistent_after_capture=torch.cuda.memory_allocated()
    # The total graph must read current buffers, not retain captured values.
    total_graph.replay(); torch.cuda.synchronize()
    def current_buffer_guard(label):
        ref_output=rtx.nvfp4_linear(x.detach(),w.detach(),scaling="jit_row_region",autotune="off",
            scale_config=scale_policy,dynamic_config=forward_config)
        torch.testing.assert_close(total_output,ref_output,atol=0,rtol=0)
        references=(dy.float()@w.detach().float(),dy.float().T@x.detach().float())
        errors={}
        for phase,actual,reference,policy in zip(("dx","dw"),total_grads,references,(dx,dw)):
            if not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
                raise RuntimeError(f"{label} {phase} nonfinite")
            nrms=float((actual.float()-reference).norm()/reference.norm().clamp_min(1e-12))
            gate=.25 if policy=="ms_eden" else .06
            if nrms>gate:
                raise RuntimeError(f"{label} {phase} current-buffer gradient NRMS {nrms} > {gate}")
            errors[phase]=dict(nrms=nrms,gate=gate)
        return errors
    guard_errors={"initial":current_buffer_guard("initial")}
    initial_output=total_output.clone()
    initial_grads=tuple(t.clone() for t in total_grads)
    total_graph.replay(); torch.cuda.synchronize()
    fresh=tuple(not torch.equal(a,b) for a,b in zip(initial_grads,total_grads))
    for changed,policy in zip(fresh,(dx,dw)):
        if policy=="ms_eden" and not changed:
            raise RuntimeError("captured FP4 phase reused stale RNG")
        if policy=="mxfp8" and changed:
            raise RuntimeError("deterministic MXFP8 phase changed on identical inputs")
    with torch.no_grad():
        x.mul_(.5)
    total_graph.replay(); torch.cuda.synchronize()
    if torch.equal(initial_output,total_output):
        raise RuntimeError("captured forward ignored changed current input")
    if torch.equal(initial_grads[1],total_grads[1]):
        raise RuntimeError("captured dW ignored changed current input")
    guard_errors["changed_x"]=current_buffer_guard("changed_x")
    changed_input_grads=tuple(t.clone() for t in total_grads)
    with torch.no_grad():
        dy.mul_(.5)
    total_graph.replay(); torch.cuda.synchronize()
    if any(torch.equal(a,b) for a,b in zip(changed_input_grads,total_grads)):
        raise RuntimeError("captured backward ignored changed upstream gradient")
    guard_errors["changed_dy"]=current_buffer_guard("changed_dy")
    with torch.no_grad():
        w.mul_(.5)
    total_graph.replay(); torch.cuda.synchronize()
    guard_errors["changed_w"]=current_buffer_guard("changed_w")
    assert all(torch.isfinite(t).all() for t in (total_output,*total_grads))
    del initial_output,initial_grads,changed_input_grads
    # Restore fixed benchmark inputs; guards and clones are outside timing.
    with torch.no_grad():
        x.mul_(2); dy.mul_(2); w.mul_(2)
    forward_graph.replay()
    backward_graph.replay()
    torch.cuda.synchronize()
    backward_previous=tuple(t.clone() for t in backward_grads)
    backward_graph.replay(); torch.cuda.synchronize()
    for a,b,policy in zip(backward_previous,backward_grads,(dx,dw)):
        if (not torch.equal(a,b)) != (policy=="ms_eden"):
            raise RuntimeError("standalone backward replay RNG policy mismatch")
    del backward_previous
    measurements={}
    for name,graph in (() if capture_only else (("forward",forward_graph),("backward",backward_graph),("total",total_graph))):
        graph.replay(); torch.cuda.synchronize()
        values,batching=measure_cuda_batches(graph.replay,samples=args.samples,
            calls_per_sample=args.calls_per_sample,device=x.device)
        measurements[name]=timing_summary(values,batching)
    torch.cuda.synchronize()
    live_before=torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(20):
        total_graph.replay()
    torch.cuda.synchronize()
    live_after=torch.cuda.memory_allocated()
    memory=dict(baseline_before_capture_bytes=baseline,capture_peak_allocated_bytes=capture_peak,
        ordinary_compiled_allocation_audit=ordinary_memory,
        capture_peak_above_baseline_bytes=capture_peak-baseline,
        persistent_after_capture_bytes=persistent_after_capture,
        persistent_above_baseline_bytes=persistent_after_capture-baseline,
        live_before_replays_bytes=live_before,live_after_replays_bytes=live_after,
        replay_live_growth_bytes=live_after-live_before,
        replay_peak_allocated_bytes=torch.cuda.max_memory_allocated())
    if live_after!=live_before:
        raise RuntimeError("live allocated memory changed during repeated graph replay")
    # All timings and replays are complete. Release only this fresh worker's
    # graph pins and frontend workspace caches; never clear another workload.
    if capture_only:
        bundle=dict(inputs=(x,w,dy),graph=total_graph,output=total_output,grads=total_grads,
                    pins=(forward_graph,backward_graph,forward_output,backward_grads,compiled))
    else:
        del forward_output,backward_grads,total_output,total_grads
        del graph,forward_graph,backward_graph,total_graph
        gc.collect()
        torch.cuda.synchronize()
        from rtx.runtime import clear_runtime_caches
        clear_runtime_caches(preserve_compiled=True)
        gc.collect()
        memory["idle_after_graph_release_and_cache_cleanup_bytes"]=torch.cuda.memory_allocated()
    result=dict(policy=args.policy,dx_policy=dx,dw_policy=dw,shape=[args.m,args.n,args.k],
        backward_integration="stable paired MXFP8" if args.policy=="stable_mxfp8" else "research independent partial phases",
        seed=args.seed,torch=torch.__version__,cuda=torch.version.cuda,device=str(torch.cuda.get_device_properties(0)),
        command=sys.argv,source_sha256=source_hashes(),forward="NVFP4 exact jit_row_region",source_policy="original_bf16",
        input_alias_contract="independent_autograd_leaves_shared_storage_v1",
        forward_config=asdict(forward_config),mxfp8_backward_config=asdict(config),
        schedule_json=args.schedule_json,dx_schedule_json=args.dx_schedule_json,dw_schedule_json=args.dw_schedule_json,
        backward_pipeline=getattr(args,"backward_pipeline","serial"),
        timings=measurements,memory=memory,guards=dict(forward_bitwise_matched=True,
        current_input_replay=True,current_upstream_replay=True,current_weight_replay=True,
        fresh_rng_per_phase=list(fresh),current_buffer_gradient_checks=guard_errors,
        gradient_gate_scope="Gaussian algorithm-quantization NRMS versus independent FP32 GEMMs on current BF16 inputs; stale-buffer guard, not packed-kernel numerical qualification"),
        scope="CUDA graph replay of public fullgraph compiled forward + autograd.grad; forward/backward separately captured; total independently captured; excludes eager host launch overhead, compile, capture and input generation; not optimizer/training wall throughput")
    return (result,bundle) if capture_only else result


def paired_statistics(rounds,*,seed=941,bootstrap_samples=4000):
    """Bootstrap entire AB/BA rounds, never individual replays as IID samples."""
    import math
    if not rounds or bootstrap_samples<1:
        raise ValueError("positive round and bootstrap sample counts required")
    if any(not math.isfinite(r[key]) or r[key]<=0 for r in rounds
           for key in ("reference_ms","candidate_ms")):
        raise ValueError("paired timing samples must be finite and positive")
    logs=[math.log(r["reference_ms"]/r["candidate_ms"]) for r in rounds]
    rng=random.Random(seed)
    estimates=sorted(math.exp(statistics.mean(rng.choices(logs,k=len(logs))))
                     for _ in range(bootstrap_samples))
    return dict(geomean_reference_over_candidate=math.exp(statistics.mean(logs)),
        round_cluster_bootstrap95=[estimates[int(.025*bootstrap_samples)],estimates[int(.975*bootstrap_samples)]],
        rounds=len(rounds),bootstrap_samples=bootstrap_samples,
        interpretation="within-session paired graph replay only; ratio>1 favors candidate; rounds are bootstrap clusters")


def run_paired(args):
    import torch
    built={}
    metadata={}
    shared=None
    specifications=(("stable", "stable_mxfp8","coalesced_wide"),
                    ("fp4_wide","fp4_fp4","coalesced_wide"),
                    ("mixed_wide","fp4_mxfp8","coalesced_wide"),
                    ("fp4_coalesced","fp4_fp4","coalesced"))
    for name,policy,load in specifications:
        options=argparse.Namespace(**vars(args))
        options.policy=policy
        schedule=json.loads(args.schedule_json)
        schedule["producer_load"]=load
        options.schedule_json=json.dumps(schedule,sort_keys=True)
        if options.dx_schedule_json is not None or options.dw_schedule_json is not None:
            raise ValueError("paired controlled load comparison uses shared schedule; omit per-phase overrides")
        metadata[name],built[name]=run_worker(options,capture_only=True,shared_inputs=shared)
        shared=built[name]["inputs"]
    # All configurations and graph pools exist before any timing sample.
    for _ in range(args.warmup):
        for item in built.values():
            item["graph"].replay()
    torch.cuda.synchronize()
    def batch(name):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.replays):
            built[name]["graph"].replay()
        end.record(); end.synchronize()
        return start.elapsed_time(end)/args.replays
    results={}
    for reference,candidate in (("stable","fp4_wide"),("stable","mixed_wide"),("fp4_coalesced","fp4_wide")):
        rounds=[]
        for index in range(args.rounds):
            orders=((reference,candidate),(candidate,reference))
            if index%2:
                orders=orders[::-1]
            windows=[]
            for order in orders:
                values={name:batch(name) for name in order}
                windows.append(dict(order=list(order),order_kind="AB" if order[0]==reference else "BA",
                    reference_ms=values[reference],candidate_ms=values[candidate],
                    ratio=values[reference]/values[candidate]))
            ref=(windows[0]["reference_ms"]*windows[1]["reference_ms"])**.5
            cand=(windows[0]["candidate_ms"]*windows[1]["candidate_ms"])**.5
            rounds.append(dict(round=index,windows=windows,reference_ms=ref,candidate_ms=cand,ratio=ref/cand))
        by_order={kind:paired_statistics([window for row in rounds for window in row["windows"]
                                          if window["order_kind"]==kind],seed=args.seed)
                  for kind in ("AB","BA")}
        results[f"{reference}__vs__{candidate}"]=dict(raw_rounds=rounds,by_order=by_order,
            **paired_statistics(rounds,seed=args.seed))
    return dict(mode="within-session paired complete public compiled graph replay",shape=[args.m,args.n,args.k],
        shared_input_storage=True,replays_per_batch=args.replays,source_sha256=source_hashes(),
        configs=metadata,comparisons=results,
        caveat="All graphs built and warmed before timing; RNG is inside complete fwd+bwd graphs; excludes Python autograd/optimizer wall throughput; CI is within-session only, not cross-session qualification")


def main(argv=None):
    args=parser().parse_args(argv)
    validate(args)
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    if args.paired:
        result=run_paired(args)
    elif args.worker:
        if args.policy=="all":
            raise ValueError("worker requires exactly one policy")
        result=run_worker(args)
    else:
        policies=POLICIES if args.policy=="all" else (args.policy,)
        results={}
        for policy in policies:
            child=output.with_name(output.stem+"."+policy+".json")
            subprocess.run(worker_command(args,policy,child),check=True)
            results[policy]=json.loads(child.read_text())
        result=dict(fresh_process_per_policy=True,results=results)
    with output.open("x") as handle:
        json.dump(result,handle,indent=2)
    print(json.dumps(result))


if __name__=="__main__":
    main()
