"""Paired public compiled MS-EDEN backward comparison against the 256-CTA baseline.

Measures caller RNG plus complete backward, and independently captured full
forward+backward CUDA graphs. Not a substitute for optimizer/training wall time.
Different RNG implementations preserve distribution, not bit-identical draws;
use test_ms_eden_rng_independent_cuda.py for identical-draw kernel equivalence.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

PUBLIC_PHASE_MEASUREMENT_CONTRACT="public_rng_graph8_samples9_replays16_warm200ms_calibrated_cudaevents_v2"

def schedule_pair(overrides=None):
    from rtx.autotune.posthoc import PosthocPreparationSchedule
    base=dict(producer_threads=128,producer_blocks=256,producer_load="coalesced_wide",
              finalizer_threads=128,finalizer_blocks=224)
    base.update(overrides or {})
    result={}
    for phase,order in (("dx","row_major"),("dw","reduction_major")):
        values=dict(base)
        values.setdefault("producer_order",order)
        result[phase]=json.dumps(asdict(PosthocPreparationSchedule(**values)),sort_keys=True)
    return result


def options(shape,schedules,seed=941,policy="fp4_fp4"):
    from benchmarks.compare_mixed_nvfp4_backward import parser
    args=parser().parse_args(["--output","unused.json"])
    args.m,args.n,args.k=shape
    args.seed=seed
    args.policy=policy
    args.warmup=3
    args.samples=3
    args.dx_schedule_json=schedules["dx"]
    args.dw_schedule_json=schedules["dw"]
    return args


def evaluate_public_phase(shape,phase,schedule_dict,samples=9,replays=16):
    """Autotuner objective: compiled caller RNG + complete requested gradient.

    Gaussian FP32 error is a broad stale/corrupt-output gate; exact numerical
    qualification belongs to the independent identical-draw reference tests.
    """
    import torch
    from rtx.autotune.posthoc import PosthocPreparationSchedule
    from rtx.autotune.measurement import measure_cuda_batches,runtime_timing_metadata
    from rtx.experimental_ms_eden import _random_gradient
    if len(shape)!=3 or any(type(v) is not int or v<=0 or v%128 for v in shape):
        raise ValueError("shape must contain positive multiples of 128")
    if phase not in ("dx","dw") or samples<3 or replays<1:
        raise ValueError("dx/dw, samples>=3 and positive replay count required")
    schedule=PosthocPreparationSchedule(**schedule_dict)
    if schedule.reader_mode!="direct":
        raise ValueError("public phase evaluator requires original BF16 direct reader")
    encoded=json.dumps(asdict(schedule),sort_keys=True)
    m,n,k=shape
    torch.manual_seed(941)
    dy=torch.randn(m,n,device="cuda",dtype=torch.bfloat16)*.25
    operand=torch.randn(n if phase=="dx" else m,k,device="cuda",dtype=torch.bfloat16)*.02
    def call(a,b): return _random_gradient(a,b,phase,encoded)
    compiled=torch.compile(call,fullgraph=True,dynamic=False,options={"triton.cudagraphs":False})
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3): temporary=compiled(dy,operand)
    stream.synchronize()
    del temporary
    before_capture=torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):
        for _ in range(8): result=compiled(dy,operand)
    torch.cuda.synchronize()
    capture_peak=torch.cuda.max_memory_allocated()
    graph.replay()
    first=result.clone()
    graph.replay()
    if torch.equal(first,result): raise RuntimeError("public phase captured stale RNG")
    del first
    reference=dy.float()@operand.float() if phase=="dx" else dy.float().T@operand.float()
    def errors():
        if not result.isfinite().all(): raise RuntimeError("nonfinite public gradient")
        error=(result.float()-reference).double()
        ref=reference.double()
        nrms=float(error.norm()/ref.norm().clamp_min(1e-12))
        if nrms>.3: raise RuntimeError(f"public gradient Gaussian NRMS {nrms} exceeds .3 sanity gate")
        return dict(nrms=nrms,max_abs=float(error.abs().max()),rms=float(error.square().mean().sqrt()),
                    reference_rms=float(ref.square().mean().sqrt()),gate=.3)
    initial=errors()
    deadline=time.monotonic()+.2
    while time.monotonic()<deadline:
        for _ in range(8): graph.replay()
        torch.cuda.synchronize()
    allocated=torch.cuda.memory_allocated()
    graph_timings,batching=measure_cuda_batches(graph.replay,samples=samples,calls_per_sample=replays,device=dy.device)
    timings=[value/8 for value in graph_timings]
    batching=dict(batching,gradients_per_graph=8,timing_unit="milliseconds_per_gradient")
    final=errors()
    torch.cuda.synchronize()
    after=torch.cuda.memory_allocated()
    if after!=allocated: raise RuntimeError("live allocation changed during phase replay")
    quality=runtime_timing_metadata(timings)
    return dict(shape=shape,phase=phase,schedule=asdict(schedule),torch=torch.__version__,
        measurement_contract=PUBLIC_PHASE_MEASUREMENT_CONTRACT if (samples,replays)==(9,16)
            else f"public_rng_graph8_samples{samples}_replays{replays}_warm200ms_calibrated_cudaevents_v2",
        cuda=torch.version.cuda,device=str(torch.cuda.get_device_properties(0)),
        timings=dict(median_ms=statistics.median(timings),min_ms=min(timings),samples_ms=timings,batching=batching),
        quality=quality,metrics=dict(initial=initial,final=final),
        memory=dict(capture_peak_above_baseline_bytes=capture_peak-before_capture,
                    replay_live_growth_bytes=after-allocated),
        guards=dict(fresh_rng=True,finite=True,fullgraph=True),
        scope="Compiled original-BF16 public random-gradient path, caller RNG and complete gradient included, eight gradients per CUDA graph normalized per gradient; no forward/optimizer/compile time")


def compare(shape,output,candidate_overrides=None,rounds=8,replays=40,include_hybrid=False):
    import torch
    from benchmarks.compare_mixed_nvfp4_backward import run_worker,paired_statistics
    if len(shape)!=3 or any(type(v) is not int or v<=0 or v%128 for v in shape):
        raise ValueError("shape must contain positive multiples of 128")
    if not 4<=rounds<=32 or not 1<=replays<=100:
        raise ValueError("require 4..32 rounds and 1..100 graph replays")
    output=Path(output)
    if output.exists(): raise FileExistsError(output)
    candidate=dict(rng_mode="philox")
    candidate.update(candidate_overrides or {})
    specs={"baseline_256":("fp4_fp4",schedule_pair()),
           "candidate":("fp4_fp4",schedule_pair(candidate))}
    if include_hybrid: specs["hybrid"]=("stable_mxfp8",schedule_pair())
    bundles,metadata={},{}
    shared=None
    for name,(policy,schedules) in specs.items():
        metadata[name],bundles[name]=run_worker(options(shape,schedules,policy=policy),
                                               capture_only=True,shared_inputs=shared)
        shared=bundles[name]["inputs"]
    x,w,dy=shared
    graphs={name:{"total":bundle["graph"],"backward":bundle["pins"][1]}
            for name,bundle in bundles.items()}
    # Separate complete graphs include RNG. Preconditioning is outside timing.
    deadline=time.monotonic()+3.
    while time.monotonic()<deadline:
        for bundle in bundles.values(): bundle["graph"].replay()
        torch.cuda.synchronize()
    def batch(graph):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays): graph.replay()
        end.record(); end.synchronize()
        return start.elapsed_time(end)/replays
    rng=random.Random(941)
    rows=[]
    before=torch.cuda.memory_allocated()
    for round_id in range(rounds):
        order=list(bundles); rng.shuffle(order)
        for scope in ("backward","total"):
            for sequence in (order,list(reversed(order))):
                values={name:batch(graphs[name][scope]) for name in sequence}
                rows.append(dict(round=round_id,scope=scope,order=sequence,ms=values))
    torch.cuda.synchronize()
    after=torch.cuda.memory_allocated()
    if after!=before: raise RuntimeError("live CUDA memory changed during replay timing")
    # Finiteness/current-gradient checks after all timings; never in timed loops.
    errors={}
    references=(dy.float()@w.detach().float(),dy.float().T@x.detach().float())
    for name,bundle in bundles.items():
        bundle["graph"].replay()
        phases={}
        for phase,actual,reference in zip(("dx","dw"),bundle["grads"],references):
            if not actual.isfinite().all(): raise RuntimeError(f"{name}/{phase}: nonfinite gradient")
            error=(actual.float()-reference).double()
            ref=reference.double()
            nrms=float(error.norm()/ref.norm().clamp_min(1e-12))
            gate=.06 if name=="hybrid" else .25
            if nrms>gate: raise RuntimeError(f"{name}/{phase}: stale/invalid gradient NRMS {nrms}")
            phases[phase]=dict(nrms=nrms,max_abs=float(error.abs().max()),
                rms=float(error.square().mean().sqrt()),reference_rms=float(ref.square().mean().sqrt()),
                gate=gate)
        errors[name]=phases
    summary={}
    for scope in ("backward","total"):
        observations=[r for r in rows if r["scope"]==scope]
        paired=[]
        for round_id in range(rounds):
            complete_round=[r for r in observations if r["round"]==round_id]
            paired.append(dict(reference_ms=statistics.mean(r["ms"]["baseline_256"] for r in complete_round),
                candidate_ms=statistics.mean(r["ms"]["candidate"] for r in complete_round)))
        summary[scope]=dict(median_ms={name:statistics.median(r["ms"][name] for r in observations)
                                      for name in bundles},paired=paired_statistics(paired))
    root=Path(__file__).resolve().parents[1]
    paths=("benchmarks/compare_ms_eden_amortization.py","benchmarks/compare_mixed_nvfp4_backward.py",
           "rtx/experimental_ms_eden.py","rtx/kernels/nvfp4_posthoc.py","rtx/kernels/nvfp4_rng.py",
           "rtx/autotune/posthoc.py")
    m,n,k=shape
    memory={}
    for name,record in metadata.items():
        allocation=record["memory"]
        ordinary=allocation["ordinary_compiled_allocation_audit"]
        memory[name]=dict(capture_persistent_above_baseline_bytes=allocation["persistent_above_baseline_bytes"],
            capture_peak_above_baseline_bytes=allocation["capture_peak_above_baseline_bytes"],
            ordinary_peak_above_baseline_bytes=ordinary["peak_allocated_bytes"]-ordinary["baseline_allocated_bytes"],
            ordinary_live_growth_bytes=ordinary["live_growth_bytes"],
            replay_live_growth_bytes=allocation["replay_live_growth_bytes"])
    result=dict(shape=shape,torch=torch.__version__,cuda=torch.version.cuda,
        device=str(torch.cuda.get_device_properties(0)),rounds=rounds,replays=replays,
        source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in paths},
        configurations=specs,build_guards=metadata,gradient_checks=errors,
        replay_live_growth_bytes=after-before,memory=memory,
        modeled_rng=dict(explicit_uniform_payload_bytes=(2*m*n+n*k+m*k)//4,
            philox_seed_payload_bytes=32,unchanged_sign_payload_bytes=1024,
            candidate_rng_mode=candidate.get("rng_mode","explicit"),
            interpretation="Alternative logical per-backward RNG payloads only; Philox saving applies only when selected, not measured HBM traffic; signs shared by two operands per phase; actual capture/peak allocation measured separately"),
        timings=summary,samples=rows,
        scope="Paired CUDA graph replay of compiled public original-BF16-source backward and full forward+backward, caller RNG included; no optimizer, data loading or compilation; not full training wall",
        numerical_scope="FP32-gradient gates are broad stale-input sanity checks; independent identical-draw CUDA tests qualify RNG fusion arithmetic")
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x") as handle: json.dump(result,handle,indent=2)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shape",default="4096,2048,512")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--candidate-json",default="{}")
    p.add_argument("--rounds",type=int,default=8)
    p.add_argument("--replays",type=int,help="defaults to16 for phase tuning or40 for paired comparison")
    p.add_argument("--include-hybrid",action="store_true")
    p.add_argument("--evaluate-phase",choices=("dx","dw"),help="single-config public-RNG autotuner objective")
    p.add_argument("--schedule-json",default="{}",help="JSON string for --evaluate-phase")
    args=p.parse_args()
    replays=args.replays if args.replays is not None else (16 if args.evaluate_phase else 40)
    shape=tuple(map(int,args.shape.split(',')))
    if args.evaluate_phase:
        if args.output.exists(): raise FileExistsError(args.output)
        result=evaluate_public_phase(shape,args.evaluate_phase,json.loads(args.schedule_json),replays=replays)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open("x") as handle: json.dump(result,handle,indent=2)
    else:
        result=compare(shape,args.output,json.loads(args.candidate_json),
            args.rounds,replays,args.include_hybrid)
    print(json.dumps(result["timings"],indent=2))


if __name__=="__main__": main()
