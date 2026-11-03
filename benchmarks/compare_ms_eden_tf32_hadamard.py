"""Diagnostic scalar/guarded-TF32 H128 transform comparison, NOT backward speed.

Both routes use the same prototype ABI and original BF16/source/sign buffers.
MMA reassociation is quality-gated against dense FP64 H128, not byte-exact.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time


def shape_value(value):
    try:shape=tuple(map(int,value.lower().replace("x",",").split(",")))
    except ValueError as error:raise argparse.ArgumentTypeError("shape requires rows,cols") from error
    if len(shape)!=2 or min(shape)<1 or shape[1]%128:
        raise argparse.ArgumentTypeError("positive rows and columns divisible by 128 required")
    return shape


def positive_int(value):
    try:number=int(value)
    except ValueError as error:raise argparse.ArgumentTypeError("positive integer required") from error
    if number<1:raise argparse.ArgumentTypeError("positive integer required")
    return number


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shape",type=shape_value,default=(4096,1536))
    p.add_argument("--threads",type=int,choices=(64,128,256),default=128)
    p.add_argument("--blocks",type=int,default=128)
    p.add_argument("--mma-kind",choices=("tf32","bf16_padded","bf16_h16"),default="tf32")
    p.add_argument('--h16-store-bits',type=int,choices=(32,64),default=32)
    p.add_argument("--seed",type=int,default=959)
    p.add_argument("--distribution",choices=("gaussian","outliers","cancellation","zero"),default="gaussian")
    p.add_argument("--warmup-seconds",type=float,default=15.)
    p.add_argument("--rounds",type=int,default=24)
    p.add_argument("--replays",type=int,default=16)
    p.add_argument("--graph-batch",type=positive_int,default=8,
                   help="same kernel calls captured per graph; times normalized per transform (1 allowed)")
    p.add_argument("--min-batch-ms",type=float,default=2.)
    p.add_argument("--output",type=Path,required=True)
    return p


def dense_fp64_oracle(source,signs):
    import torch
    h=torch.tensor([[1. if (i&j).bit_count()%2==0 else -1. for j in range(128)]
                    for i in range(128)],device=source.device,dtype=torch.float64)
    return ((source.double().reshape(-1,128)*signs.double())@h/math.sqrt(128.)).flatten()


def quality(actual,reference,gate=1e-5):
    import torch
    if not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
        raise RuntimeError("nonfinite transform output/FP64 oracle")
    error=actual.double()-reference
    nrms=float(error.norm()/reference.norm().clamp_min(1e-300))
    normalized_max=float(error.abs().max()/reference.abs().max().clamp_min(1e-300))
    result=dict(nrms=nrms,normalized_max=normalized_max,max_abs=float(error.abs().max()),
        rms=float(error.square().mean().sqrt()),reference_rms=float(reference.square().mean().sqrt()),
        denominator_floor=1e-300,gate=gate,finite=True)
    if nrms>gate or normalized_max>gate:raise RuntimeError(f"FP64 pure-transform quality gate failed: {result}")
    return result


def timing_summary(rows,seed):
    from benchmarks.compare_mixed_nvfp4_backward import paired_statistics
    from rtx.autotune.measurement import runtime_timing_metadata
    quality={name:runtime_timing_metadata([row[key]for row in rows])
             for name,key in (("baseline","reference_ms"),("candidate","candidate_ms"))}
    return dict(paired=paired_statistics(rows,seed=seed),
        median_ms={name:statistics.median(row[key]for row in rows)
                   for name,key in (("baseline","reference_ms"),("candidate","candidate_ms"))},
        quality=quality,stationary=all(item["sampling"]["collection"]["stationary"]for item in quality.values()),
        by_order={kind:paired_statistics([window for row in rows for window in row["windows"]
                                        if window["order_kind"]==kind],seed=seed)for kind in ("AB","BA")})


def window_times(elapsed_ms,replays,graph_batch):
    """CUDA-event window covers graph replays; each graph holds B transforms."""
    if replays<1 or graph_batch<1:raise ValueError("positive replay and graph batch counts required")
    return dict(elapsed_window_ms=elapsed_ms,graph_replays=replays,
                transforms=replays*graph_batch,per_graph_replay_ms=elapsed_ms/replays,
                per_transform_ms=elapsed_ms/(replays*graph_batch))


def run(args):
    import torch
    from benchmarks.compare_ms_eden_pipeline import round_orders
    from rtx.autotune.measurement import measure_cuda_batches
    from rtx.kernels.ms_eden_tf32_hadamard import TF32H8Config,TF32H8HadamardKernel,compile_tf32_h8_hadamard
    if args.output.exists():raise FileExistsError(args.output)
    if (not 4<=args.rounds<=64 or args.replays<1 or type(args.graph_batch) is not int or args.graph_batch<1 or args.min_batch_ms<2
            or not math.isfinite(args.min_batch_ms) or not math.isfinite(args.warmup_seconds) or args.warmup_seconds<0):
        raise ValueError("require 4..64 rounds, positive replays, finite batch>=2ms and nonnegative warmup")
    rows,cols=args.shape
    configs={name:TF32H8Config(args.threads,args.blocks,force_scalar=scalar,
                             mma_kind='tf32' if args.mma_kind=='bf16_h16' else args.mma_kind)
             for name,scalar in (("baseline",True),("candidate",False))}
    compilers={name:compile_tf32_h8_hadamard for name in configs}
    constructors={name:TF32H8HadamardKernel for name in configs}
    if args.mma_kind=='bf16_h16':
        from rtx.kernels.ms_eden_h16_hadamard import H16Config,H16HadamardKernel,compile_h16_hadamard
        configs['candidate']=H16Config(args.threads,args.blocks,output_store_bits=args.h16_store_bits)
        compilers['candidate']=compile_h16_hadamard
        constructors['candidate']=H16HadamardKernel
    torch.manual_seed(args.seed)
    source=torch.empty(rows,cols,device="cuda",dtype=torch.bfloat16)
    signs=torch.ones(128,device="cuda")
    outputs={name:torch.empty(rows*cols,device="cuda")for name in configs}
    routes={name:torch.empty(rows*cols//128,device="cuda",dtype=torch.int32)for name in configs}
    runners={name:compilers[name](rows,cols,config)for name,config in configs.items()}
    source_flat=source.flatten()
    def fill():
        source.normal_(0,.25);signs.random_(0,2).mul_(2).sub_(1)
        if args.distribution=="outliers":source_flat[::97].mul_(32)
        if args.distribution=="cancellation":
            source.fill_(1.);source[:,1::2].fill_(-1.);source[:,::17].add_(.0078125);signs.fill_(1.)
        if args.distribution=="zero":source.zero_();signs.fill_(1.)
    fill()
    graphs={}
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    # Warm and capture on the SAME explicit stream; no implicit capture stream.
    for name in configs:
        with torch.cuda.stream(stream):
            for _ in range(3):runners[name](source_flat,signs,outputs[name],routes[name])
        stream.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):
            for _ in range(args.graph_batch):runners[name](source_flat,signs,outputs[name],routes[name])
        graphs[name]=graph
    def routing():
        if any(not ((value==0)|(value==1)).all()for value in routes.values()):
            raise RuntimeError("invalid/unwritten transform route flag")
        expected=((source_flat.abs()==0)|((source_flat.abs()>=2.**-96)&(source_flat.abs()<=2.**96)))
        expected=expected.reshape(-1,128).all(1)&(signs.abs()==1).all()
        if args.mma_kind=='bf16_h16':
            paired=expected
            if expected.numel()%2:paired=torch.cat((expected,torch.ones(1,device=expected.device,dtype=torch.bool)))
            expected=paired.reshape(-1,2).all(1).repeat_interleave(2)[:expected.numel()]
        if not torch.equal(routes["candidate"].bool(),expected) or not (routes["baseline"]==0).all():
            raise RuntimeError("transform route flags disagree with guarded input/sign domain")
        return {name:dict(tc_tiles=int((value==1).sum()),scalar_tiles=int((value==0).sum()),
                          total_tiles=value.numel())for name,value in routes.items()}
    def validate():
        for graph in graphs.values():graph.replay()
        torch.cuda.synchronize()
        reference=dense_fp64_oracle(source,signs)
        return dict(metrics={name:quality(value,reference)for name,value in outputs.items()},routing=routing(),
            differing_fp32_words=int((outputs["baseline"].view(torch.int32)!=outputs["candidate"].view(torch.int32)).sum()))
    checks={"initial":validate()}
    fill()  # Same storage, changed even for deterministic cancellation/zero.
    source_flat[0]=1.5;signs.neg_()
    checks["changed_buffers"]=validate()
    signs[5]=.75
    checks["changed_sign_fallback"]=validate()
    if not torch.equal(outputs["baseline"].view(torch.int32),outputs["candidate"].view(torch.int32)):
        raise RuntimeError("non-Rademacher fallback is not bitwise scalar")
    fill()
    checks["timed_buffers"]=validate()
    pointers=dict(source=source.data_ptr(),signs=signs.data_ptr(),
                  **{name+"_output":value.data_ptr()for name,value in outputs.items()},
                  **{name+"_route":value.data_ptr()for name,value in routes.items()})
    deadline=time.monotonic()+args.warmup_seconds
    while time.monotonic()<deadline:
        for _ in range(16):
            for graph in graphs.values():graph.replay()
        torch.cuda.synchronize()
    calibration={}
    for name,graph in graphs.items():
        _,calibration[name]=measure_cuda_batches(graph.replay,samples=1,calls_per_sample=args.replays,
            min_batch_ms=2*args.min_batch_ms,max_calls=65536,device=source.device)
    count=max(value["actual_calls_per_sample"]for value in calibration.values())
    def batch(graph):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(count):graph.replay()
        end.record();end.synchronize()
        elapsed=float(start.elapsed_time(end))
        if elapsed<args.min_batch_ms:raise RuntimeError("timing window below requested minimum; no acceptance")
        return window_times(elapsed,count,args.graph_batch)
    torch.cuda.synchronize()
    before=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
    samples=[]
    for index,orders in round_orders(args.rounds,args.seed):
        windows=[]
        for order in orders:
            values={name:batch(graphs[name])for name in order}
            windows.append(dict(order=list(order),order_kind="AB" if order[0]=="baseline" else "BA",
                reference_ms=values["baseline"]["per_transform_ms"],candidate_ms=values["candidate"]["per_transform_ms"],
                reference_graph_replay_ms=values["baseline"]["per_graph_replay_ms"],
                candidate_graph_replay_ms=values["candidate"]["per_graph_replay_ms"],
                elapsed_window_ms={name:value["elapsed_window_ms"]for name,value in values.items()},
                raw_measurements=values))
        samples.append(dict(round=index,windows=windows,
            reference_ms=math.sqrt(math.prod(w["reference_ms"]for w in windows)),
            candidate_ms=math.sqrt(math.prod(w["candidate_ms"]for w in windows)),
            reference_graph_replay_ms=math.sqrt(math.prod(w["reference_graph_replay_ms"]for w in windows)),
            candidate_graph_replay_ms=math.sqrt(math.prod(w["candidate_graph_replay_ms"]for w in windows))))
    torch.cuda.synchronize()
    after=torch.cuda.memory_allocated();peak=torch.cuda.max_memory_allocated()
    if after!=before:raise RuntimeError("live allocation changed during transform replay timing")
    checks["after_timing"]=validate()
    root=Path(__file__).resolve().parents[1]
    paths=("benchmarks/compare_ms_eden_tf32_hadamard.py","rtx/kernels/ms_eden_tf32_hadamard.py",
           "rtx/autotune/measurement.py","benchmarks/compare_mixed_nvfp4_backward.py","benchmarks/compare_ms_eden_pipeline.py")
    if args.mma_kind=='bf16_h16':paths+=('rtx/kernels/ms_eden_h16_hadamard.py',)
    result=dict(shape=list(args.shape),seed=args.seed,distribution=args.distribution,command=sys.argv,
        executable=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,
        device=str(torch.cuda.get_device_properties(source.device)),
        environment={key:os.environ.get(key)for key in ("CUDA_VISIBLE_DEVICES","CUTE_DSL_ARCH","CUDA_MODULE_LOADING","OMP_NUM_THREADS")},
        transform_variant=args.mma_kind,
        configs={name:dict(asdict(config),actual_blocks=constructors[name](rows,cols,config).blocks)
                 for name,config in configs.items()},checks=checks,buffer_pointers=pointers,
        timing=timing_summary(samples,args.seed),samples=samples,calibration=calibration,
        calls_per_window=count,graph_replays_per_window=count,graph_batch=args.graph_batch,
        transforms_per_window=count*args.graph_batch,measurement_revision=2,
        raw_graph_replay_median_ms={name:statistics.median(row[key]for row in samples)
            for name,key in (("baseline","reference_graph_replay_ms"),("candidate","candidate_graph_replay_ms"))},
        timing_units="timing/reference_ms/candidate_ms are milliseconds per transform; raw graph replay times and whole-event windows retained separately",
        rounds=args.rounds,warmup_seconds=args.warmup_seconds,
        memory=dict(before_allocated_bytes=before,after_allocated_bytes=after,
                    replay_live_growth_bytes=after-before,replay_peak_allocated_bytes=peak),
        source_sha256={path:hashlib.sha256((root/path).read_bytes()).hexdigest()for path in paths},
        scope="DIAGNOSTIC ONLY: original-BF16 signed normalized Hadamard128 graph latency; no quantization, EDEN statistics, RNG, GEMM, gradients or complete backward. Not a backward speedup or production acceptance.",
        oracle="independent dense FP64 signed H128 / sqrt(128); NRMS and normalized-max <=1e-5; scalar fallback bitwise",
        sampling="randomized ABBA/BAAB rounds; full rounds bootstrap clusters; graph_batch calls on identical buffers per graph; event windows cover whole graph replays; compilation/oracle/capture/warmup excluded")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x")as handle:json.dump(result,handle,indent=2)
    return result


def main(argv=None):
    result=run(parser().parse_args(argv))
    print(json.dumps(dict(timing=result["timing"],scope=result["scope"]),indent=2))
    return result


if __name__=="__main__":main()
