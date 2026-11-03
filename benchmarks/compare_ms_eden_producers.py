"""Bounded paired graph timing of unchanged MS-EDEN preparation policies.

Compares coalesced vs wider coalesced source tiles, including complete GEMM.
Diagnostic explicit randomness is fixed; this is not public compiled backward.
"""
import argparse
import json
from pathlib import Path
import random
import statistics
import time

import torch


def compare(shape,phase,rounds=12,replays=40,precondition_seconds=3.):
    if phase not in ("dx","dw") or len(shape)!=3 or any(type(d) is not int or d<=0 or d%128 for d in shape):
        raise ValueError("dx/dw and positive dimensions divisible by 128 required")
    from benchmarks.study_ms_eden_backward import prepare
    from rtx.autotune.posthoc import PosthocPreparationSchedule
    from rtx.autotune.measurement import runtime_timing_metadata
    from rtx.kernels.nvfp4_posthoc import compile_posthoc_scale_product
    from rtx.configs.nvfp4 import NVFP4Problem
    from rtx.fp4 import compile_nvfp4_gemm
    if not 2 <= rounds <= 24 or not 1 <= replays <= 100 or not 0 <= precondition_seconds <= 10:
        raise ValueError("bounded rounds/replays/precondition required")
    torch.manual_seed(825)
    m,n,k=shape
    dy=torch.randn(m,n,device="cuda",dtype=torch.bfloat16)*.25
    source=torch.randn(n if phase=="dx" else m,k,device="cuda",dtype=torch.bfloat16)*.02
    signs=(torch.randint(2,(128,),device="cuda")*2-1).float()
    problem=NVFP4Problem(m if phase=="dx" else n,k,n if phase=="dx" else m)
    runners={}
    for mode in ("coalesced","coalesced_wide"):
        schedule=PosthocPreparationSchedule(producer_load=mode)
        a=prepare(dy,phase=="dw","mma128",schedule)
        b=prepare(source,True,"mma128",schedule)
        alpha=torch.empty(1,device="cuda")
        out=torch.empty(problem.m,problem.n,device="cuda",dtype=torch.bfloat16)
        product=compile_posthoc_scale_product()
        gemm=compile_nvfp4_gemm(problem,schedule.gemm)
        def full(a=a,b=b,alpha=alpha,out=out,product=product,gemm=gemm):
            a["run"](signs); b["run"](signs)
            product(a["factor"],b["factor"],alpha)
            gemm(a["packed"],b["packed"],a["scales"],b["scales"],out,alpha)
        full()
        runners[mode]=dict(full=full,a=a,b=b,out=out)
    baseline,candidate=runners.values()
    for operand in ("a","b"):
        candidate[operand]["uniforms"].copy_(baseline[operand]["uniforms"])
    def check():
        for runner in runners.values():
            runner["full"]()
        for operand in ("a","b"):
            for name in ("packed","scales","factor"):
                torch.testing.assert_close(baseline[operand][name].view(torch.uint8),
                    candidate[operand][name].view(torch.uint8),rtol=0,atol=0)
        torch.testing.assert_close(baseline["out"],candidate["out"],rtol=0,atol=0)
        assert baseline["out"].isfinite().all()
    check()
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    for runner in runners.values():
        with torch.cuda.stream(stream):
            for _ in range(3):
                runner["full"]()
        stream.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):
            for _ in range(8):
                runner["full"]()
        runner["graph"]=graph
    deadline=time.monotonic()+precondition_seconds
    while time.monotonic()<deadline:
        for runner in runners.values():
            for _ in range(8):
                runner["graph"].replay()
        torch.cuda.synchronize()
    rng=random.Random(825)
    samples=[]
    for index in range(rounds):
        # Changed sources and rotation each round, identical across candidates.
        dy.normal_(0,.25); source.normal_(0,.02)
        signs.random_(0,2).mul_(2).sub_(1)
        check()
        order=list(runners)
        rng.shuffle(order)
        for sequence in (order,list(reversed(order))):
            timings={}
            for name in sequence:
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(replays):
                    runners[name]["graph"].replay()
                end.record(); end.synchronize()
                timings[name]=start.elapsed_time(end)/(8*replays)
            samples.append(dict(round=index,order=sequence,timings_ms=timings,
                speedup=timings["coalesced"]/timings["coalesced_wide"]))
    ratios=[s["speedup"] for s in samples]
    round_ratios=[statistics.median(s["speedup"] for s in samples if s["round"]==index)
                  for index in range(rounds)]
    boot=random.Random(826)
    bootstrap=sorted(statistics.median(boot.choices(round_ratios,k=len(round_ratios))) for _ in range(2000))
    timings={name:[s["timings_ms"][name] for s in samples] for name in runners}
    return dict(shape=list(shape),phase=phase,rounds=rounds,replays_per_sample=replays,
        graph_calls=8,precondition_seconds=precondition_seconds,samples=samples,
        median_ms={name:statistics.median(values) for name,values in timings.items()},
        individual_quality={name:runtime_timing_metadata(values) for name,values in timings.items()},
        paired_speedup_median=statistics.median(ratios),paired_speedup_bootstrap95=[bootstrap[50],bootstrap[1949]],
        paired_speedup_by_first={name:statistics.median(s["speedup"] for s in samples if s["order"][0]==name)
                                 for name in runners},
        all_rounds_bitexact=True,torch=torch.__version__,cuda=torch.version.cuda,
        device=str(torch.cuda.get_device_properties(0)),seed=825,
        caveat="Paired relative diagnostic, not individually eligible absolute latencies or public compiled backward. "
               "Bootstrap resamples complete paired rounds, not independent hardware sessions.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape",default="2048,768,1536")
    parser.add_argument("--phase",choices=("dx","dw"),default="dw")
    parser.add_argument("--rounds",type=int,default=12)
    parser.add_argument("--replays",type=int,default=40)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    result=compare(tuple(map(int,args.shape.split(","))),args.phase,args.rounds,args.replays)
    with Path(args.output).open("x") as handle:
        json.dump(result,handle,indent=2)
    print(json.dumps({key:result[key] for key in ("median_ms","paired_speedup_median","paired_speedup_bootstrap95","paired_speedup_by_first")}))


if __name__=="__main__":
    main()
