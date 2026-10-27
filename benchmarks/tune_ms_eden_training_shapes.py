"""Bit-exact preparation schedule study on the 21M decoder's projections.

CUDA graph timing includes both operand preparations, finalization and GEMM.
Explicit signs/uniforms are identical between schedules; this isolates geometry,
not RNG quality. Public compiled training remains the final acceptance test.
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import statistics

import torch


def study(shape, phase, rounds=6, replays=40, portfolio="loads"):
    from benchmarks.study_ms_eden_backward import prepare
    from rtx.autotune.posthoc import preparation_resource_seeds
    from rtx.configs.nvfp4 import NVFP4Problem
    from rtx.fp4 import compile_nvfp4_gemm
    from rtx.kernels.nvfp4_posthoc import compile_posthoc_scale_product
    m,n,k=shape
    torch.manual_seed(826)
    dy=torch.randn(m,n,device="cuda",dtype=torch.bfloat16)*.25
    operand=torch.randn(n if phase=="dx" else m,k,device="cuda",dtype=torch.bfloat16)*.02
    signs=(torch.randint(2,(128,),device="cuda")*2-1).float()
    sm=torch.cuda.get_device_properties(0).multi_processor_count
    base=next(c for c in preparation_resource_seeds(shape,phase,sm)
              if c.producer_load=="coalesced_wide")
    problem=NVFP4Problem(m if phase=="dx" else n,k,n if phase=="dx" else m)
    runners={}
    candidates={mode:replace(base,producer_load=mode)
                for mode in ("coalesced_wide","coalesced_x4","coalesced_x8")}
    if portfolio=="confirmation":
        candidates={"coalesced_wide":base,
                    "coalesced_wide_256":replace(base,producer_blocks=256)}
    if portfolio=="signs":
        candidates={"reload":replace(base,producer_blocks=256),
                    "register":replace(base,producer_blocks=256,producer_signs="register")}
    if portfolio=="resources":
        candidates={"coalesced_wide":base,
                    "coalesced_wide_256":replace(base,producer_blocks=256)}
        for threads,blocks in ((64,512),(256,64),(256,128),(256,256),(512,64),(512,128)):
            candidates[f"wide_t{threads}_b{blocks}"]=replace(base,producer_threads=threads,
                                                            producer_blocks=blocks)
    if portfolio=="grids":
        for mode in ("coalesced_wide","coalesced_x8"):
            for blocks in (128,256,512):
                candidates[f"{mode}_{blocks}"]=replace(base,producer_load=mode,producer_blocks=blocks)
        candidates["coalesced_x8_256_staged"]=replace(base,producer_load="coalesced_x8",
                                                     producer_blocks=256,amax_reduction="staged")
    for mode,schedule in candidates.items():
        a=prepare(dy,phase=="dw","mma128",schedule)
        b=prepare(operand,True,"mma128",schedule)
        alpha=torch.empty(1,device="cuda")
        out=torch.empty(problem.m,problem.n,device="cuda",dtype=torch.bfloat16)
        product=compile_posthoc_scale_product()
        gemm=compile_nvfp4_gemm(problem,schedule.gemm)
        def run(a=a,b=b,alpha=alpha,out=out,product=product,gemm=gemm):
            a["run"](signs); b["run"](signs)
            product(a["factor"],b["factor"],alpha)
            gemm(a["packed"],b["packed"],a["scales"],b["scales"],out,alpha)
        runners[mode]=dict(run=run,a=a,b=b,out=out,schedule=asdict(schedule))
    baseline=next(iter(runners.values()))
    for r in runners.values():
        for key in ("a","b"):
            r[key]["uniforms"].copy_(baseline[key]["uniforms"])
    def check():
        for r in runners.values():
            r["run"]()
        for r in runners.values():
            for key in ("a","b"):
                for field in ("packed","scales","factor"):
                    torch.testing.assert_close(r[key][field].view(torch.uint8),
                        baseline[key][field].view(torch.uint8),atol=0,rtol=0)
            torch.testing.assert_close(r["out"],baseline["out"],atol=0,rtol=0)
            assert r["out"].isfinite().all()
    check()
    dy.zero_()
    check()
    dy.normal_(0,.25)
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    for r in runners.values():
        with torch.cuda.stream(stream):
            for _ in range(3): r["run"]()
        stream.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):
            for _ in range(4): r["run"]()
        r["graph"]=graph
    torch.cuda.current_stream().wait_stream(stream)
    samples=[]
    rng=random.Random(826)
    for index in range(rounds):
        dy.normal_(0,.25); operand.normal_(0,.02)
        signs.random_(0,2).mul_(2).sub_(1)
        check()
        for r in runners.values():
            for _ in range(8): r["graph"].replay()
        torch.cuda.synchronize()
        order=list(runners); rng.shuffle(order)
        for sequence in (order,list(reversed(order))):
            timings={}
            for name in sequence:
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(replays): runners[name]["graph"].replay()
                end.record(); end.synchronize()
                timings[name]=start.elapsed_time(end)*1000/(4*replays)
            samples.append(dict(round=index,order=sequence,us=timings))
    medians={name:statistics.median(s["us"][name] for s in samples) for name in runners}
    return dict(shape=shape,phase=phase,gpu=torch.cuda.get_device_name(),torch=torch.__version__,
                bit_exact=True,rounds=rounds,replays=replays,median_us=medians,samples=samples,
                schedules={name:r["schedule"] for name,r in runners.items()})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shape",default="4096,2048,512")
    p.add_argument("--phase",choices=("dx","dw"),required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--portfolio",choices=("loads","grids","resources","confirmation","signs"),default="loads")
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    result=study(tuple(map(int,args.shape.split(','))),args.phase,portfolio=args.portfolio)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ("shape","phase","bit_exact","median_us")}))


if __name__=="__main__": main()
