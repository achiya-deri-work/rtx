"""Matched aggressive torch.compile versus fused GLU, excluding cold start."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import time

import torch

from benchmarks.benchmark_torchao_fp8_rowwise import COMPILE_OPTIONS
from rtx import mxfp8_glu, mxfp8_glu_packed
from rtx.glu import DEFAULT_GLU_CONFIG
from rtx.configs.glu import glu_config_from_dict, prequant_glu_config_from_dict


def measure(fn, calls=20, samples=21):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            output = fn()
    graph.replay()
    torch.cuda.synchronize()
    timings = []
    for _ in range(samples):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) * 1000 / calls)
    return {"median_us": statistics.median(timings), "min_us": min(timings),
            "samples_us": timings}


def measure_uncaptured(fn,calls=50,samples=11):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    device_times,wall_times=[],[]
    for _ in range(samples):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        wall_start=time.perf_counter()
        start.record()
        for _ in range(calls):
            output=fn()
        end.record()
        end.synchronize()
        wall_times.append((time.perf_counter()-wall_start)*1e6/calls)
        device_times.append(start.elapsed_time(end)*1000/calls)
    return {"median_us":statistics.median(device_times),"wall_median_us":statistics.median(wall_times),
            "samples_us":device_times,"wall_samples_us":wall_times}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shape", default="512,1536,768", help="M,H,K")
    p.add_argument("--activation", default="silu", choices=("silu","gelu","relu"))
    p.add_argument("--output-format", default="bf16", choices=("bf16","mxfp8"))
    p.add_argument("--history", default="autotune_results/glu_smoke_bf16")
    p.add_argument("--config-file",help="explicit configuration or previous report; remeasured, never reuses timings")
    p.add_argument("--output", default="autotune_reports/glu_iteration/baseline.json")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--primary-mode",choices=("graph","uncaptured"),default="graph",
                   help="Primary performance target; both modes are always reported")
    p.add_argument("--target-speedup",type=float,default=1.3)
    p.add_argument("--packed-output",action="store_true",help="Use tensor-only MXFP8 output instead of MXTensor")
    args = p.parse_args()
    if args.packed_output and args.output_format!="mxfp8":
        p.error("--packed-output requires --output-format mxfp8")
    m,h,k = map(int,args.shape.split(","))
    torch.manual_seed(0)
    x = torch.randn(m,k,device="cuda",dtype=torch.bfloat16)
    w = torch.randn(2*h,k,device="cuda",dtype=torch.bfloat16)*0.02
    records = [json.loads(line) for file in Path(args.history).rglob("observations.jsonl")
               for line in file.read_text().splitlines()]
    records = [r for r in records if r["outcome"]["status"] == "ok"
               and r["features"].get("context.workload.m")==m
               and r["features"].get("context.workload.n")==2*h
               and r["features"].get("context.workload.k")==k
               and r["features"].get(f"context.workload.activation={args.activation}")==1
               and r["features"].get(f"context.workload.output_format={args.output_format}")==1]
    config = DEFAULT_GLU_CONFIG
    if records:
        value=min(records,key=lambda r:r["outcome"]["median_ms"])["config"]
        config=(prequant_glu_config_from_dict if value.get("backend")=="prequant" else glu_config_from_dict)(value)
    if args.config_file:
        value=json.loads(Path(args.config_file).read_text())
        value=value.get("config",value)
        config=(prequant_glu_config_from_dict if value.get("backend")=="prequant" else glu_config_from_dict)(value)
    def baseline(x,w):
        gate, up = torch.nn.functional.linear(x,w).chunk(2,-1)
        y = getattr(torch.nn.functional,args.activation)(gate) * up
        if args.output_format == "bf16":
            return y
        blocks = torch.nn.functional.pad(y.float(),(0,(-h)%32)).reshape(m,-1,32)
        bits = blocks.abs().amax(-1,keepdim=True).view(torch.int32)
        exponent = ((bits>>23)&255)-127-8
        exponent += (bits&0x7fffff)>0x600000
        code = (exponent.clamp(-127,128)+127).to(torch.uint8)
        scale = torch.exp2(exponent.clamp(-126,127).float())
        return (blocks/scale).clamp(-448,448).to(torch.float8_e4m3fn).flatten(1), code.squeeze(-1)
    compiled = torch.compile(baseline,fullgraph=True,dynamic=False,options=COMPILE_OPTIONS)
    def candidate(x,w):
        if args.packed_output:
            return mxfp8_glu_packed(x,w,activation=args.activation,config=config)
        return mxfp8_glu(x,w,activation=args.activation,config=config,output_format=args.output_format)
    fused = torch.compile(candidate,
                          fullgraph=True,dynamic=False,options=COMPILE_OPTIONS)
    with torch.inference_mode():
        if args.profile:
            for _ in range(5):
                fused(x,w)
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            with torch.cuda.nvtx.range("glu_candidate"):
                fused(x,w)
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            return
        started = time.monotonic()
        ref, actual = compiled(x,w), fused(x,w)
        from torch._inductor.utils import run_and_get_code
        _,code=run_and_get_code(fused,x,w)
        direct_lowering=any("torch._rtx_glu_launchers" in text for text in code)
        assert direct_lowering, "GLU used a stale opaque Inductor wrapper"
        torch.cuda.synchronize()
        cold_s = time.monotonic()-started
        changed = compiled(x,-w)
        if args.output_format=="mxfp8":
            assert not torch.equal(ref[0].float(), changed[0].float()), "baseline froze weight"
            ref = (ref[0].float()*torch.exp2(ref[1].float()-127).repeat_interleave(32,-1))[:,:h]
            if args.packed_output:
                actual = (actual[0].float()*actual[1].float().repeat_interleave(32,-1))[:,:h]
            else:
                actual = actual.dequantize()[:,:h]
        else:
            assert not torch.equal(ref, changed), "baseline froze weight"
        report = {"shape_m_h_k":[m,h,k],"activation":args.activation,"output_format":args.output_format,
                  "config_source":args.config_file or args.history,
                  "direct_lowering":direct_lowering,
                  "packed_output":args.packed_output,
                  "device":torch.cuda.get_device_name(),"torch":str(torch.__version__),
                  "compile_options":COMPILE_OPTIONS,"config":asdict(config),"cold_s":cold_s,
                  "relative_l2_vs_bf16":((actual.float()-ref.float()).norm()/ref.float().norm()).item()}
        report["baseline"] = measure(lambda:compiled(x,w))
        report["candidate"] = measure(lambda:fused(x,w))
        report["baseline_repeat"] = measure(lambda:compiled(x,w))
        report["baseline_uncaptured"] = measure_uncaptured(lambda:compiled(x,w))
        report["candidate_uncaptured"] = measure_uncaptured(lambda:fused(x,w))
        report["uncaptured_speedup"] = report["baseline_uncaptured"]["wall_median_us"]/report["candidate_uncaptured"]["wall_median_us"]
        report["speedup"] = report["baseline"]["median_us"]/report["candidate"]["median_us"]
        report["primary_mode"] = args.primary_mode
        report["primary_speedup"] = report["speedup"] if args.primary_mode=="graph" else report["uncaptured_speedup"]
        report["target_speedup"] = args.target_speedup
        report["target_met"] = report["primary_speedup"] >= args.target_speedup
    output = Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
