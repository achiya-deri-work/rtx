"""Persistent fresh-process qualification of compiled MLP first-hit effort.

Invoke separately for budgets 8,16,32,64 with the same cache root, then repeat
64 with --reuse-only and a new output. Reports refuse overwrite. Raw tuner
journals remain under the cache root; no observations are relabeled or erased.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback
from unittest.mock import patch

import torch
import rtx


def inventory(root):
    files={}
    statuses=Counter()
    count=0
    for path in sorted(root.rglob("observations.jsonl")):
        data=path.read_bytes()
        files[str(path.relative_to(root))]=dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
        if data and not data.endswith(b"\n"):
            raise ValueError(f"incomplete journal tail: {path}")
        for line in data.splitlines():
            row=json.loads(line)
            statuses[row["outcome"]["status"]]+=1
            count+=1
    return dict(files=files,count=count,statuses=dict(statuses))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--budget",type=int,choices=(8,16,32,64),required=True)
    p.add_argument("--shape",default="256,256,512,256",help="M,K,H,O")
    p.add_argument("--phase",choices=("train","forward"),default="train")
    p.add_argument("--seconds",type=float,default=180)
    p.add_argument("--reuse-only",action="store_true")
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).resolve().parents[1]
    sources={str(path.relative_to(source)):hashlib.sha256(path.read_bytes()).hexdigest()
             for path in [Path(__file__).resolve(),*sorted((source/"rtx").rglob("*.py"))]}
    report=dict(command=sys.argv,pid=os.getpid(),budget=args.budget,phase=args.phase,
        shape=list(map(int,args.shape.split(","))),reuse_only=args.reuse_only,sources=sources,
        seed=946,torch=str(torch.__version__),cuda=torch.version.cuda,passed=False)
    try:
        qualify(args,report)
        report["passed"]=True
    except Exception:
        report["error"]=traceback.format_exc()
        raise
    finally:
        with args.output.open("x") as f:json.dump(report,f,indent=2)
        print("RESULT",args.output,"passed=",report["passed"],flush=True)


def qualify(args,report):
    from torch._inductor.compile_fx import compile_fx
    from rtx.autotune.mlp import reference,tune_mxfp8_mlp
    from rtx._mlp_autotune import SELECTIONS
    report["gpu"]=torch.cuda.get_device_name()
    torch.manual_seed(report["seed"])
    m,k,h,o=report["shape"]
    policy=rtx.AutotunePolicy(max_trials=args.budget,max_seconds=args.seconds,
        search_schedule="progressive",cache_behavior="refine",finalist_rounds=2,
        confirmation_repeats=0,warmup=1,samples=5,calls_per_sample=3)
    x=(torch.randn(m,k,device="cuda",dtype=torch.bfloat16)*.1).requires_grad_(args.phase=="train")
    layer=rtx.MXFP8MLP(k,h,o,device="cuda",autotune="balanced",tuning_policy=policy,
                       autotune_cache_dir=args.cache_root)
    dy=torch.randn(m,o,device="cuda",dtype=torch.bfloat16)*.1
    inputs=x,layer.gate_up_weight,layer.down_weight
    graphs=[]
    def backend(gm,values):
        graphs.append(gm)
        return compile_fx(gm,values)
    compiled=torch.compile(layer,fullgraph=True,dynamic=False,backend=backend)
    calls=[]
    def tune(*values,**kwargs):
        if args.reuse_only:raise AssertionError("fresh-process durable reuse attempted tuning")
        calls.append(time.perf_counter())
        kwargs["progress"]=lambda line:print("TUNE",line,flush=True)
        return tune_mxfp8_mlp(*values,**kwargs)
    before=inventory(args.cache_root)
    report["before"]=before
    with torch.set_grad_enabled(args.phase=="train"),patch("rtx.autotune.mlp.tune_mxfp8_mlp",side_effect=tune):
        torch.cuda.synchronize()
        start=time.perf_counter()
        y=compiled(x)
        torch.cuda.synchronize()
        report["first_forward_seconds"]=time.perf_counter()-start
        if args.phase=="train":
            start=time.perf_counter()
            grads=torch.autograd.grad(y,inputs,dy)
            torch.cuda.synchronize()
            report["first_backward_seconds"]=time.perf_counter()-start
        else:grads=()
        actual=(y,*grads)
        with torch.no_grad():expected=reference(*inputs,dy,"silu")
        numerics={}
        for name,a,b in zip(("y","dx","dgu","ddown"),actual,expected):
            delta=a.detach().float()-b.float()
            rms=float(delta.square().mean().sqrt())
            ref=float(b.float().square().mean().sqrt())
            numerics[name]=dict(finite=bool(a.isfinite().all()),max_abs=float(delta.abs().max()),
                rms=rms,reference_rms=ref,normalized_rms=rms/max(ref,1e-12))
            assert numerics[name]["finite"] and numerics[name]["normalized_rms"]<.025,numerics
        report["numerics"]=numerics
        selected=list(SELECTIONS.values())
        assert len(selected)==1
        report["selected_config"]=json.loads(selected[0])
        after=inventory(args.cache_root)
        report["after_search"]=after
        # Dedicated roots keep the coverage assertion unambiguous.
        assert after["count"]==args.budget,(after["count"],args.budget)
        if args.reuse_only:assert after==before
        def run():
            output=compiled(x)
            gradients=torch.autograd.grad(output,inputs,dy) if args.phase=="train" else ()
            return output,*gradients
        for _ in range(3):run()
        wall=[]
        for _ in range(10):
            torch.cuda.synchronize()
            start=time.perf_counter()
            result=run()
            torch.cuda.synchronize()
            wall.append(time.perf_counter()-start)
            for a,b in zip(actual,result):torch.testing.assert_close(a,b,rtol=0,atol=0)
        report["warm_host_inclusive_seconds"]=wall
        report["warm_host_inclusive_median_seconds"]=statistics.median(wall)
        assert len(graphs)==1,len(graphs)
        assert inventory(args.cache_root)==after
        report["dynamo_compilations"]=len(graphs)
        report["tuner_invocations"]=len(calls)
        if args.reuse_only:assert not calls
    print("QUALIFIED",json.dumps({key:report[key] for key in (
        "budget","phase","first_forward_seconds","dynamo_compilations","tuner_invocations")}),flush=True)


if __name__=="__main__":main()
