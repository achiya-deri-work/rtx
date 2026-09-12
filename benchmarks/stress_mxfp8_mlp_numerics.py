"""Numerical stress qualification, not a performance benchmark.

Run: python -m benchmarks.stress_mxfp8_mlp_numerics --output-dir PATH
Every case is flushed independently; failed inputs are saved for reproduction.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import torch

from rtx import MXFP8MLPConfig, mxfp8_mlp
from tests.test_mxfp8_mlp import reference


def metrics(candidate, expected):
    a,b=candidate.detach().double(),expected.detach().double()
    finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    if not finite:
        return {"finite":False,"candidate_nonfinite":int((~torch.isfinite(a)).sum()),
                "reference_nonfinite":int((~torch.isfinite(b)).sum())}
    difference=a-b
    rms=float(difference.square().mean().sqrt())
    ref_rms=float(b.square().mean().sqrt())
    absolute=float(difference.abs().max())
    cosine=float(torch.nn.functional.cosine_similarity(a.flatten(),b.flatten(),dim=0,eps=1e-30))
    return {"finite":True,"max_abs":absolute,"rms":rms,"reference_rms":ref_rms,
            "relative_l2":rms/max(ref_rms,1e-30),
            "normalized_max":absolute/max(float(b.abs().max()),1e-30),
            "cosine":cosine,"worst_flat_index":int(difference.abs().argmax())}


def baseline(inputs, dy, activation, dtype):
    x,w,d=(v.detach().to(dtype).requires_grad_() for v in inputs)
    g,u=(x@w.T).chunk(2,-1)
    y=(getattr(torch.nn.functional,activation)(g)*u)@d.T
    return (y,*torch.autograd.grad(y,(x,w,d),dy.to(dtype)))


def inputs_for(shape, distribution, seed):
    torch.manual_seed(seed)
    m,k,h,o=shape
    x=torch.randn(m,k,device="cuda")
    w=torch.randn(2*h,k,device="cuda")*.02
    d=torch.randn(o,h,device="cuda")*.02
    dy=torch.randn(m,o,device="cuda")
    if distribution=="tiny":
        x*=1e-6; w*=1e-3; d*=1e-3
    elif distribution=="large":
        x*=32; w*=8
    elif distribution=="outliers":
        x[:,::32]*=128; w[:,::32]*=64; dy[::32]*=64
    elif distribution=="zero":
        x.zero_(); w.zero_(); d.zero_()
    elif distribution=="zero_dy":
        dy.zero_()
    elif distribution=="cancellation":
        x[1::2]=-x[:2*(m//2):2]; dy[1::2]=dy[:2*(m//2):2]
    elif distribution=="scale_boundaries":
        # Repeated positive/negative representable values around scale transitions.
        values=torch.tensor([447,448,449,-447,-448,-449,0,1],device="cuda")
        x=values.repeat((x.numel()+7)//8)[:x.numel()].reshape_as(x)/128
        w*=torch.exp2((torch.arange(k,device="cuda")//32%9-4).float())
    elif distribution=="negative_gate":
        x=x.abs(); w[:h]=-w[:h].abs()*16
    elif distribution=="tiny_x":
        x*=1e-35
    elif distribution=="tiny_w":
        w*=1e-33
    elif distribution=="subnormal_x":
        x.fill_(torch.finfo(torch.bfloat16).tiny/64)
    elif distribution=="large_dy":
        dy*=1e20
    elif distribution=="sparse":
        x.zero_();x[:,0]=1;w[:,1:]=0
    elif distribution=="signed_zero":
        x.fill_(-0.0)
    return tuple(t.bfloat16().contiguous().requires_grad_() for t in (x,w,d)),dy.bfloat16()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--seed",type=int,default=9211)
    parser.add_argument("--start",type=int,default=0)
    parser.add_argument("--limit",type=int,default=100000)
    parser.add_argument("--policy",choices=("default","packed","winner"))
    args=parser.parse_args()
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    env={"torch":torch.__version__,"cuda":torch.version.cuda,
         "device":torch.cuda.get_device_name(),"seed":args.seed,
         "gate":"finite and (relative_l2 <= .025 or max_abs <= 1e-12)",
         "oracle":"independent PyTorch quantized operational reference; BF16/FP32 separately descriptive"}
    (out/"environment.json").write_text(json.dumps(env,indent=2))
    base=MXFP8MLPConfig()
    packed=replace(base,down_backend="producer_quantized",recompute_backend="prequant",
                   backward_overlap=True,gate_up=replace(base.gate_up,glu_silu_math="approx_reciprocal"),
                   recompute=replace(base.recompute,gemm=replace(base.recompute.gemm,
                       glu_hidden_decode="bf16x2",glu_grad_load="prefetch",glu_grad_store="exchange")))
    winner=MXFP8MLPConfig.from_dict(json.loads(Path(
        "benchmarks/configs/mxfp8_mlp_5070ti_m512_k768_h1536_o768.json").read_text())["config"])
    distributions=("normal","tiny","large","outliers","zero","zero_dy",
                   "cancellation","scale_boundaries","negative_gate")
    cases=[]
    for shape in ((1,1,1,1),(17,35,97,41),(31,63,65,33),(33,129,127,65),
                  (64,128,256,64),(129,257,255,129)):
        for activation in ("silu","gelu","relu"):
            for distribution in distributions:
                cases.append((shape,activation,distribution,"packed" if len(cases)%2 else "default",False))
    for m in (512,2048,8192,32768):
        for distribution in ("normal","cancellation","outliers"):
            cases.append(((m,128,256,128),"silu",distribution,"packed",False))
    for distribution in distributions:
        cases.append(((512,768,1536,768),"silu",distribution,"winner",True))
    for shape in ((17,8192,256,129),(33,128,8193,97),(512,1536,3072,1536)):
        for activation in ("silu","gelu","relu"):
            for distribution in ("normal","outliers"):
                cases.append((shape,activation,distribution,"packed",False))
    for activation in ("silu","gelu","relu"):
        for distribution in ("tiny_x","tiny_w","subnormal_x","large_dy","sparse","signed_zero"):
            for policy in ("default","packed"):
                cases.append(((17,35,97,41),activation,distribution,policy,False))
    failures=0;started=time.monotonic()
    attempted=0
    selected=max(0,min(len(cases),args.start+args.limit)-args.start)
    (out/"policies.json").write_text(json.dumps(
        {name:config.to_dict() for name,config in (("default",base),("packed",packed),("winner",winner))},indent=2))
    compiled={}
    with (out/"cases.jsonl").open("w") as stream:
        for index,(shape,activation,distribution,policy,compile_it) in enumerate(cases):
            if index<args.start or index>=args.start+args.limit:
                continue
            attempted+=1
            policy=args.policy or policy
            record={"index":index,"shape":shape,"activation":activation,
                    "distribution":distribution,"policy":policy,"compiled":compile_it,
                    "seed":args.seed+index}
            inputs,dy=inputs_for(shape,distribution,args.seed+index)
            cpu_inputs=[t.detach().cpu() for t in inputs]
            cpu_dy=dy.cpu()
            config={"default":base,"packed":packed,"winner":winner}[policy]
            try:
                yr,gr=reference(*inputs,dy,activation)
                if compile_it:
                    key=(activation,policy)
                    if key not in compiled:
                        compiled[key]=torch.compile(
                            lambda x,w,d:mxfp8_mlp(x,w,d,activation=activation,config=config),
                            fullgraph=True,dynamic=False)
                    y=compiled[key](*inputs)
                else:
                    y=mxfp8_mlp(*inputs,activation=activation,config=config)
                actual=(y,*torch.autograd.grad(y,inputs,dy))
                record["implementation"]={name:metrics(a,b) for name,a,b in
                    zip(("y","dx","dgate_up","ddown"),actual,(yr,*gr))}
                for dtype,label in ((torch.bfloat16,"versus_bf16"),(torch.float32,"versus_fp32")):
                    expected=baseline(inputs,dy,activation,dtype)
                    record[label]={name:metrics(a,b) for name,a,b in
                        zip(("y","dx","dgate_up","ddown"),actual,expected)}
                record["passed"]=all(v["finite"] and (v["relative_l2"]<=.025 or v["max_abs"]<=1e-12)
                                     for v in record["implementation"].values())
            except Exception as error:
                record.update(passed=False,error=f"{type(error).__name__}: {error}")
            if not record["passed"]:
                failures+=1
                torch.save({"inputs":cpu_inputs,"dy":cpu_dy,
                            "case":record,"config":config.to_dict()},out/f"failure_{index:04d}.pt")
            stream.write(json.dumps(record,allow_nan=False)+"\n");stream.flush()
            print(f"{index+1}/{len(cases)} {'PASS' if record['passed'] else 'FAIL'} "
                  f"{shape} {activation} {distribution} {policy} elapsed={time.monotonic()-started:.1f}s",flush=True)
            if "error" in record and any(token in record["error"].lower() for token in
                                          ("illegal instruction","illegal memory","device-side assert")):
                break
    summary={"cases":attempted,"selected":selected,"failures":failures,
             "complete":attempted==selected,"elapsed_s":time.monotonic()-started}
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)
    raise SystemExit(bool(failures))


if __name__=="__main__":
    main()
