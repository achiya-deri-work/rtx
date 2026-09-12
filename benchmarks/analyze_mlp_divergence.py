"""Stage-wise numerical ablations; hypothetical paths are NOT performance claims."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch

from rtx import MXFP8MLPConfig,mxfp8_mlp
from benchmarks.stress_mxfp8_mlp_numerics import metrics,baseline


def quantize(t,block=32,scale_mode="e8m0"):
    t=t.float(); k=t.shape[-1]
    b=torch.nn.functional.pad(t,(0,(-k)%block)).reshape(t.shape[0],-1,block)
    scale=(b.abs().amax(-1,keepdim=True)/448).clamp_min(2.**-126)
    if scale_mode=="e8m0":
        scale=torch.exp2(torch.ceil(torch.log2(scale)))
    return ((b/scale).clamp(-448,448).to(torch.float8_e4m3fn).float()*scale).flatten(1)[:,:k]


def derivative(g,activation):
    if activation=="silu":
        s=torch.sigmoid(g); return s*(1+g*(1-s))
    if activation=="gelu":
        return .5*(1+torch.erf(g/2**.5))+g*torch.exp(-g*g/2)/(2*torch.pi)**.5
    return (g>0).float()


def pipeline(x,w,d,dy,activation,disabled=(),correction=(),block=32,scale_mode="e8m0"):
    q=lambda t:quantize(t,block,scale_mode)
    def operand(t,name):
        return t.float() if name in disabled else q(t)
    def mm(a,b,name):
        if name in disabled:
            return a.float()@b.float().T
        aq,bq=q(a),q(b)
        y=aq@bq.T
        if name in correction:
            # Three GEMMs; no residual-residual product. A proposed algorithm,
            # simulated with FP32 GEMMs over reconstructed FP8 operands.
            y=y+q(a.float()-aq)@bq.T+aq@q(b.float()-bq).T
        return y
    xq,wq=operand(x,"x"),operand(w,"wgu")
    p=xq@wq.T
    if "gu" in correction:
        p=p+q(x.float()-xq)@wq.T+xq@q(w.float()-wq).T
    if "gate_only" in disabled:
        p=torch.cat((x.float()@w[:w.shape[0]//2].float().T,p[:,p.shape[1]//2:]),-1)
    g,u=p.chunk(2,-1)
    a=getattr(torch.nn.functional,activation)(g)
    h=a*u
    hq=operand(h,"hidden")
    if "hidden" in correction:
        hq=hq+q(h-hq)
    dq=operand(d,"wdown")
    y=hq@dq.T
    if "down" in correction:
        y=y+q(h-hq)@dq.T+hq@q(d.float()-dq).T
    y=y.bfloat16()
    dh=mm(dy,d.T,"dh").bfloat16().float()
    dp=torch.cat((dh*u*derivative(g,activation),dh*a),-1).bfloat16()
    dx=mm(dp,w.T,"dx").bfloat16()
    dg=mm(dp.T,x.T,"dwgu").bfloat16()
    dd=mm(dy.T,hq.bfloat16().T,"dwdown").bfloat16()
    return {"xq":xq,"wq":wq,"projection":p,"activation":a,"hidden":h,
            "hq":hq,"downq":dq,"dh":dh,"dp":dp,
            "y":y,"dx":dx,"dgate_up":dg,"ddown":dd}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--reduction-ladder",action="store_true")
    args=parser.parse_args()
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    if args.reduction_ladder:
        torch.manual_seed(882)
        with (out/"reduction.jsonl").open("w") as stream:
            for m in (128,512,2048,8192,32768):
                x=(torch.rand(m,128,device="cuda")-.5).bfloat16()
                dy=(torch.rand(m,128,device="cuda")-.5).bfloat16()
                a,b=quantize(dy.T),quantize(x.T)
                low=a@b.T
                record={"m":m,"torch":torch.__version__,"gpu":torch.cuda.get_device_name(),
                    "fp32_accumulator_vs_fp64":metrics(low,a.double()@b.double().T),
                    "fp8_vs_bf16":metrics(low.bfloat16(),dy.T@x),
                    "bf16_vs_fp64":metrics(dy.T@x,dy.double().T@x.double())}
                stream.write(json.dumps(record)+"\n");stream.flush()
                print(json.dumps(record),flush=True)
        return
    names=("y","dx","dgate_up","ddown")
    all_sites=("x","wgu","hidden","wdown","dh","dx","dwgu","dwdown")
    ablations={"full":{},"exact_x":{"disabled":("x",)},
        "exact_wgu":{"disabled":("wgu",)},"exact_hidden":{"disabled":("hidden",)},
        "exact_wdown":{"disabled":("wdown",)},
        "exact_forward":{"disabled":all_sites[:4]},
        "exact_backward_gemms":{"disabled":all_sites[4:]},
        "exact_dh":{"disabled":("dh",)},"exact_dx":{"disabled":("dx",)},
        "exact_dwgu":{"disabled":("dwgu",)},"exact_dwdown":{"disabled":("dwdown",)},
        "corrected_backward":{"correction":("dh","dx","dwgu","dwdown")},
        "corrected_gu":{"correction":("gu",)},
        "exact_gu":{"disabled":("x","wgu")},
        "exact_gate_only":{"disabled":("gate_only",)},
        "corrected_projections":{"correction":("gu","down","dh","dx","dwgu","dwdown")},
        "corrected_projections_hidden":{"correction":("gu","down","hidden","dh","dx","dwgu","dwdown")},
        "block16_hypothesis":{"block":16},"fp32_scales_hypothesis":{"scale_mode":"fp32"}}
    (out/"manifest.json").write_text(json.dumps({"torch":torch.__version__,
        "cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(),"ablations":ablations,
        "note":"Analytical ablations, not implemented GPU kernels or timing claims"},indent=2))
    with (out/"cases.jsonl").open("w") as stream:
        for shape in ((64,128,256,128),(512,768,1536,768)):
            m,k,h,o=shape
            for distribution in ("uniform_half","uniform_eighth","init_002"):
                for activation in ("silu","gelu","relu"):
                    for seed in (17,31,59):
                        torch.manual_seed(seed)
                        tensors=[torch.rand(*s,device="cuda")-.5 for s in ((m,k),(2*h,k),(o,h),(m,o))]
                        if distribution=="uniform_eighth":
                            tensors=[t/4 for t in tensors]
                        if distribution=="init_002":
                            for i in (1,2):
                                torch.nn.init.trunc_normal_(tensors[i],std=.02,a=-.04,b=.04)
                        x,w,d,dy=[t.bfloat16() for t in tensors]
                        inputs=[t.requires_grad_() for t in (x,w,d)]
                        bf=baseline(inputs,dy,activation,torch.bfloat16)
                        fp=baseline(inputs,dy,activation,torch.float32)
                        with torch.no_grad():
                            exact=pipeline(x,w,d,dy,activation,disabled=all_sites)
                            full=pipeline(x,w,d,dy,activation)
                            record={"shape":shape,"distribution":distribution,"activation":activation,"seed":seed,
                                "stage_vs_unquantized":{key:metrics(full[key],exact[key]) for key in full},
                                "ablations":{},"bf16_vs_fp32":{n:metrics(a,b) for n,a,b in zip(names,bf,fp)}}
                            g,u=full["projection"].chunk(2,-1)
                            ge,ue=exact["projection"].chunk(2,-1)
                            record["gates"]={"sign_flip_fraction":float(((g>0)!=(ge>0)).float().mean()),
                                "reference_gate_rms":float(ge.square().mean().sqrt()),
                                "reference_near_zero_fraction":float((ge.abs()<.01).float().mean()),
                                "derivative":metrics(derivative(g,activation),derivative(ge,activation))}
                            record["local_backward_quantization"]={
                                "dh":metrics(full["dh"],(dy.float()@d.float()).bfloat16()),
                                "dx":metrics(full["dx"],(full["dp"].float()@w.float()).bfloat16()),
                                "dgate_up":metrics(full["dgate_up"],(full["dp"].T.float()@x.float()).bfloat16()),
                                "ddown":metrics(full["ddown"],(dy.T.float()@full["hq"].bfloat16().float()).bfloat16())}
                            for label,options in ablations.items():
                                result=full if label=="full" else pipeline(x,w,d,dy,activation,**options)
                                record["ablations"][label]={n:metrics(result[n],b) for n,b in zip(names,bf)}
                        # Validate the simulation against actual default CUDA implementation.
                        config=MXFP8MLPConfig()
                        y=mxfp8_mlp(*inputs,activation=activation,config=config)
                        actual=(y,*torch.autograd.grad(y,inputs,dy))
                        record["kernel_vs_simulation"]={n:metrics(a,full[n]) for n,a in zip(names,actual)}
                        # Isolate approximate reciprocal without changing any other schedule.
                        cfg=replace(config,gate_up=replace(config.gate_up,glu_silu_math="approx_reciprocal"))
                        ya=mxfp8_mlp(*inputs,activation=activation,config=cfg)
                        approx=(ya,*torch.autograd.grad(ya,inputs,dy))
                        record["approx_vs_division"]={n:metrics(a,b) for n,a,b in zip(names,approx,actual)}
                        stream.write(json.dumps(record,allow_nan=False)+"\n");stream.flush()
                        print(shape,distribution,activation,seed,"done",flush=True)


if __name__=="__main__":
    main()
