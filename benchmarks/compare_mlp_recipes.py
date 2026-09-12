"""Compare RTX with independently oriented TorchAO-quantized MLP recipes.

Uses TorchAO conversion and FP32 reconstructed-operand GEMMs, not TE kernels.
BF16-boundary recipes use real PyTorch activation/product autograd.
"""
import argparse
import json
from pathlib import Path

import torch
from torchao.prototype.mx_formats.mx_tensor import to_mx
from torchao.prototype.mx_formats.config import ScaleCalculationMode

from rtx import MXFP8MLPConfig,mxfp8_mlp
from benchmarks.stress_mxfp8_mlp_numerics import metrics,baseline
from benchmarks.analyze_mlp_divergence import derivative


def ao_quantize(t):
    k=t.shape[-1]
    padded=torch.nn.functional.pad(t.contiguous(),(0,(-k)%32))
    scales,data=to_mx(padded,torch.float8_e4m3fn,32,
                      scaling_mode=ScaleCalculationMode.RCEIL)
    scales=scales.reshape(padded.shape[0],-1).view(torch.float8_e8m0fnu)
    return (data.float()*scales.float().repeat_interleave(32,-1))[:,:k]


class AOReferenceLinear(torch.autograd.Function):
    """Fresh HP source per reduction orientation; BF16 linear boundaries."""
    @staticmethod
    def forward(ctx,x,w):
        ctx.save_for_backward(x,w)
        return (ao_quantize(x)@ao_quantize(w).T).bfloat16()

    @staticmethod
    def backward(ctx,dy):
        x,w=ctx.saved_tensors
        return ((ao_quantize(dy)@ao_quantize(w.T).T).bfloat16(),
                (ao_quantize(dy.T)@ao_quantize(x.T).T).bfloat16())


def separate_recipe(inputs,dy,activation,activation_fp32):
    x,w,d=(v.detach().requires_grad_() for v in inputs)
    p=AOReferenceLinear.apply(x,w)
    if activation_fp32:
        p=p.float()
    g,u=p.chunk(2,-1)
    h=(getattr(torch.nn.functional,activation)(g)*u).bfloat16()
    y=AOReferenceLinear.apply(h,d)
    return (y,*torch.autograd.grad(y,(x,w,d),dy))


def fused_reference(inputs,dy,activation,fresh_hidden):
    x,w,d=inputs
    g,u=(ao_quantize(x)@ao_quantize(w).T).chunk(2,-1)
    a=getattr(torch.nn.functional,activation)(g)
    h=a*u
    hq=ao_quantize(h)
    y=(hq@ao_quantize(d).T).bfloat16()
    dh=(ao_quantize(dy)@ao_quantize(d.T).T).bfloat16().float()
    dp=torch.cat((dh*u*derivative(g,activation),dh*a),-1).bfloat16()
    hidden=(h if fresh_hidden else hq).bfloat16()
    return (y,(ao_quantize(dp)@ao_quantize(w.T).T).bfloat16(),
            (ao_quantize(dp.T)@ao_quantize(x.T).T).bfloat16(),
            (ao_quantize(dy.T)@ao_quantize(hidden.T).T).bfloat16())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",required=True)
    args=parser.parse_args()
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    import importlib.metadata
    metadata={"torch":torch.__version__,"torchao":importlib.metadata.version("torchao"),
              "gpu":torch.cuda.get_device_name(),"scale_mode":"RCEIL","block":32,
              "reference":"TorchAO eager conversion + reconstructed FP32 GEMMs; no TE execution"}
    (out/"environment.json").write_text(json.dumps(metadata,indent=2))
    names=("y","dx","dgate_up","ddown")
    count=0;failures=0
    with (out/"cases.jsonl").open("w") as stream:
        for shape in ((17,35,97,35),(64,128,256,128),(512,768,1536,768),(8192,128,256,128)):
            m,k,h,o=shape
            for distribution in ("uniform_half","init_002","outliers"):
                for activation in ("silu","gelu","relu"):
                    for seed in (17,31):
                        torch.manual_seed(seed)
                        x,w,d,dy=[(torch.rand(*s,device="cuda")-.5).bfloat16()
                                  for s in ((m,k),(2*h,k),(o,h),(m,o))]
                        if distribution=="init_002":
                            torch.nn.init.trunc_normal_(w,std=.02,a=-.04,b=.04)
                            torch.nn.init.trunc_normal_(d,std=.02,a=-.04,b=.04)
                        elif distribution=="outliers":
                            x[:,::32]*=128;w[:,::32]*=64
                        inputs=tuple(t.requires_grad_() for t in (x,w,d))
                        bf=baseline(inputs,dy,activation,torch.bfloat16)
                        y=mxfp8_mlp(*inputs,activation=activation,config=MXFP8MLPConfig())
                        actual=(y,*torch.autograd.grad(y,inputs,dy))
                        with torch.no_grad():
                            current=fused_reference(inputs,dy,activation,False)
                            fresh=fused_reference(inputs,dy,activation,True)
                        recipes={"native":actual,"ao_fused_current":current,"ao_fused_fresh_hidden":fresh,
                                 "ao_bf16_linear_fp32_activation":separate_recipe(inputs,dy,activation,True),
                                 "ao_bf16_linear_bf16_activation":separate_recipe(inputs,dy,activation,False)}
                        record={"shape":shape,"distribution":distribution,"activation":activation,"seed":seed,
                            "versus_bf16":{},"versus_native":{},"residual_output":{},
                            "residual_input_gradient":{}}
                        for label,values in recipes.items():
                            record["versus_bf16"][label]={n:metrics(a,b) for n,a,b in zip(names,values,bf)}
                            record["versus_native"][label]={n:metrics(a,b) for n,a,b in zip(names,values,actual)}
                            record["residual_output"][label]=metrics((x.detach()+values[0]).bfloat16(),
                                                                    (x.detach()+bf[0]).bfloat16())
                            record["residual_input_gradient"][label]=metrics((dy+values[1]).bfloat16(),
                                                                            (dy+bf[1]).bfloat16())
                        record["kernel_gate_passed"]=all(v["finite"] and (v["relative_l2"]<.025 or v["max_abs"]<1e-12)
                            for v in record["versus_native"]["ao_fused_current"].values())
                        record["all_finite"]=all(v["finite"] for group in record["versus_bf16"].values() for v in group.values())
                        count+=1;failures+=not(record["kernel_gate_passed"] and record["all_finite"])
                        stream.write(json.dumps(record,allow_nan=False)+"\n");stream.flush()
                        print(count,shape,distribution,activation,seed,"pass" if record["kernel_gate_passed"] else "FAIL",flush=True)
    (out/"summary.json").write_text(json.dumps({"cases":count,"failures":failures},indent=2))
    raise SystemExit(bool(failures))


if __name__=="__main__":
    main()
