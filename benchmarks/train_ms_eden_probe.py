"""Small compiled teacher-regression probe, not transformer qualification."""
import argparse
import json
from pathlib import Path
import time
import torch
import rtx
from rtx.experimental_ms_eden import ms_eden_linear


def run(steps=200,width=128,forward_outer_mode="power2",forward_scaling="current"):
    if steps <= 0 or width <= 0 or width % 128:
        raise ValueError("positive steps and width divisible by 128 required")
    if forward_scaling not in ("current","jit_row_region","block"):
        raise ValueError("unsupported experimental forward scaling")
    torch.manual_seed(914)
    x=torch.randn(512,width,device="cuda",dtype=torch.bfloat16)*.25
    heldout=torch.randn_like(x)*1.
    teacher=torch.randn(width,width,device="cuda",dtype=torch.bfloat16)*.02
    initial=torch.randn_like(teacher)*.02
    target=(x.float()@teacher.float().T).bfloat16()
    validation=(heldout.float()@teacher.float().T).bfloat16()
    results={}
    forward_policy=rtx.NVFP4ScaleConfig(tensor_scale_mode=forward_outer_mode)
    modes=("bf16","nvfp4_mxfp8","nvfp4_ms_eden")
    if forward_scaling == "current":
        modes += ("nvfp4_ms_eden_packed",)
    for mode in modes:
        torch.compiler.reset()
        torch.manual_seed(915)
        weight=torch.nn.Parameter(initial.clone())
        optimizer=torch.optim.AdamW([weight],lr=.002,weight_decay=0.)
        def forward(a,b):
            if mode == "bf16":
                return a@b.T
            if mode == "nvfp4_mxfp8":
                return rtx.nvfp4_linear(a,b,scaling=forward_scaling,autotune="off",
                    scale_config=forward_policy,
                    backward_config=rtx.DEFAULT_MXFP8_BWD_CONFIG)
            return ms_eden_linear(a,b,forward_outer_mode=forward_outer_mode,forward_scaling=forward_scaling,
                source_policy="forward_current_roundtrip_bf16" if mode.endswith("packed") else "original_bf16")
        def loss_fn(a,b,t):
            return (forward(a,b).float()-t.float()).square().mean()
        loss=torch.compile(loss_fn,fullgraph=True,dynamic=False)
        history=[]
        torch.cuda.synchronize()
        started=time.perf_counter()
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            value=loss(x,weight,target)
            value.backward()
            if not torch.isfinite(weight.grad).all():
                raise RuntimeError(f"nonfinite {mode} gradient")
            optimizer.step()
            if step%20==0 or step==steps-1:
                history.append(dict(step=step,loss=value.item()))
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-started
        with torch.no_grad():
            error=forward(heldout,weight).float()-validation.float()
            nrms=(error.norm()/validation.float().norm()).item()
        results[mode]=dict(history=history,validation_nrms=nrms,
            wall_seconds_including_compile_and_checks=elapsed,
            optimizer_state_dtypes={key:str(value.dtype) for key,value in optimizer.state[weight].items()
                                    if isinstance(value,torch.Tensor)},
            weight_nrms=((weight.detach().float()-teacher.float()).norm()/teacher.float().norm()).item())
    return dict(results=results,steps=steps,width=width,seed=914,
        train_input_std=.25,validation_input_std=1.,forward_outer_mode=forward_outer_mode,
        forward_scaling=forward_scaling,
        torch=torch.__version__,cuda=torch.version.cuda,device=str(torch.cuda.get_device_properties(0)),
        caveat="One fixed teacher dataset and seed; includes compile, optimizer, and host checks; not a throughput benchmark or transformer convergence test.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps",type=int,default=200)
    parser.add_argument("--width",type=int,default=128)
    parser.add_argument("--forward-outer-mode",choices=("power2","exact"),default="power2")
    parser.add_argument("--forward-scaling",choices=("current","jit_row_region","block"),default="current")
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    result=run(args.steps,args.width,args.forward_outer_mode,args.forward_scaling)
    path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x") as handle:
        json.dump(result,handle,indent=2)
    print(json.dumps(result))


if __name__=="__main__":
    main()
