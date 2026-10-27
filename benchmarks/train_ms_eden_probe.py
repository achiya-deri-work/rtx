"""Small compiled teacher-regression probe, not transformer qualification."""
import argparse
import json
from pathlib import Path
import time
import torch
import rtx
from rtx.experimental_ms_eden import ms_eden_linear


def run_matched(steps=40,width=128):
    """Two-layer learnable probe; identical NVFP4 forward across all policies."""
    if steps <= 0 or width <= 0 or width % 128:
        raise ValueError("positive steps and width divisible by128 required")
    torch.manual_seed(914)
    inputs=torch.randn(256,width,device="cuda",dtype=torch.bfloat16)*.25
    heldout=torch.randn_like(inputs)
    first=torch.randn(width,width,device="cuda",dtype=torch.bfloat16)*.02
    second=torch.randn_like(first)*.02
    teacher1=torch.randn_like(first)*.02
    teacher2=torch.randn_like(first)*.02
    def teacher(a):
        return torch.tanh(a.float()@teacher1.float().T)@teacher2.float().T
    target,validation=teacher(inputs),teacher(heldout)
    expected=None
    results={}
    for dx,dw in (("mxfp8","mxfp8"),("ms_eden","ms_eden"),
                  ("mxfp8","ms_eden"),("ms_eden","mxfp8")):
        torch.compiler.reset()
        torch.manual_seed(915)
        w1,w2=torch.nn.Parameter(first.clone()),torch.nn.Parameter(second.clone())
        optimizer=torch.optim.AdamW([w1,w2],lr=.002,weight_decay=0.)
        def forward(a,b,c):
            hidden=torch.tanh(a@b.T)
            return ms_eden_linear(hidden,c,source_policy="original_bf16",
                forward_scaling="jit_row_region",forward_outer_mode="exact",
                mxfp8_backward_config=rtx.DEFAULT_MXFP8_BWD_CONFIG if "mxfp8" in (dx,dw) else None,
                dx_policy=dx,dw_policy=dw)
        compiled=torch.compile(forward,fullgraph=True,dynamic=False)
        initial=compiled(inputs,w1,w2).detach()
        if expected is None:
            expected=initial.clone()
        else:
            torch.testing.assert_close(initial,expected,atol=0,rtol=0)
        history=[]
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss=(compiled(inputs,w1,w2).float()-target).square().mean()
            loss.backward()
            if any(v.grad is None or not torch.isfinite(v.grad).all() for v in (w1,w2)):
                raise RuntimeError("missing/nonfinite upstream or final-layer gradient")
            if step%10==0 or step==steps-1:
                history.append(dict(step=step,loss=loss.item(),
                    upstream_grad_norm=w1.grad.float().norm().item(),
                    final_grad_norm=w2.grad.float().norm().item()))
            optimizer.step()
        with torch.no_grad():
            error=compiled(heldout,w1,w2).float()-validation
        results[f"dx_{dx}__dw_{dw}"]=dict(history=history,
            heldout_nrms=(error.norm()/validation.norm().clamp_min(1e-12)).item(),
            first_weight_update_norm=(w1.detach().float()-first.float()).norm().item(),
            second_weight_update_norm=(w2.detach().float()-second.float()).norm().item())
    return dict(results=results,steps=steps,width=width,seed=914,
        forward="NVFP4 jit_row_region exact FP32 outer scaling, bitwise identical initial output across policies",
        source_policy="original_bf16",torch=torch.__version__,cuda=torch.version.cuda,
        device=str(torch.cuda.get_device_properties(0)),
        caveat="Single-seed short compiled two-layer teacher regression; upstream BF16 layer exercises dX; not transformer qualification.")


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
    parser.add_argument("--matched-mixed",action="store_true",help="test all four original-source backward policies with identical forward")
    args=parser.parse_args()
    result=(run_matched(args.steps,args.width) if args.matched_mixed else
            run(args.steps,args.width,args.forward_outer_mode,args.forward_scaling))
    path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x") as handle:
        json.dump(result,handle,indent=2)
    print(json.dumps(result))


if __name__=="__main__":
    main()
