"""Isolated, optional Quartet-II comparison; never imported by RTX runtime.

Requires the separately installed upstream quartet2 package. One mode/shape per
process. Native FP4 backward and RTX MXFP8 backward are different contracts;
report individual samples and sample means, never equate unbiasedness with low
single-step error. Original source is pinned in the experiment report.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import time

import torch
from torch import nn

import rtx
from benchmarks.study_nvfp4_precision import metrics


class QuartetForwardMXBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, had, mode, scratch):
        from quartet2.linear import Quartet_II_fn
        ctx.save_for_backward(x, weight)
        return Quartet_II_fn.apply(x, weight, had, mode, True, None, None, scratch)

    @staticmethod
    def backward(ctx, dy):
        x, weight = ctx.saved_tensors
        dx, dw = rtx.mxfp8_linear_backward(
            dy.contiguous(), x, weight, config=rtx.DEFAULT_MXFP8_BWD_CONFIG,
            autotune="off")
        return dx, dw, None, None, None


def build(mode, k, n):
    if mode == "bf16":
        return nn.Linear(k, n, bias=False, dtype=torch.bfloat16, device="cuda")
    if mode == "mxfp8":
        return rtx.MXFP8Linear(k, n, device="cuda", autotune="off",
                               backward_config=rtx.DEFAULT_MXFP8_BWD_CONFIG)
    if mode == "rtx_nvfp4":
        return rtx.NVFP4Linear(k, n, device="cuda", autotune="off",
                               backward_config=rtx.DEFAULT_MXFP8_BWD_CONFIG)
    if mode in ("rtx_jit_four_six", "rtx_jit_amax_exact"):
        from rtx.nvfp4_inference_autotune import preferred_jit_row_region_config
        cfg=preferred_jit_row_region_config(rtx.NVFP4Problem(2048,n,k))
        outer="exact_four_over_six" if mode=="rtx_jit_four_six" else "exact"
        selection="four_over_six" if mode=="rtx_jit_four_six" else "amax"
        cfg=replace(cfg,x_scale_region_rows=8,weight_scale_region_rows=8,tensor_scale_mode=outer,
                    quant=replace(cfg.quant,block_scale_selection=selection))
        return rtx.NVFP4Linear(k,n,device="cuda",autotune="off",scaling="jit_row_region",
            dynamic_config=cfg,scale_config=rtx.NVFP4ScaleConfig(tensor_scale_mode=outer),
            backward_config=rtx.DEFAULT_MXFP8_BWD_CONFIG)
    if mode == "rtx_four_six":
        cfg = rtx.NVFP4DynamicConfig(tensor_scale_mode="exact_four_over_six")
        cfg = replace(cfg, quant=replace(cfg.quant, block_scale_selection="four_over_six"))
        return rtx.NVFP4Linear(k,n,device="cuda",autotune="off",scaling="current",
                              scale_config=rtx.NVFP4ScaleConfig(tensor_scale_mode="exact_four_over_six"),
                              dynamic_config=cfg,backward_config=rtx.DEFAULT_MXFP8_BWD_CONFIG)
    from quartet2.linear import Quartet_II_linear
    if mode == "quartet_hybrid":
        class Hybrid(Quartet_II_linear):
            def forward(self, x):
                return QuartetForwardMXBackward.apply(
                    x, self.weight, self.had, self.mode, self.scratch_amax)
        cls = Hybrid
    else:
        cls = Quartet_II_linear
    return cls(k, n, bias=False, device="cuda", dtype=torch.bfloat16,
               four_over_six=mode != "quartet_rne",disable_backward_quant=mode == "quartet_ste")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("bf16", "mxfp8", "rtx_nvfp4", "rtx_four_six", "quartet",
                                      "quartet_rne", "quartet_hybrid", "quartet_ste",
                                      "rtx_jit_four_six", "rtx_jit_amax_exact"), required=True)
    p.add_argument("--shape", default="128,256,256", help="M,N,K")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--graph", action="store_true", help="Only use with a stream/RNG-qualified backend")
    p.add_argument("--graph-unroll", type=int, default=1)
    p.add_argument("--profile", action="store_true", help="Expose five steady calls to Nsight cudaProfilerApi capture")
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--training-steps", type=int, default=0)
    p.add_argument("--seed", type=int, default=20261023)
    p.add_argument("--distribution", choices=("normal", "sparse", "outliers", "zero"), default="normal")
    p.add_argument("--magnitude", type=float, default=1.)
    p.add_argument("--noncontiguous-grad", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    m, n, k = map(int, a.shape.split(","))
    if min(m, n, k, a.samples, a.graph_unroll) <= 0 or a.training_steps < 0:
        p.error("positive shape/samples and nonnegative training steps required")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as out:
        def save(record):
            out.write(json.dumps(record, allow_nan=False) + "\n")
            out.flush()
            print(json.dumps(record, allow_nan=False), flush=True)
        save(dict(kind="environment", mode=a.mode, shape=[m,n,k], seed=a.seed,
                  compiled=a.compile, torch=torch.__version__, cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(), samples=a.samples,
                  distribution=a.distribution, magnitude=a.magnitude,
                  noncontiguous_grad=a.noncontiguous_grad,
                  graph=a.graph, graph_unroll=a.graph_unroll,
                  scope="fixed schedules; separate process per mode"))
        try:
            torch.manual_seed(a.seed)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * .02
            dy = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
            if a.noncontiguous_grad:
                dy = torch.randn(m,2*n,device="cuda",dtype=torch.bfloat16)[:,::2]
            with torch.no_grad():
                for operand in (x, weight, dy):
                    if a.distribution == "sparse":
                        operand.mul_((torch.rand_like(operand.float()) < .05).to(operand.dtype))
                    elif a.distribution == "outliers":
                        operand.mul_(torch.where(torch.rand_like(operand.float()) < .01, 20., 1.))
                    elif a.distribution == "zero":
                        operand.zero_()
                    operand.mul_(a.magnitude)
            original = build(a.mode, k, n)
            with torch.no_grad():
                original.weight.copy_(weight)
            fn = torch.compile(original, fullgraph=True, dynamic=False,
                               options={"triton.cudagraphs": False}) if a.compile else original
            ref = (x.detach().float() @ weight.float().T,
                   dy.float() @ weight.float(), dy.float().T @ x.detach().float())
            ste_ref = None
            if a.mode in ("quartet", "quartet_rne"):
                # Their unbiasedness target is the gradient using already
                # quantized forward operands, not the original BF16 tensors.
                yy = original(x, disable_backward_quant=True)
                ste_ref = tuple(g.detach().float() for g in
                                torch.autograd.grad(yy, (x, original.weight), dy))
                del yy
            sums = [torch.zeros_like(ref[1]), torch.zeros_like(ref[2])]
            first = None
            changed = [False, False]
            start = time.monotonic()
            for i in range(a.samples):
                y = fn(x)
                grads = torch.autograd.grad(y, (x, original.weight), dy)
                values = (y, *grads)
                if not all(bool(v.isfinite().all()) for v in values):
                    raise RuntimeError("nonfinite output/gradient")
                if i == 0:
                    first = [g.detach().clone() for g in grads]
                    save(dict(kind="single_sample", numerics={name: metrics(v,r)
                              for name,v,r in zip(("y","dx","dw"),values,ref)}))
                for j,g in enumerate(grads):
                    sums[j].add_(g.float())
                    changed[j] |= not torch.equal(g, first[j])
            save(dict(kind="sample_mean", samples=a.samples, gradients_changed=changed,
                      dx=metrics(sums[0]/a.samples,ref[1]), dw=metrics(sums[1]/a.samples,ref[2]),
                      setup_and_sampling_seconds=time.monotonic()-start,
                      oracle="original BF16 operands evaluated in FP32; includes forward quantization bias"))
            if ste_ref is not None:
                save(dict(kind="forward_quantized_ste_sample_mean", samples=a.samples,
                          dx=metrics(sums[0]/a.samples,ste_ref[0]),
                          dw=metrics(sums[1]/a.samples,ste_ref[1]),
                          oracle="upstream disable_backward_quant=True BF16 GEMMs"))
            del y, grads, values, first, sums
            def step():
                y = fn(x)
                return y, torch.autograd.grad(y, (x,original.weight), dy)
            # Ordinary current-stream event timing includes Python launch gaps.
            # Do not silently freeze upstream CPU randomness for graph replay.
            for _ in range(5):
                step()
            if a.graph:
                torch.cuda.synchronize()
                capture_stream = torch.cuda.Stream()
                capture_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(capture_stream):
                    for _ in range(3):
                        step()
                torch.cuda.current_stream().wait_stream(capture_stream)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=capture_stream):
                    for _ in range(a.graph_unroll):
                        captured = step()
                graph.replay()
                torch.cuda.synchronize()
                before = captured[0].detach().clone()
                with torch.no_grad():
                    x.mul_(2)
                graph.replay()
                torch.cuda.synchronize()
                observed = captured[0].detach().clone()
                expected = fn(x).detach()
                if torch.equal(before, observed) or not torch.equal(observed, expected):
                    raise RuntimeError("graph changed-input check failed; refusing timing")
                with torch.no_grad():
                    x.mul_(.5)
                timed_step = graph.replay
                save(dict(kind="graph_changed_input", matches_eager=True))
            else:
                timed_step = step
            times=[]
            for _ in range(9):
                begin,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
                begin.record()
                for _ in range(10):
                    result=timed_step()
                end.record(); end.synchronize()
                times.append(begin.elapsed_time(end)*1000/10/(a.graph_unroll if a.graph else 1))
            save(dict(kind="timing", forward_backward_median_us=statistics.median(times),
                      samples_us=times, graph_replay=a.graph, includes_cpu_launch_gaps=not a.graph))
            if a.profile:
                torch.cuda.synchronize()
                torch.cuda.profiler.start()
                for _ in range(5):
                    timed_step()
                torch.cuda.synchronize()
                torch.cuda.profiler.stop()
            if a.training_steps:
                torch.manual_seed(a.seed+100)
                target_w=torch.randn_like(weight)/k**.5
                with torch.no_grad():
                    original.weight.copy_(weight)
                opt=torch.optim.SGD(original.parameters(),lr=1.)
                losses=[]
                for i in range(a.training_steps):
                    gen=torch.Generator(device="cuda").manual_seed(a.seed+1000+i)
                    xx=torch.randn(m,k,device="cuda",dtype=torch.bfloat16,generator=gen,
                                   requires_grad=True)
                    target=xx.detach().float() @ target_w.float().T
                    opt.zero_grad(set_to_none=True)
                    loss=(fn(xx).float()-target).square().mean()
                    loss.backward()
                    if not all(bool(v.isfinite().all()) for v in (loss,xx.grad,original.weight.grad)):
                        raise RuntimeError(f"nonfinite training step {i}")
                    opt.step()
                    losses.append(float(loss.detach()))
                save(dict(kind="training", steps=a.training_steps, optimizer="BF16 SGD lr=1",
                          losses=losses, initial_median=statistics.median(losses[:5]),
                          final_median=statistics.median(losses[-5:]),
                          finite_parameters=bool(original.weight.isfinite().all())))
        except Exception as exc:
            save(dict(kind="error", error=f"{type(exc).__name__}: {exc}"))
            raise


if __name__ == "__main__":
    main()
