"""Correctness-gated outer-loop sweep; no winner installation or policy changes.

Run each family/shape in a fresh process. Captured timings include preparation;
MLP timings include all recomputation and gradients. Baseline is remeasured
after the sweep, and every result is appended independently before proceeding.
"""
from dataclasses import asdict, replace
import argparse
import json
from pathlib import Path
import statistics
import time

import torch
import rtx


def schedule_bwd(config, factor):
    return replace(config,
        dx=replace(config.dx, gemm=replace(config.dx.gemm, k_tile_unroll=factor)),
        dw=replace(config.dw, gemm=replace(config.dw.gemm, k_tile_unroll=factor)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("linear", "backward", "glu", "mlp"), required=True)
    parser.add_argument("--shape", default="512,768,1536", help="M,N,K; for MLP M,H,K with O=K")
    parser.add_argument("--factors", default="1,2,3,4,6,8,12")
    parser.add_argument("--samples", type=int, default=11)
    parser.add_argument("--calls", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    m, n, k = map(int, args.shape.split(","))
    factors = list(dict.fromkeys([1, *map(int, args.factors.split(","))]))
    if args.samples < 3 or args.calls < 1:
        parser.error("at least three samples and one call required")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(21345)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)*.2
    w = torch.randn(n if args.family in ("linear", "backward") else 2*n,
                    k, device="cuda", dtype=torch.bfloat16)*.02
    g = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)*.1
    down = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)*.02
    dy = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)*.1
    expected = None

    def make(factor):
        if args.family == "linear":
            from rtx.fp8 import _build_prequant_runner, _intern_prequant_config
            cfg = rtx.DEFAULT_MXFP8_PREQUANT_CONFIG
            cfg = replace(cfg, gemm=replace(cfg.gemm, k_tile_unroll=factor))
            out = torch.empty(m, n, device=x.device, dtype=x.dtype)
            runner = _build_prequant_runner(x, w, _intern_prequant_config(cfg))
            def call():
                runner(x, w, out)
                return (out,)
        elif args.family == "backward":
            from rtx.fp8_bwd import _build_bwd_runner
            from rtx.kernels.mxfp8 import MXFP8Problem
            cfg = schedule_bwd(rtx.DEFAULT_MXFP8_BWD_CONFIG, factor)
            runner = _build_bwd_runner(MXFP8Problem(m, n, k), cfg, x.device)
            dx, dw = torch.empty_like(x), torch.empty_like(w)
            def call():
                runner(g, x, w, dx, dw)
                return dx, dw
        elif args.family == "glu":
            from rtx.kernels.mxfp8_glu_prequant import build_prequant_glu_runner
            cfg = rtx.MXFP8GLUPrequantConfig()
            cfg = replace(cfg, gemm=replace(cfg.gemm, k_tile_unroll=factor))
            out = torch.empty(m, n, device=x.device, dtype=torch.float8_e4m3fn)
            sf = torch.empty(m, n//32, device=x.device, dtype=torch.uint8)
            runner = build_prequant_glu_runner(x, w, out, sf, cfg, "silu", "mxfp8")
            def call():
                runner()
                return out, sf
        else:
            cfg = replace(rtx.MXFP8MLPConfig(), gate_backend="prequant", down_backend="prequant",
                          recompute_backend="prequant",
                          gate_backward=schedule_bwd(rtx.DEFAULT_MXFP8_BWD_CONFIG, factor),
                          down_backward=schedule_bwd(rtx.DEFAULT_MXFP8_BWD_CONFIG, factor))
            cfg = replace(cfg,
                gate_prequant=replace(cfg.gate_prequant, gemm=replace(cfg.gate_prequant.gemm, k_tile_unroll=factor)),
                recompute=replace(cfg.recompute, gemm=replace(cfg.recompute.gemm, k_tile_unroll=factor)),
                down_gemm=replace(cfg.down_gemm, k_tile_unroll=factor))
            inputs = tuple(t.detach().requires_grad_() for t in (x, w, down))
            torch._dynamo.reset()
            fn = torch.compile(lambda a,b,c: rtx.mxfp8_mlp(a,b,c,config=cfg),
                               fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
            def call():
                y = fn(*inputs)
                return y, *torch.autograd.grad(y, inputs, dy)
        return call, cfg

    with args.output.open("x") as log:
        for index, factor in enumerate([*factors, 1]):
            started = time.monotonic()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                call, cfg = make(factor)
                outputs = call()
            stream.synchronize()
            if expected is None:
                expected = tuple(o.detach().clone() for o in outputs)
            for actual, ref in zip(outputs, expected):
                assert bool(actual.float().isfinite().all())
                torch.testing.assert_close(actual.float(), ref.float(), rtol=0, atol=0)
            del outputs
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(5):
                    call()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(args.calls):
                    outputs = call()
            warm = time.monotonic()
            while time.monotonic()-warm < .2:
                graph.replay()
                torch.cuda.synchronize()
            samples = []
            for _ in range(args.samples):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(10):
                    graph.replay()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end)*1000/(10*args.calls))
            row = dict(family=args.family, shape=[m,n,k], factor=factor,
                       baseline_confirmation=index == len(factors), config=asdict(cfg),
                       median_us=statistics.median(samples), samples_us=samples,
                       elapsed_s=time.monotonic()-started, bitwise_equal=True,
                       gpu=torch.cuda.get_device_name(), torch=torch.__version__, seed=21345)
            log.write(json.dumps(row)+"\n")
            log.flush()
            print(f"{args.family} {args.shape} unroll={factor} {row['median_us']:.3f}us", flush=True)
            del outputs, graph, call


if __name__ == "__main__":
    main()
