"""Policy-matched dX/dW preparation timing and independent numerical checks."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import statistics

import torch
import rtx
from rtx.bwd_autotune import bwd_config_from_dict, bwd_config_to_dict
from rtx.fp8_bwd import _intern_bwd_config, _mxfp8_linear_dx_op, _mxfp8_linear_dw_op
from rtx.kernels.mxfp8 import normalize_fwd_config
from rtx.kernels.mxfp8_bwd import DEFAULT_DUAL_DECOMPOSED_MXFP8_BWD_CONFIG


from rtx.autotune.preparation_mxfp8 import backward_plan_config as build_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shape", default="2048,768,1536")
    p.add_argument("--plan", choices=("staged", "fused", "fused_dx", "fused_dw"), default="staged")
    p.add_argument("--phase", choices=("dx", "dw", "backward"), default="backward")
    p.add_argument("--config-json")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    config = bwd_config_from_dict(json.loads(Path(args.config_json).read_text())) if args.config_json else build_config(args.plan)
    m, n, k = map(int, args.shape.split(","))
    torch.manual_seed(20261023)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * .25
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * .02
    dy = torch.randn(m, n, device="cuda", dtype=torch.bfloat16) * .25
    key = _intern_bwd_config(config)

    def call(dy, x, w):
        if args.phase == "dx":
            return (_mxfp8_linear_dx_op(dy, x, w, key),)
        if args.phase == "dw":
            return (_mxfp8_linear_dw_op(dy, x, w, key),)
        return rtx.mxfp8_linear_backward(dy, x, w, config=config, autotune="off")

    run = torch.compile(call, fullgraph=True, dynamic=False) if args.compile else call
    refs = {"dx": dy.float() @ w.float(), "dw": dy.float().T @ x.float()}
    names = ("dx", "dw") if args.phase == "backward" else (args.phase,)
    for _ in range(3):
        outputs = run(dy, x, w)
    torch.cuda.synchronize()
    metrics = {}
    for name, output in zip(names, outputs):
        error = output.float() - refs[name]
        rms = error.square().mean().sqrt().item()
        ref_rms = refs[name].square().mean().sqrt().item()
        metrics[name] = dict(finite=bool(output.isfinite().all()), rms=rms,
                             reference_rms=ref_rms, nrms=rms / max(ref_rms, 1e-12),
                             max_abs=error.abs().max().item())
        if not metrics[name]["finite"] or metrics[name]["nrms"] > .06:
            raise RuntimeError(f"{name} numerical gate failed: {metrics[name]}")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run(dy, x, w)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outputs = run(dy, x, w)
    dy.mul_(2)
    graph.replay()
    torch.cuda.synchronize()
    eager = call(dy, x, w)
    torch.cuda.synchronize()
    if not all(torch.equal(a, b) for a, b in zip(outputs, eager)):
        raise RuntimeError("changed-input graph replay mismatch")
    dy.mul_(.5)
    from benchmarks.profile_preparation_phases import measure
    timing = measure(lambda: run(dy, x, w), time_based_warmup=True)
    samples = timing["samples_ms"]
    result = dict(shape=[m, n, k], phase=args.phase, plan=args.plan, config=bwd_config_to_dict(config),
        compiled=args.compile, metrics=metrics, median_ms=statistics.median(samples), samples_ms=samples,
        measurement_contract="graph8x10_samples15_warm200ms_revision2",
        graph_warmup=timing["graph_warmup"],
        torch=torch.__version__, cuda=torch.version.cuda, device=str(torch.cuda.get_device_properties(0)),
        seed=20261023, timing="cuda_graph", policy="original_bf16_mxfp8_fp32_infinity")
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({k: v for k, v in result.items() if k not in ("config", "samples_ms")}), flush=True)


if __name__ == "__main__":
    main()
