"""Matched-policy MXFP8 forward preparation study; no runtime cache promotion."""
import argparse
import json
import os
from pathlib import Path
import statistics
import time
from dataclasses import asdict, replace

import torch
import rtx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shape", default="2048,768,1536")
    parser.add_argument("--backend", choices=("fused", "materialized"), required=True)
    parser.add_argument("--schedule", choices=("default", "tma"), default="default")
    parser.add_argument("--config-json")
    args = parser.parse_args()
    m, n, k = map(int, args.shape.split(","))
    torch.manual_seed(20261023)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * .25
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * .02
    # Independent reference on the exact BF16 operands, not another RTX path.
    reference = x.float() @ w.float().T
    from rtx.kernels.mxfp8 import DEFAULT_MXFP8_FWD_CONFIG, normalize_fwd_config
    from rtx.fp8 import DEFAULT_MXFP8_PREQUANT_CONFIG
    fused = DEFAULT_MXFP8_FWD_CONFIG
    materialized = DEFAULT_MXFP8_PREQUANT_CONFIG
    if args.config_json:
        from rtx.kernels.mxfp8 import fwd_config_from_dict
        from rtx.prequant_autotune import prequant_config_from_dict
        value = json.loads(Path(args.config_json).read_text())
        if args.backend == "fused":
            fused = fwd_config_from_dict(value)
        else:
            materialized = prequant_config_from_dict(value)
    if args.schedule == "tma":
        if args.backend != "fused":
            raise ValueError("tma schedule variant applies only to fused")
        fused = normalize_fwd_config(fused, cta_reuse_tile=(128, 128, 2, 1, 232, 255))
        fused = replace(fused, quant_math="fp32", quant_amax="fp32")
    started = time.monotonic()
    def run():
        return rtx.mxfp8_linear(x, w, backend=args.backend, autotune="off",
                               config=fused if args.backend == "fused" else None,
                               prequant_config=materialized if args.backend == "materialized" else None)
    with torch.no_grad():
        for _ in range(3):
            y = run()
        torch.cuda.synchronize()
        error = y.float() - reference
        metrics = dict(finite=bool(y.isfinite().all()),
                       max_abs=error.abs().max().item(),
                       rms=error.square().mean().sqrt().item(),
                       reference_rms=reference.square().mean().sqrt().item())
        metrics["nrms"] = metrics["rms"] / max(metrics["reference_rms"], 1e-12)
        if not metrics["finite"] or metrics["nrms"] > .06:
            raise RuntimeError(f"numerical gate failed: {metrics}")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            y = run()
        graph.replay()
        torch.cuda.synchronize()
        # Check real input dependence, not merely successful capture.
        x.mul_(2)
        graph.replay()
        torch.cuda.synchronize()
        expected = run()
        torch.cuda.synchronize()
        if not torch.equal(y, expected):
            raise RuntimeError("changed-input graph replay differs from eager")
        x.mul_(.5)
        from benchmarks.profile_preparation_phases import measure
        timing = measure(run, time_based_warmup=True)
        samples = timing["samples_ms"]
    document = dict(schema_version=1, shape=[m,n,k], backend=args.backend,
                    schedule=args.schedule,
                    config=asdict(fused if args.backend == "fused" else materialized),
                    policy="E4M3_E8M0_infinity_default", autotune="off",
                    phase="forward", timing="cuda_graph", samples_ms=samples,
                    measurement_contract="graph8x10_samples15_warm200ms_revision2",
                    graph_warmup=timing["graph_warmup"],
                    median_ms=statistics.median(samples), metrics=metrics,
                    torch=torch.__version__, cuda=torch.version.cuda,
                    device=str(torch.cuda.get_device_properties(0)), seed=20261023,
                    elapsed_seconds=time.monotonic()-started)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(document, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(document), flush=True)


if __name__ == "__main__":
    main()
