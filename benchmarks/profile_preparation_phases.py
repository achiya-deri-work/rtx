"""Graph-timed phase decomposition of the actual staged MXFP8 runners."""
import argparse
import json
import os
from pathlib import Path
import statistics

import torch
from rtx.fp8 import _build_prequant_runner, _intern_prequant_config, DEFAULT_MXFP8_PREQUANT_CONFIG
from rtx.fp8_bwd import _build_partial_bwd_runner
from rtx.kernels.mxfp8 import MXFP8Problem
from rtx.kernels.mxfp8_bwd import DEFAULT_DUAL_DECOMPOSED_MXFP8_BWD_CONFIG
from rtx.bwd_autotune import bwd_config_from_dict, bwd_config_to_dict
from rtx.prequant_autotune import prequant_config_from_dict, prequant_config_to_dict


def completed_winner(root, shape, implementation):
    """Read a completed research search, never a partial or timing-invalid winner."""
    root = Path(root) / "residuals"
    sessions = [json.loads(line) for line in (root / "sessions.jsonl").read_text().splitlines()]
    finish = next((row for row in reversed(sessions) if row.get("event") == "finish"), None)
    if finish is None:
        raise ValueError("preparation search has not completed")
    start = next(row for row in sessions if row.get("event") == "start"
                 and row["session_id"] == finish["session_id"])
    workload = start["context"]["workload"]
    if tuple(workload[key] for key in ("m", "n", "k")) != tuple(shape):
        raise ValueError("winner workload does not match phase study")
    from rtx.autotune.outcomes import TrialOutcome
    rows = [json.loads(line) for line in (root / "observations.jsonl").read_text().splitlines()]
    winner = next(row for row in reversed(rows)
                  if row["context_id"] == start["context_id"]
                  and row["config_id"] == finish["best_config_id"])
    outcome = winner["outcome"]
    if not TrialOutcome(**outcome).timing_eligible:
        raise ValueError("winner timing is ineligible")
    if winner["config"]["implementation"] != implementation:
        raise ValueError("fused winner cannot be decomposed into staged phases")
    return winner["config"]["config"]


def measure(fn, *, time_based_warmup=False):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(8):
            fn()
    warmup=None
    if time_based_warmup:
        from rtx.autotune.preparation_runtime import _warm_measurement_graph
        warmup=_warm_measurement_graph(graph)
    else:
        for _ in range(10):
            graph.replay()
    samples = []
    for _ in range(15):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(10):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / 80)
    result=dict(median_ms=statistics.median(samples), min_ms=min(samples), samples_ms=samples)
    if warmup is not None:
        result["graph_warmup"]=warmup
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shape", default="2048,768,1536")
    p.add_argument("--forward-config")
    p.add_argument("--backward-config")
    p.add_argument("--forward-trials", help="Completed tune_operand_preparation output")
    p.add_argument("--backward-trials", help="Completed tune_operand_preparation output")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if (args.forward_config and args.forward_trials) or (args.backward_config and args.backward_trials):
        p.error("choose config or trials per direction, not both")
    m, n, k = map(int, args.shape.split(","))
    problem = MXFP8Problem(m,n,k)
    fc = prequant_config_from_dict(json.loads(Path(args.forward_config).read_text())) if args.forward_config else DEFAULT_MXFP8_PREQUANT_CONFIG
    bc = bwd_config_from_dict(json.loads(Path(args.backward_config).read_text())) if args.backward_config else DEFAULT_DUAL_DECOMPOSED_MXFP8_BWD_CONFIG
    if args.forward_trials:
        fc = prequant_config_from_dict(completed_winner(args.forward_trials, (m,n,k), "materialized"))
    if args.backward_trials:
        bc = bwd_config_from_dict(completed_winner(args.backward_trials, (m,n,k), "staged"))
    if any(leg.backend != "decomposed" for leg in (bc.dx, bc.dw)) or bc.quant_schedule != "per_matmul" or bc.stream_schedule != "single":
        raise ValueError("phase study requires staged per-matmul single-stream backward")
    torch.manual_seed(20261023)
    x = torch.randn(m,k,device="cuda",dtype=torch.bfloat16) * .25
    w = torch.randn(n,k,device="cuda",dtype=torch.bfloat16) * .02
    dy = torch.randn(m,n,device="cuda",dtype=torch.bfloat16) * .25
    y, dx, dw = torch.empty_like(dy), torch.empty_like(x), torch.empty_like(w)
    fwd = _build_prequant_runner(x,w,_intern_prequant_config(fc))
    xr = _build_partial_bwd_runner(problem,bc,x.device,"dx")
    wr = _build_partial_bwd_runner(problem,bc,x.device,"dw")

    def quant_fwd():
        if fwd.quant_launches == "dual":
            fwd.quant_x(x,w,fwd.qx,fwd.qw,fwd.sx,fwd.sw)
        else:
            fwd.quant_x(x,fwd.qx,fwd.sx)
            fwd.quant_w(w,fwd.qw,fwd.sw)

    phases = {
        "forward_prepare": quant_fwd,
        "forward_gemm": lambda: fwd.gemm(fwd.qx,fwd.qw,fwd.sx,fwd.sw,y),
        "dx_prepare": lambda: xr.quantize(dy,w),
        "dx_gemm": lambda: xr.matmul(dx),
        "dw_prepare": lambda: wr.quantize(dy,x),
        "dw_gemm": lambda: wr.matmul(dw),
    }
    def complete():
        for fn in phases.values():
            fn()
    complete()
    torch.cuda.synchronize()
    metrics = {}
    refs = {"forward": x.float() @ w.float().T, "dx": dy.float() @ w.float(), "dw": dy.float().T @ x.float()}
    for name, out in (("forward", y), ("dx", dx), ("dw", dw)):
        err = out.float() - refs[name]
        rms, rrms = err.square().mean().sqrt().item(), refs[name].square().mean().sqrt().item()
        metrics[name] = dict(finite=bool(out.isfinite().all()), rms=rms, reference_rms=rrms,
                             nrms=rms/max(rrms,1e-12), max_abs=err.abs().max().item())
        if not metrics[name]["finite"] or metrics[name]["nrms"] > .06:
            raise RuntimeError(f"numerical gate failed: {name} {metrics[name]}")
    timings = {name: measure(fn, time_based_warmup=True) for name, fn in phases.items()}
    timings["full_sequence"] = measure(complete, time_based_warmup=True)
    from rtx.autotune.measurement import runtime_timing_metadata
    result = dict(shape=[m,n,k], metrics=metrics, timings=timings,
        forward_config=prequant_config_to_dict(fc), backward_config=bwd_config_to_dict(bc),
        source_trials={"forward": args.forward_trials, "backward": args.backward_trials},
        device=str(torch.cuda.get_device_properties(0)), torch=torch.__version__, cuda=torch.version.cuda,
        seed=20261023, measurement="graph8x10_samples15_warm200ms_revision2",
        timing_quality={name: runtime_timing_metadata(value["samples_ms"]) for name,value in timings.items()},
        caveat="isolated hot phases change cache reuse; full sequence is authoritative")
    path = Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x") as handle:
        json.dump(result,handle,indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({name: value["median_ms"] for name,value in timings.items()}),flush=True)


if __name__ == "__main__":
    main()
