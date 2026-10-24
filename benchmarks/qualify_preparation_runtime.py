"""End-to-end first-hit preparation qualification on a user-sized linear."""
import argparse
import json
import os
from pathlib import Path
import time

import torch
import rtx
from benchmarks.profile_preparation_phases import measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", default="2048,768,1536")
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--phase", choices=("backward", "dx", "dw"), default="backward")
    parser.add_argument("--compiled", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", help="Existing cache, separate from immutable report output")
    parser.add_argument("--require-cache-hit", action="store_true")
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    cache = Path(args.cache_dir) if args.cache_dir else root
    path = cache / "preparation" / "observations.jsonl"
    if args.require_cache_hit and not path.is_file():
        parser.error("--require-cache-hit needs an existing preparation journal")
    before = path.read_bytes() if path.is_file() else None
    os.environ["RTX_PREPARATION_SELECTION"] = "1"
    torch.manual_seed(20261023)
    m, n, k = map(int, args.shape.split(","))
    layer = rtx.MXFP8Linear(k, n, device="cuda", backend="auto", autotune="balanced",
        tuning_policy=rtx.AutotunePolicy(max_trials=args.trials, max_seconds=120,
                                       confirmation_repeats=1), autotune_cache_dir=cache)
    layer.weight.requires_grad_(args.phase != "dx")
    x = (torch.randn(m, k, device="cuda", dtype=torch.bfloat16)*.25).requires_grad_(args.phase != "dw")
    dy = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)*.25
    forward = torch.compile(layer, fullgraph=True, dynamic=False) if args.compiled else layer
    inputs = tuple(t for t in (x, layer.weight) if t.requires_grad)
    def step():
        y = forward(x)
        return y, torch.autograd.grad(y, inputs, dy)
    start = time.monotonic()
    y, grads = step()
    torch.cuda.synchronize()
    cold_seconds = time.monotonic()-start
    if args.require_cache_hit:
        assert path.read_bytes() == before, "fresh-process first invocation retuned"
    with torch.no_grad():
        references = [x.float() @ layer.weight.float().T]
        names = ["forward"]
        if x.requires_grad:
            references.append(dy.float() @ layer.weight.float()); names.append("dx")
        if layer.weight.requires_grad:
            references.append(dy.float().T @ x.float()); names.append("dw")
        metrics = {}
        for name, actual, reference in zip(names, (y, *grads), references):
            rms = (actual.float()-reference).square().mean().sqrt().item()
            rrms = reference.square().mean().sqrt().item()
            metrics[name] = dict(rms=rms, reference_rms=rrms, nrms=rms/max(rrms,1e-12))
            if not actual.isfinite().all() or rms > max(.06*rrms,1e-6):
                raise RuntimeError(f"{name} failed FP32 reference: {metrics[name]}")
    evidence = path.read_bytes()
    # Do not retain the cold forward's autograd graph across the benchmark's
    # side-stream warmup/capture: its AccumulateGrad nodes belong to stream 0.
    del y, grads, actual
    timing = measure(step, time_based_warmup=True)
    assert path.read_bytes() == evidence, "hot execution changed tuning observations"
    rows = [json.loads(line) for line in evidence.splitlines()]
    contexts = {}
    for row in rows:
        contexts.setdefault(row["context_id"], []).append(row)
    from rtx.autotune.measurement import runtime_timing_metadata
    result = dict(arguments=vars(args), cold_seconds=cold_seconds, metrics=metrics,
        measurement_contract="graph8x10_samples15_warm200ms_revision2",
        persistent_cache_hit=before is not None and evidence == before,
        hot_forward_backward=timing, timing_quality=runtime_timing_metadata(timing["samples_ms"]),
        contexts={key: dict(observations=len(values),
                    statuses=[r["outcome"]["status"] for r in values],
                    stationary=[r["outcome"].get("metadata", {}).get("sampling", {}).get(
                        "collection", {}).get("stationary") for r in values])
                  for key,values in contexts.items()},
        torch=torch.__version__, cuda=torch.version.cuda,
        device=str(torch.cuda.get_device_properties(0)))
    (root / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
