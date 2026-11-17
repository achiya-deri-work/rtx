"""Matched-source MoE execution benchmark: host latency versus outer graph replay.

The controller launches sequential fresh worker processes in ABBA order. It
never runs two GPU workers concurrently. Compilation, correctness and graph
capture are outside timing. Both source snapshots need the frozen
``benchmarks/profile_moe.py`` helper and the same seed-study files.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def summary(samples):
    median = statistics.median(samples)
    return dict(median_ms=median, min_ms=min(samples), max_ms=max(samples),
                relative_mad=statistics.median(abs(v-median) for v in samples)/max(median, 1e-12),
                samples_ms=samples)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("worker", "compare"):
        s = sub.add_parser(name)
        s.add_argument("--output", type=Path, required=True)
        s.add_argument("--case-id", default="d768_e48_k6_h256")
        s.add_argument("--tokens", type=int, default=2000)
        s.add_argument("--seed-studies", nargs="+", type=Path, default=[])
        s.add_argument("--config", type=Path)
        s.add_argument("--whole-layer", action="store_true")
        s.add_argument("--include-bf16", action="store_true")
        s.add_argument("--samples", type=int, default=9)
        s.add_argument("--rounds", type=int, default=2)
        s.add_argument("--device", default="cuda:0")
        s.add_argument("--phases", nargs="+", default=["expert_step", "public_expert_step"])
    worker = sub.choices["worker"]
    worker.add_argument("--source", type=Path, required=True)
    worker.add_argument("--label", default="candidate")
    worker.add_argument("--fixture", type=Path)
    worker.add_argument("--save-fixture", type=Path)
    worker.add_argument("--tune-trials", type=int, default=0)
    worker.add_argument("--tune-seconds", type=float, default=120.)
    compare = sub.choices["compare"]
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--baseline-config", type=Path)
    compare.add_argument("--candidate-config", type=Path)
    compare.add_argument("--python", type=Path, default=Path(sys.executable))
    return p


def _use_source(source):
    source = source.resolve()
    current = Path(__file__).resolve().parents[1]
    # This script can reside outside either snapshot. Do not accidentally use
    # its project's editable installation or namespace benchmark package.
    sys.path[:] = [str(source)] + [p for p in sys.path
        if p and Path(p).resolve() not in (current, current / "benchmarks", source)]
    os.chdir(source)
    rtx = importlib.import_module("rtx")
    helper = importlib.import_module("benchmarks.profile_moe")
    assert Path(rtx.__file__).resolve().is_relative_to(source), rtx.__file__
    assert Path(helper.__file__).resolve().is_relative_to(source), helper.__file__
    return helper


def _tensor_digest(tensor):
    value = tensor.detach().contiguous().cpu().reshape(-1).view(__import__("torch").uint8)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _host(call, samples):
    import torch
    values = []
    # Clock ramp is excluded. Keep no saved training graph from a prior call.
    until = time.monotonic() + .25
    while time.monotonic() < until:
        output = call()
        del output
    torch.cuda.synchronize()
    for _ in range(samples):
        start = time.perf_counter()
        output = call()
        torch.cuda.synchronize()
        values.append((time.perf_counter()-start)*1000)
        del output
    return summary(values)


def worker(args):
    args.source = args.source.resolve()
    args.output = args.output.resolve()
    args.config = args.config.resolve() if args.config else None
    args.seed_studies = [p.resolve() for p in args.seed_studies]
    args.fixture = args.fixture.resolve() if args.fixture else None
    args.save_fixture = args.save_fixture.resolve() if args.save_fixture else None
    helper = _use_source(args.source)
    import torch
    from rtx.autotune.moe import source_revision
    from rtx.moe import mxfp8_routed_experts
    from rtx.routing import BalancedRouting

    torch.cuda.set_device(args.device)
    requested_trials = args.tune_trials
    args.tune_trials = 0  # Bound the search explicitly below, outside timing.
    args.skip_baseline = True  # Here baseline means a source revision, not BF16.
    if args.whole_layer and args.fixture:
        raise ValueError("Whole-layer source comparison uses matched state/input digests, not an expert-only fixture")
    began = time.monotonic()
    case, shape, config, origin, operands, dy, calls = helper.build(args)
    x, gu, down, ids, weights, inverse = operands
    tuning = None
    if requested_trials:
        from rtx.autotune.effort import AutotunePolicy
        from rtx.autotune.moe import tune_moe
        from rtx.moe import _selection_spec, _parse_spec, _run_forward, _run_backward
        calls["frozen_expert_step"] = calls["expert_step"]
        policy = AutotunePolicy(max_trials=requested_trials, max_seconds=args.tune_seconds,
            search_schedule="progressive", candidate_mix="mixed", confirmation_repeats=0)
        cold = time.monotonic()
        result = tune_moe(operands, _parse_spec(_selection_spec(config, "balanced", policy,
            args.output.parent/"tuning")), training=True, progress=lambda message: print(message, flush=True))
        tuning = dict(cold_seconds=time.monotonic()-cold, requested_trials=requested_trials,
            max_seconds=args.tune_seconds, initial_config=config.to_dict(), selected_config=result.config.to_dict(),
            median_ms=result.median_ms, evidence=getattr(result, "tuning_evidence", None))
        config = result.config
        calls["expert_forward"] = lambda: _run_forward(*operands, config)
        calls["expert_backward"] = lambda: _run_backward(*operands, dy, config)
        calls["expert_step"] = lambda: (calls["expert_forward"](), *calls["expert_backward"]())
        if args.whole_layer:
            model = inspect.getclosurevars(calls["whole_step"]).nonlocals["native"]._orig_mod
            model.config = config
            model._expert_spec = _selection_spec(config, "off", None, None)
    public = torch.compile(lambda x, gu, down, weights: mxfp8_routed_experts(
        x, gu, down, BalancedRouting(ids, weights, inverse), config=config),
        fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
    calls["public_expert_forward"] = lambda: public(x, gu, down, weights)
    calls["public_expert_step"] = lambda: torch.autograd.grad(public(x, gu, down, weights),
                                                             (x, gu, down, weights), dy)
    if args.include_bf16:
        from rtx.moe import torch_routed_experts
        baseline = torch.compile(lambda x, gu, down, weights: torch_routed_experts(
            x, gu, down, BalancedRouting(ids, weights, inverse)), fullgraph=True, dynamic=False,
            options={"triton.cudagraphs": False})
        calls["bf16_expert_step"] = lambda: torch.autograd.grad(baseline(x, gu, down, weights),
                                                               (x, gu, down, weights), dy)
    phases = list(args.phases)
    if args.whole_layer and "whole_step" not in phases:
        phases.append("whole_step")
    for phase in phases:
        if phase not in calls:
            raise ValueError(f"Unknown/unavailable phase {phase!r}; available={sorted(calls)}")
    record = dict(status="running", label=args.label, case=case, shape=shape,
        source=str(args.source.resolve()), source_revision=source_revision(), config=config.to_dict(),
        seed_origin=origin, commands=sys.argv, torch=torch.__version__, cuda=torch.version.cuda,
        device=str(torch.cuda.get_device_properties(torch.cuda.current_device())),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        helper_sha256=hashlib.sha256(Path(helper.__file__).read_bytes()).hexdigest(),
        compile_options=dict(fullgraph=True, dynamic=False, options={"triton.cudagraphs": False}),
        timing_contract="boundary-synchronized host and externally captured CUDAgraph; no automatic per-op graphs",
        tuning=tuning,
        input_digests={name:_tensor_digest(t) for name,t in zip(
            ("x","gate_up","down","ids","weights","inverse","dy"),(*operands,dy))})
    if args.whole_layer:
        # The established frozen helper closes over all real model parameters.
        parameters = inspect.getclosurevars(calls["whole_step"]).nonlocals["params"]
        record["whole_parameter_digests"] = [_tensor_digest(t) for t in parameters]
    for phase in phases:
        output = calls[phase]()
        del output
    torch.cuda.synchronize()
    record["construction_compile_seconds"] = time.monotonic()-began
    validation_start = time.monotonic()
    actual_public = lambda: (calls["public_expert_forward"](), *calls["public_expert_step"]())
    # The helper oracle is independent FP64 quantized Torch. For large N it
    # checks both full-C weight gradients on first/last experts, all finite.
    record["validation_public"] = helper.correctness(operands, dy, {"expert_step":actual_public}, args.tokens)
    record["validation_seconds"] = time.monotonic()-validation_start
    record["timings"] = {phase:dict(host=[], graph=[]) for phase in phases}
    write(args.output, record)
    stream = torch.cuda.Stream()
    for round_id in range(args.rounds):
        order = phases if round_id % 2 == 0 else list(reversed(phases))
        for phase in order:
            call = calls[phase]
            host = _host(call, args.samples)
            graph = helper.measure(call, stream, args.samples)
            record["timings"][phase]["host"].append(host)
            record["timings"][phase]["graph"].append(graph)
            print("TIME", args.label, round_id, phase,
                  f"host={host['median_ms']:.6f}ms graph={graph['median_ms']:.6f}ms", flush=True)
            write(args.output, record)
    live = []
    for _ in range(8):
        output = calls["public_expert_step"]()
        del output
        torch.cuda.synchronize()
        live.append(torch.cuda.memory_allocated())
    record.update(status="complete", live_allocated_samples=live, live_growth=max(live)-min(live))
    write(args.output, record)


def aggregate(records):
    rows = []
    if not records:
        return rows
    expected = records[0]["input_digests"]
    for record in records:
        if record["input_digests"] != expected:
            raise ValueError("Input/route/parameter digests differ across source workers")
        if record.get("whole_parameter_digests") != records[0].get("whole_parameter_digests"):
            raise ValueError("Whole-model parameter digests differ across source workers")
        if record["status"] != "complete":
            raise ValueError("Cannot aggregate an incomplete worker")
    phases = list(records[0]["timings"])
    for phase in phases:
        for mode in ("host", "graph"):
            values = {label:[statistics.median(r["median_ms"] for r in record["timings"][phase][mode])
                             for record in records if record["label"] == label]
                      for label in ("baseline", "candidate")}
            if not all(values.values()):
                raise ValueError("Both source labels are required")
            medians = {key:statistics.median(value) for key,value in values.items()}
            rows.append(dict(phase=phase, execution=mode, baseline_ms=medians["baseline"],
                candidate_ms=medians["candidate"], speedup=medians["baseline"]/medians["candidate"],
                worker_medians=values,
                worker_relative_range={k:max(v)/min(v)-1 for k,v in values.items()}))
    return rows


def compare(args):
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    fixture = args.output / "matched-expert-inputs.pt"
    records = []
    for index, label in enumerate(("baseline", "candidate", "candidate", "baseline")):
        source = getattr(args, label).resolve()
        destination = args.output / f"visit-{index:02d}-{label}.json"
        # Resolving a venv's python symlink jumps to the system interpreter and
        # silently drops its site-packages. Keep the launcher path intact.
        command = [str(args.python.absolute()), str(script), "worker", "--source", str(source),
            "--label", label, "--output", str(destination), "--case-id", args.case_id,
            "--tokens", str(args.tokens), "--device", args.device, "--samples", str(args.samples),
            "--rounds", str(args.rounds), "--phases", *args.phases]
        config = getattr(args, label+"_config") or args.config
        if config:
            command += ["--config", str(config.resolve())]
        if args.seed_studies:
            command += ["--seed-studies", *[str(p.resolve()) for p in args.seed_studies]]
        if args.whole_layer:
            command.append("--whole-layer")
        elif index == 0:
            command += ["--save-fixture", str(fixture)]
        else:
            command += ["--fixture", str(fixture)]
        if args.include_bf16:
            command.append("--include-bf16")
        environment = dict(os.environ, TORCHINDUCTOR_COMPILE_THREADS="2",
            TORCHINDUCTOR_CACHE_DIR=str(args.output / ("inductor-"+label)), PYTHONPATH=str(source))
        print("START", index, label, flush=True)
        with destination.with_suffix(".log").open("w") as log:
            result = subprocess.run(command, cwd=source, env=environment, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"{label} worker failed ({result.returncode}); inspect {destination.with_suffix('.log')}")
        records.append(json.loads(destination.read_text()))
        if index >= 1:
            write(args.output / "comparison.json", dict(status="running", order=[r["label"] for r in records],
                                                       results=aggregate(records)))
    write(args.output / "comparison.json", dict(status="complete", order=[r["label"] for r in records],
        scope="Matched precision/config policies unless explicit per-source config override; source ABBA, not BF16 baseline",
        results=aggregate(records), workers=[str(args.output/f"visit-{i:02d}-{r['label']}.json") for i,r in enumerate(records)]))
    print("DONE", args.output / "comparison.json", flush=True)


def main():
    args = parser().parse_args()
    if args.samples < 3 or args.rounds < 1:
        raise ValueError("At least three samples and one round are required")
    worker(args) if args.command == "worker" else compare(args)


if __name__ == "__main__":
    main()
