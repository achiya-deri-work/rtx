"""Budgeted serial-worker complete-backward research autotuning.

Uses the existing first-hit search and append-only residual store. Selections
are artifacts only: no runtime defaults or deployment winners are installed.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import uuid

from benchmarks.compare_ms_eden_pipeline import shape_value
from rtx.autotune.first_hit import FirstHitPolicy, tune_first_hit
from rtx.autotune.ms_eden_pipeline import MSEdenPipelineSchedule, make_ms_eden_pipeline_adapter
from rtx.autotune.outcomes import TrialOutcome
from rtx.autotune.store import JsonlTuningStore


def preset(name):
    base = MSEdenPipelineSchedule()
    if name == "baseline":
        return base
    if name in ("paired", "operand_paired"):
        return replace(base, backward_pipeline=name)
    if name == "staged":
        return replace(base, dx=replace(base.dx, amax_reduction="staged"), dw=replace(base.dw, amax_reduction="staged"))
    if name in ("buffered2", "buffered3"):
        stages = int(name[-1])
        return replace(base, dx=replace(base.dx, producer_stages=stages, producer_async_copy=True),
                       dw=replace(base.dw, producer_stages=stages, producer_async_copy=True))
    raise ValueError("unknown pipeline preset")


def candidates(adapter, pool_size):
    """Small structural seeds first, then round-robin full adapter coordinates.

    Coupled async+stage seeds remain reachable even when a one-coordinate
    intermediate would be rejected. Never enumerate a Cartesian product.
    """
    seeds = [adapter.initial_config, preset("paired"), preset("staged"),
             preset("buffered2"), preset("buffered3")]
    base = adapter.initial_config
    paired = replace(base, backward_pipeline="paired")
    # Keep the historical first seeds, but do not spend a small paired search
    # exclusively on serial buffering or one leg's first coordinate values.
    for stages in (2, 3):
        seeds.append(replace(paired, **{phase: replace(getattr(paired, phase),
            producer_stages=stages, producer_async_copy=True) for phase in ("dx", "dw")}))
    for blocks, finalizer_blocks in ((64, 64), (128, 128), (512, 256)):
        seeds.append(replace(paired, **{phase: replace(getattr(paired, phase),
            producer_blocks=blocks, finalizer_blocks=finalizer_blocks) for phase in ("dx", "dw")}))
    # Two legal geometry challengers per leg provide GEMM coverage within a
    # 16-trial portfolio, without assuming any particular geometry is fastest.
    geometry = {}
    for phase in ("dx", "dw"):
        geometry[phase] = []
        seen_geometry = {adapter.config_id(paired)}
        for value in adapter.axes.get(f"{phase}.gemm:gemm_geometry", ()):
            config = adapter.update_fn(paired, f"{phase}.gemm:gemm_geometry", value)
            identity = adapter.config_id(config)
            if identity not in seen_geometry and adapter.rejection(config) is None:
                seen_geometry.add(identity)
                geometry[phase].append(config)
                if len(geometry[phase]) == 2:
                    break
    for index in range(2):
        for phase in ("dx", "dw"):
            if index < len(geometry[phase]):
                seeds.append(geometry[phase][index])
        if index == 0:
            for phase in ("dx", "dw"):
                seeds.append(replace(paired, **{phase: replace(getattr(paired, phase),
                    producer_blocks=64, finalizer_blocks=64)}))
    for load in ("coalesced_x4", "coalesced_x8"):
        seeds.append(replace(paired, **{phase: replace(getattr(paired, phase),
            producer_load=load, producer_stages=2, producer_async_copy=True) for phase in ("dx", "dw")}))
    coordinates = []
    for coordinate in adapter.axes:
        if coordinate.startswith("dx."):
            coordinates.extend((coordinate, "dw." + coordinate[3:]))
        elif not coordinate.startswith("dw."):
            coordinates.append(coordinate)
    portfolios = [[adapter.update_fn(base, coordinate, value)
                   for value in adapter.axes[coordinate]]
                  for coordinate in coordinates if coordinate in adapter.axes]
    seen, result = set(), []
    def add(config):
        identity = adapter.config_id(config)
        if identity not in seen and adapter.rejection(config) is None:
            seen.add(identity)
            result.append(config)
    for config in seeds:
        add(config)
    for index in range(max(map(len, portfolios), default=0)):
        for portfolio in portfolios:
            if len(result) >= pool_size:
                return result[:pool_size]
            if index < len(portfolio):
                add(portfolio[index])
    return result[:pool_size]


def evaluation_command(args, config, output):
    return [sys.executable, "-m", "benchmarks.compare_ms_eden_pipeline",
        "--shape", args.shape, "--output", str(output), "--scope", "backward",
        "--candidate-json", "{}", "--candidate-pipeline", config.backward_pipeline,
        "--candidate-dx-json", json.dumps(asdict(config.dx), sort_keys=True),
        "--candidate-dw-json", json.dumps(asdict(config.dw), sort_keys=True),
        "--rounds", str(args.rounds), "--replays", str(args.replays),
        "--min-batch-ms", str(args.min_batch_ms), "--warmup-ms", str(args.warmup_ms), "--seed", str(args.seed)]


def evaluate_subprocess(args, config, root):
    directory = root / "trials" / uuid.uuid4().hex
    directory.mkdir(parents=True)
    output = directory / "result.json"
    command = evaluation_command(args, config, output)
    with (directory / "manifest.json").open("x") as handle:
        json.dump(dict(command=command, config=asdict(config)), handle, indent=2)
    with (directory / "run.log").open("x") as log:
        try:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.trial_timeout)
        except subprocess.TimeoutExpired:
            return dict(worker_failure="isolated trial timeout", artifact=str(directory), status="runtime_error")
    if result.returncode:
        tail = (directory / "run.log").read_text()[-4000:]
        return dict(worker_failure=tail, artifact=str(directory), status="runtime_error")
    if not output.exists():
        return dict(worker_failure="worker returned no public result", artifact=str(directory), status="runtime_error")
    result = json.loads(output.read_text())
    result["artifact"] = str(output)
    return result


def runtime_environment():
    import importlib.metadata
    import torch
    from rtx.autotune.legacy import DeviceFingerprint
    root = Path(__file__).resolve().parents[1]
    paths = ("benchmarks/tune_ms_eden_pipeline.py", "benchmarks/compare_ms_eden_pipeline.py",
        "benchmarks/compare_mixed_nvfp4_backward.py", "rtx/experimental_ms_eden.py",
        "rtx/kernels/nvfp4_posthoc.py", "rtx/kernels/nvfp4_rng.py", "rtx/autotune/posthoc.py",
        "rtx/autotune/ms_eden_pipeline.py")
    device = DeviceFingerprint.current("cuda:0").as_dict()
    props = torch.cuda.get_device_properties(0)
    device["multiprocessor_count"] = props.multi_processor_count
    device["l2_cache_size"] = int(getattr(props, "L2_cache_size", 0) or 0)
    return device, dict(torch=torch.__version__, cuda=torch.version.cuda,
        cutlass_dsl=importlib.metadata.version("nvidia-cutlass-dsl"),
        source_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in paths})


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shape", default="tiny")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--trials", type=int, default=8)
    p.add_argument("--seconds", type=float, default=1800.)
    p.add_argument("--trial-timeout", type=float, default=600.)
    p.add_argument("--pool-size", type=int, default=128)
    p.add_argument("--confirmation-repeats", type=int, default=0)
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--replays", type=int, default=16)
    p.add_argument("--min-batch-ms", type=float, default=2.)
    p.add_argument("--warmup-ms", type=float, default=3000.)
    p.add_argument("--seed", type=int, default=941)
    p.add_argument("--initial-preset", choices=("baseline", "paired", "operand_paired", "staged", "buffered2", "buffered3"), default="baseline")
    p.add_argument("--initial-json", help="serialized MSEdenPipelineSchedule, overrides preset")
    p.add_argument("--refine", action="store_true", help="spend remaining cumulative budget instead of compatible cache hit")
    p.add_argument("--dry-run", action="store_true", help="print candidates and axes without CUDA or filesystem writes")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    shape = shape_value(args.shape)
    args.shape = ",".join(map(str, shape))
    if args.trials < 1 or args.pool_size < 1 or args.rounds < 4 or args.replays < 1 or args.min_batch_ms < 2 or args.warmup_ms < 0 or args.trial_timeout <= 0:
        raise ValueError("positive budgets/replays, rounds>=4, batch>=2ms and warmup>=0 required")
    policy = FirstHitPolicy(max_trials=args.trials, seconds=args.seconds, pool_size=args.pool_size,
                            confirmation_repeats=args.confirmation_repeats)
    initial = MSEdenPipelineSchedule(**json.loads(args.initial_json)) if args.initial_json else preset(args.initial_preset)
    device, environment = ({}, {}) if args.dry_run else runtime_environment()
    contract = f"public_fullgraph_complete_bwd_abba_rounds{args.rounds}_replays{args.replays}_batch{args.min_batch_ms}_warm{args.warmup_ms}_seed{args.seed}_v1"
    root = args.output.resolve()
    adapter = make_ms_eden_pipeline_adapter(shape, lambda config: evaluate_subprocess(args, config, root),
        device=device, environment=environment, measurement_contract=contract, initial=initial)
    # Preserve worker diagnostics instead of misclassifying compile/runtime
    # failures as missing numerical evidence. All successful artifacts still
    # pass the adapter's strict complete-backward converter.
    def evaluate(config):
        result = evaluate_subprocess(args, config, root)
        if "worker_failure" in result:
            return TrialOutcome(result["status"], error=result["worker_failure"], metadata=dict(artifact=result["artifact"]))
        from rtx.autotune.ms_eden_pipeline import pipeline_outcome
        outcome = pipeline_outcome(result, expected_config=config, expected_shape=shape)
        outcome.metadata["artifact"] = result["artifact"]
        return outcome
    adapter.evaluator = evaluate
    portfolio = candidates(adapter, args.pool_size)
    if args.dry_run:
        print(json.dumps(dict(context=adapter.context.as_dict(), axes=adapter.axes,
            candidates=[adapter.serialize(config) for config in portfolio], automatic_promotion=False), indent=2))
        return
    root.mkdir(parents=True, exist_ok=True)
    store = JsonlTuningStore(root / "residuals")
    # AutotuneOrchestrator.tune owns the same-context transaction lock. Taking
    # it here too self-deadlocks on a second flock descriptor in one process.
    result = tune_first_hit(adapter, store, portfolio, policy=policy, refine=args.refine)
    selection = root / f"selection-{uuid.uuid4().hex}.json"
    with selection.open("x") as handle:
        json.dump(dict(config=adapter.serialize(result.config), median_ms=result.median_ms,
            context=adapter.context.as_dict(), arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            automatic_promotion=False, scope="research selection only; no runtime winner installed"), handle, indent=2)
    print("SELECTION", selection, result.median_ms, adapter.serialize(result.config))


if __name__ == "__main__":
    main()
