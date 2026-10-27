"""Isolated full-backward search across implementation and scheduling choices."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import uuid

from rtx.autotune.ms_eden_design import (
    IMPLEMENTATIONS, MSEdenDesign, make_ms_eden_design_adapter,
)
from rtx.autotune.ms_eden_pipeline import MSEdenPipelineSchedule


def implementation_seeds(schedule):
    """Seed distinct implementations so coupled legality cannot strand search."""
    seeds = []
    for loader in IMPLEMENTATIONS["transpose_loader"]:
        base = schedule
        if loader != "native":
            base = replace(base, **{phase: replace(getattr(base, phase),
                producer_threads=128, producer_async_copy=True,
                producer_load="coalesced_x8", producer_transpose_values_per_lane=16,
                producer_copy_bits=32, producer_unroll_rows=False,
                producer_stages=int(loader[-1])) for phase in ("dx", "dw")})
        for finalizer in IMPLEMENTATIONS["finalizer"]:
            selected = base
            if finalizer != "separate":
                selected = replace(base, backward_pipeline="operand_paired",
                    **{phase: replace(getattr(base, phase), amax_reduction="staged")
                       for phase in ("dx", "dw")})
            seeds.append(MSEdenDesign(selected, loader, finalizer))
    # Interleave families: interruption after one sweep covers every loader
    # and every finalizer, rather than exhausting all native variants first.
    finals = len(IMPLEMENTATIONS["finalizer"])
    return [seeds[loader * finals + (loader + offset) % finals]
            for offset in range(finals) for loader in range(len(IMPLEMENTATIONS["transpose_loader"]))]


def worker(args):
    from benchmarks.ms_eden_design_dispatch import dispatch
    from benchmarks.experiment_ms_eden_operand_order import exact_smoke
    from benchmarks.compare_ms_eden_pipeline import compare, shape_value
    import torch._inductor.config as compiler
    design = MSEdenDesign(**json.loads(args.config_json))
    shape = shape_value(args.shape)
    schedules = {p: json.dumps(asdict(getattr(design.schedule, p))) for p in ("dx", "dw")}
    with dispatch(design) as dispatch_records:
        exact = exact_smoke(shape, schedules, 941)
        with compiler.patch(combo_kernels=True, combo_kernels_pointwise_only=True):
            result = compare(shape, args.output.with_suffix(".raw.json"), {},
                dx_overrides=asdict(design.schedule.dx), dw_overrides=asdict(design.schedule.dw),
                pipeline=design.schedule.backward_pipeline, baseline_policy="stable_mxfp8",
                scopes=("backward",), rounds=args.rounds, replays=args.replays,
                min_batch_ms=args.min_batch_ms, warmup_ms=args.warmup_ms, seed=941)
    result.update(design=asdict(design), design_same_draw=exact,
                  design_dispatch=dispatch_records)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)


def layout_seeds(adapter, anchor):
    """Cross layout/tile/epilogue constraints without illegal intermediate steps."""
    if "dx.scale_layout" not in adapter.axes:
        return []
    seeds = []
    geometries = (
        dict(tile_m=128, tile_n=128, epilogue="tma", store_vec=4),
        dict(tile_m=128, tile_n=128, epilogue="direct", store_vec=1),
        dict(tile_m=256, tile_n=128, epilogue="direct", store_vec=1),
        dict(tile_m=128, tile_n=256, atom_layout_m=4, atom_layout_n=4,
             epilogue="direct", store_vec=1),
    )
    for geometry in geometries:
        for phases in (("dx",), ("dw",), ("dx", "dw")):
            candidate = anchor
            for phase in phases:
                candidate = adapter.update_fn(candidate, f"{phase}.scale_layout", "row_major")
                leg = getattr(candidate.schedule, phase)
                candidate = replace(candidate, schedule=replace(candidate.schedule,
                    **{phase: replace(leg, gemm=replace(leg.gemm, **geometry))}))
            if adapter.rejection(candidate) is None:
                seeds.append(candidate)
            # Test producer-owned scale movement as a coupled layout/ownership
            # proposal rather than relying on a random coordinate to discover it.
            producer = candidate
            for phase in phases:
                producer = adapter.update_fn(producer, f"{phase}.gemm:scale_owner",
                    dict(scale_role="producer", scale_smem_store="scalar", scale_recycle="barrier"))
            if adapter.rejection(producer) is None:
                seeds.append(producer)
    return seeds


def tma_seeds(anchor):
    """Cover transfer geometry and buffering without changing GEMM policy."""
    seeds = [anchor]
    for rows in (32, 64, 128):
        for stages in (1, 2, 3):
            for xor in (4, 5):
                schedule = replace(anchor.schedule, **{
                    phase: replace(getattr(anchor.schedule, phase),
                        producer_threads=128, producer_async_copy=True,
                        producer_load="coalesced_x8", producer_transpose_values_per_lane=16,
                        producer_copy_bits=32, producer_unroll_rows=False,
                        producer_stages=stages) for phase in ("dx", "dw")})
                seeds.append(replace(anchor, schedule=schedule,
                    transpose_loader=f"tma_xor{xor}_{stages}", tma_tile_rows=rows))
    return seeds


def row_producer_seeds(anchor, sm_count):
    """Vary only row dY ownership; preserve all three transpose producers."""
    from rtx.autotune.posthoc import PosthocOperandGeometry
    grids = (sm_count, sm_count * 4, 256)
    seeds = [anchor]
    for blocks in grids:
        for values in (4, 8, 16):
            for threads in (64, 128, 256, 512):
                geometry = anchor.schedule.dx.a_geometry or PosthocOperandGeometry()
                leg = replace(anchor.schedule.dx,
                    a_geometry=replace(geometry, producer_threads=threads, producer_blocks=blocks),
                    producer_values_per_lane=values, producer_row_tile_batch=1)
                seeds.append(replace(anchor, schedule=replace(anchor.schedule, dx=leg)))
    return seeds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shape", default="decoder1536")
    parser.add_argument("--trials", type=int, default=64)
    parser.add_argument("--seconds", type=float, default=1800.)
    parser.add_argument("--trial-timeout", type=float, default=180.)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--allow-row-major", action="store_true",
                        help="opt into experimental row-major scale layout and broader GEMM tiles")
    parser.add_argument("--coverage", choices=("implementation", "layout", "tma", "row_producer"), default="implementation",
                        help="prioritize implementation families or the new layout/tile combinations")
    parser.add_argument("--replays", type=int, default=16)
    parser.add_argument("--min-batch-ms", type=float, default=2.)
    parser.add_argument("--warmup-ms", type=float, default=5000.)
    parser.add_argument("--confirm-only", action="store_true",
                        help="remeasure precisely the selected design, without search")
    parser.add_argument("--row-transform", choices=("native", "fma"),
                        help="explicit row butterfly implementation; recorded in the design identity")
    parser.add_argument("--config-json", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if min(args.trials, args.seconds, args.trial_timeout) <= 0:
        parser.error("trial and time budgets must be positive")
    if args.rounds < 8 or (args.confirm_only and args.rounds < 24):
        parser.error("require >=8 rounds for search and >=24 for confirmation")
    if (args.replays < 16 or not math.isfinite(args.min_batch_ms) or args.min_batch_ms < 2.
            or not math.isfinite(args.warmup_ms) or args.warmup_ms < 5000.):
        parser.error("require >=16 replays, >=2ms batches and >=5000ms warmup")
    if args.config_json:
        return worker(args)
    if args.selection is None:
        parser.error("--selection is required")
    from benchmarks.compare_ms_eden_pipeline import shape_value
    from benchmarks.tune_ms_eden_pipeline import runtime_environment
    from rtx.autotune.first_hit import PortfolioSearch
    from rtx.autotune.core import TuningBudget, Proposal
    from rtx.autotune.orchestrator import AutotuneOrchestrator, ConfirmationPolicy
    from rtx.autotune.bandit import UCB1Scheduler
    from rtx.autotune.strategies import RandomSearch, CoordinateLocalSearch
    from rtx.autotune.store import JsonlTuningStore
    from rtx.autotune.outcomes import TrialOutcome
    shape = shape_value(args.shape)
    source = json.loads(args.selection.read_text())
    selected_design = None
    if "candidate" in source:
        selected = source["candidate"]
        schedule = MSEdenPipelineSchedule(**selected.get("schedule", selected))
        if "schedule" in selected:
            selected_design = MSEdenDesign(**selected)
    else:
        schedule = MSEdenPipelineSchedule(
            backward_pipeline=source["candidate_pipeline"],
            **{p: json.loads(source["schedules"]["candidate"][p]) for p in ("dx", "dw")})
    if args.row_transform is not None:
        selected_design = replace(selected_design or MSEdenDesign(schedule), row_transform=args.row_transform)
    if args.confirm_only and selected_design is None:
        parser.error("confirmation requires a selection containing the exact design")
    if selected_design is not None and any(getattr(selected_design.schedule, p).gemm.scale_layout == "row_major"
                                           for p in ("dx", "dw")):
        args.allow_row_major = True
    if args.coverage == "layout" and not args.allow_row_major:
        parser.error("layout coverage requires --allow-row-major")
    device, environment = runtime_environment()
    repo = Path(__file__).resolve().parents[1]
    # Research dispatch imports multiple implementations. Fingerprint all local
    # source, not just the common runtime, to prevent stale result reuse.
    for folder in ("rtx", "benchmarks"):
        for path in sorted((repo / folder).rglob("*.py")):
            environment["source_sha256"][str(path.relative_to(repo))] = hashlib.sha256(path.read_bytes()).hexdigest()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)

    def evaluate(design):
        trial = root / (adapter.config_id(design) + "-" + uuid.uuid4().hex[:8])
        trial.mkdir()
        command = [sys.executable, "-m", "benchmarks.tune_ms_eden_design",
            "--shape", args.shape, "--output", str(trial / "result.json"),
            "--rounds", str(args.rounds),
            "--replays", str(args.replays), "--min-batch-ms", str(args.min_batch_ms),
            "--warmup-ms", str(args.warmup_ms),
            "--config-json", json.dumps(asdict(design), sort_keys=True)]
        (trial / "manifest.json").write_text(json.dumps({"command": command, "design": asdict(design)}, indent=2))
        print("START", adapter.config_id(design), design.transpose_loader, design.finalizer, flush=True)
        with (trial / "run.log").open("x") as log:
            try:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                           timeout=args.trial_timeout)
            except subprocess.TimeoutExpired:
                raise RuntimeError(f"isolated worker timeout: {trial}")
        if completed.returncode:
            raise RuntimeError(f"worker failed: {trial}\n" + (trial / "run.log").read_text()[-4000:])
        return json.loads((trial / "result.json").read_text())

    seeds = ([selected_design] if args.confirm_only else
             row_producer_seeds(selected_design or MSEdenDesign(schedule), device["multiprocessor_count"])
             if args.coverage == "row_producer" else
             tma_seeds(selected_design or MSEdenDesign(schedule)) if args.coverage == "tma" else
             [selected_design or MSEdenDesign(schedule)] if args.coverage == "layout" else
             ([selected_design] if selected_design is not None else []) + implementation_seeds(schedule))
    adapter = make_ms_eden_design_adapter(shape, evaluate, initial=seeds[0],
        device=device, environment=environment, allow_row_major=args.allow_row_major,
        measurement_contract=f"design_combo_stable_mxfp8_abba{args.rounds}_replays{args.replays}"
            f"_batch{args.min_batch_ms}_warm{args.warmup_ms}_order_gated_v1")
    raw_evaluate = adapter.evaluator
    def safe_evaluate(design):
        try:
            return raw_evaluate(design)
        except (RuntimeError, ValueError, OSError) as error:
            return TrialOutcome("runtime_error", error=str(error))
    adapter.evaluator = safe_evaluate
    if args.allow_row_major and not args.confirm_only:
        seeds.extend(layout_seeds(adapter, seeds[0]))
    seeds = [s for s in seeds if adapter.rejection(s) is None]
    if not seeds:
        raise ValueError("no legal implementation seeds for supplied selection")
    (root / "seeds.json").write_text(json.dumps([asdict(s) for s in seeds], indent=2))
    result = AutotuneOrchestrator(adapter, JsonlTuningStore(root / "residuals"),
        [PortfolioSearch(seeds), RandomSearch(max_pool_size=128, require_timing_eligible=True),
         CoordinateLocalSearch(shuffle_coordinates=True, require_timing_eligible=True)], UCB1Scheduler(),
        TuningBudget(1 if args.confirm_only else args.trials, args.seconds), transfer_history=False,
        confirmation=ConfirmationPolicy(0, .02, True), progress=print,
        coverage_proposals=[Proposal(s, "implementation_coverage") for s in seeds]).tune()
    (root / "selection.json").write_text(json.dumps({"candidate": asdict(result.config),
        "median_ms": result.median_ms, "automatic_promotion": False,
        "requires_independent_confirmation": not args.confirm_only,
        "confirmation_only": args.confirm_only, "rounds": args.rounds,
        "replays": args.replays, "min_batch_ms": args.min_batch_ms,
        "warmup_ms": args.warmup_ms}, indent=2))


if __name__ == "__main__":
    main()
