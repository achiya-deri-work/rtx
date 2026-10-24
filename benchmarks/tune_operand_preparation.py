"""Isolated GPU trials through the shared preparation first-hit adapter."""
import argparse
from dataclasses import replace
import json
import random
from pathlib import Path
import subprocess
import sys
import uuid

from rtx.autotune.adapters import make_mxfp8_fwd_adapter, make_mxfp8_prequant_adapter, make_mxfp8_bwd_adapter
from rtx.autotune.first_hit import FirstHitPolicy
from rtx.autotune.legacy import DeviceFingerprint
from rtx.autotune.outcomes import TrialOutcome
from rtx.autotune.preparation import PreparationCandidate, make_preparation_adapter, tune_preparation, expand_preparation_candidates
from rtx.autotune.preparation_mxfp8 import numerical_policy as policy_of, execution_signature
from rtx.autotune.store import JsonlTuningStore
from rtx.autotune.measurement import runtime_timing_metadata
from rtx.kernels.mxfp8 import MXFP8Problem, DEFAULT_MXFP8_FWD_CONFIG, normalize_fwd_config


def research_candidates(adapter, seeds, *, per_seed=8, max_examined=512,
                        portfolio_seed=None, refinement_plan=None, axis_filters=None):
    expansion_seeds = [c for c in seeds if refinement_plan is None or c.implementation == refinement_plan]
    if not expansion_seeds:
        raise ValueError("refinement plan has no registered seed")
    coordinates = None
    if axis_filters:
        coordinates = [key for key in adapter.axes if any(value in key for value in axis_filters)]
        if not coordinates:
            raise ValueError("axis filters matched no search coordinates")
    expanded = expand_preparation_candidates(adapter, expansion_seeds,
        per_seed=per_seed, max_examined=max_examined, coordinates=coordinates)
    initial_ids = {adapter.config_id(c) for c in seeds}
    refinements = [c for c in expanded if adapter.config_id(c) not in initial_ids]
    if portfolio_seed is not None:
        random.Random(portfolio_seed).shuffle(refinements)
    return [*seeds, *refinements]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--shape", default="2048,768,1536")
    p.add_argument("--trials", type=int, default=8)
    p.add_argument("--phase", choices=("forward", "backward", "dx", "dw"), default="forward")
    p.add_argument("--refine", action="store_true")
    p.add_argument("--expansion-per-seed", type=int, default=8)
    p.add_argument("--max-examined", type=int, default=512)
    p.add_argument("--portfolio-seed", type=int)
    p.add_argument("--refinement-plan", help="Expand this seed family while retaining all baseline plans")
    p.add_argument("--axis-filter", action="append", help="Expand matching coordinate substrings; repeat for OR")
    args = p.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    problem = MXFP8Problem(*map(int, args.shape.split(",")))
    device = DeviceFingerprint.current("cuda:0")
    adapters = {}

    def expand(adapter, seeds):
        return research_candidates(adapter, seeds, per_seed=args.expansion_per_seed,
            max_examined=args.max_examined, portfolio_seed=args.portfolio_seed,
            refinement_plan=args.refinement_plan, axis_filters=args.axis_filter)

    with (root / f"invocation-{uuid.uuid4()}.json").open("x") as handle:
        json.dump(vars(args), handle, indent=2)

    def evaluate(name, config):
        trial = root / str(uuid.uuid4())
        trial.mkdir()
        config_path, result_path = trial / "config.json", trial / "result.json"
        config_path.write_text(json.dumps(adapters[name].serialize(config)))
        with (trial / "run.log").open("w") as log:
            command = (["benchmarks.study_operand_preparation", "--backend", name]
                       if args.phase == "forward" else
                       ["benchmarks.study_backward_preparation", "--plan", name, "--phase", args.phase])
            try:
                process = subprocess.run([sys.executable, "-m", *command,
                    "--shape", args.shape, "--config-json", str(config_path), "--output", str(result_path)],
                    stdout=log, stderr=subprocess.STDOUT, timeout=300)
            except subprocess.TimeoutExpired:
                return TrialOutcome("runtime_error", error="isolated preparation trial exceeded 300s",
                    metadata={"artifact_directory": str(trial), "timeout_seconds": 300})
        if process.returncode:
            return TrialOutcome("runtime_error", error=(trial / "run.log").read_text()[-4000:])
        result = json.loads(result_path.read_text())
        errors = result["metrics"]
        max_abs = errors["max_abs"] if args.phase == "forward" else max(v["max_abs"] for v in errors.values())
        return TrialOutcome("ok", median_ms=result["median_ms"], timings_ms=result["samples_ms"],
            max_abs_error=max_abs, metadata={"artifact": str(result_path), "metrics": errors,
                **runtime_timing_metadata(result["samples_ms"])})

    if args.phase != "forward":
        from rtx.autotune.preparation_mxfp8 import execution_plan
        from benchmarks.study_backward_preparation import build_config
        candidates = []
        for name in ("staged", "fused_dx", "fused_dw", "fused"):
            config = build_config(name)
            adapters[name] = make_mxfp8_bwd_adapter(problem,
                lambda c, name=name: evaluate(name, c), initial=config,
                device=device, tags={"phase": args.phase,
                    "measurement_contract": "graph8x10_samples15_warm200ms_revision2"})
            candidates.append(PreparationCandidate(name, config))
        policy = policy_of("staged", candidates[0].config)
        adapter = make_preparation_adapter(adapters, numerical_policy=policy, policy_of=policy_of,
            equivalence_key=lambda c: execution_signature(c,args.phase))
        candidates = expand(adapter, candidates)
        result = tune_preparation(adapter, JsonlTuningStore(root / "residuals"), candidates,
            score=lambda c: 0, refine=args.refine,
            plan_of=lambda c: execution_plan(c, args.phase),
            policy=FirstHitPolicy(max_trials=args.trials, seconds=900, confirmation_repeats=1))
        print(json.dumps({"config": adapter.serialize(result.config), "trials": result.evaluated_trials,
                          "context": adapter.context.identifier}), flush=True)
        return

    tags = {"measurement_contract": "graph8x10_samples15_warm200ms_revision2"}
    adapters["materialized"] = make_mxfp8_prequant_adapter(problem, lambda c: evaluate("materialized", c), device=device, tags=tags)
    adapters["fused"] = make_mxfp8_fwd_adapter(problem, lambda c: evaluate("fused", c), device=device, tags=tags)
    policy = policy_of("fused", DEFAULT_MXFP8_FWD_CONFIG)
    adapter = make_preparation_adapter(adapters, numerical_policy=policy, policy_of=policy_of,
        equivalence_key=execution_signature)
    candidates = [PreparationCandidate(name, child.initial_config) for name, child in adapters.items()]
    for stages in (1, 2, 3):
        fused = normalize_fwd_config(DEFAULT_MXFP8_FWD_CONFIG,
                                    cta_reuse_tile=(128, 128, stages, 1, 232, 255))
        candidates.append(PreparationCandidate("fused", replace(fused, quant_math="fp32", quant_amax="fp32")))
    staged = adapters["materialized"].initial_config
    for launches in ("separate", "dual"):
        for tile_n in (64, 128, 256):
            candidates.append(PreparationCandidate("materialized", replace(staged,
                quant_launches=launches, gemm=replace(staged.gemm, tile_n=tile_n))))
    candidates = expand(adapter, candidates)
    result = tune_preparation(adapter, JsonlTuningStore(root / "residuals"), candidates,
        score=lambda c: 0, refine=args.refine,
        policy=FirstHitPolicy(max_trials=args.trials, seconds=900, confirmation_repeats=1))
    print(json.dumps({"config": adapter.serialize(result.config), "trials": result.evaluated_trials,
                      "context": adapter.context.identifier}), flush=True)


if __name__ == "__main__":
    main()
