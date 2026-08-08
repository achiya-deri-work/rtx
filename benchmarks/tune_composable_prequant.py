"""Run the composable learned/global + local tuner on dynamic MXFP8 forward."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from rtx.autotune import (
    CalibratedPrequantEvaluator,
    DeviceFingerprint,
    HybridTuningPolicy,
    JsonlTuningStore,
    import_legacy_json_database,
    make_hybrid_autotuner,
    make_mxfp8_prequant_adapter,
)
from rtx.prequant_autotune import prequant_config_from_dict
from rtx.prequant_experiments import (
    BenchmarkProtocol,
    PrequantBenchmarkHarness,
    ShapeSpec,
)


def _legacy_best_trial(path: Path):
    document = json.loads(path.read_text(encoding="utf-8"))
    successful = [
        trial
        for trial in document.get("trials", {}).values()
        if trial.get("status") == "ok" and trial.get("median_ms") is not None
    ]
    if not successful:
        raise RuntimeError(f"legacy database {path} has no successful trials")
    return prequant_config_from_dict(
        min(successful, key=lambda trial: float(trial["median_ms"]))["config"]
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=512)
    parser.add_argument("--n", type=int, default=1536)
    parser.add_argument("--k", type=int, default=1536)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--legacy-db", type=Path)
    parser.add_argument("--initial-config", type=Path)
    parser.add_argument("--trials", type=int, default=512)
    parser.add_argument("--seconds", type=float, default=1800.0)
    parser.add_argument("--model-trials", type=int, default=320)
    parser.add_argument("--model-warmup", type=int, default=32)
    parser.add_argument("--model-pool", type=int, default=4096)
    parser.add_argument("--model-refit", type=int, default=32)
    parser.add_argument("--model-estimators", type=int, default=24)
    parser.add_argument("--model-ensembles", type=int, default=3)
    parser.add_argument("--model-features", type=int, default=64)
    parser.add_argument("--local-model-refit", type=int, default=32)
    parser.add_argument("--confirmation-repeats", type=int, default=0)
    parser.add_argument("--confirmation-ratio", type=float, default=0.0)
    parser.add_argument("--confirm-initial", action="store_true")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--target-batch-ms", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--protocol-tag", default="calibrated_batch_v1")
    parser.add_argument(
        "--orchestration", choices=("sequential", "bandit"), default="sequential"
    )
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    shape = ShapeSpec(args.m, args.n, args.k)
    protocol = BenchmarkProtocol(
        warmup_calls=5,
        samples=args.samples,
        confirm_samples=max(11, args.samples),
        race_rounds=15,
        target_batch_ms=args.target_batch_ms,
        telemetry=False,
    )
    harness = PrequantBenchmarkHarness(shape, "hot", protocol, seed=args.seed)
    evaluator = CalibratedPrequantEvaluator(
        harness, samples=args.samples, seed=args.seed
    )
    initial = None
    if args.initial_config is not None:
        initial_document = json.loads(
            args.initial_config.read_text(encoding="utf-8")
        )
        initial = prequant_config_from_dict(initial_document["config"])
    elif args.legacy_db is not None:
        initial = _legacy_best_trial(args.legacy_db)
    adapter = make_mxfp8_prequant_adapter(
        evaluator.problem,
        evaluator,
        initial=initial,
        device=DeviceFingerprint.current(harness.device),
        regime="hot",
        tags={"protocol": args.protocol_tag},
    )
    store = JsonlTuningStore(args.store, fsync=False)
    imported = 0
    if args.legacy_db is not None:
        imported = import_legacy_json_database(args.legacy_db, adapter, store)
    policy = HybridTuningPolicy(
        orchestration=args.orchestration,
        max_trials=args.trials,
        time_budget_s=args.seconds,
        cost_model_trials=args.model_trials,
        model_warmup=args.model_warmup,
        model_pool_size=args.model_pool,
        model_refit_interval=args.model_refit,
        model_estimators=args.model_estimators,
        model_ensembles=args.model_ensembles,
        model_max_features=args.model_features,
        local_model_refit_interval=args.local_model_refit,
        confirmation_repeats=args.confirmation_repeats,
        confirmation_ratio=args.confirmation_ratio,
        confirm_initial=args.confirm_initial,
        seed=args.seed,
    )
    print(f"IMPORTED {imported}", flush=True)
    result = make_hybrid_autotuner(
        adapter, store, policy, progress=lambda message: print(message, flush=True)
    ).tune()
    payload = {
        "context_id": result.context_id,
        "session_id": result.session_id,
        "config_id": adapter.config_id(result.config),
        "median_ms": result.median_ms,
        "evaluated_trials": result.evaluated_trials,
        "elapsed_s": result.elapsed_s,
        "strategy_trials": result.strategy_trials,
        "config": adapter.serialize(result.config),
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
