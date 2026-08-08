"""Search spaces for persistent MXFP8 inference operand states."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

from .configs import (
    MXFP8FullyPrequantConfig,
    MXFP8GemmConfig,
    MXFP8QuantConfig,
    MXFP8WeightPrequantConfig,
)
from .prequant_autotune import PREQUANT_SEARCH_SPACE


INFERENCE_KERNEL_REVISION = 1

_WEIGHT_ACTIVE_AXES = (
    "layout_transport",
    "x_vector_load",
    "x_arithmetic",
    "x_launch",
    "x_registers",
    "x_scale_store",
    "gemm_geometry",
    "gemm_stages",
    "ldmatrix",
    "smem_swizzle",
    "scale_s2r",
    "scale_schedule",
    "scale_cache",
    "gemm_registers",
    "producer_registers",
    "consumer_registers",
    "gemm_maxrregcount",
    "epilogue",
    "raster_group",
    "global_l2_fetch",
)
_FULLY_ACTIVE_AXES = (
    "layout_transport",
    "gemm_geometry",
    "gemm_stages",
    "ldmatrix",
    "smem_swizzle",
    "scale_s2r",
    "scale_schedule",
    "scale_cache",
    "gemm_registers",
    "producer_registers",
    "consumer_registers",
    "gemm_maxrregcount",
    "epilogue",
    "raster_group",
    "global_l2_fetch",
)

MXFP8_WEIGHT_PREQUANT_SEARCH_SPACE = {
    name: PREQUANT_SEARCH_SPACE[name] for name in _WEIGHT_ACTIVE_AXES
}
MXFP8_FULLY_PREQUANT_SEARCH_SPACE = {
    name: PREQUANT_SEARCH_SPACE[name] for name in _FULLY_ACTIVE_AXES
}


def _identifier(value: Mapping[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def weight_prequant_config_to_dict(
    config: MXFP8WeightPrequantConfig,
) -> dict[str, object]:
    return asdict(config)


def weight_prequant_config_from_dict(
    value: Mapping[str, object],
) -> MXFP8WeightPrequantConfig:
    return MXFP8WeightPrequantConfig(
        quant_x=MXFP8QuantConfig(**dict(value["quant_x"])),  # type: ignore[arg-type]
        gemm=MXFP8GemmConfig(**dict(value["gemm"])),  # type: ignore[arg-type]
        l2_fetch_granularity=(
            None
            if value.get("l2_fetch_granularity") is None
            else int(value["l2_fetch_granularity"])
        ),
    )


def weight_prequant_config_id(config: MXFP8WeightPrequantConfig) -> str:
    return _identifier(weight_prequant_config_to_dict(config))


def update_weight_prequant_config(
    config: MXFP8WeightPrequantConfig,
    updates: Mapping[str, object],
) -> MXFP8WeightPrequantConfig:
    quant = asdict(config.quant_x)
    gemm = asdict(config.gemm)
    quant.update(dict(updates.get("quant", {})))  # type: ignore[arg-type]
    gemm.update(dict(updates.get("gemm", {})))  # type: ignore[arg-type]
    return MXFP8WeightPrequantConfig(
        quant_x=MXFP8QuantConfig(**quant),
        gemm=MXFP8GemmConfig(**gemm),
        l2_fetch_granularity=updates.get(  # type: ignore[arg-type]
            "l2_fetch_granularity", config.l2_fetch_granularity
        ),
    )


def fully_prequant_config_to_dict(
    config: MXFP8FullyPrequantConfig,
) -> dict[str, object]:
    return asdict(config)


def fully_prequant_config_from_dict(
    value: Mapping[str, object],
) -> MXFP8FullyPrequantConfig:
    return MXFP8FullyPrequantConfig(
        gemm=MXFP8GemmConfig(**dict(value["gemm"])),  # type: ignore[arg-type]
        l2_fetch_granularity=(
            None
            if value.get("l2_fetch_granularity") is None
            else int(value["l2_fetch_granularity"])
        ),
    )


def fully_prequant_config_id(config: MXFP8FullyPrequantConfig) -> str:
    return _identifier(fully_prequant_config_to_dict(config))


def update_fully_prequant_config(
    config: MXFP8FullyPrequantConfig,
    updates: Mapping[str, object],
) -> MXFP8FullyPrequantConfig:
    gemm = asdict(config.gemm)
    gemm.update(dict(updates.get("gemm", {})))  # type: ignore[arg-type]
    return MXFP8FullyPrequantConfig(
        gemm=MXFP8GemmConfig(**gemm),
        l2_fetch_granularity=updates.get(  # type: ignore[arg-type]
            "l2_fetch_granularity", config.l2_fetch_granularity
        ),
    )


def _layout_updates_for(
    axes: Mapping[str, tuple[dict[str, object], ...]],
    *,
    weight_layout: str,
    activation_layout: str | None,
) -> tuple[dict[str, object], ...]:
    from .configs.inference import gemm_operand_scale_layouts

    result = []
    for update in axes["layout_transport"]:
        gemm = dict(update.get("gemm", {}))
        layouts = gemm_operand_scale_layouts(str(gemm["scale_layout"]))
        if layouts[1] != weight_layout:
            continue
        if activation_layout is not None and layouts[0] != activation_layout:
            continue
        result.append(update)
    return tuple(result)


def tune_mxfp8_inference_state(
    problem,
    *,
    state: str,
    weight_layout: str,
    activation_layout: str | None = None,
    device="cuda",
    cache_dir: Path | str | None = None,
    policy=None,
    progress=print,
):
    """Tune one packed inference state and publish its runtime winner."""

    from .autotune import (
        CalibratedPrequantEvaluator,
        DeviceFingerprint,
        HybridTuningPolicy,
        JsonlTuningStore,
        make_hybrid_autotuner,
        make_mxfp8_fully_prequant_adapter,
        make_mxfp8_weight_prequant_adapter,
    )
    from .autotune.legacy import default_cache_dir
    from .autotune.winners import runtime_winner_key, save_runtime_winner
    from .inference_experiments import (
        FullyPrequantBenchmarkHarness,
        WeightPrequantBenchmarkHarness,
    )
    from .prequant_experiments import BenchmarkProtocol, ShapeSpec

    root = default_cache_dir() if cache_dir is None else Path(cache_dir).expanduser()
    fingerprint = DeviceFingerprint.current(device)
    samples = int(getattr(policy, "samples", 7))
    protocol = BenchmarkProtocol(
        warmup_calls=int(getattr(policy, "warmup", 5)),
        samples=samples,
        confirm_samples=max(15, samples),
        race_rounds=max(15, samples),
        target_batch_ms=50.0,
        max_calls_per_sample=max(1, int(getattr(policy, "calls_per_sample", 4096))),
        correctness_rtol=float(getattr(policy, "correctness_rtol", 5e-2)),
        correctness_atol=float(getattr(policy, "correctness_atol", 5e-1)),
        telemetry=False,
    )
    shape = ShapeSpec(problem.m, problem.n, problem.k)
    if state == "weight_prequantized":
        axes = dict(MXFP8_WEIGHT_PREQUANT_SEARCH_SPACE)
        axes["layout_transport"] = _layout_updates_for(
            axes,
            weight_layout=weight_layout,
            activation_layout=None,
        )
        if not axes["layout_transport"]:
            raise RuntimeError(
                f"no AOT-weight search family supports W layout {weight_layout}"
            )
        initial = MXFP8WeightPrequantConfig()
        for update in axes["layout_transport"]:
            candidate = update_weight_prequant_config(initial, update)
            if candidate.rejection(problem) is None:
                initial = candidate
                break
        harness = WeightPrequantBenchmarkHarness(
            shape, "hot", protocol, device=device, seed=20260809
        )
        evaluator = CalibratedPrequantEvaluator(
            harness, samples=samples, seed=20260809
        )
        adapter = make_mxfp8_weight_prequant_adapter(
            problem,
            evaluator,
            initial=initial,
            axes=axes,
            device=fingerprint,
            regime="hot",
        )
        variant = f"w-{weight_layout}"
    elif state == "fully_prequantized":
        if activation_layout is None:
            raise ValueError("fully packed tuning requires activation_layout")
        axes = dict(MXFP8_FULLY_PREQUANT_SEARCH_SPACE)
        axes["layout_transport"] = _layout_updates_for(
            axes,
            weight_layout=weight_layout,
            activation_layout=activation_layout,
        )
        if not axes["layout_transport"]:
            raise RuntimeError(
                "no fully packed search family supports layouts "
                f"{(activation_layout, weight_layout)}"
            )
        initial = MXFP8FullyPrequantConfig()
        for update in axes["layout_transport"]:
            candidate = update_fully_prequant_config(initial, update)
            if candidate.rejection(problem) is None:
                initial = candidate
                break
        harness = FullyPrequantBenchmarkHarness(
            shape, "hot", protocol, device=device, seed=20260809
        )
        evaluator = CalibratedPrequantEvaluator(
            harness, samples=samples, seed=20260809
        )
        adapter = make_mxfp8_fully_prequant_adapter(
            problem,
            evaluator,
            initial=initial,
            axes=axes,
            device=fingerprint,
            regime="hot",
        )
        variant = f"x-{activation_layout}_w-{weight_layout}"
    else:
        raise ValueError(f"unsupported inference state {state!r}")

    if isinstance(policy, HybridTuningPolicy):
        hybrid = policy
    else:
        hybrid = HybridTuningPolicy(
            max_trials=512,
            time_budget_s=float(
                getattr(
                    policy,
                    "time_budget_s",
                    os.getenv("RTX_MXFP8_AUTOTUNE_SECONDS", "1800"),
                )
            ),
            seed=int(getattr(policy, "seed", 20260809)),
        )
    store = JsonlTuningStore(
        root
        / "runtime_sessions"
        / adapter.context.family
        / fingerprint.identifier
        / f"m{problem.m}_n{problem.n}_k{problem.k}_{variant}"
    )
    result = make_hybrid_autotuner(
        adapter, store, hybrid, progress=progress
    ).tune()
    key = runtime_winner_key(
        adapter.context.family,
        problem,
        fingerprint=fingerprint,
        variant=variant,
    )
    save_runtime_winner(
        key,
        adapter.serialize(result.config),
        config_id=adapter.config_id(result.config),
        root=root,
        median_ms=result.median_ms,
        metadata={"context_id": result.context_id, "source": "runtime_tuner"},
    )
    return result.config


__all__ = [
    "INFERENCE_KERNEL_REVISION",
    "MXFP8_FULLY_PREQUANT_SEARCH_SPACE",
    "MXFP8_WEIGHT_PREQUANT_SEARCH_SPACE",
    "fully_prequant_config_from_dict",
    "fully_prequant_config_id",
    "fully_prequant_config_to_dict",
    "update_fully_prequant_config",
    "update_weight_prequant_config",
    "weight_prequant_config_from_dict",
    "weight_prequant_config_id",
    "weight_prequant_config_to_dict",
    "tune_mxfp8_inference_state",
]
