"""Composable search spaces for persistent NVFP4 inference states."""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

from .configs.nvfp4 import (
    DEFAULT_NVFP4_DYNAMIC_CONFIG,
    NVFP4DynamicConfig,
    NVFP4FullyPrequantConfig,
    NVFP4GemmConfig,
    NVFP4Problem,
    NVFP4QuantConfig,
    NVFP4WeightPrequantConfig,
)
from .prequant_autotune import PREQUANT_SEARCH_SPACE


NVFP4_INFERENCE_KERNEL_REVISION = 3
NVFP4_DYNAMIC_KERNEL_REVISION = 6


def _gemm_axes() -> dict[str, tuple[dict[str, object], ...]]:
    names = (
        "gemm_geometry",
        "gemm_stages",
        "ldmatrix",
        "mma_schedule",
        "smem_swizzle",
        "scale_s2r",
        "scale_schedule",
        "scale_recycle",
        "scale_smem_store",
        "scale_cache",
        "gemm_registers",
        "producer_registers",
        "consumer_registers",
        "gemm_maxrregcount",
        "epilogue",
        "raster_group",
        "gemm_persistence",
        "global_l2_fetch",
    )
    axes = {name: PREQUANT_SEARCH_SPACE[name] for name in names}
    axes["gemm_geometry"] = tuple(
        {
            "gemm": {
                "tile_m": tile_m,
                "tile_n": tile_n,
                "tile_k": tile_k,
                "atom_layout_m": atom_m,
                "atom_layout_n": atom_n,
            }
        }
        for tile_m, atom_m in (
            (64, 2),
            (128, 2),
            (128, 4),
            (128, 8),
            (256, 8),
        )
        for tile_n, atom_n in ((128, 2), (256, 4))
        for tile_k in (64, 128, 256)
    )
    fixed_persistence = tuple(
        {
            "gemm": {
                **dict(update["gemm"]),
                "persistent_waves": 0,
            }
        }
        for update in PREQUANT_SEARCH_SPACE["gemm_persistence"]
    )
    balanced_persistence = tuple(
        {
            "gemm": {
                "tiles_per_cta": tile_cap,
                "tile_locality": locality,
                "persistent_waves": waves,
            }
        }
        for tile_cap in (2, 4, 8)
        for waves in (1, 2, 3, 4)
        for locality in (
            "raster",
            "same_a",
            "same_b",
            "serpentine_a",
            "serpentine_b",
        )
    )
    axes["gemm_persistence"] = fixed_persistence + balanced_persistence
    return axes


_QUANT_VECTOR = tuple(
    {"quant": {"values_per_lane": values, "load_bits": bits}}
    for values in (2, 4, 8, 16)
    for bits in (16, 32, 64, 128)
    if bits <= values * 16 and (values * 16) % bits == 0
)
_QUANT_REDUCTION = tuple(
    {"quant": {"reduction": reduction}}
    for reduction in ("shuffle", "redux")
)
_QUANT_LAUNCH = tuple(
    {"quant": {"num_warps": warps, "persistent_waves": waves}}
    for warps in (4, 8, 16)
    for waves in (1, 2, 3, 4, 6, 8)
)
_QUANT_REGISTERS = tuple(
    {"quant": {"maxrregcount": registers}}
    for registers in (64, 80, 96, 112, 128, 160, 192, 224, 255)
)
_QUANT_RECIPROCAL = tuple(
    {"quant": {"scale_reciprocal": mode}}
    for mode in ("direct", "e4m3_lut", "rcp_approx")
)
_QUANT_SCALE_COMPUTE = tuple(
    {"quant": {"scale_compute": mode}}
    for mode in ("redundant", "leader_broadcast")
)

_NATIVE_DYNAMIC_ANCHOR = NVFP4DynamicConfig(
    quant=NVFP4QuantConfig(
        values_per_lane=8,
        load_bits=16,
        reduction="shuffle",
        num_warps=8,
        persistent_waves=6,
        maxrregcount=128,
        scale_layout="mma128",
    ),
    gemm=NVFP4GemmConfig(
        tile_m=128,
        tile_n=128,
        tile_k=128,
        atom_layout_m=8,
        atom_layout_n=2,
        stages=3,
        a_ldmatrix_matrices=4,
        b_ldmatrix_matrices=2,
        a_swizzle="64b",
        b_swizzle="64b",
        scale_role="tma",
        scale_layout="mma128",
        sfa_s2r_bits=0,
        sfb_s2r_bits=8,
        producer_registers=24,
        consumer_registers=128,
        maxrregcount=192,
        epilogue="tma",
        epilogue_stages=1,
        store_vec=4,
        raster="n",
        grid_swizzle=2,
        tiles_per_cta=4,
        tile_locality="same_b",
        persistent_waves=1,
    ),
    quant_launches="dual",
)


# Compound implementation anchors keep the tuner from having to rediscover
# known-good materialized-GEMM basins one independent coordinate at a time.
# They are seeds, not hard-coded winners: every field remains mutable through
# the ordinary axes below, and device/shape-specific results still have to win
# measurement and confirmation.
_DYNAMIC_IMPLEMENTATION_ANCHORS = (
    # Native physical E4M3 scales let the GEMM's producer warp move operands
    # and scales through the same TMA pipeline.  Three stages are decisive on
    # SM120: the older one-stage native-scale comparison could not overlap the
    # transfers and incorrectly made consumer-side scalar staging look faster.
    asdict(_NATIVE_DYNAMIC_ANCHOR),
    # Large CTA grids can profit from a higher persistent wave floor.  Keep a
    # second seed because the best grid is discontinuous: on the 5070 Ti the
    # 576-tile asymmetric probe drops from 68.8 to 54.9 us at three waves,
    # while the 144-tile square probe prefers the one-wave balanced grid.
    asdict(
        replace(
            _NATIVE_DYNAMIC_ANCHOR,
            gemm=replace(
                _NATIVE_DYNAMIC_ANCHOR.gemm,
                persistent_waves=3,
            ),
        )
    ),
    # Row-major scales remain useful for shapes which cannot use mma128 and as
    # an independent fallback basin on future SM12x devices.
    asdict(
        NVFP4DynamicConfig(
            quant=NVFP4QuantConfig(
                values_per_lane=8,
                load_bits=16,
                reduction="shuffle",
                num_warps=8,
                persistent_waves=6,
                maxrregcount=128,
            ),
            gemm=NVFP4GemmConfig(
                tile_m=128,
                tile_n=128,
                tile_k=128,
                atom_layout_m=8,
                atom_layout_n=2,
                stages=1,
                a_ldmatrix_matrices=4,
                b_ldmatrix_matrices=4,
                a_swizzle="64b",
                b_swizzle="64b",
                scale_role="consumers",
                scale_schedule="before_wait",
                scale_load_vec=4,
                sfa_s2r_bits=0,
                sfb_s2r_bits=8,
                producer_registers=24,
                consumer_registers=128,
                maxrregcount=192,
                epilogue="tma",
                epilogue_stages=2,
                store_vec=4,
                raster="n",
                grid_swizzle=2,
                # A cap of four lets the balanced scheduler choose one CTA per
                # SM for 144 output tiles on the 70-SM 5070 Ti. A cap of two
                # forces 72 CTAs and recreates the two-CTA tail wave.
                tiles_per_cta=4,
                tile_locality="same_b",
                persistent_waves=1,
            ),
            quant_launches="dual",
        )
    ),
)


def preferred_dynamic_config(problem: NVFP4Problem) -> NVFP4DynamicConfig:
    """Return the fastest known portable starting basin for ``problem``.

    Native MMA-layout scales are not merely a local schedule adjustment: the
    quantizer and GEMM transport must change together. Starting eligible
    searches from the row-major fallback left short, breadth-first campaigns
    unable to reach this compound basin. Keep the general row-major fallback
    for shapes which cannot represent complete 128x128 scale tiles.
    """

    native = dynamic_config_from_dict(_DYNAMIC_IMPLEMENTATION_ANCHORS[0])
    return native if native.rejection(problem) is None else DEFAULT_NVFP4_DYNAMIC_CONFIG

NVFP4_WEIGHT_PREQUANT_SEARCH_SPACE = {
    "x_vector_load": _QUANT_VECTOR,
    "x_reduction": _QUANT_REDUCTION,
    "x_launch": _QUANT_LAUNCH,
    "x_registers": _QUANT_REGISTERS,
    "x_scale_reciprocal": _QUANT_RECIPROCAL,
    "x_scale_compute": _QUANT_SCALE_COMPUTE,
    **_gemm_axes(),
}
NVFP4_FULLY_PREQUANT_SEARCH_SPACE = _gemm_axes()
NVFP4_DYNAMIC_SEARCH_SPACE = {
    "implementation_anchor": _DYNAMIC_IMPLEMENTATION_ANCHORS,
    "quant_vector_load": _QUANT_VECTOR,
    "quant_reduction": _QUANT_REDUCTION,
    "quant_launch": _QUANT_LAUNCH,
    "quant_registers": _QUANT_REGISTERS,
    "quant_scale_reciprocal": _QUANT_RECIPROCAL,
    "quant_scale_compute": _QUANT_SCALE_COMPUTE,
    "quant_launches": tuple(
        {"quant_launches": value}
        for value in ("dual", "independent", "concurrent")
    ),
    "scale_transport": (
        {
            "quant": {"scale_layout": "row_major"},
            "gemm": {
                "scale_layout": "row_major",
                "scale_role": "consumers",
            },
        },
        {
            "quant": {"scale_layout": "mma128"},
            "gemm": {
                "scale_layout": "mma128",
                "scale_role": "tma",
                "scale_schedule": "before_wait",
                "scale_load_vec": 4,
                "scale_l2_prefetch": "none",
                "scale_l1_evict": "default",
                "scale_cache": "default",
            },
        },
    ),
    **_gemm_axes(),
}


def _identifier(value: Mapping[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def dynamic_config_to_dict(config: NVFP4DynamicConfig) -> dict[str, object]:
    return asdict(config)


def dynamic_config_from_dict(
    value: Mapping[str, object],
) -> NVFP4DynamicConfig:
    return NVFP4DynamicConfig(
        quant=NVFP4QuantConfig(**dict(value["quant"])),  # type: ignore[arg-type]
        gemm=NVFP4GemmConfig(**dict(value["gemm"])),  # type: ignore[arg-type]
        quant_launches=str(value.get("quant_launches", "dual")),
        l2_fetch_granularity=(
            None
            if value.get("l2_fetch_granularity") is None
            else int(value["l2_fetch_granularity"])
        ),
    )


def dynamic_config_id(config: NVFP4DynamicConfig) -> str:
    return _identifier(dynamic_config_to_dict(config))


def update_dynamic_config(
    config: NVFP4DynamicConfig,
    updates: Mapping[str, object],
) -> NVFP4DynamicConfig:
    quant = asdict(config.quant)
    gemm = asdict(config.gemm)
    quant.update(dict(updates.get("quant", {})))  # type: ignore[arg-type]
    gemm.update(dict(updates.get("gemm", {})))  # type: ignore[arg-type]
    return NVFP4DynamicConfig(
        quant=NVFP4QuantConfig(**quant),
        gemm=NVFP4GemmConfig(**gemm),
        quant_launches=str(updates.get("quant_launches", config.quant_launches)),
        l2_fetch_granularity=updates.get(  # type: ignore[arg-type]
            "l2_fetch_granularity", config.l2_fetch_granularity
        ),
    )


def weight_prequant_config_to_dict(
    config: NVFP4WeightPrequantConfig,
) -> dict[str, object]:
    return asdict(config)


def weight_prequant_config_from_dict(
    value: Mapping[str, object],
) -> NVFP4WeightPrequantConfig:
    return NVFP4WeightPrequantConfig(
        quant_x=NVFP4QuantConfig(**dict(value["quant_x"])),  # type: ignore[arg-type]
        gemm=NVFP4GemmConfig(**dict(value["gemm"])),  # type: ignore[arg-type]
        l2_fetch_granularity=(
            None
            if value.get("l2_fetch_granularity") is None
            else int(value["l2_fetch_granularity"])
        ),
    )


def weight_prequant_config_id(config: NVFP4WeightPrequantConfig) -> str:
    return _identifier(weight_prequant_config_to_dict(config))


def update_weight_prequant_config(
    config: NVFP4WeightPrequantConfig,
    updates: Mapping[str, object],
) -> NVFP4WeightPrequantConfig:
    quant = asdict(config.quant_x)
    gemm = asdict(config.gemm)
    quant.update(dict(updates.get("quant", {})))  # type: ignore[arg-type]
    gemm.update(dict(updates.get("gemm", {})))  # type: ignore[arg-type]
    return NVFP4WeightPrequantConfig(
        quant_x=NVFP4QuantConfig(**quant),
        gemm=NVFP4GemmConfig(**gemm),
        l2_fetch_granularity=updates.get(  # type: ignore[arg-type]
            "l2_fetch_granularity", config.l2_fetch_granularity
        ),
    )


def fully_prequant_config_to_dict(
    config: NVFP4FullyPrequantConfig,
) -> dict[str, object]:
    return asdict(config)


def fully_prequant_config_from_dict(
    value: Mapping[str, object],
) -> NVFP4FullyPrequantConfig:
    return NVFP4FullyPrequantConfig(
        gemm=NVFP4GemmConfig(**dict(value["gemm"])),  # type: ignore[arg-type]
        l2_fetch_granularity=(
            None
            if value.get("l2_fetch_granularity") is None
            else int(value["l2_fetch_granularity"])
        ),
    )


def fully_prequant_config_id(config: NVFP4FullyPrequantConfig) -> str:
    return _identifier(fully_prequant_config_to_dict(config))


def update_fully_prequant_config(
    config: NVFP4FullyPrequantConfig,
    updates: Mapping[str, object],
) -> NVFP4FullyPrequantConfig:
    gemm = asdict(config.gemm)
    gemm.update(dict(updates.get("gemm", {})))  # type: ignore[arg-type]
    return NVFP4FullyPrequantConfig(
        gemm=NVFP4GemmConfig(**gemm),
        l2_fetch_granularity=updates.get(  # type: ignore[arg-type]
            "l2_fetch_granularity", config.l2_fetch_granularity
        ),
    )


def tune_nvfp4_inference_state(
    problem,
    *,
    state: str,
    device="cuda",
    cache_dir: Path | str | None = None,
    policy=None,
    progress=print,
):
    """Tune one NVFP4 materialized operand state and publish its winner.

    ``state`` is one of ``dynamic``, ``weight_prequantized``, or
    ``fully_prequantized``. Packed public operands currently use the canonical
    row-major TorchAO-compatible representation; physical tensor-core scale
    layouts remain an implementation coordinate inside the dynamic family.
    """

    from .autotune import (
        CalibratedPrequantEvaluator,
        DeviceFingerprint,
        HybridTuningPolicy,
        JsonlTuningStore,
        make_hybrid_autotuner,
        make_nvfp4_dynamic_adapter,
        make_nvfp4_fully_prequant_adapter,
        make_nvfp4_weight_prequant_adapter,
    )
    from .autotune.legacy import default_cache_dir
    from .autotune.winners import runtime_winner_key, save_runtime_winner
    from .nvfp4_inference_experiments import (
        NVFP4DynamicBenchmarkHarness,
        NVFP4FullyPrequantBenchmarkHarness,
        NVFP4WeightPrequantBenchmarkHarness,
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
    common = dict(
        regime="hot",
        protocol=protocol,
        device=device,
        seed=20260811,
    )
    if state == "dynamic":
        harness = NVFP4DynamicBenchmarkHarness(shape, **common)
        adapter_factory = make_nvfp4_dynamic_adapter
        initial = DEFAULT_NVFP4_DYNAMIC_CONFIG
        variant = "default"
    elif state == "weight_prequantized":
        harness = NVFP4WeightPrequantBenchmarkHarness(shape, **common)
        adapter_factory = make_nvfp4_weight_prequant_adapter
        initial = NVFP4WeightPrequantConfig()
        variant = "w-row_major"
    elif state == "fully_prequantized":
        harness = NVFP4FullyPrequantBenchmarkHarness(shape, **common)
        adapter_factory = make_nvfp4_fully_prequant_adapter
        initial = NVFP4FullyPrequantConfig()
        variant = "x-row_major_w-row_major"
    else:
        raise ValueError(f"unsupported NVFP4 inference state {state!r}")

    evaluator = CalibratedPrequantEvaluator(
        harness, samples=samples, seed=20260811
    )
    adapter = adapter_factory(
        problem,
        evaluator,
        initial=initial,
        device=fingerprint,
        regime="hot",
    )
    if isinstance(policy, HybridTuningPolicy):
        hybrid = policy
    else:
        hybrid = HybridTuningPolicy(
            max_trials=int(getattr(policy, "max_trials", 512)),
            time_budget_s=float(
                getattr(
                    policy,
                    "time_budget_s",
                    os.getenv(
                        "RTX_NVFP4_AUTOTUNE_SECONDS",
                        os.getenv("RTX_AUTOTUNE_SECONDS", "1800"),
                    ),
                )
            ),
            seed=int(getattr(policy, "seed", 20260811)),
        )
    suffix = f"m{problem.m}_n{problem.n}_k{problem.k}"
    if variant != "default":
        suffix += f"_{variant}"
    store = JsonlTuningStore(
        root / "runtime_sessions" / adapter.context.family / fingerprint.identifier / suffix
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
    "NVFP4_DYNAMIC_KERNEL_REVISION",
    "NVFP4_DYNAMIC_SEARCH_SPACE",
    "NVFP4_FULLY_PREQUANT_SEARCH_SPACE",
    "NVFP4_INFERENCE_KERNEL_REVISION",
    "NVFP4_WEIGHT_PREQUANT_SEARCH_SPACE",
    "dynamic_config_from_dict",
    "dynamic_config_id",
    "dynamic_config_to_dict",
    "fully_prequant_config_from_dict",
    "fully_prequant_config_id",
    "fully_prequant_config_to_dict",
    "preferred_dynamic_config",
    "update_fully_prequant_config",
    "update_dynamic_config",
    "update_weight_prequant_config",
    "weight_prequant_config_from_dict",
    "weight_prequant_config_id",
    "weight_prequant_config_to_dict",
    "tune_nvfp4_inference_state",
]
