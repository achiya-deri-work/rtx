"""Composable search spaces for persistent NVFP4 inference states."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from typing import Mapping

from .configs.nvfp4 import (
    NVFP4DynamicConfig,
    NVFP4FullyPrequantConfig,
    NVFP4GemmConfig,
    NVFP4QuantConfig,
    NVFP4WeightPrequantConfig,
)
from .prequant_autotune import PREQUANT_SEARCH_SPACE


NVFP4_INFERENCE_KERNEL_REVISION = 1
NVFP4_DYNAMIC_KERNEL_REVISION = 3


def _gemm_axes() -> dict[str, tuple[dict[str, object], ...]]:
    names = (
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

# A compound implementation anchor keeps the tuner from having to rediscover
# a known-good materialized-GEMM basin one independent coordinate at a time.
# It is deliberately a seed, not a hard-coded winner: every field remains
# mutable through the ordinary axes below, and device/shape-specific results
# still have to win measurement and confirmation.
_DYNAMIC_IMPLEMENTATION_ANCHORS = (
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
    "update_fully_prequant_config",
    "update_dynamic_config",
    "update_weight_prequant_config",
    "weight_prequant_config_from_dict",
    "weight_prequant_config_id",
    "weight_prequant_config_to_dict",
]
