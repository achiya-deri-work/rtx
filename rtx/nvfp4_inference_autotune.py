"""Composable search spaces for persistent NVFP4 inference states."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from typing import Mapping

from .configs.nvfp4 import (
    NVFP4FullyPrequantConfig,
    NVFP4GemmConfig,
    NVFP4QuantConfig,
    NVFP4WeightPrequantConfig,
)
from .prequant_autotune import PREQUANT_SEARCH_SPACE


NVFP4_INFERENCE_KERNEL_REVISION = 1


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
    return {name: PREQUANT_SEARCH_SPACE[name] for name in names}


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

NVFP4_WEIGHT_PREQUANT_SEARCH_SPACE = {
    "x_vector_load": _QUANT_VECTOR,
    "x_reduction": _QUANT_REDUCTION,
    "x_launch": _QUANT_LAUNCH,
    "x_registers": _QUANT_REGISTERS,
    **_gemm_axes(),
}
NVFP4_FULLY_PREQUANT_SEARCH_SPACE = _gemm_axes()


def _identifier(value: Mapping[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


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
    "NVFP4_FULLY_PREQUANT_SEARCH_SPACE",
    "NVFP4_INFERENCE_KERNEL_REVISION",
    "NVFP4_WEIGHT_PREQUANT_SEARCH_SPACE",
    "fully_prequant_config_from_dict",
    "fully_prequant_config_id",
    "fully_prequant_config_to_dict",
    "update_fully_prequant_config",
    "update_weight_prequant_config",
    "weight_prequant_config_from_dict",
    "weight_prequant_config_id",
    "weight_prequant_config_to_dict",
]
