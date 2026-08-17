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


NVFP4_INFERENCE_KERNEL_REVISION = 4
NVFP4_DYNAMIC_KERNEL_REVISION = 7
NVFP4_DELAYED_KERNEL_REVISION = 2
NVFP4_JIT_ROW_REGION_KERNEL_REVISION = 9


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
NVFP4_DELAYED_SEARCH_SPACE = {
    name: values
    for name, values in NVFP4_DYNAMIC_SEARCH_SPACE.items()
    if name != "quant_launches"
}


def preferred_jit_row_region_config(
    problem: NVFP4Problem,
) -> NVFP4DynamicConfig:
    """Portable seed for current local outer scales plus native NVFP4 GEMM."""

    base = preferred_dynamic_config(problem)
    use_regional_tma = problem.k >= 256
    # Dense cross-shape sweeps found that the portable optimum is asymmetric:
    # keep a smaller weight region while amortizing each weight chunk across
    # more activation rows.  Divisibility is not required; the observer masks
    # the final partial region.
    x_rows = (
        min(problem.m, 5)
        if use_regional_tma
        else 1
    )
    weight_rows = (
        min(problem.n, 4)
        if use_regional_tma
        else 1
    )
    candidate = replace(
        base,
        gemm=replace(
            base.gemm,
            epilogue=("tma" if use_regional_tma else "direct"),
            epilogue_stages=1,
            store_vec=(base.gemm.store_vec if use_regional_tma else 1),
            raster="n",
            grid_swizzle=2,
            tile_locality="same_b",
            regional_scale_epilogue=(
                "fragment_registers" if use_regional_tma else "direct"
            ),
            regional_epilogue_schedule="mma",
            regional_epilogue_warps=8,
            regional_epilogue_registers=48,
            regional_epilogue_values=(2 if use_regional_tma else 1),
            tiles_per_cta=4,
            persistent_waves=1,
        ),
        quant_launches="dual",
        x_scale_region_rows=x_rows,
        weight_scale_region_rows=weight_rows,
        tensor_scale_mode="power2",
        region_amax_load_bits=128,
        region_amax_unroll=1,
        region_waves=4,
        region_order="x_first",
        region_ownership=("cta_cached" if use_regional_tma else "warp"),
        programmatic_dependent_launch=False,
        regional_rescale_values=8,
        regional_rescale_warps=16,
        regional_rescale_waves=8,
    )
    if candidate.rejection(problem) is None:
        return candidate
    candidate = replace(candidate, region_ownership="cta")
    if candidate.rejection(problem) is None:
        return candidate
    # Ragged output tiles cannot use the current TMA epilogue, but that does
    # not invalidate current regional observation. Preserve the asymmetric
    # geometry and fall back only the store schedule.
    direct_candidate = replace(
        candidate,
        gemm=replace(
            candidate.gemm,
            epilogue="direct",
            store_vec=1,
            regional_scale_epilogue="direct",
            regional_epilogue_values=1,
        ),
    )
    if direct_candidate.rejection(problem) is None:
        return direct_candidate
    fallback = replace(
        DEFAULT_NVFP4_DYNAMIC_CONFIG,
        x_scale_region_rows=1,
        weight_scale_region_rows=1,
    )
    return fallback


_JIT_REGION_IMPLEMENTATION_ANCHORS = tuple(
    asdict(
        replace(
            _NATIVE_DYNAMIC_ANCHOR,
            gemm=replace(
                _NATIVE_DYNAMIC_ANCHOR.gemm,
                epilogue="direct",
                epilogue_stages=1,
                store_vec=1,
                persistent_waves=gemm_waves,
            ),
            quant_launches="dual",
            x_scale_region_rows=x_rows,
            weight_scale_region_rows=w_rows,
            tensor_scale_mode="power2",
            region_waves=region_waves,
        )
    )
    for x_rows, w_rows, region_waves, gemm_waves in (
        (1, 1, 4, 1),
        (4, 1, 4, 1),
        (16, 1, 6, 1),
        (32, 4, 6, 3),
        (64, 8, 8, 3),
    )
)

# Seed the proven regional TMA epilogue directly. Fine one-row scaling plus a
# direct epilogue made scale lookup dominate MMA and trapped short campaigns
# in a local minimum even though JIT observation itself was competitive.
_JIT_REGION_IMPLEMENTATION_ANCHORS += tuple(
    asdict(
        replace(
            _NATIVE_DYNAMIC_ANCHOR,
            gemm=replace(
                _NATIVE_DYNAMIC_ANCHOR.gemm,
                epilogue="tma",
                epilogue_stages=epilogue_stages,
                producer_registers=24,
                maxrregcount=192,
                raster="n",
                grid_swizzle=2,
                tile_locality="same_b",
                tiles_per_cta=4,
                persistent_waves=gemm_waves,
            ),
            quant_launches="dual",
            x_scale_region_rows=x_rows,
            weight_scale_region_rows=w_rows,
            tensor_scale_mode="power2",
            region_waves=region_waves,
            region_ownership="cta",
            programmatic_dependent_launch=use_pdl,
        )
    )
    for x_rows, w_rows, region_waves, gemm_waves, epilogue_stages, use_pdl in (
        (5, 4, 4, 1, 1, False),
        (5, 4, 4, 1, 1, True),
        (4, 5, 4, 1, 1, False),
        (8, 4, 4, 1, 1, False),
        (8, 8, 4, 1, 1, False),
        (16, 8, 4, 1, 1, False),
        (16, 16, 6, 3, 2, False),
        (32, 8, 8, 3, 1, False),
        (16, 8, 4, 1, 1, True),
    )
)

# The SM120 128x128, atom-N=2 accumulator fragment repeats two adjacent
# columns for each of two rows. Hoist its two X factors once per thread and
# one W factor per four values, then reuse the two exact FP32 products before
# the final BF16 conversion. Keep the direct anchors above as independent
# basins; layout, geometry, and register pressure can reverse small wins on a
# different SKU or shape.
_JIT_REGION_IMPLEMENTATION_ANCHORS += tuple(
    asdict(
        replace(
            _NATIVE_DYNAMIC_ANCHOR,
            gemm=replace(
                _NATIVE_DYNAMIC_ANCHOR.gemm,
                epilogue="tma",
                epilogue_stages=1,
                producer_registers=24,
                maxrregcount=192,
                raster="n",
                grid_swizzle=2,
                tile_locality="same_b",
                tiles_per_cta=4,
                persistent_waves=1,
                regional_scale_epilogue="fragment_registers",
                regional_epilogue_values=2,
            ),
            quant_launches="dual",
            x_scale_region_rows=x_rows,
            weight_scale_region_rows=w_rows,
            tensor_scale_mode="power2",
            region_waves=region_waves,
            region_ownership="cta_cached",
            programmatic_dependent_launch=False,
        )
    )
    for x_rows, w_rows, region_waves in (
        (5, 4, 4),
        (8, 4, 6),
        (8, 8, 6),
    )
)

# True current-scale quantization normally reads each BF16 region once for
# amax and again for quantization. Register-cached CTA anchors retain each
# thread's quantization fragment across the reduction, preserving identical
# FP32 scale math while removing the extra global/L2 operand pass.
_JIT_REGION_IMPLEMENTATION_ANCHORS += tuple(
    asdict(
        replace(
            _NATIVE_DYNAMIC_ANCHOR,
            gemm=replace(
                _NATIVE_DYNAMIC_ANCHOR.gemm,
                epilogue="tma",
                epilogue_stages=1,
                producer_registers=24,
                maxrregcount=192,
                raster="n",
                grid_swizzle=2,
                tile_locality="same_b",
                tiles_per_cta=4,
                persistent_waves=1,
            ),
            quant_launches="dual",
            x_scale_region_rows=x_rows,
            weight_scale_region_rows=w_rows,
            tensor_scale_mode="power2",
            region_waves=region_waves,
            region_ownership=ownership,
            programmatic_dependent_launch=False,
        )
    )
    for x_rows, w_rows, region_waves, ownership in (
        (4, 5, 4, "cta_cached"),
        (5, 4, 4, "cta_cached"),
        (8, 4, 6, "cta_cached"),
        (8, 8, 6, "cta_cached"),
    )
)

# Dedicated CUDA-core epilogue warps consume one FP32 SMEM handoff tile while
# the tensor-core warps accumulate the next persistent output tile.  A single
# operand stage is intentional: it leaves enough SM120 shared memory for the
# 64-KiB 128x128 accumulator exchange without narrowing the MMA tile.
_JIT_REGION_IMPLEMENTATION_ANCHORS += tuple(
    asdict(
        replace(
            _NATIVE_DYNAMIC_ANCHOR,
            gemm=replace(
                _NATIVE_DYNAMIC_ANCHOR.gemm,
                stages=1,
                epilogue="direct",
                epilogue_stages=1,
                store_vec=1,
                regional_scale_epilogue="product",
                regional_epilogue_schedule="warp_specialized",
                atom_layout_m=4,
                regional_epilogue_warps=epilogue_warps,
                regional_epilogue_registers=epilogue_registers,
                regional_epilogue_values=4,
                tiles_per_cta=tiles_per_cta,
                tile_locality="same_b",
                persistent_waves=(0 if epilogue_warps == 8 else gemm_waves),
            ),
            quant_launches="dual",
            x_scale_region_rows=5,
            weight_scale_region_rows=4,
            tensor_scale_mode="power2",
            region_waves=4,
            region_ownership="cta",
            programmatic_dependent_launch=False,
        )
    )
    for epilogue_warps, epilogue_registers, tiles_per_cta, gemm_waves in (
        (1, 32, 2, 1),
        (2, 40, 2, 1),
        (4, 48, 2, 1),
        (8, 64, 2, 1),
        (4, 48, 4, 2),
    )
)

_jit_gemm_axes = _gemm_axes()
NVFP4_JIT_ROW_REGION_SEARCH_SPACE = {
    "implementation_anchor": _JIT_REGION_IMPLEMENTATION_ANCHORS,
    "x_region_rows": tuple(
        {"x_scale_region_rows": rows}
        for rows in (1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 256)
    ),
    "weight_region_rows": tuple(
        {"weight_scale_region_rows": rows}
        for rows in (1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128)
    ),
    "regional_scale_epilogue": tuple(
        {"gemm": {"regional_scale_epilogue": strategy}}
        for strategy in (
            "direct",
            "expanded_factors",
            "factorized",
            "fragment_registers",
            "product",
            "separate",
        )
    ),
    "regional_epilogue_schedule": tuple(
        {"gemm": {"regional_epilogue_schedule": schedule}}
        for schedule in ("mma", "warp_specialized")
    ),
    "regional_epilogue_warps": tuple(
        {"gemm": {"regional_epilogue_warps": warps}}
        for warps in (1, 2, 4, 8)
    ),
    "regional_epilogue_registers": tuple(
        {"gemm": {"regional_epilogue_registers": registers}}
        for registers in (24, 32, 40, 48, 64, 80)
    ),
    "regional_epilogue_values": tuple(
        {"gemm": {"regional_epilogue_values": values}}
        for values in (1, 2, 4, 8)
    ),
    "regional_rescale_values": tuple(
        {"regional_rescale_values": values} for values in (1, 2, 4, 8, 16)
    ),
    "regional_rescale_warps": tuple(
        {"regional_rescale_warps": warps} for warps in (4, 8, 16)
    ),
    "regional_rescale_waves": tuple(
        {"regional_rescale_waves": waves} for waves in (1, 2, 3, 4, 6, 8)
    ),
    # Stratifying the dense neighborhood by X size prevents a bandit or an
    # interrupted campaign from sampling only the diagonal.  These coordinates
    # expose all 49 asymmetric 2..8 pairs early while the independent axes
    # above retain long-range exploration.
    **{
        f"fine_region_geometry_x{x_rows}": tuple(
            {
                "x_scale_region_rows": x_rows,
                "weight_scale_region_rows": weight_rows,
            }
            for weight_rows in range(2, 9)
        )
        for x_rows in range(2, 9)
    },
    "outer_scale_math": tuple(
        {"tensor_scale_mode": mode} for mode in ("power2", "exact")
    ),
    "region_amax_load": tuple(
        {"region_amax_load_bits": bits} for bits in (16, 32, 64, 128)
    ),
    "region_amax_unroll": tuple(
        {"region_amax_unroll": unroll} for unroll in (1, 2, 4, 8)
    ),
    "region_grid": tuple(
        {"region_waves": waves} for waves in (1, 2, 3, 4, 6, 8)
    ),
    "region_order": tuple(
        {"region_order": order} for order in ("x_first", "weight_first")
    ),
    "region_ownership": tuple(
        {"region_ownership": owner}
        for owner in ("warp", "cta", "cta_cached")
    ),
    "dependent_launch": tuple(
        {"programmatic_dependent_launch": enabled}
        for enabled in (False, True)
    ),
    "quant_vector_load": _QUANT_VECTOR,
    "quant_reduction": _QUANT_REDUCTION,
    "quant_launch": _QUANT_LAUNCH,
    "quant_registers": _QUANT_REGISTERS,
    "quant_scale_reciprocal": _QUANT_RECIPROCAL,
    "quant_scale_compute": _QUANT_SCALE_COMPUTE,
    "scale_transport": NVFP4_DYNAMIC_SEARCH_SPACE["scale_transport"],
    **_jit_gemm_axes,
}



def _identifier(value: Mapping[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def dynamic_config_to_dict(config: NVFP4DynamicConfig) -> dict[str, object]:
    value = asdict(config)
    if not config.jit_row_region:
        # Preserve the revision-6 dynamic and revision-1 delayed serialized
        # schema/config IDs. JIT region coordinates belong only to their new
        # family and must not invalidate already collected block/delayed data.
        for name in (
            "x_scale_region_rows",
            "weight_scale_region_rows",
            "tensor_scale_mode",
            "region_amax_load_bits",
            "region_amax_unroll",
            "region_waves",
            "region_order",
            "region_ownership",
            "programmatic_dependent_launch",
            "regional_rescale_values",
            "regional_rescale_warps",
            "regional_rescale_waves",
        ):
            value.pop(name)
    return value


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
        x_scale_region_rows=int(value.get("x_scale_region_rows", 0)),
        weight_scale_region_rows=int(value.get("weight_scale_region_rows", 0)),
        tensor_scale_mode=str(value.get("tensor_scale_mode", "power2")),
        region_amax_load_bits=int(value.get("region_amax_load_bits", 128)),
        region_amax_unroll=int(value.get("region_amax_unroll", 1)),
        region_waves=int(value.get("region_waves", 4)),
        region_order=str(value.get("region_order", "x_first")),
        region_ownership=str(value.get("region_ownership", "warp")),
        programmatic_dependent_launch=bool(
            value.get("programmatic_dependent_launch", False)
        ),
        regional_rescale_values=int(value.get("regional_rescale_values", 8)),
        regional_rescale_warps=int(value.get("regional_rescale_warps", 16)),
        regional_rescale_waves=int(value.get("regional_rescale_waves", 8)),
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
        x_scale_region_rows=int(
            updates.get("x_scale_region_rows", config.x_scale_region_rows)
        ),
        weight_scale_region_rows=int(
            updates.get(
                "weight_scale_region_rows", config.weight_scale_region_rows
            )
        ),
        tensor_scale_mode=str(
            updates.get("tensor_scale_mode", config.tensor_scale_mode)
        ),
        region_amax_load_bits=int(
            updates.get("region_amax_load_bits", config.region_amax_load_bits)
        ),
        region_amax_unroll=int(
            updates.get("region_amax_unroll", config.region_amax_unroll)
        ),
        region_waves=int(updates.get("region_waves", config.region_waves)),
        region_order=str(updates.get("region_order", config.region_order)),
        region_ownership=str(
            updates.get("region_ownership", config.region_ownership)
        ),
        programmatic_dependent_launch=bool(
            updates.get(
                "programmatic_dependent_launch",
                config.programmatic_dependent_launch,
            )
        ),
        regional_rescale_values=int(
            updates.get("regional_rescale_values", config.regional_rescale_values)
        ),
        regional_rescale_warps=int(
            updates.get("regional_rescale_warps", config.regional_rescale_warps)
        ),
        regional_rescale_waves=int(
            updates.get("regional_rescale_waves", config.regional_rescale_waves)
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
        make_nvfp4_jit_row_region_adapter,
        make_nvfp4_fully_prequant_adapter,
        make_nvfp4_weight_prequant_adapter,
    )
    from .autotune.legacy import default_cache_dir
    from .autotune.winners import runtime_winner_key, save_runtime_winner
    from .nvfp4_inference_experiments import (
        NVFP4DynamicBenchmarkHarness,
        NVFP4JITRowRegionBenchmarkHarness,
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
    elif state == "jit_row_region":
        harness = NVFP4JITRowRegionBenchmarkHarness(shape, **common)
        adapter_factory = make_nvfp4_jit_row_region_adapter
        initial = preferred_jit_row_region_config(problem)
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
    "NVFP4_DELAYED_KERNEL_REVISION",
    "NVFP4_DELAYED_SEARCH_SPACE",
    "NVFP4_FULLY_PREQUANT_SEARCH_SPACE",
    "NVFP4_INFERENCE_KERNEL_REVISION",
    "NVFP4_JIT_ROW_REGION_KERNEL_REVISION",
    "NVFP4_JIT_ROW_REGION_SEARCH_SPACE",
    "NVFP4_WEIGHT_PREQUANT_SEARCH_SPACE",
    "dynamic_config_from_dict",
    "dynamic_config_id",
    "dynamic_config_to_dict",
    "fully_prequant_config_from_dict",
    "fully_prequant_config_id",
    "fully_prequant_config_to_dict",
    "preferred_dynamic_config",
    "preferred_jit_row_region_config",
    "update_fully_prequant_config",
    "update_dynamic_config",
    "update_weight_prequant_config",
    "weight_prequant_config_from_dict",
    "weight_prequant_config_id",
    "weight_prequant_config_to_dict",
    "tune_nvfp4_inference_state",
]
