"""MXFP8 numerical signatures for execution-plan selection.

Arithmetic choices are deliberately not inferred from performance features.
An implementation must match this contract before its timing is eligible.
"""


def execution_signature(candidate, phase="backward"):
    """Exclude provably inactive backend fields from candidate identity."""
    from dataclasses import asdict
    config = asdict(candidate.config)
    if "dx" not in config:
        return {"implementation": candidate.implementation, "config": config}
    if phase not in ("backward", "dx", "dw"):
        raise ValueError("invalid backward phase")
    for name in ("dx", "dw"):
        if phase != "backward" and phase != name:
            config.pop(name)
            continue
        leg = config[name]
        if leg["backend"] == "decomposed":
            leg.pop("fused")
        elif leg["backend"] == "fused":
            for key in ("gemm", "quant_a", "quant_b", "quant_launches"):
                leg.pop(key, None)
    return {"phase": phase, "config": config}


def execution_plan(candidate, phase="backward"):
    """Classify actual execution, not the registry seed that produced it."""
    config = candidate.config
    if not hasattr(config, "dx"):
        return candidate.implementation
    if phase not in ("backward", "dx", "dw"):
        raise ValueError("invalid backward phase")
    legs = ("dx", "dw") if phase == "backward" else (phase,)
    return tuple(getattr(config, leg).backend for leg in legs)


def reuse_prior(candidate, m, n, k, *, phase="forward", sm_count=0, l2_bytes=0):
    """Research-only work ordering, NOT predicted latency or measured HBM.

    Count BF16 values prepared and packed workspace round trips. Cache pressure
    and CTA underfill break ties only: no unsupported bandwidth/clock model is
    inferred from SKU names. Rejected for runtime deployment by the 44-row
    retrospective study: it ranks small-shape fused candidates too optimistically.
    """
    from .preparation import PreparationGeometry, linear_phase_geometry
    config = candidate.config
    phases = ("dx", "dw") if phase == "backward" else (phase,)
    work, underfill, pressure = 0., 0., 0.
    for leg_name in phases:
        dims = linear_phase_geometry(m, n, k, leg_name)
        if leg_name == "forward":
            staged = candidate.implementation == "materialized"
            schedule = config.gemm if staged else config
        else:
            leg = getattr(config, leg_name)
            staged = leg.backend == "decomposed"
            schedule = leg.gemm if staged else leg.fused
        geometry = PreparationGeometry(*dims, schedule.tile_m, schedule.tile_n,
                                       sm_count=sm_count, l2_bytes=l2_bytes)
        features = geometry.features("staged" if staged else "fused")
        source_bytes = 2 * (dims[0] + dims[1]) * dims[2]
        work += (2 * features["prepared_elements"] + features["logical_staged_rw_bytes"]) / source_bytes
        if sm_count:
            underfill += max(0., 1. - features["cta_waves"])
        pressure += features["workspace_l2_ratio"]
    return work, underfill, pressure


def quantization_policy(quantizer):
    return {"rounding": quantizer.scale_rounding, "math": quantizer.quant_math,
            "amax": quantizer.quant_amax}


def backward_leg_policy(leg):
    quantizers = ([leg.fused, leg.fused] if leg.backend == "fused"
                  else [leg.quant_a, leg.resolved_quant_b()])
    return {"reduction": leg.reduction, "split_reduction": leg.split_reduction,
            "orientations": [leg.a_orientation, leg.b_orientation],
            "quantizers": [quantization_policy(q) for q in quantizers]}


def numerical_policy(implementation, config):
    if hasattr(config, "dx"):
        return {"sources": "original_bf16", "precision": "mxfp8",
                "legs": {name: backward_leg_policy(getattr(config, name))
                         for name in ("dx", "dw")}}
    if implementation not in ("fused", "materialized"):
        raise ValueError("unsupported MXFP8 forward preparation implementation")
    quantizers = ([config, config] if implementation == "fused"
                  else [config.quant, config.resolved_weight_quant()])
    return {"precision": "mxfp8", "sources": "original_bf16", "accumulator": "fp32",
            "quantizers": [quantization_policy(q) for q in quantizers]}


def backward_plan_config(plan, stages=2):
    """Policy-matched starting points; child schedules remain autotunable."""
    from dataclasses import replace
    from ..kernels.mxfp8 import normalize_fwd_config
    from ..kernels.mxfp8_bwd import DEFAULT_DUAL_DECOMPOSED_MXFP8_BWD_CONFIG
    config = DEFAULT_DUAL_DECOMPOSED_MXFP8_BWD_CONFIG
    choices = {"staged": (False, False), "fused": (True, True),
               "fused_dx": (True, False), "fused_dw": (False, True)}
    if plan not in choices:
        raise ValueError(f"unknown backward preparation plan: {plan}")
    legs = {}
    for name, fused in zip(("dx", "dw"), choices[plan]):
        leg = getattr(config, name)
        if fused:
            schedule = normalize_fwd_config(leg.fused,
                cta_reuse_tile=(128, 128, stages, 1, 232, 255))
            schedule = replace(schedule, quant_math="fp32", quant_amax="fp32")
            leg = replace(leg, backend="fused", fused=schedule)
        legs[name] = leg
    return replace(config, **legs)


def fused_forward_preparation_seeds():
    """Measured research seed order, not a rule selecting the winning backend.

    TMA two-stage beat scalar preparation on the four decoder shapes and the
    128x256x256 probe. All schedules still require caller legality, numerical
    policy checks and target-device measurement. Scalar remains a tail fallback.
    """
    from dataclasses import replace
    from ..kernels.mxfp8 import DEFAULT_MXFP8_FWD_CONFIG, normalize_fwd_config
    return tuple(
        replace(normalize_fwd_config(DEFAULT_MXFP8_FWD_CONFIG,
            cta_reuse_tile=(128,128,stages,1,232,255)),
            quant_math="fp32", quant_amax="fp32") for stages in (2,3,1)
    ) + (DEFAULT_MXFP8_FWD_CONFIG,)
