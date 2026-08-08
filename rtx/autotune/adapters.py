"""Adapters connecting every current MXFP8 tuning family to the shared engine."""

from __future__ import annotations

import math
from typing import Callable, Iterable, Mapping

from .core import DiscreteKernelAdapter, KernelContext
from .legacy import DeviceFingerprint, TrialOutcome
from ..kernels.mxfp8 import (
    DEFAULT_MXFP8_FWD_CONFIG,
    FWD_SEARCH_SPACE,
    MXFP8_FWD_KERNEL_REVISION,
    MXFP8FwdConfig,
    MXFP8Problem,
    fwd_config_from_dict,
    fwd_config_id,
    fwd_config_to_dict,
    normalize_fwd_config,
)


def _device_dict(
    device: DeviceFingerprint | Mapping[str, object] | None,
) -> Mapping[str, object]:
    if device is None:
        return {}
    return device.as_dict() if isinstance(device, DeviceFingerprint) else dict(device)


def _context(
    family: str,
    revision: int,
    problem: MXFP8Problem,
    device: DeviceFingerprint | Mapping[str, object] | None,
    regime: str,
    tags: Mapping[str, object] | None,
) -> KernelContext:
    return KernelContext(
        family=family,
        kernel_revision=revision,
        workload={"m": problem.m, "n": problem.n, "k": problem.k},
        device=_device_dict(device),
        regime=regime,
        tags={} if tags is None else dict(tags),
    )


def _geometry_features(
    problem: MXFP8Problem,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    device: DeviceFingerprint | Mapping[str, object] | None,
) -> dict[str, float]:
    device_values = _device_dict(device)
    sm_count = max(1, int(device_values.get("multiprocessor_count", 1)))
    ctas = math.ceil(problem.m / tile_m) * math.ceil(problem.n / tile_n)
    return {
        "m_tiles": math.ceil(problem.m / tile_m),
        "n_tiles": math.ceil(problem.n / tile_n),
        "k_tiles": math.ceil(problem.k / tile_k),
        "ctas": ctas,
        "cta_waves": ctas / sm_count,
        "m_tail_fraction": (problem.m % tile_m) / tile_m,
        "n_tail_fraction": (problem.n % tile_n) / tile_n,
        "k_tail_fraction": (problem.k % tile_k) / tile_k,
        "aspect_mn": problem.m / max(1, problem.n),
        "aspect_km": problem.k / max(1, problem.m),
        "aspect_kn": problem.k / max(1, problem.n),
    }


def make_mxfp8_fwd_adapter(
    problem: MXFP8Problem,
    evaluator: Callable[[MXFP8FwdConfig], TrialOutcome],
    *,
    initial: MXFP8FwdConfig = DEFAULT_MXFP8_FWD_CONFIG,
    axes: Mapping[str, Iterable[object]] = FWD_SEARCH_SPACE,
    device: DeviceFingerprint | Mapping[str, object] | None = None,
    regime: str = "hot",
    tags: Mapping[str, object] | None = None,
) -> DiscreteKernelAdapter[MXFP8FwdConfig]:
    axis_values = {name: tuple(values) for name, values in axes.items()}
    unknown = set(axis_values).difference(FWD_SEARCH_SPACE)
    if unknown:
        raise ValueError(f"unknown fused-forward tuning axes: {sorted(unknown)}")

    def rejection(config: MXFP8FwdConfig) -> tuple[str, str] | None:
        reason = config.architecture_rejection(problem)
        if reason is not None:
            return "architecture_rejected", reason
        reason = config.implementation_rejection(problem)
        return None if reason is None else ("implementation_rejected", reason)

    return DiscreteKernelAdapter(
        context=_context(
            "mxfp8_fused_fwd",
            MXFP8_FWD_KERNEL_REVISION,
            problem,
            device,
            regime,
            tags,
        ),
        initial_config=initial,
        axes=axis_values,
        config_id_fn=fwd_config_id,
        serialize_fn=fwd_config_to_dict,
        deserialize_fn=lambda value: fwd_config_from_dict(dict(value)),
        update_fn=lambda config, coordinate, value: normalize_fwd_config(
            config, **{coordinate: value}
        ),
        evaluator=evaluator,
        rejection_fn=rejection,
        extra_features_fn=lambda config: _geometry_features(
            problem, config.tile_m, config.tile_n, config.tile_k, device
        ),
    )


def make_mxfp8_prequant_adapter(
    problem: MXFP8Problem,
    evaluator: Callable[[object], TrialOutcome],
    *,
    initial: object | None = None,
    axes: Mapping[str, Iterable[Mapping[str, object]]] | None = None,
    device: DeviceFingerprint | Mapping[str, object] | None = None,
    regime: str = "hot",
    tags: Mapping[str, object] | None = None,
) -> DiscreteKernelAdapter[object]:
    from ..fp8 import DEFAULT_MXFP8_PREQUANT_CONFIG
    from ..prequant_autotune import (
        KERNEL_REVISION,
        PREQUANT_SEARCH_SPACE,
        prequant_config_from_dict,
        prequant_config_id,
        prequant_config_to_dict,
        update_prequant_config,
    )

    initial_config = DEFAULT_MXFP8_PREQUANT_CONFIG if initial is None else initial
    selected_axes = PREQUANT_SEARCH_SPACE if axes is None else axes
    axis_values = {name: tuple(values) for name, values in selected_axes.items()}
    unknown = set(axis_values).difference(PREQUANT_SEARCH_SPACE)
    if unknown:
        raise ValueError(f"unknown prequant-forward tuning axes: {sorted(unknown)}")

    def rejection(config: object) -> tuple[str, str] | None:
        reason = config.rejection(problem)  # type: ignore[attr-defined]
        return None if reason is None else ("implementation_rejected", reason)

    def derived(config: object) -> Mapping[str, float]:
        gemm = config.gemm  # type: ignore[attr-defined]
        return _geometry_features(
            problem, gemm.tile_m, gemm.tile_n, gemm.tile_k, device
        )

    return DiscreteKernelAdapter(
        context=_context(
            "mxfp8_prequant_fwd", KERNEL_REVISION, problem, device, regime, tags
        ),
        initial_config=initial_config,
        axes=axis_values,
        config_id_fn=prequant_config_id,
        serialize_fn=prequant_config_to_dict,
        deserialize_fn=prequant_config_from_dict,
        update_fn=lambda config, _coordinate, value: update_prequant_config(config, value),
        evaluator=evaluator,
        rejection_fn=rejection,
        extra_features_fn=derived,
    )


def make_mxfp8_bwd_adapter(
    problem: MXFP8Problem,
    evaluator: Callable[[object], TrialOutcome],
    *,
    initial: object | None = None,
    axes: Mapping[str, Iterable[Mapping[str, object]]] | None = None,
    device: DeviceFingerprint | Mapping[str, object] | None = None,
    regime: str = "hot",
    tags: Mapping[str, object] | None = None,
) -> DiscreteKernelAdapter[object]:
    from ..bwd_autotune import (
        BWD_SEARCH_SPACE,
        KERNEL_REVISION,
        bwd_config_from_dict,
        bwd_config_id,
        bwd_config_to_dict,
        update_bwd_config,
    )
    from ..kernels.mxfp8_bwd import DEFAULT_MXFP8_BWD_CONFIG

    initial_config = DEFAULT_MXFP8_BWD_CONFIG if initial is None else initial
    selected_axes = BWD_SEARCH_SPACE if axes is None else axes
    axis_values = {name: tuple(values) for name, values in selected_axes.items()}
    unknown = set(axis_values).difference(BWD_SEARCH_SPACE)
    if unknown:
        raise ValueError(f"unknown backward tuning axes: {sorted(unknown)}")

    def update(config: object, coordinate: str, value: object) -> object:
        current = config
        parts = coordinate.split("_")
        if len(parts) >= 3 and parts[0] in ("dx", "dw") and parts[1] == "b":
            matmul = current.dx if parts[0] == "dx" else current.dw  # type: ignore[attr-defined]
            if (
                matmul.quant_launches == "dual"
                and matmul.a_orientation == matmul.b_orientation
            ):
                current = update_bwd_config(
                    current, {parts[0]: {"quant_launches": "separate"}}
                )
        return update_bwd_config(current, value)  # type: ignore[arg-type]

    def rejection(config: object) -> tuple[str, str] | None:
        reason = config.implementation_rejection(problem)  # type: ignore[attr-defined]
        return None if reason is None else ("implementation_rejected", reason)

    def derived(config: object) -> Mapping[str, float]:
        dx = config.dx.gemm  # type: ignore[attr-defined]
        dw = config.dw.gemm  # type: ignore[attr-defined]
        values = {
            f"dx_{key}": value
            for key, value in _geometry_features(
                MXFP8Problem(problem.m, problem.k, problem.n),
                dx.tile_m,
                dx.tile_n,
                dx.tile_k,
                device,
            ).items()
        }
        values.update(
            {
                f"dw_{key}": value
                for key, value in _geometry_features(
                    MXFP8Problem(problem.n, problem.k, problem.m),
                    dw.tile_m,
                    dw.tile_n,
                    dw.tile_k,
                    device,
                ).items()
            }
        )
        values["dw_reduction_length"] = problem.m
        return values

    return DiscreteKernelAdapter(
        context=_context("mxfp8_bwd", KERNEL_REVISION, problem, device, regime, tags),
        initial_config=initial_config,
        axes=axis_values,
        config_id_fn=bwd_config_id,
        serialize_fn=bwd_config_to_dict,
        deserialize_fn=bwd_config_from_dict,
        update_fn=update,
        evaluator=evaluator,
        rejection_fn=rejection,
        extra_features_fn=derived,
    )


__all__ = [
    "make_mxfp8_bwd_adapter",
    "make_mxfp8_fwd_adapter",
    "make_mxfp8_prequant_adapter",
]
