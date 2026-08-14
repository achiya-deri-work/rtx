"""Stable, side-effect-free descriptions of linear runtime selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping


SelectionSource = Literal["explicit", "portable", "deferred_runtime"]


@dataclass(frozen=True, slots=True)
class LinearExecutionDecision:
    """Explain how a linear call will be routed without compiling or tuning.

    ``deferred_runtime`` means exact cache lookup or online tuning happens in
    the registered launcher after device/shape guards are known. Consequently
    ``config`` describes explicit constraints rather than claiming a winner
    that has not yet been resolved.
    """

    format: Literal["mxfp8", "nvfp4"]
    operand_state: Literal[
        "dynamic", "weight_prequantized", "fully_prequantized"
    ]
    problem: tuple[int, int, int]
    backend: Literal["fused", "materialized"]
    family: str
    selection_source: SelectionSource
    autotune: Literal["off", "cache", "coordinate"]
    scaling: str | None = None
    x_scale_region_rows: int | None = None
    weight_scale_region_rows: int | None = None
    config: Mapping[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()


__all__ = ["LinearExecutionDecision", "SelectionSource"]
