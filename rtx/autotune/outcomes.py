"""Backend-neutral trial outcome records shared by every tuning frontend."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, Mapping


TrialStatus = Literal[
    "ok",
    "architecture_rejected",
    "implementation_rejected",
    "compile_error",
    "correctness_error",
    "runtime_error",
]


@dataclass(slots=True)
class TrialOutcome:
    status: TrialStatus
    median_ms: float | None = None
    timings_ms: list[float] = field(default_factory=list)
    compile_ms: float | None = None
    max_abs_error: float | None = None
    error: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def successful(self) -> bool:
        return self.status == "ok" and self.median_ms is not None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> "TrialOutcome":
        return cls(
            status=str(values["status"]),  # type: ignore[arg-type]
            median_ms=(
                None
                if values.get("median_ms") is None
                else float(values["median_ms"])
            ),
            timings_ms=[
                float(value) for value in values.get("timings_ms", [])
            ],  # type: ignore[arg-type]
            compile_ms=(
                None
                if values.get("compile_ms") is None
                else float(values["compile_ms"])
            ),
            max_abs_error=(
                None
                if values.get("max_abs_error") is None
                else float(values["max_abs_error"])
            ),
            error=None if values.get("error") is None else str(values["error"]),
            metadata=dict(values.get("metadata", {})),  # type: ignore[arg-type]
        )


__all__ = ["TrialOutcome", "TrialStatus"]
