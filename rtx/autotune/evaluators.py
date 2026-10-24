"""Adapters for statistically calibrated benchmark harnesses."""

from __future__ import annotations

from .outcomes import TrialOutcome


class CalibratedPrequantEvaluator:
    """Convert the rigorous prequant harness result into a generic trial outcome."""

    def __init__(
        self,
        harness,
        *,
        samples: int,
        seed: int = 0,
        components: bool = False,
    ) -> None:
        self.harness = harness
        self.problem = harness.problem
        self.samples = samples
        self.seed = seed
        self.components = components
        self.evaluated = 0

    def __call__(self, config, *, validate_only=False) -> TrialOutcome:
        if validate_only:
            from ..prequant_experiments import CandidateCompileError, CandidateCorrectnessError
            try:
                prepared=self.harness.prepare(config)
            except Exception as exc:
                status=('compile_error' if isinstance(exc,CandidateCompileError) else
                        'correctness_error' if isinstance(exc,CandidateCorrectnessError) else 'runtime_error')
                return TrialOutcome(status,error=f'{type(exc).__name__}: {exc}'[:4000])
            return TrialOutcome('ok',compile_ms=prepared.compile_ms,
                                max_abs_error=prepared.max_abs_error,
                                metadata={'validation_only':True,'oracle_inputs':'harness'})
        result = self.harness.measure(
            config,
            samples=self.samples,
            seed=self.seed + self.evaluated * 104729,
            components=self.components,
        )
        self.evaluated += 1
        status = str(result.get("status", "runtime_error"))
        summary = result.get("summary_ms")
        median_ms = None
        if isinstance(summary, dict) and summary.get("median") is not None:
            median_ms = float(summary["median"])
        standard = {
            "status",
            "summary_ms",
            "timings_ms",
            "compile_ms",
            "max_abs_error",
            "error",
        }
        return TrialOutcome(
            status,  # type: ignore[arg-type]
            median_ms=median_ms,
            timings_ms=[float(value) for value in result.get("timings_ms", [])],
            compile_ms=(
                None
                if result.get("compile_ms") is None
                else float(result["compile_ms"])
            ),
            max_abs_error=(
                None
                if result.get("max_abs_error") is None
                else float(result["max_abs_error"])
            ),
            error=(
                None if result.get("error") is None else str(result["error"])
            ),
            metadata={
                str(key): value
                for key, value in result.items()
                if key not in standard
            },
        )


class CalibratedBwdEvaluator(CalibratedPrequantEvaluator):
    """Connect the calibrated backward harness to the generic engine."""

    pass


__all__ = ["CalibratedBwdEvaluator", "CalibratedPrequantEvaluator"]
