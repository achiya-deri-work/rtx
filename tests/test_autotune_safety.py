from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import tempfile
import unittest

from rtx.autotune import (
    DiscreteKernelAdapter,
    JsonlFailureLedger,
    KernelContext,
    SafetyAwareAdapter,
    TrialOutcome,
)


@dataclass(frozen=True)
class _Config:
    tile: int = 128


def _adapter(context: KernelContext, calls: list[str], status: str = "compile_error"):
    def evaluate(config: _Config) -> TrialOutcome:
        calls.append(f"{context.regime}:{config.tile}")
        if status == "ok":
            return TrialOutcome("ok", median_ms=1.0, timings_ms=[1.0])
        return TrialOutcome(status, error="synthetic deterministic failure")

    return DiscreteKernelAdapter(
        context=context,
        initial_config=_Config(),
        axes={"tile": (64, 128)},
        config_id_fn=lambda config: f"tile-{config.tile}",
        serialize_fn=asdict,
        deserialize_fn=lambda value: _Config(**value),
        update_fn=lambda config, _coordinate, value: replace(config, tile=int(value)),
        evaluator=evaluate,
        rejection_fn=lambda _config: None,
    )


class DeterministicFailureLedgerTests(unittest.TestCase):
    def _context(self, *, regime: str = "hot", m: int = 128) -> KernelContext:
        return KernelContext(
            family="toy_kernel",
            kernel_revision=7,
            workload={"m": m, "n": 512, "k": 512},
            device={
                "architecture": {"key": "sm120"},
                "compiler": {"cuda": "13.2", "cutlass": "4.3"},
            },
            regime=regime,
        )

    def test_exact_failure_reuses_across_regimes_but_not_workloads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "failures.jsonl"
            calls: list[str] = []
            hot = SafetyAwareAdapter(
                _adapter(self._context(regime="hot"), calls),
                JsonlFailureLedger(path, fsync=False),
            )
            first = hot.evaluate(_Config())
            self.assertEqual(first.status, "compile_error")
            self.assertEqual(calls, ["hot:128"])

            rotate = SafetyAwareAdapter(
                _adapter(self._context(regime="rotate"), calls),
                JsonlFailureLedger(path, fsync=False),
            )
            reused = rotate.evaluate(_Config())
            self.assertEqual(calls, ["hot:128"])
            self.assertTrue(reused.metadata["deterministic_failure_cache_hit"])

            other_workload = SafetyAwareAdapter(
                _adapter(self._context(regime="rotate", m=256), calls),
                JsonlFailureLedger(path, fsync=False),
            )
            other_workload.evaluate(_Config())
            self.assertEqual(calls[-1], "rotate:128")
            self.assertEqual(len(calls), 2)

    def test_successes_are_never_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "failures.jsonl"
            calls: list[str] = []
            adapter = SafetyAwareAdapter(
                _adapter(self._context(), calls, status="ok"),
                JsonlFailureLedger(path, fsync=False),
            )
            adapter.evaluate(_Config())
            adapter.evaluate(_Config())
            self.assertEqual(len(calls), 2)

    def test_malformed_tail_isolated_before_append(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "failures.jsonl"
            path.write_bytes(b'{"truncated":')
            calls: list[str] = []
            adapter = SafetyAwareAdapter(
                _adapter(self._context(), calls),
                JsonlFailureLedger(path, fsync=False),
            )
            adapter.evaluate(_Config())
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[0], '{"truncated":')
            cached = JsonlFailureLedger(path, fsync=False).lookup(
                adapter.context, adapter.config_id(_Config())
            )
            self.assertIsNotNone(cached)


if __name__ == "__main__":
    unittest.main()
