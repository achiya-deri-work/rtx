from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from rtx.autotune.core import Observation
from rtx.autotune.evidence import (
    PairwisePreferenceModel,
    build_evidence_study,
    failure_analysis,
    parent_move_analysis,
    timing_convergence_analysis,
)
from rtx.autotune.outcomes import TrialOutcome


def _row(
    *,
    sku: str,
    m: int,
    tile: int,
    sequence: int,
    status: str = "ok",
    parent: int | None = None,
) -> Observation[object]:
    context = f"{sku}-{m}"
    latency = m / 128 * (1.0 if tile == 128 else 1.2 + tile / 1000)
    config_id = f"tile-{tile}"
    return Observation(
        observation_id=f"{context}-{tile}-{status}",
        session_id="synthetic",
        sequence=sequence,
        context_id=context,
        family="toy_kernel",
        kernel_revision=1,
        config_id=config_id,
        config={"gemm": {"tile_m": tile, "stages": 3}},
        serialized_config={"gemm": {"tile_m": tile, "stages": 3}},
        features={
            f"context.device.sku.sku_family={sku}": 1.0,
            "context.device.multiprocessor_count": 40.0 if sku == "small" else 120.0,
            "context.device.calibration.measured_native_mxfp8_gemm_tflops": 100.0,
            "context.workload.m": float(m),
            "context.workload.n": 256.0,
            "context.workload.k": 256.0,
            "context.regime=hot": 1.0,
            "config.gemm.tile_m": float(tile),
            "config.gemm.stages": 3.0,
            "derived.effective_cta_waves": 2.0,
            "derived.smem_fraction_per_cta": 0.5,
            "derived.register_fraction_per_cta": 0.5,
            "derived.memory_roofline_ms": latency * 0.5,
            "derived.nominal_flops": 1e9,
            "derived.m_tail_fraction": float(m % tile != 0),
            "derived.n_tail_fraction": 0.0,
            "derived.k_tail_fraction": 0.0,
        },
        strategy="coordinate_local",
        outcome=TrialOutcome(
            status,  # type: ignore[arg-type]
            median_ms=latency if status == "ok" else None,
            timings_ms=(
                [latency * 1.01, latency, latency * 0.99, latency * 1.002, latency]
                if status == "ok"
                else []
            ),
            error=(
                None
                if status == "ok"
                else "candidate numerical guard nrmse cosine"
            ),
        ),
        started_at="",
        finished_at="",
        elapsed_s=0.1,
        parent_config_id=None if parent is None else f"tile-{parent}",
        coordinate=None if parent is None else "gemm.tile_m",
        coordinate_value=tile if parent is not None else None,
    )


class AutotuneEvidenceTests(unittest.TestCase):
    def _rows(self) -> list[Observation[object]]:
        rows = []
        sequence = 0
        for sku in ("small", "large"):
            for m in (128, 256, 512, 1024):
                for tile in (32, 64, 96, 128, 160, 192):
                    rows.append(
                        _row(
                            sku=sku,
                            m=m,
                            tile=tile,
                            sequence=sequence,
                            parent=None if tile == 32 else 32,
                        )
                    )
                    sequence += 1
                rows.append(
                    _row(
                        sku=sku,
                        m=m,
                        tile=224,
                        sequence=sequence,
                        status="correctness_error",
                    )
                )
                sequence += 1
        return rows

    def test_paired_failure_and_timing_analyses(self) -> None:
        rows = self._rows()
        moves = parent_move_analysis(rows, minimum_pairs=2)
        self.assertGreater(moves["paired_moves"], 20)
        self.assertTrue(moves["effects"])
        failures = failure_analysis(rows, minimum_support=2)
        self.assertEqual(failures["failure_kinds"]["numerical_contract"], 8)
        timing = timing_convergence_analysis(rows)
        self.assertEqual(timing["rows_with_raw_timings"], 48)
        self.assertTrue(timing["convergence"])

    def test_pairwise_model_round_trip(self) -> None:
        rows = self._rows()
        model = PairwisePreferenceModel()
        model.fit(rows)
        self.assertTrue(model.fitted)
        left = rows[3].features
        right = rows[0].features
        expected, _ = model.predict([left], [right])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairwise.json"
            model.save(path)
            actual, _ = PairwisePreferenceModel.load(path).predict(
                [left], [right]
            )
        self.assertAlmostEqual(float(expected[0]), float(actual[0]))

    def test_end_to_end_evidence_report(self) -> None:
        rows = self._rows()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "observations.jsonl").write_text(
                "".join(json.dumps(item.as_dict()) + "\n" for item in rows),
                encoding="utf-8",
            )
            report = build_evidence_study(
                [root],
                root / "report",
                minimum_pairs=2,
                minimum_contexts=2,
                pairwise_estimators=2,
            )
            self.assertIn("toy_kernel@1", report["pairwise_shape_heldout"])
            self.assertIn("timing_convergence", report)
            self.assertTrue(report["strategy_efficiency"])
            self.assertEqual(
                report["pairwise_artifact"]["type"],
                "rtx_pairwise_preference_bundle",
            )
            self.assertTrue((root / "report/evidence.json").is_file())
            self.assertTrue((root / "report/evidence.md").is_file())
            self.assertTrue((root / "report/pairwise_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
