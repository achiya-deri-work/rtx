from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from rtx.autotune.core import Observation
from rtx.autotune.outcomes import TrialOutcome
from rtx.autotune.sku_study import study_sku_relationships


class SKUStudyTests(unittest.TestCase):
    def test_context_normalization_exposes_sku_sensitive_coordinate(self) -> None:
        rows = []
        for sku in ("small_gpu", "large_gpu"):
            for m in (128, 512, 2048):
                context = f"{sku}-{m}"
                for tile, config_id in ((64, "tile64"), (128, "tile128")):
                    preferred = (sku == "small_gpu" and tile == 64) or (
                        sku == "large_gpu" and tile == 128
                    )
                    score = (1.0 if preferred else 1.2) * m / 128
                    rows.append(
                        Observation(
                            observation_id=f"{context}-{tile}",
                            session_id="session",
                            sequence=tile,
                            context_id=context,
                            family="toy_kernel",
                            kernel_revision=1,
                            config_id=config_id,
                            config={"tile_m": tile},
                            serialized_config={"tile_m": tile},
                            features={
                                f"context.device.sku.sku_family={sku}": 1.0,
                                "context.device.multiprocessor_count": (
                                    40.0 if sku == "small_gpu" else 120.0
                                ),
                                "context.workload.m": float(m),
                                "context.workload.n": 256.0,
                                "context.workload.k": 256.0,
                                "context.regime=hot": 1.0,
                            },
                            strategy="random",
                            outcome=TrialOutcome(
                                "ok", median_ms=score, timings_ms=[score]
                            ),
                            started_at="",
                            finished_at="",
                            elapsed_s=0.1,
                        )
                    )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            residual = root / "observations.jsonl"
            residual.write_text(
                "".join(json.dumps(row.as_dict()) + "\n" for row in rows),
                encoding="utf-8",
            )
            report = study_sku_relationships(
                [root], root / "report", minimum_contexts=2
            )
            sensitive = [
                item
                for item in report["portable_effects"]
                if item["coordinate"] == "config.tile_m"
                and item["classification"] == "sku_sensitive"
            ]
            self.assertEqual(len(sensitive), 2)
            self.assertEqual(len(report["winner_transfer"]), 2)
            self.assertTrue((root / "report/study.json").is_file())
            self.assertTrue((root / "report/study.md").is_file())


if __name__ == "__main__":
    unittest.main()
