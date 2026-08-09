from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from rtx.autotune.audit import audit_bundles
from rtx.autotune.promotion import install_verified_winners
from rtx.kernels.mxfp8 import (
    MXFP8_FWD_KERNEL_REVISION,
    fwd_config_to_dict,
    normalize_fwd_config,
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


class AutotuneOperationsTests(unittest.TestCase):
    def _bundle(self, root: Path) -> Path:
        bundle = root / "campaign" / "machine" / "shard-000-of-001"
        machine = {
            "machine_id": "machine-id",
            "device": {
                "fingerprint_id": "device-id",
                "fingerprint": {"name": "Synthetic RTX"},
            },
        }
        manifest = {
            "jobs": [{"shapes": [{"m": 64, "n": 128, "k": 128}], "regimes": ["hot"]}],
            "replicates": 1,
            "treatments": [],
            "shard_count": 1,
        }
        _write(bundle / "machine.json", machine)
        _write(bundle / "manifest.json", manifest)
        residual = bundle / "residuals" / "unit"
        _write(
            residual / "unit.json",
            {
                "context_id": "context-id",
                "family": "mxfp8_fused_fwd",
                "shape_key": "m64_n128_k128",
                "regime": "hot",
            },
        )
        observation = {
            "record_type": "observation",
            "observation_id": "observation-id",
            "context_id": "context-id",
            "config_id": "config-id",
            "family": "mxfp8_fused_fwd",
            "outcome": {"status": "ok", "median_ms": 1.0},
        }
        _write(residual / "observations.jsonl", observation)
        verification = {
            "record_type": "verification_measurement",
            "observation_key": "verify-id",
            "context_id": "context-id",
            "config_id": "config-id",
            "family": "mxfp8_fused_fwd",
            "outcome": {
                "status": "ok",
                "summary_ms": {"median": 0.9},
                "timings_ms": [0.9] * 11,
            },
        }
        _write(residual / "verification.jsonl", verification)
        _write(
            bundle / "summary.json",
            {
                "results": [
                    {
                        "context_id": "context-id",
                        "family": "mxfp8_fused_fwd",
                        "kernel_revision": MXFP8_FWD_KERNEL_REVISION,
                        "shape": "m64_n128_k128",
                        "regime": "hot",
                        "treatment": "random_local",
                        "replicate": 0,
                        "target_trials": 64,
                        "verification": {
                            "status": "ok",
                            "winner_id": "config-id",
                            "winner_config": fwd_config_to_dict(
                                normalize_fwd_config(tile_m=64)
                            ),
                        },
                    }
                ]
            },
        )
        return bundle

    def test_audit_reports_coverage_and_tolerates_only_a_crash_tail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = self._bundle(root)
            with (bundle / "residuals/unit/events.jsonl").open("w", encoding="utf-8") as sink:
                sink.write(json.dumps({"record_type": "event"}) + "\n")
                sink.write('{"record_type":')
            report = audit_bundles((root,))
            self.assertTrue(report["ok"])
            self.assertEqual(report["summary"]["contexts"], 1)
            self.assertEqual(report["summary"]["malformed_tails"], 1)

    def test_audit_rejects_duplicate_observations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = self._bundle(root)
            path = bundle / "residuals/unit/observations.jsonl"
            original = json.loads(path.read_text(encoding="utf-8"))
            conflicting = dict(original)
            conflicting["config_id"] = "other-config"
            path.write_text(
                json.dumps(original) + "\n" + json.dumps(conflicting) + "\n",
                encoding="utf-8",
            )
            report = audit_bundles((root,))
            self.assertFalse(report["ok"])
            self.assertEqual(report["summary"]["errors"], 1)

    def test_install_winners_writes_runtime_cache_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._bundle(root)
            cache = root / "cache"
            report = install_verified_winners((root,), cache_dir=cache)
            self.assertEqual(len(report["installed"]), 1)
            installed = Path(report["installed"][0]["path"])
            document = json.loads(installed.read_text(encoding="utf-8"))
            self.assertEqual(document["config_id"], "config-id")
            self.assertEqual(document["median_ms"], 0.9)
            self.assertEqual(document["metadata"]["support"], 1)


if __name__ == "__main__":
    unittest.main()
