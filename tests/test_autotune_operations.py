from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from rtx.autotune.audit import audit_bundles
from rtx.autotune.promotion import install_verified_winners
from rtx.autotune.winners import (
    RuntimeWinnerKey,
    list_runtime_winners,
    save_runtime_winner,
    runtime_tuning_lock,
)
from rtx.kernels.mxfp8 import (
    MXFP8_FWD_KERNEL_REVISION,
    MXFP8Problem,
    fwd_config_to_dict,
    normalize_fwd_config,
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


class AutotuneOperationsTests(unittest.TestCase):
    def test_runtime_tuning_lock_uses_a_context_specific_cache_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            key = RuntimeWinnerKey(
                "mxfp8_fused_fwd",
                MXFP8Problem(128, 256, 512),
                "hot",
                "device",
                7,
            )
            with runtime_tuning_lock(key, root=directory):
                locks = list(Path(directory).glob("runtime_locks/**/*.lock"))
                self.assertEqual(len(locks), 1)

    def test_runtime_winners_are_inspectable_without_deserialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key = RuntimeWinnerKey(
                "mxfp8_fused_fwd",
                MXFP8Problem(64, 128, 128),
                "hot",
                "device-id",
                MXFP8_FWD_KERNEL_REVISION,
            )
            path = save_runtime_winner(
                key,
                {"tile_m": 64},
                config_id="candidate-id",
                root=root,
                median_ms=0.01,
            )
            rows = list_runtime_winners(
                root=root,
                families=("mxfp8_fused_fwd",),
                device_ids=("device-id",),
            )
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0]["valid"])
            self.assertEqual(rows[0]["path"], str(path))
            self.assertEqual(rows[0]["config_id"], "candidate-id")
            self.assertEqual(rows[0]["kernel_revision"], MXFP8_FWD_KERNEL_REVISION)
            self.assertEqual(rows[0]["compatibility"], "compatible")

    def test_schema_v1_runtime_winner_is_visible_but_invalidated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = (
                root
                / "runtime_winners"
                / "mxfp8_fused_fwd"
                / "device-id"
                / "m64_n128_k128_hot_default.json"
            )
            _write(
                path,
                {
                    "schema_version": 1,
                    "family": "mxfp8_fused_fwd",
                    "device_id": "device-id",
                    "config": {"tile_m": 64},
                },
            )
            rows = list_runtime_winners(root=root)
            self.assertEqual(len(rows), 1)
            self.assertFalse(rows[0]["valid"])
            self.assertEqual(rows[0]["compatibility"], "schema_v1_invalidated")

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

    def test_audit_reports_interrupted_candidate_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = self._bundle(root)
            _write(
                bundle / "residuals/unit/events.jsonl",
                {
                    "kind": "candidate_started",
                    "payload": {
                        "attempt_id": "stalled-attempt",
                        "context_id": "context-id",
                        "config_id": "stalled-config",
                    },
                },
            )
            report = audit_bundles((root,))
            attempts = report["bundles"][0]["candidate_attempts"]
            self.assertEqual(attempts["started"], 1)
            self.assertEqual(attempts["orphaned"], 1)
            self.assertEqual(attempts["orphaned_config_ids"], ["stalled-config"])

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

    def test_posthoc_verification_supersedes_the_anytime_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = self._bundle(root)
            posthoc_id = "posthoc-config-id"
            posthoc_config = fwd_config_to_dict(
                normalize_fwd_config(quant_vec=2)
            )
            verification_path = bundle / "residuals/unit/verification.jsonl"
            with verification_path.open("a", encoding="utf-8") as sink:
                sink.write(
                    json.dumps(
                        {
                            "record_type": "verification_measurement",
                            "observation_key": "posthoc-verify-id",
                            "context_id": "context-id",
                            "config_id": posthoc_id,
                            "family": "mxfp8_fused_fwd",
                            "outcome": {
                                "status": "ok",
                                "summary_ms": {"median": 0.8},
                                "timings_ms": [0.8] * 11,
                            },
                        }
                    )
                    + "\n"
                )
            _write(
                bundle / "verification_summary.json",
                {
                    "type": "rtx_autotune_posthoc_verification",
                    "results": [
                        {
                            "context_id": "context-id",
                            "family": "mxfp8_fused_fwd",
                            "kernel_revision": MXFP8_FWD_KERNEL_REVISION,
                            "shape": "m64_n128_k128",
                            "regime": "hot",
                            "treatment": "random_local",
                            "replicate": 0,
                            "target_trials": 65,
                            "verification": {
                                "status": "ok",
                                "winner_id": posthoc_id,
                                "winner_config": posthoc_config,
                            },
                        }
                    ],
                },
            )

            cache = root / "cache"
            report = install_verified_winners((root,), cache_dir=cache)

            self.assertEqual(len(report["installed"]), 1)
            document = json.loads(
                Path(report["installed"][0]["path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(document["config_id"], posthoc_id)
            self.assertEqual(document["median_ms"], 0.8)

    def test_posthoc_only_bundle_installs_after_watchdog_exit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = self._bundle(root)
            summary = json.loads(
                (bundle / "summary.json").read_text(encoding="utf-8")
            )
            _write(
                bundle / "verification_summary.json",
                {
                    "type": "rtx_autotune_posthoc_verification",
                    "results": summary["results"],
                },
            )
            (bundle / "summary.json").unlink()

            cache = root / "cache"
            report = install_verified_winners((root,), cache_dir=cache)

            self.assertEqual(len(report["installed"]), 1)
            document = json.loads(
                Path(report["installed"][0]["path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(document["config_id"], "config-id")
            self.assertEqual(document["median_ms"], 0.9)


if __name__ == "__main__":
    unittest.main()
