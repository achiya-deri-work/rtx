from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from rtx.autotune import DeviceFingerprint
from rtx.fp8 import DEFAULT_MXFP8_PREQUANT_CONFIG
from rtx.kernels.mxfp8 import MXFP8Problem
from rtx.prequant_experiments import (
    analyze_observations,
    BenchmarkProtocol,
    CandidateCompileError,
    CandidateCorrectnessError,
    ExperimentJournal,
    ExperimentManifest,
    ShapeSpec,
    _reference_prequant_config,
    config_in_shard,
    collect_timing_samples,
    derived_features,
    export_journal_csv,
    generate_legal_catalog,
    merge_journals,
    robust_summary,
)
from rtx.prequant_autotune import prequant_config_id


def _fingerprint() -> DeviceFingerprint:
    return DeviceFingerprint(
        device_index=0,
        name="Synthetic SM120",
        capability=(12, 0),
        total_memory=16 << 30,
        multiprocessor_count=70,
        uuid="GPU-EXPERIMENT-TEST",
        torch_version="test",
        torch_cuda_version="test",
        cutlass_dsl_version="test",
        cuda_python_version="test",
        python_version="test",
        platform="test",
    )


class PrequantExperimentTests(unittest.TestCase):
    def test_adaptive_sampling_uses_odd_dispersion_gated_stages(self) -> None:
        protocol = BenchmarkProtocol(
            samples=5,
            confirm_samples=11,
            screen_stable_cv=0.005,
            confirm_stable_cv=0.0025,
        )
        stable = collect_timing_samples(
            lambda index: (1.0, 1.001, 0.999, 1.0, 1.0)[index],
            protocol,
            requested_samples=5,
        )
        self.assertEqual(len(stable), 3)
        self.assertEqual(
            protocol.sampling_metadata(stable, requested_samples=5)["stop_reason"],
            "stable_dispersion",
        )
        noisy_values = (1.0, 1.1, 0.9, 1.08, 0.92)
        noisy = collect_timing_samples(
            lambda index: noisy_values[index],
            protocol,
            requested_samples=5,
        )
        self.assertEqual(len(noisy), 5)
        confirmed = collect_timing_samples(
            lambda index: 1.0 + (index % 2) * 0.001,
            protocol,
            requested_samples=11,
        )
        self.assertEqual(len(confirmed), 5)

    def test_adaptive_race_extends_noisy_devices(self) -> None:
        protocol = BenchmarkProtocol(race_rounds=11, race_min_rounds=7)
        stable = [1.0, 1.001, 0.999, 1.0, 1.001, 0.999, 1.0]
        self.assertTrue(protocol.race_complete(stable, stable))
        self.assertEqual(
            protocol.race_sampling_metadata(stable, stable)["stop_reason"],
            "stable_dispersion",
        )
        noisy = [1.0, 1.05, 0.95, 1.04, 0.96, 1.03, 0.97]
        self.assertFalse(protocol.race_complete(noisy, noisy))
        self.assertTrue(protocol.race_complete(noisy + [1.02] * 4, noisy + [0.98] * 4))

    def test_reference_config_supports_64_row_campaign_shapes(self) -> None:
        problem = MXFP8Problem(64, 1536, 1536)
        config = _reference_prequant_config(problem)
        self.assertIsNone(config.rejection(problem))
        self.assertEqual(config.gemm.scale_layout, "row_major")
        regular = MXFP8Problem(128, 1536, 1536)
        self.assertEqual(
            _reference_prequant_config(regular),
            DEFAULT_MXFP8_PREQUANT_CONFIG,
        )

    def test_manifest_round_trip_and_digest(self) -> None:
        manifest = ExperimentManifest(
            name="pilot",
            shapes=(ShapeSpec(512, 1536, 1536, name="underfill"),),
            regimes=("hot",),
            candidates_per_shape=12,
            promote=3,
            seed=9,
            protocol=BenchmarkProtocol(samples=3, confirm_samples=5),
        )
        restored = ExperimentManifest.from_dict(manifest.as_dict())
        self.assertEqual(restored, manifest)
        self.assertEqual(restored.digest, manifest.digest)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest.as_dict()), encoding="utf-8")
            self.assertEqual(ExperimentManifest.load(path), manifest)

    def test_robust_summary_is_deterministic_and_resists_outlier(self) -> None:
        values = [1.0, 1.0, 1.01, 0.99, 100.0]
        first = robust_summary(values, seed=17, bootstrap_resamples=200)
        second = robust_summary(values, seed=17, bootstrap_resamples=200)
        self.assertEqual(first, second)
        self.assertEqual(first.median, 1.0)
        self.assertLess(first.median, first.mean)
        self.assertLessEqual(first.ci_low, first.median)
        self.assertGreaterEqual(first.ci_high, first.median)

    def test_conditional_catalog_is_legal_deterministic_and_diverse(self) -> None:
        problem = MXFP8Problem(512, 1536, 1536)
        first = generate_legal_catalog(problem, 40, seed=123)
        second = generate_legal_catalog(problem, 40, seed=123)
        self.assertEqual(
            [prequant_config_id(config) for config in first],
            [prequant_config_id(config) for config in second],
        )
        self.assertEqual(len(first), 40)
        self.assertTrue(all(config.rejection(problem) is None for config in first))
        self.assertEqual(first[0], DEFAULT_MXFP8_PREQUANT_CONFIG)
        self.assertGreater(len({config.gemm.raster for config in first}), 1)
        self.assertGreater(len({config.gemm.scale_layout for config in first}), 1)

    def test_hash_shards_are_disjoint_and_complete(self) -> None:
        configs = generate_legal_catalog(MXFP8Problem(512, 1536, 1536), 50, seed=8)
        memberships = [
            [index for index in range(4) if config_in_shard(config, index, 4)]
            for config in configs
        ]
        self.assertTrue(all(len(indices) == 1 for indices in memberships))
        self.assertEqual(sum(len(indices) for indices in memberships), len(configs))

    def test_features_capture_waves_reuse_and_l2_boundaries(self) -> None:
        shape = ShapeSpec(8192, 1536, 1536)
        features = derived_features(
            shape,
            DEFAULT_MXFP8_PREQUANT_CONFIG,
            _fingerprint(),
            l2_cache_size=48 << 20,
        )
        self.assertEqual(features["cta_m"], 64)
        self.assertEqual(features["cta_n"], 12)
        self.assertEqual(features["cta_count"], 768)
        self.assertEqual(features["w_reuse_ctas"], 64)
        self.assertIn("working_set_l2_ratio", features)

    def test_journal_is_append_only_resumable_and_exports_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trials.jsonl"
            journal = ExperimentJournal(path)
            journal.append(
                {
                    "record_type": "measurement",
                    "observation_key": "one",
                    "recorded_at": "now",
                    "device_id": "gpu",
                    "shape_key": "m1_n1_k32",
                    "regime": "hot",
                    "stage": "screen",
                    "config_id": "cfg",
                    "config": {"gemm": {"raster": "m"}},
                    "features": {"cta_count": 1},
                    "outcome": {
                        "status": "ok",
                        "summary_ms": {"median": 1.0, "ci_low": 0.9, "ci_high": 1.1},
                    },
                }
            )
            journal.append(
                {
                    "record_type": "session",
                    "event": "finished",
                    "recorded_at": "later",
                }
            )
            self.assertEqual(journal.completed_keys(), {"one"})
            self.assertEqual(len(journal.records()), 2)
            csv_path = Path(directory) / "trials.csv"
            export_journal_csv(path, csv_path)
            with csv_path.open(encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["median_ms"], "1.0")
            self.assertEqual(rows[0]["feature_cta_count"], "1")
            self.assertEqual(rows[0]["config_gemm_raster"], "m")
            merged_path = Path(directory) / "merged.jsonl"
            report = merge_journals((path, path), merged_path)
            self.assertEqual(report["observations"], 1)
            self.assertEqual(report["duplicates"], 1)
            merged = ExperimentJournal(merged_path).records()
            self.assertEqual(
                len([row for row in merged if row.get("record_type") == "measurement"]),
                1,
            )
            analysis = analyze_observations(merged, tolerance=0.01)
            self.assertEqual(analysis["context_count"], 1)
            self.assertEqual(analysis["portfolio_size"], 1)
            self.assertEqual(analysis["portfolio"][0]["config_id"], "cfg")

    def test_measurement_classifies_prepare_failures(self) -> None:
        # Construct without CUDA because both failures occur before timing.
        from rtx.prequant_experiments import PrequantBenchmarkHarness

        harness = object.__new__(PrequantBenchmarkHarness)
        harness.protocol = BenchmarkProtocol(telemetry=False)
        harness.device = "cuda"
        for error, expected in (
            (CandidateCompileError("compile"), "compile_error"),
            (CandidateCorrectnessError("wrong"), "correctness_error"),
            (RuntimeError("launch"), "runtime_error"),
        ):
            harness.prepare = lambda _config, error=error: (_ for _ in ()).throw(error)
            outcome = harness.measure(
                DEFAULT_MXFP8_PREQUANT_CONFIG,
                samples=1,
                seed=0,
            )
            self.assertEqual(outcome["status"], expected)


if __name__ == "__main__":
    unittest.main()
