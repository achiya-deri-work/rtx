from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from rtx.autotune import (
    CoordinateDescentPolicy,
    CoordinateDescentTuner,
    DeviceFingerprint,
    JsonTuningDatabase,
    TrialOutcome,
)
from rtx.kernels.mxfp8 import (
    DEFAULT_MXFP8_FWD_CONFIG,
    MXFP8Problem,
    normalize_fwd_config,
)


def _fingerprint() -> DeviceFingerprint:
    return DeviceFingerprint(
        device_index=0,
        name="Synthetic SM120",
        capability=(12, 0),
        total_memory=16 << 30,
        multiprocessor_count=70,
        uuid="GPU-TEST",
        torch_version="test",
        torch_cuda_version="test",
        cutlass_dsl_version="test",
        cuda_python_version="test",
        python_version="test",
        platform="test",
    )


class PersistentCoordinateDescentTests(unittest.TestCase):
    def test_dependent_launch_coordinates_are_normalized(self) -> None:
        config = normalize_fwd_config(
            atom_layout_m=2,
            atom_layout_n=2,
            schedule="warp_specialized",
        )
        self.assertEqual(config.num_mma_warps, 4)
        self.assertEqual(config.quantizer_warps, 4)
        self.assertEqual(config.producer_warps, 1)
        self.assertEqual(config.num_threads, 160)

    def test_compound_smem_rmem_and_pipeline_coordinates_are_real(self) -> None:
        config = normalize_fwd_config(
            smem_rmem_tile=(128, 128, 128, 16, 64),
            bf16_pipeline=("tma", "warp_specialized", 32, "64b", 4),
        )
        self.assertEqual(config.smem_rmem_tile, (128, 128, 128, 16, 64))
        self.assertEqual(config.atom_layout_m, 8)
        self.assertEqual(config.atom_layout_n, 2)
        self.assertEqual(
            config.bf16_pipeline,
            ("tma", "warp_specialized", 32, "64b", 4),
        )
        self.assertEqual(config.num_mma_warps, 16)
        self.assertEqual(config.producer_warps, 1)
        self.assertIsNone(
            config.implementation_rejection(MXFP8Problem(128, 128, 128))
        )

    def test_pipeline_block_move_crosses_slow_intermediate_states(self) -> None:
        scalar = ("scalar", "cooperative", 128, "128b", 1)
        deep_tma = ("tma", "warp_specialized", 32, "64b", 4)
        axes = {"bf16_pipeline": (scalar, deep_tma)}
        policy = CoordinateDescentPolicy(
            time_budget_s=10,
            max_passes=2,
            warmup=0,
            samples=1,
            calls_per_sample=1,
            min_improvement=0,
            coordinate_order=("bf16_pipeline",),
        )
        problem = MXFP8Problem(128, 128, 128)

        def evaluator(config):
            score = 0.5 if config.bf16_pipeline == deep_tma else 1.0
            return TrialOutcome("ok", median_ms=score, timings_ms=[score])

        with tempfile.TemporaryDirectory() as directory:
            database = JsonTuningDatabase(directory, _fingerprint(), problem, axes)
            result = CoordinateDescentTuner(
                problem,
                evaluator,
                database,
                policy,
                axes=axes,
            ).tune(DEFAULT_MXFP8_FWD_CONFIG)

        self.assertEqual(result.config.bf16_pipeline, deep_tma)
        self.assertEqual(result.median_ms, 0.5)

    def test_operand_ldmatrix_widths_are_independent_real_coordinates(self) -> None:
        config = normalize_fwd_config(
            a_ldmatrix_matrices=1,
            b_ldmatrix_matrices=2,
        )
        self.assertEqual(config.a_ldmatrix_matrices, 1)
        self.assertEqual(config.b_ldmatrix_matrices, 2)
        self.assertIsNone(
            config.implementation_rejection(MXFP8Problem(128, 128, 128))
        )

    def test_coordinate_descent_persists_and_resumes(self) -> None:
        axes = {
            "tile_k": (128, 256),
            "maxrregcount": (128, 160, 192, 224, 255),
        }
        policy = CoordinateDescentPolicy(
            time_budget_s=20,
            max_passes=3,
            warmup=0,
            samples=1,
            calls_per_sample=1,
            min_improvement=0,
            coordinate_order=("tile_k", "maxrregcount"),
        )
        problem = MXFP8Problem(128, 128, 128)
        calls: list[tuple[int, int]] = []

        def evaluator(config):
            calls.append((config.tile_k, config.maxrregcount))
            score = 1.0 + abs(config.tile_k - 256) / 1000
            score += abs(config.maxrregcount - 192) / 1000
            return TrialOutcome("ok", median_ms=score, timings_ms=[score])

        with tempfile.TemporaryDirectory() as directory:
            database = JsonTuningDatabase(directory, _fingerprint(), problem, axes)
            tuner = CoordinateDescentTuner(
                problem,
                evaluator,
                database,
                policy,
                axes=axes,
                architecture_validator=lambda _cfg, _problem: None,
                implementation_validator=lambda _cfg, _problem: None,
            )
            result = tuner.tune(DEFAULT_MXFP8_FWD_CONFIG)
            self.assertEqual(result.config.tile_k, 256)
            self.assertEqual(result.config.maxrregcount, 192)
            self.assertGreater(len(calls), 0)
            self.assertTrue(result.database_path.exists())

            with result.database_path.open(encoding="utf-8") as source:
                document = json.load(source)
            self.assertEqual(document["schema_version"], 1)
            self.assertEqual(document["best"]["config"]["tile_k"], 256)
            self.assertEqual(document["best"]["config"]["maxrregcount"], 192)
            self.assertTrue(document["trials"])
            self.assertEqual(document["sessions"][-1]["status"], "complete")

            resumed_calls: list[object] = []

            def should_not_run(config):
                resumed_calls.append(config)
                return TrialOutcome("runtime_error", error="unexpected evaluation")

            resumed = CoordinateDescentTuner(
                problem,
                should_not_run,
                database,
                policy,
                axes=axes,
                architecture_validator=lambda _cfg, _problem: None,
                implementation_validator=lambda _cfg, _problem: None,
            ).tune(DEFAULT_MXFP8_FWD_CONFIG)
            self.assertEqual(resumed.config, result.config)
            self.assertEqual(resumed_calls, [])
            self.assertEqual(resumed.evaluated_trials, 0)
            self.assertGreater(resumed.reused_trials, 0)

    def test_rejections_are_saved_as_trials(self) -> None:
        axes = {"schedule": ("cooperative", "pingpong")}
        policy = CoordinateDescentPolicy(
            time_budget_s=10,
            max_passes=1,
            warmup=0,
            samples=1,
            calls_per_sample=1,
            coordinate_order=("schedule",),
        )
        problem = MXFP8Problem(128, 128, 128)

        with tempfile.TemporaryDirectory() as directory:
            database = JsonTuningDatabase(directory, _fingerprint(), problem, axes)
            tuner = CoordinateDescentTuner(
                problem,
                lambda _config: TrialOutcome("ok", median_ms=1.0, timings_ms=[1.0]),
                database,
                policy,
                axes=axes,
            )
            tuner.tune()
            with Path(database.path).open(encoding="utf-8") as source:
                document = json.load(source)
            statuses = {trial["status"] for trial in document["trials"].values()}
            self.assertIn("ok", statuses)
            self.assertIn("implementation_rejected", statuses)

    def test_force_only_bypasses_previous_sessions(self) -> None:
        axes = {"tile_k": (128, 256)}
        policy = CoordinateDescentPolicy(
            time_budget_s=10,
            max_passes=1,
            warmup=0,
            samples=1,
            calls_per_sample=1,
            min_improvement=0,
            coordinate_order=("tile_k",),
            force=True,
        )
        problem = MXFP8Problem(128, 128, 256)
        calls: list[int] = []

        def evaluator(config):
            calls.append(config.tile_k)
            return TrialOutcome("ok", median_ms=1.0, timings_ms=[1.0])

        with tempfile.TemporaryDirectory() as directory:
            database = JsonTuningDatabase(directory, _fingerprint(), problem, axes)
            result = CoordinateDescentTuner(
                problem,
                evaluator,
                database,
                policy,
                axes=axes,
                architecture_validator=lambda _cfg, _problem: None,
                implementation_validator=lambda _cfg, _problem: None,
            ).tune(DEFAULT_MXFP8_FWD_CONFIG)

        self.assertEqual(calls, [128, 256])
        self.assertEqual(result.evaluated_trials, 2)
        self.assertEqual(result.reused_trials, 1)

    def test_persisted_winner_respects_minimum_improvement(self) -> None:
        axes = {"tile_k": (128, 256)}
        policy = CoordinateDescentPolicy(
            time_budget_s=10,
            max_passes=1,
            warmup=0,
            samples=1,
            calls_per_sample=1,
            min_improvement=0.002,
            coordinate_order=("tile_k",),
        )
        problem = MXFP8Problem(128, 128, 256)

        def evaluator(config):
            # The 0.1% delta is below the 0.2% acceptance threshold.
            score = 0.999 if config.tile_k == 256 else 1.0
            return TrialOutcome("ok", median_ms=score, timings_ms=[score])

        with tempfile.TemporaryDirectory() as directory:
            database = JsonTuningDatabase(directory, _fingerprint(), problem, axes)
            result = CoordinateDescentTuner(
                problem,
                evaluator,
                database,
                policy,
                axes=axes,
                architecture_validator=lambda _cfg, _problem: None,
                implementation_validator=lambda _cfg, _problem: None,
            ).tune(DEFAULT_MXFP8_FWD_CONFIG)
            cached = database.best()

        self.assertEqual(result.config.tile_k, 128)
        self.assertIsNotNone(cached)
        self.assertEqual(cached[0].tile_k, 128)
        self.assertEqual(cached[1], result.median_ms)

    def test_random_restarts_escape_a_coordinate_local_minimum(self) -> None:
        axes = {
            "tile_k": (128, 256),
            "maxrregcount": (128, 255),
        }
        policy = CoordinateDescentPolicy(
            time_budget_s=10,
            max_passes=2,
            restarts=8,
            warmup=0,
            samples=1,
            calls_per_sample=1,
            min_improvement=0,
            coordinate_order=("tile_k", "maxrregcount"),
            seed=0,
        )
        problem = MXFP8Problem(128, 128, 256)
        scores = {
            (128, 255): 1.0,
            (256, 255): 2.0,
            (128, 128): 2.0,
            (256, 128): 0.5,
        }

        def evaluator(config):
            score = scores[(config.tile_k, config.maxrregcount)]
            return TrialOutcome("ok", median_ms=score, timings_ms=[score])

        with tempfile.TemporaryDirectory() as directory:
            database = JsonTuningDatabase(directory, _fingerprint(), problem, axes)
            result = CoordinateDescentTuner(
                problem,
                evaluator,
                database,
                policy,
                axes=axes,
                architecture_validator=lambda _cfg, _problem: None,
                implementation_validator=lambda _cfg, _problem: None,
            ).tune(DEFAULT_MXFP8_FWD_CONFIG)

        self.assertEqual(result.config.tile_k, 256)
        self.assertEqual(result.config.maxrregcount, 128)
        self.assertEqual(result.median_ms, 0.5)


if __name__ == "__main__":
    unittest.main()
