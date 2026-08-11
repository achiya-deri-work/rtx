from __future__ import annotations

import tempfile
import unittest
import torch

from rtx.autotune import CoordinateDescentPolicy, DeviceFingerprint, TrialOutcome
from rtx.fp8 import DEFAULT_MXFP8_PREQUANT_CONFIG, MXFP8Linear, mxfp8_linear
from rtx.kernels.mxfp8 import MXFP8Problem
from rtx.kernels.mxfp8_gemm import MXFP8GemmConfig
from rtx.prequant_autotune import (
    PREQUANT_SEARCH_SPACE,
    PrequantCoordinateDescentTuner,
    PrequantJsonTuningDatabase,
    prequant_config_from_dict,
    prequant_config_id,
    prequant_config_to_dict,
    update_prequant_config,
    load_cached_mxfp8_prequant_config,
    tune_mxfp8_prequant,
)


def _fingerprint() -> DeviceFingerprint:
    return DeviceFingerprint(
        device_index=0,
        name="Synthetic SM120",
        capability=(12, 0),
        total_memory=16 << 30,
        multiprocessor_count=70,
        uuid="GPU-PREQUANT-TEST",
        torch_version="test",
        torch_cuda_version="test",
        cutlass_dsl_version="test",
        cuda_python_version="test",
        python_version="test",
        platform="test",
    )


class PrequantAutotuneTests(unittest.TestCase):
    problem = MXFP8Problem(512, 1536, 1536)

    def test_config_round_trip_and_structural_layouts(self) -> None:
        config = DEFAULT_MXFP8_PREQUANT_CONFIG
        self.assertEqual(
            prequant_config_from_dict(prequant_config_to_dict(config)), config
        )
        layouts = set()
        for update in PREQUANT_SEARCH_SPACE["layout_transport"]:
            candidate = update_prequant_config(config, update)
            layouts.add(
                (
                    candidate.quant.scale_layout,
                    candidate.resolved_weight_quant().scale_layout,
                    candidate.gemm.scale_layout,
                    candidate.gemm.scale_role,
                    candidate.gemm.tile_m,
                )
            )
        self.assertIn(("mma128", "mma128", "mma128", "tma", 128), layouts)
        self.assertIn(("mma64", "mma128", "mma64x128", "tma", 64), layouts)
        self.assertIn(
            ("row_major", "row_major", "row_major", "producer", 128), layouts
        )
        persistence = {
            (
                update["gemm"]["tiles_per_cta"],
                update["gemm"]["tile_locality"],
            )
            for update in PREQUANT_SEARCH_SPACE["gemm_persistence"]
        }
        self.assertIn((1, "raster"), persistence)
        self.assertIn((4, "same_a"), persistence)
        self.assertIn((8, "serpentine_b"), persistence)

    def test_dual_x_coordinate_updates_the_shared_schedule(self) -> None:
        candidate = update_prequant_config(
            DEFAULT_MXFP8_PREQUANT_CONFIG,
            {"quant": {"maxrregcount": 64}},
        )
        self.assertEqual(candidate.quant.maxrregcount, 64)
        self.assertEqual(candidate.resolved_weight_quant().maxrregcount, 64)
        self.assertIsNone(candidate.rejection(self.problem))
        restored = update_prequant_config(
            candidate,
            {"quant": {"maxrregcount": 96}},
        )
        self.assertEqual(
            prequant_config_id(restored),
            prequant_config_id(DEFAULT_MXFP8_PREQUANT_CONFIG),
        )

    def test_inactive_tma_scale_staging_knobs_are_rejected(self) -> None:
        config = MXFP8GemmConfig(
            scale_layout="mma128",
            scale_role="tma",
            scale_schedule="after_wait",
        )
        self.assertIn("inactive", config.rejection(self.problem))

    def test_scale_pipeline_coordinates_have_static_legality(self) -> None:
        for coordinate in (
            "mma_schedule",
            "scale_recycle",
            "scale_smem_store",
        ):
            self.assertIn(coordinate, PREQUANT_SEARCH_SPACE)
        self.assertIsNone(
            MXFP8GemmConfig(
                stages=2,
                scale_recycle="staged",
                scale_smem_store="packed",
                scale_load_vec=4,
                mma_schedule="preload",
            ).rejection(self.problem)
        )
        self.assertIn(
            "2+ stages",
            MXFP8GemmConfig(
                stages=1,
                scale_recycle="staged",
            ).rejection(self.problem),
        )
        self.assertIn(
            "vectorized",
            MXFP8GemmConfig(
                scale_smem_store="packed",
                scale_load_vec=1,
            ).rejection(self.problem),
        )

    def test_setmaxregister_values_are_rejected_before_compilation(self) -> None:
        self.assertIn(
            "between 24 and 256",
            MXFP8GemmConfig(producer_registers=16).rejection(self.problem),
        )
        self.assertIn(
            "multiple of 8",
            MXFP8GemmConfig(consumer_registers=130).rejection(self.problem),
        )

    def test_w_coordinate_can_cross_to_independent_launches(self) -> None:
        axes = {
            "w_registers": (
                {"weight_quant": {"maxrregcount": 96}},
                {"weight_quant": {"maxrregcount": 64}},
            )
        }
        policy = CoordinateDescentPolicy(
            time_budget_s=10,
            max_passes=2,
            warmup=0,
            samples=1,
            calls_per_sample=1,
            min_improvement=0,
            coordinate_order=("w_registers",),
        )

        def evaluator(config):
            if config.quant_launches == "dual":
                score = 1.0
            else:
                score = 0.5 + config.resolved_weight_quant().maxrregcount / 1000
            return TrialOutcome("ok", median_ms=score, timings_ms=[score])

        with tempfile.TemporaryDirectory() as directory:
            database = PrequantJsonTuningDatabase(
                directory, _fingerprint(), self.problem, axes
            )
            result = PrequantCoordinateDescentTuner(
                self.problem,
                evaluator,  # type: ignore[arg-type]
                database,
                policy,
                axes=axes,
            ).tune()
            resumed_calls = []

            def should_not_run(config):
                resumed_calls.append(config)
                return TrialOutcome("runtime_error", error="unexpected")

            resumed = PrequantCoordinateDescentTuner(
                self.problem,
                should_not_run,  # type: ignore[arg-type]
                database,
                policy,
                axes=axes,
            ).tune()

        self.assertEqual(result.config.quant_launches, "separate")
        self.assertEqual(result.config.resolved_weight_quant().maxrregcount, 64)
        self.assertEqual(resumed.config, result.config)
        self.assertEqual(resumed_calls, [])
        self.assertGreater(resumed.reused_trials, 0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class PrequantAutotuneCudaTests(unittest.TestCase):
    def test_focused_joint_search_and_cache_reuse(self) -> None:
        if torch.cuda.get_device_capability()[0] != 12:
            self.skipTest("joint tuner requires SM120/SM121")
        axes = {
            "layout_transport": tuple(
                PREQUANT_SEARCH_SPACE["layout_transport"][index]
                for index in (0, 1, 4)
            ),
            "quant_launches": PREQUANT_SEARCH_SPACE["quant_launches"],
            "scale_schedule": PREQUANT_SEARCH_SPACE["scale_schedule"],
            "global_l2_fetch": (
                {"l2_fetch_granularity": None},
                {"l2_fetch_granularity": 64},
            ),
        }
        policy = CoordinateDescentPolicy(
            time_budget_s=120,
            max_passes=1,
            warmup=2,
            samples=3,
            calls_per_sample=5,
            correctness_rtol=5e-2,
            correctness_atol=5e-1,
            coordinate_order=tuple(axes),
        )
        torch.manual_seed(991)
        x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        with tempfile.TemporaryDirectory() as directory:
            result = tune_mxfp8_prequant(
                x,
                weight,
                policy=policy,
                cache_dir=directory,
                axes=axes,
                progress=None,
            )
            cached = load_cached_mxfp8_prequant_config(
                MXFP8Problem(128, 128, 128),
                device=x.device,
                cache_dir=directory,
                axes=axes,
            )
        self.assertIsNotNone(cached)
        self.assertEqual(cached, result.config)
        self.assertGreater(result.evaluated_trials, 0)

    def test_compile_first_runtime_autotune_request(self) -> None:
        if torch.cuda.get_device_capability()[0] != 12:
            self.skipTest("joint tuner requires SM120/SM121")
        # Other frontend tests deliberately compile this shared module method
        # under many constant policies.  Isolate this fullgraph test from the
        # process-global Dynamo recompile budget.
        torch.compiler.reset()
        policy = CoordinateDescentPolicy(
            time_budget_s=30,
            max_passes=1,
            warmup=1,
            samples=2,
            calls_per_sample=2,
            correctness_rtol=5e-2,
            correctness_atol=5e-1,
            coordinate_order=("quant_launches",),
        )
        torch.manual_seed(992)
        x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        with tempfile.TemporaryDirectory() as directory:
            layer = MXFP8Linear(
                128,
                128,
                device="cuda",
                backend="auto",
                autotune="coordinate",
                tuning_policy=policy,
                autotune_cache_dir=directory,
            ).eval().requires_grad_(False)
            compiled = torch.compile(layer, fullgraph=True, dynamic=False)
            actual = compiled(x)
            expected = layer(x)
            compiled_functional = torch.compile(
                lambda value: mxfp8_linear(
                    value,
                    layer.weight,
                    backend="prequant",
                    autotune="coordinate",
                    tuning_policy=policy,
                    autotune_cache_dir=directory,
                ),
                fullgraph=True,
                dynamic=False,
            )
            functional = compiled_functional(x)
            torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(functional, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
