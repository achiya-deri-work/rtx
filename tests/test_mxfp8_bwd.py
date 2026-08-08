from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import torch

from rtx.bwd_autotune import (
    BWD_SEARCH_SPACE,
    BwdCoordinateDescentTuner,
    bwd_config_from_dict,
    bwd_config_id,
    bwd_config_to_dict,
    load_mxfp8_bwd_config,
    update_bwd_config,
)
from rtx.bwd_experiments import BwdBenchmarkHarness
from rtx.fp8_bwd import mxfp8_linear_backward
from rtx.kernels.mxfp8 import MXFP8Problem
from rtx.kernels.mxfp8_bwd import (
    DEFAULT_MXFP8_BWD_CONFIG,
)
from rtx.kernels.mxfp8_quant import (
    MXFP8QuantConfig,
    compile_mxfp8_quant,
    compile_mxfp8_transposed_quant,
)
from rtx.prequant_experiments import BenchmarkProtocol, ShapeSpec


def _has_sm120() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12


class TestMXFP8BwdConfiguration(unittest.TestCase):
    def test_default_maps_forward_axes_to_both_backward_gemms(self) -> None:
        problem = MXFP8Problem(512, 1536, 1536)
        self.assertIsNone(DEFAULT_MXFP8_BWD_CONFIG.rejection(problem))
        self.assertIsNone(
            DEFAULT_MXFP8_BWD_CONFIG.implementation_rejection(problem)
        )

    def test_forward_scales_are_not_admitted_as_backward_scales(self) -> None:
        problem = MXFP8Problem(510, 1536, 1536)
        reason = DEFAULT_MXFP8_BWD_CONFIG.rejection(problem)
        self.assertIsNotNone(reason)
        self.assertIn("become MXFP8 reduction axes", reason)

    def test_serialization_is_stable(self) -> None:
        restored = bwd_config_from_dict(
            bwd_config_to_dict(DEFAULT_MXFP8_BWD_CONFIG)
        )
        self.assertEqual(restored, DEFAULT_MXFP8_BWD_CONFIG)
        self.assertEqual(
            bwd_config_id(restored),
            bwd_config_id(DEFAULT_MXFP8_BWD_CONFIG),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "winner.json"
            path.write_text(
                json.dumps({"config": bwd_config_to_dict(restored)}),
                encoding="utf-8",
            )
            self.assertEqual(load_mxfp8_bwd_config(path), restored)

    def test_every_search_coordinate_changes_the_config(self) -> None:
        for name, variants in BWD_SEARCH_SPACE.items():
            self.assertTrue(
                any(
                    update_bwd_config(DEFAULT_MXFP8_BWD_CONFIG, variant)
                    != DEFAULT_MXFP8_BWD_CONFIG
                    for variant in variants
                ),
                name,
            )

    def test_unimplemented_reduction_is_named_not_benchmarked_as_noop(self) -> None:
        candidate = update_bwd_config(
            DEFAULT_MXFP8_BWD_CONFIG,
            {
                "dw": {
                    "reduction": "split_fp32_workspace",
                    "split_reduction": 4,
                    "reduction_tile": 512,
                    "workspace_epilogue": "tree",
                }
            },
        )
        reason = candidate.implementation_rejection(
            MXFP8Problem(512, 1536, 1536)
        )
        self.assertIsNotNone(reason)
        self.assertIn("split_fp32_workspace", reason)

    def test_no_physical_transpose_axes_exist(self) -> None:
        self.assertFalse(any("transpose" in name for name in BWD_SEARCH_SPACE))

    def test_b_only_coordinate_crosses_to_independent_quantizers(self) -> None:
        tuner = object.__new__(BwdCoordinateDescentTuner)
        candidate = tuner._candidate(
            DEFAULT_MXFP8_BWD_CONFIG,
            "dx_b_vector",
            {"dx": {"quant_b": {"quant_vec": 2, "load_bits": 32}}},
        )
        self.assertEqual(candidate.dx.quant_launches, "separate")
        self.assertEqual(candidate.dx.quant_a.quant_vec, 4)
        self.assertEqual(candidate.dx.resolved_quant_b().quant_vec, 2)

    def test_mixed_dual_coordinate_preserves_specialized_schedules(self) -> None:
        candidate = update_bwd_config(
            DEFAULT_MXFP8_BWD_CONFIG,
            {"dx": {"quant_launches": "dual"}},
        )
        self.assertEqual(candidate.dx.quant_launches, "dual")
        self.assertNotEqual(candidate.dx.quant_a, candidate.dx.resolved_quant_b())
        self.assertIsNone(
            candidate.implementation_rejection(MXFP8Problem(512, 1536, 1536))
        )

    def test_mixed_dual_b_coordinate_remains_fused(self) -> None:
        tuner = object.__new__(BwdCoordinateDescentTuner)
        dual = update_bwd_config(
            DEFAULT_MXFP8_BWD_CONFIG,
            {"dx": {"quant_launches": "dual"}},
        )
        candidate = tuner._candidate(
            dual,
            "dx_b_registers",
            {"dx": {"quant_b": {"maxrregcount": 160}}},
        )
        self.assertEqual(candidate.dx.quant_launches, "dual")
        self.assertEqual(candidate.dx.resolved_quant_b().maxrregcount, 160)

    def test_long_dw_reduction_requires_explicit_fp32_strategy(self) -> None:
        invalid = replace(
            DEFAULT_MXFP8_BWD_CONFIG,
            dw=replace(
                DEFAULT_MXFP8_BWD_CONFIG.dw,
                reduction="full_fp32",
                split_reduction=4,
            ),
        )
        reason = invalid.rejection(MXFP8Problem(512, 1536, 1536))
        self.assertIsNotNone(reason)
        self.assertIn("full reduction", reason)


@unittest.skipUnless(_has_sm120(), "requires an SM120/SM121 CUDA GPU")
class TestMXFP8BwdCuda(unittest.TestCase):
    def test_cute_logical_transpose_quant_matches_contiguous_reference(self) -> None:
        rows, k = 128, 256
        source = torch.randn(k, rows, device="cuda", dtype=torch.bfloat16)
        logical = source.T
        self.assertEqual(logical.stride(), (1, rows))
        self.assertEqual(logical.data_ptr(), source.data_ptr())
        self.assertEqual(
            logical.untyped_storage().data_ptr(),
            source.untyped_storage().data_ptr(),
        )
        config = MXFP8QuantConfig(
            quant_vec=4,
            load_bits=64,
            quant_math="bf16x2",
            quant_amax="bf16_bits",
            scale_layout="row_major",
        )
        quantized = torch.empty(
            rows, k, device="cuda", dtype=torch.float8_e4m3fn
        )
        scales = torch.empty(
            rows, k // 32, device="cuda", dtype=torch.float8_e8m0fnu
        )
        reference_q = torch.empty_like(quantized)
        reference_s = torch.empty_like(scales)
        compile_mxfp8_transposed_quant(rows, k, config)(
            logical, quantized, scales
        )
        compile_mxfp8_quant(rows, k, config)(
            logical.contiguous(), reference_q, reference_s
        )
        torch.cuda.synchronize()
        self.assertTrue(
            torch.equal(quantized.view(torch.uint8), reference_q.view(torch.uint8))
        )
        self.assertTrue(torch.equal(scales.view(torch.uint8), reference_s.view(torch.uint8)))

    def test_dual_dw_quantization_uses_two_logical_views(self) -> None:
        torch.manual_seed(11)
        x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        grad_output = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        config = replace(
            DEFAULT_MXFP8_BWD_CONFIG,
            dw=replace(
                DEFAULT_MXFP8_BWD_CONFIG.dw,
                quant_launches="dual",
            ),
        )
        self.assertIsNone(
            config.implementation_rejection(MXFP8Problem(128, 128, 128))
        )
        _grad_x, grad_weight = mxfp8_linear_backward(
            grad_output, x, weight, config=config
        )
        torch.cuda.synchronize()
        expected_weight = grad_output.float().T @ x.float()
        relative_weight = (
            (grad_weight.float() - expected_weight).norm()
            / expected_weight.norm()
        )
        self.assertLess(float(relative_weight), 0.06)

    def test_dual_dx_quantization_mixes_row_and_logical_view(self) -> None:
        torch.manual_seed(13)
        x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        grad_output = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        config = update_bwd_config(
            DEFAULT_MXFP8_BWD_CONFIG,
            {"dx": {"quant_launches": "dual"}},
        )
        self.assertIsNone(
            config.implementation_rejection(MXFP8Problem(128, 128, 128))
        )
        grad_x, _grad_weight = mxfp8_linear_backward(
            grad_output, x, weight, config=config
        )
        torch.cuda.synchronize()
        expected_x = grad_output.float() @ weight.float()
        relative_x = (grad_x.float() - expected_x).norm() / expected_x.norm()
        self.assertLess(float(relative_x), 0.06)

    def test_backward_matches_fp32_gemm_with_mxfp8_error(self) -> None:
        torch.manual_seed(7)
        x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        grad_output = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        grad_x, grad_weight = mxfp8_linear_backward(grad_output, x, weight)
        torch.cuda.synchronize()
        expected_x = grad_output.float() @ weight.float()
        expected_weight = grad_output.float().T @ x.float()
        relative_x = (grad_x.float() - expected_x).norm() / expected_x.norm()
        relative_weight = (
            (grad_weight.float() - expected_weight).norm()
            / expected_weight.norm()
        )
        self.assertLess(float(relative_x), 0.06)
        self.assertLess(float(relative_weight), 0.06)

    def test_calibrated_harness_measures_and_races_backward(self) -> None:
        protocol = BenchmarkProtocol(
            warmup_calls=1,
            samples=3,
            confirm_samples=3,
            race_rounds=3,
            target_batch_ms=1.0,
            max_calls_per_sample=128,
            correctness_rtol=0.07,
            correctness_atol=1.0,
            practical_threshold=0.0,
            bootstrap_resamples=100,
            telemetry=False,
        )
        harness = BwdBenchmarkHarness(
            ShapeSpec(128, 128, 128), protocol, seed=17
        )
        dual = update_bwd_config(
            DEFAULT_MXFP8_BWD_CONFIG,
            {
                "dx": {"quant_launches": "dual"},
                "dw": {"quant_launches": "dual"},
            },
        )
        measurement = harness.measure(
            dual, samples=3, seed=17, components=True
        )
        self.assertEqual(measurement["status"], "ok")
        self.assertEqual(len(measurement["timings_ms"]), 3)
        self.assertIn("dx_quant", measurement["components"])
        race = harness.race(DEFAULT_MXFP8_BWD_CONFIG, dual, seed=18)
        self.assertEqual(race["status"], "ok")
        self.assertEqual(len(race["incumbent_timings_ms"]), 3)


if __name__ == "__main__":
    unittest.main()
