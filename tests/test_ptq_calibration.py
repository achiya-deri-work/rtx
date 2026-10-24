"""Explicit offline NVFP4 PTQ integration and atomic FQN-plan validation."""

from dataclasses import FrozenInstanceError
import unittest
from unittest.mock import patch

import torch
from torch import nn

import rtx.ptq as ptq
from rtx.nvfp4_calibration import calibrate_nvfp4_weight


class NVFP4PTQCalibrationCpuTests(unittest.TestCase):
    def setUp(self):
        # Production's CUDA requirement is unchanged. CPU tests replace only
        # device admission/native quantization, not model selection or planning.
        selected = ptq._selected_linears
        self.selection = patch.object(ptq, "_selected_linears", side_effect=lambda model, filter_fn, **kwargs:
                                      selected(model, filter_fn, require_cuda=False))
        self.selection.start()
        self.addCleanup(self.selection.stop)
        self.native = patch("rtx.fp4.quantize_nvfp4", side_effect=lambda weight, *, tensor_scale, config:
                            calibrate_nvfp4_weight(weight, "max", outer_scale=tensor_scale))
        self.quantize = self.native.start()
        self.addCleanup(self.native.stop)

    @staticmethod
    def linear(k=17):
        return nn.Linear(k, 3, bias=False, dtype=torch.bfloat16).eval()

    @staticmethod
    def moments(k=17):
        return torch.eye(16).repeat((k + 15) // 16, 1, 1)

    def test_calibration_config_frozen_and_validated(self):
        config = ptq.NVFP4CalibrationConfig()
        self.assertEqual(config.method, "mse")
        with self.assertRaises(FrozenInstanceError):
            config.method = "max"
        for kwargs in ({"method": "unknown"}, {"hessian_shrinkage": .5},
                       {"method": "local_hessian", "hessian_shrinkage": float("nan")},
                       {"method": "local_hessian", "hessian_shrinkage": float("inf")},
                       {"method": "local_hessian", "hessian_shrinkage": -1},
                       {"method": "local_hessian", "hessian_shrinkage": 1.1},
                       {"rows_per_chunk": 0}, {"rows_per_chunk": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ptq.NVFP4CalibrationConfig(**kwargs)
        with self.assertRaises(TypeError):
            ptq.NVFP4WeightOnlyConfig(weight_calibration="mse")
        with self.assertRaises(ValueError):
            ptq.NVFP4WeightOnlyConfig(weight_calibration=config, quant_config=object())

    def test_default_keeps_native_quantizer_and_config(self):
        from rtx.configs.nvfp4 import NVFP4QuantConfig
        config = NVFP4QuantConfig()
        with patch("rtx.nvfp4_calibration.calibrate_nvfp4_weight", side_effect=AssertionError("unexpected offline calibration")):
            packed = ptq.quantize_(self.linear(), ptq.NVFP4WeightOnlyConfig(quant_config=config, autotune="off"))
        self.assertEqual(packed.weight_mode, "prequantized")
        self.quantize.assert_called_once()
        self.assertIs(self.quantize.call_args.kwargs["config"], config)
        self.assertIsNone(self.quantize.call_args.kwargs["tensor_scale"])

    def test_all_methods_preserve_block_outer_and_native_module(self):
        from rtx.fp4 import NVFP4Linear
        for method in ("max", "four_over_six", "mse", "local_hessian"):
            with self.subTest(method=method):
                source = self.linear()
                weight = source.weight.detach().clone()
                calibration = ptq.NVFP4CalibrationConfig(method=method, rows_per_chunk=2,
                                                        hessian_shrinkage=.5 if method == "local_hessian" else 0.)
                kwargs = {"calibration_moments": {"": self.moments()}} if method == "local_hessian" else {}
                result = ptq.quantize_(source, ptq.NVFP4WeightOnlyConfig(scaling="block", weight_calibration=calibration,
                                                                       autotune="off"), **kwargs)
                expected = calibrate_nvfp4_weight(weight, method, outer_scale=torch.tensor(1.), rows_per_chunk=2,
                                                  second_moments=self.moments() if method == "local_hessian" else None,
                                                  hessian_shrinkage=calibration.hessian_shrinkage)
                self.assertIsInstance(result, NVFP4Linear)
                self.assertIsNone(result.weight)
                self.assertEqual(result.scaling, "block")
                self.assertEqual(float(result.weight_tensor_scale), 1.)
                torch.testing.assert_close(result.weight_data, expected.qdata, rtol=0, atol=0)
                torch.testing.assert_close(result.weight_block_scales.float(), expected.scale.float(), rtol=0, atol=0)
        self.quantize.assert_not_called()

    def test_per_layer_overrides_and_nonhessian_moments(self):
        model = nn.Sequential(self.linear(), self.linear())
        config = ptq.NVFP4WeightOnlyConfig(weight_calibration=ptq.NVFP4CalibrationConfig("mse"))
        local = ptq.NVFP4CalibrationConfig("local_hessian", hessian_shrinkage=.5)
        with patch("rtx.nvfp4_calibration.calibrate_nvfp4_weight", wraps=calibrate_nvfp4_weight) as calibrate:
            ptq.quantize_(model, config, calibration_overrides={"1": local},
                          calibration_moments={"0": self.moments(), "1": self.moments()})
        self.assertEqual([call.args[1] for call in calibrate.call_args_list], ["mse", "local_hessian"])
        self.assertIsNone(calibrate.call_args_list[0].kwargs["second_moments"])
        self.assertEqual(calibrate.call_args_list[1].kwargs["hessian_shrinkage"], .5)
        self.quantize.assert_not_called()

    def test_override_only_plan_preserves_unplanned_native_layers(self):
        model = nn.Sequential(self.linear(), self.linear())
        ptq.quantize_(model, ptq.NVFP4WeightOnlyConfig(),
                      calibration_overrides={"1": ptq.NVFP4CalibrationConfig("mse")})
        self.quantize.assert_called_once()
        self.assertEqual(model[0].weight_mode, "prequantized")
        self.assertEqual(model[1].weight_mode, "prequantized")

    def test_invalid_plans_fail_before_any_replacement_or_packing(self):
        config = ptq.NVFP4WeightOnlyConfig(weight_calibration=ptq.NVFP4CalibrationConfig("local_hessian"))
        scenarios = (
            {"calibration_moments": {"0": self.moments()}},
            {"calibration_moments": {"0": self.moments(), "1": torch.zeros(2, 16)}},
            {"calibration_moments": {"0": self.moments(), "1": self.moments() * float("nan")}},
            {"calibration_moments": {"missing": self.moments()}},
            {"calibration_overrides": {"missing": ptq.NVFP4CalibrationConfig("max")}},
            {"calibration_overrides": {"0": None}},
        )
        for kwargs in scenarios:
            model = nn.Sequential(self.linear(), self.linear())
            originals = tuple(model)
            with patch("rtx.nvfp4_calibration.calibrate_nvfp4_weight") as calibrate:
                with self.subTest(kwargs=tuple(kwargs)), self.assertRaises((ValueError, TypeError)):
                    ptq.quantize_(model, config, **kwargs)
                calibrate.assert_not_called()
            self.assertEqual(tuple(model), originals)
        self.quantize.assert_not_called()

    def test_strict_format_and_quantizer_conflicts(self):
        for kwargs in ({"calibration_moments": {}}, {"calibration_overrides": {}}):
            with self.assertRaises(TypeError):
                ptq.quantize_(self.linear(), ptq.MXFP8WeightOnlyConfig(), **kwargs)
        with self.assertRaises(ValueError):
            ptq.quantize_(self.linear(), ptq.NVFP4WeightOnlyConfig(), calibration_moments={})
        model = nn.Sequential(self.linear(), self.linear())
        originals = tuple(model)
        with self.assertRaisesRegex(ValueError, "quant_config conflicts"):
            ptq.quantize_(model, ptq.NVFP4WeightOnlyConfig(quant_config=object()),
                          calibration_overrides={"1": ptq.NVFP4CalibrationConfig("mse")})
        self.assertEqual(tuple(model), originals)
        self.quantize.assert_not_called()

    def test_unselected_keys_and_invalid_unused_moments_are_rejected(self):
        config = ptq.NVFP4WeightOnlyConfig(weight_calibration=ptq.NVFP4CalibrationConfig("mse"))
        model = nn.Sequential(self.linear(), self.linear())
        originals = tuple(model)
        with self.assertRaisesRegex(ValueError, "unknown or unselected"):
            ptq.quantize_(model, config, filter_fn=lambda module, fqn: fqn == "0",
                          calibration_moments={"1": self.moments()})
        with self.assertRaisesRegex(ValueError, "finite"):
            ptq.quantize_(model, config, calibration_moments={"0": self.moments()*float("nan")})
        self.assertEqual(tuple(model), originals)
        self.quantize.assert_not_called()

    def test_packing_failure_does_not_install_earlier_replacements(self):
        model = nn.Sequential(self.linear(), self.linear())
        originals = tuple(model)
        first_packed = calibrate_nvfp4_weight(model[0].weight, "mse")
        with patch("rtx.nvfp4_calibration.calibrate_nvfp4_weight", side_effect=[first_packed, RuntimeError("test failure")]):
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                ptq.quantize_(model, ptq.NVFP4WeightOnlyConfig(weight_calibration=ptq.NVFP4CalibrationConfig("mse")))
        self.assertEqual(tuple(model), originals)
        self.quantize.assert_not_called()

    def test_shared_nested_aliases_consistent_plan_and_moments(self):
        model = nn.Module()
        model.a = nn.Sequential(self.linear())
        model.b = model.a
        original = model.a[0]
        config = ptq.NVFP4WeightOnlyConfig(weight_calibration=ptq.NVFP4CalibrationConfig("mse"))
        local = ptq.NVFP4CalibrationConfig("local_hessian")
        with self.assertRaisesRegex(ValueError, "some aliases"):
            ptq.quantize_(model, config, filter_fn=lambda module, fqn: fqn == "a.0")
        with self.assertRaisesRegex(ValueError, "conflicting calibration policies"):
            ptq.quantize_(model, config, calibration_overrides={"a.0": local})
        with self.assertRaisesRegex(ValueError, "conflicting calibration moments"):
            ptq.quantize_(model, config, calibration_moments={"a.0": self.moments(), "b.0": 2*self.moments()})
        self.assertIs(model.a[0], original)
        with patch("rtx.nvfp4_calibration.calibrate_nvfp4_weight", wraps=calibrate_nvfp4_weight) as calibrate:
            ptq.quantize_(model, config, calibration_overrides={"a.0": local, "b.0": local},
                          calibration_moments={"b.0": self.moments()})
        self.assertIs(model.a[0], model.b[0])
        calibrate.assert_called_once()
        self.quantize.assert_not_called()

    def test_shared_direct_alias_equal_moments_accepted(self):
        model = nn.Module()
        model.left = self.linear()
        model.right = model.left
        ptq.quantize_(model, ptq.NVFP4WeightOnlyConfig(weight_calibration=ptq.NVFP4CalibrationConfig("local_hessian")),
                      calibration_moments={"left": self.moments(), "right": self.moments().clone()})
        self.assertIs(model.left, model.right)


@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12,
                     "requires an SM12x CUDA GPU")
class NVFP4PTQCalibrationCudaTests(unittest.TestCase):
    def test_calibrated_conversion_is_native_and_fullgraph_compilable(self):
        from rtx.fp4 import NVFP4Linear
        for scaling, k in (("current", 128), ("block", 129)):
            with self.subTest(scaling=scaling, k=k):
                torch._dynamo.reset()
                model = nn.Sequential(nn.Linear(k, 33, bias=False, device="cuda", dtype=torch.bfloat16)).eval()
                moments = torch.eye(16).repeat((k+15)//16, 1, 1)
                ptq.quantize_(model, ptq.NVFP4WeightOnlyConfig(scaling=scaling, autotune="off",
                              weight_calibration=ptq.NVFP4CalibrationConfig("local_hessian", .5)),
                              calibration_moments={"0": moments})
                self.assertIsInstance(model[0], NVFP4Linear)
                self.assertIsNone(model[0].weight)
                if scaling == "block":
                    self.assertEqual(float(model[0].weight_tensor_scale), 1.)
                x = torch.randn(17, k, device="cuda", dtype=torch.bfloat16)
                with torch.inference_mode():
                    expected = model(x).clone()
                    compiled = torch.compile(model, fullgraph=True, dynamic=False,
                                             options={"triton.cudagraphs": False})
                    actual = compiled(x)
                self.assertTrue(bool(actual.isfinite().all()))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.cuda.synchronize()
                torch._dynamo.reset()


if __name__ == "__main__":
    unittest.main()
