import unittest

import torch
from torch import nn

from rtx.calibration import NVFP4CalibrationCollector
from rtx.nvfp4_calibration import local_second_moments


class CalibrationCollectionCpuTests(unittest.TestCase):
    @staticmethod
    def linear(k=17, n=4):
        return nn.Linear(k, n, bias=False, dtype=torch.bfloat16)

    def test_streaming_bounded_even_rows_root_and_uncentered_moments(self):
        model = self.linear()
        first = torch.arange(170).reshape(2, 5, 17).bfloat16()
        second = torch.arange(68).reshape(4, 17).bfloat16()
        collector = NVFP4CalibrationCollector(model, max_tokens_per_batch=3)
        with collector as calibration:
            model(first)
            model(input=second)
        expected = torch.cat((first.flatten(0, 1)[[0, 4, 9]], second[[0, 1, 3]]))
        self.assertEqual(calibration.sample_counts, {"": 6})
        torch.testing.assert_close(calibration.second_moments()[""], local_second_moments(expected), rtol=0, atol=0)
        self.assertEqual(calibration.second_moments()[""].dtype, torch.float32)
        self.assertEqual(calibration.second_moments()[""].device.type, "cpu")
        self.assertEqual(len(model._forward_pre_hooks), 0)
        # Returned moments/count maps cannot mutate the accumulated evidence.
        calibration.second_moments()[""].zero_()
        calibration.sample_counts[""] = -1
        self.assertEqual(calibration.sample_counts[""], 6)
        self.assertGreater(float(calibration.second_moments()[""].sum()), 0)

    def test_shared_alias_one_hook_combined_invocations_canonical_name(self):
        shared = self.linear(4, 4)
        class Aliased(nn.Module):
            def __init__(self):
                super().__init__()
                self.left = shared
                self.right = shared
            def forward(self, x):
                return self.left(x) + self.right(x + 1)
        model = Aliased()
        value = torch.ones(2, 4, dtype=torch.bfloat16)
        with NVFP4CalibrationCollector(model) as calibration:
            self.assertEqual(len(shared._forward_pre_hooks), 1)
            model(value)
        self.assertEqual(calibration.sample_counts, {"left": 4})
        expected = local_second_moments(torch.cat((value, value + 1)))
        torch.testing.assert_close(calibration.second_moments()["left"], expected, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "some aliases"):
            NVFP4CalibrationCollector(model, filter_fn=lambda module, name: name == "left")

    def test_does_not_change_mode_parameters_or_surrounding_gradient_mode(self):
        model = nn.Sequential(self.linear(4, 4), self.linear(4, 4))
        model.train()
        model[1].eval()
        parameters = tuple(model.parameters())
        modes = [module.training for module in model.modules()]
        original_weights = [parameter.detach().clone() for parameter in parameters]
        x = torch.ones(2, 4, dtype=torch.bfloat16, requires_grad=True)
        with NVFP4CalibrationCollector(model) as calibration:
            self.assertTrue(torch.is_grad_enabled())
            output = model(x)
            self.assertTrue(output.requires_grad)
            self.assertTrue(torch.is_grad_enabled())
        output.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual([module.training for module in model.modules()], modes)
        for current, original, saved in zip(model.parameters(), parameters, original_weights):
            self.assertIs(current, original)
            torch.testing.assert_close(current, saved, rtol=0, atol=0)
        self.assertEqual(set(calibration.second_moments()), {"0", "1"})
        self.assertFalse(any(value.requires_grad for value in calibration.second_moments().values()))

    def test_results_not_available_before_during_or_after_failed_context(self):
        model = self.linear(4, 4)
        calibration = NVFP4CalibrationCollector(model)
        with self.assertRaisesRegex(RuntimeError, "successfully completed"):
            calibration.second_moments()
        with calibration:
            with self.assertRaisesRegex(RuntimeError, "active"):
                calibration.second_moments()
            with self.assertRaisesRegex(RuntimeError, "active"):
                _ = calibration.sample_counts
            model(torch.ones(1, 4, dtype=torch.bfloat16))
        with self.assertRaisesRegex(RuntimeError, "single-use"):
            with calibration:
                pass
        broken = NVFP4CalibrationCollector(model)
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with broken:
                model(torch.ones(1, 4, dtype=torch.bfloat16))
                raise RuntimeError("injected")
        self.assertEqual(len(model._forward_pre_hooks), 0)
        with self.assertRaisesRegex(RuntimeError, "invalid"):
            broken.second_moments()
        with self.assertRaisesRegex(RuntimeError, "single-use"):
            with broken:
                pass

    def test_invalid_hook_error_caught_inside_context_still_invalidates(self):
        model = self.linear(4, 4)
        with NVFP4CalibrationCollector(model) as calibration:
            model(torch.ones(1, 4, dtype=torch.bfloat16))
            with self.assertRaisesRegex(TypeError, "BF16 tensor"):
                model(torch.ones(1, 4))
        self.assertEqual(len(model._forward_pre_hooks), 0)
        with self.assertRaisesRegex(RuntimeError, "invalid"):
            calibration.second_moments()

    def test_unobserved_layers_reject_all_partial_results(self):
        model = nn.ModuleList([self.linear(4, 4), self.linear(4, 4)])
        with NVFP4CalibrationCollector(model) as calibration:
            model[0](torch.ones(1, 4, dtype=torch.bfloat16))
        with self.assertRaisesRegex(RuntimeError, "unobserved: 1"):
            calibration.second_moments()
        with self.assertRaisesRegex(RuntimeError, "invalid"):
            _ = calibration.sample_counts

    def test_filter_validation_and_no_selection(self):
        with self.assertRaisesRegex(NotImplementedError, "bias"):
            NVFP4CalibrationCollector(nn.Linear(4, 4, dtype=torch.bfloat16))
        with self.assertRaisesRegex(TypeError, "BF16 weight"):
            NVFP4CalibrationCollector(nn.Linear(4, 4, bias=False))
        with self.assertRaisesRegex(TypeError, "only replace nn.Linear"):
            NVFP4CalibrationCollector(nn.ReLU(), filter_fn=lambda *_: True)
        with self.assertRaisesRegex(TypeError, "nn.Module"):
            NVFP4CalibrationCollector(None)
        for limit in (0, -1, 1.5, True):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                NVFP4CalibrationCollector(self.linear(), max_tokens_per_batch=limit)
        with NVFP4CalibrationCollector(nn.Linear(4, 4), filter_fn=lambda *_: False) as calibration:
            pass
        self.assertEqual(calibration.second_moments(), {})
        self.assertEqual(calibration.sample_counts, {})

    def test_nonfinite_and_empty_inputs_invalidate_and_remove_hooks(self):
        for value in (torch.full((2, 4), float("nan"), dtype=torch.bfloat16), torch.empty(0, 4, dtype=torch.bfloat16)):
            model = self.linear(4, 4)
            calibration = NVFP4CalibrationCollector(model)
            with self.assertRaises(ValueError):
                with calibration:
                    model(value)
            self.assertEqual(len(model._forward_pre_hooks), 0)
            with self.assertRaisesRegex(RuntimeError, "invalid"):
                calibration.second_moments()


if __name__ == "__main__":
    unittest.main()
