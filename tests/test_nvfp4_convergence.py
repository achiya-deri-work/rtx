"""CPU contracts for the synthetic training-check driver (no CUDA launch)."""

from dataclasses import replace
import unittest

from benchmarks.validate_nvfp4_convergence import _adaptive_configs, _median_window


class ConvergenceDriverTests(unittest.TestCase):
    def test_explicit_headroom_keeps_exact_control(self):
        for batch, features in ((128, 64), (256, 128)):
            configs = _adaptive_configs(batch, features)
            self.assertEqual(set(configs), {
                "amax", "lsq1", "mse5", "four_over_six", "four_over_six_headroom",
            })
            exact = configs["four_over_six"]
            headroom = configs["four_over_six_headroom"]
            self.assertEqual(exact.tensor_scale_mode, "exact")
            self.assertEqual(headroom.tensor_scale_mode, "exact_four_over_six")
            self.assertEqual(replace(headroom, tensor_scale_mode="exact"), exact)
            for config in configs.values():
                self.assertEqual(config.quant.values_per_lane, 16)
                self.assertEqual(config.quant.load_bits, 128)
                self.assertEqual(config.x_scale_region_rows, 8)
                self.assertEqual(config.weight_scale_region_rows, 8)

    def test_final_window_uses_last_five_or_available(self):
        self.assertEqual(_median_window([3., 1.]), 2.)
        self.assertEqual(_median_window([100., 1., 2., 3., 4., 5.]), 3.)


if __name__ == "__main__":
    unittest.main()
