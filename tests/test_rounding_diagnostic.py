import unittest

import torch

from benchmarks.diagnose_mxfp8_rounding import round_fp64_to_bf16


class RoundingDiagnosticTests(unittest.TestCase):
    def test_captured_5060_reference_midpoint(self):
        # Exact dyadic dot product from the 257x385x256 seed-321 capture.
        value=torch.tensor([4538369 / 8388608],dtype=torch.float64)
        self.assertEqual(round_fp64_to_bf16(value).item(),.54296875)
        fp32_reference=torch.tensor([.541015625],dtype=torch.float32)
        self.assertEqual(fp32_reference.bfloat16().item(),.5390625)

    def test_midpoint_neighbors_do_not_double_round(self):
        midpoint = 1.0 + 2**-8
        values = torch.tensor([midpoint - 2**-30, midpoint, midpoint + 2**-30],
                              dtype=torch.float64)
        expected = torch.tensor([1.0, 1.0, 1.0 + 2**-7], dtype=torch.bfloat16)
        torch.testing.assert_close(round_fp64_to_bf16(values), expected, rtol=0, atol=0)
        torch.testing.assert_close(round_fp64_to_bf16(-values), -expected, rtol=0, atol=0)

    def test_odd_lower_neighbor_ties_round_up_to_even(self):
        lower = 1.0 + 2**-7
        midpoint = lower + 2**-8
        actual = round_fp64_to_bf16(torch.tensor([midpoint], dtype=torch.float64))
        self.assertEqual(actual.item(), 1.0 + 2**-6)

    def test_nonfinite_or_non_fp64_input_is_rejected(self):
        for x in (torch.tensor([1.0]), torch.tensor([float("nan")], dtype=torch.float64)):
            with self.assertRaises(ValueError):
                round_fp64_to_bf16(x)
