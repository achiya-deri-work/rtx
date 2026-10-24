import unittest

import torch

from benchmarks.study_nvfp4_precision import fp4_round, quantize


class NVFP4PrecisionStudyTests(unittest.TestCase):
    def test_ties_even_and_saturation(self):
        x = torch.tensor([0., .25, .75, 1.25, 1.75, 2.5, 3.5, 5., 8.])
        y = torch.tensor([0., 0., 1., 1., 2., 2., 4., 4., 6.])
        torch.testing.assert_close(fp4_round(x), y, rtol=0, atol=0)
        torch.testing.assert_close(fp4_round(-x), -y, rtol=0, atol=0)

    def test_search_includes_baseline_on_ragged_inputs(self):
        g = torch.Generator().manual_seed(42)
        x = torch.randn(7, 33, generator=g).bfloat16()
        for rows in (1, 4, 8):
            base = quantize(x, rows, 'exact')
            searched = quantize(x, rows, 'exact', search=True)
            self.assertLessEqual(float((searched-x).square().sum()),
                                 float((base-x).square().sum()))
            self.assertEqual(searched.shape, x.shape)

    def test_zero_is_exact(self):
        x = torch.zeros(3, 17)
        for mode in ('power2', 'exact', 'block'):
            torch.testing.assert_close(quantize(x, 4, mode, search=True), x)

    def test_refinement_accepts_only_measured_improvements(self):
        x = torch.randn(7, 33, generator=torch.Generator().manual_seed(53)).bfloat16()
        for rows in (1,4,8):
            previous = float((quantize(x,rows,'exact')-x).square().sum())
            for steps in (1,2):
                q = quantize(x,rows,'exact',refine_steps=steps)
                error = float((q-x).square().sum())
                self.assertLessEqual(error,previous)
                previous=error
        torch.testing.assert_close(quantize(torch.zeros(3,17),4,'exact',refine_steps=2),torch.zeros(3,17))


if __name__ == '__main__':
    unittest.main()
