import unittest
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch

from benchmarks.study_nvfp4_calibration import (
    calibrate_weight, e4m3_candidates, fp4_round, local_second_moments,
    load_tensor_problem, main, make_problem,
)


class NVFP4CalibrationStudyTests(unittest.TestCase):
    def test_exact_format_codes_and_ties(self):
        scales = e4m3_candidates()
        self.assertEqual(scales.numel(), 126)
        self.assertEqual(float(scales.min()), 2**-9)
        self.assertEqual(float(scales.max()), 448.)
        torch.testing.assert_close(scales.to(torch.float8_e4m3fn).float(), scales,
                                   rtol=0, atol=0)
        x = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5., 99.])
        expected = torch.tensor([0., 1., 1., 2., 2., 4., 4., 6.])
        torch.testing.assert_close(fp4_round(x), expected, rtol=0, atol=0)
        torch.testing.assert_close(fp4_round(-x), -expected, rtol=0, atol=0)

    def test_identity_second_moment_agrees_with_mse(self):
        w = torch.randn(3, 33, generator=torch.Generator().manual_seed(2))
        h = torch.eye(16).repeat(3, 1, 1)
        mse = calibrate_weight(w, "mse")
        lh = calibrate_weight(w, "local_hessian", second_moments=h)
        torch.testing.assert_close(mse.block_scales, lh.block_scales, rtol=0, atol=0)
        torch.testing.assert_close(mse.dequantize(), lh.dequantize(), rtol=0, atol=0)

    def test_correlated_inputs_local_objective_and_heldout(self):
        w, calibration, heldout = make_problem(4, 16, 256, 9, "correlated")
        self.assertFalse(torch.equal(calibration, heldout))
        h = local_second_moments(calibration)
        mse = calibrate_weight(w, "mse").dequantize()
        lh = calibrate_weight(w, "local_hessian", second_moments=h).dequantize()
        # For exactly one K block, the local objective IS the full layer SSE.
        mse_error = ((calibration @ (mse-w).T)**2).sum()
        lh_error = ((calibration @ (lh-w).T)**2).sum()
        self.assertLess(float(lh_error), float(mse_error))
        # Generalization is measured separately, not an acceptance gate.
        self.assertTrue(bool((heldout @ lh.T).isfinite().all()))
        delta = lh - w
        quadratic = torch.einsum("oi,ij,oj->", delta, h[0], delta)
        torch.testing.assert_close(quadratic, lh_error / calibration.shape[0])

    def test_zeros_and_ragged_all_policies(self):
        g = torch.Generator().manual_seed(7)
        for w in (torch.zeros(3, 17), torch.randn(3, 17, generator=g)):
            x = torch.randn(128, 17, generator=g)
            h = local_second_moments(x)
            self.assertEqual(h.shape, (2, 16, 16))
            self.assertEqual(float(h[1, 1:].abs().max()), 0.)
            base = calibrate_weight(w, "max").dequantize()
            for policy in ("max", "four_over_six", "mse", "local_hessian"):
                q = calibrate_weight(w, policy, second_moments=h)
                decoded = q.dequantize()
                self.assertEqual(decoded.shape, w.shape)
                self.assertTrue(bool(decoded.isfinite().all()))
                self.assertTrue(bool(torch.isin(q.block_scales, e4m3_candidates()).all()))
                torch.testing.assert_close(q.outer_scale, calibrate_weight(w, "max").outer_scale)
                if policy in ("mse", "four_over_six"):
                    self.assertLessEqual(float((decoded-w).square().sum()),
                                         float((base-w).square().sum()) + 1e-7)
                if not bool(w.any()):
                    torch.testing.assert_close(decoded, w, rtol=0, atol=0)

    def test_moments_are_not_centered(self):
        x = torch.full((4, 16), 2.)
        torch.testing.assert_close(local_second_moments(x), torch.full((1, 16, 16), 4.))

    def test_explicit_scale_domains_and_lower_bounds(self):
        w = torch.randn(2, 33, generator=torch.Generator().manual_seed(4))
        w[:, 16:32] *= 1e-7
        h = local_second_moments(torch.eye(33))
        for domain, minimum, count in (("production-normal", 2**-6, 119),
                                        ("full-e4m3", 2**-9, 126)):
            self.assertEqual(e4m3_candidates(domain).numel(), count)
            for policy in ("max", "four_over_six", "mse", "local_hessian"):
                q = calibrate_weight(w, policy, second_moments=h, scale_domain=domain)
                self.assertGreaterEqual(float(q.block_scales.min()), minimum)
                torch.testing.assert_close(q.block_scales.to(torch.float8_e4m3fn).float(),
                                           q.block_scales, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            calibrate_weight(w, "max", scale_domain="bogus")

    def test_reject_invalid_inputs(self):
        for w in (torch.empty(0, 3), torch.ones(3), torch.tensor([[float("nan")]])):
            with self.assertRaises(ValueError):
                calibrate_weight(w, "mse")
        with self.assertRaises(ValueError):
            calibrate_weight(torch.ones(1, 16), "local_hessian")
        with self.assertRaises(ValueError):
            local_second_moments(torch.empty(0, 16))

    def test_supplied_tensor_file_and_provenance(self):
        w, calibration, heldout = make_problem(2, 17, 8, 3, "normal")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.pt"
            payload = dict(weight=w.bfloat16(), calibration_activations=calibration,
                           heldout_activations=heldout, provenance={"split": ["train", "test"]})
            torch.save(payload, path)
            with patch("torch.load", wraps=torch.load) as load:
                tensors, description = load_tensor_problem(path)
            self.assertTrue(load.call_args.kwargs["weights_only"])
            torch.testing.assert_close(tensors["weight"], w.bfloat16())
            self.assertEqual(description["provenance"], payload["provenance"])
            self.assertEqual(len(description["file_sha256"]), 64)
            self.assertEqual(description["tensors"]["weight"]["shape"], [2, 17])
            self.assertFalse(description["calibration_heldout_identical"])
            payload["heldout_activations"] = calibration
            torch.save(payload, path)
            self.assertTrue(load_tensor_problem(path)[1]["calibration_heldout_identical"])
            del payload["heldout_activations"]
            torch.save(payload, path)
            with self.assertRaises(ValueError):
                load_tensor_problem(path)

    def test_supplied_cli_preserves_full_tensor_outer_for_row_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            path, output = Path(directory) / "input.pt", Path(directory) / "results.jsonl"
            torch.save(dict(weight=torch.tensor([[1.] * 17, [2688.] * 17]),
                            calibration_activations=torch.ones(3, 17),
                            heldout_activations=torch.full((4, 17), 2.),
                            provenance={"note": "unit fixture, not independent"}), path)
            with patch("sys.argv", ["study", "--input-tensors", str(path), "--output", str(output),
                                    "--weight-rows", "1"]), redirect_stdout(io.StringIO()):
                main()
            records = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(records), 6)
            self.assertEqual(records[1]["evaluated_weight_shape"], [1, 17])
            self.assertEqual(records[1]["tensors"]["weight"]["shape"], [2, 17])
            for record in records[2:]:
                self.assertEqual(record["outer_scale"], 1.)
                self.assertEqual(record["scale_domain"], "production-normal")
                self.assertTrue(record["heldout_output"]["finite"])


if __name__ == "__main__":
    unittest.main()
