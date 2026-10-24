import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
import torch

from benchmarks.evaluate_nvfp4_dinov3_cifar import POLICIES, SELECTION_CANDIDATES, capture_calibration_inputs, capture_first_inputs, convert, decode_native_packed, move_native_packed, quantize_reference, save_frozen_selection, select_calibrated_weight, select_indices, select_window_indices, task_metrics
from benchmarks.diagnose_dinov3_compile_drift import clone_tree, compare_trees


class CifarPolicyCpuTests(unittest.TestCase):
    def test_diagnostic_tree_clone_does_not_alias_reference(self):
        source = {"tensor": torch.ones(2, 4), "optional": None, "nested": (torch.zeros(1, 3),)}
        reference = clone_tree(source, cpu=True)
        source["tensor"].zero_()
        self.assertEqual(float(reference["tensor"].sum()), 8.)
        checks = compare_trees(reference, reference)
        self.assertEqual(len(checks), 2)
        self.assertTrue(all(check["exact_equal"] for check in checks))
        self.assertTrue(all(check["normalized_rms"] == 0 for check in checks))
        with self.assertRaisesRegex(ValueError, "structures"):
            compare_trees(reference, {"different": torch.ones(2, 4)})

    def test_stratified_selection(self):
        labels = np.arange(100) % 10
        selected = select_indices(labels, 23, 17)
        self.assertEqual(selected, select_indices(labels, 23, 17))
        self.assertEqual(len(set(selected)), 23)
        counts = np.bincount(labels[selected], minlength=10)
        self.assertLessEqual(int(counts.max() - counts.min()), 1)

    def test_calibration_train_selection_excludes_gallery(self):
        labels = np.arange(100) % 10
        gallery = select_indices(labels, 30, 17)
        calibration = select_indices(labels, 25, 19, exclude=gallery)
        self.assertFalse(set(gallery) & set(calibration))
        self.assertEqual(calibration, select_indices(labels, 25, 19, exclude=gallery))
        with self.assertRaises(ValueError):
            select_indices(labels, 71, 19, exclude=gallery)

    def test_test_offset_excludes_exact_prior_stratified_prefix(self):
        labels = np.arange(100) % 10
        first = select_indices(labels, 20, 17)
        fresh, excluded = select_window_indices(labels, 30, 17, offset=20)
        self.assertEqual(excluded, first)
        self.assertEqual(fresh, select_indices(labels, 50, 17)[20:])
        self.assertFalse(set(fresh) & set(excluded))
        for offset in (-1, 100):
            with self.assertRaises(ValueError):
                select_window_indices(labels, 2, 17, offset=offset)

    @staticmethod
    def tiny_calibration_model():
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.Sequential(torch.nn.Linear(4, 4, dtype=torch.bfloat16))

            def forward_features(self, value):
                return self.blocks(value)
        return Tiny().eval()

    def test_calibration_uses_every_image_bounded_tokens_and_uncentered_moments(self):
        model = self.tiny_calibration_model()
        images = torch.arange(60).reshape(5, 3, 4).float()
        captured = capture_calibration_inputs(model, images, 2, "cpu", 7, expected_modules=1)
        data = captured["blocks.0"]
        chosen = torch.stack([images[0, 0], images[0, 2], images[1, 0], images[1, 2], images[2, 0], images[3, 0], images[4, 0]]).bfloat16().float()
        expected = chosen.T @ chosen / len(chosen)
        self.assertTrue(torch.equal(data["second_moments"][0, :4, :4], expected))
        self.assertTrue(torch.equal(data["second_moments"][0, 4:], torch.zeros(12, 16)))
        self.assertEqual(data["metadata"]["images"], 5)
        self.assertEqual(data["metadata"]["sampled_rows"], 7)
        self.assertTrue(data["metadata"]["no_padding"])
        self.assertEqual(len(model.blocks[0]._forward_pre_hooks), 0)
        # Capture and moments are batch partition invariant.
        again = capture_calibration_inputs(model, images, 3, "cpu", 7, expected_modules=1)
        self.assertEqual(data["metadata"], again["blocks.0"]["metadata"])

    def test_calibration_validation_and_failed_forward_cleanup(self):
        model = self.tiny_calibration_model()
        with self.assertRaisesRegex(ValueError, "at least two"):
            capture_calibration_inputs(model, torch.ones(1, 3, 4), 1, "cpu", 7, expected_modules=1)
        with self.assertRaisesRegex(ValueError, "expected 48"):
            capture_calibration_inputs(model, torch.ones(2, 3, 4), 2, "cpu", 7)
        def fail(_module, _inputs, _outputs):
            raise RuntimeError("injected")
        handle = model.blocks[0].register_forward_hook(fail)
        try:
            with self.assertRaisesRegex(RuntimeError, "injected"):
                capture_calibration_inputs(model, torch.ones(3, 3, 4), 2, "cpu", 7, expected_modules=1)
            self.assertEqual(len(model.blocks[0]._forward_pre_hooks), 0)
        finally:
            handle.remove()

    def test_validation_capture_quantizes_full_batch_before_sampling_and_decodes_3d(self):
        from rtx.formats.nvfp4 import make_nvfp4_tensor
        from rtx.nvfp4_calibration import calibrate_nvfp4_weight
        seen = []
        def cpu_native_standin(value, config):
            seen.append(value.detach().clone())
            self.assertEqual(config.block_scale_selection, "amax")
            self.assertEqual(config.scale_reciprocal, "direct")
            packed = calibrate_nvfp4_weight(value.reshape(-1, value.shape[-1]), "max")
            return make_nvfp4_tensor(packed.qdata, packed.scale, packed.per_tensor_scale, tuple(value.shape))
        model = self.tiny_calibration_model()
        images = torch.arange(24).reshape(2, 3, 4).float()
        images[0, 1, 0] = 1000  # Not sampled, but must set full-batch outer.
        with patch("rtx.quantize_nvfp4", side_effect=cpu_native_standin):
            data = capture_calibration_inputs(model, images, 2, "cpu", 2, expected_modules=1, retain_activations=True, retain_quantized=True)["blocks.0"]
        self.assertEqual(len(seen), 1)
        self.assertEqual(tuple(seen[0].shape), (2, 3, 4))
        self.assertEqual(float(seen[0].max()), 1000.)
        q = cpu_native_standin(seen[0], type("Config", (), {"block_scale_selection": "amax", "scale_reciprocal": "direct"})())
        decoded = decode_native_packed(q)
        self.assertEqual(tuple(decoded.shape), (6, 4))
        torch.testing.assert_close(data["quantized_activations"], decoded[[0, 3]], rtol=0, atol=0)
        torch.testing.assert_close(data["activations"].float(), images[:, 0], rtol=0, atol=0)
        self.assertEqual(len(model.blocks[0]._forward_pre_hooks), 0)

    def test_selection_uses_full_layer_validation_score_and_fixed_candidate_order(self):
        from rtx.nvfp4_calibration import calibrate_nvfp4_weight, local_second_moments
        generator = torch.Generator().manual_seed(51)
        weight = torch.randn(3, 32, generator=generator).bfloat16()
        fit = torch.randn(15, 32, generator=generator).bfloat16()
        validation = torch.randn(11, 32, generator=generator).bfloat16()
        qx = decode_native_packed(calibrate_nvfp4_weight(validation, "max"))
        moments = local_second_moments(fit)
        packed, record = select_calibrated_weight(weight, moments, validation, decoded_x=qx)
        reference = validation.float() @ weight.float().T
        scores = []
        for policy, alpha in SELECTION_CANDIDATES:
            candidate = calibrate_nvfp4_weight(weight, policy, second_moments=moments, hessian_shrinkage=alpha)
            output = qx @ decode_native_packed(candidate).T
            scores.append(float((output.double() - reference.double()).square().sum()))
        self.assertEqual([r["full_layer_output_sse"] for r in record["candidates"]], scores)
        self.assertEqual(record["selected"], record["candidates"][scores.index(min(scores))])
        self.assertFalse(record["test_access"])
        selected_output = qx @ decode_native_packed(packed).T
        self.assertEqual(float((selected_output.double() - reference.double()).square().sum()), min(scores))
        with self.assertRaisesRegex(ValueError, "finite matching"):
            select_calibrated_weight(weight, moments, validation, decoded_x=torch.full_like(qx, float("nan")))

    def test_frozen_packed_buffer_move_and_exclusive_audit_artifact(self):
        from rtx.nvfp4_calibration import calibrate_nvfp4_weight
        packed = calibrate_nvfp4_weight(torch.arange(34).reshape(2, 17).float(), "max")
        # Moving must never dispatch the fragile TorchAO subclass .to().
        with patch.object(type(packed), "to", side_effect=RuntimeError("subclass move called")):
            moved = move_native_packed(packed, "cpu")
        self.assertEqual(moved._rtx_logical_shape, (2, 17))
        torch.testing.assert_close(decode_native_packed(moved), decode_native_packed(packed), rtol=0, atol=0)
        calibration = {"blocks.0": {"selected_weight": packed, "selection": {"policy": "max"}, "metadata": {"sampled_rows": 3}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.selection.pt"
            record = save_frozen_selection(path, calibration, checkpoint_hash="abc", fit_ids=[1, 2], validation_ids=[3, 4], outer_mode="exact")
            artifact = torch.load(path, weights_only=True)
            self.assertEqual(record["module_count"], 1)
            self.assertEqual(artifact["checkpoint_sha256"], "abc")
            self.assertTrue(artifact["frozen_before_any_heldout_forward"])
            layer = artifact["layers"]["blocks.0"]
            self.assertEqual(layer["logical_shape"], (2, 17))
            torch.testing.assert_close(layer["qdata"], packed.qdata, rtol=0, atol=0)
            torch.testing.assert_close(layer["block_scales"].float(), packed.scale.float(), rtol=0, atol=0)
            with self.assertRaises(FileExistsError):
                save_frozen_selection(path, calibration, checkpoint_hash="abc", fit_ids=[1, 2], validation_ids=[3, 4], outer_mode="exact")

    @unittest.skipUnless(torch.cuda.is_available(), "optional packed CPU/CUDA buffer relocation test")
    def test_selected_packed_buffers_move_cpu_to_cuda_without_subclass_to(self):
        from rtx.nvfp4_calibration import calibrate_nvfp4_weight
        packed = calibrate_nvfp4_weight(torch.randn(2, 17), "max")
        moved = move_native_packed(packed, "cuda:0")
        self.assertEqual(moved.qdata.device.type, "cuda")
        self.assertEqual(moved.scale.device.type, "cuda")
        torch.testing.assert_close(decode_native_packed(moved), decode_native_packed(packed), rtol=0, atol=0)

    def test_ptq_conversion_constructs_native_packed_weight_and_fixed_activation_policy(self):
        from rtx import NVFP4Linear
        for policy in ("max", "mse", "local_hessian", "four_over_six"):
            model = self.tiny_calibration_model()
            calibration = capture_calibration_inputs(model, torch.ones(3, 2, 4), 2, "cpu", 6, expected_modules=1)
            converted, config = convert(model, f"ptq:{policy}", 2, 16, 8, 8, calibration=calibration)
            core = converted.blocks[0].core
            self.assertIsInstance(core, NVFP4Linear)
            self.assertIsNone(core.weight)
            self.assertEqual(core.weight_mode, "prequantized")
            self.assertEqual(core.scaling, "current")
            self.assertEqual(core.autotune, "off")
            self.assertEqual(core.weight_prequant_config.quant_x.block_scale_selection, "amax")
            self.assertEqual(core.weight_prequant_config.quant_x.scale_reciprocal, "direct")
            self.assertEqual(config["blocks.0"]["weight_policy"], policy)
        with self.assertRaisesRegex(ValueError, "missing train-only"):
            convert(self.tiny_calibration_model(), "ptq:max", 2, 16, 8, 8)

    def test_adaptive_block_sse_not_worse_than_amax(self):
        torch.manual_seed(31)
        value = torch.randn(13, 48).bfloat16()
        baseline = quantize_reference(value, 8, "amax")
        # SSE comparison before the final BF16 cast has an exact guarantee;
        # the public BF16 reconstruction can perturb it by a rounding ulp.
        value = value.float()
        baseline = quantize_reference(value, 8, "amax")
        baseline_error = (baseline - value).square().reshape(-1, 16).sum(1)
        for policy in POLICIES:
            candidate = quantize_reference(value, 8, policy)
            self.assertEqual(candidate.shape, value.shape)
            self.assertTrue(candidate.isfinite().all())
            error = (candidate - value).square().reshape(-1, 16).sum(1)
            self.assertTrue(bool((error <= baseline_error + 1e-6).all()), policy)

    def test_zero_and_ragged(self):
        for policy in POLICIES:
            value = torch.zeros(3, 19)
            self.assertTrue(torch.equal(quantize_reference(value, 5, policy), value))

    def test_headroom_is_separate_from_policy_and_bounded(self):
        torch.manual_seed(31)
        value = torch.randn(13, 48)
        for policy in POLICIES:
            ordinary = quantize_reference(value, 8, policy)
            explicit_default = quantize_reference(value, 8, policy, "exact")
            self.assertTrue(torch.equal(ordinary, explicit_default))
            headroom = quantize_reference(value, 8, policy, "exact_four_over_six")
            self.assertFalse(torch.equal(ordinary, headroom), policy)
            self.assertTrue(headroom.isfinite().all())
            # Baseline-supported error bound is generous, not a claimed
            # cross-outer improvement: headroom can help OR hurt task quality.
            self.assertLess(float((headroom - value).abs().max()), float(value.abs().max()) / 2)
            baseline = quantize_reference(value, 8, "amax", "exact_four_over_six")
            err = (headroom - value).square().reshape(-1, 16).sum(1)
            base_err = (baseline - value).square().reshape(-1, 16).sum(1)
            self.assertTrue(bool((err <= base_err + 1e-6).all()), policy)
        with self.assertRaises(ValueError):
            quantize_reference(value, 8, "amax", "unknown")

    def test_labeled_task_is_not_just_prediction_agreement(self):
        features = torch.eye(3)
        labels = torch.tensor([1, 2, 0])
        result = task_metrics(features, features, labels, labels, [0, 0, 0])
        self.assertEqual(result["accuracy"], 1.)
        self.assertAlmostEqual(result["bf16_prediction_agreement"], 1 / 3, places=6)
        self.assertEqual(result["bf16_wrong_candidate_correct"], 2)
        self.assertEqual(result["bf16_correct_candidate_wrong"], 0)

    def test_nonfinite_task_rejected(self):
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            task_metrics(torch.full((1, 2), float("nan")), torch.ones(1, 2), torch.tensor([0]), torch.tensor([0]))

    def test_capture_is_cpu_bf16_flattened_truncated_and_removes_hook(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 5, dtype=torch.bfloat16)

            def forward_features(self, value):
                return self.linear(value)

        model = Tiny()
        images = torch.arange(24).reshape(2, 3, 4).float()
        tensor, metadata = capture_first_inputs(model, "linear", images, 2, "cpu", 4)
        self.assertEqual(tensor.shape, (4, 4))
        self.assertEqual(tensor.dtype, torch.bfloat16)
        self.assertEqual(tensor.device.type, "cpu")
        self.assertTrue(metadata["truncated"])
        self.assertEqual(metadata["original_shape"], [2, 3, 4])
        self.assertEqual(len(model.linear._forward_pre_hooks), 0)
        self.assertTrue(torch.equal(tensor, images.reshape(-1, 4)[:4].bfloat16()))

    def test_capture_removes_hook_when_forward_fails(self):
        class Broken(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 5, dtype=torch.bfloat16)

            def forward_features(self, value):
                self.linear(value)
                raise RuntimeError("injected")

        model = Broken()
        with self.assertRaisesRegex(RuntimeError, "injected"):
            capture_first_inputs(model, "linear", torch.ones(2, 4), 2, "cpu", 10)
        self.assertEqual(len(model.linear._forward_pre_hooks), 0)


if __name__ == "__main__":
    unittest.main()
