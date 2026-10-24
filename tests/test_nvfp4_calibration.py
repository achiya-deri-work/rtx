"""Offline PTQ numerics and native nibble/layout contract; no GPU required."""

import unittest

import torch

from rtx.formats.nvfp4 import nvfp4_matrix_shape, validate_nvfp4_tensor
from rtx.nvfp4_calibration import (
    NVFP4SecondMomentCollector, calibrate_nvfp4_weight, local_second_moments,
)


def decode_native(value):
    """Independent nibble LUT decoder, including RTX logical/storage padding."""
    lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                        -0., -.5, -1., -1.5, -2., -3., -4., -6.])
    packed = value.qdata.cpu().view(torch.uint8)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()
    decoded = lut[codes].reshape(packed.shape[0], -1, 16)
    decoded *= value.scale.cpu().float()[..., None]
    decoded *= value.per_tensor_scale.cpu()
    return decoded.flatten(1)[:, :nvfp4_matrix_shape(value)[1]]


class OfflineNVFP4CalibrationTests(unittest.TestCase):
    def test_native_nibbles_ties_signs_and_padding(self):
        w = torch.tensor([[.25, .75, 1.25, 1.75, 2.5, 3.5, 5., 99.,
                           -.25, -.75, -1.25, -1.75, -2.5, -3.5, -5., -99., 0.]])
        q = calibrate_nvfp4_weight(w, "max", outer_scale=torch.tensor(16.))
        validate_nvfp4_tensor(q)
        # Force scale=1 via amax=6 and outer=1 to isolate exact RTNE/nibbles.
        w = w[:, :16].clamp(-6, 6)
        q = calibrate_nvfp4_weight(w, "max", outer_scale=torch.tensor(1.))
        self.assertEqual(q.qdata.shape, (1, 16))
        self.assertEqual(q.qdata[0, 0].item(), 0x20)
        torch.testing.assert_close(decode_native(q), torch.tensor(
            [[0., 1., 1., 2., 2., 4., 4., 6., -0., -1., -1., -2., -2., -4., -4., -6.]]),
            rtol=0, atol=0)
        self.assertEqual(q.qdata[0, 8:].count_nonzero().item(), 0)

    def test_matches_independent_study_and_fixed_outer(self):
        # Study is an independent test oracle only, never a production import.
        from benchmarks.study_nvfp4_calibration import calibrate_weight
        g = torch.Generator().manual_seed(22)
        w = torch.randn(3, 33, generator=g).bfloat16().float()
        h = local_second_moments(torch.randn(40, 33, generator=g))
        outer = torch.tensor(.01)
        for policy in ("max", "four_over_six", "mse", "local_hessian"):
            q = calibrate_nvfp4_weight(w, policy, second_moments=h, outer_scale=outer,
                                       rows_per_chunk=2)
            reference = calibrate_weight(w, policy, second_moments=h, outer_scale=outer)
            validate_nvfp4_tensor(q)
            torch.testing.assert_close(q.scale.float()[:, :3], reference.block_scales, rtol=0, atol=0)
            torch.testing.assert_close(decode_native(q), reference.dequantize(), rtol=1e-6, atol=1e-7)
            self.assertEqual(q.per_tensor_scale.dtype, torch.float32)
            self.assertEqual(float(q.per_tensor_scale), float(outer))
            self.assertGreaterEqual(q.scale.float().min().item(), 2**-6)

    def test_identity_hessian_mse_and_objective_controls(self):
        g = torch.Generator().manual_seed(7)
        w = torch.randn(2, 17, generator=g)
        h = torch.eye(16).repeat(2, 1, 1)
        mse = calibrate_nvfp4_weight(w, "mse")
        lh = calibrate_nvfp4_weight(w, "local_hessian", second_moments=h)
        torch.testing.assert_close(lh.qdata, mse.qdata, rtol=0, atol=0)
        torch.testing.assert_close(lh.scale.float(), mse.scale.float(), rtol=0, atol=0)
        maxq = calibrate_nvfp4_weight(w, "max")
        self.assertLessEqual(float((decode_native(mse) - w).square().sum()),
                             float((decode_native(maxq) - w).square().sum()) + 1e-6)
        # One block: local quadratic objective equals calibration layer SSE/N.
        w = w[:, :16]
        x = torch.randn(50, 1, generator=g).expand(-1, 16) + .1 * torch.randn(50, 16, generator=g)
        h = local_second_moments(x)
        mse = decode_native(calibrate_nvfp4_weight(w, "mse"))
        lh = decode_native(calibrate_nvfp4_weight(w, "local_hessian", second_moments=h))
        self.assertLessEqual(float((x @ (lh-w).T).square().mean()),
                             float((x @ (mse-w).T).square().mean()) + 1e-6)

    def test_streaming_moments_uncentered_and_ragged(self):
        x = torch.arange(85).reshape(5, 17).float()
        collector = NVFP4SecondMomentCollector(17)
        collector.update(x[:2].reshape(1, 2, 17))
        collector.update(x[2:])
        h = collector.second_moments()
        torch.testing.assert_close(h, local_second_moments(x), rtol=0, atol=0)
        torch.testing.assert_close(h[0], x[:, :16].T @ x[:, :16] / 5)
        self.assertEqual(h[1, 1:].count_nonzero().item(), 0)
        self.assertEqual(collector.num_samples, 5)
        torch.testing.assert_close(local_second_moments(torch.full((3, 16), 2.)),
                                   torch.full((1, 16, 16), 4.))

    def test_hessian_shrinkage_endpoints_and_zero_energy_fallback(self):
        g = torch.Generator().manual_seed(101)
        w = torch.randn(3, 33, generator=g)
        h = local_second_moments(torch.randn(30, 33, generator=g))
        h[1].zero_()
        original_h = h.clone()
        default = calibrate_nvfp4_weight(w, "local_hessian", second_moments=h)
        explicit_zero = calibrate_nvfp4_weight(w, "local_hessian", second_moments=h,
                                              hessian_shrinkage=0.)
        endpoint = calibrate_nvfp4_weight(w, "local_hessian", second_moments=h,
                                         hessian_shrinkage=1.)
        mse = calibrate_nvfp4_weight(w, "mse")
        for a, b in ((default, explicit_zero), (endpoint, mse)):
            torch.testing.assert_close(a.qdata, b.qdata, rtol=0, atol=0)
            torch.testing.assert_close(a.scale.float(), b.scale.float(), rtol=0, atol=0)
            torch.testing.assert_close(a.per_tensor_scale, b.per_tensor_scale, rtol=0, atol=0)
        halfway = calibrate_nvfp4_weight(w, "local_hessian", second_moments=h,
                                        hessian_shrinkage=.5, rows_per_chunk=1)
        torch.testing.assert_close(halfway.qdata[:, 8:16], mse.qdata[:, 8:16], rtol=0, atol=0)
        torch.testing.assert_close(halfway.scale.float()[:, 1], mse.scale.float()[:, 1], rtol=0, atol=0)
        torch.testing.assert_close(h, original_h, rtol=0, atol=0)

    def test_hessian_shrinkage_independent_exhaustive_objective(self):
        from benchmarks.study_nvfp4_calibration import e4m3_candidates, fp4_round

        g = torch.Generator().manual_seed(331)
        w = torch.randn(1, 17, generator=g)
        x = torch.randn(40, 17, generator=g) * torch.linspace(.2, 3., 17)
        h = local_second_moments(x)
        outer = torch.tensor(.01)
        blocks = torch.nn.functional.pad(w.double(), (0, 15)).reshape(2, 16)
        for alpha in (.05, .5, .95):
            packed = calibrate_nvfp4_weight(w, "local_hessian", second_moments=h,
                                            outer_scale=outer, hessian_shrinkage=alpha)
            selected = packed.scale.float()[0, :2]
            # Enumerate every representable normal E4M3 scale, but compute the
            # objective as scalar data-fit plus isotropic penalty, not by using
            # the production matrix shrinkage or its einsum decomposition.
            for block in range(2):
                reference_h = h[block].double()
                energy = reference_h.trace() / 16
                errors = []
                candidates = e4m3_candidates("production-normal")
                for scale in candidates:
                    effective = scale.double() * outer.double()
                    delta = fp4_round(blocks[block] / effective) * effective - blocks[block]
                    data_fit = sum(delta[i] * reference_h[i, j] * delta[j]
                                   for i in range(16) for j in range(16))
                    errors.append((1-alpha) * data_fit + alpha * energy * delta.square().sum())
                expected = candidates[torch.stack(errors).argmin()]
                torch.testing.assert_close(selected[block], expected, rtol=0, atol=0)
            self.assertEqual(float(packed.per_tensor_scale), float(outer))

    def test_hessian_shrinkage_validation(self):
        w = torch.ones(1, 16)
        h = torch.eye(16).unsqueeze(0)
        for alpha in (-.1, 1.1, float("nan"), float("inf"), -float("inf"), True, "0.5", None):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                calibrate_nvfp4_weight(w, "local_hessian", second_moments=h,
                                       hessian_shrinkage=alpha)
        for policy in ("max", "four_over_six", "mse"):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                calibrate_nvfp4_weight(w, policy, hessian_shrinkage=.1)
            calibrate_nvfp4_weight(w, policy, hessian_shrinkage=0.)

    def test_zero_ties_determinism_and_detachment(self):
        for k in (1, 16, 17, 32, 33):
            w = torch.zeros(2, k, requires_grad=True)
            h = torch.zeros((k + 15) // 16, 16, 16)
            for policy in ("max", "mse", "local_hessian"):
                q = calibrate_nvfp4_weight(w, policy, second_moments=h)
                validate_nvfp4_tensor(q)
                self.assertEqual(float(q.per_tensor_scale), 1.)
                self.assertFalse(q.per_tensor_scale.requires_grad)
                self.assertEqual(q.qdata.count_nonzero().item(), 0)
                torch.testing.assert_close(q.scale.float(), torch.full_like(q.scale.float(), 2**-6))
                torch.testing.assert_close(decode_native(q), w.detach(), rtol=0, atol=0)
        self.assertIsNone(w.grad)

    def test_invalid_inputs_fail_closed(self):
        for w in (torch.ones(16), torch.empty(0, 16), torch.tensor([[float("nan")]]),
                  torch.ones(2, 16, dtype=torch.int32)):
            with self.assertRaises(ValueError):
                calibrate_nvfp4_weight(w, "max")
        w = torch.ones(2, 17)
        for kwargs in ({"policy": "bogus"}, {"policy": "local_hessian"},
                       {"rows_per_chunk": 0}, {"outer_scale": torch.tensor(0.)},
                       {"outer_scale": torch.tensor(float("inf"))},
                       {"second_moments": torch.ones(2, 16)},
                       {"second_moments": -torch.eye(16).repeat(2, 1, 1)}):
            with self.assertRaises(ValueError):
                calibrate_nvfp4_weight(w, **({"policy": "max"} | kwargs))
        with self.assertRaises(ValueError):
            NVFP4SecondMomentCollector(17).second_moments()
        with self.assertRaises(ValueError):
            NVFP4SecondMomentCollector(17).update(torch.empty(0, 17))

    def test_module_state_dict_preserves_packed_buffers_cpu(self):
        import io
        from rtx.fp4 import NVFP4Linear

        w = torch.randn(3, 17, generator=torch.Generator().manual_seed(31))
        packed = calibrate_nvfp4_weight(w, "mse")
        layer = NVFP4Linear(17, 3, device="cpu", packed_weight=packed, autotune=False)
        buffer = io.BytesIO()
        torch.save(layer.state_dict(), buffer)
        buffer.seek(0)
        restored = NVFP4Linear(17, 3, device="cpu", packed_weight=calibrate_nvfp4_weight(w * 0, "max"),
                              autotune=False)
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        torch.testing.assert_close(decode_native(restored.packed_weight), decode_native(packed), rtol=0, atol=0)
        self.assertEqual(tuple(restored.packed_weight._rtx_logical_shape), (3, 17))


def _has_sm12x():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12


@unittest.skipUnless(_has_sm12x(), "requires an SM12x CUDA device")
class NativeCalibratedNVFP4Tests(unittest.TestCase):
    def test_cpu_calibrated_weight_device_move_preserves_native_storage(self):
        from unittest.mock import patch
        from rtx.fp4 import NVFP4Linear, nvfp4_linear, quantize_nvfp4
        from rtx.formats.nvfp4 import NVFP4Tensor
        from rtx.configs.nvfp4 import NVFP4WeightPrequantConfig, NVFP4FullyPrequantConfig

        m, n, k = 17, 33, 129
        g = torch.Generator().manual_seed(761)
        weight = torch.randn(n, k, generator=g).bfloat16()
        packed_cpu = calibrate_nvfp4_weight(weight, "mse")
        original = decode_native(packed_cpu)
        # TorchAO subclass.to historically attempted invalid cross-device
        # aliases. All migration must occur through the underlying buffers.
        with patch.object(NVFP4Tensor, "to", side_effect=AssertionError("subclass.to must not be used")):
            layer = NVFP4Linear(k, n, device="cuda", packed_weight=packed_cpu, autotune=False,
                               weight_prequant_config=NVFP4WeightPrequantConfig(),
                               fully_prequant_config=NVFP4FullyPrequantConfig()).eval()
            for destination in ("cuda", "cpu", "cuda"):
                layer.to(destination)
                packed = layer.packed_weight
                validate_nvfp4_tensor(packed)
                self.assertEqual(nvfp4_matrix_shape(packed), (n, k))
                self.assertEqual(packed._rtx_logical_shape, (n, k))
                self.assertEqual(packed.qdata.device.type, destination)
                self.assertEqual(packed.scale.device.type, destination)
                self.assertEqual(packed.per_tensor_scale.device.type, destination)
                self.assertEqual(packed.qdata.shape, (n, 80))
                torch.testing.assert_close(packed.qdata.cpu(), packed_cpu.qdata, rtol=0, atol=0)
                torch.testing.assert_close(packed.scale.cpu().float(), packed_cpu.scale.float(), rtol=0, atol=0)
                torch.testing.assert_close(packed.per_tensor_scale.cpu(), packed_cpu.per_tensor_scale,
                                           rtol=0, atol=0)
                torch.testing.assert_close(decode_native(packed), original, rtol=0, atol=0)
        x = torch.randn(m, k, generator=g).bfloat16().cuda()
        with torch.inference_mode():
            output = layer(x)
            packed_x = quantize_nvfp4(x)
            packed_output = nvfp4_linear(packed_x, layer.packed_weight, autotune=False,
                                         fully_prequant_config=NVFP4FullyPrequantConfig())
        torch.testing.assert_close(output, packed_output, rtol=0, atol=0)
        self.assertTrue(bool(output.isfinite().all()))
        reference = (decode_native(packed_x).double() @ original.double().T).float()
        torch.testing.assert_close(output.cpu().float(), reference.bfloat16().float(), rtol=0,
                                   atol=reference.abs().max().item() / 128 + 1e-4)
        torch.cuda.synchronize()

    def test_native_inference_and_compilation_aligned_ragged(self):
        from rtx.fp4 import NVFP4Linear, nvfp4_linear, quantize_nvfp4
        from rtx.configs.nvfp4 import NVFP4WeightPrequantConfig, NVFP4FullyPrequantConfig

        for m, n, k in ((16, 32, 128), (17, 33, 129)):
            torch._dynamo.reset()  # Independent fixed specializations, no cache-limit changes.
            g = torch.Generator().manual_seed(177 + k)
            w = torch.randn(n, k, generator=g).bfloat16().cuda()
            x = torch.randn(m, k, generator=g).bfloat16().cuda()
            h = local_second_moments(x)
            packed_x = quantize_nvfp4(x)
            wp = NVFP4WeightPrequantConfig()
            fp = NVFP4FullyPrequantConfig()
            for policy, shrinkage in (("max", 0.), ("mse", 0.), ("local_hessian", 0.),
                                      ("local_hessian", .5)):
                with self.subTest(shape=(m, n, k), policy=policy, shrinkage=shrinkage), torch.inference_mode():
                    packed_w = calibrate_nvfp4_weight(w, policy, second_moments=h,
                                                      hessian_shrinkage=shrinkage)
                    validate_nvfp4_tensor(packed_w)
                    layer = NVFP4Linear(k, n, device="cuda", packed_weight=packed_w, autotune=False,
                                       weight_prequant_config=wp, fully_prequant_config=fp).eval()
                    dynamic = layer(x)
                    fully_packed = nvfp4_linear(packed_x, packed_w, autotune=False,
                                                fully_prequant_config=fp)
                    reference = (decode_native(packed_x).double() @ decode_native(packed_w).double().T).float()
                    reference_bf16 = reference.bfloat16().float()
                    self.assertTrue(bool(dynamic.isfinite().all()))
                    self.assertTrue(bool(fully_packed.isfinite().all()))
                    torch.testing.assert_close(dynamic, fully_packed, rtol=0, atol=0)
                    # Decode oracle includes identical packed X, not original BF16 X.
                    # Absolute floor protects cancellation; one BF16 ulp at output scale.
                    tolerance = reference.abs().max().item() / 128 + 1e-4
                    torch.testing.assert_close(fully_packed.float().cpu(), reference_bf16,
                                               rtol=0, atol=tolerance)
                    torch._dynamo.reset()
                    compiled = torch.compile(layer, fullgraph=True, dynamic=False,
                                             options={"triton.cudagraphs": False})
                    torch.testing.assert_close(compiled(x), dynamic, rtol=0, atol=0)
                    torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
