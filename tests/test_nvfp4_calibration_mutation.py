"""Regression for in-place PTQ candidate swaps through cached native runners."""

import unittest

import torch

from rtx.formats.nvfp4 import nvfp4_matrix_shape, validate_nvfp4_tensor
from rtx.nvfp4_calibration import calibrate_nvfp4_weight


def _has_sm12x():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12


@torch.no_grad()
def _copy_packed_buffers_(layer, source):
    """Explicit buffer migration; never invoke the TorchAO subclass .to()."""
    validate_nvfp4_tensor(source)
    if nvfp4_matrix_shape(source) != (layer.out_features, layer.in_features):
        raise ValueError("candidate logical shape must match the persistent layer")
    layer.weight_data.copy_(source.qdata.to(layer.weight_data.device))
    layer.weight_block_scales.copy_(source.scale.to(layer.weight_block_scales.device))
    layer.weight_tensor_scale.copy_(source.per_tensor_scale.to(layer.weight_tensor_scale.device))


@unittest.skipUnless(_has_sm12x(), "requires an SM12x CUDA device")
class NVFP4CalibrationMutationTests(unittest.TestCase):
    def test_inplace_candidate_swap_updates_compiled_cached_weight_and_restores(self):
        from unittest.mock import patch
        from rtx.configs.nvfp4 import NVFP4WeightPrequantConfig
        from rtx.formats.nvfp4 import NVFP4Tensor
        from rtx.fp4 import NVFP4Linear

        for m, n, k in ((16, 32, 128), (17, 33, 129)):
            with self.subTest(shape=(m, n, k)):
                torch._dynamo.reset()
                g = torch.Generator().manual_seed(2013 + k)
                weight_a = torch.randn(n, k, generator=g).bfloat16()
                weight_b = (-1.75 * weight_a.float()
                            + .1 * torch.randn(n, k, generator=g)).bfloat16()
                packed_a = calibrate_nvfp4_weight(weight_a, "max")
                packed_b = calibrate_nvfp4_weight(weight_b, "mse")
                config = NVFP4WeightPrequantConfig()
                # Both calibrations begin on CPU. Constructor and updates must
                # transfer their native buffers without subclass aliasing.
                with patch.object(NVFP4Tensor, "to", side_effect=AssertionError("subclass.to is forbidden")):
                    layer = NVFP4Linear(k, n, device="cuda", packed_weight=packed_a,
                                        autotune=False, weight_prequant_config=config).eval()
                    fresh_b = NVFP4Linear(k, n, device="cuda", packed_weight=packed_b,
                                          autotune=False, weight_prequant_config=config).eval()
                buffers = (layer.weight_data, layer.weight_block_scales, layer.weight_tensor_scale)
                pointers = tuple(buffer.data_ptr() for buffer in buffers)
                identities = tuple(id(buffer) for buffer in buffers)
                original = tuple(buffer.clone() for buffer in buffers)
                x = torch.randn(m, k, generator=g).bfloat16().cuda()

                with torch.inference_mode():
                    eager_a = layer(x).clone()  # Warm the registered eager runner.
                    compiled = torch.compile(layer, fullgraph=True, dynamic=False,
                                             options={"triton.cudagraphs": False})
                    compiled_a = compiled(x).clone()  # Warm the compiled runner.
                    expected_b = fresh_b(x).clone()
                    self.assertTrue(bool(eager_a.isfinite().all()))
                    self.assertTrue(bool(expected_b.isfinite().all()))
                    self.assertTrue(bool((eager_a != expected_b).any()))
                    torch.testing.assert_close(compiled_a, eager_a, rtol=0, atol=0)

                    # Swap A -> B -> A -> B without replacing any buffer or
                    # recompiling. Both warmed runtime caches must read current
                    # storage contents, including the outer decode scale.
                    for candidate, expected in ((packed_b, expected_b),
                                                (packed_a, eager_a),
                                                (packed_b, expected_b),
                                                (packed_a, eager_a)):
                        with patch.object(NVFP4Tensor, "to", side_effect=AssertionError("subclass.to is forbidden")):
                            _copy_packed_buffers_(layer, candidate)
                        current = (layer.weight_data, layer.weight_block_scales, layer.weight_tensor_scale)
                        self.assertEqual(tuple(buffer.data_ptr() for buffer in current), pointers)
                        self.assertEqual(tuple(id(buffer) for buffer in current), identities)
                        self.assertEqual(layer.packed_weight._rtx_logical_shape, (n, k))
                        eager = layer(x).clone()
                        updated = compiled(x).clone()
                        self.assertTrue(bool(eager.isfinite().all()))
                        self.assertTrue(bool(updated.isfinite().all()))
                        torch.testing.assert_close(eager, expected, rtol=0, atol=0)
                        torch.testing.assert_close(updated, expected, rtol=0, atol=0)
                    for current, before in zip(buffers, original):
                        # Compare E4M3 as bytes to cover exact restoration.
                        torch.testing.assert_close(current.reshape(-1).view(torch.uint8),
                                                   before.reshape(-1).view(torch.uint8),
                                                   rtol=0, atol=0)
                    torch.cuda.synchronize()
                torch._dynamo.reset()


if __name__ == "__main__":
    unittest.main()
