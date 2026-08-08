from __future__ import annotations

import unittest

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import rtx


class LinearFrontendContractTests(unittest.TestCase):
    def test_mxfp8_is_no_bias_nn_linear_compatible(self) -> None:
        layer = rtx.MXFP8Linear(128, 64, bias=False, device="cpu")
        self.assertEqual(layer.weight.shape, (64, 128))
        self.assertIsNone(layer.bias)
        self.assertEqual(tuple(layer.state_dict()), ("weight",))
        with self.assertRaisesRegex(NotImplementedError, "bias=False"):
            rtx.MXFP8Linear(128, 64, bias=True, device="cpu")

    def test_nvfp4_public_contract_is_no_bias_bf16(self) -> None:
        layer = rtx.NVFP4Linear(128, 64, bias=False, device="cpu")
        self.assertEqual(layer.weight.shape, (64, 128))
        self.assertEqual(layer.weight.dtype, torch.bfloat16)
        self.assertIsNone(layer.bias)
        self.assertEqual(tuple(layer.state_dict()), ("weight",))
        self.assertIn("forward=NVFP4", layer.extra_repr())
        self.assertIn("backward=MXFP8", layer.extra_repr())

    def test_nvfp4_rejects_bias_and_non_bf16_parameters(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "bias=False"):
            rtx.NVFP4Linear(128, 64, bias=True, device="cpu")
        with self.assertRaisesRegex(TypeError, "BF16"):
            rtx.NVFP4Linear(
                128,
                64,
                bias=False,
                device="cpu",
                dtype=torch.float32,
            )

    def test_both_dispatcher_families_have_fake_and_autograd_ops(self) -> None:
        with FakeTensorMode():
            x = torch.empty(
                128, 128, device="cuda", dtype=torch.bfloat16
            )
            weight = torch.empty(
                64, 128, device="cuda", dtype=torch.bfloat16
            )
            calls = (
                (torch.ops.rtx.mxfp8_linear_fwd, (x, weight, "fake")),
                (
                    torch.ops.rtx.mxfp8_linear_train,
                    (x, weight, "fake", "fake"),
                ),
                (torch.ops.rtx.nvfp4_linear_fwd, (x, weight, "fake")),
                (
                    torch.ops.rtx.nvfp4_linear_train,
                    (x, weight, "fake", "fake"),
                ),
            )
            for op, args in calls:
                with self.subTest(op=op):
                    result = op(*args)
                    self.assertEqual(result.shape, (128, 64))
                    self.assertEqual(result.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
