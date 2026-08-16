from __future__ import annotations

import unittest

import torch
from torch import nn

import rtx


class MXFP8PTQContractTests(unittest.TestCase):
    def test_training_swaps_preserve_parameter_and_state_dict(self) -> None:
        for converter, config, expected_type in (
            (
                rtx.convert_to_mxfp8_training,
                rtx.MXFP8TrainingConfig(autotune="off"),
                rtx.MXFP8Linear,
            ),
            (
                rtx.convert_to_nvfp4_training,
                rtx.NVFP4TrainingConfig(autotune="off", scaling="block"),
                rtx.NVFP4Linear,
            ),
        ):
            with self.subTest(converter=converter.__name__):
                model = nn.Sequential(
                    nn.Linear(8, 16, bias=False, dtype=torch.bfloat16),
                    nn.ReLU(),
                ).eval()
                original_parameter = model[0].weight
                optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
                converted = converter(model, config=config)
                self.assertIs(converted, model)
                self.assertIsInstance(model[0], expected_type)
                self.assertIs(model[0].weight, original_parameter)
                self.assertIs(optimizer.param_groups[0]["params"][0], original_parameter)
                self.assertEqual(tuple(model.state_dict()), ("0.weight",))
                self.assertFalse(model[0].training)

    def test_training_root_conversion_returns_replacement(self) -> None:
        root = nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)
        parameter = root.weight
        converted = rtx.convert_to_mxfp8_training(
            root, config=rtx.MXFP8TrainingConfig(autotune="off")
        )
        self.assertIsInstance(converted, rtx.MXFP8Linear)
        self.assertIs(converted.weight, parameter)

    def test_bias_rejection_is_atomic(self) -> None:
        model = nn.Sequential(
            nn.Linear(8, 8, bias=False, dtype=torch.bfloat16),
            nn.Linear(8, 8, bias=True, dtype=torch.bfloat16),
        )
        first = model[0]
        with self.assertRaisesRegex(NotImplementedError, "do not support bias"):
            rtx.quantize_(
                model,
                rtx.MXFP8WeightOnlyConfig(),
                filter_fn=lambda module, fqn: fqn == "1",
            )
        self.assertIs(model[0], first)

    def test_filter_uses_fully_qualified_names(self) -> None:
        model = nn.Sequential(
            nn.Linear(8, 8, bias=True, dtype=torch.bfloat16),
            nn.Linear(8, 8, bias=False, dtype=torch.float32),
        )
        # Nothing is selected, so neither the bias nor dtype restriction applies.
        self.assertIs(
            rtx.quantize_(
                model,
                rtx.MXFP8WeightOnlyConfig(),
                filter_fn=lambda module, fqn: fqn == "missing",
            ),
            model,
        )

    def test_shared_module_filter_must_select_every_alias(self) -> None:
        shared = nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)
        model = nn.Module()
        model.left = shared
        model.right = shared
        with self.assertRaisesRegex(ValueError, "some aliases"):
            rtx.quantize_(
                model,
                rtx.MXFP8WeightOnlyConfig(),
                filter_fn=lambda module, fqn: fqn == "left",
            )

    def test_root_dtype_is_validated(self) -> None:
        root = nn.Linear(8, 8, bias=False, dtype=torch.float32)
        with self.assertRaisesRegex(TypeError, "BF16 weight"):
            rtx.quantize_(root, rtx.MXFP8WeightOnlyConfig())


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] == 12,
    "requires an SM120/SM121 CUDA GPU",
)
class MXFP8PTQCudaTests(unittest.TestCase):
    def test_selected_linear_is_packed_once_and_fullgraph_compileable(self) -> None:
        torch.manual_seed(2201)
        model = nn.Sequential(
            nn.Linear(
                128, 64, bias=False, device="cuda", dtype=torch.bfloat16
            ),
            nn.ReLU(),
        ).eval()
        original = model[0]
        converted = rtx.quantize_(
            model,
            rtx.MXFP8WeightOnlyConfig(autotune="off"),
            filter_fn=lambda module, fqn: fqn == "0",
        )
        self.assertIs(converted, model)
        self.assertIsInstance(model[0], rtx.MXFP8Linear)
        self.assertIsNone(model[0].weight)
        self.assertIsNot(model[0], original)
        self.assertEqual(
            tuple(model[0].state_dict()),
            ("weight_data", "weight_scales", "weight_packing_meta"),
        )
        for buffer in (model[0].weight_data, model[0].weight_scales):
            current = buffer
            while isinstance(current, torch.Tensor):
                self.assertFalse(hasattr(current, "_base_inputs"))
                current = getattr(current, "_base", None)

        x = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            eager = model(x)
            compiled = torch.compile(model, fullgraph=True, dynamic=False)
            actual = compiled(x)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, eager, rtol=0, atol=0)

    def test_root_linear_is_returned_as_packed_module(self) -> None:
        root = nn.Linear(
            128, 64, bias=False, device="cuda", dtype=torch.bfloat16
        ).eval()
        converted = rtx.quantize_(
            root, rtx.MXFP8WeightOnlyConfig(autotune="off")
        )
        self.assertIsInstance(converted, rtx.MXFP8Linear)
        self.assertEqual(converted.weight_mode, "prequantized")

    def test_nvfp4_ptq_supports_current_and_block_policies(self) -> None:
        for scaling in ("current", "block"):
            with self.subTest(scaling=scaling):
                root = nn.Linear(
                    128, 64, bias=False, device="cuda", dtype=torch.bfloat16
                ).eval()
                converted = rtx.quantize_(
                    root,
                    rtx.NVFP4WeightOnlyConfig(
                        scaling=scaling, autotune="off"
                    ),
                )
                self.assertIsInstance(converted, rtx.NVFP4Linear)
                self.assertEqual(converted.scaling, scaling)
                self.assertEqual(converted.weight_mode, "prequantized")
                self.assertIsNone(converted.weight)
                for buffer in (
                    converted.weight_data,
                    converted.weight_block_scales,
                ):
                    current = buffer
                    while isinstance(current, torch.Tensor):
                        self.assertFalse(hasattr(current, "_base_inputs"))
                        current = getattr(current, "_base", None)
                x = torch.randn(
                    16, 128, device="cuda", dtype=torch.bfloat16
                )
                with torch.inference_mode():
                    output = converted(x)
                self.assertEqual(tuple(output.shape), (16, 64))


if __name__ == "__main__":
    unittest.main()
