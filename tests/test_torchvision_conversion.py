from __future__ import annotations

import unittest

import torch
from torch import nn

import rtx

try:
    from torchvision.models.vision_transformer import VisionTransformer
except (ImportError, RuntimeError):
    VisionTransformer = None


SELECTED_MLP_FQNS = {
    "encoder.layers.encoder_layer_0.mlp.0",
    "encoder.layers.encoder_layer_0.mlp.3",
}


def _tiny_vit(*, device: str = "cpu") -> nn.Module:
    assert VisionTransformer is not None
    model = VisionTransformer(
        image_size=32,
        patch_size=8,
        num_layers=1,
        num_heads=4,
        hidden_dim=128,
        mlp_dim=256,
        num_classes=10,
    )
    # Torchvision ViT is bias-heavy. RTX deliberately does not implement
    # bias, so make only the two projections selected by this integration test
    # bias-free without changing unrelated attention/head modules.
    for fqn in SELECTED_MLP_FQNS:
        parent_fqn, name = fqn.rsplit(".", 1)
        parent = model.get_submodule(parent_fqn)
        source = parent._modules[name]
        assert isinstance(source, nn.Linear)
        replacement = nn.Linear(
            source.in_features,
            source.out_features,
            bias=False,
            device=source.weight.device,
            dtype=source.weight.dtype,
        )
        replacement.weight = source.weight
        parent._modules[name] = replacement
    return model.to(device=device, dtype=torch.bfloat16)


def _select_vit_mlp(module: nn.Module, fqn: str) -> bool:
    return isinstance(module, nn.Linear) and fqn in SELECTED_MLP_FQNS


@unittest.skipIf(VisionTransformer is None, "torchvision is not installed")
class TorchvisionConversionContractTests(unittest.TestCase):
    def test_small_vit_training_conversion_preserves_parameters(self) -> None:
        for converter, config, expected in (
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
                model = _tiny_vit()
                original = {
                    fqn: model.get_submodule(fqn).weight
                    for fqn in SELECTED_MLP_FQNS
                }
                converted = converter(
                    model, module_filter_fn=_select_vit_mlp, config=config
                )
                self.assertIs(converted, model)
                for fqn in SELECTED_MLP_FQNS:
                    module = model.get_submodule(fqn)
                    self.assertIsInstance(module, expected)
                    self.assertIs(module.weight, original[fqn])
                self.assertIsInstance(model.heads.head, nn.Linear)

    def test_default_conversion_rejects_torchvision_biases(self) -> None:
        model = _tiny_vit()
        with self.assertRaisesRegex(NotImplementedError, "do not support bias"):
            rtx.convert_to_mxfp8_training(model)

    def test_torchao_rowwise_training_model_converts_directly(self) -> None:
        from torchao.float8 import (
            Float8LinearConfig,
            Float8LinearRecipeName,
            convert_to_float8_training,
        )

        model = _tiny_vit()
        original = {
            fqn: model.get_submodule(fqn).weight for fqn in SELECTED_MLP_FQNS
        }
        model = convert_to_float8_training(
            model,
            module_filter_fn=_select_vit_mlp,
            config=Float8LinearConfig.from_recipe_name(
                Float8LinearRecipeName.ROWWISE
            ),
        )
        for fqn in SELECTED_MLP_FQNS:
            self.assertEqual(type(model.get_submodule(fqn)).__name__, "Float8Linear")
        model = rtx.convert_to_mxfp8_training(
            model,
            module_filter_fn=_select_vit_mlp,
            config=rtx.MXFP8TrainingConfig(autotune="off"),
        )
        for fqn in SELECTED_MLP_FQNS:
            module = model.get_submodule(fqn)
            self.assertIsInstance(module, rtx.MXFP8Linear)
            self.assertIs(module.weight, original[fqn])


@unittest.skipUnless(
    VisionTransformer is not None
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] == 12,
    "requires torchvision and an SM120/SM121 CUDA GPU",
)
class TorchvisionConversionCudaTests(unittest.TestCase):
    def test_small_vit_training_runs_both_formats_fullgraph(self) -> None:
        for converter, config in (
            (
                rtx.convert_to_mxfp8_training,
                rtx.MXFP8TrainingConfig(autotune="off"),
            ),
            (
                rtx.convert_to_nvfp4_training,
                rtx.NVFP4TrainingConfig(autotune="off", scaling="block"),
            ),
        ):
            with self.subTest(converter=converter.__name__):
                torch.compiler.reset()
                model = _tiny_vit(device="cuda").train()
                model = converter(
                    model,
                    module_filter_fn=_select_vit_mlp,
                    config=config,
                )
                compiled = torch.compile(model, fullgraph=True, dynamic=False)
                images = torch.randn(
                    2, 3, 32, 32, device="cuda", dtype=torch.bfloat16
                )
                output = compiled(images)
                output.float().square().mean().backward()
                torch.cuda.synchronize()
                self.assertEqual(tuple(output.shape), (2, 10))
                for fqn in SELECTED_MLP_FQNS:
                    self.assertIsNotNone(model.get_submodule(fqn).weight.grad)

    def test_small_vit_ptq_runs_both_formats(self) -> None:
        images = torch.randn(
            2, 3, 32, 32, device="cuda", dtype=torch.bfloat16
        )
        for config, expected in (
            (
                rtx.MXFP8WeightOnlyConfig(autotune="off"),
                rtx.MXFP8Linear,
            ),
            (
                rtx.NVFP4WeightOnlyConfig(
                    scaling="current", autotune="off"
                ),
                rtx.NVFP4Linear,
            ),
        ):
            with self.subTest(config=type(config).__name__):
                model = _tiny_vit(device="cuda").eval()
                model = rtx.quantize_(
                    model, config, filter_fn=_select_vit_mlp
                )
                for fqn in SELECTED_MLP_FQNS:
                    module = model.get_submodule(fqn)
                    self.assertIsInstance(module, expected)
                    self.assertIsNone(module.weight)
                compiled = torch.compile(model, fullgraph=True, dynamic=False)
                with torch.inference_mode():
                    output = compiled(images)
                torch.cuda.synchronize()
                self.assertEqual(tuple(output.shape), (2, 10))
                self.assertTrue(bool(torch.isfinite(output).all()))


if __name__ == "__main__":
    unittest.main()
