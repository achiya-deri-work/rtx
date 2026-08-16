from __future__ import annotations

import unittest

import torch
from torch import nn
from torch.nn import functional as F

import rtx
from benchmarks.dinov3_mixed_precision import (
    DEFAULT_MODES as DEFAULT_MIXED_MODES,
    mode_metadata,
    run_mixed_regression,
)
from benchmarks.dinov3_regression import (
    DEFAULT_CHECKPOINT,
    EXPECTED_LINEAR_COUNT,
    EXPECTED_PARAMETERS,
    EXPECTED_SHA256,
    DinoBiasAdapter,
    checkpoint_sha256,
    convert_dinov3_variant,
    deterministic_images,
    load_dinov3_vits16,
    run_variant,
    tensor_metrics,
)


CHECKPOINT_AVAILABLE = DEFAULT_CHECKPOINT.is_file()
HAS_SM12 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12


@unittest.skipUnless(CHECKPOINT_AVAILABLE, "local DINOv3 ViT-S/16 checkpoint missing")
class DinoV3CheckpointContractTests(unittest.TestCase):
    def test_checkpoint_hash_and_model_inventory(self) -> None:
        self.assertEqual(checkpoint_sha256(), EXPECTED_SHA256)
        model = load_dinov3_vits16()
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()),
            EXPECTED_PARAMETERS,
        )
        linears = [
            (name, module)
            for name, module in model.named_modules()
            if name.startswith("blocks.") and isinstance(module, nn.Linear)
        ]
        self.assertEqual(len(linears), EXPECTED_LINEAR_COUNT)
        self.assertTrue(all(module.bias is not None for _, module in linears))
        self.assertEqual(tuple(model.blocks[0].attn.qkv.weight.shape), (1152, 384))
        self.assertEqual(tuple(model.blocks[0].mlp.fc1.weight.shape), (1536, 384))
        self.assertEqual(tuple(model.blocks[0].mlp.fc2.weight.shape), (384, 1536))

    def test_qkv_mask_and_pretrained_bias_contract(self) -> None:
        model = load_dinov3_vits16()
        qkv = model.blocks[0].attn.qkv
        self.assertTrue(bool((qkv.bias == 0).all()))
        # The released small checkpoint freezes the complete QKV bias path:
        # both the parameter and its persisted mask are exactly zero.
        self.assertTrue(bool((qkv.bias_mask == 0).all()))
        learned = model.blocks[0].mlp.fc1.bias
        self.assertGreater(float(learned.detach().abs().max()), 0.1)

    def test_public_conversion_rejects_unadapted_dino_biases_atomically(self) -> None:
        model = load_dinov3_vits16()
        original = model.blocks[0].attn.qkv
        with self.assertRaisesRegex(NotImplementedError, "do not support bias"):
            rtx.convert_to_mxfp8_training(
                model,
                module_filter_fn=lambda module, fqn: (
                    isinstance(module, nn.Linear) and fqn.startswith("blocks.")
                ),
            )
        self.assertIs(model.blocks[0].attn.qkv, original)

    def test_training_adapter_preserves_weight_bias_and_mask_objects(self) -> None:
        model = load_dinov3_vits16()
        source = model.blocks[0].attn.qkv
        weight, bias, mask = source.weight, source.bias, source.bias_mask
        model = convert_dinov3_variant(model, "mxfp8_training")
        adapted = model.blocks[0].attn.qkv
        self.assertIsInstance(adapted, DinoBiasAdapter)
        self.assertIsInstance(adapted.core, rtx.MXFP8Linear)
        self.assertIs(adapted.core.weight, weight)
        self.assertIs(adapted.bias, bias)
        self.assertIs(adapted.bias_mask, mask)

    def test_regional_alias_is_rejected(self) -> None:
        # DINO's source projection is biased, so exercise the same bias-free
        # core construction used by the model adapter.
        source = load_dinov3_vits16().blocks[0].attn.qkv
        core = nn.Linear(
            source.in_features,
            source.out_features,
            bias=False,
            dtype=torch.bfloat16,
        )
        core.weight = source.weight
        with self.assertRaisesRegex(ValueError, "scaling"):
            rtx.convert_to_nvfp4_training(
                core,
                config=rtx.NVFP4TrainingConfig(
                    scaling="regional", autotune="off"  # type: ignore[arg-type]
                ),
            )

    def test_mixed_precision_mode_metadata_distinguishes_native_mma(self) -> None:
        native = [
            mode
            for mode in DEFAULT_MIXED_MODES
            if mode_metadata(mode)["direct_sm120_mma"]
        ]
        self.assertEqual(native, ["mxf4_w4a8"])
        for mode in DEFAULT_MIXED_MODES:
            metadata = mode_metadata(mode)
            self.assertTrue(metadata["emulated"])
            self.assertFalse(metadata["performance_result"])


@unittest.skipUnless(
    CHECKPOINT_AVAILABLE and HAS_SM12,
    "requires local DINOv3 checkpoint and SM120/SM121 CUDA GPU",
)
class DinoV3CudaRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(4401)
        torch.compiler.reset()

    def test_bias_adapter_preserves_bf16_pretrained_function(self) -> None:
        images = deterministic_images(1, 64, device="cuda", seed=4402)
        reference, _, _ = run_variant("bf16", images)
        adapted, _, _ = run_variant("bf16_bias_adapter", images)
        metrics = tensor_metrics(adapted, reference)
        self.assertTrue(metrics["finite"])
        self.assertGreater(metrics["minimum_cosine"], 0.999)
        self.assertLess(metrics["relative_l2"], 0.04)

    def test_w4a16_and_w4a8_format_ordering(self) -> None:
        report = run_mixed_regression(
            batch_size=2,
            image_size=64,
            device="cuda:0",
        )
        results = report["results"]
        cosines = {}
        for mode in DEFAULT_MIXED_MODES:
            result = results[mode]
            self.assertEqual(result["converted_linears"], 48)
            metrics = result["versus_bf16_bias_adapter"]
            self.assertTrue(metrics["finite"])
            cosines[mode] = float(metrics["mean_cosine"])
        self.assertLess(
            abs(cosines["nvfp4_w4a8"] - cosines["nvfp4_w4a16"]),
            0.02,
        )
        self.assertGreater(cosines["nvfp4_w4a16"], 0.85)
        self.assertLess(
            cosines["mxf4_w4a8"],
            cosines["nvfp4_w4a8"] - 0.02,
        )

    def test_layerwise_mxfp8_ptq_regression(self) -> None:
        images = deterministic_images(2, 224, device="cuda", seed=4403)
        reference, reference_blocks, _ = run_variant(
            "bf16_bias_adapter", images, capture_blocks=True
        )
        actual, actual_blocks, metadata = run_variant(
            "mxfp8_ptq", images, capture_blocks=True
        )
        final = tensor_metrics(actual, reference)
        self.assertTrue(final["finite"])
        self.assertGreater(final["minimum_cosine"], 0.97)
        self.assertLess(final["relative_l2"], 0.30)
        self.assertGreater(metadata["packed_bytes"], 0)
        self.assertEqual(tuple(actual_blocks), tuple(reference_blocks))
        previous_error = 0.0
        for name in reference_blocks:
            metrics = tensor_metrics(actual_blocks[name], reference_blocks[name])
            self.assertTrue(metrics["finite"], name)
            self.assertLess(metrics["relative_l2"], 0.35, name)
            # Drift need not be strictly monotonic, but a catastrophic isolated
            # block jump should fail with its exact depth in the assertion.
            self.assertLess(metrics["relative_l2"], previous_error + 0.15, name)
            previous_error = float(metrics["relative_l2"])

    def test_dense_patch_and_storage_token_fidelity(self) -> None:
        images = deterministic_images(2, 224, device="cuda", seed=4410)
        features = {}
        with torch.inference_mode():
            for variant in (
                "bf16_bias_adapter",
                "mxfp8_ptq",
                "nvfp4_training_delayed",
                "nvfp4_training_current",
                "nvfp4_training_jit",
                "nvfp4_training_block",
                "nvfp4_ptq_current",
                "nvfp4_ptq_block",
            ):
                model = convert_dinov3_variant(
                    load_dinov3_vits16(device="cuda"), variant
                ).eval()
                features[variant] = model.forward_features(images)

        reference = features["bf16_bias_adapter"]
        self.assertEqual(tuple(reference["x_norm_patchtokens"].shape), (2, 196, 384))
        self.assertEqual(tuple(reference["x_storage_tokens"].shape), (2, 4, 384))
        for variant, patch_cosine, storage_cosine in (
            ("mxfp8_ptq", 0.99, 0.995),
            ("nvfp4_training_delayed", 0.92, 0.96),
            ("nvfp4_training_current", 0.92, 0.96),
            ("nvfp4_training_jit", 0.92, 0.96),
            ("nvfp4_training_block", 0.92, 0.95),
            ("nvfp4_ptq_current", 0.93, 0.97),
            ("nvfp4_ptq_block", 0.92, 0.95),
        ):
            with self.subTest(variant=variant):
                actual = features[variant]
                patch = tensor_metrics(
                    actual["x_norm_patchtokens"],
                    reference["x_norm_patchtokens"],
                )
                storage = tensor_metrics(
                    actual["x_storage_tokens"],
                    reference["x_storage_tokens"],
                )
                self.assertTrue(patch["finite"])
                self.assertTrue(storage["finite"])
                self.assertGreater(patch["minimum_cosine"], patch_cosine)
                self.assertGreater(storage["minimum_cosine"], storage_cosine)

    def test_ragged_rectangular_and_multi_batch_shapes(self) -> None:
        shapes = (
            (1, 3, 16, 16),
            (2, 3, 48, 80),
            (3, 3, 80, 48),
            (1, 3, 240, 256),
        )
        generator = torch.Generator(device="cuda").manual_seed(4414)
        with torch.inference_mode():
            for variant in (
                "mxfp8_ptq",
                "nvfp4_training_jit",
                "nvfp4_training_block",
                "nvfp4_ptq_current",
                "nvfp4_ptq_block",
            ):
                model = convert_dinov3_variant(
                    load_dinov3_vits16(device="cuda"), variant
                ).eval()
                for shape in shapes:
                    with self.subTest(variant=variant, shape=shape):
                        images = torch.randn(
                            shape,
                            generator=generator,
                            device="cuda",
                            dtype=torch.bfloat16,
                        )
                        output = model(images)
                        self.assertEqual(tuple(output.shape), (shape[0], 384))
                        self.assertTrue(bool(torch.isfinite(output).all()))

    def test_batch_decomposition_contracts(self) -> None:
        images = deterministic_images(3, 64, device="cuda", seed=4411)
        with torch.inference_mode():
            for variant in (
                "bf16_bias_adapter",
                "mxfp8_ptq",
                "nvfp4_training_current",
                "nvfp4_training_jit",
                "nvfp4_training_block",
                "nvfp4_ptq_block",
            ):
                model = convert_dinov3_variant(
                    load_dinov3_vits16(device="cuda"), variant
                ).eval()
                batched = model(images)
                separate = torch.cat([model(image[None]) for image in images])
                with self.subTest(variant=variant):
                    torch.testing.assert_close(batched, separate, rtol=0, atol=0)

            # Tensor-current NVFP4 intentionally derives a shared runtime
            # scale from the current input. Its result may depend on batch
            # composition, but the representation must remain stable.
            model = convert_dinov3_variant(
                load_dinov3_vits16(device="cuda"), "nvfp4_ptq_current"
            ).eval()
            batched = model(images)
            separate = torch.cat([model(image[None]) for image in images])
            metrics = tensor_metrics(batched, separate)
            self.assertTrue(metrics["finite"])
            self.assertGreater(metrics["minimum_cosine"], 0.90)

            delayed = convert_dinov3_variant(
                load_dinov3_vits16(device="cuda"),
                "nvfp4_training_delayed",
            ).eval()
            delayed_batched = delayed(images)
            delayed_separate = torch.cat(
                [delayed(image[None]) for image in images]
            )
            delayed_metrics = tensor_metrics(delayed_batched, delayed_separate)
            self.assertTrue(delayed_metrics["finite"])
            self.assertGreater(delayed_metrics["minimum_cosine"], 0.96)

    def test_representation_geometry_is_preserved(self) -> None:
        images = deterministic_images(12, 64, device="cuda", seed=4413)
        embeddings = {}
        with torch.inference_mode():
            for variant in (
                "bf16_bias_adapter",
                "mxfp8_ptq",
                "nvfp4_training_delayed",
                "nvfp4_training_current",
                "nvfp4_training_jit",
                "nvfp4_training_block",
                "nvfp4_ptq_current",
                "nvfp4_ptq_block",
            ):
                model = convert_dinov3_variant(
                    load_dinov3_vits16(device="cuda"), variant
                ).eval()
                embeddings[variant] = F.normalize(model(images).float(), dim=-1)
        reference = embeddings["bf16_bias_adapter"]
        reference_gram = reference @ reference.T
        for variant, l2_ceiling in (
            ("mxfp8_ptq", 0.02),
            ("nvfp4_training_delayed", 0.12),
            ("nvfp4_training_current", 0.12),
            ("nvfp4_training_jit", 0.12),
            ("nvfp4_training_block", 0.12),
            ("nvfp4_ptq_current", 0.10),
            ("nvfp4_ptq_block", 0.12),
        ):
            with self.subTest(variant=variant):
                actual = embeddings[variant]
                metrics = tensor_metrics(actual @ actual.T, reference_gram)
                self.assertTrue(metrics["finite"])
                self.assertLess(metrics["relative_l2"], l2_ceiling)

    def test_nvfp4_ptq_current_and_block_are_finite(self) -> None:
        images = deterministic_images(1, 64, device="cuda", seed=4404)
        reference, _, _ = run_variant("bf16_bias_adapter", images)
        for variant, cosine_floor, l2_ceiling in (
            ("nvfp4_ptq_current", 0.85, 0.60),
            ("nvfp4_ptq_block", 0.80, 0.75),
        ):
            with self.subTest(variant=variant):
                actual, _, metadata = run_variant(variant, images)
                metrics = tensor_metrics(actual, reference)
                self.assertTrue(metrics["finite"])
                self.assertGreater(metrics["minimum_cosine"], cosine_floor)
                self.assertLess(metrics["relative_l2"], l2_ceiling)
                self.assertGreater(metadata["packed_bytes"], 0)

    def test_ptq_variants_are_fullgraph_compileable(self) -> None:
        images = deterministic_images(1, 32, device="cuda", seed=4405)
        for variant, cosine_floor, l2_ceiling in (
            ("mxfp8_ptq", 0.99, 0.15),
            ("nvfp4_ptq_current", 0.95, 0.40),
            ("nvfp4_ptq_block", 0.94, 0.40),
        ):
            with self.subTest(variant=variant):
                torch.compiler.reset()
                eager, _, _ = run_variant(variant, images)
                compiled, _, _ = run_variant(
                    variant, images, compile_model=True
                )
                metrics = tensor_metrics(compiled, eager)
                self.assertTrue(metrics["finite"])
                self.assertGreater(metrics["minimum_cosine"], cosine_floor)
                self.assertLess(metrics["relative_l2"], l2_ceiling)

    def test_ptq_eager_execution_is_repeatable(self) -> None:
        images = deterministic_images(1, 32, device="cuda", seed=4407)
        for variant in (
            "mxfp8_ptq",
            "nvfp4_ptq_current",
            "nvfp4_ptq_block",
        ):
            with self.subTest(variant=variant):
                first, _, _ = run_variant(variant, images)
                second, _, _ = run_variant(variant, images)
                torch.testing.assert_close(second, first, rtol=0, atol=0)

    def test_packed_model_state_dict_round_trip(self) -> None:
        images = deterministic_images(1, 64, device="cuda", seed=4408)
        for variant in (
            "mxfp8_ptq",
            "nvfp4_ptq_current",
            "nvfp4_ptq_block",
        ):
            with self.subTest(variant=variant), torch.inference_mode():
                source = convert_dinov3_variant(
                    load_dinov3_vits16(device="cuda"), variant
                ).eval()
                expected = source(images)
                state = {
                    name: value.detach().clone()
                    for name, value in source.state_dict().items()
                }
                restored = convert_dinov3_variant(
                    load_dinov3_vits16(device="cuda"), variant
                ).eval()
                restored.load_state_dict(state, strict=True)
                actual = restored(images)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_full_resolution_ptq_is_fullgraph_compileable(self) -> None:
        images = deterministic_images(1, 224, device="cuda", seed=4409)
        for variant, cosine_floor, l2_ceiling in (
            ("mxfp8_ptq", 0.99, 0.20),
            ("nvfp4_ptq_current", 0.92, 0.45),
            ("nvfp4_ptq_block", 0.93, 0.45),
        ):
            with self.subTest(variant=variant):
                torch.compiler.reset()
                eager, _, _ = run_variant(variant, images)
                compiled, _, _ = run_variant(
                    variant, images, compile_model=True
                )
                metrics = tensor_metrics(compiled, eager)
                self.assertTrue(metrics["finite"])
                self.assertGreater(metrics["minimum_cosine"], cosine_floor)
                self.assertLess(metrics["relative_l2"], l2_ceiling)

    def test_all_dynamic_nvfp4_modes_are_fullgraph_compileable(self) -> None:
        images = deterministic_images(1, 64, device="cuda", seed=4420)
        for variant in (
            "nvfp4_training_delayed",
            "nvfp4_training_current",
            "nvfp4_training_jit",
            "nvfp4_training_block",
        ):
            with self.subTest(variant=variant):
                torch.compiler.reset()
                eager, _, _ = run_variant(variant, images)
                compiled, _, _ = run_variant(
                    variant, images, compile_model=True
                )
                metrics = tensor_metrics(compiled, eager)
                self.assertTrue(metrics["finite"])
                self.assertGreater(metrics["minimum_cosine"], 0.93)
                self.assertLess(metrics["relative_l2"], 0.40)

    def test_delayed_history_matches_eager_under_fullgraph(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(4431)
        inputs = [
            torch.randn(
                (2, 17, 384),
                generator=generator,
                device="cuda",
                dtype=torch.bfloat16,
            )
            * magnitude
            for magnitude in (0.0625, 32.0, 0.0625)
        ]
        eager_model = convert_dinov3_variant(
            load_dinov3_vits16(device="cuda"),
            "nvfp4_training_delayed",
        ).eval()
        compiled_model = convert_dinov3_variant(
            load_dinov3_vits16(device="cuda"),
            "nvfp4_training_delayed",
        ).eval()
        eager = eager_model.blocks[0].attn.qkv.core
        compiled_core = compiled_model.blocks[0].attn.qkv.core
        compiled = torch.compile(compiled_core, fullgraph=True, dynamic=False)
        with torch.no_grad():
            histories = []
            for value in inputs:
                expected = eager(value)
                actual = compiled(value)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(
                    compiled_core._x_amax_state,
                    eager._x_amax_state,
                    rtol=0,
                    atol=0,
                )
                histories.append(eager._x_amax_state.detach().clone())
        self.assertGreater(float(histories[1][0]), float(histories[0][0]) * 100)
        self.assertEqual(float(histories[2][1]), float(histories[1][0]))

    def test_mxfp8_and_nvfp4_training_backward_produce_finite_gradients(self) -> None:
        for variant in (
            "mxfp8_training",
            "nvfp4_training_delayed",
            "nvfp4_training_current",
            "nvfp4_training_jit",
            "nvfp4_training_block",
        ):
            with self.subTest(variant=variant):
                torch.compiler.reset()
                model = load_dinov3_vits16(device="cuda").train()
                model = convert_dinov3_variant(model, variant)
                function = torch.compile(model, fullgraph=True, dynamic=False)
                images = deterministic_images(
                    1, 48, device="cuda", seed=4406
                ).requires_grad_(True)
                output = function(images)
                output.float().square().mean().backward()
                torch.cuda.synchronize()
                self.assertTrue(bool(torch.isfinite(output).all()))
                self.assertIsNotNone(images.grad)
                self.assertTrue(bool(torch.isfinite(images.grad).all()))
                first = model.blocks[0].mlp.fc1
                self.assertIsInstance(first, DinoBiasAdapter)
                self.assertIsNotNone(first.core.weight.grad)
                self.assertTrue(bool(torch.isfinite(first.core.weight.grad).all()))
                self.assertIsNotNone(first.bias.grad)
                self.assertTrue(bool(torch.isfinite(first.bias.grad).all()))


if __name__ == "__main__":
    unittest.main()
