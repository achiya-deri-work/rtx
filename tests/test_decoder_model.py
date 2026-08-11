from __future__ import annotations

from dataclasses import replace
import unittest

import torch
from torch import nn

import rtx
from rtx.models import (
    DecoderConfig,
    DecoderOnlyTransformer,
    FP32RMSNorm,
    LinearSpec,
    QKRMSNorm,
    causal_lm_loss,
)


def _small_config(
    precision: str = "bf16", *, dtype: torch.dtype = torch.float32
) -> DecoderConfig:
    return DecoderConfig(
        vocab_size=128,
        max_seq_len=32,
        num_layers=2,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        dtype=dtype,
        linear=LinearSpec(precision=precision, autotune="off"),
    )


class DecoderConfigurationTests(unittest.TestCase):
    def test_default_is_the_short_convergence_shape(self) -> None:
        config = DecoderConfig()
        self.assertEqual(config.hidden_size, 768)
        self.assertEqual(config.intermediate_size, 1536)
        self.assertEqual(config.head_dim, 64)
        self.assertEqual(config.max_seq_len, 512)
        self.assertTrue(config.qk_norm)
        self.assertTrue(config.post_sublayer_norm)
        self.assertTrue(config.tie_word_embeddings)

    def test_invalid_precision_geometry_is_rejected_early(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible by num_attention_heads"):
            replace(_small_config(), hidden_size=130)
        with self.assertRaisesRegex(TypeError, "require BF16"):
            _small_config("mxfp8", dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "divisible by 64"):
            replace(
                _small_config("nvfp4", dtype=torch.bfloat16),
                intermediate_size=224,
            )


class DecoderNumericsTests(unittest.TestCase):
    def test_fp32_norm_returns_activation_dtype_and_unit_rms(self) -> None:
        torch.manual_seed(11)
        norm = FP32RMSNorm(64, dtype=torch.bfloat16)
        x = torch.randn(3, 7, 64, dtype=torch.bfloat16)
        output = norm(x)
        self.assertIs(output.dtype, torch.bfloat16)
        rms = output.float().square().mean(dim=-1).sqrt()
        torch.testing.assert_close(rms, torch.ones_like(rms), atol=3e-3, rtol=3e-3)

    def test_qk_norm_controls_each_head_independently(self) -> None:
        torch.manual_seed(13)
        norm = QKRMSNorm(32, dtype=torch.float32)
        query = 100.0 * torch.randn(2, 4, 8, 32)
        key = 0.1 * torch.randn(2, 4, 8, 32)
        query, key = norm(query, key)
        for value in (query, key):
            rms = value.square().mean(dim=-1).sqrt()
            torch.testing.assert_close(
                rms, torch.ones_like(rms), atol=2e-3, rtol=2e-3
            )

    def test_causal_forward_backward_and_future_token_isolation(self) -> None:
        torch.manual_seed(17)
        model = DecoderOnlyTransformer(_small_config()).eval()
        first = torch.randint(0, 128, (2, 12))
        second = first.clone()
        second[:, 7:] = torch.randint(0, 128, second[:, 7:].shape)
        first_logits = model(first)
        second_logits = model(second)
        self.assertEqual(first_logits.shape, (2, 12, 128))
        torch.testing.assert_close(first_logits[:, :7], second_logits[:, :7])

        model.train()
        targets = torch.randint(0, 128, (2, 12))
        loss = causal_lm_loss(model(first), targets)
        self.assertIs(loss.dtype, torch.float32)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.layers[0].self_attn.qkv_proj.weight.grad)

    def test_model_is_fullgraph_traceable(self) -> None:
        model = DecoderOnlyTransformer(_small_config()).eval()
        compiled = torch.compile(
            model, backend="eager", fullgraph=True, dynamic=False
        )
        tokens = torch.randint(0, 128, (2, 8))
        torch.testing.assert_close(compiled(tokens), model(tokens))

    def test_rope_tables_survive_module_bf16_cast_in_fp32(self) -> None:
        model = DecoderOnlyTransformer(_small_config()).bfloat16()
        rope = model.layers[0].self_attn.rope
        self.assertIs(rope.cos.dtype, torch.float32)
        self.assertIs(rope.sin.dtype, torch.float32)


class DecoderPrecisionBoundaryTests(unittest.TestCase):
    def test_depth_progressive_residual_scaling_preserves_point_zero_two_base_init(
        self,
    ) -> None:
        torch.manual_seed(19)
        config = _small_config()
        model = DecoderOnlyTransformer(config)
        base_weights = [model.embed_tokens.weight, model.lm_head.weight]
        residual_weights = []
        for depth, layer in enumerate(model.layers, start=1):
            base_weights.extend(
                (layer.self_attn.qkv_proj.weight, layer.mlp.gate_up_proj.weight)
            )
            residual_weights.append(
                (
                    depth,
                    layer.self_attn.out_proj.weight,
                    layer.mlp.down_proj.weight,
                )
            )
        for weight in base_weights:
            with self.subTest(shape=tuple(weight.shape)):
                self.assertAlmostEqual(weight.float().std().item(), 0.02, delta=0.001)
        for depth, attention_weight, mlp_weight in residual_weights:
            expected = 0.02 / (2.0 * depth) ** 0.5
            self.assertAlmostEqual(
                config.residual_initializer_range(depth), expected
            )
            for weight in (attention_weight, mlp_weight):
                with self.subTest(depth=depth, residual_shape=tuple(weight.shape)):
                    self.assertAlmostEqual(
                        weight.float().std().item(), expected, delta=0.001
                    )

    def test_low_precision_models_keep_embedding_and_head_bf16(self) -> None:
        expected_types = {
            "mxfp8": rtx.MXFP8Linear,
            "nvfp4": rtx.NVFP4Linear,
        }
        bf16 = DecoderOnlyTransformer(
            _small_config("bf16", dtype=torch.bfloat16), device="cpu"
        )
        initial_state = bf16.state_dict()
        for precision, linear_type in expected_types.items():
            with self.subTest(precision=precision):
                model = DecoderOnlyTransformer(
                    _small_config(precision, dtype=torch.bfloat16), device="cpu"
                )
                self.assertIsInstance(model.embed_tokens, nn.Embedding)
                self.assertIsInstance(model.lm_head, nn.Linear)
                self.assertIs(model.embed_tokens.weight, model.lm_head.weight)
                self.assertIs(model.embed_tokens.weight.dtype, torch.bfloat16)
                layer = model.layers[0]
                self.assertIsInstance(layer.self_attn.qkv_proj, linear_type)
                self.assertIsInstance(layer.self_attn.out_proj, linear_type)
                self.assertIsInstance(layer.mlp.gate_up_proj, linear_type)
                self.assertIsInstance(layer.mlp.down_proj, linear_type)
                model.load_state_dict(initial_state, strict=True)

    def test_sandwich_norms_surround_both_quantized_sublayers(self) -> None:
        model = DecoderOnlyTransformer(
            _small_config("mxfp8", dtype=torch.bfloat16), device="cpu"
        )
        layer = model.layers[0]
        self.assertIsInstance(layer.input_layernorm, FP32RMSNorm)
        self.assertIsInstance(layer.post_attention_layernorm, FP32RMSNorm)
        self.assertIsInstance(layer.pre_feedforward_layernorm, FP32RMSNorm)
        self.assertIsInstance(layer.post_feedforward_layernorm, FP32RMSNorm)
        self.assertIsInstance(layer.self_attn.qk_norm, QKRMSNorm)


if __name__ == "__main__":
    unittest.main()
