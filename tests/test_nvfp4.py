from __future__ import annotations

from dataclasses import asdict, replace
import unittest

import torch

import rtx
from rtx.autotune.adapters import make_nvfp4_fwd_adapter
from rtx.autotune.outcomes import TrialOutcome
from rtx.autotune.promotion import _config_rejection, _current_revision
from rtx.configs.nvfp4 import (
    DEFAULT_NVFP4_FWD_CONFIG,
    NVFP4FwdConfig,
    NVFP4Problem,
    NVFP4QuantConfig,
    normalize_nvfp4_fwd_config,
)


def _has_sm12x() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12


class NVFP4ConfigTests(unittest.TestCase):
    def test_training_fallback_exposes_full_block_lane_ownership(self) -> None:
        config = DEFAULT_NVFP4_FWD_CONFIG
        self.assertEqual(config.quant_vec, 16)
        self.assertEqual(config.quant_load_bits, 128)
        self.assertEqual(config.tile_k, 256)
        self.assertEqual(config.native_operand_bits, 4)
        self.assertEqual(config.scale_vector_size, 16)
        self.assertIsNone(config.implementation_rejection(NVFP4Problem(256, 1536, 1536)))

    def test_nvfp4_normalizer_preserves_format_specific_coordinates(self) -> None:
        config = normalize_nvfp4_fwd_config(
            replace(DEFAULT_NVFP4_FWD_CONFIG, collect_amax=True),
            k_unroll=2,
            scale_reciprocal="supplied_pow2_ptx_rcp",
        )
        self.assertIsInstance(config, NVFP4FwdConfig)
        self.assertTrue(config.collect_amax)
        self.assertEqual(config.k_unroll, 2)
        self.assertEqual(config.scale_reciprocal, "supplied_pow2_ptx_rcp")

    def test_standalone_quantizer_exposes_one_lane_per_block(self) -> None:
        config = NVFP4QuantConfig(values_per_lane=16, load_bits=128)
        self.assertEqual(config.threads_per_scale, 1)
        self.assertEqual(config.blocks_per_warp, 32)
        self.assertIsNone(config.rejection(128, 128))

    def test_shared_autotuner_builds_nvfp4_family_context(self) -> None:
        problem = NVFP4Problem(256, 1536, 1536)

        def evaluator(config):
            return TrialOutcome("ok", median_ms=1.0)

        adapter = make_nvfp4_fwd_adapter(
            problem,
            evaluator,
            axes={"quant_vec": (8, 16), "collect_amax": (True,)},
        )
        self.assertEqual(adapter.context.family, "nvfp4_fused_fwd")
        self.assertTrue(adapter.initial_config.collect_amax)
        features = adapter.features(adapter.initial_config)
        self.assertEqual(features["derived.delayed_telemetry_slots"], 24.0)
        self.assertEqual(
            features["derived.delayed_telemetry_state_bytes"], 192.0
        )
        self.assertEqual(features["derived.total_kernel_launches"], 1.0)

    def test_runtime_promotion_understands_nvfp4_revision_and_schema(self) -> None:
        problem = NVFP4Problem(256, 1536, 1536)
        config = replace(DEFAULT_NVFP4_FWD_CONFIG, collect_amax=True)
        self.assertEqual(_current_revision("nvfp4_fused_fwd"), 2)
        self.assertIsNone(
            _config_rejection(
                "nvfp4_fused_fwd",
                asdict(config),
                problem,  # type: ignore[arg-type]
            )
        )


@unittest.skipUnless(_has_sm12x(), "requires an SM120/SM121 CUDA GPU")
class NVFP4CudaTests(unittest.TestCase):
    def test_public_current_scale_fused_forward_is_finite(self) -> None:
        torch.manual_seed(1901)
        x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        out = rtx.nvfp4_linear(x, weight)
        reference = x.float() @ weight.float().T
        error = (out.float() - reference).abs()
        self.assertTrue(bool(torch.isfinite(out).all()))
        self.assertLess(float(error.mean()), 2.0)

    def test_fused_current_matches_torchao_quantization_reference(self) -> None:
        from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

        torch.manual_seed(1905)
        x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        actual = rtx.nvfp4_linear(x, weight)

        def reference_quantize(tensor: torch.Tensor) -> torch.Tensor:
            amax = tensor.float().abs().amax()
            target = torch.clamp_min(
                amax / 2688.0, torch.finfo(torch.float32).tiny
            )
            scale = torch.exp2(torch.ceil(torch.log2(target)))
            scale = torch.where(amax > 0, scale, torch.ones_like(scale))
            return NVFP4Tensor.to_nvfp4(
                tensor,
                block_size=16,
                per_tensor_scale=scale,
                is_swizzled_scales=False,
                use_triton_kernel=False,
            ).dequantize(torch.float32)

        reference = reference_quantize(x) @ reference_quantize(weight).T
        error = (actual.float() - reference).abs()
        self.assertLess(float(error.mean()), 0.08)
        self.assertLess(float(error.max()), 0.6)

    def test_standalone_quant_and_packed_inference_match_torchao(self) -> None:
        from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

        torch.manual_seed(1907)
        x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)

        def exact_scale(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.float().abs().amax() / 2688.0

        quant_config = NVFP4QuantConfig(values_per_lane=16, load_bits=128)
        packed_x = rtx.quantize_nvfp4(
            x, tensor_scale=exact_scale(x), config=quant_config
        )
        packed_weight = rtx.quantize_nvfp4(
            weight, tensor_scale=exact_scale(weight), config=quant_config
        )
        torchao_x = NVFP4Tensor.to_nvfp4(
            x,
            block_size=16,
            per_tensor_scale=exact_scale(x),
            use_triton_kernel=False,
        )
        torch.testing.assert_close(
            packed_x.qdata.view(torch.uint8),
            torchao_x.qdata.view(torch.uint8),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            packed_x.scale.float(), torchao_x.scale.float(), rtol=0, atol=0
        )
        actual = rtx.nvfp4_linear(packed_x, packed_weight)
        reference = (
            packed_x.dequantize(torch.float32)
            @ packed_weight.dequantize(torch.float32).T
        )
        error = (actual.float() - reference).abs()
        self.assertLess(float(error.mean()), 0.08)
        self.assertLess(float(error.max()), 0.6)

        packed_module = rtx.NVFP4Linear(
            128, 128, device="cuda", packed_weight=packed_weight
        )
        with torch.inference_mode():
            dynamic_x = packed_module(x)
            expected_dynamic_x = rtx.nvfp4_linear(
                rtx.quantize_nvfp4(x), packed_weight
            )
        torch.testing.assert_close(dynamic_x, expected_dynamic_x, rtol=0, atol=0)

    def test_delayed_scale_generation_and_mxfp8_backward(self) -> None:
        torch.manual_seed(1902)
        layer = rtx.NVFP4Linear(128, 128, device="cuda")
        x = torch.randn(
            128, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        first = layer(x)
        first_amax = layer._x_amax_state.detach().clone()
        second = layer(x * 2.0)
        second.float().square().mean().backward()
        torch.cuda.synchronize()
        self.assertTrue(bool(torch.isfinite(first).all()))
        self.assertTrue(bool(torch.isfinite(second).all()))
        self.assertTrue(bool(torch.isfinite(x.grad).all()))
        self.assertTrue(bool(torch.isfinite(layer.weight.grad).all()))
        torch.testing.assert_close(
            layer._x_amax_state.max(),
            first_amax.max() * 2.0,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            layer._x_amax_state.max(),
            (x.detach() * 2.0).float().abs().amax(),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            layer._weight_amax_state.max(),
            layer.weight.detach().float().abs().amax(),
            rtol=0,
            atol=0,
        )

    def test_delayed_scaling_recovers_one_step_after_distribution_jump(self) -> None:
        torch.manual_seed(1903)
        layer = rtx.NVFP4Linear(128, 128, device="cuda")
        x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        layer(x)
        jumped = x * 64.0
        stale = layer(jumped)
        recovered = layer(jumped)
        current = rtx.nvfp4_linear(jumped, layer.weight)
        torch.cuda.synchronize()
        self.assertTrue(bool(torch.isfinite(stale).all()))
        torch.testing.assert_close(recovered, current, rtol=0, atol=0)

    def test_delayed_telemetry_covers_multiple_ctas_and_variable_m(self) -> None:
        torch.manual_seed(1906)
        layer = rtx.NVFP4Linear(256, 256, device="cuda")
        x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        first = layer(x)
        torch.cuda.synchronize()
        self.assertGreater(layer._x_amax_state.numel(), 1)
        torch.testing.assert_close(
            layer._x_amax_state.max(), x.float().abs().amax(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            layer._weight_amax_state.max(),
            layer.weight.detach().float().abs().amax(),
            rtol=0,
            atol=0,
        )
        changed_m = layer(x[:128])
        torch.cuda.synchronize()
        self.assertTrue(bool(torch.isfinite(first).all()))
        self.assertTrue(bool(torch.isfinite(changed_m).all()))
        self.assertEqual(layer._delayed_problem[2], 128)

    def test_delayed_training_is_fullgraph_compileable(self) -> None:
        torch.manual_seed(1904)
        layer = rtx.NVFP4Linear(128, 128, device="cuda")
        bootstrap = torch.randn(
            128,
            128,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        layer(bootstrap)
        compiled = torch.compile(
            layer,
            fullgraph=True,
            dynamic=False,
            options={"triton.cudagraphs": False},
        )
        x = torch.randn_like(bootstrap, requires_grad=True)
        out = compiled(x)
        out.float().square().mean().backward()
        torch.cuda.synchronize()
        self.assertTrue(bool(torch.isfinite(out).all()))
        self.assertTrue(bool(torch.isfinite(x.grad).all()))


if __name__ == "__main__":
    unittest.main()
