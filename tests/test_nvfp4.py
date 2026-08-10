from __future__ import annotations

from dataclasses import asdict, replace
import unittest

import torch

import rtx
from rtx.autotune.adapters import (
    make_nvfp4_dynamic_adapter,
    make_nvfp4_fully_prequant_adapter,
    make_nvfp4_fwd_adapter,
    make_nvfp4_weight_prequant_adapter,
)
from rtx.autotune.outcomes import TrialOutcome
from rtx.autotune.promotion import _config_rejection, _current_revision
from rtx.configs.nvfp4 import (
    DEFAULT_NVFP4_FWD_CONFIG,
    NVFP4DynamicConfig,
    NVFP4FullyPrequantConfig,
    NVFP4FwdConfig,
    NVFP4GemmConfig,
    NVFP4Problem,
    NVFP4QuantConfig,
    NVFP4WeightPrequantConfig,
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
        self.assertEqual(config.telemetry_layout, "scalar_atomic")
        self.assertEqual(config.telemetry_ownership, "operand_owner")
        self.assertEqual(config.amax_history_len, 16)
        self.assertEqual(config.amax_history_algo, "window_max")

    def test_row_region_scale_legality_tracks_cta_tiles(self) -> None:
        problem = NVFP4Problem(512, 512, 128)
        regional = replace(
            DEFAULT_NVFP4_FWD_CONFIG,
            tile_k=128,
            bf16_tile_k=128,
            x_scale_region_rows=256,
            weight_scale_region_rows=256,
        )
        self.assertIsNone(regional.implementation_rejection(problem))
        self.assertIn(
            "direct",
            replace(regional, epilogue="tma").implementation_rejection(problem),
        )
        self.assertIn(
            "distinct policies",
            replace(regional, collect_amax=True).implementation_rejection(problem),
        )
        normalized = normalize_nvfp4_fwd_config(
            regional, x_scale_region_rows=512, weight_scale_region_rows=256
        )
        self.assertEqual(normalized.x_scale_region_rows, 512)
        self.assertEqual(normalized.weight_scale_region_rows, 256)

    def test_row_region_scale_pack_is_region_major(self) -> None:
        from rtx.fp4 import _regional_tensor_scale_pack

        value = torch.ones(4, 8, dtype=torch.bfloat16)
        value[2:].mul_(32.0)
        pack = _regional_tensor_scale_pack(value, 2, "power2").reshape(2, 3)
        self.assertEqual(tuple(pack.shape), (2, 3))
        self.assertGreater(float(pack[1, 0]), float(pack[0, 0]))
        torch.testing.assert_close(pack[:, 0] * pack[:, 1], torch.ones(2))
        torch.testing.assert_close(pack[:, 2], pack[:, 1] / 6.0)

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
        self.assertEqual(features["derived.delayed_telemetry_slots"], 1.0)
        self.assertEqual(
            features["derived.delayed_telemetry_state_bytes"], 128.0
        )
        self.assertEqual(features["derived.delayed_telemetry_memsets"], 2.0)
        self.assertEqual(features["derived.total_kernel_launches"], 1.0)

    def test_runtime_promotion_understands_nvfp4_revision_and_schema(self) -> None:
        problem = NVFP4Problem(256, 1536, 1536)
        config = replace(DEFAULT_NVFP4_FWD_CONFIG, collect_amax=True)
        self.assertEqual(_current_revision("nvfp4_fused_fwd"), 5)
        self.assertIsNone(
            _config_rejection(
                "nvfp4_fused_fwd",
                asdict(config),
                problem,  # type: ignore[arg-type]
            )
        )
        for family, inference_config in (
            ("nvfp4_dynamic_fwd", NVFP4DynamicConfig()),
            ("nvfp4_weight_prequant_fwd", NVFP4WeightPrequantConfig()),
            ("nvfp4_fully_prequant_fwd", NVFP4FullyPrequantConfig()),
        ):
            with self.subTest(family=family):
                self.assertEqual(
                    _current_revision(family),
                    2 if family == "nvfp4_dynamic_fwd" else 1,
                )
                self.assertIsNone(
                    _config_rejection(
                        family,
                        asdict(inference_config),
                        problem,  # type: ignore[arg-type]
                    )
                )
        invalid_dynamic = replace(
            NVFP4DynamicConfig(),
            gemm=replace(NVFP4DynamicConfig().gemm, b_swizzle="128b"),
        )
        self.assertIn("128-byte swizzles", invalid_dynamic.rejection(problem))

    def test_inference_state_adapters_are_distinct_and_fp4_aware(self) -> None:
        problem = NVFP4Problem(64, 128, 128)

        def evaluator(config):
            return TrialOutcome("ok", median_ms=1.0)

        weight = make_nvfp4_weight_prequant_adapter(
            problem,
            evaluator,
            initial=NVFP4WeightPrequantConfig(),
            axes={"x_vector_load": ({"quant": {"values_per_lane": 16}},)},
        )
        fully = make_nvfp4_fully_prequant_adapter(
            problem,
            evaluator,
            initial=NVFP4FullyPrequantConfig(),
            axes={"gemm_stages": ({"gemm": {"stages": 1}},)},
        )
        dynamic = make_nvfp4_dynamic_adapter(
            problem,
            evaluator,
            initial=NVFP4DynamicConfig(),
            axes={"quant_launches": ({"quant_launches": "dual"},)},
        )
        self.assertEqual(dynamic.context.family, "nvfp4_dynamic_fwd")
        self.assertEqual(weight.context.family, "nvfp4_weight_prequant_fwd")
        self.assertEqual(fully.context.family, "nvfp4_fully_prequant_fwd")
        weight_features = weight.features(weight.initial_config)
        fully_features = fully.features(fully.initial_config)
        self.assertEqual(weight_features["derived.quant_launch_count"], 1.0)
        self.assertEqual(fully_features["derived.quant_launch_count"], 0.0)
        self.assertEqual(
            weight_features["derived.quant_x_packed_output_bytes"],
            problem.m * problem.k / 2,
        )
        self.assertLess(
            fully_features["derived.smem_bytes_per_cta"],
            100_000,
        )


@unittest.skipUnless(_has_sm12x(), "requires an SM120/SM121 CUDA GPU")
class NVFP4CudaTests(unittest.TestCase):
    def test_calibrated_inference_harnesses_execute(self) -> None:
        from rtx.nvfp4_inference_experiments import (
            NVFP4FullyPrequantBenchmarkHarness,
            NVFP4WeightPrequantBenchmarkHarness,
        )
        from rtx.prequant_experiments import BenchmarkProtocol, ShapeSpec

        protocol = BenchmarkProtocol(
            warmup_calls=1,
            samples=3,
            confirm_samples=3,
            race_rounds=3,
            target_batch_ms=1.0,
            max_calls_per_sample=16,
            correctness_rtol=0,
            correctness_atol=0,
            telemetry=False,
        )
        shape = ShapeSpec(64, 128, 64)
        cases = (
            (
                NVFP4WeightPrequantBenchmarkHarness,
                NVFP4WeightPrequantConfig(),
            ),
            (
                NVFP4FullyPrequantBenchmarkHarness,
                NVFP4FullyPrequantConfig(),
            ),
        )
        for harness_type, config in cases:
            with self.subTest(harness=harness_type.__name__):
                harness = harness_type(shape, "hot", protocol, seed=1912)
                result = harness.measure(config, samples=3, seed=1912)
                self.assertEqual(result["status"], "ok")
                self.assertGreater(result["summary_ms"]["median"], 0)

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

    def test_native_64_row_packed_gemm_matches_dequantized_reference(self) -> None:
        torch.manual_seed(1908)
        x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        packed_x = rtx.quantize_nvfp4(x)
        packed_weight = rtx.quantize_nvfp4(weight)
        from rtx.fp4 import compile_nvfp4_gemm
        from rtx.formats.nvfp4 import nvfp4_tensor_scale

        config = NVFP4GemmConfig(
            tile_m=64,
            atom_layout_m=2,
            stages=1,
        )
        problem = NVFP4Problem(64, 128, 128)
        self.assertIsNone(config.rejection(problem))
        actual = torch.empty(64, 128, device="cuda", dtype=torch.bfloat16)
        output_scale = (
            nvfp4_tensor_scale(packed_x)
            * nvfp4_tensor_scale(packed_weight)
        ).reshape(1)
        compile_nvfp4_gemm(problem, config)(
            packed_x.qdata,
            packed_weight.qdata,
            packed_x.scale,
            packed_weight.scale,
            actual,
            output_scale,
        )
        reference = (
            packed_x.dequantize(torch.float32)
            @ packed_weight.dequantize(torch.float32).T
        )
        torch.testing.assert_close(actual.float(), reference, rtol=0.03, atol=0.15)

    def test_ragged_and_minimum_k_fused_shapes(self) -> None:
        torch.manual_seed(1909)
        for m, n, k in ((65, 130, 128), (64, 128, 64)):
            with self.subTest(shape=(m, n, k)):
                x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
                actual = rtx.nvfp4_linear(x, weight)
                self.assertEqual(tuple(actual.shape), (m, n))
                self.assertTrue(bool(torch.isfinite(actual).all()))

    def test_exact_and_power2_tensor_scale_policies(self) -> None:
        torch.manual_seed(1910)
        x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        exact_config = replace(
            DEFAULT_NVFP4_FWD_CONFIG,
            tile_k=128,
            bf16_tile_k=128,
            tensor_scale_mode="exact",
        )
        exact = rtx.nvfp4_linear(x, weight, forward_config=exact_config)
        power2 = rtx.nvfp4_linear(x, weight)
        self.assertTrue(bool(torch.isfinite(exact).all()))
        self.assertTrue(bool(torch.isfinite(power2).all()))
        reference = x.float() @ weight.float().T
        self.assertLess(float((exact.float() - reference).abs().mean()), 2.0)

    def test_block_only_scaling_eliminates_global_scale_reductions(self) -> None:
        torch.manual_seed(1912)
        x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        functional = rtx.nvfp4_linear(x, weight, scaling="block")
        layer = rtx.NVFP4Linear(
            128, 128, device="cuda", scaling="block"
        ).eval()
        with torch.no_grad():
            layer.weight.copy_(weight)
            modular = layer(x)
        torch.testing.assert_close(functional, modular, rtol=0, atol=0)
        self.assertTrue(bool(torch.isfinite(functional).all()))

    def test_row_region_scaling_uses_independent_cta_scale_packs(self) -> None:
        torch.manual_seed(1915)
        x = torch.randn(512, 128, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(512, 128, device="cuda", dtype=torch.bfloat16)
        x[256:].mul_(32.0)
        weight[256:].mul_(16.0)
        regional = rtx.nvfp4_linear(
            x, weight, scaling="regional", scale_region_rows=256
        )
        current = rtx.nvfp4_linear(x, weight, scaling="current")
        reference = x.float() @ weight.float().T
        self.assertTrue(bool(torch.isfinite(regional).all()))
        self.assertLess(
            float((regional.float() - reference).abs().mean()),
            float((current.float() - reference).abs().mean()) * 2.0,
        )

    def test_rowwise_scaling_preserves_heterogeneous_row_ranges(self) -> None:
        torch.manual_seed(1917)
        rows = 512
        x = torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16)
        dynamic_range = torch.exp2(
            torch.linspace(-20, 20, rows, device="cuda")
        ).to(torch.bfloat16)
        x = x * dynamic_range[:, None]
        weight = weight * dynamic_range[:, None]
        current = rtx.nvfp4_linear(x, weight, scaling="current")
        rowwise = rtx.nvfp4_linear(x, weight, scaling="regional")
        reference = x.float() @ weight.float().T
        denominator = reference.abs().clamp_min(1.0e-30)
        current_relative = ((current.float() - reference).abs() / denominator).median()
        rowwise_relative = ((rowwise.float() - reference).abs() / denominator).median()
        self.assertLess(float(rowwise_relative), float(current_relative) * 0.25)

    def test_row_region_scaling_is_fullgraph_compileable(self) -> None:
        torch.manual_seed(1916)
        layer = rtx.NVFP4Linear(
            128,
            512,
            device="cuda",
            scaling="regional",
        ).eval()
        compiled = torch.compile(
            layer,
            fullgraph=True,
            dynamic=False,
            options={"triton.cudagraphs": False},
        )
        x = torch.randn(512, 128, device="cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            out = compiled(x)
        self.assertEqual(tuple(out.shape), (512, 512))
        self.assertTrue(bool(torch.isfinite(out).all()))

        def functional(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return rtx.nvfp4_linear(a, b, scaling="regional")

        compiled_functional = torch.compile(
            functional,
            fullgraph=True,
            dynamic=False,
            options={"triton.cudagraphs": False},
        )
        with torch.inference_mode():
            functional_out = compiled_functional(x, layer.weight)
        self.assertTrue(bool(torch.isfinite(functional_out).all()))

    def test_block_only_is_range_fast_path_not_extreme_range_reference(self) -> None:
        x = torch.full(
            (128, 128), 8192.0, device="cuda", dtype=torch.bfloat16
        )
        weight = torch.eye(128, device="cuda", dtype=torch.bfloat16)
        current = rtx.nvfp4_linear(x, weight, scaling="current")
        block = rtx.nvfp4_linear(x, weight, scaling="block")
        reference = x.float() @ weight.float().T
        current_error = (current.float() - reference).abs().mean()
        block_error = (block.float() - reference).abs().mean()
        self.assertLess(float(current_error), float(block_error))

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
        x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        for mode in ("power2", "exact"):
            with self.subTest(tensor_scale_mode=mode):
                config = replace(
                    DEFAULT_NVFP4_FWD_CONFIG,
                    tile_k=128,
                    bf16_tile_k=128,
                    tensor_scale_mode=mode,
                )
                layer = rtx.NVFP4Linear(
                    128, 128, device="cuda", forward_config=config
                )
                layer(x)
                jumped = x * 64.0
                stale = layer(jumped)
                recovered = layer(jumped)
                current = rtx.nvfp4_linear(
                    jumped,
                    layer.weight,
                    forward_config=config,
                    backend="fused",
                )
                torch.cuda.synchronize()
                self.assertTrue(bool(torch.isfinite(stale).all()))
                torch.testing.assert_close(recovered, current, rtol=0, atol=0)

    def test_zero_and_tiny_inputs_have_finite_scale_fallbacks(self) -> None:
        weight = torch.zeros(128, 128, device="cuda", dtype=torch.bfloat16)
        tiny = torch.full(
            (128, 128),
            torch.finfo(torch.bfloat16).tiny,
            device="cuda",
            dtype=torch.bfloat16,
        )
        for x in (torch.zeros_like(tiny), tiny):
            with self.subTest(amax=float(x.float().abs().amax())):
                current = rtx.nvfp4_linear(x, weight)
                layer = rtx.NVFP4Linear(128, 128, device="cuda")
                with torch.no_grad():
                    layer.weight.zero_()
                delayed_first = layer(x)
                delayed_second = layer(x)
                self.assertTrue(bool(torch.isfinite(current).all()))
                self.assertTrue(bool(torch.isfinite(delayed_first).all()))
                self.assertTrue(bool(torch.isfinite(delayed_second).all()))
                self.assertEqual(int(torch.count_nonzero(delayed_second)), 0)

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
        self.assertEqual(layer._delayed_problem[3], 128)

    def test_delayed_history_rotates_without_losing_recent_maxima(self) -> None:
        torch.manual_seed(1913)
        layer = rtx.NVFP4Linear(128, 128, device="cuda")
        x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        layer(x)
        first = layer._x_amax_state.clone()
        layer(x * 0.5)
        torch.cuda.synchronize()
        self.assertEqual(layer._x_amax_state.numel(), 16)
        torch.testing.assert_close(
            layer._x_amax_state[0], first[0] * 0.5, rtol=0, atol=0
        )
        torch.testing.assert_close(
            layer._x_amax_state[1], first[0], rtol=0, atol=0
        )

    def test_delayed_state_reboot_on_stream_change_and_state_load(self) -> None:
        torch.manual_seed(1911)
        layer = rtx.NVFP4Linear(128, 128, device="cuda")
        x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        streams = (torch.cuda.Stream(), torch.cuda.Stream())
        outputs = []
        stream_ids = []
        for stream in streams:
            with torch.cuda.stream(stream):
                outputs.append(layer(x))
                stream_ids.append(layer._delayed_problem[2])
        torch.cuda.synchronize()
        self.assertNotEqual(stream_ids[0], stream_ids[1])
        self.assertTrue(all(bool(torch.isfinite(out).all()) for out in outputs))
        layer.load_state_dict(layer.state_dict())
        self.assertFalse(layer._delayed_initialized)
        self.assertEqual(layer._x_amax_state.numel(), 0)

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
        changed = torch.randn_like(bootstrap, requires_grad=True) * 3.0
        second = compiled(changed)
        torch.cuda.synchronize()
        self.assertTrue(bool(torch.isfinite(out).all()))
        self.assertTrue(bool(torch.isfinite(second).all()))
        self.assertTrue(bool(torch.isfinite(x.grad).all()))
        torch.testing.assert_close(
            layer._x_amax_state.max(),
            changed.detach().float().abs().amax(),
            rtol=0,
            atol=0,
        )

    def test_current_training_is_fullgraph_compileable(self) -> None:
        torch.manual_seed(1912)
        layer = rtx.NVFP4Linear(
            128, 128, device="cuda", scaling="current"
        )
        compiled = torch.compile(
            layer,
            fullgraph=True,
            dynamic=False,
            options={"triton.cudagraphs": False},
        )
        x = torch.randn(
            128,
            128,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        out = compiled(x)
        out.float().square().mean().backward()
        torch.cuda.synchronize()
        self.assertTrue(bool(torch.isfinite(out).all()))
        self.assertTrue(bool(torch.isfinite(x.grad).all()))
        self.assertTrue(bool(torch.isfinite(layer.weight.grad).all()))

        def functional(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return rtx.nvfp4_linear(a, b)

        compiled_functional = torch.compile(
            functional,
            fullgraph=True,
            dynamic=False,
            options={"triton.cudagraphs": False},
        )
        with torch.inference_mode():
            functional_out = compiled_functional(x.detach(), layer.weight.detach())
        self.assertTrue(bool(torch.isfinite(functional_out).all()))

    def test_block_only_training_is_fullgraph_compileable(self) -> None:
        torch.manual_seed(1914)
        layer = rtx.NVFP4Linear(
            128, 128, device="cuda", scaling="block"
        )
        compiled = torch.compile(
            layer,
            fullgraph=True,
            dynamic=False,
            options={"triton.cudagraphs": False},
        )
        x = torch.randn(
            128,
            128,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        out = compiled(x)
        out.float().square().mean().backward()
        torch.cuda.synchronize()
        self.assertTrue(bool(torch.isfinite(out).all()))
        self.assertTrue(bool(torch.isfinite(x.grad).all()))
        self.assertTrue(bool(torch.isfinite(layer.weight.grad).all()))

        def functional(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return rtx.nvfp4_linear(a, b, scaling="block")

        compiled_functional = torch.compile(
            functional,
            fullgraph=True,
            dynamic=False,
            options={"triton.cudagraphs": False},
        )
        with torch.inference_mode():
            functional_out = compiled_functional(x.detach(), layer.weight.detach())
        self.assertTrue(bool(torch.isfinite(functional_out).all()))

    def test_prequantized_inference_states_are_fullgraph_compileable(self) -> None:
        torch.manual_seed(1913)
        x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        packed_x = rtx.quantize_nvfp4(x)
        packed_weight = rtx.quantize_nvfp4(weight)
        layer = rtx.NVFP4Linear(
            128, 128, device="cuda", packed_weight=packed_weight
        )
        dynamic_x = torch.compile(
            layer,
            fullgraph=True,
            dynamic=False,
            options={"triton.cudagraphs": False},
        )(x)

        def fully_packed() -> torch.Tensor:
            return rtx.nvfp4_linear(packed_x, packed_weight)

        packed = torch.compile(
            fully_packed,
            fullgraph=True,
            dynamic=False,
            options={"triton.cudagraphs": False},
        )()
        torch.cuda.synchronize()
        self.assertTrue(bool(torch.isfinite(dynamic_x).all()))
        self.assertTrue(bool(torch.isfinite(packed).all()))


if __name__ == "__main__":
    unittest.main()
