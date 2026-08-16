from __future__ import annotations

from dataclasses import asdict, replace
import unittest

import torch

import rtx
from rtx.autotune.adapters import (
    make_nvfp4_delayed_adapter,
    make_nvfp4_dynamic_adapter,
    make_nvfp4_fully_prequant_adapter,
    make_nvfp4_jit_row_region_adapter,
    make_nvfp4_weight_prequant_adapter,
)
from rtx.autotune.outcomes import TrialOutcome
from rtx.autotune.promotion import _config_rejection, _current_revision
from rtx.configs.nvfp4 import (
    DEFAULT_NVFP4_SCALE_CONFIG,
    NVFP4DynamicConfig,
    NVFP4FullyPrequantConfig,
    NVFP4ScaleConfig,
    NVFP4GemmConfig,
    NVFP4Problem,
    NVFP4QuantConfig,
    NVFP4WeightPrequantConfig,
)
from rtx.nvfp4_inference_autotune import (
    NVFP4_DELAYED_SEARCH_SPACE,
    NVFP4_DYNAMIC_SEARCH_SPACE,
    NVFP4_JIT_ROW_REGION_SEARCH_SPACE,
    dynamic_config_from_dict,
    dynamic_config_to_dict,
    preferred_jit_row_region_config,
    preferred_dynamic_config,
    update_dynamic_config,
)


def _has_sm12x() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12


class NVFP4ConfigTests(unittest.TestCase):
    def test_regional_epilogue_strategies_are_tunable_and_bounded(self) -> None:
        problem = NVFP4Problem(257, 769, 768)
        base = preferred_jit_row_region_config(problem)
        for strategy in (
            "direct",
            "expanded_factors",
            "factorized",
            "product",
            "separate",
        ):
            with self.subTest(strategy=strategy):
                candidate = replace(
                    base,
                    gemm=replace(
                        base.gemm, regional_scale_epilogue=strategy
                    ),
                )
                self.assertIsNone(candidate.rejection(problem))
        too_fine = replace(
            base,
            gemm=replace(base.gemm, regional_scale_epilogue="product"),
            x_scale_region_rows=1,
            weight_scale_region_rows=1,
        )
        self.assertIn("4-KiB", too_fine.rejection(problem) or "")

    def test_jit_row_region_config_roundtrips_and_exposes_deep_axes(self) -> None:
        problem = NVFP4Problem(257, 769, 768)
        config = preferred_jit_row_region_config(problem)
        self.assertTrue(config.jit_row_region)
        self.assertEqual(config.x_scale_region_rows, 5)
        self.assertEqual(config.weight_scale_region_rows, 4)
        self.assertIsNone(config.rejection(problem))
        self.assertEqual(
            dynamic_config_from_dict(dynamic_config_to_dict(config)), config
        )
        self.assertTrue(
            {
                "x_region_rows",
                "weight_region_rows",
                "outer_scale_math",
                "region_amax_load",
                "region_amax_unroll",
                "region_grid",
                "region_order",
                "region_ownership",
                "dependent_launch",
                "quant_vector_load",
                "gemm_geometry",
                "regional_epilogue_schedule",
                "regional_epilogue_warps",
                "regional_epilogue_registers",
            }.issubset(NVFP4_JIT_ROW_REGION_SEARCH_SPACE)
        )

        pdl = update_dynamic_config(
            config, {"programmatic_dependent_launch": True}
        )
        self.assertTrue(pdl.programmatic_dependent_launch)
        self.assertEqual(
            dynamic_config_from_dict(dynamic_config_to_dict(pdl)), pdl
        )

    def test_epilogue_warp_seed_is_bounded_to_measured_basin(self) -> None:
        bandwidth_heavy = preferred_jit_row_region_config(
            NVFP4Problem(32768, 2304, 768)
        )
        self.assertEqual(
            bandwidth_heavy.gemm.regional_epilogue_schedule,
            "warp_specialized",
        )
        self.assertEqual(bandwidth_heavy.gemm.regional_epilogue_warps, 8)
        self.assertEqual(bandwidth_heavy.gemm.tiles_per_cta, 8)
        self.assertEqual(bandwidth_heavy.gemm.stages, 1)
        compute_heavy = preferred_jit_row_region_config(
            NVFP4Problem(32768, 768, 1536)
        )
        self.assertEqual(compute_heavy.gemm.regional_epilogue_schedule, "mma")
        small_grid = preferred_jit_row_region_config(
            NVFP4Problem(512, 1536, 768)
        )
        self.assertEqual(small_grid.gemm.regional_epilogue_schedule, "mma")

    def test_jit_row_region_adapter_has_distinct_runtime_identity(self) -> None:
        problem = NVFP4Problem(256, 1536, 1536)
        adapter = make_nvfp4_jit_row_region_adapter(
            problem,
            lambda _config: TrialOutcome("ok", median_ms=1.0),
        )
        self.assertEqual(adapter.context.family, "nvfp4_jit_row_region_fwd")
        self.assertEqual(adapter.context.tags["scale_policy"], "jit_row_region")
        features = adapter.features(adapter.initial_config)
        self.assertEqual(features["derived.jit_row_region"], 1.0)
        self.assertGreater(features["derived.x_scale_regions"], 0.0)
        self.assertGreater(features["derived.region_observer_read_bytes"], 0.0)
        self.assertEqual(
            features["derived.programmatic_dependent_launch"], 0.0
        )
        self.assertIn("x_region_rows", adapter.coordinates())
        self.assertIn("gemm_geometry", adapter.coordinates())

    def test_jit_fields_do_not_change_existing_dynamic_dataset_schema(self) -> None:
        serialized = dynamic_config_to_dict(NVFP4DynamicConfig())
        self.assertNotIn("x_scale_region_rows", serialized)
        self.assertNotIn("region_amax_load_bits", serialized)
        self.assertNotIn("programmatic_dependent_launch", serialized)
        self.assertEqual(
            dynamic_config_from_dict(serialized), NVFP4DynamicConfig()
        )

    def test_problem_keeps_logical_k_and_exposes_minimal_storage_k(self) -> None:
        problem = NVFP4Problem(3, 5, 17)
        problem.validate()
        self.assertEqual(problem.k, 17)
        self.assertEqual(problem.storage_k, 32)

    def test_scale_config_contains_only_numeric_policy(self) -> None:
        config = NVFP4ScaleConfig(
            tensor_scale_mode="exact",
            amax_history_len=4,
            amax_history_algo="most_recent",
        )
        self.assertEqual(config.tensor_scale_mode, "exact")
        self.assertEqual(config.amax_history_len, 4)
        self.assertEqual(config.amax_history_algo, "most_recent")
        self.assertFalse(hasattr(config, "tile_m"))
        self.assertEqual(DEFAULT_NVFP4_SCALE_CONFIG.amax_history_len, 16)
        with self.assertRaisesRegex(ValueError, "amax_history_len"):
            NVFP4ScaleConfig(amax_history_len=2)  # type: ignore[arg-type]

    def test_standalone_quantizer_exposes_one_lane_per_block(self) -> None:
        config = NVFP4QuantConfig(values_per_lane=16, load_bits=128)
        self.assertEqual(config.threads_per_scale, 1)
        self.assertEqual(config.blocks_per_warp, 32)
        self.assertIsNone(config.rejection(128, 128))

    def test_runtime_promotion_understands_nvfp4_revision_and_schema(self) -> None:
        problem = NVFP4Problem(256, 1536, 1536)
        for family, inference_config in (
            ("nvfp4_dynamic_fwd", NVFP4DynamicConfig()),
            ("nvfp4_delayed_fwd", NVFP4DynamicConfig()),
            (
                "nvfp4_jit_row_region_fwd",
                preferred_jit_row_region_config(problem),
            ),
            ("nvfp4_weight_prequant_fwd", NVFP4WeightPrequantConfig()),
            ("nvfp4_fully_prequant_fwd", NVFP4FullyPrequantConfig()),
        ):
            with self.subTest(family=family):
                expected_revision = {
                    "nvfp4_dynamic_fwd": 6,
                    "nvfp4_delayed_fwd": 1,
                    "nvfp4_jit_row_region_fwd": 5,
                    "nvfp4_weight_prequant_fwd": 3,
                    "nvfp4_fully_prequant_fwd": 3,
                }[family]
                self.assertEqual(
                    _current_revision(family),
                    expected_revision,
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
        delayed = make_nvfp4_delayed_adapter(problem, evaluator)
        self.assertEqual(dynamic.context.family, "nvfp4_dynamic_fwd")
        self.assertEqual(delayed.context.family, "nvfp4_delayed_fwd")
        self.assertEqual(delayed.initial_config.quant_launches, "dual")
        self.assertNotIn("quant_launches", NVFP4_DELAYED_SEARCH_SPACE)
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

    def test_dynamic_native_scale_and_k64_legality_is_static(self) -> None:
        problem = NVFP4Problem(128, 128, 128)
        base = NVFP4DynamicConfig()
        invalid_k64 = replace(
            base,
            gemm=replace(base.gemm, tile_k=64, a_swizzle="64b"),
        )
        self.assertIn("K=64", invalid_k64.rejection(problem))
        legal_k64 = replace(
            base,
            gemm=replace(
                base.gemm,
                tile_k=64,
                a_swizzle="32b",
                b_swizzle="32b",
            ),
        )
        self.assertIsNone(legal_k64.rejection(problem))

        native = replace(
            base,
            quant=replace(base.quant, scale_layout="mma128"),
            gemm=replace(
                base.gemm,
                scale_layout="mma128",
                scale_role="tma",
            ),
        )
        self.assertIsNone(native.rejection(problem))
        self.assertIn(
            "selected together",
            replace(native, quant=base.quant).rejection(problem),
        )

    def test_dynamic_implementation_anchor_models_balanced_sm_grid(self) -> None:
        problem = NVFP4Problem(1536, 1536, 1536)

        def evaluator(config):
            return TrialOutcome("ok", median_ms=1.0)

        adapter = make_nvfp4_dynamic_adapter(
            problem,
            evaluator,
            device={"multiprocessor_count": 70},
        )
        anchor_update = NVFP4_DYNAMIC_SEARCH_SPACE[
            "implementation_anchor"
        ][0]
        anchor = update_dynamic_config(
            adapter.initial_config, anchor_update
        )
        self.assertIsNone(anchor.rejection(problem))
        self.assertEqual(anchor.quant.scale_layout, "mma128")
        self.assertEqual(anchor.gemm.scale_layout, "mma128")
        self.assertEqual(anchor.gemm.scale_role, "tma")
        self.assertEqual(anchor.gemm.stages, 3)
        self.assertEqual(anchor.gemm.epilogue_stages, 1)
        self.assertEqual(anchor.gemm.persistent_waves, 1)
        self.assertEqual(anchor.gemm.tiles_per_cta, 4)
        features = adapter.features(anchor)
        self.assertEqual(features["derived.grid_ctas"], 70.0)
        self.assertEqual(features["derived.work_tiles_per_cta"], 3.0)
        self.assertEqual(features["derived.balanced_persistent_grid"], 1.0)

        concurrent = replace(anchor, quant_launches="concurrent")
        concurrent_features = adapter.features(concurrent)
        self.assertEqual(
            concurrent_features["derived.quant_launch_concurrency"], 2.0
        )

    def test_dynamic_search_starts_in_native_scale_basin_when_legal(self) -> None:
        eligible = NVFP4Problem(32_768, 3_072, 768)
        ragged = NVFP4Problem(127, 129, 130)

        preferred = preferred_dynamic_config(eligible)
        self.assertEqual(preferred.quant.scale_layout, "mma128")
        self.assertEqual(preferred.gemm.scale_layout, "mma128")
        self.assertEqual(preferred.gemm.scale_role, "tma")
        self.assertIsNone(preferred.rejection(eligible))
        self.assertEqual(
            preferred_dynamic_config(ragged),
            NVFP4DynamicConfig(),
        )

        adapter = make_nvfp4_dynamic_adapter(
            eligible,
            lambda _config: TrialOutcome("ok", median_ms=1.0),
        )
        self.assertEqual(adapter.initial_config, preferred)


@unittest.skipUnless(_has_sm12x(), "requires an SM120/SM121 CUDA GPU")
class NVFP4CudaTests(unittest.TestCase):
    def test_jit_regional_warp_specialized_epilogue(self) -> None:
        from rtx.fp4 import _make_jit_region_dynamic_runner

        torch.manual_seed(1931)
        problem = NVFP4Problem(256, 256, 256)
        base = preferred_jit_row_region_config(problem)
        config = replace(
            base,
            gemm=replace(
                base.gemm,
                stages=1,
                epilogue="direct",
                epilogue_stages=1,
                store_vec=1,
                regional_scale_epilogue="factorized",
                regional_epilogue_schedule="warp_specialized",
                regional_epilogue_warps=4,
                regional_epilogue_registers=48,
                tiles_per_cta=2,
                tile_locality="raster",
            ),
            programmatic_dependent_launch=False,
        )
        self.assertIsNone(config.rejection(problem))
        x = torch.randn(
            problem.m, problem.k, device="cuda", dtype=torch.bfloat16
        )
        weight = torch.randn(
            problem.n, problem.k, device="cuda", dtype=torch.bfloat16
        )
        out = torch.empty(
            problem.m, problem.n, device="cuda", dtype=torch.bfloat16
        )
        mma_out = torch.empty_like(out)
        runner = _make_jit_region_dynamic_runner(problem, config, x.device)
        mma_config = replace(
            config,
            gemm=replace(
                config.gemm, regional_epilogue_schedule="mma"
            ),
        )
        mma_runner = _make_jit_region_dynamic_runner(
            problem, mma_config, x.device
        )
        runner(x, weight, out)
        mma_runner(x, weight, mma_out)
        torch.cuda.synchronize()
        reference = x.float() @ weight.float().T
        self.assertTrue(bool(torch.isfinite(out).all()))
        torch.testing.assert_close(out, mma_out, rtol=0, atol=0)
        self.assertLess(
            float(torch.linalg.vector_norm(out.float() - reference))
            / float(torch.linalg.vector_norm(reference)),
            0.25,
        )

    def test_repeated_fullgraph_linears_preserve_every_weight_gradient(self) -> None:
        torch.compiler.reset()
        torch.manual_seed(1898)

        def make_stack() -> torch.nn.Sequential:
            return torch.nn.Sequential(
                *(
                    rtx.NVFP4Linear(
                        128,
                        128,
                        device="cuda",
                        dtype=torch.bfloat16,
                        scaling="block",
                        autotune="off",
                    )
                    for _ in range(4)
                )
            )

        eager = make_stack()
        compiled_model = make_stack()
        compiled_model.load_state_dict(eager.state_dict())
        compiled = torch.compile(
            compiled_model,
            fullgraph=True,
            dynamic=False,
            options={"triton.cudagraphs": False},
        )
        eager_x = torch.randn(
            128, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        compiled_x = eager_x.detach().clone().requires_grad_(True)
        grad_output = torch.randn_like(eager_x)
        expected = eager(eager_x)
        actual = compiled(compiled_x)
        expected.backward(grad_output)
        actual.backward(grad_output)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(compiled_x.grad, eager_x.grad, rtol=0, atol=0)
        for expected_parameter, actual_parameter in zip(
            eager.parameters(), compiled_model.parameters(), strict=True
        ):
            torch.testing.assert_close(
                actual_parameter.grad,
                expected_parameter.grad,
                rtol=0,
                atol=0,
            )

    def test_materialized_native_scales_and_balanced_grid_execute(self) -> None:
        from rtx.nvfp4_inference_experiments import (
            NVFP4DynamicBenchmarkHarness,
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
        problem = NVFP4Problem(128, 512, 128)
        harness = NVFP4DynamicBenchmarkHarness(
            ShapeSpec(128, 512, 128), "hot", protocol, seed=1918
        )
        anchor = update_dynamic_config(
            NVFP4DynamicConfig(),
            NVFP4_DYNAMIC_SEARCH_SPACE["implementation_anchor"][0],
        )
        native = replace(
            NVFP4DynamicConfig(),
            quant=replace(
                NVFP4DynamicConfig().quant, scale_layout="mma128"
            ),
            gemm=replace(
                NVFP4DynamicConfig().gemm,
                scale_layout="mma128",
                scale_role="tma",
            ),
        )
        for name, config in (
            ("balanced_native", anchor),
            ("native_scales", native),
        ):
            with self.subTest(schedule=name):
                self.assertIsNone(config.rejection(problem))
                result = harness.measure(config, samples=3, seed=1918)
                self.assertEqual(result["status"], "ok")
                self.assertEqual(result["max_abs_error"], 0.0)

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

    def test_public_current_scale_forward_is_finite(self) -> None:
        torch.manual_seed(1901)
        x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        out = rtx.nvfp4_linear(x, weight)
        reference = x.float() @ weight.float().T
        error = (out.float() - reference).abs()
        self.assertTrue(bool(torch.isfinite(out).all()))
        self.assertLess(float(error.mean()), 2.0)

    def test_current_matches_torchao_quantization_reference(self) -> None:
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

    def test_default_dynamic_module_converts_to_usable_packed_weight(self) -> None:
        torch.manual_seed(1925)
        layer = rtx.NVFP4Linear(
            128, 128, device="cuda", autotune="off"
        ).eval()
        x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            packed = layer.to_quantized_weight()
            output = packed(x)
        torch.cuda.synchronize()
        self.assertEqual(layer.scaling, "jit_row_region")
        self.assertEqual(packed.scaling, "current")
        self.assertEqual(packed.weight_mode, "prequantized")
        self.assertTrue(bool(torch.isfinite(output).all()))

    def test_block_module_preserves_block_policy_when_packed(self) -> None:
        torch.manual_seed(1926)
        layer = rtx.NVFP4Linear(
            128,
            128,
            device="cuda",
            scaling="block",
            autotune="off",
        ).eval()
        packed = layer.to_quantized_weight()
        self.assertEqual(packed.scaling, "block")
        torch.testing.assert_close(
            packed.weight_tensor_scale,
            torch.ones_like(packed.weight_tensor_scale),
            rtol=0,
            atol=0,
        )
        expected = rtx.quantize_nvfp4(
            layer.weight.detach(),
            tensor_scale=torch.ones((), device="cuda", dtype=torch.float32),
        )
        self.assertTrue(torch.equal(packed.weight_data, expected.qdata))
        self.assertTrue(torch.equal(packed.weight_block_scales, expected.scale))

    def test_from_float_block_policy_packs_with_unit_tensor_scale(self) -> None:
        source = torch.nn.Linear(
            128, 128, bias=False, device="cuda", dtype=torch.bfloat16
        ).eval()
        packed = rtx.NVFP4Linear.from_float(
            source, scaling="block", autotune="off"
        )
        self.assertEqual(packed.scaling, "block")
        torch.testing.assert_close(
            packed.weight_tensor_scale,
            torch.ones_like(packed.weight_tensor_scale),
            rtol=0,
            atol=0,
        )

    def test_ragged_k_packed_inference_uses_minimal_fp4_storage(self) -> None:
        torch.manual_seed(1919)
        m, n, k = 17, 29, 33
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            packed_x = rtx.quantize_nvfp4(x)
            packed_weight = rtx.quantize_nvfp4(weight)
            fully_packed = rtx.nvfp4_linear(
                packed_x, packed_weight, autotune="off"
            )
            layer = rtx.NVFP4Linear(
                k,
                n,
                packed_weight=packed_weight,
                scaling="current",
                autotune="off",
            )
            dynamic_x = layer(x)
            torch.compiler.reset()
            compiled = torch.compile(
                layer,
                fullgraph=True,
                dynamic=False,
                options={"triton.cudagraphs": False},
            )
            compiled_dynamic_x = compiled(x)
            compiled_fully_packed = compiled(packed_x)
        torch.cuda.synchronize()
        self.assertEqual(tuple(packed_x.qdata.shape), (m, 24))
        self.assertEqual(tuple(packed_weight.qdata.shape), (n, 24))
        self.assertEqual(tuple(fully_packed.shape), (m, n))
        self.assertEqual(tuple(dynamic_x.shape), (m, n))
        self.assertEqual(tuple(compiled_dynamic_x.shape), (m, n))
        self.assertEqual(tuple(compiled_fully_packed.shape), (m, n))
        self.assertTrue(bool(torch.isfinite(fully_packed).all()))
        self.assertTrue(bool(torch.isfinite(dynamic_x).all()))
        self.assertTrue(bool(torch.isfinite(compiled_dynamic_x).all()))
        self.assertTrue(bool(torch.isfinite(compiled_fully_packed).all()))

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

    def test_ragged_and_minimum_k_materialized_shapes(self) -> None:
        torch.manual_seed(1909)
        for m, n, k in (
            (65, 130, 128),
            (64, 128, 64),
            (17, 29, 1),
            (33, 65, 17),
            (63, 127, 33),
        ):
            with self.subTest(shape=(m, n, k)):
                x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
                actual = rtx.nvfp4_linear(x, weight, backend="materialized")
                self.assertEqual(tuple(actual.shape), (m, n))
                self.assertTrue(bool(torch.isfinite(actual).all()))

    def test_ragged_nvfp4_training_is_fullgraph_compileable(self) -> None:
        torch.manual_seed(1918)
        layer = rtx.NVFP4Linear(
            33,
            29,
            device="cuda",
            backend="materialized",
            autotune="off",
        ).train()
        compiled = torch.compile(layer, fullgraph=True, dynamic=False)
        x = torch.randn(
            17, 33, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        output = compiled(x)
        output.float().square().mean().backward()
        torch.cuda.synchronize()
        self.assertEqual(tuple(output.shape), (17, 29))
        self.assertTrue(bool(torch.isfinite(x.grad).all()))
        self.assertTrue(bool(torch.isfinite(layer.weight.grad).all()))

    def test_exact_and_power2_tensor_scale_policies(self) -> None:
        torch.manual_seed(1910)
        x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        exact_config = NVFP4ScaleConfig(tensor_scale_mode="exact")
        exact = rtx.nvfp4_linear(x, weight, scale_config=exact_config)
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
            x, weight, scaling="jit_row_region", scale_region_rows=256
        )
        current = rtx.nvfp4_linear(x, weight, scaling="current")
        reference = x.float() @ weight.float().T
        self.assertTrue(bool(torch.isfinite(regional).all()))
        self.assertLess(
            float((regional.float() - reference).abs().mean()),
            float((current.float() - reference).abs().mean()) * 2.0,
        )

    def test_ragged_regional_epilogue_strategies_are_bitwise_equivalent(
        self,
    ) -> None:
        from rtx.fp4 import _make_jit_region_dynamic_runner

        torch.manual_seed(19151)
        problem = NVFP4Problem(257, 259, 256)
        x = torch.randn(
            problem.m, problem.k, device="cuda", dtype=torch.bfloat16
        )
        weight = torch.randn(
            problem.n, problem.k, device="cuda", dtype=torch.bfloat16
        )
        base = preferred_jit_row_region_config(problem)
        outputs = []
        for strategy in (
            "direct",
            "expanded_factors",
            "factorized",
            "product",
            "separate",
        ):
            config = replace(
                base,
                gemm=replace(
                    base.gemm, regional_scale_epilogue=strategy
                ),
                programmatic_dependent_launch=False,
            )
            runner = _make_jit_region_dynamic_runner(
                problem, config, x.device
            )
            out = torch.empty(
                problem.m,
                problem.n,
                device=x.device,
                dtype=torch.bfloat16,
            )
            runner(x, weight, out)
            outputs.append(out)
        torch.cuda.synchronize()
        for output in outputs[1:]:
            torch.testing.assert_close(output, outputs[0], rtol=0, atol=0)

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
        rowwise = rtx.nvfp4_linear(x, weight, scaling="jit_row_region")
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
            scaling="jit_row_region",
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
            return rtx.nvfp4_linear(a, b, scaling="jit_row_region")

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
        layer = rtx.NVFP4Linear(
            128, 128, device="cuda", scaling="delayed"
        )
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

    def test_delayed_scaling_remains_delayed_without_gradients(self) -> None:
        torch.manual_seed(1921)
        layer = rtx.NVFP4Linear(
            128, 128, device="cuda", scaling="delayed", autotune="off"
        ).eval()
        first_x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        second_x = first_x * 8.0
        with torch.inference_mode():
            first = layer(first_x)
            first_history = layer._x_amax_state.clone()
            second = layer(second_x)
        torch.cuda.synchronize()
        self.assertTrue(bool(torch.isfinite(first).all()))
        self.assertTrue(bool(torch.isfinite(second).all()))
        torch.testing.assert_close(
            first_history.max(), first_x.float().abs().amax(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            layer._x_amax_state.max(),
            second_x.float().abs().amax(),
            rtol=0,
            atol=0,
        )

    def test_compiled_delayed_eval_rotates_history(self) -> None:
        torch.manual_seed(1922)
        layer = rtx.NVFP4Linear(
            128, 128, device="cuda", scaling="delayed", autotune="off"
        ).eval()
        compiled = torch.compile(
            layer,
            fullgraph=True,
            dynamic=False,
            options={"triton.cudagraphs": False},
        )
        x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            compiled(x)
            compiled(x * 4.0)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            layer._x_amax_state.max(),
            (x * 4.0).float().abs().amax(),
            rtol=0,
            atol=0,
        )

    def test_delayed_scaling_recovers_one_step_after_distribution_jump(self) -> None:
        torch.manual_seed(1903)
        x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        for mode in ("power2", "exact"):
            with self.subTest(tensor_scale_mode=mode):
                config = NVFP4ScaleConfig(tensor_scale_mode=mode)
                layer = rtx.NVFP4Linear(
                    128,
                    128,
                    device="cuda",
                    scale_config=config,
                    scaling="delayed",
                )
                layer(x)
                jumped = x * 64.0
                stale = layer(jumped)
                recovered = layer(jumped)
                current = rtx.nvfp4_linear(
                    jumped,
                    layer.weight,
                    scale_config=config,
                    scaling="current",
                    backend="materialized",
                )
                torch.cuda.synchronize()
                self.assertTrue(bool(torch.isfinite(stale).all()))
                torch.testing.assert_close(recovered, current, rtol=0, atol=0)


    def test_jit_region_training_is_fullgraph_compileable(self) -> None:
        torch.manual_seed(1924)
        layer = rtx.NVFP4Linear(
            256,
            256,
            device="cuda",
            scaling="jit_row_region",
            autotune="off",
        ).train()
        compiled = torch.compile(
            layer,
            fullgraph=True,
            dynamic=False,
            options={"triton.cudagraphs": False},
        )
        x = torch.randn(
            256, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        output = compiled(x)
        output.float().square().mean().backward()
        torch.cuda.synchronize()
        self.assertTrue(bool(torch.isfinite(output).all()))
        self.assertTrue(bool(torch.isfinite(x.grad).all()))
        self.assertTrue(bool(torch.isfinite(layer.weight.grad).all()))

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
        layer = rtx.NVFP4Linear(
            256, 256, device="cuda", scaling="delayed"
        )
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
        layer = rtx.NVFP4Linear(
            128, 128, device="cuda", scaling="delayed"
        )
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
        layer = rtx.NVFP4Linear(
            128, 128, device="cuda", scaling="delayed"
        )
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
        layer = rtx.NVFP4Linear(
            128, 128, device="cuda", scaling="delayed"
        )
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
