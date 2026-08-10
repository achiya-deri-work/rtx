from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch
import rtx.fp8_bwd as fp8_bwd

from rtx.bwd_autotune import (
    BWD_SEARCH_SPACE,
    BwdCoordinateDescentTuner,
    bwd_config_from_dict,
    bwd_config_id,
    bwd_config_to_dict,
    load_mxfp8_bwd_config,
    update_bwd_config,
)
from rtx.bwd_experiments import BwdBenchmarkHarness
from rtx.fp8_bwd import mxfp8_linear_backward
from rtx.kernels.mxfp8 import MXFP8Problem, normalize_fwd_config
from rtx.kernels.mxfp8_fwd import (
    compile_mxfp8_atomic_split_fwd,
    compile_mxfp8_fwd,
    compile_mxfp8_split_fwd,
)
from rtx.kernels.mxfp8_reduce import compile_mxfp8_workspace_reduce
from rtx.kernels.mxfp8_bwd import (
    DEFAULT_DECOMPOSED_MXFP8_BWD_CONFIG,
    DEFAULT_DUAL_DECOMPOSED_MXFP8_BWD_CONFIG,
    DEFAULT_FUSED_MXFP8_BWD_CONFIG,
    DEFAULT_MXFP8_BWD_CONFIG,
    DEFAULT_SEPARATE_DECOMPOSED_MXFP8_BWD_CONFIG,
)
from rtx.kernels.mxfp8_quant import (
    MXFP8QuantConfig,
    compile_mxfp8_backward_quad_quant,
    compile_mxfp8_quant,
    compile_mxfp8_transposed_quant,
)
from rtx.prequant_experiments import BenchmarkProtocol, ShapeSpec


def _has_sm120() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12


class TestMXFP8BwdConfiguration(unittest.TestCase):
    def test_default_is_measured_safe_seed_and_fused_family_is_exposed(self) -> None:
        problem = MXFP8Problem(512, 1536, 1536)
        self.assertIsNone(DEFAULT_MXFP8_BWD_CONFIG.rejection(problem))
        self.assertIsNone(
            DEFAULT_MXFP8_BWD_CONFIG.implementation_rejection(problem)
        )
        self.assertEqual(DEFAULT_MXFP8_BWD_CONFIG.dx.backend, "decomposed")
        self.assertEqual(DEFAULT_MXFP8_BWD_CONFIG.dw.backend, "decomposed")
        self.assertEqual(DEFAULT_MXFP8_BWD_CONFIG.dx.quant_launches, "dual")
        self.assertEqual(DEFAULT_MXFP8_BWD_CONFIG.dw.quant_launches, "dual")
        self.assertEqual(DEFAULT_MXFP8_BWD_CONFIG.quant_schedule, "quad")
        self.assertEqual(DEFAULT_MXFP8_BWD_CONFIG.stream_schedule, "dual_stream")
        self.assertEqual(DEFAULT_FUSED_MXFP8_BWD_CONFIG.dx.backend, "fused")
        self.assertEqual(DEFAULT_FUSED_MXFP8_BWD_CONFIG.dw.backend, "fused")

    def test_partial_runner_builds_only_the_requested_gradient(self) -> None:
        problem = MXFP8Problem(512, 1536, 1536)
        sentinel = object()
        with patch.object(
            fp8_bwd, "_build_matmul_runner", return_value=sentinel
        ) as build:
            dx = fp8_bwd._build_partial_bwd_runner(
                problem,
                DEFAULT_MXFP8_BWD_CONFIG,
                torch.device("cpu"),
                "dx",
            )
            self.assertIs(dx, sentinel)
            kwargs = build.call_args.kwargs
            self.assertEqual(kwargs["problem"], MXFP8Problem(512, 1536, 1536))
            self.assertIs(kwargs["config"], DEFAULT_MXFP8_BWD_CONFIG.dx)
            self.assertTrue(kwargs["compile_quantizer"])
            build.reset_mock()
            dw = fp8_bwd._build_partial_bwd_runner(
                problem,
                DEFAULT_MXFP8_BWD_CONFIG,
                torch.device("cpu"),
                "dw",
            )
            self.assertIs(dw, sentinel)
            kwargs = build.call_args.kwargs
            self.assertEqual(kwargs["problem"], MXFP8Problem(1536, 1536, 512))
            self.assertIs(kwargs["config"], DEFAULT_MXFP8_BWD_CONFIG.dw)
            self.assertTrue(kwargs["compile_quantizer"])
        with self.assertRaisesRegex(ValueError, "unknown backward gradient"):
            fp8_bwd._build_partial_bwd_runner(
                problem,
                DEFAULT_MXFP8_BWD_CONFIG,
                torch.device("cpu"),
                "other",
            )

    def test_partial_launch_never_resolves_the_full_backward_runner(self) -> None:
        grad = torch.empty(8, 16, dtype=torch.bfloat16)
        x = torch.empty(8, 32, dtype=torch.bfloat16)
        weight = torch.empty(16, 32, dtype=torch.bfloat16)
        partial = Mock()
        with (
            patch.object(
                fp8_bwd,
                "_resolve_partial_bwd_runner",
                return_value=(partial, grad, x, weight),
            ) as resolve_partial,
            patch.object(
                fp8_bwd,
                "_resolve_bwd_runner",
                side_effect=AssertionError("full runner must not be built"),
            ),
        ):
            dx = fp8_bwd._launch_dx(grad, x, weight, "config")
            self.assertEqual(dx.shape, x.shape)
            resolve_partial.assert_called_once_with(
                grad, x, weight, "config", "dx"
            )
            partial.assert_called_once()
            resolve_partial.reset_mock()
            partial.reset_mock()
            dw = fp8_bwd._launch_dw(grad, x, weight, "config")
            self.assertEqual(dw.shape, weight.shape)
            resolve_partial.assert_called_once_with(
                grad, x, weight, "config", "dw"
            )
            partial.assert_called_once()

    def test_forward_scales_are_not_admitted_as_backward_scales(self) -> None:
        problem = MXFP8Problem(510, 1536, 1536)
        reason = DEFAULT_MXFP8_BWD_CONFIG.rejection(problem)
        self.assertIsNotNone(reason)
        self.assertIn("become MXFP8 reduction axes", reason)

    def test_serialization_is_stable(self) -> None:
        restored = bwd_config_from_dict(
            bwd_config_to_dict(DEFAULT_MXFP8_BWD_CONFIG)
        )
        self.assertEqual(restored, DEFAULT_MXFP8_BWD_CONFIG)
        self.assertEqual(
            bwd_config_id(restored),
            bwd_config_id(DEFAULT_MXFP8_BWD_CONFIG),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "winner.json"
            path.write_text(
                json.dumps({"config": bwd_config_to_dict(restored)}),
                encoding="utf-8",
            )
            self.assertEqual(load_mxfp8_bwd_config(path), restored)

    def test_every_search_coordinate_changes_the_config(self) -> None:
        for name, variants in BWD_SEARCH_SPACE.items():
            # Some forward coordinates are intentionally conditional (for
            # example quantizer_warps becomes active after entering the
            # three-role schedule). They must have a reachable changing value,
            # not necessarily change the baseline in isolation.
            bases = [
                DEFAULT_MXFP8_BWD_CONFIG,
                update_bwd_config(
                    DEFAULT_MXFP8_BWD_CONFIG,
                    {
                        "dx": {
                            "fused": {
                                "bf16_pipeline": (
                                    "tma", "three_role", 32, "none", 2
                                )
                            }
                        },
                        "dw": {
                            "fused": {
                                "bf16_pipeline": (
                                    "tma", "three_role", 32, "none", 2
                                )
                            }
                        },
                    },
                ),
            ]
            self.assertTrue(
                any(
                    update_bwd_config(base, variant) != base
                    for base in bases
                    for variant in variants
                ),
                name,
            )

    def test_cluster_reduction_is_an_executable_search_family(self) -> None:
        candidate = update_bwd_config(
            DEFAULT_DUAL_DECOMPOSED_MXFP8_BWD_CONFIG,
            {
                "dw": {
                    "reduction": "cluster_fp32",
                    "split_reduction": 4,
                    "reduction_tile": 128,
                    "gemm": {
                        "epilogue": "direct",
                        "epilogue_stages": 1,
                        "store_vec": 1,
                    },
                }
            },
        )
        reason = candidate.implementation_rejection(
            MXFP8Problem(512, 1536, 1536)
        )
        self.assertIsNone(reason)
        fused_update = next(
            value
            for value in BWD_SEARCH_SPACE["dw_reduction"]
            if value["dw"].get("backend") == "fused"
            and value["dw"]["reduction"] == "cluster_fp32"
            and value["dw"]["split_reduction"] == 4
            and value["dw"]["reduction_tile"] == 128
        )
        fused = update_bwd_config(DEFAULT_FUSED_MXFP8_BWD_CONFIG, fused_update)
        self.assertIsNone(
            fused.implementation_rejection(MXFP8Problem(512, 1536, 1536))
        )
        overfull = update_bwd_config(
            fused,
            {
                "dw": {
                    "fused": {
                        "load_engine": "tma",
                        "schedule": "three_role",
                        "bf16_tile_k": 32,
                        "bf16_swizzle": "none",
                        "bf16_stages": 2,
                        "mxfp8_stages": 2,
                        "quantizer_warps": 4,
                        "quant_load_bits": 128,
                    }
                }
            },
        )
        self.assertIn(
            "cluster reduction exceeds SM120 shared-memory capacity",
            overfull.implementation_rejection(MXFP8Problem(512, 1536, 1536)),
        )

    def test_decomposed_workspace_and_atomic_reductions_are_executable(self) -> None:
        problem = MXFP8Problem(512, 1536, 1536)
        for reduction in (
            "split_fp32_workspace",
            "split_fp32_atomic",
            "cluster_fp32",
        ):
            update = next(
                value
                for value in BWD_SEARCH_SPACE["dw_reduction"]
                if value["dw"]["reduction"] == reduction
                and value["dw"]["split_reduction"] == 4
                and value["dw"]["reduction_tile"] == 128
            )
            candidate = update_bwd_config(DEFAULT_MXFP8_BWD_CONFIG, update)
            with self.subTest(reduction=reduction):
                self.assertIsNone(candidate.implementation_rejection(problem))

    def test_no_physical_transpose_axes_exist(self) -> None:
        self.assertFalse(any("transpose" in name for name in BWD_SEARCH_SPACE))

    def test_real_prequant_gemm_persistence_is_exposed_for_dx_and_dw(self) -> None:
        for prefix in ("dx", "dw"):
            axis = BWD_SEARCH_SPACE[f"{prefix}_gemm_persistence"]
            self.assertTrue(
                any(
                    update[prefix]["gemm"]
                    == {"tiles_per_cta": 4, "tile_locality": "same_a"}
                    for update in axis
                )
            )

    def test_b_only_coordinate_crosses_to_independent_quantizers(self) -> None:
        tuner = object.__new__(BwdCoordinateDescentTuner)
        candidate = tuner._candidate(
            DEFAULT_SEPARATE_DECOMPOSED_MXFP8_BWD_CONFIG,
            "dx_b_vector",
            {"dx": {"quant_b": {"quant_vec": 2, "load_bits": 32}}},
        )
        self.assertEqual(candidate.dx.quant_launches, "separate")
        self.assertEqual(candidate.dx.quant_a.quant_vec, 4)
        self.assertEqual(candidate.dx.resolved_quant_b().quant_vec, 2)

    def test_mixed_dual_coordinate_preserves_specialized_schedules(self) -> None:
        candidate = update_bwd_config(
            DEFAULT_SEPARATE_DECOMPOSED_MXFP8_BWD_CONFIG,
            {"dx": {"quant_launches": "dual"}},
        )
        self.assertEqual(candidate.dx.quant_launches, "dual")
        self.assertNotEqual(candidate.dx.quant_a, candidate.dx.resolved_quant_b())
        self.assertIsNone(
            candidate.implementation_rejection(MXFP8Problem(512, 1536, 1536))
        )

    def test_mixed_dual_b_coordinate_remains_fused(self) -> None:
        tuner = object.__new__(BwdCoordinateDescentTuner)
        dual = update_bwd_config(
            DEFAULT_SEPARATE_DECOMPOSED_MXFP8_BWD_CONFIG,
            {"dx": {"quant_launches": "dual"}},
        )
        candidate = tuner._candidate(
            dual,
            "dx_b_registers",
            {"dx": {"quant_b": {"maxrregcount": 160}}},
        )
        self.assertEqual(candidate.dx.quant_launches, "dual")
        self.assertEqual(candidate.dx.resolved_quant_b().maxrregcount, 160)

    def test_quad_native_scale_coordinate_crosses_wide_k_legality_atomically(self) -> None:
        packed = next(
            value
            for value in BWD_SEARCH_SPACE["quad_native_scale_store"]
            if value["dx"]["quant_b"]["native_scale_store"] == "packed"
        )
        candidate = update_bwd_config(DEFAULT_MXFP8_BWD_CONFIG, packed)
        self.assertEqual(candidate.dx.resolved_quant_b().transposed_tile_k, 128)
        self.assertEqual(
            candidate.dw.quant_a.native_scale_store,
            "packed",
        )
        self.assertIsNone(
            candidate.implementation_rejection(MXFP8Problem(512, 1536, 1536))
        )

    def test_long_dw_reduction_requires_explicit_fp32_strategy(self) -> None:
        invalid = replace(
            DEFAULT_MXFP8_BWD_CONFIG,
            dw=replace(
                DEFAULT_MXFP8_BWD_CONFIG.dw,
                reduction="full_fp32",
                split_reduction=4,
            ),
        )
        reason = invalid.rejection(MXFP8Problem(512, 1536, 1536))
        self.assertIsNotNone(reason)
        self.assertIn("full reduction", reason)

    def test_fused_workspace_reduction_is_executable(self) -> None:
        candidate = update_bwd_config(
            DEFAULT_FUSED_MXFP8_BWD_CONFIG,
            {
                "dw": {
                    "reduction": "split_fp32_workspace",
                    "split_reduction": 4,
                    "reduction_tile": 128,
                    "workspace_epilogue": "tree",
                }
            },
        )
        self.assertIsNone(
            candidate.implementation_rejection(MXFP8Problem(512, 1536, 1536))
        )

    def test_fused_atomic_reduction_is_executable(self) -> None:
        candidate = update_bwd_config(
            DEFAULT_FUSED_MXFP8_BWD_CONFIG,
            {
                "dw": {
                    "reduction": "split_fp32_atomic",
                    "split_reduction": 4,
                    "reduction_tile": 128,
                    "workspace_epilogue": "none",
                }
            },
        )
        self.assertIsNone(
            candidate.implementation_rejection(MXFP8Problem(512, 1536, 1536))
        )

    def test_interleaved_is_a_real_decomposed_only_schedule(self) -> None:
        problem = MXFP8Problem(512, 1536, 1536)
        candidate = update_bwd_config(
            DEFAULT_DUAL_DECOMPOSED_MXFP8_BWD_CONFIG,
            {"execution_order": "interleaved"},
        )
        self.assertIsNone(candidate.implementation_rejection(problem))
        fused = update_bwd_config(
            DEFAULT_FUSED_MXFP8_BWD_CONFIG,
            {"execution_order": "interleaved"},
        )
        self.assertIn("requires two decomposed", fused.implementation_rejection(problem))
        dual_stream = update_bwd_config(
            candidate,
            {"stream_schedule": "dual_stream"},
        )
        self.assertIn("single-stream", dual_stream.implementation_rejection(problem))

    def test_fused_split_reductions_admit_persistent_locality_schedules(self) -> None:
        problem = MXFP8Problem(1024, 512, 512)
        for reduction, reuse in (
            ("split_fp32_workspace", "x"),
            ("split_fp32_atomic", "weight"),
        ):
            fused = normalize_fwd_config(
                DEFAULT_FUSED_MXFP8_BWD_CONFIG.dw.fused,
                persistent=True,
                persistent_waves=1,
                reuse=reuse,
            )
            candidate = update_bwd_config(
                DEFAULT_FUSED_MXFP8_BWD_CONFIG,
                {
                    "dw": {
                        "fused": asdict(fused),
                        "reduction": reduction,
                        "split_reduction": 8,
                        "reduction_tile": 128,
                        "workspace_epilogue": (
                            "tree" if reduction == "split_fp32_workspace" else "none"
                        ),
                    }
                },
            )
            with self.subTest(reduction=reduction, reuse=reuse):
                self.assertIsNone(candidate.implementation_rejection(problem))
        clustered = update_bwd_config(
            candidate,
            {
                "dw": {
                    "reduction": "cluster_fp32",
                    "workspace_epilogue": "none",
                }
            },
        )
        self.assertIn(
            "fixed CTA topology", clustered.implementation_rejection(problem)
        )

    def test_quad_quantization_is_a_real_shared_decomposed_schedule(self) -> None:
        problem = MXFP8Problem(512, 1536, 1536)
        candidate = update_bwd_config(
            DEFAULT_DECOMPOSED_MXFP8_BWD_CONFIG,
            {"quant_schedule": "quad"},
        )
        self.assertIsNone(candidate.implementation_rejection(problem))
        fused = update_bwd_config(
            DEFAULT_FUSED_MXFP8_BWD_CONFIG,
            {"quant_schedule": "quad"},
        )
        self.assertIn("decomposed", fused.implementation_rejection(problem))

    def test_shared_g_quad_is_a_reachable_square_tile_family(self) -> None:
        problem = MXFP8Problem(512, 1536, 1536)
        candidates = [
            update_bwd_config(DEFAULT_MXFP8_BWD_CONFIG, value)
            for value in BWD_SEARCH_SPACE["shared_g_quad"]
        ]
        self.assertTrue(candidates)
        self.assertTrue(
            all(candidate.quant_schedule == "shared_g_quad" for candidate in candidates)
        )
        self.assertTrue(
            all(candidate.implementation_rejection(problem) is None for candidate in candidates)
        )
        for candidate in candidates:
            shared = candidate.dx.quant_a
            self.assertEqual(shared, candidate.dx.resolved_quant_b())
            self.assertEqual(shared, candidate.dw.quant_a)
            self.assertEqual(shared, candidate.dw.resolved_quant_b())
            self.assertEqual(
                shared.transposed_tile_rows, shared.transposed_tile_k
            )

    def test_oriented_fused_kernels_compile_without_materialized_transposes(self) -> None:
        problem = MXFP8Problem(128, 128, 128)
        compile_mxfp8_fwd(
            problem,
            DEFAULT_FUSED_MXFP8_BWD_CONFIG.dx.fused,
            a_orientation="row",
            b_orientation="transpose",
        )
        compile_mxfp8_fwd(
            problem,
            DEFAULT_FUSED_MXFP8_BWD_CONFIG.dw.fused,
            a_orientation="transpose",
            b_orientation="transpose",
        )
        compile_mxfp8_split_fwd(
            MXFP8Problem(128, 128, 512),
            DEFAULT_FUSED_MXFP8_BWD_CONFIG.dw.fused,
            a_orientation="transpose",
            b_orientation="transpose",
            split_reduction=4,
            reduction_tile=128,
        )
        compile_mxfp8_atomic_split_fwd(
            MXFP8Problem(128, 128, 512),
            DEFAULT_FUSED_MXFP8_BWD_CONFIG.dw.fused,
            a_orientation="transpose",
            b_orientation="transpose",
            split_reduction=4,
            reduction_tile=128,
        )
        compile_mxfp8_workspace_reduce(
            128,
            128,
            1,
            algorithm="serial",
            threads=256,
            vector=4,
            persistent_waves=1,
        )


@unittest.skipUnless(_has_sm120(), "requires an SM120/SM121 CUDA GPU")
class TestMXFP8BwdCuda(unittest.TestCase):
    @staticmethod
    def _assert_backward_close(
        config, grad_output: torch.Tensor, x: torch.Tensor, weight: torch.Tensor
    ) -> None:
        grad_x, grad_weight = mxfp8_linear_backward(
            grad_output, x, weight, config=config, autotune="off"
        )
        torch.cuda.synchronize()
        expected_x = grad_output.float() @ weight.float()
        expected_weight = grad_output.float().T @ x.float()
        relative_x = (grad_x.float() - expected_x).norm() / expected_x.norm()
        relative_weight = (
            (grad_weight.float() - expected_weight).norm()
            / expected_weight.norm()
        )
        if float(relative_x) >= 0.07 or float(relative_weight) >= 0.07:
            raise AssertionError(
                f"backward error exceeds 7%: dX={float(relative_x)} "
                f"dW={float(relative_weight)} config={config}"
            )

    def test_cute_logical_transpose_quant_matches_contiguous_reference(self) -> None:
        rows, k = 128, 256
        source = torch.randn(k, rows, device="cuda", dtype=torch.bfloat16)
        logical = source.T
        self.assertEqual(logical.stride(), (1, rows))
        self.assertEqual(logical.data_ptr(), source.data_ptr())
        self.assertEqual(
            logical.untyped_storage().data_ptr(),
            source.untyped_storage().data_ptr(),
        )
        config = MXFP8QuantConfig(
            quant_vec=4,
            load_bits=64,
            quant_math="bf16x2",
            quant_amax="bf16_bits",
            scale_layout="row_major",
        )
        quantized = torch.empty(
            rows, k, device="cuda", dtype=torch.float8_e4m3fn
        )
        scales = torch.empty(
            rows, k // 32, device="cuda", dtype=torch.float8_e8m0fnu
        )
        reference_q = torch.empty_like(quantized)
        reference_s = torch.empty_like(scales)
        compile_mxfp8_transposed_quant(rows, k, config)(
            logical, quantized, scales
        )
        compile_mxfp8_quant(rows, k, config)(
            logical.contiguous(), reference_q, reference_s
        )
        torch.cuda.synchronize()
        self.assertTrue(
            torch.equal(quantized.view(torch.uint8), reference_q.view(torch.uint8))
        )
        self.assertTrue(torch.equal(scales.view(torch.uint8), reference_s.view(torch.uint8)))

    def test_cpasync_logical_transpose_and_packed_stores_match_reference(self) -> None:
        rows, k = 256, 256
        source = torch.randn(k, rows, device="cuda", dtype=torch.bfloat16)
        logical = source.T
        baseline = MXFP8QuantConfig(
            quant_vec=4,
            load_bits=64,
            quant_store_bits=8,
            quant_math="bf16x2",
            quant_amax="bf16_bits",
            scale_layout="row_major",
            transposed_load_engine="register",
        )
        expected_q = torch.empty(
            rows, k, device="cuda", dtype=torch.float8_e4m3fn
        )
        expected_s = torch.empty(
            rows, k // 32, device="cuda", dtype=torch.float8_e8m0fnu
        )
        compile_mxfp8_transposed_quant(rows, k, baseline)(
            logical, expected_q, expected_s
        )
        for store_bits in (8, 16, 32):
            config = replace(
                baseline,
                quant_store_bits=store_bits,
                transposed_load_engine="cp_async",
                transposed_smem_padding=0,
            )
            actual_q = torch.empty_like(expected_q)
            actual_s = torch.empty_like(expected_s)
            compile_mxfp8_transposed_quant(rows, k, config)(
                logical, actual_q, actual_s
            )
            torch.cuda.synchronize()
            with self.subTest(store_bits=store_bits):
                self.assertTrue(
                    torch.equal(
                        actual_q.view(torch.uint8), expected_q.view(torch.uint8)
                    )
                )
                self.assertTrue(
                    torch.equal(
                        actual_s.view(torch.uint8), expected_s.view(torch.uint8)
                    )
                )

    def test_wide_k_transpose_tile_packs_native_scales(self) -> None:
        rows, k = 256, 256
        source = torch.randn(k, rows, device="cuda", dtype=torch.bfloat16)
        logical = source.T
        scalar = MXFP8QuantConfig(
            quant_vec=4,
            load_bits=32,
            quant_store_bits=32,
            scale_layout="mma128",
            native_scale_store="scalar",
            transposed_tile_rows=128,
            transposed_tile_k=128,
        )
        expected_q = torch.empty(
            rows, k, device="cuda", dtype=torch.float8_e4m3fn
        )
        expected_s = torch.empty(
            rows // 128,
            k // 128,
            512,
            device="cuda",
            dtype=torch.float8_e8m0fnu,
        )
        compile_mxfp8_transposed_quant(rows, k, scalar)(
            logical, expected_q, expected_s
        )
        row = torch.arange(rows, device="cuda")[:, None]
        block = torch.arange(k // 32, device="cuda")[None, :]
        physical = (
            (row % 32) * 16 + ((row // 32) % 4) * 4 + block % 4
        )
        expected_used = expected_s[row // 128, block // 4, physical]
        for engine in ("register", "cp_async"):
            packed = replace(
                scalar,
                native_scale_store="packed",
                transposed_load_engine=engine,
                transposed_smem_padding=0 if engine == "cp_async" else 1,
            )
            actual_q = torch.empty_like(expected_q)
            actual_s = torch.empty_like(expected_s)
            compile_mxfp8_transposed_quant(rows, k, packed)(
                logical, actual_q, actual_s
            )
            torch.cuda.synchronize()
            actual_used = actual_s[row // 128, block // 4, physical]
            with self.subTest(engine=engine):
                self.assertTrue(
                    torch.equal(
                        actual_q.view(torch.uint8), expected_q.view(torch.uint8)
                    )
                )
                self.assertTrue(
                    torch.equal(
                        actual_used.view(torch.uint8),
                        expected_used.view(torch.uint8),
                    )
                )

    def test_dual_dw_quantization_uses_two_logical_views(self) -> None:
        torch.manual_seed(11)
        x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        grad_output = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        config = replace(
            DEFAULT_SEPARATE_DECOMPOSED_MXFP8_BWD_CONFIG,
            dw=replace(
                DEFAULT_SEPARATE_DECOMPOSED_MXFP8_BWD_CONFIG.dw,
                quant_launches="dual",
            ),
        )
        self.assertIsNone(
            config.implementation_rejection(MXFP8Problem(128, 128, 128))
        )
        _grad_x, grad_weight = mxfp8_linear_backward(
            grad_output, x, weight, config=config
        )
        torch.cuda.synchronize()
        expected_weight = grad_output.float().T @ x.float()
        relative_weight = (
            (grad_weight.float() - expected_weight).norm()
            / expected_weight.norm()
        )
        self.assertLess(float(relative_weight), 0.06)

    def test_dual_dx_quantization_mixes_row_and_logical_view(self) -> None:
        torch.manual_seed(13)
        x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        grad_output = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        config = update_bwd_config(
            DEFAULT_SEPARATE_DECOMPOSED_MXFP8_BWD_CONFIG,
            {"dx": {"quant_launches": "dual"}},
        )
        self.assertIsNone(
            config.implementation_rejection(MXFP8Problem(128, 128, 128))
        )
        grad_x, _grad_weight = mxfp8_linear_backward(
            grad_output, x, weight, config=config
        )
        torch.cuda.synchronize()
        expected_x = grad_output.float() @ weight.float()
        relative_x = (grad_x.float() - expected_x).norm() / expected_x.norm()
        self.assertLess(float(relative_x), 0.06)

    def test_backward_matches_fp32_gemm_with_mxfp8_error(self) -> None:
        torch.manual_seed(7)
        x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        grad_output = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        grad_x, grad_weight = mxfp8_linear_backward(grad_output, x, weight)
        torch.cuda.synchronize()
        expected_x = grad_output.float() @ weight.float()
        expected_weight = grad_output.float().T @ x.float()
        relative_x = (grad_x.float() - expected_x).norm() / expected_x.norm()
        relative_weight = (
            (grad_weight.float() - expected_weight).norm()
            / expected_weight.norm()
        )
        self.assertLess(float(relative_x), 0.06)
        self.assertLess(float(relative_weight), 0.06)

    def test_interleaved_decomposed_backward_matches_reference(self) -> None:
        torch.manual_seed(31)
        x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        grad_output = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        config = update_bwd_config(
            DEFAULT_DUAL_DECOMPOSED_MXFP8_BWD_CONFIG,
            {"execution_order": "interleaved"},
        )
        self._assert_backward_close(config, grad_output, x, weight)

    def test_quad_quantized_backward_matches_reference(self) -> None:
        torch.manual_seed(33)
        x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        grad_output = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        config = update_bwd_config(
            DEFAULT_DECOMPOSED_MXFP8_BWD_CONFIG,
            {"quant_schedule": "quad"},
        )
        self._assert_backward_close(config, grad_output, x, weight)

    def test_shared_g_quad_loads_one_tile_for_both_orientations(self) -> None:
        torch.manual_seed(133)
        m = n = k = 256
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        grad_output = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
        variants = [
            value
            for value in BWD_SEARCH_SPACE["shared_g_quad"]
            if value["dx"]["quant_a"]["transposed_tile_rows"] == 128
        ]
        for value in variants:
            config = update_bwd_config(DEFAULT_MXFP8_BWD_CONFIG, value)
            engine = config.dx.quant_a.transposed_load_engine
            scale_store = config.dx.quant_a.native_scale_store
            with self.subTest(engine=engine, scale_store=scale_store):
                self.assertIsNone(
                    config.implementation_rejection(MXFP8Problem(m, n, k))
                )
                self._assert_backward_close(config, grad_output, x, weight)

    def test_shared_g_quad_never_reads_the_duplicate_transpose_argument(self) -> None:
        torch.manual_seed(134)
        rows = 128
        value = next(
            candidate
            for candidate in BWD_SEARCH_SPACE["shared_g_quad"]
            if candidate["dx"]["quant_a"]["transposed_tile_rows"] == 128
            and candidate["dx"]["quant_a"]["transposed_load_engine"] == "cp_async"
            and candidate["dx"]["quant_a"]["native_scale_store"] == "packed"
        )
        config = update_bwd_config(DEFAULT_MXFP8_BWD_CONFIG, value)
        schedule = config.dx.quant_a
        grad_output = torch.randn(
            rows, rows, device="cuda", dtype=torch.bfloat16
        )
        unrelated = torch.zeros_like(grad_output).T
        weight_t = torch.randn_like(grad_output).T
        x_t = torch.randn_like(grad_output).T

        def q_buffer() -> torch.Tensor:
            return torch.empty(
                rows, rows, device="cuda", dtype=torch.float8_e4m3fn
            )

        def s_buffer() -> torch.Tensor:
            return torch.zeros(
                1, 1, 512, device="cuda", dtype=torch.float8_e8m0fnu
            )

        qa, qb, qc, qd = (q_buffer() for _ in range(4))
        sa, sb, sc, sd = (s_buffer() for _ in range(4))
        compile_mxfp8_backward_quad_quant(
            rows,
            rows,
            rows,
            rows,
            rows,
            rows,
            schedule,
            schedule,
            shared_g=True,
        )(
            grad_output,
            weight_t,
            unrelated,
            x_t,
            qa,
            qb,
            qc,
            qd,
            sa,
            sb,
            sc,
            sd,
        )
        expected_qa, expected_qc = q_buffer(), q_buffer()
        expected_sa, expected_sc = s_buffer(), s_buffer()
        compile_mxfp8_quant(rows, rows, schedule)(
            grad_output, expected_qa, expected_sa
        )
        compile_mxfp8_transposed_quant(rows, rows, schedule)(
            grad_output.T, expected_qc, expected_sc
        )
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(qa.view(torch.uint8), expected_qa.view(torch.uint8)))
        self.assertTrue(torch.equal(qc.view(torch.uint8), expected_qc.view(torch.uint8)))
        self.assertTrue(torch.equal(sa.view(torch.uint8), expected_sa.view(torch.uint8)))
        self.assertTrue(torch.equal(sc.view(torch.uint8), expected_sc.view(torch.uint8)))

    def test_persistent_prequantized_backward_matches_reference(self) -> None:
        torch.manual_seed(34)
        m = n = k = 256
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        grad_output = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
        config = update_bwd_config(
            DEFAULT_DECOMPOSED_MXFP8_BWD_CONFIG,
            {
                "dx": {
                    "gemm": {
                        "tiles_per_cta": 2,
                        "tile_locality": "same_a",
                    }
                },
                "dw": {
                    "gemm": {
                        "tiles_per_cta": 2,
                        "tile_locality": "same_b",
                    }
                },
            },
        )
        self.assertIsNone(config.implementation_rejection(MXFP8Problem(m, n, k)))
        self._assert_backward_close(config, grad_output, x, weight)

    def test_decomposed_split_dw_reductions_match_reference(self) -> None:
        torch.manual_seed(36)
        m, n, k = 512, 256, 256
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        grad_output = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
        problem = MXFP8Problem(m, n, k)
        for reduction in ("split_fp32_workspace", "split_fp32_atomic"):
            update = next(
                value
                for value in BWD_SEARCH_SPACE["dw_reduction"]
                if value["dw"]["reduction"] == reduction
                and value["dw"]["split_reduction"] == 4
                and value["dw"]["reduction_tile"] == 128
            )
            config = update_bwd_config(DEFAULT_MXFP8_BWD_CONFIG, update)
            with self.subTest(reduction=reduction):
                self.assertIsNone(config.implementation_rejection(problem))
                self._assert_backward_close(config, grad_output, x, weight)

    def test_quad_cpasync_logical_transport_feeds_both_matmuls(self) -> None:
        torch.manual_seed(35)
        m = n = k = 256
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        grad_output = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
        transport = {
            "transposed_load_engine": "cp_async",
            "transposed_smem_padding": 0,
            "transposed_tile_rows": 128,
            "transposed_tile_k": 128,
            "native_scale_store": "packed",
            "quant_store_bits": 32,
        }
        config = update_bwd_config(
            DEFAULT_DECOMPOSED_MXFP8_BWD_CONFIG,
            {
                "dx": {"quant_b": transport},
                "dw": {"quant_a": transport, "quant_b": transport},
            },
        )
        self.assertIsNone(config.implementation_rejection(MXFP8Problem(m, n, k)))
        self._assert_backward_close(config, grad_output, x, weight)

    def test_wide_dw_cta_reuses_quantized_a_without_global_round_trip(self) -> None:
        torch.manual_seed(37)
        m, n, k = 512, 256, 256
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        grad_output = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
        base = normalize_fwd_config(
            cta_reuse_tile=(128, 128, 2, 2, 232, 255)
        )
        wide_dw = normalize_fwd_config(
            cta_reuse_tile=(128, 256, 2, 1, 96, 160)
        )
        config = update_bwd_config(
            DEFAULT_FUSED_MXFP8_BWD_CONFIG,
            {
                "dx": {"fused": asdict(base)},
                "dw": {"fused": asdict(wide_dw)},
                "stream_schedule": "dual_stream",
            },
        )
        self.assertIsNone(config.implementation_rejection(MXFP8Problem(m, n, k)))
        self._assert_backward_close(config, grad_output, x, weight)

    def test_clustered_fused_backward_shares_native_tiles(self) -> None:
        torch.manual_seed(41)
        m = n = k = 256
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        grad_output = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
        transports = {
            "tma": normalize_fwd_config(cluster_reuse_tile=("a", 2)),
            "cpasync_ldmatrix": normalize_fwd_config(
                cpasync_cluster_reuse_tile=(
                    "a",
                    2,
                    4,
                    "bf16x2",
                    "bf16_bits",
                )
            ),
        }
        for name, clustered in transports.items():
            config = update_bwd_config(
                DEFAULT_FUSED_MXFP8_BWD_CONFIG,
                {
                    "dx": {"fused": asdict(clustered)},
                    "dw": {"fused": asdict(clustered)},
                    "stream_schedule": "dual_stream",
                },
            )
            with self.subTest(transport=name):
                self.assertIsNone(
                    config.implementation_rejection(MXFP8Problem(m, n, k))
                )
                self._assert_backward_close(config, grad_output, x, weight)

    def test_fused_tma_and_split_reduction_runtime_matrix(self) -> None:
        torch.manual_seed(29)
        m, n, k = 512, 128, 128
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        grad_output = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
        tma = normalize_fwd_config(
            load_engine="tma",
            schedule="three_role",
            bf16_tile_k=32,
            bf16_swizzle="none",
            bf16_stages=2,
            mxfp8_stages=2,
            quantizer_warps=4,
            quant_vec=8,
            quant_math="bf16x2",
            quant_amax="bf16_bits",
            quant_load_bits=128,
        )
        fused_tma = update_bwd_config(
            DEFAULT_FUSED_MXFP8_BWD_CONFIG,
            {
                "dx": {"fused": asdict(tma)},
                "dw": {"fused": asdict(tma)},
            },
        )
        tma_m64 = normalize_fwd_config(tma, tile_m=64)
        tma_cluster = normalize_fwd_config(tma, mxfp8_stages=1)
        variants = {
            "tma_full": fused_tma,
            "tma_m64": update_bwd_config(
                DEFAULT_FUSED_MXFP8_BWD_CONFIG,
                {
                    "dx": {"fused": asdict(tma_m64)},
                    "dw": {"fused": asdict(tma_m64)},
                },
            ),
            "tma_workspace": update_bwd_config(
                fused_tma,
                {
                    "dw": {
                        "reduction": "split_fp32_workspace",
                        "split_reduction": 4,
                        "reduction_tile": 128,
                        "workspace_epilogue": "tree",
                    }
                },
            ),
            "tma_atomic": update_bwd_config(
                fused_tma,
                {
                    "dw": {
                        "reduction": "split_fp32_atomic",
                        "split_reduction": 4,
                        "reduction_tile": 128,
                        "workspace_epilogue": "none",
                    }
                },
            ),
            "tma_cluster": update_bwd_config(
                fused_tma,
                {
                    "dw": {
                        "fused": asdict(tma_cluster),
                        "reduction": "cluster_fp32",
                        "split_reduction": 4,
                        "reduction_tile": 128,
                        "workspace_epilogue": "none",
                    }
                },
            ),
            "tma_dual_stream": update_bwd_config(
                fused_tma, {"stream_schedule": "dual_stream"}
            ),
        }
        for name, config in variants.items():
            with self.subTest(name=name):
                self.assertIsNone(
                    config.implementation_rejection(MXFP8Problem(m, n, k))
                )
                for _ in range(8 if name == "tma_cluster" else 1):
                    self._assert_backward_close(
                        config, grad_output, x, weight
                    )

    def test_persistent_fused_split_dw_crosses_output_and_split_tiles(self) -> None:
        torch.manual_seed(129)
        # Seven full 128-wide slices plus a 32-wide final slice exercise
        # persistent pipeline phase accounting across unequal reductions.
        m, n, k = 928, 512, 512
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        grad_output = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
        tma = normalize_fwd_config(
            load_engine="tma",
            schedule="three_role",
            bf16_tile_k=32,
            bf16_swizzle="none",
            bf16_stages=2,
            mxfp8_stages=2,
            quantizer_warps=4,
            quant_vec=8,
            quant_math="bf16x2",
            quant_amax="bf16_bits",
            quant_load_bits=128,
            persistent=True,
            persistent_waves=1,
        )
        problem = MXFP8Problem(m, n, k)
        for reduction, reuse in (
            ("split_fp32_workspace", "x"),
            ("split_fp32_atomic", "weight"),
        ):
            persistent = normalize_fwd_config(tma, reuse=reuse)
            config = update_bwd_config(
                DEFAULT_FUSED_MXFP8_BWD_CONFIG,
                {
                    "dw": {
                        "fused": asdict(persistent),
                        "reduction": reduction,
                        "split_reduction": 8,
                        "reduction_tile": 128,
                        "workspace_epilogue": (
                            "tree" if reduction == "split_fp32_workspace" else "none"
                        ),
                    }
                },
            )
            with self.subTest(reduction=reduction, reuse=reuse):
                self.assertIsNone(config.implementation_rejection(problem))
                for _ in range(4):
                    self._assert_backward_close(config, grad_output, x, weight)
        scalar = normalize_fwd_config(
            DEFAULT_FUSED_MXFP8_BWD_CONFIG.dw.fused,
            persistent=True,
            persistent_waves=1,
            reuse="none",
        )
        cpasync = normalize_fwd_config(
            load_engine="cpasync",
            bf16_tile_k=32,
            bf16_stages=4,
            bf16_swizzle="64b",
            quant_vec=8,
            persistent=True,
            persistent_waves=1,
            reuse="x",
        )
        cpasync_ldmatrix = normalize_fwd_config(
            cpasync,
            bf16_swizzle="none",
            quant_load_bits=128,
        )
        for name, transport in (
            ("scalar", scalar),
            ("cpasync", cpasync),
            ("cpasync_ldmatrix", cpasync_ldmatrix),
        ):
            config = update_bwd_config(
                DEFAULT_FUSED_MXFP8_BWD_CONFIG,
                {
                    "dw": {
                        "fused": asdict(transport),
                        "reduction": "split_fp32_workspace",
                        "split_reduction": 8,
                        "reduction_tile": 128,
                        "workspace_epilogue": "tree",
                    }
                },
            )
            with self.subTest(transport=name):
                self.assertIsNone(config.implementation_rejection(problem))
                self._assert_backward_close(config, grad_output, x, weight)

    def test_fused_cpasync_uses_physical_basis_for_all_backward_orientations(self) -> None:
        torch.manual_seed(130)
        m = n = k = 256
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        grad_output = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
        cpasync_scalar_load = normalize_fwd_config(
            load_engine="cpasync",
            bf16_tile_k=32,
            bf16_stages=4,
            bf16_swizzle="64b",
            quant_vec=8,
        )
        cpasync_ldmatrix = normalize_fwd_config(
            cpasync_scalar_load,
            bf16_swizzle="none",
            quant_load_bits=128,
        )
        for name, transport in (
            ("scalar_smem_load", cpasync_scalar_load),
            ("ldmatrix_x4", cpasync_ldmatrix),
        ):
            config = update_bwd_config(
                DEFAULT_FUSED_MXFP8_BWD_CONFIG,
                {
                    "dx": {"fused": asdict(transport)},
                    "dw": {"fused": asdict(transport)},
                },
            )
            with self.subTest(transport=name):
                self.assertIsNone(
                    config.implementation_rejection(MXFP8Problem(m, n, k))
                )
                self._assert_backward_close(config, grad_output, x, weight)

    def test_calibrated_harness_measures_and_races_backward(self) -> None:
        protocol = BenchmarkProtocol(
            warmup_calls=1,
            samples=3,
            confirm_samples=3,
            race_rounds=3,
            target_batch_ms=1.0,
            max_calls_per_sample=128,
            correctness_rtol=0.07,
            correctness_atol=1.0,
            practical_threshold=0.0,
            bootstrap_resamples=100,
            telemetry=False,
        )
        harness = BwdBenchmarkHarness(
            ShapeSpec(128, 128, 128), protocol, seed=17
        )
        dual = update_bwd_config(
            DEFAULT_SEPARATE_DECOMPOSED_MXFP8_BWD_CONFIG,
            {
                "dx": {"quant_launches": "dual"},
                "dw": {"quant_launches": "dual"},
            },
        )
        measurement = harness.measure(
            dual, samples=3, seed=17, components=True
        )
        self.assertEqual(measurement["status"], "ok")
        self.assertEqual(len(measurement["timings_ms"]), 3)
        self.assertIn("dx_quant", measurement["components"])
        race = harness.race(
            DEFAULT_SEPARATE_DECOMPOSED_MXFP8_BWD_CONFIG, dual, seed=18
        )
        self.assertEqual(race["status"], "ok")
        self.assertEqual(len(race["incumbent_timings_ms"]), 3)


if __name__ == "__main__":
    unittest.main()
