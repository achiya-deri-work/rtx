from __future__ import annotations

import unittest

import torch

import rtx
from rtx.configs import MXFP8FullyPrequantConfig, MXFP8WeightPrequantConfig
from rtx.inference_experiments import (
    FullyPrequantBenchmarkHarness,
    WeightPrequantBenchmarkHarness,
)

from rtx.kernels.mxfp8_quant import (
    MXFP8QuantConfig,
    compile_mxfp8_dual_quant,
    compile_mxfp8_quant,
)
from rtx.kernels.mxfp8 import MXFP8Problem
from rtx.kernels.mxfp8_gemm import MXFP8GemmConfig, compile_mxfp8_gemm
from rtx.kernels.mxfp8_reduce import compile_mxfp8_workspace_reduce
from rtx.prequant_experiments import BenchmarkProtocol, ShapeSpec


def _reference(src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows, k = src.shape
    blocks = src.reshape(rows, k // 32, 32)
    amax = blocks.abs().amax(dim=-1, keepdim=True).float()
    exponent = ((amax.view(torch.int32) >> 23) & 0xFF) - 127
    scale_code = ((exponent - 8).clamp(-127, 128) + 127).to(torch.uint8)
    scale_code = torch.where(
        torch.isnan(amax), torch.full_like(scale_code, 255), scale_code
    )
    scale = (scale_code.int() << 23).view(torch.float32).clamp_min(2**-126)
    quantized = (blocks.float() / scale).clamp(-448, 448)
    return (
        quantized.to(torch.float8_e4m3fn).reshape(rows, k),
        scale_code.view(torch.float8_e8m0fnu).squeeze(-1),
    )


class MXFP8QuantConfigTests(unittest.TestCase):
    def test_legality_is_explicit(self) -> None:
        self.assertIsNone(MXFP8QuantConfig().rejection(128, 256))
        self.assertIn(
            "divisible by quant_vec",
            MXFP8QuantConfig(quant_vec=8, load_bits=128).rejection(128, 128),
        )
        self.assertIn(
            "load width exceeds",
            MXFP8QuantConfig(quant_vec=2, load_bits=128).rejection(128, 256),
        )
        self.assertIsNone(
            MXFP8QuantConfig(quant_vec=4, quant_store_bits=32).rejection(
                128, 256
            )
        )
        self.assertIn(
            "quantized store width exceeds",
            MXFP8QuantConfig(
                quant_vec=2, load_bits=32, quant_store_bits=32
            ).rejection(128, 256),
        )
        self.assertIsNone(
            MXFP8QuantConfig(
                transposed_load_engine="cp_async",
                transposed_smem_padding=0,
            ).rejection(128, 256)
        )
        self.assertIn(
            "copy alignment",
            MXFP8QuantConfig(
                transposed_load_engine="cp_async",
                transposed_smem_padding=1,
            ).rejection(128, 256),
        )
        self.assertIn(
            "tile_k=128",
            MXFP8QuantConfig(
                scale_layout="mma128",
                native_scale_store="packed",
            ).transposed_rejection(128, 256),
        )
        self.assertIsNone(
            MXFP8QuantConfig(
                scale_layout="mma128",
                native_scale_store="packed",
                transposed_tile_k=128,
            ).transposed_rejection(128, 256)
        )
        self.assertIsNone(
            MXFP8GemmConfig().rejection(MXFP8Problem(128, 128, 256))
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class MXFP8QuantCudaTests(unittest.TestCase):
    def _assert_native_scales(
        self,
        native: torch.Tensor,
        row_major: torch.Tensor,
        tile_rows: int,
    ) -> None:
        rows, scale_blocks = row_major.shape
        row = torch.arange(rows, device="cuda")[:, None]
        block = torch.arange(scale_blocks, device="cuda")[None, :]
        row_group = (row // 32) % (tile_rows // 32)
        physical = (row % 32) * 16 + row_group * 4 + block % 4
        unpacked = native[row // tile_rows, block // 4, physical]
        torch.testing.assert_close(unpacked, row_major, rtol=0, atol=0)

    def test_dual_quantizer_matches_two_independent_quantizers(self) -> None:
        if torch.cuda.get_device_capability()[0] != 12:
            self.skipTest("native kernel requires SM120/SM121")
        x_rows, weight_rows, k = 64, 192, 256
        torch.manual_seed(1703)
        x = torch.randn(x_rows, k, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(
            weight_rows, k, device="cuda", dtype=torch.bfloat16
        )
        expected_qx, expected_sx = _reference(x)
        expected_qw, expected_sw = _reference(weight)
        actual_qx = torch.empty_like(expected_qx)
        actual_qw = torch.empty_like(expected_qw)
        actual_sx = torch.empty_like(expected_sx)
        actual_sw = torch.empty_like(expected_sw)
        compile_mxfp8_dual_quant(x_rows, weight_rows, k)(
            x,
            weight,
            actual_qx,
            actual_qw,
            actual_sx,
            actual_sw,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(actual_qx, expected_qx, rtol=0, atol=0)
        torch.testing.assert_close(actual_qw, expected_qw, rtol=0, atol=0)
        torch.testing.assert_close(actual_sx, expected_sx, rtol=0, atol=0)
        torch.testing.assert_close(actual_sw, expected_sw, rtol=0, atol=0)

    def test_dynamic_quantizer_variants_match_reference(self) -> None:
        if torch.cuda.get_device_capability()[0] != 12:
            self.skipTest("native kernel requires SM120/SM121")
        rows, k = 128, 256
        torch.manual_seed(1701)
        src = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
        expected_q, expected_s = _reference(src)
        variants = [
            MXFP8QuantConfig(),
            MXFP8QuantConfig(load_bits=16),
            MXFP8QuantConfig(load_bits=32),
            MXFP8QuantConfig(load_bits=64),
            MXFP8QuantConfig(quant_store_bits=16),
            MXFP8QuantConfig(quant_store_bits=32),
            # Logical-transpose transport is intentionally inactive for the
            # ordinary row-major quantizer.
            MXFP8QuantConfig(
                transposed_load_engine="cp_async",
                transposed_smem_padding=0,
            ),
            MXFP8QuantConfig(quant_math="fp32"),
            MXFP8QuantConfig(quant_amax="fp32"),
            MXFP8QuantConfig(reduction="redux"),
            MXFP8QuantConfig(num_warps=4, persistent_waves=1),
            MXFP8QuantConfig(num_warps=16, persistent_waves=8),
        ]
        for config in variants:
            with self.subTest(config=config):
                actual_q = torch.empty_like(expected_q)
                actual_s = torch.empty_like(expected_s)
                compile_mxfp8_quant(rows, k, config)(src, actual_q, actual_s)
                torch.cuda.synchronize()
                torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)
                torch.testing.assert_close(actual_s, expected_s, rtol=0, atol=0)

    def test_prequantized_gemm_matches_dequantized_reference(self) -> None:
        if torch.cuda.get_device_capability()[0] != 12:
            self.skipTest("native kernel requires SM120/SM121")
        problem = MXFP8Problem(128, 128, 256)
        torch.manual_seed(1702)
        x = torch.randn(
            problem.m, problem.k, device="cuda", dtype=torch.bfloat16
        )
        weight = torch.randn(
            problem.n, problem.k, device="cuda", dtype=torch.bfloat16
        )
        qx, sx = _reference(x)
        qw, sw = _reference(weight)
        dx = qx.float() * sx.float().repeat_interleave(32, dim=-1)
        dw = qw.float() * sw.float().repeat_interleave(32, dim=-1)
        expected = (dx @ dw.T).bfloat16()
        for config in (
            MXFP8GemmConfig(),
            MXFP8GemmConfig(stages=1, epilogue="direct", store_vec=1),
            MXFP8GemmConfig(
                tile_m=64,
                atom_layout_m=2,
                stages=3,
                consumer_registers=232,
            ),
        ):
            with self.subTest(config=config):
                actual = torch.empty_like(expected)
                compile_mxfp8_gemm(problem, config)(
                    qx, qw, sx, sw, actual
                )
                torch.cuda.synchronize()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_persistent_prequantized_gemm_locality_orders(self) -> None:
        if torch.cuda.get_device_capability()[0] != 12:
            self.skipTest("native kernel requires SM120/SM121")
        # Ten logical tiles exercise both multi-output work and a partial final
        # CTA for the four/eight-tile schedules.
        problem = MXFP8Problem(640, 256, 256)
        torch.manual_seed(1706)
        x = torch.randn(
            problem.m, problem.k, device="cuda", dtype=torch.bfloat16
        )
        weight = torch.randn(
            problem.n, problem.k, device="cuda", dtype=torch.bfloat16
        )
        qx, sx = _reference(x)
        qw, sw = _reference(weight)
        expected = torch.empty(
            problem.m, problem.n, device="cuda", dtype=torch.bfloat16
        )
        compile_mxfp8_gemm(problem)(qx, qw, sx, sw, expected)
        variants = (
            MXFP8GemmConfig(tiles_per_cta=2, tile_locality="same_a"),
            MXFP8GemmConfig(tiles_per_cta=4, tile_locality="same_b"),
            MXFP8GemmConfig(
                tiles_per_cta=4,
                tile_locality="serpentine_a",
                epilogue="direct",
                store_vec=1,
            ),
            MXFP8GemmConfig(
                tiles_per_cta=8,
                tile_locality="serpentine_b",
            ),
        )
        for config in variants:
            with self.subTest(config=config):
                actual = torch.full_like(expected, float("nan"))
                compile_mxfp8_gemm(problem, config)(qx, qw, sx, sw, actual)
                torch.cuda.synchronize()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_prequantized_split_k_workspace_and_atomic_outputs(self) -> None:
        if torch.cuda.get_device_capability()[0] != 12:
            self.skipTest("native kernel requires SM120/SM121")
        problem = MXFP8Problem(128, 128, 512)
        config = MXFP8GemmConfig(epilogue="direct", store_vec=1)
        torch.manual_seed(1707)
        x = torch.randn(
            problem.m, problem.k, device="cuda", dtype=torch.bfloat16
        )
        weight = torch.randn(
            problem.n, problem.k, device="cuda", dtype=torch.bfloat16
        )
        qx, sx = _reference(x)
        qw, sw = _reference(weight)
        expected = torch.empty(
            problem.m, problem.n, device="cuda", dtype=torch.bfloat16
        )
        compile_mxfp8_gemm(problem, config)(qx, qw, sx, sw, expected)
        reduce_four = compile_mxfp8_workspace_reduce(
            problem.m,
            problem.n,
            4,
            algorithm="tree",
            threads=256,
            vector=4,
            persistent_waves=1,
        )
        reduce_one = compile_mxfp8_workspace_reduce(
            problem.m,
            problem.n,
            1,
            algorithm="serial",
            threads=256,
            vector=4,
            persistent_waves=1,
        )
        workspace = torch.empty(
            4 * problem.m * problem.n,
            device="cuda",
            dtype=torch.float32,
        )
        workspace_out = torch.empty_like(expected)
        compile_mxfp8_gemm(
            problem,
            config,
            split_reduction=4,
            reduction_tile=128,
        )(qx, qw, sx, sw, workspace)
        reduce_four(workspace, workspace_out)
        accumulator = torch.zeros(
            problem.m * problem.n,
            device="cuda",
            dtype=torch.float32,
        )
        atomic_out = torch.empty_like(expected)
        compile_mxfp8_gemm(
            problem,
            config,
            split_reduction=4,
            reduction_tile=128,
            atomic_output=True,
        )(qx, qw, sx, sw, accumulator)
        reduce_one(accumulator, atomic_out)
        torch.cuda.synchronize()
        torch.testing.assert_close(workspace_out, expected, rtol=0, atol=1e-5)
        torch.testing.assert_close(atomic_out, expected, rtol=0, atol=1e-5)

    def test_native_scale_tma_paths_match_row_major(self) -> None:
        if torch.cuda.get_device_capability()[0] != 12:
            self.skipTest("native kernel requires SM120/SM121")
        problem = MXFP8Problem(128, 128, 256)
        torch.manual_seed(1704)
        x = torch.randn(
            problem.m, problem.k, device="cuda", dtype=torch.bfloat16
        )
        weight = torch.randn(
            problem.n, problem.k, device="cuda", dtype=torch.bfloat16
        )
        qx = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        qw = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
        sx = torch.empty(
            problem.m,
            problem.k // 32,
            device="cuda",
            dtype=torch.float8_e8m0fnu,
        )
        sw = torch.empty(
            problem.n,
            problem.k // 32,
            device="cuda",
            dtype=torch.float8_e8m0fnu,
        )
        compile_mxfp8_dual_quant(problem.m, problem.n, problem.k)(
            x, weight, qx, qw, sx, sw
        )
        row_out = torch.empty(
            problem.m, problem.n, device="cuda", dtype=torch.bfloat16
        )
        compile_mxfp8_gemm(
            problem,
            MXFP8GemmConfig(tile_m=64, atom_layout_m=2, stages=1),
        )(qx, qw, sx, sw, row_out)

        for tile_rows, scale_layout, gemm_layout, tile_m, atom_m in (
            (128, "mma128", "mma128", 128, 8),
            (64, "mma64", "mma64x128", 64, 2),
        ):
            with self.subTest(scale_layout=scale_layout):
                native_sx = torch.empty(
                    problem.m // tile_rows,
                    problem.k // 128,
                    512,
                    device="cuda",
                    dtype=torch.float8_e8m0fnu,
                )
                native_sw = torch.empty(
                    problem.n // 128,
                    problem.k // 128,
                    512,
                    device="cuda",
                    dtype=torch.float8_e8m0fnu,
                )
                compile_mxfp8_dual_quant(
                    problem.m,
                    problem.n,
                    problem.k,
                    MXFP8QuantConfig(scale_layout=scale_layout),
                    weight_scale_layout="mma128",
                )(x, weight, qx, qw, native_sx, native_sw)
                native_out = torch.empty_like(row_out)
                compile_mxfp8_gemm(
                    problem,
                    MXFP8GemmConfig(
                        tile_m=tile_m,
                        atom_layout_m=atom_m,
                        stages=1,
                        scale_role="tma",
                        scale_layout=gemm_layout,
                    ),
                )(qx, qw, native_sx, native_sw, native_out)
                torch.cuda.synchronize()
                self._assert_native_scales(native_sx, sx, tile_rows)
                self._assert_native_scales(native_sw, sw, 128)
                torch.testing.assert_close(native_out, row_out, rtol=0, atol=0)

    def test_three_public_operand_states_are_numerically_identical(self) -> None:
        if torch.cuda.get_device_capability()[0] != 12:
            self.skipTest("native kernel requires SM120/SM121")
        torch.manual_seed(1705)
        x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        config = rtx.DEFAULT_MXFP8_INFERENCE_CONFIG
        with torch.inference_mode():
            dynamic = rtx.mxfp8_linear(
                x,
                weight,
                backend="prequant",
                prequant_config=config,
            )
            packed_weight = rtx.quantize_mxfp8(
                weight, config=config.resolved_weight_quant()
            )
            aot_weight = rtx.mxfp8_linear(
                x, packed_weight, prequant_config=config
            )
            packed_x = rtx.quantize_mxfp8(x, config=config.quant)
            fully_packed = rtx.mxfp8_linear(
                packed_x, packed_weight, prequant_config=config
            )
            layer = rtx.MXFP8Linear(
                256,
                128,
                packed_weight=packed_weight,
                autotune="cache",
            )
            compiled = torch.compile(layer, fullgraph=True, dynamic=False)
            compiled_aot_weight = compiled(x)
            compiled_fully_packed = compiled(packed_x)
        torch.cuda.synchronize()
        torch.testing.assert_close(aot_weight, dynamic, rtol=0, atol=0)
        torch.testing.assert_close(fully_packed, dynamic, rtol=0, atol=0)
        torch.testing.assert_close(compiled_aot_weight, dynamic, rtol=0, atol=0)
        torch.testing.assert_close(compiled_fully_packed, dynamic, rtol=0, atol=0)

    def test_state_specific_harnesses_exclude_aot_packing_from_timing(self) -> None:
        if torch.cuda.get_device_capability()[0] != 12:
            self.skipTest("native kernel requires SM120/SM121")
        shape = ShapeSpec(128, 128, 256)
        protocol = BenchmarkProtocol(
            warmup_calls=1,
            samples=3,
            confirm_samples=3,
            race_rounds=3,
            target_batch_ms=1.0,
            max_calls_per_sample=16,
            bootstrap_resamples=100,
            telemetry=False,
        )
        cases = (
            (
                WeightPrequantBenchmarkHarness,
                MXFP8WeightPrequantConfig(),
                "x_quant",
            ),
            (
                FullyPrequantBenchmarkHarness,
                MXFP8FullyPrequantConfig(),
                None,
            ),
        )
        for harness_type, config, quant_component in cases:
            with self.subTest(harness=harness_type.__name__):
                harness = harness_type(
                    shape, "hot", protocol, device="cuda", seed=1706
                )
                result = harness.measure(
                    config, samples=3, seed=1707, components=True
                )
                self.assertEqual(result["status"], "ok", result.get("error"))
                components = result["components"]
                self.assertIn("gemm_hot_packed", components)
                self.assertEqual(
                    quant_component in components,
                    quant_component is not None,
                )
                self.assertNotIn("w_quant", components)
                self.assertNotIn("dual_quant", components)


if __name__ == "__main__":
    unittest.main()
