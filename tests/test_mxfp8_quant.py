from __future__ import annotations

import unittest

import torch

from rtx.kernels.mxfp8_quant import MXFP8QuantConfig, compile_mxfp8_quant
from rtx.kernels.mxfp8 import MXFP8Problem
from rtx.kernels.mxfp8_gemm import MXFP8GemmConfig, compile_mxfp8_gemm


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
            MXFP8GemmConfig().rejection(MXFP8Problem(128, 128, 256))
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class MXFP8QuantCudaTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
