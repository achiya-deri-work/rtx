from __future__ import annotations

import unittest

import torch

from rtx import MXFP8Linear, mxfp8_linear
from rtx.kernels.mxfp8 import MXFP8FwdConfig, MXFP8Problem, normalize_fwd_config
from rtx.kernels.mxfp8_fwd import compile_mxfp8_fwd


def _reference_to_mx(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent FLOOR-mode E4M3/E8M0 reference (scale-vector size 32)."""

    shape = x.shape
    blocks = x.reshape(*shape[:-1], shape[-1] // 32, 32)
    amax = blocks.abs().amax(dim=-1, keepdim=True).float()
    exponent = ((amax.view(torch.int32) >> 23) & 0xFF) - 127
    scale_code = ((exponent - 8).clamp(-127, 128) + 127).to(torch.uint8)
    scale_code = torch.where(
        torch.isnan(amax), torch.full_like(scale_code, 255), scale_code
    )
    scale = (scale_code.int() << 23).view(torch.float32).clamp_min(2**-126)
    values = (blocks.float() / scale).clamp(-448, 448)
    return (
        values.to(torch.float8_e4m3fn).reshape(shape),
        scale_code.view(torch.float8_e8m0fnu).squeeze(-1),
    )


def _reference_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    k = x.shape[-1]
    flat_x = x.reshape(-1, k)
    qx, sx = _reference_to_mx(flat_x)
    qw, sw = _reference_to_mx(weight)
    dx = qx.float() * sx.float().repeat_interleave(32, dim=-1)
    dw = qw.float() * sw.float().repeat_interleave(32, dim=-1)
    return (dx @ dw.T).bfloat16().reshape(*x.shape[:-1], weight.shape[0])


class MXFP8ConfigTests(unittest.TestCase):
    def test_default_config_is_explicitly_implemented(self) -> None:
        problem = MXFP8Problem(37, 70, 128)
        self.assertIsNone(MXFP8FwdConfig().implementation_rejection(problem))
        self.assertIsNone(
            normalize_fwd_config(
                tile_m=256,
                k_unroll=2,
                maxrregcount=192,
                raster="m",
                grid_swizzle=4,
            ).implementation_rejection(MXFP8Problem(512, 512, 512))
        )
        self.assertIsNone(
            MXFP8FwdConfig(load_engine="tma").implementation_rejection(problem)
        )
        self.assertIsNone(
            normalize_fwd_config(
                load_engine="tma", schedule="warp_specialized"
            ).implementation_rejection(problem)
        )
        self.assertIn(
            "requires full M/N/K tiles",
            normalize_fwd_config(load_engine="cpasync").implementation_rejection(
                problem
            ),
        )

    def test_k_must_be_scale_aligned(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible by 32"):
            MXFP8Problem(1, 1, 33).validate()

    def test_64_row_and_three_role_schedules_are_legal(self) -> None:
        problem = MXFP8Problem(512, 1536, 1536)
        tile64 = normalize_fwd_config(tile_m=64)
        self.assertEqual(tile64.num_mma_warps, 4)
        self.assertIsNone(tile64.implementation_rejection(problem))
        three_role = normalize_fwd_config(
            load_engine="tma",
            schedule="three_role",
            bf16_tile_k=32,
            bf16_stages=2,
            mxfp8_stages=2,
            quantizer_warps=4,
        )
        self.assertEqual(three_role.num_threads, 416)
        self.assertIsNone(three_role.implementation_rejection(problem))

    def test_compound_reuse_tile_enters_a_legal_wide_tma_basin(self) -> None:
        problem = MXFP8Problem(512, 1536, 1536)
        config = normalize_fwd_config(
            cta_reuse_tile=(128, 256, 2, 1, 96, 160)
        )
        self.assertEqual((config.tile_m, config.tile_n), (128, 256))
        self.assertEqual(config.mxfp8_stages, 1)
        self.assertEqual(config.quant_load_bits, 128)
        self.assertEqual(config.num_threads, 672)
        self.assertIsNone(config.implementation_rejection(problem))
        misaligned = normalize_fwd_config(config, quantizer_warps=2)
        self.assertIn(
            "complete warpgroups",
            misaligned.implementation_rejection(problem),
        )
        overprovisioned = normalize_fwd_config(config, quantizer_warps=8)
        self.assertIn(
            "four quantizer warps",
            overprovisioned.implementation_rejection(problem),
        )
        unsupported_n_atoms = normalize_fwd_config(
            config, tile_m=64, tile_n=384
        )
        self.assertIn(
            "1, 2, or 4 N atoms",
            unsupported_n_atoms.implementation_rejection(problem),
        )

    def test_tma_epilogue_rejects_ragged_output_tiles(self) -> None:
        config = normalize_fwd_config(
            tile_m=64, epilogue="tma", store_vec=4
        )
        self.assertIn(
            "full M/N output tiles",
            config.implementation_rejection(MXFP8Problem(65, 129, 128)),
        )

    def test_logical_transpose_exposes_proven_ldmatrix_vector_path(self) -> None:
        problem = MXFP8Problem(128, 128, 128)
        vector = normalize_fwd_config(
            load_engine="tma",
            schedule="three_role",
            bf16_tile_k=32,
            bf16_swizzle="none",
            quant_vec=8,
            quant_load_bits=128,
        )
        self.assertIsNone(
            vector.oriented_implementation_rejection(
                problem, "row", "transpose"
            )
        )
        unsupported = normalize_fwd_config(
            load_engine="tma",
            schedule="three_role",
            bf16_swizzle="none",
            quant_vec=4,
            quant_load_bits=64,
        )
        self.assertIn(
            "TMA/ldmatrix x4 path",
            unsupported.oriented_implementation_rejection(
                problem, "row", "transpose"
            ),
        )
        scalar = normalize_fwd_config(
            load_engine="tma", schedule="three_role", quant_load_bits=16
        )
        self.assertIsNone(
            scalar.oriented_implementation_rejection(
                problem, "transpose", "transpose"
            )
        )

    def test_baseline_compiles_without_a_visible_gpu(self) -> None:
        compiled = compile_mxfp8_fwd(
            MXFP8Problem(128, 128, 128), MXFP8FwdConfig()
        )
        self.assertIsNotNone(compiled)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class MXFP8CudaTests(unittest.TestCase):
    def test_prequant_frontend_is_fullgraph_compileable(self) -> None:
        if torch.cuda.get_device_capability()[0] != 12:
            self.skipTest("native kernel requires SM120/SM121")
        torch.manual_seed(122)
        x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        layer = MXFP8Linear(
            128, 128, device="cuda", dtype=torch.bfloat16, backend="prequant"
        ).eval().requires_grad_(False)
        expected = layer(x)
        compiled = torch.compile(layer, fullgraph=True, dynamic=False)
        actual = compiled(x)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_fused_mxfp8_matches_floor_reference(self) -> None:
        if torch.cuda.get_device_capability()[0] != 12:
            self.skipTest("native kernel requires SM120/SM121")
        cases = [
            ((128,), 128, 128),
            ((37,), 70, 128),
            ((2, 3), 77, 128),
            ((5,), 129, 256),
        ]
        for leading, n, k in cases:
            with self.subTest(leading=leading, n=n, k=k):
                torch.manual_seed(123)
                x = (torch.randn(*leading, k, device="cuda") * 0.5).bfloat16()
                weight = (torch.randn(n, k, device="cuda") * 0.5).bfloat16()
                actual = mxfp8_linear(x, weight)
                expected = _reference_linear(x, weight)
                torch.cuda.synchronize()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_real_schedule_variants_match_reference(self) -> None:
        if torch.cuda.get_device_capability()[0] != 12:
            self.skipTest("native kernel requires SM120/SM121")
        problem = MXFP8Problem(257, 385, 256)
        variants = [
            normalize_fwd_config(tile_m=64),
            normalize_fwd_config(tile_m=256),
            normalize_fwd_config(tile_n=256),
            normalize_fwd_config(tile_k=256, k_unroll=2),
            normalize_fwd_config(atom_layout_m=2),
            normalize_fwd_config(atom_layout_m=8),
            normalize_fwd_config(maxrregcount=192),
            normalize_fwd_config(quant_vec=2),
            normalize_fwd_config(quant_vec=4),
            normalize_fwd_config(quant_vec=8),
            normalize_fwd_config(quant_vec=8, quant_math="bf16x2"),
            normalize_fwd_config(quant_vec=8, quant_amax="bf16_bits"),
            normalize_fwd_config(quant_vec=8, reduction="redux"),
            normalize_fwd_config(mxfp8_stages=2),
            normalize_fwd_config(mxfp8_stages=3),
            normalize_fwd_config(load_engine="tma"),
            normalize_fwd_config(
                load_engine="tma",
                schedule="warp_specialized",
                quant_vec=4,
            ),
            normalize_fwd_config(
                load_engine="tma",
                schedule="warp_specialized",
                bf16_tile_k=32,
                bf16_stages=4,
                bf16_swizzle="64b",
                quant_vec=8,
            ),
            normalize_fwd_config(
                load_engine="tma",
                schedule="three_role",
                bf16_tile_k=32,
                bf16_stages=2,
                mxfp8_stages=2,
                quantizer_warps=4,
                quant_vec=8,
            ),
            normalize_fwd_config(a_swizzle="none"),
            normalize_fwd_config(a_swizzle="32b"),
            normalize_fwd_config(a_swizzle="64b"),
            normalize_fwd_config(b_swizzle="none"),
            normalize_fwd_config(b_swizzle="32b"),
            normalize_fwd_config(b_swizzle="64b"),
            normalize_fwd_config(a_ldmatrix_matrices=1),
            normalize_fwd_config(a_ldmatrix_matrices=2),
            normalize_fwd_config(b_ldmatrix_matrices=1),
            normalize_fwd_config(b_ldmatrix_matrices=2),
            normalize_fwd_config(raster="m"),
            normalize_fwd_config(grid_swizzle=2),
            normalize_fwd_config(raster="m", grid_swizzle=4),
            normalize_fwd_config(
                quant_vec=4,
                mxfp8_stages=2,
                a_swizzle="64b",
                b_swizzle="32b",
                raster="m",
                grid_swizzle=2,
            ),
            normalize_fwd_config(
                load_engine="tma",
                quant_vec=4,
                a_swizzle="64b",
                b_swizzle="32b",
                raster="m",
                grid_swizzle=2,
            ),
        ]
        torch.manual_seed(321)
        x = (torch.randn(problem.m, problem.k, device="cuda") * 0.5).bfloat16()
        weight = (
            torch.randn(problem.n, problem.k, device="cuda") * 0.5
        ).bfloat16()
        expected = _reference_linear(x, weight)
        for config in variants:
            with self.subTest(config=config):
                out = torch.empty_like(expected)
                compile_mxfp8_fwd(problem, config)(x, weight, out)
                torch.cuda.synchronize()
                torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_long_k_aggressive_schedule_matches_baseline(self) -> None:
        if torch.cuda.get_device_capability()[0] != 12:
            self.skipTest("native kernel requires SM120/SM121")
        problem = MXFP8Problem(129, 193, 1536)
        aggressive = normalize_fwd_config(
            atom_layout_m=8,
            quant_vec=8,
            k_unroll=2,
            maxrregcount=160,
            a_swizzle="none",
            b_swizzle="none",
            raster="m",
            grid_swizzle=2,
        )
        torch.manual_seed(991)
        x = (torch.randn(problem.m, problem.k, device="cuda") * 0.5).bfloat16()
        weight = (
            torch.randn(problem.n, problem.k, device="cuda") * 0.5
        ).bfloat16()
        baseline = torch.empty(
            (problem.m, problem.n), device="cuda", dtype=torch.bfloat16
        )
        actual = torch.empty_like(baseline)
        specialized = torch.empty_like(baseline)
        compile_mxfp8_fwd(problem, MXFP8FwdConfig())(x, weight, baseline)
        compile_mxfp8_fwd(problem, aggressive)(x, weight, actual)
        compile_mxfp8_fwd(
            problem,
            normalize_fwd_config(
                load_engine="tma",
                schedule="warp_specialized",
                atom_layout_m=8,
                quant_vec=8,
                k_unroll=2,
            ),
        )(x, weight, specialized)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
        torch.testing.assert_close(specialized, baseline, rtol=0, atol=0)

    def test_cpasync_full_tiles_match_baseline(self) -> None:
        if torch.cuda.get_device_capability()[0] != 12:
            self.skipTest("native kernel requires SM120/SM121")
        problem = MXFP8Problem(256, 256, 256)
        torch.manual_seed(733)
        x = torch.randn(
            problem.m, problem.k, device="cuda", dtype=torch.bfloat16
        )
        weight = torch.randn(
            problem.n, problem.k, device="cuda", dtype=torch.bfloat16
        )
        baseline = torch.empty(
            (problem.m, problem.n), device="cuda", dtype=torch.bfloat16
        )
        actual = torch.empty_like(baseline)
        compile_mxfp8_fwd(problem, MXFP8FwdConfig())(x, weight, baseline)
        compile_mxfp8_fwd(
            problem,
            normalize_fwd_config(
                load_engine="cpasync",
                bf16_tile_k=32,
                bf16_stages=4,
                bf16_swizzle="64b",
                quant_vec=8,
            ),
        )(x, weight, actual)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, baseline, rtol=0, atol=0)

    def test_vectorized_packed_quantizer_matches_baseline(self) -> None:
        if torch.cuda.get_device_capability()[0] != 12:
            self.skipTest("native kernel requires SM120/SM121")
        problem = MXFP8Problem(128, 128, 128)
        torch.manual_seed(811)
        x = torch.randn(
            problem.m, problem.k, device="cuda", dtype=torch.bfloat16
        )
        weight = torch.randn(
            problem.n, problem.k, device="cuda", dtype=torch.bfloat16
        )
        baseline = torch.empty(
            (problem.m, problem.n), device="cuda", dtype=torch.bfloat16
        )
        compile_mxfp8_fwd(problem, MXFP8FwdConfig())(x, weight, baseline)
        for tile_m in (64, 128):
            with self.subTest(tile_m=tile_m):
                actual = torch.empty_like(baseline)
                compile_mxfp8_fwd(
                    problem,
                    normalize_fwd_config(
                        tile_m=tile_m,
                        load_engine="tma",
                        bf16_tile_k=32,
                        bf16_swizzle="none",
                        quant_vec=8,
                        quant_math="bf16x2",
                        quant_amax="bf16_bits",
                        quant_load_bits=128,
                    ),
                )(x, weight, actual)
                torch.cuda.synchronize()
                torch.testing.assert_close(actual, baseline, rtol=0, atol=0)

    def test_tma_epilogue_full_tiles_match_direct_store(self) -> None:
        if torch.cuda.get_device_capability()[0] != 12:
            self.skipTest("native kernel requires SM120/SM121")
        problem = MXFP8Problem(128, 256, 128)
        torch.manual_seed(1229)
        x = torch.randn(
            problem.m, problem.k, device="cuda", dtype=torch.bfloat16
        )
        weight = torch.randn(
            problem.n, problem.k, device="cuda", dtype=torch.bfloat16
        )
        direct = torch.empty(
            (problem.m, problem.n), device="cuda", dtype=torch.bfloat16
        )
        base = normalize_fwd_config(tile_m=64, quant_vec=8)
        compile_mxfp8_fwd(problem, base)(x, weight, direct)
        for store_vec in (1, 2, 4):
            with self.subTest(store_vec=store_vec):
                actual = torch.empty_like(direct)
                compile_mxfp8_fwd(
                    problem,
                    normalize_fwd_config(
                        base,
                        epilogue="tma",
                        epilogue_stages=1,
                        store_vec=store_vec,
                    ),
                )(x, weight, actual)
                torch.cuda.synchronize()
                torch.testing.assert_close(actual, direct, rtol=0, atol=0)

    def test_persistent_grid_and_locality_orders_match_nonpersistent(self) -> None:
        if torch.cuda.get_device_capability()[0] != 12:
            self.skipTest("native kernel requires SM120/SM121")
        # 8 * 12 = 96 logical tiles on this 70-SM RTX: a one-wave persistent
        # launch necessarily makes some CTAs execute a second work tile.
        problem = MXFP8Problem(512, 1536, 128)
        torch.manual_seed(1881)
        x = torch.randn(
            problem.m, problem.k, device="cuda", dtype=torch.bfloat16
        )
        weight = torch.randn(
            problem.n, problem.k, device="cuda", dtype=torch.bfloat16
        )
        expected = torch.empty(
            (problem.m, problem.n), device="cuda", dtype=torch.bfloat16
        )
        base = normalize_fwd_config(tile_m=64, quant_vec=8)
        compile_mxfp8_fwd(problem, base)(x, weight, expected)
        for reuse in ("none", "x", "weight"):
            with self.subTest(reuse=reuse):
                actual = torch.empty_like(expected)
                config = normalize_fwd_config(
                    base,
                    persistent=True,
                    persistent_waves=1,
                    reuse=reuse,
                )
                compile_mxfp8_fwd(problem, config)(x, weight, actual)
                torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_wide_cta_reuse_tile_matches_reference(self) -> None:
        if torch.cuda.get_device_capability()[0] != 12:
            self.skipTest("native kernel requires SM120/SM121")
        problem = MXFP8Problem(256, 256, 256)
        config = normalize_fwd_config(
            cta_reuse_tile=(128, 256, 2, 1, 96, 160)
        )
        torch.manual_seed(127)
        x = torch.randn(problem.m, problem.k, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(
            problem.n, problem.k, device="cuda", dtype=torch.bfloat16
        )
        out = torch.empty(
            problem.m, problem.n, device="cuda", dtype=torch.bfloat16
        )
        compile_mxfp8_fwd(problem, config)(x, weight, out)
        torch.cuda.synchronize()
        expected = _reference_linear(x, weight)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
