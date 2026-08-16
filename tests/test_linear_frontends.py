from __future__ import annotations

import os
import subprocess
import sys
import unittest

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import rtx
from rtx.formats import make_mxfp8_tensor, make_nvfp4_tensor


class LinearFrontendContractTests(unittest.TestCase):
    def test_import_and_frontend_resolution_do_not_select_a_cute_arch(self) -> None:
        env = os.environ.copy()
        env.pop("CUTE_DSL_ARCH", None)
        env.pop("QUACK_ARCH", None)
        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import os, rtx; "
                    "_ = rtx.MXFP8Linear, rtx.NVFP4Linear; "
                    "import rtx.autotune, rtx.autotune.dataset; "
                    "assert 'CUTE_DSL_ARCH' not in os.environ; "
                    "assert 'QUACK_ARCH' not in os.environ"
                ),
            ],
            check=True,
            env=env,
        )

    def test_public_package_exposes_versioned_packed_operand_types(self) -> None:
        from torchao.prototype.mx_formats.mx_tensor import MXTensor
        from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

        self.assertIs(rtx.MXTensor, MXTensor)
        self.assertIs(rtx.MXFP8Tensor, MXTensor)
        self.assertIs(rtx.NVFP4Tensor, NVFP4Tensor)

    def test_runtime_workspace_caches_have_an_explicit_release_boundary(self) -> None:
        released = rtx.clear_runtime_caches(synchronize=False)
        self.assertIn("fp8", released)
        self.assertIn("fp8_bwd", released)
        self.assertIn("prequant", released["fp8"])
        self.assertIn("backward", released["fp8_bwd"])
        self.assertIn("backward_matmul", released["fp8_bwd"])

    def test_dynamic_module_defers_backward_cache_selection(self) -> None:
        cached = rtx.MXFP8Linear(128, 64, device="cpu", autotune="cache")
        self.assertTrue(cached._backward_config_key.startswith("bwd-autotune:"))
        fixed = rtx.MXFP8Linear(128, 64, device="cpu", autotune="off")
        self.assertFalse(fixed._backward_config_key.startswith("bwd-autotune:"))

    def test_external_torchao_mx_tensors_map_to_rtx_kernel_layouts(self) -> None:
        from torchao.prototype.mx_formats.mx_tensor import MXTensor

        from rtx.formats.mxfp8 import (
            mxfp8_scale_layout,
            mxfp8_scales_for_kernel,
        )

        for rows, expected_layout in ((64, "mma64"), (128, "mma128")):
            with self.subTest(rows=rows):
                value = MXTensor.from_qdata_and_scales(
                    torch.empty(rows, 128, dtype=torch.float8_e4m3fn),
                    torch.empty(32, 16, dtype=torch.float8_e8m0fnu),
                    torch.bfloat16,
                    block_size=32,
                    is_swizzled_scales=True,
                )
                self.assertEqual(mxfp8_scale_layout(value), expected_layout)
                self.assertEqual(
                    mxfp8_scales_for_kernel(value).shape,
                    (1, 1, 512),
                )

    def test_external_torchao_nvfp4_tensor_is_accepted(self) -> None:
        from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

        value = NVFP4Tensor(
            torch.empty(64, 64, dtype=torch.uint8),
            torch.empty(64, 8, dtype=torch.float8_e4m3fn),
            16,
            torch.bfloat16,
        )
        layer = rtx.NVFP4Linear(128, 64, packed_weight=value)
        self.assertIsInstance(layer.packed_weight, NVFP4Tensor)

    def test_mxfp8_is_no_bias_nn_linear_compatible(self) -> None:
        layer = rtx.MXFP8Linear(128, 64, bias=False, device="cpu")
        self.assertEqual(layer.weight.shape, (64, 128))
        self.assertIsNone(layer.bias)
        self.assertEqual(tuple(layer.state_dict()), ("weight",))
        with self.assertRaisesRegex(NotImplementedError, "bias=False"):
            rtx.MXFP8Linear(128, 64, bias=True, device="cpu")

    def test_nvfp4_public_contract_is_no_bias_bf16(self) -> None:
        layer = rtx.NVFP4Linear(128, 64, bias=False, device="cpu")
        self.assertEqual(layer.weight.shape, (64, 128))
        self.assertEqual(layer.weight.dtype, torch.bfloat16)
        self.assertIsNone(layer.bias)
        self.assertEqual(tuple(layer.state_dict()), ("weight",))
        self.assertIn("forward=NVFP4", layer.extra_repr())
        self.assertIn("backward=MXFP8", layer.extra_repr())
        self.assertEqual(layer.scaling, "jit_row_region")
        self.assertEqual(layer.x_scale_region_rows, 5)
        self.assertEqual(layer.weight_scale_region_rows, 4)

    def test_linear_explain_is_side_effect_free_and_shape_specific(self) -> None:
        x = torch.empty(2, 3, 128, dtype=torch.bfloat16)
        mx = rtx.MXFP8Linear(128, 64, device="cpu", autotune="cache")
        mx_decision = mx.explain(x)
        self.assertEqual(mx_decision.format, "mxfp8")
        self.assertEqual(mx_decision.problem, (6, 64, 128))
        self.assertEqual(mx_decision.operand_state, "dynamic")
        self.assertEqual(mx_decision.selection_source, "deferred_runtime")

        nv = rtx.NVFP4Linear(128, 64, device="cpu", autotune="off")
        nv_decision = nv.explain(x)
        self.assertEqual(nv_decision.format, "nvfp4")
        self.assertEqual(nv_decision.family, "nvfp4_jit_row_region_fwd")
        self.assertEqual(nv_decision.selection_source, "portable")
        self.assertEqual(nv_decision.x_scale_region_rows, 5)
        self.assertEqual(nv_decision.weight_scale_region_rows, 4)

        online = rtx.MXFP8Linear(
            128, 64, device="cpu", autotune="online"
        ).explain(x)
        self.assertEqual(online.autotune, "coordinate")

    def test_nvfp4_jit_row_region_is_canonical_and_materialized(self) -> None:
        layer = rtx.NVFP4Linear(
            128,
            64,
            device="cpu",
            scaling="jit_row_region",
            scale_region_rows=8,
        )
        self.assertEqual(layer.scaling, "jit_row_region")
        self.assertEqual(layer.scale_region_rows, 8)
        with self.assertRaisesRegex(ValueError, "scaling"):
            rtx.NVFP4Linear(128, 64, device="cpu", scaling="regional")
        with self.assertRaisesRegex(ValueError, "backend"):
            rtx.NVFP4Linear(
                128,
                64,
                device="cpu",
                scaling="jit_row_region",
                scale_region_rows=128,
                backend="fused",  # type: ignore[arg-type]
            )
        fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
        if fp4_dtype is not None:
            packed = make_nvfp4_tensor(
                torch.empty(64, 64, dtype=fp4_dtype),
                torch.empty(64, 8, dtype=torch.float8_e4m3fn),
                torch.ones((), dtype=torch.float32),
                (64, 128),
            )
            with self.assertRaisesRegex(ValueError, "current or block"):
                rtx.NVFP4Linear(
                    128,
                    64,
                    packed_weight=packed,
                    scaling="jit_row_region",
                )

    def test_backend_terminology_is_canonical_across_frontends(self) -> None:
        legacy = rtx.MXFP8Linear(128, 64, device="cpu", backend="prequant")
        self.assertEqual(legacy.backend, "materialized")
        self.assertEqual(
            rtx.MXFP8Linear(128, 64, device="cpu", backend="materialized").backend,
            "materialized",
        )
        with self.assertRaisesRegex(ValueError, "auto or materialized"):
            rtx.NVFP4Linear(128, 64, device="cpu", backend="fused")

    def test_nvfp4_scale_policy_has_only_effective_public_fields(self) -> None:
        policy = rtx.NVFP4ScaleConfig(
            tensor_scale_mode="exact",
            amax_history_len=4,
            amax_history_algo="most_recent",
        )
        layer = rtx.NVFP4Linear(
            128,
            64,
            device="cpu",
            scaling="delayed",
            scale_config=policy,
        )
        self.assertIs(layer.scale_config, policy)
        self.assertFalse(hasattr(policy, "tile_m"))

    def test_mxfp8_canonical_config_names_and_legacy_aliases(self) -> None:
        forward = rtx.MXFP8FwdConfig()
        dynamic = rtx.MXFP8PrequantConfig()
        layer = rtx.MXFP8Linear(
            128,
            64,
            device="cpu",
            forward_config=forward,
            dynamic_config=dynamic,
        )
        self.assertIs(layer.config, forward)
        self.assertIs(layer.prequant_config, dynamic)
        self.assertIs(layer.forward_config, forward)
        self.assertIs(layer.dynamic_config, dynamic)
        with self.assertRaisesRegex(ValueError, "forward_config"):
            rtx.MXFP8Linear(
                128,
                64,
                device="cpu",
                config=forward,
                forward_config=forward,
            )
        with self.assertRaisesRegex(ValueError, "dynamic_config"):
            rtx.MXFP8Linear(
                128,
                64,
                device="cpu",
                prequant_config=dynamic,
                dynamic_config=dynamic,
            )
        self.assertEqual(
            rtx.NVFP4Linear(
                128,
                64,
                device="cpu",
                scaling="block",
                backend="materialized",
            ).backend,
            "materialized",
        )
        with self.assertRaisesRegex(ValueError, "auto, fused, or materialized"):
            rtx.MXFP8Linear(128, 64, device="cpu", backend="unknown")
        with self.assertRaisesRegex(ValueError, "auto or materialized"):
            rtx.NVFP4Linear(128, 64, device="cpu", backend="unknown")

    def test_nvfp4_exposes_all_materialized_autotune_states(self) -> None:
        from rtx import fp4

        dynamic = rtx.NVFP4DynamicConfig()
        weight = rtx.NVFP4WeightPrequantConfig()
        fully = rtx.NVFP4FullyPrequantConfig()
        layer = rtx.NVFP4Linear(
            128,
            64,
            device="cpu",
            scaling="block",
            backend="materialized",
            autotune="off",
            dynamic_config=dynamic,
            weight_prequant_config=weight,
            fully_prequant_config=fully,
            autotune_cache_dir="test-cache",
        )
        request = fp4._AUTOTUNE_REQUESTS[layer._materialized_request_key]
        self.assertEqual(request.mode, "off")
        self.assertEqual(request.dynamic, dynamic)
        self.assertEqual(request.weight_prequantized, weight)
        self.assertEqual(request.fully_prequantized, fully)
        self.assertTrue(request.cache_dir.endswith("test-cache"))

    def test_nvfp4_explicit_dynamic_config_bypasses_runtime_winners(self) -> None:
        from rtx import fp4

        config = rtx.NVFP4DynamicConfig()
        request_key = fp4._intern_autotune_request(
            "off", None, None, dynamic=config
        )
        key = fp4._materialized_dynamic_config_key(
            rtx.NVFP4Problem(128, 64, 128),
            torch.device("cpu"),
            request_key,
        )
        self.assertEqual(
            fp4._resolve_forward_config(key),
            fp4.NVFP4ForwardConfig.from_materialized(config),
        )

    def test_jit_row_region_requested_geometry_seeds_uncached_config(self) -> None:
        from rtx import fp4

        request_key = fp4._intern_autotune_request("off", None, None)
        key = fp4._materialized_dynamic_config_key(
            rtx.NVFP4Problem(128, 64, 128),
            torch.device("cpu"),
            request_key,
            "nvfp4_jit_row_region_fwd",
            8,
            4,
        )
        config = fp4._resolve_forward_config(key)
        self.assertEqual(config.x_scale_region_rows, 8)
        self.assertEqual(config.weight_scale_region_rows, 4)

    def test_jit_row_region_explicit_geometry_overrides_fixed_winner(self) -> None:
        from dataclasses import replace

        from rtx import fp4
        from rtx.nvfp4_inference_autotune import (
            preferred_jit_row_region_config,
        )

        problem = rtx.NVFP4Problem(128, 128, 256)
        winner = replace(
            preferred_jit_row_region_config(problem),
            x_scale_region_rows=16,
            weight_scale_region_rows=8,
        )
        request_key = fp4._intern_autotune_request(
            "off", None, None, dynamic=winner
        )
        inherited_key = fp4._materialized_dynamic_config_key(
            problem,
            torch.device("cpu"),
            request_key,
            "nvfp4_jit_row_region_fwd",
        )
        inherited = fp4._resolve_forward_config(inherited_key)
        self.assertEqual(inherited.x_scale_region_rows, 16)
        self.assertEqual(inherited.weight_scale_region_rows, 8)

        explicit_key = fp4._materialized_dynamic_config_key(
            problem,
            torch.device("cpu"),
            request_key,
            "nvfp4_jit_row_region_fwd",
            7,
            3,
        )
        explicit = fp4._resolve_forward_config(explicit_key)
        self.assertEqual(explicit.x_scale_region_rows, 7)
        self.assertEqual(explicit.weight_scale_region_rows, 3)

    def test_nvfp4_rejects_bias_and_non_bf16_parameters(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "bias=False"):
            rtx.NVFP4Linear(128, 64, bias=True, device="cpu")
        with self.assertRaisesRegex(TypeError, "BF16"):
            rtx.NVFP4Linear(
                128,
                64,
                bias=False,
                device="cpu",
                dtype=torch.float32,
            )

    def test_both_dispatcher_families_have_fake_and_autograd_ops(self) -> None:
        _ = rtx.MXFP8Linear, rtx.NVFP4Linear, rtx.DEFAULT_MXFP8_BWD_CONFIG
        with FakeTensorMode():
            x = torch.empty(
                128, 128, device="cuda", dtype=torch.bfloat16
            )
            weight = torch.empty(
                64, 128, device="cuda", dtype=torch.bfloat16
            )
            nvfp4_scale = torch.ones(1, device="cuda", dtype=torch.float32)
            calls = (
                (torch.ops.rtx.mxfp8_linear_fwd, (x, weight, "fake")),
                (
                    torch.ops.rtx.mxfp8_linear_train,
                    (x, weight, "fake", "fake"),
                ),
                (
                    torch.ops.rtx.nvfp4_linear_materialized_fwd,
                    (x, weight, nvfp4_scale, nvfp4_scale, nvfp4_scale, "fake"),
                ),
                (
                    torch.ops.rtx.nvfp4_linear_materialized_train,
                    (
                        x,
                        weight,
                        nvfp4_scale,
                        nvfp4_scale,
                        nvfp4_scale,
                        "fake",
                        "fake",
                    ),
                ),
                (
                    torch.ops.rtx.nvfp4_linear_block_fwd,
                    (x, weight, "fake"),
                ),
                (
                    torch.ops.rtx.nvfp4_linear_block_train,
                    (x, weight, "fake", "fake"),
                ),
                (
                    torch.ops.rtx.nvfp4_linear_jit_row_region_fwd,
                    (x, weight, "fake"),
                ),
                (
                    torch.ops.rtx.nvfp4_linear_jit_row_region_train,
                    (x, weight, "fake", "fake"),
                ),
            )
            for op, args in calls:
                with self.subTest(op=op):
                    result = op(*args)
                    self.assertEqual(result.shape, (128, 64))
                    self.assertEqual(result.dtype, torch.bfloat16)

    def test_mxfp8_fake_dispatch_covers_both_inference_packing_states(self) -> None:
        _ = rtx.MXFP8Linear
        with FakeTensorMode():
            x = torch.empty(32, 128, device="cuda", dtype=torch.bfloat16)
            qx = torch.empty(32, 128, device="cuda", dtype=torch.float8_e4m3fn)
            qw = torch.empty(64, 128, device="cuda", dtype=torch.float8_e4m3fn)
            sx = torch.empty(32, 4, device="cuda", dtype=torch.float8_e8m0fnu)
            sw = torch.empty(64, 4, device="cuda", dtype=torch.float8_e8m0fnu)
            dynamic_x = torch.ops.rtx.mxfp8_linear_dynamic_x_prequant_w(
                x, qw, sw, 64, 128, "row_major", "fake"
            )
            packed = torch.ops.rtx.mxfp8_linear_prequantized(
                qx,
                qw,
                sx,
                sw,
                32,
                64,
                128,
                "row_major",
                "row_major",
                "fake",
            )
            self.assertEqual(dynamic_x.shape, (32, 64))
            self.assertEqual(packed.shape, (32, 64))

    def test_packed_mxfp8_module_has_no_bf16_master_weight(self) -> None:
        with FakeTensorMode():
            qw = torch.empty(64, 128, device="cuda", dtype=torch.float8_e4m3fn)
            sw = torch.empty(64, 4, device="cuda", dtype=torch.float8_e8m0fnu)
            packed = make_mxfp8_tensor(qw, sw, (64, 128))
            layer = rtx.MXFP8Linear(128, 64, packed_weight=packed)
            self.assertEqual(layer.weight_mode, "prequantized")
            self.assertIsNone(layer.weight)
            self.assertEqual(
                tuple(layer.state_dict()),
                ("weight_data", "weight_scales", "weight_packing_meta"),
            )
            self.assertEqual(layer.packed_weight.shape, (64, 128))

    def test_packed_mxfp8_state_dict_round_trip_and_cast_guard(self) -> None:
        qw = torch.zeros(64, 128, device="cpu", dtype=torch.float8_e4m3fn)
        sw = torch.zeros(64, 4, device="cpu", dtype=torch.float8_e8m0fnu)
        source = rtx.MXFP8Linear(
            128, 64, packed_weight=make_mxfp8_tensor(qw, sw, (64, 128))
        )
        target = rtx.MXFP8Linear(
            128,
            64,
            packed_weight=make_mxfp8_tensor(
                torch.empty_like(qw), torch.empty_like(sw), (64, 128)
            ),
        )
        target.load_state_dict(source.state_dict())
        self.assertTrue(torch.equal(target.weight_data, source.weight_data))
        self.assertTrue(torch.equal(target.weight_scales, source.weight_scales))
        self.assertIs(target.to(device="cpu"), target)
        self.assertFalse(target.training)
        with self.assertRaisesRegex(RuntimeError, "inference-only"):
            target.train()
        self.assertIs(target.eval(), target)
        with self.assertRaisesRegex(TypeError, "cannot be cast"):
            target.to(dtype=torch.bfloat16)
        with self.assertRaisesRegex(TypeError, "cannot be dtype-cast"):
            target.float()

    def test_mxfp8_functional_and_module_fake_paths_use_packed_contracts(self) -> None:
        with FakeTensorMode():
            x = torch.empty(2, 128, device="cuda", dtype=torch.bfloat16)
            qx = torch.empty(2, 128, device="cuda", dtype=torch.float8_e4m3fn)
            qw = torch.empty(64, 128, device="cuda", dtype=torch.float8_e4m3fn)
            sx = torch.empty(2, 4, device="cuda", dtype=torch.float8_e8m0fnu)
            sw = torch.empty(64, 4, device="cuda", dtype=torch.float8_e8m0fnu)
            packed_x = make_mxfp8_tensor(qx, sx, (2, 128))
            packed_w = make_mxfp8_tensor(qw, sw, (64, 128))
            layer = rtx.MXFP8Linear(128, 64, packed_weight=packed_w)
            self.assertEqual(layer(x).shape, (2, 64))
            self.assertEqual(layer(packed_x).shape, (2, 64))
            self.assertEqual(rtx.mxfp8_linear(x, packed_w).shape, (2, 64))
            self.assertEqual(
                rtx.mxfp8_linear(packed_x, packed_w).shape, (2, 64)
            )

    def test_mxfp8_packed_operands_preserve_ragged_logical_k(self) -> None:
        from rtx.formats.mxfp8 import mxfp8_matrix_shape, mxfp8_qdata_2d

        with FakeTensorMode():
            x = torch.empty(17, 33, device="cuda", dtype=torch.bfloat16)
            qx = torch.empty(17, 64, device="cuda", dtype=torch.float8_e4m3fn)
            qw = torch.empty(29, 64, device="cuda", dtype=torch.float8_e4m3fn)
            sx = torch.empty(17, 2, device="cuda", dtype=torch.float8_e8m0fnu)
            sw = torch.empty(29, 2, device="cuda", dtype=torch.float8_e8m0fnu)
            packed_x = make_mxfp8_tensor(qx, sx, (17, 33))
            packed_w = make_mxfp8_tensor(qw, sw, (29, 33))
            self.assertEqual(mxfp8_matrix_shape(packed_x), (17, 33))
            self.assertEqual(tuple(mxfp8_qdata_2d(packed_x).shape), (17, 64))
            layer = rtx.MXFP8Linear(33, 29, packed_weight=packed_w)
            self.assertEqual(layer(x).shape, (17, 29))
            self.assertEqual(layer(packed_x).shape, (17, 29))

    def test_packed_cache_selection_is_deferred_behind_an_opaque_token(self) -> None:
        from rtx import fp8
        from rtx.kernels.mxfp8 import MXFP8Problem

        data = torch.empty(64, 128, dtype=torch.float8_e4m3fn)
        scales = torch.empty(64, 4, dtype=torch.float8_e8m0fnu)
        weight = make_mxfp8_tensor(data, scales, (64, 128))
        key = fp8._packed_inference_config_key(
            MXFP8Problem(32, 64, 128),
            weight,
            x=None,
            explicit=None,
            autotune="cache",
            tuning_policy=None,
            cache_dir=None,
        )
        self.assertTrue(key.startswith("packed-autotune:"))
        self.assertIn(key, fp8._PACKED_INFERENCE_AUTOTUNE_REQUESTS)

    def test_nvfp4_fake_dispatch_covers_packed_states(self) -> None:
        _ = rtx.NVFP4Linear
        fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
        if fp4_dtype is None:
            self.skipTest("PyTorch does not expose packed FP4")
        with FakeTensorMode():
            x = torch.empty(32, 128, device="cuda", dtype=torch.bfloat16)
            qx = torch.empty(32, 64, device="cuda", dtype=fp4_dtype)
            qw = torch.empty(64, 64, device="cuda", dtype=fp4_dtype)
            sx = torch.empty(32, 8, device="cuda", dtype=torch.float8_e4m3fn)
            sw = torch.empty(64, 8, device="cuda", dtype=torch.float8_e4m3fn)
            gx = torch.ones((), device="cuda", dtype=torch.float32)
            gw = torch.ones((), device="cuda", dtype=torch.float32)
            dynamic_x = torch.ops.rtx.nvfp4_linear_dynamic_x_prequant_w(
                x, qw, sw, gx, gw, 64, 128, "row_major", "fake"
            )
            packed = torch.ops.rtx.nvfp4_linear_prequantized(
                qx,
                qw,
                sx,
                sw,
                gw,
                32,
                64,
                128,
                "row_major",
                "row_major",
                "fake",
            )
            self.assertEqual(dynamic_x.shape, (32, 64))
            self.assertEqual(packed.shape, (32, 64))

    def test_nvfp4_container_uses_two_logical_values_per_storage_byte(self) -> None:
        fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
        if fp4_dtype is None:
            self.skipTest("PyTorch does not expose packed FP4")
        data = torch.empty(64, 64, dtype=fp4_dtype)
        scales = torch.empty(64, 8, dtype=torch.float8_e4m3fn)
        tensor_scale = torch.ones((), dtype=torch.float32)
        packed = make_nvfp4_tensor(
            data, scales, tensor_scale, shape=(64, 128)
        )
        layer = rtx.NVFP4Linear(128, 64, packed_weight=packed)
        self.assertEqual(layer.weight_data.shape, (64, 64))
        self.assertIsNone(layer.weight)
        self.assertFalse(layer.training)
        with self.assertRaisesRegex(ValueError, "qdata storage"):
            make_nvfp4_tensor(
                torch.empty(64, 128, dtype=fp4_dtype),
                scales,
                tensor_scale,
                shape=(64, 128),
            )

    def test_nvfp4_packed_operands_preserve_ragged_logical_k(self) -> None:
        fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
        if fp4_dtype is None:
            self.skipTest("PyTorch does not expose packed FP4")
        from rtx.formats.nvfp4 import nvfp4_matrix_shape

        with FakeTensorMode():
            scale = torch.ones((), device="cuda", dtype=torch.float32)
            qx = torch.empty(17, 24, device="cuda", dtype=fp4_dtype)
            qw = torch.empty(29, 24, device="cuda", dtype=fp4_dtype)
            sx = torch.empty(17, 3, device="cuda", dtype=torch.float8_e4m3fn)
            sw = torch.empty(29, 3, device="cuda", dtype=torch.float8_e4m3fn)
            packed_x = make_nvfp4_tensor(qx, sx, scale, (17, 33))
            packed_w = make_nvfp4_tensor(qw, sw, scale, (29, 33))
            self.assertEqual(nvfp4_matrix_shape(packed_x), (17, 33))
            layer = rtx.NVFP4Linear(
                33, 29, packed_weight=packed_w, autotune="off"
            )
            self.assertEqual(layer(packed_x).shape, (17, 29))


if __name__ == "__main__":
    unittest.main()
