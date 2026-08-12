from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import unittest

from rtx.autotune import (
    DiscreteKernelAdapter,
    GradientBoostedFeasibilityModel,
    KernelContext,
    TrialOutcome,
    make_mxfp8_fwd_adapter,
    make_mxfp8_bwd_adapter,
)
from rtx.autotune.adapters import (
    _fused_smem_bytes,
    _gemm_features,
    _gemm_launch_smem_bytes,
)
from rtx.autotune.core import Proposal, evaluate_proposal
from rtx.autotune.hardware import (
    architecture_profile,
    geometry_features,
    launch_resource_features,
    sku_spec,
)
from rtx.kernels.mxfp8 import (
    DEFAULT_MXFP8_FWD_CONFIG,
    MXFP8Problem,
    normalize_fwd_config,
)
from rtx.kernels.mxfp8_bwd import (
    DEFAULT_FUSED_MXFP8_BWD_CONFIG,
    DEFAULT_MXFP8_BWD_CONFIG,
)
from rtx.kernels.mxfp8_bwd import DEFAULT_DECOMPOSED_MXFP8_BWD_CONFIG
from rtx.configs import MXFP8GemmConfig
from rtx.bwd_autotune import BWD_SEARCH_SPACE, update_bwd_config


@dataclass(frozen=True)
class _CompileConfig:
    x: int = 0
    y: int = 0


class HardwareAutotuneTests(unittest.TestCase):
    def test_persistent_gemm_features_use_actual_grid_and_reuse_edges(self) -> None:
        features = _gemm_features(
            MXFP8Problem(512, 1536, 1536),
            MXFP8GemmConfig(tiles_per_cta=4, tile_locality="same_a"),
            None,
            materialized_quant=True,
        )
        self.assertEqual(features["natural_ctas"], 48.0)
        self.assertEqual(features["grid_ctas"], 12.0)
        self.assertEqual(features["work_tiles_per_cta"], 4.0)
        self.assertEqual(features["consecutive_a_reuse_edges"], 36.0)
        self.assertEqual(features["consecutive_b_reuse_edges"], 0.0)
        split_features = _gemm_features(
            MXFP8Problem(512, 1536, 1536),
            MXFP8GemmConfig(epilogue="direct", store_vec=1),
            None,
            materialized_quant=True,
            split_reduction=4,
        )
        self.assertEqual(split_features["natural_ctas"], 48.0)
        self.assertEqual(split_features["split_work_ctas"], 192.0)
        self.assertEqual(split_features["grid_ctas"], 192.0)

    def test_staged_fused_smem_includes_pipeline_alignment_reserve(self) -> None:
        candidate = normalize_fwd_config(
            load_engine="tma",
            schedule="three_role",
            bf16_tile_k=32,
            bf16_swizzle="none",
            bf16_stages=2,
            mxfp8_stages=2,
            quantizer_warps=4,
        )
        # Explicit arrays consume 100,352 bytes. The staged CuTe struct adds
        # one complete 1-KiB alignment quantum for its pipeline state.
        self.assertEqual(_fused_smem_bytes(candidate), 101_376)

    def test_backward_adapter_rejects_runtime_smem_overhead_before_launch(self) -> None:
        raw_limit = _gemm_launch_smem_bytes(
            DEFAULT_DECOMPOSED_MXFP8_BWD_CONFIG.dx.gemm
        ) - 1024
        adapter = make_mxfp8_bwd_adapter(
            MXFP8Problem(512, 1536, 1536),
            lambda _config: TrialOutcome("ok", median_ms=1.0),
            device={
                "properties": {
                    "shared_memory_per_block_optin": raw_limit,
                }
            },
        )
        rejection = adapter.rejection(DEFAULT_DECOMPOSED_MXFP8_BWD_CONFIG)
        self.assertIsNotNone(rejection)
        self.assertIn("runtime overhead", rejection[1])

    def test_backward_features_model_shared_grad_output_traffic(self) -> None:
        problem = MXFP8Problem(4096, 4096, 4096)
        adapter = make_mxfp8_bwd_adapter(
            problem,
            lambda _config: TrialOutcome("ok", median_ms=1.0),
        )
        shared_value = next(
            value
            for value in BWD_SEARCH_SPACE["shared_g_quad"]
            if value["dx"]["quant_a"]["transposed_tile_rows"] == 128
            and value["dx"]["quant_a"]["transposed_load_engine"] == "cp_async"
            and value["dx"]["quant_a"]["native_scale_store"] == "packed"
        )
        shared = update_bwd_config(DEFAULT_MXFP8_BWD_CONFIG, shared_value)
        baseline_features = adapter.features(DEFAULT_MXFP8_BWD_CONFIG)
        shared_features = adapter.features(shared)
        self.assertEqual(
            shared_features["derived.backward_shared_g_bf16_bytes_saved"],
            4096 * 4096 * 2,
        )
        self.assertEqual(
            baseline_features["derived.backward_grad_output_bf16_read_bytes"],
            2 * shared_features["derived.backward_grad_output_bf16_read_bytes"],
        )
        self.assertEqual(
            shared_features["derived.backward_shared_g_tile_count"], 32 * 32
        )
        self.assertEqual(
            shared_features["derived.backward_total_kernel_launches"], 3
        )

    def test_backward_features_model_persistent_split_pipeline_topology(self) -> None:
        problem = MXFP8Problem(1024, 512, 512)
        persistent = normalize_fwd_config(
            DEFAULT_FUSED_MXFP8_BWD_CONFIG.dw.fused,
            persistent=True,
            persistent_waves=1,
            reuse="x",
        )
        candidate = update_bwd_config(
            DEFAULT_FUSED_MXFP8_BWD_CONFIG,
            {
                "dw": {
                    "fused": asdict(persistent),
                    "reduction": "split_fp32_workspace",
                    "split_reduction": 8,
                    "reduction_tile": 128,
                    "workspace_epilogue": "tree",
                }
            },
        )
        adapter = make_mxfp8_bwd_adapter(
            problem,
            lambda _config: TrialOutcome("ok", median_ms=1.0),
            device={"properties": {"multiprocessor_count": 70}},
        )
        features = adapter.features(candidate)
        self.assertEqual(features["derived.dw_natural_ctas"], 16)
        self.assertEqual(features["derived.dw_split_work_ctas"], 16 * 8)
        self.assertEqual(features["derived.dw_grid_ctas"], 64)
        self.assertEqual(features["derived.dw_work_tiles_per_cta"], 2)
        self.assertEqual(features["derived.dw_persistent_split"], 1)
        self.assertEqual(
            features["derived.dw_persistent_split_grid_ctas_per_slice"], 8
        )
        self.assertEqual(
            features["derived.dw_persistent_split_pipeline_tail_count"], 1
        )
        self.assertEqual(
            features["derived.dw_persistent_split_tiles_per_pipeline_tail"], 2
        )

    def test_backward_features_identify_oriented_cpasync_ldmatrix(self) -> None:
        problem = MXFP8Problem(512, 1536, 1536)
        transport = normalize_fwd_config(
            cpasync_ldmatrix_pipeline=(4, 1, "bf16x2", "bf16_bits")
        )
        candidate = update_bwd_config(
            DEFAULT_FUSED_MXFP8_BWD_CONFIG,
            {
                "dx": {"fused": asdict(transport)},
                "dw": {"fused": asdict(transport)},
            },
        )
        adapter = make_mxfp8_bwd_adapter(
            problem,
            lambda _config: TrialOutcome("ok", median_ms=1.0),
        )
        features = adapter.features(candidate)
        self.assertEqual(features["derived.dx_logical_transpose_operands"], 1)
        self.assertEqual(features["derived.dw_logical_transpose_operands"], 2)
        self.assertEqual(
            features["derived.dx_oriented_cpasync_ldmatrix_operands"], 1
        )
        self.assertEqual(
            features["derived.dw_oriented_cpasync_ldmatrix_operands"], 2
        )

    def test_backward_features_account_for_cpasync_cluster_load_elision(self) -> None:
        problem = MXFP8Problem(512, 1536, 1536)
        transport = normalize_fwd_config(
            cpasync_cluster_reuse_tile=(
                "a",
                4,
                4,
                "bf16x2",
                "bf16_bits",
            )
        )
        candidate = update_bwd_config(
            DEFAULT_FUSED_MXFP8_BWD_CONFIG,
            {
                "dx": {"fused": asdict(transport)},
                "dw": {"fused": asdict(transport)},
            },
        )
        adapter = make_mxfp8_bwd_adapter(
            problem,
            lambda _config: TrialOutcome("ok", median_ms=1.0),
        )
        features = adapter.features(candidate)
        expected_saved = 36 * 128 * 1536 * 2
        self.assertEqual(features["derived.dx_cluster_operand_reuse"], 1)
        self.assertEqual(features["derived.dx_cluster_operand_reuse_a"], 1)
        self.assertEqual(features["derived.dx_cluster_operand_reuse_size"], 4)
        self.assertEqual(
            features["derived.dx_cluster_operand_bf16_bytes_saved"],
            expected_saved,
        )
        self.assertGreater(
            features["derived.dx_cluster_dsmem_publication_bytes"], 0
        )
        self.assertEqual(
            features["derived.dx_cluster_dsmem_publication_bytes"],
            36 * (128 * 1536 + 128 * (1536 // 32)),
        )
        self.assertLess(
            features["derived.dx_estimated_operand_read_bytes"],
            features["derived.dx_x_bytes"]
            * features["derived.dx_x_reuse_ctas"]
            + features["derived.dx_weight_bytes"]
            * features["derived.dx_weight_reuse_ctas"],
        )

    def test_prequant_config_rejects_runtime_smem_reserve_statically(self) -> None:
        from rtx.configs import MXFP8GemmConfig

        candidate = MXFP8GemmConfig(
            tile_m=128,
            tile_n=128,
            tile_k=128,
            stages=3,
            epilogue="direct",
            store_vec=1,
        )
        reason = candidate.rejection(MXFP8Problem(128, 128, 128))
        self.assertIsNotNone(reason)
        self.assertIn("runtime reserve", reason)

    def test_architecture_and_sku_profiles_cover_target_devices(self) -> None:
        unsupported = architecture_profile((11, 0))
        self.assertEqual(unsupported.execution_model, "unknown")
        self.assertFalse(unsupported.supports_mxfp8)
        self.assertEqual(architecture_profile((12, 0)).tensor_accumulator, "rmem")
        self.assertEqual(
            sku_spec("NVIDIA GeForce RTX 5090")["memory_bus_width_bits"], 512
        )
        laptop = sku_spec("NVIDIA GeForce RTX 5070 Laptop GPU")
        self.assertEqual(laptop["memory_bus_width_bits"], 128)
        self.assertEqual(laptop["theoretical_memory_bandwidth_gbps"], 384.0)
        self.assertEqual(sku_spec("NVIDIA Jetson T5000")["sku_family"], "unknown")

    def test_resource_features_compute_multi_residency_waves(self) -> None:
        profile = {
            "multiprocessor_count": 70,
            "properties": {
                "max_threads_per_multiprocessor": 1536,
                "shared_memory_per_multiprocessor": 101_376,
                "regs_per_multiprocessor": 65_536,
            },
        }
        geometry = geometry_features(
            m=512,
            n=1536,
            k=1536,
            tile_m=128,
            tile_n=128,
            tile_k=128,
            profile=profile,
        )
        resources = launch_resource_features(
            profile=profile,
            grid_ctas=int(geometry["grid_ctas"]),
            threads_per_cta=256,
            smem_bytes_per_cta=32_768,
            register_budget_per_cta=32_768,
            register_limit_per_thread=128,
        )
        self.assertEqual(geometry["grid_ctas"], 48)
        self.assertEqual(resources["estimated_resident_ctas_per_sm"], 2)
        self.assertAlmostEqual(resources["effective_cta_waves"], 48 / 140)

    def test_fused_adapter_exposes_resource_and_traffic_features(self) -> None:
        profile = {
            "name": "synthetic sm120",
            "capability": [12, 0],
            "multiprocessor_count": 70,
            "properties": {
                "multiprocessor_count": 70,
                "max_threads_per_multiprocessor": 1536,
                "shared_memory_per_multiprocessor": 101_376,
                "shared_memory_per_block_optin": 101_376,
                "regs_per_multiprocessor": 65_536,
                "l2_cache_size": 48 << 20,
            },
            "sku": {"theoretical_memory_bandwidth_gbps": 896.0},
        }
        adapter = make_mxfp8_fwd_adapter(
            MXFP8Problem(512, 1536, 1536),
            lambda _config: TrialOutcome("ok", median_ms=1.0),
            device=profile,
        )
        features = adapter.features(DEFAULT_MXFP8_FWD_CONFIG)
        self.assertEqual(features["derived.grid_ctas"], 48)
        self.assertGreater(features["derived.smem_bytes_per_cta"], 0)
        self.assertGreater(features["derived.estimated_total_memory_bytes"], 0)
        self.assertGreater(features["derived.memory_roofline_ms"], 0)
        clustered = normalize_fwd_config(
            cpasync_cluster_reuse_tile=(
                "a",
                4,
                4,
                "bf16x2",
                "bf16_bits",
            )
        )
        clustered_features = adapter.features(clustered)
        self.assertEqual(clustered_features["derived.cluster_operand_reuse"], 1)
        self.assertEqual(
            clustered_features["derived.cluster_operand_bf16_bytes_saved"],
            36 * 128 * 1536 * 2,
        )
        self.assertLess(
            clustered_features["derived.estimated_operand_read_bytes"],
            features["derived.estimated_operand_read_bytes"],
        )

    def test_fused_adapter_rejects_nvvm_explosive_persistent_work(self) -> None:
        profile = {
            "capability": [12, 0],
            "multiprocessor_count": 70,
            "properties": {
                "multiprocessor_count": 70,
                "shared_memory_per_block_optin": 101_376,
            },
        }
        adapter = make_mxfp8_fwd_adapter(
            MXFP8Problem(32_768, 3_072, 768),
            lambda _config: TrialOutcome("ok", median_ms=1.0),
            device=profile,
        )
        explosive = normalize_fwd_config(
            persistent_tma_pipeline=(1, 1, 1, "none")
        )
        boundary = normalize_fwd_config(
            persistent_tma_pipeline=(1, 1, 2, "none")
        )

        rejection = adapter.rejection(explosive)
        self.assertIsNotNone(rejection)
        self.assertEqual(rejection[0], "implementation_rejected")
        self.assertIn("192 tiles per CTA", rejection[1])
        self.assertIsNone(adapter.rejection(boundary))
        self.assertEqual(
            adapter.features(boundary)["derived.work_tiles_per_cta"],
            96.0,
        )

    def test_feasibility_model_learns_compile_boundary(self) -> None:
        adapter = DiscreteKernelAdapter(
            context=KernelContext("compile", 1, {"m": 1}),
            initial_config=_CompileConfig(),
            axes={"x": tuple(range(6)), "y": tuple(range(4))},
            config_id_fn=lambda config: f"{config.x}-{config.y}",
            serialize_fn=asdict,
            deserialize_fn=lambda value: _CompileConfig(**value),
            update_fn=lambda config, coordinate, value: replace(
                config, **{coordinate: int(value)}
            ),
            evaluator=lambda config: TrialOutcome(
                "compile_error" if config.x >= 4 else "ok",
                median_ms=None if config.x >= 4 else 1.0,
            ),
            rejection_fn=lambda _config: None,
        )
        observations = []
        for x in range(6):
            for y in range(4):
                config = _CompileConfig(x, y)
                observations.append(
                    evaluate_proposal(
                        adapter,
                        Proposal(config, "grid"),
                        session_id="training",
                        sequence=len(observations),
                    )
                )
        model = GradientBoostedFeasibilityModel(
            n_estimators=24,
            ensembles=4,
            max_depth=2,
            min_leaf=2,
            seed=3,
        )
        model.fit(observations)
        probability, uncertainty = model.predict(
            [adapter.features(_CompileConfig(1, 0)), adapter.features(_CompileConfig(5, 0))]
        )
        self.assertGreater(float(probability[0]), float(probability[1]))
        self.assertTrue((uncertainty >= 0).all())


if __name__ == "__main__":
    unittest.main()
