from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import unittest

from rtx.autotune import (
    DiscreteKernelAdapter,
    GradientBoostedFeasibilityModel,
    KernelContext,
    TrialOutcome,
    make_mxfp8_fwd_adapter,
)
from rtx.autotune.core import Proposal, evaluate_proposal
from rtx.autotune.hardware import (
    architecture_profile,
    geometry_features,
    launch_resource_features,
    sku_spec,
)
from rtx.kernels.mxfp8 import DEFAULT_MXFP8_FWD_CONFIG, MXFP8Problem


@dataclass(frozen=True)
class _CompileConfig:
    x: int = 0
    y: int = 0


class HardwareAutotuneTests(unittest.TestCase):
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
