from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from rtx.autotune.core import (
    DiscreteKernelAdapter,
    KernelContext,
    Proposal,
    SearchHistory,
    evaluate_proposal,
)
from rtx.autotune.cost_model import (
    GradientBoostedCostModel,
    GradientBoostedFeasibilityModel,
)
from rtx.autotune.legacy import TrialOutcome
from rtx.autotune.pretrained import (
    ConditionalRuleSet,
    NormalizedCostModel,
    analytical_baseline_ms,
    extract_conditional_rules,
    evaluate_latency_model,
    evaluate_pretrained_bundle,
    load_offline_observations,
    load_pretrained_family,
    train_pretrained_bundle,
)
from rtx.autotune.strategies import CostModelGuidedSearch


@dataclass(frozen=True)
class _Config:
    x: int = 0
    stages: int = 1


def _adapter(device: str = "device-a") -> DiscreteKernelAdapter[_Config]:
    context = KernelContext(
        "toy",
        7,
        {"m": 512, "n": 1536, "k": 1536},
        device={
            "sku": {"sku_family": device, "memory_bus_width_bits": 128},
            "multiprocessor_count": 36,
            "calibration": {"measured_native_mxfp8_gemm_tflops": 80.0},
        },
    )
    return DiscreteKernelAdapter(
        context=context,
        initial_config=_Config(),
        axes={"x": (0, 1, 2, 3), "stages": (1, 2, 3)},
        config_id_fn=lambda config: f"{config.x}-{config.stages}",
        serialize_fn=asdict,
        deserialize_fn=lambda value: _Config(**value),
        update_fn=lambda config, coordinate, value: replace(
            config, **{coordinate: int(value)}
        ),
        evaluator=lambda config: TrialOutcome(
            "ok",
            median_ms=0.1 + (config.x - 2) ** 2 + 0.1 * (config.stages - 2) ** 2,
        ),
        rejection_fn=lambda _config: None,
        extra_features_fn=lambda config: {
            "nominal_flops": 1.0e9,
            "memory_roofline_ms": 0.02,
            "effective_cta_waves": float(config.x + 1),
        },
    )


def _observations():
    adapter = _adapter()
    result = []
    sequence = 0
    for stages in (1, 2, 3):
        parent = _Config(0, stages)
        parent_observation = evaluate_proposal(
            adapter, Proposal(parent, "random"), session_id="s", sequence=sequence
        )
        sequence += 1
        result.append(parent_observation)
        for x in (1, 2, 3):
            child = _Config(x, stages)
            result.append(
                evaluate_proposal(
                    adapter,
                    Proposal(
                        child,
                        "model_local",
                        parent_config_id=adapter.config_id(parent),
                        coordinate="x",
                        coordinate_value=x,
                    ),
                    session_id="s",
                    sequence=sequence,
                )
            )
            sequence += 1
    return adapter, result


class PretrainedAutotuneTests(unittest.TestCase):
    def test_analytical_baseline_combines_compute_and_memory(self) -> None:
        features = {
            "derived.nominal_flops": 2.0e9,
            "derived.memory_roofline_ms": 0.01,
            "context.device.calibration.measured_native_mxfp8_gemm_tflops": 100.0,
        }
        self.assertAlmostEqual(analytical_baseline_ms(features), 0.02)

    def test_normalized_model_round_trip_and_prefit_search(self) -> None:
        adapter, observations = _observations()
        model = NormalizedCostModel(
            GradientBoostedCostModel(
                n_estimators=12, ensembles=2, max_depth=2, min_leaf=2, seed=4
            )
        )
        model.fit(observations)
        expected, _ = model.predict(
            [adapter.features(_Config(2, 2)), adapter.features(_Config(0, 1))]
        )
        self.assertLess(float(expected[0]), float(expected[1]))
        metrics = evaluate_latency_model(model, observations)
        self.assertIn("catalog_replay_regret", metrics)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cost.json"
            model.save(path)
            restored = NormalizedCostModel.load(path)
            actual, _ = restored.predict(
                [adapter.features(_Config(2, 2)), adapter.features(_Config(0, 1))]
            )
            self.assertTrue((abs(actual - expected) < 1e-10).all())
            strategy = CostModelGuidedSearch(
                model=restored, min_observations=32, pool_size=64
            )
            proposals = strategy.propose(adapter, SearchHistory([], adapter.context.identifier), __import__("random").Random(2), 1)
            self.assertTrue(proposals)
            self.assertNotEqual(proposals[0].metadata.get("phase"), "warmup")

    def test_conditional_rules_use_paired_parent_moves(self) -> None:
        adapter, observations = _observations()
        rules = extract_conditional_rules(
            observations, min_support=3, minimum_effect=0.001, seed=3
        )
        self.assertTrue(rules.rules)
        effect, matches = rules.adjustment(
            adapter.features(_Config(2, 2)), coordinate="x", coordinate_value=2
        )
        self.assertLess(effect, 0.0)
        self.assertTrue(matches)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rules.json"
            rules.save(path)
            self.assertEqual(
                len(ConditionalRuleSet.load(path).rules), len(rules.rules)
            )

    def test_loaded_feasibility_prior_survives_one_class_local_data(self) -> None:
        _adapter_value, observations = _observations()
        labeled = list(observations)
        for index in range(4):
            labeled[index] = replace(
                labeled[index], outcome=TrialOutcome("compile_error", error="test")
            )
        model = GradientBoostedFeasibilityModel(
            n_estimators=8, ensembles=2, max_depth=2, min_leaf=2, seed=9
        )
        model.fit(labeled)
        self.assertTrue(model.fitted)
        parameters = model.parameter_count
        model.fit(observations)
        self.assertTrue(model.fitted)
        self.assertEqual(model.parameter_count, parameters)

    def test_zip_loader_and_bundle_training_are_cpu_only(self) -> None:
        _adapter_value, observations = _observations()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "data.zip"
            payload = "".join(json.dumps(item.as_dict()) + "\n" for item in observations)
            payload_b = "".join(
                json.dumps(
                    {
                        **item.as_dict(),
                        "observation_id": f"bundle-b-{item.observation_id}",
                    }
                )
                + "\n"
                for item in observations
            )
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(
                    "bundle-a/stores/toy/hot/observations.jsonl", payload
                )
                archive.writestr(
                    "bundle-b/stores/toy/hot/observations.jsonl", payload_b
                )
            loaded, report = load_offline_observations(
                [archive_path], campaign=["bundle-a", "bundle-b"]
            )
            self.assertEqual(len(loaded), len(observations) * 2)
            self.assertEqual(report["malformed"], 0)
            output = root / "artifact"
            manifest = train_pretrained_bundle(
                [archive_path],
                output,
                n_estimators=8,
                ensembles=2,
                max_depth=2,
                min_leaf=2,
                min_rule_support=3,
                validate_devices=False,
                campaign="bundle-a",
            )
            self.assertIn("toy@7", manifest["families"])
            self.assertEqual(len(manifest["artifact_id"]), 24)
            self.assertTrue(manifest["input"]["dataset_sha256"])
            self.assertEqual(
                len(manifest["families"]["toy@7"]["files_sha256"]), 4
            )
            self.assertEqual(
                manifest["families"]["toy@7"]["deployment"]["selected_cost_head"],
                "none",
            )
            family = load_pretrained_family(output, "toy", 7)
            self.assertTrue(family.cost_model.fitted)
            self.assertEqual(family.artifact_id, manifest["artifact_id"])
            self.assertTrue((output / "manifest.json").exists())

            heldout_path = root / "heldout" / "observations.jsonl"
            heldout_path.parent.mkdir()
            heldout_path.write_text(payload_b, encoding="utf-8")
            evaluation = evaluate_pretrained_bundle(output, [heldout_path])
            self.assertTrue(evaluation["separation"]["held_out"])
            self.assertEqual(evaluation["families"]["toy@7"]["rows"], 12)
            with self.assertRaisesRegex(ValueError, "overlaps training data"):
                evaluate_pretrained_bundle(
                    output, [archive_path], campaign="bundle-a"
                )
            copied_overlap = root / "copied" / "observations.jsonl"
            copied_overlap.parent.mkdir()
            copied_overlap.write_text(payload, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "shared observations"):
                evaluate_pretrained_bundle(output, [copied_overlap])


if __name__ == "__main__":
    unittest.main()
