from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from rtx.autotune import (
    DiscreteKernelAdapter,
    JsonlTuningStore,
    KernelContext,
    TrialOutcome,
)
from rtx.autotune.core import Proposal, evaluate_proposal
from rtx.autotune.dataset import (
    AnytimeRunPolicy,
    DatasetCampaign,
    DatasetManifest,
    export_bundle,
    export_parquet,
    normalized_rows,
)
from rtx.autotune import dataset as dataset_module
from rtx.prequant_experiments import ExperimentJournal


@dataclass(frozen=True)
class _Config:
    tile: int = 64


def _adapter() -> DiscreteKernelAdapter[_Config]:
    return DiscreteKernelAdapter(
        context=KernelContext(
            family="test_kernel",
            kernel_revision=3,
            workload={"m": 128, "n": 256, "k": 512},
            device={"name": "synthetic", "multiprocessor_count": 70},
            regime="hot",
            tags={"source_sha256": "source"},
        ),
        initial_config=_Config(),
        axes={"tile": (64, 128)},
        config_id_fn=lambda config: f"tile-{config.tile}",
        serialize_fn=asdict,
        deserialize_fn=lambda value: _Config(**value),
        update_fn=lambda config, _coordinate, value: replace(config, tile=int(value)),
        evaluator=lambda config: TrialOutcome(
            "ok",
            median_ms=1.0 / config.tile,
            timings_ms=[1.0 / config.tile, 1.01 / config.tile],
            metadata={"calls_per_sample": 17, "telemetry_after": {"temperature": 55}},
        ),
        rejection_fn=lambda _config: None,
    )


class DatasetTests(unittest.TestCase):
    def test_anytime_policy_and_duration_parsing(self) -> None:
        self.assertEqual(dataset_module._parse_duration("2h"), 7200.0)
        self.assertEqual(dataset_module._parse_duration("90m"), 5400.0)
        self.assertEqual(dataset_module._parse_duration("15"), 15.0)
        self.assertEqual(dataset_module._parse_milestones("8,32,96"), (8, 32, 96))
        with self.assertRaises(ValueError):
            AnytimeRunPolicy(10.0, trial_milestones=(32, 16))
        with self.assertRaises(ValueError):
            AnytimeRunPolicy(10.0, context_bandit_discount=0.0)

    def test_context_bandit_uses_similarity_and_forces_real_samples(self) -> None:
        shape_a = dataset_module.ShapeSpec(128, 1536, 1536, "a")
        shape_b = dataset_module.ShapeSpec(256, 1536, 1536, "b")
        shape_c = dataset_module.ShapeSpec(8192, 4096, 4096, "c")
        descriptors = {
            "a": ("mxfp8_fused_fwd", shape_a, "hot"),
            "b": ("mxfp8_fused_fwd", shape_b, "hot"),
            "c": ("mxfp8_bwd", shape_c, "rotate"),
        }
        arms = {
            key: dataset_module._ContextArmStatistics() for key in descriptors
        }
        arms["a"].update(0.8, 1.0, True)
        arms["c"].update(-0.2, 1.0, False)
        scores = dataset_module._context_bandit_scores(
            tuple(descriptors), arms, descriptors, 0.1
        )
        self.assertEqual(scores["b"], float("inf"))
        arms["b"].update(0.0, 1.0, True)
        scores = dataset_module._context_bandit_scores(
            tuple(descriptors), arms, descriptors, 0.1
        )
        self.assertGreater(scores["b"], scores["c"])

    def test_anytime_context_order_is_breadth_first_across_families(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest = DatasetManifest.load(
            root / "autotune_manifests" / "cross_device_dataset_v2.json"
        )
        campaign = DatasetCampaign.__new__(DatasetCampaign)
        campaign.manifest = manifest
        campaign.anytime = AnytimeRunPolicy(60.0)
        contexts = campaign._assigned_contexts()
        first = [(job.family, shape.name, regime) for job, shape, regime in contexts[:3]]
        self.assertEqual(
            first,
            [
                ("mxfp8_fused_fwd", "balanced", "hot"),
                ("mxfp8_prequant_fwd", "balanced", "hot"),
                ("mxfp8_bwd", "balanced", "hot"),
            ],
        )
        self.assertEqual(len(contexts), 54)

    def test_existing_v2_context_identity_is_explicitly_reusable(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest = DatasetManifest.load(
            root / "autotune_manifests" / "cross_device_dataset_v2.json"
        )
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            source = {"python_source_sha256": "old-source", "git_commit": "old"}
            (bundle / "machine.json").write_text(
                json.dumps({"machine_id": "same-machine", "source": source}),
                encoding="utf-8",
            )
            (bundle / "manifest.json").write_text(
                json.dumps(manifest.as_dict()), encoding="utf-8"
            )
            campaign = DatasetCampaign.__new__(DatasetCampaign)
            campaign.bundle = bundle
            campaign.machine = {"machine_id": "same-machine"}
            campaign.manifest = manifest
            self.assertEqual(campaign._existing_context_source(), source)

    def test_repository_manifests_validate_and_round_trip(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for name in (
            "dataset_pilot.json",
            "cross_device_dataset_v1.json",
            "cross_device_dataset_v2.json",
            "cross_device_dataset_bandit_v1.json",
            "inference_states_pilot_v1.json",
        ):
            manifest = DatasetManifest.load(root / "autotune_manifests" / name)
            restored = DatasetManifest.from_dict(manifest.as_dict())
            self.assertEqual(restored, manifest)
            self.assertEqual(restored.digest, manifest.digest)
            self.assertGreater(sum(len(job.shapes) for job in manifest.jobs), 0)

    def test_every_registered_public_family_constructs_an_adapter(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifests = (
            DatasetManifest.load(
                root / "autotune_manifests" / "cross_device_dataset_v2.json"
            ),
            DatasetManifest.load(
                root / "autotune_manifests" / "inference_states_pilot_v1.json"
            ),
        )
        jobs = {
            job.family: job for manifest in manifests for job in manifest.jobs
        }
        expected = {
            "mxfp8_fused_fwd",
            "mxfp8_prequant_fwd",
            "mxfp8_weight_prequant_fwd",
            "mxfp8_fully_prequant_fwd",
            "mxfp8_bwd",
        }
        self.assertTrue(expected.issubset(jobs))
        campaign = SimpleNamespace(
            manifest=SimpleNamespace(seed=7),
            hardware_profile={},
        )
        for family in expected:
            with self.subTest(family=family):
                job = jobs[family]
                shape = job.shapes[0]
                harness = SimpleNamespace(problem=shape.problem)
                adapter = dataset_module._BACKENDS[family].make_adapter(
                    campaign, job, shape, "hot", harness, {}
                )
                self.assertIsInstance(adapter, DiscreteKernelAdapter)
                self.assertEqual(adapter.context.family, family)

    def test_normalized_export_joins_context_and_deduplicates(self) -> None:
        adapter = _adapter()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "machine.json").write_text(
                json.dumps(
                    {
                        "machine_id": "machine-one",
                        "device": {"properties": {"l2_cache_size": 50_000_000}},
                    }
                ),
                encoding="utf-8",
            )
            adapter.context = KernelContext(
                family=adapter.context.family,
                kernel_revision=adapter.context.kernel_revision,
                workload=adapter.context.workload,
                device=adapter.context.device,
                regime=adapter.context.regime,
                tags={**adapter.context.tags, "machine_id": "machine-one"},
            )
            store = JsonlTuningStore(root / "stores" / "test", fsync=False)
            session = store.start_session(adapter.context, {"seed": 7})
            observation = evaluate_proposal(
                adapter,
                Proposal(_Config(128), "cost_model", coordinate="tile", coordinate_value=128),
                session_id=session,
                sequence=0,
            )
            store.record_observation(observation)
            store.finish_session(session, {"status": "complete"})
            verification = ExperimentJournal(root / "verification.jsonl")
            verification.append(
                {
                    "record_type": "race",
                    "observation_key": "race-one",
                    "context_id": adapter.context.identifier,
                    "context": adapter.context.as_dict(),
                    "family": "test_kernel",
                    "kernel_revision": 3,
                    "incumbent_id": "tile-64",
                    "challenger_id": "tile-128",
                    "outcome": {
                        "status": "ok",
                        "decision": "challenger",
                        "paired_speedup": {"median": 0.1, "ci_low": 0.08, "ci_high": 0.12},
                    },
                }
            )
            allocations = ExperimentJournal(root / "context_allocations.jsonl")
            allocations.append(
                {
                    "record_type": "context_allocation",
                    "observation_key": "allocation-one",
                    "context_id": adapter.context.identifier,
                    "family": "test_kernel",
                    "machine_id": "machine-one",
                    "shape": {"m": 128, "n": 256, "k": 512},
                    "reward": 0.25,
                    "success": True,
                }
            )

            rows = normalized_rows((root, root))
            self.assertEqual(len(rows), 3)
            measurement = next(row for row in rows if row["record_type"] == "measurement")
            self.assertEqual(measurement["context__workload__m"], 128)
            self.assertEqual(measurement["config__tile"], 128)
            self.assertEqual(measurement["outcome__metadata__calls_per_sample"], 17)
            self.assertEqual(
                measurement["machine__device__properties__l2_cache_size"],
                50_000_000,
            )
            self.assertEqual(measurement["strategy"], "cost_model")
            allocation = next(
                row for row in rows if row["record_type"] == "context_allocation"
            )
            self.assertEqual(allocation["allocation__reward"], 0.25)

            report = export_bundle((root,), root / "merged", export_format="csv")
            self.assertEqual(report["rows"], 3)
            self.assertEqual(report["context_allocations"], 1)
            with (root / "merged.csv").open(encoding="utf-8") as source:
                exported = list(csv.DictReader(source))
            self.assertEqual(len(exported), 3)
            self.assertTrue((root / "merged.export.json").exists())

    def test_parquet_dependency_error_is_actionable(self) -> None:
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(RuntimeError, "parquet"):
                    export_parquet([], Path(directory) / "data.parquet")

    def test_manifest_rejects_unknown_family(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported kernel family"):
            DatasetManifest.from_dict(
                {
                    "name": "bad",
                    "jobs": [
                        {
                            "family": "unknown",
                            "shapes": [{"m": 128, "n": 128, "k": 128}],
                        }
                    ],
                }
            )


if __name__ == "__main__":
    unittest.main()
