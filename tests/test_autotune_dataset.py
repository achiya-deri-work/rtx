from __future__ import annotations

from collections import Counter
import csv
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import zipfile

from rtx.autotune import (
    DiscountedArmStatistics,
    DiscreteKernelAdapter,
    JsonlTuningStore,
    KernelContext,
    ResidualTuningStore,
    TrialOutcome,
    contextual_ucb_scores,
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
from rtx.autotune.dataset import decomposed_machine_identities
from rtx.autotune import dataset as dataset_module
from rtx.autotune.optimizer_benchmark import (
    optimizer_study_rows,
    summarize_optimizer_study,
)
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
    def test_machine_identities_separate_device_compiler_calibration_and_source(self) -> None:
        device = {
            "fingerprint": {
                "uuid": "gpu-1",
                "name": "RTX test",
                "capability": [12, 0],
                "total_memory": 16,
                "multiprocessor_count": 70,
            },
            "hardware_profile": {
                "architecture": {"key": "sm120"},
                "sku": {"sku_family": "test", "memory_bus_width_bits": 256},
                "properties": {"l2_cache_size": 1024},
                "compiler": {"nvcc": "13.2"},
                "calibration": {"dram": 1.0},
                "software": {"torch": "2.13"},
            },
        }
        base = decomposed_machine_identities(
            device,
            {"python_source_sha256": "one"},
            hostname="host",
            platform_name="linux",
            python_version="3.13",
        )
        compiler_changed = json.loads(json.dumps(device))
        compiler_changed["hardware_profile"]["compiler"]["nvcc"] = "13.3"
        changed = decomposed_machine_identities(
            compiler_changed,
            {"python_source_sha256": "one"},
            hostname="host",
            platform_name="linux",
            python_version="3.13",
        )
        self.assertNotEqual(base["compiler_id"], changed["compiler_id"])
        for key in base.keys() - {"compiler_id"}:
            self.assertEqual(base[key], changed[key])
        source_changed = decomposed_machine_identities(
            device,
            {"python_source_sha256": "two"},
            hostname="host",
            platform_name="linux",
            python_version="3.13",
        )
        self.assertNotEqual(base["kernel_source_id"], source_changed["kernel_source_id"])
        for key in base.keys() - {"kernel_source_id"}:
            self.assertEqual(base[key], source_changed[key])

    def test_anytime_policy_and_duration_parsing(self) -> None:
        self.assertEqual(dataset_module._parse_duration("2h"), 7200.0)
        self.assertEqual(dataset_module._parse_duration("90m"), 5400.0)
        self.assertEqual(dataset_module._parse_duration("15"), 15.0)
        self.assertEqual(dataset_module._parse_milestones("8,32,96"), (8, 32, 96))
        with self.assertRaises(ValueError):
            AnytimeRunPolicy(10.0, trial_milestones=(32, 16))
        with self.assertRaises(ValueError):
            AnytimeRunPolicy(10.0, context_bandit_discount=0.0)

    def test_cli_reserves_exit_75_for_worker_restart(self) -> None:
        self.assertEqual(dataset_module.FATAL_DEVICE_CONTEXT_EXIT_CODE, 75)

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
            key: DiscountedArmStatistics() for key in descriptors
        }
        arms["a"].update(0.8, 1.0, True)
        arms["c"].update(-0.2, 1.0, False)
        scores = contextual_ucb_scores(
            tuple(descriptors),
            arms,
            lambda left, right: dataset_module._context_similarity(
                descriptors[left], descriptors[right]
            ),
            0.1,
        )
        self.assertEqual(scores["b"], float("inf"))
        arms["b"].update(0.0, 1.0, True)
        scores = contextual_ucb_scores(
            tuple(descriptors),
            arms,
            lambda left, right: dataset_module._context_similarity(
                descriptors[left], descriptors[right]
            ),
            0.1,
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
                json.dumps(
                    {
                        "machine_id": "old-runner-machine-id",
                        "source": source,
                        "device": {"fingerprint_id": "same-device"},
                        "identities": {"calibration_id": "same-calibration"},
                    }
                ),
                encoding="utf-8",
            )
            (bundle / "manifest.json").write_text(
                json.dumps(manifest.as_dict()), encoding="utf-8"
            )
            campaign = DatasetCampaign.__new__(DatasetCampaign)
            campaign.bundle = bundle
            campaign.machine = {
                "machine_id": "new-runner-machine-id",
                "device": {"fingerprint_id": "same-device"},
                "identities": {"calibration_id": "same-calibration"},
            }
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
            "autotuner_prospective_5070_v1.json",
            "mxfp8_bwd_revision19_calibration_v1.json",
            "blackwell_diversity_atlas_v1.json",
            "blackwell_prospective_v3.json",
        ):
            manifest = DatasetManifest.load(root / "autotune_manifests" / name)
            restored = DatasetManifest.from_dict(manifest.as_dict())
            self.assertEqual(restored, manifest)
            self.assertEqual(restored.digest, manifest.digest)
            self.assertGreater(sum(len(job.shapes) for job in manifest.jobs), 0)

    def test_manifest_expands_shared_shape_sets_and_job_defaults(self) -> None:
        manifest = DatasetManifest.from_dict(
            {
                "schema_version": 2,
                "name": "composed",
                "shape_sets": {
                    "atlas": [
                        {"m": 32, "n": 256, "k": 129, "name": "ragged"},
                        {"m": 512, "n": 768, "k": 1024, "name": "dense"},
                    ]
                },
                "job_defaults": {
                    "shape_set": "atlas",
                    "regimes": ["hot"],
                    "promote": 2,
                    "tuning": {"max_trials": 12, "model_warmup": 4},
                    "protocol": {"samples": 3},
                    "tags": {"study": "coverage"},
                },
                "jobs": [
                    {"family": "mxfp8_fused_fwd"},
                    {
                        "family": "nvfp4_dynamic_fwd",
                        "tuning": {"max_trials": 8},
                        "tags": {"precision": "nvfp4"},
                    },
                ],
            }
        )
        self.assertEqual(len(manifest.jobs), 2)
        self.assertEqual(len(manifest.jobs[0].shapes), 2)
        self.assertEqual(manifest.jobs[0].tuning.max_trials, 12)
        self.assertEqual(manifest.jobs[1].tuning.max_trials, 8)
        self.assertEqual(manifest.jobs[1].tuning.model_warmup, 4)
        self.assertEqual(manifest.jobs[1].protocol.samples, 3)
        self.assertEqual(
            manifest.jobs[1].tags,
            {"study": "coverage", "precision": "nvfp4"},
        )
        self.assertEqual(DatasetManifest.from_dict(manifest.as_dict()), manifest)

        inline = DatasetManifest.from_dict(
            {
                "name": "inline-override",
                "shape_sets": {"atlas": [{"m": 32, "n": 32, "k": 32}]},
                "job_defaults": {"shape_set": "atlas", "regimes": ["hot"]},
                "jobs": [
                    {
                        "family": "mxfp8_fused_fwd",
                        "shapes": [{"m": 48, "n": 80, "k": 112}],
                    }
                ],
            }
        )
        shape = inline.jobs[0].shapes[0]
        self.assertEqual((shape.m, shape.n, shape.k), (48, 80, 112))

        bad = {
            "name": "bad-shape-set",
            "shape_sets": {"atlas": [{"m": 32, "n": 32, "k": 32}]},
            "jobs": [
                {
                    "family": "mxfp8_fused_fwd",
                    "shape_set": "atlas",
                    "shapes": [{"m": 64, "n": 64, "k": 64}],
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "both shapes and shape_set"):
            DatasetManifest.from_dict(bad)

    def test_posthoc_verification_rechecks_current_residual_finalists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            (bundle / "manifest.json").write_text("{}", encoding="utf-8")
            campaign = DatasetCampaign.__new__(DatasetCampaign)
            campaign.bundle = bundle
            campaign.manifest = SimpleNamespace(digest="manifest-digest")
            campaign.machine = {"machine_id": "machine-one"}
            campaign.bundle_machine_id = "machine-one"
            campaign.progress = None
            job = DatasetManifest.from_dict(
                {
                    "name": "verify-test",
                    "jobs": [
                        {
                            "family": "mxfp8_bwd",
                            "shapes": [{"m": 128, "n": 128, "k": 128}],
                            "promote": 2,
                        }
                    ],
                }
            ).jobs[0]
            shape = job.shapes[0]
            adapter = SimpleNamespace(
                context=SimpleNamespace(
                    identifier="context-one", kernel_revision=19
                )
            )
            harness = object()
            store = object()
            campaign._assigned_contexts = lambda: [(job, shape, "hot")]
            campaign._make_harness = Mock(return_value=harness)
            campaign._make_adapter = Mock(return_value=adapter)
            campaign._store = Mock(return_value=store)
            campaign._context_trials = Mock(return_value=17)
            campaign._verify = Mock(
                return_value={
                    "status": "ok",
                    "screened": 17,
                    "winner_id": "winner",
                }
            )

            summary_path = campaign.verify_existing(promote=4)

            verified_job = campaign._verify.call_args.args[2]
            self.assertEqual(verified_job.promote, 4)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["contexts"], 1)
            self.assertEqual(summary["verified"], 1)
            self.assertEqual(
                summary["results"][0]["verification"]["winner_id"], "winner"
            )

    def test_verification_measurements_are_resumable_completed_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            journal = ExperimentJournal(Path(directory) / "verification.jsonl")
            for record_type, key in (
                ("verification_measurement", "confirm-one"),
                ("race", "race-one"),
            ):
                journal.append(
                    {
                        "record_type": record_type,
                        "observation_key": key,
                    }
                )
            self.assertEqual(
                journal.completed_keys(), {"confirm-one", "race-one"}
            )

    def test_prospective_matrix_rotates_balanced_residual_units(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest = DatasetManifest.load(
            root / "autotune_manifests" / "autotuner_prospective_5070_v1.json"
        )
        campaign = DatasetCampaign.__new__(DatasetCampaign)
        campaign.manifest = manifest
        campaign.anytime = AnytimeRunPolicy(60.0)
        contexts = campaign._assigned_contexts()
        self.assertEqual(len(contexts), 144)
        domains = (
            {job.tags["treatment"] for job, _shape, _regime in contexts},
            {job.family for job, _shape, _regime in contexts},
            {shape.name for _job, shape, _regime in contexts},
            {regime for _job, _shape, regime in contexts},
            {job.tags["replicate"] for job, _shape, _regime in contexts},
        )
        for prefix_size in (12, 24, 36, 72, 144):
            prefix = contexts[:prefix_size]
            dimensions = (
                [job.tags["treatment"] for job, _shape, _regime in prefix],
                [job.family for job, _shape, _regime in prefix],
                [shape.name for _job, shape, _regime in prefix],
                [regime for _job, _shape, regime in prefix],
                [job.tags["replicate"] for job, _shape, _regime in prefix],
            )
            for values, domain in zip(dimensions, domains):
                counts = Counter(values)
                loads = [counts[value] for value in domain]
                self.assertLessEqual(max(loads) - min(loads), 1)
        self.assertEqual(manifest.storage_mode, "residual_context")
        self.assertEqual(manifest.rotation_mode, "balanced_categories")

    def test_residual_store_isolates_corrupt_tail_and_reads_siblings(self) -> None:
        first_adapter = _adapter()
        second_adapter = replace(
            first_adapter,
            context=replace(
                first_adapter.context,
                workload={"m": 256, "n": 256, "k": 512},
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            family_root = Path(directory) / "residuals" / "test_kernel"
            first = ResidualTuningStore(
                family_root / "unit-a", transfer_root=family_root, fsync=False
            )
            second = ResidualTuningStore(
                family_root / "unit-b", transfer_root=family_root, fsync=False
            )
            first.record_observation(
                evaluate_proposal(
                    first_adapter,
                    Proposal(_Config(64), "random"),
                    session_id="first",
                    sequence=0,
                )
            )
            second.record_observation(
                evaluate_proposal(
                    second_adapter,
                    Proposal(_Config(128), "random"),
                    session_id="second",
                    sequence=0,
                )
            )
            with first.local.observations_path.open("a", encoding="utf-8") as sink:
                sink.write('{"truncated":')
            self.assertEqual(len(list(first.records())), 2)
            self.assertEqual(
                len(list(first.records(second_adapter.context))), 1
            )
            self.assertNotEqual(first.path, second.path)

            first.record_observation(
                evaluate_proposal(
                    first_adapter,
                    Proposal(_Config(128), "random"),
                    session_id="resumed",
                    sequence=1,
                )
            )
            self.assertEqual(len(list(first.records(first_adapter.context))), 2)

    def test_optimizer_study_reports_prospective_regret(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for treatment, timings in (
                ("random", (2.0, 1.8, 1.7, 1.6)),
                ("online_bandit", (1.7, 1.4, 1.2, 1.0)),
            ):
                adapter = _adapter()
                adapter.context = replace(
                    adapter.context,
                    tags={
                        **adapter.context.tags,
                        "machine_id": "gpu-one",
                        "treatment": treatment,
                        "replicate": 0,
                        "task_category": "balanced",
                    },
                )
                store = JsonlTuningStore(root / treatment, fsync=False)
                session = store.start_session(adapter.context, {"seed": 1})
                for sequence, timing in enumerate(timings):
                    observation = evaluate_proposal(
                        adapter,
                        Proposal(
                            _Config(64 if sequence < 2 else 128), treatment
                        ),
                        session_id=session,
                        sequence=sequence,
                    )
                    observation.outcome.median_ms = timing
                    observation.outcome.timings_ms = [timing]
                    store.record_observation(observation)

            rows = optimizer_study_rows((root,))
            self.assertEqual(len(rows), 2)
            by_treatment = {str(row["treatment"]): row for row in rows}
            self.assertEqual(by_treatment["online_bandit"]["final_regret"], 0.0)
            self.assertAlmostEqual(
                float(by_treatment["random"]["final_regret"]), 0.6
            )
            report = summarize_optimizer_study(
                (root,), root / "report" / "optimizers", export_format="csv"
            )
            self.assertEqual(report["units"], 2)
            self.assertEqual(report["schema_version"], 3)
            self.assertIn("mean_compile_failure_rate", report["aggregates"][0])
            self.assertIn(
                "ci_low_median_final_regret", report["aggregates"][0]
            )
            self.assertEqual(len(report["matched_comparisons"]), 1)
            self.assertTrue(report["shape_aggregates"])
            comparison = report["matched_comparisons"][0]
            self.assertEqual(comparison["treatment"], "online_bandit")
            self.assertLess(comparison["median_delta_final"], 0)
            self.assertEqual(
                comparison["probability_beating_random_final"], 1.0
            )
            self.assertTrue((root / "report" / "optimizers.csv").exists())

    def test_every_registered_public_family_constructs_an_adapter(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifests = (
            DatasetManifest.load(
                root / "autotune_manifests" / "cross_device_dataset_v2.json"
            ),
            DatasetManifest.load(
                root / "autotune_manifests" / "inference_states_pilot_v1.json"
            ),
            DatasetManifest.load(
                root / "autotune_manifests" / "nvfp4_inference_states_v1.json"
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
            "nvfp4_weight_prequant_fwd",
            "nvfp4_fully_prequant_fwd",
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

            archive_path = root / "dataset-export-test.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                for path in root.rglob("*.json*"):
                    archive.write(path, f"bundle/{path.relative_to(root)}")
            zipped_rows = normalized_rows((archive_path,))
            self.assertEqual(len(zipped_rows), 3)

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
