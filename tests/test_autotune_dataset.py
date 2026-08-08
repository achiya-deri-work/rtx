from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import tempfile
import unittest

from rtx.autotune import (
    DiscreteKernelAdapter,
    JsonlTuningStore,
    KernelContext,
    TrialOutcome,
)
from rtx.autotune.core import Proposal, evaluate_proposal
from rtx.autotune.dataset import (
    DatasetManifest,
    export_bundle,
    export_parquet,
    normalized_rows,
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
    def test_repository_manifests_validate_and_round_trip(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for name in ("dataset_pilot.json", "cross_device_dataset_v1.json"):
            manifest = DatasetManifest.load(root / "autotune_manifests" / name)
            restored = DatasetManifest.from_dict(manifest.as_dict())
            self.assertEqual(restored, manifest)
            self.assertEqual(restored.digest, manifest.digest)
            self.assertGreater(sum(len(job.shapes) for job in manifest.jobs), 0)

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

            rows = normalized_rows((root, root))
            self.assertEqual(len(rows), 2)
            measurement = next(row for row in rows if row["record_type"] == "measurement")
            self.assertEqual(measurement["context__workload__m"], 128)
            self.assertEqual(measurement["config__tile"], 128)
            self.assertEqual(measurement["outcome__metadata__calls_per_sample"], 17)
            self.assertEqual(
                measurement["machine__device__properties__l2_cache_size"],
                50_000_000,
            )
            self.assertEqual(measurement["strategy"], "cost_model")

            report = export_bundle((root,), root / "merged", export_format="csv")
            self.assertEqual(report["rows"], 2)
            with (root / "merged.csv").open(encoding="utf-8") as source:
                exported = list(csv.DictReader(source))
            self.assertEqual(len(exported), 2)
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
