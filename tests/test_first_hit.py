from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,MagicMock
from contextlib import redirect_stdout
import io

from tests.test_autotune_coverage import adapter,Config
from rtx.autotune.first_hit import (
    FirstHitPolicy,FrozenRanker,bounded_candidates,tune_first_hit,model_features,retrieve_candidates,
)
from rtx.autotune.store import InMemoryTuningStore
from rtx.autotune.first_hit_study import study_tasks,grouped_key


class FirstHitTests(unittest.TestCase):
    def test_bounded_dedup_and_legality(self):
        a=adapter()
        self.assertEqual(bounded_candidates(a,[Config(),Config(),Config(a=3),Config(a=2)]),
                         [Config(),Config(a=2)])

    def test_first_hit_then_cache_does_not_evaluate(self):
        a=adapter();store=InMemoryTuningStore()
        result=tune_first_hit(a,store,[Config(a=1),Config(b=2)],
            policy=FirstHitPolicy(max_trials=3,confirmation_repeats=0),progress=None)
        self.assertEqual(result.evaluated_trials,3)
        a.evaluator=lambda c: self.fail("cache hit evaluated a candidate")
        cached=tune_first_hit(a,store,[],progress=None)
        self.assertEqual(cached.evaluated_trials,0)
        self.assertEqual(cached.config,result.config)

    def test_invalid_artifact_not_silently_deployed(self):
        with self.assertRaisesRegex(ValueError,"qualification"):
            FrozenRanker({"schema_version":1,"deployment_enabled":False})

    def test_no_post_compile_feature_leak(self):
        values=model_features({"derived.compiled.registers":100,"derived.spill_bytes":128,
                               "context.workload.m":512,"config.tile_m":64})
        self.assertNotIn("derived.compiled.registers",values)
        self.assertNotIn("derived.spill_bytes",values)
        self.assertEqual(values["config.tile_m"],64)

    def test_retrieval_revalidates(self):
        a=adapter();store=InMemoryTuningStore()
        tune_first_hit(a,store,[Config(a=2)],policy=FirstHitPolicy(max_trials=2,confirmation_repeats=0),progress=None)
        found=retrieve_candidates(a,store.records())
        self.assertEqual(found,[Config(a=2)])

    def test_manifest_balanced_order(self):
        manifest=json.loads(Path("autotune_manifests/blackwell_first_hit_anchors_v1.json").read_text())
        tasks=study_tasks(manifest)
        self.assertEqual(len(tasks),32)
        self.assertEqual(len({t["id"] for t in tasks}),32)
        self.assertEqual({t["precision"] for t in tasks[:4]},{"mxfp8","bf16_gate"})
        self.assertEqual({t["phase"] for t in tasks[:4]},{"forward","train"})

    def test_shape_split_includes_hidden_and_output(self):
        from types import SimpleNamespace
        a=SimpleNamespace(features={"context.workload.m":128,"context.workload.k":128,"context.workload.h":256})
        b=SimpleNamespace(features={**a.features,"context.workload.h":512})
        self.assertNotEqual(grouped_key(a,"shape"),grouped_key(b,"shape"))

    def test_offline_fit_export_and_deployment_gate(self):
        from rtx.autotune.core import KernelContext,Proposal,evaluate_proposal
        from rtx.autotune.store import JsonlTuningStore
        from rtx.autotune.first_hit import contract
        from rtx.autotune.first_hit_study import fit,export
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for m in (128,256):
                a=adapter();a.context=KernelContext("toy",1,{"m":m,"phase":"train"})
                a.__post_init__();a.rejection_fn=lambda c:None
                store=JsonlTuningStore(root/str(m))
                session=store.start_session(a.context,{})
                for index in range(8):
                    store.record_observation(evaluate_proposal(a,Proposal(Config(a=index),"anchor"),session_id=session,sequence=index))
                (root/str(m)/"anchor.json").write_text(json.dumps({"context":a.context.as_dict(),
                    "contract":contract(a.context),"manifest_id":"test","task":{"id":str(m)},"portfolio_id":"test"}))
            fit([root],root/"model")
            report=json.loads((root/"model/evaluation.json").read_text())
            self.assertEqual(len(report["metrics"]),8)
            artifact=next((root/"model").glob("*-ranker.json"))
            with self.assertRaises(ValueError):FrozenRanker.load(artifact)
            ranker=FrozenRanker.load(artifact,allow_unqualified=True)
            self.assertTrue(ranker.compatible(a.context))
            export([root],root/"export")
            self.assertTrue((root/"export/observations.csv").exists())
            self.assertTrue((root/"export/observations.parquet").exists())

    def test_supervisor_command_and_fail_fast(self):
        from rtx.autotune.first_hit_study import run
        manifest=json.loads(Path("autotune_manifests/blackwell_first_hit_anchors_v1.json").read_text())
        manifest["shapes"]=manifest["shapes"][:1];manifest["milestones"]=[4]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"manifest.json";path.write_text(json.dumps(manifest))
            process=MagicMock();process.wait.return_value=1
            with patch("rtx.autotune.first_hit_study.subprocess.Popen",return_value=process) as launch,redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError,"three consecutive"):
                    run(path,Path(tmp)/"output",1,"cuda:0")
            self.assertEqual(launch.call_count,3)
            self.assertEqual(launch.call_args.args[0][2],"rtx.autotune.first_hit_study")
            self.assertTrue(launch.call_args.kwargs["start_new_session"])


if __name__=="__main__":unittest.main()
