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
    def test_configuration_returning_runtime_tuner_checked_fallback(self):
        from rtx.autotune.runtime import run_runtime_config_tuner
        from rtx.autotune.orchestrator import NoEligibleTimingError
        from rtx.configs.nvfp4 import NVFP4DynamicConfig
        seed=NVFP4DynamicConfig()
        self.assertIs(run_runtime_config_tuner(lambda: seed),seed)
        def noisy():
            raise NoEligibleTimingError(seed)
        with self.assertWarnsRegex(RuntimeWarning,'correctness-checked'):
            self.assertIs(run_runtime_config_tuner(noisy),seed)
        def unchecked():
            raise NoEligibleTimingError()
        with self.assertRaises(NoEligibleTimingError):
            run_runtime_config_tuner(unchecked)

    def test_runtime_noise_fallback_is_checked_initial_and_never_persisted(self):
        from rtx.autotune.outcomes import TrialOutcome
        from rtx.autotune.runtime import run_runtime_tuner
        from rtx.autotune.winners import save_runtime_winner
        a=adapter();store=InMemoryTuningStore()
        a.evaluator=lambda c: TrialOutcome("ok",median_ms=.001,max_abs_error=0.,
            metadata={"sampling":{"collection":{"stationary":False}}})
        with self.assertWarnsRegex(RuntimeWarning,"without installing"):
            result=run_runtime_tuner(tune_first_hit,a,store,[Config(a=2)],
                policy=FirstHitPolicy(max_trials=2,confirmation_repeats=0),progress=None)
        self.assertEqual(result.config,a.initial_config)
        self.assertIsNone(result.median_ms)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(save_runtime_winner(None,{},config_id="not-a-winner",root=tmp,
                tuning_evidence=result.tuning_evidence))
            self.assertEqual(list(Path(tmp).iterdir()),[])
        self.assertEqual(retrieve_candidates(a,store.records()),[])

    def test_runtime_noise_fallback_does_not_hide_failures_or_unchecked_seed(self):
        from rtx.autotune.outcomes import TrialOutcome
        from rtx.autotune.runtime import run_runtime_tuner
        for outcome in (TrialOutcome("correctness_error",error="bad"),
                        TrialOutcome("ok",median_ms=1.,metadata={"sampling":{"collection":{"stationary":False}}})):
            a=adapter();a.evaluator=lambda c:outcome
            with self.assertRaisesRegex(RuntimeError,"no successful|initial configuration failed"):
                run_runtime_tuner(tune_first_hit,a,InMemoryTuningStore(),[],
                    policy=FirstHitPolicy(max_trials=1,confirmation_repeats=0),progress=None)

    def test_benchmark_partition_ignores_provenance_not_semantics(self):
        from rtx.autotune.first_hit_study import benchmark_contract_features
        a={"context.tags.machine_id=a":1,"context.tags.manifest_digest=x":1,
           "context.tags.source_sha256=same":1,"context.tags.precision=fp4":1}
        b={"context.tags.machine_id=b":1,"context.tags.manifest_digest=y":1,
           "context.tags.source_sha256=same":1,"context.tags.precision=fp4":1}
        self.assertEqual(benchmark_contract_features(a),benchmark_contract_features(b))
        b["context.tags.scale_policy=block"]=1
        self.assertNotEqual(benchmark_contract_features(a),benchmark_contract_features(b))
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

    def test_retrieval_budget_counts_recent_compatible_measurements(self):
        a=adapter(); store=InMemoryTuningStore()
        tune_first_hit(a,store,[Config(a=2)],policy=FirstHitPolicy(max_trials=2,confirmation_repeats=0),progress=None)
        rows=list(store.records())
        winner=next(row for row in rows if row["config"] == a.serialize(Config(a=2)))
        stale=dict(winner, family="unrelated_family")
        self.assertEqual(retrieve_candidates(a,iter([stale]*10+[winner]),max_records=1),[Config(a=2)])
        older=dict(winner,config=a.serialize(Config(b=1)),context_id="older")
        self.assertEqual(retrieve_candidates(a,iter([older,winner]),max_records=1),[Config(a=2)])

    def test_unstable_timings_do_not_win_or_cache_and_can_be_retried(self):
        from rtx.autotune.outcomes import TrialOutcome
        a=adapter();store=InMemoryTuningStore()
        noisy={"sampling":{"collection":{"stationary":False}}}
        a.evaluator=lambda c: TrialOutcome("ok",median_ms=.001,metadata=noisy)
        messages=[]
        with self.assertRaisesRegex(RuntimeError,"no successful"):
            tune_first_hit(a,store,[Config()],policy=FirstHitPolicy(max_trials=1,confirmation_repeats=0),progress=messages.append)
        self.assertTrue(any("SAVE" in message and "timing_ineligible" in message for message in messages))
        self.assertEqual(retrieve_candidates(a,store.records()),[])
        self.assertTrue(TrialOutcome("ok",median_ms=.001,metadata=noisy).successful)
        a.evaluator=lambda c: TrialOutcome("ok",median_ms=1.)
        result=tune_first_hit(a,store,[Config()],policy=FirstHitPolicy(max_trials=1,confirmation_repeats=0),progress=None)
        self.assertEqual(result.evaluated_trials,1)
        self.assertEqual(result.median_ms,1.)
        a.evaluator=lambda c:self.fail("stable cache evaluated")
        self.assertEqual(tune_first_hit(a,store,[],progress=None).median_ms,1.)

    def test_progressive_refinement_counts_nonstationary_retry_in_budget(self):
        from rtx.autotune.outcomes import TrialOutcome
        a=adapter();store=InMemoryTuningStore()
        calls={}
        def evaluate(c):
            cid=a.config_id(c)
            calls[cid]=calls.get(cid,0)+1
            stationary=c != a.initial_config or calls[cid] > 1
            return TrialOutcome("ok",median_ms=1.,metadata={
                "sampling":{"collection":{"stationary":stationary}}})
        a.evaluator=evaluate
        pool=[Config(a=2),Config(b=1)]
        for budget in (2,4):
            tune_first_hit(a,store,pool,policy=FirstHitPolicy(
                max_trials=budget,confirmation_repeats=0),refine=True,progress=None)
        rows=list(store.records())
        self.assertEqual(len(rows),4)
        self.assertEqual(len({r['config_id'] for r in rows}),3)
        self.assertEqual(calls[a.config_id(a.initial_config)],2)
        self.assertEqual(calls[a.config_id(Config(a=2))],1)

    def test_unstable_confirmation_cannot_be_hidden_by_stable_screen(self):
        from rtx.autotune.outcomes import TrialOutcome
        a=adapter();store=InMemoryTuningStore()
        calls=iter((True,False))
        a.evaluator=lambda c:TrialOutcome("ok",median_ms=1.,timings_ms=[1.]*9,
            metadata={"sampling":{"collection":{"stationary":next(calls)}}})
        with self.assertRaisesRegex(RuntimeError,"no successful"):
            tune_first_hit(a,store,[Config()],policy=FirstHitPolicy(max_trials=1,confirmation_repeats=1),progress=None)
        row=list(store.records())[0]
        self.assertFalse(row["outcome"]["metadata"]["sampling"]["collection"]["stationary"])

    def test_retrieval_exact_semantics_regime_and_toolchain(self):
        a=adapter();store=InMemoryTuningStore()
        tune_first_hit(a,store,[Config(a=2)],policy=FirstHitPolicy(max_trials=2,confirmation_repeats=0),progress=None)
        rows=list(store.records())
        for key in ("context.tags.extra_policy=packed_source", "context.environment.math=fast",
                    "context.device.torch_version=other"):
            altered=[{**r,"features":{**r["features"],key:1.}} for r in rows]
            self.assertEqual(retrieve_candidates(a,altered),[])
        altered=[]
        for row in rows:
            features={k:v for k,v in row["features"].items() if not k.startswith("context.regime=")}
            features["context.regime=rotate"]=1.
            altered.append({**row,"features":features})
        self.assertEqual(retrieve_candidates(a,altered),[])

    def test_retrieval_hardware_ties_and_unique_budget(self):
        a=adapter()
        a.context=replace(a.context,device={"multiprocessor_count":170})
        def record(identity,config,sm):
            context=replace(a.context,device={"multiprocessor_count":sm})
            return {"family":a.context.family,"kernel_revision":a.context.kernel_revision,
                "context_id":identity,"features":context.features(),"config":a.serialize(config),
                "outcome":{"status":"ok","median_ms":1.}}
        rows=[record("far",Config(a=1),40),record("near1",Config(a=2),170),
              record("near2",Config(a=2),170),record("next",Config(b=1),160)]
        self.assertEqual(retrieve_candidates(a,rows,limit=2),[Config(a=2),Config(b=1)])
        self.assertEqual(retrieve_candidates(a,rows,limit=0),[])

    def test_retrieval_portfolios_interleave_sources_under_one_cap(self):
        a=adapter()
        def row(context,cost,config):
            return {"family":a.context.family,"kernel_revision":a.context.kernel_revision,
                "context_id":context,"features":a.context.features(),"config":a.serialize(config),
                "outcome":{"status":"ok","median_ms":cost}}
        rows=[row("one",1.,Config(a=1)),row("one",2.,Config(b=1)),
              row("one",3.,Config(b=1)),row("two",10.,Config(a=2)),row("two",20.,Config(b=2))]
        self.assertEqual(retrieve_candidates(a,rows,per_context=2,limit=4),
                         [Config(a=1),Config(a=2),Config(b=1),Config(b=2)])
        self.assertEqual(retrieve_candidates(a,rows,per_context=2,limit=3),
                         [Config(a=1),Config(a=2),Config(b=1)])

    def test_retrieval_bus_width_alias_and_unknown_capacity(self):
        a=adapter()
        a.context=replace(a.context,device={"memory_bus_width_bits":512,"l2_cache_size":96})
        def row(identity,config,device):
            context=replace(a.context,device=device)
            return {"family":a.context.family,"kernel_revision":a.context.kernel_revision,
                "context_id":identity,"features":context.features(),"config":a.serialize(config),
                "outcome":{"status":"ok","median_ms":1.}}
        rows=[row("unknown",Config(a=1),{}),
              row("far",Config(b=1),{"memory_bus_width_bits":128,"l2_cache_size":32}),
              row("legacy_match",Config(a=2),{"memory_bus_width":512,"l2_cache_size":96})]
        self.assertEqual(retrieve_candidates(a,rows,limit=3),[Config(a=2),Config(b=1),Config(a=1)])

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
                unstable=evaluate_proposal(a,Proposal(Config(a=99),"anchor"),session_id=session,sequence=8)
                unstable=replace(unstable,outcome=replace(unstable.outcome,median_ms=1e-12,
                    metadata={"sampling":{"collection":{"stationary":False}}}))
                store.record_observation(unstable)
                (root/str(m)/"anchor.json").write_text(json.dumps({"context":a.context.as_dict(),
                    "contract":contract(a.context),"manifest_id":"test","task":{"id":str(m)},"portfolio_id":"test"}))
            fit([root],root/"model")
            report=json.loads((root/"model/evaluation.json").read_text())
            self.assertEqual(len(report["metrics"]),8)
            self.assertEqual(report["ingestion"]["fit_selection"]["stable_successful_rows"],16)
            self.assertEqual(report["ingestion"]["latency_view"]["excluded_nonstationary_successes"],2)
            self.assertTrue(all(r["candidates"]==8 for r in report["metrics"]))
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
            process=MagicMock();process.wait.return_value=1;process.poll.return_value=1
            with patch("rtx.autotune.first_hit_study.subprocess.Popen",return_value=process) as launch,redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError,"three consecutive"):
                    run(path,Path(tmp)/"output",1,"cuda:0")
            self.assertEqual(launch.call_count,4)
            self.assertEqual(launch.call_args.args[0][2],"rtx.autotune.first_hit_study")
            self.assertTrue(launch.call_args.kwargs["start_new_session"])


if __name__=="__main__":unittest.main()
