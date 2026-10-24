from dataclasses import replace
import unittest
from unittest.mock import patch

from rtx.autotune.runtime_coordinates import process_l2_policy, runtime_schedule_adapter
from rtx.autotune.adapters import make_mxfp8_prequant_adapter
from rtx.autotune.outcomes import TrialOutcome
from rtx.kernels.mxfp8 import MXFP8Problem


class RuntimeCoordinateTests(unittest.TestCase):
    def test_dual_stream_order_alias_preserves_single_stream_axis(self):
        from rtx.autotune.adapters import make_mxfp8_bwd_adapter
        measured=[]
        a=make_mxfp8_bwd_adapter(MXFP8Problem(128,128,128),
            lambda c:measured.append(c) or TrialOutcome('ok',median_ms=1),device={})
        runtime=runtime_schedule_adapter(a)
        dx=replace(a.initial_config,stream_schedule='dual_stream',execution_order='dx_first')
        dw=replace(dx,execution_order='dw_first')
        self.assertNotEqual(a.config_id(dx),a.config_id(dw))
        self.assertEqual(runtime.config_id(dx),runtime.config_id(dw))
        runtime.evaluate(dw)
        self.assertEqual(measured,[dx])
        self.assertEqual(runtime.deserialize(a.serialize(dw)),dx)
        self.assertNotEqual(runtime.config_id(replace(dx,stream_schedule='single')),
                            runtime.config_id(replace(dw,stream_schedule='single')))
        self.assertIsNotNone(runtime.rejection(replace(dx,execution_order='interleaved')))
        self.assertNotEqual(runtime.context.identifier,a.context.identifier)
        self.assertEqual(runtime_schedule_adapter(runtime).context.identifier,runtime.context.identifier)

    def test_dual_order_only_domain_spends_one_trial(self):
        from rtx.autotune.adapters import make_mxfp8_bwd_adapter
        from rtx.autotune.effort import AutotunePolicy,make_effort_autotuner
        from rtx.autotune.store import InMemoryTuningStore
        a=make_mxfp8_bwd_adapter(MXFP8Problem(128,128,128),
            lambda c:TrialOutcome('ok',median_ms=1),device={})
        a=replace(a,initial_config=replace(a.initial_config,stream_schedule='dual_stream'),
            axes={'execution_order':({'execution_order':'dx_first'},{'execution_order':'dw_first'})})
        a=runtime_schedule_adapter(a);store=InMemoryTuningStore()
        p=AutotunePolicy(max_trials=8,search_schedule='progressive',confirmation_repeats=0)
        first=make_effort_autotuner(a,store,p).tune()
        second=make_effort_autotuner(a,store,p).tune()
        self.assertEqual((first.evaluated_trials,second.evaluated_trials),(1,0))
        self.assertEqual(first.tuning_evidence['backward_coordinate_revision'],1)

    def test_search_space_identity_includes_initial_and_ordered_axes(self):
        from tests.test_autotune_coverage import adapter
        a=adapter()
        base=runtime_schedule_adapter(a)
        changes=(replace(a,axes=dict(reversed(list(a.axes.items())))),
                 replace(a,axes={**a.axes,'a.tile':(4096,)}),
                 replace(a,initial_config=a.update_fn(a.initial_config,'a.tile',4096)))
        for changed in changes:
            with self.subTest(axes=changed.axes,initial=changed.initial_config):
                self.assertNotEqual(base.context.identifier,
                                    runtime_schedule_adapter(changed).context.identifier)
        self.assertEqual(base.context.identifier,runtime_schedule_adapter(base).context.identifier)

    def test_projected_inactive_axes_have_equivalent_search_identity(self):
        a=make_mxfp8_prequant_adapter(MXFP8Problem(128,128,128),lambda c:None,
            axes={'global_l2_fetch':({'l2_fetch_granularity':None},)},device={})
        b=replace(a,axes={'global_l2_fetch':({'l2_fetch_granularity':128},)},
                  initial_config=replace(a.initial_config,l2_fetch_granularity=128))
        self.assertEqual(runtime_schedule_adapter(a).context.identifier,
                         runtime_schedule_adapter(b).context.identifier)

    def test_changed_search_space_cannot_spend_old_trials(self):
        import tempfile
        from tests.test_autotune_coverage import adapter
        from rtx.autotune.effort import AutotunePolicy, make_effort_autotuner
        from rtx.autotune.store import JsonlTuningStore
        a=adapter()
        b=replace(a,axes=dict(reversed(list(a.axes.items()))))
        a,b=runtime_schedule_adapter(a),runtime_schedule_adapter(b)
        policy=AutotunePolicy(max_trials=4,search_schedule='progressive',confirmation_repeats=0)
        with tempfile.TemporaryDirectory() as root:
            first=make_effort_autotuner(a,JsonlTuningStore(root),policy).tune()
            second=make_effort_autotuner(b,JsonlTuningStore(root),policy).tune()
            resumed=make_effort_autotuner(b,JsonlTuningStore(root),policy).tune()
            self.assertEqual((first.evaluated_trials,second.evaluated_trials,resumed.evaluated_trials),
                             (4,4,0))
            self.assertEqual(len(list(JsonlTuningStore(root).records(a.context))),4)
            self.assertEqual(len(list(JsonlTuningStore(root).records(b.context))),4)

    def test_runtime_projects_l2_but_preserves_other_coordinates_and_offline_adapter(self):
        measured=[]
        original=make_mxfp8_prequant_adapter(MXFP8Problem(128,128,128),
            lambda c: measured.append(c) or TrialOutcome('ok',median_ms=1),device={})
        candidate=replace(original.initial_config,l2_fetch_granularity=128)
        runtime=runtime_schedule_adapter(original)
        self.assertIn('global_l2_fetch',original.axes)
        self.assertNotIn('global_l2_fetch',runtime.axes)
        self.assertEqual(runtime.config_id(candidate),runtime.config_id(original.initial_config))
        self.assertNotEqual(original.config_id(candidate),original.config_id(original.initial_config))
        runtime.evaluator(candidate)
        self.assertEqual(measured,[original.initial_config])
        self.assertIsNone(runtime.deserialize(original.serialize(candidate)).l2_fetch_granularity)
        self.assertEqual(runtime.serialize(candidate)['gemm'],original.serialize(candidate)['gemm'])
        original.evaluator(candidate)
        self.assertEqual(measured[-1].l2_fetch_granularity,128)

    def test_process_policy_is_validated_and_separates_contexts(self):
        adapter=make_mxfp8_prequant_adapter(MXFP8Problem(128,128,128),lambda c:None,device={})
        with patch.dict('os.environ',{'RTX_L2_FETCH_GRANULARITY':'32'}):
            a=runtime_schedule_adapter(adapter)
        with patch.dict('os.environ',{'RTX_L2_FETCH_GRANULARITY':'128'}):
            b=runtime_schedule_adapter(adapter)
        self.assertNotEqual(a.context.identifier,b.context.identifier)
        with patch.dict('os.environ',{'RTX_L2_FETCH_GRANULARITY':'17'}):
            with self.assertRaises(ValueError): process_l2_policy()

    def test_only_inactive_coordinate_becomes_one_executable_schedule(self):
        from rtx.autotune.effort import AutotunePolicy, make_effort_autotuner
        from rtx.autotune.store import InMemoryTuningStore
        measured=[]
        original=make_mxfp8_prequant_adapter(MXFP8Problem(128,128,128),
            lambda c: measured.append(c) or TrialOutcome('ok',median_ms=1),
            axes={'global_l2_fetch': ({'l2_fetch_granularity':None},
                                      {'l2_fetch_granularity':128})}, device={})
        runtime=runtime_schedule_adapter(original)
        store=InMemoryTuningStore()
        result=make_effort_autotuner(runtime,store,AutotunePolicy(max_trials=8,
            search_schedule='progressive',confirmation_repeats=0)).tune()
        self.assertEqual(len(measured),1)
        self.assertEqual(len(list(store.records(runtime.context))),1)
        self.assertIsNone(result.config.l2_fetch_granularity)
        make_effort_autotuner(runtime,store,AutotunePolicy(max_trials=8,
            search_schedule='progressive',confirmation_repeats=0)).tune()
        self.assertEqual(len(measured),1)

    def test_singleton_real_cuda_measurement_and_resume(self):
        import tempfile
        import torch
        import rtx
        from rtx.prequant_autotune import tune_mxfp8_prequant
        from rtx.autotune.orchestrator import NoEligibleTimingError
        if not torch.cuda.is_available():
            self.skipTest('CUDA required')
        torch.manual_seed(1729)
        x=torch.randn(128,128,device='cuda',dtype=torch.bfloat16)*.1
        w=torch.randn(128,128,device='cuda',dtype=torch.bfloat16)*.02
        policy=rtx.AutotunePolicy(max_trials=8,search_schedule='progressive',
            confirmation_repeats=0,warmup=3,samples=9,calls_per_sample=8)
        axes={'global_l2_fetch': ({'l2_fetch_granularity':None},{'l2_fetch_granularity':128})}
        with tempfile.TemporaryDirectory() as root:
            # A singleton has no alternate candidate when a timing window is
            # noisy. Resume its journal, with a bounded number of attempts;
            # do not install noisy evidence or relax the stationarity gate.
            for attempt in range(3):
                try:
                    first=tune_mxfp8_prequant(x,w,policy=policy,axes=axes,cache_dir=root,progress=None)
                    break
                except NoEligibleTimingError:
                    if attempt == 2:
                        import json
                        from pathlib import Path
                        for journal in Path(root).rglob('observations.jsonl'):
                            for line in journal.read_text().splitlines():
                                record=json.loads(line)
                                print('SINGLETON_TIMING_FAILURE',record['outcome'],flush=True)
                        raise
            second=tune_mxfp8_prequant(x,w,policy=policy,axes=axes,cache_dir=root,progress=None)
            self.assertEqual(first.evaluated_trials,1)
            self.assertEqual(second.evaluated_trials,0)
            self.assertEqual(first.config,second.config)
            self.assertIsNone(first.config.l2_fetch_granularity)
            layer=rtx.MXFP8Linear(128,128,device='cuda',backend='materialized',
                                  autotune='off',prequant_config=first.config).eval()
            with torch.no_grad():
                layer.weight.copy_(w)
            with torch.inference_mode():
                y=torch.compile(layer,fullgraph=True,dynamic=False)(x)
                reference=(x.float()@w.float().T).bfloat16().float()
                self.assertTrue(bool(y.isfinite().all()))
                difference=y.float()-reference
                rms=float(difference.square().mean().sqrt())
                ref_rms=float(reference.square().mean().sqrt())
                self.assertLess(rms/max(ref_rms,1e-12),.07)
                print('SINGLETON_NUMERICS',dict(rms=rms,reference_rms=ref_rms,
                      normalized_rms=rms/max(ref_rms,1e-12),max_abs=float(difference.abs().max())),flush=True)


if __name__=='__main__':unittest.main()
