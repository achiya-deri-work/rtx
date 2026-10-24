"""Runtime compatibility and requested effort are independent cache contracts."""
from dataclasses import replace
import json
import tempfile
import unittest
from rtx.autotune.effort import AutotunePolicy
from rtx.autotune.winners import RuntimeWinnerKey,load_runtime_winner,save_runtime_winner,list_runtime_winners
from rtx.kernels.mxfp8 import MXFP8Problem


class EffortCacheTests(unittest.TestCase):
    def test_backward_refinement_rejects_old_duplicate_coordinate_budget(self):
        key=RuntimeWinnerKey('mxfp8_bwd',MXFP8Problem(128,128,128),'hot','test',23)
        policy=AutotunePolicy(max_trials=8,cache_behavior='refine')
        evidence=dict(schema=1,contract=policy.evidence_contract(),attempted_trials=8)
        with tempfile.TemporaryDirectory() as root:
            save_runtime_winner(key,{'value':1},config_id='one',root=root,tuning_evidence=evidence)
            self.assertIsNone(load_runtime_winner(key,dict,root=root,tuning_policy=policy))
            self.assertIsNotNone(load_runtime_winner(key,dict,root=root,
                tuning_policy=replace(policy,cache_behavior='reuse')))
            evidence['backward_coordinate_revision']=1
            save_runtime_winner(key,{'value':1},config_id='one',root=root,tuning_evidence=evidence)
            self.assertIsNotNone(load_runtime_winner(key,dict,root=root,tuning_policy=policy))

    def test_balanced_refine_does_not_accept_legacy_effort(self):
        import torch
        from contextlib import nullcontext
        from unittest.mock import patch
        from rtx.fp8 import _resolve_fwd_config
        x=torch.zeros(128,128,dtype=torch.bfloat16)
        with patch('rtx.autotune.winners.runtime_winner_key',return_value=object()), \
             patch('rtx.autotune.winners.load_runtime_winner',return_value=None), \
             patch('rtx.autotune.winners.runtime_tuning_lock',return_value=nullcontext()), \
             patch('rtx.autotune.load_cached_mxfp8_fwd_config') as legacy, \
             patch('rtx.autotune.tune_mxfp8_fwd',side_effect=RuntimeError('refinement reached')):
            with self.assertRaisesRegex(RuntimeError,'refinement reached'):
                _resolve_fwd_config(x,x,config=None,autotune='balanced',
                    tuning_policy=AutotunePolicy(cache_behavior='refine'),cache_dir=None)
        legacy.assert_not_called()

    def test_backward_cache_only_refine_uses_legacy(self):
        import torch
        from types import SimpleNamespace
        from unittest.mock import patch
        import rtx.fp8_bwd as backward
        x=torch.zeros(128,128,dtype=torch.bfloat16)
        problem=MXFP8Problem(128,128,128)
        request=SimpleNamespace(mode='cache',policy=AutotunePolicy(cache_behavior='refine'),cache_dir=None)
        with patch.dict(backward._AUTOTUNE_REQUESTS,{'test-cache':request},clear=True), \
             patch.dict(backward._AUTOTUNE_SELECTIONS,{},clear=True), \
             patch.object(backward,'_check_bwd_inputs',return_value=problem), \
             patch('torch.cuda.get_device_capability',return_value=(12,0)), \
             patch('rtx.autotune.winners.runtime_winner_key',return_value=object()), \
             patch('rtx.autotune.winners.load_runtime_winner',return_value=None), \
             patch('rtx.bwd_autotune.load_cached_mxfp8_bwd_config',return_value=backward.DEFAULT_MXFP8_BWD_CONFIG) as legacy:
            result=backward._resolve_bwd_context(x,x,x,'test-cache')
        legacy.assert_called_once()
        self.assertEqual(result[2],backward.DEFAULT_MXFP8_BWD_CONFIG)

    def test_delayed_cache_only_refine_preserves_fallback(self):
        import torch
        from types import SimpleNamespace
        from unittest.mock import patch
        import rtx.fp4 as fp4
        from rtx.configs.nvfp4 import NVFP4Problem, NVFP4DynamicConfig
        request=SimpleNamespace(mode='cache',policy=AutotunePolicy(cache_behavior='refine'),cache_dir=None,dynamic=None)
        with patch.dict(fp4._AUTOTUNE_REQUESTS,{'test-cache':request},clear=True), \
             patch.dict(fp4._DYNAMIC_CONFIG_SELECTIONS,{},clear=True), \
             patch('rtx.autotune.winners.runtime_winner_key',return_value=object()), \
             patch('rtx.autotune.winners.load_runtime_winner',side_effect=[None,NVFP4DynamicConfig()]) as load:
            fp4._materialized_dynamic_config_key(NVFP4Problem(128,128,128),torch.device('cpu'),
                                                'test-cache',family='nvfp4_delayed_fwd')
        self.assertEqual(load.call_count,2)

    def test_cache_only_refine_policy_still_uses_compatible_legacy_winner(self):
        import torch
        from unittest.mock import patch
        from rtx.fp8 import _resolve_fwd_config, DEFAULT_MXFP8_FWD_CONFIG
        x=torch.zeros(128,128,dtype=torch.bfloat16)
        cached=replace(DEFAULT_MXFP8_FWD_CONFIG, grid_swizzle=2)
        policy=AutotunePolicy(max_trials=16,cache_behavior='refine')
        with patch('rtx.autotune.winners.runtime_winner_key',return_value=object()), \
             patch('rtx.autotune.winners.load_runtime_winner',return_value=None), \
             patch('rtx.autotune.load_cached_mxfp8_fwd_config',return_value=cached) as legacy, \
             patch('rtx.autotune.tune_mxfp8_fwd',side_effect=AssertionError('cache-only tuned')):
            actual=_resolve_fwd_config(x,x,config=None,autotune='cache',tuning_policy=policy,cache_dir=None)
        legacy.assert_called_once()
        self.assertIs(actual,cached)

    def test_ragged_materialized_initial_and_reference_are_legal(self):
        import torch
        from unittest.mock import patch
        from rtx.prequant_autotune import tune_mxfp8_prequant, MXFP8PrequantEvaluator
        from rtx.fp8 import _PREQUANT_CONFIGS
        x=torch.zeros(257,129,dtype=torch.bfloat16)
        w=torch.zeros(131,129,dtype=torch.bfloat16)
        policy=AutotunePolicy(max_trials=8)
        with patch('rtx.autotune.effort.tune_legacy_effort') as tune:
            tune_mxfp8_prequant(x,w,policy=policy)
        initial=tune.call_args.kwargs['initial']
        self.assertIsNone(initial.rejection(MXFP8Problem(257,131,129)))
        evaluator=MXFP8PrequantEvaluator(x,w,policy.coordinate())
        def runner(x,w,out): out.zero_()
        with patch('rtx.prequant_autotune._build_prequant_runner',return_value=runner) as build, \
             patch('torch.cuda.synchronize'):
            evaluator._reference()
        self.assertEqual(_PREQUANT_CONFIGS[build.call_args.args[2]],initial)

    def test_corrupt_documents_fail_closed(self):
        key=RuntimeWinnerKey('test',MXFP8Problem(128,128,128),'hot','test',1)
        policy=AutotunePolicy(max_trials=8,cache_behavior='refine')
        with tempfile.TemporaryDirectory() as root:
            path=save_runtime_winner(key,{'value':1},config_id='one',root=root)
            original=json.loads(path.read_text())
            for document in (None, [], 4, 'bad', True):
                with self.subTest(document=document):
                    path.write_text(json.dumps(document))
                    self.assertIsNone(load_runtime_winner(key,dict,root=root))
                    self.assertFalse(list_runtime_winners(root=root)[0]['valid'])
            for metadata in (None, [], 4, {'tuning_evidence':None},
                             {'tuning_evidence':{'schema':1,'contract':policy.evidence_contract(),
                                                 'attempted_trials':True}}):
                with self.subTest(metadata=metadata):
                    path.write_text(json.dumps({**original,'metadata':metadata}))
                    self.assertIsNone(load_runtime_winner(key,dict,root=root,tuning_policy=policy))

    def test_inspection_verdict_cannot_be_overridden_by_document(self):
        key=RuntimeWinnerKey('test',MXFP8Problem(128,128,128),'hot','test',1)
        with tempfile.TemporaryDirectory() as root:
            path=save_runtime_winner(key,{'value':1},config_id='one',root=root)
            document=json.loads(path.read_text())
            document.update(schema_version=1,valid=True,compatibility='compatible',path='incorrect')
            path.write_text(json.dumps(document))
            row=list_runtime_winners(root=root)[0]
            self.assertFalse(row['valid'])
            self.assertEqual(row['compatibility'],'schema_v1_invalidated')
            self.assertEqual(row['path'],str(path))

    def test_ragged_backward_reference_uses_a_legal_baseline(self):
        import torch
        from unittest.mock import patch
        from rtx.bwd_autotune import MXFP8BwdEvaluator
        from rtx.fp8_bwd import _default_bwd_config
        x=torch.zeros(257,129,dtype=torch.bfloat16)
        w=torch.zeros(131,129,dtype=torch.bfloat16)
        dy=torch.zeros(257,131,dtype=torch.bfloat16)
        evaluator=MXFP8BwdEvaluator(dy,x,w,AutotunePolicy().coordinate())
        def runner(dy,x,w,dx,dw): dx.zero_(); dw.zero_()
        with patch('rtx.bwd_autotune._build_bwd_runner',return_value=runner) as build, patch('torch.cuda.synchronize'):
            evaluator.reference()
        selected=build.call_args.args[1]
        self.assertEqual(selected,_default_bwd_config(evaluator.problem))
        self.assertIsNone(selected.implementation_rejection(evaluator.problem))

    def test_reuse_refinement_budget_and_confirmation(self):
        from rtx.autotune.paired_promotion import PROMOTION_REVISION
        key=RuntimeWinnerKey('test',MXFP8Problem(128,128,128),'hot','test',1)
        p=AutotunePolicy(max_trials=8,cache_behavior='refine')
        with tempfile.TemporaryDirectory() as root:
            def save(evidence=None):
                save_runtime_winner(key,{'value':1},config_id='one',root=root,tuning_evidence=evidence)
            def load(policy):
                return load_runtime_winner(key,dict,root=root,tuning_policy=policy)
            save()
            self.assertIsNotNone(load(replace(p,cache_behavior='reuse')))
            self.assertIsNone(load(p))
            evidence=dict(schema=1,contract=p.evidence_contract(),attempted_trials=8,
                          finalist_rounds=4,finalist_margin=.005,promotion_revision=PROMOTION_REVISION)
            save(evidence)
            self.assertIsNotNone(load(p))
            self.assertIsNone(load(replace(p,max_trials=16)))
            self.assertIsNotNone(load(replace(p,finalist_rounds=4)))
            self.assertIsNone(load(replace(p,finalist_rounds=6)))
            self.assertIsNone(load(replace(p,seed=1)))
            self.assertIsNone(load(replace(p,samples=9)))
            self.assertEqual(p.evidence_contract(),replace(p,max_seconds=100,max_trials=16).evidence_contract())
            evidence['promotion_revision']=2
            save(evidence)
            self.assertIsNone(load(p))
            self.assertIsNone(load(replace(p,cache_behavior='reuse')))
            evidence.pop('promotion_revision')
            save(evidence)
            self.assertIsNone(load(p))
            self.assertIsNone(load(replace(p,cache_behavior='reuse')))

    def test_backward_budget_is_used(self):
        key=RuntimeWinnerKey('test',MXFP8Problem(128,128,128),'hot','test',1)
        p=AutotunePolicy(max_trials=8,backward_trials=16,cache_behavior='refine')
        with tempfile.TemporaryDirectory() as root:
            save_runtime_winner(key,{'value':1},config_id='one',root=root,
                tuning_evidence=dict(schema=1,contract=p.evidence_contract(),attempted_trials=8))
            self.assertIsNotNone(load_runtime_winner(key,dict,root=root,tuning_policy=p))
            self.assertIsNone(load_runtime_winner(key,dict,root=root,tuning_policy=p,backward=True))


if __name__=='__main__': unittest.main()
