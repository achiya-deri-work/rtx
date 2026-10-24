"""Compiled refinement across packed operands and NVFP4 scaling policies."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import torch
import rtx


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class PackedRefinementCudaTests(unittest.TestCase):
    def setUp(self):
        # Each test is a distinct workload/policy study. Do not accumulate
        # their specializations against one shared Python forward code object.
        # No reset occurs between a study's budgets or repeated calls.
        torch.compiler.reset()

    def _case(self, family, state, scaling='current', scale_config=None, *, require_no_eligible=False):
        from torch._inductor.compile_fx import compile_fx
        torch.manual_seed(1729)
        cls=rtx.MXFP8Linear if family=='mxfp8' else rtx.NVFP4Linear
        p=rtx.AutotunePolicy(max_trials=8,max_seconds=90,search_schedule='progressive',
            cache_behavior='refine',confirmation_repeats=0,finalist_rounds=2,
            warmup=1,samples=3,calls_per_sample=2)
        with tempfile.TemporaryDirectory() as root, torch.inference_mode():
            def count():
                return sum(len(path.read_text().splitlines()) for path in Path(root).rglob('observations.jsonl'))
            for budget in (8,16):
                kwargs={} if family=='mxfp8' else dict(scaling=scaling)
                if scale_config is not None:
                    kwargs['scale_config'] = scale_config
                layer=cls(128,128,device='cuda',tuning_policy=replace(p,max_trials=budget),
                          autotune_cache_dir=root,**kwargs).eval()
                x=torch.randn(128,128,device='cuda',dtype=torch.bfloat16)*.1
                reference=(x.float()@layer.weight.float().T).bfloat16()
                if state!='dynamic': layer=layer.to_quantized_weight()
                inputs=x
                if state=='fully_prequantized':
                    inputs=(rtx.quantize_mxfp8(x) if family=='mxfp8' else rtx.quantize_nvfp4(x))
                graphs=[]
                def backend(graph,inputs):
                    graphs.append(graph)
                    return compile_fx(graph,inputs)
                compiled=torch.compile(layer,backend=backend,fullgraph=True,dynamic=False)
                before=count(); output=compiled(inputs)
                self.assertGreater(count(),before,(family,state,scaling,budget))
                self.assertTrue(bool(output.isfinite().all()))
                error=float((output.float()-reference.float()).norm()/reference.float().norm().clamp_min(1e-12))
                print('NUMERICS',family,state,scaling,budget,error,flush=True)
                self.assertLess(error,.07 if family=='mxfp8' else .20)
                before=count(); graph_count=len(graphs); compiled(inputs)
                self.assertEqual(count(),before)
                self.assertEqual(len(graphs),graph_count)
                from cache_memory_checks import assert_stable_cuda_memory
                assert_stable_cuda_memory(self, lambda: compiled(inputs),
                    label=dict(family=family, state=state, scaling=scaling,
                               scale_config=str(scale_config), budget=budget))
                self.assertEqual(count(), before)
                self.assertEqual(len(graphs), graph_count)
                winners=list(Path(root).glob('runtime_winners/**/*.json'))
                from rtx.autotune.outcomes import TrialOutcome
                rows=[json.loads(line) for path in Path(root).rglob('observations.jsonl')
                      for line in path.read_text().splitlines()]
                self.assertTrue(rows)
                eligible=[row for row in rows if TrialOutcome.from_dict(row['outcome']).timing_eligible]
                if require_no_eligible:
                    self.assertFalse(eligible)
                if not eligible:
                    # A checked process-local fallback is the public contract
                    # under unstable measurements, not a durable winner.
                    # Numerics, reuse, refinement and live-memory checks above
                    # still apply. Never turn rejected timing into cache data.
                    self.assertFalse(winners)
                    print('NO_ELIGIBLE_WINNER',family,state,scaling,budget,len(rows),flush=True)
                    continue
                self.assertTrue(winners)
                for path in winners:
                    document = json.loads(path.read_text())
                    if scaling == 'delayed':
                        from rtx.nvfp4_inference_autotune import delayed_scale_variant
                        self.assertEqual(document['variant'], delayed_scale_variant(scale_config))
                    evidence=document['metadata']['tuning_evidence']
                    self.assertEqual(evidence['attempted_trials'],budget)
                    self.assertEqual(evidence['finalist_rounds'],2)

    def test_mxfp8_weight_packed(self): self._case('mxfp8','weight_prequantized')
    def test_mxfp8_fully_packed(self): self._case('mxfp8','fully_prequantized')
    def test_nvfp4_weight_packed(self): self._case('nvfp4','weight_prequantized')
    def test_nvfp4_fully_packed(self): self._case('nvfp4','fully_prequantized')
    def test_nvfp4_current(self): self._case('nvfp4','dynamic','current')
    def test_nvfp4_block(self): self._case('nvfp4','dynamic','block')

    def test_nvfp4_block_noisy_refinement_does_not_persist(self):
        from unittest.mock import patch
        from rtx.autotune.evaluators import CalibratedPrequantEvaluator
        original=CalibratedPrequantEvaluator.__call__
        rejected=[]
        def evaluate(evaluator,config,**kwargs):
            outcome=original(evaluator,config,**kwargs)
            if not kwargs.get('validate_only') and outcome.successful:
                outcome.metadata={**outcome.metadata,'sampling':{'collection':{'stationary':False}},
                                  'test_forced_ineligible':True}
                rejected.append(True)
            return outcome
        with patch.object(CalibratedPrequantEvaluator,'__call__',evaluate):
            self._case('nvfp4','dynamic','block',require_no_eligible=True)
        self.assertTrue(rejected)

    def test_nvfp4_jit_region(self): self._case('nvfp4','dynamic','jit_row_region')
    def test_nvfp4_delayed(self): self._case('nvfp4','dynamic','delayed')

    def test_nvfp4_delayed_custom_policies(self):
        from rtx.configs.nvfp4 import NVFP4ScaleConfig
        for policy in (NVFP4ScaleConfig('exact', 4, 'most_recent'),
                       NVFP4ScaleConfig('power2', 64, 'window_max')):
            with self.subTest(policy=policy):
                torch.compiler.reset()
                self._case('nvfp4', 'dynamic', 'delayed', policy)

    def test_delayed_initialized_deepcopy_compiled_training(self):
        import copy
        from torch._inductor.compile_fx import compile_fx
        torch.manual_seed(1730)
        source=rtx.NVFP4Linear(128,128,device='cuda',scaling='delayed',autotune='off')
        with torch.no_grad():
            for magnitude in (.1,.11,.09):
                source(torch.randn(128,128,device='cuda',dtype=torch.bfloat16)*magnitude)
        clone=copy.deepcopy(source)
        self.assertTrue(clone._delayed_initialized)
        graphs=[]
        def backend(graph,inputs):
            graphs.append(graph)
            return compile_fx(graph,inputs)
        compiled=torch.compile(clone,backend=backend,fullgraph=True,dynamic=False)
        for magnitude in (.08,.1,.09):
            x=(torch.randn(128,128,device='cuda',dtype=torch.bfloat16)*magnitude).requires_grad_()
            x_control=x.detach().clone().requires_grad_()
            dy=torch.randn_like(x)*.1
            clone.weight.grad=None; source.weight.grad=None
            actual=compiled(x); expected=source(x_control)
            actual.backward(dy); expected.backward(dy)
            self.assertEqual(len(graphs),1)
            for name in ('_x_amax_state','_weight_amax_state'):
                torch.testing.assert_close(getattr(clone,name),getattr(source,name),rtol=0,atol=0)
                self.assertNotEqual(getattr(clone,name).data_ptr(),getattr(source,name).data_ptr())
            with torch.no_grad():
                references=((x.float()@clone.weight.float().T).bfloat16(),
                            (dy.float()@clone.weight.float()).bfloat16(),
                            (dy.float().T@x.float()).bfloat16())
                for label,value,control,reference in zip(('y','dx','dw'),
                    (actual,x.grad,clone.weight.grad),(expected,x_control.grad,source.weight.grad),references):
                    self.assertTrue(bool(value.isfinite().all()),label)
                    torch.testing.assert_close(value,control,rtol=0,atol=0)
                    error=float((value.float()-reference.float()).norm()/reference.float().norm().clamp_min(1e-12))
                    self.assertLess(error,.20 if label=='y' else .07,label)
        clone.reset_delayed_scales()
        self.assertFalse(clone._delayed_initialized)
        self.assertTrue(source._delayed_initialized)

    def test_nested_delayed_training_preserves_history_without_recompile(self):
        from torch._inductor.compile_fx import compile_fx
        torch.manual_seed(1729)
        layer=rtx.NVFP4Linear(128,128,device='cuda',scaling='delayed',autotune='off')
        nested=torch.nn.Sequential(layer,torch.nn.Identity())
        graphs=[]
        def backend(graph,inputs):
            graphs.append(graph)
            return compile_fx(graph,inputs)
        compiled=torch.compile(nested,backend=backend,fullgraph=True,dynamic=False)
        expected_history=None
        for index,magnitude in enumerate((.1,.11,.09)):
            x=(torch.randn(128,128,device='cuda',dtype=torch.bfloat16)*magnitude).requires_grad_()
            dy=torch.randn_like(x)*.1
            layer.weight.grad=None
            y=compiled(x); y.backward(dy)
            with torch.no_grad():
                amax=x.float().abs().amax()
                if expected_history is None:
                    expected_history=amax.expand(16).clone()
                else:
                    expected_history=torch.cat((amax.reshape(1),expected_history[:-1]))
                torch.testing.assert_close(layer._x_amax_state,expected_history,rtol=0,atol=0)
                refs=((x.float()@layer.weight.float().T).bfloat16(),
                      (dy.float()@layer.weight.float()).bfloat16(),
                      (dy.float().T@x.float()).bfloat16())
                for label,actual,ref in zip(('y','dx','dw'),(y,x.grad,layer.weight.grad),refs):
                    self.assertTrue(bool(actual.isfinite().all()),label)
                    error=float((actual.float()-ref.float()).norm()/ref.float().norm().clamp_min(1e-12))
                    self.assertLess(error,.20 if label=='y' else .07,label)
            self.assertEqual(len(graphs),1,(index,'delayed history caused recompilation'))

    def test_delayed_tuning_does_not_advance_live_history(self):
        torch.manual_seed(1729)
        with tempfile.TemporaryDirectory() as root, torch.inference_mode():
            policy=rtx.AutotunePolicy(max_trials=8,max_seconds=90,
                search_schedule='progressive',cache_behavior='refine',
                confirmation_repeats=0,finalist_rounds=2,
                warmup=1,samples=3,calls_per_sample=2)
            tuned=rtx.NVFP4Linear(128,128,device='cuda',scaling='delayed',
                                 tuning_policy=policy,autotune_cache_dir=root)
            control=rtx.NVFP4Linear(128,128,device='cuda',scaling='delayed',autotune='off')
            control.weight.copy_(tuned.weight)
            for magnitude in (.1,.3,.05):
                x=torch.randn(128,128,device='cuda',dtype=torch.bfloat16)*magnitude
                actual=tuned(x); expected=control(x)
                self.assertTrue(bool(actual.isfinite().all()))
                for name in ('_x_amax_state','_weight_amax_state'):
                    torch.testing.assert_close(getattr(tuned,name),getattr(control,name),rtol=0,atol=0)
                error=float((actual.float()-expected.float()).norm()/expected.float().norm().clamp_min(1e-12))
                self.assertLess(error,.02)


if __name__=='__main__': unittest.main()
