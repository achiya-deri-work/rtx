"""Execution mode is measurement identity, not a interchangeable timing trick."""
from dataclasses import replace
import unittest
import torch
from rtx.autotune.effort import AutotunePolicy
from rtx.fp8_bwd import _bwd_measurement_variant


class BackwardMeasurementModeTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_compiler_requested_graph_replays_joint_backward(self):
        import tempfile
        from unittest.mock import patch
        import rtx
        import rtx.fp8_bwd as bwd
        torch.compiler.reset();torch.manual_seed(4013)
        calls=0
        original=bwd._InductorMXFP8BwdPackedLauncher.__call__
        def launch(instance,*args,**kwargs):
            nonlocal calls
            calls+=1
            return original(instance,*args,**kwargs)
        with tempfile.TemporaryDirectory() as root, patch.object(bwd._InductorMXFP8BwdPackedLauncher,'__call__',launch):
            layer=rtx.MXFP8Linear(128,128,device='cuda',backend='materialized',autotune='balanced',
                tuning_policy=AutotunePolicy(max_trials=8,max_seconds=90,search_schedule='progressive',
                    cache_behavior='refine',backward_measurement_mode='cuda_graph',confirmation_repeats=0,
                    warmup=1,samples=3,calls_per_sample=3),autotune_cache_dir=root)
            x=(torch.randn(128,128,device='cuda',dtype=torch.bfloat16)*.1).requires_grad_()
            dy=torch.randn_like(x)*.1
            compiled=torch.compile(layer,fullgraph=True,dynamic=False,options={'triton.cudagraphs':True})
            counts=[]
            for step in range(8):
                torch.compiler.cudagraph_mark_step_begin()
                x.grad=None;layer.weight.grad=None
                y=compiled(x);y.backward(dy)
                torch.cuda.synchronize()
                from tools.research.matched_gpu import independent_metrics
                independent_metrics(x.grad,(dy.float()@layer.weight.float()).bfloat16(),.07)
                independent_metrics(layer.weight.grad,(dy.float().T@x.float()).bfloat16(),.07)
                counts.append(calls)
                del y
            self.assertGreater(counts[0],0)
            self.assertEqual(counts[3:], [counts[3]]*5,counts)

    def test_contract_and_winner_variant_are_separate(self):
        host=AutotunePolicy(search_schedule='progressive')
        graph=replace(host,backward_measurement_mode='cuda_graph')
        self.assertNotEqual(host.evidence_contract(),graph.evidence_contract())
        self.assertEqual(_bwd_measurement_variant(host),'default')
        self.assertNotEqual(_bwd_measurement_variant(host),_bwd_measurement_variant(graph))
        self.assertEqual(graph.evidence_contract(),replace(graph,max_trials=64).evidence_contract())
        self.assertEqual(graph.for_pass().evidence_contract(),host.for_pass().evidence_contract())
        self.assertEqual(graph.for_pass(True).backward_measurement_mode,'cuda_graph')
        with self.assertRaises(ValueError):replace(host,backward_measurement_mode='automatic')

    def test_auto_lowering_request_is_self_contained(self):
        from unittest.mock import patch
        from torch._inductor import config
        import rtx.fp8_bwd as bwd
        p=AutotunePolicy(backward_measurement_mode='auto')
        self.assertEqual(p.for_pass(True).backward_measurement_mode,'host')
        key=bwd._intern_bwd_autotune_request('balanced',p,'/tmp/rtx-mode-test')
        for enabled,expected in ((False,'host'),(True,'cuda_graph')):
            with config.patch({'triton.cudagraphs':enabled}):
                encoded=bwd._compiled_bwd_measurement_key(key)
            with patch.dict(bwd._AUTOTUNE_REQUESTS,{},clear=True):
                decoded=bwd._bwd_request_from_key(encoded)
                self.assertEqual(decoded.policy.backward_measurement_mode,expected)
                self.assertEqual(decoded.cache_dir,'/tmp/rtx-mode-test')
                self.assertEqual(decoded.policy.predictor_heads,p.predictor_heads)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_fresh_process_modes_share_forward_not_backward(self):
        self._fresh_process_modes(('host','cuda_graph','host','cuda_graph'))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_auto_fresh_process_modes_and_inductor_cache(self):
        self._fresh_process_modes(('auto_host','auto_graph','auto_host','auto_graph'))

    def _fresh_process_modes(self,modes):
        import json
        from pathlib import Path
        import subprocess
        import sys
        import tempfile
        worker=r'''
import sys
from unittest.mock import patch
from contextlib import ExitStack
import torch,rtx
root,mode,cached=sys.argv[1:]
torch.manual_seed(9151)
p=rtx.AutotunePolicy(max_trials=8,max_seconds=90,search_schedule='progressive',
    cache_behavior='refine',backward_measurement_mode='auto' if mode.startswith('auto_') else mode,confirmation_repeats=0,
    finalist_rounds=2,warmup=1,samples=3,calls_per_sample=3)
layer=rtx.MXFP8Linear(128,128,device='cuda',backend='materialized',
    autotune='balanced',tuning_policy=p,autotune_cache_dir=root)
x=(torch.randn(128,128,device='cuda',dtype=torch.bfloat16)*.1).requires_grad_()
dy=torch.randn_like(x)*.1
f=torch.compile(layer,fullgraph=True,dynamic=False,options={'triton.cudagraphs':mode=='auto_graph'})
with ExitStack() as stack:
    if cached=='yes':
        for target in ('rtx.prequant_autotune.tune_mxfp8_prequant','rtx.bwd_autotune.tune_mxfp8_backward'):
            stack.enter_context(patch(target,side_effect=AssertionError('fresh-process cache hit retuned')))
    y=f(x);y.backward(dy)
    from tools.research.matched_gpu import independent_metrics
    for actual,ref in ((y,(x.float()@layer.weight.float().T).bfloat16()),
        (x.grad,(dy.float()@layer.weight.float()).bfloat16()),
        (layer.weight.grad,(dy.float().T@x.float()).bfloat16())):
        independent_metrics(actual,ref,.07)
    if cached=='yes':
        from torch._dynamo.utils import counters
        assert counters['inductor']['fxgraph_cache_hit']+counters['aot_autograd']['autograd_cache_hit']>0,dict(counters)
'''
        with tempfile.TemporaryDirectory() as root:
            def inventory():
                return {str(p):p.read_bytes() for p in Path(root).rglob('observations.jsonl')}
            for index,mode in enumerate(modes):
                before=inventory()
                result=subprocess.run([sys.executable,'-c',worker,root,mode,'yes' if index>=2 else 'no'],
                    capture_output=True,text=True,timeout=180)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                after=inventory()
                if index>=2:self.assertEqual(before,after)
                else:
                    self.assertEqual(sum(len(data.splitlines()) for data in after.values()),16+index*8)
            rows=[json.loads(line) for data in inventory().values() for line in data.splitlines()]
            backward=[row for row in rows if row['family']=='mxfp8_bwd']
            self.assertEqual({row['outcome']['metadata']['measurement_mode'] for row in backward},{'host','cuda_graph'})

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_graph_evaluator_preserves_gradients_and_reports_capture_cost(self):
        from rtx.bwd_autotune import MXFP8BwdEvaluator
        from rtx.fp8_bwd import _default_bwd_config
        from rtx.kernels.mxfp8 import MXFP8Problem
        from tools.research.matched_gpu import independent_metrics
        torch.manual_seed(1937)
        x=torch.randn(257,129,device='cuda',dtype=torch.bfloat16)*.1
        w=torch.randn(131,129,device='cuda',dtype=torch.bfloat16)*.02
        dy=torch.randn(257,131,device='cuda',dtype=torch.bfloat16)*.1
        policy=AutotunePolicy(samples=3,calls_per_sample=4).coordinate(rtol=.07,atol=1.)
        config=_default_bwd_config(MXFP8Problem(257,131,129))
        for mode in ('host','cuda_graph'):
            evaluator=MXFP8BwdEvaluator(dy,x,w,policy,measurement_mode=mode)
            result=evaluator(config)
            self.assertEqual(result.status,'ok',result.error)
            self.assertEqual(result.metadata['measurement_mode'],mode)
            self.assertEqual(len(result.timings_ms),3)
            self.assertEqual(result.metadata['graph_capture_ms']>0,mode=='cuda_graph')
            # Independent numerical gate for the same candidate's reference
            # path, in addition to replay-versus-reference inside evaluate.
            dx,dw=evaluator.reference()
            independent_metrics(dx,(dy.float()@w.float()).bfloat16(),.07)
            independent_metrics(dw,(dy.float().T@x.float()).bfloat16(),.07)


if __name__=='__main__':unittest.main()
