"""Real compiled first-hit, paired selection and increased-effort refinement."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import torch
import rtx


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class ProgressiveCacheCudaTests(unittest.TestCase):
    def setUp(self):
        # Independent cases may reset Dynamo; a case's effort refinements may not.
        torch.compiler.reset()

    def test_compiled_training_refines_once_and_preserves_numerics(self):
        self._run_case(128,128,128)

    def test_graph_measured_backward_compiled_refinement(self):
        self._run_case(257,129,131,policy_overrides=dict(backward_measurement_mode='cuda_graph'))

    def test_ranked_opening_compiled_training_refinement(self):
        artifact=Path('autotune_reports/transport_prior_20260914_v1/predictor.json.gz')
        if not artifact.is_file():
            self.skipTest('generated research interaction prior required')
        self._run_case(512,256,384,policy_overrides=dict(
            ranked_opening_trials=2,predictor_artifact=str(artifact.resolve()),
            predictor_heads=('trees',),allow_research_prior=True))

    def test_ragged_compiled_training_refinement(self):
        self._run_case(257,129,131)

    def test_larger_compiled_training_refinement(self):
        self._run_case(1024,256,384)

    def test_long_reduction_mxfp8_full_effort_ladder(self):
        self._run_case(8192,256,128,budgets=(8,16,32,64))

    def test_long_reduction_nvfp4_full_effort_ladder(self):
        for scaling in ('block','current','jit_row_region','delayed'):
            with self.subTest(scaling=scaling):
                torch.compiler.reset()
                self._run_case(8192,256,128,family='nvfp4',scaling=scaling,
                               budgets=(8,16,32,64))

    def _run_case(self,m,k,n,*,family='mxfp8',scaling=None,budgets=(8,16),policy_overrides=None):
        from torch._inductor.compile_fx import compile_fx
        torch.manual_seed(1729)
        p=rtx.AutotunePolicy(max_trials=4,max_seconds=90,search_schedule='progressive',
            cache_behavior='refine',confirmation_repeats=0,finalist_rounds=2,
            warmup=1,samples=3,calls_per_sample=2)
        if policy_overrides:
            p=replace(p,**policy_overrides)
        with tempfile.TemporaryDirectory() as root:
            def count():
                return sum(len(path.read_text().splitlines()) for path in Path(root).rglob('observations.jsonl'))
            for budget in budgets:
                # Hold numerical inputs fixed across effort levels.
                torch.manual_seed(1729)
                graphs=[]
                def backend(graph,inputs):
                    graphs.append(graph)
                    return compile_fx(graph,inputs)
                cls=rtx.MXFP8Linear if family=='mxfp8' else rtx.NVFP4Linear
                kwargs={} if family=='mxfp8' else dict(scaling=scaling)
                layer=cls(k,n,device='cuda',tuning_policy=replace(p,max_trials=budget),
                          autotune_cache_dir=root,**kwargs)
                compiled=torch.compile(layer,backend=backend,fullgraph=True,dynamic=False)
                x=(torch.randn(m,k,device='cuda',dtype=torch.bfloat16)*.1).requires_grad_()
                dy=(torch.randn(n,m,device='cuda',dtype=torch.bfloat16)*.1).T
                self.assertFalse(dy.is_contiguous())
                before=count()
                y=compiled(x); y.backward(dy)
                self.assertGreater(count(),before)
                if policy_overrides and policy_overrides.get('ranked_opening_trials'):
                    forward_rows=[json.loads(line) for path in Path(root).rglob('observations.jsonl')
                                  for line in path.read_text().splitlines()
                                  if json.loads(line)['family']=='mxfp8_prequant_fwd']
                    self.assertEqual([r['strategy'] for r in forward_rows[:7]],
                        ['initial']+['initial_neighborhood']*2+['retrieval']*2+['portable_trees']*2)
                reference=(x.float()@layer.weight.float().T).bfloat16()
                dx=(dy.float()@layer.weight.float()).bfloat16()
                dw=(dy.float().T@x.float()).bfloat16()
                for label,actual,expected in (('y',y,reference),('dx',x.grad,dx),('dw',layer.weight.grad,dw)):
                    self.assertTrue(bool(actual.isfinite().all()),label)
                    self.assertTrue(bool(expected.isfinite().all()),label)
                    with torch.no_grad():
                        delta=actual.float()-expected.float()
                        rms=float(delta.square().mean().sqrt())
                        ref_rms=float(expected.float().square().mean().sqrt())
                        error=rms/max(ref_rms,1e-12)
                        gate=.20 if family=='nvfp4' and label=='y' else .07
                        print('NUMERICS',json.dumps(dict(family=family,scaling=scaling,
                            shape=[m,n,k],budget=budget,output=label,seed=1729,
                            max_abs=float(delta.abs().max()),rms=rms,reference_rms=ref_rms,
                            normalized_rms=error,normalized_max=float(delta.abs().max())/
                            max(float(expected.float().abs().max()),1e-12),floor=1e-12,
                            gate=gate,finite=True)),flush=True)
                    self.assertLess(error,gate,label)
                records=count(); graph_count=len(graphs)
                layer.weight.grad=None; x.grad=None
                compiled(x).backward(dy)
                self.assertEqual(count(),records)
                self.assertEqual(len(graphs),graph_count)
                from cache_memory_checks import assert_stable_cuda_memory
                def step():
                    compiled(x).backward(dy)
                    layer.weight.grad = None
                    x.grad = None
                assert_stable_cuda_memory(self, step,
                    label=dict(family=family,scaling=scaling,state='training',shape=[m,n,k],budget=budget))
                self.assertEqual(count(), records)
                self.assertEqual(len(graphs), graph_count)
            winners=list(Path(root).glob('runtime_winners/**/*.json'))
            self.assertTrue(winners)
            for path in winners:
                evidence=json.loads(path.read_text())['metadata']['tuning_evidence']
                self.assertEqual(evidence['attempted_trials'],budgets[-1])
                self.assertEqual(evidence['finalist_rounds'],2)


if __name__=='__main__': unittest.main()
