"""Real runtime workspace eviction must not evict verified tuning decisions."""
from dataclasses import fields, is_dataclass
import gc
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import weakref

import torch
import rtx
from rtx.runtime import BoundedCache


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class ShapeChurnCudaTests(unittest.TestCase):
    def test_compiled_training_eviction_reuses_durable_forward_and_backward(self):
        from contextlib import ExitStack
        from torch._inductor.compile_fx import compile_fx
        from cache_memory_checks import assert_stable_cuda_memory
        import rtx.fp8 as fp8
        import rtx.fp4 as fp4
        import rtx.fp8_bwd as bwd

        def tensor_refs(value):
            if isinstance(value, torch.Tensor):
                return [weakref.ref(value)]
            if is_dataclass(value) and not isinstance(value, type):
                return [ref for field in fields(value) for ref in tensor_refs(getattr(value, field.name))]
            if isinstance(value, (list, tuple)):
                return [ref for item in value for ref in tensor_refs(item)]
            return []

        cases=(('mxfp8',fp8._InductorPrequantLauncher,None,fp8._INDUCTOR_PREQUANT_LAUNCHERS),
               ('nvfp4',fp4._InductorNVFP4BlockLauncher,'block',fp4._INDUCTOR_BLOCK_LAUNCHERS),
               ('nvfp4',fp4._InductorNVFP4MaterializedLauncher,'current',fp4._INDUCTOR_MATERIALIZED_LAUNCHERS),
               ('nvfp4',fp4._InductorNVFP4JITRegionLauncher,'jit_row_region',fp4._INDUCTOR_JIT_REGION_LAUNCHERS),
               ('nvfp4',fp4._InductorNVFP4DelayedLauncher,'delayed',fp4._INDUCTOR_DELAYED_LAUNCHERS))
        for family,launcher_class,scaling,registry in cases:
            with self.subTest(family=family,scaling=scaling), tempfile.TemporaryDirectory() as root, ExitStack() as stack:
                torch.compiler.reset();torch.manual_seed(1729)
                caches={'forward':BoundedCache(2),'backward':BoundedCache(2),'partial_backward':BoundedCache(4)}
                original_init=launcher_class.__init__
                def initialize_launcher(instance,*args,**kwargs):
                    original_init(instance,*args,**kwargs)
                    stack.callback(setattr,instance,'runners',instance.runners)
                    instance.runners=caches['forward']
                # Earlier tests/Inductor cache hits may reuse an already
                # constructed launcher and never invoke the patched __init__.
                for instance in tuple(registry.values()):
                    stack.enter_context(patch.object(instance,'runners',caches['forward']))
                stack.enter_context(patch.object(launcher_class,'__init__',initialize_launcher))
                stack.enter_context(patch.object(bwd,'_RUNNERS',caches['backward']))
                stack.enter_context(patch.object(bwd,'_MATMUL_RUNNERS',caches['partial_backward']))
                policy=rtx.AutotunePolicy(max_trials=8,max_seconds=90,search_schedule='progressive',
                    cache_behavior='refine',confirmation_repeats=0,finalist_rounds=2,
                    warmup=1,samples=3,calls_per_sample=2)
                cls=rtx.MXFP8Linear if family=='mxfp8' else rtx.NVFP4Linear
                kwargs={} if scaling is None else {'scaling':scaling}
                layer=cls(128,128,device='cuda',backend='materialized',tuning_policy=policy,
                          autotune_cache_dir=root,**kwargs).train()
                graphs=[]
                def backend(graph, inputs):
                    graphs.append(graph)
                    return compile_fx(graph,inputs)
                compiled=torch.compile(layer,backend=backend,fullgraph=True,dynamic=False)
                inputs=[(torch.randn(m,128,device='cuda',dtype=torch.bfloat16)*.1).requires_grad_()
                        for m in (128,256,384)]
                upstream=[(torch.randn(128,m,device='cuda',dtype=torch.bfloat16)*.1).T
                          for m in (128,256,384)]
                self.assertTrue(all(not dy.is_contiguous() for dy in upstream))

                def journal_bytes():
                    return {str(p.relative_to(root)):p.read_bytes() for p in Path(root).rglob('observations.jsonl')}

                def step(i, check=True):
                    x,dy=inputs[i],upstream[i]
                    y=compiled(x);y.backward(dy)
                    if check:
                        with torch.no_grad():
                            refs=((x.float()@layer.weight.float().T).bfloat16(),
                                  (dy.float()@layer.weight.float()).bfloat16(),
                                  (dy.float().T@x.float()).bfloat16())
                            for label,actual,reference in zip(('y','dx','dw'),(y,x.grad,layer.weight.grad),refs):
                                self.assertTrue(bool(actual.isfinite().all()),label)
                                self.assertTrue(bool(reference.isfinite().all()),label)
                                diff=actual.float()-reference.float()
                                rms=float(diff.square().mean().sqrt());rrms=float(reference.float().square().mean().sqrt())
                                gate=.20 if family=='nvfp4' and label=='y' else .07
                                metrics=dict(family=family,scaling=scaling,m=x.shape[0],output=label,
                                    finite=True,rms=rms,reference_rms=rrms,max_abs=float(diff.abs().max()),
                                    normalized_rms=rms/max(rrms,1e-12),
                                    normalized_max=float(diff.abs().max())/max(float(reference.float().abs().max()),1e-12),
                                    floor=1e-12,gate=gate)
                                print('COMPILED_CHURN_NUMERICS '+json.dumps(metrics),flush=True)
                                self.assertLess(metrics['normalized_rms'],gate,label)
                    x.grad=None;layer.weight.grad=None

                step(0)
                active={name:c for name,c in caches.items() if len(c)}
                self.assertIn('forward',active)
                self.assertTrue({'backward','partial_backward'} & set(active))
                first_keys={name:tuple(c) for name,c in active.items()}
                refs={name:[ref for key in keys for ref in tensor_refs(active[name][key])]
                      for name,keys in first_keys.items()}
                self.assertTrue(all(refs.values()))
                step(1);step(2);gc.collect();torch.cuda.synchronize()
                for name,c in active.items():
                    self.assertEqual(len(c),c.max_entries,name)
                    self.assertTrue(all(key not in c for key in first_keys[name]),name)
                    self.assertTrue(all(ref() is None for ref in refs[name]),name)
                evidence=journal_bytes()
                self.assertEqual(sum(len(data.splitlines()) for data in evidence.values()),48)
                self.assertEqual(len(graphs),3)
                for target in ('rtx.prequant_autotune.tune_mxfp8_prequant',
                               'rtx.nvfp4_inference_autotune.tune_nvfp4_inference_state',
                               'rtx.bwd_autotune.tune_mxfp8_backward'):
                    stack.enter_context(patch(target,side_effect=AssertionError('eviction unexpectedly retuned')))
                step(0)
                self.assertEqual(journal_bytes(),evidence)
                self.assertEqual(len(graphs),3)
                for name,c in active.items():
                    self.assertTrue(all(key in c for key in first_keys[name]),name)
                    self.assertEqual(len(c),c.max_entries,name)
                assert_stable_cuda_memory(self,lambda:step(0,False),
                    label=dict(family=family,scaling=scaling,state='compiled_training_after_eviction'))
                self.assertEqual(journal_bytes(),evidence);self.assertEqual(len(graphs),3)
                print('COMPILED_CHURN_CACHE '+json.dumps(dict(family=family,scaling=scaling,
                    caches={name:c.stats() for name,c in active.items()},graphs=len(graphs),
                    released_tensors={name:len(values) for name,values in refs.items()})),flush=True)

    def test_materialized_runner_eviction_preserves_tuning_decisions(self):
        import rtx.fp8 as fp8
        import rtx.fp4 as fp4
        cases = (
            ('mxfp8', fp8, '_PREQUANT_RUNNERS', None),
            ('nvfp4', fp4, '_BLOCK_DYNAMIC_RUNNERS', 'block'),
            ('nvfp4', fp4, '_JIT_REGION_DYNAMIC_RUNNERS', 'jit_row_region'),
            ('nvfp4', fp4, '_DELAYED_DYNAMIC_RUNNERS', 'delayed'),
        )
        for family, module, cache_name, scaling in cases:
            with self.subTest(family=family, scaling=scaling):
                torch.manual_seed(1729)
                cache = BoundedCache(2)
                policy = rtx.AutotunePolicy(max_trials=8, max_seconds=90,
                    search_schedule='progressive', cache_behavior='refine',
                    confirmation_repeats=0, finalist_rounds=2,
                    warmup=1, samples=3, calls_per_sample=2)
                with tempfile.TemporaryDirectory() as root, \
                     patch.object(module, cache_name, cache), torch.inference_mode():
                    cls = rtx.MXFP8Linear if family == 'mxfp8' else rtx.NVFP4Linear
                    kwargs = {} if scaling is None else {'scaling': scaling}
                    layer = cls(128, 128, device='cuda', backend='materialized',
                        tuning_policy=policy, autotune_cache_dir=root, **kwargs).eval()
                    inputs = [torch.randn(m, 128, device='cuda', dtype=torch.bfloat16) * .1
                              for m in (128, 256, 384)]
                    def records():
                        return sum(len(p.read_text().splitlines())
                                   for p in Path(root).rglob('observations.jsonl'))

                    def check(x):
                        y = layer(x)
                        reference = (x.float() @ layer.weight.float().T).bfloat16()
                        self.assertTrue(bool(y.isfinite().all()))
                        error = y.float() - reference.float()
                        rms = float(error.square().mean().sqrt())
                        ref_rms = float(reference.float().square().mean().sqrt())
                        normalized = rms / max(ref_rms, 1e-12)
                        self.assertLess(normalized, .07 if family == 'mxfp8' else .20)
                        print('CHURN_NUMERICS ' + json.dumps(dict(
                            family=family, scaling=scaling, m=x.shape[0], rms=rms,
                            reference_rms=ref_rms, normalized_rms=normalized,
                            max_abs=float(error.abs().max()))), flush=True)
                        torch.cuda.synchronize()

                    check(inputs[0])
                    self.assertEqual(len(cache), 1)
                    first_key = next(iter(cache))
                    runner = cache[first_key]
                    tensor_refs = [weakref.ref(getattr(runner, field.name))
                                   for field in fields(runner)
                                   if isinstance(getattr(runner, field.name), torch.Tensor)]
                    self.assertTrue(tensor_refs)
                    del runner
                    check(inputs[1])
                    check(inputs[2])
                    gc.collect()
                    torch.cuda.synchronize()
                    self.assertEqual(len(cache), 2)
                    self.assertNotIn(first_key, tuple(cache))
                    self.assertTrue(all(ref() is None for ref in tensor_refs),
                                    'evicted runner workspace tensors remain reachable')
                    observations = records()
                    self.assertEqual(observations, 24)
                    with patch('rtx.prequant_autotune.tune_mxfp8_prequant',
                               side_effect=AssertionError('unexpected MXFP8 retuning')), \
                         patch('rtx.nvfp4_inference_autotune.tune_nvfp4_inference_state',
                               side_effect=AssertionError('unexpected NVFP4 retuning')):
                        check(inputs[0])
                    self.assertEqual(records(), observations)
                    self.assertIn(first_key, tuple(cache))
                    self.assertEqual(len(cache), 2)
                    self.assertEqual(cache.evictions, 2)
                    print('CHURN_CACHE ' + json.dumps(dict(family=family, scaling=scaling,
                          stats=cache.stats(), observations=observations,
                          released_tensors=len(tensor_refs))), flush=True)


if __name__ == '__main__':
    unittest.main()
