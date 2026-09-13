"""Small end-to-end budget/API checks; not a performance benchmark."""
import tempfile,unittest
from pathlib import Path
import torch
import rtx
from unittest import mock


@unittest.skipUnless(torch.cuda.is_available(),'requires CUDA')
class EffortCudaTests(unittest.TestCase):
    def test_cold_reduce_overhead_inference(self):
        for cls in (rtx.MXFP8Linear, rtx.NVFP4Linear):
            with self.subTest(layer=cls.__name__), tempfile.TemporaryDirectory() as tmp:
                torch._dynamo.reset()
                with mock.patch.dict('os.environ', {
                    'RTX_BALANCED_AUTOTUNE_TRIALS': '2',
                    'RTX_BALANCED_AUTOTUNE_SECONDS': '60',
                }), torch.inference_mode():
                    layer = cls(128, 128, device='cuda', autotune_cache_dir=tmp)
                    compiled = torch.compile(layer, fullgraph=True, mode='reduce-overhead')
                    x = torch.randn(128, 128, device='cuda', dtype=torch.bfloat16)
                    counts = []
                    for _ in range(4):
                        torch.compiler.cudagraph_mark_step_begin()
                        out = compiled(x)
                        self.assertTrue(out.isfinite().all())
                        counts.append(sum(len(p.read_text().splitlines())
                                          for p in Path(tmp).rglob('observations.jsonl')))
                    self.assertGreater(counts[0], 0)
                    self.assertEqual(counts, [counts[0]] * 4)

    def test_cold_fullgraph_default_tunes_once_per_shape(self):
        # No eager warmup, no explicit policy, and no explicit autotune call.
        from torch._inductor.compile_fx import compile_fx
        for cls in (rtx.MXFP8Linear, rtx.NVFP4Linear):
            with self.subTest(layer=cls.__name__), tempfile.TemporaryDirectory() as tmp:
                torch._dynamo.reset()
                compilations = []

                def backend(graph, inputs):
                    compilations.append(graph)
                    return compile_fx(graph, inputs)

                with mock.patch.dict('os.environ', {
                    'RTX_AUTOTUNE': 'balanced', 'RTX_MXFP8_AUTOTUNE': 'balanced',
                    'RTX_BALANCED_AUTOTUNE_TRIALS': '2',
                    'RTX_BALANCED_AUTOTUNE_SECONDS': '60',
                    'RTX_BALANCED_AUTOTUNE_WARMUP': '1',
                    'RTX_BALANCED_AUTOTUNE_SAMPLES': '3',
                    'RTX_BALANCED_AUTOTUNE_CALLS_PER_SAMPLE': '2',
                }):
                    layer = cls(128, 128, device='cuda', autotune_cache_dir=tmp)
                    compiled = torch.compile(layer, backend=backend, fullgraph=True, dynamic=False)

                    def records():
                        return sum(len(p.read_text().splitlines())
                                   for p in Path(tmp).rglob('observations.jsonl'))

                    self.assertEqual(records(), 0)
                    for rows in (128, 256):
                        x = torch.randn(rows, 128, device='cuda', dtype=torch.bfloat16,
                                        requires_grad=True)
                        before = records()
                        y = compiled(x)
                        dy = torch.randn_like(y)
                        y.backward(dy)
                        self.assertTrue(y.isfinite().all())
                        self.assertTrue(x.grad.isfinite().all())
                        self.assertTrue(layer.weight.grad.isfinite().all())
                        after = records()
                        self.assertGreater(after, before)
                        graph_count = len(compilations)
                        dx, dw = x.grad.clone(), layer.weight.grad.clone()
                        x.grad = None
                        layer.weight.grad = None
                        repeated = compiled(x)
                        repeated.backward(dy)
                        torch.testing.assert_close(repeated, y, rtol=0, atol=0)
                        torch.testing.assert_close(x.grad, dx, rtol=0, atol=0)
                        torch.testing.assert_close(layer.weight.grad, dw, rtol=0, atol=0)
                        self.assertEqual(records(), after)
                        self.assertEqual(len(compilations), graph_count)
                        layer.weight.grad = None

    def test_real_shape_resume_eight_to_sixteen(self):
        from dataclasses import replace
        from rtx.prequant_autotune import tune_mxfp8_prequant
        torch.manual_seed(1730)
        x=torch.randn(128,128,device='cuda',dtype=torch.bfloat16)*.2
        w=torch.randn_like(x)*.02
        p=rtx.AutotunePolicy(max_trials=8,max_seconds=120,warmup=1,samples=3,calls_per_sample=2)
        with tempfile.TemporaryDirectory() as tmp:
            first=tune_mxfp8_prequant(x,w,policy=p,cache_dir=tmp,progress=None)
            self.assertEqual(first.evaluated_trials,8)
            hit=tune_mxfp8_prequant(x,w,policy=p,cache_dir=tmp,progress=None)
            self.assertEqual(hit.evaluated_trials,0)
            extended=tune_mxfp8_prequant(x,w,policy=replace(p,max_trials=16),cache_dir=tmp,progress=None)
            self.assertEqual(extended.evaluated_trials,8)
            self.assertEqual(extended.reused_trials,8)

    def test_linear_training_and_packed_inference(self):
        torch.manual_seed(1729)
        with tempfile.TemporaryDirectory() as tmp:
            for cls,kwargs in [(rtx.MXFP8Linear,{'backend':'materialized'}),
                               (rtx.NVFP4Linear,{'scaling':'block'})]:
                with self.subTest(layer=cls.__name__):
                    policy=rtx.AutotunePolicy(max_trials=2,max_seconds=60,samples=3,
                        warmup=1,calls_per_sample=2,forward_trials=2,backward_trials=2)
                    layer=cls(128,128,bias=False,device='cuda',autotune='balanced',
                        tuning_policy=policy,autotune_cache_dir=tmp,**kwargs)
                    x=(torch.randn(128,128,device='cuda',dtype=torch.bfloat16)*.2).requires_grad_()
                    dy=torch.randn_like(x)*.1
                    y=layer(x);y.backward(dy)
                    refs=(x.float()@layer.weight.float().T,
                          dy.float()@layer.weight.float(),dy.float().T@x.float())
                    for actual,ref in zip((y,x.grad,layer.weight.grad),refs):
                        self.assertTrue(actual.isfinite().all())
                        self.assertLess(float((actual.float()-ref).norm()/ref.norm().clamp_min(1e-20)),.3)
                    with torch.no_grad():
                        packed=layer.to_quantized_weight()
                        out=packed(x.detach());self.assertTrue(out.isfinite().all())
                        compiled=torch.compile(packed,fullgraph=True,dynamic=False)
                        torch.testing.assert_close(compiled(x.detach()),out,rtol=.05,atol=.1)
            self.assertTrue(list(Path(tmp).rglob('observations.jsonl')))


if __name__=='__main__':unittest.main()
