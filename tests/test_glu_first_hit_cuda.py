from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
import rtx


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class GLUFirstHitTests(unittest.TestCase):
    def test_compiled_refinement_and_durable_reuse(self):
        from torch._inductor.compile_fx import compile_fx
        from rtx._glu_inductor import LAUNCHERS
        for output_format in ('bf16','mxfp8'):
            with self.subTest(output=output_format),tempfile.TemporaryDirectory() as root,torch.no_grad():
                torch.compiler.reset();torch.manual_seed(881)
                x=torch.randn(128,128,device='cuda',dtype=torch.bfloat16)*.1
                policy=rtx.AutotunePolicy(max_trials=8,max_seconds=90,search_schedule='progressive',
                    cache_behavior='refine',finalist_rounds=2,confirmation_repeats=0,warmup=1,samples=3,calls_per_sample=2)
                def journals():
                    return {str(p):p.read_bytes() for p in Path(root).rglob('observations.jsonl')}
                def count():return sum(len(v.splitlines()) for v in journals().values())
                for budget in (8,16):
                    graphs=[]
                    def backend(g,inputs):graphs.append(g);return compile_fx(g,inputs)
                    layer=rtx.MXFP8GLU(128,128,device='cuda',output_format=output_format,
                        packed_output=output_format=='mxfp8',autotune='balanced',
                        tuning_policy=replace(policy,max_trials=budget),autotune_cache_dir=root)
                    compiled=torch.compile(layer,backend=backend,fullgraph=True,dynamic=False)
                    y=compiled(x)
                    self.assertEqual(count(),budget)
                    p=x.float()@layer.weight.float().T
                    ref=torch.nn.functional.silu(p[:,:128])*p[:,128:]
                    if output_format=='mxfp8':
                        q,s=y
                        actual=q.float()*torch.exp2(s.view(torch.uint8).float()-127).repeat_interleave(32,dim=-1)
                    else:actual=y.float()
                    self.assertTrue(bool(actual.isfinite().all()))
                    delta=actual-ref;rms=float(delta.square().mean().sqrt());rrms=float(ref.square().mean().sqrt())
                    self.assertLess(rms/max(rrms,1e-12),.12)
                    print('GLU_FIRST_HIT',json.dumps(dict(output=output_format,budget=budget,
                        normalized_rms=rms/max(rrms,1e-12),max_abs=float(delta.abs().max()),rms=rms,reference_rms=rrms)),flush=True)
                    before=journals()
                    compiled(x)
                    self.assertEqual(journals(),before);self.assertEqual(len(graphs),1)
                    # Discard launcher runners, but retain the compiled graph.
                    # Durable winners must rebuild them without new searches.
                    for launcher in LAUNCHERS.values():launcher.runners.clear()
                    with patch('rtx.autotune.glu.tune_mxfp8_glu',side_effect=AssertionError('durable hit retuned')):
                        compiled(x)
                        eager=layer(x)
                        if output_format=='bf16':torch.testing.assert_close(eager,y,rtol=0,atol=0)
                        else:
                            for a,b in zip(eager,y):torch.testing.assert_close(a.view(torch.uint8),b.view(torch.uint8),rtol=0,atol=0)
                        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):compiled(x)
                        stream.synchronize()
                        graph=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph,stream=stream):captured=compiled(x)
                        for _ in range(3):graph.replay()
                        torch.cuda.synchronize()
                        if output_format=='bf16':torch.testing.assert_close(captured,y,rtol=0,atol=0)
                        else:
                            for a,b in zip(captured,y):torch.testing.assert_close(a.view(torch.uint8),b.view(torch.uint8),rtol=0,atol=0)
                    self.assertEqual(journals(),before);self.assertEqual(len(graphs),1)


if __name__=='__main__':unittest.main()
