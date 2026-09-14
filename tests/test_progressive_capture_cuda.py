"""Capture/replay after first-hit tuning; not a replay performance benchmark."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import rtx


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class ProgressiveCaptureCudaTests(unittest.TestCase):
    def setUp(self):
        torch.compiler.reset()

    def _case(self,family,m,n,k,scaling=None):
        import rtx.prequant_autotune as mx
        import rtx.nvfp4_inference_autotune as nv
        import rtx.bwd_autotune as bwd
        from torch._inductor.compile_fx import compile_fx
        torch.manual_seed(1729)
        policy=rtx.AutotunePolicy(max_trials=8,max_seconds=90,
            search_schedule='progressive',cache_behavior='refine',
            confirmation_repeats=0,finalist_rounds=2,warmup=1,samples=3,calls_per_sample=2)
        with tempfile.TemporaryDirectory(prefix='rtx-progressive-capture-') as root:
            cls=rtx.MXFP8Linear if family=='mxfp8' else rtx.NVFP4Linear
            layer=cls(k,n,device='cuda',backend='materialized',tuning_policy=policy,
                      autotune_cache_dir=root,**({} if scaling is None else {'scaling':scaling}))
            x=(torch.randn(m,k,device='cuda',dtype=torch.bfloat16)*.1).requires_grad_()
            dy=torch.randn(m,n,device='cuda',dtype=torch.bfloat16)*.1
            graphs=[]
            def backend(graph,inputs):
                graphs.append(graph)
                return compile_fx(graph,inputs)
            compiled=torch.compile(layer,backend=backend,fullgraph=True,dynamic=False)
            stream=torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    x.grad=None; layer.weight.grad=None
                    compiled(x).backward(dy)
            stream.synchronize()
            if scaling == 'delayed':
                histories=[layer._x_amax_state.clone(),layer._weight_amax_state.clone()]
            graph_count=len(graphs)
            def snapshots():
                return {str(p):p.read_bytes() for p in Path(root).rglob('*.json*')}
            before=snapshots()
            self.assertTrue(list(Path(root).rglob('observations.jsonl')))
            def forbidden(*args,**kwargs):
                raise AssertionError('tuner entered during capture/replay')
            with patch.object(mx,'tune_mxfp8_prequant',side_effect=forbidden), \
                 patch.object(nv,'tune_nvfp4_inference_state',side_effect=forbidden), \
                 patch.object(bwd,'tune_mxfp8_backward',side_effect=forbidden):
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):
                    x.grad.zero_(); layer.weight.grad.zero_()
                    output=compiled(x)
                    output.backward(dy)
                # Exceed the 16-entry delayed window to test full turnover.
                magnitudes=(.1,.11,.09)*7 if scaling == 'delayed' else (.1,.3,.05)
                for index,magnitude in enumerate(magnitudes):
                    with torch.no_grad():
                        x.copy_(torch.randn_like(x)*magnitude)
                        layer.weight.copy_(torch.randn_like(layer.weight)*.02)
                        dy.copy_(torch.randn_like(dy)*.1)
                    graph.replay()
                    torch.cuda.synchronize()
                    with torch.no_grad():
                        if scaling == 'delayed':
                            for slot,(source,actual_history) in enumerate((
                                (x,layer._x_amax_state),(layer.weight,layer._weight_amax_state),
                            )):
                                amax=source.float().abs().amax().reshape(1)
                                histories[slot]=torch.cat((amax,histories[slot][:-1]))
                                torch.testing.assert_close(actual_history,histories[slot],rtol=0,atol=0,
                                    msg=f'delayed history slot={slot} replay={index}')
                        refs=((x.float()@layer.weight.float().T).bfloat16(),
                              (dy.float()@layer.weight.float()).bfloat16(),
                              (dy.float().T@x.float()).bfloat16())
                        for label,actual,reference in zip(('y','dx','dw'),(output,x.grad,layer.weight.grad),refs):
                            self.assertTrue(bool(actual.isfinite().all()),(family,label))
                            delta=actual.float()-reference.float()
                            rms=float(delta.square().mean().sqrt())
                            ref_rms=float(reference.float().square().mean().sqrt())
                            error=rms/max(ref_rms,1e-12)
                            print(json.dumps(dict(family=family,shape=(m,n,k),replay=index,
                                output=label,relative_rms=error,rms_error=rms,reference_rms=ref_rms,
                                max_abs=float(delta.abs().max()))),flush=True)
                            self.assertLess(error,.20 if family=='nvfp4' and label=='y' else .07)
                self.assertEqual(len(graphs),graph_count)
                self.assertEqual(snapshots(),before)

    def test_mxfp8_aligned(self): self._case('mxfp8',128,128,256)
    def test_mxfp8_ragged(self): self._case('mxfp8',257,131,129)
    def test_nvfp4_aligned(self): self._case('nvfp4',128,128,256)
    def test_nvfp4_ragged(self): self._case('nvfp4',257,131,129)
    def test_nvfp4_delayed_history_replay(self): self._case('nvfp4',128,128,256,'delayed')


if __name__=='__main__': unittest.main()
