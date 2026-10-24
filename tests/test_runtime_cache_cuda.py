"""Cold first-hit tuning and cross-process cache reuse on a real device."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch


_WORKER = r'''
import json
from pathlib import Path
import sys
from unittest.mock import patch
import torch
import rtx
import rtx.prequant_autotune as mx
import rtx.nvfp4_inference_autotune as nv
from torch._inductor.compile_fx import compile_fx

root, phase = sys.argv[1:3]
budget = int(sys.argv[3]) if len(sys.argv) > 3 else 0
policy = None if not budget else rtx.AutotunePolicy(
    max_trials=budget, max_seconds=90, search_schedule='progressive',
    cache_behavior='refine', confirmation_repeats=0, finalist_rounds=2,
    warmup=1, samples=3, calls_per_sample=2,
)
torch.manual_seed(1729)
def forbidden(*args, **kwargs):
    raise AssertionError("cache hit unexpectedly entered the autotuner")
for family in ("mxfp8", "nvfp4"):
    module, name = ((mx, "tune_mxfp8_prequant") if family == "mxfp8"
                    else (nv, "tune_nvfp4_inference_state"))
    original = getattr(module, name)
    with patch.object(module, name, side_effect=forbidden if phase == "read" else original) as tuner:
        cls = rtx.MXFP8Linear if family == "mxfp8" else rtx.NVFP4Linear
        layer = cls(256, 128, device="cuda", autotune="balanced",
                    tuning_policy=policy,
                    backend="materialized", autotune_cache_dir=root).eval()
        x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        graphs = []
        def backend(graph, inputs):
            graphs.append(graph)
            return compile_fx(graph, inputs)
        compiled = torch.compile(layer, backend=backend, fullgraph=True, dynamic=False)
        with torch.inference_mode():
            reference = (x.float() @ layer.weight.float().T).bfloat16().float()
            first = compiled(x)
            calls = tuner.call_count
            graph_count = len(graphs)
            second = compiled(x)
        torch.cuda.synchronize()
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        assert bool(torch.isfinite(second).all())
        assert tuner.call_count == calls, "same-process cache miss"
        assert len(graphs) == graph_count, "cache mutation caused recompilation"
        assert calls == (1 if phase == "write" else 0), (family, phase, calls)
        delta = second.float() - reference
        rms = float(delta.square().mean().sqrt())
        ref_rms = float(reference.square().mean().sqrt())
        relative = rms / max(ref_rms, 1e-12)
        assert relative < (.07 if family == 'mxfp8' else .20), (family, relative)
        print(json.dumps(dict(family=family, phase=phase, budget=budget,
            rms_error=rms, reference_rms=ref_rms, relative_rms=relative,
            max_abs=float(delta.abs().max()), graphs=graph_count, tuner_calls=calls)), flush=True)
if budget:
    for path in Path(root).glob('runtime_winners/**/*.json'):
        evidence=json.loads(path.read_text())['metadata']['tuning_evidence']
        assert evidence['attempted_trials'] >= budget, (path, evidence)
        assert evidence['finalist_rounds'] == 2, (path, evidence)
print("PASS", phase)
'''


_TRAIN_WORKER = r'''
import json
from pathlib import Path
import sys
from unittest.mock import patch
import torch
import rtx
import rtx.prequant_autotune as mx
import rtx.nvfp4_inference_autotune as nv
import rtx.bwd_autotune as bwd
from torch._inductor.compile_fx import compile_fx

root,phase,budget=sys.argv[1:]
budget=int(budget)
policy=rtx.AutotunePolicy(max_trials=budget,max_seconds=90,
    search_schedule='progressive',cache_behavior='refine',
    confirmation_repeats=0,finalist_rounds=2,warmup=1,samples=3,calls_per_sample=2)
def forbidden(*args,**kwargs):
    raise AssertionError('fresh-process cache hit entered tuner')
for family in ('mxfp8','nvfp4'):
    torch.manual_seed(1729)
    cache=Path(root)/family
    module,name=(mx,'tune_mxfp8_prequant') if family=='mxfp8' else (nv,'tune_nvfp4_inference_state')
    original=getattr(module,name)
    with patch.object(module,name,side_effect=forbidden if phase=='read' else original) as fwd_tuner, \
         patch.object(bwd,'tune_mxfp8_backward',side_effect=forbidden if phase=='read' else bwd.tune_mxfp8_backward) as bwd_tuner:
        cls=rtx.MXFP8Linear if family=='mxfp8' else rtx.NVFP4Linear
        layer=cls(256,128,device='cuda',autotune='balanced',backend='materialized',
                  tuning_policy=policy,autotune_cache_dir=cache)
        x=(torch.randn(128,256,device='cuda',dtype=torch.bfloat16)*.1).requires_grad_()
        # Noncontiguous upstream gradient is an independent backward contract.
        dy=(torch.randn(128,128,device='cuda',dtype=torch.bfloat16)*.1).T
        assert not dy.is_contiguous()
        graphs=[]
        def backend(graph,inputs):
            graphs.append(graph)
            return compile_fx(graph,inputs)
        compiled=torch.compile(layer,backend=backend,fullgraph=True,dynamic=False)
        y=compiled(x)
        y.backward(dy)
        first=(y.detach().clone(),x.grad.clone(),layer.weight.grad.clone())
        calls=(fwd_tuner.call_count,bwd_tuner.call_count)
        assert calls==((1,1) if phase=='write' else (0,0)),(family,phase,calls)
        graph_count=len(graphs)
        records={str(path):path.read_bytes() for path in cache.rglob('observations.jsonl')}
        x.grad=None; layer.weight.grad=None
        y=compiled(x); y.backward(dy)
        assert len(graphs)==graph_count,'repeated training caused recompilation'
        assert (fwd_tuner.call_count,bwd_tuner.call_count)==calls
        assert records=={str(path):path.read_bytes() for path in cache.rglob('observations.jsonl')}
        with torch.no_grad():
            refs=((x.float()@layer.weight.float().T).bfloat16(),
                  (dy.float()@layer.weight.float()).bfloat16(),
                  (dy.float().T@x.float()).bfloat16())
            for label,actual,previous,ref in zip(('y','dx','dw'),(y,x.grad,layer.weight.grad),first,refs):
                assert bool(actual.isfinite().all()),(family,label)
                torch.testing.assert_close(actual,previous,atol=0,rtol=0)
                delta=actual.float()-ref.float()
                rms=float(delta.square().mean().sqrt())
                ref_rms=float(ref.float().square().mean().sqrt())
                relative=rms/max(ref_rms,1e-12)
                assert relative < (.20 if family=='nvfp4' and label=='y' else .07),(family,label,relative)
                print(json.dumps(dict(family=family,phase=phase,budget=budget,output=label,
                    relative_rms=relative,rms_error=rms,reference_rms=ref_rms,
                    max_abs=float(delta.abs().max()),graphs=graph_count,tuner_calls=calls)),flush=True)
        winners=[json.loads(path.read_text()) for path in cache.glob('runtime_winners/**/*.json')]
        assert len(winners)==2,(family,winners)
        for winner in winners:
            evidence=winner['metadata']['tuning_evidence']
            assert evidence['attempted_trials']>=budget,evidence
            assert evidence['finalist_rounds']==2,evidence
print('PASS',phase,budget)
'''


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12,
    "requires RTX Blackwell CUDA",
)
class RuntimeCacheCudaTests(unittest.TestCase):
    def test_exhausted_noisy_backward_history_revalidates_current_inputs(self):
        from unittest.mock import patch
        import rtx
        import rtx.bwd_autotune as bwd
        from rtx.autotune.runtime import run_runtime_tuner
        policy=rtx.AutotunePolicy(max_trials=1,max_seconds=60,
            search_schedule='progressive',confirmation_repeats=0,finalist_rounds=0,
            warmup=1,samples=3,calls_per_sample=2)
        torch.manual_seed(7721)
        x=torch.randn(128,128,device='cuda',dtype=torch.bfloat16)*.1
        w=torch.randn(128,128,device='cuda',dtype=torch.bfloat16)*.1
        dy=torch.randn(128,128,device='cuda',dtype=torch.bfloat16)*.1
        noisy={'sampling':{'collection':{'stationary':False}}}
        with tempfile.TemporaryDirectory() as root, patch.object(
                bwd,'runtime_timing_metadata',return_value=noisy) as timing:
            def run():
                return run_runtime_tuner(bwd.tune_mxfp8_backward,dy,x,w,
                    policy=policy,cache_dir=root,progress=None)
            with self.assertWarnsRegex(RuntimeWarning,'without installing'):
                first=run()
            records={str(p):p.read_bytes() for p in Path(root).rglob('observations.jsonl')}
            self.assertEqual(timing.call_count,1)
            x.mul_(2)
            with self.assertWarnsRegex(RuntimeWarning,'without installing'):
                second=run()
            self.assertEqual(first.config,second.config)
            self.assertIsNone(second.median_ms)
            self.assertEqual(timing.call_count,1,'fallback collected extra timing trials')
            self.assertEqual(records,{str(p):p.read_bytes() for p in Path(root).rglob('observations.jsonl')})
            x.fill_(float('nan'))
            with self.assertRaisesRegex(RuntimeError,'fallback validation failed'):
                run()

    def test_progressive_training_refinement_in_fresh_processes(self):
        with tempfile.TemporaryDirectory(prefix='rtx-progressive-training-process-') as root:
            phases=[(phase,budget) for budget in (8,16,32,64) for phase in ('write','read')]
            phases.append(('read',8))
            for phase,budget in phases:
                run=subprocess.run(
                    [sys.executable,'-c',_TRAIN_WORKER,root,phase,str(budget)],
                    capture_output=True,text=True,timeout=300,
                )
                print(run.stdout,flush=True)
                self.assertEqual(run.returncode,0,run.stdout+run.stderr)

    def test_progressive_refinement_in_fresh_processes(self):
        with tempfile.TemporaryDirectory(prefix='rtx-progressive-process-') as root:
            for phase,budget in (('write',8),('read',8),('write',16),('read',16),('read',8)):
                with self.subTest(phase=phase,budget=budget):
                    run=subprocess.run(
                        [sys.executable,'-c',_WORKER,root,phase,str(budget)],
                        capture_output=True,text=True,timeout=240,
                    )
                    print(run.stdout,flush=True)
                    self.assertEqual(run.returncode,0,run.stdout+run.stderr)

    def test_balanced_first_hit_and_cross_process_load(self):
        # Exercise orchestration, not search quality; production uses its normal
        # larger budget. Keep this integration test bounded on a fresh machine.
        env = dict(os.environ, RTX_BALANCED_AUTOTUNE_TRIALS="4",
                   RTX_BALANCED_AUTOTUNE_SECONDS="3")
        with tempfile.TemporaryDirectory(prefix="rtx-runtime-cache-") as root:
            for phase in ("write", "read"):
                run = subprocess.run(
                    [sys.executable, "-c", _WORKER, root, phase],
                    env=env, capture_output=True, text=True, timeout=180,
                )
                self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                self.assertGreaterEqual(
                    len(list((Path(root) / "runtime_winners").rglob("*.json"))), 2
                )


if __name__ == "__main__":
    unittest.main()
