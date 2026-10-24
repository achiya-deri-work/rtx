"""Real concurrent first-hit/refinement: correctness, not uncontended timing."""
import json
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import unittest

import torch

WORKER=r'''
import json,sys
from unittest.mock import patch
import torch
import rtx
import rtx.prequant_autotune as mx
import rtx.nvfp4_inference_autotune as nv
import rtx.bwd_autotune as bw
from torch._inductor.compile_fx import compile_fx

root,family,budget,seed=sys.argv[1:]
budget=int(budget)
torch.manual_seed(int(seed))
policy=rtx.AutotunePolicy(max_trials=budget,max_seconds=90,search_schedule='progressive',
    cache_behavior='refine',confirmation_repeats=0,finalist_rounds=2,
    warmup=1,samples=3,calls_per_sample=2)
cls=rtx.MXFP8Linear if family=='mxfp8' else rtx.NVFP4Linear
layer=cls(256,128,device='cuda',backend='materialized',autotune='balanced',
          tuning_policy=policy,autotune_cache_dir=root)
x=(torch.randn(128,256,device='cuda',dtype=torch.bfloat16)*.1).requires_grad_()
dy=torch.randn(128,128,device='cuda',dtype=torch.bfloat16)*.1
graphs=[]
def backend(graph,inputs):
    graphs.append(graph)
    return compile_fx(graph,inputs)
compiled=torch.compile(layer,backend=backend,fullgraph=True,dynamic=False)
forward_module=mx if family=='mxfp8' else nv
forward_name='tune_mxfp8_prequant' if family=='mxfp8' else 'tune_nvfp4_inference_state'
with patch.object(forward_module,forward_name,wraps=getattr(forward_module,forward_name)) as forward, \
     patch.object(bw,'tune_mxfp8_backward',wraps=bw.tune_mxfp8_backward) as backward:
    torch.cuda.synchronize()
    print('READY',flush=True)
    assert sys.stdin.readline().strip()=='go'
    y=compiled(x); y.backward(dy)
    first=(y.detach().clone(),x.grad.clone(),layer.weight.grad.clone())
    calls=(forward.call_count,backward.call_count); graph_count=len(graphs)
    x.grad=None;layer.weight.grad=None
    again=compiled(x);again.backward(dy)
    torch.cuda.synchronize()
    assert (forward.call_count,backward.call_count)==calls,'repeated tuning'
    assert len(graphs)==graph_count,'cache-induced recompilation'
    actual=(again,x.grad,layer.weight.grad)
    reference=((x.float()@layer.weight.float().T).bfloat16(),
               (dy.float()@layer.weight.float()).bfloat16(),
               (dy.float().T@x.float()).bfloat16())
    metrics={}
    for index,label in enumerate(('y','dx','dw')):
        torch.testing.assert_close(actual[index],first[index],rtol=0,atol=0)
        assert bool(actual[index].isfinite().all())
        difference=actual[index].float()-reference[index].float()
        rms=float(difference.square().mean().sqrt().detach())
        ref_rms=float(reference[index].float().square().mean().sqrt().detach())
        relative=rms/max(ref_rms,1e-12)
        gate=.20 if family=='nvfp4' and index==0 else .07
        assert relative<gate,(label,relative,gate)
        metrics[label]=dict(rms_error=rms,reference_rms=ref_rms,normalized_rms=relative,
                            max_abs=float(difference.abs().max().detach()),finite=True,gate=gate)
    print(json.dumps(dict(family=family,budget=budget,seed=int(seed),forward_calls=calls[0],
                         backward_calls=calls[1],graphs=graph_count,metrics=metrics)),flush=True)
'''


@unittest.skipUnless(torch.cuda.is_available() and sys.platform=='linux','CUDA and flock required')
class ConcurrentCacheCudaTests(unittest.TestCase):
    def wave(self,root,family,budget):
        workers=[]
        try:
            for seed in (1729,1730):
                workers.append(subprocess.Popen([sys.executable,'-c',WORKER,root,family,str(budget),str(seed)],
                    stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True))
            for worker in workers:
                ready,_,_=select.select([worker.stdout],[],[],60)
                self.assertTrue(ready,'worker did not reach CUDA readiness barrier')
                self.assertEqual(worker.stdout.readline().strip(),'READY')
            for worker in workers:
                worker.stdin.write('go\n');worker.stdin.flush()
            reports=[]
            for worker in workers:
                out,err=worker.communicate(timeout=180)
                self.assertEqual(worker.returncode,0,out+err)
                report=json.loads(out.strip().splitlines()[-1]);reports.append(report)
                print('CONCURRENT',json.dumps(report),flush=True)
            return reports
        finally:
            for worker in workers:
                if worker.poll() is None:worker.kill()
                worker.communicate()

    def test_dynamic_training_first_hit_refine_then_reuse(self):
        for family in ('mxfp8','nvfp4'):
            with self.subTest(family=family), tempfile.TemporaryDirectory(prefix='rtx-concurrent-cuda-') as root:
                for budget,expected in ((8,1),(16,1),(8,0)):
                    reports=self.wave(root,family,budget)
                    self.assertEqual(sum(r['forward_calls'] for r in reports),expected)
                    self.assertEqual(sum(r['backward_calls'] for r in reports),expected)
                paths=list(Path(root).rglob('observations.jsonl'))
                self.assertEqual(len(paths),2)
                for path in paths:
                    rows=[json.loads(line) for line in path.read_text().splitlines()]
                    self.assertEqual(len(rows),16)
                    # A later session may retry a numerically valid candidate
                    # whose earlier clock samples were ineligible. Such retries
                    # consume the trial budget; concurrent writers must not
                    # duplicate an already eligible candidate.
                    from rtx.autotune.outcomes import TrialOutcome
                    seen={}
                    for row in rows:
                        previous=seen.get(row['config_id'], [])
                        for old in previous:
                            outcome=TrialOutcome.from_dict(old['outcome'])
                            self.assertTrue(outcome.successful and not outcome.timing_eligible)
                            self.assertNotEqual(old['session_id'],row['session_id'])
                        seen.setdefault(row['config_id'],[]).append(row)
                for path in Path(root).glob('runtime_winners/**/*.json'):
                    evidence=json.loads(path.read_text())['metadata']['tuning_evidence']
                    self.assertEqual(evidence['attempted_trials'],16)


if __name__=='__main__':unittest.main()
