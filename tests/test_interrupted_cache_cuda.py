"""SIGKILL after real CUDA evaluation, before durable outcome publication."""
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import time
import unittest

import torch

WORKER = r'''
import json,sys
from pathlib import Path
from unittest.mock import patch
import torch,rtx
from rtx.autotune.evaluators import CalibratedPrequantEvaluator
from rtx.prequant_autotune import MXFP8PrequantEvaluator
from torch._inductor.compile_fx import compile_fx
root,family,mode=sys.argv[1:]
torch.manual_seed(1729)
policy=rtx.AutotunePolicy(max_trials=8,max_seconds=90,search_schedule='progressive',
    cache_behavior='refine',confirmation_repeats=0,finalist_rounds=2,
    warmup=1,samples=3,calls_per_sample=2)
cls=rtx.MXFP8Linear if family=='mxfp8' else rtx.NVFP4Linear
layer=cls(128,128,device='cuda',backend='materialized',tuning_policy=policy,
          autotune_cache_dir=root).eval()
x=torch.randn(128,128,device='cuda',dtype=torch.bfloat16)*.1
calls=0
evaluator_type=MXFP8PrequantEvaluator if family=='mxfp8' else CalibratedPrequantEvaluator
original=evaluator_type.__call__
def evaluate(self,config):
    global calls
    outcome=original(self,config)
    calls+=1
    if mode=='interrupt' and calls==4:
        assert outcome.status=='ok','crash boundary requires a successfully measured CUDA candidate'
        torch.cuda.synchronize()
        print('KILL_READY',flush=True)
        sys.stdin.readline()
        raise AssertionError('interrupted worker unexpectedly resumed')
    return outcome
graphs=[]
def backend(graph,inputs):
    graphs.append(graph)
    return compile_fx(graph,inputs)
compiled=torch.compile(layer,backend=backend,fullgraph=True,dynamic=False)
with torch.inference_mode(),patch.object(evaluator_type,'__call__',evaluate):
    y=compiled(x)
    reference=(x.float()@layer.weight.float().T).bfloat16()
    assert bool(y.isfinite().all())
    error=y.float()-reference.float()
    rms=float(error.square().mean().sqrt())
    ref_rms=float(reference.float().square().mean().sqrt())
    nrmse=rms/max(ref_rms,1e-12)
    assert nrmse<(.07 if family=='mxfp8' else .20)
    before=calls; graph_count=len(graphs)
    again=compiled(x)
    torch.testing.assert_close(again,y,rtol=0,atol=0)
    assert calls==before and len(graphs)==graph_count
    print('RESULT '+json.dumps(dict(family=family,calls=calls,graphs=graph_count,
        rms=rms,reference_rms=ref_rms,normalized_rms=nrmse,max_abs=float(error.abs().max()))),flush=True)
'''


@unittest.skipUnless(torch.cuda.is_available() and sys.platform == 'linux', 'CUDA/Linux required')
class InterruptedCacheCudaTests(unittest.TestCase):
    def test_killed_evaluation_consumes_budget_and_resumes(self):
        for family in ('mxfp8', 'nvfp4'):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as root:
                with tempfile.TemporaryFile(mode='w+') as errors:
                    child = subprocess.Popen([sys.executable, '-u', '-c', WORKER,
                        root, family, 'interrupt'], stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE, stderr=errors, text=True, bufsize=1)
                    try:
                        deadline = time.monotonic() + 120
                        output = b''
                        while True:
                            remaining = deadline - time.monotonic()
                            self.assertGreater(remaining, 0, 'worker did not reach crash boundary')
                            ready, _, _ = select.select([child.stdout], [], [], min(remaining, 5))
                            if not ready:
                                self.assertIsNone(child.poll(), 'worker exited before crash boundary')
                                continue
                            chunk = os.read(child.stdout.fileno(), 65536)
                            if not chunk:
                                errors.seek(0)
                                self.fail('worker exited: ' + errors.read())
                            output += chunk
                            if b'KILL_READY\n' in output:
                                break
                        paths = list(Path(root).rglob('observations.jsonl'))
                        self.assertEqual(len(paths), 1)
                        before_bytes = paths[0].read_bytes()
                        self.assertEqual(len(before_bytes.splitlines()), 3)
                        self.assertFalse(list(Path(root).glob('runtime_winners/**/*.json')))
                        child.kill()
                        child.wait(timeout=20)
                        self.assertEqual(child.returncode, -9)
                    finally:
                        if child.poll() is None:
                            child.kill()
                        child.communicate(timeout=20)
                resumed = subprocess.run([sys.executable, '-u', '-c', WORKER,
                    root, family, 'resume'], capture_output=True, text=True, timeout=150)
                self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
                after = paths[0].read_bytes()
                self.assertTrue(after.startswith(before_bytes), 'completed evidence was rewritten')
                rows = [json.loads(line) for line in after.splitlines()]
                self.assertEqual(len(rows), 7)
                self.assertEqual(len({row['config_id'] for row in rows}), 7)
                events = [json.loads(line) for p in Path(root).rglob('events.jsonl')
                          for line in p.read_text().splitlines()]
                recovered = [e for e in events if e['kind'] == 'abandoned_candidates_recovered']
                self.assertTrue(recovered)
                lost = set(recovered[-1]['payload']['config_ids'])
                self.assertEqual(len(lost), 1)
                self.assertTrue(lost.isdisjoint(row['config_id'] for row in rows))
                winners = list(Path(root).glob('runtime_winners/**/*.json'))
                self.assertEqual(len(winners), 1)
                winner = json.loads(winners[0].read_text())
                self.assertNotIn(winner['config_id'], lost)
                evidence = winner['metadata']['tuning_evidence']
                self.assertEqual(evidence['attempted_trials'], 8)
                self.assertEqual(evidence['finalist_rounds'], 2)
                print('RECOVERY ' + json.dumps(dict(family=family, completed=7,
                      abandoned=list(lost), winner=winner['config_id'])), flush=True)
                print(resumed.stdout, flush=True)


if __name__ == '__main__':
    unittest.main()
