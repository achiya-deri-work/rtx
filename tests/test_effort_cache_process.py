"""CPU process qualification of the shared cache protocol, not GPU tuning."""
import json
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import unittest


WORKER = r'''
import json
from pathlib import Path
import sys
from rtx.autotune.effort import AutotunePolicy
from rtx.autotune.winners import (
    RuntimeWinnerKey, load_runtime_winner, runtime_tuning_lock, save_runtime_winner,
)
from rtx.kernels.mxfp8 import MXFP8Problem
root, budget, revision = sys.argv[1:4]
budget, revision = int(budget), int(revision)
policy = AutotunePolicy(max_trials=budget, search_schedule='progressive', cache_behavior='refine')
key = RuntimeWinnerKey('test', MXFP8Problem(128,128,128), 'hot', 'cpu-test', revision)
sys.stdin.readline()  # Parent releases all workers after spawning them.
with runtime_tuning_lock(key, root=root):
    if len(sys.argv) > 4:
        print('locked', flush=True)
        sys.stdin.readline()  # Test parent terminates this confirmed lock holder.
    config = load_runtime_winner(key, dict, root=root, tuning_policy=policy)
    miss = config is None
    if miss:
        with (Path(root)/'tuning_attempts.jsonl').open('a') as sink:
            sink.write(json.dumps({'budget':budget, 'revision':revision})+'\n')
        config = {'budget':budget}
        save_runtime_winner(key, config, config_id=str(budget), root=root,
            tuning_evidence=dict(schema=1, contract=policy.evidence_contract(), attempted_trials=budget))
print(json.dumps({'miss':miss, 'budget':config['budget']}))
'''


class EffortCacheProcessTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == 'linux', 'production flock contract')
    def test_process_death_releases_lock_without_publishing_winner(self):
        with tempfile.TemporaryDirectory(prefix='rtx-effort-interrupted-') as root:
            holder=subprocess.Popen(
                [sys.executable,'-c',WORKER,root,'8','1','hold'],
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,
            )
            try:
                holder.stdin.write('go\n')
                holder.stdin.flush()
                ready,_,_=select.select([holder.stdout],[],[],30)
                self.assertTrue(ready,'lock holder never reported readiness')
                self.assertEqual(holder.stdout.readline().strip(),'locked')
                holder.kill()
                holder.communicate(timeout=10)
                self.assertFalse((Path(root)/'tuning_attempts.jsonl').exists())
                rows=self.wave(root,8,count=2)
                self.assertEqual(sum(row['miss'] for row in rows),1)
            finally:
                if holder.poll() is None:
                    holder.kill()
                holder.communicate()

    def wave(self, root, budget, revision=1, count=3):
        workers=[]
        try:
            for _ in range(count):
                workers.append(subprocess.Popen(
                    [sys.executable, '-c', WORKER, root, str(budget), str(revision)],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True,
                ))
            for worker in workers:
                worker.stdin.write('go\n')
                worker.stdin.flush()
            rows=[]
            for worker in workers:
                stdout, stderr=worker.communicate(timeout=60)
                self.assertEqual(worker.returncode,0,stdout+stderr)
                rows.append(json.loads(stdout))
            return rows
        finally:
            for worker in workers:
                if worker.poll() is None:
                    worker.kill()
                worker.communicate()

    @unittest.skipUnless(sys.platform == 'linux', 'production flock contract')
    def test_concurrent_first_hit_refine_reuse_and_revision(self):
        with tempfile.TemporaryDirectory(prefix='rtx-effort-process-') as root:
            for budget, revision, expected_misses, expected_budget in (
                (8,1,1,8), (16,1,1,16), (8,1,0,16), (8,2,1,8),
            ):
                with self.subTest(budget=budget,revision=revision):
                    rows=self.wave(root,budget,revision)
                    self.assertEqual(sum(row['miss'] for row in rows),expected_misses)
                    self.assertEqual({row['budget'] for row in rows},{expected_budget})
            attempts=[json.loads(line) for line in
                      (Path(root)/'tuning_attempts.jsonl').read_text().splitlines()]
            self.assertEqual(attempts,[{'budget':8,'revision':1},
                                       {'budget':16,'revision':1},
                                       {'budget':8,'revision':2}])


if __name__ == '__main__':
    unittest.main()
