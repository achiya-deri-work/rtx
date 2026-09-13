import json
from pathlib import Path
import tempfile
import unittest

from tools.coverage_gpu_study import metrics, rows, summarize
from tools.predictor_study import atomic


class CoverageGpuStudyTests(unittest.TestCase):
    def test_metrics_zero_and_nonfinite(self):
        import torch
        z=torch.zeros(4)
        self.assertEqual(metrics(z,z)['normalized_rms'],0)
        self.assertFalse(metrics(torch.full((4,),float('nan')),z)['finite'])
        json.dumps(metrics(torch.full((4,),float('inf')),z),allow_nan=False)

    def test_strict_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'observations.jsonl';p.write_text('{"a":1}\n')
            self.assertEqual(list(rows([tmp])),[{'a':1}])
            p.write_text('broken\n')
            with self.assertRaises(json.JSONDecodeError):list(rows([tmp]))

    def test_bad_candidates_do_not_enter_performance_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            atomic(p/'tasks/a/result.json',dict(task=dict(kind='panel',sources=['structural']),status='correctness_error'))
            atomic(p/'tasks/b/result.json',dict(task=dict(kind='panel',sources=['local_walk']),status='ok',
                outcome=dict(median_ms=2),verification=[dict(median_ms=1),dict(median_ms=3)]))
            summarize(p);r=json.loads((p/'summary.json').read_text())
            self.assertNotIn('structural',r['panel'])
            self.assertEqual(r['panel']['local_walk']['best_ms'],2)
            self.assertFalse(r['automatically_reinstated'])


if __name__=='__main__':unittest.main()
