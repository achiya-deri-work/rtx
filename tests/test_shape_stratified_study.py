import json
from collections import Counter
from pathlib import Path
import unittest
from tools.shape_stratified_study import tasks_for_manifest


class ShapeStudyTests(unittest.TestCase):
    def manifest(self):
        return json.loads(Path('autotune_manifests/shape_stratified_v1.json').read_text())

    def test_strata_and_family_balance_at_interruptions(self):
        m = self.manifest(); tasks = tasks_for_manifest(m)
        self.assertEqual(len(tasks), 16*8*4*4)
        self.assertEqual(len({t['id'] for t in tasks}), len(tasks))
        first = tasks[:16*4]
        self.assertEqual(len({t['category'] for t in first}), 16)
        self.assertEqual(set(Counter(t['family'] for t in first).values()), {8})
        wave = tasks[:16*8*4]
        self.assertEqual(set(Counter((t['category'],t['family']) for t in wave).values()), {4})
        self.assertEqual({t['budget'] for t in wave}, {8})
        for start in range(0, len(tasks), 4):
            group = tasks[start:start+4]
            self.assertEqual(len({t['treatment'] for t in group}), 4)
            self.assertEqual(len({(t['family'],t['category'],t['budget']) for t in group}), 1)

    def test_invalid_and_deterministic(self):
        m=self.manifest()
        self.assertEqual(tasks_for_manifest(m),tasks_for_manifest(m))
        m['budgets']=[16,8]
        with self.assertRaises(ValueError):tasks_for_manifest(m)

    def test_setup_failure_blocks_only_exact_context(self):
        import tempfile
        from tools.predictor_study import atomic, setup_quarantines, setup_context_key
        tasks=tasks_for_manifest(self.manifest())
        with tempfile.TemporaryDirectory() as tmp:
            atomic(Path(tmp)/'tasks'/'failed'/'result.json',
                   dict(task=tasks[0],status='setup_error',error='invalid reference'))
            blocked=setup_quarantines(tmp,{'quarantine_setup_failures':True})
            self.assertIn(setup_context_key(tasks[1]),blocked)
            self.assertNotIn(setup_context_key(tasks[4]),blocked)
            self.assertEqual(setup_quarantines(tmp,{}),{})

    def test_fatal_and_initial_failure_not_ordinary_search_failure(self):
        from tools.predictor_study import blocking_failure
        for error in ('FatalDeviceContextError', 'initial configuration failed: alignment'):
            self.assertTrue(blocking_failure(dict(status='search_error',error=error)))
        self.assertFalse(blocking_failure(dict(status='search_error',error='model ranking bug')))
        m=self.manifest();m['shapes'][0]['k']=0
        with self.assertRaises(ValueError):tasks_for_manifest(m)


if __name__ == '__main__':unittest.main()
