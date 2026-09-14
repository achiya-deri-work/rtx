import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.research.matched_audit import audit, valid_confirmation, TREATMENTS, BUDGETS


class MatchedAuditTests(unittest.TestCase):
    def fixture(self,root):
        def write(path,value):
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(value))
        configs=[dict(config_id=str(i),config=dict(tile=i)) for i in range(64)]
        context=dict(family='f',kernel_revision=1)
        context_id=hashlib.sha256(json.dumps(context,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:24]
        write(root/'pool.json',dict(pool_id='pool',configs=configs,protected_prefix=['1','2'],
                                   context=context))
        write(root/'rankings.json',dict(pool_id='pool',rankings={t:[str(i) for i in range(64)] for t in TREATMENTS},
            provenance=dict(pool=hashlib.sha256((root/'pool.json').read_bytes()).hexdigest())))
        for treatment in TREATMENTS:
            path=root/'results'/treatment/'observations.jsonl'
            path.parent.mkdir(parents=True)
            rows=[dict(**c,context_id=context_id,family='f',kernel_revision=1,strategy='test',elapsed_s=.01,
                       outcome=dict(status='ok',median_ms=1.,compile_ms=2.)) for c in configs]
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            confirmation=dict(status='complete',invalid=[],rounds=[
                dict(order=order,outcomes={'0':dict(outcome=dict(status='ok',median_ms=1.))})
                for order in (['0','1'],['1','0'])])
            events=[dict(kind='winner_decision',payload=dict(context_id=context_id,completed_trials=b,
                        config_id='0',confirmation=confirmation)) for b in BUDGETS]
            path.with_name('events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
            for budget,increment in zip(BUDGETS,(8,8,16,32)):
                write(root/'results'/f'{treatment}-{budget}.json',dict(pool_id='pool',budget=budget,
                    treatment=treatment,evaluated_trials=increment,verification=dict(status='ok'),winner=dict(tile=0)))

    def test_complete_and_tampered_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);self.fixture(root)
            self.assertTrue(audit(root)['complete'])
            path=root/'results'/'ordinary-8.json'
            result=json.loads(path.read_text());result['winner']=dict(tile=63)
            path.write_text(json.dumps(result))
            self.assertIn('ordinary: winner not a successful budget-prefix candidate',audit(root)['issues'])

    def test_duplicate_and_broken_tail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);self.fixture(root)
            path=root/'results'/'rules'/'observations.jsonl'
            text=path.read_text();path.write_text(text+text.splitlines()[0]+'\n')
            self.assertIn('rules: repeated candidate',audit(root)['issues'])
            path.write_text(text+'{"partial":')
            issues=audit(root)['issues']
            self.assertIn('rules: corrupt JSONL',issues)
            self.assertIn('rules: unterminated JSONL tail',issues)

    def test_missing_confirmation_cannot_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);self.fixture(root)
            (root/'results'/'ordinary'/'events.jsonl').write_text('')
            self.assertIn('ordinary: missing or invalid paired winner decision at 8',audit(root)['issues'])
        self.assertFalse(valid_confirmation(dict(config_id='x',confirmation=dict(status='deadline')), 'x'))


if __name__=='__main__':unittest.main()
