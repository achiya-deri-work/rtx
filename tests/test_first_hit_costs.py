import copy
import tempfile
from pathlib import Path
import unittest
from tools.research.first_hit_costs import summarize,read_jsonl


class FirstHitCostTests(unittest.TestCase):
    def fixture(self):
        doc=dict(mode='ml',budget=16,first_call_s=10,
                 winner={'metadata':{'tuning_evidence':dict(session_id='new',context_id='context')}})
        old=dict(session_id='old',context_id='context',observation_id='old',elapsed_s=90,
                 outcome=dict(compile_ms=80000))
        new=dict(session_id='new',context_id='context',observation_id='new',elapsed_s=4,
                 outcome=dict(compile_ms=3000),metadata=dict(proposal_elapsed_s=.5))
        confirmation=dict(session_id='new',kind='winner_confirmation',payload=dict(
            context_id='context',elapsed_s=1,outcome=dict(compile_ms=500)))
        return doc,[old,new],[confirmation,dict(session_id='new',kind='winner_decision',
                                               payload=dict(confirmation=confirmation))]

    def test_resumed_counts_and_nested_compile_not_double_counted(self):
        doc,rows,events=self.fixture();result=summarize(doc,rows,events)
        self.assertEqual(result['new_search_trials'],1)
        self.assertEqual(result['search_evaluator_s'],4)
        self.assertEqual(result['search_compile']['recorded_s'],3)
        self.assertEqual(result['finalist_evaluator_s'],1)
        self.assertEqual(result['unclassified_upper_bound_s'],4.5)
        self.assertIsNone(result['model_inference_s'])

    def test_missing_compile_is_not_reported_as_complete_zero(self):
        doc,rows,events=self.fixture();rows[-1]['outcome']['compile_ms']=None
        result=summarize(doc,rows,events)
        self.assertEqual(result['search_compile']['missing_count'],1)

    def test_failed_trials_consume_cost_without_becoming_successes(self):
        doc,rows,events=self.fixture()
        rows[-1]['outcome']['status']='compile_error'
        result=summarize(doc,rows,events)
        self.assertEqual(result['search_by_status']['compile_error']['count'],1)
        self.assertEqual(result['search_by_status']['compile_error']['compile']['recorded_s'],3)
        self.assertNotIn('ok',result['search_by_status'])

    def test_invalid_durations_duplicates_and_mixed_context_fail_closed(self):
        for value in (True,-1,float('nan'),float('inf')):
            doc,rows,events=self.fixture();rows[-1]['elapsed_s']=value
            with self.assertRaises(ValueError):summarize(doc,rows,events)
        doc,rows,events=self.fixture()
        with self.assertRaises(ValueError):summarize(doc,rows+[copy.deepcopy(rows[-1])],events)
        rows[-1]['context_id']='other'
        with self.assertRaises(ValueError):summarize(doc,rows,events)
        doc,rows,events=self.fixture();doc['first_call_s']=1
        with self.assertRaises(ValueError):summarize(doc,rows,events)

    def test_incomplete_or_corrupt_journal_is_not_silently_truncated(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'rows.jsonl'
            for content in ('{}','{}\ninvalid\n'):
                path.write_text(content)
                with self.assertRaises(ValueError):read_jsonl(path)
            path.write_text('{}\n')
            self.assertEqual(read_jsonl(path),[{}])


if __name__=='__main__':unittest.main()
