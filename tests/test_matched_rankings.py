import importlib.util
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec('sklearn'), 'research environment required')
class MatchedRankingTests(unittest.TestCase):
    def test_complete_orders_guarded_retrieval_and_no_labels(self):
        from tools.research.matched_rankings import freeze_rankings
        from tools.research.conditional_heuristics import export_rules
        train = [dict(family='test', revision=1, context='train', split='train', ms=1.+i%4,
                      features={'config.tile':float(i%4), 'context.workload.m':128.,
                                'context.workload.k':256.}) for i in range(160)]
        rules = export_rules(train)
        pool = dict(pool_id='pool', context=dict(family='test', kernel_revision=1,
                    workload=dict(m=384, n=128, k=256)),
                    configs=[dict(config_id=str(i), features={**train[i]['features'],
                              'context.workload.m':384.}) for i in range(4)])
        class Predictor:
            def bundle(self, adapter):
                return dict(contract=dict(policy={}), winners=[
                    dict(config_id=cid, features={'context.workload.m':384.})
                    for cid in ('3', '2', 'foreign')])
            def predict(self, rows, head):
                self.assert_no_labels = all('ms' not in r and 'outcome' not in r for r in rows)
                return [3., 2., 1., 0.]
        predictor = Predictor()
        result = freeze_rankings(pool, predictor, rules, excluded_mk=[(128,256)])
        self.assertTrue(predictor.assert_no_labels)
        self.assertEqual(result['portfolio_available'], 2)
        self.assertEqual(result['score_audit']['portfolio_ml']['distinct_scores'],4)
        self.assertEqual(result['score_audit']['portfolio_ml']['largest_tie'],1)
        self.assertEqual(result['rankings']['portfolio'][:2], ['2','3'])
        for order in result['rankings'].values():
            self.assertEqual(sorted(order), ['0','1','2','3'])
        reversed_pool = {**pool, 'configs':list(reversed(pool['configs']))}
        self.assertEqual(result['rankings'], freeze_rankings(
            reversed_pool, predictor, rules, excluded_mk=[(128,256)])['rankings'])
        with self.assertRaisesRegex(ValueError, 'already observed'):
            freeze_rankings(pool, predictor, rules, excluded_mk=[(384,256)])
        with patch.object(predictor, 'bundle', return_value=None):
            with self.assertRaisesRegex(ValueError, 'unsupported'):
                freeze_rankings(pool, predictor, rules, excluded_mk=[])

        projected={**pool,'context':{**pool['context'],
                   'tags':{'runtime_search_space_revision':1}},'runtime_retrieval_ids':['1','0']}
        # The live projection, not old artifact config IDs, owns the prefix.
        mapped=freeze_rankings(projected,predictor,rules,excluded_mk=[])
        self.assertEqual(mapped['portfolio_prefix'],['1','0'])
        pure=freeze_rankings(projected,predictor,rules,excluded_mk=[],retrieval_prefix=0)
        self.assertEqual(pure['portfolio_prefix'],[])
        self.assertEqual(pure['rankings']['portfolio_ml'],['3','2','1','0'])
        self.assertEqual(pure['rankings']['portfolio'],pure['rankings']['ordinary'])
        self.assertEqual(pure['rankings']['rules'],mapped['rankings']['rules'])
        for count in (True,-1,33,1.5):
            with self.assertRaises(ValueError):
                freeze_rankings(projected,predictor,rules,excluded_mk=[],retrieval_prefix=count)
        for invalid in (None,['foreign'],['1','1']):
            with self.assertRaisesRegex(ValueError,'projected retrieval'):
                freeze_rankings({**projected,'runtime_retrieval_ids':invalid},
                                predictor,rules,excluded_mk=[])


if __name__ == '__main__':
    unittest.main()
