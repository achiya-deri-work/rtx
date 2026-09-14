import unittest
import importlib.util
HAS_RESEARCH = importlib.util.find_spec('sklearn') is not None
if HAS_RESEARCH:
    from tools.research.portfolio_study import fit_portfolios, portfolio_ids, score, splits


def row(context, cid, ms, split='train', device='a', shape='00000001', shape_class='ordinary'):
    return dict(context=context,config_id=cid,ms=ms,split=split,device=device,
                shape=shape,shape_class=shape_class,family='f',revision=1,config={'tile':cid},
                features={'context.workload.m':128,'context.workload.k':128})


@unittest.skipUnless(HAS_RESEARCH, 'requires research environment')
class PortfolioTests(unittest.TestCase):
    def test_coverage_and_unmeasured_are_not_latency_labels(self):
        train=[row('one','a',1),row('one','b',2),row('two','b',1),row('two','c',2)]
        portfolio=fit_portfolios(train,size=2)
        self.assertEqual(set(portfolio_ids(portfolio,train[0])), {'a','b'})
        self.assertNotIn('unseen',portfolio_ids(portfolio,train[0]))

    def test_grouped_splits(self):
        values=[row('one','a',1,shape='00000000'),row('two','a',2,device='b',shape='00000000'),
                row('three','a',1,shape='00000001')]
        protocol,tr,te=next(x for x in splits(values) if x[0]=='unseen_shapes')
        self.assertFalse({r['shape'] for r in tr}&{r['shape'] for r in te})
        self.assertEqual(len(te),2)

    def test_budgets_and_missing_portfolio_accounting(self):
        train=[row('train','unmeasured',1)]
        test=[row('abcd',str(i),1+i) for i in range(10)]
        result=score(test,list(range(10)),fit_portfolios(train),'test','test')
        self.assertEqual(len(result),2)
        self.assertTrue(all(r['budget']==8 and r['regret']==0 for r in result))
        self.assertTrue(all(r['portfolio_measured']==0 and r['portfolio_proposed']==1 for r in result))
