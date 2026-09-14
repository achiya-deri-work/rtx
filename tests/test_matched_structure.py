import unittest
from tools.research.matched_structure import associations


class StructuralAssociationsTests(unittest.TestCase):
    def setUp(self):
        self.configs=[dict(config_id=str(i),config={'g':{'layout':v}})
                      for i,v in enumerate(('a','a','b','b'))]
        self.reference=[dict(config_id=str(i),outcome={'status':'ok','median_ms':v})
                        for i,v in enumerate((1.,1.04,2.,3.))]
        self.ranking=['2','3','0','1']

    def run_report(self,**kwargs):
        return associations(self.configs,self.reference,self.ranking,['g.layout'],**kwargs)

    def test_rank_exposure_and_reference_association_separate(self):
        report=self.run_report(opening=2)
        a,b=report['groups']
        self.assertEqual((a['near_5pct'],a['top_opening_count'],a['first_near_rank']),(2,0,3))
        self.assertEqual((b['near_5pct'],b['top_opening_count'],b['first_near_rank']),(0,2,None))
        self.assertAlmostEqual(b['median_reference_excess'],1.5)
        self.assertFalse(report['runtime_promotion'])

    def test_failure_consumes_rank_without_latency_imputation(self):
        self.reference[2]['outcome']={'status':'compile_error','median_ms':0}
        b=self.run_report(opening=2)['groups'][1]
        self.assertEqual((b['failed'],b['successful'],b['top_opening_count']),(1,1,2))
        self.assertEqual(b['median_reference_excess'],2.)

    def test_reject_incomplete_duplicate_nonfinite_missing_fields(self):
        for change in ('missing','duplicate','nan','bool','field','rank'):
            with self.subTest(change=change):
                self.setUp()
                if change=='missing':self.reference.pop()
                if change=='duplicate':self.reference[-1]=self.reference[0]
                if change=='nan':self.reference[0]['outcome']['median_ms']=float('nan')
                if change=='bool':self.reference[0]['outcome']['median_ms']=True
                if change=='field':self.configs[0]['config']={}
                if change=='rank':self.ranking[-1]=self.ranking[0]
                with self.assertRaises(ValueError):self.run_report()


if __name__=='__main__':unittest.main()
