import unittest
from tools.research.transport_calibration import summarize


class TransportCalibrationTests(unittest.TestCase):
    def row(self,context,cid,ms):
        return dict(context=context,config_id=cid,ms=ms,device='gpu',shape_class='ordinary',
            shape=context,features={'context.regime=hot':1},
            config={'gemm':dict(scale_layout='row_major',scale_role='consumers',epilogue='direct')})

    def test_contexts_equal_weight_and_residual_sign(self):
        rows=[self.row('dense',str(i),1.) for i in range(9)]+[self.row('sparse','0',1.)]
        group=summarize(rows,[-1.]*9+[0.])[0]
        self.assertEqual(group['context_median_log_residual'],.5)
        self.assertEqual((group['rows'],group['contexts']),(10,2))

    def test_reject_bad_evidence(self):
        row=self.row('a','b',1.)
        for rows,scores in [([],[]),([row],[]),([row,row],[0.,0.]),([row],[float('nan')]),
                            ([self.row('a','b',0.)],[0.])]:
            with self.subTest(rows=rows),self.assertRaises(ValueError):summarize(rows,scores)


if __name__=='__main__':unittest.main()
