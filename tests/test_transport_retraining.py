import unittest
from tools.research.transport_retraining import shape_fold


class ShapeFoldTests(unittest.TestCase):
    def test_whole_mk_across_device_width_policy(self):
        base={'features':{'context.workload.m':512,'context.workload.k':1536}}
        expected=shape_fold(base)
        for device in ('a','b','c'):
            for width in (768,3072):
                row={'device':device,'features':{**base['features'],'context.workload.n':width,
                    'context.tags.scaling=delayed':1}}
                self.assertEqual(shape_fold(row),expected)
        self.assertEqual(shape_fold({'features':{k:float(v) for k,v in base['features'].items()}}),expected)

    def test_invalid_shapes_and_fold_counts(self):
        for value in (0,-1,1.5,True):
            with self.subTest(value=value),self.assertRaises(ValueError):
                shape_fold({'features':{'context.workload.m':value,'context.workload.k':128}})
        for folds in (True,1,1.5):
            with self.assertRaises(ValueError):shape_fold({},folds)


if __name__=='__main__':unittest.main()
