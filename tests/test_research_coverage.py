from dataclasses import replace
import unittest

from tools.research.coverage_audit import classify,effective_config,context_weights
from tools.research.structural_sampling import raw_portfolio,select_portfolio,StructuralSamplingPolicy,pool_diagnostics


BAD=dict(tile_m=64,stages=1,scale_layout='row_major',scale_role='consumers')


def record(precision='mxfp8',phase='forward'):
    return dict(family='mxfp8_recompute_mlp',kernel_revision=33,
        features={f'context.workload.precision={precision}':1,f'context.workload.phase={phase}':1},
        config=dict(gate_backend='onchip',down_backend='onchip',gate_up={'a':1},down={'a':2},
                    gate_prequant={'gemm':BAD},down_gemm={'tile_m':128}))


class ResearchCoverageTests(unittest.TestCase):
    def test_active_quarantine_and_unknown_revision(self):
        r=record();self.assertEqual(classify(r),(True,False,True))
        self.assertEqual(classify(record('bf16_gate')),(True,True,True))
        r['config']['gate_backend']='cooperative_prequant'
        self.assertEqual(classify(r),(True,True,True))
        r=record();r['kernel_revision']=999
        self.assertEqual(classify(r),(True,True,False))
        r=record(phase='train');r['config'].update(recompute_backend='prequant',recompute={'gemm':BAD})
        self.assertEqual(classify(r),(True,True,True))

    def test_effective_dedup_keeps_active_backward(self):
        a=record();b=record();b['config']['gate_prequant']={'gemm':{'stages':4}}
        self.assertEqual(effective_config(a),effective_config(b))
        a=record('bf16_gate','train');b=record('bf16_gate','train')
        b['config']['gate_backward']={'tile':32}
        self.assertNotEqual(effective_config(a),effective_config(b))

    def test_equal_context_and_family_weights(self):
        rows=[dict(family='a',context='x')]*10+[dict(family='a',context='y')]*2+[dict(family='b',context='z')]
        weights=context_weights(rows)
        self.assertAlmostEqual(sum(weights[:10]),sum(weights[10:12]))
        self.assertAlmostEqual(sum(weights[:12]),weights[12])

    def test_crosses_intermediate_legality_barrier_and_replays(self):
        from types import SimpleNamespace
        a=SimpleNamespace(initial_config={'a':0,'b':0},axes={'a.tile':(0,1,2),'b.stages':(0,1,2)},
            serialize=dict,deserialize=dict,
            update_fn=lambda c,k,v:{**c,k.split('.')[0]:v},
            rejection=lambda c:None if c['a']==c['b'] else ('illegal','must match'))
        policy=StructuralSamplingPolicy(3,128)
        raw=raw_portfolio(a,policy)
        self.assertEqual(raw,raw_portfolio(a,policy))
        chosen,audit=select_portfolio(a,raw,3)
        self.assertEqual({c['a'] for c in chosen},{0,1,2})
        self.assertTrue(any(r['status']=='rejected' for r in audit))

    def test_pool_ceiling_decomposition(self):
        result=pool_diagnostics({'a':1,'b':2,'c':4},['b','c'],['c'])
        self.assertEqual(result['pool_penalty'],2)
        self.assertEqual(result['selector_penalty'],2)
        self.assertEqual(pool_diagnostics({'a':1},['missing'],['missing'])['pool_label_coverage'],0)
        with self.assertRaises(ValueError):pool_diagnostics({},['a'],['b'])


if __name__=='__main__':unittest.main()
