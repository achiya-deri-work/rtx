from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rtx.autotune.effort import AutotunePolicy,effort_residual_root
from tests.test_autotune_coverage import adapter


class CompositeEffortNamespaceTests(unittest.TestCase):
    def test_budget_extension_and_policy_change(self):
        p=AutotunePolicy(search_schedule='progressive',max_trials=8)
        root=effort_residual_root('root',p)
        self.assertEqual(root,effort_residual_root('root',replace(p,max_trials=64,max_seconds=180)))
        self.assertNotEqual(root,effort_residual_root('root',replace(p,samples=9)))
        self.assertEqual(effort_residual_root('root',replace(p,search_schedule='legacy')),Path('root'))

    def test_glu_and_mlp_use_policy_and_domain_namespaces(self):
        from rtx.autotune.glu import tune_mxfp8_glu
        from rtx.autotune.mlp import tune_mxfp8_mlp
        x=SimpleNamespace(device='cuda')
        for family in ('glu','mlp'):
            with self.subTest(family=family),tempfile.TemporaryDirectory() as root:
                paths=[];a=adapter()
                def factory(adapt,store,*args,**kwargs):
                    paths.append(store.path)
                    return SimpleNamespace(tune=lambda:SimpleNamespace(config=adapt.initial_config))
                with patch('torch.cuda.device',side_effect=lambda *_:nullcontext()), \
                     patch('torch.cuda.is_current_stream_capturing',return_value=False), \
                     patch(f'rtx.autotune.{family}.make_{family}_adapter',return_value=a), \
                     patch('rtx.autotune.mlp.mlp_structural_seeds',return_value=[]), \
                     patch('rtx.autotune.effort.make_effort_autotuner',side_effect=factory), \
                     patch('rtx.autotune.glu.make_hybrid_autotuner',side_effect=factory):
                    p=AutotunePolicy(search_schedule='progressive',max_trials=8)
                    def run(policy):
                        kwargs=dict(output_dir=root,tuning_policy=policy)
                        if family=='glu':tune_mxfp8_glu(x,None,backend='prequant',**kwargs)
                        else:tune_mxfp8_mlp(x,None,None,**kwargs)
                    run(p);run(replace(p,max_trials=16));run(replace(p,samples=9))
                    a.axes={'a.tile':(0,1),'b.stages':(0,1)}
                    run(p)
                self.assertEqual(paths[0],paths[1])
                self.assertNotEqual(paths[0],paths[2])
                self.assertNotEqual(paths[0],paths[3])
                self.assertEqual(Path(paths[0]).name,p.evidence_contract())


if __name__=='__main__':unittest.main()
