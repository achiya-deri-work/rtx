"""Optional real-artifact CUDA integration; no release performance claim."""
import json
import os
from pathlib import Path
import tempfile
import unittest

import torch
import rtx

ARTIFACT = Path(__file__).resolve().parents[1]/'autotune_priors/tournament_predictors_v1.json.gz'


def artifact_supports_record(path, row):
    from types import SimpleNamespace
    from rtx.autotune.predictors import load_predictor
    context = SimpleNamespace(family=row['family'], kernel_revision=row['kernel_revision'],
                              features=lambda: row['features'])
    return load_predictor(str(path), research=True).bundle(SimpleNamespace(context=context)) is not None


@unittest.skipUnless(torch.cuda.is_available() and ARTIFACT.is_file(), 'requires CUDA and exported research artifact')
class PredictorCudaTests(unittest.TestCase):
    def test_cold_compile_training_and_cache(self):
        policy = rtx.AutotunePolicy(max_trials=8, max_seconds=90, pool_size=64,
            candidate_mix=os.environ.get('RTX_TEST_CANDIDATE_MIX', 'legacy'),
            warmup=1, samples=3, calls_per_sample=2,
            predictor_artifact=str(ARTIFACT), allow_research_prior=True)
        with tempfile.TemporaryDirectory() as tmp:
            for cls in (rtx.MXFP8Linear, rtx.NVFP4Linear):
                with self.subTest(layer=cls.__name__):
                    torch._dynamo.reset()
                    layer = cls(256, 128, device='cuda', tuning_policy=policy, autotune_cache_dir=tmp)
                    compiled = torch.compile(layer, fullgraph=True)
                    x = (torch.randn(128, 256, device='cuda', dtype=torch.bfloat16)*.2).requires_grad_()
                    y = compiled(x)
                    dy = torch.randn_like(y)
                    y.backward(dy)
                    references = (x.detach().float()@layer.weight.detach().float().T,
                                  dy.float()@layer.weight.detach().float(), dy.float().T@x.detach().float())
                    for actual, ref in zip((y, x.grad, layer.weight.grad), references):
                        self.assertTrue(actual.isfinite().all())
                        self.assertLess(float((actual.detach().float()-ref).norm()/ref.norm()), .3)
                    def count():
                        return sum(len(p.read_text().splitlines()) for p in Path(tmp).rglob('observations.jsonl'))
                    before = count()
                    compiled(x).sum().backward()
                    self.assertEqual(count(), before)
            records = [json.loads(line) for p in Path(tmp).rglob('observations.jsonl') for line in p.read_text().splitlines()]
            compatible=[r for r in records if artifact_supports_record(ARTIFACT, r)]
            if compatible:
                self.assertTrue({'portable_mlp', 'portable_residual', 'retrieval'} <= {r['strategy'] for r in compatible})
            else:
                # Historical research artifacts must not silently cross a
                # kernel revision. Exercise cold-start/cache fallback instead.
                self.assertTrue(records)
                self.assertFalse(any(r['strategy'].startswith('portable_') or
                                     r['strategy']=='retrieval' for r in records))


if __name__ == '__main__':
    unittest.main()
