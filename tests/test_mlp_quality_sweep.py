import unittest
import json
from pathlib import Path
import tempfile

from benchmarks.sweep_mlp_quality import tasks


class QualitySweepTests(unittest.TestCase):
    def test_coverage_before_search(self):
        work=list(tasks())
        self.assertEqual(len(work),100)
        self.assertEqual({task[2] for task in work},{"silu","gelu"})
        self.assertEqual(len({task[1] for task in work}),6)
        first_search=next(i for i,t in enumerate(work) if t[0]=="search")
        self.assertEqual(first_search,60)
        self.assertEqual(len({(t[1],t[2],t[3]) for t in work[:36]}),36)

    def test_schedule_keys_are_unique(self):
        keys=[(wave,shape,act,policy,tuple(options)) for wave,shape,act,policy,options in tasks()]
        self.assertEqual(len(keys),len(set(keys)))

    def test_summary_does_not_pool_distributions(self):
        from benchmarks.summarize_mlp_quality_sweep import summarize
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for i,(distribution,latency) in enumerate((("init_002",100),("uniform_half",1))):
                trial=root/str(i);trial.mkdir()
                (trial/"outcome.json").write_text(json.dumps({"status":"ok","wave":"coverage"}))
                report={"args":{"shape":"128,256,512,256","activation":"silu",
                    "policy":"bf16_gate_fwd","distribution":distribution,"seed":17},
                    "train":{"median_us":latency},"forward":{"median_us":latency/3},
                    "bf16_error":{name:{"relative_l2":.05} for name in ("y","dx","dgu","ddown")}}
                (trial/"result.json").write_text(json.dumps(report))
            summary,rows=summarize(root)
            self.assertEqual(len(rows),2)
            winner=summary["contexts"][0]["best_search_observations"]["bf16_gate_fwd"]
            self.assertEqual(winner["train_us"],100)

    def test_tail_store_legality(self):
        from dataclasses import replace
        from rtx.fp8 import MXFP8PrequantConfig
        from rtx.configs.mxfp8 import MXFP8QuantConfig,MXFP8GemmConfig
        from rtx.kernels.mxfp8 import MXFP8Problem
        config=MXFP8PrequantConfig(quant=MXFP8QuantConfig(scale_layout="row_major"),
            gemm=MXFP8GemmConfig(tile_m=64,atom_layout_m=2,stages=1))
        problem=MXFP8Problem(129,256,128)
        self.assertIn("full M/N tiles",config.rejection(problem))
        direct=replace(config,gemm=replace(config.gemm,epilogue="direct",store_vec=1))
        self.assertIsNone(direct.rejection(problem))

    def test_fused_feature_tail_legality(self):
        from rtx.kernels.mxfp8 import MXFP8Problem,normalize_fwd_config
        config=normalize_fwd_config(tile_m=64,quant_vec=8,quant_load_bits=128,
            quant_math="bf16x2",quant_amax="bf16_bits",quant_store_bits=32)
        self.assertIsNone(config.implementation_rejection(MXFP8Problem(129,272,136)))


if __name__=="__main__":
    unittest.main()
