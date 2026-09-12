"""Independent FP32-epilogue reference for the experimental fused GLU."""
import unittest

import torch

from rtx.kernels.mxfp8_glu import (
    DEFAULT_GLU_CONFIG, compile_mxfp8_glu, glu_rejection,
)
from rtx.kernels.mxfp8 import MXFP8Problem
from rtx.configs.glu import normalize_glu_config as normalize_fwd_config
from tests.test_mxfp8_fwd import _reference_to_mx


def reference(x, w, activation):
    def dequant(t):
        q, s = _reference_to_mx(t)
        return q.float() * s.float().repeat_interleave(32, -1)
    a, b = (dequant(x) @ dequant(w).T).chunk(2, -1)
    return getattr(torch.nn.functional, activation)(a) * b


class ConfigTests(unittest.TestCase):
    def test_module_conversion_and_contract(self):
        from rtx import MXFP8GLU
        gate=torch.nn.Linear(17,19,bias=False,dtype=torch.bfloat16)
        up=torch.nn.Linear(17,19,bias=False,dtype=torch.bfloat16)
        module=MXFP8GLU.from_linears(gate,up,activation="gelu")
        torch.testing.assert_close(module.weight[:19],gate.weight)
        torch.testing.assert_close(module.weight[19:],up.weight)
        self.assertNotEqual(module.weight.data_ptr(),gate.weight.data_ptr())
        self.assertEqual(list(module.state_dict()),["weight"])
        self.assertIn("forward_only=True",repr(module))
        restored=MXFP8GLU(17,19,activation="gelu")
        restored.load_state_dict(module.state_dict())
        torch.testing.assert_close(restored.weight,module.weight)
        with self.assertRaisesRegex(ValueError,"bias"):
            MXFP8GLU.from_linears(torch.nn.Linear(17,19,dtype=torch.bfloat16),up)
        with self.assertRaises(TypeError):
            MXFP8GLU(17,19,dtype=torch.float32)
        initialized=MXFP8GLU(128,128)
        self.assertLessEqual(initialized.weight.abs().max().item(),.041)

    def test_legality(self):
        p = MXFP8Problem(37, 194, 129)
        self.assertIsNone(glu_rejection(p, DEFAULT_GLU_CONFIG))
        self.assertIsNotNone(glu_rejection(p, normalize_fwd_config(tile_m=128, tile_n=256)))

    def test_projection_search_parity(self):
        from rtx.autotune.glu import GLU_SEARCH_SPACE
        from rtx.kernels.mxfp8 import FWD_SEARCH_SPACE
        self.assertEqual(set(FWD_SEARCH_SPACE)-set(GLU_SEARCH_SPACE),{"epilogue","epilogue_stages","store_vec"})
        self.assertTrue({"glu_epilogue","glu_store","glu_store_stages"} <= set(GLU_SEARCH_SPACE))

    def test_prequant_roundtrip(self):
        from dataclasses import asdict
        from rtx.configs.glu import MXFP8GLUPrequantConfig,prequant_glu_config_from_dict
        config=MXFP8GLUPrequantConfig()
        self.assertEqual(prequant_glu_config_from_dict(asdict(config)),config)
        self.assertIsNone(config.rejection(MXFP8Problem(512,3072,768),"mxfp8"))

    def test_native_scales_reject_row_major_staging(self):
        from dataclasses import replace
        from rtx.configs.glu import MXFP8GLUGemmConfig
        problem=MXFP8Problem(512,3072,768)
        base=MXFP8GLUGemmConfig(tile_m=64,atom_layout_m=2,scale_layout="mma64x128")
        for role in ("consumers","producer"):
            self.assertIn("native scales require TMA",replace(base,scale_role=role).rejection(problem))
        self.assertIsNone(replace(base,scale_role="tma").rejection(problem))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class CudaTests(unittest.TestCase):
    def test_module_and_packed_frontend(self):
        import tempfile
        from rtx import MXFP8GLU, mxfp8_glu_packed
        from rtx.configs.glu import MXFP8GLUPrequantConfig
        from dataclasses import replace
        config=MXFP8GLUPrequantConfig()
        config=replace(config,gemm=replace(config.gemm,glu_quant_vec=4,reduction="redux"))
        x=torch.randn(2,17,129,device="cuda",dtype=torch.bfloat16)*.2
        module=MXFP8GLU(129,97,device="cuda",output_format="mxfp8",packed_output=True,config=config)
        with self.assertRaisesRegex(RuntimeError,"backward"):
            module(x)
        with torch.no_grad():
            eager=module(x)
            compiled=torch.compile(module,fullgraph=True,dynamic=False)
            actual=compiled(x)
            self.assertEqual(actual[0].shape,(2,17,128))
            self.assertEqual(actual[1].shape,(2,17,4))
            for a,b in zip(actual,eager):
                torch.testing.assert_close(a.float(),b.float(),rtol=0,atol=0)
            self.assertTrue((actual[0].float()[...,97:]==0).all())
            alternate=mxfp8_glu_packed(x,module.weight,config=replace(config,gemm=replace(config.gemm,reduction="shuffle")))
            for a,b in zip(actual,alternate):
                torch.testing.assert_close(a.float(),b.float(),rtol=0,atol=0)
            with tempfile.TemporaryDirectory() as directory:
                self.assertIs(module.autotune_(x,backend="prequant",max_trials=1,
                                              time_budget_s=5,output_dir=directory,progress=None),module)
                self.assertIsNotNone(module.tuning_result)

    def test_formats_activations_and_tails(self):
        torch.manual_seed(941)
        for m, h, k in ((64, 128, 128), (37, 97, 129), (1, 1, 17)):
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.2
            w = torch.randn(2*h, k, device="cuda", dtype=torch.bfloat16) * 0.2
            for activation in ("silu", "gelu", "relu"):
                ref = reference(x, w, activation)
                for fmt in ("bf16", "mxfp8"):
                    with self.subTest(shape=(m,h,k), activation=activation, fmt=fmt):
                        width = h if fmt == "bf16" else (h+31)//32*32
                        o = torch.empty(m, width, device="cuda", dtype=torch.bfloat16 if fmt=="bf16" else torch.float8_e4m3fn)
                        s = torch.empty(m, 0 if fmt == "bf16" else (h+31)//32, device="cuda", dtype=torch.uint8)
                        f = compile_mxfp8_glu(MXFP8Problem(m,2*h,k), DEFAULT_GLU_CONFIG, activation, fmt)
                        auxiliary = o if fmt == "bf16" else s
                        f(x,w,o,auxiliary,auxiliary,auxiliary,auxiliary)
                        torch.cuda.synchronize()
                        if fmt == "bf16":
                            torch.testing.assert_close(o.float(), ref, rtol=0.004, atol=2e-6)
                        else:
                            qref, sref = _reference_to_mx(ref)
                            torch.testing.assert_close(s, sref.view(torch.uint8), rtol=0, atol=0)
                            torch.testing.assert_close(o.float(), qref.float(), rtol=0.13, atol=0.001)
                            self.assertTrue((o.float()[:, h:] == 0).all())

    def test_api_compile_and_opcheck(self):
        import json
        from rtx import mxfp8_glu
        from rtx.glu import _glu_op
        from rtx.kernels.mxfp8 import fwd_config_to_dict
        x = torch.randn(2,17,129,device="cuda",dtype=torch.bfloat16)
        w = torch.randn(194,129,device="cuda",dtype=torch.bfloat16)
        for fmt in ("bf16", "mxfp8"):
            args = (x,w,"silu",fmt,json.dumps(fwd_config_to_dict(DEFAULT_GLU_CONFIG),sort_keys=True))
            torch.library.opcheck(_glu_op, args, test_utils=("test_schema","test_faketensor","test_aot_dispatch_dynamic"))
            compiled = torch.compile(lambda x,w: mxfp8_glu(x,w,output_format=fmt), fullgraph=True, dynamic=False)
            eager = mxfp8_glu(x,w,output_format=fmt)
            result = compiled(x,w)
            if fmt == "mxfp8":
                eager, result = eager.dequantize(), result.dequantize()
            torch.testing.assert_close(result, eager, rtol=0, atol=0)
            retained = result.clone()
            changed = compiled(x, -w)
            expected = mxfp8_glu(x, -w, output_format=fmt)
            if fmt == "mxfp8":
                changed, expected = changed.dequantize(), expected.dequantize()
            torch.testing.assert_close(changed, expected, rtol=0, atol=0)
            torch.testing.assert_close(result, retained, rtol=0, atol=0)
        with self.assertRaisesRegex(RuntimeError, "backward"):
            mxfp8_glu(x.requires_grad_(),w)

    def test_schedule_variants(self):
        from rtx.autotune.glu import make_glu_adapter
        x = torch.randn(73,256,device="cuda",dtype=torch.bfloat16)*0.2
        w = torch.randn(258,256,device="cuda",dtype=torch.bfloat16)*0.2
        adapter = make_glu_adapter(x,w,output_format="mxfp8",warmup=1,samples=2,calls_per_sample=2)
        for updates in ({"mxfp8_stages":2}, {"quant_vec":4,"quant_load_bits":64},
                        {"tile_m":128}, {"reduction":"redux"},
                        {"persistent":True,"raster":"m","grid_swizzle":2}):
            config = normalize_fwd_config(DEFAULT_GLU_CONFIG,**updates)
            with self.subTest(updates=updates):
                self.assertIsNone(adapter.rejection(config))
                outcome = adapter.evaluate(config)
                self.assertEqual(outcome.status, "ok", outcome.error)

    def test_zero_and_nondefault_stream(self):
        from rtx import mxfp8_glu
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            x = torch.zeros(19,35,device="cuda",dtype=torch.bfloat16)
            w = torch.randn(66,35,device="cuda",dtype=torch.bfloat16)
            y = mxfp8_glu(x,w,output_format="mxfp8").dequantize()
        stream.synchronize()
        self.assertTrue((y == 0).all())

    def test_async_and_register_epilogues(self):
        from rtx.autotune.glu import make_glu_adapter
        from rtx.configs.glu import normalize_glu_config
        torch.manual_seed(12)
        for m,h,k in ((128,128,256), (73,129,256)):
            x = torch.randn(m,k,device="cuda",dtype=torch.bfloat16)*0.2
            w = torch.randn(2*h,k,device="cuda",dtype=torch.bfloat16)*0.2
            for fmt in ("bf16","mxfp8"):
                for activation in ("silu","gelu","relu"):
                    adapter = make_glu_adapter(x,w,activation=activation,output_format=fmt,warmup=1,samples=1,calls_per_sample=1)
                    for epilogue in ("exchange","registers"):
                        for engine,schedule in (("scalar","cooperative"),("cpasync","cooperative"),("tma","cooperative"),("tma","warp_specialized"),("tma","three_role")):
                            updates = dict(glu_epilogue=epilogue,load_engine=engine,schedule=schedule,quant_vec=8,quant_load_bits=128)
                            if engine != "scalar":
                                updates.update(bf16_tile_k=32,bf16_swizzle="none",bf16_stages=2)
                            cfg = normalize_glu_config(**updates)
                            if adapter.rejection(cfg):
                                continue
                            with self.subTest(shape=(m,h,k),fmt=fmt,act=activation,epilogue=epilogue,engine=engine,schedule=schedule):
                                result = adapter.evaluate(cfg)
                                self.assertEqual(result.status,"ok",result.error)

    def test_prequant_glu(self):
        from dataclasses import replace
        from rtx.autotune.glu import make_glu_adapter
        from rtx.configs.glu import MXFP8GLUPrequantConfig
        from rtx import mxfp8_glu
        for m,h,k in ((128,128,256),(73,129,129)):
            x=torch.randn(m,k,device="cuda",dtype=torch.bfloat16)*.2
            w=torch.randn(2*h,k,device="cuda",dtype=torch.bfloat16)*.2
            for fmt in ("bf16","mxfp8"):
                for act in ("silu","gelu","relu"):
                    adapter=make_glu_adapter(x,w,backend="prequant",activation=act,output_format=fmt,warmup=1,samples=1,calls_per_sample=1)
                    for layout in ("row_major","mma128"):
                        for launches in ("dual","separate"):
                            base=MXFP8GLUPrequantConfig()
                            c=replace(base,quant=replace(base.quant,scale_layout=layout),
                                      gemm=replace(base.gemm,scale_layout=layout,scale_role="tma" if layout=="mma128" else "consumers"),quant_launches=launches)
                            if adapter.rejection(c):
                                continue
                            with self.subTest(shape=(m,h,k),fmt=fmt,act=act,layout=layout,launches=launches):
                                outcome=adapter.evaluate(c)
                                self.assertEqual(outcome.status,"ok",outcome.error)
            config=MXFP8GLUPrequantConfig()
            compiled=torch.compile(lambda x,w:mxfp8_glu(x,w,config=config),fullgraph=True,dynamic=False)
            from torch._inductor.utils import run_and_get_code
            actual,code=run_and_get_code(compiled,x,w)
            self.assertIn("torch._rtx_glu_launchers","\n".join(code))
            torch.testing.assert_close(actual,mxfp8_glu(x,w,config=config),rtol=0,atol=0)
            changed=compiled(x,-w)
            torch.testing.assert_close(changed,mxfp8_glu(x,-w,config=config),rtol=0,atol=0)

    def test_tma_store_recycling_and_tails(self):
        from dataclasses import replace
        from rtx.autotune.glu import make_glu_adapter
        from rtx.configs.glu import MXFP8GLUPrequantConfig,normalize_glu_config
        x=torch.randn(2049,128,device="cuda",dtype=torch.bfloat16)*.2
        w=torch.randn(2064,128,device="cuda",dtype=torch.bfloat16)*.2
        for backend in ("cta","prequant"):
            a=make_glu_adapter(x,w,backend=backend,warmup=1,samples=1,calls_per_sample=1)
            for stages in (1,2,3,4):
                if backend=="cta":
                    c=normalize_glu_config(glu_epilogue="registers",glu_store="tma",glu_store_stages=stages,
                                           quant_vec=8,quant_load_bits=128,persistent=True,raster="n")
                else:
                    b=MXFP8GLUPrequantConfig()
                    c=replace(b,gemm=replace(b.gemm,tile_m=64,atom_layout_m=2,stages=1,
                                            glu_store="tma",glu_store_stages=stages,tiles_per_cta=8))
                with self.subTest(backend=backend,stages=stages):
                    self.assertIsNone(a.rejection(c))
                    result=a.evaluate(c)
                    self.assertEqual(result.status,"ok",result.error)

    def test_vectorized_output_quantization(self):
        from dataclasses import replace
        from rtx.autotune.glu import make_glu_adapter
        from rtx.configs.glu import MXFP8GLUPrequantConfig
        for m,h,k in ((37,97,129),(128,128,256)):
            x=torch.randn(m,k,device="cuda",dtype=torch.bfloat16)*.2
            w=torch.randn(2*h,k,device="cuda",dtype=torch.bfloat16)*.2
            for backend in ("cta","prequant"):
                for activation in ("silu","gelu","relu"):
                    adapter=make_glu_adapter(x,w,backend=backend,activation=activation,
                                             output_format="mxfp8",warmup=1,samples=1,calls_per_sample=1)
                    for vec in (1,2,4,8):
                        for epilogue in ("exchange","registers"):
                            if backend=="cta":
                                config=normalize_fwd_config(glu_quant_vec=vec,glu_epilogue=epilogue)
                            else:
                                base=MXFP8GLUPrequantConfig()
                                config=replace(base,gemm=replace(base.gemm,glu_quant_vec=vec,glu_epilogue=epilogue))
                            if adapter.rejection(config):
                                continue
                            with self.subTest(shape=(m,h,k),backend=backend,act=activation,vec=vec,epilogue=epilogue):
                                result=adapter.evaluate(config)
                                self.assertEqual(result.status,"ok",result.error)

    def test_graph_replay_updates_and_memory(self):
        from dataclasses import replace
        from rtx import mxfp8_glu
        from rtx.configs.glu import MXFP8GLUPrequantConfig
        base=MXFP8GLUPrequantConfig()
        config=replace(base,gemm=replace(base.gemm,glu_quant_vec=4))
        x=torch.randn(64,128,device="cuda",dtype=torch.bfloat16)*.2
        w=torch.randn(256,128,device="cuda",dtype=torch.bfloat16)*.2
        for fmt in ("bf16","mxfp8"):
            with self.subTest(fmt=fmt),torch.inference_mode():
                fn=torch.compile(lambda x,w:mxfp8_glu(x,w,config=config,output_format=fmt),
                                 fullgraph=True,dynamic=False)
                for _ in range(3):
                    fn(x,w)
                torch.cuda.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output=fn(x,w)
                graph.replay()
                torch.cuda.synchronize()
                before=torch.cuda.memory_allocated()
                for _ in range(50):
                    graph.replay()
                torch.cuda.synchronize()
                self.assertEqual(torch.cuda.memory_allocated(),before)
                # Graphs capture addresses, not values: in-place input/weight
                # updates must be quantized again on every replay.
                x.mul_(.75)
                w.neg_()
                graph.replay()
                expected=mxfp8_glu(x,w,config=config,output_format=fmt)
                if fmt=="mxfp8":
                    actual,expected=output.dequantize(),expected.dequantize()
                else:
                    actual=output
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)


if __name__ == "__main__":
    unittest.main()
