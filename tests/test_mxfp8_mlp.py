"""Independent arithmetic and retention contracts for recomputed MXFP8 MLP."""
import unittest
import torch


class MLPConfigTests(unittest.TestCase):
    def test_revisioned_roundtrip(self):
        from rtx import MXFP8MLPConfig
        config=MXFP8MLPConfig()
        self.assertEqual(config,MXFP8MLPConfig.from_dict(config.to_dict()))
        stale=config.to_dict()
        stale["revision"]=1
        with self.assertRaisesRegex(ValueError,"revision"):
            MXFP8MLPConfig.from_dict(stale)

    def test_invalid_policy_choices(self):
        from dataclasses import replace
        from rtx import MXFP8MLPConfig
        from rtx.configs.glu import glu_rejection
        from rtx.kernels.mxfp8 import MXFP8Problem
        with self.assertRaises(ValueError):
            MXFP8MLPConfig(gate_backend="unknown")
        with self.assertRaises(ValueError):
            MXFP8MLPConfig(cooperative_ctas_per_sm=.75)
        base=MXFP8MLPConfig()
        self.assertIsNotNone(glu_rejection(MXFP8Problem(64,256,128),
                                         replace(base.gate_up,glu_hidden_decode="unknown")))
        bad=replace(base.recompute,gemm=replace(base.recompute.gemm,
                                               glu_grad_load="shared",glu_grad_store="exchange"))
        self.assertIsNotNone(bad.rejection(MXFP8Problem(64,256,128),"mxfp8"))


def quantized(t):
    from tests.test_mxfp8_fwd import _reference_to_mx
    q,s=_reference_to_mx(t.contiguous())
    return (q.float()*s.float().repeat_interleave(32,-1))[:,:t.shape[1]]


def reference(x,gu,down,dy,activation):
    shape=x.shape
    x=x.reshape(-1,x.shape[-1]);dy=dy.reshape(-1,dy.shape[-1])
    p=quantized(x)@quantized(gu).T
    g,u=p.chunk(2,-1)
    a=getattr(torch.nn.functional,activation)(g)
    h=quantized(a*u).bfloat16()
    y=(h.float()@quantized(down).T).bfloat16().view(*shape[:-1],down.shape[0])
    dh=(quantized(dy)@quantized(down.T).T).bfloat16().float()
    dd=(quantized(dy.T)@quantized(h.T).T).bfloat16()
    if activation=="silu":
        s=torch.sigmoid(g);derivative=s*(1+g*(1-s))
    elif activation=="gelu":
        derivative=.5*(1+torch.erf(g/(2**.5)))+g*torch.exp(-g*g/2)/(2*torch.pi)**.5
    else:
        derivative=(g>0).float()
    dp=torch.cat((dh*u*derivative,dh*a),-1).bfloat16()
    dx=(quantized(dp)@quantized(gu.T).T).bfloat16().view(shape)
    dg=(quantized(dp.T)@quantized(x.T).T).bfloat16()
    return y,(dx,dg,dd)


@unittest.skipUnless(torch.cuda.is_available(),"requires CUDA")
class MLPTests(unittest.TestCase):
    def test_shared_reciprocal_reference(self):
        from dataclasses import replace
        from rtx import mxfp8_mlp, MXFP8MLPConfig
        torch.manual_seed(393)
        for m,k,h,o in ((64,128,256,64),(17,35,97,41)):
            inputs=tuple((torch.randn(*s,device="cuda",dtype=torch.bfloat16)*.2).requires_grad_()
                         for s in ((m,k),(2*h,k),(o,h)))
            dy=torch.randn(m,o,device="cuda",dtype=torch.bfloat16)
            y_ref,g_ref=reference(*inputs,dy,"silu")
            for backend in ("onchip","prequant"):
                base=MXFP8MLPConfig()
                cfg=replace(base,recompute_backend=backend,
                            gate_up=replace(base.gate_up,glu_silu_math="shared_reciprocal"))
                y=mxfp8_mlp(*inputs,config=cfg)
                for a,b in zip((y,*torch.autograd.grad(y,inputs,dy)),(y_ref,*g_ref)):
                    error=((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-20)).item()
                    self.assertLess(error,.025)

    def test_cooperative_compile_graph_and_dynamic_weights(self):
        from dataclasses import replace
        from rtx import mxfp8_mlp, MXFP8MLPConfig
        torch.manual_seed(392)
        inputs=tuple((torch.randn(*s,device="cuda",dtype=torch.bfloat16)*.02).requires_grad_()
                     for s in ((64,128),(512,128),(64,256)))
        dy=torch.randn(64,64,device="cuda",dtype=torch.bfloat16)
        cfg=replace(MXFP8MLPConfig(),gate_backend="cooperative_prequant",down_backend="producer_quantized",
                    recompute_backend="prequant",backward_overlap=True)
        fn=torch.compile(lambda x,w,d:mxfp8_mlp(x,w,d,config=cfg),fullgraph=True,dynamic=False)
        def run():
            y=fn(*inputs)
            return y,*torch.autograd.grad(y,inputs,dy)
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):run()
        stream.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):actual=run()
        for step in range(3):
            with torch.no_grad():inputs[2].add_(.001)
            graph.replay()
            y=mxfp8_mlp(*inputs)
            expected=(y,*torch.autograd.grad(y,inputs,dy))
            for a,b in zip(actual,expected):
                torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_derivative_store_exchange(self):
        from dataclasses import replace
        from rtx import mxfp8_mlp, MXFP8MLPConfig
        for m,k,h,o in ((64,128,256,64),(17,35,97,41)):
            inputs=tuple((torch.randn(*s,device="cuda",dtype=torch.bfloat16)*.02).requires_grad_()
                         for s in ((m,k),(2*h,k),(o,h)))
            dy=torch.randn(m,o,device="cuda",dtype=torch.bfloat16)
            base=MXFP8MLPConfig()
            y=mxfp8_mlp(*inputs,config=base)
            expected=(y,*torch.autograd.grad(y,inputs,dy))
            for backend in ("onchip","prequant"):
                cfg=replace(base,recompute_backend=backend,
                            gate_up=replace(base.gate_up,glu_grad_load="prefetch",glu_grad_store="exchange",glu_hidden_decode="bf16x2",glu_scratch_swizzle="xor_16b"),
                            recompute=replace(base.recompute,gemm=replace(base.recompute.gemm,
                                              glu_grad_load="prefetch",glu_grad_store="exchange",glu_hidden_decode="bf16x2",glu_scratch_swizzle="xor_16b")))
                y=mxfp8_mlp(*inputs,config=cfg)
                for a,b in zip((y,*torch.autograd.grad(y,inputs,dy)),expected):
                    torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_derivative_gradient_load_schedules(self):
        from dataclasses import replace
        from rtx import mxfp8_mlp, MXFP8MLPConfig
        for m,k,h,o in ((64,128,256,64),(17,35,97,41)):
            inputs=tuple((torch.randn(*s,device="cuda",dtype=torch.bfloat16)*.02).requires_grad_()
                         for s in ((m,k),(2*h,k),(o,h)))
            dy=torch.randn(m,o,device="cuda",dtype=torch.bfloat16)
            base=MXFP8MLPConfig()
            y=mxfp8_mlp(*inputs,config=base)
            expected=(y,*torch.autograd.grad(y,inputs,dy))
            for backend in ("onchip","prequant"):
                for mode in ("prefetch","shared"):
                    cfg=replace(base,recompute_backend=backend,
                                gate_up=replace(base.gate_up,glu_grad_load=mode),
                                recompute=replace(base.recompute,gemm=replace(base.recompute.gemm,glu_grad_load=mode)))
                    y=mxfp8_mlp(*inputs,config=cfg)
                    for a,b in zip((y,*torch.autograd.grad(y,inputs,dy)),expected):
                        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_prequant_recompute(self):
        from dataclasses import replace
        from rtx import mxfp8_mlp, MXFP8MLPConfig
        torch.manual_seed(391)
        for m,k,h,o in ((64,128,256,64),(17,35,97,41)):
            inputs=tuple((torch.randn(*s,device="cuda",dtype=torch.bfloat16)*.02).requires_grad_()
                         for s in ((m,k),(2*h,k),(o,h)))
            dy=torch.randn(m,o,device="cuda",dtype=torch.bfloat16)
            for activation in ("silu","gelu","relu"):
                base=MXFP8MLPConfig()
                y=mxfp8_mlp(*inputs,activation=activation,config=base)
                expected=(y,*torch.autograd.grad(y,inputs,dy))
                for schedule in ("joint","two_pass"):
                    cfg=replace(base,recompute_backend="prequant",backward_recompute=schedule,
                                down_backend="producer_quantized",backward_overlap=True)
                    y=mxfp8_mlp(*inputs,activation=activation,config=cfg)
                    for a,b in zip((y,*torch.autograd.grad(y,inputs,dy)),expected):
                        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_producer_quantized_parallel_wide_scratch(self):
        from dataclasses import replace
        from rtx import mxfp8_mlp, MXFP8MLPConfig
        from rtx.configs.glu import normalize_glu_config
        torch.manual_seed(390)
        for m,k,h,o in ((128,128,256,128),(17,35,97,41)):
            inputs=tuple((torch.randn(*s,device="cuda",dtype=torch.bfloat16)*.02).requires_grad_()
                         for s in ((m,k),(2*h,k),(o,h)))
            dy=torch.randn(m,o,device="cuda",dtype=torch.bfloat16)
            base=MXFP8MLPConfig()
            y=mxfp8_mlp(*inputs,config=base)
            expected=(y,*torch.autograd.grad(y,inputs,dy))
            for vec in (4,16,32):
                cfg=replace(base,down_backend="producer_quantized",backward_overlap=True,
                            gate_up=normalize_glu_config(base.gate_up,glu_tile=(128,256,4),
                                quant_vec=vec,glu_quant_vec=vec,glu_scratch_swizzle="xor_16b"))
                y=mxfp8_mlp(*inputs,config=cfg)
                for a,b in zip((y,*torch.autograd.grad(y,inputs,dy)),expected):
                    torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_fragment_quantization_and_extra_cooperative_warps(self):
        from dataclasses import replace
        from rtx import mxfp8_mlp, MXFP8MLPConfig
        from rtx.configs.glu import normalize_glu_config
        from rtx.kernels.mxfp8 import normalize_fwd_config
        for m,k,h,o in ((64,128,256,64),(17,35,97,41)):
            inputs=tuple((torch.randn(*s,device="cuda",dtype=torch.bfloat16)*.02).requires_grad_()
                         for s in ((m,k),(2*h,k),(o,h)))
            dy=torch.randn(m,o,device="cuda",dtype=torch.bfloat16)
            for activation in ("silu","gelu","relu"):
                base=MXFP8MLPConfig()
                y=mxfp8_mlp(*inputs,activation=activation,config=base)
                expected=(y,*torch.autograd.grad(y,inputs,dy))
                for tm,tn,cw in ((32,128,8),(64,128,16),(128,128,16)):
                    cfg=replace(base,gate_up=normalize_glu_config(base.gate_up,tile_m=tm,tile_n=tn,
                                cooperative_warps=cw,glu_quant_epilogue="fragment_registers"),
                                down=normalize_fwd_config(base.down,tile_m=tm,cooperative_warps=cw))
                    y=mxfp8_mlp(*inputs,activation=activation,config=cfg)
                    for a,b in zip((y,*torch.autograd.grad(y,inputs,dy)),expected):
                        torch.testing.assert_close(a,b,rtol=0,atol=0,msg=f"shape={(m,k,h,o)} activation={activation} tile={(tm,tn)} warps={cw}")

    def test_32_row_logical_tiles(self):
        from dataclasses import replace
        from rtx import mxfp8_mlp, MXFP8MLPConfig
        from rtx.configs.glu import normalize_glu_config
        from rtx.kernels.mxfp8 import normalize_fwd_config
        for m,k,h,o in ((32,64,128,64),(17,35,97,41)):
            inputs=tuple((torch.randn(*s,device="cuda",dtype=torch.bfloat16)*.02).requires_grad_()
                         for s in ((m,k),(2*h,k),(o,h)))
            dy=torch.randn(m,o,device="cuda",dtype=torch.bfloat16)
            base=MXFP8MLPConfig()
            y=mxfp8_mlp(*inputs,config=base)
            expected=(y,*torch.autograd.grad(y,inputs,dy))
            cfg=replace(base,gate_up=normalize_glu_config(base.gate_up,tile_m=32),
                        down=normalize_fwd_config(base.down,tile_m=32))
            y=mxfp8_mlp(*inputs,config=cfg)
            for a,b in zip((y,*torch.autograd.grad(y,inputs,dy)),expected):
                torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_quantizer_specialization_and_wide_shared_stores(self):
        from dataclasses import replace
        from rtx import mxfp8_mlp,MXFP8MLPConfig
        from rtx.configs.glu import normalize_glu_config
        from rtx.kernels.mxfp8 import normalize_fwd_config
        base=MXFP8MLPConfig()
        for m,k,h,o in ((32,64,128,64),(17,35,97,41)):
            inputs=tuple((torch.randn(*shape,device="cuda",dtype=torch.bfloat16)*.02).requires_grad_()
                         for shape in ((m,k),(2*h,k),(o,h)))
            dy=torch.randn(m,o,device="cuda",dtype=torch.bfloat16)*.1
            y=mxfp8_mlp(*inputs,config=base)
            expected=(y,*torch.autograd.grad(y,inputs,dy))
            for schedule,bits,swizzle in (("combined",16,"none"),("split_a_b",16,"32b"),
                                          ("split_a_b",32,"64b"),("split_a_b",64,"128b")):
                options=dict(quant_vec=8,quant_load_bits=128,quant_schedule=schedule,
                             quant_store_bits=bits,a_swizzle=swizzle,b_swizzle=swizzle)
                config=replace(base,gate_up=normalize_glu_config(base.gate_up,**options),
                               down=normalize_fwd_config(base.down,**options))
                z=mxfp8_mlp(*inputs,config=config)
                actual=(z,*torch.autograd.grad(z,inputs,dy))
                for a,b in zip(actual,expected):
                    torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_packed_down_cluster_reuse(self):
        from dataclasses import replace
        from rtx import mxfp8_mlp,MXFP8MLPConfig
        from rtx.kernels.mxfp8 import normalize_fwd_config
        base=MXFP8MLPConfig()
        x=torch.randn(512,64,device="cuda",dtype=torch.bfloat16)*.2
        gu=torch.randn(512,64,device="cuda",dtype=torch.bfloat16)*.02
        down=torch.randn(512,256,device="cuda",dtype=torch.bfloat16)*.02
        expected=mxfp8_mlp(x,gu,down)
        for operand in ("a","b"):
            for size in (2,4):
                config=replace(base,down=normalize_fwd_config(
                    base.down,cluster_reuse=operand,cluster_size=size,
                    raster="n" if operand=="a" else "m",
                    a_swizzle="none",b_swizzle="none"))
                actual=mxfp8_mlp(x,gu,down,config=config)
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    def test_dead_operand_scratch_reuse(self):
        from dataclasses import replace
        from rtx import mxfp8_mlp, MXFP8MLPConfig
        from rtx.configs.glu import normalize_glu_config
        for m,k,h,o in ((32,64,128,64),(17,35,97,41)):
            inputs=tuple((torch.randn(*s,device="cuda",dtype=torch.bfloat16)*.02).requires_grad_()
                         for s in ((m,k),(2*h,k),(o,h)))
            dy=torch.randn(m,o,device="cuda",dtype=torch.bfloat16)
            base=MXFP8MLPConfig()
            expected=mxfp8_mlp(*inputs,config=base)
            expected=(expected,*torch.autograd.grad(expected,inputs,dy))
            cfg=replace(base,gate_up=normalize_glu_config(base.gate_up,glu_scratch="reuse_operands"))
            y=mxfp8_mlp(*inputs,config=cfg)
            for a,b in zip((y,*torch.autograd.grad(y,inputs,dy)),expected):
                torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_recompute_schedule_parity(self):
        from dataclasses import replace
        from rtx import mxfp8_mlp,MXFP8MLPConfig
        for m,k,h,o in ((32,64,128,64),(17,35,97,41)):
            inputs=tuple((torch.randn(*shape,device="cuda",dtype=torch.bfloat16)*scale).requires_grad_()
                         for shape,scale in (((m,k),.2),((2*h,k),.02),((o,h),.02)))
            dy=torch.randn(m,o,device="cuda",dtype=torch.bfloat16)*.1
            results=[]
            for schedule in ("joint","two_pass"):
                config=replace(MXFP8MLPConfig(),backward_recompute=schedule)
                y=mxfp8_mlp(*inputs,config=config)
                results.append((y,*torch.autograd.grad(y,inputs,dy)))
            for a,b in zip(*results):
                torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_packed_down_vector_loads(self):
        from dataclasses import replace
        from rtx import mxfp8_mlp,MXFP8MLPConfig
        from rtx.kernels.mxfp8 import normalize_fwd_config
        base=MXFP8MLPConfig()
        for m,k,h,o in ((32,64,128,64),(17,35,97,41)):
            x=torch.randn(m,k,device="cuda",dtype=torch.bfloat16)*.2
            gu=torch.randn(2*h,k,device="cuda",dtype=torch.bfloat16)*.02
            down=torch.randn(o,h,device="cuda",dtype=torch.bfloat16)*.02
            expected=mxfp8_mlp(x,gu,down)
            for bits,vec in ((32,2),(64,4),(128,8)):
                config=replace(base,down=normalize_fwd_config(base.down,quant_load_bits=bits,quant_vec=vec))
                actual=mxfp8_mlp(x,gu,down,config=config)
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    def test_forward_backward_reference(self):
        from rtx import mxfp8_mlp
        torch.manual_seed(51)
        for m,k,h,o in ((32,64,128,64),(17,35,97,41),(1,17,1,1)):
            for act in ("silu","gelu","relu"):
                with self.subTest(shape=(m,k,h,o),activation=act):
                    x=(torch.randn(m,k,device="cuda",dtype=torch.bfloat16)*.2).requires_grad_()
                    gu=(torch.randn(2*h,k,device="cuda",dtype=torch.bfloat16)*.02).requires_grad_()
                    down=(torch.randn(o,h,device="cuda",dtype=torch.bfloat16)*.02).requires_grad_()
                    dy=torch.randn(m,o,device="cuda",dtype=torch.bfloat16)*.1
                    expected,grads=reference(x.detach(),gu.detach(),down.detach(),dy,act)
                    actual=mxfp8_mlp(x,gu,down,activation=act)
                    actual.backward(dy)
                    for name,result,ref in zip(("y","dx","dgate_up","ddown"),(actual,x.grad,gu.grad,down.grad),(expected,*grads)):
                        self.assertTrue(result.isfinite().all(),name)
                        error=((result.float()-ref.float()).norm()/ref.float().norm().clamp_min(1e-20)).item()
                        self.assertLess(error,.025,(name,error))

    def test_saved_state_and_partial_gradients(self):
        from rtx import mxfp8_mlp
        x=torch.randn(2,17,35,device="cuda",dtype=torch.bfloat16)*.1
        gu=torch.randn(194,35,device="cuda",dtype=torch.bfloat16)*.02
        down=torch.randn(41,97,device="cuda",dtype=torch.bfloat16)*.02
        for needs in ((1,1,1),(1,0,0),(0,1,0),(0,0,1)):
            values=[t.detach().requires_grad_(bool(n)) for t,n in zip((x,gu,down),needs)]
            saved=[]
            with torch.autograd.graph.saved_tensors_hooks(lambda t:(saved.append(t),t)[1],lambda t:t):
                out=mxfp8_mlp(*values)
            self.assertEqual(len(saved),3)
            self.assertEqual([t.data_ptr() for t in saved],[t.data_ptr() for t in values])
            # Expanded upstream gradient is deliberately noncontiguous.
            out.backward(torch.ones(1,1,41,device="cuda",dtype=torch.bfloat16).expand_as(out))
            for t,n in zip(values,needs):
                self.assertEqual(t.grad is not None,bool(n))

    def test_compile_and_opcheck(self):
        from rtx import MXFP8MLP
        from rtx.mlp import _mlp_op,_config_key,DEFAULT_MLP_CONFIG
        module=MXFP8MLP(35,97,41,device="cuda")
        x=(torch.randn(2,17,35,device="cuda",dtype=torch.bfloat16)*.1).requires_grad_()
        args=(x,module.gate_up_weight,module.down_weight,"silu",_config_key(DEFAULT_MLP_CONFIG))
        torch.library.opcheck(_mlp_op,args)
        expected=module(x)
        grads=torch.autograd.grad(expected.float().sum(),(x,*module.parameters()))
        compiled=torch.compile(module,fullgraph=True,dynamic=False)
        from torch._inductor.utils import run_and_get_code
        result,code=run_and_get_code(compiled,x)
        self.assertIn("torch._rtx_mlp_launchers","\n".join(code))
        actual_grads=torch.autograd.grad(result.float().sum(),(x,*module.parameters()))
        torch.testing.assert_close(result,expected,rtol=0,atol=0)
        for a,b in zip(actual_grads,grads):
            torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_two_forward_device_kernels(self):
        from rtx import MXFP8MLP
        module=MXFP8MLP(64,128,device="cuda")
        x=torch.randn(32,64,device="cuda",dtype=torch.bfloat16)
        with torch.no_grad():
            for _ in range(3):module(x)
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                result=module(x)
                torch.cuda.synchronize()
        kernels=[e for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA]
        self.assertEqual(len(kernels),2,[e.name for e in kernels])
        self.assertTrue(result.isfinite().all())

    def test_training_replay_and_memory(self):
        from rtx import MXFP8MLP
        torch.manual_seed(99)
        module=MXFP8MLP(32,64,device="cuda")
        x=(torch.randn(32,32,device="cuda",dtype=torch.bfloat16)*.5).requires_grad_()
        with torch.no_grad():
            teacher1=torch.randn(128,32,device="cuda")*.08
            teacher2=torch.randn(32,64,device="cuda")*.08
            a,b=(x.float()@teacher1.T).chunk(2,-1)
            target=(torch.nn.functional.silu(a)*b)@teacher2.T
        compiled=torch.compile(module,fullgraph=True,dynamic=False)
        optimizer=torch.optim.AdamW(module.parameters(),lr=.005,fused=True,capturable=True)
        def step():
            optimizer.zero_grad(set_to_none=False)
            if x.grad is not None:x.grad.zero_()
            out=compiled(x)
            loss=(out.float()-target).square().mean()
            loss.backward()
            optimizer.step()
            return loss
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):step()
        stream.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):loss=step()
        graph.replay();torch.cuda.synchronize()
        initial=loss.item();allocated=torch.cuda.memory_allocated()
        for _ in range(30):graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(torch.cuda.memory_allocated(),allocated)
        self.assertTrue(loss.isfinite())
        self.assertLess(loss.item(),initial)
        for state in optimizer.state.values():
            self.assertEqual(state["exp_avg"].dtype,torch.bfloat16)
            self.assertEqual(state["exp_avg_sq"].dtype,torch.bfloat16)

    def test_long_reduction_and_accumulation(self):
        from rtx import mxfp8_mlp
        x=(torch.randn(2048,64,device="cuda",dtype=torch.bfloat16)*.2).requires_grad_()
        gu=(torch.randn(256,64,device="cuda",dtype=torch.bfloat16)*.02).requires_grad_()
        down=(torch.randn(64,128,device="cuda",dtype=torch.bfloat16)*.02).requires_grad_()
        dy=torch.randn(2048,64,device="cuda",dtype=torch.bfloat16)*.1
        y=mxfp8_mlp(x,gu,down)
        y.backward(dy,retain_graph=True)
        first=[t.grad.clone() for t in (x,gu,down)]
        _,expected=reference(x.detach(),gu.detach(),down.detach(),dy,"silu")
        for a,b in zip(first,expected):
            error=((a.float()-b.float()).norm()/b.float().norm()).item()
            self.assertLess(error,.03)
        y.backward(dy)
        for t,g in zip((x,gu,down),first):
            torch.testing.assert_close(t.grad,(g+g),rtol=0,atol=0)

    def test_streams_weight_updates_and_version_checks(self):
        from rtx import MXFP8MLP
        module=MXFP8MLP(35,97,41,device="cuda")
        inputs=[torch.randn(m,35,device="cuda",dtype=torch.bfloat16)*.2 for m in (17,33)]
        expected=[module(x).detach() for x in inputs]
        streams=[torch.cuda.Stream(),torch.cuda.Stream()]
        outputs=[]
        for stream,x in zip(streams,inputs):
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                outputs.append(module(x))
        for stream in streams:stream.synchronize()
        for a,b in zip(outputs,expected):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        pending=module(inputs[0])
        with torch.no_grad():module.down_weight.mul_(2)
        with self.assertRaisesRegex(RuntimeError,"modified by an inplace operation"):
            pending.sum().backward()
        with torch.no_grad():updated=module(inputs[0])
        torch.testing.assert_close(updated,expected[0]*2,rtol=0,atol=0)
        replica=MXFP8MLP(35,97,41,device="cuda")
        replica.load_state_dict(module.state_dict())
        torch.testing.assert_close(replica(inputs[0]),updated,rtol=0,atol=0)

    def test_zero_input_and_cache_rebuild(self):
        from rtx import MXFP8MLP
        from rtx.mlp import _clear_runtime_caches
        module=MXFP8MLP(35,97,41,device="cuda")
        x=torch.zeros(17,35,device="cuda",dtype=torch.bfloat16,requires_grad=True)
        compiled=torch.compile(module,fullgraph=True,dynamic=False)
        out=compiled(x)
        grads=torch.autograd.grad(out,(x,*module.parameters()),torch.ones_like(out))
        for t in (out,*grads):
            self.assertEqual(torch.count_nonzero(t).item(),0)
        torch.cuda.synchronize()
        released=_clear_runtime_caches()
        self.assertGreater(released["launchers"],0)
        torch.testing.assert_close(compiled(x),out,rtol=0,atol=0)


if __name__=="__main__":
    unittest.main()
