"""Packed vector stores must preserve bytes, including logical padding."""
from dataclasses import replace
import pytest
import torch
import rtx


def test_store_width_legality():
    assert rtx.NVFP4QuantConfig(packed_store_bits=32).rejection(128,128)
    assert rtx.NVFP4QuantConfig(values_per_lane=8,packed_store_bits=32).rejection(7,33) is None
    assert rtx.NVFP4QuantConfig(values_per_lane=16,packed_store_bits=16).rejection(128,128)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('shape',[(7,33),(128,256),(257,385)])
@pytest.mark.parametrize('policy',['amax','four_over_six','mse5','lsq1'])
@pytest.mark.parametrize('vpl',[8,16])
@pytest.mark.parametrize('decode',['lut','native'])
def test_vector_store_matches_scalar_bytes(shape,policy,vpl,decode):
    if policy=='amax' and decode=='native':
        pytest.skip('adaptive decoder is inactive for amax')
    torch.manual_seed(430)
    x=torch.randn(*shape,device='cuda',dtype=torch.bfloat16)
    x[0]=0
    x[-1,::17]*=20
    outer=torch.tensor([0.0137],device='cuda',dtype=torch.float32)
    cfg=rtx.NVFP4QuantConfig(values_per_lane=vpl,load_bits=128,
                            block_scale_selection=policy,adaptive_reuse_packed=True)
    scalar=rtx.quantize_nvfp4(x,tensor_scale=outer,config=cfg)
    vector=rtx.quantize_nvfp4(x,tensor_scale=outer,config=replace(cfg,packed_store_bits=32,adaptive_decode=decode))
    torch.testing.assert_close(vector.qdata.view(torch.uint8),scalar.qdata.view(torch.uint8),rtol=0,atol=0)
    torch.testing.assert_close(vector.scale.float(),scalar.scale.float(),rtol=0,atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('owner',['cta','cta_cached','warp'])
def test_regional_native_vector_store_matches_scalar(owner):
    from rtx.fp4 import _make_jit_region_dynamic_runner
    from rtx.nvfp4_inference_autotune import preferred_jit_row_region_config
    problem=rtx.NVFP4Problem(256,256,256)
    cfg=preferred_jit_row_region_config(problem)
    cfg=replace(cfg,region_ownership=owner,x_scale_region_rows=8,weight_scale_region_rows=8,
                tensor_scale_mode='exact_four_over_six',quant=replace(cfg.quant,
                    values_per_lane=16,load_bits=128,adaptive_reuse_packed=True,
                    block_scale_selection='four_over_six'))
    x=torch.randn(256,256,device='cuda',dtype=torch.bfloat16)
    w=torch.randn_like(x)
    results=[]
    for bits in (8,32):
        runner=_make_jit_region_dynamic_runner(problem,
            replace(cfg,quant=replace(cfg.quant,packed_store_bits=bits,
                adaptive_decode='native' if bits==32 else 'lut')),x.device)
        out=torch.empty_like(x)
        runner(x,w,out)
        results.append([t.clone() for t in (runner.qx,runner.qw,runner.sx,runner.sw,runner.region_scales,out)])
    for a,b in zip(*results):
        torch.testing.assert_close(a.view(torch.uint8),b.view(torch.uint8),rtol=0,atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_vector_store_public_compiled_forward_backward():
    from rtx.nvfp4_inference_autotune import preferred_jit_row_region_config
    torch.manual_seed(431)
    cfg=preferred_jit_row_region_config(rtx.NVFP4Problem(128,256,256))
    cfg=replace(cfg,x_scale_region_rows=8,weight_scale_region_rows=8,
                tensor_scale_mode='exact_four_over_six',quant=replace(cfg.quant,
                    values_per_lane=16,load_bits=128,adaptive_reuse_packed=True,
                    block_scale_selection='four_over_six'))
    x=torch.randn(128,256,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    w=torch.randn(256,256,device='cuda',dtype=torch.bfloat16)*.02
    grad=torch.randn_like(x)
    results=[]
    for bits in (8,32):
        layer=rtx.NVFP4Linear(256,256,device='cuda',autotune='off',scaling='jit_row_region',
            dynamic_config=replace(cfg,quant=replace(cfg.quant,packed_store_bits=bits,
                adaptive_decode='native' if bits==32 else 'lut')),
            scale_config=rtx.NVFP4ScaleConfig(tensor_scale_mode='exact_four_over_six'),
            backward_config=rtx.DEFAULT_MXFP8_BWD_CONFIG)
        with torch.no_grad(): layer.weight.copy_(w)
        compiled=torch.compile(layer,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
        y=compiled(x)
        dx,dw=torch.autograd.grad(y,(x,layer.weight),grad)
        assert all(bool(t.isfinite().all()) for t in (y,dx,dw))
        results.append((y,dx,dw))
    for a,b in zip(*results):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
