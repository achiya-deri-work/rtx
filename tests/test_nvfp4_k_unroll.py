"""Outer K-loop unrolling preserves accumulation and circular pipeline phases."""
from dataclasses import asdict, replace
import pytest
import torch
import rtx
from rtx.nvfp4_inference_autotune import preferred_jit_row_region_config


def test_unroll_legality_and_search_axis():
    from rtx.nvfp4_inference_autotune import _gemm_axes, NVFP4_JIT_ROW_REGION_SEARCH_SPACE
    problem=rtx.NVFP4Problem(256,256,256)
    base=rtx.NVFP4GemmConfig()
    for value in (0,5,-1,True,1.0):
        assert "k_tile_unroll" in replace(base,k_tile_unroll=value).rejection(problem)
    expected=tuple({"gemm":{"k_tile_unroll":v}} for v in (1,2,3,4,6,8,12))
    assert _gemm_axes()["gemm_k_unroll"] == expected
    assert NVFP4_JIT_ROW_REGION_SEARCH_SPACE["gemm_k_unroll"] == expected
    from rtx.nvfp4_inference_autotune import dynamic_config_from_dict
    original=preferred_jit_row_region_config(problem)
    candidate=replace(original,gemm=replace(original.gemm,k_tile_unroll=6))
    assert dynamic_config_from_dict(asdict(candidate)) == candidate
    legacy=asdict(original)
    legacy["gemm"].pop("k_tile_unroll")
    assert dynamic_config_from_dict(legacy).gemm.k_tile_unroll == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("shape", [(256,256,256), (384,384,768), (384,256,384)])
@pytest.mark.parametrize("atom_m", [2,4])
@pytest.mark.parametrize("stages", [1,3])
@pytest.mark.parametrize("unroll", [2,3,6])
def test_unrolled_pipeline_bitwise(shape, atom_m, stages, unroll):
    from rtx.fp4 import _make_jit_region_dynamic_runner
    torch.manual_seed(745)
    problem=rtx.NVFP4Problem(*shape)
    base=preferred_jit_row_region_config(problem)
    base=replace(base, x_scale_region_rows=5, weight_scale_region_rows=4,
        tensor_scale_mode="exact_four_over_six",
        quant=replace(base.quant,block_scale_selection="four_over_six",scale_layout="mma128"),
        gemm=replace(base.gemm,tile_m=128,tile_n=128,epilogue="tma",store_vec=2,
            stages=stages,epilogue_stages=1,regional_scale_epilogue="fragment_registers",
            regional_epilogue_values=2,atom_layout_m=atom_m,atom_layout_n=2,
            scale_layout="mma128",scale_role="tma",scale_schedule="before_wait",
            tiles_per_cta=4,tile_locality="same_a",persistent_waves=0))
    candidate=replace(base,gemm=replace(base.gemm,k_tile_unroll=unroll))
    assert candidate.rejection(problem) is None
    m,n,k=shape
    x=torch.randn(m,k,device="cuda",dtype=torch.bfloat16)
    w=torch.randn(n,k,device="cuda",dtype=torch.bfloat16)*.02
    x[::5]*=8
    w[::7]*=.125
    a=torch.empty(m,n,device=x.device,dtype=x.dtype)
    b=torch.empty_like(a)
    reference=_make_jit_region_dynamic_runner(problem,base,x.device)
    optimized=_make_jit_region_dynamic_runner(problem,candidate,x.device)
    for _ in range(3):
        reference(x,w,a)
        optimized(x,w,b)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        x.mul_(-1.5)
        w.mul_(.75)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_unroll_public_compiled_backward():
    torch.manual_seed(431)
    cfg=preferred_jit_row_region_config(rtx.NVFP4Problem(128,256,256))
    cfg=replace(cfg,x_scale_region_rows=8,weight_scale_region_rows=8,
        tensor_scale_mode="exact_four_over_six",quant=replace(cfg.quant,
            block_scale_selection="four_over_six"))
    x=torch.randn(128,256,device="cuda",dtype=torch.bfloat16,requires_grad=True)
    w=torch.randn(256,256,device="cuda",dtype=torch.bfloat16)*.02
    grad=torch.randn_like(x)
    results=[]
    for unroll in (1,6):
        torch._dynamo.reset()
        layer=rtx.NVFP4Linear(256,256,device="cuda",autotune="off",scaling="jit_row_region",
            dynamic_config=replace(cfg,gemm=replace(cfg.gemm,k_tile_unroll=unroll)),
            scale_config=rtx.NVFP4ScaleConfig(tensor_scale_mode="exact_four_over_six"),
            backward_config=rtx.DEFAULT_MXFP8_BWD_CONFIG)
        with torch.no_grad():
            layer.weight.copy_(w)
        compiled=torch.compile(layer,fullgraph=True,dynamic=False,options={"triton.cudagraphs":False})
        y=compiled(x)
        dx,dw=torch.autograd.grad(y,(x,layer.weight),grad)
        assert all(bool(t.isfinite().all()) for t in (y,dx,dw))
        results.append((y,dx,dw))
    for a,b in zip(*results):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
