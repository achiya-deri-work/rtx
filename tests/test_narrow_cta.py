"""N=64 scale partitions and tall weight-reuse search coordinates."""
from dataclasses import replace
import pytest
import torch

from rtx.kernels.mxfp8 import MXFP8Problem, normalize_fwd_config, FWD_SEARCH_SPACE
from rtx.configs.glu import normalize_glu_config, glu_rejection


def test_narrow_coordinates_are_coupled_and_reject_wrong_n_partition():
    for m in (64,128,256,384,512):
        c=normalize_fwd_config(weight_reuse_tile=(m,"scalar",1,4))
        assert (c.tile_m,c.tile_n,c.atom_layout_n,c.raster,c.grid_swizzle)==(m,64,2,"m",4)
        assert c.implementation_rejection(MXFP8Problem(4096,256,128)) is None
        assert replace(c,atom_layout_n=1).implementation_rejection(MXFP8Problem(4096,256,128))
        assert 64 in FWD_SEARCH_SPACE['tile_n']
    bad=normalize_fwd_config(weight_reuse_tile=(256,"cpasync",2,1))
    assert 'quarantined' in bad.implementation_rejection(MXFP8Problem(512,256,128))


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")
@pytest.mark.parametrize("tm,engine,stages",[
    (64,"scalar",1),(128,"scalar",1),(256,"scalar",1),
    (384,"scalar",1),(512,"scalar",1),
    (128,"cpasync",2),
    (128,"tma",3),(256,"tma",2),
])
def test_narrow_fused_linear_and_glu(tm,engine,stages):
    from rtx.fp8 import mxfp8_linear
    from rtx.kernels.mxfp8_glu import compile_mxfp8_glu
    from tests.test_mxfp8_glu import reference
    from tests.test_mxfp8_fwd import _reference_to_mx
    from benchmarks.diagnose_mxfp8_rounding import round_fp64_to_bf16
    torch.manual_seed(1963)
    m,k,h=2*tm+17,256,161
    if engine == 'cpasync':
        m,h=2*tm,160  # This backend's existing contract requires full tiles.
    x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)*.1
    w=torch.randn(2*h,k,device='cuda',dtype=torch.bfloat16)*.1
    c=normalize_fwd_config(weight_reuse_tile=(tm,engine,stages,2))
    problem=MXFP8Problem(m,2*h,k)
    assert c.implementation_rejection(problem) is None
    def dq(t):
        q,s=_reference_to_mx(t)
        return q.float()*s.float().repeat_interleave(32,-1)
    y=mxfp8_linear(x,w,config=c,autotune='off',backend='fused')
    # FP32 cuBLAS can round a dot onto a BF16 midpoint before the final cast.
    # This seed's (632,12) is one such cell: the kernel agrees with the exact
    # dot, not the double-rounded FP32 oracle. Keep the original FP32 gate;
    # a disagreement is accepted ONLY if it equals the independently correctly
    # rounded exact dot. Requiring exact dots everywhere would incorrectly
    # demand FP64 accumulation from a deliberately FP32-accumulating kernel.
    a,b=dq(x),dq(w)
    expected=(a@b.T).bfloat16()
    bad=~torch.isclose(y,expected,rtol=.004,atol=2e-6)
    if bool(bad.any()):
        exact=a.cpu().double()@b.cpu().double().T
        rounded=round_fp64_to_bf16(exact).to(y.device)
        torch.testing.assert_close(y[bad],rounded[bad],rtol=0,atol=0)
    gc=normalize_glu_config(weight_reuse_tile=(tm,engine,stages,2))
    assert glu_rejection(problem,gc,'mxfp8') is None
    width=(h+31)//32*32
    for act in ('silu','gelu'):
        out=torch.empty(m,width,device='cuda',dtype=torch.float8_e4m3fn)
        scales=torch.empty(m,width//32,device='cuda',dtype=torch.uint8)
        fn=compile_mxfp8_glu(problem,gc,act,'mxfp8')
        fn(x,w,out,scales,scales,scales,scales)
        qr,sr=_reference_to_mx(reference(x,w,act))
        torch.testing.assert_close(scales,sr.view(torch.uint8),rtol=0,atol=0)
        torch.testing.assert_close(out.float(),qr.float(),rtol=.13,atol=.001)
        assert torch.isfinite(out.float()).all()


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")
@pytest.mark.parametrize("tm,stages,tiles",[(128,1,1),(128,2,2),(128,3,1),(256,1,1),(256,2,2),(384,1,1),(512,1,1)])
def test_narrow_prequant_gemm(tm,stages,tiles):
    from rtx.configs.mxfp8 import MXFP8GemmConfig
    from rtx.kernels.mxfp8_gemm import compile_mxfp8_gemm
    from tests.test_mxfp8_fwd import _reference_to_mx
    torch.manual_seed(1964)
    m,n,k=2*tm+17,193,256
    x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)*.1
    w=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)*.1
    qx,sx=_reference_to_mx(x);qw,sw=_reference_to_mx(w)
    c=MXFP8GemmConfig(tile_m=tm,tile_n=64,atom_layout_m=4 if tm==128 else 8,
        atom_layout_n=2,stages=stages,scale_layout='row_major',scale_role='consumers',
        epilogue='direct',store_vec=1,raster='m',grid_swizzle=2,
        tiles_per_cta=tiles,tile_locality='same_b')
    p=MXFP8Problem(m,n,k)
    assert c.rejection(p) is None
    out=torch.empty(m,n,device='cuda',dtype=torch.bfloat16)
    compile_mxfp8_gemm(p,c)(qx,qw,sx,sw,out)
    ref=(qx.float()*sx.float().repeat_interleave(32,-1))@(qw.float()*sw.float().repeat_interleave(32,-1)).T
    torch.testing.assert_close(out,ref.bfloat16(),rtol=.004,atol=2e-6)
    assert torch.isfinite(out).all()


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")
def test_narrow_recomputed_mlp_and_backward():
    import rtx
    from rtx.kernels.mxfp8_bwd import DEFAULT_FUSED_MXFP8_BWD_CONFIG
    from rtx.autotune.mlp import reference
    torch.manual_seed(1965)
    c=normalize_fwd_config(weight_reuse_tile=(256,'scalar',1,2))
    gc=normalize_glu_config(weight_reuse_tile=(256,'scalar',1,2))
    b=DEFAULT_FUSED_MXFP8_BWD_CONFIG
    b=replace(b,dx=replace(b.dx,fused=c),dw=replace(b.dw,fused=c))
    policy=rtx.MXFP8MLPConfig(gate_up=gc,down=c,gate_backward=b,down_backward=b)
    layer=rtx.MXFP8MLP(128,129,97,config=policy,device='cuda')
    x=(torch.randn(513,128,device='cuda',dtype=torch.bfloat16)*.1).requires_grad_()
    dy=torch.randn(513,97,device='cuda',dtype=torch.bfloat16)*.1
    inputs=(x,layer.gate_up_weight,layer.down_weight)
    fn=torch.compile(layer,fullgraph=True,dynamic=False)
    y=fn(x)
    grads=torch.autograd.grad(y,inputs,dy)
    with torch.no_grad():expected=reference(*inputs,dy,'silu')
    for actual,ref in zip((y,*grads),expected):
        assert torch.isfinite(actual).all()
        delta=actual.float()-ref.float()
        nrms=delta.norm()/ref.float().norm().clamp_min(1e-12)
        assert nrms<.025,(float(nrms),float(delta.abs().max()))
