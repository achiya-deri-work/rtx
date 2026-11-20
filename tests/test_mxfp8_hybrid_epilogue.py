"""Bulk TMA stores on full tiles and masked scalar stores only on boundaries."""
from dataclasses import replace

import pytest
import torch

from rtx.configs.mxfp8 import MXFP8GemmConfig
from rtx.kernels.mxfp8 import MXFP8Problem


def schedule(**kwargs):
    return replace(MXFP8GemmConfig(atom_layout_m=4,stages=2,epilogue='hybrid'),**kwargs)


@pytest.mark.parametrize('shape',[(130,136,128),(257,192,256),(17568,512,384)])
def test_hybrid_accepts_tails_but_accounts_for_full_output_smem(shape):
    p=MXFP8Problem(*shape);c=schedule()
    assert c.rejection(p) is None
    assert replace(c,epilogue='tma').rejection(p)=='TMA epilogue requires full M/N tiles'


@pytest.mark.parametrize('shape',[(130,132,128),(63,128,128),(128,63,128)])
def test_hybrid_rejects_unaligned_strides_or_no_interior(shape):
    assert schedule().rejection(MXFP8Problem(*shape)) is not None


def test_hybrid_never_changes_fp32_split_workspace_output_contract():
    from rtx.configs.mxfp8 import MXFP8QuantConfig
    from rtx.kernels.mxfp8_bwd import MXFP8BwdMatmulConfig
    c=MXFP8BwdMatmulConfig(backend='decomposed',gemm=schedule(),
        quant_a=MXFP8QuantConfig(scale_layout='row_major'),
        reduction='split_fp32_workspace',split_reduction=2,reduction_tile=128,
        workspace_epilogue='tree')
    assert 'requires direct GEMM output' in c.implementation_rejection(MXFP8Problem(256,256,256))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('shape,tiles,buffers',[
    ((130,136,128),1,1),((257,192,256),2,2),((256,128,128),1,1),
    ((769,192,128),4,2),((769,200,128),8,3),((130,136,160),1,1),
])
def test_hybrid_matches_direct_and_independent_oracle_with_guarded_output(shape,tiles,buffers):
    from rtx.kernels.mxfp8_gemm import compile_mxfp8_gemm
    m,n,k=shape;torch.manual_seed(19371)
    a=(torch.randn(m,k,device='cuda')*2).to(torch.float8_e4m3fn)
    b=(torch.randn(n,k,device='cuda')*2).to(torch.float8_e4m3fn)
    sa=torch.randint(123,132,(m,k//32),device='cuda',dtype=torch.uint8).view(torch.float8_e8m0fnu)
    sb=torch.randint(123,130,(n,k//32),device='cuda',dtype=torch.uint8).view(torch.float8_e8m0fnu)
    c=schedule(tiles_per_cta=tiles,epilogue_stages=buffers)
    if buffers>1:c=replace(c,tile_m=64,atom_layout_m=2)
    if buffers>2:c=replace(c,tile_n=64,atom_layout_n=2)
    p=MXFP8Problem(m,n,k)
    hybrid=compile_mxfp8_gemm(p,c)
    direct=compile_mxfp8_gemm(p,replace(c,epilogue='direct',epilogue_stages=1,store_vec=1))
    guard=torch.full((m*n+32,),-123.,device='cuda',dtype=torch.bfloat16)
    output=guard[16:-16].view(m,n);reference=torch.empty_like(output)
    direct(a,b,sa,sb,reference);hybrid(a,b,sa,sb,output)
    torch.cuda.synchronize()
    assert torch.equal(output,reference)
    da=a.float()*sa.float().repeat_interleave(32,dim=1)
    db=b.float()*sb.float().repeat_interleave(32,dim=1)
    oracle=(da@db.T).bfloat16()
    delta=(output.float()-oracle.float())
    assert bool(output.isfinite().all())
    assert float(delta.norm()/oracle.float().norm().clamp_min(1e-12))<.003
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):hybrid(a,b,sa,sb,output)
    stream.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):hybrid(a,b,sa,sb,output)
    for _ in range(16):graph.replay()
    stream.synchronize()
    assert torch.equal(output,reference)
    # A captured tail must write fresh results on replay, not preserve stale
    # masked lanes or touch the canary surrounding the logical output.
    a.copy_((a.float()*.5).to(a.dtype));direct(a,b,sa,sb,reference)
    graph.replay();torch.cuda.synchronize()
    assert torch.equal(output,reference)
    assert bool((guard[:16]==-123).all()) and bool((guard[-16:]==-123).all())
    graph.reset()
