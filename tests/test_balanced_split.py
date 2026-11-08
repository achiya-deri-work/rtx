"""Balanced K-tile coverage, policy identity, and exact FP32 workspace contracts."""
from dataclasses import replace

import pytest
import torch

from rtx._bwd_search import balanced_dw_candidates, materialized_bwd_seed, split_leg
from rtx.bwd_autotune import bwd_config_from_dict, bwd_config_to_dict
from rtx.kernels.mxfp8 import MXFP8Problem
from rtx.kernels.mxfp8_bwd import DEFAULT_FUSED_MXFP8_BWD_CONFIG as BASE


@pytest.mark.parametrize('length',[1024,11296,11712,65568,68320,208192])
def test_balanced_ranges_cover_every_tile_once(length):
    tiles=(length+127)//128
    for parts in (2,4,8,16,32,64,128):
        if parts>tiles:continue
        ranges=[(i*tiles//parts,(i+1)*tiles//parts) for i in range(parts)]
        assert ranges[0][0]==0 and ranges[-1][1]==tiles
        assert all(a<b for a,b in ranges)
        assert all(ranges[i][1]==ranges[i+1][0] for i in range(parts-1))
        assert max(b-a for a,b in ranges)-min(b-a for a,b in ranges)<=1
        p=MXFP8Problem(length,512,128)
        c=materialized_bwd_seed(p,BASE)
        c=replace(c,dw=split_leg(c.dw,length,parts,partition='balanced'))
        assert c.implementation_rejection(p) is None
        assert bwd_config_from_dict(bwd_config_to_dict(c))==c.normalized()


def test_partition_contracts_and_legacy_identity():
    p=MXFP8Problem(11296,256,128)
    c=materialized_bwd_seed(p,BASE)
    assert 'split_partition' not in bwd_config_to_dict(c)['dw']
    balanced=replace(c,dw=split_leg(c.dw,p.m,64,partition='balanced'))
    assert balanced.implementation_rejection(p) is None
    bad=replace(balanced,dw=replace(balanced.dw,reduction_tile=256))
    assert 'zero' in bad.implementation_rejection(p)
    bad=replace(balanced,dw=replace(balanced.dw,backend='fused'))
    assert 'decomposed' in bad.implementation_rejection(p)
    bad=replace(balanced,dw=replace(balanced.dw,split_reduction=128))
    assert 'one K tile' in bad.implementation_rejection(p)
    bad=replace(balanced,dw=replace(balanced.dw,split_reduction=63))
    assert 'power of two' in bad.implementation_rejection(p)


def test_shared_search_preserves_numeric_policy_and_reaches_large_splits():
    from rtx._bwd_search import backward_search_space
    from rtx.bwd_autotune import BWD_SEARCH_SPACE, update_bwd_config
    p=MXFP8Problem(208192,512,128)
    c=materialized_bwd_seed(p,BASE)
    qa=replace(c.dw.quant_a,quant_math='bf16x2',quant_amax='bf16_bits')
    c=replace(c,dw=replace(c.dw,quant_a=qa,quant_b=qa))
    axes=backward_search_space(p,BWD_SEARCH_SPACE)
    assert backward_search_space(p,axes)==axes
    variants=[update_bwd_config(c,move) for move in axes['dw_balanced']]
    assert {64,128}<={v.dw.split_reduction for v in variants}
    assert all(v.implementation_rejection(p) is None for v in variants)
    assert all(v.dw.quant_a.quant_math=='bf16x2' and v.dw.quant_a.quant_amax=='bf16_bits'
               and v.dw.resolved_quant_b().quant_math=='bf16x2' for v in variants)
    assert all(v.dw.split_reduction*p.n*p.k*4<=128*1024**2 for v in variants)
    assert balanced_dw_candidates(p,c,36)
    distinct=replace(c,dw=replace(c.dw,quant_b=replace(qa,quant_math='fp32',quant_amax='fp32')))
    assert all(v.dw.quant_a.quant_math=='bf16x2' and v.dw.resolved_quant_b().quant_math=='fp32'
        for v in balanced_dw_candidates(p,distinct,36))
    assert all(update_bwd_config(distinct,move).dw.resolved_quant_b().quant_math=='fp32'
        for move in axes['dw_balanced'])
    query=MXFP8Problem(22528,1024,256)
    assert {64,128}<={v.dw.split_reduction for v in balanced_dw_candidates(query,BASE)}
    from rtx._bwd_search import backward_structural_seeds,backward_coverage
    seeds=backward_structural_seeds(p,c,sm_count=170)
    assert seeds[1].dw.split_partition=='balanced'
    assert backward_coverage([seeds[1]],p)['dw'][0]['grid_ctas']>=170
    assert all(v.dw.split_partition=='uniform' for v in
        backward_structural_seeds(MXFP8Problem(1024,128,64),BASE))


cuda=pytest.mark.skipif(not torch.cuda.is_available(),reason='assigned SM12x CUDA required')


def test_nvfp4_mlp_uses_shared_partition_domain_and_roundtrip():
    from rtx.autotune.nvfp4_mlp import resolved,search_space,update
    from rtx.configs.nvfp4_mlp import NVFP4MLPConfig
    m,k,h,o=68320,256,512,256
    c=resolved(NVFP4MLPConfig(),m,k,h,o)
    axes=search_space(m,k,h,o)
    for role in ('gate_backward','down_backward'):
        axis=f'{role}.dw_balanced'
        moves=[v for v in axes[axis] if v['dw']['split_reduction']==64]
        assert moves
        c=update(c,axis,moves[0])
    assert c.rejection(m,k,h,o) is None
    assert NVFP4MLPConfig.from_dict(c.to_dict())==c


@cuda
@pytest.mark.parametrize('parts',[64,128])
@pytest.mark.parametrize('algorithm',['serial','tree','persistent_tree'])
def test_large_workspace_reducer_includes_every_partial_and_replays(parts,algorithm):
    from rtx.kernels.mxfp8_reduce import compile_mxfp8_workspace_reduce
    m,n=3,17
    workspace=(torch.arange(parts*m*n,device='cuda')%17-8).float()
    out=torch.empty((m,n),device='cuda',dtype=torch.bfloat16)
    fn=compile_mxfp8_workspace_reduce(m,n,parts,algorithm=algorithm,threads=128,vector=1,persistent_waves=1)
    fn(workspace,out)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):fn(workspace,out)
    for iteration in range(3):
        workspace[-m*n:].add_(1)
        graph.replay()
        want=workspace.reshape(parts,m,n).double().sum(0).bfloat16()
        torch.testing.assert_close(out,want,rtol=0,atol=0)
    graph.reset()


def quantized(tensor):
    width=tensor.shape[1]
    blocks=torch.nn.functional.pad(tensor.float(),(0,(-width)%32)).reshape(tensor.shape[0],-1,32)
    scale=torch.exp2(torch.ceil(torch.log2((blocks.abs().amax(-1,keepdim=True)/448).clamp_min(2.**-126))))
    return ((blocks/scale).to(torch.float8_e4m3fn).float()*scale).flatten(1)[:,:width]


@cuda
@pytest.mark.parametrize('rows,parts',[(11296,64),(11712,64),(65568,128),(208192,128)])
def test_balanced_dw_independent_quantized_oracle_and_current_input_graph(rows,parts,record_property):
    from rtx import fp8_bwd as b
    torch.manual_seed(81108)
    torch.set_float32_matmul_precision('highest')
    n,k=128,64
    p=MXFP8Problem(rows,n,k)
    c=materialized_bwd_seed(p,BASE)
    c=replace(c,dw=replace(split_leg(c.dw,rows,parts,partition='balanced'),workspace_epilogue='tree'))
    x=torch.randn(rows,k,device='cuda',dtype=torch.bfloat16)*.2
    w=torch.empty(n,k,device='cuda',dtype=torch.bfloat16)
    dy=torch.randn(n,rows,device='cuda',dtype=torch.bfloat16).t()
    out=torch.empty_like(w)
    key=b._intern_bwd_config(c)
    def run():b._launch_dw_out(dy,x,w,key,out)
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        run();run()
        with torch.cuda.graph(graph,stream=stream):run()
    torch.cuda.current_stream().wait_stream(stream)
    for iteration in range(2):
        x.add_(.01);dy.add_(.02)
        graph.replay()
        want=(quantized(dy.T)@quantized(x.T).T).bfloat16()
        assert bool(out.isfinite().all())
        difference=out.float()-want.float()
        nrms=float(difference.norm()/want.float().norm().clamp_min(1e-12))
        record_property(f'iteration_{iteration}_nrms',nrms)
        record_property(f'iteration_{iteration}_max_abs',float(difference.abs().max()))
        record_property(f'iteration_{iteration}_rms_error',float(difference.square().mean().sqrt()))
        record_property(f'iteration_{iteration}_reference_rms',float(want.float().square().mean().sqrt()))
        assert nrms<.002,(rows,parts,nrms)
    torch.cuda.synchronize()
    live=torch.cuda.memory_allocated()
    for _ in range(8):graph.replay()
    torch.cuda.synchronize()
    record_property('replay_allocated_delta_bytes',torch.cuda.memory_allocated()-live)
    assert torch.cuda.memory_allocated()==live
    from rtx.runtime import clear_runtime_caches
    retained=out.clone()
    clear_runtime_caches(device=x.device,preserve_compiled=True)
    graph.replay()
    torch.testing.assert_close(out,retained,rtol=0,atol=0)
    graph.reset()


@cuda
def test_complete_compiled_recomputed_mlp_balanced_all_gradients():
    from rtx.autotune.mlp import reference,make_mlp_adapter,mlp_structural_seeds
    from rtx.mlp import MXFP8MLPConfig,mxfp8_mlp
    torch.manual_seed(81109);torch.set_float32_matmul_precision('highest')
    m,k,h,o=32800,128,256,128
    gate=materialized_bwd_seed(MXFP8Problem(m,2*h,k),BASE,4)
    down=materialized_bwd_seed(MXFP8Problem(m,o,h),BASE,4)
    c=replace(MXFP8MLPConfig(),gate_backend='prequant',recompute_backend='prequant',down_backend='prequant',
        gate_backward=replace(gate,dw=split_leg(gate.dw,m,64,partition='balanced')),
        down_backward=replace(down,dw=split_leg(down.dw,m,128,partition='balanced')))
    inputs=tuple((torch.randn(*shape,device='cuda',dtype=torch.bfloat16)*scale).requires_grad_()
        for shape,scale in (((m,k),.2),((2*h,k),.02),((o,h),.02)))
    dy=torch.randn(m,o,device='cuda',dtype=torch.bfloat16)*.1
    adapter=make_mlp_adapter(*inputs,initial=c,phase='train',warmup=1,samples=3,calls_per_sample=1)
    seeds=mlp_structural_seeds(adapter,'mxfp8','train')
    assert any(s.gradient_quantization==s.hidden_quantization=='recompute' for s in seeds[:3])
    assert any(s.gate_backward.dw.split_partition==s.down_backward.dw.split_partition=='balanced'
        for s in seeds[:3])
    fn=torch.compile(lambda *a:mxfp8_mlp(*a,config=c),fullgraph=True,dynamic=False)
    y=fn(*inputs);grads=torch.autograd.grad(y,inputs,dy)
    with torch.no_grad():want=reference(*inputs,dy,'silu')
    for a,b in zip((y,*grads),want):
        assert bool(a.isfinite().all())
        error=float((a.detach().float()-b.float()).norm()/b.float().norm().clamp_min(1e-12))
        assert error<.025,error


@cuda
def test_nvfp4_compiled_mlp_with_shared_balanced_mxfp8_backward():
    from rtx.autotune.nvfp4_mlp import resolved
    from rtx.configs.nvfp4_mlp import NVFP4MLPConfig
    from rtx._nvfp4_mlp_reference import reference
    from rtx.fp4_mlp import nvfp4_mlp
    torch.manual_seed(81111);torch.set_float32_matmul_precision('highest')
    m,k,h,o=8224,64,128,64
    c=resolved(NVFP4MLPConfig(),m,k,h,o)
    c=replace(c,gate_backward=replace(c.gate_backward,
        dw=split_leg(c.gate_backward.dw,m,64,partition='balanced')),
        down_backward=replace(c.down_backward,
        dw=split_leg(c.down_backward.dw,m,64,partition='balanced')))
    assert c.rejection(m,k,h,o) is None
    inputs=tuple((torch.randn(*shape,device='cuda',dtype=torch.bfloat16)*scale).requires_grad_()
        for shape,scale in (((m,k),.2),((2*h,k),.02),((o,h),.02)))
    dy=torch.randn(m,o,device='cuda',dtype=torch.bfloat16)*.1
    fn=torch.compile(lambda *a:nvfp4_mlp(*a,config=c),fullgraph=True,dynamic=False)
    y=fn(*inputs);grads=torch.autograd.grad(y,inputs,dy)
    with torch.no_grad():want=reference(*inputs,dy,c)
    for a,b in zip((y,*grads),want):
        assert bool(a.isfinite().all())
        error=float((a.detach().float()-b.float()).norm()/b.float().norm().clamp_min(1e-12))
        assert error<.03,error
