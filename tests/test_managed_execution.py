"""Normal torch.compile is sufficient: no RTX scopes, streams or allocation API."""
import pytest
import torch

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('graphs',[False,True])
@pytest.mark.parametrize('family',['mxfp8','nvfp4'])
def test_compiled_working_set_is_automatic(family,graphs):
    import rtx
    from rtx.runtime import runtime_cache_stats
    from torch._inductor.compile_fx import compile_fx
    torch._dynamo.reset()
    torch.manual_seed(924)
    cls=rtx.MXFP8Linear if family=='mxfp8' else rtx.NVFP4Linear
    shapes=[(k,n) for n in (128,256) for k in (128,256,384,512,640,768)]
    class Projections(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers=torch.nn.ModuleList([cls(k,n,device='cuda',autotune=False,backend='materialized') for k,n in shapes])
        def forward(self,*xs):return tuple(layer(x) for layer,x in zip(self.layers,xs))
    model=Projections()
    xs=tuple((torch.randn(128,k,device='cuda',dtype=torch.bfloat16)*.1).requires_grad_() for k,n in shapes)
    dys=tuple(torch.randn(128,n,device='cuda',dtype=torch.bfloat16)*.1 for k,n in shapes)
    parameters=tuple(model.parameters())
    compilations=[]
    def backend(graph,inputs):
        compilations.append(graph)
        if graphs:
            return compile_fx(graph,inputs,config_patches={'triton.cudagraphs':True})
        return compile_fx(graph,inputs)
    # Default compilation mode, without forcing CUDA graphs on the primary path.
    compiled=torch.compile(model,backend=backend,fullgraph=True,dynamic=False)
    live=[]
    insertions=[]
    for iteration in range(6):
        if iteration==3:
            with torch.no_grad():
                xs[0].add_(.002);parameters[0].add_(.0001)
        ys=compiled(*xs)
        grads=torch.autograd.grad(ys,(*xs,*parameters),dys)
        with torch.no_grad():
            for index,((k,n),x,dy,layer,y) in enumerate(zip(shapes,xs,dys,model.layers,ys)):
                refs=(x.float()@layer.weight.float().T,
                      dy.float()@layer.weight.float(),dy.float().T@x.float())
                actual=(y,grads[index],grads[len(xs)+index])
                for position,(a,r) in enumerate(zip(actual,refs)):
                    assert a.isfinite().all()
                    relative=float((a.float()-r).norm()/r.norm().clamp_min(1e-12))
                    assert relative < (.25 if family=='nvfp4' and position==0 else .08),(family,position,relative)
        del ys,grads,y,a,r,actual,refs
        torch.cuda.synchronize()
        if iteration>=3:
            live.append(torch.cuda.memory_allocated())
            insertions.append({name:row['insertions'] for name,row in runtime_cache_stats().items()})
    assert len(compilations)==1,'memory management must not cause recompilation'
    assert len(set(live))==1,live
    assert all(row==insertions[0] for row in insertions),insertions
    stats=rtx.runtime_memory_report()['automatic_execution']
    assert stats['compiled_functions']>=2
    if graphs:assert stats['recordings']>=2
    torch._dynamo.reset()


def test_ordinary_backward_reuses_split_storage_without_user_arena():
    import rtx
    from rtx import fp8_bwd as b
    from rtx._bwd_search import backward_structural_seeds
    from rtx.kernels.mxfp8 import MXFP8Problem
    from rtx.kernels.mxfp8_bwd import DEFAULT_FUSED_MXFP8_BWD_CONFIG
    pointers=[]
    for m in (4096,8192):
        p=MXFP8Problem(m,256,256)
        cfg=next(c for c in backward_structural_seeds(p,DEFAULT_FUSED_MXFP8_BWD_CONFIG)
                 if c.dw.backend=='decomposed' and c.dw.split_reduction==32)
        args=(torch.randn(m,256,device='cuda',dtype=torch.bfloat16)*.1,
              torch.randn(m,256,device='cuda',dtype=torch.bfloat16)*.1,
              torch.randn(256,256,device='cuda',dtype=torch.bfloat16)*.1)
        actual=rtx.mxfp8_linear_backward(*args,config=cfg,autotune=False)
        with torch.no_grad():
            refs=(args[0].float()@args[2].float(),args[0].float().T@args[1].float())
            for a,r in zip(actual,refs):assert float((a.float()-r).norm()/r.norm())<.08
        context=b._resolve_bwd_context(*args,b._intern_bwd_config(cfg))
        runner=b._bwd_runner_from_context(context)[1]
        pointers.append(runner.dw.workspace.untyped_storage().data_ptr())
    assert pointers[0]==pointers[1]


@pytest.mark.parametrize('family',['glu','mlp'])
def test_default_compiled_glu_and_recomputed_mlp(family):
    import rtx
    torch._dynamo.reset()
    torch.manual_seed(122)
    model=(rtx.MXFP8GLU(128,256,device='cuda') if family=='glu'
           else rtx.MXFP8MLP(128,256,128,device='cuda'))
    x=(torch.randn(256,128,device='cuda',dtype=torch.bfloat16)*.1)
    x.requires_grad_(family=='mlp')
    compiled=torch.compile(model,fullgraph=True,dynamic=False)
    with torch.no_grad():expected=model(x).clone()
    if family=='mlp':
        dy=torch.randn_like(expected)*.1
        eager=model(x)
        expected_grads=tuple(t.clone() for t in torch.autograd.grad(eager,(x,*model.parameters()),dy))
    live=[]
    for iteration in range(5):
        with torch.set_grad_enabled(family=='mlp'):
            y=compiled(x)
        torch.testing.assert_close(y,expected,rtol=0,atol=0)
        if family=='mlp':
            grads=torch.autograd.grad(y,(x,*model.parameters()),dy)
            for actual,reference in zip(grads,expected_grads):
                torch.testing.assert_close(actual,reference,rtol=0,atol=0)
            del grads,actual
        del y
        torch.cuda.synchronize()
        if iteration>=2:live.append(torch.cuda.memory_allocated())
    assert len(set(live))==1,live
    torch._dynamo.reset()
