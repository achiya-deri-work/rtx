"""MoE-local shared MLP first-hit binding, without changing the global MLP API."""
import pytest
import torch


cuda=pytest.mark.skipif(not torch.cuda.is_available(),reason='SM120 CUDA required')


def _spec(cache_dir=None):
    from rtx.mlp import DEFAULT_MLP_CONFIG
    from rtx._mlp_autotune import deferred_spec
    return deferred_spec(DEFAULT_MLP_CONFIG,'mxfp8','cache',None,cache_dir)


def _fake(mode,shape,requires_grad=True):
    from torch._subclasses.fake_tensor import FakeTensor
    return FakeTensor(mode,torch.empty(shape,device='meta',dtype=torch.bfloat16,
                                       requires_grad=requires_grad),torch.device('cuda:0'))


def _values(mode,n=32):
    return _fake(mode,(n,64)),_fake(mode,(128,64)),_fake(mode,(64,64))


def test_shared_cold_token_is_private_and_fixed_hot_op_has_no_cpu_output():
    import rtx._moe_shared  # noqa: F401
    import rtx.mlp  # noqa: F401
    cold=torch.ops.rtx.moe_shared_select.default
    backward=torch.ops.rtx.moe_shared_selected_backward.default
    hot=torch.ops.rtx.mxfp8_mlp.default
    assert len(cold._schema.arguments)==6 and len(cold._schema.returns)==2
    assert len(backward._schema.arguments)==9 and len(backward._schema.returns)==3
    assert torch.Tag.cudagraph_unsafe in cold.tags and torch.Tag.cudagraph_unsafe in backward.tags
    assert len(hot._schema.returns)==1 and torch.Tag.cudagraph_unsafe not in hot.tags


def test_shared_fake_cuda_transitions_once_per_shape_and_hot_has_static_autograd():
    from torch._subclasses.fake_tensor import FakeTensorMode
    from rtx import _moe_selection as cells
    from rtx._moe_shared import selected_shared,bind_shared_config
    from rtx.mlp import DEFAULT_MLP_CONFIG,_config_key
    spec=_spec();key=_config_key(DEFAULT_MLP_CONFIG)
    cells.clear_selection_cells(namespace='moe_shared:');torch.compiler.reset()
    captured=[]
    def backend(graph,_inputs):captured.append(graph);return graph.forward
    compiled=torch.compile(lambda x,gu,down:selected_shared(x,gu,down,'silu',spec,True),
                           backend=backend,fullgraph=True,dynamic=False)
    with FakeTensorMode() as mode:
        first=_values(mode);out=compiled(*first)
        assert out.shape==(32,64) and out.requires_grad
        assert len(captured)==1
        bind_shared_config(*first,'silu',spec,True,key)
        compiled(*first);compiled(*first)
        assert len(captured)==2
        code='\n'.join(m.code for m in captured[-1].modules() if isinstance(m,torch.fx.GraphModule))
        assert 'moe_shared_select' not in code and 'moe_shared_selected_backward' not in code
        assert 'mxfp8_mlp' in code and 'selected_replay' not in code
        second=_values(mode,48);compiled(*second)
        bind_shared_config(*second,'silu',spec,True,key);compiled(*second)
        assert len(captured)==4
        compiled(*first)
        assert len(captured)==4
    cells.clear_selection_cells(namespace='moe_shared:');torch.compiler.reset()


def test_shared_selection_namespace_does_not_collide_with_expert_or_router():
    from rtx import _moe_selection as cells
    from rtx._moe_shared import binding_identity,bind_shared_config
    x=torch.empty(8,4);gu=torch.empty(8,4);down=torch.empty(4,4)
    bind_shared_config(x,gu,down,'silu','policy',True,'shared')
    identity=binding_identity(x,gu,down,'silu','policy',True)
    assert cells.bound_config(*identity)=='shared'
    assert cells.clear_selection_cells(namespace='routing:')>=0
    assert cells.bound_config(*identity)=='shared'
    cells.clear_selection_cells(namespace='moe_shared:')


@cuda
def test_shared_cache_cold_and_bound_backwards_match_static_and_never_private_replay(monkeypatch,tmp_path):
    from rtx._moe_shared import selected_shared
    from rtx._moe_selection import clear_selection_cells
    from rtx.mlp import DEFAULT_MLP_CONFIG,mxfp8_mlp
    import rtx._mlp_autotune as startup
    clear_selection_cells(namespace='moe_shared:')
    torch.manual_seed(173)
    values=tuple((torch.randn(*shape,device='cuda',dtype=torch.bfloat16)*scale).requires_grad_()
                 for shape,scale in (((32,128),.2),((128,128),.02),((128,64),.02)))
    def forbidden(*args,**kwargs):raise AssertionError('MoE shared path requested private replay')
    monkeypatch.setattr(startup,'_replay_forward',forbidden)
    calls=[];original=startup._select
    def count(*args,**kwargs):calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(startup,'_select',count)
    spec=_spec(tmp_path);dy=torch.randn_like(values[0])
    cold=selected_shared(*values,'silu',spec,True)
    hot=selected_shared(*values,'silu',spec,True)
    expected=mxfp8_mlp(*values,config=DEFAULT_MLP_CONFIG)
    clear_selection_cells(namespace='moe_shared:')
    gc=torch.autograd.grad(cold,values,dy)
    gh=torch.autograd.grad(hot,values,dy)
    ge=torch.autograd.grad(expected,values,dy)
    assert len(calls)==1
    torch.testing.assert_close(cold,expected,rtol=.003,atol=2e-7)
    torch.testing.assert_close(hot,expected,rtol=.003,atol=2e-7)
    for left,right,reference in zip(gc,gh,ge):
        assert bool(left.isfinite().all()) and bool(right.isfinite().all())
        torch.testing.assert_close(left,reference,rtol=.01,atol=2e-6)
        torch.testing.assert_close(right,reference,rtol=.01,atol=2e-6)


@cuda
def test_whole_cache_mode_cold_to_bound_default_compile_auto_graph_all_parameters(tmp_path,monkeypatch):
    from rtx.moe import BalancedMoE,DEFAULT_MOE_CONFIG
    from rtx.routing import RoutingKernelConfig
    from rtx._moe_selection import clear_selection_cells
    import rtx._mlp_autotune as startup
    clear_selection_cells()
    torch.manual_seed(174)
    model=BalancedMoE(128,4,64,top_k=2,shared_hidden_features=64,device='cuda',
        config=DEFAULT_MOE_CONFIG,autotune='cache',autotune_cache_dir=tmp_path,
        router_options=dict(config=RoutingKernelConfig(),autotune='off'))
    with torch.no_grad():model.router.projection_weight.normal_(std=.02)
    def forbidden(*args,**kwargs):raise AssertionError('Whole MoE used legacy shared selection/private replay')
    monkeypatch.setattr(startup,'selected_mlp',forbidden)
    monkeypatch.setattr(startup,'_replay_forward',forbidden)
    compiled=torch.compile(model,fullgraph=True,dynamic=False,options={'triton.cudagraphs':True})
    x=(torch.randn(32,128,device='cuda',dtype=torch.bfloat16)*.2).requires_grad_()
    dy=torch.randn_like(x)
    for _ in range(4):
        result=compiled(x)
        grads=torch.autograd.grad(result,(x,*model.parameters()),dy)
        assert bool(result.isfinite().all())
        assert all(bool(grad.isfinite().all()) for grad in grads)
    torch.cuda.synchronize()
