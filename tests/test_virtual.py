"""Independent optimizer matrices, one allocation, and tensor-only consumers."""
from collections import OrderedDict
import copy
import io

import pytest
import torch
from torch import nn

from rtx import VirtualLayer


def test_public_export_is_standalone_and_stable():
    import rtx
    assert 'VirtualLayer' in rtx.__all__ and 'VirtualLayer' in dir(rtx)
    assert rtx.VirtualLayer is VirtualLayer


def make_layer(*, dtype=torch.float64, device='cpu', bank=False):
    widths = (4, 4, 4) if bank else (6, 2, 2)
    modules = OrderedDict((name, nn.Linear(4, rows, bias=False, dtype=dtype, device=device))
                          for name, rows in zip(('q','k','v'), widths))
    return VirtualLayer(modules, packed_shape=(3,4,4) if bank else None)


def test_storage_parameter_identity_and_existing_initialization():
    modules = OrderedDict((name,nn.Linear(4,rows,bias=False)) for name,rows in [('q',6),('k',2),('v',2)])
    parameters = [m.weight for m in modules.values()]
    before = [p.detach().clone() for p in parameters]
    layer = VirtualLayer(modules)
    assert [id(p) for p in layer.parameters()] == [id(p) for p in parameters]
    assert [n for n,_ in layer.named_parameters()] == ['projections.q.weight','projections.k.weight','projections.v.weight']
    weight = layer.weight
    assert not isinstance(weight,nn.Parameter) and weight.is_contiguous() and weight.shape==(10,4)
    assert all(p.is_leaf and p.ndim==2 for p in layer.parameters())
    assert all(p.untyped_storage().data_ptr()==weight.untyped_storage().data_ptr() for p in parameters)
    assert [p.storage_offset() for p in parameters] == [0,24,32]
    torch.testing.assert_close(weight,torch.cat(before),rtol=0,atol=0)
    with torch.no_grad():layer['k'].weight.add_(7)
    torch.testing.assert_close(layer.weight[6:8],before[1]+7)


def test_full_span_alias_is_nonpersistent_and_never_an_extra_parameter():
    layer=make_layer()
    storage=layer._packed_storage
    assert tuple(storage.shape)==layer.packed_shape
    assert storage.numel()==sum(p.numel() for p in layer.parameters())
    assert storage.data_ptr()==layer.weight.data_ptr()
    assert not storage.requires_grad
    assert len(tuple(layer.parameters()))==3
    assert '_packed_storage' not in layer.state_dict()
    layer.float();layer.validate()
    assert layer._packed_storage.data_ptr()==layer.weight.data_ptr()


def test_compile_exposes_full_span_storage_without_weight_copy():
    layer=make_layer(dtype=torch.float32,bank=True)
    x=torch.randn(3,5,4)
    graphs=[]
    def backend(graph,inputs):
        graphs.append(graph)
        return graph.forward
    actual=torch.compile(layer,backend=backend,fullgraph=True,dynamic=False)(x)
    assert len(graphs)==1
    storage=[n.meta['example_value'] for n in graphs[0].graph.nodes
             if n.op=='placeholder' and '_packed_storage' in n.name]
    assert len(storage)==1 and tuple(storage[0].shape)==layer.packed_shape
    assert not any('cat' in str(n.target) or 'clone' in str(n.target)
                   for n in graphs[0].graph.nodes if n.op=='call_function')
    torch.testing.assert_close(actual,layer(x))


@pytest.fixture
def isolated_compile_variants():
    # These independent static-mode cases must not exhaust the default
    # per-code-object recompile cap for subsequent, unrelated tests.
    torch.compiler.reset()
    yield
    torch.compiler.reset()


@pytest.mark.parametrize('mode',['no_grad','inference','frozen'])
@pytest.mark.parametrize('bank',[False,True])
def test_compile_without_bank_weight_adjoint_keeps_zero_copy(mode,bank,isolated_compile_variants):
    layer=make_layer(dtype=torch.float32,bank=bank)
    if mode=='frozen':layer.requires_grad_(False)
    x=torch.randn((3,5,4) if bank else (5,4),requires_grad=mode=='frozen')
    fn=torch.compile(layer,backend='aot_eager',fullgraph=True,dynamic=False)
    context=torch.no_grad() if mode=='no_grad' else torch.inference_mode() if mode=='inference' else torch.enable_grad()
    with context:
        actual=fn(x);expected=layer(x)
        torch.testing.assert_close(actual,expected)
        assert layer.weight.data_ptr()==layer._packed_storage.data_ptr()
        if mode=='frozen':
            torch.testing.assert_close(torch.autograd.grad(actual.sum(),x)[0],
                                       torch.autograd.grad(expected.sum(),x)[0])
    with torch.no_grad():layer['v'].weight.add_(.1)
    with context:torch.testing.assert_close(fn(x),layer(x))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required for Inductor subprocess autotuning')
@pytest.mark.parametrize('bank',[False,True])
def test_cuda_virtual_subprocess_autotune_uses_full_storage(bank):
    from torch.utils.checkpoint import checkpoint
    torch.manual_seed(827)
    layer=VirtualLayer({str(i):nn.Linear(128,128,bias=False,device='cuda',dtype=torch.bfloat16)
                        for i in range(4)},packed_shape=(4,128,128) if bank else None)
    reference=copy.deepcopy(layer.projections)
    x=torch.randn((4,64,128) if bank else (64,128),device='cuda',dtype=torch.bfloat16,requires_grad=True)
    xr=x.detach().clone().requires_grad_()
    def call(value):return checkpoint(layer,value,use_reentrant=True)
    fn=torch.compile(call,fullgraph=True,dynamic=False,options={'max_autotune':True,
        'autotune_in_subproc':True,'triton.cudagraphs':False})
    actual=fn(x)
    expected=(torch.stack([reference[str(i)](xr[i]) for i in range(4)]) if bank
              else torch.cat([m(xr) for m in reference.values()],dim=-1))
    dy=torch.randn_like(actual)
    grads=torch.autograd.grad(actual,(x,*layer.parameters()),dy)
    refs=torch.autograd.grad(expected,(xr,*reference.parameters()),dy)
    # BF16 accumulation order may differ between selected GEMM backends.
    for a,b in zip((actual,*grads),(expected,*refs)):
        assert bool(a.isfinite().all())
        torch.testing.assert_close(a,b,rtol=.03,atol=.08)
    with torch.no_grad():layer['3'].weight.add_(.01)
    assert bool(fn(x).isfinite().all())
    torch.cuda.synchronize()


@pytest.mark.parametrize('bank',[False,True])
@pytest.mark.parametrize('compiled',[False,True])
def test_output_all_gradients_and_changed_parameters(bank,compiled):
    torch.manual_seed(721)
    layer = make_layer(bank=bank)
    reference = copy.deepcopy(layer.projections)
    shape = (3,5,4) if bank else (2,5,4)
    x = torch.randn(shape,dtype=torch.float64,requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    fn = torch.compile(layer,backend='aot_eager',fullgraph=True,dynamic=False) if compiled else layer
    for step in range(2):
        y = fn(x)
        yr = (torch.stack([reference[n](xr[i]) for i,n in enumerate(reference)]) if bank
              else torch.cat([m(xr) for m in reference.values()],-1))
        grad = torch.randn(tuple(reversed(y.shape)),dtype=y.dtype).permute(tuple(reversed(range(y.ndim))))
        actual = torch.autograd.grad(y,(x,*layer.parameters()),grad)
        expected = torch.autograd.grad(yr,(xr,*reference.parameters()),grad)
        torch.testing.assert_close(y,yr)
        for a,b in zip(actual,expected):torch.testing.assert_close(a,b)
        with torch.no_grad():
            for p,r in zip(layer.parameters(),reference.parameters()):p.add_(.01);r.add_(.01)
    layer.validate()


@pytest.mark.parametrize('optimizer',['sgd','adamw','muon'])
def test_optimizer_updates_are_independent_2d_matrices(optimizer):
    torch.manual_seed(722)
    layer = make_layer(dtype=torch.float32)
    reference = copy.deepcopy(layer.projections)
    if optimizer=='muon' and not hasattr(torch.optim,'Muon'):pytest.skip('Muon unavailable in this Torch build')
    cls = {'sgd':torch.optim.SGD,'adamw':torch.optim.AdamW,'muon':getattr(torch.optim,'Muon',None)}[optimizer]
    kwargs = dict(lr=.003)
    if optimizer=='sgd':kwargs['momentum']=.9
    a,b = cls(layer.parameters(),**kwargs),cls(reference.parameters(),**kwargs)
    for step in range(3):
        x = torch.randn(5,4)
        a.zero_grad(set_to_none=step%2==0);b.zero_grad(set_to_none=step%2==0)
        layer(x).square().sum().backward()
        torch.cat([m(x) for m in reference.values()],-1).square().sum().backward()
        a.step();b.step();layer.validate()
        for p,r in zip(layer.parameters(),reference.parameters()):torch.testing.assert_close(p,r)
    assert len(a.param_groups[0]['params'])==3


def test_frozen_projection_individual_consumer_and_gradient_accumulation():
    layer = make_layer()
    layer['q'].requires_grad_(False)
    x = torch.randn(3,4,dtype=torch.float64,requires_grad=True)
    layer(x).sum().backward()
    first = layer['k'].weight.grad.clone()
    layer['k'](x).sum().backward()
    torch.testing.assert_close(layer['k'].weight.grad,2*first)
    assert layer['q'].weight.grad is None and layer['v'].weight.grad is not None


def test_swiglu_gate_up_compiled_outputs_and_independent_gradients():
    torch.manual_seed(724)
    packed=VirtualLayer({name:nn.Linear(8,12,bias=False,dtype=torch.float64)
                         for name in ('gate','up')})
    separate=copy.deepcopy(packed.projections)
    down=nn.Linear(12,8,bias=False,dtype=torch.float64)
    down_reference=copy.deepcopy(down)
    assert packed.weight.shape==(24,8)
    def call(x):
        gate,up=packed(x).chunk(2,dim=-1)
        return down(torch.nn.functional.silu(gate)*up)
    x=torch.randn(3,5,8,dtype=torch.float64,requires_grad=True)
    xr=x.detach().clone().requires_grad_()
    actual=torch.compile(call,backend='aot_eager',fullgraph=True,dynamic=False)(x)
    expected=down_reference(torch.nn.functional.silu(separate['gate'](xr))*separate['up'](xr))
    gradient=torch.randn_like(actual)
    actual_grads=torch.autograd.grad(actual,(x,*packed.parameters(),*down.parameters()),gradient)
    expected_grads=torch.autograd.grad(expected,(xr,*separate.parameters(),*down_reference.parameters()),gradient)
    torch.testing.assert_close(actual,expected)
    for a,b in zip(actual_grads,expected_grads):torch.testing.assert_close(a,b)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required for native MXFP8 MLP')
def test_cuda_swiglu_virtual_gate_up_matches_packed_mxfp8_mlp():
    import rtx
    torch.manual_seed(725)
    gate_up=VirtualLayer({name:nn.Linear(128,128,bias=False,device='cuda',dtype=torch.bfloat16)
                          for name in ('gate','up')})
    down=nn.Linear(128,128,bias=False,device='cuda',dtype=torch.bfloat16)
    reference=nn.Parameter(gate_up.weight.detach().clone())
    reference_down=nn.Parameter(down.weight.detach().clone())
    x=(torch.randn(64,128,device='cuda',dtype=torch.bfloat16)*.2).requires_grad_()
    gradient=torch.randn_like(x)
    fn=torch.compile(lambda value:rtx.mxfp8_mlp(value,gate_up.weight,down.weight,activation='silu'),
                     fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    actual=fn(x)
    actual_grads=torch.autograd.grad(actual,(x,*gate_up.parameters(),down.weight),gradient)
    expected=rtx.mxfp8_mlp(x,reference,reference_down,activation='silu')
    dx,dgu,ddown=torch.autograd.grad(expected,(x,reference,reference_down),gradient)
    for a,b in zip((actual,*actual_grads),(expected,dx,*dgu.chunk(2,dim=0),ddown)):
        torch.testing.assert_close(a,b,rtol=0,atol=0)


@pytest.mark.parametrize('assign',[False,True])
def test_checkpoint_roundtrip_and_parent_loading(assign):
    layer = make_layer()
    parent = nn.Sequential(layer)
    saved = {k:v.clone() for k,v in parent.state_dict().items()}
    assert len(saved)==3 and all('weight' in k for k in saved)
    with torch.no_grad():
        for p in layer.parameters():p.zero_()
    parent.load_state_dict(saved,assign=assign)
    layer.validate()
    for k,v in parent.state_dict().items():torch.testing.assert_close(v,saved[k])
    buffer = io.BytesIO();torch.save(parent.state_dict(),buffer);buffer.seek(0)
    parent.load_state_dict(torch.load(buffer,weights_only=True),assign=assign)
    layer.validate()


def test_dtype_move_preserves_identity_values_gradients_and_optimizer_references():
    layer = make_layer(dtype=torch.float32)
    parameters = tuple(layer.parameters())
    layer(torch.randn(2,4)).sum().backward()
    saved = tuple((p.detach().clone(),p.grad.clone()) for p in parameters)
    layer.double();layer.validate()
    assert all(p is q for p,q in zip(parameters,layer.parameters()))
    for p,(v,g) in zip(layer.parameters(),saved):
        torch.testing.assert_close(p,v.double());torch.testing.assert_close(p.grad,g.double())
    assert layer.weight.dtype==torch.float64


def test_meta_materialization_and_assign_loading():
    source = make_layer(dtype=torch.float32)
    target = make_layer(dtype=torch.float32,device='meta')
    assert target.weight.device.type=='meta'
    target.load_state_dict(source.state_dict(),assign=True)
    target.validate();torch.testing.assert_close(target.weight,source.weight)
    target.to('meta');target.to_empty(device='cpu');target.validate()
    target.load_state_dict(source.state_dict());torch.testing.assert_close(target.weight,source.weight)


def test_deepcopy_has_independent_packed_storage():
    layer = make_layer()
    other = copy.deepcopy(layer)
    other.validate();torch.testing.assert_close(other.weight,layer.weight)
    assert other.weight.data_ptr()!=layer.weight.data_ptr()
    with torch.no_grad():other['q'].weight.zero_()
    assert bool(layer['q'].weight.abs().sum()>0)


def test_replacement_requires_repack_and_keeps_replacement_identity():
    layer = make_layer()
    replacement = nn.Parameter(torch.ones_like(layer['k'].weight))
    layer['k'].weight = replacement
    with pytest.raises(RuntimeError,match='packing changed'):layer.weight
    layer.repack_();layer.validate()
    assert layer['k'].weight is replacement
    torch.testing.assert_close(layer.weight[6:8],replacement)


def test_generic_flat_packing_and_overlapping_weights_rejected():
    layer = VirtualLayer({'a':nn.Linear(4,3,bias=False),'b':nn.Linear(3,4,bias=False)})
    assert layer.weight.shape==(24,)
    layer.weight.sum().backward()
    assert all(bool((p.grad==1).all()) for p in layer.parameters())
    base = torch.randn(5,4)
    a,b = nn.Linear(4,3,bias=False),nn.Linear(4,3,bias=False)
    a.weight=nn.Parameter(base[:3]);b.weight=nn.Parameter(base[2:])
    with pytest.raises(ValueError,match='Overlapping'):
        VirtualLayer({'a':a,'b':b})


def test_gradcheck_and_gradgradcheck():
    layer = make_layer()
    x = torch.randn(2,4,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda a:layer(a),(x,))
    assert torch.autograd.gradgradcheck(lambda a:layer(a),(x,))


def test_all_parameter_gradcheck_and_reentrant_checkpoint():
    from rtx.virtual import _VirtualWeight
    from torch.utils.checkpoint import checkpoint
    layer = VirtualLayer({'a':nn.Linear(2,2,bias=False,dtype=torch.float64),
                          'b':nn.Linear(2,1,bias=False,dtype=torch.float64)})
    assert torch.autograd.gradcheck(lambda *p:_VirtualWeight.apply(layer.packed_shape,layer._packed_storage,*p),
                                   tuple(layer.parameters()))
    x = torch.randn(2,2,dtype=torch.float64,requires_grad=True)
    checkpoint(layer,x,use_reentrant=True).sum().backward()
    assert x.grad is not None and all(p.grad is not None for p in layer.parameters())


def test_mutating_a_nonfirst_parameter_before_backward_is_rejected():
    layer = make_layer()
    x = torch.randn(2,4,dtype=torch.float64,requires_grad=True)
    loss = layer(x).square().sum()
    with torch.no_grad():layer['v'].weight.add_(1)
    with pytest.raises(RuntimeError,match='modified by an inplace operation'):
        loss.backward()


def test_compile_replacement_guard_and_autocast():
    layer = make_layer(dtype=torch.float32)
    x = torch.randn(3,4)
    fn = torch.compile(layer,backend='aot_eager',fullgraph=True)
    fn(x)
    layer['k'].weight = nn.Parameter(torch.ones_like(layer['k'].weight))
    with pytest.raises(Exception,match='packing changed'):
        fn(x)
    layer.repack_()
    with torch.autocast('cpu',dtype=torch.bfloat16):
        y = fn(x)
    assert y.dtype==torch.bfloat16
    y.sum().backward();assert all(p.grad.dtype==torch.float32 for p in layer.parameters())


def test_inductor_one_graph_no_weight_concatenation():
    from torch._dynamo.backends.registry import lookup_backend
    graphs = []
    compiler = lookup_backend('inductor')
    def backend(gm,example_inputs):
        graphs.append(gm)
        return compiler(gm,example_inputs)
    layer = make_layer(dtype=torch.float32)
    x = torch.randn(3,4,requires_grad=True)
    fn = torch.compile(layer,backend=backend,fullgraph=True,dynamic=False)
    for _ in range(2):
        fn(x).sum().backward()
        with torch.no_grad():
            for p in layer.parameters():p.add_(.001)
    assert len(graphs)==1
    for gm in graphs[0].modules():
        if hasattr(gm,'graph'):
            assert not any('cat'==str(n.target).split('.')[-1] or 'stack'==str(n.target).split('.')[-1]
                           for n in gm.graph.nodes)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required for RTX kernels and replay')
@pytest.mark.parametrize('precision',['mxfp8','nvfp4'])
def test_cuda_rtx_compiled_leaf_gradients_graph_and_move(precision):
    import rtx
    torch.manual_seed(723)
    modules = {name:nn.Linear(128,n,bias=False,device='cuda',dtype=torch.bfloat16)
               for name,n in [('q',128),('k',64),('v',64)]}
    layer = VirtualLayer(modules)
    with torch.no_grad():
        reference = nn.Parameter(layer.weight.detach().clone())
    x = (torch.randn(128,128,device='cuda',dtype=torch.bfloat16)*.2).requires_grad_()
    dy = torch.randn(256,128,device='cuda',dtype=torch.bfloat16).T
    op = rtx.mxfp8_linear if precision=='mxfp8' else rtx.nvfp4_linear
    def call(value):return op(value,layer.weight,autotune=False)
    fn = torch.compile(call,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    def run():
        y = fn(x)
        return y,*torch.autograd.grad(y,(x,*layer.parameters()),dy)
    stream = torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):actual=run()
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(2):
        with torch.no_grad():
            x.add_(.001)
            for p in layer.parameters():p.add_(.0001)
            reference.copy_(layer.weight)
        graph.replay();torch.cuda.synchronize()
        y = op(x,reference,autotune=False)
        dx,dw = torch.autograd.grad(y,(x,reference),dy)
        expected = (y,dx,*dw.split((128,64,64),dim=0))
        for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
    graph.reset()
    layer.cpu();layer.validate();layer.cuda();layer.validate()


@pytest.mark.parametrize('bad', ['empty','bias','tied','shape','dtype','extra'])
def test_invalid_contracts_fail_before_packing(bad):
    modules = {'q':nn.Linear(4,3,bias=False),'k':nn.Linear(4,2,bias=False)}
    kwargs = {}
    if bad=='empty':modules={}
    elif bad=='bias':modules['q']=nn.Linear(4,3)
    elif bad=='tied':modules['k']=modules['q']
    elif bad=='shape':kwargs['packed_shape']=(100,)
    elif bad=='dtype':modules['k'].double()
    elif bad=='extra':modules['q'].extra=nn.Parameter(torch.ones(1))
    with pytest.raises(ValueError):VirtualLayer(modules,**kwargs)
