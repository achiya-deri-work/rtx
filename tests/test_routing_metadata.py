"""Stable expert metadata: independent ordering, gradient and compiler checks."""
import pytest
import torch


cuda=pytest.mark.skipif(not torch.cuda.is_available(),reason='SM120 CUDA required')


def assignments(n,e,k,device='cpu'):
    # Equal loads and distinct row choices, with an arbitrary expert relabeling.
    generator=torch.Generator().manual_seed(1287+n+e+k)
    permutation=torch.randperm(e,generator=generator)
    return permutation[(torch.arange(n*k)%e).reshape(n,k)].to(device)


def test_layout_legality_and_schema_have_explicit_mutation():
    from rtx._routing_metadata import layout,rejection
    assert rejection(68,16,4,65,128) is None
    assert layout(68,16,4,65,128)[1]==(48,272,48)
    for shape in ((68,16,4,69,128),(68,16,4,65,64),(1024,256,8,1000,128),(64,64,64,64,128)):
        assert rejection(*shape)
    op=torch.ops.rtx_routing.metadata_execute.default
    assert not op._schema.returns
    assert next(arg for arg in op._schema.arguments if arg.name=='buffers').alias_info.is_write
    assert torch.Tag.cudagraph_unsafe not in op.tags
    op=torch.ops.rtx_routing.metadata_backward.default
    assert next(arg for arg in op._schema.arguments if arg.name=='out').alias_info.is_write


@pytest.mark.parametrize('n,e,k,real,threads',[(68,16,4,65,128),(14,7,2,13,256),(16,4,4,16,512)])
def test_cpu_chunk_rank_contract_matches_independent_stable_sort(n,e,k,real,threads):
    chosen=assignments(n,e,k).flatten();m=chosen.numel();chunks=(m+threads-1)//threads
    ranks=torch.empty(m,dtype=torch.int64);counts=torch.zeros(e,chunks,dtype=torch.int64)
    for expert in range(e):
        for chunk in range(chunks):
            begin=chunk*threads;end=min(m,begin+threads)
            match=chosen[begin:end]==expert
            local=match.long().cumsum(0)-1
            ranks[begin:end][match]=local[match]
            counts[expert,chunk]=match.sum()
    offsets=counts.cumsum(1)-counts
    index=torch.arange(m)
    inverse=chosen*(m//e)+offsets[chosen,index//threads]+ranks
    order=chosen.argsort(stable=True)
    expected=torch.empty_like(order);expected[order]=torch.arange(m)
    torch.testing.assert_close(inverse,expected,rtol=0,atol=0)
    assert inverse.unique().numel()==m


def _fake(mode,shape,dtype,grad=False):
    from torch._subclasses.fake_tensor import FakeTensor
    return FakeTensor(mode,torch.empty(shape,device='meta',dtype=dtype,requires_grad=grad),torch.device('cuda:0'))


def test_fake_fullgraph_metadata_and_visible_allocations():
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.proxy_tensor import make_fx
    from rtx._routing_metadata import routing_metadata
    call=lambda p,a:routing_metadata(p,a,num_real_tokens=65,threads=128)
    with FakeTensorMode() as mode:
        logp=_fake(mode,(68,16),torch.float32,True)
        chosen=_fake(mode,(68,4),torch.int64)
        output=torch.compile(call,backend='eager',fullgraph=True,dynamic=False)(logp,chosen)
        assert [value.shape for value in output]==[(16,17),(16,17),(65,4)]
        assert output[1].requires_grad and not output[0].requires_grad and not output[2].requires_grad
        graph=make_fx(call,tracing_mode='fake')(logp,chosen)
    targets=[str(node.target) for node in graph.graph.nodes if node.op=='call_function']
    assert targets.count('rtx_routing.metadata_execute.default')==1
    assert sum(name.startswith('aten.empty.') for name in targets)>=6
    assert not any(name.startswith(('aten.argsort.','aten.sort.','aten._to_copy.','aten.softmax.')) for name in targets)


@cuda
@pytest.mark.parametrize('n,e,k,real,threads,compiled',[
    (68,16,4,65,128,False),(68,16,4,65,256,True),
    (14,7,2,13,512,True),(16,4,4,16,128,False),(128,24,6,125,256,True),
    (8192,48,6,8189,128,False)])
def test_cuda_metadata_and_all_log_transport_gradients(n,e,k,real,threads,compiled):
    from rtx._routing_metadata import routing_metadata
    from rtx._routing_reference import routing_metadata as reference
    torch.manual_seed(1488+n+e)
    logp=(torch.randn(n,e,device='cuda')*.8).requires_grad_()
    chosen=assignments(n,e,k,'cuda')
    call=lambda p,a:routing_metadata(p,a,num_real_tokens=real,threads=threads)
    if compiled:call=torch.compile(call,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    result=call(logp,chosen)
    oracle=logp.detach().double().requires_grad_();expected=reference(oracle,chosen,num_real_tokens=real)
    torch.testing.assert_close(result[0],expected[0],rtol=0,atol=0)
    torch.testing.assert_close(result[2],expected[2],rtol=0,atol=0)
    torch.testing.assert_close(result[1].double(),expected[1],rtol=3e-6,atol=3e-7)
    # Deliberately noncontiguous [E,C] upstream.
    dy=torch.randn(result[1].shape[1],e,device='cuda').T
    actual_gradient=torch.autograd.grad(result[1],logp,dy)[0]
    expected_gradient=torch.autograd.grad(expected[1],oracle,dy.double())[0]
    torch.testing.assert_close(actual_gradient.double(),expected_gradient,rtol=3e-5,atol=5e-7)
    assert bool(actual_gradient.isfinite().all())
    assert not bool(actual_gradient[real:].count_nonzero())
    # Outputs are dedicated owned buffers, never aliases into scratch arenas.
    assert all(value._base is None for value in result)
    assert len({value.untyped_storage().data_ptr() for value in result})==3


@cuda
def test_cuda_private_opcheck_no_hot_allocations_and_graph_replay(monkeypatch):
    import rtx._routing_metadata as metadata
    from rtx._routing_reference import routing_metadata as reference
    logp=torch.randn(68,16,device='cuda',requires_grad=True);chosen=assignments(68,16,4,'cuda')
    config,sizes=metadata.layout(68,16,4,65,128)
    buffers=[torch.empty(size,device='cuda',dtype=torch.int32) for size in sizes]
    buffers += [torch.empty(272,device='cuda',dtype=torch.int64),torch.empty(272,device='cuda'),torch.empty(260,device='cuda',dtype=torch.int64)]
    args=(logp.detach().flatten(),chosen.flatten(),buffers,list(config))
    metadata.metadata_execute(*args)
    result=torch.library.opcheck(metadata.metadata_execute,args,test_utils=('test_schema','test_faketensor'))
    assert set(result.values())=={'SUCCESS'}
    dense=torch.empty(68*16,device='cuda');upstream=torch.randn(272,device='cuda')
    backward_args=(upstream,buffers[4],buffers[5],dense,list(config))
    metadata.metadata_backward(*backward_args)
    checks=torch.library.opcheck(metadata.metadata_backward,backward_args,
                                test_utils=('test_schema','test_faketensor'))
    assert set(checks.values())=={'SUCCESS'}
    def forbidden(*args,**kwargs):raise AssertionError('Hot metadata operation allocated with Torch')
    with monkeypatch.context() as patch:
        patch.setattr(torch,'empty',forbidden);patch.setattr(torch,'empty_like',forbidden)
        metadata.metadata_execute(*args)
        metadata.metadata_backward(*backward_args)
    call=torch.compile(lambda p:metadata.routing_metadata(p,chosen,num_real_tokens=65),
                       fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    coefficient=torch.randn(16,17,device='cuda')
    def step():
        values=call(logp)
        return values,torch.autograd.grad(values[1],logp,coefficient)[0]
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):step()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):actual,gradient=step()
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(2):
        with torch.no_grad():
            logp.add_(.01*torch.randn_like(logp));coefficient.add_(.1*torch.randn_like(coefficient))
        graph.replay();torch.cuda.synchronize()
        expected=reference(logp,chosen,num_real_tokens=65)
        torch.testing.assert_close(actual[0],expected[0],rtol=0,atol=0)
        torch.testing.assert_close(actual[2],expected[2],rtol=0,atol=0)
        torch.testing.assert_close(actual[1],expected[1],rtol=3e-6,atol=3e-7)
        expected_gradient=torch.autograd.grad(expected[1],logp,coefficient)[0]
        torch.testing.assert_close(gradient,expected_gradient,rtol=3e-5,atol=5e-7)
    graph.reset()
