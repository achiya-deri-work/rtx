"""Independent metadata oracle, including conservative rows and empty pools."""
import math

import pytest
import torch

from rtx._routing_varlen import _torch_metadata, workspace_bounds


def fixture(lengths,e,k,device):
    t=sum(lengths);b=len(lengths);n,c=workspace_bounds(t,b,e,k)
    multiple=e//math.gcd(e,k);scheduled=[0];physical=[0];cu=[0]
    sequence=[];rowmap=[];assignments=[];expected_ids=torch.zeros(e,c,dtype=torch.int64)
    inverse=[];selected=[]
    for s,length in enumerate(lengths):
        ns=math.ceil(length/multiple)*multiple;capacity=ns*k//e
        scheduled.append(scheduled[-1]+ns)
        counts=[0]*e
        for row in range(ns):
            choices=[(row*k+j)%e for j in range(k)]
            slots=[a*c+physical[-1]+counts[a] for a in choices]
            token=cu[-1]+min(row,length-1)
            for a in choices:
                expected_ids[a,physical[-1]+counts[a]]=token;counts[a]+=1
            sequence.append(s);rowmap.append(cu[-1]+row if row<length else -cu[-1]-length)
            assignments.append(choices)
            if row<length:inverse.append(slots);selected.append(choices)
        physical.append(physical[-1]+math.ceil(capacity/32)*32);cu.append(cu[-1]+length)
    extra=n-len(sequence)
    sequence.extend([0]*extra);rowmap.extend([-1]*extra)
    # ExtractAssignments fills inactive slots with flattened-index % E.
    assignments.extend([[(index*k+j)%e for j in range(k)] for index in range(n-extra,n)])
    generator=torch.Generator().manual_seed(441)
    scores=torch.randn(n,e,generator=generator).to(device).requires_grad_()
    tensors=[torch.tensor(x,device=device,dtype=torch.int64) for x in
             (assignments,scheduled,physical,sequence,rowmap,cu)]
    key=(t,b,e,k,n,c)
    return scores,tensors,key,expected_ids,torch.tensor(inverse),torch.tensor(selected)


@pytest.mark.parametrize('lengths,e,k',[
    ([128,128],32,4),([5,0,11,3],8,2),([0,7,0,23],8,2),
    ([1024,1536,2048,3584],32,6),([1,2],8,2),
])
@pytest.mark.parametrize('device',['cpu','cuda'])
def test_segmented_torch_metadata_matches_independent_rank_and_adjoint(lengths,e,k,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    scores,args,key,ids,inverse,chosen=fixture(lengths,e,k,device)
    call=_torch_metadata if device=='cpu' else torch.compile(_torch_metadata,fullgraph=True,dynamic=False)
    actual=call(scores,*args,key)
    torch.testing.assert_close(actual[0].cpu(),ids,rtol=0,atol=0)
    torch.testing.assert_close(actual[2].cpu(),inverse,rtol=0,atol=0)
    # Construct real score rows independently from the production slot algebra.
    rows=[];offset=0;multiple=e//math.gcd(e,k)
    for length in lengths:
        rows.extend(range(offset,offset+length));offset+=math.ceil(length/multiple)*multiple
    oracle=scores.detach().double().requires_grad_()
    expected=oracle[rows].gather(1,chosen.to(device)).softmax(1)
    values=actual[1].flatten()[actual[2]]
    torch.testing.assert_close(values.double(),expected,rtol=2e-6,atol=2e-7)
    upstream=torch.linspace(-1,1,values.numel(),device=device).reshape_as(values)
    a=torch.autograd.grad(values,scores,upstream)[0]
    b=torch.autograd.grad(expected,oracle,upstream.double())[0]
    torch.testing.assert_close(a.double(),b,rtol=2e-5,atol=2e-7)
    assert torch.isfinite(actual[1]).all()


def test_workspace_tail_really_exists_in_tiny_regression():
    scores,args,key,*_=fixture([128,128],32,4,'cpu')
    assert key[4]==264 and int(args[1][-1])==256


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_previously_faulting_tiny_tuning_candidate_replay_and_adjoint():
    from dataclasses import replace
    from rtx._routing_varlen import routing
    from rtx.routing import RoutingKernelConfig,SinkhornKernelConfig
    config=RoutingKernelConfig(forward=SinkhornKernelConfig(64,1,2),
        backward=SinkhornKernelConfig(64,4,2),metadata_backend='torch',metadata_threads=512,
        packed_column_cache_items=64,packed_column_shards_forward=16,
        packed_column_shards_backward=32,packed_row_backend='cta',
        packed_qb_column_backend='row_major',packed_sinkhorn_backend='staged',
        qb_backend='native',qb_threads=64,quota_threads=64,segmented_quotas=False,
        quota_cache_items=0,quota_early_exit=False,quota_threshold=0)
    scores=torch.nn.functional.logsigmoid(torch.randn(256,32,device='cuda'))
    scores.requires_grad_();cu=torch.tensor([0,128,256],device='cuda',dtype=torch.int32)
    def function(value):return routing(value,cu,top_k=4,steps=5,qb=5,rounds=64,config=config,owner=91441)
    call=torch.compile(function,fullgraph=True,dynamic=False)
    # Same hard plan, independent native metadata implementation.
    reference=routing(scores,cu,top_k=4,steps=5,qb=5,rounds=64,
                      config=replace(config,metadata_backend='cute'),owner=91442)
    actual=call(scores)
    for a,b in zip(actual,reference):torch.testing.assert_close(a,b,rtol=3e-5,atol=5e-7)
    upstream=torch.randn_like(actual.route_weights)
    a=torch.autograd.grad(actual.route_weights,scores,upstream)[0]
    b=torch.autograd.grad(reference.route_weights,scores,upstream)[0]
    torch.testing.assert_close(a,b,rtol=5e-4,atol=8e-6)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):call(scores)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):replayed=call(scores)
        graph.replay()
    stream.synchronize()
    for a,b in zip(replayed,actual):torch.testing.assert_close(a,b,rtol=0,atol=0)
    graph.reset()
