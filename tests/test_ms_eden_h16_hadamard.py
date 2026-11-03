"""Pure H16 transform qualification, explicitly not an MS-EDEN speed claim."""
import os
import pytest
import torch
from rtx.kernels.ms_eden_h16_hadamard import H16Config,H16HadamardKernel,compile_h16_hadamard
from rtx.kernels.ms_eden_tf32_hadamard import TF32H8Config,compile_tf32_h8_hadamard
from test_ms_eden_tf32_hadamard import oracle,compare_metrics


def test_h16_pair_tail_and_configuration():
    k=H16HadamardKernel(3,128)
    assert k.tiles==3 and k.pairs==2 and k.blocks==1
    for kwargs in (dict(threads=32),dict(blocks=0),dict(force_scalar=1),dict(output_store_bits=True),dict(output_store_bits=128)):
        with pytest.raises(ValueError):H16Config(**kwargs)


run_cuda=pytest.mark.skipif(os.getenv('RTX_RUN_H16_CUDA')!='1',reason='assigned GPU only')


@run_cuda
@pytest.mark.parametrize('shape',[(3,128),(64,256),(5,384)])
@pytest.mark.parametrize('distribution',['gaussian','outliers','cancellation','zero'])
def test_h16_fp64_quality(shape,distribution):
    torch.manual_seed(959)
    x=torch.randn(shape,device='cuda',dtype=torch.bfloat16)*.25
    if distribution=='outliers':x.flatten()[::97].mul_(32)
    elif distribution=='cancellation':
        x.fill_(1.);x[:,1::2].fill_(-1.);x[:,::17].add_(.0078125)
    elif distribution=='zero':x.zero_()
    signs=torch.where(torch.rand(128,device='cuda')<.5,-1.,1.)
    out=torch.empty(x.numel(),device='cuda')
    route=torch.empty(x.numel()//128,device='cuda',dtype=torch.int32)
    compile_h16_hadamard(*shape)(x.flatten(),signs,out,route)
    measured=compare_metrics(out,oracle(x,signs).flatten())
    assert measured['nrms']<=1e-5 and measured['normalized_max']<=1e-5
    assert torch.all(route==1)


@run_cuda
@pytest.mark.parametrize('unsafe',['subnormal','tiny','huge','arbitrary_sign','zero_sign','nan_sign'])
def test_h16_guarded_fallback_matches_original(unsafe):
    x=torch.randn(3,128,device='cuda',dtype=torch.bfloat16)*.25
    signs=torch.ones(128,device='cuda')
    if unsafe=='subnormal':x[:,0]=2.**-133
    elif unsafe=='tiny':x[:,0]=2.**-110
    elif unsafe=='huge':x[:,0]=2.**120
    elif unsafe=='arbitrary_sign':signs[7]=.3
    elif unsafe=='zero_sign':signs[7]=-0.
    else:signs[7]=float('nan')
    out=torch.empty(x.numel(),device='cuda');expected=torch.empty_like(out)
    route=torch.empty(3,device='cuda',dtype=torch.int32);other=torch.empty_like(route)
    compile_h16_hadamard(3,128)(x.flatten(),signs,out,route)
    compile_tf32_h8_hadamard(3,128,TF32H8Config(force_scalar=True))(x.flatten(),signs,expected,other)
    assert torch.all(route==0)
    assert torch.equal(torch.isnan(out),torch.isnan(expected))
    finite=~torch.isnan(expected)
    assert torch.equal(out[finite].view(torch.int32),expected[finite].view(torch.int32))


@run_cuda
def test_h16_graph_current_inputs_and_no_live_growth():
    x=torch.randn(5,384,device='cuda',dtype=torch.bfloat16)*.25
    signs=torch.ones(128,device='cuda')
    out=torch.empty(x.numel(),device='cuda')
    route=torch.empty(x.numel()//128,device='cuda',dtype=torch.int32)
    run=compile_h16_hadamard(5,384)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):run(x.flatten(),signs,out,route)
    stream.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g,stream=stream):run(x.flatten(),signs,out,route)
    for sign in (1.,.75,-1.):
        x.mul_(.5);signs.fill_(sign);g.replay();torch.cuda.synchronize()
        assert torch.all(route==(0 if sign==.75 else 1))
        assert compare_metrics(out,oracle(x,signs).flatten())['nrms']<=1e-5
    before=torch.cuda.memory_allocated()
    for _ in range(20):g.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated()==before


@run_cuda
@pytest.mark.parametrize('shape',[(3,128),(5,384),(4096,1536)])
def test_h16_paired_stores_are_bitwise_and_replay_safe(shape):
    torch.manual_seed(960)
    x=torch.randn(shape,device='cuda',dtype=torch.bfloat16)*.25
    signs=torch.where(torch.rand(128,device='cuda')<.5,-1.,1.)
    outputs=[torch.empty(x.numel(),device='cuda') for _ in range(2)]
    routes=[torch.empty(x.numel()//128,device='cuda',dtype=torch.int32) for _ in range(2)]
    runners=[compile_h16_hadamard(*shape,H16Config(output_store_bits=bits)) for bits in (32,64)]
    graphs=[];stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    for run,out,route in zip(runners,outputs,routes):
        with torch.cuda.stream(stream):
            for _ in range(3):run(x.flatten(),signs,out,route)
        stream.synchronize();g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g,stream=stream):run(x.flatten(),signs,out,route)
        graphs.append(g)
    for iteration in range(4):
        x.mul_(.5);signs.neg_()
        if iteration==1:x.flatten()[0]=2.**-110
        elif iteration==2:x.zero_()
        elif iteration==3:signs[5]=.75
        for g in graphs:g.replay()
        torch.cuda.synchronize()
        assert torch.equal(outputs[0].view(torch.int32),outputs[1].view(torch.int32))
        assert torch.equal(routes[0],routes[1])
    before=torch.cuda.memory_allocated()
    for _ in range(10):
        for g in graphs:g.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated()==before
