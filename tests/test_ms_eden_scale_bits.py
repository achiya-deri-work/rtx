"""Standalone proof harness; no producer kernel or adapter integration yet."""
import os

import pytest
import torch


def edge_input_bits():
    values={0,1,2,3,0x007fffff,0x00800000,0x7f7fffff,0x7f800000}
    boundaries={0x800000,0xffffff,3*(1<<22)}
    boundaries.update(3*(1<<17)*q for q in range(21,43))
    for exponent in range(255):
        for boundary in boundaries:
            for delta in (-2,-1,0,1,2):
                m=boundary+delta
                if 0x800000<=m<=0xffffff:
                    values.add((exponent<<23)|(m-0x800000))
    return torch.tensor(sorted(values),dtype=torch.int32)


def cpu_formula(bits):
    """Proposed integer formula, independently checked against CPU division."""
    exponent=(bits>>23)&255
    n=((bits>>17)&63)|64
    q=(n*171)>>9
    carry=q>>5
    output_exponent=exponent-3+carry
    normalized=q>>carry
    result=(output_exponent<<23)|((normalized-16)<<19)
    result=torch.where((exponent>0)&(output_exponent>0),result,0)
    return torch.where(exponent==255,0x7f800000,result)


def test_cpu_retained_boundaries_all_finite_exponents_and_infinity():
    bits=edge_input_bits()
    value=bits.view(torch.float32)
    quotient=value/torch.full_like(value,6.)
    cross=(value.double()/torch.full_like(value.double(),6.)).float()
    assert torch.equal(quotient.view(torch.int32),cross.view(torch.int32))
    expected=quotient.view(torch.int32)&(-524288)
    expected=torch.where(expected<0x00800000,0,expected)
    assert torch.equal(cpu_formula(bits),expected)
    short=torch.arange(64,128,dtype=torch.int32)
    assert torch.equal(short//3,(short*171)>>9)
    # Minimum-normal crossing: significand M=3*2^22, not an earlier RN value.
    boundary=torch.tensor([(3<<23)|((3<<22)-1-0x800000),
                           (3<<23)|((3<<22)-0x800000)],dtype=torch.int32)
    assert cpu_formula(boundary).tolist()==[0,0x00800000]


def test_helper_import_does_not_initialize_cuda():
    initialized=torch.cuda.is_initialized()
    from rtx.kernels.nvfp4_scale_bits import producer_scale_rtz4
    assert producer_scale_rtz4 is not None
    assert torch.cuda.is_initialized()==initialized


run_cuda=pytest.mark.skipif(os.environ.get("RTX_RUN_SCALE_BITS_CUDA")!="1",
    reason="explicit assigned idle GPU qualification opt-in")


@run_cuda
def test_cuda_bits_match_literal_ptx_rn_divide6_oracle_and_graph_replay():
    import cutlass
    import cutlass.cute as cute
    import cuda.bindings.driver as cuda
    from cutlass import Float32,Int32
    from cutlass.experimental.primitives import nvvm_wrapper as nvvm
    from rtx.kernels.nvfp4_scale_bits import producer_scale_rtz4

    class ScaleProbe:
        def __init__(self,count):self.count=count
        @cute.jit
        def __call__(self,source:cute.Tensor,out:cute.Tensor,stream:cuda.CUstream):
            self.kernel(source,out).launch(grid=((self.count+255)//256,1,1),block=(256,1,1),stream=stream)
        @cute.kernel
        def kernel(self,source:cute.Tensor,out:cute.Tensor):
            thread,_,_=cute.arch.thread_idx()
            block,_,_=cute.arch.block_idx()
            i=block*256+thread
            if i<self.count:
                maximum=source[i]
                # Literal PTX denominator avoids reciprocal substitution and
                # the installed wrapper's Float32-literal read-operand bug.
                quotient=nvvm.inline_ptx_hl("div.rn.f32 {$w0}, {$r0}, 0f40c00000;",
                    write_only_types=[Float32],read_only_args=[maximum])
                expected=(quotient.bitcast(Int32)&Int32(-524288)).bitcast(Float32)
                if expected<Float32(1.1754943508222875e-38):expected=Float32(0.)
                out[2*i]=producer_scale_rtz4(maximum)
                out[2*i+1]=expected

    generator=torch.Generator().manual_seed(997)
    bits=torch.cat((edge_input_bits(),torch.randint(0,0x7f800000,(65536,),dtype=torch.int32,generator=generator)))
    source=bits.view(torch.float32).cuda()
    output=torch.empty(2*len(bits),device="cuda",dtype=torch.float32)
    fake=lambda size:cute.runtime.make_fake_tensor(Float32,(size,),stride=(1,),assumed_align=4)
    runner=cute.compile(ScaleProbe(len(bits)),fake(len(bits)),fake(2*len(bits)),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),options="--enable-tvm-ffi --opt-level 3")
    runner(source,output)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):runner(source,output)
    for current in (bits,bits.flip(0)):
        source.copy_(current.view(torch.float32))
        graph.replay();torch.cuda.synchronize()
        actual=output.view(-1,2).view(torch.int32)
        assert torch.equal(actual[:,0],actual[:,1])
        assert torch.equal(actual[:,0].cpu(),cpu_formula(current))
