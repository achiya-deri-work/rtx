"""Pure-transform feasibility/qualification, not production MS-EDEN evidence.

MMA association differences are measured against independent FP64 H128.
Only unsafe/sign fallback is required to match the original scalar route.
"""
from dataclasses import replace
import os

import numpy as np
import pytest
import torch

from rtx.kernels.ms_eden_tf32_hadamard import TF32H8Config,TF32H8HadamardKernel,compile_tf32_h8_hadamard


def logical_index(m,n):
    return ((m&7)<<4)|((m>>3)<<1)|(n&1)|((n>>1)<<2)


def hadamard_matrix(size,dtype=np.float64):
    return np.asarray([[1 if (i&j).bit_count()%2==0 else -1 for j in range(size)]
                       for i in range(size)],dtype=dtype)


def butterfly(values,bits):
    values=values.copy()
    for bit in bits:
        width=1<<bit
        groups=values.reshape(len(values),-1,2,width)
        a,b=groups[:,:,0,:].copy(),groups[:,:,1,:].copy()
        groups[:,:,0,:]=a+b;groups[:,:,1,:]=a-b
    return values


def test_installed_cute_tf32_layout_and_rank3_mma_ir_without_gpu():
    from cutlass._mlir import ir
    import cutlass._mlir.dialects.cute as cute_ir
    import cutlass._mlir.dialects.cute_nvgpu as nvgpu_ir
    import cutlass.cute as cute
    from cutlass import Float32,TFloat32
    from cutlass.cute.nvgpu.warp import MmaTF32Op
    with ir.Context() as context:
        cute_ir.register_dialect(context);nvgpu_ir.register_dialect(context)
        with ir.Location.unknown():
            module=ir.Module.create()
            with ir.InsertionPoint(module.body):
                atom=cute.make_mma_atom(MmaTF32Op((16,8,8)))
                all_a=[]
                for lane in range(32):
                    for reg in range(4):
                        offset=atom.tv_layout_A((lane,reg))
                        m,k=offset%16,offset//16
                        expected=(lane&1)+((lane&2)<<1)+((lane&28)<<2)+((reg&1)<<1)+((reg>>1)<<3)
                        assert logical_index(m,k)==expected
                        all_a.append(expected)
                        offset=atom.tv_layout_C((lane,reg))
                        assert logical_index(offset%16,offset//16)==4*lane+reg
                assert sorted(all_a)==list(range(128))
                a=cute.make_rmem_tensor((4,1,1),TFloat32)
                b=cute.make_rmem_tensor((2,1,1),TFloat32)
                acc=cute.make_rmem_tensor((4,1,1),Float32)
                a.fill(TFloat32(1.));b.fill(TFloat32(1.));acc.fill(Float32(0.))
                cute.gemm(atom,acc,a,b,acc)
                assert module.operation.verify()


def test_bit_factorization_exactly_matches_h128_over_all_basis_vectors():
    m,n=np.indices((16,8))
    index=logical_index(m,n)
    basis=np.eye(128,dtype=np.int64)
    prefix=np.einsum('brk,kn->brn',basis[:,index],hadamard_matrix(8,np.int64))
    output=np.empty_like(basis);output[:,index]=prefix
    np.testing.assert_array_equal(butterfly(output,(1,4,5,6)),hadamard_matrix(128,np.int64))


def test_h16_two_block_mapping_uses_full_bf16_mma_without_padding():
    """Feasibility proof only: two H16 MMA outputs plus three FP32 stages.

    Unlike padded H8, every A/B element is useful. Output ownership is NOT
    the scalar producer's ownership; this test does not qualify an integrated
    quantizer, floating-point association, or performance.
    """
    from cutlass._mlir import ir
    import cutlass._mlir.dialects.cute as cute_ir
    import cutlass._mlir.dialects.cute_nvgpu as nvgpu_ir
    import cutlass.cute as cute
    from cutlass import BFloat16,Float32
    from cutlass.cute.nvgpu.warp import MmaF16BF16Op
    basis=np.eye(256,dtype=np.int64)
    outputs=np.empty_like(basis)
    with ir.Context() as context:
        cute_ir.register_dialect(context);nvgpu_ir.register_dialect(context)
        with ir.Location.unknown():
            module=ir.Module.create()
            with ir.InsertionPoint(module.body):
                atom=cute.make_mma_atom(MmaF16BF16Op(BFloat16,Float32,(16,8,16)))
                a=np.empty((256,16,16),dtype=np.int64)
                owned=[]
                for lane in range(32):
                    for reg in range(8):
                        off=atom.tv_layout_A((lane,reg));m,k=off%16,off//16
                        index=(m%8)*16+k+(m//8)*128
                        expected=16*(lane//4)+2*(lane%4)+(reg&1)+128*((reg>>1)&1)+8*(reg>>2)
                        assert index==expected
                        a[:,m,k]=basis[:,index];owned.append(index)
                assert sorted(owned)==list(range(256))
                destinations=[]
                for half in range(2):
                    b=np.empty((16,8),dtype=np.int64)
                    for lane in range(32):
                        for reg in range(4):
                            off=atom.tv_layout_B((lane,reg));n,k=off%8,off//8
                            b[k,n]=1 if (k&(n+8*half)).bit_count()%2==0 else -1
                    partial=a@b
                    for lane in range(32):
                        for reg in range(4):
                            off=atom.tv_layout_C((lane,reg));m,n=off%16,off//16
                            index=(m%8)*16+n+8*half+(m//8)*128
                            expected=16*(lane//4)+2*(lane%4)+(reg&1)+128*(reg>>1)+8*half
                            assert index==expected
                            outputs[:,index]=partial[:,m,n];destinations.append(index)
                assert sorted(destinations)==list(range(256))
    transformed=butterfly(outputs,(4,5,6))
    expected=np.zeros((256,256),dtype=np.int64)
    expected[:128,:128]=expected[128:,128:]=hadamard_matrix(128,np.int64)
    np.testing.assert_array_equal(transformed,expected)


def test_bf16_padded_installed_traits_packed_mapping_and_mma_ir_without_gpu():
    from cutlass._mlir import ir
    import cutlass._mlir.dialects.cute as cute_ir
    import cutlass._mlir.dialects.cute_nvgpu as nvgpu_ir
    import cutlass.cute as cute
    from cutlass import BFloat16,Float32,Uint32
    from cutlass.cute.nvgpu.warp import MmaF16BF16Op
    with ir.Context() as context:
        cute_ir.register_dialect(context);nvgpu_ir.register_dialect(context)
        with ir.Location.unknown():
            module=ir.Module.create()
            with ir.InsertionPoint(module.body):
                atom=cute.make_mma_atom(MmaF16BF16Op(BFloat16,Float32,(16,8,16)))
                basis=np.eye(128,dtype=np.int64)
                padded_a=np.zeros((128,16,16),dtype=np.int64)
                padded_b=np.zeros((16,8),dtype=np.int64)
                for lane in range(32):
                    for reg in range(8):
                        offset=atom.tv_layout_A((lane,reg))
                        m,k=offset%16,offset//16
                        if reg<4:
                            assert logical_index(m,k)==4*lane+reg
                            padded_a[:,m,k]=basis[:,4*lane+reg]
                        else:assert k>=8  # These four BF16 elements are zero.
                    for reg in range(4):
                        offset=atom.tv_layout_B((lane,reg))
                        n,k=offset%8,offset//8
                        assert n==lane>>2
                        assert k==2*(lane&3)+(reg&1)+8*(reg>>1)
                        if k<8:padded_b[k,n]=1 if (k&n).bit_count()%2==0 else -1
                        offset=atom.tv_layout_C((lane,reg))
                        assert logical_index(offset%16,offset//16)==4*lane+reg
                a=cute.make_rmem_tensor((8,1,1),BFloat16)
                b=cute.make_rmem_tensor((4,1,1),BFloat16)
                acc=cute.make_rmem_tensor((4,1,1),Float32)
                a.fill(BFloat16(0.));b.fill(BFloat16(0.));acc.fill(Float32(0.))
                packed=cute.recast_tensor(a,Uint32)
                assert cute.size(packed)==4
                packed[0,0,0]=Uint32(0x3f803f80)
                cute.gemm(atom,acc,a,b,acc)
                assert module.operation.verify()
                partial=padded_a@padded_b
                output=np.empty_like(basis)
                m,n=np.indices((16,8));output[:,logical_index(m,n)]=partial
                np.testing.assert_array_equal(butterfly(output,(1,4,5,6)),hadamard_matrix(128,np.int64))
    # Two packed XOR masks equal independent BF16 unit-sign products,
    # including both signed zeros. No arbitrary-sign simplification is used.
    bits=np.array([0,0x8000,0x3f80,0xbf80,0x0f80,0x6f80],dtype=np.uint32)
    for low in bits:
        for high in bits:
            for sign0 in (0,1):
                for sign1 in (0,1):
                    packed=int(low)|(int(high)<<16)
                    result=packed^((sign0<<15)|(sign1<<31))
                    values=np.array([low<<16,high<<16],dtype=np.uint32).view(np.float32)
                    expected=values*np.array([(-1.)**sign0,(-1.)**sign1],dtype=np.float32)
                    np.testing.assert_array_equal(
                        np.array([result&0xffff,result>>16],dtype=np.uint32)<<16,
                        expected.view(np.uint32))


def test_guard_constants_cover_no_precision_narrowing_and_validate_api():
    # BF16-expanded FP32 values have low 16 bits zero, more than TF32 needs.
    finite=(np.arange(0x7f80,dtype=np.uint32)<<16)
    assert np.all((finite&0x1fff)==0)
    assert np.float32(2.**-96).view(np.uint32)==0x0f800000
    assert np.float32(2.**96).view(np.uint32)==0x6f800000
    assert TF32H8HadamardKernel(3,128).tiles==3
    for rows,cols in ((0,128),(1,127),(True,128)):
        with pytest.raises(ValueError):TF32H8HadamardKernel(rows,cols)
    for kwargs in (dict(threads=True),dict(threads=32),dict(blocks=0),dict(force_scalar=1),dict(mma_kind="fp16")):
        with pytest.raises(ValueError):TF32H8Config(**kwargs)


run_cuda=pytest.mark.skipif(os.environ.get("RTX_RUN_TF32_H8_CUDA")!="1",
                            reason="explicit assigned-GPU pure-transform qualification")


@pytest.fixture(params=("tf32","bf16_padded"))
def mma_kind(request):
    return request.param


def oracle(source,signs):
    h=torch.tensor(hadamard_matrix(128),device=source.device)
    return (source.double().reshape(-1,128)*signs.double())@h/np.sqrt(128.)


def compare_metrics(actual,reference):
    assert torch.isfinite(actual).all() and torch.isfinite(reference).all()
    error=actual.double()-reference
    refnorm=reference.norm().clamp_min(1e-300)
    nrms=float(error.norm()/refnorm)
    max_abs=float(error.abs().max())
    normalized_max=float(error.abs().max()/reference.abs().max().clamp_min(1e-300))
    return dict(nrms=nrms,max_abs=max_abs,normalized_max=normalized_max,denominator_floor=1e-300)


@run_cuda
@pytest.mark.parametrize("distribution",["gaussian","outliers","cancellation","zero"])
def test_tf32_full_transform_fp64_quality_and_scalar_association_report(distribution,mma_kind):
    torch.manual_seed(959)
    rows,cols=64,256
    source=torch.randn(rows,cols,device="cuda",dtype=torch.bfloat16)*.25
    if distribution=="outliers":source.flatten()[::97].mul_(32)
    elif distribution=="cancellation":
        source.fill_(1.);source[:,1::2].fill_(-1.);source[:,::17].add_(.0078125)
    elif distribution=="zero":source.zero_()
    signs=torch.ones(128,device="cuda") if distribution in ("cancellation","zero") else torch.where(torch.rand(128,device="cuda")<.5,-1.,1.)
    out=torch.empty(rows*cols,device="cuda")
    scalar=torch.empty_like(out)
    route=torch.empty(rows*cols//128,device="cuda",dtype=torch.int32)
    scalar_route=torch.empty_like(route)
    fast=compile_tf32_h8_hadamard(rows,cols,TF32H8Config(mma_kind=mma_kind))
    control=compile_tf32_h8_hadamard(rows,cols,TF32H8Config(force_scalar=True,mma_kind=mma_kind))
    fast(source.flatten(),signs,out,route);control(source.flatten(),signs,scalar,scalar_route)
    assert torch.all(route==1) and torch.all(scalar_route==0)
    reference=oracle(source,signs).flatten()
    measured=compare_metrics(out,reference)
    baseline=compare_metrics(scalar,reference)
    # This is a new pure-transform quality gate, not a relaxed gradient gate.
    # MMA IEEE rounding/order is unspecified; actual device output is measured.
    assert measured["nrms"]<=1e-5 and measured["normalized_max"]<=1e-5
    assert baseline["nrms"]<=1e-5 and baseline["normalized_max"]<=1e-5
    print(dict(distribution=distribution,mma_kind=mma_kind,mma=measured,scalar=baseline,
        changed_fp32_words=int((out.view(torch.int32)!=scalar.view(torch.int32)).sum()),
        scope="Pure original-BF16 Hadamard128; not quantization, EDEN, gradient or performance qualification"))


@run_cuda
@pytest.mark.parametrize("unsafe",["subnormal","tiny_normal","huge","non_rademacher","zero_sign","nan_sign"])
def test_unsafe_range_or_arbitrary_signs_preserve_scalar_fallback(unsafe,mma_kind):
    rows,cols=3,128  # Includes an idle fourth warp.
    source=torch.randn(rows,cols,device="cuda",dtype=torch.bfloat16)*.25
    signs=torch.ones(128,device="cuda")
    if unsafe=="subnormal":source[:,0].fill_(2.**-133)
    elif unsafe=="tiny_normal":source[:,0].fill_(2.**-110)
    elif unsafe=="huge":
        source.zero_();source[:,0].fill_(2.**120)
    elif unsafe=="non_rademacher":signs[7]=.3
    elif unsafe=="zero_sign":signs[7]=-0.
    elif unsafe=="nan_sign":signs[7]=float("nan")
    out=torch.empty(source.numel(),device="cuda");expected=torch.empty_like(out)
    route=torch.empty(rows,device="cuda",dtype=torch.int32);other=torch.empty_like(route)
    compile_tf32_h8_hadamard(rows,cols,TF32H8Config(mma_kind=mma_kind))(source.flatten(),signs,out,route)
    compile_tf32_h8_hadamard(rows,cols,TF32H8Config(force_scalar=True,mma_kind=mma_kind))(source.flatten(),signs,expected,other)
    assert torch.all(route==0)
    finite=~torch.isnan(expected)
    assert torch.equal(torch.isnan(out),torch.isnan(expected))
    torch.testing.assert_close(out[finite].view(torch.int32),expected[finite].view(torch.int32),atol=0,rtol=0)
    if unsafe!="nan_sign":assert compare_metrics(out,oracle(source,signs).flatten())["nrms"]<=1e-5


@run_cuda
def test_guarded_capture_reads_changed_sources_and_signs_and_keeps_live_memory_stable(mma_kind):
    rows,cols=5,384
    source=torch.randn(rows,cols,device="cuda",dtype=torch.bfloat16)*.25
    signs=torch.ones(128,device="cuda")
    out=torch.empty(source.numel(),device="cuda");expected=torch.empty_like(out)
    route=torch.empty(source.numel()//128,device="cuda",dtype=torch.int32);other=torch.empty_like(route)
    fast=compile_tf32_h8_hadamard(rows,cols,TF32H8Config(mma_kind=mma_kind))
    control=compile_tf32_h8_hadamard(rows,cols,TF32H8Config(force_scalar=True,mma_kind=mma_kind))
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):fast(source.flatten(),signs,out,route)
    stream.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):fast(source.flatten(),signs,out,route)
    # Scalar signs are cached only within a launch, never across graph replays.
    # Capture this control too so later sign mutations exercise that contract.
    control(source.flatten(),signs,expected,other);torch.cuda.synchronize()
    control_graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(control_graph,stream=stream):control(source.flatten(),signs,expected,other)
    graph.replay();torch.cuda.synchronize()
    assert torch.all(route==1)
    source.mul_(.5);signs[5]=.75  # Must switch all warps to exact scalar fallback.
    control_graph.replay();graph.replay();torch.cuda.synchronize()
    assert torch.all(route==0)
    torch.testing.assert_close(out.view(torch.int32),expected.view(torch.int32),atol=0,rtol=0)
    assert compare_metrics(expected,oracle(source,signs).flatten())["nrms"]<=1e-5
    signs.fill_(1.);source.flatten()[0]=2.**-110
    graph.replay();torch.cuda.synchronize()
    assert route[0]==0 and torch.all(route[1:]==1)
    before=torch.cuda.memory_allocated()
    for _ in range(20):graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated()==before
