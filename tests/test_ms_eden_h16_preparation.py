"""Fused row-preparation gates; not complete-backward qualification."""
import os
import pytest
import torch
import numpy as np
from rtx.kernels.ms_eden_h16_preparation import H16PreparationKernel,compile_h16_preparation
from rtx.kernels.nvfp4_posthoc import PosthocScaleConfig,compile_hadamard_preparation


def test_h16_preparation_is_row_only_with_warp_partial_outputs():
    k=H16PreparationKernel(3,128,PosthocScaleConfig(128,8))
    assert not k.transpose and not k.packed_source
    assert k.tiles==3 and k.pairs==2 and k.partial_count==4
    with pytest.raises(TypeError):compile_h16_preparation(128,128,transpose=True)
    with pytest.raises(ValueError,match='requires grouped'):
        compile_h16_preparation(128,128,recompute_coefficients=True)


def test_four_block_h16_register_exchange_recovers_local_scale_groups():
    # Two two-block MMA batches: physical R bits own logical {0,7,3,8};
    # physical lane bits own {1,2,4,5,6}. Swap R1/L0 and R3/L1, then R2/R3.
    values=np.asarray([[16*(lane//4)+2*(lane%4)+(reg&1)
                        +128*((reg>>1)&1)+8*((reg>>2)&1)+256*(reg>>3)
                        for reg in range(16)] for lane in range(32)])
    assert sorted(values.flatten())==list(range(512))
    for reg_bit,lane_bit in ((1,0),(3,1)):
        previous=values.copy()
        for lane in range(32):
            for reg in range(16):
                if ((reg>>reg_bit)&1)!=((lane>>lane_bit)&1):
                    values[lane,reg]=previous[lane^(1<<lane_bit),reg^(1<<reg_bit)]
    reordered=values[:,[(j&3)|((j&4)<<1)|((j&8)>>1) for j in range(16)]]
    expected=np.asarray([[(lane%4)*128+(lane//4)*16+j for j in range(16)] for lane in range(32)])
    np.testing.assert_array_equal(reordered,expected)


@pytest.mark.parametrize('grouped_output,recompute_coefficients',[(False,False),(True,False),(True,True)])
def test_research_dispatch_preserves_unsupported_policies_and_restores(monkeypatch,grouped_output,recompute_coefficients):
    import functools
    from rtx.kernels import nvfp4_posthoc,ms_eden_h16_preparation
    from benchmarks.experiment_ms_eden_h16_preparation import patched_preparation
    @functools.wraps(nvfp4_posthoc.compile_hadamard_preparation)
    def retained(*args,**kwargs):return 'retained'
    monkeypatch.setattr(nvfp4_posthoc,'compile_hadamard_preparation',retained)
    def selected(*a,**kwargs):
        expected=dict(grouped_output=grouped_output)
        if recompute_coefficients:expected['recompute_coefficients']=True
        assert kwargs==expected
        return 'h16'
    monkeypatch.setattr(ms_eden_h16_preparation,'compile_h16_preparation',selected)
    with pytest.raises(RuntimeError,match='sentinel'):
        with patched_preparation(grouped_output=grouped_output,recompute_coefficients=recompute_coefficients) as records:
            run=nvfp4_posthoc.compile_hadamard_preparation
            assert run(128,128,correction_decode='native')=='h16'
            for change in (dict(transpose=True),dict(mantissa_bits=3),
                           dict(normalization_math='refine'),dict(partial_max_scope='cta')):
                assert run(128,128,correction_decode='native',**change)=='retained'
            assert len(records)==5
            raise RuntimeError('sentinel')
    assert nvfp4_posthoc.compile_hadamard_preparation is retained


def test_h16_and_tma_wrappers_compose_with_real_signatures(monkeypatch):
    import inspect
    from rtx.kernels import nvfp4_posthoc,ms_eden_h16_preparation,ms_eden_tma_preparation
    from benchmarks.experiment_ms_eden_bank_layout import patched
    from benchmarks.experiment_ms_eden_h16_preparation import patched_preparation
    original=nvfp4_posthoc.compile_hadamard_preparation
    signature=inspect.signature(original)
    monkeypatch.setattr(ms_eden_h16_preparation,'compile_h16_preparation',lambda *a,**k:'h16')
    monkeypatch.setattr(ms_eden_tma_preparation,'compile_tma_preparation',lambda **k:'tma')
    with patched('tma_xor5_1'),patched_preparation() as records:
        run=nvfp4_posthoc.compile_hadamard_preparation
        assert inspect.signature(run)==signature
        assert run(128,128,correction_decode='native')=='h16'
        assert run(128,128,transpose=True,async_copy=True,load_mode='coalesced_x8',
                   producer_transpose_values_per_lane=16)=='tma'
        assert [r['selected'] for r in records]==['h16_row','retained']
    assert nvfp4_posthoc.compile_hadamard_preparation is original


@pytest.mark.skipif(os.getenv('RTX_RUN_H16_PREPARATION_CUDA')!='1',reason='assigned GPU only')
@pytest.mark.parametrize('shape',[(3,128),(256,384),(4096,1536)])
@pytest.mark.parametrize('distribution',['gaussian','outliers','cancellation','zero','arbitrary_signs','tiny','huge'])
@pytest.mark.parametrize('grouped_output,recompute_coefficients',[(False,False),(True,False),(True,True)])
def test_h16_fused_preparation_fp64_decode_graph_and_memory(shape,distribution,grouped_output,recompute_coefficients):
    from benchmarks.compare_ms_eden_tf32_hadamard import dense_fp64_oracle
    torch.manual_seed(973)
    source=torch.randn(shape,device='cuda',dtype=torch.bfloat16)*.25
    signs=(torch.randint(2,(128,),device='cuda')*2-1).float()
    if distribution=='outliers':source.flatten()[::97].mul_(32)
    elif distribution=='cancellation':
        source.fill_(1.);source[:,1::2].fill_(-1.);source[:,::17].add_(.0078125)
    elif distribution=='zero':source.zero_()
    elif distribution=='arbitrary_signs':signs.fill_(1.001)
    elif distribution=='tiny':source.reshape(-1,128)[:,0]=2.**-110
    elif distribution=='huge':source.reshape(-1,128)[:,0]=2.**120
    config=PosthocScaleConfig(128,8)
    reference,count_ref=compile_hadamard_preparation(*shape,config,correction_decode='native')
    candidate,count_new=compile_h16_preparation(*shape,config,grouped_output=grouped_output,
                                              recompute_coefficients=recompute_coefficients)
    outputs=[];graphs=[]
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    for run,count in ((reference,count_ref),(candidate,count_new)):
        buffers=(torch.empty(source.numel()//2,device='cuda',dtype=torch.uint8),
                 torch.empty(source.numel()//16,device='cuda',dtype=torch.bfloat16),
                 torch.empty(count,device='cuda'))
        with torch.cuda.stream(stream):
            for _ in range(3):run(source.flatten(),signs,*buffers)
        stream.synchronize();g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g,stream=stream):run(source.flatten(),signs,*buffers)
        graphs.append(g);outputs.append(buffers)
    lut=torch.tensor([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6],device='cuda')
    for iteration in range(2):
        if iteration:source.mul_(.5);signs.neg_()
        for graph in graphs:graph.replay()
        torch.cuda.synchronize()
        oracle=dense_fp64_oracle(source,signs)
        decoded=[]
        for packed,scales,partials in outputs:
            assert torch.isfinite(scales).all() and torch.isfinite(partials).all()
            codes=torch.stack((packed&15,packed>>4),1).flatten().long()
            decoded.append(lut[codes]*scales.float().repeat_interleave(16))
            torch.testing.assert_close(partials.max(),scales.float().max(),rtol=0,atol=0)
        denominator=oracle.norm().clamp_min(1e-300)
        errors=[float((d.double()-oracle).norm()/denominator) for d in decoded]
        difference=float((decoded[0].double()-decoded[1]).norm()/denominator)
        assert errors[1]<=errors[0]+1e-4,errors
        assert difference<=.005,difference
        if distribution in ('zero','arbitrary_signs','tiny','huge'):
            for a,b in zip(outputs[0][:2],outputs[1][:2]):
                assert torch.equal(a.view(torch.uint8),b.view(torch.uint8))
        print(dict(shape=shape,distribution=distribution,grouped_output=grouped_output,iteration=iteration,
                   decoded_nrms=errors,candidate_reference_nrms=difference))
    before=torch.cuda.memory_allocated()
    for _ in range(16):
        for graph in graphs:graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated()==before
