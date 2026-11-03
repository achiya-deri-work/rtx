"""Dual dY CTA-role mapping; CUDA exactness is explicitly opt-in."""
from dataclasses import replace
import os

import pytest
import torch

from rtx.kernels.nvfp4_posthoc import HadamardPreparationKernel,PosthocScaleConfig
from rtx.kernels.ms_eden_dual_dy_preparation import DualDYPreparationKernel,compile_dual_dy_preparation


def role_kernels(shape=(512,512),threads=128,blocks=256,**changes):
    common=dict(correction_decode="native",signs_mode="register",normalization_math="divide",
        partial_max_scope="warp",load_mode="coalesced_x8",async_copy=True,stages=1,
        producer_values_per_lane=4,producer_transpose_values_per_lane=16,producer_grouped_store_bits=32)
    common.update(changes)
    config=PosthocScaleConfig(threads,blocks)
    return (HadamardPreparationKernel(*shape,config,4,False,tile_order="row_major",**common),
            HadamardPreparationKernel(*shape,config,4,True,tile_order="reduction_major",**common))


@pytest.mark.parametrize("shape,threads,blocks",[((128,128),128,256),((512,512),128,3),((256,384),64,5),((128,256),512,256)])
def test_role_bijection_original_warp_tiles_partial_slots_and_shared_union(shape,threads,blocks):
    row,transposed=role_kernels(shape,threads,blocks)
    fused=DualDYPreparationKernel(row,transposed)
    warps=threads//32
    assert fused.grid_blocks==2*row.blocks and fused.partial_counts==(row.partial_count,transposed.partial_count)
    assert fused.shared_bytes==2*(127*transposed.async_row_stride+transposed.source_tile_rows)
    assert fused.shared_bytes>0  # Row role reserves this union too, despite not using it.
    assignments=[(physical%2,physical//2)for physical in range(fused.grid_blocks)]
    assert assignments[:2]==[(0,0),(1,0)]
    for role in (0,1):
        assert [block for selected,block in assignments if selected==role]==list(range(row.blocks))
        partials=[block*warps+warp for selected,block in assignments if selected==role for warp in range(warps)]
        assert partials==list(range(fused.partial_counts[role]))
    row_tiles=[];transposed_tiles=[]
    for selected,logical in assignments:
        if selected==0:
            for warp in range(warps):
                row_tiles.extend(range(logical*warps+warp,row.tiles,row.work_warp_count))
        else:
            limit=transposed.tiles//transposed.source_tile_rows
            for work in range(logical,limit,transposed.blocks):
                group=work%(transposed.rows//transposed.source_tile_rows)
                chunk=work//(transposed.rows//transposed.source_tile_rows)
                for warp in range(warps):
                    for subrow in range(8):
                        transposed_tiles.append((group*transposed.source_tile_rows+warp+subrow*warps)*(transposed.cols//128)+chunk)
    assert sorted(row_tiles)==list(range(row.tiles))
    assert sorted(transposed_tiles)==list(range(transposed.tiles))


@pytest.mark.parametrize("changes",[dict(signs_mode="reload"),dict(correction_decode="lut"),
    dict(normalization_math="refine"),dict(partial_max_scope="cta"),dict(producer_scale_math="integer_rtz4"),
    dict(stages=2),dict(producer_grouped_store_bits=16),dict(producer_transpose_values_per_lane=8),
    dict(producer_row_load_bits=32),dict(producer_unroll_rows=False)])
def test_unsupported_options_reject_before_cuda_compile(changes):
    with pytest.raises(ValueError):
        compile_dual_dy_preparation(*role_kernels(**changes))


def test_incompatible_roles_rejected_without_cuda():
    initialized=torch.cuda.is_initialized()
    row,transposed=role_kernels()
    for other in (role_kernels(threads=64)[1],role_kernels(blocks=3)[1],role_kernels((512,256))[1],row):
        with pytest.raises(ValueError):DualDYPreparationKernel(row,other)
    with pytest.raises(ValueError):DualDYPreparationKernel(transposed,row)
    assert torch.cuda.is_initialized()==initialized


run_cuda=pytest.mark.skipif(os.environ.get("RTX_RUN_DUAL_DY_PREPARATION_CUDA")!="1",
    reason="explicit assigned idle GPU dual-producer exact qualification")


@run_cuda
@pytest.mark.parametrize("shape,blocks",[((128,128),256),((512,512),3),((256,384),5)])
def test_dual_dy_all_prepared_arrays_exact_and_changed_input_graph(shape,blocks):
    from benchmarks.study_ms_eden_backward import prepare
    from rtx.autotune.posthoc import PosthocPreparationSchedule
    torch.manual_seed(996)
    source=torch.randn(*shape,device="cuda",dtype=torch.bfloat16)
    signs=[torch.ones(128,device="cuda")for _ in range(2)]
    row,transposed=role_kernels(shape,blocks=blocks)
    runner,counts=compile_dual_dy_preparation(row,transposed)
    base=PosthocPreparationSchedule(producer_blocks=blocks,producer_load="coalesced_x8",
        producer_async_copy=True,producer_stages=1,producer_signs="register",correction_decode="native",
        producer_values_per_lane=4,producer_transpose_values_per_lane=16,producer_grouped_store_bits=32)
    expected=[prepare(source,transpose,"mma128",replace(base,producer_order=order))
              for transpose,order in ((False,"row_major"),(True,"reduction_major"))]
    actual=[prepare(source,transpose,"mma128",replace(base,producer_order=order))
            for transpose,order in ((False,"row_major"),(True,"reduction_major"))]
    for index in range(2):assert actual[index]["partials"].numel()==counts[index]
    def run():
        runner(source.flatten(),signs[0],actual[0]["packed"].view(torch.uint8).flatten(),
            actual[0]["temporary_scales"],actual[0]["partials"],signs[1],
            actual[1]["packed"].view(torch.uint8).flatten(),actual[1]["temporary_scales"],actual[1]["partials"])
        # Both finalizers wait for whole fused grid, not an early per-role flag.
        for prepared in actual:prepared["finalize"]()
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):run()
    stream.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):run()
    for mode in ("gaussian","outlier","range","zero"):
        source.normal_(0,.25)
        if mode=="outlier":source[0,0]=4096
        if mode=="range":source.copy_(torch.ldexp(source.float(),torch.randint(-40,40,shape,device="cuda")).bfloat16())
        if mode=="zero":source.zero_();source.view(torch.int16)[0,0]=-32768
        for index in range(2):
            signs[index].random_(0,2).mul_(2).sub_(1)
            expected[index]["uniforms"].uniform_();actual[index]["uniforms"].copy_(expected[index]["uniforms"])
            expected[index]["run"](signs[index])
        graph.replay();torch.cuda.synchronize()
        for index in range(2):
            for key in ("packed","temporary_scales","partials","scales","factor"):
                assert torch.equal(expected[index][key].view(torch.uint8),actual[index][key].view(torch.uint8)),(mode,index,key)
