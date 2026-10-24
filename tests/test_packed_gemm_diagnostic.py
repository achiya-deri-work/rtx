import torch
import pytest
from dataclasses import replace

from benchmarks.compare_nvfp4_packed_gemm import decode, native_scales, row_major_scales


def test_packed_diagnostic_accepts_shape_checked_forward_winner():
    from benchmarks.compare_nvfp4_packed_gemm import gemm_config_from_record
    record = {'shape': [8192,768,1536], 'config': {'gemm': {'stages': 2}, 'quant': {}}}
    assert gemm_config_from_record(record, (8192,768,1536)).stages == 2
    with pytest.raises(ValueError, match='shape mismatch'):
        gemm_config_from_record(record, (2048,768,1536))
    record['config']['l2_fetch_granularity'] = 128
    with pytest.raises(ValueError, match='L2 fetch'):
        gemm_config_from_record(record, (8192,768,1536))


@pytest.mark.parametrize("shape", [(128,128),(256,768),(384,1536)])
def test_native_scale_inverse_is_bit_exact(shape):
    rows,k=shape
    values=torch.arange(rows*k//16).remainder(256).to(torch.uint8).reshape(rows,k//16)
    assert torch.equal(row_major_scales(native_scales(values),rows,k,"mma128"),values)
    assert torch.equal(row_major_scales(values,rows,k,"row_major"),values)


def test_independent_fp4_decode_nibble_order_and_both_scale_levels():
    indices=torch.arange(16,dtype=torch.uint8).repeat(2).reshape(1,32)
    packed=indices[:,::2] | (indices[:,1::2] << 4)
    grid=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.])
    result=decode(packed,torch.tensor([[1.,2.]]),torch.tensor(3.))
    torch.testing.assert_close(result,torch.cat((grid*3,grid*6)).reshape(1,32),rtol=0,atol=0)


def test_independent_fp4_decode_zero_scale_is_zero():
    result=decode(torch.full((2,16),0x77,dtype=torch.uint8),torch.ones(2,2),torch.tensor(0.))
    assert torch.equal(result,torch.zeros(2,32))


def test_native_scale_layout_matches_physical_index_contract():
    scales=torch.arange(256*16,dtype=torch.int64).remainder(251).to(torch.uint8).reshape(256,16)
    native=native_scales(scales)
    for row in range(256):
        for block in range(16):
            offset=(row%32)*16+((row//32)%4)*4+(block%4)+((block%8)//4)*512
            assert native[row//128,block//8,offset] == scales[row,block]


def test_coupled_geometry_seeds_reach_k64_legally():
    from benchmarks.tune_nvfp4_packed_diagnostic import geometry_seeds
    from rtx.autotune.adapters import make_nvfp4_fully_prequant_adapter
    from rtx.configs.nvfp4 import NVFP4Problem
    adapter=make_nvfp4_fully_prequant_adapter(NVFP4Problem(2048,768,1536),lambda c:None,device={})
    seeds=geometry_seeds(adapter)
    assert all(adapter.rejection(c) is None for c in seeds)
    assert len({adapter.config_id(c) for c in seeds})==len(seeds)
    assert any(c.gemm.tile_k==64 and c.gemm.a_swizzle in ("none","32b") for c in seeds)


def test_shared_nvfp4_geometry_coordinate_reaches_coupled_regions():
    from rtx.autotune.adapters import make_nvfp4_fully_prequant_adapter
    from rtx.configs.nvfp4 import NVFP4Problem
    adapter=make_nvfp4_fully_prequant_adapter(NVFP4Problem(2048,768,1536),lambda c:None,device={})
    legal=[]
    for update in adapter.axes["gemm_geometry"]:
        candidate=adapter.update_fn(adapter.initial_config,"gemm_geometry",update)
        if adapter.rejection(candidate) is None:
            legal.append(candidate.gemm)
    assert any(c.tile_k==64 and c.a_swizzle=="32b" and c.b_swizzle=="32b" for c in legal)
    assert any(c.stages==1 for c in legal)
    assert any(c.tile_k==128 and c.stages==2 for c in legal)


def test_jit_fragment_seed_can_reach_nonfragment_tiles():
    from rtx.nvfp4_inference_autotune import preferred_jit_row_region_config, NVFP4_JIT_ROW_REGION_SEARCH_SPACE, update_dynamic_config
    from rtx.configs.nvfp4 import NVFP4Problem
    problem=NVFP4Problem(8192,3072,768)
    initial=preferred_jit_row_region_config(problem)
    legal=[update_dynamic_config(initial,v) for v in NVFP4_JIT_ROW_REGION_SEARCH_SPACE['gemm_geometry']]
    legal=[c for c in legal if c.rejection(problem) is None]
    assert any(c.gemm.tile_m==256 for c in legal)
    assert any(c.gemm.tile_n==256 for c in legal)
    assert all(c.x_scale_region_rows==initial.x_scale_region_rows and
               c.weight_scale_region_rows==initial.weight_scale_region_rows and
               c.quant.block_scale_selection==initial.quant.block_scale_selection for c in legal)


@pytest.mark.parametrize("block_policy", ["amax", "mse5", "lsq1", "four_over_six"])
@pytest.mark.parametrize("outer_policy", ["power2", "exact", "exact_four_over_six"])
@pytest.mark.parametrize("family", ["dynamic", "delayed"])
def test_anchors_preserve_quantization_policy(block_policy, outer_policy, family):
    from rtx.autotune import adapters
    from rtx.configs.nvfp4 import NVFP4Problem, NVFP4DynamicConfig
    initial = NVFP4DynamicConfig()
    initial = replace(initial, quant=replace(initial.quant, block_scale_selection=block_policy),
                      tensor_scale_mode=outer_policy)
    factory = getattr(adapters, f"make_nvfp4_{family}_adapter")
    adapter = factory(NVFP4Problem(2048,768,1536), lambda c: None,
                      initial=initial, device={})
    layouts = set()
    for value in adapter.axes["implementation_anchor"]:
        candidate = adapter.update_fn(initial, "implementation_anchor", value)
        assert candidate.quant.block_scale_selection == block_policy
        assert candidate.tensor_scale_mode == outer_policy
        assert adapter.rejection(candidate) is None
        layouts.add(candidate.quant.scale_layout)
    assert layouts == {"row_major", "mma128"}
    for value in adapter.axes["scale_transport"]:
        candidate = adapter.update_fn(initial, "scale_transport", value)
        assert candidate.quant.block_scale_selection == block_policy
        assert candidate.tensor_scale_mode == outer_policy
        assert adapter.rejection(candidate) is None
