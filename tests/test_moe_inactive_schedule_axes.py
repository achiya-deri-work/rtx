"""Drop only proved identical programs, not weak empirical pruning rules."""
from dataclasses import replace
import pytest

from rtx.autotune.moe import make_moe_adapter
from rtx._moe_execution import Builder


@pytest.mark.parametrize('preparation',('quantize_once','indexed_transpose'))
@pytest.mark.parametrize('phase',('forward','train'))
def test_fp8_gather_modes_are_one_canonical_candidate(preparation,phase):
    shape=(16384,384,32,2048,4,320)
    adapter=make_moe_adapter(shape,None,phase=phase)
    cfg=adapter.update_fn(adapter.initial_config,'prepared.input_preparation',preparation)
    if preparation=='indexed_transpose':
        cfg=adapter.update_fn(cfg,'prepared.transpose_mode','warp')
        cfg=adapter.update_fn(cfg,'prepared.backward_preparation','joint')
    a,b=replace(cfg,gather_mode='flat'),replace(cfg,gather_mode='warp')
    assert adapter.config_id(a)==adapter.config_id(b)
    assert adapter.deserialize(adapter.serialize(b)).gather_mode=='flat'
    # Independent raw planner comparison: don't prove equivalence by calling
    # the new normalization twice. Every step and logical buffer must match.
    for direction in ('forward','backward'):
        x,y=Builder(shape,a,direction).build(),Builder(shape,b,direction).build()
        assert x.steps==y.steps and x.buffers==y.buffers and x.outputs==y.outputs


@pytest.mark.parametrize('family',('grouped_prepared','grouped_staged'))
def test_real_bf16_gather_modes_remain_distinct(family):
    adapter=make_moe_adapter((8192,384,32,1024,4,128),None)
    cfg=replace(adapter.initial_config,family=family)
    cfg=replace(cfg,prepared=replace(cfg.prepared,input_preparation='gather_bf16'))
    assert adapter.config_id(replace(cfg,gather_mode='flat'))!=adapter.config_id(replace(cfg,gather_mode='warp'))


def test_onchip_forward_preserves_real_bf16_gather_choice():
    adapter=make_moe_adapter((8192,384,32,1024,4,128),None)
    cfg=adapter.update_fn(adapter.initial_config,'prepared.forward_engine','onchip')
    cfg=replace(cfg,prepared=replace(cfg.prepared,input_preparation='quantize_once'))
    assert adapter.config_id(replace(cfg,gather_mode='flat'))!=adapter.config_id(replace(cfg,gather_mode='warp'))
