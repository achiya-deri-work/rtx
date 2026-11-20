"""Never spend a trial on on-chip fields unused by the selected backends."""
from dataclasses import replace

import pytest

from rtx.autotune.core import stable_id
from rtx.autotune.mlp import active_schedule
from rtx.kernels.mxfp8_bwd import DEFAULT_FUSED_MXFP8_BWD_CONFIG
from rtx.mlp import MXFP8MLPConfig


def identity(c,phase):
    return stable_id(active_schedule(c,'mxfp8',phase))


@pytest.mark.parametrize('gate_backend',['onchip','prequant','cooperative_prequant'])
@pytest.mark.parametrize('recompute_backend',['onchip','prequant'])
@pytest.mark.parametrize('phase',['forward','train'])
def test_onchip_geometry_is_active_only_when_executed(gate_backend,recompute_backend,phase):
    base=replace(MXFP8MLPConfig(),gate_backend=gate_backend,
        recompute_backend=recompute_backend,gate_backward=DEFAULT_FUSED_MXFP8_BWD_CONFIG,
        down_backward=DEFAULT_FUSED_MXFP8_BWD_CONFIG)
    changed=replace(base,gate_up=replace(base.gate_up,tile_m=128,
        maxrregcount=128,quant_vec=8,a_swizzle='64b'))
    active=gate_backend=='onchip' or (phase=='train' and recompute_backend=='onchip')
    assert (identity(base,phase)!=identity(changed,phase))==active
    # This field is read and overrides subordinate SiLU policies in all paths.
    arithmetic=replace(base,gate_up=replace(base.gate_up,glu_silu_math='shared_reciprocal'))
    assert identity(base,phase)!=identity(arithmetic,phase)


def test_prepared_tiles_and_quantization_remain_distinct():
    base=replace(MXFP8MLPConfig(),gate_backend='prequant',recompute_backend='prequant',
        down_backend='prequant',gate_backward=DEFAULT_FUSED_MXFP8_BWD_CONFIG,
        down_backward=DEFAULT_FUSED_MXFP8_BWD_CONFIG)
    variants=[replace(base,gate_prequant=replace(base.gate_prequant,
                gemm=replace(base.gate_prequant.gemm,stages=3))),
              replace(base,gate_prequant=replace(base.gate_prequant,
                  quant=replace(base.gate_prequant.quant,
                      quant_vec=4 if base.gate_prequant.quant.quant_vec==8 else 8))),
              replace(base,recompute=replace(base.recompute,
                gemm=replace(base.recompute.gemm,stages=3))),
              replace(base,down_gemm=replace(base.down_gemm,stages=3))]
    assert all(identity(c,'train')!=identity(base,'train') for c in variants)
    # Recompute is unused for forward-only selection.
    assert identity(variants[2],'forward')==identity(base,'forward')
