from dataclasses import replace

from rtx.configs.glu import MXFP8GLUGemmConfig, prequant_recompute_rejection


def test_single_buffer_derivative_quarantined_without_banning_other_schedules():
    cfg = MXFP8GLUGemmConfig(tile_m=128,tile_n=64,atom_layout_m=4,
                            atom_layout_n=2,stages=1,scale_layout='row_major')
    assert 'nondeterminism' in prequant_recompute_rejection(cfg)
    assert prequant_recompute_rejection(replace(cfg,stages=2)) is None
    assert prequant_recompute_rejection(replace(cfg,tile_n=128)) is None
    assert prequant_recompute_rejection(replace(cfg,scale_role='producer')) is None
