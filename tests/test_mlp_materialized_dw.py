from dataclasses import replace
from types import SimpleNamespace
from rtx.autotune.mlp import materialized_dw_config, mlp_structural_seeds
from rtx.mlp import DEFAULT_MLP_CONFIG
from rtx.fp8_bwd import _default_bwd_config
from rtx.kernels.mxfp8 import MXFP8Problem


def test_materialized_dw_ragged_reduction_preserves_dx_and_arithmetic():
    problem = MXFP8Problem(11296, 3072, 256)
    original = _default_bwd_config(problem)
    # Reproduce the structurally illegal inherited native scale layout.
    original = replace(original, dw=replace(original.dw,
        quant_a=replace(original.dw.quant_a, scale_layout='mma128'),
        gemm=replace(original.dw.gemm, scale_layout='mma128', scale_role='tma')))
    bare = replace(original, dw=replace(original.dw, backend='decomposed'))
    assert bare.rejection(problem) is not None
    candidate = materialized_dw_config(original)
    assert candidate.rejection(problem) is None
    assert candidate.implementation_rejection(problem) is None
    assert candidate.dx == original.dx
    assert candidate.dw.fused == original.dw.fused
    assert candidate.dw.quant_a.quant_math == original.dw.quant_a.quant_math
    assert candidate.dw.quant_a.quant_amax == original.dw.quant_a.quant_amax
    assert candidate.dw.quant_a.scale_rounding == original.dw.quant_a.scale_rounding
    assert candidate.dw.gemm.tile_m == original.dw.gemm.tile_m


def test_materialized_dw_preserves_separate_quantizer_policy():
    original = _default_bwd_config(MXFP8Problem(11296, 3072, 256))
    qb = replace(original.dw.quant_a, quant_vec=8)
    original = replace(original, dw=replace(original.dw, quant_b=qb))
    candidate = materialized_dw_config(original, 64)
    assert candidate.dw.quant_b.quant_vec == 8
    assert candidate.dw.quant_b.scale_layout == 'row_major'
    assert candidate.dw.quant_b.transposed_tile_rows == 64


def test_structural_seeds_include_legal_dw_for_both_mlps():
    base = replace(DEFAULT_MLP_CONFIG,
        gate_backward=_default_bwd_config(MXFP8Problem(11296,3072,256)),
        down_backward=_default_bwd_config(MXFP8Problem(11296,256,1536)))
    adapter = SimpleNamespace(initial_config=base, update_fn=lambda c,*args:c)
    seeds = mlp_structural_seeds(adapter, 'mxfp8', 'train')
    assert any(c.gate_backend == base.gate_backend and c.down_backend == 'prequant'
               for c in seeds)
    assert any(c.gate_backend == 'prequant' and c.down_backend == 'prequant'
               for c in seeds)
    assert any(c.gate_backward == materialized_dw_config(base.gate_backward) for c in seeds)
    from rtx.autotune.mlp import materialized_dx_config
    assert any(c.gate_backward == materialized_dw_config(base.gate_backward) and
               c.down_backward == materialized_dx_config(materialized_dw_config(base.down_backward))
               for c in seeds)
    for group in ('gate_backward', 'down_backward'):
        assert any(getattr(c,group) == materialized_dw_config(getattr(base,group)) for c in seeds)
    assert any(c.gate_backend == 'prequant' and c.down_backend == 'onchip'
               and c.recompute_backend == 'prequant'
               and c.gate_backward == base.gate_backward
               and c.down_backward == base.down_backward for c in seeds)
    assert any(c.recompute_backend == 'prequant'
               and c.recompute.gemm.glu_grad_load == 'prefetch'
               and c.gate_up.glu_silu_math == base.gate_up.glu_silu_math
               for c in seeds)
    complete = [c for c in seeds if c.gate_backend == 'prequant'
                and c.down_backend == 'prequant' and c.recompute_backend == 'prequant'
                and c.recompute.gemm.tile_n == 64]
    assert {c.backward_overlap for c in complete} == {False}


def test_materialized_dw_normalizes_inactive_scale_vector():
    original = _default_bwd_config(MXFP8Problem(11296, 256, 1536))
    original = replace(original, dw=replace(original.dw,
        gemm=replace(original.dw.gemm, tile_k=32, scale_load_vec=4)))
    candidate = materialized_dw_config(original)
    assert candidate.dw.gemm.scale_load_vec == 1
    assert candidate.dw.gemm.tile_k == 32


def test_materialized_dx_accepts_ragged_rows_and_preserves_dw():
    from rtx.autotune.mlp import materialized_dx_config
    problem = MXFP8Problem(11296, 256, 1536)
    original = _default_bwd_config(problem)
    candidate = materialized_dx_config(original)
    assert candidate.dx.gemm.epilogue == 'direct'
    assert candidate.dw == original.dw
    assert candidate.dx.quant_a.quant_math == original.dx.quant_a.quant_math
    assert candidate.rejection(problem) is None
    assert candidate.implementation_rejection(problem) is None


def test_decomposed_long_reduction_accepts_tile_aligned_nonpower_chunks():
    from rtx.autotune.mlp import materialized_dx_config
    problem=MXFP8Problem(11712,1536,256)
    original=materialized_dx_config(materialized_dw_config(_default_bwd_config(problem)))
    for count,tile in ((2,5888),(4,2944),(8,1536),(16,768)):
        dw=replace(original.dw,reduction='split_fp32_workspace',split_reduction=count,
                   reduction_tile=tile,workspace_epilogue='tree',
                   gemm=replace(original.dw.gemm,epilogue='direct',store_vec=1,
                                tiles_per_cta=1,epilogue_stages=1))
        candidate=replace(original,dw=dw)
        assert candidate.rejection(problem) is None
        assert candidate.implementation_rejection(problem) is None
    bad=replace(candidate,dw=replace(candidate.dw,reduction_tile=1024))
    assert 'empty partition' in bad.rejection(problem)
