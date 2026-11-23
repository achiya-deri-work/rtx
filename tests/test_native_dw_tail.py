"""Long dW native-scale transport with logical (not BF16) K padding."""
from dataclasses import replace

import pytest
import torch

from rtx._bwd_search import backward_search_space, backward_structural_seeds
from rtx.bwd_autotune import BWD_SEARCH_SPACE, bwd_config_from_dict, bwd_config_to_dict, update_bwd_config
from rtx.kernels.mxfp8 import MXFP8Problem
from rtx.kernels.mxfp8_bwd import DEFAULT_FUSED_MXFP8_BWD_CONFIG as BASE
from tests.test_balanced_split import quantized


def test_reported_shape_reaches_native_transport_in_light_opening():
    problem = MXFP8Problem(138304, 384, 384)
    seeds = backward_structural_seeds(problem, BASE, sm_count=170)
    candidates = [c for c in seeds[:3] if c.dw.gemm.scale_role == "tma"]
    assert len(candidates) == 2
    assert len({c.dw.split_reduction for c in candidates}) == 2
    dw = MXFP8Problem(384, 384, problem.m)
    for config in candidates:
        assert config.implementation_rejection(problem) is None
        assert config.dw._packed_reduction_k(dw) == 138368
        assert bwd_config_from_dict(bwd_config_to_dict(config)) == config.normalized()
        assert config.dw.quant_a.transposed_rejection(dw.m, dw.k) is not None
        for bad in (
            replace(config.dw, quant_launches="dual"),
            replace(config.dw, split_partition="uniform", split_reduction=1,
                reduction="full_fp32", workspace_epilogue="none"),
            replace(config.dw, quant_a=replace(config.dw.quant_a,
                transposed_load_engine="cp_async", transposed_smem_padding=0)),
        ):
            assert bad.implementation_rejection(dw) is not None
    axes = backward_search_space(problem, BWD_SEARCH_SPACE)
    assert any(update_bwd_config(BASE, move).dw._pads_native_reduction(dw)
               for move in axes["dw_balanced"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="assigned CUDA GPU required")
@pytest.mark.parametrize("k,tile,store,scale_layout,rows", [
    (4160, 32, "scalar", "mma128", 128),
    (4161, 64, "scalar", "mma128", 128),
    (4097, 128, "packed", "mma128", 128),
    (4128, 32, "scalar", "mma64", 64),
])
def test_quantizer_writes_complete_tail_and_matches_zero_pad_oracle(k, tile, store, scale_layout, rows):
    from rtx.configs.mxfp8 import MXFP8QuantConfig
    from rtx.kernels.mxfp8_quant import compile_mxfp8_transposed_quant

    torch.manual_seed(652791)
    x = torch.randn(k, rows, device="cuda", dtype=torch.bfloat16)
    padded_k = (k + 127) // 128 * 128
    padded_x = torch.nn.functional.pad(x, (0, 0, 0, padded_k - k))
    config = MXFP8QuantConfig(scale_layout=scale_layout, transposed_tile_rows=rows,
        transposed_tile_k=tile, native_scale_store=store)
    q = torch.empty(rows, padded_k, dtype=torch.float8_e4m3fn, device=x.device)
    sq = torch.empty(1, padded_k // 128, 512, dtype=torch.float8_e8m0fnu, device=x.device)
    expected, expected_sq = torch.empty_like(q), torch.empty_like(sq)
    sq.view(torch.uint8).fill_(255)
    expected_sq.view(torch.uint8).fill_(255)
    q.view(torch.uint8).fill_(255)
    compile_mxfp8_transposed_quant(rows, k, config, pad_native_k=True)(x, q, sq)
    compile_mxfp8_transposed_quant(rows, padded_k, config)(padded_x, expected, expected_sq)
    torch.testing.assert_close(q.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(sq.view(torch.uint8), expected_sq.view(torch.uint8), rtol=0, atol=0)
    assert not q[:, k:].float().count_nonzero()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="assigned CUDA GPU required")
@pytest.mark.parametrize("storage", ["separate", "serial_reuse"])
def test_compiled_mxfp8_linear_native_tail_weight_gradient(storage):
    import rtx
    from rtx._bwd_search import balanced_dw_candidates, materialized_bwd_seed

    torch.manual_seed(58135)
    torch.set_float32_matmul_precision("highest")
    problem = MXFP8Problem(8384, 128, 128)
    seed = materialized_bwd_seed(problem, BASE, 8)
    config = next(c for c in balanced_dw_candidates(problem, seed, sm_count=170)
                  if c.dw.gemm.scale_layout == "mma128")
    config = replace(config, operand_storage=storage)
    x = torch.randn(problem.m, problem.k, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    layer = rtx.MXFP8Linear(problem.k, problem.n, device=x.device, autotune="off",
                            backward_config=config)
    dy = torch.randn(problem.n, problem.m, device=x.device, dtype=torch.bfloat16).T
    y = torch.compile(layer, fullgraph=True, dynamic=False)(x)
    dx, dw = torch.autograd.grad(y, (x, layer.weight), dy)
    want_dw = (quantized(dy.T) @ quantized(x.T).T).bfloat16()
    want_dx = (quantized(dy) @ quantized(layer.weight.T).T).bfloat16()
    for actual, expected in ((dw, want_dw), (dx, want_dx)):
        assert actual.isfinite().all()
        error = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12)
        assert error < .002
