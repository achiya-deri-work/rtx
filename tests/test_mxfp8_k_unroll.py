"""Shared reduction-loop scheduling across linears, GLU and recomputed MLP."""
from dataclasses import asdict, replace

import pytest
import torch
import rtx
from rtx.configs.mxfp8 import MXFP8GemmConfig
from rtx.configs.glu import MXFP8GLUPrequantConfig, prequant_glu_config_from_dict
from rtx.kernels.mxfp8 import MXFP8Problem


def test_unroll_legacy_roundtrip_and_legality():
    for cls in (MXFP8GemmConfig, rtx.MXFP8GLUPrequantConfig().gemm.__class__):
        base = cls()
        values = asdict(base)
        values.pop("k_tile_unroll")
        assert cls(**values).k_tile_unroll == 1
        for factor in (0, 5, -1, True, 1.0):
            assert "k_tile_unroll" in replace(base, k_tile_unroll=factor).rejection(MXFP8Problem(256, 256, 384))
        for factor in (1, 2, 3, 4, 6, 8, 12):
            assert cls(**asdict(replace(base, k_tile_unroll=factor))).k_tile_unroll == factor
    cfg = MXFP8GLUPrequantConfig()
    cfg = replace(cfg, gemm=replace(cfg.gemm, k_tile_unroll=6))
    assert prequant_glu_config_from_dict(asdict(cfg)) == cfg


def test_forward_and_backward_search_axes():
    from rtx.prequant_autotune import PREQUANT_SEARCH_SPACE
    from rtx.bwd_autotune import BWD_SEARCH_SPACE, update_bwd_config
    expected = (1, 2, 3, 4, 6, 8, 12)
    from rtx.autotune.glu import GLU_SEARCH_SPACE
    from rtx.kernels.mxfp8 import FWD_SEARCH_SPACE, normalize_fwd_config
    assert FWD_SEARCH_SPACE['k_unroll'] == GLU_SEARCH_SPACE['k_unroll'] == expected
    for invalid in (0, -1, True, 5):
        assert 'k_unroll' in replace(normalize_fwd_config(), k_unroll=invalid).implementation_rejection(MXFP8Problem(128,128,128))
    assert tuple(v["gemm"]["k_tile_unroll"] for v in PREQUANT_SEARCH_SPACE["gemm_k_unroll"]) == expected
    for leg in ("dx", "dw"):
        axis = BWD_SEARCH_SPACE[f"{leg}_gemm_k_unroll"]
        assert tuple(v[leg]["gemm"]["k_tile_unroll"] for v in axis) == expected
        selected = update_bwd_config(rtx.DEFAULT_MXFP8_BWD_CONFIG, axis[4])
        assert getattr(selected, leg).gemm.k_tile_unroll == 6


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@cuda
@pytest.mark.parametrize("shape", [(256, 256, 384), (257, 259, 385), (2048, 128, 256)])
@pytest.mark.parametrize("factor", [2, 3, 6])
@pytest.mark.parametrize("stages", [1, 3])
def test_linear_prequant_unroll_bitwise(shape, factor, stages):
    from rtx.fp8 import _build_prequant_runner, _intern_prequant_config
    torch.manual_seed(2143)
    m, n, k = shape
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * .2
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * .02
    base = rtx.MXFP8PrequantConfig(quant=rtx.MXFP8QuantConfig(scale_layout="row_major"), gemm=MXFP8GemmConfig(
        tile_m=128 if stages == 1 else 64, atom_layout_m=4 if stages == 1 else 2, stages=stages,
        epilogue="direct", store_vec=1, tiles_per_cta=4))
    a = _build_prequant_runner(x, w, _intern_prequant_config(base))
    b = _build_prequant_runner(x, w, _intern_prequant_config(replace(base, gemm=replace(base.gemm, k_tile_unroll=factor))))
    expected = torch.empty(m, n, device=x.device, dtype=x.dtype)
    actual = torch.empty_like(expected)
    for _ in range(2):
        a(x, w, expected)
        b(x, w, actual)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        x.mul_(-.75)
        w.mul_(1.25)


@cuda
@pytest.mark.parametrize("activation", ["silu", "gelu"])
@pytest.mark.parametrize("fmt", ["bf16", "mxfp8"])
@pytest.mark.parametrize("shape", [(128, 256, 384), (37, 194, 129)])
def test_glu_prequant_unroll_bitwise(shape, activation, fmt):
    from rtx.kernels.mxfp8_glu_prequant import build_prequant_glu_runner
    torch.manual_seed(781)
    m, n, k = shape
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * .2
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * .02
    cfg = MXFP8GLUPrequantConfig()
    width = n // 2 if fmt == "bf16" else ((n // 2 + 31) // 32) * 32
    out = torch.empty(m, width, device="cuda", dtype=torch.bfloat16 if fmt == "bf16" else torch.float8_e4m3fn)
    scales = torch.empty(m, 0 if fmt == "bf16" else width // 32, device="cuda", dtype=torch.uint8)
    baseline = build_prequant_glu_runner(x, w, out, scales, cfg, activation, fmt)
    other = build_prequant_glu_runner(x, w, out, scales,
        replace(cfg, gemm=replace(cfg.gemm, k_tile_unroll=6)), activation, fmt)
    for _ in range(2):
        baseline()
        expected, sf = out.clone(), scales.clone()
        other()
        torch.testing.assert_close(out.float(), expected.float(), rtol=0, atol=0)
        torch.testing.assert_close(scales, sf, rtol=0, atol=0)
        x.mul_(-.75)


@cuda
@pytest.mark.parametrize("activation", ["silu", "gelu"])
@pytest.mark.parametrize("fmt", ["bf16", "mxfp8"])
def test_onchip_glu_unroll_bitwise(activation, fmt):
    from rtx.kernels.mxfp8_glu import compile_mxfp8_glu
    from rtx.configs.glu import DEFAULT_GLU_CONFIG, normalize_glu_config
    torch.manual_seed(871)
    x=torch.randn(37,385,device='cuda',dtype=torch.bfloat16)*.2
    w=torch.randn(194,385,device='cuda',dtype=torch.bfloat16)*.02
    out=torch.empty(37,97 if fmt=='bf16' else 128,device='cuda',
                    dtype=torch.bfloat16 if fmt=='bf16' else torch.float8_e4m3fn)
    sf=torch.empty(37,0 if fmt=='bf16' else 4,device='cuda',dtype=torch.uint8)
    ref=None
    for factor in (1,3,6):
        cfg=normalize_glu_config(DEFAULT_GLU_CONFIG,k_unroll=factor)
        fn=compile_mxfp8_glu(MXFP8Problem(37,194,385),cfg,activation,fmt)
        aux=out if fmt=='bf16' else sf
        fn(x,w,out,aux,aux,aux,aux)
        if ref is None:ref=(out.clone(),sf.clone())
        for actual,expected in zip((out,sf),ref):
            torch.testing.assert_close(actual.float(),expected.float(),rtol=0,atol=0)


@cuda
@pytest.mark.parametrize("activation", ["silu", "gelu"])
def test_compiled_mlp_all_unroll_legs_and_search(activation):
    from rtx.autotune.mlp import make_mlp_adapter, reference
    torch.manual_seed(982)
    inputs = tuple((torch.randn(*s, device="cuda", dtype=torch.bfloat16)*.1).requires_grad_()
                   for s in ((128, 384), (512, 384), (128, 256)))
    dy = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)*.1
    cfg = replace(rtx.MXFP8MLPConfig(), gate_backend="prequant", down_backend="prequant",
                  recompute_backend="prequant", gate_backward=rtx.DEFAULT_MXFP8_BWD_CONFIG,
                  down_backward=rtx.DEFAULT_MXFP8_BWD_CONFIG)
    adapter = make_mlp_adapter(*inputs, activation=activation, initial=cfg,
                               warmup=1, samples=3, calls_per_sample=1)
    for group in ("cooperative_gemm", "down_gemm", "recompute_gemm"):
        assert f"{group}.gemm_k_unroll" in adapter.axes
    for group in ("gate_backward", "down_backward"):
        for leg in ("dx", "dw"):
            assert f"{group}.{leg}_gemm_k_unroll" in adapter.axes
    def leg(c):
        return replace(c, dx=replace(c.dx, gemm=replace(c.dx.gemm, k_tile_unroll=3)),
                       dw=replace(c.dw, gemm=replace(c.dw.gemm, k_tile_unroll=6)))
    other = replace(cfg,
        gate_prequant=replace(cfg.gate_prequant, gemm=replace(cfg.gate_prequant.gemm, k_tile_unroll=6)),
        down_gemm=replace(cfg.down_gemm, k_tile_unroll=3),
        recompute=replace(cfg.recompute, gemm=replace(cfg.recompute.gemm, k_tile_unroll=6)),
        gate_backward=leg(cfg.gate_backward), down_backward=leg(cfg.down_backward))
    assert rtx.MXFP8MLPConfig.from_dict(other.to_dict()) == other
    results = []
    for config in (cfg, other):
        torch._dynamo.reset()
        def step(x, w, d):
            return rtx.mxfp8_mlp(x, w, d, activation=activation, config=config)
        fn = torch.compile(step, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
        y = fn(*inputs)
        results.append((y, *torch.autograd.grad(y, inputs, dy)))
    with torch.no_grad():
        expected = reference(*inputs, dy, activation)
    for a, b, ref in zip(*results, expected):
        assert bool(a.isfinite().all())
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert float((a.detach().float()-ref.float()).norm()/ref.float().norm().clamp_min(1e-12)) < .025
