"""Small independent oracle and graph replay for shared-panel expert dW."""

from dataclasses import replace

import pytest
import torch

from rtx.configs.moe_backward import persistent_dw_rejection
from rtx.configs.mxfp8 import MXFP8GemmConfig
from rtx.kernels.mxfp8 import MXFP8Problem


CONFIG = MXFP8GemmConfig(
    tile_m=64, tile_n=64, atom_layout_m=2, atom_layout_n=2,
    scale_load_vec=1, epilogue="direct", store_vec=1,
)


def test_persistent_dw_resource_and_ownership_legality():
    p = MXFP8Problem(256, 256, 160)
    assert persistent_dw_rejection(p, CONFIG, "n", 2) is None
    assert persistent_dw_rejection(p, CONFIG, "m", 3) is None
    assert "capacity" in persistent_dw_rejection(p, replace(CONFIG, stages=3), "m", 3)
    assert "neighboring" in persistent_dw_rejection(MXFP8Problem(64, 256, 160), CONFIG, "m", 2)
    assert "32-value" in persistent_dw_rejection(MXFP8Problem(256, 256, 161), CONFIG, "n", 2)
    assert "singleton" in persistent_dw_rejection(MXFP8Problem(70, 1, 32), CONFIG, "m", 2)
    assert "singleton" in persistent_dw_rejection(MXFP8Problem(1, 70, 32), CONFIG, "n", 2)
    assert "without generic persistence" in persistent_dw_rejection(
        p, replace(CONFIG, tiles_per_cta=2), "n", 2,
    )
    assert persistent_dw_rejection(p, CONFIG, "n", True) is not None


def _packed(experts, rows, k, rng):
    # Quantized values are exact and intentionally varied. Choose independent
    # E8M0 bytes per expert/row/block, including scale changes at K128 edges.
    q = torch.randint(-24, 25, (experts * rows, k), device="cuda", generator=rng)
    s = torch.randint(119, 132, (experts * rows, k // 32), device="cuda", generator=rng, dtype=torch.uint8)
    q = q.to(torch.float8_e4m3fn)
    return q, s


def _oracle(qa, qb, sa, sb, experts, m, n, k):
    def expand(data, scales, rows):
        return (
            data.double().reshape(experts, rows, k // 32, 32)
            * torch.exp2(scales.double() - 127).view(experts, rows, k // 32, 1)
        ).flatten(-2)
    return (expand(qa, sa, m) @ expand(qb, sb, n).mT).bfloat16().flatten(0, 1)


def _check(out, ref):
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()
    delta = (out.float() - ref.float()).abs()
    max_abs = float(delta.max())
    nrms = float(delta.norm() / ref.float().norm().clamp_min(1e-8))
    assert nrms < 0.002, (nrms, max_abs)
    assert max_abs <= 0.004 * float(ref.float().abs().max()), (nrms, max_abs)


@pytest.mark.parametrize(
    "axis,tiles,experts,m,n,k,stages,schedule",
    [
        ("n", 2, 2, 73, 149, 416, 2, "before_wait"),
        ("m", 2, 1, 131, 79, 384, 3, "after_wait"),
        ("m", 3, 3, 207, 71, 288, 2, "before_wait"),
        ("n", 3, 2, 78, 229, 160, 2, "after_wait"),
        ("n", 2, 3, 2, 70, 32, 2, "before_wait"),
        ("m", 2, 1, 70, 2, 32, 2, "after_wait"),
    ],
)
def test_persistent_dw_partition_tail_and_graph(axis, tiles, experts, m, n, k, stages, schedule):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("SM120 CUDA required")
    from rtx.kernels.mxfp8_moe_persistent_dw import compile_persistent_dw

    rng = torch.Generator(device="cuda").manual_seed(75182)
    qa, sa = _packed(experts, m, k, rng)
    qb, sb = _packed(experts, n, k, rng)
    # Simulate padded expert slots within the final quantization block.
    qb = qb.float()
    qb.reshape(experts, n, k)[0, :, -11:] = 0
    qb = qb.to(torch.float8_e4m3fn)
    # Leave aligned guards on both sides of the direct global output.
    hold = torch.full((experts * m * n + 32,), 47, device="cuda", dtype=torch.bfloat16)
    out = hold[16:-16].view(experts * m, n)
    cfg = replace(CONFIG, stages=stages, scale_schedule=schedule)
    fn = compile_persistent_dw(experts, MXFP8Problem(m, n, k), cfg, axis, tiles)
    fn(qa, qb, sa, sb, out)
    _check(out, _oracle(qa, qb, sa, sb, experts, m, n, k))
    assert (hold[:16] == 47).all() and (hold[-16:] == 47).all()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn(qa, qb, sa, sb, out)
    for _ in range(3):
        fresh_q, fresh_s = _packed(experts, m, k, rng)
        qa.copy_(fresh_q)
        sa.copy_(fresh_s)
        out.fill_(torch.nan)
        graph.replay()
        _check(out, _oracle(qa, qb, sa, sb, experts, m, n, k))
        assert (hold[:16] == 47).all() and (hold[-16:] == 47).all()
    torch.cuda.synchronize()
    live = torch.cuda.memory_allocated()
    for _ in range(8):
        graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == live
