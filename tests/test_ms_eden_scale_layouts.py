"""Opt-in scale-layout integration: CPU contracts and assigned-GPU gates."""
from dataclasses import asdict, replace
import json
import os

import pytest
import torch

from rtx.autotune.ms_eden_pipeline import make_ms_eden_pipeline_adapter
from rtx.autotune.posthoc import PosthocPreparationSchedule


def row_schedule(tile_m=128, tile_n=128, owner="consumers"):
    base = PosthocPreparationSchedule()
    return replace(base, gemm=replace(base.gemm, scale_layout="row_major",
        scale_role=owner, tile_m=tile_m, tile_n=tile_n,
        atom_layout_m=4 if tile_n == 256 else 8, atom_layout_n=tile_n // 64,
        epilogue="direct", store_vec=1))


def test_layout_search_is_opt_in_and_preserves_quarantine():
    default = make_ms_eden_pipeline_adapter((4096, 1536, 512), lambda _: {})
    assert "dx.scale_layout" not in default.axes
    a = make_ms_eden_pipeline_adapter((4096, 1536, 512), lambda _: {}, allow_row_major=True)
    c = a.update_fn(a.initial_config, "dx.scale_layout", "row_major")
    assert c.dw == a.initial_config.dw
    assert a.rejection(c) is None and default.rejection(c) is not None
    assert a.config_id(c) != a.config_id(a.initial_config)
    geometry = replace(c, dx=row_schedule(256, 128))
    assert a.rejection(geometry) is None
    back = a.update_fn(geometry, "dx.scale_layout", "mma128")
    assert back.dx.gemm.tile_m == back.dx.gemm.tile_n == back.dx.gemm.tile_k == 128
    assert a.rejection(back) is None
    bad = replace(c, dx=replace(c.dx, gemm=replace(c.dx.gemm, tile_m=64)))
    assert "quarantined" in a.rejection(bad)[1]
    narrow = replace(c, dx=replace(c.dx, gemm=replace(c.dx.gemm, tile_n=64)))
    assert a.rejection(narrow) is not None


def test_composable_design_seeds_reach_larger_tiles_without_relaxing_legality():
    from rtx.autotune.ms_eden_design import make_ms_eden_design_adapter
    from benchmarks.tune_ms_eden_design import layout_seeds
    default = make_ms_eden_design_adapter((4096, 1536, 512), lambda _: {})
    assert layout_seeds(default, default.initial_config) == []
    a = make_ms_eden_design_adapter((4096, 1536, 512), lambda _: {}, allow_row_major=True)
    seeds = layout_seeds(a, a.initial_config)
    assert len(seeds) == len({a.config_id(c) for c in seeds}) == 24
    assert all(a.rejection(c) is None for c in seeds)
    assert any(c.schedule.dx.gemm.tile_m == 256 for c in seeds)
    assert any(c.schedule.dw.gemm.tile_n == 256 for c in seeds)
    assert any(c.schedule.dx.gemm.scale_role == "producer" for c in seeds)


def test_scale_owner_is_opt_in_and_crosses_consumer_constraints():
    a = make_ms_eden_pipeline_adapter((4096, 1536, 512), lambda _: {}, allow_row_major=True)
    c = a.update_fn(a.initial_config, "dx.scale_layout", "row_major")
    c = replace(c, dx=replace(c.dx, gemm=replace(c.dx.gemm,
        scale_smem_store="packed", scale_recycle="staged")))
    producer = a.update_fn(c, "dx.gemm:scale_owner", a.axes["dx.gemm:scale_owner"][1])
    assert a.rejection(producer) is None
    assert producer.dw == c.dw
    assert producer.dx.gemm.scale_role == "producer"
    assert producer.dx.gemm.scale_smem_store == "scalar"
    assert producer.dx.gemm.scale_recycle == "barrier"
    assert a.config_id(producer) != a.config_id(c)
    native = a.update_fn(a.initial_config, "dx.gemm:scale_owner", a.axes["dx.gemm:scale_owner"][1])
    assert a.rejection(native) is not None
    default = make_ms_eden_pipeline_adapter((4096, 1536, 512), lambda _: {})
    assert "dx.gemm:scale_owner" not in default.axes


@pytest.mark.parametrize("phase", ["dx", "dw"])
def test_row_major_fake_contract(phase):
    from rtx.experimental_ms_eden import _gradient_fake
    dy = torch.empty((128, 256), device="meta", dtype=torch.bfloat16)
    operand = torch.empty((256 if phase == "dx" else 128, 128), device="meta", dtype=torch.bfloat16)
    signs = torch.empty(128, device="meta", dtype=torch.float32)
    ua = torch.empty(dy.numel() // 16, device="meta")
    ub = torch.empty(operand.numel() // 16, device="meta")
    out = _gradient_fake(dy, operand, signs, ua, ub, phase, json.dumps(asdict(row_schedule())))
    assert out.shape == (128 if phase == "dx" else 256, 128)
    assert out.dtype == torch.bfloat16 and out.device.type == "meta"


@pytest.mark.skipif(os.getenv("RTX_RUN_MS_EDEN_LAYOUT_CUDA") != "1", reason="assigned GPU only")
@pytest.mark.parametrize("phase", ["dx", "dw"])
@pytest.mark.parametrize("tile", [(128, 128), (256, 128), (128, 256)])
@pytest.mark.parametrize("shape", [(128, 256, 128), (512, 768, 256)])
@pytest.mark.parametrize("owner", ["consumers", "producer"])
def test_row_major_gpu_schema_compile_and_independent_reference(phase, tile, shape, owner):
    from tests.test_experimental_ms_eden import independent_reference
    from rtx.experimental_ms_eden import _gradient, ms_eden_gradient
    torch.manual_seed(901)
    m, n, k = shape
    dy = torch.randn(m, n, device="cuda", dtype=torch.bfloat16) * .25
    operand = torch.randn(n if phase == "dx" else m, k, device="cuda", dtype=torch.bfloat16) * .1
    signs = (torch.randint(2, (128,), device="cuda") * 2 - 1).float()
    args = (dy, operand, signs, torch.rand(dy.numel() // 16, device="cuda"),
            torch.rand(operand.numel() // 16, device="cuda"))
    schedule = json.dumps(asdict(row_schedule(*tile, owner=owner)))
    checks = torch.library.opcheck(_gradient, (*args, phase, schedule))
    assert set(checks.values()) == {"SUCCESS"}
    eager = ms_eden_gradient(*args, phase=phase, schedule_json=schedule)
    reference = independent_reference(args, phase)
    assert torch.isfinite(eager).all()
    assert (eager.float() - reference).norm() / reference.norm() < .005
    torch.compiler.reset()
    compiled = torch.compile(lambda *a: ms_eden_gradient(*a, phase=phase, schedule_json=schedule),
                             fullgraph=True, dynamic=False)
    for _ in range(3):
        torch.testing.assert_close(compiled(*args), eager, atol=0, rtol=0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            compiled(*args)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        replayed = compiled(*args)
    dy.mul_(.5)
    args[3].uniform_()
    args[4].uniform_()
    graph.replay()
    torch.cuda.synchronize()
    changed = independent_reference(args, phase)
    assert torch.isfinite(replayed).all()
    assert (replayed.float() - changed).norm() / changed.norm() < .005
