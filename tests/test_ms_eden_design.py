from dataclasses import asdict, replace

import pytest

from rtx.autotune.ms_eden_design import (
    MSEdenDesign, implementation_rejection, make_ms_eden_design_adapter,
)


def test_roundtrip_identity_and_schedule_axes():
    adapter = make_ms_eden_design_adapter((512, 512, 512), lambda _: {})
    initial = adapter.initial_config
    assert adapter.deserialize(adapter.serialize(initial)) == initial
    assert "dx.producer_blocks" in adapter.axes
    assert any(axis.startswith("dx.gemm:") for axis in adapter.axes)
    assert "transpose_loader" in adapter.axes
    assert "finalizer" in adapter.axes
    assert not any("registers" in name or "maxrregcount" in name for name in adapter.axes)
    assert adapter.config_id(initial) != adapter.config_id(replace(initial, finalizer="dual"))
    changed = adapter.update_fn(initial, "dx.producer_blocks", 128)
    assert changed.schedule.dx.producer_blocks == 128
    assert changed.schedule.dw == initial.schedule.dw


def test_dual_requires_its_actual_dispatch_path():
    initial = MSEdenDesign(finalizer="dual")
    assert implementation_rejection(initial)
    schedule = replace(initial.schedule, backward_pipeline="operand_paired")
    schedule = replace(schedule, **{p: replace(getattr(schedule, p), amax_reduction="staged")
                                    for p in ("dx", "dw")})
    assert implementation_rejection(replace(initial, schedule=schedule)) is None


def test_independent_operand_grids_include_device_waves():
    adapter = make_ms_eden_design_adapter((4096, 1536, 512), lambda _: {},
        device={"multiprocessor_count": 26})
    axis = "dx.operand.b.finalizer_blocks"
    assert {26, 52, 104, 208} <= set(adapter.axes[axis])
    initial = adapter.initial_config
    candidate = adapter.update_fn(initial, axis, 52)
    assert candidate.schedule.dx.operand_schedule("b").finalizer_blocks == 52
    assert candidate.schedule.dx.operand_schedule("b").producer_blocks == initial.schedule.dx.producer_blocks
    assert candidate.schedule.dx.operand_schedule("a") == initial.schedule.dx.operand_schedule("a")
    assert candidate.schedule.dw == initial.schedule.dw
    restored = adapter.update_fn(candidate, axis, None)
    assert restored == initial


def test_identity_deduplicates_inactive_controls_but_not_real_changes():
    from rtx.autotune.posthoc import PosthocOperandGeometry
    a = make_ms_eden_design_adapter((4096, 1536, 512), lambda _: {})
    c = a.initial_config
    alias = replace(c, schedule=replace(c.schedule, dw=replace(c.schedule.dw,
        producer_row_tile_batch=2, producer_row_load_bits=64, producer_values_per_lane=16,
        gemm_reduction_threads=512,
        a_geometry=PosthocOperandGeometry(producer_blocks=c.schedule.dw.producer_blocks))))
    assert a.config_id(alias) == a.config_id(c)
    assert a.serialize(alias) != a.serialize(c)  # Exact execution manifest remains intact.
    assert "dw.producer_row_tile_batch" not in a.axes
    assert "dw.producer_values_per_lane" not in a.axes
    assert "dw.producer_row_load_bits" not in a.axes
    active = replace(c, schedule=replace(c.schedule, dx=replace(c.schedule.dx, producer_row_tile_batch=2)))
    assert a.config_id(active) != a.config_id(c)
    assert a.config_id(replace(c, finalizer="dual")) != a.config_id(c)


@pytest.mark.parametrize("mode", ["tma_xor4_1", "tma_xor4_2", "tma_xor4_3", "tma_xor5_1", "tma_xor5_2", "tma_xor5_3"])
def test_tma_rejects_fallback_and_stage_override(mode):
    initial = MSEdenDesign(transpose_loader=mode)
    assert implementation_rejection(initial)
    schedule = replace(initial.schedule, **{
        p: replace(getattr(initial.schedule, p), producer_threads=128,
            producer_async_copy=True, producer_load="coalesced_x8",
            producer_transpose_values_per_lane=16, producer_copy_bits=32,
            producer_unroll_rows=False, producer_stages=int(mode[-1])) for p in ("dx", "dw")})
    assert implementation_rejection(replace(initial, schedule=schedule)) is None
    bad = replace(schedule, dw=replace(schedule.dw, producer_stages=1 if mode.endswith("3") else 3))
    assert implementation_rejection(replace(initial, schedule=bad))


def test_manifest_required_before_scoring():
    adapter = make_ms_eden_design_adapter((512, 512, 512), lambda _: {})
    result = adapter.evaluate(adapter.initial_config)
    assert result.status == "correctness_error"
    assert "manifest" in result.error


def test_complete_evidence_keeps_order_gates():
    import copy
    from test_ms_eden_pipeline_autotune import evidence
    def evaluate(config):
        result = evidence(config.schedule)
        result["design"] = asdict(config)
        result["baseline_policy"] = "stable_mxfp8"
        baseline = result["build_guards"]["baseline"]
        baseline.update(policy="stable_mxfp8", dx_policy="mxfp8", dw_policy="mxfp8")
        baseline["guards"]["fresh_rng_per_phase"] = [False, False]
        for metrics in result["gradient_checks"]["baseline"]["backward"].values():
            metrics.update(nrms=.04, gate=.06)
        quality = result["timings"]["backward"]["quality"]
        result["timings"]["backward"]["order_quality"] = {
            order: copy.deepcopy(quality) for order in ("AB", "BA")}
        return result
    adapter = make_ms_eden_design_adapter((512, 512, 512), evaluate)
    assert adapter.evaluate(adapter.initial_config).status == "ok"
    def missing_order(config):
        result = evaluate(config)
        del result["timings"]["backward"]["order_quality"]
        return result
    strict = make_ms_eden_design_adapter((512, 512, 512), missing_order)
    assert strict.evaluate(strict.initial_config).status != "ok"


def test_implementation_seeds_cover_coupled_loaders():
    from benchmarks.tune_ms_eden_design import implementation_seeds
    seeds = implementation_seeds(MSEdenDesign().schedule)
    assert len(seeds) == len(set(seeds)) == 28
    assert all(implementation_rejection(s) is None for s in seeds)
    assert len({s.transpose_loader for s in seeds[:7]}) == 7
    assert len({s.finalizer for s in seeds[:7]}) == 4


@pytest.mark.parametrize("kind,inline,scope,paired", [
    ("dual", False, "warp", False), ("dual_paired", False, "warp", True),
    ("dual_cta", True, "cta", False)])
def test_dispatch_arguments_and_cleanup(monkeypatch, kind, inline, scope, paired):
    from contextlib import contextmanager
    from benchmarks import experiment_ms_eden_dual_finalizer as module
    from benchmarks.ms_eden_design_dispatch import dispatch
    from benchmarks.tune_ms_eden_design import implementation_seeds
    calls = []
    @contextmanager
    def patched(mode, **kwargs):
        calls.append((mode, kwargs))
        try:
            yield
        finally:
            calls.append("restored")
    monkeypatch.setattr(module, "patched", patched)
    design = next(s for s in implementation_seeds(MSEdenDesign().schedule)
                  if s.transpose_loader == "native" and s.finalizer == kind)
    with pytest.raises(RuntimeError):
        with dispatch(design):
            raise RuntimeError("worker failure")
    assert calls == [("both", dict(inline_max=inline, inline_scope=scope, pair_values=paired)), "restored"]


def test_invalid_dispatch_never_enters_patches():
    from benchmarks.ms_eden_design_dispatch import dispatch
    with pytest.raises(ValueError, match="unknown"):
        with dispatch(MSEdenDesign(finalizer="typo")):
            pytest.fail("invalid implementation dispatched")


def test_driver_records_real_orchestrator_trials(monkeypatch, tmp_path):
    import copy
    import json
    from pathlib import Path
    from types import SimpleNamespace
    from benchmarks import tune_ms_eden_design as driver
    from benchmarks import tune_ms_eden_pipeline as existing
    from test_ms_eden_pipeline_autotune import evidence
    monkeypatch.setattr(existing, "runtime_environment", lambda: ({}, {"source_sha256": {}}))
    def run(command, **kwargs):
        assert command[command.index("--warmup-ms") + 1] in ("5000.0", "20000.0")
        assert "--min-batch-ms" in command and "--replays" in command
        config = MSEdenDesign(**json.loads(command[command.index("--config-json") + 1]))
        result = evidence(config.schedule)
        result.update(design=asdict(config), baseline_policy="stable_mxfp8")
        result["design_dispatch"] = [{"transpose": True, "selected": config.transpose_loader}]
        baseline = result["build_guards"]["baseline"]
        baseline.update(policy="stable_mxfp8", dx_policy="mxfp8", dw_policy="mxfp8")
        baseline["guards"]["fresh_rng_per_phase"] = [False, False]
        for metric in result["gradient_checks"]["baseline"]["backward"].values():
            metric.update(nrms=.04, gate=.06)
        timing = result["timings"]["backward"]
        timing["order_quality"] = {order: copy.deepcopy(timing["quality"]) for order in ("AB", "BA")}
        Path(command[command.index("--output") + 1]).write_text(json.dumps(result))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(driver.subprocess, "run", run)
    selection = tmp_path / "initial.json"
    selection.write_text(json.dumps({"candidate": asdict(MSEdenDesign().schedule)}))
    output = tmp_path / "study"
    driver.main(["--selection", str(selection), "--output", str(output),
                 "--shape", "512,512,512", "--trials", "32", "--seconds", "30"])
    records = [json.loads(s) for s in (output / "residuals" / "observations.jsonl").read_text().splitlines()]
    assert len(records) == 32
    assert all(r["outcome"]["status"] == "ok" for r in records)
    assert any(r["strategy"] == "implementation_coverage" for r in records)
    assert {"random", "coordinate_local"} <= {r["strategy"] for r in records}
    assert all("transpose_loader" in r["config"] for r in records)
    assert json.loads((output / "selection.json").read_text())["automatic_promotion"] is False
    confirmed = tmp_path / "confirmation"
    driver.main(["--selection", str(output / "selection.json"), "--output", str(confirmed),
                 "--shape", "512,512,512", "--confirm-only", "--rounds", "24",
                 "--warmup-ms", "20000", "--min-batch-ms", "5"])
    before = json.loads((output / "selection.json").read_text())
    after = json.loads((confirmed / "selection.json").read_text())
    assert before["candidate"] == after["candidate"]
    rows = (confirmed / "residuals" / "observations.jsonl").read_text().splitlines()
    assert len(rows) == 1 and after["confirmation_only"]
    assert after["warmup_ms"] == 20000. and after["min_batch_ms"] == 5.
    assert json.loads(rows[0])["context_id"] != records[0]["context_id"]
    layouts = tmp_path / "layouts"
    driver.main(["--selection", str(output / "selection.json"), "--output", str(layouts),
                 "--shape", "512,512,512", "--allow-row-major", "--coverage", "layout",
                 "--trials", "4", "--seconds", "30"])
    layout_rows = [json.loads(s) for s in (layouts / "residuals" / "observations.jsonl").read_text().splitlines()]
    assert len(layout_rows) == 4
    assert any(r["config"]["schedule"]["dx"]["gemm"]["scale_layout"] == "row_major" for r in layout_rows)
    assert all(r["outcome"]["status"] == "ok" for r in layout_rows)


@pytest.mark.parametrize("option,value", [
    ("--warmup-ms", "4999"), ("--min-batch-ms", "1"), ("--warmup-ms", "nan"),
    ("--min-batch-ms", "inf"), ("--replays", "15")])
def test_measurement_options_cannot_weaken_existing_budget(tmp_path, option, value):
    from benchmarks.tune_ms_eden_design import main
    with pytest.raises(SystemExit):
        main(["--output", str(tmp_path / "unused"), option, value])


def test_tma_audit_requires_actual_dispatch():
    from benchmarks.tune_ms_eden_design import implementation_seeds
    design = next(s for s in implementation_seeds(MSEdenDesign().schedule)
                  if s.transpose_loader == "tma_xor5_1" and s.finalizer == "separate")
    adapter = make_ms_eden_design_adapter((512, 512, 512),
        lambda c: {"design": asdict(c), "baseline_policy": "stable_mxfp8"}, initial=design)
    assert adapter.rejection(design) is None
    result = adapter.evaluate(design)
    assert result.status == "correctness_error" and "dispatch" in result.error


def test_tma_dispatch_records_and_rejects_silent_fallback(monkeypatch):
    import functools
    from rtx.kernels import nvfp4_posthoc as module
    from rtx.kernels import ms_eden_tma_preparation as tma
    from benchmarks.experiment_ms_eden_bank_layout import patched
    original = module.compile_hadamard_preparation
    @functools.wraps(original)
    def native(*args, **kwargs):
        return "native"
    monkeypatch.setattr(module, "compile_hadamard_preparation", native)
    calls = []
    def compile_tma(**kwargs):
        calls.append(kwargs)
        return "tma"
    monkeypatch.setattr(tma, "compile_tma_preparation", compile_tma)
    records = []
    with patched("tma_xor5_2", records=records, strict=True):
        assert module.compile_hadamard_preparation(128, 128) == "native"
        with pytest.raises(ValueError, match="silently fall back"):
            module.compile_hadamard_preparation(128, 128, transpose=True)
        assert module.compile_hadamard_preparation(128, 128,
            module.PosthocScaleConfig(128, 128), transpose=True, async_copy=True,
            load_mode="coalesced_x8", producer_transpose_values_per_lane=16) == "tma"
    assert module.compile_hadamard_preparation is native
    assert len(calls) == 1 and calls[0]["stages"] == 2
    assert [r["selected"] for r in records] == ["native", "tma_xor5_2"]
