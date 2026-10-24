import pytest

from rtx.autotune.preparation import (
    PreparationGeometry, linear_phase_geometry, preparation_portfolio,
)


def test_logical_reuse_and_staging():
    g = PreparationGeometry(1024, 256, 128, 128, 128)
    fused = g.features("fused")
    staged = g.features("staged")
    assert fused["a_cta_reuse"] == 2
    assert fused["b_cta_reuse"] == 8
    assert fused["prepared_elements"] == 524288
    assert staged["prepared_elements"] == 163840
    assert staged["logical_workspace_bytes"] == 1280 * 132
    assert fused["logical_workspace_bytes"] == 0


def test_hybrid_and_ragged_storage():
    g = PreparationGeometry(3, 5, 33, 2, 4, packed_bits=4, scale_group=16)
    assert g.features("stage_a")["logical_workspace_bytes"] == 3 * 20
    assert g.features("stage_b")["logical_workspace_bytes"] == 5 * 20
    assert g.features("staged")["logical_workspace_bytes"] == 8 * 20
    assert g.features("staged")["l2_known"] == 0


def test_phase_mapping():
    assert linear_phase_geometry(1000, 64, 128, "dx") == (1000, 128, 64)
    assert linear_phase_geometry(1000, 64, 128, "dw") == (64, 128, 1000)
    with pytest.raises(ValueError):
        linear_phase_geometry(1, 2, 3, "bad")


def test_execution_plan_uses_actual_backends_not_seed_names():
    from rtx.autotune.preparation import PreparationCandidate
    from rtx.autotune.preparation_mxfp8 import backward_plan_config, execution_plan
    candidate = PreparationCandidate("staged", backward_plan_config("fused_dx"))
    assert execution_plan(candidate) == ("fused", "decomposed")
    assert execution_plan(candidate, "dx") == ("fused",)
    assert execution_plan(candidate, "dw") == ("decomposed",)


def test_execution_signature_ignores_only_inactive_backend_fields():
    from dataclasses import replace
    from rtx.autotune.preparation import PreparationCandidate
    from rtx.autotune.preparation_mxfp8 import backward_plan_config, execution_signature
    config = backward_plan_config("staged")
    c = lambda cfg: PreparationCandidate("staged",cfg)
    inactive = replace(config,dx=replace(config.dx,fused=replace(config.dx.fused,tile_n=64)))
    active = replace(config,dx=replace(config.dx,gemm=replace(config.dx.gemm,tile_n=64)))
    assert execution_signature(c(config)) == execution_signature(c(inactive))
    assert execution_signature(c(config)) != execution_signature(c(active))
    assert execution_signature(c(config),"dw") == execution_signature(c(active),"dw")
    fused = backward_plan_config("fused")
    changed = replace(fused,dx=replace(fused.dx,gemm=replace(fused.dx.gemm,tile_n=64)))
    assert execution_signature(c(fused)) == execution_signature(c(changed))
    assert execution_signature(c(fused)) != execution_signature(c(config))


def test_schedule_features_exclude_inactive_fields_and_measurements():
    from benchmarks.analyze_preparation_prior import portable_features
    row={"features":{"config.config.dx.backend=decomposed":1.,
        "config.config.dx.fused.tile_n":64.,"config.config.dx.gemm.tile_n":128.,
        "config.config.dx.gemm.compiled_registers":99.,"outcome.median_ms":1.}}
    features=portable_features(row,"schedule")
    assert features["config.config.dx.gemm.tile_n"] == 128.
    assert "config.config.dx.fused.tile_n" not in features
    assert not any("compiled" in k or "outcome" in k for k in features)


def test_study_consolidation_uses_median_and_keeps_contexts_separate():
    from dataclasses import replace
    from rtx.autotune.preparation_mxfp8 import backward_plan_config
    from rtx.bwd_autotune import bwd_config_to_dict
    from benchmarks.analyze_preparation_prior import consolidate_schedules
    base = backward_plan_config("staged")
    inactive = replace(base,dx=replace(base.dx,fused=replace(base.dx.fused,tile_n=64)))
    def row(context,config,latency,sequence):
        return {"context_id":context,"sequence":sequence,
            "config":{"implementation":"staged","config":bwd_config_to_dict(config)},
            "outcome":{"median_ms":latency}}
    rows=[row("device_a",base,1.,0),row("device_a",inactive,3.,1),row("device_b",base,9.,0)]
    grouped=consolidate_schedules(rows)
    assert len(grouped)==2
    assert grouped[0]["outcome"]["median_ms"]==2.
    assert grouped[0]["repeat_count"]==2
    assert grouped[1]["outcome"]["median_ms"]==9.
    assert rows[0]["outcome"]["median_ms"]==1.


def test_reuse_prior_ranks_work_without_removing_plans():
    from rtx.autotune.preparation import PreparationCandidate
    from rtx.autotune.preparation_mxfp8 import backward_plan_config, reuse_prior, execution_plan
    candidates = [PreparationCandidate(name, backward_plan_config(name))
                  for name in ("fused", "fused_dx", "fused_dw", "staged")]
    score = lambda c: reuse_prior(c, 32768, 768, 1536, phase="backward",
                                  sm_count=170, l2_bytes=96*1024**2)
    ordered = preparation_portfolio(candidates, plan_of=execution_plan, score=score, limit=4)
    assert ordered[0].implementation == "staged"
    assert len(ordered) == len(candidates)
    # Partial-gradient ranking does not charge the unused gradient.
    assert reuse_prior(candidates[1], 32768,768,1536,phase="dw") == reuse_prior(candidates[3],32768,768,1536,phase="dw")
    assert all(value >= 0 for c in candidates for value in score(c))


def test_reuse_prior_device_capacity_changes_tie_breakers_only():
    from rtx.autotune.preparation import PreparationCandidate
    from rtx.autotune.preparation_mxfp8 import backward_plan_config, reuse_prior
    c = PreparationCandidate("staged", backward_plan_config("staged"))
    small = reuse_prior(c,128,256,256,phase="backward",sm_count=40,l2_bytes=1024**2)
    large = reuse_prior(c,128,256,256,phase="backward",sm_count=170,l2_bytes=96*1024**2)
    assert small[0] == large[0]
    assert small[1] < large[1]
    assert small[2] > large[2]


def test_expansion_filter_does_not_consume_useful_quota():
    from tests.test_autotune_coverage import adapter, Config
    from rtx.autotune.preparation import expand_preparation_candidates
    a = adapter()
    result = expand_preparation_candidates(a, [Config()], per_seed=1,
        accept=lambda c: c.a == 0)
    assert len(result) == 2
    assert all(c.a == 0 for c in result)
    assert result[1].b != 0


def test_budget_preserves_plan_breadth():
    candidates = [("fused", 1), ("fused", 2), ("staged", 3), ("stage_a", 4)]
    assert preparation_portfolio(candidates, plan_of=lambda x: x[0],
                                 score=lambda x: x[1], limit=3) == [
        ("fused", 1), ("staged", 3), ("stage_a", 4)]


@pytest.mark.parametrize("kwargs", [{"m": 0}, {"packed_bits": 3}, {"l2_bytes": -1}])
def test_invalid_geometry(kwargs):
    args = dict(m=128, n=128, k=128, tile_m=128, tile_n=128)
    args.update(kwargs)
    with pytest.raises(ValueError):
        PreparationGeometry(**args)


def test_composed_selection_cache_and_policy_isolation(tmp_path):
    from dataclasses import replace
    from tests.test_autotune_coverage import adapter, Config
    from rtx.autotune.preparation import (
        make_preparation_adapter, PreparationCandidate, tune_preparation,
    )
    from rtx.autotune.first_hit import FirstHitPolicy
    from rtx.autotune.store import JsonlTuningStore
    children = {"fused": adapter(), "staged": adapter()}
    policy = {"rounding": "infinity", "precision": "mxfp8"}
    composite = make_preparation_adapter(children, numerical_policy=policy,
        policy_of=lambda name, config: policy if config.b == 0 else {"rounding": "floor"})
    candidates = [PreparationCandidate("fused", Config()),
                  PreparationCandidate("staged", Config(a=2))]
    store = JsonlTuningStore(tmp_path)
    result = tune_preparation(composite, store, candidates, score=lambda c: 0,
        policy=FirstHitPolicy(max_trials=2, confirmation_repeats=0), progress=None)
    assert result.config == candidates[1]
    composite.evaluator = lambda c: pytest.fail("cache should not evaluate")
    cached = tune_preparation(composite, store, candidates, score=lambda c: 0, progress=None)
    assert cached.evaluated_trials == 0
    assert cached.config == result.config
    assert composite.rejection(PreparationCandidate("fused", Config(b=1)))[0] == "policy_rejected"
    children["staged"].context = replace(children["staged"].context, kernel_revision=2)
    revised = make_preparation_adapter(children, numerical_policy=policy,
                                      policy_of=lambda name, config: policy)
    assert revised.context.identifier != composite.context.identifier
    assert not list(store.records(revised.context))


def test_composed_adapter_rejects_different_workloads():
    from dataclasses import replace
    from tests.test_autotune_coverage import adapter
    from rtx.autotune.preparation import make_preparation_adapter
    a, b = adapter(), adapter()
    b.context = replace(b.context, workload={"m": 1024})
    with pytest.raises(ValueError, match="workload"):
        make_preparation_adapter({"a": a, "b": b}, numerical_policy={"p": 1},
                                 policy_of=lambda n, c: {"p": 1})


def test_backward_plans_have_identical_numerical_policy():
    from dataclasses import replace
    from benchmarks.study_backward_preparation import build_config
    from rtx.autotune.preparation_mxfp8 import numerical_policy as policy_of
    from rtx.kernels.mxfp8 import MXFP8Problem
    reference = policy_of("staged", build_config("staged"))
    for plan in ("staged", "fused_dx", "fused_dw", "fused"):
        config = build_config(plan)
        assert policy_of(plan, config) == reference
        assert config.implementation_rejection(MXFP8Problem(2048, 768, 1536)) is None
    config = build_config("fused")
    changed = replace(config, dw=replace(config.dw,
        fused=replace(config.dw.fused, scale_rounding="floor")))
    assert policy_of("fused", changed) != reference


def test_refinement_uses_cumulative_budget_and_preserves_incumbent(tmp_path):
    from tests.test_autotune_coverage import adapter, Config
    from rtx.autotune.first_hit import FirstHitPolicy, tune_first_hit
    from rtx.autotune.store import JsonlTuningStore
    a = adapter()
    store = JsonlTuningStore(tmp_path)
    pool = [Config(), Config(a=1), Config(a=2), Config(a=2,b=1)]
    def run(trials):
        return tune_first_hit(a, store, pool, refine=True, progress=None,
            policy=FirstHitPolicy(max_trials=trials, confirmation_repeats=0))
    first = run(2)
    assert len(list(store.records(a.context))) == 2
    second = run(4)
    assert second.evaluated_trials == 2
    assert len(list(store.records(a.context))) == 4
    assert second.median_ms <= first.median_ms
    a.evaluator = lambda c: pytest.fail("exhausted cumulative budget evaluated")
    third = run(4)
    assert third.evaluated_trials == 0
    assert third.config == second.config


def test_component_schedule_axes_remain_searchable():
    from tests.test_autotune_coverage import adapter, Config
    from rtx.autotune.preparation import make_preparation_adapter, PreparationCandidate, expand_preparation_candidates
    composite = make_preparation_adapter({"fused": adapter(), "staged": adapter()},
        numerical_policy={"p": 1}, policy_of=lambda name, config: {"p": 1})
    seed = PreparationCandidate("fused", Config())
    assert composite.update_fn(seed, "fused:a.tile", 2).config.a == 2
    assert composite.update_fn(seed, "staged:a.tile", 2) == seed
    expanded = expand_preparation_candidates(composite, [seed], per_seed=8)
    assert any(c.config.a == 2 for c in expanded)
    assert all(c.config.a != 3 for c in expanded)
    assert len({composite.config_id(c) for c in expanded}) == len(expanded)


def test_shape_transfer_preserves_exact_cache_identity_and_semantics():
    from dataclasses import replace
    from tests.test_autotune_coverage import adapter, Config
    from rtx.autotune.preparation import make_preparation_adapter, PreparationCandidate
    from rtx.autotune.first_hit import contract, retrieve_candidates, tune_first_hit, FirstHitPolicy
    from rtx.autotune.store import InMemoryTuningStore
    def make(m, rounding="infinity"):
        child = adapter()
        child.context = replace(child.context, workload={"m": m, "n": 128, "k": 128})
        policy = {"rounding": rounding}
        return make_preparation_adapter({"staged": child}, numerical_policy=policy,
                                        policy_of=lambda name, config: policy)
    source, target = make(128), make(256)
    assert contract(source.context) == contract(target.context)
    assert source.context.identifier != target.context.identifier
    store = InMemoryTuningStore()
    winner = PreparationCandidate("staged", Config(a=2))
    tune_first_hit(source, store, [winner],
                   policy=FirstHitPolicy(max_trials=2,confirmation_repeats=0),progress=None)
    assert retrieve_candidates(target, store.records()) == [winner]
    assert not list(store.records(target.context))
    assert not retrieve_candidates(make(256,"floor"),store.records())


def test_preparation_prior_feature_allowlist():
    from benchmarks.analyze_preparation_prior import portable_features
    row = {"features": {"context.device.multiprocessor_count": 100,
        "derived.dx_preparation_prepared_elements": 1000,
        "measured_latency": 999, "context.tags.source_path": 1,
        "derived.compiled_registers": 255}, "outcome": {"median_ms": 123}}
    a = portable_features(row)
    row["outcome"]["median_ms"] = .001
    row["features"]["measured_latency"] = 0
    assert portable_features(row) == a
    assert all("measured" not in key and "compiled" not in key for key in a)


def test_relative_cost_target_normalizes_each_training_context():
    from benchmarks.analyze_preparation_prior import training_scores
    rows = [{"context_id": context, "outcome": {"median_ms": value}}
            for context,value in (("small",1.),("small",2.),("large",100.),("large",150.))]
    assert training_scores(rows,"relative") == [1.,2.,1.,1.5]
    assert training_scores(rows,"absolute") == [1.,2.,100.,150.]
    with pytest.raises(ValueError):
        training_scores(rows,"invalid")


def test_research_refinement_keeps_baseline_plans_and_is_reproducible():
    from tests.test_autotune_coverage import adapter, Config
    from rtx.autotune.preparation import make_preparation_adapter, PreparationCandidate
    from benchmarks.tune_operand_preparation import research_candidates
    composite = make_preparation_adapter({"fused":adapter(),"staged":adapter()},
        numerical_policy={"p":1},policy_of=lambda n,c:{"p":1})
    seeds = [PreparationCandidate(name,Config()) for name in ("fused","staged")]
    def generate(seed):
        return research_candidates(composite,seeds,per_seed=8,portfolio_seed=seed,
                                   refinement_plan="staged")
    result = generate(42)
    assert result[:2] == seeds
    assert all(c.implementation == "staged" for c in result[2:])
    assert result == generate(42)
    assert {composite.config_id(c) for c in result} == {composite.config_id(c) for c in generate(99)}
    with pytest.raises(ValueError,match="no registered seed"):
        research_candidates(composite,seeds,refinement_plan="missing")


def test_candidate_expansion_rotates_coordinates():
    from tests.test_autotune_coverage import adapter, Config
    from rtx.autotune.preparation import expand_preparation_candidates
    a = adapter()
    a.axes = {"a.tile": (1,2), "b.stages": (1,2)}
    expanded = expand_preparation_candidates(a, [Config()], per_seed=2)
    assert expanded == [Config(), Config(a=1), Config(b=1)]
    assert expand_preparation_candidates(a, [Config()], per_seed=2, max_examined=1) == [Config(), Config(a=1)]


def test_candidate_expansion_can_target_late_coordinate_without_mutation():
    from tests.test_autotune_coverage import adapter, Config
    from rtx.autotune.preparation import expand_preparation_candidates
    a = adapter()
    a.axes = {"a.tile": (1,2), "b.stages": (1,2)}
    assert expand_preparation_candidates(a, [Config()], per_seed=2,
        coordinates=["b.stages"]) == [Config(), Config(b=1), Config(b=2)]
    assert list(a.axes) == ["a.tile", "b.stages"]
    with pytest.raises(KeyError):
        expand_preparation_candidates(a, [Config()], coordinates=["missing"])


def test_research_axis_filter_preserves_baseline_and_rejects_typos():
    from tests.test_autotune_coverage import adapter, Config
    from rtx.autotune.preparation import make_preparation_adapter, PreparationCandidate
    from benchmarks.tune_operand_preparation import research_candidates
    component = adapter()
    component.axes = {"a.tile": (1,2), "b.stages": (1,2)}
    composite = make_preparation_adapter({"staged": component},
        numerical_policy={"p":1}, policy_of=lambda n,c:{"p":1})
    seed = PreparationCandidate("staged", Config())
    result = research_candidates(composite, [seed], axis_filters=["b.stages"])
    assert result == [seed, PreparationCandidate("staged",Config(b=1)),
                      PreparationCandidate("staged",Config(b=2))]
    with pytest.raises(ValueError, match="matched no"):
        research_candidates(composite, [seed], axis_filters=["misspelled"])


def test_graph_warmup_is_time_based_and_bounded():
    import pytest
    from rtx.autotune.preparation_runtime import _graph_warmup_replays
    assert _graph_warmup_replays(2.) == 100
    assert _graph_warmup_replays(.1) == 2000
    assert _graph_warmup_replays(.001) == 4096
    assert _graph_warmup_replays(100.) == 10
    for invalid in (0., -1., float("nan"), float("inf")):
        with pytest.raises(ValueError):
            _graph_warmup_replays(invalid)


def test_preparation_profile_keeps_only_positive_static_resources(monkeypatch):
    from types import SimpleNamespace
    from rtx.autotune import preparation_runtime as runtime
    from rtx.autotune import hardware
    monkeypatch.setattr(runtime.DeviceFingerprint,"current",lambda device: SimpleNamespace(
        as_dict=lambda: {"multiprocessor_count":170,"uuid":"test"}))
    monkeypatch.setattr(hardware,"device_properties",lambda device: {
        "l2_cache_size":96*1024**2,"memory_bus_width_bits":512,
        "regs_per_multiprocessor":0,"clock_rate_khz":999999,
        "max_blocks_per_multiprocessor":None})
    assert runtime._preparation_device_profile("cuda:1") == {
        "multiprocessor_count":170,"uuid":"test","l2_cache_size":96*1024**2,
        "memory_bus_width_bits":512}


def test_explicit_preparation_policy_and_legacy_opt_in(monkeypatch):
    from rtx.autotune import AutotunePolicy
    from rtx.autotune.preparation import preparation_selection_enabled
    monkeypatch.delenv("RTX_PREPARATION_SELECTION", raising=False)
    assert not preparation_selection_enabled(None)
    assert not preparation_selection_enabled(AutotunePolicy())
    policy = AutotunePolicy(preparation_selection=True, backward_trials=8)
    assert preparation_selection_enabled(policy)
    assert preparation_selection_enabled(policy.for_pass(backward=True))
    for invalid in (1, "true", None):
        with pytest.raises(ValueError, match="preparation_selection"):
            AutotunePolicy(preparation_selection=invalid)
    monkeypatch.setenv("RTX_PREPARATION_SELECTION", "1")
    assert preparation_selection_enabled(AutotunePolicy())


def test_fused_seed_prior_keeps_policy_and_scalar_tail_fallback():
    from rtx.autotune.preparation_mxfp8 import fused_forward_preparation_seeds, numerical_policy
    from rtx.kernels.mxfp8 import DEFAULT_MXFP8_FWD_CONFIG, MXFP8Problem
    seeds = fused_forward_preparation_seeds()
    assert len(seeds) == 4
    assert seeds[-1] == DEFAULT_MXFP8_FWD_CONFIG
    policies = [numerical_policy("fused", c) for c in seeds]
    assert all(p == policies[0] for p in policies)
    for dims in ((128,256,256),(2048,768,1536),(8192,3072,768),(32768,768,1536)):
        assert seeds[0].implementation_rejection(MXFP8Problem(*dims)) is None
    ragged = MXFP8Problem(129,96,160)
    assert seeds[-1].implementation_rejection(ragged) is None
