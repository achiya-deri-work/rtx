from dataclasses import FrozenInstanceError, replace
import copy
import json
import pytest

from rtx.autotune.ms_eden_pipeline import MSEdenPipelineSchedule, make_ms_eden_pipeline_adapter, pipeline_outcome


def evidence(config=None):
    from dataclasses import asdict
    config = config or MSEdenPipelineSchedule()
    guards = dict(forward_bitwise_matched=True, current_input_replay=True,
                  current_upstream_replay=True, current_weight_replay=True, fresh_rng_per_phase=[True, True])
    memory = dict(ordinary_compiled_allocation_audit=dict(live_growth_bytes=0), replay_live_growth_bytes=0)
    gradients = {phase: dict(finite=True, nrms=.14, gate=.25, max_abs=1.) for phase in ("dx", "dw")}
    return dict(shape=[512, 512, 512], candidate_pipeline=config.backward_pipeline,
        schedules=dict(candidate={phase: json.dumps(asdict(getattr(config, phase))) for phase in ("dx", "dw")}),
        build_guards={name: dict(guards=copy.deepcopy(guards), memory=copy.deepcopy(memory))
                              for name in ("baseline", "candidate")},
        gradient_checks={name: dict(backward=copy.deepcopy(gradients)) for name in ("baseline", "candidate")},
        memory=dict(replay_live_growth_bytes=0),
        samples=[dict(scope="backward", candidate_ms=.1) for _ in range(4)],
        timings=dict(backward=dict(quality={name:dict(sampling=dict(collection=dict(stationary=True)))
            for name in ("baseline", "candidate")}, paired={})))


def test_pipeline_immutable_serialization_and_component_updates():
    adapter = make_ms_eden_pipeline_adapter((512, 512, 512), lambda _: evidence())
    initial = adapter.initial_config
    assert initial.backward_pipeline == "serial"
    assert initial.dx.producer_signs == initial.dw.producer_signs == "register"
    assert initial.dx.producer_order == "row_major" and initial.dw.producer_order == "reduction_major"
    with pytest.raises(FrozenInstanceError):
        initial.backward_pipeline = "paired"
    candidate = adapter.update_fn(initial, "dw.amax_reduction", "staged")
    assert candidate.dx is initial.dx and candidate.dw.amax_reduction == "staged"
    paired = adapter.update_fn(candidate, "backward_pipeline", "paired")
    assert paired.dx == candidate.dx and paired.dw == candidate.dw
    assert adapter.deserialize(adapter.serialize(paired)) == paired
    assert len({adapter.config_id(c) for c in (initial, candidate, paired)}) == 3
    assert adapter.rejection(initial) is None
    assert "derived.dx.amax_logical_load_elements" in adapter.features(initial)


def test_policy_rejection_and_component_axes_are_composed():
    adapter = make_ms_eden_pipeline_adapter((512, 512, 512), lambda _: evidence())
    assert adapter.axes["backward_pipeline"] == ("serial", "paired", "operand_paired")
    assert "dx.producer_stages" in adapter.axes and "dw.gemm:gemm_geometry" in adapter.axes
    assert "dx.rng_mode" not in adapter.axes
    initial = adapter.initial_config
    assert adapter.rejection(replace(initial, dx=replace(initial.dx, rng_mode="philox")))[0] == "policy_rejected"
    invalid = replace(initial, dw=replace(initial.dw, producer_stages=2, producer_async_copy=False))
    assert adapter.rejection(invalid)[0] == "implementation_rejected"
    # Component identity canonicalizes irrelevant fixed-explicit packet width.
    assert adapter.config_id(initial) == adapter.config_id(replace(initial, dx=replace(initial.dx, philox_packet_size=1)))


def test_context_separates_device_environment_measurement_and_shape():
    def build(**kw):
        return make_ms_eden_pipeline_adapter((512, 512, 512), lambda _: evidence(), **kw)
    contexts = [build(), build(device={"name": "other"}), build(environment={"torch": "other"}),
                build(measurement_contract="other")]
    assert len({c.context.identifier for c in contexts}) == 4
    assert contexts[0].context.tags["automatic_promotion"] is False
    with pytest.raises(ValueError):
        build(measurement_contract="")


def test_evaluator_scores_only_guarded_complete_backward_and_preserves_quality():
    result = evidence()
    outcome = pipeline_outcome(result)
    assert outcome.successful and outcome.timing_eligible
    assert outcome.median_ms == .1
    assert outcome.metadata["validated"] and outcome.metadata["automatic_promotion"] is False
    result["timings"]["backward"]["quality"]["candidate"]["sampling"]["collection"]["stationary"] = False
    assert pipeline_outcome(result).successful and not pipeline_outcome(result).timing_eligible
    result["timings"]["backward"]["quality"]["candidate"]["sampling"]["collection"]["stationary"] = True
    result["timings"]["backward"]["quality"]["baseline"]["sampling"]["collection"]["stationary"] = False
    assert not pipeline_outcome(result).timing_eligible
    adapter = make_ms_eden_pipeline_adapter((512, 512, 512), lambda _: result)
    assert adapter.evaluate(adapter.initial_config).successful


def test_deterministic_mxfp8_reference_does_not_require_fp4_rng():
    result=evidence()
    result["baseline_policy"]="stable_mxfp8"
    reference=result["build_guards"]["baseline"]
    reference.update(policy="stable_mxfp8",dx_policy="mxfp8",dw_policy="mxfp8")
    reference["guards"]["fresh_rng_per_phase"]=[False,False]
    for phase in ("dx","dw"):
        result["gradient_checks"]["baseline"]["backward"][phase].update(nrms=.038,gate=.06)
    assert pipeline_outcome(result).successful
    bad=copy.deepcopy(result)
    bad["build_guards"]["candidate"]["guards"]["fresh_rng_per_phase"]=[False,False]
    assert not pipeline_outcome(bad).successful
    bad=copy.deepcopy(result)
    bad["build_guards"]["baseline"]["dw_policy"]="ms_eden"
    assert not pipeline_outcome(bad).successful
    bad=copy.deepcopy(result)
    bad["gradient_checks"]["baseline"]["backward"]["dx"]["gate"]=.25
    assert not pipeline_outcome(bad).successful


def test_per_order_quality_cannot_be_hidden_by_stationary_aggregates():
    result=evidence()
    assert pipeline_outcome(result).timing_eligible  # Legacy evidence contract.
    assert not pipeline_outcome(result,require_order_quality=True).successful
    timing=result['timings']['backward']
    timing['order_quality']={order:copy.deepcopy(timing['quality']) for order in ('AB','BA')}
    assert pipeline_outcome(result,require_order_quality=True).timing_eligible
    timing['order_quality']['BA']['candidate']['sampling']['collection']['stationary']=False
    outcome=pipeline_outcome(result,require_order_quality=True)
    assert outcome.successful and not outcome.timing_eligible
    assert not pipeline_outcome(result).timing_eligible


@pytest.mark.parametrize("failure", ["missing", "input", "rng", "ordinary_memory", "graph_memory", "gradient", "finiteness", "rounds"])
def test_invalid_evidence_is_never_latency_success(failure):
    result = evidence()
    if failure == "missing":
        del result["gradient_checks"]
    elif failure == "input":
        result["build_guards"]["candidate"]["guards"]["current_input_replay"] = False
    elif failure == "rng":
        result["build_guards"]["candidate"]["guards"]["fresh_rng_per_phase"] = [True, False]
    elif failure == "ordinary_memory":
        result["build_guards"]["candidate"]["memory"]["ordinary_compiled_allocation_audit"]["live_growth_bytes"] = 1
    elif failure == "graph_memory":
        result["memory"]["replay_live_growth_bytes"] = 1
    elif failure in ("gradient", "finiteness"):
        result["gradient_checks"]["candidate"]["backward"]["dw"]["nrms"] = .4 if failure == "gradient" else float("nan")
    elif failure == "rounds":
        result["samples"].pop()
    outcome = pipeline_outcome(result)
    assert outcome.status == "correctness_error" and not outcome.successful


def test_adapter_rejects_stale_or_wrong_config_artifacts():
    result = evidence()
    adapter = make_ms_eden_pipeline_adapter((512, 512, 512), lambda _: result)
    paired = replace(adapter.initial_config, backward_pipeline="paired")
    assert adapter.evaluate(paired).status == "correctness_error"
    altered = replace(adapter.initial_config, dw=replace(adapter.initial_config.dw, amax_reduction="staged"))
    assert adapter.evaluate(altered).status == "correctness_error"
    result["shape"] = [128, 128, 128]
    assert adapter.evaluate(adapter.initial_config).status == "correctness_error"
