"""Composable, policy-fixed complete-backward MS-EDEN research tuning.

This adapter joins the existing per-leg schedule spaces; it neither implements
a search engine nor installs winners into runtime defaults. Evaluators must run
the public fullgraph complete backward, not sum independently timed phases.
"""
from dataclasses import asdict, dataclass, field, replace
import json
import math
import statistics

from .core import DiscreteKernelAdapter, KernelContext, stable_id
from .outcomes import TrialOutcome
from .posthoc import PosthocPreparationSchedule, make_posthoc_preparation_adapter


def _baseline(phase):
    return PosthocPreparationSchedule(producer_threads=128, producer_blocks=256,
        producer_load="coalesced_wide", producer_signs="register", rng_mode="explicit",
        producer_order="row_major" if phase == "dx" else "reduction_major",
        finalizer_threads=128, finalizer_blocks=224)


@dataclass(frozen=True)
class MSEdenPipelineSchedule:
    backward_pipeline: str = "serial"
    dx: PosthocPreparationSchedule = field(default_factory=lambda: _baseline("dx"))
    dw: PosthocPreparationSchedule = field(default_factory=lambda: _baseline("dw"))

    def __post_init__(self):
        if self.backward_pipeline not in ("serial", "paired", "operand_paired"):
            raise ValueError("backward_pipeline must be serial, paired, or operand_paired")
        for phase in ("dx", "dw"):
            value = getattr(self, phase)
            if isinstance(value, dict):
                object.__setattr__(self, phase, PosthocPreparationSchedule(**value))
            elif not isinstance(value, PosthocPreparationSchedule):
                raise ValueError("dx/dw must be immutable preparation schedules")


def pipeline_outcome(result, *, expected_config=None, expected_shape=None, require_order_quality=False):
    """Gate complete public A/B artifacts before assigning a candidate score.

    These are integration/stale-buffer and broad FP32 quantization-error gates.
    Independent identical-draw arithmetic qualification remains separate.
    """
    def failure(reason):
        return TrialOutcome("correctness_error", error=reason,
            metadata=dict(scope="complete_public_backward", validation_scope="integration_not_identical_draw"))
    try:
        baseline_policy=result.get("baseline_policy","fp4_fp4")
        if baseline_policy not in ("fp4_fp4","stable_mxfp8"):
            return failure("unknown baseline numerical policy")
        if expected_shape is not None and tuple(result["shape"]) != tuple(expected_shape):
            return failure("evaluator returned a different workload shape")
        if expected_config is not None:
            if result["candidate_pipeline"] != expected_config.backward_pipeline:
                return failure("evaluator returned a different pipeline")
            for phase in ("dx", "dw"):
                actual = PosthocPreparationSchedule(**json.loads(result["schedules"]["candidate"][phase]))
                if actual != getattr(expected_config, phase):
                    return failure(f"evaluator returned a different {phase} schedule")
        for name in ("baseline", "candidate"):
            record = result["build_guards"][name]
            guards = record["guards"]
            for key in ("forward_bitwise_matched", "current_input_replay",
                        "current_upstream_replay", "current_weight_replay"):
                if guards[key] is not True:
                    return failure(f"{name}: {key} missing or failed")
            deterministic_reference=name=="baseline" and baseline_policy=="stable_mxfp8"
            if deterministic_reference and (
                record.get("policy")!="stable_mxfp8" or
                record.get("dx_policy")!="mxfp8" or record.get("dw_policy")!="mxfp8"):
                return failure("baseline: inconsistent deterministic MXFP8 policy")
            expected_rng=[False,False] if deterministic_reference else [True,True]
            if guards["fresh_rng_per_phase"] != expected_rng:
                return failure(f"{name}: fresh RNG not established for both gradients")
            memory = record["memory"]
            if memory["ordinary_compiled_allocation_audit"]["live_growth_bytes"] > 0 or memory["replay_live_growth_bytes"] != 0:
                return failure(f"{name}: live allocation growth")
            for phase in ("dx", "dw"):
                metrics = result["gradient_checks"][name]["backward"][phase]
                if (metrics["finite"] is not True or not math.isfinite(metrics["nrms"])
                        or not math.isfinite(metrics["gate"]) or not 0 <= metrics["nrms"] <= metrics["gate"] <= (.06 if deterministic_reference else .25)
                        or not math.isfinite(metrics["max_abs"])):
                    return failure(f"{name}/{phase}: invalid current-buffer gradient")
        if result["memory"]["replay_live_growth_bytes"] != 0:
            return failure("paired measurement live allocation growth")
        samples = [float(row["candidate_ms"]) for row in result["samples"] if row["scope"] == "backward"]
        if len(samples) < 4 or any(not math.isfinite(v) or v <= 0 for v in samples):
            return failure("four finite positive complete-backward rounds required")
        qualities = result["timings"]["backward"]["quality"]
        stationary = []
        for name in ("baseline", "candidate"):
            flag = qualities[name]["sampling"]["collection"]["stationary"]
            if type(flag) is not bool:
                return failure("explicit timing quality assessment required for both paired legs")
            stationary.append(flag)
        order_quality=result['timings']['backward'].get('order_quality')
        if require_order_quality and order_quality is None:
            return failure('separate AB/BA timing quality evidence required')
        if order_quality is not None:
            for order in ('AB','BA'):
                for name in ('baseline','candidate'):
                    flag=order_quality[order][name]['sampling']['collection']['stationary']
                    if type(flag) is not bool:
                        return failure('explicit per-order timing quality assessment required')
                    stationary.append(flag)
        quality = dict(sampling=dict(actual_samples=len(samples), collection=dict(
            stationary=all(stationary), paired_quality=qualities, order_quality=order_quality)))
        return TrialOutcome("ok", median_ms=statistics.median(samples), timings_ms=samples,
            max_abs_error=max(result["gradient_checks"]["candidate"]["backward"][phase]["max_abs"] for phase in ("dx", "dw")),
            metadata=dict(scope="complete_public_backward", validated=True,
                validation_scope="integration_not_identical_draw", automatic_promotion=False,
                gradient_checks=result["gradient_checks"]["candidate"],
                memory=result["build_guards"]["candidate"]["memory"],
                paired=result["timings"]["backward"]["paired"],
                source_sha256=result.get("source_sha256", {}), **quality))
    except (KeyError, TypeError, ValueError) as error:
        return failure(f"incomplete public backward evidence: {error}")


def make_ms_eden_pipeline_adapter(shape, evaluator, *, device=None, environment=None,
        measurement_contract="public_fullgraph_paired_abba_complete_backward_v1", initial=None,
        allow_row_major=False):
    """Build a normal DiscreteKernelAdapter for existing search/ask-tell APIs.

    ``evaluator(config)`` returns a compare_ms_eden_pipeline result mapping.
    Component updates, legality checks, identities and resource features are
    delegated to the existing posthoc adapters. No Cartesian space is built.
    """
    if not isinstance(measurement_contract, str) or not measurement_contract.strip():
        raise ValueError("nonempty complete-backward measurement contract required")
    components = {phase: make_posthoc_preparation_adapter(shape, phase, "mma128", lambda _: None,
        device=device, environment=environment, randomness="public",
        measurement_contract=measurement_contract) for phase in ("dx", "dw")}
    if type(allow_row_major) is not bool:
        raise ValueError("allow_row_major must be bool")
    row_components = ({phase: make_posthoc_preparation_adapter(shape, phase, "row_major", lambda _: None,
        device=device, environment=environment, randomness="public",
        measurement_contract=measurement_contract) for phase in ("dx", "dw")} if allow_row_major else {})
    def component_for(phase, schedule):
        return (row_components[phase] if allow_row_major and schedule.gemm.scale_layout == "row_major"
                else components[phase])
    initial = initial or MSEdenPipelineSchedule()
    axes = {"backward_pipeline": ("serial", "paired", "operand_paired")}
    for phase, component in components.items():
        if allow_row_major:
            axes[f"{phase}.scale_layout"] = ("mma128", "row_major")
        for axis, values in component.axes.items():
            # This experiment holds draw generation/source semantics fixed.
            if axis in ("rng_mode", "reader_mode", "philox_packet_size", "decoder_threads", "decoder_blocks"):
                continue
            axes[f"{phase}.{axis}"] = values
        if allow_row_major:
            # Moving scale staging to the producer is a different ownership
            # design, not a register-cap tweak. Couple its scalar stores and
            # barrier policy so a packed/staged consumer candidate can reach it.
            axes[f"{phase}.gemm:scale_owner"] = tuple(
                dict(scale_role=owner, scale_smem_store="scalar", scale_recycle="barrier")
                for owner in ("consumers", "producer"))
    def update(config, axis, value):
        if axis == "backward_pipeline":
            return replace(config, backward_pipeline=value)
        phase, coordinate = axis.split(".", 1)
        schedule = getattr(config, phase)
        if coordinate == "scale_layout":
            if not allow_row_major or value not in ("mma128", "row_major"):
                raise ValueError("unsupported scale-layout transition")
            changes = dict(scale_layout=value, scale_role="tma" if value == "mma128" else "consumers")
            if value == "mma128":
                changes.update(tile_m=128, tile_n=128, tile_k=128, atom_layout_n=2)
            return replace(config, **{phase: replace(schedule, gemm=replace(schedule.gemm, **changes))})
        return replace(config, **{phase: component_for(phase, schedule).update_fn(schedule, coordinate, value)})
    def rejection(config):
        for phase, component in components.items():
            schedule = getattr(config, phase)
            if schedule.rng_mode != "explicit" or schedule.reader_mode != "direct":
                return "policy_rejected", "pipeline experiment fixes explicit RNG and original BF16 direct reader"
            reason = component_for(phase, schedule).rejection(schedule)
            if reason:
                return reason
        return None
    def features(config):
        result = {"pipeline_side_streams":3 if config.backward_pipeline=="operand_paired" else int(config.backward_pipeline=="paired")}
        for phase, component in components.items():
            schedule = getattr(config, phase)
            result.update({f"{phase}.{key}": value for key, value in component_for(phase, schedule).extra_features_fn(schedule).items()})
        return result
    return DiscreteKernelAdapter(
        context=KernelContext("ms_eden_pipeline_research", 3, dict(zip(("m", "n", "k"), shape)),
            device=device or {}, environment=environment or {}, tags=dict(
                source="original_bf16", forward="unchanged_exact_jit_row_region", rng="explicit_public",
                policy="shared_sign_hadamard128_e8m4_rtz_eden_scale_sr",
                measurement=measurement_contract, objective="complete_backward", automatic_promotion=False,
                component_contexts={**{phase: component.context.identifier for phase, component in components.items()},
                    **{phase+".row_major": component.context.identifier for phase, component in row_components.items()}})),
        initial_config=initial, axes=axes,
        config_id_fn=lambda config: stable_id(dict(backward_pipeline=config.backward_pipeline,
            **{phase: component_for(phase, getattr(config, phase)).config_id(getattr(config, phase)) for phase in components})),
        serialize_fn=asdict, deserialize_fn=lambda value: MSEdenPipelineSchedule(**value),
        update_fn=update, evaluator=lambda config: pipeline_outcome(evaluator(config),
            expected_config=config, expected_shape=shape),
        rejection_fn=rejection, extra_features_fn=features)
