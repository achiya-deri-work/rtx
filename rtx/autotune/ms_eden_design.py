"""Composable research implementations over the existing complete-backward tuner.

No runtime defaults or numerical policies are changed here. Implementation
selection is part of candidate identity, never an out-of-band CLI switch.
"""
from dataclasses import asdict, dataclass, field, replace

from .core import DiscreteKernelAdapter, KernelContext, stable_id
from .ms_eden_pipeline import (
    MSEdenPipelineSchedule, make_ms_eden_pipeline_adapter, pipeline_outcome,
)
from .outcomes import TrialOutcome
from .posthoc import PosthocOperandGeometry


@dataclass(frozen=True)
class MSEdenDesign:
    schedule: MSEdenPipelineSchedule = field(default_factory=MSEdenPipelineSchedule)
    transpose_loader: str = "native"
    finalizer: str = "separate"

    def __post_init__(self):
        if isinstance(self.schedule, dict):
            object.__setattr__(self, "schedule", MSEdenPipelineSchedule(**self.schedule))


IMPLEMENTATIONS = {
    "transpose_loader": ("native", "tma_xor4_1", "tma_xor4_2", "tma_xor4_3",
                         "tma_xor5_1", "tma_xor5_2", "tma_xor5_3"),
    "finalizer": ("separate", "dual", "dual_paired", "dual_cta"),
}


def implementation_rejection(design):
    """Reject unsupported dispatch instead of silently timing another kernel."""
    for name, choices in IMPLEMENTATIONS.items():
        if getattr(design, name) not in choices:
            return "implementation_rejected", f"unknown {name}"
    if design.finalizer != "separate":
        if design.schedule.backward_pipeline != "operand_paired":
            return "implementation_rejected", "dual finalizers require operand_paired"
        for phase in ("dx", "dw"):
            leg = getattr(design.schedule, phase)
            if leg.amax_reduction != "staged":
                return "implementation_rejected", "dual finalizers require staged maxima"
            if leg.operand_schedule("a").finalizer_threads != leg.operand_schedule("b").finalizer_threads:
                return "implementation_rejected", "dual finalizers require equal threads within each leg"
    if design.transpose_loader != "native":
        # dX B=W^T; dW A=dY^T and B=X^T. dX A is the row reader.
        for phase, operand in (("dx", "b"), ("dw", "a"), ("dw", "b")):
            s = getattr(design.schedule, phase).operand_schedule(operand)
            if not (s.producer_threads == 128 and s.producer_async_copy
                    and s.producer_load == "coalesced_x8"
                    and s.producer_transpose_values_per_lane == 16
                    and s.producer_copy_bits == 32
                    and not s.producer_unroll_rows
                    and s.producer_stages == int(design.transpose_loader[-1])):
                return "implementation_rejected", f"{phase}.{operand} cannot dispatch requested TMA loader"
    return None


def make_ms_eden_design_adapter(shape, evaluator, *, initial=None, **kwargs):
    """Use normal search/model/store APIs with implementation + schedule axes.

    Evaluators return the existing complete-backward evidence AND an exact
    ``design`` manifest. Per-order stationarity remains mandatory.
    """
    initial = initial or MSEdenDesign()
    base = make_ms_eden_pipeline_adapter(shape, lambda _: None,
                                         initial=initial.schedule, **kwargs)

    def reject(config):
        return implementation_rejection(config) or base.rejection(config.schedule)

    def update(config, axis, value):
        if axis in IMPLEMENTATIONS:
            return replace(config, **{axis: value})
        if ".operand." in axis:
            phase, _, operand, field_name = axis.split(".")
            leg = getattr(config.schedule, phase)
            geometry = getattr(leg, operand + "_geometry") or PosthocOperandGeometry()
            geometry = replace(geometry, **{field_name: value})
            if all(value is None for value in asdict(geometry).values()):
                geometry = None
            leg = replace(leg, **{operand + "_geometry": geometry})
            return replace(config, schedule=replace(config.schedule, **{phase: leg}))
        return replace(config, schedule=base.update_fn(config.schedule, axis, value))

    def evaluate(config):
        rejection = reject(config)
        if rejection:
            return TrialOutcome(rejection[0], error=rejection[1])
        result = evaluator(config)
        if result.get("design") != asdict(config):
            return TrialOutcome("correctness_error", error="missing or mismatched implementation manifest")
        if result.get("baseline_policy") != "stable_mxfp8":
            return TrialOutcome("correctness_error", error="matched stable MXFP8 baseline required")
        if config.transpose_loader != "native":
            records = result.get("design_dispatch", [])
            transposes = [record for record in records if record.get("transpose") is True]
            if not transposes or any(record.get("selected") != config.transpose_loader
                                     for record in transposes):
                return TrialOutcome("correctness_error", error="requested TMA dispatch was not established")
        return pipeline_outcome(result, expected_config=config.schedule,
                                expected_shape=shape, require_order_quality=True)

    # Keep compiler register allocation as a consequence of structural choices.
    # The underlying general-purpose adapter also serves legacy cap studies;
    # these are deliberately not research axes for this design search.
    axes = {name: values for name, values in base.axes.items()
            if "registers" not in name and "maxrregcount" not in name
            and name not in ("dw.producer_row_tile_batch", "dw.producer_row_load_bits",
                             "dw.producer_values_per_lane")}
    sm_count = base.context.device.get("multiprocessor_count")
    wave_grids = (() if type(sm_count) is not int or sm_count <= 0
                  else tuple(sm_count * waves for waves in (1, 2, 4, 8)))
    for phase in ("dx", "dw"):
        for field_name in ("producer_blocks", "finalizer_blocks"):
            shared_axis = f"{phase}.{field_name}"
            grids = tuple(sorted(set(axes[shared_axis]) | set(wave_grids)))
            axes[shared_axis] = grids
            for operand in ("a", "b"):
                axes[f"{phase}.operand.{operand}.{field_name}"] = (None, *grids)
        for field_name in ("producer_threads", "finalizer_threads"):
            for operand in ("a", "b"):
                axes[f"{phase}.operand.{operand}.{field_name}"] = (None, 64, 128, 256, 512)
    def identity(config):
        # Both dW operands are logical transposes: their row-only controls
        # cannot change executed work. Preserve the original manifest, but
        # deduplicate by active semantics and the base adapter's normalization
        # of inherited geometry, explicit RNG and inactive split-K controls.
        schedule = replace(config.schedule, dw=replace(config.schedule.dw,
            producer_row_tile_batch=1, producer_row_load_bits=16, producer_values_per_lane=4))
        return stable_id(dict(schedule=base.config_id(schedule),
            transpose_loader=config.transpose_loader, finalizer=config.finalizer))

    return DiscreteKernelAdapter(
        context=KernelContext("ms_eden_design_research", 4, base.context.workload,
            device=base.context.device, environment=base.context.environment,
            tags={"pipeline_context": base.context.identifier, "automatic_promotion": False}),
        initial_config=initial, axes={**axes, **IMPLEMENTATIONS},
        config_id_fn=identity, serialize_fn=asdict,
        deserialize_fn=lambda v: MSEdenDesign(**v), update_fn=update,
        evaluator=evaluate, rejection_fn=reject,
        extra_features_fn=lambda c: base.extra_features_fn(c.schedule),
    )
