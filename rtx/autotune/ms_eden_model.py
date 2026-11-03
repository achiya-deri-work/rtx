"""Model-context refinement of previously qualified MS-EDEN kernel designs.

The evaluator must time the complete compiled workload. This adapter changes
only per-shape orchestration; it does not replace kernel qualification or
publish runtime defaults. It composes with the normal stores and searchers.
"""
from dataclasses import asdict, dataclass, replace

from .core import DiscreteKernelAdapter, KernelContext, stable_id
from .ms_eden_design import MSEdenDesign, make_ms_eden_design_adapter
from .ms_eden_pipeline import BACKWARD_PIPELINES


@dataclass(frozen=True)
class MSEdenModelSchedule:
    pipelines: tuple[str, ...]

    def __post_init__(self):
        object.__setattr__(self,'pipelines',tuple(self.pipelines))
        if not self.pipelines or any(p not in BACKWARD_PIPELINES for p in self.pipelines):
            raise ValueError('explicit supported per-shape pipelines required')


def make_ms_eden_model_adapter(designs, evaluator, *, batch, sequence,
                               model_context, device=None, environment=None):
    """Refine stream ownership using actual model cost, not isolated-cost sums.

    ``designs`` maps canonical M,N,K strings to exact designs. ``evaluator``
    receives a new expanded design map and returns a correctness- and timing-
    checked TrialOutcome. Batch/sequence, model contract, base designs, device
    and environment all participate in context identity.
    """
    if type(batch) is not int or batch<1 or type(sequence) is not int or sequence<1:
        raise ValueError('positive integer batch and sequence required')
    if not designs or not model_context:
        raise ValueError('nonempty designs and model context required')
    keys=tuple(sorted(designs))
    bases={}
    adapters={}
    for key in keys:
        shape=tuple(map(int,key.split(',')))
        if len(shape)!=3 or any(v<1 or v%128 for v in shape) or key!=','.join(map(str,shape)):
            raise ValueError('canonical positive aligned M,N,K required')
        if shape[0]!=batch*sequence:
            raise ValueError('batch and sequence must match projection M')
        design=designs[key]
        bases[key]=MSEdenDesign(**design) if isinstance(design,dict) else design
        adapters[key]=make_ms_eden_design_adapter(shape,lambda _:None,
            initial=bases[key],device=device,environment=environment,allow_row_major=True)

    def expand(config):
        if len(config.pipelines)!=len(keys):raise ValueError('model schedule arity mismatch')
        return {key:asdict(replace(bases[key],schedule=replace(bases[key].schedule,
            backward_pipeline=pipeline))) for key,pipeline in zip(keys,config.pipelines)}

    def serialize(config):
        expand(config)
        return {'pipelines':dict(zip(keys,config.pipelines))}

    def deserialize(value):
        if set(value)!= {'pipelines'} or set(value['pipelines'])!=set(keys):
            raise ValueError('model schedule must exactly cover base shapes')
        return MSEdenModelSchedule(tuple(value['pipelines'][key] for key in keys))

    def rejection(config):
        for key,value in expand(config).items():
            reason=adapters[key].rejection(MSEdenDesign(**value))
            if reason:return reason
        return None

    def update(config,axis,value):
        key=axis.removeprefix('pipeline:')
        if key not in keys or value not in BACKWARD_PIPELINES:raise ValueError('unknown pipeline coordinate')
        values=list(config.pipelines);values[keys.index(key)]=value
        return MSEdenModelSchedule(tuple(values))

    initial=MSEdenModelSchedule(tuple(bases[key].schedule.backward_pipeline for key in keys))
    adapter=DiscreteKernelAdapter(
        context=KernelContext('ms_eden_model_schedule_research',1,
            dict(batch=batch,sequence=sequence,shapes=keys,model=model_context),
            device=device or {},environment=environment or {},
            tags=dict(base_designs=stable_id({k:asdict(v) for k,v in bases.items()}),
                      objective='complete_compiled_model',automatic_promotion=False)),
        initial_config=initial,axes={f'pipeline:{key}':BACKWARD_PIPELINES for key in keys},
        config_id_fn=lambda c:stable_id(serialize(c)),serialize_fn=serialize,deserialize_fn=deserialize,
        update_fn=update,rejection_fn=rejection,evaluator=lambda c:evaluator(expand(c)))
    reason=adapter.rejection(initial)
    if reason:raise ValueError(f'base model design is not qualified/legal: {reason[1]}')
    return adapter
