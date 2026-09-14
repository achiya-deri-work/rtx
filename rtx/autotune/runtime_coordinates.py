"""Project experimental schedules onto coordinates executed by the runtime."""
from dataclasses import replace
import os

BWD_COORDINATE_REVISION = 1


def _backward_schedule_adapter(adapter):
    if adapter.context.tags.get('runtime_backward_coordinate_revision') == BWD_COORDINATE_REVISION:
        return _search_identity(adapter)
    def canonical(config):
        # The dual-stream implementation launches dX first regardless of this
        # field. Preserve unsupported interleaved values for legality rejection.
        if config.stream_schedule == 'dual_stream' and config.execution_order == 'dw_first':
            return replace(config, execution_order='dx_first')
        return config
    projected = replace(adapter,
        context=replace(adapter.context, tags={**adapter.context.tags,
            'runtime_backward_coordinate_revision':BWD_COORDINATE_REVISION}),
        initial_config=canonical(adapter.initial_config),
        config_id_fn=lambda c:adapter.config_id(canonical(c)),
        serialize_fn=lambda c:adapter.serialize(canonical(c)),
        deserialize_fn=lambda value:canonical(adapter.deserialize(value)),
        update_fn=lambda c,k,v:canonical(adapter.update_fn(canonical(c),k,v)),
        evaluator=lambda c:adapter.evaluator(canonical(c)),
        rejection_fn=lambda c:adapter.rejection_fn(canonical(c)),
        extra_features_fn=(None if adapter.extra_features_fn is None else
            lambda c:adapter.extra_features_fn(canonical(c))))
    return _search_identity(projected)


def _search_identity(adapter):
    from .core import stable_id
    # Coordinate/value order changes protected neighborhoods and random walks.
    # A mapping hash alone sorts keys and would lose that search contract.
    domain = stable_id(dict(schema=1, initial=adapter.serialize(adapter.initial_config),
                           axes=[(name, list(values)) for name, values in adapter.axes.items()]))
    return replace(adapter, context=replace(adapter.context, tags={**adapter.context.tags,
        'runtime_search_space_revision': 1, 'runtime_search_space': domain}))


def process_l2_policy():
    value = os.environ.get('RTX_L2_FETCH_GRANULARITY')
    if value is None:
        return 'driver_default'
    try:
        numeric = int(value)
    except ValueError as exc:
        raise ValueError('invalid RTX_L2_FETCH_GRANULARITY') from exc
    if numeric not in (0, 32, 64, 128):
        raise ValueError('invalid RTX_L2_FETCH_GRANULARITY')
    return str(numeric)


def runtime_schedule_adapter(adapter):
    """Keep schedule choices, but never benchmark ignored per-op L2 limits.

    Production launchers honor only RTX_L2_FETCH_GRANULARITY. Canonicalizing
    this inactive field deduplicates retrieved/neighbor schedules without
    discarding their meaningful tile/pipeline coordinates. Offline adapters
    remain untouched. The caller must namespace journals by evidence contract.
    """
    if adapter.context.family == 'mxfp8_bwd':
        return _backward_schedule_adapter(adapter)
    if not hasattr(adapter.initial_config, 'l2_fetch_granularity'):
        return _search_identity(adapter)
    def canonical(config):
        return replace(config, l2_fetch_granularity=None)
    context = replace(adapter.context, tags={**adapter.context.tags,
        'runtime_l2_policy': process_l2_policy(), 'runtime_measurement_revision': 1})
    axes = {k:v for k,v in adapter.axes.items() if k != 'global_l2_fetch'}
    # DiscreteKernelAdapter requires an axis even for a singleton domain.
    # A no-op coordinate lets normal deduplication/exhaustion handle it.
    projected = replace(adapter, context=context, initial_config=canonical(adapter.initial_config),
        axes=axes or {'runtime_fixed_schedule': (None,)},
        config_id_fn=lambda c: adapter.config_id(canonical(c)),
        serialize_fn=lambda c: adapter.serialize(canonical(c)),
        deserialize_fn=lambda value: canonical(adapter.deserialize(value)),
        update_fn=lambda c,k,v: (canonical(adapter.update_fn(canonical(c),k,v))
                                 if axes else canonical(c)),
        evaluator=lambda c: adapter.evaluator(canonical(c)),
        rejection_fn=lambda c: adapter.rejection_fn(canonical(c)),
        extra_features_fn=(None if adapter.extra_features_fn is None else
                           lambda c: adapter.extra_features_fn(canonical(c))))
    return _search_identity(projected)
