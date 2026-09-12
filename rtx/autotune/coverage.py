"""Deterministic structural coverage, independent of measured winners.

This is a finite coverage plan, not an exhaustive Cartesian search. Legal
neighbours of supplied backend seeds are interleaved by component and axis so
an interrupted run does not spend its entire budget on the first component.
"""
from collections import defaultdict, deque

from .core import Proposal


def coverage_plan(adapter, seeds=()):
    bases = [adapter.initial_config, *seeds]
    queues = defaultdict(lambda: defaultdict(deque))
    audit = []
    seen = {adapter.config_id(adapter.initial_config)}
    visited_bases = set()
    proposals = []
    # Reserve seeds before enumerating neighbours: otherwise an earlier
    # neighbour steals a seed's identity and buries that backend in its queue.
    for base in bases[1:]:
        identifier=adapter.config_id(base)
        if identifier not in seen and adapter.rejection(base) is None:
            seen.add(identifier)
            proposals.append(Proposal(base,"coverage",coordinate="seed"))
    for base in bases:
        identifier = adapter.config_id(base)
        if identifier in visited_bases:
            continue
        visited_bases.add(identifier)
        reason = adapter.rejection(base)
        if reason is not None:
            audit.append({"coordinate": "seed", "config_id": identifier,
                          "status": "rejected", "reason": reason})
            continue
        if identifier not in seen:
            seen.add(identifier)
            proposals.append(Proposal(base, "coverage", coordinate="seed"))
        for axis, values in adapter.axes.items():
            for value in values:
                row = {"coordinate": axis, "value": value, "base_id": identifier}
                try:
                    candidate = adapter.update_fn(base, axis, value)
                    cid = adapter.config_id(candidate)
                    rejection = adapter.rejection(candidate)
                    row.update(config_id=cid, status="rejected" if rejection else
                               "inactive_or_duplicate" if cid in seen else "planned")
                    if rejection:
                        row["reason"] = rejection
                    elif cid not in seen:
                        seen.add(cid)
                        component=axis.split(".")[0]
                        suffix=axis.split(".")[-1]
                        if suffix.startswith(("dx_","dw_")):
                            component += "."+suffix[:2]
                        queues[component][axis].append(Proposal(
                            candidate, "coverage", parent_config_id=identifier,
                            coordinate=axis, coordinate_value=value))
                except (ValueError, TypeError, KeyError) as exc:
                    row.update(status="update_error", reason=str(exc))
                audit.append(row)
    # Visit each axis once before taking its second value. Prioritize structural
    # dimensions, but retain every remaining axis rather than dropping it.
    def priority(axis):
        return (not any(word in axis for word in
                        ("backend", "geometry", "stages", "split_count", "tile_m",
                         "materialized_rows", "layout_transport")), axis)
    component_queues = []
    for axes in queues.values():
        ordered = deque(axes[name] for name in sorted(axes, key=priority))
        component = deque()
        while ordered:
            queue = ordered.popleft()
            component.append(queue.popleft())
            if queue:
                ordered.append(queue)
        component_queues.append(component)
    while any(component_queues):
        for queue in component_queues:
            if queue:
                proposals.append(queue.popleft())
    return proposals, audit
