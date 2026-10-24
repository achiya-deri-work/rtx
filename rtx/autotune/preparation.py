"""Operand preparation features and coverage ordering, not a latency oracle.

Callers supply only legal, numerically equivalent implementations. Staging is
per invocation: this module never implies reusing dynamic weights across calls.
"""
from dataclasses import dataclass
from math import ceil
from typing import Literal


Preparation = Literal["fused", "stage_a", "stage_b", "staged"]
PREPARATION_REVISION = 3


def preparation_selection_enabled(policy):
    """Explicit experimental opt-in, retaining the legacy environment switch."""
    import os
    return (getattr(policy, "preparation_selection", False)
            or os.getenv("RTX_PREPARATION_SELECTION", "0") == "1")


@dataclass(frozen=True)
class PreparationGeometry:
    m: int
    n: int
    k: int
    tile_m: int
    tile_n: int
    packed_bits: int = 8
    scale_group: int = 32
    scale_bytes: int = 1
    l2_bytes: int = 0
    sm_count: int = 0

    def __post_init__(self):
        for name in ("m", "n", "k", "tile_m", "tile_n", "scale_group", "scale_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.packed_bits not in (4, 8):
            raise ValueError("packed_bits must be 4 or 8")
        if any(type(v) is not int or v < 0 for v in (self.l2_bytes, self.sm_count)):
            raise ValueError("device capacities must be nonnegative integers")

    def features(self, plan: Preparation) -> dict[str, float]:
        if plan not in ("fused", "stage_a", "stage_b", "staged"):
            raise ValueError(f"unknown preparation plan: {plan}")
        mt, nt = ceil(self.m / self.tile_m), ceil(self.n / self.tile_n)
        a, b = self.m * self.k, self.n * self.k
        # Logical useful bytes: excludes physical-layout padding, cache effects,
        # persistence, scale metadata beyond block scales, and split reductions.
        pa = self.m * (ceil(self.k * self.packed_bits / 8) + ceil(self.k / self.scale_group) * self.scale_bytes)
        pb = self.n * (ceil(self.k * self.packed_bits / 8) + ceil(self.k / self.scale_group) * self.scale_bytes)
        sa, sb = plan in ("stage_a", "staged"), plan in ("stage_b", "staged")
        prepared = (a if sa else a * nt) + (b if sb else b * mt)
        workspace = pa * sa + pb * sb
        return {
            "preparation_revision": PREPARATION_REVISION,
            "a_cta_reuse": nt, "b_cta_reuse": mt,
            "cta_count": mt * nt,
            "cta_waves": mt * nt / self.sm_count if self.sm_count else 0,
            "sm_count_known": float(bool(self.sm_count)),
            "prepared_elements": prepared,
            "avoided_preparation_elements": a * nt + b * mt - prepared,
            "logical_workspace_bytes": workspace,
            "logical_staged_rw_bytes": 2 * workspace,
            "workspace_l2_ratio": workspace / self.l2_bytes if self.l2_bytes else 0,
            "l2_known": float(bool(self.l2_bytes)),
            "stage_a": float(sa), "stage_b": float(sb),
        }


def preparation_portfolio(candidates, *, plan_of, score, limit):
    """Reserve breadth before filling with high-ranked schedule refinements.

``score`` is caller-owned (measured/model/heuristic), lower is better. No
unsupported hybrid kernels are invented. Stable ordering preserves ties.
"""
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be a positive integer")
    ordered = sorted(candidates, key=score)
    representatives, refinements, seen = [], [], set()
    for candidate in ordered:
        plan = plan_of(candidate)
        if plan not in seen:
            seen.add(plan)
            representatives.append(candidate)
        else:
            refinements.append(candidate)
    return (representatives + refinements)[:limit]


def linear_phase_geometry(m, n, k, phase):
    """Logical A @ B.T GEMM dimensions; no physical transpose implied."""
    if phase == "forward":
        return m, n, k
    if phase == "dx":
        return m, k, n
    if phase == "dw":
        return n, k, m
    raise ValueError("phase must be forward, dx, or dw")


@dataclass(frozen=True)
class PreparationCandidate:
    """One complete schedule in a registered preparation implementation."""
    implementation: str
    config: object


def make_preparation_adapter(adapters, *, numerical_policy, policy_of, equivalence_key=None):
    """Compose existing adapters without bypassing legality or observations.

    ``policy_of(name, config)`` must describe actual arithmetic semantics, not
    schedule names. Component revisions and policy become cache identity.
    Callers own GPU evaluation and must time the entire execution plan.
    """
    from .core import DiscreteKernelAdapter, KernelContext, stable_id

    if not adapters or not numerical_policy:
        raise ValueError("nonempty implementations and numerical policy required")
    adapters = dict(adapters)
    first = next(iter(adapters.values())).context
    for adapter in adapters.values():
        context = adapter.context
        for field in ("workload", "device", "environment", "regime"):
            if getattr(context, field) != getattr(first, field):
                raise ValueError(f"preparation adapters disagree on {field}")
    # Shape/device/environment are already present in the outer exact context
    # and verified equal above. Duplicating them in semantic tags would forbid
    # shape/device transfer and leak device identifiers into model features.
    components = {name: {"family": adapter.context.family,
                         "kernel_revision": adapter.context.kernel_revision,
                         "tags": dict(adapter.context.tags)}
                  for name, adapter in adapters.items()}
    context = KernelContext(
        family="operand_preparation", kernel_revision=PREPARATION_REVISION,
        workload=first.workload, device=first.device, environment=first.environment,
        regime=first.regime,
        tags={"numerical_policy": numerical_policy, "components": components},
    )

    def serialize(candidate):
        return {"implementation": candidate.implementation,
                "config": adapters[candidate.implementation].serialize(candidate.config)}

    def deserialize(value):
        name = value["implementation"]
        if name not in adapters:
            raise ValueError(f"unavailable preparation implementation: {name}")
        return PreparationCandidate(name, adapters[name].deserialize(value["config"]))

    def rejection(candidate):
        if candidate.implementation not in adapters:
            return "implementation_rejected", "unavailable preparation implementation"
        if policy_of(candidate.implementation, candidate.config) != numerical_policy:
            return "policy_rejected", "preparation candidate changes numerical policy"
        return adapters[candidate.implementation].rejection(candidate.config)

    seeds = tuple(PreparationCandidate(name, adapter.initial_config)
                  for name, adapter in adapters.items())
    legal = [candidate for candidate in seeds if rejection(candidate) is None]
    if not legal:
        raise ValueError("no legal policy-matched preparation seed")
    axes = {"implementation": seeds}
    component_axes = {}
    for name, adapter in adapters.items():
        for coordinate, values in adapter.axes.items():
            key = f"{name}:{coordinate}"
            axes[key] = values
            component_axes[key] = (name, coordinate)

    def update(candidate, axis, value):
        if axis == "implementation":
            return value
        name, coordinate = component_axes[axis]
        if candidate.implementation != name:
            return candidate
        return PreparationCandidate(name, adapters[name].update_fn(candidate.config, coordinate, value))

    return DiscreteKernelAdapter(
        context=context, initial_config=legal[0], axes=axes,
        config_id_fn=lambda c: stable_id(equivalence_key(c) if equivalence_key else serialize(c)), serialize_fn=serialize,
        deserialize_fn=deserialize, update_fn=update,
        evaluator=lambda c: adapters[c.implementation].evaluate(c.config),
        rejection_fn=rejection,
        extra_features_fn=lambda c: {
            key.removeprefix("derived."): value for key, value in adapters[c.implementation].features(c.config).items()
            if key.startswith("derived.")
        },
    )


def expand_preparation_candidates(adapter, seeds, *, per_seed=8, max_examined=512, accept=None,
                                  coordinates=None):
    """Bounded one-coordinate expansion retaining each component's search axes.

    Expansion is policy/legality filtered; candidates remain suggestions, not
    measured winners. Caps bound host work as well as compilation workload.
    """
    if per_seed < 0 or max_examined < 1:
        raise ValueError("invalid preparation expansion budget")
    axes = adapter.axes if coordinates is None else {key: adapter.axes[key] for key in coordinates}
    result, seen = [], set()
    seeds = list(seeds)
    for seed in seeds:
        cid = adapter.config_id(seed)
        if cid not in seen and adapter.rejection(seed) is None and (accept is None or accept(seed)):
            seen.add(cid)
            result.append(seed)
    for seed in list(result):
        added, examined = 0, 0
        if per_seed == 0:
            break
        active_coordinates = [(coordinate, iter(values)) for coordinate, values in axes.items()]
        while active_coordinates and examined < max_examined and added < per_seed:
            pending = []
            for coordinate, values in active_coordinates:
                if examined >= max_examined or added >= per_seed:
                    break
                try:
                    value = next(values)
                except StopIteration:
                    continue
                pending.append((coordinate, values))
                examined += 1
                try:
                    candidate = adapter.update_fn(seed, coordinate, value)
                    cid = adapter.config_id(candidate)
                    if cid in seen or adapter.rejection(candidate) is not None or (accept is not None and not accept(candidate)):
                        continue
                except (ValueError, TypeError, KeyError):
                    continue
                seen.add(cid)
                result.append(candidate)
                added += 1
            active_coordinates = pending
    return result


def tune_preparation(adapter, store, candidates, *, score, policy=None, progress=print, refine=False, plan_of=None):
    """Bounded measured selection using the shared first-hit/cache machinery.

    A soft prior supplies ordering, never a winner. The supplied candidate pool
    contains complete component schedules. Rejected candidates cannot consume
    reserved plan coverage slots. Runtime launchers must call outside capture.
    """
    from .first_hit import FirstHitPolicy, tune_first_hit

    policy = policy or FirstHitPolicy()
    plan_of = plan_of or (lambda c: c.implementation)
    pool = preparation_portfolio(
        [c for c in candidates if adapter.rejection(c) is None],
        plan_of=plan_of, score=score, limit=policy.pool_size,
    )
    # The orchestrator owns the session lock; nesting flock here deadlocks.
    return tune_first_hit(adapter, store, pool, policy=policy, progress=progress,
                          candidate_group=plan_of, refine=refine, retrieval_per_context=4)
