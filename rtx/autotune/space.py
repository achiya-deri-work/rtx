"""Backend-neutral, declarative search spaces for kernel autotuning."""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Callable, Iterable, Mapping, Protocol, Sequence, TypeVar

from .core import canonical_json, stable_id


ConfigT = TypeVar("ConfigT")
PortableConfig = dict[str, object]


class SearchSpace(Protocol[ConfigT]):
    """Config generation and mutation independent of a compiler or runtime."""

    @property
    def initial_config(self) -> ConfigT: ...

    def config_id(self, config: ConfigT) -> str: ...

    def serialize(self, config: ConfigT) -> Mapping[str, object]: ...

    def deserialize(self, value: Mapping[str, object]) -> ConfigT: ...

    def rejection(self, config: ConfigT) -> tuple[str, str] | None: ...

    def coordinates(self) -> Sequence[str]: ...

    def neighbors(self, config: ConfigT) -> Iterable[tuple[str, object, ConfigT]]: ...

    def sample(
        self,
        rng: random.Random,
        count: int,
        seeds: Sequence[ConfigT],
    ) -> list[ConfigT]: ...


@dataclass(frozen=True, slots=True)
class Condition:
    """Serializable activation condition for a dependent parameter."""

    parameter: str
    operator: str
    value: object

    def __post_init__(self) -> None:
        if self.operator not in {"eq", "ne", "in", "not_in", "lt", "le", "gt", "ge"}:
            raise ValueError(f"unsupported condition operator {self.operator!r}")

    def matches(self, config: Mapping[str, object]) -> bool:
        if self.parameter not in config:
            return False
        actual = config[self.parameter]
        if self.operator == "eq":
            return actual == self.value
        if self.operator == "ne":
            return actual != self.value
        if self.operator in {"in", "not_in"}:
            if not isinstance(self.value, (tuple, list, set, frozenset)):
                raise TypeError(f"condition {self.operator!r} requires a sequence value")
            contains = actual in self.value
            return contains if self.operator == "in" else not contains
        try:
            if self.operator == "lt":
                return actual < self.value  # type: ignore[operator]
            if self.operator == "le":
                return actual <= self.value  # type: ignore[operator]
            if self.operator == "gt":
                return actual > self.value  # type: ignore[operator]
            return actual >= self.value  # type: ignore[operator]
        except TypeError:
            return False

    def as_dict(self) -> dict[str, object]:
        return {
            "parameter": self.parameter,
            "operator": self.operator,
            "value": self.value,
        }


@dataclass(frozen=True, slots=True)
class DiscreteParameter:
    """Finite parameter whose presence may depend on earlier parameters."""

    name: str
    values: tuple[object, ...]
    default: object | None = None
    active_if: tuple[Condition, ...] = ()
    tags: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("parameter names cannot be empty")
        if not self.values:
            raise ValueError(f"parameter {self.name!r} has no values")
        canonical = [canonical_json(value) for value in self.values]
        if len(canonical) != len(set(canonical)):
            raise ValueError(f"parameter {self.name!r} contains duplicate values")
        default = self.values[0] if self.default is None else self.default
        if canonical_json(default) not in canonical:
            raise ValueError(f"default for {self.name!r} is not in its domain")
        object.__setattr__(self, "default", default)

    def active(self, config: Mapping[str, object]) -> bool:
        return all(condition.matches(config) for condition in self.active_if)

    def contains(self, value: object) -> bool:
        encoded = canonical_json(value)
        return any(canonical_json(candidate) == encoded for candidate in self.values)

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "values": list(self.values),
            "default": self.default,
            "active_if": [condition.as_dict() for condition in self.active_if],
            "tags": dict(self.tags),
        }


@dataclass(frozen=True, slots=True)
class SpaceConstraint:
    """Named project-specific legality rule with a portable descriptor."""

    name: str
    check: Callable[[Mapping[str, object]], str | None] = field(
        compare=False, repr=False
    )
    status: str = "implementation_rejected"
    description: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "status": self.status,
            "description": self.description,
        }


@dataclass(slots=True)
class ConditionalSearchSpace:
    """Hierarchical finite space suitable for schedules and kernel layouts.

    Conditions may only reference earlier parameters. Normalization therefore
    has deterministic semantics: inactive parameters disappear, while newly
    activated parameters receive their declared defaults.
    """

    parameters: Sequence[DiscreteParameter]
    constraints: Sequence[SpaceConstraint] = ()
    name: str = "conditional_discrete"
    revision: int = 1
    _by_name: dict[str, DiscreteParameter] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.parameters:
            raise ValueError("a search space requires at least one parameter")
        names = [parameter.name for parameter in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError("search-space parameter names must be unique")
        seen: set[str] = set()
        for parameter in self.parameters:
            for condition in parameter.active_if:
                if condition.parameter not in seen:
                    raise ValueError(
                        f"{parameter.name!r} depends on non-earlier parameter "
                        f"{condition.parameter!r}"
                    )
            seen.add(parameter.name)
        self._by_name = {parameter.name: parameter for parameter in self.parameters}

    @property
    def initial_config(self) -> PortableConfig:
        return self.normalize({})

    @property
    def identifier(self) -> str:
        return stable_id(self.as_dict())

    def as_dict(self) -> dict[str, object]:
        return {
            "type": self.name,
            "revision": self.revision,
            "parameters": [parameter.as_dict() for parameter in self.parameters],
            "constraints": [constraint.as_dict() for constraint in self.constraints],
        }

    def normalize(self, config: Mapping[str, object]) -> PortableConfig:
        unknown = set(config) - set(self._by_name)
        if unknown:
            raise ValueError(f"unknown search-space parameters: {sorted(unknown)}")
        normalized: PortableConfig = {}
        for parameter in self.parameters:
            if not parameter.active(normalized):
                continue
            value = config.get(parameter.name, parameter.default)
            if not parameter.contains(value):
                raise ValueError(
                    f"value {value!r} is outside the domain of {parameter.name!r}"
                )
            normalized[parameter.name] = value
        return normalized

    def config_id(self, config: Mapping[str, object]) -> str:
        return stable_id(self.normalize(config))

    def serialize(self, config: Mapping[str, object]) -> Mapping[str, object]:
        return self.normalize(config)

    def deserialize(self, value: Mapping[str, object]) -> PortableConfig:
        return self.normalize(value)

    def rejection(self, config: Mapping[str, object]) -> tuple[str, str] | None:
        try:
            normalized = self.normalize(config)
        except ValueError as exc:
            return "implementation_rejected", str(exc)
        if dict(config) != normalized:
            return (
                "implementation_rejected",
                "configuration is not normalized for its active conditional parameters",
            )
        for constraint in self.constraints:
            reason = constraint.check(normalized)
            if reason is not None:
                return constraint.status, f"{constraint.name}: {reason}"
        return None

    def coordinates(self) -> Sequence[str]:
        return tuple(parameter.name for parameter in self.parameters)

    def active_coordinates(self, config: Mapping[str, object]) -> tuple[str, ...]:
        normalized = self.normalize(config)
        return tuple(
            parameter.name
            for parameter in self.parameters
            if parameter.active(normalized)
        )

    def update(
        self, config: Mapping[str, object], coordinate: str, value: object
    ) -> PortableConfig:
        if coordinate not in self._by_name:
            raise KeyError(coordinate)
        current = self.normalize(config)
        parameter = self._by_name[coordinate]
        if not parameter.active(current):
            raise ValueError(f"parameter {coordinate!r} is inactive")
        if not parameter.contains(value):
            raise ValueError(f"value {value!r} is outside {coordinate!r}")
        changed = dict(current)
        changed[coordinate] = value
        return self.normalize(changed)

    def neighbors(
        self, config: Mapping[str, object]
    ) -> Iterable[tuple[str, object, PortableConfig]]:
        current = self.normalize(config)
        current_id = self.config_id(current)
        seen: set[str] = set()
        for coordinate in self.active_coordinates(current):
            parameter = self._by_name[coordinate]
            for value in parameter.values:
                candidate = self.update(current, coordinate, value)
                identifier = self.config_id(candidate)
                if identifier == current_id or identifier in seen:
                    continue
                if self.rejection(candidate) is not None:
                    continue
                seen.add(identifier)
                yield coordinate, value, candidate

    def sample(
        self,
        rng: random.Random,
        count: int,
        seeds: Sequence[Mapping[str, object]],
    ) -> list[PortableConfig]:
        if count <= 0:
            return []
        bases = [self.normalize(seed) for seed in seeds] or [self.initial_config]
        result: list[PortableConfig] = []
        seen: set[str] = set()
        attempts = max(64, count * 32)
        for _attempt in range(attempts):
            if len(result) >= count:
                break
            candidate = dict(rng.choice(bases))
            steps = rng.randint(1, max(1, min(8, len(self.parameters))))
            for _step in range(steps):
                active = self.active_coordinates(candidate)
                if not active:
                    break
                coordinate = rng.choice(active)
                parameter = self._by_name[coordinate]
                proposal = self.update(candidate, coordinate, rng.choice(parameter.values))
                if self.rejection(proposal) is None:
                    candidate = proposal
            identifier = self.config_id(candidate)
            if identifier in seen or self.rejection(candidate) is not None:
                continue
            seen.add(identifier)
            result.append(candidate)
        return result


__all__ = [
    "Condition",
    "ConditionalSearchSpace",
    "DiscreteParameter",
    "PortableConfig",
    "SearchSpace",
    "SpaceConstraint",
]
