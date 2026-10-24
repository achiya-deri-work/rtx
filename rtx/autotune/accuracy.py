"""Offline accuracy-policy search, never kernel/runtime autotuning.

Callbacks own model application, score definition and data provenance. Search
and guard must use representative, independent non-test data; this framework
cannot establish dataset disjointness or guarantee held-out improvements.
Every callback receives the complete policy state, not a local-layer proxy.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any

State = Mapping[str, str]
Evaluator = Callable[[dict[str, str]], float]


def _labels(values: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a nonempty sequence of unique labels")
    try:
        labels = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a nonempty sequence of unique labels") from exc
    if (not labels or any(not isinstance(value, str) or not value.strip() for value in labels)
            or len(set(labels)) != len(labels)):
        raise ValueError(f"{name} must be a nonempty sequence of unique nonempty string labels")
    return labels


def _inputs(groups, candidates, baselines, max_passes):
    groups = _labels(groups, "groups")
    candidates = _labels(candidates, "candidates")
    baselines = _labels(baselines, "baselines")
    if not set(baselines).issubset(candidates):
        raise ValueError("baselines must belong to candidates")
    if isinstance(max_passes, bool) or not isinstance(max_passes, int) or max_passes < 1:
        raise ValueError("max_passes must be a positive integer")
    return groups, candidates, baselines


def _score(evaluate: Evaluator, state: State) -> float:
    raw = evaluate(dict(state))  # Callback mutations never alter search state.
    if isinstance(raw, bool):
        raise ValueError("accuracy score must be finite and nonnegative, not boolean")
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("accuracy score must be finite and nonnegative") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError("accuracy score must be finite and nonnegative")
    return value


def _copy_trial(record):
    return {**record, "full_state": dict(record["full_state"])}


def coordinate_search(blocks, candidates, evaluate, initial_states=("max", "mse"),
                      progress=None, *, max_passes=1):
    """Compatibility helper: offline full-state coordinate search, lower is better.

    Baseline and candidate order provide deterministic ties. At each coordinate
    all trials hold the other coordinates fixed; only strict decreases commit.
    Returns a detached state, score and JSON-friendly search trial records.
    """
    blocks, candidates, baselines = _inputs(blocks, candidates, initial_states, max_passes)
    if not callable(evaluate) or (progress is not None and not callable(progress)):
        raise ValueError("evaluate and optional progress must be callable")
    records = []

    def score(state, phase, block=None, policy=None, pass_index=None):
        value = _score(evaluate, state)
        record = {"phase": phase, "block": block, "candidate": policy,
                  "score": value, "full_state": dict(state), "pass_index": pass_index}
        records.append(record)
        if progress is not None:
            progress(_copy_trial(record))
        return value

    initial = []
    for policy in baselines:
        state = dict.fromkeys(blocks, policy)
        initial.append((score(state, "initial", policy=policy), state))
    best_score, state = min(initial, key=lambda item: item[0])
    for pass_index in range(max_passes):
        for block in blocks:
            best_state = dict(state)
            for policy in candidates:
                trial = {**state, block: policy}
                value = score(trial, "coordinate", block, policy, pass_index)
                if value < best_score:
                    best_score, best_state = value, trial
            state = best_state
    return dict(state), best_score, [_copy_trial(record) for record in records]


def select_guard(scores, *, baselines=("max", "mse"), search_label="selected"):
    """Compatibility helper: baseline-first ties and optional search fallback."""
    baselines = _labels(baselines, "baselines")
    if not isinstance(search_label, str) or not search_label or search_label in baselines:
        raise ValueError("search_label must be nonempty and distinct from baselines")
    order = (*baselines, search_label)
    if set(scores) != set(order):
        raise ValueError("finite baseline/search guard scores required")
    checked = {name: _score(lambda _, value=scores[name]: value, {}) for name in order}
    return min(order, key=checked.__getitem__)


@dataclass(frozen=True)
class AccuracySelectionResult:
    """Immutable offline evidence; guard keys are baseline:<label> and search.

    ``guard_choice`` uses the same prefixed labels, avoiding reserved candidate
    names. Trials include both search and guard evaluations, in execution order.
    Use ``to_dict`` for a detached, JSON-serializable audit representation.
    """

    selected_state: State
    search_state: State
    search_score: float
    guard_scores: Mapping[str, float]
    guard_choice: str
    trials: tuple[Mapping[str, Any], ...]

    def __post_init__(self):
        object.__setattr__(self, "selected_state", MappingProxyType(dict(self.selected_state)))
        object.__setattr__(self, "search_state", MappingProxyType(dict(self.search_state)))
        object.__setattr__(self, "guard_scores", MappingProxyType(dict(self.guard_scores)))
        object.__setattr__(self, "trials", tuple(
            MappingProxyType({**trial, "full_state": MappingProxyType(dict(trial["full_state"]))})
            for trial in self.trials
        ))

    def to_dict(self) -> dict[str, Any]:
        return {"selected_state": dict(self.selected_state), "search_state": dict(self.search_state),
                "search_score": self.search_score, "guard_scores": dict(self.guard_scores),
                "guard_choice": self.guard_choice, "trials": [_copy_trial(trial) for trial in self.trials]}


def select_accuracy_policy(groups: Sequence[str], candidates: Sequence[str], evaluate: Evaluator,
                           *, guard_evaluate: Evaluator, baselines: tuple[str, ...],
                           max_passes: int = 1, progress=None) -> AccuracySelectionResult:
    """Select a complete offline policy by coordinate search and separate guard.

    Scores must be finite, nonnegative scalars; smaller is better. Start from
    the best homogeneous baseline, run bounded coordinate passes, then compare
    the search state with every homogeneous baseline on independent guard data.
    Guard ties prefer baselines in caller order. Neither callbacks nor progress
    can mutate internal states/records. No runtime policy is applied or cached.

    Caller owns independent representative search/guard datasets, must exclude
    test data from both callbacks, and must freeze this result before test use.
    A separate callback argument does not prove dataset independence.
    """
    groups, candidates, baselines = _inputs(groups, candidates, baselines, max_passes)
    if not callable(guard_evaluate):
        raise ValueError("guard_evaluate must be callable")
    search_state, search_score, trials = coordinate_search(
        groups, candidates, evaluate, baselines, progress, max_passes=max_passes)
    states = {f"baseline:{policy}": dict.fromkeys(groups, policy) for policy in baselines}
    states["search"] = dict(search_state)
    guard_scores = {}
    for label, state in states.items():
        value = _score(guard_evaluate, state)
        guard_scores[label] = value
        record = {"phase": "guard", "block": None, "candidate": label,
                  "score": value, "full_state": dict(state), "pass_index": None}
        trials.append(record)
        if progress is not None:
            progress(_copy_trial(record))
    choice = select_guard(guard_scores, baselines=tuple(f"baseline:{p}" for p in baselines),
                          search_label="search")
    return AccuracySelectionResult(states[choice], search_state, search_score,
                                   guard_scores, choice, tuple(trials))
