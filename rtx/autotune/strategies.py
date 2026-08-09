"""Composable candidate generators: exploration, learned ranking, and local search."""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Generic, Mapping, Protocol, Sequence

import numpy as np

from .core import ConfigT, KernelAdapter, Observation, Proposal, SearchHistory
from .cost_model import CostModel, GradientBoostedCostModel, GradientBoostedFeasibilityModel
from .pretrained import ConditionalRuleSet


class SearchStrategy(Protocol, Generic[ConfigT]):
    name: str

    def propose(
        self,
        adapter: KernelAdapter[ConfigT],
        history: SearchHistory[ConfigT],
        rng: random.Random,
        limit: int,
    ) -> list[Proposal[ConfigT]]: ...

    def observe(self, observation: Observation[ConfigT]) -> None: ...


@dataclass(slots=True)
class RandomSearch(Generic[ConfigT]):
    pool_multiplier: int = 4
    max_batches: int = 8
    max_pool_size: int = 4096
    name: str = "random"

    def __post_init__(self) -> None:
        if self.pool_multiplier <= 0 or self.max_batches <= 0:
            raise ValueError("random-search pool and retry counts must be positive")
        if self.max_pool_size <= 0:
            raise ValueError("random-search maximum pool size must be positive")

    def propose(
        self,
        adapter: KernelAdapter[ConfigT],
        history: SearchHistory[ConfigT],
        rng: random.Random,
        limit: int,
    ) -> list[Proposal[ConfigT]]:
        seeds = [item.config for item in sorted(history.successful, key=lambda item: item.score)[:8]]
        pool_size = max(limit * self.pool_multiplier, limit)
        eligible: list[ConfigT] = []
        eligible_ids: set[str] = set()
        for _batch in range(self.max_batches):
            candidates = adapter.sample(
                rng, min(pool_size, self.max_pool_size), seeds
            )
            for candidate in candidates:
                config_id = adapter.config_id(candidate)
                if (
                    config_id in history.seen_ids
                    or config_id in eligible_ids
                    or adapter.rejection(candidate) is not None
                ):
                    continue
                eligible_ids.add(config_id)
                eligible.append(candidate)
            if len(eligible) >= limit or pool_size >= self.max_pool_size:
                break
            pool_size = min(self.max_pool_size, pool_size * 2)
        proposals: list[Proposal[ConfigT]] = []
        for rank, candidate in enumerate(eligible):
            proposals.append(
                Proposal(
                    candidate,
                    self.name,
                    metadata={
                        "candidate_pool_size": len(eligible),
                        "candidate_rank": rank,
                        "candidate_draw_pool": pool_size,
                        "proposal_probability": (
                            None if not eligible else 1.0 / len(eligible)
                        ),
                        "proposal_probability_kind": "approximate_pool_uniform",
                    },
                )
            )
            if len(proposals) >= limit:
                break
        return proposals

    def observe(self, observation: Observation[ConfigT]) -> None:
        return None


@dataclass(slots=True)
class CoordinateLocalSearch(Generic[ConfigT]):
    """Beam coordinate search around measured incumbents."""

    beam_width: int = 2
    shuffle_coordinates: bool = False
    name: str = "coordinate_local"

    def propose(
        self,
        adapter: KernelAdapter[ConfigT],
        history: SearchHistory[ConfigT],
        rng: random.Random,
        limit: int,
    ) -> list[Proposal[ConfigT]]:
        ranked = sorted(history.successful, key=lambda item: item.score)
        bases = ranked[: self.beam_width]
        if not bases:
            return [Proposal(adapter.initial_config, self.name)]
        candidates: list[Proposal[ConfigT]] = []
        emitted: set[str] = set()
        for base in bases:
            neighbors = list(adapter.neighbors(base.config))
            if self.shuffle_coordinates:
                rng.shuffle(neighbors)
            for coordinate, value, candidate in neighbors:
                config_id = adapter.config_id(candidate)
                if config_id in history.seen_ids or config_id in emitted:
                    continue
                if adapter.rejection(candidate) is not None:
                    continue
                emitted.add(config_id)
                candidates.append(
                    Proposal(
                        candidate,
                        self.name,
                        parent_config_id=base.config_id,
                        coordinate=coordinate,
                        coordinate_value=value,
                    )
                )
                if len(candidates) >= limit:
                    return candidates
        return candidates

    def observe(self, observation: Observation[ConfigT]) -> None:
        return None


@dataclass(slots=True)
class CostModelGuidedSearch(Generic[ConfigT]):
    """Train gradient boosting on measured trials and rank a broad legal pool."""

    model: CostModel = field(default_factory=GradientBoostedCostModel)
    feasibility_model: GradientBoostedFeasibilityModel = field(
        default_factory=GradientBoostedFeasibilityModel
    )
    warmup: SearchStrategy[ConfigT] = field(default_factory=RandomSearch)
    min_observations: int = 16
    pool_size: int = 2048
    refit_interval: int = 8
    exploration: float = 0.15
    feasibility_exploration: float = 0.5
    minimum_optimistic_feasibility: float = 0.05
    include_local_neighbors: int = 4
    model_provenance: Mapping[str, object] | None = None
    name: str = "gradient_boosted"
    _fitted_count: int = field(default=0, init=False, repr=False)
    _feasibility_fitted_count: int = field(default=0, init=False, repr=False)
    _queue: list[Proposal[ConfigT]] = field(default_factory=list, init=False, repr=False)

    def propose(
        self,
        adapter: KernelAdapter[ConfigT],
        history: SearchHistory[ConfigT],
        rng: random.Random,
        limit: int,
    ) -> list[Proposal[ConfigT]]:
        current_successes = history.successful
        training_successes = history.training_successful
        if len(training_successes) < self.min_observations and not self.model.fitted:
            proposals = self.warmup.propose(adapter, history, rng, limit)
            for proposal in proposals:
                proposal.strategy = self.name
                proposal.metadata["phase"] = "warmup"
            return proposals
        if (
            not self.model.fitted
            or len(training_successes) - self._fitted_count >= self.refit_interval
        ):
            self.model.fit(history.observations)  # type: ignore[arg-type]
            self._fitted_count = len(training_successes)
            self._queue.clear()
        feasibility_count = self.feasibility_model.labeled_count(
            history.observations  # type: ignore[arg-type]
        )
        if (
            not self.feasibility_model.fitted
            or feasibility_count - self._feasibility_fitted_count >= self.refit_interval
        ):
            self.feasibility_model.fit(history.observations)  # type: ignore[arg-type]
            self._feasibility_fitted_count = feasibility_count
            self._queue.clear()
        if not self.model.fitted:
            return self.warmup.propose(adapter, history, rng, limit)

        queued: list[Proposal[ConfigT]] = []
        while self._queue and len(queued) < limit:
            proposal = self._queue.pop(0)
            if adapter.config_id(proposal.config) not in history.seen_ids:
                queued.append(proposal)
        if len(queued) >= limit:
            return queued

        seed_observations = current_successes or training_successes
        seeds = [
            item.config for item in sorted(seed_observations, key=lambda item: item.score)[:8]
        ]
        candidate_by_id: dict[str, ConfigT] = {}
        for candidate in adapter.sample(rng, self.pool_size, seeds):
            if adapter.rejection(candidate) is not None:
                continue
            candidate_by_id.setdefault(adapter.config_id(candidate), candidate)
        for base in sorted(current_successes, key=lambda item: item.score)[
            : self.include_local_neighbors
        ]:
            for _coordinate, _value, candidate in adapter.neighbors(base.config):
                if adapter.rejection(candidate) is not None:
                    continue
                candidate_by_id.setdefault(adapter.config_id(candidate), candidate)
        for config_id in history.seen_ids:
            candidate_by_id.pop(config_id, None)
        candidates = list(candidate_by_id.values())
        if not candidates:
            return []
        feature_rows = [adapter.features(item) for item in candidates]
        mean, uncertainty = self.model.predict(feature_rows)
        acquisition = mean - self.exploration * uncertainty
        compile_probability = None
        compile_uncertainty = None
        if self.feasibility_model.fitted:
            compile_probability, compile_uncertainty = self.feasibility_model.predict(
                feature_rows
            )
            optimistic_probability = np.clip(
                compile_probability
                + self.feasibility_exploration * compile_uncertainty,
                self.minimum_optimistic_feasibility,
                1.0,
            )
            # Expected latency per successful trial. Uncertain regions retain
            # an optimistic probability and are explored rather than pruned.
            acquisition = mean / optimistic_probability - self.exploration * uncertainty
        order = sorted(range(len(candidates)), key=lambda index: float(acquisition[index]))
        queue_limit = min(len(order), max(256, self.refit_interval * 8))
        best_acquisition = float(acquisition[order[0]])
        for rank, index in enumerate(order[:queue_limit]):
            metadata = {
                "predicted_ms": float(mean[index]),
                "predicted_std_ms": float(uncertainty[index]),
                "acquisition": float(acquisition[index]),
                "training_successes": len(training_successes),
                "current_successes": len(current_successes),
                "model_parameters": self.model.parameter_count,
                "compile_training_labels": feasibility_count,
                "feasibility_model_parameters": self.feasibility_model.parameter_count,
                "candidate_pool_size": len(candidates),
                "candidate_rank": rank,
                "acquisition_gap_to_best": float(acquisition[index]) - best_acquisition,
                "proposal_probability": None,
                "proposal_probability_kind": "deterministic_rank_after_stochastic_pool",
            }
            if self.model_provenance is not None:
                metadata["pretrained"] = dict(self.model_provenance)
            if compile_probability is not None and compile_uncertainty is not None:
                metadata.update(
                    predicted_compile_probability=float(compile_probability[index]),
                    predicted_compile_uncertainty=float(compile_uncertainty[index]),
                )
            self._queue.append(
                Proposal(
                    candidates[index],
                    self.name,
                    metadata=metadata,
                )
            )
        while self._queue and len(queued) < limit:
            proposal = self._queue.pop(0)
            if adapter.config_id(proposal.config) not in history.seen_ids:
                queued.append(proposal)
        return queued

    def observe(self, observation: Observation[ConfigT]) -> None:
        return None


@dataclass(slots=True)
class CostModelLocalSearch(Generic[ConfigT]):
    """Rank complete legal coordinate neighborhoods with a fitted cost model."""

    model: CostModel
    feasibility_model: GradientBoostedFeasibilityModel | None = None
    rule_prior: ConditionalRuleSet | None = None
    rule_weight: float = 0.15
    model_provenance: Mapping[str, object] | None = None
    beam_width: int = 3
    exploration: float = 0.05
    feasibility_exploration: float = 0.5
    minimum_optimistic_feasibility: float = 0.05
    refresh_interval: int = 8
    refit_interval: int = 32
    name: str = "model_local"
    _queue: list[Proposal[ConfigT]] = field(default_factory=list, init=False, repr=False)
    _observed: int = field(default=0, init=False, repr=False)
    _fitted_count: int = field(default=0, init=False, repr=False)
    _feasibility_fitted_count: int = field(default=0, init=False, repr=False)

    def propose(
        self,
        adapter: KernelAdapter[ConfigT],
        history: SearchHistory[ConfigT],
        rng: random.Random,
        limit: int,
    ) -> list[Proposal[ConfigT]]:
        training_count = len(history.training_successful)
        if self.model.fitted and training_count >= self.refit_interval and (
            self._fitted_count == 0
            or training_count - self._fitted_count >= self.refit_interval
        ):
            self.model.fit(history.observations)  # type: ignore[arg-type]
            self._fitted_count = training_count
            self._queue.clear()
        if self.feasibility_model is not None:
            feasibility_count = self.feasibility_model.labeled_count(
                history.observations  # type: ignore[arg-type]
            )
            if (
                not self.feasibility_model.fitted
                or feasibility_count - self._feasibility_fitted_count
                >= self.refit_interval
            ):
                self.feasibility_model.fit(history.observations)  # type: ignore[arg-type]
                self._feasibility_fitted_count = feasibility_count
                self._queue.clear()
        if not self.model.fitted:
            return CoordinateLocalSearch[ConfigT](
                beam_width=self.beam_width, name=self.name
            ).propose(adapter, history, rng, limit)
        selected: list[Proposal[ConfigT]] = []
        while self._queue and len(selected) < limit:
            proposal = self._queue.pop(0)
            if adapter.config_id(proposal.config) not in history.seen_ids:
                selected.append(proposal)
        if len(selected) >= limit:
            return selected

        ranked = sorted(history.successful, key=lambda item: item.score)
        candidates: list[Proposal[ConfigT]] = []
        emitted: set[str] = set()
        for base in ranked[: self.beam_width]:
            for coordinate, value, candidate in adapter.neighbors(base.config):
                config_id = adapter.config_id(candidate)
                if (
                    config_id in history.seen_ids
                    or config_id in emitted
                    or adapter.rejection(candidate) is not None
                ):
                    continue
                emitted.add(config_id)
                candidates.append(
                    Proposal(
                        candidate,
                        self.name,
                        parent_config_id=base.config_id,
                        coordinate=coordinate,
                        coordinate_value=value,
                    )
                )
        if not candidates:
            return selected
        feature_rows = [adapter.features(item.config) for item in candidates]
        mean, uncertainty = self.model.predict(feature_rows)
        adjusted_mean = mean.copy()
        rule_effects = np.zeros(len(candidates), dtype=np.float64)
        rule_matches: list[list[int]] = [[] for _ in candidates]
        if self.rule_prior is not None and self.rule_weight != 0:
            for index, proposal in enumerate(candidates):
                effect, matches = self.rule_prior.adjustment(
                    feature_rows[index],
                    coordinate=proposal.coordinate,
                    coordinate_value=proposal.coordinate_value,
                )
                rule_effects[index] = effect
                rule_matches[index] = matches
            adjusted_mean = mean * np.exp(
                np.clip(self.rule_weight * rule_effects, -0.5, 0.5)
            )
        acquisition = adjusted_mean - self.exploration * uncertainty
        compile_probability = None
        compile_uncertainty = None
        if self.feasibility_model is not None and self.feasibility_model.fitted:
            compile_probability, compile_uncertainty = self.feasibility_model.predict(
                feature_rows
            )
            optimistic_probability = np.clip(
                compile_probability
                + self.feasibility_exploration * compile_uncertainty,
                self.minimum_optimistic_feasibility,
                1.0,
            )
            acquisition = (
                adjusted_mean / optimistic_probability
                - self.exploration * uncertainty
            )
        order = sorted(range(len(candidates)), key=lambda index: float(acquisition[index]))
        best_acquisition = float(acquisition[order[0]])
        for rank, index in enumerate(order):
            proposal = candidates[index]
            proposal.metadata.update(
                predicted_ms=float(mean[index]),
                prior_adjusted_ms=float(adjusted_mean[index]),
                predicted_std_ms=float(uncertainty[index]),
                acquisition=float(acquisition[index]),
                conditional_rule_effect=float(rule_effects[index]),
                conditional_rule_matches=rule_matches[index],
                candidate_pool_size=len(candidates),
                candidate_rank=rank,
                acquisition_gap_to_best=float(acquisition[index]) - best_acquisition,
                proposal_probability=None,
                proposal_probability_kind="deterministic_coordinate_rank",
            )
            if self.model_provenance is not None:
                proposal.metadata["pretrained"] = dict(self.model_provenance)
            if compile_probability is not None and compile_uncertainty is not None:
                proposal.metadata.update(
                    predicted_compile_probability=float(compile_probability[index]),
                    predicted_compile_uncertainty=float(compile_uncertainty[index]),
                )
            self._queue.append(proposal)
        while self._queue and len(selected) < limit:
            proposal = self._queue.pop(0)
            if adapter.config_id(proposal.config) not in history.seen_ids:
                selected.append(proposal)
        return selected

    def observe(self, observation: Observation[ConfigT]) -> None:
        self._observed += 1
        if self._observed % self.refresh_interval == 0:
            self._queue.clear()


@dataclass(slots=True)
class StrategyPipeline(Generic[ConfigT]):
    """A strategy itself composed of fixed-budget strategy stages."""

    stages: Sequence[tuple[SearchStrategy[ConfigT], int | None]]
    name: str = "pipeline"
    _observed: int = field(default=0, init=False, repr=False)

    def _active(self) -> SearchStrategy[ConfigT]:
        consumed = 0
        for strategy, budget in self.stages:
            if budget is None or self._observed < consumed + budget:
                return strategy
            consumed += budget
        return self.stages[-1][0]

    def propose(
        self,
        adapter: KernelAdapter[ConfigT],
        history: SearchHistory[ConfigT],
        rng: random.Random,
        limit: int,
    ) -> list[Proposal[ConfigT]]:
        strategy = self._active()
        proposals = strategy.propose(adapter, history, rng, limit)
        for proposal in proposals:
            proposal.metadata["pipeline"] = self.name
            proposal.metadata["pipeline_stage"] = strategy.name
        return proposals

    def observe(self, observation: Observation[ConfigT]) -> None:
        active = self._active()
        self._observed += 1
        active.observe(observation)


__all__ = [
    "CoordinateLocalSearch",
    "CostModelLocalSearch",
    "CostModelGuidedSearch",
    "RandomSearch",
    "SearchStrategy",
    "StrategyPipeline",
]
