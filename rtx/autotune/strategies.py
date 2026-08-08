"""Composable candidate generators: exploration, learned ranking, and local search."""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Generic, Protocol, Sequence

from .core import ConfigT, KernelAdapter, Observation, Proposal, SearchHistory
from .cost_model import GradientBoostedCostModel


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
    name: str = "random"

    def propose(
        self,
        adapter: KernelAdapter[ConfigT],
        history: SearchHistory[ConfigT],
        rng: random.Random,
        limit: int,
    ) -> list[Proposal[ConfigT]]:
        seeds = [item.config for item in sorted(history.successful, key=lambda item: item.score)[:8]]
        candidates = adapter.sample(rng, max(limit * self.pool_multiplier, limit), seeds)
        proposals: list[Proposal[ConfigT]] = []
        for candidate in candidates:
            if adapter.config_id(candidate) in history.seen_ids:
                continue
            if adapter.rejection(candidate) is not None:
                continue
            proposals.append(Proposal(candidate, self.name))
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

    model: GradientBoostedCostModel = field(default_factory=GradientBoostedCostModel)
    warmup: SearchStrategy[ConfigT] = field(default_factory=RandomSearch)
    min_observations: int = 16
    pool_size: int = 2048
    refit_interval: int = 8
    exploration: float = 0.15
    include_local_neighbors: int = 4
    name: str = "gradient_boosted"
    _fitted_count: int = field(default=0, init=False, repr=False)
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
        if len(training_successes) < self.min_observations:
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
        mean, uncertainty = self.model.predict([adapter.features(item) for item in candidates])
        acquisition = mean - self.exploration * uncertainty
        order = sorted(range(len(candidates)), key=lambda index: float(acquisition[index]))
        queue_limit = min(len(order), max(256, self.refit_interval * 8))
        for index in order[:queue_limit]:
            self._queue.append(
                Proposal(
                    candidates[index],
                    self.name,
                    metadata={
                        "predicted_ms": float(mean[index]),
                        "predicted_std_ms": float(uncertainty[index]),
                        "acquisition": float(acquisition[index]),
                        "training_successes": len(training_successes),
                        "current_successes": len(current_successes),
                        "model_parameters": self.model.parameter_count,
                    },
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

    model: GradientBoostedCostModel
    beam_width: int = 3
    exploration: float = 0.05
    refresh_interval: int = 8
    refit_interval: int = 32
    name: str = "model_local"
    _queue: list[Proposal[ConfigT]] = field(default_factory=list, init=False, repr=False)
    _observed: int = field(default=0, init=False, repr=False)
    _fitted_count: int = field(default=0, init=False, repr=False)

    def propose(
        self,
        adapter: KernelAdapter[ConfigT],
        history: SearchHistory[ConfigT],
        rng: random.Random,
        limit: int,
    ) -> list[Proposal[ConfigT]]:
        training_count = len(history.training_successful)
        if self.model.fitted and (
            self._fitted_count == 0
            or training_count - self._fitted_count >= self.refit_interval
        ):
            self.model.fit(history.observations)  # type: ignore[arg-type]
            self._fitted_count = training_count
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
        mean, uncertainty = self.model.predict(
            [adapter.features(item.config) for item in candidates]
        )
        acquisition = mean - self.exploration * uncertainty
        order = sorted(range(len(candidates)), key=lambda index: float(acquisition[index]))
        for index in order:
            proposal = candidates[index]
            proposal.metadata.update(
                predicted_ms=float(mean[index]),
                predicted_std_ms=float(uncertainty[index]),
                acquisition=float(acquisition[index]),
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
