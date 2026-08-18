"""Composable candidate generators: exploration, learned ranking, and local search."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
import time
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


class PairwisePredictor(Protocol):
    def predict(
        self,
        left: Sequence[Mapping[str, float]],
        right: Sequence[Mapping[str, float]],
    ) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass(slots=True)
class SharedModelFitState:
    """One fit clock shared by every strategy consuming the same models."""

    cost_model: CostModel
    feasibility_model: GradientBoostedFeasibilityModel | None = None
    cost_fitted_count: int = 0
    feasibility_fitted_count: int = 0
    cost_fit_elapsed_s: float = 0.0
    feasibility_fit_elapsed_s: float = 0.0
    cost_fit_count: int = 0
    feasibility_fit_count: int = 0
    recommended_pool_size: int = 0
    last_pool_size: int = 0
    last_pool_elapsed_s: float = 0.0

    @staticmethod
    def refit_due(
        current_count: int,
        fitted_count: int,
        minimum_interval: int,
        growth_fraction: float,
    ) -> bool:
        if fitted_count <= 0:
            return True
        interval = max(
            minimum_interval,
            int(math.ceil(fitted_count * growth_fraction)),
        )
        return current_count - fitted_count >= interval


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
class PairwiseModelGuidedSearch(Generic[ConfigT]):
    """Rank a fresh legal pool by probability of beating the incumbent."""

    model: PairwisePredictor
    pool_size: int = 1024
    exploration: float = 0.1
    model_provenance: Mapping[str, object] | None = None
    name: str = "pairwise_pretrained"
    _queue: list[Proposal[ConfigT]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.pool_size <= 0 or self.exploration < 0:
            raise ValueError("invalid pairwise search policy")

    def propose(
        self,
        adapter: KernelAdapter[ConfigT],
        history: SearchHistory[ConfigT],
        rng: random.Random,
        limit: int,
    ) -> list[Proposal[ConfigT]]:
        selected: list[Proposal[ConfigT]] = []
        while self._queue and len(selected) < limit:
            proposal = self._queue.pop(0)
            if adapter.config_id(proposal.config) not in history.seen_ids:
                selected.append(proposal)
        if len(selected) >= limit:
            return selected
        incumbent = history.best
        reference = (
            adapter.initial_config if incumbent is None else incumbent.config
        )
        seeds = [
            item.config
            for item in sorted(history.successful, key=lambda item: item.score)[:8]
        ]
        candidates: dict[str, ConfigT] = {}
        for candidate in adapter.sample(rng, self.pool_size, seeds):
            config_id = adapter.config_id(candidate)
            if (
                config_id in history.seen_ids
                or config_id in candidates
                or adapter.rejection(candidate) is not None
            ):
                continue
            candidates[config_id] = candidate
        if not candidates:
            return selected
        configs = list(candidates.values())
        feature_rows = [adapter.features(config) for config in configs]
        reference_features = adapter.features(reference)
        probability, uncertainty = self.model.predict(
            feature_rows,
            [reference_features] * len(feature_rows),
        )
        acquisition = probability + self.exploration * uncertainty
        order = sorted(
            range(len(configs)),
            key=lambda index: float(acquisition[index]),
            reverse=True,
        )
        for rank, index in enumerate(order[: max(64, limit * 8)]):
            metadata: dict[str, object] = {
                "predicted_probability_beats_incumbent": float(probability[index]),
                "predicted_pairwise_uncertainty": float(uncertainty[index]),
                "pairwise_acquisition": float(acquisition[index]),
                "candidate_pool_size": len(configs),
                "candidate_rank": rank,
                "reference_config_id": adapter.config_id(reference),
                "proposal_probability": None,
                "proposal_probability_kind": "frozen_pairwise_rank",
            }
            if self.model_provenance is not None:
                metadata["pairwise_pretrained"] = dict(self.model_provenance)
            self._queue.append(
                Proposal(configs[index], self.name, metadata=metadata)
            )
        while self._queue and len(selected) < limit:
            proposal = self._queue.pop(0)
            if adapter.config_id(proposal.config) not in history.seen_ids:
                selected.append(proposal)
        return selected

    def observe(self, observation: Observation[ConfigT]) -> None:
        # Refresh the reference after every measurement; queued ranks are
        # relative to the previous incumbent.
        self._queue.clear()


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
    refit_growth_fraction: float = 0.05
    exploration: float = 0.15
    feasibility_exploration: float = 0.5
    minimum_optimistic_feasibility: float = 0.05
    include_local_neighbors: int = 4
    local_candidate_cap: int = 512
    initial_pool_cap: int = 2048
    proposal_budget_s: float = 1.0
    model_provenance: Mapping[str, object] | None = None
    name: str = "gradient_boosted"
    _fitted_count: int = field(default=0, init=False, repr=False)
    _feasibility_fitted_count: int = field(default=0, init=False, repr=False)
    _queue: list[Proposal[ConfigT]] = field(default_factory=list, init=False, repr=False)
    fit_state: SharedModelFitState | None = None

    def __post_init__(self) -> None:
        if (
            self.refit_interval <= 0
            or not 0 <= self.refit_growth_fraction <= 1
            or self.initial_pool_cap <= 0
            or self.proposal_budget_s <= 0
            or self.local_candidate_cap < 0
        ):
            raise ValueError("invalid model refit policy")
        if self.fit_state is None:
            self.fit_state = SharedModelFitState(self.model, self.feasibility_model)
        elif (
            self.fit_state.cost_model is not self.model
            or self.fit_state.feasibility_model is not self.feasibility_model
        ):
            raise ValueError("shared fit state must own the strategy models")

    def propose(
        self,
        adapter: KernelAdapter[ConfigT],
        history: SearchHistory[ConfigT],
        rng: random.Random,
        limit: int,
    ) -> list[Proposal[ConfigT]]:
        current_successes = history.successful
        training_successes = history.training_successful
        assert self.fit_state is not None
        fit_state = self.fit_state
        if len(training_successes) < self.min_observations and not self.model.fitted:
            proposals = self.warmup.propose(adapter, history, rng, limit)
            for proposal in proposals:
                proposal.strategy = self.name
                proposal.metadata["phase"] = "warmup"
            return proposals
        if self.model.fitted and fit_state.cost_fitted_count == 0:
            # A loaded pretrained model already represents the available prior
            # evidence; do not immediately overwrite it on its first proposal.
            fit_state.cost_fitted_count = len(training_successes)
        if fit_state.cost_fitted_count == 0 or fit_state.refit_due(
            len(training_successes),
            fit_state.cost_fitted_count,
            self.refit_interval,
            self.refit_growth_fraction,
        ):
            fit_started = time.monotonic()
            self.model.fit(history.observations)  # type: ignore[arg-type]
            fit_state.cost_fit_elapsed_s = time.monotonic() - fit_started
            fit_state.cost_fit_count += 1
            fit_state.cost_fitted_count = len(training_successes)
            self._fitted_count = fit_state.cost_fitted_count
            self._queue.clear()
        feasibility_count = self.feasibility_model.labeled_count(
            history.observations  # type: ignore[arg-type]
        )
        if (
            self.feasibility_model.fitted
            and fit_state.feasibility_fitted_count == 0
        ):
            fit_state.feasibility_fitted_count = feasibility_count
        if fit_state.feasibility_fitted_count == 0 or fit_state.refit_due(
            feasibility_count,
            fit_state.feasibility_fitted_count,
            self.refit_interval,
            self.refit_growth_fraction,
        ):
            fit_started = time.monotonic()
            self.feasibility_model.fit(history.observations)  # type: ignore[arg-type]
            fit_state.feasibility_fit_elapsed_s = time.monotonic() - fit_started
            fit_state.feasibility_fit_count += 1
            fit_state.feasibility_fitted_count = feasibility_count
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
        effective_pool_size = min(
            self.pool_size,
            fit_state.recommended_pool_size
            or min(self.pool_size, self.initial_pool_cap),
        )
        pool_started = time.monotonic()
        candidate_by_id: dict[str, ConfigT] = {}
        for candidate in adapter.sample(rng, effective_pool_size, seeds):
            if adapter.rejection(candidate) is not None:
                continue
            candidate_by_id.setdefault(adapter.config_id(candidate), candidate)
        # Full coordinate neighborhoods can dwarf the nominal random pool in
        # coupled kernel spaces (7K+ backward candidates in practice). Keep an
        # unbiased bounded reservoir so the adaptive pool is a real wall-time
        # control rather than a misleading lower bound.
        local_candidates: list[tuple[str, ConfigT]] = []
        local_seen: set[str] = set()
        local_population = 0
        if self.local_candidate_cap:
            for base in sorted(current_successes, key=lambda item: item.score)[
                : self.include_local_neighbors
            ]:
                for _coordinate, _value, candidate in adapter.neighbors(base.config):
                    config_id = adapter.config_id(candidate)
                    if (
                        config_id in candidate_by_id
                        or config_id in local_seen
                        or config_id in history.seen_ids
                        or adapter.rejection(candidate) is not None
                    ):
                        continue
                    local_seen.add(config_id)
                    local_population += 1
                    if len(local_candidates) < self.local_candidate_cap:
                        local_candidates.append((config_id, candidate))
                        continue
                    slot = rng.randrange(local_population)
                    if slot < self.local_candidate_cap:
                        local_candidates[slot] = (config_id, candidate)
        for config_id, candidate in local_candidates:
            candidate_by_id[config_id] = candidate
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
        pool_elapsed_s = time.monotonic() - pool_started
        fit_state.last_pool_size = effective_pool_size
        fit_state.last_pool_elapsed_s = pool_elapsed_s
        scaled_pool = int(
            effective_pool_size
            * self.proposal_budget_s
            / max(pool_elapsed_s, 1.0e-6)
        )
        # Move gradually when under budget, but react immediately to a severe
        # overrun. The configured pool remains the hard quality ceiling.
        if pool_elapsed_s <= self.proposal_budget_s:
            scaled_pool = min(effective_pool_size * 2, scaled_pool)
        fit_state.recommended_pool_size = min(
            self.pool_size,
            max(min(256, self.pool_size), scaled_pool),
        )
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
                "model_fit_count": fit_state.cost_fit_count,
                "model_fit_elapsed_s": fit_state.cost_fit_elapsed_s,
                "model_fitted_observations": fit_state.cost_fitted_count,
                "feasibility_fit_count": fit_state.feasibility_fit_count,
                "feasibility_fit_elapsed_s": fit_state.feasibility_fit_elapsed_s,
                "candidate_pool_size": len(candidates),
                "local_candidate_cap": self.local_candidate_cap,
                "local_candidate_population": local_population,
                "local_candidates_selected": len(local_candidates),
                "requested_candidate_pool_size": self.pool_size,
                "effective_candidate_pool_size": effective_pool_size,
                "candidate_pool_elapsed_s": pool_elapsed_s,
                "next_candidate_pool_size": fit_state.recommended_pool_size,
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
    refit_growth_fraction: float = 0.05
    candidate_cap: int = 1024
    name: str = "model_local"
    _queue: list[Proposal[ConfigT]] = field(default_factory=list, init=False, repr=False)
    _observed: int = field(default=0, init=False, repr=False)
    _fitted_count: int = field(default=0, init=False, repr=False)
    _feasibility_fitted_count: int = field(default=0, init=False, repr=False)
    fit_state: SharedModelFitState | None = None

    def __post_init__(self) -> None:
        if (
            self.refit_interval <= 0
            or not 0 <= self.refit_growth_fraction <= 1
            or self.candidate_cap <= 0
        ):
            raise ValueError("invalid local-model refit policy")
        if self.fit_state is None:
            self.fit_state = SharedModelFitState(self.model, self.feasibility_model)
        elif (
            self.fit_state.cost_model is not self.model
            or self.fit_state.feasibility_model is not self.feasibility_model
        ):
            raise ValueError("shared fit state must own the local strategy models")

    def propose(
        self,
        adapter: KernelAdapter[ConfigT],
        history: SearchHistory[ConfigT],
        rng: random.Random,
        limit: int,
    ) -> list[Proposal[ConfigT]]:
        training_count = len(history.training_successful)
        assert self.fit_state is not None
        fit_state = self.fit_state
        if self.model.fitted and fit_state.cost_fitted_count == 0:
            fit_state.cost_fitted_count = training_count
        if self.model.fitted and fit_state.refit_due(
            training_count,
            fit_state.cost_fitted_count,
            self.refit_interval,
            self.refit_growth_fraction,
        ):
            fit_started = time.monotonic()
            self.model.fit(history.observations)  # type: ignore[arg-type]
            fit_state.cost_fit_elapsed_s = time.monotonic() - fit_started
            fit_state.cost_fit_count += 1
            fit_state.cost_fitted_count = training_count
            self._fitted_count = training_count
            self._queue.clear()
        if self.feasibility_model is not None:
            feasibility_count = self.feasibility_model.labeled_count(
                history.observations  # type: ignore[arg-type]
            )
            if (
                self.feasibility_model.fitted
                and fit_state.feasibility_fitted_count == 0
            ):
                fit_state.feasibility_fitted_count = feasibility_count
            if fit_state.feasibility_fitted_count == 0 or fit_state.refit_due(
                feasibility_count,
                fit_state.feasibility_fitted_count,
                self.refit_interval,
                self.refit_growth_fraction,
            ):
                fit_started = time.monotonic()
                self.feasibility_model.fit(history.observations)  # type: ignore[arg-type]
                fit_state.feasibility_fit_elapsed_s = time.monotonic() - fit_started
                fit_state.feasibility_fit_count += 1
                fit_state.feasibility_fitted_count = feasibility_count
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
        candidate_population = 0
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
                candidate_population += 1
                proposal = Proposal(
                    candidate,
                    self.name,
                    parent_config_id=base.config_id,
                    coordinate=coordinate,
                    coordinate_value=value,
                )
                if len(candidates) < self.candidate_cap:
                    candidates.append(proposal)
                    continue
                slot = rng.randrange(candidate_population)
                if slot < self.candidate_cap:
                    candidates[slot] = proposal
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
                candidate_pool_cap=self.candidate_cap,
                candidate_pool_population=candidate_population,
                candidate_rank=rank,
                model_fit_count=fit_state.cost_fit_count,
                model_fit_elapsed_s=fit_state.cost_fit_elapsed_s,
                model_fitted_observations=fit_state.cost_fitted_count,
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
    "PairwiseModelGuidedSearch",
    "RandomSearch",
    "SearchStrategy",
    "SharedModelFitState",
    "StrategyPipeline",
]
