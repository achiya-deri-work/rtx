"""Reusable multi-armed-bandit policies and sufficient statistics.

This module contains policy mathematics only. Kernel execution belongs in
``orchestrator.py`` and campaign lifecycle/persistence belongs in
``dataset.py``. Keeping those boundaries separate makes the policies usable by
other RTX kernel families without importing the dataset CLI.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Callable, Mapping, Sequence

from .core import Observation, SearchHistory


@dataclass(slots=True)
class ArmStatistics:
    """Persistent and discounted state for one search-strategy arm."""

    pulls: int = 0
    successes: int = 0
    reward_sum: float = 0.0
    elapsed_s: float = 0.0
    unavailable_until: int = 0
    effective_pulls: float = 0.0
    effective_reward_sum: float = 0.0
    prior_pulls: float = 0.0
    prior_reward_sum: float = 0.0
    last_reward: float = 0.0

    @property
    def mean_reward(self) -> float:
        return 0.0 if self.pulls == 0 else self.reward_sum / self.pulls

    @property
    def bandit_pulls(self) -> float:
        return self.effective_pulls + self.prior_pulls

    @property
    def bandit_mean_reward(self) -> float:
        pulls = self.bandit_pulls
        if pulls <= 0:
            return 0.0
        return (self.effective_reward_sum + self.prior_reward_sum) / pulls

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class DiscountedArmStatistics:
    """Generic nonstationary arm state used by campaign-level allocators."""

    pulls: int = 0
    successes: int = 0
    reward_sum: float = 0.0
    elapsed_s: float = 0.0
    effective_pulls: float = 0.0
    effective_reward_sum: float = 0.0

    @property
    def mean(self) -> float:
        if self.effective_pulls <= 0:
            return 0.0
        return self.effective_reward_sum / self.effective_pulls

    def decay(self, discount: float) -> None:
        self.effective_pulls *= discount
        self.effective_reward_sum *= discount

    def update(self, reward: float, elapsed_s: float, success: bool) -> None:
        self.pulls += 1
        self.successes += int(success)
        self.reward_sum += reward
        self.elapsed_s += elapsed_s
        self.effective_pulls += 1.0
        self.effective_reward_sum += reward

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class UCB1Scheduler:
    """Small stationary UCB1 baseline retained for controlled comparisons."""

    exploration: float = 1.25
    latency_aware: bool = True
    name: str = "ucb1_bandit"

    def select(
        self,
        names: Sequence[str],
        statistics: Mapping[str, ArmStatistics],
        trial_index: int,
    ) -> str | None:
        available = [
            name for name in names if statistics[name].unavailable_until <= trial_index
        ]
        if not available:
            return None
        for name in available:
            if statistics[name].pulls == 0:
                return name
        total = max(1, sum(statistics[name].pulls for name in available))

        def score(name: str) -> float:
            arm = statistics[name]
            exploitation = arm.mean_reward
            if self.latency_aware and arm.elapsed_s > 0:
                exploitation /= max(1e-6, arm.elapsed_s / arm.pulls)
            return exploitation + self.exploration * math.sqrt(
                math.log(total + 1) / arm.pulls
            )

        return max(available, key=score)


def _feature_number(observation: Observation, name: str) -> float | None:
    value = observation.features.get(name)
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def _observation_similarity(
    observation: Observation,
    active_features: Mapping[str, float],
) -> float:
    """Conservative workload/device similarity for contextual virtual pulls."""

    distance = 0.0
    dimensions = 0
    for key in ("context.workload.m", "context.workload.n", "context.workload.k"):
        left = _feature_number(observation, key)
        right = active_features.get(key)
        if left is None or right is None or left <= 0 or right <= 0:
            continue
        distance += abs(math.log2(left / float(right)))
        dimensions += 1
    shape_similarity = math.exp(-distance / max(1, dimensions))

    regime_keys = {
        key
        for key, value in observation.features.items()
        if key.startswith("context.regime=") and value > 0
    }
    active_regime_keys = {
        key
        for key, value in active_features.items()
        if key.startswith("context.regime=") and value > 0
    }
    regime_similarity = (
        0.45
        if regime_keys and active_regime_keys and regime_keys != active_regime_keys
        else 1.0
    )

    sm_similarity = 1.0
    sm_key = "context.device.multiprocessor_count"
    left_sm = _feature_number(observation, sm_key)
    right_sm = active_features.get(sm_key)
    if left_sm is not None and right_sm is not None and left_sm > 0 and right_sm > 0:
        sm_similarity = math.exp(-abs(math.log2(left_sm / float(right_sm))))
    return max(0.02, shape_similarity * regime_similarity * sm_similarity)


@dataclass(slots=True)
class AdaptiveBanditScheduler:
    """Discounted, cost-aware contextual UCB over search strategies.

    Current-context observations are replayed on resume. Observations from
    similar contexts contribute only capped virtual pulls, so transfer can
    break ties but cannot overwhelm measurements from the active context.
    """

    exploration: float = 0.35
    discount: float = 0.985
    warmup_trials: int = 32
    warmup_arm: str = "random"
    transfer_prior_strength: float = 3.0
    cost_scale_s: float = 2.0
    name: str = "adaptive_contextual_bandit"

    def __post_init__(self) -> None:
        if self.exploration < 0:
            raise ValueError("bandit exploration cannot be negative")
        if not 0 < self.discount <= 1:
            raise ValueError("bandit discount must be in (0, 1]")
        if self.warmup_trials < 0 or self.transfer_prior_strength < 0:
            raise ValueError("bandit warmup/prior strength cannot be negative")
        if self.cost_scale_s <= 0:
            raise ValueError("bandit cost scale must be positive")

    def reward(self, before: float, observation: Observation) -> float:
        cost = 0.04 * math.tanh(max(0.0, observation.elapsed_s) / self.cost_scale_s)
        if not observation.successful:
            return -0.20 - cost
        improvement = 0.0
        if math.isfinite(before) and observation.score > 0:
            improvement = math.tanh(max(0.0, math.log(before / observation.score)))
        uncertainty = observation.metadata.get("predicted_std_ms", 0.0)
        predicted = observation.metadata.get("predicted_ms", 0.0)
        try:
            information = float(uncertainty) / max(float(predicted), 1e-9)
        except (TypeError, ValueError):
            information = 0.0
        return 0.015 + improvement + 0.015 * math.tanh(max(0.0, information)) - cost

    def _decay(self, statistics: Mapping[str, ArmStatistics]) -> None:
        for arm in statistics.values():
            arm.effective_pulls *= self.discount
            arm.effective_reward_sum *= self.discount

    def update(
        self,
        statistics: Mapping[str, ArmStatistics],
        arm_name: str,
        reward: float,
        observation: Observation,
    ) -> None:
        self._decay(statistics)
        arm = statistics[arm_name]
        arm.pulls += 1
        arm.successes += int(observation.successful)
        arm.elapsed_s += observation.elapsed_s
        arm.reward_sum += reward
        arm.effective_pulls += 1.0
        arm.effective_reward_sum += reward
        arm.last_reward = reward

    def initialize(
        self,
        statistics: Mapping[str, ArmStatistics],
        history: SearchHistory,
        active_features: Mapping[str, float],
    ) -> None:
        """Rebuild exact active state and contextual priors from JSONL history."""

        incumbents: dict[str, float] = {}
        transfer_rewards: dict[str, list[tuple[float, float]]] = {
            name: [] for name in statistics
        }
        ordered = sorted(
            history.observations,
            key=lambda item: (item.finished_at, item.session_id, item.sequence),
        )
        for observation in ordered:
            before = incumbents.get(observation.context_id, math.inf)
            if observation.successful:
                incumbents[observation.context_id] = min(before, observation.score)
            if observation.strategy not in statistics:
                continue
            reward = self.reward(before, observation)
            if observation.context_id == history.active_context_id:
                self.update(statistics, observation.strategy, reward, observation)
            else:
                similarity = _observation_similarity(observation, active_features)
                transfer_rewards[observation.strategy].append((similarity, reward))

        for name, values in transfer_rewards.items():
            if not values or self.transfer_prior_strength <= 0:
                continue
            weight = sum(item[0] for item in values)
            if weight <= 0:
                continue
            mean = sum(similarity * reward for similarity, reward in values) / weight
            pulls = self.transfer_prior_strength * weight / (weight + 8.0)
            statistics[name].prior_pulls = pulls
            statistics[name].prior_reward_sum = pulls * mean

    def scores(
        self,
        names: Sequence[str],
        statistics: Mapping[str, ArmStatistics],
        trial_index: int,
    ) -> dict[str, float | None]:
        available = [
            name for name in names if statistics[name].unavailable_until <= trial_index
        ]
        if not available:
            return {name: None for name in names}
        total = max(1.0, sum(statistics[name].bandit_pulls for name in available))
        result: dict[str, float | None] = {name: None for name in names}
        for name in available:
            arm = statistics[name]
            pulls = arm.bandit_pulls
            if pulls <= 0:
                result[name] = math.inf
            else:
                result[name] = arm.bandit_mean_reward + self.exploration * math.sqrt(
                    math.log(total + 1.0) / pulls
                )
        return result

    def select(
        self,
        names: Sequence[str],
        statistics: Mapping[str, ArmStatistics],
        trial_index: int,
    ) -> str | None:
        available = [
            name for name in names if statistics[name].unavailable_until <= trial_index
        ]
        if not available:
            return None
        if trial_index < self.warmup_trials and self.warmup_arm in available:
            return self.warmup_arm
        for name in available:
            if statistics[name].pulls == 0:
                return name
        scores = self.scores(names, statistics, trial_index)
        return max(available, key=lambda name: float(scores[name]))

    def snapshot(
        self,
        names: Sequence[str],
        statistics: Mapping[str, ArmStatistics],
        trial_index: int,
    ) -> dict[str, object]:
        raw_scores = self.scores(names, statistics, trial_index)
        return {
            "scheduler": self.name,
            "trial": trial_index,
            "discount": self.discount,
            "exploration": self.exploration,
            "scores": {
                name: None if value is None or not math.isfinite(value) else value
                for name, value in raw_scores.items()
            },
            "arms": {name: statistics[name].as_dict() for name in names},
        }


def contextual_ucb_scores(
    keys: Sequence[str],
    statistics_by_key: Mapping[str, DiscountedArmStatistics],
    similarity: Callable[[str, str], float],
    exploration: float,
    *,
    prior_cap: float = 2.0,
    prior_scale: float = 4.0,
) -> dict[str, float]:
    """Score arbitrary arms using discounted evidence and neighbor priors."""

    total = max(
        1.0,
        sum(statistics_by_key[key].effective_pulls for key in keys),
    )
    result: dict[str, float] = {}
    for key in keys:
        arm = statistics_by_key[key]
        if arm.pulls == 0:
            result[key] = math.inf
            continue
        neighbor_weight = 0.0
        neighbor_reward = 0.0
        for other in keys:
            if other == key:
                continue
            peer = statistics_by_key[other]
            if peer.effective_pulls <= 0:
                continue
            weight = max(0.0, similarity(key, other)) * min(1.0, peer.effective_pulls)
            neighbor_weight += weight
            neighbor_reward += weight * peer.mean
        prior_pulls = min(prior_cap, neighbor_weight / max(prior_scale, 1e-9))
        prior_reward = (
            0.0
            if neighbor_weight <= 0
            else prior_pulls * neighbor_reward / neighbor_weight
        )
        pulls = arm.effective_pulls + prior_pulls
        mean = (arm.effective_reward_sum + prior_reward) / max(pulls, 1e-9)
        result[key] = mean + exploration * math.sqrt(
            math.log(total + 1.0) / max(pulls, 1e-9)
        )
    return result


__all__ = [
    "AdaptiveBanditScheduler",
    "ArmStatistics",
    "DiscountedArmStatistics",
    "UCB1Scheduler",
    "contextual_ucb_scores",
]
