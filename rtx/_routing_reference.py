"""Fixed-K policy reference extracted from the BiMoE handoff.

Source: routing-research-port-20261111/bidirectional_mixture_of_experts/router.py
SHA256: f4872cd1f3df5b320bedb526e666415c8b59130560b080d84195ac43d35fada7
Only the regular no-fallback pipeline is included; no expert dependency.
This is a transparent Torch reference, not a claim of handwritten CUDA kernels.
Selected-softmax backward centers upstreams to avoid FP32 cancellation.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
import torch
import torch.nn.functional as F


def _rms_normalize(x: torch.Tensor, epsilon: float) -> torch.Tensor:
    value = x.float()
    return value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + epsilon)


def sigmoid_router_log_affinities(
    x: torch.Tensor,
    projection_weight: torch.Tensor,
    expert_tokens: torch.Tensor,
    logit_scale: torch.Tensor,
    score_bias: torch.Tensor,
    *,
    rms_epsilon: float,
    _affine=None,
) -> torch.Tensor:
    """Compute stable FP32 log-sigmoid affinities."""
    if x.ndim != 2:
        raise ValueError("router input must have shape [tokens, d_model]")
    if projection_weight.ndim != 2:
        raise ValueError("projection_weight must have shape [router_dim, d_model]")
    if expert_tokens.ndim != 2:
        raise ValueError("expert_tokens must have shape [experts, router_dim]")
    if x.shape[1] != projection_weight.shape[1]:
        raise ValueError("router input and projection dimensions must match")
    if projection_weight.shape[0] != expert_tokens.shape[1]:
        raise ValueError("projection and expert-token dimensions must match")
    if logit_scale.ndim != 0 or score_bias.ndim != 0:
        raise ValueError("logit_scale and score_bias must be scalar tensors")
    if rms_epsilon <= 0.0:
        raise ValueError("rms_epsilon must be positive")

    token_latents = F.linear(x.float(), projection_weight.float())
    normalized_tokens = _rms_normalize(token_latents, rms_epsilon)
    normalized_experts = _rms_normalize(expert_tokens, rms_epsilon)
    logits = F.linear(normalized_tokens, normalized_experts)
    if _affine is not None:
        return _affine(logits, logit_scale, score_bias)
    logits = logits * logit_scale.float().exp() + score_bias.float()
    return F.logsigmoid(logits)


class _SelectedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values):
        probabilities = values.softmax(dim=-1)
        ctx.save_for_backward(probabilities)
        return probabilities

    @staticmethod
    def backward(ctx, gradient):
        (probabilities,) = ctx.saved_tensors
        # J * 1 = 0. Remove the common component before reduction rather
        # than subtracting two nearly equal O(1) values after reduction.
        # All operations remain visible to AOTAutograd/Inductor.
        centered = gradient - gradient[..., :1]
        return probabilities * (centered - (probabilities * centered).sum(-1, keepdim=True))


def selected_softmax(values):
    """Selected-expert softmax with a cancellation-resistant saved adjoint."""
    return _SelectedSoftmax.apply(values)


def padded_token_count(num_tokens: int, num_experts: int, top_k: int) -> int:
    """Smallest token count whose fixed top-k assignments divide over experts."""
    if num_tokens <= 0 or num_experts <= 0 or top_k <= 0:
        raise ValueError("tokens, experts, and top_k must all be positive")
    if top_k > num_experts:
        raise ValueError("top_k must not exceed num_experts")
    token_multiple = num_experts // math.gcd(num_experts, top_k)
    return ((num_tokens + token_multiple - 1) // token_multiple) * token_multiple


def uniform_capacity(num_tokens: int, num_experts: int, top_k: int) -> int:
    """Derive exact per-expert capacity from a divisible routing shape."""
    if num_tokens <= 0 or num_experts <= 0 or top_k <= 0:
        raise ValueError("tokens, experts, and top_k must all be positive")
    if top_k > num_experts:
        raise ValueError("top_k must not exceed num_experts")
    total_assignments = num_tokens * top_k
    if total_assignments % num_experts:
        raise ValueError("tokens * top_k must be divisible by num_experts")
    return total_assignments // num_experts


def _deterministic_tie_break(
    tokens: int,
    experts: int,
    *,
    device: torch.device,
    magnitude: float,
) -> torch.Tensor:
    """Return a token-specific perturbation used only to order numerical ties."""
    token_ids = torch.arange(tokens, device=device, dtype=torch.int64)
    expert_ids = torch.arange(experts, device=device, dtype=torch.int64)
    # Reduce coefficients and partial products before combining them. The
    # equivalent original polynomial lets Inductor fold block offsets into
    # constants larger than int32 even though the final hash is small.
    modulus = 104_729
    token_term = (token_ids[:, None] * (1_103_515_245 % modulus)) % modulus
    expert_term = (expert_ids[None, :] * 12_345) % modulus
    cross_term = ((token_ids[:, None] * (1_013_904_223 % modulus)) % modulus
                  * expert_ids[None, :]) % modulus
    tie_hash = (token_term + expert_term + cross_term) % modulus
    return tie_hash.float() * (magnitude / 104_729.0)


def log_sinkhorn(
    logits: torch.Tensor,
    *,
    top_k: int,
    iterations: int = 5,
    epsilon_schedule: Sequence[float] | None = None,
) -> torch.Tensor:
    """Return an FP32 balanced transport in log space.

    Rows have mass ``top_k`` and columns have mass ``capacity`` up to the
    requested finite iteration error. Exact integer balance is supplied by
    the detached hard scheduler rather than by discretizing this tensor directly.
    """
    if logits.ndim != 2:
        raise ValueError("router logits must have shape [tokens, experts]")
    if iterations <= 0:
        raise ValueError("Sinkhorn iterations must be positive")
    tokens, experts = logits.shape
    capacity = uniform_capacity(tokens, experts, top_k)

    if epsilon_schedule is None:
        epsilons = (1.0,) * iterations
    else:
        epsilons = tuple(float(epsilon) for epsilon in epsilon_schedule)
        if len(epsilons) != iterations:
            raise ValueError("epsilon_schedule must contain one value per iteration")
        if any(epsilon <= 0.0 for epsilon in epsilons):
            raise ValueError("every Sinkhorn epsilon must be positive")

    scores = logits.float()
    log_row_mass = math.log(float(top_k))
    log_column_mass = math.log(float(capacity))
    row_potential = torch.zeros(
        tokens, 1, device=scores.device, dtype=torch.float32
    )
    column_potential = torch.zeros(
        1, experts, device=scores.device, dtype=torch.float32
    )
    for epsilon in epsilons:
        row_potential = epsilon * (
            log_row_mass
            - torch.logsumexp(
                (scores + column_potential) / epsilon,
                dim=1,
                keepdim=True,
            )
        )
        column_potential = epsilon * (
            log_column_mass
            - torch.logsumexp(
                (scores + row_potential) / epsilon,
                dim=0,
                keepdim=True,
            )
        )
    final_epsilon = epsilons[-1]
    return (scores + row_potential + column_potential) / final_epsilon


def dual_price_assignment(
    scores: torch.Tensor,
    *,
    top_k: int,
    iterations: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a hard candidate schedule with alternating exact dual updates.

    This is the vectorized Quantile-Balancing coordinate solver applied to an
    already balanced soft transport. It is intentionally non-differentiable;
    gradients flow through the selected Sinkhorn weights, not the schedule.
    """
    if scores.ndim != 2:
        raise ValueError("dual scores must have shape [tokens, experts]")
    if iterations <= 0:
        raise ValueError("dual iterations must be positive")
    tokens, experts = scores.shape
    capacity = uniform_capacity(tokens, experts, top_k)
    if top_k == experts:
        assignments = torch.arange(experts, device=scores.device).expand(tokens, -1)
        return assignments, torch.zeros(experts, device=scores.device)

    detached_scores = scores.detach().float()
    detached_scores = detached_scores + _deterministic_tie_break(
        tokens,
        experts,
        device=scores.device,
        magnitude=1.0e-7,
    )
    expert_prices = torch.zeros(
        experts, device=scores.device, dtype=torch.float32
    )
    for _ in range(iterations):
        token_margins = detached_scores - expert_prices[None, :]
        token_cutoffs = torch.topk(token_margins, top_k + 1, dim=1).values[
            :, top_k : top_k + 1
        ]
        expert_margins = detached_scores - token_cutoffs
        if capacity == tokens:
            expert_prices = expert_margins.amin(dim=0)
        else:
            expert_prices = torch.topk(
                expert_margins, capacity + 1, dim=0
            ).values[capacity]
    assignments = torch.topk(
        detached_scores - expert_prices[None, :], top_k, dim=1
    ).indices
    return assignments, expert_prices


def assignment_loads(assignments: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Count hard assignments without a CPU synchronization."""
    return torch.zeros(
        num_experts, device=assignments.device, dtype=torch.int64
    ).scatter_add(
        0, assignments.flatten(), torch.ones_like(assignments.flatten())
    )


def _quota_accept(
    groups: torch.Tensor,
    priorities: torch.Tensor,
    quotas: torch.Tensor,
) -> torch.Tensor:
    """Accept the highest-priority proposals within each group's exact quota.

    Stable lexicographic ordering gives every proposal a unique rank, including
    exact score ties, without constructing an experts-by-proposals matrix.
    """
    positions = torch.arange(groups.numel(), device=groups.device)
    order = torch.argsort(priorities, descending=True, stable=True)
    order = order[torch.argsort(groups[order], stable=True)]
    ordered_groups = groups[order]
    first_in_group = torch.cat(
        [
            torch.ones(1, device=groups.device, dtype=torch.bool),
            ordered_groups[1:] != ordered_groups[:-1],
        ]
    )
    starts = torch.where(first_in_group, positions, 0).cummax(dim=0).values
    ranks = positions - starts
    accepted = (ranks < quotas[ordered_groups]) & torch.isfinite(priorities[order])
    return torch.zeros_like(groups, dtype=torch.bool).scatter(0, order, accepted)


def repair_balanced_assignment(
    scores: torch.Tensor,
    assignments: torch.Tensor,
    *,
    rounds: int = 64,
) -> torch.Tensor:
    """Repair residual loads with distinct parallel proposals and exact quotas.

    Each token proposes up to K distinct underloaded targets, paired with its
    lowest-scoring overloaded source slots. After 16 greedy rounds, rotating
    token-specific priorities diversify proposals within the best remaining
    max(2*K, 16) targets to avoid persistent contention. Source and target quotas bound
    accepted moves. Every accepted move removes one unit of surplus without
    creating a duplicate expert within a token. The bounded device-side loop
    uses a fixed CUDA round budget; convergence is empirical, not assumed.
    This helper returns residual assignments for diagnostics if the budget ends.
    """
    if scores.ndim != 2 or assignments.ndim != 2:
        raise ValueError("repair inputs must be rank-two tensors")
    tokens, experts = scores.shape
    assignment_tokens, top_k = assignments.shape
    if assignment_tokens != tokens:
        raise ValueError("scores and assignments must have the same token count")
    capacity = uniform_capacity(tokens, experts, top_k)
    if rounds <= 0:
        raise ValueError("repair rounds must be positive")

    detached_scores = scores.detach().float()
    repaired = assignments.detach().clone()
    tie_break = _deterministic_tie_break(
        tokens, experts, device=scores.device, magnitude=1.0e-6
    )
    initial_loads = assignment_loads(repaired, experts)
    proposal_pool = min(experts, max(2 * top_k, 16))
    token_indices = torch.arange(tokens, device=scores.device)[:, None]

    def needs_repair(step, proposal_step, current, loads):
        del current, proposal_step
        if scores.device.type == "cuda":
            # Inductor lowers while_loop's predicate on the host. A CPU-only
            # counter avoids reading CUDA loads there and permits graph capture.
            # Balanced states are fixed points of repair_step, so later CUDA
            # rounds make no changes. CPU diagnostics can exit immediately.
            return step < rounds
        return (step < rounds) & (loads != capacity).any()

    def repair_step(step, proposal_step, current, loads):
        surplus = (loads - capacity).clamp_min(0)
        deficit = (capacity - loads).clamp_min(0)
        assigned_mask = torch.zeros_like(detached_scores, dtype=torch.bool).scatter(
            1, current, True
        )
        eligible = (deficit[None, :] > 0) & ~assigned_mask
        pool_values, pool_targets = torch.topk(
            (detached_scores + tie_break).masked_fill(~eligible, float("-inf")),
            proposal_pool,
            dim=1,
        )
        proposal_hash = (
            token_indices * 1_103_515_245
            + pool_targets * 12_345
            + token_indices * pool_targets * 1_013_904_223
            + proposal_step * (pool_targets * 2_654_435_761 + 2_246_822_519)
        ) % 104_729
        diverse_slots = torch.topk(
            proposal_hash.float().masked_fill(~torch.isfinite(pool_values), float("-inf")),
            top_k,
            dim=1,
        ).indices
        diversify = proposal_step >= 16
        targets = torch.where(
            diversify, pool_targets.gather(1, diverse_slots), pool_targets[:, :top_k]
        )
        target_values = torch.where(
            diversify, pool_values.gather(1, diverse_slots), pool_values[:, :top_k]
        )
        source_scores = detached_scores.gather(1, current)
        source_values, source_slots = torch.sort(
            source_scores.masked_fill(surplus[current] <= 0, float("inf")),
            dim=1,
            stable=True,
        )
        sources = current.gather(1, source_slots)
        gains = (target_values - source_values).flatten()
        source_accepted = _quota_accept(sources.flatten(), gains, surplus)
        target_accepted = _quota_accept(
            targets.flatten(),
            gains.masked_fill(~source_accepted, float("-inf")),
            deficit,
        )
        accepted = source_accepted & target_accepted & torch.isfinite(gains)
        next_assignments = current.scatter(
            1,
            source_slots,
            torch.where(accepted.reshape(tokens, top_k), targets, sources),
        )
        moved = accepted.to(torch.int64)
        next_loads = loads.scatter_add(0, sources.flatten(), -moved).scatter_add(
            0, targets.flatten(), moved
        )
        return step + 1, proposal_step + 1, next_assignments, next_loads

    _, _, repaired, _ = torch.while_loop(
        needs_repair,
        repair_step,
        (
            torch.zeros((), device="cpu", dtype=torch.int64),
            torch.zeros((), device=scores.device, dtype=torch.int64),
            repaired,
            initial_loads,
        ),
    )
    return repaired


def balanced_dual_price(
    scores: torch.Tensor,
    *,
    top_k: int,
    iterations: int = 5,
    repair_rounds: int = 64,
) -> torch.Tensor:
    """Run QB and residual repair without a score-independent fallback.

    The finite repair budget is statistically qualified. If it is exhausted,
    report a failure instead of silently replacing learned routes with a
    round-robin schedule or returning invalid expert-major routing metadata.
    """
    tokens, experts = scores.shape
    capacity = uniform_capacity(tokens, experts, top_k)
    assignments, _ = dual_price_assignment(scores, top_k=top_k, iterations=iterations)
    repaired = repair_balanced_assignment(scores, assignments, rounds=repair_rounds)
    exact = (assignment_loads(repaired, experts) == capacity).all()
    torch._assert_async(
        exact,
        "QB residual repair exhausted its budget; increase repair_rounds",
    )
    return repaired


def routing_metadata(
    log_transport: torch.Tensor,
    expert_assignments: torch.Tensor,
    *,
    num_real_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert token-major assignments into the expert-major kernel contract."""
    scheduled_tokens, num_experts = log_transport.shape
    assignment_tokens, top_k = expert_assignments.shape
    assert assignment_tokens == scheduled_tokens
    total_assignments = scheduled_tokens * top_k
    capacity = total_assignments // num_experts

    token_ids = torch.arange(scheduled_tokens, device=log_transport.device)
    token_ids = token_ids[:, None].expand(scheduled_tokens, top_k)
    selected_log_weights = log_transport.gather(1, expert_assignments)
    selected_weights = selected_softmax(selected_log_weights)
    real_assignment = token_ids < num_real_tokens
    selected_weights = selected_weights * real_assignment.to(selected_weights.dtype)

    flat_experts = expert_assignments.flatten()
    order = torch.argsort(flat_experts, stable=True)
    safe_token_ids = token_ids.clamp_max(num_real_tokens - 1)
    expert_token_ids = safe_token_ids.flatten()[order].reshape(
        num_experts, capacity
    )
    expert_route_weights = selected_weights.flatten()[order].reshape(
        num_experts, capacity
    )
    inverse_order = torch.empty_like(order)
    inverse_order = inverse_order.scatter(
        0, order, torch.arange(total_assignments, device=order.device)
    )
    token_assignment_ids = inverse_order.reshape(scheduled_tokens, top_k)[
        :num_real_tokens
    ]
    return expert_token_ids, expert_route_weights, token_assignment_ids
