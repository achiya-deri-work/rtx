"""Small, bounded policies for first-use runtime autotuning.

The campaign policies deliberately search for minutes or hours.  These
policies are the compile-like counterpart: enough exploration to improve a
previously unseen exact context without turning the first model iteration into
an offline campaign.
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import replace

from .legacy import CoordinateDescentPolicy
from .recipes import HybridTuningPolicy


def run_runtime_tuner(tuner, *args, **kwargs):
    """Keep a correct first invocation usable when the clock samples are noisy.

    This is an in-process seed fallback, never a measured/cacheable winner.
    Compilation, numerical, device errors and unverified seeds still raise.
    Explicit offline tuning keeps its strict no-eligible-winner exception.
    """
    from types import SimpleNamespace
    import warnings
    from .orchestrator import NoEligibleTimingError
    try:
        return tuner(*args, **kwargs)
    except NoEligibleTimingError as exc:
        if exc.fallback_config is None and exc.fallback_validation is not None:
            exc.fallback_config = exc.fallback_validation()
        if exc.fallback_config is None:
            raise
        warnings.warn("RTX timing samples were unstable; using the correctness-checked "
                      "initial schedule for this process without installing a winner.",
                      RuntimeWarning, stacklevel=2)
        return SimpleNamespace(config=exc.fallback_config, median_ms=None,
                               tuning_evidence={"timing_fallback": True})


def checked_seed_validator(adapter, evaluator, store):
    """Lazy fresh validation after noisy history exhausts search.

    This records a separate correctness-only session, never a timing trial or
    winner. Offline callers do not invoke it. Historical correctness alone is
    insufficient. MXFP8 uses current operands; dimension-only NVFP4 tuning
    reuses its established synthetic harness oracle, not application inputs.
    """
    def validate():
        import math
        from .outcomes import raise_if_fatal_device_context_error
        config=adapter.initial_config
        rejection=adapter.rejection(config)
        if rejection is not None:
            raise RuntimeError(f"runtime fallback rejected: {rejection}")
        session=store.start_session(adapter.context,{
            "purpose":"runtime_fallback_validation", "search_trials":0})
        outcome=evaluator(config,validate_only=True)
        checked=(outcome.status == 'ok' and outcome.max_abs_error is not None
                 and math.isfinite(outcome.max_abs_error))
        store.record_event(session,'runtime_fallback_validation',{
            'config_id':adapter.config_id(config),'outcome':outcome.as_dict(),
            'checked':checked})
        store.finish_session(session,{'status':'validated' if checked else 'failed'})
        raise_if_fatal_device_context_error(outcome.error)
        if not checked:
            raise RuntimeError(f"runtime fallback validation failed: {outcome.status}: {outcome.error}")
        return config
    return validate


def run_runtime_config_tuner(tuner, *args, **kwargs):
    """Equivalent fallback for NVFP4 tuners returning a configuration directly.

    Their persistence happens after successful orchestration. On this exception
    that code was never reached; only the checked seed enters the process cache.
    """
    result = run_runtime_tuner(tuner, *args, **kwargs)
    if getattr(result, "tuning_evidence", {}).get("timing_fallback"):
        return result.config
    return result


def _seconds() -> float:
    return float(os.getenv("RTX_BALANCED_AUTOTUNE_SECONDS", "30"))


def _trials() -> int:
    return int(os.getenv("RTX_BALANCED_AUTOTUNE_TRIALS", "24"))


def _pairwise_artifact() -> str | None:
    configured = os.getenv("RTX_AUTOTUNE_PAIRWISE_ARTIFACT")
    if configured is not None:
        return (
            None
            if configured.lower() in {"", "0", "false", "none", "off"}
            else configured
        )
    bundled = (
        Path(__file__).with_name("artifacts")
        / "blackwell_diversity_atlas_v1_pairwise"
    )
    return str(bundled) if (bundled / "pairwise_manifest.json").is_file() else None


def balanced_coordinate_policy(
    *,
    coordinate_order: tuple[str, ...] | None = None,
    correctness_rtol: float = 2e-2,
    correctness_atol: float = 2e-1,
) -> CoordinateDescentPolicy:
    """Return the bounded policy used by legacy coordinate kernel families."""

    policy = CoordinateDescentPolicy(
        time_budget_s=_seconds(),
        max_trials=_trials(),
        max_passes=1,
        restarts=1,
        warmup=int(os.getenv("RTX_BALANCED_AUTOTUNE_WARMUP", "3")),
        samples=int(os.getenv("RTX_BALANCED_AUTOTUNE_SAMPLES", "5")),
        calls_per_sample=int(
            os.getenv("RTX_BALANCED_AUTOTUNE_CALLS_PER_SAMPLE", "5")
        ),
        min_improvement=0.002,
        correctness_rtol=correctness_rtol,
        correctness_atol=correctness_atol,
        randomize_coordinates=True,
        seed=int(os.getenv("RTX_BALANCED_AUTOTUNE_SEED", "20260817")),
    )
    if coordinate_order is not None:
        policy = replace(policy, coordinate_order=coordinate_order)
    return policy


def balanced_hybrid_policy() -> HybridTuningPolicy:
    """Return the 24-trial learned/bandit policy used on a runtime cache miss."""

    trials = _trials()
    warmup = min(8, max(4, trials // 3))
    return HybridTuningPolicy(
        portfolio="hybrid",
        orchestration="bandit",
        max_trials=trials,
        time_budget_s=_seconds(),
        cost_model_trials=max(warmup, trials // 2),
        model_warmup=warmup,
        model_pool_size=1024,
        model_initial_pool_cap=512,
        model_proposal_budget_s=0.25,
        model_refit_interval=4,
        model_estimators=16,
        model_ensembles=2,
        local_beam_width=3,
        local_model_refit_interval=8,
        local_model_candidate_cap=256,
        bandit_coordinate_bootstrap=4,
        bandit_learned_bootstrap=2,
        bandit_model_local_bootstrap=2,
        confirmation_repeats=1,
        confirmation_ratio=0.02,
        confirm_initial=True,
        seed=int(os.getenv("RTX_BALANCED_AUTOTUNE_SEED", "20260817")),
        pretrained_artifact=os.getenv("RTX_AUTOTUNE_PRETRAINED_ARTIFACT") or None,
        pairwise_artifact=_pairwise_artifact(),
        warmup=int(os.getenv("RTX_BALANCED_AUTOTUNE_WARMUP", "3")),
        samples=int(os.getenv("RTX_BALANCED_AUTOTUNE_SAMPLES", "5")),
        calls_per_sample=int(
            os.getenv("RTX_BALANCED_AUTOTUNE_CALLS_PER_SAMPLE", "2048")
        ),
    )


__all__ = ["balanced_coordinate_policy", "balanced_hybrid_policy"]


def balanced_effort_policy():
    """Shared first-input policy; legacy factories above remain explicit APIs.

    Called by real-tensor dispatch, never by fake/meta execution. Family
    evaluators retain their own numerical correctness thresholds.
    """
    from .effort import AutotunePolicy

    return AutotunePolicy(
        max_trials=int(os.getenv("RTX_BALANCED_AUTOTUNE_TRIALS", "16")),
        max_seconds=_seconds(),
        warmup=int(os.getenv("RTX_BALANCED_AUTOTUNE_WARMUP", "3")),
        samples=int(os.getenv("RTX_BALANCED_AUTOTUNE_SAMPLES", "5")),
        calls_per_sample=int(os.getenv("RTX_BALANCED_AUTOTUNE_CALLS_PER_SAMPLE", "5")),
        seed=int(os.getenv("RTX_BALANCED_AUTOTUNE_SEED", "20260817")),
    )
