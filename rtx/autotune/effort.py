"""User budgets for retrieval-first, resumable UCB kernel autotuning."""
from dataclasses import dataclass,replace
import math
import random
import time
from pathlib import Path

from .bandit import AdaptiveBanditScheduler
from .core import Proposal,TuningBudget
from .first_hit import bounded_candidates
from .orchestrator import AutotuneOrchestrator,ConfirmationPolicy
from .strategies import CoordinateLocalSearch,RandomSearch,CostModelGuidedSearch


@dataclass(frozen=True)
class AutotunePolicy:
    """Per-context cumulative trial cap and per-invocation wall-time allowance.

    Existing layers accept this as ``tuning_policy=...``. Explicit tuning with
    a higher max_trials resumes durable history; runtime winner hits do not
    automatically refine. Time is a soft admission limit: an in-flight
    compilation/measurement is not preempted. Research priors require opt-in.
    """
    max_trials: int = 16
    max_seconds: float = 30.0
    pool_size: int = 128
    candidate_mix: str = 'legacy'
    structural_proposals: int = 512
    retrieval_trials: int | None = None
    exploration_trials: int = 1
    prior_artifact: str | None = None
    predictor_artifact: str | None = None
    predictor_heads: tuple[str, ...] = ('residual', 'mlp')
    allow_research_prior: bool = False
    exploration: float = .15
    confirmation_repeats: int = 1
    seed: int = 20260914
    warmup: int = 3
    samples: int = 5
    calls_per_sample: int = 3
    forward_trials: int | None = None
    backward_trials: int | None = None

    def __post_init__(self):
        if self.candidate_mix not in ('legacy', 'mixed'):
            raise ValueError('candidate_mix must be legacy or mixed')
        if type(self.structural_proposals) is not int or self.structural_proposals < 1:
            raise ValueError('structural_proposals must be positive')
        if (not isinstance(self.predictor_heads, tuple)
                or len(set(self.predictor_heads)) != len(self.predictor_heads)
                or any(h not in ('residual', 'mlp') for h in self.predictor_heads)):
            raise ValueError('predictor_heads must be a unique tuple of residual/mlp (empty means retrieval only)')
        if type(self.exploration_trials) is not int or self.exploration_trials < 0:
            raise ValueError('exploration_trials must be a nonnegative integer')
        for name in ('max_trials','pool_size','warmup','samples','calls_per_sample'):
            n=getattr(self,name)
            if type(n) is not int or n<1:raise ValueError(f'{name} must be a positive integer')
        if self.retrieval_trials is not None and (type(self.retrieval_trials) is not int or self.retrieval_trials < 0):
            raise ValueError('retrieval_trials must be nonnegative or None')
        for name in ('forward_trials','backward_trials'):
            n=getattr(self,name)
            if n is not None and (type(n) is not int or n<1):raise ValueError(f'{name} must be positive or None')
        if not math.isfinite(self.max_seconds) or self.max_seconds<=0:raise ValueError('max_seconds must be finite and positive')
        if not math.isfinite(self.exploration) or self.exploration<0:raise ValueError('exploration must be finite and nonnegative')
        if type(self.confirmation_repeats) is not int or self.confirmation_repeats<0:raise ValueError('confirmation_repeats must be nonnegative')

    @classmethod
    def preset(cls,name,**overrides):
        sizes={'quick':(8,15.),'balanced':(16,30.),'thorough':(32,90.),'extensive':(64,180.)}
        if name not in sizes:raise ValueError(f'unknown effort preset: {name}')
        trials,seconds=sizes[name]
        return cls(**{'max_trials':trials,'max_seconds':seconds,**overrides})

    def for_pass(self,backward=False):
        trials=self.backward_trials if backward else self.forward_trials
        return self if trials is None else replace(self,max_trials=trials)

    def coordinate(self,*,rtol=.05,atol=.5):
        from .legacy import CoordinateDescentPolicy
        return CoordinateDescentPolicy(max_trials=self.max_trials,time_budget_s=self.max_seconds,
            max_passes=1,restarts=1,warmup=self.warmup,samples=self.samples,
            calls_per_sample=self.calls_per_sample,correctness_rtol=rtol,correctness_atol=atol,seed=self.seed)


class FrozenPriorSearch:
    """Lazy bounded pool ranking; the orchestrator owns validation and timing."""
    name='frozen_ranker'
    def __init__(self,prior,pool):self.prior=prior;self.pool=pool;self.order=None
    def propose(self,adapter,history,rng,limit):
        if self.order is None:self.order=self.prior.order(adapter,self.pool,policy='ranker')
        return [Proposal(c,self.name) for c in self.order if adapter.config_id(c) not in history.seen_ids][:limit]
    def observe(self,observation):pass


def make_effort_autotuner(adapter,store,policy,*,progress=None,candidates=(),started_at=None):
    """Compose existing strategies/UCB. No new timing or persistence engine."""
    started_at=time.monotonic() if started_at is None else started_at
    prior=None
    predictor=None
    if policy.predictor_artifact:
        from .predictors import load_predictor, PortablePredictorSearch
        predictor=load_predictor(policy.predictor_artifact,research=policy.allow_research_prior)
        if predictor.bundle(adapter) is None:
            if progress:
                progress('PREDICTOR incompatible or ambiguous contract; using ordinary search')
            predictor=None
        elif any(adapter.context.family not in predictor.models.get(h,{}) for h in policy.predictor_heads):
            raise ValueError('requested predictor head is absent for this family')
    if policy.prior_artifact:
        from .prior import FrozenPrior
        prior=FrozenPrior.load(policy.prior_artifact,research=policy.allow_research_prior)
        if prior.bundle(adapter) is None:prior=None
    strategies=[CoordinateLocalSearch(beam_width=2),RandomSearch(max_pool_size=policy.pool_size)]
    coverage=[]
    pool_audit = None
    mixed = None
    if policy.candidate_mix == 'mixed':
        from .sampling import mixed_pool, StructuralSearch
        source = prior if prior is not None else predictor
        retrieved = [] if source is None else source.retrieval(adapter, limit=policy.pool_size)
        mixed, pool_audit = mixed_pool(adapter, limit=policy.pool_size,
            seed=policy.seed, retrieved=retrieved, seeds=candidates,
            structural_proposals=policy.structural_proposals)
        structural = [c for c in mixed if pool_audit['origins'][adapter.config_id(c)] == 'structural']
        if structural:
            strategies.append(StructuralSearch(structural))
        if progress:
            progress(f'POOL mixed counts={pool_audit["counts"]} static={pool_audit["structural_statuses"]}')
    if prior is not None or predictor is not None:
        retrieval_source=prior if prior is not None else predictor
        retrieved=retrieval_source.retrieval(adapter,limit=policy.pool_size)
        pool = mixed if mixed is not None else bounded_candidates(adapter,
            [*retrieved,*candidates,adapter.initial_config,
             *adapter.sample(random.Random(policy.seed),policy.pool_size,[adapter.initial_config])],policy.pool_size)
        if predictor is not None:
            strategies[:0]=[PortablePredictorSearch(predictor,pool,h) for h in policy.predictor_heads]
        else:
            strategies.insert(0,FrozenPriorSearch(prior,pool))
        # The mandatory initial candidate counts too; never spend the entire
        # user budget on warmup or mandatory arm pulls.
        # Reserve random attempts and one opportunity per other arm. This is
        # a trial-budget guarantee only: deadline/exhaustion can stop earlier.
        reserve = min(policy.exploration_trials, max(0, policy.max_trials-1))
        headroom = max(0, policy.max_trials-1-reserve-(len(strategies)-1))
        requested = min(3, max(1, policy.max_trials//8)) if policy.retrieval_trials is None else policy.retrieval_trials
        opening=min(requested, headroom)
        coverage=[Proposal(c,'retrieval') for c in retrieved if adapter.config_id(c)!=adapter.config_id(adapter.initial_config)][:opening]
    elif candidates:
        coverage=[Proposal(c,'structural') for c in bounded_candidates(adapter,candidates,policy.pool_size)
                  if adapter.config_id(c)!=adapter.config_id(adapter.initial_config)][:min(2,max(0,policy.max_trials-1))]
    if prior is None and predictor is None and policy.max_trials>=16:
        from .cost_model import GradientBoostedCostModel
        strategies.append(CostModelGuidedSearch(model=GradientBoostedCostModel(n_estimators=16,ensembles=2),
            min_observations=min(8,max(4,policy.max_trials//4)),pool_size=policy.pool_size,
            initial_pool_cap=policy.pool_size,local_candidate_cap=policy.pool_size,
            refit_interval=4,proposal_budget_s=.25))
    exploration_arm = 'structural' if any(s.name == 'structural' for s in strategies) else 'random'
    scheduler=AdaptiveBanditScheduler(exploration=policy.exploration,warmup_trials=0,
        minimum_pulls={exploration_arm: min(policy.exploration_trials, max(0, policy.max_trials-1))}
            if policy.exploration_trials else {},transfer_prior_strength=1.)
    tuner = AutotuneOrchestrator(adapter,store,strategies,scheduler,
        TuningBudget(policy.max_trials,policy.max_seconds),seed=policy.seed,
        max_trials_includes_resumed=True,transfer_history=False,coverage_proposals=coverage,
        confirmation=ConfirmationPolicy(policy.confirmation_repeats,.02,True),progress=progress,started_at=started_at)
    tuner.pool_audit = pool_audit
    return tuner


def tune_legacy_effort(kind,x,weight,*,policy,initial,axes,cache_dir,progress,grad=None):
    """Bridge old evaluator implementations into the shared orchestrator."""
    started_at=time.monotonic()
    from .legacy import DeviceFingerprint,default_cache_dir,TuningResult,MXFP8ForwardEvaluator,FWD_SEARCH_SPACE
    from .adapters import make_mxfp8_fwd_adapter,make_mxfp8_prequant_adapter,make_mxfp8_bwd_adapter
    from .store import ResidualTuningStore
    policy=policy.for_pass(kind=='backward')
    rtol,atol={'prequant':(.05,.5),'backward':(.07,1.),'forward':(.02,.2)}[kind]
    timing=policy.coordinate(rtol=rtol,atol=atol);x=x.reshape(-1,x.shape[-1])
    if kind=='prequant':
        from ..prequant_autotune import MXFP8PrequantEvaluator,PrequantTuningResult
        evaluator=MXFP8PrequantEvaluator(x,weight,timing);factory=make_mxfp8_prequant_adapter;result_type=PrequantTuningResult
    elif kind=='backward':
        from ..bwd_autotune import MXFP8BwdEvaluator,BwdTuningResult
        evaluator=MXFP8BwdEvaluator(grad.reshape(-1,grad.shape[-1]),x,weight,timing);factory=make_mxfp8_bwd_adapter;result_type=BwdTuningResult
    else:
        evaluator=MXFP8ForwardEvaluator(x,weight,timing);factory=make_mxfp8_fwd_adapter;result_type=TuningResult
        axes=FWD_SEARCH_SPACE if axes is None else axes
    adapter=factory(evaluator.problem,evaluator,initial=initial,axes=axes,device=DeviceFingerprint.current(x.device),
        tags={'measurement':'effort_v1','warmup':policy.warmup,'samples':policy.samples,
              'calls_per_sample':policy.calls_per_sample,'correctness_rtol':rtol,'correctness_atol':atol})
    root=(default_cache_dir() if cache_dir is None else Path(cache_dir))/'effort'/adapter.context.identifier
    store=ResidualTuningStore(root)
    reused=len(list(store.records(adapter.context)))
    result=make_effort_autotuner(adapter,store,policy,progress=progress,started_at=started_at).tune()
    return result_type(config=result.config,median_ms=result.median_ms,database_path=root,
        session_id=result.session_id,evaluated_trials=result.evaluated_trials,reused_trials=reused,elapsed_s=result.elapsed_s)
