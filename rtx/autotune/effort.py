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
    a higher max_trials resumes durable history; runtime winner hits refine
    only with ``cache_behavior='refine'``. Time is a soft admission limit: an in-flight
    compilation/measurement is not preempted. Research priors require opt-in.
    """
    max_trials: int = 16
    search_schedule: str = 'legacy'
    cache_behavior: str = 'reuse'
    max_seconds: float = 30.0
    pool_size: int = 128
    candidate_mix: str = 'legacy'
    structural_proposals: int = 512
    retrieval_trials: int | None = None
    ranked_opening_trials: int = 0
    exploration_trials: int = 1
    baseline_trials: int = 0
    initial_neighborhood_trials: int = 0
    local_anchor_tolerance: float = 0.0
    prior_artifact: str | None = None
    predictor_artifact: str | None = None
    predictor_heads: tuple[str, ...] = ('residual', 'mlp')
    allow_research_prior: bool = False
    exploration: float = .15
    confirmation_repeats: int = 1
    finalist_rounds: int = 0
    finalist_margin: float = 0.005
    finalist_reserve_fraction: float = 0.2
    seed: int = 20260914
    warmup: int = 3
    samples: int = 5
    calls_per_sample: int = 3
    backward_measurement_mode: str = 'host'
    forward_trials: int | None = None
    backward_trials: int | None = None
    preparation_selection: bool = False

    def __post_init__(self):
        if type(self.preparation_selection) is not bool:
            raise ValueError('preparation_selection must be bool')
        if self.backward_measurement_mode not in ('host','cuda_graph','auto'):
            raise ValueError('backward_measurement_mode must be host, cuda_graph or auto')
        if type(self.ranked_opening_trials) is not int or not 0 <= self.ranked_opening_trials <= 2:
            raise ValueError('ranked_opening_trials must be an integer in [0, 2]')
        if self.ranked_opening_trials:
            if self.search_schedule != 'progressive' or not self.predictor_artifact or not self.predictor_heads:
                raise ValueError('ranked opening requires progressive search and a predictor head')
            retrieval = 2 if self.retrieval_trials is None else self.retrieval_trials
            if (self.initial_neighborhood_trials or 2) + retrieval + self.ranked_opening_trials > 6:
                raise ValueError('ranked opening must leave an exploration slot within eight trials')
        if self.cache_behavior not in ('reuse', 'refine'):
            raise ValueError('cache_behavior must be reuse or refine')
        if type(self.finalist_rounds) is not int or self.finalist_rounds < 0 or self.finalist_rounds % 2:
            raise ValueError('finalist_rounds must be zero or a positive even integer')
        for name in ('finalist_margin', 'finalist_reserve_fraction'):
            if not math.isfinite(getattr(self,name)) or not 0 <= getattr(self,name) < 1:
                raise ValueError(f'{name} must be finite and in [0, 1)')
        if self.search_schedule not in ('legacy', 'progressive'):
            raise ValueError('search_schedule must be legacy or progressive')
        if self.candidate_mix not in ('legacy', 'mixed'):
            raise ValueError('candidate_mix must be legacy or mixed')
        if type(self.structural_proposals) is not int or self.structural_proposals < 1:
            raise ValueError('structural_proposals must be positive')
        if (not isinstance(self.predictor_heads, tuple)
                or len(set(self.predictor_heads)) != len(self.predictor_heads)
                or any(h not in ('residual', 'mlp', 'trees') for h in self.predictor_heads)):
            raise ValueError('predictor_heads must be a unique tuple of residual/mlp/trees (empty means retrieval only)')
        if type(self.exploration_trials) is not int or self.exploration_trials < 0:
            raise ValueError('exploration_trials must be a nonnegative integer')
        if type(self.baseline_trials) is not int or self.baseline_trials < 0:
            raise ValueError('baseline_trials must be a nonnegative integer')
        if type(self.initial_neighborhood_trials) is not int or self.initial_neighborhood_trials < 0:
            raise ValueError('initial_neighborhood_trials must be a nonnegative integer')
        if not math.isfinite(self.local_anchor_tolerance) or not 0 <= self.local_anchor_tolerance < 1:
            raise ValueError('local_anchor_tolerance must be finite and in [0, 1)')
        if self.initial_neighborhood_trials and (self.initial_neighborhood_trials +
                self.baseline_trials + self.exploration_trials > self.max_trials - 1):
            raise ValueError('neighborhood and arm reservations must leave the initial trial')
        if self.baseline_trials + self.exploration_trials > self.max_trials - 1 and self.baseline_trials:
            raise ValueError('baseline and exploration reservations must leave the initial trial')
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
        updates={} if trials is None else {'max_trials':trials}
        if (not backward and self.backward_measurement_mode!='host') or self.backward_measurement_mode=='auto':
            updates['backward_measurement_mode']='host'
        return replace(self,**updates) if updates else self

    def evidence_contract(self):
        """Search/measurement identity; effort caps and promotion are separate."""
        from dataclasses import asdict
        import hashlib
        from .core import stable_id
        values=asdict(self)
        if values['backward_measurement_mode']=='host':
            values.pop('backward_measurement_mode')
        if not values['ranked_opening_trials']:
            values.pop('ranked_opening_trials')
        for name in ('max_trials','max_seconds','forward_trials','backward_trials',
                     'cache_behavior','finalist_rounds','finalist_margin','finalist_reserve_fraction'):
            values.pop(name)
        for name in ('prior_artifact','predictor_artifact'):
            if values[name] and Path(values[name]).is_file():
                values[name]=hashlib.sha256(Path(values[name]).read_bytes()).hexdigest()
        from .runtime_coordinates import process_l2_policy
        return stable_id(dict(schema=1,policy=values,
            runtime_measurement_revision=1, runtime_l2_policy=process_l2_policy(),
            runtime_search_space_revision=1,
            **({'schedule_revision':2} if self.search_schedule=='progressive' else {})))

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


def effort_residual_root(root, policy):
    """Separate incompatible progressive evidence; extensions keep one path."""
    root = Path(root)
    return root / policy.evidence_contract() if policy.search_schedule == 'progressive' else root


class ProgressiveBanditScheduler(AdaptiveBanditScheduler):
    """Budget-independent exploration milestones, then ordinary contextual UCB."""
    progressive_exploration = True
    def select(self, names, statistics, trial_index):
        # One exploration opportunity by trial 8, three by 16, six by 32,
        # and ten by 64. Unavailable arms release their slots to normal UCB.
        milestones = (5, 8, 12, 16, 22, 28, 32, 40, 48, 56)
        required = sum(trial_index >= step for step in milestones) if self.progressive_exploration else 0
        arm = 'structural' if 'structural' in names else 'random'
        if (arm in statistics and statistics[arm].pulls < required and
                statistics[arm].unavailable_until <= trial_index):
            return arm
        return super().select(names, statistics, trial_index)


def make_effort_autotuner(adapter,store,policy,*,progress=None,candidates=(),started_at=None):
    """Compose existing strategies/UCB. No new timing or persistence engine."""
    started_at=time.monotonic() if started_at is None else started_at
    progressive = policy.search_schedule == 'progressive'
    planning_trials = 64 if progressive else policy.max_trials
    neighborhood_trials = policy.initial_neighborhood_trials or (2 if progressive else 0)
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
    strategies=[CoordinateLocalSearch(beam_width=2, anchor_tolerance=policy.local_anchor_tolerance),RandomSearch(max_pool_size=policy.pool_size)]
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
        reserve = min(policy.exploration_trials + policy.baseline_trials + policy.initial_neighborhood_trials, max(0, policy.max_trials-1))
        headroom = max(0, planning_trials-1-reserve-(len(strategies)-1))
        requested = (2 if progressive else min(3, max(1, policy.max_trials//8))) if policy.retrieval_trials is None else policy.retrieval_trials
        opening=min(requested, headroom)
        coverage=[Proposal(c,'retrieval') for c in retrieved if adapter.config_id(c)!=adapter.config_id(adapter.initial_config)]
        if not progressive:
            coverage=coverage[:opening]
    elif candidates:
        coverage=[Proposal(c,'structural') for c in bounded_candidates(adapter,candidates,policy.pool_size)
                  if adapter.config_id(c)!=adapter.config_id(adapter.initial_config)][:min(2,max(0,
                      planning_trials-1-policy.baseline_trials-policy.exploration_trials-policy.initial_neighborhood_trials))]
    # Fix this prefix to the initial configuration, not the noisy incumbent.
    # The orchestrator persists each proposal and skips it on resume.
    anchored = []
    seen = {adapter.config_id(adapter.initial_config)}
    if neighborhood_trials:
        for coordinate, value, candidate in adapter.neighbors(adapter.initial_config):
            cid = adapter.config_id(candidate)
            if cid in seen or adapter.rejection(candidate) is not None:
                continue
            seen.add(cid)
            anchored.append(Proposal(candidate, 'initial_neighborhood',
                parent_config_id=adapter.config_id(adapter.initial_config),
                coordinate=coordinate, coordinate_value=value))
            if len(anchored) == neighborhood_trials:
                break
    if progressive and (prior is not None or predictor is not None):
        novel = []
        for proposal in coverage:
            if len(novel) >= opening:
                break
            cid = adapter.config_id(proposal.config)
            if cid in seen or adapter.rejection(proposal.config) is not None:
                continue
            seen.add(cid)
            novel.append(proposal)
        coverage = anchored + novel
    else:
        coverage = anchored + [p for p in coverage if adapter.config_id(p.config) not in seen]
    if policy.ranked_opening_trials and predictor is not None:
        from types import SimpleNamespace
        seen.update(adapter.config_id(p.config) for p in coverage)
        heads = [s for s in strategies if isinstance(s, PortablePredictorSearch)]
        # Reuse the arm's frozen ordering/audit, rather than scoring twice.
        # Round-robin heads; duplicate or illegal ranks do not consume quota.
        added = 0
        while added < policy.ranked_opening_trials:
            progressed = False
            for arm in heads:
                proposals = arm.propose(adapter, SimpleNamespace(seen_ids=seen),
                                        random.Random(policy.seed), policy.pool_size)
                for proposal in proposals:
                    cid = adapter.config_id(proposal.config)
                    seen.add(cid)
                    if adapter.rejection(proposal.config) is not None:
                        continue
                    coverage.append(proposal)
                    added += 1
                    progressed = True
                    break
                if added == policy.ranked_opening_trials:
                    break
            if not progressed:
                break
    if prior is None and predictor is None and policy.max_trials>=16 and not progressive:
        from .cost_model import GradientBoostedCostModel
        strategies.append(CostModelGuidedSearch(model=GradientBoostedCostModel(n_estimators=16,ensembles=2),
            min_observations=min(8,max(4,policy.max_trials//4)),pool_size=policy.pool_size,
            initial_pool_cap=policy.pool_size,local_candidate_cap=policy.pool_size,
            refit_interval=4,proposal_budget_s=.25))
    exploration_arm = 'structural' if any(s.name == 'structural' for s in strategies) else 'random'
    minimum_pulls = ({exploration_arm: min(policy.exploration_trials, max(0, policy.max_trials-1))}
                     if policy.exploration_trials else {})
    if policy.baseline_trials:
        minimum_pulls['coordinate_local'] = policy.baseline_trials
    scheduler_type = ProgressiveBanditScheduler if progressive else AdaptiveBanditScheduler
    scheduler=scheduler_type(exploration=policy.exploration,warmup_trials=0,
        minimum_pulls=minimum_pulls,transfer_prior_strength=1.)
    if progressive:
        scheduler.progressive_exploration = policy.exploration_trials > 0
    tuner = AutotuneOrchestrator(adapter,store,strategies,scheduler,
        TuningBudget(policy.max_trials,policy.max_seconds),seed=policy.seed,
        max_trials_includes_resumed=True,transfer_history=False,coverage_proposals=coverage,
        confirmation=ConfirmationPolicy(policy.confirmation_repeats,.02,True),progress=progress,started_at=started_at)
    if progressive:
        from dataclasses import asdict
        from .core import stable_id
        # Budget extensions must retain the same search contract. Mutable
        # online-model fit state is not enabled in this initial schedule;
        # frozen prior arms, local search and exploration are restartable.
        contract = asdict(policy)
        if contract['backward_measurement_mode']=='host':
            contract.pop('backward_measurement_mode')
        if not contract['ranked_opening_trials']:
            contract.pop('ranked_opening_trials')
        for name in ('max_trials', 'max_seconds', 'forward_trials', 'backward_trials',
                     'cache_behavior', 'finalist_rounds', 'finalist_margin', 'finalist_reserve_fraction'):
            contract.pop(name)
        import hashlib
        for name in ('prior_artifact', 'predictor_artifact'):
            if contract[name]:
                path = Path(contract[name])
                if path.is_file():
                    contract[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        tuner.checkpoint_key = stable_id(dict(policy=contract,
            schedule_revision=2,
            coverage=[adapter.config_id(p.config) for p in coverage],
            pool=pool_audit))
    tuner.finalist_rounds = policy.finalist_rounds
    tuner.effort_contract = policy.evidence_contract()
    tuner.finalist_margin = policy.finalist_margin
    tuner.finalist_reserve_s = policy.max_seconds * policy.finalist_reserve_fraction if policy.finalist_rounds else 0.
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
        evaluator=MXFP8BwdEvaluator(grad.reshape(-1,grad.shape[-1]),x,weight,timing,
            measurement_mode=policy.backward_measurement_mode);factory=make_mxfp8_bwd_adapter;result_type=BwdTuningResult
    else:
        evaluator=MXFP8ForwardEvaluator(x,weight,timing);factory=make_mxfp8_fwd_adapter;result_type=TuningResult
        axes=FWD_SEARCH_SPACE if axes is None else axes
    adapter=factory(evaluator.problem,evaluator,initial=initial,axes=axes,device=DeviceFingerprint.current(x.device),
        tags={'measurement':'effort_v1','warmup':policy.warmup,'samples':policy.samples,
              'calls_per_sample':policy.calls_per_sample,'correctness_rtol':rtol,'correctness_atol':atol,
              **({'measurement_mode':policy.backward_measurement_mode} if kind=='backward' else {})})
    from .runtime_coordinates import runtime_schedule_adapter
    adapter=runtime_schedule_adapter(adapter)
    root=(default_cache_dir() if cache_dir is None else Path(cache_dir))/'effort'/adapter.context.identifier
    root = effort_residual_root(root, policy)
    store=ResidualTuningStore(root)
    reused=len(list(store.records(adapter.context)))
    result=make_effort_autotuner(adapter,store,policy,progress=progress,started_at=started_at).tune()
    return result_type(config=result.config,median_ms=result.median_ms,database_path=root,
        session_id=result.session_id,evaluated_trials=result.evaluated_trials,reused_trials=reused,elapsed_s=result.elapsed_s,
        tuning_evidence=result.tuning_evidence)
