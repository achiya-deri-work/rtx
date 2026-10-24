"""Frozen-model, bounded-portfolio first-hit selection for any KernelAdapter.

No online model fitting, CUDA imports, or Cartesian search. The caller owns
the semantic contract and supplies complete, potentially transferable schedules.
"""
from dataclasses import dataclass
import math
from pathlib import Path
import time

from .core import ComposableTuningResult, Proposal, TuningBudget, stable_id
from .orchestrator import AutotuneOrchestrator, ConfirmationPolicy, SequentialScheduler
from .pretrained import ContextRankingModel, portable_features


def contract(context):
    """Only timings for the same semantics/revision may train one ranker."""
    workload = context.workload
    return {
        "family": context.family, "revision": context.kernel_revision,
        "regime": context.regime,
        "policy": {k: workload[k] for k in
                   ("precision", "activation", "phase", "silu_math", "scaling") if k in workload},
        "tags": dict(context.tags),
        "toolchain":{k:context.device[k] for k in
                     ("torch_version","torch_cuda_version","cutlass_dsl_version") if k in context.device},
    }


def model_features(features):
    """Pre-compilation inputs only; never leak measured compiler resources."""
    return {k: v for k, v in portable_features(features).items()
            if not any(word in k.lower() for word in
                       ("compiled", "measured_register", "spill", "ptxas", "telemetry"))
            and not (k.startswith("context.device.") and any(word in k.lower() for word in
                         ("compiler.","version", "device_index", "platform")))}


@dataclass(frozen=True)
class FirstHitPolicy:
    max_trials: int = 8
    seconds: float = 30.0
    pool_size: int = 128
    confirmation_repeats: int = 1
    challenger_stride: int = 3

    def __post_init__(self):
        if min(self.max_trials, self.pool_size, self.challenger_stride) < 1 or self.seconds <= 0:
            raise ValueError("first-hit budgets must be positive")
        if self.confirmation_repeats < 0:
            raise ValueError("confirmation repeats must be nonnegative")


class FrozenRanker:
    """JSON-only artifact; deployment is disabled unless explicitly qualified."""
    def __init__(self, document, *, allow_unqualified=False):
        if document.get("schema_version") != 1:
            raise ValueError("unsupported first-hit artifact schema")
        if not document.get("deployment_enabled", False) and not allow_unqualified:
            raise ValueError("ranker has not passed prospective deployment qualification")
        self.document = document
        if document.get("backend","builtin")=="xgboost":
            self.model = XGBRankerModel.from_state_dict(document["model"])
        else:
            self.model = ContextRankingModel.from_state_dict(document["model"])

    @classmethod
    def load(cls, path, **kwargs):
        import json
        return cls(json.loads(Path(path).read_text()), **kwargs)

    def compatible(self, context):
        return self.document["contract"] == contract(context)

    def predict(self, adapter, candidates):
        if not self.compatible(adapter.context):
            raise ValueError("first-hit model contract/revision mismatch")
        return self.model.predict([model_features(adapter.features(c)) for c in candidates])


class XGBRankerModel:
    """Optional offline comparator; JSON serialization, no pickle execution."""
    def __init__(self):
        from xgboost import XGBRanker
        from .cost_model import SparseFeatureVectorizer
        self.model=XGBRanker(n_estimators=64,max_depth=4,learning_rate=.08,
                             objective="rank:pairwise",n_jobs=2,random_state=0)
        self.vectorizer=SparseFeatureVectorizer()

    def fit(self, observations):
        from collections import defaultdict
        import numpy as np
        groups=defaultdict(list)
        for row in observations:
            if row.successful: groups[row.context_id].append(row)
        rows=[]; labels=[]; sizes=[]
        for group in groups.values():
            sizes.append(len(group)); rows.extend(group)
            worst=max(r.score for r in group)
            labels.extend(math.log(worst/r.score) for r in group)
        features=[model_features(r.features) for r in rows]
        self.vectorizer.fit(features)
        self.model.fit(self.vectorizer.transform(features),np.asarray(labels),group=sizes)

    def predict(self, features):
        import numpy as np
        # Higher relevance is better; the shared selector expects lower cost.
        score=-self.model.predict(self.vectorizer.transform(features))
        return score,np.zeros(len(score))

    def state_dict(self):
        import json
        return {"vectorizer":self.vectorizer.state_dict(),
                "booster":json.loads(self.model.get_booster().save_raw(raw_format="json"))}

    @classmethod
    def from_state_dict(cls,state):
        import json
        result=cls();result.vectorizer.load_state_dict(state["vectorizer"])
        result.model.load_model(bytearray(json.dumps(state["booster"]).encode()))
        return result


def bounded_candidates(adapter, candidates, limit=128):
    """Bound CPU work as well as returned pool size. No hidden wide search."""
    accepted, seen = [], set()
    for index, candidate in enumerate(candidates):
        if index >= limit * 4 or len(accepted) >= limit:
            break
        try:
            cid = adapter.config_id(candidate)
            if cid in seen or adapter.rejection(candidate) is not None:
                continue
            seen.add(cid)
            accepted.append(candidate)
        except (ValueError, TypeError, KeyError):
            continue
    return accepted


def retrieve_candidates(adapter, records, limit=8, max_records=4096, per_context=1):
    """Retrieve measured nearby-context winners, never transfer their timings.

    Keep at most max_records recent compatible observations in memory. Scan
    the journal fully: unrelated/old families must not hide appended evidence.
    """
    from collections import deque
    if limit < 1 or max_records < 1 or per_context < 1:
        return []
    target=adapter.context.features()
    def semantic_features(features):
        return {k:v for k,v in features.items() if k.startswith("context.tags.")
                or k.startswith("context.regime=")
                or k.startswith("context.environment.")
                or k.startswith("context.workload.") and "=" in k
                or k.startswith("context.device.") and any(
                    k.startswith("context.device." + field + "=") for field in
                    ("torch_version", "torch_cuda_version", "cutlass_dsl_version"))}
    semantics=semantic_features(target)
    recent=deque(maxlen=max_records)
    for record in records:
        if (record.get("family")!=adapter.context.family
                or record.get("kernel_revision")!=adapter.context.kernel_revision
                or record.get("outcome",{}).get("status")!="ok"):
            continue
        features=record.get("features",{})
        if record["outcome"].get("metadata",{}).get("sampling",{}).get("collection",{}).get("stationary") is False:
            continue
        if semantic_features(features) != semantics: continue
        try:
            score=float(record["outcome"]["median_ms"])
            if not math.isfinite(score) or score<=0:continue
            if "context_id" not in record: continue
            recent.append((score,record))
        except (TypeError,KeyError,ValueError):continue
    contexts={}
    for score,record in recent:
        contexts.setdefault(record["context_id"],[]).append((score,record))
    def distance(item):
        features=item[1]["features"]
        total=0.0
        for axis in ("m","n","k","h","o"):
            key="context.workload."+axis
            if key in target:
                total+=abs(math.log2(max(1,features.get(key,1))/max(1,target[key])))
        # Shapes remain the primary prior. At equal shape distance, prefer
        # measured schedules from similar resource capacities, not SKU names
        # or device UUIDs. Unknown capacities cannot look like a perfect match.
        hardware, missing = 0., 0
        for field in ("multiprocessor_count", "l2_cache_size", "memory_bus_width_bits"):
            key = "context.device." + field
            legacy_key = "context.device.memory_bus_width" if field == "memory_bus_width_bits" else key
            target_value = target.get(key, target.get(legacy_key, 0))
            if target_value > 0:
                value = features.get(key, features.get(legacy_key, 0))
                if value > 0:
                    hardware += abs(math.log2(value / target_value))
                else:
                    missing += 1
        return total, missing, hardware
    portfolios = []
    for group in sorted(contexts.values(),key=lambda group:distance(group[0])):
        portfolio, local_seen = [], set()
        for _,record in sorted(group,key=lambda item:item[0]):
            try:
                candidate=adapter.deserialize(record["config"])
                cid=adapter.config_id(candidate)
                if cid in local_seen or adapter.rejection(candidate) is not None:
                    continue
                local_seen.add(cid)
                portfolio.append(candidate)
            except (TypeError,KeyError,ValueError):continue
            if len(portfolio)>=per_context:break
        portfolios.append(portfolio)
    candidates, seen=[], set()
    for rank in range(per_context):
        for portfolio in portfolios:
            if rank>=len(portfolio):continue
            candidate=portfolio[rank]
            cid=adapter.config_id(candidate)
            if cid not in seen:
                candidates.append(candidate)
                seen.add(cid)
            if len(candidates)>=limit:return candidates
    return candidates


class PortfolioSearch:
    name = "first_hit"

    def __init__(self, candidates, ranker=None, challenger_stride=3):
        self.candidates = candidates
        self.ranker = ranker
        self.challenger_stride = challenger_stride
        self.order = None

    def propose(self, adapter, history, rng, limit):
        if self.order is None:
            ordered = list(self.candidates)
            if self.ranker is not None:
                mean, std = self.ranker.predict(adapter, ordered)
                ranked = sorted(range(len(ordered)), key=lambda i: float(mean[i]))
                uncertain = sorted(range(len(ordered)), key=lambda i: -float(std[i]))
                indexes, used = [], set()
                while len(indexes) < len(ordered):
                    source = uncertain if (len(indexes)+1) % self.challenger_stride == 0 else ranked
                    index = next(i for i in source if i not in used)
                    used.add(index); indexes.append(index)
                ordered = [ordered[i] for i in indexes]
            self.order = ordered
        return [Proposal(c, self.name, metadata={"frozen_ranker": self.ranker is not None})
                for c in self.order if adapter.config_id(c) not in history.seen_ids][:limit]

    def observe(self, observation):
        pass


def tune_first_hit(adapter, store, candidates, *, policy=FirstHitPolicy(), ranker=None, progress=print,
                   candidate_group=None, refine=False, retrieval_per_context=1):
    """Select a measured winner; exact compatible cache hits do no evaluation.

    A single compiler invocation is not preemptible here. Campaign supervisors
    must isolate workers to enforce hard deadlines. Do not call during capture.
    """
    started = time.monotonic()
    successful = []
    for record in store.records(adapter.context):
        outcome = record.get("outcome", {})
        if outcome.get("metadata",{}).get("sampling",{}).get("collection",{}).get("stationary") is False:
            continue
        if (record.get("kernel_revision") != adapter.context.kernel_revision
                or outcome.get("status") != "ok"):
            continue
        latency = outcome.get("median_ms")
        if latency is None or not math.isfinite(latency) or latency <= 0:
            continue
        try:
            config = adapter.deserialize(record["config"])
            if adapter.rejection(config) is None:
                successful.append((latency, config))
        except (KeyError, ValueError, TypeError):
            continue
    if successful and not refine:
        score, config = min(successful, key=lambda pair: pair[0])
        if progress:
            progress(f"FIRST_HIT cache {adapter.config_id(config)} {score*1000:.3f}us")
        return ComposableTuningResult(config, score, "cache", adapter.context.identifier,
                                      0, time.monotonic()-started, {}, str(store.path))
    retrieved=retrieve_candidates(adapter,store.records(None),per_context=retrieval_per_context)
    combined = [*retrieved, *candidates]
    pool = bounded_candidates(adapter, combined,
                              max(1, len(combined)) if candidate_group is not None else policy.pool_size)
    if candidate_group is not None:
        from .preparation import preparation_portfolio
        pool = preparation_portfolio(pool, plan_of=candidate_group,
                                     score=lambda _: 0, limit=policy.pool_size)
    strategy = PortfolioSearch(pool, ranker, policy.challenger_stride)
    if progress:
        progress(f"FIRST_HIT pool={len(pool)} trials<={policy.max_trials} learned={ranker is not None}")
    return AutotuneOrchestrator(
        adapter, store, [strategy], SequentialScheduler(((strategy.name, None),)),
        TuningBudget(policy.max_trials, policy.seconds), transfer_history=False,
        max_trials_includes_resumed=refine,
        confirmation=ConfirmationPolicy(policy.confirmation_repeats, .02, True),
        progress=progress,
    ).tune()
