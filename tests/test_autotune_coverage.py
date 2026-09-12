"""CPU checks for coverage ordering, legality, identity and persistence."""
from dataclasses import dataclass, asdict, replace
import unittest

from rtx.autotune.core import DiscreteKernelAdapter, KernelContext, TuningBudget
from rtx.autotune.coverage import coverage_plan
from rtx.autotune.orchestrator import AutotuneOrchestrator, SequentialScheduler
from rtx.autotune.strategies import RandomSearch
from rtx.autotune.outcomes import TrialOutcome
from rtx.autotune.store import InMemoryTuningStore


@dataclass(frozen=True)
class Config:
    a: int = 0
    b: int = 0


def adapter():
    return DiscreteKernelAdapter(
        context=KernelContext("coverage_test",1,{}), initial_config=Config(),
        axes={"a.tile": (0,1,2,3), "b.stages": (0,1,2)},
        config_id_fn=lambda c: str((c.a,c.b)), serialize_fn=asdict,
        deserialize_fn=lambda v: Config(**v),
        update_fn=lambda c,k,v: replace(c,**{k.split('.')[0]:v}),
        rejection_fn=lambda c: ("illegal","too large") if c.a==3 else None,
        evaluator=lambda c: TrialOutcome("ok",median_ms=1/(1+c.a+c.b)))


class CoverageTests(unittest.TestCase):
    def test_balanced_deduplicated_legal(self):
        a=adapter()
        plan,audit=coverage_plan(a)
        self.assertEqual([p.coordinate for p in plan],
                         ["a.tile","b.stages","a.tile","b.stages"])
        self.assertEqual(len({a.config_id(p.config) for p in plan}),len(plan))
        self.assertTrue(any(r["status"]=="rejected" for r in audit))
        self.assertTrue(any(r["status"]=="inactive_or_duplicate" for r in audit))

    def test_budget_and_resume(self):
        a=adapter(); store=InMemoryTuningStore(); plan,_=coverage_plan(a)
        def run(trials):
            return AutotuneOrchestrator(a,store,[RandomSearch()],
                SequentialScheduler((("random",None),)),TuningBudget(trials,30),
                max_trials_includes_resumed=True,coverage_proposals=plan).tune()
        result=run(3)
        self.assertEqual(result.strategy_trials["coverage"],2)
        records=list(store.records(a.context))
        self.assertEqual(len(records),3)
        self.assertEqual([r["strategy"] for r in records],["initial","coverage","coverage"])
        run(5)
        records=list(store.records(a.context))
        self.assertEqual(len(records),5)
        self.assertEqual(len({r["config_id"] for r in records}),5)

    def test_seed_has_priority_even_when_it_is_a_neighbour(self):
        a=adapter()
        plan,_=coverage_plan(a,[Config(b=2),Config(b=2)])
        self.assertEqual(plan[0].config,Config(b=2))
        self.assertEqual(plan[0].coordinate,"seed")
        self.assertEqual(sum(p.config==Config(b=2) for p in plan),1)


if __name__=="__main__":
    unittest.main()
