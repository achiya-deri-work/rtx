from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import random
import tempfile
import unittest

from rtx.autotune import (
    AdaptiveBanditScheduler,
    ArmStatistics,
    AutotuneOrchestrator,
    ConfirmationPolicy,
    CoordinateLocalSearch,
    CostModelGuidedSearch,
    DiscreteKernelAdapter,
    GradientBoostedCostModel,
    FatalDeviceContextError,
    InMemoryTuningStore,
    JsonlTuningStore,
    KernelContext,
    HybridTuningPolicy,
    RandomSearch,
    RuntimeWinnerKey,
    SearchHistory,
    SharedModelFitState,
    SequentialScheduler,
    TrialOutcome,
    TuningBudget,
    UCB1Scheduler,
    import_legacy_json_database,
    make_mxfp8_bwd_adapter,
    make_mxfp8_fully_prequant_adapter,
    make_mxfp8_fwd_adapter,
    make_mxfp8_prequant_adapter,
    make_mxfp8_weight_prequant_adapter,
    load_runtime_winner,
    make_hybrid_autotuner,
    make_hybrid_ask_tell_runner,
    save_runtime_winner,
)
from rtx.autotune.adapters import (
    make_nvfp4_fully_prequant_adapter,
    make_nvfp4_fwd_adapter,
    make_nvfp4_weight_prequant_adapter,
)
from rtx.configs.nvfp4 import NVFP4Problem
from rtx.autotune.core import Proposal, evaluate_proposal
from rtx.kernels.mxfp8 import MXFP8Problem


@dataclass(frozen=True)
class _ToyConfig:
    x: int = 0
    y: int = 0


def _toy_score(config: _ToyConfig) -> float:
    return 0.1 + (config.x - 3) ** 2 + 1.5 * (config.y - 2) ** 2


def _toy_adapter() -> DiscreteKernelAdapter[_ToyConfig]:
    return DiscreteKernelAdapter(
        context=KernelContext(
            family="toy",
            kernel_revision=1,
            workload={"m": 512, "n": 1536, "k": 1536},
            device={"multiprocessor_count": 70, "name": "test"},
        ),
        initial_config=_ToyConfig(),
        axes={"x": tuple(range(5)), "y": tuple(range(5))},
        config_id_fn=lambda config: f"{config.x}-{config.y}",
        serialize_fn=asdict,
        deserialize_fn=lambda value: _ToyConfig(**value),
        update_fn=lambda config, coordinate, value: replace(
            config, **{coordinate: int(value)}
        ),
        evaluator=lambda config: TrialOutcome(
            "ok", median_ms=_toy_score(config), timings_ms=[_toy_score(config)]
        ),
        rejection_fn=lambda config: (
            ("implementation_rejected", "forbidden corner")
            if config == _ToyConfig(4, 4)
            else None
        ),
    )


class ComposableAutotuneTests(unittest.TestCase):
    def test_durable_ask_tell_runner_uses_store_and_resumes(self) -> None:
        adapter = _toy_adapter()
        store = InMemoryTuningStore()
        first = make_hybrid_ask_tell_runner(
            adapter,
            store,
            HybridTuningPolicy(
                portfolio="random",
                max_trials=4,
                time_budget_s=10,
                seed=19,
            ),
        ).tune()
        self.assertEqual(first.evaluated_trials, 4)
        self.assertEqual(store.sessions[0]["engine"], "durable_local_ask_tell")
        self.assertTrue(any(event["kind"] == "trial_issued" for event in store.events))
        second = make_hybrid_ask_tell_runner(
            adapter,
            store,
            HybridTuningPolicy(
                portfolio="random",
                max_trials=6,
                time_budget_s=10,
                seed=19,
            ),
        ).tune()
        self.assertEqual(second.evaluated_trials, 2)
        self.assertLessEqual(second.median_ms, first.median_ms)

    def test_sticky_device_fault_is_durable_then_aborts_worker(self) -> None:
        adapter = _toy_adapter()
        adapter.initial_config = _ToyConfig(1, 0)
        adapter.evaluator = lambda _config: TrialOutcome(
            "runtime_error",
            error="CUDA error: an illegal instruction was encountered",
        )
        store = InMemoryTuningStore()
        tuner = AutotuneOrchestrator(
            adapter,
            store,
            [RandomSearch()],
            SequentialScheduler((("random", None),)),
            TuningBudget(max_trials=2, time_budget_s=10),
        )
        with self.assertRaises(FatalDeviceContextError):
            tuner.tune()
        self.assertEqual(len(store.observations), 1)
        self.assertEqual(store.observations[0].outcome.status, "runtime_error")
        self.assertTrue(
            any(event["kind"] == "candidate_started" for event in store.events)
        )
        self.assertTrue(
            any(event["kind"] == "candidate_completed" for event in store.events)
        )

    def test_resume_excludes_candidate_interrupted_inside_cuda(self) -> None:
        adapter = _toy_adapter()
        store = InMemoryTuningStore()
        abandoned_id = adapter.config_id(adapter.initial_config)
        store.record_event(
            "dead-session",
            "candidate_started",
            {
                "attempt_id": "dead-attempt",
                "context_id": adapter.context.identifier,
                "config_id": abandoned_id,
                "config": asdict(adapter.initial_config),
                "strategy": "initial",
            },
        )
        result = AutotuneOrchestrator(
            adapter,
            store,
            [RandomSearch()],
            SequentialScheduler((("random", None),)),
            TuningBudget(max_trials=4, time_budget_s=10),
            seed=7,
        ).tune()
        self.assertEqual(result.evaluated_trials, 4)
        self.assertNotIn(
            abandoned_id,
            {observation.config_id for observation in store.observations},
        )
        self.assertTrue(
            any(
                event["kind"] == "abandoned_candidates_recovered"
                for event in store.events
            )
        )

    def test_recipe_portfolios_are_explicit_experimental_arms(self) -> None:
        random_tuner = make_hybrid_autotuner(
            _toy_adapter(),
            InMemoryTuningStore(),
            HybridTuningPolicy(portfolio="random", max_trials=4),
        )
        self.assertEqual(set(random_tuner.strategies), {"random"})

        local_tuner = make_hybrid_autotuner(
            _toy_adapter(),
            InMemoryTuningStore(),
            HybridTuningPolicy(
                portfolio="random_local",
                max_trials=8,
                cost_model_trials=4,
            ),
        )
        self.assertEqual(
            set(local_tuner.strategies), {"random", "coordinate_local"}
        )
        result = local_tuner.tune()
        self.assertEqual(result.evaluated_trials, 8)
        self.assertGreater(result.strategy_trials["random"], 0)
        self.assertGreater(result.strategy_trials["coordinate_local"], 0)

    def test_random_search_retries_after_a_seen_only_pool(self) -> None:
        base = _toy_adapter()
        seen = evaluate_proposal(
            base,
            Proposal(_ToyConfig(), "seed"),
            session_id="seed",
            sequence=0,
        )

        class RetryAdapter:
            def __init__(self):
                self.context = base.context
                self.initial_config = base.initial_config
                self.calls = 0

            def __getattr__(self, name):
                return getattr(base, name)

            def sample(self, rng, count, seeds):
                self.calls += 1
                return [_ToyConfig()] if self.calls == 1 else [_ToyConfig(1, 1)]

        adapter = RetryAdapter()
        proposals = RandomSearch(
            pool_multiplier=1, max_batches=2, max_pool_size=8
        ).propose(
            adapter,
            SearchHistory([seen], base.context.identifier),
            random.Random(3),
            1,
        )
        self.assertEqual(adapter.calls, 2)
        self.assertEqual(proposals[0].config, _ToyConfig(1, 1))

    def test_adaptive_bandit_replays_state_across_resume(self) -> None:
        adapter = _toy_adapter()
        with tempfile.TemporaryDirectory() as directory:
            store = JsonlTuningStore(Path(directory), fsync=False)
            strategies = [RandomSearch(), CoordinateLocalSearch()]
            first = AutotuneOrchestrator(
                adapter,
                store,
                strategies,
                AdaptiveBanditScheduler(
                    exploration=0.2,
                    warmup_trials=2,
                    warmup_arm="random",
                ),
                TuningBudget(max_trials=6, time_budget_s=10),
                seed=11,
                max_trials_includes_resumed=True,
            ).tune()
            self.assertEqual(first.evaluated_trials, 6)
            events_before = len(store.events_path.read_text().splitlines())
            second = AutotuneOrchestrator(
                adapter,
                store,
                [RandomSearch(), CoordinateLocalSearch()],
                AdaptiveBanditScheduler(
                    exploration=0.2,
                    warmup_trials=2,
                    warmup_arm="random",
                ),
                TuningBudget(max_trials=8, time_budget_s=10),
                seed=11,
                max_trials_includes_resumed=True,
            ).tune()
            self.assertEqual(second.evaluated_trials, 2)
            new_events = [
                json.loads(line)
                for line in store.events_path.read_text().splitlines()[events_before:]
                if json.loads(line).get("kind") == "strategy_selected"
            ]
            self.assertTrue(new_events)
            state = new_events[0]["payload"]["scheduler_state"]
            self.assertEqual(state["scheduler"], "adaptive_contextual_bandit")
            self.assertGreater(
                sum(arm["pulls"] for arm in state["arms"].values()),
                0,
            )

    def test_adaptive_bandit_reward_penalizes_expensive_failures(self) -> None:
        scheduler = AdaptiveBanditScheduler(cost_scale_s=1.0)
        fast = evaluate_proposal(
            _toy_adapter(),
            Proposal(_ToyConfig(), "random"),
            session_id="session",
            sequence=0,
        )
        fast.outcome = TrialOutcome("compile_error", error="bad")
        fast.elapsed_s = 0.01
        slow = replace(fast, elapsed_s=10.0)
        self.assertLess(scheduler.reward(float("inf"), slow), scheduler.reward(float("inf"), fast))

    def test_adaptive_bandit_charges_cpu_proposal_time(self) -> None:
        scheduler = AdaptiveBanditScheduler(cost_scale_s=1.0)
        cheap = evaluate_proposal(
            _toy_adapter(),
            Proposal(_ToyConfig(), "gradient_boosted"),
            session_id="session",
            sequence=0,
        )
        expensive = replace(
            cheap,
            metadata={**cheap.metadata, "proposal_elapsed_s": 10.0},
        )
        self.assertLess(
            scheduler.reward(float("inf"), expensive),
            scheduler.reward(float("inf"), cheap),
        )

    def test_adaptive_bandit_bootstraps_each_configured_arm_after_warmup(self) -> None:
        scheduler = AdaptiveBanditScheduler(
            warmup_trials=2,
            warmup_arm="random",
            minimum_pulls={"coordinate_local": 2, "gradient_boosted": 1},
        )
        names = ("random", "coordinate_local", "gradient_boosted")
        statistics = {name: ArmStatistics() for name in names}
        self.assertEqual(scheduler.select(names, statistics, 0), "random")
        statistics["random"].pulls = 2
        self.assertEqual(
            scheduler.select(names, statistics, 2), "coordinate_local"
        )
        statistics["coordinate_local"].pulls = 2
        self.assertEqual(
            scheduler.select(names, statistics, 4), "gradient_boosted"
        )
        statistics["gradient_boosted"].pulls = 1
        snapshot = scheduler.snapshot(names, statistics, 5)
        self.assertEqual(snapshot["minimum_pulls"]["coordinate_local"], 2)

    def test_hybrid_bandit_contains_observed_coordinate_local_arm(self) -> None:
        tuner = make_hybrid_autotuner(
            _toy_adapter(),
            InMemoryTuningStore(),
            HybridTuningPolicy(
                portfolio="hybrid",
                orchestration="bandit",
                max_trials=32,
                model_warmup=8,
            ),
        )
        self.assertEqual(
            set(tuner.strategies),
            {"random", "coordinate_local", "gradient_boosted", "model_local"},
        )
        self.assertEqual(
            tuner.scheduler.minimum_pulls["coordinate_local"], 8
        )

    def test_gradient_boosted_model_learns_schedule_order(self) -> None:
        adapter = _toy_adapter()
        observations = []
        sequence = 0
        for x in range(5):
            for y in range(5):
                config = _ToyConfig(x, y)
                observation = evaluate_proposal(
                    adapter,
                    Proposal(config, "grid"),
                    session_id="training",
                    sequence=sequence,
                )
                sequence += 1
                observations.append(observation)
        model = GradientBoostedCostModel(
            n_estimators=24, ensembles=3, max_depth=3, min_leaf=2, seed=4
        )
        model.fit(observations)
        mean, uncertainty = model.predict(
            [adapter.features(_ToyConfig(3, 2)), adapter.features(_ToyConfig(0, 4))]
        )
        self.assertLess(float(mean[0]), float(mean[1]))
        self.assertTrue((uncertainty > 0).all())
        self.assertLess(model.parameter_count, 1_000_000)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.save(path)
            restored = GradientBoostedCostModel.load(path)
            restored_mean, _ = restored.predict(
                [adapter.features(_ToyConfig(3, 2)), adapter.features(_ToyConfig(0, 4))]
            )
            self.assertTrue((abs(restored_mean - mean) < 1e-10).all())

    def test_tree_prediction_matches_scalar_reference(self) -> None:
        adapter = _toy_adapter()
        observations = [
            evaluate_proposal(
                adapter,
                Proposal(_ToyConfig(x, y), "grid"),
                session_id="training",
                sequence=x * 5 + y,
            )
            for x in range(5)
            for y in range(5)
        ]
        model = GradientBoostedCostModel(
            n_estimators=6, ensembles=2, max_depth=3, min_leaf=2, seed=9
        )
        model.fit(observations)
        features = [adapter.features(_ToyConfig(x, y)) for x in range(5) for y in range(5)]
        matrix = model.vectorizer.transform(features)
        for ensemble in model.models:
            for tree in ensemble.trees:
                expected = []
                for row in matrix:
                    node_index = 0
                    while tree.nodes[node_index].feature >= 0:
                        node = tree.nodes[node_index]
                        node_index = (
                            node.left
                            if row[node.feature] <= node.threshold
                            else node.right
                        )
                    expected.append(tree.nodes[node_index].value)
                self.assertTrue((abs(tree.predict(matrix) - expected) < 1e-12).all())

    def test_shared_fit_state_prevents_duplicate_local_refit(self) -> None:
        adapter = _toy_adapter()
        history = SearchHistory(
            [
                evaluate_proposal(
                    adapter,
                    Proposal(_ToyConfig(x, y), "grid"),
                    session_id="training",
                    sequence=x * 5 + y,
                )
                for x in range(4)
                for y in range(5)
            ],
            adapter.context.identifier,
        )
        model = GradientBoostedCostModel(
            n_estimators=4, ensembles=1, max_depth=2, min_leaf=2, seed=3
        )
        from rtx.autotune.cost_model import GradientBoostedFeasibilityModel
        from rtx.autotune.strategies import CostModelLocalSearch

        feasibility = GradientBoostedFeasibilityModel(
            n_estimators=2, ensembles=1, max_depth=1, min_leaf=2
        )
        state = SharedModelFitState(model, feasibility)
        global_search = CostModelGuidedSearch(
            model=model,
            feasibility_model=feasibility,
            min_observations=4,
            pool_size=32,
            refit_interval=4,
            fit_state=state,
        )
        local_search = CostModelLocalSearch(
            model=model,
            feasibility_model=feasibility,
            refit_interval=4,
            candidate_cap=1,
            fit_state=state,
        )
        global_search.propose(adapter, history, random.Random(3), 1)
        fit_counts = (state.cost_fit_count, state.feasibility_fit_count)
        self.assertLessEqual(state.recommended_pool_size, global_search.pool_size)
        local_proposals = local_search.propose(adapter, history, random.Random(4), 1)
        self.assertEqual(local_proposals[0].metadata["candidate_pool_size"], 1)
        self.assertGreater(
            local_proposals[0].metadata["candidate_pool_population"], 1
        )
        self.assertEqual(
            (state.cost_fit_count, state.feasibility_fit_count), fit_counts
        )

    def test_sequential_cost_model_then_coordinate_search(self) -> None:
        adapter = _toy_adapter()
        store = InMemoryTuningStore()
        cost = CostModelGuidedSearch(
            model=GradientBoostedCostModel(
                n_estimators=12, ensembles=2, max_depth=2, min_leaf=2, seed=2
            ),
            min_observations=4,
            pool_size=64,
            refit_interval=2,
        )
        local = CoordinateLocalSearch(beam_width=2)
        result = AutotuneOrchestrator(
            adapter,
            store,
            [cost, local],
            SequentialScheduler(((cost.name, 12), (local.name, None))),
            TuningBudget(max_trials=24, time_budget_s=10),
            seed=3,
        ).tune()
        self.assertLessEqual(result.median_ms, 1.6)
        self.assertGreater(result.strategy_trials[cost.name], 0)
        self.assertGreater(result.strategy_trials[local.name], 0)
        self.assertTrue(any(event["kind"] == "strategy_selected" for event in store.events))

    def test_ucb_bandit_allocates_trials_to_multiple_search_arms(self) -> None:
        adapter = _toy_adapter()
        store = InMemoryTuningStore()
        random_search = RandomSearch()
        local = CoordinateLocalSearch()
        result = AutotuneOrchestrator(
            adapter,
            store,
            [random_search, local],
            UCB1Scheduler(exploration=1.0),
            TuningBudget(max_trials=12, time_budget_s=10),
            seed=9,
        ).tune()
        self.assertGreater(result.strategy_trials[random_search.name], 0)
        self.assertGreater(result.strategy_trials[local.name], 0)

    def test_apparent_incumbent_is_confirmed_and_timings_are_merged(self) -> None:
        adapter = _toy_adapter()
        adapter.axes = {"x": (0, 1), "y": (0,)}
        calls: dict[str, int] = {}

        def noisy(config: _ToyConfig) -> TrialOutcome:
            key = adapter.config_id(config)
            index = calls.get(key, 0)
            calls[key] = index + 1
            if config == _ToyConfig():
                score = 1.0
            else:
                score = (0.9, 1.2)[min(index, 1)]
            return TrialOutcome("ok", median_ms=score, timings_ms=[score])

        adapter.evaluator = noisy
        store = InMemoryTuningStore()
        AutotuneOrchestrator(
            adapter,
            store,
            [CoordinateLocalSearch(beam_width=1)],
            UCB1Scheduler(),
            TuningBudget(max_trials=2, time_budget_s=10),
            confirmation=ConfirmationPolicy(repeats=1),
        ).tune()
        contender = next(
            item for item in store.observations if item.config == _ToyConfig(1, 0)
        )
        self.assertEqual(contender.outcome.timings_ms, [0.9, 1.2])
        self.assertAlmostEqual(float(contender.outcome.median_ms), 1.05)
        self.assertEqual(contender.metadata["confirmation"]["repeats"], 1)

    def test_jsonl_store_is_append_only_and_resumable(self) -> None:
        adapter = _toy_adapter()
        with tempfile.TemporaryDirectory() as directory:
            store = JsonlTuningStore(Path(directory), fsync=False)
            first = AutotuneOrchestrator(
                adapter,
                store,
                [RandomSearch()],
                UCB1Scheduler(),
                TuningBudget(max_trials=5, time_budget_s=10),
                seed=1,
            ).tune()
            lines_before = store.observations_path.read_text().splitlines()
            second = AutotuneOrchestrator(
                adapter,
                store,
                [CoordinateLocalSearch()],
                UCB1Scheduler(),
                TuningBudget(max_trials=4, time_budget_s=10),
                seed=1,
                resume=True,
            ).tune()
            lines_after = store.observations_path.read_text().splitlines()
            self.assertGreater(len(lines_after), len(lines_before))
            self.assertEqual(first.context_id, second.context_id)
            self.assertTrue(all('"features"' in line for line in lines_after))
            events = [
                json.loads(line)
                for line in store.events_path.read_text().splitlines()
            ]
            starts = [
                event for event in events if event["kind"] == "candidate_started"
            ]
            self.assertTrue(starts)
            self.assertTrue(
                all(event["payload"].get("config") is not None for event in starts)
            )
            self.assertEqual(
                tuple(store.incomplete_candidates(adapter.context.identifier)), ()
            )

    def test_resumed_observations_can_count_toward_total_trial_cap(self) -> None:
        adapter = _toy_adapter()
        with tempfile.TemporaryDirectory() as directory:
            store = JsonlTuningStore(Path(directory), fsync=False)
            AutotuneOrchestrator(
                adapter,
                store,
                [RandomSearch()],
                UCB1Scheduler(),
                TuningBudget(max_trials=5, time_budget_s=10),
                seed=1,
                max_trials_includes_resumed=True,
            ).tune()
            before = len(list(store.records(adapter.context)))
            resumed = AutotuneOrchestrator(
                adapter,
                store,
                [RandomSearch()],
                UCB1Scheduler(),
                TuningBudget(max_trials=5, time_budget_s=10),
                seed=1,
                max_trials_includes_resumed=True,
            ).tune()
            after = len(list(store.records(adapter.context)))
            self.assertEqual(before, 5)
            self.assertEqual(after, before)
            self.assertEqual(resumed.evaluated_trials, 0)

    def test_cross_context_history_trains_but_does_not_deduplicate(self) -> None:
        first_adapter = _toy_adapter()
        second_adapter = _toy_adapter()
        second_adapter.context = KernelContext(
            family="toy",
            kernel_revision=1,
            workload={"m": 512, "n": 1536, "k": 1536},
            device={"multiprocessor_count": 84, "name": "other-device"},
        )
        with tempfile.TemporaryDirectory() as directory:
            store = JsonlTuningStore(Path(directory), fsync=False)
            AutotuneOrchestrator(
                first_adapter,
                store,
                [RandomSearch()],
                UCB1Scheduler(),
                TuningBudget(max_trials=4, time_budget_s=10),
                seed=7,
            ).tune()
            AutotuneOrchestrator(
                second_adapter,
                store,
                [RandomSearch()],
                UCB1Scheduler(),
                TuningBudget(max_trials=2, time_budget_s=10),
                seed=7,
                transfer_history=True,
            ).tune()
            initial_records = [
                record
                for record in store.records()
                if record["config_id"] == "0-0"
            ]
            self.assertEqual(len(initial_records), 2)
            self.assertNotEqual(
                initial_records[0]["context_id"], initial_records[1]["context_id"]
            )

    def test_legacy_import_is_idempotent_and_transfer_only(self) -> None:
        adapter = _toy_adapter()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "legacy.json"
            legacy.write_text(
                json.dumps(
                    {
                        "trials": {
                            "old": {
                                "config": {"x": 3, "y": 2},
                                "status": "ok",
                                "median_ms": 0.1,
                                "timings_ms": [0.1],
                                "recorded_at": "then",
                            },
                            "nested": {
                                "config": {"x": 2, "y": 2},
                                "outcome": {
                                    "status": "ok",
                                    "median_ms": 1.1,
                                    "timings_ms": [1.1],
                                },
                                "recorded_at": "later",
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            store = JsonlTuningStore(root / "unified", fsync=False)
            self.assertEqual(import_legacy_json_database(legacy, adapter, store), 2)
            self.assertEqual(import_legacy_json_database(legacy, adapter, store), 0)
            records = list(store.records())
            self.assertTrue(
                all(
                    record["context_id"] != adapter.context.identifier
                    for record in records
                )
            )
            self.assertEqual(
                {record["config_id"] for record in records}, {"3-2", "2-2"}
            )

    def test_current_kernel_families_share_the_adapter_contract(self) -> None:
        problem = MXFP8Problem(512, 1536, 1536)
        evaluator = lambda _config: TrialOutcome("ok", median_ms=1.0)
        adapters = (
            make_mxfp8_fwd_adapter(problem, evaluator),
            make_nvfp4_fwd_adapter(
                NVFP4Problem(problem.m, problem.n, problem.k), evaluator
            ),
            make_mxfp8_prequant_adapter(problem, evaluator),
            make_mxfp8_weight_prequant_adapter(problem, evaluator),
            make_mxfp8_fully_prequant_adapter(problem, evaluator),
            make_nvfp4_weight_prequant_adapter(
                NVFP4Problem(problem.m, problem.n, problem.k), evaluator
            ),
            make_nvfp4_fully_prequant_adapter(
                NVFP4Problem(problem.m, problem.n, problem.k), evaluator
            ),
            make_mxfp8_bwd_adapter(problem, evaluator),
        )
        self.assertEqual(
            {adapter.context.family for adapter in adapters},
            {
                "mxfp8_fused_fwd",
                "nvfp4_fused_fwd",
                "mxfp8_prequant_fwd",
                "mxfp8_weight_prequant_fwd",
                "mxfp8_fully_prequant_fwd",
                "nvfp4_weight_prequant_fwd",
                "nvfp4_fully_prequant_fwd",
                "mxfp8_bwd",
            },
        )
        for adapter in adapters:
            serialized = adapter.serialize(adapter.initial_config)
            restored = adapter.deserialize(serialized)
            self.assertEqual(
                adapter.config_id(restored),
                adapter.config_id(adapter.initial_config),
            )
            self.assertTrue(adapter.features(restored))
            self.assertTrue(adapter.coordinates())

    def test_inference_state_adapters_remove_inactive_quantizer_axes(self) -> None:
        problem = MXFP8Problem(512, 1536, 1536)
        evaluator = lambda _config: TrialOutcome("ok", median_ms=1.0)
        weight = make_mxfp8_weight_prequant_adapter(problem, evaluator)
        fully = make_mxfp8_fully_prequant_adapter(problem, evaluator)
        self.assertEqual(weight.context.tags["operand_state"], "weight_prequantized")
        self.assertEqual(fully.context.tags["operand_state"], "fully_prequantized")
        self.assertTrue(any(name.startswith("x_") for name in weight.coordinates()))
        self.assertFalse(any(name.startswith("w_") for name in weight.coordinates()))
        self.assertFalse(any("quant_launch" in name for name in weight.coordinates()))
        self.assertFalse(any(name.startswith("x_") for name in fully.coordinates()))
        self.assertFalse(any(name.startswith("w_") for name in fully.coordinates()))
        self.assertEqual(
            weight.features(weight.initial_config)[
                "derived.operand_state_weight_prequantized"
            ],
            1.0,
        )
        self.assertEqual(
            fully.features(fully.initial_config)[
                "derived.operand_state_fully_prequantized"
            ],
            1.0,
        )
        candidates = weight.sample(
            random.Random(17), 12, (weight.initial_config,)
        )
        self.assertTrue(candidates)
        self.assertTrue(all(weight.rejection(item) is None for item in candidates))

    def test_runtime_winner_cache_is_state_layout_and_device_specific(self) -> None:
        problem = MXFP8Problem(128, 128, 256)
        adapter = make_mxfp8_fully_prequant_adapter(
            problem, lambda _config: TrialOutcome("ok", median_ms=1.0)
        )
        key = RuntimeWinnerKey(
            adapter.context.family,
            problem,
            "hot",
            "synthetic-device",
            "x-row_major_w-row_major",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = save_runtime_winner(
                key,
                adapter.serialize(adapter.initial_config),
                config_id=adapter.config_id(adapter.initial_config),
                root=directory,
                median_ms=0.125,
            )
            self.assertTrue(path.exists())
            restored = load_runtime_winner(
                key,
                adapter.deserialize,
                root=directory,
                rejection=lambda config: config.rejection(problem),
            )
            self.assertEqual(restored, adapter.initial_config)
            wrong_layout = replace(key, variant="x-mma128_w-mma128")
            self.assertIsNone(
                load_runtime_winner(
                    wrong_layout, adapter.deserialize, root=directory
                )
            )

    def test_fused_random_walk_preserves_legality_and_explores(self) -> None:
        adapter = make_mxfp8_fwd_adapter(
            MXFP8Problem(128, 1536, 1536),
            lambda _config: TrialOutcome("ok", median_ms=1.0),
        )
        candidates = adapter.sample(
            random.Random(20260808),
            32,
            (adapter.initial_config,),
        )
        self.assertGreater(len(candidates), 1)
        self.assertTrue(all(adapter.rejection(config) is None for config in candidates))
        self.assertGreater(
            len({adapter.config_id(config) for config in candidates}),
            1,
        )

    def test_adapter_rejects_unknown_axes_before_a_long_run(self) -> None:
        problem = MXFP8Problem(128, 128, 128)
        with self.assertRaisesRegex(ValueError, "unknown fused-forward"):
            make_mxfp8_fwd_adapter(
                problem,
                lambda _config: TrialOutcome("ok", median_ms=1.0),
                axes={"stages": (1, 2)},
            )


if __name__ == "__main__":
    unittest.main()
