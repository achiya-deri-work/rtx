from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import random
import unittest

from rtx.autotune import (
    AdapterKernelTask,
    AskTellSession,
    Condition,
    ConditionalSearchSpace,
    CoordinateLocalSearch,
    DiscreteKernelAdapter,
    DiscreteParameter,
    EvaluationPlan,
    EvaluationStage,
    FunctionKernelTask,
    KernelContext,
    LocalTrialWorker,
    RandomSearch,
    SpaceConstraint,
    StageKind,
    StageResult,
    StagedTaskAdapter,
    TrialOutcome,
    TrialRequest,
    TrialResponse,
    UCB1Scheduler,
)


def _portable_space() -> ConditionalSearchSpace:
    return ConditionalSearchSpace(
        (
            DiscreteParameter("tile", (16, 32, 64), default=32),
            DiscreteParameter("persistent", (False, True), default=False),
            DiscreteParameter(
                "resident_tiles",
                (1, 2, 4),
                default=2,
                active_if=(Condition("persistent", "eq", True),),
            ),
            DiscreteParameter("load", ("vector", "tma"), default="vector"),
            DiscreteParameter(
                "stages",
                (1, 2, 3, 4),
                default=2,
                active_if=(Condition("load", "eq", "tma"),),
            ),
        ),
        constraints=(
            SpaceConstraint(
                "smem_budget",
                lambda config: (
                    "tile-stage product exceeds the portable test budget"
                    if int(config["tile"]) * int(config.get("stages", 1)) > 128
                    else None
                ),
                description="Synthetic stand-in for compiler resource constraints.",
            ),
        ),
    )


def _portable_adapter() -> StagedTaskAdapter[dict[str, object]]:
    space = _portable_space()
    plan = EvaluationPlan(
        (
            EvaluationStage("compile", StageKind.COMPILE, 0.2),
            EvaluationStage("check", StageKind.CORRECTNESS, 0.5),
            EvaluationStage("benchmark", StageKind.BENCHMARK, 1.0),
        )
    )

    def run(config, stage):
        tile = int(config["tile"])
        if stage.kind == StageKind.COMPILE:
            if tile == 64:
                return StageResult(
                    "backend_failure", error="synthetic compiler failure"
                )
            return StageResult(
                "ok",
                metrics={"compile_ms": 3.0, "registers": tile // 2},
                artifacts={"binary_key": f"toy-{tile}"},
            )
        if stage.kind == StageKind.CORRECTNESS:
            return StageResult("ok", metrics={"max_abs_error": 1e-3})
        latency = 0.2 + abs(tile - 32) / 64
        if config.get("load") == "tma":
            latency *= 0.9
        return StageResult(
            "ok", metrics={"latency_ms": latency, "timings_ms": [latency] * 3}
        )

    task = FunctionKernelTask(
        context=KernelContext(
            "external_custom_attention",
            1,
            {"batch": 2, "heads": 16, "sequence": 4096},
            device={"backend": "test", "compute_units": 80},
        ),
        space=space,
        plan=plan,
        stage_runner=run,
        feature_fn=lambda config: {
            "estimated_work": float(int(config["tile"]) * 4096)
        },
    )
    return StagedTaskAdapter(task)


class PortableAutotuneTests(unittest.TestCase):
    def test_conditional_space_is_normalized_serializable_and_legal(self) -> None:
        space = _portable_space()
        self.assertEqual(
            space.initial_config,
            {"tile": 32, "persistent": False, "load": "vector"},
        )
        persistent = space.update(space.initial_config, "persistent", True)
        self.assertEqual(persistent["resident_tiles"], 2)
        tma = space.update(persistent, "load", "tma")
        self.assertEqual(tma["stages"], 2)
        vector = space.update(tma, "load", "vector")
        self.assertNotIn("stages", vector)
        self.assertIsNotNone(
            space.rejection(
                {
                    "tile": 64,
                    "persistent": False,
                    "load": "tma",
                    "stages": 4,
                }
            )
        )
        samples = space.sample(random.Random(3), 32, [space.initial_config])
        self.assertTrue(samples)
        self.assertTrue(all(space.rejection(config) is None for config in samples))
        self.assertTrue(
            all(space.deserialize(space.serialize(config)) == config for config in samples)
        )
        json.dumps(space.as_dict())

    def test_staged_task_records_fidelity_artifacts_and_objective(self) -> None:
        adapter = _portable_adapter()
        outcome = adapter.evaluate(adapter.initial_config)
        self.assertTrue(outcome.successful)
        self.assertAlmostEqual(float(outcome.median_ms), 0.2)
        self.assertEqual(outcome.compile_ms, 3.0)
        self.assertEqual(outcome.max_abs_error, 1e-3)
        self.assertEqual(outcome.metadata["completed_fidelity"], 1.0)
        self.assertEqual(len(outcome.metadata["stages"]), 3)
        self.assertIn("derived.estimated_work", adapter.features(adapter.initial_config))
        partial = adapter.evaluate_at_fidelity(adapter.initial_config, 0.5)
        self.assertEqual(partial.status, "ok")
        self.assertIsNone(partial.median_ms)
        self.assertTrue(partial.metadata["partial"])
        compile_failure = adapter.evaluate(
            {"tile": 64, "persistent": False, "load": "vector"}
        )
        self.assertEqual(compile_failure.status, "compile_error")
        self.assertEqual(compile_failure.metadata["failed_stage"], "compile")

    def test_existing_adapter_can_be_exposed_as_a_portable_task(self) -> None:
        @dataclass(frozen=True)
        class Config:
            x: int = 0

        original = DiscreteKernelAdapter(
            context=KernelContext("existing_project_kernel", 7, {"n": 1024}),
            initial_config=Config(),
            axes={"x": (0, 1)},
            config_id_fn=lambda config: str(config.x),
            serialize_fn=asdict,
            deserialize_fn=lambda value: Config(**value),
            update_fn=lambda config, coordinate, value: replace(
                config, **{coordinate: int(value)}
            ),
            evaluator=lambda config: TrialOutcome(
                "ok", median_ms=1.0 - 0.2 * config.x
            ),
            rejection_fn=lambda _config: None,
            extra_features_fn=lambda config: {"reuse": float(config.x)},
        )
        portable = AdapterKernelTask(original)
        bridged = StagedTaskAdapter(portable)
        self.assertEqual(bridged.config_id(Config(1)), original.config_id(Config(1)))
        self.assertAlmostEqual(float(bridged.evaluate(Config(1)).median_ms), 0.8)
        self.assertEqual(bridged.features(Config(1))["derived.reuse"], 1.0)

    def test_ask_tell_requests_round_trip_and_complete_out_of_order(self) -> None:
        adapter = _portable_adapter()
        session = AskTellSession(
            adapter,
            [RandomSearch(), CoordinateLocalSearch()],
            UCB1Scheduler(exploration=0.5),
            seed=4,
        )
        worker = LocalTrialWorker(adapter, {"worker_id": "external-gpu-0"})

        initial = session.ask(4)
        self.assertEqual(len(initial), 1)
        wire_request = json.loads(json.dumps(initial[0].as_dict()))
        decoded = TrialRequest.from_dict(adapter, wire_request)
        wire_response = json.loads(json.dumps(worker.evaluate(decoded).as_dict()))
        session.tell(TrialResponse.from_dict(wire_response))

        requests = session.ask(2)
        self.assertEqual(len(requests), 2)
        self.assertEqual(len({request.config_id for request in requests}), 2)
        state = json.loads(json.dumps(session.state_dict()))
        restored = AskTellSession.from_state_dict(
            adapter,
            [RandomSearch(), CoordinateLocalSearch()],
            UCB1Scheduler(exploration=0.5),
            state,
        )
        self.assertEqual(set(restored.pending), set(session.pending))
        responses = [worker.evaluate(request) for request in requests]
        for response in reversed(responses):
            observation = restored.tell(response)
            self.assertEqual(
                observation.metadata["worker"]["worker_id"], "external-gpu-0"
            )
        self.assertFalse(restored.pending)
        self.assertEqual(len(restored.history.current), 3)
        json.dumps(restored.state_dict())

    def test_expired_ask_tell_leases_are_reclaimable(self) -> None:
        session = AskTellSession(
            _portable_adapter(), [RandomSearch()], UCB1Scheduler(), seed=1
        )
        request = session.ask(1, lease_s=0.01)[0]
        expired = session.reclaim_expired(now=request.lease_expires_at_monotonic + 1)
        self.assertEqual([item.request_id for item in expired], [request.request_id])
        self.assertFalse(session.pending)

    def test_partial_trial_can_be_promoted_to_full_fidelity(self) -> None:
        adapter = _portable_adapter()
        session = AskTellSession(
            adapter, [RandomSearch()], UCB1Scheduler(), seed=2
        )
        worker = LocalTrialWorker(adapter)
        partial_request = session.ask(1, fidelity=0.5)[0]
        partial = session.tell(worker.evaluate(partial_request))
        self.assertFalse(partial.successful)
        self.assertTrue(partial.outcome.metadata["partial"])
        promoted_request = session.promote(partial.config_id, 1.0)
        promoted = session.tell(worker.evaluate(promoted_request))
        self.assertTrue(promoted.successful)
        self.assertEqual(promoted.metadata["promoted_from_fidelity"], 0.5)


if __name__ == "__main__":
    unittest.main()
