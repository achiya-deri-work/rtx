"""CPU-only offline selection contracts; no model/dataset qualification."""

from dataclasses import FrozenInstanceError
import json
import unittest

from rtx.autotune.accuracy import coordinate_search, select_accuracy_policy, select_guard


class AccuracySelectionTests(unittest.TestCase):
    @staticmethod
    def objective(state):
        first = {"max": 3., "mse": 1.}
        second = {"max": 2., "mse": 1.}
        return (first[state["a"]] * second[state["b"]] - 2.5) ** 2

    def test_nonlocal_full_state_and_baseline_guard_fallback(self):
        calls = []
        def evaluate(state):
            calls.append(dict(state))
            self.assertEqual(set(state), {"a", "b"})
            return self.objective(state)
        def guard(state):
            return 0. if set(state.values()) == {"mse"} else 1.
        result = select_accuracy_policy(["a", "b"], ["max", "mse"], evaluate,
            guard_evaluate=guard, baselines=("max", "mse"))
        self.assertEqual(result.search_state, {"a": "max", "b": "mse"})
        self.assertEqual(result.search_score, .25)
        self.assertEqual(result.selected_state, {"a": "mse", "b": "mse"})
        self.assertEqual(result.guard_choice, "baseline:mse")
        self.assertEqual(len(calls), 6)
        self.assertEqual(len(result.trials), 9)
        self.assertEqual(result.guard_scores, {"baseline:max": 1., "baseline:mse": 0., "search": 1.})

    def test_strict_improvements_and_baseline_order_ties(self):
        result = select_accuracy_policy(["a"], ["x", "y"], lambda _: 1.,
            guard_evaluate=lambda _: 1., baselines=("y", "x"), max_passes=2)
        self.assertEqual(result.search_state, {"a": "y"})
        self.assertEqual(result.selected_state, {"a": "y"})
        self.assertEqual(result.guard_choice, "baseline:y")
        self.assertEqual(len(result.trials), 2 + 2 * 2 + 3)

    def test_callback_progress_and_serialization_cannot_mutate_evidence(self):
        def evaluate(state):
            score = 0. if state["a"] == "y" else 1.
            state.clear()
            state["injected"] = "bad"
            return score
        def progress(trial):
            trial["full_state"].clear()
            trial["score"] = -1.
        result = select_accuracy_policy(["a"], ["x", "y"], evaluate,
            guard_evaluate=evaluate, baselines=("x",), progress=progress)
        self.assertEqual(result.selected_state, {"a": "y"})
        self.assertEqual(result.guard_choice, "search")
        self.assertTrue(all(set(t["full_state"]) == {"a"} for t in result.trials))
        self.assertTrue(all(t["score"] >= 0 for t in result.trials))
        with self.assertRaises(TypeError):
            result.selected_state["a"] = "x"
        with self.assertRaises(TypeError):
            result.trials[0]["full_state"]["a"] = "x"
        with self.assertRaises(FrozenInstanceError):
            result.search_score = 9.
        payload = result.to_dict()
        json.dumps(payload, allow_nan=False)
        payload["selected_state"].clear()
        payload["trials"][0]["full_state"].clear()
        self.assertEqual(result.selected_state, {"a": "y"})
        self.assertEqual(result.trials[0]["full_state"], {"a": "x"})

    def test_candidate_names_do_not_collide_with_guard_names(self):
        result = select_accuracy_policy(["a"], ["search", "baseline:search"], lambda _: 0.,
            guard_evaluate=lambda _: 0., baselines=("search", "baseline:search"))
        self.assertEqual(result.guard_choice, "baseline:search")
        self.assertEqual(set(result.guard_scores), {"baseline:search", "baseline:baseline:search", "search"})

    def test_invalid_inputs_rejected_before_callbacks(self):
        def fail(_):
            self.fail("invalid arguments must not invoke evaluator")
        cases = [
            {"groups": []}, {"groups": ["a", "a"]}, {"groups": "a"},
            {"groups": [""]}, {"groups": [1]}, {"groups": None}, {"candidates": []},
            {"candidates": ["x", "x"]}, {"candidates": [" "]},
            {"baselines": ()}, {"baselines": ("z",)}, {"baselines": ("x", "x")},
            {"max_passes": 0}, {"max_passes": -1}, {"max_passes": 1.5},
            {"max_passes": True}, {"evaluate": None}, {"guard_evaluate": None},
            {"progress": 2},
        ]
        for overrides in cases:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                args = dict(groups=["a"], candidates=["x"], evaluate=fail,
                            guard_evaluate=fail, baselines=("x",))
                select_accuracy_policy(**{**args, **overrides})

    def test_nonfinite_negative_and_nonnumeric_scores_rejected(self):
        for invalid in (float("nan"), float("inf"), -1., None, True, complex(1, 2)):
            for bad_stage in ("search", "guard"):
                with self.subTest(invalid=invalid, stage=bad_stage), self.assertRaises(ValueError):
                    select_accuracy_policy(["a"], ["x"],
                        lambda _: invalid if bad_stage == "search" else 0.,
                        guard_evaluate=lambda _: invalid if bad_stage == "guard" else 0.,
                        baselines=("x",))

    def test_compatibility_helpers(self):
        state, score, records = coordinate_search(["a", "b"], ["max", "mse"], self.objective)
        self.assertEqual(state, {"a": "max", "b": "mse"})
        self.assertEqual(score, .25)
        self.assertEqual(len(records), 6)
        self.assertEqual(select_guard({"max": .2, "mse": .1, "selected": .3}), "mse")
        self.assertEqual(select_guard({"max": .1, "mse": .1, "selected": .1}), "max")
        with self.assertRaises(ValueError):
            select_guard({"max": 1., "selected": 0.})


if __name__ == "__main__":
    unittest.main()
