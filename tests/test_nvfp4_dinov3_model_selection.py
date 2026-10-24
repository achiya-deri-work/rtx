import unittest
from types import SimpleNamespace

import torch

from benchmarks.select_nvfp4_dinov3_model import (
    PACKED_BUFFERS, coordinate_search, copy_packed_buffers,
    normalized_feature_score, select_guard,
)


class WholeModelSelectionCpuTests(unittest.TestCase):
    def test_coordinate_search_scores_nonlocal_interactions_and_strict_decreases(self):
        calls = []
        # Two serial transformations. Independent local preferences (mse,max)
        # would output2; model-level selection instead finds(max,mse), output3.
        first = {"max": 3., "mse": 1.}
        second = {"max": 2., "mse": 1.}
        def evaluate(state):
            self.assertEqual(set(state), {"blocks.0", "blocks.1"})
            calls.append(dict(state))
            return (first[state["blocks.0"]] * second[state["blocks.1"]] - 3.) ** 2
        state, score, records = coordinate_search(["blocks.0", "blocks.1"], ["max", "mse"], evaluate)
        self.assertEqual(state, {"blocks.0": "max", "blocks.1": "mse"})
        self.assertEqual(score, 0.)
        self.assertEqual(len(calls), 6)
        self.assertEqual(records[-1]["full_state"], state)
        repeat = coordinate_search(["blocks.0", "blocks.1"], ["max", "mse"], evaluate)
        self.assertEqual((state, score, records), repeat)
        # No acceptance on equality; global-max order is safe tie-break.
        state, _, _ = coordinate_search(["blocks.0"], ["mse", "max"], lambda _: 1.)
        self.assertEqual(state, {"blocks.0": "max"})
        with self.assertRaisesRegex(ValueError, "finite"):
            coordinate_search(["blocks.0"], ["max"], lambda _: float("nan"), initial_states=("max",))

    def test_packed_copies_preserve_identity_and_validate_before_mutation(self):
        core = SimpleNamespace(weight_data=torch.zeros(2, 16, dtype=torch.uint8), weight_block_scales=torch.ones(2, 2, dtype=torch.float8_e4m3fn), weight_tensor_scale=torch.tensor(1.))
        identities = {name: id(getattr(core, name)) for name in PACKED_BUFFERS}
        buffers = {"weight_data": torch.full((2, 16), 2, dtype=torch.uint8), "weight_block_scales": torch.full((2, 2), 2, dtype=torch.float8_e4m3fn), "weight_tensor_scale": torch.tensor(2.)}
        copy_packed_buffers(core, buffers)
        self.assertEqual(identities, {name: id(getattr(core, name)) for name in PACKED_BUFFERS})
        self.assertEqual(float(core.weight_tensor_scale), 2.)
        bad = {**buffers, "weight_data": torch.ones(2, 16, dtype=torch.uint8), "weight_tensor_scale": torch.ones(1)}
        with self.assertRaisesRegex(ValueError, "shape/dtype"):
            copy_packed_buffers(core, bad)
        self.assertEqual(int(core.weight_data.sum()), 64)
        with self.assertRaisesRegex(ValueError, "names"):
            copy_packed_buffers(core, {"weight_data": buffers["weight_data"]})
        with self.assertRaisesRegex(ValueError, "shape/dtype"):
            copy_packed_buffers(core, {**buffers, "weight_data": buffers["weight_data"].float()})

    def test_score_equal_weights_and_energy_normalization(self):
        reference = (torch.ones(2, 4), torch.full((2, 3, 4), 10.))
        candidate = (torch.full((2, 4), 2.), torch.full((2, 3, 4), 30.))
        self.assertEqual(normalized_feature_score(candidate, reference), 2.5)
        self.assertEqual(normalized_feature_score(reference, reference), 0.)
        with self.assertRaisesRegex(ValueError, "finite matching"):
            normalized_feature_score((torch.full((2, 4), float("nan")), reference[1]), reference)

    def test_compiled_guard_can_fallback_to_baseline_and_retains_ties(self):
        self.assertEqual(select_guard({"max": .2, "mse": .1, "selected": .3}), "mse")
        self.assertEqual(select_guard({"max": .1, "mse": .1, "selected": .1}), "max")
        self.assertEqual(select_guard({"max": .2, "mse": .3, "selected": .1}), "selected")
        with self.assertRaisesRegex(ValueError, "finite"):
            select_guard({"max": 0., "mse": 1., "selected": float("nan")})


if __name__ == "__main__":
    unittest.main()
