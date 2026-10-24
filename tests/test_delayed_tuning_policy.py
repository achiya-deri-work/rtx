"""Delayed tuning must measure and identify the requested numerical policy."""
from contextlib import nullcontext
from dataclasses import asdict
from itertools import product
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import torch
from rtx.configs.nvfp4 import NVFP4ScaleConfig, NVFP4Problem
from rtx.nvfp4_inference_autotune import delayed_scale_variant


class DelayedTuningPolicyTests(unittest.TestCase):
    def test_adapter_partitions_all_policies_and_accepts_legacy_tags(self):
        from rtx.autotune.adapters import make_nvfp4_delayed_adapter
        problem = NVFP4Problem(128, 128, 128)
        def build(**kwargs):
            return make_nvfp4_delayed_adapter(problem, lambda c: None, device={}, **kwargs).context
        identities = set()
        for mode, length, algo in product(
                ('power2', 'exact', 'exact_four_over_six'),
                (1, 4, 16, 64), ('most_recent', 'window_max')):
            policy = NVFP4ScaleConfig(mode, length, algo)
            context = build(scale_policy=policy)
            identities.add(context.identifier)
            self.assertEqual(context.identifier,
                             build(tags={'delayed_scale_policy': asdict(policy)}).identifier)
        self.assertEqual(len(identities), 24)
        self.assertEqual(build().identifier, build(scale_policy=NVFP4ScaleConfig()).identifier)
        self.assertNotIn('delayed_scale_policy', build().tags)
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            build(scale_policy=NVFP4ScaleConfig('exact'),
                  tags={'delayed_scale_policy': asdict(NVFP4ScaleConfig())})
        with self.assertRaises(TypeError):
            build(scale_policy={})

    def test_all_policies_have_distinct_identities(self):
        variants = {
            delayed_scale_variant(NVFP4ScaleConfig(mode, length, algo))
            for mode, length, algo in product(
                ('power2', 'exact'), (1, 4, 16, 64), ('most_recent', 'window_max'))
        }
        self.assertEqual(len(variants), 16)
        self.assertEqual(delayed_scale_variant(), 'default')
        self.assertEqual(delayed_scale_variant(NVFP4ScaleConfig()), 'default')
        with self.assertRaises(TypeError):
            delayed_scale_variant({})

    def test_harness_policy_is_set_before_reference_construction(self):
        from rtx.nvfp4_inference_experiments import (
            NVFP4DelayedBenchmarkHarness, NVFP4DynamicBenchmarkHarness)
        policy = NVFP4ScaleConfig('exact', 4, 'most_recent')
        observed = []
        def initialize(instance, *args, **kwargs):
            observed.append(instance.scale_policy)
        with patch.object(NVFP4DynamicBenchmarkHarness, '__init__', initialize):
            NVFP4DelayedBenchmarkHarness(scale_policy=policy)
            NVFP4DelayedBenchmarkHarness()
        self.assertEqual(observed, [policy, NVFP4ScaleConfig()])

    def test_non_delayed_policy_fails_before_cuda(self):
        from rtx.nvfp4_inference_autotune import tune_nvfp4_inference_state
        with self.assertRaisesRegex(ValueError, 'only for delayed'):
            tune_nvfp4_inference_state(NVFP4Problem(128, 128, 128),
                                      state='dynamic', scale_policy=NVFP4ScaleConfig())

    def test_runner_receives_policy_and_allocates_matching_histories(self):
        import rtx.nvfp4_inference_experiments as experiments
        from rtx.configs.nvfp4 import NVFP4DynamicConfig
        harness = object.__new__(experiments.NVFP4DelayedBenchmarkHarness)
        harness.scale_policy = NVFP4ScaleConfig('exact', 64, 'most_recent')
        harness.problem = NVFP4Problem(128, 128, 128)
        harness.device = torch.device('cpu')
        harness.x = torch.zeros(128, 128)
        harness.weight = torch.zeros(128, 128)
        with patch.object(experiments, '_make_delayed_dynamic_runner', return_value=object()) as build:
            runner = harness._build_runner(NVFP4DynamicConfig())
        self.assertEqual(build.call_args.args[2], harness.scale_policy)
        for value in (runner.x_history, runner.weight_history,
                      runner.next_x_history, runner.next_weight_history):
            self.assertEqual(value.shape, (64,))
            self.assertEqual(value.dtype, torch.float32)

    def test_resolver_partitions_policy_and_forwards_it_to_tuner(self):
        import rtx.fp4 as fp4
        from rtx.autotune import AutotunePolicy
        policy = NVFP4ScaleConfig('exact', 4, 'most_recent')
        public = fp4.NVFP4ForwardConfig(policy=policy)
        key = fp4._intern_forward_config(public)
        request = SimpleNamespace(mode='balanced', policy=AutotunePolicy(),
                                  cache_dir=None, dynamic=None)
        with patch.dict(fp4._AUTOTUNE_REQUESTS, {'policy-test': request}, clear=True), \
             patch.dict(fp4._DYNAMIC_CONFIG_SELECTIONS, {}, clear=True), \
             patch('rtx.autotune.winners.runtime_winner_key', return_value=object()) as identity, \
             patch('rtx.autotune.winners.load_runtime_winner', return_value=None) as load, \
             patch('rtx.autotune.winners.runtime_tuning_lock', return_value=nullcontext()), \
             patch('rtx.nvfp4_inference_autotune.tune_nvfp4_inference_state',
                   side_effect=RuntimeError('tuner reached')) as tune:
            with self.assertRaisesRegex(RuntimeError, 'tuner reached'):
                fp4._materialized_dynamic_config_key(
                    NVFP4Problem(128, 128, 128), torch.device('cpu'), 'policy-test',
                    family='nvfp4_delayed_fwd', delayed_forward_config_key=key)
        self.assertEqual(identity.call_count, 1)  # No unrelated block-policy fallback.
        self.assertEqual(identity.call_args.kwargs['variant'], delayed_scale_variant(policy))
        self.assertEqual(load.call_count, 2)  # Outside and inside the tuning lock.
        self.assertEqual(tune.call_args.kwargs['scale_policy'], policy)


if __name__ == '__main__':
    unittest.main()
