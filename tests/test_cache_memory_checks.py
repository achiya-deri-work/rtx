"""The allocator guard detects live growth without rejecting reserve changes."""
from contextlib import ExitStack
from unittest.mock import Mock, patch
import unittest

from cache_memory_checks import assert_stable_cuda_memory


class CacheMemoryGuardTests(unittest.TestCase):
    def _run(self, allocated):
        step = Mock()
        with ExitStack() as stack:
            for name in ('synchronize', 'reset_peak_memory_stats'):
                stack.enter_context(patch('torch.cuda.' + name))
            stack.enter_context(patch('torch.cuda.memory_allocated', side_effect=allocated))
            stack.enter_context(patch('torch.cuda.memory_reserved', side_effect=[1000, 2000, 3000]))
            stack.enter_context(patch('torch.cuda.max_memory_allocated', return_value=500))
            stack.enter_context(patch('torch.cuda.get_device_name', return_value='test'))
            stack.enter_context(patch('cache_memory_checks.gc.collect'))
            samples = assert_stable_cuda_memory(self, step, label='guard', iterations=10)
        self.assertEqual(step.call_count, 13)
        return samples

    def test_reserved_growth_is_not_live_growth(self):
        self.assertEqual(len(self._run([100, 100, 100])), 3)

    def test_even_transient_live_growth_is_rejected(self):
        with self.assertRaises(AssertionError):
            self._run([100, 101, 100])


if __name__ == '__main__':
    unittest.main()
