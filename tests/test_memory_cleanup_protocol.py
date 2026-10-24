"""CPU coverage: cleanup settling must never hide persistent allocation growth."""
from unittest.mock import patch

import pytest
from tests import cache_memory_checks as checks


def observe(values, *, observed=120, timeout=2.5):
    clock=[0.]
    with patch.object(checks.time,'monotonic',side_effect=lambda:clock[0]), \
         patch.object(checks.time,'sleep',side_effect=lambda _:clock.__setitem__(0,clock[0]+1)), \
         patch.object(checks.gc,'collect'), \
         patch.object(checks.torch.cuda,'synchronize'), \
         patch.object(checks.torch.cuda,'memory_allocated',side_effect=values) as read, \
         patch.object(checks.torch.cuda,'empty_cache',side_effect=AssertionError('cache eviction')):
        result=checks.wait_for_cuda_allocation_cleanup(100,observed=observed,timeout_s=timeout)
        return result,read.call_count


def test_delayed_cleanup_preserves_raw_samples_and_exact_bound():
    assert observe([110,100]) == ([120,110,100],2)


def test_persistent_growth_is_not_accepted_or_smoothed():
    samples,calls=observe([140,160,180])
    assert samples == [120,140,160,180] and calls == 3
    assert samples[-1] > 100  # The deployment test's unchanged bound must fail.


def test_stable_sample_needs_no_wait_or_new_read():
    assert observe([],observed=100) == ([100],0)


def test_zero_deadline_does_not_claim_cleanup():
    assert observe([],timeout=0) == ([120],0)


@pytest.mark.parametrize('args',[{'timeout_s':-1},{'poll_s':0}])
def test_invalid_wait_policy(args):
    with pytest.raises(ValueError):
        checks.wait_for_cuda_allocation_cleanup(100,observed=120,**args)
