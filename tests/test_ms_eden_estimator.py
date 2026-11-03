import pytest
import torch
from benchmarks.study_ms_eden_estimator import Moments


def test_moments_deterministic_error_is_not_noise():
    moment=Moments(torch.ones(4))
    for _ in range(4):
        moment.add(torch.full((4,),1.25))
    value=moment.report()
    assert value["mean_error_nrms"] == .25
    assert value["mean_standard_error_nrms"] == 0.
    assert value["first_draw_nrms"] == .25
    assert value["per_draw_nrms_rms"] == .25
    assert value["bias_squared_noise_corrected_nrms2"] == .0625


def test_moments_separates_finite_mean_and_variance():
    moment=Moments(torch.ones(2))
    for value in (0.,2.):
        moment.add(torch.full((2,),value))
    result=moment.report()
    assert result["mean_error_nrms"] == 0.
    assert result["mean_standard_error_nrms"] == 1.
    assert result["first_draw_nrms"] == 1.
    assert result["per_draw_nrms_rms"] == 1.
    assert result["bias_squared_noise_corrected_nrms2"] == -1.


def test_moments_rejects_nonfinite_and_too_few():
    moment=Moments(torch.zeros(2))
    with pytest.raises(ValueError):
        moment.report()
    with pytest.raises(ValueError):
        moment.add(torch.tensor([float("nan"),0.]))
