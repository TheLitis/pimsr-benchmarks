import numpy as np
import pytest

from pimsr_benchmarks.statistics import bootstrap_ci, calibration_summary


def test_bootstrap_is_reproducible():
    values = np.arange(10.0)
    assert bootstrap_ci(values, n_resamples=500, seed=7) == bootstrap_ci(values, n_resamples=500, seed=7)


def test_perfect_gaussian_style_calibration_shape():
    mean = np.zeros((4, 3, 2))
    sigma = np.ones_like(mean)
    truth = np.zeros_like(mean)
    out = calibration_summary(mean, sigma, truth, scenario=np.array([0, 0, 1, 1]))
    assert out["coverage"]["95"] == 1.0
    assert len(out["coverage68_by_depth"]) == 3
    assert set(out["coverage68_by_scenario"]) == {"0", "1"}


def test_statistics_reject_invalid_inputs():
    with pytest.raises(ValueError):
        bootstrap_ci([])
    with pytest.raises(ValueError):
        calibration_summary(np.zeros(1), np.zeros(1), np.zeros(1))
