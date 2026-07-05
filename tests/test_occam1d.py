"""Occam baseline must recover simple models from clean data."""

import numpy as np
from pimsr_forward.mt1d import default_period_band, mt1d_response

from pimsr_benchmarks.occam1d import occam1d_invert
from pimsr_benchmarks.metrics import profile_rmse


PERIODS = default_period_band(24)


def test_halfspace_recovery():
    rho_a, phase = mt1d_response(np.array([100.0]), np.array([]), PERIODS)
    res = occam1d_invert(np.log10(rho_a), phase, PERIODS)
    assert res.converged
    # every cell should be close to 2.0 (=log10 100)
    np.testing.assert_allclose(res.log10_rho, 2.0, atol=0.15)


def test_two_layer_recovery():
    true_rho = np.array([30.0, 1000.0])
    thick = np.array([2000.0])
    rho_a, phase = mt1d_response(true_rho, thick, PERIODS)
    res = occam1d_invert(np.log10(rho_a), phase, PERIODS)
    assert res.converged

    grid = np.logspace(1.0, np.log10(6.0e4), 64)
    prof = res.profile_on_grid(grid)
    true_prof = np.where(grid <= 2000.0, np.log10(30.0), 3.0)
    # smooth inversion blurs the interface; demand rough agreement
    assert profile_rmse(prof, true_prof) < 0.6
    # shallow cells conductive, deep cells resistive
    assert prof[grid < 1000.0].mean() < 2.0
    assert prof[grid > 20000.0].mean() > 2.3


def test_nrms_decreases():
    rho_a, phase = mt1d_response(
        np.array([50.0, 500.0, 10.0]), np.array([1000.0, 5000.0]), PERIODS
    )
    res = occam1d_invert(np.log10(rho_a), phase, PERIODS)
    assert res.nrms_history[-1] < res.nrms_history[0]
