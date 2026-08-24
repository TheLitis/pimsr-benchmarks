"""Occam baseline must recover simple models from clean data."""

import numpy as np
import pytest
from pimsr_forward.mt1d import default_period_band, mt1d_response

import pimsr_benchmarks.occam1d as occam1d_module
from pimsr_benchmarks.metrics import profile_rmse
from pimsr_benchmarks.occam1d import default_mesh, occam1d_invert

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


def test_reported_nrms_scores_the_returned_model_after_final_step():
    rho_a, phase = mt1d_response(
        np.array([50.0, 500.0, 10.0]), np.array([1000.0, 5000.0]), PERIODS
    )
    observed_log_rho = np.log10(rho_a)
    result = occam1d_invert(
        observed_log_rho,
        phase,
        PERIODS,
        err_log10_rho_a=0.05,
        max_iterations=1,
    )
    weights = np.concatenate(
        [
            np.full(PERIODS.size, 1.0 / 0.05),
            np.full(PERIODS.size, 1.0 / (2.0 / 45.0)),
        ]
    )
    residual = occam1d_module._weighted_data_residual(
        observed_log_rho,
        phase,
        occam1d_module._forward(result.log10_rho, result.thicknesses, PERIODS),
        weights,
    )
    actual_nrms = float(np.sqrt(np.mean(residual**2)))

    assert result.n_iterations == 1
    assert result.nrms == pytest.approx(actual_nrms)
    assert result.nrms_history[-1] == pytest.approx(actual_nrms)
    assert result.converged is (actual_nrms <= 1.0)


def test_occam_objective_is_invariant_to_180_degree_phase_branch():
    rho_a, phase = mt1d_response(
        np.array([50.0, 500.0, 10.0]), np.array([1000.0, 5000.0]), PERIODS
    )
    baseline = occam1d_invert(np.log10(rho_a), phase, PERIODS, max_iterations=5)
    shifted = occam1d_invert(
        np.log10(rho_a), phase + 180.0, PERIODS, max_iterations=5
    )

    np.testing.assert_allclose(shifted.log10_rho, baseline.log10_rho)
    np.testing.assert_allclose(shifted.nrms_history, baseline.nrms_history)


def test_occam_jacobian_wraps_finite_difference_across_phase_cut(monkeypatch):
    def branch_crossing_forward(model, _thicknesses, _periods):
        phase = (179.0 + 20_000.0 * model[0]) % 180.0
        return np.array([0.0, phase / 45.0])

    monkeypatch.setattr(occam1d_module, "_forward", branch_crossing_forward)
    model = np.array([0.0])
    periods = np.array([1.0])
    baseline = branch_crossing_forward(model, np.array([]), periods)
    jacobian = occam1d_module._jacobian(model, np.array([]), periods, baseline)

    assert jacobian[0, 0] == pytest.approx(0.0)
    assert jacobian[1, 0] == pytest.approx(20_000.0 / 45.0)


@pytest.mark.parametrize(
    ("updates", "error_type"),
    [
        ({"obs_log10_rho_a": np.array([]), "obs_phase": np.array([]), "periods": np.array([])}, ValueError),
        ({"obs_phase": np.zeros(2)}, ValueError),
        ({"periods": np.array([1.0, np.nan, 10.0])}, ValueError),
        ({"obs_log10_rho_a": np.array([2.0, np.inf, 2.0])}, ValueError),
        ({"err_log10_rho_a": np.nan}, ValueError),
        ({"err_phase_deg": 0.0}, ValueError),
        ({"max_iterations": 0}, ValueError),
        ({"max_iterations": 1.5}, TypeError),
        ({"mu_cool": 1.1}, ValueError),
        ({"thicknesses": np.array([100.0, -10.0])}, ValueError),
    ],
)
def test_occam_rejects_invalid_contract_inputs(updates, error_type):
    inputs = {
        "obs_log10_rho_a": np.full(3, 2.0),
        "obs_phase": np.full(3, 45.0),
        "periods": np.array([1.0, 10.0, 100.0]),
    }
    inputs.update(updates)
    with pytest.raises(error_type):
        occam1d_invert(**inputs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_cells": 0},
        {"n_cells": True},
        {"z_min": np.nan},
        {"z_min": 10.0, "z_max": 10.0},
    ],
)
def test_default_mesh_rejects_invalid_geometry(kwargs):
    with pytest.raises((TypeError, ValueError)):
        default_mesh(**kwargs)
