"""Observed-data and metric contract tests for the 2D benchmark."""

import numpy as np
import pytest

from pimsr_benchmarks.hybrid2d import (
    _interpolate_profile_field,
    _make_profiled_static_shift_misfit,
    _pack_profiled_mode_data,
    _profile_log10_rho_residual,
    multimode_nrms,
    profile_geometry_metadata,
    refine_section_2d,
    section_nrms_2d,
)


def _empty_modes(shape=(2, 1)):
    zeros = np.zeros(shape, dtype=float)
    false = np.zeros(shape, dtype=bool)
    return {
        "lr_te": zeros.copy(),
        "ph_te": zeros.copy(),
        "mask_te": false.copy(),
        "lr_tm": zeros.copy(),
        "ph_tm": zeros.copy(),
        "mask_tm": false.copy(),
    }


def test_metric_weights_rho_and_phase_equally():
    obs = _empty_modes()
    obs["lr_te"][:, 0] = [1.0, 3.0]
    obs["ph_te"][:, 0] = [2.0, 2.0]
    obs["mask_te"][:] = True
    sim = {key: np.zeros((2, 1)) for key in ("lr_te", "ph_te", "lr_tm", "ph_tm")}

    # Static-shift removal turns rho residuals into [-1, +1].  With errors
    # rho=1 and phase=2, all four scalar normalised residuals have magnitude
    # one, so the corrected nRMS is exactly one.
    score = multimode_nrms(obs, sim, rho_error=1.0, phase_error=2.0)
    assert score == pytest.approx(1.0)


def test_metric_excludes_unmasked_periods_from_score_and_static_shift():
    obs = _empty_modes()
    obs["lr_te"][:, 0] = [1.0, 1.0e6]
    obs["ph_te"][:, 0] = [2.0, 1.0e6]
    obs["mask_te"][:, 0] = [True, False]
    sim = {key: np.zeros((2, 1)) for key in ("lr_te", "ph_te", "lr_tm", "ph_tm")}

    # One valid rho datum is entirely removed as a static shift; the phase
    # contributes one unit of squared residual and rho contributes zero.
    score = multimode_nrms(obs, sim, rho_error=1.0, phase_error=2.0)
    assert score == pytest.approx(np.sqrt(0.5))


def test_metric_gives_te_and_tm_equal_weight():
    obs = _empty_modes()
    obs["ph_te"][:, 0] = 2.0
    obs["mask_te"][:] = True
    obs["mask_tm"][:] = True
    sim = {key: np.zeros((2, 1)) for key in ("lr_te", "ph_te", "lr_tm", "ph_tm")}

    # TE mode MSE is 0.5, TM mode MSE is 0.0, then the two mode MSEs are
    # averaged before the square root.
    score = multimode_nrms(obs, sim, rho_error=1.0, phase_error=2.0)
    assert score == pytest.approx(0.5)


def test_metric_uses_180_degree_periodic_phase_residual():
    obs = _empty_modes(shape=(1, 1))
    obs["ph_te"][0, 0] = 179.0
    obs["mask_te"][0, 0] = True
    sim = {
        "lr_te": np.zeros((1, 1)),
        "ph_te": np.full((1, 1), -1.0),
        "lr_tm": np.zeros((1, 1)),
        "ph_tm": np.zeros((1, 1)),
    }

    assert multimode_nrms(obs, sim) == pytest.approx(0.0)


def test_metric_requires_explicit_boolean_masks():
    obs = _empty_modes()
    del obs["mask_tm"]
    sim = {key: np.zeros((2, 1)) for key in ("lr_te", "ph_te", "lr_tm", "ph_tm")}
    with pytest.raises(ValueError, match="mask_tm"):
        multimode_nrms(obs, sim)


@pytest.mark.parametrize("rho_error", [np.nan, np.inf, 0.0, -1.0])
def test_metric_rejects_invalid_rho_error(rho_error):
    with pytest.raises(ValueError):
        multimode_nrms(_empty_modes(), {}, rho_error=rho_error)


@pytest.mark.parametrize("phase_error", [np.nan, np.inf, 0.0, -1.0])
def test_metric_rejects_invalid_phase_error(phase_error):
    with pytest.raises(ValueError):
        multimode_nrms(_empty_modes(), {}, phase_error=phase_error)


def _valid_refinement_inputs():
    shape = (2, 2)
    return {
        "section0": np.full(shape, 2.0),
        "observations": {
            "lr_tm": np.full(shape, 2.0),
            "ph_tm": np.full(shape, 45.0),
            "mask_tm": np.ones(shape, dtype=bool),
        },
        "freqs": np.array([0.1, 1.0]),
        "station_x": np.array([-1.0, 1.0]),
        "x_grid": np.array([-1.0, 1.0]),
        "depth_grid": np.array([10.0, 100.0]),
        "mode": "tm",
    }


@pytest.mark.parametrize(
    ("updates", "error_type"),
    [
        ({"section0": np.full((1, 2), 2.0)}, ValueError),
        ({"section0": np.array([[2.0, np.nan], [2.0, 2.0]])}, ValueError),
        ({"freqs": np.array([1.0, 0.1])}, ValueError),
        ({"freqs": np.array([0.0, 1.0])}, ValueError),
        ({"station_x": np.array([1.0, 1.0])}, ValueError),
        ({"depth_grid": np.array([10.0, np.inf])}, ValueError),
        ({"mode": "xy"}, ValueError),
        ({"max_iter": 0}, ValueError),
        ({"max_iter_cg": 1.5}, TypeError),
        ({"alpha_ref": np.nan}, ValueError),
        ({"beta0_ratio": 0.0}, ValueError),
    ],
)
def test_refinement_rejects_invalid_contract_before_solver(updates, error_type):
    inputs = _valid_refinement_inputs()
    inputs.update(updates)
    with pytest.raises(error_type):
        refine_section_2d(**inputs)


def test_profiled_residual_is_invariant_to_stationwise_static_shift():
    station = np.array([0, 1, 0, 1, 0])
    observed = np.array([1.0, 2.0, 1.3, 2.2, 0.8])
    predicted = np.array([0.9, 2.2, 1.1, 2.0, 1.0])
    error = np.array([0.05, 0.08, 0.07, 0.06, 0.09])

    base, base_shift = _profile_log10_rho_residual(
        observed,
        predicted,
        station,
        standard_deviation=error,
    )
    added_shift = np.array([0.37, -0.21])
    shifted, fitted_shift = _profile_log10_rho_residual(
        observed + added_shift[station],
        predicted,
        station,
        standard_deviation=error,
    )

    assert np.allclose(shifted, base)
    assert np.allclose(fitted_shift, base_shift + added_shift)
    for station_index in (0, 1):
        selected = station == station_index
        weights = 1.0 / error[selected] ** 2
        assert np.sum(weights * shifted[selected]) == pytest.approx(0.0)


def test_profiled_observation_packing_is_independent_of_start_model():
    lr = np.array([[1.0, 2.0], [1.2, 2.2]])
    ph = np.array([[20.0, 30.0], [22.0, 32.0]])
    valid_by_frequency = [
        (0, np.array([True, True])),
        (1, np.array([True, True])),
    ]

    packed = _pack_profiled_mode_data(
        lr,
        ph,
        valid_by_frequency,
        phase_offset=-180.0,
    )

    # Regression for the former section0-dependent preprocessing: the packed
    # observations are the literal measured data. No starting-model response
    # has been subtracted or folded into the inversion data vector.
    assert np.allclose(packed.observed_log10_rho, [1.0, 2.0, 1.2, 2.2])
    assert np.allclose(
        packed.dobs_native[packed.rho_indices],
        10.0**packed.observed_log10_rho,
    )
    assert np.allclose(packed.observed_phase, [-160.0, -150.0, -158.0, -148.0])


def test_profiled_simpeg_objective_recomputes_nuisance_for_current_prediction():
    simpeg = pytest.importorskip("simpeg")
    del simpeg
    from simpeg import data
    from simpeg.simulation import BaseSimulation
    from simpeg.survey import BaseRx, BaseSrc, BaseSurvey

    valid_by_frequency = [
        (0, np.array([True, True])),
        (1, np.array([True, True])),
    ]
    observed = np.array([[1.0, 2.0], [1.2, 2.2]])
    phase = np.array([[20.0, 30.0], [22.0, 32.0]])
    station_shift = np.array([0.4, -0.3])
    layouts = [
        _pack_profiled_mode_data(
            observed + shift,
            phase,
            valid_by_frequency,
            phase_offset=0.0,
        )
        for shift in (0.0, station_shift[None, :])
    ]

    predicted_lr = np.array([[0.9, 2.1], [1.4, 2.0]])
    predicted_phase = np.array([[21.0, 29.0], [20.0, 34.0]])
    predicted = np.r_[
        10.0 ** predicted_lr[0],
        predicted_phase[0],
        10.0 ** predicted_lr[1],
        predicted_phase[1],
    ]
    jacobian = np.array(
        [
            [1.0, 0.0],
            [0.0, 2.0],
            [0.2, 0.1],
            [0.1, -0.2],
            [0.5, 0.1],
            [0.2, -0.7],
            [-0.1, 0.3],
            [0.4, 0.2],
        ]
    )

    class LinearSimulation(BaseSimulation):
        def __init__(self, survey):
            super().__init__(survey=survey)

        def fields(self, model=None):
            del model
            return {}

        def dpred(self, model=None, f=None):
            del f
            model = np.zeros(2) if model is None else np.asarray(model)
            return predicted + jacobian @ model

        def Jvec(self, model, vector, f=None):
            del model, f
            return jacobian @ vector

        def Jtvec(self, model, vector, f=None):
            del model, f
            return jacobian.T @ vector

    objectives = []
    for layout in layouts:
        receiver = BaseRx(np.zeros((len(layout.dobs_native), 1)))
        survey = BaseSurvey([BaseSrc([receiver])])
        simulation = LinearSimulation(survey)
        dat = data.Data(
            survey,
            dobs=layout.dobs_native,
            standard_deviation=layout.standard_deviation_native,
        )
        objectives.append(_make_profiled_static_shift_misfit(dat, simulation, layout))

    current_model = np.array([0.03, -0.02])
    assert objectives[1](current_model) == pytest.approx(objectives[0](current_model))

    direction = np.array([0.4, -0.7])
    epsilon = 1.0e-6
    finite_difference = (
        objectives[0](current_model + epsilon * direction)
        - objectives[0](current_model - epsilon * direction)
    ) / (2.0 * epsilon)
    analytic = np.dot(objectives[0].deriv(current_model), direction)
    assert analytic == pytest.approx(finite_difference, rel=2.0e-6, abs=2.0e-6)

    other_direction = np.array([-0.2, 0.6])
    h_direction = objectives[0].deriv2(current_model, direction)
    h_other = objectives[0].deriv2(current_model, other_direction)
    assert np.dot(other_direction, h_direction) == pytest.approx(
        np.dot(direction, h_other), rel=1.0e-12, abs=1.0e-12
    )
    assert np.dot(direction, h_direction) >= 0.0


def test_profile_interpolation_does_not_extrapolate_validity():
    values = np.array([[10.0, 20.0, 30.0]])
    station_mask = np.array([[False, True, True]])
    x_km = np.array([0.0, 10.0, 20.0])
    x_model = np.array([0.0, 5.0, 10.0, 15.0, 20.0])

    result, mask = _interpolate_profile_field(values, station_mask, x_km, x_model)
    assert np.array_equal(mask[0], [False, False, True, True, True])
    assert np.isnan(result[0, :2]).all()
    assert np.allclose(result[0, 2:], [20.0, 25.0, 30.0])


def test_profile_phase_interpolation_is_circular():
    result, mask = _interpolate_profile_field(
        np.array([[179.0, 1.0]]),
        np.array([[True, True]]),
        np.array([0.0, 2.0]),
        np.array([1.0]),
        circular_period_degrees=180.0,
    )
    assert mask[0, 0]
    assert min(abs(result[0, 0]), abs(result[0, 0] - 180.0)) < 1.0e-10


def test_geometry_metadata_discloses_nonphysical_normalization():
    metadata = profile_geometry_metadata(
        {
            "geometry_policy": "affine_profile_to_model_station_grid",
            "profile_azimuth_deg": np.float64(90.0),
            "profile_span_m": np.float64(290_000.0),
            "model_station_span_m": np.float64(16_000.0),
            "horizontal_compression_factor": np.float64(18.125),
            "publishable_physical_geometry": False,
            "station_ids": ("A", "B"),
        }
    )
    assert metadata["horizontal_compression_factor"] == pytest.approx(18.125)
    assert metadata["publishable_physical_geometry"] is False
    assert metadata["station_ids"] == ["A", "B"]


def test_section_metric_maps_literal_te_to_zyx(monkeypatch):
    from pimsr_forward import mt2d

    class FakeForward:
        def __init__(self, frequencies, station_x):
            self.shape = (len(frequencies), len(station_x))

        def response_te(self, section):
            del section
            return np.full(self.shape, 100.0), np.full(self.shape, 20.0)

        def response_tm(self, section):
            del section
            return np.full(self.shape, 1000.0), np.full(self.shape, 30.0)

    monkeypatch.setattr(mt2d, "MT2DForward", FakeForward)
    observations = {
        "lr_te": np.full((2, 1), 2.0),
        "ph_te": np.full((2, 1), 20.0),
        "mask_te": np.ones((2, 1), dtype=bool),
        "lr_tm": np.full((2, 1), 3.0),
        "ph_tm": np.full((2, 1), 30.0),
        "mask_tm": np.ones((2, 1), dtype=bool),
    }

    score = section_nrms_2d(
        np.full((2, 2), 2.0),
        observations,
        np.array([1.0, 2.0]),
        np.array([0.0]),
        np.array([-1.0, 1.0]),
        np.array([10.0, 100.0]),
    )
    assert score == pytest.approx(0.0)
