"""Strict scalar metric contracts and MT phase periodicity."""

import numpy as np
import pytest

from pimsr_benchmarks.metrics import coverage, data_nrms, profile_rmse, summarize


def test_data_nrms_uses_shortest_180_degree_phase_residual():
    score = data_nrms(
        pred_log_rho_a=np.array([2.0]),
        pred_phase=np.array([1.0]),
        obs_log_rho_a=np.array([2.0]),
        obs_phase=np.array([179.0]),
        err_log_rho=1.0,
        err_phase=1.0,
    )
    assert score == pytest.approx(np.sqrt(2.0))


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("pred_phase", np.ones(2)),
        ("obs_log_rho_a", np.ones((1, 1))),
        ("obs_phase", np.ones(2)),
    ],
)
def test_data_nrms_rejects_mismatched_data_shapes(argument, value):
    inputs = {
        "pred_log_rho_a": np.ones(1),
        "pred_phase": np.ones(1),
        "obs_log_rho_a": np.ones(1),
        "obs_phase": np.ones(1),
    }
    inputs[argument] = value
    with pytest.raises(ValueError, match="matching shapes"):
        data_nrms(**inputs)


@pytest.mark.parametrize("error_name", ["err_log_rho", "err_phase"])
@pytest.mark.parametrize("invalid", [0.0, -1.0, np.nan, np.inf, [1.0]])
def test_data_nrms_rejects_invalid_errors(error_name, invalid):
    kwargs = {error_name: invalid}
    with pytest.raises(ValueError, match="finite positive scalar"):
        data_nrms(np.ones(1), np.ones(1), np.ones(1), np.ones(1), **kwargs)


def test_data_nrms_requires_nonempty_data_and_mask_selection():
    empty = np.array([], dtype=float)
    with pytest.raises(ValueError, match="non-empty"):
        data_nrms(empty, empty, empty, empty)

    values = np.ones(2)
    with pytest.raises(ValueError, match="at least one"):
        data_nrms(values, values, values, values, mask=np.zeros(2, dtype=bool))


def test_data_nrms_requires_boolean_shape_matched_mask():
    values = np.ones(2)
    with pytest.raises(ValueError, match="data shape"):
        data_nrms(values, values, values, values, mask=np.ones(1, dtype=bool))
    with pytest.raises(ValueError, match="boolean"):
        data_nrms(values, values, values, values, mask=np.ones(2, dtype=np.int8))


def test_data_nrms_rejects_nonfinite_selected_values_but_allows_masked_missing():
    missing = np.array([1.0, np.nan])
    finite = np.ones(2)
    assert np.isfinite(
        data_nrms(missing, finite, finite, finite, mask=np.array([True, False]))
    )
    with pytest.raises(ValueError, match="non-finite selected"):
        data_nrms(missing, finite, finite, finite, mask=np.ones(2, dtype=bool))


def test_profile_rmse_and_coverage_validate_shapes_and_finiteness():
    assert profile_rmse(np.array([1.0, 3.0]), np.array([1.0, 1.0])) == pytest.approx(
        np.sqrt(2.0)
    )
    assert coverage(np.zeros(2), np.ones(2), np.array([0.5, 2.0])) == 0.5

    with pytest.raises(ValueError, match="same non-empty shape"):
        profile_rmse(np.ones(2), np.ones(1))
    with pytest.raises(ValueError, match="finite"):
        profile_rmse(np.array([np.nan]), np.zeros(1))
    with pytest.raises(ValueError, match="same non-empty shape"):
        coverage(np.ones(2), np.ones(1), np.ones(2))
    with pytest.raises(ValueError, match="sigma"):
        coverage(np.zeros(1), np.zeros(1), np.zeros(1))
    with pytest.raises(ValueError, match="finite positive scalar"):
        coverage(np.zeros(1), np.ones(1), np.zeros(1), k=0.0)


def test_summarize_rejects_empty_multidimensional_and_nonfinite_values():
    summary = summarize([1.0, 2.0, 3.0])
    assert summary["median"] == 2.0
    assert summary["n"] == 3
    for invalid in ([], [[1.0]], [np.inf]):
        with pytest.raises(ValueError, match="non-empty finite 1D"):
            summarize(invalid)
