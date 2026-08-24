import numpy as np
import pytest

from pimsr_benchmarks.statistics import (
    bootstrap_ci,
    calibration_summary,
    hierarchical_paired_bootstrap,
)


def test_bootstrap_is_reproducible():
    values = np.arange(10.0)
    assert bootstrap_ci(values, n_resamples=500, seed=7) == bootstrap_ci(
        values, n_resamples=500, seed=7
    )


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


def test_hierarchical_paired_bootstrap_is_reproducible_and_paired():
    left = np.array([3.0, 5.0, 8.0, 4.0, 7.0])
    right = np.array([2.0, 3.0, 5.0, 4.0, 6.0])
    families = np.array(["fold", "fold", "rift", "rift", "rift"])
    base_models = np.array(["a", "a", "b", "c", "c"])
    sample_ids = np.array(
        ["a-noise-0", "a-noise-1", "b-noise-0", "c-noise-0", "c-noise-1"]
    )

    first = hierarchical_paired_bootstrap(
        left,
        right,
        families,
        base_models,
        n_resamples=500,
        seed=17,
        left_sample_ids=sample_ids,
        right_sample_ids=sample_ids.copy(),
    )
    second = hierarchical_paired_bootstrap(
        left,
        right,
        families,
        base_models,
        n_resamples=500,
        seed=17,
        left_sample_ids=sample_ids,
        right_sample_ids=sample_ids.copy(),
    )

    # Base-model paired effects are a=1.5, b=3.0 and c=0.5.
    assert first == second
    assert first["estimate"] == pytest.approx((1.5 + 3.0 + 0.5) / 3.0)
    assert first["n_pairs"] == 5
    assert first["n_families"] == 2
    assert first["n_base_models"] == 3


def test_hierarchical_paired_bootstrap_rejects_cross_family_base_id():
    with pytest.raises(ValueError, match="exactly one family"):
        hierarchical_paired_bootstrap(
            [1.0, 2.0],
            [0.0, 0.0],
            ["family-a", "family-b"],
            ["same-base", "same-base"],
            n_resamples=10,
            left_sample_ids=["sample-0", "sample-1"],
            right_sample_ids=["sample-0", "sample-1"],
        )


def test_hierarchical_paired_bootstrap_requires_verified_ordered_identities():
    common = ([1.0, 2.0], [0.0, 0.0], ["family", "family"], ["a", "b"])
    with pytest.raises(ValueError, match="required for verified pairing"):
        hierarchical_paired_bootstrap(*common, n_resamples=10)
    with pytest.raises(ValueError, match="not identically ordered"):
        hierarchical_paired_bootstrap(
            *common,
            n_resamples=10,
            left_sample_ids=["sample-0", "sample-1"],
            right_sample_ids=["sample-1", "sample-0"],
        )
    with pytest.raises(ValueError, match="must be unique"):
        hierarchical_paired_bootstrap(
            *common,
            n_resamples=10,
            left_sample_ids=["same", "same"],
            right_sample_ids=["same", "same"],
        )


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"confidence": 1.0}, ValueError),
        ({"n_resamples": 0}, ValueError),
        ({"n_resamples": True}, TypeError),
        ({"seed": -1}, ValueError),
    ],
)
def test_bootstrap_rejects_invalid_options(kwargs, error):
    with pytest.raises(error):
        bootstrap_ci([1.0, 2.0], **kwargs)
