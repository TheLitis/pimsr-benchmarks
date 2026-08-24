"""Reproducible uncertainty and bootstrap statistics for frozen benchmarks."""

from __future__ import annotations

import numpy as np

__all__ = ["bootstrap_ci", "calibration_summary", "hierarchical_paired_bootstrap"]


def _validate_bootstrap_options(confidence, n_resamples, seed) -> tuple[float, int, int]:
    if (
        not np.isscalar(confidence)
        or not np.isfinite(confidence)
        or not 0 < confidence < 1
    ):
        raise ValueError("confidence must be a finite scalar in (0, 1)")
    if isinstance(n_resamples, (bool, np.bool_)) or not isinstance(
        n_resamples, (int, np.integer)
    ):
        raise TypeError("n_resamples must be an integer")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    if not 0 <= seed <= np.iinfo(np.uint64).max:
        raise ValueError("seed must be in the uint64 range")
    return float(confidence), int(n_resamples), int(seed)


def bootstrap_ci(values, statistic=np.mean, confidence=0.95, n_resamples=10_000, seed=0):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("values must be a non-empty finite 1D array")
    confidence, n_resamples, seed = _validate_bootstrap_options(
        confidence,
        n_resamples,
        seed,
    )
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_resamples)
    for start in range(0, n_resamples, 512):
        size = min(512, n_resamples - start)
        samples = values[rng.integers(0, values.size, (size, values.size))]
        estimates[start : start + size] = np.apply_along_axis(statistic, 1, samples)
    alpha = (1.0 - confidence) / 2.0
    return {
        "estimate": float(statistic(values)),
        "low": float(np.quantile(estimates, alpha)),
        "high": float(np.quantile(estimates, 1.0 - alpha)),
        "confidence": confidence,
        "n": int(values.size),
        "n_resamples": n_resamples,
        "seed": int(seed),
    }


def hierarchical_paired_bootstrap(
    left,
    right,
    family_ids,
    base_model_ids,
    statistic=np.mean,
    confidence=0.95,
    n_resamples=10_000,
    seed=0,
    *,
    left_sample_ids=None,
    right_sample_ids=None,
):
    """Paired hierarchical CI over geological families and base models.

    Repeated noise or survey realizations for one base model are first averaged
    as paired differences. Each bootstrap replicate samples families with
    replacement and then samples that family's base models with replacement,
    preserving the number of independent base models in every selected family.

    Both metric arrays require their own ordered sample identities.  This is
    intentionally fail-closed: positional pairing without identities can
    silently compare different realizations after filtering or sorting.
    """
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    families = np.asarray(family_ids, dtype=object)
    base_models = np.asarray(base_model_ids, dtype=object)
    if not (
        left_values.ndim == right_values.ndim == families.ndim == base_models.ndim == 1
    ):
        raise ValueError("paired metrics and hierarchy ids must be one-dimensional")
    if not (
        left_values.size == right_values.size == families.size == base_models.size
        and left_values.size > 0
    ):
        raise ValueError(
            "paired metrics and hierarchy ids must have equal non-zero length"
        )
    if not np.isfinite(left_values).all() or not np.isfinite(right_values).all():
        raise ValueError("paired metrics must be finite")

    if left_sample_ids is None or right_sample_ids is None:
        raise ValueError(
            "left_sample_ids and right_sample_ids are required for verified pairing"
        )
    left_ids = np.asarray(left_sample_ids, dtype=object)
    right_ids = np.asarray(right_sample_ids, dtype=object)
    if left_ids.ndim != 1 or right_ids.ndim != 1:
        raise ValueError("ordered sample ids must be one-dimensional")
    if left_ids.size != left_values.size or right_ids.size != right_values.size:
        raise ValueError("ordered sample ids must match their metric array lengths")

    def canonical_id(value: object) -> tuple[str, str]:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("sample ids must be strings or integers, not booleans")
        if isinstance(value, (int, np.integer)):
            return ("integer", str(int(value)))
        if isinstance(value, str):
            if not value:
                raise ValueError("sample ids must be non-empty")
            return ("string", value)
        raise TypeError("sample ids must be strings or integers")

    canonical_left = [canonical_id(value) for value in left_ids]
    canonical_right = [canonical_id(value) for value in right_ids]
    if len(set(canonical_left)) != len(canonical_left):
        raise ValueError("left sample ids must be unique")
    if len(set(canonical_right)) != len(canonical_right):
        raise ValueError("right sample ids must be unique")
    if canonical_left != canonical_right:
        raise ValueError("left and right sample ids are not identically ordered")
    if any(not isinstance(value, str) or not value for value in families):
        raise ValueError("family ids must be non-empty strings")
    if any(not isinstance(value, str) or not value for value in base_models):
        raise ValueError("base-model ids must be non-empty strings")
    confidence, n_resamples, seed = _validate_bootstrap_options(
        confidence,
        n_resamples,
        seed,
    )

    grouped_rows: dict[tuple[str, str], list[float]] = {}
    base_family: dict[str, str] = {}
    for left_value, right_value, family, base_model in zip(
        left_values,
        right_values,
        families,
        base_models,
        strict=True,
    ):
        previous_family = base_family.setdefault(base_model, family)
        if previous_family != family:
            raise ValueError("each base-model id must belong to exactly one family")
        grouped_rows.setdefault((family, base_model), []).append(left_value - right_value)

    family_values: dict[str, np.ndarray] = {}
    for (family, _base_model), differences in grouped_rows.items():
        family_values.setdefault(family, []).append(float(np.mean(differences)))
    family_values = {
        family: np.asarray(values, dtype=np.float64)
        for family, values in family_values.items()
    }
    ordered_families = tuple(sorted(family_values))
    base_differences = np.concatenate(
        [family_values[family] for family in ordered_families]
    )
    estimate = float(statistic(base_differences))
    if not np.isfinite(estimate):
        raise ValueError("statistic must return a finite scalar")

    rng = np.random.default_rng(seed)
    estimates = np.empty(n_resamples, dtype=np.float64)
    for replicate in range(n_resamples):
        selected_families = rng.integers(0, len(ordered_families), len(ordered_families))
        selected_values: list[np.ndarray] = []
        for family_index in selected_families:
            values = family_values[ordered_families[int(family_index)]]
            selected_values.append(values[rng.integers(0, values.size, values.size)])
        estimate_value = statistic(np.concatenate(selected_values))
        if not np.isscalar(estimate_value) or not np.isfinite(estimate_value):
            raise ValueError("statistic must return a finite scalar")
        estimates[replicate] = float(estimate_value)

    alpha = (1.0 - confidence) / 2.0
    return {
        "estimate": estimate,
        "low": float(np.quantile(estimates, alpha)),
        "high": float(np.quantile(estimates, 1.0 - alpha)),
        "confidence": confidence,
        "n_pairs": int(left_values.size),
        "n_families": len(ordered_families),
        "n_base_models": len(grouped_rows),
        "n_resamples": n_resamples,
        "seed": seed,
        "difference": "left_minus_right",
        "sample_pairing": "verified_equal_ordered_identities",
        "n_unique_sample_ids": len(canonical_left),
        "repeat_aggregation": "mean_within_base_model",
    }


def calibration_summary(mean, sigma, truth, depth_axis=1, scenario=None):
    mean, sigma, truth = map(np.asarray, (mean, sigma, truth))
    if mean.shape != sigma.shape or mean.shape != truth.shape:
        raise ValueError("mean, sigma and truth shapes must match")
    if mean.ndim == 0 or not -mean.ndim <= depth_axis < mean.ndim:
        raise ValueError("depth_axis must select an existing array dimension")
    if not np.isfinite(mean).all() or not np.isfinite(truth).all():
        raise ValueError("mean and truth must be finite")
    if np.any(sigma <= 0) or not np.isfinite(sigma).all():
        raise ValueError("sigma must be finite and positive")
    levels = {"50": 0.67448975, "68": 0.99445788, "90": 1.64485363, "95": 1.95996398}
    error = np.abs(truth - mean)
    coverage = {key: float(np.mean(error <= k * sigma)) for key, k in levels.items()}
    expected = {"50": 0.50, "68": 0.68, "90": 0.90, "95": 0.95}
    result = {
        "coverage": coverage,
        "calibration_error_mean": float(
            np.mean([abs(coverage[k] - expected[k]) for k in levels])
        ),
        "sharpness_mean_sigma": float(np.mean(sigma)),
        "coverage68_by_depth": np.mean(
            error <= levels["68"] * sigma,
            axis=tuple(i for i in range(mean.ndim) if i != depth_axis),
        ).tolist(),
    }
    if scenario is not None:
        scenario = np.asarray(scenario)
        if len(scenario) != mean.shape[0]:
            raise ValueError("scenario length must match batch dimension")
        result["coverage68_by_scenario"] = {
            str(s): float(
                np.mean(error[scenario == s] <= levels["68"] * sigma[scenario == s])
            )
            for s in np.unique(scenario)
        }
    return result
