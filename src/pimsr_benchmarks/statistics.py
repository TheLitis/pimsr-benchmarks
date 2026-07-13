"""Reproducible uncertainty and bootstrap statistics for frozen benchmarks."""
from __future__ import annotations

import numpy as np

__all__ = ["bootstrap_ci", "calibration_summary"]


def bootstrap_ci(values, statistic=np.mean, confidence=0.95, n_resamples=10_000, seed=0):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("values must be a non-empty finite 1D array")
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
        "seed": int(seed),
    }


def calibration_summary(mean, sigma, truth, depth_axis=1, scenario=None):
    mean, sigma, truth = map(np.asarray, (mean, sigma, truth))
    if mean.shape != sigma.shape or mean.shape != truth.shape:
        raise ValueError("mean, sigma and truth shapes must match")
    if np.any(sigma <= 0) or not np.isfinite(sigma).all():
        raise ValueError("sigma must be finite and positive")
    levels = {"50": 0.67448975, "68": 0.99445788, "90": 1.64485363, "95": 1.95996398}
    error = np.abs(truth - mean)
    coverage = {key: float(np.mean(error <= k * sigma)) for key, k in levels.items()}
    expected = {"50": 0.50, "68": 0.68, "90": 0.90, "95": 0.95}
    result = {
        "coverage": coverage,
        "calibration_error_mean": float(np.mean([abs(coverage[k] - expected[k]) for k in levels])),
        "sharpness_mean_sigma": float(np.mean(sigma)),
        "coverage68_by_depth": np.mean(error <= levels["68"] * sigma, axis=tuple(i for i in range(mean.ndim) if i != depth_axis)).tolist(),
    }
    if scenario is not None:
        scenario = np.asarray(scenario)
        if len(scenario) != mean.shape[0]:
            raise ValueError("scenario length must match batch dimension")
        result["coverage68_by_scenario"] = {
            str(s): float(np.mean(error[scenario == s] <= levels["68"] * sigma[scenario == s]))
            for s in np.unique(scenario)
        }
    return result
