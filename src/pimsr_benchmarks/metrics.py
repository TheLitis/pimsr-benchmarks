"""Benchmark metrics shared by the synthetic and real-data studies."""

from __future__ import annotations

import numpy as np

__all__ = ["coverage", "data_nrms", "profile_rmse", "summarize"]


def _phase_residual_deg(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Shortest signed ``left - right`` residual for 180-degree MT phase."""
    return (np.asarray(left) - np.asarray(right) + 90.0) % 180.0 - 90.0


def _positive_finite_scalar(value: float, name: str) -> float:
    """Validate one scalar error floor without accepting implicit arrays."""
    array = np.asarray(value)
    if array.ndim != 0 or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite positive scalar")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive scalar") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return result


def profile_rmse(pred_log10_res: np.ndarray, true_log10_res: np.ndarray) -> float:
    """RMSE in log10-resistivity over the depth grid (lower is better)."""
    predicted = np.asarray(pred_log10_res, dtype=np.float64)
    truth = np.asarray(true_log10_res, dtype=np.float64)
    if predicted.shape != truth.shape or predicted.size == 0:
        raise ValueError("predicted and true profiles must have the same non-empty shape")
    if not np.isfinite(predicted).all() or not np.isfinite(truth).all():
        raise ValueError("predicted and true profiles must be finite")
    return float(np.sqrt(np.mean((predicted - truth) ** 2)))


def data_nrms(
    pred_log_rho_a: np.ndarray,
    pred_phase: np.ndarray,
    obs_log_rho_a: np.ndarray,
    obs_phase: np.ndarray,
    err_log_rho: float = 0.03,
    err_phase: float = 2.0,
    mask: np.ndarray | None = None,
) -> float:
    """Error-normalised RMS data misfit; ~1 means fitting to the noise level.

    MT impedance phase is 180-degree periodic, so phase differences use the
    shortest signed branch.  All four data arrays must be non-empty and have
    exactly the same shape.  A mask, when present, must be boolean, must match
    that shape and must select at least one datum.  Non-finite values are
    permitted only outside the explicit mask.
    """
    arrays = {
        "pred_log_rho_a": np.asarray(pred_log_rho_a, dtype=float),
        "pred_phase": np.asarray(pred_phase, dtype=float),
        "obs_log_rho_a": np.asarray(obs_log_rho_a, dtype=float),
        "obs_phase": np.asarray(obs_phase, dtype=float),
    }
    shape = arrays["pred_log_rho_a"].shape
    if not shape or arrays["pred_log_rho_a"].size == 0:
        raise ValueError("data arrays must be non-empty and at least one-dimensional")
    if any(array.shape != shape for array in arrays.values()):
        raise ValueError("all predicted and observed data arrays must have matching shapes")

    rho_error = _positive_finite_scalar(err_log_rho, "err_log_rho")
    phase_error = _positive_finite_scalar(err_phase, "err_phase")
    if mask is None:
        valid = np.ones(shape, dtype=bool)
    else:
        valid = np.asarray(mask)
        if valid.shape != shape:
            raise ValueError("mask must match the data shape")
        if valid.dtype.kind != "b":
            raise ValueError("mask must be boolean")
        if not np.any(valid):
            raise ValueError("mask must select at least one datum")

    for name, array in arrays.items():
        if np.any(~np.isfinite(array[valid])):
            raise ValueError(f"{name} contains non-finite selected data")

    rho_residual = (
        arrays["pred_log_rho_a"][valid] - arrays["obs_log_rho_a"][valid]
    ) / rho_error
    phase_residual = _phase_residual_deg(
        arrays["pred_phase"][valid], arrays["obs_phase"][valid]
    ) / phase_error
    residual = np.concatenate([rho_residual.ravel(), phase_residual.ravel()])
    return float(np.sqrt(np.mean(residual**2)))


def coverage(
    pred_mean: np.ndarray, pred_sigma: np.ndarray, truth: np.ndarray, k: float = 1.0
) -> float:
    """Fraction of truth inside +-k sigma. Calibrated Gaussian: 0.683 at k=1."""
    mean = np.asarray(pred_mean, dtype=np.float64)
    sigma = np.asarray(pred_sigma, dtype=np.float64)
    truth_array = np.asarray(truth, dtype=np.float64)
    if mean.shape != sigma.shape or mean.shape != truth_array.shape or mean.size == 0:
        raise ValueError("mean, sigma and truth must have the same non-empty shape")
    if (
        not np.isfinite(mean).all()
        or not np.isfinite(sigma).all()
        or not np.isfinite(truth_array).all()
        or np.any(sigma <= 0.0)
    ):
        raise ValueError("mean/truth must be finite and sigma must be finite and positive")
    scale = _positive_finite_scalar(k, "k")
    inside = np.abs(truth_array - mean) <= scale * sigma
    return float(np.mean(inside))


def summarize(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 1 or a.size == 0 or not np.isfinite(a).all():
        raise ValueError("summary values must be a non-empty finite 1D array")
    return {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p90": float(np.percentile(a, 90)),
        "std": float(a.std()),
        "n": int(a.size),
    }
