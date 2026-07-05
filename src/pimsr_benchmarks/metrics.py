"""Benchmark metrics shared by the synthetic and real-data studies."""

from __future__ import annotations

import numpy as np

__all__ = ["profile_rmse", "data_nrms", "coverage", "summarize"]


def profile_rmse(pred_log10_res: np.ndarray, true_log10_res: np.ndarray) -> float:
    """RMSE in log10-resistivity over the depth grid (lower is better)."""
    return float(np.sqrt(np.mean((pred_log10_res - true_log10_res) ** 2)))


def data_nrms(
    pred_log_rho_a: np.ndarray,
    pred_phase: np.ndarray,
    obs_log_rho_a: np.ndarray,
    obs_phase: np.ndarray,
    err_log_rho: float = 0.03,
    err_phase: float = 2.0,
    mask: np.ndarray | None = None,
) -> float:
    """Error-normalised RMS data misfit; ~1 means fitting to the noise level."""
    r1 = (pred_log_rho_a - obs_log_rho_a) / err_log_rho
    r2 = (pred_phase - obs_phase) / err_phase
    r = np.concatenate([np.atleast_1d(r1), np.atleast_1d(r2)])
    if mask is not None:
        m2 = np.concatenate([np.atleast_1d(mask), np.atleast_1d(mask)])
        r = r[m2]
    return float(np.sqrt(np.mean(r**2)))


def coverage(
    pred_mean: np.ndarray, pred_sigma: np.ndarray, truth: np.ndarray, k: float = 1.0
) -> float:
    """Fraction of truth inside +-k sigma. Calibrated Gaussian: 0.683 at k=1."""
    inside = np.abs(truth - pred_mean) <= k * pred_sigma
    return float(np.mean(inside))


def summarize(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p90": float(np.percentile(a, 90)),
        "std": float(a.std()),
        "n": int(a.size),
    }
