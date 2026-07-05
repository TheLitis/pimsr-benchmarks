"""Occam-style regularised 1D MT inversion (classical baseline).

A faithful, dependency-light implementation of the smooth ("Occam") 1D MT
inversion of Constable, Parker & Constable (1987): Gauss-Newton iterations on
log10-resistivity of a fixed layered mesh, first-difference (smoothness)
Tikhonov regularisation, and a cooling schedule on the trade-off parameter
targeting nRMS ~= 1.

The forward kernel is the exact Wait recursion from ``pimsr-forward`` so the
comparison against the neural inversion is apples-to-apples: same physics,
same data vector [log10 rho_a; phase/45].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from pimsr_forward.mt1d import mt1d_response

__all__ = ["OccamResult", "occam1d_invert", "default_mesh"]


def default_mesh(n_cells: int = 48, z_min: float = 10.0, z_max: float = 6.0e4) -> np.ndarray:
    """Log-spaced layer thicknesses (m) spanning the MT sensitivity range."""
    edges = np.logspace(np.log10(z_min), np.log10(z_max), n_cells + 1)
    return np.diff(edges)


@dataclass
class OccamResult:
    log10_rho: np.ndarray  # (n_cells + 1,) incl. terminating half-space
    thicknesses: np.ndarray  # (n_cells,)
    nrms: float
    n_iterations: int
    wall_time_s: float
    converged: bool
    nrms_history: list[float] = field(default_factory=list)

    def profile_on_grid(self, depth_grid: np.ndarray) -> np.ndarray:
        """Piecewise-constant log10-resistivity sampled on ``depth_grid``."""
        interfaces = np.cumsum(self.thicknesses)
        idx = np.searchsorted(interfaces, depth_grid, side="right")
        return self.log10_rho[idx]


def _forward(m: np.ndarray, thick: np.ndarray, periods: np.ndarray) -> np.ndarray:
    rho_a, phase = mt1d_response(10.0**m, thick, periods)
    return np.concatenate([np.log10(rho_a), phase / 45.0])


def _jacobian(
    m: np.ndarray, thick: np.ndarray, periods: np.ndarray, d0: np.ndarray
) -> np.ndarray:
    """Forward-difference Jacobian d(data)/d(log10 rho)."""
    eps = 1e-4
    J = np.zeros((d0.size, m.size))
    for j in range(m.size):
        mp = m.copy()
        mp[j] += eps
        J[:, j] = (_forward(mp, thick, periods) - d0) / eps
    return J


def occam1d_invert(
    obs_log10_rho_a: np.ndarray,
    obs_phase: np.ndarray,
    periods: np.ndarray,
    err_log10_rho_a: float = 0.03,
    err_phase_deg: float = 2.0,
    thicknesses: np.ndarray | None = None,
    max_iterations: int = 30,
    target_nrms: float = 1.0,
    mu0: float = 1.0e3,
    mu_cool: float = 0.65,
) -> OccamResult:
    """Invert one MT sounding for a smooth 1D resistivity profile.

    Parameters mirror standard Occam practice: data are weighted by their
    errors, the regulariser is the first-difference roughness of the
    log-resistivity profile, and mu is cooled geometrically until the
    chi-squared target is met (then held).
    """
    t0 = perf_counter()
    thick = default_mesh() if thicknesses is None else np.asarray(thicknesses)
    n_cells = thick.size + 1

    d_obs = np.concatenate([obs_log10_rho_a, obs_phase / 45.0])
    w = np.concatenate(
        [
            np.full(obs_log10_rho_a.size, 1.0 / err_log10_rho_a),
            np.full(obs_phase.size, 1.0 / (err_phase_deg / 45.0)),
        ]
    )

    # Roughness operator (first differences).
    R = np.zeros((n_cells - 1, n_cells))
    for i in range(n_cells - 1):
        R[i, i], R[i, i + 1] = -1.0, 1.0

    # Half-space starting model from the mean apparent resistivity.
    m = np.full(n_cells, float(np.mean(obs_log10_rho_a)))
    mu = mu0
    history: list[float] = []
    converged = False

    for it in range(max_iterations):
        d_pred = _forward(m, thick, periods)
        r = (d_obs - d_pred) * w
        nrms = float(np.sqrt(np.mean(r**2)))
        history.append(nrms)
        if nrms <= target_nrms:
            converged = True
            break

        J = _jacobian(m, thick, periods, d_pred) * w[:, None]
        # Gauss-Newton step on the regularised normal equations:
        # (J^T J + mu R^T R) dm = J^T r - mu R^T R m
        A = J.T @ J + mu * (R.T @ R)
        b = J.T @ r - mu * (R.T @ (R @ m))
        try:
            dm = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            dm = np.linalg.lstsq(A, b, rcond=None)[0]

        # Damped line search: accept the largest step that reduces nRMS.
        best_m, best_nrms = m, nrms
        for step in (1.0, 0.5, 0.25, 0.1):
            m_try = np.clip(m + step * dm, -1.0, 5.0)
            r_try = (d_obs - _forward(m_try, thick, periods)) * w
            nrms_try = float(np.sqrt(np.mean(r_try**2)))
            if nrms_try < best_nrms:
                best_m, best_nrms = m_try, nrms_try
                break
        m = best_m
        mu = max(mu * mu_cool, 1.0e-2)

    return OccamResult(
        log10_rho=m,
        thicknesses=thick,
        nrms=history[-1] if history else float("nan"),
        n_iterations=len(history),
        wall_time_s=perf_counter() - t0,
        converged=converged,
        nrms_history=history,
    )
