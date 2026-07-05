"""Hybrid inversion: neural prediction as the Occam starting model.

The amortised network gives a structured profile in milliseconds; Occam then
performs test-time refinement, restoring near-classical data misfit in a
fraction of the cold-start iterations. This is the MVP-2 "best of both"
strategy identified in the MVP-1 benchmark report.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .neural import NeuralInverter
from .occam1d import OccamResult, default_mesh, occam1d_invert

__all__ = ["HybridResult", "hybrid_invert"]


@dataclass
class HybridResult:
    occam: OccamResult
    neural_log10_rho: np.ndarray  # network prediction on its depth grid
    neural_time_s: float
    total_time_s: float


def _project_to_mesh(
    log10_rho: np.ndarray, depth_grid: np.ndarray, thicknesses: np.ndarray
) -> np.ndarray:
    """Sample the network's depth-grid profile at the Occam cell centres."""
    edges = np.concatenate([[0.0], np.cumsum(thicknesses)])
    centres = 0.5 * (edges[:-1] + edges[1:])
    # Cell centres plus one terminating half-space sampled at the last edge.
    query = np.concatenate([centres, [edges[-1]]])
    return np.interp(query, depth_grid, log10_rho)


def hybrid_invert(
    inverter: NeuralInverter,
    log_rho_a: np.ndarray,
    phase: np.ndarray,
    periods: np.ndarray,
    gravity: np.ndarray | None = None,
    thicknesses: np.ndarray | None = None,
    max_iterations: int = 12,
    neural_log_rho_a: np.ndarray | None = None,
    neural_phase: np.ndarray | None = None,
    **occam_kwargs,
) -> HybridResult:
    """Neural warm start followed by Occam Gauss-Newton refinement.

    ``log_rho_a``/``phase``/``periods`` feed the Occam refinement and may lie
    on any period grid. For real stations whose grid differs from the training
    grid, pass the response resampled onto ``inverter.periods`` via
    ``neural_log_rho_a``/``neural_phase``; synthetic data needs neither.
    """
    t0 = perf_counter()
    thick = default_mesh() if thicknesses is None else np.asarray(thicknesses)

    n_lr = log_rho_a if neural_log_rho_a is None else neural_log_rho_a
    n_ph = phase if neural_phase is None else neural_phase
    pred = inverter.invert(n_lr, n_ph, gravity)
    m0 = _project_to_mesh(pred.log10_rho, inverter.depth_grid, thick)

    occam = occam1d_invert(
        log_rho_a,
        phase,
        periods,
        thicknesses=thick,
        max_iterations=max_iterations,
        initial_model=m0,
        **occam_kwargs,
    )
    return HybridResult(
        occam=occam,
        neural_log10_rho=pred.log10_rho,
        neural_time_s=pred.wall_time_s,
        total_time_s=perf_counter() - t0,
    )
