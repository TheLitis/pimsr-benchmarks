"""Warm-start plumbing tests (no trained checkpoint required)."""

import numpy as np
import pytest

from pimsr_benchmarks.hybrid import _project_to_mesh
from pimsr_benchmarks.occam1d import default_mesh, occam1d_invert
from pimsr_forward.mt1d import mt1d_response


def _two_layer_data():
    periods = np.logspace(-2, 3, 24)
    rho = np.array([30.0, 500.0])
    thick = np.array([800.0])
    rho_a, phase = mt1d_response(rho, thick, periods)
    return periods, np.log10(rho_a), phase


class TestProjectToMesh:
    def test_shape_and_monotone_sampling(self) -> None:
        thick = default_mesh()
        depth = np.logspace(1, 4.8, 64)
        prof = np.linspace(1.0, 3.0, 64)
        m = _project_to_mesh(prof, depth, thick)
        assert m.shape == (thick.size + 1,)
        assert m[0] == pytest.approx(prof[0], abs=0.2)
        assert np.all(np.diff(m) >= -1e-12)  # monotone input stays monotone


class TestWarmStart:
    def test_good_init_converges_faster(self) -> None:
        periods, lr, ph = _two_layer_data()
        cold = occam1d_invert(lr, ph, periods)

        thick = default_mesh()
        edges = np.concatenate([[0.0], np.cumsum(thick)])
        centres = np.concatenate([0.5 * (edges[:-1] + edges[1:]), [edges[-1]]])
        # Truth-like starting model.
        init = np.where(centres < 800.0, np.log10(30.0), np.log10(500.0))
        warm = occam1d_invert(lr, ph, periods, initial_model=init)

        assert warm.converged
        assert warm.n_iterations <= cold.n_iterations
        assert warm.nrms <= max(cold.nrms, 1.0) + 1e-9

    def test_bad_init_size_raises(self) -> None:
        periods, lr, ph = _two_layer_data()
        with pytest.raises(ValueError, match="initial_model"):
            occam1d_invert(lr, ph, periods, initial_model=np.zeros(3))
