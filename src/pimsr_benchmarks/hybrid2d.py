"""2D hybrid inversion: U-Net warm start + SimPEG Gauss-Newton refinement.

The 1D hybrid (neural start -> Occam) closed the real-data gap entirely at
2.4x classical speed. This is the 2D analogue: the U-Net's laterally
coherent section becomes the starting and reference model for a few
inexact-Gauss-Newton iterations of an explicit literal SimPEG mode.

Legacy scripts normalise each real profile onto the training section's 24 km
frame.  Because that changes the 2D forward physics, scores produced on this
normalised geometry describe only internal forward consistency and are not
scale-invariant or metrically crustal.  A publishable field comparison must
preserve and report physical geometry.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from time import perf_counter

import numpy as np
from pimsr_forward.emtf import (
    YELLOWSTONE_PROFILE_IDS,
    YELLOWSTONE_PROFILES,
)
from pimsr_forward.emtf import (
    assemble_profile_modes as _assemble_profile_modes,
)
from pimsr_forward.emtf import (
    interpolate_profile_field as _shared_interpolate_profile_field,
)

from .emtf import MTMode

__all__ = [
    "LOG10_RHO_ERROR",
    "PHASE_ERROR_DEG",
    "SECTION_NRMS_METRIC_ID",
    "Hybrid2DResult",
    "assemble_profile",
    "assemble_profile_modes",
    "multimode_nrms",
    "profile_geometry_metadata",
    "refine_section_2d",
    "scoring_observation_error_contract",
    "section_nrms",
    "section_nrms_2d",
]

SECTION_NRMS_METRIC_ID = (
    "section_nrms_2d_profile_rotated_tetm_masked_normalized_geometry_v3"
)
LOG10_RHO_ERROR = 0.05
PHASE_ERROR_DEG = 2.9


def _finite_scalar(
    value: object,
    *,
    name: str,
    strictly_positive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    invalid_sign = result <= 0.0 if strictly_positive else result < 0.0
    if not np.isfinite(result) or invalid_sign:
        qualifier = "positive" if strictly_positive else "non-negative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


def _strict_axis(
    values: object,
    *,
    name: str,
    positive: bool = False,
) -> np.ndarray:
    axis = np.asarray(values, dtype=float)
    if axis.ndim != 1 or axis.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.isfinite(axis).all() or (positive and np.any(axis <= 0.0)):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    if axis.size > 1 and np.any(np.diff(axis) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    return axis


def scoring_observation_error_contract() -> dict[str, object]:
    """Return the shared uncertainty policy used by every 2D score.

    Classical solvers may require a native representation of these two
    uncertainties, but benchmark scoring always returns to these declared
    apparent-resistivity and phase standard deviations.
    """
    return {
        "log10_apparent_resistivity_std": LOG10_RHO_ERROR,
        "phase_std_degrees": PHASE_ERROR_DEG,
        "phase_residual_period_degrees": 180.0,
        "static_shift_policy": "profile_per_mode_per_station_log10_offset",
        "mode_weighting": "equal_te_tm",
        "component_weighting": "equal_log10_rho_phase",
    }


#: E-W profile at ~44.6N, west to east (same as run_2d_bench).
PROFILE_IDS = YELLOWSTONE_PROFILE_IDS

#: All five E-W USArray rows in the region, west to east. "H-YS" is the
#: original Yellowstone profile; the others are independent test lines.
PROFILES = YELLOWSTONE_PROFILES


def assemble_profile(
    emtf_dir: str,
    freqs: np.ndarray,
    station_x: np.ndarray,
    *,
    mode: MTMode,
    profile_ids: list[str] | None = None,
):
    """Interpolate one profile-coordinate mode onto the model station grid.

    ``mode="te"`` is local ``Zyx`` and ``mode="tm"`` is local ``Zxy`` after
    rotating the full geographic EMTF tensor into the fitted profile frame.
    Returns ``(lr, ph, period_mask, x_model, x_km)``.  Out-of-band values are
    ``NaN`` and never extrapolated into the returned valid-data mask.
    """
    if mode not in ("te", "tm"):
        raise ValueError(f"mode must be 'te' or 'tm', got {mode!r}")
    modes = assemble_profile_modes(emtf_dir, freqs, station_x, profile_ids=profile_ids)
    return (
        modes[f"lr_{mode}"],
        modes[f"ph_{mode}"],
        modes[f"mask_{mode}"],
        modes["x_model"],
        modes["x_km"],
    )


def assemble_profile_modes(
    emtf_dir: str,
    freqs: np.ndarray,
    station_x: np.ndarray,
    profile_ids: list[str] | None = None,
) -> dict[str, object]:
    """Assemble profile-rotated TE and TM without extrapolation.

    The result contains ``lr_*``, ``ph_*`` and ``mask_*`` arrays of shape
    ``(n_frequency, n_model_station)`` plus ``x_model`` and ``x_km``.  Spatial
    interpolation is available only inside the hull of real stations with an
    in-band measurement at that period. The full geographic EMTF tensor is
    first rotated into x-along-profile/y-strike/z-up coordinates. Unsupported
    values remain ``NaN``.
    """
    return _assemble_profile_modes(
        emtf_dir,
        freqs,
        station_x,
        profile_ids=profile_ids or PROFILE_IDS,
    )


def _interpolate_profile_field(
    values: np.ndarray,
    station_mask: np.ndarray,
    x_km: np.ndarray,
    x_model: np.ndarray,
    *,
    circular_period_degrees: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility wrapper around the shared no-extrapolation policy."""
    return _shared_interpolate_profile_field(
        values,
        station_mask,
        x_km,
        x_model,
        circular_period_degrees=circular_period_degrees,
    )


def profile_geometry_metadata(modes: Mapping[str, object]) -> dict[str, object]:
    """Return JSON-safe disclosure for normalized field-profile geometry."""
    required = (
        "geometry_policy",
        "profile_azimuth_deg",
        "profile_span_m",
        "model_station_span_m",
        "horizontal_compression_factor",
        "publishable_physical_geometry",
        "station_ids",
    )
    missing = [key for key in required if key not in modes]
    if missing:
        raise ValueError(f"profile observations lack geometry metadata: {missing}")
    return {
        "geometry_policy": str(modes["geometry_policy"]),
        "profile_azimuth_deg": float(modes["profile_azimuth_deg"]),
        "profile_span_m": float(modes["profile_span_m"]),
        "model_station_span_m": float(modes["model_station_span_m"]),
        "horizontal_compression_factor": float(modes["horizontal_compression_factor"]),
        "publishable_physical_geometry": bool(modes["publishable_physical_geometry"]),
        "station_ids": list(modes["station_ids"]),
    }


def section_nrms(
    section: np.ndarray,
    lr: np.ndarray,
    ph: np.ndarray,
    period_mask: np.ndarray,
    x_model: np.ndarray,
    x_km: np.ndarray,
    periods: np.ndarray,
    depth_grid: np.ndarray,
) -> tuple[float, list[float]]:
    """Static-shift-invariant per-station physics misfit of a section.

    Identical scoring to run_2d_bench.bench_real_profile so every method row
    in the leaderboard is comparable.
    """
    from pimsr_forward.mt1d import mt1d_response
    from pimsr_inversion.data import grid_cell_thicknesses

    thick = grid_cell_thicknesses(depth_grid)
    nrms_list = []
    cols = np.linspace(0, section.shape[1] - 1, len(x_km)).astype(int)
    for j, x in enumerate(cols):
        rho = 10.0 ** section[:, x]
        sim_rho, sim_ph = mt1d_response(rho, thick, periods)
        jx = int(np.argmin(np.abs(x_model - x_km[j])))
        valid = np.asarray(period_mask[:, jx], dtype=bool)
        if not np.any(valid):
            raise ValueError(f"station {j} has no in-band periods")
        d_lr = lr[valid, jx] - np.log10(sim_rho[valid])
        d_lr -= d_lr.mean()
        d_ph = _phase_residual(ph[valid, jx], sim_ph[valid])
        err = np.sqrt(
            np.mean(((d_lr / LOG10_RHO_ERROR) ** 2 + (d_ph / PHASE_ERROR_DEG) ** 2) / 2.0)
        )
        nrms_list.append(float(err))
    return float(np.mean(nrms_list)), nrms_list


def _phase_residual(observed: np.ndarray, simulated: np.ndarray) -> np.ndarray:
    """Shortest signed phase residual under the 180-degree MT convention."""
    return (np.asarray(observed) - np.asarray(simulated) + 90.0) % 180.0 - 90.0


def multimode_nrms(
    observations: Mapping[str, np.ndarray],
    simulations: Mapping[str, np.ndarray],
    *,
    rho_error: float = LOG10_RHO_ERROR,
    phase_error: float = PHASE_ERROR_DEG,
) -> float:
    """Masked, shift-invariant nRMS with equal TE/TM and rho/phase weight.

    Both mappings use explicit keys ``lr_te``, ``ph_te``, ``lr_tm`` and
    ``ph_tm``.  ``observations`` additionally requires ``mask_te`` and
    ``mask_tm``.  A station first averages the two normalised residual
    components within each available mode, then averages TE and TM equally.
    The final score is the arithmetic mean of per-station RMS values.
    """
    rho_error = _finite_scalar(rho_error, name="rho_error")
    phase_error = _finite_scalar(phase_error, name="phase_error")

    required_obs = {"lr_te", "ph_te", "mask_te", "lr_tm", "ph_tm", "mask_tm"}
    missing_obs = required_obs - observations.keys()
    missing_sim = {"lr_te", "ph_te", "lr_tm", "ph_tm"} - simulations.keys()
    if missing_obs:
        raise ValueError(f"observations missing keys: {sorted(missing_obs)}")
    if missing_sim:
        raise ValueError(f"simulations missing keys: {sorted(missing_sim)}")

    shape = np.asarray(observations["lr_te"]).shape
    if len(shape) != 2:
        raise ValueError("mode arrays must have shape (n_frequency, n_station)")
    station_mode_mse: list[list[float]] = [[] for _ in range(shape[1])]
    for mode in ("te", "tm"):
        lr_obs = np.asarray(observations[f"lr_{mode}"], dtype=float)
        ph_obs = np.asarray(observations[f"ph_{mode}"], dtype=float)
        mask = np.asarray(observations[f"mask_{mode}"])
        lr_sim = np.asarray(simulations[f"lr_{mode}"], dtype=float)
        ph_sim = np.asarray(simulations[f"ph_{mode}"], dtype=float)
        arrays = (lr_obs, ph_obs, mask, lr_sim, ph_sim)
        if any(a.shape != shape for a in arrays):
            raise ValueError("all observation, mask and simulation shapes must match")
        if mask.dtype.kind != "b":
            raise ValueError(f"mask_{mode} must be boolean")
        if np.any(~np.isfinite(lr_obs[mask])) or np.any(~np.isfinite(ph_obs[mask])):
            raise ValueError(f"{mode.upper()} observations contain non-finite valid data")
        if np.any(~np.isfinite(lr_sim[mask])) or np.any(~np.isfinite(ph_sim[mask])):
            raise ValueError(f"{mode.upper()} simulation contains non-finite valid data")

        for j in range(shape[1]):
            valid = mask[:, j]
            if not np.any(valid):
                continue
            d_lr = lr_obs[valid, j] - lr_sim[valid, j]
            d_lr -= d_lr.mean()  # per-mode, per-station static shift
            d_ph = _phase_residual(ph_obs[valid, j], ph_sim[valid, j])
            mse = np.mean(((d_lr / rho_error) ** 2 + (d_ph / phase_error) ** 2) / 2.0)
            station_mode_mse[j].append(float(mse))

    missing_stations = [i for i, mode_mse in enumerate(station_mode_mse) if not mode_mse]
    if missing_stations:
        raise ValueError(
            f"period masks select no observations for stations {missing_stations}"
        )
    scores = [float(np.sqrt(np.mean(mode_mse))) for mode_mse in station_mode_mse]
    return float(np.mean(scores))


def section_nrms_2d(
    section: np.ndarray,
    observations: Mapping[str, np.ndarray],
    freqs: np.ndarray,
    station_x: np.ndarray,
    x_grid: np.ndarray,
    depth_grid: np.ndarray,
) -> float:
    """Literal TE+TM 2D-forward score with explicit period masks.

    The per-column 1D score is biased toward laterally-smooth sections; this
    metric uses both polarisations and gives TE/TM and rho/phase equal weight.
    TE is local ``Zyx`` and TM is local ``Zxy``; the same profile-coordinate
    mode names are required from :mod:`pimsr_forward`.
    """
    from types import SimpleNamespace

    from pimsr_forward.mt2d import MT2DForward

    fwd = MT2DForward(frequencies=freqs, station_x=station_x)
    sec = SimpleNamespace(log10_res=section, x_grid=x_grid, depth_grid=depth_grid)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho_te, ph_te = fwd.response_te(sec)
        rho_tm, ph_tm = fwd.response_tm(sec)
    simulations = {
        "lr_te": np.log10(rho_te),
        "ph_te": ph_te,
        "lr_tm": np.log10(rho_tm),
        "ph_tm": ph_tm,
    }
    return multimode_nrms(observations, simulations)


@dataclass
class Hybrid2DResult:
    section: np.ndarray  # refined (n_z, n_x) log10 resistivity
    wall_time_s: float
    n_iterations: int


@dataclass(frozen=True)
class _ProfiledModeData:
    """Flattened SimPEG data order plus log-rho nuisance metadata."""

    dobs_native: np.ndarray
    standard_deviation_native: np.ndarray
    rho_indices: np.ndarray
    phase_indices: np.ndarray
    rho_station_indices: np.ndarray
    observed_log10_rho: np.ndarray
    observed_phase: np.ndarray
    log10_rho_error: np.ndarray
    phase_error: np.ndarray


def _profile_station_constant(
    values: np.ndarray,
    station_indices: np.ndarray,
    standard_deviation: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project station-wise constants out of a weighted residual vector.

    The returned first array is the residual after applying the weighted
    orthogonal projector.  The second contains the fitted constant for every
    station index (missing station indices are ``NaN``).  This small pure
    transform is shared by the objective residual and its Gauss-Newton
    Hessian, so nuisance handling cannot silently diverge between them.
    """
    values = np.asarray(values, dtype=float)
    station_indices = np.asarray(station_indices)
    if values.ndim != 1 or station_indices.ndim != 1:
        raise ValueError("values and station_indices must be one-dimensional")
    if values.shape != station_indices.shape:
        raise ValueError("values and station_indices must have matching shapes")
    if values.size == 0:
        raise ValueError("at least one log-resistivity datum is required")
    if station_indices.dtype.kind not in "iu" or np.any(station_indices < 0):
        raise ValueError("station_indices must contain non-negative integers")
    if np.any(~np.isfinite(values)):
        raise ValueError("values contain non-finite data")

    error = np.broadcast_to(np.asarray(standard_deviation, dtype=float), values.shape)
    if np.any(~np.isfinite(error)) or np.any(error <= 0.0):
        raise ValueError("standard_deviation must be finite and positive")

    profiled = values.copy()
    offsets = np.full(int(station_indices.max()) + 1, np.nan, dtype=float)
    for station in np.unique(station_indices):
        selected = station_indices == station
        weights = 1.0 / error[selected] ** 2
        offset = np.sum(weights * values[selected]) / np.sum(weights)
        profiled[selected] -= offset
        offsets[int(station)] = offset
    return profiled, offsets


def _profile_log10_rho_residual(
    observed: np.ndarray,
    predicted: np.ndarray,
    station_indices: np.ndarray,
    *,
    standard_deviation: float | np.ndarray = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a residual marginalized over station-wise static shifts.

    ``observed = physical + static_shift`` is the nuisance convention.  The
    second return value is the analytically fitted observed-data shift for
    each station.  Crucially, this operation depends on the *current*
    prediction supplied by the objective, never on the inversion start.
    """
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if observed.shape != predicted.shape:
        raise ValueError("observed and predicted must have matching shapes")
    profiled, prediction_minus_observation = _profile_station_constant(
        predicted - observed,
        station_indices,
        standard_deviation,
    )
    return profiled, -prediction_minus_observation


def _pack_profiled_mode_data(
    lr: np.ndarray,
    ph: np.ndarray,
    valid_by_frequency: list[tuple[int, np.ndarray]],
    *,
    phase_offset: float,
    log10_rho_error: float = 0.05,
    phase_error: float = 2.9,
) -> _ProfiledModeData:
    """Pack observations in survey order without model-dependent correction."""
    if log10_rho_error <= 0.0 or phase_error <= 0.0:
        raise ValueError("data errors must be positive")

    dobs_parts: list[np.ndarray] = []
    error_parts: list[np.ndarray] = []
    rho_indices: list[np.ndarray] = []
    phase_indices: list[np.ndarray] = []
    rho_station_indices: list[np.ndarray] = []
    observed_log10_rho: list[np.ndarray] = []
    observed_phase: list[np.ndarray] = []
    cursor = 0
    for frequency_index, valid in valid_by_frequency:
        stations = np.flatnonzero(valid)
        n_valid = len(stations)
        lr_valid = np.asarray(lr[frequency_index, valid], dtype=float)
        ph_valid = np.asarray(ph[frequency_index, valid], dtype=float) + phase_offset
        if np.any(~np.isfinite(lr_valid)) or np.any(~np.isfinite(ph_valid)):
            raise ValueError("valid observations must be finite")
        rho_native = np.power(10.0, lr_valid)
        if np.any(~np.isfinite(rho_native)) or np.any(rho_native <= 0.0):
            raise ValueError("valid apparent resistivity must be finite and positive")

        dobs_parts.append(np.r_[rho_native, ph_valid])
        # These native-space errors are metadata for SimPEG's Data container;
        # the custom objective below applies exact log10-rho and phase errors.
        error_parts.append(
            np.r_[
                np.log(10.0) * log10_rho_error * rho_native,
                np.full(n_valid, phase_error),
            ]
        )
        rho_indices.append(np.arange(cursor, cursor + n_valid, dtype=np.int64))
        phase_indices.append(
            np.arange(cursor + n_valid, cursor + 2 * n_valid, dtype=np.int64)
        )
        rho_station_indices.append(stations.astype(np.int64, copy=False))
        observed_log10_rho.append(lr_valid)
        observed_phase.append(ph_valid)
        cursor += 2 * n_valid

    return _ProfiledModeData(
        dobs_native=np.concatenate(dobs_parts),
        standard_deviation_native=np.concatenate(error_parts),
        rho_indices=np.concatenate(rho_indices),
        phase_indices=np.concatenate(phase_indices),
        rho_station_indices=np.concatenate(rho_station_indices),
        observed_log10_rho=np.concatenate(observed_log10_rho),
        observed_phase=np.concatenate(observed_phase),
        log10_rho_error=np.full(
            sum(len(values) for values in observed_log10_rho),
            log10_rho_error,
        ),
        phase_error=np.full(
            sum(len(values) for values in observed_phase),
            phase_error,
        ),
    )


def _make_profiled_static_shift_misfit(dat, simulation, layout: _ProfiledModeData):
    """Build a SimPEG objective that profiles shifts at every evaluation."""
    from simpeg import data_misfit

    class _ProfiledStaticShiftDataMisfit(data_misfit.BaseDataMisfit):
        def _predicted_components(self, model, f=None):
            predicted = np.asarray(self.simulation.dpred(model, f=f), dtype=float)
            if predicted.shape != layout.dobs_native.shape:
                raise ValueError("simulation prediction has unexpected data shape")
            rho = predicted[layout.rho_indices]
            phase = predicted[layout.phase_indices]
            if np.any(~np.isfinite(rho)) or np.any(rho <= 0.0):
                raise ValueError(
                    "simulation apparent resistivity must be finite and positive"
                )
            if np.any(~np.isfinite(phase)):
                raise ValueError("simulation phase must be finite")
            return predicted, rho, phase

        def __call__(self, model, f=None):
            _, rho, phase = self._predicted_components(model, f)
            rho_residual, _ = _profile_log10_rho_residual(
                layout.observed_log10_rho,
                np.log10(rho),
                layout.rho_station_indices,
                standard_deviation=layout.log10_rho_error,
            )
            phase_residual = _phase_residual(phase, layout.observed_phase)
            return float(
                np.sum((rho_residual / layout.log10_rho_error) ** 2)
                + np.sum((phase_residual / layout.phase_error) ** 2)
            )

        def deriv(self, model, f=None):
            if f is None:
                f = self.simulation.fields(model)
            _, rho, phase = self._predicted_components(model, f)
            rho_residual, _ = _profile_log10_rho_residual(
                layout.observed_log10_rho,
                np.log10(rho),
                layout.rho_station_indices,
                standard_deviation=layout.log10_rho_error,
            )
            phase_residual = _phase_residual(phase, layout.observed_phase)
            data_gradient = np.zeros_like(layout.dobs_native)
            data_gradient[layout.rho_indices] = (
                2.0 * rho_residual / layout.log10_rho_error**2 / (np.log(10.0) * rho)
            )
            data_gradient[layout.phase_indices] = (
                2.0 * phase_residual / layout.phase_error**2
            )
            return self.simulation.Jtvec(model, data_gradient, f=f)

        def deriv2(self, model, v, f=None):
            if f is None:
                f = self.simulation.fields(model)
            _, rho, _ = self._predicted_components(model, f)
            j_vector = np.asarray(
                self.simulation.Jvec_approx(model, v, f=f),
                dtype=float,
            )
            log_rho_j_vector = j_vector[layout.rho_indices] / (np.log(10.0) * rho)
            profiled_j_vector, _ = _profile_station_constant(
                log_rho_j_vector,
                layout.rho_station_indices,
                layout.log10_rho_error,
            )
            data_hessian_vector = np.zeros_like(layout.dobs_native)
            data_hessian_vector[layout.rho_indices] = (
                2.0 * profiled_j_vector / layout.log10_rho_error**2 / (np.log(10.0) * rho)
            )
            data_hessian_vector[layout.phase_indices] = (
                2.0 * j_vector[layout.phase_indices] / layout.phase_error**2
            )
            return self.simulation.Jtvec_approx(model, data_hessian_vector, f=f)

    return _ProfiledStaticShiftDataMisfit(data=dat, simulation=simulation)


def refine_section_2d(
    section0: np.ndarray,
    observations: Mapping[str, np.ndarray],
    freqs: np.ndarray,
    station_x: np.ndarray,
    x_grid: np.ndarray,
    depth_grid: np.ndarray,
    *,
    mode: MTMode,
    max_iter: int = 5,
    max_iter_cg: int = 10,
    alpha_ref: float = 1e-2,
    beta0_ratio: float = 1.0,
) -> Hybrid2DResult:
    """A few SimPEG inexact-GN iterations from a warm-start section.

    ``section0`` is the (n_z, n_x) log10-resistivity starting model (the
    U-Net prediction, or a half-space for the cold-start control).  ``mode``
    is explicit and follows the benchmark contract: TE is ``Zyx`` and TM is
    ``Zxy``.  Only observations selected by ``mask_<mode>`` enter the solve.
    """
    if mode not in {"te", "tm"}:
        raise ValueError(f"mode must be 'te' or 'tm', got {mode!r}")
    for value, name in ((max_iter, "max_iter"), (max_iter_cg, "max_iter_cg")):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
        if value < 1:
            raise ValueError(f"{name} must be positive")
    alpha_ref = _finite_scalar(
        alpha_ref,
        name="alpha_ref",
        strictly_positive=False,
    )
    beta0_ratio = _finite_scalar(beta0_ratio, name="beta0_ratio")
    freqs = _strict_axis(freqs, name="freqs", positive=True)
    station_x = _strict_axis(station_x, name="station_x")
    x_grid = _strict_axis(x_grid, name="x_grid")
    depth_grid = _strict_axis(depth_grid, name="depth_grid", positive=True)
    section0 = np.asarray(section0, dtype=float)
    expected_section_shape = (depth_grid.size, x_grid.size)
    if section0.shape != expected_section_shape or not np.isfinite(section0).all():
        raise ValueError(
            f"section0 must be finite with shape {expected_section_shape}"
        )
    if not isinstance(observations, Mapping):
        raise TypeError("observations must be a mapping")
    required_observations = {f"lr_{mode}", f"ph_{mode}", f"mask_{mode}"}
    missing = required_observations - observations.keys()
    if missing:
        raise ValueError(f"observations missing keys: {sorted(missing)}")
    lr = np.asarray(observations[f"lr_{mode}"], dtype=float)
    ph = np.asarray(observations[f"ph_{mode}"], dtype=float)
    period_mask = np.asarray(observations[f"mask_{mode}"])
    expected_shape = (freqs.size, station_x.size)
    if lr.shape != expected_shape or ph.shape != expected_shape:
        raise ValueError(f"{mode.upper()} observations must have shape {expected_shape}")
    if period_mask.shape != expected_shape or period_mask.dtype.kind != "b":
        raise ValueError(f"mask_{mode} must be a boolean array of shape {expected_shape}")
    if not np.any(period_mask):
        raise ValueError(f"mask_{mode} selects no observations")
    if not np.isfinite(lr[period_mask]).all() or not np.isfinite(ph[period_mask]).all():
        raise ValueError(f"valid {mode.upper()} observations must be finite")

    import discretize  # noqa: F401  (simpeg dependency check)
    from pimsr_forward._mapping import nearest_indices
    from pimsr_forward.mt2d import _AIR_SIGMA, _build_mesh
    from simpeg import (
        data,
        directives,
        inverse_problem,
        inversion,
        maps,
        optimization,
        regularization,
    )
    from simpeg.electromagnetics import natural_source as nsem

    t0 = perf_counter()
    m2 = _build_mesh()
    mesh = m2.mesh
    act = np.zeros(mesh.n_cells, dtype=bool)
    act[m2.active_idx] = True

    # model = log conductivity of subsurface cells
    sigma_map = maps.ExpMap(mesh) * maps.InjectActiveCells(mesh, act, np.log(_AIR_SIGMA))

    # warm-start model: nearest-neighbour sample of the section
    cc = m2.active_cc
    ix = nearest_indices(x_grid, cc[:, 0])
    iz = nearest_indices(depth_grid, -cc[:, 1])
    m0 = -np.log(10.0) * section0[iz, ix]  # log sigma = -ln10 * log10 rho

    # Match pimsr_forward's public profile-mode contract. A masked survey is
    # built here because the public full-grid simulations cannot represent
    # station/period-specific missing observations.
    if mode == "te":
        orientation = "yx"
        simulation_cls = nsem.simulation.Simulation2DMagneticField
        phase_offset = 0.0
    elif mode == "tm":
        orientation = "xy"
        simulation_cls = nsem.simulation.Simulation2DElectricField
        phase_offset = -180.0

    srcs = []
    valid_by_frequency = []
    for i, frequency in enumerate(freqs):
        valid = period_mask[i]
        if not np.any(valid):
            continue
        rx_locs = np.c_[station_x[valid], np.zeros(int(valid.sum()))]
        rx = [
            nsem.receivers.Impedance(
                rx_locs,
                orientation=orientation,
                component="apparent_resistivity",
            ),
            nsem.receivers.Impedance(rx_locs, orientation=orientation, component="phase"),
        ]
        srcs.append(nsem.sources.Planewave(rx, frequency=frequency))
        valid_by_frequency.append((i, valid))
    survey = nsem.survey.Survey(srcs)
    sim = simulation_cls(mesh, survey=survey, sigmaMap=sigma_map)

    # Profile (analytically marginalize) one log10-rho constant per station at
    # every objective/Jacobian evaluation. Unlike the former one-shot
    # correction, this data objective is identical for warm and cold starts.
    layout = _pack_profiled_mode_data(
        lr,
        ph,
        valid_by_frequency,
        phase_offset=phase_offset,
    )
    dat = data.Data(
        survey,
        dobs=layout.dobs_native,
        standard_deviation=layout.standard_deviation_native,
    )

    dmis = _make_profiled_static_shift_misfit(dat, sim, layout)
    reg = regularization.WeightedLeastSquares(
        mesh,
        active_cells=act,
        reference_model=m0,
        alpha_s=alpha_ref,
        alpha_x=1.0,
        alpha_y=1.0,
    )
    opt = optimization.InexactGaussNewton(maxIter=max_iter, maxIterCG=max_iter_cg)
    prob = inverse_problem.BaseInvProblem(dmis, reg, opt)
    dirs = [
        directives.BetaEstimate_ByEig(beta0_ratio=beta0_ratio),
        directives.BetaSchedule(coolingFactor=2.0, coolingRate=1),
    ]
    inv = inversion.BaseInversion(prob, directiveList=dirs)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_rec = inv.run(m0)

    # map recovered log-sigma back onto the section grid (nearest active cell)
    from scipy.spatial import cKDTree

    tree = cKDTree(cc)
    zz, xx = np.meshgrid(-depth_grid, x_grid, indexing="ij")
    _, idx = tree.query(np.c_[xx.ravel(), zz.ravel()])
    section = (-m_rec[idx] / np.log(10.0)).reshape(len(depth_grid), len(x_grid))

    return Hybrid2DResult(
        section=section,
        wall_time_s=perf_counter() - t0,
        n_iterations=int(opt.iter),
    )
