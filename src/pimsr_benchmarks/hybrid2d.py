"""2D hybrid inversion: U-Net warm start + SimPEG Gauss-Newton refinement.

The 1D hybrid (neural start -> Occam) closed the real-data gap entirely at
2.4x classical speed. This is the 2D analogue: the U-Net's laterally
coherent section becomes the starting and reference model for a few
inexact-Gauss-Newton iterations of the rigorous SimPEG TE-mode inversion.

Scale caveat (shared by every 2D row in the benchmark): the real profile is
compressed onto the training section's 24 km frame, so the recovered image
is a scaled structural sketch, not a metrically correct crustal section.
The per-station physics misfit used for scoring is invariant to that
compression.
"""

from __future__ import annotations

import glob
import warnings
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .emtf import parse_emtf_xml, resample_station

__all__ = [
    "assemble_profile",
    "assemble_profile_modes",
    "section_nrms",
    "refine_section_2d",
    "Hybrid2DResult",
]

#: E-W profile at ~44.6N, west to east (same as run_2d_bench).
PROFILE_IDS = ["MTH15", "MTH16", "WYYS1", "WYYS2", "WYYS3", "WYH18", "WYH19"]

#: All five E-W USArray rows in the region, west to east. "H-YS" is the
#: original Yellowstone profile; the others are independent test lines.
PROFILES = {
    "G": ["MTG15", "MTG16", "MTG17", "MTG18", "MTG19"],
    "H-YS": PROFILE_IDS,
    "I": ["IDI15", "IDI16", "WYI17", "WYI18", "WYI19"],
    "J": ["IDJ15", "IDJ16", "WYJ17", "WYJ18", "WYJ19"],
    "K": ["IDK15", "IDK16", "WYK17", "WYK18", "WYK19"],
}


def assemble_profile(
    emtf_dir: str,
    freqs: np.ndarray,
    station_x: np.ndarray,
    profile_ids: list[str] | None = None,
):
    """Interpolate a USArray profile onto the model station grid.

    Returns (lr, ph, x_model, x_km): log10 apparent resistivity and phase of
    shape (n_freq, n_station), model station coordinates (km), and the true
    station coordinates (km).
    """
    stations = {}
    for f in glob.glob(f"{emtf_dir}/*.xml"):
        st = parse_emtf_xml(f)
        stations[st.station_id] = st
    profile = [stations[i] for i in (profile_ids or PROFILE_IDS)]

    periods = 1.0 / freqs
    n_f, n_s = len(freqs), len(station_x)

    lon = np.array([s.longitude for s in profile])
    x_km = (lon - lon.min()) * 111.0 * np.cos(np.radians(44.6))
    x_model = np.linspace(x_km.min(), x_km.max(), n_s)

    lr_st = np.empty((n_f, len(profile)))
    ph_st = np.empty((n_f, len(profile)))
    for j, st in enumerate(profile):
        lr_st[:, j], ph_st[:, j], _ = resample_station(st, periods)
    lr = np.stack([np.interp(x_model, x_km, lr_st[i]) for i in range(n_f)])
    ph = np.stack([np.interp(x_model, x_km, ph_st[i]) for i in range(n_f)])
    return lr, ph, x_model, x_km


def assemble_profile_modes(
    emtf_dir: str,
    freqs: np.ndarray,
    station_x: np.ndarray,
    profile_ids: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Per-mode variant of :func:`assemble_profile` for v3 4-channel nets.

    Returns ``{"lr_te", "ph_te", "lr_tm", "ph_tm", "x_model", "x_km"}``.
    """
    from .emtf import resample_station_modes

    stations = {}
    for f in glob.glob(f"{emtf_dir}/*.xml"):
        st = parse_emtf_xml(f)
        stations[st.station_id] = st
    profile = [stations[i] for i in (profile_ids or PROFILE_IDS)]

    periods = 1.0 / freqs
    n_f, n_s = len(freqs), len(station_x)

    lon = np.array([s.longitude for s in profile])
    x_km = (lon - lon.min()) * 111.0 * np.cos(np.radians(44.6))
    x_model = np.linspace(x_km.min(), x_km.max(), n_s)

    out: dict[str, np.ndarray] = {"x_model": x_model, "x_km": x_km}
    st_modes = [resample_station_modes(st, periods) for st in profile]
    for key in ("lr_te", "ph_te", "lr_tm", "ph_tm"):
        arr = np.stack([m[key] for m in st_modes], axis=1)  # (n_f, n_prof)
        out[key] = np.stack(
            [np.interp(x_model, x_km, arr[i]) for i in range(n_f)]
        )
    return out


def section_nrms(
    section: np.ndarray,
    lr: np.ndarray,
    ph: np.ndarray,
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

    thick = np.diff(depth_grid)
    nrms_list = []
    cols = np.linspace(0, section.shape[1] - 1, len(x_km)).astype(int)
    for j, x in enumerate(cols):
        rho = 10.0 ** section[:, x]
        sim_rho, sim_ph = mt1d_response(rho, thick, periods)
        jx = int(np.argmin(np.abs(x_model - x_km[j])))
        d_lr = lr[:, jx] - np.log10(sim_rho)
        d_lr -= d_lr.mean()
        d_ph = ph[:, jx] - sim_ph
        err = np.sqrt(np.mean(d_lr**2 / 0.05**2 + (d_ph / 2.9) ** 2 / 2.0))
        nrms_list.append(float(err))
    return float(np.mean(nrms_list)), nrms_list


def section_nrms_2d(
    section: np.ndarray,
    lr: np.ndarray,
    ph: np.ndarray,
    freqs: np.ndarray,
    station_x: np.ndarray,
    x_grid: np.ndarray,
    depth_grid: np.ndarray,
) -> float:
    """Physics misfit using the rigorous 2D forward (same weights as 1D).

    The per-column 1D score is biased toward laterally-smooth sections; this
    is the fair metric for methods that exploit true 2D physics. Static
    shift is removed per station, matching the 1D scoring convention.
    """
    from types import SimpleNamespace

    from pimsr_forward.mt2d import MT2DForward

    fwd = MT2DForward(frequencies=freqs, station_x=station_x)
    sec = SimpleNamespace(
        log10_res=section, x_grid=x_grid, depth_grid=depth_grid
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim_rho, sim_ph = fwd.response(sec)
    d_lr = lr - np.log10(sim_rho)
    d_lr -= d_lr.mean(axis=0, keepdims=True)  # per-station static shift
    d_ph = ph - sim_ph
    per_station = np.sqrt(
        np.mean(d_lr**2 / 0.05**2 + (d_ph / 2.9) ** 2 / 2.0, axis=0)
    )
    return float(per_station.mean())


@dataclass
class Hybrid2DResult:
    section: np.ndarray  # refined (n_z, n_x) log10 resistivity
    wall_time_s: float
    n_iterations: int


def refine_section_2d(
    section0: np.ndarray,
    lr: np.ndarray,
    ph: np.ndarray,
    freqs: np.ndarray,
    station_x: np.ndarray,
    x_grid: np.ndarray,
    depth_grid: np.ndarray,
    max_iter: int = 5,
    max_iter_cg: int = 10,
    alpha_ref: float = 1e-2,
    beta0_ratio: float = 1.0,
) -> Hybrid2DResult:
    """A few SimPEG inexact-GN iterations from a warm-start section.

    ``section0`` is the (n_z, n_x) log10-resistivity starting model (the
    U-Net prediction, or a half-space for the cold-start control).
    """
    import discretize  # noqa: F401  (simpeg dependency check)
    from simpeg import (
        data,
        data_misfit,
        directives,
        inverse_problem,
        inversion,
        maps,
        optimization,
        regularization,
    )
    from simpeg.electromagnetics import natural_source as nsem

    from pimsr_forward.mt2d import _AIR_SIGMA, _build_mesh

    t0 = perf_counter()
    m2 = _build_mesh()
    mesh = m2.mesh
    act = np.zeros(mesh.n_cells, dtype=bool)
    act[m2.active_idx] = True

    # model = log conductivity of subsurface cells
    sigma_map = maps.ExpMap(mesh) * maps.InjectActiveCells(
        mesh, act, np.log(_AIR_SIGMA)
    )

    # warm-start model: nearest-neighbour sample of the section
    cc = m2.active_cc
    ix = np.clip(np.searchsorted(x_grid, cc[:, 0]), 0, len(x_grid) - 1)
    iz = np.clip(np.searchsorted(depth_grid, -cc[:, 1]), 0, len(depth_grid) - 1)
    m0 = -np.log(10.0) * section0[iz, ix]  # log sigma = -ln10 * log10 rho

    rx_locs = np.c_[station_x, np.zeros_like(station_x)]
    rx = [
        nsem.receivers.Impedance(rx_locs, orientation="xy", component="apparent_resistivity"),
        nsem.receivers.Impedance(rx_locs, orientation="xy", component="phase"),
    ]
    srcs = [nsem.sources.Planewave(rx, frequency=f) for f in freqs]
    survey = nsem.survey.Survey(srcs)
    sim = nsem.simulation.Simulation2DElectricField(
        mesh, survey=survey, sigmaMap=sigma_map
    )

    # Static-shift correction: the scoring metric is shift-invariant, so the
    # inversion must not bend structure to fit per-station DC offsets. We
    # estimate each station's shift against the starting model's response
    # (standard practice with a reference model) and remove it.
    from types import SimpleNamespace

    from pimsr_forward.mt2d import MT2DForward

    fwd0 = MT2DForward(frequencies=freqs, station_x=station_x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho0, _ = fwd0.response(
            SimpleNamespace(log10_res=section0, x_grid=x_grid, depth_grid=depth_grid)
        )
    shift = (lr - np.log10(rho0)).mean(axis=0, keepdims=True)  # (1, n_station)

    # observed data in SimPEG's native convention (phase in -180..-90)
    rho_obs = 10.0 ** (lr - shift)
    dobs = np.concatenate(
        [np.r_[rho_obs[i], ph[i] - 180.0] for i in range(len(freqs))]
    )
    std = np.concatenate(
        [np.r_[0.12 * np.abs(rho_obs[i]), np.full(len(station_x), 2.9)]
         for i in range(len(freqs))]
    )
    dat = data.Data(survey, dobs=dobs, standard_deviation=std)

    dmis = data_misfit.L2DataMisfit(data=dat, simulation=sim)
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
        n_iterations=max_iter,
    )
