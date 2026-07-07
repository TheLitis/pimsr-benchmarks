"""Unified leaderboard: every method scored with the rigorous 2D forward.

Motivation (see REPORT.md "2D hybrid experiment"): per-station 1D scoring is
biased and incomparable across method families. Here every method produces a
full (nz, nx) section, and all sections are scored with the same
shift-invariant 2D-forward misfit (``section_nrms_2d``) on the real
Yellowstone profile — plus optionally on denser station profiles.

1D per-station methods are converted to sections by inverting each station
column independently and interpolating laterally between stations, which is
exactly how 1D results are used in practice (stitched sections).
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import h5py
import numpy as np

warnings.filterwarnings("ignore")

from pimsr_benchmarks.hybrid import hybrid_invert  # noqa: E402
from pimsr_benchmarks.hybrid2d import (  # noqa: E402
    assemble_profile,
    refine_section_2d,
    section_nrms_2d,
)
from pimsr_benchmarks.neural import NeuralInverter  # noqa: E402
from pimsr_benchmarks.occam1d import occam1d_invert  # noqa: E402


def columns_to_section(
    cols: np.ndarray, station_x: np.ndarray, x_grid: np.ndarray
) -> np.ndarray:
    """Laterally interpolate per-station columns (nz, n_st) onto x_grid."""
    nz = cols.shape[0]
    return np.stack(
        [np.interp(x_grid, station_x, cols[i]) for i in range(nz)]
    )


def occam_section(lr, ph, periods, station_x, x_grid, depth_grid):
    cols = []
    for j in range(lr.shape[1]):
        res = occam1d_invert(lr[:, j], ph[:, j], periods, max_iterations=30)
        cols.append(res.profile_on_grid(depth_grid))
    return columns_to_section(np.stack(cols, axis=1), station_x, x_grid)


def neural_1d_section(inv, lr, ph, periods, station_x, x_grid, depth_grid):
    cols = []
    for j in range(lr.shape[1]):
        n_lr = np.interp(inv.periods, periods, lr[:, j])
        n_ph = np.interp(inv.periods, periods, ph[:, j])
        pred = inv.invert(n_lr, n_ph)
        cols.append(np.interp(depth_grid, inv.depth_grid, pred.log10_rho))
    return columns_to_section(np.stack(cols, axis=1), station_x, x_grid)


def hybrid_1d_section(inv, lr, ph, periods, station_x, x_grid, depth_grid):
    cols = []
    for j in range(lr.shape[1]):
        n_lr = np.interp(inv.periods, periods, lr[:, j])
        n_ph = np.interp(inv.periods, periods, ph[:, j])
        res = hybrid_invert(
            inv, lr[:, j], ph[:, j], periods,
            neural_log_rho_a=n_lr, neural_phase=n_ph,
        )
        cols.append(res.occam.profile_on_grid(depth_grid))
    return columns_to_section(np.stack(cols, axis=1), station_x, x_grid)


def unet_section(checkpoint, lr, ph, modes=None):
    """Single U-Net pass. For 4-channel (TE+TM) v3 models pass ``modes``
    from :func:`assemble_profile_modes`."""
    import sys

    import torch

    sys.path.insert(0, str(Path(__file__).parent))
    from run_2d_hybrid_bench import load_model

    model, ckpt = load_model(checkpoint)
    if model.in_channels == 4:
        if modes is None:
            raise ValueError("4-channel checkpoint requires modes=...")
        obs = np.stack(
            [modes["lr_te"], modes["ph_te"] / 45.0,
             modes["lr_tm"], modes["ph_tm"] / 45.0]
        )[None].astype(np.float32)
    else:
        obs = np.stack([lr, ph / 45.0])[None].astype(np.float32)
    obs = (obs - ckpt["stats_mean"]) / ckpt["stats_std"]
    with torch.no_grad():
        out = model(torch.from_numpy(obs.astype(np.float32)))
    return out["log_rho"][0].numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-h5", required=True)
    ap.add_argument("--emtf-dir", default="data/emtf")
    ap.add_argument("--ckpt-1d", required=True)
    ap.add_argument("--ckpt-10k", required=True)
    ap.add_argument("--ckpt-10k-ft", required=True)
    ap.add_argument("--ckpt-60k", required=True)
    ap.add_argument("--ckpt-60k-ft", required=True)
    ap.add_argument("--ckpt-v3", default=None, help="TE+TM 4-channel checkpoint")
    ap.add_argument("--ckpt-v3-ft", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-gn", action="store_true", help="skip slow 2D GN rows")
    args = ap.parse_args()

    with h5py.File(args.test_h5) as f:
        freqs = f["frequencies"][:]
        station_x = f["station_x"][:]
        x_grid = f["x_grid"][:]
        depth_grid = f["depth_grid"][:]
    periods = 1.0 / freqs

    lr, ph, _, _ = assemble_profile(args.emtf_dir, freqs, station_x)

    def score(section):
        return section_nrms_2d(
            section, lr, ph, freqs, station_x, x_grid, depth_grid
        )

    board: dict[str, dict] = {}

    def add(name, section, wall=None):
        board[name] = {"nrms_2d": score(section)}
        if wall is not None:
            board[name]["wall_time_s"] = wall
        print(f"{name:28s} | 2D nRMS {board[name]['nrms_2d']:.2f}", flush=True)

    # ---- 1D family ------------------------------------------------------
    import time

    t0 = time.perf_counter()
    add("occam1d-stitched", occam_section(lr, ph, periods, station_x, x_grid, depth_grid), time.perf_counter() - t0)

    inv = NeuralInverter(args.ckpt_1d)
    t0 = time.perf_counter()
    add("neural1d-stitched", neural_1d_section(inv, lr, ph, periods, station_x, x_grid, depth_grid), time.perf_counter() - t0)

    t0 = time.perf_counter()
    add("hybrid1d-stitched", hybrid_1d_section(inv, lr, ph, periods, station_x, x_grid, depth_grid), time.perf_counter() - t0)

    # ---- 2D neural family -----------------------------------------------
    add("unet-10k", unet_section(args.ckpt_10k, lr, ph))
    add("unet-10k-ft", unet_section(args.ckpt_10k_ft, lr, ph))
    add("unet-60k", unet_section(args.ckpt_60k, lr, ph))
    add("unet-60k-ft", unet_section(args.ckpt_60k_ft, lr, ph))

    if args.ckpt_v3:
        from pimsr_benchmarks.hybrid2d import assemble_profile_modes

        modes = assemble_profile_modes(args.emtf_dir, freqs, station_x)
        add("unet-v3-tetm", unet_section(args.ckpt_v3, lr, ph, modes=modes))
        if args.ckpt_v3_ft:
            add("unet-v3-tetm-ft", unet_section(args.ckpt_v3_ft, lr, ph, modes=modes))

    # ---- 2D iterative ----------------------------------------------------
    if not args.skip_gn:
        warm = unet_section(args.ckpt_60k, lr, ph)
        hy = refine_section_2d(
            warm, lr, ph, freqs, station_x, x_grid, depth_grid,
            max_iter=8, beta0_ratio=10.0, alpha_ref=1e-2,
        )
        add("hybrid2d-gn8", hy.section, hy.wall_time_s)

        cold = refine_section_2d(
            np.full_like(warm, 2.0), lr, ph, freqs, station_x, x_grid,
            depth_grid, max_iter=25, beta0_ratio=10.0, alpha_ref=1e-2,
        )
        add("cold-gn25", cold.section, cold.wall_time_s)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(board, f, indent=1)
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
