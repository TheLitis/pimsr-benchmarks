#!/usr/bin/env python
"""Run ModEM (Mod2DMT, open-source github.com/magnetotellurics/ModEM)
on a real USArray profile and score it with the shift-invariant
2D-forward metric used across the project.

Pipeline:
  1. per-mode TE/TM observations from EMTF XMLs (same mapping as v3:
     TE=Zyx, TM=Zxy for an E-W profile with N-S strike, phases in 0..90)
  2. rho_a/phase -> complex impedances in ModEM units ([V/m]/[T] =
     Z_SI/mu0) under the exp(-i omega t) convention:
       TE: Z = |Z| e^{-i phi}      (re>0, im<0)
       TM: Z = |Z| e^{i(180-phi)}  (re<0, im>0)
  3. Mackie-format LOGE prior (100 Ohm-m halfspace) on a padded mesh
  4. Mod2DMT -I NLCG prior.rho data.dat
  5. parse final NLCG iteration model, interpolate onto the project
     (depth_grid, x_grid) raster, score with section_nrms_2d

Usage:
  python scripts/run_modem2d.py --emtf-dir data/emtf \
      --binary /tmp/ModEM/f90/Mod2DMT \
      --test-h5 /vercel/share/pimsr-data/v3/ds2d_test.h5 \
      --profile H-YS --out results/modem2d/modem2d_HYS.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np

from pimsr_benchmarks.hybrid2d import (
    PROFILES,
    assemble_profile_modes,
    section_nrms_2d,
)

MU0 = 4.0e-7 * np.pi
ERROR_FLOOR = 0.10  # fraction of |Z|


# ----------------------------------------------------------------- mesh

def build_mesh(x_km: np.ndarray) -> dict:
    """Padded 2D mesh: core columns spanning the stations + geometric
    padding; log-spaced depth layers 50 m .. ~120 km."""
    span = (x_km.max() - x_km.min()) * 1e3  # m
    core_dy = span / 48.0
    n_core = 52  # core columns (stations sit inside with margin)
    dy_core = np.full(n_core, core_dy)
    pad = [core_dy * 1.4**i for i in range(1, 11)]
    dy = np.array(pad[::-1] + list(dy_core) + pad)

    dz = np.logspace(np.log10(50.0), np.log10(1.2e4), 40)
    dz = np.concatenate([dz, dz[-1] * 1.4 ** np.arange(1, 9)])

    # station y positions: left padding + margin + fractional position
    pad_w = sum(pad)
    margin = 2 * core_dy
    core_w = n_core * core_dy
    usable = core_w - 2 * margin
    xn = (x_km - x_km.min()) / max(x_km.max() - x_km.min(), 1e-9)
    st_y = pad_w + margin + xn * usable
    return {"dy": dy, "dz": dz, "st_y": st_y, "pad_w": pad_w}


def write_prior(path: Path, mesh: dict, rho0: float = 100.0) -> None:
    dy, dz = mesh["dy"], mesh["dz"]
    ny, nz = len(dy), len(dz)
    ln_rho = np.log(rho0)
    with open(path, "w") as f:
        f.write(f"{ny} {nz} LOGE\n")
        for arr in (dy, dz):
            vals = [f"{v:.4E}" for v in arr]
            for i in range(0, len(vals), 10):
                f.write(" ".join(vals[i : i + 10]) + "\n")
        f.write("1\n")  # skipped record before values
        row = " ".join([f"{ln_rho:.6E}"] * ny)
        for _ in range(nz):
            f.write(row + "\n")


# ----------------------------------------------------------------- data

def write_data(
    path: Path, modes: dict, periods: np.ndarray, st_y: np.ndarray
) -> int:
    """Write TE+TM impedance blocks at the real station columns."""
    n_st = len(st_y)
    lines_te, lines_tm = [], []
    for j in range(n_st):
        for i, t in enumerate(periods):
            omega = 2.0 * np.pi / t
            for mode, lr_key, ph_key, out in (
                ("TE", "lr_te", "ph_te", lines_te),
                ("TM", "lr_tm", "ph_tm", lines_tm),
            ):
                rho = 10.0 ** modes[lr_key][i, j]
                ph = np.radians(modes[ph_key][i, j])
                zmag = np.sqrt(rho * omega * MU0) / MU0
                if mode == "TE":
                    zre, zim = zmag * np.cos(ph), -zmag * np.sin(ph)
                else:
                    zre = -zmag * np.cos(ph)
                    zim = zmag * np.sin(ph)
                err = ERROR_FLOOR * zmag
                out.append(
                    f"{t:.6E} {j + 1:03d}    0.000    0.000        0.000"
                    f"   {st_y[j]:.3f}   0.000 {mode}"
                    f"    {zre:.6E}    {zim:.6E}    {err:.6E}"
                )
    with open(path, "w") as f:
        for name, lines in (("TE_Impedance", lines_te), ("TM_Impedance", lines_tm)):
            f.write("# PIMSR real-profile export\n")
            f.write(
                "# Period(s) Code GG_Lat GG_Lon X(m) Y(m) Z(m) "
                "Component Real Imag Error\n"
            )
            f.write(f"> {name}\n> exp(-i\\omega t)\n> [V/m]/[T]\n> 0.00\n")
            f.write(f"> 0.000 0.000\n> {len(periods)} {n_st}\n")
            f.write("\n".join(lines) + "\n")
    return len(lines_te) + len(lines_tm)


# ------------------------------------------------------------- parsing

def read_final_model(workdir: Path) -> tuple[np.ndarray, int]:
    """Latest Modular_NLCG_NNN.rho -> (ln_rho[nz, ny] array, iteration)."""
    files = sorted(workdir.glob("Modular_NLCG_*.rho"))
    if not files:
        raise RuntimeError("no NLCG output models found")
    final = files[-1]
    it = int(re.search(r"_(\d+)\.rho$", final.name).group(1))
    tokens = final.read_text().split()
    ny, nz = int(tokens[0]), int(tokens[1])
    assert tokens[2] == "LOGE"
    vals = np.array([float(v) for v in tokens[3:]])
    dy, vals = vals[:ny], vals[ny:]
    dz, vals = vals[:nz], vals[nz:]
    vals = vals[1:]  # skipped record token ("1")
    ln_rho = vals[: ny * nz].reshape(nz, ny)
    return ln_rho, it, dy, dz


def to_project_raster(
    ln_rho: np.ndarray, dy: np.ndarray, dz: np.ndarray,
    mesh: dict, x_km: np.ndarray, x_grid: np.ndarray, depth_grid: np.ndarray,
) -> np.ndarray:
    """Interpolate the ModEM model onto the project (depth, x) raster."""
    yc = np.cumsum(dy) - dy / 2.0
    zc = np.cumsum(dz) - dz / 2.0
    # map project x_grid (normalised to the station span) into mesh y
    xn = (x_grid - x_grid.min()) / max(x_grid.max() - x_grid.min(), 1e-9)
    st_y = mesh["st_y"]
    y_t = st_y.min() + xn * (st_y.max() - st_y.min())
    z_t = depth_grid[:-1] + np.diff(depth_grid) / 2.0
    iy = np.searchsorted(yc, y_t).clip(0, len(yc) - 1)
    iz = np.searchsorted(zc, z_t).clip(0, len(zc) - 1)
    log10_rho = ln_rho / np.log(10.0)
    sec = log10_rho[np.ix_(iz, iy)]
    # project raster has len(depth_grid) rows in the net convention
    if len(depth_grid) > sec.shape[0]:
        sec = np.vstack([sec, sec[-1:]])
    return sec


# ----------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emtf-dir", required=True)
    ap.add_argument("--binary", required=True)
    ap.add_argument("--test-h5", required=True, help="grid/frequency reference")
    ap.add_argument("--profile", default="H-YS", choices=sorted(PROFILES))
    ap.add_argument("--workdir", default="/tmp/modem2d_run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()

    with h5py.File(args.test_h5, "r") as f:
        freqs = f["frequencies"][:]
        station_x = f["station_x"][:]
        x_grid = f["x_grid"][:]
        depth_grid = f["depth_grid"][:]
    periods = 1.0 / freqs

    modes = assemble_profile_modes(
        args.emtf_dir, freqs, station_x, profile_ids=PROFILES[args.profile]
    )
    x_km = modes["x_km"]
    # station-column observations (n_freq, n_station_model): reuse the
    # model-station raster for the data so the metric reference matches
    mesh = build_mesh(np.asarray(x_km))
    # place data at the interpolated model stations (same as U-Net input)
    st_y = mesh["pad_w"] + (
        (np.asarray(station_x) - station_x.min())
        / max(station_x.max() - station_x.min(), 1e-9)
        * (mesh["st_y"].max() - mesh["st_y"].min())
        + (mesh["st_y"].min() - mesh["pad_w"])
    )

    wd = Path(args.workdir)
    wd.mkdir(parents=True, exist_ok=True)
    write_prior(wd / "prior.rho", mesh)
    n_data = write_data(wd / "data.dat", modes, periods, st_y)

    t0 = time.time()
    proc = subprocess.run(
        [args.binary, "-I", "NLCG", "prior.rho", "data.dat"],
        cwd=wd, capture_output=True, text=True, timeout=args.timeout,
    )
    runtime = time.time() - t0
    rms_hist = [float(m) for m in re.findall(r"rms=\s*([0-9.Ee+-]+)", proc.stdout)]

    ln_rho, it, dy, dz = read_final_model(wd)
    sec = to_project_raster(ln_rho, dy, dz, mesh, x_km, x_grid, depth_grid)
    nrms = section_nrms_2d(
        sec, modes["lr_te"], modes["ph_te"], freqs, station_x, x_grid, depth_grid
    )

    result = {
        "code": "ModEM Mod2DMT (open-source, NLCG)",
        "profile": args.profile,
        "n_data": n_data,
        "error_floor": ERROR_FLOOR,
        "iterations": it,
        "modem_final_rms": rms_hist[-1] if rms_hist else None,
        "section_nrms_2d": float(nrms),
        "runtime_s": round(runtime, 1),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
