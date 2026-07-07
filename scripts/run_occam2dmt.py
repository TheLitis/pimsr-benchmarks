#!/usr/bin/env python
"""Production comparison: Scripps Occam2DMT v3.0 on a real USArray profile.

Generates DATA/MESH/MODEL/startup files for the classical Occam 2D MT
inversion code (deGroot-Hedlin & Constable 1990; v3.0, Fortran 90), runs
the binary, parses the final iteration model, and scores it with the same
section_nrms_2d metric used in the unified leaderboard.

Error floors in the DATA file match the leaderboard weights exactly
(0.05 log10 rho-a, 2.9 deg phase), so Occam's own converged RMS is
directly comparable to every other leaderboard row.

Usage:
  python scripts/run_occam2dmt.py --emtf-dir data/emtf \
      --binary /tmp/occam2d/OCCAM2DMT_V3.0/Source/occam2d \
      --test-h5 /vercel/share/pimsr-data/v3/ds2d_test.h5 \
      --out results/occam2dmt/occam2dmt.json \
      [--profile H-YS] [--modes te,tm] [--max-iter 25]
"""
from __future__ import annotations

import argparse
import json
import shutil
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

RHO_ERR = 0.05  # log10 units — identical to section_nrms_2d weights
PH_ERR = 2.9  # degrees


# --------------------------------------------------------------------------
# mesh construction
# --------------------------------------------------------------------------
def build_mesh(x_km: np.ndarray, depth_max_m: float):
    """Column/row layout: uniform core columns snapped to stations plus
    geometric padding; log-growing rows. Returns dict with all geometry."""
    x_m = x_km * 1e3
    span = x_m.max() - x_m.min()
    dx = span / 40.0  # ~40 core columns across the profile
    # core nodes: uniform grid; stations snap to nearest node
    n_core = 45
    core_left = x_m.min() - 2 * dx
    core_widths = np.full(n_core, dx)
    pads = [dx * 3**i for i in range(1, 7)]  # 3x growth, 6 pads each side
    col_widths = np.array(pads[::-1] + list(core_widths) + pads)
    n_pad = len(pads)

    # rows: 300 m first, growing ~1.35x to reach several skin depths
    row_heights = [300.0]
    while sum(row_heights) < depth_max_m:
        row_heights.append(row_heights[-1] * 1.35)
    row_heights += [row_heights[-1] * 3, row_heights[-1] * 9, row_heights[-1] * 27]
    row_heights = np.array(row_heights)

    return {
        "col_widths": col_widths,
        "row_heights": row_heights,
        "n_pad": n_pad,
        "n_core": n_core,
        "core_left": core_left,
        "dx": dx,
    }


def write_mesh(path: Path, mesh: dict) -> None:
    cw, rh = mesh["col_widths"], mesh["row_heights"]
    n_col, n_row = len(cw), len(rh)
    lines = ["PIMSR auto-generated mesh for Occam2DMT"]
    lines.append(f"     0 {n_col + 1:5d} {n_row + 1:5d}     0   0   2")

    def fmt(vals):
        out, row = [], []
        for v in vals:
            row.append(f"{v:10.1f}")
            if len(row) == 8:
                out.append("".join(row))
                row = []
        if row:
            out.append("".join(row))
        return out

    lines += fmt(cw)
    lines += fmt(rh)
    lines.append("     0")
    for _ in range(4 * n_row):  # 4 triangle codes per element row (PW2D)
        lines.append("?" * n_col)
    path.write_text("\n".join(lines) + "\n")


def build_layers(mesh: dict):
    """Group mesh rows into parameter layers and mesh columns into
    parameter columns (padding merged into the edge parameters)."""
    n_row = len(mesh["row_heights"])
    layers = []
    i = 0
    step = 1
    while i < n_row:
        take = min(int(round(step)), n_row - i)
        layers.append(take)
        i += take
        step *= 1.25  # deeper layers span more mesh rows
    # column spec: merge pads + 1 core col at each edge, core cols in pairs
    n_pad, n_core = mesh["n_pad"], mesh["n_core"]
    col_spec = [n_pad + 1]
    remaining = n_core - 2
    while remaining > 0:
        take = min(2, remaining)
        col_spec.append(take)
        remaining -= take
    col_spec.append(n_pad + 1)
    return layers, col_spec


def write_model(path: Path, mesh_file: str, mesh: dict, layers, col_spec) -> None:
    lines = [
        "FORMAT:           OCCAM2MTMOD_1.0",
        "MODEL NAME:       PIMSR real-profile comparison",
        "DESCRIPTION:      auto-generated",
        f"MESH FILE:        {mesh_file}",
        "MESH TYPE:        PW2D",
        "STATICS FILE:     none",
        "PREJUDICE FILE:   none",
        f"BINDING OFFSET:   {mesh['core_left'] - sum(mesh['col_widths'][:mesh['n_pad']]):.1f}",
        f"NUM LAYERS:       {len(layers)}",
    ]
    for take in layers:
        lines.append(f"{take}  {len(col_spec)}")
        lines.append(" " + " ".join(str(c) for c in col_spec))
    lines.append("NO. EXCEPTIONS:   0")
    path.write_text("\n".join(lines) + "\n")


def write_data(
    path: Path, x_km, periods, station_modes, modes=("te", "tm"), mask=None
) -> int:
    n_st = len(x_km)
    freqs_hz = 1.0 / periods
    lines = [
        "FORMAT:           OCCAM2MTDATA_1.0",
        "TITLE:            PIMSR USArray profile",
        f"SITES:            {n_st}",
    ]
    lines += [f"site-{j + 1}" for j in range(n_st)]
    lines.append("OFFSETS (M):")
    x_m = x_km * 1e3
    lines.append("  ".join(f"{v:.1f}" for v in x_m))
    lines.append(f"FREQUENCIES:      {len(freqs_hz)}")
    lines.append("  ".join(f"{v:.6g}" for v in freqs_hz))

    type_map = {"te": (1, 2), "tm": (5, 6)}
    blocks = []
    for j in range(n_st):
        sm = station_modes[j]
        for fi in range(len(freqs_hz)):
            if mask is not None and not mask[j][fi]:
                continue
            for mode in modes:
                t_rho, t_ph = type_map[mode]
                lr = sm[f"lr_{mode}"][fi]
                ph = sm[f"ph_{mode}"][fi]
                if not (np.isfinite(lr) and np.isfinite(ph)):
                    continue
                blocks.append(f"{j + 1:6d}{fi + 1:6d}{t_rho:6d}  {lr:.5f}  {RHO_ERR:.5f}")
                blocks.append(f"{j + 1:6d}{fi + 1:6d}{t_ph:6d}  {ph:.4f}  {PH_ERR:.4f}")
    lines.append(f"DATA BLOCKS:      {len(blocks)}")
    lines.append("SITE   FREQ   DATA TYPE      DATUM              ERROR")
    lines += blocks
    path.write_text("\n".join(lines) + "\n")
    return len(blocks)


def write_startup(path: Path, n_params: int, max_iter: int) -> None:
    lines = [
        "FORMAT:           OCCAMITER_1.0",
        "DESCRIPTION:      PIMSR production comparison",
        "MODEL FILE:       MODEL",
        "DATA FILE:        DATA",
        "DATE/TIME:        auto",
        f"MAX ITER:         {max_iter}",
        "REQ TOL:          1.0",
        "IRUF:             1",
        "DEBUG LEVEL:      1",
        "ITERATION:        0",
        "PMU:              5.0",
        "RLAST:            1.0E+07",
        "TLAST:            100.",
        "IFFTOL:           0",
        f"NO. PARMS:        {n_params}",
    ]
    vals = ["2.0000000"] * n_params
    for i in range(0, n_params, 5):
        lines.append(" ".join(vals[i : i + 5]))
    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# result parsing
# --------------------------------------------------------------------------
def parse_final_iter(workdir: Path):
    iters = sorted(workdir.glob("ITER*.iter"))
    if not iters:
        return None, None
    final = iters[-1]
    txt = final.read_text().splitlines()
    misfit = None
    vals = []
    in_params = False
    for ln in txt:
        s = ln.strip()
        if s.startswith("Misfit Value:"):
            misfit = float(s.split(":")[1])
        elif in_params:
            vals += [float(v) for v in s.split()]
        elif s.startswith("Param Count:"):
            in_params = True
    return np.array(vals), misfit


def params_to_section(
    params, layers, col_spec, mesh, x_grid_km, depth_grid_m
):
    """Map Occam parameter blocks onto the (depth, x) raster of the
    neural models so section_nrms_2d scores identically-shaped input."""
    # parameter column x-centres
    col_edges = np.concatenate([[0.0], np.cumsum(mesh["col_widths"])])
    col_edges += mesh["core_left"] - sum(mesh["col_widths"][: mesh["n_pad"]])
    pcol_edges = [col_edges[0]]
    ci = 0
    for c in col_spec:
        ci += c
        pcol_edges.append(col_edges[ci])
    pcol_cent = 0.5 * (np.array(pcol_edges[:-1]) + np.array(pcol_edges[1:]))

    row_edges = np.concatenate([[0.0], np.cumsum(mesh["row_heights"])])
    play_edges = [0.0]
    ri = 0
    for take in layers:
        ri += take
        play_edges.append(row_edges[ri])
    play_cent = 0.5 * (np.array(play_edges[:-1]) + np.array(play_edges[1:]))

    grid = np.array(params).reshape(len(layers), len(col_spec))
    # bilinear interp onto model raster
    sec = np.empty((len(depth_grid_m), len(x_grid_km)))
    xq = x_grid_km * 1e3
    for zi, z in enumerate(depth_grid_m):
        li = np.interp(z, play_cent, np.arange(len(play_cent)))
        l0, l1 = int(np.floor(li)), min(int(np.floor(li)) + 1, len(layers) - 1)
        w = li - l0
        row = (1 - w) * grid[l0] + w * grid[l1]
        sec[zi] = np.interp(xq, pcol_cent, row)
    return sec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emtf-dir", required=True)
    ap.add_argument("--binary", required=True)
    ap.add_argument("--test-h5", required=True, help="grids/frequencies source")
    ap.add_argument("--out", required=True)
    ap.add_argument("--profile", default="H-YS")
    ap.add_argument("--modes", default="te,tm")
    ap.add_argument("--max-iter", type=int, default=25)
    ap.add_argument("--workdir", default="/tmp/occam2dmt_run")
    args = ap.parse_args()

    with h5py.File(args.test_h5, "r") as f:
        freqs = f["frequencies"][:]
        station_x = f["station_x"][:]
        x_grid = f["x_grid"][:]
        depth_grid = f["depth_grid"][:]

    periods = 1.0 / freqs
    m = assemble_profile_modes(
        args.emtf_dir, freqs, station_x, profile_ids=PROFILES[args.profile]
    )
    x_km = m["x_km"]

    # per-REAL-station curves (not the interpolated model stations)
    from pimsr_benchmarks.emtf import parse_emtf_xml, resample_station_modes
    import glob as _glob

    stations = {}
    for fpath in _glob.glob(f"{args.emtf_dir}/*.xml"):
        st = parse_emtf_xml(fpath)
        stations[st.station_id] = st
    profile = [stations[i] for i in PROFILES[args.profile]]
    station_modes, mask = [], []
    for st in profile:
        sm = resample_station_modes(st, periods)
        station_modes.append(sm)
        mask.append(sm["mask"])

    workdir = Path(args.workdir)
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)

    depth_max = float(depth_grid[-1])
    mesh = build_mesh(x_km, depth_max)
    layers, col_spec = build_layers(mesh)
    n_params = len(layers) * len(col_spec)

    write_mesh(workdir / "MESH", mesh)
    write_model(workdir / "MODEL", "MESH", mesh, layers, col_spec)
    n_blocks = write_data(
        workdir / "DATA", x_km, periods, station_modes,
        modes=tuple(args.modes.split(",")), mask=mask,
    )
    write_startup(workdir / "startup", n_params, args.max_iter)
    print(f"mesh: {len(mesh['col_widths'])}x{len(mesh['row_heights'])} elements, "
          f"{n_params} params, {n_blocks} data", flush=True)

    t0 = time.time()
    res = subprocess.run(
        [str(Path(args.binary).resolve()), "startup"],
        cwd=workdir, capture_output=True, text=True, timeout=7200,
    )
    dt = time.time() - t0
    tail = "\n".join(res.stdout.splitlines()[-12:])
    print(tail, flush=True)

    params, misfit = parse_final_iter(workdir)
    out: dict = {
        "profile": args.profile, "modes": args.modes,
        "n_params": n_params, "n_data_blocks": n_blocks,
        "occam_rms": misfit, "runtime_s": round(dt, 1),
        "n_iterations": len(list(workdir.glob("ITER*.iter"))) - 1,
    }
    if params is not None and len(params) == n_params:
        sec = params_to_section(params, layers, col_spec, mesh, x_grid, depth_grid)
        score = section_nrms_2d(
            sec, m["lr_te"], m["ph_te"], freqs, station_x, x_grid, depth_grid
        )
        out["section_nrms_2d"] = round(float(score), 3)
        np.save(workdir / "final_section.npy", sec)
    print(json.dumps(out, indent=1), flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
