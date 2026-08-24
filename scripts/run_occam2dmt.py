#!/usr/bin/env python
"""Fail-closed diagnostic run of Scripps Occam2DMT v3.0.

Generates DATA/MESH/MODEL/startup files for the classical Occam 2D MT
inversion code (deGroot-Hedlin & Constable 1990; v3.0, Fortran 90), runs
the binary, parses the final iteration model, and scores it with the same
section_nrms_2d metric used in the unified leaderboard.

The DATA errors use the same nominal log-rho and phase scales as the shared
score.  Occam's raw-block objective is nevertheless not equivalent to the
shared score, which profiles static shifts and balances stations, modes and
components.  Results remain diagnostic and are never headline/ranking rows.

Usage:
  python scripts/run_occam2dmt.py --emtf-dir data/emtf \
      --binary /tmp/occam2d/OCCAM2DMT_V3.0/Source/occam2d \
      --test-h5 /vercel/share/pimsr-data/v3/ds2d_test.h5 \
      --out results/occam2dmt/occam2dmt.json \
      [--profile H-YS] [--modes te,tm] [--max-iter 25]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np

from pimsr_benchmarks.hybrid2d import (
    LOG10_RHO_ERROR,
    PHASE_ERROR_DEG,
    PROFILES,
    SECTION_NRMS_METRIC_ID,
    assemble_profile_modes,
    profile_geometry_metadata,
    scoring_observation_error_contract,
    section_nrms_2d,
)
from pimsr_benchmarks.runner2d import (
    load_dataset2d,
    prepare_empty_workdir,
    publish_json_no_overwrite,
    publish_npz_no_overwrite,
    run_checked,
)

RHO_ERR = LOG10_RHO_ERROR
PH_ERR = PHASE_ERROR_DEG


class _ArtifactSnapshot:
    __slots__ = ("path", "sha256", "size_bytes", "stat_signature")

    def __init__(
        self,
        path: Path,
        sha256: str,
        size_bytes: int,
        stat_signature: tuple[int, int, int, int],
    ) -> None:
        self.path = path
        self.sha256 = sha256
        self.size_bytes = size_bytes
        self.stat_signature = stat_signature

    def record(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


def _stat_signature(path: Path) -> tuple[int, int, int, int]:
    info = path.stat()
    return (int(info.st_dev), int(info.st_ino), int(info.st_size), int(info.st_mtime_ns))


def _snapshot_file(path: str | Path, *, kind: str) -> _ArtifactSnapshot:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{kind} does not exist: {resolved}")
    before = _stat_signature(resolved)
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    after = _stat_signature(resolved)
    if before != after:
        raise RuntimeError(f"{kind} changed while it was being hashed: {resolved}")
    return _ArtifactSnapshot(resolved, digest.hexdigest(), before[2], before)


def _require_unchanged(snapshot: _ArtifactSnapshot, *, kind: str) -> None:
    current = _snapshot_file(snapshot.path, kind=kind)
    if (
        current.stat_signature != snapshot.stat_signature
        or current.sha256 != snapshot.sha256
        or current.size_bytes != snapshot.size_bytes
    ):
        raise RuntimeError(f"{kind} changed after it was read: {snapshot.path}")


def _snapshot_xml_inputs(emtf_dir: str | Path) -> tuple[_ArtifactSnapshot, ...]:
    directory = Path(emtf_dir).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"EMTF directory does not exist: {directory}")
    paths = sorted((path.resolve() for path in directory.glob("*.xml")), key=str)
    if not paths:
        raise FileNotFoundError(f"EMTF directory contains no XML files: {directory}")
    if len(paths) != len(set(paths)):
        raise ValueError("EMTF XML paths resolve to duplicate files")
    return tuple(_snapshot_file(path, kind="EMTF XML") for path in paths)


def _require_xml_inputs_unchanged(
    emtf_dir: str | Path,
    snapshots: tuple[_ArtifactSnapshot, ...],
) -> None:
    current_paths = tuple(
        sorted((path.resolve() for path in Path(emtf_dir).resolve().glob("*.xml")), key=str)
    )
    expected_paths = tuple(snapshot.path for snapshot in snapshots)
    if current_paths != expected_paths:
        raise RuntimeError("EMTF XML input set changed after it was enumerated")
    for snapshot in snapshots:
        _require_unchanged(snapshot, kind="EMTF XML")


def _section_output_path(json_path: str | Path) -> Path:
    output = Path(json_path).resolve()
    return output.with_name(f"{output.stem}.section.npz")


def _require_fresh_outputs(json_path: str | Path, section_path: Path) -> None:
    for path in (Path(json_path).resolve(), section_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing benchmark output: {path}")


def _stopping_evidence(output: str) -> tuple[str | None, str | None]:
    """Return a conservative solver-reported stopping reason and exact line."""
    negative = re.compile(
        r"\b(?:not|failed\s+to)\s+converg|target\s+not\s+(?:met|reached)|"
        r"(?:maximum|max)\s+(?:number\s+of\s+)?iterations|iteration\s+limit",
        re.IGNORECASE,
    )
    positive = re.compile(
        r"\b(?:converged|convergence\s+(?:achieved|reached|satisfied)|"
        r"target\s+(?:met|reached)|tolerance\s+(?:met|reached|satisfied))\b",
        re.IGNORECASE,
    )
    for line in reversed(output.splitlines()):
        text = line.strip()
        if text and negative.search(text):
            return "solver_reported_iteration_limit_or_non_convergence", text
        if text and positive.search(text):
            return "solver_reported_convergence", text
    return None, None


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
    step = 1.0
    while i < n_row:
        take = min(round(step), n_row - i)
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
        f"BINDING OFFSET:   {mesh['core_left'] - sum(mesh['col_widths'][: mesh['n_pad']]):.1f}",
        f"NUM LAYERS:       {len(layers)}",
    ]
    for take in layers:
        lines.append(f"{take}  {len(col_spec)}")
        lines.append(" " + " ".join(str(c) for c in col_spec))
    lines.append("NO. EXCEPTIONS:   0")
    path.write_text("\n".join(lines) + "\n")


def write_data(path: Path, x_km, periods, station_modes, modes=("te", "tm")) -> int:
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
            for mode in modes:
                if not sm[f"mask_{mode}"][fi]:
                    continue
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


def write_startup(
    path: Path,
    n_params: int,
    max_iter: int,
    target_rms: float = 1.0,
) -> None:
    lines = [
        "FORMAT:           OCCAMITER_1.0",
        "DESCRIPTION:      PIMSR diagnostic evaluation",
        "MODEL FILE:       MODEL",
        "DATA FILE:        DATA",
        "DATE/TIME:        auto",
        f"MAX ITER:         {max_iter}",
        f"REQ TOL:          {target_rms:.12g}",
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
    iters = _iteration_paths(workdir)
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


def _iteration_number(path: Path) -> int:
    match = re.search(r"(\d+)(?=\.iter$)", path.name, flags=re.IGNORECASE)
    if match is None:
        raise RuntimeError(f"unexpected Occam iteration filename: {path.name}")
    return int(match.group(1))


def _iteration_paths(workdir: Path) -> tuple[Path, ...]:
    return tuple(sorted(workdir.glob("ITER*.iter"), key=_iteration_number))


def _iteration_trace(
    workdir: Path,
) -> tuple[list[dict[str, object]], tuple[_ArtifactSnapshot, ...]]:
    paths_before = _iteration_paths(workdir)
    if not paths_before:
        raise RuntimeError("Occam2DMT produced no iteration files")
    snapshots = tuple(
        _snapshot_file(path, kind="Occam iteration output") for path in paths_before
    )
    trace: list[dict[str, object]] = []
    for path, snapshot in zip(paths_before, snapshots, strict=True):
        text = path.read_text(encoding="utf-8")
        match = re.search(
            r"^\s*Misfit\s+Value\s*:\s*"
            r"([0-9]+(?:\.[0-9]*)?(?:[Ee][+-]?\d+)?)\s*$",
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        misfit = float(match.group(1)) if match else None
        if misfit is not None and (not np.isfinite(misfit) or misfit < 0.0):
            raise ValueError("Occam iteration trace contains an invalid RMS")
        _require_unchanged(snapshot, kind="Occam iteration output")
        trace.append(
            {
                "iteration": _iteration_number(path),
                "rms": misfit,
                "artifact": snapshot.record(),
            }
        )
    if _iteration_paths(workdir) != paths_before:
        raise RuntimeError("Occam iteration output set changed while it was parsed")
    return trace, snapshots


def _require_iteration_outputs_unchanged(
    workdir: Path,
    snapshots: tuple[_ArtifactSnapshot, ...],
) -> None:
    current_paths = _iteration_paths(workdir)
    expected_paths = tuple(snapshot.path for snapshot in snapshots)
    if current_paths != expected_paths:
        raise RuntimeError("Occam iteration output set changed after solver completion")
    for snapshot in snapshots:
        _require_unchanged(snapshot, kind="Occam iteration output")


def _execution_contract(
    *,
    stdout: str,
    stderr: str,
    trace: list[dict[str, object]],
    final_rms: float | None,
    target_rms: float,
    max_iterations: int,
) -> dict[str, object]:
    reason, evidence = _stopping_evidence(f"{stdout}\n{stderr}")
    target_reached = (
        final_rms is not None and np.isfinite(final_rms) and final_rms <= target_rms
    )
    trace_final_rms = trace[-1]["rms"]
    trace_matches_final = bool(
        final_rms is not None
        and trace_final_rms is not None
        and final_rms == trace_final_rms
    )
    convergence_proven = bool(
        target_reached
        and trace_matches_final
        and reason == "solver_reported_convergence"
    )
    final_iteration = int(trace[-1]["iteration"])
    if reason is None:
        if target_reached:
            reason = "target_reached_without_explicit_solver_convergence"
        elif final_iteration >= max_iterations:
            reason = "configured_iteration_limit_reached"
        else:
            reason = "no_explicit_solver_convergence_evidence"
    stdout_bytes = stdout.encode("utf-8")
    stderr_bytes = stderr.encode("utf-8")
    return {
        "target_rms": target_rms,
        "target_reached": bool(target_reached),
        "convergence_proven": convergence_proven,
        "final_rms_matches_iteration_trace": trace_matches_final,
        "stopping_reason": reason,
        "stopping_evidence_line": evidence,
        "configured_max_iterations": max_iterations,
        "final_iteration": final_iteration,
        "iteration_trace": trace,
        "stdout_sha256": hashlib.sha256(stdout_bytes).hexdigest(),
        "stdout_size_bytes": len(stdout_bytes),
        "stderr_sha256": hashlib.sha256(stderr_bytes).hexdigest(),
        "stderr_size_bytes": len(stderr_bytes),
    }


def params_to_section(
    params,
    layers,
    col_spec,
    mesh,
    x_grid_m,
    profile_x_km,
    depth_grid_m,
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
    sec = np.empty((len(depth_grid_m), len(x_grid_m)))
    x_span = float(np.ptp(x_grid_m))
    if x_span <= 0.0:
        raise ValueError("x_grid_m must span a non-zero distance")
    x_fraction = (np.asarray(x_grid_m) - np.min(x_grid_m)) / x_span
    profile_x_m = np.asarray(profile_x_km) * 1e3
    xq = profile_x_m.min() + x_fraction * np.ptp(profile_x_m)
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
    ap.add_argument("--target-rms", type=float, default=1.0)
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--workdir", default="/tmp/occam2dmt_run")
    args = ap.parse_args()
    if args.profile not in PROFILES:
        raise ValueError(f"unknown profile: {args.profile!r}")
    requested_modes = tuple(part.strip() for part in args.modes.split(","))
    if not requested_modes or len(requested_modes) != len(set(requested_modes)):
        raise ValueError("--modes must contain unique comma-separated modes")
    if any(mode not in {"te", "tm"} for mode in requested_modes):
        raise ValueError("--modes may contain only 'te' and 'tm'")
    if args.max_iter <= 0:
        raise ValueError("--max-iter must be positive")
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    if not np.isfinite(args.target_rms) or args.target_rms <= 0.0:
        raise ValueError("--target-rms must be finite and positive")

    output_path = Path(args.out).resolve()
    section_path = _section_output_path(output_path)
    _require_fresh_outputs(output_path, section_path)

    dataset_snapshot = _snapshot_file(args.test_h5, kind="2D HDF5 dataset")
    binary_snapshot = _snapshot_file(args.binary, kind="Occam2DMT binary")
    runner_snapshot = _snapshot_file(__file__, kind="Occam2DMT runner source")
    xml_snapshots = _snapshot_xml_inputs(args.emtf_dir)
    binary = binary_snapshot.path

    contract = load_dataset2d(dataset_snapshot.path)
    _require_unchanged(dataset_snapshot, kind="2D HDF5 dataset")
    freqs = contract.frequencies
    station_x = contract.station_x
    x_grid = contract.x_grid
    depth_grid = contract.depth_grid

    periods = 1.0 / freqs
    m = assemble_profile_modes(
        args.emtf_dir, freqs, station_x, profile_ids=PROFILES[args.profile]
    )
    _require_xml_inputs_unchanged(args.emtf_dir, xml_snapshots)
    x_km = m["x_km"]

    # per-REAL-station curves (not the interpolated model stations)
    import glob as _glob

    from pimsr_benchmarks.emtf import parse_emtf_xml, resample_station_modes

    stations = {}
    for fpath in _glob.glob(f"{args.emtf_dir}/*.xml"):
        st = parse_emtf_xml(fpath)
        stations[st.station_id] = st
    profile = [stations[i] for i in PROFILES[args.profile]]
    station_modes = []
    for st in profile:
        sm = resample_station_modes(
            st,
            periods,
            profile_azimuth_deg=float(m["profile_azimuth_deg"]),
        )
        station_modes.append(sm)
    _require_xml_inputs_unchanged(args.emtf_dir, xml_snapshots)

    workdir = prepare_empty_workdir(args.workdir)

    depth_max = float(depth_grid[-1])
    mesh = build_mesh(x_km, depth_max)
    layers, col_spec = build_layers(mesh)
    n_params = len(layers) * len(col_spec)

    write_mesh(workdir / "MESH", mesh)
    write_model(workdir / "MODEL", "MESH", mesh, layers, col_spec)
    n_blocks = write_data(
        workdir / "DATA",
        x_km,
        periods,
        station_modes,
        modes=requested_modes,
    )
    write_startup(
        workdir / "startup",
        n_params,
        args.max_iter,
        target_rms=float(args.target_rms),
    )
    generated_inputs = tuple(
        _snapshot_file(workdir / name, kind=f"Occam generated {name}")
        for name in ("MESH", "MODEL", "DATA", "startup")
    )
    print(
        f"mesh: {len(mesh['col_widths'])}x{len(mesh['row_heights'])} elements, "
        f"{n_params} params, {n_blocks} data",
        flush=True,
    )

    t0 = time.time()
    command = [str(binary), "startup"]
    res = run_checked(
        command,
        cwd=workdir,
        timeout=args.timeout,
    )
    dt = time.time() - t0
    tail = "\n".join(res.stdout.splitlines()[-12:])
    print(tail, flush=True)

    trace, iteration_snapshots = _iteration_trace(workdir)
    params, misfit = parse_final_iter(workdir)
    if params is None or len(params) != n_params:
        raise RuntimeError(
            f"Occam2DMT produced no complete model: expected {n_params} parameters"
        )
    if not np.isfinite(params).all():
        raise RuntimeError("Occam2DMT final model contains non-finite parameters")
    if misfit is not None and (not np.isfinite(misfit) or misfit < 0.0):
        raise RuntimeError("Occam2DMT final RMS is not finite and non-negative")
    _require_iteration_outputs_unchanged(workdir, iteration_snapshots)
    execution = _execution_contract(
        stdout=res.stdout,
        stderr=res.stderr,
        trace=trace,
        final_rms=misfit,
        target_rms=float(args.target_rms),
        max_iterations=args.max_iter,
    )
    for snapshot, kind in (
        (dataset_snapshot, "2D HDF5 dataset"),
        (binary_snapshot, "Occam2DMT binary"),
        (runner_snapshot, "Occam2DMT runner source"),
    ):
        _require_unchanged(snapshot, kind=kind)
    _require_xml_inputs_unchanged(args.emtf_dir, xml_snapshots)
    for snapshot in generated_inputs:
        _require_unchanged(snapshot, kind="Occam generated solver input")
    for snapshot in iteration_snapshots:
        _require_unchanged(snapshot, kind="Occam iteration output")

    out: dict = {
        "schema_version": 4,
        "metric_id": SECTION_NRMS_METRIC_ID,
        "geometry": profile_geometry_metadata(m),
        "scoring_error_contract": scoring_observation_error_contract(),
        "profile": args.profile,
        "modes": ",".join(requested_modes),
        "n_params": n_params,
        "n_data_blocks": n_blocks,
        "occam_rms": misfit,
        "runtime_s": round(dt, 1),
        "n_iterations": int(trace[-1]["iteration"]),
    }
    sec = params_to_section(params, layers, col_spec, mesh, x_grid, x_km, depth_grid)
    score = section_nrms_2d(sec, m, freqs, station_x, x_grid, depth_grid)
    if not np.isfinite(sec).all() or not np.isfinite(score):
        raise RuntimeError("Occam2DMT produced non-finite section output or score")
    out["section_nrms_2d"] = round(float(score), 3)
    publish_npz_no_overwrite(
        section_path,
        section_log10_resistivity=np.asarray(sec, dtype=np.float64),
        x_grid_m=np.asarray(x_grid, dtype=np.float64),
        depth_grid_m=np.asarray(depth_grid, dtype=np.float64),
    )
    section_snapshot = _snapshot_file(section_path, kind="published Occam section")
    out.update(
        {
            "comparison_status": (
                "diagnostic_non_comparable"
                if execution["convergence_proven"]
                else "diagnostic_incomplete"
            ),
            "ranking_allowed": False,
            "headline_claim_allowed": False,
            "comparison_contract": {
                "comparable_to_shared_leaderboard": False,
                "reason": (
                    "the normalized-geometry diagnostic score and Occam raw-block "
                    "objective are not the native-geometry shared ranking contract"
                ),
                "inverse_observation_geometry": "native_emtf_stations",
                "native_emtf_station_count": len(PROFILES[args.profile]),
                "solver_station_count": len(PROFILES[args.profile]),
                "solver_frequency_count": len(freqs),
                "solver_data_block_count": int(n_blocks),
                "internal_objective_equivalent_to_shared_score": False,
            },
            "execution": execution,
            "provenance": {
                "dataset": dataset_snapshot.record(),
                "emtf_xml": [snapshot.record() for snapshot in xml_snapshots],
                "executable": binary_snapshot.record(),
                "runner_source": runner_snapshot.record(),
                "generated_solver_inputs": [
                    snapshot.record() for snapshot in generated_inputs
                ],
                "command": command,
                "cwd": str(workdir),
                "configuration": {
                    "profile_ids": list(PROFILES[args.profile]),
                    "profile": args.profile,
                    "modes": list(requested_modes),
                    "max_iterations": args.max_iter,
                    "target_rms": float(args.target_rms),
                    "timeout_s": args.timeout,
                    "rho_error_log10": RHO_ERR,
                    "phase_error_degrees": PH_ERR,
                },
            },
            "section_artifact": section_snapshot.record(),
        }
    )
    for snapshot, kind in (
        (dataset_snapshot, "2D HDF5 dataset"),
        (binary_snapshot, "Occam2DMT binary"),
        (runner_snapshot, "Occam2DMT runner source"),
        (section_snapshot, "published Occam section"),
    ):
        _require_unchanged(snapshot, kind=kind)
    _require_xml_inputs_unchanged(args.emtf_dir, xml_snapshots)
    for snapshot in generated_inputs:
        _require_unchanged(snapshot, kind="Occam generated solver input")
    _require_iteration_outputs_unchanged(workdir, iteration_snapshots)
    print(json.dumps(out, indent=1), flush=True)

    publish_json_no_overwrite(out, output_path)


if __name__ == "__main__":
    main()
