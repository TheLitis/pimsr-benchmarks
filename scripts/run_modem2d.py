#!/usr/bin/env python
"""Run a fail-closed diagnostic ModEM evaluation on a USArray profile.

This runner is deliberately not ranking-eligible: its historical input
adapter densifies the native EMTF stations onto the model station grid.  The
published result records that limitation together with immutable input,
solver-output and section identities so it cannot be mistaken for a
production/headline comparison.

Pipeline:
  1. rotate each full geographic EMTF tensor into the fitted profile frame,
     then form local TE=Zyx and TM=Zxy observations (phases modulo 180)
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

MU0 = 4.0e-7 * np.pi


def _modem_error_contract() -> dict[str, object]:
    """Map the shared score errors to ModEM's one complex-component floor.

    Mod2DMT accepts one fractional uncertainty for both Cartesian impedance
    components.  The shared score instead declares independent log10-rho and
    phase errors.  The larger exact complex-plane equivalent is therefore
    used so the inversion does not receive a tighter error than either score
    component.
    """
    amplitude_fraction = 10.0 ** (LOG10_RHO_ERROR / 2.0) - 1.0
    phase_fraction = np.tan(np.radians(PHASE_ERROR_DEG))
    return {
        "native_representation": "fraction_of_impedance_magnitude_per_component",
        "fraction": float(max(amplitude_fraction, phase_fraction)),
        "derivation": "max(10**(log10_rho_std/2)-1, tan(phase_std_radians))",
        "amplitude_equivalent_fraction": float(amplitude_fraction),
        "phase_equivalent_fraction": float(phase_fraction),
        "source_scoring_contract": scoring_observation_error_contract(),
    }


MODEM_ERROR_CONTRACT = _modem_error_contract()
ERROR_FLOOR = float(MODEM_ERROR_CONTRACT["fraction"])


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
            return "solver_reported_non_convergence", text
        if text and positive.search(text):
            return "solver_reported_convergence", text
    return None, None


def _execution_contract(
    *,
    stdout: str,
    stderr: str,
    rms_history: list[float],
    final_iteration: int,
    target_rms: float,
) -> dict[str, object]:
    if any(not np.isfinite(value) or value < 0.0 for value in rms_history):
        raise ValueError("ModEM RMS trace must contain finite non-negative values")
    reason, evidence = _stopping_evidence(f"{stdout}\n{stderr}")
    final_rms = rms_history[-1] if rms_history else None
    target_reached = final_rms is not None and final_rms <= target_rms
    convergence_proven = bool(target_reached and reason == "solver_reported_convergence")
    if reason is None:
        reason = (
            "target_reached_without_explicit_solver_convergence"
            if target_reached
            else "no_explicit_solver_convergence_evidence"
        )
    return {
        "target_rms": target_rms,
        "target_reached": target_reached,
        "convergence_proven": convergence_proven,
        "stopping_reason": reason,
        "stopping_evidence_line": evidence,
        "final_model_iteration": final_iteration,
        "rms_trace": [
            {"trace_index": index, "rms": value}
            for index, value in enumerate(rms_history, start=1)
        ],
        "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
        "stdout_size_bytes": len(stdout.encode("utf-8")),
        "stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
        "stderr_size_bytes": len(stderr.encode("utf-8")),
    }


def _require_model_outputs_unchanged(
    workdir: Path,
    snapshots: tuple[_ArtifactSnapshot, ...],
) -> None:
    current_paths = tuple(
        sorted(workdir.glob("Modular_NLCG_*.rho"), key=_modem_model_iteration)
    )
    expected_paths = tuple(snapshot.path for snapshot in snapshots)
    if current_paths != expected_paths:
        raise RuntimeError("ModEM model output set changed after solver completion")
    for snapshot in snapshots:
        _require_unchanged(snapshot, kind="ModEM iteration model")


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
            f.writelines(
                " ".join(vals[i : i + 10]) + "\n" for i in range(0, len(vals), 10)
            )
        f.write("1\n")  # skipped record before values
        row = " ".join([f"{ln_rho:.6E}"] * ny)
        f.writelines(row + "\n" for _ in range(nz))


# ----------------------------------------------------------------- data


def write_data(path: Path, modes: dict, periods: np.ndarray, st_y: np.ndarray) -> int:
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
                if not modes[f"mask_{mode.lower()}"][i, j]:
                    continue
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


def read_final_model(
    workdir: Path,
) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    """Latest Modular_NLCG_NNN.rho -> (ln_rho[nz, ny] array, iteration)."""
    files = sorted(workdir.glob("Modular_NLCG_*.rho"), key=_modem_model_iteration)
    if not files:
        raise RuntimeError("no NLCG output models found")
    final = files[-1]
    match = re.search(r"_(\d+)\.rho$", final.name)
    if match is None:
        raise RuntimeError(f"unexpected ModEM model filename: {final.name}")
    it = int(match.group(1))
    tokens = final.read_text().split()
    if len(tokens) < 4:
        raise RuntimeError(f"incomplete ModEM model file: {final}")
    ny, nz = int(tokens[0]), int(tokens[1])
    if tokens[2] != "LOGE":
        raise RuntimeError(f"unsupported ModEM model representation: {tokens[2]}")
    vals = np.array([float(v) for v in tokens[3:]])
    required = ny + nz + 1 + ny * nz
    if vals.size < required:
        raise RuntimeError(
            f"incomplete ModEM model values: expected {required}, got {vals.size}"
        )
    dy, vals = vals[:ny], vals[ny:]
    dz, vals = vals[:nz], vals[nz:]
    vals = vals[1:]  # skipped record token ("1")
    ln_rho = vals[: ny * nz].reshape(nz, ny)
    return ln_rho, it, dy, dz


def _modem_model_iteration(path: Path) -> int:
    match = re.search(r"_(\d+)\.rho$", path.name)
    if match is None:
        raise RuntimeError(f"unexpected ModEM model filename: {path.name}")
    return int(match.group(1))


def to_project_raster(
    ln_rho: np.ndarray,
    dy: np.ndarray,
    dz: np.ndarray,
    mesh: dict,
    x_km: np.ndarray,
    x_grid: np.ndarray,
    depth_grid: np.ndarray,
) -> np.ndarray:
    """Interpolate the ModEM model onto the project (depth, x) raster."""
    yc = np.cumsum(dy) - dy / 2.0
    zc = np.cumsum(dz) - dz / 2.0
    # map project x_grid (normalised to the station span) into mesh y
    xn = (x_grid - x_grid.min()) / max(x_grid.max() - x_grid.min(), 1e-9)
    st_y = mesh["st_y"]
    y_t = st_y.min() + xn * (st_y.max() - st_y.min())
    iy = np.asarray([int(np.argmin(np.abs(yc - value))) for value in y_t])
    iz = np.asarray([int(np.argmin(np.abs(zc - value))) for value in depth_grid])
    log10_rho = ln_rho / np.log(10.0)
    return log10_rho[np.ix_(iz, iy)]


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
    ap.add_argument(
        "--target-rms",
        type=float,
        default=1.0,
        help="diagnostic RMS target; does not alter the ModEM solver settings",
    )
    args = ap.parse_args()
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    if not np.isfinite(args.target_rms) or args.target_rms <= 0.0:
        raise ValueError("--target-rms must be finite and positive")

    output_path = Path(args.out).resolve()
    section_path = _section_output_path(output_path)
    _require_fresh_outputs(output_path, section_path)

    dataset_snapshot = _snapshot_file(args.test_h5, kind="2D HDF5 dataset")
    binary_snapshot = _snapshot_file(args.binary, kind="ModEM binary")
    runner_snapshot = _snapshot_file(__file__, kind="ModEM runner source")
    xml_snapshots = _snapshot_xml_inputs(args.emtf_dir)
    binary = binary_snapshot.path

    contract = load_dataset2d(dataset_snapshot.path)
    _require_unchanged(dataset_snapshot, kind="2D HDF5 dataset")
    freqs = contract.frequencies
    station_x = contract.station_x
    x_grid = contract.x_grid
    depth_grid = contract.depth_grid
    periods = 1.0 / freqs

    modes = assemble_profile_modes(
        args.emtf_dir, freqs, station_x, profile_ids=PROFILES[args.profile]
    )
    _require_xml_inputs_unchanged(args.emtf_dir, xml_snapshots)
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

    wd = prepare_empty_workdir(args.workdir)
    write_prior(wd / "prior.rho", mesh)
    n_data = write_data(wd / "data.dat", modes, periods, st_y)
    generated_inputs = (
        _snapshot_file(wd / "prior.rho", kind="ModEM prior"),
        _snapshot_file(wd / "data.dat", kind="ModEM data"),
    )
    command = [str(binary), "-I", "NLCG", "prior.rho", "data.dat"]

    t0 = time.time()
    proc = run_checked(
        command,
        cwd=wd,
        timeout=args.timeout,
    )
    runtime = time.time() - t0
    rms_hist = [
        float(match)
        for match in re.findall(
            r"\brms\s*=\s*([0-9]+(?:\.[0-9]*)?(?:[Ee][+-]?\d+)?)",
            f"{proc.stdout}\n{proc.stderr}",
            flags=re.IGNORECASE,
        )
    ]

    model_paths_before = tuple(
        sorted(wd.glob("Modular_NLCG_*.rho"), key=_modem_model_iteration)
    )
    if not model_paths_before:
        raise RuntimeError("no NLCG output models found")
    model_snapshots = tuple(
        _snapshot_file(path, kind="ModEM iteration model")
        for path in model_paths_before
    )
    final_model_snapshot = model_snapshots[-1]
    ln_rho, it, dy, dz = read_final_model(wd)
    _require_model_outputs_unchanged(wd, model_snapshots)
    sec = to_project_raster(ln_rho, dy, dz, mesh, x_km, x_grid, depth_grid)
    nrms = section_nrms_2d(sec, modes, freqs, station_x, x_grid, depth_grid)
    if not np.isfinite(sec).all() or not np.isfinite(nrms):
        raise RuntimeError("ModEM produced non-finite section output or score")

    execution = _execution_contract(
        stdout=proc.stdout,
        stderr=proc.stderr,
        rms_history=rms_hist,
        final_iteration=it,
        target_rms=float(args.target_rms),
    )

    for snapshot, kind in (
        (dataset_snapshot, "2D HDF5 dataset"),
        (binary_snapshot, "ModEM binary"),
        (runner_snapshot, "ModEM runner source"),
    ):
        _require_unchanged(snapshot, kind=kind)
    _require_xml_inputs_unchanged(args.emtf_dir, xml_snapshots)
    for snapshot in generated_inputs:
        _require_unchanged(snapshot, kind="ModEM generated solver input")

    publish_npz_no_overwrite(
        section_path,
        section_log10_resistivity=np.asarray(sec, dtype=np.float64),
        x_grid_m=np.asarray(x_grid, dtype=np.float64),
        depth_grid_m=np.asarray(depth_grid, dtype=np.float64),
    )
    section_snapshot = _snapshot_file(section_path, kind="published ModEM section")

    result = {
        "schema_version": 4,
        "metric_id": SECTION_NRMS_METRIC_ID,
        "geometry": profile_geometry_metadata(modes),
        "code": "ModEM Mod2DMT (NLCG diagnostic)",
        "profile": args.profile,
        "n_data": n_data,
        "error_floor": ERROR_FLOOR,
        "inversion_error_contract": MODEM_ERROR_CONTRACT,
        "scoring_error_contract": scoring_observation_error_contract(),
        "inverse_observation_modes": ["te", "tm"],
        "scoring_observation_modes": ["te", "tm"],
        "observation_budget_equal_to_scoring": True,
        "iterations": it,
        "modem_final_rms": rms_hist[-1] if rms_hist else None,
        "section_nrms_2d": float(nrms),
        "runtime_s": round(runtime, 1),
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
                "ModEM receives densified model-grid pseudo-stations while the "
                "native-geometry external-solver protocol requires original EMTF stations"
            ),
            "inverse_observation_geometry": "densified_model_grid_pseudo_stations",
            "native_emtf_station_count": len(PROFILES[args.profile]),
            "solver_station_count": len(station_x),
            "solver_frequency_count": len(freqs),
            "solver_datum_count": int(n_data),
            "inverse_observation_modes": ["te", "tm"],
            "scoring_observation_modes": ["te", "tm"],
            "observation_budget_equal_to_scoring": True,
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
            "final_solver_model": final_model_snapshot.record(),
            "command": command,
            "cwd": str(wd),
            "configuration": {
                "profile_ids": list(PROFILES[args.profile]),
                "profile": args.profile,
                "timeout_s": args.timeout,
                "target_rms": float(args.target_rms),
                "inversion_method": "NLCG",
                "error_floor_fraction": ERROR_FLOOR,
            },
        },
        "section_artifact": section_snapshot.record(),
    }
    result["execution"]["iteration_model_artifacts"] = [
        {
            "iteration": _modem_model_iteration(snapshot.path),
            "artifact": snapshot.record(),
        }
        for snapshot in model_snapshots
    ]
    for snapshot, kind in (
        (dataset_snapshot, "2D HDF5 dataset"),
        (binary_snapshot, "ModEM binary"),
        (runner_snapshot, "ModEM runner source"),
        (section_snapshot, "published ModEM section"),
    ):
        _require_unchanged(snapshot, kind=kind)
    _require_xml_inputs_unchanged(args.emtf_dir, xml_snapshots)
    for snapshot in generated_inputs:
        _require_unchanged(snapshot, kind="ModEM generated solver input")
    _require_model_outputs_unchanged(wd, model_snapshots)
    publish_json_no_overwrite(result, output_path)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
