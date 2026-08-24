#!/usr/bin/env python3
"""Run the legacy neural diagnostic on synthetic and USArray inputs.

The outputs from this standalone runner are deliberately non-rankable.  Its
synthetic neural path consumes gravity in addition to MT, unlike the legacy
MT-only Occam baseline, and the real-data path has no resistivity ground truth.

Usage:
    python scripts/run_neural_bench.py --checkpoint /path/to/best.pt \
        --test-h5 /path/to/ds_test.h5 --emtf-dir data/emtf --out-dir /path/out
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import h5py
import numpy as np
from pimsr_forward.mt1d import mt1d_response
from pimsr_inversion.data import grid_cell_thicknesses

from pimsr_benchmarks.emtf import parse_emtf_xml, resample_station_determinant
from pimsr_benchmarks.metrics import coverage, profile_rmse, summarize
from pimsr_benchmarks.neural import NeuralInverter
from pimsr_benchmarks.runner2d import (
    file_artifact_provenance,
    publish_json_no_overwrite,
    require_file_artifact_unchanged,
)

SCHEMA = "pimsr-legacy-1d-neural-diagnostic"
SCHEMA_VERSION = 1


def _xml_paths(emtf_dir: str | Path) -> tuple[Path, list[Path]]:
    root = Path(emtf_dir).resolve(strict=True)
    paths = sorted(path.resolve(strict=True) for path in root.glob("*.xml"))
    if not paths:
        raise ValueError(f"no EMTF XML inputs found in {root}")
    return root, paths


def _snapshot_xml_inputs(
    emtf_dir: str | Path,
) -> tuple[Path, list[dict[str, object]]]:
    root, paths = _xml_paths(emtf_dir)
    return root, [file_artifact_provenance(path) for path in paths]


def _require_xml_inputs_unchanged(
    root: Path,
    provenance: Sequence[Mapping[str, object]],
) -> None:
    _, current_paths = _xml_paths(root)
    expected_paths = [Path(str(item["path"])) for item in provenance]
    if current_paths != expected_paths:
        raise RuntimeError(f"EMTF XML input set changed after snapshot: {root}")
    for item in provenance:
        require_file_artifact_unchanged(item, role="EMTF XML input")


def _require_output_paths_absent(paths: Sequence[Path]) -> None:
    existing = [path for path in paths if path.exists() or path.is_symlink()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing diagnostic output(s): "
            + ", ".join(str(path) for path in existing)
        )


def _diagnostic_result(
    result: Mapping[str, object],
    *,
    provenance: Mapping[str, object],
    reasons: Sequence[str],
    inverse_observation_budget: Mapping[str, object],
) -> dict[str, object]:
    return {
        **result,
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "comparison_status": "diagnostic_non_comparable",
        "ranking_allowed": False,
        "headline_claim_allowed": False,
        "diagnostic_reasons": list(reasons),
        "inverse_observation_budget": dict(inverse_observation_budget),
        "provenance": dict(provenance),
    }


def mt_nrms(
    log10_rho: np.ndarray,
    depth_grid: np.ndarray,
    periods: np.ndarray,
    obs_log_rho_a: np.ndarray,
    obs_phase: np.ndarray,
    mask: np.ndarray | None = None,
    err_log_rho: float = 0.05,
    err_phase_deg: float = 2.5,
) -> float:
    """nRMS of the predicted profile's MT response against observations."""
    rho = np.power(10.0, log10_rho)
    thick = grid_cell_thicknesses(depth_grid)
    rho_a, phase = mt1d_response(rho, thick, periods)
    r_lr = (np.log10(rho_a) - obs_log_rho_a) / err_log_rho
    r_ph = ((phase - obs_phase + 90.0) % 180.0 - 90.0) / err_phase_deg
    res = np.concatenate([r_lr, r_ph])
    if mask is not None:
        res = np.concatenate([r_lr[mask], r_ph[mask]])
    return float(np.sqrt(np.mean(res**2)))


def bench_synthetic(inv: NeuralInverter, test_h5: str, n: int) -> dict:
    with h5py.File(test_h5) as f:
        inv.require_dataset(f)
        lr = f["obs_mt_log10_rho"][:n].astype(np.float64)
        ph = f["obs_mt_phase"][:n].astype(np.float64)
        gz = f["obs_gravity"][:n].astype(np.float64)
        tgt_res = f["target_log10_res"][:n]
        scen = f["scenario"][:n]

    rmses, sigmas, preds, times, scen_hits = [], [], [], [], []
    for i in range(n):
        p = inv.invert(lr[i], ph[i], gz[i])
        rmses.append(profile_rmse(p.log10_rho, tgt_res[i]))
        sigmas.append(p.sigma_log10_rho)
        preds.append(p.log10_rho)
        times.append(p.wall_time_s)
        scen_hits.append(int(np.argmax(p.scenario_probs)) == int(scen[i]))

    cov = coverage(np.asarray(preds), np.asarray(sigmas), tgt_res)
    return {
        "method": "pimsr-neural",
        "n": int(n),
        "rmse_log10_res": summarize(rmses),
        "sigma_coverage_1": cov,
        "scenario_accuracy": float(np.mean(scen_hits)),
        "time_per_station_s": float(np.mean(times)),
        "per_scenario_rmse": {
            str(s): float(np.mean([rm for rm, sc in zip(rmses, scen) if sc == s]))
            for s in sorted(set(scen.tolist()))
        },
    }


def bench_real(
    inv: NeuralInverter,
    emtf_dir: str,
    *,
    xml_paths: Sequence[str | Path] | None = None,
) -> dict:
    stations = []
    paths = (
        [Path(path) for path in xml_paths]
        if xml_paths is not None
        else _xml_paths(emtf_dir)[1]
    )
    for path in paths:
        st = parse_emtf_xml(path)
        lr, ph, mask = resample_station_determinant(st, inv.periods)
        p = inv.invert(lr, ph, gravity=None)
        nrms = mt_nrms(p.log10_rho, inv.depth_grid, inv.periods, lr, ph, mask=mask)
        stations.append(
            {
                "station": st.station_id,
                "lat": st.latitude,
                "lon": st.longitude,
                "n_periods_in_band": int(mask.sum()),
                "nrms": nrms,
                "scenario": int(np.argmax(p.scenario_probs)),
                "mean_sigma": float(p.sigma_log10_rho.mean()),
                "time_s": p.wall_time_s,
            }
        )
        print(
            f"{st.station_id}: nRMS={nrms:.2f} "
            f"scen={stations[-1]['scenario']} sigma={stations[-1]['mean_sigma']:.3f}"
        )

    nrms = [s["nrms"] for s in stations]
    return {
        "method": "pimsr-neural",
        "dataset": "USArray EMTF Yellowstone box",
        "n_stations": len(stations),
        "nrms": summarize(nrms),
        "stations": stations,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test-h5", required=True)
    ap.add_argument("--emtf-dir", default="data/emtf")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    output_paths = [out / "neural_synthetic.json", out / "neural_real.json"]
    _require_output_paths_absent(output_paths)

    checkpoint = file_artifact_provenance(args.checkpoint)
    dataset = file_artifact_provenance(args.test_h5)
    emtf_root, emtf_xml = _snapshot_xml_inputs(args.emtf_dir)

    inv = NeuralInverter(str(checkpoint["path"]))
    require_file_artifact_unchanged(checkpoint, role="neural checkpoint")

    syn_metrics = bench_synthetic(inv, str(dataset["path"]), args.n)
    require_file_artifact_unchanged(dataset, role="synthetic dataset")
    require_file_artifact_unchanged(checkpoint, role="neural checkpoint")

    real_metrics = bench_real(
        inv,
        str(emtf_root),
        xml_paths=[str(item["path"]) for item in emtf_xml],
    )
    _require_xml_inputs_unchanged(emtf_root, emtf_xml)
    require_file_artifact_unchanged(dataset, role="synthetic dataset")
    require_file_artifact_unchanged(checkpoint, role="neural checkpoint")

    syn = _diagnostic_result(
        syn_metrics,
        provenance={"checkpoint": checkpoint, "dataset": dataset},
        reasons=(
            "legacy standalone run has no validated versioned execution manifest",
            (
                "synthetic neural inversion consumes MT and gravity while the legacy "
                "Occam reference consumes MT only"
            ),
            "unequal inverse observation budgets prohibit a method ranking",
        ),
        inverse_observation_budget={
            "candidate": ["mt_apparent_resistivity", "mt_phase", "gravity"],
            "legacy_occam_reference": ["mt_apparent_resistivity", "mt_phase"],
            "equal": False,
        },
    )
    real = _diagnostic_result(
        real_metrics,
        provenance={"checkpoint": checkpoint, "emtf_xml": emtf_xml},
        reasons=(
            "legacy standalone run has no validated versioned execution manifest",
            "field stations have no resistivity ground truth for an inversion score",
            (
                "reported nRMS is forward consistency of the predicted model, not "
                "reconstruction accuracy"
            ),
        ),
        inverse_observation_budget={
            "candidate": ["mt_apparent_resistivity", "mt_phase"],
            "gravity_handling": "unobserved_mean_fill",
            "equal_to_synthetic_path": False,
        },
    )

    _require_output_paths_absent(output_paths)
    publish_json_no_overwrite(syn, output_paths[0])
    publish_json_no_overwrite(real, output_paths[1])
    print(
        json.dumps({k: v for k, v in syn.items() if k != "per_scenario_rmse"}, indent=2)
    )
    print("real mean nRMS:", real["nrms"]["mean"])


if __name__ == "__main__":
    main()
