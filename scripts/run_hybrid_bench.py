"""Run the legacy hybrid (neural warm start + Occam refinement) diagnostic.

The standalone outputs are deliberately non-rankable.  In particular, the
synthetic neural warm start sees MT plus gravity while the Occam refinement
and legacy Occam baseline see MT only.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import h5py
import numpy as np

from pimsr_benchmarks.emtf import parse_emtf_xml, resample_station_determinant
from pimsr_benchmarks.hybrid import hybrid_invert
from pimsr_benchmarks.metrics import profile_rmse, summarize
from pimsr_benchmarks.neural import NeuralInverter
from pimsr_benchmarks.runner2d import (
    file_artifact_provenance,
    publish_json_no_overwrite,
    require_file_artifact_unchanged,
)

SCHEMA = "pimsr-legacy-1d-hybrid-diagnostic"
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


def bench_synthetic(inverter: NeuralInverter, test_h5: str, n: int) -> dict:
    with h5py.File(test_h5) as f:
        inverter.require_dataset(f)
        periods = f["periods"][:]
        depth = f["depth_grid"][:]
        obs_lr = f["obs_mt_log10_rho"][:n].astype(np.float64)
        obs_ph = f["obs_mt_phase"][:n].astype(np.float64)
        obs_gz = f["obs_gravity"][:n].astype(np.float64)
        tgt = f["target_log10_res"][:n]
        scen = f["scenario"][:n]

    t0 = time.time()
    rmses, nrms, iters, conv = [], [], [], 0
    for i in range(n):
        r = hybrid_invert(inverter, obs_lr[i], obs_ph[i], periods, gravity=obs_gz[i])
        rmses.append(profile_rmse(r.occam.profile_on_grid(depth), tgt[i]))
        nrms.append(r.occam.nrms)
        iters.append(r.occam.n_iterations)
        conv += r.occam.converged
    dt = time.time() - t0

    return {
        "method": "hybrid",
        "n": n,
        "rmse_log10_res": summarize(rmses),
        "nrms": summarize(nrms),
        "iterations": summarize(iters),
        "converged_frac": conv / n,
        "time_per_station_s": dt / n,
        "per_scenario_rmse": {
            str(s): float(np.mean([rm for rm, sc in zip(rmses, scen) if sc == s]))
            for s in sorted(set(scen.tolist()))
        },
    }


def bench_real(
    inverter: NeuralInverter,
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
    for f in paths:
        st = parse_emtf_xml(f)
        lr = np.log10(st.rho_a_det)
        n_lr, n_ph, _ = resample_station_determinant(st, inverter.periods)
        r = hybrid_invert(
            inverter,
            lr,
            st.phase_det,
            st.periods,
            neural_log_rho_a=n_lr,
            neural_phase=n_ph,
        )
        stations.append(
            {
                "station": st.station_id,
                "nrms": r.occam.nrms,
                "iters": r.occam.n_iterations,
                "converged": bool(r.occam.converged),
                "time_s": r.total_time_s,
            }
        )
        print(
            f"{st.station_id}: nRMS={r.occam.nrms:.2f} iters={r.occam.n_iterations} "
            f"t={r.total_time_s:.2f}s"
        )
    nrms = [x["nrms"] for x in stations]
    return {
        "method": "hybrid",
        "dataset": "USArray EMTF Yellowstone box",
        "n_stations": len(stations),
        "nrms": summarize(nrms),
        "converged_frac": float(np.mean([x["converged"] for x in stations])),
        "stations": stations,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test-h5", required=True)
    ap.add_argument("--emtf-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    output_paths = [out / "hybrid_synthetic.json", out / "hybrid_real.json"]
    _require_output_paths_absent(output_paths)

    checkpoint = file_artifact_provenance(args.checkpoint)
    dataset = file_artifact_provenance(args.test_h5)
    emtf_root, emtf_xml = _snapshot_xml_inputs(args.emtf_dir)

    inverter = NeuralInverter(str(checkpoint["path"]))
    require_file_artifact_unchanged(checkpoint, role="neural checkpoint")

    syn_metrics = bench_synthetic(inverter, str(dataset["path"]), args.n)
    require_file_artifact_unchanged(dataset, role="synthetic dataset")
    require_file_artifact_unchanged(checkpoint, role="neural checkpoint")

    real_metrics = bench_real(
        inverter,
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
                "synthetic neural warm start consumes MT and gravity while Occam "
                "refinement and the legacy Occam reference consume MT only"
            ),
            "unequal inverse observation budgets prohibit a method ranking",
        ),
        inverse_observation_budget={
            "neural_warm_start": [
                "mt_apparent_resistivity",
                "mt_phase",
                "gravity",
            ],
            "occam_refinement": ["mt_apparent_resistivity", "mt_phase"],
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
                "reported nRMS measures MT forward consistency after a distinct "
                "hybrid optimization path"
            ),
        ),
        inverse_observation_budget={
            "neural_warm_start": ["mt_apparent_resistivity", "mt_phase"],
            "occam_refinement": ["mt_apparent_resistivity", "mt_phase"],
            "gravity_handling": "unobserved_mean_fill",
        },
    )

    _require_output_paths_absent(output_paths)
    publish_json_no_overwrite(syn, output_paths[0])
    publish_json_no_overwrite(real, output_paths[1])
    print("\nSYNTHETIC:", json.dumps(syn["rmse_log10_res"]))
    print("\nREAL nRMS:", json.dumps(real["nrms"]))


if __name__ == "__main__":
    main()
