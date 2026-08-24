"""Benchmark runner CLI.

Subcommands
-----------
synthetic : neural vs Occam on the held-out synthetic test split
real      : neural vs Occam on real EMTF stations (USArray)
"""

from __future__ import annotations

import argparse
import json

import h5py
import numpy as np

from .emtf import parse_emtf_xml, resample_station_determinant
from .metrics import coverage, data_nrms, profile_rmse, summarize
from .neural import NeuralInverter
from .occam1d import occam1d_invert
from .runner2d import (
    file_artifact_provenance,
    publish_json_no_overwrite,
    require_file_artifact_unchanged,
)


def run_synthetic(args: argparse.Namespace) -> dict:
    inv = NeuralInverter(args.checkpoint)
    if not getattr(args, "allow_mixed_budget_diagnostic", False):
        raise RuntimeError(
            "synthetic neural inversion consumes MT plus gravity while Occam consumes "
            "MT only; pass --allow-mixed-budget-diagnostic to emit an explicitly "
            "non-rankable diagnostic"
        )
    with h5py.File(args.dataset, "r") as f:
        inv.require_dataset(f)
        obs_rho = f["obs_mt_log10_rho"][:]
        obs_phase = f["obs_mt_phase"][:]
        obs_grav = f["obs_gravity"][:]
        tgt = f["target_log10_res"][:]
        periods = f["periods"][:]
        depth_grid = f["depth_grid"][:]

    n = min(args.n_stations, obs_rho.shape[0])
    idx = np.random.default_rng(0).choice(obs_rho.shape[0], n, replace=False)

    res: dict[str, list[float]] = {
        "neural_rmse": [],
        "occam_rmse": [],
        "neural_time": [],
        "occam_time": [],
        "neural_cov68": [],
        "occam_nrms": [],
    }
    for i in idx:
        pred = inv.invert(obs_rho[i], obs_phase[i], obs_grav[i])
        res["neural_rmse"].append(profile_rmse(pred.log10_rho, tgt[i]))
        res["neural_time"].append(pred.wall_time_s)
        res["neural_cov68"].append(coverage(pred.log10_rho, pred.sigma_log10_rho, tgt[i]))

        oc = occam1d_invert(obs_rho[i], obs_phase[i], periods)
        res["occam_rmse"].append(profile_rmse(oc.profile_on_grid(depth_grid), tgt[i]))
        res["occam_time"].append(oc.wall_time_s)
        res["occam_nrms"].append(oc.nrms)

    return {
        "schema": "pimsr-1d-mixed-budget-diagnostic",
        "schema_version": 1,
        "comparison_status": "diagnostic_non_comparable",
        "ranking_allowed": False,
        "inverse_observation_budget": {
            "neural": ["mt_log10_apparent_resistivity", "mt_phase", "gravity"],
            "occam": ["mt_log10_apparent_resistivity", "mt_phase"],
        },
        **{key: summarize(values) for key, values in res.items()},
    }


def run_real(args: argparse.Namespace) -> dict:
    inv = NeuralInverter(args.checkpoint)
    out: dict[str, dict] = {}
    for xml in args.xml:
        st = parse_emtf_xml(xml)
        log_rho, phase, mask = resample_station_determinant(st, inv.periods)
        pred = inv.invert(log_rho, phase, None)
        oc = occam1d_invert(log_rho[mask], phase[mask], inv.periods[mask])

        # Data misfit of each recovered profile against the observations.
        from pimsr_forward.mt1d import mt1d_response
        from pimsr_inversion.data import grid_cell_thicknesses

        thick = grid_cell_thicknesses(inv.depth_grid)
        nn_rho_a, nn_phase = mt1d_response(10.0**pred.log10_rho, thick, inv.periods[mask])
        oc_rho_a, oc_phase = mt1d_response(
            10.0**oc.log10_rho, oc.thicknesses, inv.periods[mask]
        )
        out[st.station_id] = {
            "lat": st.latitude,
            "lon": st.longitude,
            "n_periods_in_band": int(mask.sum()),
            "neural": {
                "nrms": data_nrms(
                    np.log10(nn_rho_a), nn_phase, log_rho[mask], phase[mask]
                ),
                "time_s": pred.wall_time_s,
                "scenario_probs": pred.scenario_probs.tolist(),
                "profile_log10_rho": pred.log10_rho.tolist(),
                "profile_sigma": pred.sigma_log10_rho.tolist(),
            },
            "occam": {
                "nrms": data_nrms(
                    np.log10(oc_rho_a), oc_phase, log_rho[mask], phase[mask]
                ),
                "time_s": oc.wall_time_s,
                "iterations": oc.n_iterations,
                "profile_log10_rho": oc.profile_on_grid(inv.depth_grid).tolist(),
            },
            "depth_grid": inv.depth_grid.tolist(),
        }
    return {
        "schema": "pimsr-1d-real-data-diagnostic",
        "schema_version": 1,
        "comparison_status": "diagnostic_no_ground_truth",
        "ranking_allowed": False,
        "inverse_observation_budget": {
            "neural": ["mt_log10_apparent_resistivity", "mt_phase"],
            "occam": ["mt_log10_apparent_resistivity", "mt_phase"],
        },
        "stations": out,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="pimsr-bench")
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("synthetic", help="benchmark on the synthetic test split")
    ps.add_argument("--dataset", required=True)
    ps.add_argument("--checkpoint", required=True)
    ps.add_argument("--n-stations", type=int, default=200)
    ps.add_argument("--out", required=True)
    ps.add_argument(
        "--allow-mixed-budget-diagnostic",
        action="store_true",
        help="allow a non-rankable MT+gravity neural versus MT-only Occam diagnostic",
    )

    pr = sub.add_parser("real", help="benchmark on real EMTF XML stations")
    pr.add_argument("--xml", nargs="+", required=True)
    pr.add_argument("--checkpoint", required=True)
    pr.add_argument("--out", required=True)

    args = p.parse_args(argv)
    artifacts: dict[str, object] = {
        "checkpoint": file_artifact_provenance(args.checkpoint)
    }
    if args.cmd == "synthetic":
        artifacts["dataset"] = file_artifact_provenance(args.dataset)
    else:
        artifacts["emtf_xml"] = [
            file_artifact_provenance(path) for path in args.xml
        ]
    result = run_synthetic(args) if args.cmd == "synthetic" else run_real(args)
    result["artifacts"] = artifacts
    require_file_artifact_unchanged(artifacts["checkpoint"], role="checkpoint")
    if args.cmd == "synthetic":
        require_file_artifact_unchanged(artifacts["dataset"], role="dataset")
    else:
        for xml in artifacts["emtf_xml"]:
            require_file_artifact_unchanged(xml, role="EMTF XML")
    publish_json_no_overwrite(result, args.out)
    print(json.dumps({k: v for k, v in list(result.items())[:3]}, indent=1))


if __name__ == "__main__":
    main()
