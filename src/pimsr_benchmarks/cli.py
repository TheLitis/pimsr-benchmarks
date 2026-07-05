"""Benchmark runner CLI.

Subcommands
-----------
synthetic : neural vs Occam on the held-out synthetic test split
real      : neural vs Occam on real EMTF stations (USArray)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from .emtf import parse_emtf_xml, resample_station
from .metrics import coverage, data_nrms, profile_rmse, summarize
from .neural import NeuralInverter
from .occam1d import occam1d_invert


def run_synthetic(args: argparse.Namespace) -> dict:
    inv = NeuralInverter(args.checkpoint)
    with h5py.File(args.dataset, "r") as f:
        obs_rho = f["obs_mt_log10_rho"][:]
        obs_phase = f["obs_mt_phase"][:]
        obs_grav = f["obs_gravity"][:]
        tgt = f["target_log10_res"][:]
        periods = f["periods"][:]
        depth_grid = f["depth_grid"][:]

    n = min(args.n_stations, obs_rho.shape[0])
    idx = np.random.default_rng(0).choice(obs_rho.shape[0], n, replace=False)

    res: dict[str, list[float]] = {
        "neural_rmse": [], "occam_rmse": [],
        "neural_time": [], "occam_time": [],
        "neural_cov68": [], "occam_nrms": [],
    }
    for i in idx:
        pred = inv.invert(obs_rho[i], obs_phase[i], obs_grav[i])
        res["neural_rmse"].append(profile_rmse(pred.log10_rho, tgt[i]))
        res["neural_time"].append(pred.wall_time_s)
        res["neural_cov68"].append(
            coverage(pred.log10_rho, pred.sigma_log10_rho, tgt[i])
        )

        oc = occam1d_invert(obs_rho[i], obs_phase[i], periods)
        res["occam_rmse"].append(profile_rmse(oc.profile_on_grid(depth_grid), tgt[i]))
        res["occam_time"].append(oc.wall_time_s)
        res["occam_nrms"].append(oc.nrms)

    return {k: summarize(v) for k, v in res.items()}


def run_real(args: argparse.Namespace) -> dict:
    inv = NeuralInverter(args.checkpoint)
    out: dict[str, dict] = {}
    for xml in args.xml:
        st = parse_emtf_xml(xml)
        log_rho, phase, mask = resample_station(st, inv.periods)
        pred = inv.invert(log_rho, phase, None)
        oc = occam1d_invert(log_rho[mask], phase[mask], inv.periods[mask])

        # Data misfit of each recovered profile against the observations.
        from pimsr_forward.mt1d import mt1d_response
        from pimsr_inversion.data import grid_cell_thicknesses

        thick = grid_cell_thicknesses(inv.depth_grid)
        nn_rho_a, nn_phase = mt1d_response(
            10.0 ** pred.log10_rho, thick, inv.periods[mask]
        )
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
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="pimsr-bench")
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("synthetic", help="benchmark on the synthetic test split")
    ps.add_argument("--dataset", required=True)
    ps.add_argument("--checkpoint", required=True)
    ps.add_argument("--n-stations", type=int, default=200)
    ps.add_argument("--out", required=True)

    pr = sub.add_parser("real", help="benchmark on real EMTF XML stations")
    pr.add_argument("--xml", nargs="+", required=True)
    pr.add_argument("--checkpoint", required=True)
    pr.add_argument("--out", required=True)

    args = p.parse_args(argv)
    result = run_synthetic(args) if args.cmd == "synthetic" else run_real(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in list(result.items())[:3]}, indent=1))


if __name__ == "__main__":
    main()
