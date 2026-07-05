#!/usr/bin/env python3
"""Run the trained PIMSR neural inverter on the synthetic test split and the
real USArray stations, writing JSON results next to the Occam baselines.

Usage:
    python scripts/run_neural_bench.py --checkpoint /path/to/best.pt \
        --test-h5 /path/to/ds_test.h5 --emtf-dir data/emtf --out-dir /path/out
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import h5py
import numpy as np
from pimsr_forward.mt1d import mt1d_response

from pimsr_benchmarks.emtf import parse_emtf_xml, resample_station
from pimsr_benchmarks.metrics import coverage, profile_rmse, summarize
from pimsr_benchmarks.neural import NeuralInverter


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
    thick = np.diff(depth_grid)
    rho_a, phase = mt1d_response(rho, thick, periods)
    r_lr = (np.log10(rho_a) - obs_log_rho_a) / err_log_rho
    r_ph = (phase - obs_phase) / err_phase_deg
    res = np.concatenate([r_lr, r_ph])
    if mask is not None:
        res = np.concatenate([r_lr[mask], r_ph[mask]])
    return float(np.sqrt(np.mean(res**2)))


def bench_synthetic(
    inv: NeuralInverter, test_h5: str, n: int
) -> dict:
    with h5py.File(test_h5) as f:
        lr = f["obs_mt_log10_rho"][:n].astype(np.float64)
        ph = f["obs_mt_phase"][:n].astype(np.float64)
        gz = f["obs_gravity_mgal"][:n].astype(np.float64)
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

    cov = coverage(
        np.asarray(preds), np.asarray(sigmas), tgt_res
    )
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


def bench_real(inv: NeuralInverter, emtf_dir: str) -> dict:
    stations = []
    for path in sorted(glob.glob(os.path.join(emtf_dir, "*.xml"))):
        st = parse_emtf_xml(path)
        lr, ph, mask = resample_station(st, inv.periods)
        p = inv.invert(lr, ph, gravity=None)
        nrms = mt_nrms(
            p.log10_rho, inv.depth_grid, inv.periods, lr, ph, mask=mask
        )
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

    inv = NeuralInverter(args.checkpoint)
    os.makedirs(args.out_dir, exist_ok=True)

    syn = bench_synthetic(inv, args.test_h5, args.n)
    with open(os.path.join(args.out_dir, "neural_synthetic.json"), "w") as fh:
        json.dump(syn, fh, indent=2)
    print(
        json.dumps(
            {k: v for k, v in syn.items() if k != "per_scenario_rmse"}, indent=2
        )
    )

    real = bench_real(inv, args.emtf_dir)
    with open(os.path.join(args.out_dir, "neural_real.json"), "w") as fh:
        json.dump(real, fh, indent=2)
    print("real mean nRMS:", real["nrms"]["mean"])


if __name__ == "__main__":
    main()
