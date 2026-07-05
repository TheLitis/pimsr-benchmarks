"""Benchmark the hybrid (neural warm start + Occam refinement) inversion.

Runs both the synthetic test split and the real USArray stations, emitting
JSON files compatible with make_report.py.
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import h5py
import numpy as np

from pimsr_benchmarks.emtf import parse_emtf_xml, resample_station
from pimsr_benchmarks.hybrid import hybrid_invert
from pimsr_benchmarks.metrics import profile_rmse, summarize
from pimsr_benchmarks.neural import NeuralInverter


def bench_synthetic(inverter: NeuralInverter, test_h5: str, n: int) -> dict:
    with h5py.File(test_h5) as f:
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


def bench_real(inverter: NeuralInverter, emtf_dir: str) -> dict:
    stations = []
    for f in sorted(glob.glob(str(Path(emtf_dir) / "*.xml"))):
        st = parse_emtf_xml(f)
        lr = np.log10(st.rho_a_det)
        n_lr, n_ph, _ = resample_station(st, inverter.periods)
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

    inverter = NeuralInverter(args.checkpoint)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    syn = bench_synthetic(inverter, args.test_h5, args.n)
    json.dump(syn, open(out / "hybrid_synthetic.json", "w"), indent=2)
    print("\nSYNTHETIC:", json.dumps(syn["rmse_log10_res"]))

    real = bench_real(inverter, args.emtf_dir)
    json.dump(real, open(out / "hybrid_real.json", "w"), indent=2)
    print("\nREAL nRMS:", json.dumps(real["nrms"]))


if __name__ == "__main__":
    main()
