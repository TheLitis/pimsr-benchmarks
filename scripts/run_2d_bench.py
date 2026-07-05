"""Benchmark the conv-2D inversion network.

Synthetic: section RMSE + sigma coverage on the 2D test split.
Real: assemble an E-W USArray profile (~44.6N through Yellowstone) into a
pseudo-section, invert, and report the physics misfit of the recovered
section re-simulated station-by-station with the 1D forward (a conservative
check: the 2D network may legitimately disagree with per-station 1D).

Usage:
    python scripts/run_2d_bench.py --checkpoint best2d.pt \
        --test-h5 ds2d_test.h5 --emtf-dir data/emtf --out-dir results
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from pimsr_benchmarks.emtf import parse_emtf_xml, resample_station
from pimsr_benchmarks.metrics import summarize
from pimsr_forward.mt1d import mt1d_response
from pimsr_inversion.network2d import PimsrNet2D

#: E-W profile at ~44.6N, west to east.
PROFILE_IDS = ["MTH15", "MTH16", "WYYS1", "WYYS2", "WYYS3", "WYH18", "WYH19"]


def load_model(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = PimsrNet2D(
        n_freq=int(ckpt["n_freq"]),
        n_stations=int(ckpt["n_stations"]),
        n_depth=int(ckpt["n_depth"]),
        n_x=int(ckpt["n_x"]),
        n_scenarios=int(ckpt["n_scenarios"]),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def bench_synthetic(model, ckpt, test_h5: str, n: int) -> dict:
    with h5py.File(test_h5, "r") as f:
        lr = f["obs_mt_log10_rho"][:n].astype(np.float32)
        ph = f["obs_mt_phase"][:n].astype(np.float32) / 45.0
        tgt = f["target_log10_res"][:n].astype(np.float32)
        scen = f["scenario"][:n]
    obs = np.stack([lr, ph], axis=1)
    obs = (obs - ckpt["stats_mean"]) / ckpt["stats_std"]

    t0 = time.time()
    with torch.no_grad():
        out = model(torch.from_numpy(obs.astype(np.float32)))
    dt = time.time() - t0

    pred = out["log_rho"].numpy()
    sigma = np.exp(0.5 * out["log_sigma_rho"].numpy())
    rmses = np.sqrt(((pred - tgt) ** 2).mean(axis=(1, 2)))
    cov1 = float((np.abs(pred - tgt) < sigma).mean())
    acc = float(
        (out["scenario_logits"].argmax(dim=1).numpy() == scen).mean()
    )
    return {
        "method": "conv2d",
        "n": int(len(rmses)),
        "rmse_log10_res": summarize(rmses.tolist()),
        "sigma_coverage_1": cov1,
        "scenario_accuracy": acc,
        "time_per_section_s": dt / len(rmses),
        "per_scenario_rmse": {
            str(s): float(rmses[scen == s].mean()) for s in np.unique(scen)
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test-h5", required=True)
    ap.add_argument("--emtf-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()

    model, ckpt = load_model(args.checkpoint)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    syn = bench_synthetic(model, ckpt, args.test_h5, args.n)
    (out_dir / "conv2d_synthetic.json").write_text(json.dumps(syn, indent=2))
    print("synthetic:", json.dumps(syn["rmse_log10_res"]))
    print("coverage:", syn["sigma_coverage_1"], "| scen acc:", syn["scenario_accuracy"])

    with h5py.File(args.test_h5, "r") as f:
        freqs = f["frequencies"][:]
        station_x = f["station_x"][:]

    real = bench_real_profile(model, ckpt, args.emtf_dir, freqs, station_x)
    np.savez(
        out_dir / "conv2d_real_profile.npz",
        section=real["section"],
        lr_obs=real["lr_obs"],
        ph_obs=real["ph_obs"],
        x_model=real["x_model"],
    )
    (out_dir / "conv2d_real.json").write_text(
        json.dumps({k: v for k, v in real.items() if isinstance(v, (list, float, int))},
                   indent=2)
    )
    print("real profile nRMS:", real.get("nrms_mean"))


def bench_real_profile(model, ckpt, emtf_dir, freqs, station_x) -> dict:
    """Invert the USArray profile and physics-check the recovered section."""
    stations = {}
    for f in glob.glob(f"{emtf_dir}/*.xml"):
        st = parse_emtf_xml(f)
        stations[st.station_id] = st
    profile = [stations[i] for i in PROFILE_IDS]

    periods = 1.0 / freqs
    n_f, n_s = len(freqs), len(station_x)

    lon = np.array([s.longitude for s in profile])
    x_km = (lon - lon.min()) * 111.0 * np.cos(np.radians(44.6))
    x_model = np.linspace(x_km.min(), x_km.max(), n_s)

    lr_st = np.empty((n_f, len(profile)))
    ph_st = np.empty((n_f, len(profile)))
    for j, st in enumerate(profile):
        lr_st[:, j], ph_st[:, j], _ = resample_station(st, periods)
    lr = np.stack([np.interp(x_model, x_km, lr_st[i]) for i in range(n_f)])
    ph = np.stack([np.interp(x_model, x_km, ph_st[i]) for i in range(n_f)])

    obs = np.stack([lr, ph / 45.0])[None].astype(np.float32)
    obs = (obs - ckpt["stats_mean"]) / ckpt["stats_std"]
    with torch.no_grad():
        out = model(torch.from_numpy(obs.astype(np.float32)))
    section = out["log_rho"][0].numpy()

    # physics check: re-simulate each station column with the 1D forward on
    # the depth grid and compare to the observed response at that x.
    # 48-node log depth grid must match the dataset's depth_grid; we rebuild
    # layer thicknesses from consecutive node spacing.
    from pimsr_geogen.model import DEFAULT_DEPTH_GRID

    z = DEFAULT_DEPTH_GRID
    thick = np.diff(z)
    nrms_list = []
    for j_model, x in enumerate(np.linspace(0, section.shape[1] - 1, len(profile)).astype(int)):
        col = section[:, x]
        rho = 10.0 ** col
        # len(z) resistivities with len(z) - 1 thicknesses: the last grid
        # value acts as the terminating half-space.
        sim_rho, sim_ph = mt1d_response(rho, thick, periods)
        jx = int(np.argmin(np.abs(x_model - x_km[j_model])))
        d_lr = lr[:, jx] - np.log10(sim_rho)
        d_lr -= d_lr.mean()  # static-shift invariant
        d_ph = ph[:, jx] - sim_ph
        err = np.sqrt(np.mean(d_lr**2 / 0.05**2 + (d_ph / 2.9) ** 2 / 2.0))
        nrms_list.append(float(err))

    return {
        "profile": PROFILE_IDS,
        "nrms_mean": float(np.mean(nrms_list)),
        "nrms_per_station": nrms_list,
        "section": section,
        "lr_obs": lr,
        "ph_obs": ph,
        "x_model": x_model,
    }


if __name__ == "__main__":
    main()
