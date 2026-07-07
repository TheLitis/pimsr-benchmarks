"""Benchmark the 2D hybrid (U-Net warm start + SimPEG GN) on the real profile.

Three rows, all scored with the identical per-station physics nRMS:
  1. U-Net alone (single pass) — reproduces the existing leaderboard row.
  2. Hybrid: U-Net section -> a few SimPEG inexact-GN iterations.
  3. Cold control: 100 ohm-m half-space -> the same GN budget.

Usage:
    python scripts/run_2d_hybrid_bench.py --checkpoint best2d.pt \
        --test-h5 ds2d_test.h5 --emtf-dir data/emtf --out-dir results \
        [--max-iter 5]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from pimsr_benchmarks.hybrid2d import (
    PROFILE_IDS,
    assemble_profile,
    refine_section_2d,
    section_nrms,
    section_nrms_2d,
)
from pimsr_inversion.network2d import PimsrNet2D


def load_model(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = PimsrNet2D.from_checkpoint(ckpt)
    model.eval()
    return model, ckpt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test-h5", required=True)
    ap.add_argument("--emtf-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-iter", type=int, default=5)
    ap.add_argument("--skip-cold", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.test_h5, "r") as f:
        freqs = f["frequencies"][:]
        station_x = f["station_x"][:]
        x_grid = f["x_grid"][:]
        depth_grid = f["depth_grid"][:]
    periods = 1.0 / freqs

    lr, ph, x_model, x_km = assemble_profile(args.emtf_dir, freqs, station_x)

    # ---- row 1: U-Net single pass -------------------------------------
    model, ckpt = load_model(args.checkpoint)
    obs = np.stack([lr, ph / 45.0])[None].astype(np.float32)
    obs = (obs - ckpt["stats_mean"]) / ckpt["stats_std"]
    with torch.no_grad():
        out = model(torch.from_numpy(obs.astype(np.float32)))
    net_section = out["log_rho"][0].numpy()
    net_nrms, net_list = section_nrms(
        net_section, lr, ph, x_model, x_km, periods, depth_grid
    )
    net_2d = section_nrms_2d(net_section, lr, ph, freqs, station_x, x_grid, depth_grid)
    print(f"unet single pass    | 1D-col nRMS {net_nrms:.2f} | 2D nRMS {net_2d:.2f}", flush=True)

    # ---- row 2: hybrid = warm start + GN -------------------------------
    hy = refine_section_2d(
        net_section, lr, ph, freqs, station_x, x_grid, depth_grid,
        max_iter=args.max_iter,
    )
    hy_nrms, hy_list = section_nrms(
        hy.section, lr, ph, x_model, x_km, periods, depth_grid
    )
    hy_2d = section_nrms_2d(hy.section, lr, ph, freqs, station_x, x_grid, depth_grid)
    print(
        f"hybrid (net + GN{args.max_iter}) | 1D-col nRMS {hy_nrms:.2f} "
        f"| 2D nRMS {hy_2d:.2f} | {hy.wall_time_s:.0f} s",
        flush=True,
    )

    results = {
        "profile": PROFILE_IDS,
        "max_iter": args.max_iter,
        "unet": {
            "nrms_mean": net_nrms,
            "nrms_2d": net_2d,
            "nrms_per_station": net_list,
        },
        "hybrid": {
            "nrms_mean": hy_nrms,
            "nrms_2d": hy_2d,
            "nrms_per_station": hy_list,
            "wall_time_s": hy.wall_time_s,
        },
    }

    # ---- row 3: cold-start control -------------------------------------
    if not args.skip_cold:
        cold0 = np.full_like(net_section, 2.0)  # 100 ohm-m half-space
        cold = refine_section_2d(
            cold0, lr, ph, freqs, station_x, x_grid, depth_grid,
            max_iter=args.max_iter,
        )
        cold_nrms, cold_list = section_nrms(
            cold.section, lr, ph, x_model, x_km, periods, depth_grid
        )
        cold_2d = section_nrms_2d(
            cold.section, lr, ph, freqs, station_x, x_grid, depth_grid
        )
        print(
            f"cold GN{args.max_iter} control   | 1D-col nRMS {cold_nrms:.2f} "
            f"| 2D nRMS {cold_2d:.2f} | {cold.wall_time_s:.0f} s",
            flush=True,
        )
        results["cold"] = {
            "nrms_mean": cold_nrms,
            "nrms_2d": cold_2d,
            "nrms_per_station": cold_list,
            "wall_time_s": cold.wall_time_s,
        }
        np.savez(
            out_dir / "hybrid2d_sections.npz",
            unet=net_section,
            hybrid=hy.section,
            cold=cold.section,
        )
    else:
        np.savez(
            out_dir / "hybrid2d_sections.npz", unet=net_section, hybrid=hy.section
        )

    (out_dir / "hybrid2d_real.json").write_text(json.dumps(results, indent=2))
    print("saved to", out_dir, flush=True)


if __name__ == "__main__":
    main()
