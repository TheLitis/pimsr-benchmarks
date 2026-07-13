"""Publication-grade uncertainty analysis for a 2D checkpoint and test split."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from pimsr_benchmarks.statistics import bootstrap_ci, calibration_summary
from pimsr_inversion.network2d import PimsrNet2D


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-h5", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n", type=int, default=0, help="0 means all samples")
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument(
        "--adaptation", required=True,
        choices=("zero-shot", "profile-adapted", "regional/joint-adapted"),
    )
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = PimsrNet2D.from_checkpoint(ckpt).eval()
    sl = slice(None if args.n == 0 else args.n)
    with h5py.File(args.test_h5) as f:
        chans = [
            f["obs_mt_log10_rho"][sl].astype(np.float32),
            f["obs_mt_phase"][sl].astype(np.float32) / 45.0,
        ]
        if model.in_channels == 4:
            chans += [
                f["obs_mt_log10_rho_tm"][sl].astype(np.float32),
                f["obs_mt_phase_tm"][sl].astype(np.float32) / 45.0,
            ]
        target = f["target_log10_res"][sl].astype(np.float32)
        scenario = f["scenario"][sl]
        depth = f["depth_grid"][:]
    obs = np.stack(chans, axis=1)
    obs = (obs - ckpt["stats_mean"]) / ckpt["stats_std"]
    with torch.inference_mode():
        pred = model(torch.from_numpy(obs))
    mean = pred["log_rho"].numpy()
    log_sigma = pred["log_sigma_rho"].numpy()
    affine = ckpt.get("sigma_affine2d")
    if affine:
        log_sigma = affine["a"] * log_sigma + affine["b"]
    sigma = np.exp(0.5 * log_sigma)
    rmse = np.sqrt(np.mean((mean - target) ** 2, axis=(1, 2)))
    calibration = calibration_summary(mean, sigma, target, depth_axis=1, scenario=scenario)
    output = {
        "schema_version": 1,
        "method": "conv2d",
        "adaptation": args.adaptation,
        "score_interpretation": "synthetic geological error; real-profile nRMS is reported separately as forward consistency",
        "n": int(len(rmse)),
        "rmse": bootstrap_ci(rmse, seed=args.seed),
        "calibration": calibration,
        "checkpoint": str(Path(args.checkpoint).name),
        "test_dataset": str(Path(args.test_h5).name),
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "uncertainty2d.json").write_text(json.dumps(output, indent=2) + "\n")
    with (out / "coverage_by_depth.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("depth_m", "coverage68"))
        writer.writerows(zip(depth, calibration["coverage68_by_depth"], strict=True))
    print(json.dumps(output["rmse"], indent=2))


if __name__ == "__main__":
    main()
