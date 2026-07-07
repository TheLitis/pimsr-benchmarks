"""Joint multi-profile fine-tuning study (out-of-row generalisation fix).

The sigma-reg 60k model fine-tuned on H-YS alone is sharp on H-YS (3.99)
but degrades on unseen rows (5.29 mean). Here we test whether averaging the
physics misfit over several profiles during fine-tuning recovers
generalisation without giving up the target-profile gain:

  1. joint-all : fine-tune on all five rows, score on each row.
  2. LOO       : for each row, fine-tune on the other four and score on the
                 held-out row — a true out-of-sample generalisation test.

Scored with the shift-invariant 2D-forward misfit (section_nrms_2d).
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import h5py
import numpy as np

warnings.filterwarnings("ignore")

from pimsr_benchmarks.hybrid2d import (  # noqa: E402
    PROFILES,
    assemble_profile,
    section_nrms_2d,
)
from run_unified_leaderboard import unet_section  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-h5", required=True, help="grids/frequencies source")
    ap.add_argument("--emtf-dir", default="data/emtf")
    ap.add_argument("--ckpt", required=True, help="base (pretrained) checkpoint")
    ap.add_argument("--ckpt-ys-ft", default=None, help="existing H-YS-only ft")
    ap.add_argument("--ft-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--anchor-weight", type=float, default=3.0)
    ap.add_argument("--skip-loo", action="store_true")
    args = ap.parse_args()

    from pimsr_inversion.finetune2d import finetune2d

    with h5py.File(args.test_h5) as f:
        freqs = f["frequencies"][:]
        station_x = f["station_x"][:]
        x_grid = f["x_grid"][:]
        depth_grid = f["depth_grid"][:]

    ft_dir = Path(args.ft_dir)
    ft_dir.mkdir(parents=True, exist_ok=True)

    obs = {
        p: assemble_profile(args.emtf_dir, freqs, station_x, profile_ids=ids)[:2]
        for p, ids in PROFILES.items()
    }

    def score(ckpt_path: str, pname: str) -> float:
        lr, ph = obs[pname]
        return section_nrms_2d(
            unet_section(ckpt_path, lr, ph),
            lr, ph, freqs, station_x, x_grid, depth_grid,
        )

    board: dict[str, dict[str, float]] = {p: {} for p in PROFILES}

    # baselines
    for p in PROFILES:
        board[p]["pretrained"] = score(args.ckpt, p)
        if args.ckpt_ys_ft:
            board[p]["ft-YS-only"] = score(args.ckpt_ys_ft, p)

    # 1. joint fine-tune on all five profiles
    joint_path = ft_dir / "best2d_ft_joint_all.pt"
    if not joint_path.exists():
        finetune2d(
            checkpoint=args.ckpt,
            emtf_dir=args.emtf_dir,
            data_h5=args.test_h5,
            out=str(joint_path),
            steps=args.steps,
            anchor_weight=args.anchor_weight,
            profiles=[PROFILES[p] for p in PROFILES],
        )
    for p in PROFILES:
        board[p]["ft-joint-all"] = score(str(joint_path), p)

    # 2. leave-one-out: fine-tune on the other four, score held-out row
    if not args.skip_loo:
        for held in PROFILES:
            loo_path = ft_dir / f"best2d_ft_loo_{held}.pt"
            if not loo_path.exists():
                finetune2d(
                    checkpoint=args.ckpt,
                    emtf_dir=args.emtf_dir,
                    data_h5=args.test_h5,
                    out=str(loo_path),
                    steps=args.steps,
                    anchor_weight=args.anchor_weight,
                    profiles=[PROFILES[p] for p in PROFILES if p != held],
                )
            board[held]["ft-loo"] = score(str(loo_path), held)
            print(f"LOO {held}: {board[held]['ft-loo']:.2f}", flush=True)

    methods = list(next(iter(board.values())).keys())
    summary = {
        m: float(np.mean([board[p][m] for p in board if m in board[p]]))
        for m in methods
    }
    for p, row in board.items():
        print(f"{p:5s} | " + " | ".join(f"{k} {v:.2f}" for k, v in row.items()))
    print("MEAN  | " + " | ".join(f"{k} {v:.2f}" for k, v in summary.items()))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"profiles": board, "mean": summary}, f, indent=1)
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
