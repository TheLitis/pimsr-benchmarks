"""Multi-profile evaluation: five independent USArray lines, one metric.

For each of the five E-W rows (G, H-YS, I, J, K) we score:
  - occam1d-stitched   (classical per-station, laterally stitched)
  - neural1d-stitched  (1D net per station, stitched)
  - unet-60k           (2D net, pretrained only)
  - unet-60k-ft-YS     (fine-tuned on H-YS: transfer to unseen lines)
  - unet-60k-ft-self   (fine-tuned on the evaluated line itself)

Everything is scored with the rigorous shift-invariant 2D-forward misfit.
The ft-YS row on lines != H-YS measures whether single-profile fine-tuning
generalises or merely memorises the profile it saw.
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
from pimsr_benchmarks.neural import NeuralInverter  # noqa: E402
from pimsr_benchmarks.occam1d import occam1d_invert  # noqa: E402
from run_unified_leaderboard import (  # noqa: E402
    columns_to_section,
    neural_1d_section,
    unet_section,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-h5", required=True)
    ap.add_argument("--emtf-dir", default="data/emtf")
    ap.add_argument("--ckpt-1d", required=True)
    ap.add_argument("--ckpt-60k", required=True)
    ap.add_argument("--ckpt-60k-ft", required=True, help="fine-tuned on H-YS")
    ap.add_argument("--ft-dir", required=True, help="dir for per-profile ft ckpts")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with h5py.File(args.test_h5) as f:
        freqs = f["frequencies"][:]
        station_x = f["station_x"][:]
        x_grid = f["x_grid"][:]
        depth_grid = f["depth_grid"][:]
    periods = 1.0 / freqs

    inv1d = NeuralInverter(args.ckpt_1d)
    Path(args.ft_dir).mkdir(parents=True, exist_ok=True)

    board: dict[str, dict[str, float]] = {}
    for pname, pids in PROFILES.items():
        lr, ph, _, _ = assemble_profile(
            args.emtf_dir, freqs, station_x, profile_ids=pids
        )

        def score(section, lr=lr, ph=ph):
            return section_nrms_2d(
                section, lr, ph, freqs, station_x, x_grid, depth_grid
            )

        row: dict[str, float] = {}

        cols = []
        for j in range(lr.shape[1]):
            res = occam1d_invert(lr[:, j], ph[:, j], periods, max_iterations=30)
            cols.append(res.profile_on_grid(depth_grid))
        row["occam1d"] = score(
            columns_to_section(np.stack(cols, axis=1), station_x, x_grid)
        )

        row["neural1d"] = score(
            neural_1d_section(inv1d, lr, ph, periods, station_x, x_grid, depth_grid)
        )

        row["unet-60k"] = score(unet_section(args.ckpt_60k, lr, ph))
        row["unet-60k-ft-YS"] = score(unet_section(args.ckpt_60k_ft, lr, ph))

        # per-profile fine-tune (self)
        ft_path = Path(args.ft_dir) / f"best2d_ft_{pname}.pt"
        if not ft_path.exists():
            from pimsr_inversion.finetune2d import finetune2d

            finetune2d(
                checkpoint=args.ckpt_60k,
                emtf_dir=args.emtf_dir,
                data_h5=args.test_h5,
                out=str(ft_path),
                steps=600,
                lr=2.0e-5,
                anchor_weight=3.0,
                jitter=0.02,
                profile_ids=pids,
            )
        row["unet-60k-ft-self"] = score(unet_section(str(ft_path), lr, ph))

        board[pname] = row
        print(
            f"{pname:5s} | "
            + " | ".join(f"{k} {v:.2f}" for k, v in row.items()),
            flush=True,
        )

    # summary: mean over profiles
    methods = list(next(iter(board.values())).keys())
    summary = {
        m: float(np.mean([board[p][m] for p in board])) for m in methods
    }
    print("MEAN  | " + " | ".join(f"{k} {v:.2f}" for k, v in summary.items()))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"profiles": board, "mean": summary}, f, indent=1)
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
