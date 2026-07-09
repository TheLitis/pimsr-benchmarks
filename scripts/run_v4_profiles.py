"""Score v4 (TE+TM, per-mode distortion augmentation) checkpoints on all
five USArray rows with the rigorous shift-invariant 2D-forward metric.

Rows: pretrained (zero-shot), ft-YS (single-profile), ft-joint (regional).
Key question vs v3: do the unseen rows I/K improve (v3 mean 7.42) while
keeping the H-YS zero-shot gain (v3 4.36)?
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import h5py
import numpy as np

warnings.filterwarnings("ignore")

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))

from pimsr_benchmarks.hybrid2d import (  # noqa: E402
    PROFILES,
    assemble_profile,
    assemble_profile_modes,
    section_nrms_2d,
)
from run_unified_leaderboard import unet_section  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-h5", required=True)
    ap.add_argument("--emtf-dir", default="data/emtf")
    ap.add_argument("--ckpt-pre", required=True)
    ap.add_argument("--ckpt-ft-ys", required=True)
    ap.add_argument("--ckpt-ft-joint", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with h5py.File(args.test_h5) as f:
        freqs = f["frequencies"][:]
        station_x = f["station_x"][:]
        x_grid = f["x_grid"][:]
        depth_grid = f["depth_grid"][:]

    ckpts = {
        "v4-pre": args.ckpt_pre,
        "v4-ft-YS": args.ckpt_ft_ys,
        "v4-ft-joint": args.ckpt_ft_joint,
    }

    board: dict[str, dict[str, float]] = {}
    for pname, pids in PROFILES.items():
        lr, ph, _, _ = assemble_profile(
            args.emtf_dir, freqs, station_x, profile_ids=pids
        )
        modes = assemble_profile_modes(
            args.emtf_dir, freqs, station_x, profile_ids=pids
        )
        row = {}
        for label, ck in ckpts.items():
            sec = unet_section(ck, lr, ph, modes=modes, profile_name=pname)
            row[label] = float(
                section_nrms_2d(
                    sec, lr, ph, freqs, station_x, x_grid, depth_grid
                )
            )
        board[pname] = row
        print(
            f"{pname:5s} | "
            + " | ".join(f"{k} {v:.2f}" for k, v in row.items()),
            flush=True,
        )

    methods = list(next(iter(board.values())).keys())
    summary = {
        m: float(np.mean([board[p][m] for p in board])) for m in methods
    }
    print("MEAN  | " + " | ".join(f"{k} {v:.2f}" for k, v in summary.items()))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"profiles": board, "mean": summary}, indent=1)
    )


if __name__ == "__main__":
    main()
