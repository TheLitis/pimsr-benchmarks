"""Score v4 (TE+TM, per-mode distortion augmentation) checkpoints on all
five USArray rows with the rigorous shift-invariant 2D-forward metric.

Rows: pretrained (zero-shot), ft-YS (single-profile), ft-joint (regional).
Key question vs v3: do the unseen rows I/K improve (v3 mean 7.42) while
keeping the H-YS zero-shot gain (v3 4.36)?
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

import sys

sys.path.insert(0, str(Path(__file__).parent))

from run_unified_leaderboard import unet_section

from pimsr_benchmarks.hybrid2d import (
    PROFILES,
    SECTION_NRMS_METRIC_ID,
    assemble_profile_modes,
    profile_geometry_metadata,
    section_nrms_2d,
)
from pimsr_benchmarks.runner2d import (
    checkpoint_adaptation_kind,
    load_model2d,
    publish_json_no_overwrite,
    require_finetune2d_lineage,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-h5", required=True)
    ap.add_argument("--emtf-dir", default="data/emtf")
    ap.add_argument("--ckpt-pre", required=True)
    ap.add_argument("--ckpt-ft-ys", required=True)
    ap.add_argument("--ckpt-ft-joint", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    models = {
        "v4-pre": load_model2d(args.ckpt_pre, args.test_h5),
        "v4-ft-YS": load_model2d(args.ckpt_ft_ys, args.test_h5),
        "v4-ft-joint": load_model2d(args.ckpt_ft_joint, args.test_h5),
    }
    if checkpoint_adaptation_kind(models["v4-pre"].checkpoint) != "zero-shot":
        raise ValueError("--ckpt-pre is not a zero-shot checkpoint")
    require_finetune2d_lineage(
        models["v4-ft-YS"],
        base=models["v4-pre"],
        emtf_dir=args.emtf_dir,
        expected_profiles=[PROFILES["H-YS"]],
    )
    require_finetune2d_lineage(
        models["v4-ft-joint"],
        base=models["v4-pre"],
        emtf_dir=args.emtf_dir,
        expected_profiles=[PROFILES[p] for p in PROFILES],
    )
    contract = models["v4-pre"].contract
    freqs = contract.frequencies
    station_x = contract.station_x
    x_grid = contract.x_grid
    depth_grid = contract.depth_grid

    board: dict[str, dict[str, float]] = {}
    geometry: dict[str, dict[str, object]] = {}
    for pname, pids in PROFILES.items():
        modes = assemble_profile_modes(args.emtf_dir, freqs, station_x, profile_ids=pids)
        geometry[pname] = profile_geometry_metadata(modes)
        row = {}
        for label, loaded in models.items():
            sec = unet_section(loaded, modes, profile_name=pname)
            row[label] = float(
                section_nrms_2d(sec, modes, freqs, station_x, x_grid, depth_grid)
            )
        board[pname] = row
        print(
            f"{pname:5s} | " + " | ".join(f"{k} {v:.2f}" for k, v in row.items()),
            flush=True,
        )

    methods = list(next(iter(board.values())).keys())
    summary = {m: float(np.mean([board[p][m] for p in board])) for m in methods}
    print("MEAN  | " + " | ".join(f"{k} {v:.2f}" for k, v in summary.items()))

    for loaded in models.values():
        loaded.require_artifacts_unchanged()
    publish_json_no_overwrite(
        {
            "schema_version": 3,
            "comparison_status": "diagnostic_normalized_geometry",
            "ranking_allowed": False,
            "diagnostic_reasons": [
                "field profiles are assembled on the synthetic model station grid rather than native physical geometry",
                "the joint checkpoint adapts on every scored profile and is not a held-out estimate",
            ],
            "metric_id": SECTION_NRMS_METRIC_ID,
            "geometry": geometry,
            "profiles": board,
            "mean": summary,
            "artifacts": {
                label: loaded.artifact_provenance() for label, loaded in models.items()
            },
        },
        args.out,
    )


if __name__ == "__main__":
    main()
