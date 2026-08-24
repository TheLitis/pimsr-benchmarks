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
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from run_unified_leaderboard import unet_section

from pimsr_benchmarks.hybrid2d import (
    PROFILES,
    SECTION_NRMS_METRIC_ID,
    assemble_profile_modes,
    profile_geometry_metadata,
    section_nrms_2d,
)
from pimsr_benchmarks.runner2d import (
    LoadedModel2D,
    load_model2d,
    publish_json_no_overwrite,
    require_finetune2d_lineage,
)


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

    base_model = load_model2d(args.ckpt, args.test_h5)
    ys_model = load_model2d(args.ckpt_ys_ft, args.test_h5) if args.ckpt_ys_ft else None
    loaded_models = {"pretrained": base_model}
    if ys_model is not None:
        loaded_models["ft-YS-only"] = ys_model
        require_finetune2d_lineage(
            ys_model,
            base=base_model,
            emtf_dir=args.emtf_dir,
            expected_profiles=[PROFILES["H-YS"]],
        )
    contract = base_model.contract
    freqs = contract.frequencies
    station_x = contract.station_x
    x_grid = contract.x_grid
    depth_grid = contract.depth_grid

    ft_dir = Path(args.ft_dir)
    ft_dir.mkdir(parents=True, exist_ok=True)

    obs = {
        p: assemble_profile_modes(args.emtf_dir, freqs, station_x, profile_ids=ids)
        for p, ids in PROFILES.items()
    }

    def score(loaded: LoadedModel2D, pname: str) -> float:
        modes = obs[pname]
        return section_nrms_2d(
            unet_section(loaded, modes),
            modes,
            freqs,
            station_x,
            x_grid,
            depth_grid,
        )

    board: dict[str, dict[str, float]] = {p: {} for p in PROFILES}

    # baselines
    for p in PROFILES:
        board[p]["pretrained"] = score(base_model, p)
        if ys_model is not None:
            board[p]["ft-YS-only"] = score(ys_model, p)

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
    joint_model = load_model2d(joint_path, args.test_h5)
    require_finetune2d_lineage(
        joint_model,
        base=base_model,
        emtf_dir=args.emtf_dir,
        expected_profiles=[PROFILES[p] for p in PROFILES],
        expected_options={
            "seed": 0,
            "steps": args.steps,
            "anchor_weight": float(args.anchor_weight),
        },
    )
    loaded_models["ft-joint-all"] = joint_model
    for p in PROFILES:
        board[p]["ft-joint-all"] = score(joint_model, p)

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
            loo_model = load_model2d(loo_path, args.test_h5)
            require_finetune2d_lineage(
                loo_model,
                base=base_model,
                emtf_dir=args.emtf_dir,
                expected_profiles=[PROFILES[p] for p in PROFILES if p != held],
                expected_options={
                    "seed": 0,
                    "steps": args.steps,
                    "anchor_weight": float(args.anchor_weight),
                },
            )
            loaded_models[f"ft-loo-{held}"] = loo_model
            board[held]["ft-loo"] = score(loo_model, held)
            print(f"LOO {held}: {board[held]['ft-loo']:.2f}", flush=True)

    methods = list(next(iter(board.values())).keys())
    summary = {
        m: float(np.mean([board[p][m] for p in board if m in board[p]])) for m in methods
    }
    for p, row in board.items():
        print(f"{p:5s} | " + " | ".join(f"{k} {v:.2f}" for k, v in row.items()))
    print("MEAN  | " + " | ".join(f"{k} {v:.2f}" for k, v in summary.items()))

    for loaded in loaded_models.values():
        loaded.require_artifacts_unchanged()
    publish_json_no_overwrite(
        {
            "schema_version": 3,
            "comparison_status": "diagnostic_normalized_geometry",
            "ranking_allowed": False,
            "diagnostic_reasons": [
                "field profiles are assembled on the synthetic model station grid rather than native physical geometry",
                "joint-all rows adapt on every scored profile and are not held-out estimates",
            ],
            "metric_id": SECTION_NRMS_METRIC_ID,
            "geometry": {
                profile: profile_geometry_metadata(modes)
                for profile, modes in obs.items()
            },
            "profiles": board,
            "mean": summary,
            "artifacts": {
                label: loaded.artifact_provenance()
                for label, loaded in loaded_models.items()
            },
        },
        args.out,
    )
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
