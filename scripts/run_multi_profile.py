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
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from run_unified_leaderboard import (
    columns_to_section,
    neural_1d_section,
    unet_section,
)

from pimsr_benchmarks.hybrid2d import (
    PROFILES,
    SECTION_NRMS_METRIC_ID,
    assemble_profile_modes,
    profile_geometry_metadata,
    section_nrms_2d,
)
from pimsr_benchmarks.neural import NeuralInverter
from pimsr_benchmarks.occam1d import occam1d_invert
from pimsr_benchmarks.runner2d import (
    load_model2d,
    publish_json_no_overwrite,
    require_finetune2d_lineage,
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

    model_60k = load_model2d(args.ckpt_60k, args.test_h5)
    model_60k_ft = load_model2d(args.ckpt_60k_ft, args.test_h5)
    loaded_models = {"unet-60k": model_60k, "unet-60k-ft-YS": model_60k_ft}
    adaptation_lineage = {
        "unet-60k-ft-YS": require_finetune2d_lineage(
            model_60k_ft,
            base=model_60k,
            emtf_dir=args.emtf_dir,
            expected_profiles=[PROFILES["H-YS"]],
        )["lineage_sha256"]
    }
    contract = model_60k.contract
    freqs = contract.frequencies
    station_x = contract.station_x
    x_grid = contract.x_grid
    depth_grid = contract.depth_grid
    periods = 1.0 / freqs

    inv1d = NeuralInverter(args.ckpt_1d)
    Path(args.ft_dir).mkdir(parents=True, exist_ok=True)

    board: dict[str, dict[str, float]] = {}
    geometry: dict[str, dict[str, object]] = {}
    for pname, pids in PROFILES.items():
        modes = assemble_profile_modes(args.emtf_dir, freqs, station_x, profile_ids=pids)
        geometry[pname] = profile_geometry_metadata(modes)
        lr, ph, mask = (modes["lr_tm"], modes["ph_tm"], modes["mask_tm"])

        def score(section, modes=modes):
            return section_nrms_2d(section, modes, freqs, station_x, x_grid, depth_grid)

        row: dict[str, float] = {}

        cols = []
        for j in range(lr.shape[1]):
            valid = mask[:, j]
            res = occam1d_invert(
                lr[valid, j],
                ph[valid, j],
                periods[valid],
                max_iterations=30,
            )
            cols.append(res.profile_on_grid(depth_grid))
        row["occam1d"] = score(
            columns_to_section(np.stack(cols, axis=1), station_x, x_grid)
        )

        row["neural1d"] = score(
            neural_1d_section(inv1d, lr, ph, mask, periods, station_x, x_grid, depth_grid)
        )

        row["unet-60k"] = score(unet_section(model_60k, modes))
        row["unet-60k-ft-YS"] = score(unet_section(model_60k_ft, modes))

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
        self_model = load_model2d(ft_path, args.test_h5)
        adaptation_lineage[f"unet-60k-ft-self-{pname}"] = require_finetune2d_lineage(
            self_model,
            base=model_60k,
            emtf_dir=args.emtf_dir,
            expected_profiles=[pids],
            expected_options={
                "seed": 0,
                "steps": 600,
                "learning_rate": 2.0e-5,
                "anchor_weight": 3.0,
                "jitter": 0.02,
            },
        )["lineage_sha256"]
        loaded_models[f"unet-60k-ft-self-{pname}"] = self_model
        row["unet-60k-ft-self"] = score(unet_section(self_model, modes))

        board[pname] = row
        print(
            f"{pname:5s} | " + " | ".join(f"{k} {v:.2f}" for k, v in row.items()),
            flush=True,
        )

    # summary: mean over profiles
    methods = list(next(iter(board.values())).keys())
    summary = {m: float(np.mean([board[p][m] for p in board])) for m in methods}
    print("MEAN  | " + " | ".join(f"{k} {v:.2f}" for k, v in summary.items()))
    print("DIAGNOSTIC ONLY: mixed inverse budgets and normalized geometry", flush=True)

    for loaded in loaded_models.values():
        loaded.require_artifacts_unchanged()
    publish_json_no_overwrite(
        {
            "schema_version": 3,
            "comparison_status": "diagnostic_non_comparable",
            "ranking_allowed": False,
            "diagnostic_reasons": [
                "Occam1D and neural1D consume TM-only station curves while the 2D U-Net consumes TE+TM",
                "field profiles are assembled on the synthetic model station grid rather than native geometry",
                "self-adapted rows optimize on the same profile that they score",
            ],
            "metric_id": SECTION_NRMS_METRIC_ID,
            "geometry": geometry,
            "profiles": board,
            "mean": summary,
            "artifacts": {
                label: loaded.artifact_provenance()
                for label, loaded in loaded_models.items()
            },
            "adaptation_lineage_sha256": adaptation_lineage,
        },
        args.out,
    )
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
