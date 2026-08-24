"""Diagnostic mixed-budget board scored with the same 2D forward.

Motivation (see REPORT.md "2D hybrid experiment"): per-station 1D scoring is
biased and incomparable across method families. Here every method produces a
full (nz, nx) section, and all sections are scored with the same
shift-invariant 2D-forward misfit (``section_nrms_2d``) on the real
Yellowstone profile — plus optionally on denser station profiles.

The inverse observation budget is not common: legacy stitched and cold-GN
methods use TM only, the strict U-Net uses TE+TM, and the hybrid 2D path uses
a TE+TM warm start before TM-only refinement. The command therefore fails
closed unless the caller explicitly requests a non-comparable diagnostic.

1D per-station methods are converted to sections by inverting each station
column independently and interpolating laterally between stations, which is
exactly how 1D results are used in practice (stitched sections).
"""

from __future__ import annotations

import argparse
import time
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from pimsr_benchmarks.hybrid import hybrid_invert
from pimsr_benchmarks.hybrid2d import (
    PROFILES,
    SECTION_NRMS_METRIC_ID,
    assemble_profile_modes,
    profile_geometry_metadata,
    refine_section_2d,
    section_nrms_2d,
)
from pimsr_benchmarks.neural import NeuralInverter
from pimsr_benchmarks.occam1d import occam1d_invert
from pimsr_benchmarks.runner2d import (
    LoadedModel2D,
    file_artifact_provenance,
    interpolate_periods_in_band,
    load_model2d,
    prepare_profile_observation,
    publish_json_no_overwrite,
    require_file_artifact_unchanged,
    require_finetune2d_lineage,
)

SCHEMA = "pimsr-unified-2d-diagnostic"
SCHEMA_VERSION = 4


def _xml_paths(emtf_dir: str | Path) -> tuple[Path, list[Path]]:
    root = Path(emtf_dir).resolve(strict=True)
    paths = sorted(path.resolve(strict=True) for path in root.glob("*.xml"))
    if not paths:
        raise ValueError(f"no EMTF XML inputs found in {root}")
    return root, paths


def _snapshot_xml_inputs(
    emtf_dir: str | Path,
) -> tuple[Path, list[dict[str, object]]]:
    root, paths = _xml_paths(emtf_dir)
    return root, [file_artifact_provenance(path) for path in paths]


def _require_xml_inputs_unchanged(
    root: Path,
    provenance: Sequence[Mapping[str, object]],
) -> None:
    _, current_paths = _xml_paths(root)
    expected_paths = [Path(str(item["path"])) for item in provenance]
    if current_paths != expected_paths:
        raise RuntimeError(f"EMTF XML input set changed after snapshot: {root}")
    for item in provenance:
        require_file_artifact_unchanged(item, role="EMTF XML input")


def columns_to_section(
    cols: np.ndarray, station_x: np.ndarray, x_grid: np.ndarray
) -> np.ndarray:
    """Laterally interpolate per-station columns (nz, n_st) onto x_grid."""
    nz = cols.shape[0]
    return np.stack([np.interp(x_grid, station_x, cols[i]) for i in range(nz)])


def occam_section(lr, ph, mask, periods, station_x, x_grid, depth_grid):
    cols = []
    for j in range(lr.shape[1]):
        valid = mask[:, j]
        res = occam1d_invert(
            lr[valid, j], ph[valid, j], periods[valid], max_iterations=30
        )
        cols.append(res.profile_on_grid(depth_grid))
    return columns_to_section(np.stack(cols, axis=1), station_x, x_grid)


def _neural_grid_input(inv, lr, ph, mask, periods):
    """Interpolate valid samples and mean-fill unsupported model periods."""
    raw_mean = inv.stats.obs_mean
    n = inv.n_periods
    out_lr = interpolate_periods_in_band(periods, lr, mask, inv.periods, raw_mean[:n])
    out_ph = interpolate_periods_in_band(
        periods, ph, mask, inv.periods, raw_mean[n : 2 * n] * 45.0
    )
    return out_lr, out_ph


def neural_1d_section(inv, lr, ph, mask, periods, station_x, x_grid, depth_grid):
    cols = []
    for j in range(lr.shape[1]):
        n_lr, n_ph = _neural_grid_input(inv, lr[:, j], ph[:, j], mask[:, j], periods)
        pred = inv.invert(n_lr, n_ph)
        cols.append(np.interp(depth_grid, inv.depth_grid, pred.log10_rho))
    return columns_to_section(np.stack(cols, axis=1), station_x, x_grid)


def hybrid_1d_section(inv, lr, ph, mask, periods, station_x, x_grid, depth_grid):
    cols = []
    for j in range(lr.shape[1]):
        valid = mask[:, j]
        n_lr, n_ph = _neural_grid_input(inv, lr[:, j], ph[:, j], valid, periods)
        res = hybrid_invert(
            inv,
            lr[valid, j],
            ph[valid, j],
            periods[valid],
            neural_log_rho_a=n_lr,
            neural_phase=n_ph,
        )
        cols.append(res.occam.profile_on_grid(depth_grid))
    return columns_to_section(np.stack(cols, axis=1), station_x, x_grid)


def unet_section(
    loaded: LoadedModel2D,
    modes,
    profile_name=None,
):
    """Single strict four-channel U-Net pass on profile-frame TE+TM."""
    import torch

    model, ckpt = loaded.model, loaded.checkpoint
    obs = prepare_profile_observation(modes, ckpt)
    film = None
    adapters = ckpt.get("film_adapters")
    if adapters and profile_name in adapters:
        a = adapters[profile_name]
        film = (a["gamma"].float(), a["beta"].float())
    with torch.no_grad():
        out = model(torch.from_numpy(obs.astype(np.float32)), film=film)
    return out["log_rho"][0].numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-h5", required=True)
    ap.add_argument("--emtf-dir", default="data/emtf")
    ap.add_argument("--ckpt-1d", required=True)
    ap.add_argument("--ckpt-10k", required=True)
    ap.add_argument("--ckpt-10k-ft", required=True)
    ap.add_argument("--ckpt-60k", required=True)
    ap.add_argument("--ckpt-60k-ft", required=True)
    ap.add_argument("--ckpt-v3", default=None, help="TE+TM 4-channel checkpoint")
    ap.add_argument("--ckpt-v3-ft", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-gn", action="store_true", help="skip slow 2D GN rows")
    ap.add_argument(
        "--allow-mixed-budget-diagnostic",
        action="store_true",
        help="write a non-rankable diagnostic with TM-only and TE+TM methods",
    )
    args = ap.parse_args()
    if args.ckpt_v3_ft and not args.ckpt_v3:
        ap.error("--ckpt-v3-ft requires --ckpt-v3")
    if not args.allow_mixed_budget_diagnostic:
        ap.error(
            "this driver mixes TM-only and TE+TM inversion budgets and cannot "
            "produce a fair leaderboard; pass --allow-mixed-budget-diagnostic "
            "only to write an explicitly non-comparable artifact"
        )

    checkpoint_1d = file_artifact_provenance(args.ckpt_1d)
    emtf_root, emtf_xml = _snapshot_xml_inputs(args.emtf_dir)

    loaded_models = {
        "unet-10k": load_model2d(args.ckpt_10k, args.test_h5),
        "unet-10k-ft": load_model2d(args.ckpt_10k_ft, args.test_h5),
        "unet-60k": load_model2d(args.ckpt_60k, args.test_h5),
        "unet-60k-ft": load_model2d(args.ckpt_60k_ft, args.test_h5),
    }
    if args.ckpt_v3:
        loaded_models["unet-v3-tetm"] = load_model2d(args.ckpt_v3, args.test_h5)
    if args.ckpt_v3_ft:
        loaded_models["unet-v3-tetm-ft"] = load_model2d(args.ckpt_v3_ft, args.test_h5)

    adaptation_lineage: dict[str, object] = {}
    fine_tuned_pairs = [
        ("unet-10k-ft", "unet-10k"),
        ("unet-60k-ft", "unet-60k"),
    ]
    if args.ckpt_v3_ft:
        fine_tuned_pairs.append(("unet-v3-tetm-ft", "unet-v3-tetm"))
    for adapted_name, base_name in fine_tuned_pairs:
        lineage = require_finetune2d_lineage(
            loaded_models[adapted_name],
            base=loaded_models[base_name],
            emtf_dir=emtf_root,
            expected_profiles=[PROFILES["H-YS"]],
        )
        adaptation_lineage[adapted_name] = lineage["lineage_sha256"]

    contract = loaded_models["unet-60k"].contract
    freqs = contract.frequencies
    station_x = contract.station_x
    x_grid = contract.x_grid
    depth_grid = contract.depth_grid
    periods = 1.0 / freqs

    modes = assemble_profile_modes(str(emtf_root), freqs, station_x)
    # Legacy 1D baselines and the GN objective consume literal TM. Strict
    # U-Nets, including the hybrid warm start, consume all four TE+TM channels.
    lr, ph, mask = modes["lr_tm"], modes["ph_tm"], modes["mask_tm"]

    def score(section):
        return section_nrms_2d(section, modes, freqs, station_x, x_grid, depth_grid)

    board: dict[str, object] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "metric_id": SECTION_NRMS_METRIC_ID,
        "geometry": profile_geometry_metadata(modes),
        "comparison_status": "diagnostic_non_comparable",
        "ranking_allowed": False,
        "headline_claim_allowed": False,
        "diagnostic_reasons": [
            (
                "stitched 1D and cold-GN rows invert TM only, U-Net rows invert "
                "TE+TM, and hybrid2d uses a TE+TM warm start followed by TM-only "
                "refinement"
            ),
            "all sections are scored on TE+TM even when their inversion used TM only",
            (
                "field observations are mapped onto normalized synthetic geometry "
                "rather than preserved native physical geometry"
            ),
            "fine-tuned rows adapt on the same H-YS profile used for this diagnostic",
        ],
        "score_observation_modes": ["te", "tm"],
        "artifacts": {
            name: loaded.artifact_provenance() for name, loaded in loaded_models.items()
        },
        "provenance": {
            "checkpoint_1d": checkpoint_1d,
            "emtf_xml": emtf_xml,
        },
        "adaptation_lineage_sha256": adaptation_lineage,
        "methods": {},
    }
    methods = board["methods"]

    def add(
        name,
        section,
        *,
        inversion_modes,
        wall=None,
        inversion_stages=None,
        timing=None,
    ):
        entry = {
            "nrms_2d": score(section),
            "inversion_observation_modes": list(inversion_modes),
        }
        if wall is not None:
            entry["wall_time_s"] = wall
        if inversion_stages is not None:
            entry["inversion_observation_stages"] = {
                stage: list(stage_modes)
                for stage, stage_modes in inversion_stages.items()
            }
        if timing is not None:
            entry["timing"] = dict(timing)
        methods[name] = entry
        print(f"{name:28s} | 2D nRMS {entry['nrms_2d']:.2f}", flush=True)

    # ---- 1D family ------------------------------------------------------
    t0 = time.perf_counter()
    add(
        "occam1d-stitched",
        occam_section(lr, ph, mask, periods, station_x, x_grid, depth_grid),
        inversion_modes=("tm",),
        wall=time.perf_counter() - t0,
    )

    inv = NeuralInverter(str(checkpoint_1d["path"]))
    require_file_artifact_unchanged(checkpoint_1d, role="1D checkpoint")
    t0 = time.perf_counter()
    add(
        "neural1d-stitched",
        neural_1d_section(inv, lr, ph, mask, periods, station_x, x_grid, depth_grid),
        inversion_modes=("tm",),
        wall=time.perf_counter() - t0,
    )

    t0 = time.perf_counter()
    add(
        "hybrid1d-stitched",
        hybrid_1d_section(inv, lr, ph, mask, periods, station_x, x_grid, depth_grid),
        inversion_modes=("tm",),
        wall=time.perf_counter() - t0,
    )

    # ---- 2D neural family -----------------------------------------------
    for name in ("unet-10k", "unet-10k-ft", "unet-60k", "unet-60k-ft"):
        add(
            name,
            unet_section(loaded_models[name], modes),
            inversion_modes=("te", "tm"),
        )

    if args.ckpt_v3:
        add(
            "unet-v3-tetm",
            unet_section(loaded_models["unet-v3-tetm"], modes),
            inversion_modes=("te", "tm"),
        )
        if args.ckpt_v3_ft:
            add(
                "unet-v3-tetm-ft",
                unet_section(loaded_models["unet-v3-tetm-ft"], modes),
                inversion_modes=("te", "tm"),
            )

    # ---- 2D iterative ----------------------------------------------------
    if not args.skip_gn:
        hybrid_t0 = time.perf_counter()
        warm = unet_section(loaded_models["unet-60k"], modes)
        warm_wall_time = time.perf_counter() - hybrid_t0
        hy = refine_section_2d(
            warm,
            modes,
            freqs,
            station_x,
            x_grid,
            depth_grid,
            mode="tm",
            max_iter=8,
            beta0_ratio=10.0,
            alpha_ref=1e-2,
        )
        add(
            "hybrid2d-gn8",
            hy.section,
            inversion_modes=("te", "tm"),
            wall=time.perf_counter() - hybrid_t0,
            inversion_stages={
                "warm_start": ("te", "tm"),
                "refinement": ("tm",),
            },
            timing={
                "scope": "warm_start_plus_refinement",
                "warm_start_wall_time_s": warm_wall_time,
                "refinement_wall_time_s": hy.wall_time_s,
            },
        )

        cold = refine_section_2d(
            np.full_like(warm, 2.0),
            modes,
            freqs,
            station_x,
            x_grid,
            depth_grid,
            mode="tm",
            max_iter=25,
            beta0_ratio=10.0,
            alpha_ref=1e-2,
        )
        add(
            "cold-gn25",
            cold.section,
            inversion_modes=("tm",),
            wall=cold.wall_time_s,
        )

    require_file_artifact_unchanged(checkpoint_1d, role="1D checkpoint")
    _require_xml_inputs_unchanged(emtf_root, emtf_xml)
    for loaded in loaded_models.values():
        loaded.require_artifacts_unchanged()
    publish_json_no_overwrite(board, args.out)
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
