"""Run the non-comparable 2D hybrid diagnostic on the real profile.

The score function is shared, but the inverse observation budgets are not:
the U-Net warm start consumes TE+TM while Gauss-Newton refinement consumes
TM only. Field coordinates are also normalized to the synthetic model span.
The output is therefore diagnostic-only and can never be a leaderboard row.

Usage:
    python scripts/run_2d_hybrid_bench.py --checkpoint best2d.pt \
        --test-h5 ds2d_test.h5 --emtf-dir data/emtf --out-dir results \
        --allow-mixed-budget-diagnostic [--max-iter 5]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from pimsr_benchmarks.hybrid2d import (
    PROFILE_IDS,
    SECTION_NRMS_METRIC_ID,
    assemble_profile_modes,
    profile_geometry_metadata,
    refine_section_2d,
    section_nrms,
    section_nrms_2d,
)
from pimsr_benchmarks.runner2d import (
    file_artifact_provenance,
    load_model2d,
    prepare_profile_observation,
    publish_json_no_overwrite,
    publish_npz_no_overwrite,
    require_file_artifact_unchanged,
)

COMPARISON_STATUS = "diagnostic_non_comparable"
DIAGNOSTIC_REASONS = [
    (
        "inverse budgets differ: the U-Net warm start uses TE+TM while "
        "Gauss-Newton refinement and the cold control use TM only"
    ),
    (
        "field station coordinates are normalized to the synthetic model span "
        "rather than preserved on their native physical scale"
    ),
]
LEGACY_1D_TM_METRIC_ID = "section_nrms_1d_tm_masked_v2"


def _snapshot_emtf_sources(emtf_dir: str | Path) -> list[dict[str, object]]:
    directory = Path(emtf_dir).resolve(strict=True)
    paths = sorted(directory.glob("*.xml"), key=lambda path: path.name.casefold())
    if not paths:
        raise FileNotFoundError(f"no EMTF XML inputs found in {directory}")
    return [file_artifact_provenance(path) for path in paths]


def _require_emtf_sources_unchanged(
    emtf_dir: str | Path,
    sources: list[dict[str, object]],
) -> None:
    directory = Path(emtf_dir).resolve(strict=True)
    current = sorted(directory.glob("*.xml"), key=lambda path: path.name.casefold())
    if [str(path.resolve()) for path in current] != [
        str(source["path"]) for source in sources
    ]:
        raise RuntimeError("EMTF XML input set changed during the diagnostic")
    for source in sources:
        require_file_artifact_unchanged(source, role="EMTF XML input")


def _method_metrics(
    nrms_2d: float,
    nrms_1d_tm: float,
    per_station_1d_tm: list[float],
) -> dict[str, object]:
    return {
        SECTION_NRMS_METRIC_ID: nrms_2d,
        LEGACY_1D_TM_METRIC_ID: {
            "mean": nrms_1d_tm,
            "per_station": per_station_1d_tm,
        },
    }


def _diagnostic_contract(
    geometry: dict[str, object], *, include_cold: bool
) -> dict[str, object]:
    if geometry.get("publishable_physical_geometry") is not False:
        raise ValueError(
            "this diagnostic driver requires explicitly non-publishable "
            "normalized geometry"
        )
    budgets: dict[str, object] = {
        "unet": ["te", "tm"],
        "hybrid": {
            "warm_start": ["te", "tm"],
            "gauss_newton_refinement": ["tm"],
        },
    }
    if include_cold:
        budgets["cold"] = ["tm"]
    return {
        "comparison_status": COMPARISON_STATUS,
        "ranking_allowed": False,
        "headline_claim_allowed": False,
        "diagnostic_reasons": DIAGNOSTIC_REASONS.copy(),
        "score_observation_modes": ["te", "tm"],
        "inverse_observation_budget": budgets,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test-h5", required=True)
    ap.add_argument("--emtf-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-iter", type=int, default=5)
    ap.add_argument("--skip-cold", action="store_true")
    ap.add_argument(
        "--allow-mixed-budget-diagnostic",
        action="store_true",
        help="write a non-rankable normalized-geometry diagnostic",
    )
    args = ap.parse_args(argv)
    if not args.allow_mixed_budget_diagnostic:
        ap.error(
            "this driver mixes TE+TM and TM-only inverse budgets on normalized "
            "geometry; pass --allow-mixed-budget-diagnostic only to write an "
            "explicitly non-comparable artifact"
        )
    if args.max_iter < 1:
        ap.error("--max-iter must be positive")

    out_dir = Path(args.out_dir).resolve()
    output_paths = (
        out_dir / "hybrid2d_sections.npz",
        out_dir / "hybrid2d_real.json",
    )
    existing = next((path for path in output_paths if path.exists()), None)
    if existing is not None:
        raise FileExistsError(
            f"refusing to overwrite existing benchmark output: {existing}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_model2d(args.checkpoint, args.test_h5)
    freqs = loaded.contract.frequencies
    station_x = loaded.contract.station_x
    x_grid = loaded.contract.x_grid
    depth_grid = loaded.contract.depth_grid
    periods = 1.0 / freqs

    emtf_sources = _snapshot_emtf_sources(args.emtf_dir)
    modes = assemble_profile_modes(args.emtf_dir, freqs, station_x)
    lr, ph, period_mask = (modes["lr_tm"], modes["ph_tm"], modes["mask_tm"])
    x_model, x_km = modes["x_model"], modes["x_km"]

    # ---- row 1: U-Net single pass -------------------------------------
    model, ckpt = loaded.model, loaded.checkpoint
    obs = prepare_profile_observation(modes, ckpt)
    network_started = perf_counter()
    with torch.no_grad():
        out = model(torch.from_numpy(obs.astype(np.float32)))
    network_wall_time_s = perf_counter() - network_started
    net_section = out["log_rho"][0].numpy()
    net_nrms, net_list = section_nrms(
        net_section, lr, ph, period_mask, x_model, x_km, periods, depth_grid
    )
    net_2d = section_nrms_2d(net_section, modes, freqs, station_x, x_grid, depth_grid)
    print(
        f"unet single pass    | 1D-col nRMS {net_nrms:.2f} | 2D nRMS {net_2d:.2f}",
        flush=True,
    )

    # ---- row 2: hybrid = warm start + GN -------------------------------
    hy = refine_section_2d(
        net_section,
        modes,
        freqs,
        station_x,
        x_grid,
        depth_grid,
        mode="tm",
        max_iter=args.max_iter,
    )
    hy_nrms, hy_list = section_nrms(
        hy.section, lr, ph, period_mask, x_model, x_km, periods, depth_grid
    )
    hy_2d = section_nrms_2d(hy.section, modes, freqs, station_x, x_grid, depth_grid)
    print(
        f"hybrid (net + GN{args.max_iter}) | 1D-col nRMS {hy_nrms:.2f} "
        f"| 2D nRMS {hy_2d:.2f} | {hy.wall_time_s:.0f} s",
        flush=True,
    )

    geometry = profile_geometry_metadata(modes)
    results = {
        "schema": "pimsr-2d-hybrid-diagnostic",
        "schema_version": 3,
        "metric_id": SECTION_NRMS_METRIC_ID,
        "supplementary_metric_id": LEGACY_1D_TM_METRIC_ID,
        "geometry": geometry,
        "profile": PROFILE_IDS,
        "max_iter": args.max_iter,
        "artifacts": {
            **loaded.artifact_provenance(),
            "emtf_xml": emtf_sources,
        },
        "unet": {
            "metrics": _method_metrics(net_2d, net_nrms, net_list),
            "wall_time_s": network_wall_time_s,
        },
        "hybrid": {
            "metrics": _method_metrics(hy_2d, hy_nrms, hy_list),
            "warm_start_wall_time_s": network_wall_time_s,
            "refinement_wall_time_s": hy.wall_time_s,
            "wall_time_s": network_wall_time_s + hy.wall_time_s,
            "n_refinement_iterations": hy.n_iterations,
        },
        **_diagnostic_contract(geometry, include_cold=not args.skip_cold),
    }

    # ---- row 3: cold-start control -------------------------------------
    if not args.skip_cold:
        cold0 = np.full_like(net_section, 2.0)  # 100 ohm-m half-space
        cold = refine_section_2d(
            cold0,
            modes,
            freqs,
            station_x,
            x_grid,
            depth_grid,
            mode="tm",
            max_iter=args.max_iter,
        )
        cold_nrms, cold_list = section_nrms(
            cold.section,
            lr,
            ph,
            period_mask,
            x_model,
            x_km,
            periods,
            depth_grid,
        )
        cold_2d = section_nrms_2d(
            cold.section, modes, freqs, station_x, x_grid, depth_grid
        )
        print(
            f"cold GN{args.max_iter} control   | 1D-col nRMS {cold_nrms:.2f} "
            f"| 2D nRMS {cold_2d:.2f} | {cold.wall_time_s:.0f} s",
            flush=True,
        )
        results["cold"] = {
            "metrics": _method_metrics(cold_2d, cold_nrms, cold_list),
            "wall_time_s": cold.wall_time_s,
            "n_refinement_iterations": cold.n_iterations,
        }
        section_arrays = {
            "unet": net_section,
            "hybrid": hy.section,
            "cold": cold.section,
        }
    else:
        section_arrays = {"unet": net_section, "hybrid": hy.section}

    loaded.require_artifacts_unchanged()
    _require_emtf_sources_unchanged(args.emtf_dir, emtf_sources)
    section_path = publish_npz_no_overwrite(output_paths[0], **section_arrays)
    results["section_artifact"] = file_artifact_provenance(section_path)
    loaded.require_artifacts_unchanged()
    _require_emtf_sources_unchanged(args.emtf_dir, emtf_sources)
    publish_json_no_overwrite(results, output_paths[1])
    print("saved to", out_dir, flush=True)


if __name__ == "__main__":
    main()
