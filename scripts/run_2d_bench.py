"""Evaluate the conv-2D inversion network without making a SOTA claim.

Synthetic: section RMSE + sigma coverage on the 2D test split.
Real: assemble an E-W USArray profile (~44.6N through Yellowstone) into a
pseudo-section and report a legacy 1D-column TM diagnostic.  The real-data
number is not a comparable 2D TE+TM score and is never rankable.

Usage:
    python scripts/run_2d_bench.py --checkpoint best2d.pt \
        --test-h5 ds2d_test.h5 --emtf-dir data/emtf --out-dir results
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from pimsr_benchmarks.hybrid2d import (
    assemble_profile_modes,
    profile_geometry_metadata,
    section_nrms,
)
from pimsr_benchmarks.metrics import summarize
from pimsr_benchmarks.runner2d import (
    file_artifact_provenance,
    load_model2d,
    prepare_profile_observation,
    publish_json_no_overwrite,
    publish_npz_no_overwrite,
    require_file_artifact_unchanged,
    stack_dataset_observations,
)

#: E-W profile at ~44.6N, west to east.
PROFILE_IDS = ["MTH15", "MTH16", "WYYS1", "WYYS2", "WYYS3", "WYH18", "WYH19"]


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
    expected_paths = [str(source["path"]) for source in sources]
    if [str(path.resolve()) for path in current] != expected_paths:
        raise RuntimeError("EMTF XML input set changed during the benchmark")
    for source in sources:
        require_file_artifact_unchanged(source, role="EMTF XML input")


def bench_synthetic(model, ckpt, test_h5: str, n: int) -> dict:
    with h5py.File(test_h5, "r") as f:
        obs = stack_dataset_observations(f, slice(0, n))
        tgt = f["target_log10_res"][:n].astype(np.float32)
        scen = f["scenario"][:n]
    obs = (obs - ckpt["stats_mean"]) / ckpt["stats_std"]

    t0 = time.time()
    with torch.no_grad():
        out = model(torch.from_numpy(obs.astype(np.float32)))
    dt = time.time() - t0

    pred = out["log_rho"].numpy()
    ls = out["log_sigma_rho"].numpy()
    aff = ckpt.get("sigma_affine2d")
    if aff:  # post-hoc affine recalibration fitted on the val split
        ls = aff["a"] * ls + aff["b"]
    sigma = np.exp(0.5 * ls)
    rmses = np.sqrt(((pred - tgt) ** 2).mean(axis=(1, 2)))
    cov1 = float((np.abs(pred - tgt) < sigma).mean())
    acc = float((out["scenario_logits"].argmax(dim=1).numpy() == scen).mean())
    return {
        "schema": "pimsr-2d-synthetic-method-evaluation",
        "schema_version": 1,
        "comparison_status": "single_method_evaluation",
        "ranking_allowed": False,
        "completion_requirements": [
            "run every comparator through the same held-out split and metric contract",
            "publish validated versioned SOTA execution manifests",
        ],
        "method": "conv2d",
        "n": len(rmses),
        "rmse_log10_res": summarize(rmses.tolist()),
        "sigma_coverage_1": cov1,
        "scenario_accuracy": acc,
        "time_per_section_s": dt / len(rmses),
        "per_scenario_rmse": {
            str(s): float(rmses[scen == s].mean()) for s in np.unique(scen)
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test-h5", required=True)
    ap.add_argument("--emtf-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()

    loaded = load_model2d(args.checkpoint, args.test_h5)
    model, ckpt = loaded.model, loaded.checkpoint
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts = loaded.artifact_provenance()
    emtf_sources = _snapshot_emtf_sources(args.emtf_dir)
    syn = bench_synthetic(model, ckpt, args.test_h5, args.n)
    syn["artifacts"] = artifacts
    print("synthetic:", json.dumps(syn["rmse_log10_res"]))
    print("coverage:", syn["sigma_coverage_1"], "| scen acc:", syn["scenario_accuracy"])

    freqs = loaded.contract.frequencies
    station_x = loaded.contract.station_x
    depth_grid = loaded.contract.depth_grid

    real = bench_real_profile(model, ckpt, args.emtf_dir, freqs, station_x, depth_grid)
    array_fields = {"section", "lr_obs", "ph_obs", "period_mask", "x_model"}
    real_public = {key: value for key, value in real.items() if key not in array_fields}
    real_public["artifacts"] = {**artifacts, "emtf_xml": emtf_sources}
    output_paths = (
        out_dir / "conv2d_real_profile.npz",
        out_dir / "conv2d_synthetic.json",
        out_dir / "conv2d_real.json",
    )
    existing = next((path for path in output_paths if path.exists()), None)
    if existing is not None:
        raise FileExistsError(
            f"refusing to overwrite existing benchmark output: {existing}"
        )
    loaded.require_artifacts_unchanged()
    _require_emtf_sources_unchanged(args.emtf_dir, emtf_sources)
    prediction_path = publish_npz_no_overwrite(
        output_paths[0],
        section=real["section"],
        lr_obs=real["lr_obs"],
        ph_obs=real["ph_obs"],
        period_mask=real["period_mask"],
        x_model=real["x_model"],
    )
    real_public["prediction_artifact"] = file_artifact_provenance(prediction_path)
    loaded.require_artifacts_unchanged()
    _require_emtf_sources_unchanged(args.emtf_dir, emtf_sources)
    publish_json_no_overwrite(syn, output_paths[1])
    publish_json_no_overwrite(real_public, output_paths[2])
    print("real profile diagnostic nRMS:", real.get("nrms_mean"))


def bench_real_profile(model, ckpt, emtf_dir, freqs, station_x, depth_grid) -> dict:
    """Invert the USArray profile and physics-check the recovered section."""
    periods = 1.0 / freqs
    modes = assemble_profile_modes(emtf_dir, freqs, station_x, profile_ids=PROFILE_IDS)
    x_model, x_km = modes["x_model"], modes["x_km"]

    obs = prepare_profile_observation(modes, ckpt)
    with torch.no_grad():
        out = model(torch.from_numpy(obs.astype(np.float32)))
    section = out["log_rho"][0].numpy()

    # Legacy 1D-column diagnostic, explicitly referenced to literal TM.
    lr, ph = modes["lr_tm"], modes["ph_tm"]
    nrms_mean, nrms_list = section_nrms(
        section,
        lr,
        ph,
        modes["mask_tm"],
        x_model,
        x_km,
        periods,
        depth_grid,
    )

    return {
        "schema": "pimsr-2d-real-profile-diagnostic",
        "schema_version": 1,
        "comparison_status": "diagnostic_non_comparable",
        "ranking_allowed": False,
        "diagnostic_reasons": [
            "the network consumes TE+TM but this legacy score evaluates only TM",
            "the score re-simulates independent 1D columns rather than a 2D TE+TM forward",
            "the physical field profile is normalized onto the synthetic model station grid",
        ],
        "metric_id": "section_nrms_1d_tm_masked_v2",
        "mode": "TM/Zxy",
        "inverse_observation_budget": ["TE/Zyx", "TM/Zxy"],
        "scoring_observation_budget": ["TM/Zxy"],
        "profile": PROFILE_IDS,
        "geometry": profile_geometry_metadata(modes),
        "nrms_mean": nrms_mean,
        "nrms_per_station": nrms_list,
        "section": section,
        "lr_obs": lr,
        "ph_obs": ph,
        "period_mask": modes["mask_tm"],
        "x_model": x_model,
    }


if __name__ == "__main__":
    main()
