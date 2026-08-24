"""Fail-closed single-method uncertainty diagnostic for a 2D checkpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
from pathlib import Path

import h5py
import numpy as np
import pimsr_inversion.contracts2d as contracts2d_module
import pimsr_inversion.network2d as network2d_module
import torch

import pimsr_benchmarks.runner2d as runner2d_module
import pimsr_benchmarks.statistics as statistics_module
from pimsr_benchmarks.runner2d import (
    checkpoint_adaptation_kind,
    file_artifact_provenance,
    load_model2d,
    publish_json_no_overwrite,
    publish_text_no_overwrite,
    require_file_artifact_unchanged,
    stack_dataset_observations,
)
from pimsr_benchmarks.statistics import bootstrap_ci, calibration_summary


def _inclusive_ranges(values: np.ndarray) -> list[list[int]]:
    indices = np.asarray(values, dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("evaluated sample indices must be a non-empty vector")
    if np.any(np.diff(indices) <= 0):
        raise ValueError("evaluated sample indices must be strictly increasing")
    groups = np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)
    return [[int(group[0]), int(group[-1])] for group in groups]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-h5", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n", type=int, default=0, help="0 means all samples")
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument(
        "--adaptation",
        required=True,
        choices=("zero-shot", "profile-adapted", "regional/joint-adapted"),
    )
    args = parser.parse_args()
    if args.n < 0:
        raise ValueError("--n must be non-negative")
    if not 0 <= args.seed <= np.iinfo(np.uint64).max:
        raise ValueError("--seed must be in the uint64 range")

    out = Path(args.out_dir).resolve()
    json_path = out / "uncertainty2d.json"
    csv_path = out / "coverage_by_depth.csv"
    existing = next((path for path in (json_path, csv_path) if path.exists()), None)
    if existing is not None:
        raise FileExistsError(
            f"refusing to overwrite existing benchmark output: {existing}"
        )

    loaded = load_model2d(args.checkpoint, args.test_h5)
    ckpt, model = loaded.checkpoint, loaded.model
    actual_adaptation = checkpoint_adaptation_kind(ckpt)
    if args.adaptation != actual_adaptation:
        raise ValueError(
            "--adaptation does not match checkpoint lineage: "
            f"declared {args.adaptation!r}, actual {actual_adaptation!r}"
        )
    sl = slice(None if args.n == 0 else args.n)
    with h5py.File(loaded.dataset_path) as f:
        obs = stack_dataset_observations(f, sl)
        target = f["target_log10_res"][sl].astype(np.float32)
        scenario = f["scenario"][sl]
        sample_indices = f["sample_index"][sl].astype(np.int64)
        generator_seed = int(np.asarray(f.attrs["generator_seed"]))
    loaded.require_artifacts_unchanged()
    depth = loaded.contract.depth_grid
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
    calibration = calibration_summary(
        mean, sigma, target, depth_axis=1, scenario=scenario
    )
    canonical_indices = np.asarray(sample_indices, dtype="<i8")
    source_artifacts = {
        "runner_source": file_artifact_provenance(__file__),
        "runner2d_source": file_artifact_provenance(runner2d_module.__file__),
        "statistics_source": file_artifact_provenance(statistics_module.__file__),
        "contracts2d_source": file_artifact_provenance(contracts2d_module.__file__),
        "network2d_source": file_artifact_provenance(network2d_module.__file__),
    }
    output = {
        "schema_version": 3,
        "result_kind": "single_method_synthetic_uncertainty_diagnostic",
        "comparison_status": "diagnostic_non_comparable",
        "ranking_allowed": False,
        "headline_claim_allowed": False,
        "diagnostic_reasons": [
            "single-method uncertainty analysis does not define a cross-method comparison",
            "synthetic geological error is not real-profile geological ground truth",
            "calibration is conditional on one checkpoint and one held-out dataset",
        ],
        "method": "conv2d",
        "adaptation": actual_adaptation,
        "score_interpretation": "synthetic geological error; real-profile nRMS is reported separately as forward consistency",
        "n": len(rmse),
        "rmse": bootstrap_ci(rmse, seed=args.seed),
        "calibration": calibration,
        "checkpoint": str(Path(args.checkpoint).name),
        "test_dataset": str(Path(args.test_h5).name),
        "artifacts": loaded.artifact_provenance(),
        "evaluator_sources": source_artifacts,
        "runtime_environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "h5py": h5py.__version__,
            "hdf5": h5py.version.hdf5_version,
            "torch": str(torch.__version__),
            "device": "cpu",
        },
        "evaluation_contract": {
            "requested_sample_limit": args.n,
            "sample_selection": "all" if args.n == 0 else "ordered_prefix",
            "evaluated_sample_count": len(sample_indices),
            "sample_index_ranges_inclusive": _inclusive_ranges(sample_indices),
            "sample_indices_dtype": "int64-little-endian",
            "sample_indices_sha256": hashlib.sha256(
                canonical_indices.tobytes(order="C")
            ).hexdigest(),
            "generator_seed": generator_seed,
            "bootstrap_seed": args.seed,
            "bootstrap_resamples": 10_000,
            "bootstrap_confidence": 0.95,
            "geometry": "versioned_synthetic_2d_dataset_physical_axes",
            "field_profile_normalization_applied": False,
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    loaded.require_artifacts_unchanged()
    for role, provenance in source_artifacts.items():
        require_file_artifact_unchanged(provenance, role=role)
    csv_payload = io.StringIO(newline="")
    writer = csv.writer(csv_payload)
    writer.writerow(("depth_m", "coverage68"))
    writer.writerows(zip(depth, calibration["coverage68_by_depth"], strict=True))
    publish_text_no_overwrite(csv_payload.getvalue(), csv_path)
    coverage_artifact = file_artifact_provenance(csv_path)
    output["coverage_by_depth_artifact"] = coverage_artifact
    loaded.require_artifacts_unchanged()
    for role, provenance in source_artifacts.items():
        require_file_artifact_unchanged(provenance, role=role)
    require_file_artifact_unchanged(
        coverage_artifact,
        role="coverage-by-depth output",
    )
    publish_json_no_overwrite(output, json_path)
    print(json.dumps(output["rmse"], indent=2))


if __name__ == "__main__":
    main()
