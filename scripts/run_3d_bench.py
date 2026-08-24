"""Score a trained PIMSR 3D checkpoint on held-out 3D samples.

Reports physical-volume-weighted model errors and 68% sigma coverage on the
held-out dataset grid. The exact checkpoint byte snapshot used for inference
is hashed before it is decoded, so concurrent checkpoint publication cannot
make the recorded digest disagree with the evaluated model.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pimsr_benchmarks.statistics import bootstrap_ci

RESULT_STATUS = "diagnostic_incomplete"
RESULT_SCOPE = (
    "single-method held-out model-space diagnostic; no external baseline or "
    "common forward-data scoring is executed"
)


def _stat_signature(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return (int(stat.st_dev), int(stat.st_ino), int(stat.st_size), int(stat.st_mtime_ns))


def _load_checkpoint_snapshot(
    path: Path,
    *,
    map_location: torch.device | str,
) -> tuple[object, str, int]:
    """Decode one stable byte snapshot and return its digest and size."""
    before = _stat_signature(path)
    payload = path.read_bytes()
    after = _stat_signature(path)
    if before != after or len(payload) != before[2]:
        raise RuntimeError(
            f"3D checkpoint changed while it was being snapshotted: {path}"
        )
    digest = hashlib.sha256(payload).hexdigest()
    state = torch.load(io.BytesIO(payload), map_location=map_location, weights_only=False)
    return state, digest, len(payload)


def _cell_widths_from_centers(values: object, *, name: str) -> np.ndarray:
    centers = np.asarray(values, dtype=np.float64)
    if (
        centers.ndim != 1
        or centers.size < 2
        or not np.isfinite(centers).all()
        or np.any(np.diff(centers) <= 0.0)
    ):
        raise ValueError(
            f"3D scoring axis {name!r} must contain increasing finite centers"
        )
    edges = np.empty(centers.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    widths = np.diff(edges)
    if not np.isfinite(widths).all() or np.any(widths <= 0.0):
        raise ValueError(f"3D scoring axis {name!r} has invalid midpoint cell widths")
    return widths


def _volume_weights(coordinates: object) -> np.ndarray:
    if not isinstance(coordinates, dict) or not all(
        name in coordinates for name in ("depth", "y", "x")
    ):
        raise ValueError("3D scoring coordinates must provide depth, y and x centers")
    depth = _cell_widths_from_centers(coordinates["depth"], name="depth")
    y_width = _cell_widths_from_centers(coordinates["y"], name="y")
    x_width = _cell_widths_from_centers(coordinates["x"], name="x")
    return depth[:, None, None] * y_width[None, :, None] * x_width[None, None, :]


def _weighted_sample_metrics(
    mean: object,
    sigma: object,
    truth: object,
    weights: object,
) -> dict[str, float]:
    mean_array, sigma_array, truth_array, weight_array = (
        np.asarray(value, dtype=np.float64) for value in (mean, sigma, truth, weights)
    )
    if not (
        mean_array.shape == sigma_array.shape == truth_array.shape == weight_array.shape
    ):
        raise ValueError(
            "3D prediction, uncertainty, truth and volume weights must match"
        )
    if (
        not np.isfinite(mean_array).all()
        or not np.isfinite(sigma_array).all()
        or not np.isfinite(truth_array).all()
        or not np.isfinite(weight_array).all()
        or np.any(sigma_array <= 0.0)
        or np.any(weight_array <= 0.0)
    ):
        raise ValueError(
            "3D scoring arrays must be finite with positive sigma and weights"
        )
    residual = mean_array - truth_array
    total_weight = float(weight_array.sum())
    return {
        "log10_resistivity_rmse": float(
            np.sqrt(np.sum(weight_array * residual**2) / total_weight)
        ),
        "log10_resistivity_mae": float(
            np.sum(weight_array * np.abs(residual)) / total_weight
        ),
        "coverage68": float(
            np.sum(weight_array * (np.abs(residual) <= sigma_array)) / total_weight
        ),
    }


def _publish_json(value: object, path: Path) -> None:
    """Atomically publish a new benchmark result without overwriting a run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing benchmark result: {path}")
    if part.exists():
        raise FileExistsError(f"temporary benchmark result already exists: {part}")
    owned = False
    try:
        with part.open("x", encoding="utf-8", newline="\n") as stream:
            owned = True
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(part, path)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite existing benchmark result: {path}"
            ) from error
        part.unlink()
    except Exception:
        if owned:
            part.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data", required=True, help="directory of held-out sample_*.h5")
    parser.add_argument(
        "--preset",
        default=None,
        help="optional assertion against the preset recorded by the checkpoint",
    )
    parser.add_argument("--out", default="results/3d/bench3d.json")
    args = parser.parse_args()

    from pimsr_inversion.network3d import PimsrNet3D
    from pimsr_inversion.train3d import (
        Volume3DDataset,
        validate_checkpoint3d_for_dataset,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Volume3DDataset(args.data)
    checkpoint_path = Path(args.ckpt).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"3D checkpoint does not exist: {checkpoint_path}")
    state, checkpoint_sha256, checkpoint_size = _load_checkpoint_snapshot(
        checkpoint_path,
        map_location="cpu",
    )
    model_config = validate_checkpoint3d_for_dataset(
        state,
        dataset,
        expected_preset=args.preset,
    )
    model = PimsrNet3D(
        model_config["in_channels"],
        model_config["width"],
        checkpoint_blocks=model_config["checkpoint_blocks"],
    ).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()

    coordinates = dataset.data_contract["coordinates"]
    weights = _volume_weights(coordinates)
    sample_metrics: list[dict[str, object]] = []
    with torch.no_grad():
        for index in range(len(dataset)):
            obs, target = dataset[index]
            obs = obs.unsqueeze(0).to(device)
            pred = model(obs, output_shape=target.shape[-3:])
            mu = pred["log_rho"].squeeze(0).cpu().numpy()
            sigma = np.exp(0.5 * pred["log_sigma_rho"].squeeze(0).cpu().numpy())
            metrics = _weighted_sample_metrics(mu, sigma, target.numpy(), weights)
            sample = dataset.data_contract["samples"][index]
            sample_metrics.append(
                {
                    "identity": sample["identity"],
                    "source_sha256": sample["sha256"],
                    **metrics,
                }
            )

    rmses = [float(item["log10_resistivity_rmse"]) for item in sample_metrics]
    maes = [float(item["log10_resistivity_mae"]) for item in sample_metrics]
    coverages = [float(item["coverage68"]) for item in sample_metrics]
    rmse_summary = bootstrap_ci(rmses)
    mae_summary = bootstrap_ci(maes)
    coverage_summary = bootstrap_ci(coverages)

    summary = {
        "schema": "pimsr-3d-benchmark-result",
        "schema_version": 1,
        "status": RESULT_STATUS,
        "comparison_scope": RESULT_SCOPE,
        "ranking_allowed": False,
        "completion_requirements": [
            "evaluate at least one external 3D baseline on the identical samples",
            "apply a common forward-observation scoring contract to every method",
        ],
        "n_samples": len(dataset),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_size_bytes": checkpoint_size,
        "checkpoint_schema": state["checkpoint_schema"],
        "checkpoint_schema_version": state["checkpoint_schema_version"],
        "checkpoint_epoch": int(state["epoch"]),
        "preset": state["preset"],
        "model_config": model_config,
        "heldout_data_contract": dataset.data_contract,
        "scoring_contract": {
            "mesh": "heldout_dataset_target_centers",
            "prediction_operator": "network_decoder_trilinear_to_heldout_target_shape",
            "cell_volume_rule": "midpoint_edges_from_physical_centers",
            "target": "log10_resistivity_ohm_m",
            "uncertainty": "one_predicted_standard_deviation",
            "sample_aggregation": "unweighted_bootstrap_over_heldout_samples",
            "bootstrap_resamples": 10_000,
            "bootstrap_seed": 0,
        },
        "per_sample": sample_metrics,
        "volume_weighted_log10_resistivity_rmse": rmse_summary,
        "volume_weighted_log10_resistivity_mae": mae_summary,
        "volume_weighted_coverage68": coverage_summary,
        # Compatibility aliases now point to the explicitly physical metrics.
        "rmse": rmse_summary,
        "coverage68": coverage_summary,
    }
    out = Path(args.out)
    _publish_json(summary, out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
