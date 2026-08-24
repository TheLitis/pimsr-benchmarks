"""Strict, training-independent evaluation of 2-D SOTA predictions.

The evaluator deliberately accepts one prediction grid only: predictions must
already be expressed on the withheld truth grid.  Resampling belongs in a
method adapter and must therefore be captured before the prediction artifact is
hashed.  Samples are paired by their integer ``sample_index`` rather than by
archive row order.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import stat
import subprocess
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

TRUTH_SCHEMA = "pimsr-sota-2d-truth"
PREDICTION_SCHEMA = "pimsr-sota-2d-predictions"
EVALUATION_SCHEMA = "pimsr-sota-2d-evaluation"
OBSERVATION_SCHEMA_VERSION = 1
TRUTH_SCHEMA_VERSION = 2
PREDICTION_SCHEMA_VERSION = 2
EVALUATION_SCHEMA_VERSION = 2
# Backwards-compatible public alias for callers that construct evaluator
# fixtures.  Artifact-specific code below never relies on this alias.
SCHEMA_VERSION = EVALUATION_SCHEMA_VERSION

_TRUTH_KEY_ORDER = (
    "schema",
    "schema_version",
    "sample_index",
    "observations_sha256",
    "scenario",
    "has_fault",
    "x_cell_centers_m",
    "depth_cell_centers_m",
    "truth_log10_resistivity",
)
_TRUTH_KEYS = frozenset(_TRUTH_KEY_ORDER)
_PREDICTION_KEY_ORDER = (
    "schema",
    "schema_version",
    "observations_sha256",
    "sample_index",
    "x_cell_centers_m",
    "depth_cell_centers_m",
    "predicted_log10_resistivity",
)
_PREDICTION_KEYS = frozenset(_PREDICTION_KEY_ORDER)
_SHA256_LENGTH = 64


class Evaluation2DValidationError(ValueError):
    """Raised when an input artifact violates the evaluator contract."""


class Evaluation2DPublicationError(RuntimeError):
    """Raised when an immutable evaluation report cannot be published."""


@dataclass(frozen=True)
class _ArtifactSnapshot:
    path: Path
    payload: bytes
    sha256: str

    @property
    def size_bytes(self) -> int:
        return len(self.payload)


@dataclass(frozen=True)
class Truth2D:
    sample_index: np.ndarray
    observations_sha256: str
    scenario: np.ndarray
    has_fault: np.ndarray
    x_cell_centers_m: np.ndarray
    depth_cell_centers_m: np.ndarray
    log10_resistivity: np.ndarray
    artifact_sha256: str
    artifact_size_bytes: int


@dataclass(frozen=True)
class Predictions2D:
    sample_index: np.ndarray
    observations_sha256: str
    x_cell_centers_m: np.ndarray
    depth_cell_centers_m: np.ndarray
    log10_resistivity: np.ndarray
    artifact_sha256: str
    artifact_size_bytes: int


def _validated_expected_sha256(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Evaluation2DValidationError(
            f"{name} must be 64 lowercase hexadecimal characters"
        )
    return value


def _read_snapshot(
    path: str | Path,
    *,
    expected_sha256: str | None,
    artifact_name: str,
) -> _ArtifactSnapshot:
    source = Path(path)
    expected = _validated_expected_sha256(
        expected_sha256, f"expected_{artifact_name}_sha256"
    )
    if source.is_symlink():
        raise Evaluation2DValidationError(
            f"{artifact_name} artifact must not be a symbolic link"
        )
    try:
        info = source.stat()
    except OSError as exc:
        raise Evaluation2DValidationError(
            f"cannot stat {artifact_name} artifact {source}: {exc}"
        ) from exc
    if not stat.S_ISREG(info.st_mode):
        raise Evaluation2DValidationError(
            f"{artifact_name} artifact must be a regular file"
        )
    try:
        payload = source.read_bytes()
    except OSError as exc:
        raise Evaluation2DValidationError(
            f"cannot read {artifact_name} artifact {source}: {exc}"
        ) from exc
    digest = hashlib.sha256(payload).hexdigest()
    if expected is not None and digest != expected:
        raise Evaluation2DValidationError(
            f"{artifact_name} artifact SHA-256 does not match the pinned digest"
        )
    return _ArtifactSnapshot(source, payload, digest)


def _validate_npz_members(
    snapshot: _ArtifactSnapshot,
    *,
    expected_keys: frozenset[str],
    expected_order: Sequence[str],
    artifact_name: str,
) -> None:
    expected_members = {f"{key}.npy" for key in expected_keys}
    try:
        with zipfile.ZipFile(io.BytesIO(snapshot.payload), "r") as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if len(names) != len(set(names)):
                raise Evaluation2DValidationError(
                    f"{artifact_name} NPZ contains duplicate archive members"
                )
            if set(names) != expected_members:
                missing = sorted(expected_members - set(names))
                extra = sorted(set(names) - expected_members)
                raise Evaluation2DValidationError(
                    f"{artifact_name} NPZ members mismatch; "
                    f"missing={missing}, extra={extra}"
                )
            canonical_names = [f"{key}.npy" for key in expected_order]
            if names != canonical_names:
                raise Evaluation2DValidationError(
                    f"{artifact_name} NPZ members are not in canonical order"
                )
            if any(member.flag_bits & 0x1 for member in members):
                raise Evaluation2DValidationError(
                    f"{artifact_name} NPZ must not contain encrypted members"
                )
            if any(member.compress_type != zipfile.ZIP_STORED for member in members):
                raise Evaluation2DValidationError(
                    f"{artifact_name} NPZ members must use ZIP_STORED"
                )
            bad_member = archive.testzip()
            if bad_member is not None:
                raise Evaluation2DValidationError(
                    f"{artifact_name} NPZ has a corrupt member {bad_member!r}"
                )
    except Evaluation2DValidationError:
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise Evaluation2DValidationError(
            f"cannot decode {artifact_name} artifact as NPZ: {exc}"
        ) from exc


def _load_npz_arrays(
    snapshot: _ArtifactSnapshot,
    *,
    expected_keys: frozenset[str],
    expected_order: Sequence[str],
    artifact_name: str,
) -> dict[str, np.ndarray]:
    _validate_npz_members(
        snapshot,
        expected_keys=expected_keys,
        expected_order=expected_order,
        artifact_name=artifact_name,
    )
    try:
        with np.load(io.BytesIO(snapshot.payload), allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in sorted(expected_keys)}
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise Evaluation2DValidationError(
            f"cannot load arrays from {artifact_name} NPZ: {exc}"
        ) from exc
    return arrays


def _require_scalar_schema(
    array: np.ndarray, *, expected: str, artifact_name: str
) -> None:
    if array.ndim != 0 or array.dtype.kind != "U" or array.item() != expected:
        raise Evaluation2DValidationError(
            f"{artifact_name}.schema must be scalar Unicode {expected!r}"
        )


def _require_version(
    array: np.ndarray, artifact_name: str, expected_version: int
) -> None:
    if (
        array.ndim != 0
        or array.dtype != np.dtype("<i8")
        or int(array.item()) != expected_version
    ):
        raise Evaluation2DValidationError(
            f"{artifact_name}.schema_version must be scalar int64 "
            f"equal to {expected_version}"
        )


def _require_sha256_scalar(array: np.ndarray, name: str) -> str:
    if array.ndim != 0 or array.dtype != np.dtype("<U64"):
        raise Evaluation2DValidationError(
            f"{name} must be a scalar Unicode[64] SHA-256 digest"
        )
    value = str(array.item())
    validated = _validated_expected_sha256(value, name)
    assert validated is not None
    return validated


def _require_array(
    array: np.ndarray,
    *,
    name: str,
    dtype: str | np.dtype[Any],
    ndim: int,
) -> np.ndarray:
    expected_dtype = np.dtype(dtype)
    if array.dtype != expected_dtype:
        raise Evaluation2DValidationError(
            f"{name} must have dtype {expected_dtype}, got {array.dtype}"
        )
    if array.ndim != ndim:
        raise Evaluation2DValidationError(
            f"{name} must be {ndim}-dimensional, got shape {array.shape}"
        )
    return array


def _validate_sample_indices(array: np.ndarray, name: str) -> np.ndarray:
    values = _require_array(array, name=name, dtype="<i8", ndim=1)
    if values.size == 0:
        raise Evaluation2DValidationError(f"{name} must not be empty")
    if np.any(values < 0):
        raise Evaluation2DValidationError(f"{name} must contain non-negative ids")
    if np.unique(values).size != values.size:
        raise Evaluation2DValidationError(f"{name} contains duplicate ids")
    return values


def _validate_centers(array: np.ndarray, name: str) -> np.ndarray:
    values = _require_array(array, name=name, dtype="<f8", ndim=1)
    if values.size < 2:
        raise Evaluation2DValidationError(
            f"{name} must contain at least two cell centers"
        )
    if not np.isfinite(values).all():
        raise Evaluation2DValidationError(f"{name} must be finite")
    differences = np.diff(values)
    if not np.isfinite(differences).all() or np.any(differences <= 0):
        raise Evaluation2DValidationError(f"{name} must be strictly increasing")
    return values


def load_truth_2d(
    path: str | Path, *, expected_sha256: str | None = None
) -> Truth2D:
    """Load and strictly validate a withheld 2-D truth NPZ artifact."""
    snapshot = _read_snapshot(
        path,
        expected_sha256=expected_sha256,
        artifact_name="truth",
    )
    arrays = _load_npz_arrays(
        snapshot,
        expected_keys=_TRUTH_KEYS,
        expected_order=_TRUTH_KEY_ORDER,
        artifact_name="truth",
    )
    _require_scalar_schema(
        arrays["schema"], expected=TRUTH_SCHEMA, artifact_name="truth"
    )
    _require_version(arrays["schema_version"], "truth", TRUTH_SCHEMA_VERSION)
    sample_index = _validate_sample_indices(
        arrays["sample_index"], "truth.sample_index"
    )
    observations_sha256 = _require_sha256_scalar(
        arrays["observations_sha256"], "truth.observations_sha256"
    )
    scenario = arrays["scenario"]
    if scenario.ndim != 1 or scenario.dtype.kind != "U":
        raise Evaluation2DValidationError(
            "truth.scenario must be a one-dimensional Unicode array"
        )
    if scenario.shape != sample_index.shape:
        raise Evaluation2DValidationError(
            "truth.scenario length must match truth.sample_index"
        )
    scenario_values = scenario.tolist()
    if any(not value or "\x00" in value for value in scenario_values):
        raise Evaluation2DValidationError(
            "truth.scenario values must be non-empty and contain no NUL"
        )
    has_fault = _require_array(
        arrays["has_fault"], name="truth.has_fault", dtype=np.bool_, ndim=1
    )
    if has_fault.shape != sample_index.shape:
        raise Evaluation2DValidationError(
            "truth.has_fault length must match truth.sample_index"
        )
    x_centers = _validate_centers(
        arrays["x_cell_centers_m"], "truth.x_cell_centers_m"
    )
    depth_centers = _validate_centers(
        arrays["depth_cell_centers_m"], "truth.depth_cell_centers_m"
    )
    truth = _require_array(
        arrays["truth_log10_resistivity"],
        name="truth.truth_log10_resistivity",
        dtype="<f4",
        ndim=3,
    )
    expected_shape = (sample_index.size, depth_centers.size, x_centers.size)
    if truth.shape != expected_shape:
        raise Evaluation2DValidationError(
            "truth.truth_log10_resistivity shape must be "
            f"{expected_shape}, got {truth.shape}"
        )
    if not np.isfinite(truth).all():
        raise Evaluation2DValidationError(
            "truth.truth_log10_resistivity must be finite"
        )
    return Truth2D(
        sample_index=sample_index,
        observations_sha256=observations_sha256,
        scenario=scenario,
        has_fault=has_fault,
        x_cell_centers_m=x_centers,
        depth_cell_centers_m=depth_centers,
        log10_resistivity=truth,
        artifact_sha256=snapshot.sha256,
        artifact_size_bytes=snapshot.size_bytes,
    )


def load_predictions_2d(
    path: str | Path, *, expected_sha256: str | None = None
) -> Predictions2D:
    """Load and strictly validate a minimal 2-D prediction NPZ artifact."""
    snapshot = _read_snapshot(
        path,
        expected_sha256=expected_sha256,
        artifact_name="prediction",
    )
    arrays = _load_npz_arrays(
        snapshot,
        expected_keys=_PREDICTION_KEYS,
        expected_order=_PREDICTION_KEY_ORDER,
        artifact_name="prediction",
    )
    _require_scalar_schema(
        arrays["schema"], expected=PREDICTION_SCHEMA, artifact_name="prediction"
    )
    _require_version(
        arrays["schema_version"], "prediction", PREDICTION_SCHEMA_VERSION
    )
    sample_index = _validate_sample_indices(
        arrays["sample_index"], "prediction.sample_index"
    )
    observations_sha256 = _require_sha256_scalar(
        arrays["observations_sha256"], "prediction.observations_sha256"
    )
    x_centers = _validate_centers(
        arrays["x_cell_centers_m"], "prediction.x_cell_centers_m"
    )
    depth_centers = _validate_centers(
        arrays["depth_cell_centers_m"], "prediction.depth_cell_centers_m"
    )
    prediction = _require_array(
        arrays["predicted_log10_resistivity"],
        name="prediction.predicted_log10_resistivity",
        dtype="<f4",
        ndim=3,
    )
    expected_shape = (sample_index.size, depth_centers.size, x_centers.size)
    if prediction.shape != expected_shape:
        raise Evaluation2DValidationError(
            "prediction.predicted_log10_resistivity shape must be "
            f"{expected_shape}, got {prediction.shape}"
        )
    if not np.isfinite(prediction).all():
        raise Evaluation2DValidationError(
            "prediction.predicted_log10_resistivity must be finite"
        )
    return Predictions2D(
        sample_index=sample_index,
        observations_sha256=observations_sha256,
        x_cell_centers_m=x_centers,
        depth_cell_centers_m=depth_centers,
        log10_resistivity=prediction,
        artifact_sha256=snapshot.sha256,
        artifact_size_bytes=snapshot.size_bytes,
    )


def cell_edges_from_centers(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Infer Voronoi-cell edges by midpoint and half-spacing extrapolation."""
    centers = np.asarray(values)
    if centers.dtype != np.dtype("<f8") or centers.ndim != 1:
        raise Evaluation2DValidationError(
            "cell centers must be a one-dimensional float64 array"
        )
    if centers.size < 2 or not np.isfinite(centers).all():
        raise Evaluation2DValidationError(
            "cell centers must contain at least two finite values"
        )
    differences = np.diff(centers)
    if not np.isfinite(differences).all() or np.any(differences <= 0):
        raise Evaluation2DValidationError("cell centers must be strictly increasing")
    edges = np.empty(centers.size + 1, dtype=np.float64)
    edges[1:-1] = centers[:-1] + differences / 2.0
    edges[0] = centers[0] - differences[0] / 2.0
    edges[-1] = centers[-1] + differences[-1] / 2.0
    widths = np.diff(edges)
    if not np.isfinite(widths).all() or np.any(widths <= 0):
        raise Evaluation2DValidationError(
            "cell-center coordinates do not imply finite positive widths"
        )
    return edges


def cell_widths_from_centers(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Infer positive Voronoi-cell widths using midpoint/extrapolated edges."""
    return np.diff(cell_edges_from_centers(values))


def _validate_bootstrap_options(
    confidence: float, n_resamples: int, seed: int
) -> tuple[float, int, int]:
    if (
        isinstance(confidence, (bool, np.bool_))
        or not np.isscalar(confidence)
        or not np.isfinite(confidence)
        or not 0.0 < float(confidence) < 1.0
    ):
        raise Evaluation2DValidationError(
            "bootstrap confidence must be a finite scalar in (0, 1)"
        )
    if isinstance(n_resamples, (bool, np.bool_)) or not isinstance(
        n_resamples, (int, np.integer)
    ):
        raise Evaluation2DValidationError("bootstrap n_resamples must be an integer")
    if int(n_resamples) < 1:
        raise Evaluation2DValidationError("bootstrap n_resamples must be positive")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise Evaluation2DValidationError("bootstrap seed must be an integer")
    if not 0 <= int(seed) <= np.iinfo(np.uint64).max:
        raise Evaluation2DValidationError("bootstrap seed must be in uint64 range")
    return float(confidence), int(n_resamples), int(seed)


def _estimate_record(estimate: float, samples: np.ndarray, alpha: float) -> dict[str, float]:
    return {
        "estimate": float(estimate),
        "ci_lower": float(np.quantile(samples, alpha, method="linear")),
        "ci_upper": float(np.quantile(samples, 1.0 - alpha, method="linear")),
    }


def _paired_bootstrap_summary(
    rmse: np.ndarray,
    mae: np.ndarray,
    *,
    confidence: float,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    if (
        rmse.ndim != 1
        or mae.ndim != 1
        or rmse.shape != mae.shape
        or rmse.size == 0
        or not np.isfinite(rmse).all()
        or not np.isfinite(mae).all()
    ):
        raise Evaluation2DValidationError(
            "paired per-sample metrics must be equal-length finite 1-D arrays"
        )
    confidence, n_resamples, seed = _validate_bootstrap_options(
        confidence, n_resamples, seed
    )
    rng = np.random.Generator(np.random.PCG64(seed))
    estimates = {
        "rmse_mean": np.empty(n_resamples, dtype=np.float64),
        "rmse_median": np.empty(n_resamples, dtype=np.float64),
        "mae_mean": np.empty(n_resamples, dtype=np.float64),
        "mae_median": np.empty(n_resamples, dtype=np.float64),
    }
    for start in range(0, n_resamples, 256):
        size = min(256, n_resamples - start)
        indices = rng.integers(0, rmse.size, size=(size, rmse.size))
        rmse_samples = rmse[indices]
        mae_samples = mae[indices]
        estimates["rmse_mean"][start : start + size] = np.mean(
            rmse_samples, axis=1
        )
        estimates["rmse_median"][start : start + size] = np.median(
            rmse_samples, axis=1
        )
        estimates["mae_mean"][start : start + size] = np.mean(mae_samples, axis=1)
        estimates["mae_median"][start : start + size] = np.median(
            mae_samples, axis=1
        )
    alpha = (1.0 - confidence) / 2.0
    return {
        "n_samples": int(rmse.size),
        "rmse_log10_resistivity": {
            "mean": _estimate_record(
                float(np.mean(rmse)), estimates["rmse_mean"], alpha
            ),
            "median": _estimate_record(
                float(np.median(rmse)), estimates["rmse_median"], alpha
            ),
        },
        "mae_log10_resistivity": {
            "mean": _estimate_record(float(np.mean(mae)), estimates["mae_mean"], alpha),
            "median": _estimate_record(
                float(np.median(mae)), estimates["mae_median"], alpha
            ),
        },
    }


def _pair_and_sort(
    truth: Truth2D, predictions: Predictions2D
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if predictions.observations_sha256 != truth.observations_sha256:
        raise Evaluation2DValidationError(
            "prediction and withheld truth must bind the same observations SHA-256"
        )
    if not np.array_equal(predictions.x_cell_centers_m, truth.x_cell_centers_m):
        raise Evaluation2DValidationError(
            "prediction x grid must exactly match withheld truth coordinates"
        )
    if not np.array_equal(
        predictions.depth_cell_centers_m, truth.depth_cell_centers_m
    ):
        raise Evaluation2DValidationError(
            "prediction depth grid must exactly match withheld truth coordinates"
        )
    truth_ids = {int(value) for value in truth.sample_index}
    prediction_ids = {int(value) for value in predictions.sample_index}
    if truth_ids != prediction_ids:
        missing = sorted(truth_ids - prediction_ids)
        extra = sorted(prediction_ids - truth_ids)
        raise Evaluation2DValidationError(
            "prediction sample ids must exactly match withheld truth; "
            f"missing={missing}, extra={extra}"
        )
    if predictions.log10_resistivity.shape[1:] != truth.log10_resistivity.shape[1:]:
        raise Evaluation2DValidationError(
            "prediction grid shape must exactly match withheld truth; "
            f"prediction={predictions.log10_resistivity.shape[1:]}, "
            f"truth={truth.log10_resistivity.shape[1:]}"
        )
    truth_positions = {int(value): index for index, value in enumerate(truth.sample_index)}
    prediction_positions = {
        int(value): index for index, value in enumerate(predictions.sample_index)
    }
    ordered_ids = np.asarray(sorted(truth_ids), dtype=np.int64)
    truth_order = np.asarray(
        [truth_positions[int(value)] for value in ordered_ids], dtype=np.int64
    )
    prediction_order = np.asarray(
        [prediction_positions[int(value)] for value in ordered_ids], dtype=np.int64
    )
    return (
        ordered_ids,
        truth.log10_resistivity[truth_order],
        predictions.log10_resistivity[prediction_order],
        truth.scenario[truth_order],
        truth.has_fault[truth_order],
    )


def _implementation_identity() -> dict[str, Any]:
    source = Path(__file__).resolve(strict=True)
    before = source.stat()
    payload = source.read_bytes()
    after = source.stat()
    signatures = (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns),
        (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
    )
    if signatures[0] != signatures[1] or len(payload) != before.st_size:
        raise Evaluation2DValidationError(
            "evaluator source changed while provenance was captured"
        )
    repository = source.parents[2]
    commit: str | None = None
    dirty_tree: bool | None = None
    try:
        top = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=30,
        ).stdout.strip()
        if Path(top).resolve(strict=True) == repository:
            commit = subprocess.run(
                ["git", "-C", str(repository), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                encoding="ascii",
                errors="strict",
                timeout=30,
            ).stdout.strip()
            status = subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository),
                    "status",
                    "--porcelain=v1",
                    "--untracked-files=normal",
                ],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="strict",
                timeout=30,
            ).stdout
            dirty_tree = bool(status)
    except (OSError, subprocess.SubprocessError, UnicodeError, ValueError):
        commit = None
        dirty_tree = None
    try:
        distribution_version = importlib.metadata.version("pimsr-benchmarks")
    except importlib.metadata.PackageNotFoundError:
        distribution_version = None
    return {
        "distribution_version": distribution_version,
        "git_commit": commit,
        "git_dirty_tree": dirty_tree,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "source_file": source.name,
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "source_size_bytes": len(payload),
    }


def evaluate_predictions_2d(
    truth_path: str | Path,
    prediction_path: str | Path,
    *,
    expected_truth_sha256: str | None = None,
    expected_prediction_sha256: str | None = None,
    expected_observations_sha256: str | None = None,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Evaluate predictions using cell-area weights and paired sample bootstrap.

    The same resampled sample rows are used for RMSE, MAE, mean and median,
    preserving their pairing.  Omitting expected SHA-256 values is diagnostic
    API usage only; the production CLI requires all three pins.  This function
    does not calculate a physics/data misfit; that must be produced by an
    independently pinned forward solver.
    """
    confidence, n_resamples, seed = _validate_bootstrap_options(
        confidence, n_resamples, seed
    )
    observations_pin = _validated_expected_sha256(
        expected_observations_sha256, "expected_observations_sha256"
    )
    truth = load_truth_2d(truth_path, expected_sha256=expected_truth_sha256)
    predictions = load_predictions_2d(
        prediction_path, expected_sha256=expected_prediction_sha256
    )
    if observations_pin is not None and truth.observations_sha256 != observations_pin:
        raise Evaluation2DValidationError(
            "truth observations SHA-256 does not match the pinned digest"
        )
    ordered_ids, expected, predicted, scenarios, has_fault = _pair_and_sort(
        truth, predictions
    )
    x_widths = cell_widths_from_centers(truth.x_cell_centers_m)
    depth_widths = cell_widths_from_centers(truth.depth_cell_centers_m)
    x_edges = cell_edges_from_centers(truth.x_cell_centers_m)
    depth_edges = cell_edges_from_centers(truth.depth_cell_centers_m)
    normalized_x = x_widths / np.sum(x_widths)
    normalized_depth = depth_widths / np.sum(depth_widths)
    weights = normalized_depth[:, None] * normalized_x[None, :]
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise Evaluation2DValidationError("cell-area weights must be finite and positive")
    difference = predicted.astype(np.float64) - expected.astype(np.float64)
    squared_error = difference * difference
    rmse = np.sqrt(np.sum(squared_error * weights[None, :, :], axis=(1, 2)))
    mae = np.sum(np.abs(difference) * weights[None, :, :], axis=(1, 2))
    if not np.isfinite(rmse).all() or not np.isfinite(mae).all():
        raise Evaluation2DValidationError("computed metrics must be finite")

    overall = _paired_bootstrap_summary(
        rmse,
        mae,
        confidence=confidence,
        n_resamples=n_resamples,
        seed=seed,
    )
    by_scenario: dict[str, dict[str, Any]] = {}
    for scenario in sorted(set(scenarios.tolist())):
        selected = scenarios == scenario
        by_scenario[scenario] = _paired_bootstrap_summary(
            rmse[selected],
            mae[selected],
            confidence=confidence,
            n_resamples=n_resamples,
            seed=seed,
        )
    depth_rmse = np.sqrt(
        np.sum(squared_error * normalized_x[None, None, :], axis=2)
    )
    depth_mae = np.sum(
        np.abs(difference) * normalized_x[None, None, :], axis=2
    )
    per_depth = [
        {
            "depth_cell_center_m": float(truth.depth_cell_centers_m[index]),
            "depth_cell_lower_edge_m": float(depth_edges[index]),
            "depth_cell_upper_edge_m": float(depth_edges[index + 1]),
            "depth_index": index,
            "mae_log10_resistivity": {
                "mean": float(np.mean(depth_mae[:, index])),
                "median": float(np.median(depth_mae[:, index])),
            },
            "n_samples": int(ordered_ids.size),
            "rmse_log10_resistivity": {
                "mean": float(np.mean(depth_rmse[:, index])),
                "median": float(np.median(depth_rmse[:, index])),
            },
        }
        for index in range(truth.depth_cell_centers_m.size)
    ]
    per_sample = [
        {
            "has_fault": bool(fault),
            "mae_log10_resistivity": float(mae_value),
            "rmse_log10_resistivity": float(rmse_value),
            "sample_index": int(sample_id),
            "scenario": str(scenario),
        }
        for sample_id, scenario, fault, rmse_value, mae_value in zip(
            ordered_ids, scenarios, has_fault, rmse, mae, strict=True
        )
    ]
    return {
        "audience": "benchmark_operator_only_until_predictions_locked",
        "schema": EVALUATION_SCHEMA,
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "inputs": {
            "observations": {
                "schema": "pimsr-sota-2d-observations",
                "schema_version": OBSERVATION_SCHEMA_VERSION,
                "sha256": truth.observations_sha256,
            },
            "prediction": {
                "schema": PREDICTION_SCHEMA,
                "schema_version": PREDICTION_SCHEMA_VERSION,
                "sha256": predictions.artifact_sha256,
                "size_bytes": predictions.artifact_size_bytes,
            },
            "truth": {
                "schema": TRUTH_SCHEMA,
                "schema_version": TRUTH_SCHEMA_VERSION,
                "sha256": truth.artifact_sha256,
                "size_bytes": truth.artifact_size_bytes,
            },
        },
        "metric_contract": {
            "aggregation_across_samples": "equal_sample_weight",
            "campaign_binding": "exact_observations_sha256_in_truth_and_prediction",
            "cell_edges": "midpoints_with_half_spacing_boundary_extrapolation",
            "grid_weighting": "normalized_physical_cell_area",
            "scoring_domain": {
                "depth_cell_edges_m": depth_edges.tolist(),
                "mask": "all_truth_grid_cells",
                "support": "full_grid_voronoi_cells_from_centers",
                "x_cell_edges_m": x_edges.tolist(),
            },
            "prediction_grid": "must_exactly_match_withheld_truth_grid",
            "quantity": "log10_resistivity_ohm_m",
            "sample_pairing": "exact_unique_sample_index",
        },
        "bootstrap_contract": {
            "algorithm": "numpy_random_PCG64_percentile_linear",
            "confidence": confidence,
            "cross_method_effect_ci": False,
            "headline_eligible": False,
            "hierarchical": False,
            "n_resamples": n_resamples,
            "resampled_fields": [
                "rmse_log10_resistivity",
                "mae_log10_resistivity",
            ],
            "pairing": "identical_resampled_sample_rows_for_all_metrics",
            "seed": seed,
            "scope": "single_method_single_campaign_sample_level_descriptive",
        },
        "implementation": _implementation_identity(),
        "physics_misfit": {
            "included": False,
            "reason": "requires_a_separate_independently_pinned_forward_solver",
        },
        "release_gate": {
            "predictions_locked": False,
            "public_release_allowed": False,
            "required_next_step": (
                "lock all method predictions, run the paired hierarchical comparator, "
                "then create a redacted public report"
            ),
        },
        "overall": overall,
        "by_scenario": by_scenario,
        "per_depth": per_depth,
        "per_sample": per_sample,
    }


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Encode a deterministic, finite, compact UTF-8 JSON document."""
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise Evaluation2DPublicationError(
            f"evaluation is not canonical JSON data: {exc}"
        ) from exc
    return (encoded + "\n").encode("utf-8")


def _path_identity(path: Path) -> tuple[int, int]:
    info = path.stat(follow_symlinks=False)
    return int(info.st_dev), int(info.st_ino)


def _unlink_owned_path(
    path: Path, expected_identity: tuple[int, int], *, role: str
) -> None:
    if not os.path.lexists(path):
        return
    info = path.stat(follow_symlinks=False)
    if (
        path.is_symlink()
        or not stat.S_ISREG(info.st_mode)
        or (int(info.st_dev), int(info.st_ino)) != expected_identity
    ):
        raise Evaluation2DPublicationError(
            f"refusing to delete {role} replaced during publication: {path}"
        )
    path.unlink()


def _unlink_ambiguous_link_if_owned(
    path: Path, expected_identity: tuple[int, int]
) -> None:
    """Remove a link only when an interrupted ``os.link`` created our inode."""
    if not os.path.lexists(path):
        return
    info = path.stat(follow_symlinks=False)
    if (
        not path.is_symlink()
        and stat.S_ISREG(info.st_mode)
        and (int(info.st_dev), int(info.st_ino)) == expected_identity
    ):
        path.unlink()


def publish_evaluation_2d(
    evaluation: Mapping[str, Any], output_path: str | Path
) -> Path:
    """Atomically publish canonical JSON without replacing any existing path."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    if destination.exists():
        raise Evaluation2DPublicationError(f"refusing to overwrite {destination}")
    if partial.exists():
        raise Evaluation2DPublicationError(f"refusing stale partial file {partial}")
    payload = canonical_json_bytes(evaluation)
    partial_identity: tuple[int, int] | None = None
    destination_identity: tuple[int, int] | None = None
    try:
        try:
            with partial.open("xb") as stream:
                info = os.fstat(stream.fileno())
                partial_identity = (int(info.st_dev), int(info.st_ino))
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as exc:
            raise Evaluation2DPublicationError(
                f"cannot write partial evaluation {partial}: {exc}"
            ) from exc
        if partial.read_bytes() != payload:
            raise Evaluation2DPublicationError("partial evaluation verification failed")
        try:
            os.link(partial, destination)
        except FileExistsError as exc:
            raise Evaluation2DPublicationError(
                f"publication race: destination appeared: {destination}"
            ) from exc
        except OSError as exc:
            raise Evaluation2DPublicationError(
                f"cannot publish evaluation {destination}: {exc}"
            ) from exc
        assert partial_identity is not None
        destination_identity = partial_identity
        if _path_identity(destination) != destination_identity or destination.is_symlink():
            raise Evaluation2DPublicationError(
                f"published evaluation identity mismatch: {destination}"
            )
        if destination.read_bytes() != payload:
            raise Evaluation2DPublicationError("published evaluation verification failed")
    except BaseException as exc:
        if destination_identity is not None:
            try:
                _unlink_owned_path(
                    destination,
                    destination_identity,
                    role="evaluation destination",
                )
            except Evaluation2DPublicationError as cleanup_error:
                raise cleanup_error from exc
        elif partial_identity is not None:
            _unlink_ambiguous_link_if_owned(destination, partial_identity)
        raise
    finally:
        if partial_identity is not None:
            _unlink_owned_path(
                partial,
                partial_identity,
                role="evaluation partial",
            )
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strictly score one PIMSR SOTA 2-D prediction artifact"
    )
    parser.add_argument("--truth", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--truth-sha256", required=True)
    parser.add_argument("--prediction-sha256", required=True)
    parser.add_argument("--observations-sha256", required=True)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point used by the standalone evaluation script."""
    args = _parser().parse_args(argv)
    evaluation = evaluate_predictions_2d(
        args.truth,
        args.predictions,
        expected_truth_sha256=args.truth_sha256,
        expected_prediction_sha256=args.prediction_sha256,
        expected_observations_sha256=args.observations_sha256,
        confidence=args.confidence,
        n_resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )
    destination = publish_evaluation_2d(evaluation, args.output)
    print(
        f"published {destination} "
        f"sha256={hashlib.sha256(destination.read_bytes()).hexdigest()}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the wrapper script
    raise SystemExit(main())
