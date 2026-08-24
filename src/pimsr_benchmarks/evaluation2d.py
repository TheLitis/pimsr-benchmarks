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
import re
import stat
import subprocess
import sys
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ._publication_io import (
    close_publication_descriptor,
    ensure_real_directory,
    open_exclusive_publication,
    open_verified_publication,
    set_publication_descriptor_read_only,
)
from .prediction_lock2d import (
    FAMILY_COMMITMENT_CONTRACT,
    FAMILY_PARTITION_SCHEMA,
    FAMILY_PARTITION_SCHEMA_VERSION,
    GEOLOGICAL_FAMILIES,
    LOCK_SCHEMA,
    LOCK_SCHEMA_VERSION,
    PredictionLock2DValidationError,
    snapshot_regular_file,
    validate_locked_run_artifacts_2d,
    validate_prediction_lock_2d,
)

TRUTH_SCHEMA = "pimsr-sota-2d-truth"
PREDICTION_SCHEMA = "pimsr-sota-2d-predictions"
EVALUATION_SCHEMA = "pimsr-sota-2d-evaluation"
OBSERVATION_SCHEMA_VERSION = 1
TRUTH_SCHEMA_VERSION = 2
PREDICTION_SCHEMA_VERSION = 2
EVALUATION_SCHEMA_VERSION = 3
OBSERVATION_MANIFEST_SCHEMA = "pimsr-sota-2d-observation-manifest"
OBSERVATION_MANIFEST_SCHEMA_VERSION = 3
OPERATOR_MANIFEST_SCHEMA = "pimsr-sota-2d-scoring-manifest"
OPERATOR_MANIFEST_SCHEMA_VERSION = 3
FAMILY_REVEAL_SCHEMA = "pimsr-sota-2d-family-partition-reveal"
FAMILY_REVEAL_SCHEMA_VERSION = 1
FAMILY_COMMITMENT_DOMAIN = b"pimsr-sota-2d-family-partition/v1\x00"
SAMPLES_PER_CAMPAIGN = 500
BASE_MODELS_PER_FAMILY = 20
NOISE_REALIZATIONS_PER_BASE = 5
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
class Evaluation2DPublicationReceipt:
    """Bytes verified through the final, reopened publication descriptor."""

    path: Path
    sha256: str
    size_bytes: int


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


@dataclass(frozen=True)
class _OperatorBinding:
    snapshot: _ArtifactSnapshot
    truth_sha256: str
    truth_size_bytes: int | None
    family_commitment_sha256: str
    family_by_sample: Mapping[int, str]


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
    expected = _validated_expected_sha256(
        expected_sha256, f"expected_{artifact_name}_sha256"
    )
    try:
        snapshot = snapshot_regular_file(
            path,
            expected_sha256=expected,
            role=f"{artifact_name} artifact",
        )
    except PredictionLock2DValidationError as exc:
        raise Evaluation2DValidationError(
            f"cannot read locked {artifact_name} artifact: {exc}"
        ) from exc
    return _ArtifactSnapshot(snapshot.path, snapshot.payload, snapshot.sha256)


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


def load_truth_2d(path: str | Path, *, expected_sha256: str | None = None) -> Truth2D:
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
    _require_scalar_schema(arrays["schema"], expected=TRUTH_SCHEMA, artifact_name="truth")
    _require_version(arrays["schema_version"], "truth", TRUTH_SCHEMA_VERSION)
    sample_index = _validate_sample_indices(arrays["sample_index"], "truth.sample_index")
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
    x_centers = _validate_centers(arrays["x_cell_centers_m"], "truth.x_cell_centers_m")
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
        raise Evaluation2DValidationError("truth.truth_log10_resistivity must be finite")
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
    _require_version(arrays["schema_version"], "prediction", PREDICTION_SCHEMA_VERSION)
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


def _estimate_record(
    estimate: float, samples: np.ndarray, alpha: float
) -> dict[str, float]:
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
        estimates["rmse_mean"][start : start + size] = np.mean(rmse_samples, axis=1)
        estimates["rmse_median"][start : start + size] = np.median(rmse_samples, axis=1)
        estimates["mae_mean"][start : start + size] = np.mean(mae_samples, axis=1)
        estimates["mae_median"][start : start + size] = np.median(mae_samples, axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "n_samples": int(rmse.size),
        "rmse_log10_resistivity": {
            "mean": _estimate_record(float(np.mean(rmse)), estimates["rmse_mean"], alpha),
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
    if not np.array_equal(predictions.depth_cell_centers_m, truth.depth_cell_centers_m):
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
    truth_positions = {
        int(value): index for index, value in enumerate(truth.sample_index)
    }
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


def _normal_path(path: Path) -> str:
    return os.path.normcase(os.path.normpath(str(path)))


def _direct_absolute_path(path: str | Path, *, role: str) -> Path:
    requested = Path(os.path.abspath(os.fspath(path)))
    try:
        resolved = requested.resolve(strict=True)
    except OSError as exc:
        raise Evaluation2DValidationError(f"cannot resolve {role}: {requested}") from exc
    if _normal_path(requested) != _normal_path(resolved):
        raise Evaluation2DValidationError(
            f"{role} must not traverse a symbolic link or redirected parent"
        )
    return requested


def _directory_identities(paths: Sequence[Path]) -> tuple[tuple[int, int], ...]:
    identities: list[tuple[int, int]] = []
    for path in paths:
        try:
            info = os.lstat(path)
        except OSError as exc:
            raise Evaluation2DValidationError(
                f"cannot inspect evaluator provenance parent: {path}"
            ) from exc
        if not stat.S_ISDIR(info.st_mode) or path.is_symlink():
            raise Evaluation2DValidationError(
                f"evaluator provenance parent must be a direct directory: {path}"
            )
        identities.append((int(info.st_dev), int(info.st_ino)))
    return tuple(identities)


def _git_bytes(
    repository: Path,
    *arguments: str,
    input_payload: bytes | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=check,
            capture_output=True,
            input=input_payload,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        detail = getattr(exc, "stderr", b"")
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", errors="replace")
        raise Evaluation2DValidationError(
            f"cannot establish evaluator Git provenance: {str(detail).strip() or exc}"
        ) from exc


def _git_text(repository: Path, *arguments: str, encoding: str = "ascii") -> str:
    payload = _git_bytes(repository, *arguments).stdout
    try:
        return payload.decode(encoding, errors="strict").strip()
    except UnicodeError as exc:
        raise Evaluation2DValidationError(
            "evaluator Git provenance contains invalid text"
        ) from exc


def _git_blob_at(repository: Path, commit: str, relative_source: str) -> str:
    record = _git_bytes(repository, "ls-tree", "-z", commit, "--", relative_source).stdout
    rows = [row for row in record.split(b"\0") if row]
    try:
        metadata, encoded_path = rows[0].split(b"\t", 1)
        mode, object_type, object_id = metadata.split(b" ")
        recorded_path = encoded_path.decode("utf-8", errors="strict")
        object_id_text = object_id.decode("ascii", errors="strict")
    except (IndexError, ValueError, UnicodeError) as exc:
        raise Evaluation2DValidationError(
            "evaluator source is not a unique regular blob in the pinned commit"
        ) from exc
    if (
        len(rows) != 1
        or mode not in {b"100644", b"100755"}
        or object_type != b"blob"
        or recorded_path != relative_source
        or not re.fullmatch(r"[0-9a-f]{40}", object_id_text)
    ):
        raise Evaluation2DValidationError(
            "evaluator source is not a unique regular blob in the pinned commit"
        )
    return object_id_text


def _implementation_identity() -> dict[str, Any]:
    source = _direct_absolute_path(__file__, role="evaluator source")
    repository = source.parents[2]
    relative_source = source.relative_to(repository).as_posix()
    provenance_parents = (
        repository,
        repository / relative_source.split("/", maxsplit=1)[0],
        source.parent,
    )
    parent_identities = _directory_identities(provenance_parents)
    try:
        source_snapshot = snapshot_regular_file(
            source,
            expected_sha256=None,
            role="evaluator implementation source",
        )
    except PredictionLock2DValidationError as exc:
        raise Evaluation2DValidationError(
            f"cannot capture evaluator implementation source: {exc}"
        ) from exc

    top = _direct_absolute_path(
        _git_text(repository, "rev-parse", "--show-toplevel", encoding="utf-8"),
        role="evaluator Git repository",
    )
    if _normal_path(top) != _normal_path(repository):
        raise Evaluation2DValidationError(
            "evaluator source is not inside the expected Git repository root"
        )
    head_commit = _git_text(repository, "rev-parse", "HEAD^{commit}")
    commit = _git_text(
        repository,
        "log",
        "-1",
        "--format=%H",
        "--",
        relative_source,
    )
    if not re.fullmatch(r"[0-9a-f]{40}", head_commit) or not re.fullmatch(
        r"[0-9a-f]{40}", commit
    ):
        raise Evaluation2DValidationError("evaluator Git provenance is incomplete")
    ancestor = _git_bytes(
        repository, "merge-base", "--is-ancestor", commit, head_commit, check=False
    )
    if ancestor.returncode != 0:
        raise Evaluation2DValidationError(
            "evaluator source commit is not an ancestor of the executed HEAD"
        )
    pinned_blob = _git_blob_at(repository, commit, relative_source)
    if _git_blob_at(repository, head_commit, relative_source) != pinned_blob:
        raise Evaluation2DValidationError(
            "executed HEAD changes the evaluator source after its pinned commit"
        )
    filtered_blob = _git_bytes(
        repository,
        "hash-object",
        "--stdin",
        "--path",
        relative_source,
        input_payload=source_snapshot.payload,
    ).stdout
    try:
        filtered_blob_text = filtered_blob.decode("ascii", errors="strict").strip()
    except UnicodeError as exc:
        raise Evaluation2DValidationError(
            "evaluator clean-filter object id is not ASCII"
        ) from exc
    if filtered_blob_text != pinned_blob:
        raise Evaluation2DValidationError(
            "captured evaluator source does not match the pinned commit blob "
            "after Git clean filtering"
        )
    status = _git_bytes(
        repository, "status", "--porcelain=v1", "--untracked-files=normal"
    ).stdout
    dirty_tree = bool(status)

    if _directory_identities(provenance_parents) != parent_identities:
        raise Evaluation2DValidationError(
            "evaluator source parent changed while provenance was captured"
        )
    if _normal_path(
        _direct_absolute_path(source, role="evaluator source")
    ) != _normal_path(source):
        raise Evaluation2DValidationError(
            "evaluator source pathname changed while provenance was captured"
        )
    try:
        final_snapshot = snapshot_regular_file(
            source,
            expected_sha256=source_snapshot.sha256,
            role="evaluator implementation source recheck",
        )
    except PredictionLock2DValidationError as exc:
        raise Evaluation2DValidationError(
            f"evaluator source changed while provenance was captured: {exc}"
        ) from exc
    if final_snapshot.identity != source_snapshot.identity:
        raise Evaluation2DValidationError(
            "evaluator source pathname was replaced while provenance was captured"
        )
    if (
        _git_text(repository, "rev-parse", "HEAD^{commit}") != head_commit
        or _git_text(
            repository,
            "log",
            "-1",
            "--format=%H",
            "--",
            relative_source,
        )
        != commit
        or _git_bytes(
            repository, "status", "--porcelain=v1", "--untracked-files=normal"
        ).stdout
        != status
    ):
        raise Evaluation2DValidationError(
            "evaluator Git state changed while provenance was captured"
        )
    try:
        distribution_version = importlib.metadata.version("pimsr-benchmarks")
    except importlib.metadata.PackageNotFoundError:
        distribution_version = None
    return {
        "distribution_version": distribution_version,
        "git_commit": commit,
        "git_dirty_tree": dirty_tree,
        "git_head_commit": head_commit,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "source_file": source.name,
        "source_sha256": source_snapshot.sha256,
        "source_size_bytes": source_snapshot.size_bytes,
    }


def _strict_json_document(payload: bytes, artifact_name: str) -> Mapping[str, Any]:
    def object_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise Evaluation2DValidationError(
                    f"{artifact_name} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=object_hook,
            parse_constant=lambda value: (_ for _ in ()).throw(
                Evaluation2DValidationError(
                    f"{artifact_name} contains non-finite constant {value!r}"
                )
            ),
        )
    except Evaluation2DValidationError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Evaluation2DValidationError(
            f"cannot decode {artifact_name}: {exc}"
        ) from exc
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise Evaluation2DValidationError(
            f"{artifact_name} root must be an object with string keys"
        )
    return value


def _json_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise Evaluation2DValidationError(f"{path} must be an object with string keys")
    return value


def _json_sequence(value: Any, path: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise Evaluation2DValidationError(f"{path} must be an array")
    return value


def _json_integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise Evaluation2DValidationError(
            f"{path} must be an integer greater than or equal to {minimum}"
        )
    return value


def _json_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise Evaluation2DValidationError(f"{path} must be a non-empty NUL-free string")
    return value


def _exact_json_keys(
    value: Mapping[str, Any], expected: frozenset[str], path: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise Evaluation2DValidationError(
            f"{path} keys mismatch; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _operator_artifact_record(
    value: Any,
    *,
    path: str,
    schema: str,
    schema_version: int,
) -> tuple[str, int]:
    record = _json_mapping(value, path)
    _exact_json_keys(
        record,
        frozenset({"schema", "schema_version", "sha256", "size_bytes"}),
        path,
    )
    digest = _validated_expected_sha256(record["sha256"], f"{path}.sha256")
    size_bytes = _json_integer(record["size_bytes"], f"{path}.size_bytes", minimum=1)
    if record["schema"] != schema or record["schema_version"] != schema_version:
        raise Evaluation2DValidationError(f"{path} schema identity is wrong")
    assert digest is not None
    return digest, size_bytes


def _public_family_commitment(
    locked_artifacts: Any,
    *,
    campaign_id: str,
    observations_sha256: str,
) -> str:
    try:
        manifest_snapshot = locked_artifacts.observation_manifest
        observations_snapshot = locked_artifacts.observations
    except AttributeError as exc:
        raise Evaluation2DValidationError(
            "locked public artifacts are incomplete"
        ) from exc
    manifest = _strict_json_document(
        manifest_snapshot.payload, "public observation manifest"
    )
    _exact_json_keys(
        manifest,
        frozenset(
            {
                "audience",
                "declared_evaluation_floors",
                "family_partition_commitment",
                "observation_payload",
                "physical_contract",
                "sample_count",
                "schema",
                "schema_version",
                "split_id",
            }
        ),
        "public observation manifest",
    )
    if (
        manifest["audience"] != "method_input_public"
        or manifest["schema"] != OBSERVATION_MANIFEST_SCHEMA
        or manifest["schema_version"] != OBSERVATION_MANIFEST_SCHEMA_VERSION
        or manifest["split_id"] != campaign_id
        or manifest["sample_count"] != SAMPLES_PER_CAMPAIGN
    ):
        raise Evaluation2DValidationError(
            "public observation manifest campaign identity is wrong"
        )
    payload = _json_mapping(
        manifest["observation_payload"],
        "public observation manifest.observation_payload",
    )
    payload_digest = _validated_expected_sha256(
        payload.get("sha256"), "observation_payload.sha256"
    )
    if (
        payload.get("schema") != "pimsr-sota-2d-observations"
        or payload.get("schema_version") != OBSERVATION_SCHEMA_VERSION
        or payload_digest != observations_sha256
        or payload_digest != observations_snapshot.sha256
        or payload.get("size_bytes") != observations_snapshot.size_bytes
    ):
        raise Evaluation2DValidationError(
            "public observation manifest does not bind the locked observations"
        )
    commitment = _json_mapping(
        manifest["family_partition_commitment"],
        "public observation manifest.family_partition_commitment",
    )
    _exact_json_keys(
        commitment,
        frozenset({"contract", "schema", "schema_version", "sha256"}),
        "public observation manifest.family_partition_commitment",
    )
    digest = _validated_expected_sha256(
        commitment["sha256"], "family_partition_commitment.sha256"
    )
    if (
        commitment["schema"] != FAMILY_PARTITION_SCHEMA
        or commitment["schema_version"] != FAMILY_PARTITION_SCHEMA_VERSION
        or commitment["contract"] != FAMILY_COMMITMENT_CONTRACT
    ):
        raise Evaluation2DValidationError(
            "public family commitment contract is not the preregistered contract"
        )
    assert digest is not None
    return digest


def _family_commitment_digest(
    *, campaign_id: str, nonce_hex: str, rows: Sequence[Mapping[str, Any]]
) -> str:
    if len(nonce_hex) != 64 or any(
        character not in "0123456789abcdef" for character in nonce_hex
    ):
        raise Evaluation2DValidationError(
            "family_partition_reveal.nonce_hex must encode exactly 32 bytes "
            "as lowercase hexadecimal"
        )
    body = {
        "campaign_id": campaign_id,
        "rows": [dict(row) for row in rows],
        "schema": "pimsr-sota-2d-family-partition-reveal-body",
        "schema_version": 1,
    }
    try:
        body_bytes = canonical_json_bytes(body)
    except Evaluation2DPublicationError as exc:
        raise Evaluation2DValidationError(
            "family partition reveal body is not canonical JSON"
        ) from exc
    digest = hashlib.sha256()
    digest.update(FAMILY_COMMITMENT_DOMAIN)
    digest.update(bytes.fromhex(nonce_hex))
    digest.update(body_bytes)
    return digest.hexdigest()


def _operator_family_partition(
    split: Mapping[str, Any],
    *,
    campaign_id: str,
    expected_commitment_sha256: str,
) -> Mapping[int, str]:
    _exact_json_keys(
        split,
        frozenset(
            {
                "family_partition_reveal",
                "groups",
                "opaque_sample_id_contract",
                "payload_row_order",
                "sample_count",
                "sample_id_mapping",
                "scenario_groups",
                "split_id",
            }
        ),
        "operator split",
    )
    if split["split_id"] != campaign_id or split["sample_count"] != SAMPLES_PER_CAMPAIGN:
        raise Evaluation2DValidationError("operator manifest campaign binding is wrong")
    groups = _json_sequence(split["groups"], "operator split.groups")
    if len(groups) != SAMPLES_PER_CAMPAIGN:
        raise Evaluation2DValidationError(
            "operator hierarchy requires exactly 500 sample groups"
        )
    sample_rows: dict[int, dict[str, Any]] = {}
    base_families: dict[str, str] = {}
    base_noise: dict[str, set[int]] = {}
    family_bases: dict[str, set[str]] = {family: set() for family in GEOLOGICAL_FAMILIES}
    for index, raw_group in enumerate(groups):
        path = f"operator split.groups[{index}]"
        group = _json_mapping(raw_group, path)
        _exact_json_keys(
            group,
            frozenset({"base_model_id", "family_id", "noise_id", "sample_ids"}),
            path,
        )
        family_id = _json_string(group["family_id"], f"{path}.family_id")
        if family_id not in GEOLOGICAL_FAMILIES:
            raise Evaluation2DValidationError(
                f"{path}.family_id is not a preregistered geological family"
            )
        base_model_id = _json_string(group["base_model_id"], f"{path}.base_model_id")
        noise_id = _json_string(group["noise_id"], f"{path}.noise_id")
        noise_match = re.fullmatch(r"noise-([0-4])", noise_id, flags=re.ASCII)
        if noise_match is None:
            raise Evaluation2DValidationError(
                "operator noise ids must be exactly noise-0 through noise-4"
            )
        noise_index = int(noise_match.group(1))
        sample_names = _json_sequence(group["sample_ids"], f"{path}.sample_ids")
        if len(sample_names) != 1:
            raise Evaluation2DValidationError(
                "each operator family/base/noise group must contain one sample"
            )
        sample_name = _json_string(sample_names[0], f"{path}.sample_ids[0]")
        if not re.fullmatch(r"sample-[0-9]+", sample_name, flags=re.ASCII):
            raise Evaluation2DValidationError(
                "operator sample id is not an opaque int64 encoding"
            )
        sample_index = int(sample_name[7:])
        if sample_index > np.iinfo(np.int64).max or sample_index in sample_rows:
            raise Evaluation2DValidationError(
                "operator opaque sample ids are invalid or duplicated"
            )
        previous_family = base_families.setdefault(base_model_id, family_id)
        if previous_family != family_id:
            raise Evaluation2DValidationError(
                "one operator base model appears in multiple families"
            )
        noises = base_noise.setdefault(base_model_id, set())
        if noise_index in noises:
            raise Evaluation2DValidationError(
                "one operator base model repeats a noise realization"
            )
        noises.add(noise_index)
        family_bases[family_id].add(base_model_id)
        sample_rows[sample_index] = {
            "base_model_id": base_model_id,
            "family_id": family_id,
            "noise_index": noise_index,
            "sample_index": sample_index,
        }
    if len(base_families) != len(GEOLOGICAL_FAMILIES) * BASE_MODELS_PER_FAMILY:
        raise Evaluation2DValidationError(
            "operator hierarchy requires exactly 100 base models"
        )
    if any(
        noises != set(range(NOISE_REALIZATIONS_PER_BASE))
        for noises in base_noise.values()
    ):
        raise Evaluation2DValidationError(
            "each operator base model requires noise realizations 0 through 4"
        )
    if any(
        len(family_bases[family]) != BASE_MODELS_PER_FAMILY
        for family in GEOLOGICAL_FAMILIES
    ):
        raise Evaluation2DValidationError(
            "operator hierarchy requires exactly 20 bases in every family"
        )
    opaque_contract = _json_mapping(
        split["opaque_sample_id_contract"], "operator opaque_sample_id_contract"
    )
    if dict(opaque_contract) != {
        "algorithm": "HMAC-SHA256",
        "digest_projection": "first_64_bits_big_endian_clear_sign_bit",
        "key_material": "external_secret_not_recorded",
        "message": (
            "domain_separator || generator_seed_uint64_be || "
            "source_sample_index_uint64_be || split_id_length_uint32_be || "
            "split_id_ascii"
        ),
        "version": 1,
    }:
        raise Evaluation2DValidationError("operator opaque sample-id contract is wrong")
    if split["payload_row_order"] != "strictly_increasing_opaque_sample_index":
        raise Evaluation2DValidationError("operator payload row-order contract is wrong")
    mappings = _json_sequence(split["sample_id_mapping"], "operator sample_id_mapping")
    mapped_opaque: set[int] = set()
    opaque_ids: list[int] = []
    source_ids: list[int] = []
    for index, raw_mapping in enumerate(mappings):
        path = f"operator sample_id_mapping[{index}]"
        mapping = _json_mapping(raw_mapping, path)
        _exact_json_keys(
            mapping,
            frozenset({"opaque_sample_index", "source_generator_sample_index"}),
            path,
        )
        opaque = _json_integer(
            mapping["opaque_sample_index"], f"{path}.opaque_sample_index"
        )
        source = _json_integer(
            mapping["source_generator_sample_index"],
            f"{path}.source_generator_sample_index",
        )
        if (
            opaque > np.iinfo(np.int64).max
            or opaque in mapped_opaque
            or source > np.iinfo(np.int64).max
            or source in source_ids
        ):
            raise Evaluation2DValidationError(
                "operator sample mapping contains duplicate or invalid ids"
            )
        mapped_opaque.add(opaque)
        opaque_ids.append(opaque)
        source_ids.append(source)
    if (
        len(mappings) != SAMPLES_PER_CAMPAIGN
        or mapped_opaque != set(sample_rows)
        or opaque_ids != sorted(opaque_ids)
        or sorted(source_ids) != list(range(SAMPLES_PER_CAMPAIGN))
    ):
        raise Evaluation2DValidationError(
            "operator sample mapping differs from the exact hierarchy"
        )
    scenarios = _json_sequence(split["scenario_groups"], "operator scenario_groups")
    grouped_samples: dict[str, set[int]] = {}
    scenario_indices: set[int] = set()
    for index, raw_scenario in enumerate(scenarios):
        path = f"operator scenario_groups[{index}]"
        scenario = _json_mapping(raw_scenario, path)
        _exact_json_keys(
            scenario,
            frozenset({"opaque_sample_indices", "scenario", "scenario_index"}),
            path,
        )
        family_id = _json_string(scenario["scenario"], f"{path}.scenario")
        scenario_index = _json_integer(
            scenario["scenario_index"], f"{path}.scenario_index"
        )
        sample_values = _json_sequence(
            scenario["opaque_sample_indices"], f"{path}.opaque_sample_indices"
        )
        samples = {
            _json_integer(value, f"{path}.opaque_sample_indices[{sample_position}]")
            for sample_position, value in enumerate(sample_values)
        }
        if (
            family_id not in GEOLOGICAL_FAMILIES
            or scenario_index != GEOLOGICAL_FAMILIES.index(family_id)
            or family_id in grouped_samples
            or scenario_index in scenario_indices
            or len(samples) != len(sample_values)
        ):
            raise Evaluation2DValidationError(
                "operator scenario groups are invalid or duplicated"
            )
        grouped_samples[family_id] = samples
        scenario_indices.add(scenario_index)
    expected_groups = {
        family: {
            sample_index
            for sample_index, row in sample_rows.items()
            if row["family_id"] == family
        }
        for family in GEOLOGICAL_FAMILIES
    }
    if grouped_samples != expected_groups:
        raise Evaluation2DValidationError(
            "operator scenario groups differ from the exact hierarchy"
        )
    reveal = _json_mapping(
        split["family_partition_reveal"],
        "operator split.family_partition_reveal",
    )
    _exact_json_keys(
        reveal,
        frozenset({"campaign_id", "nonce_hex", "rows", "schema", "schema_version"}),
        "operator split.family_partition_reveal",
    )
    if (
        reveal["schema"] != FAMILY_REVEAL_SCHEMA
        or reveal["schema_version"] != FAMILY_REVEAL_SCHEMA_VERSION
        or reveal["campaign_id"] != campaign_id
    ):
        raise Evaluation2DValidationError(
            "operator family partition reveal identity is wrong"
        )
    raw_rows = _json_sequence(
        reveal["rows"], "operator split.family_partition_reveal.rows"
    )
    rows: list[dict[str, Any]] = []
    for index, raw_row in enumerate(raw_rows):
        path = f"operator split.family_partition_reveal.rows[{index}]"
        row = _json_mapping(raw_row, path)
        _exact_json_keys(
            row,
            frozenset({"base_model_id", "family_id", "noise_index", "sample_index"}),
            path,
        )
        rows.append(
            {
                "base_model_id": _json_string(
                    row["base_model_id"], f"{path}.base_model_id"
                ),
                "family_id": _json_string(row["family_id"], f"{path}.family_id"),
                "noise_index": _json_integer(row["noise_index"], f"{path}.noise_index"),
                "sample_index": _json_integer(
                    row["sample_index"], f"{path}.sample_index"
                ),
            }
        )
    expected_rows = [sample_rows[sample_id] for sample_id in sorted(sample_rows)]
    if rows != expected_rows:
        raise Evaluation2DValidationError(
            "operator family partition reveal differs from the exact hierarchy"
        )
    digest = _family_commitment_digest(
        campaign_id=campaign_id,
        nonce_hex=_json_string(
            reveal["nonce_hex"], "operator split.family_partition_reveal.nonce_hex"
        ),
        rows=rows,
    )
    if digest != expected_commitment_sha256:
        raise Evaluation2DValidationError(
            "operator family partition reveal does not open the public commitment"
        )
    return {sample_id: str(row["family_id"]) for sample_id, row in sample_rows.items()}


def _load_operator_binding(
    path: str | Path,
    *,
    expected_sha256: str,
    campaign_id: str,
    observations_sha256: str,
    expected_truth_sha256: str,
    locked_artifacts: Any,
) -> _OperatorBinding:
    """Validate the operator hierarchy/reveal after the prediction lock gate."""
    snapshot = _read_snapshot(
        path,
        expected_sha256=expected_sha256,
        artifact_name="operator_manifest",
    )
    value = _strict_json_document(snapshot.payload, "operator manifest")
    _exact_json_keys(
        value,
        frozenset(
            {"artifacts", "audience", "schema", "schema_version", "source", "split"}
        ),
        "operator manifest",
    )
    if (
        value["audience"] != "benchmark_operator_only"
        or value["schema"] != OPERATOR_MANIFEST_SCHEMA
        or value["schema_version"] != OPERATOR_MANIFEST_SCHEMA_VERSION
    ):
        raise Evaluation2DValidationError("operator manifest identity is wrong")
    artifacts = _json_mapping(value["artifacts"], "operator artifacts")
    _exact_json_keys(
        artifacts,
        frozenset({"observations", "public_observation_manifest", "withheld_truth"}),
        "operator artifacts",
    )
    observation_digest, observation_size = _operator_artifact_record(
        artifacts["observations"],
        path="operator artifacts.observations",
        schema="pimsr-sota-2d-observations",
        schema_version=OBSERVATION_SCHEMA_VERSION,
    )
    public_manifest_digest, public_manifest_size = _operator_artifact_record(
        artifacts["public_observation_manifest"],
        path="operator artifacts.public_observation_manifest",
        schema=OBSERVATION_MANIFEST_SCHEMA,
        schema_version=OBSERVATION_MANIFEST_SCHEMA_VERSION,
    )
    truth_digest, truth_size = _operator_artifact_record(
        artifacts["withheld_truth"],
        path="operator artifacts.withheld_truth",
        schema=TRUTH_SCHEMA,
        schema_version=TRUTH_SCHEMA_VERSION,
    )
    try:
        locked_observations = locked_artifacts.observations
        locked_manifest = locked_artifacts.observation_manifest
    except AttributeError as exc:
        raise Evaluation2DValidationError(
            "locked public artifacts are incomplete"
        ) from exc
    if (
        observation_digest != observations_sha256
        or observation_digest != locked_observations.sha256
        or observation_size != locked_observations.size_bytes
        or public_manifest_digest != locked_manifest.sha256
        or public_manifest_size != locked_manifest.size_bytes
    ):
        raise Evaluation2DValidationError(
            "operator manifest public artifacts differ from the locked campaign"
        )
    if truth_digest != expected_truth_sha256:
        raise Evaluation2DValidationError(
            "operator manifest truth differs from the external truth pin"
        )
    family_commitment = _public_family_commitment(
        locked_artifacts,
        campaign_id=campaign_id,
        observations_sha256=observations_sha256,
    )
    family_by_sample = _operator_family_partition(
        _json_mapping(value["split"], "operator split"),
        campaign_id=campaign_id,
        expected_commitment_sha256=family_commitment,
    )
    source = _json_mapping(value["source"], "operator source")
    if dict(source) != {
        "production_generation_closure": "post_score_manifest.campaign.hidden_generation"
    }:
        raise Evaluation2DValidationError(
            "operator source must defer to the material hidden-generation closure"
        )
    return _OperatorBinding(
        snapshot=snapshot,
        truth_sha256=truth_digest,
        truth_size_bytes=truth_size,
        family_commitment_sha256=family_commitment,
        family_by_sample=family_by_sample,
    )


def evaluate_predictions_2d(
    truth_path: str | Path,
    prediction_path: str | Path,
    *,
    preregistration_path: str | Path,
    expected_preregistration_sha256: str,
    predictions_lock_path: str | Path,
    expected_predictions_lock_sha256: str,
    campaign_id: str,
    method_id: str,
    training_seed: int,
    observations_path: str | Path,
    observation_manifest_path: str | Path,
    runtime_path: str | Path,
    checkpoint_path: str | Path,
    source_path: str | Path,
    operator_manifest_path: str | Path,
    expected_operator_manifest_sha256: str,
    expected_truth_sha256: str,
) -> dict[str, Any]:
    """Evaluate one run only after validating the complete pre-score lock.

    The same resampled sample rows are used for RMSE, MAE, mean and median,
    preserving their pairing.  Lock/preregistration pins and the run coordinate
    are mandatory in the API itself.  The complete lock is validated before
    this function calls ``load_truth_2d`` or otherwise opens withheld truth.
    """
    try:
        prediction_lock = validate_prediction_lock_2d(
            preregistration_path,
            predictions_lock_path,
            expected_preregistration_sha256=expected_preregistration_sha256,
            expected_lock_sha256=expected_predictions_lock_sha256,
        )
        locked_run = prediction_lock.require_run(campaign_id, method_id, training_seed)
    except PredictionLock2DValidationError as exc:
        raise Evaluation2DValidationError(
            f"prediction lock validation failed before truth access: {exc}"
        ) from exc
    truth_pin = _validated_expected_sha256(expected_truth_sha256, "expected_truth_sha256")
    if truth_pin is None:
        raise Evaluation2DValidationError("expected_truth_sha256 is required")
    try:
        locked_artifacts = validate_locked_run_artifacts_2d(
            locked_run,
            observations_path=observations_path,
            observation_manifest_path=observation_manifest_path,
            prediction_path=prediction_path,
            runtime_path=runtime_path,
            checkpoint_path=checkpoint_path,
            source_path=source_path,
        )
    except PredictionLock2DValidationError as exc:
        raise Evaluation2DValidationError(
            f"locked run artifact validation failed before operator/truth access: {exc}"
        ) from exc
    statistical = prediction_lock.statistical_options
    confidence, n_resamples, seed = _validate_bootstrap_options(
        statistical["confidence"],
        statistical["n_resamples"],
        statistical["rng_seed"],
    )
    predictions = load_predictions_2d(
        prediction_path, expected_sha256=locked_run.prediction_sha256
    )
    if predictions.observations_sha256 != locked_run.observations_sha256:
        raise Evaluation2DValidationError(
            "prediction observations SHA-256 does not match the locked campaign"
        )
    operator_binding = _load_operator_binding(
        operator_manifest_path,
        expected_sha256=expected_operator_manifest_sha256,
        campaign_id=campaign_id,
        observations_sha256=locked_run.observations_sha256,
        expected_truth_sha256=truth_pin,
        locked_artifacts=locked_artifacts,
    )
    truth = load_truth_2d(truth_path, expected_sha256=operator_binding.truth_sha256)
    if (
        operator_binding.truth_size_bytes is not None
        and truth.artifact_size_bytes != operator_binding.truth_size_bytes
    ):
        raise Evaluation2DValidationError(
            "withheld truth size differs from the operator binding"
        )
    if truth.observations_sha256 != locked_run.observations_sha256:
        raise Evaluation2DValidationError(
            "truth observations SHA-256 does not match the locked campaign"
        )
    ordered_ids, expected, predicted, scenarios, has_fault = _pair_and_sort(
        truth, predictions
    )
    if set(operator_binding.family_by_sample) != {
        int(sample_index) for sample_index in ordered_ids
    }:
        raise Evaluation2DValidationError(
            "operator family reveal sample ids differ from withheld truth"
        )
    revealed_families = np.asarray(
        [
            operator_binding.family_by_sample[int(sample_index)]
            for sample_index in ordered_ids
        ]
    )
    if not np.array_equal(scenarios, revealed_families):
        raise Evaluation2DValidationError(
            "withheld truth scenarios differ from the committed family reveal"
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
    depth_rmse = np.sqrt(np.sum(squared_error * normalized_x[None, None, :], axis=2))
    depth_mae = np.sum(np.abs(difference) * normalized_x[None, None, :], axis=2)
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
        "audience": "benchmark_operator_only_after_predictions_locked",
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
            "prediction_lock": {
                "input_manifest_sha256": prediction_lock.input_manifest_sha256,
                "preregistration_sha256": prediction_lock.preregistration_sha256,
                "schema": LOCK_SCHEMA,
                "schema_version": LOCK_SCHEMA_VERSION,
                "sha256": prediction_lock.lock_sha256,
            },
            "operator_manifest": {
                "schema": OPERATOR_MANIFEST_SCHEMA,
                "schema_version": OPERATOR_MANIFEST_SCHEMA_VERSION,
                "sha256": operator_binding.snapshot.sha256,
                "size_bytes": operator_binding.snapshot.size_bytes,
            },
        },
        "run": {
            "adapter_source_sha256": locked_run.adapter_source_sha256,
            "campaign_id": locked_run.campaign_id,
            "checkpoint_sha256": locked_run.checkpoint_sha256,
            "method_id": locked_run.method_id,
            "runtime_sha256": locked_run.runtime_sha256,
            "source_commit": locked_run.source_commit,
            "source_sha256": locked_run.source_sha256,
            "training_seed": locked_run.training_seed,
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
            "predictions_locked": True,
            "public_release_allowed": False,
            "required_next_step": (
                "run the preregistered paired hierarchical comparator with this exact "
                "lock, then create a separately redacted public report"
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


def _publication_parent_identity(path: Path) -> tuple[int, int]:
    try:
        info = os.lstat(path)
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise Evaluation2DPublicationError(
            f"cannot inspect publication parent {path}: {exc}"
        ) from exc
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise Evaluation2DPublicationError(
            f"publication parent must be a real directory: {path}"
        )
    if os.path.normcase(str(resolved)) != os.path.normcase(str(path.absolute())):
        raise Evaluation2DPublicationError(
            f"publication parent must not traverse a symbolic link: {path}"
        )
    return int(info.st_dev), int(info.st_ino)


def _write_all_descriptor(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("zero-byte write while publishing evaluation")
        view = view[written:]


def _read_all_descriptor(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while chunk := os.read(descriptor, 1024 * 1024):
        chunks.append(chunk)
    return b"".join(chunks)


def _publication_signature(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
    )


def _seal_publication_descriptor(descriptor: int) -> None:
    set_publication_descriptor_read_only(descriptor)
    os.fsync(descriptor)


def _verify_new_evaluation_path(
    destination: Path,
    *,
    expected_identity: tuple[int, int],
    expected_parent_identity: tuple[int, int],
) -> None:
    try:
        current = os.lstat(destination)
    except OSError as exc:
        raise Evaluation2DPublicationError(
            f"new evaluation disappeared before writing: {destination}"
        ) from exc
    if (
        not stat.S_ISREG(current.st_mode)
        or stat.S_ISLNK(current.st_mode)
        or (int(current.st_dev), int(current.st_ino)) != expected_identity
        or int(current.st_nlink) != 1
    ):
        raise Evaluation2DPublicationError(
            "new evaluation path was replaced or acquired a hardlink alias"
        )
    if _publication_parent_identity(destination.parent) != expected_parent_identity:
        raise Evaluation2DPublicationError(
            "evaluation publication parent was replaced before writing"
        )


def _stable_evaluation_receipt(
    destination: Path,
    *,
    expected_identity: tuple[int, int],
    expected_parent_identity: tuple[int, int],
    expected_payload: bytes,
) -> Evaluation2DPublicationReceipt:
    """Reopen and twice snapshot the sealed output through one descriptor."""
    descriptor: int | None = None
    try:
        preopen = os.lstat(destination)
        if not stat.S_ISREG(preopen.st_mode) or stat.S_ISLNK(preopen.st_mode):
            raise Evaluation2DPublicationError(
                f"published evaluation is not a regular non-link file: {destination}"
            )
        descriptor = open_verified_publication(destination)
        presealed = os.fstat(descriptor)
        if (int(presealed.st_dev), int(presealed.st_ino)) != expected_identity or int(
            presealed.st_nlink
        ) != 1:
            raise Evaluation2DPublicationError(
                "published evaluation was replaced or acquired a hardlink alias"
            )
        set_publication_descriptor_read_only(descriptor)
        before = os.lstat(destination)
        opened = os.fstat(descriptor)
        if stat.S_IMODE(opened.st_mode) & 0o222:
            raise Evaluation2DPublicationError(
                "published evaluation is not sealed read-only"
            )
        first = _read_all_descriptor(descriptor)
        middle = os.fstat(descriptor)
        if _publication_parent_identity(destination.parent) != expected_parent_identity:
            raise Evaluation2DPublicationError(
                "evaluation publication parent was replaced"
            )
        second = _read_all_descriptor(descriptor)
        after_fd = os.fstat(descriptor)
        after_path = os.lstat(destination)
        signatures = {
            _publication_signature(value)
            for value in (before, opened, middle, after_fd, after_path)
        }
        if (
            len(signatures) != 1
            or first != second
            or second != expected_payload
            or len(second) != int(opened.st_size)
        ):
            raise Evaluation2DPublicationError(
                "published evaluation changed during final descriptor verification"
            )
        digest = hashlib.sha256(second).hexdigest()
        return Evaluation2DPublicationReceipt(destination, digest, len(second))
    except Evaluation2DPublicationError:
        raise
    except OSError as exc:
        raise Evaluation2DPublicationError(
            f"cannot verify published evaluation {destination}: {exc}"
        ) from exc
    finally:
        if descriptor is not None:
            suppress_close_error = sys.exception() is not None
            if suppress_close_error:
                try:
                    set_publication_descriptor_read_only(descriptor)
                except OSError:
                    pass
            close_publication_descriptor(descriptor, suppress_errors=suppress_close_error)


def publish_evaluation_2d_receipt(
    evaluation: Mapping[str, Any], output_path: str | Path
) -> Evaluation2DPublicationReceipt:
    """Exclusively create, seal, and descriptor-verify canonical JSON.

    No cleanup mutates a pathname.  An interrupted or ambiguous publication is
    retained read-only so an operator can inspect it without risking deletion
    of a concurrently substituted file.
    """
    destination = Path(os.path.abspath(output_path))
    ensure_real_directory(
        destination.parent,
        error_type=Evaluation2DPublicationError,
        role="evaluation publication parent",
    )
    parent_identity = _publication_parent_identity(destination.parent)
    if os.path.lexists(destination):
        raise Evaluation2DPublicationError(f"refusing to overwrite {destination}")
    payload = canonical_json_bytes(evaluation)
    descriptor: int | None = None
    identity: tuple[int, int] | None = None
    try:
        try:
            descriptor = open_exclusive_publication(destination)
        except FileExistsError as exc:
            raise Evaluation2DPublicationError(
                f"publication race: refusing to overwrite {destination}"
            ) from exc
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or int(opened.st_nlink) != 1:
            raise Evaluation2DPublicationError(
                "new evaluation output is not a unique regular file"
            )
        identity = int(opened.st_dev), int(opened.st_ino)
        _verify_new_evaluation_path(
            destination,
            expected_identity=identity,
            expected_parent_identity=parent_identity,
        )
        _write_all_descriptor(descriptor, payload)
        os.fsync(descriptor)
        _seal_publication_descriptor(descriptor)
        sealed = os.fstat(descriptor)
        if (
            (int(sealed.st_dev), int(sealed.st_ino)) != identity
            or int(sealed.st_nlink) != 1
            or int(sealed.st_size) != len(payload)
        ):
            raise Evaluation2DPublicationError(
                "evaluation changed before final verification"
            )
    except BaseException:
        if descriptor is not None:
            try:
                _seal_publication_descriptor(descriptor)
            except OSError:
                pass
        raise
    finally:
        if descriptor is not None:
            close_publication_descriptor(
                descriptor, suppress_errors=sys.exception() is not None
            )
    assert identity is not None
    return _stable_evaluation_receipt(
        destination,
        expected_identity=identity,
        expected_parent_identity=parent_identity,
        expected_payload=payload,
    )


def publish_evaluation_2d(evaluation: Mapping[str, Any], output_path: str | Path) -> Path:
    """Publish an evaluation and retain the original path-returning API."""
    requested_path = Path(output_path)
    publish_evaluation_2d_receipt(evaluation, requested_path)
    return requested_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strictly score one PIMSR SOTA 2-D prediction artifact"
    )
    parser.add_argument("--truth", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--observation-manifest", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--operator-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--predictions-lock", required=True, type=Path)
    parser.add_argument("--predictions-lock-sha256", required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--training-seed", required=True, type=int)
    parser.add_argument("--operator-manifest-sha256", required=True)
    parser.add_argument("--truth-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point used by the standalone evaluation script."""
    args = _parser().parse_args(argv)
    evaluation = evaluate_predictions_2d(
        args.truth,
        args.predictions,
        preregistration_path=args.preregistration,
        expected_preregistration_sha256=args.preregistration_sha256,
        predictions_lock_path=args.predictions_lock,
        expected_predictions_lock_sha256=args.predictions_lock_sha256,
        campaign_id=args.campaign_id,
        method_id=args.method_id,
        training_seed=args.training_seed,
        observations_path=args.observations,
        observation_manifest_path=args.observation_manifest,
        runtime_path=args.runtime,
        checkpoint_path=args.checkpoint,
        source_path=args.source,
        operator_manifest_path=args.operator_manifest,
        expected_operator_manifest_sha256=args.operator_manifest_sha256,
        expected_truth_sha256=args.truth_sha256,
    )
    receipt = publish_evaluation_2d_receipt(evaluation, args.output)
    print(f"published {receipt.path} sha256={receipt.sha256} size={receipt.size_bytes}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the wrapper script
    raise SystemExit(main())
