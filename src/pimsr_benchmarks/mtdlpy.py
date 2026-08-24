"""Fail-closed common-retraining adapter for the pinned MTDLPy DinkNet50.

Training and inference are deliberately separate operations.  A training run
can see only the public train/validation datasets and the fixed ImageNet
initialization.  Each held-out campaign is then inferred from the same
immutable seed checkpoint through the observation-only payload emitted by
:mod:`dataset2d_materialization`.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
import platform
import random
import re
import ssl
import stat
import subprocess
import sys
import time
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from pimsr_benchmarks.dataset2d_materialization import (
    OBSERVATION_CHANNEL_ORDER,
    OBSERVATION_SCHEMA,
    PAYLOAD_SCHEMA_VERSION,
)
from pimsr_benchmarks.runner2d import require_file_artifact_unchanged

MTDLPY_REPOSITORY_URL = "https://github.com/Yuan-Chongxin/MTDLPy.git"
MTDLPY_COMMIT = "b01f72a53078a9dc8d452fa53ea5009639d00b04"
MTDLPY_DINKNET_PATH = "func/dinknet.py"
MTDLPY_DINKNET_GIT_BLOB = "5551d6b598f9934db4d4beb17f475c6da36b4a53"
MTDLPY_DINKNET_SHA256 = "838d1271c6987fdac05daf53f2408827d86d493f1fa2ec73a6ecd4753d42ebae"

IMAGENET_RESNET50_V1_URL = "https://download.pytorch.org/models/resnet50-0676ba61.pth"
IMAGENET_RESNET50_V1_SHA256 = (
    "0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a"
)

COMMON_RETRAIN_SEEDS = (101, 102, 103, 104, 105)
INPUT_GRID_SHAPE = (8, 12)
NETWORK_GRID_SHAPE = (32, 32)
OUTPUT_GRID_SHAPE = (64, 48)

ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
WEIGHT_DECAY = 0.0
GRAD_CLIP_NORM = 0.1


@dataclass(frozen=True)
class TrainingRecipe:
    """A named, immutable MTDLPy training schedule selected before scoring."""

    recipe_id: str
    epochs: int
    batch_size: int
    learning_rate: float
    schedule_origin: str


REVIEWED_RECIPE = TrainingRecipe(
    recipe_id="benchmark_reviewed_v1",
    epochs=10,
    batch_size=4,
    learning_rate=1e-4,
    schedule_origin=(
        "preregistered benchmark-native reviewed adapter schedule; "
        "not an MTDLPy upstream default"
    ),
)
UPSTREAM_CONFIG_RECIPE = TrainingRecipe(
    recipe_id="upstream_paramconfig_b01f72a_v1",
    epochs=200,
    batch_size=8,
    learning_rate=1e-8,
    schedule_origin=(
        "verbatim active Epochs, BatchSize and LearnRate values in pinned "
        "MTDLPy ParamConfig.py; best-validation selection and no scheduler "
        "match pinned MT_train.py"
    ),
)
TRAINING_RECIPES = {
    recipe.recipe_id: recipe for recipe in (REVIEWED_RECIPE, UPSTREAM_CONFIG_RECIPE)
}
DEFAULT_RECIPE_ID = REVIEWED_RECIPE.recipe_id

# Backwards-compatible aliases for the reviewed development recipe. Production
# run manifests always carry the explicit recipe id and resolved values.
EPOCHS = REVIEWED_RECIPE.epochs
BATCH_SIZE = REVIEWED_RECIPE.batch_size
LEARNING_RATE = REVIEWED_RECIPE.learning_rate

PREDICTION_SCHEMA = "pimsr-sota-2d-predictions"
PREDICTION_SCHEMA_VERSION = 2
PREDICTION_KEYS = (
    "schema",
    "schema_version",
    "observations_sha256",
    "sample_index",
    "x_cell_centers_m",
    "depth_cell_centers_m",
    "predicted_log10_resistivity",
)
RUNTIME_SCHEMA = "pimsr-mtdlpy-common-retrain-runtime"
RUNTIME_SCHEMA_VERSION = 2
TRAINING_RUNTIME_SCHEMA = "pimsr-mtdlpy-common-retrain-training-runtime"
TRAINING_RUNTIME_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA = "pimsr-mtdlpy-common-retrain-checkpoint"
CHECKPOINT_SCHEMA_VERSION = 1
DEPENDENCY_CLOSURE_SCHEMA = "pimsr-mtdlpy-dependency-closure"
DEPENDENCY_CLOSURE_SCHEMA_VERSION = 3
MAX_CHECKPOINT_SIZE_BYTES = 1024 * 1024 * 1024

_COMMON_BACKEND_RUNTIME_KEYS = frozenset(
    {
        "python",
        "platform",
        "torch",
        "torchvision",
        "torch_cuda_build",
        "cuda_available",
        "device",
        "cuda_device_name",
        "peak_cuda_memory_bytes",
        "preprocessing_wall_time_s",
        "model_initialization_wall_time_s",
        "backend_wall_time_s",
    }
)
_TRAINING_BACKEND_RUNTIME_KEYS = _COMMON_BACKEND_RUNTIME_KEYS | {"training_wall_time_s"}
_INFERENCE_BACKEND_RUNTIME_KEYS = _COMMON_BACKEND_RUNTIME_KEYS | {"inference_wall_time_s"}

CHECKPOINT_KEYS = frozenset(
    {
        "checkpoint_schema",
        "checkpoint_schema_version",
        "method",
        "track",
        "seed",
        "recipe_id",
        "model_state",
        "training_config",
        "preprocessing",
        "training_summary",
        "training_runtime",
        "source",
        "imagenet_weights",
        "dataset_identities",
        "training_geometry",
        "dependency_closure",
        "truth_keys_accepted",
        "contains_truth",
        "contains_observation_campaign",
    }
)

_OBSERVATION_VALUE_KEYS = (
    "observed_log10_rho_te",
    "observed_phase_te_degrees",
    "observed_log10_rho_tm",
    "observed_phase_tm_degrees",
)
_OBSERVATION_FLOOR_KEYS = (
    "declared_evaluation_floor_log10_rho_te",
    "declared_evaluation_floor_phase_te_degrees",
    "declared_evaluation_floor_log10_rho_tm",
    "declared_evaluation_floor_phase_tm_degrees",
)
_OBSERVATION_KEYS = (
    "schema",
    "schema_version",
    "sample_index",
    "frequency_hz",
    "station_x_m",
    "x_cell_centers_m",
    "depth_cell_centers_m",
    "observation_channel_order",
    *_OBSERVATION_VALUE_KEYS,
    *_OBSERVATION_FLOOR_KEYS,
    "valid_mask",
)


class MTDLPyAdapterError(RuntimeError):
    """Raised when an MTDLPy run cannot prove its comparison contract."""


def training_recipe(recipe_id: str) -> TrainingRecipe:
    """Resolve one closed-set recipe id without accepting free hyperparameters."""
    if not isinstance(recipe_id, str) or recipe_id not in TRAINING_RECIPES:
        raise ValueError(f"recipe_id must be one of {tuple(sorted(TRAINING_RECIPES))}")
    return TRAINING_RECIPES[recipe_id]


@dataclass(frozen=True)
class TrainingSplit:
    """In-memory train/validation arrays from one validated schema-v2 HDF5."""

    observations: np.ndarray
    targets: np.ndarray
    sample_index: np.ndarray
    generator_seed: int
    frequencies: np.ndarray
    station_x: np.ndarray
    x_grid: np.ndarray
    depth_grid: np.ndarray
    provenance: Mapping[str, object]


@dataclass(frozen=True)
class HeldoutObservations:
    """Truth-free, materialized test observations and declared score floors."""

    observations: np.ndarray
    evaluation_floors: np.ndarray
    sample_index: np.ndarray
    frequencies: np.ndarray
    station_x: np.ndarray
    x_grid: np.ndarray
    depth_grid: np.ndarray
    provenance: Mapping[str, object]


@dataclass(frozen=True)
class TrainingOutcome:
    """Backend result before immutable publication."""

    state_dict: Mapping[str, Any]
    training_summary: Mapping[str, object]
    runtime: Mapping[str, object]


@dataclass(frozen=True)
class InferenceOutcome:
    """Truth-free backend inference result before immutable publication."""

    predicted_log10_resistivity: np.ndarray
    runtime: Mapping[str, object]


@dataclass(frozen=True)
class _ArtifactSnapshot:
    path: Path
    payload: bytes
    sha256: str
    device: int
    inode: int

    @property
    def size_bytes(self) -> int:
        return len(self.payload)


def _path_is_link(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction()) if callable(is_junction) else False


def _stat_identity(info: os.stat_result) -> tuple[int, int, int, int]:
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_size),
        int(info.st_mtime_ns),
    )


def _snapshot_regular_file(
    path: str | Path,
    *,
    role: str,
    expected_sha256: str | None = None,
    max_size_bytes: int | None = None,
) -> _ArtifactSnapshot:
    requested = Path(os.path.abspath(os.fspath(path)))
    if expected_sha256 is not None and (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise MTDLPyAdapterError(
            f"expected {role} SHA-256 must be 64 lowercase hexadecimal characters"
        )
    try:
        path_before = requested.lstat()
    except OSError as exc:
        raise MTDLPyAdapterError(f"cannot inspect {role}: {requested}") from exc
    if _path_is_link(requested):
        raise MTDLPyAdapterError(f"{role} must be a non-symlink regular file")
    if not stat.S_ISREG(path_before.st_mode):
        raise MTDLPyAdapterError(f"{role} must be a regular file")
    if max_size_bytes is not None and int(path_before.st_size) > max_size_bytes:
        raise MTDLPyAdapterError(f"{role} exceeds the fail-closed size limit")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(requested, flags)
    except OSError as exc:
        raise MTDLPyAdapterError(f"cannot open {role} without following links") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise MTDLPyAdapterError(f"{role} must be a regular file")
        if _stat_identity(path_before) != _stat_identity(opened):
            raise MTDLPyAdapterError(f"{role} changed before it was opened")
        if max_size_bytes is not None and int(opened.st_size) > max_size_bytes:
            raise MTDLPyAdapterError(f"{role} exceeds the fail-closed size limit")
        try:
            resolved = requested.resolve(strict=True)
            resolved_info = resolved.stat(follow_symlinks=False)
        except OSError as exc:
            raise MTDLPyAdapterError(f"cannot resolve opened {role}") from exc
        if _path_is_link(resolved) or _stat_identity(resolved_info) != _stat_identity(
            opened
        ):
            raise MTDLPyAdapterError(f"{role} path does not identify the opened file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
        after_descriptor = os.fstat(descriptor)
        try:
            path_after = requested.lstat()
        except OSError as exc:
            raise MTDLPyAdapterError(
                f"{role} path disappeared while it was read"
            ) from exc
        if (
            _path_is_link(requested)
            or _stat_identity(opened) != _stat_identity(after_descriptor)
            or _stat_identity(opened) != _stat_identity(path_after)
            or len(payload) != int(opened.st_size)
        ):
            raise MTDLPyAdapterError(f"{role} changed while it was read")
    finally:
        os.close(descriptor)
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise MTDLPyAdapterError(f"{role} SHA-256 differs from the pinned digest")
    return _ArtifactSnapshot(
        path=resolved,
        payload=payload,
        sha256=digest,
        device=int(opened.st_dev),
        inode=int(opened.st_ino),
    )


def _snapshot_core(snapshot: _ArtifactSnapshot) -> dict[str, object]:
    return {
        "path": str(snapshot.path),
        "sha256": snapshot.sha256,
        "size_bytes": snapshot.size_bytes,
    }


def _artifact_core(value: Mapping[str, object]) -> dict[str, object]:
    """Select the exact three fields understood by the mutation guard."""
    try:
        return {
            "path": value["path"],
            "sha256": value["sha256"],
            "size_bytes": value["size_bytes"],
        }
    except KeyError as exc:
        raise MTDLPyAdapterError("artifact provenance is incomplete") from exc


def _run_git(repo: Path, *arguments: str, binary: bool = False) -> str | bytes:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repo,
            check=True,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise MTDLPyAdapterError(
            f"failed to verify pinned MTDLPy repository with git {' '.join(arguments)}"
        ) from exc
    if binary:
        return result.stdout
    try:
        return result.stdout.decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise MTDLPyAdapterError("git returned non-UTF-8 repository metadata") from exc


def verify_pinned_repository(path: str | Path) -> dict[str, object]:
    """Prove URL, commit, cleanliness and exact upstream DinkNet50 source."""
    repo = Path(path).resolve(strict=True)
    if not repo.is_dir():
        raise NotADirectoryError(f"MTDLPy repository is not a directory: {repo}")

    top = Path(str(_run_git(repo, "rev-parse", "--show-toplevel"))).resolve(strict=True)
    if not os.path.samefile(repo, top):
        raise MTDLPyAdapterError(
            f"MTDLPy path must be the repository root, got {repo} with root {top}"
        )
    commit = str(_run_git(repo, "rev-parse", "--verify", "HEAD^{commit}"))
    if commit != MTDLPY_COMMIT:
        raise MTDLPyAdapterError(
            f"MTDLPy HEAD is {commit}; required pinned commit is {MTDLPY_COMMIT}"
        )

    remotes = str(_run_git(repo, "remote", "get-url", "--all", "origin")).splitlines()
    if remotes != [MTDLPY_REPOSITORY_URL]:
        raise MTDLPyAdapterError(
            "MTDLPy origin URL must be exactly "
            f"{MTDLPY_REPOSITORY_URL!r}, got {remotes!r}"
        )
    status = str(_run_git(repo, "status", "--porcelain=v1", "--untracked-files=all"))
    if status:
        raise MTDLPyAdapterError("MTDLPy repository must have a clean worktree")

    tree_entry = str(
        _run_git(repo, "ls-tree", "--full-tree", "HEAD", "--", MTDLPY_DINKNET_PATH)
    )
    expected_entry = f"100644 blob {MTDLPY_DINKNET_GIT_BLOB}\t{MTDLPY_DINKNET_PATH}"
    if tree_entry != expected_entry:
        raise MTDLPyAdapterError(
            "pinned MTDLPy DinkNet source tree entry is not the reviewed blob"
        )
    blob = _run_git(
        repo,
        "cat-file",
        "blob",
        f"{MTDLPY_COMMIT}:{MTDLPY_DINKNET_PATH}",
        binary=True,
    )
    assert isinstance(blob, bytes)
    blob_sha256 = hashlib.sha256(blob).hexdigest()
    if blob_sha256 != MTDLPY_DINKNET_SHA256:
        raise MTDLPyAdapterError("pinned MTDLPy DinkNet Git blob SHA-256 changed")

    source = repo / MTDLPY_DINKNET_PATH
    source_identity = _snapshot_core(
        _snapshot_regular_file(
            source,
            role="pinned MTDLPy DinkNet source",
            expected_sha256=MTDLPY_DINKNET_SHA256,
        )
    )
    if source_identity["sha256"] != MTDLPY_DINKNET_SHA256:
        raise MTDLPyAdapterError(
            "checked-out MTDLPy DinkNet source bytes differ from the reviewed blob"
        )
    return {
        "path": str(repo),
        "origin_url": MTDLPY_REPOSITORY_URL,
        "commit": commit,
        "clean_worktree": True,
        "dinknet_source": source_identity,
        "dinknet_git_blob_sha1": MTDLPY_DINKNET_GIT_BLOB,
        "dinknet_git_blob_sha256": blob_sha256,
    }


def validate_local_imagenet_weights(
    path: str | Path,
    declared_sha256: str,
) -> dict[str, object]:
    """Require the reviewed local ResNet50 V1 weights; never fetch or cache-fallback."""
    if declared_sha256 != IMAGENET_RESNET50_V1_SHA256:
        raise MTDLPyAdapterError(
            "declared ImageNet weights SHA-256 is not the preregistered ResNet50 V1 hash"
        )
    identity = _snapshot_core(
        _snapshot_regular_file(
            path,
            role="local ImageNet weights",
            expected_sha256=declared_sha256,
        )
    )
    return {
        **identity,
        "source_url": IMAGENET_RESNET50_V1_URL,
        "artifact": "torchvision ResNet50 IMAGENET1K_V1",
    }


def _axis(values: np.ndarray, *, name: str, positive: bool = False) -> np.ndarray:
    result = np.asarray(values)
    if result.dtype != np.dtype("<f8") or result.ndim != 1 or result.size == 0:
        raise MTDLPyAdapterError(
            f"{name} must be a non-empty little-endian float64 vector"
        )
    if not np.isfinite(result).all() or np.any(np.diff(result) <= 0):
        raise MTDLPyAdapterError(f"{name} must be finite and strictly increasing")
    if positive and np.any(result <= 0):
        raise MTDLPyAdapterError(f"{name} must be positive")
    return result.copy()


def _require_geometry_shape(
    frequencies: np.ndarray,
    station_x: np.ndarray,
    x_grid: np.ndarray,
    depth_grid: np.ndarray,
    *,
    where: str,
) -> None:
    if (frequencies.size, station_x.size) != INPUT_GRID_SHAPE:
        raise MTDLPyAdapterError(
            f"{where} observation geometry must be exactly {INPUT_GRID_SHAPE}"
        )
    if (depth_grid.size, x_grid.size) != OUTPUT_GRID_SHAPE:
        raise MTDLPyAdapterError(
            f"{where} target geometry must be exactly {OUTPUT_GRID_SHAPE}"
        )
    if station_x[0] < x_grid[0] or station_x[-1] > x_grid[-1]:
        raise MTDLPyAdapterError(f"{where} stations must lie inside the x grid")


def load_training_split(path: str | Path, *, role: str) -> TrainingSplit:
    """Load train/validation truth only after the complete producer validation."""
    snapshot = _snapshot_regular_file(path, role=role)
    artifact = _snapshot_core(snapshot)
    from pimsr_inversion.contracts2d import validate_dataset2d

    try:
        with h5py.File(io.BytesIO(snapshot.payload), "r") as h5:
            contract = validate_dataset2d(h5)
            generator_seed = int(np.asarray(h5.attrs["generator_seed"]).item())
            observations = np.stack(
                [
                    h5["obs_mt_log10_rho"][:],
                    h5["obs_mt_phase"][:],
                    h5["obs_mt_log10_rho_tm"][:],
                    h5["obs_mt_phase_tm"][:],
                ],
                axis=1,
            ).astype(np.float32, copy=False)
            targets = np.asarray(h5["target_log10_res"][:], dtype=np.float32)
            sample_index = np.asarray(h5["sample_index"][:], dtype=np.int64)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise MTDLPyAdapterError(f"cannot load {role} from pinned bytes: {exc}") from exc

    frequencies = np.asarray(contract.frequencies, dtype="<f8")
    station_x = np.asarray(contract.station_x, dtype="<f8")
    x_grid = np.asarray(contract.x_grid, dtype="<f8")
    depth_grid = np.asarray(contract.depth_grid, dtype="<f8")
    _require_geometry_shape(
        frequencies,
        station_x,
        x_grid,
        depth_grid,
        where=role,
    )
    if observations.shape != (sample_index.size, 4, *INPUT_GRID_SHAPE):
        raise MTDLPyAdapterError(f"{role} has an unexpected observation tensor shape")
    if targets.shape != (sample_index.size, *OUTPUT_GRID_SHAPE):
        raise MTDLPyAdapterError(f"{role} has an unexpected target tensor shape")
    if not np.isfinite(observations).all() or not np.isfinite(targets).all():
        raise MTDLPyAdapterError(f"{role} arrays must be finite")
    return TrainingSplit(
        observations=observations,
        targets=targets,
        sample_index=sample_index,
        generator_seed=generator_seed,
        frequencies=frequencies,
        station_x=station_x,
        x_grid=x_grid,
        depth_grid=depth_grid,
        provenance=artifact,
    )


def _scalar_string(value: np.ndarray, name: str) -> str:
    if value.ndim != 0 or value.dtype.kind != "U":
        raise MTDLPyAdapterError(f"{name} must be a scalar Unicode array")
    return str(value.item())


def _check_npz_member_names(payload: bytes, *, path: Path) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
            names = archive.namelist()
    except (OSError, zipfile.BadZipFile) as exc:
        raise MTDLPyAdapterError(f"invalid observation NPZ: {path}") from exc
    expected = [f"{name}.npy" for name in _OBSERVATION_KEYS]
    if names != expected or len(names) != len(set(names)):
        raise MTDLPyAdapterError(
            "observation NPZ members must exactly match the ordered truth-free contract"
        )


def load_heldout_observations(path: str | Path) -> HeldoutObservations:
    """Parse the exact truth-free NPZ contract without guessing missing fields."""
    snapshot = _snapshot_regular_file(path, role="held-out observation payload")
    artifact = _snapshot_core(snapshot)
    artifact_path = snapshot.path
    _check_npz_member_names(snapshot.payload, path=artifact_path)
    try:
        with np.load(io.BytesIO(snapshot.payload), allow_pickle=False) as payload:
            if tuple(payload.files) != _OBSERVATION_KEYS:
                raise MTDLPyAdapterError(
                    "observation NPZ keys are not in the canonical contract order"
                )
            arrays = {name: np.asarray(payload[name]).copy() for name in payload.files}
    except (OSError, ValueError) as exc:
        raise MTDLPyAdapterError(f"cannot load observation NPZ: {artifact_path}") from exc
    if _scalar_string(arrays["schema"], "schema") != OBSERVATION_SCHEMA:
        raise MTDLPyAdapterError("unsupported held-out observation schema")
    schema_version = arrays["schema_version"]
    if (
        schema_version.shape != ()
        or schema_version.dtype != np.dtype("<i8")
        or int(schema_version) != PAYLOAD_SCHEMA_VERSION
    ):
        raise MTDLPyAdapterError("unsupported held-out observation schema version")

    order = arrays["observation_channel_order"]
    if (
        order.dtype.kind != "U"
        or order.shape != (4,)
        or tuple(order.tolist()) != (OBSERVATION_CHANNEL_ORDER)
    ):
        raise MTDLPyAdapterError("held-out observation channel order is not canonical")

    sample_index = arrays["sample_index"]
    if (
        sample_index.dtype != np.dtype("<i8")
        or sample_index.ndim != 1
        or sample_index.size == 0
        or np.any(sample_index < 0)
        or np.unique(sample_index).size != sample_index.size
    ):
        raise MTDLPyAdapterError(
            "held-out sample_index must contain unique non-negative opaque IDs"
        )
    frequencies = _axis(arrays["frequency_hz"], name="frequency_hz", positive=True)
    station_x = _axis(arrays["station_x_m"], name="station_x_m")
    x_grid = _axis(arrays["x_cell_centers_m"], name="x_cell_centers_m")
    depth_grid = _axis(
        arrays["depth_cell_centers_m"],
        name="depth_cell_centers_m",
        positive=True,
    )
    _require_geometry_shape(
        frequencies,
        station_x,
        x_grid,
        depth_grid,
        where="held-out observations",
    )

    shape = (sample_index.size, *INPUT_GRID_SHAPE)
    values: list[np.ndarray] = []
    floors: list[np.ndarray] = []
    for name in _OBSERVATION_VALUE_KEYS:
        array = arrays[name]
        if array.dtype != np.dtype("<f4") or array.shape != shape:
            raise MTDLPyAdapterError(f"{name} must be float32 with shape {shape}")
        if not np.isfinite(array).all():
            raise MTDLPyAdapterError(f"{name} must be finite")
        if "phase" in name and np.any((array < 0.0) | (array >= 180.0)):
            raise MTDLPyAdapterError(f"{name} violates the [0, 180) convention")
        values.append(array)
    for name in _OBSERVATION_FLOOR_KEYS:
        array = arrays[name]
        if array.dtype != np.dtype("<f4") or array.shape != shape:
            raise MTDLPyAdapterError(f"{name} must be float32 with shape {shape}")
        if not np.isfinite(array).all() or np.any(array <= 0):
            raise MTDLPyAdapterError(f"{name} must be finite and strictly positive")
        floors.append(array)

    mask = arrays["valid_mask"]
    expected_mask_shape = (sample_index.size, 4, *INPUT_GRID_SHAPE)
    if mask.dtype != np.dtype(bool) or mask.shape != expected_mask_shape:
        raise MTDLPyAdapterError(
            f"valid_mask must be bool with shape {expected_mask_shape}"
        )
    if not bool(mask.all()):
        raise MTDLPyAdapterError(
            "MTDLPy common retraining does not impute or interpolate missing data; "
            "valid_mask must be all true"
        )
    return HeldoutObservations(
        observations=np.stack(values, axis=1),
        evaluation_floors=np.stack(floors, axis=1),
        sample_index=sample_index,
        frequencies=frequencies,
        station_x=station_x,
        x_grid=x_grid,
        depth_grid=depth_grid,
        provenance=artifact,
    )


def resize_bilinear_half_pixel(
    values: np.ndarray,
    output_shape: tuple[int, int],
    *,
    _output_dtype: np.dtype[Any] | type[np.floating[Any]] = np.float32,
) -> np.ndarray:
    """Deterministic CPU bilinear resize with a fixed half-pixel convention."""
    array = np.asarray(values)
    if array.ndim < 2 or array.shape[-2] < 1 or array.shape[-1] < 1:
        raise ValueError("resize input must have two non-empty spatial dimensions")
    if not np.isfinite(array).all():
        raise ValueError(
            "resize input must be finite; missing values are not interpolated"
        )
    out_h, out_w = output_shape
    if out_h < 1 or out_w < 1:
        raise ValueError("resize output dimensions must be positive")
    in_h, in_w = array.shape[-2:]

    y = (np.arange(out_h, dtype=np.float64) + 0.5) * in_h / out_h - 0.5
    x = (np.arange(out_w, dtype=np.float64) + 0.5) * in_w / out_w - 0.5
    y = np.clip(y, 0.0, in_h - 1.0)
    x = np.clip(x, 0.0, in_w - 1.0)
    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    y1 = np.minimum(y0 + 1, in_h - 1)
    x1 = np.minimum(x0 + 1, in_w - 1)
    wy = y - y0
    wx = x - x0

    work = array.astype(np.float64, copy=False)
    row_shape = (1,) * (work.ndim - 2) + (out_h, 1)
    col_shape = (1,) * (work.ndim - 1) + (out_w,)
    rows = np.take(work, y0, axis=-2) * (1.0 - wy).reshape(row_shape)
    rows += np.take(work, y1, axis=-2) * wy.reshape(row_shape)
    result = np.take(rows, x0, axis=-1) * (1.0 - wx).reshape(col_shape)
    result += np.take(rows, x1, axis=-1) * wx.reshape(col_shape)
    result = result.astype(_output_dtype)
    if not np.isfinite(result).all():
        raise MTDLPyAdapterError("bilinear preprocessing produced non-finite values")
    return result


def _resize_log10_resistivity(
    values: np.ndarray,
    output_shape: tuple[int, int],
) -> np.ndarray:
    """Match upstream's linear-resistivity resize followed by ``log10``."""
    log10_values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(log10_values).all():
        raise ValueError("log10 resistivity values must be finite")
    try:
        with np.errstate(over="raise", under="ignore", invalid="raise"):
            linear = np.power(10.0, log10_values)
    except FloatingPointError as exc:
        raise MTDLPyAdapterError(
            "log10 resistivity cannot be represented in linear float64"
        ) from exc
    if not np.isfinite(linear).all() or np.any(linear <= 0.0):
        raise MTDLPyAdapterError(
            "log10 resistivity cannot be represented as positive linear values"
        )
    resized = resize_bilinear_half_pixel(
        linear,
        output_shape,
        _output_dtype=np.float64,
    )
    result = np.log10(resized).astype(np.float32)
    if not np.isfinite(result).all():
        raise MTDLPyAdapterError(
            "linear-resistivity resize followed by log10 produced non-finite values"
        )
    return result


def _preprocess_observations(values: np.ndarray) -> np.ndarray:
    """Match upstream scale-aware resize and final transpose for TE/TM data."""
    array = np.asarray(values)
    if array.ndim != 4 or array.shape[1] != len(OBSERVATION_CHANNEL_ORDER):
        raise MTDLPyAdapterError(
            "MTDLPy observations must have sample, four-channel, frequency, station axes"
        )
    resized = np.empty(
        (array.shape[0], array.shape[1], *NETWORK_GRID_SHAPE),
        dtype=np.float32,
    )
    for channel in (0, 2):
        resized[:, channel] = _resize_log10_resistivity(
            array[:, channel], NETWORK_GRID_SHAPE
        )
    for channel in (1, 3):
        resized[:, channel] = resize_bilinear_half_pixel(
            array[:, channel], NETWORK_GRID_SHAPE
        )
    return np.ascontiguousarray(np.swapaxes(resized, -2, -1), dtype=np.float32)


def _require_same_geometry(
    left: TrainingSplit | HeldoutObservations,
    right: TrainingSplit | HeldoutObservations,
    *,
    where: str,
) -> None:
    for name in ("frequencies", "station_x", "x_grid", "depth_grid"):
        a = getattr(left, name)
        b = getattr(right, name)
        if a.shape != b.shape or not np.array_equal(a, b):
            raise MTDLPyAdapterError(f"{where} have different {name} axes")


def _require_training_disjoint(
    train: TrainingSplit,
    validation: TrainingSplit,
) -> None:
    if (
        train.generator_seed == validation.generator_seed
        and np.intersect1d(train.sample_index, validation.sample_index).size
    ):
        raise MTDLPyAdapterError(
            "train and validation (generator_seed, sample_index) identities overlap"
        )
    identities = [entry.provenance["sha256"] for entry in (train, validation)]
    if len(set(identities)) != 2:
        raise MTDLPyAdapterError("train and validation artifacts must differ")


def _training_config(seed: int, recipe: TrainingRecipe) -> dict[str, object]:
    return {
        "campaign_seeds": list(COMMON_RETRAIN_SEEDS),
        "recipe_id": recipe.recipe_id,
        "seed": seed,
        "epochs": recipe.epochs,
        "batch_size": recipe.batch_size,
        "optimizer": {
            "name": "Adam",
            "learning_rate": recipe.learning_rate,
            "betas": list(ADAM_BETAS),
            "eps": ADAM_EPS,
            "weight_decay": WEIGHT_DECAY,
        },
        "scheduler": None,
        "early_stopping": None,
        "gradient_clip_max_norm": GRAD_CLIP_NORM,
        "loss": "mean_squared_error_mean",
        "checkpoint_selection": "lowest validation MSE; strict less-than; first tie",
        "normalization": "none",
        "schedule_origin": recipe.schedule_origin,
    }


def _preprocessing_contract() -> dict[str, object]:
    return {
        "input_channel_order": list(OBSERVATION_CHANNEL_ORDER),
        "observation_axis_order": ["sample", "channel", "frequency", "station"],
        "network_spatial_axis_order": ["station_resampled", "frequency_resampled"],
        "transpose_observations": True,
        "observation_transform_order": {
            "apparent_resistivity": (
                "pow10_to_linear_then_bilinear_resize_then_log10_then_transpose"
            ),
            "phase": "bilinear_resize_then_transpose",
        },
        "input_units": ["log10_ohm_m", "degree", "log10_ohm_m", "degree"],
        "observation_resize": "8x12_to_32x32",
        "observation_resize_origin": (
            "benchmark-native geometry adaptation preserving upstream transform order"
        ),
        "training_target_resize": ("pow10_to_linear_then_64x48_to_32x32_then_log10"),
        "training_target_axis_order": ["sample", "depth", "x"],
        "prediction_resize": "pow10_to_linear_then_32x32_to_64x48_then_log10",
        "prediction_resize_origin": (
            "benchmark-native evaluation-grid adaptation; upstream does not define "
            "this inverse regridding"
        ),
        "interpolation": {
            "kind": "bilinear",
            "coordinate_transform": "half_pixel",
            "boundary": "clamp_to_edge",
            "implementation": "pimsr_benchmarks.mtdlpy.resize_bilinear_half_pixel",
            "dtype": "float64_accumulation_float32_output",
            "antialias": False,
        },
        "missing_data_policy": "reject_if_valid_mask_is_not_all_true",
        "phase_domain_adaptation": (
            "retain benchmark canonical [0,180) phases; do not apply the upstream "
            "loader's [0,90] sample rejection"
        ),
        "upstream_pipeline_claim": "semantic preprocessing order only, not byte-exact",
        "test_tuning": False,
        "evaluation_floors_used_as_model_input": False,
    }


def _module_source_artifact(module_name: str) -> dict[str, object]:
    module = sys.modules.get(module_name)
    if module is None:
        module = __import__(module_name, fromlist=["__name__"])
    source = getattr(module, "__file__", None)
    if not isinstance(source, str):
        raise MTDLPyAdapterError(f"cannot identify source for dependency {module_name}")
    path = Path(source)
    if path.suffix == ".pyc":
        path = path.with_suffix(".py")
    return _snapshot_core(
        _snapshot_regular_file(path, role=f"dependency source {module_name}")
    )


def _benchmark_runner_source_path() -> Path:
    return Path(__file__).resolve().parents[2] / "scripts" / "run_mtdlpy_common.py"


def _validated_runner_source_artifact(path: str | Path) -> dict[str, object]:
    snapshot = _snapshot_regular_file(path, role="MTDLPy CLI runner source")
    try:
        expected = _benchmark_runner_source_path().resolve(strict=True)
    except OSError as exc:
        raise MTDLPyAdapterError(
            "cannot identify benchmark scripts/run_mtdlpy_common.py"
        ) from exc
    if snapshot.path != expected:
        raise MTDLPyAdapterError(
            "runner_source must be the exact benchmark scripts/run_mtdlpy_common.py"
        )
    expected_info = expected.stat(follow_symlinks=False)
    if (snapshot.device, snapshot.inode) != (
        int(expected_info.st_dev),
        int(expected_info.st_ino),
    ):
        raise MTDLPyAdapterError(
            "runner_source identity changed during dependency closure capture"
        )
    return _snapshot_core(snapshot)


def _dependency_closure(
    *,
    repository: Mapping[str, object],
    weights: Mapping[str, object],
    runner_source: str | Path | None,
) -> dict[str, object]:
    cli_entrypoint_source_included = runner_source is not None
    local_sources: dict[str, Mapping[str, object]] = {
        "adapter": _snapshot_core(
            _snapshot_regular_file(__file__, role="MTDLPy adapter source")
        ),
        "dataset2d_materialization": _module_source_artifact(
            "pimsr_benchmarks.dataset2d_materialization"
        ),
        "runner2d": _module_source_artifact("pimsr_benchmarks.runner2d"),
        "pimsr_inversion_contracts2d": _module_source_artifact(
            "pimsr_inversion.contracts2d"
        ),
        "upstream_dinknet": repository["dinknet_source"],
    }
    if runner_source is not None:
        local_sources["cli_runner"] = _validated_runner_source_artifact(runner_source)
    packages: dict[str, str] = {}
    for distribution in ("h5py", "numpy", "torch", "torchvision", "pimsr-inversion"):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise MTDLPyAdapterError(
                f"required dependency distribution is missing: {distribution}"
            ) from exc
    return {
        "schema": DEPENDENCY_CLOSURE_SCHEMA,
        "schema_version": DEPENDENCY_CLOSURE_SCHEMA_VERSION,
        "evidence_scope": (
            "direct_python_source_artifacts_and_distribution_version_strings"
        ),
        "python": platform.python_version(),
        "packages": packages,
        "local_source_artifacts": {
            key: dict(value) for key, value in local_sources.items()
        },
        "fixed_imagenet_weights": dict(weights),
        "cli_entrypoint_source_included": cli_entrypoint_source_included,
        "required_local_python_source_artifacts_recorded": (
            cli_entrypoint_source_included
        ),
        "native_binary_environment_complete": False,
    }


def _canonical_object_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _strict_metadata_equal(actual: object, expected: object) -> bool:
    """Compare JSON-like evidence without Python's bool/int/float coercions."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, Mapping):
        return set(actual) == set(expected) and all(
            _strict_metadata_equal(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(
            _strict_metadata_equal(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return bool(actual == expected)


def _configure_determinism(torch: Any, seed: int) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def _load_upstream_model(
    repository: Mapping[str, object],
    weights: Mapping[str, object],
    *,
    torch: Any,
) -> tuple[Any, Any]:
    """Instantiate exact upstream DinkNet50 with an injected local ResNet50."""
    try:
        import torchvision
    except ImportError as exc:
        raise MTDLPyAdapterError(
            "torchvision is required to instantiate pinned MTDLPy DinkNet50"
        ) from exc

    weight_snapshot = _snapshot_regular_file(
        str(weights["path"]),
        role="local ImageNet weights",
        expected_sha256=str(weights["sha256"]),
    )
    if _snapshot_core(weight_snapshot) != _artifact_core(weights):
        raise MTDLPyAdapterError(
            "local ImageNet weights identity differs from the validated artifact"
        )
    state = torch.load(
        io.BytesIO(weight_snapshot.payload), map_location="cpu", weights_only=True
    )
    if not isinstance(state, Mapping):
        raise MTDLPyAdapterError("ImageNet weights root must be a plain state dictionary")

    resnet = torchvision.models.resnet50(weights=None)
    try:
        resnet.load_state_dict(state, strict=True)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise MTDLPyAdapterError(
            "local ImageNet artifact is not an exact torchvision ResNet50 state dict"
        ) from exc

    source_artifact = repository["dinknet_source"]
    if not isinstance(source_artifact, Mapping):
        raise MTDLPyAdapterError("pinned MTDLPy DinkNet source identity is missing")
    source_snapshot = _snapshot_regular_file(
        str(source_artifact["path"]),
        role="pinned MTDLPy DinkNet source",
        expected_sha256=str(source_artifact["sha256"]),
    )
    if _snapshot_core(source_snapshot) != _artifact_core(source_artifact):
        raise MTDLPyAdapterError(
            "pinned MTDLPy DinkNet source identity differs from validation"
        )
    module_name = "_pimsr_pinned_mtdlpy_dinknet"
    spec = importlib.util.spec_from_loader(
        module_name, loader=None, origin=str(source_snapshot.path)
    )
    if spec is None:
        raise MTDLPyAdapterError("cannot load pinned MTDLPy DinkNet source")
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(source_snapshot.path)
    original_ssl_context = ssl._create_default_https_context
    original_dont_write_bytecode = sys.dont_write_bytecode
    try:
        # Compile and execute the descriptor-pinned bytes.  A pathname loader
        # could otherwise reopen a different inode after provenance capture.
        sys.dont_write_bytecode = True
        code = compile(
            source_snapshot.payload,
            str(source_snapshot.path),
            "exec",
            dont_inherit=True,
        )
        exec(code, module.__dict__)  # noqa: S102 - exact descriptor-pinned source bytes
    except (OSError, SyntaxError, TypeError) as exc:
        raise MTDLPyAdapterError("cannot execute pinned MTDLPy DinkNet source") from exc
    finally:
        sys.dont_write_bytecode = original_dont_write_bytecode
        ssl._create_default_https_context = original_ssl_context

    calls = 0

    def local_resnet50(*args: object, **kwargs: object) -> Any:
        nonlocal calls
        calls += 1
        if args or kwargs != {"pretrained": True}:
            raise MTDLPyAdapterError(
                "pinned DinkNet50 changed its expected pretrained ResNet50 call"
            )
        if calls != 1:
            raise MTDLPyAdapterError("DinkNet50 requested ResNet50 more than once")
        return resnet

    original_resnet50 = module.models.resnet50
    try:
        module.models.resnet50 = local_resnet50
        model = module.DinkNet50(num_classes=1, num_channels=4)
    finally:
        module.models.resnet50 = original_resnet50
    if calls != 1:
        raise MTDLPyAdapterError("DinkNet50 did not consume the injected local weights")
    require_file_artifact_unchanged(
        _snapshot_core(source_snapshot), role="pinned MTDLPy DinkNet source"
    )
    require_file_artifact_unchanged(
        _snapshot_core(weight_snapshot), role="local ImageNet weights"
    )
    return model, torchvision


def _train_model(
    train: TrainingSplit,
    validation: TrainingSplit,
    repository: Mapping[str, object],
    weights: Mapping[str, object],
    *,
    seed: int,
    device_name: str,
    recipe: TrainingRecipe,
) -> TrainingOutcome:
    backend_start = time.perf_counter()
    import torch
    from torch.nn import functional
    from torch.utils.data import DataLoader, TensorDataset

    _configure_determinism(torch, seed)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise MTDLPyAdapterError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    preprocessing_start = time.perf_counter()
    observations_train = _preprocess_observations(train.observations)
    observations_validation = _preprocess_observations(validation.observations)
    targets_train = _resize_log10_resistivity(train.targets, NETWORK_GRID_SHAPE)
    targets_validation = _resize_log10_resistivity(validation.targets, NETWORK_GRID_SHAPE)
    preprocessing_wall_time_s = time.perf_counter() - preprocessing_start

    initialization_start = time.perf_counter()
    model, torchvision = _load_upstream_model(repository, weights, torch=torch)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=recipe.learning_rate,
        betas=ADAM_BETAS,
        eps=ADAM_EPS,
        weight_decay=WEIGHT_DECAY,
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(observations_train),
            torch.from_numpy(targets_train[:, None]),
        ),
        batch_size=recipe.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )
    validation_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(observations_validation),
            torch.from_numpy(targets_validation[:, None]),
        ),
        batch_size=recipe.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    model_initialization_wall_time_s = time.perf_counter() - initialization_start

    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    history: list[dict[str, object]] = []
    training_start = time.perf_counter()
    for epoch in range(1, recipe.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for observations, targets in train_loader:
            observations = observations.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(observations)
            if predictions.shape != targets.shape or not bool(
                torch.isfinite(predictions).all()
            ):
                raise MTDLPyAdapterError(
                    "DinkNet50 produced an invalid training prediction"
                )
            loss = functional.mse_loss(predictions, targets, reduction="mean")
            if not bool(torch.isfinite(loss)):
                raise MTDLPyAdapterError("DinkNet50 training loss is non-finite")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=GRAD_CLIP_NORM
            )
            if not bool(torch.isfinite(grad_norm)):
                raise MTDLPyAdapterError("DinkNet50 gradient norm is non-finite")
            optimizer.step()
            batch_count = int(observations.shape[0])
            train_loss_sum += float(loss.detach().cpu()) * batch_count
            train_count += batch_count

        model.eval()
        validation_loss_sum = 0.0
        validation_count = 0
        with torch.no_grad():
            for observations, targets in validation_loader:
                observations = observations.to(device)
                targets = targets.to(device)
                predictions = model(observations)
                if predictions.shape != targets.shape or not bool(
                    torch.isfinite(predictions).all()
                ):
                    raise MTDLPyAdapterError(
                        "DinkNet50 produced an invalid validation prediction"
                    )
                loss = functional.mse_loss(predictions, targets, reduction="mean")
                batch_count = int(observations.shape[0])
                validation_loss_sum += float(loss.cpu()) * batch_count
                validation_count += batch_count
        train_loss = train_loss_sum / train_count
        validation_loss = validation_loss_sum / validation_count
        history.append(
            {
                "epoch": epoch,
                "train_mse": train_loss,
                "validation_mse": validation_loss,
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_wall_time_s = time.perf_counter() - training_start
    if best_state is None:
        raise MTDLPyAdapterError("training produced no validation-selected checkpoint")

    peak_cuda_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    cuda_device = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    runtime = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
        "cuda_device_name": cuda_device,
        "peak_cuda_memory_bytes": peak_cuda_memory,
        "preprocessing_wall_time_s": preprocessing_wall_time_s,
        "model_initialization_wall_time_s": model_initialization_wall_time_s,
        "training_wall_time_s": training_wall_time_s,
        "backend_wall_time_s": time.perf_counter() - backend_start,
    }
    return TrainingOutcome(
        state_dict=best_state,
        training_summary={
            "best_epoch": best_epoch,
            "best_validation_mse": best_loss,
            "history": history,
        },
        runtime=runtime,
    )


def _validate_model_state(model: Any, state: Mapping[str, Any], *, torch: Any) -> None:
    if type(state) is not dict:
        raise MTDLPyAdapterError("checkpoint model_state must be a plain dictionary")
    expected = model.state_dict()
    if set(state) != set(expected) or any(not isinstance(name, str) for name in state):
        raise MTDLPyAdapterError(
            "checkpoint model_state keys do not exactly match pinned DinkNet50"
        )
    for name, expected_tensor in expected.items():
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise MTDLPyAdapterError(f"checkpoint model_state[{name!r}] is not a tensor")
        if (
            tuple(value.shape) != tuple(expected_tensor.shape)
            or value.dtype != expected_tensor.dtype
            or value.layout != expected_tensor.layout
            or value.device.type != expected_tensor.device.type
        ):
            raise MTDLPyAdapterError(
                f"checkpoint model_state[{name!r}] shape, dtype, layout or device is wrong"
            )
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all()
        ):
            raise MTDLPyAdapterError(
                f"checkpoint model_state[{name!r}] contains non-finite values"
            )
    try:
        model.load_state_dict(dict(state), strict=True)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise MTDLPyAdapterError(
            "checkpoint model_state cannot be loaded into pinned DinkNet50"
        ) from exc


def _infer_model(
    test: HeldoutObservations,
    state: Mapping[str, Any],
    repository: Mapping[str, object],
    weights: Mapping[str, object],
    *,
    seed: int,
    device_name: str,
    recipe: TrainingRecipe,
) -> InferenceOutcome:
    backend_start = time.perf_counter()
    import torch

    _configure_determinism(torch, seed)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise MTDLPyAdapterError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    preprocessing_start = time.perf_counter()
    observations_test = _preprocess_observations(test.observations)
    preprocessing_wall_time_s = time.perf_counter() - preprocessing_start

    initialization_start = time.perf_counter()
    model, torchvision = _load_upstream_model(repository, weights, torch=torch)
    _validate_model_state(model, state, torch=torch)
    model.to(device)
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    model_initialization_wall_time_s = time.perf_counter() - initialization_start

    inference_start = time.perf_counter()
    prediction_batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, observations_test.shape[0], recipe.batch_size):
            batch = torch.from_numpy(
                observations_test[start : start + recipe.batch_size]
            ).to(device)
            prediction = model(batch)
            expected = (batch.shape[0], 1, *NETWORK_GRID_SHAPE)
            if tuple(prediction.shape) != expected or not bool(
                torch.isfinite(prediction).all()
            ):
                raise MTDLPyAdapterError("DinkNet50 produced an invalid test prediction")
            prediction_batches.append(prediction[:, 0].cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_wall_time_s = time.perf_counter() - inference_start
    predictions = _resize_log10_resistivity(
        np.concatenate(prediction_batches, axis=0), OUTPUT_GRID_SHAPE
    )
    peak_cuda_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    cuda_device = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    return InferenceOutcome(
        predicted_log10_resistivity=predictions,
        runtime={
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__),
            "torch_cuda_build": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "device": str(device),
            "cuda_device_name": cuda_device,
            "peak_cuda_memory_bytes": peak_cuda_memory,
            "preprocessing_wall_time_s": preprocessing_wall_time_s,
            "model_initialization_wall_time_s": model_initialization_wall_time_s,
            "inference_wall_time_s": inference_wall_time_s,
            "backend_wall_time_s": time.perf_counter() - backend_start,
        },
    )


def _canonical_npy_bytes(array: np.ndarray) -> bytes:
    payload = io.BytesIO()
    np.lib.format.write_array(payload, np.asarray(array), allow_pickle=False)
    return payload.getvalue()


def _prediction_npz_bytes(
    observations_sha256: str,
    sample_index: np.ndarray,
    x_cell_centers_m: np.ndarray,
    depth_cell_centers_m: np.ndarray,
    predictions: np.ndarray,
) -> bytes:
    arrays = (
        ("schema", np.asarray(PREDICTION_SCHEMA)),
        ("schema_version", np.asarray(PREDICTION_SCHEMA_VERSION, dtype="<i8")),
        (
            "observations_sha256",
            np.asarray(observations_sha256, dtype="<U64"),
        ),
        ("sample_index", np.asarray(sample_index, dtype="<i8")),
        (
            "x_cell_centers_m",
            np.asarray(x_cell_centers_m, dtype="<f8"),
        ),
        (
            "depth_cell_centers_m",
            np.asarray(depth_cell_centers_m, dtype="<f8"),
        ),
        (
            "predicted_log10_resistivity",
            np.asarray(predictions, dtype="<f4"),
        ),
    )
    payload = io.BytesIO()
    with zipfile.ZipFile(
        payload,
        mode="w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
        strict_timestamps=True,
    ) as archive:
        for name, array in arrays:
            member = zipfile.ZipInfo(f"{name}.npy", (1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_STORED
            member.create_system = 3
            member.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(member, _canonical_npy_bytes(array))
    return payload.getvalue()


def _remove_owned_part(path: Path, expected_identity: tuple[int, int]) -> None:
    """Remove only the exact staged inode created by this invocation."""
    if not os.path.lexists(path):
        return
    try:
        current = path.stat(follow_symlinks=False)
    except OSError:
        return
    if (
        _path_is_link(path)
        or not stat.S_ISREG(current.st_mode)
        or (int(current.st_dev), int(current.st_ino)) != expected_identity
    ):
        return
    path.unlink()


def _write_bytes_new(path: Path, payload: bytes) -> tuple[int, int]:
    identity: tuple[int, int] | None = None
    try:
        with path.open("xb") as stream:
            opened = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened.st_mode):
                raise MTDLPyAdapterError(f"new staged artifact is not regular: {path}")
            identity = (int(opened.st_dev), int(opened.st_ino))
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            after = os.fstat(stream.fileno())
            if (int(after.st_dev), int(after.st_ino)) != identity:
                raise MTDLPyAdapterError(
                    f"new staged artifact identity changed while writing: {path}"
                )
        return identity
    except BaseException:
        if identity is not None:
            _remove_owned_part(path, identity)
        raise


def _require_owned_snapshot(
    snapshot: _ArtifactSnapshot,
    expected_identity: tuple[int, int],
    *,
    role: str,
) -> None:
    if (snapshot.device, snapshot.inode) != expected_identity:
        raise MTDLPyAdapterError(f"{role} pathname was replaced after creation")


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MTDLPyAdapterError(f"runtime metadata is not strict JSON: {exc}") from exc


def _require_snapshot_matches(
    actual: _ArtifactSnapshot,
    expected: _ArtifactSnapshot,
    *,
    role: str,
) -> None:
    if (
        actual.sha256 != expected.sha256
        or actual.size_bytes != expected.size_bytes
        or actual.device != expected.device
        or actual.inode != expected.inode
    ):
        raise MTDLPyAdapterError(f"{role} changed after validation")


def _publish_parts(
    parts: Sequence[Path],
    destinations: Sequence[Path],
    *,
    expected_snapshots: Sequence[_ArtifactSnapshot] | None = None,
) -> None:
    """Publish with no overwrite and identity-aware rollback on BaseException."""
    if expected_snapshots is None:
        snapshots = tuple(
            _snapshot_regular_file(part, role=f"staged artifact {part}") for part in parts
        )
    else:
        snapshots = tuple(expected_snapshots)
    if len(parts) != len(destinations) or len(parts) != len(snapshots):
        raise ValueError("publication parts, destinations, and snapshots must align")
    published: list[tuple[Path, tuple[int, int]]] = []
    try:
        for part, destination, expected in zip(
            parts, destinations, snapshots, strict=True
        ):
            current_part = _snapshot_regular_file(part, role=f"staged artifact {part}")
            _require_snapshot_matches(
                current_part, expected, role=f"staged artifact {part}"
            )
            expected_identity = (expected.device, expected.inode)
            try:
                os.link(part, destination)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"publication race: refusing to overwrite {destination}"
                ) from exc
            except BaseException:
                if os.path.lexists(destination):
                    current = destination.stat(follow_symlinks=False)
                    if (
                        not _path_is_link(destination)
                        and stat.S_ISREG(current.st_mode)
                        and (int(current.st_dev), int(current.st_ino))
                        == expected_identity
                    ):
                        destination.unlink()
                raise
            published.append((destination, expected_identity))
            published_snapshot = _snapshot_regular_file(
                destination, role=f"published artifact {destination}"
            )
            _require_snapshot_matches(
                published_snapshot,
                expected,
                role=f"published artifact {destination}",
            )
        for destination, expected in zip(destinations, snapshots, strict=True):
            final_snapshot = _snapshot_regular_file(
                destination, role=f"published artifact {destination}"
            )
            _require_snapshot_matches(
                final_snapshot,
                expected,
                role=f"published artifact {destination}",
            )
    except BaseException as exc:
        unsafe: list[str] = []
        for destination, expected_identity in reversed(published):
            if not os.path.lexists(destination):
                continue
            current = destination.stat(follow_symlinks=False)
            if (
                _path_is_link(destination)
                or not stat.S_ISREG(current.st_mode)
                or (int(current.st_dev), int(current.st_ino)) != expected_identity
            ):
                unsafe.append(str(destination))
                continue
            destination.unlink()
        if unsafe:
            raise MTDLPyAdapterError(
                "refusing to delete outputs replaced during rollback: "
                + ", ".join(unsafe)
            ) from exc
        raise


def _output_paths(
    values: Sequence[str | Path],
    suffixes: Sequence[str],
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    paths = tuple(Path(os.path.abspath(os.fspath(path))) for path in values)
    if not paths or len(paths) != len(suffixes) or len(set(paths)) != len(paths):
        raise ValueError("MTDLPy output paths must be non-empty and distinct")
    for path, suffix in zip(paths, suffixes, strict=True):
        if os.path.lexists(path) and _path_is_link(path):
            raise MTDLPyAdapterError(
                f"MTDLPy output must not be a symbolic link or junction: {path}"
            )
        if path.suffix.lower() != suffix:
            raise ValueError(f"MTDLPy output {path} must use the {suffix} suffix")
        path.parent.mkdir(parents=True, exist_ok=True)
    existing = [path for path in paths if os.path.lexists(path)]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing MTDLPy output(s): "
            + ", ".join(str(path) for path in existing)
        )
    parts = tuple(path.with_name(path.name + ".part") for path in paths)
    stale = [path for path in parts if os.path.lexists(path)]
    if stale:
        raise FileExistsError(
            "stale MTDLPy partial output(s) require inspection: "
            + ", ".join(str(path) for path in stale)
        )
    return paths, parts


def _regular_input_artifact(path: str | Path, *, role: str) -> dict[str, object]:
    return _snapshot_core(_snapshot_regular_file(path, role=role))


def _checkpoint_bytes(value: Mapping[str, object]) -> bytes:
    import torch

    payload = io.BytesIO()
    torch.save(dict(value), payload)
    return payload.getvalue()


def _geometry_contract(value: TrainingSplit | HeldoutObservations) -> dict[str, object]:
    return {
        "frequency_hz": np.asarray(value.frequencies, dtype="<f8").tolist(),
        "station_x_m": np.asarray(value.station_x, dtype="<f8").tolist(),
        "x_cell_centers_m": np.asarray(value.x_grid, dtype="<f8").tolist(),
        "depth_cell_centers_m": np.asarray(value.depth_grid, dtype="<f8").tolist(),
        "input_grid_shape": list(INPUT_GRID_SHAPE),
        "output_grid_shape": list(OUTPUT_GRID_SHAPE),
    }


def _require_geometry_contract(
    expected: Mapping[str, object], value: HeldoutObservations
) -> None:
    if not _strict_metadata_equal(expected, _geometry_contract(value)):
        raise MTDLPyAdapterError(
            "held-out observation geometry differs from checkpoint training geometry"
        )


def _validate_training_geometry_metadata(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "frequency_hz",
        "station_x_m",
        "x_cell_centers_m",
        "depth_cell_centers_m",
        "input_grid_shape",
        "output_grid_shape",
    }:
        raise MTDLPyAdapterError("MTDLPy checkpoint training geometry is not exact")
    if value["input_grid_shape"] != list(INPUT_GRID_SHAPE) or value[
        "output_grid_shape"
    ] != list(OUTPUT_GRID_SHAPE):
        raise MTDLPyAdapterError("MTDLPy checkpoint training geometry shape is wrong")
    expected_sizes = {
        "frequency_hz": INPUT_GRID_SHAPE[0],
        "station_x_m": INPUT_GRID_SHAPE[1],
        "x_cell_centers_m": OUTPUT_GRID_SHAPE[1],
        "depth_cell_centers_m": OUTPUT_GRID_SHAPE[0],
    }
    for name, size in expected_sizes.items():
        if not isinstance(value[name], list) or any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            for item in value[name]
        ):
            raise MTDLPyAdapterError(
                f"MTDLPy checkpoint training geometry {name} must be a numeric list"
            )
        array = np.asarray(value[name], dtype=np.float64)
        if (
            array.shape != (size,)
            or not np.isfinite(array).all()
            or np.any(np.diff(array) <= 0)
        ):
            raise MTDLPyAdapterError(
                f"MTDLPy checkpoint training geometry {name} is invalid"
            )
        if name in {"frequency_hz", "depth_cell_centers_m"} and np.any(array <= 0):
            raise MTDLPyAdapterError(
                f"MTDLPy checkpoint training geometry {name} must be positive"
            )


_SAFE_FALSE_PRESCORE_DECLARATIONS = frozenset(
    {
        "contains_truth",
        "heldout_truth_available_to_adapter",
        "truth_keys_accepted",
    }
)


def _metadata_key_is_forbidden(key: str, value: object) -> bool:
    camel_split = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key)
    normalized = re.sub(r"[^a-z0-9]+", "_", camel_split.lower()).strip("_")
    if normalized in _SAFE_FALSE_PRESCORE_DECLARATIONS and value is False:
        return False
    tokens = normalized.split("_")
    if any(
        left == "generator" and right in {"seed", "seeds"}
        for left, right in pairwise(tokens)
    ):
        return True
    if any(
        token in {"hidden", "operator", "secret", "truth", "withheld"} for token in tokens
    ):
        return True
    return any(
        left == "sample" and right in {"id", "ids"} for left, right in pairwise(tokens)
    )


def _require_no_prescore_metadata(value: object, *, where: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise MTDLPyAdapterError(f"{where} contains a non-string metadata key")
            if _metadata_key_is_forbidden(key, child):
                raise MTDLPyAdapterError(
                    f"{where} exposes forbidden pre-score metadata key {key!r}"
                )
            _require_no_prescore_metadata(child, where=where)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            _require_no_prescore_metadata(child, where=where)


def _validate_backend_runtime(value: object, *, phase: str) -> None:
    if not isinstance(value, Mapping):
        raise MTDLPyAdapterError(f"MTDLPy {phase} runtime must be a mapping")
    expected = {
        "training": _TRAINING_BACKEND_RUNTIME_KEYS,
        "inference": _INFERENCE_BACKEND_RUNTIME_KEYS,
    }.get(phase)
    if expected is None:  # pragma: no cover - internal closed call sites
        raise AssertionError(f"unsupported MTDLPy runtime phase: {phase}")
    if set(value) != set(expected):
        raise MTDLPyAdapterError(f"MTDLPy {phase} runtime schema is not exact")
    for name in ("python", "platform", "torch", "torchvision"):
        item = value[name]
        if not isinstance(item, str) or not item:
            raise MTDLPyAdapterError(
                f"MTDLPy {phase} runtime {name} must be a non-empty string"
            )
    torch_cuda_build = value["torch_cuda_build"]
    if torch_cuda_build is not None and (
        not isinstance(torch_cuda_build, str) or not torch_cuda_build
    ):
        raise MTDLPyAdapterError(
            f"MTDLPy {phase} runtime torch_cuda_build must be null or a non-empty string"
        )
    if not isinstance(value["cuda_available"], bool):
        raise MTDLPyAdapterError(f"MTDLPy {phase} runtime cuda_available must be boolean")
    device = value["device"]
    if device not in {"cpu", "cuda"}:
        raise MTDLPyAdapterError(f"MTDLPy {phase} runtime device must be cpu or cuda")
    cuda_device_name = value["cuda_device_name"]
    if device == "cuda":
        if (
            value["cuda_available"] is not True
            or not isinstance(cuda_device_name, str)
            or not cuda_device_name
        ):
            raise MTDLPyAdapterError(
                f"MTDLPy {phase} CUDA runtime must identify an available CUDA device"
            )
    elif cuda_device_name is not None:
        raise MTDLPyAdapterError(
            f"MTDLPy {phase} CPU runtime must not identify a CUDA execution device"
        )
    peak = value["peak_cuda_memory_bytes"]
    if isinstance(peak, bool) or not isinstance(peak, int) or peak < 0:
        raise MTDLPyAdapterError(
            f"MTDLPy {phase} runtime peak_cuda_memory_bytes must be a non-negative integer"
        )
    if device == "cpu" and peak != 0:
        raise MTDLPyAdapterError(
            f"MTDLPy {phase} CPU runtime peak_cuda_memory_bytes must be zero"
        )
    for name in {
        "preprocessing_wall_time_s",
        "model_initialization_wall_time_s",
        "backend_wall_time_s",
        f"{phase}_wall_time_s",
    }:
        item = value[name]
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not np.isfinite(float(item))
            or float(item) < 0.0
        ):
            raise MTDLPyAdapterError(
                f"MTDLPy {phase} runtime {name} must be finite and non-negative"
            )
    component_sum = sum(
        float(value[name])
        for name in (
            "preprocessing_wall_time_s",
            "model_initialization_wall_time_s",
            f"{phase}_wall_time_s",
        )
    )
    if float(value["backend_wall_time_s"]) + 1e-12 < component_sum:
        raise MTDLPyAdapterError(
            f"MTDLPy {phase} backend_wall_time_s is shorter than its components"
        )
    _canonical_json_bytes(value)
    _require_no_prescore_metadata(value, where=f"MTDLPy {phase} runtime")


def _decode_checkpoint_snapshot(snapshot: _ArtifactSnapshot) -> dict[str, object]:
    try:
        import torch

        loaded = torch.load(
            io.BytesIO(snapshot.payload),
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise MTDLPyAdapterError(
            "MTDLPy checkpoint is not loadable by the restricted weights-only loader"
        ) from exc
    if type(loaded) is not dict or set(loaded) != CHECKPOINT_KEYS:
        raise MTDLPyAdapterError("MTDLPy checkpoint root schema is not exact")
    if type(loaded["model_state"]) is not dict:
        raise MTDLPyAdapterError(
            "MTDLPy checkpoint model_state must be a plain dictionary"
        )
    metadata = {key: value for key, value in loaded.items() if key != "model_state"}
    _require_no_prescore_metadata(metadata, where="MTDLPy checkpoint")
    return loaded


def _load_checkpoint_safely(
    path: str | Path,
) -> tuple[dict[str, object], dict[str, object]]:
    snapshot = _snapshot_regular_file(
        path,
        role="MTDLPy checkpoint",
        max_size_bytes=MAX_CHECKPOINT_SIZE_BYTES,
    )
    return _decode_checkpoint_snapshot(snapshot), _snapshot_core(snapshot)


def _validate_checkpoint_metadata(
    checkpoint: Mapping[str, object],
    *,
    repository: Mapping[str, object],
    weights: Mapping[str, object],
    train_artifact: Mapping[str, object],
    validation_artifact: Mapping[str, object],
    dependency_closure: Mapping[str, object],
    seed: int,
    recipe: TrainingRecipe,
) -> None:
    if type(checkpoint) is not dict or set(checkpoint) != CHECKPOINT_KEYS:
        raise MTDLPyAdapterError("MTDLPy checkpoint root schema is not exact")
    _require_no_prescore_metadata(
        {key: value for key, value in checkpoint.items() if key != "model_state"},
        where="MTDLPy checkpoint",
    )
    if (
        checkpoint["checkpoint_schema"] != CHECKPOINT_SCHEMA
        or not isinstance(checkpoint["checkpoint_schema"], str)
        or checkpoint["checkpoint_schema_version"] != CHECKPOINT_SCHEMA_VERSION
        or isinstance(checkpoint["checkpoint_schema_version"], bool)
        or not isinstance(checkpoint["checkpoint_schema_version"], int)
        or checkpoint["method"] != "MTDLPy/DinkNet50"
        or checkpoint["track"] != "common-retrain"
        or checkpoint["seed"] != seed
        or isinstance(checkpoint["seed"], bool)
        or not isinstance(checkpoint["seed"], int)
        or checkpoint["recipe_id"] != recipe.recipe_id
    ):
        raise MTDLPyAdapterError("MTDLPy checkpoint identity, seed or recipe is wrong")
    if (
        checkpoint["truth_keys_accepted"] is not False
        or checkpoint["contains_truth"] is not False
        or checkpoint["contains_observation_campaign"] is not False
    ):
        raise MTDLPyAdapterError(
            "MTDLPy checkpoint does not prove campaign-independent truth-free state"
        )
    expected_datasets = {
        "train": dict(train_artifact),
        "validation": dict(validation_artifact),
    }
    if not _strict_metadata_equal(checkpoint["dataset_identities"], expected_datasets):
        raise MTDLPyAdapterError(
            "MTDLPy checkpoint train/validation identities do not match exact inputs"
        )
    if not _strict_metadata_equal(checkpoint["source"], repository):
        raise MTDLPyAdapterError("MTDLPy checkpoint upstream source identity is wrong")
    if not _strict_metadata_equal(checkpoint["imagenet_weights"], weights):
        raise MTDLPyAdapterError("MTDLPy checkpoint ImageNet identity is wrong")
    if not _strict_metadata_equal(
        checkpoint["training_config"], _training_config(seed, recipe)
    ):
        raise MTDLPyAdapterError("MTDLPy checkpoint training recipe is not exact")
    if not _strict_metadata_equal(checkpoint["preprocessing"], _preprocessing_contract()):
        raise MTDLPyAdapterError("MTDLPy checkpoint preprocessing contract is not exact")
    if not _strict_metadata_equal(checkpoint["dependency_closure"], dependency_closure):
        raise MTDLPyAdapterError("MTDLPy checkpoint dependency closure is not exact")
    geometry = checkpoint["training_geometry"]
    _validate_training_geometry_metadata(geometry)
    summary = checkpoint["training_summary"]
    if type(summary) is not dict:
        raise MTDLPyAdapterError("MTDLPy checkpoint training summary is invalid")
    best_epoch = summary.get("best_epoch")
    best_loss = summary.get("best_validation_mse")
    history = summary.get("history")
    if (
        set(summary) != {"best_epoch", "best_validation_mse", "history"}
        or isinstance(best_epoch, bool)
        or not isinstance(best_epoch, int)
        or not 1 <= best_epoch <= recipe.epochs
        or isinstance(best_loss, bool)
        or not isinstance(best_loss, (int, float))
        or not np.isfinite(float(best_loss))
        or not isinstance(history, list)
        or len(history) != recipe.epochs
    ):
        raise MTDLPyAdapterError("MTDLPy checkpoint training summary is not exact")
    validation_losses: list[float] = []
    for expected_epoch, entry in enumerate(history, start=1):
        if type(entry) is not dict or set(entry) != {
            "epoch",
            "train_mse",
            "validation_mse",
        }:
            raise MTDLPyAdapterError("MTDLPy checkpoint training history is not exact")
        epoch = entry["epoch"]
        train_mse = entry["train_mse"]
        validation_mse = entry["validation_mse"]
        if (
            isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or epoch != expected_epoch
            or isinstance(train_mse, bool)
            or not isinstance(train_mse, (int, float))
            or not np.isfinite(float(train_mse))
            or float(train_mse) < 0
            or isinstance(validation_mse, bool)
            or not isinstance(validation_mse, (int, float))
            or not np.isfinite(float(validation_mse))
            or float(validation_mse) < 0
        ):
            raise MTDLPyAdapterError(
                "MTDLPy checkpoint training history contains invalid values"
            )
        validation_losses.append(float(validation_mse))
    selected_loss = min(validation_losses)
    selected_epoch = validation_losses.index(selected_loss) + 1
    if best_epoch != selected_epoch or float(best_loss) != selected_loss:
        raise MTDLPyAdapterError(
            "MTDLPy checkpoint best validation selection does not match history"
        )
    _validate_backend_runtime(checkpoint["training_runtime"], phase="training")


def train_common_retrain(
    *,
    repository_path: str | Path,
    imagenet_weights_path: str | Path,
    imagenet_weights_sha256: str,
    train_h5: str | Path,
    validation_h5: str | Path,
    seed: int,
    device: str,
    checkpoint_out: str | Path,
    runtime_out: str | Path,
    recipe_id: str = DEFAULT_RECIPE_ID,
    command: Sequence[str] | None = None,
    runner_source: str | Path | None = None,
) -> dict[str, object]:
    """Train exactly one seed checkpoint without opening any test campaign."""
    if isinstance(seed, bool) or seed not in COMMON_RETRAIN_SEEDS:
        raise ValueError(f"seed must be one of {COMMON_RETRAIN_SEEDS}")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'")
    recipe = training_recipe(recipe_id)
    destinations, parts = _output_paths((checkpoint_out, runtime_out), (".pt", ".json"))
    checkpoint_path, _runtime_path = destinations
    checkpoint_part, runtime_part = parts

    started_at = datetime.now(UTC)
    wall_start = time.perf_counter()
    repository = verify_pinned_repository(repository_path)
    weights = validate_local_imagenet_weights(
        imagenet_weights_path, imagenet_weights_sha256
    )
    train = load_training_split(train_h5, role="MTDLPy training dataset")
    validation = load_training_split(validation_h5, role="MTDLPy validation dataset")
    _require_same_geometry(train, validation, where="train and validation datasets")
    _require_training_disjoint(train, validation)
    closure = _dependency_closure(
        repository=repository,
        weights=weights,
        runner_source=runner_source,
    )
    closure_sources = closure["local_source_artifacts"]
    assert isinstance(closure_sources, Mapping)
    source_artifacts: dict[str, Mapping[str, object]] = {
        "train_dataset": train.provenance,
        "validation_dataset": validation.provenance,
        "imagenet_weights": weights,
        "dinknet_source": repository["dinknet_source"],
        "adapter_source": closure_sources["adapter"],
        "dataset_contract_loader_source": closure_sources["pimsr_inversion_contracts2d"],
        "materializer_contract_source": closure_sources["dataset2d_materialization"],
        "artifact_guard_source": closure_sources["runner2d"],
    }
    if "cli_runner" in closure_sources:
        source_artifacts["runner_source"] = closure_sources["cli_runner"]

    outcome = _train_model(
        train,
        validation,
        repository,
        weights,
        seed=seed,
        device_name=device,
        recipe=recipe,
    )
    if (
        type(outcome) is not TrainingOutcome
        or type(outcome.state_dict) is not dict
        or type(outcome.training_summary) is not dict
        or type(outcome.runtime) is not dict
    ):
        raise MTDLPyAdapterError(
            "MTDLPy training backend returned a malformed TrainingOutcome"
        )
    _validate_backend_runtime(outcome.runtime, phase="training")
    training_config = _training_config(seed, recipe)
    preprocessing = _preprocessing_contract()
    checkpoint: dict[str, object] = {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "method": "MTDLPy/DinkNet50",
        "track": "common-retrain",
        "seed": seed,
        "recipe_id": recipe.recipe_id,
        "model_state": dict(outcome.state_dict),
        "training_config": training_config,
        "preprocessing": preprocessing,
        "training_summary": dict(outcome.training_summary),
        "training_runtime": dict(outcome.runtime),
        "source": repository,
        "imagenet_weights": weights,
        "dataset_identities": {
            "train": dict(train.provenance),
            "validation": dict(validation.provenance),
        },
        "training_geometry": _geometry_contract(train),
        "dependency_closure": closure,
        "truth_keys_accepted": False,
        "contains_truth": False,
        "contains_observation_campaign": False,
    }
    _validate_checkpoint_metadata(
        checkpoint,
        repository=repository,
        weights=weights,
        train_artifact=train.provenance,
        validation_artifact=validation.provenance,
        dependency_closure=closure,
        seed=seed,
        recipe=recipe,
    )
    import torch

    validation_model, _torchvision = _load_upstream_model(
        repository, weights, torch=torch
    )
    _validate_model_state(validation_model, checkpoint["model_state"], torch=torch)
    owned_parts: dict[Path, tuple[int, int]] = {}
    try:
        for role, artifact in source_artifacts.items():
            require_file_artifact_unchanged(_artifact_core(artifact), role=role)
        if verify_pinned_repository(repository_path) != repository:
            raise MTDLPyAdapterError("pinned MTDLPy repository changed during training")
        owned_parts[checkpoint_part] = _write_bytes_new(
            checkpoint_part, _checkpoint_bytes(checkpoint)
        )
        checkpoint_snapshot = _snapshot_regular_file(
            checkpoint_part,
            role="staged MTDLPy checkpoint",
            max_size_bytes=MAX_CHECKPOINT_SIZE_BYTES,
        )
        _require_owned_snapshot(
            checkpoint_snapshot,
            owned_parts[checkpoint_part],
            role="staged MTDLPy checkpoint",
        )
        roundtrip_checkpoint = _decode_checkpoint_snapshot(checkpoint_snapshot)
        checkpoint_staged = _snapshot_core(checkpoint_snapshot)
        _validate_checkpoint_metadata(
            roundtrip_checkpoint,
            repository=repository,
            weights=weights,
            train_artifact=train.provenance,
            validation_artifact=validation.provenance,
            dependency_closure=closure,
            seed=seed,
            recipe=recipe,
        )
        _validate_model_state(
            validation_model,
            roundtrip_checkpoint["model_state"],
            torch=torch,
        )
        checkpoint_identity = {**checkpoint_staged, "path": str(checkpoint_path)}
        finished_at = datetime.now(UTC)
        bindings: dict[str, object] = {
            "training_seed": seed,
            "source_commit": repository["commit"],
            "source_clean_worktree": True,
            "upstream_source_sha256": repository["dinknet_source"]["sha256"],
            "adapter_source_sha256": closure_sources["adapter"]["sha256"],
            "train_sha256": train.provenance["sha256"],
            "validation_sha256": validation.provenance["sha256"],
            "imagenet_weights_sha256": weights["sha256"],
            "dependency_closure_sha256": _canonical_object_sha256(closure),
            "checkpoint_sha256": checkpoint_staged["sha256"],
        }
        if "cli_runner" in closure_sources:
            bindings["runner_source_sha256"] = closure_sources["cli_runner"]["sha256"]
        runtime = {
            "schema": TRAINING_RUNTIME_SCHEMA,
            "schema_version": TRAINING_RUNTIME_SCHEMA_VERSION,
            "method": "MTDLPy/DinkNet50",
            "track": "common-retrain",
            "operation": "train_checkpoint_once",
            "seed": seed,
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "adapter_wall_time_s": time.perf_counter() - wall_start,
            "command": list(command) if command is not None else None,
            "working_directory": str(Path.cwd().resolve()),
            "repository": repository,
            "source_artifacts": {
                key: dict(value) for key, value in source_artifacts.items()
            },
            "training_config": training_config,
            "preprocessing": preprocessing,
            "training_summary": dict(outcome.training_summary),
            "runtime": dict(outcome.runtime),
            "dependency_closure": closure,
            "truth_keys_accepted": False,
            "contains_truth": False,
            "observation_campaigns_accessed": False,
            "bindings": bindings,
            "outputs": {"checkpoint": checkpoint_identity},
        }
        _require_no_prescore_metadata(runtime, where="MTDLPy training runtime")
        for role, artifact in source_artifacts.items():
            require_file_artifact_unchanged(_artifact_core(artifact), role=role)
        require_file_artifact_unchanged(
            checkpoint_staged, role="staged MTDLPy checkpoint"
        )
        if verify_pinned_repository(repository_path) != repository:
            raise MTDLPyAdapterError(
                "pinned MTDLPy repository changed before checkpoint publication"
            )
        runtime_payload = _canonical_json_bytes(runtime)
        owned_parts[runtime_part] = _write_bytes_new(runtime_part, runtime_payload)
        runtime_snapshot = _snapshot_regular_file(
            runtime_part, role="staged MTDLPy training runtime"
        )
        _require_owned_snapshot(
            runtime_snapshot,
            owned_parts[runtime_part],
            role="staged MTDLPy training runtime",
        )
        if runtime_snapshot.payload != runtime_payload:
            raise MTDLPyAdapterError("staged training runtime changed after writing")
        for role, artifact in source_artifacts.items():
            require_file_artifact_unchanged(_artifact_core(artifact), role=role)
        require_file_artifact_unchanged(
            checkpoint_staged, role="staged MTDLPy checkpoint"
        )
        if verify_pinned_repository(repository_path) != repository:
            raise MTDLPyAdapterError(
                "pinned MTDLPy repository changed during runtime staging"
            )
        _publish_parts(
            parts,
            destinations,
            expected_snapshots=(checkpoint_snapshot, runtime_snapshot),
        )
        return runtime
    finally:
        for part, identity in owned_parts.items():
            _remove_owned_part(part, identity)


def infer_common_retrain(
    *,
    repository_path: str | Path,
    imagenet_weights_path: str | Path,
    imagenet_weights_sha256: str,
    train_h5: str | Path,
    validation_h5: str | Path,
    checkpoint: str | Path,
    observations_npz: str | Path,
    seed: int,
    device: str,
    predictions_out: str | Path,
    runtime_out: str | Path,
    recipe_id: str = DEFAULT_RECIPE_ID,
    command: Sequence[str] | None = None,
    runner_source: str | Path | None = None,
) -> dict[str, object]:
    """Infer one truth-free campaign from an immutable seed checkpoint."""
    if isinstance(seed, bool) or seed not in COMMON_RETRAIN_SEEDS:
        raise ValueError(f"seed must be one of {COMMON_RETRAIN_SEEDS}")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'")
    recipe = training_recipe(recipe_id)
    destinations, parts = _output_paths((predictions_out, runtime_out), (".npz", ".json"))
    prediction_path, _runtime_path = destinations
    prediction_part, runtime_part = parts

    started_at = datetime.now(UTC)
    wall_start = time.perf_counter()
    repository = verify_pinned_repository(repository_path)
    weights = validate_local_imagenet_weights(
        imagenet_weights_path, imagenet_weights_sha256
    )
    train_artifact = _regular_input_artifact(train_h5, role="MTDLPy training dataset")
    validation_artifact = _regular_input_artifact(
        validation_h5, role="MTDLPy validation dataset"
    )
    if train_artifact["sha256"] == validation_artifact["sha256"]:
        raise MTDLPyAdapterError("train and validation artifacts must differ")
    closure = _dependency_closure(
        repository=repository,
        weights=weights,
        runner_source=runner_source,
    )
    checkpoint_snapshot = _snapshot_regular_file(
        checkpoint,
        role="MTDLPy checkpoint",
        max_size_bytes=MAX_CHECKPOINT_SIZE_BYTES,
    )
    checkpoint_state = _decode_checkpoint_snapshot(checkpoint_snapshot)
    checkpoint_artifact = _snapshot_core(checkpoint_snapshot)
    _validate_checkpoint_metadata(
        checkpoint_state,
        repository=repository,
        weights=weights,
        train_artifact=train_artifact,
        validation_artifact=validation_artifact,
        dependency_closure=closure,
        seed=seed,
        recipe=recipe,
    )
    test = load_heldout_observations(observations_npz)
    geometry = checkpoint_state["training_geometry"]
    assert isinstance(geometry, Mapping)
    _require_geometry_contract(geometry, test)
    if test.provenance["sha256"] in {
        train_artifact["sha256"],
        validation_artifact["sha256"],
        checkpoint_artifact["sha256"],
    }:
        raise MTDLPyAdapterError(
            "observation campaign must differ from training, validation and checkpoint"
        )

    outcome = _infer_model(
        test,
        checkpoint_state["model_state"],
        repository,
        weights,
        seed=seed,
        device_name=device,
        recipe=recipe,
    )
    if type(outcome) is not InferenceOutcome or type(outcome.runtime) is not dict:
        raise MTDLPyAdapterError(
            "MTDLPy inference backend returned a malformed InferenceOutcome"
        )
    _validate_backend_runtime(outcome.runtime, phase="inference")
    predictions = np.asarray(outcome.predicted_log10_resistivity, dtype="<f4")
    expected_prediction_shape = (test.sample_index.size, *OUTPUT_GRID_SHAPE)
    if (
        predictions.shape != expected_prediction_shape
        or not np.isfinite(predictions).all()
    ):
        raise MTDLPyAdapterError(
            f"backend predictions must be finite with shape {expected_prediction_shape}"
        )

    closure_sources = closure["local_source_artifacts"]
    assert isinstance(closure_sources, Mapping)
    source_artifacts: dict[str, Mapping[str, object]] = {
        "train_dataset": train_artifact,
        "validation_dataset": validation_artifact,
        "heldout_observations": test.provenance,
        "imagenet_weights": weights,
        "dinknet_source": repository["dinknet_source"],
        "adapter_source": closure_sources["adapter"],
        "dataset_contract_loader_source": closure_sources["pimsr_inversion_contracts2d"],
        "materializer_contract_source": closure_sources["dataset2d_materialization"],
        "artifact_guard_source": closure_sources["runner2d"],
    }
    if "cli_runner" in closure_sources:
        source_artifacts["runner_source"] = closure_sources["cli_runner"]
    for role, artifact in source_artifacts.items():
        require_file_artifact_unchanged(_artifact_core(artifact), role=role)
    require_file_artifact_unchanged(
        checkpoint_artifact, role="MTDLPy reusable checkpoint"
    )
    if verify_pinned_repository(repository_path) != repository:
        raise MTDLPyAdapterError("pinned MTDLPy repository changed during inference")

    owned_parts: dict[Path, tuple[int, int]] = {}
    try:
        owned_parts[prediction_part] = _write_bytes_new(
            prediction_part,
            _prediction_npz_bytes(
                str(test.provenance["sha256"]),
                test.sample_index,
                test.x_grid,
                test.depth_grid,
                predictions,
            ),
        )
        prediction_snapshot = _snapshot_regular_file(
            prediction_part, role="staged MTDLPy predictions"
        )
        _require_owned_snapshot(
            prediction_snapshot,
            owned_parts[prediction_part],
            role="staged MTDLPy predictions",
        )
        prediction_staged = _snapshot_core(prediction_snapshot)
        prediction_identity = {**prediction_staged, "path": str(prediction_path)}
        training_config = _training_config(seed, recipe)
        preprocessing = _preprocessing_contract()
        bindings: dict[str, object] = {
            "training_seed": seed,
            "source_commit": repository["commit"],
            "source_clean_worktree": True,
            "upstream_source_sha256": repository["dinknet_source"]["sha256"],
            "adapter_source_sha256": closure_sources["adapter"]["sha256"],
            "train_sha256": train_artifact["sha256"],
            "validation_sha256": validation_artifact["sha256"],
            "imagenet_weights_sha256": weights["sha256"],
            "dependency_closure_sha256": _canonical_object_sha256(closure),
            "checkpoint_sha256": checkpoint_artifact["sha256"],
            "observations_sha256": test.provenance["sha256"],
            "prediction_sha256": prediction_staged["sha256"],
        }
        if "cli_runner" in closure_sources:
            bindings["runner_source_sha256"] = closure_sources["cli_runner"]["sha256"]
        finished_at = datetime.now(UTC)
        runtime = {
            "schema": RUNTIME_SCHEMA,
            "schema_version": RUNTIME_SCHEMA_VERSION,
            "method": "MTDLPy/DinkNet50",
            "track": "common-retrain",
            "operation": "inference_from_reusable_checkpoint",
            "comparison_status": "unscored_prediction_artifact",
            "ranking_allowed": False,
            "seed": seed,
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "adapter_wall_time_s": time.perf_counter() - wall_start,
            "command": list(command) if command is not None else None,
            "working_directory": str(Path.cwd().resolve()),
            "repository": repository,
            "source_artifacts": {
                key: dict(value) for key, value in source_artifacts.items()
            },
            "training_config": training_config,
            "preprocessing": preprocessing,
            "determinism": {
                "cublas_workspace_config": ":4096:8",
                "python_random_seed": seed,
                "numpy_legacy_global_seed": seed,
                "torch_deterministic_algorithms": True,
                "cudnn_deterministic": True,
                "cudnn_benchmark": False,
                "tf32": False,
                "data_loader_workers": 0,
            },
            "training_summary": dict(checkpoint_state["training_summary"]),
            "runtime": dict(outcome.runtime),
            "dependency_closure": closure,
            "bindings": bindings,
            "truth_keys_accepted": False,
            "contains_truth": False,
            "observation_contract": {
                "schema": OBSERVATION_SCHEMA,
                "schema_version": PAYLOAD_SCHEMA_VERSION,
                "observations_sha256": test.provenance["sha256"],
                "truth_keys_accepted": False,
                "contains_truth": False,
                "sample_count": int(test.sample_index.size),
                "sample_index_sha256": hashlib.sha256(
                    np.asarray(test.sample_index, dtype="<i8").tobytes()
                ).hexdigest(),
                "evaluation_floor_role": "scorer_only_not_model_input",
            },
            "checkpoint_contract": {
                "schema": CHECKPOINT_SCHEMA,
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "safe_load": "torch.load(weights_only=True)",
                "seed": seed,
                "recipe_id": recipe.recipe_id,
                "contains_truth": False,
                "contains_observation_campaign": False,
                "dataset_identities": checkpoint_state["dataset_identities"],
            },
            "prediction_contract": {
                "schema": PREDICTION_SCHEMA,
                "schema_version": PREDICTION_SCHEMA_VERSION,
                "keys_exact": list(PREDICTION_KEYS),
                "observations_sha256": str(test.provenance["sha256"]),
                "observations_sha256_dtype": "<U64",
                "sample_index_dtype": "<i8",
                "x_cell_centers_dtype": "<f8",
                "depth_cell_centers_dtype": "<f8",
                "prediction_dtype": "<f4",
                "prediction_shape": list(predictions.shape),
                "prediction_axis_order": ["sample", "depth", "x"],
                "prediction_unit": "log10_ohm_m",
                "truth_keys_accepted": False,
                "contains_truth": False,
            },
            "outputs": {
                "checkpoint": dict(checkpoint_artifact),
                "predictions": prediction_identity,
            },
        }
        _require_no_prescore_metadata(runtime, where="MTDLPy inference runtime")
        for role, artifact in source_artifacts.items():
            require_file_artifact_unchanged(_artifact_core(artifact), role=role)
        require_file_artifact_unchanged(
            checkpoint_artifact, role="MTDLPy reusable checkpoint"
        )
        require_file_artifact_unchanged(
            prediction_staged, role="staged MTDLPy predictions"
        )
        if verify_pinned_repository(repository_path) != repository:
            raise MTDLPyAdapterError(
                "pinned MTDLPy repository changed before inference publication"
            )
        runtime_payload = _canonical_json_bytes(runtime)
        owned_parts[runtime_part] = _write_bytes_new(runtime_part, runtime_payload)
        runtime_snapshot = _snapshot_regular_file(
            runtime_part, role="staged MTDLPy inference runtime"
        )
        _require_owned_snapshot(
            runtime_snapshot,
            owned_parts[runtime_part],
            role="staged MTDLPy inference runtime",
        )
        if runtime_snapshot.payload != runtime_payload:
            raise MTDLPyAdapterError("staged inference runtime changed after writing")
        for role, artifact in source_artifacts.items():
            require_file_artifact_unchanged(_artifact_core(artifact), role=role)
        require_file_artifact_unchanged(
            checkpoint_artifact, role="MTDLPy reusable checkpoint"
        )
        require_file_artifact_unchanged(
            prediction_staged, role="staged MTDLPy predictions"
        )
        if verify_pinned_repository(repository_path) != repository:
            raise MTDLPyAdapterError(
                "pinned MTDLPy repository changed during runtime staging"
            )
        _publish_parts(
            parts,
            destinations,
            expected_snapshots=(prediction_snapshot, runtime_snapshot),
        )
        return runtime
    finally:
        for part, identity in owned_parts.items():
            _remove_owned_part(part, identity)


__all__ = [
    "COMMON_RETRAIN_SEEDS",
    "DEFAULT_RECIPE_ID",
    "REVIEWED_RECIPE",
    "TRAINING_RECIPES",
    "UPSTREAM_CONFIG_RECIPE",
    "HeldoutObservations",
    "InferenceOutcome",
    "MTDLPyAdapterError",
    "TrainingOutcome",
    "TrainingRecipe",
    "TrainingSplit",
    "infer_common_retrain",
    "load_heldout_observations",
    "load_training_split",
    "resize_bilinear_half_pixel",
    "train_common_retrain",
    "training_recipe",
    "validate_local_imagenet_weights",
    "verify_pinned_repository",
]
