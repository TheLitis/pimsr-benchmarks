"""Fail-closed common-retraining adapter for the pinned MTDLPy DinkNet50.

The adapter deliberately accepts held-out observations only through the
observation-only payload emitted by :mod:`dataset2d_materialization`.  Test
truth therefore cannot enter the process that constructs or trains the model.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import platform
import random
import ssl
import stat
import subprocess
import sys
import time
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from pimsr_benchmarks.dataset2d_materialization import (
    OBSERVATION_CHANNEL_ORDER,
    OBSERVATION_SCHEMA,
    PAYLOAD_SCHEMA_VERSION,
)
from pimsr_benchmarks.runner2d import (
    file_artifact_provenance,
    require_file_artifact_unchanged,
)

MTDLPY_REPOSITORY_URL = "https://github.com/Yuan-Chongxin/MTDLPy.git"
MTDLPY_COMMIT = "b01f72a53078a9dc8d452fa53ea5009639d00b04"
MTDLPY_DINKNET_PATH = "func/dinknet.py"
MTDLPY_DINKNET_GIT_BLOB = "5551d6b598f9934db4d4beb17f475c6da36b4a53"
MTDLPY_DINKNET_SHA256 = (
    "838d1271c6987fdac05daf53f2408827d86d493f1fa2ec73a6ecd4753d42ebae"
)

IMAGENET_RESNET50_V1_URL = (
    "https://download.pytorch.org/models/resnet50-0676ba61.pth"
)
IMAGENET_RESNET50_V1_SHA256 = (
    "0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a"
)

COMMON_RETRAIN_SEEDS = (101, 102, 103, 104, 105)
INPUT_GRID_SHAPE = (8, 12)
NETWORK_GRID_SHAPE = (32, 32)
OUTPUT_GRID_SHAPE = (64, 48)

EPOCHS = 10
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
WEIGHT_DECAY = 0.0
GRAD_CLIP_NORM = 0.1

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
RUNTIME_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA = "pimsr-mtdlpy-common-retrain-checkpoint"
CHECKPOINT_SCHEMA_VERSION = 1

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
    predicted_log10_resistivity: np.ndarray
    training_summary: Mapping[str, object]
    runtime: Mapping[str, object]


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
    status = str(
        _run_git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    )
    if status:
        raise MTDLPyAdapterError("MTDLPy repository must have a clean worktree")

    tree_entry = str(
        _run_git(repo, "ls-tree", "--full-tree", "HEAD", "--", MTDLPY_DINKNET_PATH)
    )
    expected_entry = (
        f"100644 blob {MTDLPY_DINKNET_GIT_BLOB}\t{MTDLPY_DINKNET_PATH}"
    )
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
    source_identity = file_artifact_provenance(source)
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
    identity = file_artifact_provenance(path)
    if identity["sha256"] != declared_sha256:
        raise MTDLPyAdapterError(
            "local ImageNet weights do not match the declared SHA-256"
        )
    return {
        **identity,
        "source_url": IMAGENET_RESNET50_V1_URL,
        "artifact": "torchvision ResNet50 IMAGENET1K_V1",
    }


def _axis(values: np.ndarray, *, name: str, positive: bool = False) -> np.ndarray:
    result = np.asarray(values)
    if result.dtype != np.dtype("<f8") or result.ndim != 1 or result.size == 0:
        raise MTDLPyAdapterError(f"{name} must be a non-empty little-endian float64 vector")
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
    artifact = file_artifact_provenance(path)
    artifact_path = Path(str(artifact["path"]))
    from pimsr_inversion.contracts2d import validate_dataset2d

    with h5py.File(artifact_path, "r") as h5:
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
    require_file_artifact_unchanged(artifact, role=role)

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


def _check_npz_member_names(path: Path) -> None:
    try:
        with zipfile.ZipFile(path, "r") as archive:
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
    artifact = file_artifact_provenance(path)
    artifact_path = Path(str(artifact["path"]))
    _check_npz_member_names(artifact_path)
    try:
        with np.load(artifact_path, allow_pickle=False) as payload:
            if tuple(payload.files) != _OBSERVATION_KEYS:
                raise MTDLPyAdapterError(
                    "observation NPZ keys are not in the canonical contract order"
                )
            arrays = {name: np.asarray(payload[name]).copy() for name in payload.files}
    except (OSError, ValueError) as exc:
        raise MTDLPyAdapterError(f"cannot load observation NPZ: {artifact_path}") from exc
    require_file_artifact_unchanged(artifact, role="held-out observation payload")

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
    if order.dtype.kind != "U" or order.shape != (4,) or tuple(order.tolist()) != (
        OBSERVATION_CHANNEL_ORDER
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
        raise ValueError("resize input must be finite; missing values are not interpolated")
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


def _require_disjoint_samples(
    train: TrainingSplit,
    validation: TrainingSplit,
    test: HeldoutObservations,
) -> None:
    if train.generator_seed == validation.generator_seed and np.intersect1d(
        train.sample_index, validation.sample_index
    ).size:
        raise MTDLPyAdapterError(
            "train and validation (generator_seed, sample_index) identities overlap"
        )
    identities = [entry.provenance["sha256"] for entry in (train, validation, test)]
    if len(set(identities)) != len(identities):
        raise MTDLPyAdapterError("train, validation and held-out artifacts must differ")


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

    weight_path = Path(str(weights["path"]))
    weight_bytes = weight_path.read_bytes()
    require_file_artifact_unchanged(
        _artifact_core(weights), role="local ImageNet weights"
    )
    state = torch.load(io.BytesIO(weight_bytes), map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise MTDLPyAdapterError("ImageNet weights root must be a plain state dictionary")

    resnet = torchvision.models.resnet50(weights=None)
    try:
        resnet.load_state_dict(state, strict=True)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise MTDLPyAdapterError(
            "local ImageNet artifact is not an exact torchvision ResNet50 state dict"
        ) from exc

    source_path = Path(str(repository["dinknet_source"]["path"]))
    spec = importlib.util.spec_from_file_location("_pimsr_pinned_mtdlpy_dinknet", source_path)
    if spec is None or spec.loader is None:
        raise MTDLPyAdapterError("cannot load pinned MTDLPy DinkNet source")
    module = importlib.util.module_from_spec(spec)
    original_ssl_context = ssl._create_default_https_context
    original_dont_write_bytecode = sys.dont_write_bytecode
    try:
        # Executing a reviewed source file must not dirty the pinned checkout.
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
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
        repository["dinknet_source"], role="pinned MTDLPy DinkNet source"
    )
    require_file_artifact_unchanged(
        _artifact_core(weights), role="local ImageNet weights"
    )
    return model, torchvision


def _train_and_predict(
    train: TrainingSplit,
    validation: TrainingSplit,
    test: HeldoutObservations,
    repository: Mapping[str, object],
    weights: Mapping[str, object],
    *,
    seed: int,
    device_name: str,
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
    observations_test = _preprocess_observations(test.observations)
    targets_train = _resize_log10_resistivity(
        train.targets, NETWORK_GRID_SHAPE
    )
    targets_validation = _resize_log10_resistivity(
        validation.targets, NETWORK_GRID_SHAPE
    )
    preprocessing_wall_time_s = time.perf_counter() - preprocessing_start

    initialization_start = time.perf_counter()
    model, torchvision = _load_upstream_model(repository, weights, torch=torch)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
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
        batch_size=BATCH_SIZE,
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
        batch_size=BATCH_SIZE,
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
    for epoch in range(1, EPOCHS + 1):
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
    model.load_state_dict(best_state, strict=True)
    model.eval()

    inference_start = time.perf_counter()
    prediction_batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, observations_test.shape[0], BATCH_SIZE):
            batch = torch.from_numpy(
                observations_test[start : start + BATCH_SIZE]
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
    cuda_device = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else None
    )
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
        "inference_wall_time_s": inference_wall_time_s,
        "backend_wall_time_s": time.perf_counter() - backend_start,
    }
    return TrainingOutcome(
        state_dict=best_state,
        predicted_log10_resistivity=predictions,
        training_summary={
            "best_epoch": best_epoch,
            "best_validation_mse": best_loss,
            "history": history,
        },
        runtime=runtime,
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


def _write_bytes_new(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MTDLPyAdapterError(f"runtime metadata is not strict JSON: {exc}") from exc


def _publish_parts(parts: Sequence[Path], destinations: Sequence[Path]) -> None:
    published: list[tuple[Path, tuple[int, int]]] = []
    try:
        for part, destination in zip(parts, destinations, strict=True):
            part_info = part.stat(follow_symlinks=False)
            if part.is_symlink() or not stat.S_ISREG(part_info.st_mode):
                raise MTDLPyAdapterError(
                    f"staged artifact must be a regular file: {part}"
                )
            expected_identity = (int(part_info.st_dev), int(part_info.st_ino))
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
                        not destination.is_symlink()
                        and stat.S_ISREG(current.st_mode)
                        and (int(current.st_dev), int(current.st_ino))
                        == expected_identity
                    ):
                        destination.unlink()
                raise
            published.append((destination, expected_identity))
            current = destination.stat(follow_symlinks=False)
            if (
                destination.is_symlink()
                or not stat.S_ISREG(current.st_mode)
                or (int(current.st_dev), int(current.st_ino)) != expected_identity
            ):
                raise MTDLPyAdapterError(
                    f"published artifact identity mismatch: {destination}"
                )
    except BaseException as exc:
        unsafe: list[str] = []
        for destination, expected_identity in reversed(published):
            if not os.path.lexists(destination):
                continue
            current = destination.stat(follow_symlinks=False)
            if (
                destination.is_symlink()
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
    checkpoint: str | Path,
    predictions: str | Path,
    runtime: str | Path,
) -> tuple[tuple[Path, Path, Path], tuple[Path, Path, Path]]:
    paths = tuple(Path(path).resolve() for path in (checkpoint, predictions, runtime))
    if len(set(paths)) != 3:
        raise ValueError("checkpoint, prediction and runtime outputs must be distinct")
    for path, suffix in zip(paths, (".pt", ".npz", ".json"), strict=True):
        if path.suffix.lower() != suffix:
            raise ValueError(f"MTDLPy output {path} must use the {suffix} suffix")
        path.parent.mkdir(parents=True, exist_ok=True)
    existing = [path for path in paths if path.exists() or path.is_symlink()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing MTDLPy output(s): "
            + ", ".join(str(path) for path in existing)
        )
    parts = tuple(path.with_name(path.name + ".part") for path in paths)
    stale = [path for path in parts if path.exists() or path.is_symlink()]
    if stale:
        raise FileExistsError(
            "stale MTDLPy partial output(s) require inspection: "
            + ", ".join(str(path) for path in stale)
        )
    return paths, parts


def _checkpoint_bytes(value: Mapping[str, object]) -> bytes:
    import torch

    payload = io.BytesIO()
    torch.save(dict(value), payload)
    return payload.getvalue()


def run_common_retrain(
    *,
    repository_path: str | Path,
    imagenet_weights_path: str | Path,
    imagenet_weights_sha256: str,
    train_h5: str | Path,
    validation_h5: str | Path,
    observations_npz: str | Path,
    seed: int,
    device: str,
    checkpoint_out: str | Path,
    predictions_out: str | Path,
    runtime_out: str | Path,
    command: Sequence[str] | None = None,
    runner_source: str | Path | None = None,
) -> dict[str, object]:
    """Run one preregistered MTDLPy seed and immutably publish its artifacts."""
    if isinstance(seed, bool) or seed not in COMMON_RETRAIN_SEEDS:
        raise ValueError(f"seed must be one of {COMMON_RETRAIN_SEEDS}")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'")
    destinations, parts = _output_paths(
        checkpoint_out, predictions_out, runtime_out
    )
    checkpoint_path, prediction_path, _runtime_path = destinations
    checkpoint_part, prediction_part, runtime_part = parts

    started_at = datetime.now(UTC)
    wall_start = time.perf_counter()
    repository = verify_pinned_repository(repository_path)
    weights = validate_local_imagenet_weights(
        imagenet_weights_path, imagenet_weights_sha256
    )
    train = load_training_split(train_h5, role="MTDLPy training dataset")
    validation = load_training_split(
        validation_h5, role="MTDLPy validation dataset"
    )
    test = load_heldout_observations(observations_npz)
    _require_same_geometry(train, validation, where="train and validation datasets")
    _require_same_geometry(train, test, where="train and held-out observations")
    _require_disjoint_samples(train, validation, test)

    source_artifacts: dict[str, Mapping[str, object]] = {
        "train_dataset": train.provenance,
        "validation_dataset": validation.provenance,
        "heldout_observations": test.provenance,
        "imagenet_weights": weights,
        "dinknet_source": repository["dinknet_source"],
        "adapter_source": file_artifact_provenance(__file__),
    }
    if runner_source is not None:
        source_artifacts["runner_source"] = file_artifact_provenance(runner_source)

    outcome = _train_and_predict(
        train,
        validation,
        test,
        repository,
        weights,
        seed=seed,
        device_name=device,
    )
    predictions = np.asarray(outcome.predicted_log10_resistivity, dtype="<f4")
    expected_prediction_shape = (test.sample_index.size, *OUTPUT_GRID_SHAPE)
    if predictions.shape != expected_prediction_shape or not np.isfinite(
        predictions
    ).all():
        raise MTDLPyAdapterError(
            f"backend predictions must be finite with shape {expected_prediction_shape}"
        )

    for role, artifact in source_artifacts.items():
        require_file_artifact_unchanged(_artifact_core(artifact), role=role)
    if verify_pinned_repository(repository_path) != repository:
        raise MTDLPyAdapterError("pinned MTDLPy repository changed during the run")

    training_config = {
        "campaign_seeds": list(COMMON_RETRAIN_SEEDS),
        "seed": seed,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": {
            "name": "Adam",
            "learning_rate": LEARNING_RATE,
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
        "schedule_origin": (
            "preregistered benchmark-native reviewed adapter schedule; "
            "not an MTDLPy upstream default"
        ),
    }
    preprocessing = {
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
        "training_target_resize": (
            "pow10_to_linear_then_64x48_to_32x32_then_log10"
        ),
        "training_target_axis_order": ["sample", "depth", "x"],
        "prediction_resize": (
            "pow10_to_linear_then_32x32_to_64x48_then_log10"
        ),
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
    checkpoint = {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "method": "MTDLPy/DinkNet50",
        "track": "common-retrain",
        "seed": seed,
        "model_state": outcome.state_dict,
        "training_config": training_config,
        "preprocessing": preprocessing,
        "training_summary": dict(outcome.training_summary),
        "source": repository,
        "imagenet_weights": weights,
        "dataset_identities": {
            "train": {
                **dict(train.provenance),
                "generator_seed": train.generator_seed,
                "sample_identity": "(generator_seed,sample_index)",
            },
            "validation": {
                **dict(validation.provenance),
                "generator_seed": validation.generator_seed,
                "sample_identity": "(generator_seed,sample_index)",
            },
            "heldout_observations": dict(test.provenance),
        },
        "heldout_truth_available_to_adapter": False,
    }
    try:
        _write_bytes_new(checkpoint_part, _checkpoint_bytes(checkpoint))
        _write_bytes_new(
            prediction_part,
            _prediction_npz_bytes(
                str(test.provenance["sha256"]),
                test.sample_index,
                test.x_grid,
                test.depth_grid,
                predictions,
            ),
        )
        checkpoint_staged = file_artifact_provenance(checkpoint_part)
        prediction_staged = file_artifact_provenance(prediction_part)
        checkpoint_identity = {
            **checkpoint_staged,
            "path": str(checkpoint_path),
        }
        prediction_identity = {
            **prediction_staged,
            "path": str(prediction_path),
        }

        finished_at = datetime.now(UTC)
        runtime = {
            "schema": RUNTIME_SCHEMA,
            "schema_version": RUNTIME_SCHEMA_VERSION,
            "method": "MTDLPy/DinkNet50",
            "track": "common-retrain",
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
            "training_summary": dict(outcome.training_summary),
            "runtime": dict(outcome.runtime),
            "observation_contract": {
                "schema": OBSERVATION_SCHEMA,
                "schema_version": PAYLOAD_SCHEMA_VERSION,
                "truth_keys_accepted": False,
                "sample_count": int(test.sample_index.size),
                "sample_index_sha256": hashlib.sha256(
                    np.asarray(test.sample_index, dtype="<i8").tobytes()
                ).hexdigest(),
                "evaluation_floor_role": "scorer_only_not_model_input",
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
                "contains_truth": False,
            },
            "outputs": {
                "checkpoint": checkpoint_identity,
                "predictions": prediction_identity,
            },
        }
        for role, artifact in source_artifacts.items():
            require_file_artifact_unchanged(_artifact_core(artifact), role=role)
        require_file_artifact_unchanged(
            checkpoint_staged, role="staged checkpoint"
        )
        require_file_artifact_unchanged(
            prediction_staged, role="staged predictions"
        )
        if verify_pinned_repository(repository_path) != repository:
            raise MTDLPyAdapterError(
                "pinned MTDLPy repository changed before publication"
            )
        runtime_payload = _canonical_json_bytes(runtime)
        _write_bytes_new(runtime_part, runtime_payload)
        if runtime_part.read_bytes() != runtime_payload:
            raise MTDLPyAdapterError("staged runtime metadata changed after writing")
        _publish_parts(parts, destinations)
        return runtime
    finally:
        for part in parts:
            part.unlink(missing_ok=True)


__all__ = [
    "COMMON_RETRAIN_SEEDS",
    "HeldoutObservations",
    "MTDLPyAdapterError",
    "TrainingSplit",
    "load_heldout_observations",
    "load_training_split",
    "resize_bilinear_half_pixel",
    "run_common_retrain",
    "validate_local_imagenet_weights",
    "verify_pinned_repository",
]
