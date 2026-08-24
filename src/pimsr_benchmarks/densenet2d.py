"""Fail-closed common-retraining adapter for MT2DInv-DenseNet v1.2.

Only six reviewed architecture class definitions are compiled from the pinned
upstream source.  In particular, the upstream module's dataset reads, training
loop, logging, and other top-level side effects are never imported or executed.
Training accepts only the public train and validation artifacts.  A separately
invoked inference phase reads one immutable checkpoint plus one truth-free
observation NPZ, so the same seed checkpoint is reusable across every campaign.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
import platform
import random
import stat
import subprocess
import time
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from pimsr_benchmarks import dataset2d_materialization as _materializer_contracts
from pimsr_benchmarks import mtdlpy as _shared_contracts
from pimsr_benchmarks import runner2d as _artifact_guards
from pimsr_benchmarks.dataset2d_materialization import (
    OBSERVATION_CHANNEL_ORDER,
    OBSERVATION_SCHEMA,
    PAYLOAD_SCHEMA_VERSION,
)
from pimsr_benchmarks.mtdlpy import (
    HeldoutObservations,
    MTDLPyAdapterError,
    TrainingSplit,
)
from pimsr_benchmarks.runner2d import (
    file_artifact_provenance,
    require_file_artifact_unchanged,
)

MT2DINV_DENSENET_REPOSITORY_URL = "https://github.com/Geo-huang/MT2DInv-DenseNet.git"
MT2DINV_DENSENET_COMMIT = "9fc46d91c40f8a1a73155c84950689d0fb92662a"
MT2DINV_DENSENET_RELEASE_TAGS = ("v1.1", "v1.2")
MT2DINV_DENSENET_SOURCE_PATH = "Improved Densenet/train_MTinv_iDenseNet.py"
MT2DINV_DENSENET_SOURCE_GIT_BLOB = "5e491a574e9b74f0d41854ac8c606c1aa80dbcb8"
MT2DINV_DENSENET_SOURCE_SHA256 = (
    "79d6d712b01e8b4a4fc14b046b5a2c277e1bde4ab56c34080a80b81da7237043"
)

ARCHITECTURE_CLASS_NAMES = (
    "ICBAM",
    "ChannelAttention",
    "SpatialAttention",
    "DenseBlock",
    "TransitionLayer",
    "DenseNetWithICBAM",
)
MODEL_BLOCKS = (6, 12, 24, 16)
MODEL_GROWTH_RATE = 32
MODEL_OUTPUT_FEATURES = 2176
MODEL_PARAMETER_COUNT = 25_908_034

COMMON_RETRAIN_SEEDS = (101, 102, 103, 104, 105)
INPUT_GRID_SHAPE = (8, 12)
NETWORK_INPUT_SHAPE = (16, 33, 4)
NETWORK_OUTPUT_SHAPE = (34, 64)
OUTPUT_GRID_SHAPE = (64, 48)

EPOCHS = 200
BATCH_SIZE = 100
LEARNING_RATE = 1e-4
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
WEIGHT_DECAY = 0.0
AMSGRAD = False
BACKGROUND_RESISTIVITY_OHM_M = 300.0
BACKGROUND_LOG10_RESISTIVITY = float(np.log10(BACKGROUND_RESISTIVITY_OHM_M))
BACKGROUND_LOSS_MULTIPLIER = 1.0
NON_BACKGROUND_LOSS_MULTIPLIER = 10.0

METHOD_ID = "mt2dinv_densenet"
METHOD_NAME = "MT2DInv-DenseNet/iDenseNet"
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
RUNTIME_SCHEMA = "pimsr-mt2dinv-densenet-common-retrain-runtime"
RUNTIME_SCHEMA_VERSION = 2
CHECKPOINT_SCHEMA = "pimsr-mt2dinv-densenet-common-retrain-checkpoint"
CHECKPOINT_SCHEMA_VERSION = 2


class DenseNet2DAdapterError(RuntimeError):
    """Raised when an MT2DInv-DenseNet run cannot prove its contract."""


@dataclass(frozen=True)
class TrainingOutcome:
    """Small internal value object kept free of Torch-specific types."""

    state_dict: Mapping[str, Any]
    training_summary: Mapping[str, object]
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


def _artifact_core(value: Mapping[str, object]) -> dict[str, object]:
    try:
        return {
            "path": value["path"],
            "sha256": value["sha256"],
            "size_bytes": value["size_bytes"],
        }
    except KeyError as exc:
        raise DenseNet2DAdapterError("artifact provenance is incomplete") from exc


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
        command = " ".join(arguments)
        raise DenseNet2DAdapterError(
            f"failed to verify pinned MT2DInv-DenseNet repository with git {command}"
        ) from exc
    if binary:
        return result.stdout
    try:
        return result.stdout.decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise DenseNet2DAdapterError(
            "git returned non-UTF-8 repository metadata"
        ) from exc


def verify_pinned_repository(path: str | Path) -> dict[str, object]:
    """Prove exact origin, commit, clean tree, and reviewed source bytes."""
    repo = Path(path).resolve(strict=True)
    if not repo.is_dir():
        raise NotADirectoryError(
            f"MT2DInv-DenseNet repository is not a directory: {repo}"
        )

    top = Path(str(_run_git(repo, "rev-parse", "--show-toplevel"))).resolve(strict=True)
    if not os.path.samefile(repo, top):
        raise DenseNet2DAdapterError("MT2DInv-DenseNet path must be the repository root")
    commit = str(_run_git(repo, "rev-parse", "--verify", "HEAD^{commit}"))
    if commit != MT2DINV_DENSENET_COMMIT:
        raise DenseNet2DAdapterError(
            f"MT2DInv-DenseNet HEAD is {commit}; required {MT2DINV_DENSENET_COMMIT}"
        )
    tag_commits: dict[str, str] = {}
    for tag in MT2DINV_DENSENET_RELEASE_TAGS:
        tag_commit = str(
            _run_git(repo, "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}")
        )
        if tag_commit != MT2DINV_DENSENET_COMMIT:
            raise DenseNet2DAdapterError(
                f"MT2DInv-DenseNet tag {tag} does not resolve to the pinned commit"
            )
        tag_commits[tag] = tag_commit
    remotes = str(_run_git(repo, "remote", "get-url", "--all", "origin")).splitlines()
    if remotes != [MT2DINV_DENSENET_REPOSITORY_URL]:
        raise DenseNet2DAdapterError(
            "MT2DInv-DenseNet origin URL must be exactly "
            f"{MT2DINV_DENSENET_REPOSITORY_URL!r}, got {remotes!r}"
        )
    status = str(_run_git(repo, "status", "--porcelain=v1", "--untracked-files=all"))
    if status:
        raise DenseNet2DAdapterError(
            "MT2DInv-DenseNet repository must have a clean worktree"
        )

    tree_entry = str(
        _run_git(
            repo,
            "ls-tree",
            "--full-tree",
            "HEAD",
            "--",
            MT2DINV_DENSENET_SOURCE_PATH,
        )
    )
    expected_entry = (
        f"100644 blob {MT2DINV_DENSENET_SOURCE_GIT_BLOB}\t{MT2DINV_DENSENET_SOURCE_PATH}"
    )
    if tree_entry != expected_entry:
        raise DenseNet2DAdapterError(
            "pinned MT2DInv-DenseNet source tree entry is not the reviewed blob"
        )
    blob = _run_git(
        repo,
        "cat-file",
        "blob",
        f"{MT2DINV_DENSENET_COMMIT}:{MT2DINV_DENSENET_SOURCE_PATH}",
        binary=True,
    )
    assert isinstance(blob, bytes)
    blob_sha256 = hashlib.sha256(blob).hexdigest()
    if blob_sha256 != MT2DINV_DENSENET_SOURCE_SHA256:
        raise DenseNet2DAdapterError("pinned MT2DInv-DenseNet Git blob SHA-256 changed")

    source = repo / MT2DINV_DENSENET_SOURCE_PATH
    source_identity = file_artifact_provenance(source)
    if source_identity["sha256"] != MT2DINV_DENSENET_SOURCE_SHA256:
        raise DenseNet2DAdapterError(
            "checked-out MT2DInv-DenseNet source differs from the reviewed blob"
        )
    return {
        "path": str(repo),
        "origin_url": MT2DINV_DENSENET_REPOSITORY_URL,
        "commit": commit,
        "clean_worktree": True,
        "release_tags_reviewed": list(MT2DINV_DENSENET_RELEASE_TAGS),
        "release_tag_commits": tag_commits,
        "architecture_source": source_identity,
        "architecture_git_blob_sha1": MT2DINV_DENSENET_SOURCE_GIT_BLOB,
        "architecture_git_blob_sha256": blob_sha256,
    }


def load_training_split(path: str | Path, *, role: str) -> TrainingSplit:
    """Load one schema-v2 split from bytes pinned by a single file descriptor."""
    snapshot = _snapshot_regular_file(path, role=role)
    return _load_training_split_snapshot(snapshot, role=role)


def load_heldout_observations(path: str | Path) -> HeldoutObservations:
    """Load the exact truth-free NPZ from one descriptor-pinned byte snapshot."""
    snapshot = _snapshot_regular_file(path, role="truth-free observations")
    return _load_heldout_observations_snapshot(snapshot)


def _linear_interpolate_axis(
    values: np.ndarray,
    source_coordinates: np.ndarray,
    target_coordinates: np.ndarray,
    *,
    axis: int,
) -> np.ndarray:
    """Vectorized, deterministic linear interpolation without extrapolation."""
    array = np.asarray(values, dtype=np.float64)
    source = np.asarray(source_coordinates, dtype=np.float64)
    target = np.asarray(target_coordinates, dtype=np.float64)
    if source.ndim != 1 or target.ndim != 1 or source.size < 2 or target.size < 1:
        raise ValueError("interpolation coordinates must be non-empty vectors")
    if (
        not np.isfinite(array).all()
        or not np.isfinite(source).all()
        or not np.isfinite(target).all()
    ):
        raise DenseNet2DAdapterError("interpolation inputs must be finite")
    if np.any(np.diff(source) <= 0) or np.any(np.diff(target) <= 0):
        raise DenseNet2DAdapterError(
            "interpolation coordinates must be strictly increasing"
        )
    if target[0] < source[0] or target[-1] > source[-1]:
        raise DenseNet2DAdapterError("coordinate interpolation cannot extrapolate")
    normalized_axis = axis % array.ndim
    if array.shape[normalized_axis] != source.size:
        raise ValueError("source coordinate size does not match interpolation axis")

    upper = np.searchsorted(source, target, side="right")
    upper = np.clip(upper, 1, source.size - 1)
    lower = upper - 1
    fraction = (target - source[lower]) / (source[upper] - source[lower])
    shape = [1] * array.ndim
    shape[normalized_axis] = target.size
    low_values = np.take(array, lower, axis=normalized_axis)
    high_values = np.take(array, upper, axis=normalized_axis)
    result = low_values + (high_values - low_values) * fraction.reshape(shape)
    if not np.isfinite(result).all():
        raise DenseNet2DAdapterError("coordinate interpolation produced non-finite data")
    return result


def _network_observation_axes(
    frequencies: np.ndarray,
    station_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    frequency_axis = np.geomspace(
        float(frequencies[0]),
        float(frequencies[-1]),
        NETWORK_INPUT_SHAPE[0],
        dtype=np.float64,
    )
    station_axis = np.linspace(
        float(station_x[0]),
        float(station_x[-1]),
        NETWORK_INPUT_SHAPE[1],
        dtype=np.float64,
    )
    return frequency_axis, station_axis


def preprocess_observations(
    values: np.ndarray,
    frequencies: np.ndarray,
    station_x: np.ndarray,
) -> np.ndarray:
    """Adapt PIMSR observations to native ``N,C=16,H=33,W=4`` semantics."""
    array = np.asarray(values)
    expected_tail = (len(OBSERVATION_CHANNEL_ORDER), *INPUT_GRID_SHAPE)
    if array.ndim != 4 or array.shape[1:] != expected_tail:
        raise DenseNet2DAdapterError(
            "observations must have sample, component, frequency, station axes "
            f"with tail {expected_tail}"
        )
    if not np.isfinite(array).all():
        raise DenseNet2DAdapterError("observations must be finite")
    frequency = np.asarray(frequencies, dtype=np.float64)
    station = np.asarray(station_x, dtype=np.float64)
    if frequency.shape != (INPUT_GRID_SHAPE[0],) or np.any(frequency <= 0):
        raise DenseNet2DAdapterError("frequency axis is invalid")
    if station.shape != (INPUT_GRID_SHAPE[1],):
        raise DenseNet2DAdapterError("station axis is invalid")
    target_frequency, target_station = _network_observation_axes(frequency, station)
    resampled = _linear_interpolate_axis(
        array,
        np.log10(frequency),
        np.log10(target_frequency),
        axis=2,
    )
    resampled = _linear_interpolate_axis(
        resampled,
        station,
        target_station,
        axis=3,
    )
    native = np.transpose(resampled, (0, 2, 3, 1))
    centered = native - native.mean(axis=(1, 2), keepdims=True, dtype=np.float64)
    scale = np.max(np.abs(centered), axis=(1, 2), keepdims=True)
    if not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise DenseNet2DAdapterError(
            "per-sample component max-abs normalization is undefined for a "
            "constant or non-finite channel"
        )
    normalized = (centered / scale).astype(np.float32)
    if normalized.shape[1:] != NETWORK_INPUT_SHAPE or not np.isfinite(normalized).all():
        raise DenseNet2DAdapterError("observation preprocessing produced invalid data")
    return np.ascontiguousarray(normalized)


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
        raise DenseNet2DAdapterError("resize input must be finite")
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
        raise DenseNet2DAdapterError("bilinear resize produced non-finite data")
    return result


def resize_log10_resistivity(
    values: np.ndarray,
    output_shape: tuple[int, int],
) -> np.ndarray:
    """Resize resistivity in linear ohm-m, then return finite log10 values."""
    log10_values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(log10_values).all():
        raise DenseNet2DAdapterError("log10 resistivity values must be finite")
    try:
        with np.errstate(over="raise", under="ignore", invalid="raise"):
            linear = np.power(10.0, log10_values)
    except FloatingPointError as exc:
        raise DenseNet2DAdapterError(
            "log10 resistivity cannot be represented in linear float64"
        ) from exc
    if not np.isfinite(linear).all() or np.any(linear <= 0.0):
        raise DenseNet2DAdapterError(
            "log10 resistivity must map to positive finite linear values"
        )
    resized = resize_bilinear_half_pixel(
        linear,
        output_shape,
        _output_dtype=np.float64,
    )
    result = np.log10(resized).astype(np.float32)
    if not np.isfinite(result).all():
        raise DenseNet2DAdapterError("resized log10 resistivity is non-finite")
    return result


def _require_same_geometry(
    left: TrainingSplit | HeldoutObservations,
    right: TrainingSplit | HeldoutObservations,
    *,
    where: str,
) -> None:
    for name in ("frequencies", "station_x", "x_grid", "depth_grid"):
        left_axis = getattr(left, name)
        right_axis = getattr(right, name)
        if left_axis.shape != right_axis.shape or not np.array_equal(
            left_axis, right_axis
        ):
            raise DenseNet2DAdapterError(f"{where} have different {name} axes")


def _require_disjoint_training_splits(
    train: TrainingSplit,
    validation: TrainingSplit,
) -> None:
    if (
        train.generator_seed == validation.generator_seed
        and np.intersect1d(train.sample_index, validation.sample_index).size
    ):
        raise DenseNet2DAdapterError(
            "train and validation (generator_seed, sample_index) identities overlap"
        )
    if train.provenance["sha256"] == validation.provenance["sha256"]:
        raise DenseNet2DAdapterError("train and validation artifacts must differ")


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


def _architecture_nodes(source: bytes, *, filename: str) -> list[ast.ClassDef]:
    """Select exactly the six reviewed architecture definitions from an AST."""
    try:
        tree = ast.parse(source, filename=filename)
    except (SyntaxError, ValueError) as exc:
        raise DenseNet2DAdapterError(
            "reviewed architecture source cannot be parsed"
        ) from exc
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in ARCHITECTURE_CLASS_NAMES
    ]
    names = tuple(node.name for node in selected)
    if names != ARCHITECTURE_CLASS_NAMES or len(set(names)) != len(names):
        raise DenseNet2DAdapterError(
            "reviewed source does not contain the exact six architecture classes"
        )
    return selected


def _load_upstream_model(
    repository: Mapping[str, object],
    *,
    torch: Any,
) -> Any:
    """Compile only reviewed class definitions and instantiate exact iDenseNet."""
    source_identity = repository.get("architecture_source")
    if not isinstance(source_identity, Mapping):
        raise DenseNet2DAdapterError("repository architecture provenance is missing")
    source_path = Path(str(source_identity.get("path", "")))
    source = source_path.read_bytes()
    require_file_artifact_unchanged(
        _artifact_core(source_identity), role="pinned MT2DInv-DenseNet source"
    )
    if hashlib.sha256(source).hexdigest() != MT2DINV_DENSENET_SOURCE_SHA256:
        raise DenseNet2DAdapterError("reviewed architecture source SHA-256 changed")
    nodes = _architecture_nodes(source, filename=str(source_path))
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    namespace: dict[str, object] = {
        "__name__": "_pimsr_reviewed_mt2dinv_densenet_architecture",
        "nn": torch.nn,
        "torch": torch,
        "F": torch.nn.functional,
    }
    try:
        exec(compile(module, str(source_path), "exec"), namespace)  # noqa: S102
        model_type = namespace["DenseNetWithICBAM"]
        model = model_type(
            num_blocks=list(MODEL_BLOCKS),
            growth_rate=MODEL_GROWTH_RATE,
            num_classes=MODEL_OUTPUT_FEATURES,
        )
    except (AssertionError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise DenseNet2DAdapterError(
            "cannot instantiate the reviewed MT2DInv-DenseNet architecture"
        ) from exc
    parameter_count = sum(int(parameter.numel()) for parameter in model.parameters())
    if parameter_count != MODEL_PARAMETER_COUNT:
        raise DenseNet2DAdapterError(
            f"architecture has {parameter_count} parameters; expected "
            f"{MODEL_PARAMETER_COUNT}"
        )
    require_file_artifact_unchanged(
        _artifact_core(source_identity), role="pinned MT2DInv-DenseNet source"
    )
    return model


def _weighted_mse(torch: Any, predictions: Any, targets: Any) -> Any:
    """Reproduce the upstream 10x non-background elementwise MSE."""
    background = targets.new_tensor(BACKGROUND_LOG10_RESISTIVITY)
    weights = torch.where(
        targets != background,
        targets.new_tensor(NON_BACKGROUND_LOSS_MULTIPLIER),
        targets.new_tensor(BACKGROUND_LOSS_MULTIPLIER),
    )
    return ((predictions - targets).square() * weights).mean()


def _train_model(
    train: TrainingSplit,
    validation: TrainingSplit,
    repository: Mapping[str, object],
    *,
    seed: int,
    device_name: str,
) -> TrainingOutcome:
    backend_start = time.perf_counter()
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    _configure_determinism(torch, seed)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise DenseNet2DAdapterError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    preprocessing_start = time.perf_counter()
    observations_train = preprocess_observations(
        train.observations, train.frequencies, train.station_x
    )
    observations_validation = preprocess_observations(
        validation.observations, validation.frequencies, validation.station_x
    )
    targets_train = resize_log10_resistivity(train.targets, NETWORK_OUTPUT_SHAPE)
    targets_validation = resize_log10_resistivity(
        validation.targets, NETWORK_OUTPUT_SHAPE
    )
    preprocessing_wall_time_s = time.perf_counter() - preprocessing_start

    initialization_start = time.perf_counter()
    model = _load_upstream_model(repository, torch=torch)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=ADAM_BETAS,
        eps=ADAM_EPS,
        weight_decay=WEIGHT_DECAY,
        amsgrad=AMSGRAD,
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(observations_train),
            torch.from_numpy(targets_train),
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
            torch.from_numpy(targets_validation),
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    initialization_wall_time_s = time.perf_counter() - initialization_start

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
            raw_predictions = model(observations)
            expected = (observations.shape[0], MODEL_OUTPUT_FEATURES)
            if tuple(raw_predictions.shape) != expected or not bool(
                torch.isfinite(raw_predictions).all()
            ):
                raise DenseNet2DAdapterError(
                    "iDenseNet produced an invalid training prediction"
                )
            predictions = raw_predictions.reshape(
                observations.shape[0], *NETWORK_OUTPUT_SHAPE
            )
            loss = _weighted_mse(torch, predictions, targets)
            if not bool(torch.isfinite(loss)):
                raise DenseNet2DAdapterError("iDenseNet training loss is non-finite")
            loss.backward()
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
                raw_predictions = model(observations)
                expected = (observations.shape[0], MODEL_OUTPUT_FEATURES)
                if tuple(raw_predictions.shape) != expected or not bool(
                    torch.isfinite(raw_predictions).all()
                ):
                    raise DenseNet2DAdapterError(
                        "iDenseNet produced an invalid validation prediction"
                    )
                predictions = raw_predictions.reshape(
                    observations.shape[0], *NETWORK_OUTPUT_SHAPE
                )
                loss = _weighted_mse(torch, predictions, targets)
                if not bool(torch.isfinite(loss)):
                    raise DenseNet2DAdapterError(
                        "iDenseNet validation loss is non-finite"
                    )
                batch_count = int(observations.shape[0])
                validation_loss_sum += float(loss.cpu()) * batch_count
                validation_count += batch_count
        train_loss = train_loss_sum / train_count
        validation_loss = validation_loss_sum / validation_count
        history.append(
            {
                "epoch": epoch,
                "train_weighted_mse": train_loss,
                "validation_weighted_mse": validation_loss,
            }
        )
        # Strict inequality deliberately makes the first exact tie win.
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
        raise DenseNet2DAdapterError(
            "training produced no validation-selected checkpoint"
        )
    model.load_state_dict(best_state, strict=True)
    model.eval()

    peak_cuda_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    cuda_device = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    return TrainingOutcome(
        state_dict=best_state,
        training_summary={
            "best_epoch": best_epoch,
            "best_validation_weighted_mse": best_loss,
            "history": history,
        },
        runtime={
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": str(torch.__version__),
            "torch_cuda_build": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "device": str(device),
            "cuda_device_name": cuda_device,
            "peak_cuda_memory_bytes": peak_cuda_memory,
            "preprocessing_wall_time_s": preprocessing_wall_time_s,
            "model_initialization_wall_time_s": initialization_wall_time_s,
            "training_wall_time_s": training_wall_time_s,
            "backend_wall_time_s": time.perf_counter() - backend_start,
        },
    )


def _predict_from_checkpoint(
    test: HeldoutObservations,
    repository: Mapping[str, object],
    model_state: Mapping[str, Any],
    *,
    seed: int,
    device_name: str,
) -> tuple[np.ndarray, dict[str, object]]:
    """Run inference without accepting a training target or campaign truth."""
    backend_start = time.perf_counter()
    import torch

    _configure_determinism(torch, seed)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise DenseNet2DAdapterError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    preprocessing_start = time.perf_counter()
    observations = preprocess_observations(
        test.observations, test.frequencies, test.station_x
    )
    preprocessing_wall_time_s = time.perf_counter() - preprocessing_start

    initialization_start = time.perf_counter()
    model = _load_upstream_model(repository, torch=torch)
    try:
        model.load_state_dict(dict(model_state), strict=True)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise DenseNet2DAdapterError(
            "checkpoint model_state is incompatible with the pinned architecture"
        ) from exc
    model.to(device)
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    initialization_wall_time_s = time.perf_counter() - initialization_start

    inference_start = time.perf_counter()
    native_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, observations.shape[0], BATCH_SIZE):
            batch = torch.from_numpy(observations[start : start + BATCH_SIZE]).to(device)
            raw_predictions = model(batch)
            expected = (batch.shape[0], MODEL_OUTPUT_FEATURES)
            if tuple(raw_predictions.shape) != expected or not bool(
                torch.isfinite(raw_predictions).all()
            ):
                raise DenseNet2DAdapterError(
                    "iDenseNet produced an invalid held-out prediction"
                )
            native_batches.append(
                raw_predictions.reshape(batch.shape[0], *NETWORK_OUTPUT_SHAPE)
                .cpu()
                .numpy()
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_wall_time_s = time.perf_counter() - inference_start
    predictions = resize_log10_resistivity(
        np.concatenate(native_batches, axis=0), OUTPUT_GRID_SHAPE
    )
    peak_cuda_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    cuda_device = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    runtime = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": str(torch.__version__),
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
        "cuda_device_name": cuda_device,
        "peak_cuda_memory_bytes": peak_cuda_memory,
        "preprocessing_wall_time_s": preprocessing_wall_time_s,
        "model_initialization_wall_time_s": initialization_wall_time_s,
        "inference_wall_time_s": inference_wall_time_s,
        "backend_wall_time_s": time.perf_counter() - backend_start,
    }
    return predictions, runtime


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
        ("x_cell_centers_m", np.asarray(x_cell_centers_m, dtype="<f8")),
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
        raise DenseNet2DAdapterError(
            f"runtime metadata is not strict JSON: {exc}"
        ) from exc


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
        raise DenseNet2DAdapterError(f"{role} changed after validation")


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
            raise DenseNet2DAdapterError(
                "refusing to delete outputs replaced during rollback: "
                + ", ".join(unsafe)
            ) from exc
        raise


def _checkpoint_bytes(value: Mapping[str, object]) -> bytes:
    import torch

    payload = io.BytesIO()
    torch.save(dict(value), payload)
    return payload.getvalue()


def run_common_retrain(*_args: object, **_kwargs: object) -> dict[str, object]:
    """Reject the retired campaign-specific train-and-infer API."""
    raise DenseNet2DAdapterError(
        "combined training and campaign inference was removed; use "
        "train_common_retrain followed by run_common_inference"
    )


_CHECKPOINT_KEYS = frozenset(
    {
        "checkpoint_schema",
        "checkpoint_schema_version",
        "method_id",
        "method",
        "track",
        "seed",
        "model",
        "model_state",
        "training_config",
        "preprocessing",
        "training_summary",
        "training_runtime",
        "source",
        "dependency_closure",
        "dataset_identities",
        "data_geometry",
        "campaign_observations_accepted_for_training",
        "truth_keys_accepted",
        "contains_truth",
        "contains_observation_campaign",
    }
)
_MODEL_KEYS = frozenset(
    {"class", "blocks", "growth_rate", "output_features", "parameter_count"}
)
_DATASET_IDENTITY_KEYS = frozenset(
    {"path", "sha256", "size_bytes", "sample_count", "sample_index_sha256"}
)
_GEOMETRY_KEYS = frozenset(
    {"frequency_hz", "station_x_m", "x_cell_centers_m", "depth_cell_centers_m"}
)
_DEPENDENCY_CLOSURE_SCHEMA = "pimsr-mt2dinv-densenet-source-dependency-closure"
_DEPENDENCY_CLOSURE_VERSION = 2
_COMMON_BACKEND_RUNTIME_KEYS = frozenset(
    {
        "python",
        "platform",
        "numpy",
        "torch",
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
_TRAINING_RUNTIME_KEYS = _COMMON_BACKEND_RUNTIME_KEYS | {"training_wall_time_s"}
_INFERENCE_RUNTIME_KEYS = _COMMON_BACKEND_RUNTIME_KEYS | {"inference_wall_time_s"}


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], role: str
) -> None:
    actual = set(value)
    if actual != set(expected):
        raise DenseNet2DAdapterError(
            f"{role} keys mismatch; missing={sorted(set(expected) - actual)}, "
            f"extra={sorted(actual - set(expected))}"
        )


def _require_sha256(value: object, role: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DenseNet2DAdapterError(
            f"{role} must be 64 lowercase hexadecimal characters"
        )
    return value


_SAFE_FALSE_PRESCORE_DECLARATIONS = frozenset(
    {
        "contains_truth",
        "heldout_truth_available_to_adapter",
        "truth_keys_accepted",
    }
)


def _metadata_key_is_forbidden(key: str, value: object) -> bool:
    normalized = key.lower().replace("-", "_")
    if normalized in _SAFE_FALSE_PRESCORE_DECLARATIONS and value is False:
        return False
    if normalized in {"generator_seed", "generator_seeds"}:
        return True
    tokens = normalized.split("_")
    if any(
        token in {"hidden", "operator", "secret", "truth", "withheld"} for token in tokens
    ):
        return True
    return any(
        left == "sample" and right in {"id", "ids"} for left, right in pairwise(tokens)
    )


def _require_no_prescore_metadata(value: object, *, role: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise DenseNet2DAdapterError(f"{role} contains a non-string metadata key")
            if _metadata_key_is_forbidden(key, child):
                raise DenseNet2DAdapterError(
                    f"{role} exposes forbidden pre-score metadata key {key!r}"
                )
            _require_no_prescore_metadata(child, role=role)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            _require_no_prescore_metadata(child, role=role)


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
) -> _ArtifactSnapshot:
    requested = Path(os.path.abspath(os.fspath(path)))
    expected = (
        None
        if expected_sha256 is None
        else _require_sha256(expected_sha256, f"expected {role} SHA-256")
    )
    try:
        path_before = requested.lstat()
    except OSError as exc:
        raise DenseNet2DAdapterError(f"cannot stat {role}: {requested}") from exc
    if _path_is_link(requested):
        raise DenseNet2DAdapterError(f"{role} must not be a symbolic link or junction")
    if not stat.S_ISREG(path_before.st_mode):
        raise DenseNet2DAdapterError(f"{role} must be a regular file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(requested, flags)
    except OSError as exc:
        raise DenseNet2DAdapterError(
            f"cannot open {role} without following links"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise DenseNet2DAdapterError(f"{role} must be a regular file")
        if _stat_identity(path_before) != _stat_identity(opened):
            raise DenseNet2DAdapterError(f"{role} changed before it was opened")
        try:
            resolved = requested.resolve(strict=True)
            resolved_info = resolved.stat(follow_symlinks=False)
        except OSError as exc:
            raise DenseNet2DAdapterError(f"cannot resolve opened {role}") from exc
        if _path_is_link(resolved) or _stat_identity(resolved_info) != _stat_identity(
            opened
        ):
            raise DenseNet2DAdapterError(f"{role} path does not identify the opened file")
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
            raise DenseNet2DAdapterError(
                f"{role} path disappeared while it was read"
            ) from exc
        if (
            _path_is_link(requested)
            or _stat_identity(opened) != _stat_identity(after_descriptor)
            or _stat_identity(opened) != _stat_identity(path_after)
            or len(payload) != int(opened.st_size)
        ):
            raise DenseNet2DAdapterError(f"{role} changed while it was read")
    finally:
        os.close(descriptor)
    digest = hashlib.sha256(payload).hexdigest()
    if expected is not None and digest != expected:
        raise DenseNet2DAdapterError(f"{role} SHA-256 differs from the pinned digest")
    return _ArtifactSnapshot(
        path=resolved,
        payload=payload,
        sha256=digest,
        device=int(opened.st_dev),
        inode=int(opened.st_ino),
    )


def _load_training_split_snapshot(
    snapshot: _ArtifactSnapshot,
    *,
    role: str,
) -> TrainingSplit:
    import h5py
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
        raise DenseNet2DAdapterError(
            f"cannot load {role} from pinned bytes: {exc}"
        ) from exc

    frequencies = np.asarray(contract.frequencies, dtype="<f8")
    station_x = np.asarray(contract.station_x, dtype="<f8")
    x_grid = np.asarray(contract.x_grid, dtype="<f8")
    depth_grid = np.asarray(contract.depth_grid, dtype="<f8")
    try:
        _shared_contracts._require_geometry_shape(
            frequencies,
            station_x,
            x_grid,
            depth_grid,
            where=role,
        )
    except MTDLPyAdapterError as exc:
        raise DenseNet2DAdapterError(str(exc).replace("MTDLPy", METHOD_NAME)) from exc
    if observations.shape != (sample_index.size, 4, *INPUT_GRID_SHAPE):
        raise DenseNet2DAdapterError(f"{role} has an unexpected observation tensor shape")
    if targets.shape != (sample_index.size, *OUTPUT_GRID_SHAPE):
        raise DenseNet2DAdapterError(f"{role} has an unexpected target tensor shape")
    if not np.isfinite(observations).all() or not np.isfinite(targets).all():
        raise DenseNet2DAdapterError(f"{role} arrays must be finite")
    return TrainingSplit(
        observations=observations,
        targets=targets,
        sample_index=sample_index,
        generator_seed=generator_seed,
        frequencies=frequencies,
        station_x=station_x,
        x_grid=x_grid,
        depth_grid=depth_grid,
        provenance=_snapshot_core(snapshot),
    )


def _load_heldout_observations_snapshot(
    snapshot: _ArtifactSnapshot,
) -> HeldoutObservations:
    expected_keys = tuple(_shared_contracts._OBSERVATION_KEYS)
    try:
        with zipfile.ZipFile(io.BytesIO(snapshot.payload), "r") as archive:
            names = archive.namelist()
        expected_names = [f"{name}.npy" for name in expected_keys]
        if names != expected_names or len(names) != len(set(names)):
            raise DenseNet2DAdapterError(
                "observation NPZ members must exactly match the ordered truth-free contract"
            )
        with np.load(io.BytesIO(snapshot.payload), allow_pickle=False) as payload:
            if tuple(payload.files) != expected_keys:
                raise DenseNet2DAdapterError(
                    "observation NPZ keys are not in the canonical contract order"
                )
            arrays = {name: np.asarray(payload[name]).copy() for name in payload.files}
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise DenseNet2DAdapterError(
            f"cannot load truth-free observations from pinned bytes: {exc}"
        ) from exc

    try:
        schema = _shared_contracts._scalar_string(arrays["schema"], "schema")
        if schema != OBSERVATION_SCHEMA:
            raise DenseNet2DAdapterError("unsupported held-out observation schema")
        schema_version = arrays["schema_version"]
        if (
            schema_version.shape != ()
            or schema_version.dtype != np.dtype("<i8")
            or int(schema_version) != PAYLOAD_SCHEMA_VERSION
        ):
            raise DenseNet2DAdapterError(
                "unsupported held-out observation schema version"
            )
        order = arrays["observation_channel_order"]
        if (
            order.dtype.kind != "U"
            or order.shape != (4,)
            or tuple(order.tolist()) != OBSERVATION_CHANNEL_ORDER
        ):
            raise DenseNet2DAdapterError(
                "held-out observation channel order is not canonical"
            )
        sample_index = arrays["sample_index"]
        if (
            sample_index.dtype != np.dtype("<i8")
            or sample_index.ndim != 1
            or sample_index.size == 0
            or np.any(sample_index < 0)
            or np.unique(sample_index).size != sample_index.size
        ):
            raise DenseNet2DAdapterError(
                "held-out sample_index must contain unique non-negative opaque IDs"
            )
        frequencies = _shared_contracts._axis(
            arrays["frequency_hz"], name="frequency_hz", positive=True
        )
        station_x = _shared_contracts._axis(arrays["station_x_m"], name="station_x_m")
        x_grid = _shared_contracts._axis(
            arrays["x_cell_centers_m"], name="x_cell_centers_m"
        )
        depth_grid = _shared_contracts._axis(
            arrays["depth_cell_centers_m"],
            name="depth_cell_centers_m",
            positive=True,
        )
        _shared_contracts._require_geometry_shape(
            frequencies,
            station_x,
            x_grid,
            depth_grid,
            where="held-out observations",
        )
    except (KeyError, MTDLPyAdapterError) as exc:
        raise DenseNet2DAdapterError(str(exc).replace("MTDLPy", METHOD_NAME)) from exc

    shape = (sample_index.size, *INPUT_GRID_SHAPE)
    values: list[np.ndarray] = []
    floors: list[np.ndarray] = []
    for name in _shared_contracts._OBSERVATION_VALUE_KEYS:
        array = arrays[name]
        if array.dtype != np.dtype("<f4") or array.shape != shape:
            raise DenseNet2DAdapterError(f"{name} must be float32 with shape {shape}")
        if not np.isfinite(array).all():
            raise DenseNet2DAdapterError(f"{name} must be finite")
        if "phase" in name and np.any((array < 0.0) | (array >= 180.0)):
            raise DenseNet2DAdapterError(f"{name} violates the [0, 180) convention")
        values.append(array)
    for name in _shared_contracts._OBSERVATION_FLOOR_KEYS:
        array = arrays[name]
        if array.dtype != np.dtype("<f4") or array.shape != shape:
            raise DenseNet2DAdapterError(f"{name} must be float32 with shape {shape}")
        if not np.isfinite(array).all() or np.any(array <= 0):
            raise DenseNet2DAdapterError(f"{name} must be finite and strictly positive")
        floors.append(array)
    mask = arrays["valid_mask"]
    expected_mask_shape = (sample_index.size, 4, *INPUT_GRID_SHAPE)
    if mask.dtype != np.dtype(bool) or mask.shape != expected_mask_shape:
        raise DenseNet2DAdapterError(
            f"valid_mask must be bool with shape {expected_mask_shape}"
        )
    if not bool(mask.all()):
        raise DenseNet2DAdapterError(
            f"{METHOD_NAME} common retraining requires valid_mask to be all true"
        )
    return HeldoutObservations(
        observations=np.stack(values, axis=1),
        evaluation_floors=np.stack(floors, axis=1),
        sample_index=sample_index,
        frequencies=frequencies,
        station_x=station_x,
        x_grid=x_grid,
        depth_grid=depth_grid,
        provenance=_snapshot_core(snapshot),
    )


def _snapshot_core(snapshot: _ArtifactSnapshot) -> dict[str, object]:
    return {
        "path": str(snapshot.path),
        "sha256": snapshot.sha256,
        "size_bytes": snapshot.size_bytes,
    }


def _json_sha256(value: object) -> str:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DenseNet2DAdapterError(
            f"dependency closure is not canonical JSON: {exc}"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _training_config(seed: int) -> dict[str, object]:
    return {
        "seed": seed,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": {
            "name": "Adam",
            "learning_rate": LEARNING_RATE,
            "betas": list(ADAM_BETAS),
            "eps": ADAM_EPS,
            "weight_decay": WEIGHT_DECAY,
            "amsgrad": AMSGRAD,
        },
        "scheduler": None,
        "early_stopping": None,
        "gradient_clipping": None,
        "loss": {
            "name": "weighted_mean_squared_error",
            "background_log10_resistivity": BACKGROUND_LOG10_RESISTIVITY,
            "background_multiplier": BACKGROUND_LOSS_MULTIPLIER,
            "non_background_multiplier": NON_BACKGROUND_LOSS_MULTIPLIER,
            "mask_rule": "target != log10(300)",
        },
        "checkpoint_selection": (
            "lowest validation weighted MSE; strict less-than; first tie"
        ),
        "checkpoint_selection_origin": (
            "benchmark-native validation-only adaptation; upstream saves the "
            "last epoch and does not select by validation loss"
        ),
        "normalization": (
            "per-sample per-component mean-center then max-absolute divide"
        ),
        "schedule_origin": (
            "pinned upstream semantic method-specific schedule adapted to the "
            "common train/validation split"
        ),
        "equal_compute_claim": False,
        "upstream_cli_unchanged_claim": False,
    }


def _preprocessing_contract() -> dict[str, object]:
    return {
        "input_channel_order_before_native_reorder": list(OBSERVATION_CHANNEL_ORDER),
        "benchmark_observation_axis_order": [
            "sample",
            "component",
            "frequency",
            "station",
        ],
        "network_tensor_semantics": [
            "sample",
            "conv2d_channel_frequency_16",
            "conv2d_height_station_33",
            "conv2d_width_component_4",
        ],
        "network_input_shape_excluding_batch": list(NETWORK_INPUT_SHAPE),
        "native_architecture_note": (
            "upstream passes N,16,33,4 directly to Conv2d(16,...); the four "
            "TE/TM components are the spatial W dimension, not Conv2d channels"
        ),
        "observation_geometry_adaptation": {
            "source": "8 frequencies x 12 stations",
            "target": "16 frequencies x 33 stations",
            "frequency_target": "geomspace between exact source endpoints",
            "frequency_coordinate": "log10(frequency_hz)",
            "station_target": "linspace between exact source endpoints",
            "station_coordinate": "station_x_m",
            "interpolation": "piecewise linear float64; no extrapolation",
            "origin": "benchmark-native geometry adaptation",
        },
        "normalization": {
            "scope": "independently for each sample and each of four components",
            "order": "subtract spatial mean, then divide by spatial max absolute",
            "zero_channel_policy": (
                "fail closed before Torch; upstream division would be undefined/NaN"
            ),
        },
        "training_target_adaptation": (
            "log10 64x48 -> pow10 linear ohm-m -> deterministic half-pixel "
            "bilinear 34x64 -> log10"
        ),
        "native_prediction_axis_order": ["sample", "depth_34", "x_64"],
        "prediction_geometry_adaptation": (
            "native log10 34x64 -> pow10 linear ohm-m -> deterministic "
            "half-pixel bilinear evaluation 64x48 -> log10"
        ),
        "target_and_prediction_interpolation": {
            "kind": "bilinear",
            "coordinate_transform": "half_pixel",
            "boundary": "clamp_to_edge",
            "accumulation_dtype": "float64",
            "output_dtype": "float32",
            "antialias": False,
        },
        "phase_domain_adaptation": (
            "retain benchmark canonical [0,180) phase values before the upstream "
            "per-component normalization"
        ),
        "geometry_and_phase_adaptations_are_benchmark_specific": True,
        "missing_data_policy": "reject unless valid_mask is all true",
        "test_tuning": False,
        "evaluation_floors_used_as_model_input": False,
        "upstream_cli_unchanged_claim": False,
    }


def _model_contract() -> dict[str, object]:
    return {
        "class": "DenseNetWithICBAM",
        "blocks": list(MODEL_BLOCKS),
        "growth_rate": MODEL_GROWTH_RATE,
        "output_features": MODEL_OUTPUT_FEATURES,
        "parameter_count": MODEL_PARAMETER_COUNT,
    }


def _dataset_identity(split: TrainingSplit) -> dict[str, object]:
    provenance = _artifact_core(split.provenance)
    return {
        **provenance,
        "sample_count": int(split.sample_index.size),
        "sample_index_sha256": hashlib.sha256(
            np.asarray(split.sample_index, dtype="<i8").tobytes()
        ).hexdigest(),
    }


def _data_geometry(split: TrainingSplit) -> dict[str, object]:
    return {
        "frequency_hz": np.asarray(split.frequencies, dtype="<f8").tolist(),
        "station_x_m": np.asarray(split.station_x, dtype="<f8").tolist(),
        "x_cell_centers_m": np.asarray(split.x_grid, dtype="<f8").tolist(),
        "depth_cell_centers_m": np.asarray(split.depth_grid, dtype="<f8").tolist(),
    }


def _dependency_closure(
    artifacts: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    required_sources = {
        "adapter_source",
        "artifact_guard_source",
        "architecture_source",
        "inversion_dataset_contract_source",
        "materializer_contract_source",
        "shared_contract_loader_source",
    }
    actual_sources = set(artifacts)
    if actual_sources not in (required_sources, required_sources | {"runner_source"}):
        raise DenseNet2DAdapterError(
            "dependency source artifact set is not the exact required local set"
        )
    exact_artifacts = {
        name: _artifact_core(artifacts[name]) for name in sorted(artifacts)
    }
    packages: dict[str, str] = {}
    for distribution in ("h5py", "numpy", "torch", "pimsr-inversion"):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise DenseNet2DAdapterError(
                f"required dependency distribution is missing: {distribution}"
            ) from exc
    cli_entrypoint_source_included = "runner_source" in exact_artifacts
    if cli_entrypoint_source_included:
        expected_runner = _validated_runner_source_artifact(
            _benchmark_runner_source_path()
        )
        if exact_artifacts["runner_source"] != expected_runner:
            raise DenseNet2DAdapterError(
                "dependency closure runner source is not the exact benchmark CLI"
            )
    body: dict[str, object] = {
        "schema": _DEPENDENCY_CLOSURE_SCHEMA,
        "schema_version": _DEPENDENCY_CLOSURE_VERSION,
        "evidence_scope": (
            "direct_python_source_artifacts_and_distribution_version_strings"
        ),
        "python": platform.python_version(),
        "packages": packages,
        "local_source_artifacts": exact_artifacts,
        "cli_entrypoint_source_included": cli_entrypoint_source_included,
        "required_local_python_source_artifacts_recorded": (
            cli_entrypoint_source_included
        ),
        "native_binary_environment_complete": False,
    }
    return {**body, "closure_sha256": _json_sha256(body)}


def _module_source_artifact(module_name: str) -> dict[str, object]:
    module = importlib.import_module(module_name)
    source = getattr(module, "__file__", None)
    if not isinstance(source, str):
        raise DenseNet2DAdapterError(
            f"cannot identify source for dependency {module_name}"
        )
    path = Path(source)
    if path.suffix == ".pyc":
        path = path.with_suffix(".py")
    return _snapshot_core(
        _snapshot_regular_file(path, role=f"dependency source {module_name}")
    )


def _benchmark_runner_source_path() -> Path:
    return Path(__file__).resolve().parents[2] / "scripts" / "run_densenet2d_common.py"


def _validated_runner_source_artifact(path: str | Path) -> dict[str, object]:
    snapshot = _snapshot_regular_file(path, role="DenseNet CLI runner source")
    try:
        expected = _benchmark_runner_source_path().resolve(strict=True)
    except OSError as exc:
        raise DenseNet2DAdapterError(
            "cannot identify benchmark scripts/run_densenet2d_common.py"
        ) from exc
    if snapshot.path != expected:
        raise DenseNet2DAdapterError(
            "runner_source must be the exact benchmark scripts/run_densenet2d_common.py"
        )
    expected_info = expected.stat(follow_symlinks=False)
    if (snapshot.device, snapshot.inode) != (
        int(expected_info.st_dev),
        int(expected_info.st_ino),
    ):
        raise DenseNet2DAdapterError(
            "runner_source identity changed during dependency closure capture"
        )
    return _snapshot_core(snapshot)


def _source_dependency_artifacts(
    repository: Mapping[str, object],
    *,
    runner_source: str | Path | None = None,
) -> dict[str, Mapping[str, object]]:
    architecture = repository.get("architecture_source")
    if not isinstance(architecture, Mapping):
        raise DenseNet2DAdapterError("repository architecture provenance is missing")
    result: dict[str, Mapping[str, object]] = {
        "adapter_source": _snapshot_core(
            _snapshot_regular_file(__file__, role="DenseNet adapter source")
        ),
        "artifact_guard_source": _snapshot_core(
            _snapshot_regular_file(
                _artifact_guards.__file__, role="artifact guard source"
            )
        ),
        "architecture_source": architecture,
        "inversion_dataset_contract_source": _module_source_artifact(
            "pimsr_inversion.contracts2d"
        ),
        "materializer_contract_source": _snapshot_core(
            _snapshot_regular_file(
                _materializer_contracts.__file__, role="materializer contract source"
            )
        ),
        "shared_contract_loader_source": _snapshot_core(
            _snapshot_regular_file(
                _shared_contracts.__file__, role="shared contract loader source"
            )
        ),
    }
    if runner_source is not None:
        result["runner_source"] = _validated_runner_source_artifact(runner_source)
    return result


def _validate_dataset_identity(value: object, role: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise DenseNet2DAdapterError(f"{role} must be a mapping")
    _require_exact_keys(value, _DATASET_IDENTITY_KEYS, role)
    if not isinstance(value["path"], str) or not value["path"]:
        raise DenseNet2DAdapterError(f"{role}.path must be non-empty")
    _require_sha256(value["sha256"], f"{role}.sha256")
    _require_sha256(value["sample_index_sha256"], f"{role}.sample_index_sha256")
    for name in ("size_bytes", "sample_count"):
        item = value[name]
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            raise DenseNet2DAdapterError(f"{role}.{name} must be a positive integer")
    return value


def _validate_data_geometry(value: object) -> dict[str, np.ndarray]:
    if not isinstance(value, Mapping):
        raise DenseNet2DAdapterError("checkpoint.data_geometry must be a mapping")
    _require_exact_keys(value, _GEOMETRY_KEYS, "checkpoint.data_geometry")
    expected_shapes = {
        "frequency_hz": (INPUT_GRID_SHAPE[0],),
        "station_x_m": (INPUT_GRID_SHAPE[1],),
        "x_cell_centers_m": (OUTPUT_GRID_SHAPE[1],),
        "depth_cell_centers_m": (OUTPUT_GRID_SHAPE[0],),
    }
    result: dict[str, np.ndarray] = {}
    for name, shape in expected_shapes.items():
        if not isinstance(value[name], list):
            raise DenseNet2DAdapterError(
                f"checkpoint.data_geometry.{name} must be a JSON list"
            )
        axis = np.asarray(value[name], dtype=np.float64)
        if axis.shape != shape or not np.isfinite(axis).all():
            raise DenseNet2DAdapterError(
                f"checkpoint.data_geometry.{name} has invalid shape or values"
            )
        if np.any(np.diff(axis) <= 0):
            raise DenseNet2DAdapterError(
                f"checkpoint.data_geometry.{name} must be strictly increasing"
            )
        if name in {"frequency_hz", "depth_cell_centers_m"} and np.any(axis <= 0):
            raise DenseNet2DAdapterError(
                f"checkpoint.data_geometry.{name} must be positive"
            )
        result[name] = axis
    return result


def _validate_training_summary(value: object) -> None:
    if not isinstance(value, Mapping):
        raise DenseNet2DAdapterError("checkpoint.training_summary must be a mapping")
    expected = frozenset({"best_epoch", "best_validation_weighted_mse", "history"})
    _require_exact_keys(value, expected, "checkpoint.training_summary")
    history = value["history"]
    if not isinstance(history, list) or len(history) != EPOCHS:
        raise DenseNet2DAdapterError(
            "checkpoint training history must contain every configured epoch"
        )
    best_epoch = value["best_epoch"]
    if isinstance(best_epoch, bool) or not isinstance(best_epoch, int):
        raise DenseNet2DAdapterError("checkpoint best_epoch must be an integer")
    validation_losses: list[float] = []
    for index, record in enumerate(history, start=1):
        if not isinstance(record, Mapping):
            raise DenseNet2DAdapterError("checkpoint history record must be a mapping")
        _require_exact_keys(
            record,
            frozenset({"epoch", "train_weighted_mse", "validation_weighted_mse"}),
            f"checkpoint.training_summary.history[{index - 1}]",
        )
        if record["epoch"] != index:
            raise DenseNet2DAdapterError("checkpoint history epochs are not sequential")
        for name in ("train_weighted_mse", "validation_weighted_mse"):
            item = record[name]
            if (
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not np.isfinite(float(item))
                or float(item) < 0
            ):
                raise DenseNet2DAdapterError(
                    f"checkpoint history {name} must be finite and non-negative"
                )
        validation_losses.append(float(record["validation_weighted_mse"]))
    selected_epoch = min(range(EPOCHS), key=validation_losses.__getitem__) + 1
    selected_loss = validation_losses[selected_epoch - 1]
    declared_loss = value["best_validation_weighted_mse"]
    if (
        best_epoch != selected_epoch
        or isinstance(declared_loss, bool)
        or not isinstance(declared_loss, (int, float))
        or float(declared_loss) != selected_loss
    ):
        raise DenseNet2DAdapterError(
            "checkpoint validation-selected epoch/loss is inconsistent with history"
        )


def _validate_backend_runtime(value: object, *, phase: str) -> None:
    if not isinstance(value, Mapping):
        raise DenseNet2DAdapterError(f"{phase} runtime must be a mapping")
    expected = {
        "training": _TRAINING_RUNTIME_KEYS,
        "inference": _INFERENCE_RUNTIME_KEYS,
    }.get(phase)
    if expected is None:  # pragma: no cover - internal closed call sites
        raise AssertionError(f"unsupported DenseNet runtime phase: {phase}")
    _require_exact_keys(value, frozenset(expected), f"{phase} runtime")
    for name in ("python", "platform", "numpy", "torch"):
        item = value[name]
        if not isinstance(item, str) or not item:
            raise DenseNet2DAdapterError(
                f"{phase} runtime {name} must be a non-empty string"
            )
    torch_cuda_build = value["torch_cuda_build"]
    if torch_cuda_build is not None and (
        not isinstance(torch_cuda_build, str) or not torch_cuda_build
    ):
        raise DenseNet2DAdapterError(
            f"{phase} runtime torch_cuda_build must be null or a non-empty string"
        )
    if not isinstance(value["cuda_available"], bool):
        raise DenseNet2DAdapterError(f"{phase} runtime cuda_available must be boolean")
    device = value["device"]
    if device not in {"cpu", "cuda"}:
        raise DenseNet2DAdapterError(f"{phase} runtime device must be cpu or cuda")
    cuda_device_name = value["cuda_device_name"]
    if device == "cuda":
        if (
            value["cuda_available"] is not True
            or not isinstance(cuda_device_name, str)
            or not cuda_device_name
        ):
            raise DenseNet2DAdapterError(
                f"{phase} CUDA runtime must identify an available CUDA device"
            )
    elif cuda_device_name is not None:
        raise DenseNet2DAdapterError(
            f"{phase} CPU runtime must not identify a CUDA execution device"
        )
    peak = value["peak_cuda_memory_bytes"]
    if isinstance(peak, bool) or not isinstance(peak, int) or peak < 0:
        raise DenseNet2DAdapterError(
            f"{phase} runtime peak_cuda_memory_bytes must be a non-negative integer"
        )
    if device == "cpu" and peak != 0:
        raise DenseNet2DAdapterError(
            f"{phase} CPU runtime peak_cuda_memory_bytes must be zero"
        )
    time_keys = {
        "preprocessing_wall_time_s",
        "model_initialization_wall_time_s",
        "backend_wall_time_s",
        f"{phase}_wall_time_s",
    }
    for name in time_keys:
        item = value[name]
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not np.isfinite(float(item))
            or float(item) < 0.0
        ):
            raise DenseNet2DAdapterError(
                f"{phase} runtime {name} must be finite and non-negative"
            )
    _canonical_json_bytes(value)
    _require_no_prescore_metadata(value, role=f"{phase} runtime")


def _validate_exact_model_state(
    value: object,
    *,
    candidate: Any,
    torch: Any,
) -> Mapping[str, Any]:
    if type(value) is not dict or not value:
        raise DenseNet2DAdapterError(
            "checkpoint model_state must be a non-empty plain dictionary"
        )
    expected_state = candidate.state_dict()
    if set(value) != set(expected_state):
        raise DenseNet2DAdapterError(
            "checkpoint model_state keys differ from the pinned architecture"
        )
    for name, expected_tensor in expected_state.items():
        tensor = value[name]
        if not isinstance(tensor, torch.Tensor):
            raise DenseNet2DAdapterError(
                f"checkpoint model_state[{name!r}] must be a tensor"
            )
        if (
            tuple(tensor.shape) != tuple(expected_tensor.shape)
            or tensor.dtype != expected_tensor.dtype
            or tensor.layout != expected_tensor.layout
            or tensor.device.type != expected_tensor.device.type
        ):
            raise DenseNet2DAdapterError(
                f"checkpoint model_state[{name!r}] shape/dtype/layout/device is not exact"
            )
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
            torch.isfinite(tensor).all()
        ):
            raise DenseNet2DAdapterError(
                f"checkpoint model_state[{name!r}] is non-finite"
            )
    try:
        candidate.load_state_dict(dict(value), strict=True)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise DenseNet2DAdapterError(
            "checkpoint model_state is incompatible with the pinned architecture"
        ) from exc
    return value


def _validate_dependency_closure(
    value: object,
    expected: Mapping[str, object],
) -> None:
    if not isinstance(value, Mapping):
        raise DenseNet2DAdapterError("checkpoint dependency_closure must be a mapping")
    if dict(value) != dict(expected):
        raise DenseNet2DAdapterError(
            "checkpoint source dependency closure differs from current pinned sources"
        )
    body = {key: value[key] for key in value if key != "closure_sha256"}
    if value.get("closure_sha256") != _json_sha256(body):
        raise DenseNet2DAdapterError("checkpoint dependency closure digest is invalid")


def _validate_checkpoint_state(
    state: object,
    *,
    repository: Mapping[str, object],
    dependency_closure: Mapping[str, object],
    expected_seed: int | None = None,
) -> Mapping[str, Any]:
    import torch

    if type(state) is not dict:
        raise DenseNet2DAdapterError("checkpoint root must be a plain dictionary")
    _require_exact_keys(state, _CHECKPOINT_KEYS, "checkpoint")
    if (
        state["checkpoint_schema"] != CHECKPOINT_SCHEMA
        or state["checkpoint_schema_version"] != CHECKPOINT_SCHEMA_VERSION
    ):
        raise DenseNet2DAdapterError("checkpoint schema/version is unsupported")
    if (
        state["method_id"] != METHOD_ID
        or state["method"] != METHOD_NAME
        or state["track"] != "common-retrain"
    ):
        raise DenseNet2DAdapterError("checkpoint method identity is invalid")
    seed = state["seed"]
    if isinstance(seed, bool) or seed not in COMMON_RETRAIN_SEEDS:
        raise DenseNet2DAdapterError(
            "checkpoint seed is outside the locked campaign seeds"
        )
    if expected_seed is not None and seed != expected_seed:
        raise DenseNet2DAdapterError("checkpoint seed differs from the requested seed")
    model = state["model"]
    if not isinstance(model, Mapping):
        raise DenseNet2DAdapterError("checkpoint.model must be a mapping")
    _require_exact_keys(model, _MODEL_KEYS, "checkpoint.model")
    if dict(model) != _model_contract():
        raise DenseNet2DAdapterError("checkpoint model contract is invalid")
    training_config = state["training_config"]
    if not isinstance(training_config, Mapping) or dict(
        training_config
    ) != _training_config(int(seed)):
        raise DenseNet2DAdapterError("checkpoint training_config is not exact")
    preprocessing = state["preprocessing"]
    if (
        not isinstance(preprocessing, Mapping)
        or dict(preprocessing) != _preprocessing_contract()
    ):
        raise DenseNet2DAdapterError("checkpoint preprocessing contract is not exact")
    if not isinstance(state["source"], Mapping) or dict(state["source"]) != dict(
        repository
    ):
        raise DenseNet2DAdapterError("checkpoint pinned source identity is not exact")
    _validate_dependency_closure(state["dependency_closure"], dependency_closure)
    identities = state["dataset_identities"]
    if not isinstance(identities, Mapping) or set(identities) != {
        "train",
        "validation",
    }:
        raise DenseNet2DAdapterError(
            "checkpoint must identify exactly train and validation artifacts"
        )
    train_identity = _validate_dataset_identity(
        identities["train"], "checkpoint.dataset_identities.train"
    )
    validation_identity = _validate_dataset_identity(
        identities["validation"], "checkpoint.dataset_identities.validation"
    )
    if train_identity["sha256"] == validation_identity["sha256"]:
        raise DenseNet2DAdapterError("checkpoint train and validation identities overlap")
    _validate_data_geometry(state["data_geometry"])
    _validate_training_summary(state["training_summary"])
    _validate_backend_runtime(state["training_runtime"], phase="training")
    checkpoint_metadata = {
        key: value for key, value in state.items() if key != "model_state"
    }
    _require_no_prescore_metadata(checkpoint_metadata, role="checkpoint metadata")
    if state["campaign_observations_accepted_for_training"] is not False:
        raise DenseNet2DAdapterError(
            "checkpoint must declare that campaign observations were not accepted"
        )
    for declaration in (
        "truth_keys_accepted",
        "contains_truth",
        "contains_observation_campaign",
    ):
        if state[declaration] is not False:
            raise DenseNet2DAdapterError(
                f"checkpoint {declaration} declaration must be false"
            )
    candidate = _load_upstream_model(repository, torch=torch)
    _validate_exact_model_state(state["model_state"], candidate=candidate, torch=torch)
    return state


def _safe_load_checkpoint(
    snapshot: _ArtifactSnapshot,
    *,
    repository: Mapping[str, object],
    dependency_closure: Mapping[str, object],
    expected_seed: int | None = None,
) -> Mapping[str, Any]:
    if snapshot.size_bytes > 512 * 1024 * 1024:
        raise DenseNet2DAdapterError("checkpoint exceeds the fail-closed size limit")
    import torch

    try:
        state = torch.load(
            io.BytesIO(snapshot.payload),
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise DenseNet2DAdapterError(
            f"cannot safely decode checkpoint with weights_only=True: {exc}"
        ) from exc
    return _validate_checkpoint_state(
        state,
        repository=repository,
        dependency_closure=dependency_closure,
        expected_seed=expected_seed,
    )


def _prepare_new_outputs(
    outputs: Sequence[tuple[str | Path, str]],
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    paths = tuple(Path(os.path.abspath(os.fspath(path))) for path, _suffix in outputs)
    if len(set(paths)) != len(paths):
        raise ValueError("DenseNet output paths must be distinct")
    for path, (_raw, suffix) in zip(paths, outputs, strict=True):
        if os.path.lexists(path) and _path_is_link(path):
            raise DenseNet2DAdapterError(
                f"MT2DInv-DenseNet output must not be a symbolic link or junction: {path}"
            )
        if path.suffix.lower() != suffix:
            raise ValueError(f"MT2DInv-DenseNet output {path} must use {suffix}")
        path.parent.mkdir(parents=True, exist_ok=True)
    existing = [path for path in paths if os.path.lexists(path)]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing MT2DInv-DenseNet output(s): "
            + ", ".join(str(path) for path in existing)
        )
    parts = tuple(path.with_name(path.name + ".part") for path in paths)
    stale = [path for path in parts if os.path.lexists(path)]
    if stale:
        raise FileExistsError(
            "stale MT2DInv-DenseNet partial output(s) require inspection: "
            + ", ".join(str(path) for path in stale)
        )
    return paths, parts


def train_common_retrain(
    *,
    repository_path: str | Path,
    train_h5: str | Path,
    validation_h5: str | Path,
    seed: int,
    device: str,
    checkpoint_out: str | Path,
    command: Sequence[str] | None = None,
    runner_source: str | Path | None = None,
) -> dict[str, object]:
    """Train one seed once using only train and validation artifacts."""
    if isinstance(seed, bool) or seed not in COMMON_RETRAIN_SEEDS:
        raise ValueError(f"seed must be one of {COMMON_RETRAIN_SEEDS}")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'")
    destinations, parts = _prepare_new_outputs(((checkpoint_out, ".pt"),))
    checkpoint_path = destinations[0]
    checkpoint_part = parts[0]
    input_paths = {
        Path(train_h5).resolve(),
        Path(validation_h5).resolve(),
        Path(repository_path).resolve(),
    }
    if runner_source is not None:
        input_paths.add(Path(runner_source).resolve())
    if checkpoint_path in input_paths:
        raise ValueError("checkpoint output must differ from every training input")

    repository = verify_pinned_repository(repository_path)
    dependency_artifacts = _source_dependency_artifacts(
        repository, runner_source=runner_source
    )
    dependency_closure = _dependency_closure(dependency_artifacts)
    train = load_training_split(train_h5, role="MT2DInv-DenseNet training dataset")
    validation = load_training_split(
        validation_h5, role="MT2DInv-DenseNet validation dataset"
    )
    _require_same_geometry(train, validation, where="train and validation datasets")
    _require_disjoint_training_splits(train, validation)

    outcome = _train_model(
        train,
        validation,
        repository,
        seed=seed,
        device_name=device,
    )
    for role, artifact in {
        **dependency_artifacts,
        "train_dataset": train.provenance,
        "validation_dataset": validation.provenance,
    }.items():
        require_file_artifact_unchanged(_artifact_core(artifact), role=role)
    if verify_pinned_repository(repository_path) != repository:
        raise DenseNet2DAdapterError(
            "pinned MT2DInv-DenseNet repository changed during training"
        )

    checkpoint: dict[str, object] = {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "method": METHOD_NAME,
        "track": "common-retrain",
        "seed": seed,
        "model": _model_contract(),
        "model_state": dict(outcome.state_dict),
        "training_config": _training_config(seed),
        "preprocessing": _preprocessing_contract(),
        "training_summary": dict(outcome.training_summary),
        "training_runtime": dict(outcome.runtime),
        "source": dict(repository),
        "dependency_closure": dependency_closure,
        "dataset_identities": {
            "train": _dataset_identity(train),
            "validation": _dataset_identity(validation),
        },
        "data_geometry": _data_geometry(train),
        "campaign_observations_accepted_for_training": False,
        "truth_keys_accepted": False,
        "contains_truth": False,
        "contains_observation_campaign": False,
    }
    _validate_checkpoint_state(
        checkpoint,
        repository=repository,
        dependency_closure=dependency_closure,
        expected_seed=seed,
    )
    try:
        _write_bytes_new(checkpoint_part, _checkpoint_bytes(checkpoint))
        staged = _snapshot_regular_file(checkpoint_part, role="staged checkpoint")
        _safe_load_checkpoint(
            staged,
            repository=repository,
            dependency_closure=dependency_closure,
            expected_seed=seed,
        )
        for role, artifact in {
            **dependency_artifacts,
            "train_dataset": train.provenance,
            "validation_dataset": validation.provenance,
        }.items():
            require_file_artifact_unchanged(_artifact_core(artifact), role=role)
        if verify_pinned_repository(repository_path) != repository:
            raise DenseNet2DAdapterError(
                "pinned MT2DInv-DenseNet repository changed before publication"
            )
        _publish_parts(parts, destinations, expected_snapshots=(staged,))
    finally:
        checkpoint_part.unlink(missing_ok=True)

    checkpoint_identity = {
        **_snapshot_core(staged),
        "path": str(checkpoint_path),
    }
    return {
        "schema": "pimsr-mt2dinv-densenet-common-retrain-training-result",
        "schema_version": 1,
        "method_id": METHOD_ID,
        "method": METHOD_NAME,
        "seed": seed,
        "checkpoint": checkpoint_identity,
        "dependency_closure_sha256": dependency_closure["closure_sha256"],
        "train_sha256": train.provenance["sha256"],
        "validation_sha256": validation.provenance["sha256"],
        "command": list(command) if command is not None else None,
    }


def _require_checkpoint_geometry_matches_observations(
    checkpoint: Mapping[str, Any],
    observations: HeldoutObservations,
) -> None:
    axes = _validate_data_geometry(checkpoint["data_geometry"])
    expected = {
        "frequency_hz": observations.frequencies,
        "station_x_m": observations.station_x,
        "x_cell_centers_m": observations.x_grid,
        "depth_cell_centers_m": observations.depth_grid,
    }
    for name, values in expected.items():
        if not np.array_equal(axes[name], np.asarray(values, dtype=np.float64)):
            raise DenseNet2DAdapterError(
                f"checkpoint and truth-free observations have different {name}"
            )


def run_common_inference(
    *,
    repository_path: str | Path,
    checkpoint_path: str | Path,
    observations_npz: str | Path,
    device: str,
    predictions_out: str | Path,
    runtime_out: str | Path,
    expected_checkpoint_sha256: str | None = None,
    expected_observations_sha256: str | None = None,
    command: Sequence[str] | None = None,
    runner_source: str | Path | None = None,
) -> dict[str, object]:
    """Infer from one reusable checkpoint and one truth-free campaign payload."""
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'")
    destinations, parts = _prepare_new_outputs(
        ((predictions_out, ".npz"), (runtime_out, ".json"))
    )
    prediction_path, _runtime_path = destinations
    prediction_part, runtime_part = parts
    input_paths = {
        Path(checkpoint_path).resolve(),
        Path(observations_npz).resolve(),
        Path(repository_path).resolve(),
    }
    if runner_source is not None:
        input_paths.add(Path(runner_source).resolve())
    if any(path in input_paths for path in destinations):
        raise ValueError("inference outputs must differ from every inference input")

    started_at = datetime.now(UTC)
    wall_start = time.perf_counter()
    repository = verify_pinned_repository(repository_path)
    dependency_artifacts = _source_dependency_artifacts(
        repository, runner_source=runner_source
    )
    dependency_closure = _dependency_closure(dependency_artifacts)
    checkpoint_snapshot = _snapshot_regular_file(
        checkpoint_path,
        role="checkpoint",
        expected_sha256=expected_checkpoint_sha256,
    )
    checkpoint = _safe_load_checkpoint(
        checkpoint_snapshot,
        repository=repository,
        dependency_closure=dependency_closure,
    )
    observations_snapshot = _snapshot_regular_file(
        observations_npz,
        role="truth-free observations",
        expected_sha256=expected_observations_sha256,
    )
    test = _load_heldout_observations_snapshot(observations_snapshot)
    if (
        test.provenance["sha256"] != observations_snapshot.sha256
        or test.provenance["size_bytes"] != observations_snapshot.size_bytes
    ):
        raise DenseNet2DAdapterError(
            "truth-free observation loader returned a different artifact identity"
        )
    _require_checkpoint_geometry_matches_observations(checkpoint, test)
    training_identities = checkpoint["dataset_identities"]
    assert isinstance(training_identities, Mapping)
    if observations_snapshot.sha256 in {
        training_identities["train"]["sha256"],
        training_identities["validation"]["sha256"],
    }:
        raise DenseNet2DAdapterError(
            "campaign observation artifact must differ from train and validation"
        )

    seed = int(checkpoint["seed"])
    predictions, inference_runtime = _predict_from_checkpoint(
        test,
        repository,
        checkpoint["model_state"],
        seed=seed,
        device_name=device,
    )
    _validate_backend_runtime(inference_runtime, phase="inference")
    predictions = np.asarray(predictions, dtype="<f4")
    expected_shape = (test.sample_index.size, *OUTPUT_GRID_SHAPE)
    if predictions.shape != expected_shape or not np.isfinite(predictions).all():
        raise DenseNet2DAdapterError(
            f"backend predictions must be finite with shape {expected_shape}"
        )

    for role, artifact in dependency_artifacts.items():
        require_file_artifact_unchanged(_artifact_core(artifact), role=role)
    require_file_artifact_unchanged(
        _snapshot_core(checkpoint_snapshot), role="checkpoint"
    )
    require_file_artifact_unchanged(
        _snapshot_core(observations_snapshot), role="truth-free observations"
    )
    if verify_pinned_repository(repository_path) != repository:
        raise DenseNet2DAdapterError(
            "pinned MT2DInv-DenseNet repository changed during inference"
        )

    try:
        _write_bytes_new(
            prediction_part,
            _prediction_npz_bytes(
                observations_snapshot.sha256,
                test.sample_index,
                test.x_grid,
                test.depth_grid,
                predictions,
            ),
        )
        prediction_staged = _snapshot_regular_file(
            prediction_part, role="staged predictions"
        )
        checkpoint_identity = _snapshot_core(checkpoint_snapshot)
        prediction_identity = {
            **_snapshot_core(prediction_staged),
            "path": str(prediction_path),
        }
        train_identity = dict(training_identities["train"])
        validation_identity = dict(training_identities["validation"])
        architecture = dependency_artifacts["architecture_source"]
        adapter = dependency_artifacts["adapter_source"]
        shared_loader = dependency_artifacts["shared_contract_loader_source"]
        bindings: dict[str, object] = {
            "training_seed": seed,
            "source_commit": repository["commit"],
            "source_clean_worktree": repository["clean_worktree"],
            "upstream_source_sha256": architecture["sha256"],
            "adapter_source_sha256": adapter["sha256"],
            "shared_contract_loader_source_sha256": shared_loader["sha256"],
            "dependency_closure_sha256": dependency_closure["closure_sha256"],
            "train_sha256": train_identity["sha256"],
            "validation_sha256": validation_identity["sha256"],
            "observations_sha256": observations_snapshot.sha256,
            "checkpoint_sha256": checkpoint_snapshot.sha256,
            "prediction_sha256": prediction_staged.sha256,
        }
        if "runner_source" in dependency_artifacts:
            bindings["runner_source_sha256"] = dependency_artifacts["runner_source"][
                "sha256"
            ]
        source_artifacts = {
            name: dict(value) for name, value in dependency_artifacts.items()
        }
        source_artifacts.update(
            {
                "train_dataset": train_identity,
                "validation_dataset": validation_identity,
                "heldout_observations": _snapshot_core(observations_snapshot),
            }
        )
        finished_at = datetime.now(UTC)
        runtime = {
            "schema": RUNTIME_SCHEMA,
            "schema_version": RUNTIME_SCHEMA_VERSION,
            "method_id": METHOD_ID,
            "method": METHOD_NAME,
            "track": "common-retrain",
            "operation": "inference_from_reusable_checkpoint",
            "comparison_status": "unscored_prediction_artifact",
            "ranking_allowed": False,
            "seed": seed,
            "training_seed": seed,
            "bindings": bindings,
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "adapter_wall_time_s": time.perf_counter() - wall_start,
            "command": list(command) if command is not None else None,
            "working_directory": str(Path.cwd().resolve()),
            "repository": dict(repository),
            "source_artifacts": source_artifacts,
            "dependency_closure": dict(dependency_closure),
            "model": dict(checkpoint["model"]),
            "training_config": dict(checkpoint["training_config"]),
            "preprocessing": dict(checkpoint["preprocessing"]),
            "dataset_identities": {
                "train": train_identity,
                "validation": validation_identity,
            },
            "training_summary": dict(checkpoint["training_summary"]),
            "training_runtime": dict(checkpoint["training_runtime"]),
            "runtime": dict(inference_runtime),
            "checkpoint_contract": {
                "schema": CHECKPOINT_SCHEMA,
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "safe_load": "torch.load(weights_only=True)",
                "seed": seed,
                "campaign_observations_accepted_for_training": False,
                "truth_keys_accepted": False,
                "contains_truth": False,
                "contains_observation_campaign": False,
                "dataset_identities": {
                    "train": train_identity,
                    "validation": validation_identity,
                },
                "checkpoint_sha256": checkpoint_snapshot.sha256,
            },
            "observation_contract": {
                "schema": OBSERVATION_SCHEMA,
                "schema_version": PAYLOAD_SCHEMA_VERSION,
                "truth_keys_accepted": False,
                "observations_sha256": observations_snapshot.sha256,
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
                "observations_sha256": observations_snapshot.sha256,
                "observations_sha256_dtype": "<U64",
                "sample_index_dtype": "<i8",
                "x_cell_centers_dtype": "<f8",
                "depth_cell_centers_dtype": "<f8",
                "prediction_dtype": "<f4",
                "prediction_shape": list(predictions.shape),
                "prediction_axis_order": ["sample", "depth", "x"],
                "prediction_unit": "log10_ohm_m",
                "contains_truth": False,
                "prediction_sha256": prediction_staged.sha256,
            },
            "outputs": {
                "checkpoint": checkpoint_identity,
                "predictions": prediction_identity,
            },
            "truth_keys_accepted": False,
            "contains_truth": False,
            "heldout_truth_available_to_adapter": False,
        }
        _require_no_prescore_metadata(runtime, role="DenseNet inference runtime")
        for role, artifact in dependency_artifacts.items():
            require_file_artifact_unchanged(_artifact_core(artifact), role=role)
        require_file_artifact_unchanged(
            _snapshot_core(checkpoint_snapshot), role="checkpoint"
        )
        require_file_artifact_unchanged(
            _snapshot_core(observations_snapshot), role="truth-free observations"
        )
        require_file_artifact_unchanged(
            _snapshot_core(prediction_staged), role="staged predictions"
        )
        if verify_pinned_repository(repository_path) != repository:
            raise DenseNet2DAdapterError(
                "pinned MT2DInv-DenseNet repository changed before publication"
            )
        runtime_payload = _canonical_json_bytes(runtime)
        _write_bytes_new(runtime_part, runtime_payload)
        runtime_staged = _snapshot_regular_file(runtime_part, role="staged runtime")
        if runtime_staged.payload != runtime_payload:
            raise DenseNet2DAdapterError("staged runtime changed after writing")
        _publish_parts(
            parts,
            destinations,
            expected_snapshots=(prediction_staged, runtime_staged),
        )
        return runtime
    finally:
        for part in parts:
            part.unlink(missing_ok=True)


__all__ = [
    "COMMON_RETRAIN_SEEDS",
    "DenseNet2DAdapterError",
    "HeldoutObservations",
    "TrainingOutcome",
    "TrainingSplit",
    "load_heldout_observations",
    "load_training_split",
    "preprocess_observations",
    "resize_bilinear_half_pixel",
    "resize_log10_resistivity",
    "run_common_inference",
    "train_common_retrain",
    "verify_pinned_repository",
]
