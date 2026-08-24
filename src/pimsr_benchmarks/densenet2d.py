"""Fail-closed common-retraining adapter for MT2DInv-DenseNet v1.2.

Only six reviewed architecture class definitions are compiled from the pinned
upstream source.  In particular, the upstream module's dataset reads, training
loop, logging, and other top-level side effects are never imported or executed.
Held-out data enter this adapter only through the truth-free observation NPZ.
"""

from __future__ import annotations

import ast
import hashlib
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
from pathlib import Path
from typing import Any

import numpy as np

from pimsr_benchmarks import mtdlpy as _shared_contracts
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
from pimsr_benchmarks.mtdlpy import (
    load_heldout_observations as _load_heldout_observations,
)
from pimsr_benchmarks.mtdlpy import (
    load_training_split as _load_training_split,
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
RUNTIME_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA = "pimsr-mt2dinv-densenet-common-retrain-checkpoint"
CHECKPOINT_SCHEMA_VERSION = 1


class DenseNet2DAdapterError(RuntimeError):
    """Raised when an MT2DInv-DenseNet run cannot prove its contract."""


@dataclass(frozen=True)
class TrainingOutcome:
    """Small internal value object kept free of Torch-specific types."""

    state_dict: Mapping[str, Any]
    predicted_log10_resistivity: np.ndarray
    training_summary: Mapping[str, object]
    runtime: Mapping[str, object]


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
    """Reuse the exact schema-v2 loader while presenting method-local errors."""
    try:
        return _load_training_split(path, role=role)
    except MTDLPyAdapterError as exc:
        raise DenseNet2DAdapterError(str(exc).replace("MTDLPy", METHOD_NAME)) from exc


def load_heldout_observations(path: str | Path) -> HeldoutObservations:
    """Load the exact canonical truth-free observation NPZ v1."""
    try:
        return _load_heldout_observations(path)
    except MTDLPyAdapterError as exc:
        raise DenseNet2DAdapterError(str(exc).replace("MTDLPy", METHOD_NAME)) from exc


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


def _require_disjoint_samples(
    train: TrainingSplit,
    validation: TrainingSplit,
    test: HeldoutObservations,
) -> None:
    if (
        train.generator_seed == validation.generator_seed
        and np.intersect1d(train.sample_index, validation.sample_index).size
    ):
        raise DenseNet2DAdapterError(
            "train and validation (generator_seed, sample_index) identities overlap"
        )
    digests = [entry.provenance["sha256"] for entry in (train, validation, test)]
    if len(set(digests)) != len(digests):
        raise DenseNet2DAdapterError(
            "train, validation, and held-out artifacts must differ"
        )


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


def _train_and_predict(
    train: TrainingSplit,
    validation: TrainingSplit,
    test: HeldoutObservations,
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
    observations_test = preprocess_observations(
        test.observations, test.frequencies, test.station_x
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

    inference_start = time.perf_counter()
    native_batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, observations_test.shape[0], BATCH_SIZE):
            batch = torch.from_numpy(observations_test[start : start + BATCH_SIZE]).to(
                device
            )
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
    return TrainingOutcome(
        state_dict=best_state,
        predicted_log10_resistivity=predictions,
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


def _publish_parts(parts: Sequence[Path], destinations: Sequence[Path]) -> None:
    """Publish with no overwrite and identity-aware rollback on BaseException."""
    published: list[tuple[Path, tuple[int, int]]] = []
    try:
        for part, destination in zip(parts, destinations, strict=True):
            part_info = part.stat(follow_symlinks=False)
            if part.is_symlink() or not stat.S_ISREG(part_info.st_mode):
                raise DenseNet2DAdapterError(
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
                raise DenseNet2DAdapterError(
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
            raise DenseNet2DAdapterError(
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
        raise ValueError("checkpoint, prediction, and runtime outputs must be distinct")
    for path, suffix in zip(paths, (".pt", ".npz", ".json"), strict=True):
        if path.suffix.lower() != suffix:
            raise ValueError(f"MT2DInv-DenseNet output {path} must use {suffix}")
        path.parent.mkdir(parents=True, exist_ok=True)
    existing = [path for path in paths if path.exists() or path.is_symlink()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing MT2DInv-DenseNet output(s): "
            + ", ".join(str(path) for path in existing)
        )
    parts = tuple(path.with_name(path.name + ".part") for path in paths)
    stale = [path for path in parts if path.exists() or path.is_symlink()]
    if stale:
        raise FileExistsError(
            "stale MT2DInv-DenseNet partial output(s) require inspection: "
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
    """Run one fixed common-retraining seed and immutably publish artifacts."""
    if isinstance(seed, bool) or seed not in COMMON_RETRAIN_SEEDS:
        raise ValueError(f"seed must be one of {COMMON_RETRAIN_SEEDS}")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'")
    destinations, parts = _output_paths(checkpoint_out, predictions_out, runtime_out)
    checkpoint_path, prediction_path, _runtime_path = destinations
    checkpoint_part, prediction_part, runtime_part = parts

    started_at = datetime.now(UTC)
    wall_start = time.perf_counter()
    adapter_source = file_artifact_provenance(__file__)
    shared_contract_loader_source = file_artifact_provenance(_shared_contracts.__file__)
    runner_identity = (
        file_artifact_provenance(runner_source) if runner_source is not None else None
    )
    repository = verify_pinned_repository(repository_path)
    train = load_training_split(train_h5, role="MT2DInv-DenseNet training dataset")
    validation = load_training_split(
        validation_h5, role="MT2DInv-DenseNet validation dataset"
    )
    test = load_heldout_observations(observations_npz)
    _require_same_geometry(train, validation, where="train and validation datasets")
    _require_same_geometry(train, test, where="train and held-out observations")
    _require_disjoint_samples(train, validation, test)

    architecture_source = repository.get("architecture_source")
    if not isinstance(architecture_source, Mapping):
        raise DenseNet2DAdapterError("repository architecture provenance is missing")
    source_artifacts: dict[str, Mapping[str, object]] = {
        "train_dataset": train.provenance,
        "validation_dataset": validation.provenance,
        "heldout_observations": test.provenance,
        "architecture_source": architecture_source,
        "adapter_source": adapter_source,
        "shared_contract_loader_source": shared_contract_loader_source,
    }
    if runner_identity is not None:
        source_artifacts["runner_source"] = runner_identity

    outcome = _train_and_predict(
        train,
        validation,
        test,
        repository,
        seed=seed,
        device_name=device,
    )
    predictions = np.asarray(outcome.predicted_log10_resistivity, dtype="<f4")
    expected_prediction_shape = (test.sample_index.size, *OUTPUT_GRID_SHAPE)
    if (
        predictions.shape != expected_prediction_shape
        or not np.isfinite(predictions).all()
    ):
        raise DenseNet2DAdapterError(
            f"backend predictions must be finite with shape {expected_prediction_shape}"
        )

    for role, artifact in source_artifacts.items():
        require_file_artifact_unchanged(_artifact_core(artifact), role=role)
    if verify_pinned_repository(repository_path) != repository:
        raise DenseNet2DAdapterError(
            "pinned MT2DInv-DenseNet repository changed during the run"
        )

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
    preprocessing = {
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
    dataset_identities = {
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
    }
    checkpoint = {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "method": METHOD_NAME,
        "track": "common-retrain",
        "seed": seed,
        "model": {
            "class": "DenseNetWithICBAM",
            "blocks": list(MODEL_BLOCKS),
            "growth_rate": MODEL_GROWTH_RATE,
            "output_features": MODEL_OUTPUT_FEATURES,
            "parameter_count": MODEL_PARAMETER_COUNT,
        },
        "model_state": outcome.state_dict,
        "training_config": training_config,
        "preprocessing": preprocessing,
        "training_summary": dict(outcome.training_summary),
        "source": repository,
        "dataset_identities": dataset_identities,
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
        checkpoint_identity = {**checkpoint_staged, "path": str(checkpoint_path)}
        prediction_identity = {**prediction_staged, "path": str(prediction_path)}

        finished_at = datetime.now(UTC)
        bindings = {
            "training_seed": seed,
            "source_commit": repository["commit"],
            "source_clean_worktree": repository["clean_worktree"],
            "upstream_source_sha256": architecture_source["sha256"],
            "adapter_source_sha256": source_artifacts["adapter_source"]["sha256"],
            "shared_contract_loader_source_sha256": source_artifacts[
                "shared_contract_loader_source"
            ]["sha256"],
            "train_sha256": train.provenance["sha256"],
            "validation_sha256": validation.provenance["sha256"],
            "observations_sha256": test.provenance["sha256"],
            "checkpoint_sha256": checkpoint_identity["sha256"],
            "prediction_sha256": prediction_identity["sha256"],
        }
        runtime = {
            "schema": RUNTIME_SCHEMA,
            "schema_version": RUNTIME_SCHEMA_VERSION,
            "method_id": METHOD_ID,
            "method": METHOD_NAME,
            "track": "common-retrain",
            "comparison_status": "unscored_prediction_artifact",
            "ranking_allowed": False,
            "seed": seed,
            "bindings": bindings,
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "adapter_wall_time_s": time.perf_counter() - wall_start,
            "command": list(command) if command is not None else None,
            "working_directory": str(Path.cwd().resolve()),
            "repository": repository,
            "source_artifacts": {
                key: dict(value) for key, value in source_artifacts.items()
            },
            "model": checkpoint["model"],
            "training_config": training_config,
            "preprocessing": preprocessing,
            "dataset_identities": dataset_identities,
            "determinism": {
                "cublas_workspace_config": ":4096:8",
                "python_random_seed": seed,
                "numpy_legacy_global_seed": seed,
                "torch_manual_seed": seed,
                "torch_cuda_manual_seed_all": seed,
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
                "observations_sha256": str(test.provenance["sha256"]),
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
                "prediction_sha256": prediction_identity["sha256"],
            },
            "outputs": {
                "checkpoint": checkpoint_identity,
                "predictions": prediction_identity,
            },
        }
        for role, artifact in source_artifacts.items():
            require_file_artifact_unchanged(_artifact_core(artifact), role=role)
        require_file_artifact_unchanged(checkpoint_staged, role="staged checkpoint")
        require_file_artifact_unchanged(prediction_staged, role="staged predictions")
        if verify_pinned_repository(repository_path) != repository:
            raise DenseNet2DAdapterError(
                "pinned MT2DInv-DenseNet repository changed before publication"
            )
        runtime_payload = _canonical_json_bytes(runtime)
        _write_bytes_new(runtime_part, runtime_payload)
        if runtime_part.read_bytes() != runtime_payload:
            raise DenseNet2DAdapterError("staged runtime metadata changed after writing")
        _publish_parts(parts, destinations)
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
    "run_common_retrain",
    "verify_pinned_repository",
]
