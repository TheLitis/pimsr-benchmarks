"""Fail-closed PIMSR 2-D inference for the observation-only SOTA payload.

The adapter is deliberately narrower than the training data loader.  It accepts
only the public observation payload emitted by ``dataset2d_materialization``;
clean responses, geological labels and truth arrays are rejected as unknown
members.  A versioned training checkpoint supplies both the model geometry and
the training-only normalization statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import io
import json
import os
import platform
import stat
import subprocess
import time
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pimsr_inversion.network2d import PimsrNet2D

OBSERVATION_SCHEMA = "pimsr-sota-2d-observations"
PREDICTION_SCHEMA = "pimsr-sota-2d-predictions"
RUNTIME_SCHEMA = "pimsr-sota-2d-pimsr-runtime"
OBSERVATION_SCHEMA_VERSION = 1
PREDICTION_SCHEMA_VERSION = 2
RUNTIME_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA = "pimsr-train-2d"
CHECKPOINT_SCHEMA_VERSION = 1

OBSERVATION_CHANNEL_ORDER = (
    "log10_rho_te",
    "phase_te_degrees",
    "log10_rho_tm",
    "phase_tm_degrees",
)

_OBSERVATION_KEYS = frozenset(
    {
        "schema",
        "schema_version",
        "sample_index",
        "frequency_hz",
        "station_x_m",
        "x_cell_centers_m",
        "depth_cell_centers_m",
        "observation_channel_order",
        "observed_log10_rho_te",
        "observed_phase_te_degrees",
        "observed_log10_rho_tm",
        "observed_phase_tm_degrees",
        "declared_evaluation_floor_log10_rho_te",
        "declared_evaluation_floor_phase_te_degrees",
        "declared_evaluation_floor_log10_rho_tm",
        "declared_evaluation_floor_phase_tm_degrees",
        "valid_mask",
    }
)
_PREDICTION_KEYS = (
    "schema",
    "schema_version",
    "observations_sha256",
    "sample_index",
    "x_cell_centers_m",
    "depth_cell_centers_m",
    "predicted_log10_resistivity",
)
_CHECKPOINT_KEYS = frozenset(
    {
        "checkpoint_schema",
        "checkpoint_schema_version",
        "n_freq",
        "n_stations",
        "n_depth",
        "n_x",
        "n_scenarios",
        "width",
        "in_channels",
        "scen_head",
        "beta",
        "data_contract",
        "dataset_identities",
        "normalization_sha256",
        "model_config",
        "training_config",
        "epoch",
        "best_epoch",
        "best_val_loss",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "history",
        "rng_state",
        "stats_mean",
        "stats_std",
    }
)
_DATA_CONTRACT_KEYS = frozenset(
    {
        "schema",
        "schema_version",
        "mode_order",
        "impedance_components",
        "scenario_order",
        "phase_convention",
        "resistivity_representation",
        "frequencies_unit",
        "station_x_unit",
        "x_grid_unit",
        "depth_grid_unit",
        "phase_unit",
        "frequencies",
        "station_x",
        "x_grid",
        "depth_grid",
    }
)
_MODEL_CONFIG_KEYS = frozenset(
    {
        "architecture",
        "n_freq",
        "n_stations",
        "n_depth",
        "n_x",
        "in_channels",
        "width",
        "n_scenarios",
        "scen_head",
    }
)
_TRAINING_CONFIG_KEYS = frozenset(
    {
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "gradient_clip_norm",
        "sigma_warmup",
        "sigma_regularization",
        "beta_nll",
        "seed",
        "workers",
        "class_weights",
        "optimizer",
        "scheduler",
        "scheduler_t_max",
        "loss",
        "validation_loss",
        "normalization",
        "runtime",
    }
)
_DATASET_IDENTITY_KEYS = frozenset(
    {
        "identity_schema",
        "identity_schema_version",
        "artifact_sha256",
        "artifact_size_bytes",
        "contract_sha256",
        "provenance",
        "provenance_sha256",
        "identity_sha256",
    }
)
_SHA256_LENGTH = 64
_GIT_COMMIT_LENGTH = 40


class Pimsr2DValidationError(ValueError):
    """Raised when an input, checkpoint or source tree is not exact."""


class Pimsr2DPublicationError(RuntimeError):
    """Raised when immutable output publication cannot be completed."""


@dataclass(frozen=True)
class _ArtifactSnapshot:
    path: Path
    payload: bytes
    sha256: str

    @property
    def size_bytes(self) -> int:
        return len(self.payload)


@dataclass(frozen=True)
class Observations2D:
    """Validated public observations with no target-bearing fields."""

    sample_index: np.ndarray
    frequency_hz: np.ndarray
    station_x_m: np.ndarray
    x_cell_centers_m: np.ndarray
    depth_cell_centers_m: np.ndarray
    log10_rho_te: np.ndarray
    phase_te_degrees: np.ndarray
    log10_rho_tm: np.ndarray
    phase_tm_degrees: np.ndarray
    artifact_sha256: str
    artifact_size_bytes: int


@dataclass(frozen=True)
class ValidatedCheckpoint2D:
    state: Mapping[str, Any]
    model: PimsrNet2D
    mean: np.ndarray
    std: np.ndarray
    artifact_sha256: str
    artifact_size_bytes: int


@dataclass(frozen=True)
class Pimsr2DInferenceResult:
    prediction_path: Path
    runtime_path: Path
    observation_sha256: str
    checkpoint_sha256: str
    prediction_sha256: str
    runtime_sha256: str


def _canonical_json_bytes(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise Pimsr2DValidationError(f"value is not canonical JSON data: {exc}") from exc
    return (encoded + "\n").encode("utf-8")


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value).rstrip(b"\n")).hexdigest()


def _valid_sha256(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Pimsr2DValidationError(
            f"{name} must be 64 lowercase hexadecimal characters"
        )
    return value


def _stat_signature(path: Path) -> tuple[int, int, int, int]:
    value = path.stat()
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
    )


def _snapshot(
    path: str | Path,
    *,
    expected_sha256: str | None,
    role: str,
) -> _ArtifactSnapshot:
    requested = Path(path)
    expected = _valid_sha256(expected_sha256, f"expected_{role}_sha256")
    if requested.is_symlink():
        raise Pimsr2DValidationError(f"{role} artifact must not be a symbolic link")
    try:
        resolved = requested.resolve(strict=True)
        before = _stat_signature(resolved)
    except OSError as exc:
        raise Pimsr2DValidationError(f"cannot stat {role} artifact {requested}: {exc}") from exc
    if not stat.S_ISREG(resolved.stat().st_mode):
        raise Pimsr2DValidationError(f"{role} artifact must be a regular file")
    try:
        payload = resolved.read_bytes()
        after = _stat_signature(resolved)
    except OSError as exc:
        raise Pimsr2DValidationError(f"cannot read {role} artifact {resolved}: {exc}") from exc
    if before != after or len(payload) != after[2]:
        raise Pimsr2DValidationError(f"{role} artifact changed while it was read")
    digest = hashlib.sha256(payload).hexdigest()
    if expected is not None and digest != expected:
        raise Pimsr2DValidationError(
            f"{role} artifact SHA-256 does not match the pinned digest"
        )
    return _ArtifactSnapshot(resolved, payload, digest)


def _require_exact_keys(value: Mapping[str, Any], expected: set[str] | frozenset[str], name: str) -> None:
    actual = set(value)
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        extra = sorted(actual - set(expected))
        raise Pimsr2DValidationError(
            f"{name} keys mismatch; missing={missing}, extra={extra}"
        )


def _npz_arrays(
    snapshot: _ArtifactSnapshot,
    *,
    keys: set[str] | frozenset[str],
    role: str,
) -> dict[str, np.ndarray]:
    expected_members = {f"{key}.npy" for key in keys}
    try:
        with zipfile.ZipFile(io.BytesIO(snapshot.payload), "r") as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if len(names) != len(set(names)):
                raise Pimsr2DValidationError(f"{role} NPZ contains duplicate members")
            if set(names) != expected_members:
                missing = sorted(expected_members - set(names))
                extra = sorted(set(names) - expected_members)
                raise Pimsr2DValidationError(
                    f"{role} NPZ members mismatch; missing={missing}, extra={extra}"
                )
            if any(member.flag_bits & 0x1 for member in members):
                raise Pimsr2DValidationError(f"{role} NPZ contains encrypted members")
            if any(member.compress_type != zipfile.ZIP_STORED for member in members):
                raise Pimsr2DValidationError(
                    f"{role} NPZ must use the materializer's uncompressed member contract"
                )
            bad_member = archive.testzip()
            if bad_member is not None:
                raise Pimsr2DValidationError(
                    f"{role} NPZ has a corrupt member {bad_member!r}"
                )
        with np.load(io.BytesIO(snapshot.payload), allow_pickle=False) as archive:
            return {name: archive[name] for name in sorted(keys)}
    except Pimsr2DValidationError:
        raise
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
        raise Pimsr2DValidationError(f"cannot decode {role} NPZ: {exc}") from exc


def _require_schema(array: np.ndarray, expected: str, role: str) -> None:
    if array.ndim != 0 or array.dtype.kind != "U" or array.item() != expected:
        raise Pimsr2DValidationError(
            f"{role}.schema must be scalar Unicode {expected!r}"
        )


def _require_version(array: np.ndarray, role: str, expected: int) -> None:
    if array.ndim != 0 or array.dtype != np.dtype("<i8") or int(array) != expected:
        raise Pimsr2DValidationError(
            f"{role}.schema_version must be scalar int64 equal to {expected}"
        )


def _array(
    value: np.ndarray,
    *,
    name: str,
    dtype: str | np.dtype[Any],
    ndim: int,
) -> np.ndarray:
    expected = np.dtype(dtype)
    if value.dtype != expected or value.ndim != ndim:
        raise Pimsr2DValidationError(
            f"{name} must have dtype {expected} and rank {ndim}, got {value.dtype} {value.shape}"
        )
    return value


def _axis(value: np.ndarray, *, name: str, positive: bool) -> np.ndarray:
    result = _array(value, name=name, dtype="<f8", ndim=1)
    if result.size == 0 or not np.isfinite(result).all():
        raise Pimsr2DValidationError(f"{name} must be non-empty and finite")
    if result.size > 1 and np.any(np.diff(result) <= 0):
        raise Pimsr2DValidationError(f"{name} must be strictly increasing")
    if positive and np.any(result <= 0):
        raise Pimsr2DValidationError(f"{name} must be strictly positive")
    return result


def load_observations_2d(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> Observations2D:
    """Load the exact observation-only materialization contract."""
    snapshot = _snapshot(
        path,
        expected_sha256=expected_sha256,
        role="observation",
    )
    arrays = _npz_arrays(
        snapshot,
        keys=_OBSERVATION_KEYS,
        role="observation",
    )
    _require_schema(arrays["schema"], OBSERVATION_SCHEMA, "observation")
    _require_version(
        arrays["schema_version"], "observation", OBSERVATION_SCHEMA_VERSION
    )

    sample_index = _array(
        arrays["sample_index"], name="observation.sample_index", dtype="<i8", ndim=1
    )
    if (
        sample_index.size == 0
        or np.any(sample_index < 0)
        or np.unique(sample_index).size != sample_index.size
    ):
        raise Pimsr2DValidationError(
            "observation.sample_index must contain unique non-negative opaque ids"
        )
    frequency = _axis(
        arrays["frequency_hz"], name="observation.frequency_hz", positive=True
    )
    station = _axis(
        arrays["station_x_m"], name="observation.station_x_m", positive=False
    )
    x_centers = _axis(
        arrays["x_cell_centers_m"],
        name="observation.x_cell_centers_m",
        positive=False,
    )
    depth_centers = _axis(
        arrays["depth_cell_centers_m"],
        name="observation.depth_cell_centers_m",
        positive=True,
    )
    if station[0] < x_centers[0] or station[-1] > x_centers[-1]:
        raise Pimsr2DValidationError("observation stations must lie inside the x grid")

    order = arrays["observation_channel_order"]
    if (
        order.ndim != 1
        or order.dtype.kind != "U"
        or tuple(order.tolist()) != OBSERVATION_CHANNEL_ORDER
    ):
        raise Pimsr2DValidationError(
            "observation channel order must be exactly TE rho/phase then TM rho/phase"
        )
    shape = (sample_index.size, frequency.size, station.size)

    def observed(name: str, *, phase: bool = False) -> np.ndarray:
        values = _array(arrays[name], name=f"observation.{name}", dtype="<f4", ndim=3)
        if values.shape != shape or not np.isfinite(values).all():
            raise Pimsr2DValidationError(
                f"observation.{name} must be finite with shape {shape}"
            )
        if phase and np.any((values < 0.0) | (values >= 180.0)):
            raise Pimsr2DValidationError(
                f"observation.{name} violates the declared [0, 180) convention"
            )
        return values

    rho_te = observed("observed_log10_rho_te")
    phase_te = observed("observed_phase_te_degrees", phase=True)
    rho_tm = observed("observed_log10_rho_tm")
    phase_tm = observed("observed_phase_tm_degrees", phase=True)
    for name in (
        "declared_evaluation_floor_log10_rho_te",
        "declared_evaluation_floor_phase_te_degrees",
        "declared_evaluation_floor_log10_rho_tm",
        "declared_evaluation_floor_phase_tm_degrees",
    ):
        floor = _array(arrays[name], name=f"observation.{name}", dtype="<f4", ndim=3)
        if floor.shape != shape or not np.isfinite(floor).all() or np.any(floor <= 0):
            raise Pimsr2DValidationError(
                f"observation.{name} must be finite, positive and have shape {shape}"
            )
    mask = _array(
        arrays["valid_mask"], name="observation.valid_mask", dtype=np.bool_, ndim=4
    )
    expected_mask_shape = (sample_index.size, 4, frequency.size, station.size)
    if mask.shape != expected_mask_shape or not np.all(mask):
        raise Pimsr2DValidationError(
            "PIMSR 2D requires an explicit all-true valid mask with shape "
            f"{expected_mask_shape}"
        )
    return Observations2D(
        sample_index=sample_index,
        frequency_hz=frequency,
        station_x_m=station,
        x_cell_centers_m=x_centers,
        depth_cell_centers_m=depth_centers,
        log10_rho_te=rho_te,
        phase_te_degrees=phase_te,
        log10_rho_tm=rho_tm,
        phase_tm_degrees=phase_tm,
        artifact_sha256=snapshot.sha256,
        artifact_size_bytes=snapshot.size_bytes,
    )


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise Pimsr2DValidationError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise Pimsr2DValidationError(f"{name} must be >= {minimum}")
    return result


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise Pimsr2DValidationError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise Pimsr2DValidationError(f"{name} must be finite")
    return result


def _normalization_sha256(mean: np.ndarray, std: np.ndarray) -> str:
    digest = hashlib.sha256()
    for key, values in (("mean", mean), ("std", std)):
        digest.update(key.encode("ascii"))
        shape = json.dumps(list(values.shape), sort_keys=True, separators=(",", ":"))
        digest.update(shape.encode("ascii"))
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def _require_hash_field(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise Pimsr2DValidationError(f"{name} must be a SHA-256 string")
    validated = _valid_sha256(value, name)
    assert validated is not None
    return validated


def _validate_dataset_identity(
    value: object,
    *,
    name: str,
    contract_sha256: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Pimsr2DValidationError(f"{name} must be a mapping")
    _require_exact_keys(value, _DATASET_IDENTITY_KEYS, name)
    if value["identity_schema"] != "pimsr-mt-2d-artifact-identity":
        raise Pimsr2DValidationError(f"{name} has an unsupported identity schema")
    if _integer(value["identity_schema_version"], f"{name}.identity_schema_version", minimum=1) != 1:
        raise Pimsr2DValidationError(f"{name} has an unsupported identity version")
    _require_hash_field(value["artifact_sha256"], f"{name}.artifact_sha256")
    _integer(value["artifact_size_bytes"], f"{name}.artifact_size_bytes", minimum=1)
    if _require_hash_field(value["contract_sha256"], f"{name}.contract_sha256") != contract_sha256:
        raise Pimsr2DValidationError(f"{name} is bound to a different data contract")
    provenance = value["provenance"]
    if not isinstance(provenance, Mapping) or not provenance:
        raise Pimsr2DValidationError(f"{name}.provenance must be a non-empty mapping")
    provenance_sha = _require_hash_field(
        value["provenance_sha256"], f"{name}.provenance_sha256"
    )
    if provenance_sha != _json_sha256(provenance):
        raise Pimsr2DValidationError(f"{name} has a bad provenance digest")
    base = {key: value[key] for key in value if key != "identity_sha256"}
    identity_sha = _require_hash_field(value["identity_sha256"], f"{name}.identity_sha256")
    if identity_sha != _json_sha256(base):
        raise Pimsr2DValidationError(f"{name} has a bad identity digest")
    return value


def _validate_training_metadata(state: Mapping[str, Any]) -> None:
    model_config = state["model_config"]
    if not isinstance(model_config, Mapping):
        raise Pimsr2DValidationError("checkpoint.model_config must be a mapping")
    _require_exact_keys(model_config, _MODEL_CONFIG_KEYS, "checkpoint.model_config")
    if model_config["architecture"] != "pimsr_inversion.PimsrNet2D/v3":
        raise Pimsr2DValidationError("checkpoint has an unsupported model architecture")
    for key in (
        "n_freq",
        "n_stations",
        "n_depth",
        "n_x",
        "in_channels",
        "width",
        "n_scenarios",
    ):
        if _integer(model_config[key], f"checkpoint.model_config.{key}", minimum=1) != state[key]:
            raise Pimsr2DValidationError(f"checkpoint.model_config.{key} is inconsistent")
    if model_config["scen_head"] != state["scen_head"]:
        raise Pimsr2DValidationError("checkpoint.model_config.scen_head is inconsistent")

    training = state["training_config"]
    if not isinstance(training, Mapping):
        raise Pimsr2DValidationError("checkpoint.training_config must be a mapping")
    _require_exact_keys(training, _TRAINING_CONFIG_KEYS, "checkpoint.training_config")
    for key, minimum in (
        ("epochs", 1),
        ("batch_size", 1),
        ("sigma_warmup", 0),
        ("seed", 0),
        ("workers", 0),
        ("scheduler_t_max", 1),
    ):
        _integer(training[key], f"checkpoint.training_config.{key}", minimum=minimum)
    for key in (
        "learning_rate",
        "weight_decay",
        "gradient_clip_norm",
        "sigma_regularization",
        "beta_nll",
    ):
        _finite_real(training[key], f"checkpoint.training_config.{key}")
    if training["normalization"] != "per-channel-train-mean-std/v1":
        raise Pimsr2DValidationError("checkpoint normalization policy is unsupported")
    if training["optimizer"] != "torch.optim.AdamW" or training["scheduler"] != "CosineAnnealingLR":
        raise Pimsr2DValidationError("checkpoint optimizer/scheduler metadata is unsupported")
    weights = np.asarray(training["class_weights"], dtype=np.float64)
    if weights.shape != (5,) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise Pimsr2DValidationError("checkpoint class_weights must contain five positives")
    if not isinstance(training["runtime"], Mapping) or not training["runtime"]:
        raise Pimsr2DValidationError("checkpoint training runtime metadata is missing")
    if _finite_real(state["beta"], "checkpoint.beta") != _finite_real(
        training["beta_nll"], "checkpoint.training_config.beta_nll"
    ):
        raise Pimsr2DValidationError("checkpoint beta metadata is inconsistent")


def _validate_checkpoint_state(
    state: Mapping[str, Any], observations: Observations2D
) -> tuple[np.ndarray, np.ndarray]:
    _require_exact_keys(state, _CHECKPOINT_KEYS, "checkpoint")
    if state["checkpoint_schema"] != CHECKPOINT_SCHEMA:
        raise Pimsr2DValidationError("legacy or unsupported PIMSR 2D checkpoint schema")
    if _integer(
        state["checkpoint_schema_version"],
        "checkpoint.checkpoint_schema_version",
        minimum=1,
    ) != CHECKPOINT_SCHEMA_VERSION:
        raise Pimsr2DValidationError("legacy or unsupported PIMSR 2D checkpoint version")

    for key in ("n_freq", "n_stations", "n_depth", "n_x", "n_scenarios", "width", "in_channels"):
        _integer(state[key], f"checkpoint.{key}", minimum=1)
    if state["in_channels"] != 4:
        raise Pimsr2DValidationError("checkpoint.in_channels must equal four")
    if state["n_scenarios"] != 5 or state["scen_head"] not in {"gap", "multiscale"}:
        raise Pimsr2DValidationError("checkpoint scenario-head metadata is unsupported")

    contract = state["data_contract"]
    if not isinstance(contract, Mapping):
        raise Pimsr2DValidationError("checkpoint.data_contract must be a mapping")
    _require_exact_keys(contract, _DATA_CONTRACT_KEYS, "checkpoint.data_contract")
    expected_axes = {
        "frequencies": observations.frequency_hz,
        "station_x": observations.station_x_m,
        "x_grid": observations.x_cell_centers_m,
        "depth_grid": observations.depth_cell_centers_m,
    }
    for key, expected in expected_axes.items():
        actual = np.asarray(contract[key], dtype=np.float64)
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            raise Pimsr2DValidationError(
                f"checkpoint and observation payload have different {key} geometry"
            )
    dimensions = {
        "n_freq": observations.frequency_hz.size,
        "n_stations": observations.station_x_m.size,
        "n_depth": observations.depth_cell_centers_m.size,
        "n_x": observations.x_cell_centers_m.size,
    }
    for key, expected in dimensions.items():
        if state[key] != expected:
            raise Pimsr2DValidationError(f"checkpoint.{key} does not match observations")

    # The inversion package owns the physical contract validation.  Calling it
    # here also rejects checkpoints from the pre-TE=Zyx/TM=Zxy era.
    try:
        from pimsr_inversion.contracts2d import validate_checkpoint2d

        validate_checkpoint2d(state)
    except Exception as exc:
        raise Pimsr2DValidationError(f"checkpoint physical contract is invalid: {exc}") from exc

    mean = np.asarray(state["stats_mean"])
    std = np.asarray(state["stats_std"])
    for name, values in (("stats_mean", mean), ("stats_std", std)):
        if values.dtype != np.dtype("<f4") or values.shape != (1, 4, 1, 1):
            raise Pimsr2DValidationError(
                f"checkpoint.{name} must be float32 with shape (1, 4, 1, 1)"
            )
        if not np.isfinite(values).all():
            raise Pimsr2DValidationError(f"checkpoint.{name} must be finite")
    if np.any(std <= 0):
        raise Pimsr2DValidationError("checkpoint.stats_std must be strictly positive")
    normalized_digest = _require_hash_field(
        state["normalization_sha256"], "checkpoint.normalization_sha256"
    )
    if normalized_digest != _normalization_sha256(mean, std):
        raise Pimsr2DValidationError("checkpoint normalization statistics have a bad digest")

    _validate_training_metadata(state)
    contract_sha = _json_sha256(contract)
    identities = state["dataset_identities"]
    if not isinstance(identities, Mapping) or set(identities) != {"train", "val"}:
        raise Pimsr2DValidationError("checkpoint must identify exact train and val artifacts")
    train_identity = _validate_dataset_identity(
        identities["train"], name="checkpoint.dataset_identities.train", contract_sha256=contract_sha
    )
    val_identity = _validate_dataset_identity(
        identities["val"], name="checkpoint.dataset_identities.val", contract_sha256=contract_sha
    )
    if train_identity["artifact_sha256"] == val_identity["artifact_sha256"]:
        raise Pimsr2DValidationError("checkpoint train and val artifacts must be distinct")

    epoch = _integer(state["epoch"], "checkpoint.epoch")
    epochs = int(state["training_config"]["epochs"])
    if epoch >= epochs:
        raise Pimsr2DValidationError("checkpoint.epoch is outside the training schedule")
    best_epoch = _integer(state["best_epoch"], "checkpoint.best_epoch")
    if best_epoch > epoch:
        raise Pimsr2DValidationError("checkpoint.best_epoch cannot exceed checkpoint.epoch")
    _finite_real(state["best_val_loss"], "checkpoint.best_val_loss")
    history = state["history"]
    if not isinstance(history, list) or len(history) != epoch + 1:
        raise Pimsr2DValidationError("checkpoint history is inconsistent with checkpoint.epoch")
    for index, record in enumerate(history):
        if not isinstance(record, Mapping) or set(record) != {
            "epoch",
            "train_loss",
            "val_loss",
            "val_rmse",
        }:
            raise Pimsr2DValidationError("checkpoint history contains an invalid record")
        if record["epoch"] != index:
            raise Pimsr2DValidationError("checkpoint history epochs are not sequential")
        for key in ("train_loss", "val_loss", "val_rmse"):
            _finite_real(record[key], f"checkpoint.history[{index}].{key}")
    if not isinstance(state["model_state"], Mapping) or not state["model_state"]:
        raise Pimsr2DValidationError("checkpoint.model_state must be a non-empty mapping")
    for name, tensor in state["model_state"].items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise Pimsr2DValidationError("checkpoint.model_state must map strings to tensors")
        if (tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(tensor).all():
            raise Pimsr2DValidationError(f"checkpoint.model_state[{name!r}] is non-finite")
    for key in ("optimizer_state", "scheduler_state", "rng_state"):
        if not isinstance(state[key], Mapping):
            raise Pimsr2DValidationError(f"checkpoint.{key} must be a mapping")
    return np.ascontiguousarray(mean), np.ascontiguousarray(std)


def load_checkpoint_2d(
    path: str | Path,
    observations: Observations2D,
    *,
    expected_sha256: str | None = None,
) -> ValidatedCheckpoint2D:
    """Load an exact ``pimsr-train-2d`` schema-v1 checkpoint."""
    snapshot = _snapshot(path, expected_sha256=expected_sha256, role="checkpoint")
    safe_globals = getattr(torch.serialization, "safe_globals", None)
    if safe_globals is None:
        raise Pimsr2DValidationError(
            "installed PyTorch cannot safely load the NumPy-bearing checkpoint"
        )
    numpy_core = getattr(np, "_core", np.core)
    allowed_numpy_globals = [
        numpy_core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        type(np.dtype(np.float32)),
        type(np.dtype(np.uint32)),
    ]
    try:
        with safe_globals(allowed_numpy_globals):
            state = torch.load(
                io.BytesIO(snapshot.payload),
                map_location="cpu",
                weights_only=True,
            )
    except Exception as exc:
        raise Pimsr2DValidationError(
            f"cannot safely decode PIMSR 2D checkpoint: {exc}"
        ) from exc
    if not isinstance(state, Mapping):
        raise Pimsr2DValidationError("checkpoint root must be a mapping")
    mean, std = _validate_checkpoint_state(state, observations)
    try:
        model = PimsrNet2D.from_checkpoint(dict(state))
    except Exception as exc:
        raise Pimsr2DValidationError(f"checkpoint model state is incompatible: {exc}") from exc
    return ValidatedCheckpoint2D(
        state=state,
        model=model,
        mean=mean,
        std=std,
        artifact_sha256=snapshot.sha256,
        artifact_size_bytes=snapshot.size_bytes,
    )


def normalized_observation_tensor(
    observations: Observations2D,
    checkpoint: ValidatedCheckpoint2D,
) -> np.ndarray:
    """Apply the exact ``Section2DDataset`` channel and normalization transform."""
    channels = np.stack(
        (
            observations.log10_rho_te,
            observations.phase_te_degrees / np.float32(45.0),
            observations.log10_rho_tm,
            observations.phase_tm_degrees / np.float32(45.0),
        ),
        axis=1,
    ).astype("<f4", copy=False)
    normalized = (channels - checkpoint.mean) / checkpoint.std
    normalized = np.ascontiguousarray(normalized, dtype="<f4")
    if not np.isfinite(normalized).all():
        raise Pimsr2DValidationError("normalized PIMSR 2D input is non-finite")
    return normalized


def _resolve_device(requested: str) -> torch.device:
    if not isinstance(requested, str) or not requested:
        raise Pimsr2DValidationError("device must be a non-empty string")
    value = "cuda" if requested == "auto" and torch.cuda.is_available() else requested
    if requested == "auto" and not torch.cuda.is_available():
        value = "cpu"
    try:
        device = torch.device(value)
    except (RuntimeError, ValueError) as exc:
        raise Pimsr2DValidationError(f"invalid device {requested!r}: {exc}") from exc
    if device.type not in {"cpu", "cuda"}:
        raise Pimsr2DValidationError("PIMSR 2D inference supports only cpu or cuda")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise Pimsr2DValidationError("CUDA was requested but is not available")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index < 0 or index >= torch.cuda.device_count():
            raise Pimsr2DValidationError(f"CUDA device index is unavailable: {index}")
        device = torch.device("cuda", index)
    return device


def _configure_determinism() -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    for option in (
        "allow_fp16_reduced_precision_reduction",
        "allow_bf16_reduced_precision_reduction",
    ):
        if hasattr(torch.backends.cuda.matmul, option):
            setattr(torch.backends.cuda.matmul, option, False)
    torch.backends.cudnn.allow_tf32 = False
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.mkldnn, "deterministic"):
        torch.backends.mkldnn.deterministic = True


def _module_source_identity() -> dict[str, Any]:
    network_path = Path(inspect.getfile(PimsrNet2D)).resolve(strict=True)
    package_root = network_path.parent
    files = {
        "network2d.py": network_path,
        "contracts2d.py": package_root / "contracts2d.py",
        "train2d.py": package_root / "train2d.py",
    }
    records: dict[str, Any] = {}
    for name, path in files.items():
        snapshot = _snapshot(path, expected_sha256=None, role=f"source_{name}")
        records[name] = {
            "sha256": snapshot.sha256,
            "size_bytes": snapshot.size_bytes,
        }
    try:
        version = importlib.metadata.version("pimsr-inversion")
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {"distribution_version": version, "module_files": records}


def _git(
    repository: Path,
    *arguments: str,
) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
        raise Pimsr2DValidationError(f"cannot inspect source repository: {exc}") from exc
    return result.stdout.strip()


def _source_identity(
    source_repository: str | Path | None,
    expected_source_commit: str | None,
) -> dict[str, Any]:
    installed = _module_source_identity()
    if source_repository is None:
        if expected_source_commit is not None:
            raise Pimsr2DValidationError(
                "expected_source_commit requires source_repository"
            )
        return {
            **installed,
            "repository_checked": False,
            "head_commit": None,
            "dirty_tree": None,
        }
    if expected_source_commit is not None and (
        not isinstance(expected_source_commit, str)
        or len(expected_source_commit) != _GIT_COMMIT_LENGTH
        or any(character not in "0123456789abcdef" for character in expected_source_commit)
    ):
        raise Pimsr2DValidationError(
            "expected_source_commit must be 40 lowercase hexadecimal characters"
        )
    requested = Path(source_repository)
    if requested.is_symlink():
        raise Pimsr2DValidationError("source_repository must not be a symbolic link")
    try:
        repository = requested.resolve(strict=True)
    except OSError as exc:
        raise Pimsr2DValidationError(f"cannot resolve source_repository: {exc}") from exc
    top = Path(_git(repository, "rev-parse", "--show-toplevel")).resolve(strict=True)
    if top != repository:
        raise Pimsr2DValidationError("source_repository must name the exact Git root")
    commit = _git(repository, "rev-parse", "HEAD")
    if len(commit) != _GIT_COMMIT_LENGTH:
        raise Pimsr2DValidationError("source repository returned an invalid HEAD")
    if expected_source_commit is not None and commit != expected_source_commit:
        raise Pimsr2DValidationError("source repository HEAD does not match the pinned commit")
    dirty_output = _git(repository, "status", "--porcelain", "--untracked-files=normal")
    dirty = bool(dirty_output)
    if dirty:
        raise Pimsr2DValidationError("source repository must be clean for benchmark inference")
    repository_files = {
        name: repository / "src" / "pimsr_inversion" / name
        for name in installed["module_files"]
    }
    for name, path in repository_files.items():
        snapshot = _snapshot(path, expected_sha256=None, role=f"repository_source_{name}")
        if snapshot.sha256 != installed["module_files"][name]["sha256"]:
            raise Pimsr2DValidationError(
                f"imported pimsr-inversion {name} differs from the checked repository"
            )
    return {
        **installed,
        "repository_checked": True,
        "head_commit": commit,
        "dirty_tree": False,
    }


def _npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, array, allow_pickle=False)
    return stream.getvalue()


def _write_prediction_npz(
    path: Path,
    observations_sha256: str,
    sample_index: np.ndarray,
    x_cell_centers_m: np.ndarray,
    depth_cell_centers_m: np.ndarray,
    prediction: np.ndarray,
) -> None:
    arrays = {
        "schema": np.asarray(PREDICTION_SCHEMA, dtype=f"<U{len(PREDICTION_SCHEMA)}"),
        "schema_version": np.asarray(PREDICTION_SCHEMA_VERSION, dtype="<i8"),
        "observations_sha256": np.asarray(observations_sha256, dtype="<U64"),
        "sample_index": np.ascontiguousarray(sample_index, dtype="<i8"),
        "x_cell_centers_m": np.ascontiguousarray(x_cell_centers_m, dtype="<f8"),
        "depth_cell_centers_m": np.ascontiguousarray(
            depth_cell_centers_m, dtype="<f8"
        ),
        "predicted_log10_resistivity": np.ascontiguousarray(prediction, dtype="<f4"),
    }
    with path.open("xb") as raw:
        with zipfile.ZipFile(
            raw,
            mode="w",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
            strict_timestamps=True,
        ) as archive:
            for name in _PREDICTION_KEYS:
                member = zipfile.ZipInfo(f"{name}.npy", (1980, 1, 1, 0, 0, 0))
                member.compress_type = zipfile.ZIP_STORED
                member.create_system = 3
                member.external_attr = (stat.S_IFREG | 0o644) << 16
                archive.writestr(member, _npy_bytes(arrays[name]))
        raw.flush()
        os.fsync(raw.fileno())


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _prepare_outputs(
    observation_path: str | Path,
    checkpoint_path: str | Path,
    prediction_path: str | Path,
    runtime_path: str | Path,
) -> tuple[tuple[Path, Path], tuple[Path, Path]]:
    inputs = {
        Path(observation_path).resolve(strict=True),
        Path(checkpoint_path).resolve(strict=True),
    }
    destinations: list[Path] = []
    parts: list[Path] = []
    for requested, suffix in ((prediction_path, ".npz"), (runtime_path, ".json")):
        target = Path(requested)
        if target.suffix.lower() != suffix:
            raise Pimsr2DValidationError(f"output {target} must use the {suffix} suffix")
        target.parent.mkdir(parents=True, exist_ok=True)
        resolved = target.resolve(strict=False)
        part = resolved.with_name(resolved.name + ".part")
        if resolved in inputs:
            raise Pimsr2DValidationError("output must not replace an input artifact")
        if resolved.exists():
            raise Pimsr2DPublicationError(f"refusing to overwrite output: {resolved}")
        if part.exists():
            raise Pimsr2DPublicationError(f"stale partial output requires inspection: {part}")
        destinations.append(resolved)
        parts.append(part)
    if destinations[0] == destinations[1]:
        raise Pimsr2DValidationError("prediction and runtime outputs must be distinct")
    return (destinations[0], destinations[1]), (parts[0], parts[1])


def _publish(parts: Sequence[Path], destinations: Sequence[Path]) -> None:
    published: list[tuple[Path, tuple[int, int]]] = []
    try:
        for part, destination in zip(parts, destinations, strict=True):
            part_info = part.stat(follow_symlinks=False)
            if part.is_symlink() or not stat.S_ISREG(part_info.st_mode):
                raise Pimsr2DPublicationError(
                    f"staged artifact must be a regular file: {part}"
                )
            expected_identity = (int(part_info.st_dev), int(part_info.st_ino))
            try:
                os.link(part, destination)
            except FileExistsError as exc:
                raise Pimsr2DPublicationError(
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
                raise Pimsr2DPublicationError(
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
            raise Pimsr2DPublicationError(
                "refusing to delete outputs replaced during rollback: "
                + ", ".join(unsafe)
            ) from exc
        raise


def run_pimsr2d_inference(
    observations_path: str | Path,
    checkpoint_path: str | Path,
    prediction_path: str | Path,
    runtime_path: str | Path,
    *,
    expected_observations_sha256: str | None = None,
    expected_checkpoint_sha256: str | None = None,
    batch_size: int = 64,
    device: str = "auto",
    source_repository: str | Path | None = None,
    expected_source_commit: str | None = None,
) -> Pimsr2DInferenceResult:
    """Run deterministic inference without loading or accepting withheld truth."""
    batch = _integer(batch_size, "batch_size", minimum=1)
    destinations, parts = _prepare_outputs(
        observations_path,
        checkpoint_path,
        prediction_path,
        runtime_path,
    )
    wall_started = time.perf_counter()
    try:
        observations = load_observations_2d(
            observations_path,
            expected_sha256=expected_observations_sha256,
        )
        checkpoint = load_checkpoint_2d(
            checkpoint_path,
            observations,
            expected_sha256=expected_checkpoint_sha256,
        )
        source = _source_identity(source_repository, expected_source_commit)
        resolved_device = _resolve_device(device)
        _configure_determinism()
        inputs = normalized_observation_tensor(observations, checkpoint)
        model = checkpoint.model.to(device=resolved_device, dtype=torch.float32)
        model.eval()
        prediction = np.empty(
            (
                observations.sample_index.size,
                observations.depth_cell_centers_m.size,
                observations.x_cell_centers_m.size,
            ),
            dtype="<f4",
        )
        if resolved_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(resolved_device)
            torch.cuda.synchronize(resolved_device)
        inference_started = time.perf_counter()
        with torch.inference_mode():
            for start in range(0, inputs.shape[0], batch):
                stop = min(start + batch, inputs.shape[0])
                tensor = torch.from_numpy(inputs[start:stop]).to(
                    resolved_device,
                    dtype=torch.float32,
                    non_blocking=False,
                )
                output = model(tensor)
                if not isinstance(output, Mapping) or "log_rho" not in output:
                    raise Pimsr2DValidationError("PIMSR model did not return log_rho")
                values = output["log_rho"]
                expected_shape = (stop - start, prediction.shape[1], prediction.shape[2])
                if values.dtype != torch.float32 or tuple(values.shape) != expected_shape:
                    raise Pimsr2DValidationError(
                        f"PIMSR output must be float32 with shape {expected_shape}"
                    )
                if not torch.isfinite(values).all():
                    raise Pimsr2DValidationError("PIMSR output contains non-finite values")
                prediction[start:stop] = values.detach().cpu().numpy()
        if resolved_device.type == "cuda":
            torch.cuda.synchronize(resolved_device)
        inference_seconds = time.perf_counter() - inference_started
        peak_memory = (
            int(torch.cuda.max_memory_allocated(resolved_device))
            if resolved_device.type == "cuda"
            else None
        )
        if not np.isfinite(prediction).all():
            raise Pimsr2DValidationError("prediction artifact would contain non-finite values")

        _write_prediction_npz(
            parts[0],
            observations.artifact_sha256,
            observations.sample_index,
            observations.x_cell_centers_m,
            observations.depth_cell_centers_m,
            prediction,
        )
        prediction_snapshot = _snapshot(
            parts[0], expected_sha256=None, role="staged_prediction"
        )
        prediction_arrays = _npz_arrays(
            prediction_snapshot,
            keys=frozenset(_PREDICTION_KEYS),
            role="staged_prediction",
        )
        _require_schema(
            prediction_arrays["schema"], PREDICTION_SCHEMA, "staged_prediction"
        )
        _require_version(
            prediction_arrays["schema_version"],
            "staged_prediction",
            PREDICTION_SCHEMA_VERSION,
        )
        prediction_observations_sha256 = prediction_arrays["observations_sha256"]
        if (
            prediction_observations_sha256.ndim != 0
            or prediction_observations_sha256.dtype.kind != "U"
            or prediction_observations_sha256.item()
            != observations.artifact_sha256
        ):
            raise Pimsr2DPublicationError(
                "staged prediction is not bound to the exact observation artifact"
            )
        prediction_x_centers = _axis(
            prediction_arrays["x_cell_centers_m"],
            name="staged_prediction.x_cell_centers_m",
            positive=False,
        )
        prediction_depth_centers = _axis(
            prediction_arrays["depth_cell_centers_m"],
            name="staged_prediction.depth_cell_centers_m",
            positive=True,
        )
        if not np.array_equal(
            prediction_x_centers, observations.x_cell_centers_m
        ) or not np.array_equal(
            prediction_depth_centers, observations.depth_cell_centers_m
        ):
            raise Pimsr2DPublicationError(
                "staged prediction grid differs from the observation payload"
            )

        wall_seconds = time.perf_counter() - wall_started
        state = checkpoint.state
        runtime = {
            "schema": RUNTIME_SCHEMA,
            "schema_version": RUNTIME_SCHEMA_VERSION,
            "method": "pimsr-2d",
            "inputs": {
                "observations": {
                    "schema": OBSERVATION_SCHEMA,
                    "schema_version": OBSERVATION_SCHEMA_VERSION,
                    "sha256": observations.artifact_sha256,
                    "size_bytes": observations.artifact_size_bytes,
                    "sample_count": int(observations.sample_index.size),
                },
                "checkpoint": {
                    "schema": CHECKPOINT_SCHEMA,
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "sha256": checkpoint.artifact_sha256,
                    "size_bytes": checkpoint.artifact_size_bytes,
                    "epoch": int(state["epoch"]),
                    "best_epoch": int(state["best_epoch"]),
                    "best_val_loss": float(state["best_val_loss"]),
                    "normalization_sha256": state["normalization_sha256"],
                    "data_contract_sha256": _json_sha256(state["data_contract"]),
                    "dataset_identities_sha256": _json_sha256(
                        state["dataset_identities"]
                    ),
                    "training_config_sha256": _json_sha256(state["training_config"]),
                    "model_config": dict(state["model_config"]),
                },
            },
            "output": {
                "schema": PREDICTION_SCHEMA,
                "schema_version": PREDICTION_SCHEMA_VERSION,
                "sha256": prediction_snapshot.sha256,
                "size_bytes": prediction_snapshot.size_bytes,
            },
            "execution": {
                "batch_size": batch,
                "device_requested": device,
                "device_resolved": str(resolved_device),
                "device_name": (
                    torch.cuda.get_device_name(resolved_device)
                    if resolved_device.type == "cuda"
                    else "cpu"
                ),
                "precision": "float32",
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "inference_seconds": float(inference_seconds),
                "wall_seconds": float(wall_seconds),
                "peak_cuda_memory_bytes": peak_memory,
            },
            "software": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "torch": str(torch.__version__),
                "cuda_runtime": str(torch.version.cuda) if torch.version.cuda else None,
                "cudnn": torch.backends.cudnn.version(),
            },
            "source": source,
        }
        runtime_payload = _canonical_json_bytes(runtime)
        _write_bytes(parts[1], runtime_payload)
        runtime_snapshot = _snapshot(parts[1], expected_sha256=None, role="staged_runtime")
        if runtime_snapshot.payload != runtime_payload:
            raise Pimsr2DPublicationError("runtime JSON changed while staged")
        _publish(parts, destinations)
    finally:
        for part in parts:
            part.unlink(missing_ok=True)

    return Pimsr2DInferenceResult(
        prediction_path=destinations[0],
        runtime_path=destinations[1],
        observation_sha256=observations.artifact_sha256,
        checkpoint_sha256=checkpoint.artifact_sha256,
        prediction_sha256=prediction_snapshot.sha256,
        runtime_sha256=runtime_snapshot.sha256,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strict deterministic PIMSR 2D SOTA inference adapter"
    )
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--expected-observations-sha256")
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--source-repository", type=Path)
    parser.add_argument("--expected-source-commit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_pimsr2d_inference(
        args.observations,
        args.checkpoint,
        args.output,
        args.runtime,
        expected_observations_sha256=args.expected_observations_sha256,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        batch_size=args.batch_size,
        device=args.device,
        source_repository=args.source_repository,
        expected_source_commit=args.expected_source_commit,
    )
    print(f"prediction: {result.prediction_path} sha256={result.prediction_sha256}")
    print(f"runtime: {result.runtime_path} sha256={result.runtime_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
