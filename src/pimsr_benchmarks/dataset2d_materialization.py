"""Fail-closed materialization of a PIMSR schema-v2 2D benchmark.

The source HDF5 artifact contains noisy observations, clean forward responses
and geological truth.  External methods must never receive the latter two.
This module therefore publishes two deterministic NPZ payloads plus two
canonical JSON manifests:

* an observation-only method input;
* a separately withheld truth payload for the evaluator; and
* a sanitized public observation manifest for external methods; and
* an operator-only scoring manifest with secret generation provenance.

Materialization intentionally snapshots the complete evaluation set in
memory.  It is an exporter for bounded held-out evaluation sets, not for large
training splits.

NPZ members are stored as canonical little-endian NPY streams inside a ZIP
archive with fixed metadata.  Repeating a materialization with identical
inputs and output basenames therefore produces byte-identical artifacts.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import os
import re
import stat
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import h5py
import numpy as np

PUBLIC_MANIFEST_SCHEMA = "pimsr-sota-2d-observation-manifest"
OPERATOR_MANIFEST_SCHEMA = "pimsr-sota-2d-scoring-manifest"
MANIFEST_SCHEMA_VERSION = 2
OBSERVATION_SCHEMA = "pimsr-sota-2d-observations"
TRUTH_SCHEMA = "pimsr-sota-2d-truth"
PAYLOAD_SCHEMA_VERSION = 1
UNCERTAINTY_POLICY_ID = "declared_evaluation_floors_log10_rho_phase_v1"

DEFAULT_RHO_LOG10_FLOOR = 0.05
DEFAULT_PHASE_DEGREE_FLOOR = 2.9

OBSERVATION_CHANNEL_ORDER = (
    "log10_rho_te",
    "phase_te_degrees",
    "log10_rho_tm",
    "phase_tm_degrees",
)

_OBSERVATION_AXES = ("sample", "frequency", "station")
_TRUTH_AXES = ("sample", "depth", "x")
_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_SOURCE_OBSERVATIONS = {
    "observed_log10_rho_te": "obs_mt_log10_rho",
    "observed_phase_te_degrees": "obs_mt_phase",
    "observed_log10_rho_tm": "obs_mt_log10_rho_tm",
    "observed_phase_tm_degrees": "obs_mt_phase_tm",
}


@dataclass(frozen=True)
class MaterializationResult:
    """Paths and exact payload identities for one completed publication."""

    observations_path: Path
    truth_path: Path
    public_manifest_path: Path
    operator_manifest_path: Path
    observations_sha256: str
    truth_sha256: str
    public_manifest_sha256: str
    operator_manifest_sha256: str


class MaterializationError(RuntimeError):
    """Raised when a source or output violates the materialization contract."""


def _text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def _text_tuple(value: object) -> tuple[str, ...]:
    return tuple(_text(item) for item in np.asarray(value).reshape(-1))


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer scalar")
    result = int(array)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _positive_float32(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    source = float(value)
    if not np.isfinite(source) or source <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    stored = np.float32(source)
    if not np.isfinite(stored) or stored <= 0.0:
        raise ValueError(f"{name} is not representable as a positive float32")
    return float(stored)


def _split_id(value: object) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError("split_id must match ^[a-z0-9][a-z0-9_.-]*$")
    return value


def _load_sample_id_key(
    value: bytes | bytearray | memoryview | str | Path,
) -> bytes:
    if isinstance(value, (bytes, bytearray, memoryview)):
        key = bytes(value)
    elif isinstance(value, (str, Path)):
        requested = Path(value)
        if requested.is_symlink():
            raise ValueError("sample_id_key path must not be a symbolic link")
        try:
            before = requested.stat()
            key = requested.read_bytes()
            after = requested.stat()
        except OSError as exc:
            raise ValueError(f"cannot read sample_id_key file: {requested}") from exc
        before_signature = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        after_signature = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or before_signature != after_signature
            or len(key) != before.st_size
        ):
            raise MaterializationError("sample_id_key file changed while it was read")
    else:
        raise TypeError("sample_id_key must be secret bytes or a path to a secret file")
    if len(key) < 32:
        raise ValueError("sample_id_key must contain at least 32 bytes")
    return key


def _opaque_sample_indices(
    source_indices: np.ndarray,
    *,
    generator_seed: int,
    split_id: str,
    key: bytes,
) -> np.ndarray:
    """Map private generator indices to stable, unlinkable non-negative int64 IDs."""
    prefix = b"pimsr-sota-2d-opaque-sample-id-v1\x00"
    split = split_id.encode("ascii")
    values: list[int] = []
    for source_index in source_indices:
        index = int(source_index)
        message = (
            prefix
            + generator_seed.to_bytes(8, "big", signed=False)
            + index.to_bytes(8, "big", signed=False)
            + len(split).to_bytes(4, "big", signed=False)
            + split
        )
        digest = hmac.digest(key, message, "sha256")
        values.append(int.from_bytes(digest[:8], "big") & np.iinfo(np.int64).max)
    opaque = np.asarray(values, dtype="<i8")
    if len(np.unique(opaque)) != opaque.size:
        raise MaterializationError("HMAC-derived opaque sample_index collision")
    return opaque


def _canonical_array(value: object, dtype: str | np.dtype[Any]) -> np.ndarray:
    array = np.asarray(value, dtype=np.dtype(dtype))
    if array.ndim == 0:
        return array.copy()
    return np.ascontiguousarray(array)


def _unicode_scalar(value: str) -> np.ndarray:
    return np.asarray(value, dtype=f"<U{len(value)}")


def _unicode_vector(values: Sequence[str]) -> np.ndarray:
    width = max((len(value) for value in values), default=1)
    return np.asarray(values, dtype=f"<U{width}")


def _npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, array, allow_pickle=False)
    return stream.getvalue()


def _array_record(
    array: np.ndarray,
    payload: bytes,
    *,
    axis_order: Sequence[str],
) -> dict[str, object]:
    if len(axis_order) != array.ndim:
        raise ValueError("array axis metadata does not match array rank")
    return {
        "axis_order": list(axis_order),
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _write_deterministic_npz(
    path: Path,
    arrays: Sequence[tuple[str, np.ndarray, Sequence[str]]],
) -> dict[str, dict[str, object]]:
    """Write fixed-metadata, uncompressed NPZ and return member identities."""
    records: dict[str, dict[str, object]] = {}
    with path.open("xb") as raw_stream:
        with zipfile.ZipFile(
            raw_stream,
            mode="w",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
            strict_timestamps=True,
        ) as archive:
            for name, array, axis_order in arrays:
                if name in records:
                    raise ValueError(f"duplicate NPZ array name: {name}")
                payload = _npy_bytes(array)
                member = zipfile.ZipInfo(f"{name}.npy", (1980, 1, 1, 0, 0, 0))
                member.compress_type = zipfile.ZIP_STORED
                member.create_system = 3
                member.external_attr = (stat.S_IFREG | 0o644) << 16
                archive.writestr(member, payload)
                records[name] = _array_record(
                    array,
                    payload,
                    axis_order=axis_order,
                )
        raw_stream.flush()
        os.fsync(raw_stream.fileno())
    return records


def _canonical_json_bytes(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise MaterializationError(f"manifest is not canonical JSON data: {exc}") from exc
    return (encoded + "\n").encode("utf-8")


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _stable_file_identity(path: Path, *, role: str) -> tuple[str, int]:
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or path.is_symlink():
        raise ValueError(f"{role} must be a regular non-symlink file: {path}")
    digest, size = _sha256_file(path)
    after = path.stat()
    before_signature = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_signature = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_signature != after_signature or size != before.st_size:
        raise MaterializationError(f"{role} changed while it was hashed: {path}")
    return digest, size


def _source_contract(path: Path) -> None:
    """Delegate the complete producer schema-v2 check to the pinned producer."""
    try:
        from pimsr_forward.dataset2d import validate_dataset_2d
    except ImportError as exc:
        raise MaterializationError(
            "pimsr-forward is required to validate the exact PIMSR 2D schema-v2 contract"
        ) from exc
    validate_dataset_2d(path)


def _strict_axis(values: object, name: str, *, positive: bool) -> np.ndarray:
    axis = _canonical_array(values, "<f8")
    if axis.ndim != 1 or axis.size == 0 or not np.isfinite(axis).all():
        raise ValueError(f"{name} must be a non-empty finite vector")
    if positive and np.any(axis <= 0.0):
        raise ValueError(f"{name} must be positive")
    if np.any(np.diff(axis) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    return axis


def _finite_array(
    values: object,
    name: str,
    *,
    dtype: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    array = _canonical_array(values, dtype)
    if array.shape != shape:
        raise ValueError(f"{name} has shape {array.shape}, expected {shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _load_source_arrays(
    path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]]:
    with h5py.File(path, "r") as h5:
        if _text(h5.attrs.get("schema", "")) != "pimsr-mt-2d":
            raise ValueError("source HDF5 schema must be pimsr-mt-2d")
        if _integer(h5.attrs.get("schema_version", -1), "schema_version") != 2:
            raise ValueError("source HDF5 must use PIMSR 2D schema version 2")
        if _text_tuple(h5.attrs.get("mode_order", ())) != ("te", "tm"):
            raise ValueError("source HDF5 mode order must be TE then TM")
        if _text_tuple(h5.attrs.get("impedance_components", ())) != ("Zyx", "Zxy"):
            raise ValueError("source HDF5 must declare TE=Zyx and TM=Zxy")
        if _text(h5.attrs.get("phase_convention", "")) != ("degrees_modulo_180_[0,180)"):
            raise ValueError("source HDF5 phase convention is not canonical")

        frequencies = _strict_axis(h5["frequencies"][:], "frequencies", positive=True)
        station_x = _strict_axis(h5["station_x"][:], "station_x", positive=False)
        x_grid = _strict_axis(h5["x_grid"][:], "x_grid", positive=False)
        depth_grid = _strict_axis(h5["depth_grid"][:], "depth_grid", positive=True)

        sample_index = _canonical_array(h5["sample_index"][:], "<i8")
        if sample_index.ndim != 1 or sample_index.size == 0:
            raise ValueError("sample_index must be a non-empty vector")
        if np.any(sample_index < 0) or np.any(np.diff(sample_index) <= 0):
            raise ValueError("sample_index must be non-negative and strictly increasing")
        n = sample_index.size
        observation_shape = (n, frequencies.size, station_x.size)
        truth_shape = (n, depth_grid.size, x_grid.size)

        observations: dict[str, np.ndarray] = {
            "schema": _unicode_scalar(OBSERVATION_SCHEMA),
            "schema_version": _canonical_array(PAYLOAD_SCHEMA_VERSION, "<i8"),
            "sample_index": sample_index,
            "frequency_hz": frequencies,
            "station_x_m": station_x,
            "x_cell_centers_m": x_grid,
            "depth_cell_centers_m": depth_grid,
            "observation_channel_order": _unicode_vector(OBSERVATION_CHANNEL_ORDER),
        }
        for output_name, source_name in _SOURCE_OBSERVATIONS.items():
            values = _finite_array(
                h5[source_name][:],
                source_name,
                dtype="<f4",
                shape=observation_shape,
            )
            if "phase" in output_name and np.any((values < 0.0) | (values >= 180.0)):
                raise ValueError(f"{source_name} violates the [0, 180) phase convention")
            observations[output_name] = values

        scenario_indices = _canonical_array(h5["scenario"][:], "<i8")
        has_fault = _canonical_array(h5["has_fault"][:], np.dtype("bool"))
        if scenario_indices.shape != (n,) or has_fault.shape != (n,):
            raise ValueError("source labels must match the ordered sample count")
        scenario_order = _text_tuple(h5.attrs["scenario_order"])
        if not scenario_order or np.any(
            (scenario_indices < 0) | (scenario_indices >= len(scenario_order))
        ):
            raise ValueError("source scenario labels are outside scenario_order")
        scenario = _unicode_vector(
            [scenario_order[int(index)] for index in scenario_indices]
        )
        target = _finite_array(
            h5["target_log10_res"][:],
            "target_log10_res",
            dtype="<f4",
            shape=truth_shape,
        )
        truth = {
            "schema": _unicode_scalar(TRUTH_SCHEMA),
            "schema_version": _canonical_array(PAYLOAD_SCHEMA_VERSION, "<i8"),
            "sample_index": sample_index.copy(),
            "scenario": scenario,
            "has_fault": has_fault,
            "x_cell_centers_m": x_grid.copy(),
            "depth_cell_centers_m": depth_grid.copy(),
            "truth_log10_resistivity": target,
        }
        metadata: dict[str, object] = {
            "generator_seed": _integer(h5.attrs["generator_seed"], "generator_seed"),
            "generation_start_index": _integer(
                h5.attrs["generation_start_index"], "generation_start_index"
            ),
            "source_shard_count": _integer(
                h5.attrs["source_shard_count"], "source_shard_count", minimum=1
            ),
            "generation_contract": _text(h5.attrs["generation_contract"]),
            "generator_rng": _text(h5.attrs["generator_rng"]),
            "forward_contract": _text(h5.attrs["forward_contract"]),
            "sensor_contract": _text(h5.attrs["sensor_contract"]),
            "sensor_rng": _text(h5.attrs["sensor_rng"]),
            "scenario_order": scenario_order,
            "scenario_indices": scenario_indices,
            "sample_index": sample_index,
        }
        return observations, truth, metadata


def _add_declared_evaluation_floors(
    observations: dict[str, np.ndarray],
    *,
    rho_log10_floor: float,
    phase_degree_floor: float,
) -> None:
    shape = observations["observed_log10_rho_te"].shape
    for mode in ("te", "tm"):
        observations[f"declared_evaluation_floor_log10_rho_{mode}"] = np.full(
            shape, rho_log10_floor, dtype="<f4"
        )
        observations[f"declared_evaluation_floor_phase_{mode}_degrees"] = np.full(
            shape, phase_degree_floor, dtype="<f4"
        )
    observations["valid_mask"] = np.ones((shape[0], 4, *shape[1:]), dtype=bool)


def _observation_members(
    arrays: Mapping[str, np.ndarray],
) -> list[tuple[str, np.ndarray, Sequence[str]]]:
    metadata_axes: dict[str, tuple[str, ...]] = {
        "schema": (),
        "schema_version": (),
        "sample_index": ("sample",),
        "frequency_hz": ("frequency",),
        "station_x_m": ("station",),
        "x_cell_centers_m": ("x",),
        "depth_cell_centers_m": ("depth",),
        "observation_channel_order": ("channel",),
        "valid_mask": ("sample", "channel", "frequency", "station"),
    }
    order = (
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
    )
    return [
        (name, arrays[name], metadata_axes.get(name, _OBSERVATION_AXES)) for name in order
    ]


def _truth_members(
    arrays: Mapping[str, np.ndarray],
) -> list[tuple[str, np.ndarray, Sequence[str]]]:
    axes = {
        "schema": (),
        "schema_version": (),
        "sample_index": ("sample",),
        "scenario": ("sample",),
        "has_fault": ("sample",),
        "x_cell_centers_m": ("x",),
        "depth_cell_centers_m": ("depth",),
        "truth_log10_resistivity": _TRUTH_AXES,
    }
    order = (
        "schema",
        "schema_version",
        "sample_index",
        "scenario",
        "has_fault",
        "x_cell_centers_m",
        "depth_cell_centers_m",
        "truth_log10_resistivity",
    )
    return [(name, arrays[name], axes[name]) for name in order]


def _sample_groups(metadata: Mapping[str, object]) -> list[dict[str, object]]:
    generator_seed = int(metadata["generator_seed"])
    source_indices = np.asarray(metadata["sample_index"], dtype=np.int64)
    opaque_indices = np.asarray(metadata["opaque_sample_index"], dtype=np.int64)
    scenarios = np.asarray(metadata["scenario_indices"], dtype=np.int64)
    groups: list[dict[str, object]] = []
    rows = zip(source_indices, opaque_indices, scenarios, strict=True)
    for source_index, opaque_index, scenario_index in rows:
        groups.append(
            {
                "base_model_id": (f"geology-g{generator_seed}-i{int(source_index)}"),
                "family_id": f"scenario-{int(scenario_index)}",
                "noise_id": f"sensor-g{generator_seed}-i{int(source_index)}",
                "sample_ids": [f"sample-{int(opaque_index)}"],
            }
        )
    return groups


def _sample_id_mapping(metadata: Mapping[str, object]) -> list[dict[str, int]]:
    source_indices = np.asarray(metadata["sample_index"], dtype=np.int64)
    opaque_indices = np.asarray(metadata["opaque_sample_index"], dtype=np.int64)
    return [
        {
            "opaque_sample_index": int(opaque),
            "source_generator_sample_index": int(source),
        }
        for source, opaque in zip(source_indices, opaque_indices, strict=True)
    ]


def _scenario_groups(metadata: Mapping[str, object]) -> list[dict[str, object]]:
    scenario_order = tuple(str(value) for value in metadata["scenario_order"])
    scenarios = np.asarray(metadata["scenario_indices"], dtype=np.int64)
    opaque = np.asarray(metadata["opaque_sample_index"], dtype=np.int64)
    return [
        {
            "opaque_sample_indices": [
                int(sample_id)
                for sample_id, sample_scenario in zip(opaque, scenarios, strict=True)
                if int(sample_scenario) == scenario_index
            ],
            "scenario": scenario,
            "scenario_index": scenario_index,
        }
        for scenario_index, scenario in enumerate(scenario_order)
        if np.any(scenarios == scenario_index)
    ]


def _physical_contract() -> dict[str, object]:
    return {
        "axes": ["x", "z"],
        "axis_units": {"x": "m", "z": "m"},
        "coordinate_system": "pimsr_2d_profile_cartesian",
        "dataset_schema": "pimsr-mt-2d",
        "dataset_schema_version": 2,
        "dimensionality": "2d",
        "handedness": "right_handed",
        "mode_component_mapping": {"TE": "Zyx", "TM": "Zxy"},
        "model_parameter": "log10_resistivity",
        "model_parameter_unit": "log10_ohm_m",
        "observation_axis_order": list(_OBSERVATION_AXES),
        "observation_channel_order": list(OBSERVATION_CHANNEL_ORDER),
        "phase_convention": "degrees_modulo_180_[0,180)",
        "phase_unit": "degree",
        "resistivity_unit": "ohm_m",
        "rotation_degrees": 0.0,
        "spectral_axis": "frequency",
        "spectral_order": "strictly_increasing",
        "spectral_unit": "Hz",
        "time_convention": "exp(+i_omega_t)",
        "truth_axis_order": list(_TRUTH_AXES),
        "vertical_positive": "down",
    }


def _payload_record(
    path: Path,
    *,
    schema: str,
    digest: str,
    size: int,
    arrays: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    return {
        "arrays": dict(arrays),
        "file_name": path.name,
        "media_type": "application/x-npz",
        "schema": schema,
        "schema_version": PAYLOAD_SCHEMA_VERSION,
        "sha256": digest,
        "size_bytes": size,
    }


def _public_payload_record(
    *,
    digest: str,
    size: int,
    arrays: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    return {
        "arrays": dict(arrays),
        "media_type": "application/x-npz",
        "schema": OBSERVATION_SCHEMA,
        "schema_version": PAYLOAD_SCHEMA_VERSION,
        "sha256": digest,
        "size_bytes": size,
    }


def _declared_evaluation_floors(
    *,
    rho_log10_floor: float,
    phase_degree_floor: float,
) -> dict[str, object]:
    return {
        "interpretation": (
            "declared evaluation floors, not empirical or source-noise standard "
            "deviations"
        ),
        "log10_rho_floor": rho_log10_floor,
        "phase_degree_floor": phase_degree_floor,
        "policy_id": UNCERTAINTY_POLICY_ID,
        "storage": "explicit_full_shape_float32_arrays_in_observations_payload",
        "validity": "explicit_all_true_boolean_mask",
    }


def _public_manifest(
    *,
    observations_sha256: str,
    observations_size: int,
    observation_records: Mapping[str, Mapping[str, object]],
    split_id: str,
    sample_count: int,
    rho_log10_floor: float,
    phase_degree_floor: float,
) -> dict[str, object]:
    """Build the exact sanitized manifest delivered to external methods."""
    return {
        "audience": "method_input_public",
        "declared_evaluation_floors": _declared_evaluation_floors(
            rho_log10_floor=rho_log10_floor,
            phase_degree_floor=phase_degree_floor,
        ),
        "observation_payload": _public_payload_record(
            digest=observations_sha256,
            size=observations_size,
            arrays=observation_records,
        ),
        "physical_contract": _physical_contract(),
        "sample_count": sample_count,
        "schema": PUBLIC_MANIFEST_SCHEMA,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "split_id": split_id,
    }


def _operator_manifest(
    *,
    source: Path,
    source_sha256: str,
    source_size: int,
    observations_path: Path,
    observations_sha256: str,
    observations_size: int,
    observation_records: Mapping[str, Mapping[str, object]],
    truth_path: Path,
    truth_sha256: str,
    truth_size: int,
    truth_records: Mapping[str, Mapping[str, object]],
    public_manifest_path: Path,
    public_manifest_sha256: str,
    public_manifest_size: int,
    metadata: Mapping[str, object],
    split_id: str,
) -> dict[str, object]:
    return {
        "artifacts": {
            "observations": _payload_record(
                observations_path,
                schema=OBSERVATION_SCHEMA,
                digest=observations_sha256,
                size=observations_size,
                arrays=observation_records,
            ),
            "withheld_truth": _payload_record(
                truth_path,
                schema=TRUTH_SCHEMA,
                digest=truth_sha256,
                size=truth_size,
                arrays=truth_records,
            ),
            "public_observation_manifest": {
                "file_name": public_manifest_path.name,
                "media_type": "application/json",
                "schema": PUBLIC_MANIFEST_SCHEMA,
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "sha256": public_manifest_sha256,
                "size_bytes": public_manifest_size,
            },
        },
        "audience": "benchmark_operator_only",
        "schema": OPERATOR_MANIFEST_SCHEMA,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source": {
            "dataset_schema": "pimsr-mt-2d",
            "dataset_schema_version": 2,
            "file_name": source.name,
            "forward_contract": metadata["forward_contract"],
            "generation_contract": metadata["generation_contract"],
            "generation_start_index": metadata["generation_start_index"],
            "generator_rng": metadata["generator_rng"],
            "generator_seed": metadata["generator_seed"],
            "media_type": "application/x-hdf5",
            "sensor_contract": metadata["sensor_contract"],
            "sensor_rng": metadata["sensor_rng"],
            "sha256": source_sha256,
            "size_bytes": source_size,
            "source_shard_count": metadata["source_shard_count"],
        },
        "split": {
            "groups": _sample_groups(metadata),
            "opaque_sample_id_contract": {
                "algorithm": "HMAC-SHA256",
                "digest_projection": "first_64_bits_big_endian_clear_sign_bit",
                "key_material": "external_secret_not_recorded",
                "message": (
                    "domain_separator || generator_seed_uint64_be || "
                    "source_sample_index_uint64_be || split_id_length_uint32_be || "
                    "split_id_ascii"
                ),
                "version": 1,
            },
            "sample_id_mapping": _sample_id_mapping(metadata),
            "sample_count": int(np.asarray(metadata["sample_index"]).size),
            "scenario_groups": _scenario_groups(metadata),
            "payload_row_order": "strictly_increasing_source_sample_index",
            "split_id": split_id,
        },
    }


def _prepare_outputs(
    source: Path,
    outputs: Sequence[Path],
) -> tuple[list[Path], list[Path]]:
    requested_parents = [output.parent.resolve(strict=False) for output in outputs]
    if len(set(requested_parents)) != 1:
        raise ValueError(
            "all four materialization outputs must share one parent directory"
        )
    common_parent = requested_parents[0]
    common_parent.mkdir(parents=True, exist_ok=True)
    common_parent = common_parent.resolve(strict=True)

    resolved = [(common_parent / output.name).resolve(strict=False) for output in outputs]
    parts = [
        destination.with_name(destination.name + ".part") for destination in resolved
    ]
    source_resolved = source.resolve(strict=True)
    for destination in resolved:
        if destination == source_resolved:
            raise ValueError("materialization output cannot replace the source HDF5")
    if len(set(resolved)) != len(resolved):
        raise ValueError("materialization output paths must be distinct")
    for destination, part in zip(resolved, parts, strict=True):
        if destination.exists():
            raise FileExistsError(
                f"refusing to overwrite materialization output: {destination}"
            )
        if part.exists():
            raise FileExistsError(f"refusing stale partial output: {part}")
    return resolved, parts


def _verify_npz(
    path: Path,
    expected: Mapping[str, Mapping[str, object]],
) -> None:
    with zipfile.ZipFile(path, "r") as archive:
        names = archive.namelist()
        expected_names = [f"{name}.npy" for name in expected]
        if names != expected_names:
            raise MaterializationError(f"NPZ member order or names changed: {path}")
        for name, record in expected.items():
            payload = archive.read(f"{name}.npy")
            if hashlib.sha256(payload).hexdigest() != record["sha256"]:
                raise MaterializationError(f"NPZ array digest mismatch: {path}:{name}")
            array = np.lib.format.read_array(io.BytesIO(payload), allow_pickle=False)
            if array.dtype.str != record["dtype"] or list(array.shape) != record["shape"]:
                raise MaterializationError(f"NPZ array contract mismatch: {path}:{name}")


def _publish_parts(
    parts: Sequence[Path],
    destinations: Sequence[Path],
) -> None:
    published: list[tuple[Path, tuple[int, int]]] = []
    try:
        for part, destination in zip(parts, destinations, strict=True):
            part_info = part.stat()
            expected_identity = (part_info.st_dev, part_info.st_ino)
            try:
                os.link(part, destination)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"publication race: refusing to overwrite {destination}"
                ) from exc
            published.append((destination, expected_identity))
            destination_info = destination.stat(follow_symlinks=False)
            if (
                destination.is_symlink()
                or not stat.S_ISREG(destination_info.st_mode)
                or (destination_info.st_dev, destination_info.st_ino) != expected_identity
            ):
                raise MaterializationError(
                    f"published output file identity mismatch: {destination}"
                )
    except BaseException as exc:
        unsafe: list[str] = []
        for destination, expected_identity in reversed(published):
            if not os.path.lexists(destination):
                continue
            current = destination.stat(follow_symlinks=False)
            current_identity = (current.st_dev, current.st_ino)
            if (
                destination.is_symlink()
                or not stat.S_ISREG(current.st_mode)
                or current_identity != expected_identity
            ):
                unsafe.append(str(destination))
                continue
            destination.unlink()
        if unsafe:
            raise MaterializationError(
                "refusing to delete outputs replaced during rollback: "
                + ", ".join(unsafe)
            ) from exc
        raise


def materialize_dataset2d(
    source_h5: str | Path,
    observations_npz: str | Path,
    truth_npz: str | Path,
    public_manifest_json: str | Path,
    operator_manifest_json: str | Path,
    *,
    split_id: str,
    sample_id_key: bytes | bytearray | memoryview | str | Path,
    rho_log10_floor: float = DEFAULT_RHO_LOG10_FLOOR,
    phase_degree_floor: float = DEFAULT_PHASE_DEGREE_FLOOR,
) -> MaterializationResult:
    """Publish a four-artifact, access-separated schema-v2 evaluation set.

    Every destination is new-only.  Source mutation, ambiguous physical
    conventions, non-finite data, unordered sample indices, invalid phases,
    stale partial outputs and publication races all fail closed.  The secret
    ``sample_id_key`` must be supplied out of band and is never persisted or
    hashed into an output.
    """
    source = Path(source_h5).resolve(strict=True)
    if source.suffix.lower() not in {".h5", ".hdf5"}:
        raise ValueError("source_h5 must use an .h5 or .hdf5 suffix")
    requested_outputs = [
        Path(observations_npz),
        Path(truth_npz),
        Path(public_manifest_json),
        Path(operator_manifest_json),
    ]
    if requested_outputs[0].suffix.lower() != ".npz":
        raise ValueError("observations_npz must use an .npz suffix")
    if requested_outputs[1].suffix.lower() != ".npz":
        raise ValueError("truth_npz must use an .npz suffix")
    if requested_outputs[2].suffix.lower() != ".json":
        raise ValueError("public_manifest_json must use a .json suffix")
    if requested_outputs[3].suffix.lower() != ".json":
        raise ValueError("operator_manifest_json must use a .json suffix")
    split = _split_id(split_id)
    key = _load_sample_id_key(sample_id_key)
    rho_floor = _positive_float32(rho_log10_floor, "rho_log10_floor")
    phase_floor = _positive_float32(phase_degree_floor, "phase_degree_floor")
    destinations, parts = _prepare_outputs(source, requested_outputs)

    source_sha256, source_size = _stable_file_identity(source, role="source HDF5")
    _source_contract(source)
    observations, truth, metadata = _load_source_arrays(source)
    final_source_sha256, final_source_size = _stable_file_identity(
        source, role="source HDF5"
    )
    if (final_source_sha256, final_source_size) != (source_sha256, source_size):
        raise MaterializationError("source HDF5 changed during materialization")
    opaque_indices = _opaque_sample_indices(
        np.asarray(metadata["sample_index"], dtype="<i8"),
        generator_seed=int(metadata["generator_seed"]),
        split_id=split,
        key=key,
    )
    observations["sample_index"] = opaque_indices.copy()
    truth["sample_index"] = opaque_indices.copy()
    metadata["opaque_sample_index"] = opaque_indices
    _add_declared_evaluation_floors(
        observations,
        rho_log10_floor=rho_floor,
        phase_degree_floor=phase_floor,
    )

    observation_records: dict[str, dict[str, object]]
    truth_records: dict[str, dict[str, object]]
    try:
        observation_records = _write_deterministic_npz(
            parts[0], _observation_members(observations)
        )
        truth_records = _write_deterministic_npz(parts[1], _truth_members(truth))
        _verify_npz(parts[0], observation_records)
        _verify_npz(parts[1], truth_records)
        observations_sha256, observations_size = _sha256_file(parts[0])
        truth_sha256, truth_size = _sha256_file(parts[1])
        public_manifest = _public_manifest(
            observations_sha256=observations_sha256,
            observations_size=observations_size,
            observation_records=observation_records,
            split_id=split,
            sample_count=int(opaque_indices.size),
            rho_log10_floor=rho_floor,
            phase_degree_floor=phase_floor,
        )
        public_manifest_payload = _canonical_json_bytes(public_manifest)
        _write_bytes(parts[2], public_manifest_payload)
        if parts[2].read_bytes() != public_manifest_payload:
            raise MaterializationError("public canonical manifest verification failed")
        public_manifest_sha256, public_manifest_size = _sha256_file(parts[2])
        operator_manifest = _operator_manifest(
            source=source,
            source_sha256=source_sha256,
            source_size=source_size,
            observations_path=destinations[0],
            observations_sha256=observations_sha256,
            observations_size=observations_size,
            observation_records=observation_records,
            truth_path=destinations[1],
            truth_sha256=truth_sha256,
            truth_size=truth_size,
            truth_records=truth_records,
            public_manifest_path=destinations[2],
            public_manifest_sha256=public_manifest_sha256,
            public_manifest_size=public_manifest_size,
            metadata=metadata,
            split_id=split,
        )
        operator_manifest_payload = _canonical_json_bytes(operator_manifest)
        _write_bytes(parts[3], operator_manifest_payload)
        if parts[3].read_bytes() != operator_manifest_payload:
            raise MaterializationError("operator canonical manifest verification failed")
        _publish_parts(parts, destinations)
    finally:
        for part in parts:
            part.unlink(missing_ok=True)

    operator_manifest_sha256, _ = _sha256_file(destinations[3])
    return MaterializationResult(
        observations_path=destinations[0],
        truth_path=destinations[1],
        public_manifest_path=destinations[2],
        operator_manifest_path=destinations[3],
        observations_sha256=observations_sha256,
        truth_sha256=truth_sha256,
        public_manifest_sha256=public_manifest_sha256,
        operator_manifest_sha256=operator_manifest_sha256,
    )
