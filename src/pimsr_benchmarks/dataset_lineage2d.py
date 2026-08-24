"""Fail-closed lineage evidence for existing public PIMSR 2-D datasets.

This module does not generate geological models and does not run an MT
forward solve.  It attests the byte identities of an already merged HDF5
dataset, its ordered source shards and logs, and the clean source repositories
whose identities are supplied by the caller.  It then proves, by chunked exact
comparison, that the merged logical HDF5 arrays equal the ordered shard data.

The resulting evidence deliberately has the narrower scope
``artifact_lineage_and_source_identity_without_forward_regeneration``.  A
matching artifact and clean source tree are not proof that those exact sources
were executing when the artifacts were originally generated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import h5py
import numpy as np

LINEAGE_SCHEMA = "pimsr-public-dataset-lineage-2d"
LINEAGE_SCHEMA_VERSION = 1
SHARD_PINS_SCHEMA = "pimsr-public-dataset-shard-pins-2d"
SHARD_PINS_SCHEMA_VERSION = 1
EVIDENCE_SCOPE = "artifact_lineage_and_source_identity_without_forward_regeneration"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHARD_RE = re.compile(r"^shard-(?P<start>[0-9]{6})-(?P<end>[0-9]{6})\.h5$")
_CHUNK_ROWS = 100

_ROW_KEYS = (
    "obs_mt_log10_rho",
    "obs_mt_phase",
    "clean_mt_log10_rho",
    "clean_mt_phase",
    "target_log10_res",
    "scenario",
    "has_fault",
    "obs_mt_log10_rho_tm",
    "obs_mt_phase_tm",
    "clean_mt_log10_rho_tm",
    "clean_mt_phase_tm",
    "sample_index",
)
_META_KEYS = ("frequencies", "station_x", "x_grid", "depth_grid")
_DATASET_KEYS = _ROW_KEYS + _META_KEYS
_OBSERVATION_KEYS = (
    "obs_mt_log10_rho",
    "obs_mt_phase",
    "clean_mt_log10_rho",
    "clean_mt_phase",
    "obs_mt_log10_rho_tm",
    "obs_mt_phase_tm",
    "clean_mt_log10_rho_tm",
    "clean_mt_phase_tm",
)
_PHASE_KEYS = tuple(name for name in _OBSERVATION_KEYS if "phase" in name)

_ROOT_STRING_ATTRS = {
    "schema": "pimsr-mt-2d",
    "phase_convention": "degrees_modulo_180_[0,180)",
    "resistivity_representation": "log10_ohm_m",
    "frequencies_unit": "Hz",
    "station_x_unit": "m",
    "x_grid_unit": "m",
    "depth_grid_unit": "m",
    "phase_unit": "degree",
    "generation_contract": "pimsr-geogen.SectionGenerator/default-grid/v1",
    "generator_rng": "numpy.default_rng([generator_seed,2,sample_index])",
    "forward_contract": "pimsr-forward.MT2DForward/default-mesh/v2",
    "sensor_contract": "pimsr-forward.SensorModel/mt-noise+tm-severity-v5/v1",
    "sensor_rng": "numpy.default_rng([generator_seed,3,sample_index])",
}
_INTEGER_ROOT_ATTRS = (
    "schema_version",
    "generator_seed",
    "generation_start_index",
    "expected_row_count",
    "source_shard_count",
    "generation_complete",
)
_ROOT_ATTRS = frozenset(
    (*_ROOT_STRING_ATTRS, *_INTEGER_ROOT_ATTRS)
    + (
        "mode_order",
        "impedance_components",
        "scenario_order",
        "sensor_parameters_json",
        "software_versions_json",
    )
)
_SCENARIOS = ("background", "aquifer", "hydrocarbon", "salt", "geothermal")
_SOFTWARE_VERSION_KEYS = {
    "discretize",
    "h5py",
    "numpy",
    "pimsr_forward",
    "pimsr_geogen",
    "simpeg",
}
_DEFAULT_SENSOR_PARAMETERS = {
    "application_order": "station_major_te_then_tm",
    "sensor_model": {
        "distort_lag1": 0.46,
        "distort_log10rho_hi": 0.25,
        "distort_log10rho_lo": 0.02,
        "distort_phase_scale": 40.0,
        "grav_drift_mgal": 0.05,
        "grav_white_mgal": 0.03,
        "mt_dead_band_extra": 0.02,
        "mt_phase_floor_deg": 1.0,
        "mt_rel_floor": 0.03,
        "static_shift_sigma": 0.15,
    },
    "te_overrides": None,
    "tm_overrides": {
        "distort_hi": {"distribution": "log_uniform", "high": 0.45, "low": 0.25},
        "shift_sigma": {"distribution": "uniform", "high": 0.32, "low": 0.15},
    },
}
_SOURCE_FILES = {
    "pimsr_forward": (
        "src/pimsr_forward/dataset2d.py",
        "src/pimsr_forward/mt2d.py",
        "src/pimsr_forward/sensors.py",
    ),
    "pimsr_geogen": ("src/pimsr_geogen/section2d.py",),
}


class DatasetLineageError(RuntimeError):
    """Raised when a lineage input or publication violates the contract."""


@dataclass(frozen=True)
class DatasetLineageResult:
    """Identity of one immutable, canonical lineage publication."""

    path: Path
    sha256: str
    size_bytes: int
    manifest: dict[str, object]


@dataclass(frozen=True)
class _FileSnapshot:
    path: Path
    sha256: str
    size_bytes: int
    signature: tuple[int, int, int, int]
    payload: bytes | None = None
    _stream: BinaryIO | None = None

    def close(self) -> None:
        """Close an optional descriptor retained for stable structured reads."""
        if self._stream is not None:
            self._stream.close()

    @contextmanager
    def open_hdf5(self) -> Iterator[h5py.File]:
        """Open HDF5 through the pinned descriptor, never through its pathname."""
        if self._stream is None or self._stream.closed:
            raise DatasetLineageError(
                f"stable descriptor is unavailable for HDF5 snapshot: {self.path}"
            )
        try:
            duplicate = os.fdopen(os.dup(self._stream.fileno()), "rb")
        except OSError as exc:
            raise DatasetLineageError(
                f"cannot duplicate pinned HDF5 descriptor: {self.path}"
            ) from exc
        try:
            with h5py.File(duplicate, "r") as h5:
                yield h5
        finally:
            duplicate.close()

    def verify_unchanged(self, *, role: str) -> None:
        current = _snapshot_file(self.path, role=role)
        if (
            current.signature != self.signature
            or current.sha256 != self.sha256
            or current.size_bytes != self.size_bytes
        ):
            raise DatasetLineageError(f"{role} was replaced or changed: {self.path}")


@dataclass(frozen=True)
class _DirectorySnapshot:
    path: Path
    entries: tuple[str, ...]
    signature: tuple[int, int, int, int]

    def verify_unchanged(self, *, role: str) -> None:
        current = _snapshot_directory(self.path, role=role)
        if current.entries != self.entries or current.signature != self.signature:
            raise DatasetLineageError(f"{role} was replaced or changed: {self.path}")


@dataclass(frozen=True)
class _DatasetContract:
    generator_seed: int
    start_index: int
    sample_count: int
    source_shard_count: int
    root_attributes: dict[str, object]


@dataclass(frozen=True)
class _ShardPin:
    hdf5_filename: str
    hdf5_sha256: str
    hdf5_size_bytes: int
    log_filename: str
    log_sha256: str
    log_size_bytes: int


def _file_signature(value: os.stat_result) -> tuple[int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
    )


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_commit(value: object, name: str) -> str:
    if not isinstance(value, str) or not _COMMIT_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase 40-character Git commit")
    return value


def _require_nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _require_positive_integer(value: object, name: str) -> int:
    result = _require_nonnegative_integer(value, name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _sha256_stream(stream: Any) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
        size += len(block)
    return digest.hexdigest(), size


def _snapshot_file(
    path: str | Path,
    *,
    role: str,
    capture_payload: bool = False,
    keep_open: bool = False,
) -> _FileSnapshot:
    requested = Path(path).absolute()
    try:
        before = requested.lstat()
    except OSError as exc:
        raise ValueError(f"cannot inspect {role}: {requested}") from exc
    if (
        stat.S_ISLNK(before.st_mode)
        or requested.is_symlink()
        or not stat.S_ISREG(before.st_mode)
    ):
        raise ValueError(f"{role} must be a regular non-symlink file: {requested}")
    stream: BinaryIO | None = None
    try:
        stream = requested.open("rb")
        opened_before = os.fstat(stream.fileno())
        digest, size = _sha256_stream(stream)
        payload = None
        if capture_payload:
            stream.seek(0)
            payload = stream.read()
            if len(payload) != size or hashlib.sha256(payload).hexdigest() != digest:
                raise DatasetLineageError(
                    f"{role} changed while its payload was captured: {requested}"
                )
        opened_after = os.fstat(stream.fileno())
        after = requested.lstat()
    except OSError as exc:
        if stream is not None:
            stream.close()
        raise DatasetLineageError(f"cannot read {role}: {requested}") from exc
    except BaseException:
        if stream is not None:
            stream.close()
        raise
    signatures = {
        _file_signature(before),
        _file_signature(opened_before),
        _file_signature(opened_after),
        _file_signature(after),
    }
    if len(signatures) != 1 or size != before.st_size:
        stream.close()
        raise DatasetLineageError(f"{role} changed while it was hashed: {requested}")
    retained = stream if keep_open else None
    if not keep_open:
        stream.close()
    return _FileSnapshot(
        requested,
        digest,
        size,
        _file_signature(after),
        payload,
        retained,
    )


def _snapshot_pinned_file(
    path: str | Path,
    *,
    role: str,
    expected_sha256: str,
    expected_size_bytes: int | None = None,
    capture_payload: bool = False,
    keep_open: bool = False,
) -> _FileSnapshot:
    expected = _require_sha256(expected_sha256, f"expected {role} SHA-256")
    if expected_size_bytes is not None:
        expected_size_bytes = _require_positive_integer(
            expected_size_bytes, f"expected {role} size"
        )
    snapshot = _snapshot_file(
        path,
        role=role,
        capture_payload=capture_payload,
        keep_open=keep_open,
    )
    if snapshot.sha256 != expected:
        snapshot.close()
        raise DatasetLineageError(
            f"{role} SHA-256 mismatch: expected {expected}, got {snapshot.sha256}"
        )
    if expected_size_bytes is not None and snapshot.size_bytes != expected_size_bytes:
        snapshot.close()
        raise DatasetLineageError(
            f"{role} size mismatch: expected {expected_size_bytes}, "
            f"got {snapshot.size_bytes}"
        )
    return snapshot


def _snapshot_directory(path: str | Path, *, role: str) -> _DirectorySnapshot:
    requested = Path(path).absolute()
    try:
        before = requested.lstat()
        entries = tuple(sorted(entry.name for entry in requested.iterdir()))
        after = requested.lstat()
    except OSError as exc:
        raise ValueError(f"cannot inspect {role}: {requested}") from exc
    if (
        stat.S_ISLNK(before.st_mode)
        or requested.is_symlink()
        or not stat.S_ISDIR(before.st_mode)
    ):
        raise ValueError(f"{role} must be a non-symlink directory: {requested}")
    if _file_signature(before) != _file_signature(after):
        raise DatasetLineageError(f"{role} changed while it was listed: {requested}")
    return _DirectorySnapshot(requested, entries, _file_signature(after))


def _canonical_json_bytes(value: object) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise DatasetLineageError(f"lineage is not canonical JSON data: {exc}") from exc
    return (text + "\n").encode("utf-8")


def _load_shard_pins(snapshot: _FileSnapshot) -> tuple[_ShardPin, ...]:
    raw = snapshot.payload
    if raw is None:
        raise DatasetLineageError("shard pin manifest snapshot has no captured payload")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("shard pin manifest must be UTF-8 JSON") from exc
    if raw != _canonical_json_bytes(value):
        raise ValueError("shard pin manifest must use canonical JSON encoding")
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "schema_version",
        "shards",
    }:
        raise ValueError("shard pin manifest has invalid root keys")
    if value["schema"] != SHARD_PINS_SCHEMA or value["schema_version"] != 1:
        raise ValueError("unsupported shard pin manifest schema")
    records = value["shards"]
    if not isinstance(records, list) or not records:
        raise ValueError("shard pin manifest must contain a non-empty shards list")
    pins: list[_ShardPin] = []
    names: set[str] = set()
    keys = {
        "hdf5_filename",
        "hdf5_sha256",
        "hdf5_size_bytes",
        "log_filename",
        "log_sha256",
        "log_size_bytes",
    }
    for ordinal, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != keys:
            raise ValueError(f"invalid shard pin record at ordinal {ordinal}")
        hdf5_name = record["hdf5_filename"]
        log_name = record["log_filename"]
        if (
            not isinstance(hdf5_name, str)
            or Path(hdf5_name).name != hdf5_name
            or _SHARD_RE.fullmatch(hdf5_name) is None
        ):
            raise ValueError(f"invalid HDF5 shard filename at ordinal {ordinal}")
        if not isinstance(log_name, str) or log_name != f"{hdf5_name}.log":
            raise ValueError(f"invalid matching log filename at ordinal {ordinal}")
        if hdf5_name in names or log_name in names:
            raise ValueError("shard pin manifest contains duplicate filenames")
        names.update((hdf5_name, log_name))
        pins.append(
            _ShardPin(
                hdf5_filename=hdf5_name,
                hdf5_sha256=_require_sha256(
                    record["hdf5_sha256"], f"shard {ordinal} HDF5 SHA-256"
                ),
                hdf5_size_bytes=_require_positive_integer(
                    record["hdf5_size_bytes"], f"shard {ordinal} HDF5 size"
                ),
                log_filename=log_name,
                log_sha256=_require_sha256(
                    record["log_sha256"], f"shard {ordinal} log SHA-256"
                ),
                log_size_bytes=_require_positive_integer(
                    record["log_size_bytes"], f"shard {ordinal} log size"
                ),
            )
        )
    return tuple(pins)


def _text(value: object) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    return str(value)


def _text_tuple(value: object) -> tuple[str, ...]:
    return tuple(_text(item) for item in np.asarray(value).reshape(-1))


def _require_text_vector(value: object, expected: tuple[str, ...], *, name: str) -> None:
    array = np.asarray(value)
    if array.ndim != 1 or array.shape != (len(expected),):
        raise ValueError(f"invalid PIMSR 2D {name} shape")
    if _text_tuple(array) != expected:
        raise ValueError(f"invalid PIMSR 2D {name}")


def _json_value(value: object) -> object:
    array = np.asarray(value)
    if array.ndim == 0:
        item = array.item()
        if isinstance(item, bytes):
            return item.decode("utf-8")
        if isinstance(item, (np.integer, int)) and not isinstance(item, bool):
            return int(item)
        if isinstance(item, (np.floating, float)):
            result = float(item)
            if not np.isfinite(result):
                raise ValueError("HDF5 metadata contains a non-finite scalar")
            return result
        if isinstance(item, (np.bool_, bool)):
            return bool(item)
        if isinstance(item, str):
            return item
        raise ValueError(f"unsupported HDF5 metadata scalar: {type(item).__name__}")
    return [_json_value(item) for item in array.reshape(-1)]


def _integer_attr(h5: h5py.File, key: str) -> int:
    try:
        value = np.asarray(h5.attrs[key])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid PIMSR 2D generation attribute: {key}") from exc
    if value.ndim != 0 or value.dtype.kind not in "iu":
        raise ValueError(f"invalid PIMSR 2D generation attribute: {key}")
    return int(value)


def _require_canonical_json_attr(value: object, *, name: str) -> object:
    if np.asarray(value).ndim != 0:
        raise ValueError(f"invalid PIMSR 2D {name}")
    text = _text(value)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid PIMSR 2D {name}") from exc
    canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
    if text != canonical:
        raise ValueError(f"PIMSR 2D {name} must be canonical JSON")
    return parsed


def _iter_dataset_chunks(dataset: h5py.Dataset, chunk_rows: int = _CHUNK_ROWS):
    for start in range(0, dataset.shape[0], chunk_rows):
        end = min(start + chunk_rows, dataset.shape[0])
        yield start, end, np.asarray(dataset[start:end])


def _validate_dataset_attrs(h5: h5py.File) -> None:
    expected: dict[str, dict[str, object]] = {
        "frequencies": {"unit": "Hz"},
        "station_x": {"unit": "m"},
        "x_grid": {"unit": "m"},
        "depth_grid": {"unit": "m"},
        "target_log10_res": {"unit": "log10_ohm_m"},
        "scenario": {"labels": list(_SCENARIOS)},
        "sample_index": {"role": "generator_sample_index"},
        "has_fault": {},
    }
    for mode, component, suffix in (("TE", "Zyx", ""), ("TM", "Zxy", "_tm")):
        for stem in (
            "obs_mt_log10_rho",
            "obs_mt_phase",
            "clean_mt_log10_rho",
            "clean_mt_phase",
        ):
            name = stem + suffix
            expected[name] = {
                "impedance_component": component,
                "mode": mode,
                "unit": "degree" if "phase" in name else "log10_ohm_m",
            }
    for key in _DATASET_KEYS:
        actual = {name: _json_value(value) for name, value in h5[key].attrs.items()}
        if actual != expected[key]:
            raise ValueError(f"invalid PIMSR 2D dataset metadata: {key}")
    labels = np.asarray(h5["scenario"].attrs["labels"])
    if labels.ndim != 1 or labels.shape != (len(_SCENARIOS),):
        raise ValueError("invalid PIMSR 2D scenario labels metadata shape")


def _validate_open_dataset(
    h5: h5py.File,
    *,
    role: str,
    expected_generator_seed: int,
) -> _DatasetContract:
    if set(h5.attrs) != _ROOT_ATTRS:
        raise ValueError(f"{role} has invalid PIMSR 2D root attributes")
    for key, expected in _ROOT_STRING_ATTRS.items():
        if _text(h5.attrs[key]) != expected:
            raise ValueError(f"{role} has unsupported PIMSR 2D contract: {key}")
    integers = {key: _integer_attr(h5, key) for key in _INTEGER_ROOT_ATTRS}
    if integers["schema_version"] != 2:
        raise ValueError(f"{role} must use PIMSR 2D schema version 2")
    if integers["generation_complete"] != 1:
        raise ValueError(f"{role} is not marked generation_complete")
    if integers["generator_seed"] != expected_generator_seed:
        raise ValueError(f"{role} generator_seed does not match the external pin")
    if integers["generation_start_index"] < 0:
        raise ValueError(f"{role} generation_start_index must be non-negative")
    if integers["source_shard_count"] < 1:
        raise ValueError(f"{role} source_shard_count must be positive")
    try:
        _require_text_vector(h5.attrs["mode_order"], ("te", "tm"), name="mode order")
        _require_text_vector(
            h5.attrs["impedance_components"],
            ("Zyx", "Zxy"),
            name="impedance components",
        )
        _require_text_vector(
            h5.attrs["scenario_order"], _SCENARIOS, name="scenario order"
        )
    except ValueError as exc:
        raise ValueError(f"{role} has {exc}") from exc
    sensor_parameters = _require_canonical_json_attr(
        h5.attrs["sensor_parameters_json"], name="sensor parameters"
    )
    if sensor_parameters != _DEFAULT_SENSOR_PARAMETERS:
        raise ValueError(f"{role} has unsupported PIMSR 2D sensor parameters")
    software_versions = _require_canonical_json_attr(
        h5.attrs["software_versions_json"], name="software versions"
    )
    if (
        not isinstance(software_versions, dict)
        or set(software_versions) != _SOFTWARE_VERSION_KEYS
        or any(
            not isinstance(software_versions[key], str) or not software_versions[key]
            for key in _SOFTWARE_VERSION_KEYS
        )
    ):
        raise ValueError(f"{role} has invalid PIMSR 2D software versions")
    if set(h5) != set(_DATASET_KEYS):
        raise ValueError(f"{role} has invalid PIMSR 2D dataset members")

    expected_dtypes = {
        **{key: np.dtype("float32") for key in _OBSERVATION_KEYS},
        "target_log10_res": np.dtype("float32"),
        "scenario": np.dtype("int32"),
        "has_fault": np.dtype("uint8"),
        "sample_index": np.dtype("int64"),
        **{key: np.dtype("float64") for key in _META_KEYS},
    }
    for key, expected_dtype in expected_dtypes.items():
        if h5[key].dtype != expected_dtype:
            raise ValueError(f"{role} has invalid PIMSR 2D dtype: {key}")
    _validate_dataset_attrs(h5)

    coordinates = {key: np.asarray(h5[key][:]) for key in _META_KEYS}
    for key, values in coordinates.items():
        if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
            raise ValueError(f"{role} has invalid PIMSR 2D coordinate: {key}")
        if np.any(np.diff(values) <= 0):
            raise ValueError(f"{role} coordinate is not strictly increasing: {key}")
    if np.any(coordinates["frequencies"] <= 0) or np.any(coordinates["depth_grid"] <= 0):
        raise ValueError(f"{role} frequencies/depth must be positive")
    if (
        coordinates["station_x"][0] < coordinates["x_grid"][0]
        or coordinates["station_x"][-1] > coordinates["x_grid"][-1]
    ):
        raise ValueError(f"{role} stations must lie inside x_grid")

    n = h5["sample_index"].shape[0]
    if n == 0 or integers["expected_row_count"] != n:
        raise ValueError(f"{role} expected_row_count does not match stored rows")
    obs_shape = (n, len(coordinates["frequencies"]), len(coordinates["station_x"]))
    truth_shape = (n, len(coordinates["depth_grid"]), len(coordinates["x_grid"]))
    for key in _OBSERVATION_KEYS:
        if h5[key].shape != obs_shape:
            raise ValueError(f"{role} dimensions do not match coordinates: {key}")
    if h5["target_log10_res"].shape != truth_shape:
        raise ValueError(f"{role} target dimensions do not match coordinates")
    for key in ("scenario", "has_fault", "sample_index"):
        if h5[key].shape != (n,):
            raise ValueError(f"{role} row count mismatch: {key}")

    for key in (*_OBSERVATION_KEYS, "target_log10_res"):
        for _start, _end, chunk in _iter_dataset_chunks(h5[key]):
            if not np.isfinite(chunk).all():
                raise ValueError(f"{role} contains non-finite values: {key}")
            if key in _PHASE_KEYS and np.any((chunk < 0.0) | (chunk >= 180.0)):
                raise ValueError(f"{role} violates the [0, 180) phase convention: {key}")
    for _start, _end, chunk in _iter_dataset_chunks(h5["scenario"]):
        if np.any((chunk < 0) | (chunk >= len(_SCENARIOS))):
            raise ValueError(f"{role} contains invalid scenario labels")
    for _start, _end, chunk in _iter_dataset_chunks(h5["has_fault"]):
        if np.any((chunk != 0) & (chunk != 1)):
            raise ValueError(f"{role} contains invalid has_fault values")

    indices = np.asarray(h5["sample_index"][:], dtype=np.int64)
    start = integers["generation_start_index"]
    if not np.array_equal(indices, np.arange(start, start + n, dtype=np.int64)):
        raise ValueError(f"{role} sample_index must be contiguous and ordered")
    root_attributes = {key: _json_value(h5.attrs[key]) for key in sorted(_ROOT_ATTRS)}
    return _DatasetContract(
        generator_seed=integers["generator_seed"],
        start_index=start,
        sample_count=n,
        source_shard_count=integers["source_shard_count"],
        root_attributes=root_attributes,
    )


def _logical_digest(dataset: h5py.Dataset, *, chunk_rows: int) -> str:
    digest = hashlib.sha256()
    for _start, _end, chunk in _iter_dataset_chunks(dataset, chunk_rows):
        digest.update(np.ascontiguousarray(chunk).tobytes(order="C"))
    return digest.hexdigest()


def _compare_hdf5(
    merged_snapshot: _FileSnapshot,
    shard_snapshots: tuple[_FileSnapshot, ...],
    *,
    expected_generator_seed: int,
    expected_sample_start: int,
    expected_sample_count: int,
    chunk_rows: int,
) -> tuple[_DatasetContract, list[_DatasetContract], dict[str, object]]:
    shard_contracts: list[_DatasetContract] = []
    shard_ranges: list[tuple[int, int]] = []
    for ordinal, shard_snapshot in enumerate(shard_snapshots):
        match = _SHARD_RE.fullmatch(shard_snapshot.path.name)
        if match is None:  # guarded by the pin parser
            raise ValueError(f"invalid shard filename: {shard_snapshot.path.name}")
        filename_start = int(match.group("start"))
        filename_end = int(match.group("end"))
        if filename_end < filename_start:
            raise ValueError(f"invalid shard filename range: {shard_snapshot.path.name}")
        with shard_snapshot.open_hdf5() as shard:
            contract = _validate_open_dataset(
                shard,
                role=f"shard {ordinal}",
                expected_generator_seed=expected_generator_seed,
            )
        if contract.source_shard_count != 1:
            raise ValueError(f"shard {ordinal} source_shard_count must equal 1")
        if (
            contract.start_index != filename_start
            or contract.sample_count != filename_end - filename_start + 1
        ):
            raise ValueError(f"shard {ordinal} content does not match its filename range")
        if shard_ranges and filename_start != shard_ranges[-1][1] + 1:
            raise ValueError("ordered shard ranges contain a gap or overlap")
        shard_ranges.append((filename_start, filename_end))
        shard_contracts.append(contract)

    expected_end = expected_sample_start + expected_sample_count - 1
    if shard_ranges[0][0] != expected_sample_start or shard_ranges[-1][1] != expected_end:
        raise ValueError("ordered shard ranges do not cover the externally pinned range")
    if (
        sum(contract.sample_count for contract in shard_contracts)
        != expected_sample_count
    ):
        raise ValueError("ordered shard sample counts do not match the external pin")

    with merged_snapshot.open_hdf5() as merged:
        merged_contract = _validate_open_dataset(
            merged,
            role="merged dataset",
            expected_generator_seed=expected_generator_seed,
        )
        if (
            merged_contract.start_index != expected_sample_start
            or merged_contract.sample_count != expected_sample_count
            or merged_contract.source_shard_count != len(shard_snapshots)
        ):
            raise ValueError(
                "merged dataset generation counts do not match external pins"
            )

        ignored_root_attrs = {
            "generation_start_index",
            "expected_row_count",
            "source_shard_count",
        }
        merged_shared_attrs = {
            key: value
            for key, value in merged_contract.root_attributes.items()
            if key not in ignored_root_attrs
        }
        for ordinal, contract in enumerate(shard_contracts):
            shared_attrs = {
                key: value
                for key, value in contract.root_attributes.items()
                if key not in ignored_root_attrs
            }
            if shared_attrs != merged_shared_attrs:
                raise ValueError(
                    f"shard {ordinal} root metadata differs from merged dataset"
                )

        arrays: dict[str, object] = {}
        for key in _META_KEYS:
            merged_dataset = merged[key]
            digest = _logical_digest(merged_dataset, chunk_rows=chunk_rows)
            for ordinal, shard_snapshot in enumerate(shard_snapshots):
                with shard_snapshot.open_hdf5() as shard:
                    if shard[key].shape != merged_dataset.shape or shard[key].dtype != (
                        merged_dataset.dtype
                    ):
                        raise ValueError(
                            f"shard {ordinal} metadata shape/dtype differs: {key}"
                        )
                    for start, end, values in _iter_dataset_chunks(
                        shard[key], chunk_rows
                    ):
                        if not np.array_equal(values, merged_dataset[start:end]):
                            raise ValueError(f"shard {ordinal} metadata differs: {key}")
            arrays[key] = {
                "dtype": merged_dataset.dtype.str,
                "shape": list(merged_dataset.shape),
                "logical_c_order_bytes_sha256": digest,
                "shard_equality": "exact_repetition_in_every_shard",
            }

        for key in _ROW_KEYS:
            merged_dataset = merged[key]
            merged_digest = _logical_digest(merged_dataset, chunk_rows=chunk_rows)
            shard_digest = hashlib.sha256()
            row_start = 0
            for ordinal, shard_snapshot in enumerate(shard_snapshots):
                with shard_snapshot.open_hdf5() as shard:
                    source = shard[key]
                    if source.shape[1:] != merged_dataset.shape[1:] or source.dtype != (
                        merged_dataset.dtype
                    ):
                        raise ValueError(
                            f"shard {ordinal} row shape/dtype differs: {key}"
                        )
                    for start, end, values in _iter_dataset_chunks(source, chunk_rows):
                        expected = np.asarray(
                            merged_dataset[row_start + start : row_start + end]
                        )
                        if not np.array_equal(values, expected):
                            raise ValueError(
                                f"shard {ordinal} data differs from merged dataset: {key}"
                            )
                        shard_digest.update(
                            np.ascontiguousarray(values).tobytes(order="C")
                        )
                    row_start += source.shape[0]
            if row_start != merged_dataset.shape[0] or shard_digest.hexdigest() != (
                merged_digest
            ):
                raise DatasetLineageError(f"ordered shard digest mismatch: {key}")
            arrays[key] = {
                "dtype": merged_dataset.dtype.str,
                "shape": list(merged_dataset.shape),
                "logical_c_order_bytes_sha256": merged_digest,
                "shard_equality": "exact_ordered_concatenation",
            }
    return merged_contract, shard_contracts, arrays


def _snapshot_and_compare_hdf5(
    merged_dataset_path: str | Path,
    shard_directory: Path,
    pins: tuple[_ShardPin, ...],
    *,
    expected_merged_sha256: str,
    expected_merged_size_bytes: int,
    expected_generator_seed: int,
    expected_sample_start: int,
    expected_sample_count: int,
    chunk_rows: int,
) -> tuple[
    _FileSnapshot,
    list[_FileSnapshot],
    _DatasetContract,
    list[_DatasetContract],
    dict[str, object],
]:
    """Hash and parse every HDF5 artifact through one retained descriptor."""
    with ExitStack() as descriptors:
        merged_snapshot = _snapshot_pinned_file(
            merged_dataset_path,
            role="merged dataset",
            expected_sha256=expected_merged_sha256,
            expected_size_bytes=expected_merged_size_bytes,
            keep_open=True,
        )
        descriptors.callback(merged_snapshot.close)
        shard_snapshots: list[_FileSnapshot] = []
        for ordinal, pin in enumerate(pins):
            snapshot = _snapshot_pinned_file(
                shard_directory / pin.hdf5_filename,
                role=f"shard {ordinal} HDF5",
                expected_sha256=pin.hdf5_sha256,
                expected_size_bytes=pin.hdf5_size_bytes,
                keep_open=True,
            )
            descriptors.callback(snapshot.close)
            shard_snapshots.append(snapshot)

        merged_contract, shard_contracts, arrays = _compare_hdf5(
            merged_snapshot,
            tuple(shard_snapshots),
            expected_generator_seed=expected_generator_seed,
            expected_sample_start=expected_sample_start,
            expected_sample_count=expected_sample_count,
            chunk_rows=chunk_rows,
        )

        # The structured reads above used only the already-hashed descriptors.
        # A final pathname check additionally rejects a persistent replacement.
        merged_snapshot.verify_unchanged(role="merged dataset")
        for ordinal, snapshot in enumerate(shard_snapshots):
            snapshot.verify_unchanged(role=f"shard {ordinal} HDF5")
        return (
            merged_snapshot,
            shard_snapshots,
            merged_contract,
            shard_contracts,
            arrays,
        )


def _git(repo: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *arguments],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "")
        raise DatasetLineageError(
            f"Git repository check failed for {repo}: {str(detail).strip()}"
        ) from exc
    return result.stdout.strip()


def _git_bytes(repo: Path, *arguments: str, input_payload: bytes | None = None) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *arguments],
            check=True,
            capture_output=True,
            input=input_payload,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", b"")
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", errors="replace")
        raise DatasetLineageError(
            f"Git repository check failed for {repo}: {str(detail).strip()}"
        ) from exc
    return result.stdout


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(str(left.resolve(strict=True))) == os.path.normcase(
        str(right.resolve(strict=True))
    )


def _check_repository(
    path: str | Path,
    *,
    role: str,
    expected_commit: str,
    expected_origin_remote: str,
    expected_source_sha256: dict[str, str],
) -> tuple[dict[str, object], tuple[_FileSnapshot, ...]]:
    expected_commit = _require_commit(expected_commit, f"{role} expected commit")
    if not isinstance(expected_origin_remote, str) or not expected_origin_remote:
        raise ValueError(f"{role} expected origin remote must be non-empty")
    required_files = _SOURCE_FILES[role]
    if set(expected_source_sha256) != set(required_files):
        raise ValueError(
            f"{role} expected source hashes must name the exact required files"
        )
    requested = Path(path).absolute()
    directory = _snapshot_directory(requested, role=f"{role} repository")
    top_level = Path(_git(requested, "rev-parse", "--show-toplevel"))
    if not _same_path(requested, top_level):
        raise DatasetLineageError(f"{role} path is not the repository top level")
    commit = _git(requested, "rev-parse", "HEAD")
    if commit != expected_commit:
        raise DatasetLineageError(
            f"{role} commit mismatch: expected {expected_commit}, got {commit}"
        )
    status_output = _git(requested, "status", "--porcelain=v1", "--untracked-files=all")
    if status_output:
        raise DatasetLineageError(f"{role} repository must have a clean worktree")
    remotes = tuple(
        line
        for line in _git(
            requested, "config", "--get-all", "remote.origin.url"
        ).splitlines()
        if line
    )
    if remotes != (expected_origin_remote,):
        raise DatasetLineageError(f"{role} origin remote does not match the external pin")

    snapshots: list[_FileSnapshot] = []
    source_files: dict[str, object] = {}
    for relative_path in required_files:
        tracked = _git(requested, "ls-files", "--error-unmatch", "--", relative_path)
        if tracked.replace("\\", "/") != relative_path:
            raise DatasetLineageError(
                f"{role} source is not uniquely tracked: {relative_path}"
            )
        tree_record = _git(requested, "ls-tree", "HEAD", "--", relative_path).split()
        if (
            len(tree_record) < 4
            or tree_record[0] not in {"100644", "100755"}
            or tree_record[1] != "blob"
            or not re.fullmatch(r"[0-9a-f]{40,64}", tree_record[2])
        ):
            raise DatasetLineageError(
                f"{role} source is not a regular Git blob: {relative_path}"
            )
        snapshot = _snapshot_pinned_file(
            requested / Path(relative_path),
            role=f"{role} source {relative_path}",
            expected_sha256=expected_source_sha256[relative_path],
            capture_payload=True,
        )
        if snapshot.payload is None:  # pragma: no cover - local invariant
            raise DatasetLineageError("repository source snapshot has no payload")
        filtered_object_id = (
            _git_bytes(
                requested,
                "hash-object",
                "--stdin",
                "--path",
                relative_path,
                input_payload=snapshot.payload,
            )
            .decode("ascii")
            .strip()
        )
        if filtered_object_id != tree_record[2]:
            raise DatasetLineageError(
                f"{role} source does not match pinned commit blob: {relative_path}"
            )
        snapshots.append(snapshot)
        source_files[relative_path] = {
            "matches_commit_blob_after_git_clean_filter": True,
            "sha256": snapshot.sha256,
            "size_bytes": snapshot.size_bytes,
        }
    directory.verify_unchanged(role=f"{role} repository")
    return (
        {
            "path": str(requested),
            "commit": commit,
            "clean_worktree": True,
            "origin_remote": expected_origin_remote,
            "source_files": source_files,
        },
        tuple(snapshots),
    )


def _verify_repository_unchanged(
    path: Path,
    *,
    role: str,
    expected_commit: str,
    expected_origin_remote: str,
) -> None:
    if _git(path, "rev-parse", "HEAD") != expected_commit:
        raise DatasetLineageError(f"{role} commit changed during lineage verification")
    if _git(path, "status", "--porcelain=v1", "--untracked-files=all"):
        raise DatasetLineageError(f"{role} worktree changed during lineage verification")
    remotes = tuple(
        line
        for line in _git(path, "config", "--get-all", "remote.origin.url").splitlines()
        if line
    )
    if remotes != (expected_origin_remote,):
        raise DatasetLineageError(f"{role} origin changed during lineage verification")


def _path_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _remove_ours(path: Path, signature: tuple[int, int, int, int]) -> bool:
    if not _path_exists(path):
        return True
    try:
        current = path.lstat()
    except OSError:
        return False
    if _file_signature(current) != signature or not stat.S_ISREG(current.st_mode):
        return False
    try:
        path.unlink()
    except OSError:
        return False
    return True


def _publish_canonical_json(
    path: str | Path, manifest: dict[str, object]
) -> tuple[Path, str, int]:
    destination = Path(path).absolute()
    parent = destination.parent
    if parent.is_symlink():
        raise ValueError("lineage output parent must not be a symbolic link")
    parent.mkdir(parents=True, exist_ok=True)
    if not parent.is_dir():
        raise ValueError("lineage output parent must be a directory")
    partial = destination.with_name(destination.name + ".part")
    if _path_exists(destination):
        raise FileExistsError(f"refusing to overwrite lineage output: {destination}")
    if _path_exists(partial):
        raise FileExistsError(f"refusing to overwrite stale lineage partial: {partial}")
    payload = _canonical_json_bytes(manifest)
    digest = hashlib.sha256(payload).hexdigest()
    partial_signature: tuple[int, int, int, int] | None = None
    try:
        with partial.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        partial_signature = _file_signature(partial.lstat())
        try:
            os.link(partial, destination)
        except FileExistsError as exc:
            raise FileExistsError(
                f"lineage publication race at destination: {destination}"
            ) from exc
        destination_signature = _file_signature(destination.lstat())
        if destination_signature != partial_signature:
            raise DatasetLineageError(
                "lineage destination was replaced during publication"
            )
        partial.unlink()
        final = _snapshot_file(destination, role="published lineage")
        if final.sha256 != digest or final.size_bytes != len(payload):
            raise DatasetLineageError(
                "published lineage identity is not the canonical payload"
            )
        return destination, digest, len(payload)
    except BaseException as exc:
        replacement = False
        if partial_signature is not None:
            if not _remove_ours(destination, partial_signature):
                replacement = _path_exists(destination)
            _remove_ours(partial, partial_signature)
        if replacement:
            raise DatasetLineageError(
                "lineage destination was replaced; refusing destructive rollback"
            ) from exc
        raise


def build_dataset_lineage_2d(
    merged_dataset_path: str | Path,
    shard_directory: str | Path,
    shard_pins_path: str | Path,
    forward_repository_path: str | Path,
    geogen_repository_path: str | Path,
    output_path: str | Path,
    *,
    split: str,
    expected_merged_sha256: str,
    expected_merged_size_bytes: int,
    expected_shard_pins_sha256: str,
    expected_generator_seed: int,
    expected_sample_start: int,
    expected_sample_count: int,
    expected_forward_commit: str,
    expected_forward_origin_remote: str,
    expected_forward_source_sha256: dict[str, str],
    expected_geogen_commit: str,
    expected_geogen_origin_remote: str,
    expected_geogen_source_sha256: dict[str, str],
    chunk_rows: int = _CHUNK_ROWS,
) -> DatasetLineageResult:
    """Verify existing artifacts and publish immutable canonical lineage JSON.

    Every external file is SHA-256 pinned.  Shard/log pins are supplied in a
    separately SHA-256-pinned canonical JSON document so a large shard set can
    be passed without an unsafe directory-discovery mode.
    """
    if split not in {"train", "validation"}:
        raise ValueError("split must be train or validation")
    expected_generator_seed = _require_nonnegative_integer(
        expected_generator_seed, "expected_generator_seed"
    )
    expected_sample_start = _require_nonnegative_integer(
        expected_sample_start, "expected_sample_start"
    )
    expected_sample_count = _require_positive_integer(
        expected_sample_count, "expected_sample_count"
    )
    chunk_rows = _require_positive_integer(chunk_rows, "chunk_rows")

    pin_snapshot = _snapshot_pinned_file(
        shard_pins_path,
        role="shard pin manifest",
        expected_sha256=expected_shard_pins_sha256,
        capture_payload=True,
    )
    pins = _load_shard_pins(pin_snapshot)
    shard_directory_snapshot = _snapshot_directory(
        shard_directory, role="shard directory"
    )
    expected_entries = tuple(
        sorted(name for pin in pins for name in (pin.hdf5_filename, pin.log_filename))
    )
    if shard_directory_snapshot.entries != expected_entries:
        raise DatasetLineageError(
            "shard directory entries do not exactly match the externally pinned inventory"
        )

    log_snapshots: list[_FileSnapshot] = []
    for ordinal, pin in enumerate(pins):
        log_path = shard_directory_snapshot.path / pin.log_filename
        log_snapshots.append(
            _snapshot_pinned_file(
                log_path,
                role=f"shard {ordinal} log",
                expected_sha256=pin.log_sha256,
                expected_size_bytes=pin.log_size_bytes,
            )
        )

    forward_record, forward_sources = _check_repository(
        forward_repository_path,
        role="pimsr_forward",
        expected_commit=expected_forward_commit,
        expected_origin_remote=expected_forward_origin_remote,
        expected_source_sha256=expected_forward_source_sha256,
    )
    geogen_record, geogen_sources = _check_repository(
        geogen_repository_path,
        role="pimsr_geogen",
        expected_commit=expected_geogen_commit,
        expected_origin_remote=expected_geogen_origin_remote,
        expected_source_sha256=expected_geogen_source_sha256,
    )

    (
        merged_snapshot,
        shard_snapshots,
        merged_contract,
        shard_contracts,
        arrays,
    ) = _snapshot_and_compare_hdf5(
        merged_dataset_path,
        shard_directory_snapshot.path,
        pins,
        expected_merged_sha256=expected_merged_sha256,
        expected_merged_size_bytes=expected_merged_size_bytes,
        expected_generator_seed=expected_generator_seed,
        expected_sample_start=expected_sample_start,
        expected_sample_count=expected_sample_count,
        chunk_rows=chunk_rows,
    )

    # Re-hash every external file after HDF5 and Git reads.  This is
    # intentionally stronger than relying only on timestamps or inode numbers.
    pin_snapshot.verify_unchanged(role="shard pin manifest")
    for ordinal, snapshot in enumerate(log_snapshots):
        snapshot.verify_unchanged(role=f"shard {ordinal} log")
    for snapshot in (*forward_sources, *geogen_sources):
        snapshot.verify_unchanged(role="pinned repository source")
    shard_directory_snapshot.verify_unchanged(role="shard directory")
    _verify_repository_unchanged(
        Path(forward_record["path"]),
        role="pimsr_forward",
        expected_commit=expected_forward_commit,
        expected_origin_remote=expected_forward_origin_remote,
    )
    _verify_repository_unchanged(
        Path(geogen_record["path"]),
        role="pimsr_geogen",
        expected_commit=expected_geogen_commit,
        expected_origin_remote=expected_geogen_origin_remote,
    )

    shards: list[dict[str, object]] = []
    for ordinal, (pin, contract, hdf5_snapshot, log_snapshot) in enumerate(
        zip(pins, shard_contracts, shard_snapshots, log_snapshots, strict=True)
    ):
        sample_end = contract.start_index + contract.sample_count - 1
        shards.append(
            {
                "ordinal": ordinal,
                "sample_start": contract.start_index,
                "sample_end": sample_end,
                "sample_count": contract.sample_count,
                "hdf5": {
                    "filename": pin.hdf5_filename,
                    "sha256": hdf5_snapshot.sha256,
                    "size_bytes": hdf5_snapshot.size_bytes,
                },
                "log": {
                    "filename": pin.log_filename,
                    "sha256": log_snapshot.sha256,
                    "size_bytes": log_snapshot.size_bytes,
                },
            }
        )
    forward_public_record = {
        **forward_record,
        "path": Path(str(forward_record["path"])).name,
    }
    geogen_public_record = {
        **geogen_record,
        "path": Path(str(geogen_record["path"])).name,
    }
    manifest: dict[str, object] = {
        "schema": LINEAGE_SCHEMA,
        "schema_version": LINEAGE_SCHEMA_VERSION,
        "evidence_scope": EVIDENCE_SCOPE,
        "split": split,
        "inputs": {
            "merged_dataset": {
                "path": merged_snapshot.path.name,
                "sha256": merged_snapshot.sha256,
                "size_bytes": merged_snapshot.size_bytes,
            },
            "shard_directory": {
                "path": shard_directory_snapshot.path.name,
                "entry_count": len(shard_directory_snapshot.entries),
            },
            "shard_pin_manifest": {
                "path": pin_snapshot.path.name,
                "sha256": pin_snapshot.sha256,
                "size_bytes": pin_snapshot.size_bytes,
            },
            "shards": shards,
        },
        "repositories": {
            "pimsr_forward": forward_public_record,
            "pimsr_geogen": geogen_public_record,
        },
        "verification": {
            "arrays": arrays,
            "chunk_rows": chunk_rows,
            "concatenation": "exact_ordered_array_and_metadata_equality",
            "forward_regeneration_performed": False,
            "generation_complete": True,
            "generation_start_index": merged_contract.start_index,
            "generation_time_execution_proven": False,
            "generator_seed": merged_contract.generator_seed,
            "root_attributes": merged_contract.root_attributes,
            "sample_count": merged_contract.sample_count,
            "sample_end_index": (
                merged_contract.start_index + merged_contract.sample_count - 1
            ),
            "schema_contract": "pimsr-mt-2d/v2",
            "source_shard_count": merged_contract.source_shard_count,
        },
    }
    published_path, digest, size = _publish_canonical_json(output_path, manifest)
    return DatasetLineageResult(published_path, digest, size, manifest)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify an existing public PIMSR 2-D merged dataset against pinned "
            "shards/logs and clean pinned source repositories; no generation is run"
        )
    )
    parser.add_argument("--merged-dataset", type=Path, required=True)
    parser.add_argument("--shard-directory", type=Path, required=True)
    parser.add_argument("--shard-pins", type=Path, required=True)
    parser.add_argument("--forward-repository", type=Path, required=True)
    parser.add_argument("--geogen-repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--expected-merged-sha256", required=True)
    parser.add_argument("--expected-merged-size-bytes", type=int, required=True)
    parser.add_argument("--expected-shard-pins-sha256", required=True)
    parser.add_argument("--expected-generator-seed", type=int, required=True)
    parser.add_argument("--expected-sample-start", type=int, required=True)
    parser.add_argument("--expected-sample-count", type=int, required=True)
    parser.add_argument("--expected-forward-commit", required=True)
    parser.add_argument("--expected-forward-origin-remote", required=True)
    parser.add_argument("--expected-forward-dataset2d-sha256", required=True)
    parser.add_argument("--expected-forward-mt2d-sha256", required=True)
    parser.add_argument("--expected-forward-sensors-sha256", required=True)
    parser.add_argument("--expected-geogen-commit", required=True)
    parser.add_argument("--expected-geogen-origin-remote", required=True)
    parser.add_argument("--expected-geogen-section2d-sha256", required=True)
    parser.add_argument("--chunk-rows", type=int, default=_CHUNK_ROWS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = build_dataset_lineage_2d(
        args.merged_dataset,
        args.shard_directory,
        args.shard_pins,
        args.forward_repository,
        args.geogen_repository,
        args.output,
        split=args.split,
        expected_merged_sha256=args.expected_merged_sha256,
        expected_merged_size_bytes=args.expected_merged_size_bytes,
        expected_shard_pins_sha256=args.expected_shard_pins_sha256,
        expected_generator_seed=args.expected_generator_seed,
        expected_sample_start=args.expected_sample_start,
        expected_sample_count=args.expected_sample_count,
        expected_forward_commit=args.expected_forward_commit,
        expected_forward_origin_remote=args.expected_forward_origin_remote,
        expected_forward_source_sha256={
            "src/pimsr_forward/dataset2d.py": (args.expected_forward_dataset2d_sha256),
            "src/pimsr_forward/mt2d.py": args.expected_forward_mt2d_sha256,
            "src/pimsr_forward/sensors.py": args.expected_forward_sensors_sha256,
        },
        expected_geogen_commit=args.expected_geogen_commit,
        expected_geogen_origin_remote=args.expected_geogen_origin_remote,
        expected_geogen_source_sha256={
            "src/pimsr_geogen/section2d.py": args.expected_geogen_section2d_sha256,
        },
        chunk_rows=args.chunk_rows,
    )
    print(f"lineage: {result.path} sha256={result.sha256} size={result.size_bytes}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the wrapper script
    raise SystemExit(main())
