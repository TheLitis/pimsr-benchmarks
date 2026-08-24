"""Materialize one fail-closed hidden 2-D ModEM campaign.

The production contract is deliberately different from the public training
dataset builder.  It creates exactly 100 forced-family geological bases,
runs one pinned ModEM forward solve for each base, and derives five seeded
noise realizations from that one clean response.  No SimPEG forward solve is
used by this module.

Materialization has two phases.  Immutable raw ModEM bundles are accumulated
under an operator-private work directory and may be resumed.  Once every base
has been materially verified, deterministic public observations and private
truth/operator evidence are published into separate, new-only directories.
The public directory contains neither generation seeds nor family reveals.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import importlib
import importlib.metadata
import io
import json
import math
import os
import platform
import re
import stat
import sys
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from . import modem2d_forward as modem2d_bridge
from ._publication_io import (
    close_publication_descriptor,
    ensure_real_directory,
    open_exclusive_publication,
    set_publication_descriptor_read_only,
)
from .modem2d_forward import (
    CANONICAL_MODEL_SHAPE,
    CANONICAL_RESPONSE_SHAPE,
    PINNED_CONTAINER_REF,
    ArtifactSnapshot,
    CanonicalTruth,
    ModEMResponse,
    NestedMeshConfig,
    VerifiedRuntime,
    canonical_json_sha256,
    mapped_model,
    parse_modem_response,
    require_snapshot_unchanged,
    run_modem_forward,
    snapshot_file,
)
from .prediction_lock2d import (
    FAMILY_COMMITMENT_CONTRACT,
    FAMILY_PARTITION_SCHEMA,
    FAMILY_PARTITION_SCHEMA_VERSION,
    GEOLOGICAL_FAMILIES,
)

CAMPAIGN_SCHEMA = "pimsr-modem2d-hidden-generation-closure"
CAMPAIGN_SCHEMA_VERSION = 3
GENERATION_CONTRACT_SCHEMA = "pimsr-modem2d-hidden-generation-contract"
GENERATION_CONTRACT_SCHEMA_VERSION = 2
OBSERVATION_SCHEMA = "pimsr-sota-2d-observations"
OBSERVATION_SCHEMA_VERSION = 1
TRUTH_SCHEMA = "pimsr-sota-2d-truth"
TRUTH_SCHEMA_VERSION = 2
PUBLIC_MANIFEST_SCHEMA = "pimsr-sota-2d-observation-manifest"
PUBLIC_MANIFEST_SCHEMA_VERSION = 3
OPERATOR_MANIFEST_SCHEMA = "pimsr-sota-2d-scoring-manifest"
OPERATOR_MANIFEST_SCHEMA_VERSION = 3
FAMILY_REVEAL_SCHEMA = "pimsr-sota-2d-family-partition-reveal"
FAMILY_REVEAL_SCHEMA_VERSION = 1

BASES_PER_FAMILY = 20
BASE_COUNT = len(GEOLOGICAL_FAMILIES) * BASES_PER_FAMILY
NOISE_REALIZATIONS_PER_BASE = 5
SAMPLE_COUNT = BASE_COUNT * NOISE_REALIZATIONS_PER_BASE

BASE_LAYER_RNG = "numpy.default_rng([generator_seed,base_index])"
SECTION_RNG = "numpy.default_rng([generator_seed,2,base_index])"
NOISE_RNG = "numpy.default_rng([generator_seed,3,base_index,noise_index])"
BASE_LAYER_SCENARIO = "forced_background_before_2d_scenario_injection"
SCENARIO_POLICY = "SectionGenerator.sample(base_index,scenario=family_id)"
GEOLOGY_CONTRACT = "pimsr-geogen.SectionGenerator/default-grid/v1"
CLEAN_FORWARD_CONTRACT = "pinned_modem_2d_raw_forward_per_unique_base/v1"
NOISE_CONTRACT = "pimsr-forward.SensorModel/mt-noise+tm-severity-v5/v1"
FROZEN_GENERATION_RUNTIME = {
    "python_version": "3.11.15",
    "numpy_version": "2.4.6",
    "pimsr_geogen_version": "0.2.0",
    "pimsr_forward_version": "0.2.0",
}
FROZEN_RUNTIME_DISTRIBUTIONS = {
    "numpy": "2.4.6",
    "h5py": "3.16.0",
    "pimsr_benchmarks": "0.2.0",
    "pimsr_geogen": "0.2.0",
    "pimsr_forward": "0.2.0",
}

DEFAULT_RHO_LOG10_FLOOR = np.float32(0.05)
DEFAULT_PHASE_DEGREE_FLOOR = np.float32(2.9)
FAMILY_COMMITMENT_DOMAIN = b"pimsr-sota-2d-family-partition/v1\x00"
SAMPLE_ID_DOMAIN = b"pimsr-sota-2d-opaque-sample-id-v1\x00"

OBSERVATION_MEMBER_ORDER = (
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
TRUTH_MEMBER_ORDER = (
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
RAW_BUNDLE_FILES = (
    "model.rho",
    "template.dat",
    "forward.dat",
    "responses.npz",
    "solver.stdout.txt",
    "solver.stderr.txt",
    "provenance.json",
)
OUTPUT_FILES = RAW_BUNDLE_FILES[:-1]

_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}$", flags=re.ASCII)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$", flags=re.ASCII)
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$", flags=re.ASCII)


class HiddenCampaign2DError(RuntimeError):
    """An input, raw solve, or publication violated the hidden contract."""


@dataclass(frozen=True)
class CampaignGeometry2D:
    """Pinned canonical model and observation axes."""

    x_cell_centers_m: np.ndarray
    depth_cell_centers_m: np.ndarray
    frequencies_hz: np.ndarray
    station_x_m: np.ndarray
    source: ArtifactSnapshot


@dataclass(frozen=True)
class HiddenBase2D:
    """One forced-family geology and its five logical sample rows."""

    base_index: int
    family_id: str
    base_model_id: str
    source_sample_indices: tuple[int, ...]
    opaque_sample_indices: tuple[int, ...]
    truth: CanonicalTruth
    has_fault: bool


@dataclass(frozen=True)
class FileIdentity2D:
    path: Path
    sha256: str
    size_bytes: int

    def reference(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class VerifiedBaseForward2D:
    """Stable material identity of one raw ModEM base bundle."""

    base: HiddenBase2D
    bundle_path: Path
    files: Mapping[str, FileIdentity2D]
    response: ModEMResponse
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class HiddenCampaign2DResult:
    campaign_id: str
    public_directory: Path
    operator_directory: Path
    observations: FileIdentity2D
    public_manifest: FileIdentity2D
    truth: FileIdentity2D
    operator_manifest: FileIdentity2D
    family_reveal: FileIdentity2D
    hidden_generation: FileIdentity2D


@dataclass(frozen=True)
class GenerationRuntimeManifest2D:
    snapshot: ArtifactSnapshot
    value: Mapping[str, Any]

    @property
    def identity(self) -> FileIdentity2D:
        return FileIdentity2D(
            self.snapshot.path, self.snapshot.sha256, self.snapshot.size_bytes
        )


@dataclass(frozen=True)
class _CoreCampaign2D:
    observations_arrays: Mapping[str, np.ndarray]
    observations_payload: bytes
    observations_identity: FileIdentity2D
    public_manifest_payload: bytes
    public_manifest_identity: FileIdentity2D
    truth_payload: bytes
    truth_identity: FileIdentity2D
    operator_manifest_payload: bytes
    operator_manifest_identity: FileIdentity2D
    family_reveal_payload: bytes
    family_reveal_identity: FileIdentity2D
    family_commitment_sha256: str


@dataclass(frozen=True)
class _FinalEvidencePlan2D:
    provenance_payloads: Mapping[str, bytes]
    final_files: Mapping[str, Mapping[str, FileIdentity2D]]
    closure_payload: bytes
    closure_identity: FileIdentity2D


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"{name} must be a canonical lowercase identifier")
    return value


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _commit(value: object, name: str) -> str:
    if not isinstance(value, str) or not _COMMIT_RE.fullmatch(value):
        raise ValueError(f"{name} must be a full lowercase Git commit")
    return value


def _canonical_json_bytes(value: object, *, pretty: bool = False) -> bytes:
    try:
        if pretty:
            text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
        else:
            text = json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
    except (TypeError, ValueError) as exc:
        raise HiddenCampaign2DError(f"value is not finite JSON: {exc}") from exc
    return (text + "\n").encode("utf-8")


def _canonical_object_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_identity(path: Path, *, role: str) -> FileIdentity2D:
    snapshot = snapshot_file(path, role=role)
    return FileIdentity2D(snapshot.path, snapshot.sha256, snapshot.size_bytes)


def _artifact_record(
    identity: FileIdentity2D, *, schema: str, version: int
) -> dict[str, object]:
    return {
        "schema": schema,
        "schema_version": version,
        "sha256": identity.sha256,
        "size_bytes": identity.size_bytes,
    }


def _write_exclusive(path: Path, payload: bytes) -> FileIdentity2D:
    ensure_real_directory(
        path.parent,
        error_type=HiddenCampaign2DError,
        role="hidden campaign publication parent",
    )
    descriptor = open_exclusive_publication(path)
    failed = True
    try:
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise HiddenCampaign2DError(
                    f"zero-byte write while publishing hidden artifact: {path}"
                )
            written += count
        os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        captured = bytearray()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            captured.extend(block)
        if bytes(captured) != payload:
            raise HiddenCampaign2DError(f"published bytes changed while writing: {path}")
        set_publication_descriptor_read_only(descriptor)
        failed = False
    finally:
        close_publication_descriptor(descriptor, suppress_errors=failed)
    snapshot = snapshot_file(path, role="new hidden campaign artifact")
    if snapshot.payload != payload:
        raise HiddenCampaign2DError(f"published bytes changed while writing: {path}")
    return FileIdentity2D(snapshot.path, snapshot.sha256, snapshot.size_bytes)


def _npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, np.asarray(array), allow_pickle=False)
    return stream.getvalue()


def _npz_bytes(
    arrays: Mapping[str, np.ndarray],
    order: Sequence[str],
    *,
    compressed: bool,
) -> tuple[bytes, dict[str, dict[str, object]]]:
    if set(arrays) != set(order):
        raise HiddenCampaign2DError("NPZ array set is not exact")
    output = io.BytesIO()
    records: dict[str, dict[str, object]] = {}
    compression = zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=compression,
        compresslevel=9 if compressed else None,
        allowZip64=True,
        strict_timestamps=True,
    ) as archive:
        for name in order:
            array = np.asarray(arrays[name])
            payload = _npy_bytes(array)
            member = zipfile.ZipInfo(f"{name}.npy", (1980, 1, 1, 0, 0, 0))
            member.compress_type = compression
            member.create_system = 3
            member.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(member, payload)
            records[name] = {
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
    return output.getvalue(), records


def _unicode_scalar(value: str) -> np.ndarray:
    return np.asarray(value, dtype=f"<U{len(value)}")


def _unicode_vector(values: Sequence[str]) -> np.ndarray:
    width = max(len(value) for value in values)
    return np.asarray(values, dtype=f"<U{width}")


def _array_identity(value: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _strict_source_lineage(value: Mapping[str, Any]) -> dict[str, Any]:
    if set(value) != {"pimsr_forward", "pimsr_geogen"}:
        raise ValueError("source_lineage must contain exact forward/geogen repositories")
    forward = value["pimsr_forward"]
    geogen = value["pimsr_geogen"]
    if not isinstance(forward, Mapping) or set(forward) != {
        "repository_commit",
        "dataset2d_source_sha256",
        "sensors_source_sha256",
    }:
        raise ValueError("pimsr_forward source lineage is incomplete")
    if not isinstance(geogen, Mapping) or set(geogen) != {
        "repository_commit",
        "generator_source_sha256",
        "model_source_sha256",
        "rock_physics_source_sha256",
        "section2d_source_sha256",
    }:
        raise ValueError("pimsr_geogen source lineage is incomplete")
    normalized = {
        "pimsr_forward": {
            "repository_commit": _commit(
                forward["repository_commit"], "pimsr_forward.repository_commit"
            ),
            "dataset2d_source_sha256": _sha256(
                forward["dataset2d_source_sha256"], "pimsr_forward.dataset2d"
            ),
            "sensors_source_sha256": _sha256(
                forward["sensors_source_sha256"], "pimsr_forward.sensors"
            ),
        },
        "pimsr_geogen": {
            "repository_commit": _commit(
                geogen["repository_commit"], "pimsr_geogen.repository_commit"
            ),
            **{
                name: _sha256(geogen[name], f"pimsr_geogen.{name}")
                for name in (
                    "generator_source_sha256",
                    "model_source_sha256",
                    "rock_physics_source_sha256",
                    "section2d_source_sha256",
                )
            },
        },
    }
    return normalized


def generation_contract_2d(
    *, generator_seed: int, source_lineage: Mapping[str, Any]
) -> dict[str, Any]:
    """Return the exact comparator schema-v2 hidden generation contract."""
    if (
        type(generator_seed) is not int
        or not 0 <= generator_seed <= np.iinfo(np.int64).max
    ):
        raise ValueError("generator_seed must be a non-negative int64")
    return {
        "schema": GENERATION_CONTRACT_SCHEMA,
        "schema_version": GENERATION_CONTRACT_SCHEMA_VERSION,
        "generator_seed": generator_seed,
        "base_layer_rng": BASE_LAYER_RNG,
        "base_layer_scenario": BASE_LAYER_SCENARIO,
        "section_rng": SECTION_RNG,
        "scenario_policy": SCENARIO_POLICY,
        "noise_rng": NOISE_RNG,
        "geology_contract": GEOLOGY_CONTRACT,
        "clean_forward_contract": CLEAN_FORWARD_CONTRACT,
        "noise_contract": NOISE_CONTRACT,
        "source_lineage": _strict_source_lineage(source_lineage),
        "base_count": BASE_COUNT,
        "noise_realizations_per_base": NOISE_REALIZATIONS_PER_BASE,
    }


def _generation_runtime() -> dict[str, str]:
    try:
        benchmarks_version = importlib.metadata.version("pimsr-benchmarks")
        geogen_version = importlib.metadata.version("pimsr-geogen")
        forward_version = importlib.metadata.version("pimsr-forward")
    except importlib.metadata.PackageNotFoundError as exc:
        raise HiddenCampaign2DError(
            "pinned pimsr-geogen/pimsr-forward distributions are required"
        ) from exc
    if platform.python_implementation() != "CPython":
        raise HiddenCampaign2DError(
            "hidden generation requires the frozen CPython runtime"
        )
    observed = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pimsr_geogen_version": geogen_version,
        "pimsr_forward_version": forward_version,
    }
    observed_distributions = {
        "numpy": np.__version__,
        "h5py": h5py.__version__,
        "pimsr_benchmarks": benchmarks_version,
        "pimsr_geogen": geogen_version,
        "pimsr_forward": forward_version,
    }
    if (
        observed != FROZEN_GENERATION_RUNTIME
        or observed_distributions != FROZEN_RUNTIME_DISTRIBUTIONS
    ):
        raise HiddenCampaign2DError(
            "hidden generation runtime differs from the frozen preregistered runtime: "
            f"observed={observed!r}, distributions={observed_distributions!r}"
        )
    return observed


def _tree_file_records(root: Path, *, package: str) -> list[dict[str, Any]]:
    root = root.resolve(strict=True)
    try:
        root_info = os.lstat(root)
    except OSError as exc:
        raise HiddenCampaign2DError(f"cannot inspect installed {package} root") from exc
    if not stat.S_ISDIR(root_info.st_mode) or root.is_symlink():
        raise HiddenCampaign2DError(
            f"installed {package} root must be a direct directory"
        )
    records: list[dict[str, Any]] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as exc:
            raise HiddenCampaign2DError(
                f"cannot enumerate installed {package} tree"
            ) from exc
        for entry in entries:
            if entry.name == "__pycache__" or entry.name.endswith((".pyc", ".pyo")):
                continue
            path = Path(entry.path)
            try:
                resolved = path.resolve(strict=True)
            except OSError as exc:
                raise HiddenCampaign2DError(
                    f"cannot resolve installed {package} tree entry"
                ) from exc
            if not resolved.is_relative_to(root):
                raise HiddenCampaign2DError(
                    f"installed {package} tree entry escapes its package root: {path}"
                )
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise HiddenCampaign2DError(
                    f"cannot inspect installed {package} file"
                ) from exc
            if entry.is_symlink():
                raise HiddenCampaign2DError(
                    f"installed {package} tree contains a symbolic link: {path}"
                )
            if stat.S_ISDIR(info.st_mode):
                pending.append(path)
                continue
            if not stat.S_ISREG(info.st_mode):
                raise HiddenCampaign2DError(
                    f"installed {package} tree contains a non-regular entry: {path}"
                )
            snapshot = snapshot_file(path, role=f"installed {package} tree file")
            records.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "sha256": snapshot.sha256,
                    "size_bytes": snapshot.size_bytes,
                }
            )
    records.sort(key=lambda item: str(item["relative_path"]))
    if not records:
        raise HiddenCampaign2DError(f"installed {package} tree is empty")
    return records


def _installed_distribution_roots() -> tuple[tuple[str, Path], ...]:
    import pimsr_forward
    import pimsr_geogen

    import pimsr_benchmarks

    roots: list[tuple[str, Path]] = []
    for name, module in (
        ("numpy", np),
        ("h5py", h5py),
        ("pimsr_benchmarks", pimsr_benchmarks),
        ("pimsr_geogen", pimsr_geogen),
        ("pimsr_forward", pimsr_forward),
    ):
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            raise HiddenCampaign2DError(f"installed {name} has no material package root")
        roots.append((name, Path(module_file).resolve(strict=True).parent))
    return tuple(roots)


def _runtime_manifest_value(source_lineage: Mapping[str, Any]) -> dict[str, Any]:
    versions = _generation_runtime()
    lineage = _strict_source_lineage(source_lineage)
    source_snapshots = _verify_generation_sources(lineage)
    for snapshot in source_snapshots:
        require_snapshot_unchanged(snapshot, role="runtime manifest source closure")
    executable = snapshot_file(sys.executable, role="hidden generation Python executable")
    tree_records: list[dict[str, Any]] = []
    distributions: dict[str, dict[str, str]] = {}
    version_by_package = {
        "numpy": versions["numpy_version"],
        "h5py": FROZEN_RUNTIME_DISTRIBUTIONS["h5py"],
        "pimsr_benchmarks": FROZEN_RUNTIME_DISTRIBUTIONS["pimsr_benchmarks"],
        "pimsr_geogen": versions["pimsr_geogen_version"],
        "pimsr_forward": versions["pimsr_forward_version"],
    }
    for package, root in _installed_distribution_roots():
        files = _tree_file_records(root, package=package)
        tree_body = {
            "package": package,
            "files": files,
            "schema": "pimsr-hidden-generation-installed-package-tree-2d",
            "schema_version": 1,
        }
        tree_sha = _canonical_object_sha256(tree_body)
        distributions[package] = {
            "version": version_by_package[package],
            "installed_tree_sha256": tree_sha,
        }
        tree_records.append(
            {
                "package": package,
                "installed_tree_sha256": tree_sha,
                "files": files,
            }
        )
    ordered_tree = {
        "schema": "pimsr-hidden-generation-installed-tree-record-2d",
        "schema_version": 1,
        "python_executable_sha256": executable.sha256,
        "distributions": tree_records,
    }
    value = {
        "schema": "pimsr-hidden-generation-runtime-2d",
        "schema_version": 1,
        "python": {
            "implementation": "CPython",
            "version": versions["python_version"],
            "executable_sha256": executable.sha256,
        },
        "distributions": distributions,
        "source_closure": lineage,
        "tree_manifest_sha256": _canonical_object_sha256(ordered_tree),
    }
    require_snapshot_unchanged(executable, role="hidden generation Python executable")
    for snapshot in source_snapshots:
        require_snapshot_unchanged(snapshot, role="runtime manifest source closure")
    return value


def build_hidden_generation_runtime_manifest_2d(
    destination: str | Path, *, source_lineage: Mapping[str, Any]
) -> FileIdentity2D:
    """Publish a deterministic runtime identity before any hidden seed exists."""
    path = Path(destination).resolve(strict=False)
    if path.suffix.lower() != ".json":
        raise ValueError("runtime manifest destination must use a .json suffix")
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite runtime manifest: {path}")
    value = _runtime_manifest_value(source_lineage)
    return _write_exclusive(path, _canonical_json_bytes(value))


def validate_hidden_generation_runtime_manifest_2d(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
) -> GenerationRuntimeManifest2D:
    """Recompute the complete environment and compare it with an external pin."""
    expected_digest = _sha256(expected_sha256, "expected runtime manifest SHA-256")
    if type(expected_size_bytes) is not int or expected_size_bytes <= 0:
        raise ValueError("expected runtime manifest size must be positive")
    snapshot = snapshot_file(path, role="hidden generation runtime manifest")
    if snapshot.sha256 != expected_digest or snapshot.size_bytes != expected_size_bytes:
        raise HiddenCampaign2DError("runtime manifest differs from its external pin")
    value = _strict_json_payload(
        snapshot.payload, role="hidden generation runtime manifest"
    )
    expected_keys = {
        "schema",
        "schema_version",
        "python",
        "distributions",
        "source_closure",
        "tree_manifest_sha256",
    }
    if set(value) != expected_keys or (
        value["schema"] != "pimsr-hidden-generation-runtime-2d"
        or value["schema_version"] != 1
        or snapshot.payload != _canonical_json_bytes(value)
    ):
        raise HiddenCampaign2DError("runtime manifest schema/canonicalization is wrong")
    source_closure = value["source_closure"]
    if not isinstance(source_closure, Mapping):
        raise HiddenCampaign2DError("runtime manifest source closure is not an object")
    recomputed = _runtime_manifest_value(source_closure)
    if dict(value) != recomputed:
        raise HiddenCampaign2DError(
            "installed environment differs from the pinned runtime manifest"
        )
    require_snapshot_unchanged(snapshot, role="hidden generation runtime manifest")
    return GenerationRuntimeManifest2D(snapshot, dict(value))


def _verify_generation_sources(
    source_lineage: Mapping[str, Any],
) -> tuple[ArtifactSnapshot, ...]:
    lineage = _strict_source_lineage(source_lineage)
    specifications = (
        (
            "pimsr_forward.dataset2d",
            lineage["pimsr_forward"]["dataset2d_source_sha256"],
        ),
        (
            "pimsr_forward.sensors",
            lineage["pimsr_forward"]["sensors_source_sha256"],
        ),
        (
            "pimsr_geogen.generator",
            lineage["pimsr_geogen"]["generator_source_sha256"],
        ),
        (
            "pimsr_geogen.model",
            lineage["pimsr_geogen"]["model_source_sha256"],
        ),
        (
            "pimsr_geogen.rock_physics",
            lineage["pimsr_geogen"]["rock_physics_source_sha256"],
        ),
        (
            "pimsr_geogen.section2d",
            lineage["pimsr_geogen"]["section2d_source_sha256"],
        ),
    )
    snapshots: list[ArtifactSnapshot] = []
    for module_name, expected_sha256 in specifications:
        module = importlib.import_module(module_name)
        module_path = getattr(module, "__file__", None)
        if not isinstance(module_path, str) or not module_path.endswith(".py"):
            raise HiddenCampaign2DError(
                f"{module_name} must resolve to a material Python source file"
            )
        snapshot = snapshot_file(module_path, role=f"hidden source {module_name}")
        if snapshot.sha256 != expected_sha256:
            raise HiddenCampaign2DError(
                f"installed {module_name} differs from the transitive source pin"
            )
        snapshots.append(snapshot)
    return tuple(snapshots)


def load_campaign_geometry_2d(
    path: str | Path, *, expected_sha256: str
) -> CampaignGeometry2D:
    """Load only canonical axes from one externally pinned schema-v2 HDF5."""
    expected = _sha256(expected_sha256, "expected geometry SHA-256")
    source = snapshot_file(path, role="hidden campaign geometry source")
    if source.sha256 != expected:
        raise HiddenCampaign2DError("geometry source differs from its external pin")
    try:
        with h5py.File(io.BytesIO(source.payload), "r") as h5:
            if (
                h5.attrs.get("schema") != "pimsr-mt-2d"
                or int(h5.attrs.get("schema_version", -1)) != 2
            ):
                raise HiddenCampaign2DError(
                    "geometry source must be pimsr-mt-2d schema version 2"
                )
            names = ("x_grid", "depth_grid", "frequencies", "station_x")
            if any(name not in h5 for name in names):
                raise HiddenCampaign2DError("geometry source is missing a canonical axis")
            arrays = {
                "x": np.asarray(h5["x_grid"][:], dtype="<f8"),
                "depth": np.asarray(h5["depth_grid"][:], dtype="<f8"),
                "frequency": np.asarray(h5["frequencies"][:], dtype="<f8"),
                "station": np.asarray(h5["station_x"][:], dtype="<f8"),
            }
    except HiddenCampaign2DError:
        raise
    except (OSError, ValueError) as exc:
        raise HiddenCampaign2DError(f"cannot decode geometry HDF5: {exc}") from exc
    shapes = {
        "x": (CANONICAL_MODEL_SHAPE[1],),
        "depth": (CANONICAL_MODEL_SHAPE[0],),
        "frequency": (CANONICAL_RESPONSE_SHAPE[0],),
        "station": (CANONICAL_RESPONSE_SHAPE[1],),
    }
    for name, array in arrays.items():
        if (
            array.shape != shapes[name]
            or not np.isfinite(array).all()
            or np.any(np.diff(array) <= 0.0)
        ):
            raise HiddenCampaign2DError(f"geometry {name} axis is not canonical")
        array.setflags(write=False)
    if np.any(arrays["depth"] <= 0.0) or np.any(arrays["frequency"] <= 0.0):
        raise HiddenCampaign2DError("depth/frequency axes must be positive")
    if arrays["station"][0] < arrays["x"][0] or arrays["station"][-1] > arrays["x"][-1]:
        raise HiddenCampaign2DError("stations must lie inside the canonical x grid")
    require_snapshot_unchanged(source, role="hidden campaign geometry source")
    return CampaignGeometry2D(
        arrays["x"], arrays["depth"], arrays["frequency"], arrays["station"], source
    )


def _secret_bytes(
    value: bytes | bytearray | memoryview | str | Path, *, role: str, exact: int | None
) -> bytes:
    if isinstance(value, (bytes, bytearray, memoryview)):
        payload = bytes(value)
    elif isinstance(value, (str, Path)):
        payload = snapshot_file(value, role=role).payload
    else:
        raise TypeError(f"{role} must be bytes or a regular-file path")
    if exact is not None and len(payload) != exact:
        raise ValueError(f"{role} must contain exactly {exact} bytes")
    if exact is None and len(payload) < 32:
        raise ValueError(f"{role} must contain at least 32 bytes")
    return payload


def _opaque_sample_indices(
    *, generator_seed: int, campaign_id: str, key: bytes
) -> tuple[int, ...]:
    split = campaign_id.encode("ascii")
    values: list[int] = []
    for source_index in range(SAMPLE_COUNT):
        message = (
            SAMPLE_ID_DOMAIN
            + generator_seed.to_bytes(8, "big", signed=False)
            + source_index.to_bytes(8, "big", signed=False)
            + len(split).to_bytes(4, "big", signed=False)
            + split
        )
        digest = hmac.digest(key, message, "sha256")
        values.append(int.from_bytes(digest[:8], "big") & np.iinfo(np.int64).max)
    if len(set(values)) != SAMPLE_COUNT:
        raise HiddenCampaign2DError("HMAC-derived opaque sample id collision")
    return tuple(values)


def _family_for_base(base_index: int) -> str:
    return GEOLOGICAL_FAMILIES[base_index // BASES_PER_FAMILY]


def _build_hidden_bases(
    *,
    generator_seed: int,
    campaign_id: str,
    sample_id_key: bytes,
    geometry: CampaignGeometry2D,
    section_generator_factory: Callable[[int], Any] | None = None,
) -> tuple[HiddenBase2D, ...]:
    if section_generator_factory is None:
        from pimsr_geogen.model import DEFAULT_DEPTH_GRID
        from pimsr_geogen.section2d import DEFAULT_X_GRID, SectionGenerator

        if not np.array_equal(geometry.x_cell_centers_m, DEFAULT_X_GRID.astype("<f8")):
            raise HiddenCampaign2DError(
                "geometry x axis differs from SectionGenerator default"
            )
        if not np.array_equal(
            geometry.depth_cell_centers_m, DEFAULT_DEPTH_GRID.astype("<f8")
        ):
            raise HiddenCampaign2DError(
                "geometry depth axis differs from SectionGenerator default"
            )
        section_generator_factory = SectionGenerator
    generator = section_generator_factory(generator_seed)
    opaque = _opaque_sample_indices(
        generator_seed=generator_seed, campaign_id=campaign_id, key=sample_id_key
    )
    bases: list[HiddenBase2D] = []
    geology_digests: set[str] = set()
    for base_index in range(BASE_COUNT):
        family = _family_for_base(base_index)
        section = generator.sample(base_index, scenario=family)
        if section.scenario != family or int(section.seed) != base_index:
            raise HiddenCampaign2DError(
                "SectionGenerator returned the wrong forced family/index"
            )
        if not np.array_equal(
            np.asarray(section.x_grid, dtype="<f8"), geometry.x_cell_centers_m
        ) or not np.array_equal(
            np.asarray(section.depth_grid, dtype="<f8"),
            geometry.depth_cell_centers_m,
        ):
            raise HiddenCampaign2DError(
                "generated geology axes differ from campaign geometry"
            )
        grid = np.ascontiguousarray(section.log10_res, dtype="<f4")
        if grid.shape != CANONICAL_MODEL_SHAPE or not np.isfinite(grid).all():
            raise HiddenCampaign2DError("generated geology grid is invalid")
        digest = hashlib.sha256(grid.tobytes(order="C")).hexdigest()
        if digest in geology_digests:
            raise HiddenCampaign2DError(
                "two hidden bases generated byte-identical geology"
            )
        geology_digests.add(digest)
        source_indices = tuple(
            base_index * NOISE_REALIZATIONS_PER_BASE + noise_index
            for noise_index in range(NOISE_REALIZATIONS_PER_BASE)
        )
        opaque_indices = tuple(opaque[index] for index in source_indices)
        base_model_id = f"base-{base_index:03d}"
        truth = CanonicalTruth(
            log10_resistivity=grid,
            x_centres_m=geometry.x_cell_centers_m,
            depth_centres_m=geometry.depth_cell_centers_m,
            frequencies_hz=geometry.frequencies_hz,
            station_x_m=geometry.station_x_m,
            sample_id=base_model_id,
        )
        bases.append(
            HiddenBase2D(
                base_index,
                family,
                base_model_id,
                source_indices,
                opaque_indices,
                truth,
                bool(section.has_fault),
            )
        )
    if len(bases) != BASE_COUNT or len(geology_digests) != BASE_COUNT:
        raise HiddenCampaign2DError("hidden base campaign is incomplete")
    return tuple(bases)


def _sensor_ar1_curve(size: int, lag1: float, rng: np.random.Generator) -> np.ndarray:
    innovations = rng.normal(0.0, 1.0, size)
    result = np.empty(size, dtype=np.float64)
    result[0] = innovations[0]
    coefficient = float(np.clip(lag1, 0.0, 0.95))
    scale = math.sqrt(1.0 - coefficient * coefficient)
    for index in range(1, size):
        result[index] = coefficient * result[index - 1] + scale * innovations[index]
    return result


def _fold_phase_float32_safe(values: np.ndarray) -> np.ndarray:
    maximum = float(np.nextafter(np.float32(180.0), np.float32(0.0)))
    folded = np.remainder(np.asarray(values, dtype=np.float64), 180.0)
    return np.where(folded > maximum, 0.0, folded)


def _apply_frozen_mt_noise(
    rho_app: np.ndarray,
    phase_degrees: np.ndarray,
    periods_seconds: np.ndarray,
    rng: np.random.Generator,
    *,
    shift_sigma: float | None = None,
    distort_hi: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rho = np.asarray(rho_app, dtype=np.float64)
    phase = np.asarray(phase_degrees, dtype=np.float64)
    periods = np.asarray(periods_seconds, dtype=np.float64)
    relative = np.full(rho.shape, 0.03, dtype=np.float64)
    relative += 0.02 * ((periods >= 0.1) & (periods <= 10.0))
    rho_noisy = rho * np.exp(rng.normal(0.0, relative))
    phase_noisy = phase + rng.normal(0.0, 1.0, phase.shape)
    upper = 0.25 if distort_hi is None else distort_hi
    if upper > 0.0:
        lower = min(0.02, upper)
        amplitude = float(np.exp(rng.uniform(np.log(lower), np.log(upper))))
        rho_noisy *= 10.0 ** (amplitude * _sensor_ar1_curve(rho.size, 0.46, rng))
        phase_noisy += 40.0 * amplitude * _sensor_ar1_curve(phase.size, 0.46, rng)
    sigma = 0.15 if shift_sigma is None else shift_sigma
    rho_noisy *= 10.0 ** rng.normal(0.0, sigma)
    return rho_noisy, _fold_phase_float32_safe(phase_noisy)


def _noisy_response(
    response: ModEMResponse,
    *,
    generator_seed: int,
    base_index: int,
    noise_index: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng([generator_seed, 3, base_index, noise_index])
    tm_shift_sigma = float(rng.uniform(0.15, 0.32))
    tm_distort_hi = float(np.exp(rng.uniform(np.log(0.25), np.log(0.45))))
    periods = 1.0 / response.frequencies_hz
    rho_te = np.power(10.0, response.log10_rho_te)
    rho_tm = np.power(10.0, response.log10_rho_tm)
    noisy_rho_te = np.empty_like(rho_te)
    noisy_phase_te = np.empty_like(response.phase_te_deg)
    noisy_rho_tm = np.empty_like(rho_tm)
    noisy_phase_tm = np.empty_like(response.phase_tm_deg)
    for station_index in range(response.station_x_m.size):
        noisy_rho_te[:, station_index], noisy_phase_te[:, station_index] = (
            _apply_frozen_mt_noise(
                rho_te[:, station_index],
                response.phase_te_deg[:, station_index],
                periods,
                rng,
            )
        )
        noisy_rho_tm[:, station_index], noisy_phase_tm[:, station_index] = (
            _apply_frozen_mt_noise(
                rho_tm[:, station_index],
                response.phase_tm_deg[:, station_index],
                periods,
                rng,
                shift_sigma=tm_shift_sigma,
                distort_hi=tm_distort_hi,
            )
        )
    return {
        "observed_log10_rho_te": np.log10(noisy_rho_te).astype("<f4"),
        "observed_phase_te_degrees": noisy_phase_te.astype("<f4"),
        "observed_log10_rho_tm": np.log10(noisy_rho_tm).astype("<f4"),
        "observed_phase_tm_degrees": noisy_phase_tm.astype("<f4"),
    }


def _render_modem_inputs(
    truth: CanonicalTruth, mesh: NestedMeshConfig
) -> tuple[bytes, bytes]:
    """Reproduce the protected bridge's exact model/template bytes."""
    mapped_log10, dy, dz = mapped_model(truth, mesh)
    mapped_ln = mapped_log10 * math.log(10.0)
    model_lines = [f"{dy.size:d} {dz.size:d} LOGE"]
    for values in (dy, dz):
        for start in range(0, values.size, 10):
            model_lines.append(
                " ".join(f"{item:.12e}" for item in values[start : start + 10])
            )
    model_lines.append("0")
    model_lines.extend(" ".join(f"{item:.12e}" for item in row) for row in mapped_ln)
    model_payload = ("\n".join(model_lines) + "\n").encode("ascii")

    width = float(dy.sum())
    station_y = 0.5 * width + truth.station_x_m
    periods = 1.0 / truth.frequencies_hz
    template_lines: list[str] = []
    for mode in ("TE", "TM"):
        template_lines.extend(
            (
                f"# PIMSR canonical sample {truth.sample_id} ModEM forward template",
                (
                    "# Period(s) Code GG_Lat GG_Lon X(m) Y(m) Z(m) "
                    "Component Real Imag Error"
                ),
                f"> {mode}_Impedance",
                "> exp(+i\\omega t)",
                "> [V/m]/[T]",
                "> 0.00",
                "> 0.000 0.000",
                f"> {periods.size:d} {station_y.size:d}",
            )
        )
        for site_index, y_value in enumerate(station_y, start=1):
            for period in periods:
                template_lines.append(
                    f"{period:.12e} S{site_index:02d} 0.0 0.0 0.0 "
                    f"{y_value:.12e} 0.0 {mode} 0.0 0.0 1.0"
                )
    return model_payload, ("\n".join(template_lines) + "\n").encode("ascii")


def _pending_truth_source(
    *,
    campaign_id: str,
    base: HiddenBase2D,
    generator_seed: int,
    generation_contract_sha256: str,
    geometry: CampaignGeometry2D,
) -> dict[str, Any]:
    return {
        "schema": "pimsr-modem2d-hidden-base-generation-source-pending",
        "schema_version": 1,
        "campaign_id": campaign_id,
        "base_model_id": base.base_model_id,
        "family_id": base.family_id,
        "base_index": base.base_index,
        "generator_seed": generator_seed,
        "base_layer_rng_key": [generator_seed, base.base_index],
        "section_rng_key": [generator_seed, 2, base.base_index],
        "generation_contract_sha256": generation_contract_sha256,
        "source_generator_sample_indices": list(base.source_sample_indices),
        "geometry_source_sha256": geometry.source.sha256,
        "geometry_source_size_bytes": geometry.source.size_bytes,
    }


def _strict_json_payload(payload: bytes, *, role: str) -> Mapping[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise HiddenCampaign2DError(f"{role} duplicates key {key!r}")
            value[key] = item
        return value

    def reject_constant(value: str) -> None:
        raise HiddenCampaign2DError(f"{role} contains non-finite {value}")

    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except HiddenCampaign2DError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise HiddenCampaign2DError(f"cannot decode {role}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise HiddenCampaign2DError(f"{role} must contain a JSON object")
    return value


def _record_matches_snapshot(record: Any, snapshot: ArtifactSnapshot) -> bool:
    return (
        isinstance(record, Mapping)
        and set(record) in ({"sha256", "size_bytes"}, {"path", "sha256", "size_bytes"})
        and record.get("sha256") == snapshot.sha256
        and record.get("size_bytes") == snapshot.size_bytes
    )


def _validate_response_npz(snapshot: ArtifactSnapshot, response: ModEMResponse) -> None:
    expected_keys = (
        "schema",
        "schema_version",
        "frequencies_hz",
        "station_x_m",
        "z_eb_te_real",
        "z_eb_te_imag",
        "z_eb_tm_real",
        "z_eb_tm_imag",
        "log10_rho_te",
        "phase_te_deg",
        "log10_rho_tm",
        "phase_tm_deg",
    )
    try:
        with np.load(io.BytesIO(snapshot.payload), allow_pickle=False) as archive:
            if tuple(archive.files) != expected_keys:
                raise HiddenCampaign2DError("responses.npz member order is not exact")
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    except HiddenCampaign2DError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise HiddenCampaign2DError(f"cannot decode responses.npz: {exc}") from exc
    expected = {
        "frequencies_hz": response.frequencies_hz,
        "station_x_m": response.station_x_m,
        "z_eb_te_real": response.z_eb_te.real,
        "z_eb_te_imag": response.z_eb_te.imag,
        "z_eb_tm_real": response.z_eb_tm.real,
        "z_eb_tm_imag": response.z_eb_tm.imag,
        "log10_rho_te": response.log10_rho_te,
        "phase_te_deg": response.phase_te_deg,
        "log10_rho_tm": response.log10_rho_tm,
        "phase_tm_deg": response.phase_tm_deg,
    }
    if (
        arrays["schema"].shape != ()
        or arrays["schema"].item() != "pimsr-modem2d-response"
        or arrays["schema_version"].shape != ()
        or int(arrays["schema_version"].item()) != 1
        or any(
            not np.array_equal(arrays[name], values) for name, values in expected.items()
        )
    ):
        raise HiddenCampaign2DError("responses.npz differs from parsed forward.dat")


def _verify_forward_bundle(
    path: Path,
    *,
    base: HiddenBase2D,
    mesh: NestedMeshConfig,
    runtime: VerifiedRuntime,
    expected_truth_source: Mapping[str, Any],
    timeout_seconds: float,
) -> VerifiedBaseForward2D:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise HiddenCampaign2DError(f"missing raw ModEM bundle: {path}") from exc
    if not stat.S_ISDIR(info.st_mode) or path.is_symlink():
        raise HiddenCampaign2DError("raw ModEM bundle must be a direct directory")
    entries = sorted(item.name for item in path.iterdir())
    if entries != sorted(RAW_BUNDLE_FILES):
        raise HiddenCampaign2DError(f"raw ModEM bundle has unexpected files: {entries}")
    snapshots = {
        name: snapshot_file(path / name, role=f"raw ModEM {base.base_model_id} {name}")
        for name in RAW_BUNDLE_FILES
    }
    provenance = _strict_json_payload(
        snapshots["provenance.json"].payload,
        role=f"{base.base_model_id} raw ModEM provenance",
    )
    expected_provenance_payload = (
        json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    if snapshots["provenance.json"].payload != expected_provenance_payload:
        raise HiddenCampaign2DError("raw ModEM provenance is not deterministic JSON")
    expected_root = {
        "schema",
        "schema_version",
        "truth",
        "truth_source",
        "mesh",
        "runtime",
        "runtime_identity_sha256",
        "bridge_source",
        "input_contract",
        "response_contract",
        "execution",
        "outputs",
    }
    if set(provenance) != expected_root:
        raise HiddenCampaign2DError("raw ModEM provenance key set is not exact")
    expected_mesh = {**mesh.canonical_record(), "mesh_config_sha256": mesh.sha256}
    if (
        provenance["schema"] != "pimsr-modem2d-forward-run"
        or provenance["schema_version"] != 1
        or provenance["truth"] != base.truth.identity_record()
        or provenance["truth_source"] != expected_truth_source
        or provenance["mesh"] != expected_mesh
        or provenance["runtime"] != dict(runtime.record)
        or provenance["runtime_identity_sha256"] != runtime.identity_sha256
        or canonical_json_sha256(provenance["runtime"]) != runtime.identity_sha256
    ):
        raise HiddenCampaign2DError("raw ModEM provenance identity is wrong")
    outputs = provenance["outputs"]
    if not isinstance(outputs, Mapping) or set(outputs) != set(OUTPUT_FILES):
        raise HiddenCampaign2DError("raw ModEM provenance output set is not exact")
    if any(
        not isinstance(outputs[name], Mapping)
        or set(outputs[name]) != {"sha256", "size_bytes"}
        or not _record_matches_snapshot(outputs[name], snapshots[name])
        for name in OUTPUT_FILES
    ):
        raise HiddenCampaign2DError("raw ModEM output identity differs from provenance")
    contracts = provenance["input_contract"]
    response_contract = provenance["response_contract"]
    bridge_path = getattr(modem2d_bridge, "__file__", None)
    if not isinstance(bridge_path, str):
        raise HiddenCampaign2DError("installed ModEM bridge has no material source")
    bridge_snapshot = snapshot_file(bridge_path, role="installed ModEM bridge source")
    dy, dz = mesh.cell_widths(base.truth.depth_centres_m)
    model_contract = contracts.get("model") if isinstance(contracts, Mapping) else None
    template_contract = (
        contracts.get("template") if isinstance(contracts, Mapping) else None
    )
    expected_model_contract = {
        "representation": "LOGE natural_log_resistivity_ohm_m",
        "ny": int(dy.size),
        "nz_earth": int(dz.size),
        "total_width_m": float(dy.sum()),
        "total_depth_m": float(dz.sum()),
        "mapping": mesh.canonical_record()["mapping"],
        "spatial_operation_order": (
            "map canonical log10(rho) piecewise-constantly by physical centres, "
            "then multiply mapped values by ln(10)"
        ),
    }
    expected_template_contract = {
        "time_convention_requested": "exp(+i omega t)",
        "manual_conjugation": False,
        "units": "[V/m]/[T] (E/B)",
        "coordinate_mapping": {
            "ModEM_X_m": 0.0,
            "ModEM_Y_m": "total_width_m/2 + PIMSR_station_x_m",
            "ModEM_Z_m": 0.0,
        },
        "mode_mapping": {
            "ModEM_TE_Ex_over_By": "PIMSR_TE_Ey_over_Hx_Zyx_no_mode_swap",
            "ModEM_TM_Ey_over_Bx": "PIMSR_TM_Ex_over_Hy_Zxy_no_mode_swap",
        },
        "period_count": int(base.truth.frequencies_hz.size),
        "station_count": int(base.truth.station_x_m.size),
        "rows_per_mode": int(
            base.truth.frequencies_hz.size * base.truth.station_x_m.size
        ),
    }
    expected_response_contract = {
        "rows": {"TE": 96, "TM": 96},
        "all_rows_finite": True,
        "time_convention": "exp(+i omega t) as written by ModEM DataIO",
        "manual_conjugation": False,
        "native_units": "[V/m]/[T] (E/B)",
        "rho_formula": "mu0 * abs(E_over_B)**2 / omega",
        "phase_formula": "degrees(angle(E_over_B)) modulo 180",
        "canonical_mode_order": ["TE_Zyx", "TM_Zxy"],
    }
    if (
        not isinstance(contracts, Mapping)
        or set(contracts) != {"model", "template"}
        or not isinstance(model_contract, Mapping)
        or {key: model_contract.get(key) for key in expected_model_contract}
        != expected_model_contract
        or set(model_contract) != {*expected_model_contract, "artifact"}
        or not _record_matches_snapshot(
            model_contract.get("artifact"),
            snapshots["model.rho"],
        )
        or not isinstance(template_contract, Mapping)
        or {key: template_contract.get(key) for key in expected_template_contract}
        != expected_template_contract
        or set(template_contract) != {*expected_template_contract, "artifact"}
        or not _record_matches_snapshot(
            template_contract.get("artifact"),
            snapshots["template.dat"],
        )
        or not isinstance(response_contract, Mapping)
        or {key: response_contract.get(key) for key in expected_response_contract}
        != expected_response_contract
        or set(response_contract) != {*expected_response_contract, "artifact"}
        or not _record_matches_snapshot(
            response_contract.get("artifact"), snapshots["forward.dat"]
        )
    ):
        raise HiddenCampaign2DError("raw ModEM input/response contracts are not material")
    execution = provenance["execution"]
    command = execution.get("command") if isinstance(execution, Mapping) else None
    command_valid = (
        isinstance(command, list)
        and len(command) == 24
        and all(isinstance(value, str) for value in command)
        and command[:13]
        == [
            runtime.docker_executable,
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
            "--workdir",
            "/tmp",
            "--env",
            "LD_LIBRARY_PATH=/runtime/lib",
            "--mount",
        ]
        and command[13]
        == f"type=bind,source={runtime.runtime_path},target=/runtime,readonly"
        and command[14] == "--mount"
        and command[15].startswith("type=bind,source=")
        and command[15].endswith(",target=/input,readonly")
        and command[16] == "--mount"
        and command[17].startswith("type=bind,source=")
        and command[17].endswith(",target=/output")
        and command[18:]
        == [
            PINNED_CONTAINER_REF,
            "/runtime/bin/Mod2DMT",
            "-F",
            "/input/model.rho",
            "/input/template.dat",
            "/output/forward.dat",
        ]
    )
    if (
        not _record_matches_snapshot(provenance["bridge_source"], bridge_snapshot)
        or not isinstance(execution, Mapping)
        or set(execution)
        != {
            "command",
            "container_network",
            "container_root_filesystem",
            "input_mount",
            "runtime_mount",
            "timeout_seconds",
            "elapsed_seconds",
            "returncode",
        }
        or not command_valid
        or execution["container_network"] != "none"
        or execution["container_root_filesystem"] != "read_only"
        or execution["input_mount"] != "read_only"
        or execution["runtime_mount"] != "read_only"
        or execution["timeout_seconds"] != timeout_seconds
        or type(execution["elapsed_seconds"]) not in (int, float)
        or not math.isfinite(float(execution["elapsed_seconds"]))
        or float(execution["elapsed_seconds"]) < 0.0
        or execution["returncode"] != 0
    ):
        raise HiddenCampaign2DError("raw ModEM bridge/execution provenance is invalid")
    expected_model, expected_template = _render_modem_inputs(base.truth, mesh)
    if (
        snapshots["model.rho"].payload != expected_model
        or snapshots["template.dat"].payload != expected_template
    ):
        raise HiddenCampaign2DError(
            "raw ModEM model/template differ from generated truth"
        )
    response, _record = parse_modem_response(path / "forward.dat", base.truth, mesh)
    _validate_response_npz(snapshots["responses.npz"], response)
    for name, snapshot in snapshots.items():
        require_snapshot_unchanged(snapshot, role=f"verified raw ModEM {name}")
    require_snapshot_unchanged(bridge_snapshot, role="installed ModEM bridge source")
    return VerifiedBaseForward2D(
        base=base,
        bundle_path=path.resolve(strict=True),
        files={
            name: FileIdentity2D(snapshot.path, snapshot.sha256, snapshot.size_bytes)
            for name, snapshot in snapshots.items()
        },
        response=response,
        provenance=copy.deepcopy(dict(provenance)),
    )


def _work_contract(
    *,
    campaign_id: str,
    generator_seed: int,
    generation_contract_sha256: str,
    geometry: CampaignGeometry2D,
    mesh: NestedMeshConfig,
    runtime: VerifiedRuntime,
    generation_runtime: Mapping[str, str],
    generation_runtime_manifest: GenerationRuntimeManifest2D,
    sample_id_key: bytes,
    family_nonce: bytes,
) -> dict[str, Any]:
    return {
        "schema": "pimsr-modem2d-hidden-materialization-work",
        "schema_version": 1,
        "campaign_id": campaign_id,
        "generator_seed": generator_seed,
        "generation_contract_sha256": generation_contract_sha256,
        "geometry_source_sha256": geometry.source.sha256,
        "geometry_source_size_bytes": geometry.source.size_bytes,
        "mesh_config_sha256": mesh.sha256,
        "runtime_identity_sha256": runtime.identity_sha256,
        "generation_runtime": dict(generation_runtime),
        "generation_runtime_manifest": (generation_runtime_manifest.identity.reference()),
        "generation_runtime_tree_manifest_sha256": (
            generation_runtime_manifest.value["tree_manifest_sha256"]
        ),
        "sample_id_key_commitment_sha256": hashlib.sha256(sample_id_key).hexdigest(),
        "family_nonce_commitment_sha256": hashlib.sha256(family_nonce).hexdigest(),
        "base_count": BASE_COUNT,
        "noise_realizations_per_base": NOISE_REALIZATIONS_PER_BASE,
    }


def _prepare_work_directory(work_dir: str | Path, contract: Mapping[str, Any]) -> Path:
    requested = Path(work_dir)
    if requested.name in {"", ".", ".."}:
        raise ValueError("work_dir must have a concrete leaf name")
    requested.parent.mkdir(parents=True, exist_ok=True)
    work = requested.resolve(strict=False)
    if not os.path.lexists(work):
        work.mkdir()
    info = os.lstat(work)
    if not stat.S_ISDIR(info.st_mode) or work.is_symlink():
        raise HiddenCampaign2DError("work_dir must be a direct directory")
    contract_path = work / "work-contract.json"
    expected = _canonical_json_bytes(contract)
    if os.path.lexists(contract_path):
        snapshot = snapshot_file(contract_path, role="hidden work contract")
        if snapshot.payload != expected:
            raise HiddenCampaign2DError("work_dir belongs to a different hidden campaign")
    else:
        _write_exclusive(contract_path, expected)
    base_root = work / "base-forward-runs"
    if not os.path.lexists(base_root):
        base_root.mkdir()
    base_info = os.lstat(base_root)
    if not stat.S_ISDIR(base_info.st_mode) or base_root.is_symlink():
        raise HiddenCampaign2DError("base-forward-runs must be a direct directory")
    allowed = {"work-contract.json", "base-forward-runs"}
    unexpected = {item.name for item in work.iterdir()} - allowed
    if unexpected:
        raise HiddenCampaign2DError(
            f"work_dir contains foreign entries: {sorted(unexpected)}"
        )
    return work


def _materialize_raw_forwards(
    *,
    bases: Sequence[HiddenBase2D],
    work: Path,
    campaign_id: str,
    generator_seed: int,
    generation_contract_sha256: str,
    geometry: CampaignGeometry2D,
    mesh: NestedMeshConfig,
    runtime: VerifiedRuntime,
    timeout_seconds: float,
    progress: Callable[[int, int], None] | None,
) -> tuple[VerifiedBaseForward2D, ...]:
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be finite and positive")
    if len(bases) != BASE_COUNT or any(
        base.base_index != index or base.base_model_id != f"base-{index:03d}"
        for index, base in enumerate(bases)
    ):
        raise HiddenCampaign2DError(
            "raw materialization requires the exact ordered 100-base campaign"
        )
    raw_root = work / "base-forward-runs"
    expected_names = {base.base_model_id for base in bases}
    existing_names = {item.name for item in raw_root.iterdir()}
    if not existing_names.issubset(expected_names):
        raise HiddenCampaign2DError("base-forward-runs contains an unknown base")
    verified: list[VerifiedBaseForward2D] = []
    for position, base in enumerate(bases, start=1):
        output = raw_root / base.base_model_id
        pending = _pending_truth_source(
            campaign_id=campaign_id,
            base=base,
            generator_seed=generator_seed,
            generation_contract_sha256=generation_contract_sha256,
            geometry=geometry,
        )
        if not os.path.lexists(output):
            runtime.require_unchanged()
            published, _response, _provenance = run_modem_forward(
                runtime=runtime,
                truth=base.truth,
                mesh=mesh,
                output_dir=output,
                source_provenance=pending,
                timeout_seconds=timeout_seconds,
            )
            if Path(published).resolve(strict=True) != output.resolve(strict=True):
                raise HiddenCampaign2DError("ModEM runner published an unexpected path")
        verified.append(
            _verify_forward_bundle(
                output,
                base=base,
                mesh=mesh,
                runtime=runtime,
                expected_truth_source=pending,
                timeout_seconds=timeout_seconds,
            )
        )
        runtime.require_unchanged()
        if progress is not None:
            progress(position, BASE_COUNT)
    return tuple(verified)


def _identity_for_payload(path: Path, payload: bytes) -> FileIdentity2D:
    return FileIdentity2D(
        path.resolve(strict=False), hashlib.sha256(payload).hexdigest(), len(payload)
    )


def _physical_contract() -> dict[str, Any]:
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
        "observation_axis_order": ["sample", "frequency", "station"],
        "observation_channel_order": [
            "log10_rho_te",
            "phase_te_degrees",
            "log10_rho_tm",
            "phase_tm_degrees",
        ],
        "phase_convention": "degrees_modulo_180_[0,180)",
        "phase_unit": "degree",
        "resistivity_unit": "ohm_m",
        "rotation_degrees": 0.0,
        "spectral_axis": "frequency",
        "spectral_order": "strictly_increasing",
        "spectral_unit": "Hz",
        "time_convention": "exp(+i_omega_t)",
        "truth_axis_order": ["sample", "depth", "x"],
        "vertical_positive": "down",
    }


def _family_commitment(
    *, campaign_id: str, nonce: bytes, rows: Sequence[Mapping[str, Any]]
) -> str:
    body = {
        "schema": "pimsr-sota-2d-family-partition-reveal-body",
        "schema_version": 1,
        "campaign_id": campaign_id,
        "rows": [dict(row) for row in rows],
    }
    digest = hashlib.sha256()
    digest.update(FAMILY_COMMITMENT_DOMAIN)
    digest.update(nonce)
    digest.update(_canonical_json_bytes(body))
    return digest.hexdigest()


def _clean_response_identity(
    response: ModEMResponse,
    *,
    mesh: NestedMeshConfig,
    depth_cell_centers_m: np.ndarray,
) -> str:
    domain_midpoint = 0.5 * float(mesh.cell_widths(depth_cell_centers_m)[0].sum())
    return _canonical_object_sha256(
        {
            "schema": "pimsr-modem2d-clean-response-identity",
            "schema_version": 1,
            "arrays": {
                "frequency_hz": _array_identity(response.frequencies_hz.astype("<f8")),
                "station_x_m": _array_identity(
                    (response.station_x_m + domain_midpoint).astype("<f8")
                ),
                "log10_rho_te": _array_identity(response.log10_rho_te.astype("<f8")),
                "phase_te_degrees": _array_identity(response.phase_te_deg.astype("<f8")),
                "log10_rho_tm": _array_identity(response.log10_rho_tm.astype("<f8")),
                "phase_tm_degrees": _array_identity(response.phase_tm_deg.astype("<f8")),
            },
        }
    )


def _observation_row_identity(
    *, sample_index: int, arrays: Mapping[str, np.ndarray]
) -> str:
    return _canonical_object_sha256(
        {
            "schema": "pimsr-sota-2d-observation-row-identity",
            "schema_version": 1,
            "sample_index": sample_index,
            "arrays": {
                name: _array_identity(arrays[name])
                for name in (
                    "observed_log10_rho_te",
                    "observed_phase_te_degrees",
                    "observed_log10_rho_tm",
                    "observed_phase_tm_degrees",
                    "valid_mask",
                )
            },
        }
    )


def _noise_delta_identity(
    *, observed: Mapping[str, np.ndarray], clean: ModEMResponse
) -> str:
    deltas = {
        "log10_rho_te": observed["observed_log10_rho_te"].astype("<f8")
        - clean.log10_rho_te,
        "phase_te_degrees": (
            observed["observed_phase_te_degrees"].astype("<f8")
            - clean.phase_te_deg
            + 90.0
        )
        % 180.0
        - 90.0,
        "log10_rho_tm": observed["observed_log10_rho_tm"].astype("<f8")
        - clean.log10_rho_tm,
        "phase_tm_degrees": (
            observed["observed_phase_tm_degrees"].astype("<f8")
            - clean.phase_tm_deg
            + 90.0
        )
        % 180.0
        - 90.0,
    }
    if not all(np.isfinite(array).all() for array in deltas.values()):
        raise HiddenCampaign2DError("hidden observation noise delta is non-finite")
    return _canonical_object_sha256(
        {
            "schema": "pimsr-modem2d-noise-delta-identity",
            "schema_version": 1,
            "arrays": {
                name: _array_identity(array.astype("<f8"))
                for name, array in deltas.items()
            },
        }
    )


def _split_contract(
    *,
    campaign_id: str,
    bases: Sequence[HiddenBase2D],
    reveal: Mapping[str, Any],
) -> dict[str, Any]:
    groups: list[dict[str, Any]] = []
    mappings: list[dict[str, int]] = []
    scenario_groups: dict[str, list[int]] = {family: [] for family in GEOLOGICAL_FAMILIES}
    for base in bases:
        for noise_index, (source_index, opaque_index) in enumerate(
            zip(
                base.source_sample_indices,
                base.opaque_sample_indices,
                strict=True,
            )
        ):
            groups.append(
                {
                    "base_model_id": base.base_model_id,
                    "family_id": base.family_id,
                    "noise_id": f"noise-{noise_index}",
                    "sample_ids": [f"sample-{opaque_index}"],
                }
            )
            mappings.append(
                {
                    "opaque_sample_index": opaque_index,
                    "source_generator_sample_index": source_index,
                }
            )
            scenario_groups[base.family_id].append(opaque_index)
    groups.sort(key=lambda item: int(str(item["sample_ids"][0]).removeprefix("sample-")))
    mappings.sort(key=lambda item: item["opaque_sample_index"])
    return {
        "groups": groups,
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
        "payload_row_order": "strictly_increasing_opaque_sample_index",
        "sample_count": SAMPLE_COUNT,
        "sample_id_mapping": mappings,
        "scenario_groups": [
            {
                "opaque_sample_indices": scenario_groups[family],
                "scenario": family,
                "scenario_index": family_index,
            }
            for family_index, family in enumerate(GEOLOGICAL_FAMILIES)
        ],
        "family_partition_reveal": dict(reveal),
        "split_id": campaign_id,
    }


def _build_core_campaign(
    *,
    campaign_id: str,
    generator_seed: int,
    family_nonce: bytes,
    geometry: CampaignGeometry2D,
    mesh: NestedMeshConfig,
    bases: Sequence[HiddenBase2D],
    forwards: Sequence[VerifiedBaseForward2D],
    public_directory: Path,
    operator_directory: Path,
) -> _CoreCampaign2D:
    if len(bases) != BASE_COUNT or len(forwards) != BASE_COUNT:
        raise HiddenCampaign2DError("core campaign requires exactly 100 verified bases")
    if any(item.base is not base for item, base in zip(forwards, bases, strict=True)):
        raise HiddenCampaign2DError(
            "verified ModEM bases are not in exact campaign order"
        )
    source_order_sample_ids = np.asarray(
        [sample for base in bases for sample in base.opaque_sample_indices],
        dtype="<i8",
    )
    if (
        source_order_sample_ids.shape != (SAMPLE_COUNT,)
        or np.unique(source_order_sample_ids).size != SAMPLE_COUNT
    ):
        raise HiddenCampaign2DError("campaign opaque sample ids are invalid")
    response_shape = (SAMPLE_COUNT, *CANONICAL_RESPONSE_SHAPE)
    observed = {
        name: np.empty(response_shape, dtype="<f4")
        for name in (
            "observed_log10_rho_te",
            "observed_phase_te_degrees",
            "observed_log10_rho_tm",
            "observed_phase_tm_degrees",
        )
    }
    truth_grid = np.empty((SAMPLE_COUNT, *CANONICAL_MODEL_SHAPE), dtype="<f4")
    scenarios: list[str] = []
    has_fault = np.empty(SAMPLE_COUNT, dtype=np.bool_)
    row = 0
    for base, forward in zip(bases, forwards, strict=True):
        response = forward.response
        if not np.array_equal(
            response.frequencies_hz, geometry.frequencies_hz
        ) or not np.array_equal(response.station_x_m, geometry.station_x_m):
            raise HiddenCampaign2DError(
                "ModEM response axes differ from campaign geometry"
            )
        base_grid = base.truth.log10_resistivity.astype("<f4")
        for noise_index in range(NOISE_REALIZATIONS_PER_BASE):
            noisy = _noisy_response(
                response,
                generator_seed=generator_seed,
                base_index=base.base_index,
                noise_index=noise_index,
            )
            for name, values in observed.items():
                values[row] = noisy[name]
            truth_grid[row] = base_grid
            scenarios.append(base.family_id)
            has_fault[row] = base.has_fault
            row += 1
    if row != SAMPLE_COUNT:
        raise HiddenCampaign2DError("campaign row assembly is incomplete")
    # The source generator order contains five adjacent noise realizations per
    # base and twenty adjacent bases per family.  Publishing that order would
    # disclose the withheld hierarchy even though the ids themselves are HMAC
    # opaque.  Sorting by the secret-key-derived ids gives a deterministic keyed
    # permutation while retaining a simple, independently checkable row order.
    public_order = np.argsort(source_order_sample_ids, kind="stable")
    sample_ids = source_order_sample_ids[public_order]
    if np.any(np.diff(sample_ids) <= 0):
        raise HiddenCampaign2DError("opaque-id publication permutation is invalid")
    observed = {name: values[public_order] for name, values in observed.items()}
    truth_grid = truth_grid[public_order]
    scenarios = [scenarios[int(index)] for index in public_order]
    has_fault = has_fault[public_order]
    floor_shape = response_shape
    observation_arrays: dict[str, np.ndarray] = {
        "schema": _unicode_scalar(OBSERVATION_SCHEMA),
        "schema_version": np.asarray(OBSERVATION_SCHEMA_VERSION, dtype="<i8"),
        "sample_index": sample_ids,
        "frequency_hz": geometry.frequencies_hz.astype("<f8"),
        "station_x_m": geometry.station_x_m.astype("<f8"),
        "x_cell_centers_m": geometry.x_cell_centers_m.astype("<f8"),
        "depth_cell_centers_m": geometry.depth_cell_centers_m.astype("<f8"),
        "observation_channel_order": _unicode_vector(
            (
                "log10_rho_te",
                "phase_te_degrees",
                "log10_rho_tm",
                "phase_tm_degrees",
            )
        ),
        **observed,
        "declared_evaluation_floor_log10_rho_te": np.full(
            floor_shape, DEFAULT_RHO_LOG10_FLOOR, dtype="<f4"
        ),
        "declared_evaluation_floor_phase_te_degrees": np.full(
            floor_shape, DEFAULT_PHASE_DEGREE_FLOOR, dtype="<f4"
        ),
        "declared_evaluation_floor_log10_rho_tm": np.full(
            floor_shape, DEFAULT_RHO_LOG10_FLOOR, dtype="<f4"
        ),
        "declared_evaluation_floor_phase_tm_degrees": np.full(
            floor_shape, DEFAULT_PHASE_DEGREE_FLOOR, dtype="<f4"
        ),
        "valid_mask": np.ones((SAMPLE_COUNT, 4, 8, 12), dtype=np.bool_),
    }
    observations_payload, observation_records = _npz_bytes(
        observation_arrays, OBSERVATION_MEMBER_ORDER, compressed=True
    )
    observations_identity = _identity_for_payload(
        public_directory / "observations.npz", observations_payload
    )
    truth_arrays = {
        "schema": _unicode_scalar(TRUTH_SCHEMA),
        "schema_version": np.asarray(TRUTH_SCHEMA_VERSION, dtype="<i8"),
        "sample_index": sample_ids.copy(),
        "observations_sha256": _unicode_scalar(observations_identity.sha256),
        "scenario": _unicode_vector(scenarios),
        "has_fault": has_fault,
        "x_cell_centers_m": geometry.x_cell_centers_m.astype("<f8"),
        "depth_cell_centers_m": geometry.depth_cell_centers_m.astype("<f8"),
        "truth_log10_resistivity": truth_grid,
    }
    truth_payload, _truth_records = _npz_bytes(
        truth_arrays, TRUTH_MEMBER_ORDER, compressed=False
    )
    truth_identity = _identity_for_payload(
        operator_directory / "truth.npz", truth_payload
    )
    reveal_rows = sorted(
        (
            {
                "base_model_id": base.base_model_id,
                "family_id": base.family_id,
                "noise_index": noise_index,
                "sample_index": opaque_index,
            }
            for base in bases
            for noise_index, opaque_index in enumerate(base.opaque_sample_indices)
        ),
        key=lambda item: int(item["sample_index"]),
    )
    reveal = {
        "campaign_id": campaign_id,
        "nonce_hex": family_nonce.hex(),
        "rows": reveal_rows,
        "schema": FAMILY_REVEAL_SCHEMA,
        "schema_version": FAMILY_REVEAL_SCHEMA_VERSION,
    }
    commitment_sha256 = _family_commitment(
        campaign_id=campaign_id, nonce=family_nonce, rows=reveal_rows
    )
    public_manifest = {
        "audience": "method_input_public",
        "declared_evaluation_floors": {
            "interpretation": (
                "declared evaluation floors, not empirical or source-noise standard "
                "deviations"
            ),
            "log10_rho_floor": float(DEFAULT_RHO_LOG10_FLOOR),
            "phase_degree_floor": float(DEFAULT_PHASE_DEGREE_FLOOR),
            "policy_id": "declared_evaluation_floors_log10_rho_phase_v1",
            "storage": "explicit_full_shape_float32_arrays_in_observations_payload",
            "validity": "explicit_all_true_boolean_mask",
        },
        "family_partition_commitment": {
            "contract": dict(FAMILY_COMMITMENT_CONTRACT),
            "schema": FAMILY_PARTITION_SCHEMA,
            "schema_version": FAMILY_PARTITION_SCHEMA_VERSION,
            "sha256": commitment_sha256,
        },
        "observation_payload": {
            "arrays": observation_records,
            "media_type": "application/x-npz",
            "schema": OBSERVATION_SCHEMA,
            "schema_version": OBSERVATION_SCHEMA_VERSION,
            "sha256": observations_identity.sha256,
            "size_bytes": observations_identity.size_bytes,
        },
        "physical_contract": _physical_contract(),
        "sample_count": SAMPLE_COUNT,
        "schema": PUBLIC_MANIFEST_SCHEMA,
        "schema_version": PUBLIC_MANIFEST_SCHEMA_VERSION,
        "split_id": campaign_id,
    }
    public_manifest_payload = _canonical_json_bytes(public_manifest)
    public_manifest_identity = _identity_for_payload(
        public_directory / "observations.public.json", public_manifest_payload
    )
    operator_manifest = {
        "artifacts": {
            "observations": _artifact_record(
                observations_identity,
                schema=OBSERVATION_SCHEMA,
                version=OBSERVATION_SCHEMA_VERSION,
            ),
            "public_observation_manifest": _artifact_record(
                public_manifest_identity,
                schema=PUBLIC_MANIFEST_SCHEMA,
                version=PUBLIC_MANIFEST_SCHEMA_VERSION,
            ),
            "withheld_truth": _artifact_record(
                truth_identity, schema=TRUTH_SCHEMA, version=TRUTH_SCHEMA_VERSION
            ),
        },
        "audience": "benchmark_operator_only",
        "schema": OPERATOR_MANIFEST_SCHEMA,
        "schema_version": OPERATOR_MANIFEST_SCHEMA_VERSION,
        "source": {
            "production_generation_closure": (
                "post_score_manifest.campaign.hidden_generation"
            )
        },
        "split": _split_contract(campaign_id=campaign_id, bases=bases, reveal=reveal),
    }
    operator_manifest_payload = _canonical_json_bytes(operator_manifest)
    operator_manifest_identity = _identity_for_payload(
        operator_directory / "operator.json", operator_manifest_payload
    )
    reveal_payload = _canonical_json_bytes(reveal)
    reveal_identity = _identity_for_payload(
        operator_directory / "family-reveal.json", reveal_payload
    )
    return _CoreCampaign2D(
        observations_arrays=observation_arrays,
        observations_payload=observations_payload,
        observations_identity=observations_identity,
        public_manifest_payload=public_manifest_payload,
        public_manifest_identity=public_manifest_identity,
        truth_payload=truth_payload,
        truth_identity=truth_identity,
        operator_manifest_payload=operator_manifest_payload,
        operator_manifest_identity=operator_manifest_identity,
        family_reveal_payload=reveal_payload,
        family_reveal_identity=reveal_identity,
        family_commitment_sha256=commitment_sha256,
    )


def _final_truth_source(
    *,
    campaign_id: str,
    base: HiddenBase2D,
    generator_seed: int,
    generation_contract_sha256: str,
    core: _CoreCampaign2D,
) -> dict[str, Any]:
    return {
        "schema": "pimsr-modem2d-hidden-base-generation-source",
        "schema_version": 2,
        "campaign_id": campaign_id,
        "operator_manifest_sha256": core.operator_manifest_identity.sha256,
        "base_model_id": base.base_model_id,
        "family_id": base.family_id,
        "base_index": base.base_index,
        "generator_seed": generator_seed,
        "base_layer_rng_key": [generator_seed, base.base_index],
        "section_rng_key": [generator_seed, 2, base.base_index],
        "generation_contract_sha256": generation_contract_sha256,
        "source_generator_sample_indices": list(base.source_sample_indices),
        "observations_sha256": core.observations_identity.sha256,
        "public_observation_manifest_sha256": core.public_manifest_identity.sha256,
        "withheld_truth_sha256": core.truth_identity.sha256,
        "family_partition_commitment_sha256": core.family_commitment_sha256,
    }


def _rebind_provenance(
    forward: VerifiedBaseForward2D,
    *,
    final_directory: Path,
    truth_source: Mapping[str, Any],
) -> bytes:
    value = copy.deepcopy(dict(forward.provenance))
    value["truth_source"] = dict(truth_source)
    value["input_contract"]["model"]["artifact"]["path"] = str(
        (final_directory / "model.rho").resolve(strict=False)
    )
    value["input_contract"]["template"]["artifact"]["path"] = str(
        (final_directory / "template.dat").resolve(strict=False)
    )
    value["response_contract"]["artifact"]["path"] = str(
        (final_directory / "forward.dat").resolve(strict=False)
    )
    return _canonical_json_bytes(value, pretty=True)


def _build_final_evidence_plan(
    *,
    campaign_id: str,
    generator_seed: int,
    generation_contract: Mapping[str, Any],
    geometry: CampaignGeometry2D,
    mesh: NestedMeshConfig,
    runtime: VerifiedRuntime,
    generation_runtime: Mapping[str, str],
    generation_runtime_manifest: GenerationRuntimeManifest2D,
    bases: Sequence[HiddenBase2D],
    forwards: Sequence[VerifiedBaseForward2D],
    core: _CoreCampaign2D,
    operator_directory: Path,
) -> _FinalEvidencePlan2D:
    generation_sha = _canonical_object_sha256(generation_contract)
    final_files: dict[str, dict[str, FileIdentity2D]] = {}
    provenance_payloads: dict[str, bytes] = {}
    base_rows: list[dict[str, Any]] = []
    forward_by_base: dict[str, VerifiedBaseForward2D] = {}
    for base, forward in zip(bases, forwards, strict=True):
        final_dir = operator_directory / "modem" / base.base_model_id
        truth_source = _final_truth_source(
            campaign_id=campaign_id,
            base=base,
            generator_seed=generator_seed,
            generation_contract_sha256=generation_sha,
            core=core,
        )
        provenance_payload = _rebind_provenance(
            forward, final_directory=final_dir, truth_source=truth_source
        )
        provenance_identity = _identity_for_payload(
            final_dir / "provenance.json", provenance_payload
        )
        identities = {
            name: FileIdentity2D(
                (final_dir / name).resolve(strict=False),
                forward.files[name].sha256,
                forward.files[name].size_bytes,
            )
            for name in OUTPUT_FILES
        }
        identities["provenance.json"] = provenance_identity
        final_files[base.base_model_id] = identities
        provenance_payloads[base.base_model_id] = provenance_payload
        clean_sha = _clean_response_identity(
            forward.response,
            mesh=mesh,
            depth_cell_centers_m=geometry.depth_cell_centers_m,
        )
        base_rows.append(
            {
                "base_index": base.base_index,
                "base_layer_rng_key": [generator_seed, base.base_index],
                "section_rng_key": [generator_seed, 2, base.base_index],
                "family_id": base.family_id,
                "base_model_id": base.base_model_id,
                "source_generator_sample_indices": list(base.source_sample_indices),
                "clean_response_sha256": clean_sha,
                "model": identities["model.rho"].reference(),
                "template": identities["template.dat"].reference(),
                "forward": identities["forward.dat"].reference(),
                "provenance": provenance_identity.reference(),
            }
        )
        forward_by_base[base.base_model_id] = forward
    noise_rows: list[dict[str, Any]] = []
    by_opaque: dict[int, tuple[HiddenBase2D, int, int]] = {}
    for base in bases:
        for noise_index, (source_index, opaque_index) in enumerate(
            zip(
                base.source_sample_indices,
                base.opaque_sample_indices,
                strict=True,
            )
        ):
            by_opaque[opaque_index] = (base, noise_index, source_index)
    valid_mask = core.observations_arrays["valid_mask"]
    for opaque_index in sorted(by_opaque):
        base, noise_index, source_index = by_opaque[opaque_index]
        forward = forward_by_base[base.base_model_id]
        row_arrays = {
            name: core.observations_arrays[name][source_index]
            for name in (
                "observed_log10_rho_te",
                "observed_phase_te_degrees",
                "observed_log10_rho_tm",
                "observed_phase_tm_degrees",
            )
        }
        row_arrays["valid_mask"] = valid_mask[source_index]
        clean_sha = _clean_response_identity(
            forward.response,
            mesh=mesh,
            depth_cell_centers_m=geometry.depth_cell_centers_m,
        )
        noise_rows.append(
            {
                "sample_index": opaque_index,
                "source_generator_sample_index": source_index,
                "base_index": base.base_index,
                "noise_rng_key": [
                    generator_seed,
                    3,
                    base.base_index,
                    noise_index,
                ],
                "base_model_id": base.base_model_id,
                "family_id": base.family_id,
                "noise_index": noise_index,
                "base_forward_sha256": final_files[base.base_model_id][
                    "forward.dat"
                ].sha256,
                "clean_response_sha256": clean_sha,
                "observation_row_sha256": _observation_row_identity(
                    sample_index=opaque_index, arrays=row_arrays
                ),
                "noise_delta_sha256": _noise_delta_identity(
                    observed=row_arrays, clean=forward.response
                ),
            }
        )
    closure = {
        "schema": CAMPAIGN_SCHEMA,
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "mesh": {**mesh.canonical_record(), "mesh_config_sha256": mesh.sha256},
        "runtime": dict(runtime.record),
        "runtime_identity_sha256": runtime.identity_sha256,
        "generation_runtime": dict(generation_runtime),
        "generation_runtime_manifest": (generation_runtime_manifest.identity.reference()),
        "bindings": {
            "operator_manifest_sha256": core.operator_manifest_identity.sha256,
            "observations_sha256": core.observations_identity.sha256,
            "public_observation_manifest_sha256": core.public_manifest_identity.sha256,
            "withheld_truth_sha256": core.truth_identity.sha256,
            "family_partition_commitment_sha256": core.family_commitment_sha256,
        },
        "generation_contract": dict(generation_contract),
        "observation_payload": core.observations_identity.reference(),
        "base_forward_runs": base_rows,
        "noise_rows": noise_rows,
    }
    closure_payload = _canonical_json_bytes(closure)
    closure_identity = _identity_for_payload(
        operator_directory / "hidden-generation.json", closure_payload
    )
    return _FinalEvidencePlan2D(
        provenance_payloads=provenance_payloads,
        final_files=final_files,
        closure_payload=closure_payload,
        closure_identity=closure_identity,
    )


def _ensure_direct_directory(path: Path, *, role: str) -> Path:
    requested = path.resolve(strict=False)
    ensure_real_directory(requested, error_type=HiddenCampaign2DError, role=role)
    info = os.lstat(requested)
    if not stat.S_ISDIR(info.st_mode) or requested.is_symlink():
        raise HiddenCampaign2DError(f"{role} must be a direct directory")
    return requested


def _require_published_identity(
    path: Path, expected: FileIdentity2D, *, role: str
) -> None:
    snapshot = snapshot_file(path, role=role)
    if (
        snapshot.path != expected.path
        or snapshot.sha256 != expected.sha256
        or snapshot.size_bytes != expected.size_bytes
    ):
        raise HiddenCampaign2DError(f"published {role} differs from its final plan")
    require_snapshot_unchanged(snapshot, role=role)


def _ensure_exact_file(path: Path, payload: bytes, *, role: str) -> FileIdentity2D:
    if os.path.lexists(path):
        snapshot = snapshot_file(path, role=role)
        if snapshot.payload != payload:
            raise HiddenCampaign2DError(
                f"refusing to overwrite differing existing {role}: {path}"
            )
        return FileIdentity2D(snapshot.path, snapshot.sha256, snapshot.size_bytes)
    return _write_exclusive(path, payload)


def _ensure_copied_file(
    source: FileIdentity2D, destination: Path, *, role: str
) -> FileIdentity2D:
    snapshot = snapshot_file(source.path, role=f"source {role}")
    if snapshot.sha256 != source.sha256 or snapshot.size_bytes != source.size_bytes:
        raise HiddenCampaign2DError(f"source {role} changed after verification")
    result = _ensure_exact_file(destination, snapshot.payload, role=role)
    if result.sha256 != source.sha256 or result.size_bytes != source.size_bytes:
        raise HiddenCampaign2DError(f"copied {role} differs from raw ModEM evidence")
    require_snapshot_unchanged(snapshot, role=f"source {role}")
    return result


def _publish_operator_directory(
    *,
    operator_directory: Path,
    core: _CoreCampaign2D,
    forwards: Sequence[VerifiedBaseForward2D],
    evidence: _FinalEvidencePlan2D,
) -> None:
    destination = _ensure_direct_directory(
        operator_directory, role="operator output directory"
    )
    completion = destination / "hidden-generation.json"
    if os.path.lexists(completion):
        snapshot = snapshot_file(completion, role="hidden generation completion manifest")
        if snapshot.payload != evidence.closure_payload:
            raise HiddenCampaign2DError(
                "completed operator directory differs from campaign"
            )
    _ensure_exact_file(
        destination / "truth.npz", core.truth_payload, role="withheld truth"
    )
    _ensure_exact_file(
        destination / "operator.json",
        core.operator_manifest_payload,
        role="operator manifest",
    )
    _ensure_exact_file(
        destination / "family-reveal.json",
        core.family_reveal_payload,
        role="family reveal",
    )
    modem_root = _ensure_direct_directory(
        destination / "modem", role="ModEM evidence root"
    )
    forward_by_id = {item.base.base_model_id: item for item in forwards}
    if set(forward_by_id) != set(evidence.final_files):
        raise HiddenCampaign2DError("final ModEM evidence plan is incomplete")
    for base_model_id in sorted(forward_by_id):
        final_dir = _ensure_direct_directory(
            modem_root / base_model_id, role=f"{base_model_id} final evidence"
        )
        forward = forward_by_id[base_model_id]
        expected = evidence.final_files[base_model_id]
        for name in OUTPUT_FILES:
            copied = _ensure_copied_file(
                forward.files[name],
                final_dir / name,
                role=f"{base_model_id} {name}",
            )
            if copied != expected[name]:
                raise HiddenCampaign2DError("final ModEM raw identity changed")
        provenance = _ensure_exact_file(
            final_dir / "provenance.json",
            evidence.provenance_payloads[base_model_id],
            role=f"{base_model_id} final provenance",
        )
        if provenance != expected["provenance.json"]:
            raise HiddenCampaign2DError("final ModEM provenance identity changed")
        if sorted(item.name for item in final_dir.iterdir()) != sorted(RAW_BUNDLE_FILES):
            raise HiddenCampaign2DError("final ModEM bundle contains foreign files")
    if sorted(item.name for item in modem_root.iterdir()) != sorted(forward_by_id):
        raise HiddenCampaign2DError("ModEM evidence root contains foreign bases")
    # Completion manifest is intentionally published last.
    _ensure_exact_file(
        completion, evidence.closure_payload, role="hidden generation completion manifest"
    )
    expected_root = {
        "truth.npz",
        "operator.json",
        "family-reveal.json",
        "hidden-generation.json",
        "modem",
    }
    if {item.name for item in destination.iterdir()} != expected_root:
        raise HiddenCampaign2DError("operator directory contains foreign artifacts")


def _verify_published_operator_directory(
    *,
    operator_directory: Path,
    core: _CoreCampaign2D,
    evidence: _FinalEvidencePlan2D,
) -> None:
    """Reopen and hash every final artifact, including all 700 raw files."""
    destination = _ensure_direct_directory(
        operator_directory, role="published operator output directory"
    )
    planned_root = {
        "truth.npz": core.truth_identity,
        "operator.json": core.operator_manifest_identity,
        "family-reveal.json": core.family_reveal_identity,
        "hidden-generation.json": evidence.closure_identity,
    }
    for name, identity in planned_root.items():
        _require_published_identity(
            destination / name, identity, role=f"published operator {name}"
        )
    modem_root = _ensure_direct_directory(
        destination / "modem", role="published ModEM evidence root"
    )
    if sorted(item.name for item in modem_root.iterdir()) != sorted(evidence.final_files):
        raise HiddenCampaign2DError("published ModEM evidence base set changed")
    verified = 0
    for base_model_id, planned in sorted(evidence.final_files.items()):
        base_dir = _ensure_direct_directory(
            modem_root / base_model_id,
            role=f"published {base_model_id} ModEM evidence",
        )
        if sorted(item.name for item in base_dir.iterdir()) != sorted(RAW_BUNDLE_FILES):
            raise HiddenCampaign2DError(
                f"published {base_model_id} ModEM evidence file set changed"
            )
        for name in RAW_BUNDLE_FILES:
            _require_published_identity(
                base_dir / name,
                planned[name],
                role=f"published {base_model_id} {name}",
            )
            verified += 1
    if verified != BASE_COUNT * len(RAW_BUNDLE_FILES):
        raise HiddenCampaign2DError("published raw ModEM evidence count is not 700")


def _publish_public_directory(*, public_directory: Path, core: _CoreCampaign2D) -> None:
    destination = _ensure_direct_directory(
        public_directory, role="public output directory"
    )
    completion = destination / "observations.public.json"
    if os.path.lexists(completion):
        snapshot = snapshot_file(completion, role="public observation manifest")
        if snapshot.payload != core.public_manifest_payload:
            raise HiddenCampaign2DError(
                "completed public directory differs from campaign"
            )
    _ensure_exact_file(
        destination / "observations.npz",
        core.observations_payload,
        role="public observations",
    )
    # Sanitized public manifest is the publication completion marker.
    _ensure_exact_file(
        completion, core.public_manifest_payload, role="public observation manifest"
    )
    if {item.name for item in destination.iterdir()} != {
        "observations.npz",
        "observations.public.json",
    }:
        raise HiddenCampaign2DError("public directory contains non-public artifacts")


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _preflight_existing_directory(
    path: Path, *, role: str, allowed_root_entries: set[str]
) -> None:
    if not os.path.lexists(path):
        return
    info = os.lstat(path)
    if not stat.S_ISDIR(info.st_mode) or path.is_symlink():
        raise HiddenCampaign2DError(f"{role} must be a direct directory")
    unexpected = {item.name for item in path.iterdir()} - allowed_root_entries
    if unexpected:
        raise HiddenCampaign2DError(
            f"{role} contains foreign entries: {sorted(unexpected)}"
        )


def materialize_hidden_campaign2d(
    *,
    campaign_id: str,
    generator_seed: int,
    geometry_h5: str | Path,
    expected_geometry_sha256: str,
    source_lineage: Mapping[str, Any],
    generation_runtime_manifest_path: str | Path,
    expected_generation_runtime_manifest_sha256: str,
    expected_generation_runtime_manifest_size_bytes: int,
    sample_id_key: bytes | bytearray | memoryview | str | Path,
    family_nonce: bytes | bytearray | memoryview | str | Path,
    runtime: VerifiedRuntime,
    mesh: NestedMeshConfig,
    work_dir: str | Path,
    public_output_dir: str | Path,
    operator_output_dir: str | Path,
    timeout_seconds: float = 1_800.0,
    progress: Callable[[int, int], None] | None = None,
) -> HiddenCampaign2DResult:
    """Resume 100 ModEM solves and publish one separated 500-row campaign.

    ``work_dir`` and both output directories are new-only logical identities.
    Exact partial files are accepted only to resume the same committed work
    contract; differing or foreign bytes are never replaced.  The private
    completion manifest is published before either public file is exposed.
    """
    campaign = _identifier(campaign_id, "campaign_id")
    if (
        type(generator_seed) is not int
        or not 0 <= generator_seed <= np.iinfo(np.int64).max
    ):
        raise ValueError("generator_seed must be a non-negative int64")
    if not isinstance(mesh, NestedMeshConfig):
        raise TypeError("hidden campaign requires a protected NestedMeshConfig")
    key = _secret_bytes(sample_id_key, role="sample id key", exact=None)
    nonce = _secret_bytes(family_nonce, role="family reveal nonce", exact=32)
    geometry = load_campaign_geometry_2d(
        geometry_h5, expected_sha256=expected_geometry_sha256
    )
    generation_contract = generation_contract_2d(
        generator_seed=generator_seed, source_lineage=source_lineage
    )
    generation_sha = _canonical_object_sha256(generation_contract)
    runtime_manifest = validate_hidden_generation_runtime_manifest_2d(
        generation_runtime_manifest_path,
        expected_sha256=expected_generation_runtime_manifest_sha256,
        expected_size_bytes=expected_generation_runtime_manifest_size_bytes,
    )
    normalized_lineage = _strict_source_lineage(source_lineage)
    if runtime_manifest.value["source_closure"] != normalized_lineage:
        raise HiddenCampaign2DError(
            "runtime manifest source closure differs from campaign source lineage"
        )
    generation_runtime = {
        "python_version": runtime_manifest.value["python"]["version"],
        "numpy_version": runtime_manifest.value["distributions"]["numpy"]["version"],
        "pimsr_geogen_version": (
            runtime_manifest.value["distributions"]["pimsr_geogen"]["version"]
        ),
        "pimsr_forward_version": (
            runtime_manifest.value["distributions"]["pimsr_forward"]["version"]
        ),
    }
    if generation_runtime != FROZEN_GENERATION_RUNTIME:
        raise HiddenCampaign2DError(
            "runtime manifest headline versions differ from the frozen contract"
        )
    runtime.require_unchanged()
    if canonical_json_sha256(runtime.record) != runtime.identity_sha256:
        raise HiddenCampaign2DError("runtime record differs from its identity SHA-256")

    work_path = Path(work_dir).resolve(strict=False)
    public_path = Path(public_output_dir).resolve(strict=False)
    operator_path = Path(operator_output_dir).resolve(strict=False)
    if any(
        _paths_overlap(left, right)
        for left, right in (
            (work_path, public_path),
            (work_path, operator_path),
            (public_path, operator_path),
        )
    ):
        raise ValueError("work, public, and operator roots must not overlap")
    _preflight_existing_directory(
        public_path,
        role="public output directory",
        allowed_root_entries={"observations.npz", "observations.public.json"},
    )
    _preflight_existing_directory(
        operator_path,
        role="operator output directory",
        allowed_root_entries={
            "truth.npz",
            "operator.json",
            "family-reveal.json",
            "hidden-generation.json",
            "modem",
        },
    )
    if os.path.lexists(public_path) and not os.path.lexists(
        operator_path / "hidden-generation.json"
    ):
        raise HiddenCampaign2DError(
            "public output exists before private campaign completion"
        )
    contract = _work_contract(
        campaign_id=campaign,
        generator_seed=generator_seed,
        generation_contract_sha256=generation_sha,
        geometry=geometry,
        mesh=mesh,
        runtime=runtime,
        generation_runtime=generation_runtime,
        generation_runtime_manifest=runtime_manifest,
        sample_id_key=key,
        family_nonce=nonce,
    )
    work = _prepare_work_directory(work_path, contract)
    bases = _build_hidden_bases(
        generator_seed=generator_seed,
        campaign_id=campaign,
        sample_id_key=key,
        geometry=geometry,
    )
    forwards = _materialize_raw_forwards(
        bases=bases,
        work=work,
        campaign_id=campaign,
        generator_seed=generator_seed,
        generation_contract_sha256=generation_sha,
        geometry=geometry,
        mesh=mesh,
        runtime=runtime,
        timeout_seconds=timeout_seconds,
        progress=progress,
    )
    core = _build_core_campaign(
        campaign_id=campaign,
        generator_seed=generator_seed,
        family_nonce=nonce,
        geometry=geometry,
        mesh=mesh,
        bases=bases,
        forwards=forwards,
        public_directory=public_path,
        operator_directory=operator_path,
    )
    evidence = _build_final_evidence_plan(
        campaign_id=campaign,
        generator_seed=generator_seed,
        generation_contract=generation_contract,
        geometry=geometry,
        mesh=mesh,
        runtime=runtime,
        generation_runtime=generation_runtime,
        generation_runtime_manifest=runtime_manifest,
        bases=bases,
        forwards=forwards,
        core=core,
        operator_directory=operator_path,
    )
    runtime_manifest_before_publication = validate_hidden_generation_runtime_manifest_2d(
        generation_runtime_manifest_path,
        expected_sha256=expected_generation_runtime_manifest_sha256,
        expected_size_bytes=expected_generation_runtime_manifest_size_bytes,
    )
    if runtime_manifest_before_publication != runtime_manifest:
        raise HiddenCampaign2DError(
            "generation runtime manifest identity changed before publication"
        )
    runtime.require_unchanged()
    require_snapshot_unchanged(geometry.source, role="hidden campaign geometry source")
    _publish_operator_directory(
        operator_directory=operator_path,
        core=core,
        forwards=forwards,
        evidence=evidence,
    )
    _verify_published_operator_directory(
        operator_directory=operator_path, core=core, evidence=evidence
    )
    runtime.require_unchanged()
    runtime_manifest_before_public = validate_hidden_generation_runtime_manifest_2d(
        generation_runtime_manifest_path,
        expected_sha256=expected_generation_runtime_manifest_sha256,
        expected_size_bytes=expected_generation_runtime_manifest_size_bytes,
    )
    if runtime_manifest_before_public != runtime_manifest:
        raise HiddenCampaign2DError(
            "generation runtime manifest changed before public exposure"
        )
    _publish_public_directory(public_directory=public_path, core=core)
    _verify_published_operator_directory(
        operator_directory=operator_path, core=core, evidence=evidence
    )
    runtime.require_unchanged()
    runtime_manifest_at_end = validate_hidden_generation_runtime_manifest_2d(
        generation_runtime_manifest_path,
        expected_sha256=expected_generation_runtime_manifest_sha256,
        expected_size_bytes=expected_generation_runtime_manifest_size_bytes,
    )
    if runtime_manifest_at_end != runtime_manifest:
        raise HiddenCampaign2DError(
            "generation runtime manifest identity changed during publication"
        )
    require_snapshot_unchanged(geometry.source, role="hidden campaign geometry source")

    result = HiddenCampaign2DResult(
        campaign_id=campaign,
        public_directory=public_path.resolve(strict=True),
        operator_directory=operator_path.resolve(strict=True),
        observations=_file_identity(
            public_path / "observations.npz", role="published observations"
        ),
        public_manifest=_file_identity(
            public_path / "observations.public.json", role="published public manifest"
        ),
        truth=_file_identity(operator_path / "truth.npz", role="published truth"),
        operator_manifest=_file_identity(
            operator_path / "operator.json", role="published operator manifest"
        ),
        family_reveal=_file_identity(
            operator_path / "family-reveal.json", role="published family reveal"
        ),
        hidden_generation=_file_identity(
            operator_path / "hidden-generation.json",
            role="published hidden generation closure",
        ),
    )
    expected = (
        (result.observations, core.observations_identity),
        (result.public_manifest, core.public_manifest_identity),
        (result.truth, core.truth_identity),
        (result.operator_manifest, core.operator_manifest_identity),
        (result.family_reveal, core.family_reveal_identity),
        (result.hidden_generation, evidence.closure_identity),
    )
    if any(actual != wanted for actual, wanted in expected):
        raise HiddenCampaign2DError(
            "final publication receipt differs from planned bytes"
        )
    return result


__all__ = [
    "BASES_PER_FAMILY",
    "BASE_COUNT",
    "CAMPAIGN_SCHEMA",
    "CAMPAIGN_SCHEMA_VERSION",
    "FROZEN_GENERATION_RUNTIME",
    "FROZEN_RUNTIME_DISTRIBUTIONS",
    "GENERATION_CONTRACT_SCHEMA",
    "GENERATION_CONTRACT_SCHEMA_VERSION",
    "NOISE_REALIZATIONS_PER_BASE",
    "SAMPLE_COUNT",
    "GenerationRuntimeManifest2D",
    "HiddenCampaign2DError",
    "HiddenCampaign2DResult",
    "build_hidden_generation_runtime_manifest_2d",
    "generation_contract_2d",
    "load_campaign_geometry_2d",
    "materialize_hidden_campaign2d",
    "validate_hidden_generation_runtime_manifest_2d",
]
