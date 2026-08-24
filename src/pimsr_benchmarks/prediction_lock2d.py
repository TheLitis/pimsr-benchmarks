"""Pre-score cryptographic lock for the complete 2-D learned-method matrix.

The lock is deliberately built without opening an operator manifest, withheld
truth, or evaluation report.  Its externally recorded SHA-256 is the capability
that permits the evaluator to open truth later.  The module therefore keeps the
two phases structurally separate instead of relying on an operator convention.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
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

LOCK_INPUT_SCHEMA = "pimsr-sota-2d-prediction-lock-input"
LOCK_INPUT_SCHEMA_VERSION = 1
LOCK_SCHEMA = "pimsr-sota-2d-predictions-lock"
LOCK_SCHEMA_VERSION = 2
LOCK_AUDIENCE = "benchmark_prescore_prediction_artifacts"
PREREGISTRATION_SCHEMA = "pimsr-sota-2d-common-retrain-preregistration"
PREREGISTRATION_SCHEMA_VERSION = 1

METHOD_IDS = ("pimsr", "mtdlpy", "mt2dinv_densenet")
TRAINING_SEEDS = (101, 102, 103, 104, 105)
CAMPAIGN_COUNT = 5
SAMPLES_PER_CAMPAIGN = 500
RUN_COUNT = CAMPAIGN_COUNT * len(METHOD_IDS) * len(TRAINING_SEEDS)
GEOLOGICAL_FAMILIES = (
    "background",
    "aquifer",
    "hydrocarbon",
    "salt",
    "geothermal",
)
FAMILY_PARTITION_SCHEMA = "pimsr-sota-2d-family-partition-commitment"
FAMILY_PARTITION_SCHEMA_VERSION = 1
FAMILY_COMMITMENT_CONTRACT = {
    "algorithm": "SHA-256",
    "canonicalization": "utf8-canonical-json-sort-keys-compact-newline-v1",
    "domain_separator": "pimsr-sota-2d-family-partition/v1",
    "nonce_encoding": "lowercase_hex_32_bytes",
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$", flags=re.ASCII)
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$", flags=re.ASCII)
_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}$", flags=re.ASCII)
_FORBIDDEN_PRESCORE_TERMS = frozenset(
    {
        "evaluation",
        "evaluation_sha256",
        "generator_seed",
        "generator_seeds",
        "operator_manifest",
        "operator_scoring_manifest",
        "sample_id_key",
        "truth",
        "truth_path",
        "truth_sha256",
        "withheld_truth",
    }
)
_FALSE_PRESCORE_DECLARATIONS = frozenset(
    {
        "contains_truth",
        "heldout_truth_available_to_adapter",
        "truth_keys_accepted",
    }
)
_IDENTITY_SUFFIXES = (
    "_sha256",
    "_size_bytes",
    "_path",
    "_hash",
    "_digest",
    "_file",
    "_filename",
    "_uri",
    "_url",
)
_OBSERVATION_MEMBER_ORDER = (
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
_PREDICTION_MEMBER_ORDER = (
    "schema",
    "schema_version",
    "observations_sha256",
    "sample_index",
    "x_cell_centers_m",
    "depth_cell_centers_m",
    "predicted_log10_resistivity",
)
_LOCK_INPUT_KEYS = frozenset(
    {
        "audience",
        "campaigns",
        "preregistration_sha256",
        "schema",
        "schema_version",
    }
)
_CAMPAIGN_INPUT_KEYS = frozenset(
    {"campaign_id", "observation_manifest", "observations", "runs"}
)
_RUN_INPUT_KEYS = frozenset(
    {
        "checkpoint",
        "method_id",
        "prediction",
        "runtime",
        "source",
        "training_seed",
    }
)
_ARTIFACT_REFERENCE_KEYS = frozenset({"path", "sha256"})
_SOURCE_REFERENCE_KEYS = frozenset({"path", "repository_path", "sha256"})
_LOCK_KEYS = frozenset(
    {
        "audience",
        "campaigns",
        "design",
        "input_manifest",
        "locked",
        "preregistration",
        "runs",
        "schema",
        "schema_version",
        "statistical_options",
    }
)
_LOCK_RUN_KEYS = frozenset(
    {
        "adapter_source_sha256",
        "campaign_id",
        "checkpoint_sha256",
        "checkpoint_size_bytes",
        "method_id",
        "observation_manifest_sha256",
        "observations_sha256",
        "prediction_sha256",
        "prediction_size_bytes",
        "runtime_sha256",
        "runtime_size_bytes",
        "source_commit",
        "source_sha256",
        "training_seed",
    }
)
_STATISTICAL_OPTION_KEYS = frozenset(
    {
        "candidate_method_id",
        "confidence",
        "dominance_gate",
        "multiplicity_policy",
        "n_resamples",
        "point_aggregation",
        "reference_method_ids",
        "resampling_levels",
        "rng_seed",
    }
)
_POINT_AGGREGATION = (
    "equal_family_equal_base_equal_noise_mean_across_paired_training_seeds_and_campaigns"
)
_DOMINANCE_GATE = "one_sided_95_percent_iut_upper_below_zero_against_both_references"
_MULTIPLICITY_POLICY = (
    "none_for_single_intersection_union_claim_individual_pairwise_descriptive"
)
_RESAMPLING_LEVELS = (
    "training_seed",
    "campaign",
    "geological_family",
    "base_model_within_family",
    "noise_realization_within_base_model",
)

_PIMSR_RUNTIME_KEYS = frozenset(
    {
        "adapter_source",
        "checkpoint_contract",
        "comparison_status",
        "contains_truth",
        "execution",
        "heldout_truth_available_to_adapter",
        "inputs",
        "method",
        "observation_contract",
        "operation",
        "output",
        "prediction_contract",
        "ranking_allowed",
        "schema",
        "schema_version",
        "software",
        "source",
        "training_contract",
        "training_seed",
        "truth_keys_accepted",
    }
)
_MTDLPY_RUNTIME_KEYS = frozenset(
    {
        "adapter_wall_time_s",
        "bindings",
        "checkpoint_contract",
        "command",
        "comparison_status",
        "contains_truth",
        "dependency_closure",
        "determinism",
        "finished_at_utc",
        "method",
        "observation_contract",
        "operation",
        "outputs",
        "prediction_contract",
        "preprocessing",
        "ranking_allowed",
        "repository",
        "runtime",
        "schema",
        "schema_version",
        "seed",
        "source_artifacts",
        "started_at_utc",
        "track",
        "training_config",
        "training_summary",
        "truth_keys_accepted",
        "working_directory",
    }
)
_DENSENET_RUNTIME_KEYS = frozenset(
    {
        "adapter_wall_time_s",
        "bindings",
        "checkpoint_contract",
        "command",
        "comparison_status",
        "contains_truth",
        "dataset_identities",
        "dependency_closure",
        "finished_at_utc",
        "heldout_truth_available_to_adapter",
        "method",
        "method_id",
        "model",
        "observation_contract",
        "operation",
        "outputs",
        "prediction_contract",
        "preprocessing",
        "ranking_allowed",
        "repository",
        "runtime",
        "schema",
        "schema_version",
        "seed",
        "source_artifacts",
        "started_at_utc",
        "track",
        "training_config",
        "training_runtime",
        "training_seed",
        "training_summary",
        "truth_keys_accepted",
        "working_directory",
    }
)


class PredictionLock2DValidationError(ValueError):
    """Raised when pre-score evidence is incomplete or inconsistent."""


class PredictionLock2DPublicationError(RuntimeError):
    """Raised when a lock cannot be published without replacing a path."""


@dataclass(frozen=True)
class PredictionLock2DPublicationReceipt:
    """Bytes verified through the final, reopened publication descriptor."""

    path: Path
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class ArtifactSnapshot:
    """Bytes read through one stable regular-file descriptor."""

    path: Path
    payload: bytes
    sha256: str
    device: int
    inode: int

    @property
    def size_bytes(self) -> int:
        return len(self.payload)

    @property
    def identity(self) -> tuple[int, int]:
        return self.device, self.inode


@dataclass(frozen=True)
class LockedRun2D:
    campaign_id: str
    method_id: str
    training_seed: int
    observations_sha256: str
    observation_manifest_sha256: str
    prediction_sha256: str
    prediction_size_bytes: int
    runtime_sha256: str
    runtime_size_bytes: int
    checkpoint_sha256: str
    checkpoint_size_bytes: int
    source_commit: str
    source_sha256: str
    adapter_source_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "adapter_source_sha256": self.adapter_source_sha256,
            "campaign_id": self.campaign_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_size_bytes": self.checkpoint_size_bytes,
            "method_id": self.method_id,
            "observation_manifest_sha256": self.observation_manifest_sha256,
            "observations_sha256": self.observations_sha256,
            "prediction_sha256": self.prediction_sha256,
            "prediction_size_bytes": self.prediction_size_bytes,
            "runtime_sha256": self.runtime_sha256,
            "runtime_size_bytes": self.runtime_size_bytes,
            "source_commit": self.source_commit,
            "source_sha256": self.source_sha256,
            "training_seed": self.training_seed,
        }


@dataclass(frozen=True)
class ValidatedPredictionLock2D:
    preregistration_sha256: str
    lock_sha256: str
    input_manifest_sha256: str
    campaign_ids: tuple[str, ...]
    method_ids: tuple[str, ...]
    training_seeds: tuple[int, ...]
    statistical_options: Mapping[str, Any]
    runs: tuple[LockedRun2D, ...]

    def require_run(
        self, campaign_id: str, method_id: str, training_seed: int
    ) -> LockedRun2D:
        wanted = (campaign_id, method_id, training_seed)
        for run in self.runs:
            if (run.campaign_id, run.method_id, run.training_seed) == wanted:
                return run
        raise PredictionLock2DValidationError(
            "prediction lock has no run for "
            f"campaign={campaign_id!r}, method={method_id!r}, seed={training_seed!r}"
        )


@dataclass(frozen=True)
class ValidatedLockedArtifacts2D:
    observations: ArtifactSnapshot
    observation_manifest: ArtifactSnapshot
    prediction: ArtifactSnapshot
    runtime: ArtifactSnapshot
    checkpoint: ArtifactSnapshot
    source: ArtifactSnapshot


@dataclass(frozen=True)
class _Preregistration:
    snapshot: ArtifactSnapshot
    value: Mapping[str, Any]
    campaign_ids: tuple[str, ...]
    method_by_id: Mapping[str, Mapping[str, Any]]
    train_sha256: str
    validation_sha256: str
    statistical_options: Mapping[str, Any]
    family_partition: Mapping[str, Any]


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], path: str) -> None:
    actual = set(value)
    if actual != expected:
        raise PredictionLock2DValidationError(
            f"{path} keys mismatch; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise PredictionLock2DValidationError(
            f"{path} must be an object with string keys"
        )
    return value


def _sequence(value: Any, path: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise PredictionLock2DValidationError(f"{path} must be an array")
    return value


def _identifier(value: Any, path: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise PredictionLock2DValidationError(f"{path} is not a portable identifier")
    return value


def _integer(
    value: Any, path: str, *, minimum: int = 0, maximum: int | None = None
) -> int:
    if (
        type(value) is not int
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        bound = (
            f"between {minimum} and {maximum}"
            if maximum is not None
            else f"greater than or equal to {minimum}"
        )
        raise PredictionLock2DValidationError(f"{path} must be an integer {bound}")
    return value


def _finite(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PredictionLock2DValidationError(f"{path} must be a finite JSON number")
    result = float(value)
    if not np.isfinite(result):
        raise PredictionLock2DValidationError(f"{path} must be finite")
    return result


def _sha256(value: Any, path: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise PredictionLock2DValidationError(
            f"{path} must be 64 lowercase hexadecimal characters"
        )
    return value


def _git_commit(value: Any, path: str) -> str:
    if not isinstance(value, str) or not _GIT_COMMIT_RE.fullmatch(value):
        raise PredictionLock2DValidationError(
            f"{path} must be a full lowercase Git commit"
        )
    return value


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Return finite, sorted, compact, newline-terminated UTF-8 JSON."""
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise PredictionLock2DPublicationError(
            f"lock data is not canonical finite JSON: {exc}"
        ) from exc
    return (text + "\n").encode("utf-8")


def _canonical_object_sha256(value: Any) -> str:
    """Match the adapter dependency-closure digest without accepting NaN."""
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PredictionLock2DValidationError(
            f"dependency closure is not canonical finite JSON: {exc}"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def snapshot_regular_file(
    path: str | Path, *, expected_sha256: str | None, role: str
) -> ArtifactSnapshot:
    """Read one inode through one fd and reject links and replacement races."""
    source = Path(path)
    expected = None if expected_sha256 is None else _sha256(expected_sha256, role)
    try:
        before = os.lstat(source)
    except OSError as exc:
        raise PredictionLock2DValidationError(
            f"cannot lstat {role} {source}: {exc}"
        ) from exc
    if not stat.S_ISREG(before.st_mode):
        raise PredictionLock2DValidationError(f"{role} must be a regular non-link file")
    flags = os.O_RDONLY
    flags |= getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(source, flags)
        opened = os.fstat(descriptor)
        before_identity = (int(before.st_dev), int(before.st_ino))
        opened_identity = (int(opened.st_dev), int(opened.st_ino))
        if not stat.S_ISREG(opened.st_mode) or opened_identity != before_identity:
            raise PredictionLock2DValidationError(f"{role} changed before it was opened")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after_fd = os.fstat(descriptor)
        payload = b"".join(chunks)
    except PredictionLock2DValidationError:
        raise
    except OSError as exc:
        raise PredictionLock2DValidationError(
            f"cannot read {role} {source}: {exc}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    try:
        after_path = os.lstat(source)
    except OSError as exc:
        raise PredictionLock2DValidationError(
            f"cannot re-lstat {role} {source}: {exc}"
        ) from exc
    signatures = (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns),
        (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns),
        (after_fd.st_dev, after_fd.st_ino, after_fd.st_size, after_fd.st_mtime_ns),
        (
            after_path.st_dev,
            after_path.st_ino,
            after_path.st_size,
            after_path.st_mtime_ns,
        ),
    )
    if len(set(signatures)) != 1 or len(payload) != int(opened.st_size):
        raise PredictionLock2DValidationError(f"{role} changed while it was read")
    digest = hashlib.sha256(payload).hexdigest()
    if expected is not None and digest != expected:
        raise PredictionLock2DValidationError(f"{role} SHA-256 differs from its pin")
    return ArtifactSnapshot(
        path=source,
        payload=payload,
        sha256=digest,
        device=int(opened.st_dev),
        inode=int(opened.st_ino),
    )


def _strict_json(snapshot: ArtifactSnapshot, role: str) -> Mapping[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PredictionLock2DValidationError(
                    f"{role} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise PredictionLock2DValidationError(
            f"{role} contains non-finite JSON constant {value!r}"
        )

    try:
        value = json.loads(
            snapshot.payload.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except PredictionLock2DValidationError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PredictionLock2DValidationError(f"cannot decode {role}: {exc}") from exc
    return _mapping(value, role)


def _normalized_metadata_key(key: str) -> str:
    snake = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key)
    return re.sub(r"[^A-Za-z0-9]+", "_", snake).strip("_").lower()


def _forbidden_prescore_key(normalized: str) -> bool:
    identity_base = normalized
    stripped = True
    while stripped:
        stripped = False
        for suffix in _IDENTITY_SUFFIXES:
            if identity_base.endswith(suffix):
                identity_base = identity_base[: -len(suffix)]
                stripped = True
                break
    terms = frozenset(term for term in identity_base.split("_") if term)
    compact = identity_base.replace("_", "")
    return (
        normalized in _FORBIDDEN_PRESCORE_TERMS
        or identity_base in _FORBIDDEN_PRESCORE_TERMS
        or bool(terms & {"hidden", "secret", "blind"})
        or {"ground", "truth"}.issubset(terms)
        or {"operator", "manifest"}.issubset(terms)
        or {"generator", "seed"}.issubset(terms)
        or {"withheld", "truth"}.issubset(terms)
        or compact.startswith(("groundtruth", "operatormanifest", "generatorseed"))
    )


def _reject_prescore_secrets(value: Any, path: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = _normalized_metadata_key(key)
            if normalized in _FALSE_PRESCORE_DECLARATIONS:
                if child is not False:
                    raise PredictionLock2DValidationError(
                        f"{path}.{key} must be explicitly false before scoring"
                    )
                continue
            if _forbidden_prescore_key(normalized):
                raise PredictionLock2DValidationError(
                    f"{path} contains forbidden pre-score key {key!r}"
                )
            _reject_prescore_secrets(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _reject_prescore_secrets(child, f"{path}[{index}]")


def _artifact_reference(
    value: Any,
    *,
    base: Path,
    role: str,
    source: bool = False,
) -> tuple[ArtifactSnapshot, Path | None]:
    reference = _mapping(value, role)
    _exact_keys(
        reference,
        _SOURCE_REFERENCE_KEYS if source else _ARTIFACT_REFERENCE_KEYS,
        role,
    )
    path_text = reference["path"]
    if not isinstance(path_text, str) or not path_text or "\x00" in path_text:
        raise PredictionLock2DValidationError(f"{role}.path is invalid")
    requested = Path(path_text)
    path = requested if requested.is_absolute() else base / requested
    snapshot = snapshot_regular_file(
        path,
        expected_sha256=_sha256(reference["sha256"], f"{role}.sha256"),
        role=role,
    )
    repository: Path | None = None
    if source:
        repository_text = reference["repository_path"]
        if (
            not isinstance(repository_text, str)
            or not repository_text
            or "\x00" in repository_text
        ):
            raise PredictionLock2DValidationError(f"{role}.repository_path is invalid")
        repository_requested = Path(repository_text)
        repository = (
            repository_requested
            if repository_requested.is_absolute()
            else base / repository_requested
        ).resolve(strict=True)
    return snapshot, repository


def _statistical_options(prereg: Mapping[str, Any]) -> Mapping[str, Any]:
    analysis = _mapping(prereg.get("statistical_analysis"), "prereg.statistical_analysis")
    effect = _mapping(analysis.get("effect"), "prereg.statistical_analysis.effect")
    bootstrap = _mapping(
        analysis.get("hierarchical_paired_bootstrap"),
        "prereg.statistical_analysis.hierarchical_paired_bootstrap",
    )
    candidate = _identifier(effect.get("candidate"), "prereg effect candidate")
    raw_references = effect.get("references")
    if raw_references is None:
        legacy = effect.get("reference")
        raw_references = [] if legacy is None else [legacy]
    references = [
        _identifier(value, f"prereg effect references[{index}]")
        for index, value in enumerate(
            _sequence(raw_references, "prereg effect references")
        )
    ]
    if candidate != "pimsr" or references != ["mtdlpy", "mt2dinv_densenet"]:
        raise PredictionLock2DValidationError(
            "prereg effect must compare pimsr against mtdlpy and mt2dinv_densenet"
        )
    confidence = _finite(bootstrap.get("confidence"), "prereg bootstrap confidence")
    if confidence != 0.95:
        raise PredictionLock2DValidationError(
            "prereg bootstrap confidence must be exactly 0.95"
        )
    levels = list(
        _sequence(bootstrap.get("resampling_levels"), "prereg bootstrap levels")
    )
    if levels != list(_RESAMPLING_LEVELS):
        raise PredictionLock2DValidationError(
            "prereg bootstrap levels differ from the frozen paired hierarchy"
        )
    point = bootstrap.get("point_aggregation", bootstrap.get("repeat_aggregation"))
    dominance = analysis.get("dominance_gate", effect.get("dominance_gate"))
    multiplicity = analysis.get("multiplicity_policy", effect.get("multiplicity_policy"))
    expected_policies = (
        (point, _POINT_AGGREGATION, "point_aggregation"),
        (dominance, _DOMINANCE_GATE, "dominance_gate"),
        (multiplicity, _MULTIPLICITY_POLICY, "multiplicity_policy"),
    )
    for value, expected, path in expected_policies:
        if value != expected:
            raise PredictionLock2DValidationError(
                f"prereg statistical_analysis {path} differs from the frozen policy"
            )
    result = {
        "candidate_method_id": candidate,
        "confidence": confidence,
        "dominance_gate": dominance,
        "multiplicity_policy": multiplicity,
        "n_resamples": _integer(
            bootstrap.get("n_resamples"), "prereg bootstrap n_resamples", minimum=10_000
        ),
        "point_aggregation": point,
        "reference_method_ids": references,
        "resampling_levels": levels,
        "rng_seed": _integer(
            bootstrap.get("rng_seed"),
            "prereg bootstrap rng_seed",
            minimum=0,
            maximum=int(np.iinfo(np.uint64).max),
        ),
    }
    _exact_keys(result, _STATISTICAL_OPTION_KEYS, "statistical_options")
    return result


def _family_partition_policy(prereg: Mapping[str, Any]) -> Mapping[str, Any]:
    value = _mapping(prereg.get("family_partition"), "prereg.family_partition")
    _exact_keys(
        value,
        frozenset(
            {
                "bases_per_family",
                "commitment_contract",
                "families",
                "noise_realizations_per_base",
                "schema",
                "schema_version",
            }
        ),
        "prereg.family_partition",
    )
    contract = _mapping(
        value.get("commitment_contract"),
        "prereg.family_partition.commitment_contract",
    )
    _exact_keys(
        contract,
        frozenset(FAMILY_COMMITMENT_CONTRACT),
        "prereg.family_partition.commitment_contract",
    )
    if (
        value.get("schema") != FAMILY_PARTITION_SCHEMA
        or value.get("schema_version") != FAMILY_PARTITION_SCHEMA_VERSION
        or list(_sequence(value.get("families"), "prereg family list"))
        != list(GEOLOGICAL_FAMILIES)
        or value.get("bases_per_family") != 20
        or value.get("noise_realizations_per_base") != 5
        or dict(contract) != FAMILY_COMMITMENT_CONTRACT
    ):
        raise PredictionLock2DValidationError(
            "prereg family partition differs from the frozen 5x20x5 design"
        )
    return dict(value)


def _load_preregistration(path: str | Path, expected_sha256: str) -> _Preregistration:
    snapshot = snapshot_regular_file(
        path,
        expected_sha256=_sha256(expected_sha256, "expected preregistration SHA-256"),
        role="preregistration",
    )
    value = _strict_json(snapshot, "preregistration")
    if (
        value.get("schema") != PREREGISTRATION_SCHEMA
        or value.get("schema_version") != PREREGISTRATION_SCHEMA_VERSION
    ):
        raise PredictionLock2DValidationError("preregistration schema identity is wrong")
    seeds = tuple(
        _integer(seed, f"prereg.run_seeds[{index}]")
        for index, seed in enumerate(
            _sequence(value.get("run_seeds"), "prereg.run_seeds")
        )
    )
    if seeds != TRAINING_SEEDS:
        raise PredictionLock2DValidationError("preregistration run seeds are not exact")
    datasets = _mapping(value.get("datasets"), "prereg.datasets")
    hidden = _mapping(datasets.get("hidden_test"), "prereg.datasets.hidden_test")
    campaigns = _mapping(hidden.get("campaigns"), "prereg hidden campaigns")
    if (
        campaigns.get("count") != CAMPAIGN_COUNT
        or campaigns.get("samples_per_campaign") != SAMPLES_PER_CAMPAIGN
        or campaigns.get("total_samples") != CAMPAIGN_COUNT * SAMPLES_PER_CAMPAIGN
    ):
        raise PredictionLock2DValidationError(
            "prereg hidden observation budget is not exact"
        )
    ids_raw = campaigns.get("campaign_ids")
    if ids_raw is None:
        raise PredictionLock2DValidationError(
            "prereg hidden campaigns require an explicit campaign_ids array before locking"
        )
    campaign_ids = tuple(
        _identifier(item, f"prereg campaign_ids[{index}]")
        for index, item in enumerate(_sequence(ids_raw, "prereg campaign_ids"))
    )
    if len(campaign_ids) != CAMPAIGN_COUNT or len(set(campaign_ids)) != CAMPAIGN_COUNT:
        raise PredictionLock2DValidationError(
            "prereg campaign_ids must be five unique ids"
        )
    gate = _mapping(hidden.get("prediction_lock_gate"), "prereg prediction_lock_gate")
    if gate.get("locked_artifact_count") != RUN_COUNT:
        raise PredictionLock2DValidationError(
            f"prereg prediction lock must declare exactly {RUN_COUNT} run cells"
        )
    methods_raw = _sequence(value.get("methods"), "prereg.methods")
    method_by_id: dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(methods_raw):
        method = _mapping(item, f"prereg.methods[{index}]")
        method_id = _identifier(method.get("id"), f"prereg.methods[{index}].id")
        if method_id in method_by_id:
            raise PredictionLock2DValidationError("prereg methods contain duplicate ids")
        _mapping(
            method.get("implementation"), f"prereg method {method_id}.implementation"
        )
        _mapping(method.get("training"), f"prereg method {method_id}.training")
        method_by_id[method_id] = method
    if tuple(method_by_id) != METHOD_IDS:
        raise PredictionLock2DValidationError(
            f"prereg methods must be ordered exactly as {METHOD_IDS!r}"
        )
    train = _mapping(
        _mapping(datasets.get("train"), "prereg train").get("artifact"), "train artifact"
    )
    validation = _mapping(
        _mapping(datasets.get("validation"), "prereg validation").get("artifact"),
        "validation artifact",
    )
    return _Preregistration(
        snapshot=snapshot,
        value=value,
        campaign_ids=campaign_ids,
        method_by_id=method_by_id,
        train_sha256=_sha256(train.get("sha256"), "prereg train sha256"),
        validation_sha256=_sha256(validation.get("sha256"), "prereg validation sha256"),
        statistical_options=_statistical_options(value),
        family_partition=_family_partition_policy(value),
    )


def _npz_arrays(
    snapshot: ArtifactSnapshot,
    *,
    member_order: Sequence[str],
    role: str,
) -> dict[str, np.ndarray]:
    expected_names = [f"{name}.npy" for name in member_order]
    try:
        with zipfile.ZipFile(io.BytesIO(snapshot.payload), "r") as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if names != expected_names or len(names) != len(set(names)):
                raise PredictionLock2DValidationError(
                    f"{role} NPZ member set/order is not exact"
                )
            if any(member.flag_bits & 0x1 for member in members):
                raise PredictionLock2DValidationError(f"{role} NPZ is encrypted")
            if archive.testzip() is not None:
                raise PredictionLock2DValidationError(f"{role} NPZ is corrupt")
        with np.load(io.BytesIO(snapshot.payload), allow_pickle=False) as archive:
            return {name: archive[name] for name in member_order}
    except PredictionLock2DValidationError:
        raise
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
        raise PredictionLock2DValidationError(f"cannot decode {role} NPZ: {exc}") from exc


def _scalar_text(array: np.ndarray, expected: str, path: str) -> None:
    if array.ndim != 0 or array.dtype.kind != "U" or array.item() != expected:
        raise PredictionLock2DValidationError(f"{path} identity is wrong")


def _observation_identity(
    snapshot: ArtifactSnapshot,
    manifest: Mapping[str, Any],
    campaign_id: str,
    family_partition: Mapping[str, Any],
) -> tuple[int, tuple[int, ...], np.ndarray, np.ndarray]:
    arrays = _npz_arrays(
        snapshot, member_order=_OBSERVATION_MEMBER_ORDER, role="public observations"
    )
    _scalar_text(arrays["schema"], "pimsr-sota-2d-observations", "observations.schema")
    if (
        arrays["schema_version"].ndim != 0
        or arrays["schema_version"].dtype != np.dtype("<i8")
        or int(arrays["schema_version"].item()) != 1
    ):
        raise PredictionLock2DValidationError("observations.schema_version is wrong")
    sample_ids = arrays["sample_index"]
    if (
        sample_ids.dtype != np.dtype("<i8")
        or sample_ids.ndim != 1
        or sample_ids.size != SAMPLES_PER_CAMPAIGN
        or np.any(sample_ids < 0)
        or np.unique(sample_ids).size != sample_ids.size
    ):
        raise PredictionLock2DValidationError(
            f"each observation campaign must have exactly {SAMPLES_PER_CAMPAIGN} unique ids"
        )
    axes: dict[str, np.ndarray] = {}
    for name, positive in (
        ("frequency_hz", True),
        ("station_x_m", False),
        ("x_cell_centers_m", False),
        ("depth_cell_centers_m", True),
    ):
        axis = arrays[name]
        if (
            axis.dtype != np.dtype("<f8")
            or axis.ndim != 1
            or axis.size < 2
            or not np.isfinite(axis).all()
            or np.any(np.diff(axis) <= 0)
            or (positive and np.any(axis <= 0))
        ):
            raise PredictionLock2DValidationError(
                f"observations.{name} must be a finite increasing float64 axis"
            )
        axes[name] = axis
    if (
        axes["station_x_m"][0] < axes["x_cell_centers_m"][0]
        or axes["station_x_m"][-1] > axes["x_cell_centers_m"][-1]
    ):
        raise PredictionLock2DValidationError(
            "observation stations must lie inside the model x axis"
        )
    channel_order = arrays["observation_channel_order"]
    if (
        channel_order.ndim != 1
        or channel_order.dtype.kind != "U"
        or tuple(channel_order.tolist())
        != (
            "log10_rho_te",
            "phase_te_degrees",
            "log10_rho_tm",
            "phase_tm_degrees",
        )
    ):
        raise PredictionLock2DValidationError(
            "observation channel order is not the exact TE/TM contract"
        )
    response_shape = (
        SAMPLES_PER_CAMPAIGN,
        axes["frequency_hz"].size,
        axes["station_x_m"].size,
    )
    for name in (
        "observed_log10_rho_te",
        "observed_phase_te_degrees",
        "observed_log10_rho_tm",
        "observed_phase_tm_degrees",
    ):
        values = arrays[name]
        if (
            values.dtype != np.dtype("<f4")
            or values.shape != response_shape
            or not np.isfinite(values).all()
        ):
            raise PredictionLock2DValidationError(
                f"observations.{name} must be finite float32 with shape {response_shape}"
            )
        if "phase" in name and np.any((values < 0.0) | (values >= 180.0)):
            raise PredictionLock2DValidationError(
                f"observations.{name} violates the [0, 180) phase convention"
            )
    for name in (
        "declared_evaluation_floor_log10_rho_te",
        "declared_evaluation_floor_phase_te_degrees",
        "declared_evaluation_floor_log10_rho_tm",
        "declared_evaluation_floor_phase_tm_degrees",
    ):
        values = arrays[name]
        if (
            values.dtype != np.dtype("<f4")
            or values.shape != response_shape
            or not np.isfinite(values).all()
            or np.any(values <= 0)
        ):
            raise PredictionLock2DValidationError(
                f"observations.{name} must be positive finite float32 with shape "
                f"{response_shape}"
            )
    valid_mask = arrays["valid_mask"]
    expected_mask_shape = (
        SAMPLES_PER_CAMPAIGN,
        4,
        axes["frequency_hz"].size,
        axes["station_x_m"].size,
    )
    if (
        valid_mask.dtype != np.dtype(np.bool_)
        or valid_mask.shape != expected_mask_shape
        or not np.all(valid_mask)
    ):
        raise PredictionLock2DValidationError(
            "observations.valid_mask must be all-true bool with shape "
            f"{expected_mask_shape}"
        )
    _exact_keys(
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
        or manifest["schema"] != "pimsr-sota-2d-observation-manifest"
        or manifest["schema_version"] != 3
        or manifest["split_id"] != campaign_id
        or manifest["sample_count"] != SAMPLES_PER_CAMPAIGN
    ):
        raise PredictionLock2DValidationError(
            "public observation manifest identity is wrong"
        )
    commitment = _mapping(
        manifest["family_partition_commitment"],
        "public observation family_partition_commitment",
    )
    _exact_keys(
        commitment,
        frozenset({"contract", "schema", "schema_version", "sha256"}),
        "public observation family_partition_commitment",
    )
    commitment_contract = _mapping(
        commitment["contract"],
        "public observation family_partition_commitment.contract",
    )
    prereg_contract = _mapping(
        family_partition["commitment_contract"],
        "prereg family_partition.commitment_contract",
    )
    _sha256(
        commitment["sha256"],
        "public observation family partition commitment SHA-256",
    )
    if (
        commitment["schema"] != FAMILY_PARTITION_SCHEMA
        or commitment["schema_version"] != FAMILY_PARTITION_SCHEMA_VERSION
        or dict(commitment_contract) != dict(prereg_contract)
    ):
        raise PredictionLock2DValidationError(
            "public observation family partition commitment differs from preregistration"
        )
    payload = _mapping(manifest["observation_payload"], "observation_payload")
    _exact_keys(
        payload,
        frozenset(
            {"arrays", "media_type", "schema", "schema_version", "sha256", "size_bytes"}
        ),
        "observation_payload",
    )
    if (
        payload["media_type"] != "application/x-npz"
        or payload["schema"] != "pimsr-sota-2d-observations"
        or payload["schema_version"] != 1
        or payload["sha256"] != snapshot.sha256
        or payload["size_bytes"] != snapshot.size_bytes
    ):
        raise PredictionLock2DValidationError(
            "public manifest does not bind the exact observation payload"
        )
    if set(_mapping(payload["arrays"], "observation_payload.arrays")) != set(
        _OBSERVATION_MEMBER_ORDER
    ):
        raise PredictionLock2DValidationError(
            "public manifest array declarations are not exact"
        )
    _reject_prescore_secrets(manifest, "public observation manifest")
    return (
        int(sample_ids.size),
        tuple(sorted(int(value) for value in sample_ids)),
        axes["x_cell_centers_m"],
        axes["depth_cell_centers_m"],
    )


def _prediction_identity(
    snapshot: ArtifactSnapshot,
    *,
    observations_sha256: str,
    sample_ids: tuple[int, ...],
    x_cell_centers_m: np.ndarray,
    depth_cell_centers_m: np.ndarray,
) -> None:
    arrays = _npz_arrays(
        snapshot, member_order=_PREDICTION_MEMBER_ORDER, role="prediction"
    )
    _scalar_text(arrays["schema"], "pimsr-sota-2d-predictions", "prediction.schema")
    version = arrays["schema_version"]
    if version.ndim != 0 or version.dtype != np.dtype("<i8") or int(version.item()) != 2:
        raise PredictionLock2DValidationError("prediction.schema_version is wrong")
    observation = arrays["observations_sha256"]
    if (
        observation.ndim != 0
        or observation.dtype != np.dtype("<U64")
        or str(observation.item()) != observations_sha256
    ):
        raise PredictionLock2DValidationError("prediction observations binding is wrong")
    prediction_ids = arrays["sample_index"]
    if prediction_ids.dtype != np.dtype("<i8") or prediction_ids.ndim != 1:
        raise PredictionLock2DValidationError("prediction sample ids have wrong type")
    if tuple(sorted(int(value) for value in prediction_ids)) != sample_ids:
        raise PredictionLock2DValidationError(
            "prediction sample ids differ from observations"
        )
    for name, expected_axis in (
        ("x_cell_centers_m", x_cell_centers_m),
        ("depth_cell_centers_m", depth_cell_centers_m),
    ):
        axis = arrays[name]
        if (
            axis.dtype != np.dtype("<f8")
            or axis.shape != expected_axis.shape
            or not np.array_equal(axis, expected_axis)
        ):
            raise PredictionLock2DValidationError(
                f"prediction {name} differs from public observations"
            )
    values = arrays["predicted_log10_resistivity"]
    expected_shape = (
        len(sample_ids),
        depth_cell_centers_m.size,
        x_cell_centers_m.size,
    )
    if values.dtype != np.dtype("<f4") or values.shape != expected_shape:
        raise PredictionLock2DValidationError("prediction grid has wrong shape or dtype")
    if not np.isfinite(values).all():
        raise PredictionLock2DValidationError(
            "prediction grid contains non-finite values"
        )


def _git_output(repository: Path, *arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
        raise PredictionLock2DValidationError(
            f"cannot validate source repository {repository}: {exc}"
        ) from exc


def _git_bytes(
    repository: Path, *arguments: str, input_payload: bytes | None = None
) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            input=input_payload,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        detail = getattr(exc, "stderr", b"")
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", errors="replace")
        raise PredictionLock2DValidationError(
            f"cannot validate source repository {repository}: {str(detail).strip()}"
        ) from exc


def _validate_source_repository(
    source: ArtifactSnapshot,
    repository: Path,
    *,
    expected_commit: str,
    method_id: str,
    allow_descendant_head: bool = False,
    protected_paths: Sequence[str] = (),
) -> None:
    top = Path(_git_output(repository, "rev-parse", "--show-toplevel")).resolve(
        strict=True
    )
    if top != repository:
        raise PredictionLock2DValidationError(
            f"{method_id} source repository_path is not the exact Git root"
        )
    head_commit = _git_output(repository, "rev-parse", "HEAD")
    if allow_descendant_head:
        try:
            _git_output(
                repository,
                "merge-base",
                "--is-ancestor",
                expected_commit,
                head_commit,
            )
        except PredictionLock2DValidationError as exc:
            raise PredictionLock2DValidationError(
                f"{method_id} repository HEAD is not a descendant of its pinned commit"
            ) from exc
    elif head_commit != expected_commit:
        raise PredictionLock2DValidationError(
            f"{method_id} source commit is not preregistered"
        )
    if _git_output(repository, "status", "--porcelain=v1", "--untracked-files=normal"):
        raise PredictionLock2DValidationError(f"{method_id} source repository is dirty")
    try:
        relative = source.path.resolve(strict=True).relative_to(repository).as_posix()
    except ValueError as exc:
        raise PredictionLock2DValidationError(
            f"{method_id} source artifact is outside its repository"
        ) from exc
    tracked = _git_output(
        repository, "ls-files", "--error-unmatch", "--", relative
    ).replace("\\", "/")
    if tracked != relative:
        raise PredictionLock2DValidationError(
            f"{method_id} source artifact is not uniquely tracked"
        )
    tree_fields = _git_output(
        repository, "ls-tree", expected_commit, "--", relative
    ).split(maxsplit=3)
    if (
        len(tree_fields) != 4
        or tree_fields[0] not in {"100644", "100755"}
        or tree_fields[1] != "blob"
        or not re.fullmatch(r"[0-9a-f]{40,64}", tree_fields[2])
        or tree_fields[3].replace("\\", "/") != relative
    ):
        raise PredictionLock2DValidationError(
            f"{method_id} pinned source is not a regular Git blob"
        )
    try:
        filtered_object_id = (
            _git_bytes(
                repository,
                "hash-object",
                "--stdin",
                "--path",
                relative,
                input_payload=source.payload,
            )
            .decode("ascii", errors="strict")
            .strip()
        )
    except UnicodeError as exc:
        raise PredictionLock2DValidationError(
            f"{method_id} Git returned a non-ASCII source object id"
        ) from exc
    if filtered_object_id != tree_fields[2]:
        raise PredictionLock2DValidationError(
            f"{method_id} captured source bytes do not match the pinned commit blob"
        )
    if allow_descendant_head and head_commit != expected_commit:
        scopes = tuple(dict.fromkeys((relative, *protected_paths)))
        changed = _git_output(
            repository,
            "diff",
            "--name-only",
            expected_commit,
            head_commit,
            "--",
            *scopes,
        )
        if changed:
            raise PredictionLock2DValidationError(
                f"{method_id} source artifact changed after its pinned commit"
            )
    current = snapshot_regular_file(
        source.path,
        expected_sha256=source.sha256,
        role=f"{method_id} source artifact after Git validation",
    )
    if current.identity != source.identity or current.payload != source.payload:
        raise PredictionLock2DValidationError(
            f"{method_id} source artifact changed during Git validation"
        )


def _artifact_digest(record: Any, path: str) -> tuple[str, int | None]:
    value = _mapping(record, path)
    digest = _sha256(value.get("sha256"), f"{path}.sha256")
    size_raw = value.get("size_bytes")
    size = (
        None if size_raw is None else _integer(size_raw, f"{path}.size_bytes", minimum=1)
    )
    return digest, size


def _exact_object(value: Any, keys: frozenset[str], path: str) -> Mapping[str, Any]:
    result = _mapping(value, path)
    _exact_keys(result, keys, path)
    return result


def _validate_pimsr_training(
    expected: Mapping[str, Any], actual: Mapping[str, Any], training_seed: int
) -> None:
    _exact_keys(
        actual,
        frozenset(
            {
                "batch_size",
                "beta_nll",
                "class_weights",
                "epochs",
                "gradient_clip_norm",
                "learning_rate",
                "loss",
                "normalization",
                "optimizer",
                "runtime",
                "scheduler",
                "scheduler_t_max",
                "seed",
                "sigma_regularization",
                "sigma_warmup",
                "validation_loss",
                "weight_decay",
                "workers",
            }
        ),
        "pimsr resolved training_config",
    )
    optimizer = _exact_object(
        expected.get("optimizer"),
        frozenset({"betas", "eps", "learning_rate", "name", "weight_decay"}),
        "prereg pimsr optimizer",
    )
    scheduler = _exact_object(
        expected.get("scheduler"),
        frozenset({"eta_min", "name", "step_timing", "t_max"}),
        "prereg pimsr scheduler",
    )
    loss = _exact_object(
        expected.get("loss"),
        frozenset(
            {
                "scenario_cross_entropy_weight",
                "sigma_epochs_15_through_79",
                "total_variation_weight",
                "validation",
                "warmup_epochs_0_through_14",
            }
        ),
        "prereg pimsr loss",
    )
    resolved = {
        "batch_size": expected.get("batch_size"),
        "beta_nll": 0.5,
        "epochs": expected.get("epochs"),
        "gradient_clip_norm": expected.get("gradient_clip_max_norm"),
        "learning_rate": optimizer.get("learning_rate"),
        "loss": "beta_nll+tv0.05+scenario_ce0.1/v1",
        "normalization": "per-channel-train-mean-std/v1",
        "optimizer": "torch.optim.AdamW",
        "scheduler": "CosineAnnealingLR",
        "scheduler_t_max": scheduler.get("t_max"),
        "seed": training_seed,
        "sigma_regularization": 0.05,
        "sigma_warmup": 15,
        "validation_loss": "plain_nll+tv0.05+scenario_ce0.1/v1",
        "weight_decay": optimizer.get("weight_decay"),
        "workers": expected.get("workers"),
    }
    for key, expected_value in resolved.items():
        if actual.get(key) != expected_value:
            raise PredictionLock2DValidationError(
                f"pimsr resolved training_config.{key} differs from preregistration"
            )
    if optimizer != {
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "learning_rate": 0.0003,
        "name": "AdamW",
        "weight_decay": 0.0001,
    }:
        raise PredictionLock2DValidationError("prereg pimsr optimizer is not frozen")
    if scheduler != {
        "eta_min": 0.0,
        "name": "CosineAnnealingLR",
        "step_timing": "after_each_epoch",
        "t_max": 80,
    }:
        raise PredictionLock2DValidationError("prereg pimsr scheduler is not frozen")
    if loss != {
        "scenario_cross_entropy_weight": 0.1,
        "sigma_epochs_15_through_79": "beta_nll_0.5_plus_log_sigma_l2_0.05",
        "total_variation_weight": 0.05,
        "validation": "plain_nll_plus_tv_0.05_plus_scenario_ce_0.1",
        "warmup_epochs_0_through_14": (
            "half_mean_squared_error_plus_tv_0.05_plus_scenario_ce_0.1"
        ),
    }:
        raise PredictionLock2DValidationError("prereg pimsr loss schedule is not frozen")
    class_counts_raw = expected.get("class_counts")
    if (
        not isinstance(class_counts_raw, list)
        or len(class_counts_raw) != 5
        or any(type(value) is not int or value <= 0 for value in class_counts_raw)
        or sum(class_counts_raw) != 10_000
        or expected.get("class_weight_formula") != "count_sum/(5*max(class_count,1))"
    ):
        raise PredictionLock2DValidationError(
            "prereg pimsr class counts/formula are not frozen to the 10k train split"
        )
    class_counts = np.asarray(class_counts_raw, dtype=np.float64)
    expected_class_weights = class_counts.sum() / (5.0 * np.maximum(class_counts, 1.0))
    prereg_class_weights = np.asarray(expected.get("class_weights"), dtype=np.float64)
    if prereg_class_weights.shape != (5,) or not np.array_equal(
        prereg_class_weights, expected_class_weights
    ):
        raise PredictionLock2DValidationError(
            "prereg pimsr class weights do not match the frozen class counts"
        )
    class_weights = np.asarray(actual.get("class_weights"), dtype=np.float64)
    if (
        class_weights.shape != (5,)
        or not np.isfinite(class_weights).all()
        or np.any(class_weights <= 0)
        or not np.array_equal(class_weights, expected_class_weights)
    ):
        raise PredictionLock2DValidationError(
            "pimsr resolved class weights differ from the pinned train split"
        )
    _mapping(actual.get("runtime"), "pimsr resolved training runtime")


def _validate_mtdlpy_training(
    expected: Mapping[str, Any], actual: Mapping[str, Any], training_seed: int
) -> None:
    expected_optimizer = _exact_object(
        expected.get("optimizer"),
        frozenset({"betas", "eps", "learning_rate", "name", "weight_decay"}),
        "prereg mtdlpy optimizer",
    )
    if expected.get("recipe_id") != "benchmark_reviewed_v1":
        raise PredictionLock2DValidationError(
            "MTDLPy preregistration must select benchmark_reviewed_v1"
        )
    if expected_optimizer != {
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "learning_rate": 0.0001,
        "name": "Adam",
        "weight_decay": 0.0,
    }:
        raise PredictionLock2DValidationError("prereg MTDLPy optimizer is not frozen")
    resolved = {
        "batch_size": 4,
        "campaign_seeds": list(TRAINING_SEEDS),
        "checkpoint_selection": "lowest validation MSE; strict less-than; first tie",
        "early_stopping": None,
        "epochs": 10,
        "gradient_clip_max_norm": 0.1,
        "loss": "mean_squared_error_mean",
        "normalization": "none",
        "optimizer": {
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "learning_rate": 0.0001,
            "name": "Adam",
            "weight_decay": 0.0,
        },
        "recipe_id": "benchmark_reviewed_v1",
        "schedule_origin": (
            "preregistered benchmark-native reviewed adapter schedule; "
            "not an MTDLPy upstream default"
        ),
        "scheduler": None,
        "seed": training_seed,
    }
    if dict(actual) != resolved:
        raise PredictionLock2DValidationError(
            "MTDLPy resolved training_config differs from the closed reviewed recipe"
        )
    if (
        expected.get("epochs") != 10
        or expected.get("batch_size") != 4
        or expected.get("gradient_clip_max_norm") != 0.1
        or expected.get("scheduler") != "none"
        or expected.get("early_stopping") != "none_run_all_10_epochs"
    ):
        raise PredictionLock2DValidationError("prereg MTDLPy schedule is not frozen")


def _validate_densenet_training(
    expected: Mapping[str, Any], actual: Mapping[str, Any], training_seed: int
) -> None:
    resolved = {
        "batch_size": 100,
        "checkpoint_selection": "lowest validation weighted MSE; strict less-than; first tie",
        "checkpoint_selection_origin": (
            "benchmark-native validation-only adaptation; upstream saves the "
            "last epoch and does not select by validation loss"
        ),
        "early_stopping": None,
        "epochs": 200,
        "equal_compute_claim": False,
        "gradient_clipping": None,
        "loss": {
            "background_log10_resistivity": float(np.log10(300.0)),
            "background_multiplier": 1.0,
            "mask_rule": "target != log10(300)",
            "name": "weighted_mean_squared_error",
            "non_background_multiplier": 10.0,
        },
        "normalization": "per-sample per-component mean-center then max-absolute divide",
        "optimizer": {
            "amsgrad": False,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "learning_rate": 0.0001,
            "name": "Adam",
            "weight_decay": 0.0,
        },
        "schedule_origin": (
            "pinned upstream semantic method-specific schedule adapted to the "
            "common train/validation split"
        ),
        "scheduler": None,
        "seed": training_seed,
        "upstream_cli_unchanged_claim": False,
    }
    if dict(actual) != resolved:
        raise PredictionLock2DValidationError(
            "DenseNet resolved training_config differs from the preregistered recipe"
        )
    expected_resolved = dict(resolved)
    expected_resolved.pop("seed")
    if dict(expected) != expected_resolved:
        raise PredictionLock2DValidationError(
            "DenseNet preregistered training_config is not the exact frozen recipe"
        )


def _validate_training_recipe(
    method_id: str,
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    training_seed: int | None = None,
) -> None:
    if "recipe_id" in expected and actual.get("recipe_id") != expected["recipe_id"]:
        raise PredictionLock2DValidationError(
            f"{method_id} runtime recipe_id differs from preregistration"
        )
    if training_seed is None:
        return
    if method_id == "pimsr":
        _validate_pimsr_training(expected, actual, training_seed)
    elif method_id == "mtdlpy":
        _validate_mtdlpy_training(expected, actual, training_seed)
    elif method_id == "mt2dinv_densenet":
        _validate_densenet_training(expected, actual, training_seed)


def _validate_mtdlpy_dependency_closure_identity(
    value: Any,
) -> Mapping[str, Any]:
    closure = _mapping(value, "MTDLPy dependency_closure")
    _exact_keys(
        closure,
        frozenset(
            {
                "cli_entrypoint_source_included",
                "required_local_python_source_artifacts_recorded",
                "evidence_scope",
                "fixed_imagenet_weights",
                "local_source_artifacts",
                "native_binary_environment_complete",
                "packages",
                "python",
                "schema",
                "schema_version",
            }
        ),
        "MTDLPy dependency_closure",
    )
    if (
        closure.get("schema") != "pimsr-mtdlpy-dependency-closure"
        or closure.get("schema_version") != 3
        or closure.get("evidence_scope")
        != "direct_python_source_artifacts_and_distribution_version_strings"
        or closure.get("cli_entrypoint_source_included") is not True
        or closure.get("required_local_python_source_artifacts_recorded") is not True
        or closure.get("native_binary_environment_complete") is not False
    ):
        raise PredictionLock2DValidationError(
            "MTDLPy dependency closure scope/completeness is wrong"
        )
    return closure


def _runtime_bindings(
    runtime: Mapping[str, Any],
    *,
    method_id: str,
    training_seed: int,
    method: Mapping[str, Any],
    observations: ArtifactSnapshot,
    prediction: ArtifactSnapshot,
    checkpoint: ArtifactSnapshot,
    source: ArtifactSnapshot,
    adapter: ArtifactSnapshot,
    train_sha256: str,
    validation_sha256: str,
) -> None:
    runtime_keys = {
        "pimsr": _PIMSR_RUNTIME_KEYS,
        "mtdlpy": _MTDLPY_RUNTIME_KEYS,
        "mt2dinv_densenet": _DENSENET_RUNTIME_KEYS,
    }[method_id]
    _exact_keys(runtime, runtime_keys, f"{method_id} runtime")
    implementation = _mapping(
        method["implementation"], f"prereg {method_id}.implementation"
    )
    training = _mapping(method["training"], f"prereg {method_id}.training")
    expected_commit = _git_commit(
        implementation.get("repository_commit"), f"prereg {method_id} repository_commit"
    )
    if method_id == "pimsr":
        if (
            runtime.get("schema") != "pimsr-sota-2d-pimsr-runtime"
            or runtime.get("schema_version") != 3
            or runtime.get("method") != "pimsr-2d"
            or runtime.get("operation") != "inference_from_reusable_checkpoint"
            or runtime.get("comparison_status") != "unscored_prediction_artifact"
            or runtime.get("ranking_allowed") is not False
            or runtime.get("training_seed") != training_seed
            or runtime.get("truth_keys_accepted") is not False
            or runtime.get("contains_truth") is not False
            or runtime.get("heldout_truth_available_to_adapter") is not False
        ):
            raise PredictionLock2DValidationError(
                "PIMSR runtime v3 identity/seed is wrong"
            )
        source_record = _mapping(runtime.get("source"), "pimsr runtime.source")
        if (
            source_record.get("repository_checked") is not True
            or source_record.get("dirty_tree") is not False
            or source_record.get("head_commit") != expected_commit
        ):
            raise PredictionLock2DValidationError("PIMSR runtime source binding is wrong")
        inputs = _mapping(runtime.get("inputs"), "pimsr runtime.inputs")
        observation_digest, _ = _artifact_digest(
            inputs.get("observations"), "pimsr observations"
        )
        checkpoint_digest, checkpoint_size = _artifact_digest(
            inputs.get("checkpoint"), "pimsr checkpoint"
        )
        prediction_digest, prediction_size = _artifact_digest(
            runtime.get("output"), "pimsr output"
        )
        contract = _mapping(runtime.get("training_contract"), "pimsr training_contract")
        train_digest, _ = _artifact_digest(
            contract.get("train_dataset"), "pimsr train dataset"
        )
        validation_digest, _ = _artifact_digest(
            contract.get("validation_dataset"), "pimsr validation dataset"
        )
        adapter_digest, _ = _artifact_digest(
            runtime.get("adapter_source"), "pimsr adapter"
        )
        module_files = _mapping(
            source_record.get("module_files"), "pimsr source.module_files"
        )
        source_digest, _ = _artifact_digest(
            module_files.get("network2d.py"), "pimsr network2d"
        )
        actual_training = _mapping(
            contract.get("training_config"), "pimsr training config"
        )
        if actual_training.get("seed") != training_seed:
            raise PredictionLock2DValidationError(
                "PIMSR checkpoint training_config seed differs from the run cell"
            )
        observation_contract = _mapping(
            runtime.get("observation_contract"), "pimsr observation_contract"
        )
        prediction_contract = _mapping(
            runtime.get("prediction_contract"), "pimsr prediction_contract"
        )
        checkpoint_contract = _mapping(
            runtime.get("checkpoint_contract"), "pimsr checkpoint_contract"
        )
        if (
            observation_contract.get("schema") != "pimsr-sota-2d-observations"
            or observation_contract.get("schema_version") != 1
            or observation_contract.get("observations_sha256") != observations.sha256
            or observation_contract.get("truth_keys_accepted") is not False
            or observation_contract.get("contains_truth") is not False
            or observation_contract.get("evaluation_floor_role")
            != "scorer_only_not_model_input"
        ):
            raise PredictionLock2DValidationError(
                "PIMSR observation contract is not exact/truth-free"
            )
        if (
            prediction_contract.get("schema") != "pimsr-sota-2d-predictions"
            or prediction_contract.get("schema_version") != 2
            or prediction_contract.get("observations_sha256") != observations.sha256
            or prediction_contract.get("prediction_sha256") != prediction.sha256
            or prediction_contract.get("truth_keys_accepted") is not False
            or prediction_contract.get("contains_truth") is not False
        ):
            raise PredictionLock2DValidationError(
                "PIMSR prediction contract is not exact/truth-free"
            )
        if (
            checkpoint_contract.get("schema") != "pimsr-train-2d"
            or checkpoint_contract.get("schema_version") != 1
            or checkpoint_contract.get("safe_load") != "torch.load(weights_only=True)"
            or checkpoint_contract.get("seed") != training_seed
            or checkpoint_contract.get("contains_observation_campaign") is not False
            or checkpoint_contract.get("truth_keys_accepted") is not False
            or checkpoint_contract.get("contains_truth") is not False
            or checkpoint_contract.get("checkpoint_sha256") != checkpoint.sha256
        ):
            raise PredictionLock2DValidationError(
                "PIMSR checkpoint contract is not the restricted reusable schema"
            )
        checkpoint_datasets = _mapping(
            checkpoint_contract.get("dataset_identities"),
            "pimsr checkpoint dataset_identities",
        )
        if (
            _artifact_digest(checkpoint_datasets.get("train"), "pimsr checkpoint train")[
                0
            ]
            != train_sha256
            or _artifact_digest(
                checkpoint_datasets.get("validation"),
                "pimsr checkpoint validation",
            )[0]
            != validation_sha256
        ):
            raise PredictionLock2DValidationError(
                "PIMSR checkpoint datasets differ from the frozen common split"
            )
    elif method_id in {"mtdlpy", "mt2dinv_densenet"}:
        expected_schema = {
            "mtdlpy": "pimsr-mtdlpy-common-retrain-runtime",
            "mt2dinv_densenet": "pimsr-mt2dinv-densenet-common-retrain-runtime",
        }[method_id]
        expected_name = {
            "mtdlpy": "MTDLPy/DinkNet50",
            "mt2dinv_densenet": "MT2DInv-DenseNet/iDenseNet",
        }[method_id]
        expected_schema_version = {
            "mtdlpy": 2,
            "mt2dinv_densenet": 2,
        }[method_id]
        if (
            runtime.get("schema") != expected_schema
            or runtime.get("schema_version") != expected_schema_version
            or runtime.get("method") != expected_name
            or runtime.get("track") != "common-retrain"
            or runtime.get("operation") != "inference_from_reusable_checkpoint"
            or runtime.get("comparison_status") != "unscored_prediction_artifact"
            or runtime.get("ranking_allowed") is not False
            or runtime.get("seed") != training_seed
            or runtime.get("truth_keys_accepted") is not False
            or runtime.get("contains_truth") is not False
        ):
            raise PredictionLock2DValidationError(
                f"{method_id} split inference runtime identity/seed is wrong"
            )
        repository = _mapping(
            runtime.get("repository"), f"{method_id} runtime.repository"
        )
        if (
            repository.get("clean_worktree") is not True
            or repository.get("commit") != expected_commit
        ):
            raise PredictionLock2DValidationError(f"{method_id} source binding is wrong")
        artifacts = _mapping(
            runtime.get("source_artifacts"), f"{method_id} source_artifacts"
        )
        if method_id == "mtdlpy":
            _exact_keys(
                artifacts,
                frozenset(
                    {
                        "adapter_source",
                        "artifact_guard_source",
                        "dataset_contract_loader_source",
                        "dinknet_source",
                        "heldout_observations",
                        "imagenet_weights",
                        "materializer_contract_source",
                        "runner_source",
                        "train_dataset",
                        "validation_dataset",
                    }
                ),
                "MTDLPy source_artifacts",
            )
        else:
            _exact_keys(
                artifacts,
                frozenset(
                    {
                        "adapter_source",
                        "architecture_source",
                        "artifact_guard_source",
                        "heldout_observations",
                        "inversion_dataset_contract_source",
                        "materializer_contract_source",
                        "runner_source",
                        "shared_contract_loader_source",
                        "train_dataset",
                        "validation_dataset",
                    }
                ),
                "DenseNet source_artifacts",
            )
        train_digest, _ = _artifact_digest(
            artifacts.get("train_dataset"), f"{method_id} train"
        )
        validation_digest, _ = _artifact_digest(
            artifacts.get("validation_dataset"), f"{method_id} validation"
        )
        adapter_digest, _ = _artifact_digest(
            artifacts.get("adapter_source"), f"{method_id} adapter"
        )
        source_key = "dinknet_source" if method_id == "mtdlpy" else "architecture_source"
        source_digest, _ = _artifact_digest(
            artifacts.get(source_key), f"{method_id} source"
        )
        heldout_digest, _ = _artifact_digest(
            artifacts.get("heldout_observations"), f"{method_id} heldout"
        )
        observation_contract = _mapping(
            runtime.get("observation_contract"), f"{method_id} observation_contract"
        )
        if observation_contract.get("truth_keys_accepted") is not False:
            raise PredictionLock2DValidationError(
                f"{method_id} runtime does not prove truth-free observations"
            )
        contract_observation = observation_contract.get(
            "observations_sha256", heldout_digest
        )
        prediction_contract = _mapping(
            runtime.get("prediction_contract"), f"{method_id} prediction_contract"
        )
        if prediction_contract.get("contains_truth") is not False or (
            method_id == "mtdlpy"
            and prediction_contract.get("truth_keys_accepted") is not False
        ):
            raise PredictionLock2DValidationError(
                f"{method_id} runtime does not prove truth-free predictions"
            )
        observation_digest = prediction_contract.get("observations_sha256")
        outputs = _mapping(runtime.get("outputs"), f"{method_id} outputs")
        checkpoint_digest, checkpoint_size = _artifact_digest(
            outputs.get("checkpoint"), f"{method_id} checkpoint"
        )
        prediction_digest, prediction_size = _artifact_digest(
            outputs.get("predictions"), f"{method_id} predictions"
        )
        actual_training = _mapping(
            runtime.get("training_config"), f"{method_id} training"
        )
        if contract_observation != heldout_digest:
            raise PredictionLock2DValidationError(
                f"{method_id} observation contract differs from heldout artifact"
            )
        checkpoint_contract = _mapping(
            runtime.get("checkpoint_contract"), f"{method_id} checkpoint_contract"
        )
        if (
            checkpoint_contract.get("contains_truth") is not False
            or checkpoint_contract.get("contains_observation_campaign") is not False
            or checkpoint_contract.get("seed") != training_seed
        ):
            raise PredictionLock2DValidationError(
                f"{method_id} checkpoint is not proven campaign-independent"
            )
        if method_id == "mtdlpy":
            if (
                checkpoint_contract.get("schema")
                != "pimsr-mtdlpy-common-retrain-checkpoint"
                or checkpoint_contract.get("schema_version") != 1
                or checkpoint_contract.get("safe_load") != "torch.load(weights_only=True)"
            ):
                raise PredictionLock2DValidationError(
                    "MTDLPy checkpoint contract is not the restricted reusable schema"
                )
            checkpoint_datasets = _mapping(
                checkpoint_contract.get("dataset_identities"),
                "MTDLPy checkpoint dataset_identities",
            )
            if (
                _artifact_digest(
                    checkpoint_datasets.get("train"), "MTDLPy checkpoint train"
                )[0]
                != train_sha256
                or _artifact_digest(
                    checkpoint_datasets.get("validation"),
                    "MTDLPy checkpoint validation",
                )[0]
                != validation_sha256
            ):
                raise PredictionLock2DValidationError(
                    "MTDLPy checkpoint datasets differ from the frozen common split"
                )
            closure = _validate_mtdlpy_dependency_closure_identity(
                runtime.get("dependency_closure")
            )
            local_sources = _mapping(
                closure.get("local_source_artifacts"),
                "MTDLPy dependency_closure.local_source_artifacts",
            )
            _exact_keys(
                local_sources,
                frozenset(
                    {
                        "adapter",
                        "cli_runner",
                        "dataset2d_materialization",
                        "pimsr_inversion_contracts2d",
                        "runner2d",
                        "upstream_dinknet",
                    }
                ),
                "MTDLPy dependency_closure.local_source_artifacts",
            )
            _exact_keys(
                _mapping(closure.get("packages"), "MTDLPy dependency packages"),
                frozenset({"h5py", "numpy", "pimsr-inversion", "torch", "torchvision"}),
                "MTDLPy dependency packages",
            )
            runner_digest, _ = _artifact_digest(
                artifacts.get("runner_source"), "MTDLPy runner source"
            )
            weights_digest, _ = _artifact_digest(
                artifacts.get("imagenet_weights"), "MTDLPy ImageNet weights"
            )
            initialization = _mapping(
                method.get("initialization"), "prereg MTDLPy initialization"
            )
            expected_weights_digest = _sha256(
                initialization.get("sha256"),
                "prereg MTDLPy initialization.sha256",
            )
            expected_runner_digest = _sha256(
                implementation.get("runner_source_sha256"),
                "prereg MTDLPy runner_source_sha256",
            )
            closure_digest = _canonical_object_sha256(closure)
            expected_closure_digest = _sha256(
                implementation.get("dependency_closure_sha256"),
                "prereg MTDLPy dependency_closure_sha256",
            )
            if (
                dict(_mapping(artifacts.get("adapter_source"), "MTDLPy adapter artifact"))
                != dict(_mapping(local_sources.get("adapter"), "MTDLPy closure adapter"))
                or dict(
                    _mapping(artifacts.get("dinknet_source"), "MTDLPy source artifact")
                )
                != dict(
                    _mapping(
                        local_sources.get("upstream_dinknet"), "MTDLPy closure source"
                    )
                )
                or dict(
                    _mapping(artifacts.get("runner_source"), "MTDLPy runner artifact")
                )
                != dict(
                    _mapping(local_sources.get("cli_runner"), "MTDLPy closure runner")
                )
                or dict(
                    _mapping(artifacts.get("imagenet_weights"), "MTDLPy weights artifact")
                )
                != dict(
                    _mapping(
                        closure.get("fixed_imagenet_weights"), "MTDLPy closure weights"
                    )
                )
                or weights_digest != expected_weights_digest
                or runner_digest != expected_runner_digest
                or closure_digest != expected_closure_digest
                or _mapping(
                    closure.get("fixed_imagenet_weights"),
                    "MTDLPy fixed_imagenet_weights",
                ).get("sha256")
                != expected_weights_digest
                or _mapping(
                    local_sources.get("cli_runner"), "MTDLPy closure cli_runner"
                ).get("sha256")
                != expected_runner_digest
            ):
                raise PredictionLock2DValidationError(
                    "MTDLPy initialization/runner closure differs from preregistration"
                )
            bindings = _mapping(runtime.get("bindings"), "MTDLPy runtime.bindings")
            expected_bindings = {
                "adapter_source_sha256": adapter.sha256,
                "checkpoint_sha256": checkpoint.sha256,
                "dependency_closure_sha256": closure_digest,
                "imagenet_weights_sha256": expected_weights_digest,
                "observations_sha256": observations.sha256,
                "prediction_sha256": prediction.sha256,
                "runner_source_sha256": expected_runner_digest,
                "source_clean_worktree": True,
                "source_commit": expected_commit,
                "train_sha256": train_sha256,
                "training_seed": training_seed,
                "upstream_source_sha256": source.sha256,
                "validation_sha256": validation_sha256,
            }
            if dict(bindings) != expected_bindings:
                raise PredictionLock2DValidationError(
                    "MTDLPy runtime bindings are not the exact frozen evidence set"
                )
        if method_id == "mt2dinv_densenet":
            if (
                runtime.get("method_id") != "mt2dinv_densenet"
                or runtime.get("training_seed") != training_seed
                or runtime.get("heldout_truth_available_to_adapter") is not False
            ):
                raise PredictionLock2DValidationError(
                    "DenseNet split inference identity/truth declaration is wrong"
                )
            if (
                checkpoint_contract.get("schema")
                != "pimsr-mt2dinv-densenet-common-retrain-checkpoint"
                or checkpoint_contract.get("schema_version") != 2
                or checkpoint_contract.get("safe_load") != "torch.load(weights_only=True)"
                or checkpoint_contract.get("campaign_observations_accepted_for_training")
                is not False
                or checkpoint_contract.get("truth_keys_accepted") is not False
                or checkpoint_contract.get("checkpoint_sha256") != checkpoint.sha256
            ):
                raise PredictionLock2DValidationError(
                    "DenseNet checkpoint contract is not the restricted reusable schema"
                )
            checkpoint_datasets = _mapping(
                checkpoint_contract.get("dataset_identities"),
                "DenseNet checkpoint dataset_identities",
            )
            checkpoint_train, _ = _artifact_digest(
                checkpoint_datasets.get("train"), "DenseNet checkpoint train"
            )
            checkpoint_validation, _ = _artifact_digest(
                checkpoint_datasets.get("validation"),
                "DenseNet checkpoint validation",
            )
            if (
                checkpoint_train != train_sha256
                or checkpoint_validation != validation_sha256
            ):
                raise PredictionLock2DValidationError(
                    "DenseNet checkpoint datasets differ from the frozen common split"
                )
            closure = _mapping(
                runtime.get("dependency_closure"), "DenseNet dependency_closure"
            )
            _exact_keys(
                closure,
                frozenset(
                    {
                        "cli_entrypoint_source_included",
                        "closure_sha256",
                        "required_local_python_source_artifacts_recorded",
                        "evidence_scope",
                        "local_source_artifacts",
                        "native_binary_environment_complete",
                        "packages",
                        "python",
                        "schema",
                        "schema_version",
                    }
                ),
                "DenseNet dependency_closure",
            )
            closure_body = dict(closure)
            closure_digest = _sha256(
                closure_body.pop("closure_sha256", None),
                "DenseNet dependency_closure.closure_sha256",
            )
            if (
                closure.get("schema")
                != "pimsr-mt2dinv-densenet-source-dependency-closure"
                or closure.get("schema_version") != 2
                or closure.get("evidence_scope")
                != "direct_python_source_artifacts_and_distribution_version_strings"
                or closure.get("cli_entrypoint_source_included") is not True
                or closure.get("required_local_python_source_artifacts_recorded")
                is not True
                or closure.get("native_binary_environment_complete") is not False
                or closure_digest != _canonical_object_sha256(closure_body)
            ):
                raise PredictionLock2DValidationError(
                    "DenseNet dependency closure identity/completeness is wrong"
                )
            local_sources = _mapping(
                closure.get("local_source_artifacts"),
                "DenseNet dependency_closure.local_source_artifacts",
            )
            _exact_keys(
                local_sources,
                frozenset(
                    {
                        "adapter_source",
                        "architecture_source",
                        "artifact_guard_source",
                        "inversion_dataset_contract_source",
                        "materializer_contract_source",
                        "runner_source",
                        "shared_contract_loader_source",
                    }
                ),
                "DenseNet dependency_closure.local_source_artifacts",
            )
            _exact_keys(
                _mapping(closure.get("packages"), "DenseNet dependency packages"),
                frozenset({"h5py", "numpy", "pimsr-inversion", "torch"}),
                "DenseNet dependency packages",
            )
            for key in local_sources:
                if dict(_mapping(local_sources[key], f"DenseNet closure {key}")) != dict(
                    _mapping(artifacts[key], f"DenseNet source artifact {key}")
                ):
                    raise PredictionLock2DValidationError(
                        f"DenseNet closure/source artifact {key!r} differs"
                    )
            expected_runner_digest = _sha256(
                implementation.get("runner_source_sha256"),
                "prereg DenseNet runner_source_sha256",
            )
            expected_shared_digest = _sha256(
                implementation.get("shared_contract_loader_source_sha256"),
                "prereg DenseNet shared_contract_loader_source_sha256",
            )
            expected_closure_digest = _sha256(
                implementation.get("dependency_closure_sha256"),
                "prereg DenseNet dependency_closure_sha256",
            )
            if (
                _artifact_digest(artifacts["runner_source"], "DenseNet runner")[0]
                != expected_runner_digest
                or _artifact_digest(
                    artifacts["shared_contract_loader_source"],
                    "DenseNet shared loader",
                )[0]
                != expected_shared_digest
                or closure_digest != expected_closure_digest
            ):
                raise PredictionLock2DValidationError(
                    "DenseNet dependency sources differ from preregistration"
                )
            bindings = _mapping(runtime.get("bindings"), "DenseNet runtime.bindings")
            expected_bindings = {
                "adapter_source_sha256": adapter.sha256,
                "checkpoint_sha256": checkpoint.sha256,
                "dependency_closure_sha256": closure_digest,
                "observations_sha256": observations.sha256,
                "prediction_sha256": prediction.sha256,
                "runner_source_sha256": expected_runner_digest,
                "shared_contract_loader_source_sha256": expected_shared_digest,
                "source_clean_worktree": True,
                "source_commit": expected_commit,
                "train_sha256": train_sha256,
                "training_seed": training_seed,
                "upstream_source_sha256": source.sha256,
                "validation_sha256": validation_sha256,
            }
            if dict(bindings) != expected_bindings:
                raise PredictionLock2DValidationError(
                    "DenseNet runtime bindings are not the exact frozen evidence set"
                )
    else:  # pragma: no cover - callers are closed by prereg parsing
        raise PredictionLock2DValidationError(f"unsupported method {method_id!r}")
    expected_values = (
        (observation_digest, observations.sha256, "observations"),
        (prediction_digest, prediction.sha256, "prediction"),
        (checkpoint_digest, checkpoint.sha256, "checkpoint"),
        (source_digest, source.sha256, "source"),
        (adapter_digest, adapter.sha256, "adapter"),
        (train_digest, train_sha256, "train dataset"),
        (validation_digest, validation_sha256, "validation dataset"),
    )
    for actual, expected, role in expected_values:
        if actual != expected:
            raise PredictionLock2DValidationError(
                f"{method_id} runtime {role} binding differs from locked evidence"
            )
    if checkpoint_size is not None and checkpoint_size != checkpoint.size_bytes:
        raise PredictionLock2DValidationError(
            f"{method_id} checkpoint size binding is wrong"
        )
    if prediction_size is not None and prediction_size != prediction.size_bytes:
        raise PredictionLock2DValidationError(
            f"{method_id} prediction size binding is wrong"
        )
    _validate_training_recipe(
        method_id, training, actual_training, training_seed=training_seed
    )
    _reject_prescore_secrets(runtime, f"{method_id} runtime")


def _preregistered_source_sha256(
    method_id: str, implementation: Mapping[str, Any]
) -> str:
    key = {
        "pimsr": "network_source_sha256",
        "mtdlpy": "dinknet_source_sha256",
        "mt2dinv_densenet": "architecture_source_sha256",
    }[method_id]
    return _sha256(implementation.get(key), f"prereg {method_id} implementation.{key}")


def _adapter_snapshot(
    preregistration_path: Path, method_id: str, method: Mapping[str, Any]
) -> ArtifactSnapshot:
    implementation = _mapping(
        method["implementation"], f"prereg {method_id}.implementation"
    )
    relative_text = implementation.get("adapter_source_path")
    if not isinstance(relative_text, str) or not relative_text:
        raise PredictionLock2DValidationError(
            f"{method_id} adapter_source_path is missing"
        )
    repository = preregistration_path.resolve(strict=True).parent.parent
    source = (repository / relative_text).resolve(strict=True)
    try:
        source.relative_to(repository)
    except ValueError as exc:
        raise PredictionLock2DValidationError(
            f"{method_id} adapter source escapes the preregistration repository"
        ) from exc
    snapshot = snapshot_regular_file(
        source,
        expected_sha256=_sha256(
            implementation.get("adapter_source_sha256"),
            f"prereg {method_id} adapter_source_sha256",
        ),
        role=f"{method_id} adapter source",
    )
    _validate_source_repository(
        snapshot,
        repository,
        expected_commit=_git_commit(
            implementation.get("adapter_repository_commit"),
            f"prereg {method_id} adapter_repository_commit",
        ),
        method_id=f"{method_id} adapter",
        allow_descendant_head=True,
        protected_paths=("src", "scripts", "pyproject.toml"),
    )
    return snapshot


def _parse_lock_input(
    prereg: _Preregistration,
    input_snapshot: ArtifactSnapshot,
    input_value: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[LockedRun2D]]:
    _exact_keys(input_value, _LOCK_INPUT_KEYS, "prediction lock input")
    if (
        input_value["audience"] != LOCK_AUDIENCE
        or input_value["schema"] != LOCK_INPUT_SCHEMA
        or input_value["schema_version"] != LOCK_INPUT_SCHEMA_VERSION
        or input_value["preregistration_sha256"] != prereg.snapshot.sha256
    ):
        raise PredictionLock2DValidationError("prediction lock input identity is wrong")
    _reject_prescore_secrets(input_value, "prediction lock input")
    base = input_snapshot.path.parent
    campaigns_raw = _sequence(input_value["campaigns"], "prediction lock input campaigns")
    if len(campaigns_raw) != CAMPAIGN_COUNT:
        raise PredictionLock2DValidationError(
            "prediction lock input requires five campaigns"
        )
    campaigns_output: list[dict[str, Any]] = []
    locked_runs: list[LockedRun2D] = []
    seen_unique_identities: set[tuple[int, int]] = {
        input_snapshot.identity,
        prereg.snapshot.identity,
    }
    seen_observation_hashes: set[str] = set()
    adapter_by_method = {
        method_id: _adapter_snapshot(prereg.snapshot.path, method_id, method)
        for method_id, method in prereg.method_by_id.items()
    }
    if len({snapshot.identity for snapshot in adapter_by_method.values()}) != len(
        METHOD_IDS
    ):
        raise PredictionLock2DValidationError(
            "method adapter files must have distinct identities"
        )
    if any(
        snapshot.identity in {input_snapshot.identity, prereg.snapshot.identity}
        for snapshot in adapter_by_method.values()
    ):
        raise PredictionLock2DValidationError(
            "adapter files must not alias preregistration or lock input"
        )
    seen_unique_identities.update(
        snapshot.identity for snapshot in adapter_by_method.values()
    )
    checkpoint_by_cell: dict[tuple[str, int], ArtifactSnapshot] = {}
    source_by_method: dict[str, ArtifactSnapshot] = {}
    run_unique_identities: set[tuple[int, int]] = set()
    seen_cells: set[tuple[str, str, int]] = set()
    for campaign_index, raw_campaign in enumerate(campaigns_raw):
        campaign_path = f"campaigns[{campaign_index}]"
        campaign = _mapping(raw_campaign, campaign_path)
        _exact_keys(campaign, _CAMPAIGN_INPUT_KEYS, campaign_path)
        campaign_id = _identifier(campaign["campaign_id"], f"{campaign_path}.campaign_id")
        if campaign_id != prereg.campaign_ids[campaign_index]:
            raise PredictionLock2DValidationError(
                "lock input campaign order/ids differ from preregistration"
            )
        observation_snapshot, _ = _artifact_reference(
            campaign["observations"], base=base, role=f"{campaign_path}.observations"
        )
        manifest_snapshot, _ = _artifact_reference(
            campaign["observation_manifest"],
            base=base,
            role=f"{campaign_path}.observation_manifest",
        )
        for snapshot in (observation_snapshot, manifest_snapshot):
            if (
                snapshot.identity in seen_unique_identities
                or snapshot.identity in run_unique_identities
            ):
                raise PredictionLock2DValidationError(
                    "campaign observation payload/manifest identities must be distinct"
                )
            seen_unique_identities.add(snapshot.identity)
        if observation_snapshot.sha256 in seen_observation_hashes:
            raise PredictionLock2DValidationError(
                "five campaigns require distinct observation payload hashes"
            )
        seen_observation_hashes.add(observation_snapshot.sha256)
        manifest = _strict_json(
            manifest_snapshot, f"{campaign_path}.observation_manifest"
        )
        sample_count, sample_ids, x_cell_centers_m, depth_cell_centers_m = (
            _observation_identity(
                observation_snapshot,
                manifest,
                campaign_id,
                prereg.family_partition,
            )
        )
        runs_raw = _sequence(campaign["runs"], f"{campaign_path}.runs")
        expected_per_campaign = len(METHOD_IDS) * len(TRAINING_SEEDS)
        if len(runs_raw) != expected_per_campaign:
            raise PredictionLock2DValidationError(
                f"each campaign requires exactly {expected_per_campaign} run cells"
            )
        for run_index, raw_run in enumerate(runs_raw):
            run_path = f"{campaign_path}.runs[{run_index}]"
            run = _mapping(raw_run, run_path)
            _exact_keys(run, _RUN_INPUT_KEYS, run_path)
            method_id = _identifier(run["method_id"], f"{run_path}.method_id")
            training_seed = _integer(run["training_seed"], f"{run_path}.training_seed")
            cell = (campaign_id, method_id, training_seed)
            if cell in seen_cells:
                raise PredictionLock2DValidationError(
                    "prediction lock input has duplicate cells"
                )
            seen_cells.add(cell)
            if (
                method_id not in prereg.method_by_id
                or training_seed not in TRAINING_SEEDS
            ):
                raise PredictionLock2DValidationError(
                    f"{run_path} is outside prereg design"
                )
            prediction, _ = _artifact_reference(
                run["prediction"], base=base, role=f"{run_path}.prediction"
            )
            runtime_snapshot, _ = _artifact_reference(
                run["runtime"], base=base, role=f"{run_path}.runtime"
            )
            checkpoint, _ = _artifact_reference(
                run["checkpoint"], base=base, role=f"{run_path}.checkpoint"
            )
            source, source_repository = _artifact_reference(
                run["source"], base=base, role=f"{run_path}.source", source=True
            )
            assert source_repository is not None
            for unique in (prediction, runtime_snapshot):
                if (
                    unique.identity in seen_unique_identities
                    or unique.identity in run_unique_identities
                ):
                    raise PredictionLock2DValidationError(
                        "prediction/runtime paths and hardlink identities must be unique"
                    )
                run_unique_identities.add(unique.identity)
            checkpoint_key = (method_id, training_seed)
            prior_checkpoint = checkpoint_by_cell.setdefault(checkpoint_key, checkpoint)
            if (
                prior_checkpoint.identity != checkpoint.identity
                or prior_checkpoint.sha256 != checkpoint.sha256
            ):
                raise PredictionLock2DValidationError(
                    "one method/seed must reuse exactly one checkpoint across campaigns"
                )
            prior_source = source_by_method.setdefault(method_id, source)
            if (
                prior_source.identity != source.identity
                or prior_source.sha256 != source.sha256
            ):
                raise PredictionLock2DValidationError(
                    "one method must reuse exactly one upstream source artifact"
                )
            method = prereg.method_by_id[method_id]
            implementation = _mapping(
                method["implementation"], f"prereg {method_id}.implementation"
            )
            source_commit = _git_commit(
                implementation.get("repository_commit"),
                f"prereg {method_id}.repository_commit",
            )
            if source.sha256 != _preregistered_source_sha256(method_id, implementation):
                raise PredictionLock2DValidationError(
                    f"{method_id} upstream source hash differs from preregistration"
                )
            _validate_source_repository(
                source,
                source_repository,
                expected_commit=source_commit,
                method_id=method_id,
            )
            _prediction_identity(
                prediction,
                observations_sha256=observation_snapshot.sha256,
                sample_ids=sample_ids,
                x_cell_centers_m=x_cell_centers_m,
                depth_cell_centers_m=depth_cell_centers_m,
            )
            runtime = _strict_json(runtime_snapshot, f"{run_path}.runtime")
            adapter = adapter_by_method[method_id]
            _runtime_bindings(
                runtime,
                method_id=method_id,
                training_seed=training_seed,
                method=method,
                observations=observation_snapshot,
                prediction=prediction,
                checkpoint=checkpoint,
                source=source,
                adapter=adapter,
                train_sha256=prereg.train_sha256,
                validation_sha256=prereg.validation_sha256,
            )
            locked_runs.append(
                LockedRun2D(
                    campaign_id=campaign_id,
                    method_id=method_id,
                    training_seed=training_seed,
                    observations_sha256=observation_snapshot.sha256,
                    observation_manifest_sha256=manifest_snapshot.sha256,
                    prediction_sha256=prediction.sha256,
                    prediction_size_bytes=prediction.size_bytes,
                    runtime_sha256=runtime_snapshot.sha256,
                    runtime_size_bytes=runtime_snapshot.size_bytes,
                    checkpoint_sha256=checkpoint.sha256,
                    checkpoint_size_bytes=checkpoint.size_bytes,
                    source_commit=source_commit,
                    source_sha256=source.sha256,
                    adapter_source_sha256=adapter.sha256,
                )
            )
        campaigns_output.append(
            {
                "campaign_id": campaign_id,
                "observation_manifest_sha256": manifest_snapshot.sha256,
                "observations_sha256": observation_snapshot.sha256,
                "observations_size_bytes": observation_snapshot.size_bytes,
                "sample_count": sample_count,
            }
        )
    expected_cells = {
        (campaign_id, method_id, seed)
        for campaign_id in prereg.campaign_ids
        for method_id in METHOD_IDS
        for seed in TRAINING_SEEDS
    }
    if seen_cells != expected_cells or len(locked_runs) != RUN_COUNT:
        raise PredictionLock2DValidationError(
            "prediction lock input matrix is incomplete"
        )
    checkpoints = list(checkpoint_by_cell.values())
    if len(checkpoints) != len(METHOD_IDS) * len(TRAINING_SEEDS):
        raise PredictionLock2DValidationError(
            "exactly fifteen checkpoint identities are required"
        )
    if len({item.identity for item in checkpoints}) != len(checkpoints):
        raise PredictionLock2DValidationError(
            "checkpoint path/hardlink identities must differ across method/seed cells"
        )
    if len({item.sha256 for item in checkpoints}) != len(checkpoints):
        raise PredictionLock2DValidationError(
            "checkpoint digests must differ across method/seed cells"
        )
    if len({item.identity for item in source_by_method.values()}) != len(METHOD_IDS):
        raise PredictionLock2DValidationError(
            "upstream source identities must differ by method"
        )
    checkpoint_identities = {item.identity for item in checkpoints}
    source_identities = {item.identity for item in source_by_method.values()}
    if (
        checkpoint_identities & run_unique_identities
        or source_identities & run_unique_identities
        or checkpoint_identities & source_identities
        or checkpoint_identities & seen_unique_identities
        or source_identities & seen_unique_identities
    ):
        raise PredictionLock2DValidationError(
            "checkpoint/source identities must not alias run, adapter, manifest, or input files"
        )
    locked_runs.sort(
        key=lambda item: (item.campaign_id, item.method_id, item.training_seed)
    )
    return campaigns_output, locked_runs


def create_prediction_lock_2d(
    preregistration_path: str | Path,
    input_manifest_path: str | Path,
    *,
    expected_preregistration_sha256: str,
    expected_input_manifest_sha256: str,
) -> dict[str, Any]:
    """Validate all 75 unscored run cells and return a path-free lock document."""
    prereg = _load_preregistration(preregistration_path, expected_preregistration_sha256)
    input_snapshot = snapshot_regular_file(
        input_manifest_path,
        expected_sha256=_sha256(
            expected_input_manifest_sha256, "expected prediction lock input SHA-256"
        ),
        role="prediction lock input manifest",
    )
    if input_snapshot.identity == prereg.snapshot.identity:
        raise PredictionLock2DValidationError(
            "preregistration and lock input must have distinct file identities"
        )
    input_value = _strict_json(input_snapshot, "prediction lock input manifest")
    campaigns, runs = _parse_lock_input(prereg, input_snapshot, input_value)
    return {
        "audience": LOCK_AUDIENCE,
        "campaigns": campaigns,
        "design": {
            "campaign_count": CAMPAIGN_COUNT,
            "campaign_ids": list(prereg.campaign_ids),
            "checkpoint_count": len(METHOD_IDS) * len(TRAINING_SEEDS),
            "method_count": len(METHOD_IDS),
            "method_ids": list(METHOD_IDS),
            "run_count": RUN_COUNT,
            "samples_per_campaign": SAMPLES_PER_CAMPAIGN,
            "training_seeds": list(TRAINING_SEEDS),
        },
        "input_manifest": {
            "schema": LOCK_INPUT_SCHEMA,
            "schema_version": LOCK_INPUT_SCHEMA_VERSION,
            "sha256": input_snapshot.sha256,
            "size_bytes": input_snapshot.size_bytes,
        },
        "locked": True,
        "preregistration": {
            "id": prereg.value.get("preregistration_id"),
            "schema": PREREGISTRATION_SCHEMA,
            "schema_version": PREREGISTRATION_SCHEMA_VERSION,
            "sha256": prereg.snapshot.sha256,
            "size_bytes": prereg.snapshot.size_bytes,
        },
        "runs": [run.as_dict() for run in runs],
        "schema": LOCK_SCHEMA,
        "schema_version": LOCK_SCHEMA_VERSION,
        "statistical_options": dict(prereg.statistical_options),
    }


def _locked_run(value: Any, path: str) -> LockedRun2D:
    record = _mapping(value, path)
    _exact_keys(record, _LOCK_RUN_KEYS, path)
    return LockedRun2D(
        campaign_id=_identifier(record["campaign_id"], f"{path}.campaign_id"),
        method_id=_identifier(record["method_id"], f"{path}.method_id"),
        training_seed=_integer(record["training_seed"], f"{path}.training_seed"),
        observations_sha256=_sha256(
            record["observations_sha256"], f"{path}.observations_sha256"
        ),
        observation_manifest_sha256=_sha256(
            record["observation_manifest_sha256"], f"{path}.observation_manifest_sha256"
        ),
        prediction_sha256=_sha256(
            record["prediction_sha256"], f"{path}.prediction_sha256"
        ),
        prediction_size_bytes=_integer(
            record["prediction_size_bytes"], f"{path}.prediction_size_bytes", minimum=1
        ),
        runtime_sha256=_sha256(record["runtime_sha256"], f"{path}.runtime_sha256"),
        runtime_size_bytes=_integer(
            record["runtime_size_bytes"], f"{path}.runtime_size_bytes", minimum=1
        ),
        checkpoint_sha256=_sha256(
            record["checkpoint_sha256"], f"{path}.checkpoint_sha256"
        ),
        checkpoint_size_bytes=_integer(
            record["checkpoint_size_bytes"], f"{path}.checkpoint_size_bytes", minimum=1
        ),
        source_commit=_git_commit(record["source_commit"], f"{path}.source_commit"),
        source_sha256=_sha256(record["source_sha256"], f"{path}.source_sha256"),
        adapter_source_sha256=_sha256(
            record["adapter_source_sha256"], f"{path}.adapter_source_sha256"
        ),
    )


def validate_prediction_lock_2d(
    preregistration_path: str | Path,
    lock_path: str | Path,
    *,
    expected_preregistration_sha256: str,
    expected_lock_sha256: str,
) -> ValidatedPredictionLock2D:
    """Validate a committed path-free lock without touching any truth artifact."""
    prereg = _load_preregistration(preregistration_path, expected_preregistration_sha256)
    snapshot = snapshot_regular_file(
        lock_path,
        expected_sha256=_sha256(expected_lock_sha256, "expected prediction lock SHA-256"),
        role="prediction lock",
    )
    if snapshot.identity == prereg.snapshot.identity:
        raise PredictionLock2DValidationError(
            "preregistration and prediction lock must have distinct file identities"
        )
    value = _strict_json(snapshot, "prediction lock")
    _exact_keys(value, _LOCK_KEYS, "prediction lock")
    _reject_prescore_secrets(value, "prediction lock")
    if (
        value["audience"] != LOCK_AUDIENCE
        or value["schema"] != LOCK_SCHEMA
        or value["schema_version"] != LOCK_SCHEMA_VERSION
        or value["locked"] is not True
    ):
        raise PredictionLock2DValidationError("prediction lock identity is wrong")
    prereg_record = _mapping(value["preregistration"], "prediction lock preregistration")
    _exact_keys(
        prereg_record,
        frozenset({"id", "schema", "schema_version", "sha256", "size_bytes"}),
        "prediction lock preregistration",
    )
    if (
        prereg_record.get("schema") != PREREGISTRATION_SCHEMA
        or prereg_record.get("schema_version") != PREREGISTRATION_SCHEMA_VERSION
        or prereg_record.get("sha256") != prereg.snapshot.sha256
        or prereg_record.get("size_bytes") != prereg.snapshot.size_bytes
        or prereg_record.get("id") != prereg.value.get("preregistration_id")
    ):
        raise PredictionLock2DValidationError(
            "prediction lock preregistration binding is wrong"
        )
    input_record = _mapping(value["input_manifest"], "prediction lock input_manifest")
    _exact_keys(
        input_record,
        frozenset({"schema", "schema_version", "sha256", "size_bytes"}),
        "prediction lock input_manifest",
    )
    input_sha = _sha256(
        input_record.get("sha256"), "prediction lock input manifest sha256"
    )
    if (
        input_record.get("schema") != LOCK_INPUT_SCHEMA
        or input_record.get("schema_version") != LOCK_INPUT_SCHEMA_VERSION
        or type(input_record.get("size_bytes")) is not int
        or input_record["size_bytes"] < 1
    ):
        raise PredictionLock2DValidationError("prediction lock input binding is wrong")
    design = _mapping(value["design"], "prediction lock design")
    expected_design = {
        "campaign_count": CAMPAIGN_COUNT,
        "campaign_ids": list(prereg.campaign_ids),
        "checkpoint_count": len(METHOD_IDS) * len(TRAINING_SEEDS),
        "method_count": len(METHOD_IDS),
        "method_ids": list(METHOD_IDS),
        "run_count": RUN_COUNT,
        "samples_per_campaign": SAMPLES_PER_CAMPAIGN,
        "training_seeds": list(TRAINING_SEEDS),
    }
    if design != expected_design:
        raise PredictionLock2DValidationError(
            "prediction lock design differs from preregistration"
        )
    statistical_options = _mapping(value["statistical_options"], "statistical_options")
    _exact_keys(statistical_options, _STATISTICAL_OPTION_KEYS, "statistical_options")
    if statistical_options != prereg.statistical_options:
        raise PredictionLock2DValidationError(
            "prediction lock statistical options differ from prereg"
        )
    campaigns = _sequence(value["campaigns"], "prediction lock campaigns")
    if len(campaigns) != CAMPAIGN_COUNT:
        raise PredictionLock2DValidationError("prediction lock campaign count is wrong")
    campaign_ids: list[str] = []
    observation_by_campaign: dict[str, tuple[str, str]] = {}
    for index, raw in enumerate(campaigns):
        campaign = _mapping(raw, f"prediction lock campaigns[{index}]")
        expected_keys = frozenset(
            {
                "campaign_id",
                "observation_manifest_sha256",
                "observations_sha256",
                "observations_size_bytes",
                "sample_count",
            }
        )
        _exact_keys(campaign, expected_keys, f"prediction lock campaigns[{index}]")
        campaign_id = _identifier(
            campaign["campaign_id"], f"campaigns[{index}].campaign_id"
        )
        if (
            campaign_id != prereg.campaign_ids[index]
            or campaign["sample_count"] != SAMPLES_PER_CAMPAIGN
        ):
            raise PredictionLock2DValidationError(
                "prediction lock campaign identity is wrong"
            )
        observation_by_campaign[campaign_id] = (
            _sha256(campaign["observations_sha256"], "campaign observations sha256"),
            _sha256(campaign["observation_manifest_sha256"], "campaign manifest sha256"),
        )
        _integer(
            campaign["observations_size_bytes"],
            "campaign observations_size_bytes",
            minimum=1,
        )
        campaign_ids.append(campaign_id)
    if len({value[0] for value in observation_by_campaign.values()}) != CAMPAIGN_COUNT:
        raise PredictionLock2DValidationError(
            "prediction lock campaigns must bind distinct observation payloads"
        )
    if len({value[1] for value in observation_by_campaign.values()}) != CAMPAIGN_COUNT:
        raise PredictionLock2DValidationError(
            "prediction lock campaigns must bind distinct public manifests"
        )
    runs = tuple(
        _locked_run(raw, f"prediction lock runs[{index}]")
        for index, raw in enumerate(_sequence(value["runs"], "prediction lock runs"))
    )
    if len(runs) != RUN_COUNT:
        raise PredictionLock2DValidationError(
            f"prediction lock requires exactly {RUN_COUNT} runs"
        )
    expected_cells = {
        (campaign_id, method_id, seed)
        for campaign_id in prereg.campaign_ids
        for method_id in METHOD_IDS
        for seed in TRAINING_SEEDS
    }
    actual_cells = {(run.campaign_id, run.method_id, run.training_seed) for run in runs}
    if actual_cells != expected_cells or len(actual_cells) != len(runs):
        raise PredictionLock2DValidationError(
            "prediction lock run matrix is incomplete/duplicate"
        )
    if list(runs) != sorted(
        runs, key=lambda item: (item.campaign_id, item.method_id, item.training_seed)
    ):
        raise PredictionLock2DValidationError(
            "prediction lock runs are not canonical ordered"
        )
    checkpoint_by_cell: dict[tuple[str, int], tuple[str, int]] = {}
    source_by_method: dict[str, tuple[str, str, str]] = {}
    for run in runs:
        observation = observation_by_campaign.get(run.campaign_id)
        if observation != (run.observations_sha256, run.observation_manifest_sha256):
            raise PredictionLock2DValidationError(
                "run observation binding differs from campaign"
            )
        checkpoint_key = (run.method_id, run.training_seed)
        previous_checkpoint = checkpoint_by_cell.setdefault(
            checkpoint_key, (run.checkpoint_sha256, run.checkpoint_size_bytes)
        )
        if previous_checkpoint != (run.checkpoint_sha256, run.checkpoint_size_bytes):
            raise PredictionLock2DValidationError(
                "checkpoint identity changes across campaigns for one method/seed"
            )
        source_key = (
            run.source_commit,
            run.source_sha256,
            run.adapter_source_sha256,
        )
        previous_source = source_by_method.setdefault(run.method_id, source_key)
        if previous_source != source_key:
            raise PredictionLock2DValidationError(
                "method source identity changes within lock"
            )
        implementation = _mapping(
            prereg.method_by_id[run.method_id]["implementation"],
            f"prereg {run.method_id}.implementation",
        )
        if (
            run.source_commit != implementation.get("repository_commit")
            or run.adapter_source_sha256 != implementation.get("adapter_source_sha256")
            or run.source_sha256
            != _preregistered_source_sha256(run.method_id, implementation)
        ):
            raise PredictionLock2DValidationError(
                "run source/adapter differs from prereg"
            )
    checkpoint_hashes = {value[0] for value in checkpoint_by_cell.values()}
    if len(checkpoint_hashes) != len(METHOD_IDS) * len(TRAINING_SEEDS):
        raise PredictionLock2DValidationError(
            "different method/seeds share checkpoint digest"
        )
    return ValidatedPredictionLock2D(
        preregistration_sha256=prereg.snapshot.sha256,
        lock_sha256=snapshot.sha256,
        input_manifest_sha256=input_sha,
        campaign_ids=tuple(campaign_ids),
        method_ids=METHOD_IDS,
        training_seeds=TRAINING_SEEDS,
        statistical_options=dict(statistical_options),
        runs=runs,
    )


def validate_locked_run_artifacts_2d(
    run: LockedRun2D,
    *,
    observations_path: str | Path,
    observation_manifest_path: str | Path,
    prediction_path: str | Path,
    runtime_path: str | Path,
    checkpoint_path: str | Path,
    source_path: str | Path,
) -> ValidatedLockedArtifacts2D:
    """Re-snapshot one locked run's public artifacts before operator access."""
    snapshots = ValidatedLockedArtifacts2D(
        observations=snapshot_regular_file(
            observations_path,
            expected_sha256=run.observations_sha256,
            role="locked observations",
        ),
        observation_manifest=snapshot_regular_file(
            observation_manifest_path,
            expected_sha256=run.observation_manifest_sha256,
            role="locked public observation manifest",
        ),
        prediction=snapshot_regular_file(
            prediction_path,
            expected_sha256=run.prediction_sha256,
            role="locked prediction",
        ),
        runtime=snapshot_regular_file(
            runtime_path,
            expected_sha256=run.runtime_sha256,
            role="locked runtime",
        ),
        checkpoint=snapshot_regular_file(
            checkpoint_path,
            expected_sha256=run.checkpoint_sha256,
            role="locked checkpoint",
        ),
        source=snapshot_regular_file(
            source_path,
            expected_sha256=run.source_sha256,
            role="locked upstream source",
        ),
    )
    if snapshots.prediction.size_bytes != run.prediction_size_bytes:
        raise PredictionLock2DValidationError("locked prediction size changed")
    if snapshots.runtime.size_bytes != run.runtime_size_bytes:
        raise PredictionLock2DValidationError("locked runtime size changed")
    if snapshots.checkpoint.size_bytes != run.checkpoint_size_bytes:
        raise PredictionLock2DValidationError("locked checkpoint size changed")
    identities = [
        snapshots.observations.identity,
        snapshots.observation_manifest.identity,
        snapshots.prediction.identity,
        snapshots.runtime.identity,
        snapshots.checkpoint.identity,
        snapshots.source.identity,
    ]
    if len(set(identities)) != len(identities):
        raise PredictionLock2DValidationError(
            "locked run artifacts alias one another by path or hardlink"
        )
    return snapshots


def _publication_parent_identity(path: Path) -> tuple[int, int]:
    try:
        info = os.lstat(path)
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise PredictionLock2DPublicationError(
            f"cannot inspect publication parent {path}: {exc}"
        ) from exc
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise PredictionLock2DPublicationError(
            f"publication parent must be a real directory: {path}"
        )
    if os.path.normcase(str(resolved)) != os.path.normcase(str(path.absolute())):
        raise PredictionLock2DPublicationError(
            f"publication parent must not traverse a symbolic link: {path}"
        )
    return int(info.st_dev), int(info.st_ino)


def _write_all_descriptor(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("zero-byte write while publishing prediction lock")
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


def _verify_new_prediction_lock_path(
    destination: Path,
    *,
    expected_identity: tuple[int, int],
    expected_parent_identity: tuple[int, int],
) -> None:
    try:
        current = os.lstat(destination)
    except OSError as exc:
        raise PredictionLock2DPublicationError(
            f"new prediction lock disappeared before writing: {destination}"
        ) from exc
    if (
        not stat.S_ISREG(current.st_mode)
        or stat.S_ISLNK(current.st_mode)
        or (int(current.st_dev), int(current.st_ino)) != expected_identity
        or int(current.st_nlink) != 1
    ):
        raise PredictionLock2DPublicationError(
            "new prediction lock path was replaced or acquired a hardlink alias"
        )
    if _publication_parent_identity(destination.parent) != expected_parent_identity:
        raise PredictionLock2DPublicationError(
            "prediction lock publication parent was replaced before writing"
        )


def _stable_prediction_lock_receipt(
    destination: Path,
    *,
    expected_identity: tuple[int, int],
    expected_parent_identity: tuple[int, int],
    expected_payload: bytes,
) -> PredictionLock2DPublicationReceipt:
    """Reopen and twice snapshot the sealed output through one descriptor."""
    descriptor: int | None = None
    try:
        preopen = os.lstat(destination)
        if not stat.S_ISREG(preopen.st_mode) or stat.S_ISLNK(preopen.st_mode):
            raise PredictionLock2DPublicationError(
                f"published lock is not a regular non-link file: {destination}"
            )
        descriptor = open_verified_publication(destination)
        presealed = os.fstat(descriptor)
        identity = (int(presealed.st_dev), int(presealed.st_ino))
        if identity != expected_identity or int(presealed.st_nlink) != 1:
            raise PredictionLock2DPublicationError(
                "published lock was replaced or acquired a hardlink alias"
            )
        set_publication_descriptor_read_only(descriptor)
        before = os.lstat(destination)
        opened = os.fstat(descriptor)
        if stat.S_IMODE(opened.st_mode) & 0o222:
            raise PredictionLock2DPublicationError(
                "published prediction lock is not sealed read-only"
            )
        first = _read_all_descriptor(descriptor)
        middle = os.fstat(descriptor)
        if _publication_parent_identity(destination.parent) != expected_parent_identity:
            raise PredictionLock2DPublicationError(
                "prediction lock publication parent was replaced"
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
            raise PredictionLock2DPublicationError(
                "published lock changed during final descriptor verification"
            )
        digest = hashlib.sha256(second).hexdigest()
        return PredictionLock2DPublicationReceipt(destination, digest, len(second))
    except PredictionLock2DPublicationError:
        raise
    except OSError as exc:
        raise PredictionLock2DPublicationError(
            f"cannot verify published prediction lock {destination}: {exc}"
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


def publish_prediction_lock_2d(
    lock: Mapping[str, Any], output_path: str | Path
) -> PredictionLock2DPublicationReceipt:
    """Exclusively create, seal, and descriptor-verify one canonical lock.

    Publication intentionally has no pathname rollback.  If an interruption or
    adversarial replacement makes ownership ambiguous, the uniquely and
    exclusively created artifact is left sealed for operator inspection.
    """
    destination = Path(os.path.abspath(output_path))
    ensure_real_directory(
        destination.parent,
        error_type=PredictionLock2DPublicationError,
        role="prediction lock publication parent",
    )
    parent_identity = _publication_parent_identity(destination.parent)
    if os.path.lexists(destination):
        raise PredictionLock2DPublicationError(f"refusing to overwrite {destination}")
    payload = canonical_json_bytes(lock)
    descriptor: int | None = None
    identity: tuple[int, int] | None = None
    try:
        try:
            descriptor = open_exclusive_publication(destination)
        except FileExistsError as exc:
            raise PredictionLock2DPublicationError(
                f"publication race: refusing to overwrite {destination}"
            ) from exc
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or int(opened.st_nlink) != 1:
            raise PredictionLock2DPublicationError(
                "new prediction lock is not a unique regular file"
            )
        identity = int(opened.st_dev), int(opened.st_ino)
        _verify_new_prediction_lock_path(
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
            raise PredictionLock2DPublicationError(
                "prediction lock changed before final verification"
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
    return _stable_prediction_lock_receipt(
        destination,
        expected_identity=identity,
        expected_parent_identity=parent_identity,
        expected_payload=payload,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create the complete pre-score PIMSR SOTA 2-D prediction lock"
    )
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--input-manifest-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    lock = create_prediction_lock_2d(
        args.preregistration,
        args.input_manifest,
        expected_preregistration_sha256=args.preregistration_sha256,
        expected_input_manifest_sha256=args.input_manifest_sha256,
    )
    receipt = publish_prediction_lock_2d(lock, args.output)
    print(f"published {receipt.path} sha256={receipt.sha256} size={receipt.size_bytes}")
    return 0


__all__ = [
    "LOCK_INPUT_SCHEMA",
    "LOCK_INPUT_SCHEMA_VERSION",
    "LOCK_SCHEMA",
    "LOCK_SCHEMA_VERSION",
    "ArtifactSnapshot",
    "LockedRun2D",
    "PredictionLock2DPublicationError",
    "PredictionLock2DPublicationReceipt",
    "PredictionLock2DValidationError",
    "ValidatedLockedArtifacts2D",
    "ValidatedPredictionLock2D",
    "canonical_json_bytes",
    "create_prediction_lock_2d",
    "publish_prediction_lock_2d",
    "snapshot_regular_file",
    "validate_locked_run_artifacts_2d",
    "validate_prediction_lock_2d",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
