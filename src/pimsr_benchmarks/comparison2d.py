"""Post-score three-method comparison for the locked PIMSR 2-D benchmark.

Truth, operator manifests and evaluation reports are opened only after the
global pre-score prediction lock has been validated.  Statistics are taken
verbatim from the externally pinned preregistration through the validated lock;
the comparator exposes no confidence, seed or resample-count overrides.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import stat
import subprocess
import sys
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from pimsr_benchmarks import evaluation2d as protected_evaluation2d
from pimsr_benchmarks._publication_io import (
    close_publication_descriptor,
    ensure_real_directory,
    open_exclusive_publication,
    open_verified_publication,
)
from pimsr_benchmarks.modem2d_forward import (
    CANONICAL_MODEL_SHAPE,
    CanonicalTruth,
    NestedMeshConfig,
    mapped_model,
)
from pimsr_benchmarks.prediction_lock2d import (
    ArtifactSnapshot,
    LockedRun2D,
    ValidatedPredictionLock2D,
    snapshot_regular_file,
    validate_prediction_lock_2d,
)

POST_SCORE_SCHEMA = "pimsr-sota-2d-post-score-comparison-manifest"
POST_SCORE_SCHEMA_VERSION = 3
COMPARISON_SCHEMA = "pimsr-sota-2d-three-method-comparison"
COMPARISON_SCHEMA_VERSION = 3
EVALUATION_SCHEMA = "pimsr-sota-2d-evaluation"
EVALUATION_SCHEMA_VERSION = 3
OPERATOR_MANIFEST_SCHEMA = "pimsr-sota-2d-scoring-manifest"
OPERATOR_MANIFEST_SCHEMA_VERSION = 3

METHOD_IDS = ("pimsr", "mtdlpy", "mt2dinv_densenet")
CANDIDATE_METHOD_ID = "pimsr"
REFERENCE_METHOD_IDS = ("mtdlpy", "mt2dinv_densenet")
TRAINING_SEEDS = (101, 102, 103, 104, 105)
CAMPAIGN_COUNT = 5
SAMPLES_PER_CAMPAIGN = 500
BASE_MODELS_PER_CAMPAIGN = 100
NOISE_REALIZATIONS_PER_BASE = 5
RUN_COUNT = CAMPAIGN_COUNT * len(METHOD_IDS) * len(TRAINING_SEEDS)
FAMILY_IDS = ("background", "aquifer", "hydrocarbon", "salt", "geothermal")
BASE_MODELS_PER_FAMILY = 20
PUBLIC_CONVERGENCE_RAW_RUN_COUNT = 25 * 3 + 4 + 1
_MU0 = 4.0e-7 * math.pi

METRIC_IDS = ("rmse_log10_resistivity", "mae_log10_resistivity")
PRIMARY_METRIC_ID = METRIC_IDS[0]
EXPECTED_RESAMPLING_LEVELS = (
    "training_seed",
    "campaign",
    "geological_family",
    "base_model_within_family",
    "noise_realization_within_base_model",
)
EXPECTED_POINT_AGGREGATION = (
    "equal_family_equal_base_equal_noise_mean_across_paired_training_seeds_and_campaigns"
)
EXPECTED_DOMINANCE_GATE = (
    "one_sided_95_percent_iut_upper_below_zero_against_both_references"
)
EXPECTED_MULTIPLICITY_POLICY = (
    "none_for_single_intersection_union_claim_individual_pairwise_descriptive"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$", flags=re.ASCII)
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$", flags=re.ASCII)
_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}$", flags=re.ASCII)
_OCI_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$", flags=re.ASCII)

_POST_SCORE_KEYS = frozenset(
    {
        "audience",
        "schema",
        "schema_version",
        "preregistration_sha256",
        "prediction_lock_sha256",
        "prediction_lock_input_manifest_sha256",
        "evaluator_implementation",
        "headline_evidence",
        "evidence_artifacts",
        "public_convergence_raw_runs",
        "campaigns",
    }
)
_CAMPAIGN_KEYS = frozenset(
    {
        "campaign_id",
        "public_observation_manifest",
        "operator_manifest",
        "withheld_truth",
        "hidden_generation",
        "evaluations",
    }
)
_EVALUATION_REFERENCE_KEYS = frozenset(
    {"method_id", "training_seed", "evaluation", "prediction"}
)
_ARTIFACT_REFERENCE_KEYS = frozenset({"path", "sha256", "size_bytes"})
_EVALUATOR_KEYS = frozenset({"repository_commit", "source_sha256"})
_EVALUATION_ROOT_KEYS = frozenset(
    {
        "audience",
        "schema",
        "schema_version",
        "inputs",
        "run",
        "metric_contract",
        "bootstrap_contract",
        "implementation",
        "physics_misfit",
        "release_gate",
        "overall",
        "by_scenario",
        "per_depth",
        "per_sample",
    }
)
_RUN_BINDING_KEYS = frozenset(
    {
        "adapter_source_sha256",
        "campaign_id",
        "checkpoint_sha256",
        "method_id",
        "runtime_sha256",
        "source_commit",
        "source_sha256",
        "training_seed",
    }
)
_PER_SAMPLE_KEYS = frozenset(
    {
        "has_fault",
        "mae_log10_resistivity",
        "rmse_log10_resistivity",
        "sample_index",
        "scenario",
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

_CONVERGENCE_THRESHOLDS = {
    "log10_rho_dex": {"median": 0.005, "p95": 0.015, "max": 0.05},
    "phase_circular_180_degrees": {
        "median": 0.10,
        "p95": 0.50,
        "max": 1.50,
    },
    "padding": {
        "log10_rho_p95_dex": 0.005,
        "phase_circular_180_p95_degrees": 0.20,
    },
}

_CONVERGENCE_REPORT_SCHEMA = "pimsr-modem2d-convergence-validation"
_CONVERGENCE_REPORT_SCHEMA_VERSION = 1
_CONVERGENCE_FAMILIES = FAMILY_IDS
_CONVERGENCE_CHANNELS = (
    "te_log10_rho",
    "te_phase_deg",
    "tm_log10_rho",
    "tm_phase_deg",
)
_CONVERGENCE_RESIDUAL_MEMBERS = (
    "sample_index",
    "scenario_index",
    "candidate_vs_reference_te_log10_rho",
    "candidate_vs_reference_te_phase_deg",
    "candidate_vs_reference_tm_log10_rho",
    "candidate_vs_reference_tm_phase_deg",
    "candidate_vs_padding_te_log10_rho",
    "candidate_vs_padding_te_phase_deg",
    "candidate_vs_padding_tm_log10_rho",
    "candidate_vs_padding_tm_phase_deg",
)
_MESH_RECORD_KEYS = frozenset(
    {
        "schema",
        "schema_version",
        "mesh_id",
        "version",
        "base_core_width_m",
        "base_core_count",
        "base_padding_count_each_side",
        "base_padding_growth",
        "minimum_vertical_subdivisions",
        "maximum_base_dz_m",
        "deep_padding_growth",
        "maximum_deep_macro_dz_m",
        "minimum_depth_m",
        "horizontal_refinement_factor",
        "vertical_refinement_factor",
        "canonical_depth_centres_sha256",
        "canonical_x_centres_sha256",
        "horizontal_partition",
        "vertical_partition",
        "mapping",
    }
)
_MESH_MAPPING = (
    "nearest canonical physical cell centre, tie to lower index; "
    "piecewise constant edge extension with invariant interfaces"
)
_CANONICAL_DEPTH_CENTRES_SHA256 = (
    "d6382014a4672008ffda4952e31ab91123b0b70e44610d8040625ec3c424636f"
)
_CANONICAL_X_CENTRES_SHA256 = (
    "a0a28ac3698ab0a519fc51e59a2bce7f92a0a3793ae1d3adb9d0567e80fd860b"
)
_HORIZONTAL_PARTITION = (
    "62.5m-aligned 24km core plus common geometric padding; exact "
    "equal-width subdivision by horizontal_refinement_factor"
)
_VERTICAL_PARTITION = (
    "surface zero, arithmetic midpoint canonical interfaces, "
    "extrapolated bottom edge, capped deep edge-extension; every base "
    "cell exact equal-width subdivision by vertical_refinement_factor"
)
_COMPARISON_IMPLEMENTATION_SCHEMA = "pimsr-sota-2d-comparison-implementation"
_COMPARISON_IMPLEMENTATION_SCHEMA_VERSION = 1
_COMPARISON_SOURCE_PATH = "src/pimsr_benchmarks/comparison2d.py"
_REQUIRED_COMPARISON_DEPENDENCIES = (
    "src/pimsr_benchmarks/_publication_io.py",
    "src/pimsr_benchmarks/dataset_lineage2d.py",
    "src/pimsr_benchmarks/evaluation2d.py",
    "src/pimsr_benchmarks/hidden_campaign2d.py",
    "src/pimsr_benchmarks/modem2d_forward.py",
    "src/pimsr_benchmarks/prediction_lock2d.py",
)
_PUBLIC_LINEAGE_SCHEMA = "pimsr-public-dataset-lineage-2d"
_PUBLIC_LINEAGE_SCHEMA_VERSION = 2
_PUBLIC_LINEAGE_EVIDENCE_SCOPE = (
    "artifact_lineage_and_transitive_generator_source_identity_"
    "without_forward_regeneration"
)
_SOURCE_DERIVED_GENERATION_SEMANTICS = {
    "base_layer_rng": "numpy.default_rng([generator_seed,sample_index])",
    "base_layer_scenario": "forced_background_before_2d_scenario_injection",
    "scenario_policy": "SectionGenerator.sample(sample_index,scenario=None)",
    "section_rng": "numpy.default_rng([generator_seed,2,sample_index])",
    "sensor_rng": "numpy.default_rng([generator_seed,3,sample_index])",
    "status": "derived_from_exact_pinned_source_closure_not_generation_time_execution",
}
_LINEAGE_ROW_ARRAYS = frozenset(
    {
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
    }
)
_LINEAGE_COORDINATE_ARRAYS = frozenset(
    {"frequencies", "station_x", "x_grid", "depth_grid"}
)
_LINEAGE_ROOT_STRING_ATTRIBUTES = {
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
_LINEAGE_SENSOR_PARAMETERS = {
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
        "distort_hi": {
            "distribution": "log_uniform",
            "high": 0.45,
            "low": 0.25,
        },
        "shift_sigma": {
            "distribution": "uniform",
            "high": 0.32,
            "low": 0.15,
        },
    },
}
_FAMILY_COMMITMENT_SCHEMA = "pimsr-sota-2d-family-partition-commitment"
_FAMILY_COMMITMENT_SCHEMA_VERSION = 1
_FAMILY_REVEAL_SCHEMA = "pimsr-sota-2d-family-partition-reveal"
_FAMILY_REVEAL_SCHEMA_VERSION = 1
_FAMILY_COMMITMENT_DOMAIN = b"pimsr-sota-2d-family-partition/v1\x00"
_PUBLIC_RAW_RUN_SCHEMA = "pimsr-modem2d-public-convergence-raw-run-set"
_PUBLIC_RAW_RUN_SCHEMA_VERSION = 2
_HIDDEN_GENERATION_SCHEMA = "pimsr-modem2d-hidden-generation-closure"
_HIDDEN_GENERATION_SCHEMA_VERSION = 3
_HIDDEN_GENERATION_CONTRACT_SCHEMA = "pimsr-modem2d-hidden-generation-contract"
_HIDDEN_GENERATION_CONTRACT_SCHEMA_VERSION = 2
_HIDDEN_BASE_LAYER_RNG = "numpy.default_rng([generator_seed,base_index])"
_HIDDEN_SECTION_RNG = "numpy.default_rng([generator_seed,2,base_index])"
_HIDDEN_BASE_LAYER_SCENARIO = "forced_background_before_2d_scenario_injection"
_HIDDEN_SCENARIO_POLICY = "SectionGenerator.sample(base_index,scenario=family_id)"
_HIDDEN_NOISE_RNG = "numpy.default_rng([generator_seed,3,base_index,noise_index])"
_HIDDEN_GENERATION_RUNTIME = {
    "python_version": "3.11.15",
    "numpy_version": "2.4.6",
    "pimsr_geogen_version": "0.2.0",
    "pimsr_forward_version": "0.2.0",
}
_HIDDEN_RUNTIME_MANIFEST_SCHEMA = "pimsr-hidden-generation-runtime-2d"
_HIDDEN_RUNTIME_MANIFEST_SCHEMA_VERSION = 1
_HIDDEN_RUNTIME_DISTRIBUTIONS = {
    "numpy": "2.4.6",
    "h5py": "3.16.0",
    "pimsr_benchmarks": "0.2.0",
    "pimsr_geogen": "0.2.0",
    "pimsr_forward": "0.2.0",
}
_OBSERVATION_ARRAY_MEMBERS = (
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


class Comparison2DValidationError(ValueError):
    """Raised when post-score evidence is incomplete or inconsistent."""


class Comparison2DPublicationError(RuntimeError):
    """Raised when a comparison cannot be immutably published."""


@dataclass(frozen=True)
class EffectRow2D:
    """One paired method-effect row at the deepest hierarchy level."""

    campaign_id: str
    training_seed: int
    family_id: str
    base_model_id: str
    noise_id: str
    # Shape is reference x metric, ordered by REFERENCE_METHOD_IDS/METRIC_IDS.
    effects: np.ndarray


@dataclass(frozen=True)
class HierarchicalBootstrap2D:
    point: np.ndarray
    two_sided_lower: np.ndarray
    two_sided_upper: np.ndarray
    one_sided_upper: np.ndarray
    samples: np.ndarray
    family_points: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class _Hierarchy:
    sample_ids: tuple[int, ...]
    families: tuple[str, ...]
    # family -> ((base_model_id, ((noise_id, sample_id), ...)), ...)
    tree: Mapping[str, tuple[tuple[str, tuple[tuple[str, int], ...]], ...]]


@dataclass(frozen=True)
class _Campaign:
    campaign_id: str
    operator_sha256: str
    operator_size_bytes: int
    observations_sha256: str
    observation_manifest_sha256: str
    truth_sha256: str
    truth_size_bytes: int
    hierarchy: _Hierarchy
    source_sample_ids: Mapping[int, int]
    family_commitment_sha256: str | None
    generation_evidence_proven: bool
    generation_evidence_reason: str | None


@dataclass(frozen=True)
class _Evaluation:
    sha256: str
    metrics: np.ndarray
    sample_ids: tuple[int, ...]
    metric_contract: Mapping[str, Any]
    physics_misfit_included: bool


@dataclass(frozen=True)
class _MaterialTruth2D:
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
class _MaterialPredictions2D:
    sample_index: np.ndarray
    observations_sha256: str
    x_cell_centers_m: np.ndarray
    depth_cell_centers_m: np.ndarray
    log10_resistivity: np.ndarray
    artifact_sha256: str
    artifact_size_bytes: int


@dataclass(frozen=True)
class _RawModEMResponse:
    frequencies_hz: np.ndarray
    station_y_m: np.ndarray
    log10_rho_te: np.ndarray
    phase_te_deg: np.ndarray
    log10_rho_tm: np.ndarray
    phase_tm_deg: np.ndarray


@dataclass(frozen=True)
class _PreregContracts:
    evaluator: Mapping[str, Any]
    headline_evidence: Mapping[str, Any]
    comparison_implementation: Mapping[str, Any]
    public_lineage: Mapping[str, Mapping[str, Any]]
    family_partition: Mapping[str, Any]
    hidden_seed_commitment: Mapping[str, Any]


@dataclass(frozen=True)
class _PublicRawValidation:
    residuals: Mapping[str, np.ndarray]
    analytic_residuals: Mapping[tuple[str, str], Mapping[str, np.ndarray]]
    identities_sha256: str
    artifact_count: int


@dataclass(frozen=True)
class _PublicShardMaterial:
    artifact_sha256: str
    artifact_size_bytes: int
    generator_seed: int
    generation_contract: str
    forward_contract: str
    row_by_sample: Mapping[int, int]
    scenario_by_sample: Mapping[int, int]
    truth_identity_by_sample: Mapping[int, Mapping[str, Any]]


@dataclass(frozen=True)
class PublishedComparison2D:
    """Receipt for the final bytes observed through a reopened descriptor."""

    path: Path
    sha256: str
    size_bytes: int


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], path: str) -> None:
    actual = set(value)
    if actual != expected:
        raise Comparison2DValidationError(
            f"{path} keys mismatch; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise Comparison2DValidationError(f"{path} must be a JSON object")
    return value


def _sequence(value: Any, path: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise Comparison2DValidationError(f"{path} must be a JSON array")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise Comparison2DValidationError(f"{path} must be a non-empty NUL-free string")
    return value


def _identifier(value: Any, path: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise Comparison2DValidationError(f"{path} is not a canonical identifier")
    return value


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise Comparison2DValidationError(
            f"{path} must be an integer greater than or equal to {minimum}"
        )
    return value


def _finite(value: Any, path: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Comparison2DValidationError(f"{path} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise Comparison2DValidationError(f"{path} is not in its finite domain")
    return result


def _sha256(value: Any, path: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise Comparison2DValidationError(f"{path} must be a lowercase SHA-256")
    return value


def _git_commit(value: Any, path: str) -> str:
    if not isinstance(value, str) or not _GIT_COMMIT_RE.fullmatch(value):
        raise Comparison2DValidationError(f"{path} must be a full lowercase Git commit")
    return value


def _canonical_json_attribute(value: Any, path: str) -> Any:
    text = _string(value, path)
    try:
        parsed = json.loads(
            text,
            parse_constant=lambda item: (_ for _ in ()).throw(
                Comparison2DValidationError(f"{path} contains {item}")
            ),
        )
        canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
    except Comparison2DValidationError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise Comparison2DValidationError(f"{path} is not canonical JSON: {exc}") from exc
    if text != canonical:
        raise Comparison2DValidationError(f"{path} is not canonical JSON")
    return parsed


def _json_bytes(value: Any, *, publication: bool) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        error = (
            Comparison2DPublicationError if publication else Comparison2DValidationError
        )
        raise error(f"value is not finite canonical JSON: {exc}") from exc
    return (text + "\n").encode("utf-8")


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Return compact, finite, deterministic newline-terminated JSON."""
    return _json_bytes(value, publication=True)


def _strict_json(
    snapshot: ArtifactSnapshot,
    role: str,
    *,
    require_canonical: bool = True,
) -> Mapping[str, Any]:
    def object_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise Comparison2DValidationError(
                    f"{role} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise Comparison2DValidationError(
            f"{role} contains non-finite constant {value!r}"
        )

    try:
        decoded = json.loads(
            snapshot.payload.decode("utf-8", errors="strict"),
            object_pairs_hook=object_hook,
            parse_constant=reject_constant,
        )
    except Comparison2DValidationError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Comparison2DValidationError(f"cannot decode {role}: {exc}") from exc
    result = _mapping(decoded, role)
    if require_canonical and _json_bytes(result, publication=False) != snapshot.payload:
        raise Comparison2DValidationError(f"{role} is not canonical JSON")
    return result


def _snapshot_unique(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int | None,
    role: str,
    seen_identities: set[tuple[int, int]],
) -> ArtifactSnapshot:
    try:
        snapshot = snapshot_regular_file(
            path,
            expected_sha256=_sha256(expected_sha256, f"{role}.sha256"),
            role=role,
        )
    except Exception as exc:
        raise Comparison2DValidationError(str(exc)) from exc
    if expected_size_bytes is not None and snapshot.size_bytes != _integer(
        expected_size_bytes, f"{role}.size_bytes", minimum=1
    ):
        raise Comparison2DValidationError(f"{role} size differs from its pin")
    info = os.lstat(snapshot.path)
    if (
        not stat.S_ISREG(info.st_mode)
        or (int(info.st_dev), int(info.st_ino)) != snapshot.identity
    ):
        raise Comparison2DValidationError(f"{role} changed after its safe snapshot")
    if int(info.st_nlink) != 1:
        raise Comparison2DValidationError(f"{role} must not have hardlink aliases")
    if snapshot.identity in seen_identities:
        raise Comparison2DValidationError(f"{role} aliases another artifact inode")
    seen_identities.add(snapshot.identity)
    return snapshot


def _artifact_reference(
    value: Any,
    *,
    base: Path,
    role: str,
    seen_identities: set[tuple[int, int]],
) -> ArtifactSnapshot:
    reference = _mapping(value, role)
    _exact_keys(reference, _ARTIFACT_REFERENCE_KEYS, role)
    path_text = _string(reference["path"], f"{role}.path")
    requested = Path(path_text)
    path = requested if requested.is_absolute() else base / requested
    return _snapshot_unique(
        path,
        expected_sha256=_sha256(reference["sha256"], f"{role}.sha256"),
        expected_size_bytes=_integer(
            reference["size_bytes"], f"{role}.size_bytes", minimum=1
        ),
        role=role,
        seen_identities=seen_identities,
    )


def _portable_artifact_reference(
    value: Any,
    *,
    base: Path,
    role: str,
    seen_identities: set[tuple[int, int]],
) -> ArtifactSnapshot:
    reference = _mapping(value, role)
    _exact_keys(reference, _ARTIFACT_REFERENCE_KEYS, role)
    path_text = _string(reference["path"], f"{role}.path")
    requested = Path(path_text)
    if (
        requested.is_absolute()
        or bool(requested.drive)
        or bool(requested.root)
        or not requested.parts
        or any(part in {".", ".."} for part in requested.parts)
        or requested.as_posix() != path_text
    ):
        raise Comparison2DValidationError(
            f"{role}.path must be a normalized portable relative path"
        )
    return _artifact_reference(
        reference,
        base=base,
        role=role,
        seen_identities=seen_identities,
    )


def _evaluation_snapshot(snapshot: ArtifactSnapshot) -> Any:
    return protected_evaluation2d._ArtifactSnapshot(
        snapshot.path,
        snapshot.payload,
        snapshot.sha256,
    )


def _load_material_truth(snapshot: ArtifactSnapshot) -> _MaterialTruth2D:
    """Decode withheld truth only from the already captured stable bytes."""
    try:
        arrays = protected_evaluation2d._load_npz_arrays(
            _evaluation_snapshot(snapshot),
            expected_keys=protected_evaluation2d._TRUTH_KEYS,
            expected_order=protected_evaluation2d._TRUTH_KEY_ORDER,
            artifact_name="truth",
        )
        protected_evaluation2d._require_scalar_schema(
            arrays["schema"],
            expected=protected_evaluation2d.TRUTH_SCHEMA,
            artifact_name="truth",
        )
        protected_evaluation2d._require_version(
            arrays["schema_version"],
            "truth",
            protected_evaluation2d.TRUTH_SCHEMA_VERSION,
        )
        sample_index = protected_evaluation2d._validate_sample_indices(
            arrays["sample_index"], "truth.sample_index"
        )
        observations_sha256 = protected_evaluation2d._require_sha256_scalar(
            arrays["observations_sha256"], "truth.observations_sha256"
        )
        scenario = arrays["scenario"]
        if (
            scenario.ndim != 1
            or scenario.dtype.kind != "U"
            or scenario.shape != sample_index.shape
            or any(not value or "\x00" in value for value in scenario.tolist())
        ):
            raise Comparison2DValidationError("truth scenario array is invalid")
        has_fault = protected_evaluation2d._require_array(
            arrays["has_fault"],
            name="truth.has_fault",
            dtype=np.bool_,
            ndim=1,
        )
        if has_fault.shape != sample_index.shape:
            raise Comparison2DValidationError("truth fault flags do not match sample ids")
        x_centres = protected_evaluation2d._validate_centers(
            arrays["x_cell_centers_m"], "truth.x_cell_centers_m"
        )
        depth_centres = protected_evaluation2d._validate_centers(
            arrays["depth_cell_centers_m"], "truth.depth_cell_centers_m"
        )
        values = protected_evaluation2d._require_array(
            arrays["truth_log10_resistivity"],
            name="truth.truth_log10_resistivity",
            dtype="<f4",
            ndim=3,
        )
        if (
            values.shape != (sample_index.size, *CANONICAL_MODEL_SHAPE)
            or x_centres.shape != (CANONICAL_MODEL_SHAPE[1],)
            or depth_centres.shape != (CANONICAL_MODEL_SHAPE[0],)
            or not np.isfinite(values).all()
        ):
            raise Comparison2DValidationError(
                "truth material must use the exact finite canonical 64x48 grid"
            )
    except Comparison2DValidationError:
        raise
    except Exception as exc:
        raise Comparison2DValidationError(
            f"cannot validate withheld truth material: {exc}"
        ) from exc
    return _MaterialTruth2D(
        sample_index=sample_index,
        observations_sha256=observations_sha256,
        scenario=scenario,
        has_fault=has_fault,
        x_cell_centers_m=x_centres,
        depth_cell_centers_m=depth_centres,
        log10_resistivity=values,
        artifact_sha256=snapshot.sha256,
        artifact_size_bytes=snapshot.size_bytes,
    )


def _load_material_predictions(snapshot: ArtifactSnapshot) -> _MaterialPredictions2D:
    """Decode locked predictions only from the already captured stable bytes."""
    try:
        arrays = protected_evaluation2d._load_npz_arrays(
            _evaluation_snapshot(snapshot),
            expected_keys=protected_evaluation2d._PREDICTION_KEYS,
            expected_order=protected_evaluation2d._PREDICTION_KEY_ORDER,
            artifact_name="prediction",
        )
        protected_evaluation2d._require_scalar_schema(
            arrays["schema"],
            expected=protected_evaluation2d.PREDICTION_SCHEMA,
            artifact_name="prediction",
        )
        protected_evaluation2d._require_version(
            arrays["schema_version"],
            "prediction",
            protected_evaluation2d.PREDICTION_SCHEMA_VERSION,
        )
        sample_index = protected_evaluation2d._validate_sample_indices(
            arrays["sample_index"], "prediction.sample_index"
        )
        observations_sha256 = protected_evaluation2d._require_sha256_scalar(
            arrays["observations_sha256"], "prediction.observations_sha256"
        )
        x_centres = protected_evaluation2d._validate_centers(
            arrays["x_cell_centers_m"], "prediction.x_cell_centers_m"
        )
        depth_centres = protected_evaluation2d._validate_centers(
            arrays["depth_cell_centers_m"], "prediction.depth_cell_centers_m"
        )
        values = protected_evaluation2d._require_array(
            arrays["predicted_log10_resistivity"],
            name="prediction.predicted_log10_resistivity",
            dtype="<f4",
            ndim=3,
        )
        if (
            values.shape != (sample_index.size, *CANONICAL_MODEL_SHAPE)
            or x_centres.shape != (CANONICAL_MODEL_SHAPE[1],)
            or depth_centres.shape != (CANONICAL_MODEL_SHAPE[0],)
            or not np.isfinite(values).all()
        ):
            raise Comparison2DValidationError(
                "prediction material must use the exact finite canonical 64x48 grid"
            )
    except Comparison2DValidationError:
        raise
    except Exception as exc:
        raise Comparison2DValidationError(
            f"cannot validate locked prediction material: {exc}"
        ) from exc
    return _MaterialPredictions2D(
        sample_index=sample_index,
        observations_sha256=observations_sha256,
        x_cell_centers_m=x_centres,
        depth_cell_centers_m=depth_centres,
        log10_resistivity=values,
        artifact_sha256=snapshot.sha256,
        artifact_size_bytes=snapshot.size_bytes,
    )


def _recomputed_material_metrics(
    truth: _MaterialTruth2D,
    predictions: _MaterialPredictions2D,
) -> tuple[tuple[int, ...], np.ndarray]:
    if (
        predictions.observations_sha256 != truth.observations_sha256
        or not np.array_equal(predictions.x_cell_centers_m, truth.x_cell_centers_m)
        or not np.array_equal(
            predictions.depth_cell_centers_m, truth.depth_cell_centers_m
        )
    ):
        raise Comparison2DValidationError(
            "prediction material does not bind the exact withheld truth grid"
        )
    truth_positions = {
        int(sample_id): index for index, sample_id in enumerate(truth.sample_index)
    }
    prediction_positions = {
        int(sample_id): index for index, sample_id in enumerate(predictions.sample_index)
    }
    if set(truth_positions) != set(prediction_positions):
        raise Comparison2DValidationError(
            "prediction material sample ids differ from withheld truth"
        )
    sample_ids = tuple(sorted(truth_positions))
    truth_rows = truth.log10_resistivity[
        [truth_positions[sample_id] for sample_id in sample_ids]
    ].astype(np.float64)
    prediction_rows = predictions.log10_resistivity[
        [prediction_positions[sample_id] for sample_id in sample_ids]
    ].astype(np.float64)
    x_widths = protected_evaluation2d.cell_widths_from_centers(truth.x_cell_centers_m)
    depth_widths = protected_evaluation2d.cell_widths_from_centers(
        truth.depth_cell_centers_m
    )
    weights = (depth_widths / np.sum(depth_widths))[:, None] * (
        x_widths / np.sum(x_widths)
    )[None, :]
    difference = prediction_rows - truth_rows
    rmse = np.sqrt(np.sum(difference * difference * weights[None, :, :], axis=(1, 2)))
    mae = np.sum(np.abs(difference) * weights[None, :, :], axis=(1, 2))
    metrics = np.column_stack((rmse, mae))
    if not np.isfinite(metrics).all():
        raise Comparison2DValidationError("recomputed material metrics are non-finite")
    return sample_ids, metrics


def _validate_statistical_options(value: Mapping[str, Any]) -> dict[str, Any]:
    options = _mapping(value, "validated lock statistical_options")
    _exact_keys(options, _STATISTICAL_OPTION_KEYS, "statistical_options")
    if options["candidate_method_id"] != CANDIDATE_METHOD_ID:
        raise Comparison2DValidationError("statistical candidate is not pimsr")
    if tuple(options["reference_method_ids"]) != REFERENCE_METHOD_IDS:
        raise Comparison2DValidationError("statistical references are not exact")
    confidence = _finite(options["confidence"], "statistical confidence")
    if confidence != 0.95:
        raise Comparison2DValidationError("pairwise/IUT confidence must be exactly 0.95")
    n_resamples = _integer(options["n_resamples"], "statistical n_resamples")
    if n_resamples < 10_000:
        raise Comparison2DValidationError("hierarchical bootstrap requires >=10000 draws")
    rng_seed = _integer(options["rng_seed"], "statistical rng_seed")
    if rng_seed > np.iinfo(np.uint64).max:
        raise Comparison2DValidationError("statistical rng_seed exceeds uint64")
    if tuple(options["resampling_levels"]) != EXPECTED_RESAMPLING_LEVELS:
        raise Comparison2DValidationError("preregistered hierarchy levels are not exact")
    expected_literals = {
        "point_aggregation": EXPECTED_POINT_AGGREGATION,
        "dominance_gate": EXPECTED_DOMINANCE_GATE,
        "multiplicity_policy": EXPECTED_MULTIPLICITY_POLICY,
    }
    for key, expected in expected_literals.items():
        if _string(options[key], f"statistical_options.{key}") != expected:
            raise Comparison2DValidationError(
                f"statistical_options.{key} is not the preregistered exact protocol"
            )
    return dict(options)


def _hierarchy_index(
    rows: Sequence[EffectRow2D],
    *,
    training_seeds: tuple[int, ...],
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    Mapping[str, Any],
]:
    if not rows:
        raise Comparison2DValidationError("paired effect rows must not be empty")
    nested: dict[
        str,
        dict[str, dict[str, dict[str, dict[int, np.ndarray]]]],
    ] = {}
    for index, row in enumerate(rows):
        if row.training_seed not in training_seeds:
            raise Comparison2DValidationError(f"effect row {index} has an unknown seed")
        values = np.asarray(row.effects, dtype=np.float64)
        if values.shape != (len(REFERENCE_METHOD_IDS), len(METRIC_IDS)):
            raise Comparison2DValidationError(f"effect row {index} shape is not 2x2")
        if not np.isfinite(values).all():
            raise Comparison2DValidationError(f"effect row {index} is non-finite")
        seed_map = (
            nested.setdefault(row.campaign_id, {})
            .setdefault(row.family_id, {})
            .setdefault(row.base_model_id, {})
            .setdefault(row.noise_id, {})
        )
        if row.training_seed in seed_map:
            raise Comparison2DValidationError(
                "paired effect hierarchy has duplicate cells"
            )
        seed_map[row.training_seed] = values
    campaigns = tuple(sorted(nested))
    family_sets = {tuple(sorted(nested[campaign])) for campaign in campaigns}
    if len(family_sets) != 1:
        raise Comparison2DValidationError("campaigns must preserve the same family ids")
    families = next(iter(family_sets))
    for campaign in campaigns:
        for family in families:
            bases = nested[campaign][family]
            if not bases:
                raise Comparison2DValidationError("every family needs a base model")
            noise_counts = {len(noises) for noises in bases.values()}
            if len(noise_counts) != 1:
                raise Comparison2DValidationError(
                    "base models within a stratum require the same noise count"
                )
            for noises in bases.values():
                if not noises:
                    raise Comparison2DValidationError(
                        "every base needs a noise realization"
                    )
                for seed_map in noises.values():
                    if set(seed_map) != set(training_seeds):
                        raise Comparison2DValidationError(
                            "every noise realization requires all paired training seeds"
                        )
    return campaigns, families, nested


def hierarchical_paired_bootstrap_2d(
    rows: Sequence[EffectRow2D],
    *,
    training_seeds: tuple[int, ...],
    confidence: float,
    n_resamples: int,
    rng_seed: int,
) -> HierarchicalBootstrap2D:
    """Run the preregistered globally paired five-level hierarchy.

    Training-seed draws are global for a replicate and are therefore shared by
    every campaign, geological group and both method references.  Campaigns,
    families, bases within family and noise rows within base are sampled with
    replacement; the same draws are shared by both references.
    """
    if training_seeds != TRAINING_SEEDS:
        raise Comparison2DValidationError("training seeds must be exactly 101..105")
    confidence_value = _finite(confidence, "bootstrap confidence")
    if confidence_value != 0.95:
        raise Comparison2DValidationError("bootstrap confidence must be 0.95")
    count = _integer(n_resamples, "bootstrap n_resamples")
    if count < 10_000:
        raise Comparison2DValidationError("bootstrap requires at least 10000 draws")
    seed_value = _integer(rng_seed, "bootstrap rng_seed")
    campaigns, families, nested = _hierarchy_index(rows, training_seeds=training_seeds)
    tensors: dict[str, dict[str, np.ndarray]] = {}
    for campaign in campaigns:
        tensors[campaign] = {}
        for family in families:
            bases = nested[campaign][family]
            tensors[campaign][family] = np.stack(
                [
                    np.stack(
                        [
                            np.stack(
                                [noises[noise][seed] for seed in training_seeds],
                                axis=0,
                            )
                            for noise in sorted(noises)
                        ],
                        axis=0,
                    )
                    for base, noises in sorted(bases.items())
                ],
                axis=0,
            )

    family_points = {
        family: np.mean(
            np.stack(
                [
                    np.mean(tensors[campaign][family], axis=(0, 1, 2))
                    for campaign in campaigns
                ]
            ),
            axis=0,
        )
        for family in families
    }
    point = np.mean(np.stack(list(family_points.values())), axis=0)
    rng = np.random.Generator(np.random.PCG64(seed_value))
    samples = np.empty(
        (count, len(REFERENCE_METHOD_IDS), len(METRIC_IDS)), dtype=np.float64
    )
    for replicate in range(count):
        selected_seed_indices = rng.integers(
            0, len(training_seeds), size=len(training_seeds)
        )
        selected_campaign_indices = rng.integers(0, len(campaigns), size=len(campaigns))
        campaign_values: list[np.ndarray] = []
        for raw_campaign_index in selected_campaign_indices:
            campaign = campaigns[int(raw_campaign_index)]
            family_values: list[np.ndarray] = []
            selected_family_indices = rng.integers(0, len(families), size=len(families))
            for raw_family_index in selected_family_indices:
                family = families[int(raw_family_index)]
                tensor = tensors[campaign][family]
                base_count, noise_count = tensor.shape[:2]
                selected_base_indices = rng.integers(0, base_count, size=base_count)
                selected_noise_indices = rng.integers(
                    0, noise_count, size=(base_count, noise_count)
                )
                selected = tensor[
                    selected_base_indices[:, None, None],
                    selected_noise_indices[:, :, None],
                    selected_seed_indices[None, None, :],
                ]
                family_values.append(np.mean(selected, axis=(0, 1, 2)))
            campaign_values.append(np.mean(np.stack(family_values), axis=0))
        samples[replicate] = np.mean(np.stack(campaign_values), axis=0)
    alpha = (1.0 - confidence_value) / 2.0
    return HierarchicalBootstrap2D(
        point=point,
        two_sided_lower=np.quantile(samples, alpha, axis=0, method="linear"),
        two_sided_upper=np.quantile(samples, 1.0 - alpha, axis=0, method="linear"),
        one_sided_upper=np.quantile(samples, confidence_value, axis=0, method="linear"),
        samples=samples,
        family_points=family_points,
    )


def _relative_source_pin(value: Any, path: str) -> dict[str, Any]:
    record = _mapping(value, path)
    _exact_keys(record, _ARTIFACT_REFERENCE_KEYS, path)
    path_text = _string(record["path"], f"{path}.path")
    candidate = Path(path_text)
    if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
        raise Comparison2DValidationError(
            f"{path}.path must be a repository-relative path"
        )
    return {
        "path": candidate.as_posix(),
        "sha256": _sha256(record["sha256"], f"{path}.sha256"),
        "size_bytes": _integer(record["size_bytes"], f"{path}.size_bytes", minimum=1),
    }


def _comparison_implementation_contract(value: Any) -> Mapping[str, Any]:
    contract = _mapping(value, "preregistration.comparison_contract")
    _exact_keys(
        contract,
        frozenset(
            {
                "schema",
                "schema_version",
                "repository_commit",
                "source",
                "dependencies",
                "protected_paths",
            }
        ),
        "preregistration.comparison_contract",
    )
    if (
        contract["schema"] != _COMPARISON_IMPLEMENTATION_SCHEMA
        or contract["schema_version"] != _COMPARISON_IMPLEMENTATION_SCHEMA_VERSION
    ):
        raise Comparison2DValidationError(
            "comparison implementation contract identity is wrong"
        )
    commit = _git_commit(contract["repository_commit"], "comparison repository_commit")
    source = _relative_source_pin(contract["source"], "comparison_contract.source")
    if source["path"] != _COMPARISON_SOURCE_PATH:
        raise Comparison2DValidationError("comparison source path is not exact")
    raw_dependencies = _sequence(
        contract["dependencies"], "comparison_contract.dependencies"
    )
    dependencies = tuple(
        _relative_source_pin(item, f"comparison_contract.dependencies[{index}]")
        for index, item in enumerate(raw_dependencies)
    )
    dependency_paths = tuple(item["path"] for item in dependencies)
    if dependency_paths != _REQUIRED_COMPARISON_DEPENDENCIES:
        raise Comparison2DValidationError(
            "comparison dependency source paths are not exact"
        )
    protected_values = _sequence(
        contract["protected_paths"], "comparison_contract.protected_paths"
    )
    protected = tuple(
        _string(item, f"comparison_contract.protected_paths[{index}]")
        for index, item in enumerate(protected_values)
    )
    expected_protected = (_COMPARISON_SOURCE_PATH, *_REQUIRED_COMPARISON_DEPENDENCIES)
    if protected != expected_protected:
        raise Comparison2DValidationError("comparison protected paths are not exact")
    return {
        "schema": contract["schema"],
        "schema_version": contract["schema_version"],
        "repository_commit": commit,
        "source": source,
        "dependencies": dependencies,
        "protected_paths": protected,
    }


def _public_lineage_contract(value: Any) -> Mapping[str, Mapping[str, Any]]:
    contract = _mapping(value, "preregistration.public_dataset_lineage")
    _exact_keys(contract, frozenset({"train", "validation"}), "public_dataset_lineage")
    result: dict[str, Mapping[str, Any]] = {}
    for split in ("train", "validation"):
        record = _relative_source_pin(contract[split], f"public_dataset_lineage.{split}")
        _require_public_path(record["path"], f"public_dataset_lineage.{split}.path")
        result[split] = record
    return result


def _family_partition_contract(value: Any) -> Mapping[str, Any]:
    contract = _mapping(value, "preregistration.family_partition")
    _exact_keys(
        contract,
        frozenset(
            {
                "schema",
                "schema_version",
                "families",
                "bases_per_family",
                "noise_realizations_per_base",
                "commitment_contract",
            }
        ),
        "family_partition",
    )
    if (
        contract["schema"] != _FAMILY_COMMITMENT_SCHEMA
        or contract["schema_version"] != _FAMILY_COMMITMENT_SCHEMA_VERSION
        or tuple(_sequence(contract["families"], "family_partition.families"))
        != FAMILY_IDS
        or contract["bases_per_family"] != BASE_MODELS_PER_FAMILY
        or contract["noise_realizations_per_base"] != NOISE_REALIZATIONS_PER_BASE
    ):
        raise Comparison2DValidationError("family partition design is not exact")
    commitment_contract = _mapping(
        contract["commitment_contract"], "family_partition.commitment_contract"
    )
    expected_commitment_contract = {
        "algorithm": "SHA-256",
        "canonicalization": "utf8-canonical-json-sort-keys-compact-newline-v1",
        "domain_separator": "pimsr-sota-2d-family-partition/v1",
        "nonce_encoding": "lowercase_hex_32_bytes",
    }
    if dict(commitment_contract) != expected_commitment_contract:
        raise Comparison2DValidationError(
            "family partition commitment contract is not exact"
        )
    return {
        **dict(contract),
        "commitment_contract": expected_commitment_contract,
    }


def _hidden_seed_commitment(preregistration: Mapping[str, Any]) -> Mapping[str, Any]:
    datasets = _mapping(preregistration.get("datasets"), "preregistration.datasets")
    hidden = _mapping(datasets.get("hidden_test"), "datasets.hidden_test")
    commitment = _mapping(
        hidden.get("seed_commitment"), "datasets.hidden_test.seed_commitment"
    )
    expected_encoding = "utf8-canonical-json-int64-array-no-newline-v1"
    _exact_keys(
        commitment,
        frozenset({"encoding", "sha256"}),
        "datasets.hidden_test.seed_commitment",
    )
    if commitment["encoding"] != expected_encoding:
        raise Comparison2DValidationError(
            "hidden campaign seed commitment encoding is not frozen"
        )
    return {
        "encoding": expected_encoding,
        "sha256": _sha256(
            commitment["sha256"], "datasets.hidden_test.seed_commitment.sha256"
        ),
    }


def _validate_hidden_campaign_seed_reveal(
    seeds: Sequence[int], commitment: Mapping[str, Any]
) -> Mapping[str, Any]:
    if len(seeds) != CAMPAIGN_COUNT or len(set(seeds)) != CAMPAIGN_COUNT:
        raise Comparison2DValidationError(
            "hidden campaign generator seeds must be five distinct int64 values"
        )
    normalized = [
        _integer(seed, f"hidden campaign generator seeds[{index}]")
        for index, seed in enumerate(seeds)
    ]
    if any(seed > np.iinfo(np.int64).max for seed in normalized):
        raise Comparison2DValidationError("hidden campaign generator seed exceeds int64")
    expected_encoding = "utf8-canonical-json-int64-array-no-newline-v1"
    if commitment.get("encoding") != expected_encoding:
        raise Comparison2DValidationError(
            "hidden campaign seed commitment encoding is not frozen"
        )
    payload = json.dumps(
        normalized,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != commitment.get("sha256"):
        raise Comparison2DValidationError(
            "revealed hidden campaign generator seeds do not match preregistration"
        )
    return {
        "encoding": expected_encoding,
        "sha256": actual_sha256,
        "campaign_count": CAMPAIGN_COUNT,
        "distinct_seeds": True,
        "revealed_after_prediction_lock": True,
        "verified": True,
    }


def _prereg_contracts(
    preregistration: Mapping[str, Any],
    validated_lock: ValidatedPredictionLock2D,
) -> _PreregContracts:
    evaluation = _mapping(
        preregistration.get("evaluation_contract"),
        "preregistration.evaluation_contract",
    )
    _exact_keys(
        evaluation,
        frozenset({"schema", "schema_version", "repository_commit", "source_sha256"}),
        "preregistration.evaluation_contract",
    )
    if (
        evaluation["schema"] != EVALUATION_SCHEMA
        or evaluation["schema_version"] != EVALUATION_SCHEMA_VERSION
    ):
        raise Comparison2DValidationError("preregistered evaluator is not schema v3")
    _git_commit(evaluation["repository_commit"], "evaluation repository_commit")
    _sha256(evaluation["source_sha256"], "evaluation source_sha256")

    evidence = _mapping(
        preregistration.get("headline_evidence"),
        "preregistration.headline_evidence",
    )
    _exact_keys(
        evidence,
        frozenset(
            {
                "hidden_observation_generator",
                "public_mesh_convergence",
                "training_solver_commits",
            }
        ),
        "preregistration.headline_evidence",
    )
    generator = _mapping(
        evidence["hidden_observation_generator"],
        "headline_evidence.hidden_observation_generator",
    )
    generator_keys = frozenset(
        {
            "name",
            "repository_url",
            "repository_commit",
            "source_sha256",
            "source_size_bytes",
            "binary_sha256",
            "binary_size_bytes",
            "container_image_digest",
            "mesh_artifact_sha256",
            "mesh_artifact_size_bytes",
            "converter_sha256",
            "converter_size_bytes",
            "converter_repository_commit",
            "generation_runtime",
            "generation_runtime_manifest_sha256",
            "generation_runtime_manifest_size_bytes",
        }
    )
    _exact_keys(generator, generator_keys, "hidden_observation_generator")
    if generator["name"] != "ModEM":
        raise Comparison2DValidationError("headline generator must be ModEM")
    _string(generator["repository_url"], "generator repository_url")
    _git_commit(generator["repository_commit"], "generator repository_commit")
    _git_commit(
        generator["converter_repository_commit"],
        "generator converter_repository_commit",
    )
    for prefix in ("source", "binary", "converter"):
        _sha256(generator[f"{prefix}_sha256"], f"generator {prefix}_sha256")
        _integer(
            generator[f"{prefix}_size_bytes"],
            f"generator {prefix}_size_bytes",
            minimum=1,
        )
    _sha256(generator["mesh_artifact_sha256"], "generator mesh_artifact_sha256")
    _integer(
        generator["mesh_artifact_size_bytes"],
        "generator mesh_artifact_size_bytes",
        minimum=1,
    )
    generation_runtime = _mapping(
        generator["generation_runtime"], "generator generation_runtime"
    )
    _exact_keys(
        generation_runtime,
        frozenset(_HIDDEN_GENERATION_RUNTIME),
        "generator generation_runtime",
    )
    if dict(generation_runtime) != _HIDDEN_GENERATION_RUNTIME:
        raise Comparison2DValidationError(
            "hidden generation runtime versions differ from the frozen replay runtime"
        )
    _sha256(
        generator["generation_runtime_manifest_sha256"],
        "generator generation_runtime_manifest_sha256",
    )
    _integer(
        generator["generation_runtime_manifest_size_bytes"],
        "generator generation_runtime_manifest_size_bytes",
        minimum=1,
    )
    if not isinstance(
        generator["container_image_digest"], str
    ) or not _OCI_DIGEST_RE.fullmatch(generator["container_image_digest"]):
        raise Comparison2DValidationError(
            "generator container image digest is not pinned"
        )

    convergence = _mapping(
        evidence["public_mesh_convergence"],
        "headline_evidence.public_mesh_convergence",
    )
    _exact_keys(
        convergence,
        frozenset(
            {
                "criterion_id",
                "report_sha256",
                "report_size_bytes",
                "residuals_sha256",
                "residuals_size_bytes",
                "refined_mesh_sha256",
                "refined_mesh_size_bytes",
                "thresholds",
                "analytic_1d_contract",
            }
        ),
        "public_mesh_convergence",
    )
    _identifier(convergence["criterion_id"], "convergence criterion_id")
    _sha256(convergence["report_sha256"], "convergence report_sha256")
    _integer(convergence["report_size_bytes"], "convergence report_size_bytes", minimum=1)
    _sha256(convergence["residuals_sha256"], "convergence residuals_sha256")
    _integer(
        convergence["residuals_size_bytes"],
        "convergence residuals_size_bytes",
        minimum=1,
    )
    _sha256(convergence["refined_mesh_sha256"], "refined mesh_sha256")
    _integer(
        convergence["refined_mesh_size_bytes"],
        "refined mesh_size_bytes",
        minimum=1,
    )
    if convergence["thresholds"] != _CONVERGENCE_THRESHOLDS:
        raise Comparison2DValidationError("mesh convergence thresholds are not exact")
    _analytic_1d_contract(convergence["analytic_1d_contract"])

    commits = _mapping(
        evidence["training_solver_commits"],
        "headline_evidence.training_solver_commits",
    )
    _exact_keys(commits, frozenset(METHOD_IDS), "training_solver_commits")
    source_by_method: dict[str, tuple[str, str, str]] = {}
    for run in validated_lock.runs:
        identity = (run.source_commit, run.source_sha256, run.adapter_source_sha256)
        previous = source_by_method.setdefault(run.method_id, identity)
        if previous != identity:
            raise Comparison2DValidationError("lock method source identity is unstable")
    for method_id in METHOD_IDS:
        commit = _git_commit(commits[method_id], f"training_solver_commits.{method_id}")
        if commit != source_by_method[method_id][0]:
            raise Comparison2DValidationError(
                f"training solver commit for {method_id} differs from lock"
            )
    generator_identity = (
        generator["repository_commit"],
        generator["source_sha256"],
    )
    for method_id, (commit, source_sha, adapter_sha) in source_by_method.items():
        if generator_identity[0] == commit or generator_identity[1] in {
            source_sha,
            adapter_sha,
        }:
            raise Comparison2DValidationError(
                f"ModEM generator is not distinct from {method_id} training source"
            )
    comparison_implementation = _comparison_implementation_contract(
        preregistration.get("comparison_contract")
    )
    dependency_by_path = {
        record["path"]: record for record in comparison_implementation["dependencies"]
    }
    evaluator_dependency = dependency_by_path["src/pimsr_benchmarks/evaluation2d.py"]
    converter_dependency = dependency_by_path["src/pimsr_benchmarks/modem2d_forward.py"]
    if evaluator_dependency["sha256"] != evaluation["source_sha256"]:
        raise Comparison2DValidationError(
            "preregistered evaluator differs from the protected recomputation dependency"
        )
    if (
        converter_dependency["sha256"] != generator["converter_sha256"]
        or converter_dependency["size_bytes"] != generator["converter_size_bytes"]
    ):
        raise Comparison2DValidationError(
            "preregistered ModEM converter differs from the protected dependency"
        )
    return _PreregContracts(
        evaluator=evaluation,
        headline_evidence=evidence,
        comparison_implementation=comparison_implementation,
        public_lineage=_public_lineage_contract(
            preregistration.get("public_dataset_lineage")
        ),
        family_partition=_family_partition_contract(
            preregistration.get("family_partition")
        ),
        hidden_seed_commitment=_hidden_seed_commitment(preregistration),
    )


def _git_result(
    repository: Path,
    arguments: Sequence[str],
    *,
    check: bool,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=check,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
        raise Comparison2DValidationError(
            f"cannot validate pinned comparison repository state: {exc}"
        ) from exc


def _git_bytes(
    repository: Path,
    arguments: Sequence[str],
    *,
    check: bool,
    input_payload: bytes | None = None,
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
        raise Comparison2DValidationError(
            "cannot validate pinned comparison Git objects: "
            + (str(detail).strip() or str(exc))
        ) from exc


def _normal_path(path: Path) -> str:
    return os.path.normcase(os.path.normpath(str(path)))


def _direct_absolute_path(path: str | Path, *, role: str) -> Path:
    requested = Path(os.path.abspath(os.fspath(path)))
    try:
        resolved = requested.resolve(strict=True)
    except OSError as exc:
        raise Comparison2DValidationError(f"cannot resolve {role}: {requested}") from exc
    if _normal_path(requested) != _normal_path(resolved):
        raise Comparison2DValidationError(
            f"{role} must not traverse a symbolic link or redirected parent"
        )
    return requested


def _directory_identities(paths: Sequence[Path]) -> tuple[tuple[int, int], ...]:
    result: list[tuple[int, int]] = []
    for path in paths:
        try:
            info = os.lstat(path)
        except OSError as exc:
            raise Comparison2DValidationError(
                f"cannot inspect comparison source parent: {path}"
            ) from exc
        if not stat.S_ISDIR(info.st_mode) or path.is_symlink():
            raise Comparison2DValidationError(
                f"comparison source parent must be a direct directory: {path}"
            )
        result.append((int(info.st_dev), int(info.st_ino)))
    return tuple(result)


def _git_blob_at(repository: Path, commit: str, relative_path: str) -> str:
    result = _git_bytes(
        repository,
        ["ls-tree", "-z", commit, "--", relative_path],
        check=True,
    ).stdout
    rows = [row for row in result.split(b"\0") if row]
    try:
        metadata, encoded_path = rows[0].split(b"\t", 1)
        mode, object_type, object_id = metadata.split(b" ")
        recorded_path = encoded_path.decode("utf-8", errors="strict")
        object_id_text = object_id.decode("ascii", errors="strict")
    except (IndexError, ValueError, UnicodeError) as exc:
        raise Comparison2DValidationError(
            f"comparison protected path {relative_path!r} is not a unique Git blob"
        ) from exc
    if (
        len(rows) != 1
        or mode not in {b"100644", b"100755"}
        or object_type != b"blob"
        or recorded_path != relative_path
        or not re.fullmatch(r"[0-9a-f]{40,64}", object_id_text)
    ):
        raise Comparison2DValidationError(
            f"comparison protected path {relative_path!r} is not a unique Git blob"
        )
    return object_id_text


def _validate_comparison_implementation(
    contract: Mapping[str, Any],
) -> Mapping[str, Any]:
    source = _direct_absolute_path(__file__, role="comparison source")
    repository = source.parents[2]
    expected_records = (contract["source"], *contract["dependencies"])
    protected = list(contract["protected_paths"])
    parent_paths: list[Path] = [repository]
    for record in expected_records:
        candidate = repository / record["path"]
        current = candidate.parent
        while current != repository:
            parent_paths.append(current)
            current = current.parent
    unique_parents = tuple(dict.fromkeys(parent_paths))
    parent_identities = _directory_identities(unique_parents)
    seen: set[tuple[int, int]] = set()
    snapshots: list[ArtifactSnapshot] = []
    for index, record in enumerate(expected_records):
        role = "comparison source" if index == 0 else f"comparison dependency {index}"
        snapshot = _snapshot_unique(
            repository / record["path"],
            expected_sha256=record["sha256"],
            expected_size_bytes=record["size_bytes"],
            role=role,
            seen_identities=seen,
        )
        snapshots.append(snapshot)
    top = _direct_absolute_path(
        _git_result(
            repository, ["rev-parse", "--show-toplevel"], check=True
        ).stdout.strip(),
        role="comparison Git repository",
    )
    if _normal_path(top) != _normal_path(repository):
        raise Comparison2DValidationError(
            "comparison sources are not in the expected Git repository root"
        )
    head = _git_result(
        repository, ["rev-parse", "HEAD^{commit}"], check=True
    ).stdout.strip()
    _git_commit(head, "comparison current HEAD")
    pinned = contract["repository_commit"]
    ancestor = _git_result(
        repository, ["merge-base", "--is-ancestor", pinned, head], check=False
    )
    if ancestor.returncode != 0:
        raise Comparison2DValidationError(
            "comparison HEAD is not the pinned commit or an allowed descendant"
        )
    pinned_blobs: dict[str, str] = {}
    for path, snapshot in zip(protected, snapshots, strict=True):
        pinned_blob = _git_blob_at(repository, pinned, path)
        pinned_blobs[path] = pinned_blob
        if _git_blob_at(repository, head, path) != pinned_blob:
            raise Comparison2DValidationError(
                f"comparison HEAD changes protected path {path!r}"
            )
        filtered = _git_bytes(
            repository,
            ["hash-object", "--stdin", "--path", path],
            check=True,
            input_payload=snapshot.payload,
        ).stdout
        try:
            filtered_id = filtered.decode("ascii", errors="strict").strip()
        except UnicodeError as exc:
            raise Comparison2DValidationError(
                "comparison clean-filter object id is not ASCII"
            ) from exc
        if filtered_id != pinned_blob:
            raise Comparison2DValidationError(
                f"captured comparison protected path {path!r} does not match "
                "the pinned commit blob after Git clean filtering"
            )
    committed_diff = _git_result(
        repository,
        ["diff", "--quiet", f"{pinned}..{head}", "--", *protected],
        check=False,
    )
    if committed_diff.returncode != 0:
        raise Comparison2DValidationError(
            "an allowed descendant changes a comparison protected path"
        )
    status_arguments = [
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *protected,
    ]
    worktree_status = _git_result(repository, status_arguments, check=True).stdout
    if worktree_status:
        raise Comparison2DValidationError("comparison protected paths are not clean")
    if _directory_identities(unique_parents) != parent_identities:
        raise Comparison2DValidationError(
            "comparison source parent changed while provenance was captured"
        )
    for record, snapshot in zip(expected_records, snapshots, strict=True):
        path = _direct_absolute_path(
            repository / record["path"], role=f"comparison source {record['path']}"
        )
        final = _snapshot_unique(
            path,
            expected_sha256=snapshot.sha256,
            expected_size_bytes=snapshot.size_bytes,
            role=f"comparison source {record['path']} final recheck",
            seen_identities=set(),
        )
        if final.identity != snapshot.identity or final.payload != snapshot.payload:
            raise Comparison2DValidationError(
                f"comparison protected path {record['path']!r} changed during validation"
            )
    if (
        _git_result(repository, ["rev-parse", "HEAD^{commit}"], check=True).stdout.strip()
        != head
        or _git_result(repository, status_arguments, check=True).stdout != worktree_status
        or _directory_identities(unique_parents) != parent_identities
    ):
        raise Comparison2DValidationError(
            "comparison Git state changed while provenance was captured"
        )
    return {
        "schema": _COMPARISON_IMPLEMENTATION_SCHEMA,
        "schema_version": _COMPARISON_IMPLEMENTATION_SCHEMA_VERSION,
        "pinned_repository_commit": pinned,
        "executed_repository_commit": head,
        "allowed_descendant": True,
        "protected_paths_clean": True,
        "sources": [
            {
                "path": record["path"],
                "sha256": snapshot.sha256,
                "size_bytes": snapshot.size_bytes,
                "git_blob": pinned_blobs[record["path"]],
                "matches_pinned_blob_after_git_clean_filter": True,
            }
            for record, snapshot in zip(expected_records, snapshots, strict=True)
        ],
    }


def _validate_lineage_identity(
    value: Mapping[str, Any],
    *,
    split: str,
    expected_dataset: Mapping[str, Any],
) -> Mapping[str, Any]:
    _exact_keys(
        value,
        frozenset(
            {
                "schema",
                "schema_version",
                "evidence_scope",
                "split",
                "source_derived_generation_semantics",
                "inputs",
                "repositories",
                "verification",
            }
        ),
        f"{split} public lineage",
    )
    if (
        value["schema"] != _PUBLIC_LINEAGE_SCHEMA
        or value["schema_version"] != _PUBLIC_LINEAGE_SCHEMA_VERSION
        or value["evidence_scope"] != _PUBLIC_LINEAGE_EVIDENCE_SCOPE
        or value["split"] != split
    ):
        raise Comparison2DValidationError(f"{split} public lineage identity is wrong")
    if value["source_derived_generation_semantics"] != (
        _SOURCE_DERIVED_GENERATION_SEMANTICS
    ):
        raise Comparison2DValidationError(
            f"{split} public lineage source-derived generation semantics are incomplete"
        )
    inputs = _mapping(value["inputs"], f"{split} lineage.inputs")
    _exact_keys(
        inputs,
        frozenset({"merged_dataset", "shard_pin_manifest", "shard_directory", "shards"}),
        f"{split} lineage.inputs",
    )
    merged = _mapping(inputs["merged_dataset"], f"{split} lineage merged_dataset")
    _exact_keys(merged, _ARTIFACT_REFERENCE_KEYS, f"{split} lineage merged_dataset")
    if (
        _sha256(merged["sha256"], f"{split} merged_dataset.sha256")
        != expected_dataset["sha256"]
        or _integer(merged["size_bytes"], f"{split} merged_dataset.size_bytes", minimum=1)
        != expected_dataset["size_bytes"]
    ):
        raise Comparison2DValidationError(
            f"{split} lineage merged dataset differs from preregistration"
        )
    merged_path = _string(merged["path"], f"{split} merged_dataset.path")
    _require_public_path(merged_path, f"{split} merged_dataset.path")
    if Path(merged_path).name != expected_dataset["filename"]:
        raise Comparison2DValidationError(
            f"{split} lineage merged filename differs from preregistration"
        )
    pin_manifest = _mapping(
        inputs["shard_pin_manifest"], f"{split} lineage shard_pin_manifest"
    )
    _exact_keys(
        pin_manifest,
        _ARTIFACT_REFERENCE_KEYS,
        f"{split} lineage shard_pin_manifest",
    )
    _require_public_path(pin_manifest["path"], f"{split} lineage shard_pin_manifest.path")
    _sha256(pin_manifest["sha256"], f"{split} lineage shard_pin_manifest.sha256")
    _integer(
        pin_manifest["size_bytes"],
        f"{split} lineage shard_pin_manifest.size_bytes",
        minimum=1,
    )
    shard_directory = _mapping(
        inputs["shard_directory"], f"{split} lineage shard_directory"
    )
    _exact_keys(
        shard_directory,
        frozenset({"path", "entry_count"}),
        f"{split} lineage shard_directory",
    )
    _require_public_path(shard_directory["path"], f"{split} lineage shard_directory.path")
    if (
        _integer(
            shard_directory["entry_count"],
            f"{split} lineage shard_directory.entry_count",
            minimum=1,
        )
        != expected_dataset["source_shard_count"] * 2
    ):
        raise Comparison2DValidationError(
            f"{split} lineage shard directory entry count is not exact"
        )
    raw_shards = _sequence(inputs["shards"], f"{split} lineage.shards")
    if len(raw_shards) != expected_dataset["source_shard_count"]:
        raise Comparison2DValidationError(
            f"{split} lineage does not cover every source shard"
        )
    next_start = expected_dataset["start_index"]
    shard_identities: list[dict[str, Any]] = []
    for index, raw_shard in enumerate(raw_shards):
        path = f"{split} lineage.shards[{index}]"
        shard = _mapping(raw_shard, path)
        _exact_keys(
            shard,
            frozenset(
                {
                    "ordinal",
                    "sample_start",
                    "sample_end",
                    "sample_count",
                    "hdf5",
                    "log",
                }
            ),
            path,
        )
        count = _integer(shard["sample_count"], f"{path}.sample_count", minimum=1)
        start = _integer(shard["sample_start"], f"{path}.sample_start")
        end = _integer(shard["sample_end"], f"{path}.sample_end")
        if (
            shard["ordinal"] != index
            or start != next_start
            or end != start + count - 1
            or count != expected_dataset["shard_rows"]
        ):
            raise Comparison2DValidationError(
                f"{path} range/order differs from the public generation schedule"
            )
        next_start = end + 1
        record: dict[str, Any] = {
            "ordinal": index,
            "sample_start": start,
            "sample_end": end,
        }
        for role in ("hdf5", "log"):
            artifact_path = f"{path}.{role}"
            artifact = _mapping(shard[role], artifact_path)
            _exact_keys(
                artifact,
                frozenset({"filename", "sha256", "size_bytes"}),
                artifact_path,
            )
            filename = _string(artifact["filename"], f"{artifact_path}.filename")
            _require_public_path(filename, f"{artifact_path}.filename")
            if Path(filename).name != filename:
                raise Comparison2DValidationError(
                    f"{artifact_path}.filename must not contain directories"
                )
            expected_filename = f"shard-{start:06d}-{end:06d}.h5"
            if role == "log":
                expected_filename += ".log"
            if filename != expected_filename:
                raise Comparison2DValidationError(
                    f"{artifact_path}.filename differs from its exact sample range"
                )
            record[f"{role}_filename"] = filename
            record[f"{role}_sha256"] = _sha256(
                artifact["sha256"], f"{artifact_path}.sha256"
            )
            record[f"{role}_size_bytes"] = _integer(
                artifact["size_bytes"], f"{artifact_path}.size_bytes", minimum=1
            )
        shard_identities.append(record)
    if next_start != expected_dataset["sample_end_index"] + 1:
        raise Comparison2DValidationError(
            f"{split} lineage shard ranges do not cover the merged dataset"
        )
    repositories = _mapping(value["repositories"], f"{split} lineage.repositories")
    _exact_keys(
        repositories,
        frozenset({"pimsr_forward", "pimsr_geogen"}),
        f"{split} lineage.repositories",
    )
    identities: dict[str, dict[str, Any]] = {}
    mandatory_sources = {
        "pimsr_forward": frozenset(
            {
                "src/pimsr_forward/dataset2d.py",
                "src/pimsr_forward/mt2d.py",
                "src/pimsr_forward/sensors.py",
            }
        ),
        "pimsr_geogen": frozenset(
            {
                "src/pimsr_geogen/generator.py",
                "src/pimsr_geogen/model.py",
                "src/pimsr_geogen/rock_physics.py",
                "src/pimsr_geogen/section2d.py",
            }
        ),
    }
    for name, required_sources in mandatory_sources.items():
        path = f"{split} lineage.repositories.{name}"
        repository = _mapping(repositories[name], path)
        _exact_keys(
            repository,
            frozenset(
                {"path", "commit", "clean_worktree", "origin_remote", "source_files"}
            ),
            path,
        )
        if repository["clean_worktree"] is not True:
            raise Comparison2DValidationError(
                f"{path} was not captured from a clean tree"
            )
        commit = _git_commit(repository["commit"], f"{path}.commit")
        origin = _string(repository["origin_remote"], f"{path}.origin_remote")
        _string(repository["path"], f"{path}.path")
        sources = _mapping(repository["source_files"], f"{path}.source_files")
        if set(sources) != required_sources:
            raise Comparison2DValidationError(f"{path}.source_files are not exact")
        source_hashes: dict[str, str] = {}
        for source_path in sorted(required_sources):
            record = _mapping(sources[source_path], f"{path}.source_files.{source_path}")
            _exact_keys(
                record,
                frozenset(
                    {
                        "sha256",
                        "size_bytes",
                        "matches_commit_blob_after_git_clean_filter",
                    }
                ),
                f"{path}.source_files.{source_path}",
            )
            if record["matches_commit_blob_after_git_clean_filter"] is not True:
                raise Comparison2DValidationError(
                    f"{path}.source_files.{source_path} does not match its pinned commit blob"
                )
            source_hashes[source_path] = _sha256(
                record["sha256"], f"{path}.source_files.{source_path}.sha256"
            )
            _integer(
                record["size_bytes"],
                f"{path}.source_files.{source_path}.size_bytes",
                minimum=1,
            )
        identities[name] = {
            "commit": commit,
            "origin_remote": origin,
            "source_hashes": source_hashes,
        }
    if identities["pimsr_forward"]["commit"] != expected_dataset["generator_commit"]:
        raise Comparison2DValidationError(
            f"{split} lineage pimsr-forward commit differs from preregistration"
        )
    verification = _mapping(value["verification"], f"{split} lineage.verification")
    verification_keys = frozenset(
        {
            "arrays",
            "chunk_rows",
            "concatenation",
            "forward_regeneration_performed",
            "generation_complete",
            "generation_start_index",
            "generation_time_execution_proven",
            "generator_seed",
            "root_attributes",
            "sample_count",
            "sample_end_index",
            "schema_contract",
            "source_shard_count",
        }
    )
    _exact_keys(verification, verification_keys, f"{split} lineage.verification")
    expected_verification = {
        "concatenation": "exact_ordered_array_and_metadata_equality",
        "forward_regeneration_performed": False,
        "generation_complete": True,
        "generation_start_index": expected_dataset["start_index"],
        "generation_time_execution_proven": False,
        "generator_seed": expected_dataset["generator_seed"],
        "sample_count": expected_dataset["sample_count"],
        "sample_end_index": expected_dataset["sample_end_index"],
        "schema_contract": "pimsr-mt-2d/v2",
        "source_shard_count": expected_dataset["source_shard_count"],
    }
    if any(
        verification[name] != expected for name, expected in expected_verification.items()
    ):
        raise Comparison2DValidationError(
            f"{split} lineage verification differs from the frozen public schedule"
        )
    chunk_rows = _integer(
        verification["chunk_rows"],
        f"{split} lineage.verification.chunk_rows",
        minimum=1,
    )
    if chunk_rows != 100:
        raise Comparison2DValidationError(
            f"{split} lineage verification chunk size is not the frozen 100"
        )
    root_attributes = _mapping(
        verification["root_attributes"], f"{split} lineage.root_attributes"
    )
    expected_root_keys = frozenset(
        {
            *_LINEAGE_ROOT_STRING_ATTRIBUTES,
            "schema_version",
            "generator_seed",
            "generation_start_index",
            "expected_row_count",
            "source_shard_count",
            "generation_complete",
            "mode_order",
            "impedance_components",
            "scenario_order",
            "sensor_parameters_json",
            "software_versions_json",
        }
    )
    _exact_keys(root_attributes, expected_root_keys, f"{split} lineage.root_attributes")
    expected_root_values = {
        **_LINEAGE_ROOT_STRING_ATTRIBUTES,
        "schema_version": 2,
        "generator_seed": expected_dataset["generator_seed"],
        "generation_start_index": expected_dataset["start_index"],
        "expected_row_count": expected_dataset["sample_count"],
        "source_shard_count": expected_dataset["source_shard_count"],
        "generation_complete": 1,
        "mode_order": ["te", "tm"],
        "impedance_components": ["Zyx", "Zxy"],
        "scenario_order": list(FAMILY_IDS),
    }
    if any(
        root_attributes[name] != expected
        for name, expected in expected_root_values.items()
    ):
        raise Comparison2DValidationError(
            f"{split} lineage root generation contract is not exact"
        )
    sensor_parameters = _canonical_json_attribute(
        root_attributes["sensor_parameters_json"],
        f"{split} lineage.root_attributes.sensor_parameters_json",
    )
    if sensor_parameters != _LINEAGE_SENSOR_PARAMETERS:
        raise Comparison2DValidationError(
            f"{split} lineage sensor parameters are not frozen"
        )
    software_versions = _canonical_json_attribute(
        root_attributes["software_versions_json"],
        f"{split} lineage.root_attributes.software_versions_json",
    )
    expected_version_keys = {
        "discretize",
        "h5py",
        "numpy",
        "pimsr_forward",
        "pimsr_geogen",
        "simpeg",
    }
    if (
        not isinstance(software_versions, Mapping)
        or set(software_versions) != expected_version_keys
        or any(
            not isinstance(software_versions[name], str) or not software_versions[name]
            for name in expected_version_keys
        )
    ):
        raise Comparison2DValidationError(
            f"{split} lineage software version closure is invalid"
        )
    arrays = _mapping(verification["arrays"], f"{split} lineage.arrays")
    expected_array_names = _LINEAGE_ROW_ARRAYS | _LINEAGE_COORDINATE_ARRAYS
    _exact_keys(arrays, expected_array_names, f"{split} lineage.arrays")
    sample_count = expected_dataset["sample_count"]
    expected_shapes = {
        **{
            name: (sample_count, 8, 12)
            for name in _LINEAGE_ROW_ARRAYS
            if name.startswith(("obs_mt_", "clean_mt_"))
        },
        "target_log10_res": (sample_count, *CANONICAL_MODEL_SHAPE),
        "scenario": (sample_count,),
        "has_fault": (sample_count,),
        "sample_index": (sample_count,),
        "frequencies": (8,),
        "station_x": (12,),
        "x_grid": (CANONICAL_MODEL_SHAPE[1],),
        "depth_grid": (CANONICAL_MODEL_SHAPE[0],),
    }
    expected_dtypes = {
        **{
            name: "<f4"
            for name in _LINEAGE_ROW_ARRAYS
            if name.startswith(("obs_mt_", "clean_mt_"))
        },
        "target_log10_res": "<f4",
        "scenario": "<i4",
        "has_fault": "|u1",
        "sample_index": "<i8",
        **{name: "<f8" for name in _LINEAGE_COORDINATE_ARRAYS},
    }
    for name in sorted(expected_array_names):
        path = f"{split} lineage.arrays.{name}"
        record = _mapping(arrays[name], path)
        _exact_keys(
            record,
            frozenset(
                {"dtype", "shape", "logical_c_order_bytes_sha256", "shard_equality"}
            ),
            path,
        )
        dtype = _string(record["dtype"], f"{path}.dtype")
        shape_values = _sequence(record["shape"], f"{path}.shape")
        shape = tuple(
            _integer(item, f"{path}.shape[{axis}]", minimum=1)
            for axis, item in enumerate(shape_values)
        )
        if dtype != expected_dtypes[name] or shape != expected_shapes[name]:
            raise Comparison2DValidationError(
                f"{path} shape differs from the merged data"
            )
        _sha256(
            record["logical_c_order_bytes_sha256"],
            f"{path}.logical_c_order_bytes_sha256",
        )
        expected_equality = (
            "exact_ordered_concatenation"
            if name in _LINEAGE_ROW_ARRAYS
            else "exact_repetition_in_every_shard"
        )
        if record["shard_equality"] != expected_equality:
            raise Comparison2DValidationError(f"{path} shard equality is wrong")
    return {
        "merged_dataset_sha256": expected_dataset["sha256"],
        "merged_dataset_size_bytes": expected_dataset["size_bytes"],
        "merged_dataset_filename": expected_dataset["filename"],
        "repositories": identities,
        "shards": tuple(shard_identities),
        "shard_identity_sha256": _canonical_object_sha256({"shards": shard_identities}),
        "verification_sha256": _canonical_object_sha256(dict(verification)),
    }


def _load_public_lineages(
    references: Mapping[str, Mapping[str, Any]],
    *,
    preregistration: Mapping[str, Any],
    base: Path,
    seen_identities: set[tuple[int, int]],
) -> Mapping[str, Any]:
    datasets = _mapping(preregistration.get("datasets"), "preregistration.datasets")
    result: dict[str, Any] = {}
    for split in ("train", "validation"):
        dataset = _mapping(datasets.get(split), f"datasets.{split}")
        artifact = _mapping(dataset.get("artifact"), f"datasets.{split}.artifact")
        generator = _mapping(dataset.get("generator"), f"datasets.{split}.generator")
        raw_range = _sequence(
            dataset.get("sample_index_range_inclusive"),
            f"datasets.{split}.sample_index_range_inclusive",
        )
        if len(raw_range) != 2:
            raise Comparison2DValidationError(
                f"datasets.{split}.sample_index_range_inclusive must have two values"
            )
        range_start = _integer(
            raw_range[0], f"datasets.{split}.sample_index_range_inclusive[0]"
        )
        range_end = _integer(
            raw_range[1], f"datasets.{split}.sample_index_range_inclusive[1]"
        )
        start_index = _integer(
            generator.get("start_index"), f"datasets.{split}.generator.start_index"
        )
        sample_count = _integer(dataset.get("rows"), f"datasets.{split}.rows", minimum=1)
        frozen_seed, frozen_count, frozen_shards = (
            (20260820, 10_000, 100) if split == "train" else (20260821, 1_000, 10)
        )
        generator_seed = _integer(
            generator.get("seed"), f"datasets.{split}.generator.seed"
        )
        if range_start != start_index or range_end != start_index + sample_count - 1:
            raise Comparison2DValidationError(
                f"datasets.{split} public sample range is inconsistent"
            )
        if (
            generator_seed != frozen_seed
            or start_index != 0
            or sample_count != frozen_count
        ):
            raise Comparison2DValidationError(
                f"datasets.{split} public generation schedule is not frozen"
            )
        expected_dataset = {
            "filename": _string(
                artifact.get("filename"), f"datasets.{split}.artifact.filename"
            ),
            "sha256": _sha256(
                artifact.get("sha256"), f"datasets.{split}.artifact.sha256"
            ),
            "size_bytes": _integer(
                artifact.get("size_bytes"),
                f"datasets.{split}.artifact.size_bytes",
                minimum=1,
            ),
            "generator_commit": _git_commit(
                generator.get("repository_commit"),
                f"datasets.{split}.generator.repository_commit",
            ),
            "generator_seed": generator_seed,
            "start_index": start_index,
            "sample_count": sample_count,
            "sample_end_index": range_end,
            "source_shard_count": frozen_shards,
            "shard_rows": 100,
        }
        reference = references[split]
        snapshot = _snapshot_unique(
            base / reference["path"],
            expected_sha256=reference["sha256"],
            expected_size_bytes=reference["size_bytes"],
            role=f"{split} public dataset lineage manifest",
            seen_identities=seen_identities,
        )
        value = _strict_json(snapshot, f"{split} public dataset lineage manifest")
        result[split] = {
            **_validate_lineage_identity(
                value, split=split, expected_dataset=expected_dataset
            ),
            "manifest_sha256": snapshot.sha256,
            "manifest_size_bytes": snapshot.size_bytes,
        }
    for repository_name in ("pimsr_forward", "pimsr_geogen"):
        source_identities = {
            (
                result[split]["repositories"][repository_name]["commit"],
                tuple(
                    sorted(
                        result[split]["repositories"][repository_name][
                            "source_hashes"
                        ].items()
                    )
                ),
            )
            for split in ("train", "validation")
        }
        if len(source_identities) != 1:
            raise Comparison2DValidationError(
                "train and validation lineage use different "
                f"{repository_name} source identities"
            )
    return result


def _evidence_reference_specs(
    evidence: Mapping[str, Any],
) -> dict[str, tuple[str, int]]:
    generator = evidence["hidden_observation_generator"]
    convergence = evidence["public_mesh_convergence"]
    return {
        "generator_source": (
            generator["source_sha256"],
            generator["source_size_bytes"],
        ),
        "generator_binary": (
            generator["binary_sha256"],
            generator["binary_size_bytes"],
        ),
        "production_mesh_artifact": (
            generator["mesh_artifact_sha256"],
            generator["mesh_artifact_size_bytes"],
        ),
        "converter_source": (
            generator["converter_sha256"],
            generator["converter_size_bytes"],
        ),
        "generation_runtime_manifest": (
            generator["generation_runtime_manifest_sha256"],
            generator["generation_runtime_manifest_size_bytes"],
        ),
        "mesh_convergence_report": (
            convergence["report_sha256"],
            convergence["report_size_bytes"],
        ),
        "mesh_convergence_residuals": (
            convergence["residuals_sha256"],
            convergence["residuals_size_bytes"],
        ),
        "refined_mesh_artifact": (
            convergence["refined_mesh_sha256"],
            convergence["refined_mesh_size_bytes"],
        ),
    }


def _validate_generation_runtime_manifest(
    value: Mapping[str, Any],
    *,
    public_lineage: Mapping[str, Any],
) -> Mapping[str, Any]:
    path = "hidden generation runtime manifest"
    _exact_keys(
        value,
        frozenset(
            {
                "schema",
                "schema_version",
                "python",
                "distributions",
                "source_closure",
                "tree_manifest_sha256",
            }
        ),
        path,
    )
    if (
        value["schema"] != _HIDDEN_RUNTIME_MANIFEST_SCHEMA
        or value["schema_version"] != _HIDDEN_RUNTIME_MANIFEST_SCHEMA_VERSION
    ):
        raise Comparison2DValidationError(
            "hidden generation runtime manifest identity is wrong"
        )
    python = _mapping(value["python"], f"{path}.python")
    _exact_keys(
        python,
        frozenset({"implementation", "version", "executable_sha256"}),
        f"{path}.python",
    )
    if (
        python["implementation"] != "CPython"
        or python["version"] != (_HIDDEN_GENERATION_RUNTIME["python_version"])
    ):
        raise Comparison2DValidationError(
            "hidden generation Python runtime differs from the frozen replay runtime"
        )
    executable_sha256 = _sha256(
        python["executable_sha256"], f"{path}.python.executable_sha256"
    )
    distributions = _mapping(value["distributions"], f"{path}.distributions")
    _exact_keys(
        distributions,
        frozenset(_HIDDEN_RUNTIME_DISTRIBUTIONS),
        f"{path}.distributions",
    )
    distribution_trees: dict[str, str] = {}
    for name, expected_version in _HIDDEN_RUNTIME_DISTRIBUTIONS.items():
        record = _mapping(distributions[name], f"{path}.distributions.{name}")
        _exact_keys(
            record,
            frozenset({"version", "installed_tree_sha256"}),
            f"{path}.distributions.{name}",
        )
        if record["version"] != expected_version:
            raise Comparison2DValidationError(
                f"hidden generation distribution {name} version is not frozen"
            )
        distribution_trees[name] = _sha256(
            record["installed_tree_sha256"],
            f"{path}.distributions.{name}.installed_tree_sha256",
        )
    source_closure = _mapping(value["source_closure"], f"{path}.source_closure")
    if dict(source_closure) != dict(_hidden_source_lineage_identity(public_lineage)):
        raise Comparison2DValidationError(
            "hidden runtime installed sources differ from public source closure"
        )
    tree_manifest_sha256 = _sha256(
        value["tree_manifest_sha256"], f"{path}.tree_manifest_sha256"
    )
    return {
        "schema": _HIDDEN_RUNTIME_MANIFEST_SCHEMA,
        "schema_version": _HIDDEN_RUNTIME_MANIFEST_SCHEMA_VERSION,
        "python_executable_sha256": executable_sha256,
        "distribution_tree_sha256": distribution_trees,
        "source_closure_sha256": _canonical_object_sha256(source_closure),
        "tree_manifest_sha256": tree_manifest_sha256,
    }


def _canonical_object_sha256(value: Mapping[str, Any]) -> str:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Comparison2DValidationError(
            f"cannot hash canonical evidence object: {exc}"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _file_identity_record(value: Any, path: str) -> tuple[str, int]:
    record = _mapping(value, path)
    _exact_keys(record, _ARTIFACT_REFERENCE_KEYS, path)
    _string(record["path"], f"{path}.path")
    return (
        _sha256(record["sha256"], f"{path}.sha256"),
        _integer(record["size_bytes"], f"{path}.size_bytes", minimum=1),
    )


def _require_public_path(value: Any, path: str) -> None:
    text = _string(value, path)
    tokens = {
        token
        for part in re.split(r"[/\\]+", text.casefold())
        for token in part.replace("_", "-").split("-")
    }
    forbidden = tokens & {"hidden", "secret", "blind"}
    if forbidden:
        raise Comparison2DValidationError(
            f"{path} is not a public-only evidence path: {sorted(forbidden)}"
        )


def _mesh_config_record(
    value: Any,
    *,
    path: str,
    reference_only: bool,
) -> Mapping[str, Any]:
    record = _mapping(value, path)
    extra = {"mesh_config_sha256"}
    if reference_only:
        extra.add("reference_only_not_automatically_production_eligible")
    _exact_keys(record, _MESH_RECORD_KEYS | frozenset(extra), path)
    if record["schema"] != "pimsr-modem2d-nested-mesh" or record["schema_version"] != 1:
        raise Comparison2DValidationError(f"{path} schema identity is wrong")
    _identifier(record["mesh_id"], f"{path}.mesh_id")
    _integer(record["version"], f"{path}.version", minimum=1)
    integer_names = (
        "base_core_count",
        "base_padding_count_each_side",
        "minimum_vertical_subdivisions",
        "horizontal_refinement_factor",
        "vertical_refinement_factor",
    )
    integers = {
        name: _integer(record[name], f"{path}.{name}", minimum=1)
        for name in integer_names
    }
    dimensions = {
        name: _finite(record[name], f"{path}.{name}")
        for name in (
            "base_core_width_m",
            "base_padding_growth",
            "maximum_base_dz_m",
            "deep_padding_growth",
            "maximum_deep_macro_dz_m",
            "minimum_depth_m",
        )
    }
    if any(value <= 0.0 for value in dimensions.values()):
        raise Comparison2DValidationError(f"{path} dimensions must be positive")
    if (
        dimensions["base_padding_growth"] <= 1.0
        or dimensions["deep_padding_growth"] <= 1.0
    ):
        raise Comparison2DValidationError(f"{path} mesh growth must exceed one")
    if dimensions["maximum_deep_macro_dz_m"] < dimensions["maximum_base_dz_m"]:
        raise Comparison2DValidationError(f"{path} deep dz cap is too small")
    if not math.isclose(
        dimensions["base_core_width_m"] * integers["base_core_count"],
        24_000.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise Comparison2DValidationError(f"{path} physical core is not exactly 24 km")
    expected_literals = {
        "canonical_depth_centres_sha256": _CANONICAL_DEPTH_CENTRES_SHA256,
        "canonical_x_centres_sha256": _CANONICAL_X_CENTRES_SHA256,
        "horizontal_partition": _HORIZONTAL_PARTITION,
        "vertical_partition": _VERTICAL_PARTITION,
        "mapping": _MESH_MAPPING,
    }
    if any(record[name] != expected for name, expected in expected_literals.items()):
        raise Comparison2DValidationError(f"{path} nested mesh contract is wrong")
    if (
        reference_only
        and record["reference_only_not_automatically_production_eligible"] is not True
    ):
        raise Comparison2DValidationError(f"{path} is not marked reference-only")
    canonical = {key: record[key] for key in _MESH_RECORD_KEYS}
    digest = _canonical_object_sha256(canonical)
    if record["mesh_config_sha256"] != digest:
        raise Comparison2DValidationError(f"{path} canonical SHA-256 is wrong")
    return record


def _finite_vector(value: Any, *, path: str, length: int, positive: bool) -> np.ndarray:
    raw = _sequence(value, path)
    if len(raw) != length:
        raise Comparison2DValidationError(f"{path} must contain exactly {length} values")
    result = np.asarray(
        [_finite(item, f"{path}[{index}]") for index, item in enumerate(raw)],
        dtype="<f8",
    )
    if (positive and np.any(result <= 0.0)) or np.any(np.diff(result) <= 0.0):
        raise Comparison2DValidationError(
            f"{path} must be positive and strictly increasing"
        )
    return result


def _analytic_1d_contract(value: Any) -> Mapping[str, Any]:
    contract = _mapping(value, "public_mesh_convergence.analytic_1d_contract")
    _exact_keys(
        contract,
        frozenset(
            {
                "schema",
                "schema_version",
                "time_convention",
                "response_shape",
                "frequencies_hz",
                "frequencies_sha256",
                "canonical_depth_centres_m",
                "canonical_depth_centres_sha256",
                "cases",
            }
        ),
        "analytic_1d_contract",
    )
    if (
        contract["schema"] != "pimsr-modem2d-analytic-1d-contract"
        or contract["schema_version"] != 1
        or contract["time_convention"] != "exp(+i omega t)"
        or contract["response_shape"] != [8, 12]
    ):
        raise Comparison2DValidationError("analytic 1-D contract identity is wrong")
    frequencies = _finite_vector(
        contract["frequencies_hz"],
        path="analytic_1d_contract.frequencies_hz",
        length=8,
        positive=True,
    )
    depth = _finite_vector(
        contract["canonical_depth_centres_m"],
        path="analytic_1d_contract.canonical_depth_centres_m",
        length=64,
        positive=True,
    )
    if hashlib.sha256(frequencies.tobytes()).hexdigest() != _sha256(
        contract["frequencies_sha256"], "analytic frequencies_sha256"
    ) or hashlib.sha256(depth.tobytes()).hexdigest() != _sha256(
        contract["canonical_depth_centres_sha256"],
        "analytic canonical_depth_centres_sha256",
    ):
        raise Comparison2DValidationError(
            "analytic 1-D axes do not match their byte pins"
        )
    cases = _sequence(contract["cases"], "analytic_1d_contract.cases")
    expected_profiles = (
        (
            "analytic-halfspace-100",
            ((None, 100.0),),
        ),
        (
            "analytic-layered-100-10-500",
            ((1_000.0, 100.0), (3_000.0, 10.0), (None, 500.0)),
        ),
    )
    normalized_cases: dict[str, tuple[tuple[float | None, float], ...]] = {}
    if len(cases) != len(expected_profiles):
        raise Comparison2DValidationError("analytic 1-D cases are not exact")
    for index, (raw_case, (expected_id, expected_profile)) in enumerate(
        zip(cases, expected_profiles, strict=True)
    ):
        path = f"analytic_1d_contract.cases[{index}]"
        case = _mapping(raw_case, path)
        _exact_keys(case, frozenset({"truth_id", "depth_profile"}), path)
        if case["truth_id"] != expected_id:
            raise Comparison2DValidationError("analytic truth ids/order are not exact")
        raw_profile = _sequence(case["depth_profile"], f"{path}.depth_profile")
        observed_profile: list[tuple[float | None, float]] = []
        for layer_index, raw_layer in enumerate(raw_profile):
            layer_path = f"{path}.depth_profile[{layer_index}]"
            layer = _mapping(raw_layer, layer_path)
            _exact_keys(
                layer,
                frozenset({"maximum_depth_m_exclusive", "resistivity_ohm_m"}),
                layer_path,
            )
            maximum = layer["maximum_depth_m_exclusive"]
            if maximum is not None:
                maximum = _finite(maximum, f"{layer_path}.maximum_depth_m_exclusive")
            resistivity = _finite(
                layer["resistivity_ohm_m"],
                f"{layer_path}.resistivity_ohm_m",
            )
            observed_profile.append((maximum, resistivity))
        if tuple(observed_profile) != expected_profile:
            raise Comparison2DValidationError("analytic depth profile is not frozen")
        normalized_cases[expected_id] = tuple(observed_profile)
    return {
        "frequencies_hz": frequencies,
        "canonical_depth_centres_m": depth,
        "profiles": normalized_cases,
    }


def _next_header_value(lines: Sequence[str], start: int, *, role: str) -> tuple[str, int]:
    index = start
    while index < len(lines):
        stripped = lines[index].strip()
        index += 1
        if not stripped or stripped.startswith("#"):
            continue
        if not stripped.startswith(">"):
            raise Comparison2DValidationError(f"missing {role} header")
        return stripped[1:].strip(), index
    raise Comparison2DValidationError(f"missing {role} header")


def _parse_modem_forward_snapshot(
    snapshot: ArtifactSnapshot, *, role: str
) -> _RawModEMResponse:
    try:
        lines = snapshot.payload.decode("ascii", errors="strict").splitlines()
    except UnicodeError as exc:
        raise Comparison2DValidationError(f"{role} is not strict ASCII") from exc
    rows: dict[str, list[tuple[float, int, float, complex]]] = {
        "TE": [],
        "TM": [],
    }
    blocks: set[str] = set()
    current_mode: str | None = None
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        index += 1
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith(">"):
            match = re.fullmatch(r"(TE|TM)_Impedance", stripped[1:].strip())
            if match is None:
                raise Comparison2DValidationError(
                    f"{role} has an unexpected response block"
                )
            current_mode = match.group(1)
            if current_mode in blocks:
                raise Comparison2DValidationError(
                    f"{role} duplicates {current_mode} block"
                )
            blocks.add(current_mode)
            sign, index = _next_header_value(lines, index, role=f"{role} time convention")
            units, index = _next_header_value(lines, index, role=f"{role} units")
            orientation, index = _next_header_value(
                lines, index, role=f"{role} orientation"
            )
            origin, index = _next_header_value(lines, index, role=f"{role} origin")
            counts, index = _next_header_value(lines, index, role=f"{role} counts")
            if (
                sign.replace(" ", "") not in {"exp(+i\\omegat)", "exp(+iomegat)"}
                or units.replace(" ", "") != "[V/m]/[T]"
                or orientation.split() != ["0.00"]
                or origin.split() != ["0.000", "0.000"]
                or counts.split() != ["8", "12"]
            ):
                raise Comparison2DValidationError(
                    f"{role} ModEM header contract is wrong"
                )
            continue
        if current_mode is None:
            raise Comparison2DValidationError(f"{role} has a data row outside a block")
        fields = stripped.split()
        if len(fields) != 11 or fields[7] != current_mode:
            raise Comparison2DValidationError(f"{role} has a malformed ModEM row")
        site = re.fullmatch(r"S(\d{2})", fields[1])
        if site is None:
            raise Comparison2DValidationError(f"{role} has a noncanonical station id")
        station_index = int(site.group(1)) - 1
        if not 0 <= station_index < 12:
            raise Comparison2DValidationError(f"{role} station index is outside 1..12")
        try:
            period = float(fields[0])
            latitude, longitude, x_coord, y_coord, z_coord = map(float, fields[2:7])
            real, imaginary, error = map(float, fields[8:11])
        except ValueError as exc:
            raise Comparison2DValidationError(
                f"{role} has a nonnumeric ModEM row"
            ) from exc
        numeric = (
            period,
            latitude,
            longitude,
            x_coord,
            y_coord,
            z_coord,
            real,
            imaginary,
            error,
        )
        if (
            not all(math.isfinite(item) for item in numeric)
            or period <= 0.0
            or error <= 0.0
            or abs(latitude) > 0.002
            or abs(longitude) > 0.002
            or abs(x_coord) > 0.002
            or abs(z_coord) > 0.002
            or complex(real, imaginary) == 0.0
        ):
            raise Comparison2DValidationError(
                f"{role} ModEM row violates the finite geometry contract"
            )
        rows[current_mode].append(
            (period, station_index, y_coord, complex(real, imaginary))
        )
    if blocks != {"TE", "TM"} or any(len(rows[mode]) != 96 for mode in ("TE", "TM")):
        raise Comparison2DValidationError(
            f"{role} must contain exactly 96 TE and 96 TM rows"
        )
    periods = sorted({item[0] for item in rows["TE"]}, reverse=True)
    if len(periods) != 8 or {item[0] for item in rows["TM"]} != set(periods):
        raise Comparison2DValidationError(
            f"{role} periods are not an exact shared 8-value grid"
        )
    frequencies = np.asarray([1.0 / period for period in periods], dtype=np.float64)
    if np.any(np.diff(frequencies) <= 0.0):
        raise Comparison2DValidationError(f"{role} frequency grid is not increasing")
    period_index = {period: position for position, period in enumerate(periods)}
    station_y = np.full(12, np.nan, dtype=np.float64)
    z_by_mode = {
        mode: np.full((8, 12), np.nan + 1j * np.nan, dtype=np.complex128)
        for mode in ("TE", "TM")
    }
    seen: dict[str, set[tuple[int, int]]] = {"TE": set(), "TM": set()}
    for mode in ("TE", "TM"):
        for period, station_index, y_coord, impedance in rows[mode]:
            key = (period_index[period], station_index)
            if key in seen[mode]:
                raise Comparison2DValidationError(
                    f"{role} duplicates a period/station row"
                )
            seen[mode].add(key)
            if np.isnan(station_y[station_index]):
                station_y[station_index] = y_coord
            elif not math.isclose(
                float(station_y[station_index]), y_coord, rel_tol=0.0, abs_tol=0.002
            ):
                raise Comparison2DValidationError(
                    f"{role} station coordinates are unstable"
                )
            z_by_mode[mode][key] = impedance
    expected = {(f, s) for f in range(8) for s in range(12)}
    if (
        any(seen[mode] != expected for mode in ("TE", "TM"))
        or not np.isfinite(station_y).all()
        or np.any(np.diff(station_y) <= 0.0)
    ):
        raise Comparison2DValidationError(f"{role} response grid is incomplete")
    derived: dict[str, np.ndarray] = {}
    omega = 2.0 * math.pi * frequencies[:, None]
    for mode in ("TE", "TM"):
        impedance = z_by_mode[mode]
        rho = _MU0 * np.abs(impedance) ** 2 / omega
        if not np.isfinite(rho).all() or np.any(rho <= 0.0):
            raise Comparison2DValidationError(f"{role} apparent resistivity is invalid")
        derived[f"log10_rho_{mode.lower()}"] = np.log10(rho)
        derived[f"phase_{mode.lower()}_deg"] = np.mod(
            np.degrees(np.angle(impedance)), 180.0
        )
    return _RawModEMResponse(
        frequencies_hz=frequencies,
        station_y_m=station_y,
        **derived,
    )


def _circular_phase_residual(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.abs((left - right + 90.0) % 180.0 - 90.0)


def _response_residuals(
    left: _RawModEMResponse, right: _RawModEMResponse, *, path: str
) -> dict[str, np.ndarray]:
    if not np.allclose(
        left.frequencies_hz, right.frequencies_hz, rtol=2e-6, atol=0.0
    ) or not np.allclose(left.station_y_m, right.station_y_m, rtol=0.0, atol=0.002):
        raise Comparison2DValidationError(f"{path} response axes differ")
    return {
        "te_log10_rho": np.abs(left.log10_rho_te - right.log10_rho_te),
        "te_phase_deg": _circular_phase_residual(left.phase_te_deg, right.phase_te_deg),
        "tm_log10_rho": np.abs(left.log10_rho_tm - right.log10_rho_tm),
        "tm_phase_deg": _circular_phase_residual(left.phase_tm_deg, right.phase_tm_deg),
    }


def _mesh_vertical_widths(
    mesh: Mapping[str, Any], canonical_depth_centres_m: np.ndarray
) -> np.ndarray:
    depth = np.ascontiguousarray(canonical_depth_centres_m, dtype="<f8")
    if (
        depth.shape != (64,)
        or hashlib.sha256(depth.tobytes()).hexdigest()
        != mesh["canonical_depth_centres_sha256"]
    ):
        raise Comparison2DValidationError(
            "nested mesh does not bind the analytic canonical depth axis"
        )
    internal = 0.5 * (depth[:-1] + depth[1:])
    bottom = depth[-1] + 0.5 * (depth[-1] - depth[-2])
    edges = np.concatenate((np.asarray([0.0]), internal, np.asarray([bottom])))
    physical_widths = np.diff(edges)
    minimum_subdivisions = int(mesh["minimum_vertical_subdivisions"])
    maximum_base_dz = float(mesh["maximum_base_dz_m"])
    counts = np.maximum(
        minimum_subdivisions,
        np.ceil(physical_widths / maximum_base_dz).astype(np.int64),
    )
    base_parts = [
        np.repeat(width / count, count)
        for width, count in zip(physical_widths, counts, strict=True)
    ]
    current_depth = float(bottom)
    macro_width = float(physical_widths[-1])
    minimum_depth = float(mesh["minimum_depth_m"])
    while current_depth < minimum_depth:
        macro_width = min(
            macro_width * float(mesh["deep_padding_growth"]),
            float(mesh["maximum_deep_macro_dz_m"]),
        )
        actual_width = min(macro_width, minimum_depth - current_depth)
        count = max(minimum_subdivisions, math.ceil(actual_width / maximum_base_dz))
        base_parts.append(np.repeat(actual_width / count, count))
        current_depth += actual_width
    base = np.concatenate(base_parts).astype(np.float64, copy=False)
    factor = int(mesh["vertical_refinement_factor"])
    result = np.repeat(base / factor, factor)
    if (
        not np.isfinite(result).all()
        or np.any(result <= 0.0)
        or not math.isclose(
            float(np.sum(result)), minimum_depth, rel_tol=0.0, abs_tol=1e-7
        )
    ):
        raise Comparison2DValidationError(
            "nested mesh analytic vertical partition is invalid"
        )
    return result


def _mesh_horizontal_widths(mesh: Mapping[str, Any]) -> np.ndarray:
    core = np.full(
        int(mesh["base_core_count"]),
        float(mesh["base_core_width_m"]),
        dtype=np.float64,
    )
    exponent = np.arange(
        1, int(mesh["base_padding_count_each_side"]) + 1, dtype=np.float64
    )
    near_to_far = (
        float(mesh["base_core_width_m"]) * float(mesh["base_padding_growth"]) ** exponent
    )
    base = np.concatenate((near_to_far[::-1], core, near_to_far))
    factor = int(mesh["horizontal_refinement_factor"])
    result = np.repeat(base / factor, factor)
    if not np.isfinite(result).all() or np.any(result <= 0.0):
        raise Comparison2DValidationError(
            "nested mesh analytic horizontal partition is invalid"
        )
    return result


def _nearest_lower_tie(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    right = np.searchsorted(reference, query, side="left")
    right = np.clip(right, 0, reference.size - 1)
    left = np.clip(right - 1, 0, reference.size - 1)
    choose_right = np.abs(reference[right] - query) < np.abs(reference[left] - query)
    return np.where(choose_right, right, left)


def _layered_impedance_h(
    frequencies_hz: np.ndarray,
    resistivity_ohm_m: np.ndarray,
    thickness_m: np.ndarray,
) -> np.ndarray:
    omega = 2.0 * math.pi * frequencies_hz
    eta = np.sqrt(1j * omega[:, None] * _MU0 * resistivity_ohm_m[None, :])
    gamma = np.sqrt(1j * omega[:, None] * _MU0 / resistivity_ohm_m[None, :])
    impedance = eta[:, -1]
    for layer in range(resistivity_ohm_m.size - 2, -1, -1):
        tangent = np.tanh(gamma[:, layer] * thickness_m[layer])
        intrinsic = eta[:, layer]
        impedance = (
            intrinsic
            * (impedance + intrinsic * tangent)
            / (intrinsic + impedance * tangent)
        )
    return impedance


def _analytic_expected_response(
    *,
    truth_id: str,
    mesh: Mapping[str, Any],
    contract: Mapping[str, Any],
    station_y_m: np.ndarray,
) -> _RawModEMResponse:
    depth_axis = contract["canonical_depth_centres_m"]
    profile = contract["profiles"].get(truth_id)
    if profile is None:
        raise Comparison2DValidationError(f"unknown analytic truth {truth_id!r}")
    canonical_rho = np.empty(depth_axis.size, dtype=np.float64)
    unassigned = np.ones(depth_axis.size, dtype=np.bool_)
    for maximum, rho in profile:
        selected = unassigned if maximum is None else unassigned & (depth_axis < maximum)
        canonical_rho[selected] = rho
        unassigned[selected] = False
    if np.any(unassigned):
        raise Comparison2DValidationError(
            "analytic profile does not cover the depth axis"
        )
    dz = _mesh_vertical_widths(mesh, depth_axis)
    mesh_centres = np.cumsum(dz) - 0.5 * dz
    resistivity = canonical_rho[_nearest_lower_tie(depth_axis, mesh_centres)]
    z_h = _layered_impedance_h(contract["frequencies_hz"], resistivity, dz[:-1])
    z_eb = z_h / _MU0
    omega = 2.0 * math.pi * contract["frequencies_hz"]
    log10_rho = np.log10(_MU0 * np.abs(z_eb) ** 2 / omega)
    phase = np.mod(np.degrees(np.angle(z_eb)), 180.0)
    repeated_log = np.repeat(log10_rho[:, None], 12, axis=1)
    repeated_phase = np.repeat(phase[:, None], 12, axis=1)
    return _RawModEMResponse(
        frequencies_hz=contract["frequencies_hz"],
        station_y_m=station_y_m,
        log10_rho_te=repeated_log,
        phase_te_deg=repeated_phase,
        log10_rho_tm=repeated_log,
        phase_tm_deg=repeated_phase,
    )


def _expected_report_gates() -> dict[str, dict[str, float]]:
    return {
        "log10_rho": dict(_CONVERGENCE_THRESHOLDS["log10_rho_dex"]),
        "phase_deg": dict(_CONVERGENCE_THRESHOLDS["phase_circular_180_degrees"]),
        "padding_log10_rho": {
            "p95": _CONVERGENCE_THRESHOLDS["padding"]["log10_rho_p95_dex"]
        },
        "padding_phase_deg": {
            "p95": _CONVERGENCE_THRESHOLDS["padding"]["phase_circular_180_p95_degrees"]
        },
    }


def _summary_from_array(values: np.ndarray) -> dict[str, float | int]:
    array = np.abs(np.asarray(values, dtype=np.float64)).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise Comparison2DValidationError(
            "mesh convergence residuals must be non-empty and finite"
        )
    return {
        "n": int(array.size),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "rmse": float(np.sqrt(np.mean(array**2))),
    }


def _validate_summary(
    value: Any,
    *,
    path: str,
    expected: Mapping[str, float | int] | None,
    expected_n: int,
) -> dict[str, float | int]:
    summary = _mapping(value, path)
    keys = frozenset({"n", "median", "p95", "max", "mean", "rmse"})
    _exact_keys(summary, keys, path)
    observed: dict[str, float | int] = {
        "n": _integer(summary["n"], f"{path}.n", minimum=1)
    }
    for name in ("median", "p95", "max", "mean", "rmse"):
        observed[name] = _finite(summary[name], f"{path}.{name}", nonnegative=True)
    if observed["n"] != expected_n:
        raise Comparison2DValidationError(f"{path}.n differs from the response contract")
    if not (
        float(observed["median"]) <= float(observed["p95"]) <= float(observed["max"])
    ):
        raise Comparison2DValidationError(f"{path} quantiles are inconsistent")
    if float(observed["rmse"]) + 1e-15 < float(observed["mean"]):
        raise Comparison2DValidationError(f"{path} RMSE is smaller than its mean")
    if expected is not None:
        for name in keys:
            if name == "n":
                if observed[name] != expected[name]:
                    raise Comparison2DValidationError(
                        f"{path}.{name} differs from raw paired residuals"
                    )
            elif not math.isclose(
                float(observed[name]),
                float(expected[name]),
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                raise Comparison2DValidationError(
                    f"{path}.{name} differs from raw paired residuals"
                )
    return observed


def _channel_thresholds(channel: str, *, padding: bool) -> Mapping[str, float]:
    quantity = "phase_deg" if channel.endswith("phase_deg") else "log10_rho"
    key = f"padding_{quantity}" if padding else quantity
    return _expected_report_gates()[key]


def _validate_gate_record(
    value: Any,
    *,
    path: str,
    summary: Mapping[str, float | int],
    thresholds: Mapping[str, float],
    padding: bool,
) -> None:
    record = _mapping(value, path)
    expected_keys = (
        frozenset({"thresholds", "passed"})
        if padding
        else frozenset({"thresholds", "checks", "passed"})
    )
    _exact_keys(record, expected_keys, path)
    if record["thresholds"] != thresholds:
        raise Comparison2DValidationError(f"{path} thresholds are not frozen")
    checks = {
        name: bool(float(summary[name]) <= limit) for name, limit in thresholds.items()
    }
    if not padding:
        check_record = _mapping(record["checks"], f"{path}.checks")
        _exact_keys(check_record, frozenset(thresholds), f"{path}.checks")
        if dict(check_record) != checks:
            raise Comparison2DValidationError(f"{path} checks differ from metrics")
    passed = all(checks.values())
    if record["passed"] is not passed or not passed:
        raise Comparison2DValidationError(f"{path} exceeds a convergence threshold")


def _validate_residual_section(
    value: Any,
    *,
    path: str,
    arrays: Mapping[str, np.ndarray],
    padding: bool,
) -> None:
    section = _mapping(value, path)
    _exact_keys(section, frozenset({"aggregate", "gates", "passed"}), path)
    aggregate = _mapping(section["aggregate"], f"{path}.aggregate")
    gates = _mapping(section["gates"], f"{path}.gates")
    _exact_keys(aggregate, frozenset(_CONVERGENCE_CHANNELS), f"{path}.aggregate")
    _exact_keys(gates, frozenset(_CONVERGENCE_CHANNELS), f"{path}.gates")
    for channel in _CONVERGENCE_CHANNELS:
        expected = _summary_from_array(arrays[channel])
        summary = _validate_summary(
            aggregate[channel],
            path=f"{path}.aggregate.{channel}",
            expected=expected,
            expected_n=int(arrays[channel].size),
        )
        _validate_gate_record(
            gates[channel],
            path=f"{path}.gates.{channel}",
            summary=summary,
            thresholds=_channel_thresholds(channel, padding=padding),
            padding=padding,
        )
    if section["passed"] is not True:
        raise Comparison2DValidationError(f"{path} did not pass")


def _validate_modem_run_provenance(
    value: Mapping[str, Any],
    *,
    path: str,
    forward_snapshot: ArtifactSnapshot,
    expected_truth_id: str,
    expected_mesh: Mapping[str, Any],
    expected_runtime: Mapping[str, Any],
    expected_canonical_depth_centres_m: np.ndarray,
    generator: Mapping[str, Any],
    expected_truth_identity: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    _exact_keys(
        value,
        frozenset(
            {
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
        ),
        path,
    )
    if value["schema"] != "pimsr-modem2d-forward-run" or value["schema_version"] != 1:
        raise Comparison2DValidationError(f"{path} schema identity is wrong")
    truth = _mapping(value["truth"], f"{path}.truth")
    _exact_keys(
        truth,
        frozenset(
            {
                "schema",
                "schema_version",
                "sample_id",
                "model_shape",
                "response_shape",
                "arrays_sha256",
            }
        ),
        f"{path}.truth",
    )
    if (
        truth["schema"] != "pimsr-canonical-truth-2d"
        or truth["schema_version"] != 1
        or truth["sample_id"] != expected_truth_id
        or truth["model_shape"] != list(CANONICAL_MODEL_SHAPE)
        or truth["response_shape"] != [8, 12]
    ):
        raise Comparison2DValidationError(f"{path} truth identity is wrong")
    _sha256(truth["arrays_sha256"], f"{path}.truth.arrays_sha256")
    if expected_truth_identity is not None and dict(truth) != dict(
        expected_truth_identity
    ):
        raise Comparison2DValidationError(
            f"{path} truth identity was not recomputed from material geology"
        )
    mesh = _mesh_config_record(value["mesh"], path=f"{path}.mesh", reference_only=False)
    if any(mesh[key] != expected_mesh[key] for key in _MESH_RECORD_KEYS) or (
        mesh["mesh_config_sha256"] != expected_mesh["mesh_config_sha256"]
    ):
        raise Comparison2DValidationError(f"{path} mesh differs from the report")
    runtime = _mapping(value["runtime"], f"{path}.runtime")
    if dict(runtime) != dict(expected_runtime):
        raise Comparison2DValidationError(
            f"{path} runtime differs from convergence report"
        )
    runtime_sha = _sha256(
        value["runtime_identity_sha256"], f"{path}.runtime_identity_sha256"
    )
    if runtime_sha != _canonical_object_sha256(runtime):
        raise Comparison2DValidationError(f"{path} runtime identity SHA-256 is wrong")
    bridge = _file_identity_record(value["bridge_source"], f"{path}.bridge_source")
    if bridge != (generator["converter_sha256"], generator["converter_size_bytes"]):
        raise Comparison2DValidationError(
            f"{path} bridge source differs from preregistration"
        )
    response = _mapping(value["response_contract"], f"{path}.response_contract")
    _exact_keys(
        response,
        frozenset(
            {
                "artifact",
                "rows",
                "all_rows_finite",
                "time_convention",
                "manual_conjugation",
                "native_units",
                "rho_formula",
                "phase_formula",
                "canonical_mode_order",
            }
        ),
        f"{path}.response_contract",
    )
    response_identity = _file_identity_record(
        response["artifact"], f"{path}.response_contract.artifact"
    )
    if (
        response_identity != (forward_snapshot.sha256, forward_snapshot.size_bytes)
        or response["rows"] != {"TE": 96, "TM": 96}
        or response["all_rows_finite"] is not True
        or response["time_convention"] != "exp(+i omega t) as written by ModEM DataIO"
        or response["manual_conjugation"] is not False
        or response["native_units"] != "[V/m]/[T] (E/B)"
        or response["rho_formula"] != "mu0 * abs(E_over_B)**2 / omega"
        or response["phase_formula"] != "degrees(angle(E_over_B)) modulo 180"
        or response["canonical_mode_order"] != ["TE_Zyx", "TM_Zxy"]
    ):
        raise Comparison2DValidationError(f"{path} response contract is wrong")
    outputs = _mapping(value["outputs"], f"{path}.outputs")
    expected_outputs = frozenset(
        {
            "model.rho",
            "template.dat",
            "forward.dat",
            "responses.npz",
            "solver.stdout.txt",
            "solver.stderr.txt",
        }
    )
    _exact_keys(outputs, expected_outputs, f"{path}.outputs")
    forward_output = _mapping(outputs["forward.dat"], f"{path}.outputs.forward.dat")
    _exact_keys(
        forward_output,
        frozenset({"sha256", "size_bytes"}),
        f"{path}.outputs.forward.dat",
    )
    if (
        forward_output["sha256"] != forward_snapshot.sha256
        or forward_output["size_bytes"] != forward_snapshot.size_bytes
    ):
        raise Comparison2DValidationError(f"{path} does not bind its forward.dat")
    for name in expected_outputs - {"forward.dat"}:
        output = _mapping(outputs[name], f"{path}.outputs.{name}")
        _exact_keys(output, frozenset({"sha256", "size_bytes"}), f"{path}.outputs.{name}")
        _sha256(output["sha256"], f"{path}.outputs.{name}.sha256")
        _integer(output["size_bytes"], f"{path}.outputs.{name}.size_bytes", minimum=0)
    execution = _mapping(value["execution"], f"{path}.execution")
    _exact_keys(
        execution,
        frozenset(
            {
                "command",
                "container_network",
                "container_root_filesystem",
                "input_mount",
                "runtime_mount",
                "timeout_seconds",
                "elapsed_seconds",
                "returncode",
            }
        ),
        f"{path}.execution",
    )
    if (
        execution["container_network"] != "none"
        or execution["container_root_filesystem"] != "read_only"
        or execution["input_mount"] != "read_only"
        or execution["runtime_mount"] != "read_only"
        or execution["returncode"] != 0
    ):
        raise Comparison2DValidationError(f"{path} execution isolation is wrong")
    _sequence(execution["command"], f"{path}.execution.command")
    command = list(execution["command"])
    expected_tail = [
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
        None,
        "--mount",
        None,
        "--mount",
        None,
        generator["container_image_digest"],
        "/runtime/bin/Mod2DMT",
        "-F",
        "/input/model.rho",
        "/input/template.dat",
        "/output/forward.dat",
    ]
    if (
        len(command) != 24
        or not all(isinstance(item, str) and item for item in command)
        or Path(command[0]).name.casefold() not in {"docker", "docker.exe"}
        or any(
            wanted is not None and command[index + 1] != wanted
            for index, wanted in enumerate(expected_tail)
        )
        or "target=/runtime" not in command[13]
        or not command[13].endswith(",readonly")
        or "target=/input" not in command[15]
        or not command[15].endswith(",readonly")
        or "target=/output" not in command[17]
        or command[17].endswith(",readonly")
    ):
        raise Comparison2DValidationError(f"{path} execution command is not exact")
    _finite(execution["timeout_seconds"], f"{path}.execution.timeout_seconds")
    _finite(execution["elapsed_seconds"], f"{path}.execution.elapsed_seconds")
    input_contract = _mapping(value["input_contract"], f"{path}.input_contract")
    _exact_keys(
        input_contract, frozenset({"model", "template"}), f"{path}.input_contract"
    )
    model_contract = _mapping(input_contract["model"], f"{path}.input_contract.model")
    template_contract = _mapping(
        input_contract["template"], f"{path}.input_contract.template"
    )
    _exact_keys(
        model_contract,
        frozenset(
            {
                "representation",
                "ny",
                "nz_earth",
                "total_width_m",
                "total_depth_m",
                "mapping",
                "spatial_operation_order",
                "artifact",
            }
        ),
        f"{path}.input_contract.model",
    )
    _exact_keys(
        template_contract,
        frozenset(
            {
                "time_convention_requested",
                "manual_conjugation",
                "units",
                "coordinate_mapping",
                "mode_mapping",
                "period_count",
                "station_count",
                "rows_per_mode",
                "artifact",
            }
        ),
        f"{path}.input_contract.template",
    )
    if (
        model_contract["representation"] != "LOGE natural_log_resistivity_ohm_m"
        or model_contract["mapping"] != _MESH_MAPPING
        or model_contract["spatial_operation_order"]
        != (
            "map canonical log10(rho) piecewise-constantly by physical centres, "
            "then multiply mapped values by ln(10)"
        )
        or template_contract["time_convention_requested"] != "exp(+i omega t)"
        or template_contract["manual_conjugation"] is not False
        or template_contract["units"] != "[V/m]/[T] (E/B)"
        or template_contract["coordinate_mapping"]
        != {
            "ModEM_X_m": 0.0,
            "ModEM_Y_m": "total_width_m/2 + PIMSR_station_x_m",
            "ModEM_Z_m": 0.0,
        }
        or template_contract["mode_mapping"]
        != {
            "ModEM_TE_Ex_over_By": "PIMSR_TE_Ey_over_Hx_Zyx_no_mode_swap",
            "ModEM_TM_Ey_over_Bx": "PIMSR_TM_Ex_over_Hy_Zxy_no_mode_swap",
        }
        or template_contract["period_count"] != 8
        or template_contract["station_count"] != 12
        or template_contract["rows_per_mode"] != 96
    ):
        raise Comparison2DValidationError(f"{path} solver input contract is wrong")
    dy = _mesh_horizontal_widths(expected_mesh)
    dz = _mesh_vertical_widths(expected_mesh, expected_canonical_depth_centres_m)
    expected_model_dimensions = {
        "ny": int(dy.size),
        "nz_earth": int(dz.size),
        "total_width_m": float(np.sum(dy)),
        "total_depth_m": float(np.sum(dz)),
    }
    for name, expected_dimension in expected_model_dimensions.items():
        observed_dimension = (
            _integer(model_contract[name], f"{path}.input_contract.model.{name}")
            if name in {"ny", "nz_earth"}
            else _finite(model_contract[name], f"{path}.input_contract.model.{name}")
        )
        if not math.isclose(
            float(observed_dimension),
            float(expected_dimension),
            rel_tol=0.0,
            abs_tol=1e-7,
        ):
            raise Comparison2DValidationError(
                f"{path} solver model dimensions differ from the frozen mesh"
            )
    model_artifact = _file_identity_record(
        model_contract.get("artifact"), f"{path}.input_contract.model.artifact"
    )
    template_artifact = _file_identity_record(
        template_contract.get("artifact"), f"{path}.input_contract.template.artifact"
    )
    model_output = _mapping(outputs["model.rho"], f"{path}.outputs.model.rho")
    template_output = _mapping(outputs["template.dat"], f"{path}.outputs.template.dat")
    model_output_identity = (
        model_output["sha256"],
        model_output["size_bytes"],
    )
    template_output_identity = (
        template_output["sha256"],
        template_output["size_bytes"],
    )
    if (
        model_artifact != model_output_identity
        or template_artifact != template_output_identity
    ):
        raise Comparison2DValidationError(
            f"{path} generated input artifacts differ from the published solver inputs"
        )
    return _mapping(value["truth_source"], f"{path}.truth_source")


def _load_public_shard_material(
    snapshot: ArtifactSnapshot,
    *,
    expected_generator_seed: int,
    expected_sample_start: int,
    expected_sample_end: int,
) -> _PublicShardMaterial:
    required = {
        "target_log10_res",
        "x_grid",
        "depth_grid",
        "frequencies",
        "station_x",
        "sample_index",
        "scenario",
    }
    try:
        with h5py.File(io.BytesIO(snapshot.payload), "r") as h5:
            if (
                h5.attrs.get("schema") != "pimsr-mt-2d"
                or int(h5.attrs.get("schema_version", -1)) != 2
                or not required.issubset(h5)
            ):
                raise Comparison2DValidationError(
                    "public convergence source shard schema is invalid"
                )
            target = np.asarray(h5["target_log10_res"][:])
            x_grid = np.asarray(h5["x_grid"][:])
            depth_grid = np.asarray(h5["depth_grid"][:])
            frequencies = np.asarray(h5["frequencies"][:])
            station_x = np.asarray(h5["station_x"][:])
            sample_index = np.asarray(h5["sample_index"][:])
            scenarios = np.asarray(h5["scenario"][:])
            generator_seed = int(h5.attrs.get("generator_seed", -1))
            generation_contract = str(h5.attrs.get("generation_contract", ""))
            forward_contract = str(h5.attrs.get("forward_contract", ""))
    except Comparison2DValidationError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise Comparison2DValidationError(
            f"cannot decode public convergence source shard: {exc}"
        ) from exc
    count = int(sample_index.size)
    if (
        count != 100
        or generator_seed != expected_generator_seed
        or not generation_contract
        or not forward_contract
        or target.dtype != np.dtype("<f4")
        or target.shape != (count, *CANONICAL_MODEL_SHAPE)
        or x_grid.dtype != np.dtype("<f8")
        or x_grid.shape != (CANONICAL_MODEL_SHAPE[1],)
        or depth_grid.dtype != np.dtype("<f8")
        or depth_grid.shape != (CANONICAL_MODEL_SHAPE[0],)
        or frequencies.dtype != np.dtype("<f8")
        or frequencies.shape != (8,)
        or station_x.dtype != np.dtype("<f8")
        or station_x.shape != (12,)
        or sample_index.dtype != np.dtype("<i8")
        or sample_index.shape != (count,)
        or expected_sample_end - expected_sample_start + 1 != count
        or not np.array_equal(
            sample_index,
            np.arange(expected_sample_start, expected_sample_end + 1, dtype="<i8"),
        )
        or scenarios.dtype != np.dtype("<i4")
        or scenarios.shape != (count,)
        or np.unique(sample_index).size != count
        or not np.isfinite(target).all()
        or not np.isfinite(x_grid).all()
        or not np.isfinite(depth_grid).all()
        or not np.isfinite(frequencies).all()
        or not np.isfinite(station_x).all()
        or hashlib.sha256(x_grid.tobytes(order="C")).hexdigest()
        != _CANONICAL_X_CENTRES_SHA256
        or hashlib.sha256(depth_grid.tobytes(order="C")).hexdigest()
        != _CANONICAL_DEPTH_CENTRES_SHA256
        or np.any((scenarios < 0) | (scenarios >= len(FAMILY_IDS)))
    ):
        raise Comparison2DValidationError(
            "public convergence source shard material contract is invalid"
        )
    row_by_sample: dict[int, int] = {}
    scenario_by_sample: dict[int, int] = {}
    identities: dict[int, Mapping[str, Any]] = {}
    for row, raw_sample_id in enumerate(sample_index):
        sample_id = int(raw_sample_id)
        scenario_index = int(scenarios[row])
        truth = CanonicalTruth(
            log10_resistivity=target[row],
            x_centres_m=x_grid,
            depth_centres_m=depth_grid,
            frequencies_hz=frequencies,
            station_x_m=station_x,
            sample_id=f"sample-{sample_id:06d}",
        )
        row_by_sample[sample_id] = row
        scenario_by_sample[sample_id] = scenario_index
        identities[sample_id] = truth.identity_record()
    return _PublicShardMaterial(
        artifact_sha256=snapshot.sha256,
        artifact_size_bytes=snapshot.size_bytes,
        generator_seed=generator_seed,
        generation_contract=generation_contract,
        forward_contract=forward_contract,
        row_by_sample=row_by_sample,
        scenario_by_sample=scenario_by_sample,
        truth_identity_by_sample=identities,
    )


def _public_truth_source(
    value: Mapping[str, Any],
    *,
    path: str,
    sample_index: int,
    scenario_index: int,
    scenario: str,
    source_material: _PublicShardMaterial,
    expected_source_reference: Mapping[str, Any],
) -> None:
    _exact_keys(
        value,
        frozenset(
            {
                "source",
                "row",
                "sample_index",
                "scenario_index",
                "generator_seed",
                "generation_contract",
                "forward_contract",
                "public_validation",
            }
        ),
        path,
    )
    source_reference = _mapping(value["source"], f"{path}.source")
    source_identity = _file_identity_record(source_reference, f"{path}.source")
    expected_reference = _mapping(
        expected_source_reference,
        "public_convergence_raw_runs.source_shard",
    )
    _file_identity_record(
        expected_reference,
        "public_convergence_raw_runs.source_shard",
    )
    validation = _mapping(value["public_validation"], f"{path}.public_validation")
    _exact_keys(
        validation,
        frozenset({"selection_policy", "scenario_name", "validator_source"}),
        f"{path}.public_validation",
    )
    if (
        source_identity
        != (source_material.artifact_sha256, source_material.artifact_size_bytes)
        or dict(source_reference) != dict(expected_reference)
        or value["row"] != source_material.row_by_sample.get(sample_index)
        or value["sample_index"] != sample_index
        or value["scenario_index"] != scenario_index
        or source_material.scenario_by_sample.get(sample_index) != scenario_index
        or validation["scenario_name"] != scenario
        or validation["selection_policy"] != "lowest sample_index per frozen scenario"
        or value["generator_seed"] != source_material.generator_seed
        or value["generation_contract"] != source_material.generation_contract
        or value["forward_contract"] != source_material.forward_contract
    ):
        raise Comparison2DValidationError(f"{path} public selection provenance is wrong")
    _integer(value["row"], f"{path}.row")
    _integer(value["generator_seed"], f"{path}.generator_seed")
    _string(value["generation_contract"], f"{path}.generation_contract")
    _string(value["forward_contract"], f"{path}.forward_contract")
    _file_identity_record(validation["validator_source"], f"{path}.validator_source")


def _analytic_truth_source(
    value: Mapping[str, Any], *, path: str, validator_identity: tuple[str, int]
) -> None:
    _exact_keys(value, frozenset({"scope", "validator_source"}), path)
    if (
        value["scope"] != "analytic_public_validation"
        or _file_identity_record(value["validator_source"], f"{path}.validator_source")
        != validator_identity
    ):
        raise Comparison2DValidationError(f"{path} analytic provenance is wrong")


def _convergence_family_index(value: Any, path: str) -> int:
    family = _string(value, path)
    if family not in _CONVERGENCE_FAMILIES:
        raise Comparison2DValidationError(f"{path} is not a frozen public family")
    return _CONVERGENCE_FAMILIES.index(family)


def _validated_public_selection(
    public_rows: Sequence[Any],
    *,
    source_material: _PublicShardMaterial,
) -> tuple[int, ...]:
    selected_samples = tuple(
        sample_id
        for scenario_index in range(len(FAMILY_IDS))
        for sample_id in sorted(
            sample
            for sample, scenario in source_material.scenario_by_sample.items()
            if scenario == scenario_index
        )[:5]
    )
    report_samples = tuple(
        _integer(
            _mapping(row, f"per_geology[{index}]")["sample_index"],
            f"per_geology[{index}].sample_index",
        )
        for index, row in enumerate(public_rows)
    )
    if len(selected_samples) != 25 or report_samples != selected_samples:
        raise Comparison2DValidationError(
            "public convergence rows are not the deterministic lowest five per family "
            "from the pinned lineage shard"
        )
    return selected_samples


def _validate_public_convergence_raw_runs(
    value: Any,
    *,
    report: Mapping[str, Any],
    evidence: Mapping[str, Any],
    public_lineage: Mapping[str, Any],
    base: Path,
    seen_identities: set[tuple[int, int]],
) -> _PublicRawValidation:
    raw_set = _mapping(value, "post-score public_convergence_raw_runs")
    _exact_keys(
        raw_set,
        frozenset(
            {
                "schema",
                "schema_version",
                "source_lineage_split",
                "source_shard_ordinal",
                "source_shard",
                "runs",
            }
        ),
        "public_convergence_raw_runs",
    )
    if (
        raw_set["schema"] != _PUBLIC_RAW_RUN_SCHEMA
        or raw_set["schema_version"] != _PUBLIC_RAW_RUN_SCHEMA_VERSION
    ):
        raise Comparison2DValidationError(
            "public convergence raw-run set identity is wrong"
        )
    raw_runs = _sequence(raw_set["runs"], "public_convergence_raw_runs.runs")
    if len(raw_runs) != PUBLIC_CONVERGENCE_RAW_RUN_COUNT:
        raise Comparison2DValidationError(
            "public convergence requires exactly 80 raw runs"
        )
    split = _string(
        raw_set["source_lineage_split"],
        "public_convergence_raw_runs.source_lineage_split",
    )
    ordinal = _integer(
        raw_set["source_shard_ordinal"],
        "public_convergence_raw_runs.source_shard_ordinal",
    )
    if split != "train" or ordinal != 0:
        raise Comparison2DValidationError(
            "public convergence must use frozen train lineage shard ordinal zero"
        )
    lineage_split = _mapping(
        public_lineage.get(split), "public convergence lineage split"
    )
    lineage_shards = _sequence(
        lineage_split.get("shards"), "public convergence lineage shards"
    )
    if ordinal >= len(lineage_shards):
        raise Comparison2DValidationError(
            "public convergence source shard is absent from validated lineage"
        )
    lineage_shard = _mapping(lineage_shards[ordinal], "public convergence lineage shard")
    source_shard = _portable_artifact_reference(
        raw_set["source_shard"],
        base=base,
        role="public_convergence_raw_runs.source_shard",
        seen_identities=seen_identities,
    )
    if (
        source_shard.sha256 != lineage_shard.get("hdf5_sha256")
        or source_shard.size_bytes != lineage_shard.get("hdf5_size_bytes")
        or Path(
            _mapping(
                raw_set["source_shard"],
                "public_convergence_raw_runs.source_shard",
            )["path"]
        ).name
        != lineage_shard.get("hdf5_filename")
    ):
        raise Comparison2DValidationError(
            "public convergence source shard differs from validated public lineage"
        )
    source_material = _load_public_shard_material(
        source_shard,
        expected_generator_seed=20260820,
        expected_sample_start=_integer(
            lineage_shard.get("sample_start"),
            "public convergence lineage shard sample_start",
        ),
        expected_sample_end=_integer(
            lineage_shard.get("sample_end"),
            "public convergence lineage shard sample_end",
        ),
    )
    production = _mesh_config_record(
        report["production_candidate"], path="production_candidate", reference_only=False
    )
    reference = _mesh_config_record(
        report["next_finer_reference"], path="next_finer_reference", reference_only=True
    )
    padding = _mesh_config_record(
        report["padding_perturbation"], path="padding_perturbation", reference_only=False
    )
    mesh_by_role = {
        "production-candidate": production,
        "next-finer-reference": reference,
        "padding-perturbation": padding,
    }
    report_provenance = _mapping(report["provenance"], "convergence report provenance")
    expected_runtime = _mapping(report_provenance["runtime"], "convergence runtime")
    validator_identity = _file_identity_record(
        report_provenance["validator_source"], "convergence validator source"
    )
    generator = evidence["hidden_observation_generator"]
    analytic_contract = _analytic_1d_contract(
        evidence["public_mesh_convergence"]["analytic_1d_contract"]
    )
    descriptors: list[dict[str, Any]] = []
    public_rows = _sequence(report["per_geology"], "per_geology")
    _validated_public_selection(public_rows, source_material=source_material)
    for raw_row in public_rows:
        row = _mapping(raw_row, "per_geology raw-run descriptor")
        sample_index = _integer(row["sample_index"], "per_geology.sample_index")
        truth_id = f"sample-{sample_index:06d}"
        for role in (
            "production-candidate",
            "next-finer-reference",
            "padding-perturbation",
        ):
            descriptors.append(
                {
                    "case_id": f"public:{sample_index}:{role}",
                    "case_kind": "public_geology",
                    "sample_index": sample_index,
                    "truth_id": truth_id,
                    "role": role,
                    "mesh_id": mesh_by_role[role]["mesh_id"],
                    "report_row": row,
                    "mesh": mesh_by_role[role],
                }
            )
    analytic_records = _sequence(
        _mapping(report["analytic_checks"], "analytic_checks")["records"],
        "analytic_checks.records",
    )
    for raw_record in analytic_records:
        record = _mapping(raw_record, "analytic raw-run descriptor")
        truth_id = _string(record["truth_id"], "analytic truth_id")
        mesh_id = _identifier(record["mesh_id"], "analytic mesh_id")
        if mesh_id == production["mesh_id"]:
            role = "production-candidate"
            mesh = production
        elif mesh_id == reference["mesh_id"]:
            role = "next-finer-reference"
            mesh = reference
        else:
            raise Comparison2DValidationError("analytic raw run uses an unknown mesh")
        descriptors.append(
            {
                "case_id": f"analytic:{truth_id}:{role}",
                "case_kind": "analytic",
                "sample_index": None,
                "truth_id": truth_id,
                "role": role,
                "mesh_id": mesh_id,
                "report_record": record,
                "mesh": mesh,
            }
        )
    first_row = _mapping(public_rows[0], "per_geology[0]")
    first_index = _integer(first_row["sample_index"], "per_geology[0].sample_index")
    descriptors.append(
        {
            "case_id": f"determinism:{first_index}:repeat",
            "case_kind": "determinism_repeat",
            "sample_index": first_index,
            "truth_id": f"sample-{first_index:06d}",
            "role": "determinism-repeat",
            "mesh_id": production["mesh_id"],
            "report_row": first_row,
            "mesh": production,
        }
    )
    if len(descriptors) != PUBLIC_CONVERGENCE_RAW_RUN_COUNT:
        raise Comparison2DValidationError(
            "convergence report does not describe exact 25x3+4+1 runs"
        )
    responses: dict[str, _RawModEMResponse] = {}
    forward_snapshots: dict[str, ArtifactSnapshot] = {}
    identity_records: list[dict[str, Any]] = []
    for index, (raw_run, expected) in enumerate(zip(raw_runs, descriptors, strict=True)):
        path = f"public_convergence_raw_runs.runs[{index}]"
        run = _mapping(raw_run, path)
        _exact_keys(
            run,
            frozenset(
                {
                    "case_id",
                    "case_kind",
                    "sample_index",
                    "truth_id",
                    "role",
                    "mesh_id",
                    "forward",
                    "provenance",
                }
            ),
            path,
        )
        for key in (
            "case_id",
            "case_kind",
            "sample_index",
            "truth_id",
            "role",
            "mesh_id",
        ):
            if run[key] != expected[key]:
                raise Comparison2DValidationError(
                    f"{path}.{key} differs from exact run order"
                )
        forward = _portable_artifact_reference(
            run["forward"],
            base=base,
            role=f"{path}.forward",
            seen_identities=seen_identities,
        )
        provenance = _portable_artifact_reference(
            run["provenance"],
            base=base,
            role=f"{path}.provenance",
            seen_identities=seen_identities,
        )
        provenance_value = _strict_json(
            provenance, f"{path}.provenance", require_canonical=False
        )
        expected_provenance_payload = (
            json.dumps(provenance_value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        if provenance.payload != expected_provenance_payload:
            raise Comparison2DValidationError(
                f"{path}.provenance is not deterministic JSON"
            )
        response = _parse_modem_forward_snapshot(forward, role=f"{path}.forward")
        truth_source = _validate_modem_run_provenance(
            provenance_value,
            path=f"{path}.provenance",
            forward_snapshot=forward,
            expected_truth_id=expected["truth_id"],
            expected_mesh=expected["mesh"],
            expected_runtime=expected_runtime,
            expected_canonical_depth_centres_m=analytic_contract[
                "canonical_depth_centres_m"
            ],
            generator=generator,
            expected_truth_identity=(
                source_material.truth_identity_by_sample[expected["sample_index"]]
                if expected["case_kind"] in {"public_geology", "determinism_repeat"}
                else None
            ),
        )
        if expected["case_kind"] in {"public_geology", "determinism_repeat"}:
            report_row = expected["report_row"]
            scenario = _string(report_row["scenario"], "public convergence scenario")
            scenario_index = _convergence_family_index(
                scenario, "public convergence scenario"
            )
            _public_truth_source(
                truth_source,
                path=f"{path}.provenance.truth_source",
                sample_index=expected["sample_index"],
                scenario_index=scenario_index,
                scenario=scenario,
                source_material=source_material,
                expected_source_reference=_mapping(
                    raw_set["source_shard"],
                    "public_convergence_raw_runs.source_shard",
                ),
            )
            if report_row["source_shard_sha256"] != source_material.artifact_sha256:
                raise Comparison2DValidationError(
                    f"{path} report source shard differs from material lineage"
                )
        else:
            _analytic_truth_source(
                truth_source,
                path=f"{path}.provenance.truth_source",
                validator_identity=validator_identity,
            )
        responses[expected["case_id"]] = response
        forward_snapshots[expected["case_id"]] = forward
        identity_records.append(
            {
                "case_id": expected["case_id"],
                "forward_sha256": forward.sha256,
                "forward_size_bytes": forward.size_bytes,
                "provenance_sha256": provenance.sha256,
                "provenance_size_bytes": provenance.size_bytes,
            }
        )
        if expected["case_kind"] == "public_geology":
            output = _mapping(
                expected["report_row"]["outputs"][expected["role"]],
                f"per_geology outputs {expected['role']}",
            )
            if _file_identity_record(output["forward"], "report raw forward") != (
                forward.sha256,
                forward.size_bytes,
            ) or _file_identity_record(output["provenance"], "report raw provenance") != (
                provenance.sha256,
                provenance.size_bytes,
            ):
                raise Comparison2DValidationError(
                    f"{path} differs from convergence report"
                )
    for response in responses.values():
        if not np.allclose(
            response.frequencies_hz,
            analytic_contract["frequencies_hz"],
            rtol=2e-6,
            atol=0.0,
        ):
            raise Comparison2DValidationError(
                "raw ModEM frequency grid differs from preregistration"
            )
    residual_lists = {
        f"{prefix}_{channel}": []
        for prefix in ("candidate_vs_reference", "candidate_vs_padding")
        for channel in _CONVERGENCE_CHANNELS
    }
    for raw_row in public_rows:
        row = _mapping(raw_row, "per_geology recomputation")
        sample_index = int(row["sample_index"])
        candidate = responses[f"public:{sample_index}:production-candidate"]
        reference_response = responses[f"public:{sample_index}:next-finer-reference"]
        padding_response = responses[f"public:{sample_index}:padding-perturbation"]
        for prefix, values in (
            (
                "candidate_vs_reference",
                _response_residuals(
                    candidate, reference_response, path=f"sample {sample_index} reference"
                ),
            ),
            (
                "candidate_vs_padding",
                _response_residuals(
                    candidate, padding_response, path=f"sample {sample_index} padding"
                ),
            ),
        ):
            for channel, array in values.items():
                residual_lists[f"{prefix}_{channel}"].append(array)
    recomputed_residuals: dict[str, np.ndarray] = {
        "sample_index": np.asarray(
            [int(_mapping(item, "per_geology")["sample_index"]) for item in public_rows],
            dtype=np.int64,
        ),
        "scenario_index": np.asarray(
            [
                _convergence_family_index(
                    _mapping(item, "per_geology")["scenario"], "scenario"
                )
                for item in public_rows
            ],
            dtype=np.int64,
        ),
    }
    recomputed_residuals.update(
        {name: np.stack(values, axis=0) for name, values in residual_lists.items()}
    )
    analytic_residuals: dict[tuple[str, str], Mapping[str, np.ndarray]] = {}
    for raw_record in analytic_records:
        record = _mapping(raw_record, "analytic recomputation")
        truth_id = str(record["truth_id"])
        mesh_id = str(record["mesh_id"])
        role = (
            "production-candidate"
            if mesh_id == production["mesh_id"]
            else "next-finer-reference"
        )
        raw_response = responses[f"analytic:{truth_id}:{role}"]
        mesh = production if role == "production-candidate" else reference
        expected_response = _analytic_expected_response(
            truth_id=truth_id,
            mesh=mesh,
            contract=analytic_contract,
            station_y_m=raw_response.station_y_m,
        )
        analytic_residuals[(truth_id, mesh_id)] = _response_residuals(
            raw_response, expected_response, path=f"analytic {truth_id}/{mesh_id}"
        )
    first_case = f"public:{first_index}:production-candidate"
    repeat_case = f"determinism:{first_index}:repeat"
    if forward_snapshots[first_case].payload != forward_snapshots[repeat_case].payload:
        raise Comparison2DValidationError(
            "raw ModEM determinism repeat is not byte-identical"
        )
    determinism = _mapping(report["determinism"], "determinism")
    if _file_identity_record(
        determinism["first_forward"], "determinism.first_forward"
    ) != (
        forward_snapshots[first_case].sha256,
        forward_snapshots[first_case].size_bytes,
    ) or _file_identity_record(
        determinism["repeat_forward"], "determinism.repeat_forward"
    ) != (
        forward_snapshots[repeat_case].sha256,
        forward_snapshots[repeat_case].size_bytes,
    ):
        raise Comparison2DValidationError(
            "determinism report differs from raw forward files"
        )
    return _PublicRawValidation(
        residuals=recomputed_residuals,
        analytic_residuals=analytic_residuals,
        identities_sha256=_canonical_object_sha256({"runs": identity_records}),
        artifact_count=len(identity_records) * 2 + 1,
    )


def _load_convergence_residuals(
    snapshot: ArtifactSnapshot,
    *,
    report: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    raw_record = _mapping(report.get("raw_paired_residuals"), "raw_paired_residuals")
    _exact_keys(
        raw_record,
        frozenset({"filename", "sha256", "size_bytes"}),
        "raw_paired_residuals",
    )
    if (
        raw_record["filename"] != "paired-residuals.npz"
        or raw_record["sha256"] != snapshot.sha256
        or raw_record["size_bytes"] != snapshot.size_bytes
    ):
        raise Comparison2DValidationError(
            "mesh convergence report does not bind the pinned raw residual archive"
        )
    ensemble = _mapping(report.get("public_ensemble"), "public_ensemble")
    sample_count = _integer(
        ensemble.get("sample_count"), "public_ensemble.sample_count", minimum=25
    )
    if sample_count > 10_000:
        raise Comparison2DValidationError(
            "public convergence residual archive exceeds the bounded schema"
        )
    expected_zip_names = tuple(f"{name}.npy" for name in _CONVERGENCE_RESIDUAL_MEMBERS)
    try:
        with zipfile.ZipFile(io.BytesIO(snapshot.payload), mode="r") as archive:
            infos = archive.infolist()
            if tuple(info.filename for info in infos) != expected_zip_names:
                raise Comparison2DValidationError(
                    "mesh convergence NPZ member order/schema is not exact"
                )
            if archive.comment or archive.testzip() is not None:
                raise Comparison2DValidationError(
                    "mesh convergence NPZ ZIP structure is invalid"
                )
            maximum_uncompressed = sample_count * 6_400 + 65_536
            if (
                any(
                    info.compress_type != zipfile.ZIP_DEFLATED
                    or info.flag_bits & 0x1
                    or info.file_size <= 0
                    for info in infos
                )
                or sum(info.file_size for info in infos) > maximum_uncompressed
            ):
                raise Comparison2DValidationError(
                    "mesh convergence NPZ compression contract is invalid"
                )
        with np.load(
            io.BytesIO(snapshot.payload),
            allow_pickle=False,
            max_header_size=4_096,
        ) as archive:
            if tuple(archive.files) != _CONVERGENCE_RESIDUAL_MEMBERS:
                raise Comparison2DValidationError(
                    "mesh convergence NPZ array schema is not exact"
                )
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    except Comparison2DValidationError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise Comparison2DValidationError(
            f"cannot decode pinned mesh convergence residuals: {exc}"
        ) from exc
    for name in ("sample_index", "scenario_index"):
        array = arrays[name]
        if array.dtype != np.dtype(np.int64) or array.shape != (sample_count,):
            raise Comparison2DValidationError(f"mesh convergence {name} schema is wrong")
    for name in _CONVERGENCE_RESIDUAL_MEMBERS[2:]:
        array = arrays[name]
        if (
            array.dtype != np.dtype(np.float64)
            or array.shape != (sample_count, 8, 12)
            or not array.flags.c_contiguous
            or not np.isfinite(array).all()
            or np.any(array < 0.0)
        ):
            raise Comparison2DValidationError(
                f"mesh convergence {name} must be finite nonnegative float64 [N,8,12]"
            )
    return arrays


def _validate_report_success(
    report: Mapping[str, Any],
    *,
    evidence: Mapping[str, Any],
    residuals: Mapping[str, np.ndarray],
    residual_archive: Mapping[str, np.ndarray],
    analytic_residuals: Mapping[tuple[str, str], Mapping[str, np.ndarray]],
    residuals_snapshot: ArtifactSnapshot,
) -> None:
    expected_root = frozenset(
        {
            "schema",
            "schema_version",
            "passed",
            "headline_eligible",
            "scope",
            "production_candidate",
            "next_finer_reference",
            "padding_perturbation",
            "paired_residual_definition",
            "frozen_gates",
            "public_ensemble",
            "candidate_vs_reference",
            "candidate_vs_padding",
            "per_geology",
            "analytic_checks",
            "determinism",
            "response_contract",
            "provenance",
            "raw_paired_residuals",
            "blocker",
        }
    )
    _exact_keys(report, expected_root, "mesh convergence report")
    if (
        report["schema"] != _CONVERGENCE_REPORT_SCHEMA
        or report["schema_version"] != _CONVERGENCE_REPORT_SCHEMA_VERSION
        or report["scope"] != "public_only_no_hidden_or_secret_access"
        or report["passed"] is not True
        or report["headline_eligible"] is not True
        or report["blocker"] is not None
    ):
        raise Comparison2DValidationError(
            "mesh convergence report is not a successful public-only validation"
        )
    convergence = evidence["public_mesh_convergence"]
    generator = evidence["hidden_observation_generator"]
    if convergence["criterion_id"] != "modem_public_mesh_convergence_v1":
        raise Comparison2DValidationError("mesh convergence criterion_id is not exact")
    production = _mesh_config_record(
        report["production_candidate"],
        path="production_candidate",
        reference_only=False,
    )
    reference = _mesh_config_record(
        report["next_finer_reference"],
        path="next_finer_reference",
        reference_only=True,
    )
    padding = _mesh_config_record(
        report["padding_perturbation"],
        path="padding_perturbation",
        reference_only=False,
    )
    if (
        production["mesh_config_sha256"] != generator["mesh_artifact_sha256"]
        or reference["mesh_config_sha256"] != convergence["refined_mesh_sha256"]
        or production["mesh_id"] == reference["mesh_id"]
    ):
        raise Comparison2DValidationError(
            "mesh convergence production/reference binding differs from preregistration"
        )
    for key in _MESH_RECORD_KEYS - {
        "mesh_id",
        "base_padding_count_each_side",
    }:
        if padding[key] != production[key]:
            raise Comparison2DValidationError(
                "padding perturbation changes more than the frozen horizontal padding"
            )
    if (
        padding["mesh_id"] != f"{production['mesh_id']}-padding-plus2"
        or padding["base_padding_count_each_side"]
        != production["base_padding_count_each_side"] + 2
    ):
        raise Comparison2DValidationError("padding perturbation is not exact")
    if report["paired_residual_definition"] != {
        "log10_rho": "absolute per station/frequency response difference",
        "phase": "absolute circular-180 per station/frequency response difference",
    }:
        raise Comparison2DValidationError("paired residual definition is not exact")
    if report["frozen_gates"] != _expected_report_gates():
        raise Comparison2DValidationError("mesh convergence frozen gates are not exact")
    if set(residual_archive) != set(residuals) or any(
        not np.array_equal(residual_archive[name], residuals[name]) for name in residuals
    ):
        raise Comparison2DValidationError(
            "pinned residual archive differs from raw forward recomputation"
        )

    ensemble = _mapping(report["public_ensemble"], "public_ensemble")
    _exact_keys(
        ensemble,
        frozenset(
            {
                "selection_policy",
                "sample_count",
                "family_counts",
                "minimum_required",
                "sufficient",
                "source_shards",
                "sample_indices",
            }
        ),
        "public_ensemble",
    )
    sample_count = _integer(
        ensemble["sample_count"], "public_ensemble.sample_count", minimum=25
    )
    sample_values = _sequence(
        ensemble["sample_indices"], "public_ensemble.sample_indices"
    )
    sample_indices = tuple(
        _integer(value, f"public_ensemble.sample_indices[{index}]")
        for index, value in enumerate(sample_values)
    )
    family_counts = _mapping(ensemble["family_counts"], "public_ensemble.family_counts")
    _exact_keys(
        family_counts, frozenset(_CONVERGENCE_FAMILIES), "public_ensemble.family_counts"
    )
    counts = tuple(
        _integer(family_counts[family], f"family_counts.{family}", minimum=5)
        for family in _CONVERGENCE_FAMILIES
    )
    if (
        ensemble["selection_policy"] != "lowest sample_index per frozen scenario"
        or ensemble["minimum_required"] != "25 total and >=5 per family"
        or ensemble["sufficient"] is not True
        or len(sample_indices) != sample_count
        or len(set(sample_indices)) != sample_count
        or sum(counts) != sample_count
        or not np.array_equal(residuals["sample_index"], sample_indices)
    ):
        raise Comparison2DValidationError("public convergence ensemble is inconsistent")
    scenario_indices = residuals["scenario_index"]
    if np.any((scenario_indices < 0) | (scenario_indices >= len(counts))):
        raise Comparison2DValidationError("public convergence scenario ids are invalid")
    if (
        tuple(int(np.count_nonzero(scenario_indices == index)) for index in range(5))
        != counts
    ):
        raise Comparison2DValidationError(
            "public convergence family counts differ from raw data"
        )
    shards = _sequence(ensemble["source_shards"], "public_ensemble.source_shards")
    if not shards:
        raise Comparison2DValidationError("public convergence source shards are missing")
    for index, shard in enumerate(shards):
        shard_path = f"public_ensemble.source_shards[{index}]"
        shard_record = _mapping(shard, shard_path)
        _require_public_path(shard_record.get("path"), f"{shard_path}.path")
        _file_identity_record(shard_record, shard_path)

    reference_arrays = {
        channel: residuals[f"candidate_vs_reference_{channel}"]
        for channel in _CONVERGENCE_CHANNELS
    }
    padding_arrays = {
        channel: residuals[f"candidate_vs_padding_{channel}"]
        for channel in _CONVERGENCE_CHANNELS
    }
    _validate_residual_section(
        report["candidate_vs_reference"],
        path="candidate_vs_reference",
        arrays=reference_arrays,
        padding=False,
    )
    _validate_residual_section(
        report["candidate_vs_padding"],
        path="candidate_vs_padding",
        arrays=padding_arrays,
        padding=True,
    )

    per_geology = _sequence(report["per_geology"], "per_geology")
    if len(per_geology) != sample_count:
        raise Comparison2DValidationError(
            "per_geology does not cover the public ensemble"
        )
    for index, raw_row in enumerate(per_geology):
        path = f"per_geology[{index}]"
        row = _mapping(raw_row, path)
        _exact_keys(
            row,
            frozenset(
                {
                    "sample_index",
                    "scenario",
                    "source_shard_sha256",
                    "candidate_vs_reference",
                    "candidate_vs_padding",
                    "exact_96_te_and_96_tm_finite",
                    "outputs",
                }
            ),
            path,
        )
        if (
            row["sample_index"] != sample_indices[index]
            or row["scenario"] != _CONVERGENCE_FAMILIES[int(scenario_indices[index])]
            or row["exact_96_te_and_96_tm_finite"] is not True
        ):
            raise Comparison2DValidationError(
                f"{path} identity/response contract is wrong"
            )
        _sha256(row["source_shard_sha256"], f"{path}.source_shard_sha256")
        for label, arrays in (
            ("candidate_vs_reference", reference_arrays),
            ("candidate_vs_padding", padding_arrays),
        ):
            summaries = _mapping(row[label], f"{path}.{label}")
            _exact_keys(summaries, frozenset(_CONVERGENCE_CHANNELS), f"{path}.{label}")
            for channel in _CONVERGENCE_CHANNELS:
                _validate_summary(
                    summaries[channel],
                    path=f"{path}.{label}.{channel}",
                    expected=_summary_from_array(arrays[channel][index]),
                    expected_n=96,
                )
        outputs = _mapping(row["outputs"], f"{path}.outputs")
        roles = (
            "production-candidate",
            "next-finer-reference",
            "padding-perturbation",
        )
        _exact_keys(outputs, frozenset(roles), f"{path}.outputs")
        for role in roles:
            output = _mapping(outputs[role], f"{path}.outputs.{role}")
            _exact_keys(
                output,
                frozenset({"directory", "forward", "provenance"}),
                f"{path}.outputs.{role}",
            )
            _string(output["directory"], f"{path}.outputs.{role}.directory")
            _file_identity_record(output["forward"], f"{path}.outputs.{role}.forward")
            _file_identity_record(
                output["provenance"], f"{path}.outputs.{role}.provenance"
            )

    analytic = _mapping(report["analytic_checks"], "analytic_checks")
    _exact_keys(analytic, frozenset({"records", "passed"}), "analytic_checks")
    analytic_records = _sequence(analytic["records"], "analytic_checks.records")
    if len(analytic_records) != 4 or analytic["passed"] is not True:
        raise Comparison2DValidationError(
            "independent 1-D analytic checks are incomplete"
        )
    expected_analytic_pairs = {
        (truth, mesh)
        for truth in ("analytic-halfspace-100", "analytic-layered-100-10-500")
        for mesh in (production["mesh_id"], reference["mesh_id"])
    }
    observed_analytic_pairs: set[tuple[str, str]] = set()
    for index, raw_record in enumerate(analytic_records):
        path = f"analytic_checks.records[{index}]"
        record = _mapping(raw_record, path)
        _exact_keys(
            record,
            frozenset(
                {"truth_id", "mesh_id", "output_dir", "summaries", "gates", "passed"}
            ),
            path,
        )
        pair = (
            _string(record["truth_id"], f"{path}.truth_id"),
            _identifier(record["mesh_id"], f"{path}.mesh_id"),
        )
        observed_analytic_pairs.add(pair)
        if pair not in analytic_residuals:
            raise Comparison2DValidationError(
                f"{path} has no independently recomputed analytic raw response"
            )
        _string(record["output_dir"], f"{path}.output_dir")
        summaries = _mapping(record["summaries"], f"{path}.summaries")
        gates = _mapping(record["gates"], f"{path}.gates")
        _exact_keys(summaries, frozenset(_CONVERGENCE_CHANNELS), f"{path}.summaries")
        _exact_keys(gates, frozenset(_CONVERGENCE_CHANNELS), f"{path}.gates")
        for channel in _CONVERGENCE_CHANNELS:
            summary = _validate_summary(
                summaries[channel],
                path=f"{path}.summaries.{channel}",
                expected=_summary_from_array(analytic_residuals[pair][channel]),
                expected_n=96,
            )
            _validate_gate_record(
                gates[channel],
                path=f"{path}.gates.{channel}",
                summary=summary,
                thresholds=_channel_thresholds(channel, padding=False),
                padding=False,
            )
        if record["passed"] is not True:
            raise Comparison2DValidationError(f"{path} failed")
    if observed_analytic_pairs != expected_analytic_pairs:
        raise Comparison2DValidationError("independent 1-D analytic cases are not exact")

    determinism = _mapping(report["determinism"], "determinism")
    _exact_keys(
        determinism,
        frozenset({"passed", "first_forward", "repeat_forward"}),
        "determinism",
    )
    first_identity = _file_identity_record(
        determinism["first_forward"], "determinism.first_forward"
    )
    repeat_identity = _file_identity_record(
        determinism["repeat_forward"], "determinism.repeat_forward"
    )
    if determinism["passed"] is not True or first_identity != repeat_identity:
        raise Comparison2DValidationError("ModEM public determinism check failed")
    response = _mapping(report["response_contract"], "response_contract")
    _exact_keys(
        response,
        frozenset({"required_rows_per_mode", "all_cases_exact_and_finite"}),
        "response_contract",
    )
    if (
        response["required_rows_per_mode"] != {"TE": 96, "TM": 96}
        or response["all_cases_exact_and_finite"] is not True
    ):
        raise Comparison2DValidationError("response contract is not exact finite TE/TM")

    provenance = _mapping(report["provenance"], "provenance")
    _exact_keys(
        provenance,
        frozenset(
            {"validator_source", "bridge_source", "runtime", "runtime_identity_sha256"}
        ),
        "provenance",
    )
    _file_identity_record(provenance["validator_source"], "provenance.validator_source")
    bridge_identity = _file_identity_record(
        provenance["bridge_source"], "provenance.bridge_source"
    )
    if bridge_identity != (
        generator["converter_sha256"],
        generator["converter_size_bytes"],
    ):
        raise Comparison2DValidationError(
            "convergence bridge differs from pinned converter"
        )
    runtime = _mapping(provenance["runtime"], "provenance.runtime")
    if (
        runtime.get("schema") != "pimsr-modem2d-runtime-provenance"
        or runtime.get("schema_version") != 1
        or _canonical_object_sha256(runtime) != provenance["runtime_identity_sha256"]
    ):
        raise Comparison2DValidationError("ModEM runtime identity is invalid")
    _sha256(provenance["runtime_identity_sha256"], "runtime_identity_sha256")
    modem = _mapping(runtime.get("modem"), "provenance.runtime.modem")
    container = _mapping(runtime.get("container"), "provenance.runtime.container")
    if (
        modem.get("commit") != generator["repository_commit"]
        or modem.get("checkout_clean") is not True
        or container.get("image_id") != generator["container_image_digest"]
        or not isinstance(container.get("reference"), str)
        or not container["reference"].endswith("@" + generator["container_image_digest"])
    ):
        raise Comparison2DValidationError(
            "ModEM source/container provenance differs from pins"
        )
    runtime_artifacts = _mapping(runtime.get("artifacts"), "provenance.runtime.artifacts")
    identities = {
        _file_identity_record(value, f"provenance.runtime.artifacts.{name}")
        for name, value in runtime_artifacts.items()
    }
    required_identities = {
        (generator["source_sha256"], generator["source_size_bytes"]),
        (generator["binary_sha256"], generator["binary_size_bytes"]),
    }
    if not required_identities.issubset(identities):
        raise Comparison2DValidationError(
            "ModEM source/binary artifacts differ from pins"
        )
    raw_record = _mapping(report["raw_paired_residuals"], "raw_paired_residuals")
    if raw_record != {
        "filename": "paired-residuals.npz",
        "sha256": residuals_snapshot.sha256,
        "size_bytes": residuals_snapshot.size_bytes,
    }:
        raise Comparison2DValidationError(
            "raw residual record differs from pinned artifact"
        )


def _convergence_report_valid(
    report: Mapping[str, Any],
    *,
    evidence: Mapping[str, Any],
    residuals: Mapping[str, np.ndarray],
    residual_archive: Mapping[str, np.ndarray],
    analytic_residuals: Mapping[tuple[str, str], Mapping[str, np.ndarray]],
    residuals_snapshot: ArtifactSnapshot,
) -> tuple[bool, str | None]:
    try:
        _validate_report_success(
            report,
            evidence=evidence,
            residuals=residuals,
            residual_archive=residual_archive,
            analytic_residuals=analytic_residuals,
            residuals_snapshot=residuals_snapshot,
        )
    except Comparison2DValidationError as exc:
        return False, str(exc)
    return True, None


def _validate_headline_evidence_files(
    raw_references: Any,
    public_raw_runs: Any,
    *,
    evidence: Mapping[str, Any],
    public_lineage: Mapping[str, Any],
    base: Path,
    seen_identities: set[tuple[int, int]],
) -> tuple[bool, list[str], dict[str, dict[str, Any]]]:
    references = _mapping(raw_references, "post-score evidence_artifacts")
    specs = _evidence_reference_specs(evidence)
    _exact_keys(references, frozenset(specs), "post-score evidence_artifacts")
    identities: dict[str, dict[str, Any]] = {}
    snapshots: dict[str, ArtifactSnapshot] = {}
    try:
        for role, (expected_sha, expected_size) in specs.items():
            reference = _mapping(references[role], f"evidence_artifacts.{role}")
            _exact_keys(reference, _ARTIFACT_REFERENCE_KEYS, f"evidence_artifacts.{role}")
            if (
                reference["sha256"] != expected_sha
                or reference["size_bytes"] != expected_size
            ):
                raise Comparison2DValidationError(
                    f"evidence artifact {role} differs from preregistration"
                )
            snapshot = _artifact_reference(
                reference,
                base=base,
                role=f"evidence_artifacts.{role}",
                seen_identities=seen_identities,
            )
            snapshots[role] = snapshot
            identities[role] = {
                "sha256": snapshot.sha256,
                "size_bytes": snapshot.size_bytes,
            }
        runtime_manifest_snapshot = snapshots["generation_runtime_manifest"]
        if runtime_manifest_snapshot.size_bytes > 1024 * 1024:
            raise Comparison2DValidationError(
                "hidden generation runtime manifest exceeds 1 MiB"
            )
        runtime_manifest = _strict_json(
            runtime_manifest_snapshot,
            "hidden generation runtime manifest",
        )
        identities["generation_runtime_manifest"] = {
            **identities["generation_runtime_manifest"],
            **_validate_generation_runtime_manifest(
                runtime_manifest,
                public_lineage=public_lineage,
            ),
        }
        report_snapshot = snapshots["mesh_convergence_report"]
        report = _strict_json(
            report_snapshot,
            "mesh convergence report",
            require_canonical=False,
        )
        expected_report_payload = (
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        if report_snapshot.payload != expected_report_payload:
            raise Comparison2DValidationError(
                "mesh convergence report is not the deterministic schema-v1 JSON"
            )
        residuals_snapshot = snapshots["mesh_convergence_residuals"]
        residual_archive = _load_convergence_residuals(
            residuals_snapshot,
            report=report,
        )
        raw_validation = _validate_public_convergence_raw_runs(
            public_raw_runs,
            report=report,
            evidence=evidence,
            public_lineage=public_lineage,
            base=base,
            seen_identities=seen_identities,
        )
        identities["public_convergence_raw_runs"] = {
            "artifact_count": raw_validation.artifact_count,
            "identities_sha256": raw_validation.identities_sha256,
        }
        valid, reason = _convergence_report_valid(
            report,
            evidence=evidence,
            residuals=raw_validation.residuals,
            residual_archive=residual_archive,
            analytic_residuals=raw_validation.analytic_residuals,
            residuals_snapshot=residuals_snapshot,
        )
        return valid, ([] if reason is None else [reason]), identities
    except Comparison2DValidationError as exc:
        return False, [str(exc)], identities


def _artifact_digest_record(
    value: Any,
    *,
    path: str,
    schema: str,
    schema_version: int,
    require_size: bool,
) -> tuple[str, int | None]:
    record = _mapping(value, path)
    if record.get("schema") != schema or record.get("schema_version") != schema_version:
        raise Comparison2DValidationError(f"{path} schema identity is wrong")
    digest = _sha256(record.get("sha256"), f"{path}.sha256")
    size: int | None = None
    if require_size:
        size = _integer(record.get("size_bytes"), f"{path}.size_bytes", minimum=1)
    return digest, size


def _family_commitment_digest(
    *, campaign_id: str, nonce_hex: str, rows: Sequence[Mapping[str, Any]]
) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", nonce_hex, flags=re.ASCII):
        raise Comparison2DValidationError(
            "family partition nonce must be 32 lowercase-hex bytes"
        )
    reveal_body = {
        "schema": "pimsr-sota-2d-family-partition-reveal-body",
        "schema_version": 1,
        "campaign_id": campaign_id,
        "rows": [dict(row) for row in rows],
    }
    digest = hashlib.sha256()
    digest.update(_FAMILY_COMMITMENT_DOMAIN)
    digest.update(bytes.fromhex(nonce_hex))
    digest.update(_json_bytes(reveal_body, publication=False))
    return digest.hexdigest()


def _public_observation_family_commitment(
    snapshot: ArtifactSnapshot,
    *,
    campaign_id: str,
    expected_observations_sha256: str,
    expected_contract: Mapping[str, Any],
) -> tuple[str, int]:
    value = _strict_json(snapshot, "public observation manifest")
    _exact_keys(
        value,
        frozenset(
            {
                "audience",
                "declared_evaluation_floors",
                "observation_payload",
                "physical_contract",
                "sample_count",
                "schema",
                "schema_version",
                "split_id",
                "family_partition_commitment",
            }
        ),
        "public observation manifest",
    )
    if (
        value["audience"] != "method_input_public"
        or value["schema"] != "pimsr-sota-2d-observation-manifest"
        or value["schema_version"] != 3
        or value["split_id"] != campaign_id
        or value["sample_count"] != SAMPLES_PER_CAMPAIGN
    ):
        raise Comparison2DValidationError("public observation manifest identity is wrong")
    commitment = _mapping(
        value["family_partition_commitment"], "family_partition_commitment"
    )
    payload = _mapping(value["observation_payload"], "observation_payload")
    if (
        payload.get("schema") != "pimsr-sota-2d-observations"
        or payload.get("schema_version") != 1
        or _sha256(payload.get("sha256"), "observation_payload.sha256")
        != expected_observations_sha256
    ):
        raise Comparison2DValidationError(
            "public observation payload differs from the pre-score lock"
        )
    payload_size = _integer(
        payload.get("size_bytes"), "observation_payload.size_bytes", minimum=1
    )
    _exact_keys(
        commitment,
        frozenset({"schema", "schema_version", "sha256", "contract"}),
        "family_partition_commitment",
    )
    digest = _sha256(commitment["sha256"], "family_partition_commitment.sha256")
    if (
        commitment["schema"] != _FAMILY_COMMITMENT_SCHEMA
        or commitment["schema_version"] != _FAMILY_COMMITMENT_SCHEMA_VERSION
        or commitment["contract"] != expected_contract
    ):
        raise Comparison2DValidationError(
            "public observation family commitment contract differs from preregistration"
        )
    return digest, payload_size


def _validate_family_partition_reveal(
    value: Any,
    *,
    campaign_id: str,
    expected_commitment_sha256: str,
    expected_rows: Sequence[Mapping[str, Any]],
) -> None:
    reveal = _mapping(value, "operator split.family_partition_reveal")
    _exact_keys(
        reveal,
        frozenset({"schema", "schema_version", "campaign_id", "nonce_hex", "rows"}),
        "family_partition_reveal",
    )
    if (
        reveal["schema"] != _FAMILY_REVEAL_SCHEMA
        or reveal["schema_version"] != _FAMILY_REVEAL_SCHEMA_VERSION
        or reveal["campaign_id"] != campaign_id
    ):
        raise Comparison2DValidationError("family partition reveal identity is wrong")
    raw_rows = _sequence(reveal["rows"], "family_partition_reveal.rows")
    rows: list[dict[str, Any]] = []
    for index, raw_row in enumerate(raw_rows):
        path = f"family_partition_reveal.rows[{index}]"
        row = _mapping(raw_row, path)
        _exact_keys(
            row,
            frozenset({"sample_index", "base_model_id", "family_id", "noise_index"}),
            path,
        )
        rows.append(
            {
                "sample_index": _integer(row["sample_index"], f"{path}.sample_index"),
                "base_model_id": _string(row["base_model_id"], f"{path}.base_model_id"),
                "family_id": _string(row["family_id"], f"{path}.family_id"),
                "noise_index": _integer(row["noise_index"], f"{path}.noise_index"),
            }
        )
    if rows != [dict(row) for row in expected_rows]:
        raise Comparison2DValidationError(
            "family partition reveal differs from the exact operator hierarchy"
        )
    digest = _family_commitment_digest(
        campaign_id=campaign_id,
        nonce_hex=_string(reveal["nonce_hex"], "family_partition_reveal.nonce_hex"),
        rows=rows,
    )
    if digest != expected_commitment_sha256:
        raise Comparison2DValidationError(
            "family partition reveal does not open its commitment"
        )


def _validate_runtime_against_generator(
    runtime: Mapping[str, Any],
    *,
    runtime_identity_sha256: Any,
    generator: Mapping[str, Any],
    path: str,
) -> None:
    if (
        runtime.get("schema") != "pimsr-modem2d-runtime-provenance"
        or runtime.get("schema_version") != 1
        or _canonical_object_sha256(runtime)
        != _sha256(runtime_identity_sha256, f"{path}.runtime_identity_sha256")
    ):
        raise Comparison2DValidationError(f"{path} runtime identity is invalid")
    modem = _mapping(runtime.get("modem"), f"{path}.runtime.modem")
    container = _mapping(runtime.get("container"), f"{path}.runtime.container")
    if (
        modem.get("commit") != generator["repository_commit"]
        or modem.get("checkout_clean") is not True
        or container.get("image_id") != generator["container_image_digest"]
        or not isinstance(container.get("reference"), str)
        or not container["reference"].endswith("@" + generator["container_image_digest"])
    ):
        raise Comparison2DValidationError(
            f"{path} ModEM source/container differs from pins"
        )
    runtime_artifacts = _mapping(runtime.get("artifacts"), f"{path}.runtime.artifacts")
    identities = {
        _file_identity_record(value, f"{path}.runtime.artifacts.{name}")
        for name, value in runtime_artifacts.items()
    }
    required = {
        (generator["source_sha256"], generator["source_size_bytes"]),
        (generator["binary_sha256"], generator["binary_size_bytes"]),
    }
    if not required.issubset(identities):
        raise Comparison2DValidationError(
            f"{path} ModEM source/binary artifacts differ from pins"
        )


def _validate_generator_distinct_from_lineage(
    generator: Mapping[str, Any], lineage: Mapping[str, Any]
) -> None:
    for split in ("train", "validation"):
        for repository_name, repository in lineage[split]["repositories"].items():
            if generator["repository_commit"] == repository["commit"] or generator[
                "source_sha256"
            ] in set(repository["source_hashes"].values()):
                raise Comparison2DValidationError(
                    "ModEM scoring source is not materially distinct from "
                    f"{split} {repository_name} lineage"
                )


def _array_identity(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _response_identity(response: _RawModEMResponse) -> str:
    return _canonical_object_sha256(
        {
            "schema": "pimsr-modem2d-clean-response-identity",
            "schema_version": 1,
            "arrays": {
                "frequency_hz": _array_identity(response.frequencies_hz.astype("<f8")),
                "station_x_m": _array_identity(response.station_y_m.astype("<f8")),
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
    *, observed: Mapping[str, np.ndarray], clean: _RawModEMResponse
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
        raise Comparison2DValidationError(
            "hidden observation noise deltas are non-finite"
        )
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
    parameters = _LINEAGE_SENSOR_PARAMETERS["sensor_model"]
    rho = np.asarray(rho_app, dtype=np.float64)
    phase = np.asarray(phase_degrees, dtype=np.float64)
    periods = np.asarray(periods_seconds, dtype=np.float64)
    relative = np.full(rho.shape, parameters["mt_rel_floor"], dtype=np.float64)
    relative += parameters["mt_dead_band_extra"] * ((periods >= 0.1) & (periods <= 10.0))
    rho_noisy = rho * np.exp(rng.normal(0.0, relative))
    phase_noisy = phase + rng.normal(0.0, parameters["mt_phase_floor_deg"], phase.shape)
    upper = parameters["distort_log10rho_hi"] if distort_hi is None else distort_hi
    if upper > 0.0:
        lower = min(parameters["distort_log10rho_lo"], upper)
        amplitude = float(np.exp(rng.uniform(np.log(lower), np.log(upper))))
        rho_noisy *= 10.0 ** (
            amplitude * _sensor_ar1_curve(rho.size, parameters["distort_lag1"], rng)
        )
        phase_noisy += (
            parameters["distort_phase_scale"]
            * amplitude
            * _sensor_ar1_curve(phase.size, parameters["distort_lag1"], rng)
        )
    sigma = parameters["static_shift_sigma"] if shift_sigma is None else shift_sigma
    rho_noisy *= 10.0 ** rng.normal(0.0, sigma)
    return rho_noisy, _fold_phase_float32_safe(phase_noisy)


def _expected_hidden_noisy_response(
    clean: _RawModEMResponse,
    *,
    generator_seed: int,
    base_index: int,
    noise_index: int,
) -> Mapping[str, np.ndarray]:
    rng = np.random.default_rng([generator_seed, 3, base_index, noise_index])
    tm_shift_sigma = float(rng.uniform(0.15, 0.32))
    tm_distort_hi = float(np.exp(rng.uniform(np.log(0.25), np.log(0.45))))
    periods = 1.0 / clean.frequencies_hz
    rho_te = np.power(10.0, clean.log10_rho_te)
    rho_tm = np.power(10.0, clean.log10_rho_tm)
    noisy_rho_te = np.empty_like(rho_te)
    noisy_phase_te = np.empty_like(clean.phase_te_deg)
    noisy_rho_tm = np.empty_like(rho_tm)
    noisy_phase_tm = np.empty_like(clean.phase_tm_deg)
    for station_index in range(clean.station_y_m.size):
        noisy_rho_te[:, station_index], noisy_phase_te[:, station_index] = (
            _apply_frozen_mt_noise(
                rho_te[:, station_index],
                clean.phase_te_deg[:, station_index],
                periods,
                rng,
            )
        )
        noisy_rho_tm[:, station_index], noisy_phase_tm[:, station_index] = (
            _apply_frozen_mt_noise(
                rho_tm[:, station_index],
                clean.phase_tm_deg[:, station_index],
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


def _validate_hidden_noise_realization(
    observed: Mapping[str, np.ndarray],
    clean: _RawModEMResponse,
    *,
    generator_seed: int,
    base_index: int,
    noise_index: int,
    path: str,
) -> None:
    expected = _expected_hidden_noisy_response(
        clean,
        generator_seed=generator_seed,
        base_index=base_index,
        noise_index=noise_index,
    )
    if any(
        not np.array_equal(observed[name], expected_values)
        for name, expected_values in expected.items()
    ):
        raise Comparison2DValidationError(
            f"{path} observations differ from the frozen seeded noise realization"
        )


def _load_hidden_observation_rows(
    snapshot: ArtifactSnapshot,
    *,
    campaign: _Campaign,
    material_truth: _MaterialTruth2D | None = None,
) -> Mapping[int, Mapping[str, np.ndarray]]:
    if snapshot.size_bytes > 64 * 1024 * 1024:
        raise Comparison2DValidationError("hidden observation payload exceeds 64 MiB")
    expected_names = tuple(f"{name}.npy" for name in _OBSERVATION_ARRAY_MEMBERS)
    try:
        with zipfile.ZipFile(io.BytesIO(snapshot.payload), mode="r") as archive:
            infos = archive.infolist()
            if (
                tuple(info.filename for info in infos) != expected_names
                or archive.comment
                or archive.testzip() is not None
                or any(
                    info.compress_type != zipfile.ZIP_DEFLATED
                    or info.flag_bits & 0x1
                    or info.file_size <= 0
                    for info in infos
                )
                or sum(info.file_size for info in infos) > 32 * 1024 * 1024
            ):
                raise Comparison2DValidationError(
                    "hidden observation NPZ member/compression contract is invalid"
                )
        with np.load(
            io.BytesIO(snapshot.payload), allow_pickle=False, max_header_size=4_096
        ) as archive:
            if tuple(archive.files) != _OBSERVATION_ARRAY_MEMBERS:
                raise Comparison2DValidationError(
                    "hidden observation NPZ array schema/order is not exact"
                )
            values = {name: np.array(archive[name], copy=True) for name in archive.files}
    except Comparison2DValidationError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise Comparison2DValidationError(
            f"cannot decode hidden observation payload: {exc}"
        ) from exc
    if (
        values["schema"].shape != ()
        or values["schema"].item() != "pimsr-sota-2d-observations"
        or values["schema_version"].shape != ()
        or values["schema_version"].dtype != np.dtype("<i8")
        or int(values["schema_version"].item()) != 1
    ):
        raise Comparison2DValidationError("hidden observation payload identity is wrong")
    sample_index = values["sample_index"]
    frequency = values["frequency_hz"]
    station = values["station_x_m"]
    x_centres = values["x_cell_centers_m"]
    depth_centres = values["depth_cell_centers_m"]
    channel_order = values["observation_channel_order"]
    if (
        sample_index.dtype != np.dtype("<i8")
        or sample_index.shape != (SAMPLES_PER_CAMPAIGN,)
        or len({int(item) for item in sample_index}) != SAMPLES_PER_CAMPAIGN
        or {int(item) for item in sample_index} != set(campaign.hierarchy.sample_ids)
        or frequency.dtype != np.dtype("<f8")
        or frequency.shape != (8,)
        or not np.isfinite(frequency).all()
        or np.any(frequency <= 0.0)
        or np.any(np.diff(frequency) <= 0.0)
        or station.dtype != np.dtype("<f8")
        or station.shape != (12,)
        or not np.isfinite(station).all()
        or np.any(np.diff(station) <= 0.0)
        or x_centres.dtype != np.dtype("<f8")
        or x_centres.shape != (CANONICAL_MODEL_SHAPE[1],)
        or not np.isfinite(x_centres).all()
        or np.any(np.diff(x_centres) <= 0.0)
        or depth_centres.dtype != np.dtype("<f8")
        or depth_centres.shape != (CANONICAL_MODEL_SHAPE[0],)
        or not np.isfinite(depth_centres).all()
        or np.any(depth_centres <= 0.0)
        or np.any(np.diff(depth_centres) <= 0.0)
        or station[0] < x_centres[0]
        or station[-1] > x_centres[-1]
        or channel_order.tolist()
        != [
            "log10_rho_te",
            "phase_te_degrees",
            "log10_rho_tm",
            "phase_tm_degrees",
        ]
    ):
        raise Comparison2DValidationError("hidden observation axes/ids are not exact")
    if material_truth is not None and (
        not np.array_equal(x_centres, material_truth.x_cell_centers_m)
        or not np.array_equal(depth_centres, material_truth.depth_cell_centers_m)
    ):
        raise Comparison2DValidationError(
            "hidden observation raster axes differ from withheld truth material"
        )
    observed_names = (
        "observed_log10_rho_te",
        "observed_phase_te_degrees",
        "observed_log10_rho_tm",
        "observed_phase_tm_degrees",
    )
    if any(
        values[name].dtype != np.dtype("<f4")
        or values[name].shape != (SAMPLES_PER_CAMPAIGN, 8, 12)
        or not np.isfinite(values[name]).all()
        for name in observed_names
    ):
        raise Comparison2DValidationError(
            "hidden observation response arrays are invalid"
        )
    for name in (
        "observed_phase_te_degrees",
        "observed_phase_tm_degrees",
    ):
        if np.any((values[name] < 0.0) | (values[name] >= 180.0)):
            raise Comparison2DValidationError(
                "hidden observation phases violate the [0,180) contract"
            )
    floor_names = (
        "declared_evaluation_floor_log10_rho_te",
        "declared_evaluation_floor_phase_te_degrees",
        "declared_evaluation_floor_log10_rho_tm",
        "declared_evaluation_floor_phase_tm_degrees",
    )
    if any(
        values[name].dtype != np.dtype("<f4")
        or values[name].shape != (SAMPLES_PER_CAMPAIGN, 8, 12)
        or not np.isfinite(values[name]).all()
        or np.any(values[name] <= 0.0)
        for name in floor_names
    ):
        raise Comparison2DValidationError(
            "hidden observation declared evaluation floors are invalid"
        )
    valid = values["valid_mask"]
    if (
        valid.dtype != np.dtype("bool")
        or valid.shape != (SAMPLES_PER_CAMPAIGN, 4, 8, 12)
        or not np.all(valid)
    ):
        raise Comparison2DValidationError("hidden observation valid mask is not exact")
    result: dict[int, Mapping[str, np.ndarray]] = {}
    for row_index, opaque_id in enumerate(sample_index):
        result[int(opaque_id)] = {
            "frequency_hz": frequency,
            "station_x_m": station,
            **{name: values[name][row_index] for name in observed_names},
            "valid_mask": valid[row_index],
        }
    return result


def _hidden_source_lineage_identity(
    public_lineage: Mapping[str, Any],
) -> Mapping[str, Any]:
    repositories = _mapping(
        _mapping(public_lineage.get("train"), "train public lineage identity").get(
            "repositories"
        ),
        "train public lineage repositories",
    )
    forward = _mapping(repositories.get("pimsr_forward"), "pimsr_forward lineage")
    geogen = _mapping(repositories.get("pimsr_geogen"), "pimsr_geogen lineage")
    forward_hashes = _mapping(
        forward.get("source_hashes"), "pimsr_forward lineage source_hashes"
    )
    geogen_hashes = _mapping(
        geogen.get("source_hashes"), "pimsr_geogen lineage source_hashes"
    )
    return {
        "pimsr_forward": {
            "repository_commit": _git_commit(
                forward.get("commit"), "pimsr_forward lineage commit"
            ),
            "dataset2d_source_sha256": _sha256(
                forward_hashes.get("src/pimsr_forward/dataset2d.py"),
                "pimsr_forward dataset2d source SHA-256",
            ),
            "sensors_source_sha256": _sha256(
                forward_hashes.get("src/pimsr_forward/sensors.py"),
                "pimsr_forward sensors source SHA-256",
            ),
        },
        "pimsr_geogen": {
            "repository_commit": _git_commit(
                geogen.get("commit"), "pimsr_geogen lineage commit"
            ),
            "generator_source_sha256": _sha256(
                geogen_hashes.get("src/pimsr_geogen/generator.py"),
                "pimsr_geogen generator source SHA-256",
            ),
            "model_source_sha256": _sha256(
                geogen_hashes.get("src/pimsr_geogen/model.py"),
                "pimsr_geogen model source SHA-256",
            ),
            "rock_physics_source_sha256": _sha256(
                geogen_hashes.get("src/pimsr_geogen/rock_physics.py"),
                "pimsr_geogen rock_physics source SHA-256",
            ),
            "section2d_source_sha256": _sha256(
                geogen_hashes.get("src/pimsr_geogen/section2d.py"),
                "pimsr_geogen section2d source SHA-256",
            ),
        },
    }


def _hidden_generation_contract(
    value: Any, *, expected_source_lineage: Mapping[str, Any]
) -> tuple[Mapping[str, Any], str]:
    contract = _mapping(value, "hidden_generation.generation_contract")
    _exact_keys(
        contract,
        frozenset(
            {
                "schema",
                "schema_version",
                "generator_seed",
                "base_layer_rng",
                "base_layer_scenario",
                "section_rng",
                "scenario_policy",
                "noise_rng",
                "geology_contract",
                "clean_forward_contract",
                "noise_contract",
                "source_lineage",
                "base_count",
                "noise_realizations_per_base",
            }
        ),
        "hidden_generation.generation_contract",
    )
    generator_seed = _integer(
        contract["generator_seed"], "hidden_generation.generator_seed"
    )
    if generator_seed > np.iinfo(np.int64).max:
        raise Comparison2DValidationError("hidden generator seed exceeds int64")
    expected = {
        "schema": _HIDDEN_GENERATION_CONTRACT_SCHEMA,
        "schema_version": _HIDDEN_GENERATION_CONTRACT_SCHEMA_VERSION,
        "generator_seed": generator_seed,
        "base_layer_rng": _HIDDEN_BASE_LAYER_RNG,
        "base_layer_scenario": _HIDDEN_BASE_LAYER_SCENARIO,
        "section_rng": _HIDDEN_SECTION_RNG,
        "scenario_policy": _HIDDEN_SCENARIO_POLICY,
        "noise_rng": _HIDDEN_NOISE_RNG,
        "geology_contract": "pimsr-geogen.SectionGenerator/default-grid/v1",
        "clean_forward_contract": "pinned_modem_2d_raw_forward_per_unique_base/v1",
        "noise_contract": "pimsr-forward.SensorModel/mt-noise+tm-severity-v5/v1",
        "source_lineage": dict(expected_source_lineage),
        "base_count": BASE_MODELS_PER_CAMPAIGN,
        "noise_realizations_per_base": NOISE_REALIZATIONS_PER_BASE,
    }
    if dict(contract) != expected:
        raise Comparison2DValidationError(
            "hidden generation/RNG contract is not the frozen material protocol"
        )
    return expected, _canonical_object_sha256(expected)


def _hidden_base_truth_source(
    value: Mapping[str, Any],
    *,
    path: str,
    campaign: _Campaign,
    base_model_id: str,
    family: str,
    base_index: int,
    source_sample_indices: Sequence[int],
    generator_seed: int,
    generation_contract_sha256: str,
) -> None:
    expected = {
        "schema": "pimsr-modem2d-hidden-base-generation-source",
        "schema_version": 2,
        "campaign_id": campaign.campaign_id,
        "operator_manifest_sha256": campaign.operator_sha256,
        "base_model_id": base_model_id,
        "family_id": family,
        "base_index": base_index,
        "generator_seed": generator_seed,
        "base_layer_rng_key": [generator_seed, base_index],
        "section_rng_key": [generator_seed, 2, base_index],
        "generation_contract_sha256": generation_contract_sha256,
        "source_generator_sample_indices": list(source_sample_indices),
        "observations_sha256": campaign.observations_sha256,
        "public_observation_manifest_sha256": campaign.observation_manifest_sha256,
        "withheld_truth_sha256": campaign.truth_sha256,
        "family_partition_commitment_sha256": campaign.family_commitment_sha256,
    }
    _exact_keys(value, frozenset(expected), path)
    if dict(value) != expected:
        raise Comparison2DValidationError(f"{path} hidden base-source binding is wrong")


def _nested_mesh_from_record(value: Mapping[str, Any]) -> NestedMeshConfig:
    fields = {
        name: value[name]
        for name in (
            "mesh_id",
            "version",
            "base_core_width_m",
            "base_core_count",
            "base_padding_count_each_side",
            "base_padding_growth",
            "minimum_vertical_subdivisions",
            "maximum_base_dz_m",
            "deep_padding_growth",
            "maximum_deep_macro_dz_m",
            "minimum_depth_m",
            "horizontal_refinement_factor",
            "vertical_refinement_factor",
        )
    }
    try:
        mesh = NestedMeshConfig(**fields)
    except (TypeError, ValueError) as exc:
        raise Comparison2DValidationError(
            f"cannot reconstruct protected ModEM mesh: {exc}"
        ) from exc
    if (
        any(mesh.canonical_record()[key] != value[key] for key in _MESH_RECORD_KEYS)
        or mesh.sha256 != value["mesh_config_sha256"]
    ):
        raise Comparison2DValidationError(
            "reconstructed protected ModEM mesh differs from its material record"
        )
    return mesh


def _render_modem_input_bytes(
    truth: CanonicalTruth,
    mesh: NestedMeshConfig,
) -> tuple[bytes, bytes]:
    """Reproduce protected ModEM writer bytes without opening another path."""
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
    template_payload = ("\n".join(template_lines) + "\n").encode("ascii")
    return model_payload, template_payload


def _replay_hidden_base_truths(
    *,
    generator_seed: int,
    material_truth: _MaterialTruth2D,
    expected_bases: Sequence[tuple[str, str, tuple[int, ...]]],
    section_generator_factory: Callable[[int], Any] | None = None,
) -> Mapping[str, tuple[np.ndarray, bool]]:
    """Regenerate every forced-family base and require byte-exact geology.

    Distinctness and consistency with the raw ModEM input are insufficient: an
    operator could otherwise substitute any 100 distinct grids.  The revealed
    seed and pinned generation runtime must reproduce the withheld truth itself.
    """
    if section_generator_factory is None:
        from pimsr_geogen.model import DEFAULT_DEPTH_GRID
        from pimsr_geogen.section2d import DEFAULT_X_GRID, SectionGenerator

        if not np.array_equal(
            material_truth.x_cell_centers_m, np.asarray(DEFAULT_X_GRID, dtype="<f8")
        ) or not np.array_equal(
            material_truth.depth_cell_centers_m,
            np.asarray(DEFAULT_DEPTH_GRID, dtype="<f8"),
        ):
            raise Comparison2DValidationError(
                "withheld truth axes differ from the pinned SectionGenerator defaults"
            )
        section_generator_factory = SectionGenerator
    generator = section_generator_factory(generator_seed)
    replayed: dict[str, tuple[np.ndarray, bool]] = {}
    for base_index, (family, base_model_id, _source_indices) in enumerate(
        expected_bases
    ):
        section = generator.sample(base_index, scenario=family)
        if str(section.scenario) != family or int(section.seed) != base_index:
            raise Comparison2DValidationError(
                "pinned SectionGenerator did not reproduce the forced-family plan"
            )
        if not np.array_equal(
            np.asarray(section.x_grid, dtype="<f8"), material_truth.x_cell_centers_m
        ) or not np.array_equal(
            np.asarray(section.depth_grid, dtype="<f8"),
            material_truth.depth_cell_centers_m,
        ):
            raise Comparison2DValidationError(
                "replayed hidden geology axes differ from withheld truth"
            )
        grid = np.ascontiguousarray(section.log10_res, dtype="<f4")
        if grid.shape != CANONICAL_MODEL_SHAPE or not np.isfinite(grid).all():
            raise Comparison2DValidationError("replayed hidden geology is invalid")
        replayed[base_model_id] = (grid, bool(section.has_fault))
    if len(replayed) != BASE_MODELS_PER_CAMPAIGN:
        raise Comparison2DValidationError(
            "pinned SectionGenerator did not replay exactly 100 hidden bases"
        )
    return replayed


def _material_hidden_base_truths(
    *,
    generator_seed: int,
    material_truth: _MaterialTruth2D,
    campaign: _Campaign,
    expected_bases: Sequence[tuple[str, str, tuple[int, ...]]],
    sample_ids_by_base: Mapping[str, tuple[int, ...]],
    observation_rows: Mapping[int, Mapping[str, np.ndarray]],
    section_generator_factory: Callable[[int], Any] | None = None,
) -> Mapping[str, CanonicalTruth]:
    truth_position = {
        int(sample_id): index
        for index, sample_id in enumerate(material_truth.sample_index)
    }
    if material_truth.observations_sha256 != campaign.observations_sha256 or set(
        truth_position
    ) != set(campaign.hierarchy.sample_ids):
        raise Comparison2DValidationError(
            "hidden truth material differs from the locked campaign hierarchy"
        )
    replayed = _replay_hidden_base_truths(
        generator_seed=generator_seed,
        material_truth=material_truth,
        expected_bases=expected_bases,
        section_generator_factory=section_generator_factory,
    )
    base_truths: dict[str, CanonicalTruth] = {}
    geology_identities: set[str] = set()
    for family, base_model_id, _source_indices in expected_bases:
        sample_ids = sample_ids_by_base[base_model_id]
        row_positions = [truth_position[sample_id] for sample_id in sample_ids]
        rows = material_truth.log10_resistivity[row_positions]
        if any(
            rows[index].tobytes(order="C") != rows[0].tobytes(order="C")
            for index in range(1, NOISE_REALIZATIONS_PER_BASE)
        ):
            raise Comparison2DValidationError(
                "withheld truth does not repeat one byte-identical geology for five noise rows"
            )
        if any(
            str(material_truth.scenario[position]) != family for position in row_positions
        ):
            raise Comparison2DValidationError(
                "withheld truth family labels differ from the operator hierarchy"
            )
        replayed_grid, replayed_has_fault = replayed[base_model_id]
        if rows[0].tobytes(order="C") != replayed_grid.tobytes(order="C"):
            raise Comparison2DValidationError(
                "withheld truth is not the byte-exact revealed-seed SectionGenerator replay"
            )
        if any(
            bool(material_truth.has_fault[position]) != replayed_has_fault
            for position in row_positions
        ):
            raise Comparison2DValidationError(
                "withheld fault labels differ from the revealed-seed generator replay"
            )
        geology_sha = hashlib.sha256(rows[0].tobytes(order="C")).hexdigest()
        if geology_sha in geology_identities:
            raise Comparison2DValidationError(
                "withheld truth reuses geology across distinct hidden bases"
            )
        geology_identities.add(geology_sha)
        observed = observation_rows[sample_ids[0]]
        base_truths[base_model_id] = CanonicalTruth(
            log10_resistivity=rows[0],
            x_centres_m=material_truth.x_cell_centers_m,
            depth_centres_m=material_truth.depth_cell_centers_m,
            frequencies_hz=observed["frequency_hz"],
            station_x_m=observed["station_x_m"],
            sample_id=base_model_id,
        )
    if len(base_truths) != BASE_MODELS_PER_CAMPAIGN:
        raise Comparison2DValidationError(
            "hidden truth material does not prove exactly 100 distinct bases"
        )
    return base_truths


def _validate_hidden_generation_closure(
    value: Any,
    *,
    campaign: _Campaign,
    material_truth: _MaterialTruth2D,
    evidence: Mapping[str, Any],
    public_lineage: Mapping[str, Any],
    expected_generation_runtime_manifest: Mapping[str, Any],
    expected_observation_payload_size: int,
    base: Path,
    seen_identities: set[tuple[int, int]],
) -> tuple[_Campaign, Mapping[str, Any]]:
    closure = _mapping(value, f"{campaign.campaign_id} hidden_generation")
    _exact_keys(
        closure,
        frozenset(
            {
                "schema",
                "schema_version",
                "campaign_id",
                "mesh",
                "runtime",
                "runtime_identity_sha256",
                "bindings",
                "generation_contract",
                "generation_runtime",
                "generation_runtime_manifest",
                "observation_payload",
                "base_forward_runs",
                "noise_rows",
            }
        ),
        "hidden_generation",
    )
    if (
        closure["schema"] != _HIDDEN_GENERATION_SCHEMA
        or closure["schema_version"] != _HIDDEN_GENERATION_SCHEMA_VERSION
        or closure["campaign_id"] != campaign.campaign_id
    ):
        raise Comparison2DValidationError("hidden generation closure identity is wrong")
    bindings = _mapping(closure["bindings"], "hidden_generation.bindings")
    expected_bindings = {
        "operator_manifest_sha256": campaign.operator_sha256,
        "observations_sha256": campaign.observations_sha256,
        "public_observation_manifest_sha256": campaign.observation_manifest_sha256,
        "withheld_truth_sha256": campaign.truth_sha256,
        "family_partition_commitment_sha256": campaign.family_commitment_sha256,
    }
    _exact_keys(bindings, frozenset(expected_bindings), "hidden_generation.bindings")
    if dict(bindings) != expected_bindings:
        raise Comparison2DValidationError(
            "hidden generation bindings differ from the locked operator campaign"
        )
    source_lineage = _hidden_source_lineage_identity(public_lineage)
    generation_contract, generation_contract_sha256 = _hidden_generation_contract(
        closure["generation_contract"], expected_source_lineage=source_lineage
    )
    generator_seed = int(generation_contract["generator_seed"])
    runtime_manifest_reference = _mapping(
        closure["generation_runtime_manifest"],
        "hidden_generation.generation_runtime_manifest",
    )
    _file_identity_record(
        runtime_manifest_reference,
        "hidden_generation.generation_runtime_manifest",
    )
    expected_runtime_manifest_reference = _mapping(
        expected_generation_runtime_manifest,
        "evidence_artifacts.generation_runtime_manifest",
    )
    _file_identity_record(
        expected_runtime_manifest_reference,
        "evidence_artifacts.generation_runtime_manifest",
    )
    # The manifest was already opened and validated as headline evidence.  Requiring
    # the complete reference here binds every campaign to that exact captured inode
    # without reopening an attacker-controlled alias after validation.
    if dict(runtime_manifest_reference) != dict(expected_runtime_manifest_reference):
        raise Comparison2DValidationError(
            "hidden generation runtime manifest differs from captured preregistered evidence"
        )
    base_runs = _sequence(
        closure["base_forward_runs"], "hidden_generation.base_forward_runs"
    )
    raw_noise_records = _sequence(closure["noise_rows"], "hidden_generation.noise_rows")
    if len(base_runs) != BASE_MODELS_PER_CAMPAIGN:
        raise Comparison2DValidationError(
            "hidden generation requires exactly 100 clean ModEM base solves"
        )
    if len(raw_noise_records) != SAMPLES_PER_CAMPAIGN:
        raise Comparison2DValidationError(
            "hidden generation requires exactly 500 noise-row bindings"
        )
    generator = evidence["hidden_observation_generator"]
    generation_runtime = _mapping(
        closure["generation_runtime"], "hidden_generation.generation_runtime"
    )
    _exact_keys(
        generation_runtime,
        frozenset(_HIDDEN_GENERATION_RUNTIME),
        "hidden_generation.generation_runtime",
    )
    if dict(generation_runtime) != dict(generator["generation_runtime"]):
        raise Comparison2DValidationError(
            "hidden generation runtime differs from the preregistered replay runtime"
        )
    analytic_contract = _analytic_1d_contract(
        evidence["public_mesh_convergence"]["analytic_1d_contract"]
    )
    mesh = _mesh_config_record(
        closure["mesh"], path="hidden_generation.mesh", reference_only=False
    )
    if mesh["mesh_config_sha256"] != generator["mesh_artifact_sha256"]:
        raise Comparison2DValidationError(
            "hidden generation mesh differs from preregistration"
        )
    runtime = _mapping(closure["runtime"], "hidden_generation.runtime")
    _validate_runtime_against_generator(
        runtime,
        runtime_identity_sha256=closure["runtime_identity_sha256"],
        generator=generator,
        path="hidden_generation",
    )
    observation_snapshot = _artifact_reference(
        closure["observation_payload"],
        base=base,
        role="hidden_generation.observation_payload",
        seen_identities=seen_identities,
    )
    if (
        observation_snapshot.sha256 != campaign.observations_sha256
        or observation_snapshot.size_bytes != expected_observation_payload_size
    ):
        raise Comparison2DValidationError(
            "hidden generation observation payload differs from the lock/manifest"
        )
    observation_rows = _load_hidden_observation_rows(
        observation_snapshot,
        campaign=campaign,
        material_truth=material_truth,
    )

    expected_bases: list[tuple[str, str, tuple[int, ...]]] = []
    sample_ids_by_base: dict[str, tuple[int, ...]] = {}
    coordinates: dict[int, tuple[str, str, int, int, int]] = {}
    for family in campaign.hierarchy.families:
        for base_model_id, base_noise_rows in campaign.hierarchy.tree[family]:
            ordered_noise = sorted(base_noise_rows)
            source_indices = tuple(
                campaign.source_sample_ids[sample_id]
                for _noise_id, sample_id in ordered_noise
            )
            base_index = len(expected_bases)
            expected_bases.append((family, base_model_id, source_indices))
            sample_ids_by_base[base_model_id] = tuple(
                sample_id for _noise_id, sample_id in ordered_noise
            )
            for noise_id, sample_id in ordered_noise:
                noise_index = int(noise_id.removeprefix("noise-"))
                coordinates[sample_id] = (
                    family,
                    base_model_id,
                    base_index,
                    noise_index,
                    campaign.source_sample_ids[sample_id],
                )
    if len(expected_bases) != BASE_MODELS_PER_CAMPAIGN:
        raise Comparison2DValidationError("hidden hierarchy is not exactly 100 bases")
    base_truths = _material_hidden_base_truths(
        generator_seed=generator_seed,
        material_truth=material_truth,
        campaign=campaign,
        expected_bases=expected_bases,
        sample_ids_by_base=sample_ids_by_base,
        observation_rows=observation_rows,
    )
    protected_mesh = _nested_mesh_from_record(mesh)
    hidden_domain_midpoint_m = 0.5 * float(
        protected_mesh.cell_widths(material_truth.depth_cell_centers_m)[0].sum()
    )
    responses: dict[str, _RawModEMResponse] = {}
    forward_by_base: dict[str, ArtifactSnapshot] = {}
    base_identities: list[dict[str, Any]] = []
    for index, (raw_run, expected) in enumerate(
        zip(base_runs, expected_bases, strict=True)
    ):
        path = f"hidden_generation.base_forward_runs[{index}]"
        run = _mapping(raw_run, path)
        _exact_keys(
            run,
            frozenset(
                {
                    "base_index",
                    "base_layer_rng_key",
                    "section_rng_key",
                    "family_id",
                    "base_model_id",
                    "source_generator_sample_indices",
                    "clean_response_sha256",
                    "model",
                    "template",
                    "forward",
                    "provenance",
                }
            ),
            path,
        )
        family, base_model_id, source_indices = expected
        if (
            run["base_index"] != index
            or run["base_layer_rng_key"] != [generator_seed, index]
            or run["section_rng_key"] != [generator_seed, 2, index]
            or run["family_id"] != family
            or run["base_model_id"] != base_model_id
            or run["source_generator_sample_indices"] != list(source_indices)
        ):
            raise Comparison2DValidationError(
                f"{path} differs from the exact 100-base hierarchy"
            )
        forward = _artifact_reference(
            run["forward"],
            base=base,
            role=f"{path}.forward",
            seen_identities=seen_identities,
        )
        model = _artifact_reference(
            run["model"],
            base=base,
            role=f"{path}.model",
            seen_identities=seen_identities,
        )
        template = _artifact_reference(
            run["template"],
            base=base,
            role=f"{path}.template",
            seen_identities=seen_identities,
        )
        provenance = _artifact_reference(
            run["provenance"],
            base=base,
            role=f"{path}.provenance",
            seen_identities=seen_identities,
        )
        response = _parse_modem_forward_snapshot(forward, role=f"{path}.forward")
        response_sha = _response_identity(response)
        if run["clean_response_sha256"] != response_sha:
            raise Comparison2DValidationError(
                f"{path} clean response identity was not recomputed from forward.dat"
            )
        provenance_value = _strict_json(
            provenance, f"{path}.provenance", require_canonical=False
        )
        expected_payload = (
            json.dumps(provenance_value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        if provenance.payload != expected_payload:
            raise Comparison2DValidationError(
                f"{path}.provenance is not deterministic JSON"
            )
        truth_source = _validate_modem_run_provenance(
            provenance_value,
            path=f"{path}.provenance",
            forward_snapshot=forward,
            expected_truth_id=base_model_id,
            expected_mesh=mesh,
            expected_runtime=runtime,
            expected_canonical_depth_centres_m=analytic_contract[
                "canonical_depth_centres_m"
            ],
            generator=generator,
            expected_truth_identity=base_truths[base_model_id].identity_record(),
        )
        outputs = _mapping(provenance_value["outputs"], f"{path}.provenance.outputs")
        model_output = _mapping(outputs["model.rho"], f"{path}.outputs.model.rho")
        template_output = _mapping(
            outputs["template.dat"], f"{path}.outputs.template.dat"
        )
        _exact_keys(
            model_output,
            frozenset({"sha256", "size_bytes"}),
            f"{path}.outputs.model.rho",
        )
        _exact_keys(
            template_output,
            frozenset({"sha256", "size_bytes"}),
            f"{path}.outputs.template.dat",
        )
        if (
            _sha256(model_output["sha256"], f"{path}.outputs.model.rho.sha256"),
            _integer(
                model_output["size_bytes"],
                f"{path}.outputs.model.rho.size_bytes",
                minimum=1,
            ),
        ) != (model.sha256, model.size_bytes) or (
            _sha256(
                template_output["sha256"],
                f"{path}.outputs.template.dat.sha256",
            ),
            _integer(
                template_output["size_bytes"],
                f"{path}.outputs.template.dat.size_bytes",
                minimum=1,
            ),
        ) != (template.sha256, template.size_bytes):
            raise Comparison2DValidationError(
                f"{path} raw ModEM inputs differ from solver provenance"
            )
        expected_model, expected_template = _render_modem_input_bytes(
            base_truths[base_model_id], protected_mesh
        )
        if model.payload != expected_model or template.payload != expected_template:
            raise Comparison2DValidationError(
                f"{path} raw ModEM inputs were not derived from withheld truth material"
            )
        _hidden_base_truth_source(
            truth_source,
            path=f"{path}.provenance.truth_source",
            campaign=campaign,
            base_model_id=base_model_id,
            family=family,
            base_index=index,
            source_sample_indices=source_indices,
            generator_seed=generator_seed,
            generation_contract_sha256=generation_contract_sha256,
        )
        responses[base_model_id] = response
        forward_by_base[base_model_id] = forward
        base_identities.append(
            {
                "base_model_id": base_model_id,
                "base_index": index,
                "forward_sha256": forward.sha256,
                "forward_size_bytes": forward.size_bytes,
                "model_sha256": model.sha256,
                "model_size_bytes": model.size_bytes,
                "template_sha256": template.sha256,
                "template_size_bytes": template.size_bytes,
                "provenance_sha256": provenance.sha256,
                "provenance_size_bytes": provenance.size_bytes,
                "clean_response_sha256": response_sha,
            }
        )

    noise_identities: list[dict[str, Any]] = []
    seen_noise_coordinates: set[tuple[str, int]] = set()
    for index, (raw_row, sample_id) in enumerate(
        zip(raw_noise_records, sorted(coordinates), strict=True)
    ):
        path = f"hidden_generation.noise_rows[{index}]"
        row = _mapping(raw_row, path)
        _exact_keys(
            row,
            frozenset(
                {
                    "sample_index",
                    "source_generator_sample_index",
                    "base_index",
                    "noise_rng_key",
                    "base_model_id",
                    "family_id",
                    "noise_index",
                    "base_forward_sha256",
                    "clean_response_sha256",
                    "observation_row_sha256",
                    "noise_delta_sha256",
                }
            ),
            path,
        )
        (
            family,
            base_model_id,
            base_index,
            noise_index,
            source_index,
        ) = coordinates[sample_id]
        expected_fields = {
            "sample_index": sample_id,
            "source_generator_sample_index": source_index,
            "base_index": base_index,
            "noise_rng_key": [generator_seed, 3, base_index, noise_index],
            "base_model_id": base_model_id,
            "family_id": family,
            "noise_index": noise_index,
            "base_forward_sha256": forward_by_base[base_model_id].sha256,
            "clean_response_sha256": _response_identity(responses[base_model_id]),
        }
        if any(row[name] != expected for name, expected in expected_fields.items()):
            raise Comparison2DValidationError(
                f"{path} differs from its base solve/operator mapping"
            )
        coordinate = (base_model_id, noise_index)
        if coordinate in seen_noise_coordinates:
            raise Comparison2DValidationError("a base repeats a noise index")
        seen_noise_coordinates.add(coordinate)
        observed = observation_rows[sample_id]
        clean = responses[base_model_id]
        if not np.allclose(
            observed["frequency_hz"], clean.frequencies_hz, rtol=2e-6, atol=0.0
        ) or not np.allclose(
            observed["station_x_m"],
            clean.station_y_m - hidden_domain_midpoint_m,
            rtol=0.0,
            atol=0.002,
        ):
            raise Comparison2DValidationError(
                f"{path} observation axes differ from its clean ModEM solve"
            )
        _validate_hidden_noise_realization(
            observed,
            clean,
            generator_seed=generator_seed,
            base_index=base_index,
            noise_index=noise_index,
            path=path,
        )
        observation_sha = _observation_row_identity(
            sample_index=sample_id, arrays=observed
        )
        noise_sha = _noise_delta_identity(observed=observed, clean=clean)
        if (
            row["observation_row_sha256"] != observation_sha
            or row["noise_delta_sha256"] != noise_sha
        ):
            raise Comparison2DValidationError(
                f"{path} does not materially bind the observation/noise arrays"
            )
        noise_identities.append(
            {
                "sample_index": sample_id,
                "base_model_id": base_model_id,
                "base_index": row["base_index"],
                "noise_index": noise_index,
                "observation_row_sha256": observation_sha,
                "noise_delta_sha256": noise_sha,
            }
        )
    expected_noise_coordinates = {
        (base_model_id, noise_index)
        for _family, base_model_id, _source_indices in expected_bases
        for noise_index in range(NOISE_REALIZATIONS_PER_BASE)
    }
    if seen_noise_coordinates != expected_noise_coordinates:
        raise Comparison2DValidationError(
            "hidden generation does not prove five noise rows for every base"
        )
    validated = replace(
        campaign,
        generation_evidence_proven=True,
        generation_evidence_reason=None,
    )
    return validated, {
        "base_solve_count": len(base_identities),
        "noise_row_count": len(noise_identities),
        "raw_artifact_count": len(base_identities) * 4,
        "base_identities_sha256": _canonical_object_sha256(
            {"base_forward_runs": base_identities}
        ),
        "noise_identities_sha256": _canonical_object_sha256(
            {"noise_rows": noise_identities}
        ),
        "observation_payload_sha256": observation_snapshot.sha256,
        "runtime_identity_sha256": closure["runtime_identity_sha256"],
        "family_partition_commitment_sha256": campaign.family_commitment_sha256,
        "generator_seed": generator_seed,
        "generation_contract_sha256": generation_contract_sha256,
        "generation_runtime_sha256": _canonical_object_sha256(generation_runtime),
        "generation_runtime_manifest_sha256": runtime_manifest_reference["sha256"],
    }


def _operator_campaign(
    value: Mapping[str, Any],
    *,
    snapshot: ArtifactSnapshot,
    campaign_id: str,
    locked_observations_sha256: str,
    locked_observation_manifest_sha256: str,
    family_commitment_sha256: str,
) -> _Campaign:
    required_root = {
        "artifacts",
        "audience",
        "schema",
        "schema_version",
        "source",
        "split",
    }
    _exact_keys(value, frozenset(required_root), "operator manifest")
    if (
        value["audience"] != "benchmark_operator_only"
        or value["schema"] != OPERATOR_MANIFEST_SCHEMA
        or value["schema_version"] != OPERATOR_MANIFEST_SCHEMA_VERSION
    ):
        raise Comparison2DValidationError("operator scoring manifest identity is wrong")
    artifacts = _mapping(value["artifacts"], "operator artifacts")
    observations_sha, _ = _artifact_digest_record(
        artifacts.get("observations"),
        path="operator artifacts.observations",
        schema="pimsr-sota-2d-observations",
        schema_version=1,
        require_size=True,
    )
    truth_sha, truth_size = _artifact_digest_record(
        artifacts.get("withheld_truth"),
        path="operator artifacts.withheld_truth",
        schema="pimsr-sota-2d-truth",
        schema_version=2,
        require_size=True,
    )
    public_manifest_sha, _ = _artifact_digest_record(
        artifacts.get("public_observation_manifest"),
        path="operator artifacts.public_observation_manifest",
        schema="pimsr-sota-2d-observation-manifest",
        schema_version=3,
        require_size=True,
    )
    if observations_sha != locked_observations_sha256:
        raise Comparison2DValidationError(
            "operator observations differ from pre-score lock"
        )
    if public_manifest_sha != locked_observation_manifest_sha256:
        raise Comparison2DValidationError(
            "operator public observation manifest differs from pre-score lock"
        )
    split = _mapping(value["split"], "operator split")
    _exact_keys(
        split,
        frozenset(
            {
                "groups",
                "opaque_sample_id_contract",
                "sample_id_mapping",
                "sample_count",
                "scenario_groups",
                "family_partition_reveal",
                "payload_row_order",
                "split_id",
            }
        ),
        "operator split",
    )
    if split.get("split_id") != campaign_id:
        raise Comparison2DValidationError("operator split id differs from campaign id")
    if split.get("sample_count") != SAMPLES_PER_CAMPAIGN:
        raise Comparison2DValidationError(
            "operator campaign must contain exactly 500 samples"
        )
    groups = _sequence(split.get("groups"), "operator split.groups")
    if len(groups) != SAMPLES_PER_CAMPAIGN:
        raise Comparison2DValidationError(
            "operator hierarchy requires one family/base/noise record per sample"
        )
    tree: dict[str, dict[str, list[tuple[str, int]]]] = {}
    all_ids: set[int] = set()
    base_family: dict[str, str] = {}
    reveal_by_sample: dict[int, dict[str, Any]] = {}
    for index, raw_group in enumerate(groups):
        path = f"operator split.groups[{index}]"
        group = _mapping(raw_group, path)
        _exact_keys(
            group,
            frozenset({"base_model_id", "family_id", "noise_id", "sample_ids"}),
            path,
        )
        family = _string(group["family_id"], f"{path}.family_id")
        base = _string(group["base_model_id"], f"{path}.base_model_id")
        noise = _string(group["noise_id"], f"{path}.noise_id")
        noise_match = re.fullmatch(r"noise-([0-4])", noise, flags=re.ASCII)
        if noise_match is None:
            raise Comparison2DValidationError(
                "operator noise ids must be exactly noise-0 through noise-4"
            )
        noise_index = int(noise_match.group(1))
        previous_family = base_family.setdefault(base, family)
        if previous_family != family:
            raise Comparison2DValidationError(
                "one base model appears in multiple families"
            )
        sample_names = _sequence(group["sample_ids"], f"{path}.sample_ids")
        if len(sample_names) != 1:
            raise Comparison2DValidationError(
                "every base/noise realization must map to exactly one opaque sample"
            )
        sample_name = _string(sample_names[0], f"{path}.sample_ids[0]")
        if not sample_name.startswith("sample-") or not sample_name[7:].isdigit():
            raise Comparison2DValidationError(
                "operator sample id is not opaque int64 encoding"
            )
        sample_id = int(sample_name[7:])
        if sample_id > np.iinfo(np.int64).max or sample_id in all_ids:
            raise Comparison2DValidationError(
                "operator opaque sample ids are invalid/duplicate"
            )
        all_ids.add(sample_id)
        reveal_by_sample[sample_id] = {
            "sample_index": sample_id,
            "base_model_id": base,
            "family_id": family,
            "noise_index": noise_index,
        }
        noise_rows = tree.setdefault(family, {}).setdefault(base, [])
        if any(existing_noise == noise for existing_noise, _ in noise_rows):
            raise Comparison2DValidationError("one base repeats a noise realization id")
        noise_rows.append((noise, sample_id))
    if len(base_family) != BASE_MODELS_PER_CAMPAIGN:
        raise Comparison2DValidationError(
            "operator campaign requires exactly 100 base models"
        )
    if any(
        len(noise_rows) != NOISE_REALIZATIONS_PER_BASE
        for family in tree.values()
        for noise_rows in family.values()
    ):
        raise Comparison2DValidationError(
            "every base model requires exactly five noise realizations"
        )
    if set(tree) != set(FAMILY_IDS) or any(
        len(tree[family]) != BASE_MODELS_PER_FAMILY for family in FAMILY_IDS
    ):
        raise Comparison2DValidationError(
            "operator campaign requires exact five families with 20 bases each"
        )
    canonical_tree = {
        family: tuple(
            (base, tuple(sorted(noise_rows)))
            for base, noise_rows in sorted(bases.items())
        )
        for family in FAMILY_IDS
        for bases in (tree[family],)
    }
    opaque_contract = _mapping(
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
        raise Comparison2DValidationError("operator opaque sample-id contract is wrong")
    if split["payload_row_order"] != "strictly_increasing_opaque_sample_index":
        raise Comparison2DValidationError("operator payload row-order contract is wrong")
    mapping_rows = _sequence(split.get("sample_id_mapping"), "operator sample_id_mapping")
    mapped_ids: set[int] = set()
    opaque_ids: list[int] = []
    source_ids: list[int] = []
    source_by_opaque: dict[int, int] = {}
    for index, raw_mapping in enumerate(mapping_rows):
        path = f"operator sample_id_mapping[{index}]"
        mapping = _mapping(raw_mapping, path)
        _exact_keys(
            mapping,
            frozenset({"opaque_sample_index", "source_generator_sample_index"}),
            path,
        )
        opaque_id = _integer(
            mapping["opaque_sample_index"], f"{path}.opaque_sample_index"
        )
        source_id = _integer(
            mapping["source_generator_sample_index"],
            f"{path}.source_generator_sample_index",
        )
        if opaque_id > np.iinfo(np.int64).max or opaque_id in mapped_ids:
            raise Comparison2DValidationError(
                "operator sample mapping duplicates an opaque id"
            )
        if source_id > np.iinfo(np.int64).max or source_id in source_ids:
            raise Comparison2DValidationError(
                "operator sample mapping duplicates a source id"
            )
        mapped_ids.add(opaque_id)
        opaque_ids.append(opaque_id)
        source_ids.append(source_id)
        source_by_opaque[opaque_id] = source_id
    if (
        len(mapping_rows) != SAMPLES_PER_CAMPAIGN
        or mapped_ids != all_ids
        or opaque_ids != sorted(opaque_ids)
        or sorted(source_ids) != list(range(SAMPLES_PER_CAMPAIGN))
    ):
        raise Comparison2DValidationError(
            "operator sample mapping differs from hierarchy"
        )
    scenario_rows = _sequence(split["scenario_groups"], "operator scenario_groups")
    if len(scenario_rows) != len(tree):
        raise Comparison2DValidationError(
            "operator scenario groups do not cover families"
        )
    grouped_samples: dict[str, set[int]] = {}
    scenario_indices: set[int] = set()
    for index, raw_scenario in enumerate(scenario_rows):
        path = f"operator scenario_groups[{index}]"
        scenario = _mapping(raw_scenario, path)
        _exact_keys(
            scenario,
            frozenset({"opaque_sample_indices", "scenario", "scenario_index"}),
            path,
        )
        family = _string(scenario["scenario"], f"{path}.scenario")
        scenario_index = _integer(scenario["scenario_index"], f"{path}.scenario_index")
        sample_values = _sequence(
            scenario["opaque_sample_indices"], f"{path}.opaque_sample_indices"
        )
        samples = {
            _integer(value, f"{path}.opaque_sample_indices[{sample_index}]")
            for sample_index, value in enumerate(sample_values)
        }
        if (
            family in grouped_samples
            or scenario_index in scenario_indices
            or len(samples) != len(sample_values)
        ):
            raise Comparison2DValidationError(
                "operator scenario group ids are duplicated"
            )
        grouped_samples[family] = samples
        scenario_indices.add(scenario_index)
        if family not in FAMILY_IDS or scenario_index != FAMILY_IDS.index(family):
            raise Comparison2DValidationError(
                "operator scenario index differs from the frozen family order"
            )
    expected_grouped_samples = {
        family: {
            sample_id
            for _base, noise_rows in canonical_tree[family]
            for _noise, sample_id in noise_rows
        }
        for family in canonical_tree
    }
    if grouped_samples != expected_grouped_samples:
        raise Comparison2DValidationError(
            "operator scenario groups differ from hierarchy"
        )
    _validate_family_partition_reveal(
        split["family_partition_reveal"],
        campaign_id=campaign_id,
        expected_commitment_sha256=family_commitment_sha256,
        expected_rows=[reveal_by_sample[sample_id] for sample_id in sorted(all_ids)],
    )
    source = _mapping(value["source"], "operator source")
    if dict(source) != {
        "production_generation_closure": "post_score_manifest.campaign.hidden_generation"
    }:
        raise Comparison2DValidationError(
            "operator source must defer to the material hidden-generation closure"
        )
    return _Campaign(
        campaign_id=campaign_id,
        operator_sha256=snapshot.sha256,
        operator_size_bytes=snapshot.size_bytes,
        observations_sha256=observations_sha,
        observation_manifest_sha256=public_manifest_sha,
        truth_sha256=truth_sha,
        truth_size_bytes=int(truth_size),
        hierarchy=_Hierarchy(
            sample_ids=tuple(sorted(all_ids)),
            families=tuple(canonical_tree),
            tree=canonical_tree,
        ),
        source_sample_ids=source_by_opaque,
        family_commitment_sha256=family_commitment_sha256,
        generation_evidence_proven=False,
        generation_evidence_reason="hidden generation closure has not been validated",
    )


def _validate_metric_contract(
    value: Any,
    path: str,
    *,
    x_cell_centers_m: np.ndarray,
    depth_cell_centers_m: np.ndarray,
) -> Mapping[str, Any]:
    contract = _mapping(value, path)
    expected_values = {
        "aggregation_across_samples": "equal_sample_weight",
        "campaign_binding": "exact_observations_sha256_in_truth_and_prediction",
        "cell_edges": "midpoints_with_half_spacing_boundary_extrapolation",
        "grid_weighting": "normalized_physical_cell_area",
        "prediction_grid": "must_exactly_match_withheld_truth_grid",
        "quantity": "log10_resistivity_ohm_m",
        "sample_pairing": "exact_unique_sample_index",
    }
    expected_keys = frozenset({*expected_values, "scoring_domain"})
    _exact_keys(contract, expected_keys, path)
    for key, expected in expected_values.items():
        if contract[key] != expected:
            raise Comparison2DValidationError(f"{path}.{key} is not exact")
    domain = _mapping(contract["scoring_domain"], f"{path}.scoring_domain")
    _exact_keys(
        domain,
        frozenset({"depth_cell_edges_m", "mask", "support", "x_cell_edges_m"}),
        f"{path}.scoring_domain",
    )
    if (
        domain["mask"] != "all_truth_grid_cells"
        or domain["support"] != "full_grid_voronoi_cells_from_centers"
    ):
        raise Comparison2DValidationError("evaluation scoring domain is not full-grid")
    expected_edges = {
        "depth_cell_edges_m": protected_evaluation2d.cell_edges_from_centers(
            depth_cell_centers_m
        ),
        "x_cell_edges_m": protected_evaluation2d.cell_edges_from_centers(
            x_cell_centers_m
        ),
    }
    for name, material_edges in expected_edges.items():
        raw = _sequence(domain[name], f"{path}.scoring_domain.{name}")
        edges = np.asarray(
            [
                _finite(value, f"{path}.scoring_domain.{name}[{index}]")
                for index, value in enumerate(raw)
            ]
        )
        if not np.array_equal(edges, material_edges):
            raise Comparison2DValidationError(
                "evaluation scoring edges differ from the material truth grid"
            )
    return contract


def _validate_evaluation_overall(value: Any, metrics: np.ndarray, *, path: str) -> None:
    overall = _mapping(value, path)
    if overall.get("n_samples") != SAMPLES_PER_CAMPAIGN:
        raise Comparison2DValidationError("evaluation overall sample count is not 500")
    for metric_index, metric_id in enumerate(METRIC_IDS):
        record = _mapping(overall.get(metric_id), f"{path}.{metric_id}")
        expected = {
            "mean": float(np.mean(metrics[:, metric_index])),
            "median": float(np.median(metrics[:, metric_index])),
        }
        for aggregate, wanted in expected.items():
            aggregate_record = _mapping(
                record.get(aggregate), f"{path}.{metric_id}.{aggregate}"
            )
            estimate = _finite(
                aggregate_record.get("estimate"),
                f"{path}.{metric_id}.{aggregate}.estimate",
            )
            if not math.isclose(estimate, wanted, rel_tol=1e-12, abs_tol=1e-12):
                raise Comparison2DValidationError(
                    "evaluation overall estimate differs from per-sample metrics"
                )


def _evaluation_report(
    value: Mapping[str, Any],
    *,
    snapshot: ArtifactSnapshot,
    campaign: _Campaign,
    run: LockedRun2D,
    validated_lock: ValidatedPredictionLock2D,
    evaluator_contract: Mapping[str, Any],
    material_truth: _MaterialTruth2D,
    material_predictions: _MaterialPredictions2D,
) -> _Evaluation:
    _exact_keys(value, _EVALUATION_ROOT_KEYS, "evaluation")
    if (
        value["audience"] != "benchmark_operator_only_after_predictions_locked"
        or value["schema"] != EVALUATION_SCHEMA
        or value["schema_version"] != EVALUATION_SCHEMA_VERSION
    ):
        raise Comparison2DValidationError("evaluation identity is not schema v3")
    inputs = _mapping(value["inputs"], "evaluation.inputs")
    _exact_keys(
        inputs,
        frozenset(
            {
                "prediction_lock",
                "observations",
                "prediction",
                "truth",
                "operator_manifest",
            }
        ),
        "evaluation.inputs",
    )
    lock_record = _mapping(inputs["prediction_lock"], "evaluation.inputs.prediction_lock")
    _exact_keys(
        lock_record,
        frozenset(
            {
                "input_manifest_sha256",
                "preregistration_sha256",
                "schema",
                "schema_version",
                "sha256",
            }
        ),
        "evaluation.inputs.prediction_lock",
    )
    expected_lock_record = {
        "input_manifest_sha256": validated_lock.input_manifest_sha256,
        "preregistration_sha256": validated_lock.preregistration_sha256,
        "schema": "pimsr-sota-2d-predictions-lock",
        "schema_version": 2,
        "sha256": validated_lock.lock_sha256,
    }
    if dict(lock_record) != expected_lock_record:
        raise Comparison2DValidationError("evaluation lock context is wrong")
    observations_sha, _ = _artifact_digest_record(
        inputs["observations"],
        path="evaluation.inputs.observations",
        schema="pimsr-sota-2d-observations",
        schema_version=1,
        require_size=False,
    )
    prediction_sha, prediction_size = _artifact_digest_record(
        inputs["prediction"],
        path="evaluation.inputs.prediction",
        schema="pimsr-sota-2d-predictions",
        schema_version=2,
        require_size=True,
    )
    truth_sha, truth_size = _artifact_digest_record(
        inputs["truth"],
        path="evaluation.inputs.truth",
        schema="pimsr-sota-2d-truth",
        schema_version=2,
        require_size=True,
    )
    operator_sha, operator_size = _artifact_digest_record(
        inputs["operator_manifest"],
        path="evaluation.inputs.operator_manifest",
        schema=OPERATOR_MANIFEST_SCHEMA,
        schema_version=OPERATOR_MANIFEST_SCHEMA_VERSION,
        require_size=True,
    )
    if (
        observations_sha != run.observations_sha256
        or prediction_sha != run.prediction_sha256
        or prediction_size != run.prediction_size_bytes
        or truth_sha != campaign.truth_sha256
        or truth_size != campaign.truth_size_bytes
        or operator_sha != campaign.operator_sha256
        or operator_size != campaign.operator_size_bytes
    ):
        raise Comparison2DValidationError(
            "evaluation artifact bindings differ from lock/operator"
        )
    if (
        material_truth.artifact_sha256 != truth_sha
        or material_truth.artifact_size_bytes != campaign.truth_size_bytes
        or material_predictions.artifact_sha256 != prediction_sha
        or material_predictions.artifact_size_bytes != prediction_size
        or material_predictions.observations_sha256 != run.observations_sha256
    ):
        raise Comparison2DValidationError(
            "evaluation material artifacts differ from their locked digest records"
        )
    run_record = _mapping(value["run"], "evaluation.run")
    _exact_keys(run_record, _RUN_BINDING_KEYS, "evaluation.run")
    expected_run = {
        "adapter_source_sha256": run.adapter_source_sha256,
        "campaign_id": run.campaign_id,
        "checkpoint_sha256": run.checkpoint_sha256,
        "method_id": run.method_id,
        "runtime_sha256": run.runtime_sha256,
        "source_commit": run.source_commit,
        "source_sha256": run.source_sha256,
        "training_seed": run.training_seed,
    }
    if dict(run_record) != expected_run:
        raise Comparison2DValidationError(
            "evaluation run binding differs from prediction lock"
        )
    implementation = _mapping(value["implementation"], "evaluation.implementation")
    _exact_keys(
        implementation,
        frozenset(
            {
                "distribution_version",
                "git_commit",
                "git_head_commit",
                "git_dirty_tree",
                "numpy_version",
                "python_version",
                "source_file",
                "source_sha256",
                "source_size_bytes",
            }
        ),
        "evaluation.implementation",
    )
    _git_commit(
        implementation["git_head_commit"],
        "evaluation implementation git_head_commit",
    )
    if (
        implementation["git_commit"] != evaluator_contract["repository_commit"]
        or implementation["source_sha256"] != evaluator_contract["source_sha256"]
        or implementation["git_dirty_tree"] is not False
        or implementation["source_file"] != "evaluation2d.py"
        or _integer(
            implementation["source_size_bytes"],
            "evaluation implementation source_size_bytes",
            minimum=1,
        )
        < 1
    ):
        raise Comparison2DValidationError(
            "evaluation implementation is not the pinned clean source"
        )
    release = _mapping(value["release_gate"], "evaluation.release_gate")
    if (
        release.get("predictions_locked") is not True
        or release.get("public_release_allowed") is not False
    ):
        raise Comparison2DValidationError(
            "evaluation release gate is not post-lock/operator-only"
        )
    bootstrap = _mapping(value["bootstrap_contract"], "evaluation.bootstrap_contract")
    if (
        bootstrap.get("headline_eligible") is not False
        or bootstrap.get("cross_method_effect_ci") is not False
        or bootstrap.get("hierarchical") is not False
    ):
        raise Comparison2DValidationError(
            "single-run evaluation cannot claim cross-method inference"
        )
    physics = _mapping(value["physics_misfit"], "evaluation.physics_misfit")
    if type(physics.get("included")) is not bool:
        raise Comparison2DValidationError(
            "evaluation physics_misfit.included must be boolean"
        )
    metric_contract = _validate_metric_contract(
        value["metric_contract"],
        "evaluation.metric_contract",
        x_cell_centers_m=material_truth.x_cell_centers_m,
        depth_cell_centers_m=material_truth.depth_cell_centers_m,
    )
    rows = _sequence(value["per_sample"], "evaluation.per_sample")
    if len(rows) != SAMPLES_PER_CAMPAIGN:
        raise Comparison2DValidationError(
            "evaluation must contain exactly 500 scored rows"
        )
    sample_family = {
        sample_id: family
        for family in campaign.hierarchy.families
        for _base, noise_rows in campaign.hierarchy.tree[family]
        for _noise, sample_id in noise_rows
    }
    truth_position = {
        int(sample_id): index
        for index, sample_id in enumerate(material_truth.sample_index)
    }
    if set(truth_position) != set(campaign.hierarchy.sample_ids):
        raise Comparison2DValidationError(
            "withheld truth material ids differ from the operator hierarchy"
        )
    by_id: dict[int, tuple[float, float]] = {}
    for index, raw_row in enumerate(rows):
        path = f"evaluation.per_sample[{index}]"
        row = _mapping(raw_row, path)
        _exact_keys(row, _PER_SAMPLE_KEYS, path)
        sample_id = _integer(row["sample_index"], f"{path}.sample_index")
        if sample_id > np.iinfo(np.int64).max or sample_id in by_id:
            raise Comparison2DValidationError(
                "evaluation opaque sample ids are invalid/duplicate"
            )
        if type(row["has_fault"]) is not bool:
            raise Comparison2DValidationError(f"{path}.has_fault must be boolean")
        scenario = _string(row["scenario"], f"{path}.scenario")
        truth_row = truth_position.get(sample_id)
        if (
            sample_family.get(sample_id) != scenario
            or truth_row is None
            or material_truth.scenario[truth_row] != scenario
            or bool(material_truth.has_fault[truth_row]) != row["has_fault"]
        ):
            raise Comparison2DValidationError(
                "evaluation scenario/fault fields differ from material truth/operator"
            )
        by_id[sample_id] = (
            _finite(
                row["rmse_log10_resistivity"],
                f"{path}.rmse_log10_resistivity",
                nonnegative=True,
            ),
            _finite(
                row["mae_log10_resistivity"],
                f"{path}.mae_log10_resistivity",
                nonnegative=True,
            ),
        )
    sample_ids = tuple(sorted(by_id))
    if sample_ids != campaign.hierarchy.sample_ids:
        raise Comparison2DValidationError("evaluation ids differ from operator hierarchy")
    metrics = np.asarray([by_id[sample_id] for sample_id in sample_ids], dtype=np.float64)
    material_sample_ids, material_metrics = _recomputed_material_metrics(
        material_truth, material_predictions
    )
    if material_sample_ids != sample_ids or not np.array_equal(material_metrics, metrics):
        raise Comparison2DValidationError(
            "evaluation per-sample RMSE/MAE were not recomputed from raw truth/prediction bytes"
        )
    _validate_evaluation_overall(value["overall"], metrics, path="evaluation.overall")
    return _Evaluation(
        sha256=snapshot.sha256,
        metrics=metrics,
        sample_ids=sample_ids,
        metric_contract=metric_contract,
        physics_misfit_included=physics["included"],
    )


def _validated_lock(
    preregistration_path: str | Path,
    prediction_lock_path: str | Path,
    *,
    expected_preregistration_sha256: str,
    expected_prediction_lock_sha256: str,
) -> ValidatedPredictionLock2D:
    try:
        result = validate_prediction_lock_2d(
            preregistration_path,
            prediction_lock_path,
            expected_preregistration_sha256=expected_preregistration_sha256,
            expected_lock_sha256=expected_prediction_lock_sha256,
        )
    except Exception as exc:
        raise Comparison2DValidationError(
            f"global pre-score prediction lock validation failed: {exc}"
        ) from exc
    if (
        result.method_ids != METHOD_IDS
        or result.training_seeds != TRAINING_SEEDS
        or len(result.campaign_ids) != CAMPAIGN_COUNT
        or len(set(result.campaign_ids)) != CAMPAIGN_COUNT
        or len(result.runs) != RUN_COUNT
    ):
        raise Comparison2DValidationError("validated lock design is not exact 5x5x3")
    expected_cells = {
        (campaign, method, seed)
        for campaign in result.campaign_ids
        for method in METHOD_IDS
        for seed in TRAINING_SEEDS
    }
    actual_cells = {
        (run.campaign_id, run.method_id, run.training_seed) for run in result.runs
    }
    if actual_cells != expected_cells or len(actual_cells) != len(result.runs):
        raise Comparison2DValidationError("validated lock matrix is incomplete/duplicate")
    return result


def _pairwise_report(result: HierarchicalBootstrap2D) -> dict[str, Any]:
    pairwise: dict[str, Any] = {}
    for reference_index, reference in enumerate(REFERENCE_METHOD_IDS):
        metrics: dict[str, Any] = {}
        for metric_index, metric in enumerate(METRIC_IDS):
            metrics[metric] = {
                "candidate_better_at_point_estimate": bool(
                    result.point[reference_index, metric_index] < 0.0
                ),
                "candidate_minus_reference": float(
                    result.point[reference_index, metric_index]
                ),
                "direction": "negative_favors_pimsr",
                "two_sided_95_ci": {
                    "lower": float(result.two_sided_lower[reference_index, metric_index]),
                    "upper": float(result.two_sided_upper[reference_index, metric_index]),
                },
                "one_sided_upper_95": float(
                    result.one_sided_upper[reference_index, metric_index]
                ),
            }
        pairwise[reference] = {
            "metrics": metrics,
            "individual_superiority_claim": (
                "descriptive_only_no_separate_multiplicity_adjusted_claim"
            ),
        }
    return pairwise


def compare_evaluations_2d(
    preregistration_path: str | Path,
    prediction_lock_path: str | Path,
    post_score_manifest_path: str | Path,
    *,
    expected_preregistration_sha256: str,
    expected_prediction_lock_sha256: str,
    expected_post_score_manifest_sha256: str,
) -> dict[str, Any]:
    """Validate locked evidence and compute the preregistered 3-method effects."""
    # This call is intentionally first.  No post-score/operator/truth path has
    # been opened when the pre-score capability gate is validated.
    locked = _validated_lock(
        preregistration_path,
        prediction_lock_path,
        expected_preregistration_sha256=expected_preregistration_sha256,
        expected_prediction_lock_sha256=expected_prediction_lock_sha256,
    )
    options = _validate_statistical_options(locked.statistical_options)
    seen_identities: set[tuple[int, int]] = set()
    preregistration_snapshot = _snapshot_unique(
        preregistration_path,
        expected_sha256=locked.preregistration_sha256,
        expected_size_bytes=None,
        role="preregistration after lock validation",
        seen_identities=seen_identities,
    )
    preregistration = _strict_json(
        preregistration_snapshot,
        "preregistration",
        require_canonical=False,
    )
    contracts = _prereg_contracts(preregistration, locked)
    evaluator_contract = contracts.evaluator
    headline_evidence = contracts.headline_evidence
    # These identities are all pre-score/public.  Validate them before the
    # first byte of the post-score capability is opened.
    implementation_identity = _validate_comparison_implementation(
        contracts.comparison_implementation
    )
    lineage_identities = _load_public_lineages(
        contracts.public_lineage,
        preregistration=preregistration,
        base=preregistration_snapshot.path.parent,
        seen_identities=seen_identities,
    )
    _validate_generator_distinct_from_lineage(
        headline_evidence["hidden_observation_generator"], lineage_identities
    )
    final_implementation_identity = _validate_comparison_implementation(
        contracts.comparison_implementation
    )
    if final_implementation_identity != implementation_identity:
        raise Comparison2DValidationError(
            "comparison implementation identity changed before post-score access"
        )
    post_snapshot = _snapshot_unique(
        post_score_manifest_path,
        expected_sha256=_sha256(
            expected_post_score_manifest_sha256,
            "expected post-score manifest SHA-256",
        ),
        expected_size_bytes=None,
        role="post-score comparison manifest",
        seen_identities=seen_identities,
    )
    post = _strict_json(post_snapshot, "post-score comparison manifest")
    _exact_keys(post, _POST_SCORE_KEYS, "post-score comparison manifest")
    if (
        post["audience"] != "benchmark_operator_only_after_predictions_locked"
        or post["schema"] != POST_SCORE_SCHEMA
        or post["schema_version"] != POST_SCORE_SCHEMA_VERSION
        or post["preregistration_sha256"] != locked.preregistration_sha256
        or post["prediction_lock_sha256"] != locked.lock_sha256
        or post["prediction_lock_input_manifest_sha256"] != locked.input_manifest_sha256
    ):
        raise Comparison2DValidationError("post-score manifest lock binding is wrong")
    evaluator = _mapping(post["evaluator_implementation"], "evaluator_implementation")
    _exact_keys(evaluator, _EVALUATOR_KEYS, "evaluator_implementation")
    if dict(evaluator) != {
        "repository_commit": evaluator_contract["repository_commit"],
        "source_sha256": evaluator_contract["source_sha256"],
    }:
        raise Comparison2DValidationError(
            "post-score evaluator differs from preregistration"
        )
    if post["headline_evidence"] != headline_evidence:
        raise Comparison2DValidationError(
            "post-score headline evidence differs from preregistration"
        )
    evidence_valid, evidence_reasons, evidence_identities = (
        _validate_headline_evidence_files(
            post["evidence_artifacts"],
            post["public_convergence_raw_runs"],
            evidence=headline_evidence,
            public_lineage=lineage_identities,
            base=post_snapshot.path.parent,
            seen_identities=seen_identities,
        )
    )
    if not evidence_valid:
        raise Comparison2DValidationError(
            "headline evidence validation failed: " + "; ".join(evidence_reasons)
        )

    raw_campaigns = _sequence(post["campaigns"], "post-score campaigns")
    if len(raw_campaigns) != CAMPAIGN_COUNT:
        raise Comparison2DValidationError("post-score manifest requires five campaigns")
    campaigns: list[_Campaign] = []
    evaluations: dict[tuple[str, str, int], _Evaluation] = {}
    campaign_inputs: list[dict[str, Any]] = []
    hidden_campaign_seeds: list[int] = []
    common_metric_contract: Mapping[str, Any] | None = None
    for campaign_index, raw_campaign in enumerate(raw_campaigns):
        path = f"post-score campaigns[{campaign_index}]"
        campaign_record = _mapping(raw_campaign, path)
        _exact_keys(campaign_record, _CAMPAIGN_KEYS, path)
        campaign_id = _identifier(campaign_record["campaign_id"], f"{path}.campaign_id")
        if campaign_id != locked.campaign_ids[campaign_index]:
            raise Comparison2DValidationError(
                "post-score campaigns are not exact lock-ordered ids"
            )
        representative = locked.require_run(campaign_id, METHOD_IDS[0], TRAINING_SEEDS[0])
        for method_id in METHOD_IDS:
            for training_seed in TRAINING_SEEDS:
                cell = locked.require_run(campaign_id, method_id, training_seed)
                if (
                    cell.observations_sha256 != representative.observations_sha256
                    or cell.observation_manifest_sha256
                    != representative.observation_manifest_sha256
                ):
                    raise Comparison2DValidationError(
                        "one campaign changes observations within the lock"
                    )
        public_manifest_snapshot = _artifact_reference(
            campaign_record["public_observation_manifest"],
            base=post_snapshot.path.parent,
            role=f"{path}.public_observation_manifest",
            seen_identities=seen_identities,
        )
        if public_manifest_snapshot.sha256 != representative.observation_manifest_sha256:
            raise Comparison2DValidationError(
                "post-score public observation manifest differs from the pre-score lock"
            )
        (
            expected_family_commitment,
            expected_observation_payload_size,
        ) = _public_observation_family_commitment(
            public_manifest_snapshot,
            campaign_id=campaign_id,
            expected_observations_sha256=representative.observations_sha256,
            expected_contract=contracts.family_partition["commitment_contract"],
        )
        operator_snapshot = _artifact_reference(
            campaign_record["operator_manifest"],
            base=post_snapshot.path.parent,
            role=f"{path}.operator_manifest",
            seen_identities=seen_identities,
        )
        operator_value = _strict_json(operator_snapshot, f"{path}.operator_manifest")
        campaign = _operator_campaign(
            operator_value,
            snapshot=operator_snapshot,
            campaign_id=campaign_id,
            locked_observations_sha256=representative.observations_sha256,
            locked_observation_manifest_sha256=representative.observation_manifest_sha256,
            family_commitment_sha256=expected_family_commitment,
        )
        truth_snapshot = _artifact_reference(
            campaign_record["withheld_truth"],
            base=post_snapshot.path.parent,
            role=f"{path}.withheld_truth",
            seen_identities=seen_identities,
        )
        if (
            truth_snapshot.sha256 != campaign.truth_sha256
            or truth_snapshot.size_bytes != campaign.truth_size_bytes
        ):
            raise Comparison2DValidationError(
                "post-score withheld truth differs from the operator binding"
            )
        material_truth = _load_material_truth(truth_snapshot)
        if material_truth.observations_sha256 != campaign.observations_sha256:
            raise Comparison2DValidationError(
                "withheld truth material differs from the locked observations"
            )
        campaign, hidden_generation_identity = _validate_hidden_generation_closure(
            campaign_record["hidden_generation"],
            campaign=campaign,
            material_truth=material_truth,
            evidence=headline_evidence,
            public_lineage=lineage_identities,
            expected_generation_runtime_manifest=post["evidence_artifacts"][
                "generation_runtime_manifest"
            ],
            expected_observation_payload_size=expected_observation_payload_size,
            base=post_snapshot.path.parent,
            seen_identities=seen_identities,
        )
        hidden_campaign_seeds.append(int(hidden_generation_identity["generator_seed"]))
        raw_evaluations = _sequence(campaign_record["evaluations"], f"{path}.evaluations")
        if len(raw_evaluations) != len(METHOD_IDS) * len(TRAINING_SEEDS):
            raise Comparison2DValidationError(
                "every campaign requires exactly 15 post-score evaluations"
            )
        evaluation_inputs: list[dict[str, Any]] = []
        seen_cells: set[tuple[str, int]] = set()
        for evaluation_index, raw_reference in enumerate(raw_evaluations):
            reference_path = f"{path}.evaluations[{evaluation_index}]"
            reference = _mapping(raw_reference, reference_path)
            _exact_keys(reference, _EVALUATION_REFERENCE_KEYS, reference_path)
            method_id = _identifier(reference["method_id"], f"{reference_path}.method_id")
            training_seed = _integer(
                reference["training_seed"], f"{reference_path}.training_seed"
            )
            cell_key = (method_id, training_seed)
            if cell_key in seen_cells:
                raise Comparison2DValidationError(
                    "post-score evaluations duplicate a run cell"
                )
            seen_cells.add(cell_key)
            run = locked.require_run(campaign_id, method_id, training_seed)
            prediction_snapshot = _artifact_reference(
                reference["prediction"],
                base=post_snapshot.path.parent,
                role=f"{reference_path}.prediction",
                seen_identities=seen_identities,
            )
            if (
                prediction_snapshot.sha256 != run.prediction_sha256
                or prediction_snapshot.size_bytes != run.prediction_size_bytes
            ):
                raise Comparison2DValidationError(
                    f"{reference_path} prediction differs from the pre-score lock"
                )
            material_predictions = _load_material_predictions(prediction_snapshot)
            evaluation_snapshot = _artifact_reference(
                reference["evaluation"],
                base=post_snapshot.path.parent,
                role=f"{reference_path}.evaluation",
                seen_identities=seen_identities,
            )
            evaluation_value = _strict_json(
                evaluation_snapshot, f"{reference_path}.evaluation"
            )
            evaluation = _evaluation_report(
                evaluation_value,
                snapshot=evaluation_snapshot,
                campaign=campaign,
                run=run,
                validated_lock=locked,
                evaluator_contract=evaluator_contract,
                material_truth=material_truth,
                material_predictions=material_predictions,
            )
            if common_metric_contract is None:
                common_metric_contract = evaluation.metric_contract
            elif evaluation.metric_contract != common_metric_contract:
                raise Comparison2DValidationError(
                    "metric contract changes across post-score evaluations"
                )
            evaluations[(campaign_id, method_id, training_seed)] = evaluation
            evaluation_inputs.append(
                {
                    "method_id": method_id,
                    "sha256": evaluation.sha256,
                    "prediction_sha256": prediction_snapshot.sha256,
                    "training_seed": training_seed,
                }
            )
        expected_cells = {
            (method_id, training_seed)
            for method_id in METHOD_IDS
            for training_seed in TRAINING_SEEDS
        }
        if seen_cells != expected_cells:
            raise Comparison2DValidationError(
                "post-score campaign evaluation matrix is incomplete"
            )
        campaigns.append(campaign)
        campaign_inputs.append(
            {
                "base_model_count": BASE_MODELS_PER_CAMPAIGN,
                "campaign_id": campaign_id,
                "evaluations": sorted(
                    evaluation_inputs,
                    key=lambda item: (item["method_id"], item["training_seed"]),
                ),
                "family_ids": list(campaign.hierarchy.families),
                "family_partition_commitment_sha256": expected_family_commitment,
                "hidden_generation": dict(hidden_generation_identity),
                "observation_manifest_sha256": representative.observation_manifest_sha256,
                "public_observation_manifest_size_bytes": (
                    public_manifest_snapshot.size_bytes
                ),
                "observations_sha256": representative.observations_sha256,
                "operator_manifest_sha256": campaign.operator_sha256,
                "sample_count": SAMPLES_PER_CAMPAIGN,
                "withheld_truth_sha256": truth_snapshot.sha256,
                "withheld_truth_size_bytes": truth_snapshot.size_bytes,
            }
        )
    if len({campaign.observations_sha256 for campaign in campaigns}) != CAMPAIGN_COUNT:
        raise Comparison2DValidationError("hidden campaigns reuse observation artifacts")
    family_sets = {campaign.hierarchy.families for campaign in campaigns}
    if len(family_sets) != 1:
        raise Comparison2DValidationError("campaigns do not preserve family ids")
    hidden_seed_identity = _validate_hidden_campaign_seed_reveal(
        hidden_campaign_seeds, contracts.hidden_seed_commitment
    )

    effect_rows: list[EffectRow2D] = []
    for campaign in campaigns:
        positions = {
            sample_id: index
            for index, sample_id in enumerate(campaign.hierarchy.sample_ids)
        }
        for family in campaign.hierarchy.families:
            for base_model_id, noise_rows in campaign.hierarchy.tree[family]:
                for noise_id, sample_id in noise_rows:
                    row_index = positions[sample_id]
                    for training_seed in TRAINING_SEEDS:
                        candidate_metrics = evaluations[
                            (campaign.campaign_id, CANDIDATE_METHOD_ID, training_seed)
                        ].metrics[row_index]
                        reference_metrics = np.stack(
                            [
                                evaluations[
                                    (campaign.campaign_id, reference, training_seed)
                                ].metrics[row_index]
                                for reference in REFERENCE_METHOD_IDS
                            ],
                            axis=0,
                        )
                        effect_rows.append(
                            EffectRow2D(
                                campaign_id=campaign.campaign_id,
                                training_seed=training_seed,
                                family_id=family,
                                base_model_id=base_model_id,
                                noise_id=noise_id,
                                effects=reference_metrics * -1.0
                                + candidate_metrics[None, :],
                            )
                        )
    bootstrap = hierarchical_paired_bootstrap_2d(
        effect_rows,
        training_seeds=TRAINING_SEEDS,
        confidence=options["confidence"],
        n_resamples=options["n_resamples"],
        rng_seed=options["rng_seed"],
    )
    pairwise = _pairwise_report(bootstrap)
    primary_index = METRIC_IDS.index(PRIMARY_METRIC_ID)
    dominance_components = {
        reference: float(bootstrap.one_sided_upper[index, primary_index])
        for index, reference in enumerate(REFERENCE_METHOD_IDS)
    }
    dominance_passed = all(value < 0.0 for value in dominance_components.values())
    operator_generation_proven = all(
        campaign.generation_evidence_proven for campaign in campaigns
    )
    claim_reasons = list(evidence_reasons)
    claim_reasons.extend(
        campaign.generation_evidence_reason
        for campaign in campaigns
        if campaign.generation_evidence_reason is not None
    )
    if not dominance_passed:
        claim_reasons.append(
            "one-sided 95% upper effect is not below zero versus both references"
        )
    headline_eligible = bool(
        evidence_valid and operator_generation_proven and dominance_passed
    )
    physics_count = sum(
        evaluation.physics_misfit_included for evaluation in evaluations.values()
    )
    family_effects = [
        {
            "family_id": family,
            "pairwise": {
                reference: {
                    metric: float(
                        bootstrap.family_points[family][reference_index, metric_index]
                    )
                    for metric_index, metric in enumerate(METRIC_IDS)
                }
                for reference_index, reference in enumerate(REFERENCE_METHOD_IDS)
            },
        }
        for family in sorted(bootstrap.family_points)
    ]
    assert common_metric_contract is not None
    return {
        "audience": "benchmark_operator_only_pending_redacted_release",
        "schema": COMPARISON_SCHEMA,
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "methods": {
            "candidate": CANDIDATE_METHOD_ID,
            "references": list(REFERENCE_METHOD_IDS),
        },
        "design": {
            "base_models_per_campaign": BASE_MODELS_PER_CAMPAIGN,
            "campaign_count": CAMPAIGN_COUNT,
            "campaign_ids": list(locked.campaign_ids),
            "family_weighting": "equal",
            "method_count": len(METHOD_IDS),
            "noise_realizations_per_base": NOISE_REALIZATIONS_PER_BASE,
            "run_count": RUN_COUNT,
            "samples_per_campaign": SAMPLES_PER_CAMPAIGN,
            "training_seeds": list(TRAINING_SEEDS),
        },
        "inputs": {
            "campaigns": campaign_inputs,
            "evidence_artifacts": evidence_identities,
            "hidden_campaign_seed_commitment": hidden_seed_identity,
            "public_dataset_lineage": lineage_identities,
            "post_score_manifest": {
                "schema": POST_SCORE_SCHEMA,
                "schema_version": POST_SCORE_SCHEMA_VERSION,
                "sha256": post_snapshot.sha256,
                "size_bytes": post_snapshot.size_bytes,
            },
            "prediction_lock": {
                "input_manifest_sha256": locked.input_manifest_sha256,
                "sha256": locked.lock_sha256,
                "validated_before_operator_access": True,
            },
            "preregistration_sha256": locked.preregistration_sha256,
        },
        "metric_contract": dict(common_metric_contract),
        "statistical_contract": {
            **options,
            "effect": "pimsr_minus_reference_negative_is_better",
            "family_policy": "equal_weight_resampled_with_replacement",
            "method_pairing": "same_draws_jointly_for_both_references",
            "pairwise_interval": "two_sided_percentile_95",
            "dominance_interval": "one_sided_upper_percentile_95",
            "dominance_inference": (
                "single_intersection_union_test_no_multiplicity_correction"
            ),
        },
        "pairwise_effects": pairwise,
        "family_effects": family_effects,
        "global_dominance": {
            "candidate": CANDIDATE_METHOD_ID,
            "intersection_union_passed": dominance_passed,
            "one_sided_upper_95_primary_metric": dominance_components,
            "primary_metric": PRIMARY_METRIC_ID,
            "rule": "upper_95_below_zero_against_both_references",
        },
        "physics_misfit": {
            "evaluation_count_including_secondary_physics_misfit": physics_count,
            "required_for_truth_known_primary_rmse_headline": False,
            "status": (
                "available_for_all_runs"
                if physics_count == RUN_COUNT
                else "absent_or_partial_disclosed"
            ),
            "total_evaluation_count": RUN_COUNT,
        },
        "claim_gate": {
            "all_75_locked_runs_complete": True,
            "comparison_implementation_preregistered_and_clean": True,
            "distinct_generation_and_training_lineage_materially_proven": True,
            "hidden_campaign_seed_commitment_verified": True,
            "headline_evidence_artifacts_valid": evidence_valid,
            "headline_model_rmse_eligible": headline_eligible,
            "independent_modem_generation_proven_for_all_campaigns": (
                operator_generation_proven
            ),
            "public_mesh_convergence_proven": evidence_valid,
            "reasons": sorted(set(claim_reasons)),
            "statistical_global_dominance_passed": dominance_passed,
        },
        "release_gate": {
            "headline_eligible": headline_eligible,
            "prediction_lock_validated": True,
            "public_release_allowed": False,
            "required_next_step": "produce_and_review_a_redacted_public_report",
        },
        "implementation": implementation_identity,
    }


def _read_exact_fd(stream: Any, size_bytes: int) -> bytes:
    stream.seek(0)
    payload = stream.read(size_bytes + 1)
    if len(payload) != size_bytes:
        raise Comparison2DPublicationError("published comparison size changed")
    return payload


def _make_publication_descriptor_read_only(descriptor: int) -> None:
    if hasattr(os, "fchmod"):
        current_mode = stat.S_IMODE(os.fstat(descriptor).st_mode)
        os.fchmod(descriptor, current_mode & ~0o222)
    elif os.name == "nt":  # pragma: win32 cover - exercised by Windows tests
        import ctypes
        import msvcrt

        class _FileBasicInfo(ctypes.Structure):
            _fields_ = (
                ("creation_time", ctypes.c_longlong),
                ("last_access_time", ctypes.c_longlong),
                ("last_write_time", ctypes.c_longlong),
                ("change_time", ctypes.c_longlong),
                ("file_attributes", ctypes.c_uint32),
            )

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = ctypes.c_void_p(msvcrt.get_osfhandle(descriptor))
        basic = _FileBasicInfo()
        if not kernel32.GetFileInformationByHandleEx(
            handle, 0, ctypes.byref(basic), ctypes.sizeof(basic)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        basic.file_attributes |= 0x1  # FILE_ATTRIBUTE_READONLY
        if not kernel32.SetFileInformationByHandle(
            handle, 0, ctypes.byref(basic), ctypes.sizeof(basic)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
    else:  # pragma: no cover - supported targets expose one branch above
        raise OSError("descriptor-safe comparison sealing is unavailable")


def _seal_publication_descriptor(stream: Any, identity: tuple[int, int]) -> None:
    descriptor = stream.fileno()
    _make_publication_descriptor_read_only(descriptor)
    os.fsync(descriptor)
    descriptor_state = os.fstat(descriptor)
    if (
        not stat.S_ISREG(descriptor_state.st_mode)
        or (int(descriptor_state.st_dev), int(descriptor_state.st_ino)) != identity
        or stat.S_IMODE(descriptor_state.st_mode)
        & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    ):
        raise Comparison2DPublicationError(
            "comparison descriptor could not be immutably sealed"
        )


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise Comparison2DPublicationError(
            f"cannot open publication directory for fsync: {exc}"
        ) from exc
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publication_parent_identities(
    paths: Sequence[Path],
) -> tuple[tuple[int, int], ...]:
    try:
        return _directory_identities(paths)
    except Comparison2DValidationError as exc:
        raise Comparison2DPublicationError(str(exc)) from exc


def _publication_destination(
    output_path: str | Path,
) -> tuple[Path, tuple[Path, ...], tuple[tuple[int, int], ...]]:
    try:
        requested = Path(os.path.abspath(os.fspath(output_path)))
    except (OSError, TypeError, ValueError) as exc:
        raise Comparison2DPublicationError(
            f"comparison publication destination is invalid: {exc}"
        ) from exc
    if requested.name in {"", ".", ".."}:
        raise Comparison2DPublicationError(
            "comparison publication requires a concrete destination filename"
        )
    if os.name == "nt":
        windows_leaf = requested.name
        windows_stem = windows_leaf.split(".", 1)[0].upper()
        reserved_stems = {
            "CON",
            "PRN",
            "AUX",
            "NUL",
            *(f"COM{index}" for index in range(1, 10)),
            *(f"LPT{index}" for index in range(1, 10)),
        }
        if (
            ":" in windows_leaf
            or windows_leaf.rstrip(" .") != windows_leaf
            or windows_stem in reserved_stems
        ):
            raise Comparison2DPublicationError(
                "comparison publication requires a normal Windows file path"
            )
    ensure_real_directory(
        requested.parent,
        error_type=Comparison2DPublicationError,
        role="comparison publication parent",
    )
    try:
        parent = _direct_absolute_path(
            requested.parent,
            role="comparison publication parent",
        )
    except Comparison2DValidationError as exc:
        raise Comparison2DPublicationError(str(exc)) from exc
    destination = parent / requested.name
    parent_paths = (parent, *parent.parents)
    identities = _publication_parent_identities(parent_paths)
    try:
        final_parent = _direct_absolute_path(
            parent,
            role="comparison publication parent",
        )
    except Comparison2DValidationError as exc:
        raise Comparison2DPublicationError(str(exc)) from exc
    if (
        _normal_path(final_parent) != _normal_path(parent)
        or _publication_parent_identities(parent_paths) != identities
    ):
        raise Comparison2DPublicationError(
            "comparison publication parent changed during validation"
        )
    return destination, parent_paths, identities


def _publication_state_signature(state: os.stat_result) -> tuple[int, ...]:
    return (
        int(state.st_dev),
        int(state.st_ino),
        int(state.st_size),
        int(state.st_mtime_ns),
        int(state.st_mode),
        int(state.st_nlink),
    )


def _require_sealed_publication_state(
    state: os.stat_result,
    *,
    identity: tuple[int, int],
    size_bytes: int,
    role: str,
) -> None:
    if (
        not stat.S_ISREG(state.st_mode)
        or (int(state.st_dev), int(state.st_ino)) != identity
        or int(state.st_nlink) != 1
        or int(state.st_size) != size_bytes
        or stat.S_IMODE(state.st_mode) & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    ):
        raise Comparison2DPublicationError(f"{role} is not the exact sealed inode")


def _capture_publication_receipt(
    destination: Path,
    *,
    expected_identity: tuple[int, int],
    expected_payload: bytes,
    parent_paths: Sequence[Path],
    parent_identities: tuple[tuple[int, int], ...],
) -> PublishedComparison2D:
    """Reopen once, read twice, and derive a receipt from that stable fd."""
    if _publication_parent_identities(parent_paths) != parent_identities:
        raise Comparison2DPublicationError(
            "comparison publication parent changed before receipt capture"
        )
    try:
        before_path = os.lstat(destination)
    except OSError as exc:
        raise Comparison2DPublicationError(
            f"cannot inspect published comparison for receipt: {exc}"
        ) from exc
    _require_sealed_publication_state(
        before_path,
        identity=expected_identity,
        size_bytes=len(expected_payload),
        role="published comparison before receipt",
    )
    descriptor: int | None = None
    try:
        descriptor = open_verified_publication(destination)
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            opened = os.fstat(stream.fileno())
            _require_sealed_publication_state(
                opened,
                identity=expected_identity,
                size_bytes=len(expected_payload),
                role="reopened comparison receipt descriptor",
            )
            if _publication_state_signature(opened) != _publication_state_signature(
                before_path
            ):
                raise Comparison2DPublicationError(
                    "published comparison changed while its receipt descriptor opened"
                )
            first_before = os.fstat(stream.fileno())
            first_payload = _read_exact_fd(stream, len(expected_payload))
            first_after = os.fstat(stream.fileno())
            second_payload = _read_exact_fd(stream, len(expected_payload))
            second_after = os.fstat(stream.fileno())
    except Comparison2DPublicationError:
        raise
    except OSError as exc:
        raise Comparison2DPublicationError(
            f"cannot capture stable published comparison receipt: {exc}"
        ) from exc
    finally:
        if descriptor is not None:
            close_publication_descriptor(
                descriptor,
                suppress_errors=sys.exception() is not None,
            )
    try:
        after_path = os.lstat(destination)
    except OSError as exc:
        raise Comparison2DPublicationError(
            f"cannot re-inspect published comparison for receipt: {exc}"
        ) from exc
    states = (opened, first_before, first_after, second_after, after_path)
    for index, state in enumerate(states):
        _require_sealed_publication_state(
            state,
            identity=expected_identity,
            size_bytes=len(expected_payload),
            role=f"published comparison receipt state {index}",
        )
    if len({_publication_state_signature(state) for state in states}) != 1:
        raise Comparison2DPublicationError(
            "published comparison metadata changed during receipt capture"
        )
    if _publication_parent_identities(parent_paths) != parent_identities:
        raise Comparison2DPublicationError(
            "comparison publication parent changed during receipt capture"
        )
    if first_payload != expected_payload or second_payload != expected_payload:
        raise Comparison2DPublicationError(
            "stable published comparison receipt bytes differ from the publication"
        )
    receipt_sha256 = hashlib.sha256(second_payload).hexdigest()
    return PublishedComparison2D(
        path=destination,
        sha256=receipt_sha256,
        size_bytes=len(second_payload),
    )


def publish_comparison_2d(
    comparison: Mapping[str, Any], output_path: str | Path
) -> PublishedComparison2D:
    """Publish once and return a receipt from a stable reopened descriptor.

    The final path itself is created with ``O_EXCL`` and read-only mode.  No
    rollback ever performs a pathname-based chmod or unlink: after creation,
    any failure leaves the exact sealed inode for explicit operator review.
    """
    destination, parent_paths, parent_identities = _publication_destination(output_path)
    if os.path.lexists(destination):
        raise Comparison2DPublicationError(f"refusing to overwrite {destination}")
    payload = canonical_json_bytes(comparison)
    descriptor: int | None = None
    identity: tuple[int, int] | None = None
    try:
        try:
            # The exclusive descriptor denies concurrent Windows writers and
            # remains the only writable capability until descriptor sealing.
            descriptor = open_exclusive_publication(destination)
            with os.fdopen(descriptor, "w+b", closefd=False) as stream:
                initial = os.fstat(stream.fileno())
                if not stat.S_ISREG(initial.st_mode) or int(initial.st_nlink) != 1:
                    raise Comparison2DPublicationError(
                        "comparison destination is not a unique regular file"
                    )
                identity = (int(initial.st_dev), int(initial.st_ino))
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
                if _read_exact_fd(stream, len(payload)) != payload:
                    raise Comparison2DPublicationError(
                        "comparison descriptor verification failed"
                    )
                _seal_publication_descriptor(stream, identity)
                stable = os.fstat(stream.fileno())
                if (
                    (int(stable.st_dev), int(stable.st_ino)) != identity
                    or stable.st_size != len(payload)
                    or int(stable.st_nlink) != 1
                    or stat.S_IMODE(stable.st_mode)
                    & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
                ):
                    raise Comparison2DPublicationError(
                        "comparison destination changed while it was written"
                    )
        except FileExistsError as exc:
            raise Comparison2DPublicationError(
                f"publication race: destination appeared: {destination}"
            ) from exc
        except Comparison2DPublicationError:
            raise
        except OSError as exc:
            raise Comparison2DPublicationError(
                f"cannot publish comparison {destination}: {exc}"
            ) from exc
        assert identity is not None
        if _publication_parent_identities(parent_paths) != parent_identities:
            raise Comparison2DPublicationError(
                "comparison publication parent changed while it was written"
            )
        try:
            final = os.lstat(destination)
        except OSError as exc:
            raise Comparison2DPublicationError(
                f"cannot inspect published comparison path: {exc}"
            ) from exc
        _require_sealed_publication_state(
            final,
            identity=identity,
            size_bytes=len(payload),
            role="published comparison path",
        )
        _fsync_directory(destination.parent)
        if _publication_parent_identities(parent_paths) != parent_identities:
            raise Comparison2DPublicationError(
                "comparison publication parent changed after directory sync"
            )
    finally:
        if descriptor is not None:
            try:
                state = os.fstat(descriptor)
                if stat.S_ISREG(state.st_mode):
                    _make_publication_descriptor_read_only(descriptor)
                    os.fsync(descriptor)
            except OSError:
                pass
            finally:
                close_publication_descriptor(
                    descriptor,
                    suppress_errors=sys.exception() is not None,
                )
    assert identity is not None
    return _capture_publication_receipt(
        destination,
        expected_identity=identity,
        expected_payload=payload,
        parent_paths=parent_paths,
        parent_identities=parent_identities,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the locked post-score PIMSR 2-D three-method comparison"
    )
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--prediction-lock", required=True, type=Path)
    parser.add_argument("--prediction-lock-sha256", required=True)
    parser.add_argument("--post-score-manifest", required=True, type=Path)
    parser.add_argument("--post-score-manifest-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the post-score comparator using externally pinned inputs only."""
    args = _parser().parse_args(argv)
    report = compare_evaluations_2d(
        args.preregistration,
        args.prediction_lock,
        args.post_score_manifest,
        expected_preregistration_sha256=args.preregistration_sha256,
        expected_prediction_lock_sha256=args.prediction_lock_sha256,
        expected_post_score_manifest_sha256=args.post_score_manifest_sha256,
    )
    receipt = publish_comparison_2d(report, args.output)
    print(
        f"published {receipt.path} sha256={receipt.sha256} "
        f"size_bytes={receipt.size_bytes}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
