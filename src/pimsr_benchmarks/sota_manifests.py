"""Canonical, fail-closed manifests for executable SOTA comparisons.

The registry describes what *could* be run.  These manifests record what was
planned, materialized and actually executed.  A pinned artifact, a passing
adapter smoke test and a complete benchmark are deliberately different states.
Schema version 1 is intentionally capped at adapter smoke: it cannot prove the
typed payload, campaign-wide seed, runtime and metric contracts needed for a
complete benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

from .sota import DEFAULT_REGISTRY_PATH, load_registry

MANIFEST_SCHEMA_VERSION = 1
EXPERIMENT_SCHEMA = "pimsr-sota-experiment"
OBSERVATION_SCHEMA = "pimsr-sota-observations"
PREDICTION_SCHEMA = "pimsr-sota-predictions"
RUN_SCHEMA = "pimsr-sota-run"

EXECUTION_STATUSES = frozenset(
    {"artifact_pinned", "adapter_smoke_passed", "benchmark_complete"}
)

_SCHEMA_V1_COMPLETION_ERROR = (
    "schema version 1 cannot claim benchmark_complete; it does not prove typed "
    "HDF5/NPZ arrays, the complete ordered seed campaign, or complete runtime "
    "and metric provenance"
)

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_CONTAINER_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

_ARTIFACT_KEYS = {"path", "sha256", "size_bytes", "media_type"}
_SOURCE_KEYS = {"repository_url", "commit", "artifact", "dirty_tree"}
_GROUP_KEYS = {"family_id", "base_model_id", "noise_id", "sample_ids"}
_SPLIT_KEYS = {"split_id", "groups"}
_PHYSICAL_KEYS = {
    "dimensionality",
    "coordinate_system",
    "handedness",
    "vertical_positive",
    "rotation_degrees",
    "axes",
    "axis_units",
    "components",
    "component_units",
    "spectral_axis",
    "spectral_unit",
    "spectral_order",
    "phase_unit",
    "phase_convention",
    "time_convention",
    "resistivity_unit",
    "model_parameter",
    "model_parameter_unit",
}
_COMMAND_KEYS = {"prepare", "execute", "evaluate"}

_EXPERIMENT_KEYS = {
    "schema",
    "schema_version",
    "experiment_id",
    "protocol_version",
    "registry",
    "method_id",
    "dataset_id",
    "dimensionality",
    "track",
    "execution_status",
    "dataset_artifact_state",
    "source",
    "commands",
    "split",
    "physical_contract",
    "observation_manifest",
    "random_seeds",
    "created_utc",
    "notes",
}
_OBSERVATION_KEYS = {
    "schema",
    "schema_version",
    "manifest_id",
    "dataset_id",
    "dimensionality",
    "artifact_state",
    "split",
    "physical_contract",
    "payload",
    "array_sha256",
    "created_utc",
}
_OBSERVATION_ARRAY_KEYS = {
    "station_coordinates",
    "spectral_axis",
    "observations",
    "uncertainties",
    "valid_mask",
    "truth",
}
_PREDICTION_KEYS = {
    "schema",
    "schema_version",
    "manifest_id",
    "experiment_id",
    "method_id",
    "dataset_id",
    "dimensionality",
    "track",
    "execution_status",
    "experiment_manifest",
    "observation_manifest",
    "split",
    "physical_contract",
    "outputs",
    "created_utc",
}
_RUN_KEYS = {
    "schema",
    "schema_version",
    "run_id",
    "method_id",
    "dataset_id",
    "dimensionality",
    "track",
    "execution_status",
    "experiment_manifest",
    "observation_manifest",
    "prediction_manifest",
    "source",
    "command",
    "working_directory",
    "inputs",
    "outputs",
    "environment",
    "execution",
    "resources",
}
_ENVIRONMENT_KEYS = {
    "python_version",
    "platform",
    "lockfile_sha256",
    "package_inventory_sha256",
    "container_image_digest",
}
_EXECUTION_KEYS = {
    "started_utc",
    "finished_utc",
    "exit_status",
    "exit_code",
    "converged",
    "warnings",
}
_RESOURCE_KEYS = {
    "wall_time_s",
    "cpu_time_s",
    "peak_host_ram_bytes",
    "peak_accelerator_memory_bytes",
    "cpu_count",
    "accelerator_count",
    "threads",
    "mpi_ranks",
    "precision",
    "energy_joules",
}

_DIMENSION_AXES = {
    "1d": ("z",),
    "2d": ("x", "z"),
    "3d": ("x", "y", "z"),
}
_COMPONENT_UNITS = {
    "Zxx": "ohm",
    "Zxy": "ohm",
    "Zyx": "ohm",
    "Zyy": "ohm",
    "Tx": "dimensionless",
    "Ty": "dimensionless",
    "log10_rho_te": "log10_ohm_m",
    "phase_te": "degree",
    "log10_rho_tm": "log10_ohm_m",
    "phase_tm": "degree",
}


class ManifestValidationError(ValueError):
    """Raised when a SOTA execution manifest violates its schema or context."""


class ManifestPublicationError(RuntimeError):
    """Raised when atomic, no-overwrite publication cannot be completed."""


class SnapshotError(RuntimeError):
    """Raised when a source file changes or a content-addressed snapshot conflicts."""


def _error(path: str, message: str) -> ManifestValidationError:
    return ManifestValidationError(f"{path}: {message}")


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(path, "must be an object")
    return value


def _array(value: Any, path: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise _error(path, "must be an array")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    missing = expected - value.keys()
    unknown = value.keys() - expected
    if missing:
        raise _error(path, f"missing keys: {', '.join(sorted(missing))}")
    if unknown:
        raise _error(path, f"unknown keys: {', '.join(sorted(unknown))}")


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise _error(path, "must be a non-empty, trimmed string")
    return value


def _identifier(value: Any, path: str) -> str:
    text = _string(value, path)
    if not _ID_RE.fullmatch(text):
        raise _error(path, "must be a lowercase identifier")
    return text


def _enum(value: Any, allowed: set[str] | frozenset[str], path: str) -> str:
    text = _string(value, path)
    if text not in allowed:
        raise _error(path, f"must be one of {sorted(allowed)}")
    return text


def _integer(value: Any, path: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise _error(path, "must be an integer")
    if minimum is not None and value < minimum:
        raise _error(path, f"must be >= {minimum}")
    return value


def _number(value: Any, path: str, *, nullable: bool = False) -> float | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _error(path, "must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise _error(path, "must be a finite non-negative number")
    return result


def _finite_number(value: Any, path: str, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _error(path, "must be a finite number")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result < maximum:
        raise _error(path, f"must be in [{minimum}, {maximum})")
    return result


def _boolean(value: Any, path: str) -> bool:
    if type(value) is not bool:
        raise _error(path, "must be a boolean")
    return value


def _sha256(value: Any, path: str) -> str:
    text = _string(value, path)
    if not _SHA256_RE.fullmatch(text):
        raise _error(path, "must be a lowercase 64-character SHA-256")
    return text


def _git_sha(value: Any, path: str) -> str:
    text = _string(value, path)
    if not _GIT_SHA_RE.fullmatch(text):
        raise _error(path, "must be a lowercase full 40-character Git commit")
    return text


def _https_url(value: Any, path: str) -> str:
    text = _string(value, path)
    parsed = urlparse(text)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username
        or parsed.password
    ):
        raise _error(path, "must be an absolute HTTPS URL without credentials")
    return text


def _timestamp(value: Any, path: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    text = _string(value, path)
    if not _UTC_RE.fullmatch(text):
        raise _error(path, "must be UTC YYYY-MM-DDTHH:MM:SSZ")
    try:
        datetime.fromisoformat(text.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise _error(path, "must be a valid UTC timestamp") from exc
    return text


def _portable_path(value: Any, path: str) -> str:
    text = _string(value, path)
    if "\\" in text:
        raise _error(path, "must use portable forward slashes")
    parsed = PurePosixPath(text)
    if (
        parsed.is_absolute()
        or parsed.as_posix() != text
        or any(part in {"", ".", ".."} or ":" in part for part in parsed.parts)
    ):
        raise _error(path, "must be a normalized relative path without '..'")
    return text


def _string_array(
    value: Any,
    path: str,
    *,
    nonempty: bool = True,
    identifiers: bool = False,
    unique: bool = True,
) -> list[str]:
    values = _array(value, path)
    if nonempty and not values:
        raise _error(path, "must not be empty")
    check = _identifier if identifiers else _string
    result = [check(item, f"{path}[{index}]") for index, item in enumerate(values)]
    if unique and len(result) != len(set(result)):
        raise _error(path, "must not contain duplicates")
    return result


def _nullable_sha256(value: Any, path: str) -> str | None:
    return None if value is None else _sha256(value, path)


def _validate_artifact(value: Any, path: str) -> Mapping[str, Any]:
    artifact = _mapping(value, path)
    _exact_keys(artifact, _ARTIFACT_KEYS, path)
    _portable_path(artifact["path"], f"{path}.path")
    _sha256(artifact["sha256"], f"{path}.sha256")
    _integer(artifact["size_bytes"], f"{path}.size_bytes", minimum=0)
    media_type = _string(artifact["media_type"], f"{path}.media_type")
    if "/" not in media_type or any(char.isspace() for char in media_type):
        raise _error(f"{path}.media_type", "must be an IANA-style media type")
    return artifact


def _validate_source(value: Any, path: str, *, require_artifact: bool) -> None:
    source = _mapping(value, path)
    _exact_keys(source, _SOURCE_KEYS, path)
    _https_url(source["repository_url"], f"{path}.repository_url")
    _git_sha(source["commit"], f"{path}.commit")
    _boolean(source["dirty_tree"], f"{path}.dirty_tree")
    if source["artifact"] is None:
        if require_artifact:
            raise _error(
                f"{path}.artifact", "is required after the artifact-pinned stage"
            )
    else:
        _validate_artifact(source["artifact"], f"{path}.artifact")


def _validate_split(value: Any, path: str) -> None:
    split = _mapping(value, path)
    _exact_keys(split, _SPLIT_KEYS, path)
    _identifier(split["split_id"], f"{path}.split_id")
    groups = _array(split["groups"], f"{path}.groups")
    if not groups:
        raise _error(f"{path}.groups", "must not be empty")
    group_ids: set[tuple[str, str, str]] = set()
    sample_ids: set[str] = set()
    for index, raw_group in enumerate(groups):
        group_path = f"{path}.groups[{index}]"
        group = _mapping(raw_group, group_path)
        _exact_keys(group, _GROUP_KEYS, group_path)
        family = _identifier(group["family_id"], f"{group_path}.family_id")
        base = _identifier(group["base_model_id"], f"{group_path}.base_model_id")
        noise = _identifier(group["noise_id"], f"{group_path}.noise_id")
        key = (family, base, noise)
        if key in group_ids:
            raise _error(group_path, "duplicate family/base/noise group")
        group_ids.add(key)
        samples = _string_array(
            group["sample_ids"], f"{group_path}.sample_ids", identifiers=True
        )
        overlap = sample_ids.intersection(samples)
        if overlap:
            raise _error(
                group_path, f"sample IDs reused across groups: {sorted(overlap)}"
            )
        sample_ids.update(samples)


def _validate_physical_contract(value: Any, path: str, dimensionality: str) -> None:
    contract = _mapping(value, path)
    _exact_keys(contract, _PHYSICAL_KEYS, path)
    if contract["dimensionality"] != dimensionality:
        raise _error(f"{path}.dimensionality", "must match the manifest")
    axes = _string_array(contract["axes"], f"{path}.axes")
    expected_axes = _DIMENSION_AXES[dimensionality]
    if tuple(axes) != expected_axes:
        raise _error(f"{path}.axes", f"must be {list(expected_axes)}")
    _string(contract["coordinate_system"], f"{path}.coordinate_system")
    if contract["handedness"] != "right_handed":
        raise _error(f"{path}.handedness", "must be 'right_handed'")
    if contract["vertical_positive"] != "down":
        raise _error(f"{path}.vertical_positive", "must be 'down'")
    _finite_number(
        contract["rotation_degrees"],
        f"{path}.rotation_degrees",
        minimum=-360.0,
        maximum=360.0,
    )
    axis_units = _mapping(contract["axis_units"], f"{path}.axis_units")
    _exact_keys(axis_units, set(axes), f"{path}.axis_units")
    if any(axis_units[axis] != "m" for axis in axes):
        raise _error(f"{path}.axis_units", "all physical axes must use metres")

    components = _string_array(contract["components"], f"{path}.components")
    unknown_components = set(components) - _COMPONENT_UNITS.keys()
    if unknown_components:
        raise _error(f"{path}.components", f"unsupported: {sorted(unknown_components)}")
    component_units = _mapping(contract["component_units"], f"{path}.component_units")
    _exact_keys(component_units, set(components), f"{path}.component_units")
    for component in components:
        expected = _COMPONENT_UNITS[component]
        if component_units[component] != expected:
            raise _error(f"{path}.component_units.{component}", f"must be {expected!r}")

    spectral_axis = _enum(
        contract["spectral_axis"], {"frequency", "period"}, f"{path}.spectral_axis"
    )
    expected_unit = "Hz" if spectral_axis == "frequency" else "s"
    if contract["spectral_unit"] != expected_unit:
        raise _error(f"{path}.spectral_unit", f"must be {expected_unit!r}")
    _enum(
        contract["spectral_order"], {"ascending", "descending"}, f"{path}.spectral_order"
    )
    if contract["phase_unit"] != "degree":
        raise _error(f"{path}.phase_unit", "must be 'degree'")
    if contract["phase_convention"] != "degrees_modulo_180_[0,180)":
        raise _error(
            f"{path}.phase_convention",
            "must be 'degrees_modulo_180_[0,180)'",
        )
    if contract["time_convention"] != "exp(+i_omega_t)":
        raise _error(f"{path}.time_convention", "must be 'exp(+i_omega_t)'")
    if contract["resistivity_unit"] != "ohm_m":
        raise _error(f"{path}.resistivity_unit", "must be 'ohm_m'")
    if contract["model_parameter"] != "log10_resistivity":
        raise _error(f"{path}.model_parameter", "must be 'log10_resistivity'")
    if contract["model_parameter_unit"] != "log10_ohm_m":
        raise _error(f"{path}.model_parameter_unit", "must be 'log10_ohm_m'")


def _validate_schema_header(value: Mapping[str, Any], schema: str, path: str) -> None:
    if value["schema"] != schema:
        raise _error(f"{path}.schema", f"must be {schema!r}")
    if type(value["schema_version"]) is not int:
        raise _error(f"{path}.schema_version", "must be an integer")
    if value["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise _error(
            f"{path}.schema_version",
            f"unsupported legacy/unknown version {value['schema_version']!r}",
        )


def _registry_indexes(
    registry: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    methods = {method["id"]: method for method in registry["methods"]}
    datasets = {dataset["id"]: dataset for dataset in registry["datasets"]}
    return methods, datasets


def _validate_registry_compatibility(
    value: Mapping[str, Any], path: str, registry: Mapping[str, Any]
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    methods, datasets = _registry_indexes(registry)
    method_id = _identifier(value["method_id"], f"{path}.method_id")
    dataset_id = _identifier(value["dataset_id"], f"{path}.dataset_id")
    if method_id not in methods:
        raise _error(f"{path}.method_id", "is absent from the selected registry")
    if dataset_id not in datasets:
        raise _error(f"{path}.dataset_id", "is absent from the selected registry")
    dimensionality = _enum(
        value["dimensionality"], set(_DIMENSION_AXES), f"{path}.dimensionality"
    )
    method = methods[method_id]
    dataset = datasets[dataset_id]
    if dimensionality not in method["dimensionality"]:
        raise _error(f"{path}.dimensionality", "is unsupported by the method")
    if dimensionality != dataset["dimensionality"]:
        raise _error(f"{path}.dimensionality", "does not match the dataset")
    track = _string(value["track"], f"{path}.track")
    if track not in method["tracks"]:
        raise _error(f"{path}.track", "is unsupported by the method")
    return method, dataset


def validate_experiment(
    value: Any,
    registry: Mapping[str, Any],
    *,
    registry_sha256: str,
    observation: Mapping[str, Any] | None = None,
) -> None:
    """Validate an immutable experiment preregistration."""
    experiment = _mapping(value, "experiment")
    _exact_keys(experiment, _EXPERIMENT_KEYS, "experiment")
    _validate_schema_header(experiment, EXPERIMENT_SCHEMA, "experiment")
    _identifier(experiment["experiment_id"], "experiment.experiment_id")
    if experiment["protocol_version"] != "1.0":
        raise _error("experiment.protocol_version", "must be '1.0'")
    registry_ref = _validate_artifact(experiment["registry"], "experiment.registry")
    if registry_ref["sha256"] != _sha256(registry_sha256, "registry_sha256"):
        raise _error("experiment.registry.sha256", "does not match selected registry")

    method, dataset = _validate_registry_compatibility(experiment, "experiment", registry)
    dimensionality = experiment["dimensionality"]
    if experiment["execution_status"] != "artifact_pinned":
        raise _error(
            "experiment.execution_status",
            "preregistrations must start at 'artifact_pinned'",
        )
    artifact_state = _enum(
        experiment["dataset_artifact_state"],
        {"conditional_not_materialized", "materialized"},
        "experiment.dataset_artifact_state",
    )
    _validate_source(experiment["source"], "experiment.source", require_artifact=False)
    if experiment["source"]["commit"] != method["source"]["ref"]["resolved_commit"]:
        raise _error("experiment.source.commit", "does not match registry method commit")
    if experiment["source"]["repository_url"] != method["source"]["repository_url"]:
        raise _error("experiment.source.repository_url", "does not match registry method")
    if experiment["source"]["dirty_tree"]:
        raise _error("experiment.source.dirty_tree", "pinned experiments must be clean")

    commands = _mapping(experiment["commands"], "experiment.commands")
    _exact_keys(commands, _COMMAND_KEYS, "experiment.commands")
    for name in sorted(_COMMAND_KEYS):
        _string_array(commands[name], f"experiment.commands.{name}", unique=False)
    _validate_split(experiment["split"], "experiment.split")
    _validate_physical_contract(
        experiment["physical_contract"], "experiment.physical_contract", dimensionality
    )
    seeds = _array(experiment["random_seeds"], "experiment.random_seeds")
    parsed_seeds = [
        _integer(seed, f"experiment.random_seeds[{index}]", minimum=0)
        for index, seed in enumerate(seeds)
    ]
    if len(parsed_seeds) < 5 or len(parsed_seeds) != len(set(parsed_seeds)):
        raise _error("experiment.random_seeds", "must contain at least five unique seeds")
    _timestamp(experiment["created_utc"], "experiment.created_utc")
    _string_array(experiment["notes"], "experiment.notes", nonempty=False, unique=False)

    observation_ref = experiment["observation_manifest"]
    if artifact_state == "conditional_not_materialized":
        if dataset["status"] != "conditional":
            raise _error(
                "experiment.dataset_artifact_state",
                "only a conditional registry dataset may be not materialized",
            )
        if observation_ref is not None or observation is not None:
            raise _error(
                "experiment.observation_manifest",
                "must be null until dataset bytes are materialized and checksummed",
            )
    else:
        if observation_ref is None:
            raise _error(
                "experiment.observation_manifest", "is required when materialized"
            )
        if (
            dataset.get("generator") is not None
            and dataset["generator"]["source_status"] != "artifact_pinned"
        ):
            raise _error(
                "experiment.dataset_artifact_state",
                "generated data cannot materialize before its public generator is pinned",
            )
        _validate_artifact(observation_ref, "experiment.observation_manifest")
        if observation is not None:
            _compare_observation_to_experiment(observation, experiment)


def validate_observation_manifest(value: Any, registry: Mapping[str, Any]) -> None:
    """Validate a materialized, checksummed observation contract."""
    manifest = _mapping(value, "observations")
    _exact_keys(manifest, _OBSERVATION_KEYS, "observations")
    _validate_schema_header(manifest, OBSERVATION_SCHEMA, "observations")
    _identifier(manifest["manifest_id"], "observations.manifest_id")
    datasets = _registry_indexes(registry)[1]
    dataset_id = _identifier(manifest["dataset_id"], "observations.dataset_id")
    if dataset_id not in datasets:
        raise _error("observations.dataset_id", "is absent from the selected registry")
    dimensionality = _enum(
        manifest["dimensionality"], set(_DIMENSION_AXES), "observations.dimensionality"
    )
    if dimensionality != datasets[dataset_id]["dimensionality"]:
        raise _error("observations.dimensionality", "does not match registry dataset")
    if (
        datasets[dataset_id].get("generator") is not None
        and datasets[dataset_id]["generator"]["source_status"] != "artifact_pinned"
    ):
        raise _error(
            "observations.dataset_id",
            "generated observations require a pinned public generator implementation",
        )
    if manifest["artifact_state"] != "materialized":
        raise _error(
            "observations.artifact_state",
            "an observation manifest only exists for materialized bytes",
        )
    _validate_split(manifest["split"], "observations.split")
    _validate_physical_contract(
        manifest["physical_contract"], "observations.physical_contract", dimensionality
    )
    _validate_artifact(manifest["payload"], "observations.payload")
    hashes = _mapping(manifest["array_sha256"], "observations.array_sha256")
    _exact_keys(hashes, _OBSERVATION_ARRAY_KEYS, "observations.array_sha256")
    for name in sorted(_OBSERVATION_ARRAY_KEYS - {"truth"}):
        _sha256(hashes[name], f"observations.array_sha256.{name}")
    truth = datasets[dataset_id]["truth"]
    if hashes["truth"] is None:
        if truth not in {"withheld", "not_applicable", "unknown"}:
            raise _error(
                "observations.array_sha256.truth",
                "materialized synthetic truth requires a SHA-256",
            )
    else:
        _sha256(hashes["truth"], "observations.array_sha256.truth")
        if truth == "not_applicable":
            raise _error(
                "observations.array_sha256.truth", "field data cannot claim truth"
            )
    _timestamp(manifest["created_utc"], "observations.created_utc")


def _compare_observation_to_experiment(
    observation: Mapping[str, Any], experiment: Mapping[str, Any]
) -> None:
    for key in ("dataset_id", "dimensionality", "split", "physical_contract"):
        if observation[key] != experiment[key]:
            raise _error(f"experiment.{key}", "does not match observation manifest")


def validate_prediction_manifest(
    value: Any,
    registry: Mapping[str, Any],
    *,
    experiment: Mapping[str, Any] | None = None,
    observation: Mapping[str, Any] | None = None,
) -> None:
    """Validate predictions without promoting a smoke test to a benchmark result."""
    manifest = _mapping(value, "predictions")
    _exact_keys(manifest, _PREDICTION_KEYS, "predictions")
    _validate_schema_header(manifest, PREDICTION_SCHEMA, "predictions")
    _identifier(manifest["manifest_id"], "predictions.manifest_id")
    _identifier(manifest["experiment_id"], "predictions.experiment_id")
    _validate_registry_compatibility(manifest, "predictions", registry)
    dimensionality = manifest["dimensionality"]
    execution_status = _enum(
        manifest["execution_status"],
        {"adapter_smoke_passed", "benchmark_complete"},
        "predictions.execution_status",
    )
    if execution_status == "benchmark_complete":
        raise _error("predictions.execution_status", _SCHEMA_V1_COMPLETION_ERROR)
    _validate_artifact(manifest["experiment_manifest"], "predictions.experiment_manifest")
    _validate_artifact(
        manifest["observation_manifest"], "predictions.observation_manifest"
    )
    _validate_split(manifest["split"], "predictions.split")
    _validate_physical_contract(
        manifest["physical_contract"], "predictions.physical_contract", dimensionality
    )
    outputs = _array(manifest["outputs"], "predictions.outputs")
    if not outputs:
        raise _error("predictions.outputs", "must not be empty")
    roles: set[str] = set()
    for index, raw_output in enumerate(outputs):
        output = _validate_artifact(raw_output, f"predictions.outputs[{index}]")
        role = _identifier(
            output["path"].split("/")[-1].split(".")[0],
            f"predictions.outputs[{index}].path",
        )
        if role in roles:
            raise _error("predictions.outputs", "output basenames must be unique")
        roles.add(role)
    _timestamp(manifest["created_utc"], "predictions.created_utc")

    if experiment is not None:
        for key in (
            "method_id",
            "dataset_id",
            "dimensionality",
            "track",
            "split",
            "physical_contract",
        ):
            if manifest[key] != experiment[key]:
                raise _error(f"predictions.{key}", "does not match experiment")
        if manifest["experiment_id"] != experiment["experiment_id"]:
            raise _error("predictions.experiment_id", "does not match experiment")
        if experiment["dataset_artifact_state"] != "materialized":
            raise _error(
                "predictions.execution_status",
                "conditional, unmaterialized datasets cannot produce predictions",
            )
    if observation is not None:
        for key in ("dataset_id", "dimensionality", "split", "physical_contract"):
            if manifest[key] != observation[key]:
                raise _error(f"predictions.{key}", "does not match observations")


def _validate_artifact_array(
    value: Any, path: str, *, nonempty: bool
) -> list[Mapping[str, Any]]:
    raw_items = _array(value, path)
    if nonempty and not raw_items:
        raise _error(path, "must not be empty")
    items = [
        _validate_artifact(item, f"{path}[{index}]")
        for index, item in enumerate(raw_items)
    ]
    paths = [item["path"] for item in items]
    if len(paths) != len(set(paths)):
        raise _error(path, "must not repeat paths")
    return items


def _artifact_identity(artifact: Mapping[str, Any]) -> tuple[str, int, str]:
    """Return the path-independent identity of one validated artifact.

    Artifact paths are relative to the manifest that contains them, so the
    same immutable bytes can legitimately have different paths in an
    observation, prediction and run manifest.  SHA-256, byte size and media
    type together provide the cross-manifest identity used by promotion gates.
    """
    return (
        str(artifact["sha256"]),
        int(artifact["size_bytes"]),
        str(artifact["media_type"]),
    )


def _artifact_identity_multiset(
    artifacts: Sequence[Mapping[str, Any]],
) -> list[tuple[str, int, str]]:
    """Return a sorted multiset without collapsing equal-byte output roles."""
    return sorted(_artifact_identity(artifact) for artifact in artifacts)


def validate_run_manifest(
    value: Any,
    registry: Mapping[str, Any],
    *,
    experiment: Mapping[str, Any] | None = None,
    observation: Mapping[str, Any] | None = None,
    prediction: Mapping[str, Any] | None = None,
) -> None:
    """Validate execution provenance and enforce status promotion gates."""
    manifest = _mapping(value, "run")
    _exact_keys(manifest, _RUN_KEYS, "run")
    _validate_schema_header(manifest, RUN_SCHEMA, "run")
    _identifier(manifest["run_id"], "run.run_id")
    method, _dataset = _validate_registry_compatibility(manifest, "run", registry)
    status = _enum(
        manifest["execution_status"], EXECUTION_STATUSES, "run.execution_status"
    )
    if status == "benchmark_complete":
        raise _error("run.execution_status", _SCHEMA_V1_COMPLETION_ERROR)
    _validate_artifact(manifest["experiment_manifest"], "run.experiment_manifest")
    if manifest["observation_manifest"] is not None:
        _validate_artifact(manifest["observation_manifest"], "run.observation_manifest")
    if manifest["prediction_manifest"] is not None:
        _validate_artifact(manifest["prediction_manifest"], "run.prediction_manifest")
    _validate_source(
        manifest["source"], "run.source", require_artifact=status != "artifact_pinned"
    )
    if manifest["source"]["commit"] != method["source"]["ref"]["resolved_commit"]:
        raise _error("run.source.commit", "does not match registry method commit")
    if manifest["source"]["repository_url"] != method["source"]["repository_url"]:
        raise _error("run.source.repository_url", "does not match registry method")
    command = _string_array(
        manifest["command"],
        "run.command",
        nonempty=status != "artifact_pinned",
        unique=False,
    )
    _portable_path(manifest["working_directory"], "run.working_directory")
    inputs = _validate_artifact_array(
        manifest["inputs"], "run.inputs", nonempty=status != "artifact_pinned"
    )
    outputs = _validate_artifact_array(
        manifest["outputs"], "run.outputs", nonempty=status != "artifact_pinned"
    )

    environment = _mapping(manifest["environment"], "run.environment")
    _exact_keys(environment, _ENVIRONMENT_KEYS, "run.environment")
    _string(environment["python_version"], "run.environment.python_version")
    _string(environment["platform"], "run.environment.platform")
    _sha256(environment["lockfile_sha256"], "run.environment.lockfile_sha256")
    _sha256(
        environment["package_inventory_sha256"],
        "run.environment.package_inventory_sha256",
    )
    container = environment["container_image_digest"]
    if container is not None and (
        not isinstance(container, str) or not _CONTAINER_DIGEST_RE.fullmatch(container)
    ):
        raise _error(
            "run.environment.container_image_digest",
            "must be null or sha256:<64 lowercase hex>",
        )

    execution = _mapping(manifest["execution"], "run.execution")
    _exact_keys(execution, _EXECUTION_KEYS, "run.execution")
    exit_status = _enum(
        execution["exit_status"],
        {"not_run", "succeeded", "failed", "timeout"},
        "run.execution.exit_status",
    )
    started = _timestamp(
        execution["started_utc"], "run.execution.started_utc", nullable=True
    )
    finished = _timestamp(
        execution["finished_utc"], "run.execution.finished_utc", nullable=True
    )
    exit_code = execution["exit_code"]
    if exit_code is not None:
        _integer(exit_code, "run.execution.exit_code")
    converged = execution["converged"]
    if converged is not None:
        _boolean(converged, "run.execution.converged")
    _string_array(
        execution["warnings"], "run.execution.warnings", nonempty=False, unique=False
    )
    if (started is None) != (finished is None):
        raise _error(
            "run.execution", "start and finish timestamps must be both set or null"
        )
    if started is not None and finished < started:
        raise _error("run.execution.finished_utc", "must not precede start")
    if exit_status == "not_run":
        if any(item is not None for item in (started, finished, exit_code, converged)):
            raise _error("run.execution", "not_run requires null execution fields")
        if command:
            raise _error("run.command", "must be empty for not_run")
    elif started is None or (exit_status != "timeout" and exit_code is None):
        raise _error(
            "run.execution",
            "executed runs require timestamps and non-timeout runs require an exit code",
        )
    if exit_status == "succeeded" and exit_code != 0:
        raise _error("run.execution.exit_code", "successful runs require exit code 0")
    if exit_status == "failed" and exit_code == 0:
        raise _error(
            "run.execution.exit_code", "failed runs require a non-zero exit code"
        )
    if exit_status in {"failed", "timeout"} and status != "artifact_pinned":
        raise _error(
            "run.execution_status",
            "failed/timeout runs cannot claim a passed adapter or complete benchmark",
        )

    resources = _mapping(manifest["resources"], "run.resources")
    _exact_keys(resources, _RESOURCE_KEYS, "run.resources")
    wall_time = _number(resources["wall_time_s"], "run.resources.wall_time_s")
    for name in ("cpu_time_s", "energy_joules"):
        _number(resources[name], f"run.resources.{name}", nullable=True)
    for name in (
        "peak_host_ram_bytes",
        "peak_accelerator_memory_bytes",
        "accelerator_count",
        "mpi_ranks",
    ):
        item = resources[name]
        if item is not None:
            _integer(item, f"run.resources.{name}", minimum=0)
    for name in ("cpu_count", "threads"):
        item = resources[name]
        if item is not None:
            _integer(item, f"run.resources.{name}", minimum=1)
    _enum(
        resources["precision"],
        {"float16", "bfloat16", "float32", "float64", "mixed", "not_applicable"},
        "run.resources.precision",
    )
    if exit_status == "not_run" and wall_time != 0.0:
        raise _error("run.resources.wall_time_s", "not_run requires zero wall time")
    if exit_status != "not_run" and wall_time == 0.0:
        raise _error(
            "run.resources.wall_time_s", "executed runs require positive wall time"
        )

    if experiment is not None:
        for key in ("method_id", "dataset_id", "dimensionality", "track"):
            if manifest[key] != experiment[key]:
                raise _error(f"run.{key}", "does not match experiment")
        expected_command = list(experiment["commands"]["execute"])
        if command and command != expected_command:
            raise _error(
                "run.command",
                "does not match experiment.commands.execute",
            )
    if observation is not None and experiment is not None:
        _compare_observation_to_experiment(observation, experiment)
    if prediction is not None:
        if prediction["execution_status"] != status:
            raise _error("run.execution_status", "does not match prediction manifest")
        for key in ("method_id", "dataset_id", "dimensionality", "track"):
            if manifest[key] != prediction[key]:
                raise _error(f"run.{key}", "does not match prediction manifest")

    if status == "artifact_pinned":
        if manifest["prediction_manifest"] is not None or prediction is not None:
            raise _error(
                "run.prediction_manifest", "must be null at artifact-pinned stage"
            )
    else:
        if exit_status != "succeeded":
            raise _error("run.execution.exit_status", "promoted statuses require success")
        if converged is False:
            raise _error(
                "run.execution.converged",
                "adapter_smoke_passed cannot claim an explicitly non-converged run",
            )
        if (
            manifest["observation_manifest"] is None
            or manifest["prediction_manifest"] is None
        ):
            raise _error(
                "run.execution_status",
                "promotion requires observation and prediction manifest references",
            )
        if any(item is not None for item in (experiment, observation, prediction)) and (
            experiment is None or observation is None or prediction is None
        ):
            raise _error(
                "run.execution_status", "referenced manifest context is incomplete"
            )
        if manifest["source"]["dirty_tree"]:
            raise _error("run.source.dirty_tree", "promoted runs require a clean tree")
        if not inputs or not outputs:
            raise _error("run", "promoted runs require hashed inputs and outputs")
        if experiment is not None and command != list(experiment["commands"]["execute"]):
            raise _error(
                "run.command",
                "promoted runs must exactly match experiment.commands.execute",
            )
        if observation is not None:
            observation_identity = _artifact_identity(observation["payload"])
            input_identities = {
                _artifact_identity(artifact) for artifact in inputs
            }
            if observation_identity not in input_identities:
                raise _error(
                    "run.inputs",
                    "must include the exact observation payload artifact",
                )
        if prediction is not None:
            expected_outputs = _artifact_identity_multiset(prediction["outputs"])
            actual_outputs = _artifact_identity_multiset(outputs)
            if actual_outputs != expected_outputs:
                raise _error(
                    "run.outputs",
                    "must exactly match prediction manifest output identities",
                )


def validate_manifest(
    value: Any,
    registry: Mapping[str, Any],
    *,
    registry_sha256: str,
    experiment: Mapping[str, Any] | None = None,
    observation: Mapping[str, Any] | None = None,
    prediction: Mapping[str, Any] | None = None,
) -> None:
    """Dispatch validation by the exact schema identifier."""
    manifest = _mapping(value, "manifest")
    schema = manifest.get("schema")
    if schema == EXPERIMENT_SCHEMA:
        validate_experiment(
            manifest,
            registry,
            registry_sha256=registry_sha256,
            observation=observation,
        )
    elif schema == OBSERVATION_SCHEMA:
        validate_observation_manifest(manifest, registry)
    elif schema == PREDICTION_SCHEMA:
        validate_prediction_manifest(
            manifest, registry, experiment=experiment, observation=observation
        )
    elif schema == RUN_SCHEMA:
        validate_run_manifest(
            manifest,
            registry,
            experiment=experiment,
            observation=observation,
            prediction=prediction,
        )
    else:
        raise _error(
            "manifest.schema",
            f"unknown or legacy schema {schema!r}; expected a versioned SOTA manifest",
        )


def canonical_json_bytes(value: Any) -> bytes:
    """Encode one canonical JSON document (UTF-8, sorted keys, no NaN)."""
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ManifestValidationError(
            f"manifest is not canonical JSON data: {exc}"
        ) from exc
    return (encoded + "\n").encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Return SHA-256 of the canonical JSON representation."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash a regular file without following a changing symbolic-link contract."""
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_json(data: bytes, path: Path) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ManifestValidationError(f"{path}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ManifestValidationError(f"{path}: non-finite JSON constant {value!r}")

    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestValidationError(f"cannot decode manifest {path}: {exc}") from exc


def _resolve_record(record: Mapping[str, Any], root: Path) -> Path:
    return root.joinpath(*PurePosixPath(record["path"]).parts)


def _verify_artifact(record: Mapping[str, Any], root: Path, path: str) -> Path:
    target = _resolve_record(record, root)
    current = root
    for part in PurePosixPath(record["path"]).parts:
        current = current / part
        if current.is_symlink():
            raise _error(path, "referenced artifacts must not use symbolic links")
    try:
        target.resolve(strict=False).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise _error(path, "referenced artifact escapes its manifest directory") from exc
    if target.is_symlink():
        raise _error(path, "referenced artifacts must not be symbolic links")
    try:
        info = target.stat()
    except OSError as exc:
        raise _error(path, f"cannot stat referenced artifact {target}: {exc}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise _error(path, "referenced artifact must be a regular file")
    if info.st_size != record["size_bytes"]:
        raise _error(path, "referenced artifact size does not match")
    if sha256_file(target) != record["sha256"]:
        raise _error(path, "referenced artifact SHA-256 does not match")
    return target


def _reference_fields(manifest: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    schema = manifest["schema"]
    result: list[tuple[str, Mapping[str, Any]]] = []
    if schema == EXPERIMENT_SCHEMA:
        result.append(("experiment.registry", manifest["registry"]))
        if manifest["source"]["artifact"] is not None:
            result.append(("experiment.source.artifact", manifest["source"]["artifact"]))
        if manifest["observation_manifest"] is not None:
            result.append(
                ("experiment.observation_manifest", manifest["observation_manifest"])
            )
    elif schema == OBSERVATION_SCHEMA:
        result.append(("observations.payload", manifest["payload"]))
    elif schema == PREDICTION_SCHEMA:
        result.extend(
            [
                ("predictions.experiment_manifest", manifest["experiment_manifest"]),
                ("predictions.observation_manifest", manifest["observation_manifest"]),
            ]
        )
        result.extend(
            (f"predictions.outputs[{index}]", item)
            for index, item in enumerate(manifest["outputs"])
        )
    elif schema == RUN_SCHEMA:
        result.append(("run.experiment_manifest", manifest["experiment_manifest"]))
        for name in ("observation_manifest", "prediction_manifest"):
            if manifest[name] is not None:
                result.append((f"run.{name}", manifest[name]))
        if manifest["source"]["artifact"] is not None:
            result.append(("run.source.artifact", manifest["source"]["artifact"]))
        result.extend(
            (f"run.inputs[{index}]", item)
            for index, item in enumerate(manifest["inputs"])
        )
        result.extend(
            (f"run.outputs[{index}]", item)
            for index, item in enumerate(manifest["outputs"])
        )
    return result


def load_manifest(
    path: str | Path,
    *,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    verify_artifacts: bool = True,
    _seen: set[Path] | None = None,
) -> dict[str, Any]:
    """Load canonical JSON, recursively verify references and validate semantics."""
    manifest_path = Path(path).resolve()
    seen = set() if _seen is None else _seen
    if manifest_path in seen:
        raise ManifestValidationError(f"cyclic manifest reference: {manifest_path}")
    seen.add(manifest_path)
    try:
        raw = manifest_path.read_bytes()
    except OSError as exc:
        raise ManifestValidationError(
            f"cannot load manifest {manifest_path}: {exc}"
        ) from exc
    value = _decode_json(raw, manifest_path)
    if raw != canonical_json_bytes(value):
        raise ManifestValidationError(
            f"{manifest_path}: JSON is not in canonical sorted, compact form"
        )

    registry_path = Path(registry_path).resolve()
    registry = load_registry(registry_path)
    registry_hash = sha256_file(registry_path)
    schema = value.get("schema") if isinstance(value, Mapping) else None
    root = manifest_path.parent
    referenced: dict[str, dict[str, Any]] = {}
    validate_manifest(value, registry, registry_sha256=registry_hash)
    if verify_artifacts and isinstance(value, Mapping):
        for field, record in _reference_fields(value):
            _validate_artifact(record, field)
            target = _verify_artifact(record, root, field)
            if field.endswith("experiment_manifest"):
                referenced["experiment"] = load_manifest(
                    target,
                    registry_path=registry_path,
                    verify_artifacts=True,
                    _seen=seen,
                )
            elif field.endswith("observation_manifest"):
                referenced["observation"] = load_manifest(
                    target,
                    registry_path=registry_path,
                    verify_artifacts=True,
                    _seen=seen,
                )
            elif field.endswith("prediction_manifest"):
                referenced["prediction"] = load_manifest(
                    target,
                    registry_path=registry_path,
                    verify_artifacts=True,
                    _seen=seen,
                )
    try:
        validate_manifest(
            value,
            registry,
            registry_sha256=registry_hash,
            experiment=referenced.get("experiment"),
            observation=referenced.get("observation"),
            prediction=referenced.get("prediction"),
        )
    finally:
        seen.remove(manifest_path)
    if schema == EXPERIMENT_SCHEMA and value["registry"]["sha256"] != registry_hash:
        raise _error("experiment.registry.sha256", "does not match registry bytes")
    return dict(value)


def publish_manifest(
    value: Any,
    path: str | Path,
    *,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> Path:
    """Validate and atomically publish canonical JSON without overwriting."""
    registry_path = Path(registry_path).resolve()
    registry = load_registry(registry_path)
    validate_manifest(
        value,
        registry,
        registry_sha256=sha256_file(registry_path),
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    if destination.exists():
        raise ManifestPublicationError(f"refusing to overwrite {destination}")
    if partial.exists():
        raise ManifestPublicationError(f"refusing stale partial file {partial}")
    payload = canonical_json_bytes(value)
    try:
        with partial.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if partial.read_bytes() != payload:
            raise ManifestPublicationError("partial manifest verification failed")
        load_manifest(partial, registry_path=registry_path, verify_artifacts=True)
        try:
            os.link(partial, destination)
        except FileExistsError as exc:
            raise ManifestPublicationError(
                f"publication race: destination appeared: {destination}"
            ) from exc
        except OSError as exc:
            raise ManifestPublicationError(
                f"cannot publish {destination}: {exc}"
            ) from exc
    finally:
        try:
            partial.unlink()
        except FileNotFoundError:
            pass
    return destination


def snapshot_file(
    source: str | Path,
    snapshot_directory: str | Path,
    *,
    media_type: str = "application/octet-stream",
) -> dict[str, Any]:
    """Publish an immutable ``<sha256>.blob`` snapshot and return its record.

    The source identity is checked before and after copying.  A concurrent
    publisher may win only with identical bytes; a conflicting existing object
    is treated as corruption rather than overwritten.
    """
    source_path = Path(source)
    snapshot_dir = Path(snapshot_directory)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    _validate_artifact(
        {
            "path": "placeholder.blob",
            "sha256": "0" * 64,
            "size_bytes": 0,
            "media_type": media_type,
        },
        "snapshot",
    )
    if source_path.is_symlink() or not source_path.is_file():
        raise SnapshotError(f"source must be a regular non-symlink file: {source_path}")
    before = source_path.stat()
    partial = snapshot_dir / f".snapshot-{uuid.uuid4().hex}.part"
    digest = hashlib.sha256()
    copied = 0
    try:
        with source_path.open("rb") as reader, partial.open("xb") as writer:
            while chunk := reader.read(1024 * 1024):
                writer.write(chunk)
                digest.update(chunk)
                copied += len(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        after = source_path.stat()
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if identity_before != identity_after or copied != before.st_size:
            raise SnapshotError(f"source changed while snapshotting: {source_path}")
        sha = digest.hexdigest()
        destination = snapshot_dir / f"{sha}.blob"
        if destination.is_symlink():
            raise SnapshotError(f"snapshot target must not be a symlink: {destination}")
        if destination.exists():
            if destination.stat().st_size != copied or sha256_file(destination) != sha:
                raise SnapshotError(f"corrupt/conflicting snapshot exists: {destination}")
        else:
            try:
                os.link(partial, destination)
            except FileExistsError:
                if destination.is_symlink():
                    raise SnapshotError(
                        f"snapshot publication race created a symlink: {destination}"
                    )
                if (
                    destination.stat().st_size != copied
                    or sha256_file(destination) != sha
                ):
                    raise SnapshotError(
                        f"snapshot publication race conflict: {destination}"
                    )
            except OSError as exc:
                raise SnapshotError(
                    f"cannot publish snapshot {destination}: {exc}"
                ) from exc
        record = {
            "path": destination.name,
            "sha256": sha,
            "size_bytes": copied,
            "media_type": media_type,
        }
        _validate_artifact(record, "snapshot")
        return record
    finally:
        try:
            partial.unlink()
        except FileNotFoundError:
            pass


def _load_noncanonical_json(path: Path) -> Any:
    try:
        return _decode_json(path.read_bytes(), path)
    except OSError as exc:
        raise ManifestValidationError(f"cannot load manifest {path}: {exc}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    """CLI for validation, canonical publication and content snapshots."""
    parser = argparse.ArgumentParser(prog="pimsr-sota-manifest")
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    validate_parser = subparsers.add_parser(
        "validate", help="validate a canonical manifest"
    )
    validate_parser.add_argument("manifest")
    validate_parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))

    publish_parser = subparsers.add_parser("publish", help="publish canonical JSON once")
    publish_parser.add_argument("source_json")
    publish_parser.add_argument("destination")
    publish_parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))

    snapshot_parser = subparsers.add_parser("snapshot", help="snapshot a file by SHA-256")
    snapshot_parser.add_argument("source")
    snapshot_parser.add_argument("directory")
    snapshot_parser.add_argument("--media-type", default="application/octet-stream")

    args = parser.parse_args(argv)
    if args.command_name == "validate":
        manifest = load_manifest(args.manifest, registry_path=args.registry)
        print(
            f"valid {manifest['schema']} schema={manifest['schema_version']} "
            f"sha256={sha256_file(args.manifest)}"
        )
    elif args.command_name == "publish":
        value = _load_noncanonical_json(Path(args.source_json))
        destination = publish_manifest(
            value, args.destination, registry_path=args.registry
        )
        print(f"published {destination} sha256={sha256_file(destination)}")
    else:
        record = snapshot_file(args.source, args.directory, media_type=args.media_type)
        print(canonical_json_bytes(record).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
