"""Fail-closed validation for the versioned SOTA method registry."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

SCHEMA_VERSION = 2
_PACKAGE_REGISTRY_PATH = Path(__file__).resolve().parent / "data" / "sota_methods.json"
_PACKAGE_PROTOCOL_PATH = Path(__file__).resolve().parent / "data" / "SOTA_PROTOCOL.md"
_REPOSITORY_REGISTRY_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "sota_methods.json"
)
_REPOSITORY_PROTOCOL_PATH = (
    Path(__file__).resolve().parents[2] / "docs" / "SOTA_PROTOCOL.md"
)
DEFAULT_REGISTRY_PATH = (
    _PACKAGE_REGISTRY_PATH
    if _PACKAGE_REGISTRY_PATH.is_file()
    else _REPOSITORY_REGISTRY_PATH
)
DEFAULT_PROTOCOL_PATH = (
    _PACKAGE_PROTOCOL_PATH
    if _PACKAGE_PROTOCOL_PATH.is_file()
    else _REPOSITORY_PROTOCOL_PATH
)

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DOI_RE = re.compile(r"^10\.\d{4,9}/[-._;()/:a-z0-9]+$", re.IGNORECASE)
_SPDX_RE = re.compile(r"^[A-Za-z0-9.+-]+$")
_GIT_ARTIFACT_RE = re.compile(r"^git:([0-9a-f]{40})(?::[A-Za-z0-9._/+:-]+)?$")
_RESOLVED_ARTIFACT_RE = re.compile(r"^(?:tag|version):([^@]+)@([0-9a-f]{40})$")

_ROOT_KEYS = {
    "schema_version",
    "registry_id",
    "as_of_date",
    "protocol_document",
    "methods",
    "datasets",
}
_METHOD_KEYS = {
    "id",
    "name",
    "dimensionality",
    "status",
    "tracks",
    "publications",
    "source",
    "artifacts",
    "license",
    "caveats",
}
_PUBLICATION_KEYS = {"title", "doi", "url"}
_SOURCE_KEYS = {"repository_url", "ref", "release"}
_REF_KEYS = {"kind", "value", "resolved_commit"}
_ARTIFACT_KEYS = {"kind", "availability", "url", "immutable_id", "caveat"}
_LICENSE_KEYS = {"spdx", "status", "caveat"}
_DATASET_KEYS = {
    "id",
    "name",
    "dimensionality",
    "kind",
    "status",
    "source_url",
    "artifact_availability",
    "truth",
    "checksum_policy",
    "license",
    "caveats",
}
_GENERATED_DATASET_KEYS = _DATASET_KEYS | {"generator"}
_GENERATOR_KEYS = {
    "schema_version",
    "producer_id",
    "repository_url",
    "source_commit",
    "source_status",
    "entrypoint",
    "materialization_status",
    "command_template",
    "campaign_count",
    "samples_per_campaign",
    "seed_policy",
    "seed_commitment_encoding",
    "seed_commitment_sha256",
    "sample_id_policy",
    "sample_id_key_commitment_sha256",
    "start_index",
    "grouping_contract",
    "physical_contract",
    "required_snapshot_roles",
}
_GENERATOR_PHYSICAL_KEYS = {
    "dataset_schema",
    "dataset_schema_version",
    "axes",
    "axis_unit",
    "handedness",
    "vertical_positive",
    "rotation_degrees",
    "components",
    "representations",
    "frequency_unit",
    "phase_unit",
    "phase_convention",
    "time_convention",
    "resistivity_unit",
}

ALLOWED_DIMENSIONALITIES = frozenset({"1d", "2d", "3d"})
ALLOWED_METHOD_STATUSES = frozenset(
    {"reproducible_first_wave", "reference_only", "paper_only"}
)
ALLOWED_TRACKS = frozenset({"frozen_artifact", "common_retrain", "refinement"})
ALLOWED_ARTIFACT_AVAILABILITY = frozenset({"available", "metadata_only", "unavailable"})
ALLOWED_LICENSE_STATUSES = frozenset(
    {"verified_repository_file", "reported_no_license_file", "unknown"}
)
ALLOWED_DATASET_KINDS = frozenset({"synthetic", "field"})
ALLOWED_DATASET_STATUSES = frozenset({"eligible", "conditional"})
ALLOWED_DATASET_AVAILABILITY = frozenset(
    {"downloadable", "official_page", "restricted", "not_yet_materialized"}
)
ALLOWED_TRUTH_STATES = frozenset(
    {"available", "withheld", "not_applicable", "unknown", "not_yet_materialized"}
)
_FLOATING_REFS = frozenset({"head", "latest", "main", "master", "trunk"})


class RegistryValidationError(ValueError):
    """Raised when a registry violates the benchmark contract."""


def _error(path: str, message: str) -> RegistryValidationError:
    return RegistryValidationError(f"{path}: {message}")


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(path, "must be an object")
    return value


def _sequence(value: Any, path: str) -> Sequence[Any]:
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
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise _error(path, "must be a non-empty, trimmed string")
    return value


def _nullable_string(value: Any, path: str) -> str | None:
    if value is None:
        return None
    return _string(value, path)


def _enum(value: Any, allowed: frozenset[str], path: str) -> str:
    text = _string(value, path)
    if text not in allowed:
        raise _error(path, f"must be one of {sorted(allowed)}")
    return text


def _identifier(value: Any, path: str) -> str:
    text = _string(value, path)
    if not _ID_RE.fullmatch(text):
        raise _error(path, "must match ^[a-z0-9][a-z0-9_-]*$")
    return text


def _https_url(value: Any, path: str) -> str:
    text = _string(value, path)
    parsed = urlparse(text)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise _error(path, "must be an absolute HTTPS URL without credentials")
    return text


def _doi(value: Any, path: str) -> str:
    text = _string(value, path)
    if not _DOI_RE.fullmatch(text):
        raise _error(path, "must be a bare DOI beginning with '10.'")
    return text


def _sha(value: Any, path: str) -> str:
    text = _string(value, path)
    if not _SHA_RE.fullmatch(text):
        raise _error(path, "must be a full, lowercase 40-character Git SHA")
    return text


def _no_floating_ref(value: str, path: str) -> None:
    if value.casefold() in _FLOATING_REFS:
        raise _error(path, "floating refs (main/master/latest/HEAD/trunk) are forbidden")


def _string_list(value: Any, path: str, *, nonempty: bool = True) -> list[str]:
    items = _sequence(value, path)
    if nonempty and not items:
        raise _error(path, "must not be empty")
    result = [_string(item, f"{path}[{index}]") for index, item in enumerate(items)]
    if len(result) != len(set(result)):
        raise _error(path, "must not contain duplicates")
    return result


def _sha256(value: Any, path: str) -> str:
    result = _string(value, path)
    if not _SHA256_RE.fullmatch(result):
        raise _error(path, "must be a 64-character lowercase SHA-256 digest")
    return result


def _validate_ref(value: Any, path: str) -> None:
    ref = _mapping(value, path)
    _exact_keys(ref, _REF_KEYS, path)
    kind = _enum(ref["kind"], frozenset({"commit", "tag", "version"}), f"{path}.kind")
    ref_value = _string(ref["value"], f"{path}.value")
    resolved = _sha(ref["resolved_commit"], f"{path}.resolved_commit")
    _no_floating_ref(ref_value, f"{path}.value")
    if kind == "commit":
        commit = _sha(ref_value, f"{path}.value")
        if commit != resolved:
            raise _error(path, "commit value and resolved_commit must be identical")


def _validate_publication(value: Any, path: str) -> None:
    publication = _mapping(value, path)
    _exact_keys(publication, _PUBLICATION_KEYS, path)
    _string(publication["title"], f"{path}.title")
    doi = _doi(publication["doi"], f"{path}.doi")
    url = _https_url(publication["url"], f"{path}.url")
    if url.casefold() != f"https://doi.org/{doi}".casefold():
        raise _error(path, "url must be the canonical https://doi.org/<doi> URL")


def _validate_source(value: Any, path: str) -> None:
    source = _mapping(value, path)
    _exact_keys(source, _SOURCE_KEYS, path)
    _https_url(source["repository_url"], f"{path}.repository_url")
    _validate_ref(source["ref"], f"{path}.ref")
    if source["release"] is not None:
        _validate_ref(source["release"], f"{path}.release")
        if source["release"]["kind"] == "commit":
            raise _error(f"{path}.release.kind", "a release must be a tag or version")


def _validate_immutable_id(value: Any, path: str) -> None:
    text = _string(value, path)
    if text.startswith("doi:"):
        _doi(text.removeprefix("doi:"), path)
        return
    match = _GIT_ARTIFACT_RE.fullmatch(text)
    if match:
        _sha(match.group(1), path)
        return
    match = _RESOLVED_ARTIFACT_RE.fullmatch(text)
    if match:
        _no_floating_ref(match.group(1), path)
        _sha(match.group(2), path)
        return
    raise _error(
        path,
        "must be doi:<doi>, git:<full-sha>[:path], or tag/version:<ref>@<full-sha>",
    )


def _validate_artifact(value: Any, path: str) -> str:
    artifact = _mapping(value, path)
    _exact_keys(artifact, _ARTIFACT_KEYS, path)
    _identifier(artifact["kind"], f"{path}.kind")
    availability = _enum(
        artifact["availability"],
        ALLOWED_ARTIFACT_AVAILABILITY,
        f"{path}.availability",
    )
    _string(artifact["caveat"], f"{path}.caveat")

    url = artifact["url"]
    immutable_id = artifact["immutable_id"]
    if availability == "unavailable":
        if url is not None or immutable_id is not None:
            raise _error(path, "unavailable artifacts require null url and immutable_id")
    else:
        _https_url(url, f"{path}.url")
        _validate_immutable_id(immutable_id, f"{path}.immutable_id")
    return availability


def _validate_license(value: Any, path: str) -> None:
    license_info = _mapping(value, path)
    _exact_keys(license_info, _LICENSE_KEYS, path)
    status = _enum(license_info["status"], ALLOWED_LICENSE_STATUSES, f"{path}.status")
    spdx = _nullable_string(license_info["spdx"], f"{path}.spdx")
    _string(license_info["caveat"], f"{path}.caveat")
    if status == "verified_repository_file":
        if spdx is None or not _SPDX_RE.fullmatch(spdx):
            raise _error(f"{path}.spdx", "a verified license requires an SPDX id")
    elif spdx is not None:
        raise _error(f"{path}.spdx", "unverified or unknown licenses require null SPDX")


def _validate_method(value: Any, index: int) -> str:
    path = f"methods[{index}]"
    method = _mapping(value, path)
    _exact_keys(method, _METHOD_KEYS, path)
    method_id = _identifier(method["id"], f"{path}.id")
    _string(method["name"], f"{path}.name")

    dimensions = _string_list(method["dimensionality"], f"{path}.dimensionality")
    invalid_dimensions = set(dimensions) - ALLOWED_DIMENSIONALITIES
    if invalid_dimensions:
        raise _error(
            f"{path}.dimensionality",
            f"unsupported values: {sorted(invalid_dimensions)}",
        )

    status = _enum(method["status"], ALLOWED_METHOD_STATUSES, f"{path}.status")
    tracks = _string_list(method["tracks"], f"{path}.tracks", nonempty=False)
    invalid_tracks = set(tracks) - ALLOWED_TRACKS
    if invalid_tracks:
        raise _error(f"{path}.tracks", f"unsupported values: {sorted(invalid_tracks)}")

    publications = _sequence(method["publications"], f"{path}.publications")
    if not publications and method_id != "pimsr":
        raise _error(f"{path}.publications", "must not be empty")
    for publication_index, publication in enumerate(publications):
        _validate_publication(publication, f"{path}.publications[{publication_index}]")

    artifacts = _sequence(method["artifacts"], f"{path}.artifacts")
    if not artifacts:
        raise _error(f"{path}.artifacts", "must not be empty")
    availability = [
        _validate_artifact(artifact, f"{path}.artifacts[{artifact_index}]")
        for artifact_index, artifact in enumerate(artifacts)
    ]

    if status == "paper_only":
        if method["source"] is not None:
            raise _error(f"{path}.source", "paper-only methods require null source")
        if tracks:
            raise _error(f"{path}.tracks", "paper-only methods cannot enter run tracks")
        if any(item != "unavailable" for item in availability):
            raise _error(
                f"{path}.artifacts",
                "paper-only methods may only register unavailable artifacts",
            )
    else:
        if method["source"] is None:
            raise _error(f"{path}.source", "runnable methods require a pinned source")
        _validate_source(method["source"], f"{path}.source")
        if not tracks:
            raise _error(f"{path}.tracks", "runnable methods require at least one track")
        if "available" not in availability:
            raise _error(
                f"{path}.artifacts", "runnable methods need an available artifact"
            )

    _validate_license(method["license"], f"{path}.license")
    _string_list(method["caveats"], f"{path}.caveats")
    return method_id


def _validate_dataset(value: Any, index: int) -> str:
    path = f"datasets[{index}]"
    dataset = _mapping(value, path)
    expected_keys = (
        _GENERATED_DATASET_KEYS
        if dataset.get("id") == "pimsr_generated_2d_v1"
        else _DATASET_KEYS
    )
    _exact_keys(dataset, expected_keys, path)
    dataset_id = _identifier(dataset["id"], f"{path}.id")
    _string(dataset["name"], f"{path}.name")
    _enum(
        dataset["dimensionality"],
        ALLOWED_DIMENSIONALITIES,
        f"{path}.dimensionality",
    )
    kind = _enum(dataset["kind"], ALLOWED_DATASET_KINDS, f"{path}.kind")
    _enum(dataset["status"], ALLOWED_DATASET_STATUSES, f"{path}.status")
    _https_url(dataset["source_url"], f"{path}.source_url")
    _enum(
        dataset["artifact_availability"],
        ALLOWED_DATASET_AVAILABILITY,
        f"{path}.artifact_availability",
    )
    truth = _enum(dataset["truth"], ALLOWED_TRUTH_STATES, f"{path}.truth")
    if dataset["checksum_policy"] != "sha256_required_before_run":
        raise _error(
            f"{path}.checksum_policy",
            "must be 'sha256_required_before_run'",
        )
    if kind == "field" and truth != "not_applicable":
        raise _error(f"{path}.truth", "field datasets require 'not_applicable'")
    if kind == "synthetic" and truth == "not_applicable":
        raise _error(f"{path}.truth", "synthetic datasets require a truth state")
    _validate_license(dataset["license"], f"{path}.license")
    _string_list(dataset["caveats"], f"{path}.caveats")
    if dataset_id == "pimsr_generated_2d_v1":
        _validate_generated_dataset(dataset, path)
    return dataset_id


def _validate_generated_dataset(dataset: Mapping[str, Any], path: str) -> None:
    if dataset["dimensionality"] != "2d" or dataset["kind"] != "synthetic":
        raise _error(path, "the PIMSR generated benchmark must be 2-D synthetic")
    if (
        dataset["status"] != "conditional"
        or dataset["artifact_availability"] != "not_yet_materialized"
        or dataset["truth"] != "not_yet_materialized"
    ):
        raise _error(
            path,
            "unmaterialized PIMSR data must remain conditional/not_yet_materialized",
        )
    generator = _mapping(dataset["generator"], f"{path}.generator")
    _exact_keys(generator, _GENERATOR_KEYS, f"{path}.generator")
    if type(generator["schema_version"]) is not int or generator["schema_version"] != 2:
        raise _error(f"{path}.generator.schema_version", "must be integer 2")
    if generator["producer_id"] != "pimsr-forward":
        raise _error(f"{path}.generator.producer_id", "must be 'pimsr-forward'")
    _https_url(generator["repository_url"], f"{path}.generator.repository_url")
    source_commit = _sha(generator["source_commit"], f"{path}.generator.source_commit")
    expected_source_url = f"{generator['repository_url']}/tree/{source_commit}"
    if dataset["source_url"] != expected_source_url:
        raise _error(
            f"{path}.source_url",
            "must point to the exact pinned generator commit",
        )
    _enum(
        generator["source_status"],
        frozenset({"artifact_pinned", "pre_release_implementation_not_public"}),
        f"{path}.generator.source_status",
    )
    if generator["entrypoint"] != "pimsr-forward-dataset2d":
        raise _error(f"{path}.generator.entrypoint", "must be 'pimsr-forward-dataset2d'")
    if generator["materialization_status"] != "seed_committed_not_materialized":
        raise _error(
            f"{path}.generator.materialization_status",
            "must be 'seed_committed_not_materialized'",
        )
    campaign_count = generator["campaign_count"]
    samples_per_campaign = generator["samples_per_campaign"]
    start_index = generator["start_index"]
    for name, number, expected in (
        ("campaign_count", campaign_count, 5),
        ("samples_per_campaign", samples_per_campaign, 500),
        ("start_index", start_index, 0),
    ):
        if type(number) is not int or number != expected:
            raise _error(f"{path}.generator.{name}", f"must be integer {expected}")
    if generator["seed_policy"] != "operator_withheld_until_predictions_locked":
        raise _error(
            f"{path}.generator.seed_policy",
            "must keep seeds operator-withheld until predictions are locked",
        )
    if (
        generator["seed_commitment_encoding"]
        != "utf8-canonical-json-int64-array-no-newline-v1"
    ):
        raise _error(
            f"{path}.generator.seed_commitment_encoding",
            "must use the canonical hidden-seed commitment encoding",
        )
    _sha256(
        generator["seed_commitment_sha256"],
        f"{path}.generator.seed_commitment_sha256",
    )
    if generator["sample_id_policy"] != "hmac_sha256_opaque_nonnegative_int64_v1":
        raise _error(
            f"{path}.generator.sample_id_policy",
            "must use opaque HMAC-derived sample identifiers",
        )
    _sha256(
        generator["sample_id_key_commitment_sha256"],
        f"{path}.generator.sample_id_key_commitment_sha256",
    )
    command = _string_list(
        generator["command_template"], f"{path}.generator.command_template"
    )
    expected_command = [
        "pimsr-forward-dataset2d",
        "--out",
        "<operator-only-new-output.h5>",
        "--n",
        str(samples_per_campaign),
        "--seed",
        "<operator-withheld-seed>",
        "--start-index",
        str(start_index),
    ]
    if command != expected_command:
        raise _error(
            f"{path}.generator.command_template",
            "does not match the declared campaign size and withheld-seed policy",
        )
    if (
        generator["grouping_contract"]
        != "withheld_scenario/opaque_sample_id/noise_realization_v2"
    ):
        raise _error(
            f"{path}.generator.grouping_contract",
            "must use the hidden grouped-split v2 contract",
        )
    physical = _mapping(
        generator["physical_contract"], f"{path}.generator.physical_contract"
    )
    _exact_keys(physical, _GENERATOR_PHYSICAL_KEYS, f"{path}.generator.physical_contract")
    expected_physical = {
        "dataset_schema": "pimsr-mt-2d",
        "dataset_schema_version": 2,
        "axes": ["x", "z"],
        "axis_unit": "m",
        "handedness": "right_handed",
        "vertical_positive": "down",
        "rotation_degrees": 0.0,
        "components": ["Zyx", "Zxy"],
        "representations": ["log10_rho", "phase"],
        "frequency_unit": "Hz",
        "phase_unit": "degree",
        "phase_convention": "degrees_modulo_180_[0,180)",
        "time_convention": "exp(+i_omega_t)",
        "resistivity_unit": "ohm_m",
    }
    if dict(physical) != expected_physical:
        raise _error(
            f"{path}.generator.physical_contract",
            "must match the canonical PIMSR 2-D physical contract",
        )
    roles = _string_list(
        generator["required_snapshot_roles"],
        f"{path}.generator.required_snapshot_roles",
    )
    if set(roles) != {
        "coordinates",
        "operator_scoring_manifest",
        "operator_source_dataset",
        "public_observation_manifest",
        "public_observations",
        "split_groups",
        "withheld_truth",
    }:
        raise _error(
            f"{path}.generator.required_snapshot_roles",
            "must separate public method inputs from operator-only source and truth",
        )


def _ensure_unique(values: Sequence[str], path: str) -> None:
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise _error(path, f"duplicate ids: {', '.join(duplicates)}")


def validate_registry(value: Any) -> None:
    """Validate a decoded registry and reject omissions or ambiguous references."""

    registry = _mapping(value, "registry")
    _exact_keys(registry, _ROOT_KEYS, "registry")
    if type(registry["schema_version"]) is not int:  # bool is not a schema integer
        raise _error("registry.schema_version", "must be an integer")
    if registry["schema_version"] != SCHEMA_VERSION:
        raise _error(
            "registry.schema_version",
            f"unsupported version {registry['schema_version']!r}; expected {SCHEMA_VERSION}",
        )
    if registry["registry_id"] != "pimsr-sota-methods":
        raise _error("registry.registry_id", "must be 'pimsr-sota-methods'")
    as_of = _string(registry["as_of_date"], "registry.as_of_date")
    try:
        if date.fromisoformat(as_of).isoformat() != as_of:
            raise ValueError
    except ValueError as exc:
        raise _error("registry.as_of_date", "must be an ISO YYYY-MM-DD date") from exc
    if registry["protocol_document"] != "docs/SOTA_PROTOCOL.md":
        raise _error("registry.protocol_document", "must point to docs/SOTA_PROTOCOL.md")

    methods = _sequence(registry["methods"], "registry.methods")
    datasets = _sequence(registry["datasets"], "registry.datasets")
    if not methods or not datasets:
        raise _error("registry", "methods and datasets must both be non-empty")
    method_ids = [_validate_method(method, index) for index, method in enumerate(methods)]
    dataset_ids = [
        _validate_dataset(dataset, index) for index, dataset in enumerate(datasets)
    ]
    _ensure_unique(method_ids, "registry.methods")
    _ensure_unique(dataset_ids, "registry.datasets")


def load_registry(path: str | Path = DEFAULT_REGISTRY_PATH) -> dict[str, Any]:
    """Load and validate a registry JSON file before returning it."""

    registry_path = Path(path)

    def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise RegistryValidationError(f"duplicate JSON key {key!r}")
            value[key] = item
        return value

    def reject_constant(value: str) -> None:
        raise RegistryValidationError(f"non-finite JSON constant {value!r}")

    try:
        data = json.loads(
            registry_path.read_text(encoding="utf-8"),
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
    except RegistryValidationError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryValidationError(
            f"cannot load registry {registry_path}: {exc}"
        ) from exc
    validate_registry(data)
    return data


def main(argv: Sequence[str] | None = None) -> int:
    """Validate a registry from the command line and print a compact summary."""
    parser = argparse.ArgumentParser(
        description="Validate the fail-closed PIMSR SOTA method registry"
    )
    parser.add_argument(
        "registry",
        nargs="?",
        default=str(DEFAULT_REGISTRY_PATH),
        help="registry JSON path (defaults to the packaged registry)",
    )
    args = parser.parse_args(argv)
    registry = load_registry(args.registry)
    print(
        f"valid {registry['registry_id']} schema={registry['schema_version']} "
        f"methods={len(registry['methods'])} datasets={len(registry['datasets'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
