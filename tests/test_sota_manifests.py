"""Executable SOTA manifest contracts and immutable publication tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from pimsr_benchmarks.sota import load_registry
from pimsr_benchmarks.sota_manifests import (
    EXPERIMENT_SCHEMA,
    MANIFEST_SCHEMA_VERSION,
    ManifestPublicationError,
    ManifestValidationError,
    SnapshotError,
    canonical_json_bytes,
    load_manifest,
    main,
    publish_manifest,
    sha256_file,
    snapshot_file,
    validate_experiment,
    validate_prediction_manifest,
    validate_run_manifest,
)

REGISTRY_PATH = Path(__file__).resolve().parents[1] / "config" / "sota_methods.json"
PIMSR_COMMIT = "2e4d636736762bb3e7c8e2fe66ddbc98297c6a0b"


def _physical_contract() -> dict:
    return {
        "dimensionality": "2d",
        "coordinate_system": "right_handed_x_profile_y_strike_z_down",
        "handedness": "right_handed",
        "vertical_positive": "down",
        "rotation_degrees": 0.0,
        "axes": ["x", "z"],
        "axis_units": {"x": "m", "z": "m"},
        "components": ["Zyx", "Zxy"],
        "component_units": {"Zyx": "ohm", "Zxy": "ohm"},
        "spectral_axis": "frequency",
        "spectral_unit": "Hz",
        "spectral_order": "ascending",
        "phase_unit": "degree",
        "phase_convention": "degrees_modulo_180_[0,180)",
        "time_convention": "exp(+i_omega_t)",
        "resistivity_unit": "ohm_m",
        "model_parameter": "log10_resistivity",
        "model_parameter_unit": "log10_ohm_m",
    }


def _split() -> dict:
    return {
        "split_id": "hidden_test",
        "groups": [
            {
                "family_id": "family_a",
                "base_model_id": "base_001",
                "noise_id": "noise_001",
                "sample_ids": ["sample_001", "sample_002"],
            }
        ],
    }


def _registry_snapshot(tmp_path: Path) -> dict:
    return snapshot_file(REGISTRY_PATH, tmp_path, media_type="application/json")


def _source_snapshot(tmp_path: Path) -> dict:
    source = tmp_path / "source.tar"
    source.write_bytes(b"pinned source tree\n")
    return snapshot_file(source, tmp_path, media_type="application/x-tar")


def _source(snapshot: dict) -> dict:
    return {
        "repository_url": "https://github.com/TheLitis/pimsr-inversion",
        "commit": PIMSR_COMMIT,
        "artifact": snapshot,
        "dirty_tree": False,
    }


def _observation(tmp_path: Path) -> tuple[dict, dict]:
    payload = tmp_path / "observations.h5"
    payload.write_bytes(b"test-only materialized observations\n")
    payload_ref = snapshot_file(payload, tmp_path, media_type="application/x-hdf5")
    manifest = {
        "schema": "pimsr-sota-observations",
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "manifest_id": "obs_coprod2s1_v1",
        "dataset_id": "coprod2s1",
        "dimensionality": "2d",
        "artifact_state": "materialized",
        "split": _split(),
        "physical_contract": _physical_contract(),
        "payload": payload_ref,
        "array_sha256": {
            "station_coordinates": "1" * 64,
            "spectral_axis": "2" * 64,
            "observations": "3" * 64,
            "uncertainties": "4" * 64,
            "valid_mask": "5" * 64,
            "truth": "6" * 64,
        },
        "created_utc": "2026-08-23T12:00:00Z",
    }
    path = publish_manifest(manifest, tmp_path / "observations.json")
    return manifest, snapshot_file(path, tmp_path, media_type="application/json")


def _experiment(
    tmp_path: Path,
    observation_ref: dict | None,
    source_ref: dict,
    *,
    conditional: bool = False,
) -> dict:
    return {
        "schema": EXPERIMENT_SCHEMA,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": "pimsr_coprod2s1_common_v1",
        "protocol_version": "1.0",
        "registry": _registry_snapshot(tmp_path),
        "method_id": "pimsr",
        "dataset_id": "pimsr_generated_2d_v1" if conditional else "coprod2s1",
        "dimensionality": "2d",
        "track": "common_retrain",
        "execution_status": "artifact_pinned",
        "dataset_artifact_state": (
            "conditional_not_materialized" if conditional else "materialized"
        ),
        "source": _source(source_ref),
        "commands": {
            "prepare": ["python", "prepare.py", "--frozen"],
            "execute": ["python", "adapter.py", "--no-tune"],
            "evaluate": ["python", "evaluate.py", "--independent-solver"],
        },
        "split": _split(),
        "physical_contract": _physical_contract(),
        "observation_manifest": observation_ref,
        "random_seeds": [101, 102, 103, 104, 105],
        "created_utc": "2026-08-23T12:01:00Z",
        "notes": ["No test-set tuning."],
    }


def _prediction(
    tmp_path: Path, experiment_ref: dict, observation_ref: dict
) -> tuple[dict, dict]:
    output = tmp_path / "predictions.bin"
    output.write_bytes(b"predicted models and responses\n")
    output_ref = snapshot_file(output, tmp_path)
    manifest = {
        "schema": "pimsr-sota-predictions",
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "manifest_id": "pred_pimsr_coprod2s1_v1",
        "experiment_id": "pimsr_coprod2s1_common_v1",
        "method_id": "pimsr",
        "dataset_id": "coprod2s1",
        "dimensionality": "2d",
        "track": "common_retrain",
        "execution_status": "adapter_smoke_passed",
        "experiment_manifest": experiment_ref,
        "observation_manifest": observation_ref,
        "split": _split(),
        "physical_contract": _physical_contract(),
        "outputs": [output_ref],
        "created_utc": "2026-08-23T12:03:00Z",
    }
    path = publish_manifest(manifest, tmp_path / "predictions.json")
    return manifest, snapshot_file(path, tmp_path, media_type="application/json")


def _run(
    experiment_ref: dict,
    observation_ref: dict,
    prediction_ref: dict,
    source_ref: dict,
    input_ref: dict,
    output_ref: dict,
) -> dict:
    return {
        "schema": "pimsr-sota-run",
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": "run_pimsr_coprod2s1_v1",
        "method_id": "pimsr",
        "dataset_id": "coprod2s1",
        "dimensionality": "2d",
        "track": "common_retrain",
        "execution_status": "adapter_smoke_passed",
        "experiment_manifest": experiment_ref,
        "observation_manifest": observation_ref,
        "prediction_manifest": prediction_ref,
        "source": _source(source_ref),
        "command": ["python", "adapter.py", "--no-tune"],
        "working_directory": "work/pimsr",
        "inputs": [input_ref],
        "outputs": [output_ref],
        "environment": {
            "python_version": "3.11.13",
            "platform": "windows-amd64",
            "lockfile_sha256": "7" * 64,
            "package_inventory_sha256": "8" * 64,
            "container_image_digest": None,
        },
        "execution": {
            "started_utc": "2026-08-23T12:01:30Z",
            "finished_utc": "2026-08-23T12:02:30Z",
            "exit_status": "succeeded",
            "exit_code": 0,
            "converged": True,
            "warnings": [],
        },
        "resources": {
            "wall_time_s": 60.0,
            "cpu_time_s": 41.25,
            "peak_host_ram_bytes": 1048576,
            "peak_accelerator_memory_bytes": None,
            "cpu_count": 8,
            "accelerator_count": 0,
            "threads": 4,
            "mpi_ranks": 0,
            "precision": "float32",
            "energy_joules": None,
        },
    }


def _adapter_smoke_tree(tmp_path: Path) -> tuple[Path, dict, dict, dict]:
    source_ref = _source_snapshot(tmp_path)
    observation, observation_ref = _observation(tmp_path)
    experiment = _experiment(tmp_path, observation_ref, source_ref)
    experiment_path = publish_manifest(experiment, tmp_path / "experiment.json")
    experiment_ref = snapshot_file(
        experiment_path, tmp_path, media_type="application/json"
    )
    prediction, prediction_ref = _prediction(tmp_path, experiment_ref, observation_ref)
    run = _run(
        experiment_ref,
        observation_ref,
        prediction_ref,
        source_ref,
        observation["payload"],
        prediction["outputs"][0],
    )
    run_path = publish_manifest(run, tmp_path / "run.json")
    return run_path, experiment, observation, prediction


def test_registry_marks_pimsr_and_generated_dataset_honestly():
    registry = load_registry(REGISTRY_PATH)
    methods = {item["id"]: item for item in registry["methods"]}
    datasets = {item["id"]: item for item in registry["datasets"]}

    assert methods["pimsr"]["source"]["ref"]["resolved_commit"] == PIMSR_COMMIT
    assert "artifact_pinned only" in methods["pimsr"]["caveats"][0]
    generated = datasets["pimsr_generated_2d_v1"]
    assert generated["status"] == "conditional"
    assert generated["artifact_availability"] == "not_yet_materialized"
    assert generated["truth"] == "not_yet_materialized"
    assert generated["generator"]["source_status"] == "artifact_pinned"
    assert generated["generator"]["source_commit"] == (
        "dc36edac75dbd51cc92679a35f38d42d0e276299"
    )
    assert "sha256" not in generated
    assert generated["checksum_policy"] == "sha256_required_before_run"


def test_adapter_smoke_manifest_tree_roundtrips_and_cli_validates(tmp_path, capsys):
    run_path, _experiment_value, _observation_value, _prediction_value = (
        _adapter_smoke_tree(tmp_path)
    )

    loaded = load_manifest(run_path)
    assert loaded["execution_status"] == "adapter_smoke_passed"
    assert main(["validate", str(run_path)]) == 0
    output = capsys.readouterr().out
    assert "valid pimsr-sota-run schema=1 sha256=" in output


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda run: run.update(command=["python", "totally_unrelated.py"]),
            r"run.command.*experiment.commands.execute",
        ),
        (
            lambda run: run.update(inputs=[run["source"]["artifact"]]),
            r"run.inputs.*exact observation payload",
        ),
        (
            lambda run: run.update(outputs=[run["source"]["artifact"]]),
            r"run.outputs.*prediction manifest output identities",
        ),
    ],
)
def test_adapter_smoke_promotion_binds_command_inputs_and_outputs(
    tmp_path, mutation, message
):
    run_path, _experiment_value, _observation_value, _prediction_value = (
        _adapter_smoke_tree(tmp_path)
    )
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["run_id"] = "run_unrelated_promotion"
    mutation(run)

    with pytest.raises(ManifestValidationError, match=message):
        publish_manifest(run, tmp_path / "invalid-run.json")


def test_artifact_pinned_not_run_may_remain_unexecuted(tmp_path):
    run_path, experiment, _observation, _prediction = _adapter_smoke_tree(tmp_path)
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run.update(
        execution_status="artifact_pinned",
        observation_manifest=None,
        prediction_manifest=None,
        command=[],
        inputs=[],
        outputs=[],
    )
    run["source"]["artifact"] = None
    run["execution"] = {
        "started_utc": None,
        "finished_utc": None,
        "exit_status": "not_run",
        "exit_code": None,
        "converged": None,
        "warnings": [],
    }
    run["resources"].update(wall_time_s=0.0, cpu_time_s=None, energy_joules=None)

    validate_run_manifest(
        run,
        load_registry(REGISTRY_PATH),
        experiment=experiment,
    )


def test_conditional_experiment_does_not_claim_dataset_bytes(tmp_path):
    registry = load_registry(REGISTRY_PATH)
    source_ref = _source_snapshot(tmp_path)
    experiment = _experiment(tmp_path, None, source_ref, conditional=True)

    validate_experiment(
        experiment,
        registry,
        registry_sha256=sha256_file(REGISTRY_PATH),
    )
    path = publish_manifest(experiment, tmp_path / "conditional.json")
    assert load_manifest(path)["observation_manifest"] is None

    claimed = copy.deepcopy(experiment)
    claimed["dataset_artifact_state"] = "materialized"
    with pytest.raises(ManifestValidationError, match="observation_manifest.*required"):
        validate_experiment(
            claimed,
            registry,
            registry_sha256=sha256_file(REGISTRY_PATH),
        )


def test_conditional_experiment_cannot_promote_predictions(tmp_path):
    registry = load_registry(REGISTRY_PATH)
    source_ref = _source_snapshot(tmp_path)
    experiment = _experiment(tmp_path, None, source_ref, conditional=True)
    fake = {
        "schema": "pimsr-sota-predictions",
        "schema_version": 1,
        "manifest_id": "invalid_prediction",
        "experiment_id": experiment["experiment_id"],
        "method_id": "pimsr",
        "dataset_id": "pimsr_generated_2d_v1",
        "dimensionality": "2d",
        "track": "common_retrain",
        "execution_status": "adapter_smoke_passed",
        "experiment_manifest": experiment["registry"],
        "observation_manifest": experiment["registry"],
        "split": _split(),
        "physical_contract": _physical_contract(),
        "outputs": [experiment["registry"]],
        "created_utc": "2026-08-23T12:03:00Z",
    }
    with pytest.raises(ManifestValidationError, match="unmaterialized datasets"):
        validate_prediction_manifest(fake, registry, experiment=experiment)


def test_tampered_referenced_bytes_are_rejected(tmp_path):
    run_path, _experiment_value, observation, _prediction_value = (
        _adapter_smoke_tree(tmp_path)
    )
    payload = tmp_path / observation["payload"]["path"]
    payload.write_bytes(b"tampered\n")

    with pytest.raises(ManifestValidationError, match="size does not match|SHA-256"):
        load_manifest(run_path)


def test_legacy_unknown_and_noncanonical_manifests_are_rejected(tmp_path):
    legacy = tmp_path / "legacy.json"
    legacy.write_bytes(
        canonical_json_bytes({"schema": "pimsr-sota-run", "schema_version": 0})
    )
    with pytest.raises(ManifestValidationError, match="missing keys|legacy"):
        load_manifest(legacy)

    unknown = tmp_path / "unknown.json"
    unknown.write_bytes(
        canonical_json_bytes({"schema": "old-benchmark", "schema_version": 1})
    )
    with pytest.raises(ManifestValidationError, match="unknown or legacy schema"):
        load_manifest(unknown)

    formatted = tmp_path / "formatted.json"
    formatted.write_text(
        json.dumps({"schema": "old-benchmark"}, indent=2), encoding="utf-8"
    )
    with pytest.raises(ManifestValidationError, match="not in canonical"):
        load_manifest(formatted)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema":"old-benchmark","schema":"pimsr-sota-run"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ManifestValidationError, match="duplicate JSON key"):
        load_manifest(duplicate)


def test_failed_run_cannot_claim_adapter_status(tmp_path):
    _run_path, experiment, observation, prediction = _adapter_smoke_tree(tmp_path)
    source_ref = _source_snapshot(tmp_path)
    run = _run(
        experiment["registry"],
        observation["payload"],
        prediction["outputs"][0],
        source_ref,
        observation["payload"],
        prediction["outputs"][0],
    )
    run["execution"]["exit_status"] = "failed"
    run["execution"]["exit_code"] = 1
    with pytest.raises(ManifestValidationError, match="cannot claim"):
        validate_run_manifest(run, load_registry(REGISTRY_PATH))


def test_schema_v1_cannot_claim_benchmark_complete(tmp_path):
    _run_path, experiment, observation, prediction = _adapter_smoke_tree(tmp_path)
    registry = load_registry(REGISTRY_PATH)

    completed_prediction = copy.deepcopy(prediction)
    completed_prediction["execution_status"] = "benchmark_complete"
    with pytest.raises(
        ManifestValidationError,
        match=r"schema version 1 cannot claim benchmark_complete.*typed HDF5/NPZ",
    ):
        validate_prediction_manifest(completed_prediction, registry)

    source_ref = _source_snapshot(tmp_path)
    completed_run = _run(
        experiment["registry"],
        observation["payload"],
        prediction["outputs"][0],
        source_ref,
        observation["payload"],
        prediction["outputs"][0],
    )
    completed_run["execution_status"] = "benchmark_complete"
    with pytest.raises(
        ManifestValidationError,
        match=r"schema version 1 cannot claim benchmark_complete.*ordered seed",
    ):
        validate_run_manifest(completed_run, registry)


def test_adapter_smoke_cannot_be_explicitly_non_converged(tmp_path):
    _run_path, experiment, observation, prediction = _adapter_smoke_tree(tmp_path)
    source_ref = _source_snapshot(tmp_path)
    run = _run(
        experiment["registry"],
        observation["payload"],
        prediction["outputs"][0],
        source_ref,
        observation["payload"],
        prediction["outputs"][0],
    )
    run["execution"]["converged"] = False

    with pytest.raises(
        ManifestValidationError,
        match="adapter_smoke_passed cannot claim an explicitly non-converged run",
    ):
        validate_run_manifest(run, load_registry(REGISTRY_PATH))


def test_atomic_publication_refuses_overwrite_and_stale_partial(tmp_path):
    source_ref = _source_snapshot(tmp_path)
    experiment = _experiment(tmp_path, None, source_ref, conditional=True)
    destination = publish_manifest(experiment, tmp_path / "experiment.json")

    with pytest.raises(ManifestPublicationError, match="overwrite"):
        publish_manifest(experiment, destination)

    second = tmp_path / "second.json"
    second.with_name(second.name + ".part").write_bytes(b"stale")
    with pytest.raises(ManifestPublicationError, match="stale partial"):
        publish_manifest(experiment, second)


def test_snapshot_is_content_addressed_idempotent_and_detects_corruption(tmp_path):
    source = tmp_path / "input.bin"
    source.write_bytes(b"immutable input")
    snapshots = tmp_path / "snapshots"

    first = snapshot_file(source, snapshots)
    second = snapshot_file(source, snapshots)
    assert second == first
    target = snapshots / first["path"]
    assert target.name == f"{first['sha256']}.blob"

    target.write_bytes(b"corrupt")
    with pytest.raises(SnapshotError, match="corrupt/conflicting"):
        snapshot_file(source, snapshots)


def test_exact_keys_lowercase_hashes_and_group_ids_fail_closed(tmp_path):
    registry = load_registry(REGISTRY_PATH)
    source_ref = _source_snapshot(tmp_path)
    experiment = _experiment(tmp_path, None, source_ref, conditional=True)
    experiment["score"] = 0.1
    with pytest.raises(ManifestValidationError, match="unknown keys"):
        validate_experiment(
            experiment,
            registry,
            registry_sha256=sha256_file(REGISTRY_PATH),
        )

    del experiment["score"]
    experiment["registry"]["sha256"] = "A" * 64
    with pytest.raises(ManifestValidationError, match="lowercase 64-character"):
        validate_experiment(
            experiment,
            registry,
            registry_sha256=sha256_file(REGISTRY_PATH),
        )

    experiment["registry"] = _registry_snapshot(tmp_path)
    experiment["split"]["groups"].append(copy.deepcopy(experiment["split"]["groups"][0]))
    with pytest.raises(ManifestValidationError, match="duplicate family/base/noise"):
        validate_experiment(
            experiment,
            registry,
            registry_sha256=sha256_file(REGISTRY_PATH),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(dataset_id="mt3dinv4_sphere"), "does not match"),
        (
            lambda value: value.update(track="frozen_artifact"),
            "unsupported by the method",
        ),
        (
            lambda value: value["source"].update(commit="0" * 40),
            "does not match registry method commit",
        ),
        (
            lambda value: value["physical_contract"].update(
                phase_convention="degrees_first_quadrant"
            ),
            "degrees_modulo_180",
        ),
        (
            lambda value: value["registry"].update(path="../registry.json"),
            "normalized relative path",
        ),
    ],
)
def test_method_dataset_track_and_physical_contract_compatibility(
    tmp_path, mutation, message
):
    registry = load_registry(REGISTRY_PATH)
    experiment = _experiment(tmp_path, None, _source_snapshot(tmp_path), conditional=True)
    mutation(experiment)
    with pytest.raises(ManifestValidationError, match=message):
        validate_experiment(
            experiment,
            registry,
            registry_sha256=sha256_file(REGISTRY_PATH),
        )
