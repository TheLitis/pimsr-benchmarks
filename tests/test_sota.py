"""Tests for the versioned, fail-closed SOTA registry."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from pimsr_benchmarks.sota import (
    ALLOWED_METHOD_STATUSES,
    ALLOWED_TRACKS,
    DEFAULT_PROTOCOL_PATH,
    RegistryValidationError,
    load_registry,
    main,
    validate_registry,
)

REGISTRY_PATH = Path(__file__).resolve().parents[1] / "config" / "sota_methods.json"


@pytest.fixture
def registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def test_repository_registry_is_valid_and_complete():
    registry = load_registry(REGISTRY_PATH)
    method_ids = {method["id"] for method in registry["methods"]}
    dataset_ids = {dataset["id"] for dataset in registry["datasets"]}

    assert method_ids == {
        "rdon",
        "mt2dinv_densenet",
        "mtdlpy",
        "mare2dem",
        "simpeg",
        "modem",
        "mt3d_cnn",
        "gemmie",
        "femtic",
        "deva3dmt",
        "gan_mt1dinv",
        "guided_1d",
        "res_formernet",
        "mt2d_inr",
        "mffd_unet",
        "mt2d_autodiff",
        "mt_mamba",
        "p_physinv",
        "mt2dinv_unet",
        "mt3d_net",
        "mt1d_inr",
        "trans_scale_mt",
        "pimsr",
    }
    assert dataset_ids == {
        "coprod2",
        "coprod2s1",
        "coprod2s2",
        "mt3dinv4_sphere",
        "mt3dinv4_secret",
        "mt3dinv4_raglan",
        "pimsr_generated_2d_v1",
    }
    assert DEFAULT_PROTOCOL_PATH.is_file()
    assert "Equal observation budget" in DEFAULT_PROTOCOL_PATH.read_text(encoding="utf-8")


def test_registry_cli_reports_valid_summary(capsys):
    assert main([str(REGISTRY_PATH)]) == 0
    assert capsys.readouterr().out.strip() == (
        "valid pimsr-sota-methods schema=2 methods=23 datasets=7"
    )


def test_registry_loader_rejects_duplicate_json_keys(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_version":999,"schema_version":1}',
        encoding="utf-8",
    )
    with pytest.raises(RegistryValidationError, match="duplicate JSON key"):
        load_registry(duplicate)


def test_registry_loader_rejects_nonfinite_json_constants(tmp_path):
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"schema_version":NaN}', encoding="utf-8")
    with pytest.raises(RegistryValidationError, match="non-finite JSON constant"):
        load_registry(nonfinite)


def test_registry_pins_expected_source_commits():
    registry = load_registry(REGISTRY_PATH)
    by_id = {method["id"]: method for method in registry["methods"]}
    expected = {
        "rdon": "c6a34f78ef1adf9e663d909851acbc5e5be81fab",
        "mt2dinv_densenet": "56f3d30f42daadb87eea94a0d4c73b05abfdae41",
        "mtdlpy": "b01f72a53078a9dc8d452fa53ea5009639d00b04",
        "mare2dem": "f6b9f2d60bbf4b0272cc7f8bcb6b496d832697d6",
        "simpeg": "5f2d643c484ebe8b8ccfa029aa7377d133f2f53f",
        "modem": "55a4aa62f7e8366fbf78a23ee8a19c1d4561d0c3",
        "mt3d_cnn": "9843ba52b517c38692c96990b9e4b843b055f735",
        "gemmie": "32019342e8c5d9728ec0ad39019ad64c14546e01",
        "femtic": "e3a665b4fa58e77734f074b1fc7f18a5a03708ee",
        "deva3dmt": "a46355a4da119fdd70db7a513672faa449899e0f",
        "gan_mt1dinv": "cefccf36019a11bfa1a62ee27bea30c65aab5cb3",
        "guided_1d": "2731cebbdc163d1df26f9e88673ddfdd8a38469a",
        "pimsr": "2e4d636736762bb3e7c8e2fe66ddbc98297c6a0b",
    }

    assert {
        method_id: by_id[method_id]["source"]["ref"]["resolved_commit"]
        for method_id in expected
    } == expected
    assert by_id["mt2dinv_densenet"]["source"]["release"] == {
        "kind": "tag",
        "value": "v1.2",
        "resolved_commit": "9fc46d91c40f8a1a73155c84950689d0fb92662a",
    }


def test_allowed_status_and_track_sets_are_closed():
    assert ALLOWED_METHOD_STATUSES == {
        "reproducible_first_wave",
        "reference_only",
        "paper_only",
    }
    assert ALLOWED_TRACKS == {
        "frozen_artifact",
        "common_retrain",
        "refinement",
    }


def test_duplicate_method_ids_are_rejected(registry):
    registry["methods"][1]["id"] = registry["methods"][0]["id"]
    with pytest.raises(RegistryValidationError, match="duplicate ids"):
        validate_registry(registry)


def test_duplicate_dataset_ids_are_rejected(registry):
    registry["datasets"][1]["id"] = registry["datasets"][0]["id"]
    with pytest.raises(RegistryValidationError, match="duplicate ids"):
        validate_registry(registry)


@pytest.mark.parametrize("floating_ref", ["main", "master", "HEAD", "latest"])
def test_floating_source_refs_are_rejected(registry, floating_ref):
    source_ref = registry["methods"][0]["source"]["ref"]
    source_ref["kind"] = "tag"
    source_ref["value"] = floating_ref
    with pytest.raises(RegistryValidationError, match="floating refs"):
        validate_registry(registry)


def test_short_git_sha_is_rejected(registry):
    source_ref = registry["methods"][0]["source"]["ref"]
    source_ref["value"] = "c6a34f78"
    source_ref["resolved_commit"] = "c6a34f78"
    with pytest.raises(RegistryValidationError, match="40-character Git SHA"):
        validate_registry(registry)


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    [
        ("status", "claimed_sota", "must be one of"),
        ("track", "private_tuning", "unsupported values"),
    ],
)
def test_unknown_method_enums_are_rejected(registry, field, invalid, message):
    if field == "status":
        registry["methods"][0]["status"] = invalid
    else:
        registry["methods"][0]["tracks"].append(invalid)
    with pytest.raises(RegistryValidationError, match=message):
        validate_registry(registry)


def test_malformed_doi_is_rejected(registry):
    registry["methods"][0]["publications"][0]["doi"] = "doi:RDON"
    with pytest.raises(RegistryValidationError, match="bare DOI"):
        validate_registry(registry)


def test_non_https_repository_is_rejected(registry):
    registry["methods"][0]["source"]["repository_url"] = (
        "http://github.com/zhangheng-1/RDON"
    )
    with pytest.raises(RegistryValidationError, match="absolute HTTPS URL"):
        validate_registry(registry)


def test_unknown_fields_are_rejected(registry):
    registry["methods"][0]["score_from_paper"] = 0.01
    with pytest.raises(RegistryValidationError, match="unknown keys"):
        validate_registry(registry)


def test_paper_only_method_cannot_enter_executable_track(registry):
    method = next(item for item in registry["methods"] if item["id"] == "mt_mamba")
    method["tracks"] = ["frozen_artifact"]
    with pytest.raises(RegistryValidationError, match="paper-only methods cannot"):
        validate_registry(registry)


def test_paper_only_method_may_register_context_data_but_not_only_available_artifacts(
    registry,
):
    method = next(item for item in registry["methods"] if item["id"] == "mt2d_inr")
    validate_registry(registry)
    method["artifacts"] = [method["artifacts"][1]]
    with pytest.raises(RegistryValidationError, match="unavailable execution artifact"):
        validate_registry(registry)


def test_unavailable_artifact_cannot_claim_a_url(registry):
    method = next(item for item in registry["methods"] if item["id"] == "mt_mamba")
    method["artifacts"][0]["url"] = method["publications"][0]["url"]
    with pytest.raises(RegistryValidationError, match="require null url"):
        validate_registry(registry)


def test_dataset_hash_policy_is_mandatory(registry):
    registry["datasets"][0]["checksum_policy"] = "trust_official_url"
    with pytest.raises(RegistryValidationError, match="sha256_required_before_run"):
        validate_registry(registry)


def test_hidden_generated_dataset_never_publishes_seed_or_source_indices(registry):
    generated = next(
        item for item in registry["datasets"] if item["id"] == "pimsr_generated_2d_v1"
    )
    generator = generated["generator"]

    assert generator["schema_version"] == 2
    assert "master_seed" not in generator
    assert "sample_count" not in generator
    assert generator["campaign_count"] == 5
    assert generator["samples_per_campaign"] == 500
    assert generator["seed_policy"] == "operator_withheld_until_predictions_locked"
    assert generator["command_template"][6] == "<operator-withheld-seed>"
    assert len(generator["seed_commitment_sha256"]) == 64
    assert len(generator["sample_id_key_commitment_sha256"]) == 64


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("seed_commitment_sha256", "a" * 63, "64-character"),
        ("sample_id_key_commitment_sha256", "A" * 64, "lowercase"),
        ("seed_policy", "public_seed", "operator-withheld"),
        ("campaign_count", 4, "integer 5"),
    ],
)
def test_hidden_generated_dataset_commitment_contract_is_fail_closed(
    registry, field, value, message
):
    generated = next(
        item for item in registry["datasets"] if item["id"] == "pimsr_generated_2d_v1"
    )
    generated["generator"][field] = value
    with pytest.raises(RegistryValidationError, match=message):
        validate_registry(registry)


def test_input_is_not_mutated_by_validation(registry):
    before = copy.deepcopy(registry)
    validate_registry(registry)
    assert registry == before


def test_registry_does_not_overstate_dense_or_gemmie_artifacts(registry):
    methods = {method["id"]: method for method in registry["methods"]}
    dense = methods["mt2dinv_densenet"]
    assert dense["tracks"] == ["common_retrain"]
    assert dense["artifacts"][0]["kind"] == "software_source_release"
    assert "no training data or checkpoints" in dense["artifacts"][0]["caveat"]

    gemmie = methods["gemmie"]
    assert gemmie["artifacts"][0]["kind"] == "synthetic_impedance_dataset"
    assert "not source code" in gemmie["artifacts"][0]["caveat"]


def test_recent_2d_methods_without_reproduction_bundles_stay_paper_only(registry):
    methods = {method["id"]: method for method in registry["methods"]}
    for method_id in ("mffd_unet", "mt2d_autodiff"):
        method = methods[method_id]
        assert method["status"] == "paper_only"
        assert method["tracks"] == []
        assert method["source"] is None
        assert any(
            artifact["availability"] == "unavailable"
            for artifact in method["artifacts"]
        )


def test_invalid_json_is_reported_as_registry_error(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(RegistryValidationError, match="cannot load registry"):
        load_registry(path)
