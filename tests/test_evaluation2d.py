from __future__ import annotations

import hashlib
import json
import os
import subprocess
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pimsr_benchmarks import evaluation2d
from pimsr_benchmarks.evaluation2d import (
    EVALUATION_SCHEMA,
    PREDICTION_SCHEMA,
    PREDICTION_SCHEMA_VERSION,
    TRUTH_SCHEMA,
    TRUTH_SCHEMA_VERSION,
    Evaluation2DPublicationError,
    Evaluation2DPublicationReceipt,
    Evaluation2DValidationError,
    canonical_json_bytes,
    cell_edges_from_centers,
    cell_widths_from_centers,
    evaluate_predictions_2d,
    load_predictions_2d,
    load_truth_2d,
    publish_evaluation_2d,
    publish_evaluation_2d_receipt,
)
from pimsr_benchmarks.prediction_lock2d import PredictionLock2DValidationError

_OBSERVATIONS_SHA256 = "1" * 64
_LOAD_OPERATOR_BINDING = evaluation2d._load_operator_binding
_IMPLEMENTATION_IDENTITY = evaluation2d._implementation_identity


@pytest.fixture(autouse=True)
def _validated_lock_gate(monkeypatch: pytest.MonkeyPatch):
    source_payload = Path(evaluation2d.__file__).read_bytes()
    source_sha256 = hashlib.sha256(source_payload).hexdigest()
    monkeypatch.setattr(
        evaluation2d,
        "_implementation_identity",
        lambda: {
            "distribution_version": None,
            "git_commit": "5" * 40,
            "git_dirty_tree": False,
            "git_head_commit": "5" * 40,
            "numpy_version": np.__version__,
            "python_version": "test",
            "source_file": "evaluation2d.py",
            "source_sha256": source_sha256,
            "source_size_bytes": len(source_payload),
        },
    )
    run = SimpleNamespace(
        adapter_source_sha256="a" * 64,
        campaign_id="campaign-1",
        checkpoint_sha256="b" * 64,
        method_id="pimsr",
        observations_sha256=_OBSERVATIONS_SHA256,
        prediction_sha256=None,
        runtime_sha256="c" * 64,
        source_commit="d" * 40,
        source_sha256="e" * 64,
        training_seed=101,
    )
    lock = SimpleNamespace(
        input_manifest_sha256="f" * 64,
        lock_sha256="0" * 64,
        preregistration_sha256="2" * 64,
        statistical_options={"confidence": 0.8, "n_resamples": 100, "rng_seed": 17},
        require_run=lambda campaign_id, method_id, training_seed: run,
    )
    monkeypatch.setattr(evaluation2d, "validate_prediction_lock_2d", lambda *a, **k: lock)
    monkeypatch.setattr(
        evaluation2d, "validate_locked_run_artifacts_2d", lambda *a, **k: None
    )

    def operator(*args, **kwargs):
        return SimpleNamespace(
            family_by_sample={10: "background", 20: "salt"},
            family_commitment_sha256="4" * 64,
            snapshot=SimpleNamespace(sha256="3" * 64, size_bytes=123),
            truth_sha256=kwargs["expected_truth_sha256"],
            truth_size_bytes=None,
        )

    monkeypatch.setattr(evaluation2d, "_load_operator_binding", operator)


def _evaluate_locked(truth_path: Path, prediction_path: Path, **overrides) -> dict:
    truth_sha256 = overrides.pop("expected_truth_sha256", None)
    if truth_sha256 is None:
        truth_sha256 = hashlib.sha256(truth_path.read_bytes()).hexdigest()
    options = {
        "preregistration_path": "prereg.json",
        "expected_preregistration_sha256": "2" * 64,
        "predictions_lock_path": "predictions-lock.json",
        "expected_predictions_lock_sha256": "0" * 64,
        "campaign_id": "campaign-1",
        "method_id": "pimsr",
        "training_seed": 101,
        "observations_path": "observations.npz",
        "observation_manifest_path": "observations.json",
        "runtime_path": "runtime.json",
        "checkpoint_path": "checkpoint.pt",
        "source_path": "network2d.py",
        "operator_manifest_path": "operator.json",
        "expected_operator_manifest_sha256": "3" * 64,
        "expected_truth_sha256": truth_sha256,
    }
    options.update(overrides)
    return evaluate_predictions_2d(truth_path, prediction_path, **options)


def _truth_arrays() -> dict[str, np.ndarray]:
    return {
        "schema": np.asarray(TRUTH_SCHEMA),
        "schema_version": np.asarray(TRUTH_SCHEMA_VERSION, dtype="<i8"),
        "sample_index": np.asarray([20, 10], dtype="<i8"),
        "observations_sha256": np.asarray(_OBSERVATIONS_SHA256, dtype="<U64"),
        "scenario": np.asarray(["salt", "background"]),
        "has_fault": np.asarray([True, False], dtype=np.bool_),
        "x_cell_centers_m": np.asarray([0.0, 1.0, 4.0], dtype="<f8"),
        "depth_cell_centers_m": np.asarray([1.0, 3.0], dtype="<f8"),
        "truth_log10_resistivity": np.zeros((2, 2, 3), dtype="<f4"),
    }


def _prediction_arrays() -> dict[str, np.ndarray]:
    values = np.zeros((2, 2, 3), dtype="<f4")
    # Prediction rows are intentionally in the opposite order from truth rows.
    # Sample 10 has error only in the widest x cell; sample 20 has uniform error.
    values[0, :, -1] = 1.0
    values[1, :, :] = 2.0
    return {
        "schema": np.asarray(PREDICTION_SCHEMA),
        "schema_version": np.asarray(PREDICTION_SCHEMA_VERSION, dtype="<i8"),
        "observations_sha256": np.asarray(_OBSERVATIONS_SHA256, dtype="<U64"),
        "sample_index": np.asarray([10, 20], dtype="<i8"),
        "x_cell_centers_m": np.asarray([0.0, 1.0, 4.0], dtype="<f8"),
        "depth_cell_centers_m": np.asarray([1.0, 3.0], dtype="<f8"),
        "predicted_log10_resistivity": values,
    }


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> Path:
    np.savez(path, **arrays)
    return path


def _artifacts(tmp_path: Path) -> tuple[Path, Path]:
    return (
        _write_npz(tmp_path / "truth.npz", _truth_arrays()),
        _write_npz(tmp_path / "prediction.npz", _prediction_arrays()),
    )


def _git(repository: Path, *arguments: str, input_payload: bytes | None = None) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        input=input_payload,
        timeout=30,
    ).stdout


def _implementation_repository(
    tmp_path: Path,
    *,
    source_payload: bytes,
    attributes: str | None = None,
) -> tuple[Path, Path, str]:
    repository = tmp_path / "implementation-repository"
    source = repository / "src" / "pimsr_benchmarks" / "evaluation2d.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(source_payload)
    if attributes is not None:
        (repository / ".gitattributes").write_text(attributes, encoding="utf-8")
    subprocess.run(
        ["git", "init", str(repository)],
        check=True,
        capture_output=True,
        timeout=30,
    )
    _git(repository, "config", "user.name", "Evaluation Test")
    _git(repository, "config", "user.email", "evaluation@example.invalid")
    _git(repository, "add", ".")
    _git(repository, "commit", "--no-gpg-sign", "-m", "pin evaluator")
    commit = _git(repository, "rev-parse", "HEAD").decode("ascii").strip()
    return repository, source, commit


def _operator_fixture(tmp_path: Path) -> tuple[Path, str, SimpleNamespace, dict]:
    campaign_id = "campaign-1"
    families = evaluation2d.GEOLOGICAL_FAMILIES
    rows: list[dict] = []
    groups: list[dict] = []
    sample_id_mapping: list[dict] = []
    grouped_samples: dict[str, list[int]] = {family: [] for family in families}
    sample_index = 0
    for family_id in families:
        for base_index in range(evaluation2d.BASE_MODELS_PER_FAMILY):
            base_model_id = f"{family_id}-base-{base_index:02d}"
            for noise_index in range(evaluation2d.NOISE_REALIZATIONS_PER_BASE):
                rows.append(
                    {
                        "base_model_id": base_model_id,
                        "family_id": family_id,
                        "noise_index": noise_index,
                        "sample_index": sample_index,
                    }
                )
                groups.append(
                    {
                        "base_model_id": base_model_id,
                        "family_id": family_id,
                        "noise_id": f"noise-{noise_index}",
                        "sample_ids": [f"sample-{sample_index}"],
                    }
                )
                sample_id_mapping.append(
                    {
                        "opaque_sample_index": sample_index,
                        "source_generator_sample_index": sample_index,
                    }
                )
                grouped_samples[family_id].append(sample_index)
                sample_index += 1
    nonce_hex = "01" * 32
    commitment_sha256 = evaluation2d._family_commitment_digest(
        campaign_id=campaign_id,
        nonce_hex=nonce_hex,
        rows=rows,
    )
    observations = SimpleNamespace(
        payload=b"locked-observations",
        sha256=_OBSERVATIONS_SHA256,
        size_bytes=19,
    )
    public_manifest = {
        "audience": "method_input_public",
        "declared_evaluation_floors": {},
        "family_partition_commitment": {
            "contract": dict(evaluation2d.FAMILY_COMMITMENT_CONTRACT),
            "schema": evaluation2d.FAMILY_PARTITION_SCHEMA,
            "schema_version": evaluation2d.FAMILY_PARTITION_SCHEMA_VERSION,
            "sha256": commitment_sha256,
        },
        "observation_payload": {
            "schema": "pimsr-sota-2d-observations",
            "schema_version": evaluation2d.OBSERVATION_SCHEMA_VERSION,
            "sha256": observations.sha256,
            "size_bytes": observations.size_bytes,
        },
        "physical_contract": {},
        "sample_count": evaluation2d.SAMPLES_PER_CAMPAIGN,
        "schema": evaluation2d.OBSERVATION_MANIFEST_SCHEMA,
        "schema_version": evaluation2d.OBSERVATION_MANIFEST_SCHEMA_VERSION,
        "split_id": campaign_id,
    }
    public_payload = canonical_json_bytes(public_manifest)
    public_snapshot = SimpleNamespace(
        payload=public_payload,
        sha256=hashlib.sha256(public_payload).hexdigest(),
        size_bytes=len(public_payload),
    )
    truth_sha256 = "9" * 64
    operator = {
        "artifacts": {
            "observations": {
                "schema": "pimsr-sota-2d-observations",
                "schema_version": evaluation2d.OBSERVATION_SCHEMA_VERSION,
                "sha256": observations.sha256,
                "size_bytes": observations.size_bytes,
            },
            "public_observation_manifest": {
                "schema": evaluation2d.OBSERVATION_MANIFEST_SCHEMA,
                "schema_version": evaluation2d.OBSERVATION_MANIFEST_SCHEMA_VERSION,
                "sha256": public_snapshot.sha256,
                "size_bytes": public_snapshot.size_bytes,
            },
            "withheld_truth": {
                "schema": TRUTH_SCHEMA,
                "schema_version": TRUTH_SCHEMA_VERSION,
                "sha256": truth_sha256,
                "size_bytes": 12345,
            },
        },
        "audience": "benchmark_operator_only",
        "schema": evaluation2d.OPERATOR_MANIFEST_SCHEMA,
        "schema_version": evaluation2d.OPERATOR_MANIFEST_SCHEMA_VERSION,
        "source": {
            "production_generation_closure": (
                "post_score_manifest.campaign.hidden_generation"
            )
        },
        "split": {
            "family_partition_reveal": {
                "campaign_id": campaign_id,
                "nonce_hex": nonce_hex,
                "rows": rows,
                "schema": evaluation2d.FAMILY_REVEAL_SCHEMA,
                "schema_version": evaluation2d.FAMILY_REVEAL_SCHEMA_VERSION,
            },
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
            "sample_count": evaluation2d.SAMPLES_PER_CAMPAIGN,
            "sample_id_mapping": sample_id_mapping,
            "scenario_groups": [
                {
                    "opaque_sample_indices": grouped_samples[family_id],
                    "scenario": family_id,
                    "scenario_index": family_index,
                }
                for family_index, family_id in enumerate(families)
            ],
            "split_id": campaign_id,
        },
    }
    operator_path = tmp_path / "operator.json"
    operator_payload = canonical_json_bytes(operator)
    operator_path.write_bytes(operator_payload)
    return (
        operator_path,
        truth_sha256,
        SimpleNamespace(
            observation_manifest=public_snapshot,
            observations=observations,
        ),
        operator,
    )


def test_operator_family_reveal_opens_public_commitment(tmp_path: Path):
    operator_path, truth_sha256, locked_artifacts, _ = _operator_fixture(tmp_path)
    operator_sha256 = hashlib.sha256(operator_path.read_bytes()).hexdigest()

    binding = _LOAD_OPERATOR_BINDING(
        operator_path,
        expected_sha256=operator_sha256,
        campaign_id="campaign-1",
        observations_sha256=_OBSERVATIONS_SHA256,
        expected_truth_sha256=truth_sha256,
        locked_artifacts=locked_artifacts,
    )

    assert binding.truth_sha256 == truth_sha256
    assert binding.truth_size_bytes == 12345
    assert len(binding.family_by_sample) == 500
    assert binding.family_by_sample[0] == "background"
    assert binding.family_by_sample[499] == "geothermal"


def test_operator_family_reveal_nonce_tampering_fails_closed(tmp_path: Path):
    operator_path, truth_sha256, locked_artifacts, operator = _operator_fixture(tmp_path)
    operator["split"]["family_partition_reveal"]["nonce_hex"] = "02" * 32
    operator_payload = canonical_json_bytes(operator)
    operator_path.write_bytes(operator_payload)

    with pytest.raises(Evaluation2DValidationError, match="does not open"):
        _LOAD_OPERATOR_BINDING(
            operator_path,
            expected_sha256=hashlib.sha256(operator_payload).hexdigest(),
            campaign_id="campaign-1",
            observations_sha256=_OBSERVATIONS_SHA256,
            expected_truth_sha256=truth_sha256,
            locked_artifacts=locked_artifacts,
        )


def test_operator_family_reveal_requires_family_id_key(tmp_path: Path):
    operator_path, truth_sha256, locked_artifacts, operator = _operator_fixture(tmp_path)
    first_row = operator["split"]["family_partition_reveal"]["rows"][0]
    first_row["family"] = first_row.pop("family_id")
    operator_payload = canonical_json_bytes(operator)
    operator_path.write_bytes(operator_payload)

    with pytest.raises(Evaluation2DValidationError, match="keys mismatch"):
        _LOAD_OPERATOR_BINDING(
            operator_path,
            expected_sha256=hashlib.sha256(operator_payload).hexdigest(),
            campaign_id="campaign-1",
            observations_sha256=_OBSERVATIONS_SHA256,
            expected_truth_sha256=truth_sha256,
            locked_artifacts=locked_artifacts,
        )


def test_truth_scenarios_must_match_committed_family_reveal(tmp_path: Path):
    truth = _truth_arrays()
    truth["scenario"] = np.asarray(["background", "background"])
    truth_path = _write_npz(tmp_path / "truth.npz", truth)
    prediction_path = _write_npz(tmp_path / "prediction.npz", _prediction_arrays())

    with pytest.raises(Evaluation2DValidationError, match="committed family reveal"):
        _evaluate_locked(truth_path, prediction_path)


def test_area_weighting_pairing_bootstrap_and_scenario_strata(tmp_path):
    truth_path, prediction_path = _artifacts(tmp_path)

    first = _evaluate_locked(truth_path, prediction_path)
    second = _evaluate_locked(truth_path, prediction_path)

    assert first == second
    assert first["audience"] == "benchmark_operator_only_after_predictions_locked"
    assert first["schema"] == EVALUATION_SCHEMA
    assert [row["sample_index"] for row in first["per_sample"]] == [10, 20]
    sample_10, sample_20 = first["per_sample"]
    assert sample_10["scenario"] == "background"
    assert sample_10["has_fault"] is False
    assert sample_10["mae_log10_resistivity"] == pytest.approx(0.5)
    assert sample_10["rmse_log10_resistivity"] == pytest.approx(np.sqrt(0.5))
    assert sample_10["rmse_log10_resistivity"] != pytest.approx(np.sqrt(1.0 / 3.0))
    assert sample_20["mae_log10_resistivity"] == pytest.approx(2.0)
    assert sample_20["rmse_log10_resistivity"] == pytest.approx(2.0)

    expected_mean = (np.sqrt(0.5) + 2.0) / 2.0
    assert first["overall"]["rmse_log10_resistivity"]["mean"][
        "estimate"
    ] == pytest.approx(expected_mean)
    assert first["overall"]["rmse_log10_resistivity"]["median"][
        "estimate"
    ] == pytest.approx(expected_mean)
    assert first["by_scenario"]["background"]["n_samples"] == 1
    assert first["by_scenario"]["salt"]["n_samples"] == 1
    assert first["by_scenario"]["background"]["mae_log10_resistivity"]["mean"][
        "estimate"
    ] == pytest.approx(0.5)
    assert first["bootstrap_contract"] == {
        "algorithm": "numpy_random_PCG64_percentile_linear",
        "confidence": 0.8,
        "cross_method_effect_ci": False,
        "headline_eligible": False,
        "hierarchical": False,
        "n_resamples": 100,
        "resampled_fields": [
            "rmse_log10_resistivity",
            "mae_log10_resistivity",
        ],
        "pairing": "identical_resampled_sample_rows_for_all_metrics",
        "seed": 17,
        "scope": "single_method_single_campaign_sample_level_descriptive",
    }
    assert first["release_gate"]["public_release_allowed"] is False
    assert (
        first["implementation"]["source_sha256"]
        == hashlib.sha256(
            Path("src/pimsr_benchmarks/evaluation2d.py").read_bytes()
        ).hexdigest()
    )
    assert first["implementation"]["numpy_version"] == np.__version__
    assert set(first["implementation"]) == {
        "distribution_version",
        "git_commit",
        "git_dirty_tree",
        "git_head_commit",
        "numpy_version",
        "python_version",
        "source_file",
        "source_sha256",
        "source_size_bytes",
    }
    assert first["physics_misfit"]["included"] is False
    assert (
        first["inputs"]["truth"]["sha256"]
        == hashlib.sha256(truth_path.read_bytes()).hexdigest()
    )
    assert (
        first["inputs"]["prediction"]["sha256"]
        == hashlib.sha256(prediction_path.read_bytes()).hexdigest()
    )
    assert first["inputs"]["observations"]["sha256"] == _OBSERVATIONS_SHA256
    assert len(first["per_depth"]) == 2
    assert first["per_depth"][0]["depth_cell_lower_edge_m"] == pytest.approx(0.0)
    assert first["per_depth"][0]["depth_cell_upper_edge_m"] == pytest.approx(2.0)
    assert first["metric_contract"]["scoring_domain"] == {
        "depth_cell_edges_m": [0.0, 2.0, 4.0],
        "mask": "all_truth_grid_cells",
        "support": "full_grid_voronoi_cells_from_centers",
        "x_cell_edges_m": [-0.5, 0.5, 2.5, 5.5],
    }


def test_cell_widths_use_midpoints_and_extrapolated_boundaries():
    widths = cell_widths_from_centers(np.asarray([0.0, 1.0, 4.0], dtype="<f8"))
    np.testing.assert_allclose(widths, [1.0, 2.0, 3.0])
    edges = cell_edges_from_centers(np.asarray([0.0, 1.0, 4.0], dtype="<f8"))
    np.testing.assert_allclose(edges, [-0.5, 0.5, 2.5, 5.5])


def test_implementation_identity_matches_clean_filtered_pinned_blob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source_payload = b"value = 1\r\n"
    repository, source, commit = _implementation_repository(
        tmp_path,
        source_payload=source_payload,
        attributes="src/pimsr_benchmarks/evaluation2d.py text eol=lf\n",
    )
    relative_source = "src/pimsr_benchmarks/evaluation2d.py"
    raw_blob = (
        _git(repository, "hash-object", "--stdin", input_payload=source_payload)
        .decode("ascii")
        .strip()
    )
    pinned_blob = (
        _git(repository, "rev-parse", f"{commit}:{relative_source}")
        .decode("ascii")
        .strip()
    )
    assert raw_blob != pinned_blob
    monkeypatch.setattr(evaluation2d, "__file__", str(source))

    identity = _IMPLEMENTATION_IDENTITY()

    assert identity["git_commit"] == commit
    assert identity["git_head_commit"] == commit
    assert identity["git_dirty_tree"] is False
    assert identity["source_sha256"] == hashlib.sha256(source_payload).hexdigest()
    assert identity["source_size_bytes"] == len(source_payload)


def test_implementation_identity_rejects_payload_outside_pinned_blob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _, source, _ = _implementation_repository(
        tmp_path,
        source_payload=b"value = 1\n",
    )
    source.write_bytes(b"value = 'tampered'\n")
    monkeypatch.setattr(evaluation2d, "__file__", str(source))

    with pytest.raises(Evaluation2DValidationError, match="pinned commit blob"):
        _IMPLEMENTATION_IDENTITY()


def test_implementation_identity_rejects_same_payload_path_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source_payload = b"value = 1\n"
    _, source, _ = _implementation_repository(
        tmp_path,
        source_payload=source_payload,
    )
    monkeypatch.setattr(evaluation2d, "__file__", str(source))
    real_git_bytes = evaluation2d._git_bytes
    swapped = False

    def swapping_git(*args, **kwargs):
        nonlocal swapped
        result = real_git_bytes(*args, **kwargs)
        arguments = args[1:]
        if not swapped and arguments[:2] == ("hash-object", "--stdin"):
            replacement = source.with_name("replacement.py")
            replacement.write_bytes(source_payload)
            os.replace(replacement, source)
            swapped = True
        return result

    monkeypatch.setattr(evaluation2d, "_git_bytes", swapping_git)

    with pytest.raises(Evaluation2DValidationError, match="pathname was replaced"):
        _IMPLEMENTATION_IDENTITY()


def test_implementation_identity_rejects_parent_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source_payload = b"value = 1\n"
    _, source, _ = _implementation_repository(
        tmp_path,
        source_payload=source_payload,
    )
    monkeypatch.setattr(evaluation2d, "__file__", str(source))
    real_git_bytes = evaluation2d._git_bytes
    swapped = False

    def swapping_git(*args, **kwargs):
        nonlocal swapped
        result = real_git_bytes(*args, **kwargs)
        arguments = args[1:]
        if not swapped and arguments[:2] == ("hash-object", "--stdin"):
            original_parent = source.parent
            moved_parent = original_parent.with_name("pimsr_benchmarks-original")
            original_parent.rename(moved_parent)
            original_parent.mkdir()
            source.write_bytes(source_payload)
            swapped = True
        return result

    monkeypatch.setattr(evaluation2d, "_git_bytes", swapping_git)

    with pytest.raises(Evaluation2DValidationError, match="source parent changed"):
        _IMPLEMENTATION_IDENTITY()


def test_report_publication_is_canonical_sealed_and_no_overwrite(tmp_path):
    truth_path, prediction_path = _artifacts(tmp_path)
    report = _evaluate_locked(truth_path, prediction_path)
    destination = tmp_path / "results" / "evaluation.json"

    stale_fixed_partial = destination.with_name("evaluation.json.part")
    stale_fixed_partial.parent.mkdir(parents=True, exist_ok=True)
    stale_fixed_partial.write_bytes(b"foreign old partial")

    receipt = publish_evaluation_2d_receipt(report, destination)

    assert receipt == Evaluation2DPublicationReceipt(
        destination.absolute(),
        hashlib.sha256(destination.read_bytes()).hexdigest(),
        destination.stat().st_size,
    )
    assert destination.stat().st_mode & 0o222 == 0
    assert stale_fixed_partial.read_bytes() == b"foreign old partial"
    parsed = json.loads(destination.read_text(encoding="utf-8"))
    assert destination.read_bytes() == canonical_json_bytes(parsed)
    with pytest.raises(Evaluation2DPublicationError, match="refusing to overwrite"):
        publish_evaluation_2d(report, destination)

    stale_target = tmp_path / "stale.json"
    stale_partial = stale_target.with_name("stale.json.part")
    stale_partial.write_bytes(b"unfinished")
    stale_receipt = publish_evaluation_2d_receipt(report, stale_target)
    assert stale_receipt.path == stale_target.absolute()
    assert stale_partial.read_bytes() == b"unfinished"
    assert stale_target.exists()

    compatibility_target = tmp_path / "compatible-path-return.json"
    published_path = publish_evaluation_2d(report, compatibility_target)
    assert isinstance(published_path, Path)
    assert published_path == compatibility_target.absolute()


def test_legacy_publication_api_preserves_a_relative_return_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    segment = Path("existing-segment")
    segment.mkdir()
    requested = segment / ".." / "relative-evaluation.json"

    returned = publish_evaluation_2d({"value": 1}, requested)

    assert returned == requested
    assert not returned.is_absolute()
    assert returned.exists()
    assert (tmp_path / "relative-evaluation.json").exists()


def test_report_publication_early_failure_leaves_exclusive_artifact_sealed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    truth_path, prediction_path = _artifacts(tmp_path)
    report = _evaluate_locked(truth_path, prediction_path)
    destination = tmp_path / "evaluation.json"
    expected = canonical_json_bytes(report)

    def interrupted_write(descriptor: int, payload: bytes) -> None:
        assert payload == expected
        assert os.write(descriptor, payload[:11]) == 11
        raise KeyboardInterrupt

    monkeypatch.setattr(evaluation2d, "_write_all_descriptor", interrupted_write)

    with pytest.raises(KeyboardInterrupt):
        publish_evaluation_2d(report, destination)

    assert destination.read_bytes() == expected[:11]
    assert destination.stat().st_mode & 0o222 == 0


def test_report_receipt_reseals_mode_changed_before_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    destination = tmp_path / "resealed.json"
    real_receipt = evaluation2d._stable_evaluation_receipt

    def change_mode_then_reopen(*args, **kwargs):
        os.chmod(destination, 0o600)
        return real_receipt(*args, **kwargs)

    monkeypatch.setattr(
        evaluation2d, "_stable_evaluation_receipt", change_mode_then_reopen
    )
    receipt = publish_evaluation_2d_receipt({"value": 1}, destination)

    assert destination.stat().st_mode & 0o222 == 0
    assert receipt.sha256 == hashlib.sha256(destination.read_bytes()).hexdigest()
    assert receipt.size_bytes == destination.stat().st_size


def test_report_publication_detects_same_inode_same_size_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    truth_path, prediction_path = _artifacts(tmp_path)
    report = _evaluate_locked(truth_path, prediction_path)
    destination = tmp_path / "evaluation.json"
    real_read = evaluation2d._read_all_descriptor
    calls = 0

    def mutate_after_first_read(descriptor: int) -> bytes:
        nonlocal calls
        payload = real_read(descriptor)
        calls += 1
        if calls == 1:
            changed = bytearray(payload)
            changed[-2] ^= 1
            os.chmod(destination, 0o600)
            with destination.open("r+b") as stream:
                stream.write(changed)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(destination, 0o444)
        return payload

    monkeypatch.setattr(evaluation2d, "_read_all_descriptor", mutate_after_first_read)

    with pytest.raises(
        Evaluation2DPublicationError, match="changed during|cannot verify"
    ):
        publish_evaluation_2d(report, destination)

    assert destination.exists()
    assert destination.stat().st_mode & 0o222 == 0


def test_report_receipt_detects_retained_writer_during_final_parent_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    destination = tmp_path / "retained-writer.json"
    replacement = canonical_json_bytes({"value": 2})
    real_seal = evaluation2d._seal_publication_descriptor
    real_parent_identity = evaluation2d._publication_parent_identity
    writer: int | None = None
    parent_checks = 0
    writer_denied = False

    def seal_with_retained_writer(descriptor: int) -> None:
        nonlocal writer, writer_denied
        os.chmod(destination, 0o600)
        try:
            writer = os.open(destination, os.O_RDWR | getattr(os, "O_BINARY", 0))
        except OSError:
            if os.name != "nt":
                raise
            writer_denied = True
        real_seal(descriptor)

    def mutate_during_parent_check(path: Path) -> tuple[int, int]:
        nonlocal parent_checks
        identity = real_parent_identity(path)
        parent_checks += 1
        if parent_checks == 3 and writer is not None:
            os.lseek(writer, 0, os.SEEK_SET)
            assert os.write(writer, replacement) == len(replacement)
            os.fsync(writer)
        return identity

    monkeypatch.setattr(
        evaluation2d, "_seal_publication_descriptor", seal_with_retained_writer
    )
    monkeypatch.setattr(
        evaluation2d, "_publication_parent_identity", mutate_during_parent_check
    )
    try:
        if os.name == "nt":
            receipt = publish_evaluation_2d_receipt({"value": 1}, destination)
            assert writer_denied
            assert receipt.sha256 == hashlib.sha256(destination.read_bytes()).hexdigest()
            assert receipt.size_bytes == destination.stat().st_size
        else:
            with pytest.raises(Evaluation2DPublicationError, match="changed during"):
                publish_evaluation_2d_receipt({"value": 1}, destination)
    finally:
        if writer is not None:
            os.close(writer)

    if os.name == "nt":
        assert destination.read_bytes() == canonical_json_bytes({"value": 1})
    else:
        assert destination.read_bytes() == replacement


@pytest.mark.skipif(os.name != "nt", reason="Windows share-mode contract")
def test_report_receipt_denies_a_writer_after_the_last_path_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    destination = tmp_path / "last-snapshot.json"
    real_signature = evaluation2d._publication_signature
    signature_calls = 0
    writer_denied = False

    def signature_with_writer_attempt(value: os.stat_result) -> tuple[int, ...]:
        nonlocal signature_calls, writer_denied
        signature_calls += 1
        if signature_calls == 5:
            os.chmod(destination, 0o600)
            try:
                writer = os.open(destination, os.O_RDWR | getattr(os, "O_BINARY", 0))
            except OSError:
                writer_denied = True
            else:  # pragma: no cover - this is the security regression
                os.close(writer)
        return real_signature(value)

    monkeypatch.setattr(
        evaluation2d, "_publication_signature", signature_with_writer_attempt
    )
    receipt = publish_evaluation_2d_receipt({"value": 1}, destination)

    assert signature_calls == 5
    assert writer_denied
    assert receipt.sha256 == hashlib.sha256(destination.read_bytes()).hexdigest()


@pytest.mark.skipif(os.name != "nt", reason="Windows share-mode contract")
def test_report_receipt_rejects_a_writer_retained_before_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    destination = tmp_path / "writer-before-reopen.json"
    real_receipt = evaluation2d._stable_evaluation_receipt
    writer: int | None = None

    def reopen_with_retained_writer(*args, **kwargs):
        nonlocal writer
        os.chmod(destination, 0o600)
        writer = os.open(destination, os.O_RDWR | getattr(os, "O_BINARY", 0))
        return real_receipt(*args, **kwargs)

    monkeypatch.setattr(
        evaluation2d, "_stable_evaluation_receipt", reopen_with_retained_writer
    )
    try:
        with pytest.raises(Evaluation2DPublicationError, match="cannot verify"):
            publish_evaluation_2d_receipt({"value": 1}, destination)
    finally:
        if writer is not None:
            os.close(writer)
        os.chmod(destination, 0o444)

    assert destination.stat().st_mode & 0o222 == 0


def test_report_publication_never_deletes_a_replacement_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    truth_path, prediction_path = _artifacts(tmp_path)
    report = _evaluate_locked(truth_path, prediction_path)
    destination = tmp_path / "evaluation.json"

    def replace_then_fail(*_args, **_kwargs):
        os.chmod(destination, 0o600)
        destination.unlink()
        destination.write_bytes(b"foreign replacement")
        raise Evaluation2DPublicationError("injected replacement")

    monkeypatch.setattr(evaluation2d, "_stable_evaluation_receipt", replace_then_fail)

    with pytest.raises(Evaluation2DPublicationError, match="injected replacement"):
        publish_evaluation_2d(report, destination)

    assert destination.read_bytes() == b"foreign replacement"


def test_report_publication_rejects_hardlink_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    truth_path, prediction_path = _artifacts(tmp_path)
    report = _evaluate_locked(truth_path, prediction_path)
    destination = tmp_path / "aliased.json"
    alias = tmp_path / "alias.json"
    real_seal = evaluation2d._seal_publication_descriptor
    calls = 0

    def seal_then_alias(descriptor: int) -> None:
        nonlocal calls
        real_seal(descriptor)
        calls += 1
        if calls == 1:
            os.link(destination, alias)

    monkeypatch.setattr(evaluation2d, "_seal_publication_descriptor", seal_then_alias)
    with pytest.raises(Evaluation2DPublicationError, match="changed before"):
        publish_evaluation_2d(report, destination)

    assert destination.samefile(alias)


def test_report_publication_detects_parent_replacement_before_final_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    truth_path, prediction_path = _artifacts(tmp_path)
    report = _evaluate_locked(truth_path, prediction_path)
    parent = tmp_path / "publication"
    destination = parent / "evaluation.json"
    displaced = tmp_path / "publication-displaced"
    real_receipt = evaluation2d._stable_evaluation_receipt

    def replace_parent(*args, **kwargs):
        parent.rename(displaced)
        parent.mkdir()
        return real_receipt(*args, **kwargs)

    monkeypatch.setattr(evaluation2d, "_stable_evaluation_receipt", replace_parent)
    with pytest.raises(Evaluation2DPublicationError, match="cannot verify"):
        publish_evaluation_2d(report, destination)

    assert (displaced / destination.name).exists()
    assert not destination.exists()


def test_report_publication_rejects_a_symlinked_parent(tmp_path: Path):
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    try:
        linked_parent.symlink_to(real_parent, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation is unavailable")

    with pytest.raises(Evaluation2DPublicationError, match="must (not|be a real)"):
        publish_evaluation_2d({"value": 1}, linked_parent / "evaluation.json")

    assert not (real_parent / "evaluation.json").exists()


def test_report_publication_does_not_create_through_a_linked_ancestor(tmp_path: Path):
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_ancestor = tmp_path / "linked-ancestor"
    try:
        linked_ancestor.symlink_to(real_parent, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation is unavailable")

    with pytest.raises(Evaluation2DPublicationError, match="ancestor.*(real|link)"):
        publish_evaluation_2d(
            {"value": 1}, linked_ancestor / "missing" / "evaluation.json"
        )

    assert not (real_parent / "missing").exists()


def test_cli_reports_canonical_digest_without_reopening_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    evaluation = {"schema": "test-evaluation", "value": 7}
    destination = tmp_path / "evaluation.json"
    published = False

    monkeypatch.setattr(
        evaluation2d,
        "evaluate_predictions_2d",
        lambda *args, **kwargs: evaluation,
    )

    def fake_publish(value, output_path):
        nonlocal published
        assert value is evaluation
        assert output_path == destination
        published = True
        return Evaluation2DPublicationReceipt(destination, "d" * 64, 321)

    def forbidden_read(path: Path) -> bytes:
        if published and path == destination:
            raise AssertionError("CLI reopened the immutable publication")
        return b""

    monkeypatch.setattr(evaluation2d, "publish_evaluation_2d_receipt", fake_publish)
    monkeypatch.setattr(Path, "read_bytes", forbidden_read)
    result = evaluation2d.main(
        [
            "--truth",
            "truth.npz",
            "--predictions",
            "predictions.npz",
            "--observations",
            "observations.npz",
            "--observation-manifest",
            "observations.json",
            "--runtime",
            "runtime.json",
            "--checkpoint",
            "checkpoint.pt",
            "--source",
            "source.py",
            "--operator-manifest",
            "operator.json",
            "--output",
            str(destination),
            "--preregistration",
            "preregistration.json",
            "--preregistration-sha256",
            "1" * 64,
            "--predictions-lock",
            "predictions-lock.json",
            "--predictions-lock-sha256",
            "2" * 64,
            "--campaign-id",
            "campaign-1",
            "--method-id",
            "pimsr",
            "--training-seed",
            "101",
            "--operator-manifest-sha256",
            "3" * 64,
            "--truth-sha256",
            "4" * 64,
        ]
    )

    assert result == 0
    assert capsys.readouterr().out == (
        f"published {destination} sha256={'d' * 64} size=321\n"
    )


def test_expected_hash_detects_artifact_tampering(tmp_path):
    truth_path, prediction_path = _artifacts(tmp_path)
    pinned_truth = hashlib.sha256(truth_path.read_bytes()).hexdigest()
    pinned_prediction = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
    _evaluate_locked(
        truth_path,
        prediction_path,
        expected_truth_sha256=pinned_truth,
    )

    changed = _prediction_arrays()
    changed["predicted_log10_resistivity"][0, 0, 0] = 0.25
    _write_npz(prediction_path, changed)

    with pytest.raises(Evaluation2DValidationError, match="SHA-256"):
        load_predictions_2d(prediction_path, expected_sha256=pinned_prediction)
    with pytest.raises(Evaluation2DValidationError, match="64 lowercase"):
        load_truth_2d(truth_path, expected_sha256="A" * 64)
    mismatched = _truth_arrays()
    mismatched["observations_sha256"] = np.asarray("2" * 64, dtype="<U64")
    mismatched_truth = _write_npz(tmp_path / "mismatched-truth.npz", mismatched)
    with pytest.raises(Evaluation2DValidationError, match="locked campaign"):
        _evaluate_locked(mismatched_truth, prediction_path)


@pytest.mark.parametrize(
    ("sample_ids", "error"),
    [
        ([10, 10], "duplicate"),
        ([10, 30], "exactly match"),
    ],
)
def test_prediction_rejects_duplicate_missing_or_extra_ids(tmp_path, sample_ids, error):
    truth_path = _write_npz(tmp_path / "truth.npz", _truth_arrays())
    prediction = _prediction_arrays()
    prediction["sample_index"] = np.asarray(sample_ids, dtype="<i8")
    prediction_path = _write_npz(tmp_path / "prediction.npz", prediction)

    with pytest.raises(Evaluation2DValidationError, match=error):
        _evaluate_locked(truth_path, prediction_path)


def test_truth_rejects_duplicate_ids(tmp_path):
    truth = _truth_arrays()
    truth["sample_index"] = np.asarray([20, 20], dtype="<i8")
    path = _write_npz(tmp_path / "truth.npz", truth)

    with pytest.raises(Evaluation2DValidationError, match="duplicate"):
        load_truth_2d(path)


@pytest.mark.parametrize(
    ("field", "replacement", "error"),
    [
        ("schema", np.asarray("wrong-schema"), "schema must"),
        ("schema_version", np.asarray(1, dtype="<i4"), "schema_version"),
        ("sample_index", np.asarray([10, 20], dtype="<i4"), "dtype int64"),
        ("sample_index", np.asarray([10, 20], dtype=">i8"), "dtype int64"),
        ("observations_sha256", np.asarray("bad"), r"Unicode\[64\]"),
        (
            "x_cell_centers_m",
            np.asarray([0.0, 1.0, 4.0], dtype="<f4"),
            "dtype float64",
        ),
        (
            "predicted_log10_resistivity",
            np.zeros((2, 2, 3), dtype="<f8"),
            "dtype float32",
        ),
        (
            "predicted_log10_resistivity",
            np.zeros((2, 2, 3), dtype=">f4"),
            "dtype float32",
        ),
        (
            "predicted_log10_resistivity",
            np.zeros((2, 2), dtype="<f4"),
            "3-dimensional",
        ),
        (
            "predicted_log10_resistivity",
            np.full((2, 2, 3), np.nan, dtype="<f4"),
            "must be finite",
        ),
    ],
)
def test_prediction_schema_shape_dtype_and_finiteness_fail_closed(
    tmp_path, field, replacement, error
):
    prediction = _prediction_arrays()
    prediction[field] = replacement
    path = _write_npz(tmp_path / "prediction.npz", prediction)

    with pytest.raises(Evaluation2DValidationError, match=error):
        load_predictions_2d(path)


@pytest.mark.parametrize(
    ("field", "replacement", "error"),
    [
        ("schema_version", np.asarray(1, dtype="<i8"), "schema_version"),
        ("observations_sha256", np.asarray("bad"), r"Unicode\[64\]"),
        ("scenario", np.asarray(["salt"]), "length must match"),
        ("has_fault", np.asarray([1, 0], dtype="u1"), "dtype bool"),
        (
            "x_cell_centers_m",
            np.asarray([0.0, 0.0, 4.0], dtype="<f8"),
            "strictly increasing",
        ),
        (
            "x_cell_centers_m",
            np.asarray([0.0, 1.0, 4.0], dtype=">f8"),
            "dtype float64",
        ),
        (
            "depth_cell_centers_m",
            np.asarray([1.0, np.inf], dtype="<f8"),
            "must be finite",
        ),
        (
            "truth_log10_resistivity",
            np.zeros((2, 2, 3), dtype="<f8"),
            "dtype float32",
        ),
        (
            "truth_log10_resistivity",
            np.zeros((2, 2, 3), dtype=">f4"),
            "dtype float32",
        ),
        (
            "truth_log10_resistivity",
            np.zeros((2, 3, 2), dtype="<f4"),
            "shape must be",
        ),
        (
            "truth_log10_resistivity",
            np.full((2, 2, 3), np.inf, dtype="<f4"),
            "must be finite",
        ),
    ],
)
def test_truth_schema_shape_dtype_and_finiteness_fail_closed(
    tmp_path, field, replacement, error
):
    truth = _truth_arrays()
    truth[field] = replacement
    path = _write_npz(tmp_path / "truth.npz", truth)

    with pytest.raises(Evaluation2DValidationError, match=error):
        load_truth_2d(path)


def test_npz_exact_keys_and_duplicate_members_are_rejected(tmp_path):
    extra = _prediction_arrays()
    extra["truth_log10_resistivity"] = np.zeros((2, 2, 3), dtype="<f4")
    extra_path = _write_npz(tmp_path / "extra.npz", extra)
    with pytest.raises(Evaluation2DValidationError, match="members mismatch"):
        load_predictions_2d(extra_path)

    duplicate_path = _write_npz(tmp_path / "duplicate.npz", _prediction_arrays())
    with (
        pytest.warns(UserWarning, match="Duplicate name"),
        zipfile.ZipFile(duplicate_path, "a") as archive,
    ):
        archive.writestr("schema.npy", b"duplicate")
    with pytest.raises(Evaluation2DValidationError, match="duplicate archive"):
        load_predictions_2d(duplicate_path)

    compressed_path = tmp_path / "compressed.npz"
    np.savez_compressed(compressed_path, **_prediction_arrays())
    with pytest.raises(Evaluation2DValidationError, match="ZIP_STORED"):
        load_predictions_2d(compressed_path)

    reordered = _prediction_arrays()
    reordered = {"sample_index": reordered.pop("sample_index"), **reordered}
    reordered_path = _write_npz(tmp_path / "reordered.npz", reordered)
    with pytest.raises(Evaluation2DValidationError, match="canonical order"):
        load_predictions_2d(reordered_path)


def test_prediction_grid_shape_must_exactly_match_truth(tmp_path):
    truth_path = _write_npz(tmp_path / "truth.npz", _truth_arrays())
    prediction = _prediction_arrays()
    prediction["predicted_log10_resistivity"] = np.zeros((2, 2, 4), dtype="<f4")
    prediction_path = _write_npz(tmp_path / "prediction.npz", prediction)

    with pytest.raises(Evaluation2DValidationError, match="shape must be"):
        _evaluate_locked(truth_path, prediction_path)


@pytest.mark.parametrize(
    ("field", "replacement", "error"),
    [
        ("observations_sha256", np.asarray("2" * 64, dtype="<U64"), "locked campaign"),
        (
            "x_cell_centers_m",
            np.asarray([0.0, 2.0, 4.0], dtype="<f8"),
            "x grid must exactly match",
        ),
        (
            "depth_cell_centers_m",
            np.asarray([1.0, 4.0], dtype="<f8"),
            "depth grid must exactly match",
        ),
    ],
)
def test_prediction_must_bind_exact_campaign_and_grid(
    tmp_path: Path, field: str, replacement: np.ndarray, error: str
):
    truth_path = _write_npz(tmp_path / "truth.npz", _truth_arrays())
    prediction = _prediction_arrays()
    prediction[field] = replacement
    prediction_path = _write_npz(tmp_path / "prediction.npz", prediction)

    with pytest.raises(Evaluation2DValidationError, match=error):
        _evaluate_locked(truth_path, prediction_path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"confidence": 1.0},
        {"confidence": np.nan},
        {"n_resamples": 0},
        {"n_resamples": True},
        {"seed": -1},
        {"seed": 1.5},
    ],
)
def test_bootstrap_options_are_not_caller_selectable(tmp_path, kwargs):
    truth_path, prediction_path = _artifacts(tmp_path)

    with pytest.raises(TypeError, match="unexpected keyword"):
        _evaluate_locked(truth_path, prediction_path, **kwargs)


def test_invalid_or_missing_lock_never_opens_operator_or_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    truth_path, prediction_path = _artifacts(tmp_path)
    events: list[str] = []

    def invalid_lock(*args, **kwargs):
        events.append("lock")
        raise PredictionLock2DValidationError("invalid lock")

    monkeypatch.setattr(evaluation2d, "validate_prediction_lock_2d", invalid_lock)
    monkeypatch.setattr(
        evaluation2d,
        "_load_operator_binding",
        lambda *a, **k: events.append("operator"),
    )
    monkeypatch.setattr(
        evaluation2d,
        "load_truth_2d",
        lambda *a, **k: events.append("truth"),
    )

    with pytest.raises(Evaluation2DValidationError, match="before truth access"):
        _evaluate_locked(
            truth_path,
            prediction_path,
            expected_truth_sha256="9" * 64,
        )
    assert events == ["lock"]


def test_artifact_operator_truth_access_order_is_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    truth_path, prediction_path = _artifacts(tmp_path)
    events: list[str] = []
    original_lock = evaluation2d.validate_prediction_lock_2d

    def checked_lock(*args, **kwargs):
        events.append("lock")
        return original_lock(*args, **kwargs)

    def checked_artifacts(*args, **kwargs):
        events.append("artifacts")

    def checked_operator(*args, **kwargs):
        events.append("operator")
        return SimpleNamespace(
            family_by_sample={10: "background", 20: "salt"},
            family_commitment_sha256="4" * 64,
            snapshot=SimpleNamespace(sha256="3" * 64, size_bytes=1),
            truth_sha256=kwargs["expected_truth_sha256"],
            truth_size_bytes=None,
        )

    def checked_predictions(*args, **kwargs):
        events.append("prediction_parse")
        return load_predictions_2d(*args, **kwargs)

    def stop_at_truth(*args, **kwargs):
        events.append("truth")
        raise RuntimeError("truth-open-sentinel")

    monkeypatch.setattr(evaluation2d, "validate_prediction_lock_2d", checked_lock)
    monkeypatch.setattr(
        evaluation2d, "validate_locked_run_artifacts_2d", checked_artifacts
    )
    monkeypatch.setattr(evaluation2d, "_load_operator_binding", checked_operator)
    monkeypatch.setattr(evaluation2d, "load_predictions_2d", checked_predictions)
    monkeypatch.setattr(evaluation2d, "load_truth_2d", stop_at_truth)

    with pytest.raises(RuntimeError, match="truth-open-sentinel"):
        _evaluate_locked(truth_path, prediction_path)
    assert events == ["lock", "artifacts", "prediction_parse", "operator", "truth"]


def test_invalid_locked_artifact_never_opens_operator_or_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    truth_path, prediction_path = _artifacts(tmp_path)
    events: list[str] = []

    def invalid_artifacts(*args, **kwargs):
        events.append("artifacts")
        raise PredictionLock2DValidationError("runtime hash mismatch")

    monkeypatch.setattr(
        evaluation2d, "validate_locked_run_artifacts_2d", invalid_artifacts
    )
    monkeypatch.setattr(
        evaluation2d,
        "_load_operator_binding",
        lambda *a, **k: events.append("operator"),
    )
    monkeypatch.setattr(
        evaluation2d,
        "load_truth_2d",
        lambda *a, **k: events.append("truth"),
    )
    with pytest.raises(Evaluation2DValidationError, match="before operator/truth"):
        _evaluate_locked(truth_path, prediction_path)
    assert events == ["artifacts"]


def test_canonical_json_rejects_non_finite_values():
    with pytest.raises(Evaluation2DPublicationError, match="canonical JSON"):
        canonical_json_bytes({"invalid": np.nan})
