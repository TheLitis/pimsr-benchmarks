from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from pimsr_benchmarks import prediction_lock2d
from pimsr_benchmarks.prediction_lock2d import (
    LOCK_INPUT_SCHEMA,
    LOCK_SCHEMA,
    LockedRun2D,
    PredictionLock2DPublicationError,
    PredictionLock2DPublicationReceipt,
    PredictionLock2DValidationError,
    canonical_json_bytes,
    create_prediction_lock_2d,
    publish_prediction_lock_2d,
    snapshot_regular_file,
    validate_locked_run_artifacts_2d,
    validate_prediction_lock_2d,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: bytes) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {"path": str(path), "sha256": _sha(path)}


def _write_json(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))
    return _sha(path)


def _method(
    root: Path,
    method_id: str,
    *,
    source_sha256: str,
    adapter_sha256: str,
) -> dict:
    source_key = {
        "pimsr": "network_source_sha256",
        "mtdlpy": "dinknet_source_sha256",
        "mt2dinv_densenet": "architecture_source_sha256",
    }[method_id]
    schedules = {
        "pimsr": {"batch_size": 64, "epochs": 80},
        "mtdlpy": {
            "batch_size": 4,
            "epochs": 10,
            "optimizer": {"learning_rate": 0.0001, "name": "Adam"},
            "recipe_id": "benchmark_reviewed_v1",
        },
        "mt2dinv_densenet": {
            "batch_size": 100,
            "epochs": 200,
            "optimizer": {"learning_rate": 0.0001, "name": "Adam"},
        },
    }
    return {
        "id": method_id,
        "implementation": {
            "adapter_repository_commit": "a" * 40,
            "adapter_source_path": f"adapters/{method_id}.py",
            "adapter_source_sha256": adapter_sha256,
            "repository_commit": {
                "pimsr": "1" * 40,
                "mtdlpy": "2" * 40,
                "mt2dinv_densenet": "3" * 40,
            }[method_id],
            source_key: source_sha256,
        },
        "role": "candidate" if method_id == "pimsr" else "reference",
        "training": schedules[method_id],
    }


def _matrix(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, str, Path, str, dict]:
    methods = ("pimsr", "mtdlpy", "mt2dinv_densenet")
    seeds = (101, 102, 103, 104, 105)
    campaign_ids = tuple(f"campaign-{index}" for index in range(1, 6))
    adapter_hashes: dict[str, str] = {}
    source_hashes: dict[str, str] = {}
    source_paths: dict[str, Path] = {}
    source_repositories: dict[str, Path] = {}
    for method_id in methods:
        adapter = root / "adapters" / f"{method_id}.py"
        _write(adapter, f"adapter:{method_id}".encode())
        adapter_hashes[method_id] = _sha(adapter)
        repository = root / f"source-repo-{method_id}"
        source = repository / "source.py"
        _write(source, f"source:{method_id}".encode())
        source_hashes[method_id] = _sha(source)
        source_paths[method_id] = source
        source_repositories[method_id] = repository
    prereg = {
        "datasets": {
            "hidden_test": {
                "campaigns": {
                    "campaign_ids": list(campaign_ids),
                    "count": 5,
                    "samples_per_campaign": 500,
                    "total_samples": 2500,
                },
                "prediction_lock_gate": {"locked_artifact_count": 75},
            },
            "train": {"artifact": {"sha256": "4" * 64}},
            "validation": {"artifact": {"sha256": "5" * 64}},
        },
        "methods": [
            _method(
                root,
                method_id,
                source_sha256=source_hashes[method_id],
                adapter_sha256=adapter_hashes[method_id],
            )
            for method_id in methods
        ],
        "family_partition": {
            "schema": "pimsr-sota-2d-family-partition-commitment",
            "schema_version": 1,
            "families": [
                "background",
                "aquifer",
                "hydrocarbon",
                "salt",
                "geothermal",
            ],
            "bases_per_family": 20,
            "noise_realizations_per_base": 5,
            "commitment_contract": {
                "algorithm": "SHA-256",
                "canonicalization": ("utf8-canonical-json-sort-keys-compact-newline-v1"),
                "domain_separator": "pimsr-sota-2d-family-partition/v1",
                "nonce_encoding": "lowercase_hex_32_bytes",
            },
        },
        "preregistration_id": "test-prereg-v1",
        "run_seeds": list(seeds),
        "schema": "pimsr-sota-2d-common-retrain-preregistration",
        "schema_version": 1,
        "statistical_analysis": {
            "dominance_gate": (
                "one_sided_95_percent_iut_upper_below_zero_against_both_references"
            ),
            "effect": {
                "candidate": "pimsr",
                "references": ["mtdlpy", "mt2dinv_densenet"],
            },
            "hierarchical_paired_bootstrap": {
                "confidence": 0.95,
                "n_resamples": 10_000,
                "point_aggregation": (
                    "equal_family_equal_base_equal_noise_mean_across_"
                    "paired_training_seeds_and_campaigns"
                ),
                "resampling_levels": [
                    "training_seed",
                    "campaign",
                    "geological_family",
                    "base_model_within_family",
                    "noise_realization_within_base_model",
                ],
                "rng_seed": 20260824,
            },
            "multiplicity_policy": (
                "none_for_single_intersection_union_claim_individual_pairwise_descriptive"
            ),
        },
    }
    prereg_path = root / "config" / "prereg.json"
    prereg_sha = _write_json(prereg_path, prereg)
    checkpoints: dict[tuple[str, int], dict[str, str]] = {}
    for method_id in methods:
        for seed in seeds:
            checkpoints[(method_id, seed)] = _write(
                root / "artifacts" / f"{method_id}-{seed}.checkpoint",
                f"checkpoint:{method_id}:{seed}".encode(),
            )
    campaigns: list[dict] = []
    for campaign_id in campaign_ids:
        observations = _write(
            root / "artifacts" / f"{campaign_id}.observations",
            f"observations:{campaign_id}".encode(),
        )
        manifest = root / "artifacts" / f"{campaign_id}.public.json"
        manifest_ref = _write(manifest, f'{{"split_id":"{campaign_id}"}}'.encode())
        runs: list[dict] = []
        for method_id in methods:
            for seed in seeds:
                stem = f"{campaign_id}-{method_id}-{seed}"
                prediction = _write(
                    root / "artifacts" / f"{stem}.prediction",
                    f"prediction:{stem}".encode(),
                )
                runtime = root / "artifacts" / f"{stem}.runtime.json"
                runtime_ref = _write(runtime, b"{}")
                runs.append(
                    {
                        "checkpoint": checkpoints[(method_id, seed)],
                        "method_id": method_id,
                        "prediction": prediction,
                        "runtime": runtime_ref,
                        "source": {
                            "path": str(source_paths[method_id]),
                            "repository_path": str(source_repositories[method_id]),
                            "sha256": source_hashes[method_id],
                        },
                        "training_seed": seed,
                    }
                )
        campaigns.append(
            {
                "campaign_id": campaign_id,
                "observation_manifest": manifest_ref,
                "observations": observations,
                "runs": runs,
            }
        )
    input_value = {
        "audience": prediction_lock2d.LOCK_AUDIENCE,
        "campaigns": campaigns,
        "preregistration_sha256": prereg_sha,
        "schema": LOCK_INPUT_SCHEMA,
        "schema_version": 1,
    }
    input_path = root / "lock-input.json"
    input_sha = _write_json(input_path, input_value)
    monkeypatch.setattr(
        prediction_lock2d,
        "_observation_identity",
        lambda snapshot, manifest, campaign_id, family_partition: (
            500,
            tuple(range(500)),
            np.asarray([0.0, 1.0], dtype="<f8"),
            np.asarray([1.0, 2.0], dtype="<f8"),
        ),
    )
    monkeypatch.setattr(prediction_lock2d, "_prediction_identity", lambda *a, **k: None)
    monkeypatch.setattr(
        prediction_lock2d, "_validate_source_repository", lambda *a, **k: None
    )
    monkeypatch.setattr(prediction_lock2d, "_runtime_bindings", lambda *a, **k: None)
    return prereg_path, prereg_sha, input_path, input_sha, input_value


def _create(matrix: tuple[Path, str, Path, str, dict]) -> dict:
    prereg_path, prereg_sha, input_path, input_sha, _ = matrix
    return create_prediction_lock_2d(
        prereg_path,
        input_path,
        expected_preregistration_sha256=prereg_sha,
        expected_input_manifest_sha256=input_sha,
    )


def test_complete_matrix_lock_is_path_free_publishable_and_validatable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    matrix = _matrix(tmp_path, monkeypatch)
    lock = _create(matrix)
    assert lock["schema"] == LOCK_SCHEMA
    assert lock["schema_version"] == 2
    assert lock["design"]["run_count"] == 75
    assert lock["design"]["checkpoint_count"] == 15
    assert len(lock["runs"]) == 75
    serialized = canonical_json_bytes(lock).lower()
    for prohibited in (b"truth", b"operator", b"evaluation", b'"path"'):
        assert prohibited not in serialized
    lock_path = tmp_path / "predictions-lock.json"
    publish_prediction_lock_2d(lock, lock_path)
    validated = validate_prediction_lock_2d(
        matrix[0],
        lock_path,
        expected_preregistration_sha256=matrix[1],
        expected_lock_sha256=_sha(lock_path),
    )
    assert len(validated.runs) == 75
    assert validated.require_run("campaign-1", "pimsr", 101).method_id == "pimsr"


def test_family_partition_policy_rejects_posthoc_relabelling():
    policy = {
        "schema": prediction_lock2d.FAMILY_PARTITION_SCHEMA,
        "schema_version": prediction_lock2d.FAMILY_PARTITION_SCHEMA_VERSION,
        "families": list(prediction_lock2d.GEOLOGICAL_FAMILIES),
        "bases_per_family": 20,
        "noise_realizations_per_base": 5,
        "commitment_contract": dict(prediction_lock2d.FAMILY_COMMITMENT_CONTRACT),
    }
    assert (
        prediction_lock2d._family_partition_policy({"family_partition": policy}) == policy
    )
    policy["families"] = [f"family-{index}" for index in range(100)]
    with pytest.raises(PredictionLock2DValidationError, match="5x20x5"):
        prediction_lock2d._family_partition_policy({"family_partition": policy})


def test_missing_or_relabelled_cell_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    matrix = _matrix(tmp_path, monkeypatch)
    input_value = matrix[4]
    input_value["campaigns"][0]["runs"].pop()
    input_sha = _write_json(matrix[2], input_value)
    with pytest.raises(PredictionLock2DValidationError, match="exactly 15"):
        create_prediction_lock_2d(
            matrix[0],
            matrix[2],
            expected_preregistration_sha256=matrix[1],
            expected_input_manifest_sha256=input_sha,
        )


def test_mtdlpy_runtime_bindings_accept_the_published_v3_closure():
    def snapshot(label: str, inode: int) -> prediction_lock2d.ArtifactSnapshot:
        payload = label.encode("utf-8")
        return prediction_lock2d.ArtifactSnapshot(
            path=Path(label),
            payload=payload,
            sha256=hashlib.sha256(payload).hexdigest(),
            device=1,
            inode=inode,
        )

    def artifact(digest: str) -> dict[str, str]:
        return {"sha256": digest}

    observations = snapshot("observations.npz", 1)
    prediction = snapshot("predictions.npz", 2)
    checkpoint = snapshot("checkpoint.pt", 3)
    source = snapshot("DinkNet.py", 4)
    adapter = snapshot("mtdlpy.py", 5)
    train_sha256 = "4" * 64
    validation_sha256 = "5" * 64
    runner_sha256 = "6" * 64
    weights_sha256 = "7" * 64
    adapter_artifact = artifact(adapter.sha256)
    source_artifact = artifact(source.sha256)
    runner_artifact = artifact(runner_sha256)
    weights_artifact = artifact(weights_sha256)
    source_artifacts = {
        "adapter_source": adapter_artifact,
        "artifact_guard_source": artifact("8" * 64),
        "dataset_contract_loader_source": artifact("9" * 64),
        "dinknet_source": source_artifact,
        "heldout_observations": artifact(observations.sha256),
        "imagenet_weights": weights_artifact,
        "materializer_contract_source": artifact("a" * 64),
        "runner_source": runner_artifact,
        "train_dataset": artifact(train_sha256),
        "validation_dataset": artifact(validation_sha256),
    }
    closure = {
        "cli_entrypoint_source_included": True,
        "required_local_python_source_artifacts_recorded": True,
        "evidence_scope": (
            "direct_python_source_artifacts_and_distribution_version_strings"
        ),
        "fixed_imagenet_weights": weights_artifact,
        "local_source_artifacts": {
            "adapter": adapter_artifact,
            "cli_runner": runner_artifact,
            "dataset2d_materialization": artifact("a" * 64),
            "pimsr_inversion_contracts2d": artifact("b" * 64),
            "runner2d": artifact("c" * 64),
            "upstream_dinknet": source_artifact,
        },
        "native_binary_environment_complete": False,
        "packages": {
            "h5py": "test",
            "numpy": "test",
            "pimsr-inversion": "test",
            "torch": "test",
            "torchvision": "test",
        },
        "python": {"implementation": "CPython", "version": "test"},
        "schema": "pimsr-mtdlpy-dependency-closure",
        "schema_version": 3,
    }
    closure_sha256 = prediction_lock2d._canonical_object_sha256(closure)
    expected_training = {
        "batch_size": 4,
        "early_stopping": "none_run_all_10_epochs",
        "epochs": 10,
        "gradient_clip_max_norm": 0.1,
        "optimizer": {
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "learning_rate": 0.0001,
            "name": "Adam",
            "weight_decay": 0.0,
        },
        "recipe_id": "benchmark_reviewed_v1",
        "scheduler": "none",
    }
    actual_training = {
        "batch_size": 4,
        "campaign_seeds": [101, 102, 103, 104, 105],
        "checkpoint_selection": "lowest validation MSE; strict less-than; first tie",
        "early_stopping": None,
        "epochs": 10,
        "gradient_clip_max_norm": 0.1,
        "loss": "mean_squared_error_mean",
        "normalization": "none",
        "optimizer": expected_training["optimizer"],
        "recipe_id": "benchmark_reviewed_v1",
        "schedule_origin": (
            "preregistered benchmark-native reviewed adapter schedule; "
            "not an MTDLPy upstream default"
        ),
        "scheduler": None,
        "seed": 101,
    }
    method = {
        "implementation": {
            "dependency_closure_sha256": closure_sha256,
            "repository_commit": "2" * 40,
            "runner_source_sha256": runner_sha256,
        },
        "initialization": {"sha256": weights_sha256},
        "training": expected_training,
    }
    runtime = {
        "adapter_wall_time_s": 1.0,
        "bindings": {
            "adapter_source_sha256": adapter.sha256,
            "checkpoint_sha256": checkpoint.sha256,
            "dependency_closure_sha256": closure_sha256,
            "imagenet_weights_sha256": weights_sha256,
            "observations_sha256": observations.sha256,
            "prediction_sha256": prediction.sha256,
            "runner_source_sha256": runner_sha256,
            "source_clean_worktree": True,
            "source_commit": "2" * 40,
            "train_sha256": train_sha256,
            "training_seed": 101,
            "upstream_source_sha256": source.sha256,
            "validation_sha256": validation_sha256,
        },
        "checkpoint_contract": {
            "contains_observation_campaign": False,
            "contains_truth": False,
            "dataset_identities": {
                "train": artifact(train_sha256),
                "validation": artifact(validation_sha256),
            },
            "safe_load": "torch.load(weights_only=True)",
            "schema": "pimsr-mtdlpy-common-retrain-checkpoint",
            "schema_version": 1,
            "seed": 101,
        },
        "command": ["python", "run_mtdlpy_common.py"],
        "comparison_status": "unscored_prediction_artifact",
        "contains_truth": False,
        "dependency_closure": closure,
        "determinism": {
            "numpy_legacy_global_seed": 101,
            "python_random_seed": 101,
        },
        "finished_at_utc": "2026-08-24T00:00:01+00:00",
        "method": "MTDLPy/DinkNet50",
        "observation_contract": {
            "contains_truth": False,
            "observations_sha256": observations.sha256,
            "truth_keys_accepted": False,
        },
        "operation": "inference_from_reusable_checkpoint",
        "outputs": {
            "checkpoint": artifact(checkpoint.sha256),
            "predictions": artifact(prediction.sha256),
        },
        "prediction_contract": {
            "contains_truth": False,
            "observations_sha256": observations.sha256,
            "truth_keys_accepted": False,
        },
        "preprocessing": {"contract": "test"},
        "ranking_allowed": False,
        "repository": {"clean_worktree": True, "commit": "2" * 40},
        "runtime": {"inference_wall_time_s": 1.0},
        "schema": "pimsr-mtdlpy-common-retrain-runtime",
        "schema_version": 2,
        "seed": 101,
        "source_artifacts": source_artifacts,
        "started_at_utc": "2026-08-24T00:00:00+00:00",
        "track": "common-retrain",
        "training_config": actual_training,
        "training_summary": {"best_epoch": 10},
        "truth_keys_accepted": False,
        "working_directory": "D:/benchmark",
    }

    prediction_lock2d._runtime_bindings(
        runtime,
        method_id="mtdlpy",
        training_seed=101,
        method=method,
        observations=observations,
        prediction=prediction,
        checkpoint=checkpoint,
        source=source,
        adapter=adapter,
        train_sha256=train_sha256,
        validation_sha256=validation_sha256,
    )

    runtime["ground_truth_path"] = "withheld.npz"
    with pytest.raises(PredictionLock2DValidationError, match="keys mismatch"):
        prediction_lock2d._runtime_bindings(
            runtime,
            method_id="mtdlpy",
            training_seed=101,
            method=method,
            observations=observations,
            prediction=prediction,
            checkpoint=checkpoint,
            source=source,
            adapter=adapter,
            train_sha256=train_sha256,
            validation_sha256=validation_sha256,
        )
    runtime.pop("ground_truth_path")

    closure["required_local_python_source_artifacts_recorded"] = False
    with pytest.raises(PredictionLock2DValidationError, match="scope/completeness"):
        prediction_lock2d._runtime_bindings(
            runtime,
            method_id="mtdlpy",
            training_seed=101,
            method=method,
            observations=observations,
            prediction=prediction,
            checkpoint=checkpoint,
            source=source,
            adapter=adapter,
            train_sha256=train_sha256,
            validation_sha256=validation_sha256,
        )


def test_prediction_hardlink_alias_and_checkpoint_retrain_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    matrix = _matrix(tmp_path, monkeypatch)
    value = matrix[4]
    first, second = value["campaigns"][0]["runs"][:2]
    second["prediction"] = dict(first["prediction"])
    input_sha = _write_json(matrix[2], value)
    with pytest.raises(
        PredictionLock2DValidationError, match="identities must be unique"
    ):
        create_prediction_lock_2d(
            matrix[0],
            matrix[2],
            expected_preregistration_sha256=matrix[1],
            expected_input_manifest_sha256=input_sha,
        )

    matrix = _matrix(tmp_path / "retrain", monkeypatch)
    value = matrix[4]
    replacement = _write(
        tmp_path / "retrain" / "artifacts" / "replacement.checkpoint",
        b"campaign-specific-retrain",
    )
    value["campaigns"][1]["runs"][0]["checkpoint"] = replacement
    input_sha = _write_json(matrix[2], value)
    with pytest.raises(
        PredictionLock2DValidationError, match="reuse exactly one checkpoint"
    ):
        create_prediction_lock_2d(
            matrix[0],
            matrix[2],
            expected_preregistration_sha256=matrix[1],
            expected_input_manifest_sha256=input_sha,
        )


def test_alternate_mtdlpy_recipe_is_rejected():
    expected = {
        "batch_size": 4,
        "epochs": 10,
        "optimizer": {"learning_rate": 0.0001, "name": "Adam"},
        "recipe_id": "benchmark_reviewed_v1",
    }
    alternate = {
        "batch_size": 8,
        "epochs": 200,
        "optimizer": {"learning_rate": 1e-8, "name": "Adam"},
        "recipe_id": "upstream_paramconfig_b01f72a_v1",
    }
    with pytest.raises(PredictionLock2DValidationError, match="recipe_id"):
        prediction_lock2d._validate_training_recipe("mtdlpy", expected, alternate)


def test_pimsr_class_weights_are_exactly_bound_to_frozen_train_counts():
    counts = [1909, 2097, 2001, 2048, 1945]
    weights = (np.sum(counts) / (5.0 * np.asarray(counts, dtype=np.float64))).tolist()
    expected = {
        "batch_size": 64,
        "class_counts": counts,
        "class_weight_formula": "count_sum/(5*max(class_count,1))",
        "class_weights": weights,
        "epochs": 80,
        "gradient_clip_max_norm": 1.0,
        "loss": {
            "scenario_cross_entropy_weight": 0.1,
            "sigma_epochs_15_through_79": ("beta_nll_0.5_plus_log_sigma_l2_0.05"),
            "total_variation_weight": 0.05,
            "validation": "plain_nll_plus_tv_0.05_plus_scenario_ce_0.1",
            "warmup_epochs_0_through_14": (
                "half_mean_squared_error_plus_tv_0.05_plus_scenario_ce_0.1"
            ),
        },
        "optimizer": {
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "learning_rate": 0.0003,
            "name": "AdamW",
            "weight_decay": 0.0001,
        },
        "scheduler": {
            "eta_min": 0.0,
            "name": "CosineAnnealingLR",
            "step_timing": "after_each_epoch",
            "t_max": 80,
        },
        "workers": 2,
    }
    actual = {
        "batch_size": 64,
        "beta_nll": 0.5,
        "class_weights": weights,
        "epochs": 80,
        "gradient_clip_norm": 1.0,
        "learning_rate": 0.0003,
        "loss": "beta_nll+tv0.05+scenario_ce0.1/v1",
        "normalization": "per-channel-train-mean-std/v1",
        "optimizer": "torch.optim.AdamW",
        "runtime": {},
        "scheduler": "CosineAnnealingLR",
        "scheduler_t_max": 80,
        "seed": 101,
        "sigma_regularization": 0.05,
        "sigma_warmup": 15,
        "validation_loss": "plain_nll+tv0.05+scenario_ce0.1/v1",
        "weight_decay": 0.0001,
        "workers": 2,
    }
    prediction_lock2d._validate_pimsr_training(expected, actual, 101)

    actual["class_weights"] = [1.0] * 5
    with pytest.raises(PredictionLock2DValidationError, match="pinned train split"):
        prediction_lock2d._validate_pimsr_training(expected, actual, 101)


def test_locked_artifact_alias_is_rejected(tmp_path: Path):
    shared = tmp_path / "shared.bin"
    shared.write_bytes(b"shared")
    distinct = []
    for index in range(4):
        path = tmp_path / f"artifact-{index}.bin"
        path.write_bytes(f"artifact-{index}".encode())
        distinct.append(path)
    digest = _sha(shared)
    run = LockedRun2D(
        campaign_id="campaign-1",
        method_id="pimsr",
        training_seed=101,
        observations_sha256=digest,
        observation_manifest_sha256=digest,
        prediction_sha256=_sha(distinct[0]),
        prediction_size_bytes=distinct[0].stat().st_size,
        runtime_sha256=_sha(distinct[1]),
        runtime_size_bytes=distinct[1].stat().st_size,
        checkpoint_sha256=_sha(distinct[2]),
        checkpoint_size_bytes=distinct[2].stat().st_size,
        source_commit="1" * 40,
        source_sha256=_sha(distinct[3]),
        adapter_source_sha256="2" * 64,
    )
    with pytest.raises(PredictionLock2DValidationError, match="alias"):
        validate_locked_run_artifacts_2d(
            run,
            observations_path=shared,
            observation_manifest_path=shared,
            prediction_path=distinct[0],
            runtime_path=distinct[1],
            checkpoint_path=distinct[2],
            source_path=distinct[3],
        )


def test_snapshot_rejects_symlink_and_detects_hash_change(tmp_path: Path):
    target = tmp_path / "target.bin"
    target.write_bytes(b"safe")
    with pytest.raises(PredictionLock2DValidationError, match="differs from its pin"):
        snapshot_regular_file(target, expected_sha256="0" * 64, role="target")
    link = tmp_path / "link.bin"
    try:
        os.symlink(target, link)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(PredictionLock2DValidationError, match="regular non-link"):
        snapshot_regular_file(link, expected_sha256=None, role="link")


def test_lock_publication_returns_final_descriptor_receipt_and_never_overwrites(
    tmp_path: Path,
):
    destination = tmp_path / "lock.json"
    lock = {"schema": LOCK_SCHEMA, "value": 1}
    stale_fixed_partial = destination.with_name(destination.name + ".part")
    stale_fixed_partial.write_bytes(b"foreign old partial")

    receipt = publish_prediction_lock_2d(lock, destination)

    assert receipt == PredictionLock2DPublicationReceipt(
        destination.absolute(),
        hashlib.sha256(destination.read_bytes()).hexdigest(),
        destination.stat().st_size,
    )
    assert receipt.sha256 == hashlib.sha256(canonical_json_bytes(lock)).hexdigest()
    assert destination.stat().st_mode & 0o222 == 0
    assert stale_fixed_partial.read_bytes() == b"foreign old partial"
    with pytest.raises(PredictionLock2DPublicationError, match="overwrite"):
        publish_prediction_lock_2d(lock, destination)


def test_lock_publication_early_failure_leaves_exclusive_artifact_sealed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    destination = tmp_path / "interrupted.json"

    def interrupted(descriptor: int, payload: bytes) -> None:
        assert os.write(descriptor, payload[:7]) == 7
        raise KeyboardInterrupt

    monkeypatch.setattr(prediction_lock2d, "_write_all_descriptor", interrupted)
    with pytest.raises(KeyboardInterrupt):
        publish_prediction_lock_2d({"value": 1}, destination)

    assert destination.exists()
    assert destination.read_bytes() == canonical_json_bytes({"value": 1})[:7]
    assert destination.stat().st_mode & 0o222 == 0


def test_lock_receipt_reseals_mode_changed_before_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    destination = tmp_path / "resealed.json"
    real_receipt = prediction_lock2d._stable_prediction_lock_receipt

    def change_mode_then_reopen(*args, **kwargs):
        os.chmod(destination, 0o600)
        return real_receipt(*args, **kwargs)

    monkeypatch.setattr(
        prediction_lock2d, "_stable_prediction_lock_receipt", change_mode_then_reopen
    )
    receipt = publish_prediction_lock_2d({"value": 1}, destination)

    assert destination.stat().st_mode & 0o222 == 0
    assert receipt.sha256 == hashlib.sha256(destination.read_bytes()).hexdigest()
    assert receipt.size_bytes == destination.stat().st_size


def test_lock_publication_detects_same_inode_same_size_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    destination = tmp_path / "mutated.json"
    real_read = prediction_lock2d._read_all_descriptor
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

    monkeypatch.setattr(
        prediction_lock2d, "_read_all_descriptor", mutate_after_first_read
    )
    with pytest.raises(
        PredictionLock2DPublicationError, match="changed during|cannot verify"
    ):
        publish_prediction_lock_2d({"value": 1}, destination)

    assert destination.exists()
    assert destination.stat().st_mode & 0o222 == 0


def test_lock_receipt_detects_retained_writer_during_final_parent_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    destination = tmp_path / "retained-writer.json"
    replacement = canonical_json_bytes({"value": 2})
    real_seal = prediction_lock2d._seal_publication_descriptor
    real_parent_identity = prediction_lock2d._publication_parent_identity
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
        prediction_lock2d, "_seal_publication_descriptor", seal_with_retained_writer
    )
    monkeypatch.setattr(
        prediction_lock2d,
        "_publication_parent_identity",
        mutate_during_parent_check,
    )
    try:
        if os.name == "nt":
            receipt = publish_prediction_lock_2d({"value": 1}, destination)
            assert writer_denied
            assert receipt.sha256 == hashlib.sha256(destination.read_bytes()).hexdigest()
            assert receipt.size_bytes == destination.stat().st_size
        else:
            with pytest.raises(PredictionLock2DPublicationError, match="changed during"):
                publish_prediction_lock_2d({"value": 1}, destination)
    finally:
        if writer is not None:
            os.close(writer)

    if os.name == "nt":
        assert destination.read_bytes() == canonical_json_bytes({"value": 1})
    else:
        assert destination.read_bytes() == replacement


def test_lock_publication_never_deletes_a_replacement_on_verification_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    destination = tmp_path / "replaced.json"

    def replace_then_fail(*_args, **_kwargs):
        os.chmod(destination, 0o600)
        destination.unlink()
        destination.write_bytes(b"foreign replacement")
        raise PredictionLock2DPublicationError("injected replacement")

    monkeypatch.setattr(
        prediction_lock2d, "_stable_prediction_lock_receipt", replace_then_fail
    )
    with pytest.raises(PredictionLock2DPublicationError, match="injected replacement"):
        publish_prediction_lock_2d({"value": 1}, destination)

    assert destination.read_bytes() == b"foreign replacement"


def test_lock_publication_rejects_hardlink_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    destination = tmp_path / "aliased.json"
    alias = tmp_path / "alias.json"
    real_seal = prediction_lock2d._seal_publication_descriptor
    calls = 0

    def seal_then_alias(descriptor: int) -> None:
        nonlocal calls
        real_seal(descriptor)
        calls += 1
        if calls == 1:
            os.link(destination, alias)

    monkeypatch.setattr(
        prediction_lock2d, "_seal_publication_descriptor", seal_then_alias
    )
    with pytest.raises(PredictionLock2DPublicationError, match="changed before"):
        publish_prediction_lock_2d({"value": 1}, destination)

    assert destination.samefile(alias)


def test_lock_publication_detects_parent_replacement_before_final_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    parent = tmp_path / "publication"
    destination = parent / "lock.json"
    displaced = tmp_path / "publication-displaced"
    real_receipt = prediction_lock2d._stable_prediction_lock_receipt

    def replace_parent(*args, **kwargs):
        parent.rename(displaced)
        parent.mkdir()
        return real_receipt(*args, **kwargs)

    monkeypatch.setattr(
        prediction_lock2d, "_stable_prediction_lock_receipt", replace_parent
    )
    with pytest.raises(PredictionLock2DPublicationError, match="cannot verify"):
        publish_prediction_lock_2d({"value": 1}, destination)

    assert (displaced / destination.name).exists()
    assert not destination.exists()


def test_lock_publication_rejects_a_symlinked_parent(tmp_path: Path):
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    try:
        linked_parent.symlink_to(real_parent, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation is unavailable")

    with pytest.raises(PredictionLock2DPublicationError, match="must (not|be a real)"):
        publish_prediction_lock_2d({"value": 1}, linked_parent / "lock.json")

    assert not (real_parent / "lock.json").exists()


def test_lock_publication_does_not_create_through_a_linked_ancestor(tmp_path: Path):
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_ancestor = tmp_path / "linked-ancestor"
    try:
        linked_ancestor.symlink_to(real_parent, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation is unavailable")

    with pytest.raises(PredictionLock2DPublicationError, match="ancestor.*(real|link)"):
        publish_prediction_lock_2d(
            {"value": 1}, linked_ancestor / "missing" / "lock.json"
        )

    assert not (real_parent / "missing").exists()


def test_cli_prints_the_verified_payload_digest_without_reopening_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    lock = {"schema": LOCK_SCHEMA, "value": 1}
    destination = tmp_path / "lock.json"
    monkeypatch.setattr(
        prediction_lock2d, "create_prediction_lock_2d", lambda *_args, **_kwargs: lock
    )
    monkeypatch.setattr(
        prediction_lock2d,
        "publish_prediction_lock_2d",
        lambda *_args, **_kwargs: PredictionLock2DPublicationReceipt(
            destination, "f" * 64, 123
        ),
    )

    def reject_reopen(_path: Path) -> bytes:
        raise AssertionError("published capability must not reopen a pathname")

    monkeypatch.setattr(Path, "read_bytes", reject_reopen)
    assert (
        prediction_lock2d.main(
            [
                "--preregistration",
                str(tmp_path / "prereg.json"),
                "--preregistration-sha256",
                "a" * 64,
                "--input-manifest",
                str(tmp_path / "input.json"),
                "--input-manifest-sha256",
                "b" * 64,
                "--output",
                str(destination),
            ]
        )
        == 0
    )
    assert capsys.readouterr().out == (
        f"published {destination} sha256={'f' * 64} size=123\n"
    )


def test_strict_lock_json_rejects_duplicate_keys(tmp_path: Path):
    path = tmp_path / "duplicate.json"
    path.write_bytes(b'{"schema":"a","schema":"b"}\n')
    snapshot = snapshot_regular_file(path, expected_sha256=None, role="duplicate")
    with pytest.raises(PredictionLock2DValidationError, match="duplicate JSON"):
        prediction_lock2d._strict_json(snapshot, "duplicate")


@pytest.mark.parametrize(
    "record",
    [
        {"operator_manifest_sha256": "a" * 64},
        {"operatorManifestHash": "a" * 64},
        {"ground_truth_path": "withheld.npz"},
        {"generator_random_seed": 123},
        {"hidden_generator_seed": 123},
        {"prediction_contract": {"contains_truth": True}},
        {"observation_contract": {"truth_keys_accepted": True}},
    ],
)
def test_prescore_secret_variants_and_true_truth_declarations_are_rejected(record):
    with pytest.raises(PredictionLock2DValidationError):
        prediction_lock2d._reject_prescore_secrets(record, "runtime")

    prediction_lock2d._reject_prescore_secrets(
        {
            "observation_contract": {
                "evaluation_floor_role": "scorer_only_not_model_input",
                "truth_keys_accepted": False,
            },
            "prediction_contract": {"contains_truth": False},
        },
        "runtime",
    )


def test_adapter_source_accepts_only_unchanged_clean_descendant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repository = tmp_path.resolve()
    source_path = repository / "adapters" / "pimsr.py"
    _write(source_path, b"pinned adapter\n")
    snapshot = snapshot_regular_file(
        source_path, expected_sha256=_sha(source_path), role="adapter"
    )
    pinned = "a" * 40
    descendant = "b" * 40
    blob = "c" * 40
    changed = False
    blob_matches = True

    def git_output(repo: Path, *arguments: str) -> str:
        assert repo == repository
        if arguments == ("rev-parse", "--show-toplevel"):
            return str(repository)
        if arguments == ("rev-parse", "HEAD"):
            return descendant
        if arguments == (
            "merge-base",
            "--is-ancestor",
            pinned,
            descendant,
        ):
            return ""
        if arguments == ("status", "--porcelain=v1", "--untracked-files=normal"):
            return ""
        if arguments == (
            "ls-files",
            "--error-unmatch",
            "--",
            "adapters/pimsr.py",
        ):
            return "adapters/pimsr.py"
        if arguments == (
            "ls-tree",
            pinned,
            "--",
            "adapters/pimsr.py",
        ):
            return f"100644 blob {blob}\tadapters/pimsr.py"
        if arguments == (
            "diff",
            "--name-only",
            pinned,
            descendant,
            "--",
            "adapters/pimsr.py",
            "src",
            "scripts",
            "pyproject.toml",
        ):
            return "adapters/pimsr.py" if changed else ""
        raise AssertionError(arguments)

    def git_bytes(
        repo: Path, *arguments: str, input_payload: bytes | None = None
    ) -> bytes:
        assert repo == repository
        assert arguments == (
            "hash-object",
            "--stdin",
            "--path",
            "adapters/pimsr.py",
        )
        assert input_payload == b"pinned adapter\n"
        return (blob if blob_matches else "d" * 40).encode("ascii") + b"\n"

    monkeypatch.setattr(prediction_lock2d, "_git_output", git_output)
    monkeypatch.setattr(prediction_lock2d, "_git_bytes", git_bytes)
    prediction_lock2d._validate_source_repository(
        snapshot,
        repository,
        expected_commit=pinned,
        method_id="pimsr adapter",
        allow_descendant_head=True,
        protected_paths=("src", "scripts", "pyproject.toml"),
    )

    changed = True
    with pytest.raises(PredictionLock2DValidationError, match="changed after"):
        prediction_lock2d._validate_source_repository(
            snapshot,
            repository,
            expected_commit=pinned,
            method_id="pimsr adapter",
            allow_descendant_head=True,
            protected_paths=("src", "scripts", "pyproject.toml"),
        )

    changed = False
    blob_matches = False
    with pytest.raises(PredictionLock2DValidationError, match="pinned commit blob"):
        prediction_lock2d._validate_source_repository(
            snapshot,
            repository,
            expected_commit=pinned,
            method_id="pimsr adapter",
            allow_descendant_head=True,
            protected_paths=("src", "scripts", "pyproject.toml"),
        )


def test_lock_validator_rejects_nested_paths_and_duplicate_campaign_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    matrix = _matrix(tmp_path / "matrix", monkeypatch)
    lock = _create(matrix)
    lock["input_manifest"]["path"] = "local-lock-input.json"
    lock_path = tmp_path / "lock-with-path.json"
    _write_json(lock_path, lock)
    with pytest.raises(PredictionLock2DValidationError, match="input_manifest keys"):
        validate_prediction_lock_2d(
            matrix[0],
            lock_path,
            expected_preregistration_sha256=matrix[1],
            expected_lock_sha256=_sha(lock_path),
        )

    lock = _create(matrix)
    first = lock["campaigns"][0]
    second = lock["campaigns"][1]
    second["observations_sha256"] = first["observations_sha256"]
    second["observation_manifest_sha256"] = first["observation_manifest_sha256"]
    for run in lock["runs"]:
        if run["campaign_id"] == second["campaign_id"]:
            run["observations_sha256"] = first["observations_sha256"]
            run["observation_manifest_sha256"] = first["observation_manifest_sha256"]
    lock_path = tmp_path / "duplicate-campaign-evidence.json"
    _write_json(lock_path, lock)
    with pytest.raises(PredictionLock2DValidationError, match="distinct observation"):
        validate_prediction_lock_2d(
            matrix[0],
            lock_path,
            expected_preregistration_sha256=matrix[1],
            expected_lock_sha256=_sha(lock_path),
        )


def test_observation_lock_validation_checks_all_public_response_arrays(tmp_path: Path):
    response_shape = (500, 2, 2)
    arrays = {
        "schema": np.asarray("pimsr-sota-2d-observations"),
        "schema_version": np.asarray(1, dtype="<i8"),
        "sample_index": np.arange(500, dtype="<i8"),
        "frequency_hz": np.asarray([1.0, 2.0], dtype="<f8"),
        "station_x_m": np.asarray([-0.5, 0.5], dtype="<f8"),
        "x_cell_centers_m": np.asarray([-1.0, 1.0], dtype="<f8"),
        "depth_cell_centers_m": np.asarray([1.0, 2.0], dtype="<f8"),
        "observation_channel_order": np.asarray(
            [
                "log10_rho_te",
                "phase_te_degrees",
                "log10_rho_tm",
                "phase_tm_degrees",
            ]
        ),
        "observed_log10_rho_te": np.zeros(response_shape, dtype="<f4"),
        "observed_phase_te_degrees": np.zeros(response_shape, dtype="<f4"),
        "observed_log10_rho_tm": np.zeros(response_shape, dtype="<f4"),
        "observed_phase_tm_degrees": np.zeros(response_shape, dtype="<f4"),
        "declared_evaluation_floor_log10_rho_te": np.ones(response_shape, dtype="<f4"),
        "declared_evaluation_floor_phase_te_degrees": np.ones(
            response_shape, dtype="<f4"
        ),
        "declared_evaluation_floor_log10_rho_tm": np.ones(response_shape, dtype="<f4"),
        "declared_evaluation_floor_phase_tm_degrees": np.ones(
            response_shape, dtype="<f4"
        ),
        "valid_mask": np.ones((500, 4, 2, 2), dtype=np.bool_),
    }
    path = tmp_path / "observations.npz"
    np.savez(path, **arrays)
    snapshot = snapshot_regular_file(
        path, expected_sha256=_sha(path), role="observations"
    )
    family_partition = {
        "schema": prediction_lock2d.FAMILY_PARTITION_SCHEMA,
        "schema_version": prediction_lock2d.FAMILY_PARTITION_SCHEMA_VERSION,
        "families": list(prediction_lock2d.GEOLOGICAL_FAMILIES),
        "bases_per_family": 20,
        "noise_realizations_per_base": 5,
        "commitment_contract": dict(prediction_lock2d.FAMILY_COMMITMENT_CONTRACT),
    }
    manifest = {
        "audience": "method_input_public",
        "declared_evaluation_floors": {},
        "family_partition_commitment": {
            "schema": prediction_lock2d.FAMILY_PARTITION_SCHEMA,
            "schema_version": prediction_lock2d.FAMILY_PARTITION_SCHEMA_VERSION,
            "sha256": "c" * 64,
            "contract": dict(prediction_lock2d.FAMILY_COMMITMENT_CONTRACT),
        },
        "observation_payload": {
            "arrays": {name: {} for name in prediction_lock2d._OBSERVATION_MEMBER_ORDER},
            "media_type": "application/x-npz",
            "schema": "pimsr-sota-2d-observations",
            "schema_version": 1,
            "sha256": snapshot.sha256,
            "size_bytes": snapshot.size_bytes,
        },
        "physical_contract": {},
        "sample_count": 500,
        "schema": "pimsr-sota-2d-observation-manifest",
        "schema_version": 3,
        "split_id": "campaign-1",
    }
    sample_count, sample_ids, x_axis, depth_axis = (
        prediction_lock2d._observation_identity(
            snapshot, manifest, "campaign-1", family_partition
        )
    )
    assert sample_count == len(sample_ids) == 500
    np.testing.assert_array_equal(x_axis, [-1.0, 1.0])
    np.testing.assert_array_equal(depth_axis, [1.0, 2.0])

    manifest["family_partition_commitment"]["contract"]["domain_separator"] = (
        "posthoc-relabel/v1"
    )
    with pytest.raises(PredictionLock2DValidationError, match="commitment differs"):
        prediction_lock2d._observation_identity(
            snapshot, manifest, "campaign-1", family_partition
        )
    manifest["family_partition_commitment"]["contract"] = dict(
        prediction_lock2d.FAMILY_COMMITMENT_CONTRACT
    )

    arrays["observed_phase_tm_degrees"][0, 0, 0] = 180.0
    invalid_path = tmp_path / "invalid-observations.npz"
    np.savez(invalid_path, **arrays)
    invalid = snapshot_regular_file(
        invalid_path, expected_sha256=_sha(invalid_path), role="observations"
    )
    manifest["observation_payload"]["sha256"] = invalid.sha256
    manifest["observation_payload"]["size_bytes"] = invalid.size_bytes
    with pytest.raises(PredictionLock2DValidationError, match="phase convention"):
        prediction_lock2d._observation_identity(
            invalid, manifest, "campaign-1", family_partition
        )
