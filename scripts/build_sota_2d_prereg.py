from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from pimsr_benchmarks import comparison2d, densenet2d, mtdlpy
from pimsr_benchmarks.modem2d_forward import (
    canonical_depth_centres_m,
    canonical_frequencies_hz,
)

ROOT = Path(__file__).resolve().parents[1]
FORWARD_COMMIT = "dc36edac75dbd51cc92679a35f38d42d0e276299"
TRAIN_SHA256 = "b9f1fce44012abe522e1b238ab67ccf9a3c7d9f81890c9911c9e25279b590051"
VALIDATION_SHA256 = "19ee9df2c4e0d57494e424948f0088064590c3a5e517b0369b1e18c6a26c905d"
SEED_COMMITMENT_SHA256 = (
    "b4ff00c55a1798e01fc85a980edf0642be27d31938a5112f3f2b63f8d5cdba0b"
)
SAMPLE_KEY_COMMITMENT_SHA256 = (
    "67a65552f07d792cf079f9fc57b4b523812f174bcc1ffe52abf38a33bb050e09"
)
COMPARISON_DEPENDENCIES = (
    "src/pimsr_benchmarks/_publication_io.py",
    "src/pimsr_benchmarks/dataset_lineage2d.py",
    "src/pimsr_benchmarks/evaluation2d.py",
    "src/pimsr_benchmarks/hidden_campaign2d.py",
    "src/pimsr_benchmarks/modem2d_forward.py",
    "src/pimsr_benchmarks/prediction_lock2d.py",
)


def _run_git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    return result.stdout.strip()


def _repository_identity(repository: Path, *, origin: str | None = None) -> str:
    root = Path(_run_git(repository, "rev-parse", "--show-toplevel")).resolve(strict=True)
    if not os.path.samefile(root, repository.resolve(strict=True)):
        raise RuntimeError(f"repository path is not its Git root: {repository}")
    if _run_git(repository, "status", "--porcelain=v1", "--untracked-files=no"):
        raise RuntimeError(f"repository has tracked changes: {repository}")
    if origin is not None:
        remotes = _run_git(repository, "remote", "get-url", "--all", "origin")
        if remotes.splitlines() != [origin]:
            raise RuntimeError(f"repository origin differs from {origin!r}: {repository}")
    return _run_git(repository, "rev-parse", "--verify", "HEAD^{commit}")


def _snapshot(path: Path) -> dict[str, Any]:
    absolute = path.resolve(strict=True)
    before = absolute.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or absolute.is_symlink():
        raise RuntimeError(
            f"artifact must be one unique regular non-link file: {absolute}"
        )
    payload = absolute.read_bytes()
    after = absolute.stat(follow_symlinks=False)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError(f"artifact changed while hashing: {absolute}")
    return {
        "path": str(absolute),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _relative_snapshot(path: str) -> dict[str, Any]:
    artifact = _snapshot(ROOT / path)
    return {
        "path": path,
        "sha256": artifact["sha256"],
        "size_bytes": artifact["size_bytes"],
    }


def _load_json(artifact: dict[str, Any], *, role: str) -> dict[str, Any]:
    value = json.loads(Path(artifact["path"]).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{role} must contain a JSON object")
    return value


def _analytic_contract() -> dict[str, Any]:
    frequencies = canonical_frequencies_hz()
    depth = canonical_depth_centres_m()
    return {
        "schema": "pimsr-modem2d-analytic-1d-contract",
        "schema_version": 1,
        "time_convention": "exp(+i omega t)",
        "response_shape": [8, 12],
        "frequencies_hz": frequencies.tolist(),
        "frequencies_sha256": hashlib.sha256(frequencies.tobytes()).hexdigest(),
        "canonical_depth_centres_m": depth.tolist(),
        "canonical_depth_centres_sha256": hashlib.sha256(depth.tobytes()).hexdigest(),
        "cases": [
            {
                "truth_id": "analytic-halfspace-100",
                "depth_profile": [
                    {
                        "maximum_depth_m_exclusive": None,
                        "resistivity_ohm_m": 100.0,
                    }
                ],
            },
            {
                "truth_id": "analytic-layered-100-10-500",
                "depth_profile": [
                    {
                        "maximum_depth_m_exclusive": 1_000.0,
                        "resistivity_ohm_m": 100.0,
                    },
                    {
                        "maximum_depth_m_exclusive": 3_000.0,
                        "resistivity_ohm_m": 10.0,
                    },
                    {
                        "maximum_depth_m_exclusive": None,
                        "resistivity_ohm_m": 500.0,
                    },
                ],
            },
        ],
    }


def _dataset(
    *, split: str, filename: str, sha256: str, size_bytes: int, rows: int, seed: int
) -> dict[str, Any]:
    role = "training" if split == "train" else "validation_and_checkpoint_selection"
    return {
        "artifact": {
            "filename": filename,
            "sha256": sha256,
            "size_bytes": size_bytes,
        },
        "dataset_id": f"pimsr_generated_2d_v1_{split}",
        "generator": {
            "entrypoint": "pimsr-forward-dataset2d",
            "repository_commit": FORWARD_COMMIT,
            "repository_url": "https://github.com/TheLitis/pimsr-forward",
            "seed": seed,
            "start_index": 0,
        },
        "identity_contract": "generator_seed_and_sample_index/v1",
        "role": role,
        "rows": rows,
        "sample_index_range_inclusive": [0, rows - 1],
    }


def _pimsr_training() -> dict[str, Any]:
    counts = np.asarray([1909, 2097, 2001, 2048, 1945], dtype=np.float64)
    return {
        "batch_size": 64,
        "class_counts": counts.astype(np.int64).tolist(),
        "class_weight_formula": "count_sum/(5*max(class_count,1))",
        "class_weights": (counts.sum() / (5.0 * counts)).tolist(),
        "epochs": 80,
        "gradient_clip_max_norm": 1.0,
        "loss": {
            "scenario_cross_entropy_weight": 0.1,
            "sigma_epochs_15_through_79": "beta_nll_0.5_plus_log_sigma_l2_0.05",
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


def _mtdlpy_training() -> dict[str, Any]:
    return {
        "batch_size": 4,
        "checkpoint_selection": "lowest validation MSE; strict less-than; first tie",
        "early_stopping": "none_run_all_10_epochs",
        "epochs": 10,
        "gradient_clip_max_norm": 0.1,
        "loss": "mean_squared_error_mean",
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
        "scheduler": "none",
    }


def _methods(args: argparse.Namespace, repository_commit: str) -> list[dict[str, Any]]:
    pimsr_repository = args.pimsr_inversion_repository.resolve(strict=True)
    pimsr_commit = _repository_identity(
        pimsr_repository,
        origin="https://github.com/TheLitis/pimsr-inversion.git",
    )
    network = _snapshot(pimsr_repository / "src/pimsr_inversion/network2d.py")

    mtdlpy_repository = mtdlpy.verify_pinned_repository(args.mtdlpy_repository)
    imagenet = mtdlpy.validate_local_imagenet_weights(
        args.imagenet_weights, mtdlpy.IMAGENET_RESNET50_V1_SHA256
    )
    mtdlpy_runner = ROOT / "scripts/run_mtdlpy_common.py"
    mtdlpy_closure = mtdlpy._dependency_closure(
        repository=mtdlpy_repository,
        weights=imagenet,
        runner_source=mtdlpy_runner,
    )
    mtdlpy_adapter = _relative_snapshot("src/pimsr_benchmarks/mtdlpy.py")
    mtdlpy_runner_artifact = _relative_snapshot("scripts/run_mtdlpy_common.py")

    densenet_repository = densenet2d.verify_pinned_repository(args.densenet_repository)
    densenet_runner = ROOT / "scripts/run_densenet2d_common.py"
    densenet_sources = densenet2d._source_dependency_artifacts(
        densenet_repository, runner_source=densenet_runner
    )
    densenet_closure = densenet2d._dependency_closure(densenet_sources)
    densenet_adapter = _relative_snapshot("src/pimsr_benchmarks/densenet2d.py")
    densenet_runner_artifact = _relative_snapshot("scripts/run_densenet2d_common.py")
    shared_loader = _relative_snapshot("src/pimsr_benchmarks/mtdlpy.py")

    pimsr_adapter = _relative_snapshot("src/pimsr_benchmarks/pimsr2d_adapter.py")
    dense_training = densenet2d._training_config(101)
    dense_training.pop("seed")
    return [
        {
            "id": "pimsr",
            "role": "candidate",
            "architecture": {
                "class": "pimsr_inversion.network2d.PimsrNet2D",
                "input_channels": 4,
                "scenario_classes": 5,
                "scenario_head": "multiscale",
                "width": 48,
            },
            "data_contract": {
                "hidden_input": "public_observation_npz_only",
                "input_channel_order": list(mtdlpy.OBSERVATION_CHANNEL_ORDER),
                "input_geometry": "native_8_frequency_by_12_station_grid",
                "interpolation": "none",
                "missing_data_policy": "reject_if_valid_mask_is_not_all_true",
                "normalization": (
                    "phase_channels_divided_by_45_then_per_channel_training_mean_std_v1"
                ),
                "output_geometry": "native_64_depth_by_48_x_grid",
            },
            "implementation": {
                "adapter_repository_commit": repository_commit,
                "adapter_source_path": pimsr_adapter["path"],
                "adapter_source_sha256": pimsr_adapter["sha256"],
                "network_source_path": "src/pimsr_inversion/network2d.py",
                "network_source_sha256": network["sha256"],
                "repository_commit": pimsr_commit,
                "repository_url": "https://github.com/TheLitis/pimsr-inversion",
            },
            "training": _pimsr_training(),
        },
        {
            "id": "mtdlpy",
            "role": "reference",
            "architecture": {
                "class": "func.dinknet.DinkNet50",
                "input_channels": 4,
                "network_grid_shape": list(mtdlpy.NETWORK_GRID_SHAPE),
                "num_classes": 1,
            },
            "data_contract": mtdlpy._preprocessing_contract(),
            "implementation": {
                "adapter_repository_commit": repository_commit,
                "adapter_source_path": mtdlpy_adapter["path"],
                "adapter_source_sha256": mtdlpy_adapter["sha256"],
                "dependency_closure_sha256": mtdlpy._canonical_object_sha256(
                    mtdlpy_closure
                ),
                "dinknet_git_blob_sha1": mtdlpy.MTDLPY_DINKNET_GIT_BLOB,
                "dinknet_source_path": mtdlpy.MTDLPY_DINKNET_PATH,
                "dinknet_source_sha256": mtdlpy.MTDLPY_DINKNET_SHA256,
                "repository_commit": mtdlpy.MTDLPY_COMMIT,
                "repository_url": mtdlpy.MTDLPY_REPOSITORY_URL,
                "runner_source_sha256": mtdlpy_runner_artifact["sha256"],
            },
            "initialization": {
                "artifact": "torchvision_ResNet50_IMAGENET1K_V1",
                "sha256": imagenet["sha256"],
                "size_bytes": imagenet["size_bytes"],
                "url": mtdlpy.IMAGENET_RESNET50_V1_URL,
            },
            "training": _mtdlpy_training(),
        },
        {
            "id": "mt2dinv_densenet",
            "role": "reference",
            "architecture": densenet2d._model_contract(),
            "data_contract": densenet2d._preprocessing_contract(),
            "implementation": {
                "adapter_repository_commit": repository_commit,
                "adapter_source_path": densenet_adapter["path"],
                "adapter_source_sha256": densenet_adapter["sha256"],
                "architecture_git_blob_sha1": (
                    densenet2d.MT2DINV_DENSENET_SOURCE_GIT_BLOB
                ),
                "architecture_source_path": densenet2d.MT2DINV_DENSENET_SOURCE_PATH,
                "architecture_source_sha256": (densenet2d.MT2DINV_DENSENET_SOURCE_SHA256),
                "dependency_closure_sha256": densenet_closure["closure_sha256"],
                "repository_commit": densenet2d.MT2DINV_DENSENET_COMMIT,
                "repository_url": densenet2d.MT2DINV_DENSENET_REPOSITORY_URL,
                "runner_source_sha256": densenet_runner_artifact["sha256"],
                "shared_contract_loader_source_sha256": shared_loader["sha256"],
            },
            "training": dense_training,
        },
    ]


def build(args: argparse.Namespace) -> dict[str, Any]:
    repository_commit = _repository_identity(ROOT)
    report_artifact = _snapshot(args.convergence_report)
    report = _load_json(report_artifact, role="convergence report")
    if (
        report.get("schema") != "pimsr-modem2d-convergence-validation"
        or report.get("schema_version") != 1
        or report.get("passed") is not True
    ):
        raise RuntimeError("public convergence report has not passed")
    residuals = _snapshot(args.convergence_residuals)
    generator_source = _snapshot(args.modem_source)
    generator_binary = _snapshot(args.modem_binary)
    production_mesh = _snapshot(args.production_mesh)
    reference_mesh = _snapshot(args.reference_mesh)
    runtime_manifest = _snapshot(args.generation_runtime_manifest)
    converter = _relative_snapshot("src/pimsr_benchmarks/modem2d_forward.py")
    evaluation = _relative_snapshot("src/pimsr_benchmarks/evaluation2d.py")
    registry = _relative_snapshot("config/sota_methods.json")
    train_lineage = _relative_snapshot("evidence/public-lineage/train-lineage-v2.json")
    validation_lineage = _relative_snapshot(
        "evidence/public-lineage/validation-lineage-v2.json"
    )
    source = _relative_snapshot("src/pimsr_benchmarks/comparison2d.py")
    dependencies = [_relative_snapshot(path) for path in COMPARISON_DEPENDENCIES]
    methods = _methods(args, repository_commit)
    training_commits = {
        method["id"]: method["implementation"]["repository_commit"] for method in methods
    }
    return {
        "schema": "pimsr-sota-2d-common-retrain-preregistration",
        "schema_version": 1,
        "preregistration_id": "pimsr_2d_common_retrain_three_method_v2",
        "locked_date": args.locked_date,
        "status": "locked_before_hidden_materialization_and_method_execution",
        "claim_eligibility": {
            "benchmark_label": "in_distribution_generator_benchmark",
            "headline_sota_claim": (
                "allowed_only_if_all_prelocked_evidence_and_dominance_gates_pass"
            ),
            "hidden_distribution": (
                "same_geological_generator_family_with_independent_modem_observations"
            ),
            "model_space_use": "paired_in_distribution_method_comparison_only",
        },
        "protocol": {
            "document": "docs/SOTA_PROTOCOL.md",
            "registry_path": registry["path"],
            "registry_schema_version": 2,
            "registry_sha256": registry["sha256"],
            "version": "1.1",
        },
        "datasets": {
            "train": _dataset(
                split="train",
                filename="pimsr-generated-2d-v1-train.h5",
                sha256=TRAIN_SHA256,
                size_bytes=153_745_920,
                rows=10_000,
                seed=20_260_820,
            ),
            "validation": _dataset(
                split="validation",
                filename="pimsr-generated-2d-v1-val.h5",
                sha256=VALIDATION_SHA256,
                size_bytes=15_388_920,
                rows=1_000,
                seed=20_260_821,
            ),
            "hidden_test": {
                "campaigns": {
                    "campaign_ids": [
                        f"sota2d-hidden-campaign-{index:02d}" for index in range(1, 6)
                    ],
                    "count": 5,
                    "samples_per_campaign": 500,
                    "total_samples": 2_500,
                },
                "dataset_id": "pimsr_generated_2d_v1",
                "generator": {
                    "entrypoint": "pimsr-forward-dataset2d",
                    "repository_commit": FORWARD_COMMIT,
                    "repository_url": "https://github.com/TheLitis/pimsr-forward",
                    "source_schema": "pimsr-mt-2d",
                    "source_schema_version": 2,
                    "start_index": 0,
                },
                "grouping_contract": (
                    "campaign/geological_family/base_model/noise_realization/v1"
                ),
                "prediction_lock_gate": {
                    "artifact_granularity": (
                        "one_prediction_npz_per_method_training_seed_campaign"
                    ),
                    "gate_condition": (
                        "all_75_prediction_files_locked_before_any_truth_access"
                    ),
                    "locked_artifact_count": 75,
                    "missing_artifact_policy": "fail_closed_without_scoring_or_retraining",
                    "post_lock_action": (
                        "activate_operator_only_scoring_with_exact_locked_predictions"
                    ),
                },
                "sample_id_contract": {
                    "key_commitment_sha256": SAMPLE_KEY_COMMITMENT_SHA256,
                    "policy": "hmac_sha256_opaque_nonnegative_int64_v1",
                },
                "seed_commitment": {
                    "encoding": "utf8-canonical-json-int64-array-no-newline-v1",
                    "sha256": SEED_COMMITMENT_SHA256,
                },
            },
        },
        "public_dataset_lineage": {
            "train": train_lineage,
            "validation": validation_lineage,
        },
        "family_partition": {
            "schema": "pimsr-sota-2d-family-partition-commitment",
            "schema_version": 1,
            "families": list(comparison2d.FAMILY_IDS),
            "bases_per_family": 20,
            "noise_realizations_per_base": 5,
            "commitment_contract": {
                "algorithm": "SHA-256",
                "canonicalization": ("utf8-canonical-json-sort-keys-compact-newline-v1"),
                "domain_separator": "pimsr-sota-2d-family-partition/v1",
                "nonce_encoding": "lowercase_hex_32_bytes",
            },
        },
        "run_seeds": list(comparison2d.TRAINING_SEEDS),
        "methods": methods,
        "statistical_analysis": {
            "dominance_gate": comparison2d.EXPECTED_DOMINANCE_GATE,
            "effect": {
                "candidate": "pimsr",
                "definition": "candidate_minus_reference",
                "favorable_direction": "negative",
                "pairing_keys": [
                    "training_seed",
                    "campaign_id",
                    "geological_family",
                    "base_model_id",
                    "noise_realization_id",
                ],
                "references": list(comparison2d.REFERENCE_METHOD_IDS),
            },
            "hierarchical_paired_bootstrap": {
                "confidence": 0.95,
                "n_resamples": 10_000,
                "point_aggregation": comparison2d.EXPECTED_POINT_AGGREGATION,
                "resampling_levels": list(comparison2d.EXPECTED_RESAMPLING_LEVELS),
                "rng_seed": 20_260_824,
            },
            "multiplicity_policy": comparison2d.EXPECTED_MULTIPLICITY_POLICY,
            "primary_metric": {
                "name": "area_weighted_per_sample_rmse_log10_resistivity",
                "optimization_direction": "lower_is_better",
                "unit": "log10_ohm_m",
            },
        },
        "evaluation_contract": {
            "schema": "pimsr-sota-2d-evaluation",
            "schema_version": 3,
            "repository_commit": repository_commit,
            "source_sha256": evaluation["sha256"],
        },
        "comparison_contract": {
            "schema": "pimsr-sota-2d-comparison-implementation",
            "schema_version": 1,
            "repository_commit": repository_commit,
            "source": source,
            "dependencies": dependencies,
            "protected_paths": [source["path"], *COMPARISON_DEPENDENCIES],
        },
        "headline_evidence": {
            "hidden_observation_generator": {
                "name": "ModEM",
                "repository_url": "https://github.com/magnetotellurics/ModEM",
                "repository_commit": args.modem_commit,
                "source_sha256": generator_source["sha256"],
                "source_size_bytes": generator_source["size_bytes"],
                "binary_sha256": generator_binary["sha256"],
                "binary_size_bytes": generator_binary["size_bytes"],
                "container_image_digest": args.container_image_digest,
                "mesh_artifact_sha256": production_mesh["sha256"],
                "mesh_artifact_size_bytes": production_mesh["size_bytes"],
                "converter_sha256": converter["sha256"],
                "converter_size_bytes": converter["size_bytes"],
                "converter_repository_commit": repository_commit,
                "generation_runtime": dict(comparison2d._HIDDEN_GENERATION_RUNTIME),
                "generation_runtime_manifest_sha256": runtime_manifest["sha256"],
                "generation_runtime_manifest_size_bytes": runtime_manifest["size_bytes"],
            },
            "public_mesh_convergence": {
                "criterion_id": "modem_public_mesh_convergence_v1",
                "report_sha256": report_artifact["sha256"],
                "report_size_bytes": report_artifact["size_bytes"],
                "residuals_sha256": residuals["sha256"],
                "residuals_size_bytes": residuals["size_bytes"],
                "refined_mesh_sha256": reference_mesh["sha256"],
                "refined_mesh_size_bytes": reference_mesh["size_bytes"],
                "thresholds": dict(comparison2d._CONVERGENCE_THRESHOLDS),
                "analytic_1d_contract": _analytic_contract(),
            },
            "training_solver_commits": training_commits,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the immutable three-method 2-D preregistration"
    )
    parser.add_argument("--convergence-report", type=Path, required=True)
    parser.add_argument("--convergence-residuals", type=Path, required=True)
    parser.add_argument("--modem-source", type=Path, required=True)
    parser.add_argument("--modem-binary", type=Path, required=True)
    parser.add_argument("--production-mesh", type=Path, required=True)
    parser.add_argument("--reference-mesh", type=Path, required=True)
    parser.add_argument("--generation-runtime-manifest", type=Path, required=True)
    parser.add_argument("--pimsr-inversion-repository", type=Path, required=True)
    parser.add_argument("--mtdlpy-repository", type=Path, required=True)
    parser.add_argument("--densenet-repository", type=Path, required=True)
    parser.add_argument("--imagenet-weights", type=Path, required=True)
    parser.add_argument("--container-image-digest", required=True)
    parser.add_argument(
        "--modem-commit",
        default="55a4aa62f7e8366fbf78a23ee8a19c1d4561d0c3",
    )
    parser.add_argument("--locked-date", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite preregistration: {output}")
    value = build(args)
    payload = (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    print(
        json.dumps(
            {
                "path": str(output),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
