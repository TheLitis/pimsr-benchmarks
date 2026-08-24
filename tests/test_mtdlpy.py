"""Fail-closed tests for the pinned MTDLPy common-retraining adapter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from pimsr_benchmarks import mtdlpy
from pimsr_benchmarks.mtdlpy import (
    HeldoutObservations,
    MTDLPyAdapterError,
    TrainingOutcome,
    TrainingSplit,
    load_heldout_observations,
    load_training_split,
    resize_bilinear_half_pixel,
    run_common_retrain,
    validate_local_imagenet_weights,
    verify_pinned_repository,
)
from pimsr_benchmarks.runner2d import file_artifact_provenance


def _observation_arrays(n: int = 2) -> dict[str, np.ndarray]:
    shape = (n, *mtdlpy.INPUT_GRID_SHAPE)
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 100.0
    return {
        "schema": np.asarray("pimsr-sota-2d-observations"),
        "schema_version": np.asarray(1, dtype="<i8"),
        "sample_index": np.arange(200, 200 + n, dtype="<i8"),
        "frequency_hz": np.geomspace(0.01, 100.0, 8).astype("<f8"),
        "station_x_m": np.linspace(0.0, 11_000.0, 12).astype("<f8"),
        "x_cell_centers_m": np.linspace(-500.0, 11_500.0, 48).astype("<f8"),
        "depth_cell_centers_m": np.geomspace(10.0, 10_000.0, 64).astype("<f8"),
        "observation_channel_order": np.asarray(
            [
                "log10_rho_te",
                "phase_te_degrees",
                "log10_rho_tm",
                "phase_tm_degrees",
            ]
        ),
        "observed_log10_rho_te": (1.0 + values).astype("<f4"),
        "observed_phase_te_degrees": (20.0 + values).astype("<f4"),
        "observed_log10_rho_tm": (1.5 + values).astype("<f4"),
        "observed_phase_tm_degrees": (40.0 + values).astype("<f4"),
        "declared_evaluation_floor_log10_rho_te": np.full(shape, 0.05, "<f4"),
        "declared_evaluation_floor_phase_te_degrees": np.full(
            shape, 2.9, "<f4"
        ),
        "declared_evaluation_floor_log10_rho_tm": np.full(shape, 0.05, "<f4"),
        "declared_evaluation_floor_phase_tm_degrees": np.full(
            shape, 2.9, "<f4"
        ),
        "valid_mask": np.ones((n, 4, *mtdlpy.INPUT_GRID_SHAPE), dtype=bool),
    }


def _write_observations(path: Path, arrays: dict[str, np.ndarray] | None = None) -> Path:
    np.savez(path, **(arrays or _observation_arrays()))
    return path


def _schema_v2_h5(path: Path, *, sample_start: int) -> Path:
    from pimsr_forward.dataset2d import (
        _DEFAULT_SENSOR_PARAMETERS_JSON,
        _write_contract_attrs,
        _write_dataset_attrs,
    )

    n = 2
    observation_shape = (n, *mtdlpy.INPUT_GRID_SHAPE)
    base = np.arange(np.prod(observation_shape), dtype=np.float32).reshape(
        observation_shape
    )
    software_versions = json.dumps(
        {
            "discretize": "test",
            "h5py": "test",
            "numpy": "test",
            "pimsr_forward": "test",
            "pimsr_geogen": "test",
            "simpeg": "test",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    with h5py.File(path, "x") as h5:
        _write_contract_attrs(
            h5,
            generator_seed=91,
            generation_start_index=sample_start,
            expected_row_count=n,
            source_shard_count=1,
            generation_complete=True,
            sensor_parameters_json=_DEFAULT_SENSOR_PARAMETERS_JSON,
            software_versions_json=software_versions,
        )
        arrays = {
            "obs_mt_log10_rho": 1.0 + base / 100.0,
            "obs_mt_phase": 20.0 + base / 100.0,
            "clean_mt_log10_rho": 1.1 + base / 100.0,
            "clean_mt_phase": 21.0 + base / 100.0,
            "obs_mt_log10_rho_tm": 1.5 + base / 100.0,
            "obs_mt_phase_tm": 40.0 + base / 100.0,
            "clean_mt_log10_rho_tm": 1.6 + base / 100.0,
            "clean_mt_phase_tm": 41.0 + base / 100.0,
        }
        for name, values in arrays.items():
            h5.create_dataset(name, data=values.astype(np.float32))
        h5.create_dataset(
            "target_log10_res",
            data=np.linspace(
                0.0,
                1.0,
                n * np.prod(mtdlpy.OUTPUT_GRID_SHAPE),
                dtype=np.float32,
            ).reshape(n, *mtdlpy.OUTPUT_GRID_SHAPE),
        )
        h5.create_dataset("scenario", data=np.asarray([0, 1], dtype=np.int32))
        h5.create_dataset("has_fault", data=np.asarray([0, 1], dtype=np.uint8))
        h5.create_dataset(
            "sample_index",
            data=np.arange(sample_start, sample_start + n, dtype=np.int64),
        )
        h5.create_dataset(
            "frequencies", data=np.geomspace(0.01, 100.0, 8).astype(np.float64)
        )
        h5.create_dataset(
            "station_x", data=np.linspace(0.0, 11_000.0, 12).astype(np.float64)
        )
        h5.create_dataset(
            "x_grid", data=np.linspace(-500.0, 11_500.0, 48).astype(np.float64)
        )
        h5.create_dataset(
            "depth_grid", data=np.geomspace(10.0, 10_000.0, 64).astype(np.float64)
        )
        _write_dataset_attrs(h5)
    return path


def test_repository_verifier_requires_exact_url_commit_clean_tree_and_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo = tmp_path / "MTDLPy"
    source = repo / "func" / "dinknet.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"reviewed source\n")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(mtdlpy, "MTDLPY_DINKNET_SHA256", source_sha)
    monkeypatch.setattr(mtdlpy, "MTDLPY_DINKNET_GIT_BLOB", "a" * 40)
    dirty = False

    def fake_git(_repo: Path, *arguments: str, binary: bool = False):
        nonlocal dirty
        command = tuple(arguments)
        if command == ("rev-parse", "--show-toplevel"):
            return str(repo)
        if command == ("rev-parse", "--verify", "HEAD^{commit}"):
            return mtdlpy.MTDLPY_COMMIT
        if command == ("remote", "get-url", "--all", "origin"):
            return mtdlpy.MTDLPY_REPOSITORY_URL
        if command == ("status", "--porcelain=v1", "--untracked-files=all"):
            return "?? mutation.py" if dirty else ""
        if command == (
            "ls-tree",
            "--full-tree",
            "HEAD",
            "--",
            mtdlpy.MTDLPY_DINKNET_PATH,
        ):
            return (
                f"100644 blob {mtdlpy.MTDLPY_DINKNET_GIT_BLOB}"
                f"\t{mtdlpy.MTDLPY_DINKNET_PATH}"
            )
        if command == (
            "cat-file",
            "blob",
            f"{mtdlpy.MTDLPY_COMMIT}:{mtdlpy.MTDLPY_DINKNET_PATH}",
        ):
            assert binary
            return source.read_bytes()
        raise AssertionError(command)

    monkeypatch.setattr(mtdlpy, "_run_git", fake_git)
    result = verify_pinned_repository(repo)
    assert result["commit"] == mtdlpy.MTDLPY_COMMIT
    assert result["dinknet_source"]["sha256"] == source_sha

    dirty = True
    with pytest.raises(MTDLPyAdapterError, match="clean worktree"):
        verify_pinned_repository(repo)


def test_local_weights_require_preregistered_and_matching_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"local-only-weights")
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    monkeypatch.setattr(mtdlpy, "IMAGENET_RESNET50_V1_SHA256", digest)
    result = validate_local_imagenet_weights(weights, digest)
    assert result["sha256"] == digest
    with pytest.raises(MTDLPyAdapterError, match="preregistered"):
        validate_local_imagenet_weights(weights, "0" * 64)
    weights.write_bytes(b"mutated")
    with pytest.raises(MTDLPyAdapterError, match="declared SHA-256"):
        validate_local_imagenet_weights(weights, digest)


def test_observation_parser_accepts_exact_truth_free_materializer_contract(
    tmp_path: Path,
):
    arrays = _observation_arrays()
    arrays["sample_index"] = np.asarray([901, 117], dtype="<i8")
    path = _write_observations(tmp_path / "observations.npz", arrays)
    parsed = load_heldout_observations(path)
    assert parsed.observations.shape == (2, 4, 8, 12)
    assert parsed.evaluation_floors.shape == (2, 4, 8, 12)
    np.testing.assert_array_equal(parsed.sample_index, [901, 117])
    assert parsed.observations[0, 1, 0, 0] == pytest.approx(20.0)


@pytest.mark.parametrize("mutation", ["truth", "missing", "invalid_mask", "phase"])
def test_observation_parser_fails_closed_without_missing_data_interpolation(
    tmp_path: Path, mutation: str
):
    arrays = _observation_arrays()
    if mutation == "truth":
        arrays["truth_log10_resistivity"] = np.zeros((2, 64, 48), dtype="<f4")
    elif mutation == "missing":
        del arrays["observed_log10_rho_tm"]
    elif mutation == "invalid_mask":
        arrays["valid_mask"][0, 0, 0, 0] = False
    else:
        arrays["observed_phase_te_degrees"][0, 0, 0] = 180.0
    path = _write_observations(tmp_path / "invalid.npz", arrays)
    with pytest.raises(MTDLPyAdapterError):
        load_heldout_observations(path)


def test_training_loader_uses_schema_v2_four_channel_te_tm_order(tmp_path: Path):
    path = _schema_v2_h5(tmp_path / "train.h5", sample_start=10)
    split = load_training_split(path, role="training fixture")
    assert split.observations.shape == (2, 4, 8, 12)
    assert split.targets.shape == (2, 64, 48)
    assert split.generator_seed == 91
    assert split.observations[0, :, 0, 0].tolist() == pytest.approx(
        [1.0, 20.0, 1.5, 40.0]
    )


def test_fixed_half_pixel_resize_is_deterministic_and_rejects_nonfinite():
    source = np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    first = resize_bilinear_half_pixel(source, (4, 4))
    second = resize_bilinear_half_pixel(source, (4, 4))
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(
        first,
        [
            [0.0, 0.25, 0.75, 1.0],
            [0.5, 0.75, 1.25, 1.5],
            [1.5, 1.75, 2.25, 2.5],
            [2.0, 2.25, 2.75, 3.0],
        ],
    )
    source[0, 0] = np.nan
    with pytest.raises(ValueError, match="missing values are not interpolated"):
        resize_bilinear_half_pixel(source, (4, 4))


def test_observation_preprocessing_matches_upstream_resize_then_transpose():
    source = np.empty((1, 4, 8, 12), dtype=np.float32)
    base = np.arange(8 * 12, dtype=np.float32).reshape(1, 8, 12) / 100.0
    source[:, 0] = base
    source[:, 1] = 20.0 + base
    source[:, 2] = 1.0 + base
    source[:, 3] = 40.0 + base
    expected = np.empty((1, 4, *mtdlpy.NETWORK_GRID_SHAPE), dtype=np.float32)
    for channel in (0, 2):
        expected[:, channel] = np.log10(
            resize_bilinear_half_pixel(
                np.power(10.0, source[:, channel]), mtdlpy.NETWORK_GRID_SHAPE
            )
        )
    for channel in (1, 3):
        expected[:, channel] = resize_bilinear_half_pixel(
            source[:, channel], mtdlpy.NETWORK_GRID_SHAPE
        )

    transformed = mtdlpy._preprocess_observations(source)

    np.testing.assert_allclose(
        transformed, np.swapaxes(expected, -2, -1), rtol=1e-6, atol=1e-6
    )
    naive_log_resize = np.swapaxes(
        resize_bilinear_half_pixel(source, mtdlpy.NETWORK_GRID_SHAPE), -2, -1
    )
    assert not np.array_equal(transformed[:, 0], naive_log_resize[:, 0])
    assert not np.array_equal(transformed, expected)


def test_log10_resistivity_resize_operates_in_linear_scale():
    source = np.asarray([[0.0, 2.0]], dtype=np.float32)

    resized = mtdlpy._resize_log10_resistivity(source, (1, 3))

    np.testing.assert_allclose(resized, [[0.0, np.log10(50.5), 2.0]])
    assert resized[0, 1] != pytest.approx(1.0)
    with pytest.raises(MTDLPyAdapterError, match="positive linear values"):
        mtdlpy._resize_log10_resistivity(np.asarray([[-400.0]]), (1, 1))


def _split(path: Path, start: int, *, generator_seed: int = 91) -> TrainingSplit:
    path.write_bytes(f"split-{generator_seed}-{start}".encode())
    return TrainingSplit(
        observations=np.ones((2, 4, 8, 12), dtype=np.float32),
        targets=np.ones((2, 64, 48), dtype=np.float32),
        sample_index=np.arange(start, start + 2, dtype=np.int64),
        generator_seed=generator_seed,
        frequencies=np.geomspace(0.01, 100.0, 8),
        station_x=np.linspace(0.0, 11_000.0, 12),
        x_grid=np.linspace(-500.0, 11_500.0, 48),
        depth_grid=np.geomspace(10.0, 10_000.0, 64),
        provenance=file_artifact_provenance(path),
    )


def _heldout(path: Path) -> HeldoutObservations:
    path.write_bytes(b"truth-free-observations")
    return HeldoutObservations(
        observations=np.ones((2, 4, 8, 12), dtype=np.float32),
        evaluation_floors=np.ones((2, 4, 8, 12), dtype=np.float32),
        sample_index=np.asarray([901, 117], dtype=np.int64),
        frequencies=np.geomspace(0.01, 100.0, 8),
        station_x=np.linspace(0.0, 11_000.0, 12),
        x_grid=np.linspace(-500.0, 11_500.0, 48),
        depth_grid=np.geomspace(10.0, 10_000.0, 64),
        provenance=file_artifact_provenance(path),
    )


def test_split_identity_includes_generator_seed(tmp_path: Path):
    heldout = _heldout(tmp_path / "observations.npz")
    train = _split(tmp_path / "train.h5", 10, generator_seed=91)
    same_indices_new_campaign = _split(
        tmp_path / "validation-new-campaign.h5", 10, generator_seed=92
    )

    mtdlpy._require_disjoint_samples(train, same_indices_new_campaign, heldout)

    overlapping_same_campaign = _split(
        tmp_path / "validation-overlap.h5", 11, generator_seed=91
    )
    with pytest.raises(MTDLPyAdapterError, match="identities overlap"):
        mtdlpy._require_disjoint_samples(train, overlapping_same_campaign, heldout)


def test_common_retrain_publishes_exact_prediction_contract_and_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = tmp_path / "dinknet.py"
    source.write_bytes(b"source")
    repository = {
        "path": str(tmp_path / "repo"),
        "origin_url": mtdlpy.MTDLPY_REPOSITORY_URL,
        "commit": mtdlpy.MTDLPY_COMMIT,
        "clean_worktree": True,
        "dinknet_source": file_artifact_provenance(source),
        "dinknet_git_blob_sha1": "a" * 40,
        "dinknet_git_blob_sha256": "b" * 64,
    }
    weights_path = tmp_path / "weights.pth"
    weights_path.write_bytes(b"weights")
    weights = {
        **file_artifact_provenance(weights_path),
        "source_url": mtdlpy.IMAGENET_RESNET50_V1_URL,
        "artifact": "test weights",
    }
    train = _split(tmp_path / "train.h5", 10)
    validation = _split(tmp_path / "validation.h5", 20)
    heldout = _heldout(tmp_path / "observations.npz")

    monkeypatch.setattr(mtdlpy, "verify_pinned_repository", lambda _path: repository)
    monkeypatch.setattr(
        mtdlpy,
        "validate_local_imagenet_weights",
        lambda _path, _digest: weights,
    )

    def fake_split(_path: str | Path, *, role: str) -> TrainingSplit:
        return train if "training" in role and "validation" not in role else validation

    monkeypatch.setattr(mtdlpy, "load_training_split", fake_split)
    monkeypatch.setattr(mtdlpy, "load_heldout_observations", lambda _path: heldout)
    monkeypatch.setattr(mtdlpy, "_checkpoint_bytes", lambda _value: b"checkpoint")
    monkeypatch.setattr(
        mtdlpy,
        "_train_and_predict",
        lambda *_args, **_kwargs: TrainingOutcome(
            state_dict={},
            predicted_log10_resistivity=np.full((2, 64, 48), 2.5, np.float32),
            training_summary={"best_epoch": 3, "best_validation_mse": 0.25},
            runtime={
                "torch": "fake",
                "torchvision": "fake",
                "device": "cpu",
                "peak_cuda_memory_bytes": 0,
                "training_wall_time_s": 1.0,
                "inference_wall_time_s": 0.5,
            },
        ),
    )

    checkpoint = tmp_path / "outputs" / "seed101.pt"
    predictions = tmp_path / "outputs" / "seed101.npz"
    runtime = tmp_path / "outputs" / "seed101.json"
    result = run_common_retrain(
        repository_path=tmp_path / "repo",
        imagenet_weights_path=weights_path,
        imagenet_weights_sha256=mtdlpy.IMAGENET_RESNET50_V1_SHA256,
        train_h5=train.provenance["path"],
        validation_h5=validation.provenance["path"],
        observations_npz=heldout.provenance["path"],
        seed=101,
        device="cpu",
        checkpoint_out=checkpoint,
        predictions_out=predictions,
        runtime_out=runtime,
        command=["python", "run_mtdlpy_common.py", "--seed", "101"],
    )
    assert checkpoint.read_bytes() == b"checkpoint"
    with np.load(predictions, allow_pickle=False) as payload:
        assert tuple(payload.files) == mtdlpy.PREDICTION_KEYS
        assert payload["schema"].item() == mtdlpy.PREDICTION_SCHEMA
        assert mtdlpy.PREDICTION_SCHEMA_VERSION == 2
        assert payload["schema_version"].item() == mtdlpy.PREDICTION_SCHEMA_VERSION
        assert payload["observations_sha256"].shape == ()
        assert payload["observations_sha256"].dtype == np.dtype("<U64")
        assert payload["observations_sha256"].item() == heldout.provenance["sha256"]
        assert payload["sample_index"].dtype == np.dtype("<i8")
        assert payload["x_cell_centers_m"].dtype == np.dtype("<f8")
        assert payload["depth_cell_centers_m"].dtype == np.dtype("<f8")
        np.testing.assert_array_equal(payload["x_cell_centers_m"], heldout.x_grid)
        np.testing.assert_array_equal(
            payload["depth_cell_centers_m"], heldout.depth_grid
        )
        assert payload["predicted_log10_resistivity"].dtype == np.dtype("<f4")
        assert payload["predicted_log10_resistivity"].shape == (2, 64, 48)
    assert predictions.read_bytes() == mtdlpy._prediction_npz_bytes(
        str(heldout.provenance["sha256"]),
        heldout.sample_index,
        heldout.x_grid,
        heldout.depth_grid,
        np.full((2, 64, 48), 2.5, np.float32),
    )
    from pimsr_benchmarks.evaluation2d import load_predictions_2d

    evaluator_input = load_predictions_2d(predictions)
    np.testing.assert_array_equal(evaluator_input.sample_index, [901, 117])
    assert evaluator_input.observations_sha256 == heldout.provenance["sha256"]
    np.testing.assert_array_equal(evaluator_input.x_cell_centers_m, heldout.x_grid)
    np.testing.assert_array_equal(
        evaluator_input.depth_cell_centers_m, heldout.depth_grid
    )
    assert evaluator_input.log10_resistivity.shape == (2, 64, 48)
    published = json.loads(runtime.read_text(encoding="utf-8"))
    assert published == result
    assert published["training_config"]["optimizer"] == {
        "name": "Adam",
        "learning_rate": 1e-4,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.0,
    }
    assert published["training_config"]["normalization"] == "none"
    assert published["preprocessing"]["test_tuning"] is False
    assert published["preprocessing"]["transpose_observations"] is True
    assert published["preprocessing"]["observation_transform_order"] == {
        "apparent_resistivity": (
            "pow10_to_linear_then_bilinear_resize_then_log10_then_transpose"
        ),
        "phase": "bilinear_resize_then_transpose",
    }
    assert "not an MTDLPy upstream default" in published["training_config"][
        "schedule_origin"
    ]
    assert "[0,180)" in published["preprocessing"]["phase_domain_adaptation"]
    assert "upstream does not define" in published["preprocessing"][
        "prediction_resize_origin"
    ]
    assert published["observation_contract"]["truth_keys_accepted"] is False
    assert published["prediction_contract"]["keys_exact"] == list(
        mtdlpy.PREDICTION_KEYS
    )
    assert published["prediction_contract"]["schema_version"] == 2
    assert published["prediction_contract"]["observations_sha256"] == (
        heldout.provenance["sha256"]
    )
    assert published["prediction_contract"]["observations_sha256_dtype"] == "<U64"
    assert published["prediction_contract"]["x_cell_centers_dtype"] == "<f8"
    assert published["prediction_contract"]["depth_cell_centers_dtype"] == "<f8"
    assert published["outputs"]["predictions"]["sha256"] == hashlib.sha256(
        predictions.read_bytes()
    ).hexdigest()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_common_retrain(
            repository_path=tmp_path / "repo",
            imagenet_weights_path=weights_path,
            imagenet_weights_sha256=mtdlpy.IMAGENET_RESNET50_V1_SHA256,
            train_h5=train.provenance["path"],
            validation_h5=validation.provenance["path"],
            observations_npz=heldout.provenance["path"],
            seed=101,
            device="cpu",
            checkpoint_out=checkpoint,
            predictions_out=predictions,
            runtime_out=runtime,
        )

    rollback_outputs = (
        tmp_path / "rollback" / "checkpoint.pt",
        tmp_path / "rollback" / "predictions.npz",
        tmp_path / "rollback" / "runtime.json",
    )
    real_link = mtdlpy.os.link
    link_calls = 0

    def fail_second_link(source_path: Path, destination_path: Path) -> None:
        nonlocal link_calls
        link_calls += 1
        if link_calls == 2:
            raise OSError("injected publication failure")
        real_link(source_path, destination_path)

    monkeypatch.setattr(mtdlpy.os, "link", fail_second_link)
    with pytest.raises(OSError, match="injected publication failure"):
        run_common_retrain(
            repository_path=tmp_path / "repo",
            imagenet_weights_path=weights_path,
            imagenet_weights_sha256=mtdlpy.IMAGENET_RESNET50_V1_SHA256,
            train_h5=train.provenance["path"],
            validation_h5=validation.provenance["path"],
            observations_npz=heldout.provenance["path"],
            seed=101,
            device="cpu",
            checkpoint_out=rollback_outputs[0],
            predictions_out=rollback_outputs[1],
            runtime_out=rollback_outputs[2],
        )
    assert not any(path.exists() for path in rollback_outputs)
    assert not any(path.with_name(path.name + ".part").exists() for path in rollback_outputs)


def test_publication_rolls_back_when_link_succeeds_then_interrupts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    parts = [tmp_path / f"artifact-{index}.part" for index in range(3)]
    destinations = [tmp_path / f"artifact-{index}.out" for index in range(3)]
    for part in parts:
        part.write_bytes(b"staged")
    real_link = mtdlpy.os.link

    def interrupted_link(source: Path, destination: Path) -> None:
        real_link(source, destination)
        raise KeyboardInterrupt

    monkeypatch.setattr(mtdlpy.os, "link", interrupted_link)

    with pytest.raises(KeyboardInterrupt):
        mtdlpy._publish_parts(parts, destinations)

    assert all(part.exists() for part in parts)
    assert not any(destination.exists() for destination in destinations)


def test_common_retrain_rejects_noncampaign_seed_before_work(tmp_path: Path):
    with pytest.raises(ValueError, match="seed must be one of"):
        run_common_retrain(
            repository_path=tmp_path,
            imagenet_weights_path=tmp_path / "weights",
            imagenet_weights_sha256="0" * 64,
            train_h5=tmp_path / "train",
            validation_h5=tmp_path / "validation",
            observations_npz=tmp_path / "observations",
            seed=999,
            device="cpu",
            checkpoint_out=tmp_path / "checkpoint",
            predictions_out=tmp_path / "predictions",
            runtime_out=tmp_path / "runtime",
        )
