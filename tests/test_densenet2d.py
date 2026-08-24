"""Fail-closed tests for the pinned MT2DInv-DenseNet adapter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pimsr_benchmarks import densenet2d
from pimsr_benchmarks.densenet2d import (
    DenseNet2DAdapterError,
    HeldoutObservations,
    TrainingOutcome,
    TrainingSplit,
    preprocess_observations,
    resize_log10_resistivity,
    run_common_retrain,
    verify_pinned_repository,
)
from pimsr_benchmarks.runner2d import file_artifact_provenance


def _observation_arrays(n: int = 2) -> dict[str, np.ndarray]:
    shape = (n, *densenet2d.INPUT_GRID_SHAPE)
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 100.0
    return {
        "schema": np.asarray("pimsr-sota-2d-observations"),
        "schema_version": np.asarray(1, dtype="<i8"),
        "sample_index": np.asarray([901, 117][:n], dtype="<i8"),
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
        "declared_evaluation_floor_phase_te_degrees": np.full(shape, 2.9, "<f4"),
        "declared_evaluation_floor_log10_rho_tm": np.full(shape, 0.05, "<f4"),
        "declared_evaluation_floor_phase_tm_degrees": np.full(shape, 2.9, "<f4"),
        "valid_mask": np.ones((n, 4, *densenet2d.INPUT_GRID_SHAPE), dtype=bool),
    }


def _write_observations(path: Path, arrays: dict[str, np.ndarray] | None = None) -> Path:
    np.savez(path, **(arrays or _observation_arrays()))
    return path


def _artifact(path: Path, payload: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return file_artifact_provenance(path)


def _split(
    path: Path,
    start: int,
    *,
    generator_seed: int,
) -> TrainingSplit:
    provenance = _artifact(path, f"split-{start}-{generator_seed}".encode())
    frequencies = np.geomspace(0.01, 100.0, 8).astype("<f8")
    station = np.linspace(0.0, 11_000.0, 12).astype("<f8")
    x_grid = np.linspace(-500.0, 11_500.0, 48).astype("<f8")
    depth = np.geomspace(10.0, 10_000.0, 64).astype("<f8")
    return TrainingSplit(
        observations=np.ones((2, 4, 8, 12), dtype=np.float32),
        targets=np.full((2, 64, 48), 2.0, dtype=np.float32),
        sample_index=np.arange(start, start + 2, dtype=np.int64),
        generator_seed=generator_seed,
        frequencies=frequencies,
        station_x=station,
        x_grid=x_grid,
        depth_grid=depth,
        provenance=provenance,
    )


def _heldout(path: Path) -> HeldoutObservations:
    provenance = _artifact(path, b"truth-free observations")
    return HeldoutObservations(
        observations=np.ones((2, 4, 8, 12), dtype=np.float32),
        evaluation_floors=np.ones((2, 4, 8, 12), dtype=np.float32),
        sample_index=np.asarray([901, 117], dtype=np.int64),
        frequencies=np.geomspace(0.01, 100.0, 8).astype("<f8"),
        station_x=np.linspace(0.0, 11_000.0, 12).astype("<f8"),
        x_grid=np.linspace(-500.0, 11_500.0, 48).astype("<f8"),
        depth_grid=np.geomspace(10.0, 10_000.0, 64).astype("<f8"),
        provenance=provenance,
    )


def test_repository_verifier_requires_url_commit_tags_clean_tree_and_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo = tmp_path / "MT2DInv-DenseNet"
    source = repo / densenet2d.MT2DINV_DENSENET_SOURCE_PATH
    source.parent.mkdir(parents=True)
    source.write_bytes(b"reviewed architecture\n")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(densenet2d, "MT2DINV_DENSENET_SOURCE_SHA256", source_sha)
    monkeypatch.setattr(densenet2d, "MT2DINV_DENSENET_SOURCE_GIT_BLOB", "a" * 40)
    dirty = False
    bad_tag = False
    expected_blob = (
        f"{densenet2d.MT2DINV_DENSENET_COMMIT}:{densenet2d.MT2DINV_DENSENET_SOURCE_PATH}"
    )

    def fake_git(_repo: Path, *arguments: str, binary: bool = False):
        command = tuple(arguments)
        if command == ("rev-parse", "--show-toplevel"):
            return str(repo)
        if command == ("rev-parse", "--verify", "HEAD^{commit}"):
            return densenet2d.MT2DINV_DENSENET_COMMIT
        if (
            len(command) == 3
            and command[:2] == ("rev-parse", "--verify")
            and command[2].startswith("refs/tags/")
        ):
            return "b" * 40 if bad_tag else densenet2d.MT2DINV_DENSENET_COMMIT
        if command == ("remote", "get-url", "--all", "origin"):
            return densenet2d.MT2DINV_DENSENET_REPOSITORY_URL
        if command == ("status", "--porcelain=v1", "--untracked-files=all"):
            return "?? mutation.py" if dirty else ""
        if command == (
            "ls-tree",
            "--full-tree",
            "HEAD",
            "--",
            densenet2d.MT2DINV_DENSENET_SOURCE_PATH,
        ):
            return (
                f"100644 blob {densenet2d.MT2DINV_DENSENET_SOURCE_GIT_BLOB}"
                f"\t{densenet2d.MT2DINV_DENSENET_SOURCE_PATH}"
            )
        if command == (
            "cat-file",
            "blob",
            expected_blob,
        ):
            assert binary
            return source.read_bytes()
        raise AssertionError(command)

    monkeypatch.setattr(densenet2d, "_run_git", fake_git)
    result = verify_pinned_repository(repo)
    assert result["commit"] == densenet2d.MT2DINV_DENSENET_COMMIT
    assert result["clean_worktree"] is True
    assert result["release_tags_reviewed"] == ["v1.1", "v1.2"]
    assert set(result["release_tag_commits"]) == {"v1.1", "v1.2"}
    assert result["architecture_source"]["sha256"] == source_sha

    bad_tag = True
    with pytest.raises(DenseNet2DAdapterError, match="tag v1.1"):
        verify_pinned_repository(repo)
    bad_tag = False
    dirty = True
    with pytest.raises(DenseNet2DAdapterError, match="clean worktree"):
        verify_pinned_repository(repo)


def test_ast_loader_executes_only_exact_six_classes_and_checks_parameter_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = tmp_path / "reviewed.py"
    definitions = """raise RuntimeError('top-level training must never execute')
class ICBAM: pass
class ChannelAttention: pass
class SpatialAttention: pass
class DenseBlock: pass
class TransitionLayer: pass
class DenseNetWithICBAM:
    def __init__(self, num_blocks, growth_rate, num_classes):
        self.arguments = (num_blocks, growth_rate, num_classes)
    def parameters(self):
        class Parameter:
            def numel(self): return 25908034
        return [Parameter()]
"""
    source.write_text(definitions, encoding="utf-8")
    identity = file_artifact_provenance(source)
    monkeypatch.setattr(densenet2d, "MT2DINV_DENSENET_SOURCE_SHA256", identity["sha256"])
    fake_torch = SimpleNamespace(nn=SimpleNamespace(functional=SimpleNamespace()))
    model = densenet2d._load_upstream_model(
        {"architecture_source": identity}, torch=fake_torch
    )
    assert model.arguments == ([6, 12, 24, 16], 32, 2176)

    source.write_text(
        definitions.replace("class SpatialAttention: pass\n", ""),
        encoding="utf-8",
    )
    changed = file_artifact_provenance(source)
    monkeypatch.setattr(densenet2d, "MT2DINV_DENSENET_SOURCE_SHA256", changed["sha256"])
    with pytest.raises(DenseNet2DAdapterError, match="exact six"):
        densenet2d._load_upstream_model(
            {"architecture_source": changed}, torch=fake_torch
        )


def test_truth_free_loader_rejects_extra_truth_member(tmp_path: Path):
    path = _write_observations(tmp_path / "observations.npz")
    heldout = densenet2d.load_heldout_observations(path)
    assert heldout.observations.shape == (2, 4, 8, 12)
    assert heldout.sample_index.tolist() == [901, 117]

    arrays = _observation_arrays()
    arrays["target_log10_resistivity"] = np.zeros((2, 64, 48), dtype="<f4")
    path = _write_observations(tmp_path / "truth-leak.npz", arrays)
    with pytest.raises(DenseNet2DAdapterError, match="truth-free contract"):
        densenet2d.load_heldout_observations(path)


def test_coordinate_interpolation_reorder_and_upstream_normalization_are_exact():
    frequencies = np.geomspace(0.01, 100.0, 8).astype(np.float64)
    stations = np.linspace(-5_000.0, 6_000.0, 12).astype(np.float64)
    log_f = np.log10(frequencies)
    components = []
    for component in range(4):
        components.append(
            (component + 1.0) * log_f[:, None]
            + (component + 0.5) * stations[None, :] / 10_000.0
            + component * 9.0
        )
    source = np.stack(components, axis=0)[None].astype(np.float32)
    actual = preprocess_observations(source, frequencies, stations)
    assert actual.shape == (1, 16, 33, 4)

    target_f, target_x = densenet2d._network_observation_axes(frequencies, stations)
    expected_components = []
    for component in range(4):
        expected_components.append(
            (component + 1.0) * np.log10(target_f)[:, None]
            + (component + 0.5) * target_x[None, :] / 10_000.0
            + component * 9.0
        )
    expected = np.stack(expected_components, axis=-1)[None]
    expected -= expected.mean(axis=(1, 2), keepdims=True)
    expected /= np.max(np.abs(expected), axis=(1, 2), keepdims=True)
    np.testing.assert_allclose(actual, expected.astype(np.float32), atol=2e-7)
    np.testing.assert_allclose(actual.mean(axis=(1, 2)), 0.0, atol=2e-7)
    np.testing.assert_allclose(np.max(np.abs(actual), axis=(1, 2)), 1.0)


def test_normalization_rejects_constant_or_nonfinite_component():
    frequencies = np.geomspace(0.01, 100.0, 8)
    stations = np.linspace(0.0, 11_000.0, 12)
    values = np.arange(4 * 8 * 12, dtype=np.float32).reshape(1, 4, 8, 12)
    values[:, 2] = 5.0
    with pytest.raises(DenseNet2DAdapterError, match="undefined"):
        preprocess_observations(values, frequencies, stations)
    values[:, 2, 0, 0] = np.nan
    with pytest.raises(DenseNet2DAdapterError, match="finite"):
        preprocess_observations(values, frequencies, stations)


def test_target_and_prediction_resize_operates_in_linear_resistivity():
    source = np.asarray([[0.0, 2.0]], dtype=np.float32)
    resized = resize_log10_resistivity(source, (1, 3))
    expected = np.log10(np.asarray([[1.0, 50.5, 100.0]])).astype(np.float32)
    np.testing.assert_allclose(resized, expected, atol=1e-6)
    assert not np.isclose(resized[0, 1], 1.0)


def test_determinism_seeds_all_backends_and_disables_tf32(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple[str, object]] = []

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def manual_seed_all(seed: int) -> None:
            calls.append(("cuda_seed", seed))

    class _Cudnn:
        deterministic = False
        benchmark = True
        allow_tf32 = True

    class _Matmul:
        allow_tf32 = True

    class _Backends:
        cudnn = _Cudnn()
        cuda = SimpleNamespace(matmul=_Matmul())

    class _Torch:
        cuda = _Cuda()
        backends = _Backends()

        @staticmethod
        def manual_seed(seed: int) -> None:
            calls.append(("torch_seed", seed))

        @staticmethod
        def use_deterministic_algorithms(value: bool) -> None:
            calls.append(("deterministic", value))

        @staticmethod
        def set_float32_matmul_precision(value: str) -> None:
            calls.append(("matmul_precision", value))

    python_seeds: list[int] = []
    numpy_seeds: list[int] = []
    monkeypatch.setattr(densenet2d.random, "seed", python_seeds.append)
    monkeypatch.setattr(densenet2d.np.random, "seed", numpy_seeds.append)
    densenet2d._configure_determinism(_Torch(), 101)
    assert python_seeds == [101]
    assert numpy_seeds == [101]
    assert calls == [
        ("torch_seed", 101),
        ("cuda_seed", 101),
        ("deterministic", True),
        ("matmul_precision", "highest"),
    ]
    assert _Torch.backends.cudnn.deterministic is True
    assert _Torch.backends.cudnn.benchmark is False
    assert _Torch.backends.cudnn.allow_tf32 is False
    assert _Torch.backends.cuda.matmul.allow_tf32 is False


def _fake_run_inputs(tmp_path: Path):
    source = tmp_path / "repo" / "Improved Densenet" / "train.py"
    source_identity = _artifact(source, b"reviewed source")
    repository = {
        "path": str(tmp_path / "repo"),
        "origin_url": densenet2d.MT2DINV_DENSENET_REPOSITORY_URL,
        "commit": densenet2d.MT2DINV_DENSENET_COMMIT,
        "clean_worktree": True,
        "release_tags_reviewed": ["v1.1", "v1.2"],
        "release_tag_commits": {
            "v1.1": densenet2d.MT2DINV_DENSENET_COMMIT,
            "v1.2": densenet2d.MT2DINV_DENSENET_COMMIT,
        },
        "architecture_source": source_identity,
        "architecture_git_blob_sha1": "a" * 40,
        "architecture_git_blob_sha256": "b" * 64,
    }
    train = _split(tmp_path / "train.h5", 10, generator_seed=91)
    validation = _split(tmp_path / "validation.h5", 20, generator_seed=92)
    heldout = _heldout(tmp_path / "observations.npz")
    return repository, train, validation, heldout


def _patch_fake_run(
    monkeypatch: pytest.MonkeyPatch,
    repository: dict[str, object],
    train: TrainingSplit,
    validation: TrainingSplit,
    heldout: HeldoutObservations,
) -> None:
    monkeypatch.setattr(densenet2d, "verify_pinned_repository", lambda _path: repository)

    def fake_split(_path: str | Path, *, role: str) -> TrainingSplit:
        return train if "training" in role and "validation" not in role else validation

    monkeypatch.setattr(densenet2d, "load_training_split", fake_split)
    monkeypatch.setattr(densenet2d, "load_heldout_observations", lambda _path: heldout)
    monkeypatch.setattr(densenet2d, "_checkpoint_bytes", lambda _value: b"checkpoint")
    monkeypatch.setattr(
        densenet2d,
        "_train_and_predict",
        lambda *_args, **_kwargs: TrainingOutcome(
            state_dict={},
            predicted_log10_resistivity=np.full((2, 64, 48), 2.5, np.float32),
            training_summary={
                "best_epoch": 17,
                "best_validation_weighted_mse": 0.25,
            },
            runtime={"torch": "fake", "device": "cpu"},
        ),
    )


def test_common_retrain_publishes_bound_prediction_v2_and_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repository, train, validation, heldout = _fake_run_inputs(tmp_path)
    _patch_fake_run(monkeypatch, repository, train, validation, heldout)
    checkpoint = tmp_path / "outputs" / "seed101.pt"
    predictions = tmp_path / "outputs" / "seed101.npz"
    runtime = tmp_path / "outputs" / "seed101.json"
    result = run_common_retrain(
        repository_path=tmp_path / "repo",
        train_h5=train.provenance["path"],
        validation_h5=validation.provenance["path"],
        observations_npz=heldout.provenance["path"],
        seed=101,
        device="cpu",
        checkpoint_out=checkpoint,
        predictions_out=predictions,
        runtime_out=runtime,
        command=["python", "run_densenet2d_common.py", "--seed", "101"],
    )
    assert checkpoint.read_bytes() == b"checkpoint"
    with np.load(predictions, allow_pickle=False) as payload:
        assert tuple(payload.files) == densenet2d.PREDICTION_KEYS
        assert payload["schema"].item() == densenet2d.PREDICTION_SCHEMA
        assert payload["schema_version"].item() == 2
        assert payload["observations_sha256"].item() == heldout.provenance["sha256"]
        assert payload["observations_sha256"].dtype == np.dtype("<U64")
        assert payload["sample_index"].dtype == np.dtype("<i8")
        assert payload["x_cell_centers_m"].dtype == np.dtype("<f8")
        assert payload["depth_cell_centers_m"].dtype == np.dtype("<f8")
        assert payload["predicted_log10_resistivity"].dtype == np.dtype("<f4")
        assert payload["predicted_log10_resistivity"].shape == (2, 64, 48)
        np.testing.assert_array_equal(payload["sample_index"], [901, 117])
        np.testing.assert_array_equal(payload["x_cell_centers_m"], heldout.x_grid)

    from pimsr_benchmarks.evaluation2d import load_predictions_2d

    evaluator_input = load_predictions_2d(predictions)
    assert evaluator_input.observations_sha256 == heldout.provenance["sha256"]
    published = json.loads(runtime.read_text(encoding="utf-8"))
    assert published == result
    assert published["schema"] == densenet2d.RUNTIME_SCHEMA
    assert published["schema_version"] == 1
    assert published["method_id"] == "mt2dinv_densenet"
    assert published["method"] == "MT2DInv-DenseNet/iDenseNet"
    assert published["seed"] == 101
    assert published["training_config"]["epochs"] == 200
    assert published["training_config"]["batch_size"] == 100
    assert published["training_config"]["optimizer"] == {
        "name": "Adam",
        "learning_rate": 1e-4,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.0,
        "amsgrad": False,
    }
    assert published["training_config"]["loss"]["non_background_multiplier"] == 10
    assert (
        "upstream saves the last epoch"
        in published["training_config"]["checkpoint_selection_origin"]
    )
    assert published["training_config"]["equal_compute_claim"] is False
    assert published["preprocessing"]["network_input_shape_excluding_batch"] == [
        16,
        33,
        4,
    ]
    assert published["preprocessing"][
        "geometry_and_phase_adaptations_are_benchmark_specific"
    ]
    assert (
        "undefined/NaN"
        in published["preprocessing"]["normalization"]["zero_channel_policy"]
    )
    assert published["observation_contract"]["truth_keys_accepted"] is False
    assert published["prediction_contract"]["contains_truth"] is False
    assert published["bindings"]["training_seed"] == 101
    assert published["bindings"]["source_commit"] == repository["commit"]
    assert published["bindings"]["source_clean_worktree"] is True
    assert published["bindings"]["train_sha256"] == train.provenance["sha256"]
    assert published["bindings"]["validation_sha256"] == validation.provenance["sha256"]
    assert published["bindings"]["observations_sha256"] == heldout.provenance["sha256"]
    assert (
        published["bindings"]["adapter_source_sha256"]
        == published["source_artifacts"]["adapter_source"]["sha256"]
    )
    assert (
        published["bindings"]["shared_contract_loader_source_sha256"]
        == published["source_artifacts"]["shared_contract_loader_source"]["sha256"]
    )
    assert (
        published["bindings"]["checkpoint_sha256"]
        == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    )
    assert (
        published["bindings"]["prediction_sha256"]
        == hashlib.sha256(predictions.read_bytes()).hexdigest()
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_common_retrain(
            repository_path=tmp_path / "repo",
            train_h5=train.provenance["path"],
            validation_h5=validation.provenance["path"],
            observations_npz=heldout.provenance["path"],
            seed=101,
            device="cpu",
            checkpoint_out=checkpoint,
            predictions_out=predictions,
            runtime_out=runtime,
        )


def test_publication_failure_and_interrupt_rollback_owned_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    parts = [tmp_path / f"artifact-{index}.part" for index in range(3)]
    destinations = [tmp_path / f"artifact-{index}.out" for index in range(3)]
    for part in parts:
        part.write_bytes(b"staged")
    real_link = densenet2d.os.link
    calls = 0

    def fail_second(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected failure")
        real_link(source, destination)

    monkeypatch.setattr(densenet2d.os, "link", fail_second)
    with pytest.raises(OSError, match="injected failure"):
        densenet2d._publish_parts(parts, destinations)
    assert all(part.exists() for part in parts)
    assert not any(destination.exists() for destination in destinations)

    calls = 0

    def interrupt_after_link(source: Path, destination: Path) -> None:
        real_link(source, destination)
        raise KeyboardInterrupt

    monkeypatch.setattr(densenet2d.os, "link", interrupt_after_link)
    with pytest.raises(KeyboardInterrupt):
        densenet2d._publish_parts(parts, destinations)
    assert all(part.exists() for part in parts)
    assert not any(destination.exists() for destination in destinations)


def test_source_mutation_aborts_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repository, train, validation, heldout = _fake_run_inputs(tmp_path)
    _patch_fake_run(monkeypatch, repository, train, validation, heldout)
    source_path = Path(str(repository["architecture_source"]["path"]))

    def mutate_then_return(*_args, **_kwargs) -> TrainingOutcome:
        source_path.write_bytes(b"mutated source")
        return TrainingOutcome(
            state_dict={},
            predicted_log10_resistivity=np.full((2, 64, 48), 2.5, np.float32),
            training_summary={},
            runtime={},
        )

    monkeypatch.setattr(densenet2d, "_train_and_predict", mutate_then_return)
    outputs = (
        tmp_path / "mutated" / "checkpoint.pt",
        tmp_path / "mutated" / "predictions.npz",
        tmp_path / "mutated" / "runtime.json",
    )
    with pytest.raises(RuntimeError, match="changed after it was loaded"):
        run_common_retrain(
            repository_path=tmp_path / "repo",
            train_h5=train.provenance["path"],
            validation_h5=validation.provenance["path"],
            observations_npz=heldout.provenance["path"],
            seed=101,
            device="cpu",
            checkpoint_out=outputs[0],
            predictions_out=outputs[1],
            runtime_out=outputs[2],
        )
    assert not any(path.exists() for path in outputs)
    assert not any(path.with_name(path.name + ".part").exists() for path in outputs)


def test_rejects_noncampaign_seed_before_any_filesystem_work(tmp_path: Path):
    with pytest.raises(ValueError, match="seed must be one of"):
        run_common_retrain(
            repository_path=tmp_path / "missing-repo",
            train_h5=tmp_path / "missing-train",
            validation_h5=tmp_path / "missing-validation",
            observations_npz=tmp_path / "missing-observations",
            seed=999,
            device="cpu",
            checkpoint_out=tmp_path / "checkpoint.pt",
            predictions_out=tmp_path / "predictions.npz",
            runtime_out=tmp_path / "runtime.json",
        )
    assert not (tmp_path / "checkpoint.pt").exists()
