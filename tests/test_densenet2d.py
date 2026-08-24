"""Fail-closed tests for the pinned MT2DInv-DenseNet adapter."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import h5py
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
    run_common_inference,
    train_common_retrain,
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


def _schema_v2_h5(path: Path, *, sample_start: int) -> Path:
    from pimsr_forward.dataset2d import (
        _DEFAULT_SENSOR_PARAMETERS_JSON,
        _write_contract_attrs,
        _write_dataset_attrs,
    )

    n = 2
    observation_shape = (n, *densenet2d.INPUT_GRID_SHAPE)
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
                n * np.prod(densenet2d.OUTPUT_GRID_SHAPE),
                dtype=np.float32,
            ).reshape(n, *densenet2d.OUTPUT_GRID_SHAPE),
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


def _heldout(
    path: Path,
    *,
    payload: bytes = b"truth-free observations",
    sample_index: np.ndarray | None = None,
) -> HeldoutObservations:
    provenance = _artifact(path, payload)
    return HeldoutObservations(
        observations=np.ones((2, 4, 8, 12), dtype=np.float32),
        evaluation_floors=np.ones((2, 4, 8, 12), dtype=np.float32),
        sample_index=(
            np.asarray([901, 117], dtype=np.int64)
            if sample_index is None
            else np.asarray(sample_index, dtype=np.int64)
        ),
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


def test_descriptor_snapshot_rejects_rename_between_lstat_and_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    artifact = tmp_path / "artifact.bin"
    original = tmp_path / "original.bin"
    replacement = tmp_path / "replacement.bin"
    artifact.write_bytes(b"artifact-a")
    replacement.write_bytes(b"artifact-b")
    real_open = os.open

    def swap_before_open(path: str | bytes | os.PathLike[str], flags: int) -> int:
        if Path(path) == artifact:
            os.replace(artifact, original)
            os.replace(replacement, artifact)
        return real_open(path, flags)

    monkeypatch.setattr(densenet2d.os, "open", swap_before_open)
    with pytest.raises(DenseNet2DAdapterError, match="changed before it was opened"):
        densenet2d._snapshot_regular_file(artifact, role="adversarial artifact")


def test_pinned_npz_and_h5_bytes_survive_path_rename_without_provenance_drift(
    tmp_path: Path,
):
    observations_a = _write_observations(tmp_path / "observations-a.npz")
    arrays_b = _observation_arrays()
    arrays_b["sample_index"] = np.asarray([7001, 7002], dtype="<i8")
    observations_b = _write_observations(tmp_path / "observations-b.npz", arrays_b)
    observation_snapshot = densenet2d._snapshot_regular_file(
        observations_a, role="truth-free observations"
    )
    observations_backup = tmp_path / "observations-a.original.npz"
    os.replace(observations_a, observations_backup)
    os.replace(observations_b, observations_a)
    heldout = densenet2d._load_heldout_observations_snapshot(observation_snapshot)
    assert heldout.sample_index.tolist() == [901, 117]
    assert (
        heldout.provenance["sha256"]
        == hashlib.sha256(observations_backup.read_bytes()).hexdigest()
    )

    training_a = _schema_v2_h5(tmp_path / "training-a.h5", sample_start=100)
    training_b = _schema_v2_h5(tmp_path / "training-b.h5", sample_start=200)
    training_snapshot = densenet2d._snapshot_regular_file(
        training_a, role="training dataset"
    )
    training_backup = tmp_path / "training-a.original.h5"
    os.replace(training_a, training_backup)
    os.replace(training_b, training_a)
    split = densenet2d._load_training_split_snapshot(
        training_snapshot, role="training dataset"
    )
    assert split.sample_index.tolist() == [100, 101]
    assert (
        split.provenance["sha256"]
        == hashlib.sha256(training_backup.read_bytes()).hexdigest()
    )


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


def test_publication_rejects_staged_replacement_and_post_link_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    replaced_part = tmp_path / "replaced.part"
    replaced_part.write_bytes(b"validated")
    replaced_snapshot = densenet2d._snapshot_regular_file(
        replaced_part, role="validated staged artifact"
    )
    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"replacement")
    os.replace(replacement, replaced_part)
    replaced_destination = tmp_path / "replaced.out"
    with pytest.raises(DenseNet2DAdapterError, match="changed after validation"):
        densenet2d._publish_parts(
            (replaced_part,),
            (replaced_destination,),
            expected_snapshots=(replaced_snapshot,),
        )
    assert not replaced_destination.exists()

    mutated_part = tmp_path / "mutated.part"
    mutated_part.write_bytes(b"validated")
    mutated_snapshot = densenet2d._snapshot_regular_file(
        mutated_part, role="validated staged artifact"
    )
    mutated_destination = tmp_path / "mutated.out"
    real_link = densenet2d.os.link

    def mutate_after_link(source: Path, destination: Path) -> None:
        real_link(source, destination)
        destination.write_bytes(b"post-link mutation")

    monkeypatch.setattr(densenet2d.os, "link", mutate_after_link)
    with pytest.raises(DenseNet2DAdapterError, match="changed after validation"):
        densenet2d._publish_parts(
            (mutated_part,),
            (mutated_destination,),
            expected_snapshots=(mutated_snapshot,),
        )
    assert not mutated_destination.exists()


def test_output_symlink_is_rejected_before_path_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    requested = tmp_path / "redirect.pt"
    real_lexists = densenet2d.os.path.lexists
    real_is_link = densenet2d._path_is_link

    monkeypatch.setattr(
        densenet2d.os.path,
        "lexists",
        lambda path: Path(path) == requested or real_lexists(path),
    )
    monkeypatch.setattr(
        densenet2d,
        "_path_is_link",
        lambda path: Path(path) == requested or real_is_link(path),
    )
    with pytest.raises(DenseNet2DAdapterError, match="must not be a symbolic link"):
        densenet2d._prepare_new_outputs(((requested, ".pt"),))


class _StateOnlyModel:
    def state_dict(self) -> dict[str, object]:
        import torch

        return {"weight": torch.zeros(1, dtype=torch.float32)}

    def load_state_dict(self, state: dict[str, object], *, strict: bool) -> None:
        import torch

        assert strict is True
        if set(state) != {"weight"} or not isinstance(state["weight"], torch.Tensor):
            raise RuntimeError("incompatible fake state")
        if tuple(state["weight"].shape) != (1,):
            raise RuntimeError("incompatible fake tensor shape")


def _training_history() -> list[dict[str, object]]:
    return [
        {
            "epoch": epoch,
            "train_weighted_mse": float(300 - epoch),
            "validation_weighted_mse": float(abs(epoch - 17) + 0.25),
        }
        for epoch in range(1, densenet2d.EPOCHS + 1)
    ]


def _training_runtime(**updates: object) -> dict[str, object]:
    runtime: dict[str, object] = {
        "python": "test-python",
        "platform": "test-platform",
        "numpy": "test-numpy",
        "torch": "test-torch",
        "torch_cuda_build": None,
        "cuda_available": False,
        "device": "cpu",
        "cuda_device_name": None,
        "peak_cuda_memory_bytes": 0,
        "preprocessing_wall_time_s": 0.1,
        "model_initialization_wall_time_s": 0.2,
        "training_wall_time_s": 1.0,
        "backend_wall_time_s": 1.3,
    }
    runtime.update(updates)
    return runtime


def _inference_runtime(**updates: object) -> dict[str, object]:
    runtime: dict[str, object] = {
        "python": "test-python",
        "platform": "test-platform",
        "numpy": "test-numpy",
        "torch": "test-torch",
        "torch_cuda_build": None,
        "cuda_available": False,
        "device": "cpu",
        "cuda_device_name": None,
        "peak_cuda_memory_bytes": 0,
        "preprocessing_wall_time_s": 0.1,
        "model_initialization_wall_time_s": 0.2,
        "inference_wall_time_s": 0.3,
        "backend_wall_time_s": 0.6,
    }
    runtime.update(updates)
    return runtime


def _patch_split_training(
    monkeypatch: pytest.MonkeyPatch,
    repository: dict[str, object],
    train: TrainingSplit,
    validation: TrainingSplit,
) -> None:
    import torch

    monkeypatch.setattr(densenet2d, "verify_pinned_repository", lambda _path: repository)

    def fake_split(_path: str | Path, *, role: str) -> TrainingSplit:
        return train if "training dataset" in role else validation

    monkeypatch.setattr(densenet2d, "load_training_split", fake_split)
    monkeypatch.setattr(
        densenet2d,
        "_load_upstream_model",
        lambda *_args, **_kwargs: _StateOnlyModel(),
    )
    monkeypatch.setattr(
        densenet2d,
        "_train_model",
        lambda *_args, **_kwargs: TrainingOutcome(
            state_dict={"weight": torch.asarray([1.0], dtype=torch.float32)},
            training_summary={
                "best_epoch": 17,
                "best_validation_weighted_mse": 0.25,
                "history": _training_history(),
            },
            runtime=_training_runtime(),
        ),
    )


def _train_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, dict[str, object], TrainingSplit, TrainingSplit, Path]:
    repository, train, validation, _heldout_value = _fake_run_inputs(tmp_path)
    _patch_split_training(monkeypatch, repository, train, validation)
    runner = densenet2d._benchmark_runner_source_path()
    checkpoint = tmp_path / "outputs" / "seed101.pt"
    result = train_common_retrain(
        repository_path=tmp_path / "repo",
        train_h5=train.provenance["path"],
        validation_h5=validation.provenance["path"],
        seed=101,
        device="cpu",
        checkpoint_out=checkpoint,
        command=["python", "run_densenet2d_common.py", "train"],
        runner_source=runner,
    )
    assert (
        result["checkpoint"]["sha256"]
        == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    )
    return checkpoint, repository, train, validation, Path(str(runner))


def test_train_once_checkpoint_contains_only_train_and_validation_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import inspect

    import torch

    checkpoint, repository, train, validation, runner = _train_checkpoint(
        tmp_path, monkeypatch
    )
    assert "observations_npz" not in inspect.signature(train_common_retrain).parameters
    assert "train_h5" not in inspect.signature(run_common_inference).parameters
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert set(state) == densenet2d._CHECKPOINT_KEYS
    assert state["checkpoint_schema"] == densenet2d.CHECKPOINT_SCHEMA
    assert state["checkpoint_schema_version"] == 2
    assert state["seed"] == 101
    assert "campaign_seeds" not in state["training_config"]
    assert set(state["dataset_identities"]) == {"train", "validation"}
    assert state["dataset_identities"]["train"]["sha256"] == train.provenance["sha256"]
    assert (
        state["dataset_identities"]["validation"]["sha256"]
        == (validation.provenance["sha256"])
    )
    assert state["campaign_observations_accepted_for_training"] is False
    assert state["truth_keys_accepted"] is False
    assert state["contains_truth"] is False
    assert state["contains_observation_campaign"] is False
    assert state["dependency_closure"]["schema"] == (
        "pimsr-mt2dinv-densenet-source-dependency-closure"
    )
    assert state["dependency_closure"]["schema_version"] == 2
    assert set(state["dependency_closure"]["local_source_artifacts"]) == {
        "adapter_source",
        "artifact_guard_source",
        "architecture_source",
        "inversion_dataset_contract_source",
        "materializer_contract_source",
        "runner_source",
        "shared_contract_loader_source",
    }
    assert state["dependency_closure"]["local_source_artifacts"]["runner_source"][
        "sha256"
    ] == (hashlib.sha256(runner.read_bytes()).hexdigest())
    assert state["dependency_closure"]["cli_entrypoint_source_included"] is True
    assert (
        state["dependency_closure"]["required_local_python_source_artifacts_recorded"]
        is True
    )
    assert state["dependency_closure"]["native_binary_environment_complete"] is False
    assert state["source"] == repository
    assert b"generator_seed" not in checkpoint.read_bytes()


def test_dependency_closure_is_incomplete_without_cli_and_rejects_fake_runner(
    tmp_path: Path,
):
    repository, _train, _validation, _heldout_value = _fake_run_inputs(tmp_path)
    direct_artifacts = densenet2d._source_dependency_artifacts(
        repository, runner_source=None
    )
    direct_closure = densenet2d._dependency_closure(direct_artifacts)
    assert direct_closure["cli_entrypoint_source_included"] is False
    assert direct_closure["required_local_python_source_artifacts_recorded"] is False
    assert "runner_source" not in direct_closure["local_source_artifacts"]

    fake_runner = _artifact(tmp_path / "not-the-cli.py", b"not the CLI")["path"]
    with pytest.raises(DenseNet2DAdapterError, match="exact benchmark scripts"):
        densenet2d._source_dependency_artifacts(
            repository, runner_source=str(fake_runner)
        )
    forged_artifacts = dict(direct_artifacts)
    forged_artifacts["runner_source"] = file_artifact_provenance(str(fake_runner))
    with pytest.raises(DenseNet2DAdapterError, match="not the exact benchmark CLI"):
        densenet2d._dependency_closure(forged_artifacts)


@pytest.mark.parametrize(
    "metadata",
    [
        {"truth_path": "operator-only-truth.npz"},
        {"nested": {"withheld_truth": "forbidden"}},
        {"nested": {"operator_manifest": "forbidden"}},
        {"nested": {"sample_id_key": "forbidden"}},
        {"nested": {"generator_seed": 20260824}},
    ],
)
def test_recursive_prescore_metadata_vocabulary_is_rejected(
    metadata: dict[str, object],
):
    with pytest.raises(DenseNet2DAdapterError, match="forbidden pre-score metadata"):
        densenet2d._require_no_prescore_metadata(metadata, role="adversarial metadata")
    densenet2d._require_no_prescore_metadata(
        {
            "truth_keys_accepted": False,
            "contains_truth": False,
            "heldout_truth_available_to_adapter": False,
        },
        role="safe declarations",
    )


def test_training_rejects_malformed_outcome_before_checkpoint_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import torch

    repository, train, validation, _heldout_value = _fake_run_inputs(tmp_path)
    _patch_split_training(monkeypatch, repository, train, validation)
    good_outcome = densenet2d._train_model()
    malformed = (
        (
            TrainingOutcome(
                state_dict={"weight": torch.ones(1, dtype=torch.float64)},
                training_summary=good_outcome.training_summary,
                runtime=good_outcome.runtime,
            ),
            "shape/dtype/layout/device is not exact",
        ),
        (
            TrainingOutcome(
                state_dict=good_outcome.state_dict,
                training_summary=good_outcome.training_summary,
                runtime=_training_runtime(truth_path="operator-only-truth.npz"),
            ),
            "training runtime keys mismatch",
        ),
    )
    for index, (outcome, message) in enumerate(malformed):
        monkeypatch.setattr(
            densenet2d,
            "_train_model",
            lambda *_args, _outcome=outcome, **_kwargs: _outcome,
        )
        checkpoint = tmp_path / f"malformed-{index}" / "checkpoint.pt"
        with pytest.raises(DenseNet2DAdapterError, match=message):
            train_common_retrain(
                repository_path=tmp_path / "repo",
                train_h5=train.provenance["path"],
                validation_h5=validation.provenance["path"],
                seed=101,
                device="cpu",
                checkpoint_out=checkpoint,
                runner_source=densenet2d._benchmark_runner_source_path(),
            )
        assert not checkpoint.exists()
        assert not checkpoint.with_name(checkpoint.name + ".part").exists()


def test_two_campaigns_reuse_identical_checkpoint_and_publish_truth_free_runtime_v2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkpoint, _repository, train, validation, runner = _train_checkpoint(
        tmp_path, monkeypatch
    )
    campaign_one = _heldout(
        tmp_path / "campaign-one.npz",
        payload=b"truth-free campaign one",
        sample_index=np.asarray([10101, 10102]),
    )
    campaign_two = _heldout(
        tmp_path / "campaign-two.npz",
        payload=b"truth-free campaign two",
        sample_index=np.asarray([20201, 20202]),
    )
    campaigns = {
        str(Path(str(campaign_one.provenance["path"])).resolve()): campaign_one,
        str(Path(str(campaign_two.provenance["path"])).resolve()): campaign_two,
    }
    monkeypatch.setattr(
        densenet2d,
        "_load_heldout_observations_snapshot",
        lambda snapshot: campaigns[str(snapshot.path.resolve())],
    )

    def fake_predict(
        heldout: HeldoutObservations,
        _repository: dict[str, object],
        state: dict[str, object],
        **_kwargs: object,
    ) -> tuple[np.ndarray, dict[str, object]]:
        assert set(state) == {"weight"}
        value = 2.0 if heldout is campaign_one else 3.0
        return (
            np.full((2, 64, 48), value, dtype=np.float32),
            _inference_runtime(),
        )

    monkeypatch.setattr(densenet2d, "_predict_from_checkpoint", fake_predict)
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    # Inference must use only checkpoint lineage and must never reopen the
    # target-bearing train/validation artifacts.
    Path(str(train.provenance["path"])).unlink()
    Path(str(validation.provenance["path"])).unlink()
    runtimes: list[dict[str, object]] = []
    for index, heldout in enumerate((campaign_one, campaign_two), start=1):
        prediction = tmp_path / f"campaign-{index}" / "prediction.npz"
        runtime_path = tmp_path / f"campaign-{index}" / "runtime.json"
        runtime = run_common_inference(
            repository_path=tmp_path / "repo",
            checkpoint_path=checkpoint,
            observations_npz=heldout.provenance["path"],
            expected_checkpoint_sha256=checkpoint_sha256,
            expected_observations_sha256=str(heldout.provenance["sha256"]),
            device="cpu",
            predictions_out=prediction,
            runtime_out=runtime_path,
            command=["python", "run_densenet2d_common.py", "infer"],
            runner_source=runner,
        )
        assert json.loads(runtime_path.read_text(encoding="utf-8")) == runtime
        assert runtime["schema"] == densenet2d.RUNTIME_SCHEMA
        assert runtime["schema_version"] == 2
        assert runtime["method_id"] == "mt2dinv_densenet"
        assert runtime["operation"] == "inference_from_reusable_checkpoint"
        assert runtime["seed"] == runtime["training_seed"] == 101
        assert runtime["bindings"]["checkpoint_sha256"] == checkpoint_sha256
        assert runtime["bindings"]["train_sha256"] == train.provenance["sha256"]
        assert runtime["bindings"]["validation_sha256"] == validation.provenance["sha256"]
        assert runtime["bindings"]["observations_sha256"] == heldout.provenance["sha256"]
        assert (
            runtime["bindings"]["prediction_sha256"]
            == hashlib.sha256(prediction.read_bytes()).hexdigest()
        )
        assert runtime["outputs"]["checkpoint"]["sha256"] == checkpoint_sha256
        assert runtime["observation_contract"]["truth_keys_accepted"] is False
        assert runtime["prediction_contract"]["contains_truth"] is False
        assert runtime["heldout_truth_available_to_adapter"] is False
        assert runtime["truth_keys_accepted"] is False
        assert runtime["contains_truth"] is False
        assert runtime["checkpoint_contract"]["safe_load"] == (
            "torch.load(weights_only=True)"
        )
        assert (
            runtime["checkpoint_contract"]["campaign_observations_accepted_for_training"]
            is False
        )
        assert runtime["checkpoint_contract"]["truth_keys_accepted"] is False
        assert runtime["checkpoint_contract"]["contains_truth"] is False
        assert runtime["checkpoint_contract"]["contains_observation_campaign"] is False
        assert set(runtime["dataset_identities"]) == {"train", "validation"}
        assert (
            runtime["dependency_closure"]["closure_sha256"]
            == runtime["bindings"]["dependency_closure_sha256"]
        )
        serialized = json.dumps(runtime, sort_keys=True)
        assert "generator_seed" not in serialized
        assert "hidden_" not in serialized
        with np.load(prediction, allow_pickle=False) as payload:
            assert tuple(payload.files) == densenet2d.PREDICTION_KEYS
            assert payload["observations_sha256"].item() == heldout.provenance["sha256"]
            np.testing.assert_array_equal(payload["sample_index"], heldout.sample_index)
        runtimes.append(runtime)
    assert {runtime["bindings"]["checkpoint_sha256"] for runtime in runtimes} == {
        checkpoint_sha256
    }
    assert {runtime["bindings"]["observations_sha256"] for runtime in runtimes} == {
        campaign_one.provenance["sha256"],
        campaign_two.provenance["sha256"],
    }


def test_inference_rejects_malformed_backend_runtime_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkpoint, _repository, _train, _validation, runner = _train_checkpoint(
        tmp_path, monkeypatch
    )
    heldout = _heldout(tmp_path / "campaign.npz", payload=b"campaign")
    monkeypatch.setattr(
        densenet2d, "_load_heldout_observations_snapshot", lambda _snapshot: heldout
    )
    monkeypatch.setattr(
        densenet2d,
        "_predict_from_checkpoint",
        lambda *_args, **_kwargs: (
            np.full((2, 64, 48), 2.0, np.float32),
            _inference_runtime(operator_manifest="operator-only.json"),
        ),
    )
    prediction = tmp_path / "malformed-inference" / "prediction.npz"
    runtime = tmp_path / "malformed-inference" / "runtime.json"
    with pytest.raises(DenseNet2DAdapterError, match="inference runtime keys mismatch"):
        run_common_inference(
            repository_path=tmp_path / "repo",
            checkpoint_path=checkpoint,
            observations_npz=heldout.provenance["path"],
            device="cpu",
            predictions_out=prediction,
            runtime_out=runtime,
            runner_source=runner,
        )
    assert not prediction.exists()
    assert not runtime.exists()


def test_inference_rejects_checkpoint_hash_and_exact_contract_mutations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import torch

    checkpoint, _repository, _train, _validation, runner = _train_checkpoint(
        tmp_path, monkeypatch
    )
    heldout = _heldout(tmp_path / "campaign.npz", payload=b"campaign")
    monkeypatch.setattr(
        densenet2d, "_load_heldout_observations_snapshot", lambda _snapshot: heldout
    )
    monkeypatch.setattr(
        densenet2d,
        "_predict_from_checkpoint",
        lambda *_args, **_kwargs: (
            np.full((2, 64, 48), 2.0, np.float32),
            _inference_runtime(),
        ),
    )
    with pytest.raises(DenseNet2DAdapterError, match="pinned digest"):
        run_common_inference(
            repository_path=tmp_path / "repo",
            checkpoint_path=checkpoint,
            observations_npz=heldout.provenance["path"],
            expected_checkpoint_sha256="0" * 64,
            device="cpu",
            predictions_out=tmp_path / "bad-hash" / "prediction.npz",
            runtime_out=tmp_path / "bad-hash" / "runtime.json",
            runner_source=runner,
        )

    original = torch.load(checkpoint, map_location="cpu", weights_only=True)

    def extra_key(state: dict[str, object]) -> None:
        state["unexpected"] = True

    def bad_source(state: dict[str, object]) -> None:
        state["source"]["commit"] = "0" * 40

    def bad_seed(state: dict[str, object]) -> None:
        state["seed"] = 999

    def bad_training(state: dict[str, object]) -> None:
        state["training_config"]["epochs"] = 199

    def bad_data(state: dict[str, object]) -> None:
        state["dataset_identities"]["validation"]["sha256"] = state["dataset_identities"][
            "train"
        ]["sha256"]

    def bad_model_state(state: dict[str, object]) -> None:
        state["model_state"]["weight"] = torch.asarray([float("nan")])

    def wrong_model_dtype(state: dict[str, object]) -> None:
        state["model_state"]["weight"] = state["model_state"]["weight"].to(
            dtype=torch.float64
        )

    def wrong_model_layout(state: dict[str, object]) -> None:
        state["model_state"]["weight"] = state["model_state"]["weight"].to_sparse()

    def leaked_generator_metadata(state: dict[str, object]) -> None:
        state["training_runtime"]["generator_seed"] = 12345

    def leaked_truth_metadata(state: dict[str, object]) -> None:
        state["training_runtime"]["truth_path"] = "operator-only-truth.npz"

    cases = (
        (extra_key, "keys mismatch"),
        (bad_source, "source identity"),
        (bad_seed, "seed"),
        (bad_training, "training_config"),
        (bad_data, "identities overlap"),
        (bad_model_state, "non-finite"),
        (wrong_model_dtype, "shape/dtype/layout/device is not exact"),
        (wrong_model_layout, "shape/dtype/layout/device is not exact"),
        (leaked_generator_metadata, "training runtime keys mismatch"),
        (leaked_truth_metadata, "training runtime keys mismatch"),
    )
    for index, (mutate, message) in enumerate(cases):
        state = copy.deepcopy(original)
        mutate(state)
        mutated = tmp_path / f"mutation-{index}.pt"
        torch.save(state, mutated)
        with pytest.raises(DenseNet2DAdapterError, match=message):
            run_common_inference(
                repository_path=tmp_path / "repo",
                checkpoint_path=mutated,
                observations_npz=heldout.provenance["path"],
                device="cpu",
                predictions_out=tmp_path / f"mutation-{index}" / "prediction.npz",
                runtime_out=tmp_path / f"mutation-{index}" / "runtime.json",
                runner_source=runner,
            )


def _unsafe_checkpoint_side_effect(path: str) -> None:
    Path(path).write_text("unsafe load executed", encoding="utf-8")


class _UnsafeCheckpointValue:
    def __init__(self, marker: Path):
        self.marker = marker

    def __reduce__(self) -> tuple[object, tuple[str]]:
        return _unsafe_checkpoint_side_effect, (str(self.marker),)


def test_checkpoint_loader_uses_weights_only_and_never_executes_pickle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import torch

    _checkpoint, repository, _train, _validation, runner = _train_checkpoint(
        tmp_path, monkeypatch
    )
    marker = tmp_path / "unsafe-marker"
    unsafe = tmp_path / "unsafe.pt"
    torch.save({"payload": _UnsafeCheckpointValue(marker)}, unsafe)
    heldout = _heldout(tmp_path / "campaign.npz", payload=b"campaign")
    monkeypatch.setattr(
        densenet2d, "_load_heldout_observations_snapshot", lambda _snapshot: heldout
    )
    with pytest.raises(DenseNet2DAdapterError, match="weights_only=True"):
        run_common_inference(
            repository_path=repository["path"],
            checkpoint_path=unsafe,
            observations_npz=heldout.provenance["path"],
            device="cpu",
            predictions_out=tmp_path / "unsafe" / "prediction.npz",
            runtime_out=tmp_path / "unsafe" / "runtime.json",
            runner_source=runner,
        )
    assert not marker.exists()


def test_train_rejects_noncampaign_seed_before_any_filesystem_work(tmp_path: Path):
    with pytest.raises(ValueError, match="seed must be one of"):
        train_common_retrain(
            repository_path=tmp_path / "missing-repo",
            train_h5=tmp_path / "missing-train",
            validation_h5=tmp_path / "missing-validation",
            seed=999,
            device="cpu",
            checkpoint_out=tmp_path / "checkpoint.pt",
        )
    assert not (tmp_path / "checkpoint.pt").exists()
