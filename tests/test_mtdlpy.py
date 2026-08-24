"""Fail-closed tests for the pinned MTDLPy common-retraining adapter."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import types
from pathlib import Path

import h5py
import numpy as np
import pytest

from pimsr_benchmarks import mtdlpy
from pimsr_benchmarks.mtdlpy import (
    HeldoutObservations,
    InferenceOutcome,
    MTDLPyAdapterError,
    TrainingOutcome,
    TrainingSplit,
    infer_common_retrain,
    load_heldout_observations,
    load_training_split,
    resize_bilinear_half_pixel,
    train_common_retrain,
    validate_local_imagenet_weights,
    verify_pinned_repository,
)
from pimsr_benchmarks.runner2d import file_artifact_provenance

_TEST_DISTRIBUTION_VERSIONS = {
    "h5py": "test-h5py",
    "numpy": "test-numpy",
    "pimsr-inversion": "test-pimsr-inversion",
    "torch": "test-torch",
    "torchvision": "test-torchvision",
}


def _pin_dependency_versions(monkeypatch: pytest.MonkeyPatch) -> None:
    def exact_version(distribution: str) -> str:
        if distribution not in _TEST_DISTRIBUTION_VERSIONS:
            raise AssertionError(f"unexpected distribution lookup: {distribution}")
        return _TEST_DISTRIBUTION_VERSIONS[distribution]

    monkeypatch.setattr(mtdlpy.importlib.metadata, "version", exact_version)


def _unsafe_checkpoint_side_effect(path: str) -> None:
    Path(path).write_text("unsafe load executed", encoding="utf-8")


class _UnsafeCheckpointValue:
    def __init__(self, marker: Path):
        self.marker = marker

    def __reduce__(self):
        return (_unsafe_checkpoint_side_effect, (str(self.marker),))


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
        "declared_evaluation_floor_phase_te_degrees": np.full(shape, 2.9, "<f4"),
        "declared_evaluation_floor_log10_rho_tm": np.full(shape, 0.05, "<f4"),
        "declared_evaluation_floor_phase_tm_degrees": np.full(shape, 2.9, "<f4"),
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
    with pytest.raises(MTDLPyAdapterError, match="pinned digest"):
        validate_local_imagenet_weights(weights, digest)


def test_upstream_loader_executes_pinned_source_bytes_not_replaced_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import torch

    source = tmp_path / "dinknet.py"
    source.write_text(
        "from torchvision import models\n"
        "class DinkNet50:\n"
        "    def __init__(self, num_classes, num_channels):\n"
        "        self.encoder = models.resnet50(pretrained=True)\n",
        encoding="utf-8",
    )
    source_artifact = file_artifact_provenance(source)
    marker = tmp_path / "replaced-source-executed.txt"
    replacement = tmp_path / "replacement.py"
    replacement.write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed', encoding='utf-8')\n"
        "from torchvision import models\n"
        "class DinkNet50:\n"
        "    def __init__(self, num_classes, num_channels):\n"
        "        self.encoder = models.resnet50(pretrained=True)\n",
        encoding="utf-8",
    )
    weights_path = tmp_path / "weights.pth"
    torch.save(torch.nn.Linear(2, 1).state_dict(), weights_path)
    weights = file_artifact_provenance(weights_path)

    fake_models = types.SimpleNamespace()

    def fake_resnet50(*, weights: object = None):
        assert weights is None
        return torch.nn.Linear(2, 1)

    fake_models.resnet50 = fake_resnet50
    fake_torchvision = types.ModuleType("torchvision")
    fake_torchvision.models = fake_models
    monkeypatch.setitem(sys.modules, "torchvision", fake_torchvision)

    real_snapshot = mtdlpy._snapshot_regular_file
    swapped = False

    def swap_source_after_snapshot(path: str | Path, **kwargs: object):
        nonlocal swapped
        snapshot = real_snapshot(path, **kwargs)
        if Path(path) == source and not swapped:
            swapped = True
            os.replace(source, tmp_path / "dinknet.original.py")
            os.replace(replacement, source)
        return snapshot

    monkeypatch.setattr(mtdlpy, "_snapshot_regular_file", swap_source_after_snapshot)
    with pytest.raises(RuntimeError, match="changed after it was loaded"):
        mtdlpy._load_upstream_model(
            {"dinknet_source": source_artifact}, weights, torch=torch
        )
    assert not marker.exists()


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


def test_descriptor_snapshot_rejects_rename_between_lstat_and_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    artifact = tmp_path / "artifact.bin"
    original = tmp_path / "artifact.original.bin"
    replacement = tmp_path / "artifact.replacement.bin"
    artifact.write_bytes(b"artifact-a")
    replacement.write_bytes(b"artifact-b")
    real_open = os.open

    def swap_before_open(path: str | bytes | os.PathLike[str], flags: int) -> int:
        if Path(path) == artifact:
            os.replace(artifact, original)
            os.replace(replacement, artifact)
        return real_open(path, flags)

    monkeypatch.setattr(mtdlpy.os, "open", swap_before_open)
    with pytest.raises(MTDLPyAdapterError, match="changed before it was opened"):
        mtdlpy._snapshot_regular_file(artifact, role="adversarial artifact")


def test_h5_and_npz_parsers_use_descriptor_pinned_bytes_after_path_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    training_a = _schema_v2_h5(tmp_path / "training-a.h5", sample_start=100)
    training_b = _schema_v2_h5(tmp_path / "training-b.h5", sample_start=200)
    real_snapshot = mtdlpy._snapshot_regular_file
    training_swapped = False

    def swap_training_after_snapshot(path: str | Path, **kwargs: object):
        nonlocal training_swapped
        snapshot = real_snapshot(path, **kwargs)
        if Path(path) == training_a and not training_swapped:
            training_swapped = True
            os.replace(training_a, tmp_path / "training-a.original.h5")
            os.replace(training_b, training_a)
        return snapshot

    monkeypatch.setattr(mtdlpy, "_snapshot_regular_file", swap_training_after_snapshot)
    split = load_training_split(training_a, role="training dataset")
    assert split.sample_index.tolist() == [100, 101]
    assert (
        split.provenance["sha256"]
        == hashlib.sha256((tmp_path / "training-a.original.h5").read_bytes()).hexdigest()
    )

    observations_a = _write_observations(tmp_path / "observations-a.npz")
    observations_b_arrays = _observation_arrays()
    observations_b_arrays["sample_index"] = np.asarray([7001, 7002], dtype="<i8")
    observations_b = _write_observations(
        tmp_path / "observations-b.npz", observations_b_arrays
    )
    observations_swapped = False

    def swap_observations_after_snapshot(path: str | Path, **kwargs: object):
        nonlocal observations_swapped
        snapshot = real_snapshot(path, **kwargs)
        if Path(path) == observations_a and not observations_swapped:
            observations_swapped = True
            os.replace(observations_a, tmp_path / "observations-a.original.npz")
            os.replace(observations_b, observations_a)
        return snapshot

    monkeypatch.setattr(
        mtdlpy, "_snapshot_regular_file", swap_observations_after_snapshot
    )
    heldout = load_heldout_observations(observations_a)
    assert heldout.sample_index.tolist() == [200, 201]
    assert (
        heldout.provenance["sha256"]
        == hashlib.sha256(
            (tmp_path / "observations-a.original.npz").read_bytes()
        ).hexdigest()
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


def _heldout(
    path: Path,
    *,
    payload: bytes = b"truth-free-observations",
    sample_index: tuple[int, int] = (901, 117),
) -> HeldoutObservations:
    path.write_bytes(payload)
    return HeldoutObservations(
        observations=np.ones((2, 4, 8, 12), dtype=np.float32),
        evaluation_floors=np.ones((2, 4, 8, 12), dtype=np.float32),
        sample_index=np.asarray(sample_index, dtype=np.int64),
        frequencies=np.geomspace(0.01, 100.0, 8),
        station_x=np.linspace(0.0, 11_000.0, 12),
        x_grid=np.linspace(-500.0, 11_500.0, 48),
        depth_grid=np.geomspace(10.0, 10_000.0, 64),
        provenance=file_artifact_provenance(path),
    )


def _training_runtime(**updates: object) -> dict[str, object]:
    runtime: dict[str, object] = {
        "python": "test-python",
        "platform": "test-platform",
        "torch": "test-torch",
        "torchvision": "test-torchvision",
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
        "torch": "test-torch",
        "torchvision": "test-torchvision",
        "torch_cuda_build": None,
        "cuda_available": False,
        "device": "cpu",
        "cuda_device_name": None,
        "peak_cuda_memory_bytes": 0,
        "preprocessing_wall_time_s": 0.1,
        "model_initialization_wall_time_s": 0.2,
        "inference_wall_time_s": 0.5,
        "backend_wall_time_s": 0.8,
    }
    runtime.update(updates)
    return runtime


def test_split_identity_includes_generator_seed(tmp_path: Path):
    train = _split(tmp_path / "train.h5", 10, generator_seed=91)
    same_indices_new_campaign = _split(
        tmp_path / "validation-new-campaign.h5", 10, generator_seed=92
    )

    mtdlpy._require_training_disjoint(train, same_indices_new_campaign)

    overlapping_same_campaign = _split(
        tmp_path / "validation-overlap.h5", 11, generator_seed=91
    )
    with pytest.raises(MTDLPyAdapterError, match="identities overlap"):
        mtdlpy._require_training_disjoint(train, overlapping_same_campaign)


def test_determinism_seeds_python_numpy_and_torch(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[str, object]] = []

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class _Cudnn:
        deterministic = False
        benchmark = True
        allow_tf32 = True

    class _Matmul:
        allow_tf32 = True

    class _Backends:
        cudnn = _Cudnn()

        class cuda:
            matmul = _Matmul()

    class _Torch:
        cuda = _Cuda()
        backends = _Backends()

        @staticmethod
        def manual_seed(seed: int) -> None:
            calls.append(("manual_seed", seed))

        @staticmethod
        def use_deterministic_algorithms(value: bool) -> None:
            calls.append(("deterministic", value))

        @staticmethod
        def set_float32_matmul_precision(value: str) -> None:
            calls.append(("matmul_precision", value))

    python_seeds: list[int] = []
    numpy_seeds: list[int] = []
    monkeypatch.setattr(mtdlpy.random, "seed", python_seeds.append)
    monkeypatch.setattr(mtdlpy.np.random, "seed", numpy_seeds.append)

    mtdlpy._configure_determinism(_Torch(), 101)

    assert python_seeds == [101]
    assert numpy_seeds == [101]
    assert calls == [
        ("manual_seed", 101),
        ("deterministic", True),
        ("matmul_precision", "highest"),
    ]
    assert _Torch.backends.cudnn.deterministic is True
    assert _Torch.backends.cudnn.benchmark is False
    assert _Torch.backends.cudnn.allow_tf32 is False
    assert _Torch.backends.cuda.matmul.allow_tf32 is False


def test_training_recipes_pin_reviewed_and_verbatim_upstream_schedules():
    assert mtdlpy.DEFAULT_RECIPE_ID == "benchmark_reviewed_v1"
    assert mtdlpy.training_recipe(mtdlpy.DEFAULT_RECIPE_ID) == mtdlpy.TrainingRecipe(
        recipe_id="benchmark_reviewed_v1",
        epochs=10,
        batch_size=4,
        learning_rate=1e-4,
        schedule_origin=mtdlpy.REVIEWED_RECIPE.schedule_origin,
    )
    assert mtdlpy.UPSTREAM_CONFIG_RECIPE == mtdlpy.TrainingRecipe(
        recipe_id="upstream_paramconfig_b01f72a_v1",
        epochs=200,
        batch_size=8,
        learning_rate=1e-8,
        schedule_origin=mtdlpy.UPSTREAM_CONFIG_RECIPE.schedule_origin,
    )
    with pytest.raises(ValueError, match="recipe_id must be one of"):
        mtdlpy.training_recipe("test-tuned")


def test_train_once_checkpoint_is_reused_for_two_truth_free_campaigns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import inspect

    import torch

    _pin_dependency_versions(monkeypatch)
    training_parameters = inspect.signature(train_common_retrain).parameters
    assert "observations_npz" not in training_parameters
    assert not any(
        "campaign" in name or "heldout" in name for name in training_parameters
    )
    source = tmp_path / "dinknet.py"
    source.write_bytes(b"source")
    runner = mtdlpy._benchmark_runner_source_path()
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
    campaign_a = _heldout(
        tmp_path / "campaign-a.npz",
        payload=b"truth-free-campaign-a",
        sample_index=(901, 117),
    )
    campaign_b = _heldout(
        tmp_path / "campaign-b.npz",
        payload=b"truth-free-campaign-b",
        sample_index=(310, 804),
    )

    monkeypatch.setattr(mtdlpy, "verify_pinned_repository", lambda _path: repository)
    monkeypatch.setattr(
        mtdlpy,
        "validate_local_imagenet_weights",
        lambda _path, _digest: weights,
    )

    def fake_split(_path: str | Path, *, role: str) -> TrainingSplit:
        return train if "training" in role and "validation" not in role else validation

    monkeypatch.setattr(mtdlpy, "load_training_split", fake_split)
    monkeypatch.setattr(
        mtdlpy,
        "_load_upstream_model",
        lambda *_args, **_kwargs: (torch.nn.Linear(2, 1), object()),
    )
    campaigns = {
        str(Path(campaign_a.provenance["path"]).resolve()): campaign_a,
        str(Path(campaign_b.provenance["path"]).resolve()): campaign_b,
    }
    loaded_campaigns: list[str] = []

    def fake_heldout(path: str | Path) -> HeldoutObservations:
        resolved = str(Path(path).resolve())
        loaded_campaigns.append(resolved)
        return campaigns[resolved]

    monkeypatch.setattr(
        mtdlpy,
        "load_heldout_observations",
        fake_heldout,
    )
    history = [
        {"epoch": epoch, "train_mse": 1.0 / epoch, "validation_mse": 2.0 / epoch}
        for epoch in range(1, mtdlpy.REVIEWED_RECIPE.epochs + 1)
    ]
    monkeypatch.setattr(
        mtdlpy,
        "_train_model",
        lambda *_args, **_kwargs: TrainingOutcome(
            state_dict={
                name: value.clone()
                for name, value in torch.nn.Linear(2, 1).state_dict().items()
            },
            training_summary={
                "best_epoch": mtdlpy.REVIEWED_RECIPE.epochs,
                "best_validation_mse": history[-1]["validation_mse"],
                "history": history,
            },
            runtime=_training_runtime(),
        ),
    )
    monkeypatch.setattr(
        mtdlpy,
        "_infer_model",
        lambda test, *_args, **_kwargs: InferenceOutcome(
            predicted_log10_resistivity=np.full(
                (test.sample_index.size, *mtdlpy.OUTPUT_GRID_SHAPE),
                float(test.sample_index[0]) / 1000.0,
                np.float32,
            ),
            runtime=_inference_runtime(),
        ),
    )

    checkpoint = tmp_path / "outputs" / "seed101.pt"
    training_runtime = tmp_path / "outputs" / "seed101-training.json"
    training = train_common_retrain(
        repository_path=tmp_path / "repo",
        imagenet_weights_path=weights_path,
        imagenet_weights_sha256=mtdlpy.IMAGENET_RESNET50_V1_SHA256,
        train_h5=train.provenance["path"],
        validation_h5=validation.provenance["path"],
        seed=101,
        device="cpu",
        checkpoint_out=checkpoint,
        runtime_out=training_runtime,
        runner_source=runner,
    )
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert training["operation"] == "train_checkpoint_once"
    assert training["outputs"]["checkpoint"]["sha256"] == checkpoint_sha256
    assert "heldout_observations" not in training["source_artifacts"]
    assert training["observation_campaigns_accessed"] is False
    assert training["truth_keys_accepted"] is False
    assert training["contains_truth"] is False
    assert training["dependency_closure"]["schema_version"] == 3
    assert (
        training["dependency_closure"]["required_local_python_source_artifacts_recorded"]
        is True
    )
    assert training["dependency_closure"]["cli_entrypoint_source_included"] is True
    assert training["dependency_closure"]["native_binary_environment_complete"] is False
    assert "complete_for_adapter" not in training["dependency_closure"]
    assert (
        training["bindings"]["runner_source_sha256"]
        == hashlib.sha256(runner.read_bytes()).hexdigest()
    )
    assert loaded_campaigns == []
    loaded_checkpoint, _ = mtdlpy._load_checkpoint_safely(checkpoint)
    assert set(loaded_checkpoint) == mtdlpy.CHECKPOINT_KEYS
    assert loaded_checkpoint["checkpoint_schema_version"] == 1
    assert set(loaded_checkpoint["dataset_identities"]) == {"train", "validation"}
    assert loaded_checkpoint["contains_observation_campaign"] is False
    checkpoint_metadata = {
        key: value for key, value in loaded_checkpoint.items() if key != "model_state"
    }
    assert "generator_seed" not in json.dumps(checkpoint_metadata, sort_keys=True)
    assert campaign_a.provenance["sha256"] not in json.dumps(
        checkpoint_metadata, sort_keys=True
    )
    assert campaign_b.provenance["sha256"] not in json.dumps(
        checkpoint_metadata, sort_keys=True
    )

    published: list[dict[str, object]] = []
    for label, campaign in (("a", campaign_a), ("b", campaign_b)):
        predictions = tmp_path / "outputs" / f"campaign-{label}.npz"
        runtime = tmp_path / "outputs" / f"campaign-{label}.json"
        result = infer_common_retrain(
            repository_path=tmp_path / "repo",
            imagenet_weights_path=weights_path,
            imagenet_weights_sha256=mtdlpy.IMAGENET_RESNET50_V1_SHA256,
            train_h5=train.provenance["path"],
            validation_h5=validation.provenance["path"],
            checkpoint=checkpoint,
            observations_npz=campaign.provenance["path"],
            seed=101,
            device="cpu",
            predictions_out=predictions,
            runtime_out=runtime,
            runner_source=runner,
        )
        assert json.loads(runtime.read_text(encoding="utf-8")) == result
        assert result["operation"] == "inference_from_reusable_checkpoint"
        assert result["schema_version"] == 2
        assert result["outputs"]["checkpoint"]["sha256"] == checkpoint_sha256
        assert result["bindings"]["checkpoint_sha256"] == checkpoint_sha256
        assert result["bindings"]["train_sha256"] == train.provenance["sha256"]
        assert (
            result["bindings"]["validation_sha256"] == (validation.provenance["sha256"])
        )
        assert result["bindings"]["imagenet_weights_sha256"] == weights["sha256"]
        assert (
            result["bindings"]["upstream_source_sha256"]
            == (repository["dinknet_source"]["sha256"])
        )
        assert result["observation_contract"]["truth_keys_accepted"] is False
        assert result["observation_contract"]["contains_truth"] is False
        assert result["prediction_contract"]["truth_keys_accepted"] is False
        assert result["prediction_contract"]["contains_truth"] is False
        assert result["truth_keys_accepted"] is False
        assert result["contains_truth"] is False
        assert "generator_seed" not in runtime.read_text(encoding="utf-8")
        with np.load(predictions, allow_pickle=False) as payload:
            assert tuple(payload.files) == mtdlpy.PREDICTION_KEYS
            assert (
                payload["observations_sha256"].item() == (campaign.provenance["sha256"])
            )
            np.testing.assert_array_equal(payload["sample_index"], campaign.sample_index)
        published.append(result)

    assert (
        published[0]["outputs"]["checkpoint"]["sha256"]
        == (published[1]["outputs"]["checkpoint"]["sha256"])
    )
    assert (
        published[0]["bindings"]["observations_sha256"]
        != (published[1]["bindings"]["observations_sha256"])
    )
    assert (
        published[0]["outputs"]["predictions"]["sha256"]
        != (published[1]["outputs"]["predictions"]["sha256"])
    )
    assert loaded_campaigns == [
        str(Path(campaign_a.provenance["path"]).resolve()),
        str(Path(campaign_b.provenance["path"]).resolve()),
    ]

    monkeypatch.setattr(
        mtdlpy,
        "_infer_model",
        lambda test, *_args, **_kwargs: InferenceOutcome(
            predicted_log10_resistivity=np.ones(
                (test.sample_index.size, *mtdlpy.OUTPUT_GRID_SHAPE),
                dtype=np.float32,
            ),
            runtime=_inference_runtime(operator_manifest="operator.json"),
        ),
    )
    bad_predictions = tmp_path / "outputs" / "malformed-inference.npz"
    bad_runtime = tmp_path / "outputs" / "malformed-inference.json"
    with pytest.raises(MTDLPyAdapterError, match="runtime schema is not exact"):
        infer_common_retrain(
            repository_path=tmp_path / "repo",
            imagenet_weights_path=weights_path,
            imagenet_weights_sha256=mtdlpy.IMAGENET_RESNET50_V1_SHA256,
            train_h5=train.provenance["path"],
            validation_h5=validation.provenance["path"],
            checkpoint=checkpoint,
            observations_npz=campaign_a.provenance["path"],
            seed=101,
            device="cpu",
            predictions_out=bad_predictions,
            runtime_out=bad_runtime,
            runner_source=runner,
        )
    assert not bad_predictions.exists()
    assert not bad_runtime.exists()


def test_checkpoint_loader_rejects_extra_schema_and_symlink(tmp_path: Path):
    import torch

    exact = {key: None for key in mtdlpy.CHECKPOINT_KEYS}
    exact["model_state"] = {}
    exact["truth_keys_accepted"] = False
    exact["contains_truth"] = False
    exact["contains_observation_campaign"] = False
    exact_path = tmp_path / "exact.pt"
    exact_path.write_bytes(mtdlpy._checkpoint_bytes(exact))
    loaded, identity = mtdlpy._load_checkpoint_safely(exact_path)
    assert set(loaded) == mtdlpy.CHECKPOINT_KEYS
    assert identity["sha256"] == hashlib.sha256(exact_path.read_bytes()).hexdigest()

    extra_path = tmp_path / "extra.pt"
    torch.save({**exact, "unexpected": "field"}, extra_path)
    with pytest.raises(MTDLPyAdapterError, match="root schema is not exact"):
        mtdlpy._load_checkpoint_safely(extra_path)

    link = tmp_path / "link.pt"
    try:
        link.symlink_to(exact_path)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(MTDLPyAdapterError, match="non-symlink regular file"):
        mtdlpy._load_checkpoint_safely(link)


def test_checkpoint_restricted_loader_rejects_unsafe_pickle_without_execution(
    tmp_path: Path,
):
    import torch

    marker = tmp_path / "unsafe-load-executed.txt"
    checkpoint = {key: None for key in mtdlpy.CHECKPOINT_KEYS}
    checkpoint["model_state"] = {}
    checkpoint["training_summary"] = _UnsafeCheckpointValue(marker)
    checkpoint["truth_keys_accepted"] = False
    checkpoint["contains_truth"] = False
    checkpoint["contains_observation_campaign"] = False
    path = tmp_path / "unsafe.pt"
    torch.save(checkpoint, path)

    with pytest.raises(MTDLPyAdapterError, match="restricted weights-only loader"):
        mtdlpy._load_checkpoint_safely(path)
    assert not marker.exists()


def test_model_state_validator_requires_exact_finite_tensor_contract():
    import torch

    model = torch.nn.Linear(2, 1)
    valid = {name: value.clone() for name, value in model.state_dict().items()}
    mtdlpy._validate_model_state(model, valid, torch=torch)
    with pytest.raises(MTDLPyAdapterError, match="keys do not exactly match"):
        mtdlpy._validate_model_state(model, {"unexpected": torch.zeros(1)}, torch=torch)
    nonfinite = {name: value.clone() for name, value in valid.items()}
    nonfinite["weight"][0, 0] = torch.nan
    with pytest.raises(MTDLPyAdapterError, match="non-finite"):
        mtdlpy._validate_model_state(model, nonfinite, torch=torch)
    wrong_dtype = {name: value.clone() for name, value in valid.items()}
    wrong_dtype["weight"] = wrong_dtype["weight"].to(dtype=torch.float64)
    with pytest.raises(MTDLPyAdapterError, match="dtype, layout or device"):
        mtdlpy._validate_model_state(model, wrong_dtype, torch=torch)
    wrong_layout = {name: value.clone() for name, value in valid.items()}
    wrong_layout["weight"] = wrong_layout["weight"].to_sparse()
    with pytest.raises(MTDLPyAdapterError, match="dtype, layout or device"):
        mtdlpy._validate_model_state(model, wrong_layout, torch=torch)


def test_dependency_closure_scopes_source_and_native_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _pin_dependency_versions(monkeypatch)
    source = tmp_path / "dinknet.py"
    source.write_bytes(b"source")
    weights_path = tmp_path / "weights.pth"
    weights_path.write_bytes(b"weights")
    repository = {"dinknet_source": file_artifact_provenance(source)}
    weights = file_artifact_provenance(weights_path)

    incomplete = mtdlpy._dependency_closure(
        repository=repository,
        weights=weights,
        runner_source=None,
    )
    assert incomplete["schema_version"] == 3
    assert incomplete["evidence_scope"] == (
        "direct_python_source_artifacts_and_distribution_version_strings"
    )
    assert incomplete["cli_entrypoint_source_included"] is False
    assert incomplete["required_local_python_source_artifacts_recorded"] is False
    assert incomplete["native_binary_environment_complete"] is False
    assert "complete_for_adapter" not in incomplete
    assert "cli_runner" not in incomplete["local_source_artifacts"]

    fake_runner = tmp_path / "run_mtdlpy_common.py"
    fake_runner.write_bytes(b"not the benchmark CLI")
    with pytest.raises(MTDLPyAdapterError, match="exact benchmark scripts"):
        mtdlpy._dependency_closure(
            repository=repository,
            weights=weights,
            runner_source=fake_runner,
        )

    runner = mtdlpy._benchmark_runner_source_path()
    complete_sources = mtdlpy._dependency_closure(
        repository=repository,
        weights=weights,
        runner_source=runner,
    )
    assert complete_sources["cli_entrypoint_source_included"] is True
    assert complete_sources["required_local_python_source_artifacts_recorded"] is True
    assert complete_sources["native_binary_environment_complete"] is False
    assert complete_sources["local_source_artifacts"]["cli_runner"] == (
        file_artifact_provenance(runner)
    )

    def missing_torchvision(distribution: str) -> str:
        if distribution == "torchvision":
            raise mtdlpy.importlib.metadata.PackageNotFoundError(distribution)
        return _TEST_DISTRIBUTION_VERSIONS[distribution]

    monkeypatch.setattr(
        mtdlpy.importlib.metadata,
        "version",
        missing_torchvision,
    )
    with pytest.raises(MTDLPyAdapterError, match="distribution is missing: torchvision"):
        mtdlpy._dependency_closure(
            repository=repository,
            weights=weights,
            runner_source=runner,
        )


@pytest.mark.parametrize(
    "metadata",
    [
        {"truth_path": "truth.npz"},
        {"nested": {"withheld_manifest": "withheld.json"}},
        {"nested": {"operatorManifest": "operator.json"}},
        {"nested": {"sample id key": "opaque"}},
        {"nested": {"generator_seed_value": 1}},
    ],
)
def test_recursive_prescore_metadata_vocabulary_is_rejected(
    metadata: dict[str, object],
):
    with pytest.raises(MTDLPyAdapterError, match="forbidden pre-score metadata"):
        mtdlpy._require_no_prescore_metadata(metadata, where="adversarial metadata")
    mtdlpy._require_no_prescore_metadata(
        {
            "truth_keys_accepted": False,
            "contains_truth": False,
            "heldout_truth_available_to_adapter": False,
        },
        where="safe declarations",
    )


def test_backend_runtime_schemas_are_exact_and_semantically_consistent():
    mtdlpy._validate_backend_runtime(_training_runtime(), phase="training")
    mtdlpy._validate_backend_runtime(_inference_runtime(), phase="inference")
    with pytest.raises(MTDLPyAdapterError, match="schema is not exact"):
        mtdlpy._validate_backend_runtime(
            _training_runtime(operator_manifest="operator.json"), phase="training"
        )
    with pytest.raises(MTDLPyAdapterError, match="must be zero"):
        mtdlpy._validate_backend_runtime(
            _inference_runtime(peak_cuda_memory_bytes=1), phase="inference"
        )
    with pytest.raises(MTDLPyAdapterError, match="finite and non-negative"):
        mtdlpy._validate_backend_runtime(
            _inference_runtime(inference_wall_time_s=float("nan")),
            phase="inference",
        )
    with pytest.raises(MTDLPyAdapterError, match="shorter than its components"):
        mtdlpy._validate_backend_runtime(
            _inference_runtime(backend_wall_time_s=0.1), phase="inference"
        )


def test_metadata_equality_rejects_numeric_type_masquerading():
    assert mtdlpy._strict_metadata_equal({"schema_version": 3}, {"schema_version": 3})
    assert not mtdlpy._strict_metadata_equal(
        {"schema_version": 3.0}, {"schema_version": 3}
    )


def test_training_rejects_bad_outcome_and_bad_restricted_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import torch

    _pin_dependency_versions(monkeypatch)
    source = tmp_path / "dinknet.py"
    source.write_bytes(b"source")
    runner = mtdlpy._benchmark_runner_source_path()
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
    history = [
        {"epoch": epoch, "train_mse": 1.0 / epoch, "validation_mse": 2.0 / epoch}
        for epoch in range(1, mtdlpy.REVIEWED_RECIPE.epochs + 1)
    ]
    runtime = _training_runtime()
    outcome: dict[str, TrainingOutcome] = {
        "value": TrainingOutcome(
            state_dict={},
            training_summary={
                "best_epoch": mtdlpy.REVIEWED_RECIPE.epochs,
                "best_validation_mse": history[-1]["validation_mse"],
                "history": history,
            },
            runtime=runtime,
        )
    }

    monkeypatch.setattr(mtdlpy, "verify_pinned_repository", lambda _path: repository)
    monkeypatch.setattr(
        mtdlpy,
        "validate_local_imagenet_weights",
        lambda _path, _digest: weights,
    )
    monkeypatch.setattr(
        mtdlpy,
        "load_training_split",
        lambda _path, *, role: validation if "validation" in role else train,
    )
    monkeypatch.setattr(
        mtdlpy,
        "_train_model",
        lambda *_args, **_kwargs: outcome["value"],
    )
    monkeypatch.setattr(
        mtdlpy,
        "_load_upstream_model",
        lambda *_args, **_kwargs: (torch.nn.Linear(2, 1), object()),
    )

    bad_state_outputs = (
        tmp_path / "bad-state" / "checkpoint.pt",
        tmp_path / "bad-state" / "runtime.json",
    )
    with pytest.raises(MTDLPyAdapterError, match="keys do not exactly match"):
        train_common_retrain(
            repository_path=tmp_path / "repo",
            imagenet_weights_path=weights_path,
            imagenet_weights_sha256=mtdlpy.IMAGENET_RESNET50_V1_SHA256,
            train_h5=train.provenance["path"],
            validation_h5=validation.provenance["path"],
            seed=101,
            device="cpu",
            checkpoint_out=bad_state_outputs[0],
            runtime_out=bad_state_outputs[1],
            runner_source=runner,
        )
    assert not any(path.exists() for path in bad_state_outputs)
    assert not any(
        path.with_name(path.name + ".part").exists() for path in bad_state_outputs
    )

    valid_state = {
        name: value.clone() for name, value in torch.nn.Linear(2, 1).state_dict().items()
    }
    outcome["value"] = TrainingOutcome(
        state_dict=valid_state,
        training_summary=outcome["value"].training_summary,
        runtime=_training_runtime(withheld_truth="forbidden"),
    )
    bad_runtime_outputs = (
        tmp_path / "bad-runtime" / "checkpoint.pt",
        tmp_path / "bad-runtime" / "runtime.json",
    )
    with pytest.raises(MTDLPyAdapterError, match="runtime schema is not exact"):
        train_common_retrain(
            repository_path=tmp_path / "repo",
            imagenet_weights_path=weights_path,
            imagenet_weights_sha256=mtdlpy.IMAGENET_RESNET50_V1_SHA256,
            train_h5=train.provenance["path"],
            validation_h5=validation.provenance["path"],
            seed=101,
            device="cpu",
            checkpoint_out=bad_runtime_outputs[0],
            runtime_out=bad_runtime_outputs[1],
            runner_source=runner,
        )
    assert not any(path.exists() for path in bad_runtime_outputs)

    outcome["value"] = TrainingOutcome(
        state_dict=valid_state,
        training_summary=outcome["value"].training_summary,
        runtime=runtime,
    )
    real_decode_checkpoint = mtdlpy._decode_checkpoint_snapshot

    def corrupt_roundtrip(snapshot: mtdlpy._ArtifactSnapshot):
        loaded = real_decode_checkpoint(snapshot)
        loaded["recipe_id"] = "corrupted-after-restricted-load"
        return loaded

    monkeypatch.setattr(mtdlpy, "_decode_checkpoint_snapshot", corrupt_roundtrip)
    bad_roundtrip_outputs = (
        tmp_path / "bad-roundtrip" / "checkpoint.pt",
        tmp_path / "bad-roundtrip" / "runtime.json",
    )
    with pytest.raises(MTDLPyAdapterError, match="identity, seed or recipe is wrong"):
        train_common_retrain(
            repository_path=tmp_path / "repo",
            imagenet_weights_path=weights_path,
            imagenet_weights_sha256=mtdlpy.IMAGENET_RESNET50_V1_SHA256,
            train_h5=train.provenance["path"],
            validation_h5=validation.provenance["path"],
            seed=101,
            device="cpu",
            checkpoint_out=bad_roundtrip_outputs[0],
            runtime_out=bad_roundtrip_outputs[1],
            runner_source=runner,
        )
    assert not any(path.exists() for path in bad_roundtrip_outputs)
    assert not any(
        path.with_name(path.name + ".part").exists() for path in bad_roundtrip_outputs
    )


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


def test_publication_rejects_staged_replacement_and_post_link_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    replaced_part = tmp_path / "replaced.part"
    replaced_part.write_bytes(b"validated")
    replaced_snapshot = mtdlpy._snapshot_regular_file(
        replaced_part, role="validated staged artifact"
    )
    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"replacement")
    os.replace(replacement, replaced_part)
    replaced_destination = tmp_path / "replaced.out"
    with pytest.raises(MTDLPyAdapterError, match="changed after validation"):
        mtdlpy._publish_parts(
            (replaced_part,),
            (replaced_destination,),
            expected_snapshots=(replaced_snapshot,),
        )
    assert not replaced_destination.exists()

    mutated_part = tmp_path / "mutated.part"
    mutated_part.write_bytes(b"validated")
    mutated_snapshot = mtdlpy._snapshot_regular_file(
        mutated_part, role="validated staged artifact"
    )
    mutated_destination = tmp_path / "mutated.out"
    real_link = mtdlpy.os.link

    def mutate_after_link(source: Path, destination: Path) -> None:
        real_link(source, destination)
        destination.write_bytes(b"post-link mutation")

    monkeypatch.setattr(mtdlpy.os, "link", mutate_after_link)
    with pytest.raises(MTDLPyAdapterError, match="changed after validation"):
        mtdlpy._publish_parts(
            (mutated_part,),
            (mutated_destination,),
            expected_snapshots=(mutated_snapshot,),
        )
    assert not mutated_destination.exists()


def test_staged_cleanup_never_deletes_a_replacement(
    tmp_path: Path,
):
    part = tmp_path / "owned.part"
    owned_identity = mtdlpy._write_bytes_new(part, b"owned")
    owned_snapshot = mtdlpy._snapshot_regular_file(part, role="owned staged artifact")
    mtdlpy._require_owned_snapshot(
        owned_snapshot, owned_identity, role="owned staged artifact"
    )
    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"external replacement")
    os.replace(replacement, part)

    mtdlpy._remove_owned_part(part, owned_identity)

    assert part.read_bytes() == b"external replacement"


def test_output_symlink_is_rejected_before_path_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    requested = tmp_path / "redirect.pt"
    real_lexists = mtdlpy.os.path.lexists
    real_is_link = mtdlpy._path_is_link
    monkeypatch.setattr(
        mtdlpy.os.path,
        "lexists",
        lambda path: Path(path) == requested or real_lexists(path),
    )
    monkeypatch.setattr(
        mtdlpy,
        "_path_is_link",
        lambda path: Path(path) == requested or real_is_link(path),
    )
    with pytest.raises(MTDLPyAdapterError, match="must not be a symbolic link"):
        mtdlpy._output_paths((requested,), (".pt",))


def test_split_operations_reject_noncampaign_seed_before_work(tmp_path: Path):
    with pytest.raises(ValueError, match="seed must be one of"):
        train_common_retrain(
            repository_path=tmp_path,
            imagenet_weights_path=tmp_path / "weights",
            imagenet_weights_sha256="0" * 64,
            train_h5=tmp_path / "train",
            validation_h5=tmp_path / "validation",
            seed=999,
            device="cpu",
            checkpoint_out=tmp_path / "checkpoint.pt",
            runtime_out=tmp_path / "training.json",
        )
    with pytest.raises(ValueError, match="seed must be one of"):
        infer_common_retrain(
            repository_path=tmp_path,
            imagenet_weights_path=tmp_path / "weights",
            imagenet_weights_sha256="0" * 64,
            train_h5=tmp_path / "train",
            validation_h5=tmp_path / "validation",
            checkpoint=tmp_path / "checkpoint.pt",
            observations_npz=tmp_path / "observations.npz",
            seed=999,
            device="cpu",
            predictions_out=tmp_path / "predictions.npz",
            runtime_out=tmp_path / "inference.json",
        )
