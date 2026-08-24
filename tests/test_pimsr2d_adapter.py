from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from pimsr_inversion.network2d import PimsrNet2D

from pimsr_benchmarks import pimsr2d_adapter
from pimsr_benchmarks.evaluation2d import load_predictions_2d
from pimsr_benchmarks.pimsr2d_adapter import (
    CHECKPOINT_SCHEMA,
    OBSERVATION_CHANNEL_ORDER,
    OBSERVATION_SCHEMA,
    PREDICTION_SCHEMA,
    RUNTIME_SCHEMA,
    Pimsr2DPublicationError,
    Pimsr2DValidationError,
    load_checkpoint_2d,
    load_observations_2d,
    normalized_observation_tensor,
    run_pimsr2d_inference,
)


def _canonical_sha(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalization_sha(mean: np.ndarray, std: np.ndarray) -> str:
    digest = hashlib.sha256()
    for key, values in (("mean", mean), ("std", std)):
        digest.update(key.encode("ascii"))
        digest.update(
            json.dumps(list(values.shape), sort_keys=True, separators=(",", ":")).encode(
                "ascii"
            )
        )
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def _observation_arrays() -> dict[str, np.ndarray]:
    n, n_frequency, n_station = 2, 4, 4
    shape = (n, n_frequency, n_station)
    return {
        "schema": np.asarray(OBSERVATION_SCHEMA),
        "schema_version": np.asarray(1, dtype="<i8"),
        "sample_index": np.asarray([10, 11], dtype="<i8"),
        "frequency_hz": np.asarray([0.1, 1.0, 10.0, 100.0], dtype="<f8"),
        "station_x_m": np.asarray([-1.0, -0.25, 0.25, 1.0], dtype="<f8"),
        "x_cell_centers_m": np.asarray([-2.0, 0.0, 2.0], dtype="<f8"),
        "depth_cell_centers_m": np.asarray([10.0, 100.0], dtype="<f8"),
        "observation_channel_order": np.asarray(OBSERVATION_CHANNEL_ORDER),
        "observed_log10_rho_te": np.full(shape, 1.0, dtype="<f4"),
        "observed_phase_te_degrees": np.full(shape, 45.0, dtype="<f4"),
        "observed_log10_rho_tm": np.full(shape, 2.0, dtype="<f4"),
        "observed_phase_tm_degrees": np.full(shape, 90.0, dtype="<f4"),
        "declared_evaluation_floor_log10_rho_te": np.full(
            shape, 0.05, dtype="<f4"
        ),
        "declared_evaluation_floor_phase_te_degrees": np.full(
            shape, 2.9, dtype="<f4"
        ),
        "declared_evaluation_floor_log10_rho_tm": np.full(
            shape, 0.05, dtype="<f4"
        ),
        "declared_evaluation_floor_phase_tm_degrees": np.full(
            shape, 2.9, dtype="<f4"
        ),
        "valid_mask": np.ones((n, 4, n_frequency, n_station), dtype=np.bool_),
    }


def _write_observations(
    path: Path, arrays: dict[str, np.ndarray] | None = None
) -> Path:
    np.savez(path, **(_observation_arrays() if arrays is None else arrays))
    return path


def _data_contract() -> dict[str, object]:
    arrays = _observation_arrays()
    return {
        "schema": "pimsr-mt-2d",
        "schema_version": 2,
        "mode_order": ["te", "tm"],
        "impedance_components": ["Zyx", "Zxy"],
        "scenario_order": [
            "background",
            "aquifer",
            "hydrocarbon",
            "salt",
            "geothermal",
        ],
        "phase_convention": "degrees_modulo_180_[0,180)",
        "resistivity_representation": "log10_ohm_m",
        "frequencies_unit": "Hz",
        "station_x_unit": "m",
        "x_grid_unit": "m",
        "depth_grid_unit": "m",
        "phase_unit": "degree",
        "frequencies": arrays["frequency_hz"].tolist(),
        "station_x": arrays["station_x_m"].tolist(),
        "x_grid": arrays["x_cell_centers_m"].tolist(),
        "depth_grid": arrays["depth_cell_centers_m"].tolist(),
    }


def _identity(artifact_digest: str, contract_digest: str) -> dict[str, object]:
    provenance = {"generator_seed": 7, "sample_index": [0, 1]}
    base = {
        "identity_schema": "pimsr-mt-2d-artifact-identity",
        "identity_schema_version": 1,
        "artifact_sha256": artifact_digest,
        "artifact_size_bytes": 123,
        "contract_sha256": contract_digest,
        "provenance": provenance,
        "provenance_sha256": _canonical_sha(provenance),
    }
    return {**base, "identity_sha256": _canonical_sha(base)}


def _checkpoint_state() -> dict[str, object]:
    torch.manual_seed(101)
    model = PimsrNet2D(
        n_freq=4,
        n_stations=4,
        n_depth=2,
        n_x=3,
        n_scenarios=5,
        width=3,
        in_channels=4,
        scen_head="gap",
    )
    mean = np.asarray([1.0, 1.0, 2.0, 2.0], dtype="<f4").reshape(1, 4, 1, 1)
    std = np.ones((1, 4, 1, 1), dtype="<f4")
    contract = _data_contract()
    contract_digest = _canonical_sha(contract)
    model_shape = {
        "n_freq": 4,
        "n_stations": 4,
        "n_depth": 2,
        "n_x": 3,
        "in_channels": 4,
        "width": 3,
    }
    training_config = {
        "epochs": 2,
        "batch_size": 2,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "gradient_clip_norm": 1.0,
        "sigma_warmup": 0,
        "sigma_regularization": 0.0,
        "beta_nll": 0.5,
        "seed": 101,
        "workers": 0,
        "class_weights": [1.0] * 5,
        "optimizer": "torch.optim.AdamW",
        "scheduler": "CosineAnnealingLR",
        "scheduler_t_max": 2,
        "loss": "beta_nll+tv0.05+scenario_ce0.1/v1",
        "validation_loss": "plain_nll+tv0.05+scenario_ce0.1/v1",
        "normalization": "per-channel-train-mean-std/v1",
        "runtime": {"device_type": "cpu", "torch": str(torch.__version__)},
    }
    return {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "checkpoint_schema_version": 1,
        **model_shape,
        "n_scenarios": 5,
        "scen_head": "gap",
        "beta": 0.5,
        "data_contract": contract,
        "dataset_identities": {
            "train": _identity("a" * 64, contract_digest),
            "val": _identity("b" * 64, contract_digest),
        },
        "normalization_sha256": _normalization_sha(mean, std),
        "model_config": {
            "architecture": "pimsr_inversion.PimsrNet2D/v3",
            **model_shape,
            "n_scenarios": 5,
            "scen_head": "gap",
        },
        "training_config": training_config,
        "epoch": 0,
        "best_epoch": 0,
        "best_val_loss": 1.0,
        "model_state": model.state_dict(),
        "optimizer_state": {},
        "scheduler_state": {},
        "history": [
            {"epoch": 0, "train_loss": 1.1, "val_loss": 1.0, "val_rmse": 0.9}
        ],
        "rng_state": {},
        "stats_mean": mean,
        "stats_std": std,
    }


def _write_checkpoint(path: Path, state: dict[str, object] | None = None) -> Path:
    torch.save(_checkpoint_state() if state is None else state, path)
    return path


def _artifacts(tmp_path: Path) -> tuple[Path, Path]:
    return (
        _write_observations(tmp_path / "observations.npz"),
        _write_checkpoint(tmp_path / "checkpoint.pt"),
    )


def test_exact_section2d_channel_and_phase_normalization(tmp_path: Path):
    observations_path, checkpoint_path = _artifacts(tmp_path)
    observations = load_observations_2d(observations_path)
    checkpoint = load_checkpoint_2d(checkpoint_path, observations)

    normalized = normalized_observation_tensor(observations, checkpoint)

    assert normalized.dtype == np.dtype("<f4")
    assert normalized.shape == (2, 4, 4, 4)
    np.testing.assert_array_equal(normalized, np.zeros_like(normalized))


def test_opaque_sample_ids_need_not_be_sorted_and_order_is_preserved(tmp_path: Path):
    arrays = _observation_arrays()
    arrays["sample_index"] = np.asarray([91, 12], dtype="<i8")
    observations_path = _write_observations(tmp_path / "observations.npz", arrays)
    checkpoint_path = _write_checkpoint(tmp_path / "checkpoint.pt")

    result = run_pimsr2d_inference(
        observations_path,
        checkpoint_path,
        tmp_path / "predictions.npz",
        tmp_path / "runtime.json",
        device="cpu",
    )

    predictions = load_predictions_2d(result.prediction_path)
    assert predictions.sample_index.tolist() == [91, 12]


def test_cpu_inference_emits_evaluator_prediction_and_runtime(tmp_path: Path):
    observations_path, checkpoint_path = _artifacts(tmp_path)
    output = tmp_path / "predictions.npz"
    runtime = tmp_path / "runtime.json"
    expected_observations = hashlib.sha256(observations_path.read_bytes()).hexdigest()
    expected_checkpoint = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()

    result = run_pimsr2d_inference(
        observations_path,
        checkpoint_path,
        output,
        runtime,
        expected_observations_sha256=expected_observations,
        expected_checkpoint_sha256=expected_checkpoint,
        batch_size=1,
        device="cpu",
    )

    predictions = load_predictions_2d(output, expected_sha256=result.prediction_sha256)
    assert predictions.sample_index.tolist() == [10, 11]
    assert predictions.log10_resistivity.shape == (2, 2, 3)
    assert np.isfinite(predictions.log10_resistivity).all()
    report = json.loads(runtime.read_text(encoding="utf-8"))
    assert report["schema"] == RUNTIME_SCHEMA
    assert report["schema_version"] == 3
    assert report["operation"] == "inference_from_reusable_checkpoint"
    assert report["ranking_allowed"] is False
    assert report["truth_keys_accepted"] is False
    assert report["contains_truth"] is False
    assert report["heldout_truth_available_to_adapter"] is False
    assert report["training_seed"] == 101
    assert report["adapter_source"]["sha256"] == hashlib.sha256(
        Path(pimsr2d_adapter.__file__).read_bytes()
    ).hexdigest()
    assert report["inputs"]["observations"]["sha256"] == expected_observations
    assert report["inputs"]["observations"]["schema_version"] == 1
    assert report["inputs"]["checkpoint"]["sha256"] == expected_checkpoint
    assert report["output"]["sha256"] == result.prediction_sha256
    assert report["output"]["schema_version"] == 2
    assert report["training_contract"]["training_config"]["seed"] == 101
    assert report["training_contract"]["train_dataset"]["sha256"] == "a" * 64
    assert report["training_contract"]["validation_dataset"]["sha256"] == "b" * 64
    assert report["checkpoint_contract"]["contains_observation_campaign"] is False
    assert report["checkpoint_contract"]["safe_load"] == "torch.load(weights_only=True)"
    assert report["observation_contract"]["truth_keys_accepted"] is False
    assert report["prediction_contract"]["contains_truth"] is False
    assert report["execution"]["device_resolved"] == "cpu"
    assert report["execution"]["precision"] == "float32"
    assert report["execution"]["peak_cuda_memory_bytes"] is None
    assert report["source"]["repository_checked"] is False
    with np.load(output, allow_pickle=False) as archive:
        assert tuple(archive.files) == (
            "schema",
            "schema_version",
            "observations_sha256",
            "sample_index",
            "x_cell_centers_m",
            "depth_cell_centers_m",
            "predicted_log10_resistivity",
        )
        assert archive["schema"].item() == PREDICTION_SCHEMA
        assert archive["schema_version"].shape == ()
        assert archive["schema_version"].dtype == np.dtype("<i8")
        assert int(archive["schema_version"]) == 2
        observation_binding = archive["observations_sha256"]
        assert observation_binding.shape == ()
        assert observation_binding.dtype == np.dtype("<U64")
        assert observation_binding.item() == expected_observations
        assert archive["x_cell_centers_m"].dtype == np.dtype("<f8")
        assert archive["depth_cell_centers_m"].dtype == np.dtype("<f8")
        np.testing.assert_array_equal(
            archive["x_cell_centers_m"], _observation_arrays()["x_cell_centers_m"]
        )
        np.testing.assert_array_equal(
            archive["depth_cell_centers_m"],
            _observation_arrays()["depth_cell_centers_m"],
        )
    with zipfile.ZipFile(output, "r") as archive:
        assert [member.filename for member in archive.infolist()] == [
            "schema.npy",
            "schema_version.npy",
            "observations_sha256.npy",
            "sample_index.npy",
            "x_cell_centers_m.npy",
            "depth_cell_centers_m.npy",
            "predicted_log10_resistivity.npy",
        ]
        assert all(
            member.compress_type == zipfile.ZIP_STORED
            for member in archive.infolist()
        )


def test_prediction_bytes_are_deterministic(tmp_path: Path):
    observations_path, checkpoint_path = _artifacts(tmp_path)
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"

    first = run_pimsr2d_inference(
        observations_path,
        checkpoint_path,
        first_root / "predictions.npz",
        first_root / "runtime.json",
        batch_size=2,
        device="cpu",
    )
    second = run_pimsr2d_inference(
        observations_path,
        checkpoint_path,
        second_root / "predictions.npz",
        second_root / "runtime.json",
        batch_size=2,
        device="cpu",
    )

    assert first.prediction_sha256 == second.prediction_sha256
    assert first.prediction_path.read_bytes() == second.prediction_path.read_bytes()


def test_observation_contract_rejects_leakage_extra_key_and_bad_schema(tmp_path: Path):
    arrays = _observation_arrays()
    arrays["truth_log10_resistivity"] = np.zeros((2, 2, 3), dtype="<f4")
    leakage = _write_observations(tmp_path / "leakage.npz", arrays)
    with pytest.raises(Pimsr2DValidationError, match="members mismatch"):
        load_observations_2d(leakage)

    arrays = _observation_arrays()
    arrays["schema"] = np.asarray("legacy-observations")
    legacy = _write_observations(tmp_path / "legacy.npz", arrays)
    with pytest.raises(Pimsr2DValidationError, match="schema must be"):
        load_observations_2d(legacy)


def test_hash_pins_reject_tampering(tmp_path: Path):
    observations_path, checkpoint_path = _artifacts(tmp_path)
    with pytest.raises(Pimsr2DValidationError, match="pinned digest"):
        load_observations_2d(observations_path, expected_sha256="0" * 64)
    observations = load_observations_2d(observations_path)
    with pytest.raises(Pimsr2DValidationError, match="pinned digest"):
        load_checkpoint_2d(
            checkpoint_path,
            observations,
            expected_sha256="0" * 64,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda state: state.update(checkpoint_schema="legacy"), "legacy or unsupported"),
        (lambda state: state.update(stats_std=np.zeros((1, 4, 1, 1), dtype="<f4")), "strictly positive"),
        (lambda state: state.update(extra_metadata=True), "keys mismatch"),
    ],
)
def test_checkpoint_rejects_legacy_bad_stats_and_extra_keys(
    tmp_path: Path, mutation, message: str
):
    observations_path = _write_observations(tmp_path / "observations.npz")
    state = _checkpoint_state()
    mutation(state)
    checkpoint_path = _write_checkpoint(tmp_path / "bad.pt", state)
    observations = load_observations_2d(observations_path)

    with pytest.raises(Pimsr2DValidationError, match=message):
        load_checkpoint_2d(checkpoint_path, observations)


def test_no_overwrite_and_stale_part_fail_before_inference(tmp_path: Path):
    observations_path, checkpoint_path = _artifacts(tmp_path)
    output = tmp_path / "predictions.npz"
    runtime = tmp_path / "runtime.json"
    output.write_bytes(b"owned")

    with pytest.raises(Pimsr2DPublicationError, match="refusing to overwrite"):
        run_pimsr2d_inference(
            observations_path,
            checkpoint_path,
            output,
            runtime,
            device="cpu",
        )
    assert output.read_bytes() == b"owned"
    assert not runtime.exists()

    output.unlink()
    stale = runtime.with_name(runtime.name + ".part")
    stale.write_bytes(b"unfinished")
    with pytest.raises(Pimsr2DPublicationError, match="stale partial"):
        run_pimsr2d_inference(
            observations_path,
            checkpoint_path,
            output,
            runtime,
            device="cpu",
        )
    assert stale.read_bytes() == b"unfinished"
    assert not output.exists()


def test_publication_rolls_back_first_artifact_on_base_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    parts = [tmp_path / "prediction.part", tmp_path / "runtime.part"]
    destinations = [tmp_path / "prediction.npz", tmp_path / "runtime.json"]
    for part in parts:
        part.write_bytes(b"staged")
    real_link = pimsr2d_adapter.os.link
    calls = 0

    def interrupted_link(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        real_link(source, destination)

    monkeypatch.setattr(pimsr2d_adapter.os, "link", interrupted_link)

    with pytest.raises(KeyboardInterrupt):
        pimsr2d_adapter._publish(parts, destinations)

    assert all(part.exists() for part in parts)
    assert not any(destination.exists() for destination in destinations)


def test_publication_rolls_back_when_link_succeeds_then_interrupts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    parts = [tmp_path / "prediction.part", tmp_path / "runtime.part"]
    destinations = [tmp_path / "prediction.npz", tmp_path / "runtime.json"]
    for part in parts:
        part.write_bytes(b"staged")
    real_link = pimsr2d_adapter.os.link

    def interrupted_link(source: Path, destination: Path) -> None:
        real_link(source, destination)
        raise KeyboardInterrupt

    monkeypatch.setattr(pimsr2d_adapter.os, "link", interrupted_link)

    with pytest.raises(KeyboardInterrupt):
        pimsr2d_adapter._publish(parts, destinations)

    assert all(part.exists() for part in parts)
    assert not any(destination.exists() for destination in destinations)
