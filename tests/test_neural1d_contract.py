"""Strict 1D checkpoint and observation input tests for benchmark inference."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
from pimsr_inversion.contracts1d import (
    CHECKPOINT_SCHEMA,
    CHECKPOINT_SCHEMA_VERSION,
    DATASET_IDENTITY_SCHEMA,
    DATASET_IDENTITY_SCHEMA_VERSION,
    Contract1DError,
    Dataset1DContract,
)
from pimsr_inversion.data import NormStats

import pimsr_benchmarks.cli as cli_module
import pimsr_benchmarks.neural as neural_module
from pimsr_benchmarks.cli import run_synthetic
from pimsr_benchmarks.neural import NeuralInverter


def _contract() -> Dataset1DContract:
    source_identities = np.array([[11, 101], [11, 102]], dtype=np.int64)
    return Dataset1DContract(
        periods=np.array([0.1, 1.0]),
        grav_offsets=np.array([-10.0, 10.0]),
        depth_grid=np.array([10.0, 100.0, 1000.0]),
        sensor_parameters_json='{"noise":1}',
        source_geology_sha256="a" * 64,
        source_geology_size_bytes=1024,
        source_identity_summary=Dataset1DContract.summarize_source_identities(
            source_identities
        ),
        source_identities=source_identities,
    )


def _checkpoint(contract: Dataset1DContract) -> dict:
    n_obs = contract.n_observations
    checkpoint = {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_state": {},
        "n_obs": n_obs,
        "n_depth": contract.depth_grid.size,
        "n_scenarios": 5,
        "norm_stats": {"obs_mean": [0.0] * n_obs, "obs_std": [1.0] * n_obs},
        "periods": contract.periods.tolist(),
        "depth_grid": contract.depth_grid.tolist(),
        "data_contract": contract.checkpoint_metadata(),
        "input_contract": contract.input_metadata(),
        "epoch": 0,
    }

    def identity(source_identities: np.ndarray, artifact_digest: str) -> dict:
        source = {
            **Dataset1DContract.summarize_source_identities(source_identities),
            "pairs": source_identities.tolist(),
        }

        def digest(value: object) -> str:
            payload = json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
            return hashlib.sha256(payload).hexdigest()

        contract_sha256 = digest(contract.checkpoint_metadata())
        provenance = {
            "source_identities": source,
            "contract_sha256": contract_sha256,
        }
        base = {
            "identity_schema": DATASET_IDENTITY_SCHEMA,
            "identity_schema_version": DATASET_IDENTITY_SCHEMA_VERSION,
            "artifact_sha256": artifact_digest,
            "artifact_size_bytes": 1024,
            "contract_sha256": contract_sha256,
            "source_identities": source,
            "provenance": provenance,
            "provenance_sha256": digest(provenance),
        }
        return {**base, "identity_sha256": digest(base)}

    assert contract.source_identities is not None
    checkpoint["dataset_identities"] = {
        "train": identity(contract.source_identities, "b" * 64),
        "val": identity(np.array([[12, 201], [12, 202]], dtype=np.int64), "c" * 64),
    }
    return checkpoint


def test_neural_inverter_rejects_legacy_checkpoint(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save({"model_state": {}}, path)

    with pytest.raises(Contract1DError, match="checkpoint_schema"):
        NeuralInverter(path, device="cpu")


def _load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_contract_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cli_and_direct_checkpoint_exporter_reject_legacy_checkpoint(
    tmp_path, monkeypatch
):
    path = tmp_path / "legacy.pt"
    torch.save({"model_state": {}}, path)
    with pytest.raises(Contract1DError, match="checkpoint_schema"):
        run_synthetic(
            SimpleNamespace(
                checkpoint=str(path),
                dataset=str(tmp_path / "missing.h5"),
                n_stations=1,
            )
        )

    script = _load_script("export_real_npz")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_real_npz.py",
            "--emtf-dir",
            str(tmp_path),
            "--checkpoint",
            str(path),
            "--out",
            str(tmp_path / "real.npz"),
        ],
    )
    with pytest.raises(Contract1DError, match="checkpoint_schema"):
        script.main()


def test_real_emtf_export_is_content_addressed_and_no_overwrite(tmp_path, monkeypatch):
    checkpoint = tmp_path / "strict.pt"
    torch.save(_checkpoint(_contract()), checkpoint)
    source = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "emtf"
        / "USArray_IDI15_2008.xml"
    )
    emtf_dir = tmp_path / "emtf"
    emtf_dir.mkdir()
    (emtf_dir / source.name).write_bytes(source.read_bytes())
    output = tmp_path / "real.npz"

    script = _load_script("export_real_npz")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_real_npz.py",
            "--emtf-dir",
            str(emtf_dir),
            "--checkpoint",
            str(checkpoint),
            "--out",
            str(output),
        ],
    )
    script.main()

    manifest_path = output.with_suffix(".npz.manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["artifact_role"] == "fine_tuning_input_not_benchmark_result"
    assert manifest["output_artifact"]["sha256"] == hashlib.sha256(
        output.read_bytes()
    ).hexdigest()
    with np.load(output, allow_pickle=False) as exported:
        assert exported["stations"].tolist() == ["IDI15"]
        assert exported["mask"].dtype == np.dtype(bool)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        script.main()


def test_public_synthetic_cli_fails_closed_on_mixed_observation_budget(monkeypatch):
    monkeypatch.setattr(cli_module, "NeuralInverter", lambda _checkpoint: object())

    with pytest.raises(
        RuntimeError, match="MT plus gravity while Occam consumes MT only"
    ):
        run_synthetic(
            SimpleNamespace(
                checkpoint="strict.pt",
                dataset="heldout.h5",
                n_stations=1,
                allow_mixed_budget_diagnostic=False,
            )
        )


def test_installed_cli_publishes_input_provenance_without_overwrite(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.pt"
    dataset = tmp_path / "heldout.h5"
    output = tmp_path / "diagnostic.json"
    checkpoint.write_bytes(b"checkpoint")
    dataset.write_bytes(b"dataset")
    monkeypatch.setattr(
        cli_module,
        "run_synthetic",
        lambda _args: {
            "schema": "pimsr-1d-mixed-budget-diagnostic",
            "schema_version": 1,
            "comparison_status": "diagnostic_non_comparable",
            "ranking_allowed": False,
        },
    )

    arguments = [
        "synthetic",
        "--checkpoint",
        str(checkpoint),
        "--dataset",
        str(dataset),
        "--out",
        str(output),
        "--allow-mixed-budget-diagnostic",
    ]
    cli_module.main(arguments)
    published = json.loads(output.read_text(encoding="utf-8"))

    assert published["artifacts"]["checkpoint"]["size_bytes"] == len(b"checkpoint")
    assert published["artifacts"]["dataset"]["size_bytes"] == len(b"dataset")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        cli_module.main(arguments)


def test_explicit_mixed_budget_result_is_non_rankable(tmp_path, monkeypatch):
    dataset = tmp_path / "heldout.h5"
    with h5py.File(dataset, "w") as h5:
        h5.create_dataset("obs_mt_log10_rho", data=np.zeros((1, 2)))
        h5.create_dataset("obs_mt_phase", data=np.full((1, 2), 45.0))
        h5.create_dataset("obs_gravity", data=np.zeros((1, 1)))
        h5.create_dataset("target_log10_res", data=np.zeros((1, 3)))
        h5.create_dataset("periods", data=np.array([1.0, 10.0]))
        h5.create_dataset("depth_grid", data=np.array([10.0, 100.0, 1000.0]))

    class FakeInverter:
        def __init__(self, _checkpoint):
            pass

        def require_dataset(self, _h5):
            return None

        def invert(self, _rho, _phase, _gravity):
            return SimpleNamespace(
                log10_rho=np.zeros(3),
                sigma_log10_rho=np.ones(3),
                wall_time_s=0.01,
            )

    monkeypatch.setattr(cli_module, "NeuralInverter", FakeInverter)
    monkeypatch.setattr(
        cli_module,
        "occam1d_invert",
        lambda *_args, **_kwargs: SimpleNamespace(
            profile_on_grid=lambda _depth: np.zeros(3),
            wall_time_s=0.02,
            nrms=1.0,
        ),
    )

    result = run_synthetic(
        SimpleNamespace(
            checkpoint="strict.pt",
            dataset=dataset,
            n_stations=1,
            allow_mixed_budget_diagnostic=True,
        )
    )

    assert result["comparison_status"] == "diagnostic_non_comparable"
    assert result["ranking_allowed"] is False
    assert result["inverse_observation_budget"]["neural"][-1] == "gravity"
    assert "gravity" not in result["inverse_observation_budget"]["occam"]


def test_neural_inverter_accepts_only_versioned_checkpoint(tmp_path, monkeypatch):
    class FakeNet:
        def __init__(self, *, n_obs, n_depth, n_scenarios):
            assert (n_obs, n_depth, n_scenarios) == (6, 3, 5)

        def load_state_dict(self, state):
            assert state == {}

        def to(self, _device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(neural_module, "PimsrNet", FakeNet)
    path = tmp_path / "strict.pt"
    torch.save(_checkpoint(_contract()), path)

    inverter = NeuralInverter(path, device="cpu")

    assert inverter.n_periods == 2
    assert inverter.n_grav == 2
    assert inverter.n_obs == 6


def _packing_inverter() -> NeuralInverter:
    inverter = NeuralInverter.__new__(NeuralInverter)
    inverter.n_periods = 2
    inverter.n_grav = 1
    inverter.stats = NormStats(np.zeros(5), np.ones(5))
    return inverter


@pytest.mark.parametrize("phase", [np.array([-0.1, 45.0]), np.array([45.0, 180.0])])
def test_pack_rejects_noncanonical_phase(phase):
    with pytest.raises(ValueError, match=r"\[0, 180\)"):
        _packing_inverter()._pack(np.array([1.0, 2.0]), phase, np.zeros(1))


def test_pack_requires_paired_missing_mt_and_finite_gravity():
    inverter = _packing_inverter()
    with pytest.raises(ValueError, match="paired NaN"):
        inverter._pack(np.array([np.nan, 2.0]), np.array([45.0, 45.0]), np.zeros(1))
    with pytest.raises(ValueError, match="gravity input must be finite"):
        inverter._pack(np.array([1.0, 2.0]), np.array([45.0, 45.0]), np.array([np.inf]))

    packed = inverter._pack(
        np.array([np.nan, 2.0]),
        np.array([np.nan, 45.0]),
        gravity=None,
    )
    assert np.isfinite(packed).all()
