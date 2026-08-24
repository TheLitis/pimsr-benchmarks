"""Fail-closed contracts for the legacy 1D standalone diagnostics."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_script(name: str) -> ModuleType:
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint-v1")
    dataset = tmp_path / "heldout.h5"
    dataset.write_bytes(b"heldout-v1")
    emtf = tmp_path / "emtf"
    emtf.mkdir()
    (emtf / "B.xml").write_bytes(b"station-b")
    (emtf / "A.xml").write_bytes(b"station-a")
    out = tmp_path / "out"
    return checkpoint, dataset, emtf, out


def _argv(
    script_name: str,
    checkpoint: Path,
    dataset: Path,
    emtf: Path,
    out: Path,
) -> list[str]:
    return [
        f"{script_name}.py",
        "--checkpoint",
        str(checkpoint),
        "--test-h5",
        str(dataset),
        "--emtf-dir",
        str(emtf),
        "--out-dir",
        str(out),
        "--n",
        "1",
    ]


@pytest.mark.parametrize(
    ("script_name", "prefix", "synthetic_metrics"),
    [
        (
            "run_neural_bench",
            "neural",
            {"method": "pimsr-neural", "n": 1, "rmse_log10_res": {"mean": 0.1}},
        ),
        (
            "run_hybrid_bench",
            "hybrid",
            {"method": "hybrid", "n": 1, "rmse_log10_res": {"mean": 0.1}},
        ),
    ],
)
def test_legacy_outputs_are_nonrankable_bound_and_no_overwrite(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    script_name: str,
    prefix: str,
    synthetic_metrics: dict[str, object],
) -> None:
    script = _load_script(script_name)
    checkpoint, dataset, emtf, out = _inputs(tmp_path)
    xml_paths_seen: list[Path] = []

    monkeypatch.setattr(script, "NeuralInverter", lambda _path: object())
    monkeypatch.setattr(
        script,
        "bench_synthetic",
        lambda _inverter, _dataset, _n: synthetic_metrics,
    )

    def fake_real(_inverter, _emtf_dir, *, xml_paths):
        xml_paths_seen.extend(Path(path) for path in xml_paths)
        return {
            "method": prefix,
            "nrms": {"mean": 0.2},
            "n_stations": len(xml_paths),
        }

    monkeypatch.setattr(script, "bench_real", fake_real)
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(script_name, checkpoint, dataset, emtf, out),
    )

    script.main()

    synthetic_path = out / f"{prefix}_synthetic.json"
    real_path = out / f"{prefix}_real.json"
    synthetic = json.loads(synthetic_path.read_text(encoding="utf-8"))
    real = json.loads(real_path.read_text(encoding="utf-8"))

    assert synthetic["schema"] == script.SCHEMA
    assert synthetic["schema_version"] == 1
    assert synthetic["comparison_status"] == "diagnostic_non_comparable"
    assert synthetic["ranking_allowed"] is False
    assert synthetic["headline_claim_allowed"] is False
    assert "gravity" in " ".join(synthetic["diagnostic_reasons"])
    assert synthetic["inverse_observation_budget"]["equal"] is False

    checkpoint_identity = synthetic["provenance"]["checkpoint"]
    assert checkpoint_identity["path"] == str(checkpoint.resolve())
    assert checkpoint_identity["size_bytes"] == len(b"checkpoint-v1")
    assert checkpoint_identity["sha256"] == hashlib.sha256(
        b"checkpoint-v1"
    ).hexdigest()
    dataset_identity = synthetic["provenance"]["dataset"]
    assert dataset_identity["path"] == str(dataset.resolve())
    assert dataset_identity["size_bytes"] == len(b"heldout-v1")
    assert dataset_identity["sha256"] == hashlib.sha256(b"heldout-v1").hexdigest()

    assert real["comparison_status"] == "diagnostic_non_comparable"
    assert real["ranking_allowed"] is False
    xml_identities = real["provenance"]["emtf_xml"]
    assert [Path(item["path"]).name for item in xml_identities] == ["A.xml", "B.xml"]
    assert xml_paths_seen == [Path(item["path"]) for item in xml_identities]
    for item in xml_identities:
        payload = Path(item["path"]).read_bytes()
        assert item["size_bytes"] == len(payload)
        assert item["sha256"] == hashlib.sha256(payload).hexdigest()

    before = (synthetic_path.read_bytes(), real_path.read_bytes())
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        script.main()
    assert (synthetic_path.read_bytes(), real_path.read_bytes()) == before


@pytest.mark.parametrize("script_name", ["run_neural_bench", "run_hybrid_bench"])
def test_legacy_runner_rejects_checkpoint_mutation_after_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    script_name: str,
) -> None:
    script = _load_script(script_name)
    checkpoint, dataset, emtf, out = _inputs(tmp_path)

    def mutate_checkpoint(path):
        Path(path).write_bytes(b"checkpoint-mutated")
        return object()

    monkeypatch.setattr(script, "NeuralInverter", mutate_checkpoint)
    monkeypatch.setattr(
        script,
        "bench_synthetic",
        lambda _inverter, _dataset, _n: {"rmse_log10_res": {"mean": 0.1}},
    )
    monkeypatch.setattr(
        script,
        "bench_real",
        lambda _inverter, _emtf_dir, *, xml_paths: {"nrms": {"mean": 0.2}},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(script_name, checkpoint, dataset, emtf, out),
    )

    with pytest.raises(RuntimeError, match="neural checkpoint changed after"):
        script.main()


@pytest.mark.parametrize("script_name", ["run_neural_bench", "run_hybrid_bench"])
def test_legacy_runner_rejects_dataset_mutation_after_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    script_name: str,
) -> None:
    script = _load_script(script_name)
    checkpoint, dataset, emtf, out = _inputs(tmp_path)
    monkeypatch.setattr(script, "NeuralInverter", lambda _path: object())

    def mutate_dataset(_inverter, path, _n):
        Path(path).write_bytes(b"heldout-mutated")
        return {"rmse_log10_res": {"mean": 0.1}}

    monkeypatch.setattr(script, "bench_synthetic", mutate_dataset)
    monkeypatch.setattr(
        script,
        "bench_real",
        lambda _inverter, _emtf_dir, *, xml_paths: {"nrms": {"mean": 0.2}},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(script_name, checkpoint, dataset, emtf, out),
    )

    with pytest.raises(RuntimeError, match="synthetic dataset changed after"):
        script.main()
    assert not out.joinpath(
        "neural_synthetic.json"
        if script_name == "run_neural_bench"
        else "hybrid_synthetic.json"
    ).exists()


@pytest.mark.parametrize("script_name", ["run_neural_bench", "run_hybrid_bench"])
def test_legacy_runner_rejects_xml_mutation_after_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    script_name: str,
) -> None:
    script = _load_script(script_name)
    checkpoint, dataset, emtf, out = _inputs(tmp_path)
    monkeypatch.setattr(script, "NeuralInverter", lambda _path: object())
    monkeypatch.setattr(
        script,
        "bench_synthetic",
        lambda _inverter, _dataset, _n: {"rmse_log10_res": {"mean": 0.1}},
    )

    def mutate_xml(_inverter, _emtf_dir, *, xml_paths):
        Path(xml_paths[0]).write_bytes(b"station-mutated")
        return {"nrms": {"mean": 0.2}}

    monkeypatch.setattr(script, "bench_real", mutate_xml)
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(script_name, checkpoint, dataset, emtf, out),
    )

    with pytest.raises(RuntimeError, match="EMTF XML input changed after"):
        script.main()
    assert not out.joinpath(
        "neural_real.json"
        if script_name == "run_neural_bench"
        else "hybrid_real.json"
    ).exists()
