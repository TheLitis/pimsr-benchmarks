"""Contracts for the legacy mixed-budget unified diagnostic."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_unified_leaderboard.py"
)


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("test_run_unified_leaderboard", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeLoaded:
    def __init__(self, label: str, contract: SimpleNamespace) -> None:
        self.label = label
        self.contract = contract
        self.checkpoint = {"label": label}
        self.unchanged_checks = 0

    def artifact_provenance(self) -> dict[str, object]:
        return {"checkpoint": {"label": self.label}, "dataset": {"label": "test"}}

    def require_artifacts_unchanged(self) -> None:
        self.unchanged_checks += 1


def test_unified_diagnostic_binds_lineage_inputs_and_hybrid_timing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    script = _load_script()
    dataset = tmp_path / "test.h5"
    dataset.write_bytes(b"dataset-v1")
    checkpoint_1d = tmp_path / "one-d.pt"
    checkpoint_1d.write_bytes(b"one-d-v1")
    emtf = tmp_path / "emtf"
    emtf.mkdir()
    (emtf / "B.xml").write_bytes(b"station-b")
    (emtf / "A.xml").write_bytes(b"station-a")
    output = tmp_path / "unified.json"

    names = (
        "unet-10k",
        "unet-10k-ft",
        "unet-60k",
        "unet-60k-ft",
        "unet-v3-tetm",
        "unet-v3-tetm-ft",
    )
    checkpoint_paths: dict[str, Path] = {}
    for name in names:
        path = tmp_path / f"{name}.pt"
        path.write_bytes(name.encode("ascii"))
        checkpoint_paths[name] = path

    contract = SimpleNamespace(
        frequencies=np.array([1.0]),
        station_x=np.array([0.0]),
        x_grid=np.array([0.0]),
        depth_grid=np.array([100.0]),
    )
    loaded_by_path = {
        str(path): _FakeLoaded(name, contract)
        for name, path in checkpoint_paths.items()
    }
    monkeypatch.setattr(
        script,
        "load_model2d",
        lambda checkpoint, _dataset: loaded_by_path[str(checkpoint)],
    )

    lineage_calls: list[tuple[str, str, Path, list[list[str]], object]] = []

    def fake_lineage(
        adapted,
        *,
        base,
        emtf_dir,
        expected_profiles,
        expected_options=None,
    ):
        lineage_calls.append(
            (
                adapted.label,
                base.label,
                Path(emtf_dir),
                expected_profiles,
                expected_options,
            )
        )
        return {"lineage_sha256": f"digest-{adapted.label}"}

    monkeypatch.setattr(script, "require_finetune2d_lineage", fake_lineage)
    modes = {
        "lr_tm": np.zeros((1, 1)),
        "ph_tm": np.zeros((1, 1)),
        "mask_tm": np.ones((1, 1), dtype=bool),
    }
    assembled_dirs: list[Path] = []

    def fake_assemble(emtf_dir, _freqs, _station_x):
        assembled_dirs.append(Path(emtf_dir))
        return modes

    monkeypatch.setattr(script, "assemble_profile_modes", fake_assemble)
    monkeypatch.setattr(script, "profile_geometry_metadata", lambda _modes: {})
    monkeypatch.setattr(script, "section_nrms_2d", lambda *_args: 1.25)
    section = np.zeros((1, 1))
    monkeypatch.setattr(script, "occam_section", lambda *_args: section.copy())
    monkeypatch.setattr(script, "neural_1d_section", lambda *_args: section.copy())
    monkeypatch.setattr(script, "hybrid_1d_section", lambda *_args: section.copy())
    monkeypatch.setattr(script, "NeuralInverter", lambda _path: object())

    clock = [0.0]
    unet_calls: dict[str, int] = {}

    def fake_unet(loaded, _modes):
        unet_calls[loaded.label] = unet_calls.get(loaded.label, 0) + 1
        if loaded.label == "unet-60k" and unet_calls[loaded.label] == 2:
            clock[0] += 5.0
        return section.copy()

    def fake_refine(initial, *_args, **_kwargs):
        clock[0] += 7.0
        return SimpleNamespace(
            section=np.asarray(initial).copy(),
            wall_time_s=7.0,
            n_iterations=1,
        )

    monkeypatch.setattr(script, "unet_section", fake_unet)
    monkeypatch.setattr(script, "refine_section_2d", fake_refine)
    monkeypatch.setattr(
        script,
        "time",
        SimpleNamespace(perf_counter=lambda: clock[0]),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_unified_leaderboard.py",
            "--test-h5",
            str(dataset),
            "--emtf-dir",
            str(emtf),
            "--ckpt-1d",
            str(checkpoint_1d),
            "--ckpt-10k",
            str(checkpoint_paths["unet-10k"]),
            "--ckpt-10k-ft",
            str(checkpoint_paths["unet-10k-ft"]),
            "--ckpt-60k",
            str(checkpoint_paths["unet-60k"]),
            "--ckpt-60k-ft",
            str(checkpoint_paths["unet-60k-ft"]),
            "--ckpt-v3",
            str(checkpoint_paths["unet-v3-tetm"]),
            "--ckpt-v3-ft",
            str(checkpoint_paths["unet-v3-tetm-ft"]),
            "--out",
            str(output),
            "--allow-mixed-budget-diagnostic",
        ],
    )

    script.main()

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["schema"] == "pimsr-unified-2d-diagnostic"
    assert result["comparison_status"] == "diagnostic_non_comparable"
    assert result["ranking_allowed"] is False
    assert result["headline_claim_allowed"] is False
    assert any("warm start" in reason for reason in result["diagnostic_reasons"])

    assert [(adapted, base) for adapted, base, *_rest in lineage_calls] == [
        ("unet-10k-ft", "unet-10k"),
        ("unet-60k-ft", "unet-60k"),
        ("unet-v3-tetm-ft", "unet-v3-tetm"),
    ]
    for _adapted, _base, lineage_emtf, profiles, options in lineage_calls:
        assert lineage_emtf == emtf.resolve()
        assert profiles == [script.PROFILES["H-YS"]]
        assert options is None
    assert result["adaptation_lineage_sha256"] == {
        "unet-10k-ft": "digest-unet-10k-ft",
        "unet-60k-ft": "digest-unet-60k-ft",
        "unet-v3-tetm-ft": "digest-unet-v3-tetm-ft",
    }

    hybrid = result["methods"]["hybrid2d-gn8"]
    assert hybrid["inversion_observation_modes"] == ["te", "tm"]
    assert hybrid["inversion_observation_stages"] == {
        "warm_start": ["te", "tm"],
        "refinement": ["tm"],
    }
    assert hybrid["wall_time_s"] == pytest.approx(12.0)
    assert hybrid["timing"] == {
        "scope": "warm_start_plus_refinement",
        "warm_start_wall_time_s": 5.0,
        "refinement_wall_time_s": 7.0,
    }

    checkpoint_identity = result["provenance"]["checkpoint_1d"]
    assert checkpoint_identity["path"] == str(checkpoint_1d.resolve())
    assert checkpoint_identity["size_bytes"] == len(b"one-d-v1")
    assert checkpoint_identity["sha256"] == hashlib.sha256(b"one-d-v1").hexdigest()
    xml_identities = result["provenance"]["emtf_xml"]
    assert [Path(item["path"]).name for item in xml_identities] == ["A.xml", "B.xml"]
    for item in xml_identities:
        payload = Path(item["path"]).read_bytes()
        assert item["size_bytes"] == len(payload)
        assert item["sha256"] == hashlib.sha256(payload).hexdigest()
    assert assembled_dirs == [emtf.resolve()]
    assert all(loaded.unchanged_checks == 1 for loaded in loaded_by_path.values())


def test_unified_xml_snapshot_rejects_content_and_set_mutation(tmp_path: Path) -> None:
    script = _load_script()
    emtf = tmp_path / "emtf"
    emtf.mkdir()
    xml = emtf / "A.xml"
    xml.write_bytes(b"station-a")

    root, provenance = script._snapshot_xml_inputs(emtf)
    xml.write_bytes(b"station-mutated")
    with pytest.raises(RuntimeError, match="EMTF XML input changed after"):
        script._require_xml_inputs_unchanged(root, provenance)

    xml.write_bytes(b"station-a")
    root, provenance = script._snapshot_xml_inputs(emtf)
    (emtf / "B.xml").write_bytes(b"station-b")
    with pytest.raises(RuntimeError, match="XML input set changed after snapshot"):
        script._require_xml_inputs_unchanged(root, provenance)
