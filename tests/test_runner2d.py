"""Fail-closed 2D runner plumbing and script regression tests."""

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

import pimsr_benchmarks.runner2d as runner2d_module
from pimsr_benchmarks.runner2d import (
    LoadedModel2D,
    checkpoint_adaptation_kind,
    file_artifact_provenance,
    interpolate_periods_in_band,
    prepare_empty_workdir,
    prepare_profile_observation,
    publish_json_no_overwrite,
    publish_npz_no_overwrite,
    require_file_artifact_unchanged,
    require_finetune2d_lineage,
    run_checked,
    stack_dataset_observations,
)


def test_file_provenance_detects_post_read_mutation(tmp_path: Path):
    artifact = tmp_path / "input.bin"
    artifact.write_bytes(b"original")
    provenance = file_artifact_provenance(artifact)

    assert provenance["path"] == str(artifact.resolve())
    assert provenance["size_bytes"] == 8
    require_file_artifact_unchanged(provenance, role="test input")

    artifact.write_bytes(b"mutated")
    with pytest.raises(RuntimeError, match="changed after it was loaded"):
        require_file_artifact_unchanged(provenance, role="test input")


def _lineage_digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def test_finetune_lineage_binds_base_profiles_options_and_emtf(tmp_path: Path):
    emtf = tmp_path / "emtf"
    emtf.mkdir()
    xml = emtf / "A.xml"
    xml.write_bytes(b"station")
    xml_identity = file_artifact_provenance(xml)
    base = LoadedModel2D(
        model=None,
        checkpoint={},
        contract=None,
        checkpoint_path=tmp_path / "base.pt",
        dataset_path=tmp_path / "data.h5",
        checkpoint_sha256="a" * 64,
        checkpoint_size_bytes=10,
        dataset_sha256="b" * 64,
        dataset_size_bytes=20,
    )
    lineage_base = {
        "checkpoint": {
            "artifact_sha256": "a" * 64,
            "artifact_size_bytes": 10,
        },
        "data_h5": {
            "artifact_sha256": "b" * 64,
            "artifact_size_bytes": 20,
        },
        "emtf_sources": {
            "files": [
                {
                    "relative_path": "A.xml",
                    "artifact_sha256": xml_identity["sha256"],
                    "artifact_size_bytes": xml_identity["size_bytes"],
                }
            ]
        },
        "observations": {"profiles": [{"profile_ids": ["A", "B"]}]},
        "training_options": {"steps": 3},
    }
    lineage = {**lineage_base, "lineage_sha256": _lineage_digest(lineage_base)}
    adapted = LoadedModel2D(
        model=None,
        checkpoint={
            "finetune2d": {
                "finetune_schema": "pimsr-finetune-2d",
                "finetune_schema_version": 1,
                "input_lineage": lineage,
            }
        },
        contract=None,
        checkpoint_path=tmp_path / "adapted.pt",
        dataset_path=tmp_path / "data.h5",
        checkpoint_sha256="c" * 64,
        checkpoint_size_bytes=30,
        dataset_sha256="b" * 64,
        dataset_size_bytes=20,
    )

    assert checkpoint_adaptation_kind(adapted.checkpoint) == "profile-adapted"
    require_finetune2d_lineage(
        adapted,
        base=base,
        emtf_dir=emtf,
        expected_profiles=[["A", "B"]],
        expected_options={"steps": 3},
    )
    with pytest.raises(ValueError, match="profile lineage"):
        require_finetune2d_lineage(
            adapted,
            base=base,
            emtf_dir=emtf,
            expected_profiles=[["B", "A"]],
        )

    xml.write_bytes(b"changed")
    with pytest.raises(ValueError, match="EMTF lineage changed"):
        require_finetune2d_lineage(
            adapted,
            base=base,
            emtf_dir=emtf,
            expected_profiles=[["A", "B"]],
        )


def _checkpoint_identities(*, generator_seed: int = 7) -> dict:
    def identity(start: int, end: int, digest: str) -> dict:
        return {
            "artifact_sha256": digest,
            "provenance": {
                "generator_seed": generator_seed,
                "sample_index": {"contiguous_ranges_inclusive": [[start, end]]},
            },
        }

    return {
        "dataset_identities": {
            "train": identity(0, 99, "a" * 64),
            "val": identity(100, 119, "b" * 64),
        }
    }


def test_profile_observation_uses_canonical_four_channel_order():
    shape = (2, 3)
    modes = {
        "lr_te": np.full(shape, 1.0),
        "ph_te": np.full(shape, 90.0),
        "mask_te": np.ones(shape, dtype=bool),
        "lr_tm": np.full(shape, 3.0),
        "ph_tm": np.full(shape, 135.0),
        "mask_tm": np.ones(shape, dtype=bool),
    }
    checkpoint = {
        "stats_mean": np.zeros((1, 4, 1, 1), dtype=np.float32),
        "stats_std": np.ones((1, 4, 1, 1), dtype=np.float32),
    }

    obs = prepare_profile_observation(modes, checkpoint)
    assert obs.shape == (1, 4, 2, 3)
    assert np.all(obs[0, 0] == 1.0)  # TE/Zyx log-rho
    assert np.all(obs[0, 1] == 2.0)  # TE/Zyx phase / 45
    assert np.all(obs[0, 2] == 3.0)  # TM/Zxy log-rho
    assert np.all(obs[0, 3] == 3.0)  # TM/Zxy phase / 45


def test_dataset_observation_uses_canonical_four_channel_order(tmp_path):
    path = tmp_path / "channels.h5"
    with h5py.File(path, "w") as h5:
        for name, value in (
            ("obs_mt_log10_rho", 1.0),
            ("obs_mt_phase", 90.0),
            ("obs_mt_log10_rho_tm", 3.0),
            ("obs_mt_phase_tm", 135.0),
        ):
            h5.create_dataset(name, data=np.full((1, 2, 3), value))
        obs = stack_dataset_observations(h5, slice(None))

    assert obs.shape == (1, 4, 2, 3)
    assert np.all(obs[:, 0] == 1.0)
    assert np.all(obs[:, 1] == 2.0)
    assert np.all(obs[:, 2] == 3.0)
    assert np.all(obs[:, 3] == 3.0)


def test_heldout_split_rejects_training_overlap_and_exact_artifact():
    checkpoint = _checkpoint_identities()
    with pytest.raises(ValueError, match="overlaps checkpoint train"):
        runner2d_module._assert_heldout_disjoint(
            checkpoint,
            generator_seed=7,
            sample_indices=np.array([90, 120], dtype=np.int64),
            artifact_sha256="c" * 64,
        )
    with pytest.raises(ValueError, match="checkpoint val artifact"):
        runner2d_module._assert_heldout_disjoint(
            checkpoint,
            generator_seed=99,
            sample_indices=np.array([500], dtype=np.int64),
            artifact_sha256="b" * 64,
        )


def test_heldout_split_accepts_disjoint_identity_ranges():
    runner2d_module._assert_heldout_disjoint(
        _checkpoint_identities(),
        generator_seed=7,
        sample_indices=np.array([120, 121, 200], dtype=np.int64),
        artifact_sha256="c" * 64,
    )


def test_dataset_observation_integer_selection_preserves_sample_axis(tmp_path):
    path = tmp_path / "channels-int-selection.h5"
    with h5py.File(path, "w") as h5:
        for name, value in (
            ("obs_mt_log10_rho", 1.0),
            ("obs_mt_phase", 90.0),
            ("obs_mt_log10_rho_tm", 3.0),
            ("obs_mt_phase_tm", 135.0),
        ):
            data = np.stack([np.full((2, 3), -value), np.full((2, 3), value)], axis=0)
            h5.create_dataset(name, data=data)
        obs = stack_dataset_observations(h5, 1)

    assert obs.shape == (1, 4, 2, 3)
    assert np.all(obs[0, 0] == 1.0)
    assert np.all(obs[0, 1] == 2.0)
    assert np.all(obs[0, 2] == 3.0)
    assert np.all(obs[0, 3] == 3.0)


@pytest.mark.parametrize("value", [-0.01, 180.0])
def test_dataset_observation_rejects_noncanonical_phase(tmp_path, value):
    path = tmp_path / "bad-phase.h5"
    with h5py.File(path, "w") as h5:
        for name, channel_value in (
            ("obs_mt_log10_rho", 1.0),
            ("obs_mt_phase", value),
            ("obs_mt_log10_rho_tm", 3.0),
            ("obs_mt_phase_tm", 45.0),
        ):
            h5.create_dataset(name, data=np.full((1, 2, 3), channel_value))
        with pytest.raises(ValueError, match=r"\[0, 180\)"):
            stack_dataset_observations(h5, slice(None))


@pytest.mark.parametrize("value", [-0.01, 180.0])
def test_profile_observation_rejects_noncanonical_valid_phase(value):
    shape = (2, 3)
    modes = {
        "lr_te": np.ones(shape),
        "ph_te": np.full(shape, value),
        "mask_te": np.ones(shape, dtype=bool),
        "lr_tm": np.ones(shape),
        "ph_tm": np.full(shape, 45.0),
        "mask_tm": np.ones(shape, dtype=bool),
    }
    checkpoint = {
        "stats_mean": np.zeros((1, 4, 1, 1), dtype=np.float32),
        "stats_std": np.ones((1, 4, 1, 1), dtype=np.float32),
    }

    with pytest.raises(ValueError, match=r"\[0, 180\)"):
        prepare_profile_observation(modes, checkpoint)


def test_descending_period_interpolation_is_sorted_before_numpy_interp():
    result = interpolate_periods_in_band(
        source_periods=np.array([100.0, 10.0, 1.0]),
        source_values=np.array([2.0, 1.0, 0.0]),
        source_mask=np.ones(3, dtype=bool),
        target_periods=np.array([1.0, np.sqrt(10.0), 10.0, 100.0]),
        fill_values=np.full(4, -9.0),
    )
    assert np.allclose(result, [0.0, 0.5, 1.0, 2.0])


def test_workdir_rejects_stale_outputs(tmp_path):
    workdir = prepare_empty_workdir(tmp_path / "solver-run")
    (workdir / "stale.rho").write_text("old")
    with pytest.raises(FileExistsError, match="not empty"):
        prepare_empty_workdir(workdir)


def test_external_solver_failure_is_not_ignored(tmp_path):
    with pytest.raises(RuntimeError, match="status 7"):
        run_checked(
            [sys.executable, "-c", "raise SystemExit(7)"],
            cwd=tmp_path,
            timeout=10,
        )


def test_loaded_model_reports_artifact_hashes_and_detects_post_read_mutation(tmp_path):
    checkpoint_path = tmp_path / "model.pt"
    dataset_path = tmp_path / "heldout.h5"
    checkpoint_path.write_bytes(b"checkpoint-v1")
    dataset_path.write_bytes(b"dataset-v1")
    loaded = LoadedModel2D(
        model=object(),
        checkpoint={},
        contract=object(),
        checkpoint_path=checkpoint_path,
        dataset_path=dataset_path,
        checkpoint_sha256=runner2d_module.hashlib.sha256(b"checkpoint-v1").hexdigest(),
        checkpoint_size_bytes=len(b"checkpoint-v1"),
        dataset_sha256=runner2d_module.hashlib.sha256(b"dataset-v1").hexdigest(),
        dataset_size_bytes=len(b"dataset-v1"),
    )

    provenance = loaded.artifact_provenance()
    assert provenance["checkpoint"]["sha256"] == loaded.checkpoint_sha256
    assert provenance["checkpoint"]["size_bytes"] == len(b"checkpoint-v1")
    assert provenance["dataset"]["sha256"] == loaded.dataset_sha256
    assert provenance["dataset"]["size_bytes"] == len(b"dataset-v1")
    loaded.require_artifacts_unchanged()

    dataset_path.write_bytes(b"dataset-v2")
    with pytest.raises(RuntimeError, match="dataset changed after it was loaded"):
        loaded.require_artifacts_unchanged()


def test_2d_publication_helpers_refuse_overwrite(tmp_path):
    json_path = tmp_path / "result.json"
    npz_path = tmp_path / "section.npz"
    publish_json_no_overwrite({"value": 1}, json_path)
    publish_npz_no_overwrite(npz_path, section=np.ones((2, 3)))

    assert json.loads(json_path.read_text(encoding="utf-8")) == {"value": 1}
    with np.load(npz_path) as bundle:
        assert np.array_equal(bundle["section"], np.ones((2, 3)))
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        publish_json_no_overwrite({"value": 2}, json_path)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        publish_npz_no_overwrite(npz_path, section=np.zeros((2, 3)))


def _load_occam_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_occam2dmt.py"
    spec = importlib.util.spec_from_file_location("run_occam2dmt_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_occam_raster_maps_model_metres_to_profile_extent():
    script = _load_occam_script()
    mesh = {
        "col_widths": np.array([1000.0, 1000.0]),
        "row_heights": np.array([1000.0]),
        "core_left": 0.0,
        "n_pad": 0,
    }
    section = script.params_to_section(
        params=np.array([1.0, 3.0]),
        layers=[1],
        col_spec=[1, 1],
        mesh=mesh,
        x_grid_m=np.array([-12_000.0, 0.0, 12_000.0]),
        profile_x_km=np.array([0.0, 2.0]),
        depth_grid_m=np.array([100.0]),
    )
    assert section.shape == (1, 3)
    assert np.allclose(section[0], [1.0, 2.0, 3.0])


def test_modem_raster_maps_directly_to_cell_centre_depths():
    script = _load_script("run_modem2d")
    log10_rho = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    section = script.to_project_raster(
        ln_rho=log10_rho * np.log(10.0),
        dy=np.array([1000.0, 1000.0]),
        dz=np.array([100.0, 300.0, 600.0]),
        mesh={"st_y": np.array([500.0, 1500.0])},
        x_km=np.array([0.0, 1.0]),
        x_grid=np.array([-12_000.0, 12_000.0]),
        depth_grid=np.array([100.0, 260.0, 700.0, 1000.0]),
    )
    assert section.shape == (4, 2)
    assert np.allclose(section, [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [5.0, 6.0]])


def test_modem_native_error_floor_is_derived_from_common_scoring_policy(tmp_path):
    script = _load_script("run_modem2d")
    occam = _load_occam_script()
    contract = script.MODEM_ERROR_CONTRACT
    scoring = contract["source_scoring_contract"]
    expected_amplitude = 10.0 ** (scoring["log10_apparent_resistivity_std"] / 2.0) - 1.0
    expected_phase = np.tan(np.radians(scoring["phase_std_degrees"]))

    assert script.ERROR_FLOOR == pytest.approx(max(expected_amplitude, expected_phase))
    assert script.ERROR_FLOOR != pytest.approx(0.10)
    assert contract["derivation"].startswith("max(")
    assert occam.RHO_ERR == scoring["log10_apparent_resistivity_std"]
    assert occam.PH_ERR == scoring["phase_std_degrees"]

    path = tmp_path / "modem.dat"
    shape = (1, 1)
    modes = {
        "lr_te": np.full(shape, 2.0),
        "ph_te": np.full(shape, 45.0),
        "mask_te": np.ones(shape, dtype=bool),
        "lr_tm": np.full(shape, 2.0),
        "ph_tm": np.full(shape, 45.0),
        "mask_tm": np.ones(shape, dtype=bool),
    }
    assert script.write_data(path, modes, np.array([1.0]), np.array([0.0])) == 2
    rows = [
        line.split()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith(("#", ">"))
    ]
    for row in rows:
        magnitude = np.hypot(float(row[-3]), float(row[-2]))
        assert float(row[-1]) / magnitude == pytest.approx(script.ERROR_FLOOR)


@pytest.mark.parametrize("script_name", ["run_modem2d", "run_occam2dmt"])
def test_external_runner_snapshots_hash_size_and_detect_mutation(tmp_path, script_name):
    script = _load_script(script_name)
    artifact = tmp_path / "input.bin"
    artifact.write_bytes(b"immutable-input")

    snapshot = script._snapshot_file(artifact, kind="test input")
    assert snapshot.record() == {
        "path": str(artifact.resolve()),
        "sha256": script.hashlib.sha256(b"immutable-input").hexdigest(),
        "size_bytes": len(b"immutable-input"),
    }
    script._require_unchanged(snapshot, kind="test input")

    artifact.write_bytes(b"mutated-input")
    with pytest.raises(RuntimeError, match="changed after it was read"):
        script._require_unchanged(snapshot, kind="test input")


@pytest.mark.parametrize("script_name", ["run_modem2d", "run_occam2dmt"])
def test_external_runner_detects_emtf_input_set_change(tmp_path, script_name):
    script = _load_script(script_name)
    emtf_dir = tmp_path / "emtf"
    emtf_dir.mkdir()
    (emtf_dir / "A.xml").write_text("A", encoding="utf-8")
    snapshots = script._snapshot_xml_inputs(emtf_dir)

    (emtf_dir / "B.xml").write_text("B", encoding="utf-8")
    with pytest.raises(RuntimeError, match="input set changed"):
        script._require_xml_inputs_unchanged(emtf_dir, snapshots)


def test_modem_requires_explicit_convergence_evidence():
    script = _load_script("run_modem2d")
    incomplete = script._execution_contract(
        stdout="iteration 7 rms=0.8",
        stderr="",
        rms_history=[1.2, 0.8],
        final_iteration=7,
        target_rms=1.0,
    )
    assert incomplete["target_reached"] is True
    assert incomplete["convergence_proven"] is False
    assert incomplete["stopping_reason"] == (
        "target_reached_without_explicit_solver_convergence"
    )

    complete = script._execution_contract(
        stdout="iteration 7 rms=0.8\nConvergence achieved",
        stderr="",
        rms_history=[1.2, 0.8],
        final_iteration=7,
        target_rms=1.0,
    )
    assert complete["convergence_proven"] is True
    assert complete["stopping_evidence_line"] == "Convergence achieved"


def _fake_external_modes() -> dict[str, object]:
    shape = (2, 3)
    return {
        "lr_te": np.full(shape, 2.0),
        "ph_te": np.full(shape, 45.0),
        "mask_te": np.ones(shape, dtype=bool),
        "lr_tm": np.full(shape, 2.2),
        "ph_tm": np.full(shape, 55.0),
        "mask_tm": np.ones(shape, dtype=bool),
        "x_km": np.array([0.0, 1.0]),
        "x_model": np.array([0.0, 0.5, 1.0]),
        "geometry_policy": "normalized_model_station_span",
        "profile_azimuth_deg": 90.0,
        "profile_span_m": 1000.0,
        "model_station_span_m": 2000.0,
        "horizontal_compression_factor": 2.0,
        "publishable_physical_geometry": False,
        "station_ids": ["A", "B"],
    }


def _fake_2d_contract() -> SimpleNamespace:
    return SimpleNamespace(
        frequencies=np.array([1.0, 0.1]),
        station_x=np.array([-1000.0, 0.0, 1000.0]),
        x_grid=np.array([-1000.0, 1000.0]),
        depth_grid=np.array([100.0, 1000.0]),
    )


def test_modem_main_publishes_complete_provenance_as_non_comparable(
    tmp_path, monkeypatch
):
    script = _load_script("run_modem2d")
    dataset = tmp_path / "test.h5"
    binary = tmp_path / "Mod2DMT"
    emtf_dir = tmp_path / "emtf"
    output = tmp_path / "modem.json"
    workdir = tmp_path / "work"
    dataset.write_bytes(b"exact-hdf5")
    binary.write_bytes(b"exact-binary")
    emtf_dir.mkdir()
    (emtf_dir / "A.xml").write_bytes(b"station-A")
    (emtf_dir / "B.xml").write_bytes(b"station-B")
    monkeypatch.setattr(script, "PROFILES", {"H-YS": ["A", "B"]})
    monkeypatch.setattr(script, "load_dataset2d", lambda _path: _fake_2d_contract())
    monkeypatch.setattr(script, "assemble_profile_modes", lambda *_a, **_k: _fake_external_modes())

    def fake_run(command, *, cwd, timeout):
        del command, timeout
        (Path(cwd) / "Modular_NLCG_001.rho").write_text(
            "solver-output", encoding="utf-8"
        )
        return SimpleNamespace(
            stdout="iteration 1 rms=0.8\nConvergence achieved",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(script, "run_checked", fake_run)
    monkeypatch.setattr(
        script,
        "read_final_model",
        lambda _wd: (
            np.full((2, 2), np.log(100.0)),
            1,
            np.ones(2),
            np.ones(2),
        ),
    )
    monkeypatch.setattr(
        script,
        "to_project_raster",
        lambda *_a, **_k: np.full((2, 2), 2.0),
    )
    monkeypatch.setattr(script, "section_nrms_2d", lambda *_a, **_k: 0.75)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_modem2d.py",
            "--emtf-dir",
            str(emtf_dir),
            "--binary",
            str(binary),
            "--test-h5",
            str(dataset),
            "--workdir",
            str(workdir),
            "--out",
            str(output),
        ],
    )

    script.main()

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["comparison_status"] == "diagnostic_non_comparable"
    assert result["ranking_allowed"] is False
    assert result["headline_claim_allowed"] is False
    assert result["execution"]["convergence_proven"] is True
    assert result["comparison_contract"]["inverse_observation_geometry"] == (
        "densified_model_grid_pseudo_stations"
    )
    assert result["provenance"]["dataset"]["sha256"] == script.hashlib.sha256(
        b"exact-hdf5"
    ).hexdigest()
    assert len(result["provenance"]["emtf_xml"]) == 2
    assert result["provenance"]["executable"]["size_bytes"] == len(b"exact-binary")
    assert result["provenance"]["command"] == [
        str(binary.resolve()),
        "-I",
        "NLCG",
        "prior.rho",
        "data.dat",
    ]
    assert len(result["provenance"]["generated_solver_inputs"]) == 2
    section = Path(result["section_artifact"]["path"])
    assert section == script._section_output_path(output)
    assert result["section_artifact"]["sha256"] == script.hashlib.sha256(
        section.read_bytes()
    ).hexdigest()
    with np.load(section) as bundle:
        assert np.array_equal(bundle["section_log10_resistivity"], np.full((2, 2), 2.0))


def test_occam_main_publishes_trace_and_non_equivalent_objective(
    tmp_path, monkeypatch
):
    script = _load_script("run_occam2dmt")
    import pimsr_benchmarks.emtf as emtf_module

    dataset = tmp_path / "test.h5"
    binary = tmp_path / "occam2d"
    emtf_dir = tmp_path / "emtf"
    output = tmp_path / "occam.json"
    workdir = tmp_path / "work"
    dataset.write_bytes(b"exact-hdf5")
    binary.write_bytes(b"exact-occam-binary")
    emtf_dir.mkdir()
    (emtf_dir / "A.xml").write_bytes(b"station-A")
    (emtf_dir / "B.xml").write_bytes(b"station-B")
    monkeypatch.setattr(script, "PROFILES", {"H-YS": ["A", "B"]})
    monkeypatch.setattr(script, "load_dataset2d", lambda _path: _fake_2d_contract())
    monkeypatch.setattr(script, "assemble_profile_modes", lambda *_a, **_k: _fake_external_modes())
    monkeypatch.setattr(
        emtf_module,
        "parse_emtf_xml",
        lambda path: SimpleNamespace(station_id=Path(path).stem),
    )
    monkeypatch.setattr(
        emtf_module,
        "resample_station_modes",
        lambda *_a, **_k: {
            "lr_te": np.array([2.0, 2.1]),
            "ph_te": np.array([45.0, 46.0]),
            "mask_te": np.ones(2, dtype=bool),
            "lr_tm": np.array([2.2, 2.3]),
            "ph_tm": np.array([55.0, 56.0]),
            "mask_tm": np.ones(2, dtype=bool),
        },
    )
    monkeypatch.setattr(
        script,
        "build_mesh",
        lambda *_a, **_k: {
            "col_widths": np.array([1000.0]),
            "row_heights": np.array([1000.0]),
            "n_pad": 0,
            "n_core": 1,
            "core_left": 0.0,
            "dx": 1000.0,
        },
    )
    monkeypatch.setattr(script, "build_layers", lambda _mesh: ([1], [1]))

    def fake_run(command, *, cwd, timeout):
        del command, timeout
        (Path(cwd) / "ITER01.iter").write_text(
            "Misfit Value: 0.8\nParam Count: 1\n2.0\n",
            encoding="utf-8",
        )
        return SimpleNamespace(
            stdout="Tolerance satisfied",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(script, "run_checked", fake_run)
    monkeypatch.setattr(
        script,
        "params_to_section",
        lambda *_a, **_k: np.full((2, 2), 2.0),
    )
    monkeypatch.setattr(script, "section_nrms_2d", lambda *_a, **_k: 0.6)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_occam2dmt.py",
            "--emtf-dir",
            str(emtf_dir),
            "--binary",
            str(binary),
            "--test-h5",
            str(dataset),
            "--workdir",
            str(workdir),
            "--out",
            str(output),
        ],
    )

    script.main()

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["comparison_status"] == "diagnostic_non_comparable"
    assert result["ranking_allowed"] is False
    assert result["headline_claim_allowed"] is False
    assert result["execution"]["convergence_proven"] is True
    assert result["execution"]["iteration_trace"][0]["iteration"] == 1
    iteration_artifact = result["execution"]["iteration_trace"][0]["artifact"]
    assert iteration_artifact["sha256"] == script.hashlib.sha256(
        Path(iteration_artifact["path"]).read_bytes()
    ).hexdigest()
    assert result["comparison_contract"][
        "internal_objective_equivalent_to_shared_score"
    ] is False
    assert result["comparison_contract"]["inverse_observation_geometry"] == (
        "native_emtf_stations"
    )
    assert len(result["provenance"]["generated_solver_inputs"]) == 4
    section = Path(result["section_artifact"]["path"])
    assert result["section_artifact"]["sha256"] == script.hashlib.sha256(
        section.read_bytes()
    ).hexdigest()


def test_legacy_neural_nrms_uses_cell_centre_layers_and_periodic_phase(monkeypatch):
    script = _load_script("run_neural_bench")
    captured = {}

    def fake_forward(_rho, thicknesses, _periods):
        captured["thicknesses"] = thicknesses
        return np.ones(2), np.array([-1.0, -1.0])

    monkeypatch.setattr(script, "mt1d_response", fake_forward)
    score = script.mt_nrms(
        log10_rho=np.zeros(3),
        depth_grid=np.array([10.0, 100.0, 1000.0]),
        periods=np.array([1.0, 10.0]),
        obs_log_rho_a=np.zeros(2),
        obs_phase=np.array([179.0, 179.0]),
    )
    expected_edges = np.array([0.0, np.sqrt(1000.0), np.sqrt(100_000.0)])
    assert np.allclose(captured["thicknesses"], np.diff(expected_edges))
    assert score == pytest.approx(0.0)


def test_legacy_neural_synthetic_runner_reads_producer_gravity_key(tmp_path):
    script = _load_script("run_neural_bench")
    path = tmp_path / "producer-compatible.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("obs_mt_log10_rho", data=np.zeros((1, 2)))
        h5.create_dataset("obs_mt_phase", data=np.full((1, 2), 45.0))
        h5.create_dataset("obs_gravity", data=np.zeros((1, 3)))
        h5.create_dataset("target_log10_res", data=np.zeros((1, 4)))
        h5.create_dataset("scenario", data=np.zeros(1, dtype=np.int64))

    class FakeInverter:
        def require_dataset(self, _file):
            return None

        def invert(self, _lr, _ph, _gz):
            return SimpleNamespace(
                log10_rho=np.zeros(4),
                sigma_log10_rho=np.ones(4),
                scenario_probs=np.array([1.0]),
                wall_time_s=0.01,
            )

    result = script.bench_synthetic(FakeInverter(), str(path), 1)
    assert result["n"] == 1
    assert result["scenario_accuracy"] == 1.0
