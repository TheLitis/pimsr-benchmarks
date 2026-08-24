from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from pimsr_benchmarks import dataset_lineage2d as lineage

_SEED = 20260820
_REMOTE = "https://github.com/TheLitis/example.git"
_SENSOR_PARAMETERS = {
    "application_order": "station_major_te_then_tm",
    "sensor_model": {
        "distort_lag1": 0.46,
        "distort_log10rho_hi": 0.25,
        "distort_log10rho_lo": 0.02,
        "distort_phase_scale": 40.0,
        "grav_drift_mgal": 0.05,
        "grav_white_mgal": 0.03,
        "mt_dead_band_extra": 0.02,
        "mt_phase_floor_deg": 1.0,
        "mt_rel_floor": 0.03,
        "static_shift_sigma": 0.15,
    },
    "te_overrides": None,
    "tm_overrides": {
        "distort_hi": {"distribution": "log_uniform", "high": 0.45, "low": 0.25},
        "shift_sigma": {"distribution": "uniform", "high": 0.32, "low": 0.15},
    },
}
_SOFTWARE_VERSIONS = {
    "discretize": "0.test",
    "h5py": "0.test",
    "numpy": "0.test",
    "pimsr_forward": "0.test",
    "pimsr_geogen": "0.test",
    "simpeg": "0.test",
}


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sample_arrays(indices: np.ndarray) -> dict[str, np.ndarray]:
    n = len(indices)
    base = indices.astype(np.float32)[:, None, None]
    shape = (n, 2, 2)
    surface = np.broadcast_to(base, shape).copy()
    target = np.broadcast_to(base, (n, 2, 3)).copy()
    return {
        "obs_mt_log10_rho": surface + np.float32(0.1),
        "obs_mt_phase": surface + np.float32(10.0),
        "clean_mt_log10_rho": surface + np.float32(0.2),
        "clean_mt_phase": surface + np.float32(20.0),
        "target_log10_res": target + np.float32(1.0),
        "scenario": (indices % 5).astype(np.int32),
        "has_fault": (indices % 2).astype(np.uint8),
        "obs_mt_log10_rho_tm": surface + np.float32(0.3),
        "obs_mt_phase_tm": surface + np.float32(30.0),
        "clean_mt_log10_rho_tm": surface + np.float32(0.4),
        "clean_mt_phase_tm": surface + np.float32(40.0),
        "sample_index": indices.astype(np.int64),
    }


def _write_dataset(
    path: Path,
    *,
    start: int,
    count: int,
    source_shard_count: int,
) -> None:
    indices = np.arange(start, start + count, dtype=np.int64)
    arrays = _sample_arrays(indices)
    with h5py.File(path, "w") as h5:
        attrs: dict[str, object] = {
            "schema": "pimsr-mt-2d",
            "schema_version": np.int64(2),
            "phase_convention": "degrees_modulo_180_[0,180)",
            "resistivity_representation": "log10_ohm_m",
            "frequencies_unit": "Hz",
            "station_x_unit": "m",
            "x_grid_unit": "m",
            "depth_grid_unit": "m",
            "phase_unit": "degree",
            "generation_contract": "pimsr-geogen.SectionGenerator/default-grid/v1",
            "generator_rng": "numpy.default_rng([generator_seed,2,sample_index])",
            "forward_contract": "pimsr-forward.MT2DForward/default-mesh/v2",
            "sensor_contract": ("pimsr-forward.SensorModel/mt-noise+tm-severity-v5/v1"),
            "sensor_rng": "numpy.default_rng([generator_seed,3,sample_index])",
            "mode_order": np.asarray(["te", "tm"], dtype="S2"),
            "impedance_components": np.asarray(["Zyx", "Zxy"], dtype="S3"),
            "scenario_order": np.asarray(
                ["background", "aquifer", "hydrocarbon", "salt", "geothermal"],
                dtype="S16",
            ),
            "generator_seed": np.int64(_SEED),
            "generation_start_index": np.int64(start),
            "expected_row_count": np.int64(count),
            "source_shard_count": np.int64(source_shard_count),
            "generation_complete": np.uint8(1),
            "sensor_parameters_json": json.dumps(
                _SENSOR_PARAMETERS, sort_keys=True, separators=(",", ":")
            ),
            "software_versions_json": json.dumps(
                _SOFTWARE_VERSIONS, sort_keys=True, separators=(",", ":")
            ),
        }
        for key, value in attrs.items():
            h5.attrs[key] = value
        h5.create_dataset("frequencies", data=np.asarray([1.0, 10.0], dtype=np.float64))
        h5.create_dataset("station_x", data=np.asarray([-1.0, 1.0], dtype=np.float64))
        h5.create_dataset("x_grid", data=np.asarray([-2.0, 0.0, 2.0], dtype=np.float64))
        h5.create_dataset("depth_grid", data=np.asarray([1.0, 3.0], dtype=np.float64))
        for key, value in arrays.items():
            h5.create_dataset(key, data=value)
        for key, unit in (
            ("frequencies", "Hz"),
            ("station_x", "m"),
            ("x_grid", "m"),
            ("depth_grid", "m"),
        ):
            h5[key].attrs["unit"] = unit
        for mode, component, suffix in (("TE", "Zyx", ""), ("TM", "Zxy", "_tm")):
            for stem in (
                "obs_mt_log10_rho",
                "obs_mt_phase",
                "clean_mt_log10_rho",
                "clean_mt_phase",
            ):
                key = stem + suffix
                h5[key].attrs["mode"] = mode
                h5[key].attrs["impedance_component"] = component
                h5[key].attrs["unit"] = "degree" if "phase" in key else "log10_ohm_m"
        h5["target_log10_res"].attrs["unit"] = "log10_ohm_m"
        h5["scenario"].attrs["labels"] = np.asarray(
            ["background", "aquifer", "hydrocarbon", "salt", "geothermal"],
            dtype="S16",
        )
        h5["sample_index"].attrs["role"] = "generator_sample_index"


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def _make_repository(tmp_path: Path, role: str) -> tuple[Path, str, dict[str, str]]:
    repo = tmp_path / role
    repo.mkdir()
    relative_paths = lineage._SOURCE_FILES[role]
    for ordinal, relative in enumerate(relative_paths):
        source = repo / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"# pinned {role} source {ordinal}\n", encoding="utf-8")
    _git(repo, "init", "--quiet")
    _git(repo, "remote", "add", "origin", _REMOTE)
    _git(repo, "add", ".")
    _git(
        repo,
        "-c",
        "user.name=PIMSR Test",
        "-c",
        "user.email=pimsr-test@example.invalid",
        "commit",
        "--quiet",
        "--no-gpg-sign",
        "-m",
        "fixture",
    )
    commit = _git(repo, "rev-parse", "HEAD")
    hashes = {relative: _sha256(repo / relative) for relative in relative_paths}
    return repo, commit, hashes


def _write_pin_manifest(shard_directory: Path, path: Path) -> str:
    records: list[dict[str, object]] = []
    for hdf5_path in sorted(shard_directory.glob("*.h5")):
        log_path = hdf5_path.with_name(hdf5_path.name + ".log")
        records.append(
            {
                "hdf5_filename": hdf5_path.name,
                "hdf5_sha256": _sha256(hdf5_path),
                "hdf5_size_bytes": hdf5_path.stat().st_size,
                "log_filename": log_path.name,
                "log_sha256": _sha256(log_path),
                "log_size_bytes": log_path.stat().st_size,
            }
        )
    value = {
        "schema": lineage.SHARD_PINS_SCHEMA,
        "schema_version": lineage.SHARD_PINS_SCHEMA_VERSION,
        "shards": records,
    }
    path.write_bytes(_canonical_bytes(value))
    return _sha256(path)


@pytest.fixture
def case(tmp_path: Path) -> dict[str, Any]:
    shards = tmp_path / "shards"
    shards.mkdir()
    for start in (0, 2):
        hdf5_path = shards / f"shard-{start:06d}-{start + 1:06d}.h5"
        _write_dataset(hdf5_path, start=start, count=2, source_shard_count=1)
        hdf5_path.with_name(hdf5_path.name + ".log").write_text(
            f"generated samples {start}-{start + 1}\n", encoding="utf-8"
        )
    merged = tmp_path / "merged.h5"
    _write_dataset(merged, start=0, count=4, source_shard_count=2)
    pin_manifest = tmp_path / "shard-pins.json"
    pin_sha = _write_pin_manifest(shards, pin_manifest)
    forward_repo, forward_commit, forward_hashes = _make_repository(
        tmp_path, "pimsr_forward"
    )
    geogen_repo, geogen_commit, geogen_hashes = _make_repository(tmp_path, "pimsr_geogen")
    return {
        "merged": merged,
        "shards": shards,
        "pin_manifest": pin_manifest,
        "forward_repo": forward_repo,
        "geogen_repo": geogen_repo,
        "output": tmp_path / "lineage.json",
        "kwargs": {
            "split": "train",
            "expected_merged_sha256": _sha256(merged),
            "expected_merged_size_bytes": merged.stat().st_size,
            "expected_shard_pins_sha256": pin_sha,
            "expected_generator_seed": _SEED,
            "expected_sample_start": 0,
            "expected_sample_count": 4,
            "expected_forward_commit": forward_commit,
            "expected_forward_origin_remote": _REMOTE,
            "expected_forward_source_sha256": forward_hashes,
            "expected_geogen_commit": geogen_commit,
            "expected_geogen_origin_remote": _REMOTE,
            "expected_geogen_source_sha256": geogen_hashes,
            "chunk_rows": 1,
        },
    }


def _build(case: dict[str, Any], output: Path | None = None):
    return lineage.build_dataset_lineage_2d(
        case["merged"],
        case["shards"],
        case["pin_manifest"],
        case["forward_repo"],
        case["geogen_repo"],
        output or case["output"],
        **case["kwargs"],
    )


def _refresh_pins(case: dict[str, Any]) -> None:
    case["pin_manifest"].unlink()
    case["kwargs"]["expected_shard_pins_sha256"] = _write_pin_manifest(
        case["shards"], case["pin_manifest"]
    )


def test_builds_canonical_exact_lineage_without_regeneration(case: dict[str, Any]):
    result = _build(case)
    raw = result.path.read_bytes()
    manifest = json.loads(raw)
    assert raw == _canonical_bytes(manifest)
    assert result.sha256 == hashlib.sha256(raw).hexdigest()
    assert result.size_bytes == len(raw)
    assert set(manifest) == {
        "schema",
        "schema_version",
        "evidence_scope",
        "split",
        "source_derived_generation_semantics",
        "inputs",
        "repositories",
        "verification",
    }
    assert manifest["schema"] == lineage.LINEAGE_SCHEMA
    assert manifest["schema_version"] == 2
    assert manifest["evidence_scope"] == (
        "artifact_lineage_and_transitive_generator_source_identity_without_forward_regeneration"
    )
    assert manifest["verification"]["forward_regeneration_performed"] is False
    assert manifest["verification"]["generation_time_execution_proven"] is False
    assert manifest["verification"]["generation_complete"] is True
    assert manifest["verification"]["generator_seed"] == _SEED
    assert manifest["verification"]["sample_count"] == 4
    assert manifest["verification"]["sample_end_index"] == 3
    assert manifest["inputs"]["merged_dataset"]["path"] == "merged.h5"
    assert manifest["inputs"]["shard_directory"]["path"] == "shards"
    assert manifest["inputs"]["shard_pin_manifest"]["path"] == "shard-pins.json"
    assert manifest["repositories"]["pimsr_forward"]["path"] == "pimsr_forward"
    assert manifest["repositories"]["pimsr_geogen"]["path"] == "pimsr_geogen"
    assert manifest["source_derived_generation_semantics"] == {
        "base_layer_rng": "numpy.default_rng([generator_seed,sample_index])",
        "base_layer_scenario": "forced_background_before_2d_scenario_injection",
        "scenario_policy": "SectionGenerator.sample(sample_index,scenario=None)",
        "section_rng": "numpy.default_rng([generator_seed,2,sample_index])",
        "sensor_rng": "numpy.default_rng([generator_seed,3,sample_index])",
        "status": (
            "derived_from_exact_pinned_source_closure_not_generation_time_execution"
        ),
    }
    assert set(manifest["verification"]["arrays"]) == set(lineage._DATASET_KEYS)
    for key in lineage._ROW_KEYS:
        assert manifest["verification"]["arrays"][key]["shard_equality"] == (
            "exact_ordered_concatenation"
        )
    for key in lineage._META_KEYS:
        assert manifest["verification"]["arrays"][key]["shard_equality"] == (
            "exact_repetition_in_every_shard"
        )
    assert [record["sample_start"] for record in manifest["inputs"]["shards"]] == [
        0,
        2,
    ]
    assert all(record["log"]["sha256"] for record in manifest["inputs"]["shards"])
    assert manifest["repositories"]["pimsr_forward"]["clean_worktree"] is True
    assert set(manifest["repositories"]["pimsr_forward"]["source_files"]) == set(
        lineage._SOURCE_FILES["pimsr_forward"]
    )
    assert set(manifest["repositories"]["pimsr_geogen"]["source_files"]) == set(
        lineage._SOURCE_FILES["pimsr_geogen"]
    )


def test_identical_inputs_publish_byte_identical_lineage(case: dict[str, Any]):
    first = _build(case, case["output"])
    second = _build(case, case["output"].with_name("lineage-second.json"))
    assert first.path.read_bytes() == second.path.read_bytes()


def test_swapped_shard_pin_order_is_rejected(case: dict[str, Any]):
    value = json.loads(case["pin_manifest"].read_bytes())
    value["shards"].reverse()
    case["pin_manifest"].write_bytes(_canonical_bytes(value))
    case["kwargs"]["expected_shard_pins_sha256"] = _sha256(case["pin_manifest"])
    with pytest.raises(ValueError, match="gap or overlap"):
        _build(case)
    assert not case["output"].exists()


def test_shard_pin_parse_uses_the_already_hashed_payload(
    case: dict[str, Any], monkeypatch: pytest.MonkeyPatch
):
    """A pathname A/B view after hashing cannot alter the parsed inventory."""
    alternate = json.loads(case["pin_manifest"].read_bytes())
    alternate["shards"].reverse()
    alternate_payload = _canonical_bytes(alternate)
    target = case["pin_manifest"].absolute()
    real_read_bytes = Path.read_bytes
    pathname_reads = 0

    def read_alternate(path: Path) -> bytes:
        nonlocal pathname_reads
        if path.absolute() == target:
            pathname_reads += 1
            return alternate_payload
        return real_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read_alternate)
    result = _build(case)
    assert result.path.exists()
    assert pathname_reads == 0


def test_tampered_shard_bound_by_updated_pin_still_fails_merge(case: dict[str, Any]):
    shard = case["shards"] / "shard-000002-000003.h5"
    with h5py.File(shard, "r+") as h5:
        h5["target_log10_res"][0, 0, 0] += np.float32(1.0)
    _refresh_pins(case)
    with pytest.raises(ValueError, match="differs from merged dataset"):
        _build(case)
    assert not case["output"].exists()


def test_tampered_log_is_rejected_by_its_external_hash_pin(case: dict[str, Any]):
    log = case["shards"] / "shard-000002-000003.h5.log"
    log.write_bytes(log.read_bytes() + b"tampered")
    with pytest.raises(lineage.DatasetLineageError, match="log SHA-256 mismatch"):
        _build(case)
    assert not case["output"].exists()


@pytest.mark.parametrize(("start", "end"), ((3, 4), (1, 2)))
def test_shard_gap_or_overlap_is_rejected(case: dict[str, Any], start: int, end: int):
    old_hdf5 = case["shards"] / "shard-000002-000003.h5"
    old_log = old_hdf5.with_name(old_hdf5.name + ".log")
    old_hdf5.unlink()
    old_log.unlink()
    new_hdf5 = case["shards"] / f"shard-{start:06d}-{end:06d}.h5"
    _write_dataset(new_hdf5, start=start, count=2, source_shard_count=1)
    new_hdf5.with_name(new_hdf5.name + ".log").write_text(
        "replacement range\n", encoding="utf-8"
    )
    _refresh_pins(case)
    with pytest.raises(ValueError, match="gap or overlap"):
        _build(case)
    assert not case["output"].exists()


def test_shard_metadata_mismatch_is_rejected(case: dict[str, Any]):
    shard = case["shards"] / "shard-000002-000003.h5"
    with h5py.File(shard, "r+") as h5:
        h5["frequencies"][1] = 11.0
    _refresh_pins(case)
    with pytest.raises(ValueError, match="metadata differs: frequencies"):
        _build(case)
    assert not case["output"].exists()


def test_hdf5_comparison_never_reopens_a_pinned_pathname(
    case: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A decoy returned for a later pathname open cannot affect comparison."""
    target = (case["shards"] / "shard-000002-000003.h5").absolute()
    decoy = tmp_path / "decoy.h5"
    shutil.copyfile(target, decoy)
    with h5py.File(decoy, "r+") as h5:
        h5["target_log10_res"][0, 0, 0] += np.float32(1.0)

    real_file = lineage.h5py.File
    redirected_path_opens = 0

    def swap_pathname_for_decoy(name: Any, *args: Any, **kwargs: Any):
        nonlocal redirected_path_opens
        if (
            isinstance(name, (str, bytes, os.PathLike))
            and Path(name).absolute() == target
        ):
            redirected_path_opens += 1
            return real_file(decoy, *args, **kwargs)
        return real_file(name, *args, **kwargs)

    monkeypatch.setattr(lineage.h5py, "File", swap_pathname_for_decoy)
    result = _build(case)
    assert result.path.exists()
    assert redirected_path_opens == 0


def test_dirty_source_repository_is_rejected(case: dict[str, Any]):
    (case["forward_repo"] / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(lineage.DatasetLineageError, match="clean worktree"):
        _build(case)
    assert not case["output"].exists()


def test_source_payload_must_match_the_pinned_commit_blob_even_if_status_lies(
    case: dict[str, Any], monkeypatch: pytest.MonkeyPatch
):
    relative = lineage._SOURCE_FILES["pimsr_forward"][0]
    source = case["forward_repo"] / relative
    source.write_bytes(source.read_bytes() + b"# pathname replacement\n")
    case["kwargs"]["expected_forward_source_sha256"][relative] = _sha256(source)
    real_git = lineage._git

    def hide_dirty_status(repo: Path, *arguments: str) -> str:
        if arguments[:2] == ("status", "--porcelain=v1"):
            return ""
        return real_git(repo, *arguments)

    monkeypatch.setattr(lineage, "_git", hide_dirty_status)
    with pytest.raises(lineage.DatasetLineageError, match="pinned commit blob"):
        _build(case)
    assert not case["output"].exists()


def test_wrong_source_commit_is_rejected(case: dict[str, Any]):
    case["kwargs"]["expected_geogen_commit"] = "0" * 40
    with pytest.raises(lineage.DatasetLineageError, match="commit mismatch"):
        _build(case)
    assert not case["output"].exists()


def test_incomplete_merged_dataset_is_rejected_even_when_re_pinned(case: dict[str, Any]):
    with h5py.File(case["merged"], "r+") as h5:
        h5.attrs["generation_complete"] = np.uint8(0)
    case["kwargs"]["expected_merged_sha256"] = _sha256(case["merged"])
    case["kwargs"]["expected_merged_size_bytes"] = case["merged"].stat().st_size
    with pytest.raises(ValueError, match="generation_complete"):
        _build(case)
    assert not case["output"].exists()


def test_symlinked_external_input_is_rejected(case: dict[str, Any], tmp_path: Path):
    log = case["shards"] / "shard-000000-000001.h5.log"
    target = tmp_path / "outside.log"
    shutil.copyfile(log, target)
    log.unlink()
    try:
        log.symlink_to(target)
    except OSError as exc:  # pragma: no cover - Windows policy dependent
        pytest.skip(f"symbolic links are unavailable: {exc}")
    with pytest.raises(ValueError, match="regular non-symlink"):
        _build(case)
    assert not case["output"].exists()


def test_reported_symlink_is_rejected_on_platforms_without_symlink_privilege(
    case: dict[str, Any], monkeypatch: pytest.MonkeyPatch
):
    target = case["shards"] / "shard-000000-000001.h5.log"
    real_is_symlink = Path.is_symlink

    def report_target_as_symlink(path: Path) -> bool:
        return path == target or real_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", report_target_as_symlink)
    with pytest.raises(ValueError, match="regular non-symlink"):
        _build(case)
    assert not case["output"].exists()


def test_replacement_after_comparison_is_detected(
    case: dict[str, Any], monkeypatch: pytest.MonkeyPatch
):
    real_compare = lineage._compare_hdf5
    log = case["shards"] / "shard-000000-000001.h5.log"

    def compare_then_replace(*args: Any, **kwargs: Any):
        result = real_compare(*args, **kwargs)
        log.write_bytes(log.read_bytes() + b"replacement")
        return result

    monkeypatch.setattr(lineage, "_compare_hdf5", compare_then_replace)
    with pytest.raises(lineage.DatasetLineageError, match="replaced or changed"):
        _build(case)
    assert not case["output"].exists()


def test_existing_output_is_never_overwritten(case: dict[str, Any]):
    case["output"].write_bytes(b"foreign evidence")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _build(case)
    assert case["output"].read_bytes() == b"foreign evidence"
    assert not case["output"].with_name(case["output"].name + ".part").exists()


def test_publication_failure_rolls_back_own_hardlink(
    case: dict[str, Any], monkeypatch: pytest.MonkeyPatch
):
    real_link = os.link

    def link_then_fail(source: Path, destination: Path) -> None:
        real_link(source, destination)
        raise OSError("injected post-link failure")

    monkeypatch.setattr(lineage.os, "link", link_then_fail)
    with pytest.raises(OSError, match="injected"):
        _build(case)
    assert not case["output"].exists()
    assert not case["output"].with_name(case["output"].name + ".part").exists()


def test_publication_rollback_preserves_foreign_replacement(
    case: dict[str, Any], monkeypatch: pytest.MonkeyPatch
):
    def foreign_then_fail(source: Path, destination: Path) -> None:
        Path(destination).write_bytes(b"foreign replacement")
        raise OSError("publication race")

    monkeypatch.setattr(lineage.os, "link", foreign_then_fail)
    with pytest.raises(
        lineage.DatasetLineageError, match="refusing destructive rollback"
    ):
        _build(case)
    assert case["output"].read_bytes() == b"foreign replacement"
    assert not case["output"].with_name(case["output"].name + ".part").exists()


def test_publication_detects_replacement_after_successful_hardlink(
    case: dict[str, Any], monkeypatch: pytest.MonkeyPatch
):
    real_link = os.link

    def link_then_replace(source: Path, destination: Path) -> None:
        real_link(source, destination)
        Path(destination).unlink()
        Path(destination).write_bytes(b"foreign after link")

    monkeypatch.setattr(lineage.os, "link", link_then_replace)
    with pytest.raises(
        lineage.DatasetLineageError, match="refusing destructive rollback"
    ):
        _build(case)
    assert case["output"].read_bytes() == b"foreign after link"
    assert not case["output"].with_name(case["output"].name + ".part").exists()


def test_cli_requires_external_hash_and_commit_pins():
    parser = lineage._parser()
    required = {
        action.dest for action in parser._actions if getattr(action, "required", False)
    }
    assert {
        "expected_merged_sha256",
        "expected_shard_pins_sha256",
        "expected_forward_commit",
        "expected_forward_dataset2d_sha256",
        "expected_forward_mt2d_sha256",
        "expected_forward_sensors_sha256",
        "expected_geogen_commit",
        "expected_geogen_generator_sha256",
        "expected_geogen_model_sha256",
        "expected_geogen_rock_physics_sha256",
        "expected_geogen_section2d_sha256",
    } <= required
