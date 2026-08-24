"""Regression tests for the pinned production ModEM 2-D forward bridge."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import pimsr_benchmarks.modem2d_forward as modem


@pytest.fixture
def truth() -> modem.CanonicalTruth:
    x = np.arange(-11_750.0, 12_000.0, 500.0)
    depth = np.geomspace(10.0, 60_000.0, 64)
    frequencies = np.geomspace(0.01, 10.0, 8)
    stations = np.linspace(-8_000.0, 8_000.0, 12)
    values = 2.0 + np.arange(64)[:, None] / 100.0 + np.arange(48)[None, :] / 1_000.0
    return modem.CanonicalTruth(
        values, x, depth, frequencies, stations, "public-test-000"
    )


@pytest.fixture
def small_mesh() -> modem.MeshConfig:
    return modem.MeshConfig(
        "test-mesh",
        1,
        6_000.0,
        4,
        1,
        1.4,
        100.0,
        1.2,
        1_000.0,
        3_000.0,
    )


def _response_text(
    truth: modem.CanonicalTruth,
    mesh: modem.MeshConfig,
    *,
    omit_last_tm: bool = False,
    nonfinite: bool = False,
) -> str:
    dy, _ = mesh.cell_widths()
    station_y = 0.5 * dy.sum() + truth.station_x_m
    rows: list[str] = []
    for mode, rho, phase in (("TE", 100.0, 30.0), ("TM", 1_000.0, 120.0)):
        rows.extend(
            (
                "# generated test response",
                "# columns",
                f"> {mode}_Impedance",
                "> exp(+i\\omega t)",
                "> [V/m]/[T]",
                "> 0.00",
                "> 0.000 0.000",
                "> 8 12",
            )
        )
        for station_index, y_value in enumerate(station_y, start=1):
            for frequency_index, frequency in enumerate(truth.frequencies_hz):
                if (
                    omit_last_tm
                    and mode == "TM"
                    and station_index == 12
                    and frequency_index == 7
                ):
                    continue
                omega = 2.0 * math.pi * frequency
                magnitude = math.sqrt(rho * omega / modem.MU0)
                value = magnitude * np.exp(1j * math.radians(phase))
                real = (
                    "nan"
                    if nonfinite and mode == "TE" and station_index == 1
                    else f"{value.real:.9e}"
                )
                rows.append(
                    f"{1.0 / frequency:.6f} S{station_index:02d} 0.0 0.0 0.0 "
                    f"{y_value:.3f} 0.0 {mode} {real} {value.imag:.9e} 1.0"
                )
    return "\n".join(rows) + "\n"


def test_mesh_configs_are_explicit_hashable_and_include_ultra2():
    ultra2 = modem.MESH_CONFIGS["ultra2"]
    dy, dz = ultra2.cell_widths()

    assert ultra2.core_width_m == 31.25
    assert ultra2.first_dz_m == 6.25
    assert ultra2.core_count == 768
    assert ultra2.sha256 == modem.canonical_json_sha256(ultra2.canonical_record())
    assert dy.size == 768 + 2 * 22
    assert dz[0] == 6.25
    assert dz.sum() >= 220_000.0
    assert ultra2.padding_perturbation().padding_count_each_side == 24


def test_canonical_public_axes_are_bitwise_frozen_and_return_copies():
    depth = modem.canonical_depth_centres_m()
    frequencies = modem.canonical_frequencies_hz()

    assert depth.shape == (64,)
    assert frequencies.shape == (8,)
    assert hashlib.sha256(depth.tobytes()).hexdigest() == (
        modem.CANONICAL_DEPTH_CENTRES_SHA256
    )
    assert hashlib.sha256(frequencies.tobytes()).hexdigest() == (
        modem.CANONICAL_FREQUENCIES_SHA256
    )
    depth[0] = -1.0
    frequencies[0] = -1.0
    assert modem.canonical_depth_centres_m()[0] == 10.0
    assert modem.canonical_frequencies_hz()[0] == 0.01


def test_nested_mesh_pair_has_identical_domain_and_exact_factor_two_cells(
    truth: modem.CanonicalTruth, monkeypatch: pytest.MonkeyPatch
):
    canonical_depth = np.asarray(truth.depth_centres_m, dtype="<f8")
    monkeypatch.setattr(
        modem,
        "CANONICAL_DEPTH_CENTRES_SHA256",
        modem.hashlib.sha256(canonical_depth.tobytes()).hexdigest(),
    )
    candidate = modem.MESH_CONFIGS["nested-base-v1"]
    horizontal = modem.MESH_CONFIGS["nested-horizontal-only-v1"]
    vertical = modem.MESH_CONFIGS["nested-production-v1"]
    reference = modem.MESH_CONFIGS["nested-reference-x2-v1"]
    candidate_dy, candidate_dz = candidate.cell_widths(canonical_depth)
    horizontal_dy, horizontal_dz = horizontal.cell_widths(canonical_depth)
    vertical_dy, vertical_dz = vertical.cell_widths(canonical_depth)
    reference_dy, reference_dz = reference.cell_widths(canonical_depth)

    assert reference_dy.size == 2 * candidate_dy.size
    assert reference_dz.size == 2 * candidate_dz.size
    np.testing.assert_array_equal(horizontal_dy, reference_dy)
    np.testing.assert_array_equal(horizontal_dz, candidate_dz)
    np.testing.assert_array_equal(vertical_dy, candidate_dy)
    np.testing.assert_array_equal(vertical_dz, reference_dz)
    assert reference_dy.size == 2 * vertical_dy.size
    np.testing.assert_allclose(
        reference_dy.reshape(-1, 2).sum(axis=1),
        vertical_dy,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        reference_dy.reshape(-1, 2).sum(axis=1),
        candidate_dy,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        reference_dz.reshape(-1, 2).sum(axis=1),
        candidate_dz,
        rtol=0.0,
        atol=1e-12,
    )
    assert candidate_dz.size == 196
    assert reference_dz.size == 392
    assert candidate_dz.sum() == pytest.approx(220_000.0, abs=1e-7)
    assert reference_dz.sum() == pytest.approx(220_000.0, abs=1e-7)
    internal_boundaries = 0.5 * (canonical_depth[:-1] + canonical_depth[1:])
    candidate_edges = np.cumsum(candidate_dz)
    for boundary in internal_boundaries:
        assert np.min(np.abs(candidate_edges - boundary)) < 1e-9

    mapped_candidate, _, _ = modem.mapped_model(truth, candidate)
    mapped_production, _, _ = modem.mapped_model(truth, vertical)
    mapped_reference, _, _ = modem.mapped_model(truth, reference)
    expected_reference = np.repeat(np.repeat(mapped_candidate, 2, axis=0), 2, axis=1)
    assert np.array_equal(mapped_reference, expected_reference)
    assert np.array_equal(
        mapped_reference,
        np.repeat(mapped_production, 2, axis=1),
    )


def test_canonical_truth_rejects_wrong_shape_or_unsorted_axes(
    truth: modem.CanonicalTruth,
):
    with pytest.raises(ValueError, match="shape"):
        modem.CanonicalTruth(
            truth.log10_resistivity[:, :-1],
            truth.x_centres_m,
            truth.depth_centres_m,
            truth.frequencies_hz,
            truth.station_x_m,
            "bad",
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        modem.CanonicalTruth(
            truth.log10_resistivity,
            truth.x_centres_m[::-1],
            truth.depth_centres_m,
            truth.frequencies_hz,
            truth.station_x_m,
            "bad",
        )


def test_model_writer_is_loge_ln_rho_and_no_overwrite(tmp_path: Path, small_mesh):
    base = np.full((64, 48), 2.0)
    truth = modem.CanonicalTruth(
        base,
        np.arange(-11_750.0, 12_000.0, 500.0),
        np.geomspace(10.0, 60_000.0, 64),
        np.geomspace(0.01, 10.0, 8),
        np.linspace(-8_000.0, 8_000.0, 12),
        "uniform",
    )
    path = tmp_path / "model.rho"
    snapshot, record = modem.write_modem_model(path, truth, small_mesh)
    tokens = path.read_text(encoding="ascii").split()
    dy, dz = small_mesh.cell_widths()
    assert tokens[:3] == [str(dy.size), str(dz.size), "LOGE"]
    offset = 3 + dy.size + dz.size
    assert tokens[offset] == "0"
    values = np.asarray([float(value) for value in tokens[offset + 1 :]])
    assert values.size == dy.size * dz.size
    assert np.allclose(values, math.log(100.0), rtol=0.0, atol=1e-11)
    assert record["representation"] == "LOGE natural_log_resistivity_ohm_m"
    assert snapshot.sha256 == modem.snapshot_file(path, role="test model").sha256
    with pytest.raises(FileExistsError):
        modem.write_modem_model(path, truth, small_mesh)


def test_piecewise_mapping_uses_physical_centres_and_ties_left(
    truth: modem.CanonicalTruth, small_mesh: modem.MeshConfig
):
    mapped, dy, dz = modem.mapped_model(truth, small_mesh)
    y_centres = np.cumsum(dy) - 0.5 * dy - 0.5 * dy.sum()
    depth_centres = np.cumsum(dz) - 0.5 * dz
    ix = modem.nearest_indices(truth.x_centres_m, y_centres)
    iz = modem.nearest_indices(truth.depth_centres_m, depth_centres)
    assert np.array_equal(mapped, truth.log10_resistivity[np.ix_(iz, ix)])
    assert modem.nearest_indices(np.array([0.0, 2.0]), np.array([1.0])).item() == 0


def test_template_requests_plus_iwt_and_keeps_te_tm_unswapped(
    tmp_path: Path, truth: modem.CanonicalTruth, small_mesh: modem.MeshConfig
):
    path = tmp_path / "template.dat"
    _snapshot, record = modem.write_modem_template(path, truth, small_mesh)
    text = path.read_text(encoding="ascii")
    assert text.count("> exp(+i\\omega t)") == 2
    assert "exp(-i" not in text
    assert text.count("> TE_Impedance") == 1
    assert text.count("> TM_Impedance") == 1
    assert record["manual_conjugation"] is False
    assert record["rows_per_mode"] == 96
    data_rows = [
        line.split() for line in text.splitlines() if line and line[0] not in "#>"
    ]
    assert len(data_rows) == 192
    assert {row[7] for row in data_rows[:96]} == {"TE"}
    assert {row[7] for row in data_rows[96:]} == {"TM"}


def test_parser_consumes_plus_iwt_directly_and_returns_canonical_modes(
    tmp_path: Path, truth: modem.CanonicalTruth, small_mesh: modem.MeshConfig
):
    output = tmp_path / "forward.dat"
    output.write_text(_response_text(truth, small_mesh), encoding="ascii", newline="\n")
    response, record = modem.parse_modem_response(output, truth, small_mesh)

    assert response.log10_rho_te.shape == (8, 12)
    assert np.allclose(response.log10_rho_te, 2.0, atol=2e-8)
    assert np.allclose(response.phase_te_deg, 30.0, atol=2e-8)
    assert np.allclose(response.log10_rho_tm, 3.0, atol=2e-8)
    assert np.allclose(response.phase_tm_deg, 120.0, atol=2e-8)
    assert record["rows"] == {"TE": 96, "TM": 96}
    assert record["manual_conjugation"] is False
    assert record["canonical_mode_order"] == ["TE_Zyx", "TM_Zxy"]


def test_parser_uses_pinned_snapshot_bytes_during_a_b_swap(
    tmp_path: Path,
    truth: modem.CanonicalTruth,
    small_mesh: modem.MeshConfig,
    monkeypatch: pytest.MonkeyPatch,
):
    output = tmp_path / "forward.dat"
    original_payload = _response_text(truth, small_mesh).encode("ascii")
    output.write_bytes(original_payload)
    saved = tmp_path / "forward.saved"
    replacement = tmp_path / "forward.replacement"
    replacement.write_bytes(b"not a ModEM response\n")
    real_snapshot = modem.snapshot_file
    calls = 0

    def swapping_snapshot(path: Path, *, role: str):
        nonlocal calls
        if calls == 0:
            snapshot = real_snapshot(path, role=role)
            Path(path).replace(saved)
            replacement.replace(path)
            calls += 1
            return snapshot
        Path(path).unlink()
        saved.replace(path)
        calls += 1
        return real_snapshot(path, role=role)

    monkeypatch.setattr(modem, "snapshot_file", swapping_snapshot)

    response, _record = modem.parse_modem_response(output, truth, small_mesh)

    assert calls == 2
    assert np.allclose(response.log10_rho_te, 2.0, atol=2e-8)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (({"omit_last_tm": True}, "exactly 96"), ({"nonfinite": True}, "non-finite")),
)
def test_parser_rejects_incomplete_or_nonfinite_output(
    tmp_path: Path,
    truth: modem.CanonicalTruth,
    small_mesh: modem.MeshConfig,
    kwargs: dict[str, bool],
    message: str,
):
    output = tmp_path / "bad-forward.dat"
    output.write_text(
        _response_text(truth, small_mesh, **kwargs), encoding="ascii", newline="\n"
    )
    with pytest.raises(ValueError, match=message):
        modem.parse_modem_response(output, truth, small_mesh)


def test_analytic_uniform_halfspace_is_100_ohm_m_45_degrees(
    truth: modem.CanonicalTruth, small_mesh: modem.MeshConfig
):
    uniform = modem.CanonicalTruth(
        np.full((64, 48), 2.0),
        truth.x_centres_m,
        truth.depth_centres_m,
        truth.frequencies_hz,
        truth.station_x_m,
        "halfspace",
    )
    response = modem.analytic_response_for_mapped_1d(uniform, small_mesh)
    assert np.allclose(response.log10_rho_te, 2.0, atol=1e-12)
    assert np.allclose(response.phase_te_deg, 45.0, atol=1e-12)
    assert np.array_equal(response.log10_rho_te, response.log10_rho_tm)


def test_circular_phase_and_frozen_gates():
    error = modem.circular_phase_error_deg(np.array([179.0, 1.0]), np.array([1.0, 179.0]))
    assert np.array_equal(error, [2.0, 2.0])
    summary = {"median": 0.005, "p95": 0.015, "max": 0.05}
    assert modem.gate_summary(summary, quantity="log10_rho")["passed"] is True
    summary["p95"] = 0.0150001
    assert modem.gate_summary(summary, quantity="log10_rho")["passed"] is False


def test_snapshot_detects_binary_mutation(tmp_path: Path):
    binary = tmp_path / "Mod2DMT"
    binary.write_bytes(b"pinned")
    snapshot = modem.snapshot_file(binary, role="test binary")
    binary.write_bytes(b"mutated")
    with pytest.raises(RuntimeError, match="changed after"):
        modem.require_snapshot_unchanged(snapshot, role="test binary")


def test_snapshot_rejects_symlink(tmp_path: Path):
    target = tmp_path / "target.bin"
    target.write_bytes(b"pinned")
    link = tmp_path / "link.bin"
    try:
        os.symlink(target, link)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(FileNotFoundError, match="regular non-link"):
        modem.snapshot_file(link, role="linked input")


def _fake_git(repo: Path, *arguments: str) -> str:
    del repo
    if arguments == ("rev-parse", "HEAD^{commit}"):
        return modem.PINNED_MODEM_COMMIT
    if arguments == ("rev-parse", "HEAD^{tree}"):
        return modem.PINNED_MODEM_TREE
    if arguments == ("rev-parse", f"refs/tags/{modem.PINNED_MODEM_TAG}^{{commit}}"):
        return modem.PINNED_MODEM_COMMIT
    if arguments == ("describe", "--tags", "--exact-match", "HEAD"):
        return modem.PINNED_MODEM_TAG
    if arguments == ("status", "--porcelain=v1", "--untracked-files=all"):
        return ""
    raise AssertionError(arguments)


def test_runtime_verifier_binds_commit_tree_tag_clean_and_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo = tmp_path / "repo"
    root = tmp_path / "build"
    repo.mkdir()
    root.mkdir()
    monkeypatch.setattr(modem, "_git", _fake_git)
    monkeypatch.setattr(modem, "_verify_build_recipe", lambda build, source: ())

    def capture(command, **kwargs):
        del kwargs
        if command[1:3] == ("image", "inspect"):
            payload = {
                "Id": modem.PINNED_CONTAINER_DIGEST,
                "RepoDigests": [f"ubuntu@{modem.PINNED_CONTAINER_DIGEST}"],
            }
            return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")
        return subprocess.CompletedProcess(command, 0, '"29.7.2" "29.7.2"', "")

    monkeypatch.setattr(modem, "_run_capture", capture)
    runtime = modem.verify_pinned_runtime(modem_repo=repo, build_root=root)
    assert runtime.record["modem"]["checkout_clean"] is True
    assert runtime.record["container"]["image_id"] == modem.PINNED_CONTAINER_DIGEST


def test_runtime_verifier_rejects_dirty_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo = tmp_path / "repo"
    root = tmp_path / "build"
    repo.mkdir()
    root.mkdir()

    def dirty_git(source: Path, *arguments: str) -> str:
        if arguments == ("status", "--porcelain=v1", "--untracked-files=all"):
            return " M f90/2D_MT/DataIO.f90"
        return _fake_git(source, *arguments)

    monkeypatch.setattr(modem, "_git", dirty_git)
    with pytest.raises(modem.ProvenanceError, match="not clean"):
        modem.verify_pinned_runtime(modem_repo=repo, build_root=root)


def test_atomic_bundle_no_overwrite_and_manifest_last(tmp_path: Path):
    output = modem.publish_artifact_bundle(
        tmp_path / "result",
        {"data.bin": b"data", "provenance.json": b"manifest"},
        manifest_name="provenance.json",
    )
    assert (output / "data.bin").read_bytes() == b"data"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        modem.publish_artifact_bundle(
            output,
            {"data.bin": b"other", "provenance.json": b"other"},
            manifest_name="provenance.json",
        )


def test_atomic_bundle_rolls_back_on_link_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    destination = tmp_path / "rollback"
    real_link = os.link
    calls = 0

    def fail_second(source: Path, target: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected publication failure")
        real_link(source, target)

    monkeypatch.setattr(modem.os, "link", fail_second)
    with pytest.raises(OSError, match="injected"):
        modem.publish_artifact_bundle(
            destination,
            {"data.bin": b"data", "provenance.json": b"manifest"},
            manifest_name="provenance.json",
        )
    assert not destination.exists()
    assert not list(tmp_path.glob("*.part"))


def test_atomic_bundle_rejects_staged_payload_mutation_during_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    destination = tmp_path / "mutated-stage"
    real_link = os.link
    mutated = False

    def mutate_then_link(source: Path, target: Path) -> None:
        nonlocal mutated
        if not mutated:
            source.write_bytes(b"tampered-stage-payload")
            mutated = True
        real_link(source, target)

    monkeypatch.setattr(modem.os, "link", mutate_then_link)
    with pytest.raises(modem.PublicationError, match="changed while linking"):
        modem.publish_artifact_bundle(
            destination,
            {"data.bin": b"data", "provenance.json": b"manifest"},
            manifest_name="provenance.json",
        )
    assert mutated is True
    assert not destination.exists()
    assert not list(tmp_path.glob("*.part"))


def test_atomic_bundle_rejects_dangling_output_symlink(tmp_path: Path):
    destination = tmp_path / "result"
    try:
        os.symlink(tmp_path / "missing-target", destination, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(FileExistsError, match="overwrite"):
        modem.publish_artifact_bundle(
            destination,
            {"data.bin": b"data", "provenance.json": b"manifest"},
            manifest_name="provenance.json",
        )


def test_hdf_loader_requires_canonical_mode_order(
    tmp_path: Path, truth: modem.CanonicalTruth
):
    path = tmp_path / "public.h5"
    with h5py.File(path, "w") as h5:
        h5.attrs["schema"] = "pimsr-mt-2d"
        h5.attrs["schema_version"] = 2
        h5.attrs["impedance_components"] = np.asarray(["Zyx", "Zxy"], dtype="S3")
        h5.attrs["mode_order"] = np.asarray(["te", "tm"], dtype="S2")
        h5.attrs["phase_convention"] = "degrees_modulo_180_[0,180)"
        h5.create_dataset("target_log10_res", data=truth.log10_resistivity[None])
        h5.create_dataset("x_grid", data=truth.x_centres_m)
        h5.create_dataset("depth_grid", data=truth.depth_centres_m)
        h5.create_dataset("frequencies", data=truth.frequencies_hz)
        h5.create_dataset("station_x", data=truth.station_x_m)
        h5.create_dataset("sample_index", data=[17])
        h5.create_dataset("scenario", data=[2])
    loaded, record = modem.load_canonical_hdf5(path, row=0)
    assert loaded.sample_id == "sample-000017"
    assert record["scenario_index"] == 2

    with h5py.File(path, "r+") as h5:
        h5.attrs["mode_order"] = np.asarray(["tm", "te"], dtype="S2")
    with pytest.raises(ValueError, match="mode order"):
        modem.load_canonical_hdf5(path, row=0)


def test_run_forward_rejects_preexisting_output_without_solver(
    tmp_path: Path, truth: modem.CanonicalTruth, small_mesh: modem.MeshConfig
):
    output = tmp_path / "existing"
    output.mkdir()
    runtime = SimpleNamespace()
    with pytest.raises(FileExistsError, match="overwrite"):
        modem.run_modem_forward(
            runtime=runtime,
            truth=truth,
            mesh=small_mesh,
            output_dir=output,
            source_provenance={},
        )


def test_run_forward_publishes_only_verified_snapshot_payloads(
    tmp_path: Path,
    truth: modem.CanonicalTruth,
    small_mesh: modem.MeshConfig,
    monkeypatch: pytest.MonkeyPatch,
):
    runtime = SimpleNamespace(
        record={},
        identity_sha256="0" * 64,
        require_unchanged=lambda: None,
    )
    response_payload = _response_text(truth, small_mesh)

    def fake_solver(
        _runtime: object,
        *,
        input_dir: Path,
        solver_dir: Path,
        timeout_seconds: float,
    ):
        del input_dir, timeout_seconds
        (solver_dir / "forward.dat").write_text(
            response_payload, encoding="ascii", newline="\n"
        )
        return subprocess.CompletedProcess([], 0, "stdout", ""), ["fake"], 0.01

    monkeypatch.setattr(modem, "_run_solver", fake_solver)

    def reject_path_reopen(_path: Path) -> bytes:
        raise AssertionError("verified artifacts must not be reopened by pathname")

    monkeypatch.setattr(Path, "read_bytes", reject_path_reopen)
    published, _response, provenance = modem.run_modem_forward(
        runtime=runtime,
        truth=truth,
        mesh=small_mesh,
        output_dir=tmp_path / "published",
        source_provenance={"scope": "public_test"},
    )

    assert published.is_dir()
    assert (
        provenance["outputs"]["forward.dat"]["sha256"]
        == hashlib.sha256(response_payload.encode("ascii")).hexdigest()
    )
