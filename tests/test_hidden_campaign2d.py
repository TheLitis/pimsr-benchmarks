"""Contract tests for hidden 2-D materialization; no real ModEM is invoked."""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pimsr_geogen.section2d import DEFAULT_X_GRID

from pimsr_benchmarks import evaluation2d, prediction_lock2d
from pimsr_benchmarks import hidden_campaign2d as hidden_campaign
from pimsr_benchmarks.hidden_campaign2d import (
    BASE_COUNT,
    BASES_PER_FAMILY,
    CAMPAIGN_SCHEMA,
    CAMPAIGN_SCHEMA_VERSION,
    FROZEN_GENERATION_RUNTIME,
    SAMPLE_COUNT,
    CampaignGeometry2D,
    FileIdentity2D,
    GenerationRuntimeManifest2D,
    HiddenCampaign2DError,
    VerifiedBaseForward2D,
    _build_core_campaign,
    _build_final_evidence_plan,
    _build_hidden_bases,
    _materialize_raw_forwards,
    _noisy_response,
    _prepare_work_directory,
    _publish_operator_directory,
    _verify_published_operator_directory,
    _work_contract,
    build_hidden_generation_runtime_manifest_2d,
    generation_contract_2d,
    validate_hidden_generation_runtime_manifest_2d,
)
from pimsr_benchmarks.modem2d_forward import (
    MESH_CONFIGS,
    ArtifactSnapshot,
    ModEMResponse,
    canonical_depth_centres_m,
    canonical_frequencies_hz,
)
from pimsr_benchmarks.prediction_lock2d import snapshot_regular_file


def _source_lineage() -> dict:
    return {
        "pimsr_forward": {
            "repository_commit": "1" * 40,
            "dataset2d_source_sha256": "2" * 64,
            "sensors_source_sha256": "3" * 64,
        },
        "pimsr_geogen": {
            "repository_commit": "4" * 40,
            "generator_source_sha256": "5" * 64,
            "model_source_sha256": "6" * 64,
            "rock_physics_source_sha256": "7" * 64,
            "section2d_source_sha256": "8" * 64,
        },
    }


def _geometry(tmp_path: Path) -> CampaignGeometry2D:
    payload = b"mock geometry; not a production hidden artifact"
    source_path = tmp_path / "geometry.h5"
    source_path.write_bytes(payload)
    info = os.stat(source_path)
    source = ArtifactSnapshot(
        source_path.resolve(),
        hashlib.sha256(payload).hexdigest(),
        len(payload),
        (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns),
        payload,
    )
    return CampaignGeometry2D(
        x_cell_centers_m=DEFAULT_X_GRID.astype("<f8"),
        depth_cell_centers_m=canonical_depth_centres_m(),
        frequencies_hz=canonical_frequencies_hz(),
        station_x_m=np.linspace(-10_000.0, 10_000.0, 12, dtype="<f8"),
        source=source,
    )


class _FakeSectionGenerator:
    def __init__(self, seed: int, geometry: CampaignGeometry2D) -> None:
        self.seed = seed
        self.geometry = geometry

    def sample(self, index: int, scenario: str):
        grid = np.full((64, 48), 2.0 + index / 1000.0, dtype=np.float64)
        grid[index % 64, index % 48] += 0.125
        return SimpleNamespace(
            scenario=scenario,
            seed=index,
            x_grid=self.geometry.x_cell_centers_m,
            depth_grid=self.geometry.depth_cell_centers_m,
            log10_res=grid,
            has_fault=bool(index % 2),
        )


def _bases(tmp_path: Path):
    geometry = _geometry(tmp_path)
    return geometry, _build_hidden_bases(
        generator_seed=20260830,
        campaign_id="campaign-a",
        sample_id_key=b"k" * 32,
        geometry=geometry,
        section_generator_factory=lambda seed: _FakeSectionGenerator(seed, geometry),
    )


def _response(geometry: CampaignGeometry2D, base_index: int) -> ModEMResponse:
    shape = (8, 12)
    log_rho_te = np.full(shape, 1.4 + base_index * 1.0e-4)
    log_rho_tm = np.full(shape, 1.8 + base_index * 1.0e-4)
    phase_te = np.full(shape, 35.0 + base_index * 1.0e-3)
    phase_tm = np.full(shape, 55.0 + base_index * 1.0e-3)
    return ModEMResponse(
        frequencies_hz=geometry.frequencies_hz,
        station_x_m=geometry.station_x_m,
        z_eb_te=np.full(shape, 1.0 + 2.0j),
        z_eb_tm=np.full(shape, 2.0 + 3.0j),
        log10_rho_te=log_rho_te,
        phase_te_deg=phase_te,
        log10_rho_tm=log_rho_tm,
        phase_tm_deg=phase_tm,
    )


def _identity(path: Path, character: str, size: int = 10) -> FileIdentity2D:
    payload = character.encode("ascii") * size
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return FileIdentity2D(
        path.resolve(), hashlib.sha256(payload).hexdigest(), len(payload)
    )


def _forwards(tmp_path: Path, geometry: CampaignGeometry2D, bases):
    result = []
    for base in bases:
        root = tmp_path / "raw" / base.base_model_id
        files = {
            "model.rho": _identity(root / "model.rho", "1"),
            "template.dat": _identity(root / "template.dat", "2"),
            "forward.dat": _identity(root / "forward.dat", "3"),
            "responses.npz": _identity(root / "responses.npz", "4"),
            "solver.stdout.txt": _identity(root / "solver.stdout.txt", "5"),
            "solver.stderr.txt": _identity(root / "solver.stderr.txt", "6"),
            "provenance.json": _identity(root / "provenance.json", "7"),
        }
        outputs = {
            name: {"sha256": identity.sha256, "size_bytes": identity.size_bytes}
            for name, identity in files.items()
            if name != "provenance.json"
        }
        provenance = {
            "truth_source": {},
            "outputs": outputs,
            "input_contract": {
                "model": {
                    "artifact": {
                        "path": str(files["model.rho"].path),
                        **dict(outputs["model.rho"]),
                    }
                },
                "template": {
                    "artifact": {
                        "path": str(files["template.dat"].path),
                        **dict(outputs["template.dat"]),
                    }
                },
            },
            "response_contract": {
                "artifact": {
                    "path": str(files["forward.dat"].path),
                    **dict(outputs["forward.dat"]),
                }
            },
        }
        result.append(
            VerifiedBaseForward2D(
                base=base,
                bundle_path=root,
                files=files,
                response=_response(geometry, base.base_index),
                provenance=provenance,
            )
        )
    return tuple(result)


def _runtime():
    record = {"schema": "mock-runtime", "schema_version": 1}
    payload = json.dumps(
        record, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return SimpleNamespace(
        record=record,
        identity_sha256=hashlib.sha256(payload).hexdigest(),
    )


def _runtime_manifest_value() -> dict:
    return {
        "schema": "pimsr-hidden-generation-runtime-2d",
        "schema_version": 1,
        "python": {
            "implementation": "CPython",
            "version": "3.11.15",
            "executable_sha256": "9" * 64,
        },
        "distributions": {
            "numpy": {"version": "2.4.6", "installed_tree_sha256": "a" * 64},
            "h5py": {"version": "3.16.0", "installed_tree_sha256": "b" * 64},
            "pimsr_geogen": {
                "version": "0.2.0",
                "installed_tree_sha256": "c" * 64,
            },
            "pimsr_forward": {
                "version": "0.2.0",
                "installed_tree_sha256": "d" * 64,
            },
        },
        "source_closure": _source_lineage(),
        "tree_manifest_sha256": "e" * 64,
    }


def _runtime_manifest(tmp_path: Path) -> GenerationRuntimeManifest2D:
    value = _runtime_manifest_value()
    payload = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    path = tmp_path / "generation-runtime.json"
    path.write_bytes(payload)
    info = os.stat(path)
    return GenerationRuntimeManifest2D(
        ArtifactSnapshot(
            path.resolve(),
            hashlib.sha256(payload).hexdigest(),
            len(payload),
            (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns),
            payload,
        ),
        value,
    )


def test_generation_contract_records_both_geology_rng_streams():
    contract = generation_contract_2d(
        generator_seed=20260830, source_lineage=_source_lineage()
    )
    assert contract["schema_version"] == 2
    assert contract["base_layer_rng"] == "numpy.default_rng([generator_seed,base_index])"
    assert contract["section_rng"] == "numpy.default_rng([generator_seed,2,base_index])"
    assert contract["noise_rng"].endswith("base_index,noise_index])")
    assert contract["scenario_policy"] == (
        "SectionGenerator.sample(base_index,scenario=family_id)"
    )
    assert contract["base_count"] == 100
    assert contract["noise_realizations_per_base"] == 5

    incomplete = _source_lineage()
    del incomplete["pimsr_geogen"]["rock_physics_source_sha256"]
    with pytest.raises(ValueError, match="geogen source lineage"):
        generation_contract_2d(generator_seed=1, source_lineage=incomplete)


def test_forced_family_plan_is_exact_5_by_20_by_5(tmp_path: Path):
    _geometry_value, bases = _bases(tmp_path)
    assert len(bases) == BASE_COUNT == 100
    assert len({base.base_model_id for base in bases}) == 100
    assert len({base.truth.log10_resistivity.tobytes() for base in bases}) == 100
    for family in prediction_lock2d.GEOLOGICAL_FAMILIES:
        selected = [base for base in bases if base.family_id == family]
        assert len(selected) == BASES_PER_FAMILY == 20
        assert all(len(base.source_sample_indices) == 5 for base in selected)
        assert all(len(base.opaque_sample_indices) == 5 for base in selected)
    assert [index for base in bases for index in base.source_sample_indices] == list(
        range(SAMPLE_COUNT)
    )
    assert len({value for base in bases for value in base.opaque_sample_indices}) == 500


def test_noise_is_exactly_keyed_by_base_and_noise_index(tmp_path: Path):
    geometry = _geometry(tmp_path)
    response = _response(geometry, 0)
    first = _noisy_response(
        response, generator_seed=20260830, base_index=7, noise_index=3
    )
    repeat = _noisy_response(
        response, generator_seed=20260830, base_index=7, noise_index=3
    )
    different = _noisy_response(
        response, generator_seed=20260830, base_index=7, noise_index=4
    )
    assert all(np.array_equal(first[name], repeat[name]) for name in first)
    assert any(not np.array_equal(first[name], different[name]) for name in first)
    assert all(values.dtype == np.dtype("<f4") for values in first.values())


def test_core_artifacts_are_schema_v3_evaluator_compatible(tmp_path: Path):
    geometry, bases = _bases(tmp_path)
    forwards = _forwards(tmp_path, geometry, bases)
    public = tmp_path / "public"
    operator = tmp_path / "operator"
    core = _build_core_campaign(
        campaign_id="campaign-a",
        generator_seed=20260830,
        family_nonce=b"n" * 32,
        geometry=geometry,
        mesh=MESH_CONFIGS["nested-production-v1"],
        bases=bases,
        forwards=forwards,
        public_directory=public,
        operator_directory=operator,
    )
    public.mkdir()
    operator.mkdir()
    observation_path = public / "observations.npz"
    manifest_path = public / "observations.public.json"
    truth_path = operator / "truth.npz"
    operator_path = operator / "operator.json"
    observation_path.write_bytes(core.observations_payload)
    manifest_path.write_bytes(core.public_manifest_payload)
    truth_path.write_bytes(core.truth_payload)
    operator_path.write_bytes(core.operator_manifest_payload)

    with zipfile.ZipFile(observation_path) as archive:
        assert all(
            info.compress_type == zipfile.ZIP_DEFLATED for info in archive.infolist()
        )
    truth = evaluation2d.load_truth_2d(
        truth_path, expected_sha256=core.truth_identity.sha256
    )
    assert truth.sample_index.size == SAMPLE_COUNT
    assert np.all(np.diff(truth.sample_index) > 0)
    row_by_sample = {
        int(sample_id): row for row, sample_id in enumerate(truth.sample_index)
    }
    source_order = [
        sample_id for base in bases for sample_id in base.opaque_sample_indices
    ]
    assert truth.sample_index.tolist() != source_order
    for base in bases:
        rows = truth.log10_resistivity[
            [row_by_sample[sample_id] for sample_id in base.opaque_sample_indices]
        ]
        assert all(np.array_equal(rows[0], row) for row in rows[1:])

    observation_snapshot = snapshot_regular_file(
        observation_path,
        expected_sha256=core.observations_identity.sha256,
        role="mock public observations",
    )
    manifest = json.loads(core.public_manifest_payload)
    family_policy = {
        "schema": prediction_lock2d.FAMILY_PARTITION_SCHEMA,
        "schema_version": prediction_lock2d.FAMILY_PARTITION_SCHEMA_VERSION,
        "families": list(prediction_lock2d.GEOLOGICAL_FAMILIES),
        "bases_per_family": 20,
        "noise_realizations_per_base": 5,
        "commitment_contract": dict(prediction_lock2d.FAMILY_COMMITMENT_CONTRACT),
    }
    count, sample_ids, x_axis, depth_axis = prediction_lock2d._observation_identity(
        observation_snapshot, manifest, "campaign-a", family_policy
    )
    assert count == len(sample_ids) == 500
    np.testing.assert_array_equal(x_axis, geometry.x_cell_centers_m)
    np.testing.assert_array_equal(depth_axis, geometry.depth_cell_centers_m)

    operator_manifest = json.loads(core.operator_manifest_payload)
    family_by_sample = evaluation2d._operator_family_partition(
        operator_manifest["split"],
        campaign_id="campaign-a",
        expected_commitment_sha256=core.family_commitment_sha256,
    )
    assert len(family_by_sample) == 500
    assert set(family_by_sample.values()) == set(prediction_lock2d.GEOLOGICAL_FAMILIES)
    for forbidden in (
        str(20260830).encode(),
        b"generator_seed",
        b"base_model_id",
        b"noise_index",
        b"source_generator_sample_index",
        b"key_material",
        b"nonce_hex",
    ):
        assert forbidden not in core.public_manifest_payload


def test_hidden_closure_has_100_material_four_artifact_refs(tmp_path: Path):
    geometry, bases = _bases(tmp_path)
    forwards = _forwards(tmp_path, geometry, bases)
    public = tmp_path / "public"
    operator = tmp_path / "operator"
    core = _build_core_campaign(
        campaign_id="campaign-a",
        generator_seed=20260830,
        family_nonce=b"n" * 32,
        geometry=geometry,
        mesh=MESH_CONFIGS["nested-production-v1"],
        bases=bases,
        forwards=forwards,
        public_directory=public,
        operator_directory=operator,
    )
    contract = generation_contract_2d(
        generator_seed=20260830, source_lineage=_source_lineage()
    )
    plan = _build_final_evidence_plan(
        campaign_id="campaign-a",
        generator_seed=20260830,
        generation_contract=contract,
        geometry=geometry,
        mesh=MESH_CONFIGS["nested-production-v1"],
        runtime=_runtime(),
        generation_runtime=FROZEN_GENERATION_RUNTIME,
        generation_runtime_manifest=_runtime_manifest(tmp_path),
        bases=bases,
        forwards=forwards,
        core=core,
        operator_directory=operator,
    )
    closure = json.loads(plan.closure_payload)
    assert closure["schema"] == CAMPAIGN_SCHEMA
    assert closure["schema_version"] == CAMPAIGN_SCHEMA_VERSION
    assert closure["generation_runtime"] == FROZEN_GENERATION_RUNTIME
    assert set(closure["generation_runtime_manifest"]) == {
        "path",
        "sha256",
        "size_bytes",
    }
    assert len(closure["base_forward_runs"]) == 100
    assert len(closure["noise_rows"]) == 500
    for index, row in enumerate(closure["base_forward_runs"]):
        assert row["base_layer_rng_key"] == [20260830, index]
        assert row["section_rng_key"] == [20260830, 2, index]
        assert {"model", "template", "forward", "provenance"}.issubset(row)
        assert all(
            set(row[name]) == {"path", "sha256", "size_bytes"}
            for name in ("model", "template", "forward", "provenance")
        )
    assert {tuple(row["noise_rng_key"]) for row in closure["noise_rows"]} == {
        (20260830, 3, base_index, noise_index)
        for base_index in range(100)
        for noise_index in range(5)
    }

    _publish_operator_directory(
        operator_directory=operator,
        core=core,
        forwards=forwards,
        evidence=plan,
    )
    _verify_published_operator_directory(
        operator_directory=operator,
        core=core,
        evidence=plan,
    )
    tampered = operator / "modem" / "base-000" / "forward.dat"
    os.chmod(tampered, 0o666)
    tampered.write_bytes(b"tampered")
    with pytest.raises(HiddenCampaign2DError, match="differs from its final plan"):
        _verify_published_operator_directory(
            operator_directory=operator,
            core=core,
            evidence=plan,
        )


def test_work_contract_resume_rejects_secret_or_campaign_change(tmp_path: Path):
    geometry = _geometry(tmp_path)
    mesh = MESH_CONFIGS["nested-production-v1"]
    runtime = _runtime()
    first = _work_contract(
        campaign_id="campaign-a",
        generator_seed=20260830,
        generation_contract_sha256="a" * 64,
        geometry=geometry,
        mesh=mesh,
        runtime=runtime,
        generation_runtime=FROZEN_GENERATION_RUNTIME,
        generation_runtime_manifest=_runtime_manifest(tmp_path),
        sample_id_key=b"k" * 32,
        family_nonce=b"n" * 32,
    )
    work = _prepare_work_directory(tmp_path / "work", first)
    assert _prepare_work_directory(work, first) == work
    changed = dict(first)
    changed["sample_id_key_commitment_sha256"] = "b" * 64
    with pytest.raises(HiddenCampaign2DError, match="different hidden campaign"):
        _prepare_work_directory(work, changed)


def test_runtime_manifest_is_new_only_pinned_and_recomputed(monkeypatch, tmp_path: Path):
    value = _runtime_manifest_value()
    monkeypatch.setattr(
        hidden_campaign, "_runtime_manifest_value", lambda _lineage: value
    )
    path = tmp_path / "runtime.json"
    identity = build_hidden_generation_runtime_manifest_2d(
        path, source_lineage=_source_lineage()
    )
    validated = validate_hidden_generation_runtime_manifest_2d(
        path,
        expected_sha256=identity.sha256,
        expected_size_bytes=identity.size_bytes,
    )
    assert validated.identity == identity
    assert validated.value == value
    assert path.read_bytes().endswith(b"\n")
    with pytest.raises(HiddenCampaign2DError, match="external pin"):
        validate_hidden_generation_runtime_manifest_2d(
            path,
            expected_sha256="f" * 64,
            expected_size_bytes=identity.size_bytes,
        )
    with pytest.raises(FileExistsError, match="overwrite"):
        build_hidden_generation_runtime_manifest_2d(
            path, source_lineage=_source_lineage()
        )


def test_runtime_manifest_hashes_ordered_installed_trees(monkeypatch, tmp_path: Path):
    roots = []
    for index, package in enumerate(("numpy", "h5py", "pimsr_geogen", "pimsr_forward")):
        root = tmp_path / package
        root.mkdir()
        (root / "z.bin").write_bytes(bytes([index, 1]))
        (root / "a.py").write_text(f"PACKAGE = {package!r}\n", encoding="utf-8")
        roots.append((package, root))
    monkeypatch.setattr(
        hidden_campaign,
        "_generation_runtime",
        lambda: dict(FROZEN_GENERATION_RUNTIME),
    )
    monkeypatch.setattr(hidden_campaign, "_verify_generation_sources", lambda _value: ())
    monkeypatch.setattr(
        hidden_campaign, "_installed_distribution_roots", lambda: tuple(roots)
    )
    first = hidden_campaign._runtime_manifest_value(_source_lineage())
    assert set(first) == {
        "schema",
        "schema_version",
        "python",
        "distributions",
        "source_closure",
        "tree_manifest_sha256",
    }
    assert list(first["distributions"]) == [
        "numpy",
        "h5py",
        "pimsr_geogen",
        "pimsr_forward",
    ]
    assert all(
        set(record) == {"version", "installed_tree_sha256"}
        for record in first["distributions"].values()
    )
    (roots[-1][1] / "z.bin").write_bytes(b"changed")
    second = hidden_campaign._runtime_manifest_value(_source_lineage())
    assert (
        second["distributions"]["pimsr_forward"]["installed_tree_sha256"]
        != (first["distributions"]["pimsr_forward"]["installed_tree_sha256"])
    )
    assert second["tree_manifest_sha256"] != first["tree_manifest_sha256"]


def test_runtime_manifest_resolves_stable_python_launcher_symlink(
    monkeypatch, tmp_path: Path
):
    executable = tmp_path / "python-real"
    executable.write_bytes(b"exact interpreter bytes")
    launcher = tmp_path / "python"
    try:
        launcher.symlink_to(executable)
    except OSError:
        pytest.skip("creating symlinks is unavailable")
    monkeypatch.setattr(hidden_campaign.sys, "executable", str(launcher))

    snapshot = hidden_campaign._snapshot_python_executable()

    assert snapshot.path == executable.resolve(strict=True)
    assert snapshot.payload == b"exact interpreter bytes"


def test_materializer_never_calls_simpeg_forward(monkeypatch, tmp_path: Path):
    geometry, bases = _bases(tmp_path)
    work = tmp_path / "work"
    (work / "base-forward-runs").mkdir(parents=True)
    called = []

    def forbidden_runner(**_kwargs):
        called.append("modem")
        raise RuntimeError("mock stop before any solve")

    monkeypatch.setattr(
        "pimsr_benchmarks.hidden_campaign2d.run_modem_forward", forbidden_runner
    )
    runtime = SimpleNamespace(require_unchanged=lambda: None)
    with pytest.raises(RuntimeError, match="mock stop"):
        _materialize_raw_forwards(
            bases=bases,
            work=work,
            campaign_id="campaign-a",
            generator_seed=20260830,
            generation_contract_sha256="a" * 64,
            geometry=geometry,
            mesh=MESH_CONFIGS["nested-production-v1"],
            runtime=runtime,
            timeout_seconds=1.0,
            progress=None,
        )
    assert called == ["modem"]


def test_resume_revalidates_every_existing_base_without_solving(monkeypatch, tmp_path):
    geometry, bases = _bases(tmp_path)
    expected = _forwards(tmp_path, geometry, bases)
    by_id = {item.base.base_model_id: item for item in expected}
    raw_root = tmp_path / "work" / "base-forward-runs"
    for base in bases:
        (raw_root / base.base_model_id).mkdir(parents=True)
    verified_ids = []

    def verify(path, *, base, **_kwargs):
        assert path == raw_root / base.base_model_id
        verified_ids.append(base.base_model_id)
        return by_id[base.base_model_id]

    def forbidden_runner(**_kwargs):
        raise AssertionError("resume must not repeat a completed ModEM solve")

    unchanged_calls = []
    runtime = SimpleNamespace(require_unchanged=lambda: unchanged_calls.append("runtime"))
    monkeypatch.setattr(hidden_campaign, "_verify_forward_bundle", verify)
    monkeypatch.setattr(hidden_campaign, "run_modem_forward", forbidden_runner)
    resumed = _materialize_raw_forwards(
        bases=bases,
        work=tmp_path / "work",
        campaign_id="campaign-a",
        generator_seed=20260830,
        generation_contract_sha256="a" * 64,
        geometry=geometry,
        mesh=MESH_CONFIGS["nested-production-v1"],
        runtime=runtime,
        timeout_seconds=1.0,
        progress=None,
    )
    assert resumed == expected
    assert verified_ids == [base.base_model_id for base in bases]
    assert len(unchanged_calls) == BASE_COUNT


def test_fresh_campaign_runs_exactly_one_modem_solve_per_base(monkeypatch, tmp_path):
    geometry, bases = _bases(tmp_path)
    expected = _forwards(tmp_path, geometry, bases)
    by_id = {item.base.base_model_id: item for item in expected}
    work = tmp_path / "work"
    (work / "base-forward-runs").mkdir(parents=True)
    solved_ids = []

    def runner(*, output_dir, truth, **_kwargs):
        output_dir.mkdir()
        solved_ids.append(truth.sample_id)
        return output_dir, None, None

    def verify(_path, *, base, **_kwargs):
        return by_id[base.base_model_id]

    monkeypatch.setattr(hidden_campaign, "run_modem_forward", runner)
    monkeypatch.setattr(hidden_campaign, "_verify_forward_bundle", verify)
    runtime = SimpleNamespace(require_unchanged=lambda: None)
    materialized = _materialize_raw_forwards(
        bases=bases,
        work=work,
        campaign_id="campaign-a",
        generator_seed=20260830,
        generation_contract_sha256="a" * 64,
        geometry=geometry,
        mesh=MESH_CONFIGS["nested-production-v1"],
        runtime=runtime,
        timeout_seconds=1.0,
        progress=None,
    )
    assert materialized == expected
    assert solved_ids == [base.base_model_id for base in bases]
    assert len(solved_ids) == BASE_COUNT == 100
