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

_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "validate_modem2d_convergence.py"
)
_SPEC = importlib.util.spec_from_file_location("validate_modem2d_convergence", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
validator = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = validator
_SPEC.loader.exec_module(validator)


def test_raw_run_bundle_uses_portable_pinned_snapshot_payloads(tmp_path: Path):
    output = tmp_path / "case"
    output.mkdir()
    forward_payload = b"# deterministic ModEM forward\n"
    provenance_payload = b'{"schema":"test"}\n'
    (output / "forward.dat").write_bytes(forward_payload)
    (output / "provenance.json").write_bytes(provenance_payload)

    run, artifacts = validator._bundle_raw_run(
        ordinal=7,
        case_id="public:42:production-candidate",
        case_kind="public_geology",
        sample_index=42,
        truth_id="sample-000042",
        role="production-candidate",
        mesh_id="nested-production-v1",
        output_dir=output,
    )

    assert set(run) == {
        "case_id",
        "case_kind",
        "sample_index",
        "truth_id",
        "role",
        "mesh_id",
        "forward",
        "provenance",
    }
    assert run["forward"] == {
        "path": "raw-007-forward.dat",
        "sha256": hashlib.sha256(forward_payload).hexdigest(),
        "size_bytes": len(forward_payload),
    }
    assert run["provenance"] == {
        "path": "raw-007-provenance.json",
        "sha256": hashlib.sha256(provenance_payload).hexdigest(),
        "size_bytes": len(provenance_payload),
    }
    assert artifacts == {
        "raw-007-forward.dat": forward_payload,
        "raw-007-provenance.json": provenance_payload,
    }

    (output / "forward.dat").write_bytes(b"mutated after snapshot")
    assert artifacts["raw-007-forward.dat"] == forward_payload


def test_raw_run_set_contract_is_exact():
    assert validator.RAW_RUN_SET_SCHEMA == (
        "pimsr-modem2d-public-convergence-raw-run-set"
    )
    assert validator.RAW_RUN_SET_SCHEMA_VERSION == 1


def _cached_case_fixture(tmp_path: Path) -> tuple[Path, object, object, dict, dict]:
    cache_root = tmp_path / "cache"
    output = cache_root / "cases" / "sample-000042" / "production-candidate"
    output.mkdir(parents=True)
    payloads = {
        name: f"{name}\n".encode()
        for name in validator._CACHED_RUN_FILES
        if name != "provenance.json"
    }
    for name, payload in payloads.items():
        (output / name).write_bytes(payload)
    source = {
        "source": {"path": "public.h5", "sha256": "1" * 64, "size_bytes": 10},
        "row": 0,
        "sample_index": 42,
        "scenario_index": 1,
        "generator_seed": validator.PUBLIC_GENERATOR_SEED,
        "generation_contract": validator.PUBLIC_GENERATION_CONTRACT,
        "forward_contract": validator.PUBLIC_FORWARD_CONTRACT,
        "public_validation": {
            "selection_policy": "lowest sample_index per frozen scenario",
            "scenario_name": "aquifer",
            "validator_source": {
                "path": "current-validator.py",
                "sha256": "2" * 64,
                "size_bytes": 100,
            },
        },
    }
    cached_source = json.loads(json.dumps(source))
    cached_source["public_validation"]["validator_source"] = {
        "path": "historical-validator.py",
        "sha256": "3" * 64,
        "size_bytes": 99,
    }
    truth_record = {"schema": "test-truth", "sample_id": "sample-000042"}
    truth = SimpleNamespace(identity_record=lambda: truth_record)
    runtime = SimpleNamespace(record={"runtime": "pinned"}, identity_sha256="4" * 64)
    bridge = {"path": "current-bridge.py", "sha256": "5" * 64, "size_bytes": 77}
    response_record = {"artifact": {"sha256": "6" * 64, "size_bytes": 1}}
    provenance = {
        "schema": "pimsr-modem2d-forward-run",
        "schema_version": 1,
        "truth": truth_record,
        "truth_source": cached_source,
        "mesh": {
            **validator.PRODUCTION_MESH.canonical_record(),
            "mesh_config_sha256": validator.PRODUCTION_MESH.sha256,
        },
        "runtime": runtime.record,
        "runtime_identity_sha256": runtime.identity_sha256,
        "bridge_source": {
            "path": "historical-bridge.py",
            "sha256": bridge["sha256"],
            "size_bytes": bridge["size_bytes"],
        },
        "input_contract": {},
        "response_contract": response_record,
        "execution": {"returncode": 0},
        "outputs": {
            name: {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
            for name, payload in payloads.items()
        },
    }
    (output / "provenance.json").write_bytes(
        (json.dumps(provenance, indent=2, sort_keys=True) + "\n").encode()
    )
    return cache_root, truth, runtime, source, bridge


def test_run_case_reuses_only_a_fully_validated_historical_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cache_root, truth, runtime, source, bridge = _cached_case_fixture(tmp_path)
    response = SimpleNamespace()
    response_record = {"artifact": {"sha256": "6" * 64, "size_bytes": 1}}
    monkeypatch.setattr(
        validator, "parse_modem_response", lambda *_args: (response, response_record)
    )
    monkeypatch.setattr(
        validator,
        "run_modem_forward",
        lambda **_kwargs: pytest.fail("validated cached run should avoid a new solve"),
    )

    result = validator._run_case(
        runtime=runtime,
        truth=truth,
        selection=validator.PublicSelection(Path("public.h5"), 0, 42, 1, "aquifer", {}),
        source=source,
        mesh=validator.PRODUCTION_MESH,
        role="production-candidate",
        work_root=tmp_path / "new-work",
        timeout_seconds=10.0,
        reuse_roots=(cache_root,),
        bridge_identity=bridge,
    )

    assert result.response is response
    assert result.output_dir == (
        cache_root / "cases" / "sample-000042" / "production-candidate"
    ).resolve()


def test_cached_run_rejects_any_artifact_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cache_root, truth, runtime, source, bridge = _cached_case_fixture(tmp_path)
    output = cache_root / "cases" / "sample-000042" / "production-candidate"
    (output / "forward.dat").write_bytes(b"tampered")
    monkeypatch.setattr(
        validator,
        "parse_modem_response",
        lambda *_args: pytest.fail("mismatched output pins must fail before parsing"),
    )

    with pytest.raises(ValueError, match="provenance does not match"):
        validator._load_cached_case(
            cache_root=cache_root,
            runtime=runtime,
            truth=truth,
            selection=validator.PublicSelection(
                Path("public.h5"), 0, 42, 1, "aquifer", {}
            ),
            source=source,
            mesh=validator.PRODUCTION_MESH,
            role="production-candidate",
            bridge_identity=bridge,
        )


def _write_public_catalog_shard(path: Path) -> None:
    with h5py.File(path, "w") as h5:
        h5.attrs["schema"] = "pimsr-mt-2d"
        h5.attrs["schema_version"] = 2
        h5.attrs["generation_complete"] = 1
        h5.attrs["generator_seed"] = validator.PUBLIC_GENERATOR_SEED
        h5.attrs["generation_contract"] = validator.PUBLIC_GENERATION_CONTRACT
        h5.attrs["forward_contract"] = validator.PUBLIC_FORWARD_CONTRACT
        h5.attrs["scenario_order"] = np.asarray(validator.SCENARIO_NAMES, dtype="S16")
        h5.attrs["mode_order"] = np.asarray(["te", "tm"], dtype="S2")
        h5.attrs["impedance_components"] = np.asarray(["Zyx", "Zxy"], dtype="S3")
        h5.create_dataset("sample_index", data=np.asarray([42], dtype=np.int64))
        h5.create_dataset("scenario", data=np.asarray([1], dtype=np.int64))


def test_public_catalog_parses_the_pinned_snapshot_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    shard = tmp_path / "public.h5"
    _write_public_catalog_shard(shard)
    snapshot = validator.snapshot_file(shard, role="test public shard")
    shard.write_bytes(b"replacement pathname payload")
    monkeypatch.setattr(validator, "snapshot_file", lambda *_args, **_kwargs: snapshot)
    monkeypatch.setattr(
        validator, "require_snapshot_unchanged", lambda *_args, **_kwargs: None
    )

    selections, record = validator._catalog_public_shard(shard)

    assert [(item.sample_index, item.scenario_name) for item in selections] == [
        (42, "aquifer")
    ]
    assert record["sha256"] == snapshot.sha256


def test_validate_rejects_nonexact_family_count_before_any_expensive_work():
    args = SimpleNamespace(per_family=6, jobs=1, timeout_seconds=3_600.0)

    with pytest.raises(ValueError, match=r"exact 25x3\+4\+1"):
        validator.validate(args)
