from __future__ import annotations

import hashlib
import importlib.util
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
