from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

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
