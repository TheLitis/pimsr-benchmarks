from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from pimsr_benchmarks.hidden_campaign2d import HiddenCampaign2DError
from pimsr_benchmarks.modem2d_forward import MESH_CONFIGS


def _load_cli():
    path = Path(__file__).parents[1] / "scripts" / "materialize_hidden_campaign2d.py"
    spec = importlib.util.spec_from_file_location("materialize_hidden_campaign2d_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, payload: bytes) -> tuple[str, int]:
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest(), len(payload)


def _authorized_inputs(tmp_path: Path, *, report_passed: bool = True):
    mesh = MESH_CONFIGS["nested-production-v1"]
    seeds_payload = b"[701,702,703,704,705]"
    seeds_sha, _ = _write(tmp_path / "seeds.json", seeds_payload)
    key_payload = b"k" * 32
    key_sha, _ = _write(tmp_path / "sample-key.bin", key_payload)
    nonce_payload = b"n" * 32
    _write(tmp_path / "nonce.bin", nonce_payload)
    residuals_payload = b"frozen-residuals"
    residuals_sha, residuals_size = _write(
        tmp_path / "paired-residuals.npz", residuals_payload
    )
    report = {
        "schema": "pimsr-modem2d-convergence-validation",
        "schema_version": 1,
        "passed": report_passed,
        "headline_eligible": report_passed,
        "scope": "public_only_no_hidden_or_secret_access",
        "production_candidate": {"mesh_config_sha256": mesh.sha256},
        "raw_paired_residuals": {
            "sha256": residuals_sha,
            "size_bytes": residuals_size,
        },
    }
    report_payload = json.dumps(report, sort_keys=True).encode()
    report_sha, report_size = _write(tmp_path / "convergence-report.json", report_payload)
    prereg = {
        "schema": "pimsr-sota-2d-common-retrain-preregistration",
        "schema_version": 1,
        "status": "locked_before_hidden_materialization_and_method_execution",
        "datasets": {
            "hidden_test": {
                "campaigns": {
                    "campaign_ids": [f"campaign-{index}" for index in range(5)],
                    "count": 5,
                    "samples_per_campaign": 500,
                    "total_samples": 2500,
                },
                "seed_commitment": {
                    "encoding": "utf8-canonical-json-int64-array-no-newline-v1",
                    "sha256": seeds_sha,
                },
                "sample_id_contract": {"key_commitment_sha256": key_sha},
            }
        },
        "headline_evidence": {
            "hidden_observation_generator": {"mesh_artifact_sha256": mesh.sha256},
            "public_mesh_convergence": {
                "report_sha256": report_sha,
                "report_size_bytes": report_size,
                "residuals_sha256": residuals_sha,
                "residuals_size_bytes": residuals_size,
            },
        },
    }
    prereg_payload = json.dumps(prereg, sort_keys=True).encode()
    prereg_sha, prereg_size = _write(tmp_path / "prereg.json", prereg_payload)
    args = argparse.Namespace(
        campaign_id="campaign-2",
        generator_seeds_file=tmp_path / "seeds.json",
        preregistration_json=tmp_path / "prereg.json",
        preregistration_sha256=prereg_sha,
        preregistration_size_bytes=prereg_size,
        convergence_report_json=tmp_path / "convergence-report.json",
        convergence_residuals_npz=tmp_path / "paired-residuals.npz",
        sample_id_key_file=tmp_path / "sample-key.bin",
        family_nonce_file=tmp_path / "nonce.bin",
    )
    return args, mesh, key_payload, nonce_payload


def test_campaign_authorization_precedes_expensive_generation(tmp_path: Path):
    cli = _load_cli()
    args, mesh, key, nonce = _authorized_inputs(tmp_path)
    seed, captured_key, captured_nonce, snapshots = cli._campaign_authorization(
        args, mesh=mesh
    )
    assert seed == 703
    assert captured_key == key
    assert captured_nonce == nonce
    assert len(snapshots) == 6


def test_campaign_authorization_rejects_failed_mesh_qualification(tmp_path: Path):
    cli = _load_cli()
    args, mesh, _key, _nonce = _authorized_inputs(tmp_path, report_passed=False)
    with pytest.raises(HiddenCampaign2DError, match="does not authorize"):
        cli._campaign_authorization(args, mesh=mesh)


def test_campaign_authorization_rejects_uncommitted_sample_key(tmp_path: Path):
    cli = _load_cli()
    args, mesh, _key, _nonce = _authorized_inputs(tmp_path)
    args.sample_id_key_file.write_bytes(b"x" * 32)
    with pytest.raises(HiddenCampaign2DError, match="does not open"):
        cli._campaign_authorization(args, mesh=mesh)
