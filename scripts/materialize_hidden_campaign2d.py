#!/usr/bin/env python
"""Build the pinned generation-runtime manifest or resume a hidden 2-D campaign."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from pimsr_benchmarks.hidden_campaign2d import (
    HiddenCampaign2DError,
    _strict_json_payload,
    _strict_source_lineage,
    build_hidden_generation_runtime_manifest_2d,
    materialize_hidden_campaign2d,
)
from pimsr_benchmarks.modem2d_forward import (
    MESH_CONFIGS,
    NestedMeshConfig,
    require_snapshot_unchanged,
    snapshot_file,
    verify_pinned_runtime,
)

NESTED_MESH_CONFIGS = {
    name: mesh
    for name, mesh in MESH_CONFIGS.items()
    if isinstance(mesh, NestedMeshConfig)
}


def _add_pinned_source_lineage(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-lineage-json", type=Path, required=True)
    parser.add_argument("--source-lineage-sha256", required=True)
    parser.add_argument("--source-lineage-size-bytes", type=int, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed tooling for a 100-base, 500-row hidden ModEM 2-D campaign. "
            "No production generation occurs without the campaign subcommand."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    runtime = commands.add_parser(
        "runtime-manifest",
        help="capture the exact generation environment before preregistration",
    )
    _add_pinned_source_lineage(runtime)
    runtime.add_argument("--output-json", type=Path, required=True)

    campaign = commands.add_parser(
        "campaign",
        help="resume exactly one ModEM solve per base and publish separated outputs",
    )
    campaign.add_argument("--campaign-id", required=True)
    campaign.add_argument(
        "--generator-seeds-file",
        type=Path,
        required=True,
        help=(
            "operator-only canonical JSON array containing all five committed seeds; "
            "the campaign id selects its corresponding entry"
        ),
    )
    campaign.add_argument("--preregistration-json", type=Path, required=True)
    campaign.add_argument("--preregistration-sha256", required=True)
    campaign.add_argument("--preregistration-size-bytes", type=int, required=True)
    campaign.add_argument("--convergence-report-json", type=Path, required=True)
    campaign.add_argument("--convergence-residuals-npz", type=Path, required=True)
    campaign.add_argument("--geometry-h5", type=Path, required=True)
    campaign.add_argument("--geometry-sha256", required=True)
    _add_pinned_source_lineage(campaign)
    campaign.add_argument("--generation-runtime-manifest", type=Path, required=True)
    campaign.add_argument("--generation-runtime-manifest-sha256", required=True)
    campaign.add_argument(
        "--generation-runtime-manifest-size-bytes", type=int, required=True
    )
    campaign.add_argument("--sample-id-key-file", type=Path, required=True)
    campaign.add_argument("--family-nonce-file", type=Path, required=True)
    campaign.add_argument("--modem-repo", type=Path, required=True)
    campaign.add_argument("--build-root", type=Path, required=True)
    campaign.add_argument("--docker", default="docker")
    campaign.add_argument("--mesh", choices=sorted(NESTED_MESH_CONFIGS), required=True)
    campaign.add_argument("--work-dir", type=Path, required=True)
    campaign.add_argument("--public-output-dir", type=Path, required=True)
    campaign.add_argument("--operator-output-dir", type=Path, required=True)
    campaign.add_argument("--timeout-seconds", type=float, default=1_800.0)
    return parser


def _pinned_source_lineage(args: argparse.Namespace) -> tuple[dict[str, Any], Any]:
    snapshot = snapshot_file(args.source_lineage_json, role="source lineage manifest")
    if (
        snapshot.sha256 != args.source_lineage_sha256
        or snapshot.size_bytes != args.source_lineage_size_bytes
    ):
        raise HiddenCampaign2DError("source lineage differs from its external pin")
    value = _strict_json_payload(snapshot.payload, role="source lineage manifest")
    if not isinstance(value, Mapping):
        raise HiddenCampaign2DError("source lineage manifest must be an object")
    return _strict_source_lineage(value), snapshot


def _pinned_snapshot(
    path: Path, *, sha256: object, size_bytes: object, role: str
) -> Any:
    snapshot = snapshot_file(path, role=role)
    if snapshot.sha256 != sha256 or snapshot.size_bytes != size_bytes:
        raise HiddenCampaign2DError(f"{role} differs from preregistration")
    return snapshot


def _campaign_authorization(
    args: argparse.Namespace, *, mesh: NestedMeshConfig
) -> tuple[int, bytes, bytes, tuple[Any, ...]]:
    prereg_snapshot = _pinned_snapshot(
        args.preregistration_json,
        sha256=args.preregistration_sha256,
        size_bytes=args.preregistration_size_bytes,
        role="locked preregistration",
    )
    prereg = _strict_json_payload(
        prereg_snapshot.payload, role="locked preregistration"
    )
    if (
        prereg.get("schema") != "pimsr-sota-2d-common-retrain-preregistration"
        or prereg.get("schema_version") != 1
        or prereg.get("status")
        != "locked_before_hidden_materialization_and_method_execution"
    ):
        raise HiddenCampaign2DError("preregistration is not locked for hidden generation")
    datasets = prereg.get("datasets")
    hidden = datasets.get("hidden_test") if isinstance(datasets, Mapping) else None
    campaigns = hidden.get("campaigns") if isinstance(hidden, Mapping) else None
    if not isinstance(campaigns, Mapping):
        raise HiddenCampaign2DError("preregistration hidden campaign contract is missing")
    campaign_ids = campaigns.get("campaign_ids")
    if (
        not isinstance(campaign_ids, list)
        or len(campaign_ids) != 5
        or len(set(campaign_ids)) != 5
        or args.campaign_id not in campaign_ids
        or campaigns.get("count") != 5
        or campaigns.get("samples_per_campaign") != 500
        or campaigns.get("total_samples") != 2500
    ):
        raise HiddenCampaign2DError("campaign is not authorized by preregistration")

    seed_commitment = hidden.get("seed_commitment")
    if not isinstance(seed_commitment, Mapping) or seed_commitment.get("encoding") != (
        "utf8-canonical-json-int64-array-no-newline-v1"
    ):
        raise HiddenCampaign2DError("hidden seed commitment contract is invalid")
    seeds_snapshot = snapshot_file(
        args.generator_seeds_file, role="committed hidden generator seed reveal"
    )
    if seeds_snapshot.sha256 != seed_commitment.get("sha256"):
        raise HiddenCampaign2DError(
            "hidden generator seed reveal does not open the preregistered commitment"
        )
    try:
        seeds = json.loads(seeds_snapshot.payload.decode("ascii", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HiddenCampaign2DError("hidden generator seeds are not strict JSON") from exc
    canonical_seeds = json.dumps(
        seeds, ensure_ascii=True, allow_nan=False, separators=(",", ":")
    ).encode("ascii")
    if (
        seeds_snapshot.payload != canonical_seeds
        or not isinstance(seeds, list)
        or len(seeds) != 5
        or len(set(seeds)) != 5
        or any(
            type(seed) is not int or seed < 0 or seed > np.iinfo(np.int64).max
            for seed in seeds
        )
    ):
        raise HiddenCampaign2DError("hidden generator seed reveal is not canonical")

    evidence = prereg.get("headline_evidence")
    if not isinstance(evidence, Mapping):
        raise HiddenCampaign2DError("preregistered headline evidence is missing")
    convergence = evidence.get("public_mesh_convergence")
    generator = evidence.get("hidden_observation_generator")
    if not isinstance(convergence, Mapping) or not isinstance(generator, Mapping):
        raise HiddenCampaign2DError("preregistered ModEM qualification is missing")
    report_snapshot = _pinned_snapshot(
        args.convergence_report_json,
        sha256=convergence.get("report_sha256"),
        size_bytes=convergence.get("report_size_bytes"),
        role="public mesh convergence report",
    )
    residuals_snapshot = _pinned_snapshot(
        args.convergence_residuals_npz,
        sha256=convergence.get("residuals_sha256"),
        size_bytes=convergence.get("residuals_size_bytes"),
        role="public mesh convergence residuals",
    )
    report = _strict_json_payload(
        report_snapshot.payload, role="public mesh convergence report"
    )
    production = report.get("production_candidate")
    raw_residuals = report.get("raw_paired_residuals")
    if (
        report.get("schema") != "pimsr-modem2d-convergence-validation"
        or report.get("schema_version") != 1
        or report.get("passed") is not True
        or report.get("headline_eligible") is not True
        or report.get("scope") != "public_only_no_hidden_or_secret_access"
        or not isinstance(production, Mapping)
        or production.get("mesh_config_sha256") != mesh.sha256
        or generator.get("mesh_artifact_sha256") != mesh.sha256
        or not isinstance(raw_residuals, Mapping)
        or raw_residuals.get("sha256") != residuals_snapshot.sha256
        or raw_residuals.get("size_bytes") != residuals_snapshot.size_bytes
    ):
        raise HiddenCampaign2DError(
            "public ModEM mesh qualification does not authorize the selected mesh"
        )

    key_snapshot = snapshot_file(args.sample_id_key_file, role="sample id key")
    sample_contract = hidden.get("sample_id_contract")
    if (
        not isinstance(sample_contract, Mapping)
        or key_snapshot.sha256 != sample_contract.get("key_commitment_sha256")
        or len(key_snapshot.payload) < 32
    ):
        raise HiddenCampaign2DError("sample id key does not open its preregistered commitment")
    nonce_snapshot = snapshot_file(args.family_nonce_file, role="family reveal nonce")
    if len(nonce_snapshot.payload) != 32:
        raise HiddenCampaign2DError("family reveal nonce must contain exactly 32 bytes")
    selected_seed = int(seeds[campaign_ids.index(args.campaign_id)])
    return (
        selected_seed,
        key_snapshot.payload,
        nonce_snapshot.payload,
        (
            prereg_snapshot,
            seeds_snapshot,
            report_snapshot,
            residuals_snapshot,
            key_snapshot,
            nonce_snapshot,
        ),
    )


def _runtime_manifest(args: argparse.Namespace) -> int:
    lineage, lineage_snapshot = _pinned_source_lineage(args)
    identity = build_hidden_generation_runtime_manifest_2d(
        args.output_json, source_lineage=lineage
    )
    require_snapshot_unchanged(lineage_snapshot, role="source lineage manifest")
    print(
        json.dumps(
            {
                "path": str(identity.path),
                "sha256": identity.sha256,
                "size_bytes": identity.size_bytes,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _campaign(args: argparse.Namespace) -> int:
    lineage, lineage_snapshot = _pinned_source_lineage(args)
    selected_mesh = NESTED_MESH_CONFIGS[args.mesh]
    generator_seed, sample_id_key, family_nonce, authorization_snapshots = (
        _campaign_authorization(args, mesh=selected_mesh)
    )
    runtime = verify_pinned_runtime(
        modem_repo=args.modem_repo,
        build_root=args.build_root,
        docker_executable=args.docker,
    )

    def progress(completed: int, total: int) -> None:
        print(f"verified ModEM base {completed}/{total}", flush=True)

    result = materialize_hidden_campaign2d(
        campaign_id=args.campaign_id,
        generator_seed=generator_seed,
        geometry_h5=args.geometry_h5,
        expected_geometry_sha256=args.geometry_sha256,
        source_lineage=lineage,
        generation_runtime_manifest_path=args.generation_runtime_manifest,
        expected_generation_runtime_manifest_sha256=(
            args.generation_runtime_manifest_sha256
        ),
        expected_generation_runtime_manifest_size_bytes=(
            args.generation_runtime_manifest_size_bytes
        ),
        sample_id_key=sample_id_key,
        family_nonce=family_nonce,
        runtime=runtime,
        mesh=selected_mesh,
        work_dir=args.work_dir,
        public_output_dir=args.public_output_dir,
        operator_output_dir=args.operator_output_dir,
        timeout_seconds=args.timeout_seconds,
        progress=progress,
    )
    require_snapshot_unchanged(lineage_snapshot, role="source lineage manifest")
    for snapshot in authorization_snapshots:
        require_snapshot_unchanged(snapshot, role="hidden campaign authorization input")
    print(
        json.dumps(
            {
                "campaign_id": result.campaign_id,
                "public_directory": str(result.public_directory),
                "operator_directory": str(result.operator_directory),
                "observations_sha256": result.observations.sha256,
                "public_manifest_sha256": result.public_manifest.sha256,
                "truth_sha256": result.truth.sha256,
                "operator_manifest_sha256": result.operator_manifest.sha256,
                "family_reveal_sha256": result.family_reveal.sha256,
                "hidden_generation_sha256": result.hidden_generation.sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "runtime-manifest":
        return _runtime_manifest(args)
    if args.command == "campaign":
        return _campaign(args)
    raise AssertionError("argparse accepted an unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
