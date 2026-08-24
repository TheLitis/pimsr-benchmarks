#!/usr/bin/env python
"""Public-only convergence qualification for the pinned ModEM 2-D bridge."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from pimsr_benchmarks.modem2d_forward import (
    CONVERGENCE_GATES,
    MESH_CONFIGS,
    CanonicalTruth,
    MeshConfig,
    ModEMResponse,
    NestedMeshConfig,
    analytic_response_for_mapped_1d,
    circular_phase_error_deg,
    gate_summary,
    load_canonical_hdf5,
    publish_artifact_bundle,
    require_snapshot_unchanged,
    run_modem_forward,
    snapshot_file,
    summarize_absolute,
    verify_pinned_runtime,
)

PUBLIC_GENERATOR_SEED = 20260820
PUBLIC_GENERATION_CONTRACT = "pimsr-geogen.SectionGenerator/default-grid/v1"
PUBLIC_FORWARD_CONTRACT = "pimsr-forward.MT2DForward/default-mesh/v2"
SCENARIO_NAMES = ("background", "aquifer", "hydrocarbon", "salt", "geothermal")
PUBLIC_SAMPLES_PER_FAMILY = 5
PRODUCTION_MESH = MESH_CONFIGS["nested-production-v1"]
NEXT_FINER_REFERENCE = MESH_CONFIGS["nested-reference-x2-v1"]
RAW_RUN_SET_SCHEMA = "pimsr-modem2d-public-convergence-raw-run-set"
RAW_RUN_SET_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PublicSelection:
    path: Path
    row: int
    sample_index: int
    scenario_index: int
    scenario_name: str
    source_record: dict[str, object]


@dataclass(frozen=True)
class CaseResult:
    selection: PublicSelection
    role: str
    mesh: MeshConfig | NestedMeshConfig
    output_dir: Path
    response: ModEMResponse
    provenance: dict[str, object]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the frozen vertically refined nested production mesh against "
            "its exact horizontal factor-two nested reference on public geologies."
        )
    )
    parser.add_argument("--public-shard", action="append", type=Path, required=True)
    parser.add_argument("--modem-repo", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument(
        "--per-family",
        type=int,
        default=PUBLIC_SAMPLES_PER_FAMILY,
        help="frozen public truths per family; must be exactly 5",
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=3_600.0)
    parser.add_argument("--docker", default="docker")
    return parser


def _decode_list(value: Any) -> list[str]:
    return [
        item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item)
        for item in np.asarray(value).reshape(-1)
    ]


def _reject_nonpublic_path(path: Path) -> None:
    forbidden = {"hidden", "secret", "blind"}
    tokens = {
        token
        for part in path.resolve().parts
        for token in part.casefold().replace("_", "-").split("-")
    }
    intersection = forbidden & tokens
    if intersection:
        raise ValueError(
            "public convergence validator refuses a path marked "
            + ", ".join(sorted(intersection))
        )


def _catalog_public_shard(path: Path) -> tuple[list[PublicSelection], dict[str, object]]:
    _reject_nonpublic_path(path)
    source = snapshot_file(path, role="public convergence HDF5 shard")
    selections: list[PublicSelection] = []
    with h5py.File(io.BytesIO(source.payload), "r") as h5:
        checks = {
            "schema": h5.attrs.get("schema") == "pimsr-mt-2d",
            "schema_version": int(h5.attrs.get("schema_version", -1)) == 2,
            "generation_complete": int(h5.attrs.get("generation_complete", 0)) == 1,
            "generator_seed": int(h5.attrs.get("generator_seed", -1))
            == PUBLIC_GENERATOR_SEED,
            "generation_contract": h5.attrs.get("generation_contract")
            == PUBLIC_GENERATION_CONTRACT,
            "forward_contract": h5.attrs.get("forward_contract")
            == PUBLIC_FORWARD_CONTRACT,
            "scenario_order": _decode_list(h5.attrs.get("scenario_order", []))
            == list(SCENARIO_NAMES),
            "mode_order": _decode_list(h5.attrs.get("mode_order", [])) == ["te", "tm"],
            "components": _decode_list(h5.attrs.get("impedance_components", []))
            == ["Zyx", "Zxy"],
        }
        failed = sorted(name for name, passed in checks.items() if not passed)
        if failed:
            raise ValueError(f"shard is not the frozen public corpus: {failed}")
        sample_indices = np.asarray(h5["sample_index"][:], dtype=np.int64)
        scenarios = np.asarray(h5["scenario"][:], dtype=np.int64)
        if sample_indices.ndim != 1 or scenarios.shape != sample_indices.shape:
            raise ValueError("public shard sample/scenario indices are malformed")
        if len(set(map(int, sample_indices))) != sample_indices.size:
            raise ValueError("public shard contains duplicate sample indices")
        if not np.all(np.diff(sample_indices) > 0):
            raise ValueError("public shard sample indices must be strictly increasing")
        if not np.all((scenarios >= 0) & (scenarios < len(SCENARIO_NAMES))):
            raise ValueError("public shard contains an unknown scenario index")
        source_record = source.record()
        for row, (sample_index, scenario) in enumerate(
            zip(sample_indices, scenarios, strict=True)
        ):
            selections.append(
                PublicSelection(
                    source.path,
                    row,
                    int(sample_index),
                    int(scenario),
                    SCENARIO_NAMES[int(scenario)],
                    source_record,
                )
            )
    require_snapshot_unchanged(source, role="public convergence HDF5 shard")
    return selections, source.record()


def select_public_geologies(
    shards: list[Path], *, per_family: int
) -> tuple[list[PublicSelection], list[dict[str, object]]]:
    if per_family <= 0:
        raise ValueError("--per-family must be positive")
    catalog: list[PublicSelection] = []
    source_records: list[dict[str, object]] = []
    for shard in sorted({path.resolve() for path in shards}, key=str):
        entries, record = _catalog_public_shard(shard)
        catalog.extend(entries)
        source_records.append(record)
    sample_indices = [entry.sample_index for entry in catalog]
    if len(sample_indices) != len(set(sample_indices)):
        raise ValueError("public shards overlap in sample_index")
    selected: list[PublicSelection] = []
    for scenario_index, scenario_name in enumerate(SCENARIO_NAMES):
        candidates = sorted(
            (entry for entry in catalog if entry.scenario_index == scenario_index),
            key=lambda entry: entry.sample_index,
        )
        if len(candidates) < per_family:
            raise ValueError(
                f"public corpus has only {len(candidates)} {scenario_name} truths, "
                f"need {per_family}"
            )
        selected.extend(candidates[:per_family])
    selected.sort(key=lambda entry: (entry.scenario_index, entry.sample_index))
    return selected, source_records


def _source_for_case(
    selection: PublicSelection,
    hdf_record: dict[str, object],
    validator_source: dict[str, object],
) -> dict[str, object]:
    return {
        **hdf_record,
        "public_validation": {
            "selection_policy": "lowest sample_index per frozen scenario",
            "scenario_name": selection.scenario_name,
            "validator_source": validator_source,
        },
    }


def _run_case(
    *,
    runtime: Any,
    truth: CanonicalTruth,
    selection: PublicSelection,
    source: dict[str, object],
    mesh: MeshConfig | NestedMeshConfig,
    role: str,
    work_root: Path,
    timeout_seconds: float,
) -> CaseResult:
    output = work_root / "cases" / f"sample-{selection.sample_index:06d}" / role
    published, response, provenance = run_modem_forward(
        runtime=runtime,
        truth=truth,
        mesh=mesh,
        output_dir=output,
        source_provenance=source,
        timeout_seconds=timeout_seconds,
    )
    return CaseResult(selection, role, mesh, published, response, dict(provenance))


def _response_residuals(
    left: ModEMResponse, right: ModEMResponse
) -> dict[str, np.ndarray]:
    return {
        "te_log10_rho": np.abs(left.log10_rho_te - right.log10_rho_te),
        "te_phase_deg": circular_phase_error_deg(left.phase_te_deg, right.phase_te_deg),
        "tm_log10_rho": np.abs(left.log10_rho_tm - right.log10_rho_tm),
        "tm_phase_deg": circular_phase_error_deg(left.phase_tm_deg, right.phase_tm_deg),
    }


def _quantity_from_key(key: str) -> str:
    return "phase_deg" if key.endswith("phase_deg") else "log10_rho"


def _summaries(residuals: dict[str, np.ndarray]) -> dict[str, dict[str, object]]:
    return {key: summarize_absolute(value) for key, value in residuals.items()}


def _gates(summaries: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    return {
        key: gate_summary(summary, quantity=_quantity_from_key(key))
        for key, summary in summaries.items()
    }


def _padding_gates(
    summaries: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for key, summary in summaries.items():
        threshold_key = (
            "padding_phase_deg" if key.endswith("phase_deg") else "padding_log10_rho"
        )
        threshold = CONVERGENCE_GATES[threshold_key]["p95"]
        passed = bool(float(summary["p95"]) <= threshold)
        result[key] = {"thresholds": {"p95": threshold}, "passed": passed}
    return result


def _all_passed(gates: dict[str, dict[str, object]]) -> bool:
    return all(bool(gate["passed"]) for gate in gates.values())


def _make_uniform_truth(base: CanonicalTruth, *, name: str, rho: float) -> CanonicalTruth:
    return CanonicalTruth(
        log10_resistivity=np.full(base.log10_resistivity.shape, math.log10(rho)),
        x_centres_m=base.x_centres_m,
        depth_centres_m=base.depth_centres_m,
        frequencies_hz=base.frequencies_hz,
        station_x_m=base.station_x_m,
        sample_id=name,
    )


def _make_layered_truth(base: CanonicalTruth) -> CanonicalTruth:
    values = np.full(base.log10_resistivity.shape, math.log10(500.0))
    values[base.depth_centres_m < 3_000.0, :] = math.log10(10.0)
    values[base.depth_centres_m < 1_000.0, :] = math.log10(100.0)
    return CanonicalTruth(
        log10_resistivity=values,
        x_centres_m=base.x_centres_m,
        depth_centres_m=base.depth_centres_m,
        frequencies_hz=base.frequencies_hz,
        station_x_m=base.station_x_m,
        sample_id="analytic-layered-100-10-500",
    )


def _run_analytic_case(
    *,
    runtime: Any,
    truth: CanonicalTruth,
    mesh: MeshConfig | NestedMeshConfig,
    role: str,
    work_root: Path,
    timeout_seconds: float,
    validator_source: dict[str, object],
) -> tuple[CaseResult, dict[str, object]]:
    selection = PublicSelection(
        Path("analytic"),
        -1,
        -1,
        -1,
        truth.sample_id,
        {},
    )
    output = work_root / "analytic" / truth.sample_id / role
    published, response, provenance = run_modem_forward(
        runtime=runtime,
        truth=truth,
        mesh=mesh,
        output_dir=output,
        source_provenance={
            "scope": "analytic_public_validation",
            "validator_source": validator_source,
        },
        timeout_seconds=timeout_seconds,
    )
    analytic = analytic_response_for_mapped_1d(truth, mesh)
    residuals = _response_residuals(response, analytic)
    summaries = _summaries(residuals)
    gates = _gates(summaries)
    return (
        CaseResult(selection, role, mesh, published, response, dict(provenance)),
        {
            "truth_id": truth.sample_id,
            "mesh_id": mesh.mesh_id,
            "output_dir": str(published),
            "summaries": summaries,
            "gates": gates,
            "passed": _all_passed(gates),
        },
    )


def _npz_payload(arrays: dict[str, np.ndarray]) -> bytes:
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def _file_identity(path: Path) -> dict[str, object]:
    return snapshot_file(path, role="validation case artifact").record()


def _bundle_raw_run(
    *,
    ordinal: int,
    case_id: str,
    case_kind: str,
    sample_index: int | None,
    truth_id: str,
    role: str,
    mesh_id: str,
    output_dir: Path,
) -> tuple[dict[str, object], dict[str, bytes]]:
    prefix = f"raw-{ordinal:03d}"
    forward_name = f"{prefix}-forward.dat"
    provenance_name = f"{prefix}-provenance.json"
    forward = snapshot_file(
        output_dir / "forward.dat", role=f"raw run {ordinal} ModEM response"
    )
    provenance = snapshot_file(
        output_dir / "provenance.json", role=f"raw run {ordinal} provenance"
    )
    run = {
        "case_id": case_id,
        "case_kind": case_kind,
        "sample_index": sample_index,
        "truth_id": truth_id,
        "role": role,
        "mesh_id": mesh_id,
        "forward": {
            "path": forward_name,
            "sha256": forward.sha256,
            "size_bytes": forward.size_bytes,
        },
        "provenance": {
            "path": provenance_name,
            "sha256": provenance.sha256,
            "size_bytes": provenance.size_bytes,
        },
    }
    require_snapshot_unchanged(forward, role=f"raw run {ordinal} ModEM response")
    require_snapshot_unchanged(provenance, role=f"raw run {ordinal} provenance")
    return run, {
        forward_name: forward.payload,
        provenance_name: provenance.payload,
    }


def validate(
    args: argparse.Namespace,
) -> tuple[
    dict[str, object],
    bytes,
    dict[str, object],
    dict[str, bytes],
]:
    if args.per_family != PUBLIC_SAMPLES_PER_FAMILY:
        raise ValueError(
            "--per-family is frozen at 5 for the exact 25x3+4+1 raw-run contract"
        )
    if args.jobs <= 0 or args.jobs > 4:
        raise ValueError("--jobs must be between 1 and 4")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0.0:
        raise ValueError("--timeout-seconds must be finite and positive")
    validator_snapshot = snapshot_file(__file__, role="convergence validator source")
    bridge_snapshot = snapshot_file(
        Path(__file__).resolve().parents[1]
        / "src"
        / "pimsr_benchmarks"
        / "modem2d_forward.py",
        role="ModEM bridge source",
    )
    selected, shard_records = select_public_geologies(
        args.public_shard, per_family=args.per_family
    )
    if len(selected) != len(SCENARIO_NAMES) * PUBLIC_SAMPLES_PER_FAMILY:
        raise RuntimeError("public convergence selection must contain exactly 25 truths")
    work_root = args.work_root.resolve()
    if work_root.exists():
        raise FileExistsError(f"refusing to reuse convergence work root: {work_root}")
    work_root.mkdir(parents=True)
    runtime = verify_pinned_runtime(
        modem_repo=args.modem_repo,
        build_root=args.build_root,
        docker_executable=args.docker,
    )
    validator_record = validator_snapshot.record()
    loaded: list[tuple[PublicSelection, CanonicalTruth, dict[str, object]]] = []
    for selection in selected:
        truth, hdf_record = load_canonical_hdf5(selection.path, row=selection.row)
        if int(hdf_record["sample_index"]) != selection.sample_index:
            raise RuntimeError("public sample identity changed after selection")
        loaded.append(
            (
                selection,
                truth,
                _source_for_case(selection, hdf_record, validator_record),
            )
        )

    padding_mesh = PRODUCTION_MESH.padding_perturbation()
    tasks: list[
        tuple[PublicSelection, CanonicalTruth, dict[str, object], MeshConfig, str]
    ] = []
    for selection, truth, source in loaded:
        tasks.extend(
            (
                (selection, truth, source, PRODUCTION_MESH, "production-candidate"),
                (selection, truth, source, NEXT_FINER_REFERENCE, "next-finer-reference"),
                (selection, truth, source, padding_mesh, "padding-perturbation"),
            )
        )
    cases: dict[tuple[int, str], CaseResult] = {}
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        future_map = {
            executor.submit(
                _run_case,
                runtime=runtime,
                truth=truth,
                selection=selection,
                source=source,
                mesh=mesh,
                role=role,
                work_root=work_root,
                timeout_seconds=args.timeout_seconds,
            ): (selection.sample_index, role)
            for selection, truth, source, mesh, role in tasks
        }
        for future in as_completed(future_map):
            key = future_map[future]
            cases[key] = future.result()

    raw: dict[str, np.ndarray] = {
        "sample_index": np.asarray(
            [entry.sample_index for entry in selected], dtype=np.int64
        ),
        "scenario_index": np.asarray(
            [entry.scenario_index for entry in selected], dtype=np.int64
        ),
    }
    per_geology: list[dict[str, object]] = []
    convergence_values: dict[str, list[np.ndarray]] = {
        key: []
        for key in ("te_log10_rho", "te_phase_deg", "tm_log10_rho", "tm_phase_deg")
    }
    padding_values: dict[str, list[np.ndarray]] = {key: [] for key in convergence_values}
    all_exact_rows = True
    for sample_position, selection in enumerate(selected):
        production = cases[(selection.sample_index, "production-candidate")]
        reference = cases[(selection.sample_index, "next-finer-reference")]
        padding = cases[(selection.sample_index, "padding-perturbation")]
        paired = _response_residuals(production.response, reference.response)
        padding_residual = _response_residuals(production.response, padding.response)
        for key, values in paired.items():
            convergence_values[key].append(values)
            raw[f"candidate_vs_reference_{key}"] = (
                np.stack(convergence_values[key], axis=0)
                if sample_position == len(selected) - 1
                else raw.get(f"candidate_vs_reference_{key}", np.empty(0))
            )
        for key, values in padding_residual.items():
            padding_values[key].append(values)
            raw[f"candidate_vs_padding_{key}"] = (
                np.stack(padding_values[key], axis=0)
                if sample_position == len(selected) - 1
                else raw.get(f"candidate_vs_padding_{key}", np.empty(0))
            )
        roles = (production, reference, padding)
        exact_rows = all(
            result.provenance["response_contract"]["rows"] == {"TE": 96, "TM": 96}
            and result.provenance["response_contract"]["all_rows_finite"] is True
            for result in roles
        )
        all_exact_rows = all_exact_rows and exact_rows
        per_geology.append(
            {
                "sample_index": selection.sample_index,
                "scenario": selection.scenario_name,
                "source_shard_sha256": selection.source_record["sha256"],
                "candidate_vs_reference": _summaries(paired),
                "candidate_vs_padding": _summaries(padding_residual),
                "exact_96_te_and_96_tm_finite": exact_rows,
                "outputs": {
                    result.role: {
                        "directory": str(result.output_dir),
                        "forward": _file_identity(result.output_dir / "forward.dat"),
                        "provenance": _file_identity(
                            result.output_dir / "provenance.json"
                        ),
                    }
                    for result in roles
                },
            }
        )

    aggregate = {
        key: summarize_absolute(np.concatenate(values, axis=None))
        for key, values in convergence_values.items()
    }
    aggregate_padding = {
        key: summarize_absolute(np.concatenate(values, axis=None))
        for key, values in padding_values.items()
    }
    aggregate_gates = _gates(aggregate)
    padding_gates = _padding_gates(aggregate_padding)

    base_truth = loaded[0][1]
    analytic_truths = (
        _make_uniform_truth(base_truth, name="analytic-halfspace-100", rho=100.0),
        _make_layered_truth(base_truth),
    )
    analytic_records: list[dict[str, object]] = []
    analytic_cases: list[CaseResult] = []
    for analytic_truth in analytic_truths:
        for role, mesh in (
            ("production-candidate", PRODUCTION_MESH),
            ("next-finer-reference", NEXT_FINER_REFERENCE),
        ):
            analytic_case, record = _run_analytic_case(
                runtime=runtime,
                truth=analytic_truth,
                mesh=mesh,
                role=role,
                work_root=work_root,
                timeout_seconds=args.timeout_seconds,
                validator_source=validator_record,
            )
            analytic_cases.append(analytic_case)
            analytic_records.append(record)

    first = selected[0]
    first_production = cases[(first.sample_index, "production-candidate")]
    repeat_output = work_root / "determinism" / f"sample-{first.sample_index:06d}-repeat"
    repeat_published, _repeat_response, repeat_provenance = run_modem_forward(
        runtime=runtime,
        truth=loaded[0][1],
        mesh=PRODUCTION_MESH,
        output_dir=repeat_output,
        source_provenance=loaded[0][2],
        timeout_seconds=args.timeout_seconds,
    )
    first_forward = _file_identity(first_production.output_dir / "forward.dat")
    repeat_forward = _file_identity(repeat_published / "forward.dat")
    deterministic = first_forward["sha256"] == repeat_forward["sha256"]
    deterministic = deterministic and (
        repeat_provenance["response_contract"]["rows"] == {"TE": 96, "TM": 96}
    )

    family_counts = {
        family: sum(entry.scenario_name == family for entry in selected)
        for family in SCENARIO_NAMES
    }
    ensemble_sufficient = len(selected) >= 25 and all(
        count >= 5 for count in family_counts.values()
    )
    analytic_passed = all(bool(record["passed"]) for record in analytic_records)
    convergence_passed = _all_passed(aggregate_gates)
    padding_passed = _all_passed(padding_gates)
    passed = bool(
        ensemble_sufficient
        and convergence_passed
        and padding_passed
        and analytic_passed
        and deterministic
        and all_exact_rows
    )
    raw_payload = _npz_payload(raw)
    report: dict[str, object] = {
        "schema": "pimsr-modem2d-convergence-validation",
        "schema_version": 1,
        "passed": passed,
        "headline_eligible": passed,
        "scope": "public_only_no_hidden_or_secret_access",
        "production_candidate": {
            **PRODUCTION_MESH.canonical_record(),
            "mesh_config_sha256": PRODUCTION_MESH.sha256,
        },
        "next_finer_reference": {
            **NEXT_FINER_REFERENCE.canonical_record(),
            "mesh_config_sha256": NEXT_FINER_REFERENCE.sha256,
            "reference_only_not_automatically_production_eligible": True,
        },
        "padding_perturbation": {
            **padding_mesh.canonical_record(),
            "mesh_config_sha256": padding_mesh.sha256,
        },
        "paired_residual_definition": {
            "log10_rho": "absolute per station/frequency response difference",
            "phase": "absolute circular-180 per station/frequency response difference",
        },
        "frozen_gates": CONVERGENCE_GATES,
        "public_ensemble": {
            "selection_policy": "lowest sample_index per frozen scenario",
            "sample_count": len(selected),
            "family_counts": family_counts,
            "minimum_required": "25 total and >=5 per family",
            "sufficient": ensemble_sufficient,
            "source_shards": shard_records,
            "sample_indices": [entry.sample_index for entry in selected],
        },
        "candidate_vs_reference": {
            "aggregate": aggregate,
            "gates": aggregate_gates,
            "passed": convergence_passed,
        },
        "candidate_vs_padding": {
            "aggregate": aggregate_padding,
            "gates": padding_gates,
            "passed": padding_passed,
        },
        "per_geology": per_geology,
        "analytic_checks": {
            "records": analytic_records,
            "passed": analytic_passed,
        },
        "determinism": {
            "passed": deterministic,
            "first_forward": first_forward,
            "repeat_forward": repeat_forward,
        },
        "response_contract": {
            "required_rows_per_mode": {"TE": 96, "TM": 96},
            "all_cases_exact_and_finite": all_exact_rows,
        },
        "provenance": {
            "validator_source": validator_snapshot.record(),
            "bridge_source": bridge_snapshot.record(),
            "runtime": runtime.record,
            "runtime_identity_sha256": runtime.identity_sha256,
        },
        "raw_paired_residuals": {
            "filename": "paired-residuals.npz",
            "sha256": hashlib.sha256(raw_payload).hexdigest(),
            "size_bytes": len(raw_payload),
        },
        "blocker": (
            None
            if passed
            else (
                "production candidate is not qualified; do not generate hidden observations. "
                "If the nested candidate fails against its exact factor-two reference, "
                "the reference itself requires a separately frozen next-level check "
                "before production use."
            )
        ),
    }
    raw_runs: list[dict[str, object]] = []
    raw_artifacts: dict[str, bytes] = {}
    ordinal = 0
    for selection in selected:
        for role in (
            "production-candidate",
            "next-finer-reference",
            "padding-perturbation",
        ):
            case = cases[(selection.sample_index, role)]
            run, artifacts = _bundle_raw_run(
                ordinal=ordinal,
                case_id=f"public:{selection.sample_index}:{role}",
                case_kind="public_geology",
                sample_index=selection.sample_index,
                truth_id=f"sample-{selection.sample_index:06d}",
                role=role,
                mesh_id=case.mesh.mesh_id,
                output_dir=case.output_dir,
            )
            raw_runs.append(run)
            raw_artifacts.update(artifacts)
            ordinal += 1
    for case in analytic_cases:
        truth_id = case.selection.scenario_name
        run, artifacts = _bundle_raw_run(
            ordinal=ordinal,
            case_id=f"analytic:{truth_id}:{case.role}",
            case_kind="analytic",
            sample_index=None,
            truth_id=truth_id,
            role=case.role,
            mesh_id=case.mesh.mesh_id,
            output_dir=case.output_dir,
        )
        raw_runs.append(run)
        raw_artifacts.update(artifacts)
        ordinal += 1
    repeat_run, repeat_artifacts = _bundle_raw_run(
        ordinal=ordinal,
        case_id=f"determinism:{first.sample_index}:repeat",
        case_kind="determinism_repeat",
        sample_index=first.sample_index,
        truth_id=f"sample-{first.sample_index:06d}",
        role="determinism-repeat",
        mesh_id=PRODUCTION_MESH.mesh_id,
        output_dir=repeat_published,
    )
    raw_runs.append(repeat_run)
    raw_artifacts.update(repeat_artifacts)
    if len(raw_runs) != 80 or len(raw_artifacts) != 160:
        raise RuntimeError("public convergence raw bundle must contain 80 exact runs")
    raw_run_set: dict[str, object] = {
        "schema": RAW_RUN_SET_SCHEMA,
        "schema_version": RAW_RUN_SET_SCHEMA_VERSION,
        "runs": raw_runs,
    }
    require_snapshot_unchanged(validator_snapshot, role="convergence validator source")
    require_snapshot_unchanged(bridge_snapshot, role="ModEM bridge source")
    runtime.require_unchanged()
    return report, raw_payload, raw_run_set, raw_artifacts


def _failure_report(
    *, args: argparse.Namespace, error: BaseException, started: float
) -> dict[str, object]:
    return {
        "schema": "pimsr-modem2d-convergence-validation",
        "schema_version": 1,
        "passed": False,
        "headline_eligible": False,
        "scope": "public_only_no_hidden_or_secret_access",
        "production_candidate": PRODUCTION_MESH.canonical_record(),
        "next_finer_reference": NEXT_FINER_REFERENCE.canonical_record(),
        "frozen_gates": CONVERGENCE_GATES,
        "error": {"type": type(error).__name__, "message": str(error)},
        "elapsed_seconds_before_failure": time.monotonic() - started,
        "work_root": str(args.work_root.resolve()),
        "blocker": "validation failed closed; do not generate hidden observations",
    }


def main() -> None:
    args = _parser().parse_args()
    started = time.monotonic()
    exit_code = 0
    try:
        report, raw_payload, raw_run_set, raw_artifacts = validate(args)
        if not bool(report["passed"]):
            exit_code = 2
    except BaseException as error:  # noqa: BLE001 - failure report must include interrupts
        report = _failure_report(args=args, error=error, started=started)
        raw_payload = _npz_payload(
            {
                "validation_failed": np.asarray(True),
                "error_type": np.asarray(type(error).__name__),
            }
        )
        raw_run_set = {
            "schema": RAW_RUN_SET_SCHEMA,
            "schema_version": RAW_RUN_SET_SCHEMA_VERSION,
            "runs": [],
        }
        raw_artifacts = {}
        exit_code = 2
    report_payload = (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    raw_run_set_payload = (
        json.dumps(raw_run_set, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    publish_artifact_bundle(
        args.report_dir,
        {
            **raw_artifacts,
            "paired-residuals.npz": raw_payload,
            "public-convergence-raw-runs.json": raw_run_set_payload,
            "convergence-report.json": report_payload,
        },
        manifest_name="convergence-report.json",
    )
    print(report_payload.decode("utf-8"), end="")
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
