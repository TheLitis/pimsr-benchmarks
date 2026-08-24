from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import pimsr_benchmarks.comparison2d as comparison
from pimsr_benchmarks import dataset_lineage2d, modem2d_forward
from pimsr_benchmarks.comparison2d import (
    COMPARISON_SCHEMA,
    Comparison2DPublicationError,
    Comparison2DValidationError,
    EffectRow2D,
    canonical_json_bytes,
    hierarchical_paired_bootstrap_2d,
    publish_comparison_2d,
)
from pimsr_benchmarks.prediction_lock2d import (
    ArtifactSnapshot,
    LockedRun2D,
    ValidatedPredictionLock2D,
)

_SEEDS = (101, 102, 103, 104, 105)
_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_COMMIT = "d" * 40


def _effect_rows(*, seed_values: list[float] | None = None) -> list[EffectRow2D]:
    values = seed_values or [-0.5] * 5
    rows: list[EffectRow2D] = []
    for campaign in ("campaign-a", "campaign-b"):
        for family, base_count in (("family-small", 1), ("family-large", 3)):
            for base_index in range(base_count):
                for noise_index in range(2):
                    for seed_index, seed in enumerate(_SEEDS):
                        effect = values[seed_index]
                        if seed_values is None and family == "family-large":
                            effect = -1.5
                        rows.append(
                            EffectRow2D(
                                campaign_id=campaign,
                                training_seed=seed,
                                family_id=family,
                                base_model_id=f"{family}-base-{base_index}",
                                noise_id=f"noise-{noise_index}",
                                effects=np.asarray(
                                    [[effect, effect / 2], [effect - 0.25, effect / 3]],
                                    dtype=np.float64,
                                ),
                            )
                        )
    return rows


def _mesh_record(mesh_id: str, *, reference: bool = False) -> dict:
    values = {
        "nested-production-v1": (20, 1, 2),
        "nested-reference-x2-v1": (20, 2, 2),
        "nested-production-v1-padding-plus2": (22, 1, 2),
    }
    padding_count, horizontal_factor, vertical_factor = values[mesh_id]
    record = {
        "schema": "pimsr-modem2d-nested-mesh",
        "schema_version": 1,
        "mesh_id": mesh_id,
        "version": 1,
        "base_core_width_m": 62.5,
        "base_core_count": 384,
        "base_padding_count_each_side": padding_count,
        "base_padding_growth": 1.4,
        "minimum_vertical_subdivisions": 2,
        "maximum_base_dz_m": 2_500.0,
        "deep_padding_growth": 1.35,
        "maximum_deep_macro_dz_m": 10_000.0,
        "minimum_depth_m": 220_000.0,
        "horizontal_refinement_factor": horizontal_factor,
        "vertical_refinement_factor": vertical_factor,
        "canonical_depth_centres_sha256": comparison._CANONICAL_DEPTH_CENTRES_SHA256,
        "canonical_x_centres_sha256": comparison._CANONICAL_X_CENTRES_SHA256,
        "horizontal_partition": comparison._HORIZONTAL_PARTITION,
        "vertical_partition": comparison._VERTICAL_PARTITION,
        "mapping": comparison._MESH_MAPPING,
    }
    digest = comparison._canonical_object_sha256(record)
    result = {**record, "mesh_config_sha256": digest}
    if reference:
        result["reference_only_not_automatically_production_eligible"] = True
    return result


def _residual_archive() -> tuple[dict[str, np.ndarray], ArtifactSnapshot]:
    sample_index = np.arange(25, dtype=np.int64)
    scenario_index = np.repeat(np.arange(5, dtype=np.int64), 5)
    arrays: dict[str, np.ndarray] = {
        "sample_index": sample_index,
        "scenario_index": scenario_index,
    }
    channel_values = {
        "te_log10_rho": 0.004,
        "te_phase_deg": 0.09,
        "tm_log10_rho": 0.0045,
        "tm_phase_deg": 0.095,
    }
    for prefix, scale in (("candidate_vs_reference", 1.0), ("candidate_vs_padding", 0.5)):
        for channel, value in channel_values.items():
            arrays[f"{prefix}_{channel}"] = np.full(
                (25, 8, 12), value * scale, dtype=np.float64
            )
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    payload = stream.getvalue()
    return arrays, ArtifactSnapshot(
        Path("paired-residuals.npz"),
        payload,
        hashlib.sha256(payload).hexdigest(),
        10,
        20,
    )


def _headline_evidence(
    residuals_snapshot: ArtifactSnapshot | None = None,
) -> dict:
    if residuals_snapshot is None:
        _, residuals_snapshot = _residual_archive()
    return {
        "hidden_observation_generator": {
            "name": "ModEM",
            "repository_url": "https://example.test/modem",
            "repository_commit": "1" * 40,
            "source_sha256": "1" * 64,
            "source_size_bytes": 11,
            "binary_sha256": "2" * 64,
            "binary_size_bytes": 12,
            "container_image_digest": "sha256:" + "3" * 64,
            "mesh_artifact_sha256": _mesh_record("nested-production-v1")[
                "mesh_config_sha256"
            ],
            "mesh_artifact_size_bytes": 14,
            "converter_sha256": "5" * 64,
            "converter_size_bytes": 15,
            "converter_repository_commit": "6" * 40,
            "generation_runtime": dict(comparison._HIDDEN_GENERATION_RUNTIME),
            "generation_runtime_manifest_sha256": "6" * 64,
            "generation_runtime_manifest_size_bytes": 16,
        },
        "public_mesh_convergence": {
            "criterion_id": "modem_public_mesh_convergence_v1",
            "report_sha256": "7" * 64,
            "report_size_bytes": 17,
            "residuals_sha256": residuals_snapshot.sha256,
            "residuals_size_bytes": residuals_snapshot.size_bytes,
            "refined_mesh_sha256": _mesh_record("nested-reference-x2-v1", reference=True)[
                "mesh_config_sha256"
            ],
            "refined_mesh_size_bytes": 18,
            "thresholds": {
                "log10_rho_dex": {"median": 0.005, "p95": 0.015, "max": 0.05},
                "phase_circular_180_degrees": {
                    "median": 0.10,
                    "p95": 0.50,
                    "max": 1.50,
                },
                "padding": {
                    "log10_rho_p95_dex": 0.005,
                    "phase_circular_180_p95_degrees": 0.20,
                },
            },
            "analytic_1d_contract": _analytic_contract(),
        },
        "training_solver_commits": {
            "pimsr": "a" * 40,
            "mtdlpy": "b" * 40,
            "mt2dinv_densenet": "c" * 40,
        },
    }


def _analytic_contract() -> dict:
    frequencies = np.geomspace(0.01, 100.0, 8).astype("<f8")
    depth = np.logspace(np.log10(10.0), np.log10(60_000.0), 64).astype("<f8")
    return {
        "schema": "pimsr-modem2d-analytic-1d-contract",
        "schema_version": 1,
        "time_convention": "exp(+i omega t)",
        "response_shape": [8, 12],
        "frequencies_hz": frequencies.tolist(),
        "frequencies_sha256": hashlib.sha256(frequencies.tobytes()).hexdigest(),
        "canonical_depth_centres_m": depth.tolist(),
        "canonical_depth_centres_sha256": hashlib.sha256(depth.tobytes()).hexdigest(),
        "cases": [
            {
                "truth_id": "analytic-halfspace-100",
                "depth_profile": [
                    {
                        "maximum_depth_m_exclusive": None,
                        "resistivity_ohm_m": 100.0,
                    }
                ],
            },
            {
                "truth_id": "analytic-layered-100-10-500",
                "depth_profile": [
                    {
                        "maximum_depth_m_exclusive": 1_000.0,
                        "resistivity_ohm_m": 100.0,
                    },
                    {
                        "maximum_depth_m_exclusive": 3_000.0,
                        "resistivity_ohm_m": 10.0,
                    },
                    {
                        "maximum_depth_m_exclusive": None,
                        "resistivity_ohm_m": 500.0,
                    },
                ],
            },
        ],
    }


def test_nested_mesh_recomputation_matches_pinned_modem_bridge():
    contract = comparison._analytic_1d_contract(_analytic_contract())
    for mesh_id in (
        "nested-production-v1",
        "nested-reference-x2-v1",
    ):
        record = comparison._mesh_config_record(
            _mesh_record(mesh_id, reference=mesh_id == "nested-reference-x2-v1"),
            path=mesh_id,
            reference_only=mesh_id == "nested-reference-x2-v1",
        )
        bridge_mesh = modem2d_forward.MESH_CONFIGS[mesh_id]
        expected_record = {
            **bridge_mesh.canonical_record(),
            "mesh_config_sha256": bridge_mesh.sha256,
        }
        if mesh_id == "nested-reference-x2-v1":
            expected_record["reference_only_not_automatically_production_eligible"] = True
        assert record == expected_record
        expected_horizontal, expected_vertical = bridge_mesh.cell_widths(
            contract["canonical_depth_centres_m"]
        )
        np.testing.assert_array_equal(
            comparison._mesh_horizontal_widths(record), expected_horizontal
        )
        np.testing.assert_array_equal(
            comparison._mesh_vertical_widths(
                record, contract["canonical_depth_centres_m"]
            ),
            expected_vertical,
        )


def _gate_record(summary: dict, channel: str, *, padding: bool) -> dict:
    thresholds = comparison._channel_thresholds(channel, padding=padding)
    checks = {name: summary[name] <= limit for name, limit in thresholds.items()}
    result = {"thresholds": dict(thresholds), "passed": all(checks.values())}
    if not padding:
        result["checks"] = checks
    return result


def _section(arrays: dict[str, np.ndarray], *, prefix: str, padding: bool) -> dict:
    aggregate = {
        channel: comparison._summary_from_array(arrays[f"{prefix}_{channel}"])
        for channel in comparison._CONVERGENCE_CHANNELS
    }
    return {
        "aggregate": aggregate,
        "gates": {
            channel: _gate_record(aggregate[channel], channel, padding=padding)
            for channel in comparison._CONVERGENCE_CHANNELS
        },
        "passed": True,
    }


def _identity(path: str, sha: str, size: int) -> dict:
    return {"path": path, "sha256": sha, "size_bytes": size}


def _convergence_report() -> tuple[dict, dict[str, np.ndarray], ArtifactSnapshot, dict]:
    arrays, residuals_snapshot = _residual_archive()
    evidence = _headline_evidence(residuals_snapshot)
    generator = evidence["hidden_observation_generator"]
    production = _mesh_record("nested-production-v1")
    reference = _mesh_record("nested-reference-x2-v1", reference=True)
    padding = _mesh_record("nested-production-v1-padding-plus2")
    reference_section = _section(arrays, prefix="candidate_vs_reference", padding=False)
    padding_section = _section(arrays, prefix="candidate_vs_padding", padding=True)
    per_geology = []
    for index in range(25):
        summaries = {}
        for label in ("candidate_vs_reference", "candidate_vs_padding"):
            summaries[label] = {
                channel: comparison._summary_from_array(
                    arrays[f"{label}_{channel}"][index]
                )
                for channel in comparison._CONVERGENCE_CHANNELS
            }
        per_geology.append(
            {
                "sample_index": index,
                "scenario": comparison._CONVERGENCE_FAMILIES[index // 5],
                "source_shard_sha256": "a" * 64,
                **summaries,
                "exact_96_te_and_96_tm_finite": True,
                "outputs": {
                    role: {
                        "directory": f"case-{index}/{role}",
                        "forward": _identity("forward.dat", "b" * 64, 10),
                        "provenance": _identity("provenance.json", "c" * 64, 20),
                    }
                    for role in (
                        "production-candidate",
                        "next-finer-reference",
                        "padding-perturbation",
                    )
                },
            }
        )
    analytic_records = []
    for truth in ("analytic-halfspace-100", "analytic-layered-100-10-500"):
        for mesh_id in ("nested-production-v1", "nested-reference-x2-v1"):
            summaries = {
                channel: comparison._summary_from_array(
                    np.full((8, 12), 0.001, dtype=np.float64)
                )
                for channel in comparison._CONVERGENCE_CHANNELS
            }
            analytic_records.append(
                {
                    "truth_id": truth,
                    "mesh_id": mesh_id,
                    "output_dir": f"analytic/{truth}/{mesh_id}",
                    "summaries": summaries,
                    "gates": {
                        channel: _gate_record(summaries[channel], channel, padding=False)
                        for channel in comparison._CONVERGENCE_CHANNELS
                    },
                    "passed": True,
                }
            )
    runtime = {
        "schema": "pimsr-modem2d-runtime-provenance",
        "schema_version": 1,
        "modem": {
            "commit": generator["repository_commit"],
            "checkout_clean": True,
        },
        "container": {
            "image_id": generator["container_image_digest"],
            "reference": "modem@" + generator["container_image_digest"],
        },
        "artifacts": {
            "source": _identity(
                "source.tar",
                generator["source_sha256"],
                generator["source_size_bytes"],
            ),
            "binary": _identity(
                "Mod2DMT",
                generator["binary_sha256"],
                generator["binary_size_bytes"],
            ),
        },
    }
    report = {
        "schema": "pimsr-modem2d-convergence-validation",
        "schema_version": 1,
        "passed": True,
        "headline_eligible": True,
        "scope": "public_only_no_hidden_or_secret_access",
        "production_candidate": production,
        "next_finer_reference": reference,
        "padding_perturbation": padding,
        "paired_residual_definition": {
            "log10_rho": "absolute per station/frequency response difference",
            "phase": "absolute circular-180 per station/frequency response difference",
        },
        "frozen_gates": comparison._expected_report_gates(),
        "public_ensemble": {
            "selection_policy": "lowest sample_index per frozen scenario",
            "sample_count": 25,
            "family_counts": {family: 5 for family in comparison._CONVERGENCE_FAMILIES},
            "minimum_required": "25 total and >=5 per family",
            "sufficient": True,
            "source_shards": [_identity("public.h5", "a" * 64, 100)],
            "sample_indices": list(range(25)),
        },
        "candidate_vs_reference": reference_section,
        "candidate_vs_padding": padding_section,
        "per_geology": per_geology,
        "analytic_checks": {"records": analytic_records, "passed": True},
        "determinism": {
            "passed": True,
            "first_forward": _identity("first.dat", "d" * 64, 30),
            "repeat_forward": _identity("repeat.dat", "d" * 64, 30),
        },
        "response_contract": {
            "required_rows_per_mode": {"TE": 96, "TM": 96},
            "all_cases_exact_and_finite": True,
        },
        "provenance": {
            "validator_source": _identity("validator.py", "e" * 64, 40),
            "bridge_source": _identity(
                "modem2d_forward.py",
                generator["converter_sha256"],
                generator["converter_size_bytes"],
            ),
            "runtime": runtime,
            "runtime_identity_sha256": comparison._canonical_object_sha256(runtime),
        },
        "raw_paired_residuals": {
            "filename": "paired-residuals.npz",
            "sha256": residuals_snapshot.sha256,
            "size_bytes": residuals_snapshot.size_bytes,
        },
        "blocker": None,
    }
    return report, arrays, residuals_snapshot, evidence


def _operator_manifest() -> dict:
    groups: list[dict] = []
    mapping: list[dict] = []
    reveal_rows: list[dict] = []
    sample_id = 0
    for base_index in range(100):
        for noise_index in range(5):
            family = comparison.FAMILY_IDS[base_index // 20]
            groups.append(
                {
                    "base_model_id": f"base-{base_index}",
                    "family_id": family,
                    "noise_id": f"noise-{noise_index}",
                    "sample_ids": [f"sample-{sample_id}"],
                }
            )
            reveal_rows.append(
                {
                    "sample_index": sample_id,
                    "base_model_id": f"base-{base_index}",
                    "family_id": family,
                    "noise_index": noise_index,
                }
            )
            mapping.append(
                {
                    "opaque_sample_index": sample_id,
                    "source_generator_sample_index": sample_id,
                }
            )
            sample_id += 1
    return {
        "artifacts": {
            "observations": {
                "schema": "pimsr-sota-2d-observations",
                "schema_version": 1,
                "sha256": _SHA_A,
                "size_bytes": 100,
            },
            "withheld_truth": {
                "schema": "pimsr-sota-2d-truth",
                "schema_version": 2,
                "sha256": _SHA_B,
                "size_bytes": 101,
            },
            "public_observation_manifest": {
                "schema": "pimsr-sota-2d-observation-manifest",
                "schema_version": 3,
                "sha256": _SHA_C,
                "size_bytes": 102,
            },
        },
        "audience": "benchmark_operator_only",
        "schema": "pimsr-sota-2d-scoring-manifest",
        "schema_version": 3,
        "source": {
            "production_generation_closure": (
                "post_score_manifest.campaign.hidden_generation"
            )
        },
        "split": {
            "split_id": "campaign-a",
            "sample_count": 500,
            "groups": groups,
            "opaque_sample_id_contract": {
                "algorithm": "HMAC-SHA256",
                "digest_projection": "first_64_bits_big_endian_clear_sign_bit",
                "key_material": "external_secret_not_recorded",
                "message": (
                    "domain_separator || generator_seed_uint64_be || "
                    "source_sample_index_uint64_be || split_id_length_uint32_be || "
                    "split_id_ascii"
                ),
                "version": 1,
            },
            "sample_id_mapping": mapping,
            "scenario_groups": [
                {
                    "opaque_sample_indices": [
                        index
                        for index in range(500)
                        if (index // 5) // 20 == family_index
                    ],
                    "scenario": comparison.FAMILY_IDS[family_index],
                    "scenario_index": family_index,
                }
                for family_index in range(5)
            ],
            "family_partition_reveal": {
                "schema": "pimsr-sota-2d-family-partition-reveal",
                "schema_version": 1,
                "campaign_id": "campaign-a",
                "nonce_hex": "9" * 64,
                "rows": reveal_rows,
            },
            "payload_row_order": "strictly_increasing_opaque_sample_index",
        },
    }


def _operator_commitment(operator: dict) -> str:
    reveal = operator["split"]["family_partition_reveal"]
    return comparison._family_commitment_digest(
        campaign_id=reveal["campaign_id"],
        nonce_hex=reveal["nonce_hex"],
        rows=reveal["rows"],
    )


def _hidden_observation_snapshot(
    *, phase_value: float = 30.0, floor_value: float = 0.05
) -> ArtifactSnapshot:
    shape = (500, 8, 12)
    arrays = {
        "schema": np.asarray("pimsr-sota-2d-observations"),
        "schema_version": np.asarray(1, dtype="<i8"),
        "sample_index": np.arange(500, dtype="<i8"),
        "frequency_hz": np.geomspace(0.01, 100.0, 8).astype("<f8"),
        "station_x_m": np.linspace(-5_500.0, 5_500.0, 12).astype("<f8"),
        "x_cell_centers_m": np.linspace(-11_750.0, 11_750.0, 48).astype("<f8"),
        "depth_cell_centers_m": np.linspace(10.0, 60_000.0, 64).astype("<f8"),
        "observation_channel_order": np.asarray(
            [
                "log10_rho_te",
                "phase_te_degrees",
                "log10_rho_tm",
                "phase_tm_degrees",
            ]
        ),
        "observed_log10_rho_te": np.full(shape, 2.0, dtype="<f4"),
        "observed_phase_te_degrees": np.full(shape, phase_value, dtype="<f4"),
        "observed_log10_rho_tm": np.full(shape, 3.0, dtype="<f4"),
        "observed_phase_tm_degrees": np.full(shape, phase_value, dtype="<f4"),
        "declared_evaluation_floor_log10_rho_te": np.full(
            shape, floor_value, dtype="<f4"
        ),
        "declared_evaluation_floor_phase_te_degrees": np.full(
            shape, floor_value, dtype="<f4"
        ),
        "declared_evaluation_floor_log10_rho_tm": np.full(
            shape, floor_value, dtype="<f4"
        ),
        "declared_evaluation_floor_phase_tm_degrees": np.full(
            shape, floor_value, dtype="<f4"
        ),
        "valid_mask": np.ones((500, 4, 8, 12), dtype=np.bool_),
    }
    assert tuple(arrays) == comparison._OBSERVATION_ARRAY_MEMBERS
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    payload = stream.getvalue()
    return ArtifactSnapshot(
        Path("observations.npz"),
        payload,
        hashlib.sha256(payload).hexdigest(),
        81,
        82,
    )


def _hidden_public_lineage_identity() -> dict:
    repositories = {
        "pimsr_forward": {
            "commit": "1" * 40,
            "source_hashes": {
                "src/pimsr_forward/dataset2d.py": "2" * 64,
                "src/pimsr_forward/mt2d.py": "3" * 64,
                "src/pimsr_forward/sensors.py": "4" * 64,
            },
        },
        "pimsr_geogen": {
            "commit": "5" * 40,
            "source_hashes": {
                "src/pimsr_geogen/generator.py": "7" * 64,
                "src/pimsr_geogen/model.py": "8" * 64,
                "src/pimsr_geogen/rock_physics.py": "9" * 64,
                "src/pimsr_geogen/section2d.py": "6" * 64,
            },
        },
    }
    return {
        "train": {"repositories": repositories},
        "validation": {"repositories": repositories},
    }


def _generation_runtime_manifest(public_lineage: dict) -> dict:
    return {
        "schema": "pimsr-hidden-generation-runtime-2d",
        "schema_version": 1,
        "python": {
            "implementation": "CPython",
            "version": "3.11.15",
            "executable_sha256": "a" * 64,
        },
        "distributions": {
            name: {
                "version": version,
                "installed_tree_sha256": hashlib.sha256(name.encode()).hexdigest(),
            }
            for name, version in comparison._HIDDEN_RUNTIME_DISTRIBUTIONS.items()
        },
        "source_closure": comparison._hidden_source_lineage_identity(public_lineage),
        "tree_manifest_sha256": "b" * 64,
    }


def test_generation_runtime_manifest_v1_binds_versions_and_source_closure():
    public_lineage = _hidden_public_lineage_identity()
    manifest = _generation_runtime_manifest(public_lineage)
    payload = canonical_json_bytes(manifest)
    snapshot = ArtifactSnapshot(
        Path("generation-runtime.json"),
        payload,
        hashlib.sha256(payload).hexdigest(),
        91,
        92,
    )
    parsed = comparison._strict_json(snapshot, "hidden generation runtime manifest")
    identity = comparison._validate_generation_runtime_manifest(
        parsed,
        public_lineage=public_lineage,
    )
    assert identity["schema"] == "pimsr-hidden-generation-runtime-2d"
    assert identity["schema_version"] == 1
    assert identity["python_executable_sha256"] == "a" * 64
    assert set(identity["distribution_tree_sha256"]) == {
        "numpy",
        "h5py",
        "pimsr_benchmarks",
        "pimsr_geogen",
        "pimsr_forward",
    }

    wrong_version = json.loads(json.dumps(manifest))
    wrong_version["distributions"]["numpy"]["version"] = "2.4.5"
    with pytest.raises(Comparison2DValidationError, match="version is not frozen"):
        comparison._validate_generation_runtime_manifest(
            wrong_version,
            public_lineage=public_lineage,
        )

    wrong_source = json.loads(json.dumps(manifest))
    wrong_source["source_closure"]["pimsr_geogen"]["generator_source_sha256"] = "f" * 64
    with pytest.raises(Comparison2DValidationError, match="installed sources differ"):
        comparison._validate_generation_runtime_manifest(
            wrong_source,
            public_lineage=public_lineage,
        )

    extra_key = json.loads(json.dumps(manifest))
    extra_key["unpreregistered"] = True
    with pytest.raises(Comparison2DValidationError, match="keys mismatch"):
        comparison._validate_generation_runtime_manifest(
            extra_key,
            public_lineage=public_lineage,
        )


def _forward_snapshot(
    *, rho_te: float = 100.0, rho_tm: float = 1_000.0
) -> ArtifactSnapshot:
    frequencies = np.geomspace(0.01, 100.0, 8)
    stations = np.linspace(-5_500.0, 5_500.0, 12)
    rows: list[str] = []
    for mode, rho, phase in (("TE", rho_te, 30.0), ("TM", rho_tm, 120.0)):
        rows.extend(
            (
                f"> {mode}_Impedance",
                "> exp(+i\\omega t)",
                "> [V/m]/[T]",
                "> 0.00",
                "> 0.000 0.000",
                "> 8 12",
            )
        )
        for station_index, station in enumerate(stations, start=1):
            for frequency in frequencies:
                omega = 2.0 * np.pi * frequency
                magnitude = np.sqrt(rho * omega / comparison._MU0)
                value = magnitude * np.exp(1j * np.deg2rad(phase))
                rows.append(
                    f"{1.0 / frequency:.12e} S{station_index:02d} "
                    f"0 0 0 {station:.12e} 0 {mode} "
                    f"{value.real:.12e} {value.imag:.12e} 1"
                )
    payload = ("\n".join(rows) + "\n").encode("ascii")
    return ArtifactSnapshot(
        Path("forward.dat"),
        payload,
        hashlib.sha256(payload).hexdigest(),
        91,
        92,
    )


def _commitment_contract() -> dict:
    return {
        "algorithm": "SHA-256",
        "canonicalization": "utf8-canonical-json-sort-keys-compact-newline-v1",
        "domain_separator": "pimsr-sota-2d-family-partition/v1",
        "nonce_encoding": "lowercase_hex_32_bytes",
    }


def _lineage_manifest() -> tuple[dict, dict]:
    sample_count = 10_000
    arrays: dict[str, dict] = {}
    row_response_names = {
        name
        for name in comparison._LINEAGE_ROW_ARRAYS
        if name.startswith(("obs_mt_", "clean_mt_"))
    }
    for name in comparison._LINEAGE_ROW_ARRAYS | comparison._LINEAGE_COORDINATE_ARRAYS:
        if name in row_response_names:
            dtype, shape = "<f4", [sample_count, 8, 12]
        elif name == "target_log10_res":
            dtype, shape = "<f4", [sample_count, 64, 48]
        elif name == "scenario":
            dtype, shape = "<i4", [sample_count]
        elif name == "has_fault":
            dtype, shape = "|u1", [sample_count]
        elif name == "sample_index":
            dtype, shape = "<i8", [sample_count]
        else:
            lengths = {"frequencies": 8, "station_x": 12, "x_grid": 48, "depth_grid": 64}
            dtype, shape = "<f8", [lengths[name]]
        arrays[name] = {
            "dtype": dtype,
            "shape": shape,
            "logical_c_order_bytes_sha256": hashlib.sha256(name.encode()).hexdigest(),
            "shard_equality": (
                "exact_ordered_concatenation"
                if name in comparison._LINEAGE_ROW_ARRAYS
                else "exact_repetition_in_every_shard"
            ),
        }
    shards = []
    for ordinal in range(100):
        start = ordinal * 100
        end = start + 99
        filename = f"shard-{start:06d}-{end:06d}.h5"
        shards.append(
            {
                "ordinal": ordinal,
                "sample_start": start,
                "sample_end": end,
                "sample_count": 100,
                "hdf5": {
                    "filename": filename,
                    "sha256": hashlib.sha256((filename + "h5").encode()).hexdigest(),
                    "size_bytes": 1,
                },
                "log": {
                    "filename": filename + ".log",
                    "sha256": hashlib.sha256((filename + "log").encode()).hexdigest(),
                    "size_bytes": 1,
                },
            }
        )
    root_attributes = {
        **comparison._LINEAGE_ROOT_STRING_ATTRIBUTES,
        "schema_version": 2,
        "generator_seed": 20260820,
        "generation_start_index": 0,
        "expected_row_count": sample_count,
        "source_shard_count": 100,
        "generation_complete": 1,
        "mode_order": ["te", "tm"],
        "impedance_components": ["Zyx", "Zxy"],
        "scenario_order": list(comparison.FAMILY_IDS),
        "sensor_parameters_json": json.dumps(
            comparison._LINEAGE_SENSOR_PARAMETERS,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "software_versions_json": json.dumps(
            {
                name: "1"
                for name in (
                    "discretize",
                    "h5py",
                    "numpy",
                    "pimsr_forward",
                    "pimsr_geogen",
                    "simpeg",
                )
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
    }
    forward_sources = {
        name: {
            "sha256": hashlib.sha256(name.encode()).hexdigest(),
            "size_bytes": 1,
            "matches_commit_blob_after_git_clean_filter": True,
        }
        for name in (
            "src/pimsr_forward/dataset2d.py",
            "src/pimsr_forward/mt2d.py",
            "src/pimsr_forward/sensors.py",
        )
    }
    geogen_sources = {
        name: {
            "sha256": hashlib.sha256(name.encode()).hexdigest(),
            "size_bytes": 1,
            "matches_commit_blob_after_git_clean_filter": True,
        }
        for name in (
            "src/pimsr_geogen/generator.py",
            "src/pimsr_geogen/model.py",
            "src/pimsr_geogen/rock_physics.py",
            "src/pimsr_geogen/section2d.py",
        )
    }
    manifest = {
        "schema": "pimsr-public-dataset-lineage-2d",
        "schema_version": 2,
        "evidence_scope": comparison._PUBLIC_LINEAGE_EVIDENCE_SCOPE,
        "split": "train",
        "source_derived_generation_semantics": dict(
            comparison._SOURCE_DERIVED_GENERATION_SEMANTICS
        ),
        "inputs": {
            "merged_dataset": {
                "path": "pimsr-generated-2d-v1-train.h5",
                "sha256": "1" * 64,
                "size_bytes": 10,
            },
            "shard_directory": {"path": "train-shards", "entry_count": 200},
            "shard_pin_manifest": {
                "path": "train-shard-pins.json",
                "sha256": "2" * 64,
                "size_bytes": 10,
            },
            "shards": shards,
        },
        "repositories": {
            "pimsr_forward": {
                "path": "pimsr-forward",
                "commit": "a" * 40,
                "clean_worktree": True,
                "origin_remote": "https://github.com/TheLitis/pimsr-forward",
                "source_files": forward_sources,
            },
            "pimsr_geogen": {
                "path": "pimsr-geogen",
                "commit": "b" * 40,
                "clean_worktree": True,
                "origin_remote": "https://github.com/TheLitis/pimsr-geogen",
                "source_files": geogen_sources,
            },
        },
        "verification": {
            "arrays": arrays,
            "chunk_rows": 100,
            "concatenation": "exact_ordered_array_and_metadata_equality",
            "forward_regeneration_performed": False,
            "generation_complete": True,
            "generation_start_index": 0,
            "generation_time_execution_proven": False,
            "generator_seed": 20260820,
            "root_attributes": root_attributes,
            "sample_count": sample_count,
            "sample_end_index": 9_999,
            "schema_contract": "pimsr-mt-2d/v2",
            "source_shard_count": 100,
        },
    }
    expected = {
        "filename": "pimsr-generated-2d-v1-train.h5",
        "sha256": "1" * 64,
        "size_bytes": 10,
        "generator_commit": "a" * 40,
        "generator_seed": 20260820,
        "start_index": 0,
        "sample_count": sample_count,
        "sample_end_index": 9_999,
        "source_shard_count": 100,
        "shard_rows": 100,
    }
    return manifest, expected


def _locked_run() -> LockedRun2D:
    return LockedRun2D(
        campaign_id="campaign-a",
        method_id="pimsr",
        training_seed=101,
        observations_sha256=_SHA_A,
        observation_manifest_sha256=_SHA_C,
        prediction_sha256="e" * 64,
        prediction_size_bytes=103,
        runtime_sha256="f" * 64,
        runtime_size_bytes=104,
        checkpoint_sha256="0" * 64,
        checkpoint_size_bytes=105,
        source_commit=_COMMIT,
        source_sha256="1" * 64,
        adapter_source_sha256="2" * 64,
    )


def _validated_lock(run: LockedRun2D) -> ValidatedPredictionLock2D:
    return ValidatedPredictionLock2D(
        preregistration_sha256="3" * 64,
        lock_sha256="4" * 64,
        input_manifest_sha256="5" * 64,
        campaign_ids=("campaign-a",),
        method_ids=("pimsr",),
        training_seeds=(101,),
        statistical_options={},
        runs=(run,),
    )


def _material_evaluation_inputs(campaign, run: LockedRun2D):
    sample_ids = np.asarray(campaign.hierarchy.sample_ids, dtype="<i8")
    x_grid = np.linspace(-11_750.0, 11_750.0, 48, dtype="<f8")
    depth_grid = np.linspace(100.0, 6_400.0, 64, dtype="<f8")
    truth_values = np.zeros((500, 64, 48), dtype="<f4")
    prediction_values = np.zeros_like(truth_values)
    prediction_values[:, :16, :] = np.float32(2.0)
    family_by_sample = {
        sample_id: family
        for family in campaign.hierarchy.families
        for _base, noise_rows in campaign.hierarchy.tree[family]
        for _noise, sample_id in noise_rows
    }
    truth = comparison._MaterialTruth2D(
        sample_index=sample_ids,
        observations_sha256=run.observations_sha256,
        scenario=np.asarray(
            [family_by_sample[int(sample_id)] for sample_id in sample_ids]
        ),
        has_fault=np.zeros(500, dtype=np.bool_),
        x_cell_centers_m=x_grid,
        depth_cell_centers_m=depth_grid,
        log10_resistivity=truth_values,
        artifact_sha256=campaign.truth_sha256,
        artifact_size_bytes=campaign.truth_size_bytes,
    )
    predictions = comparison._MaterialPredictions2D(
        sample_index=sample_ids,
        observations_sha256=run.observations_sha256,
        x_cell_centers_m=x_grid,
        depth_cell_centers_m=depth_grid,
        log10_resistivity=prediction_values,
        artifact_sha256=run.prediction_sha256,
        artifact_size_bytes=run.prediction_size_bytes,
    )
    return truth, predictions


def _material_npz_snapshots(campaign, run: LockedRun2D):
    truth, predictions = _material_evaluation_inputs(campaign, run)
    truth_stream = io.BytesIO()
    np.savez(
        truth_stream,
        schema=np.asarray("pimsr-sota-2d-truth"),
        schema_version=np.asarray(2, dtype="<i8"),
        sample_index=truth.sample_index,
        observations_sha256=np.asarray(truth.observations_sha256, dtype="<U64"),
        scenario=truth.scenario,
        has_fault=truth.has_fault,
        x_cell_centers_m=truth.x_cell_centers_m,
        depth_cell_centers_m=truth.depth_cell_centers_m,
        truth_log10_resistivity=truth.log10_resistivity,
    )
    prediction_stream = io.BytesIO()
    np.savez(
        prediction_stream,
        schema=np.asarray("pimsr-sota-2d-predictions"),
        schema_version=np.asarray(2, dtype="<i8"),
        observations_sha256=np.asarray(predictions.observations_sha256, dtype="<U64"),
        sample_index=predictions.sample_index,
        x_cell_centers_m=predictions.x_cell_centers_m,
        depth_cell_centers_m=predictions.depth_cell_centers_m,
        predicted_log10_resistivity=predictions.log10_resistivity,
    )
    result = []
    for name, stream in (
        ("truth.npz", truth_stream),
        ("prediction.npz", prediction_stream),
    ):
        payload = stream.getvalue()
        result.append(
            ArtifactSnapshot(
                Path(name),
                payload,
                hashlib.sha256(payload).hexdigest(),
                len(result) + 301,
                len(result) + 401,
            )
        )
    return tuple(result)


def _metric_contract() -> dict:
    x_centres = np.linspace(-11_750.0, 11_750.0, 48, dtype="<f8")
    depth_centres = np.linspace(100.0, 6_400.0, 64, dtype="<f8")
    return {
        "aggregation_across_samples": "equal_sample_weight",
        "campaign_binding": "exact_observations_sha256_in_truth_and_prediction",
        "cell_edges": "midpoints_with_half_spacing_boundary_extrapolation",
        "grid_weighting": "normalized_physical_cell_area",
        "scoring_domain": {
            "depth_cell_edges_m": comparison.protected_evaluation2d.cell_edges_from_centers(
                depth_centres
            ).tolist(),
            "mask": "all_truth_grid_cells",
            "support": "full_grid_voronoi_cells_from_centers",
            "x_cell_edges_m": comparison.protected_evaluation2d.cell_edges_from_centers(
                x_centres
            ).tolist(),
        },
        "prediction_grid": "must_exactly_match_withheld_truth_grid",
        "quantity": "log10_resistivity_ohm_m",
        "sample_pairing": "exact_unique_sample_index",
    }


def _evaluation(run: LockedRun2D, campaign, evaluator: dict) -> dict:
    rmse = [1.0] * 500
    mae = [0.5] * 500
    return {
        "audience": "benchmark_operator_only_after_predictions_locked",
        "schema": "pimsr-sota-2d-evaluation",
        "schema_version": 3,
        "inputs": {
            "prediction_lock": {
                "input_manifest_sha256": "5" * 64,
                "preregistration_sha256": "3" * 64,
                "schema": "pimsr-sota-2d-predictions-lock",
                "schema_version": 2,
                "sha256": "4" * 64,
            },
            "observations": {
                "schema": "pimsr-sota-2d-observations",
                "schema_version": 1,
                "sha256": run.observations_sha256,
            },
            "prediction": {
                "schema": "pimsr-sota-2d-predictions",
                "schema_version": 2,
                "sha256": run.prediction_sha256,
                "size_bytes": run.prediction_size_bytes,
            },
            "truth": {
                "schema": "pimsr-sota-2d-truth",
                "schema_version": 2,
                "sha256": campaign.truth_sha256,
                "size_bytes": 101,
            },
            "operator_manifest": {
                "schema": "pimsr-sota-2d-scoring-manifest",
                "schema_version": 3,
                "sha256": campaign.operator_sha256,
                "size_bytes": campaign.operator_size_bytes,
            },
        },
        "run": {
            "adapter_source_sha256": run.adapter_source_sha256,
            "campaign_id": run.campaign_id,
            "checkpoint_sha256": run.checkpoint_sha256,
            "method_id": run.method_id,
            "runtime_sha256": run.runtime_sha256,
            "source_commit": run.source_commit,
            "source_sha256": run.source_sha256,
            "training_seed": run.training_seed,
        },
        "metric_contract": _metric_contract(),
        "bootstrap_contract": {
            "headline_eligible": False,
            "cross_method_effect_ci": False,
            "hierarchical": False,
        },
        "implementation": {
            "distribution_version": "test",
            "git_commit": evaluator["repository_commit"],
            "git_head_commit": "f" * 40,
            "git_dirty_tree": False,
            "numpy_version": np.__version__,
            "python_version": "test",
            "source_file": "evaluation2d.py",
            "source_sha256": evaluator["source_sha256"],
            "source_size_bytes": 1000,
        },
        "physics_misfit": {"included": False, "reason": "not computed"},
        "release_gate": {
            "predictions_locked": True,
            "public_release_allowed": False,
            "required_next_step": "compare",
        },
        "overall": {
            "n_samples": 500,
            "rmse_log10_resistivity": {
                "mean": {"estimate": 1.0},
                "median": {"estimate": 1.0},
            },
            "mae_log10_resistivity": {
                "mean": {"estimate": 0.5},
                "median": {"estimate": 0.5},
            },
        },
        "by_scenario": {},
        "per_depth": [],
        "per_sample": [
            {
                "has_fault": False,
                "mae_log10_resistivity": mae[index],
                "rmse_log10_resistivity": rmse[index],
                "sample_index": index,
                "scenario": comparison.FAMILY_IDS[(index // 5) // 20],
            }
            for index in range(500)
        ],
    }


def test_equal_family_weight_seed_mean_joint_pairing_and_permutation():
    rows = _effect_rows(seed_values=[-10.0, 1.0, 1.0, 1.0, 1.0])
    first = hierarchical_paired_bootstrap_2d(
        rows,
        training_seeds=_SEEDS,
        confidence=0.95,
        n_resamples=10_000,
        rng_seed=17,
    )
    second = hierarchical_paired_bootstrap_2d(
        list(reversed(rows)),
        training_seeds=_SEEDS,
        confidence=0.95,
        n_resamples=10_000,
        rng_seed=17,
    )
    # Mean of seeds is -1.2; median would be +1 and is intentionally not used.
    assert first.point[0, 0] == pytest.approx(-1.2)
    np.testing.assert_array_equal(first.point, second.point)
    np.testing.assert_array_equal(first.two_sided_lower, second.two_sided_lower)
    np.testing.assert_array_equal(first.one_sided_upper, second.one_sided_upper)


def test_constant_negative_effect_has_negative_pairwise_iut_upper_bound():
    result = hierarchical_paired_bootstrap_2d(
        _effect_rows(),
        training_seeds=_SEEDS,
        confidence=0.95,
        n_resamples=10_000,
        rng_seed=23,
    )
    # Equal family weighting: (-0.5 + -1.5) / 2, despite unequal base counts.
    assert result.point[0, 0] == pytest.approx(-1.0)
    assert np.all(result.one_sided_upper[:, 0] < 0.0)


def test_missing_seed_cell_is_rejected_before_bootstrap():
    rows = _effect_rows()
    rows.pop()
    with pytest.raises(Comparison2DValidationError, match="all paired training seeds"):
        hierarchical_paired_bootstrap_2d(
            rows,
            training_seeds=_SEEDS,
            confidence=0.95,
            n_resamples=10_000,
            rng_seed=1,
        )


def test_operator_requires_exact_500_100_by_5_and_preserves_families():
    value = _operator_manifest()
    snapshot = ArtifactSnapshot(Path("operator.json"), b"operator", _SHA_A, 1, 2)
    campaign = comparison._operator_campaign(
        value,
        snapshot=snapshot,
        campaign_id="campaign-a",
        locked_observations_sha256=_SHA_A,
        locked_observation_manifest_sha256=_SHA_C,
        family_commitment_sha256=_operator_commitment(value),
    )
    assert campaign.hierarchy.families == comparison.FAMILY_IDS
    assert len(campaign.hierarchy.sample_ids) == 500
    assert campaign.generation_evidence_proven is False

    value["split"]["groups"].pop()
    with pytest.raises(Comparison2DValidationError, match="one family/base/noise"):
        comparison._operator_campaign(
            value,
            snapshot=snapshot,
            campaign_id="campaign-a",
            locked_observations_sha256=_SHA_A,
            locked_observation_manifest_sha256=_SHA_C,
            family_commitment_sha256=_operator_commitment(value),
        )


def test_multidimensional_convergence_is_computed_not_trusted():
    report, residuals, residuals_snapshot, evidence = _convergence_report()
    analytic = {
        (truth, mesh): {
            channel: np.full((8, 12), 0.001, dtype=np.float64)
            for channel in comparison._CONVERGENCE_CHANNELS
        }
        for truth in (
            "analytic-halfspace-100",
            "analytic-layered-100-10-500",
        )
        for mesh in ("nested-production-v1", "nested-reference-x2-v1")
    }
    valid, reason = comparison._convergence_report_valid(
        report,
        evidence=evidence,
        residuals=residuals,
        residual_archive=residuals,
        analytic_residuals=analytic,
        residuals_snapshot=residuals_snapshot,
    )
    assert valid is True and reason is None

    report["candidate_vs_reference"]["aggregate"]["te_log10_rho"]["mean"] = 0.003
    valid, reason = comparison._convergence_report_valid(
        report,
        evidence=evidence,
        residuals=residuals,
        residual_archive=residuals,
        analytic_residuals=analytic,
        residuals_snapshot=residuals_snapshot,
    )
    assert valid is False
    assert "raw paired residuals" in str(reason)


def test_convergence_npz_schema_and_values_are_verified():
    report, _, residuals_snapshot, _ = _convergence_report()
    arrays = comparison._load_convergence_residuals(
        residuals_snapshot,
        report=report,
    )
    assert arrays["candidate_vs_reference_te_log10_rho"].shape == (25, 8, 12)

    bad_stream = io.BytesIO()
    np.savez_compressed(
        bad_stream,
        scenario_index=np.repeat(np.arange(5, dtype=np.int64), 5),
        sample_index=np.arange(25, dtype=np.int64),
    )
    payload = bad_stream.getvalue()
    bad_snapshot = ArtifactSnapshot(
        Path("bad.npz"), payload, hashlib.sha256(payload).hexdigest(), 11, 21
    )
    report["raw_paired_residuals"]["sha256"] = bad_snapshot.sha256
    report["raw_paired_residuals"]["size_bytes"] = bad_snapshot.size_bytes
    with pytest.raises(Comparison2DValidationError, match="member order/schema"):
        comparison._load_convergence_residuals(bad_snapshot, report=report)


def test_evaluator_v3_exactly_binds_lock_operator_run_and_source():
    operator = _operator_manifest()
    operator_snapshot = ArtifactSnapshot(Path("operator.json"), b"operator", _SHA_A, 1, 2)
    campaign = comparison._operator_campaign(
        operator,
        snapshot=operator_snapshot,
        campaign_id="campaign-a",
        locked_observations_sha256=_SHA_A,
        locked_observation_manifest_sha256=_SHA_C,
        family_commitment_sha256=_operator_commitment(operator),
    )
    run = _locked_run()
    locked = _validated_lock(run)
    evaluator = {"repository_commit": "e" * 40, "source_sha256": "6" * 64}
    value = _evaluation(run, campaign, evaluator)
    material_truth, material_predictions = _material_evaluation_inputs(campaign, run)
    result = comparison._evaluation_report(
        value,
        snapshot=ArtifactSnapshot(Path("evaluation.json"), b"eval", "7" * 64, 1, 3),
        campaign=campaign,
        run=run,
        validated_lock=locked,
        evaluator_contract=evaluator,
        material_truth=material_truth,
        material_predictions=material_predictions,
    )
    assert result.metrics.shape == (500, 2)

    fabricated_domain = json.loads(json.dumps(value))
    fabricated_domain["metric_contract"]["scoring_domain"]["depth_cell_edges_m"] = [
        0.0,
        1.0,
        2.0,
    ]
    with pytest.raises(Comparison2DValidationError, match="material truth grid"):
        comparison._evaluation_report(
            fabricated_domain,
            snapshot=ArtifactSnapshot(Path("evaluation.json"), b"eval", "7" * 64, 1, 3),
            campaign=campaign,
            run=run,
            validated_lock=locked,
            evaluator_contract=evaluator,
            material_truth=material_truth,
            material_predictions=material_predictions,
        )

    value["run"]["runtime_sha256"] = "8" * 64
    with pytest.raises(Comparison2DValidationError, match="run binding"):
        comparison._evaluation_report(
            value,
            snapshot=ArtifactSnapshot(Path("evaluation.json"), b"eval", "7" * 64, 1, 3),
            campaign=campaign,
            run=run,
            validated_lock=locked,
            evaluator_contract=evaluator,
            material_truth=material_truth,
            material_predictions=material_predictions,
        )


def test_material_truth_and_prediction_bytes_recompute_every_metric():
    operator = _operator_manifest()
    campaign = comparison._operator_campaign(
        operator,
        snapshot=ArtifactSnapshot(Path("operator.json"), b"operator", _SHA_A, 1, 2),
        campaign_id="campaign-a",
        locked_observations_sha256=_SHA_A,
        locked_observation_manifest_sha256=_SHA_C,
        family_commitment_sha256=_operator_commitment(operator),
    )
    run = _locked_run()
    truth_snapshot, prediction_snapshot = _material_npz_snapshots(campaign, run)
    truth = comparison._load_material_truth(truth_snapshot)
    predictions = comparison._load_material_predictions(prediction_snapshot)
    sample_ids, metrics = comparison._recomputed_material_metrics(truth, predictions)
    assert sample_ids == campaign.hierarchy.sample_ids
    np.testing.assert_array_equal(metrics[:, 0], np.ones(500))
    np.testing.assert_array_equal(metrics[:, 1], np.full(500, 0.5))

    evaluator = {"repository_commit": "e" * 40, "source_sha256": "6" * 64}
    report = _evaluation(run, campaign, evaluator)
    report["per_sample"][0]["rmse_log10_resistivity"] = 0.999
    report["overall"]["rmse_log10_resistivity"]["mean"]["estimate"] = 499.999 / 500.0
    # Bind fixture metadata to the exact captured material byte identities.
    report["inputs"]["truth"]["sha256"] = truth.artifact_sha256
    report["inputs"]["truth"]["size_bytes"] = truth.artifact_size_bytes
    report["inputs"]["prediction"]["sha256"] = predictions.artifact_sha256
    report["inputs"]["prediction"]["size_bytes"] = predictions.artifact_size_bytes
    material_campaign = comparison.replace(
        campaign,
        truth_sha256=truth.artifact_sha256,
        truth_size_bytes=truth.artifact_size_bytes,
    )
    material_run = comparison.replace(
        run,
        prediction_sha256=predictions.artifact_sha256,
        prediction_size_bytes=predictions.artifact_size_bytes,
    )
    with pytest.raises(Comparison2DValidationError, match="not recomputed"):
        comparison._evaluation_report(
            report,
            snapshot=ArtifactSnapshot(Path("evaluation.json"), b"eval", "7" * 64, 11, 12),
            campaign=material_campaign,
            run=material_run,
            validated_lock=_validated_lock(material_run),
            evaluator_contract=evaluator,
            material_truth=truth,
            material_predictions=predictions,
        )


def test_lock_validation_occurs_before_any_post_score_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    touched = False

    def fail_lock(*args, **kwargs):
        raise Comparison2DValidationError("lock rejected")

    def forbidden_snapshot(*args, **kwargs):
        nonlocal touched
        touched = True
        raise AssertionError("post-score path opened before lock")

    monkeypatch.setattr(comparison, "_validated_lock", fail_lock)
    monkeypatch.setattr(comparison, "_snapshot_unique", forbidden_snapshot)
    with pytest.raises(Comparison2DValidationError, match="lock rejected"):
        comparison.compare_evaluations_2d(
            tmp_path / "prereg.json",
            tmp_path / "lock.json",
            tmp_path / "post.json",
            expected_preregistration_sha256="0" * 64,
            expected_prediction_lock_sha256="1" * 64,
            expected_post_score_manifest_sha256="2" * 64,
        )
    assert touched is False


def test_publication_returns_reopened_receipt_and_leaves_sealed_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    destination = tmp_path / "comparison.json"
    report = {"schema": COMPARISON_SCHEMA, "value": 1}
    read_count = 0
    original_read_exact_fd = comparison._read_exact_fd

    def tracked_read(stream, size_bytes):
        nonlocal read_count
        read_count += 1
        return original_read_exact_fd(stream, size_bytes)

    monkeypatch.setattr(comparison, "_read_exact_fd", tracked_read)
    receipt = publish_comparison_2d(report, destination)
    assert receipt.path == destination
    assert receipt.sha256 == hashlib.sha256(canonical_json_bytes(report)).hexdigest()
    assert receipt.size_bytes == len(canonical_json_bytes(report))
    assert destination.read_bytes() == canonical_json_bytes(report)
    assert read_count == 3  # one writer verification plus two reopened-fd reads
    with pytest.raises(Comparison2DPublicationError, match="overwrite"):
        publish_comparison_2d(report, destination)

    interrupted = tmp_path / "interrupted.json"
    monkeypatch.setattr(
        comparison,
        "_read_exact_fd",
        lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    with pytest.raises(KeyboardInterrupt):
        publish_comparison_2d(report, interrupted)
    assert interrupted.exists()
    assert stat.S_IMODE(interrupted.stat().st_mode) & 0o222 == 0
    assert not list(tmp_path.glob(f".{interrupted.name}.*.part"))
    interrupted.chmod(stat.S_IWRITE)
    destination.chmod(stat.S_IWRITE)


def test_raw_modem_forward_is_independently_parsed_and_fabrication_rejected():
    snapshot = _forward_snapshot()
    response = comparison._parse_modem_forward_snapshot(
        snapshot, role="adversarial raw forward"
    )
    np.testing.assert_allclose(response.log10_rho_te, 2.0, atol=2e-10)
    np.testing.assert_allclose(response.log10_rho_tm, 3.0, atol=2e-10)
    np.testing.assert_allclose(response.phase_te_deg, 30.0, atol=2e-10)
    np.testing.assert_allclose(response.phase_tm_deg, 120.0, atol=2e-10)

    lines = snapshot.payload.decode("ascii").splitlines()
    payload = ("\n".join(lines[:-1]) + "\n").encode("ascii")
    fabricated = ArtifactSnapshot(
        Path("fabricated.dat"),
        payload,
        hashlib.sha256(payload).hexdigest(),
        93,
        94,
    )
    with pytest.raises(Comparison2DValidationError, match="96 TE and 96 TM"):
        comparison._parse_modem_forward_snapshot(
            fabricated, role="fabricated raw forward"
        )


def test_public_raw_manifest_requires_exact_80_and_real_artifacts(tmp_path: Path):
    with pytest.raises(Comparison2DValidationError, match="exactly 80"):
        comparison._validate_public_convergence_raw_runs(
            {
                "schema": "pimsr-modem2d-public-convergence-raw-run-set",
                "schema_version": 2,
                "source_lineage_split": "train",
                "source_shard_ordinal": 0,
                "source_shard": _identity("shard.h5", "d" * 64, 1),
                "runs": [{}] * 79,
            },
            report={},
            evidence={},
            public_lineage={},
            base=tmp_path,
            seen_identities=set(),
        )
    with pytest.raises(Comparison2DValidationError):
        comparison._artifact_reference(
            {
                "path": "nonexistent-forward.dat",
                "sha256": "a" * 64,
                "size_bytes": 1,
            },
            base=tmp_path,
            role="nonexistent raw forward",
            seen_identities=set(),
        )
    for nonportable in (str((tmp_path / "forward.dat").resolve()), "../forward.dat"):
        with pytest.raises(Comparison2DValidationError, match="portable relative"):
            comparison._portable_artifact_reference(
                {
                    "path": nonportable,
                    "sha256": "a" * 64,
                    "size_bytes": 1,
                },
                base=tmp_path,
                role="nonportable raw forward",
                seen_identities=set(),
            )


def test_public_convergence_source_is_recomputed_from_pinned_shard_material():
    stream = io.BytesIO()
    x_grid = np.arange(-11_750.0, 12_000.0, 500.0, dtype="<f8")
    depth_grid = np.asarray(
        _analytic_contract()["canonical_depth_centres_m"], dtype="<f8"
    )
    frequencies = np.geomspace(0.01, 100.0, 8).astype("<f8")
    station_x = np.linspace(-5_500.0, 5_500.0, 12).astype("<f8")
    sample_index = np.arange(100, dtype="<i8")
    scenarios = np.repeat(np.arange(5, dtype="<i4"), 20)
    target = np.zeros((100, 64, 48), dtype="<f4")
    for row in range(100):
        target[row].fill(np.float32(1.0 + row / 100.0))
    with h5py.File(stream, "w") as h5:
        h5.attrs["schema"] = "pimsr-mt-2d"
        h5.attrs["schema_version"] = 2
        h5.attrs["generator_seed"] = 20260820
        h5.attrs["generation_contract"] = "pimsr-geogen.SectionGenerator/default-grid/v1"
        h5.attrs["forward_contract"] = "pimsr-forward.MT2DForward/default-mesh/v2"
        h5.create_dataset("target_log10_res", data=target)
        h5.create_dataset("x_grid", data=x_grid)
        h5.create_dataset("depth_grid", data=depth_grid)
        h5.create_dataset("frequencies", data=frequencies)
        h5.create_dataset("station_x", data=station_x)
        h5.create_dataset("sample_index", data=sample_index)
        h5.create_dataset("scenario", data=scenarios)
    payload = stream.getvalue()
    snapshot = ArtifactSnapshot(
        Path("shard-000000-000099.h5"),
        payload,
        hashlib.sha256(payload).hexdigest(),
        501,
        502,
    )
    material = comparison._load_public_shard_material(
        snapshot,
        expected_generator_seed=20260820,
        expected_sample_start=0,
        expected_sample_end=99,
    )
    assert material.row_by_sample[40] == 40
    assert material.scenario_by_sample[40] == 2
    assert material.truth_identity_by_sample[40]["model_shape"] == [64, 48]
    selected = tuple(
        sample_id
        for scenario_index in range(5)
        for sample_id in range(scenario_index * 20, scenario_index * 20 + 5)
    )
    assert (
        comparison._validated_public_selection(
            [{"sample_index": sample_id} for sample_id in selected],
            source_material=material,
        )
        == selected
    )
    cherry_picked = [{"sample_index": sample_id} for sample_id in selected]
    cherry_picked[-1] = {"sample_index": 99}
    with pytest.raises(Comparison2DValidationError, match="lowest five per family"):
        comparison._validated_public_selection(
            cherry_picked,
            source_material=material,
        )
    with pytest.raises(Comparison2DValidationError, match="material contract"):
        comparison._load_public_shard_material(
            snapshot,
            expected_generator_seed=20260820,
            expected_sample_start=100,
            expected_sample_end=199,
        )
    source = {
        "source": _identity("shard-000000-000099.h5", snapshot.sha256, len(payload)),
        "row": 40,
        "sample_index": 40,
        "scenario_index": 2,
        "generator_seed": 20260820,
        "generation_contract": "pimsr-geogen.SectionGenerator/default-grid/v1",
        "forward_contract": "pimsr-forward.MT2DForward/default-mesh/v2",
        "public_validation": {
            "selection_policy": "lowest sample_index per frozen scenario",
            "scenario_name": "hydrocarbon",
            "validator_source": _identity("validator.py", "f" * 64, 3),
        },
    }
    comparison._public_truth_source(
        source,
        path="public truth",
        sample_index=40,
        scenario_index=2,
        scenario="hydrocarbon",
        source_material=material,
        expected_source_reference=source["source"],
    )
    source["row"] = 41
    with pytest.raises(Comparison2DValidationError, match="selection provenance"):
        comparison._public_truth_source(
            source,
            path="public truth",
            sample_index=40,
            scenario_index=2,
            scenario="hydrocarbon",
            source_material=material,
            expected_source_reference=source["source"],
        )
    source["row"] = 40
    with pytest.raises(Comparison2DValidationError, match="selection provenance"):
        comparison._public_truth_source(
            source,
            path="public truth",
            sample_index=40,
            scenario_index=2,
            scenario="hydrocarbon",
            source_material=material,
            expected_source_reference=_identity(
                "different-shard-path.h5", snapshot.sha256, len(payload)
            ),
        )


def test_locked_family_commitment_rejects_post_lock_relabel():
    operator = _operator_manifest()
    commitment = _operator_commitment(operator)
    snapshot = ArtifactSnapshot(Path("operator.json"), b"operator", _SHA_A, 1, 2)
    comparison._operator_campaign(
        operator,
        snapshot=snapshot,
        campaign_id="campaign-a",
        locked_observations_sha256=_SHA_A,
        locked_observation_manifest_sha256=_SHA_C,
        family_commitment_sha256=commitment,
    )
    operator["split"]["family_partition_reveal"]["rows"][0]["family_id"] = "salt"
    with pytest.raises(Comparison2DValidationError, match="exact operator hierarchy"):
        comparison._operator_campaign(
            operator,
            snapshot=snapshot,
            campaign_id="campaign-a",
            locked_observations_sha256=_SHA_A,
            locked_observation_manifest_sha256=_SHA_C,
            family_commitment_sha256=commitment,
        )


def test_lock_pinned_observation_manifest_supplies_commitment_not_prereg():
    operator = _operator_manifest()
    commitment = _operator_commitment(operator)
    manifest = {
        "audience": "method_input_public",
        "declared_evaluation_floors": {},
        "observation_payload": {
            "schema": "pimsr-sota-2d-observations",
            "schema_version": 1,
            "sha256": _SHA_A,
            "size_bytes": 123,
        },
        "physical_contract": {},
        "sample_count": 500,
        "schema": "pimsr-sota-2d-observation-manifest",
        "schema_version": 3,
        "split_id": "campaign-a",
        "family_partition_commitment": {
            "schema": "pimsr-sota-2d-family-partition-commitment",
            "schema_version": 1,
            "sha256": commitment,
            "contract": _commitment_contract(),
        },
    }
    payload = canonical_json_bytes(manifest)
    snapshot = ArtifactSnapshot(
        Path("observations.json"),
        payload,
        hashlib.sha256(payload).hexdigest(),
        1,
        2,
    )
    assert comparison._public_observation_family_commitment(
        snapshot,
        campaign_id="campaign-a",
        expected_observations_sha256=_SHA_A,
        expected_contract=_commitment_contract(),
    ) == (commitment, 123)

    manifest["family_partition_commitment"]["contract"]["nonce_encoding"] = (
        "caller_selected"
    )
    payload = canonical_json_bytes(manifest)
    tampered = ArtifactSnapshot(
        Path("observations.json"),
        payload,
        hashlib.sha256(payload).hexdigest(),
        3,
        4,
    )
    with pytest.raises(Comparison2DValidationError, match="differs from preregistration"):
        comparison._public_observation_family_commitment(
            tampered,
            campaign_id="campaign-a",
            expected_observations_sha256=_SHA_A,
            expected_contract=_commitment_contract(),
        )


def test_hidden_generation_v3_requires_100_solves_and_500_noise_rows():
    operator = _operator_manifest()
    commitment = _operator_commitment(operator)
    campaign = comparison._operator_campaign(
        operator,
        snapshot=ArtifactSnapshot(Path("operator.json"), b"operator", _SHA_A, 1, 2),
        campaign_id="campaign-a",
        locked_observations_sha256=_SHA_A,
        locked_observation_manifest_sha256=_SHA_C,
        family_commitment_sha256=commitment,
    )
    public_lineage = _hidden_public_lineage_identity()
    material_truth, _ = _material_evaluation_inputs(campaign, _locked_run())
    closure = {
        "schema": "pimsr-modem2d-hidden-generation-closure",
        "schema_version": 3,
        "campaign_id": "campaign-a",
        "mesh": {},
        "runtime": {},
        "runtime_identity_sha256": "1" * 64,
        "bindings": {
            "operator_manifest_sha256": _SHA_A,
            "observations_sha256": _SHA_A,
            "public_observation_manifest_sha256": _SHA_C,
            "withheld_truth_sha256": _SHA_B,
            "family_partition_commitment_sha256": commitment,
        },
        "generation_contract": {
            "schema": "pimsr-modem2d-hidden-generation-contract",
            "schema_version": 2,
            "generator_seed": 20260830,
            "base_layer_rng": "numpy.default_rng([generator_seed,base_index])",
            "base_layer_scenario": ("forced_background_before_2d_scenario_injection"),
            "section_rng": "numpy.default_rng([generator_seed,2,base_index])",
            "scenario_policy": ("SectionGenerator.sample(base_index,scenario=family_id)"),
            "noise_rng": ("numpy.default_rng([generator_seed,3,base_index,noise_index])"),
            "geology_contract": "pimsr-geogen.SectionGenerator/default-grid/v1",
            "clean_forward_contract": ("pinned_modem_2d_raw_forward_per_unique_base/v1"),
            "noise_contract": "pimsr-forward.SensorModel/mt-noise+tm-severity-v5/v1",
            "source_lineage": comparison._hidden_source_lineage_identity(public_lineage),
            "base_count": 100,
            "noise_realizations_per_base": 5,
        },
        "generation_runtime": dict(comparison._HIDDEN_GENERATION_RUNTIME),
        "generation_runtime_manifest": _identity("generation-runtime.json", "6" * 64, 16),
        "observation_payload": {},
        "base_forward_runs": [{}] * 99,
        "noise_rows": [{}] * 500,
    }
    tampered_source = json.loads(json.dumps(closure))
    tampered_source["generation_contract"]["source_lineage"]["pimsr_geogen"][
        "section2d_source_sha256"
    ] = "f" * 64
    with pytest.raises(Comparison2DValidationError, match="material protocol"):
        comparison._validate_hidden_generation_closure(
            tampered_source,
            campaign=campaign,
            material_truth=material_truth,
            evidence={},
            public_lineage=public_lineage,
            expected_generation_runtime_manifest=closure["generation_runtime_manifest"],
            expected_observation_payload_size=1,
            base=Path("."),
            seen_identities=set(),
        )
    different_runtime_manifest = _identity(
        "different-generation-runtime.json", "7" * 64, 16
    )
    with pytest.raises(
        Comparison2DValidationError,
        match="runtime manifest differs from captured preregistered evidence",
    ):
        comparison._validate_hidden_generation_closure(
            closure,
            campaign=campaign,
            material_truth=material_truth,
            evidence={},
            public_lineage=public_lineage,
            expected_generation_runtime_manifest=different_runtime_manifest,
            expected_observation_payload_size=1,
            base=Path("."),
            seen_identities=set(),
        )
    with pytest.raises(Comparison2DValidationError, match="100 clean ModEM"):
        comparison._validate_hidden_generation_closure(
            closure,
            campaign=campaign,
            material_truth=material_truth,
            evidence={},
            public_lineage=public_lineage,
            expected_generation_runtime_manifest=closure["generation_runtime_manifest"],
            expected_observation_payload_size=1,
            base=Path("."),
            seen_identities=set(),
        )
    closure["base_forward_runs"] = [{}] * 100
    closure["noise_rows"] = [{}] * 499
    with pytest.raises(Comparison2DValidationError, match="500 noise-row"):
        comparison._validate_hidden_generation_closure(
            closure,
            campaign=campaign,
            material_truth=material_truth,
            evidence={},
            public_lineage=public_lineage,
            expected_generation_runtime_manifest=closure["generation_runtime_manifest"],
            expected_observation_payload_size=1,
            base=Path("."),
            seen_identities=set(),
        )


def test_hidden_observation_material_validation_rejects_phase_and_floor_fabrication():
    operator = _operator_manifest()
    campaign = comparison._operator_campaign(
        operator,
        snapshot=ArtifactSnapshot(Path("operator.json"), b"operator", _SHA_A, 1, 2),
        campaign_id="campaign-a",
        locked_observations_sha256=_SHA_A,
        locked_observation_manifest_sha256=_SHA_C,
        family_commitment_sha256=_operator_commitment(operator),
    )
    rows = comparison._load_hidden_observation_rows(
        _hidden_observation_snapshot(), campaign=campaign
    )
    assert len(rows) == 500

    with pytest.raises(Comparison2DValidationError, match=r"\[0,180\)"):
        comparison._load_hidden_observation_rows(
            _hidden_observation_snapshot(phase_value=180.0), campaign=campaign
        )
    with pytest.raises(Comparison2DValidationError, match="floors are invalid"):
        comparison._load_hidden_observation_rows(
            _hidden_observation_snapshot(floor_value=0.0), campaign=campaign
        )


def test_hidden_truth_proves_100_distinct_bases_and_five_identical_noise_rows():
    operator = _operator_manifest()
    campaign = comparison._operator_campaign(
        operator,
        snapshot=ArtifactSnapshot(Path("operator.json"), b"operator", _SHA_A, 1, 2),
        campaign_id="campaign-a",
        locked_observations_sha256=_SHA_A,
        locked_observation_manifest_sha256=_SHA_C,
        family_commitment_sha256=_operator_commitment(operator),
    )
    truth, _ = _material_evaluation_inputs(campaign, _locked_run())
    values = np.array(truth.log10_resistivity, copy=True)
    expected_bases = []
    sample_ids_by_base = {}
    for family in campaign.hierarchy.families:
        for base_model_id, noise_rows in campaign.hierarchy.tree[family]:
            sample_ids = tuple(sample_id for _noise, sample_id in sorted(noise_rows))
            base_index = len(expected_bases)
            for sample_id in sample_ids:
                values[sample_id].fill(np.float32(1.0 + base_index / 100.0))
            expected_bases.append((family, base_model_id, sample_ids))
            sample_ids_by_base[base_model_id] = sample_ids
    material = comparison.replace(truth, log10_resistivity=values)
    frequencies = np.geomspace(0.01, 100.0, 8).astype("<f8")
    stations = np.linspace(-5_500.0, 5_500.0, 12).astype("<f8")
    observation_rows = {
        sample_id: {"frequency_hz": frequencies, "station_x_m": stations}
        for sample_id in campaign.hierarchy.sample_ids
    }

    class FakeGenerator:
        def sample(self, base_index: int, *, scenario: str):
            return SimpleNamespace(
                scenario=scenario,
                seed=base_index,
                x_grid=material.x_cell_centers_m,
                depth_grid=material.depth_cell_centers_m,
                log10_res=np.full(
                    (64, 48), np.float32(1.0 + base_index / 100.0), dtype="<f4"
                ),
                has_fault=False,
            )

    factory = lambda _seed: FakeGenerator()
    base_truths = comparison._material_hidden_base_truths(
        generator_seed=20260830,
        material_truth=material,
        campaign=campaign,
        expected_bases=expected_bases,
        sample_ids_by_base=sample_ids_by_base,
        observation_rows=observation_rows,
        section_generator_factory=factory,
    )
    assert len(base_truths) == 100

    changed = np.array(values, copy=True)
    changed[1, 0, 0] += np.float32(0.1)
    with pytest.raises(Comparison2DValidationError, match="byte-identical"):
        comparison._material_hidden_base_truths(
            generator_seed=20260830,
            material_truth=comparison.replace(material, log10_resistivity=changed),
            campaign=campaign,
            expected_bases=expected_bases,
            sample_ids_by_base=sample_ids_by_base,
            observation_rows=observation_rows,
            section_generator_factory=factory,
        )

    duplicated = np.array(values, copy=True)
    duplicated[5:10] = duplicated[0]
    with pytest.raises(Comparison2DValidationError, match="byte-exact"):
        comparison._material_hidden_base_truths(
            generator_seed=20260830,
            material_truth=comparison.replace(material, log10_resistivity=duplicated),
            campaign=campaign,
            expected_bases=expected_bases,
            sample_ids_by_base=sample_ids_by_base,
            observation_rows=observation_rows,
            section_generator_factory=factory,
        )

    substituted = np.array(values, copy=True)
    substituted[:5] += np.float32(0.01)
    with pytest.raises(Comparison2DValidationError, match="byte-exact"):
        comparison._material_hidden_base_truths(
            generator_seed=20260830,
            material_truth=comparison.replace(material, log10_resistivity=substituted),
            campaign=campaign,
            expected_bases=expected_bases,
            sample_ids_by_base=sample_ids_by_base,
            observation_rows=observation_rows,
            section_generator_factory=factory,
        )


def test_hidden_raw_modem_inputs_are_exact_protected_writer_bytes(tmp_path: Path):
    mesh_record = _mesh_record("nested-production-v1")
    mesh = comparison._nested_mesh_from_record(mesh_record)
    truth = modem2d_forward.CanonicalTruth(
        log10_resistivity=np.full((64, 48), 2.0, dtype="<f4"),
        x_centres_m=np.arange(-11_750.0, 12_000.0, 500.0, dtype="<f8"),
        depth_centres_m=np.asarray(
            _analytic_contract()["canonical_depth_centres_m"], dtype="<f8"
        ),
        frequencies_hz=np.geomspace(0.01, 100.0, 8).astype("<f8"),
        station_x_m=np.linspace(-5_500.0, 5_500.0, 12).astype("<f8"),
        sample_id="base-0",
    )
    expected_model, expected_template = comparison._render_modem_input_bytes(truth, mesh)
    model_snapshot, _ = modem2d_forward.write_modem_model(
        tmp_path / "model.rho", truth, mesh
    )
    template_snapshot, _ = modem2d_forward.write_modem_template(
        tmp_path / "template.dat", truth, mesh
    )
    assert model_snapshot.payload == expected_model
    assert template_snapshot.payload == expected_template

    changed = np.array(truth.log10_resistivity, copy=True)
    changed[0, 0] += 0.25
    changed_truth = modem2d_forward.CanonicalTruth(
        log10_resistivity=changed,
        x_centres_m=truth.x_centres_m,
        depth_centres_m=truth.depth_centres_m,
        frequencies_hz=truth.frequencies_hz,
        station_x_m=truth.station_x_m,
        sample_id=truth.sample_id,
    )
    changed_model, changed_template = comparison._render_modem_input_bytes(
        changed_truth, mesh
    )
    assert changed_model != expected_model
    assert changed_template == expected_template


def test_hidden_noise_is_recomputed_from_exact_seed_not_self_attested():
    clean = comparison._parse_modem_forward_snapshot(
        _forward_snapshot(), role="clean hidden ModEM response"
    )
    expected = comparison._expected_hidden_noisy_response(
        clean,
        generator_seed=20260830,
        base_index=7,
        noise_index=3,
    )
    comparison._validate_hidden_noise_realization(
        expected,
        clean,
        generator_seed=20260830,
        base_index=7,
        noise_index=3,
        path="noise row",
    )
    tampered = {name: np.array(values, copy=True) for name, values in expected.items()}
    tampered["observed_log10_rho_te"][0, 0] += np.float32(0.001)
    with pytest.raises(Comparison2DValidationError, match="seeded noise"):
        comparison._validate_hidden_noise_realization(
            tampered,
            clean,
            generator_seed=20260830,
            base_index=7,
            noise_index=3,
            path="noise row",
        )


def test_hidden_campaign_seeds_must_match_preregistered_ordered_commitment():
    seeds = [20260830, 20260831, 20260832, 20260833, 20260834]
    payload = json.dumps(seeds, separators=(",", ":")).encode("utf-8")
    commitment = {
        "encoding": "utf8-canonical-json-int64-array-no-newline-v1",
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    identity = comparison._validate_hidden_campaign_seed_reveal(seeds, commitment)
    assert identity["verified"] is True
    assert identity["campaign_count"] == 5

    with pytest.raises(Comparison2DValidationError, match="do not match"):
        comparison._validate_hidden_campaign_seed_reveal(
            [*seeds[:-2], seeds[-1], seeds[-2]], commitment
        )
    with pytest.raises(Comparison2DValidationError, match="five distinct"):
        comparison._validate_hidden_campaign_seed_reveal([seeds[0]] * 5, commitment)


def test_comparison_implementation_gate_precedes_post_score_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    locked = _validated_lock(_locked_run())
    prereg_path = tmp_path / "prereg.json"
    post_path = tmp_path / "post.json"
    opened: list[Path] = []

    monkeypatch.setattr(comparison, "_validated_lock", lambda *args, **kwargs: locked)
    monkeypatch.setattr(comparison, "_validate_statistical_options", lambda value: {})

    def snapshot(path, **kwargs):
        opened.append(Path(path))
        return ArtifactSnapshot(Path(path), b"{}", "3" * 64, 11, 12)

    monkeypatch.setattr(comparison, "_snapshot_unique", snapshot)
    monkeypatch.setattr(comparison, "_strict_json", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        comparison,
        "_prereg_contracts",
        lambda *args: SimpleNamespace(
            evaluator={},
            headline_evidence={},
            comparison_implementation={},
            public_lineage={},
            family_partition={},
        ),
    )
    monkeypatch.setattr(
        comparison,
        "_validate_comparison_implementation",
        lambda value: (_ for _ in ()).throw(
            Comparison2DValidationError("implementation pin rejected")
        ),
    )
    with pytest.raises(Comparison2DValidationError, match="implementation pin rejected"):
        comparison.compare_evaluations_2d(
            prereg_path,
            tmp_path / "lock.json",
            post_path,
            expected_preregistration_sha256="3" * 64,
            expected_prediction_lock_sha256="4" * 64,
            expected_post_score_manifest_sha256="5" * 64,
        )
    assert opened == [prereg_path]


def test_comparison_git_gate_matches_captured_bytes_to_exact_pinned_blobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repository = tmp_path / "repository"
    source_path = repository / comparison._COMPARISON_SOURCE_PATH
    paths = (
        comparison._COMPARISON_SOURCE_PATH,
        *comparison._REQUIRED_COMPARISON_DEPENDENCIES,
    )
    assert "src/pimsr_benchmarks/_publication_io.py" in paths
    payloads = {}
    for index, relative in enumerate(paths):
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = f"# protected {index}\nVALUE = {index}\n".encode()
        path.write_bytes(payload)
        payloads[relative] = payload
    (repository / ".gitattributes").write_text("*.py text eol=lf\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.name", "Comparator Test"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "test@example.test"],
        check=True,
    )
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-q", "-m", "fixture"],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    records = [
        {
            "path": relative,
            "sha256": hashlib.sha256(payloads[relative]).hexdigest(),
            "size_bytes": len(payloads[relative]),
        }
        for relative in paths
    ]
    contract = {
        "repository_commit": commit,
        "source": records[0],
        "dependencies": tuple(records[1:]),
        "protected_paths": paths,
    }
    monkeypatch.setattr(comparison, "__file__", str(source_path))
    identity = comparison._validate_comparison_implementation(contract)
    assert all(
        record["matches_pinned_blob_after_git_clean_filter"] is True
        for record in identity["sources"]
    )

    dependency = repository / comparison._REQUIRED_COMPARISON_DEPENDENCIES[0]
    dependency.write_bytes(
        payloads[comparison._REQUIRED_COMPARISON_DEPENDENCIES[0]][:-1] + b"X"
    )
    with pytest.raises(Comparison2DValidationError):
        comparison._validate_comparison_implementation(contract)


def test_artifact_snapshots_reject_hardlink_aliases(tmp_path: Path):
    first = tmp_path / "first.json"
    alias = tmp_path / "alias.json"
    first.write_bytes(b"{}\n")
    os.link(first, alias)
    digest = hashlib.sha256(b"{}\n").hexdigest()
    with pytest.raises(Comparison2DValidationError, match="hardlink aliases"):
        comparison._snapshot_unique(
            first,
            expected_sha256=digest,
            expected_size_bytes=3,
            role="adversarial alias",
            seen_identities=set(),
        )


def test_publication_never_path_rereads_and_seals_final_inode_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    destination = tmp_path / "sealed.json"
    report = {"schema": COMPARISON_SCHEMA, "value": 2}

    def forbidden_read_bytes(self):
        raise AssertionError(f"path re-read is forbidden: {self}")

    with monkeypatch.context() as context:
        context.setattr(Path, "read_bytes", forbidden_read_bytes)
        publish_comparison_2d(report, destination)
    assert stat.S_IMODE(destination.stat().st_mode) & 0o222 == 0
    with destination.open("rb") as stream:
        assert stream.read() == canonical_json_bytes(report)
    destination.chmod(stat.S_IWRITE)


def test_publication_receipt_rejects_same_inode_same_size_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    destination = tmp_path / "mutated.json"
    report = {"schema": COMPARISON_SCHEMA, "value": 3}
    payload = canonical_json_bytes(report)
    retained_descriptor: int | None = None
    original_seal = comparison._seal_publication_descriptor

    def retain_writable_descriptor(stream, identity):
        nonlocal retained_descriptor
        retained_descriptor = os.dup(stream.fileno())
        original_seal(stream, identity)

    def mutate_after_path_verification(_parent):
        assert retained_descriptor is not None
        os.lseek(retained_descriptor, 0, os.SEEK_SET)
        os.write(retained_descriptor, b"X" * len(payload))
        os.fsync(retained_descriptor)

    monkeypatch.setattr(
        comparison,
        "_seal_publication_descriptor",
        retain_writable_descriptor,
    )
    monkeypatch.setattr(comparison, "_fsync_directory", mutate_after_path_verification)
    try:
        with pytest.raises(
            Comparison2DPublicationError,
            match="published comparison",
        ):
            publish_comparison_2d(report, destination)
    finally:
        if retained_descriptor is not None:
            os.close(retained_descriptor)
    assert destination.stat().st_size == len(payload)
    assert stat.S_IMODE(destination.stat().st_mode) & 0o222 == 0
    assert destination.read_bytes() == b"X" * len(payload)
    destination.chmod(stat.S_IWRITE)


@pytest.mark.skipif(os.name != "nt", reason="Windows share-mode contract")
def test_publication_receipt_rejects_a_retained_writable_windows_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    destination = tmp_path / "retained-writer.json"
    report = {"schema": COMPARISON_SCHEMA, "value": 33}
    retained_descriptor: int | None = None
    original_seal = comparison._seal_publication_descriptor

    def retain_writable_descriptor(stream, identity):
        nonlocal retained_descriptor
        retained_descriptor = os.dup(stream.fileno())
        original_seal(stream, identity)

    monkeypatch.setattr(
        comparison,
        "_seal_publication_descriptor",
        retain_writable_descriptor,
    )
    try:
        with pytest.raises(
            Comparison2DPublicationError,
            match="stable published comparison receipt",
        ):
            publish_comparison_2d(report, destination)
    finally:
        if retained_descriptor is not None:
            os.close(retained_descriptor)
    assert destination.read_bytes() == canonical_json_bytes(report)
    assert stat.S_IMODE(destination.stat().st_mode) & 0o222 == 0
    destination.chmod(stat.S_IWRITE)


def test_publication_parent_identity_change_leaves_sealed_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    destination = tmp_path / "parent-race.json"
    report = {"schema": COMPARISON_SCHEMA, "value": 4}
    original_identities = comparison._publication_parent_identities
    call_count = 0

    def simulate_parent_replacement(paths):
        nonlocal call_count
        call_count += 1
        identities = original_identities(paths)
        if call_count == 4:
            return ((identities[0][0], identities[0][1] + 1), *identities[1:])
        return identities

    monkeypatch.setattr(
        comparison,
        "_publication_parent_identities",
        simulate_parent_replacement,
    )
    with pytest.raises(Comparison2DPublicationError, match="parent changed"):
        publish_comparison_2d(report, destination)
    assert destination.exists()
    assert stat.S_IMODE(destination.stat().st_mode) & 0o222 == 0
    assert destination.read_bytes() == canonical_json_bytes(report)
    destination.chmod(stat.S_IWRITE)


def test_publication_reopened_receipt_is_stable_across_repeated_windows_opens(
    tmp_path: Path,
):
    report = {"schema": COMPARISON_SCHEMA, "value": 5}
    expected = canonical_json_bytes(report)
    for index in range(20):
        destination = tmp_path / f"reopened-{index:02d}.json"
        receipt = publish_comparison_2d(report, destination)
        assert receipt.sha256 == hashlib.sha256(expected).hexdigest()
        assert destination.read_bytes() == expected
        destination.chmod(stat.S_IWRITE)


def test_public_lineage_is_exact_and_modem_must_be_materially_distinct():
    assert comparison._PUBLIC_LINEAGE_EVIDENCE_SCOPE == dataset_lineage2d.EVIDENCE_SCOPE
    manifest, expected = _lineage_manifest()
    identity = comparison._validate_lineage_identity(
        manifest, split="train", expected_dataset=expected
    )
    lineage = {"train": identity, "validation": identity}
    generator = {
        "repository_commit": "c" * 40,
        "source_sha256": "e" * 64,
    }
    comparison._validate_generator_distinct_from_lineage(generator, lineage)

    generator["repository_commit"] = "a" * 40
    with pytest.raises(Comparison2DValidationError, match="not materially distinct"):
        comparison._validate_generator_distinct_from_lineage(generator, lineage)

    manifest["verification"]["forward_regeneration_performed"] = True
    with pytest.raises(Comparison2DValidationError, match="frozen public schedule"):
        comparison._validate_lineage_identity(
            manifest, split="train", expected_dataset=expected
        )

    manifest, expected = _lineage_manifest()
    manifest["repositories"]["pimsr_forward"]["source_files"][
        "src/pimsr_forward/dataset2d.py"
    ]["matches_commit_blob_after_git_clean_filter"] = False
    with pytest.raises(Comparison2DValidationError, match="pinned commit blob"):
        comparison._validate_lineage_identity(
            manifest, split="train", expected_dataset=expected
        )
