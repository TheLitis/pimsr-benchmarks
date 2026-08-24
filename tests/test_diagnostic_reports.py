"""Regression tests for explicitly non-rankable legacy diagnostics."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for source_root in (
    ROOT / "src",
    ROOT.parent / "pimsr-geogen" / "src",
    ROOT.parent / "pimsr-forward" / "src",
    ROOT.parent / "pimsr-inversion" / "src",
):
    sys.path.insert(0, str(source_root))


def _load_script(name: str) -> ModuleType:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"diagnostic_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _copy_legacy_inputs(destination: Path) -> None:
    destination.mkdir()
    for name in (
        "occam_synthetic.json",
        "neural_synthetic.json",
        "occam_real.json",
        "neural_real.json",
    ):
        (destination / name).write_bytes((ROOT / "results" / name).read_bytes())


def test_make_report_requires_explicit_legacy_opt_in(tmp_path: Path) -> None:
    script = _load_script("make_report.py")
    with pytest.raises(SystemExit, match="2"):
        script.main(
            [
                "--results-dir",
                str(ROOT / "results"),
                "--out",
                str(tmp_path / "report.md"),
            ]
        )
    assert not (tmp_path / "report.md").exists()


def test_make_report_is_non_rankable_and_refuses_overwrite(tmp_path: Path) -> None:
    script = _load_script("make_report.py")
    output = tmp_path / "report.md"
    args = [
        "--results-dir",
        str(ROOT / "results"),
        "--out",
        str(output),
        "--allow-legacy-diagnostic",
    ]

    script.main(args)
    report = output.read_text(encoding="utf-8")
    assert "comparison_status: `diagnostic_non_comparable`" in report
    assert "ranking_allowed: `false`" in report
    assert "Occam 1D (MT-only)" in report
    assert "PIMSR neural (MT+gravity)" in report
    assert "not a valid cross-method conclusion" in report

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        script.main(args)
    assert output.read_text(encoding="utf-8") == report


@pytest.mark.parametrize(
    ("filename", "field", "value", "message"),
    [
        (
            "neural_synthetic.json",
            "ranking_allowed",
            True,
            "ranking_allowed must be false",
        ),
        (
            "occam_real.json",
            "comparison_status",
            "benchmark_complete",
            "must mark a non-comparable diagnostic",
        ),
        (
            "neural_real.json",
            "metric_id",
            "different_metric",
            "metric_id must be declared by both inputs",
        ),
    ],
)
def test_make_report_rejects_incompatible_input_declarations(
    tmp_path: Path,
    filename: str,
    field: str,
    value: object,
    message: str,
) -> None:
    script = _load_script("make_report.py")
    inputs = tmp_path / "inputs"
    _copy_legacy_inputs(inputs)
    path = inputs / filename
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        script.render_diagnostic_report(inputs)


def test_hybrid2d_requires_opt_in_before_loading_artifacts(tmp_path: Path) -> None:
    script = _load_script("run_2d_hybrid_bench.py")
    with pytest.raises(SystemExit, match="2"):
        script.main(
            [
                "--checkpoint",
                "missing.pt",
                "--test-h5",
                "missing.h5",
                "--emtf-dir",
                "missing-emtf",
                "--out-dir",
                str(tmp_path),
            ]
        )


def test_hybrid2d_contract_discloses_budgets_and_geometry() -> None:
    script = _load_script("run_2d_hybrid_bench.py")
    geometry = {
        "geometry_policy": "affine_normalized_to_model_span",
        "publishable_physical_geometry": False,
    }

    contract = script._diagnostic_contract(geometry, include_cold=True)

    assert contract["comparison_status"] == "diagnostic_non_comparable"
    assert contract["ranking_allowed"] is False
    assert contract["headline_claim_allowed"] is False
    assert contract["score_observation_modes"] == ["te", "tm"]
    budgets = contract["inverse_observation_budget"]
    assert budgets["unet"] == ["te", "tm"]
    assert budgets["hybrid"]["warm_start"] == ["te", "tm"]
    assert budgets["hybrid"]["gauss_newton_refinement"] == ["tm"]
    assert budgets["cold"] == ["tm"]
    assert any("normalized" in reason for reason in contract["diagnostic_reasons"])

    with pytest.raises(ValueError, match="non-publishable normalized geometry"):
        script._diagnostic_contract(
            {"publishable_physical_geometry": True},
            include_cold=False,
        )


def _write_frozen_fixture(root: Path) -> Path:
    manifest = {
        "schema_version": 2,
        "status": "legacy_non_comparable",
        "metric": {
            "id": "section_nrms_2d_legacy_zxy_unmasked_v1",
            "ground_truth_claim": False,
            "comparable_to_current_metric": False,
        },
        "repositories": {"pimsr-benchmarks": "deadbeef"},
        "datasets": {"legacy": {"artifact_id": 1}},
        "checkpoints": {"legacy": {"artifact_id": 2}},
        "profiles": ["G", "H-YS", "I", "J", "K"],
        "limitations": ["legacy diagnostic"],
    }
    profiles = {
        "profiles": {
            profile: {"legacy-method": float(index + 1)}
            for index, profile in enumerate(manifest["profiles"])
        }
    }
    (root / "config").mkdir(parents=True)
    (root / "results/unified").mkdir(parents=True)
    (root / "results/v4").mkdir(parents=True)
    manifest_path = root / "config/frozen_2d.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (root / "results/unified/unified.json").write_text(
        json.dumps({"legacy-method": {"nrms_2d": 2.0}}),
        encoding="utf-8",
    )
    (root / "results/v4/v4_profiles_bal.json").write_text(
        json.dumps(profiles),
        encoding="utf-8",
    )
    return manifest_path


def test_frozen_2d_publishes_atomic_non_rankable_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _load_script("run_frozen_2d.py")
    manifest_path = _write_frozen_fixture(tmp_path)
    output_path = tmp_path / "frozen.json"
    monkeypatch.setattr(script, "ROOT", tmp_path)
    monkeypatch.setattr(
        script,
        "validate_manifest",
        lambda _manifest: {
            "pimsr-benchmarks": {
                "path": str(tmp_path),
                "audited_ancestor_commit": "deadbeef",
                "actual_head_commit": "f" * 40,
                "worktree_dirty": True,
            }
        },
    )
    monkeypatch.setattr(
        script,
        "bootstrap_ci",
        lambda values, **_kwargs: {"estimate": float(np.mean(values))},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_frozen_2d.py",
            "--manifest",
            str(manifest_path),
            "--out",
            str(output_path),
            "--allow-legacy",
        ],
    )

    script.main()

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["schema_version"] == 3
    assert result["comparison_status"] == "diagnostic_non_comparable"
    assert result["ranking_allowed"] is False
    assert result["headline_claim_allowed"] is False
    assert any("normalized" in reason for reason in result["diagnostic_reasons"])
    artifacts = result["provenance"]["input_artifacts"]
    assert set(artifacts) == {
        "manifest",
        "unified_result",
        "regional_profiles",
        "runner_source",
        "runner2d_source",
        "statistics_source",
    }
    for artifact in artifacts.values():
        payload = Path(artifact["path"]).read_bytes()
        assert artifact["sha256"] == hashlib.sha256(payload).hexdigest()
        assert artifact["size_bytes"] == len(payload)
    assert result["provenance"]["render_configuration"] == {
        "bootstrap_confidence": 0.95,
        "bootstrap_resamples": 10_000,
        "bootstrap_seed": 20260713,
    }

    original = output_path.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        script.main()
    assert output_path.read_bytes() == original


def test_frozen_2d_detects_post_read_input_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _load_script("run_frozen_2d.py")
    manifest_path = _write_frozen_fixture(tmp_path)
    output_path = tmp_path / "frozen.json"
    profiles_path = tmp_path / "results/v4/v4_profiles_bal.json"
    original_render = script.render

    def mutate_after_render(*args, **kwargs):
        result = original_render(*args, **kwargs)
        profiles_path.write_text("{}", encoding="utf-8")
        return result

    monkeypatch.setattr(script, "ROOT", tmp_path)
    monkeypatch.setattr(script, "validate_manifest", lambda _manifest: {})
    monkeypatch.setattr(script, "render", mutate_after_render)
    monkeypatch.setattr(
        script,
        "bootstrap_ci",
        lambda values, **_kwargs: {"estimate": float(np.mean(values))},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_frozen_2d.py",
            "--manifest",
            str(manifest_path),
            "--out",
            str(output_path),
            "--allow-legacy",
        ],
    )

    with pytest.raises(RuntimeError, match="changed after it was loaded"):
        script.main()
    assert not output_path.exists()


def test_uncertainty2d_is_atomic_single_method_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _load_script("run_uncertainty2d.py")
    checkpoint_path = tmp_path / "model.pt"
    dataset_path = tmp_path / "heldout.h5"
    output_dir = tmp_path / "uncertainty"
    checkpoint_path.write_bytes(b"checkpoint")
    with h5py.File(dataset_path, "w") as h5:
        h5.attrs["generator_seed"] = 17
        h5.create_dataset("sample_index", data=np.array([100, 101], dtype=np.int64))
        h5.create_dataset("scenario", data=np.array([0, 1], dtype=np.int64))
        h5.create_dataset("target_log10_res", data=np.zeros((2, 2, 2)))
        for name, value in (
            ("obs_mt_log10_rho", 2.0),
            ("obs_mt_phase", 45.0),
            ("obs_mt_log10_rho_tm", 2.2),
            ("obs_mt_phase_tm", 55.0),
        ):
            h5.create_dataset(name, data=np.full((2, 2, 3), value))

    class FakeModel:
        def __call__(self, observation):
            count = observation.shape[0]
            return {
                "log_rho": torch.zeros((count, 2, 2)),
                "log_sigma_rho": torch.zeros((count, 2, 2)),
            }

    class FakeLoaded:
        def __init__(self):
            self.checkpoint = {
                "stats_mean": np.zeros((1, 4, 1, 1), dtype=np.float32),
                "stats_std": np.ones((1, 4, 1, 1), dtype=np.float32),
            }
            self.model = FakeModel()
            self.contract = SimpleNamespace(depth_grid=np.array([100.0, 1000.0]))
            self.dataset_path = dataset_path
            self.unchanged_checks = 0
            self._artifacts = {
                "checkpoint": script.file_artifact_provenance(checkpoint_path),
                "dataset": script.file_artifact_provenance(dataset_path),
            }

        def artifact_provenance(self):
            return self._artifacts

        def require_artifacts_unchanged(self):
            self.unchanged_checks += 1
            for role, artifact in self._artifacts.items():
                script.require_file_artifact_unchanged(artifact, role=role)

    loaded = FakeLoaded()
    load_calls = 0

    def fake_load(*_args):
        nonlocal load_calls
        load_calls += 1
        return loaded

    monkeypatch.setattr(script, "load_model2d", fake_load)
    monkeypatch.setattr(
        script,
        "bootstrap_ci",
        lambda values, **_kwargs: {
            "estimate": float(np.mean(values)),
            "n": len(values),
        },
    )
    monkeypatch.setattr(
        script,
        "calibration_summary",
        lambda *_args, **_kwargs: {
            "coverage68": 1.0,
            "coverage68_by_depth": [1.0, 1.0],
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_uncertainty2d.py",
            "--checkpoint",
            str(checkpoint_path),
            "--test-h5",
            str(dataset_path),
            "--out-dir",
            str(output_dir),
            "--n",
            "1",
            "--adaptation",
            "zero-shot",
        ],
    )

    script.main()

    json_path = output_dir / "uncertainty2d.json"
    csv_path = output_dir / "coverage_by_depth.csv"
    result = json.loads(json_path.read_text(encoding="utf-8"))
    assert result["schema_version"] == 3
    assert result["comparison_status"] == "diagnostic_non_comparable"
    assert result["ranking_allowed"] is False
    assert result["headline_claim_allowed"] is False
    assert result["evaluation_contract"]["sample_selection"] == "ordered_prefix"
    assert result["evaluation_contract"]["sample_index_ranges_inclusive"] == [
        [100, 100]
    ]
    expected_indices = np.asarray([100], dtype="<i8").tobytes()
    assert result["evaluation_contract"]["sample_indices_sha256"] == hashlib.sha256(
        expected_indices
    ).hexdigest()
    assert result["evaluation_contract"]["generator_seed"] == 17
    assert result["artifacts"] == loaded.artifact_provenance()
    coverage_artifact = result["coverage_by_depth_artifact"]
    assert coverage_artifact["sha256"] == hashlib.sha256(csv_path.read_bytes()).hexdigest()
    assert loaded.unchanged_checks >= 3

    original_json = json_path.read_bytes()
    original_csv = csv_path.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        script.main()
    assert load_calls == 1
    assert json_path.read_bytes() == original_json
    assert csv_path.read_bytes() == original_csv


@pytest.mark.parametrize("filename", ["REPORT.md", "COMPARISON.md"])
def test_archived_documents_apply_global_notice(filename: str) -> None:
    text = (ROOT / "results" / filename).read_text(encoding="utf-8")
    first_table = text.find("|")
    notice = text if first_table == -1 else text[:first_table]
    assert "every" in notice.lower()
    assert "1D synthetic MT+gravity-vs-MT-only" in notice
    assert "comparison_status: diagnostic_non_comparable" in notice
    assert "ranking_allowed: false" in notice
    assert "state-of-the-art" in notice
