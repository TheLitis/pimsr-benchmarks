from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest

from pimsr_benchmarks.evaluation2d import (
    EVALUATION_SCHEMA,
    PREDICTION_SCHEMA,
    SCHEMA_VERSION,
    TRUTH_SCHEMA,
    Evaluation2DPublicationError,
    Evaluation2DValidationError,
    canonical_json_bytes,
    cell_edges_from_centers,
    cell_widths_from_centers,
    evaluate_predictions_2d,
    load_predictions_2d,
    load_truth_2d,
    publish_evaluation_2d,
)

_OBSERVATIONS_SHA256 = "1" * 64


def _truth_arrays() -> dict[str, np.ndarray]:
    return {
        "schema": np.asarray(TRUTH_SCHEMA),
        "schema_version": np.asarray(SCHEMA_VERSION, dtype="<i8"),
        "sample_index": np.asarray([20, 10], dtype="<i8"),
        "observations_sha256": np.asarray(_OBSERVATIONS_SHA256, dtype="<U64"),
        "scenario": np.asarray(["salt", "background"]),
        "has_fault": np.asarray([True, False], dtype=np.bool_),
        "x_cell_centers_m": np.asarray([0.0, 1.0, 4.0], dtype="<f8"),
        "depth_cell_centers_m": np.asarray([1.0, 3.0], dtype="<f8"),
        "truth_log10_resistivity": np.zeros((2, 2, 3), dtype="<f4"),
    }


def _prediction_arrays() -> dict[str, np.ndarray]:
    values = np.zeros((2, 2, 3), dtype="<f4")
    # Prediction rows are intentionally in the opposite order from truth rows.
    # Sample 10 has error only in the widest x cell; sample 20 has uniform error.
    values[0, :, -1] = 1.0
    values[1, :, :] = 2.0
    return {
        "schema": np.asarray(PREDICTION_SCHEMA),
        "schema_version": np.asarray(SCHEMA_VERSION, dtype="<i8"),
        "observations_sha256": np.asarray(_OBSERVATIONS_SHA256, dtype="<U64"),
        "sample_index": np.asarray([10, 20], dtype="<i8"),
        "x_cell_centers_m": np.asarray([0.0, 1.0, 4.0], dtype="<f8"),
        "depth_cell_centers_m": np.asarray([1.0, 3.0], dtype="<f8"),
        "predicted_log10_resistivity": values,
    }


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> Path:
    np.savez(path, **arrays)
    return path


def _artifacts(tmp_path: Path) -> tuple[Path, Path]:
    return (
        _write_npz(tmp_path / "truth.npz", _truth_arrays()),
        _write_npz(tmp_path / "prediction.npz", _prediction_arrays()),
    )


def test_area_weighting_pairing_bootstrap_and_scenario_strata(tmp_path):
    truth_path, prediction_path = _artifacts(tmp_path)

    first = evaluate_predictions_2d(
        truth_path,
        prediction_path,
        confidence=0.8,
        n_resamples=100,
        seed=17,
    )
    second = evaluate_predictions_2d(
        truth_path,
        prediction_path,
        confidence=0.8,
        n_resamples=100,
        seed=17,
    )

    assert first == second
    assert first["audience"] == "benchmark_operator_only_until_predictions_locked"
    assert first["schema"] == EVALUATION_SCHEMA
    assert [row["sample_index"] for row in first["per_sample"]] == [10, 20]
    sample_10, sample_20 = first["per_sample"]
    assert sample_10["scenario"] == "background"
    assert sample_10["has_fault"] is False
    assert sample_10["mae_log10_resistivity"] == pytest.approx(0.5)
    assert sample_10["rmse_log10_resistivity"] == pytest.approx(np.sqrt(0.5))
    assert sample_10["rmse_log10_resistivity"] != pytest.approx(np.sqrt(1.0 / 3.0))
    assert sample_20["mae_log10_resistivity"] == pytest.approx(2.0)
    assert sample_20["rmse_log10_resistivity"] == pytest.approx(2.0)

    expected_mean = (np.sqrt(0.5) + 2.0) / 2.0
    assert first["overall"]["rmse_log10_resistivity"]["mean"][
        "estimate"
    ] == pytest.approx(expected_mean)
    assert first["overall"]["rmse_log10_resistivity"]["median"][
        "estimate"
    ] == pytest.approx(expected_mean)
    assert first["by_scenario"]["background"]["n_samples"] == 1
    assert first["by_scenario"]["salt"]["n_samples"] == 1
    assert first["by_scenario"]["background"]["mae_log10_resistivity"][
        "mean"
    ]["estimate"] == pytest.approx(0.5)
    assert first["bootstrap_contract"] == {
        "algorithm": "numpy_random_PCG64_percentile_linear",
        "confidence": 0.8,
        "cross_method_effect_ci": False,
        "headline_eligible": False,
        "hierarchical": False,
        "n_resamples": 100,
        "resampled_fields": [
            "rmse_log10_resistivity",
            "mae_log10_resistivity",
        ],
        "pairing": "identical_resampled_sample_rows_for_all_metrics",
        "seed": 17,
        "scope": "single_method_single_campaign_sample_level_descriptive",
    }
    assert first["release_gate"]["public_release_allowed"] is False
    assert first["implementation"]["source_sha256"] == hashlib.sha256(
        Path("src/pimsr_benchmarks/evaluation2d.py").read_bytes()
    ).hexdigest()
    assert first["implementation"]["numpy_version"] == np.__version__
    assert first["physics_misfit"]["included"] is False
    assert first["inputs"]["truth"]["sha256"] == hashlib.sha256(
        truth_path.read_bytes()
    ).hexdigest()
    assert first["inputs"]["prediction"]["sha256"] == hashlib.sha256(
        prediction_path.read_bytes()
    ).hexdigest()
    assert first["inputs"]["observations"]["sha256"] == _OBSERVATIONS_SHA256
    assert len(first["per_depth"]) == 2
    assert first["per_depth"][0]["depth_cell_lower_edge_m"] == pytest.approx(0.0)
    assert first["per_depth"][0]["depth_cell_upper_edge_m"] == pytest.approx(2.0)
    assert first["metric_contract"]["scoring_domain"] == {
        "depth_cell_edges_m": [0.0, 2.0, 4.0],
        "mask": "all_truth_grid_cells",
        "support": "full_grid_voronoi_cells_from_centers",
        "x_cell_edges_m": [-0.5, 0.5, 2.5, 5.5],
    }


def test_cell_widths_use_midpoints_and_extrapolated_boundaries():
    widths = cell_widths_from_centers(np.asarray([0.0, 1.0, 4.0], dtype="<f8"))
    np.testing.assert_allclose(widths, [1.0, 2.0, 3.0])
    edges = cell_edges_from_centers(np.asarray([0.0, 1.0, 4.0], dtype="<f8"))
    np.testing.assert_allclose(edges, [-0.5, 0.5, 2.5, 5.5])


def test_report_publication_is_canonical_atomic_and_no_overwrite(tmp_path):
    truth_path, prediction_path = _artifacts(tmp_path)
    report = evaluate_predictions_2d(
        truth_path, prediction_path, n_resamples=10, seed=2
    )
    destination = tmp_path / "results" / "evaluation.json"

    published = publish_evaluation_2d(report, destination)

    assert published == destination
    assert not destination.with_name("evaluation.json.part").exists()
    parsed = json.loads(destination.read_text(encoding="utf-8"))
    assert destination.read_bytes() == canonical_json_bytes(parsed)
    with pytest.raises(Evaluation2DPublicationError, match="refusing to overwrite"):
        publish_evaluation_2d(report, destination)

    stale_target = tmp_path / "stale.json"
    stale_partial = stale_target.with_name("stale.json.part")
    stale_partial.write_bytes(b"unfinished")
    with pytest.raises(Evaluation2DPublicationError, match="stale partial"):
        publish_evaluation_2d(report, stale_target)
    assert stale_partial.read_bytes() == b"unfinished"
    assert not stale_target.exists()


def test_report_publication_rolls_back_on_base_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    truth_path, prediction_path = _artifacts(tmp_path)
    report = evaluate_predictions_2d(
        truth_path, prediction_path, n_resamples=10, seed=2
    )
    destination = tmp_path / "evaluation.json"
    partial = destination.with_name("evaluation.json.part")
    real_read_bytes = Path.read_bytes
    calls = 0

    def interrupted_read(path: Path) -> bytes:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return real_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", interrupted_read)

    with pytest.raises(KeyboardInterrupt):
        publish_evaluation_2d(report, destination)

    assert not destination.exists()
    assert not partial.exists()


def test_report_publication_rolls_back_when_link_succeeds_then_interrupts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    truth_path, prediction_path = _artifacts(tmp_path)
    report = evaluate_predictions_2d(
        truth_path, prediction_path, n_resamples=10, seed=2
    )
    destination = tmp_path / "evaluation.json"
    partial = destination.with_name("evaluation.json.part")
    real_link = os.link

    def interrupted_link(source: Path, target: Path) -> None:
        real_link(source, target)
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "link", interrupted_link)

    with pytest.raises(KeyboardInterrupt):
        publish_evaluation_2d(report, destination)

    assert not destination.exists()
    assert not partial.exists()


def test_expected_hash_detects_artifact_tampering(tmp_path):
    truth_path, prediction_path = _artifacts(tmp_path)
    pinned_truth = hashlib.sha256(truth_path.read_bytes()).hexdigest()
    pinned_prediction = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
    evaluate_predictions_2d(
        truth_path,
        prediction_path,
        expected_truth_sha256=pinned_truth,
        expected_prediction_sha256=pinned_prediction,
        expected_observations_sha256=_OBSERVATIONS_SHA256,
        n_resamples=5,
    )

    changed = _prediction_arrays()
    changed["predicted_log10_resistivity"][0, 0, 0] = 0.25
    _write_npz(prediction_path, changed)

    with pytest.raises(Evaluation2DValidationError, match="pinned digest"):
        evaluate_predictions_2d(
            truth_path,
            prediction_path,
            expected_truth_sha256=pinned_truth,
            expected_prediction_sha256=pinned_prediction,
            n_resamples=5,
        )
    with pytest.raises(Evaluation2DValidationError, match="64 lowercase"):
        load_truth_2d(truth_path, expected_sha256="A" * 64)
    with pytest.raises(Evaluation2DValidationError, match="pinned digest"):
        evaluate_predictions_2d(
            truth_path,
            prediction_path,
            expected_observations_sha256="2" * 64,
            n_resamples=5,
        )


@pytest.mark.parametrize(
    ("sample_ids", "error"),
    [
        ([10, 10], "duplicate"),
        ([10, 30], "exactly match"),
    ],
)
def test_prediction_rejects_duplicate_missing_or_extra_ids(
    tmp_path, sample_ids, error
):
    truth_path = _write_npz(tmp_path / "truth.npz", _truth_arrays())
    prediction = _prediction_arrays()
    prediction["sample_index"] = np.asarray(sample_ids, dtype="<i8")
    prediction_path = _write_npz(tmp_path / "prediction.npz", prediction)

    with pytest.raises(Evaluation2DValidationError, match=error):
        evaluate_predictions_2d(
            truth_path, prediction_path, n_resamples=5, seed=0
        )


def test_truth_rejects_duplicate_ids(tmp_path):
    truth = _truth_arrays()
    truth["sample_index"] = np.asarray([20, 20], dtype="<i8")
    path = _write_npz(tmp_path / "truth.npz", truth)

    with pytest.raises(Evaluation2DValidationError, match="duplicate"):
        load_truth_2d(path)


@pytest.mark.parametrize(
    ("field", "replacement", "error"),
    [
        ("schema", np.asarray("wrong-schema"), "schema must"),
        ("schema_version", np.asarray(1, dtype="<i4"), "schema_version"),
        ("sample_index", np.asarray([10, 20], dtype="<i4"), "dtype int64"),
        ("sample_index", np.asarray([10, 20], dtype=">i8"), "dtype int64"),
        ("observations_sha256", np.asarray("bad"), r"Unicode\[64\]"),
        (
            "x_cell_centers_m",
            np.asarray([0.0, 1.0, 4.0], dtype="<f4"),
            "dtype float64",
        ),
        (
            "predicted_log10_resistivity",
            np.zeros((2, 2, 3), dtype="<f8"),
            "dtype float32",
        ),
        (
            "predicted_log10_resistivity",
            np.zeros((2, 2, 3), dtype=">f4"),
            "dtype float32",
        ),
        (
            "predicted_log10_resistivity",
            np.zeros((2, 2), dtype="<f4"),
            "3-dimensional",
        ),
        (
            "predicted_log10_resistivity",
            np.full((2, 2, 3), np.nan, dtype="<f4"),
            "must be finite",
        ),
    ],
)
def test_prediction_schema_shape_dtype_and_finiteness_fail_closed(
    tmp_path, field, replacement, error
):
    prediction = _prediction_arrays()
    prediction[field] = replacement
    path = _write_npz(tmp_path / "prediction.npz", prediction)

    with pytest.raises(Evaluation2DValidationError, match=error):
        load_predictions_2d(path)


@pytest.mark.parametrize(
    ("field", "replacement", "error"),
    [
        ("schema_version", np.asarray(1, dtype="<i8"), "schema_version"),
        ("observations_sha256", np.asarray("bad"), r"Unicode\[64\]"),
        ("scenario", np.asarray(["salt"]), "length must match"),
        ("has_fault", np.asarray([1, 0], dtype="u1"), "dtype bool"),
        (
            "x_cell_centers_m",
            np.asarray([0.0, 0.0, 4.0], dtype="<f8"),
            "strictly increasing",
        ),
        (
            "x_cell_centers_m",
            np.asarray([0.0, 1.0, 4.0], dtype=">f8"),
            "dtype float64",
        ),
        (
            "depth_cell_centers_m",
            np.asarray([1.0, np.inf], dtype="<f8"),
            "must be finite",
        ),
        (
            "truth_log10_resistivity",
            np.zeros((2, 2, 3), dtype="<f8"),
            "dtype float32",
        ),
        (
            "truth_log10_resistivity",
            np.zeros((2, 2, 3), dtype=">f4"),
            "dtype float32",
        ),
        (
            "truth_log10_resistivity",
            np.zeros((2, 3, 2), dtype="<f4"),
            "shape must be",
        ),
        (
            "truth_log10_resistivity",
            np.full((2, 2, 3), np.inf, dtype="<f4"),
            "must be finite",
        ),
    ],
)
def test_truth_schema_shape_dtype_and_finiteness_fail_closed(
    tmp_path, field, replacement, error
):
    truth = _truth_arrays()
    truth[field] = replacement
    path = _write_npz(tmp_path / "truth.npz", truth)

    with pytest.raises(Evaluation2DValidationError, match=error):
        load_truth_2d(path)


def test_npz_exact_keys_and_duplicate_members_are_rejected(tmp_path):
    extra = _prediction_arrays()
    extra["truth_log10_resistivity"] = np.zeros((2, 2, 3), dtype="<f4")
    extra_path = _write_npz(tmp_path / "extra.npz", extra)
    with pytest.raises(Evaluation2DValidationError, match="members mismatch"):
        load_predictions_2d(extra_path)

    duplicate_path = _write_npz(tmp_path / "duplicate.npz", _prediction_arrays())
    with (
        pytest.warns(UserWarning, match="Duplicate name"),
        zipfile.ZipFile(duplicate_path, "a") as archive,
    ):
        archive.writestr("schema.npy", b"duplicate")
    with pytest.raises(Evaluation2DValidationError, match="duplicate archive"):
        load_predictions_2d(duplicate_path)

    compressed_path = tmp_path / "compressed.npz"
    np.savez_compressed(compressed_path, **_prediction_arrays())
    with pytest.raises(Evaluation2DValidationError, match="ZIP_STORED"):
        load_predictions_2d(compressed_path)

    reordered = _prediction_arrays()
    reordered = {"sample_index": reordered.pop("sample_index"), **reordered}
    reordered_path = _write_npz(tmp_path / "reordered.npz", reordered)
    with pytest.raises(Evaluation2DValidationError, match="canonical order"):
        load_predictions_2d(reordered_path)


def test_prediction_grid_shape_must_exactly_match_truth(tmp_path):
    truth_path = _write_npz(tmp_path / "truth.npz", _truth_arrays())
    prediction = _prediction_arrays()
    prediction["predicted_log10_resistivity"] = np.zeros((2, 2, 4), dtype="<f4")
    prediction_path = _write_npz(tmp_path / "prediction.npz", prediction)

    with pytest.raises(Evaluation2DValidationError, match="shape must be"):
        evaluate_predictions_2d(
            truth_path, prediction_path, n_resamples=5, seed=0
        )


@pytest.mark.parametrize(
    ("field", "replacement", "error"),
    [
        ("observations_sha256", np.asarray("2" * 64, dtype="<U64"), "same observations"),
        (
            "x_cell_centers_m",
            np.asarray([0.0, 2.0, 4.0], dtype="<f8"),
            "x grid must exactly match",
        ),
        (
            "depth_cell_centers_m",
            np.asarray([1.0, 4.0], dtype="<f8"),
            "depth grid must exactly match",
        ),
    ],
)
def test_prediction_must_bind_exact_campaign_and_grid(
    tmp_path: Path, field: str, replacement: np.ndarray, error: str
):
    truth_path = _write_npz(tmp_path / "truth.npz", _truth_arrays())
    prediction = _prediction_arrays()
    prediction[field] = replacement
    prediction_path = _write_npz(tmp_path / "prediction.npz", prediction)

    with pytest.raises(Evaluation2DValidationError, match=error):
        evaluate_predictions_2d(
            truth_path, prediction_path, n_resamples=5, seed=0
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"confidence": 1.0},
        {"confidence": np.nan},
        {"n_resamples": 0},
        {"n_resamples": True},
        {"seed": -1},
        {"seed": 1.5},
    ],
)
def test_bootstrap_options_fail_closed(tmp_path, kwargs):
    truth_path, prediction_path = _artifacts(tmp_path)

    with pytest.raises(Evaluation2DValidationError, match="bootstrap"):
        evaluate_predictions_2d(truth_path, prediction_path, **kwargs)


def test_canonical_json_rejects_non_finite_values():
    with pytest.raises(Evaluation2DPublicationError, match="canonical JSON"):
        canonical_json_bytes({"invalid": np.nan})
