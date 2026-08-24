import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


def _load_runner3d():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_3d_bench.py"
    spec = importlib.util.spec_from_file_location("run_3d_bench_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_result_publication_is_atomic_and_no_overwrite(tmp_path):
    runner = _load_runner3d()
    path = tmp_path / "result.json"

    runner._publish_json({"status": "complete"}, path)

    assert path.read_text(encoding="utf-8") == '{\n  "status": "complete"\n}\n'
    assert not (tmp_path / "result.json.part").exists()
    with pytest.raises(FileExistsError, match="overwrite"):
        runner._publish_json({"status": "replaced"}, path)
    assert '"complete"' in path.read_text(encoding="utf-8")


def test_single_method_3d_result_cannot_claim_complete_comparison():
    runner = _load_runner3d()

    assert runner.RESULT_STATUS == "diagnostic_incomplete"
    assert "no external baseline" in runner.RESULT_SCOPE


def test_result_publication_refuses_stale_part(tmp_path):
    runner = _load_runner3d()
    path = tmp_path / "result.json"
    part = tmp_path / "result.json.part"
    part.write_text("active", encoding="utf-8")

    with pytest.raises(FileExistsError, match="temporary"):
        runner._publish_json({}, path)

    assert part.read_text(encoding="utf-8") == "active"


def test_checkpoint_snapshot_hashes_the_loaded_bytes(tmp_path):
    runner = _load_runner3d()
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"epoch": 7, "model_state": {"weight": torch.arange(3)}}, checkpoint)

    state, digest, size = runner._load_checkpoint_snapshot(
        checkpoint,
        map_location="cpu",
    )

    assert state["epoch"] == 7
    assert torch.equal(state["model_state"]["weight"], torch.arange(3))
    assert digest == runner.hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert size == checkpoint.stat().st_size


def test_volume_weighted_metrics_use_physical_cell_sizes():
    runner = _load_runner3d()
    coordinates = {
        "depth": [1.0, 2.0, 5.0],
        "y": [0.0, 1.0],
        "x": [0.0, 2.0],
    }
    weights = runner._volume_weights(coordinates)
    assert weights.shape == (3, 2, 2)
    assert np.allclose(weights[:, 0, 0], [2.0, 4.0, 6.0])

    truth = np.zeros_like(weights)
    mean = np.zeros_like(weights)
    mean[-1] = 2.0
    sigma = np.ones_like(weights)
    metrics = runner._weighted_sample_metrics(mean, sigma, truth, weights)

    expected_fraction = float(weights[-1].sum() / weights.sum())
    assert metrics["log10_resistivity_rmse"] == pytest.approx(
        np.sqrt(4.0 * expected_fraction)
    )
    assert metrics["log10_resistivity_mae"] == pytest.approx(2.0 * expected_fraction)
    assert metrics["coverage68"] == pytest.approx(1.0 - expected_fraction)


def test_weighted_metrics_reject_invalid_uncertainty():
    runner = _load_runner3d()
    shape = (2, 2, 2)
    with pytest.raises(ValueError, match="positive sigma"):
        runner._weighted_sample_metrics(
            np.zeros(shape),
            np.zeros(shape),
            np.zeros(shape),
            np.ones(shape),
        )
