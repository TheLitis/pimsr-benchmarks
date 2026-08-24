"""Fail-closed contracts for the legacy 2D profile diagnostic runner."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import scripts.run_2d_bench as runner


def _profile_modes() -> dict[str, object]:
    shape = (2, 2)
    return {
        "lr_tm": np.full(shape, 2.0),
        "ph_tm": np.full(shape, 45.0),
        "mask_tm": np.ones(shape, dtype=bool),
        "x_model": np.array([-1.0, 1.0]),
        "x_km": np.array([-10.0, 10.0]),
        "geometry_policy": "affine_profile_to_model_station_grid",
        "profile_azimuth_deg": 90.0,
        "profile_span_m": 20_000.0,
        "model_station_span_m": 2.0,
        "horizontal_compression_factor": 10_000.0,
        "publishable_physical_geometry": False,
        "station_ids": runner.PROFILE_IDS,
    }


def test_real_profile_is_explicitly_non_rankable(monkeypatch):
    modes = _profile_modes()
    monkeypatch.setattr(runner, "assemble_profile_modes", lambda *args, **kwargs: modes)
    monkeypatch.setattr(
        runner,
        "prepare_profile_observation",
        lambda profile, checkpoint: np.zeros((1, 4, 2, 2), dtype=np.float32),
    )
    monkeypatch.setattr(runner, "section_nrms", lambda *args, **kwargs: (2.5, [2.0, 3.0]))

    class Model:
        def __call__(self, observation):
            assert observation.shape == (1, 4, 2, 2)
            return {"log_rho": torch.full((1, 2, 2), 2.0)}

    result = runner.bench_real_profile(
        Model(),
        {},
        "unused",
        np.array([0.1, 1.0]),
        np.array([-1.0, 1.0]),
        np.array([10.0, 100.0]),
    )

    assert result["comparison_status"] == "diagnostic_non_comparable"
    assert result["ranking_allowed"] is False
    assert result["metric_id"] == "section_nrms_1d_tm_masked_v2"
    assert result["inverse_observation_budget"] == ["TE/Zyx", "TM/Zxy"]
    assert result["scoring_observation_budget"] == ["TM/Zxy"]
    assert result["geometry"]["publishable_physical_geometry"] is False


def test_emtf_snapshot_rejects_set_or_content_mutation(tmp_path):
    first = tmp_path / "first.xml"
    first.write_text("<EM_TF>first</EM_TF>", encoding="utf-8")
    sources = runner._snapshot_emtf_sources(tmp_path)
    runner._require_emtf_sources_unchanged(tmp_path, sources)

    second = tmp_path / "second.xml"
    second.write_text("<EM_TF>second</EM_TF>", encoding="utf-8")
    with pytest.raises(RuntimeError, match="set changed"):
        runner._require_emtf_sources_unchanged(tmp_path, sources)

    second.unlink()
    first.write_text("<EM_TF>changed</EM_TF>", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed after"):
        runner._require_emtf_sources_unchanged(tmp_path, sources)
