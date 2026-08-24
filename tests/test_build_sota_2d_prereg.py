from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pimsr_benchmarks import comparison2d

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_sota_2d_prereg.py"
_SPEC = importlib.util.spec_from_file_location("build_sota_2d_prereg", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
builder = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(builder)


def test_builder_freezes_exact_analytic_and_pimsr_training_contracts() -> None:
    analytic = builder._analytic_contract()
    normalized = comparison2d._analytic_1d_contract(analytic)
    assert normalized["frequencies_hz"].shape == (8,)
    assert normalized["canonical_depth_centres_m"].shape == (64,)
    assert tuple(normalized["profiles"]) == (
        "analytic-halfspace-100",
        "analytic-layered-100-10-500",
    )

    training = builder._pimsr_training()
    counts = np.asarray(training["class_counts"], dtype=np.float64)
    expected = counts.sum() / (5.0 * counts)
    assert training["class_weight_formula"] == "count_sum/(5*max(class_count,1))"
    assert np.array_equal(np.asarray(training["class_weights"]), expected)
    assert training["epochs"] == 80


def test_builder_rejects_nonpassing_convergence_before_other_inputs(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = tmp_path / "convergence-report.json"
    report.write_text(
        json.dumps(
            {
                "schema": "pimsr-modem2d-convergence-validation",
                "schema_version": 1,
                "passed": False,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(builder, "_repository_identity", lambda *_a, **_k: "a" * 40)
    with pytest.raises(RuntimeError, match="has not passed"):
        builder.build(SimpleNamespace(convergence_report=report))
