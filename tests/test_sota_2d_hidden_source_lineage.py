from __future__ import annotations

import json
from pathlib import Path

from pimsr_benchmarks.hidden_campaign2d import _strict_source_lineage

ROOT = Path(__file__).resolve().parents[1]
LINEAGE_PATH = ROOT / "config" / "sota_2d_hidden_source_lineage.json"
REGISTRY_PATH = ROOT / "config" / "sota_methods.json"


def test_hidden_source_lineage_is_canonical_and_strict():
    raw = LINEAGE_PATH.read_bytes()
    value = json.loads(raw.decode("utf-8", errors="strict"))
    canonical = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    assert raw == canonical
    assert _strict_source_lineage(value) == value


def test_hidden_forward_commit_matches_the_public_registry_commitment():
    lineage = json.loads(LINEAGE_PATH.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    dataset = next(
        item for item in registry["datasets"] if item["id"] == "pimsr_generated_2d_v1"
    )
    assert lineage["pimsr_forward"]["repository_commit"] == dataset["generator"][
        "source_commit"
    ]
