from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "public-lineage"

EXPECTED = {
    "train": {
        "path": EVIDENCE / "train-lineage-v2.json",
        "lineage_sha256": (
            "3f671bf72f2cf40dcdf4f50d9ebf16e24b4cbbdb4b1fde4526c353ff24e65735"
        ),
        "dataset_sha256": (
            "b9f1fce44012abe522e1b238ab67ccf9a3c7d9f81890c9911c9e25279b590051"
        ),
        "dataset_size_bytes": 153_745_920,
        "generator_seed": 20_260_820,
        "sample_count": 10_000,
    },
    "validation": {
        "path": EVIDENCE / "validation-lineage-v2.json",
        "lineage_sha256": (
            "0d4f46884e4664475a490471ae7ee1eb64cd4569dc42c73409e918adadf67725"
        ),
        "dataset_sha256": (
            "19ee9df2c4e0d57494e424948f0088064590c3a5e517b0369b1e18c6a26c905d"
        ),
        "dataset_size_bytes": 15_388_920,
        "generator_seed": 20_260_821,
        "sample_count": 1_000,
    },
}


def test_public_lineage_evidence_is_canonical_complete_and_pinned() -> None:
    for split, expected in EXPECTED.items():
        payload = expected["path"].read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected["lineage_sha256"]
        value = json.loads(payload.decode("utf-8"))
        assert payload == (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
        assert value["schema"] == "pimsr-public-dataset-lineage-2d"
        assert value["schema_version"] == 2
        assert value["split"] == split
        assert value["evidence_scope"] == (
            "artifact_lineage_and_transitive_generator_source_identity_without_"
            "forward_regeneration"
        )
        merged = value["inputs"]["merged_dataset"]
        assert merged["sha256"] == expected["dataset_sha256"]
        assert merged["size_bytes"] == expected["dataset_size_bytes"]
        verification = value["verification"]
        assert verification["forward_regeneration_performed"] is False
        assert verification["generation_complete"] is True
        assert verification["generator_seed"] == expected["generator_seed"]
        assert verification["sample_count"] == expected["sample_count"]
        assert verification["sample_end_index"] == expected["sample_count"] - 1
        assert value["repositories"]["pimsr_forward"]["clean_worktree"] is True
        assert value["repositories"]["pimsr_geogen"]["clean_worktree"] is True
        assert b"operator" not in payload.lower()
        assert b"hidden" not in payload.lower()
