"""Validate and render the frozen 2D PIMSR benchmark from its manifest.

This entry point intentionally consumes committed machine-readable results.
Expensive recomputation remains in the method-specific scripts; publishing a
leaderboard with a different metric or missing provenance fails closed here.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path

import numpy as np

import pimsr_benchmarks.runner2d as runner2d_module
import pimsr_benchmarks.statistics as statistics_module
from pimsr_benchmarks.runner2d import (
    file_artifact_provenance,
    publish_json_no_overwrite,
    require_file_artifact_unchanged,
)
from pimsr_benchmarks.statistics import bootstrap_ci

ROOT = Path(__file__).resolve().parents[1]
LEGACY_METRIC_ID = "section_nrms_2d_legacy_zxy_unmasked_v1"


def _git_sha(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def validate_manifest(
    manifest: dict,
    repo_root: Path = ROOT,
) -> dict[str, dict[str, object]]:
    metric = manifest["metric"]
    if manifest.get("status") != "legacy_non_comparable":
        raise ValueError("the archived manifest must be explicitly marked legacy")
    if metric["id"] != LEGACY_METRIC_ID or metric.get("ground_truth_claim") is not False:
        raise ValueError("unexpected archived metric contract")
    if metric.get("comparable_to_current_metric") is not False:
        raise ValueError("archived results must not claim current-metric comparability")
    if set(manifest["profiles"]) != {"G", "H-YS", "I", "J", "K"}:
        raise ValueError("frozen profile set changed")
    repository_validation: dict[str, dict[str, object]] = {}
    for name, expected in manifest["repositories"].items():
        repo = repo_root if name == "pimsr-benchmarks" else repo_root.parent / name
        actual = _git_sha(repo)
        # Manifest records the audited base commit; reproducibility permits descendants.
        ok = (
            subprocess.run(
                ["git", "-C", str(repo), "merge-base", "--is-ancestor", expected, actual],
                check=False,
            ).returncode
            == 0
        )
        if not ok:
            raise ValueError(
                f"{name}: audited commit {expected} is not an ancestor of {actual}"
            )
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(repo), "status", "--porcelain"],
                text=True,
            ).strip()
        )
        repository_validation[name] = {
            "path": str(repo.resolve()),
            "audited_ancestor_commit": expected,
            "actual_head_commit": actual,
            "worktree_dirty": dirty,
        }
    return repository_validation


def render(
    manifest: dict,
    *,
    unified: dict | None = None,
    profiles: dict | None = None,
) -> dict:
    if unified is None:
        unified = json.loads(
            (ROOT / "results/unified/unified.json").read_text(encoding="utf-8")
        )
    if profiles is None:
        profiles = json.loads(
            (ROOT / "results/v4/v4_profiles_bal.json").read_text(encoding="utf-8")
        )["profiles"]
    values_by_method = {
        method: [float(profiles[p][method]) for p in manifest["profiles"]]
        for method in next(iter(profiles.values()))
    }
    return {
        "schema_version": 3,
        "status": "legacy_non_comparable",
        "result_kind": "frozen_legacy_2d_diagnostic",
        "comparison_status": "diagnostic_non_comparable",
        "ranking_allowed": False,
        "headline_claim_allowed": False,
        "diagnostic_reasons": [
            "archived Zxy-only metric is incompatible with the current TE/TM metric",
            "legacy period extrapolation and component weighting differ from the current contract",
            "normalized field geometry is not publishable physical survey geometry",
            "adapted checkpoints include observations from evaluation profiles",
        ],
        "metric": manifest["metric"],
        "provenance": {
            "repositories": manifest["repositories"],
            "datasets": manifest["datasets"],
            "checkpoints": manifest["checkpoints"],
        },
        "yellowstone_unified": unified,
        "regional_profiles": profiles,
        "regional_bootstrap_95": {
            method: bootstrap_ci(values, n_resamples=10_000, seed=20260713)
            for method, values in values_by_method.items()
        },
        "limitations": manifest["limitations"],
    }


def _read_json_artifact(
    provenance: dict[str, object],
    *,
    role: str,
) -> object:
    path = Path(str(provenance["path"]))
    value = json.loads(path.read_text(encoding="utf-8"))
    require_file_artifact_unchanged(provenance, role=role)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(ROOT / "config/frozen_2d.json"))
    parser.add_argument("--out", default=str(ROOT / "results/frozen_2d.json"))
    parser.add_argument(
        "--allow-legacy",
        action="store_true",
        help="acknowledge that the archived scores are not comparable to v2",
    )
    args = parser.parse_args()
    if not args.allow_legacy:
        parser.error("refusing to render legacy results without --allow-legacy")
    output_path = Path(args.out).resolve()
    if output_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing benchmark output: {output_path}"
        )

    input_artifacts = {
        "manifest": file_artifact_provenance(args.manifest),
        "unified_result": file_artifact_provenance(
            ROOT / "results/unified/unified.json"
        ),
        "regional_profiles": file_artifact_provenance(
            ROOT / "results/v4/v4_profiles_bal.json"
        ),
        "runner_source": file_artifact_provenance(__file__),
        "runner2d_source": file_artifact_provenance(runner2d_module.__file__),
        "statistics_source": file_artifact_provenance(statistics_module.__file__),
    }
    manifest = _read_json_artifact(input_artifacts["manifest"], role="frozen manifest")
    unified = _read_json_artifact(
        input_artifacts["unified_result"], role="frozen unified result"
    )
    profiles_document = _read_json_artifact(
        input_artifacts["regional_profiles"], role="frozen regional profiles"
    )
    if not isinstance(manifest, dict):
        raise TypeError("frozen manifest root must be an object")
    if not isinstance(unified, dict):
        raise TypeError("frozen unified result root must be an object")
    if not isinstance(profiles_document, dict) or not isinstance(
        profiles_document.get("profiles"), dict
    ):
        raise TypeError("frozen regional profile result lacks a profiles object")

    repository_validation = validate_manifest(manifest)
    output = render(
        manifest,
        unified=unified,
        profiles=profiles_document["profiles"],
    )
    output["provenance"].update(
        {
            "input_artifacts": input_artifacts,
            "repository_validation": repository_validation,
            "render_configuration": {
                "bootstrap_confidence": 0.95,
                "bootstrap_resamples": 10_000,
                "bootstrap_seed": 20260713,
            },
            "runtime_environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
            },
        }
    )
    for role, provenance in input_artifacts.items():
        require_file_artifact_unchanged(provenance, role=role)
    publish_json_no_overwrite(output, output_path)
    print(f"validated frozen diagnostic -> {output_path}")


if __name__ == "__main__":
    main()
