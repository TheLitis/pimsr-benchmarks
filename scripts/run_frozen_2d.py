"""Validate and render the frozen 2D PIMSR benchmark from its manifest.

This entry point intentionally consumes committed machine-readable results.
Expensive recomputation remains in the method-specific scripts; publishing a
leaderboard with a different metric or missing provenance fails closed here.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from pimsr_benchmarks.statistics import bootstrap_ci

ROOT = Path(__file__).resolve().parents[1]


def _git_sha(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def validate_manifest(manifest: dict, repo_root: Path = ROOT) -> None:
    metric = manifest["metric"]
    if metric["id"] != "section_nrms_2d" or metric.get("ground_truth_claim") is not False:
        raise ValueError("frozen real benchmark must use 2D forward consistency without ground-truth claim")
    if set(manifest["profiles"]) != {"G", "H-YS", "I", "J", "K"}:
        raise ValueError("frozen profile set changed")
    for name, expected in manifest["repositories"].items():
        repo = repo_root if name == "pimsr-benchmarks" else repo_root.parent / name
        actual = _git_sha(repo)
        # Manifest records the audited base commit; reproducibility permits descendants.
        ok = subprocess.run(
            ["git", "-C", str(repo), "merge-base", "--is-ancestor", expected, actual],
            check=False,
        ).returncode == 0
        if not ok:
            raise ValueError(f"{name}: audited commit {expected} is not an ancestor of {actual}")


def render(manifest: dict) -> dict:
    unified = json.loads((ROOT / "results/unified/unified.json").read_text())
    profiles = json.loads((ROOT / "results/v4/v4_profiles_bal.json").read_text())["profiles"]
    values_by_method = {
        method: [float(profiles[p][method]) for p in manifest["profiles"]]
        for method in next(iter(profiles.values()))
    }
    return {
        "schema_version": 1,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(ROOT / "config/frozen_2d.json"))
    parser.add_argument("--out", default=str(ROOT / "results/frozen_2d.json"))
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    validate_manifest(manifest)
    output = render(manifest)
    Path(args.out).write_text(json.dumps(output, indent=2) + "\n")
    print(f"validated frozen 2D benchmark -> {args.out}")


if __name__ == "__main__":
    main()
