"""Materialize separate method-input and withheld-truth PIMSR 2D payloads."""

from __future__ import annotations

import argparse
from pathlib import Path

from pimsr_benchmarks.dataset2d_materialization import (
    DEFAULT_PHASE_DEGREE_FLOOR,
    DEFAULT_RHO_LOG10_FLOOR,
    materialize_dataset2d,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed schema-v2 PIMSR 2D materialization into deterministic "
            "observation-only and withheld-truth NPZ payloads"
        )
    )
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--observations-npz", type=Path, required=True)
    parser.add_argument("--truth-npz", type=Path, required=True)
    parser.add_argument("--public-manifest-json", type=Path, required=True)
    parser.add_argument("--operator-manifest-json", type=Path, required=True)
    parser.add_argument(
        "--split-id",
        required=True,
        help="stable lowercase benchmark split identifier (for example, test)",
    )
    parser.add_argument(
        "--sample-id-key-file",
        type=Path,
        required=True,
        help=(
            "operator secret containing at least 32 bytes; used only to HMAC-map "
            "private generator indices to opaque sample IDs"
        ),
    )
    parser.add_argument(
        "--rho-log10-floor",
        type=float,
        default=DEFAULT_RHO_LOG10_FLOOR,
        help=(
            "declared evaluation floor for log10 apparent resistivity "
            f"(default: {DEFAULT_RHO_LOG10_FLOOR})"
        ),
    )
    parser.add_argument(
        "--phase-degree-floor",
        type=float,
        default=DEFAULT_PHASE_DEGREE_FLOOR,
        help=(
            "declared evaluation floor for phase in degrees "
            f"(default: {DEFAULT_PHASE_DEGREE_FLOOR})"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = materialize_dataset2d(
        args.source_h5,
        args.observations_npz,
        args.truth_npz,
        args.public_manifest_json,
        args.operator_manifest_json,
        split_id=args.split_id,
        sample_id_key=args.sample_id_key_file,
        rho_log10_floor=args.rho_log10_floor,
        phase_degree_floor=args.phase_degree_floor,
    )
    print(f"observations: {result.observations_path} sha256={result.observations_sha256}")
    print(f"withheld truth: {result.truth_path} sha256={result.truth_sha256}")
    print(
        "public manifest: "
        f"{result.public_manifest_path} sha256={result.public_manifest_sha256}"
    )
    print(
        "operator manifest: "
        f"{result.operator_manifest_path} sha256={result.operator_manifest_sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
