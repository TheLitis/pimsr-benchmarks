#!/usr/bin/env python
"""Run one production-grade pinned ModEM 2-D forward solve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pimsr_benchmarks.modem2d_forward import (
    MESH_CONFIGS,
    load_canonical_hdf5,
    load_canonical_npz,
    require_snapshot_unchanged,
    run_modem_forward,
    snapshot_file,
    verify_pinned_runtime,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Map one canonical 64x48 truth onto a frozen independent ModEM mesh and "
            "publish exactly 8x12 TE/TM responses without overwriting outputs."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-h5", type=Path)
    source.add_argument("--input-npz", type=Path)
    parser.add_argument(
        "--row", type=int, help="zero-based HDF5 row; required with --input-h5"
    )
    parser.add_argument("--mesh", choices=sorted(MESH_CONFIGS), required=True)
    parser.add_argument("--modem-repo", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--timeout-seconds", type=float, default=1_800.0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    runner_snapshot = snapshot_file(__file__, role="ModEM forward runner source")
    if args.input_h5 is not None:
        if args.row is None:
            raise ValueError("--row is required with --input-h5")
        truth, source = load_canonical_hdf5(args.input_h5, row=args.row)
    else:
        if args.row is not None:
            raise ValueError("--row is only valid with --input-h5")
        truth, source = load_canonical_npz(args.input_npz)
    source = {**source, "runner_source": runner_snapshot.record()}
    runtime = verify_pinned_runtime(
        modem_repo=args.modem_repo,
        build_root=args.build_root,
        docker_executable=args.docker,
    )
    output, response, provenance = run_modem_forward(
        runtime=runtime,
        truth=truth,
        mesh=MESH_CONFIGS[args.mesh],
        output_dir=args.output_dir,
        source_provenance=source,
        timeout_seconds=args.timeout_seconds,
    )
    require_snapshot_unchanged(runner_snapshot, role="ModEM forward runner source")
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "sample_id": truth.sample_id,
                "mesh_id": args.mesh,
                "mesh_config_sha256": MESH_CONFIGS[args.mesh].sha256,
                "runtime_identity_sha256": runtime.identity_sha256,
                "rows": provenance["response_contract"]["rows"],
                "all_finite": bool(
                    response.log10_rho_te.size == 96 and response.log10_rho_tm.size == 96
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
