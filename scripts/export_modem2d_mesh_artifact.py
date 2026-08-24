#!/usr/bin/env python
"""Publish a canonical ModEM nested-mesh record with its config digest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pimsr_benchmarks.hidden_campaign2d import _write_exclusive
from pimsr_benchmarks.modem2d_forward import MESH_CONFIGS, NestedMeshConfig

NESTED_MESH_CONFIGS = {
    name: mesh
    for name, mesh in MESH_CONFIGS.items()
    if isinstance(mesh, NestedMeshConfig)
}


def canonical_mesh_payload(mesh: NestedMeshConfig) -> bytes:
    payload = json.dumps(
        mesh.canonical_record(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    if hashlib.sha256(payload).hexdigest() != mesh.sha256:
        raise RuntimeError("canonical mesh payload does not match the config digest")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish one no-overwrite canonical ModEM nested-mesh artifact"
    )
    parser.add_argument("--mesh", choices=sorted(NESTED_MESH_CONFIGS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.output.suffix.lower() != ".json":
        raise ValueError("mesh artifact output must use a .json suffix")
    mesh = NESTED_MESH_CONFIGS[args.mesh]
    identity = _write_exclusive(args.output, canonical_mesh_payload(mesh))
    print(
        json.dumps(
            {
                "mesh_id": mesh.mesh_id,
                "path": str(identity.path),
                "sha256": identity.sha256,
                "size_bytes": identity.size_bytes,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
