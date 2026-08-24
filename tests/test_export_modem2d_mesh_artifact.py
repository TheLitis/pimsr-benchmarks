from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from pimsr_benchmarks.modem2d_forward import MESH_CONFIGS


def _load_cli():
    path = Path(__file__).parents[1] / "scripts" / "export_modem2d_mesh_artifact.py"
    spec = importlib.util.spec_from_file_location("export_modem2d_mesh_artifact", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mesh_artifact_bytes_are_the_config_digest(tmp_path: Path):
    cli = _load_cli()
    mesh = MESH_CONFIGS["nested-production-v1"]
    payload = cli.canonical_mesh_payload(mesh)
    assert hashlib.sha256(payload).hexdigest() == mesh.sha256
    assert json.loads(payload) == mesh.canonical_record()

    output = tmp_path / "production-mesh.json"
    assert cli.main(["--mesh", mesh.mesh_id, "--output", str(output)]) == 0
    assert output.read_bytes() == payload
    with pytest.raises(FileExistsError):
        cli.main(["--mesh", mesh.mesh_id, "--output", str(output)])


def test_mesh_artifact_requires_json_suffix(tmp_path: Path):
    cli = _load_cli()
    with pytest.raises(ValueError, match="json suffix"):
        cli.main(
            [
                "--mesh",
                "nested-reference-x2-v1",
                "--output",
                str(tmp_path / "mesh.txt"),
            ]
        )
