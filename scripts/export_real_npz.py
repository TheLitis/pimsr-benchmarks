"""Export real EMTF stations resampled onto the training period grid.

Produces an .npz consumed by ``pimsr-inversion``'s self-supervised fine-tune:
  log_rho_a : (n_st, n_periods) log10 apparent resistivity (det average)
  phase     : (n_st, n_periods) degrees
  mask      : (n_st, n_periods) 1 where the station band covers the period
  periods   : (n_periods,) training period grid, s
  stations  : (n_st,) station ids
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import torch
from pimsr_inversion.contracts1d import validate_checkpoint1d

from pimsr_benchmarks.emtf import parse_emtf_xml, resample_station_determinant
from pimsr_benchmarks.runner2d import (
    file_artifact_provenance,
    publish_json_no_overwrite,
    publish_npz_no_overwrite,
    require_file_artifact_unchanged,
)


def _snapshot_xml_sources(emtf_dir: str | Path) -> list[dict[str, object]]:
    directory = Path(emtf_dir).resolve(strict=True)
    paths = sorted(directory.glob("*.xml"), key=lambda path: path.name.casefold())
    if not paths:
        raise FileNotFoundError(f"no EMTF XML inputs found in {directory}")
    return [file_artifact_provenance(path) for path in paths]


def _require_xml_sources_unchanged(
    emtf_dir: str | Path,
    sources: list[dict[str, object]],
) -> None:
    directory = Path(emtf_dir).resolve(strict=True)
    current = sorted(directory.glob("*.xml"), key=lambda path: path.name.casefold())
    if [str(path.resolve()) for path in current] != [
        str(source["path"]) for source in sources
    ]:
        raise RuntimeError("EMTF XML input set changed during export")
    for source in sources:
        require_file_artifact_unchanged(source, role="EMTF XML input")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emtf-dir", required=True)
    ap.add_argument("--checkpoint", required=True, help="to read the period grid")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve(strict=True)
    checkpoint_artifact = file_artifact_provenance(checkpoint_path)
    checkpoint_payload = checkpoint_path.read_bytes()
    require_file_artifact_unchanged(checkpoint_artifact, role="1D checkpoint")
    ckpt = torch.load(
        io.BytesIO(checkpoint_payload),
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(ckpt, dict):
        raise TypeError("1D checkpoint root must be a dictionary")
    periods = validate_checkpoint1d(ckpt).periods

    sources = _snapshot_xml_sources(args.emtf_dir)
    lrs, phs, masks, ids = [], [], [], []
    seen_ids: set[str] = set()
    for source in sources:
        st = parse_emtf_xml(str(source["path"]))
        if st.station_id in seen_ids:
            raise ValueError(f"duplicate EMTF station ID {st.station_id!r}")
        seen_ids.add(st.station_id)
        lr, ph, mask = resample_station_determinant(st, periods)
        if not np.isfinite(lr[mask]).all() or not np.isfinite(ph[mask]).all():
            raise ValueError(f"station {st.station_id!r} contains invalid in-band MT values")
        if np.any((ph[mask] < 0.0) | (ph[mask] >= 180.0)):
            raise ValueError(
                f"station {st.station_id!r} phase does not follow [0, 180) degrees"
            )
        lrs.append(lr)
        phs.append(ph)
        masks.append(mask)
        ids.append(st.station_id)

    output = Path(args.out).resolve()
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    collision = next((path for path in (output, manifest_path) if path.exists()), None)
    if collision is not None:
        raise FileExistsError(f"refusing to overwrite existing export: {collision}")
    require_file_artifact_unchanged(checkpoint_artifact, role="1D checkpoint")
    _require_xml_sources_unchanged(args.emtf_dir, sources)
    exported = publish_npz_no_overwrite(
        output,
        log_rho_a=np.asarray(lrs, dtype=np.float32),
        phase=np.asarray(phs, dtype=np.float32),
        mask=np.asarray(masks, dtype=bool),
        periods=periods,
        stations=np.asarray(ids),
    )
    export_artifact = file_artifact_provenance(exported)
    require_file_artifact_unchanged(checkpoint_artifact, role="1D checkpoint")
    _require_xml_sources_unchanged(args.emtf_dir, sources)
    manifest = {
        "schema": "pimsr-real-emtf-input-export",
        "schema_version": 1,
        "artifact_role": "fine_tuning_input_not_benchmark_result",
        "source_artifacts": {
            "checkpoint": checkpoint_artifact,
            "emtf_xml": sources,
        },
        "output_artifact": export_artifact,
        "array_contract": {
            "stations": len(ids),
            "periods": int(periods.size),
            "log_rho_a": [len(ids), int(periods.size)],
            "phase": [len(ids), int(periods.size)],
            "mask": [len(ids), int(periods.size)],
            "phase_convention": "degrees_modulo_180_[0,180)",
        },
    }
    publish_json_no_overwrite(manifest, manifest_path)
    print(f"exported {len(ids)} stations x {periods.size} periods -> {args.out}")


if __name__ == "__main__":
    main()
