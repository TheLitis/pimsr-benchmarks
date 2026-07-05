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
import glob
from pathlib import Path

import numpy as np
import torch

from pimsr_benchmarks.emtf import parse_emtf_xml, resample_station


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emtf-dir", required=True)
    ap.add_argument("--checkpoint", required=True, help="to read the period grid")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    periods = np.asarray(ckpt["periods"], dtype=np.float64)

    lrs, phs, masks, ids = [], [], [], []
    for f in sorted(glob.glob(str(Path(args.emtf_dir) / "*.xml"))):
        st = parse_emtf_xml(f)
        lr, ph, mask = resample_station(st, periods)
        lrs.append(lr)
        phs.append(ph)
        masks.append(mask)
        ids.append(st.station_id)

    np.savez(
        args.out,
        log_rho_a=np.asarray(lrs, dtype=np.float32),
        phase=np.asarray(phs, dtype=np.float32),
        mask=np.asarray(masks, dtype=np.float32),
        periods=periods,
        stations=np.asarray(ids),
    )
    print(f"exported {len(ids)} stations x {periods.size} periods -> {args.out}")


if __name__ == "__main__":
    main()
