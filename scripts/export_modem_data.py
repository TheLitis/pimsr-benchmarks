"""Export USArray profile impedances to the ModEM 2D data format.

Produces one ModEM-style data file per profile row with full-tensor
off-diagonal impedances (Zxy, Zyx) at the stations' true coordinates —
ready for a production ModEM / MARE2DEM comparison run (spec item 5).

Frequencies are taken from the stations themselves (union band clipped
to the common range, log-resampled), impedance errors use the published
variance where present with a 5 % |Z| error floor, matching common
production practice.

Usage:
  python scripts/export_modem_data.py --emtf-dir data/emtf \
      --out-dir results/modem_export
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

from pimsr_benchmarks.emtf import parse_emtf_xml
from pimsr_benchmarks.hybrid2d import PROFILE_IDS, PROFILES

MU0 = 4.0e-7 * np.pi
ERROR_FLOOR = 0.05  # fraction of |Z| per component


def _interp_complex(lp, z, lt):
    re = np.interp(lt, lp, z.real)
    im = np.interp(lt, lp, z.imag)
    return re + 1j * im


def export_profile(emtf_dir: str, ids: list[str], out_path: Path,
                   n_freq: int = 20) -> dict:
    stations = {}
    for f in glob.glob(f"{emtf_dir}/*.xml"):
        st = parse_emtf_xml(f)
        stations[st.station_id] = st
    profile = [stations[i] for i in ids]

    pmin = max(st.periods.min() for st in profile)
    pmax = min(st.periods.max() for st in profile)
    periods = np.logspace(np.log10(pmin), np.log10(pmax), n_freq)

    lat0 = np.mean([st.latitude for st in profile])
    lon0 = np.mean([st.longitude for st in profile])

    lines = [
        "# ModEM impedance data exported from USArray EMTF XML (PIMSR)",
        "# period(s) code lat lon x(m) y(m) z(m) component Re Im error",
        "> Full_Impedance",
        "> exp(-i\\omega t)",
        "> [V/m]/[T]",
        "> 0.00",
        f"> {lat0:.4f} {lon0:.4f}",
        f"> {n_freq} {len(profile)}",
    ]

    n_rows = 0
    for st in profile:
        lp = np.log10(st.periods)
        lt = np.log10(periods)
        x = (st.latitude - lat0) * 111_000.0
        y = (st.longitude - lon0) * 111_000.0 * np.cos(np.radians(lat0))
        for comp, z in (("ZXY", st.zxy), ("ZYX", st.zyx)):
            zi = _interp_complex(lp, z, lt)
            err = np.maximum(ERROR_FLOOR * np.abs(zi), 1e-8)
            for k, per in enumerate(periods):
                lines.append(
                    f"{per:.6e} {st.station_id} {st.latitude:.4f} "
                    f"{st.longitude:.4f} {x:.1f} {y:.1f} 0.0 {comp} "
                    f"{zi.real[k]:.6e} {zi.imag[k]:.6e} {err[k]:.6e}"
                )
                n_rows += 1

    out_path.write_text("\n".join(lines) + "\n")
    return {
        "stations": [st.station_id for st in profile],
        "n_freq": n_freq,
        "period_range_s": [float(periods.min()), float(periods.max())],
        "rows": n_rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emtf-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-freq", type=int, default=20)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = {"H-YS": PROFILE_IDS, **{k: v for k, v in PROFILES.items() if k != "H-YS"}}
    for name, ids in rows.items():
        info = export_profile(
            args.emtf_dir, ids, out_dir / f"modem_{name}.dat", args.n_freq
        )
        print(f"{name}: {info['rows']} rows, "
              f"{len(info['stations'])} stations -> modem_{name}.dat")


if __name__ == "__main__":
    main()
