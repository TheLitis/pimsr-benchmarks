"""Assemble the final PIMSR vs classical benchmark report.

Inputs (JSON files produced by the bench runs):
  occam_synthetic.json  - Occam baseline on the synthetic test split
  neural_synthetic.json - neural inverter on the same split
  occam_real.json       - Occam on USArray EMTF stations
  neural_real.json      - neural inverter on the same stations

Output: REPORT.md (markdown summary committed to the repo).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True)
    p.add_argument("--out", default="REPORT.md")
    args = p.parse_args()

    d = Path(args.results_dir)
    load = lambda name: json.loads((d / name).read_text())  # noqa: E731

    occ_s = load("occam_synthetic.json")
    neu_s = load("neural_synthetic.json")
    occ_r = load("occam_real.json")
    neu_r = load("neural_real.json")

    lines = [
        "# PIMSR benchmark report",
        "",
        "Physics-informed neural inversion (`pimsr-inversion`) vs a classical",
        "Occam-style regularised Gauss-Newton 1D MT inversion (`occam1d`),",
        "evaluated on the held-out synthetic test split and on real USArray",
        "EMTF stations (Yellowstone / Snake River Plain).",
        "",
        "## Synthetic test split",
        "",
        f"Samples: {occ_s['n']} (Occam) / {neu_s['n']} (neural)",
        "",
        "| metric | Occam 1D | PIMSR neural |",
        "|---|---|---|",
        f"| RMSE log10(rho) mean | {fmt(occ_s['rmse_log10_res']['mean'])} | {fmt(neu_s['rmse_log10_res']['mean'])} |",
        f"| RMSE log10(rho) median | {fmt(occ_s['rmse_log10_res']['median'])} | {fmt(neu_s['rmse_log10_res']['median'])} |",
        f"| RMSE log10(rho) p90 | {fmt(occ_s['rmse_log10_res']['p90'])} | {fmt(neu_s['rmse_log10_res']['p90'])} |",
        f"| time per station (s) | {fmt(occ_s['time_per_station_s'], 4)} | {fmt(neu_s['time_per_station_s'], 4)} |",
    ]
    if "scenario_accuracy" in neu_s:
        lines.append(
            f"| scenario accuracy | n/a | {fmt(neu_s['scenario_accuracy'])} |"
        )
    if "sigma_coverage_1" in neu_s:
        lines.append(
            f"| 1-sigma coverage (ideal 0.683) | n/a | {fmt(neu_s['sigma_coverage_1'])} |"
        )

    lines += [
        "",
        "### Per-scenario RMSE (log10 rho)",
        "",
        "| scenario | Occam 1D | PIMSR neural |",
        "|---|---|---|",
    ]
    for k in sorted(occ_s.get("per_scenario_rmse", {})):
        neu_v = neu_s.get("per_scenario_rmse", {}).get(k)
        lines.append(
            f"| {k} | {fmt(occ_s['per_scenario_rmse'][k])} |"
            f" {fmt(neu_v) if neu_v is not None else 'n/a'} |"
        )

    lines += [
        "",
        "## Real USArray EMTF stations",
        "",
        f"Stations: {occ_r['n_stations']} (box 42.5-45.5N, 108.5-113W)",
        "",
        "| metric | Occam 1D | PIMSR neural |",
        "|---|---|---|",
        f"| data misfit nRMS mean | {fmt(occ_r['nrms']['mean'])} | {fmt(neu_r['nrms']['mean'])} |",
        f"| data misfit nRMS median | {fmt(occ_r['nrms']['median'])} | {fmt(neu_r['nrms']['median'])} |",
        "",
        "Real-data ground truth is unknown; the comparison metric is the",
        "normalised misfit between each method's predicted-profile forward",
        "response and the measured station response.",
        "",
    ]
    Path(args.out).write_text("\n".join(lines))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
