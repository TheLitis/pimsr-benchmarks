# pimsr-benchmarks

Benchmarks of the PIMSR physics-informed neural inversion
([pimsr-inversion](https://github.com/TheLitis/pimsr-inversion)) against
classical and **production** inversion codes, on synthetic data and on
**real USArray magnetotelluric stations**.

Part of the PIMSR platform:
[pimsr-geogen](https://github.com/TheLitis/pimsr-geogen) ·
[pimsr-forward](https://github.com/TheLitis/pimsr-forward) ·
[pimsr-inversion](https://github.com/TheLitis/pimsr-inversion) ·
pimsr-benchmarks

## Headline results

All numbers below are the shift-invariant 2D-forward data misfit
(`section_nrms_2d`, lower is better) on real USArray/EMTF stations across
five E-W profiles in the Yellowstone region. Full history, methodology and
honest negative results: [results/REPORT.md](results/REPORT.md).

| Profile | ModEM NLCG | Occam2DMT v3.0 | PIMSR U-Net (joint-ft) |
|---|---|---|---|
| G | 5.32 | 3.92 | **3.59** |
| H-YS | 5.90 | 4.68 | **4.10** |
| I | 10.98 | 9.26 | **5.62** |
| J | 6.28 | 6.40 | **3.49** |
| K | 6.99 | 6.03 | **4.69** |
| **mean** | 7.09 | 6.06 | **4.30** |

The neural model wins on every profile against both production codes, at
~4 orders of magnitude less compute (ms vs 5-210 s per profile). Both
production codes were compiled from official sources and driven by scripts
in this repo (`scripts/run_occam2dmt.py`, `scripts/run_modem2d.py`) for a
fully reproducible comparison.

## Baselines

| Method | Implementation | Notes |
|---|---|---|
| Occam2DMT v3.0 | official Scripps Fortran source | production standard since de Groot-Hedlin & Constable (1990) |
| ModEM 2D NLCG | official open source (github.com/magnetotellurics/ModEM) | the most-cited modern MT code |
| SimPEG 2D Gauss-Newton | `pimsr_benchmarks.hybrid2d` | our in-repo classical baseline |
| Occam-style 1D | `pimsr_benchmarks.occam1d` | Tikhonov GN, per-station |
| PIMSR neural (1D + 2D) | checkpoints from pimsr-inversion | single forward pass, amortised |

## Metrics

- `section_nrms_2d` — shift-invariant data misfit through a 2D forward (the
  fair cross-method metric; introduced after the stitched-1D metric was
  shown to be an artifact — see REPORT.md)
- RMSE of log10-resistivity vs ground truth (synthetic)
- Uncertainty calibration (1-sigma coverage), scenario classification accuracy
- Wall-clock time per profile

## Real data

27 USArray/EMTF transfer-function stations (Yellowstone box, 42.5-45.5N,
108.5-113W) from IRIS/EarthScope SPUD, committed under `data/emtf/`.
`pimsr_benchmarks.emtf` parses the XML into per-mode (TE=Zyx, TM=Zxy)
apparent resistivity / phase curves.

## Usage

```bash
pip install -e .
# synthetic benchmark against the held-out test split
python scripts/run_2d_bench.py --checkpoint best2d.pt --test-h5 ds2d_test.h5 \
    --emtf-dir data/emtf --out-dir results/my_run --n 500
# unified real-data leaderboard
python scripts/run_unified_leaderboard.py --help
# production-code comparisons
python scripts/run_occam2dmt.py --help
python scripts/run_modem2d.py --help
```

## License

MIT (code). Real EMTF data courtesy of IRIS/EarthScope, US National Science
Foundation.
