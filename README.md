# pimsr-benchmarks

Benchmarks of the PIMSR physics-informed neural inversion
([pimsr-inversion](https://github.com/TheLitis/pimsr-inversion)) against
classical deterministic inversion on both synthetic and **real** magnetotelluric data.

Part of the PIMSR (Physics-Informed Multi-modal Subsurface Reconstruction) project:
[pimsr-geogen](https://github.com/TheLitis/pimsr-geogen) ·
[pimsr-forward](https://github.com/TheLitis/pimsr-forward) ·
[pimsr-inversion](https://github.com/TheLitis/pimsr-inversion) ·
pimsr-benchmarks

## Baselines

| Method | Implementation | Notes |
|---|---|---|
| Occam-style 1D MT inversion | `pimsr_benchmarks.occam1d` | Tikhonov-regularised Gauss-Newton with chi-squared target, the standard of the industry since Constable et al. (1987) |
| SimPEG 1D MT inversion | `scripts/simpeg_baseline.py` | Runs where SimPEG is installed (heavy dependency, optional) |
| PIMSR neural inversion | checkpoint from pimsr-inversion | single forward pass, amortised |

## Metrics

- RMSE of log10-resistivity profile vs ground truth (synthetic)
- Data misfit (nRMS) vs observed responses (real data)
- Interface-depth error for layered scenarios
- Wall-clock time per station
- Uncertainty calibration (PIT / coverage of predicted sigma)

## Real data

USArray MagNet/EMTF transfer functions distributed by IRIS/EarthScope SPUD
(`http://ds.iris.edu/spud/emtf`). `pimsr_benchmarks.emtf` parses the XML
transfer-function format into apparent resistivity / phase curves matching the
PIMSR observation vector (Berdichevsky-average of the off-diagonal impedances).

## Usage

```bash
pip install -e .
# synthetic benchmark against the held-out test split
pimsr-bench synthetic --dataset ds_test.h5 --checkpoint model_best.pt --out results/synthetic.json
# real-data case study
pimsr-bench real --xml data/USArray.*.xml --checkpoint model_best.pt --out results/real.json
```

## License

MIT
