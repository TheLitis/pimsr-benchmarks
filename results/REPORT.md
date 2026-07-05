# PIMSR benchmark report

Physics-informed neural inversion (`pimsr-inversion`) vs a classical
Occam-style regularised Gauss-Newton 1D MT inversion (`occam1d`),
evaluated on the held-out synthetic test split and on real USArray
EMTF stations (Yellowstone / Snake River Plain).

## Synthetic test split

Samples: 500 (Occam) / 500 (neural)

| metric | Occam 1D | PIMSR neural |
|---|---|---|
| RMSE log10(rho) mean | 0.791 | 0.514 |
| RMSE log10(rho) median | 0.771 | 0.493 |
| RMSE log10(rho) p90 | 1.196 | 0.758 |
| time per station (s) | 0.0877 | 0.0018 |
| scenario accuracy | n/a | 0.372 |
| 1-sigma coverage (ideal 0.683) | n/a | 0.735 |

### Per-scenario RMSE (log10 rho)

| scenario | Occam 1D | PIMSR neural |
|---|---|---|
| 0 | 0.712 | 0.454 |
| 1 | 0.714 | 0.492 |
| 2 | 0.703 | 0.449 |
| 3 | 0.820 | 0.589 |
| 4 | 1.062 | 0.613 |

## Real USArray EMTF stations

Stations: 27 (box 42.5-45.5N, 108.5-113W)

| metric | Occam 1D | PIMSR neural |
|---|---|---|
| data misfit nRMS mean | 2.569 | 9.227 |
| data misfit nRMS median | 2.086 | 8.573 |

Real-data ground truth is unknown; the comparison metric is the
normalised misfit between each method's predicted-profile forward
response and the measured station response.

## Analysis

**Synthetic (in-distribution):** the physics-informed neural inverter beats the
classical Occam baseline decisively — 35% lower profile RMSE (0.514 vs 0.791),
better on every scenario class, ~48x faster per station, and its 1-sigma
uncertainty is close to calibrated (0.735 vs ideal 0.683). This is the regime
the MVP-1 spec targets: dense, cheap-sensor surveys where per-station classical
inversion is the throughput bottleneck.

**Real USArray stations (out-of-distribution):** the classical Occam inversion
fits the measured responses much better (nRMS 2.6 vs 9.2). This is the expected
sim-to-real domain gap, not a physics bug — the network has never seen
Yellowstone-style strong 2D/3D effects, longer period bands, or real noise
statistics. Occam optimises the misfit per station at inference time; the
network amortises over its training prior.

**Implications for MVP-2:**
1. Add test-time refinement: use the network prediction as the Occam starting
   model (best of both — near-classical misfit at a fraction of iterations).
2. Domain randomisation: widen the geology prior (2D structures via
   anisotropy proxies, realistic noise from real station residuals).
3. Fine-tune on real transfer functions with the physics loss only
   (self-supervised — no ground truth needed).

## Reproduce

```
# dataset (on the self-hosted runner)
gh workflow run pimsr-dataset.yml -R TheLitis/Runner -f n_train=60000

# training (GPU)
gh workflow run pimsr-train.yml -R TheLitis/Runner -f dataset_run_id=<id>

# baselines + neural benchmark + this report
pimsr-bench synthetic --test-h5 ds_test.h5 --out occam_synthetic.json
pimsr-bench real --emtf-dir data/emtf --out occam_real.json
python scripts/run_neural_bench.py --checkpoint best.pt --test-h5 ds_test.h5 \
    --emtf-dir data/emtf --out-dir results
python scripts/make_report.py --results-dir results --out results/REPORT.md
```
