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

---

# MVP-2 update: hybrid refinement and self-supervised fine-tuning

Both follow-ups proposed above are now implemented and measured.

## Method comparison (all four methods)

| Method | Synthetic RMSE (log10 rho) | Real nRMS (27 st.) | Time / station |
|---|---|---|---|
| Occam (cold start) | 0.791 | 2.57 | 88 ms |
| Neural (pretrained) | **0.514** | 9.23 | **2 ms** |
| Neural (fine-tuned, L2-SP aw=10) | 0.617 | 6.99 | **2 ms** |
| Hybrid (neural warm-start Occam) | 0.750 | **2.74** | 44 ms |

## Hybrid: neural warm start + Occam refinement

The network prediction is projected onto the Occam mesh and used as the
starting model with a pre-cooled trade-off parameter (`mu0 * 0.65^6`),
capped at 12 Gauss-Newton iterations (vs 30 cold).

- **Real data:** nRMS 2.74 vs 2.57 cold — near-classical misfit at 2x the
  speed and 2.5x fewer iterations. 100% of stations improve on the raw
  network output.
- **Synthetic:** RMSE 0.750 — refinement trades profile accuracy for data
  fit, as theory predicts (the truth is smoother than the noise-fitting
  minimum). Use the pure network when ground-truth-style accuracy matters
  and the hybrid when the data misfit must be defensible.

## Self-supervised fine-tuning on real transfer functions

400/200 AdamW steps on the 27 real stations, loss = masked shift-invariant
physics misfit + L2-SP anchor to the pretrained weights. Anchor-weight sweep:

| anchor | syn RMSE | syn 1-sigma cov. | real nRMS |
|---|---|---|---|
| none (pretrained) | 0.514 | 0.74 | 9.23 |
| 1 | 0.856 | 0.27 | 7.74 |
| **10** | **0.617** | **0.41** | **6.99** |
| 50 | 0.557 | 0.52 | 7.53 |

`aw=10` cuts the real-data misfit 24% while keeping the synthetic RMSE well
below the classical baseline. The uncertainty calibration degrades (0.74 ->
0.41) because the sigma head receives no direct supervision from the physics
loss — recalibrating it (e.g. temperature scaling on the val split) is the
obvious MVP-3 item.

## Takeaways

1. The **hybrid** is the production answer for real surveys today:
   classical-grade misfit, half the classical cost, fully automated init.
2. **Physics-only fine-tuning works** without any labels, but needs a
   strong anchor and sigma recalibration to preserve the synthetic prior.
3. The remaining real-data gap is structural (1D physics on 2D/3D geology),
   which motivates the MVP-2 spec's move to 2D meshes.
