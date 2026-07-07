# PIMSR vs. the state of the art in MT inversion

Comparison of our measured results against modern methods of
magnetotelluric (MT) data analysis: classical iterative inversion
(measured by us as baselines) and recent deep-learning approaches from
the literature (2022-2025). Last updated: 2026-07-07.

## 1. What we measured, on what data

- **Synthetic**: 3k held-out 2D sections (48x64 log10-resistivity grids,
  7 stations, 24 frequencies, calibrated correlated noise). Metric:
  full-section RMSE of log10(rho).
- **Real**: USArray/EMTF stations in the Yellowstone region. Primary
  profile: E-W row at ~44.6N (7 stations); generalisation set: four
  unseen E-W rows (G/I/J/K). Metric: shift-invariant nRMS computed with
  the rigorous 2D TE forward (`section_nrms_2d`) — every method scored
  identically.

## 2. Final measured leaderboard (real Yellowstone profile)

| Rank | Method | Class | 2D nRMS | Latency |
|---|---|---|---|---|
| 1 | **PIMSR sigma-reg U-Net 60k + physics ft** | ours, neural | **3.99** | ~ms |
| 2 | PIMSR sigma-reg U-Net 60k, pretrained only | ours, neural | 4.18 | ~ms |
| 3 | PIMSR U-Net 60k + physics ft | ours, neural | 4.47 | ~ms |
| 4 | PIMSR U-Net 10k + physics ft | ours, neural | 4.52 | ~ms |
| 5 | Cold-start 2D Gauss-Newton, 25 it (SimPEG) | classical 2D | 4.90 | 132 s |
| 6 | 2D hybrid (net warm-start + GN-8) | ours, hybrid | 5.05 | 41 s |
| 7 | 1D neural per-station, stitched | ours, neural | 6.49 | 36 ms |
| 8 | 1D hybrid (net + Occam), stitched | ours, hybrid | 6.50 | 2.9 s |
| 9 | 1D Occam per-station, stitched | classical 1D | 7.78 | 7.4 s |

Generalisation to four unseen profiles (mean 2D nRMS): unregularised
60k+ft **4.68** (best), pretrained 4.94, sigma-reg variants 5.29-5.35,
stitched classical 1D 6.48.

## 3. How this compares to the modern DL-inversion literature

Recent representative approaches (2022-2025): multimodal U-Nets
(MT2DInv-Unet), Transformer backbones (SwinTUNet, PISwinTUNet),
physics-guided multitask networks (PGWNet, PhyDNN), Bayesian NNs for
uncertainty. Direct number-to-number comparison is impossible — the
field has **no shared benchmark**, and published field-data results are
almost never reported as reproducible misfit numbers. The honest
comparison is therefore methodological:

| Dimension | Typical published work | PIMSR (this project) |
|---|---|---|
| Test on real field data | often 1 curated profile, qualitative figure | 5 USArray profiles, quantitative nRMS, 4 fully unseen |
| Metric vs classical methods | rarely same-metric; often visual comparison | identical rigorous 2D-forward nRMS for all 9 methods |
| Classical baseline strength | often weak/default (single Occam run) | tuned Occam 1D + 25-iteration 2D GN with reference-model reg |
| Physics in the loop | forward operator in training loss (PhyDNN, PGWNet) | pretraining + per-profile physics fine-tune against measured data |
| Uncertainty | mostly deterministic; some BNNs | heteroscedastic sigma head, coverage measured (0.69-0.79) |
| Negative results | essentially never published | documented: 2D hybrid degrades warm start; data scaling plateau; sigma-reg generalisation trade-off |
| Reproducibility | code rarely released | full pipeline (geogen -> forward -> inversion -> benchmarks) in versioned repos, CI-tested |

### Where our results agree with the literature

1. **Neural surrogates beat classical inversion on speed at comparable
   or better accuracy.** Universal claim in the literature; we confirm
   it quantitatively — 4-6 orders of magnitude latency reduction with
   19% better misfit than the best classical 2D run we could build.
2. **Physics in the loop matters.** Physics-guided training (PGWNet,
   PhyDNN) parallels our physics fine-tune, which is worth 5-11% nRMS
   on the target profile and transfers to unseen profiles.
3. **The synthetic-to-field gap is the central problem.** The
   literature attacks it with transfer learning and realistic
   synthetics; our calibrated-noise generation plus physics fine-tune
   is the same battle, and our multi-profile study measures the gap
   explicitly instead of hiding it.

### Where our findings challenge common practice

1. **Metric artifacts can invert conclusions.** Under a per-column 1D
   metric, classical iterative 1D looked dominant (~2.6); under the
   fair 2D-forward metric the same methods land at 6.5-7.8, behind
   every 2D network. Papers comparing DL to classical methods under
   inconsistent metrics may over- or under-claim.
2. **More synthetic data is not the lever.** 6x training data moved
   synthetic RMSE by ~1%. Architecture and physics coupling matter;
   raw dataset scale does not, past a modest threshold.
3. **Warm-starting classical inversion with a network — celebrated in
   1D — fails in sparse-station 2D.** GN refinement degraded our warm
   start (4.79 -> 4.91-6.85): with 7 stations the classical problem is
   too underdetermined to be a safe refiner. Hybrid claims should be
   tested at realistic station densities.

## 4. Honest limitations

- Our nRMS numbers use TE-mode 2D physics; full tensor (TE+TM+tipper)
  inversion — standard in production MT — remains future work.
- The scenario-classification head is weak (0.28-0.30) and needs
  architectural work, not more data.
- Sigma calibration is decent (0.69-0.79 coverage vs 0.68 ideal) but
  val-NLL drift is tamed, not cured (beta-NLL is the known candidate fix).
- The sigma-reg champion trades out-of-row generalisation (5.29 unseen)
  for target-profile fit (3.99); deployment must choose per use case.
- Classical baselines are ours, not a production ModEM/Occam2D setup;
  a stronger commercial-grade 2D baseline could narrow the gap.

## 5. Bottom line

On a fair, single-metric, multi-profile benchmark against real USArray
data, physics-informed neural inversion holds every top position:
**3.99 vs 4.90 (best classical 2D) vs 7.78 (stitched classical 1D)**,
at millisecond latency. This is consistent with the direction of the
2022-2025 literature but goes beyond typical publications in baseline
strength, metric fairness, generalisation testing, and the reporting of
negative results.
