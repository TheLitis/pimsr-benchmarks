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

---

# MVP-3 update: calibrated uncertainty + real-residual noise model

Two changes, then a full regeneration/retraining cycle on the self-hosted
runner (dataset run 28752222641, training run 28752390483):

1. **Temperature scaling** (`pimsr_inversion.calibrate`): closed-form
   per-head scale fitted on the val split, stored in the checkpoint and
   applied transparently at inference. Now part of the training workflow.
2. **Real-residual noise model** (`pimsr_forward.sensors`): correlated AR(1)
   distortion along the period axis calibrated on Occam residuals of the 27
   real stations (pooled std 0.085 log10rho / 5.9 deg phase, lag-1 0.46,
   per-station amplitude 0.02-0.26). Emulates the 2D/3D-effect misfit that
   dominates real errors.

## v2 results (dataset with real-calibrated noise)

| Method | Syn RMSE | 1-sigma cov. (ideal 0.68) | Real nRMS | Time/st |
|---|---|---|---|---|
| Occam (cold) | 0.951 | — | 2.57 | 391 ms |
| Neural v2 (calibrated) | **0.545** | **0.77** | 8.36 | **2 ms** |
| Neural v2 + physics fine-tune | 0.654 | 0.71 | 6.14 | **2 ms** |
| Hybrid v2 (warm-start Occam) | 0.798 | — | **2.59** | 164 ms |

## What changed vs v1

- **Harder benchmark, wider gap.** The realistic noise makes classical
  inversion much worse (RMSE 0.79 -> 0.95, 88 -> 391 ms/st as Gauss-Newton
  fights correlated distortion), while the network degrades only mildly
  (0.514 -> 0.545). The neural advantage grows from 35% to **43%**.
- **Uncertainty is now calibrated end-to-end.** Pretrained coverage 0.77 and,
  crucially, fine-tuned coverage 0.71 vs 0.41 before — temperature scaling
  fixed the main MVP-2 defect.
- **Sim-to-real gap narrows at the source.** Domain-randomised training alone
  cuts real nRMS 9.23 -> 8.36; with physics fine-tuning 6.14 (was 6.99).
- **Hybrid now matches the classical misfit** (2.59 vs 2.57) — warm-started
  refinement is strictly dominant on real data: same fit, 2.4x faster.

## Remaining gap and the case for 2D

Even with calibrated noise and fine-tuning, a 1D network cannot express
lateral structure: nRMS 6.14 vs 2.6 for per-station optimisation. The residual
gap is structural, confirming the MVP-2 spec decision to move to 2D meshes
(SimPEG 2D MT forward + conv-2D inversion) as the next milestone.

---

# 2D milestone: SimPEG forward + U-Net section inversion

Full pipeline executed on the self-hosted runner: 12,000 stochastic 2D
sections (SimPEG TE-mode, run 28753497334, ~3.5 h on 8 workers) ->
auto-triggered GPU training (run 28757073497, 80 epochs, best at ~ep. 20 by
val NLL) -> benchmark below.

## Components (all committed)

- `pimsr-geogen.section2d`: stochastic sections — undulating interfaces,
  normal faults, finite scenario lenses on a 64 x 48 (x, log-depth) grid.
- `pimsr-forward.mt2d`: TE-mode `Simulation2DElectricField`, validated
  against the analytic 1D solution (0.6 % median mismatch).
- `pimsr-inversion.network2d`: U-Net, pseudo-section (2, 24, 16) ->
  resistivity section (48, 64) + heteroscedastic sigma + scenario head.
- CI chain: `pimsr-dataset2d.yml` (parallel shards, Python 3.11) triggers
  `pimsr-train2d.yml` on success via `workflow_run`.

## Results (500 test sections + real Yellowstone E-W profile)

| Metric | 2D U-Net | Best 1D reference |
|---|---|---|
| Section RMSE (log10 rho, full 48x64 image) | 0.646 | 0.545 (per-column task, easier) |
| 1-sigma coverage (ideal 0.68) | 0.75 | 0.77 |
| Real-profile physics nRMS | 6.54 | 6.14 (1D fine-tuned) / 2.59 (hybrid) |
| Scenario accuracy | 0.26 | 0.71 (1D) |

## Honest read

- **The 2D pipeline works end-to-end** and predicts full sections with
  calibrated uncertainty (0.75 coverage) in a single forward pass.
- **It does not yet beat 1D on the real profile** (6.54 vs 6.14). Three
  identified causes: (a) 6x less training data than the 1D model (10k vs
  60k); (b) no real-data fine-tuning stage yet — the 1D number is after
  fine-tuning, the raw pretrained 1D was 8.36; (c) val NLL divergence after
  epoch 20 (sigma overfit) — earlier stopping or sigma warm-up would help.
- **Scenario head underperforms** (0.26): scenario lenses occupy a small
  fraction of each section; needs class-balanced loss or larger lenses.
- Next steps in order of expected value: real-profile fine-tuning for the 2D
  net (analogous to the 1D anchored fine-tune, which cut nRMS by 27 %),
  60k-section dataset, sigma warm-up schedule.

---

# 2D fine-tune: the 2D net now leads all pure-neural methods on real data

Applied the proven 1D recipe to the U-Net (`pimsr_inversion.finetune2d`,
committed): per-station-column physics chi^2 (masked to each station's
measured band, static-shift invariant) + L2-SP anchor + input jitter as
augmentation for the single-profile training signal. 200 steps, CPU-fast.

## Anchor sweep (real profile nRMS / synthetic RMSE / 1-sigma coverage)

| anchor_weight | Real nRMS | Syn RMSE | Coverage (ideal 0.68) |
|---|---|---|---|
| none (baseline) | 6.54 | 0.646 | 0.75 |
| 30 | 5.30 | 0.650 | 0.73 |
| 10 | 4.82 | 0.658 | 0.71 |
| **3 (chosen)** | **4.59** | 0.664 | **0.69** |

## Updated method leaderboard (real Yellowstone profile)

| Method | Real nRMS | Time |
|---|---|---|
| Hybrid (neural warm-start + Occam) | 2.59 | 164 ms/st |
| Classical Occam (cold) | 2.57 | 391 ms/st |
| **2D U-Net fine-tuned** | **4.59** | ~ms (full section, single pass) |
| 1D neural fine-tuned | 6.14 | 2 ms/st |
| 2D U-Net pretrained | 6.54 | ~ms |
| 1D neural pretrained | 8.36 | 2 ms/st |

## Takeaways

- Fine-tuning cut the 2D real-profile misfit by **30 %** (6.54 -> 4.59) at a
  negligible synthetic cost (+0.018 RMSE) — and coverage actually moved
  *toward* ideal (0.75 -> 0.69), unlike the 1D case which needed
  recalibration after fine-tuning.
- The 2D net is now the **best single-pass method on real data**, overtaking
  the fine-tuned 1D net (4.59 vs 6.14) despite 6x less pretraining data —
  lateral context is worth more than dataset size here.
- Per-station iterative methods (hybrid/Occam, ~2.6) remain ahead in raw
  misfit; they optimise each station independently at run time. The
  remaining 2D gap is the 10k-section dataset and the sigma-overfit early
  stopping — both queued (60k dataset, sigma warm-up).

---

# 60k 2D cycle: scaling results and an honest surprise

60k train / 3k val / 3k test sections (12 h generation, chunk-resumable
after two infra failures — 7 h timeout kill, then an artifact-path bug;
zero sections were ever recomputed thanks to chunk resume). Training with
the two fixes from the 10k post-mortem: sigma warm-up (15 epochs MSE ->
NLL) and inverse-frequency scenario class weights.

## Pretrained model (vs 10k baseline)

| Metric | 10k model | 60k model | Verdict |
|---|---|---|---|
| Synthetic RMSE (500 sections) | 0.646 | 0.637 | ~flat |
| 1-sigma coverage (ideal 0.68) | 0.75 | 0.755 | ~flat |
| Scenario accuracy | 0.26 | 0.29 | small gain |
| Real profile nRMS (no ft) | 6.54 | **6.16** | **-6 %** |

## Fine-tune sweep (200-600 steps, anchor 0.3-10)

Best config aw=3 / 600 steps: real nRMS **5.12**, coverage 0.69,
synthetic RMSE 0.635 (unchanged).

| Model | Real nRMS after ft | Coverage after ft |
|---|---|---|
| 10k + ft (aw=3/200) | **4.59** | 0.69 |
| 60k + ft (aw=3/600) | 5.12 | 0.69 |

## Honest findings

1. **Data scaling hit a wall.** 6x data gave ~0 synthetic improvement.
   The bottleneck is not dataset size but model capacity / task noise
   floor: val RMSE plateaued at ~0.65 within 16 epochs on both datasets.
2. **Sigma warm-up did not fix NLL divergence** — it delayed it (~epoch
   30 vs ~20), then val NLL still exploded (0.005 -> 32 by ep. 79).
   The divergence is driven by the train/val NLL gap, not early sigma
   noise. Proper fix: sigma regularisation or early stop on val NLL.
3. **The 10k fine-tuned model remains the real-data champion (4.59).**
   The 60k pretrained model starts closer to the real data (6.16 vs
   6.54) but fine-tunes to a worse endpoint (5.12 vs 4.59). A plausible
   reading: the 60k model's stronger prior is also stiffer — the anchor
   pulls it back to a sharper synthetic optimum, leaving less room to
   adapt. Single-profile fine-tuning has high variance either way.
4. Scenario head stays weak (0.29-0.30) despite class weights — the
   lenses are simply hard to detect at this resolution; needs
   architectural work (deeper decoder or attention), not loss tweaks.

## Where the leaderboard stands

| Method | Real nRMS | Type |
|---|---|---|
| Occam / Hybrid | 2.57 / 2.59 | iterative per-station |
| **2D U-Net 10k + ft** | **4.59** | single pass |
| 2D U-Net 60k + ft | 5.12 | single pass |
| 2D U-Net 60k pretrained | 6.16 | single pass |
| 1D neural + ft | 6.14 | single pass |

**Conclusion of the scaling experiment:** more synthetic data is not the
lever. The remaining gap to iterative methods is structural (single pass
vs per-station optimisation). The highest-value next step is the 2D
hybrid: U-Net warm-start + a few SimPEG Gauss-Newton iterations, which
in the 1D case closed the gap entirely at 2.4x the classical speed.

---

# 2D hybrid experiment: GN refinement does NOT transfer to 2D

Built `pimsr_benchmarks.hybrid2d`: U-Net warm-start + SimPEG 2D TE
Gauss-Newton refinement (reference-model regularisation toward the net
prediction, per-station static-shift correction against the starting
model, beta cooling). Also added `section_nrms_2d` — the rigorous
2D-forward misfit (shift-invariant, same weights as the 1D metric),
which is the fair score for 2D methods.

## Results, real Yellowstone profile (2D-forward nRMS)

| Method | 2D nRMS | Wall time |
|---|---|---|
| **U-Net 60k + real-profile ft** | **4.47** | ~ms |
| U-Net 60k pretrained | 4.79 | ~ms |
| Cold GN (25 it, control) | 5.15 | 111 s |
| Hybrid GN-8 (best sweep cfg) | 4.91 | 39 s |
| Hybrid GN-25 | 6.85 | 110 s |

## Honest findings

1. **The 1D hybrid recipe did not transfer to 2D.** GN refinement makes
   the warm start *worse* (4.79 -> 4.91 at 8 iters; 6.85 at 25). More
   iterations = more damage: phi_d keeps falling while the shift-
   invariant score rises, i.e. the inversion spends its freedom fitting
   per-station static offsets and noise, not structure. The static-shift
   pre-correction helped but did not fix this: with only 7 stations and
   2 x 24 data points each, the 2D GN problem is badly underdetermined.
2. **Note the metric asymmetry**: on the per-column 1D metric cold GN
   looks better (5.64) than warm (8.00) — but the per-column metric is
   itself biased toward laterally-smooth sections. On the rigorous 2D
   metric every GN variant loses to the neural net.
3. **New champion, properly measured: fine-tuned 2D U-Net at 4.47**
   (2D-forward metric). The earlier 1D-col numbers (4.59 for the 10k
   model) are not directly comparable to iterative 1D methods' 2.6 —
   those fit each station independently, which the 2D forward cannot
   reproduce for a laterally-coherent section.

## Where this leaves the project

The neural path (pretrain + physics fine-tune) is the strongest 2D
method we have; classical 2D refinement on 7 sparse stations hurts more
than it helps. Remaining levers: more stations / denser profiles (data,
not method), sigma regularisation for long training, and scenario-head
architecture. The 1D-vs-2D metric mismatch documented here should also
be fixed in any future leaderboard by scoring everything with the 2D
forward.
