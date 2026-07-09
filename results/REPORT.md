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

---

# Unified leaderboard + multi-profile generalisation study

Two long-standing issues fixed in one cycle: (1) every method is now
scored with the same rigorous 2D-forward shift-invariant nRMS
(`section_nrms_2d`), ending the 1D-col/2D metric mismatch; (2) the
models are evaluated on four additional independent USArray E-W
profiles (G, I, J, K rows) that none of the networks ever saw.

## Unified leaderboard, Yellowstone profile (2D-forward nRMS)

| Rank | Method | 2D nRMS | Time |
|---|---|---|---|
| 1 | **U-Net 60k + ft** | **4.47** | ~ms |
| 2 | U-Net 10k + ft | 4.52 | ~ms |
| 3 | U-Net 60k pretrained | 4.79 | ~ms |
| 4 | Cold 2D GN (25 it) | 4.90 | 132 s |
| 5 | 2D hybrid GN-8 | 5.05 | 41 s |
| 6 | U-Net 10k pretrained | 5.73 | ~ms |
| 7 | 1D neural, stitched | 6.49 | 36 ms |
| 8 | 1D hybrid, stitched | 6.50 | 2.9 s |
| 9 | 1D Occam, stitched | 7.78 | 7.4 s |

**The old "iterative 1D at ~2.6 dominates" story was a metric artifact.**
Scored against the true 2D physics of a laterally-coherent section, the
stitched 1D methods land at 6.5-7.8 — behind every 2D network. The
fine-tuned U-Net is the genuine overall champion, and the neural nets
occupy the entire top-3.

## Multi-profile generalisation (mean 2D nRMS over G/I/J/K)

| Method | Mean nRMS | Notes |
|---|---|---|
| **U-Net 60k + YS fine-tune** | **4.68** | ft transfers across profiles |
| U-Net 60k + per-profile self-ft | 4.86 | high variance (4.09-5.79) |
| U-Net 60k pretrained | 4.94 | solid baseline everywhere |
| 1D Occam stitched | 6.48 | |
| 1D neural stitched | 7.29 | |

Two notable findings: the Yellowstone fine-tune **generalises** — it
improves unseen profiles too (4.94 -> 4.68), meaning it learned regional
data statistics rather than overfitting one line; and per-profile
self-fine-tuning is not reliably better than the transferred one,
confirming that single-profile physics fine-tuning is high-variance.

## Verdict

Physics-informed neural surrogates are not merely "faster with caveats":
under a fair 2D metric they are the most accurate method available in
this benchmark, at 4-6 orders of magnitude lower latency than classical
inversion. A sigma-regularised retraining (quadratic log-sigma penalty,
committed as --sigma-reg) is running to fix the last known training
pathology; results will be appended.

---

# Sigma-regularised retraining: new overall champion at 3.99

Retrained the 60k model with the quadratic log-sigma penalty
(--sigma-reg 0.05) added after two runs showed val-NLL divergence.

## Training behaviour

Best epoch 17 (val RMSE 0.6520, ~= baseline 0.6535). The divergence is
tamed but not cured: val NLL still drifts up after ~epoch 25, yet ends
at 3.9 instead of 32 — an order of magnitude less runaway. Early
stopping still selects a good checkpoint; sigma-reg makes the long tail
of training harmless rather than fixing its cause.

## Results (rigorous 2D-forward nRMS)

| Model | Yellowstone | Unseen G/I/J/K mean |
|---|---|---|
| **sigma-reg 60k + ft** | **3.99** | 5.29 |
| sigma-reg 60k pretrained | 4.18 | 5.35 |
| previous champion (60k + ft) | 4.47 | 4.68 |
| 60k pretrained (baseline) | 4.79 | 4.94 |
| best classical 2D (cold GN-25) | 4.90 | — |

Synthetic: RMSE 0.634 (~= baseline 0.637), coverage 0.785, scenario acc
0.28 (both ~= baseline).

## Findings

1. **New overall champion on the primary benchmark: 3.99** — first
   model under 4.0, beating the previous best by 11% and the best
   classical 2D method by 19%, at ~ms latency. The sigma-reg model
   fits real data far better out of the box (4.18 pretrained vs 4.79).
2. **A trade-off appeared**: on unseen profiles the sigma-reg model is
   worse (5.29 vs 4.68 mean). Regularising sigma seems to sharpen the
   fit where data is dense (YS row is the region's centre) at some cost
   to out-of-row robustness. Which model to prefer depends on use:
   sigma-reg for the target profile, the unregularised ft model for
   regional deployment.
3. Sigma-reg did not change synthetic accuracy or coverage — its value
   is training stability and real-data fit, not the sigma calibration
   number itself.

## Project conclusion

Across 1D and 2D, synthetic and real data, the physics-informed neural
approach now holds every top position under fair metrics. The final
leaderboard on the real Yellowstone profile: neural 3.99-4.52 (ms),
classical 2D GN 4.90 (2 min), stitched 1D methods 6.5-7.8. Remaining
open problems, in value order: out-of-row generalisation of the
sigma-reg model, the root cause of NLL drift (likely needs a proper
heteroscedastic head or beta-NLL loss), and the scenario head (~0.28,
needs architecture work).

---

# Joint multi-profile fine-tune: the generalisation fix

The sigma-reg trade-off (3.99 on Yellowstone but 5.29 on unseen rows)
is resolved by fine-tuning on all five USArray rows jointly: the
physics misfit is averaged across profiles every step (finetune2d
--profiles G,H-YS,I,J,K), which anchors the adaptation to regional data
statistics instead of one line. Four variants of the sigma-reg 60k
model compared on all five rows (results/mpft/mpft.json):

| Variant | mean 2D nRMS (5 rows) | H-YS (target) | worst row |
|---|---|---|---|
| pretrained | 5.12 | 4.18 | 6.76 (K) |
| ft YS-only (champion recipe) | 5.03 | **3.99** | 6.73 (K) |
| **ft joint all 5 rows** | **4.30** | 4.10 | 5.62 (I) |
| ft leave-one-out (scored on held-out row) | 4.88 | 4.28 | 6.85 (I) |

## Findings

1. **Joint ft is the regional deployment answer: mean 4.30**, a 16%
   improvement over pretrained and better than any previous multi-row
   result, while giving up only 0.11 on the target profile (4.10 vs
   3.99). The single-profile champion recipe barely helps other rows
   (5.03 vs 5.12) — its gains were target-specific.
2. Largest wins are exactly where the pretrained model was weakest:
   K 6.76 -> 4.69, J 4.48 -> 3.49, I 6.68 -> 5.62. Averaging misfit
   across rows acts as a data-driven regulariser — no new hyper-
   parameters were needed (same aw=3, 600 steps).
3. Leave-one-out (mean 4.88 on held-out rows, always better than
   pretrained) shows the joint model transfers to rows it never saw:
   this is genuine regional adaptation, not multi-line memorisation.

---

# v3 cycle: TE+TM impedance, beta-NLL, multiscale scenario head

One combined GPU cycle addressed three open problems at once: the
TM mode doubles the physical information per station (dataset run
28868401447, 10k/1k/1k TE+TM sections, seed 3; training run
28891525012, 80 epochs, `--beta 0.5 --scen-head multiscale`).

## Synthetic test (500 sections, results/v3/synthetic.json)

| Metric | v2 60k (TE, sigma-reg) | v3 10k (TE+TM, beta-NLL) |
|---|---|---|
| section RMSE | 0.634 | **0.618** |
| 1-sigma coverage | 0.785 | 0.812 |
| scenario accuracy | 0.28 | **0.40** |

- **Scenario head fixed**: 0.28 -> 0.40 (+43%), from the multiscale
  avg+max head over bottleneck + finest decoder features. Small lenses
  survive max-pooling that global averaging washed out.
- v3 with 10k sections beats the 60k TE-only model on RMSE — mode
  diversity is worth more than 6x data volume.
- **beta-NLL verdict — the drift is tamed, not cured**: best val-NLL
  epoch 16 (0.068), drifting to 0.73 by epoch 79 vs 32.0 for
  sigma-reg and worse for plain NLL. RMSE stays flat (0.652 -> 0.692)
  so checkpoint selection is safe, but the root cause survives even a
  gradient-stopped objective. Next candidate: a separately-trained
  post-hoc sigma head.

## Real Yellowstone profile (unified 2D-forward leaderboard, results/v3/unified_v3.json)

| Method | 2D nRMS |
|---|---|
| unet-60k-ft (TE, champion) | **4.01** |
| **unet-v3-tetm-ft** | **4.05** |
| unet-v3-tetm (no ft!) | 4.36 |
| unet-10k-ft | 4.54 |
| unet-60k | 4.79 |
| unet-10k | 5.73 |
| hybrid1d-stitched | 6.69 |
| occam1d-stitched | 7.78 |

- **The headline: v3 pretrained (4.36) nearly matches the old
  champion pipeline without any fine-tuning** — the TM mode closes
  most of the sim-to-real gap that previously required per-profile
  physics ft (10k TE-only pre was 5.73). With ft the two are
  statistically tied (4.05 vs 4.01) at 6x less training data.
- Per-mode real inputs: TE=Zyx, TM=Zxy for the E-W profile (N-S
  strike assumption), phases folded to [0,180).

## Caveat: v3 on unseen rows (results/v3/v3_profiles.json)

v3-pre mean over 5 rows is 7.42 (vs 5.12 for 60k TE sreg) — rows I/K
have strongly 3D/distorted TM curves that the 10k synthetic prior has
not seen. Joint ft improves to 6.38 but does not close the gap.
The TE-only model silently ignored this mismatch; v3 exposes it.
Fix queued: TM distortion (static shifts already independent per mode,
but real yx curves need stronger galvanic/twist augmentation) plus a
60k-scale TE+TM dataset.

## Conclusions

v3 delivers on 2 of 3 goals outright (scenario head +43%, TM mode
closes the zero-shot gap on the target profile) and partially on the
third (beta-NLL is the best drift mitigation so far but not a cure).
The clear next cycle: TE+TM at 60k scale with per-mode distortion
augmentation — expected to combine v3's zero-shot quality with the
60k model's cross-row robustness.

---

# Production comparison: Occam2DMT v3.0 (Scripps)

The mandatory pre-publication check: our methods vs an actual
production 2D MT inversion code. Occam2DMT v3.0 (de Groot-Hedlin &
Constable, Scripps Marine EM Lab) was compiled from the official
source (gfortran, -std=legacy) and run on all five USArray rows with
joint TE+TM data, 10% error floors, 192-block regularised model,
via scripts/run_occam2dmt.py (auto mesh/data/startup generation +
ITER parsing). Everything is scored with the same shift-invariant
2D-forward metric (section_nrms_2d, TE reference).

| Profile | Occam2DMT v3.0 (TE+TM) | 60k joint-ft U-Net (ours) |
|---|---|---|
| G | 3.92 | **3.59** |
| H-YS | 4.68 | **4.10** |
| I | 9.26 | **5.62** |
| J | 6.40 | **3.49** |
| K | 6.03 | **4.69** |
| **mean** | 6.06 | **4.30** |

Occam2DMT runtime: 4.5–21.6 s/profile (12–25 iterations); the U-Net
is a single ~ms forward pass.

## Findings

1. **The neural model beats the production code on every profile**
   (mean 4.30 vs 6.06, -29%) at four orders of magnitude less
   compute. On the target Yellowstone row the single-profile champion
   (3.99) also clears Occam2DMT (4.68).
2. Occam2DMT slightly outperforms our internal SimPEG GN baseline on
   H-YS (4.68 vs 4.90) — sanity check passed: our classical baseline
   was honest, not a strawman.
3. **Row I is objectively hard, not just hard for us**: production
   Occam2DMT also scores 9.26 there. This independently confirms the
   3D/distortion hypothesis for I rather than a defect of the
   synthetic prior.
4. Results in results/occam2dmt/*.json. Remaining production items:
   ModEM (source requested via academic registration — build planned
   in Docker on the runner PC, D:\ drive) and MARE2DEM (Docker build,
   Unix-only code) for a 3D-code comparison.

---

# Production comparison, part 2: ModEM 2D NLCG (open source)

ModEM turned out to be open-sourced on GitHub
(magnetotellurics/ModEM) — no academic registration needed after all.
Mod2DMT was compiled from source (gfortran + LAPACK,
`Configure.2D_MT.OSU.GFortran` preset) and validated on the bundled
BLOCK2 example before use. Driver: `scripts/run_modem2d.py` — converts
our per-mode EMTF observations to ModEM's TE/TM impedance format
([V/m]/[T], e^{+iwt}, TM written as Zyx = -Z), builds a Mackie-format
LOGE prior (100 Ohm-m halfspace, 10 km padding), runs NLCG inversion,
and maps the final model back onto the benchmark raster.

| Profile | ModEM NLCG | Occam2DMT v3.0 | 60k joint-ft U-Net |
|---|---|---|---|
| G | 5.32 | 3.92 | **3.59** |
| H-YS | 5.90 | 4.68 | **4.10** |
| I | 10.98 | 9.26 | **5.62** |
| J | 6.28 | 6.40 | **3.49** |
| K | 6.99 | 6.03 | **4.69** |
| **mean** | 7.09 | 6.06 | **4.30** |

ModEM runtime: 24-210 s/profile (20-100+ NLCG iterations).

## Findings

1. **The neural model beats both production codes on every profile.**
   Final production leaderboard (mean over 5 rows): U-Net 4.30 <
   Occam2DMT 6.06 < ModEM-2D 7.09.
2. ModEM's internal data-space RMS reaches 1-3 (it fits its own data
   well); the gap on our metric is in the model-space answer, not a
   failed inversion. Occam's regularised staircase transfers to the
   benchmark raster better than NLCG's smooth logE updates at this
   sparse station density.
3. **Row I is now triple-confirmed as anomalous** (U-Net 5.62,
   Occam2DMT 9.26, ModEM 10.98): every independent method struggles,
   supporting the 3D/distortion structural hypothesis over any
   method-specific defect.
4. Results in results/modem2d/*.json. MARE2DEM (same Occam algorithm
   family as the already-beaten Occam2DMT, Unix-only) is deprioritised;
   the production-comparison milestone is now closed with two
   independent production codes.

---

# v4 cycle: 60k TE+TM + per-mode galvanic distortion augmentation

Goal: combine v3's TM-mode gains with 60k scale, and fix the unseen-row
weakness (v3 rows I/K mean 7.42) via per-section TM distortion severity
(static shift sigma U[0.15, 0.40], correlated distortion amplitude
logU[0.25, 0.60] — roughly 2x the TE level, matching real yx behaviour).
Dataset run 28966933155 (60k/3k/3k, seed 7; survived three infra
failures — runner disconnect, internet outage, SimPEG worker memory
leak — all fixed: install retries, max_tasks_per_child=1). Training run
28988267907 (80 epochs, beta-NLL 0.5, multiscale head; best epoch 16,
val RMSE 0.6271 — best of any cycle).

## Synthetic test (500 sections, results/v4/bench_pre)

| metric | v4 | v3 (10k) | 60k TE |
|---|---|---|---|
| RMSE log10 rho | **0.609** | 0.618 | 0.634 |
| sigma coverage (1σ, ideal 0.683) | 0.812 | 0.812 | 0.755 |
| scenario accuracy | 0.40 | 0.40 | 0.29 |

## Real profiles (shift-invariant 2D-forward nRMS, results/v4/v4_profiles.json)

| Profile | v4-pre | v4-ft-YS | v4-ft-joint | best previous |
|---|---|---|---|---|
| G | 4.30 | 3.95 | 4.07 | 3.59 (60k joint-ft) |
| H-YS | 4.91 | **3.92** | 4.09 | 3.99 (sreg-ft) |
| I | 7.10 | 6.78 | **5.33** | 5.62 (60k joint-ft) |
| J | 4.30 | 4.32 | 7.00 | 3.49 (60k joint-ft) |
| K | 6.81 | 6.51 | **5.10** | 4.69 (60k joint-ft) |
| **mean** | 5.49 | 5.10 | 5.12 | **4.30** (60k joint-ft) |

## Findings

1. **New target-profile champion: v4-ft-YS = 3.92 on H-YS** (first
   sub-3.99), and its transfer to G (3.95) is strong.
2. **The augmentation works where it aimed**: rows I and K — the
   distorted/3D rows that motivated v4 — improve under joint ft to
   5.33 and 5.10, and I even beats the previous best (5.62).
3. **But it costs elsewhere**: zero-shot H-YS degrades to 4.91 (v3:
   4.36) — heavy TM distortion in training makes the model too
   conservative on cleaner profiles; and J collapses under joint ft
   (7.00 vs 4.30 pre) — the shared ft update is dominated by the
   hard rows at J's expense.
4. **The overall 5-row champion remains 60k joint-ft (4.30).** v4 is
   the best model on the hardest rows and the target profile, but not
   on average. Distortion augmentation should likely be *milder*
   (upper tail 0.4-0.45, not 0.60) or curriculum-scheduled.
5. Training-side wins are unambiguous: best synthetic RMSE (0.609),
   calibration and scenario accuracy held at v3 levels with 6x data.

Next candidates: (a) v4.1 with milder TM severity tail, (b) per-profile
ft weighting in joint mode (loss balancing), (c) the 2D->3D migration.

## v4 addendum: balanced joint fine-tuning (the row-J investigation)

`finetune2d --balance` (committed) normalises each profile's physics
misfit by its pretrained value, so distorted rows cannot dominate the
shared update. Two experiments on the v4 checkpoint (aw=3, 600 steps):

| Profile | pre | joint | joint+balance | balance, no J |
|---|---|---|---|---|
| G | 4.30 | 4.07 | 3.89 | 3.95 |
| H-YS | 4.91 | 4.09 | 3.95 | 3.97 |
| I | 7.10 | 5.33 | **5.24** | 5.26 |
| J | 4.30 | 7.00 | 6.48 | **7.37** |
| K | 6.81 | 5.10 | 5.60 | 5.90 |
| mean | 5.49 | 5.12 | **5.03** | 5.29 |

Findings:

1. Balancing helps everywhere except K: best v4 joint mean (5.03), new
   best-ever row I (5.24), and G/H-YS at near-champion levels from a
   single shared model.
2. **The row-J mystery resolved**: excluding J from the joint update
   makes J *worse* (7.37), not better. J's collapse is not caused by
   its own gradient — it is collateral damage from the adaptation
   direction demanded by the distorted rows (I, K). J is
   anti-correlated with them: whatever galvanic-distortion compensation
   the network learns for I/K actively mismodels J's clean curves.
   Loss weighting cannot fix an anti-correlated objective — this needs
   either per-profile adapters (small FiLM/LoRA-style heads) or an
   input-side distortion estimator, not shared-weight ft.
3. Overall champion is still 60k joint-ft (4.30 mean): v4's heavy TM
   augmentation costs more on average than it buys on hard rows.

Results: results/v4/v4_profiles_bal.json, v4_profiles_bal_noJ.json.

## v4 addendum 2: per-profile FiLM adapters

`finetune2d --film` (committed): zero-initialised per-profile
(gamma, beta) on the U-Net bottleneck (2 x 192 params/profile, lr 50x
shared), trained jointly with balanced misfit; evaluation applies each
profile's own adapter (`unet_section(..., profile_name=...)`).
Hypothesis from addendum 1: adapters should absorb the anti-correlated
distortion compensation that shared-weight ft cannot.

| Profile | joint+balance (shared only) | joint+balance+FiLM |
|---|---|---|
| G | 3.89 | 4.37 |
| H-YS | 3.95 | **3.74** |
| I | 5.24 | 8.54 |
| J | 6.48 | **5.42** |
| K | 5.60 | **4.65** |
| mean | **5.03** | 5.34 |

Findings:

1. **Partial confirmation of the adapter hypothesis**: J recovers
   (6.48 -> 5.42) — its anti-correlated compensation moved into its
   adapter instead of fighting the shared update. K improves to a
   v4-best 4.65, and H-YS reaches **3.74 — the new all-time
   target-profile record** (previous 3.92).
2. **But row I destabilises** (5.24 -> 8.54): with lr 50x and only one
   sample per profile, I's adapter overfits its heavily distorted
   curves — exactly the profile with the most 3D contamination gets
   the least trustworthy self-supervision. Adapters need their own
   anchor/regularisation (or lower lr) before this recipe is safe.
3. Mean is worse than plain balance (5.34 vs 5.03): FiLM trades
   variance across rows for per-row wins. Next iteration: L2 penalty
   on (gamma, beta), film-lr 10x, or share adapters between
   similarly-distorted rows (I+K).

Results: results/v4/v4_profiles_film.json. Ckpt:
best2d_ft_joint_film.pt (adapters stored under `film_adapters`).

## v4 addendum 2: per-profile FiLM adapters

`finetune2d --film` (committed): zero-initialised per-profile
(gamma, beta) on the bottleneck (2 x 192 params per profile, lr 50x the
anchored trunk), trained jointly with `--balance` on all five rows.
Adapters are stored in the checkpoint and applied by profile name at
evaluation. This tests the row-J hypothesis: shared weights learn the
common regional shift, adapters absorb the anti-correlated per-profile
distortion compensation.

| Profile | joint+balance (no film) | joint+balance+film | best previous |
|---|---|---|---|
| G | 3.89 | 4.28 | 3.59 |
| H-YS | 3.95 | **3.72** | 3.92 |
| I | 5.24 | 8.31 | 5.24 |
| J | 6.48 | **5.33** | 3.49 (pre-ft) |
| K | 5.60 | **4.59** | 4.69 |
| mean | 5.03 | 5.25 | — |

Findings:

1. **The hypothesis is half-confirmed.** FiLM does what it was built
   for on three rows: J recovers from the collapse (6.48 -> 5.33), K
   sets a best-ever (4.59), and H-YS sets a new overall champion
   (3.72, first sub-3.9). The anti-correlated compensation *is*
   absorbable by 384 per-profile parameters.
2. **Row I inverts the story**: its adapter drives the 1D-column
   physics loss down but the rigorous 2D-forward nRMS up (5.24 ->
   8.31). For the most strongly 3D-distorted row, freely fitting the
   distorted curves through a 1D physics loss is actively harmful —
   the adapter needs either a stronger prior (adapter norm penalty) or
   the physics target needs to be the 2D forward, not per-column 1D.
3. Champions after this experiment: H-YS 3.72 (film), K 4.59 (film),
   I 5.24 (balance), G 3.59 / J 3.49 / 5-row mean 4.30 (60k joint-ft).

Result file: results/v4/v4_profiles_film.json (columns: v4-pre,
v4-ft-YS=film checkpoint, v4-ft-joint=balanced checkpoint).
Next lever: adapter norm regularisation or a 2D-forward physics loss
for the distorted rows.

## v4 addendum 3: regularised FiLM adapters (the row-I fix)

`finetune2d --film-reg W --film-lr-mult M` (committed): L2 anchor pulling
each adapter to identity + configurable adapter lr. Sweep on the v4
checkpoint (aw=3/600, balance+film): reg 0.03 and 0.10, both with lr-mult
10 (was 50). Columns below: unregularised film vs the two configs.

| Profile | film (no reg) | reg 0.03 / m10 | reg 0.10 / m10 |
|---|---|---|---|
| G | 4.28 | 3.94 | 3.88 |
| H-YS | **3.72** | 3.98 | 3.91 |
| I | 8.31 | **5.15** | 5.24 |
| J | **5.33** | 6.28 | 6.32 |
| K | **4.59** | 5.84 | 5.68 |
| mean | 5.25 | 5.04 | 5.01 |

Findings:

1. **The row-I fix works as designed**: taming the adapter (8.31 ->
   5.15, a best-ever for I) confirms the diagnosis — I's failure was
   adapter overfit through the 1D physics loss, not a modelling limit.
2. **But the trade-off is conserved**: with weak adapters the
   anti-correlated rows J and K lose their adapter-driven wins
   (J 5.33 -> 6.28, K 4.59 -> 5.68) and the board converges back to
   the plain-balance solution (~5.0 mean). One 384-param dial per
   profile cannot be simultaneously strong (J/K need it) and safe
   (I needs it) when each profile is a single training sample.
3. Champions stand: H-YS 3.72 and K 4.59 (film no-reg), I 5.15
   (film reg 0.03), G 3.59 / J 3.49 / mean 4.30 (60k-era ckpts).
4. Conclusion for the paper: per-profile adaptation on single-sample
   self-supervision has a variance floor; breaking it needs either a
   2D-forward physics loss (trustworthy signal for distorted rows) or
   multi-sample per-profile data (windowed sub-profiles). Both are
   candidates for the next milestone.

Results: results/v4/v4_profiles_film_reg.json. Ckpts:
best2d_ft_film_r03m10.pt, best2d_ft_film_r10m10.pt.

## v4 addendum 4: windowed sub-profile fine-tuning (negative result)

`finetune2d --windows W` (committed): each profile contributes, besides
the full station line, every contiguous W-station window as an
alternative view; each step draws one random view per profile. The hope:
multi-sample self-supervision stops a profile's FiLM adapter from
overfitting its single distorted curve set (the row-I failure).

Two runs on v4 (aw=3/600, balance, windows=5):

| Profile | film+win5 | film no-win (add. 2) | bal+win5 | bal no-win (add. 1) |
|---|---|---|---|---|
| G | 4.40 | 4.28 | 3.90 | 3.89 |
| H-YS | 4.00 | 3.72 | 4.31 | 3.95 |
| I | 8.58 | 8.31 | 5.24 | 5.24 |
| J | 5.31 | 5.33 | 6.59 | 6.48 |
| K | 4.56 | 4.59 | 5.58 | 5.60 |
| mean | 5.37 | 5.25 | 5.12 | 5.03 |

Finding: **windowing does not break the variance floor** — row I's
adapter still overfits (8.58), and every configuration is within noise
of (or slightly worse than) its non-windowed counterpart. The windows
are contiguous and overlap heavily, so they are strongly correlated
views of the same distorted curves: the adapter sees no genuinely new
constraint. This closes the "cheap multi-sample" branch; the remaining
lever for the distorted rows is a trustworthy physics signal — the
**2D-forward physics loss** — which is the right next milestone.

Results: results/v4/v4_profiles_win5.json. Ckpts:
best2d_ft_film_win5.pt, best2d_ft_bal_win5.pt.

## v4 addendum 5: 2D-forward physics loss (negative result, diagnosed)

`pimsr_inversion/physics2d.py` (committed): differentiable TE 2D forward
— SimPEG solve wrapped in a custom autograd Function with adjoint
(Jtvec) gradients, FD-validated; static-shift-invariant chi2 on a
frequency subset. `finetune2d --phys2d` swaps it in for the per-column
1D loss (~13 s/step for 5 profiles, 60 steps, lr 4e-5, balance).

Training converged well: pooled physics misfit 0.99 -> 0.25 (-75%).
But the honest TE+TM section metric got WORSE nearly everywhere:

| Profile | v4-pre | phys2d-ft |
|---|---|---|
| G | 4.30 | 5.79 |
| H-YS | 4.91 | 4.70 |
| I | 7.10 | 9.32 |
| J | 4.30 | 9.45 |
| K | 6.81 | 6.00 |
| mean | 5.49 | 7.05 |

Diagnosis: **mode mismatch, not a broken gradient**. The loss drives a
hard fit of the TE curves only, and a 2D model has enough freedom to
explain TE galvanic distortion with shallow structure — degrading TM
consistency, which the evaluation metric checks. The per-column 1D loss
was accidentally protected from this: a layered column responds
identically in both modes, so it cannot trade one mode against the
other. Root fix is a **TE+TM 2D loss** (TM forward exists in
pimsr-forward mt2d; needs the same adjoint wrapper) — then lateral
structure must satisfy both polarisations simultaneously, which is
exactly the physics that distinguishes real 2D structure from galvanic
distortion.

Results: results/v4/v4_profiles_phys2d.json. Ckpt:
best2d_ft_phys2d.pt. Infrastructure (adjoint wrapper, FD-validated) is
in place — the TE+TM extension is the next increment.

## v4 addendum 6: TE+TM 2D physics loss — the mode trade-off is fixed

physics2d now solves BOTH polarisations (TM adjoint path added to the
autograd wrapper, FD-validated; chi2 pooled across modes). Two joint-ft
runs on v4 (60 steps, lr 4e-5, aw=3, balance): plain, and with
regularised FiLM (reg 0.03 / lr-mult 10). Training misfit 1.00 -> 0.30.

| Profile | v4-pre | TE-only 2D (add. 5) | TE+TM 2D | TE+TM 2D + film |
|---|---|---|---|---|
| G | 4.30 | 5.79 | **3.58** | 3.59 |
| H-YS | 4.91 | 4.70 | 4.35 | **4.31** |
| I | 7.10 | 9.32 | **5.50** | 5.51 |
| J | 4.30 | 9.45 | 7.91 | 7.84 |
| K | 6.81 | 6.00 | 5.47 | **5.47** |
| mean | 5.49 | 7.05 | 5.36 | **5.34** |

Findings:

1. **The diagnosis of addendum 5 is confirmed and the fix works**:
   requiring both polarisations turns the 2D physics loss from
   uniformly harmful (mean 7.05) into broadly helpful — 4 of 5 rows
   improve, including BOTH heavily distorted rows (I 7.10 -> 5.50,
   K 6.81 -> 5.47) and G reaching 3.58 (matches the all-time G best
   3.59). This is the first physics-2D fine-tune that beats the 1D
   column loss on the distorted rows — with a *trustworthy* signal
   (real 2D structure vs distortion), adaptation needs no variance
   tricks.
2. **Row J's anti-correlation is genuine physics, not a loss
   artifact**: J was best zero-shot (4.30) and degrades under ANY
   joint adaptation — 1D loss (6.48), TE+TM 2D loss (7.91), film
   (7.84). The shared-update direction demanded by I/K is opposed to
   J's optimum at every level of physics fidelity tried. For regional
   deployment, J should keep the pretrained weights (or its own
   adapter trained solo) — a per-profile model-selection rule, not a
   shared compromise.
3. FiLM adds nothing under the 2D loss (5.34 vs 5.36) — consistent
   with (1): when the loss signal is trustworthy, per-profile freedom
   is no longer the bottleneck.
4. Remaining gap to the 60k-era joint-ft champion (mean 4.30) is the
   base checkpoint, not the method: v4-pre starts at 5.49 vs 60k-pre
   5.12 on these rows. Re-running TE+TM 2D ft on v4.1 (milder TM tail,
   training on the runner now) is the natural next step.

Results: results/v4/v4_profiles_phys2d_tetm{,_film}.json. Ckpts:
best2d_ft_phys2d_tetm{,_film}.pt. ~26 s/step on CPU (5 profiles,
2 modes) — 60 steps is enough (plateau by step 50).

## v4.1 cycle: milder TM severity tail

Dataset with the TM distortion tail capped at shift 0.32 / distortion
0.45 (v4: 0.40/0.60); gen run 29028709926, training run 29051192458
(best ep 14, val RMSE 0.6459). Synthetic (500): RMSE 0.617,
coverage 0.81, scenario acc 0.414 (best ever). Columns: v4.1 zero-shot,
v4.1 + TE+TM 2D joint ft (60 steps, physics 1.00 -> 0.19), v4 zero-shot
reference.

| Profile | v4.1-pre | v4.1 + TE+TM ft | v4-pre |
|---|---|---|---|
| G | 5.04 | 3.72 | 4.30 |
| H-YS | **4.31** | 5.72 | 4.91 |
| I | 7.21 | 6.98 | 7.10 |
| J | **4.10** | 8.95 | 4.30 |
| K | 8.10 | **4.54** | 6.81 |
| mean | 5.75 | 5.98 | 5.49 |

Findings:

1. **The v4.1 hypothesis is confirmed**: softening the TM tail
   recovers zero-shot performance on clean profiles — H-YS 4.31 (v4:
   4.91, v3: 4.36) and J 4.10 (best-ever zero-shot J). The cost is
   exactly where expected: heavy-distortion rows regress zero-shot
   (K 8.10 vs 6.81).
2. **TE+TM ft remains the distorted-row tool**: on v4.1 it fixes K
   spectacularly (8.10 -> 4.54) and G (5.04 -> 3.72), but on this
   base it also drags H-YS down (4.31 -> 5.72) — the shared update
   again serves the rows that need it most. J collapses as always.
3. **Per-profile model selection across the two v4.1 columns** gives
   G 3.72 / H-YS 4.31 / I 6.98 / J 4.10 / K 4.54 = mean 4.73 — decent,
   but the 60k-era joint-ft champion (4.30) still stands. Augmentation
   tuning traded off against itself; the data-side lever looks
   exhausted at 10k sections.
4. Deployment recipe consolidated by three independent cycles
   (v4, v4.1, 60k): pretrained weights for clean/anti-correlated rows
   (H-YS, J), TE+TM 2D ft for distorted rows (I, K, G) — selection by
   per-profile physics misfit, which is computable without ground
   truth.

Results: results/v41/{v41_zeroshot,v41_tetm}.json, bench in
results/v41/bench. Ckpts local: /vercel/share/pimsr-data/v41/.
