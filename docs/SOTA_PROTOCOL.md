# PIMSR SOTA comparison protocol

Version: 1.1 (2026-08-24)
Registry schema: `config/sota_methods.json`, version 2

## Purpose and admissibility

This protocol defines when a PIMSR result may be compared with a modern MT
inversion method. A result is **publishable on the main leaderboard** only when
all of the following hold:

1. the method and dataset pass the fail-closed registry validator;
2. the exact source, executable, checkpoint, data and configuration artifacts
   are content-addressed in the run manifest;
3. both methods receive the same scored observations in physical coordinates;
4. training, selection, refinement and evaluation data are disjoint as required
   by the declared track;
5. every required metric, seed and provenance field below is present.

Paper-reported numbers are never copied into the executable leaderboard.
Methods without independently runnable artifacts stay on a clearly labelled
paper-only shadow board. Historical PIMSR runs using different component,
geometry, masking or weighting contracts stay outside this protocol.

## Registry contract

`pimsr_benchmarks.sota.load_registry` is the normative validator. It rejects
unknown or missing fields, duplicate IDs, short Git hashes, floating refs such
as `main`, `master`, `HEAD` and `latest`, malformed DOI/URL fields, unsupported
tracks/statuses, and inconsistent artifact or license records.

Every runnable source is selected by either a full 40-character commit SHA or a
tag/version plus its resolved full SHA. A tag alone is not sufficient. Every
downloaded archive, package, dataset and model file receives a SHA-256 digest in
the run manifest even when its landing page or DOI is persistent. Registry
license fields are operational constraints, not legal advice: an unknown or
missing license forbids redistribution in a benchmark image unless permission
is obtained.

The first executable wave is GAN-MT1DInv and the physically guided unsupervised
notebook in 1D; RDON, MT2DInv-DenseNet, MTDLPy and MARE2DEM in 2D; and ModEM,
MT3D_CNN, GEMMIE, FEMTIC and the public DEVA3DMT research edition in 3D. SimPEG
is a numerical reference, not a learned-SOTA claim. MT-Mamba, P-PhysInv,
MT2DInv-Unet, MT3D-Net, the 1D implicit-neural-representation preprint and the
trans-scale 2D method remain paper-only until an independent source/checkpoint
bundle can be pinned and executed.

Artifact availability is track-specific. In particular, the DenseNet v1.2
Zenodo record contains source scripts but no released training data or weights,
so it is eligible for common retraining but not the frozen-artifact track.
GEMMIE's Zenodo record contains synthetic impedance examples; its executable
source is the separately pinned GitLab commit.

## Separate comparison tracks

Each result row belongs to exactly one track. Scores from different tracks must
not be averaged, ranked together or described as head-to-head equivalents.

| Track | Allowed adaptation | Required disclosure |
| --- | --- | --- |
| `frozen_artifact` | Released checkpoint is unchanged. Classical inversion uses a pinned executable and preregistered inversion configuration. No evaluation-profile tuning is allowed. | Upstream artifact IDs, preprocessing state, fixed solver settings and whether the method was originally trained on any related data. |
| `common_retrain` | A trainable method is initialized according to a preregistered rule and trained only on the common training split and observation budget. | Initialization, optimizer, schedule, selection metric, training compute and all attempted configurations. |
| `refinement` | A frozen prediction or common starting model may be adapted to observations assigned to the refinement split. | Starting model, optimized parameters, objective, iteration/early-stop rule, refinement observations and compute. |

Fine-tuning on a scored field profile is not a frozen or field-heldout result.
If reported for an adaptation study, that profile is moved to the refinement
split and evaluated on a different held-out profile. A classical inversion
started from a PIMSR prediction belongs to `refinement`; the same solver started
from the common reference model belongs to `frozen_artifact`.

## Observation and geometry contract

### Physical geometry

Publishable field and synthetic scores use actual station coordinates, survey
azimuth, elevations/topography when applicable, periods/frequencies and model
cell coordinates in SI units. Full impedance tensors are rotated once from the
archive convention into the declared survey frame. The run manifest records the
input/output axes, handedness, vertical sign, electric/magnetic units and time
dependence. In the PIMSR 2-D profile frame, TE/TM labels must follow the shared
forward-model contract after rotation.

The following are diagnostic only and cannot enter the physical leaderboard:

- affine compression of a hundreds-of-kilometres field profile into the fixed
  synthetic model width;
- interpolation of real stations into a denser pseudo-station array;
- evaluating a 2-D result with geographic tensor components that were not
  rotated to the profile frame;
- extrapolating periods outside a station's measured band;
- comparing rasters through index coordinates rather than common physical cell
  volumes or centres.

Methods unable to consume irregular physical geometry may be evaluated on a
separate, explicitly normalized-geometry diagnostic board. That board cannot
support field-resolution or geological superiority claims.

### Equal observation budget

For each paired comparison, construct one immutable observation manifest before
running any method. It contains the exact station IDs and coordinates, period
indices, impedance/tipper components, valid-data mask, uncertainty/error floor,
noise realization, static-shift policy and preprocessing transform. Every method
is scored on that same intersection. Missing values stay missing; method-specific
edge extrapolation is forbidden.

Joint TE+TM comparisons give all methods both modes and score both with the same
component weights. A method that accepts only one mode enters a separately named
single-mode track; its model cannot be scored against a joint-mode method under
the same headline. Tipper and other components follow the same rule. Nuisance
handling (static shift, distortion, topography and error floors) is either common
preprocessing or part of every method's declared inverse problem, never a
method-dependent preprocessing step derived from that method's starting model.

## Splits and inverse-crime prevention

Synthetic examples are grouped by geological family before splitting. Variants
of one base model, including alternate noise, station, frequency or mesh
realizations, remain in one group. At minimum, keep immutable train, validation,
refinement and hidden test manifests. Test truth and responses are unavailable
to training, hyperparameter selection, early stopping and manual configuration.

For PIMSR-generated tests, the independent scoring forward solve must differ
materially from the training generator. Record at least:

- independent model/family seeds and no overlapping sample IDs;
- a separately implemented or independently verified forward solver;
- a different mesh/discretization, with convergence checked by refinement;
- a hidden noise seed and a preregistered sensor/noise distribution;
- no checkpoint or normalization statistic fitted on test observations.

If a benchmark method's released training set overlaps a selected public test,
either prove non-overlap from manifests or mark that row contaminated. Public
forward benchmarks such as the MT3DINV4 sphere are solver checks, not suitable
hidden learned-inversion tests after they have been used for development.

Field evaluation is held out by survey/profile, not by individual station. No
field-heldout profile may influence architecture, checkpoint choice,
normalization, error floors, rotation choices, regularization or stopping rules.
COPROD2 original/corrected variants, COPROD2S1/S2, MT3DINV4 noise variants and
Raglan raw/corrected/repeated-site choices are distinct immutable inputs.

## Required metrics

Report individual samples and aggregates. A single mean is insufficient.

### Model-space metrics (synthetic truth only)

- volume/area-weighted log10-resistivity RMSE and MAE on a declared common
  physical support;
- per-depth and per-geological-family error;
- anomaly intersection-over-union and boundary distance using thresholds fixed
  before evaluation;
- structural similarity only as a supplementary metric, with data range fixed
  globally rather than per sample.

Interpolation to the scoring mesh is performed once by a method-independent
script, conserves physical coordinates and records the interpolation operator.
Air, padding, fixed cells and regions outside the common support are masked
identically. Model-space metrics are `N/A`, never zero, for field data or a
withheld secret model.

### Data-space and physics metrics

- uncertainty-whitened complex-impedance NRMS on the common mask;
- log-apparent-resistivity error and circular phase error, with phase residuals
  wrapped modulo 180 degrees;
- per-component, per-period and per-station summaries before any joint average;
- forward consistency through one independent scoring solver on one common
  scoring mesh, not through a learned surrogate owned by the evaluated method.

Data fit is not geological truth. Field results must be labelled as predictive
or forward consistency and interpreted alongside held-out data, sensitivity and
uncertainty; lower training-data misfit alone is not evidence of a better Earth
model.

### Uncertainty metrics

Probabilistic methods report empirical interval coverage at preregistered levels,
interval width/sharpness, calibration error and proper scores (NLL and CRPS when
the predictive representation permits them). Calibration is measured on held-out
models/families. Deterministic methods report uncertainty metrics as `N/A` and
are not assigned artificial zero uncertainty. Bootstrap ensembles must include
all member seeds and their training/inference cost.

### Runtime and resource metrics

Report training, preprocessing, compilation/mesh construction, refinement and
inference/inversion separately. Include wall-clock time, CPU time when available,
peak host RAM, peak accelerator memory and energy if measured. Report both cold
and warm inference where caching matters, batch size, precision, thread count,
MPI ranks and convergence/iteration counts. A timeout or failed convergence is
a failure with its consumed resources, not a silently dropped sample.

## Repeats, aggregation and statistical inference

Use at least five independent seeds for every stochastic trainable method and
every stochastic dataset/noise realization. Deterministic solvers run on the
same five or more test/noise seeds. Report every seed, median, mean, standard
deviation, interquartile range and a 95% confidence interval.

Comparisons are paired: methods use the same model/noise instance. The primary
confidence interval for a method difference uses a hierarchical paired
bootstrap with at least 10,000 replicates:

1. sample geological families with replacement;
2. within each selected family, sample base models with replacement;
3. keep each model's paired method results and aggregate their difference.

Do not treat correlated noise/station variants of one base model as independent
models. For field studies, bootstrap held-out profiles/surveys first and stations
within profile second; a single region with five nearby profiles does not support
a universal field-generalization claim. Report the paired effect, interval and
number of independent families/profiles, not only a p-value. Multiple primary
metrics require a preregistered correction or a declared single primary metric.

## Reproducibility manifest

Each run directory contains a machine-readable manifest and immutable outputs.
The manifest must include:

- protocol and registry schema versions plus the exact registry file SHA-256;
- repository URL, full source SHA, resolved tag/version SHA and dirty-tree state;
- SHA-256, byte size and origin URL/DOI for every input, checkpoint and output;
- exact command, working directory, configuration and ordered random seeds;
- dataset split/family/sample IDs and the observation/geometry/mask contract;
- component, units, axes, rotation angle, time convention and phase convention;
- lockfile or complete package inventory, Python/compiler/MPI/CUDA versions and
  container image digest (a mutable image tag is insufficient);
- OS, CPU model/count, RAM, accelerator model/count/driver, storage class and
  thread/rank/precision settings;
- start/end timestamps, exit status, warnings, convergence trace and resource
  measurements;
- metric implementation ID and source SHA, individual predictions and bootstrap
  seed/replicate count.

Runs write to a new empty directory and publish atomically after validation.
Existing results are never overwritten. Any manual intervention creates a new
manifest with the intervention recorded. A leaderboard row links to the manifest,
not merely to a prose report.

### Executable manifest schemas and promotion states

The normative implementation is `pimsr_benchmarks.sota_manifests`, schema
version 1. It validates and recursively links four canonical JSON documents:

- `pimsr-sota-experiment` preregisters the exact method/dataset/track pair,
  pinned source, commands, five or more seeds, grouped
  family/base-model/noise/sample split IDs and the physical contract;
- `pimsr-sota-observations` exists only after dataset bytes are materialized
  and records their SHA-256 plus hashes for coordinates, spectral axis,
  observations, uncertainty, mask and (when available) truth;
- `pimsr-sota-predictions` binds output hashes to the same experiment,
  observation groups and physical contract;
- `pimsr-sota-run` records the executed argv, source snapshot, hashed inputs
  and outputs, environment, timestamps, exit/convergence state, runtime and
  measured resources.

Schema v1 provides byte-level provenance, not semantic completeness. In
particular, it does not open HDF5/NPZ payloads to prove array names, dtypes,
shapes and finite values; it does not bind every preregistered seed to a complete
campaign of runs and predictions; and it does not require the full hardware,
compiler, convergence-trace, evaluator and per-metric provenance listed above.
Consequently its highest valid promotion state is `adapter_smoke_passed`.
`benchmark_complete` is reserved for a future stronger schema and is rejected
fail-closed by both prediction and run validators in schema v1.

Every object has exact keys and types. Unknown fields, legacy or unknown schema
versions, uppercase/short hashes, duplicate JSON keys, non-finite numbers,
incompatible method/dataset/track combinations, mismatched grouped splits and
phase/component/axis/unit/time contracts fail validation. Manifest JSON is
canonical UTF-8 (sorted compact keys and one final newline). Publication is
atomic and no-overwrite; referenced files are content-addressed and rehashed
when a manifest is loaded.

Readiness is monotonic and never inferred from prose:

| Status | Meaning | Leaderboard eligibility |
| --- | --- | --- |
| `artifact_pinned` | Exact public source/artifact is selected; execution may not have occurred. | No |
| `adapter_smoke_passed` | A hashed adapter input/output completed successfully under the declared byte-level contract. An explicitly non-converged run cannot claim this status. This is the maximum schema-v1 state. | No |
| `benchmark_complete` | Reserved for a future schema that proves typed payload arrays, the complete seed campaign, runtime/resources and metric provenance. Schema v1 rejects this status. | No under schema v1. |

The registered `pimsr_generated_2d_v1` dataset is intentionally conditional.
Its schema-v2 generator commit, campaign size and physical contract are pinned,
but hidden generator seeds and source sample indices are not public method
inputs. Five campaign seeds and the opaque sample-ID key are fixed before any
run through public SHA-256 commitments. Their values, the source HDF5, truth,
geological families and source-to-opaque-ID mapping remain operator-only until
all compared predictions are immutable. A separate public observation manifest
contains no truth-derived labels or recoverable generator identity.

A dataset made from the previously preregistered public seed `20260823` is
development/smoke data only. Because the open deterministic generator can
reconstruct its truth from that seed and generator row index, it is ineligible
as a hidden test regardless of file permissions. Hidden materialization remains
`seed_committed_not_materialized` until public-observation and operator-scoring
artifacts are separately hashed; prediction promotion is rejected before then.

## Execution gates

Before publishing a row:

1. validate the registry and all input manifests;
2. run forward/coordinate/component unit tests and an analytic or cross-solver
   sanity case;
3. freeze split, observation, configuration and stopping manifests;
4. execute every method on the paired sample set without method-specific data
   changes;
5. validate output completeness, finiteness, masks and physical coordinates;
6. compute metrics once with the independent evaluator and run the paired
   hierarchical bootstrap;
7. archive manifests, logs and predictions, then generate the leaderboard.

If any gate fails, the row is `incomplete` and excluded from ranking. Unsupported
method geometry, missing weights, a license block, a timeout or unavailable truth
is reported explicitly rather than repaired by silently changing the protocol.

## Primary artifact references

The registry carries the exact machine-readable references. Human-readable
starting points include [RDON](https://github.com/zhangheng-1/RDON),
[MT2DInv-DenseNet](https://github.com/Geo-huang/MT2DInv-DenseNet),
[MTDLPy](https://github.com/Yuan-Chongxin/MTDLPy),
[GAN-MT1DInv](https://github.com/TuanfuGui/CG),
[physically guided 1D inversion](https://github.com/PAULGOYES/MT_guided1DInversion),
[MARE2DEM](https://bitbucket.org/mare2dem/mare2dem_source),
[SimPEG](https://github.com/simpeg/simpeg),
[ModEM](https://github.com/magnetotellurics/ModEM),
[MT3D_CNN](https://github.com/Jon-GSC/MT3D_CNN),
[GEMMIE](https://gitlab.com/m.kruglyakov/gemmie), and
[FEMTIC](https://github.com/yoshiya-usui/femtic), and
[DEVA3DMT](https://github.com/varilsuhad/DEVA3DMT). Public benchmark sources are
[COPROD2](https://www.mtnet.info/data/coprod2/coprod2.html),
[COPROD2S](https://www.mtnet.info/data/coprod2s/coprod2s.html), and the
[MT3DINV4 workshop](https://mt3dinv4.mtnet.info/).
