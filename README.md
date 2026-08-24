# pimsr-benchmarks

Benchmarks of the PIMSR physics-informed neural inversion
([pimsr-inversion](https://github.com/TheLitis/pimsr-inversion)) against
classical and independently runnable research methods, on synthetic data and
on **real USArray magnetotelluric stations**.

Part of the PIMSR platform:
[pimsr-geogen](https://github.com/TheLitis/pimsr-geogen) ·
[pimsr-forward](https://github.com/TheLitis/pimsr-forward) ·
[pimsr-inversion](https://github.com/TheLitis/pimsr-inversion) ·
pimsr-benchmarks

## Current benchmark status

The committed headline table below is a **legacy, non-comparable artifact**.
Its score used only `Zxy` (historically labelled TE), included periods outside
the measured EMTF band through edge extrapolation, and gave log-resistivity
twice the weight of phase. It must not be used to claim a current advantage.

The current diagnostic metric is
`section_nrms_2d_profile_rotated_tetm_masked_normalized_geometry_v3`: the full
geographic EMTF tensor is rotated into each fitted profile frame before
assigning TE=`Zyx` and TM=`Zxy`. It uses explicit in-band masks, no period
extrapolation, equal TE/TM and rho/phase weight, and 180-degree-periodic phase
residuals. The identifier deliberately discloses that 283--299 km field lines
are still affinely mapped to the 16 km synthetic station span. Such scores are
diagnostic only and are not publishable field comparisons. A headline also
requires an equal TE+TM observation budget for every method and a mesh on the
native physical station coordinates. No current headline is published until
those requirements are met and every method has been rerun.

The executable comparison plan is defined in
[`docs/SOTA_PROTOCOL.md`](docs/SOTA_PROTOCOL.md) and the fail-closed registry
[`config/sota_methods.json`](config/sota_methods.json). The first reproducible
wave covers GAN-MT1DInv and the physically guided unsupervised notebook in 1D;
RDON, MT2DInv-DenseNet, MTDLPy, MARE2DEM and SimPEG in 2D; plus ModEM,
MT3D_CNN, GEMMIE, FEMTIC and the public DEVA3DMT research edition in 3D.
Methods without independently runnable source/weights remain on a paper-only
shadow board. Frozen artifacts, common retraining and physics refinement are
separate tracks and are never ranked together.

Validate the installed immutable registry with `pimsr-sota-validate` (or pass
an alternate registry JSON path as its only argument).

The registry also pins PIMSR itself and a conditional
`pimsr_generated_2d_v1` generation definition. The latter is deliberately
`not_yet_materialized`: it has no registered dataset bytes or checksum and
cannot produce a leaderboard row. Execution provenance uses four strict,
versioned JSON schemas (experiment, observations, predictions and run). Schema
v1 is deliberately capped at `adapter_smoke_passed`: it hashes opaque payload
files but does not inspect and prove typed HDF5/NPZ arrays, every seed in a
complete campaign, or the full runtime/metric record required for
`benchmark_complete`. The validator therefore rejects that status fail-closed
until a stronger schema is implemented. A source pin or smoke test is never
reported as a completed comparison.

Use `pimsr-sota-manifest validate MANIFEST.json` to recursively validate
canonical JSON and every referenced file hash. `publish` writes a new manifest
atomically and refuses overwrites; `snapshot` creates a content-addressed input
copy:

```bash
pimsr-sota-manifest snapshot observations.h5 run \
  --media-type application/x-hdf5
pimsr-sota-manifest publish experiment.draft.json run/experiment.json
pimsr-sota-manifest validate run/experiment.json
```

The manifest command does not download or execute third-party software. It is
the fail-closed provenance layer used before adapters and benchmark runs are
added.

## Archived headline (legacy v1; not comparable)

| Profile | ModEM NLCG | Occam2DMT v3.0 | PIMSR U-Net (joint-ft) |
|---|---|---|---|
| G | 5.32 | 3.92 | **3.59** |
| H-YS | 5.90 | 4.68 | **4.10** |
| I | 10.98 | 9.26 | **5.62** |
| J | 6.28 | 6.40 | **3.49** |
| K | 6.99 | 6.03 | **4.69** |
| **mean** | 7.09 | 6.06 | **4.30** |

These values are retained only for provenance. The external diagnostic drivers now
write schema-v3 results with explicit geometry metadata; the table cannot
be updated without rerunning the neural checkpoints and external binaries.

## Baselines

| Method | Implementation | Notes |
|---|---|---|
| Occam2DMT v3.0 | official Scripps Fortran source | established smooth-inversion reference |
| ModEM 2D NLCG | official open source (github.com/magnetotellurics/ModEM) | the most-cited modern MT code |
| SimPEG 2D Gauss-Newton | `pimsr_benchmarks.hybrid2d` | our in-repo classical baseline |
| Occam-style 1D | `pimsr_benchmarks.occam1d` | Tikhonov GN, per-station |
| PIMSR neural (1D + 2D) | checkpoints from pimsr-inversion | single forward pass, amortised |

## Metrics

- `section_nrms_2d_profile_rotated_tetm_masked_normalized_geometry_v3` —
  profile-rotated, shift-invariant TE+TM data misfit through a 2D forward,
  restricted to measured periods. Static shift is removed separately per mode
  and station. It is explicitly a normalized-geometry diagnostic, not a
  physical-scale field leaderboard.
- RMSE of log10-resistivity vs ground truth (synthetic)
- Uncertainty calibration (1-sigma coverage), scenario classification accuracy
- Wall-clock time per profile

## Real data

27 USArray/EMTF transfer-function stations (Yellowstone box, 42.5-45.5N,
108.5-113W) from IRIS/EarthScope SPUD, committed under `data/emtf/`.
The parser reads all four geographic impedance components, verifies the EMTF
orientation/sign convention, fits each profile bearing, and rotates the tensor
into PIMSR's x-profile/y-strike/z-up frame before deriving local TE=Zyx and
TM=Zxy apparent-resistivity/phase curves. Unsupported periods are represented
as `NaN`; scoring never extrapolates them.

## Usage

```bash
pip install -e .
# synthetic benchmark against the held-out test split
python scripts/run_2d_bench.py --checkpoint best2d.pt --test-h5 ds2d_test.h5 \
    --emtf-dir data/emtf --out-dir results/my_run --n 500
# mixed-budget diagnostic only (fails closed without the explicit flag)
python scripts/run_unified_leaderboard.py --help
# external-code diagnostics
python scripts/run_occam2dmt.py --help
python scripts/run_modem2d.py --help
```

## License

MIT (code). Real EMTF data courtesy of IRIS/EarthScope, US National Science
Foundation.
