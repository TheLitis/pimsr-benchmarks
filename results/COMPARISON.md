# Archived PIMSR method comparison

> **Global legacy notice (2026-08-23):** this notice applies to **every**
> numerical cross-method table and comparison formerly published in this
> document, including 1D synthetic MT+gravity-vs-MT-only, 1D field, 2D neural,
> SimPEG, Occam2DMT and ModEM values.
>
> `comparison_status: diagnostic_non_comparable`
>
> `ranking_allowed: false`
>
> The archived numbers and their relative ordering are not a leaderboard or a
> state-of-the-art result. They must not support accuracy, speed, superiority,
> production-readiness or deployment claims. Earlier statements that PIMSR
> “beat”, “led” or “outperformed” another method are retracted as conclusions
> drawn without one common publishable benchmark contract.

## Scope of the historical measurements

The project evaluated development versions of PIMSR and classical baselines
on synthetic sections and USArray/EMTF profiles around Yellowstone. It also
catalogued recent neural MT-inversion publications. These activities remain
useful as engineering history and as inputs to the current executable
protocol, but they did not constitute a controlled state-of-the-art study.

The historical comparisons had several incompatible regimes:

- synthetic 1D PIMSR used MT plus gravity, while Occam used MT only;
- old 1D field scripts did not prove one shared nRMS error-floor contract;
- early 2D runs used the retired Zxy-only, unmasked metric;
- later 2D runs rotated and masked TE+TM, but normalized hundreds of
  kilometres of field geometry onto a 16 km synthetic span;
- some neural warm starts consumed TE+TM while their Gauss-Newton refiners or
  stitched 1D controls consumed TM only;
- external program runs and neural runs lacked a single complete immutable
  execution and hardware manifest.

Applying one score after inversion does not make unequal inverse observation
budgets or altered geometry comparable.

## What remains valid

The archive can support narrowly scoped statements about implementation and
experiment history:

- the four repositories contain generators, forward solvers, inversion
  models and benchmark adapters;
- real EMTF profiles were exercised quantitatively rather than only plotted;
- negative optimization outcomes were recorded alongside successful runs;
- raw historical outputs remain available in `results/` for provenance;
- recent literature and executable sources are registered in
  `config/sota_methods.json`.

It cannot support a numerical ordering of PIMSR, classical solvers or modern
published methods.

## Current route to a valid comparison

The executable rules are in `docs/SOTA_PROTOCOL.md`. A rankable result must,
at minimum, use:

1. immutable observation, prediction, model and runtime artifacts;
2. held-out data with leakage checks;
3. identical TE/TM/tipper/gravity observation budgets for compared methods;
4. native physical station coordinates and a disclosed mesh;
5. the same versioned scoring and uncertainty contract;
6. preregistered tuning, stopping and failure handling;
7. repeated seeds or an appropriate spatial/block uncertainty analysis.

Frozen third-party artifacts, common retraining and physics-refinement tracks
remain separate. Paper-reported numbers stay on a paper-only shadow board and
are not copied into an executable ranking.

Until those gates are satisfied and the registered methods are rerun, PIMSR
has no current numerical state-of-the-art claim. The detailed historical
narrative is recoverable from Git history; the working-tree document records
the scientifically valid interpretation of that archive.
