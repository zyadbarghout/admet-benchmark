# Product vision and analysis principles

## The product

This project is a chemistry-aware error analysis layer for molecular machine
learning. A cheminformatician supplies predictions from any regression or
classification model. The package validates the experiment context, retains
every observation and fold, quantifies model performance and uncertainty, and
connects failures back to molecules and chemical subspaces.

It is not tied to OpenADMET training code. OpenADMET is the preserved origin,
reference dataset, and first benchmark case.

## Non-negotiable analysis principles

1. **Keep the observations.** Report repeat/fold values and molecule-level
   errors directly. Violin, KDE, box, and point layers are preferred to bars of
   averages. Medians and quantiles may describe a distribution but never
   replace it.
2. **Keep experiment context explicit.** Task, endpoint, model, preprocessing,
   split strategy, partition, repeat, fold, sample ID, molecule ID, and SMILES
   form the canonical context. Validation and locked test results are never
   silently pooled.
3. **Pair model comparisons.** Compare models only where the same fold and
   molecule coverage are available. Report effect direction and magnitude,
   repeated-result distributions, multiplicity-adjusted tests, and practical
   relevance—not only a p-value or rank.
4. **Treat chemistry as a first-class axis.** Users must be able to find a
   molecule, filter with SMARTS, select a region of an embedding, inspect
   structures, and see how error and uncertainty change in that chemical
   cohort across endpoints and models.
5. **Evaluate uncertainty against error.** An uncertainty score is useful when
   larger values identify molecules with larger realized errors. Ranking
   correlations and error-retention curves are the minimum. Prediction
   intervals additionally require coverage and width. Future benchmark suites
   must include distribution shift and property-cliff cohorts.
6. **Separate exploration from claims.** Interactive filtering explains an
   already-saved experiment. It must not alter splits, tune thresholds, select
   models, or turn the locked test set into a development set.
7. **Preserve provenance.** Source hashes, resolved configuration, validation
   audit, canonical predictions, fold metrics, comparisons, and software
   versions travel together as one evaluation bundle.

## Current foundation

- Stable long- and wide-form prediction ingestion for regression and
  classification.
- Semantic validation of folds, partitions, chemical identity, transforms,
  probabilities, thresholds, model coverage, uncertainty, and intervals.
- Molecule-level bootstrap intervals and paired repeated-fold comparisons.
- Distribution-first evaluation report with every fold point.
- Multi-endpoint chemical-space report with structures, SMARTS filters, lasso
  selection, validation/test and repeat/fold controls, task-specific
  diagnostics, uncertainty colouring, error-ranking scatter, and an
  error-retention curve. Its grouped controls, jump navigation, validation-first
  default, and collapsed raw/provenance sections use progressive disclosure
  rather than presenting every table at once.
- Molecule-coherent report sampling that retains complete endpoint/fold
  histories for every displayed canonical structure, plus portable hashed
  source-run signatures that do not expose absolute host paths.
- The original OpenADMET report workspace and generated reports preserved under
  `legacy/` and `outputs/current_benchmark/`.

## Work still required before a research benchmark claim

- Repeated outer-fold generation is implemented. Add nested tuning under a
  declared compute budget and untouched outer evaluation.
- Conformal intervals/sets, proper probability scores and descriptive chemical
  coverage are implemented. Extend calibration assessment and uncertainty around
  cohort effects under a declared inferential protocol.
- Define shift benchmarks: scaffold/feature shift, target-label shift,
  activity/property cliffs, temporal shift when real timestamps exist, and
  deliberately enriched error cohorts.
- Add chemical cohort comparison tables with minimum-support rules and
  uncertainty around cohort effects; do not mine and report the same cohort
  without correction or external validation.
- Add experiment indexing and an explicit benchmark protocol that locks data,
  seeds, folds, preprocessing, metrics, and practical-effect thresholds before
  running final comparisons.
- Offline external assets, on-demand maps/structures and lazy plotting are
  implemented. Measure further scaling before expanding endpoint/repeat counts.
