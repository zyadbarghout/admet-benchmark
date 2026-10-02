# Real CV, uncertainty and local prediction

Implemented 2026-09-09. The classical benchmark uses measured KSOL/BBBP data,
frozen outer folds and calibration roles. Historical tests are excluded from
training, calibration and this evaluation. It is a development experiment;
the previous historical tests have already appeared in reports.

## Run or resume

```sh
python -m admet_benchmark.cross_validation register configs/real_cv_uq.yaml \
  --output outputs/my_registered_cv
python -m admet_benchmark.cross_validation run outputs/my_registered_cv
python -m admet_benchmark.cross_validation status outputs/my_registered_cv
```

`admet-cv` provides the same commands after installing the updated package.
Registration requires a new directory; `run` resumes that registration and
checks completed artifact hashes. A changed registered input or runner version
fails validation. It never overwrites a completed model to hide an error.
`--max-jobs 3` provides a bounded integration/profile run. The local
`progress.html` reports planned, running/interrupted, completed and failed jobs;
refresh to read saved progress. It is a status view, not a background scheduler.

The new `protocol.json`/`registration.json` contract is separate from the original
`admet-study` repeated-holdout schema. The original commands remain compatible.
This version uses fixed model parameters, not nested hyperparameter search.
Five or ten outer folds, multiple declared seeds, molecule/scaffold splits and
Morgan/RDKit/combined features are configurable. All compared model families and
representations consume the same saved role assignments.

The first measured configuration has 5 folds, 1 repeat, 2 split schemes,
2 feature representations and 3 model families: 120 jobs / 440 estimators.
Linear models use one fit; each tree ensemble uses five separately seeded
150-tree estimators with minimum leaf size 2 and square-root feature sampling.
The two report representations are feature representations used for training;
the three display-map representations remain independently switchable.

## Data and calibration boundaries

Input endpoint sources contain SMILES, prepared evaluation-scale targets and
optionally original-scale targets. Configure the column names explicitly.
For `evaluation_scale: log10`, the adapter checks that the prepared target equals
log10 of the positive raw value; it does not transform the target again.
This initial producer expects prepared targets; it is not a general assay ETL.

Same-label canonical-isomeric replicates collapse to one representative. Conflicting
identities are quarantined in the new snapshot with an audit, preserving source
files. This leaves 5,128 KSOL and 1,569 BBBP development chemicals. Parent/salt/
tautomer standardization is not silently added. Endpoint source hashes, original
IDs, curation actions, scaffold groups and every role assignment are retained.

Each outer fold reserves approximately 20% of its training pool for conformal
calibration. Classification also reserves 20% of the remainder for probability
calibration. Realized scaffold fractions vary with group size. Fitting,
probability calibration, conformal calibration and outer validation are disjoint.
Every chemical is evaluated once per repeat in each split scheme. Scaffold
identity is disjoint across roles for scaffold experiments, including the common
empty scaffold group. Class/calibration support is checked before training.

The fit subset alone determines estimator imputation/scaling and model parameters.
Classical parameters are fixed in this release; no validation-based tuning,
threshold selection or epoch selection is performed. The historical-test audit
reads only structures to check overlap. Previously reported test labels do not
supply new calibration or a fresh confirmatory claim.

## Uncertainty semantics

Regression exports ensemble standard deviation and two calibrated intervals:
absolute-error split conformal and ensemble-normalized split conformal. The
normalization floor is fixed at 5% of the fitting-target standard deviation,
with a numerical minimum of 1e-6. Linear regression has no ensemble dispersion;
its score is the constant absolute-conformal radius and its normalized interval
reduces to the absolute baseline. Finite-sample corrected quantiles fail explicitly
when calibration support is insufficient for a finite interval.

Classification exports raw mean probability, sigmoid-calibrated probability,
raw ensemble standard deviation and predictive entropy. Entropy is the primary
UQ ranking score, not an estimate of epistemic variance. Sigmoid calibration uses
only the probability-calibration role. Class-conditional conformal scores use
the separate conformal role; sets retain `{0}`, `{1}`, `{0,1}` and empty outcomes.

Saved metrics include Brier/log loss, set and class-specific coverage, mean set
size, singleton rate, interval coverage/width/score and uncertainty/error rank
correlation. The common classification metrics now persist Brier/log loss for
prepared-prediction workflows too. Prediction intervals are distinct from
confidence intervals around evaluation metrics. This run does not report
confirmatory p-values or treat folds as independent experimental replicates.

Nominal coverage is 90%. Exchangeability is needed for ordinary conformal
coverage theory; scaffold shifts and selected chemical cohorts can break that
interpretation. All cohort-conditioned coverage here is descriptive. See the
[research proposal](proposals/cv_uq_2026_09_09.md) for primary sources and rationale.

## Report and chemical analysis

```sh
python scripts/prepare_cv_report_bundle.py outputs/my_registered_cv
admet-chemical-report outputs/my_registered_cv/report_data/KSOL \
  outputs/my_registered_cv/report_data/BBBP \
  --output outputs/my_registered_cv/chemical_model_diagnostics.html \
  --max-points 30000 --map-methods tsne gtm pca \
  --representations morgan rdkit_descriptors morgan_rdkit \
  --map-cache outputs/my_registered_cv/map_cache
```

The display bundle restores original names/source IDs from hash-verified sources;
its provenance records the relationship to evaluated predictions. Metrics,
predictions and training artifacts remain unchanged. Reports start with one
training representation and split scheme to make the five-fold comparison clear.

New scientific controls provide:

- Original Morgan nearest-fitting-neighbour similarity and scaffold novelty.
  These references belong to the individual fold, unlike the shared display atlas.
- Error and interval/set coverage versus similarity, with at least ten selected
  observations for a plotted coverage bin. Counts remain visible.
- Producer-labelled property-cliff stress cohorts. The delivered KSOL study uses
  similarity at least 0.8 and an observed gap at least 1 log10 unit against a
  fitting molecule. The labels use evaluation outcomes after prediction and cannot
  flag a new molecule before its outcome is measured.
- A property-cutoff decision view: below, above or crossing the cutoff according
  to the chosen interval. Equality counts as crossing. This cutoff is distinct
  from the existing residual/error bands and does not imply a probability of
  exceeding it. No universal assay cutoff is supplied.
- Classification prediction-set coverage and membership, and misclassification
  risk versus fraction retained. Regression interval methods can be switched.

View recipes are now version 3 and include the scientific cohort, interval method
and property cutoff. Cohort exports include interval/set and applicability fields.
The old reports retain their own recipe versions.

## Predict an input SMILES batch locally

```sh
python -m admet_benchmark.cross_validation predict \
  outputs/my_registered_cv/jobs/KSOL/molecule/r0_f0/rdkit_descriptors/random_forest \
  queries.csv --output predictions.csv --cutoff 1.0
```

`queries.csv` needs a `smiles` column. The optional cutoff is in evaluation units.
The output includes model hash, evaluation scale, nominal level, neighbour
reference and interval/set outputs. Log10 regression also returns monotonically
back-transformed raw endpoints. Only load trusted locally generated joblib model
bundles. This command uses the selected saved **outer-fold model**, whose fitting
and calibration populations are explicit; it is not an unreported full-data refit.
A final deployment-model fit/calibration policy remains a separate scientific step.

## CV+ pilot

```sh
python -m admet_benchmark.cvplus outputs/my_registered_cv \
  --output outputs/my_cvplus_pilot
```

This initial pilot targets KSOL/Morgan/molecule split, repeat 0. For each saved
outer fold it fits fresh inner K-fold random-forest point models on the complete
outer training pool, constructs residual/prediction candidates and evaluates only
on the untouched outer validation fold. It therefore never assesses CV+ coverage
on its calibration residual observations. `fit_cvplus`, `predict_cvplus` and
`cvplus_bounds` are reusable Python functions; quantiles are order statistics
with explicit finite-support checks and bounded temporary memory.

The resulting predictor differs from the split-conformal five-forest ensemble:
it averages inner-fold point models and uses a different allocation of training
data. Its empirical comparison is useful but does not isolate the interval method.
The nominal 90% quantile setting is not an exact worst-case 90% or conditional
coverage guarantee; CV+ theory has different bounds. No scaffold claim is made.

## Neural bridge

```sh
# Use the environment containing the optional models dependencies.
python -m admet_benchmark.neural_cv outputs/my_registered_cv \
  --output outputs/my_neural_smoke --endpoint BBBP --fold 0 \
  --model chemprop --epochs 2 --members 2
```

The adapter supports `chemprop` and `chemeleon`, consumes the exact registered
roles, trains only rows labelled `train` with fixed epochs, and calibrates and
predicts each held-out role separately. Commands, logs, role tables, checkpoints,
calibration and an addon manifest are saved. No old checkpoint is reused.

The delivered smoke test exercised Chemprop on one BBBP outer fold with two
members and two epochs. CheMeleon uses the shared adapter path but was not run;
a full neural CV benchmark, inner tuning and calibrated deployment refit remain
future work. Do not rank this short smoke test against tuned neural baselines.

## Evaluate the saved outcome

```sh
python scripts/evaluate_real_cv.py outputs/my_registered_cv
```

This independently verifies every completed artifact, exact outer-fold membership,
labels and one OOF row per molecule/method/split/repeat. It reloads 12 representative
checkpoint bundles and reproduces predictions/intervals/sets within recorded CSV
floating-point tolerances. It writes pooled descriptive outcomes and an audit.
See [the measured result](real_cv_uq_outcome_2026_09_09.md) for findings and limits.
