# Complete real-data example

Open [`outputs/complete_real_example_2026_09_29/index.html`](../outputs/complete_real_example_2026_09_29/index.html)
after building. The landing page links the full interactive report, measured
results, every final predictor, inference examples and verification artifacts.

## What is trained

The frozen configuration is [`configs/complete_real_example.yaml`](../configs/complete_real_example.yaml).
All labels are measured real-data outcomes:

| Endpoint | Curated chemicals | Task | Evaluation scale |
| --- | ---: | --- | --- |
| KSOL | 5,128 | Solubility regression | log10 source assay units |
| LogD | 4,934 | Lipophilicity regression | source LogD values |
| BBBP | 1,569 | Brain-permeability classification | binary label / predicted probability |

KSOL and LogD share real chemical identities, enabling multi-endpoint exploration.
LogD is restricted to measured rows in the preserved KSOL development population.
Registration audits duplicate identities and excludes historical test structures.
Historical test labels are not included in the training snapshots.

- Five outer folds, two seeds, molecule/scaffold splits and Morgan/RDKit features.
- Linear, random-forest and ExtraTrees families: 360 jobs, 1,320 fitted estimators.
- Tree ensembles have five independently seeded 150-tree members; linear models
  use one fitted estimator. All parameters are fixed before evaluation.
- A BBBP neural addon: five molecule folds at the first seed, two Chemprop
  members per fold, 50 epochs/member and hidden dimensions of 300. It uses the
  same registered fitting/probability-calibration/conformal-calibration/validation
  roles as the classical comparison. It is not a two-epoch smoke run.
- An independently outer-evaluated KSOL CV+ pilot with five inner models per
  outer fold. Its point learner/training allocation differs from split conformal.
- Eighteen final calibrated predictors: each endpoint × two representations ×
  three classical families, fitted using all development chemicals assigned to
  fitting or reserved calibration. They are separate from outer-fold models.

“Complete” describes finished registered training budgets, saved artifacts and
working inference. It does not assert optimal hyperparameters, prospective
performance, or that neural/shift coverage is guaranteed.

## Build or resume

Use Python 3.12 with the models, visual and test dependencies. In this workspace:

```sh
.venv-models/bin/python scripts/build_complete_example.py
```

For a new environment and a fresh training run:

```sh
uv sync --locked --python 3.12 --extra models --extra visual --extra test --extra browser
uv run python scripts/build_complete_example.py --output outputs/my_complete_example
```

Runtime versions are enforced. To load the supplied trained checkpoints, recreate
their exact recorded environment, including NumPy, instead of assuming a newly
resolved environment is identical:

```sh
uv venv .venv-trained --python 3.12
uv pip install --python .venv-trained/bin/python -r configs/trained_example_requirements.txt -e .
```

The study also saves `runtime-requirements.txt` and runtime/source/lock fingerprints.
Changing the implementation or runtime requires a fresh registration. Historical
registrations are preserved; do not rewrite their hashes to bypass compatibility.

`--validate-only` registers/checks the data and split roles without fitting.
`--classical-only` explicitly builds without the neural addon. The default includes
it. Completed classical jobs are checked and resumed; a partially failed neural
addon directory requires inspection before rerunning in a fresh directory.

## Explore the report

1. Open the landing page, then **Explore the full chemical report**. The report
   works offline from `file://`; keep its sibling `.assets` directory with it.
2. Compare KSOL molecule and scaffold splits for the same model and model features.
   Inspect all repeat/fold points; switch to fold facets for shared coordinates.
3. Select novel scaffolds, distant training neighbours or property-cliff stress
   observations. Inspect large errors and their molecular structures.
4. Compare normalized and absolute conformal interval coverage/width. Enter an
   assay-specific property cutoff to distinguish above/below/crossing intervals.
5. Keep KSOL and LogD selected to explore shared chemistry. Switch map features
   independently from model features; all nine t-SNE/GTM/PCA combinations exist.
6. On BBBP, examine probability distributions, ROC/PR, confusion, calibrated sets
   and retained-risk curves. The neural results use **graph** model features and
   **chemprop**, with molecule split / repeat 0 only.
7. Save a view, reload it, and export a selected cohort. Full-fold metrics retain
   their original population while cohort diagnostics follow the filters.

The assets report uses compact records, external molecular depictions, on-demand
maps and lazy plotting. No molecules or folds are discarded by the example's
report generation. Single-file delivery remains available with `--delivery inline`.

## Predict using complete fitted models

From the repository root:

```sh
.venv-models/bin/python -m admet_benchmark.cross_validation predict \
  outputs/complete_real_example_2026_09_29/deployment/KSOL/rdkit_descriptors/random_forest \
  outputs/complete_real_example_2026_09_29/inference/queries.csv \
  --output outputs/my_solubility_predictions.csv
```

The query file contains five real structures with names and preserved identifiers.
Prediction output includes intervals or class sets, training-neighbour similarity,
model hash and role, evaluation scale and nominal coverage. KSOL additionally
exports back-transformed raw predictions and interval bounds. These query examples
do not include fabricated ground-truth labels.

`admet-cv refit STUDY --output NEW_DIRECTORY` reproduces the final fitting policy.
No model is selected using historical test labels. Regression reserves 20% of
development molecules for conformal calibration. Classification additionally
reserves 20% of the remainder for probability calibration.

The example also reloads a neural ensemble with `predict_neural` from
`admet_benchmark.neural_cv`, using its saved probability calibrator and conformal
calibration. It writes `inference/BBBP_chemprop.csv` and preserves its outer-fold
role in output metadata.

## Evidence and limits

`outcome_audit.json` verifies exact OOF membership and labels, checks every job's
artifact hashes and reloads representative classical checkpoints. Neural OOF
membership, checkpoint hashes and batch inference are also checked during build.
`pooled_oof_outcomes.csv` contains descriptive per-repeat results; final refits do
not inherit a claim of independent prospective validation from those CV results.

The example covers real overlapping endpoints, classification/regression,
classical/neural models, calibration, chemical stress, map exploration, reproducible
views and deployment-style inference. Temporal/external prospective validation,
nested HPO, validated CheMeleon results, confirmatory cohort inference and dataset
publication rights remain separate work. None is simulated or implied by the demo.

The completed neural run achieved BBBP PR AUC 0.8957 and specificity 34.67%,
compared with PR AUC 0.9651 for the matching first-seed RDKit random forest.
Its class-set coverage was 89.61%. The report retains these measured weaknesses;
finished training does not imply a competitive or prospectively validated model.

To repeat the full-example browser acceptance check after building:

```sh
.venv-models/bin/python scripts/check_complete_example.py \
  outputs/complete_real_example_2026_09_29
```

It writes screenshots and `verification/browser_audit.json`, exercising the
unsampled report, nine maps, chemical cohorts, interval/cutoff controls, view
round-trips, exports, neural results and mobile layout. See the
[completion report](audit_resolution_2026_10_02.md) for fixed defects and measured
verification evidence.
