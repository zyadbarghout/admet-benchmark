# Evaluating Prepared Predictions

## Scope

`admet-evaluate` evaluates predictions that have already been generated. It does
not split data, tune hyperparameters, select a model, or train a model. This
separation lets predictions from notebooks, command-line programs, external
teams, and future training adapters use the same evaluation rules.

The evaluator performs two kinds of validation:

1. Structural checks verify required columns and numeric values.
2. Semantic checks detect duplicate prediction keys, inconsistent observed
   values, invalid probabilities, invalid log transforms, validation/test
   overlap, and incomplete molecule coverage between models.

## Input Shapes

Long-form input has one row per molecule and model:

```csv
sample_id,molecule_id,smiles,endpoint,model,partition,repeat,fold,observed,prediction
row-1,mol-1,CCO,KSOL,random_forest,validation,0,0,42.0,38.1
row-1,mol-1,CCO,KSOL,chemprop,validation,0,0,42.0,40.4
```

Map these columns with `predictions.columns`:

```yaml
schema_version: 1
project_name: Solubility comparison
task: regression
output_dir: outputs/solubility_evaluation

predictions:
  path: predictions.csv
  columns:
    sample_id: sample_id
    molecule_id: molecule_id
    smiles: smiles
    endpoint: endpoint
    model: model
    partition: partition
    observed: observed
    prediction: prediction
    uncertainty: uncertainty
    prediction_lower: prediction_lower
    prediction_upper: prediction_upper
    repeat: repeat
    fold: fold
  defaults:
    split_strategy: scaffold
    preprocessing: fragment_parent

target_transform:
  default: log10
  endpoints:
    LogD: none

uncertainty:
  higher_is_more_uncertain: true
  scale: evaluation
  interval_confidence: 0.9

statistics:
  bootstrap_iterations: 1000
  bootstrap_metrics: [mae, relative_absolute_error]
  confidence_level: 0.95
  seed: 42
```

Wide input has one prediction column per model. Replace the `model` and
`prediction` mappings with `prediction_columns`. Optional UQ columns can be
mapped per model in the same way:

```yaml
predictions:
  path: predictions.csv
  columns:
    sample_id: sample_id
    molecule_id: molecule_id
    smiles: smiles
    endpoint: endpoint
    partition: partition
    observed: label
  prediction_columns:
    random_forest: probability_rf
    chemprop: probability_chemprop
  uncertainty_columns:
    random_forest: uncertainty_rf
    chemprop: uncertainty_chemprop
  defaults:
    repeat: 0
    fold: 0
    split_strategy: provided_holdout
    preprocessing: raw
```

Paths declared in YAML are resolved relative to the YAML file. An explicit
CLI `--output-dir` override is resolved relative to the working directory. CSV, tab-separated, and Parquet
files are accepted. Parquet input requires a pandas-compatible Parquet engine.

`sample_id` identifies one measured row and must be unique within an endpoint,
partition, repeat, and fold. `molecule_id` identifies the chemical entity and
may be repeated when independent measurements exist. If `sample_id` is omitted,
the evaluator falls back to `molecule_id` and records a warning; this is safe
only when there is exactly one observation per molecule.

SMILES are parsed and canonicalised for auditing. Reuse of the same sample ID
between validation and test is an error. Exact canonical structures appearing
in both partitions are reported as a warning because they may represent either
intentional independent replicates or chemical leakage that must be reviewed in
the context of the dataset.

## Scale Contract

`observed` and `prediction` must be on the same submitted scale. The evaluator
preserves both columns and creates separate evaluation-scale columns.

- Use `target_transform.default: none` when values are already on the desired
  evaluation scale.
- Use `log10` when both submitted columns contain positive values on the
  original scale.
- Use `challenge` for OpenADMET-style regression: `LogD` remains unchanged and
  every other endpoint receives `log10`.

The evaluator rejects non-positive values before a log transform instead of
silently dropping molecules. It cannot determine automatically whether a file
has already been transformed.

## Uncertainty contract

`uncertainty` is a method-neutral ranking score: it must be finite,
non-negative, and larger must mean less certain. It is interpreted on the
evaluation scale and ranked independently within every model, partition,
repeat, and fold context. Coverage must be complete within such a context; a
partially populated uncertainty column is rejected.

`prediction_lower` and `prediction_upper` are optional paired interval bounds.
They use the submitted target scale and receive the same configured target
transform as observations and predictions. The evaluator reports:

- Spearman and Kendall association between uncertainty and realized absolute
  prediction error;
- empirical interval coverage;
- median interval width;
- molecule-bootstrap confidence intervals for these quantities.

These metrics test whether uncertainty tracks realized error. They do not prove
robustness under distribution shift. A benchmark claim should additionally
evaluate scaffold/feature shift, label shift, property-cliff or discontinuity
cohorts, temporal shift when genuine timestamps exist, and enriched error
cohorts.

## Classification Thresholds

Classification inputs must contain probabilities in `[0, 1]`. A threshold must
be supplied for every endpoint or as an explicit default:

```yaml
thresholds:
  endpoints:
    BBBP: 0.42
    ClinTox: 0.31
```

ROC AUC and PR AUC remain threshold-free. MCC, F1, accuracy, precision, recall,
specificity, and confusion counts use the declared threshold. Thresholds should
be chosen using training or validation data and then locked before final-test
evaluation. The threshold is stored as `decision_threshold` alongside each
fold's metrics; it is not treated as a performance metric.

## Commands

Check a file without writing an evaluation:

```bash
admet-evaluate validate evaluation.yaml
```

Validation exits with status `2` when semantic checks fail. Warnings, such as
unequal molecule coverage between models, do not stop evaluation.

Write the complete bundle:

```bash
admet-evaluate run evaluation.yaml
admet-evaluate run evaluation.yaml --output-dir outputs/alternative_location
```

## Evaluation Bundle

```text
output_dir/
  predictions.csv
  fold_metrics.csv
  metric_distributions.csv
  coverage_audit.csv
  statistical_comparisons.csv
  validation_report.json
  evaluation_manifest.json
  resolved_config.yaml
  report.html
```

`predictions.csv` is the canonical long-form record and contains submitted and
evaluation-scale values. `fold_metrics.csv` contains every fold or held-out
score with molecule-bootstrap confidence intervals for the configured
`bootstrap_metrics`. Defaults are MAE and RAE for regression and ROC AUC, PR AUC,
and MCC for classification. The report displays every repeat/fold point and
uses violin and embedded box layers as context; it deliberately does not draw
an average bar or mean line.

For multi-endpoint regression, complete model/repeat/fold contexts also receive
an `endpoint: __macro__`, `metric: ma_rae` row. This is the unweighted mean of
endpoint RAE values and matches the OpenADMET macro-averaging convention.

Pairwise comparisons use only matching repeat/fold records. With at least three
complete models, the evaluator reports a Friedman omnibus test. Pairwise
comparisons then use Conover–Friedman with Holm correction. A two-model
comparison uses the paired Wilcoxon signed-rank test. These are nominal,
exploratory p-values: repeated folds can share molecules and training data, so
independence is not established. CSV columns `inference_status` and
`inference_caveat` make this explicit. Holm correction does not fix dependence.
A single held-out test does not become replicated
evidence through molecule bootstrapping.

If model coverage differs within a fold, its descriptive metrics remain in the
report, but that fold is excluded from inferential model comparisons. This
avoids treating scores calculated from different molecules as paired evidence.

`evaluation_manifest.json` records SHA-256 hashes of all source files and the
software versions used. Keep the complete directory when sharing results.

## Validation and Final Test

Use `partition: validation` for predictions that were available during model or
threshold selection. Use `partition: test` only for a locked final evaluation.
The report lets users switch between them but never combines the two.

A sample ID cannot occur in validation and test within the same endpoint,
preprocessing profile, split strategy, repeat, and fold. For repeated
cross-validation, use explicit repeat and fold numbers and provide the same
fold assignments for every compared model.

## Chemical Analysis Boundary

Every row requires a molecule ID and SMILES so the canonical predictions can be
used by the chemical-space report. Interactive SMARTS selections are analysis
filters: they describe performance within a saved evaluation and do not change
the original experiment. A chemical cohort used for fitting or split creation
must be applied before predictions are generated and recorded as a distinct
preprocessing or cohort configuration.

Generate the chemical report directly from one or more completed evaluation
bundles:

```bash
admet-chemical-report \
  outputs/solubility_evaluation \
  outputs/permeability_evaluation \
  --output outputs/chemical_analysis.html \
  --smarts 'c1ccccc1'
```

The report exposes partition, repeat, and fold controls; actual fold values in
violin plots; molecule structures and SMARTS cohorts; uncertainty colouring;
an uncertainty-versus-error view; and an error-retention curve ordered from the
most certain to least certain chemistry.
