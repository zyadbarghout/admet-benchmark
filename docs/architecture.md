# Chemistry-aware ADMET evaluation architecture

## Purpose

The public boundary is prepared predictions rather than trained estimators.
This makes the package useful for any molecular model while preserving the
OpenADMET training and report work as one producer and historical benchmark.
The repository has one reproducible evaluation backbone and several producers
and consumers of its outputs:

- `admet-evaluate` validates externally prepared predictions and writes the
  canonical evaluation bundle. It is the primary user-facing command.

- `admet_benchmark` performs data preparation, SMARTS filtering, feature
  construction, validation, HPO, model selection, and final test evaluation.
- `admet-chemical-report` consumes one or more canonical bundles or completed
  model-arena runs and provides structure-, SMARTS-, endpoint-, partition-,
  repeat-, and fold-aware exploration. It never retrains or selects a model.
- `admet_benchmark.dashboard` remains a compatibility consumer for the older
  arena and split-suite views.
- `legacy/openadmet_report` is the immutable local archive of the original
  leaderboard/submission/report workspace. It is not a package dependency.

The training commands are example prediction producers. They write the same
prepared-prediction contract directly and invoke the evaluator, but external
predictions do not need to use the training code.

## Canonical prediction contract

Every evaluated row is identified by task, endpoint, preprocessing, split
strategy, partition, repeat, fold, model, and sample ID. Molecule ID and SMILES
bind the measurement to its chemistry. The same sample context must resolve to
the same observed value and canonical structure for every model.

Optional uncertainty fields are intentionally method-neutral:

- `uncertainty`: a non-negative ranking score on the evaluation scale, where a
  larger value means less certainty;
- `prediction_lower` and `prediction_upper`: paired prediction-interval bounds.

The evaluator preserves submitted and evaluation-scale values, and ranks
uncertainty within each model/fold context. Method metadata and future
distributional predictions can extend this versioned contract without tying
the report to one UQ algorithm.

## Training producer contract

Training runs are defined by YAML. They specify dataset, target, representation,
optional SMARTS filter, candidate models, HPO budget, scoring metric, random
seed, split strategy, and output directory. Paths resolve relative to YAML.

The outer evaluation split is also configuration: `random`, `scaffold`,
`butina`, or `temporal`. Chemical strategies keep the relevant group in one
partition, while temporal splits require an explicit real time column. The
split manifest records the exact strategy, requested and actual holdout size,
class counts, chemical-group overlap, or temporal boundaries.

For regression, `target_transform: challenge` makes the log-scale policy
explicit: all OpenADMET endpoints are transformed with log10 except `LogD`.
The transformed values are used consistently for fitting, cross-validation,
and test metrics; predictions are also exported in the original scale.

For classification, the reusable components from the classification project
are represented in the output contract: ROC AUC, PR AUC, MCC, F1, confusion
counts, class balance, ROC/PR diagnostic figure, and a standalone HTML report.

## Chemistry Filter Contract

The filter accepts any RDKit SMARTS. `substitution: substituted` means a SMARTS
match has one or more external heavy-atom attachments, while `unsubstituted`
means it has none. The generated `*_filter_audit.csv` reports the match count,
external attachment count, and decision for every usable molecule. This avoids
an ambiguous chemical filter hidden inside notebook cells.

This definition intentionally includes fused-ring attachments. If the research
question needs to distinguish fused, linked, and pendant substitutions, add a
new explicit attachment-class field rather than changing the current meaning.

## Evaluation bundle

```text
outputs/<experiment>/
  resolved_config.yaml
  predictions.csv
  fold_metrics.csv
  metric_distributions.csv
  coverage_audit.csv
  statistical_comparisons.csv
  validation_report.json
  evaluation_manifest.json
  report.html
```

`predictions.csv` is the lossless analysis table. `fold_metrics.csv` keeps each
repeat/fold result; `metric_distributions.csv` is descriptive convenience, not
the primary evidence. Model comparisons are paired by repeat/fold and exclude
contexts with unequal molecule coverage. Bootstrap intervals resample molecule
IDs and retain all observations for each sampled chemical entity.

## Report boundary

The static evaluation report compares models with raw repeat/fold points and
violin distributions. The chemical report joins those results to structures,
embeddings, SMARTS cohorts, molecule selections, errors, intervals, and
uncertainty rankings. Report filters are descriptive views over saved records;
they cannot feed back into model choice or threshold tuning. Validation is the
initial view, complete tables and provenance use progressive disclosure, and
interactive downsampling retains whole canonical-molecule histories rather
than slicing arbitrary folds. Embedded source provenance uses artifact
signatures and run names instead of absolute local filesystem paths.

See [product vision and analysis principles](product_vision.md) for the staged
roadmap and research-design requirements.
