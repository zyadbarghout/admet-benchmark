# Using the ADMET Benchmark Pipeline and Reports

## Install

Create an isolated Python environment, then install the package and the model
and visual-report extras:

```bash
python3.12 -m venv .venv-models
. .venv-models/bin/activate
python -m pip install -e '.[models,visual,test]'
```

Python 3.11 or newer is required by Chemprop 2.3. On macOS the model extra
pins PyTorch below 2.11 because later releases currently reject Chemprop's
deterministic scatter-reduce operation on MPS.

The `models` extra enables Chemprop, XGBoost, and LightGBM. CheMeleon is used
through Chemprop's `--from-foundation CheMeleon` option, so it does not need a
separate model-package dependency. `visual` supplies the standalone Plotly
reports; no web server is required to use the generated HTML.

## Run a benchmark

Every benchmark is defined by YAML. Relative paths are resolved from the YAML
file, so a configuration can be moved with its data directory. Environment
variables may be used for paths that must remain local to a machine.

Run one model arena:

```bash
admet-benchmark-arena configs/benchmarks/ksol_model_arena.yaml
```

Run a suite of arenas:

```bash
PYTHONPATH=src python -m admet_benchmark.suite configs/benchmark_suite.yaml
```

The arena protocol is fixed for every model in a run:

1. Create or load the configured outer train/test split.
2. Reserve a shared validation subset from the outer training data.
3. Tune classical models by split-consistent cross-validation and neural
   models on a separate inner holdout, using only the development subset.
4. Select a model using the shared validation metric.
5. Refit each model on all outer training data and report its untouched final-test result.

The test columns in `model_comparison.csv` are descriptive comparisons. They
must not be used to select the reported model; `benchmark_metadata.json`
records the validation-based choice. The supplied arenas are screening runs:
their bootstrap intervals quantify molecule-sampling uncertainty, but a single
split does not support fold-paired significance claims. Use repeated, paired
CV predictions with `admet-evaluate` for confirmatory model comparisons.

Configure the shared validation and neural inner partitions explicitly. The
same split family should normally be used for both:

```yaml
benchmark:
  validation:
    strategy: scaffold
    size: 0.2
  neural:
    inner_validation_size: 0.15
    inner_split:
      strategy: scaffold
```

Classical feature representations are `morgan`, `rdkit_descriptors`, and
`morgan_rdkit`. Full RDKit descriptors are imputed inside each CV fold; Morgan
bits are not standardized.

```yaml
features:
  representation: rdkit_descriptors
  descriptor_set: full
```

## Configure data splits

Set `dataset.split` in an arena configuration. A CSV dataset defaults to a
seeded random split when no strategy is supplied. OpenADMET uses its official
predefined split by default; specifying another strategy intentionally creates
a fresh endpoint-specific split from the combined labelled data.

```yaml
dataset:
  kind: csv
  path: ../../data/example.csv
  smiles_column: smiles
  target: measured_value
  split:
    strategy: scaffold
    test_size: 0.2
    group_split_attempts: 128
```

Available outer split strategies are:

| Strategy | Behaviour | Important settings |
| --- | --- | --- |
| `random` | Seeded random holdout. Binary tasks are stratified. | `test_size` |
| `scaffold` | Entire RDKit Bemis-Murcko scaffold groups are assigned to either train or test. | `test_size`, `group_split_attempts` |
| `butina` | Entire Butina clusters based on Morgan-fingerprint Tanimoto distance are assigned together. | `test_size`, `butina_radius`, `butina_n_bits`, `butina_distance_threshold`, `group_split_attempts` |
| `temporal` | Earlier timestamps train the model and later timestamps form the final test. Equal timestamps are never split. | `test_size`, required `time_column` |

Temporal evaluation is deliberately unavailable without a real measurement,
registration, or publication timestamp. It is not valid to infer time from CSV
row order, molecule IDs, or shuffled dataset indices. The bundled OpenADMET,
BBBP, and ClinTox inputs do not provide a suitable time column, so the supplied
split suite covers random, scaffold, and Butina only.

For a temporal dataset, configure the source column explicitly:

```yaml
dataset:
  split:
    strategy: temporal
    time_column: measured_at
    test_size: 0.2
```

Each run writes `data/split_manifest.json`. Check its group-overlap count for
chemical splits, or `train_time_max` and `test_time_min` for a temporal split.
The independent validation policy is recorded in
`data/validation_split_manifest.json`.

## Run a split matrix

`configs/split_comparison_suite.yaml` performs independent HPO, validation,
training, and final testing for random, scaffold, and Butina splits of KSOL and
BBBP:

```bash
PYTHONPATH=src python -m admet_benchmark.suite configs/split_comparison_suite.yaml
```

The suite output has one directory per endpoint and split, plus:

```text
outputs/split_benchmark_suite/
  suite_summary.csv
  cross_split_model_comparison.csv
  openadmet_ksol__random/
  openadmet_ksol__scaffold/
  openadmet_ksol__butina/
  bbbp_permeability__random/
  bbbp_permeability__scaffold/
  bbbp_permeability__butina/
```

`cross_split_model_comparison.csv` is the concise comparison input: every row
is one model evaluated for one endpoint and one split strategy. Do not average
raw MAE values across chemically different endpoints. Compare a fixed endpoint
and metric, or use within-split ranks in the suite report.

## Generate and use reports

Each model arena automatically writes a validated evaluation bundle under
`evaluation/`; open `evaluation/report.html` for bootstrap intervals and
validation/test comparisons. `evaluation_predictions.csv` and
`evaluation.yaml` can be passed back to `admet-evaluate` without reading model
checkpoint directories.

For the full chemical review, combine one or more completed arenas into a
single report:

```bash
admet-chemical-report \
  outputs/current_benchmark/ksol \
  outputs/current_benchmark/bbbp_50epoch \
  --output outputs/current_benchmark/chemical_model_diagnostics_v2.html
```

This report embeds RDKit molecule drawings directly and creates one t-SNE map
over the union of displayed canonical molecules. Endpoint checkboxes can show
several endpoints together, while the diagnostics endpoint chooses the task
whose axes and metrics are displayed. It opens on validation and all recorded
endpoint contexts. Exact preprocessing and split choices are offered for the
active diagnostics endpoint, avoiding an empty diagnostic state when endpoints
use different legitimate split labels. For classification, residual means
`predicted positive-class probability - observed class`; absolute error is its
absolute value. The report includes probability-versus-class, calibration,
ROC, and precision-recall views. Regression endpoints include observed versus
predicted, residual versus predicted, train/test/prediction density, and
residual density views.

Box and lasso selections update the molecule count, median error summary, residual
distribution, endpoint diagnostics, and best-molecule cards without disabling
hover. Multi-endpoint molecule rankings require evaluation coverage in every
selected endpoint and use median endpoint-normalised absolute errors. The
normalisation divisor is the endpoint/model 95th-percentile absolute error;
signed residual and absolute-error colour limits use the same percentile rule.
The complete model table and bootstrap confidence intervals remain available
in a collapsed section below the chemical plots, alongside collapsed protocol
and portable source signatures. Canonical evaluator directories can be passed
to the same command; for these, validation/test, repeat, and fold controls are
retained and uncertainty-enabled runs add uncertainty/error and error-retention
views. The Uncertainty navigation item is hidden when the current model has no
uncertainty data.

If the report must sample records to honor `--max-points`, it samples canonical
molecules rather than isolated rows. Every retained molecule therefore keeps
its complete endpoint/repeat/fold history. This can leave a little unused
capacity and makes the limit a soft ceiling for unusually large single-molecule
histories. A self-contained report can still become large because it embeds
Plotly and deduplicated structure images; use the subtitle to disclose sampling
and do not interpret a sampled map as the complete dataset.

Generate a molecule-level interactive review for a completed arena:

```bash
PYTHONPATH=src python -m admet_benchmark.dashboard \
  outputs/model_benchmark_suite/bbbp_permeability \
  --smarts 'c1ccccc1' \
  --smarts 'N'
```

The resulting `interactive_report.html` is self-contained apart from the
structure-image assets stored beside it. It reads saved predictions and never
trains or recalculates metrics. In the report:

1. Select a model and whether points are coloured by signed residual or
   absolute error. Both modes use the same red-blue colour family, and the
   colour magnitude is clipped at the 95th percentile of visible evaluation
   points.
2. Choose a supplied SMARTS filter. SMARTS matching is calculated by RDKit
   during report generation, keeping the browser report fast and reproducible.
3. Hover a validation or test point to inspect its molecule structure, SMILES,
   observed value, prediction, residual, and absolute error.
4. Box- or lasso-select points to update selection count, mean absolute error,
   mean residual, and residual distribution. Hover remains available after
   selection.
5. Hide a trace from the legend if needed. Its visibility is retained when
   model, metric, or SMARTS controls redraw the plot.

Generate the suite-level comparison report after a suite completes:

```bash
PYTHONPATH=src python -m admet_benchmark.dashboard outputs/split_benchmark_suite
```

`interactive_suite_report.html` groups each model's saved result by split
strategy, lets the reader choose a task/endpoint and metric, and calculates a
within-split rank summary. For error metrics lower is better; for AUC, MCC, F1,
and related scores higher is better. Use the report to identify robustness
across split strategies, then inspect the corresponding arena's molecule-level
report to understand chemical-space failures.

## Chemical filtering

Training configurations can also apply a SMARTS filter before fitting:

```yaml
chemical_filter:
  smarts: c1ccccc1
  substitution: substituted
```

`substituted` requires an external heavy-atom attachment to a SMARTS match;
`unsubstituted` requires none. The run's filter-audit CSV records each match,
external attachment count, and retain/remove decision. Fused-ring attachments
count as external attachments. If fused, linked, and pendant substitutions need
to be distinguished, add an explicit new attachment category rather than
silently redefining this rule.

## Reproducibility checklist

- Commit the YAML configuration used for every reported run.
- Retain `resolved_config.yaml`, `data/split_manifest.json`,
  `feature_schema.json`, HPO candidate tables, and prediction CSVs with the
  report.
- Use only validation results for model selection.
- Report the split strategy with the final metric, especially for chemical
  generalisation claims.
- Treat t-SNE as a local diagnostic visualization, not as a quantitative proof
  of chemical coverage or model performance.
