# Chemistry-aware ADMET model diagnostics

This project helps cheminformaticians understand **where a regression or
classification model fails chemically**, whether its uncertainty identifies
those failures, and how competing models behave across endpoints, validation
folds, and a locked test set.

The primary input is a table of prepared predictions—not a trained estimator.
Models from Python, notebooks, external teams, or another framework therefore
use the same validation, statistics, and chemical analysis. The original
OpenADMET leaderboard and report project is preserved under
[`legacy/openadmet_report`](legacy/openadmet_report/README.md); the newer
package does not overwrite it.

## What works now

- Regression and classification with explicit scale/threshold policies.
- Long-form or wide prediction files from multiple models and endpoints.
- Validation/test, repeat/fold, preprocessing, and split-strategy context.
- Chemical-identity, leakage, coverage, and schema audits.
- Molecule-bootstrap intervals and paired fold summaries; nominal test p-values
  are exploratory because fold dependence is not resolved.
- Method-neutral uncertainty scores and prediction intervals, including
  error-ranking correlations, empirical coverage, and interval width.
- Interactive chemical-space inspection with structures, SMARTS filters,
  molecule search, lasso selection, error/uncertainty colouring, fold controls,
  task-specific diagnostics, and error-retention curves.
- Violin/KDE/point views that retain the actual observations rather than
  replacing them with average bars.

The research principles and remaining work are explicit in
[`docs/product_vision.md`](docs/product_vision.md). The evidence, risk, and
implementation audit is maintained in
[`docs/project_assessment.md`](docs/project_assessment.md). For the current
project status, ordered roadmap, integration recommendations, and open
decisions, see [`PROJECT_STATUS.md`](PROJECT_STATUS.md).

## Complete real-data example and audit fixes

The [complete example](docs/complete_real_example.md) trains KSOL, LogD and BBBP
on registered five-fold/two-seed molecule and scaffold splits, adds full-budget
BBBP Chemprop CV and a CV+ pilot, and creates final calibrated predictors.
Its [local landing page](outputs/complete_real_example_2026_09_29/index.html)
links the full report, trained models, measured outcomes and batch inference.

```sh
python scripts/build_complete_example.py --output outputs/my_complete_example
```

The default requires the `models` and `visual` extras and runs real training.
Use `--validate-only` to check the protocol or `--classical-only` to explicitly
exclude the neural addon. The guide distinguishes a new locked-environment run
from reuse of the supplied checkpoints in their exact recorded environment.

The September 28 audit defects have regression coverage: text identities are
preserved, canonical chemistry determines bootstrap units, fitting and locked-test
leakage checks are stricter, classification plots share threshold policy, and
registered runs/checkpoints enforce runtime and feature compatibility. Reports
also support `--delivery assets` for on-demand local maps and depictions.
The [completion report](docs/audit_resolution_2026_10_02.md) records the fixes,
measured verification and remaining work.

## Earlier September 9 cross-validation study

The earlier measured development benchmark completed 120 registered jobs on KSOL
and BBBP: five outer folds, molecule/scaffold splits, Morgan/RDKit features and
three classical model families. It produces disjoint-calibration regression
intervals and classification prediction sets, with artifact hashes and resumable
training. The report adds chemical applicability, property-cliff stress cohorts
and interval decisions around an assay cutoff. A local SMILES prediction command,
CV+ pilot and short Chemprop integration run are included.

Open the [real CV report](outputs/real_cv_uq_2026_09_09/chemical_model_diagnostics.html),
read the [measured findings](docs/real_cv_uq_outcome_2026_09_09.md), or follow the
[run/resume/predict guide](docs/real_cv_uq.md). Overall KSOL interval coverage is
near 90%, but the RDKit forest's scaffold cliff cohort reaches only 68%:
chemical stress diagnostics change the interpretation of the aggregate result.

## Earlier workspace refinement

- Register a development study with `admet-study preflight`, `freeze` and
  `verify`; save shared molecular partitions before fitting models.
- Inspect context/model coverage, save and restore views tied to their report, export
  selected cohorts, and inspect structures with the keyboard.
- Switch Morgan, RDKit or combined representations across t-SNE, GTM and PCA.
- Work in separate regression/classification tabs, or opt into merged exploration.
- Search all included molecule IDs/SMILES with paginated inspection; define
  regression error bands and inspect clickable GTM node pies.
- Compare fold contexts and submitted UQ, including a synthetic 5/10-fold demo.
- Reproduce dependencies with `uv.lock` and run offline browser acceptance tests.

Start with the [five-minute tutorial](docs/five_minute_tutorial.md), then read
the [study contract](docs/study_registration.md) and the
[research and delivery plan](docs/research_and_delivery_plan.md).
The [workspace method and research notes](docs/chemical_workspace_v3.md) explain
GTM, thresholds, UQ and the new reports. The original
[verification record](docs/verification_2026_09_08.md) and latest
[workspace verification](docs/verification_workspace_v3.md) record the checks.
The original `admet-study` registry supports repeated holdouts. The new
`admet-cv` contract drives true outer-fold training with fixed parameters; neither
contract by itself certifies confirmatory inference.

## Install

Python 3.11 or newer is required; Python 3.12 is the clean-environment test target.
The reproducible installation uses `uv sync --locked --python 3.12 --extra visual --extra test`.
Standard pip packaging is also supported:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[visual,test]'
```

The optional training adapters for Chemprop, CheMeleon, XGBoost, and LightGBM
are installed with `python -m pip install -e '.[models,visual,test]'`.

## Evaluate predictions

Map columns and experiment metadata in YAML, validate without writing results,
then produce the evaluation bundle:

```bash
admet-evaluate validate configs/evaluation_regression_example.yaml
admet-evaluate run configs/evaluation_regression_example.yaml
```

For a long-form file, each row represents one model prediction for one measured
sample. The core mappings are:

```yaml
schema_version: 1
task: regression
output_dir: outputs/my_evaluation

predictions:
  path: predictions.csv
  columns:
    sample_id: sample_id
    molecule_id: molecule_id
    smiles: smiles
    endpoint: endpoint
    model: model
    partition: partition
    repeat: repeat
    fold: fold
    observed: observed
    prediction: prediction
    uncertainty: uncertainty
    prediction_lower: prediction_lower
    prediction_upper: prediction_upper
  defaults:
    split_strategy: scaffold
    preprocessing: fragment_parent

target_transform:
  default: log10

uncertainty:
  higher_is_more_uncertain: true
  scale: evaluation
  interval_confidence: 0.9
```

Uncertainty is optional. When supplied, it must be complete within a model/fold
context and larger values must mean less certainty. Interval bounds are also
optional but must be supplied as a valid pair.

See [`docs/evaluation_cli.md`](docs/evaluation_cli.md) for wide-form mapping,
classification thresholds, exact validation rules, output files, and
statistical comparisons.

## Inspect performance chemically

The chemical report accepts canonical evaluator output directories directly,
or the older completed model-arena directories:

```bash
admet-chemical-report \
  outputs/evaluation_a \
  outputs/evaluation_b \
  --output outputs/chemical_analysis.html \
  --smarts 'c1ccccc1'
```

Use the partition, repeat, and fold controls to separate validation behavior
from locked-test behavior. Select a chemical region or SMARTS motif to update
errors, distributions, structures, uncertainty diagnostics, and shared-molecule
rankings without recalculating or selecting a model. The report opens on
validation, shows all endpoint-specific preprocessing/split contexts by
default, and offers exact context labels for the selected diagnostics endpoint.
Context/model coverage appears before ranking. Save/load view recipes bind the
filters and lasso to the report signature; cohort CSV export retains context.
Raw results, the data dictionary and provenance stay collapsed until requested.
The error card and selection distribution use only the diagnostics endpoint.
Full-fold model metrics are never recomputed from chemical filters.
Use `--embedding pca` for an alternative to the default t-SNE map.

`--max-points` limits interactive payload size using complete chemical
histories: all retained folds and endpoints for a sampled canonical molecule
stay together. The limit is therefore a soft ceiling when one molecule alone
has more observations than the configured value. Self-contained reports are
convenient to share but can be tens of megabytes because Plotly and molecule
depictions are embedded; externalized assets remain planned for larger studies.

## Produce predictions with the bundled model arena

The training workflow remains available as one example producer:

```bash
admet-benchmark-arena configs/benchmarks/ksol_model_arena.yaml
admet-benchmark-arena configs/benchmarks/bbbp_model_arena.yaml
```

It performs split-consistent tuning on development data, chooses a model from
validation performance, and evaluates the locked test only afterward. Each run
writes the same canonical `evaluation/` bundle consumed by the reports.

More detail is available in the
[`pipeline and reporting guide`](docs/using_pipeline_and_reports.md) and
[`architecture note`](docs/architecture.md).

## Repository layout

```text
src/admet_benchmark/       maintained package
configs/                   reproducible evaluation and training examples
data/                      small example inputs
tests/                     behavioral and report-contract tests
docs/                      product, architecture, and usage guides
legacy/openadmet_report/   preserved original challenge/report workspace
references/                supplied research references (local PDFs ignored)
outputs/                   generated runs and preserved current reports (ignored)
```
