# ADMET Benchmark

Evaluate molecular property predictions and explore **which chemistry a model
gets wrong**. ADMET refers to absorption, distribution, metabolism, excretion,
and toxicity: properties that help characterize potential drug molecules.

The main workflow accepts CSV prediction tables from any model or framework.
It validates molecular identities and experiment metadata, computes regression
or classification metrics, compares models across matching folds, and produces
interactive HTML reports with molecular structures and chemical-space maps.
Optional training workflows generate predictions with classical models and
Chemprop, register cross-validation splits, and calibrate uncertainty.

## What you can do

- Evaluate regression predictions or binary classification probabilities using
  explicit target transforms and decision thresholds.
- Compare models and endpoints while keeping validation/test, split strategy,
  preprocessing, repeat, and fold context attached to each result.
- Audit duplicate chemical identities, missing model coverage, and conflicting
  observations. Registered studies also check split leakage.
- Inspect error distributions, structures, SMARTS substructures, chemical-space
  selections, and optional uncertainty scores or prediction intervals.
- Export selected molecular cohorts and save report views for later inspection.
- Train linear, random forest, and ExtraTrees baselines; install optional model
  dependencies for XGBoost, LightGBM, and Chemprop adapters.

## Installation

Use **Python 3.11 or newer**. Python 3.12 and the committed `uv.lock` are the
recommended reproducible setup. Run all examples from the repository root.

```sh
git clone https://github.com/zyadbarghout/admet-benchmark.git
cd admet-benchmark
uv sync --locked --python 3.12 --extra visual --extra test
```

The repository is private, so cloning requires access through your GitHub account.
The `visual` extra enables interactive Plotly reports. The examples below use
`uv run --no-sync` to keep the installed environment unchanged.

If you prefer pip, create and activate an environment first:

```sh
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```sh
source .venv/bin/activate
```

Then install the package:

```sh
python -m pip install -e ".[visual,test]"
```

With an activated pip environment, omit `uv run --no-sync` from the commands below.
Installation needs internet access; the bundled quickstart runs locally afterward.

## Working examples

### 1. Evaluate regression predictions

The bundled table contains illustrative predictions for KSOL (solubility) and
LogD (lipophilicity), with two models and repeated validation folds.
The configuration applies `log10` to KSOL and leaves LogD on its supplied scale.

```sh
uv run --no-sync admet-evaluate validate configs/evaluation_regression_example.yaml
uv run --no-sync admet-evaluate run configs/evaluation_regression_example.yaml
```

Open `outputs/evaluation_regression_example/report.html` to inspect metrics and
prediction distributions. `validate` checks inputs without writing a result bundle.

### 2. Evaluate classification probabilities

This example reads a wide CSV with one probability column per model. It evaluates
BBBP (blood-brain barrier permeability) using the configured threshold of **0.42**.

```sh
uv run --no-sync admet-evaluate validate configs/evaluation_classification_example.yaml
uv run --no-sync admet-evaluate run configs/evaluation_classification_example.yaml
```

Open `outputs/evaluation_classification_example/report.html`. Outputs include
ROC AUC, PR AUC, MCC, and threshold-dependent metrics.

The prediction examples are small software demonstrations. Their model names
are labels for supplied predictions; these commands do not train those models
or establish their scientific performance.

### 3. Explore errors across chemical space

After running examples 1 and 2:

```sh
uv run --no-sync admet-chemical-report outputs/evaluation_regression_example outputs/evaluation_classification_example --embedding pca --output outputs/quickstart/chemical_diagnostics.html
```

Open `outputs/quickstart/chemical_diagnostics.html` directly in your browser.
The default delivery embeds the plots and structures for offline use.
Choose an endpoint and model, search molecular IDs or SMILES, filter by a
substructure, or select a region of the map. Inspect the selected molecules,
save the view, and export a cohort CSV. Full-fold metrics keep their original
population when you change chemical filters.

To run all three examples with one command and store everything under a single
directory:

```sh
uv run --no-sync python examples/quickstart.py
```

You can override the destination with `--output-dir outputs/my_demo`.
The script can be rerun and replaces its generated evaluation files.
With the bundled inputs, it writes 32 regression predictions with 40 metric rows
and 32 classification predictions with 120 metric rows, then prints the report's
local browser URL.

### 4. Register reproducible study splits

The synthetic study example freezes molecule-level assignments and records
data/configuration hashes before any model fitting:

```sh
uv run --no-sync admet-study preflight configs/study_example.yaml
uv run --no-sync admet-study freeze configs/study_example.yaml --output outputs/tutorial_study
uv run --no-sync admet-study verify outputs/tutorial_study
```

Choose a new output directory each time you run `freeze`. This example uses
artificial observations and is independent of the prepared prediction examples.
See the [study registration guide](docs/study_registration.md) for the contract.

## Use your own predictions

Copy one of the two `configs/evaluation_*_example.yaml` files and map your CSV
column names. Dataset and output paths in YAML are relative to the configuration
file. An explicit CLI `--output-dir` is relative to your current directory.

A long-form table typically has one row per sample/model/endpoint/fold:

```csv
sample_id,molecule_id,smiles,endpoint,model,partition,repeat,fold,observed,prediction
sample-001,mol-001,CCO,solubility,baseline,validation,0,0,1.2,1.0
sample-002,mol-002,CCN,solubility,baseline,validation,0,0,2.1,2.4
```

This snippet shows the shape; use a full dataset with sufficient molecules for
metrics and uncertainty estimates. Configure target transforms explicitly, and
set a decision threshold for every classification endpoint. Optional uncertainty
columns must have declared direction/scale, and interval bounds must form valid
pairs. The [evaluation guide](docs/evaluation_cli.md) covers long/wide mapping,
validation requirements, threshold policies, and uncertainty handling.

Each evaluation writes:

| File | Contents |
| --- | --- |
| `report.html` | Interactive evaluation report |
| `predictions.csv` | Canonical prediction records and derived errors |
| `fold_metrics.csv` | Metrics per model, endpoint, partition, repeat, and fold |
| `metric_distributions.csv` | Summaries across matching result contexts |
| `coverage_audit.csv` | Model and molecule coverage |
| `statistical_comparisons.csv` | Exploratory paired model comparisons |
| `validation_report.json` | Input audit findings |
| `resolved_config.yaml` | Configuration used for the evaluation |
| `evaluation_manifest.json` | Source and output provenance |

## Training and larger studies

The quickstart needs no trained checkpoints or external datasets. For model
training, see the [pipeline guide](docs/using_pipeline_and_reports.md),
[cross-validation guide](docs/real_cv_uq.md), and
[complete real-data example](docs/complete_real_example.md).

The larger example uses the bundled KSOL, LogD, and BBBP snapshots. To check its
protocol without fitting models:

```sh
uv run --no-sync python scripts/build_complete_example.py --validate-only --output outputs/protocol_check
```

For a fresh full training run, install the optional models extra first:

```sh
uv sync --locked --python 3.12 --extra models --extra visual --extra test
uv run --no-sync python scripts/build_complete_example.py --output outputs/my_complete_example
```

This runs a substantial cross-validation study, including a neural addon; it is
not a quickstart. `--classical-only` excludes that addon. Generated reports,
trained models, and previous local study results are excluded from Git and must
be built locally. Checkpoint loading enforces its recorded runtime and feature
compatibility; use a new registration after changing code or dependencies.

## Verification

Run the unit and integration suite:

```sh
uv run --no-sync pytest -q
```

For the offline Chromium acceptance tests:

```sh
uv sync --locked --python 3.12 --extra visual --extra test --extra browser
uv run --no-sync playwright install chromium
```

Then on PowerShell:

```powershell
$env:RUN_BROWSER_TESTS = "1"
uv run --no-sync pytest -q
```

Or on macOS/Linux:

```sh
RUN_BROWSER_TESTS=1 uv run --no-sync pytest -q
```

GitHub Actions runs the tests, browser checks, study CLI workflow, and wheel
resource checks. Browser tests are skipped unless `RUN_BROWSER_TESTS=1`.

## Repository layout

```text
src/admet_benchmark/   Maintained package, CLI tools, report assets, and schemas
configs/              Evaluation, registration, and training configurations
data/                 Small prediction examples, synthetic data, real snapshots
examples/             Runnable quickstart
tests/                Scientific behavior and browser acceptance tests
scripts/              Larger study builders and artifact checks
docs/                 Usage guides, architecture, and historical audit records
references/           Research reference index
outputs/              Generated results (ignored by Git)
```

The original submission workspace and private research archives remain local
and are excluded from this repository. They are not required by the quickstart
or maintained package. Historical documents may refer to those local artifacts.

## Interpretation and data provenance

Reports support development diagnostics. Overlapping validation folds are
dependent, so nominal comparison p-values are exploratory. Chemical-space
distance is not a calibrated applicability guarantee, and empirical uncertainty
coverage can change under chemical shift or post-hoc cohort selection.

The [synthetic study data](data/tutorial/README.md) is explicitly artificial and
dedicated to CC0. Real snapshots retain their providers' terms; see the
[classification data notes](data/classification/README.md) and
[real-data provenance notes](data/real_demo/README.md). No repository-wide license
has been declared. The [architecture](docs/architecture.md) and
[project assessment](docs/project_assessment.md) describe design and limitations.
