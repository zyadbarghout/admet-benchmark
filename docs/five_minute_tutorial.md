# Five-minute local tutorial

Start from a checkout and Python 3.12. With uv installed:

```bash
uv sync --locked --python 3.12 --extra visual --extra test
uv run --no-sync admet-study preflight configs/study_example.yaml
uv run --no-sync admet-study freeze configs/study_example.yaml --output outputs/tutorial_study
uv run --no-sync admet-study verify outputs/tutorial_study
uv run --no-sync admet-evaluate run configs/evaluation_regression_example.yaml
uv run --no-sync admet-evaluate run configs/evaluation_classification_example.yaml
uv run --no-sync admet-chemical-report \
  outputs/evaluation_regression_example \
  outputs/evaluation_classification_example \
  --embedding pca --output outputs/tutorial_diagnostics.html
```

Use a new study output directory when repeating `freeze`. The prediction
examples and the synthetic study demonstrate two separate interfaces. They
are not asserted to belong to the same registered experiment. No model is
trained by these commands. Source observations and study assignments are
available in the registration directory for inspection.

Open `outputs/tutorial_diagnostics.html` in a browser. No server or network is
required by the generated file.

1. Start on **Validation**. Inspect the context and molecule coverage table.
2. Select the diagnostics endpoint. The median-error card uses only its scale.
3. Search for a molecule, choose a motif or lasso a region. Empty views explain
   the active filters; **Clear lasso** and **Reset filters** recover the view.
4. Use **Inspect a selected molecule** to inspect structures with the keyboard.
5. Choose **Save view**, then change filters and **Load view** to restore them.
   A view belongs to the exact report signature, including its sample and
   embedding; loading it into different report data is rejected.
6. Choose **Export cohort CSV** for selected evaluation observations and the
   active model. The export includes identity, context, predictions, residuals,
   uncertainty when available and the report signature. Save the view alongside
   the CSV to preserve the filter recipe. CSV text fields beginning with formula
   characters receive a leading apostrophe for spreadsheet safety.

The report contains its displayed molecular structures; handle the HTML, CSV
and recipes under the same data policy as the underlying study. A recipe can
include search text and observation IDs. These controls export local downloads
and do not transmit data to an external service.

Full-fold model metrics retain their original scope. They are never recomputed
from a motif, search or lasso selection. The locked-test control only displays
already saved results; it does not enforce scientific test blinding.

## Reproduce the checks

```bash
uv sync --locked --python 3.12 --extra visual --extra test --extra browser
uv run --no-sync playwright install chromium
RUN_BROWSER_TESTS=1 uv run --no-sync pytest -q
```

The default unit suite skips browser tests unless `RUN_BROWSER_TESTS=1`.
On Linux, `playwright install --with-deps chromium` also installs system
libraries. CI uses this mode. The pipeline has been added locally; a passing
hosted CI result requires pushing to a configured remote.
