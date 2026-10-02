# Code and design audit — 28 September 2026

The maintained package is a capable scientific diagnostics alpha. Its strongest design choice is accepting prepared molecular predictions from any model, then retaining experiment context through evaluation and chemical exploration. The immediate priority is tightening correctness at that input boundary and reducing report complexity and load time.

This audit inspected source, configuration, tests, architecture/status documents and the real CV report. It ran the existing tests, targeted adversarial input probes, and Chromium checks at desktop and mobile sizes. It did not retrain the 120-job study, reproduce every historical result, certify the optional neural adapters, or conduct a complete security/accessibility audit. Application code and existing reports were not changed.

## What the project does

ADMET refers to absorption, distribution, metabolism, excretion and toxicity. This project evaluates molecular machine-learning predictions and helps scientists inspect where models fail chemically.

The principal workflow is:

1. Supply CSV/TSV/Parquet predictions and YAML column/experiment mappings.
2. Validate structures, labels, scales, identifiers, model coverage and experiment context.
3. Calculate regression/classification metrics, molecule-bootstrap intervals, uncertainty diagnostics and exploratory paired fold comparisons.
4. Generate a report connecting those results to molecular structures, fingerprints/descriptors, chemical motifs and projected chemical maps.
5. Filter molecules, inspect errors and intervals, compare saved fold results, and export a cohort or reproducible view recipe.

Optional producers train classical models, generate registered cross-validation predictions and calibrated uncertainty, and perform local SMILES inference from saved fold checkpoints. The neural integration and CV+ implementation are narrower extensions. The older OpenADMET notebooks/submission workspace is preserved separately under `legacy/`.

The current demonstration concerns KSOL solubility regression and BBBP blood–brain-barrier classification; it does not establish performance across all ADMET properties.

## Verification performed

| Check | Result |
| --- | --- |
| `.venv/bin/python -m pytest -q` | 54 passed, 8 browser tests skipped; 19.91 seconds |
| `RUN_BROWSER_TESTS=1 .venv/bin/python -m pytest -q tests/test_report_browser.py` | 8 passed; 57.90 seconds |
| `RUN_BROWSER_TESTS=1 .venv-models/bin/python -m pytest -q` | 62 passed on Python 3.12.12, including all browser tests; 69.12 seconds |
| Actual real CV HTML in Chromium | No page errors observed |
| Report size | 114,310,289 bytes, approximately 109 MiB |
| Initial load and map readiness | 34.28 seconds in this local run; a single measurement, not a controlled benchmark |
| Desktop default view, 1440 × 1000 | Document height 8,714 px; 18 Plotly plots instantiated |
| Mobile viewport, 390 × 844 | No horizontal document overflow; visually inspected |
| Task-tab keyboard navigation | ArrowRight correctly activated and focused Classification |
| Git baseline | Repository exists but has no commits; project source is untracked |

The initial `.venv` checks used Python 3.10.0, below the package's declared minimum of 3.11. Its installed libraries were NumPy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2 and RDKit 2025.09.6. The entire suite subsequently passed in the existing Python 3.12.12 `.venv-models` environment. Identifier loss, both overlap cases and chemical-alias overcounting were also reproduced on Python 3.12. This audit does not claim a fresh installation from the lockfile.

Screenshots: [desktop controls](../outputs/audit_2026_09_28/desktop-top.png), [chemical map](../outputs/audit_2026_09_28/desktop-map.png), [mobile](../outputs/audit_2026_09_28/mobile-top.png).

## Code findings

### 1. High: prepared-prediction leakage validation has blind spots

Evidence: `src/admet_benchmark/evaluation.py:410` and `:543`.

The sample overlap check considers validation/test membership within the same repeat and fold. Its chemical overlap check has the same fold boundary. Neither checks supplied training predictions against validation/test chemistry within a fold.

Two probes passed with no warnings:

- Identical samples/structures in validation fold 0 and test fold 1.
- Identical samples in train and validation within the same fold.

For a declared locked test, the first case defeats the intended development/test separation. The second is direct overlap of fitting and evaluation membership when the submitted train role represents fitting data. Repeated CV legitimately moves molecules between training and validation in different folds; a fix must preserve that behavior.

Recommendation: add an explicit experiment/role contract, enforce fit-versus-evaluation disjointness within each fold, and enforce locked-test isolation across all development folds/repeats of the same experiment. Use canonical chemical identity as well as sample IDs. When fitting assignments are absent, report that training leakage is unverified. The registered CV producer already has stronger partition checks; this finding is about external prepared predictions.

### 2. High: bootstrap entities are not reconciled with canonical chemistry

Evidence: `evaluation.py:699`, `:765`, and `:929`.

Bootstrap resampling groups by submitted `molecule_id`, and `n_molecules` counts those IDs. Validation checks identity agreement across models for a sample context, but does not reconcile different IDs for the same structure.

A probe with three IDs and structures `CCO`, `OCC`, `CCC` passed with no warning and reported three molecules, although canonicalization yielded two chemicals. Those two ethanol records can therefore be independently resampled, contrary to the documented chemical-entity bootstrap. This can distort uncertainty and support counts when sources use different identifiers for replicates.

Recommendation: preserve submitted IDs, add a stable canonical chemical key, enforce the declared identity/replicate policy, and bootstrap at the declared chemical unit. Counts should separately expose measurements and unique chemicals. If scientific identity intentionally includes more than structure, make that rule explicit in configuration.

### 3. Medium: CSV parsing silently changes identifiers

Evidence: `evaluation.py:125` and `chemical_report.py:288`.

CSV files are read with pandas type inference before identity columns are converted to strings. The input IDs `001`, `002`, `003` became `1`, `2`, `3`, while validation returned `passed`. This breaks traceability and may collapse distinct identifiers. Default missing-value parsing can also affect textual identifiers.

Recommendation: preserve mapped identity/text columns at read time, including report reloads, and parse numeric fields explicitly afterward. Verify leading zeros, mixed identifiers and NA-like strings through evaluation, report and export round trips.

### 4. Medium: classification risk curves ignore the saved threshold

Evidence: `report_assets/scientific.js:40`; compare `report_assets/workspace.js:69`.

The prediction-set risk/retention curve hardcodes a probability threshold of 0.5, while the confusion matrix respects the saved per-observation/endpoint threshold.

In a browser probe using threshold 0.8, negative-class predictions of 0.6 and positive-class predictions of 0.9, the confusion matrix was `[[4,0],[0,4]]`, but the risk curves reported 0.5 error at full retention. Both panels should report zero classification error. The current registered producer uses 0.5, so its default demonstration does not expose this discrepancy; richer external bundles can.

Recommendation: share a single threshold-resolution helper across all classification panels and test agreement at non-default thresholds.

### 5. Medium: runtime and feature compatibility are recorded incompletely and not enforced

Evidence: `cross_validation.py:136`, `:164`, `:178`, and `:365`; `features.py:34`.

Registration records NumPy, scikit-learn and RDKit versions, but verification does not compare them with the active environment. The implementation fingerprint covers five modules, omitting evaluator/metric code used to assemble results. Prediction checks the model file hash, then rebuilds descriptors from the current RDKit descriptor list without validating stored descriptor names/order or runtime compatibility.

Consequences include mixed-runtime resumed studies, changed metrics under an unchanged registration, and incompatible feature generation when checkpoints are moved between environments. These are code-path risks; this audit did not establish that the existing study suffered them.

Recommendation: bind the environment lock, relevant source/build identity, and complete feature schema to registration/checkpoints. Check compatibility before resuming or predicting, with an explicit migration path. Keep numerical reload-equivalence checks for representative checkpoints.

### 6. Medium: the advertised scikit-learn minimum is too low

Evidence: `pyproject.toml:19`; `cross_validation.py:98`.

The package permits scikit-learn 1.3, but regression scaffold assignment uses `GroupKFold(..., shuffle=True, random_state=...)`. The installed library's documentation marks these options as introduced in 1.6. An otherwise permitted 1.3–1.5 environment cannot execute this path.

Recommendation: raise the minimum to match the API requirement or provide compatible deterministic splitting, and add a minimum-supported-environment check. The locked current environment passing tests does not verify the advertised lower bound.

### 7. Medium: report work is eager and tightly coupled

Evidence: `chemical_report.py:640`, `report_assets/report.js:37`, and `report_assets/workspace.js:36`.

The report embeds data, structures, maps and Plotly into one approximately 109 MiB document. Selection updates call a broad chain of diagnostics/render functions, while map redraw rebuilds lookups over the report records. The observed default page instantiated 18 plots and took 34.28 seconds to load.

Recommendation: preserve portable HTML for small studies and add a local assets mode for larger ones. Load endpoint/context data and depictions on demand, render panels when visible, index common queries once, and update only affected panels. Define performance acceptance criteria on the real report, not only the small browser fixtures.

## Architecture assessment

Good foundations include the model-neutral prediction boundary, explicit task/context columns, dedicated fitting and calibration roles in registered CV, retained fold observations, saved artifacts and hashes, isolated legacy material, packaged report assets and offline browser tests. The report also distinguishes chemical exploration from saved full-fold model metrics.

The architecture is accumulating parallel paths: older pipeline/arena workflows, repeated-holdout study registration, newer registered CV, and multiple report consumers. `cross_validation.assemble` validates through evaluator helpers but then manually assembles a richer bundle instead of using one common writer. This makes metadata, metrics and uncertainty semantics easier to diverge.

The newer Python modules and roughly 77 KB of report JavaScript use many long, compressed statements. Frontend modules share globals and initialization order; scientific helpers are connected directly to controls and rendering. Existing tests protect useful behavior, but maintenance would benefit from readable formatting, typed contracts and pure calculation functions.

Recommended structure: common typed prediction/role/UQ contracts; one validated bundle reader/writer; separate training adapters; shared statistical functions; explicit frontend state and selectors; independently testable chart adapters. A framework rewrite is optional—the state and contract separation matters first.

## Visual and interaction design

The visual direction is coherent: warm backgrounds, dark green accents, readable serif headings, grouped controls and substantial chart space. Molecular inspection beside the map works well. Validation-first defaults, selected counts, exploratory labels, saved views, cohort exports and collapsed provenance are good choices. Responsive layouts and keyboard tab navigation already work in the checks performed.

The main weakness is hierarchy. At desktop size, the first screen is almost entirely title/navigation/controls; the scientific evidence starts below it. The default page is approximately nine screens long. Mobile reflows correctly but makes the controls consume still more vertical space. Major analytical panels compete for attention.

Specific enhancements:

- Put a compact study summary and the primary result near the top, then offer workspaces for model comparison, chemical failures, uncertainty and molecule inspection.
- Keep endpoint, model and validation context visible; move embedding options, fold display and advanced cohort controls into expandable sections.
- Add a persistent context/selection strip. Explicitly distinguish “selected cohort” panels from “full-fold benchmark” panels, because filters intentionally affect them differently.
- Rename ambiguous controls. In this study, “Preprocessing” contains model feature representations, while “Representation” controls the chemical map. Use “Model features” and “Map features” where appropriate.
- Expand endpoint names and show units/scales beside metrics and cutoffs. KSOL/BBBP and raw internal model names assume prior knowledge.
- Prioritize worst errors, interval misses and high-confidence failures in the molecule list; offer best-predicted molecules as a secondary sort.
- Provide a visible loading state and progress for large reports. Disable controls until the relevant data/panels are ready.
- Continue accessibility work with chart data alternatives, screen-reader checks, contrast measurements and equivalent ways to define cohorts without lasso. Existing keyboard and overflow checks are useful but do not establish complete accessibility.

## Missing capabilities and evidence

The source/status documents already acknowledge many of these limitations. They should remain distinct from confirmed software defects:

- A confirmatory research protocol and dependence-aware model comparison; existing nominal fold p-values remain exploratory.
- Broader repeated-seed evidence, nested tuning where needed, and temporal/external evaluation with an untouched analysis policy.
- A completed neural comparison and exercised CheMeleon workflow; a smoke run is integration evidence only.
- A declared full-development deployment/refit/calibration policy and prospective inference validation; current inference uses outer-fold checkpoints.
- Cohort effect uncertainty and declared support/post-selection policies. Registered assembly already omits cohorts below ten observations, so basic support filtering is not entirely absent.
- A real overlapping multi-endpoint demonstration, rather than assuming KSOL and BBBP share the same chemicals.
- An experiment index and a simpler first-run workflow. CLI/YAML is usable by technical researchers, but non-programmers lack a prediction-import/mapping interface.
- A committed release baseline and hosted CI evidence. A workflow and lockfile exist, but this checkout has no commits. Establish publication scope before committing or publishing historical material.
- Clear project/data/model licensing and distribution metadata; there is no root project license file in the inspected tree.
- Current documentation: some older roadmap pages still describe generated UQ and shift diagnostics as future work, although newer implementation/status files contain them.

## Recommended order

1. Repair identity preservation, chemical bootstrap grouping, role/leakage validation and threshold consistency; add regression checks for the reproduced cases.
2. Enforce runtime/feature compatibility, correct dependency bounds and consolidate bundle validation. Establish a reviewed source-control baseline and CI run.
3. Reduce report load time and reorganize the default screen around the analysis task. Retain the current visual language.
4. Expand scientific evidence and deployment support under a frozen protocol. Additional model families should follow the correctness and reproducibility work.
