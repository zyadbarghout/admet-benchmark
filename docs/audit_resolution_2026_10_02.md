# Audit resolution and completed real-data example — 2 October 2026

This completes the implementation and demonstration work interrupted by usage
limits. The original [September 28 audit](code_and_design_audit_2026_09_28.md)
is retained as the record of the defects found. The example directory retains
its September 29 start date.

## What the tool does

The package validates molecular regression/classification predictions, compares
models in matching evaluation contexts, and shows where their errors and
uncertainty vary across chemical space. It accepts externally prepared
predictions and can also train registered classical and neural models. The
interactive offline report connects structures, chemical maps, held-out metrics,
calibrated intervals/class sets, chemical cohorts and reproducible view exports.

## Defects fixed

| Audit finding | Resolution |
| --- | --- |
| External prediction leakage | Reject fitting/evaluation overlap within a fold and locked-test overlap across development folds, using sample and canonical chemical identities. When fitting roles are absent, training leakage remains explicitly unverified. |
| Incorrect bootstrap entity | Canonical chemical keys determine bootstrap groups and molecule counts; aliases are surfaced while submitted identifiers remain intact. |
| CSV identity loss | Identity/text columns preserve leading zeros and NA-like strings through ingestion, reporting and batch inference. |
| Inconsistent classification threshold | Confusion and retained-risk diagnostics share the saved threshold policy; a non-default-threshold browser regression test checks agreement. |
| Checkpoint/runtime incompatibility | Registration and prediction enforce source/runtime fingerprints and the complete feature schema. The exact trained environment is recorded separately from the lock used for new runs. |
| Unsupported dependency minimum | scikit-learn minimum raised to 1.6; a separate 1.6.1 environment and CI job check that boundary. |
| Eager report loading and weak hierarchy | Offline assets delivery, compact record transport, on-demand maps/depictions, lazy plots, a density-calculation fix, a compact overview and context strip, clearer feature labels and highest-error-first molecule inspection. |

The stricter chemistry checks also exposed duplicate-molecule leakage in the
older benchmark producer. Random splitting and inner tuning now group those
duplicate chemicals. Existing historical studies were preserved rather than
rewriting their compatibility metadata.
Report generation now also reuses canonical structures across repeated fold
records and parses each unique structure only once for substructure matching.

## Completed training and artifacts

Open the [example landing page](../outputs/complete_real_example_2026_09_29/index.html)
and follow the [reproduction and exploration guide](complete_real_example.md).

- Real measured endpoints: 5,128 KSOL, 4,934 LogD and 1,569 curated BBBP chemicals.
- Five folds, two seeds, molecule/scaffold splits, Morgan/RDKit features and
  linear/random-forest/ExtraTrees families: **360 jobs, 1,320 CV estimators**.
- Full BBBP Chemprop addon: **five outer folds, two models per fold, 50 epochs per
  model**. Its scope is molecule splits at the first seed.
- An independently outer-evaluated KSOL CV+ pilot.
- **18 final calibrated classical predictors**, with development molecules
  assigned to disjoint fitting/calibration roles; 66 underlying estimators.
- Saved checkpoints, calibration files, role assignments, data/source hashes,
  complete OOF predictions and inference on five named real structures. Query
  examples have no invented assay labels.

“Complete” means the registered training budgets finished and the saved models
perform inference. It does not mean optimal hyperparameters or prospectively
validated performance. Neural batch inference uses an explicitly identified
outer-fold ensemble, not a claimed full-development neural refit.

## Verification

The final local suite passed **73 tests**, including ten offline browser checks.
The separate scikit-learn 1.6.1 suite passed **63 tests** (ten browser checks
intentionally skipped). Wheel assets and schemas passed packaging validation.
The independent outcome audit verified **360 jobs, 279,144 prediction rows and
36 checkpoint reloads**. The largest observed reload discrepancy was
`4.77e-7`; classification sets agreed exactly.

The follow-up report suite passed **17 tests** after the parsing optimization.
The complete real report passed offline Chromium acceptance with **94,617
unsampled observation records**, all nine map combinations, saved-view
round-trips, CSV export, interval/property-cutoff/cohort controls and all 1,569
neural OOF observations. No JavaScript errors, external requests or mobile
horizontal overflow were observed. Desktop, mobile, map and neural uncertainty
screenshots were rendered and inspected.

Initial report readiness took **9.09 seconds** on this host, with two plots
instantiated initially, versus 34.28 seconds and 18 plots for the earlier audit
report. This is an indicative local comparison across different report sizes,
not a controlled benchmark. The HTML shell is **2.09 MiB**; sibling assets occupy
about **174 MiB**, including a 118 MiB complete record table. The directory must
stay together for offline use. The smaller HTML is not the total payload size.

Evidence is saved under
[`verification/`](../outputs/complete_real_example_2026_09_29/verification/),
including `browser_audit.json`, screenshots and local test/build logs.
All **6,714 report artifacts** passed their manifest hash checks. Registration
verification also passed after the final report changes.

The maintained package, tests, guides, configuration and local demonstration
snapshots form a local Git baseline on `codex/audit-fixes-real-demo`. Generated
model/report artifacts stay under the ignored `outputs/` directory; the original
legacy archive remains separate. No remote repository or hosted CI run was used.

The local unit/browser suite and minimum-supported-dependency suite cover the
correctness fixes. The builder independently checks all completed job hashes,
exact held-out membership and observed labels, and reloads representative
checkpoints. Final predictors and neural checkpoints are also hash-checked and
used for real inference.

Reload checks retain a numerical tolerance of `rtol=1e-6, atol=1e-6` for point
predictions/intervals and exact equality for class sets. Float32 Morgan linear
models can differ in their last bits across BLAS batch sizes. Each reload's
measured maximum discrepancy is saved in `outcome_audit.json`.

## What the measured results show

For the RDKit random forest, per-repeat KSOL MAE is 0.389–0.391 on molecule
splits and 0.426–0.434 on scaffold splits (log10 source assay units). LogD MAE
is 0.418–0.421 versus 0.474–0.480. The harder chemistry split therefore exposes
degradation hidden by a single random-split score.

BBBP forest PR AUC is 0.965–0.967 on molecule splits and 0.951–0.959 on scaffold
splits. The completed Chemprop addon scores 0.8957 PR AUC, 34.67% specificity
and 89.61% class-set coverage. It underperforms the matching first-seed RDKit
forest (PR AUC 0.9651); the demonstration preserves that finding. These are
descriptive development results, not a tuned winner selection or a clinical
claim. Full metrics and per-context predictions remain available in the example.

## Work still open

These are research, architecture or release extensions, not unfinished training
jobs in this example:

1. Fresh external/temporal/prospective evaluation and validation of the final
   refits on untouched future data.
2. Nested hyperparameter tuning, full neural scaffold/repeat coverage and a
   validated CheMeleon benchmark; the current budgets are fixed in advance.
3. Confirmatory inference accounting for dependent folds and prespecified
   chemical cohorts. Property-cliff cohorts use observed outcomes and remain
   exploratory stress diagnostics.
4. Further frontend state/calculation separation and endpoint-level data paging.
   Maps and plots are lazy, but the complete packed record table is still loaded
   into browser memory. Shared bundle contracts could also be consolidated.
5. Dataset publication/license decisions, hosted CI execution, release packaging
   policy and any public deployment. No external publication was performed.
