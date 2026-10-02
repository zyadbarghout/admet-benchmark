# Project status and future roadmap

Current implementation update: 2026-10-02. See [the complete real-data example](docs/complete_real_example.md)
and [the audit resolution](docs/audit_resolution_2026_10_02.md) for the new code,
verification and demonstration. The September 9 measured study below is retained
as historical evidence; its immutable artifacts were not rewritten to bypass the
new compatibility checks.

## Current implementation

The current example covers KSOL, LogD and BBBP: 360 classical CV jobs, 1,320
fitted CV estimators, ten Chemprop models across five BBBP folds, a KSOL CV+
pilot and 18 final calibrated classical predictors. All training budgets have
finished. The [landing page](outputs/complete_real_example_2026_09_29/index.html)
links the complete report, model artifacts, verification and inference examples.

The audit fixes preserve textual IDs, validate canonical chemical identity and
leakage, bootstrap chemical entities, share classification threshold semantics,
and enforce runtime/source/feature compatibility. The supported scikit-learn
minimum is now 1.6. Reports support offline assets, on-demand maps, lazy plotting,
a compact study overview and failure-oriented molecular inspection.

Remaining work includes fresh prospective/temporal validation, nested tuning,
validated CheMeleon results, confirmatory dependent-fold/cohort inference and
further frontend modularization. Publication rights and hosted CI/release remain
unresolved. Finished training does not establish prospective performance.

## Historical September 9 status

The project is a working chemistry-aware model diagnostics alpha with a registered
classical cross-validation and uncertainty pipeline. The first measured KSOL/BBBP
study is complete: 120 jobs, 440 fitted estimators and 80,364 out-of-fold prediction
rows. Results remain development evidence; a confirmatory benchmark and final
production model are not yet established.

Open the [current real-data report](outputs/real_cv_uq_2026_09_09/chemical_model_diagnostics.html),
read the [measured outcome](docs/real_cv_uq_outcome_2026_09_09.md), and use the
[run/resume/predict guide](docs/real_cv_uq.md) to reproduce or extend the workflow.
The older [workspace guide](docs/chemical_workspace_v3.md) explains the chemical
maps, GTM and threshold interactions. The synthetic 5/10-fold demonstration is
preserved as a UI fixture; the new report uses measured outcomes.

## What the project does

The common public boundary is a prepared prediction table. External submissions,
classical models and neural producers can share validation and reporting. The
optional `admet-cv` producer now connects saved fold roles, fitted artifacts,
calibration outputs and evaluated OOF predictions. It is separate from the older
`admet-study` repeated-holdout registry; both contracts remain supported.

The report helps identify chemical regions where predictions fail, compare models
under matching contexts, inspect uncertainty, and distinguish exploratory
chemical selections from declared evaluation populations.

| Area | Current capability | Remaining limitation |
| --- | --- | --- |
| Prepared predictions | Regression/classification, explicit scales, transforms, thresholds, models, partitions and folds; schema and semantic validation | New producer-specific uncertainty semantics need explicit adapters |
| Real CV producer | Saved molecule/scaffold roles, repeated-seed configuration, 5/10-fold support, shared assignments, artifact hashing and resume | Measured sweep used one seed and five folds; nested HPO remains open |
| Model families | Linear, random forest, ExtraTrees; fresh neural adapter consuming registered roles | Full measured neural CV and CheMeleon execution remain open |
| UQ generation | Seed ensembles, normalized/absolute regression conformal, sigmoid probability calibration, class-conditional sets | Coverage under shift or post-hoc cohorts is empirical, not guaranteed |
| CV+ | Independently outer-evaluated KSOL/Morgan pilot; fold-specific candidate order statistics | One endpoint/split; not an isolated interval-method comparison |
| Chemistry | Morgan nearest-fitting similarity, scaffold novelty, post-hoc property-cliff labels and cohort metrics | Temporal/external shift and confirmatory cohorts remain open |
| Chemical space | Morgan/RDKit/combined × t-SNE/GTM/PCA, shared atlas and fold facets | Embedding distances are not calibrated applicability; independent GTM validation remains open |
| UI | Separate task tabs, optional merged exploration, original ID/SMILES search, thresholds, cutoff decisions, UQ and prediction sets, saved views and cohort exports | Large offline HTML needs a lazy-loading mode for larger studies |
| Local inference | Hashed saved-fold checkpoint prediction for SMILES batches; interval/set, scale and applicability outputs | No full-development deployment refit or upload UI |
| Reproducibility | Locked dependencies, source/data hashes, completion manifests, saved members and calibrations, test suite and packaged assets | Hosted CI/release have not run; source tree has no baseline commit |
| Statistical claims | Raw fold observations, descriptive paired summaries and molecule-level metrics | Overlapping folds are dependent; existing nominal p-values are exploratory |

## Measured findings

The curated development pools contain 5,128 KSOL and 1,569 BBBP chemicals. Nine
conflicting BBBP identities were quarantined and consistent replicates collapsed
in a new snapshot. Original files were preserved. Historical test labels were
excluded from training and calibration; previously inspected tests are not a
fresh confirmatory resource.

For the RDKit random forest, KSOL normalized-interval coverage is 90.04% on
molecule splits and 89.84% on scaffold splits. The scaffold property-cliff cohort
has only **68.32% coverage** across 202 observations. This outcome-dependent cohort
is a useful stress diagnostic, not a deployable cliff classifier.

BBBP/RDKit/random forest yields PR AUC 0.9651 and set coverage 89.61% on molecule
splits, versus PR AUC 0.9592 and coverage 89.04% on scaffold splits. Calibration
improves some contexts and slightly worsens others; raw and calibrated proper
scores are both retained.

CV+ reaches 90.50% empirical coverage with mean KSOL interval width 1.9303; the
Morgan split-conformal ensemble's width is 1.8320 at 90.09% coverage. This pilot
does not justify replacing the simpler baseline. Different training allocations
and point learners limit causal interpretation of that comparison.

The two-member, two-epoch Chemprop smoke run completed one BBBP fold. Its
specificity is only 0.0533 despite recall 1.0. It verifies integration and must
not be presented as a competitive neural benchmark.

## Verification

- Full implementation suite: **62 passed**, including eight offline browser tests.
- Follow-up report suite: **15 passed**, including checks for the updated inspector,
  interval controls, scientific exports and task-specific visibility.
- Independent audit: all 120 completed job manifests, exact OOF membership and
  labels; 12 representative checkpoint reloads reproduced outputs within recorded
  numerical tolerances.
- Full-data Chromium checks: all nine map combinations, five-fold contexts,
  original-ID search, exact cliff cohort/coverage, interval switching, property
  cutoff, classification sets/risk curves and CSV export. No page errors or
  external requests; mobile viewport has no horizontal overflow.
- Scientific outcome figure rendered and inspected. Historical report checksums
  still match `legacy/openadmet_report/REPORT_MANIFEST.md`.
- Wheel resource validation includes all three report JavaScript assets and schemas.

Detailed evidence and reproducible commands are linked from the
[outcome document](docs/real_cv_uq_outcome_2026_09_09.md).
The current HTML is about 109 MiB because it embeds all records, structures and
nine maps. Measured initial Chromium loads took 28–35 seconds on this host. This is usable offline desktop delivery but
not yet a scalable frontend architecture.

## Principles retained

1. Retain raw molecule and fold observations; do not substitute mean bars for
   their distributions.
2. Carry model, endpoint, units, transform, preprocessing, split, partition,
   repeat and fold context with every result.
3. Compare matching molecules and roles. Do not treat folds as independent
   experimental replicates or repeated OOF rows as new chemicals.
4. Keep fitting, probability calibration, conformal calibration and outer
   validation disjoint. All fitting preprocessing is learned inside its role.
5. Keep exploration separate from model/threshold selection and test claims.
6. Evaluate UQ for error ranking, coverage and sharpness with declared semantics.
7. Preserve input, software, model, calibration and output provenance.

## Next evidence-driven steps

1. Freeze a repeated-seed extension and quantify stability of model/UQ differences
   with a dependence-aware inferential plan. Add nested tuning only with an
   explicit compute budget and untouched outer evaluation.
2. Run a full neural comparison with matching registered roles and adequate fixed
   training budgets. The CheMeleon adapter must be exercised before claiming support
   validated by a benchmark.
3. Investigate cliff failures through training-only chemical signals, and evaluate
   any proposed calibration rule on new declared splits or external chemistry.
   Do not optimize a method on the post-hoc cliff observations in this report.
4. Define the final deployment model's training/calibration policy, fit it from the
   declared population, and validate prospective SMILES inference independently.
5. Add lazy loading or separate report assets, then measure load/interaction time
   at larger endpoint and repeat counts. Complete independent GTM stability checks.
6. Resolve dataset/publication licenses, overlap in a measured multi-endpoint
   demonstration, prospective/temporal data, a release baseline and hosted CI.

The historical OpenADMET workspace and reports remain under `legacy/` and
`outputs/current_benchmark/`. Historical submission metadata includes identity-like
fields and remains private. Publication requires its own explicit scope and review.

## Research-ready criterion

A stronger benchmark claim requires a reproducible complete protocol, matched
model coverage, validated leakage boundaries, UQ baselines assessed under chemical
shift, declared confirmatory cohorts and inference, untouched evaluation choices,
a clean-environment reproduction of the full run, report checks, and recorded
versions/licenses. The completed development run advances these requirements but
does not by itself satisfy all of them.
