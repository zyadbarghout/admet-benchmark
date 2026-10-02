# Current state, risks, and implementation roadmap

Updated 2026-09-08. See the newer
[research and delivery plan](research_and_delivery_plan.md) for source-backed
findings, the implemented registration/report iteration, and its acceptance
gates. The original 27-test baseline is preserved as historical evidence.

## Intended product

The maintained package is a model-neutral analysis layer for molecular machine
learning. A cheminformatician supplies regression or classification predictions
with molecule identity and experiment context. The package validates those
records, calculates fold- and molecule-aware statistics, and connects model
error and uncertainty to structures, motifs, and chemical-space regions.

The original OpenADMET challenge workspace is an immutable historical producer
and benchmark case, not the runtime of the new product. Its scripts, notebooks,
submission corpus, leaderboards, and reports remain under
`legacy/openadmet_report/`; the generated current reports remain under
`outputs/current_benchmark/`.

## Reliable foundation now

- A versioned YAML and prediction-table boundary supports external models,
  regression and classification, multiple endpoints, preprocessing profiles,
  split strategies, validation/test partitions, repeats, and folds.
- Schema and semantic checks cover chemical identity, invalid structures,
  missing or unequal model coverage, leakage between validation and test,
  target transforms, classification probabilities, uncertainty direction, and
  prediction-interval completeness.
- Evaluation output retains molecule-level predictions and every repeat/fold
  metric. Confidence intervals use molecule bootstrap resampling; repeated
  model comparisons are paired and multiplicity-adjusted when enough repeated
  results exist.
- The static evaluation report uses raw fold points, violins, embedded boxes,
  and distributions instead of replacing observations with mean bars.
- Chemical Model Diagnostics combines endpoints on a shared molecular map,
  provides structures, motif/search filters, lasso selection, validation/test
  and repeat/fold controls, task-specific diagnostics, uncertainty views, and
  model comparison.
- The chemical report is validation-first and uses grouped controls, jump
  navigation, responsive layouts, and collapsed raw/provenance sections. Its
  self-contained provenance uses run names and artifact signatures instead of
  leaking absolute local paths.
- Interactive sampling retains complete canonical-molecule histories across
  endpoints and folds. Repeated observations are only spread by a small visual
  offset; the chemical coordinates themselves are not changed.

## Problem assessment

| Priority | Problem | Why it matters | Current safeguard | Required resolution |
| --- | --- | --- | --- | --- |
| Critical for legacy reuse | The historical leaderboard performs an inner join and can score incomplete submissions. | A partial prediction file can receive an invalidly optimistic score. | The legacy code is isolated and labelled as historical; the new evaluator audits coverage. | Never use the old scorer for new claims. If historical scores must be regenerated, add exact row/endpoint coverage checks in a compatibility wrapper without rewriting the archived script. |
| High | The preserved KSOL and BBBP screening runs contain one validation and one test result per model, not repeated outer evaluations. | A one-point “violin” is descriptive only; bootstrap resampling does not create independent experimental repeats. | Reports show the actual points and avoid inferential claims when repeats are absent. | Add a first-class repeated/nested split producer and rerun every model on identical saved folds. |
| High | UQ can be ingested and diagnosed, but the package does not yet generate or calibrate uncertainty methods. | Ranking error is necessary but does not establish calibrated probabilistic uncertainty. | Method-neutral uncertainty/interval fields, completeness checks, ranking correlations, coverage, width, and retention curves exist. | Implement at least ensemble/disagreement and split-conformal baselines, then add calibration error, sharpness, proper scoring rules where applicable, and cohort-specific coverage. |
| High | Chemical cohorts are interactive and descriptive, without support thresholds or corrected cohort inference. | Searching many motifs or selected regions can produce unstable or post-selection findings. | Filters never feed back into model selection; selected counts and actual molecules remain visible. | Add preregistered cohort definitions, minimum molecule support, molecule-bootstrap intervals for cohort effects, and multiplicity control or external validation. |
| High | Distribution-shift and property-cliff benchmarks are not yet implemented. | UQ and error behavior in random splits does not establish usefulness on novel chemistry. | Split strategy is explicit and t-SNE is labelled exploratory. | Add scaffold/feature shift, target-label shift, activity/property cliffs, temporal shift where timestamps are real, and enriched-error cohorts. |
| High | Training adapters are not yet bound to registered study assignments. | A saved protocol cannot enforce external model behavior by itself. | New preflight/freeze/verify commands save hashed study inputs and shared repeated holdouts with separate calibration data. | Verify each completed prediction artifact against the registration and persist the binding in a completed-run index. |
| Medium | Self-contained chemical reports are large; the current two-endpoint report is about 31 MB. | Initial load, sharing, and browser memory become limiting as endpoints/folds grow. | Structures are deduplicated, a molecule-coherent point limit is disclosed, and dense tables are collapsed. | Add a second assets-directory mode, cached scalable embeddings, and optional lazy loading while preserving a portable offline mode. |
| Medium | t-SNE is stochastic and distorts global distances. | Empty regions and inter-cluster distances are not quantitative evidence of model coverage. | Fixed seeds, explicit method labelling, structures, motif filters, and raw error views keep it an exploratory map. | PCA is now available. Add UMAP, nearest-neighbour diagnostics in the original fingerprint space, and embedding stability checks. |
| Medium | Cross-endpoint molecule ranking is empty when endpoint datasets share no canonical molecules, as in the current KSOL/BBBP sample. | The feature is scientifically useful only for multi-endpoint measurements on overlapping chemistry. | The report says when no molecule is shared and never manufactures a match. | A coverage/overlap panel and synthetic overlapping tutorial now exist. Add a real licensed overlapping multi-endpoint benchmark. |
| Medium | Exact preprocessing/split names can legitimately differ by endpoint. | A global exact filter previously made a checked endpoint disappear silently. | The default now shows all endpoint contexts; exact choices come only from the active diagnostics endpoint, and the summary counts endpoints with data. | In the scalable report, represent context as an explicit endpoint-to-context mapping and show it beside cross-endpoint comparisons. |
| Low | Several old dashboard paths remain as compatibility views with different interaction and asset conventions. | Users may not know which report is authoritative. | Documentation names `admet-chemical-report` as the maintained chemical-analysis entry point. | Deprecate compatibility commands gradually after reproducing every essential historical view in the new report. |

## Ordered implementation plan

### 1. Research-grade experiment production

Development registration and shared repeated holdouts now exist. Bind training
adapters and add nested tuning within outer training data. Save
fold assignments before model fitting, enforce identical molecule coverage for
paired models, and distinguish screening, validation, and locked confirmatory
runs in metadata and report language.

### 2. UQ baselines and evaluation

Implement ensemble standard deviation/disagreement and split-conformal
intervals first because they provide complementary ranking and coverage
baselines without tying the evaluator to one model family. Benchmark them under
in-distribution and shifted cohorts using retention curves, rank correlation,
coverage, interval width/sharpness, and calibration diagnostics.

### 3. Chemical shift and cohort layer

Create saved cohort tables for scaffold novelty, nearest-neighbour distance,
property cliffs, target ranges/classes, and curated SMARTS. Report molecule
support, effect distributions, bootstrap intervals, and corrected comparisons.
Interactive selection remains exploratory and is clearly separated from these
predeclared cohort results.

### 4. Scalable report delivery

Keep the current self-contained HTML for small studies. Add an assets mode for
large studies with cached coordinates, columnar records, lazy structure loading,
and endpoint-level panels. Preserve validation-first defaults and progressive
disclosure in both modes.

### 5. Benchmark and publication audit

Run classical, graph, and foundation-model baselines over the same registered
folds. Compare raw repeated results with paired effects and practical-significance
thresholds. Verify manifests, rerun from a clean environment, and publish the
protocol and complete artifacts—not only aggregate tables.

## Definition of a research-ready release

- At least one regression, one classification, and one overlapping
  multi-endpoint benchmark use saved repeated folds.
- Every compared model predicts the same molecules within each fold or is
  explicitly excluded from paired inference.
- At least two UQ baselines are evaluated in-distribution and under chemical
  shift, including cohort coverage and retention behavior.
- Chemical cohort claims satisfy declared support and multiplicity rules.
- A complete study can be reproduced from a registry/configuration and produces
  canonical bundles with hashes and software versions.
- Both offline and scalable reports pass interaction, accessibility, visual,
  and performance checks without exposing private paths or data.

The supplied literature and classification-code references are indexed in
`references/README.md` and should remain design constraints for each work
package.
