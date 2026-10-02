# Research assessment and delivery plan

Reviewed 2026-09-08. Scope: maintained package, evaluation contract, report,
tests, supplied status and references, and primary literature/documentation.
Historical submissions and reports remain preserved. This is an engineering
and methods review, not a systematic literature review or a new benchmark.

## Product decision

Build a **model-neutral molecular diagnostics workbench**: validated prediction
tables in; auditable fold results, chemical cohorts and portable reports out.
The first user is a cheminformatician with prepared predictions. Keep the CLI
and offline report as the initial product. The training arena is one producer,
and external model frameworks should use the same evaluation contract.

The differentiator is connecting prediction failures to chemistry, uncertainty
and experimental context. A larger model leaderboard or another training
framework would not resolve the scientific gaps in the current evidence.

## What the research changes

| Evidence | Implication for this project | Delivery decision |
| --- | --- | --- |
| TDC provides a defined ADMET benchmark interface and scaffold-based partitions. [Official benchmark specification](https://tdcommons.ai/benchmark/admet_group/overview/) | Reuse recognized datasets and preserve official splits when claiming comparability. Custom splits answer a different question and need their own labels. | Add a TDC input adapter after the study-to-prediction contract is enforced. Do not call custom repeated holdouts TDC leaderboard results. |
| Coverage bias persists within the limited chemical space of a dataset. Scaffold separation alone does not establish broad extrapolation. [Coverage bias in small molecule machine learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC11718084/) | Report original-space chemical distance and coverage; do not interpret empty map regions as evidence of novelty. | Next chemical feature: saved maximum train-set Morgan Tanimoto similarity, scaffold novelty and support counts. Compare PCA/t-SNE only as exploratory projections. |
| Activity-cliff evaluation reveals failures hidden by aggregate performance. [MoleculeACE paper](https://pubs.acs.org/doi/10.1021/acs.jcim.2c01073) | Keep molecule-level errors and evaluate predeclared cliff definitions separately. | Integrate the authors' benchmark definitions instead of inventing a threshold from whichever selection looks worst. |
| Hyperparameter selection and performance evaluation need separate data roles. [scikit-learn nested evaluation example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html) | Shared splits must be saved before model fitting, with any tuning inside outer training data. | First slice supports fixed models and saved repeated holdouts. Nested tuning remains an explicit next stage. |
| Classical split-conformal coverage relies on suitable exchangeability assumptions and is marginal, not automatically conditional on chemistry. [Distribution-free predictive inference for regression](https://www.stat.berkeley.edu/~ryantibs/papers/conformal.pdf) | A scaffold-shifted test cannot inherit an unconditional claim of nominal coverage from ordinary conformal calibration. | Reserve calibration data now. Next add a simple absolute-residual split-conformal baseline with finite-sample quantiles and report observed coverage by shift cohort. |
| Chemprop already exposes ensemble uncertainty and calibration methods. [Chemprop prediction documentation](https://chemprop.readthedocs.io/en/latest/tutorial/cli/predict.html) | Reuse established producers where possible while verifying what each uncertainty field means. | Adapt score, standard deviation, interval and quantile semantics explicitly; do not assume every uncertainty field is a standard deviation. |
| Demšar's familiar Wilcoxon/Friedman recommendations address comparisons across datasets. [Original JMLR paper](https://www.jmlr.org/papers/v7/demsar06a.html) | Reusing those tests across overlapping folds of one endpoint does not establish independent blocks. Multiplicity correction cannot repair the underlying sampling assumptions. | Existing p-values are retained for compatibility but explicitly marked exploratory. A dependence-aware inferential protocol must precede confirmatory claims. |
| RDKit distinguishes canonicalization from fragment, charge and tautomer standardization. [Official MolStandardize API](https://www.rdkit.org/docs/source/rdkit.Chem.MolStandardize.rdMolStandardize.html) | Changing the identity policy can change duplicate detection and apparent leakage. | Version the implemented canonical-isomeric policy; do not imply it performs salt removal or tautomer unification. A parent-structure policy needs a separate study version and chemistry review. |
| WCAG describes keyboard access, visible focus, descriptive labels and status messages. [W3C guidance](https://www.w3.org/WAI/WCAG22/quickref/) | A hover/lasso-only interface excludes keyboard users and obscures empty states. | Add keyboard molecule inspection, focus/skip controls, live status, explicit coverage and empty-state explanations. This is improvement toward accessibility, not a WCAG certification. |

These recommendations are inferences from the sources and the inspected code.
The current benchmark data has not been rerun or retrospectively promoted to a
stronger evidence category.

Verification: the final suite passes 48 tests in a fresh locked environment.
See [the verification record](verification_2026_09_08.md) for exact scope and
full-report measurements.

## Confirmed issues and fixes in this iteration

The original suite passed 27 tests. Inspection confirmed the prepared-prediction
boundary, context-aware validation and report, but revealed additional issues:

- The report's median-error card and selection distribution pooled different
  endpoint scales. They now use the diagnostics endpoint; total cohort counts
  still span selected endpoints.
- Classification curves advanced through tied scores one record at a time.
  Threshold ties now enter together, and a one-class cohort displays an
  explanation instead of a misleading discrimination curve.
- One- and two-molecule inputs could fail the t-SNE/PCA pipeline. Tiny inputs now
  receive disclosed placeholder coordinates; PCA is also available explicitly.
- Clearing plot HTML for empty diagnostics could interfere with subsequent
  redraws. Empty charts now stay in the Plotly lifecycle.
- The report's model dropdown required a model common to every endpoint, hiding
  useful partial coverage. It now exposes the union and reports per-model
  molecule support in the active context.
- Plot controls and source text were buried in a Python literal. HTML, CSS and
  JavaScript are now packaged separately while the generated report remains
  one offline file.
- The evaluation CLI resolved a relative `--output-dir` against the config
  directory. Explicit CLI paths now resolve against the working directory;
  YAML paths retain config-relative semantics.
- There was no protocol registration command. `admet-study` now validates,
  freezes and verifies shared repeated holdouts and dataset hashes.

## Build sequence and acceptance gates

| Sequence | Deliverable | Acceptance gate |
| --- | --- | --- |
| Implemented foundation | Study schema, preflight, non-overwriting registration, run plan, shared train/calibration/validation/test assignments, content hashes | Row order does not change partitions; no chemical/group crosses a partition in a repeat; fixed test membership across repeats; classification support and modified-artifact rejection tested. |
| Implemented report iteration | Separate assets, coverage, signed view recipes, CSV export, scoped error summaries, keyboard inspection, PCA option, clearer evidence labels | Offline browser tests cover save/load, lasso, CSV rows, empty recovery, tied classification curves and desktop/mobile overflow. |
| Next: bind actual runs | Match each canonical prediction bundle to a registration and planned model/repeat; persist artifact hashes | Exact expected sample, chemical, endpoint, partition, transform and model coverage; reject unregistered or incomplete artifacts. No adapter may generate its own split. |
| Next: repeated training and UQ | Fixed descriptor/dummy baseline, tree baseline, then graph and pretrained baseline; seed disagreement and split conformal | Fit only training rows, calibrate only calibration rows, score identical validation rows, capture every seed and failed fit. Nested search, when added, remains inside outer training. |
| Next: chemical shift | Original-fingerprint nearest neighbours, scaffold novelty, target/class strata, predefined cliff cohorts | Distances computed against the appropriate training fold only; minimum unique-molecule support; descriptive cohort intervals with declared resampling units. |
| Next: real overlapping endpoints | Select an assay panel with multiple measurements per compound | Verify source data license, units, missing-label rules and canonical intersections; never impute missing outcomes to create overlap. |
| Next: scale | External local assets and lazy endpoint/structure loading, cached features | Measure at 1k, 10k and 100k observations; compare startup time, redraw latency, bytes and browser memory to agreed hardware budgets. |
| Release gate | Clean locked environment, CI, governance, release metadata and reproducible evidence bundle | CI actually passes in the target environment; dataset publication rights and redaction audited; independent scientific review approves claim wording. |

The new registry deliberately stops at development evidence. Its declared CPU
budget, model names and UQ methods are protocol metadata, not a scheduler or
implemented UQ producer. The run index contains planned runs, not completed
model artifacts. Test locking is a recorded policy, not filesystem access
control. Hashes detect modifications relative to a manifest; they are not a
cryptographic attestation against an adversary who can rewrite that manifest.

## Dataset and model choices

Keep KSOL and BBBP as existing smoke tests. For real endpoint overlap, inspect
an assay panel such as Tox21 before downloading or integrating it; its usefulness
is a candidate decision, not a verified licensing conclusion in this review.
Use the synthetic overlapping tutorial to test software now. Synthetic outcomes
must never appear in a scientific model leaderboard.

Begin a real benchmark with a dummy baseline, a descriptor or fingerprint tree
baseline, a graph model and a pretrained representation. Select model families
and fixed settings before evaluating the locked test. More models are valuable
only after saved shared folds, failure logging and runtime accounting work.
Existing Chemprop adapters make reuse more practical than building a new neural
training stack.

## Product priorities and non-goals

The default report should answer: what evidence is present, what context is
active, where the model fails, and how to reproduce that selection. Advanced
statistics and raw tables belong behind disclosure controls. Display full-fold
metrics and filtered molecule diagnostics with their different scopes stated.

Keep the offline artifact for small studies. External-assets delivery should
be a second mode after actual load measurements. A hosted dashboard, user
accounts, cloud GPU scheduling and external collaboration integrations can
follow a reproducible local product; none is a prerequisite for the evaluator.

## Remaining study-owner decisions

Real confirmatory work still needs authoritative endpoint units, acceptable
chemical standardization, practical effect thresholds, data licenses, a final
holdout policy and a compute budget. This iteration uses explicitly synthetic
values to make the workflow reviewable without inventing those scientific
choices. It does not authorize publication or transmit private submissions.
