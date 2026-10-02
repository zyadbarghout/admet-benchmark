# Measured cross-validation and UQ outcome

Evaluated 2026-09-09. This is a completed **development benchmark** on measured
KSOL and BBBP data. It supports comparative diagnostics, not a confirmatory
winner or a final deployment-model claim.

## Delivered result

Open the [interactive report](../outputs/real_cv_uq_2026_09_09/chemical_model_diagnostics.html),
read the [reproduction guide](real_cv_uq.md), or inspect the
[pooled descriptive metrics](../outputs/real_cv_uq_2026_09_09/pooled_oof_outcomes.csv).
The report includes all 26,788 endpoint/representation/split/fold records, each
with three model predictions, covering 6,697 chemicals. Its nine chemical maps
are display atlases; training applicability uses original fingerprints against
the individual fold's fitting population.

- 120 classical endpoint/split/representation/model/fold jobs completed:
  five outer folds, one repeat, molecule and scaffold splits, Morgan and full
  RDKit features, linear/random-forest/extra-trees models.
- 440 estimators were fit. Tree methods each use five independently seeded
  150-tree estimators; an ensemble member is not a validation fold.
- 80,364 out-of-fold prediction rows passed membership and artifact audits.
  Each chemical has exactly one OOF prediction per method, split and repeat.
- Regression has normalized and absolute-error conformal intervals.
  Classification has independently calibrated probabilities and
  class-conditional prediction sets, preserving empty and ambiguous sets.
- Applicability, scaffold novelty, post-hoc property cliffs, interval decisions
  around a property cutoff, local SMILES inference, a resumable experiment view,
  an independently evaluated CV+ pilot and a neural adapter are implemented.

The registered snapshot contains 5,128 KSOL and 1,569 BBBP chemicals. BBBP had
1,631 source rows and 1,578 canonical identities; same-label duplicates were
collapsed and nine conflicting identities were quarantined. Source files were
preserved. Historical test structures were checked for identity overlap, and
historical test labels did not enter fitting or calibration.

## What the outcomes show

The random forest below was the fixed default for report inspection. It was
not selected after examining results. Regression metrics and widths use log10
KSOL units; classification PR AUC is average precision. Coverage is a measured
frequency at a 90% nominal setting.

| Endpoint / representation | Split | Point prediction | Interval/set coverage | Interval width / set size |
| --- | --- | --- | ---: | ---: |
| KSOL / Morgan | Molecule | MAE 0.4151 | 90.09% | 1.8320 |
| KSOL / Morgan | Scaffold | MAE 0.4513 | 89.55% | 1.9639 |
| KSOL / RDKit | Molecule | MAE 0.3912 | 90.04% | 1.7186 |
| KSOL / RDKit | Scaffold | MAE 0.4337 | 89.84% | 1.8876 |
| BBBP / Morgan | Molecule | PR AUC 0.9560; Brier 0.0937 | 90.12% | 1.2702 |
| BBBP / Morgan | Scaffold | PR AUC 0.9331; Brier 0.1109 | 87.76% | 1.3499 |
| BBBP / RDKit | Molecule | PR AUC 0.9651; Brier 0.0872 | 89.61% | 1.1517 |
| BBBP / RDKit | Scaffold | PR AUC 0.9592; Brier 0.0984 | 89.04% | 1.2314 |

Scaffold splits generally expose worse point-prediction performance than molecule
splits. RDKit tree models are promising in this fixed experiment; the full table
also includes ExtraTrees and linear baselines. These observations do not establish
statistical superiority, and no hyperparameters or model choices were changed
in response to the outer-fold results.

### Aggregate coverage hides chemical failure

For the RDKit random forest, KSOL normalized-interval coverage is **89.84% overall
on scaffold splits but 68.32% for the 202 property-cliff observations**. Their MAE
is 0.7937, versus 0.4337 overall. On molecule splits, the 415 cliff observations
have 70.12% coverage versus 90.04% overall.

The cliff rule compares an evaluation molecule with a fitting molecule at Morgan
Tanimoto ≥ 0.8 and observed property difference ≥ 1 log10 unit. This is an
endpoint-specific exploratory stress rule. It uses the observed target after
prediction, so it cannot identify a new molecule's cliff status before that
outcome is measured. It is not a predictor, selection criterion or conditional
coverage guarantee.

![Observed coverage per outer fold](../outputs/real_cv_uq_2026_09_09/uq_outcome.png)

Each point is one outer fold. Folds overlap in training population and are not
independent experimental replicates. The dashed line is a nominal reference,
not a confidence bound. The [PDF figure](../outputs/real_cv_uq_2026_09_09/uq_outcome.pdf)
and [underlying summary](../outputs/real_cv_uq_2026_09_09/uq_outcome_summary.json)
are saved separately.

### UQ methods require empirical comparison

For KSOL/RDKit/random forest/molecule, normalized intervals achieve 90.04%
coverage with mean width 1.7186, compared with 90.04% and 1.8524 for the absolute
baseline. Mean interval score improves from 2.5874 to 2.4699 (lower is better).
This is useful empirical evidence for the normalized baseline in this context;
it does not imply uniformly better coverage across chemical cohorts.

Sigmoid calibration also is not uniformly beneficial. BBBP/Morgan/random forest
on molecule splits improves Brier score from 0.0984 to 0.0937. RDKit/random forest
on scaffold splits changes from 0.0976 to 0.0984, a small worsening. Both raw and
calibrated scores are retained so the calibration step cannot conceal this.

The separate CV+ pilot evaluated all 5,128 KSOL observations using five outer
folds and five fresh inner-fold models within each outer training population.
It achieved MAE **0.4130**, coverage **90.50%**, and mean width **1.9303**. The
Morgan split-conformal forest ensemble had MAE 0.4151, coverage 90.09% and width
1.8320. CV+ provides no clear interval-width improvement here. The learners and
training allocations differ, so this is not an isolated interval-method ablation.
CV+ theory also differs from split conformal; a 90% setting is not an exact
worst-case 90% or chemical-conditional guarantee.

## Neural and inference checks

The neural adapter consumed the registered role assignments and completed a
Chemprop smoke test on BBBP molecule-split fold 0, with two members trained for
two fixed epochs. Its 314 outer-validation observations gave ROC AUC 0.6254,
PR AUC 0.8403, Brier 0.1691, set coverage 88.85% and mean set size 1.6561.
Recall was 1.0 but specificity only 0.0533. This validates the training/calibration/
checkpoint path; it is an undertrained integration check, not a competitive
neural result. The adapter supports CheMeleon, but that branch has not been run.

[Example SMILES input](../outputs/real_cv_uq_2026_09_09/query_examples/input.csv),
[KSOL predictions](../outputs/real_cv_uq_2026_09_09/query_examples/KSOL.csv) and
[BBBP predictions](../outputs/real_cv_uq_2026_09_09/query_examples/BBBP.csv) demonstrate
local batch inference with saved fold models. These examples have no attached
outcome labels and do not measure prospective accuracy. Outputs identify model
hash, scale, nominal level, nearest-fitting similarity and interval/set. KSOL
also exports back-transformed endpoints and cutoff decisions. A saved outer-fold
predictor has a limited fitting/calibration population; a full-development
production refit has not been performed.

## Verification and limits

- The implementation suite passed **62 tests**, including eight offline browser
  tests. A subsequent report-focused run passed **15 tests** after the display
  bundle and inspector changes; the final report run also checks task-specific control visibility. Full-data browser evidence is saved under
  [browser verification](../outputs/real_cv_uq_2026_09_09/browser_verification/result.json).
- The independent [outcome audit](../outputs/real_cv_uq_2026_09_09/outcome_audit.json)
  checked all 120 job artifact manifests and exact OOF membership. It reloaded
  12 representative checkpoints and reproduced predictions, intervals and sets
  within recorded CSV precision tolerances.
- Tests cover disjoint 5/10-fold roles, curation, changed-input rejection,
  finite-sample quantiles, interval ordering, model reload/resume, inference and
  CV+ order statistics. The measured sweep used five folds and one repeat;
  ten-fold support was tested rather than fully benchmarked.
- The evaluator reported no validation errors. Report construction preserves
  original searchable IDs through a separate display bundle and does not mutate
  evaluated predictions. Historical report checksums remain unchanged.
- This large standalone HTML embeds all chemistry, all nine maps and all saved
  predictions. It is approximately 109 MiB. It works offline, but desktop use
  remains preferable; measured first-load times were 28–35 seconds on this host.
  A lazy-loading mode is still needed for larger studies.

The next evidence-driven work is repeated-seed evaluation, a fully budgeted
neural comparison with the same roles, and prespecified chemical-shift validation.
Investigate weak cliff coverage using training-only signals and independent
validation before proposing a new calibration rule. Final deployment refitting,
prospective external data, nested HPO, independent GTM validation, licensing for
publication and confirmatory inference remain open. This pass did not tune on
historical tests or publish private artifacts.

The [delivery verification manifest](../outputs/real_cv_uq_2026_09_09/delivery_verification.json)
records artifact hashes and source/report/wheel parity.

Scientific rationale and primary sources are retained in the
[original research proposal](proposals/cv_uq_2026_09_09.md).
