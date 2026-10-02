# Chemical workspace: maps, task diagnostics, thresholds and folds

Implemented and reviewed 2026-09-08. This extends the prepared-prediction report;
it does not upgrade the scientific evidence of the saved screening runs.

## Open the delivered reports

- `outputs/workspace_v3/chemical_model_diagnostics.html`: real saved KSOL and
  BBBP predictions, nine chemical maps, 6,000 of 9,337 observations spanning
  5,961 unique molecules. No models were retrained for this report.
- `outputs/workspace_v3/fold_uq_demo/fold_uq_demo.html`: **synthetic** targets on
  80 tutorial molecules, actual fitted five-forest ensembles, both 5-fold and
  10-fold schemes, uncertainty scores and regression intervals. This is a
  demonstration, not a measured ADMET benchmark.

Both reports work locally without a server or network. Map fitting happens in
Python at generation time. Representation/technique switches in the browser
select saved coordinates; they never refit against the currently selected errors.

## Workflow and controls

Regression and classification have separate tabs. **Merged exploration** is
optional; diagnostics still refer to one explicitly selected endpoint so that
incompatible units and tasks are not silently averaged. Classification includes
probabilities versus labels, reliability bins with counts, ROC and PR curves,
a confusion matrix using saved threshold policies, probability distributions,
Brier score and log loss. These scores describe probability quality; neither
Brier nor log loss is a pure calibration statistic. Single-class selections
have an explicit unavailable discrimination state. See the
[scikit-learn metrics documentation](https://scikit-learn.org/stable/modules/model_evaluation.html).

The molecule inspector searches every molecule within the selected scope, with
8 results per page. Search original/canonical SMILES, molecule IDs or sample IDs;
there is no 500-result cutoff. Search is case-insensitive text matching, not
SMILES equivalence or arbitrary substructure execution. Existing precomputed
SMARTS filters remain separate. Repeated observations are grouped by chemical
identity, with a representative record and its endpoint/fold shown on inspection.
The report's overall generation sample still limits which molecules are present.

Regression thresholds define low `< lower`, medium `[lower, upper)`, high
`>= upper` bands for either absolute error or signed residual. The evaluation
transform/scale is shown. Default cutoffs 0.25 and 0.75 are editable examples,
not endorsed assay tolerances. They affect plots and GTM pies, never predictions,
saved metrics, or model selection. Outcome-derived exploratory cutoffs must not
be presented as a prospectively validated decision policy.

Choose fold overlays, fold colours, or shared-coordinate small multiples. The
fold evidence panel retains every saved metric and available interval. Molecule
filters change error distributions; the full-fold metric points retain their
original cohort. Small multiples show at most 10 contexts; use the split/repeat/
fold selectors when there are more. A chemical has one underlying coordinate
across tasks/folds; slight radial offsets make coincident observations selectable
on the main map. These offsets do not imply chemical differences.

UQ views show submitted score versus error, per-fold error-retention curves with
ties admitted together, empirical regression interval coverage and interval
widths. A nominal line appears only when interval confidence is declared. Fold
identity includes preprocessing, split strategy, partition and repeat. Filtered
coverage is descriptive and can differ from nominal coverage. Forest dispersion
is a heuristic; it is not automatically calibrated total predictive uncertainty.
No classification prediction-set coverage is claimed.

View recipes version 2 retain the task, map, cutoffs, context, molecule query,
cohort IDs and selected GTM node, bound to the original report's payload hash.
Older recipes should be opened with their original report. This hash is an
integrity binding, not a cryptographic signature establishing an author's identity.

## GTM method and chemical-data choices

The implementation follows the Gaussian generative topographic mapping model:
a regularized radial-basis manifold constrains an equal-prior Gaussian mixture;
EM estimates weights and noise precision. The posterior over map nodes yields
mean coordinates, modal assignments, entropy and log density. Fitting and
projection are separate. These are the model's latent-space quantities, not
error bars on a property prediction. The original paper also discusses other
observation distributions, so direct Gaussian modelling of binary fingerprints
should not be treated as a Bernoulli model.
[Bishop, Svensén and Williams, 1998](https://www.microsoft.com/en-us/research/wp-content/uploads/1998/01/bishop-gtm-ncomp-98.pdf).

The chemistry-focused `ugtm` implementation documents separate map resolution,
RBF resolution, width and regularization controls, together with responsibility
based projection and landscapes. It is a useful independent implementation for
future benchmark comparison; this project currently contains its own NumPy/SciPy
reference implementation and does not claim numerical equivalence to `ugtm`.
[ugtm paper](https://openresearchsoftware.metajnl.com/articles/10.5334/jors.235),
[ugtm source](https://github.com/hagax8/ugtm).

The implemented choices are:

1. Generate Morgan radius-2, 1,024-bit fingerprints; full RDKit 2D descriptors;
   or combined blocks. RDKit's averaged Ipc avoids extreme extensive values.
2. For continuous maps, fit descriptor median imputation and standardization on
   reference chemistry and remove reference-constant descriptors. Keep Morgan
   bits unscaled per bit. Scale each feature block by its total reference
   variance, then fit up to 32 PCA components on that reference and apply one
   scalar variance normalization. This prevents the combined block sizes from
   determining their relative weight by accident. It remains a representation
   choice and should be assessed for the intended chemical domain.
3. Fit Gaussian GTM on these **continuous PCA scores**. This is an explicit
   approximation for Morgan chemistry, not a Tanimoto GTM or a binary likelihood.
   In contrast, Morgan t-SNE uses original fingerprint Tanimoto distances.
   Descriptor/combined t-SNE uses Euclidean distances in continuous PCA space.
4. Use a 15×15 latent grid, 5×5 RBF centres, width 1 centre-spacing unit,
   regularization 0.01, at most 150 EM updates, relative objective tolerance
   1e-5, and PCA initialization. The fit records its regularized objective,
   precision, iterations and stop reason. Numerical objective decreases trigger
   rollback. These defaults have not been optimized against assay performance.
5. Prefer supplied development chemistry; otherwise validation chemistry; with
   fewer than four molecules in either pool, explicitly disclose an all-chemistry
   exploratory reference. Fit at most 2,000 deterministically sampled unique
   reference molecules. Reference membership, preprocessing summaries, settings
   and dependency versions participate in provenance/cache identity. Labels,
   predictions, row order and repeated fold observations do not determine maps.
6. Keep the fitted atlas shared across folds. This is a pooled reference atlas,
   **not a fold-specific applicability-domain estimate**. Development chemistry
   pooled from folds may include another fold's held-out molecules. t-SNE is
   transductive and offers no claimed out-of-sample transform.

The delivered GTMs converged in 32 (Morgan), 87 (RDKit) and 91 (combined) updates.
The 32-component preprocessing retains about 50.6%, 75.6% and 57.2% of reference
variance respectively. Thus especially the Morgan/combined maps discard a
material amount of feature variation; nearby projected points need not be
chemically interchangeable. These percentages concern preprocessing, not the
fidelity of the final two-dimensional GTM. They are exposed in map metadata.

Soft landscapes sum node responsibilities and optionally display weighted mean
molecule error or submitted uncertainty above a minimum soft support. Repeated
observations first reduce to one median value per chemical. Error-band pies use
**modal node membership**, counting each chemical once, with square-root/log
count scaling. Pies are an intentionally different aggregation from soft maps.
Clicking a node selects its modal members. The original local `gtm_ml.html`
inspired this interaction; none of its embedded molecules or labels were imported.

Likelihood can support GTM applicability-domain research, but useful cutoffs need
separate validation. This release stores reference log density and entropy and
does not claim that either predicts ADMET confidence.
[GTM applicability-domain study](https://pubmed.ncbi.nlm.nih.gov/27490381/).

## Reproduce

```sh
admet-chemical-report outputs/current_benchmark/ksol outputs/current_benchmark/bbbp_50epoch \
  --output outputs/workspace_v3/chemical_model_diagnostics.html --max-points 6000 \
  --map-methods tsne gtm pca \
  --representations morgan rdkit_descriptors morgan_rdkit \
  --map-cache outputs/workspace_v3/map_cache
python scripts/build_fold_uq_demo.py
RUN_BROWSER_TESTS=1 pytest -q
```

The standalone full report is about 54 MiB because it embeds structures, Plotly
and all node responsibilities. It is appropriate for this sampled local example;
a much larger deployment needs binary/external assets, cached server-side map
jobs and lazy endpoint loading. Saving fitted feature transformers and GTM
weights for new batch projection is also a next step: the current HTML/cache
stores this cohort's projections, not a deployable prediction service.

## What remains before a scientific benchmark

Bind completed models/predictions to the frozen study contract; fit all transforms,
tuning and UQ calibration inside the relevant training boundaries; obtain real
5/10-fold and shifted-domain predictions with uncertainty; validate threshold
policies on development data; and evaluate stability across map settings and
held-out reference chemistry. Compare the GTM implementation with an established
implementation and measure neighbourhood preservation before making map-based
scientific claims. Do not treat this UI or the synthetic demo as completing those
studies.

The demo uses a disjoint calibration split and the finite-sample corrected
absolute-error quantile for 90% split-conformal regression intervals. Coverage
is marginal under exchangeability, not guaranteed for every fold or subgroup;
chemical distribution shifts require additional evaluation.
[Angelopoulos and Bates tutorial](https://arxiv.org/abs/2107.07511).
