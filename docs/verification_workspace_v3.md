# Workspace verification — 2026-09-08

This record covers the task tabs, chemical map atlas, GTM, molecule inspector,
regression thresholds and fold/UQ views added after the original refinement.

## Automated checks

- Full suite: **54 passed in 56.57 s**, including 7 offline Chromium tests,
  using the locked Python 3.12 environment from the previous pass.
- After layout/legend refinements: all 7 browser tests passed in 47.67 s;
  all 11 chemical-report/GTM unit tests passed in 4.78 s.
- Browser checks cover all 9 map combinations; regression/classification/merged
  tabs; correct confusion counts; threshold equality boundaries and invalid
  cutoffs; preservation of saved metrics; GTM node selection and view round trips;
  paginated molecule search; missing/empty cohort recovery; 5/10-fold plots and
  uncertainty curves; nominal interval coverage lines; and disjoint synthetic
  fit/calibration/evaluation roles.
- GTM tests cover normalized responsibilities, bounded latent coordinates and
  entropy, deterministic projection without refitting, regularized EM objective
  behavior, low density for extreme outliers, invalid inputs, reference selection
  and cached geometry independent of outcomes and repeated observation rows.
- JavaScript syntax checks passed. Wheel build and asset checks include the new
  `workspace.js` resource as well as existing HTML/CSS/JS and JSON schemas.
- All three historical report checksums still match their manifest. The delivered
  real-data HTML contains no absolute workspace path.

The installed optional neural-model adapters were not exercised in this pass.
The NumPy/SciPy GTM has not been independently benchmarked against `ugtm`.

## Real-data browser measurements

Output: `outputs/workspace_v3/chemical_model_diagnostics.html`.
Sources: saved KSOL and BBBP 50-epoch screening predictions; no retraining.

| Measurement | Result |
| --- | ---: |
| Source observations | 9,337 |
| Included observations | 6,000 |
| Included unique chemicals | 5,961 |
| Deterministic reference molecules | 2,000 |
| Saved representation/technique combinations | 9 |
| Initial regression validation observations | 1,025 |
| Classification validation observations | 328 |
| Classification task inspector matches | 1,191 |
| ID search for `ifoxetine` | 1 matching chemical |
| Browser page errors | 0 |
| Mobile viewport/document width | 390 / 390 px |

The measured initial readiness time was approximately 2.7 seconds at 1440×1000
in local headless Chromium. The HTML is approximately 54 MiB. This is one smoke
measurement, not a portable performance budget or memory benchmark. The final
JSON/screenshot evidence is under `outputs/workspace_v3/`.

All three GTMs reached their relative-objective tolerance: 32 Morgan, 87 RDKit,
91 combined updates. Convergence does not establish an optimal or chemically
faithful map; preprocessing discards a material amount of feature variation.
See `docs/chemical_workspace_v3.md` for the retained-variance values and caveats.

## Synthetic fold/UQ exercise

Output: `outputs/workspace_v3/fold_uq_demo/fold_uq_demo.html`.
The reproducible producer is `scripts/build_fold_uq_demo.py`.

- 80 unique tutorial chemicals, artificial regression and classification labels.
- Both 5-fold and 10-fold schemes: 320 endpoint observations, 118 metric rows,
  1,200 assignment-role rows. Undefined ROC/PR metrics for single-class folds
  are omitted; probability diagnostics remain available.
- Five separately seeded forests fit inside each fold. The score is dispersion
  across forest means; regression intervals use disjoint calibration data and
  a finite-sample corrected 90% split-conformal absolute-error quantile.
- Browser validation verifies exactly 5 or 10 shared-coordinate panels and
  retention curves for the selected scheme, and the declared 0.9 coverage line.
- The real benchmark has no submitted UQ: its explicit missing-data state was
  checked. Synthetic uncertainty was not attached to real predictions.

## Visual corrections

Desktop, 390-pixel mobile, classification, GTM pies, 10-fold maps and UQ screenshots
were reviewed. The review led to short fold labels with full context retained on
hover, horizontal uncertainty legends, wider chemical-map controls, an explicit
GTM band legend, and full-width fold maps. A further regression check asserts
that Plotly resizes to the widened container after changing fold display.

This completes the requested workspace refinement. It does not complete real
multi-fold UQ experiments, deployment infrastructure, independent map validation,
training/registration integration or confirmatory statistical inference.
