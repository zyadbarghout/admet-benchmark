# Verification record — 2026-09-08

## Completed checks

- Original baseline: 27 tests passed in the pre-existing environment.
- Final suite: **48 passed in 26.17 seconds**, including 5 offline Chromium
  acceptance tests, in a fresh Python 3.12.12 environment created from `uv.lock`.
- Lockfile check: 121 packages resolved; `uv lock --check` passed. The tested
  installation includes the `visual`, `test` and `browser` extras. Optional
  neural training adapters were resolved by the lock but not exercised here.
- Study preflight, freeze and verify passed in both the existing Python 3.12
  environment and the fresh environment. Registration hashes matched:
  `fe9387620f3146e61fa072bdcd454cd1c5704c2368392ce5cb6d0ef2be101571`.
- Both prepared-prediction tutorial configurations evaluated without semantic
  warnings; each produced 32 canonical prediction rows. Regression produced
  40 metric rows and classification produced 104.
- The combined PCA tutorial opened offline with 32 visible validation
  observations and no page errors.
- Wheel build succeeded and contains the report HTML/CSS/JavaScript plus both
  JSON schemas. `scripts/check_wheel_assets.py` verifies resource inclusion.
- All three historical artifact SHA-256 values in
  `legacy/openadmet_report/REPORT_MANIFEST.md` matched.
- The new full report contains no absolute workspace path.

## Full-report browser check

Output: `outputs/refinement_2026_09_08/chemical_model_diagnostics.html`.
Sources: the saved KSOL and BBBP 50-epoch screening runs. No models were retrained.

| Measurement | Result |
| --- | ---: |
| Stored observations | 9,337 |
| Sampled observations | 6,000 |
| Initial visible validation observations | 1,353 |
| Initial unique validation molecules | 1,336 |
| HTML bytes | 32,996,291 |
| Initial counts ready | 1.748 seconds |
| One search update | 0.283 seconds |
| Browser page errors | 0 |

Measured once on this macOS ARM machine in local headless Chromium, using a
1440 × 1000 desktop viewport. The search measurement includes its 180 ms
input debounce. These are smoke measurements, not portable performance
budgets, paint-completion measurements, repeated latency percentiles or a
memory benchmark. The file remains approximately 31.5 MiB; external-assets
and endpoint-level lazy loading are still needed for substantially larger data.

Desktop overview, chemical-space and 390-pixel mobile screenshots were inspected.
They are saved alongside `browser_check.json` in the output directory.

## Refinements triggered by verification

1. Preserved the diagnostics chart lifecycle through empty selections and
   verified restoration of filtered views.
2. Grouped equal classification probabilities at a single threshold and
   suppressed discrimination curves for one-class selections.
3. Restricted pooled error summaries to the active diagnostics endpoint.
4. Corrected Plotly title objects and restored ordinary point opacity when no
   lasso is active; browser checks now assert the axis title and selection state.
5. Corrected CLI `--output-dir` resolution to the working directory. YAML paths
   still resolve relative to their configuration file. An actual subprocess
   regression test checks the output location.

## Limits of this verification

The added GitHub Actions workflow has not executed on a hosted runner; the
repository has no remote or baseline commit. This local run tests Python 3.12
on macOS, not the entire declared Python/platform range. It does not certify
WCAG conformance, malicious-input security, model quality, scientific test
blinding or confirmatory statistical validity. Browser tests exercise a small
synthetic fixture in addition to the separate full-report smoke check.

No remote publication, submission upload or new model training occurred.
Real repeated/nested model execution, registration-to-prediction enforcement,
UQ producers, chemical-shift benchmarks and release governance remain in the
research and delivery plan.
