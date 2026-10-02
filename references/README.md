# Research references

The product and statistical design are guided by:

1. [Zyad thesis ADMET classification code](https://github.com/volkamerlab/Zyad_thesis/tree/main/ADMET_classification) — classification diagnostics, imbalance-aware metrics, thresholds, ROC/PR analysis, and pairwise performance distributions.
2. [Error models for uncertainty quantification in molecular machine learning](https://chemrxiv.org/doi/pdf/10.26434/chemrxiv.15008122/v1?redirectToLatest=false) — uncertainty as a ranking of likely prediction error and the complementarity of distance and ensemble signals.
3. [Uncertainty Quantification in Molecular Machine Learning for Property Predictions under Data Shifts](https://pmc.ncbi.nlm.nih.gov/articles/PMC12848971/) — evaluation of UQ under feature, label, property-cliff, and error shifts.
4. [Practically significant method comparison protocols for machine learning in small molecule drug discovery](https://chemrxiv.org/doi/pdf/10.26434/chemrxiv-2024-6dbwv-v2?redirectToLatest=false) — repeated paired comparisons, distributions, effect sizes, multiplicity control, and practical significance.

The locally supplied PDF for reference 4 is retained as
`chemrxiv-2024-6dbwv-v2.pdf`. Large reference files are intentionally ignored by
source control.


## Research review, 2026-09-08

Primary sources and their implementation implications are recorded in
[`docs/research_and_delivery_plan.md`](../docs/research_and_delivery_plan.md),
including coverage bias, MoleculeACE, TDC, conformal inference, fold dependence,
RDKit standardization and accessible report interactions.
