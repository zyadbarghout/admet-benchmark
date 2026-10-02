"""Build or resume the complete local real-data demonstration.

Run with the models environment: python scripts/build_complete_example.py.
The default includes full-budget neural CV. --classical-only omits the neural
addon explicitly; --validate-only freezes and checks the input protocol only.
"""
from pathlib import Path
import argparse
import json
import html

import numpy as np
import pandas as pd
import yaml

from admet_benchmark.cross_validation import register, run, verify_registration, digest, predict, evaluate_predictions
from admet_benchmark.deployment import refit
from admet_benchmark.chemical_report import generate_chemical_report
from admet_benchmark.tables import read_csv
from evaluate_real_cv import evaluate
from prepare_cv_report_bundle import prepare

REPOSITORY = Path(__file__).resolve().parents[1]


def verify_manifest(directory, filename="manifest.json"):
    manifest = json.loads((directory / filename).read_text())
    for name, expected in {**manifest.get("artifacts", {}), **manifest.get("checkpoints", {})}.items():
        if digest(directory / name) != expected:
            raise ValueError(f"Artifact changed: {directory / name}")
    return manifest


def add_neural_report(root, report_paths, config):
    endpoint = "BBBP"
    frames = []
    for fold in range(config["folds"]):
        directory = root / "neural" / endpoint / f"fold_{fold}"
        verify_manifest(directory)
        frame = read_csv(directory / "predictions.csv")
        observations = read_csv(root / endpoint / "observations.csv")
        assignments = read_csv(root / endpoint / "assignments.csv")
        expected = assignments[(assignments.repeat == 0) & (assignments.fold == fold)
            & (assignments.split_strategy == "molecule") & (assignments.role == "validation")]
        if not frame.sample_id.is_unique or set(frame.sample_id) != set(expected.sample_id):
            raise ValueError("Neural OOF membership mismatch")
        np.testing.assert_allclose(frame.observed, observations.set_index("sample_id").loc[frame.sample_id].observed)
        frames.append(frame)
    neural = pd.concat(frames, ignore_index=True)
    if not neural.sample_id.is_unique:
        raise ValueError("Neural OOF rows must identify each chemical once")
    destination = next(p for p in report_paths if p.name == endpoint)
    classical = read_csv(destination / "predictions.csv")
    names = classical.drop_duplicates("sample_id").set_index("sample_id").molecule_id
    neural["molecule_id"] = neural.sample_id.map(names)
    neural["uncertainty_percentile"] = neural.groupby(["repeat", "fold"]).uncertainty.rank(pct=True)
    pd.concat([classical, neural], ignore_index=True).to_csv(destination / "predictions.csv", index=False)
    metrics = []
    for fold, frame in neural.groupby("fold"):
        values = evaluate_predictions(frame, config["confidence"])
        metrics.extend({"task": "classification", "endpoint": endpoint, "model": "chemprop",
            "preprocessing": "graph", "split_strategy": "molecule", "partition": "validation",
            "repeat": 0, "fold": int(fold), "n_molecules": len(frame), "metric": key, "value": value}
            for key, value in values.items() if np.isfinite(value))
    pd.concat([read_csv(destination / "fold_metrics.csv"), pd.DataFrame(metrics)], ignore_index=True).to_csv(destination / "fold_metrics.csv", index=False)
    neural.to_csv(root / "neural" / "BBBP_oof_predictions.csv", index=False)
    values = evaluate_predictions(neural, config["confidence"])
    (root / "neural" / "outcome.json").write_text(json.dumps({"n_molecules": len(neural), "folds": config["folds"],
        "epochs_per_member": 50, "members_per_fold": 2, "metrics": values}, indent=2))
    provenance_path = destination / "display_provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance.update(display_prediction_sha256=digest(destination / "predictions.csv"),
        neural_manifests={f"fold_{fold}": digest(root / "neural" / endpoint / f"fold_{fold}" / "manifest.json") for fold in range(config["folds"])},
        neural_scope="Five molecule folds, first registered seed; separate graph representation. All classical repetitions remain included.")
    provenance_path.write_text(json.dumps(provenance, indent=2))


def portal(root, config, outcomes, neural):
    counts = {spec["name"]: len(read_csv(root / spec["name"] / "observations.csv")) for spec in config["endpoints"]}
    jobs = len(list(root.glob("jobs/**/complete.json")))
    estimators = sum(1 if json.loads(p.read_text())["context"]["model"] == "linear" else config["members"] for p in root.glob("jobs/**/complete.json"))
    final_models = len(list((root / "deployment").rglob("model.joblib")))
    count_summary = " · ".join(f"{html.escape(name)}: {count:,}" for name, count in counts.items())
    rows = []
    for _, row in outcomes.iterrows():
        metric = "pr_auc" if row.task == "classification" else "mae"
        coverage = "set_coverage" if row.task == "classification" else "normalized_coverage"
        rows.append(f"<tr><td>{html.escape(row.endpoint)}</td><td>{html.escape(row.representation)}</td><td>{html.escape(row.split)}</td><td>{int(row["repeat"])}</td><td>{html.escape(row.model)}</td><td>{metric}: {row[metric]:.4f}</td><td>{row[coverage]:.1%}</td></tr>")
    neural_note = ""
    if neural:
        measured = json.loads((root / "neural/outcome.json").read_text())["metrics"]
        rows.append(f'<tr><td>BBBP</td><td>graph</td><td>molecule</td><td>0</td><td>chemprop</td><td>pr_auc: {measured["pr_auc"]:.4f}</td><td>{measured["set_coverage"]:.1%}</td></tr>')
        forest = outcomes[(outcomes.endpoint == "BBBP") & (outcomes.representation == "rdkit_descriptors")
            & (outcomes.split == "molecule") & (outcomes["repeat"] == 0) & (outcomes.model == "random_forest")]
        comparison = f' The matching first-seed RDKit forest has PR AUC {forest.iloc[0].pr_auc:.4f}.' if len(forest) else ""
        neural_note = f'<p>Completed neural training yields PR AUC {measured["pr_auc"]:.4f} and specificity {measured["specificity"]:.1%}.{comparison} See the <a href="neural/outcome.json">full neural outcome</a>; completed training is not a claim of superior performance.</p>'
    model_links = ''.join(f'<li><a href="{p.parent.relative_to(root)}/complete.json">{html.escape(str(p.parent.relative_to(root / "deployment")))}</a> · <a href="{p.relative_to(root)}">checkpoint</a></li>' for p in (root / "deployment").rglob("model.joblib"))
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Real molecular modelling — complete example</title>
<style>body{{margin:0;background:#eee9dd;color:#15231f;font:16px/1.6 system-ui}}main{{max-width:1120px;margin:auto;padding:35px 20px}}h1{{font:700 clamp(2rem,4vw,3.8rem)/1.1 Georgia}}h2{{font-family:Georgia}}section{{background:#fffdf7;border:1px solid #d6cfbf;border-radius:16px;padding:24px;margin:20px 0}}a{{color:#125948}}.button{{display:inline-block;background:#123f38;color:white;padding:12px 18px;border-radius:9px;text-decoration:none}}.stats{{display:flex;gap:20px;flex-wrap:wrap}}.stats b{{font-size:1.6rem;display:block}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{text-align:left;padding:8px;border-bottom:1px solid #ddd}}.table{{overflow:auto}}code{{overflow-wrap:anywhere}}.note{{color:#52655d}}</style>
<main><p>REAL DATA · SAVED MODELS · REPRODUCIBLE ANALYSIS</p><h1>From molecules to measured model reliability</h1>
<p>Solubility, lipophilicity and blood–brain-barrier permeability. Every result comes from a fitted estimator and registered held-out predictions.</p>
<section><div class="stats"><div><b>{sum(counts.values()):,}</b>endpoint–chemical measurements</div><div><b>{jobs}</b>completed CV jobs</div><div><b>{estimators:,}</b>classical fitted estimators</div><div><b>{final_models}</b>final calibrated predictors</div></div><p>{count_summary}. Five folds × two seeds; molecule and scaffold splits. 90% nominal intervals/sets.</p>
<a class="button" href="chemical_model_diagnostics.html">Explore the full chemical report</a> <a href="progress.html">Training registry</a></section>
<section><h2>A guided investigation</h2><ol>
<li><strong>Compare chemistry shifts.</strong> Choose KSOL, random_forest and RDKit model features. Compare molecule and scaffold splits, then examine the separate repeat/fold observations.</li>
<li><strong>Investigate failures.</strong> Select the property-cliff cohort or low training similarity. Inspect high-error structures and export the selected cohort.</li>
<li><strong>Check uncertainty.</strong> Compare absolute and ensemble-normalized conformal intervals; inspect coverage and width together. Enter a scientifically justified property cutoff to inspect interval decisions.</li>
<li><strong>Explore measured multi-endpoint chemistry.</strong> Keep KSOL and LogD selected. Lasso a region, inspect shared structures, and switch among fingerprint/descriptor maps and t-SNE, PCA and GTM.</li>
<li><strong>Study classification.</strong> Open BBBP; inspect ROC/PR curves, the confusion matrix, probability distributions, prediction sets and risk versus retention.</li>
<li><strong>Reproduce your view.</strong> Save/load a view recipe and export cohort CSV. Saved full-fold metrics retain their original populations.</li></ol>
<p>{'The neural addon contains five full BBBP folds, two Chemprop members per fold, 50 epochs each. Choose graph model features and chemprop; only repeat 0 / molecule split is available for that addon.' if neural else 'This build explicitly excludes the neural addon.'}</p>{neural_note}</section>
<section><h2>Use the trained predictors</h2><p>Final predictors assign the complete development pool to fitting or disjoint calibration roles. These are complete fitted artifacts, separate from the outer-fold checkpoints.</p>
<p><a href="inference/queries.csv">Example SMILES batch</a> · <a href="deployment/protocol.json">Final fitting policy</a> · <a href="inference/index.html">Batch prediction results</a></p>
<code>admet-cv predict deployment/KSOL/rdkit_descriptors/random_forest inference/queries.csv --output predictions.csv</code><details><summary>All final checkpoints</summary><ul>{model_links}</ul></details></section>
<section><h2>Measured out-of-fold results</h2><p class="note">Each row pools one prediction per chemical within one repeat. Repeats/folds share data; these descriptive results do not establish independent experimental replication.</p><div class="table"><table><thead><tr><th>Endpoint</th><th>Features</th><th>Split</th><th>Repeat</th><th>Model</th><th>Performance</th><th>Coverage</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>
<section><h2>Evidence and reproducibility</h2><p><a href="protocol.json">Frozen protocol</a> · <a href="outcome_audit.json">Artifact / OOF / checkpoint audit</a> · <a href="pooled_oof_outcomes.csv">Machine-readable outcomes</a> · <a href="chemical_cohorts.csv">Chemical cohort results</a> · <a href="cvplus/metrics.csv">Independent outer-evaluated CV+ pilot</a></p>
<p class="note">Historical test structures were used only to exclude overlap; their labels were not used. This is development evidence, and prospective performance of final refits remains unverified. LogD uses measured rows from the KSOL development pool. Post-hoc property-cliff labels use observed outcomes. Dataset licenses and publication rights remain with their original providers; this local demonstration is not a publication release.</p></section></main></html>'''
    (root / "index.html").write_text(document)
    (root / "example_summary.json").write_text(json.dumps({"chemicals_per_endpoint": counts, "cv_jobs": jobs,
        "fitted_classical_estimators": estimators, "final_predictors": final_models, "neural_included": neural}, indent=2))


def finish(root, *, neural=True):
    config = verify_registration(root)
    print("Auditing all held-out predictions and reloading representative checkpoints…", flush=True)
    outcomes = evaluate(root)
    final = root / "deployment"
    if not final.exists(): refit(root, final)
    verify_manifest(final)
    print("Running batch inference with every final predictor…", flush=True)
    inference = root / "inference"
    inference.mkdir(exist_ok=True)
    # These are real structures, without fabricated property labels.
    queries = pd.DataFrame({"sample_id": ["001", "002", "003", "004", "005"],
        "molecule_id": ["caffeine", "aspirin", "acetaminophen", "ethanol", "benzene"],
        "smiles": ["Cn1c(=O)c2c(ncn2C)n(C)c1=O", "CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Nc1ccc(O)cc1", "CCO", "c1ccccc1"]})
    queries.to_csv(inference / "queries.csv", index=False)
    links = []
    for model in sorted(final.rglob("model.joblib")):
        name = "_".join(model.parent.relative_to(final).parts)
        result = predict(model.parent, inference / "queries.csv", inference / f"{name}.csv")
        if result.sample_id.tolist() != queries.sample_id.tolist(): raise ValueError("Inference ID round-trip failed")
        links.append(f'<li><a href="{name}.csv">{html.escape(name)}</a></li>')
    if neural:
        from admet_benchmark.neural_cv import predict_neural
        directory = root / "neural/BBBP/fold_0"
        predict_neural(directory, inference / "queries.csv", inference / "BBBP_chemprop.csv")
        links.append('<li><a href="BBBP_chemprop.csv">BBBP / Chemprop outer-fold ensemble</a></li>')
    (inference / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Batch inference</title><h1>Predictions for five real structures</h1><p>These queries have no supplied ground-truth labels. Inspect intervals, sets and applicability before interpretation.</p><ul>'+''.join(links)+'</ul>')
    report_paths = prepare(root)
    if neural: add_neural_report(root, report_paths, config)
    print("Building the complete offline chemical report…", flush=True)
    generate_chemical_report(report_paths, output_path=root / "chemical_model_diagnostics.html", max_points=1_000_000,
        map_methods=["tsne", "gtm", "pca"], representations=["morgan", "rdkit_descriptors", "morgan_rdkit"],
        map_cache=root / "map_cache", delivery="assets", smarts_patterns=["c1ccccc1", "C(=O)N"])
    portal(root, config, outcomes, neural)
    return root / "index.html"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPOSITORY / "outputs/complete_real_example_2026_09_29")
    parser.add_argument("--config", type=Path, default=REPOSITORY / "configs/complete_real_example.yaml")
    parser.add_argument("--classical-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    if not root.exists(): register(args.config, root)
    config = verify_registration(root)
    if args.validate_only:
        print(f"Validated registered inputs and roles: {root}")
        return
    print("Verifying completed jobs and resuming registered training…", flush=True)
    run(root)
    if not args.classical_only:
        from admet_benchmark.neural_cv import run_neural_fold
        for fold in range(config["folds"]):
            directory = root / "neural/BBBP" / f"fold_{fold}"
            if directory.exists(): verify_manifest(directory)
            else: run_neural_fold(root, directory, endpoint="BBBP", fold=fold, epochs=50, members=2)
    from admet_benchmark.cvplus import run_pilot
    if (root / "cvplus").exists(): verify_manifest(root / "cvplus")
    else: run_pilot(root, root / "cvplus")
    print(finish(root, neural=not args.classical_only))


if __name__ == "__main__": main()
