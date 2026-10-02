"""Build a reproducible synthetic 5/10-fold report; no measured ADMET claims.

Run from the repository root with the visual dependencies installed:
    python scripts/build_fold_uq_demo.py
Labels below are artificial. Forests are actually fitted separately inside each
fold; calibration and evaluation molecules are excluded from model fitting.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, mean_absolute_error, root_mean_squared_error, roc_auc_score)
from sklearn.model_selection import KFold, train_test_split
from threadpoolctl import threadpool_limits

from admet_benchmark.chemical_report import generate_chemical_report
from admet_benchmark.chemical_space import GTMConfig


def build_demo(destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv('data/tutorial/study_observations.csv')
    molecules = source.drop_duplicates('smiles').reset_index(drop=True)
    smiles = molecules.smiles.tolist()
    features = np.array([[Descriptors.MolWt(m), Descriptors.MolLogP(m),
                          Descriptors.TPSA(m), Descriptors.NumHDonors(m),
                          Descriptors.NumHAcceptors(m)]
                         for m in map(Chem.MolFromSmiles, smiles)])
    rng = np.random.default_rng(20260908)
    regression = (.008 * features[:,0] - .2 * features[:,1] + .015 * features[:,2]
                  + rng.normal(0, .18, len(smiles)))
    classification = (features[:,2] + rng.normal(0, 9, len(smiles)) > 15).astype(int)
    assert len(set(map(lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s)), smiles))) == len(smiles)
    predictions, metrics, assignments, calibration_audit = [], [], [], []
    for nfolds in (5, 10):
        strategy = f'synthetic_{nfolds:02d}fold'
        # Splits depend only on seeded molecule indices, shared by both tasks.
        for fold, (outer_train, evaluation) in enumerate(KFold(nfolds, shuffle=True, random_state=17).split(features)):
            train, calibration = train_test_split(outer_train, test_size=.25, random_state=31+fold)
            assert not set(train) & set(calibration) and not set(outer_train) & set(evaluation)
            for role, indices in [('fit', train), ('calibration', calibration), ('validation', evaluation)]:
                assignments.extend(dict(split_strategy=strategy, fold=fold, role=role,
                                        sample_id=f'synthetic-{i}', smiles=smiles[i]) for i in indices)
            for task, targets, estimator in [('regression', regression, RandomForestRegressor),
                                              ('classification', classification, RandomForestClassifier)]:
                endpoint = f'SYNTHETIC {task} · demonstration only'
                if task == 'classification':
                    assert set(targets[train]) == {0, 1}
                members = [estimator(n_estimators=40, max_depth=6, min_samples_leaf=2,
                                     random_state=seed, n_jobs=1).fit(features[train], targets[train])
                           for seed in (101, 202, 303, 404, 505)]
                def predict(indices):
                    return np.array([member.predict(features[indices]) if task == 'regression'
                                     else member.predict_proba(features[indices])[:,1] for member in members])
                member_predictions = predict(evaluation)
                predicted = member_predictions.mean(axis=0)
                uncertainty = member_predictions.std(axis=0, ddof=1)
                if task == 'regression':
                    scores = np.sort(np.abs(targets[calibration] - predict(calibration).mean(axis=0)))
                    order = int(np.ceil((len(scores)+1)*.9))
                    assert order <= len(scores), 'Insufficient calibration observations for a finite 90% interval'
                    radius = float(scores[order-1])
                    calibration_audit.append(dict(split_strategy=strategy, fold=fold, n_calibration=len(scores),
                                                  order_statistic=order, nominal_coverage=.9, interval_radius=radius))
                common = dict(task=task, endpoint=endpoint, model='five_forest_ensemble',
                              partition='validation', repeat=0, fold=fold,
                              preprocessing='raw_descriptors', split_strategy=strategy, target_transform='none')
                for j, i in enumerate(evaluation):
                    row = dict(**common, sample_id=f'synthetic-{i}', molecule_id=f'synthetic-{i}',
                               smiles=smiles[i], observed=float(targets[i]), prediction=float(predicted[j]),
                               uncertainty=float(uncertainty[j]))
                    if task == 'regression':
                        row.update(prediction_lower=float(predicted[j]-radius), prediction_upper=float(predicted[j]+radius))
                    else:
                        row['threshold'] = .5
                    predictions.append(row)
                if task == 'regression':
                    values = dict(mae=mean_absolute_error(targets[evaluation], predicted),
                                  rmse=root_mean_squared_error(targets[evaluation], predicted),
                                  interval_coverage=np.mean(np.abs(targets[evaluation]-predicted) <= radius),
                                  interval_width=2*radius)
                else:
                    values = dict(brier=brier_score_loss(targets[evaluation], predicted),
                                  log_loss=log_loss(targets[evaluation], predicted, labels=[0,1]))
                    if len(set(targets[evaluation])) == 2:
                        values['roc_auc'] = roc_auc_score(targets[evaluation], predicted)
                        values['pr_auc'] = average_precision_score(targets[evaluation], predicted)
                metrics.extend(dict(**common, metric=name, value=float(value), n_molecules=len(evaluation))
                               for name, value in values.items())
    bundle = destination/'synthetic_predictions'
    bundle.mkdir(exist_ok=True)
    pd.DataFrame(predictions).to_csv(bundle/'predictions.csv', index=False)
    pd.DataFrame(metrics).to_csv(bundle/'fold_metrics.csv', index=False)
    pd.DataFrame(assignments).to_csv(destination/'molecule_roles.csv', index=False)
    (destination/'calibration_audit.json').write_text(json.dumps(calibration_audit, indent=2)+'\n')
    (bundle/'resolved_config.yaml').write_text('uncertainty:\n  interval_confidence: 0.9\n')
    (destination/'README.md').write_text('''# Synthetic fold and uncertainty demonstration

All targets are artificial and all structures come from the repository's synthetic
tutorial. This is a UI and pipeline demonstration, not evidence of ADMET accuracy.
Select Split strategy **synthetic_05fold** or **synthetic_10fold** and Fold display
**Small multiples by fold**. The two schemes reuse the same molecules and must not
be pooled as independent experiments.

Each outer fold uses 75% of its training pool for fitting and 25% for calibration.
Five separately seeded 40-tree forests produce a mean prediction and an uncertainty
score (standard deviation across forest means). This dispersion is a heuristic,
not total predictive uncertainty. No calibration/evaluation targets enter fitting.
Regression intervals use the finite-sample corrected 90% split-conformal absolute
calibration-error quantile; the interval radius is constant within a fold. No
hyperparameter selection is performed. A nominal line is not a guarantee for any
particular fold, molecule, chemical subgroup or nonexchangeable population.

`molecule_roles.csv` records every fit/calibration/validation assignment;
`calibration_audit.json` records each quantile. The report excludes training and
calibration targets and fits its exploratory GTM atlas to validation chemistry.
No classification prediction sets are generated; inspect probability calibration,
Brier/log loss, discrimination, confusion and dispersion versus error instead.

Reproduce from the project root: `python scripts/build_fold_uq_demo.py`.
''')
    report = generate_chemical_report([bundle], output_path=destination/'fold_uq_demo.html',
                                     map_methods=['tsne','gtm','pca'],
                                     representations=['morgan','rdkit_descriptors','morgan_rdkit'],
                                     map_cache=destination/'map_cache',
                                     gtm_config=GTMConfig(nodes=9,rbfs=3))
    print(json.dumps(dict(report=str(report), molecules=len(smiles), observations=len(predictions),
                          saved_fold_metrics=len(metrics), role_rows=len(assignments))))
    return report


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        build_demo(Path('outputs/workspace_v3/fold_uq_demo'))
