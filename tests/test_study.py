from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from admet_benchmark.study import prepare_study, freeze_study, verify_study


def write_study(path: Path, *, classification=False, scaffold=False):
    # Synthetic measurements; the two endpoints deliberately share every molecule.
    smiles = ['C' * n + suffix for n in range(1, 21) for suffix in ('O', 'N', 'F', 'Cl')]
    if scaffold:
        smiles = ['C1' + 'C' * n + '1' + suffix for n in range(2, 22) for suffix in ('O', 'N', 'F', 'Cl')]
    data = pd.DataFrame([{'sample_id': f'mol-{i}', 'endpoint': endpoint, 'smiles': value,
                          'observed': i % 2 if classification else float(i + 1)}
                         for endpoint in ('A', 'B') for i, value in enumerate(smiles)])
    source = path.parent / 'observations.csv'
    data.to_csv(source, index=False)
    config = {'schema_version': 1, 'study_id': 'synthetic-demo', 'evidence_level': 'screening',
              'dataset': {'path': source.name, 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                          'source': 'Synthetic test observations', 'license': 'CC0-1.0'},
              'identity_policy': 'rdkit_canonical_isomeric_v1',
              'endpoints': [{'name': name, 'task': 'classification' if classification else 'regression',
                             'units': 'synthetic', 'transform': 'none',
                             'primary_metric': 'pr_auc' if classification else 'mae', 'practical_difference': 0.1}
                            for name in ('A', 'B')],
              'models': ['baseline', 'tree'],
              'split': {'strategy': 'scaffold' if scaffold else 'molecule', 'test_seed': 42,
                        'repeat_seeds': [11, 22, 33], 'test_fraction': 0.2,
                        'validation_fraction': 0.2, 'calibration_fraction': 0.2,
                        'minimum_molecules': 2, 'minimum_per_class': 1},
              'selection_policy': 'fixed_models_no_tuning', 'test_policy': 'locked_until_protocol_final',
              'uq_methods': ['none'], 'cohorts': ['all'], 'compute_budget_cpu_hours': 1}
    path.write_text(yaml.safe_dump(config))
    return config, data


def test_shared_partitions_and_locked_test_are_reproducible(tmp_path):
    path = tmp_path / 'study.yaml'
    write_study(path)
    study = prepare_study(path)
    assignments = study.assignments
    assert study.audit['distinct_partitions'] == 3
    assert assignments.groupby(['repeat', 'chemical_id']).partition.nunique().eq(1).all()
    assert assignments.groupby(['repeat', 'split_group']).partition.nunique().eq(1).all()
    test = assignments.query("partition == 'test'")
    assert test.groupby('chemical_id').repeat.nunique().eq(3).all()
    assert set(test.chemical_id).isdisjoint(assignments.query("partition != 'test'").chemical_id)
    assert set(assignments.partition) == {'train', 'calibration', 'validation', 'test'}
    pd.testing.assert_frame_equal(assignments, prepare_study(path).assignments)
    # Preflight is read only.
    assert {p.name for p in tmp_path.iterdir()} == {'observations.csv', 'study.yaml'}


def test_row_order_does_not_change_assignments(tmp_path):
    path = tmp_path / 'study.yaml'
    config, data = write_study(path)
    expected = prepare_study(path).assignments
    source = tmp_path / 'observations.csv'
    data.sample(frac=1, random_state=9).to_csv(source, index=False)
    config['dataset']['sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    path.write_text(yaml.safe_dump(config))
    pd.testing.assert_frame_equal(expected, prepare_study(path).assignments)


def test_freeze_portable_verifiable_and_never_overwrites(tmp_path):
    path = tmp_path / 'study.yaml'
    write_study(path)
    destination = freeze_study(path, tmp_path / 'registration')
    assert verify_study(destination)['status'] == 'verified'
    assert str(tmp_path) not in (destination / 'manifest.json').read_text()
    assert len(json.loads((destination / 'run_index.json').read_text())) == 12
    before = (destination / 'manifest.json').read_bytes()
    with pytest.raises(FileExistsError):
        freeze_study(path, destination)
    assert (destination / 'manifest.json').read_bytes() == before
    with (destination / 'fold_assignments.csv').open('a') as handle:
        handle.write('\n')
    with pytest.raises(ValueError, match='artifact changed'):
        verify_study(destination)


def test_scaffold_groups_never_cross_partitions(tmp_path):
    path = tmp_path / 'study.yaml'
    write_study(path, scaffold=True)
    study = prepare_study(path)
    assert study.assignments.groupby(['repeat', 'split_group']).partition.nunique().eq(1).all()
    assert study.audit['groups'] == 20


@pytest.mark.parametrize('change,match', [
    (lambda c: c['dataset'].update(sha256='0'*64), 'SHA-256'),
    (lambda c: c.update(evidence_level='confirmatory'), 'schema'),
    (lambda c: c['split'].update(repeat_seeds=[1, 1]), 'schema'),
    (lambda c: c['split'].update(minimum_molecules=500), 'minimum support'),
    (lambda c: c['split'].update(validation_fraction=.7, calibration_fraction=.4), 'leave training'),
    (lambda c: c['endpoints'][0].update(primary_metric='pr_auc'), 'match endpoint'),
])
def test_preflight_rejects_invalid_protocol(tmp_path, change, match):
    path = tmp_path / 'study.yaml'
    config, _ = write_study(path)
    change(config)
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match=match):
        prepare_study(path)


def test_duplicate_chemistry_rejected_even_with_another_sample_id(tmp_path):
    path = tmp_path / 'study.yaml'
    config, data = write_study(path)
    data.loc[data.sample_id.eq('mol-1'), 'smiles'] = data.loc[0, 'smiles']
    source = tmp_path / 'observations.csv'
    data.to_csv(source, index=False)
    config['dataset']['sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='Duplicate chemical'):
        prepare_study(path)


def test_classification_requires_support_in_every_partition(tmp_path):
    path = tmp_path / 'study.yaml'
    config, _ = write_study(path, classification=True)
    study = prepare_study(path)
    assert all(set(row['class_counts']) == {'0', '1'} for row in study.audit['support'])
    config['split']['minimum_per_class'] = 80
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='insufficient class support'):
        prepare_study(path)
