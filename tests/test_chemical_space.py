import numpy as np
import pytest

from admet_benchmark.chemical_space import GaussianGTM, GTMConfig, build_map_atlas


def test_gtm_probabilities_and_regularized_em_objective():
    rng = np.random.default_rng(4)
    uv = rng.uniform(-1, 1, (100, 2))
    x = np.column_stack([uv, uv[:,0]**2 + uv[:,1]**2, uv[:,0]*uv[:,1]]) + rng.normal(0,.03,(100,4))
    model = GaussianGTM(GTMConfig(nodes=7,rbfs=3,max_iterations=80)).fit(x)
    projected = model.project(x)
    np.testing.assert_allclose(projected['responsibilities'].sum(axis=1),1,atol=1e-12)
    assert np.isfinite(projected['log_density']).all()
    assert (projected['entropy']>=0).all() and (projected['entropy']<=1).all()
    assert np.max(np.abs(projected['coordinates'])) <= 1
    assert np.diff(model.objective_history).min() >= -1e-5
    before=model.weights.copy();model.project(x+1)
    np.testing.assert_array_equal(model.weights,before)
    np.testing.assert_allclose(model.project(x)['coordinates'],GaussianGTM(model.config).fit(x).project(x)['coordinates'])


def test_gtm_far_chemistry_has_lower_reference_density():
    x=np.random.default_rng(1).normal(0,1,(100,5))
    model=GaussianGTM(GTMConfig(nodes=5,rbfs=3,max_iterations=40)).fit(x)
    assert model.project(x+100)['log_density'].mean() < model.project(x)['log_density'].mean()


def test_gtm_rejects_constant_or_invalid_reference():
    with pytest.raises(ValueError,match='constant'):
        GaussianGTM().fit(np.ones((10,4)))
    with pytest.raises(ValueError,match='finite'):
        GaussianGTM().fit(np.full((10,4),np.nan))
    with pytest.raises(ValueError):
        GaussianGTM(GTMConfig(nodes=2))


def test_atlas_retains_one_coordinate_per_molecule_and_cache(tmp_path):
    smiles=['CCO','CCN','CCC','CCCl','CCCO','CCCN','CCCC','CCCCl']
    records=[dict(chemical_id=str(i),canonical_smiles=s,split='development' if i<6 else 'test',target_model=0) for i,s in enumerate(smiles)]
    settings=dict(methods=['gtm','pca'],representations=['morgan','rdkit_descriptors','morgan_rdkit'],gtm_config=GTMConfig(nodes=5,rbfs=3,max_iterations=20),cache_dir=tmp_path)
    first=build_map_atlas(records,**settings)
    assert len(first['views'])==6
    assert first['reference_molecules']==6
    assert set(first['reference_chemical_ids'])==set(map(str,range(6)))
    # Outcome changes, extra fold observations and input order do not alter geometry.
    second=build_map_atlas([{**r,'target_model':999} for r in records[::-1]]+records[:2],**settings)
    assert first==second
    assert len(list(tmp_path.glob('*.json')))==1
    for view in first['views']:
        assert len(view['coordinates'])==len(smiles)
