from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
import yaml
from admet_benchmark.uq import conformal_quantile, regression_intervals, prediction_sets, classification_calibration, ProbabilityCalibrator
from admet_benchmark.cross_validation import register, run, predict, verify_registration, assignments, verify_assignments
from admet_benchmark.applicability import training_neighbours, cliff_annotations


def test_corrected_quantile_and_prediction_sets():
    assert conformal_quantile(np.arange(1,10),.9)==9
    with pytest.raises(ValueError,match='Insufficient'):conformal_quantile(np.arange(8),.9)
    with pytest.raises(ValueError):conformal_quantile([np.nan],.5)
    sets=prediction_sets(np.array([0,.5,1]),{'class_q':[.2,.2]})
    np.testing.assert_array_equal(sets,[[1,0],[0,0],[0,1]])
    sets=prediction_sets(np.array([.5]),{'class_q':[.7,.7]})
    np.testing.assert_array_equal(sets,[[1,1]])
    lower,upper=regression_intervals(np.array([1,1]),np.array([0,2]),{'floor':.5,'normalized_q':2})
    np.testing.assert_array_equal(lower,[0,-3]);np.testing.assert_array_equal(upper,[2,5])


def test_original_training_reference_and_cliffs():
    app=training_neighbours(['CCO','c1ccccc1'],['CCO','c1ccncc1'])
    assert app['nearest_train_similarity'][0]==1 and app['nearest_train_index'][0]==0
    assert app['novel_scaffold']==[0,1]
    result=cliff_annotations(['CCO'],[1],['CCO'],[3],difference=1)
    assert result['property_cliff']==[1]


@pytest.fixture
def registered(tmp_path):
    source=pd.read_csv(Path(__file__).resolve().parents[1]/'data/tutorial/study_observations.csv').drop_duplicates('smiles').reset_index(drop=True)
    source['target_model']=np.arange(len(source))%2
    # One same-label duplicate and one conflicting identity; original rows retained in audit.
    source=pd.concat([source,source.iloc[[0]],source.iloc[[1]].assign(target_model=0)],ignore_index=True)
    source.to_csv(tmp_path/'input.csv',index=False)
    config={'study_id':'test','folds':5,'seeds':[4],'strategies':['molecule'],'representations':['morgan'],
        'models':['random_forest'],'members':3,'trees':5,'workers':1,'confidence':.7,
        'endpoints':[{'name':'TEST','task':'classification','sources':['input.csv'],'evaluation_scale':'none'}]}
    (tmp_path/'config.yaml').write_text(yaml.safe_dump(config))
    return register(tmp_path/'config.yaml',tmp_path/'study')


def test_registered_run_resume_and_predict(registered,tmp_path):
    root=registered
    pre=json.loads((root/'TEST/preflight.json').read_text())
    assert pre['retained_chemicals']==79 and pre['quarantined_chemical_identities']==1
    assert run(root,limit=1)=={'complete':1,'planned':4}
    first=next((root/'jobs').rglob('model.joblib'));before=first.stat().st_mtime_ns
    assert run(root)=={'complete':5}
    assert first.stat().st_mtime_ns==before
    predictions=pd.read_csv(root/'evaluation/TEST/predictions.csv')
    assert len(predictions)==79 and predictions.sample_id.is_unique
    assert predictions[['set_0','set_1']].isin([0,1]).all().all()
    assert ((predictions.prediction>=0)&(predictions.prediction<=1)).all()
    validation=json.loads((root/'evaluation/TEST/validation_report.json').read_text())
    assert not validation['errors']
    pd.DataFrame({'smiles':['CCO','CCN']}).to_csv(tmp_path/'queries.csv',index=False)
    result=predict(first.parent,tmp_path/'queries.csv',tmp_path/'new.csv')
    assert len(result)==2 and result.model_sha256.nunique()==1
    assert result[['set_0','set_1']].isin([0,1]).all().all()
    (first.parent/'predictions.csv').write_text('tampered')
    with pytest.raises(ValueError,match='artifact changed'):run(root)


def test_registration_tampering_rejected(registered):
    with (registered/'TEST/observations.csv').open('a') as f:f.write('\n')
    with pytest.raises(ValueError,match='artifact changed'):verify_registration(registered)


def test_ten_fold_oof_and_role_leakage():
    frame=pd.DataFrame({'sample_id':[str(i) for i in range(200)],'observed':np.arange(200)%2,'scaffold':[str(i//2) for i in range(200)]})
    roles=assignments(frame,'classification',10,[42],['molecule','scaffold'])
    assert len(roles)==4000
    assert len(roles[roles.role=='validation'])==400
    bad=roles.copy();bad.loc[0,'sample_id']='unknown'
    with pytest.raises(ValueError):verify_assignments(frame,bad,'classification',10)


def test_regression_pipeline_scale_and_new_molecule_intervals(tmp_path):
    source=pd.read_csv(Path(__file__).resolve().parents[1]/'data/tutorial/study_observations.csv').drop_duplicates('smiles').reset_index(drop=True)
    source['target_model']=np.log10(np.arange(len(source))+1)
    source['target_raw']=10**source.target_model
    source.to_csv(tmp_path/'input.csv',index=False)
    config={'study_id':'reg','folds':5,'seeds':[4],'strategies':['molecule'],'representations':['rdkit_descriptors'],
        'models':['linear'],'members':2,'trees':5,'workers':1,'confidence':.8,
        'endpoints':[{'name':'REG','task':'regression','sources':['input.csv'],'evaluation_scale':'log10'}]}
    (tmp_path/'config.yaml').write_text(yaml.safe_dump(config));root=register(tmp_path/'config.yaml',tmp_path/'study')
    assert run(root)=={'complete':5}
    saved=pd.read_csv(root/'evaluation/REG/predictions.csv')
    assert saved.sample_id.is_unique and len(saved)==80
    assert (saved.prediction_lower<=saved.prediction_upper).all()
    np.testing.assert_allclose(saved.prediction_lower,saved.absolute_lower)
    pd.DataFrame({'smiles':['CCO']}).to_csv(tmp_path/'query.csv',index=False)
    model=next((root/'jobs').rglob('model.joblib'))
    result=predict(model.parent,tmp_path/'query.csv',tmp_path/'pred.csv',cutoff=1)
    assert result.cutoff_decision.iloc[0] in {'above','below','crosses'}
    np.testing.assert_allclose(result.prediction_raw,10**result.prediction)


def test_cvplus_order_statistics_and_permutation():
    from admet_benchmark.cvplus import cvplus_bounds,fit_cvplus,predict_cvplus
    residual=np.arange(1,11,dtype=float);folds=np.arange(10)%2;query=np.array([[10.,20.],[12.,22.]])
    lo,hi=cvplus_bounds(query,residual,folds,alpha=.2)
    expected_lo=np.sort(query[folds]-residual[:,None],axis=0)[1]
    expected_hi=np.sort(query[folds]+residual[:,None],axis=0)[8]
    np.testing.assert_array_equal(lo,expected_lo);np.testing.assert_array_equal(hi,expected_hi)
    shuffled=np.random.default_rng(1).permutation(10)
    lo2,hi2=cvplus_bounds(query,residual[shuffled],folds[shuffled],alpha=.2)
    np.testing.assert_array_equal(lo,lo2);np.testing.assert_array_equal(hi,hi2)
    with pytest.raises(ValueError):cvplus_bounds(query,residual[:2],folds[:2],alpha=.01)
    x=np.random.default_rng(1).normal(size=(40,4));y=x[:,0]+.1*x[:,1]
    bundle=fit_cvplus(x,y,folds=5,trees=5)
    p,lo,hi=predict_cvplus(bundle,x[:3],alpha=.1)
    assert len(p)==3 and (lo<=hi).all()


def test_final_development_refit_and_prediction(registered, tmp_path):
    from admet_benchmark.deployment import refit
    from admet_benchmark.provenance import verify_compatibility
    import joblib
    output = refit(registered, tmp_path / 'deployment')
    roles = pd.read_csv(output / 'TEST/assignments.csv')
    assert roles.sample_id.is_unique and len(roles) == 79
    assert set(roles.role) == {'fit', 'probability_calibration', 'conformal_calibration'}
    model = next(output.rglob('model.joblib'))
    pd.DataFrame({'sample_id':['001','NA'], 'smiles':['CCO','CCN']}).to_csv(tmp_path/'queries.csv',index=False)
    result = predict(model.parent, tmp_path/'queries.csv', tmp_path/'result.csv')
    assert result.sample_id.tolist() == ['001','NA']
    assert result.model_role.str.contains('final development predictor').all()
    assert result[['set_0','set_1']].isin([0,1]).all().all()
