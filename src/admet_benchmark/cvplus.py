"""Regression CV+ candidate intervals and a nested, outer-held-out pilot.

The pilot uses fresh inner K-fold point models. It never estimates CV+ coverage
on the same observations that supplied the inner residuals. Nominal tail settings
are not claims of exact conditional or scaffold-shift coverage.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from .models import model_specs
from .features import build_features
from .uq import interval_score


def cvplus_bounds(query_predictions, residuals, residual_folds, alpha=.1):
    prediction=np.asarray(query_predictions,dtype=float)
    residuals=np.asarray(residuals,dtype=float);folds=np.asarray(residual_folds,dtype=int)
    if prediction.ndim!=2 or residuals.ndim!=1 or len(residuals)!=len(folds) or not 0<alpha<1:
        raise ValueError('Invalid CV+ arrays or tail probability')
    if not np.isfinite(prediction).all() or not np.isfinite(residuals).all() or np.any(residuals<0) or np.any(folds<0) or np.any(folds>=len(prediction)):
        raise ValueError('Nonfinite or invalid CV+ values')
    n=len(residuals);lo=int(np.floor(round((n+1)*alpha,12)));hi=int(np.ceil(round((n+1)*(1-alpha),12)))
    if lo<1 or hi>n:raise ValueError('Insufficient residual support for finite CV+ bounds')
    # Bound temporary arrays for batch prediction; kth order statistics, no interpolated quantiles.
    lower=[];upper=[]
    for start in range(0,prediction.shape[1],128):
        centre=prediction[folds,start:start+128]
        lower.extend(np.partition(centre-residuals[:,None],lo-1,axis=0)[lo-1])
        upper.extend(np.partition(centre+residuals[:,None],hi-1,axis=0)[hi-1])
    return np.asarray(lower),np.asarray(upper)


def fit_cvplus(x,y,folds=5,seed=42,trees=150):
    x=np.asarray(x);y=np.asarray(y);models=[];residual=np.zeros(len(y));membership=np.zeros(len(y),dtype=int)
    for fold,(fit,validation) in enumerate(KFold(folds,shuffle=True,random_state=seed).split(x)):
        model=model_specs('regression',['random_forest'],seed)[0].estimator
        model.set_params(model__n_estimators=trees,model__min_samples_leaf=2,model__max_features='sqrt',model__n_jobs=2)
        model.fit(x[fit],y[fit]);residual[validation]=np.abs(y[validation]-model.predict(x[validation]));membership[validation]=fold;models.append(model)
    return {'models':models,'residuals':residual,'residual_folds':membership}


def predict_cvplus(bundle,x,alpha=.1):
    predictions=np.array([model.predict(x) for model in bundle['models']])
    lower,upper=cvplus_bounds(predictions,bundle['residuals'],bundle['residual_folds'],alpha)
    return predictions.mean(axis=0),lower,upper


def run_pilot(study,output):
    import joblib
    from threadpoolctl import threadpool_limits
    from .cross_validation import digest,verify_registration,write_json
    study=Path(study);output=Path(output);config=verify_registration(study)
    if output.exists():raise FileExistsError('CV+ pilot requires a new output directory')
    output.mkdir(parents=True)
    source=pd.read_csv(study/'KSOL/observations.csv');roles=pd.read_csv(study/'KSOL/assignments.csv')
    roles=roles[(roles.split_strategy=='molecule')&(roles.repeat==0)]
    x=build_features(source.smiles.tolist(),representation='morgan').values;y=source.observed.to_numpy();rows=[];metrics=[]
    with threadpool_limits(limits=2):
        for fold in sorted(roles.fold.unique()):
            assignment=roles[roles.fold==fold];train=np.flatnonzero(source.sample_id.isin(assignment[assignment.role!='validation'].sample_id));test=np.flatnonzero(source.sample_id.isin(assignment[assignment.role=='validation'].sample_id))
            assert not set(train)&set(test)
            bundle=fit_cvplus(x[train],y[train],folds=config['folds'],seed=config['seeds'][0],trees=config['trees'])
            p,lo,hi=predict_cvplus(bundle,x[test],alpha=1-config['confidence'])
            predictions=pd.DataFrame({'sample_id':source.iloc[test].sample_id.to_numpy(),'smiles':source.iloc[test].smiles.to_numpy(),'fold':fold,'observed':y[test],'prediction':p,'lower':lo,'upper':hi})
            rows.append(predictions);metrics.append({'fold':int(fold),'n':len(test),'mae':float(np.abs(y[test]-p).mean()),'coverage':float(np.mean((y[test]>=lo)&(y[test]<=hi))),'mean_width':float(np.mean(hi-lo)),'interval_score':float(interval_score(y[test],lo,hi,config['confidence']).mean())})
            bundle.update(training_ids=source.iloc[train].sample_id.tolist(),representation='morgan',evaluation_scale='log10',outer_fold=int(fold))
            joblib.dump(bundle,output/f'fold_{fold}.joblib',compress=3)
            print(json.dumps(metrics[-1]),flush=True)
    pd.concat(rows).to_csv(output/'predictions.csv',index=False);pd.DataFrame(metrics).to_csv(output/'metrics.csv',index=False)
    write_json(output/'manifest.json',{'implementation_sha256':digest(__file__),'parent_registration_sha256':digest(study/'registration.json'),'method':'CV+ with inner K-fold RF point models; separate outer OOF evaluation','confidence_setting':config['confidence'],'coverage_caveat':'Nominal finite-sample quantile setting; CV+ worst-case bounds differ from exact 1-alpha coverage; no conditional or scaffold-shift guarantee','artifacts':{p.name:digest(p) for p in output.iterdir() if p.is_file()}})
    return pd.DataFrame(metrics)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('study');parser.add_argument('--output',required=True);args=parser.parse_args();run_pilot(args.study,args.output)
