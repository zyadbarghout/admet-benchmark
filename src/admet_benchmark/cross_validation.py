"""Registered outer CV, isolated calibration, resumable fits and local prediction.

Execute with python -m admet_benchmark.cross_validation --help. Registrations and
job completion manifests bind inputs and outputs; no historical test is scored.
Only load trusted locally generated model bundles (joblib uses Python pickle).
"""
from __future__ import annotations
import argparse
import hashlib
import html
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import yaml
from rdkit import rdBase
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, GroupShuffleSplit, train_test_split, KFold, StratifiedKFold
from sklearn.metrics import brier_score_loss, log_loss
from threadpoolctl import threadpool_limits
from .applicability import chemical, scaffold, training_neighbours, cliff_annotations
from .features import build_features
from .tables import read_csv
from .provenance import runtime_signature, implementation_signature, feature_schema, verify_compatibility, environment_lock
from .models import model_specs
from .metrics import regression_metrics, classification_metrics
from .uq import ProbabilityCalibrator, regression_calibration, regression_intervals, classification_calibration, prediction_sets, interval_score


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def clean_pool(spec, base):
    frames, source_hashes = [], {}
    for filename in spec['sources']:
        path = (base/filename).resolve(); frame = read_csv(path, text_columns=[spec.get("smiles_column", "smiles"), "sample_id"])
        source_hashes[str(path)] = digest(path)
        smiles_column = spec.get('smiles_column', 'smiles')
        target_column = spec.get('target_column', 'target_model')
        raw_column = spec.get('raw_column', 'target_raw')
        frame = pd.DataFrame({'smiles': frame[smiles_column], 'observed': pd.to_numeric(frame[target_column]),
                              'observed_raw': pd.to_numeric(frame[raw_column]) if raw_column in frame else pd.to_numeric(frame[target_column]),
                              'source_id': frame['sample_id'].astype(str) if 'sample_id' in frame else frame.index.astype(str),
                              'source_file': path.name})
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    if not np.isfinite(raw[['observed','observed_raw']].to_numpy()).all():
        raise ValueError('Development targets must be finite')
    if spec['task']=='classification' and not set(raw.observed.unique()) <= {0,1}:
        raise ValueError('Classification requires binary labels')
    if spec['evaluation_scale']=='log10' and (np.any(raw.observed_raw<=0) or not np.allclose(np.log10(raw.observed_raw),raw.observed)):
        raise ValueError('Prepared log10 targets do not match the declared raw scale')
    raw['smiles'] = raw.smiles.map(chemical)
    conflicting = set(raw.groupby('smiles')['observed'].nunique().loc[lambda s:s>1].index)
    audit = raw.copy(); audit['action'] = 'retained'
    audit.loc[raw.smiles.duplicated(), 'action'] = 'same_label_replicate'
    audit.loc[raw.smiles.isin(conflicting), 'action'] = 'quarantined_conflicting_labels'
    retained = raw.loc[~raw.smiles.isin(conflicting)].drop_duplicates('smiles').sort_values('smiles').reset_index(drop=True)
    retained['sample_id'] = retained.smiles.map(lambda s:hashlib.sha256(s.encode()).hexdigest()[:20])
    retained['scaffold'] = retained.smiles.map(scaffold)
    # Read only test structures for exclusion. Historical labels never enter fitting or calibration.
    historical_overlap = 0
    if spec.get('historical_test'):
        path = (base/spec['historical_test']).resolve()
        historical = pd.read_csv(path, usecols=[spec.get('smiles_column','smiles')])
        heldout = set(historical.iloc[:,0].map(chemical))
        historical_overlap = len(set(retained.smiles)&heldout)
        source_hashes[str(path)] = digest(path)
        if historical_overlap:
            raise ValueError('Development chemistry overlaps the historical test')
    return retained, audit, {'source_hashes':source_hashes,'source_rows':len(raw), 'retained_chemicals':len(retained),
        'quarantined_chemical_identities':len(conflicting), 'historical_test_overlap':historical_overlap,
        'duplicate_policy':'Collapse same-label canonical-isomeric replicates; quarantine conflicting identities',
        'evaluation_scale':spec['evaluation_scale']}


def inner_holdout(indices, frame, strategy, fraction, seed):
    if strategy=='scaffold':
        groups=frame.iloc[indices].scaffold.to_numpy()
        a,b=next(GroupShuffleSplit(n_splits=1,test_size=fraction,random_state=seed).split(indices,groups=groups))
        return indices[a],indices[b]
    stratify=frame.iloc[indices].observed if frame.attrs['task']=='classification' else None
    return train_test_split(indices,test_size=fraction,random_state=seed,stratify=stratify)


def assignments(frame, task, folds, seeds, strategies):
    frame.attrs['task']=task; output=[]
    for strategy in strategies:
        for repeat,seed in enumerate(seeds):
            if strategy=='scaffold':
                splitter=StratifiedGroupKFold(folds,shuffle=True,random_state=seed) if task=='classification' else GroupKFold(folds,shuffle=True,random_state=seed)
                splits=splitter.split(frame,frame.observed,frame.scaffold)
            else:
                splitter=StratifiedKFold(folds,shuffle=True,random_state=seed) if task=='classification' else KFold(folds,shuffle=True,random_state=seed)
                splits=splitter.split(frame,frame.observed)
            for fold,(outer,evaluation) in enumerate(splits):
                fitting,calibration=inner_holdout(outer,frame,strategy,.2,seed+1000+fold)
                roles={'fit':fitting,'conformal_calibration':calibration,'validation':evaluation}
                if task=='classification':
                    fitting,probability=inner_holdout(fitting,frame,strategy,.2,seed+2000+fold)
                    roles.update(fit=fitting,probability_calibration=probability)
                for role,index in roles.items():
                    if len(index)<2 or task=='classification' and set(frame.iloc[index].observed)!={0,1}:
                        raise ValueError(f'Unsupported {strategy} fold {fold}, role {role}; choose a different declared protocol')
                    for i in index:
                        output.append(dict(split_strategy=strategy,repeat=repeat,fold=fold,seed=seed,role=role,
                                           sample_id=frame.iloc[i].sample_id))
    result=pd.DataFrame(output)
    verify_assignments(frame,result,task,folds)
    return result


def verify_assignments(frame, roles, task, folds):
    known=set(frame.sample_id)
    for key,group in roles.groupby(['split_strategy','repeat','fold']):
        if set(group.sample_id)!=known or group.sample_id.duplicated().any():
            raise ValueError(f'Role membership does not partition development molecules: {key}')
        merged=group.merge(frame[['sample_id','scaffold','observed']],on='sample_id',validate='one_to_one')
        if key[0]=='scaffold' and (merged.groupby('scaffold').role.nunique()>1).any():
            raise ValueError('Scaffolds cross fitting/calibration/evaluation roles')
        expected={'fit','conformal_calibration','validation'}|({'probability_calibration'} if task=='classification' else set())
        if set(group.role)!=expected:
            raise ValueError('Missing required roles')
    for _,group in roles[roles.role=='validation'].groupby(['split_strategy','repeat']):
        if set(group.sample_id)!=known or group.sample_id.duplicated().any() or group.fold.nunique()!=folds:
            raise ValueError('Every chemical must have exactly one OOF prediction per repeat')


def implementation_hash():
    return implementation_signature()


def register(config_path, output):
    config_path=Path(config_path).resolve(); config=yaml.safe_load(config_path.read_text()); output=Path(output).resolve()
    if output.exists(): raise FileExistsError('Use a new registration directory; run resumes registered jobs')
    required={'study_id','folds','seeds','strategies','representations','models','members','trees','confidence','endpoints'}
    if not required <= config.keys():raise ValueError(f'Missing configuration keys: {required-config.keys()}')
    if config['folds']<2 or config['members']<2 or not 0<config['confidence']<1:raise ValueError('Invalid fold/member/confidence settings')
    if set(config['strategies'])-{'molecule','scaffold'}:raise ValueError('Unsupported split strategy')
    if set(config['representations'])-{'morgan','rdkit_descriptors','morgan_rdkit'}:raise ValueError('Unsupported representation')
    if set(config['models'])-{'linear','random_forest','extra_trees'}:raise ValueError('This registered runner supports classical models only')
    for key in ('seeds','strategies','representations','models','endpoints'):
        if not config[key]:raise ValueError(f'{key} must not be empty')
    for key in ('seeds','strategies','representations','models'):
        if len(set(config[key]))!=len(config[key]):raise ValueError(f'{key} must be unique')
    if len({s['name'] for s in config['endpoints']})!=len(config['endpoints']):raise ValueError('Endpoint names must be unique')
    prepared=[]
    with rdBase.BlockLogs():
        for spec in config['endpoints']:
            if not str(spec['name']).replace('_','').isalnum():raise ValueError('Endpoint names must be alphanumeric or underscore')
            frame,audit,summary=clean_pool(spec,config_path.parent)
            roles=assignments(frame,spec['task'],config['folds'],config['seeds'],config['strategies'])
            # Fail early if class-conditional or regression conformal quantiles would be infinite.
            for _,group in roles[roles.role=='conformal_calibration'].groupby(['split_strategy','repeat','fold']):
                targets=frame.set_index('sample_id').loc[group.sample_id].observed
                counts=targets.value_counts().tolist() if spec['task']=='classification' else [len(targets)]
                if any(np.ceil((n+1)*config['confidence'])>n for n in counts):raise ValueError('Insufficient conformal calibration support')
            prepared.append((spec,frame,audit,summary,roles))
    output.mkdir(parents=True)
    config['implementation_hash']=implementation_hash()
    config['runtime']=runtime_signature()
    config['environment_lock_sha256']=environment_lock()
    write_json(output/'protocol.json',config)
    for spec,frame,audit,summary,roles in prepared:
        directory=output/spec['name'];directory.mkdir()
        frame.to_csv(directory/'observations.csv',index=False)
        audit.to_csv(directory/'curation_audit.csv',index=False)
        roles.to_csv(directory/'assignments.csv',index=False)
        write_json(directory/'preflight.json',summary)
    bound=[p for p in output.rglob('*') if p.is_file()]
    write_json(output/'registration.json',{'study_id':config['study_id'],'artifacts':{str(p.relative_to(output)):digest(p) for p in bound},
        'evidence':'development; historical tests excluded; scaffold-shift conformal coverage is empirical'})
    return output


def verify_registration(root):
    root=Path(root); manifest=json.loads((root/'registration.json').read_text())
    for path,expected in manifest['artifacts'].items():
        if digest(root/path)!=expected:raise ValueError(f'Registered artifact changed: {path}')
    config=json.loads((root/'protocol.json').read_text())
    verify_compatibility(config)
    if config.get('environment_lock_sha256') != environment_lock():
        raise ValueError('Environment lock changed; restore it or create a new registration')
    return config


def job_paths(root,config):
    for spec in config['endpoints']:
        for strategy in config['strategies']:
            for repeat,_ in enumerate(config['seeds']):
                for fold in range(config['folds']):
                    for representation in config['representations']:
                        for model in config['models']:
                            context=dict(endpoint=spec['name'],task=spec['task'],split_strategy=strategy,repeat=repeat,fold=fold,preprocessing=representation,model=model)
                            key=f"{spec['name']}/{strategy}/r{repeat}_f{fold}/{representation}/{model}"
                            yield context,spec,Path(root)/'jobs'/key


def _predict(members,x,task):
    return np.array([m.predict(x) if task=='regression' else m.predict_proba(x)[:,1] for m in members])


def fit_members(x, y, task, name, config, seed):
    family = ('ridge' if task == 'regression' else 'logistic_regression') if name == 'linear' else name
    members = []
    for member in range(1 if name == 'linear' else config['members']):
        estimator = model_specs(task, [family], seed + member * 997)[0].estimator
        if name != 'linear':
            estimator.set_params(model__n_estimators=config['trees'], model__min_samples_leaf=2,
                model__max_features='sqrt', model__n_jobs=config.get('workers', 2))
        estimator.fit(x, y)
        members.append(estimator)
    return members


def fit_job(frame,roles,context,spec,config,x,directory,applicability):
    directory.mkdir(parents=True,exist_ok=True); start=time.monotonic()
    index={role:frame.index[frame.sample_id.isin(group.sample_id)].to_numpy() for role,group in roles.groupby('role')}
    fit,cal,validation=index['fit'],index['conformal_calibration'],index['validation']
    task=spec['task'];y=frame.observed.to_numpy(); seed=config['seeds'][context['repeat']]+10000*context['fold']
    name=context['model']; family=('ridge' if task=='regression' else 'logistic_regression') if name=='linear' else name
    members=fit_members(x[fit],y[fit],task,name,config,seed)
    cal_members=_predict(members,x[cal],task); val_members=_predict(members,x[validation],task)
    predicted=val_members.mean(axis=0);spread=val_members.std(axis=0,ddof=1) if len(members)>1 else np.zeros(len(validation))
    raw_probability=predicted.copy(); probability_calibrator=None
    if task=='regression':
        floor=max(float(np.std(y[fit]))*.05,1e-6) # training targets only, fixed policy
        cal_spread=cal_members.std(axis=0,ddof=1) if len(members)>1 else np.zeros(len(cal))
        calibration=regression_calibration(y[cal],cal_members.mean(axis=0),cal_spread,config['confidence'],floor)
        lower,upper=regression_intervals(predicted,spread,calibration)
        extra={'prediction_lower':lower,'prediction_upper':upper,
               'absolute_lower':predicted-calibration['absolute_q'],'absolute_upper':predicted+calibration['absolute_q']}
        uncertainty=spread if name!='linear' else np.full(len(validation),calibration['absolute_q'])
    else:
        prob=index['probability_calibration'];mean_prob=_predict(members,x[prob],task).mean(axis=0)
        probability_calibrator=ProbabilityCalibrator().fit(mean_prob,y[prob])
        predicted=probability_calibrator.predict(predicted)
        calibration=classification_calibration(y[cal],probability_calibrator.predict(cal_members.mean(axis=0)),config['confidence'])
        sets=prediction_sets(predicted,calibration)
        extra={'set_0':sets[:,0].astype(int),'set_1':sets[:,1].astype(int),'raw_probability':raw_probability}
        p=np.clip(predicted,1e-12,1-1e-12)
        uncertainty=-(p*np.log(p)+(1-p)*np.log1p(-p))
    output=pd.DataFrame({**context,'partition':'validation','sample_id':frame.iloc[validation].sample_id.to_numpy(),
        'molecule_id':frame.iloc[validation].sample_id.to_numpy(),'smiles':frame.iloc[validation].smiles.to_numpy(),
        'observed':y[validation],'observed_submitted':frame.iloc[validation].observed_raw.to_numpy(),
        'prediction':predicted,'uncertainty':uncertainty,'ensemble_std':spread,'threshold':.5,
        'target_transform':spec['evaluation_scale'],**extra,**applicability})
    output['nominal_coverage']=config['confidence']
    output.to_csv(directory/'predictions.csv',index=False)
    pd.DataFrame(val_members.T,columns=[f'member_{i}' for i in range(len(members))]).assign(sample_id=output.sample_id.to_numpy()).to_csv(directory/'member_predictions.csv',index=False)
    write_json(directory/'calibration.json',calibration)
    bundle={'members':members,'task':task,'representation':context['preprocessing'],'calibrator':probability_calibrator,
        'calibration':calibration,'context':context,'evaluation_scale':spec['evaluation_scale'],
        'training_smiles':frame.iloc[fit].smiles.tolist(),'training_ids':frame.iloc[fit].sample_id.tolist(),
        'implementation_hash':config['implementation_hash'],'runtime':config['runtime'],
        'feature_schema':feature_schema(context['preprocessing']),'model_role':'outer-fold predictor; not a full-development refit'}
    joblib.dump(bundle,directory/'model.joblib',compress=3)
    write_json(directory/'complete.json',{'context':context,'seconds':time.monotonic()-start,
        'role_counts':{k:len(v) for k,v in index.items()},'implementation_hash':config['implementation_hash'],
        'artifacts':{name:digest(directory/name) for name in ['predictions.csv','member_predictions.csv','calibration.json','model.joblib']}})
    return output


def status_report(root,config):
    jobs=[]
    for context,_,directory in job_paths(root,config):
        complete=directory/'complete.json';failed=directory/'failed.json';running=directory/'running.json'
        state='complete' if complete.exists() else 'failed' if failed.exists() else 'running or interrupted' if running.exists() else 'planned'
        detail=json.loads(complete.read_text()) if complete.exists() else {}
        jobs.append({**context,'state':state,'seconds':detail.get('seconds'),'path':str(directory.relative_to(root))})
    write_json(Path(root)/'status.json',jobs)
    columns=['endpoint','split_strategy','repeat','fold','preprocessing','model','state','seconds']
    table='<table><thead><tr>'+''.join(f'<th>{c}</th>' for c in columns)+'</tr></thead><tbody>'
    for job in jobs:table+='<tr>'+''.join('<td>'+html.escape(str(job.get(c,'')))+'</td>' for c in columns)+'</tr>'
    counts={state:sum(j['state']==state for j in jobs) for state in sorted({j['state'] for j in jobs})}
    document='<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>CV experiment progress</title><style>body{font:15px system-ui;margin:3%;color:#153b36;background:#faf8f1}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}th{position:sticky;top:0;background:#e5eee8}</style><h1>Real CV and uncertainty</h1><p>Local experiment status. Refresh to read updated progress. Resume with the run command; completed artifacts are hash-checked.</p><p>'+html.escape(str(counts))+'</p>'+table+'</tbody></table>'
    (Path(root)/'progress.html').write_text(document)
    return counts


def run(root,limit=None):
    root=Path(root).resolve();config=verify_registration(root);cache={};completed=0
    with rdBase.BlockLogs(),threadpool_limits(limits=config.get('workers',2)):
        for context,spec,directory in job_paths(root,config):
            if (directory/'complete.json').exists():
                manifest=json.loads((directory/'complete.json').read_text())
                for name,expected in manifest['artifacts'].items():
                    if digest(directory/name)!=expected:raise ValueError(f'Completed job artifact changed: {directory/name}')
                continue
            if limit is not None and completed>=limit:break
            endpoint=spec['name']
            if endpoint not in cache:
                frame=pd.read_csv(root/endpoint/'observations.csv',keep_default_na=False)
                roles=pd.read_csv(root/endpoint/'assignments.csv')
                cache[endpoint]={'frame':frame,'roles':roles,'features':{},'applicability':{}}
            item=cache[endpoint];frame=item['frame'];all_roles=item['roles']
            role=all_roles[(all_roles.split_strategy==context['split_strategy'])&(all_roles.repeat==context['repeat'])&(all_roles.fold==context['fold'])]
            rep=context['preprocessing']
            if rep not in item['features']:item['features'][rep]=build_features(frame.smiles.tolist(),representation=rep,descriptor_set='full').values
            appkey=(context['split_strategy'],context['repeat'],context['fold'])
            if appkey not in item['applicability']:
                fit=frame[frame.sample_id.isin(role[role.role=='fit'].sample_id)];val=frame[frame.sample_id.isin(role[role.role=='validation'].sample_id)]
                app=training_neighbours(fit.smiles.tolist(),val.smiles.tolist())
                app['nearest_train_id']=[fit.iloc[i].sample_id for i in app.pop('nearest_train_index')]
                if spec['task']=='regression':app.update(cliff_annotations(fit.smiles.tolist(),fit.observed.to_numpy(),val.smiles.tolist(),val.observed.to_numpy(),difference=spec.get('cliff_difference',1.)))
                item['applicability'][appkey]=app
            directory.mkdir(parents=True,exist_ok=True);write_json(directory/'running.json',{'started_at':time.time(),'context':context});status_report(root,config)
            try:
                result=fit_job(frame,role,context,spec,config,item['features'][rep],directory,item['applicability'][appkey])
                print(json.dumps({'completed':context,'observations':len(result)}),flush=True)
                (directory/'running.json').unlink(missing_ok=True);(directory/'failed.json').unlink(missing_ok=True)
            except Exception as error:
                write_json(directory/'failed.json',{'error':str(error),'context':context});status_report(root,config);raise
            completed+=1;status_report(root,config)
    counts=status_report(root,config)
    if counts.get('complete',0)==len(list(job_paths(root,config))):assemble(root,config)
    return counts


def evaluate_predictions(frame,confidence):
    task=frame.task.iloc[0];y=frame.observed.to_numpy();p=frame.prediction.to_numpy()
    values=regression_metrics(y,p) if task=='regression' else classification_metrics(y,p)
    if task=='regression':
        for prefix,lo,hi in [('normalized','prediction_lower','prediction_upper'),('absolute','absolute_lower','absolute_upper')]:
            lower=frame[lo].to_numpy();upper=frame[hi].to_numpy()
            values.update({prefix+'_coverage':float(np.mean((y>=lower)&(y<=upper))),prefix+'_mean_width':float(np.mean(upper-lower)),prefix+'_interval_score':float(interval_score(y,lower,upper,confidence).mean())})
    else:
        sets=frame[['set_0','set_1']].to_numpy(dtype=int);covered=sets[np.arange(len(y)),y.astype(int)]
        values.update(brier=brier_score_loss(y,p),log_loss=log_loss(y,p,labels=[0,1]),raw_brier=brier_score_loss(y,frame.raw_probability),raw_log_loss=log_loss(y,frame.raw_probability,labels=[0,1]),set_coverage=float(covered.mean()),mean_set_size=float(sets.sum(axis=1).mean()),singleton_rate=float(np.mean(sets.sum(axis=1)==1)))
        for c in (0,1):values[f'class_{c}_coverage']=float(covered[y==c].mean()) if np.any(y==c) else np.nan
    from .evaluation import _uncertainty_metrics
    values.update(_uncertainty_metrics(y,p,frame.uncertainty.to_numpy()))
    return values


def assemble(root,config):
    metric_rows=[];cohorts=[];bundles=[]
    for spec in config['endpoints']:
        endpoint=spec['name'];frames=[pd.read_csv(path/'predictions.csv') for context,_,path in job_paths(root,config) if context['endpoint']==endpoint]
        frame=pd.concat(frames,ignore_index=True)
        # Reuse the prepared-prediction boundary before attaching richer UQ fields.
        from .evaluation import canonicalize_predictions, validate_predictions
        bundle=Path(root)/'evaluation'/endpoint;bundle.mkdir(parents=True,exist_ok=True)
        frame.to_csv(bundle/'submitted_predictions.csv',index=False)
        fields=['sample_id','molecule_id','smiles','endpoint','model','partition','observed','prediction','repeat','fold','split_strategy','preprocessing','uncertainty']
        if spec['task']=='regression':fields+=['prediction_lower','prediction_upper']
        evaluation_config={'schema_version':1,'task':spec['task'],'predictions':{'path':str(bundle/'submitted_predictions.csv'),'columns':{c:c for c in fields}},
            'thresholds':{'default':.5},'target_transform':{'default':'none'},'uncertainty':{'higher_is_more_uncertain':True,'scale':'evaluation','interval_confidence':config['confidence']},
            '_config_dir':str(root)}
        canonical,_=canonicalize_predictions(evaluation_config)
        validated,validation=validate_predictions(canonical,evaluation_config)
        # Preserve producer-specific calibrated outputs alongside common identity/UQ fields.
        for name in ('canonical_smiles','chemical_id','uncertainty_percentile'):
            frame[name]=validated[name].to_numpy()
        if validation['errors']:raise ValueError(str(validation['errors']))
        write_json(bundle/'validation_report.json',validation)
        keys=['task','endpoint','preprocessing','split_strategy','repeat','fold','model']
        for context,group in frame.groupby(keys):
            base=dict(zip(keys,context,strict=True)); values=evaluate_predictions(group,config['confidence'])
            metric_rows.extend({**base,'partition':'validation','metric':metric,'value':float(value),'n_molecules':len(group)} for metric,value in values.items() if np.isfinite(value))
            for cohort,mask in [('all',np.ones(len(group),dtype=bool)),('similarity_lt_0.5',group.nearest_train_similarity<.5),('similarity_ge_0.5',group.nearest_train_similarity>=.5),('novel_scaffold',group.novel_scaffold==1),('property_cliff',group.get('property_cliff',pd.Series(0,index=group.index))==1)]:
                selected=group[mask]
                if len(selected)<10:continue
                y=selected.observed.to_numpy();p=selected.prediction.to_numpy()
                covered=(y>=selected.prediction_lower)&(y<=selected.prediction_upper) if spec['task']=='regression' else selected[['set_0','set_1']].to_numpy()[np.arange(len(y)),y.astype(int)]
                cohorts.append({**base,'cohort':cohort,'n_molecules':len(selected),'mean_absolute_error':float(np.abs(y-p).mean()),'coverage':float(np.mean(covered))})
        bundle=Path(root)/'evaluation'/endpoint;bundle.mkdir(parents=True,exist_ok=True)
        frame.to_csv(bundle/'predictions.csv',index=False)
        pd.DataFrame([r for r in metric_rows if r['endpoint']==endpoint]).to_csv(bundle/'fold_metrics.csv',index=False)
        (bundle/'resolved_config.yaml').write_text(yaml.safe_dump({'uncertainty':{'interval_confidence':config['confidence']},'evidence':'real outer CV; development only; historical tests excluded','cv_evidence':True,'selected_model':'random_forest' if 'random_forest' in config['models'] else config['models'][0], 'uq_description':'Primary score: calibrated predictive entropy; class-conditional conformal sets' if spec['task']=='classification' else 'Primary score: ensemble standard deviation (linear baseline: constant absolute conformal radius); normalized and absolute conformal intervals'}))
        bundles.append(bundle)
    pd.DataFrame(metric_rows).to_csv(Path(root)/'fold_metrics.csv',index=False)
    pd.DataFrame(cohorts).to_csv(Path(root)/'chemical_cohorts.csv',index=False)
    write_json(Path(root)/'evaluation_manifest.json',{'registration_sha256':digest(Path(root)/'registration.json'),'artifacts':{str(p.relative_to(root)):digest(p) for bundle in bundles for p in bundle.iterdir()},'evidence':'Development OOF evaluation; fold metrics are dependent; chemical cohorts are post hoc'})
    return bundles


def predict(model_dir,input_csv,output_csv,cutoff=None):
    directory=Path(model_dir).resolve();manifest=json.loads((directory/'complete.json').read_text())
    if digest(directory/'model.joblib')!=manifest['artifacts']['model.joblib']:raise ValueError('Model artifact hash mismatch')
    model=joblib.load(directory/'model.joblib') # trusted local artifact only
    verify_compatibility(model, representation=model['representation'])
    frame=read_csv(input_csv)
    if frame.empty or 'smiles' not in frame:raise ValueError('Prediction input requires at least one smiles row')
    smiles=frame.smiles.map(chemical).tolist()
    x=build_features(smiles,representation=model['representation'],descriptor_set='full').values
    members=_predict(model['members'],x,model['task']);p=members.mean(axis=0);std=members.std(axis=0,ddof=1) if len(members)>1 else np.zeros(len(smiles))
    result=pd.DataFrame({'smiles':smiles,'prediction':p,'ensemble_std':std})
    for identity in ('sample_id','molecule_id'):
        if identity in frame:result.insert(0,identity,frame[identity].to_numpy())
    if model['task']=='regression':
        lower,upper=regression_intervals(p,std,model['calibration']);result['prediction_lower']=lower;result['prediction_upper']=upper
        if cutoff is not None:result['cutoff_decision']=np.where(lower>cutoff,'above',np.where(upper<cutoff,'below','crosses'))
        if model['evaluation_scale']=='log10':
            for c in ['prediction','prediction_lower','prediction_upper']:result[c+'_raw']=10**result[c]
    else:
        result['prediction']=model['calibrator'].predict(p);sets=prediction_sets(result.prediction.to_numpy(),model['calibration'])
        result['set_0']=sets[:,0].astype(int);result['set_1']=sets[:,1].astype(int)
    app=training_neighbours(model['training_smiles'],smiles);app['nearest_train_id']=[model['training_ids'][i] for i in app.pop('nearest_train_index')]
    for key,value in app.items():result[key]=value
    result['model_sha256']=manifest['artifacts']['model.joblib'];result['nominal_coverage']=model['calibration']['confidence'];result['model_role']=model['model_role'];result['evaluation_scale']=model['evaluation_scale']
    Path(output_csv).parent.mkdir(parents=True,exist_ok=True);result.to_csv(output_csv,index=False)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    prepare=sub.add_parser('register');prepare.add_argument('config');prepare.add_argument('--output',required=True)
    run_parser=sub.add_parser('run');run_parser.add_argument('study');run_parser.add_argument('--max-jobs',type=int)
    status=sub.add_parser('status');status.add_argument('study')
    final=sub.add_parser('refit');final.add_argument('study');final.add_argument('--output',required=True)
    prediction=sub.add_parser('predict');prediction.add_argument('model_dir');prediction.add_argument('input_csv');prediction.add_argument('--output',required=True);prediction.add_argument('--cutoff',type=float)
    args=parser.parse_args()
    if args.command=='register':print(register(args.config,args.output))
    elif args.command=='run':print(run(args.study,args.max_jobs))
    elif args.command=='refit':
        from .deployment import refit
        print(refit(args.study,args.output))
    elif args.command=='status':print(status_report(Path(args.study),verify_registration(args.study)))
    else:print(predict(args.model_dir,args.input_csv,args.output,args.cutoff).to_string(index=False))

if __name__=='__main__':main()
