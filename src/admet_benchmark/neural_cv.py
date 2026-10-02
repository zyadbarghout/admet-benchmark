"""Optional fixed-epoch Chemprop/CheMeleon adapter for registered outer folds.

Uses the optional models environment. It creates its own addon manifest and
never reuses historical checkpoints or changes the registered role assignments.
A short run is a plumbing smoke test, not a tuned neural benchmark.
"""
from __future__ import annotations
import json
import sys
from importlib.metadata import version
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from .cross_validation import verify_registration,digest,write_json
from .benchmark import _write_chemprop_data,_chemprop_train_command,_run_command,_chemprop_predict,_chemprop_accelerator
from .tables import read_csv
from .applicability import training_neighbours, cliff_annotations
from .provenance import runtime_signature
from .uq import ProbabilityCalibrator,regression_calibration,regression_intervals,classification_calibration,prediction_sets


def run_neural_fold(study,output,endpoint='BBBP',strategy='molecule',fold=0,model='chemprop',epochs=50,members=2,accelerator='auto'):
    study=Path(study).resolve();output=Path(output).resolve();config=verify_registration(study)
    if output.exists():raise FileExistsError('Use a fresh neural addon directory')
    if model not in {'chemprop','chemeleon'} or epochs<1 or members<2:raise ValueError('Invalid neural model or training settings')
    spec=next(s for s in config['endpoints'] if s['name']==endpoint);task=spec['task']
    frame=read_csv(study/endpoint/'observations.csv');roles=read_csv(study/endpoint/'assignments.csv')
    roles=roles[(roles.split_strategy==strategy)&(roles.repeat==0)&(roles.fold==fold)]
    parts={role:frame[frame.sample_id.isin(group.sample_id)].rename(columns={'observed':'target_model'}).reset_index(drop=True) for role,group in roles.groupby('role')}
    if not {'fit','conformal_calibration','validation'} <= parts.keys():raise ValueError('Requested fold is unavailable')
    output.mkdir(parents=True);accelerator=_chemprop_accelerator(accelerator)
    write_json(output/'protocol.json',{'registration_sha256':digest(study/'registration.json'),'endpoint':endpoint,'strategy':strategy,'repeat':0,'fold':fold,'model':model,'epochs':epochs,'members':members,'accelerator':accelerator,'selection':'Fixed epochs; no early stopping, tuning, calibration or outer validation feedback','python':sys.version,'runtime':runtime_signature(),'neural_runtime':{n:version(n) for n in ('torch','chemprop','lightning')},'task':task,'evaluation_scale':spec['evaluation_scale'],'confidence':config['confidence'],'implementation_sha256':digest(__file__),'adapter_sha256':digest(Path(__file__).with_name('benchmark.py'))})
    train=parts['fit'].copy();train['split']='train';_write_chemprop_data(train,'smiles',output/'fit.csv')
    roles.to_csv(output/'roles.csv',index=False)
    predictions={role:[] for role in parts if role!='fit'};checkpoints=[]
    for member in range(members):
        directory=output/f'member_{member}'
        candidate={'epochs':epochs,'patience':0,'batch_size':128,'depth':3,'message_hidden_dim':300,'ffn_hidden_dim':300,'ffn_num_layers':1,'dropout':.05,'max_lr':.001,'init_lr':.0001,'final_lr':.0001}
        command=_chemprop_train_command(task=task,train_path=output/'fit.csv',use_splits_column=True,output_dir=directory/'training',candidate=candidate,model_name=model,seed=config['seeds'][0]+member*997,accelerator=accelerator,final_refit=True)
        _run_command(command,directory,'train');checkpoint=directory/'training/model_0/best.pt';checkpoints.append(checkpoint)
        for role in predictions:predictions[role].append(_chemprop_predict(checkpoint,parts[role],'smiles',directory/role,accelerator))
        print(f'{model} member {member+1}/{members} trained and projected on registered roles',flush=True)
    values={role:np.asarray(p) for role,p in predictions.items()};validation=parts['validation'];cal=parts['conformal_calibration'];p=values['validation'].mean(axis=0);spread=values['validation'].std(axis=0,ddof=1)
    if task=='regression':
        calibration=regression_calibration(cal.target_model.to_numpy(),values['conformal_calibration'].mean(axis=0),values['conformal_calibration'].std(axis=0,ddof=1),config['confidence'],max(float(parts['fit'].target_model.std())*.05,1e-6))
        lower,upper=regression_intervals(p,spread,calibration);extras={'prediction_lower':lower,'prediction_upper':upper,'absolute_lower':p-calibration['absolute_q'],'absolute_upper':p+calibration['absolute_q']};uq=spread
    else:
        mapping=ProbabilityCalibrator().fit(values['probability_calibration'].mean(axis=0),parts['probability_calibration'].target_model.to_numpy())
        raw=p.copy();p=mapping.predict(p);calibration=classification_calibration(cal.target_model.to_numpy(),mapping.predict(values['conformal_calibration'].mean(axis=0)),config['confidence'])
        sets=prediction_sets(p,calibration);extras={'set_0':sets[:,0].astype(int),'set_1':sets[:,1].astype(int),'raw_probability':raw}
        clipped=np.clip(p,1e-12,1-1e-12);uq=-(clipped*np.log(clipped)+(1-clipped)*np.log1p(-clipped))
        import joblib
        joblib.dump(mapping,output/'probability_calibrator.joblib')
    result=pd.DataFrame({'sample_id':validation.sample_id,'smiles':validation.smiles,'observed':validation.target_model,'prediction':p,'uncertainty':uq,'ensemble_std':spread,**extras})
    result['molecule_id']=result.sample_id
    result['endpoint']=endpoint;result['task']=task;result['model']=model
    result['preprocessing']='graph';result['split_strategy']=strategy
    result['repeat']=0;result['fold']=fold;result['partition']='validation'
    result['threshold']=.5;result['target_transform']=spec['evaluation_scale']
    result['observed_submitted']=validation.observed_raw.to_numpy()
    result['nominal_coverage']=config['confidence']
    app=training_neighbours(parts['fit'].smiles.tolist(),validation.smiles.tolist())
    app['nearest_train_id']=[parts['fit'].iloc[i].sample_id for i in app.pop('nearest_train_index')]
    if task=='regression':app.update(cliff_annotations(parts['fit'].smiles.tolist(),parts['fit'].target_model.to_numpy(),validation.smiles.tolist(),validation.target_model.to_numpy(),difference=spec.get('cliff_difference',1.)))
    for key,value in app.items():result[key]=value
    result.to_csv(output/'predictions.csv',index=False);write_json(output/'calibration.json',calibration)
    write_json(output/'manifest.json',{'protocol_sha256':digest(output/'protocol.json'),'prediction_sha256':digest(output/'predictions.csv'),'checkpoints':{str(p.relative_to(output)):digest(p) for p in checkpoints},'artifacts':{str(p.relative_to(output)):digest(p) for p in [output/'protocol.json',output/'predictions.csv',output/'calibration.json',output/'roles.csv',output/'fit.csv',*([output/'probability_calibrator.joblib'] if task=='classification' else [])]},'evidence':f'Registered outer fold; {epochs} fixed epochs, {members} independently seeded members; no tuning'})
    return result


def predict_neural(directory, input_csv, output_csv, accelerator='auto'):
    directory=Path(directory).resolve()
    manifest=json.loads((directory/'manifest.json').read_text())
    for name,expected in {**manifest['artifacts'],**manifest['checkpoints']}.items():
        if digest(directory/name)!=expected:raise ValueError(f'Neural artifact changed: {name}')
    protocol=json.loads((directory/'protocol.json').read_text())
    if protocol['runtime']!=runtime_signature() or protocol['neural_runtime']!={n:version(n) for n in ('torch','chemprop','lightning')}:
        raise ValueError('Neural runtime compatibility mismatch')
    if protocol['implementation_sha256']!=digest(__file__) or protocol['adapter_sha256']!=digest(Path(__file__).with_name('benchmark.py')):
        raise ValueError('Neural adapter implementation changed')
    frame=read_csv(input_csv)
    if frame.empty or 'smiles' not in frame:raise ValueError('Neural prediction requires smiles rows')
    accelerator=_chemprop_accelerator(accelerator)
    import tempfile
    with tempfile.TemporaryDirectory(prefix='admet-neural-predict-') as temporary:
        values=np.array([_chemprop_predict(directory/name,frame,'smiles',Path(temporary)/str(i),accelerator)
            for i,name in enumerate(manifest['checkpoints'])])
    p=values.mean(axis=0);spread=values.std(axis=0,ddof=1)
    result=frame[[c for c in ('sample_id','molecule_id','smiles') if c in frame]].copy()
    calibration=json.loads((directory/'calibration.json').read_text())
    if protocol['task']=='classification':
        p=joblib.load(directory/'probability_calibrator.joblib').predict(p)
        sets=prediction_sets(p,calibration);result['set_0']=sets[:,0].astype(int);result['set_1']=sets[:,1].astype(int)
    else:
        lo,hi=regression_intervals(p,spread,calibration);result['prediction_lower']=lo;result['prediction_upper']=hi
    result['prediction']=p;result['ensemble_std']=spread
    result['evaluation_scale']=protocol['evaluation_scale'];result['nominal_coverage']=protocol['confidence']
    result['model_role']='registered outer-fold neural ensemble';result['manifest_sha256']=digest(directory/'manifest.json')
    Path(output_csv).parent.mkdir(parents=True,exist_ok=True);result.to_csv(output_csv,index=False)
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('study');parser.add_argument('--output',required=True);parser.add_argument('--endpoint',default='BBBP');parser.add_argument('--strategy',default='molecule');parser.add_argument('--fold',type=int,default=0);parser.add_argument('--model',choices=['chemprop','chemeleon'],default='chemprop');parser.add_argument('--epochs',type=int,default=50);parser.add_argument('--members',type=int,default=2);parser.add_argument('--accelerator',default='auto');args=parser.parse_args()
    print(run_neural_fold(args.study,args.output,args.endpoint,args.strategy,args.fold,args.model,args.epochs,args.members,args.accelerator).shape)
