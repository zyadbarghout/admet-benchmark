"""Audit registered OOF outputs and write the measured outcome, without model tuning."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from admet_benchmark.cross_validation import verify_registration,job_paths,digest,predict,evaluate_predictions


def evaluate(root):
    root=Path(root).resolve();config=verify_registration(root);audits=[];summary=[];reloaded=[]
    # Morgan linear models use float32 inputs; BLAS batch sizes can change
    # last-bit rounding, including interval endpoints near zero. Keep an
    # absolute floor and report measured discrepancies, not just pass/fail.
    tolerance = {"rtol": 1e-6, "atol": 1e-6}
    for context,spec,directory in job_paths(root,config):
        complete=json.loads((directory/'complete.json').read_text())
        assert complete['context']==context
        for file,expected in complete['artifacts'].items():assert digest(directory/file)==expected
        predictions=pd.read_csv(directory/'predictions.csv');observations=pd.read_csv(root/spec['name']/'observations.csv');roles=pd.read_csv(root/spec['name']/'assignments.csv')
        assigned=roles[(roles.split_strategy==context['split_strategy'])&(roles.repeat==context['repeat'])&(roles.fold==context['fold'])&(roles.role=='validation')]
        assert predictions.sample_id.is_unique and set(predictions.sample_id)==set(assigned.sample_id)
        target=observations.set_index('sample_id').loc[predictions.sample_id].observed.to_numpy()
        np.testing.assert_allclose(target,predictions.observed)
        audits.append({'context':context,'rows':len(predictions),'artifacts':'verified','oof_membership':'exact'})
        if context['fold']==0 and context['split_strategy']=='molecule':
            queries=root/'inference_checks'/spec['name']/context['preprocessing']/context['model']/f"repeat_{context['repeat']}";queries.mkdir(parents=True,exist_ok=True)
            predictions[['smiles']].head(5).to_csv(queries/'input.csv',index=False)
            result=predict(directory,queries/'input.csv',queries/'predictions.csv',cutoff=1 if spec['task']=='regression' else None)
            differences = {}
            columns = ['prediction'] + (['prediction_lower','prediction_upper'] if spec['task']=='regression' else ['set_0','set_1'])
            for column in columns:
                actual, expected = result[column].to_numpy(), predictions[column].head(5).to_numpy()
                if column.startswith('set_'):
                    np.testing.assert_array_equal(actual, expected)
                else:
                    np.testing.assert_allclose(actual, expected, **tolerance)
                differences[column] = float(np.max(np.abs(actual - expected)))
            reloaded.append({'context': context, 'max_absolute_differences': differences})
    for spec in config['endpoints']:
        predictions=pd.read_csv(root/'evaluation'/spec['name']/'predictions.csv')
        expected=len(pd.read_csv(root/spec['name']/'observations.csv'))
        for keys,group in predictions.groupby(['preprocessing','split_strategy','repeat','model']):
            assert len(group)==expected and group.sample_id.is_unique
            metrics=evaluate_predictions(group,config['confidence'])
            summary.append({'endpoint':spec['name'],'task':spec['task'],'representation':keys[0],'split':keys[1],'repeat':int(keys[2]),'model':keys[3],'n_chemicals':len(group),**metrics})
    table=pd.DataFrame(summary);table.to_csv(root/'pooled_oof_outcomes.csv',index=False)
    (root/'outcome_audit.json').write_text(json.dumps({'jobs_verified':len(audits),'prediction_rows':sum(a['rows'] for a in audits),'reload_checks':len(reloaded),'reload_tolerance':tolerance,'reload_results':reloaded,'checks':audits,'interpretation':'One OOF prediction per chemical in each method/split/repeat. Summaries are descriptive; model choices were not changed after evaluation.'},indent=2)+'\n')
    return table


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('study');a=p.parse_args()
    table=evaluate(a.study);print(table[['endpoint','representation','split','model','mae' if 'mae' in table else 'pr_auc','normalized_coverage','set_coverage']].to_string(index=False))
