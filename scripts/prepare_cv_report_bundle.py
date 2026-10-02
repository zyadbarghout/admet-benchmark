"""Make a display bundle with original molecule names, preserving evaluated artifacts."""
from pathlib import Path
import json
import shutil
import yaml
import pandas as pd
from admet_benchmark.cross_validation import digest
from admet_benchmark.tables import read_csv


def prepare(study):
    study=Path(study).resolve();config=json.loads((study/'protocol.json').read_text());outputs=[]
    for spec in config['endpoints']:
        endpoint=spec['name'];observations=pd.read_csv(study/endpoint/'observations.csv',dtype={'source_id':str,'sample_id':str})
        preflight=json.loads((study/endpoint/'preflight.json').read_text());names={}
        for source,expected in preflight['source_hashes'].items():
            source=Path(source)
            if source.name not in set(observations.source_file):continue
            assert digest(source)==expected
            original=read_csv(source)
            name=next((c for c in ['molecule_id','Molecule Name','name'] if c in original),None)
            if name:
                for _,row in original.iterrows():names[(source.name,str(row['sample_id']))]=str(row[name])
        mapping={row.sample_id:names.get((row.source_file,row.source_id),row.source_id) for row in observations.itertuples()}
        original_ids=observations.set_index('sample_id').source_id.to_dict()
        source=study/'evaluation'/endpoint;destination=study/'report_data'/endpoint;destination.mkdir(parents=True,exist_ok=True)
        predictions=pd.read_csv(source/'predictions.csv',dtype={'sample_id':str});predictions['molecule_id']=predictions.sample_id.map(mapping);predictions['original_sample_id']=predictions.sample_id.map(original_ids)
        predictions.to_csv(destination/'predictions.csv',index=False)
        for name in ['fold_metrics.csv','resolved_config.yaml','validation_report.json']:shutil.copyfile(source/name,destination/name)
        display = yaml.safe_load((destination/'resolved_config.yaml').read_text()) or {}
        labels = {'KSOL': ('KSOL · kinetic solubility', 'log10 source assay units'),
                  'LogD': ('LogD · lipophilicity', 'LogD (source logarithmic scale)'),
                  'BBBP': ('BBBP · brain permeability', 'probability of class 1')}
        if endpoint in labels:
            display['display_name'], display['evaluation_units'] = labels[endpoint]
            (destination/'resolved_config.yaml').write_text(yaml.safe_dump(display, sort_keys=False))
        (destination/'display_provenance.json').write_text(json.dumps({'source_prediction_sha256':digest(source/'predictions.csv'),'display_prediction_sha256':digest(destination/'predictions.csv'),'change':'Restored original display names and source IDs from hash-verified source data; predictions and role membership unchanged'},indent=2)+'\n')
        outputs.append(destination)
    return outputs


if __name__=='__main__':
    import sys
    print('\n'.join(map(str,prepare(sys.argv[1]))))
