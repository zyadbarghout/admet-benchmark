"""Descriptive fold-level UQ evidence; no independent-fold confidence claims."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot(root):
    root=Path(root)
    rows=[]
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for panel,endpoint in enumerate(['KSOL','BBBP']):
        frame=pd.read_csv(root/'evaluation'/endpoint/'predictions.csv')
        frame=frame[(frame.preprocessing=='rdkit_descriptors')&(frame.model=='random_forest')]
        groups=[]
        for split in ['molecule','scaffold']:
            selected=frame[frame.split_strategy==split]
            cohorts=[('All',selected)]
            if endpoint=='KSOL':cohorts.append(('Property cliffs',selected[selected.property_cliff==1]))
            for cohort,group in cohorts:
                coverage=((group.observed>=group.prediction_lower)&(group.observed<=group.prediction_upper)) if endpoint=='KSOL' else np.where(group.observed==0,group.set_0,group.set_1)
                by_fold=pd.DataFrame({'fold':group.fold,'covered':coverage}).groupby('fold').covered.mean()
                groups.append((f'{split.title()}\n{cohort}\nn={len(group):,}',by_fold))
                rows.append(dict(endpoint=endpoint,split=split,cohort=cohort,n=len(group),pooled_coverage=float(np.mean(coverage))))
        ax=axes[panel]
        for x,(label,values) in enumerate(groups):
            ax.scatter(np.full(len(values),x)+np.linspace(-.09,.09,len(values)),values,s=42,color='#16617a',alpha=.85)
        ax.axhline(.9,color='#BC4565',ls='--',lw=1.4,label='90% nominal reference')
        ax.set(xticks=range(len(groups)),xticklabels=[label for label,_ in groups],ylim=(.45,1.01),ylabel='Observed coverage in each outer fold',title=f'{endpoint} · '+('normalized regression intervals' if endpoint=='KSOL' else 'classification prediction sets'))
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18);ax.legend(loc='lower left',frameon=False)
    fig.suptitle('RDKit random forest · five outer folds · measured development data',fontsize=14)
    fig.savefig(root/'uq_outcome.png',dpi=180);fig.savefig(root/'uq_outcome.pdf');plt.close(fig)
    cvplus=pd.read_csv(root.parent/'cvplus_pilot_2026_09_09'/'predictions.csv')
    cv={'n':len(cvplus),'mae':float(abs(cvplus.prediction-cvplus.observed).mean()),'coverage':float(((cvplus.observed>=cvplus.lower)&(cvplus.observed<=cvplus.upper)).mean()),'mean_width':float((cvplus.upper-cvplus.lower).mean())}
    (root/'uq_outcome_summary.json').write_text(json.dumps({'cohorts':rows,'cvplus':cv},indent=2)+'\n')
    print(json.dumps({'cohorts':rows,'cvplus':cv},indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('study');args=parser.parse_args();plot(args.study)
