"""Offline browser acceptance tests. Run with RUN_BROWSER_TESTS=1 pytest -q."""
from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(os.environ.get('RUN_BROWSER_TESTS') != '1', reason='Set RUN_BROWSER_TESTS=1 and install the browser extra plus Chromium')


def make_report(tmp_path):
    from admet_benchmark.chemical_report import generate_chemical_report
    run = tmp_path / 'bundle'
    run.mkdir()
    observations, metrics = [], []
    for endpoint, task in [('Solubility demo', 'regression'), ('Permeability demo', 'classification')]:
        for partition, smiles_list in [('validation', ['CCO','CCN','CCC','CCCl']), ('test', ['CCCO','CCCN','CCCC','CCCCl'])]:
            for repeat in (0, 1):
                for model, delta in [('baseline', .1), ('tree', .2)]:
                    for index, smiles in enumerate(smiles_list):
                        observed = float(index % 2) if task == 'classification' else float(index * 10)
                        prediction = .5 if task == 'classification' else observed + delta * (index + 1)
                        observations.append(dict(task=task,endpoint=endpoint,model=model,partition=partition,repeat=repeat,fold=0,sample_id=f'{partition}-{index}',molecule_id=f'{partition}-{index}',smiles=smiles,observed=observed,prediction=prediction,preprocessing='raw',split_strategy='molecule',target_transform='none',uncertainty=delta*(index+1),uncertainty_percentile=(index+1)/4,prediction_lower=prediction-1,prediction_upper=prediction+1))
                    metrics.append(dict(task=task,endpoint=endpoint,model=model,partition=partition,repeat=repeat,fold=0,metric='mae' if task=='regression' else 'pr_auc',value=delta,preprocessing='raw',split_strategy='molecule',n_molecules=4))
    pd.DataFrame(observations).to_csv(run/'predictions.csv',index=False)
    pd.DataFrame(metrics).to_csv(run/'fold_metrics.csv',index=False)
    return generate_chemical_report([run],output_path=tmp_path/'report.html',embedding='pca')


@pytest.fixture(scope='module')
def report(tmp_path_factory):
    return make_report(tmp_path_factory.mktemp('browser-report'))


@pytest.fixture
def page(report):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as driver:
        browser = driver.chromium.launch()
        page = browser.new_page(viewport={'width':1440,'height':1000},accept_downloads=True)
        errors=[]
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto(report.as_uri())
        page.wait_for_function("document.getElementById('evaluation-count').textContent === '8'")
        page.wait_for_function("document.getElementById('space').data?.length > 0")
        yield page
        assert errors == [], errors
        browser.close()


def test_offline_view_roundtrip_export_and_empty_recovery(page, tmp_path):
    assert page.locator('#partition').input_value() == 'validation'
    assert page.evaluate("document.getElementById('space')._fullLayout.xaxis.title.text") == 'PC 1'
    assert page.evaluate("document.getElementById('space').data[1].selectedpoints") is None
    page.locator('#search').fill('validation-1')
    page.wait_for_function("document.getElementById('evaluation-count').textContent === '2'")
    with page.expect_download() as saved:
        page.locator('#save-view').click()
    recipe_path=tmp_path/'view.json'
    saved.value.save_as(recipe_path)
    with page.expect_download() as exported:
        page.locator('#export-cohort').click()
    export_path=tmp_path/'cohort.csv'
    exported.value.save_as(export_path)
    rows=list(csv.DictReader(io.StringIO(export_path.read_text())))
    assert len(rows)==2
    assert {row['partition'] for row in rows}=={'validation'}
    assert {row['sample_id'] for row in rows}=={'validation-1'}
    assert all(row['report_signature'] for row in rows)
    page.locator('#search').fill('missing molecule')
    page.wait_for_function("document.getElementById('evaluation-count').textContent === '0'")
    assert page.locator('#empty-reason').is_visible()
    assert page.locator('#export-cohort').is_disabled()
    page.locator('#recipe-file').set_input_files(recipe_path)
    page.wait_for_function("document.getElementById('view-status').textContent === 'Saved view restored.'")
    assert page.locator('#evaluation-count').inner_text()=='2'
    assert page.locator('#diagnostic-a .plot-container').count()==1
    page.locator('#reset-view').click()
    assert page.locator('#evaluation-count').inner_text()=='8'
    assert not page.locator('#empty-reason').is_visible()
    page.locator('#inspector-search').fill('validation-1')
    page.wait_for_timeout(200)
    page.locator('.inspector-result').first.click()
    assert 'validation' in page.locator('#details').inner_text()


def test_lasso_recipe_integrity_and_no_partial_restore(page,tmp_path):
    # Emit the same event Plotly emits, with two fold observations of one molecule.
    page.evaluate("""() => {const rows=evaluationRecords().filter(r=>r.endpoint===diagnosticEndpoint().id).slice(0,2);document.getElementById('space').emit('plotly_selected',{points:rows.map(r=>({customdata:r.id}))})}""")
    assert page.locator('#selected-count').inner_text()=='2'
    original=page.evaluate('viewRecipe()')
    page.locator('#clear-selection').click()
    assert page.locator('#selected-count').inner_text()=='8'
    page.evaluate('(recipe)=>restoreRecipe(recipe)',original)
    assert page.locator('#selected-count').inner_text()=='2'
    assert page.evaluate('viewRecipe()')==original
    invalid=json.loads(json.dumps(original));invalid['controls']['model']='does-not-exist'
    result=page.evaluate('(recipe)=>{try{restoreRecipe(recipe);return null}catch(error){return error.message}}',invalid)
    assert 'Unavailable model' in result
    assert page.evaluate('viewRecipe()')==original
    invalid['report_signature']='wrong'
    assert 'different report' in page.evaluate('(recipe)=>{try{restoreRecipe(recipe)}catch(error){return error.message}}',invalid)


def test_classification_ties_endpoint_scale_and_test_context(page):
    page.locator('#diagnostic-endpoint').select_option(label='Solubility demo')
    # Solubility errors are .1 .2 .3 .4 (median .25); classification errors are .5.
    assert page.locator('#selection-error').inner_text()=='0.2500'
    page.locator('#tab-classification').click()
    assert page.locator('#selection-error').inner_text()=='0.5000'
    curve=page.evaluate('rocCurve(activeDiagnosticRows(),document.getElementById("model").value)')
    assert curve=={'x':[0,1],'y':[0,1]}
    page.locator('#search').fill('validation-1')
    page.wait_for_function("document.getElementById('evaluation-count').textContent === '2'")
    assert 'Both observed classes' in page.locator('#diagnostic-c').inner_text()
    page.locator('#reset-view').click()
    page.locator('#partition').select_option('test')
    assert 'Locked test included' in page.locator('#evidence-note').inner_text()
    assert page.evaluate('selectedEvaluation().every(r=>r.split==="test")')
    # CSV text escaping protects formula-like identifiers, while negative numbers remain numeric.
    assert page.evaluate('csvCell("=1+1")')=='"\'=1+1"'
    assert page.evaluate('csvCell(-1.5)')=='"-1.5"'


@pytest.mark.parametrize('width',[1440,390])
def test_layout_keyboard_and_no_external_requests(page,width,tmp_path):
    requests=[]
    page.on('request',lambda request:requests.append(request.url))
    page.set_viewport_size({'width':width,'height':900})
    page.reload()
    page.wait_for_function("document.getElementById('evaluation-count').textContent === '8'")
    assert not any(url.startswith(('https://','http://')) for url in requests)
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1')
    page.keyboard.press('Tab')
    assert page.locator('.skip-link').evaluate('(element)=>element===document.activeElement')
    page.screenshot(path=str(tmp_path/f'report-{width}.png'),full_page=True)


@pytest.fixture(scope='module')
def atlas_report(tmp_path_factory):
    from admet_benchmark.chemical_report import generate_chemical_report
    from admet_benchmark.chemical_space import GTMConfig
    directory=tmp_path_factory.mktemp('atlas-browser')
    make_report(directory)
    return generate_chemical_report([directory/'bundle'], output_path=directory/'atlas.html',
        map_methods=['tsne','gtm','pca'], representations=['morgan','rdkit_descriptors','morgan_rdkit'],
        gtm_config=GTMConfig(nodes=5,rbfs=3,max_iterations=30))


def test_switchable_atlas_threshold_boundaries_and_node_recipe(page,atlas_report):
    page.goto(atlas_report.as_uri())
    page.wait_for_function("document.getElementById('map-technique').options.length===3")
    for representation in ['morgan','rdkit_descriptors','morgan_rdkit']:
        page.locator('#representation').select_option(representation)
        for method in ['tsne','pca','gtm']:
            page.locator('#map-technique').select_option(method)
            assert page.evaluate('currentMap().id')==f'{representation}:{method}'
            assert page.evaluate("document.getElementById('space')._fullLayout.xaxis.title.text")==f'{method.upper()} 1'
    assert page.evaluate('bandIndex(.25,{low:.25,high:.75})')==1
    assert page.evaluate('bandIndex(.75,{low:.25,high:.75})')==2
    before=page.evaluate('JSON.stringify(dashboard.endpoints.map(e=>e.metrics))')
    page.locator('#threshold-low').fill('0.15')
    page.locator('#threshold-high').fill('0.35')
    page.locator('#apply-thresholds').click()
    assert 'Low < 0.1500: 2' in page.locator('#threshold-summary').inner_text()
    assert 'High ≥ 0.3500: 2' in page.locator('#threshold-summary').inner_text()
    page.locator('#gtm-landscape').select_option('bands')
    page.locator('.gtm-node:not(:disabled)').first.click()
    recipe=page.evaluate('viewRecipe()')
    assert recipe['selected_gtm_node'] is not None and recipe['selected_record_ids']
    page.locator('#clear-selection').click()
    assert page.evaluate('selectedNode') is None
    page.evaluate('(recipe)=>restoreRecipe(recipe)',recipe)
    assert page.evaluate('viewRecipe()')==recipe
    assert page.locator('.gtm-node[aria-pressed="true"]').count()==1
    assert page.evaluate('JSON.stringify(dashboard.endpoints.map(e=>e.metrics))')==before
    page.locator('#threshold-high').fill('0.1')
    page.locator('#apply-thresholds').click()
    assert 'finite ordered cutoffs' in page.locator('#threshold-summary').inner_text()
    page.locator('#tab-classification').click()
    assert page.locator('#colour').input_value()=='prediction'
    assert page.locator('#threshold-panel').is_hidden()
    assert page.locator('#diagnostic-e').is_visible()
    assert sum(map(sum,page.evaluate("document.getElementById('diagnostic-e').data[0].z")))==8
    page.locator('#tab-merged').click()
    assert page.locator('#evaluation-count').inner_text()=='16'


def test_real_synthetic_five_and_ten_fold_uncertainty_views(page,tmp_path):
    import runpy
    build=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/build_fold_uq_demo.py'))['build_demo']
    report=build(tmp_path/'demo')
    roles=pd.read_csv(tmp_path/'demo/molecule_roles.csv')
    for _,group in roles.groupby(['split_strategy','fold']):
        assert group.sample_id.is_unique
        assert set(group.role)=={'fit','calibration','validation'}
    page.goto(report.as_uri())
    page.wait_for_function("document.getElementById('strategy').options.length===3")
    for strategy,count in [('synthetic_10fold',10),('synthetic_05fold',5)]:
        page.locator('#strategy').select_option(strategy)
        page.locator('#fold-display').select_option('facets')
        page.wait_for_function("Math.abs(document.getElementById('fold-errors')._fullLayout.width-document.getElementById('fold-errors').clientWidth)<3")
        assert page.evaluate("document.getElementById('fold-errors').data.length")==count
        assert page.evaluate("document.getElementById('uncertainty-b').data.length")==count
        assert page.evaluate("document.getElementById('uncertainty-c').data[1].y.every(v=>v===.9)")
        assert page.evaluate("document.getElementById('fold-errors').data.every(t=>t.x.length===80/"+str(count)+")")
        assert page.evaluate("document.getElementById('uncertainty-b').data.every(t=>t.x.at(-1)===1)")
    page.locator('#tab-classification').click()
    assert page.locator('#diagnostic-e').is_visible()
    assert page.locator('#threshold-panel').is_hidden()


def test_scientific_cohorts_sets_decisions_and_export(page,tmp_path):
    import yaml
    from admet_benchmark.chemical_report import generate_chemical_report
    make_report(tmp_path)
    run=tmp_path/'bundle';predictions=pd.read_csv(run/'predictions.csv')
    predictions['nearest_train_similarity']=.4
    predictions['novel_scaffold']=1
    predictions['property_cliff']=(predictions.task=='regression').astype(int)
    predictions['max_similar_property_gap']=1.5
    predictions['absolute_lower']=predictions.prediction-2
    predictions['absolute_upper']=predictions.prediction+2
    predictions['set_0']=1
    predictions['set_1']=(predictions.observed==1).astype(int)
    predictions.loc[predictions.task=='regression',['set_0','set_1']]=float('nan')
    predictions.to_csv(run/'predictions.csv',index=False)
    (run/'resolved_config.yaml').write_text(yaml.safe_dump({'cv_evidence':True,'uncertainty':{'interval_confidence':.9}}))
    report=generate_chemical_report([run],output_path=tmp_path/'scientific.html',embedding='pca')
    page.goto(report.as_uri());page.wait_for_function("document.getElementById('evaluation-count').textContent==='8'")
    assert page.locator('#scientific-panel').is_visible()
    page.locator('#property-cutoff').fill('10')
    page.locator('#apply-property-cutoff').click()
    assert 'Crosses cutoff: 2' in page.locator('#decision-note').inner_text()
    page.locator('#interval-method').select_option('absolute')
    assert page.evaluate("document.getElementById('uncertainty-d').data.every(t=>t.y.every(v=>Math.abs(v-4)<1e-10))")
    page.locator('#scientific-cohort').select_option('distant')
    recipe=page.evaluate('viewRecipe()')
    assert recipe['controls']['scientific-cohort']=='distant'
    page.locator('#reset-view').click();page.evaluate('(r)=>restoreRecipe(r)',recipe)
    assert page.evaluate('viewRecipe()')==recipe
    assert 'nearest_train_similarity' in page.evaluate('cohortCsv()')
    page.locator('#tab-classification').click()
    assert page.evaluate("document.getElementById('uncertainty-d').data[0].y")==[0,4,0,4]
    assert page.evaluate("document.getElementById('uncertainty-b').data.every(t=>t.y.at(-1)===.5)")
    assert page.locator('#decision-panel').is_hidden()
    assert page.locator('#interval-method-control').is_hidden()


def test_nondefault_threshold_matches_confusion_and_risk(page):
    page.locator('#tab-classification').click()
    result=page.evaluate('''() => {
        const rows=activeDiagnosticRows(),model=$('model').value,endpoint=diagnosticEndpoint();
        rows.forEach(r=>Object.assign(r.predictions[model], {prediction:r.target_model===1?.9:.6,
            threshold:.8,set_0:1,set_1:1,uncertainty:.1}));
        drawDiagnostics();drawPredictionSets(rows,endpoint,model);
        return {matrix:$('diagnostic-e').data[0].z,risk:$('uncertainty-b').data.map(t=>t.y.at(-1))};
    }''')
    assert result['matrix']==[[4,0],[0,4]]
    assert result['risk']==[0,0]


def test_assets_delivery_lazy_maps_and_recipe(page, report, tmp_path):
    from admet_benchmark.chemical_report import generate_chemical_report
    from admet_benchmark.chemical_space import GTMConfig
    output=generate_chemical_report([report.parent/'bundle'],output_path=tmp_path/'report.html',
        delivery='assets',map_methods=['tsne','pca','gtm'],representations=['morgan','rdkit_descriptors'],
        gtm_config=GTMConfig(nodes=5,rbfs=3,max_iterations=20))
    assert output.stat().st_size<200_000
    requests=[]
    page.on('request',lambda request:requests.append(request.url))
    page.goto(output.as_uri())
    page.wait_for_function("document.documentElement.dataset.reportReady==='true'")
    assert page.locator('#evaluation-count').inner_text()=='8'
    assert not any('/morgan-gtm.js' in url for url in requests)
    assert page.evaluate("document.querySelectorAll('.js-plotly-plot').length")<15
    page.locator('#map-technique').select_option('gtm')
    page.wait_for_function("currentMap()?.method==='gtm' && currentMap()?.coordinates?.length>0 && document.getElementById('loading-state').hidden")
    page.locator('#gtm-panel').scroll_into_view_if_needed()
    page.wait_for_function("document.getElementById('gtm-landscape-chart').data?.length>0")
    assert any('/morgan-gtm.js' in url for url in requests)
    page.locator('#tab-classification').click()
    page.locator('#uncertainty-panel').scroll_into_view_if_needed()
    page.wait_for_function("document.getElementById('uncertainty-a').data?.length>0")
    assert not any(url.startswith(('http://','https://')) for url in requests)
    assert not page.locator('#loading-state').is_visible()
