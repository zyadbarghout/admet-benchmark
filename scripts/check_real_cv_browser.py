"""Acceptance check of a generated measured CV report, including saved evidence."""
from pathlib import Path
import json
import time
from playwright.sync_api import sync_playwright


def check(root):
    root=Path(root).resolve();output=root/'browser_verification';output.mkdir(exist_ok=True)
    evidence={'checks':[]};errors=[];external=[]
    with sync_playwright() as pw:
        browser=pw.chromium.launch()
        page=browser.new_page(viewport={'width':1440,'height':1000},accept_downloads=True)
        page.set_default_timeout(60000)
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.on('request',lambda r:external.append(r.url) if r.url.startswith(('http://','https://')) else None)
        start=time.monotonic();page.goto((root/'chemical_model_diagnostics.html').as_uri(),timeout=120000)
        page.wait_for_function("$('evaluation-count').textContent==='5128' && $('space').data?.length>0",timeout=120000)
        evidence['initial_load_seconds']=round(time.monotonic()-start,2)
        info=page.evaluate("({records:dashboard.records.length,chemicals:dashboard.atlas.chemical_ids.length,maps:dashboard.atlas.views.length,model:$('model').value,context:sortedContexts(activeDiagnosticRows()).length,originalIDs:dashboard.records.every(r=>!!r.original_sample_id)})")
        assert info=={'records':26788,'chemicals':6697,'maps':9,'model':'random_forest','context':5,'originalIDs':True},info
        evidence['initial_state']=info
        page.screenshot(path=str(output/'desktop.png'))
        page.locator('#preprocessing').select_option('rdkit_descriptors');page.locator('#strategy').select_option('scaffold')
        page.locator('#scientific-cohort').select_option('cliff')
        page.wait_for_function("$('evaluation-count').textContent==='202'")
        coverage=page.evaluate("mean(activeDiagnosticRows().map(r=>{const p=r.predictions[$('model').value];return Number(r.target_model>=p.prediction_lower&&r.target_model<=p.prediction_upper)}))")
        assert abs(coverage-138/202)<1e-12
        evidence['cliff_cohort']={'n':202,'coverage':coverage}
        page.locator('#property-cutoff').fill('1');page.locator('#apply-property-cutoff').click()
        assert 'Crosses cutoff:' in page.locator('#decision-note').inner_text()
        normalized=page.evaluate("$('uncertainty-d').data[0].y")
        page.locator('#interval-method').select_option('absolute')
        assert page.evaluate("$('uncertainty-d').data[0].y")!=normalized
        page.evaluate('showMolecule(activeDiagnosticRows()[0].id)')
        assert 'interval (absolute)' in page.locator('#details').inner_text()
        assert 'nearest fitting Tanimoto' in page.locator('#details').inner_text()
        recipe=page.evaluate('viewRecipe()');assert recipe['controls']['scientific-cohort']=='cliff'
        with page.expect_download() as downloaded:page.locator('#export-cohort').click()
        downloaded.value.save_as(output/'cliff_cohort.csv')
        csv=(output/'cliff_cohort.csv').read_text();assert 'displayed_interval_method' in csv and 'nearest_train_similarity' in csv
        evidence['checks'].extend(['cohort filtering and exact coverage','property cutoff','interval switch and inspector','recipe and CSV export'])
        page.locator('#scientific-panel').screenshot(path=str(output/'chemical_stress.png'))
        page.locator('#scientific-cohort').select_option('all')
        sample=page.evaluate('activeDiagnosticRows()[0].original_sample_id')
        page.locator('#inspector-search').fill(sample)
        page.wait_for_function("$('inspector-count').textContent.startsWith('1 matches')")
        expected_name=page.evaluate('activeDiagnosticRows()[0].molecule_id')
        assert page.locator('#inspector-results button strong').first.inner_text()==expected_name
        page.locator('#inspector-results button').first.click()
        evidence['searched_original_id']=sample
        page.locator('#inspector-search').fill('')
        for representation in ['morgan','rdkit_descriptors','morgan_rdkit']:
            page.locator('#representation').select_option(representation)
            for technique in ['pca','gtm','tsne']:
                page.locator('#map-technique').select_option(technique)
                assert page.evaluate('currentMap().method')==technique
        page.locator('#fold-display').select_option('facets')
        evidence['checks'].append('all nine map options and fold facets')
        page.locator('#tab-classification').click()
        page.wait_for_function("$('evaluation-count').textContent==='1569'")
        assert page.locator('#decision-panel').is_hidden()
        assert page.locator('#interval-method-control').is_hidden()
        assert page.evaluate("$('uncertainty-b').layout.yaxis.title.text")=='Misclassification fraction'
        sets=page.evaluate("$('uncertainty-d').data[0].y");assert sum(sets)==1569
        assert page.evaluate('sortedContexts(activeDiagnosticRows()).length')==5
        page.evaluate('showMolecule(activeDiagnosticRows()[0].id)')
        assert 'prediction set:' in page.locator('#details').inner_text()
        evidence['classification_set_counts']=sets
        page.locator('#uncertainty-panel').screenshot(path=str(output/'classification_uq.png'))
        page.set_viewport_size({'width':390,'height':844});page.locator('header').scroll_into_view_if_needed()
        page.wait_for_function('document.documentElement.scrollWidth<=innerWidth+1',timeout=15000)
        page.screenshot(path=str(output/'mobile.png'))
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
        evidence['checks'].extend(['classification sets and five-fold risk curves','mobile viewport without horizontal overflow'])
        assert not errors,errors
        assert not external,external
        evidence.update(page_errors=errors,external_requests=external,html_bytes=(root/'chemical_model_diagnostics.html').stat().st_size)
        browser.close()
    (output/'result.json').write_text(json.dumps(evidence,indent=2)+'\n');print(json.dumps(evidence,indent=2))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('study');args=p.parse_args();check(args.study)
