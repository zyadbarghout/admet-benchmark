"""Static HTML report for canonical evaluation bundles."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    cleaned = frame.replace({np.nan: None})
    return cleaned.to_dict(orient="records")


def _json(value: Any) -> str:
    # Avoid allowing user-provided labels to close the script element.
    return json.dumps(value, allow_nan=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )


def write_evaluation_report(
    output_dir: Path,
    config: dict[str, Any],
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    validation: dict[str, Any],
) -> Path:
    """Write a portable, distribution-first report from saved evaluation records."""

    try:
        from plotly.offline import get_plotlyjs

        plotly = get_plotlyjs()
    except ImportError:
        plotly = ""

    title = str(config.get("project_name", "ADMET evaluation"))
    prediction_columns = [
        "molecule_id",
        "endpoint",
        "model",
        "partition",
        "repeat",
        "fold",
        "preprocessing",
        "split_strategy",
        "observed",
        "prediction",
        "residual",
        "absolute_error",
        "uncertainty",
        "uncertainty_percentile",
        "prediction_lower",
        "prediction_upper",
        "threshold",
    ]
    data = {
        "task": str(config["task"]),
        "predictions": _records(
            predictions.loc[:, [column for column in prediction_columns if column in predictions]]
        ),
        "metrics": _records(metrics),
        "summary": _records(summary),
        "validation": validation,
    }
    path = output_dir / "report.html"
    path.write_text(
        _template()
        .replace("__TITLE__", escape(title))
        .replace("__PLOTLY__", plotly)
        .replace("__DATA__", _json(data)),
        encoding="utf-8",
    )
    return path


def _template() -> str:
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--ink:#16211d;--muted:#607068;--paper:#f2efe6;--panel:#fffdf7;--line:#d8d1c2;--green:#0b6550;--rust:#b64b32;--gold:#d6a437}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 90% 0,#dce8df 0,transparent 34%),var(--paper);color:var(--ink);font-family:"Avenir Next","Gill Sans",sans-serif}
header{padding:2.4rem clamp(1.2rem,5vw,4.5rem) 1.6rem;border-bottom:1px solid var(--line)}h1{font-family:"Iowan Old Style",Georgia,serif;font-size:clamp(2rem,4vw,3.7rem);letter-spacing:-.045em;margin:.15rem 0}.eyebrow{color:var(--green);font-weight:700;font-size:.72rem;letter-spacing:.15em;text-transform:uppercase}.subtitle{color:var(--muted);max-width:75ch}
main{display:grid;grid-template-columns:minmax(0,1.55fr) minmax(260px,.65fr);gap:1rem;padding:1rem clamp(1rem,4vw,3.5rem) 3rem}.panel{min-width:0;background:rgba(255,253,247,.94);border:1px solid var(--line);border-radius:16px;padding:1rem;box-shadow:0 13px 32px rgba(50,45,35,.07)}.wide{grid-column:1/-1}.controls{display:flex;flex-wrap:wrap;gap:.65rem;margin:.6rem 0 1rem}.controls label{display:grid;gap:.25rem;color:var(--muted);font-size:.7rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase}.controls select{min-width:165px;border:1px solid var(--line);border-radius:7px;background:#fff;padding:.45rem;color:var(--ink)}
.chart{height:430px}.small-chart{height:340px}.cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.55rem}.card{background:#edf3ed;border-radius:10px;padding:.7rem}.card:nth-child(even){background:#f3ead6}.card span{display:block}.label{color:var(--muted);font-size:.66rem;font-weight:700;text-transform:uppercase}.value{font-family:"Iowan Old Style",Georgia,serif;font-size:1.25rem;font-weight:700;margin-top:.15rem}.hint{color:var(--muted);font-size:.86rem;line-height:1.45}.table-wrap{overflow:auto;max-height:390px}table{border-collapse:collapse;width:100%;font-size:.78rem}th,td{border-bottom:1px solid var(--line);padding:.45rem;text-align:left;white-space:nowrap}th{position:sticky;top:0;background:#e4ece5}a{color:var(--green);font-weight:650;margin-right:1rem}
@media(max-width:900px){main{grid-template-columns:1fr}.chart{height:390px}}@media(max-width:560px){header{padding:1.3rem}.cards{grid-template-columns:1fr}.controls label,.controls select{width:100%}.panel{padding:.75rem}}
</style>
</head>
<body>
<header><div class="eyebrow">Prepared predictions · reproducible evaluation</div><h1>__TITLE__</h1><p class="subtitle" id="subtitle"></p></header>
<main>
<section class="panel wide"><div class="controls"><label>Endpoint<select id="endpoint"></select></label><label>Preprocessing<select id="preprocessing"></select></label><label>Split strategy<select id="strategy"></select></label><label>Partition<select id="partition"></select></label><label>Metric<select id="metric"></select></label></div><div id="distribution" class="chart"></div><p class="hint">Every repeat/fold value is displayed. Violin density and an embedded box summarize the observed distribution; no averaged bar replaces the fold-level results.</p></section>
<section class="panel"><h2>Selected metric</h2><div class="cards"><div class="card"><span class="label">Results</span><span class="value" id="result-count">-</span></div><div class="card"><span class="label">Models</span><span class="value" id="model-count">-</span></div><div class="card"><span class="label">Median range</span><span class="value" id="median-range">-</span></div><div class="card"><span class="label">Validation</span><span class="value" id="validation-status">-</span></div></div></section>
<section class="panel"><h2>Audit</h2><p class="hint" id="audit"></p><p><a href="validation_report.json">Validation</a><a href="coverage_audit.csv">Coverage</a><a href="evaluation_manifest.json">Manifest</a></p></section>
<section class="panel"><h2>Observed vs predicted</h2><div id="observed-predicted" class="small-chart"></div></section>
<section class="panel"><h2 id="error-title">Error distribution</h2><div id="errors" class="small-chart"></div></section>
<section class="panel wide"><h2>Metric distribution table</h2><div class="table-wrap"><table id="summary-table"></table></div></section>
<section class="panel wide"><h2>Evaluation files</h2><p class="hint">Paired comparisons are exploratory. Fold/repeat blocks may share molecules and training data; their nominal p-values do not establish confirmatory significance. Multiplicity correction does not fix this dependence.</p><p><a href="predictions.csv">Canonical predictions</a><a href="fold_metrics.csv">Fold metrics</a><a href="metric_distributions.csv">Metric distributions</a><a href="statistical_comparisons.csv">Paired comparisons</a><a href="resolved_config.yaml">Configuration</a></p></section>
</main>
<script>__PLOTLY__</script>
<script>
const report=__DATA__;const $=id=>document.getElementById(id);const unique=values=>[...new Set(values)].sort();const fmt=value=>Number.isFinite(Number(value))?Number(value).toFixed(4):'-';const median=values=>{const sorted=[...values].sort((a,b)=>a-b),middle=Math.floor(sorted.length/2);return sorted.length%2?sorted[middle]:(sorted[middle-1]+sorted[middle])/2};
function option(select,value){select.add(new Option(value,value))}function refill(select,values,preferred){const previous=select.value;select.replaceChildren();values.forEach(value=>option(select,value));select.value=values.includes(previous)?previous:(values.includes(preferred)?preferred:values[0]||'')}
function baseRows(rows){return rows.filter(row=>row.endpoint===$('endpoint').value&&row.preprocessing===$('preprocessing').value&&row.split_strategy===$('strategy').value&&row.partition===$('partition').value)}
function refreshDependent(){let rows=report.metrics.filter(row=>row.endpoint===$('endpoint').value);refill($('preprocessing'),unique(rows.map(row=>row.preprocessing)));rows=rows.filter(row=>row.preprocessing===$('preprocessing').value);refill($('strategy'),unique(rows.map(row=>row.split_strategy)));rows=rows.filter(row=>row.split_strategy===$('strategy').value);refill($('partition'),unique(rows.map(row=>row.partition)),'validation');rows=baseRows(report.metrics);const preferred=report.task==='regression'?'mae':'pr_auc';refill($('metric'),unique(rows.map(row=>row.metric)),preferred);draw()}
function draw(){if(typeof Plotly==='undefined'){document.querySelectorAll('.chart,.small-chart').forEach(node=>node.innerHTML='<p class="hint">Install the visual extra and regenerate this report to enable interactive plots.</p>');return}const rows=baseRows(report.metrics).filter(row=>row.metric===$('metric').value);const models=unique(rows.map(row=>row.model));const traces=models.map(model=>{const values=rows.filter(row=>row.model===model);return{type:'violin',name:model,y:values.map(row=>row.value),customdata:values.map(row=>[row.repeat,row.fold,row.n_molecules]),points:'all',jitter:.16,pointpos:0,box:{visible:true},meanline:{visible:false},spanmode:'hard',hovertemplate:`${model}<br>value %{y:.5f}<br>repeat %{customdata[0]} · fold %{customdata[1]}<br>molecules %{customdata[2]}<extra></extra>`}});Plotly.react('distribution',traces,{template:'none',paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',margin:{l:55,r:16,t:20,b:90},xaxis:{tickangle:-25},yaxis:{title:$('metric').value,zeroline:false},showlegend:false},{responsive:true});drawPredictions();renderSummary();$('result-count').textContent=rows.length;$('model-count').textContent=models.length;const medians=models.map(model=>median(rows.filter(row=>row.model===model).map(row=>Number(row.value)))).filter(Number.isFinite);$('median-range').textContent=medians.length?`${fmt(Math.min(...medians))}–${fmt(Math.max(...medians))}`:'-'}
function drawPredictions(){const rows=baseRows(report.predictions);const models=unique(rows.map(row=>row.model));const scatter=models.map((model,index)=>({type:'scattergl',mode:'markers',name:model,x:rows.filter(row=>row.model===model).map(row=>row.observed),y:rows.filter(row=>row.model===model).map(row=>row.prediction),text:rows.filter(row=>row.model===model).map(row=>row.molecule_id),marker:{size:6,opacity:.58},hovertemplate:'%{text}<br>Observed %{x:.4f}<br>Predicted %{y:.4f}<extra>'+model+'</extra>'}));Plotly.react('observed-predicted',scatter,{template:'none',paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',margin:{l:50,r:12,t:15,b:45},xaxis:{title:'Observed'},yaxis:{title:'Predicted'},legend:{orientation:'h'}},{responsive:true});let errorTraces;if(report.task==='regression'){$('error-title').textContent='Residual distribution';errorTraces=models.map(model=>({type:'histogram',histnorm:'probability density',opacity:.45,name:model,x:rows.filter(row=>row.model===model).map(row=>row.residual)}))}else{$('error-title').textContent='Predicted probability by class';errorTraces=[];models.forEach(model=>[0,1].forEach(label=>errorTraces.push({type:'violin',name:`${model} · class ${label}`,y:rows.filter(row=>row.model===model&&Number(row.observed)===label).map(row=>row.prediction),points:'all',jitter:.12,box:{visible:true}})))}Plotly.react('errors',errorTraces,{template:'none',barmode:'overlay',violinmode:'group',paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',margin:{l:50,r:12,t:15,b:70},legend:{orientation:'h'}},{responsive:true})}
function renderSummary(){const rows=baseRows(report.summary).filter(row=>row.metric===$('metric').value);const columns=['model','n_results','median','q1','q3','minimum','maximum'];$('summary-table').innerHTML=`<thead><tr>${columns.map(column=>`<th>${column}</th>`).join('')}</tr></thead><tbody>${rows.map(row=>`<tr>${columns.map(column=>`<td>${typeof row[column]==='number'?fmt(row[column]):row[column]??''}</td>`).join('')}</tr>`).join('')}</tbody>`}
function initialise(){refill($('endpoint'),unique(report.metrics.map(row=>row.endpoint)));['endpoint','preprocessing','strategy','partition'].forEach(id=>$(id).addEventListener('change',refreshDependent));$('metric').addEventListener('change',draw);$('validation-status').textContent=report.validation.status.replaceAll('_',' ');$('audit').textContent=report.validation.warnings.length?report.validation.warnings.join(' '):'No semantic validation warnings were recorded.';$('subtitle').textContent=`${report.predictions.length} canonical prediction records across ${unique(report.predictions.map(row=>row.endpoint)).length} endpoint(s). Evaluation scale and thresholds come from the saved configuration.`;refreshDependent()}
initialise();
</script>
</body></html>"""
