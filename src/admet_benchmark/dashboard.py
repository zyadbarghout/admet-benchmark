"""Interactive, file-based reports for completed model-arena benchmarks.

The dashboard deliberately reads only saved benchmark artifacts. It never
trains models or recalculates metrics, which keeps the visual review separate
from the evaluation protocol and makes it safe to share with collaborators.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from plotly.offline import get_plotlyjs
from rdkit import Chem
from rdkit.Chem import Draw
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from .features import build_features


def generate_run_dashboard(
    run_dir: str | Path,
    *,
    smarts_patterns: list[str] | None = None,
    max_points: int = 5_000,
    max_structure_images: int = 1_500,
    perplexity: float = 30.0,
    output_path: str | Path | None = None,
) -> Path:
    """Create a self-contained interactive chemical-space review for one run.

    All validation and final-test molecules are retained before development
    molecules are sampled, so error review never silently drops evaluation
    points. SMARTS patterns are evaluated with RDKit during generation and then
    exposed as instant client-side filters in the HTML file.
    """

    run = Path(run_dir).expanduser().resolve()
    comparison = _read_csv(run / "model_comparison.csv")
    metadata = _read_json(run / "benchmark_metadata.json")
    split_manifest_path = run / "data" / "split_manifest.json"
    split_manifest = (
        _read_json(split_manifest_path)
        if split_manifest_path.exists()
        else _legacy_split_manifest(run, metadata)
    )
    config = _read_yaml(run / "resolved_config.yaml")
    smiles_column = str(
        dict(metadata.get("split") or {}).get(
            "smiles_column", config["dataset"]["smiles_column"]
        )
    )
    task = str(metadata["task"])
    target = str(metadata["target"])
    models = comparison["model"].astype(str).tolist()
    records = _load_records(run, smiles_column, models)
    records = _sample_records(records, max_points=max_points, seed=int(config.get("seed", 42)))

    features_config = config.get("features", {})
    features = build_features(
        [record["smiles"] for record in records],
        radius=int(features_config.get("radius", 2)),
        n_bits=int(features_config.get("n_bits", 1024)),
        include_rdkit_descriptors=bool(features_config.get("include_rdkit_descriptors", True)),
    ).values
    embedding = _tsne_embedding(features, perplexity=perplexity, seed=int(config.get("seed", 42)))
    for record, coordinate in zip(records, embedding, strict=True):
        record["x"] = float(coordinate[0])
        record["y"] = float(coordinate[1])

    patterns = _validate_smarts(smarts_patterns or [])
    _attach_smarts_matches(records, patterns)
    asset_dir = run / "interactive_report_assets" / "structures"
    _write_structure_images(records, asset_dir, max_images=max_structure_images)

    dashboard = {
        "title": f"{target}: interactive {task} benchmark review",
        "task": task,
        "target": target,
        "models": models,
        "comparison": _records_for_json(comparison),
        "split_manifest": split_manifest,
        "records": records,
        "smarts_patterns": patterns,
        "default_metric": "validation_mae" if task == "regression" else "validation_pr_auc",
        "structure_asset_dir": "interactive_report_assets/structures",
        "split_manifest_href": "data/split_manifest.json" if split_manifest_path.exists() else None,
        "sampled_points": len(records),
        "embedding": "t-SNE on scaled Morgan fingerprints and RDKit descriptors",
    }
    destination = (
        Path(output_path).expanduser().resolve()
        if output_path
        else run / "interactive_report.html"
    )
    destination.write_text(_dashboard_html(dashboard), encoding="utf-8")
    return destination


def generate_suite_dashboard(
    suite_dir: str | Path, *, output_path: str | Path | None = None
) -> Path:
    """Create an interactive comparison of completed suite runs.

    The suite report intentionally consumes ``cross_split_model_comparison.csv``
    rather than recomputing scores. It therefore remains a transparent view of
    the same numbers used in the machine-readable benchmark artifacts.
    """

    suite = Path(suite_dir).expanduser().resolve()
    comparison = _read_csv(suite / "cross_split_model_comparison.csv")
    required = {"run_name", "task", "target", "split_strategy", "model"}
    missing = required.difference(comparison.columns)
    if missing:
        raise ValueError(
            "cross_split_model_comparison.csv is missing required column(s): "
            + ", ".join(sorted(missing))
        )
    if comparison.empty:
        raise ValueError("cross_split_model_comparison.csv does not contain any results")

    summary_path = suite / "suite_summary.csv"
    dashboard = {
        "title": "Interactive cross-split model comparison",
        "comparison": _records_for_json(comparison),
        "summary": _records_for_json(_read_csv(summary_path)) if summary_path.exists() else [],
        "source_csv": "cross_split_model_comparison.csv",
        "summary_csv": "suite_summary.csv" if summary_path.exists() else None,
    }
    destination = (
        Path(output_path).expanduser().resolve()
        if output_path
        else suite / "interactive_suite_report.html"
    )
    destination.write_text(_suite_dashboard_html(dashboard), encoding="utf-8")
    return destination


def _load_records(run: Path, smiles_column: str, models: list[str]) -> list[dict[str, Any]]:
    development = _read_csv(run / "data" / "development.csv").assign(_split="development")
    validation = _read_csv(run / "validation_predictions.csv").assign(_split="validation")
    test = _read_csv(run / "held_out_test_predictions.csv").assign(_split="test")
    frames = [development, validation, test]
    records: list[dict[str, Any]] = []
    for frame in frames:
        for values in frame.to_dict(orient="records"):
            identifier = f"{values['_split']}:{values['source_row']}"
            prediction_data: dict[str, dict[str, float] | None] = {}
            for model in models:
                column = f"prediction__{model}"
                prediction = values.get(column)
                if prediction is None or pd.isna(prediction):
                    prediction_data[model] = None
                    continue
                predicted_value = float(prediction)
                observed = float(values["target_model"])
                prediction_data[model] = {
                    "prediction": predicted_value,
                    "residual": predicted_value - observed,
                    "absolute_error": abs(predicted_value - observed),
                }
            records.append(
                {
                    "id": identifier,
                    "split": str(values["_split"]),
                    "source_row": int(values["source_row"]),
                    "molecule_id": str(values["molecule_id"]),
                    "smiles": str(values[smiles_column]),
                    "target_raw": float(values["target_raw"]),
                    "target_model": float(values["target_model"]),
                    "predictions": prediction_data,
                    "structure": None,
                    "smarts": {},
                }
            )
    return records


def _sample_records(records: list[dict[str, Any]], *, max_points: int, seed: int) -> list[dict[str, Any]]:
    if max_points < 3:
        raise ValueError("max_points must be at least three")
    if len(records) <= max_points:
        return records
    evaluation = [record for record in records if record["split"] != "development"]
    development = [record for record in records if record["split"] == "development"]
    if len(evaluation) >= max_points:
        rng = np.random.default_rng(seed)
        indexes = np.sort(rng.choice(len(evaluation), size=max_points, replace=False))
        return [evaluation[index] for index in indexes]
    rng = np.random.default_rng(seed)
    remaining = max_points - len(evaluation)
    indexes = np.sort(rng.choice(len(development), size=remaining, replace=False))
    return evaluation + [development[index] for index in indexes]


def _tsne_embedding(features: np.ndarray, *, perplexity: float, seed: int) -> np.ndarray:
    if len(features) < 3:
        raise ValueError("At least three molecules are required for a t-SNE report")
    components = min(50, features.shape[1], len(features) - 1)
    reduced = PCA(n_components=components, random_state=seed).fit_transform(
        StandardScaler().fit_transform(features)
    )
    valid_perplexity = min(float(perplexity), max(2.0, (len(features) - 1) / 3))
    return TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=valid_perplexity,
        random_state=seed,
    ).fit_transform(reduced)


def _validate_smarts(patterns: list[str]) -> list[dict[str, str]]:
    validated: list[dict[str, str]] = []
    for pattern in patterns:
        molecule = Chem.MolFromSmarts(pattern)
        if molecule is None:
            raise ValueError(f"Invalid SMARTS pattern: {pattern!r}")
        key = hashlib.sha1(pattern.encode("utf-8")).hexdigest()[:12]
        validated.append({"key": key, "label": pattern})
    return validated


def _attach_smarts_matches(records: list[dict[str, Any]], patterns: list[dict[str, str]]) -> None:
    queries = [(pattern["key"], Chem.MolFromSmarts(pattern["label"])) for pattern in patterns]
    for record in records:
        molecule = Chem.MolFromSmiles(record["smiles"])
        record["smarts"] = {
            key: bool(molecule.HasSubstructMatch(query)) for key, query in queries
        }


def _write_structure_images(
    records: list[dict[str, Any]], asset_dir: Path, *, max_images: int) -> None:
    if max_images < 0:
        raise ValueError("max_structure_images cannot be negative")
    asset_dir.mkdir(parents=True, exist_ok=True)
    # Evaluation molecules matter most because they have predictions to inspect.
    ordered = sorted(records, key=lambda record: record["split"] == "development")[:max_images]
    for record in ordered:
        molecule = Chem.MolFromSmiles(record["smiles"])
        if molecule is None:
            continue
        name = hashlib.sha1(record["id"].encode("utf-8")).hexdigest() + ".png"
        path = asset_dir / name
        Draw.MolToFile(molecule, str(path), size=(260, 180))
        record["structure"] = name


def _records_for_json(table: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in table.to_dict(orient="records"):
        records.append(
            {
                key: (None if pd.isna(value) else value.item() if isinstance(value, np.generic) else value)
                for key, value in row.items()
            }
        )
    return records


def _dashboard_html(dashboard: dict[str, Any]) -> str:
    data = json.dumps(dashboard, allow_nan=False)
    title = escape(str(dashboard["title"]))
    manifest_link = (
        "<a href='data/split_manifest.json'>Split manifest</a>"
        if dashboard["split_manifest_href"]
        else ""
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ --ink:#17212b; --muted:#52616b; --paper:#f5f2ea; --panel:#fffdf8; --line:#d8d1c3; --accent:#004e64; --warm:#e07a5f; --train:#5f554d; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:radial-gradient(circle at 10% 0%,#dcebe4,transparent 34%),var(--paper); color:var(--ink); font-family:Georgia,'Times New Roman',serif; }}
header {{ padding:2.8rem max(5vw,2rem) 1.8rem; border-bottom:1px solid var(--line); }} h1 {{ margin:0; font-size:clamp(2rem,4vw,3.6rem); letter-spacing:-.04em; }} h2 {{ margin:0 0 .5rem; font-size:1.3rem; }} p {{ margin:.3rem 0; }} .eyebrow {{ color:var(--accent); font:700 .74rem/1.2 system-ui,sans-serif; letter-spacing:.14em; text-transform:uppercase; }}
.layout {{ display:grid; grid-template-columns:minmax(0,1.7fr) minmax(300px,.8fr); gap:1.25rem; padding:1.25rem max(4vw,1.5rem) 3rem; }} .panel {{ background:color-mix(in srgb,var(--panel) 92%,transparent); border:1px solid var(--line); border-radius:18px; padding:1.05rem; box-shadow:0 16px 40px rgba(50,45,35,.08); }}
.controls {{ display:flex; flex-wrap:wrap; gap:.75rem; align-items:end; margin-bottom:.8rem; }} label {{ display:grid; gap:.25rem; color:var(--muted); font:600 .73rem/1.2 system-ui,sans-serif; letter-spacing:.04em; text-transform:uppercase; }} select {{ min-width:160px; background:#fffdf8; border:1px solid var(--line); border-radius:8px; padding:.45rem; color:var(--ink); font:500 .9rem system-ui,sans-serif; }}
.wide {{ grid-column:1/-1; }} #space {{ height:650px; }} #comparison {{ height:300px; }} #distribution {{ height:250px; }} .cards {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.6rem; margin:.7rem 0; }} .card {{ background:#edf4ef; border-radius:10px; padding:.7rem; }} .card span {{ display:block; }} .card-label {{ color:var(--muted); font:600 .68rem/1.2 system-ui,sans-serif; text-transform:uppercase; letter-spacing:.05em; }} .card-value {{ margin-top:.15rem; font-size:1.25rem; font-weight:700; }}
.structure {{ width:100%; min-height:180px; border:1px dashed var(--line); border-radius:12px; padding:.7rem; background:#fff; }} .structure img {{ width:100%; height:auto; display:block; }} .details {{ margin-top:.6rem; font:400 .83rem/1.45 ui-monospace,SFMono-Regular,Menlo,monospace; overflow-wrap:anywhere; }} .hint {{ color:var(--muted); font-size:.9rem; }}
.table-wrap {{ overflow:auto; max-height:380px; }} table {{ border-collapse:collapse; width:100%; font:400 .8rem/1.35 system-ui,sans-serif; }} th {{ cursor:pointer; position:sticky; top:0; background:#e5eee8; }} th,td {{ border-bottom:1px solid var(--line); padding:.45rem; text-align:left; white-space:nowrap; }} th:hover {{ background:#d5e5da; }} .links a {{ margin-right:1rem; color:var(--accent); font:600 .85rem system-ui,sans-serif; }}
@media(max-width:980px) {{ .layout {{ grid-template-columns:1fr; }} #space {{ height:520px; }} }} @media(max-width:560px) {{ .cards {{ grid-template-columns:1fr; }} header {{ padding:1.6rem; }} }}
</style>
</head>
<body>
<header><div class="eyebrow">Reproducible molecular benchmark</div><h1>{title}</h1><p id="subtitle"></p></header>
<main class="layout">
  <section class="panel wide"><div class="controls"><label>Model<select id="model"></select></label><label>Colour<select id="colour"><option value="absolute_error">Absolute error</option><option value="residual">Signed residual</option></select></label><label>Substructure<select id="smarts"></select></label></div><div id="space"></div><p class="hint">Hover an evaluation molecule for its structure and prediction. Lasso or box-select points to update the selected-region statistics; selection does not remove hover support.</p></section>
  <section class="panel"><h2>Selection</h2><div class="cards"><div class="card"><span class="card-label">Molecules</span><span class="card-value" id="count">0</span></div><div class="card"><span class="card-label">Mean absolute error</span><span class="card-value" id="mae">-</span></div><div class="card"><span class="card-label">Mean residual</span><span class="card-value" id="residual">-</span></div></div><div id="distribution"></div></section>
  <section class="panel"><h2>Molecule</h2><div class="structure" id="structure"><span class="hint">Hover a validation or test molecule in chemical space.</span></div><div class="details" id="details"></div></section>
  <section class="panel wide"><h2>Model comparison</h2><div class="controls"><label>Metric<select id="metric"></select></label></div><div id="comparison"></div></section>
  <section class="panel wide"><h2>Benchmark table</h2><div class="table-wrap"><table id="benchmark-table"></table></div></section>
  <section class="panel wide"><h2>Files and protocol</h2><p id="manifest"></p><p class="links"><a href="model_comparison.csv">Comparison CSV</a><a href="validation_predictions.csv">Validation predictions</a><a href="held_out_test_predictions.csv">Held-out predictions</a>{manifest_link}<a href="resolved_config.yaml">Resolved configuration</a></p></section>
</main>
<script>{get_plotlyjs()}</script>
<script>const dashboard={data};
const recordsById=new Map(dashboard.records.map(record=>[record.id,record])); let selectedIds=new Set();
const $=id=>document.getElementById(id); const number=value=>Number.isFinite(value)?value.toFixed(4):'-';
function escapeHtml(value){{return String(value).replace(/[&<>'"]/g,char=>({{'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}}[char]));}}
function quantile(values,q){{const sorted=[...values].sort((a,b)=>a-b);if(!sorted.length)return 1;const index=(sorted.length-1)*q;const lower=Math.floor(index),upper=Math.ceil(index);return sorted[lower]+(sorted[upper]-sorted[lower])*(index-lower);}}
function activeModel(){{return $('model').value;}} function activeFilter(){{return $('smarts').value;}} function isVisible(record){{const filter=activeFilter();return filter==='all'||record.smarts[filter];}}
function evaluationRecords(){{return dashboard.records.filter(record=>record.split!=='development'&&isVisible(record)&&record.predictions[activeModel()]);}}
function renderedRecords(){{const visible=evaluationRecords();return selectedIds.size?visible.filter(record=>selectedIds.has(record.id)):visible;}}
function drawSpace(){{const model=activeModel(),colour=$('colour').value;const train=dashboard.records.filter(record=>record.split==='development'&&isVisible(record));const evaluation=evaluationRecords();const values=evaluation.map(record=>colour==='residual'?record.predictions[model].residual:record.predictions[model].absolute_error);const bound=Math.max(quantile(values.map(value=>Math.abs(value)),.95),1e-9);const colors=colour==='residual'?values.map(value=>Math.max(-bound,Math.min(bound,value))):values.map(value=>Math.min(bound,value));const selected=evaluation.map((record,index)=>selectedIds.has(record.id)?index:null).filter(index=>index!==null);const traces=[{{uid:'development',name:'Development',x:train.map(record=>record.x),y:train.map(record=>record.y),customdata:train.map(record=>record.id),mode:'markers',type:'scattergl',marker:{{color:'#5f554d',size:6,opacity:.35}},hovertemplate:'Development molecule<extra></extra>'}},{{uid:'evaluation',name:'Validation / test',x:evaluation.map(record=>record.x),y:evaluation.map(record=>record.y),customdata:evaluation.map(record=>record.id),mode:'markers',type:'scattergl',selectedpoints:selected,marker:{{color:colors,colorscale:'RdBu',cmin:colour==='residual'?-bound:0,cmax:bound,colorbar:{{title:colour==='residual'?'Residual':'Absolute error'}},size:8,opacity:.9,line:{{color:'#fff',width:.4}}}},selected:{{marker:{{opacity:1,size:10,line:{{color:'#17212b',width:1.4}}}}}},unselected:{{marker:{{opacity:.45}}}},hovertemplate:'%{{customdata}}<extra></extra>'}}];Plotly.react('space',traces,{{template:'none',uirevision:'chemical-space',paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',margin:{{l:45,r:25,t:25,b:45}},xaxis:{{title:'t-SNE 1',showgrid:false,zeroline:false}},yaxis:{{title:'t-SNE 2',showgrid:false,zeroline:false}},legend:{{orientation:'h',uirevision:'chemical-space'}}}},{{responsive:true}});updateSelection();}}
function updateSelection(){{const model=activeModel(),records=renderedRecords(),errors=records.map(record=>record.predictions[model].absolute_error),residuals=records.map(record=>record.predictions[model].residual);$('count').textContent=records.length;$('mae').textContent=records.length?number(errors.reduce((a,b)=>a+b,0)/errors.length):'-';$('residual').textContent=records.length?number(residuals.reduce((a,b)=>a+b,0)/residuals.length):'-';Plotly.react('distribution',[{{x:residuals,type:'histogram',marker:{{color:'#004e64'}},nbinsx:30,name:'Residual'}}],{{template:'none',paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',margin:{{l:45,r:15,t:15,b:42}},xaxis:{{title:'Prediction - observed'}},yaxis:{{title:'Count'}}}},{{responsive:true}});}}
function showMolecule(id){{const record=recordsById.get(id);if(!record)return;const result=record.predictions[activeModel()];const image=record.structure?`<img src="${{dashboard.structure_asset_dir}}/${{record.structure}}" alt="Chemical structure">`:'<span class="hint">Structure image was not materialised for this sampled development molecule.</span>';$('structure').innerHTML=image;$('details').innerHTML=`<strong>${{escapeHtml(record.molecule_id)}}</strong><br>${{escapeHtml(record.smiles)}}<br>split: ${{record.split}}<br>observed (evaluation scale): ${{number(record.target_model)}}${{result?`<br>prediction: ${{number(result.prediction)}}<br>residual: ${{number(result.residual)}}<br>absolute error: ${{number(result.absolute_error)}}`:''}}`;}}
function drawComparison(){{const metric=$('metric').value;const rows=[...dashboard.comparison].sort((a,b)=>(Number(a[metric])||-Infinity)-(Number(b[metric])||-Infinity));Plotly.react('comparison',[{{x:rows.map(row=>row[metric]),y:rows.map(row=>row.model),type:'bar',orientation:'h',marker:{{color:'#e07a5f'}},hovertemplate:'%{{y}}: %{{x:.4f}}<extra></extra>'}}],{{template:'none',paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',margin:{{l:130,r:20,t:15,b:42}},xaxis:{{title:metric}},yaxis:{{automargin:true}}}},{{responsive:true}});}}
function renderTable(){{const table=$('benchmark-table'),rows=dashboard.comparison,columns=Object.keys(rows[0]).filter(column=>column!=='parameters');table.innerHTML=`<thead><tr>${{columns.map(column=>`<th>${{escapeHtml(column)}}</th>`).join('')}}</tr></thead><tbody>${{rows.map(row=>`<tr>${{columns.map(column=>`<td>${{typeof row[column]==='number'?number(row[column]):escapeHtml(row[column]??'')}}</td>`).join('')}}</tr>`).join('')}}</tbody>`;[...table.tHead.rows[0].cells].forEach((header,index)=>header.addEventListener('click',()=>{{const body=[...table.tBodies[0].rows];body.sort((left,right)=>left.cells[index].textContent.localeCompare(right.cells[index].textContent,undefined,{{numeric:true}}));body.forEach(row=>table.tBodies[0].append(row));}}));}}
function initialise(){{dashboard.models.forEach(model=>$('model').add(new Option(model,model)));dashboard.smarts_patterns.forEach(pattern=>$('smarts').add(new Option(pattern.label,pattern.key)));$('smarts').add(new Option('All molecules','all'),undefined);$('smarts').value='all';const metrics=Object.keys(dashboard.comparison[0]).filter(column=>column.startsWith('validation_')||column.startsWith('test_'));metrics.forEach(metric=>$('metric').add(new Option(metric,metric)));$('metric').value=dashboard.default_metric;$('subtitle').textContent=`${{dashboard.embedding}} | ${{dashboard.sampled_points}} displayed molecules | split: ${{dashboard.split_manifest.strategy}}`;$('manifest').textContent=`Split strategy: ${{dashboard.split_manifest.strategy}}. Train: ${{dashboard.split_manifest.train_rows}}; held-out test: ${{dashboard.split_manifest.test_rows}}; actual holdout: ${{(100*dashboard.split_manifest.actual_test_size).toFixed(1)}}%. The report reads saved predictions only.`;renderTable();drawSpace();drawComparison();$('model').addEventListener('change',()=>{{selectedIds.clear();drawSpace();}});$('colour').addEventListener('change',drawSpace);$('smarts').addEventListener('change',()=>{{selectedIds.clear();drawSpace();}});$('metric').addEventListener('change',drawComparison);const plot=$('space');plot.on('plotly_hover',event=>showMolecule(event.points[0].customdata));plot.on('plotly_selected',event=>{{selectedIds=new Set(event.points.map(point=>point.customdata));updateSelection();}});plot.on('plotly_deselect',()=>{{selectedIds.clear();updateSelection();}});}}
initialise();</script></body></html>"""


def _suite_dashboard_html(dashboard: dict[str, Any]) -> str:
    """Render a standalone cross-split report without a server dependency."""

    template = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root { --ink:#17212b; --muted:#52616b; --paper:#f5f2ea; --panel:#fffdf8; --line:#d8d1c3; --accent:#004e64; --warm:#e07a5f; --good:#276749; --rank:#f0e6cc; }
* { box-sizing:border-box; } body { margin:0; background:radial-gradient(circle at 88% 4%,#dcebe4,transparent 35%),var(--paper); color:var(--ink); font-family:Georgia,'Times New Roman',serif; }
header { padding:2.8rem max(5vw,2rem) 1.8rem; border-bottom:1px solid var(--line); } h1 { margin:0; font-size:clamp(2rem,4vw,3.6rem); letter-spacing:-.04em; } h2 { margin:0 0 .5rem; font-size:1.3rem; } p { margin:.35rem 0; } .eyebrow { color:var(--accent); font:700 .74rem/1.2 system-ui,sans-serif; letter-spacing:.14em; text-transform:uppercase; }
.layout { display:grid; grid-template-columns:minmax(0,1.55fr) minmax(285px,.75fr); gap:1.25rem; padding:1.25rem max(4vw,1.5rem) 3rem; } .panel { min-width:0; background:color-mix(in srgb,var(--panel) 92%,transparent); border:1px solid var(--line); border-radius:18px; padding:1.05rem; box-shadow:0 16px 40px rgba(50,45,35,.08); } .wide { grid-column:1/-1; }
.controls { display:flex; flex-wrap:wrap; gap:.75rem; align-items:end; margin-bottom:.8rem; } label { display:grid; gap:.25rem; color:var(--muted); font:600 .73rem/1.2 system-ui,sans-serif; letter-spacing:.04em; text-transform:uppercase; } select { min-width:210px; background:#fffdf8; border:1px solid var(--line); border-radius:8px; padding:.45rem; color:var(--ink); font:500 .9rem system-ui,sans-serif; }
#comparison { height:480px; } .cards { display:grid; gap:.6rem; grid-template-columns:repeat(2,minmax(0,1fr)); margin:.7rem 0; } .card { background:#edf4ef; border-radius:10px; padding:.75rem; } .card:nth-child(2) { background:var(--rank); } .card span { display:block; } .card-label { color:var(--muted); font:600 .68rem/1.2 system-ui,sans-serif; text-transform:uppercase; letter-spacing:.05em; } .card-value { margin-top:.15rem; font-size:1.2rem; font-weight:700; }
.table-wrap { overflow:auto; max-height:430px; } table { border-collapse:collapse; width:100%; font:400 .82rem/1.35 system-ui,sans-serif; } th { cursor:pointer; position:sticky; top:0; background:#e5eee8; } th,td { border-bottom:1px solid var(--line); padding:.48rem; text-align:left; white-space:nowrap; } th:hover { background:#d5e5da; } .best { color:var(--good); font-weight:700; } .hint { color:var(--muted); font-size:.9rem; } a { color:var(--accent); font:600 .85rem system-ui,sans-serif; margin-right:1rem; }
@media(max-width:900px) { .layout { grid-template-columns:1fr; } #comparison { height:410px; } } @media(max-width:560px) { header { padding:1.6rem; } .cards { grid-template-columns:1fr; } select { min-width:100%; } }
</style>
</head>
<body>
<header><div class="eyebrow">Reproducible molecular benchmark</div><h1>__TITLE__</h1><p id="subtitle"></p></header>
<main class="layout">
  <section class="panel wide"><div class="controls"><label>Endpoint and task<select id="target"></select></label><label>Metric<select id="metric"></select></label></div><p class="hint" id="direction"></p><div id="comparison"></div></section>
  <section class="panel"><h2>Best mean rank</h2><div class="cards"><div class="card"><span class="card-label">Model</span><span class="card-value" id="best-model">-</span></div><div class="card"><span class="card-label">Mean rank across splits</span><span class="card-value" id="best-rank">-</span></div></div><p class="hint">Ranks are calculated separately within each split using the selected metric, then averaged. Ties receive the same rank.</p></section>
  <section class="panel"><h2>Protocol</h2><p class="hint" id="protocol"></p><p class="hint">Each bar is a completed, independently tuned run. The interactive view does not retrain models or recalculate metrics.</p></section>
  <section class="panel wide"><h2>Split-aware model table</h2><div class="table-wrap"><table id="result-table"></table></div></section>
  <section class="panel wide"><h2>Files</h2><p><a href="__SOURCE_CSV__">Cross-split comparison CSV</a>__SUMMARY_LINK__</p></section>
</main>
<script>__PLOTLY__</script>
<script>
const suiteData=__DATA__;
const $=id=>document.getElementById(id);
const number=value=>Number.isFinite(Number(value))?Number(value).toFixed(4):'-';
function escapeHtml(value){return String(value).replace(/[&<>'"]/g,char=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char]));}
function keyFor(row){return `${row.task} :: ${row.target}`;}
function lowerIsBetter(metric){return /(mae|rmse|mse|error|loss|brier)/i.test(metric);}
function targets(){return [...new Set(suiteData.comparison.map(keyFor))].sort();}
function selectedRows(){const target=$('target').value,metric=$('metric').value;return suiteData.comparison.filter(row=>keyFor(row)===target&&Number.isFinite(Number(row[metric])));}
function supportedMetrics(){const target=$('target').value;const rows=suiteData.comparison.filter(row=>keyFor(row)===target);return Object.keys(rows[0]||{}).filter(column=>(column.startsWith('validation_')||column.startsWith('test_'))&&rows.some(row=>Number.isFinite(Number(row[column]))));}
function rankRows(rows,metric){const direction=lowerIsBetter(metric)?1:-1;const ranks=new Map();for(const split of [...new Set(rows.map(row=>row.split_strategy))]){const ordered=rows.filter(row=>row.split_strategy===split).sort((left,right)=>direction*(Number(left[metric])-Number(right[metric])));let rank=0,last=null;ordered.forEach((row,index)=>{const value=Number(row[metric]);if(index===0||value!==last)rank=index+1;last=value;ranks.set(`${split}:${row.model}`,rank);});}return ranks;}
function drawComparison(){const metric=$('metric').value,rows=selectedRows(),lower=lowerIsBetter(metric),splits=[...new Set(rows.map(row=>row.split_strategy))].sort(),ranks=rankRows(rows,metric);const models=[...new Set(rows.map(row=>row.model))];const meanRank=model=>{const values=splits.map(split=>ranks.get(`${split}:${model}`)).filter(Number.isFinite);return values.reduce((sum,value)=>sum+value,0)/values.length;};models.sort((left,right)=>meanRank(left)-meanRank(right));const traces=splits.map((split,index)=>({name:split,x:models,y:models.map(model=>{const row=rows.find(candidate=>candidate.split_strategy===split&&candidate.model===model);return row?Number(row[metric]):null;}),type:'bar',marker:{color:['#004e64','#e07a5f','#7b5e7b','#6f8f72'][index%4]},hovertemplate:`${escapeHtml(split)}<br>%{x}: %{y:.4f}<extra></extra>`}));Plotly.react('comparison',traces,{template:'none',barmode:'group',paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',margin:{l:58,r:20,t:24,b:115},xaxis:{tickangle:-35,automargin:true},yaxis:{title:metric,zeroline:false},legend:{orientation:'h'}},{responsive:true});const ranked=models.filter(model=>Number.isFinite(meanRank(model))).sort((left,right)=>meanRank(left)-meanRank(right));$('best-model').textContent=ranked[0]||'-';$('best-rank').textContent=ranked.length?number(meanRank(ranked[0])):'-';$('direction').textContent=`${lower?'Lower':'Higher'} ${metric.replace(/_/g,' ')} is better. Bars show each split independently; rank is the across-split summary.`;$('protocol').textContent=`${rows.length} model-by-split results are available for ${$('target').value}. ${splits.length} ${splits.length===1?'split strategy':'split strategies'}: ${splits.join(', ')||'-'}.`;renderTable(rows,metric,ranks);}
function renderTable(rows,metric,ranks){const table=$('result-table');const ordered=[...rows].sort((left,right)=>{const rankLeft=ranks.get(`${left.split_strategy}:${left.model}`),rankRight=ranks.get(`${right.split_strategy}:${right.model}`);return left.split_strategy.localeCompare(right.split_strategy)||(rankLeft-rankRight);});const columns=['split_strategy','model',metric,'validation_selection_score','test_mae','test_pr_auc','test_roc_auc','test_mcc'];const present=columns.filter(column=>ordered.some(row=>row[column]!==null&&row[column]!==undefined&&row[column]!==''));table.innerHTML=`<thead><tr>${present.map(column=>`<th>${escapeHtml(column)}</th>`).join('')}</tr></thead><tbody>${ordered.map(row=>{const best=ranks.get(`${row.split_strategy}:${row.model}`)===1;return `<tr>${present.map(column=>`<td class="${column===metric&&best?'best':''}">${typeof row[column]==='number'?number(row[column]):escapeHtml(row[column]??'')}</td>`).join('')}</tr>`;}).join('')}</tbody>`;[...table.tHead.rows[0].cells].forEach((header,index)=>header.addEventListener('click',()=>{const body=[...table.tBodies[0].rows];body.sort((left,right)=>left.cells[index].textContent.localeCompare(right.cells[index].textContent,undefined,{numeric:true}));body.forEach(row=>table.tBodies[0].append(row));}));}
function initialise(){targets().forEach(target=>$('target').add(new Option(target,target)));const refreshMetrics=()=>{const previous=$('metric').value;$('metric').replaceChildren();supportedMetrics().forEach(metric=>$('metric').add(new Option(metric,metric)));const preferred=['validation_mae','validation_pr_auc','validation_roc_auc'].find(metric=>[...$('metric').options].some(option=>option.value===metric));$('metric').value=[...$('metric').options].some(option=>option.value===previous)?previous:preferred||$('metric').value;drawComparison();};$('target').addEventListener('change',refreshMetrics);$('metric').addEventListener('change',drawComparison);$('subtitle').textContent=`${suiteData.comparison.length} recorded model-by-split results. Choose a metric to compare the completed experiments.`;refreshMetrics();}
initialise();
</script></body></html>"""
    summary_link = (
        '<a href="suite_summary.csv">Suite summary CSV</a>'
        if dashboard["summary_csv"]
        else ""
    )
    return (
        template.replace("__TITLE__", escape(str(dashboard["title"])))
        .replace("__SOURCE_CSV__", str(dashboard["source_csv"]))
        .replace("__SUMMARY_LINK__", summary_link)
        .replace("__PLOTLY__", get_plotlyjs())
        .replace("__DATA__", json.dumps(dashboard, allow_nan=False))
    )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required benchmark artifact is missing: {path}")
    return pd.read_csv(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required benchmark artifact is missing: {path}")
    return json.loads(path.read_text())


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required benchmark artifact is missing: {path}")
    return yaml.safe_load(path.read_text()) or {}


def _legacy_split_manifest(run: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    """Give reports generated before split manifests an explicit legacy label."""

    development = _read_csv(run / "data" / "development.csv")
    validation = _read_csv(run / "data" / "validation.csv")
    test = _read_csv(run / "data" / "held_out_test.csv")
    train_rows = len(development) + len(validation)
    total = train_rows + len(test)
    return {
        "strategy": "legacy_unrecorded",
        "train_rows": train_rows,
        "test_rows": len(test),
        "actual_test_size": len(test) / total if total else float("nan"),
        **dict(metadata.get("split") or {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an interactive report for a model-arena run or suite."
    )
    parser.add_argument(
        "artifact_dir",
        type=Path,
        help=(
            "Benchmark directory containing model_comparison.csv, or suite directory "
            "containing cross_split_model_comparison.csv"
        ),
    )
    parser.add_argument("--smarts", action="append", default=[], help="SMARTS pattern available as a dashboard filter; repeat as needed")
    parser.add_argument("--max-points", type=int, default=5_000, help="Maximum displayed molecules; evaluation molecules are kept first")
    parser.add_argument("--max-structure-images", type=int, default=1_500, help="Maximum structure PNG assets to materialise")
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity")
    parser.add_argument("--output", type=Path, help="Optional HTML destination")
    arguments = parser.parse_args()
    artifact_dir = arguments.artifact_dir.expanduser().resolve()
    if (artifact_dir / "cross_split_model_comparison.csv").exists():
        if arguments.smarts:
            parser.error("--smarts is available for individual run dashboards, not suite dashboards")
        output = generate_suite_dashboard(artifact_dir, output_path=arguments.output)
    else:
        output = generate_run_dashboard(
            artifact_dir,
            smarts_patterns=arguments.smarts,
            max_points=arguments.max_points,
            max_structure_images=arguments.max_structure_images,
            perplexity=arguments.perplexity,
            output_path=arguments.output,
        )
    print(output)


if __name__ == "__main__":
    main()
