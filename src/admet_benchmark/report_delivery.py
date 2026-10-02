"""Offline assets delivery: compact records and on-demand maps/depictions."""
import base64
import copy
import hashlib
import json
from html import escape
from pathlib import Path

from plotly.offline import get_plotlyjs


def _json(value):
    return json.dumps(value, allow_nan=False, separators=(",", ":")).replace("</", "<\\/")


def write_assets_report(dashboard, destination):
    from .chemical_report import _report_html
    assets = destination.with_name(destination.stem + ".assets")
    assets.mkdir(parents=True, exist_ok=True)
    data = copy.deepcopy(dashboard)
    data["delivery"] = "assets"
    data["lazy_panels"] = True
    structure_dir = assets / "structures"
    structure_dir.mkdir(exist_ok=True)
    for key, value in data["structures"].items():
        header, encoded = value.split(",", 1)
        extension = "webp" if "webp" in header else "png"
        filename = f"{key}.{extension}"
        (structure_dir / filename).write_bytes(base64.b64decode(encoded))
        data["structures"][key] = f"{assets.name}/structures/{filename}"
    for view in (data.get("atlas") or {}).get("views", []):
        filename = view["id"].replace(":", "-") + ".js"
        (assets / filename).write_text(f"window.reportChunks[{_json(view['id'])}]={_json(view)};", encoding="utf-8")
        metadata = {key: view[key] for key in ("id", "representation", "method", "axes", "note", "preprocessing")}
        metadata["asset_url"] = f"{assets.name}/{filename}"
        view.clear()
        view.update(metadata)
    # Columnar transport removes repeated keys without dropping precision/rows.
    records = data.pop("records")
    columns = sorted(set().union(*(r.keys() for r in records)) - {"predictions"})
    fields = sorted(set().union(*(p.keys() for r in records for p in r["predictions"].values())))
    packed = {"columns": columns, "prediction_fields": fields, "rows": [
        [[r.get(key) for key in columns], {m: [p.get(key) for key in fields] for m, p in r["predictions"].items()}]
        for r in records]}
    data["records"] = []
    data["record_asset"] = f"{assets.name}/records.js"
    (assets / "records.js").write_text("window.reportChunks.records=" + _json(packed) + ";", encoding="utf-8")
    (assets / "plotly.js").write_text(get_plotlyjs(), encoding="utf-8")
    source = Path(__file__).parent / "report_assets"
    for name in ("report.js", "workspace.js", "scientific.js", "delivery.js", "report.css"):
        (assets / name).write_bytes((source / name).read_bytes())
    html = _report_html(data)
    html = html.replace(f"<style>\n{(source/'report.css').read_text()}\n</style>", f'<link rel="stylesheet" href="{escape(assets.name)}/report.css">')
    for name, content in (("plotly.js", get_plotlyjs()), *[(n, (source/n).read_text()) for n in ("report.js", "workspace.js", "scientific.js")]):
        html = html.replace(f"<script>{content}</script>", f'<script defer src="{escape(assets.name)}/{name}"></script>')
    html = html.replace("<script>initialise(); initialiseWorkspace(); initialiseScientific(); initialiseViewTools();</script>",
        f'<script defer src="{escape(assets.name)}/delivery.js"></script>')
    destination.write_text(html, encoding="utf-8")
    manifest = {str(p.relative_to(destination.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in assets.rglob("*") if p.is_file() and p != assets / "manifest.json"}
    manifest[destination.name] = hashlib.sha256(destination.read_bytes()).hexdigest()
    (assets / "manifest.json").write_text(json.dumps({"report_signature": dashboard["report_signature"], "artifacts": manifest}, indent=2))
