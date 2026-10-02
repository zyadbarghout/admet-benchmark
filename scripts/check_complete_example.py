"""Acceptance checks against the complete, unsampled real-data assets report."""
from pathlib import Path
import argparse
import json
import time
from urllib.parse import unquote, urlparse

from playwright.sync_api import sync_playwright


def check(root):
    root = Path(root).resolve()
    evidence = root / "verification"
    evidence.mkdir(exist_ok=True)
    report = root / "chemical_model_diagnostics.html"
    with sync_playwright() as driver:
        browser = driver.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, accept_downloads=True)
        errors, requests = [], []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("request", lambda request: requests.append(request.url))
        page.goto((root / "index.html").as_uri())
        links = page.locator("a[href]").evaluate_all("nodes=>nodes.map(node=>node.getAttribute('href'))")
        for link in links:
            target = urlparse(link)
            if not target.scheme and target.path:
                assert (root / unquote(target.path)).is_file(), f"Broken landing-page link: {link}"
        page.screenshot(path=str(evidence / "landing-page.png"))
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth<=innerWidth+1")
        page.screenshot(path=str(evidence / "landing-mobile.png"))
        page.set_viewport_size({"width": 1440, "height": 1000})
        started = time.monotonic()
        page.goto(report.as_uri(), timeout=180_000)
        page.wait_for_function("document.documentElement.dataset.reportReady==='true'", timeout=180_000)
        load_seconds = time.monotonic() - started
        initial = page.evaluate("({records:dashboard.records.length,sampled:dashboard.sampling_applied,charts:document.querySelectorAll('.js-plotly-plot').length,height:document.documentElement.scrollHeight,visible:document.getElementById('evaluation-count').textContent})")
        assert not initial["sampled"]
        assert initial["records"] == 94617, initial
        assert initial["charts"] < 18
        page.screenshot(path=str(evidence / "desktop-overview.png"))
        page.locator("#fold").select_option("0")
        page.locator("#repeat").select_option("0")
        for representation in ["morgan", "rdkit_descriptors", "morgan_rdkit"]:
            page.locator("#representation").select_option(representation)
            page.wait_for_function("document.getElementById('loading-state').hidden")
            for method in ["tsne", "gtm", "pca"]:
                page.locator("#map-technique").select_option(method)
                page.wait_for_function("expected=>currentMap()?.id===expected && !!currentMap().coordinates && document.getElementById('loading-state').hidden", arg=f"{representation}:{method}")
                assert page.evaluate("document.getElementById('space').data.some(t=>t.x?.length)")
        page.locator("#chemical-space").scroll_into_view_if_needed()
        page.screenshot(path=str(evidence / "chemical-map.png"))
        page.locator("#scientific-cohort").select_option("cliff")
        page.locator("#scientific-panel").scroll_into_view_if_needed()
        page.wait_for_function("document.getElementById('applicability-coverage').data?.length>0")
        assert page.evaluate("activeDiagnosticRows().every(r=>r.predictions[document.getElementById('model').value].property_cliff===1)")
        page.locator("#scientific-cohort").select_option("all")
        page.locator("#interval-method").select_option("absolute")
        page.locator("#property-cutoff").fill("1")
        page.locator("#apply-property-cutoff").click()
        assert "Cutoff 1" in page.locator("#decision-note").inner_text()
        with page.expect_download() as download:
            page.locator("#save-view").click()
        recipe = evidence / "saved-view.json"
        download.value.save_as(recipe)
        page.locator("#recipe-file").set_input_files(recipe)
        page.wait_for_function("document.getElementById('view-status').textContent==='Saved view restored.'")
        with page.expect_download() as download:
            page.locator("#export-cohort").click()
        download.value.save_as(evidence / "cohort.csv")
        page.locator("#tab-classification").click()
        page.locator("#preprocessing").select_option("graph")
        page.locator("#model").select_option("chemprop")
        assert page.locator("#evaluation-count").inner_text() == "1569"
        page.locator("#uncertainty-panel").scroll_into_view_if_needed()
        page.wait_for_function("document.getElementById('uncertainty-c').data?.[0]?.name==='Set coverage'")
        page.screenshot(path=str(evidence / "neural-uncertainty.png"))
        page.set_viewport_size({"width": 390, "height": 844})
        page.evaluate("scrollTo(0,0)")
        page.screenshot(path=str(evidence / "mobile-overview.png"))
        assert page.evaluate("document.documentElement.scrollWidth<=innerWidth+1")
        assert not errors, errors
        assert not any(url.startswith(("http://", "https://")) for url in requests)
        browser.close()
    outcome = {"initial_load_seconds": round(load_seconds, 2), "html_bytes": report.stat().st_size,
        "initial": initial, "page_errors": errors, "external_requests": 0, "map_combinations_verified": 9,
        "landing_links_verified": len(links),
        "checks": ["landing page links and mobile layout", "no sampling", "chemical cohorts", "interval switching", "property cutoff",
            "view round-trip", "CSV export", "complete neural OOF", "prediction sets", "mobile overflow"]}
    (evidence / "browser_audit.json").write_text(json.dumps(outcome, indent=2))
    return outcome


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    print(json.dumps(check(parser.parse_args().study), indent=2))
