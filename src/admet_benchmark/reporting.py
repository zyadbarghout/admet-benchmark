"""Small static reports for reviewing model choice and held-out performance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialise {type(value)!r}")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, default=_json_default) + "\n")


def _markdown_metric_table(table: pd.DataFrame) -> str:
    """Render a tiny Markdown table without making ``tabulate`` a dependency."""

    rows = ["| metric | held_out_test |", "| --- | ---: |"]
    for metric, value in table.iloc[:, 0].items():
        rows.append(f"| {metric} | {float(value):.5f} |")
    return "\n".join(rows)


def plot_regression_diagnostics(
    y_true: np.ndarray, y_pred: np.ndarray, path: Path, endpoint: str
) -> None:
    figure, axis = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    axis.scatter(y_true, y_pred, alpha=0.55, s=18, color="#1f77b4", edgecolors="none")
    limits = [min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))]
    axis.plot(limits, limits, color="#2d2d2d", linewidth=1, linestyle="--")
    axis.set(xlabel="Observed target (evaluation scale)", ylabel="Predicted target (evaluation scale)")
    axis.set_title(f"Held-out prediction: {endpoint}")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_classification_diagnostics(
    y_true: np.ndarray, probability: np.ndarray, path: Path, endpoint: str
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    RocCurveDisplay.from_predictions(y_true, probability, ax=axes[0])
    PrecisionRecallDisplay.from_predictions(y_true, probability, ax=axes[1])
    axes[0].set_title(f"ROC: {endpoint}")
    axes[1].set_title(f"Precision-recall: {endpoint}")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def write_report(
    output_dir: Path,
    *,
    task: str,
    target: str,
    selected_model: str,
    validation_score: float,
    metrics: dict[str, float],
    data_summary: dict[str, Any],
) -> None:
    """Write a concise Markdown/HTML record of the locked test evaluation."""

    metric_table = pd.DataFrame([metrics]).T.rename(columns={0: "held_out_test"})
    metric_table.index.name = "metric"
    metric_table.to_csv(output_dir / "held_out_metrics.csv")
    metric_html = metric_table.to_html(float_format=lambda value: f"{value:.5f}")
    details = pd.DataFrame([data_summary]).to_html(index=False)
    (output_dir / "report.html").write_text(
        "<html><head><title>ADMET benchmark report</title>"
        "<style>body{font-family:system-ui;margin:2rem;max-width:1000px}"
        "table{border-collapse:collapse}th,td{padding:.45rem .7rem;border:1px solid #ddd}"
        "th{background:#f4f5f6}</style></head><body>"
        f"<h1>{target}: {task} benchmark</h1>"
        f"<p><strong>Selected model:</strong> {selected_model}<br>"
        f"<strong>Cross-validation selection score:</strong> {validation_score:.5f}</p>"
        "<p>The model was selected using cross-validation on the training split. "
        "The figures and metrics below use the held-out test split only.</p>"
        "<h2>Held-out metrics</h2>"
        f"{metric_html}<h2>Data summary</h2>{details}"
        "<p>See <code>validation_leaderboard.csv</code> for model selection and "
        "<code>hpo_candidates.csv</code> for every evaluated configuration.</p>"
        "</body></html>"
    )
    lines = [
        f"# {target}: {task} benchmark",
        "",
        f"Selected model: `{selected_model}`",
        f"Cross-validation selection score: `{validation_score:.5f}`",
        "",
        "The selected model was chosen on cross-validation over the training split. "
        "The following metrics are from the untouched test split.",
        "",
        _markdown_metric_table(metric_table),
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")
