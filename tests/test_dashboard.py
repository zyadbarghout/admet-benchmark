from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("plotly")

from admet_benchmark.dashboard import generate_suite_dashboard


def test_suite_dashboard_reads_saved_comparison_without_recomputing_metrics(tmp_path: Path):
    comparison = pd.DataFrame(
        {
            "run_name": ["toy__random", "toy__random", "toy__scaffold", "toy__scaffold"],
            "task": ["classification"] * 4,
            "target": ["label"] * 4,
            "split_strategy": ["random", "random", "scaffold", "scaffold"],
            "model": ["random_forest", "xgboost", "random_forest", "xgboost"],
            "validation_pr_auc": [0.70, 0.75, 0.61, 0.68],
            "test_pr_auc": [0.69, 0.74, 0.58, 0.64],
            "validation_selection_score": [0.70, 0.75, 0.61, 0.68],
        }
    )
    comparison.to_csv(tmp_path / "cross_split_model_comparison.csv", index=False)
    pd.DataFrame({"run_name": ["toy__random"]}).to_csv(
        tmp_path / "suite_summary.csv", index=False
    )

    output = generate_suite_dashboard(tmp_path)

    assert output.exists()
    html = output.read_text(encoding="utf-8")
    assert "Interactive cross-split model comparison" in html
    assert "cross_split_model_comparison.csv" in html
    assert "random_forest" in html
