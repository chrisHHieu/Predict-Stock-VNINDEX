"""Evidently report generation on the synthetic feature frame."""

from __future__ import annotations

import pytest

pytest.importorskip("evidently")

from conftest import make_config, make_prices  # noqa: E402

from vnindex_forecast.features.build import build_features  # noqa: E402
from vnindex_forecast.monitor.evidently_report import run_evidently  # noqa: E402


def test_report_writes_html_and_json(tmp_path):
    config = make_config(tmp_path)
    features = build_features(make_prices(rows=300, seed=5), config)
    summary = run_evidently(
        config,
        reference=features.iloc[:200],
        current=features.iloc[-60:],
    )
    html = config.path("reports_dir") / "evidently_drift.html"
    payload = config.path("reports_dir") / "evidently_drift.json"
    assert html.exists() and html.stat().st_size > 1000
    assert payload.exists()
    assert summary["reference_rows"] == 200
    assert summary["current_rows"] == 60
    assert summary["features"] == len(features.columns) - 3
