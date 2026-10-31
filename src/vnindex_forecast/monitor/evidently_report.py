"""Evidently data-drift report between the training reference and recent rows.

The hand-written PSI report stays the quick check. Evidently adds a HTML/JSON
report with per-column statistical tests, intended for humans reviewing a
retraining decision.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from vnindex_forecast.config import AppConfig
from vnindex_forecast.features.build import feature_columns
from vnindex_forecast.models import registry


def _find_values(node: Any, key: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(node, dict):
        for name, value in node.items():
            if name == key and not isinstance(value, (dict, list)):
                found.append(value)
            else:
                found.extend(_find_values(value, key))
    elif isinstance(node, list):
        for item in node:
            found.extend(_find_values(item, key))
    return found


def run_evidently(
    config: AppConfig,
    reference: pd.DataFrame | None = None,
    current: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Run Evidently's DataDriftPreset and write HTML plus JSON."""
    import importlib.util

    if importlib.util.find_spec("evidently") is None:
        raise RuntimeError("evidently is not installed; install the `modern` extra")

    names = feature_columns(config)
    if reference is None or current is None:
        features_path = config.path("processed_dir") / "features.csv"
        if not features_path.exists():
            raise FileNotFoundError(
                f"features not found: {features_path}. Run `features` first."
            )
        features = pd.read_csv(features_path, parse_dates=["date"]).sort_values("date")
        split = int(len(features) * config.monitor.reference_fraction)
        reference = features.iloc[:split]
        current = features.iloc[-config.monitor.window_rows :]

    reference_frame = reference[names].reset_index(drop=True)
    current_frame = current[names].reset_index(drop=True)

    from evidently import Report
    from evidently.presets import DataDriftPreset

    report = Report([DataDriftPreset()])
    snapshot = report.run(reference_data=reference_frame, current_data=current_frame)

    reports_dir = config.path("reports_dir")
    reports_dir.mkdir(parents=True, exist_ok=True)
    html_path = reports_dir / "evidently_drift.html"
    json_path = reports_dir / "evidently_drift.json"
    snapshot.save_html(str(html_path))
    payload = snapshot.dict()
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    drifted = _find_values(payload, "number_of_drifted_columns")
    share = _find_values(payload, "share_of_drifted_columns")
    summary = {
        "generated_at": registry.utc_now(),
        "reference_rows": int(len(reference_frame)),
        "current_rows": int(len(current_frame)),
        "features": len(names),
        "drifted_columns": int(drifted[0]) if drifted else None,
        "share_of_drifted_columns": float(share[0]) if share else None,
        "html": str(html_path),
        "json": str(json_path),
    }
    (reports_dir / "evidently_drift_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def format_report(summary: dict[str, Any]) -> str:
    lines = [
        f"reference rows   {summary['reference_rows']}",
        f"current rows     {summary['current_rows']}",
        f"features         {summary['features']}",
        f"drifted columns  {summary['drifted_columns']}"
        f" (share {summary['share_of_drifted_columns']})",
        f"html             {summary['html']}",
        f"json             {summary['json']}",
    ]
    return "\n".join(lines)
