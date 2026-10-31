"""Feature drift monitoring.

Population Stability Index is computed per feature between a reference slice
(the earlier part of the training history) and a current slice (the most recent
``window_rows`` sessions). Thresholds follow the usual reading: below 0.10
stable, 0.10 to 0.25 moderate, above 0.25 significant.
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from vnindex_forecast.config import AppConfig  # noqa: E402
from vnindex_forecast.features.build import feature_columns  # noqa: E402
from vnindex_forecast.models import registry  # noqa: E402


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, bins: int = 10, epsilon: float = 1e-4
) -> float:
    """PSI with quantile bins taken from the reference distribution."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if len(reference) < 20 or len(current) < 5:
        return 0.0

    edges = np.unique(np.quantile(reference, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0] = min(edges[0], current.min()) if len(current) else edges[0]
    edges[-1] = max(edges[-1], current.max()) if len(current) else edges[-1]

    reference_counts, _ = np.histogram(reference, bins=edges)
    current_counts, _ = np.histogram(np.clip(current, edges[0], edges[-1]), bins=edges)
    reference_share = np.clip(reference_counts / max(reference_counts.sum(), 1), epsilon, None)
    current_share = np.clip(current_counts / max(current_counts.sum(), 1), epsilon, None)
    ratio = np.log(current_share / reference_share)
    return float(np.sum((current_share - reference_share) * ratio))


def empty_current_bins(
    reference: np.ndarray, current: np.ndarray, bins: int = 10
) -> int:
    """Count reference bins that receive no current observation.

    A large PSI with empty bins means the two samples barely overlap; the value
    is then a separation signal, not a calibrated distance.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if len(reference) < 20 or len(current) < 5:
        return 0
    edges = np.unique(np.quantile(reference, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 3:
        return 0
    edges[0] = min(edges[0], current.min())
    edges[-1] = max(edges[-1], current.max())
    counts, _ = np.histogram(np.clip(current, edges[0], edges[-1]), bins=edges)
    return int((counts == 0).sum())


def _status(value: float, config: AppConfig) -> str:
    if value >= config.monitor.psi_significant:
        return "significant"
    if value >= config.monitor.psi_moderate:
        return "moderate"
    return "stable"


def run_drift(config: AppConfig) -> dict[str, Any]:
    features_path = config.path("processed_dir") / "features.csv"
    if not features_path.exists():
        raise FileNotFoundError(f"features not found: {features_path}. Run `features` first.")
    features = pd.read_csv(features_path, parse_dates=["date"]).sort_values("date")
    names = feature_columns(config)

    split = int(len(features) * config.monitor.reference_fraction)
    if len(features) <= config.monitor.window_rows or split < 60:
        raise ValueError("not enough feature rows for a drift report")

    reference = features.iloc[:split]
    current = features.iloc[-config.monitor.window_rows :]

    per_feature = []
    for name in names:
        value = population_stability_index(
            reference[name].to_numpy(), current[name].to_numpy(), config.monitor.psi_bins
        )
        reference_values = reference[name].to_numpy(dtype=float)
        current_values = current[name].to_numpy(dtype=float)
        std = float(np.nanstd(reference_values))
        shift = (
            float((np.nanmean(current_values) - np.nanmean(reference_values)) / std)
            if std > 0
            else 0.0
        )
        empty_bins = empty_current_bins(
            reference[name].to_numpy(), current[name].to_numpy(), config.monitor.psi_bins
        )
        per_feature.append(
            {
                "feature": name,
                "psi": round(value, 4),
                "status": _status(value, config),
                "empty_current_bins": empty_bins,
                "mean_reference": round(float(np.nanmean(reference_values)), 6),
                "mean_current": round(float(np.nanmean(current_values)), 6),
                "standardized_mean_shift": round(shift, 4),
            }
        )

    per_feature.sort(key=lambda item: item["psi"], reverse=True)
    summary = {
        "stable": sum(item["status"] == "stable" for item in per_feature),
        "moderate": sum(item["status"] == "moderate" for item in per_feature),
        "significant": sum(item["status"] == "significant" for item in per_feature),
        "max_psi": per_feature[0]["psi"] if per_feature else 0.0,
    }

    report = {
        "generated_at": registry.utc_now(),
        "method": "population stability index, quantile bins from the reference slice",
        "note": (
            "PSI is unbounded. When a reference bin receives no current observations the value "
            "mostly measures separation, not a calibrated distance; empty_current_bins says how "
            "many bins were empty."
        ),
        "thresholds": {
            "moderate": config.monitor.psi_moderate,
            "significant": config.monitor.psi_significant,
        },
        "reference": {
            "rows": int(len(reference)),
            "start": pd.Timestamp(reference["date"].min()).date().isoformat(),
            "end": pd.Timestamp(reference["date"].max()).date().isoformat(),
        },
        "current": {
            "rows": int(len(current)),
            "start": pd.Timestamp(current["date"].min()).date().isoformat(),
            "end": pd.Timestamp(current["date"].max()).date().isoformat(),
        },
        "summary": summary,
        "features": per_feature,
    }

    reports_dir = config.path("reports_dir")
    figures_dir = config.path("figures_dir")
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "drift.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    top = per_feature[: min(12, len(per_feature))]
    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=120)
    ax.barh([item["feature"] for item in reversed(top)], [item["psi"] for item in reversed(top)])
    ax.axvline(config.monitor.psi_moderate, color="orange", linestyle="--", linewidth=1)
    ax.axvline(config.monitor.psi_significant, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("PSI (reference vs last window)")
    ax.set_title("Feature drift")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "drift_psi.png")
    plt.close(fig)

    return report


def format_report(report: dict[str, Any]) -> str:
    lines = [
        f"reference {report['reference']['start']} -> {report['reference']['end']} "
        f"({report['reference']['rows']} rows)",
        f"current   {report['current']['start']} -> {report['current']['end']} "
        f"({report['current']['rows']} rows)",
        "",
        f"{'feature':<24}{'psi':>8}  {'status':<12}{'mean shift (sd)':>16}",
    ]
    for item in report["features"]:
        lines.append(
            f"{item['feature']:<24}{item['psi']:>8.4f}  {item['status']:<12}"
            f"{item['standardized_mean_shift']:>16.3f}"
        )
    summary = report["summary"]
    lines.append("")
    lines.append(
        f"stable={summary['stable']} moderate={summary['moderate']} "
        f"significant={summary['significant']} max_psi={summary['max_psi']}"
    )
    return "\n".join(lines)
