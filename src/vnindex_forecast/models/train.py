"""Training entry point.

``train`` does three things:

1. runs the walk-forward backtest to get honest out-of-sample metrics,
2. selects the served decision threshold on a time ordered holdout that is
   never touched by the backtest,
3. fits the final classifier and regressor on all labelled rows and writes a
   versioned artifact.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import pandas as pd

from vnindex_forecast.config import AppConfig
from vnindex_forecast.features.build import feature_columns, load_or_build_features
from vnindex_forecast.models import registry
from vnindex_forecast.models.evaluate import (
    choose_threshold,
    compute_data_sha256,
    compute_metrics,
    run_walk_forward,
    write_predictions,
)
from vnindex_forecast.models.factory import build_classifier, build_regressor


@dataclasses.dataclass
class TrainResult:
    artifact: registry.Artifact
    metrics: dict[str, Any]
    predictions: pd.DataFrame


def select_served_threshold(features: pd.DataFrame, config: AppConfig) -> float:
    """Threshold chosen on the most recent training rows, before deployment."""
    names = feature_columns(config)
    holdout = config.model.holdout_rows
    if len(features) < holdout + 60:
        return 0.5

    fit_part = features.iloc[:-holdout]
    hold_part = features.iloc[-holdout:]
    if fit_part["direction"].nunique() < 2 or hold_part["direction"].nunique() < 2:
        return 0.5

    model = build_classifier(config, names)
    model.fit(fit_part[names], fit_part["direction"].astype(int))
    probabilities = model.predict_proba(hold_part[names])[:, 1]
    return choose_threshold(probabilities, hold_part["next_return"].to_numpy(), config)


def train(config: AppConfig, rebuild_features: bool = False) -> TrainResult:
    """Run the backtest, fit the final models and register the artifact."""
    features = load_or_build_features(config, rebuild=rebuild_features)
    names = feature_columns(config)
    labelled = features.dropna(subset=names + ["direction", "next_return"]).reset_index(drop=True)

    walk_forward = run_walk_forward(labelled, config)
    metrics = compute_metrics(walk_forward.predictions, config)
    data_sha256 = compute_data_sha256(config)
    write_predictions(config, walk_forward, data_sha256)

    threshold = select_served_threshold(labelled, config)

    classifier = build_classifier(config, names)
    classifier.fit(labelled[names], labelled["direction"].astype(int))
    regressor = build_regressor(config, names)
    regressor.fit(labelled[names], labelled["next_return"].astype(float))

    payload = {
        "classifier": classifier,
        "regressor": regressor,
        "feature_names": names,
        "threshold": float(threshold),
        "window_bars": int(config.features.min_history),
        "config_hash": config.hash(),
    }

    metadata = {
        "model_type": config.model.name,
        "task": "next session direction (binary) and next session log return (regression)",
        "seed": config.seed,
        "config_hash": config.hash(),
        "config_path": config.config_path,
        "data_sha256": data_sha256,
        "n_train_rows": int(len(labelled)),
        "features": names,
        "window_bars": int(config.features.min_history),
        "threshold": float(threshold),
        "trained_at": registry.utc_now(),
        "git_commit": registry.git_commit(),
        "target_timing": "direction and next_return at row t refer to session t+1",
        "metrics": metrics,
        "walk_forward": {
            "min_train_rows": config.model.walk_forward.min_train_rows,
            "refit_every": config.model.walk_forward.refit_every,
            "folds": walk_forward.folds,
        },
        "training_window": {
            "first": pd.Timestamp(labelled["date"].min()).date().isoformat(),
            "last": pd.Timestamp(labelled["date"].max()).date().isoformat(),
        },
        "package_version": "1.0.0",
    }

    artifact = registry.save_artifact(config, payload, metadata)
    metrics["model_version"] = artifact.version
    metrics["generated_at"] = registry.utc_now()
    metrics["served_threshold"] = float(threshold)
    metrics["walk_forward"] = {
        "folds": walk_forward.folds,
        "min_train_rows": config.model.walk_forward.min_train_rows,
        "refit_every": config.model.walk_forward.refit_every,
        "threshold_grid": [
            round(float(walk_forward.thresholds[0]), 4),
            round(float(sorted(walk_forward.thresholds)[len(walk_forward.thresholds) // 2]), 4),
            round(float(walk_forward.thresholds[-1]), 4),
        ],
        "predictions_reused": False,
    }
    (config.path("reports_dir") / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    return TrainResult(artifact=artifact, metrics=metrics, predictions=walk_forward.predictions)
