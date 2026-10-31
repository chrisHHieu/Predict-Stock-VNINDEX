"""MLflow logging and registry alias, against a throwaway SQLite backend."""

from __future__ import annotations

import pytest

pytest.importorskip("mlflow")

from conftest import make_config, make_prices, materialise  # noqa: E402

from vnindex_forecast.features.build import (  # noqa: E402
    feature_columns,
    load_or_build_features,
)
from vnindex_forecast.models.train import train  # noqa: E402
from vnindex_forecast.tracking.mlflow import track_training  # noqa: E402


def test_training_run_is_logged_and_registered(tmp_path, monkeypatch):
    config = make_config(
        tmp_path,
        model={
            "tracking": {
                "experiment": "pytest-vnindex",
                "registered_model_name": "vnindex_direction_test",
                "alias": "champion",
            }
        },
    )
    materialise(config, make_prices())
    result = train(config)
    names = feature_columns(config)
    features = load_or_build_features(config).dropna(subset=names)
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)

    tracking = track_training(
        config,
        artifact=result.artifact,
        metrics=result.metrics,
        feature_names=names,
        input_example=features[names].tail(2),
    )
    assert tracking["status"] == "ok", tracking
    assert tracking["run_id"]
    assert tracking["model_version"] is not None
    assert tracking["alias"] == "champion"
