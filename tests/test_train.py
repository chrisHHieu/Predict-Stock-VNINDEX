"""Walk-forward training, artifact registry and reporting."""

from __future__ import annotations

import json
import math

from conftest import materialise

from vnindex_forecast.models import registry
from vnindex_forecast.models.evaluate import evaluate, run_walk_forward
from vnindex_forecast.models.train import train


def test_walk_forward_predictions_are_out_of_sample(features, config):
    result = run_walk_forward(features, config)
    predictions = result.predictions

    assert len(predictions) > 0
    assert result.folds >= 2  # 60 -> 81 -> 90 rows
    assert predictions["date"].is_monotonic_increasing
    assert predictions["p_up"].between(0.0, 1.0).all()
    assert predictions["threshold"].isin(config.model.threshold_grid).all()
    assert predictions["train_rows"].is_monotonic_increasing
    assert predictions["train_rows"].iloc[0] == config.model.walk_forward.min_train_rows


def test_train_is_reproducible_and_registers_artifact(prices, config):
    materialise(config, prices)
    first = train(config)
    second = train(config)

    for key in ("direction", "return_regression", "strategy", "buy_and_hold", "baselines"):
        assert first.metrics[key] == second.metrics[key], f"{key} differs between runs"
    assert first.metrics["direction"]["accuracy"] == second.metrics["direction"]["accuracy"]

    directory = second.artifact.directory
    assert (directory / "model.joblib").exists()
    assert (directory / "metadata.json").exists()

    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    for key in ("version", "model_type", "seed", "config_hash", "data_sha256", "trained_at",
                "metrics", "threshold", "features", "git_commit"):
        assert key in metadata
    assert metadata["seed"] == 42
    assert metadata["model_type"] == "lightgbm"
    assert len(metadata["data_sha256"]) == 64

    assert registry.latest_version(config) == second.artifact.version
    assert len(registry.list_versions(config)) == 2

    artifact = registry.load_artifact(config)
    assert artifact.version == second.artifact.version
    assert artifact.threshold == second.artifact.threshold
    assert set(artifact.payload["feature_names"]) == set(metadata["features"])
    assert artifact.payload["conformal_interval"].coverage == 0.8
    assert metadata["conformal"]["calibration_rows"] > 0


def test_evaluate_writes_metrics_and_figures(prices, config):
    materialise(config, prices)
    train(config)
    metrics = evaluate(config)

    assert metrics["walk_forward"]["predictions_reused"] is True
    assert math.isfinite(metrics["direction"]["accuracy"])
    assert 0.0 <= metrics["direction"]["accuracy"] <= 1.0

    reports = config.path("reports_dir")
    assert (reports / "metrics.json").exists()
    stored = json.loads((reports / "metrics.json").read_text(encoding="utf-8"))
    assert stored["model_version"] == metrics["model_version"]
    assert stored["strategy"]["total_return"] == metrics["strategy"]["total_return"]

    figures = sorted(path.name for path in config.path("figures_dir").glob("*.png"))
    assert figures == ["calibration.png", "equity_curve.png", "feature_importance.png"]

    assert len(metrics["method_comparison"]) == 4
    methods = {item["method"]: item for item in metrics["method_comparison"]}
    assert set(methods) == {"naive_persistence", "logistic", "lightgbm", "chronos2_zero_shot"}
    assert methods["lightgbm"]["coverage_80"] is not None
    assert metrics["chronos"]["status"] == "skipped"
    assert 0.0 <= metrics["conformal"]["walk_forward"]["coverage"] <= 1.0
    assert (reports / "method_comparison.json").exists()
    assert (reports / "confusion.json").exists()
