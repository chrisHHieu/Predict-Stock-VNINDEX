"""Quantile models and split-conformal intervals, on synthetic prices."""

from __future__ import annotations

import numpy as np
from conftest import make_config, make_prices

from vnindex_forecast.features.build import build_features, feature_columns
from vnindex_forecast.models.conformal import (
    fit_served_interval,
    holdout_conformal,
    pinball_loss,
    run_conformal_walk_forward,
)


def _fixture(tmp_path):
    config = make_config(
        tmp_path,
        data={"min_rows": 50},
        model={
            "n_estimators": 40,
            "regressor_n_estimators": 40,
            "num_leaves": 7,
            "min_child_samples": 5,
            "n_jobs": 1,
            "holdout_rows": 20,
            "walk_forward": {
                "min_train_rows": 250,
                "refit_every": 50,
                "inner_validation_rows": 40,
            },
            "conformal": {"alpha": 0.8, "calibration_rows": 60},
        },
    )
    prices = make_prices(rows=500, seed=11)
    features = build_features(prices, config)
    return config, prices, features


def test_pinball_loss_is_zero_for_a_perfect_forecast():
    actual = np.array([0.01, -0.02, 0.0])
    assert pinball_loss(actual, actual, 0.9) == 0.0
    assert pinball_loss(actual, actual, 0.1) == 0.0


def test_walk_forward_intervals_have_measured_coverage(tmp_path):
    config, _, features = _fixture(tmp_path)
    result = run_conformal_walk_forward(features, config)

    labelled = len(features.dropna(subset=feature_columns(config) + ["next_return"]))
    assert result.folds == 4
    assert len(result.frame) == labelled - 250
    assert (result.frame["lower"] < result.frame["upper"]).all()
    assert 0.5 <= result.coverage <= 1.0
    assert result.average_width > 0
    assert set(result.pinball) == {"0.1", "0.5", "0.9"}
    assert all(value > 0 for value in result.pinball.values())
    assert result.elapsed_seconds >= 0


def test_holdout_interval_reports_its_own_coverage(tmp_path):
    config, _, features = _fixture(tmp_path)
    result = holdout_conformal(features, config)
    assert result is not None
    assert result.folds == 1
    assert result.target_coverage == 0.8
    assert 0.0 <= result.coverage <= 1.0
    assert len(result.frame) == 20


def test_served_interval_orders_quantiles(tmp_path):
    config, _, features = _fixture(tmp_path)
    interval = fit_served_interval(features, config)
    names = feature_columns(config)
    latest = features.dropna(subset=names).tail(1)
    bounds = interval.predict(latest, names).iloc[0]
    assert bounds["lower"] < bounds["q10"] <= bounds["q50"] <= bounds["q90"] < bounds["upper"]
    assert interval.adjustment >= 0
    assert interval.calibration_rows == 60
