"""Chronos-2 adapter tests. The pipeline is a fake: no weights are downloaded."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import make_config, make_prices

from vnindex_forecast.features.build import build_features
from vnindex_forecast.models.chronos import (
    ChronosUnavailable,
    _distribution_stats,
    run_chronos_walk_forward,
)


class FakePipeline:
    """Mimics the documented output: list of (1, 21, 1) quantile tensors."""

    quantiles = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]

    def __init__(self) -> None:
        self.calls: list[int] = []

    def predict(self, inputs, prediction_length=1, batch_size=32):
        self.calls.append(len(inputs))
        outputs = []
        for series in inputs:
            last = float(series[-1])
            curve = last * np.exp(np.linspace(-0.02, 0.02, 21))
            outputs.append(np.repeat(curve.reshape(1, 21, 1), prediction_length, axis=0))
        return outputs


def _fixture(tmp_path):
    config = make_config(
        tmp_path,
        data={"min_rows": 50},
        chronos={
            "enabled": True,
            "model_id": "fake/chronos",
            "context_length": 64,
            "prediction_length": 1,
            "batch_size": 4,
        },
    )
    prices = make_prices(rows=150, seed=3)
    features = build_features(prices, config)
    return config, prices, features


def test_distribution_stats_are_directionally_sane():
    levels = np.asarray(FakePipeline.quantiles)
    last = 1000.0
    up_curve = last * np.exp(np.linspace(-0.01, 0.03, 21))
    down_curve = last * np.exp(np.linspace(-0.03, 0.01, 21))
    assert _distribution_stats(up_curve, last, levels)["p_up"] > 0.5
    assert _distribution_stats(down_curve, last, levels)["p_up"] < 0.5


def test_walk_forward_uses_batches_and_returns_all_points(tmp_path):
    config, prices, features = _fixture(tmp_path)
    fake = FakePipeline()
    result = run_chronos_walk_forward(features, prices, config, pipeline=fake)

    labelled = len(features.dropna(subset=["next_return"]))
    expected = labelled - config.model.walk_forward.min_train_rows
    assert len(result.frame) == expected
    assert sum(fake.calls) == expected
    assert max(fake.calls) <= 4
    assert result.points == expected
    assert result.inference_seconds >= 0
    assert result.per_point_ms >= 0
    assert result.frame["p_up"].between(0.0, 1.0).all()
    assert (result.frame["lower"] < result.frame["upper"]).all()
    assert result.frame["covered"].dtype == bool


def test_disabled_config_is_a_skip_not_a_crash(tmp_path):
    config, prices, features = _fixture(tmp_path)
    config.chronos.enabled = False
    with pytest.raises(ChronosUnavailable, match="enabled is false"):
        run_chronos_walk_forward(features, prices, config, pipeline=FakePipeline())
