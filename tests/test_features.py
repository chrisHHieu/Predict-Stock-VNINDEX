"""Feature construction: alignment, leakage safety, determinism, inference path."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vnindex_forecast.features.build import (
    build_feature_frame_from_bars,
    build_features,
    feature_columns,
)


def test_feature_frame_has_no_missing_values(features, config):
    names = feature_columns(config)
    assert len(names) == 18
    assert features[names].isna().sum().sum() == 0
    assert np.isfinite(features[names].to_numpy()).all()
    assert features["date"].is_monotonic_increasing


def test_targets_point_at_the_next_session(features, prices):
    merged = features.merge(prices, on="date", how="left", suffixes=("", "_px"))
    ordered = prices.sort_values("date").reset_index(drop=True)
    close = ordered.set_index("date")["close"]
    expected = np.log(close.shift(-1) / close)
    for _, row in merged.iterrows():
        target = expected.loc[row["date"]]
        if pd.notna(target):
            assert row["next_return"] == pytest.approx(target, rel=1e-9, abs=1e-12)
            assert row["direction"] == int(target > 0)


def test_first_row_respects_min_history(features, prices, config):
    assert features["date"].min() == prices["date"].iloc[config.features.min_history - 1]


def test_future_prices_do_not_change_past_features(prices, config):
    baseline = build_features(prices, config)
    cutoff = prices["date"].iloc[120]

    mutated = prices.copy()
    future = mutated["date"] > cutoff
    for column in ("open", "high", "low", "close"):
        mutated.loc[future, column] = mutated.loc[future, column] * 1.5
    mutated.loc[future, "volume"] = mutated.loc[future, "volume"] * 3

    after = build_features(mutated, config)
    columns = feature_columns(config)
    left = baseline[baseline["date"] <= cutoff][["date", *columns]].reset_index(drop=True)
    right = after[after["date"] <= cutoff][["date", *columns]].reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)


def test_build_is_deterministic(prices, config):
    first = build_features(prices, config)
    second = build_features(prices, config)
    pd.testing.assert_frame_equal(first, second)


def test_inference_frame_from_sixty_bars(prices, config):
    bars = prices.tail(60).to_dict("records")
    frame = build_feature_frame_from_bars(bars, config)
    assert len(frame) == 1
    names = feature_columns(config)
    assert frame[names].isna().sum().sum() == 0


def test_inference_rejects_short_window(prices, config):
    bars = prices.tail(50).to_dict("records")
    with pytest.raises(ValueError, match="at least 60 bars"):
        build_feature_frame_from_bars(bars, config)


def test_inference_rejects_inconsistent_bar(prices, config):
    bars = prices.tail(60).to_dict("records")
    bars[-1]["high"] = bars[-1]["close"] - 1.0
    with pytest.raises(ValueError, match="high must be"):
        build_feature_frame_from_bars(bars, config)
