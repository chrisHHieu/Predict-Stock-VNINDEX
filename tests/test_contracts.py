"""Pandera contracts catch the failures the hand-written checks catch."""

from __future__ import annotations

import pytest

pytest.importorskip("pandera")

from conftest import make_prices  # noqa: E402

from vnindex_forecast.data.contracts import (  # noqa: E402
    validate_processed,
    validate_raw,
)
from vnindex_forecast.features.build import build_features  # noqa: E402


def test_valid_raw_frame_passes(config):
    ok, detail = validate_raw(make_prices())
    assert ok, detail


def test_close_above_high_fails():
    frame = make_prices()
    frame.loc[3, "close"] = frame.loc[3, "high"] + 1.0
    ok, detail = validate_raw(frame)
    assert not ok
    assert "high_is_max" in detail


def test_duplicate_dates_fail():
    frame = make_prices()
    frame.loc[4, "date"] = frame.loc[3, "date"]
    ok, detail = validate_raw(frame)
    assert not ok
    assert "date" in detail


def test_extra_column_fails_the_strict_contract():
    frame = make_prices()
    frame["turnover"] = 1.0
    ok, _ = validate_raw(frame)
    assert not ok


def test_processed_feature_frame_passes(config):
    features = build_features(make_prices(), config)
    features["direction"] = features["direction"].astype("Int64")
    ok, detail = validate_processed(features, config)
    assert ok, detail


def test_processed_frame_rejects_a_non_binary_direction(config):
    features = build_features(make_prices(), config).copy()
    features["direction"] = features["direction"].astype("Int64")
    features.loc[features.index[0], "direction"] = 2
    ok, _ = validate_processed(features, config)
    assert not ok
