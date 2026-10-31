"""Data contract checks."""

from __future__ import annotations

import pandas as pd

from vnindex_forecast.data import validate as validate_module
from vnindex_forecast.data.validate import format_report, run_validate, validate_frame


def _failed(checks) -> set[str]:
    return {check.name for check in checks if not check.passed}


def test_clean_frame_passes_every_check(prices, config):
    checks = validate_frame(prices, config)
    assert _failed(checks) == set(), format_report(checks)
    assert all(check.passed for check in checks)


def test_duplicate_dates_fail(prices, config):
    frame = pd.concat([prices, prices.iloc[[10]]], ignore_index=True)
    frame = frame.sort_values("date").reset_index(drop=True)
    failed = _failed(validate_frame(frame, config))
    assert "unique_dates" in failed
    assert "dates_increasing" in failed


def test_close_above_high_fails(prices, config):
    frame = prices.copy()
    frame.loc[5, "close"] = frame.loc[5, "high"] + 10.0
    assert "ohlc_bounds" in _failed(validate_frame(frame, config))


def test_too_few_rows_fails(prices, config):
    config.data.min_rows = 1000
    assert "min_rows" in _failed(validate_frame(prices, config))


def test_negative_volume_fails(prices, config):
    frame = prices.copy()
    frame.loc[3, "volume"] = -1
    assert "volume_non_negative" in _failed(validate_frame(frame, config))


def test_schema_is_monkey_patchable(prices, config, monkeypatch):
    monkeypatch.setattr(
        validate_module, "REQUIRED_COLUMNS", [*validate_module.REQUIRED_COLUMNS, "turnover"]
    )
    checks = validate_frame(prices, config)
    assert "schema" in _failed(checks)
    assert any("turnover" in check.detail for check in checks)


def test_run_validate_returns_exit_signal(prices, config):
    ok, checks = run_validate(config, prices)
    assert ok is True
    assert len(checks) >= 10
