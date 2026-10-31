"""Offline fixtures: every test runs on synthetic bars built in code."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from vnindex_forecast.config import AppConfig, load_config, set_seed
from vnindex_forecast.features.build import build_features


def make_prices(rows: int = 150, seed: int = 7) -> pd.DataFrame:
    """Deterministic random-walk OHLCV bars that satisfy the data contract."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=rows)
    log_returns = rng.normal(0.0004, 0.011, rows)
    close = np.round(1000.0 * np.exp(np.cumsum(log_returns)), 2)
    open_ = np.round(close * (1.0 + rng.normal(0.0, 0.002, rows)), 2)
    high = np.round(np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.003, rows))), 2)
    low = np.round(np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.003, rows))), 2)

    high = np.maximum.reduce([high, open_, close])
    low = np.minimum.reduce([low, open_, close])
    volume = rng.integers(1_000_000, 9_000_000, rows)

    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume.astype("int64"),
        }
    )


def make_config(tmp_path: Path, **overrides: Any) -> AppConfig:
    """Config pointing every writable path into a temporary directory."""
    base: dict[str, Any] = {
        "seed": 42,
        "data": {"min_rows": 50},
        "model": {
            "n_estimators": 40,
            "regressor_n_estimators": 40,
            "num_leaves": 7,
            "min_child_samples": 5,
            "n_jobs": 1,
            "holdout_rows": 20,
            "walk_forward": {
                "min_train_rows": 60,
                "refit_every": 21,
                "inner_validation_rows": 30,
            },
        },
        "paths": {
            "raw_dir": str(tmp_path / "data" / "raw"),
            "processed_dir": str(tmp_path / "data" / "processed"),
            "models_dir": str(tmp_path / "models"),
            "reports_dir": str(tmp_path / "reports"),
            "figures_dir": str(tmp_path / "reports" / "figures"),
        },
    }
    merged = _merge(base, overrides)
    config = load_config(overrides=merged)
    config.ensure_dirs()
    set_seed(config.seed)
    return config


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def materialise(config: AppConfig, prices: pd.DataFrame) -> pd.DataFrame:
    """Write the processed price CSV and build the feature CSV for a temp config."""
    processed = config.path("processed_dir")
    processed.mkdir(parents=True, exist_ok=True)
    prices.to_csv(processed / "vnindex_daily.csv", index=False)
    features = build_features(prices, config)
    features.to_csv(processed / "features.csv", index=False)
    return features


@pytest.fixture
def prices() -> pd.DataFrame:
    return make_prices()


@pytest.fixture
def config(tmp_path: Path) -> AppConfig:
    return make_config(tmp_path)


@pytest.fixture
def features(prices: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    return build_features(prices, config)
