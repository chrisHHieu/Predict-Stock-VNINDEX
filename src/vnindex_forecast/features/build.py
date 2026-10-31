"""Leakage-safe feature construction for the daily VNINDEX frame.

Every feature at row ``t`` is a function of observations up to and including the
close of day ``t``. The targets are the next session's log return and its sign,
so the last row of any frame has no target and is only usable for inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from vnindex_forecast.config import AppConfig

TARGET_COLUMNS = ["next_return", "direction"]


def feature_columns(config: AppConfig) -> list[str]:
    """Feature names in a fixed order. Kept in sync with :func:`build_features`."""
    names: list[str] = [f"ret_lag_{lag}" for lag in config.features.return_lags]
    for window in config.features.rolling_windows:
        names.append(f"ret_mean_{window}")
        names.append(f"ret_std_{window}")
    names.append(f"rsi_{config.features.rsi_window}")
    names.append(f"atr_{config.features.atr_window}_norm")
    names.extend(f"close_ma{window}_ratio" for window in config.features.rolling_windows)
    names.append(f"vol_z_{config.features.volume_window}")
    names.append("dow")
    return names


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    alpha = 1.0 / window
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.where(avg_loss != 0.0, 100.0)
    rsi = rsi.where(avg_gain != 0.0, 0.0)
    return rsi


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return true_range.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def build_features(
    prices: pd.DataFrame,
    config: AppConfig,
    require_targets: bool = True,
) -> pd.DataFrame:
    """Build the model frame from a tidy OHLCV frame.

    Parameters
    ----------
    prices:
        Columns ``date, open, high, low, close, volume`` sorted ascending.
    config:
        Feature windows and history requirement.
    require_targets:
        When False the two target columns are kept as NaN on the final row so
        that the frame can be used for inference.
    """
    config.features.validate_windows()
    frame = prices.copy()
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"missing price columns: {missing}")

    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    if frame.empty:
        raise ValueError("cannot build features from an empty frame")

    close = frame["close"].astype(float)
    log_return = np.log(close).diff()

    out = pd.DataFrame({"date": frame["date"]})
    for lag in config.features.return_lags:
        out[f"ret_lag_{lag}"] = log_return.shift(lag - 1)

    for window in config.features.rolling_windows:
        # A window of w bars supplies w - 1 log returns because the first bar
        # has no previous close. Using w - 1 as the minimum keeps exactly
        # `min_history` bars sufficient for inference.
        rolling = log_return.rolling(window, min_periods=max(2, window - 1))
        out[f"ret_mean_{window}"] = rolling.mean()
        out[f"ret_std_{window}"] = rolling.std(ddof=0)

    out[f"rsi_{config.features.rsi_window}"] = _rsi(close, config.features.rsi_window)
    atr = _atr(
        frame["high"].astype(float),
        frame["low"].astype(float),
        close,
        config.features.atr_window,
    )
    out[f"atr_{config.features.atr_window}_norm"] = atr / close

    for window in config.features.rolling_windows:
        mean_close = close.rolling(window, min_periods=window).mean()
        out[f"close_ma{window}_ratio"] = close / mean_close - 1.0

    volume = frame["volume"].astype(float)
    volume_window = config.features.volume_window
    volume_mean = volume.rolling(volume_window, min_periods=volume_window).mean()
    volume_std = volume.rolling(volume_window, min_periods=volume_window).std(ddof=0)
    out[f"vol_z_{volume_window}"] = (volume - volume_mean) / volume_std.replace(0.0, np.nan)

    out["dow"] = frame["date"].dt.dayofweek.astype(float)
    out["next_return"] = log_return.shift(-1)
    out["direction"] = (out["next_return"] > 0).astype(float)

    feature_names = feature_columns(config)
    out = out.iloc[config.features.min_history - 1 :].reset_index(drop=True)
    out[feature_names] = out[feature_names].astype(float)
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=feature_names)

    if require_targets:
        out = out.dropna(subset=["next_return"])
        out = out[out["next_return"] != 0.0]
        out["direction"] = out["direction"].astype(int)
    else:
        out["direction"] = out["direction"].astype("Int64")

    return out.reset_index(drop=True)


def build_feature_frame_from_bars(
    bars: Iterable[dict[str, Any]], config: AppConfig
) -> pd.DataFrame:
    """Build an inference frame from client supplied OHLCV bars.

    Bars may carry an optional ISO date. When dates are absent, business days
    ending today are synthesised; the index is irrelevant to the features, only
    the order and the values matter.
    """
    records = [dict(bar) for bar in bars]
    if not records:
        raise ValueError("no bars supplied")
    frame = pd.DataFrame.from_records(records)
    for column in ("open", "high", "low", "close", "volume"):
        if column not in frame.columns:
            raise ValueError(f"every bar needs '{column}'")
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["open", "high", "low", "close", "volume"]].isna().any().any():
        raise ValueError("bars contain non numeric or missing values")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("prices must be positive")
    if (frame["high"] < frame[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("high must be the highest price in a bar")
    if (frame["low"] > frame[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("low must be the lowest price in a bar")
    if (frame["volume"] < 0).any():
        raise ValueError("volume must be non negative")
    if len(frame) < config.features.min_history:
        raise ValueError(
            f"at least {config.features.min_history} bars are required, received {len(frame)}"
        )

    if "date" in frame.columns:
        parsed = pd.to_datetime(frame["date"], errors="coerce")
        if parsed.isna().any():
            raise ValueError("bar dates could not be parsed")
        frame["date"] = parsed
    else:
        frame["date"] = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=len(frame))

    frame = frame.sort_values("date").reset_index(drop=True)
    return build_features(frame, config, require_targets=False)


def features_path(config: AppConfig) -> Path:
    return config.path("processed_dir") / "features.csv"


def load_or_build_features(config: AppConfig, rebuild: bool = False) -> pd.DataFrame:
    """Read the cached feature frame, building it from the price CSV if needed."""
    from vnindex_forecast.data.validate import load_processed

    path = features_path(config)
    if path.exists() and not rebuild:
        return pd.read_csv(path, parse_dates=["date"])

    prices = load_processed(config)
    frame = build_features(prices, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.10f")
    return frame
