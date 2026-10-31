"""Pandera data contracts for the raw OHLCV frame and the feature frame.

The hand-written checks in :mod:`vnindex_forecast.data.validate` stay in place:
they produce readable per-rule output for the CLI. Pandera adds a declarative,
fail-loud schema that also guards direct callers of the feature builder.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pandera.pandas as pa

from vnindex_forecast.config import AppConfig
from vnindex_forecast.features.build import feature_columns


def _high_is_max(frame: pd.DataFrame) -> bool:
    return bool((frame["high"] >= frame[["open", "low", "close"]].max(axis=1)).all())


def _low_is_min(frame: pd.DataFrame) -> bool:
    return bool((frame["low"] <= frame[["open", "high", "close"]].min(axis=1)).all())


RAW_SCHEMA = pa.DataFrameSchema(
    {
        "date": pa.Column(pa.DateTime, nullable=False),
        "open": pa.Column(float, pa.Check.gt(0), nullable=False),
        "high": pa.Column(float, pa.Check.gt(0), nullable=False),
        "low": pa.Column(float, pa.Check.gt(0), nullable=False),
        "close": pa.Column(float, pa.Check.gt(0), nullable=False),
        "volume": pa.Column(float, pa.Check.ge(0), nullable=False),
    },
    checks=[
        pa.Check(_high_is_max, name="high_is_max"),
        pa.Check(_low_is_min, name="low_is_min"),
    ],
    unique=["date"],
    strict=True,
    coerce=True,
    name="vnindex_raw_ohlcv",
)


def processed_feature_schema(config: AppConfig) -> pa.DataFrameSchema:
    """Feature frame contract, with the feature list taken from the config."""
    columns: dict[str, Any] = {
        "date": pa.Column(pa.DateTime, nullable=False),
        "next_return": pa.Column(float, nullable=False),
        "direction": pa.Column(pa.Int64, pa.Check.isin([0, 1]), nullable=False),
    }
    for name in feature_columns(config):
        columns[name] = pa.Column(
            float,
            pa.Check(lambda series: np.isfinite(series.to_numpy(dtype=float)).all()),
            nullable=False,
        )
    return pa.DataFrameSchema(
        columns,
        checks=[pa.Check(lambda frame: len(frame) > 0, name="non_empty")],
        unique=["date"],
        strict=False,
        coerce=True,
        name="vnindex_processed_features",
    )


def _failure_detail(error: pa.errors.SchemaErrors, limit: int = 400) -> str:
    cases = getattr(error, "failure_cases", None)
    if cases is None or len(cases) == 0:
        return str(error)[:limit]
    labels = []
    for _, row in cases.iterrows():
        labels.append(f"{row.get('column')}:{row.get('check')}")
    unique = sorted(set(labels))
    return f"{len(cases)} failure(s): " + ", ".join(unique)[:limit]


def validate_raw(frame: pd.DataFrame) -> tuple[bool, str]:
    """Validate a tidy OHLCV frame against the raw contract."""
    try:
        RAW_SCHEMA.validate(frame, lazy=True)
        return True, "ok"
    except pa.errors.SchemaErrors as exc:
        return False, _failure_detail(exc)


def validate_processed(frame: pd.DataFrame, config: AppConfig) -> tuple[bool, str]:
    """Validate a processed feature frame against its contract."""
    try:
        processed_feature_schema(config).validate(frame, lazy=True)
        return True, "ok"
    except pa.errors.SchemaErrors as exc:
        return False, _failure_detail(exc)
