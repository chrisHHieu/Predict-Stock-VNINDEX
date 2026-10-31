"""Zero-shot Chronos-2 forecasts for the walk-forward comparison.

Chronos-2 is a pretrained time-series foundation model: nothing here fits it on
VNINDEX. For every out-of-sample date the adapter sends the previous
``context_length`` closes and asks for a one-session forecast distribution. The
21 returned quantiles are treated as a predictive CDF: the probability of an up
session is ``1 - F(last_close)``, linearly interpolated between quantiles.

Loading the weights needs a network connection (or a warm HuggingFace cache).
When that fails the adapter raises :class:`ChronosUnavailable` and the caller
records a skip with the real reason instead of inventing a result.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any

import numpy as np
import pandas as pd

from vnindex_forecast.config import AppConfig


class ChronosUnavailable(RuntimeError):
    """Raised when the foundation model cannot be loaded."""


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _levels(pipeline: Any) -> np.ndarray:
    levels = getattr(pipeline, "quantiles", None)
    if levels is None:
        # The documented default of amazon/chronos-2 is these 21 levels.
        return np.linspace(0.01, 0.99, 21)
    return np.asarray(list(levels), dtype=float)


def _distribution_stats(
    forecast: np.ndarray, last_close: float, levels: np.ndarray
) -> dict[str, float]:
    """Turn one forecast's quantile curve into p_up and return quantiles."""
    values = np.asarray(forecast, dtype=float).reshape(-1)
    if len(values) != len(levels):
        raise ValueError(f"expected {len(levels)} quantiles, received {len(values)}")
    order = np.argsort(values)
    values = values[order]
    levels = levels[order]
    # np.interp needs a monotone x axis; forecast quantiles should already be
    # monotone, this only guards against tiny numerical wiggles.
    values = np.maximum.accumulate(values)

    p_below = float(np.interp(last_close, values, levels))
    p_up = float(np.clip(1.0 - p_below, 1e-4, 1.0 - 1e-4))
    q10 = float(np.interp(0.1, levels, values))
    q50 = float(np.interp(0.5, levels, values))
    q90 = float(np.interp(0.9, levels, values))
    return {
        "p_up": p_up,
        "q10_return": float(np.log(q10 / last_close)),
        "q50_return": float(np.log(q50 / last_close)),
        "q90_return": float(np.log(q90 / last_close)),
    }


@dataclasses.dataclass
class ChronosResult:
    frame: pd.DataFrame
    model_id: str
    context_length: int
    load_seconds: float
    inference_seconds: float
    points: int

    @property
    def per_point_ms(self) -> float:
        return 1000.0 * self.inference_seconds / max(self.points, 1)

    def summary(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model_id": self.model_id,
            "context_length": self.context_length,
            "points": int(self.points),
            "load_seconds": round(self.load_seconds, 3),
            "inference_seconds": round(self.inference_seconds, 3),
            "per_point_ms": round(self.per_point_ms, 2),
        }


class ChronosForecaster:
    """Thin, mockable wrapper around ``Chronos2Pipeline``."""

    def __init__(self, config: AppConfig, pipeline: Any | None = None) -> None:
        self.config = config
        self._pipeline = pipeline
        self.load_seconds = 0.0

    @property
    def pipeline(self) -> Any:
        if self._pipeline is None:
            started = time.perf_counter()
            try:
                from chronos import Chronos2Pipeline

                self._pipeline = Chronos2Pipeline.from_pretrained(
                    self.config.chronos.model_id,
                    device_map=self.config.chronos.device,
                )
            except Exception as exc:  # network, missing torch, bad cache, ...
                raise ChronosUnavailable(
                    f"could not load {self.config.chronos.model_id}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            self.load_seconds = time.perf_counter() - started
        return self._pipeline

    def predict_batch(self, series: list[np.ndarray]) -> list[dict[str, float]]:
        """Forecast one step for each 1-D close series and summarise each."""
        if not series:
            return []
        pipeline = self.pipeline
        outputs = pipeline.predict(
            [np.asarray(item, dtype=float) for item in series],
            prediction_length=self.config.chronos.prediction_length,
            batch_size=self.config.chronos.batch_size,
        )
        levels = _levels(pipeline)
        results = []
        for item, forecast in zip(series, outputs, strict=True):
            array = _to_numpy(forecast)
            # (prediction_length, n_quantiles, n_variates) -> first step, univariate
            curve = array[0, :, 0] if array.ndim == 3 else array.reshape(-1)
            results.append(_distribution_stats(curve, float(item[-1]), levels))
        return results


def _price_series(prices: pd.DataFrame) -> tuple[pd.DatetimeIndex, np.ndarray]:
    ordered = prices.sort_values("date").reset_index(drop=True)
    dates = pd.DatetimeIndex(pd.to_datetime(ordered["date"]))
    return dates, ordered["close"].to_numpy(dtype=float)


def run_chronos_walk_forward(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    config: AppConfig,
    pipeline: Any | None = None,
) -> ChronosResult:
    """Forecast every walk-forward test date with zero-shot Chronos-2."""
    if not config.chronos.enabled:
        raise ChronosUnavailable("chronos.enabled is false in this config")

    forecaster = ChronosForecaster(config, pipeline=pipeline)
    dates, closes = _price_series(prices)
    positions = {date: index for index, date in enumerate(dates)}
    context_length = min(config.chronos.context_length, len(closes))

    frame = features.sort_values("date").reset_index(drop=True)
    frame = frame.dropna(subset=["next_return"]).reset_index(drop=True)
    # The first forecastable date is the first walk-forward test row; earlier
    # rows are training data only and are not part of the 83-point comparison.
    minimum = config.model.walk_forward.min_train_rows
    targets = frame.iloc[minimum:].reset_index(drop=True)

    contexts: list[np.ndarray] = []
    last_closes: list[float] = []
    target_rows: list[dict[str, Any]] = []
    for _, row in targets.iterrows():
        date = pd.Timestamp(row["date"])
        position = positions.get(date)
        if position is None:
            raise ValueError(f"feature date {date.date()} is missing from the price frame")
        start = max(0, position + 1 - context_length)
        window = closes[start : position + 1]
        if len(window) < 64:
            continue
        contexts.append(window)
        last_closes.append(float(window[-1]))
        target_rows.append(
            {
                "date": date,
                "actual": float(row["next_return"]),
                "direction": int(row["direction"]),
            }
        )

    started = time.perf_counter()
    summaries: list[dict[str, float]] = []
    batch_size = config.chronos.batch_size
    for offset in range(0, len(contexts), batch_size):
        if time.perf_counter() - started > config.chronos.time_budget_seconds:
            raise TimeoutError(
                f"chronos inference exceeded {config.chronos.time_budget_seconds:.0f}s "
                f"after {len(summaries)} of {len(contexts)} points"
            )
        summaries.extend(forecaster.predict_batch(contexts[offset : offset + batch_size]))
    inference_seconds = time.perf_counter() - started

    records = []
    for row, summary in zip(target_rows, summaries, strict=True):
        covered = summary["q10_return"] <= row["actual"] <= summary["q90_return"]
        records.append(
            {
                **row,
                **summary,
                "lower": summary["q10_return"],
                "upper": summary["q90_return"],
                "covered": bool(covered),
                "width": summary["q90_return"] - summary["q10_return"],
            }
        )
    result_frame = pd.DataFrame(records)
    return ChronosResult(
        frame=result_frame,
        model_id=config.chronos.model_id,
        context_length=context_length,
        load_seconds=forecaster.load_seconds,
        inference_seconds=inference_seconds,
        points=len(result_frame),
    )
