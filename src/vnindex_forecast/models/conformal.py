"""Quantile LightGBM with split-conformal intervals.

Three pinball-loss regressors estimate the 10th, 50th and 90th percentile of the
next-session log return. A split-conformal step widens the raw interval with the
80th percentile of the conformity scores on a calibration slice::

    score = max(q10 - y, y - q90)
    interval = [q10 - q, q90 + q]

The calibration slice is always the tail of the fold's training window, so no
out-of-sample row is ever used to fit the interval. Coverage is measured, not
assumed, and reported even when it misses the 80% target.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any

import numpy as np
import pandas as pd

from vnindex_forecast.config import AppConfig
from vnindex_forecast.features.build import feature_columns
from vnindex_forecast.models.factory import build_quantile_regressor

QUANTILES: tuple[float, float, float] = (0.1, 0.5, 0.9)
INTERVAL_COLUMNS = [
    "date",
    "actual",
    "q10",
    "q50",
    "q90",
    "lower",
    "upper",
    "covered",
    "width",
]


def pinball_loss(actual: np.ndarray, prediction: np.ndarray, alpha: float) -> float:
    """Mean pinball (quantile) loss for one quantile level."""
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    delta = actual - prediction
    return float(np.mean(np.maximum(alpha * delta, (alpha - 1.0) * delta)))


def fit_quantile_models(
    config: AppConfig, feature_names: list[str], train: pd.DataFrame
) -> dict[float, Any]:
    """Fit one LightGBM per quantile level on a training slice."""
    target = train["next_return"].astype(float)
    models: dict[float, Any] = {}
    for alpha in QUANTILES:
        model = build_quantile_regressor(config, feature_names, alpha)
        model.fit(train[feature_names], target)
        models[alpha] = model
    return models


def _predict(
    models: dict[float, Any], frame: pd.DataFrame, feature_names: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q10 = np.asarray(models[0.1].predict(frame[feature_names]), dtype=float)
    q50 = np.asarray(models[0.5].predict(frame[feature_names]), dtype=float)
    q90 = np.asarray(models[0.9].predict(frame[feature_names]), dtype=float)
    return q10, q50, q90


def conformal_adjustment(
    models: dict[float, Any],
    calibration: pd.DataFrame,
    feature_names: list[str],
    coverage: float,
) -> float:
    """Quantile of the conformity scores on a held-out calibration slice."""
    q10, _, q90 = _predict(models, calibration, feature_names)
    actual = calibration["next_return"].to_numpy(dtype=float)
    scores = np.maximum(q10 - actual, actual - q90)
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return 0.0
    return float(max(0.0, np.quantile(scores, coverage, method="higher")))


@dataclasses.dataclass
class ConformalInterval:
    """Served interval: quantile models plus their conformal widening."""

    coverage: float
    adjustment: float
    calibration_rows: int
    models: dict[float, Any]

    def predict(self, frame: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
        q10, q50, q90 = _predict(self.models, frame, feature_names)
        lower = q10 - self.adjustment
        upper = q90 + self.adjustment
        return pd.DataFrame(
            {
                "q10": q10,
                "q50": q50,
                "q90": q90,
                "lower": lower,
                "upper": upper,
            },
            index=frame.index,
        )


@dataclasses.dataclass
class ConformalEvaluation:
    frame: pd.DataFrame
    folds: int
    coverage: float
    target_coverage: float
    average_width: float
    pinball: dict[str, float]
    elapsed_seconds: float = 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "rows": int(len(self.frame)),
            "folds": int(self.folds),
            "coverage": round(self.coverage, 4),
            "target_coverage": self.target_coverage,
            "average_width": round(self.average_width, 6),
            "pinball_loss": {key: round(value, 8) for key, value in self.pinball.items()},
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }


def _evaluate_frame(frame: pd.DataFrame, folds: int, coverage: float) -> ConformalEvaluation:
    actual = frame["actual"].to_numpy(dtype=float)
    covered = frame["covered"].to_numpy(dtype=bool)
    widths = frame["width"].to_numpy(dtype=float)
    pinball = {
        str(alpha): pinball_loss(actual, frame[f"q{int(alpha * 100)}"].to_numpy(), alpha)
        for alpha in QUANTILES
    }
    return ConformalEvaluation(
        frame=frame,
        folds=folds,
        coverage=float(covered.mean()) if len(covered) else 0.0,
        target_coverage=coverage,
        average_width=float(widths.mean()) if len(widths) else 0.0,
        pinball=pinball,
    )


def summarize_intervals(
    frame: pd.DataFrame, folds: int, coverage: float, elapsed_seconds: float = 0.0
) -> ConformalEvaluation:
    """Rebuild the aggregate metrics from an interval frame (used by caches)."""
    evaluation = _evaluate_frame(frame, folds, coverage)
    evaluation.elapsed_seconds = elapsed_seconds
    return evaluation


def _calibration_rows(config: AppConfig, train_rows: int) -> int:
    return min(
        config.model.conformal.calibration_rows,
        max(20, train_rows // 3),
    )


def run_conformal_walk_forward(
    features: pd.DataFrame, config: AppConfig
) -> ConformalEvaluation:
    """Conformal intervals for the same 83 walk-forward points as the classifier."""
    names = feature_columns(config)
    frame = features.sort_values("date").reset_index(drop=True)
    frame = frame.dropna(subset=names + ["next_return"]).reset_index(drop=True)

    settings = config.model.walk_forward
    minimum = settings.min_train_rows
    if len(frame) <= minimum + 1:
        raise ValueError(
            f"conformal walk-forward needs more than {minimum + 1} rows, got {len(frame)}"
        )

    coverage = float(config.model.conformal.alpha)
    rows: list[dict[str, Any]] = []
    folds = 0
    cursor = minimum
    started = time.perf_counter()

    while cursor < len(frame):
        stop = min(cursor + settings.refit_every, len(frame))
        train = frame.iloc[:cursor]
        test = frame.iloc[cursor:stop]
        if len(test) == 0:
            break

        calibration_rows = _calibration_rows(config, len(train))
        fit_part = train.iloc[:-calibration_rows]
        calibration = train.iloc[-calibration_rows:]
        models = fit_quantile_models(config, names, fit_part)
        adjustment = conformal_adjustment(models, calibration, names, coverage)
        q10, q50, q90 = _predict(models, test, names)
        actual = test["next_return"].to_numpy(dtype=float)
        lower = q10 - adjustment
        upper = q90 + adjustment

        for offset, (_, row) in enumerate(test.iterrows()):
            rows.append(
                {
                    "date": row["date"],
                    "actual": float(actual[offset]),
                    "q10": float(q10[offset]),
                    "q50": float(q50[offset]),
                    "q90": float(q90[offset]),
                    "lower": float(lower[offset]),
                    "upper": float(upper[offset]),
                    "covered": bool(lower[offset] <= actual[offset] <= upper[offset]),
                    "width": float(upper[offset] - lower[offset]),
                }
            )

        folds += 1
        cursor = stop

    result = pd.DataFrame(rows, columns=INTERVAL_COLUMNS)
    result["date"] = pd.to_datetime(result["date"])
    evaluation = _evaluate_frame(result, folds, coverage)
    evaluation.elapsed_seconds = time.perf_counter() - started
    return evaluation


def holdout_conformal(
    features: pd.DataFrame, config: AppConfig
) -> ConformalEvaluation | None:
    """Coverage on the time-ordered holdout, never used by the backtest."""
    names = feature_columns(config)
    frame = features.sort_values("date").reset_index(drop=True)
    frame = frame.dropna(subset=names + ["next_return"]).reset_index(drop=True)
    holdout = config.model.holdout_rows
    calibration_rows = config.model.conformal.calibration_rows
    if len(frame) < holdout + calibration_rows + 60:
        return None

    train = frame.iloc[: -(holdout + calibration_rows)]
    calibration = frame.iloc[-(holdout + calibration_rows) : -holdout]
    test = frame.iloc[-holdout:]
    coverage = float(config.model.conformal.alpha)
    models = fit_quantile_models(config, names, train)
    adjustment = conformal_adjustment(models, calibration, names, coverage)
    q10, q50, q90 = _predict(models, test, names)
    actual = test["next_return"].to_numpy(dtype=float)
    lower = q10 - adjustment
    upper = q90 + adjustment
    result = pd.DataFrame(
        {
            "date": test["date"].to_numpy(),
            "actual": actual,
            "q10": q10,
            "q50": q50,
            "q90": q90,
            "lower": lower,
            "upper": upper,
            "covered": (lower <= actual) & (actual <= upper),
            "width": upper - lower,
        }
    )
    return _evaluate_frame(result, 1, coverage)


def fit_served_interval(
    features: pd.DataFrame, config: AppConfig
) -> ConformalInterval:
    """Fit the interval that will be stored in the artifact and served.

    The models are fitted on every labelled row except the last
    ``calibration_rows`` sessions, which calibrate the widening.
    """
    names = feature_columns(config)
    frame = features.sort_values("date").reset_index(drop=True)
    frame = frame.dropna(subset=names + ["next_return"]).reset_index(drop=True)
    calibration_rows = _calibration_rows(config, len(frame))
    train = frame.iloc[:-calibration_rows]
    calibration = frame.iloc[-calibration_rows:]
    coverage = float(config.model.conformal.alpha)
    models = fit_quantile_models(config, names, train)
    adjustment = conformal_adjustment(models, calibration, names, coverage)
    return ConformalInterval(
        coverage=coverage,
        adjustment=adjustment,
        calibration_rows=int(calibration_rows),
        models=models,
    )
