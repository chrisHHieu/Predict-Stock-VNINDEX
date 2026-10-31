"""Walk-forward evaluation, metrics and report writing.

The only validation scheme in this repository is walk-forward with an expanding
window and a refit every ``refit_every`` sessions. Nothing is shuffled, nothing
is fitted on data that is not in the past of the row being predicted.

Two baselines are always computed alongside the model:

* the majority direction observed in the training window,
* a naive forecast of exactly zero next-session return.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from vnindex_forecast.config import AppConfig  # noqa: E402
from vnindex_forecast.features.build import feature_columns  # noqa: E402
from vnindex_forecast.models import registry  # noqa: E402
from vnindex_forecast.models.chronos import (  # noqa: E402
    ChronosUnavailable,
    run_chronos_walk_forward,
)
from vnindex_forecast.models.conformal import (  # noqa: E402
    ConformalEvaluation,
    holdout_conformal,
    run_conformal_walk_forward,
    summarize_intervals,
)
from vnindex_forecast.models.factory import (  # noqa: E402
    build_classifier,
    build_logistic_classifier,
    build_regressor,
)

PREDICTION_COLUMNS = [
    "date",
    "direction",
    "next_return",
    "p_up",
    "pred_return",
    "threshold",
    "p_base",
    "majority_pred",
    "train_rows",
]


def _positive_proba(model: Any, features: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(features)
    if proba.ndim == 1 or proba.shape[1] == 1:
        only = float(model.classes_[0])
        return np.full(len(features), only)
    positive_index = list(model.classes_).index(1)
    return proba[:, positive_index]


def _net_returns(
    probabilities: np.ndarray,
    next_returns: np.ndarray,
    threshold: float,
    cost_bps: float,
) -> tuple[np.ndarray, np.ndarray]:
    position = (probabilities > threshold).astype(float)
    previous = np.concatenate([[0.0], position[:-1]])
    turnover = np.abs(position - previous)
    cost = turnover * (cost_bps / 10_000.0)
    return position * next_returns - cost, position


def choose_threshold(
    probabilities: np.ndarray,
    next_returns: np.ndarray,
    config: AppConfig,
) -> float:
    """Pick the strategy threshold maximising net return on a held-out slice.

    Only rows that were in the training part of a fold are ever passed here.
    """
    if len(probabilities) == 0:
        return 0.5
    best_threshold = 0.5
    best_score = -np.inf
    for threshold in config.model.threshold_grid:
        net, position = _net_returns(
            probabilities, next_returns, threshold, config.model.cost_bps
        )
        if position.sum() < max(5, int(0.05 * len(position))):
            continue
        score = float(np.expm1(net.sum()))
        if score > best_score + 1e-12:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


@dataclasses.dataclass
class WalkForwardResult:
    predictions: pd.DataFrame
    folds: int
    thresholds: list[float]
    elapsed_seconds: float = 0.0


def run_walk_forward(features: pd.DataFrame, config: AppConfig) -> WalkForwardResult:
    """Expanding-window walk-forward predictions for every out-of-sample row."""
    names = feature_columns(config)
    frame = features.sort_values("date").reset_index(drop=True)
    frame = frame.dropna(subset=names + ["direction", "next_return"]).reset_index(drop=True)

    settings = config.model.walk_forward
    minimum = settings.min_train_rows
    if len(frame) <= minimum + 1:
        raise ValueError(
            f"walk-forward needs more than {minimum + 1} rows, the frame has {len(frame)}"
        )

    rows: list[dict[str, Any]] = []
    thresholds: list[float] = []
    folds = 0
    cursor = minimum
    started = time.perf_counter()

    while cursor < len(frame):
        stop = min(cursor + settings.refit_every, len(frame))
        train = frame.iloc[:cursor]
        test = frame.iloc[cursor:stop]
        if len(test) == 0:
            break

        inner_rows = min(settings.inner_validation_rows, max(20, len(train) // 3))
        inner_train = train.iloc[:-inner_rows]
        inner_valid = train.iloc[-inner_rows:]

        fold_threshold = 0.5
        if len(inner_train) >= 30 and inner_valid["direction"].nunique() == 2:
            inner_model = build_classifier(config, names)
            inner_model.fit(inner_train[names], inner_train["direction"].astype(int))
            inner_probabilities = _positive_proba(inner_model, inner_valid[names])
            fold_threshold = choose_threshold(
                inner_probabilities, inner_valid["next_return"].to_numpy(), config
            )

        model = build_classifier(config, names)
        model.fit(train[names], train["direction"].astype(int))
        probabilities = _positive_proba(model, test[names])

        regressor = build_regressor(config, names)
        regressor.fit(train[names], train["next_return"].astype(float))
        predicted_return = regressor.predict(test[names])

        base_rate = float(train["direction"].mean())
        majority = int(base_rate > 0.5)

        for offset in range(len(test)):
            row = test.iloc[offset]
            rows.append(
                {
                    "date": row["date"],
                    "direction": int(row["direction"]),
                    "next_return": float(row["next_return"]),
                    "p_up": float(probabilities[offset]),
                    "pred_return": float(predicted_return[offset]),
                    "threshold": float(fold_threshold),
                    "p_base": base_rate,
                    "majority_pred": majority,
                    "train_rows": int(len(train)),
                }
            )

        thresholds.append(fold_threshold)
        folds += 1
        cursor = stop

    predictions = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
    predictions["date"] = pd.to_datetime(predictions["date"])
    return WalkForwardResult(
        predictions=predictions,
        folds=folds,
        thresholds=thresholds,
        elapsed_seconds=time.perf_counter() - started,
    )


def _equity_metrics(returns: np.ndarray, annualization: int) -> dict[str, float]:
    equity = np.exp(np.cumsum(returns))
    peak = np.maximum.accumulate(equity)
    drawdown = float((equity / peak - 1.0).min()) if len(equity) else 0.0
    total = float(np.expm1(returns.sum())) if len(returns) else 0.0
    std = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    sharpe = float(np.mean(returns) / std * np.sqrt(annualization)) if std > 0 else 0.0
    periods = max(len(returns), 1)
    annualized = (1.0 + total) ** (annualization / periods) - 1.0 if total > -1 else -1.0
    return {
        "total_return": round(total, 6),
        "annualized_return": round(float(annualized), 6),
        "annualized_sharpe": round(sharpe, 4),
        "max_drawdown": round(drawdown, 6),
    }


def compute_metrics(predictions: pd.DataFrame, config: AppConfig) -> dict[str, Any]:
    """All reported numbers, computed once from the out-of-sample predictions."""
    if predictions.empty:
        raise ValueError("no out-of-sample predictions to score")

    y = predictions["direction"].to_numpy(dtype=int)
    p = predictions["p_up"].to_numpy(dtype=float)
    base = predictions["p_base"].to_numpy(dtype=float)
    majority = predictions["majority_pred"].to_numpy(dtype=int)
    next_return = predictions["next_return"].to_numpy(dtype=float)
    predicted = predictions["pred_return"].to_numpy(dtype=float)
    threshold = predictions["threshold"].to_numpy(dtype=float)

    accuracy = float(np.mean((p > 0.5).astype(int) == y))
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    brier = float(np.mean((p - y) ** 2))
    constant_brier = float(np.mean((base - y) ** 2))
    majority_accuracy = float(np.mean(majority == y))

    mae_model = float(np.mean(np.abs(predicted - next_return)))
    mae_naive = float(np.mean(np.abs(next_return)))

    strategy_returns, position = _net_returns(p, next_return, threshold, config.model.cost_bps)
    fixed_returns, _ = _net_returns(p, next_return, 0.5, config.model.cost_bps)
    strategy = _equity_metrics(strategy_returns, config.model.annualization)
    fixed = _equity_metrics(fixed_returns, config.model.annualization)
    buy_hold = _equity_metrics(next_return, config.model.annualization)

    long_days = position > 0
    hit_rate = float(np.mean(next_return[long_days] > 0)) if long_days.any() else 0.0

    dates = pd.to_datetime(predictions["date"])
    return {
        "data": {
            "oos_rows": int(len(predictions)),
            "oos_start": dates.min().date().isoformat(),
            "oos_end": dates.max().date().isoformat(),
            "up_share": round(float(y.mean()), 4),
            "cost_bps": config.model.cost_bps,
        },
        "direction": {
            "accuracy": round(accuracy, 4),
            "auc": round(auc, 4) if np.isfinite(auc) else None,
            "brier": round(brier, 6),
            "base_rate": round(float(y.mean()), 4),
        },
        "baselines": {
            "majority_accuracy": round(majority_accuracy, 4),
            "constant_brier": round(constant_brier, 6),
            "naive_zero_mae": round(mae_naive, 6),
            "buy_and_hold_total_return": buy_hold["total_return"],
        },
        "return_regression": {
            "mae_model": round(mae_model, 6),
            "mae_naive_zero": round(mae_naive, 6),
            "rmse_model": round(float(np.sqrt(np.mean((predicted - next_return) ** 2))), 6),
        },
        "strategy": {
            **strategy,
            "hit_rate": round(hit_rate, 4),
            "exposure": round(float(position.mean()), 4),
            "n_trades": int(np.abs(np.diff(np.concatenate([[0.0], position]))).sum()),
        },
        "strategy_fixed_0_5": {
            **fixed,
            "exposure": round(float((p > 0.5).mean()), 4),
        },
        "buy_and_hold": {
            **buy_hold,
            "hit_rate": round(float(np.mean(next_return > 0)), 4),
        },
        "threshold": {
            "selected_median": round(float(np.median(threshold)), 4),
            "min": round(float(threshold.min()), 4),
            "max": round(float(threshold.max()), 4),
        },
    }


def _plot_equity(predictions: pd.DataFrame, config: AppConfig, path: Path) -> None:
    p = predictions["p_up"].to_numpy()
    r = predictions["next_return"].to_numpy()
    threshold = predictions["threshold"].to_numpy()
    strategy, _ = _net_returns(p, r, threshold, config.model.cost_bps)
    dates = pd.to_datetime(predictions["date"])

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=120)
    ax.plot(dates, np.exp(np.cumsum(strategy)), label="Model strategy (long when p > threshold)")
    ax.plot(dates, np.exp(np.cumsum(r)), label="Buy and hold")
    ax.set_ylabel("Growth of 1.0 (net of 10 bps per trade)")
    ax.set_title("Out-of-sample equity, walk-forward")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_calibration(predictions: pd.DataFrame, path: Path) -> None:
    p = predictions["p_up"].to_numpy()
    y = predictions["direction"].to_numpy()
    bins = np.linspace(0.0, 1.0, 11)
    index = np.clip(np.digitize(p, bins) - 1, 0, len(bins) - 2)
    means, observed, counts = [], [], []
    for bucket in range(len(bins) - 1):
        mask = index == bucket
        if mask.sum() >= 10:
            means.append(float(p[mask].mean()))
            observed.append(float(y[mask].mean()))
            counts.append(int(mask.sum()))

    fig, ax = plt.subplots(figsize=(4.8, 4.4), dpi=120)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1, label="Perfect calibration")
    if means:
        ax.plot(means, observed, marker="o", label="Model")
    ax.set_xlabel("Mean predicted probability of an up session")
    ax.set_ylabel("Observed share of up sessions")
    ax.set_title("Calibration, out-of-sample")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_importance(artifact: registry.Artifact, path: Path) -> None:
    pipeline = artifact.payload["classifier"]
    names = artifact.payload["feature_names"]
    importances = pipeline.named_steps["model"].feature_importances_.astype(float)
    order = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=120)
    ax.barh([names[i] for i in order], importances[order], color="#3b6fb6")
    ax.set_xlabel("Split gain importance")
    ax.set_title(f"LightGBM feature importance, {artifact.version}")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def compute_data_sha256(config: AppConfig) -> str:
    """SHA256 of the raw fetch payload when present, else of the processed CSV."""
    raw = config.path("raw_dir") / f"{config.data.symbol.lower()}_daily.json"
    if raw.exists():
        from vnindex_forecast.data.fetch import file_sha256

        return file_sha256(raw)
    processed = config.path("processed_dir") / f"{config.data.symbol.lower()}_daily.csv"
    if processed.exists():
        from vnindex_forecast.data.fetch import file_sha256

        return file_sha256(processed)
    return "unknown"


def predictions_cache_path(config: AppConfig) -> Path:
    return config.path("reports_dir") / "walk_forward_predictions.csv"


def predictions_meta_path(config: AppConfig) -> Path:
    return config.path("reports_dir") / "walk_forward_predictions.meta.json"


def write_predictions(config: AppConfig, result: WalkForwardResult, data_sha256: str) -> None:
    path = predictions_cache_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = result.predictions.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    frame.to_csv(path, index=False, float_format="%.8f")
    meta = {
        "data_sha256": data_sha256,
        "config_hash": config.hash(),
        "folds": result.folds,
        "rows": int(len(frame)),
        "elapsed_seconds": round(result.elapsed_seconds, 3),
        "written_at": registry.utc_now(),
    }
    predictions_meta_path(config).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_cached_predictions(
    config: AppConfig, data_sha256: str
) -> WalkForwardResult | None:
    """Reuse predictions when they were produced from the same data and config."""
    path = predictions_cache_path(config)
    meta_path = predictions_meta_path(config)
    if not path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if meta.get("data_sha256") != data_sha256 or meta.get("config_hash") != config.hash():
        return None
    frame = pd.read_csv(path, parse_dates=["date"])
    thresholds = frame["threshold"].drop_duplicates().tolist()
    return WalkForwardResult(
        predictions=frame,
        folds=int(meta.get("folds", len(thresholds))),
        thresholds=[float(value) for value in thresholds],
        elapsed_seconds=float(meta.get("elapsed_seconds", 0.0)),
    )


def run_logistic_walk_forward(
    features: pd.DataFrame, config: AppConfig
) -> tuple[pd.DataFrame, float]:
    """The same expanding-window folds, with a scaled logistic regression."""
    names = feature_columns(config)
    frame = features.sort_values("date").reset_index(drop=True)
    frame = frame.dropna(subset=names + ["direction", "next_return"]).reset_index(drop=True)
    settings = config.model.walk_forward
    cursor = settings.min_train_rows
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    while cursor < len(frame):
        stop = min(cursor + settings.refit_every, len(frame))
        train = frame.iloc[:cursor]
        test = frame.iloc[cursor:stop]
        if train["direction"].nunique() < 2:
            probabilities = np.full(len(test), float(train["direction"].mean()))
        else:
            model = build_logistic_classifier(config, names)
            model.fit(train[names], train["direction"].astype(int))
            probabilities = _positive_proba(model, test[names])
        for offset in range(len(test)):
            rows.append({"date": test.iloc[offset]["date"], "p_up": float(probabilities[offset])})
        cursor = stop

    return pd.DataFrame(rows), time.perf_counter() - started


def naive_probabilities(features: pd.DataFrame) -> pd.DataFrame:
    """Persistence: tomorrow repeats today's direction with probability 0 or 1."""
    frame = features.sort_values("date").reset_index(drop=True)
    frame = frame.dropna(subset=["ret_lag_1", "next_return"]).reset_index(drop=True)
    return pd.DataFrame(
        {
            "date": frame["date"],
            "p_up": (frame["ret_lag_1"] > 0).astype(float),
        }
    )


def _fixed_threshold_metrics(
    probabilities: np.ndarray, labels: pd.DataFrame, config: AppConfig
) -> dict[str, Any]:
    probabilities = np.asarray(probabilities, dtype=float)
    direction = labels["direction"].to_numpy(dtype=int)
    next_return = labels["next_return"].to_numpy(dtype=float)
    accuracy = float(np.mean((probabilities > 0.5).astype(int) == direction))
    auc = (
        float(roc_auc_score(direction, probabilities))
        if len(np.unique(direction)) == 2
        else float("nan")
    )
    brier = float(np.mean((probabilities - direction) ** 2))
    net, position = _net_returns(probabilities, next_return, 0.5, config.model.cost_bps)
    equity = _equity_metrics(net, config.model.annualization)
    previous = np.concatenate([[0.0], position[:-1]])
    trades = int(np.abs(position - previous).sum())
    return {
        "accuracy": round(accuracy, 4),
        "auc": round(auc, 4) if np.isfinite(auc) else None,
        "brier": round(brier, 6),
        **equity,
        "exposure": round(float(position.mean()), 4),
        "n_trades": trades,
    }


def _aligned(labels: pd.DataFrame, frame: pd.DataFrame, column: str = "p_up") -> np.ndarray:
    merged = labels[["date"]].merge(frame[["date", column]], on="date", how="left")
    if merged[column].isna().any():
        raise ValueError("method predictions do not cover every walk-forward date")
    return merged[column].to_numpy(dtype=float)


def _conformal_cache_paths(config: AppConfig) -> tuple[Path, Path]:
    reports = config.path("reports_dir")
    return reports / "conformal_walk_forward.csv", reports / "conformal_walk_forward.meta.json"


def write_conformal_cache(
    config: AppConfig, evaluation: ConformalEvaluation, data_sha256: str
) -> None:
    path, meta_path = _conformal_cache_paths(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = evaluation.frame.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    frame.to_csv(path, index=False, float_format="%.10f")
    meta = {
        "data_sha256": data_sha256,
        "config_hash": config.hash(),
        "folds": evaluation.folds,
        "coverage_target": evaluation.target_coverage,
        "elapsed_seconds": evaluation.elapsed_seconds,
        "written_at": registry.utc_now(),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_cached_conformal(
    config: AppConfig, data_sha256: str
) -> ConformalEvaluation | None:
    path, meta_path = _conformal_cache_paths(config)
    if not path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if (
        meta.get("data_sha256") != data_sha256
        or meta.get("config_hash") != config.hash()
    ):
        return None
    frame = pd.read_csv(path, parse_dates=["date"])
    return summarize_intervals(
        frame,
        int(meta.get("folds", 0)),
        float(meta.get("coverage_target", config.model.conformal.alpha)),
        float(meta.get("elapsed_seconds", 0.0)),
    )


def _chronos_cache_paths(config: AppConfig) -> tuple[Path, Path]:
    reports = config.path("reports_dir")
    return reports / "chronos_walk_forward.csv", reports / "chronos_walk_forward.meta.json"


def _chronos_key(config: AppConfig, data_sha256: str) -> dict[str, Any]:
    return {
        "data_sha256": data_sha256,
        "model_id": config.chronos.model_id,
        "context_length": config.chronos.context_length,
        "prediction_length": config.chronos.prediction_length,
    }


def _load_chronos_cache(
    config: AppConfig, data_sha256: str
) -> tuple[dict[str, Any], pd.DataFrame] | None:
    path, meta_path = _chronos_cache_paths(config)
    if not path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if meta.get("key") != _chronos_key(config, data_sha256):
        return None
    return meta["summary"], pd.read_csv(path, parse_dates=["date"])


def _write_chronos_cache(
    config: AppConfig, data_sha256: str, summary: dict[str, Any], frame: pd.DataFrame
) -> None:
    path, meta_path = _chronos_cache_paths(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    stored = frame.copy()
    stored["date"] = pd.to_datetime(stored["date"]).dt.strftime("%Y-%m-%d")
    stored.to_csv(path, index=False, float_format="%.10f")
    meta_path.write_text(
        json.dumps(
            {"key": _chronos_key(config, data_sha256), "summary": summary,
             "written_at": registry.utc_now()},
            indent=2,
        ),
        encoding="utf-8",
    )


def _run_chronos(
    config: AppConfig, features: pd.DataFrame, data_sha256: str, reuse: bool
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    if not config.chronos.enabled:
        return {"status": "skipped", "reason": "chronos.enabled is false"}, None
    if reuse:
        cached = _load_chronos_cache(config, data_sha256)
        if cached is not None:
            return cached

    prices_path = config.path("processed_dir") / f"{config.data.symbol.lower()}_daily.csv"
    if not prices_path.exists():
        return {
            "status": "skipped",
            "reason": f"price frame not found: {prices_path}",
        }, None
    prices = pd.read_csv(prices_path, parse_dates=["date"])
    try:
        result = run_chronos_walk_forward(features, prices, config)
    except (ChronosUnavailable, TimeoutError) as exc:
        return {"status": "skipped", "reason": str(exc)}, None
    except Exception as exc:  # unexpected backend errors must not hide the rest
        return {
            "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
        }, None

    summary = result.summary()
    _write_chronos_cache(config, data_sha256, summary, result.frame)
    return summary, result.frame


def run_method_comparison(
    config: AppConfig,
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    data_sha256: str,
    reuse: bool = True,
) -> dict[str, Any]:
    """Four methods on the same out-of-sample dates, threshold 0.5, 10 bps."""
    labels = predictions.sort_values("date").reset_index(drop=True)
    labels["date"] = pd.to_datetime(labels["date"])
    names = feature_columns(config)
    methods: list[dict[str, Any]] = []

    naive = naive_probabilities(features)
    naive_probabilities_array = _aligned(labels, naive)
    methods.append(
        {
            "method": "naive_persistence",
            "coverage_80": None,
            "seconds": 0.0,
            **_fixed_threshold_metrics(naive_probabilities_array, labels, config),
        }
    )

    logistic, logistic_seconds = run_logistic_walk_forward(features, config)
    logistic_probabilities = _aligned(labels, logistic)
    methods.append(
        {
            "method": "logistic",
            "coverage_80": None,
            "seconds": round(logistic_seconds, 3),
            **_fixed_threshold_metrics(logistic_probabilities, labels, config),
        }
    )

    lightgbm = _fixed_threshold_metrics(labels["p_up"].to_numpy(), labels, config)
    methods.append(
        {
            "method": "lightgbm",
            "model_detail": f"{config.model.n_estimators} trees, {len(names)} features",
            "coverage_80": None,
            "seconds": round(float(getattr(predictions, "attrs", {}).get("elapsed", 0.0)), 3),
            **lightgbm,
        }
    )

    conformal: ConformalEvaluation | None = load_cached_conformal(config, data_sha256) if reuse else None
    if conformal is None:
        conformal = run_conformal_walk_forward(features, config)
        write_conformal_cache(config, conformal, data_sha256)
    lightgbm_coverage = float(
        labels[["date"]]
        .merge(
            conformal.frame[["date", "covered"]],
            on="date",
            how="left",
        )["covered"]
        .mean()
    )
    methods[2]["coverage_80"] = round(lightgbm_coverage, 4)

    holdout = holdout_conformal(features, config)
    conformal_block = {
        "walk_forward": conformal.summary(),
        "holdout": holdout.summary() if holdout is not None else None,
    }

    chronos_summary, chronos_frame = _run_chronos(config, features, data_sha256, reuse)
    if chronos_frame is not None and len(chronos_frame):
        chronos_probabilities = _aligned(
            labels, chronos_frame.rename(columns={"p_up": "p_up"})
        )
        chronos_metrics = _fixed_threshold_metrics(chronos_probabilities, labels, config)
        methods.append(
            {
                "method": "chronos2_zero_shot",
                "model_detail": f"{config.chronos.model_id}, {config.chronos.context_length}-bar context",
                "coverage_80": round(float(chronos_frame["covered"].mean()), 4),
                "seconds": chronos_summary.get("inference_seconds"),
                **chronos_metrics,
            }
        )
    else:
        methods.append(
            {
                "method": "chronos2_zero_shot",
                "status": chronos_summary.get("status"),
                "reason": chronos_summary.get("reason"),
            }
        )

    return {
        "protocol": {
            "oos_rows": int(len(labels)),
            "oos_start": pd.Timestamp(labels["date"].min()).date().isoformat(),
            "oos_end": pd.Timestamp(labels["date"].max()).date().isoformat(),
            "threshold": 0.5,
            "cost_bps": config.model.cost_bps,
            "generated_at": registry.utc_now(),
            "note": (
                "All methods use the same out-of-sample dates, a fixed 0.5 threshold "
                "and 10 bps per position change. The LightGBM row is the "
                "walk-forward classifier; its per-fold threshold path is reported "
                "separately in the strategy block."
            ),
        },
        "methods": methods,
        "conformal": conformal_block,
        "chronos": chronos_summary,
    }


def format_comparison(comparison: dict[str, Any]) -> str:
    width = 22
    header = (
        f"{'method':<{width}}{'accuracy':>10}{'AUC':>8}{'Sharpe':>9}"
        f"{'net ret':>10}{'cov80':>8}{'seconds':>9}"
    )
    lines = [header]
    for item in comparison["methods"]:
        if "accuracy" not in item:
            lines.append(f"{item['method']:<{width}}{'skipped':>10}  {item.get('reason', '')}")
            continue
        auc = f"{item['auc']:.4f}" if item["auc"] is not None else "n/a"
        coverage = (
            f"{item['coverage_80']:.4f}" if item.get("coverage_80") is not None else "n/a"
        )
        seconds = item.get("seconds")
        seconds_text = f"{seconds:.2f}" if isinstance(seconds, (int, float)) else "n/a"
        lines.append(
            f"{item['method']:<{width}}{item['accuracy']:>10.4f}{auc:>8}"
            f"{item['annualized_sharpe']:>9.3f}{item['total_return']:>10.4f}"
            f"{coverage:>8}{seconds_text:>9}"
        )
    return "\n".join(lines)


def evaluate(config: AppConfig, reuse_predictions: bool = True) -> dict[str, Any]:
    """Score the latest artifact and write reports/metrics.json plus figures."""
    artifact = registry.load_artifact(config)
    features_path = config.path("processed_dir") / "features.csv"
    if not features_path.exists():
        raise FileNotFoundError(f"features not found: {features_path}. Run `features` first.")
    features = pd.read_csv(features_path, parse_dates=["date"])
    data_sha256 = artifact.metadata.get("data_sha256") or compute_data_sha256(config)

    result = load_cached_predictions(config, data_sha256) if reuse_predictions else None
    reused = result is not None
    if result is None:
        result = run_walk_forward(features, config)
        write_predictions(config, result, data_sha256)
    result.predictions.attrs["elapsed"] = result.elapsed_seconds

    metrics = compute_metrics(result.predictions, config)
    metrics["model_version"] = artifact.version
    metrics["generated_at"] = registry.utc_now()
    metrics["config_hash"] = config.hash()
    metrics["walk_forward"] = {
        "folds": result.folds,
        "min_train_rows": config.model.walk_forward.min_train_rows,
        "refit_every": config.model.walk_forward.refit_every,
        "threshold_grid": [
            round(float(result.thresholds[0]), 4),
            round(float(np.median(result.thresholds)), 4),
            round(float(result.thresholds[-1]), 4),
        ],
        "predictions_reused": reused,
        "elapsed_seconds": round(result.elapsed_seconds, 3),
    }
    metrics["served_threshold"] = artifact.threshold

    comparison = run_method_comparison(
        config, features, result.predictions, data_sha256, reuse=reuse_predictions
    )
    metrics["method_comparison"] = comparison["methods"]
    metrics["method_protocol"] = comparison["protocol"]
    metrics["conformal"] = comparison["conformal"]
    metrics["chronos"] = comparison["chronos"]

    reports_dir = config.path("reports_dir")
    figures_dir = config.path("figures_dir")
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (reports_dir / "method_comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    direction = result.predictions["direction"].to_numpy(dtype=int)
    probability = result.predictions["p_up"].to_numpy(dtype=float)
    confusion = {}
    for name, threshold in (("fixed_0_5", 0.5), ("served", artifact.threshold)):
        predicted = (probability > threshold).astype(int)
        confusion[name] = {
            "threshold": threshold,
            "true_positive": int(((predicted == 1) & (direction == 1)).sum()),
            "false_positive": int(((predicted == 1) & (direction == 0)).sum()),
            "true_negative": int(((predicted == 0) & (direction == 0)).sum()),
            "false_negative": int(((predicted == 0) & (direction == 1)).sum()),
        }
    (reports_dir / "confusion.json").write_text(
        json.dumps(confusion, indent=2), encoding="utf-8"
    )
    _plot_equity(result.predictions, config, figures_dir / "equity_curve.png")
    _plot_calibration(result.predictions, figures_dir / "calibration.png")
    _plot_importance(artifact, figures_dir / "feature_importance.png")

    metrics["_figure_dir"] = str(figures_dir)
    return metrics


def format_metrics(metrics: dict[str, Any]) -> str:
    direction = metrics["direction"]
    regression = metrics["return_regression"]
    baselines = metrics["baselines"]
    strategy = metrics["strategy"]
    buy_hold = metrics["buy_and_hold"]
    walk_forward = metrics.get("walk_forward", {})
    lines = [
        f"model version      {metrics['model_version']}",
        f"out-of-sample      {metrics['data']['oos_start']} -> {metrics['data']['oos_end']} "
        f"({metrics['data']['oos_rows']} sessions, {walk_forward.get('folds', 0)} refits)",
        "",
        f"{'metric':<26}{'model':>12}{'baseline':>14}",
        f"{'direction accuracy':<26}{direction['accuracy']:>12.4f}"
        f"{baselines['majority_accuracy']:>14.4f}",
        f"{'direction AUC':<26}{(direction['auc'] or float('nan')):>12.4f}{0.5:>14.4f}",
        f"{'direction Brier':<26}{direction['brier']:>12.4f}"
        f"{baselines['constant_brier']:>14.4f}",
        f"{'return MAE':<26}{regression['mae_model']:>12.6f}"
        f"{baselines['naive_zero_mae']:>14.6f}",
        "",
        f"{'strategy':<26}{'model':>12}{'buy and hold':>14}",
        f"{'total return':<26}{strategy['total_return']:>12.4f}"
        f"{buy_hold['total_return']:>14.4f}",
        f"{'annualized Sharpe':<26}{strategy['annualized_sharpe']:>12.3f}"
        f"{buy_hold['annualized_sharpe']:>14.3f}",
        f"{'max drawdown':<26}{strategy['max_drawdown']:>12.4f}"
        f"{buy_hold['max_drawdown']:>14.4f}",
        f"{'hit rate (long days)':<26}{strategy['hit_rate']:>12.4f}"
        f"{buy_hold['hit_rate']:>14.4f}",
    ]
    return "\n".join(lines)
