"""Model factories.

Preprocessing lives inside an sklearn ``Pipeline`` so that it is fitted on
training folds only. The models consume a fixed, ordered column set selected by
a ``ColumnTransformer``; that also makes an out of order or extra column a hard
error rather than a silent reshape.
"""

from __future__ import annotations

from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from vnindex_forecast.config import AppConfig


def _preprocessor(feature_names: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [("numeric", "passthrough", feature_names)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_classifier(config: AppConfig, feature_names: list[str]) -> Pipeline:
    """LightGBM direction classifier wrapped with its column selection."""
    model = LGBMClassifier(
        n_estimators=config.model.n_estimators,
        learning_rate=config.model.learning_rate,
        num_leaves=config.model.num_leaves,
        min_child_samples=config.model.min_child_samples,
        subsample=config.model.subsample,
        subsample_freq=config.model.subsample_freq,
        colsample_bytree=config.model.colsample_bytree,
        reg_lambda=config.model.reg_lambda,
        max_depth=config.model.max_depth,
        n_jobs=config.model.n_jobs,
        random_state=config.seed,
        deterministic=True,
        force_col_wise=True,
        verbose=-1,
    )
    return Pipeline([("features", _preprocessor(feature_names)), ("model", model)])


def build_regressor(config: AppConfig, feature_names: list[str]) -> Pipeline:
    """LightGBM regressor for the next session log return."""
    model = LGBMRegressor(
        n_estimators=config.model.regressor_n_estimators,
        learning_rate=config.model.learning_rate,
        num_leaves=config.model.num_leaves,
        min_child_samples=config.model.min_child_samples,
        subsample=config.model.subsample,
        subsample_freq=config.model.subsample_freq,
        colsample_bytree=config.model.colsample_bytree,
        reg_lambda=config.model.reg_lambda,
        max_depth=config.model.max_depth,
        n_jobs=config.model.n_jobs,
        random_state=config.seed,
        deterministic=True,
        force_col_wise=True,
        verbose=-1,
    )
    return Pipeline([("features", _preprocessor(feature_names)), ("model", model)])


def build_quantile_regressor(
    config: AppConfig, feature_names: list[str], alpha: float
) -> Pipeline:
    """LightGBM pinball-loss regressor for one quantile level."""
    model = LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_estimators=config.model.regressor_n_estimators,
        learning_rate=config.model.learning_rate,
        num_leaves=config.model.num_leaves,
        min_child_samples=config.model.min_child_samples,
        subsample=config.model.subsample,
        subsample_freq=config.model.subsample_freq,
        colsample_bytree=config.model.colsample_bytree,
        reg_lambda=config.model.reg_lambda,
        max_depth=config.model.max_depth,
        n_jobs=config.model.n_jobs,
        random_state=config.seed,
        deterministic=True,
        force_col_wise=True,
        verbose=-1,
    )
    return Pipeline([("features", _preprocessor(feature_names)), ("model", model)])


def build_logistic_classifier(config: AppConfig, feature_names: list[str]) -> Pipeline:
    """Scaled logistic regression: the transparent linear reference model."""
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    max_iter=2000,
                    random_state=config.seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    return Pipeline([("features", _preprocessor(feature_names)), ("model", model)])
