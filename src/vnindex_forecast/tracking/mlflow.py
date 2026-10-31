"""MLflow tracking and registry, optional and fail-soft.

The local JSON registry in :mod:`vnindex_forecast.models.registry` remains the
source of truth for serving: the service must work offline. MLflow is enabled
by ``train --track`` and stores runs in ``mlruns/`` (SQLite backend so the model
registry works), logging:

* parameters from the typed config,
* out-of-sample metrics,
* the equity curve, calibration plot and metrics JSON,
* the fitted sklearn pipeline as ``model`` in the ``vnindex_direction`` registry.

The ``@champion`` alias follows the highest walk-forward **accuracy** among
registered versions. Accuracy is chosen over Sharpe because the strategy return
on 83 noisy points is unstable; the README states the criterion explicitly.
Any registry or network failure is reported as a warning and the pipeline
continues with the local artifact.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Any

from vnindex_forecast.config import REPO_ROOT, AppConfig
from vnindex_forecast.models import registry

# MLflow 3 serialises sklearn pipelines with skops, which refuses unknown types
# by default. These are the exact classes produced by our ColumnTransformer +
# LightGBM pipeline; nothing else is trusted.
SKOPS_TRUSTED_TYPES = [
    "collections.OrderedDict",
    "lightgbm.basic.Booster",
    "lightgbm.sklearn.LGBMClassifier",
    "sklearn.compose._column_transformer._RemainderColsList",
]


def available() -> bool:
    return importlib.util.find_spec("mlflow") is not None


def default_tracking_uri() -> str:
    configured = os.environ.get("MLFLOW_TRACKING_URI")
    if configured:
        return configured
    root = REPO_ROOT / "mlruns"
    root.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(root / 'mlflow.db').as_posix()}"


def _params(config: AppConfig) -> dict[str, Any]:
    model = config.model
    walk = model.walk_forward
    return {
        "seed": config.seed,
        "model_name": model.name,
        "n_estimators": model.n_estimators,
        "regressor_n_estimators": model.regressor_n_estimators,
        "learning_rate": model.learning_rate,
        "num_leaves": model.num_leaves,
        "min_child_samples": model.min_child_samples,
        "subsample": model.subsample,
        "colsample_bytree": model.colsample_bytree,
        "reg_lambda": model.reg_lambda,
        "cost_bps": model.cost_bps,
        "min_train_rows": walk.min_train_rows,
        "refit_every": walk.refit_every,
        "inner_validation_rows": walk.inner_validation_rows,
        "conformal_alpha": model.conformal.alpha,
        "chronos_model_id": config.chronos.model_id,
        "config_hash": config.hash(),
    }


def _metrics(metrics: dict[str, Any]) -> dict[str, float]:
    direction = metrics.get("direction", {})
    strategy = metrics.get("strategy", {})
    buy_hold = metrics.get("buy_and_hold", {})
    conformal = metrics.get("conformal", {}).get("walk_forward", {})
    values = {
        "accuracy": direction.get("accuracy"),
        "auc": direction.get("auc"),
        "brier": direction.get("brier"),
        "strategy_total_return": strategy.get("total_return"),
        "strategy_sharpe": strategy.get("annualized_sharpe"),
        "buy_hold_total_return": buy_hold.get("total_return"),
        "buy_hold_sharpe": buy_hold.get("annualized_sharpe"),
        "conformal_coverage_80": conformal.get("coverage"),
    }
    return {key: float(value) for key, value in values.items() if value is not None}


def track_training(
    config: AppConfig,
    *,
    artifact: registry.Artifact,
    metrics: dict[str, Any],
    feature_names: list[str],
    input_example: Any,
) -> dict[str, Any]:
    """Log one training run; never raise."""
    if not available():
        return {"status": "skipped", "reason": "mlflow is not installed"}

    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        uri = default_tracking_uri()
        mlflow.set_tracking_uri(uri)
        experiment = config.model.tracking.experiment
        mlflow.set_experiment(experiment)
        registered_name = config.model.tracking.registered_model_name
        accuracy = float(metrics["direction"]["accuracy"])

        with mlflow.start_run(run_name=f"{config.model.name}-{artifact.version}") as run:
            mlflow.log_params(_params(config))
            mlflow.log_metrics(_metrics(metrics))
            mlflow.set_tags(
                {
                    "git_commit": registry.git_commit(),
                    "artifact_version": artifact.version,
                    "data_sha256": artifact.metadata.get("data_sha256", "unknown"),
                    "config_path": str(config.config_path),
                }
            )

            reports_dir = config.path("reports_dir")
            figures_dir = config.path("figures_dir")
            for path in sorted(figures_dir.glob("*.png")):
                mlflow.log_artifact(str(path), artifact_path="figures")
            for name in ("metrics.json", "method_comparison.json", "confusion.json"):
                path = reports_dir / name
                if path.exists():
                    mlflow.log_artifact(str(path))

            info = mlflow.sklearn.log_model(
                artifact.payload["classifier"],
                name="model",
                registered_model_name=registered_name,
                input_example=input_example,
                skops_trusted_types=SKOPS_TRUSTED_TYPES,
            )
            version = getattr(info, "registered_model_version", None)
            alias = config.model.tracking.alias
            champion_before = None
            if version is not None:
                client = MlflowClient()
                client.set_model_version_tag(
                    registered_name, str(version), "walk_forward_accuracy", f"{accuracy:.6f}"
                )
                try:
                    champion_before = client.get_model_version_by_alias(registered_name, alias)
                except Exception:
                    champion_before = None
                previous = -1.0
                if champion_before is not None:
                    try:
                        previous = float(
                            champion_before.tags.get("walk_forward_accuracy", "-1")
                        )
                    except (TypeError, ValueError):
                        previous = -1.0
                if accuracy >= previous:
                    client.set_registered_model_alias(
                        registered_name, alias, str(version)
                    )

            return {
                "status": "ok",
                "tracking_uri": uri,
                "experiment": experiment,
                "run_id": run.info.run_id,
                "model_name": registered_name,
                "model_version": str(version) if version is not None else None,
                "alias": alias,
                "champion_before": (
                    champion_before.version if champion_before is not None else None
                ),
                "champion_criterion": "highest walk-forward direction accuracy",
                "feature_count": len(feature_names),
            }
    except Exception as exc:  # registry/network/backend problems must not kill training
        warning = f"mlflow tracking failed ({type(exc).__name__}: {exc}); local artifact kept"
        print(f"[mlflow] {warning}")
        return {"status": "failed", "reason": warning, "tracking_uri": default_tracking_uri()}
