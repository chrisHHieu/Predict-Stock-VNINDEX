"""Typed configuration tree loaded from YAML.

The tree is a set of pydantic models. Nothing outside this module is allowed to
read the YAML directly: every command receives an :class:`AppConfig` and asks it
for resolved paths, so the pipeline behaves the same from any working
directory.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str = "VNINDEX"
    source_url: str = "https://dchart-api.vndirect.com.vn/dchart/history"
    resolution: str = "D"
    start_date: str = "2017-01-01"
    end_date: str | None = None
    chunk_years: int = Field(default=5, ge=1, le=10)
    user_agent: str = "vnindex-forecast/1.0 (research; contact hoanghieu1642002@gmail.com)"
    timeout_seconds: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=3, ge=0, le=6)
    request_interval_seconds: float = Field(default=1.0, ge=0)
    min_rows: int = Field(default=1500, ge=1)
    max_calendar_gap_days: int = Field(default=10, ge=1)
    max_large_gaps_last_year: int = Field(default=6, ge=0)


class FeatureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    return_lags: list[int] = Field(default_factory=lambda: [1, 2, 3, 5, 10])
    rolling_windows: list[int] = Field(default_factory=lambda: [5, 20, 60])
    rsi_window: int = Field(default=14, ge=2)
    atr_window: int = Field(default=14, ge=1)
    volume_window: int = Field(default=20, ge=2)
    min_history: int = Field(default=60, ge=5)

    def validate_windows(self) -> None:
        if self.min_history < max(self.rolling_windows):
            raise ValueError("min_history must be at least the largest rolling window")
        if len(set(self.return_lags)) != len(self.return_lags):
            raise ValueError("return_lags contains duplicates")
        if sorted(self.rolling_windows) != self.rolling_windows:
            raise ValueError("rolling_windows must be sorted ascending")


class WalkForwardConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_train_rows: int = Field(default=500, ge=50)
    refit_every: int = Field(default=21, ge=1)
    inner_validation_rows: int = Field(default=126, ge=20)


class ConformalConfig(BaseModel):
    """Split-conformal settings for the LightGBM return interval."""

    model_config = ConfigDict(extra="forbid")

    alpha: float = Field(default=0.8, gt=0.5, lt=1.0)
    calibration_rows: int = Field(default=126, ge=20)


class TuningConfig(BaseModel):
    """Optuna search settings.

    Every validation block lives inside the *initial* training window
    (``walk_forward.min_train_rows``), so hyper-parameter search never sees the
    holdout or a walk-forward test row.
    """

    model_config = ConfigDict(extra="forbid")

    trials: int = Field(default=30, ge=1)
    blocks: int = Field(default=5, ge=1, le=10)
    block_rows: int = Field(default=40, ge=5)
    warmup_rows: int = Field(default=200, ge=10)
    sampler_seed: int = Field(default=42, ge=0)


class TrackingConfig(BaseModel):
    """MLflow experiment, registry name and champion alias."""

    model_config = ConfigDict(extra="forbid")

    experiment: str = "vnindex-direction"
    registered_model_name: str = "vnindex_direction"
    alias: str = "champion"


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "lightgbm"
    n_estimators: int = Field(default=300, ge=10)
    regressor_n_estimators: int = Field(default=200, ge=10)
    learning_rate: float = Field(default=0.03, gt=0)
    num_leaves: int = Field(default=15, ge=2)
    min_child_samples: int = Field(default=40, ge=1)
    subsample: float = Field(default=0.8, gt=0, le=1)
    subsample_freq: int = Field(default=1, ge=0)
    colsample_bytree: float = Field(default=0.8, gt=0, le=1)
    reg_lambda: float = Field(default=1.0, ge=0)
    max_depth: int = -1
    n_jobs: int = -1
    cost_bps: float = Field(default=10.0, ge=0)
    annualization: int = Field(default=252, ge=1)
    holdout_rows: int = Field(default=252, ge=20)
    threshold_grid: list[float] = Field(
        default_factory=lambda: [round(0.45 + 0.01 * i, 2) for i in range(16)]
    )
    walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)
    conformal: ConformalConfig = Field(default_factory=ConformalConfig)
    tuning: TuningConfig = Field(default_factory=TuningConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)


class ChronosConfig(BaseModel):
    """Zero-shot Chronos-2 settings. Weights are downloaded, never committed."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    model_id: str = "amazon/chronos-2"
    context_length: int = Field(default=512, ge=64)
    prediction_length: int = Field(default=1, ge=1)
    batch_size: int = Field(default=32, ge=1)
    device: str = "cpu"
    time_budget_seconds: float = Field(default=900.0, gt=0)


class MonitorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window_rows: int = Field(default=252, ge=20)
    reference_fraction: float = Field(default=0.7, gt=0, lt=1)
    psi_bins: int = Field(default=10, ge=2, le=20)
    psi_moderate: float = Field(default=0.10, gt=0)
    psi_significant: float = Field(default=0.25, gt=0)


class ServingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "127.0.0.1"
    port: int = Field(default=8123, ge=1, le=65535)


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    models_dir: str = "models"
    reports_dir: str = "reports"
    figures_dir: str = "reports/figures"


PathKey = str


class AppConfig(BaseModel):
    """Root configuration object passed to every pipeline stage."""

    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    monitor: MonitorConfig = Field(default_factory=MonitorConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)
    chronos: ChronosConfig = Field(default_factory=ChronosConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    config_path: str | None = None

    def path(self, key: PathKey) -> Path:
        """Resolve a configured directory against the repository root."""
        raw = getattr(self.paths, key)
        candidate = Path(raw)
        return candidate if candidate.is_absolute() else REPO_ROOT / candidate

    def as_dict(self) -> dict[str, Any]:
        data = self.model_dump(mode="json")
        data.pop("config_path", None)
        return data

    def hash(self) -> str:
        canonical = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def ensure_dirs(self) -> None:
        for key in ("raw_dir", "processed_dir", "models_dir", "reports_dir", "figures_dir"):
            self.path(key).mkdir(parents=True, exist_ok=True)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(
    path: str | Path | None = None, overrides: dict[str, Any] | None = None
) -> AppConfig:
    """Load the YAML config, apply nested overrides, and validate it."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"config file must contain a mapping: {config_path}")
    if overrides:
        payload = _deep_merge(payload, overrides)

    config = AppConfig(**payload)
    config.features.validate_windows()
    config.config_path = str(config_path)
    return config


def set_seed(seed: int) -> None:
    """Seed every random source used by the pipeline."""
    random.seed(seed)
    np.random.seed(seed)
