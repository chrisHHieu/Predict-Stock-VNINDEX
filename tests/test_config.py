"""Configuration loading, overrides and hashing."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from vnindex_forecast.config import REPO_ROOT, load_config


def test_defaults_load_from_yaml():
    config = load_config()
    assert config.seed == 42
    assert config.data.symbol == "VNINDEX"
    assert config.model.walk_forward.refit_every == 21
    assert config.model.walk_forward.min_train_rows == 500
    assert config.data.user_agent.startswith("vnindex-forecast/1.0")
    assert config.config_path == str(REPO_ROOT / "configs" / "base.yaml")


def test_relative_paths_resolve_against_repo_root():
    config = load_config()
    assert config.path("models_dir") == REPO_ROOT / "models"
    assert config.path("models_dir").is_absolute()


def test_deep_merge_keeps_sibling_values():
    config = load_config(overrides={"model": {"n_estimators": 12}})
    assert config.model.n_estimators == 12
    assert config.model.learning_rate == 0.03
    assert config.model.walk_forward.refit_every == 21


def test_unknown_key_is_rejected():
    with pytest.raises(ValidationError):
        load_config(overrides={"model": {"not_a_parameter": 1}})


def test_hash_is_stable_and_sensitive():
    first = load_config()
    second = load_config()
    assert first.hash() == second.hash()
    assert len(first.hash()) == 64
    assert load_config(overrides={"seed": 7}).hash() != first.hash()


def test_min_history_must_cover_rolling_windows():
    with pytest.raises(ValueError):
        load_config(overrides={"features": {"min_history": 10}})


def test_absolute_path_is_preserved(tmp_path):
    config = load_config(overrides={"paths": {"reports_dir": str(tmp_path / "reports")}})
    assert config.path("reports_dir") == tmp_path / "reports"
