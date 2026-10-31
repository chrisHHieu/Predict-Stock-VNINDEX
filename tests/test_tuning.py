"""Three-trial Optuna smoke test with the search confined to the train window."""

from __future__ import annotations

import pytest
import yaml
from conftest import make_config, make_prices, materialise

pytest.importorskip("optuna")

from vnindex_forecast.config import load_config  # noqa: E402
from vnindex_forecast.tuning.search import tune, validation_windows  # noqa: E402


def test_search_windows_stay_before_walk_forward(tmp_path):
    config = make_config(
        tmp_path,
        model={
            "walk_forward": {"min_train_rows": 80, "refit_every": 21,
                             "inner_validation_rows": 30},
            "tuning": {"warmup_rows": 20, "block_rows": 12, "blocks": 3},
        },
    )
    prices = make_prices()
    features = materialise(config, prices)
    windows = validation_windows(features, config)
    assert len(windows) == 3
    for train, valid in windows:
        assert len(train) >= 30
        assert valid["date"].max() <= features.sort_values("date")["date"].iloc[80]


def test_tune_writes_a_loadable_config(tmp_path):
    config = make_config(
        tmp_path,
        model={
            "walk_forward": {"min_train_rows": 80, "refit_every": 21,
                             "inner_validation_rows": 30},
            "tuning": {"warmup_rows": 20, "block_rows": 12, "blocks": 3, "trials": 3},
        },
    )
    materialise(config, make_prices())
    output = tmp_path / "tuned.yaml"
    result = tune(config, trials=3, output=output)

    assert result.trials == 3
    assert result.best_value > 0
    assert output.exists()
    payload = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert 7 <= payload["model"]["num_leaves"] <= 63
    assert 0.01 <= payload["model"]["learning_rate"] <= 0.20
    reloaded = load_config(output)
    assert reloaded.model.num_leaves == payload["model"]["num_leaves"]
    assert (config.path("reports_dir") / "optuna.db").exists()
    assert len(result.windows) == 3
