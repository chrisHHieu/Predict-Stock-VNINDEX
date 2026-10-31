"""Serving contract, against a model trained on the synthetic fixture."""

from __future__ import annotations

import pytest
from conftest import make_config, make_prices, materialise
from fastapi.testclient import TestClient

from vnindex_forecast.models.train import train
from vnindex_forecast.serving.app import create_app


@pytest.fixture(scope="module")
def service(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("serving")
    config = make_config(tmp)
    prices = make_prices()
    materialise(config, prices)
    result = train(config)
    with TestClient(create_app(config)) as client:
        yield client, config, result, prices


def _bars(prices, count: int = 60) -> list[dict]:
    window = prices.tail(count).copy()
    window["date"] = window["date"].dt.strftime("%Y-%m-%d")
    return window.to_dict("records")


def test_health_and_model_info(service):
    client, _, result, _ = service
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["model_version"] == result.artifact.version

    info = client.get("/model/info")
    assert info.status_code == 200
    body = info.json()
    assert body["version"] == result.artifact.version
    assert body["threshold"] == result.artifact.threshold
    assert body["feature_count"] == len(body["features"])
    assert "metrics" in body

    alias = client.get("/model-info")
    assert alias.status_code == 200
    assert alias.json()["version"] == body["version"]


def test_predict_returns_a_probability(service):
    client, _, result, prices = service
    response = client.post("/predict", json={"bars": _bars(prices)})
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["probability_up"] <= 1.0
    assert body["decision"] in {"long", "flat"}
    assert body["model_version"] == result.artifact.version
    assert body["threshold"] == result.artifact.threshold
    assert body["bars_used"] == 60
    assert isinstance(body["predicted_next_return"], float)
    assert body["p_up"] == body["probability_up"]
    assert body["interval_80"]["low"] < body["interval_80"]["high"]
    assert body["interval_80"]["coverage"] == 0.8
    import importlib.util

    if importlib.util.find_spec("shap") is None:
        # The core install has no SHAP; the endpoint still answers, without reasons.
        assert body["top_reasons"] == []
    else:
        assert len(body["top_reasons"]) == 3
        assert {"feature", "value", "shap", "effect"} <= set(body["top_reasons"][0])


def test_batch_predict(service):
    client, _, _, prices = service
    response = client.post(
        "/predict/batch",
        json={"windows": [_bars(prices, 60), _bars(prices.iloc[:-5], 60)]},
    )
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == 2
    assert all(0.0 <= item["probability_up"] <= 1.0 for item in predictions)


def test_short_window_is_rejected(service):
    client, _, _, prices = service
    response = client.post("/predict", json={"bars": _bars(prices, 30)})
    assert response.status_code == 422
    assert "at least 60 bars" in response.json()["detail"]


def test_missing_artifact_reports_unavailable(tmp_path):
    config = make_config(tmp_path)
    with TestClient(create_app(config)) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_loaded"] is False
        assert client.get("/model/info").status_code == 503

        prices = make_prices()
        response = client.post("/predict", json={"bars": _bars(prices)})
        assert response.status_code == 503
