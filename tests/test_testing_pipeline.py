"""Issue #11: model boundary and real SQLite/model/API wiring, without Docker."""

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import httpx
import numpy as np
import pytest
from sklearn.ensemble import IsolationForest

from app import metrics_storage as storage
from app.app import create_app
from ml import anomaly_detector as detector


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(storage.settings, "MONITOR_DB_URL", f"sqlite:///{tmp_path / 'metrics.db'}")
    monkeypatch.setattr(storage.settings, "DATABASE_URL", "sqlite://")
    monkeypatch.setattr(storage.settings, "NIM_API_KEY", "")
    monkeypatch.setattr(storage.settings, "TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setattr(storage.settings, "SENTRY_DSN", "")
    monkeypatch.setattr(storage, "_engine", None)
    monkeypatch.setattr(storage, "_initialized", False)
    monkeypatch.setattr(detector, "MODELS_DIR", tmp_path / "models")
    start = datetime.now(UTC).replace(microsecond=0) - timedelta(hours=5)
    rows = []
    for i in range(241):
        for metric, value in (
            ("row_count", 1000 + i),
            ("null_rate", 0.8 if i == 240 else 0.02),
        ):
            rows.append({
                "ts": start + timedelta(minutes=i), "table_name": "orders",
                "metric_name": metric, "value": value,
            })
    storage.save_metrics(rows, "legacy")
    app = create_app({"TESTING": True})
    yield app.test_client()
    if storage._engine is not None:
        storage._engine.dispose()


def _persist_and_read(client, scores):
    assert storage.save_anomaly_scores([
        {"table_name": "orders", **score} for score in scores
    ]) == len(scores)
    response = client.get("/api/anomalies/orders?range=24h")
    assert response.status_code == 200
    return response.get_json()


def test_mock_model_postprocessing_persists_through_api(pipeline, monkeypatch):
    """Saved scaler only; raw scores retain sign/rounding and predict drives labels."""
    model, scaler = Mock(), Mock()
    scaler.transform.side_effect = lambda matrix: matrix + 1
    raw = np.zeros(240)
    raw[:3] = [-0.123456789, 0.0, 0.123456789]
    predictions = np.ones(240, dtype=int)
    predictions[0] = -1
    model.decision_function.return_value = raw
    model.predict.return_value = predictions
    monkeypatch.setattr(detector, "_load_model", lambda *args: {
        "model": model, "scaler": scaler,
    })

    scores = detector.score_table("orders", window_days=1)
    assert [row["score"] for row in scores[:3]] == [-0.123457, 0.0, 0.123457]
    assert [row["is_anomaly"] for row in scores[:3]] == [1, 0, 0]
    scaler.fit.assert_not_called()
    scaler.fit_transform.assert_not_called()
    inputs = scaler.transform.call_args.args[0]
    np.testing.assert_allclose(inputs[0], [1001, 0.02, 1, 0])
    np.testing.assert_allclose(model.decision_function.call_args.args[0], inputs + 1)
    np.testing.assert_allclose(model.predict.call_args.args[0], inputs + 1)
    body = _persist_and_read(pipeline, scores)
    assert [{key: row[key] for key in scores[0]} for row in body] == scores


def test_real_isolation_forest_sqlite_to_api(pipeline):
    """No patched model, scaler, metric reads, score persistence or HTTP handler."""
    metadata = detector.train("orders")
    assert metadata["n_points"] == 240  # production MIN_POINTS=200 is unchanged
    persisted = detector._load_model("orders")
    assert isinstance(persisted["model"], IsolationForest)
    assert persisted["scaler"].n_samples_seen_ == 240
    scores = detector.score_table("orders", window_days=1)
    assert len(scores) == 240
    assert all(np.isfinite(row["score"]) for row in scores)
    assert scores[-1]["is_anomaly"] == 1
    assert scores[-1]["score"] < 0
    body = _persist_and_read(pipeline, scores)
    assert [{key: row[key] for key in scores[0]} for row in body] == scores
    assert "features" in body[-1]


@pytest.mark.parametrize("host", ["integrate.api.nvidia.com", "api.telegram.org"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_external_service_transport_is_blocked(host, asynchronous):
    """The common guard rejects both real HTTPX transports before network I/O."""
    async def request():
        async with httpx.AsyncClient(trust_env=False) as client:
            await client.get(f"https://{host}/")

    with pytest.raises(AssertionError, match="Unmocked external HTTPX request"):
        if asynchronous:
            asyncio.run(request())
        else:
            with httpx.Client(trust_env=False) as client:
                client.get(f"https://{host}/")
