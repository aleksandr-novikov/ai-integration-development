"""Actual ASGI HTTP, real artifact, Flask/session/SQLite and version isolation."""

import copy
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from app import inference_adapter as adapter
from app import metrics_storage as storage
from app import source_storage
from app.app import create_app as flask_app
from inference.service import create_app
from ml import model_registry as registry
from ml.inference_contract import canonical_ts, document, validator

SOURCE = {
    "project_id": "project-a",
    "connection_id": "connection-a",
    "schema_name": "public",
    "table_name": "orders",
}
TOKEN = "synthetic-test-token"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture(scope="module")
def bundles(tmp_path_factory):
    root = tmp_path_factory.mktemp("inference-registry")
    now = datetime.now(UTC) - timedelta(days=1)
    rc = [{"ts": (now + timedelta(minutes=i)).isoformat(), "value": 100 + i} for i in range(205)]
    nr = [{"ts": row["ts"], "value": 0.02 + (i % 3) * 0.001} for i, row in enumerate(rc)]
    candidates = [
        registry.train_candidate(
            root,
            source=SOURCE,
            row_counts=rc,
            null_rates=nr,
            git_sha="a" * 40,
            seed=seed,
            parameters={"n_estimators": 3},
        )
        for seed in (41, 42)
    ]
    return root, candidates


@pytest.fixture
def client(bundles):
    root, _ = bundles
    return TestClient(
        create_app(
            {
                "registry_path": str(root),
                "demo": True,
                "principals": [{"token": TOKEN, "sources": [SOURCE]}],
            }
        )
    )


def payload(bundles, version=0):
    result = copy.deepcopy(
        document()["paths"]["/v1/predict"]["post"]["requestBody"]["content"]["application/json"][
            "example"
        ]
    )
    result["model_version"] = bundles[1][version]["model_version"]
    return result


def test_actual_predict_matches_selected_artifact_without_fit(client, bundles, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("HTTP prediction must never fit")

    monkeypatch.setattr("sklearn.ensemble.IsolationForest.fit", forbidden)
    monkeypatch.setattr("sklearn.preprocessing.StandardScaler.fit", forbidden)
    request = payload(bundles)
    response = client.post("/v1/predict", json=request, headers=HEADERS)
    assert response.status_code == 200, response.text
    validator("PredictionResponse").validate(response.json())
    expected = registry.load_version(
        bundles[0], request["model_version"], source=SOURCE, demo=True
    ).predict(request["features"])
    assert response.json() == {
        **expected,
        "observed_at": request["observed_at"],
        "demo": True,
        "quality_admitted": False,
    }
    model = {key: request[key] for key in ("source", "model_version", "feature_schema_version")}
    info = client.post("/v1/model", json=model, headers=HEADERS)
    assert info.status_code == 200
    assert info.json()["demo"] is True
    assert info.json()["quality_admitted"] is False
    assert client.get("/healthz").json() == {"status": "ok", "demo": True}
    assert client.get("/docs").status_code == 200
    assert set(client.get("/openapi.json").json()["paths"]) == {
        "/healthz",
        "/v1/model",
        "/v1/predict",
    }


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e400", "1" + "0" * 400])
def test_real_http_decoder_rejects_nonfinite_numbers(client, bundles, token):
    raw = json.dumps(payload(bundles)).replace('"row_count": 110.0', f'"row_count": {token}')
    response = client.post(
        "/v1/predict", content=raw, headers={**HEADERS, "Content-Type": "application/json"}
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_json"


@pytest.mark.parametrize("value", ["110", True, None, -1])
def test_real_http_types_are_not_coerced(client, bundles, value):
    request = payload(bundles)
    request["features"]["row_count"] = value
    assert client.post("/v1/predict", json=request, headers=HEADERS).status_code == 422


def test_validation_priority_and_source_scope(client, bundles):
    assert client.post("/v1/predict", content="bad").status_code == 401
    assert client.post("/v1/predict", content="bad", headers=HEADERS).status_code == 415
    request = payload(bundles)
    request["source"]["connection_id"] = "other-connection"
    request["model_version"] = "missing"
    assert client.post("/v1/predict", json=request, headers=HEADERS).status_code == 403
    request["source"] = SOURCE
    request["feature_schema_version"] = "2.0.0"
    assert client.post("/v1/predict", json=request, headers=HEADERS).status_code == 409
    request["feature_schema_version"] = "1.0.0"
    assert client.post("/v1/predict", json=request, headers=HEADERS).status_code == 503


@pytest.mark.parametrize("change", ["time", "delta", "extra", "timezone"])
def test_cross_field_and_structural_errors(client, bundles, change):
    request = payload(bundles)
    if change == "time":
        request["previous_observed_at"] = request["observed_at"]
    elif change == "delta":
        request["features"]["d_row_count"] = 999
    elif change == "extra":
        request["unexpected"] = 1
    else:
        request["observed_at"] = "2026-10-25T12:00:00"
    assert client.post("/v1/predict", json=request, headers=HEADERS).status_code == 422


@pytest.mark.parametrize("field", ["observed_at", "previous_observed_at"])
@pytest.mark.parametrize("stamp", ["0001-01-01T00:00:00+01:00", "9999-12-31T23:59:59-01:00"])
def test_timestamp_outside_utc_range_returns_invalid_input(client, bundles, field, stamp):
    request = payload(bundles)
    request[field] = stamp
    response = client.post("/v1/predict", json=request, headers=HEADERS)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_input"


@pytest.mark.parametrize("stamp", ["0001-01-01T00:00:00+01:00", "9999-12-31T23:59:59-01:00"])
def test_explanation_timestamp_outside_utc_range_is_rejected_before_reads(ui, monkeypatch, stamp):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid timestamp must not read history/cache or call the LLM")

    for name in ("_has_project_table_metrics", "get_cached_explanation"):
        monkeypatch.setattr(f"app.api.{name}", forbidden)
    monkeypatch.setattr(source_storage, "metrics", forbidden)
    monkeypatch.setattr("app.llm.explain_anomaly", forbidden)
    response = ui[1].post(
        "/api/explain", json={"table": "orders", "metric": "row_count", "ts": stamp}
    )
    assert response.status_code == 400
    assert response.json == {"error": "ts must be RFC 3339 with a timezone"}


@pytest.mark.parametrize("stamp", ["0001-01-01T00:00:00Z", "0001-01-01T01:00:00Z"])
def test_explanation_context_window_stays_in_datetime_range(ui, monkeypatch, stamp):
    from app import llm

    monkeypatch.setattr(llm.settings, "NIM_API_KEY", "")
    response = ui[1].post(
        "/api/explain",
        json={"source": SOURCE, "table": "orders", "metric": "row_count", "ts": stamp},
    )
    assert response.status_code == 200
    assert response.json["confidence"] == 0.3


def test_source_bound_model_cannot_be_reused_by_other_allowed_source(bundles):
    other = {**SOURCE, "connection_id": "other-connection"}
    http = TestClient(
        create_app(
            {
                "registry_path": str(bundles[0]),
                "demo": True,
                "principals": [{"token": TOKEN, "sources": [SOURCE, other]}],
            }
        )
    )
    request = payload(bundles)
    request["source"] = other
    assert http.post("/v1/predict", json=request, headers=HEADERS).status_code == 503


def test_no_positive_admission_requires_explicit_demo_mode(bundles):
    http = TestClient(
        create_app(
            {
                "registry_path": str(bundles[0]),
                "demo": False,
                "principals": [{"token": TOKEN, "sources": [SOURCE]}],
            }
        )
    )
    assert http.post("/v1/predict", json=payload(bundles), headers=HEADERS).status_code == 503


def test_incompatible_verified_manifest_precedes_model_deserialization(tmp_path, bundles):
    manifest = copy.deepcopy(bundles[1][0])
    manifest.pop("model_version")
    manifest["feature_schema_version"] = "2.0.0"
    version = "if-" + registry.sha256(registry.canonical(manifest))
    directory = tmp_path / "versions" / version
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_bytes(
        registry.canonical({**manifest, "model_version": version})
    )
    http = TestClient(
        create_app(
            {
                "registry_path": str(tmp_path),
                "demo": True,
                "principals": [{"token": TOKEN, "sources": [SOURCE]}],
            }
        )
    )
    request = payload(bundles)
    request["model_version"] = version
    assert http.post("/v1/predict", json=request, headers=HEADERS).status_code == 409
    (directory / "manifest.json").write_text("{}")
    assert http.post("/v1/predict", json=request, headers=HEADERS).status_code == 503


def test_service_import_has_no_flask_or_scheduler_side_effects():
    code = "import sys; from inference.service import create_app; create_app({'registry_path': '/missing', 'principals': []}); assert 'app.app' not in sys.modules; assert 'app.metrics_storage' not in sys.modules; assert 'collectors.scheduler' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)


def test_cached_model_still_rejects_artifact_corruption(tmp_path, bundles, monkeypatch):
    shutil.copytree(bundles[0], tmp_path / "registry")
    http = TestClient(
        create_app(
            {
                "registry_path": str(tmp_path / "registry"),
                "demo": True,
                "principals": [{"token": TOKEN, "sources": [SOURCE]}],
            }
        )
    )
    loads = []
    original = registry._load_pipeline

    def tracked(*args):
        loads.append(1)
        return original(*args)

    monkeypatch.setattr(registry, "_load_pipeline", tracked)
    request = payload(bundles)
    for _ in range(2):
        assert http.post("/v1/predict", json=request, headers=HEADERS).status_code == 200
    assert len(loads) == 1
    artifact = tmp_path / "registry/versions" / request["model_version"] / "pipeline.skops"
    artifact.write_bytes(artifact.read_bytes() + b"corruption")
    assert http.post("/v1/predict", json=request, headers=HEADERS).status_code == 503


@pytest.fixture
def ui(tmp_path, monkeypatch, client, bundles):
    monkeypatch.setattr(storage.settings, "MONITOR_DB_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setattr(storage.settings, "DATABASE_URL", "sqlite://")
    monkeypatch.setattr(storage, "_engine", None)
    monkeypatch.setattr(storage, "_initialized", False)
    storage.create_user("u", "u@example.test", "unused")
    storage.create_project(SOURCE["project_id"], "u", "A", "a")
    storage.create_connection(
        connection_id=SOURCE["connection_id"],
        project_id=SOURCE["project_id"],
        name="A",
        dsn_encrypted=b"unused",
    )
    config = {
        "url": "http://testserver",
        "demo": True,
        "token_env": "TEST_INFERENCE_TOKEN",
        "bindings": [
            {
                "source": SOURCE,
                "model_version": bundles[1][0]["model_version"],
                "feature_schema_version": "1.0.0",
            }
        ],
    }
    filename = tmp_path / "adapter.json"
    filename.write_text(json.dumps(config))
    monkeypatch.setenv("INFERENCE_ADAPTER_CONFIG", str(filename))
    monkeypatch.setenv("TEST_INFERENCE_TOKEN", TOKEN)
    app = flask_app({"TESTING": True, "LOGIN_DISABLED": False, "INFERENCE_HTTP_CLIENT": client})
    http = app.test_client()
    with http.session_transaction() as session:
        session["_user_id"] = "u"
        session["_fresh"] = True
        session["current_project_id"] = SOURCE["project_id"]
    for metric, values in [("row_count", [100, 110]), ("null_rate", [0.02, 0.03])]:
        storage.save_metrics(
            [
                {
                    "ts": (
                        datetime.now(UTC).replace(second=0, microsecond=0)
                        - timedelta(minutes=15 * (2 - i))
                    ).isoformat(),
                    "table_name": "orders",
                    "metric_name": metric,
                    "value": value,
                }
                for i, value in enumerate(values)
            ],
            SOURCE["project_id"],
            connection_id=SOURCE["connection_id"],
            schema_name=SOURCE["schema_name"],
        )
    yield app, http, config
    storage._engine.dispose()


def test_flask_adapter_dashboard_two_versions_and_idempotence(ui, bundles):
    app, http, config = ui
    entry = config["bindings"][0]
    first = http.post("/api/inference/predict", json=entry)
    assert first.status_code == 200, first.text
    assert http.post("/api/inference/predict", json=entry).json == first.json
    second_entry = {**entry, "model_version": bundles[1][1]["model_version"]}
    second = http.post("/api/inference/predict", json=second_entry)
    assert second.status_code == 200
    for result in [first.json, second.json]:
        rows = source_storage.scores(
            SOURCE, result["model_version"], result["feature_schema_version"]
        )
        assert len(rows) == 1
        assert rows[0]["score"] == result["score"]
        assert rows[0]["demo"] is True
        query = {
            **SOURCE,
            "model_version": result["model_version"],
            "feature_schema_version": "1.0.0",
        }
        assert http.get("/api/inference/scores", query_string=query).json == rows
        page = http.get(
            "/dashboard/inference", query_string={"version": result["model_version"] + "|1.0.0"}
        )
        assert page.status_code == 200
        assert str(result["score"]) in page.text
    assert http.post("/api/inference/predict", json=entry).json == first.json  # Explicit rollback.
    with storage.get_engine().connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM anomaly_scores")).scalar() == 2
    assert storage.get_anomaly_scores("orders", SOURCE["project_id"]) == []


def test_flask_denies_anonymous_and_foreign_source(ui):
    app, http, config = ui
    entry = config["bindings"][0]
    assert app.test_client().post("/api/inference/predict", json=entry).status_code == 401
    foreign = {**entry, "source": {**SOURCE, "project_id": "other"}}
    assert http.post("/api/inference/predict", json=foreign).status_code == 403


def test_dashboard_sends_a_valid_csrf_token(ui):
    app, http, config = ui
    app.config["WTF_CSRF_ENABLED"] = True
    page = http.get("/dashboard/inference")
    token = re.search(r"'X-CSRFToken': \"([^\"]+)\"", page.text).group(1)
    assert http.post("/api/inference/predict", json=config["bindings"][0]).status_code == 400
    assert (
        http.post(
            "/api/inference/predict", json=config["bindings"][0], headers={"X-CSRFToken": token}
        ).status_code
        == 200
    )


def test_demo_result_remains_marked_after_configuration_switch(ui, monkeypatch):
    _, http, config = ui
    assert http.post("/api/inference/predict", json=config["bindings"][0]).status_code == 200
    monkeypatch.setattr(adapter, "configuration", lambda: {**config, "demo": False})
    assert "Учебная демонстрация" in http.get("/dashboard/inference").text
    response = http.post("/api/inference/predict", json=config["bindings"][0])
    assert response.status_code == 503
    assert response.json["error"]["code"] == "invalid_inference_response"


def test_missing_previous_and_duplicates_do_not_call_service(ui, monkeypatch):
    app, http, config = ui
    with storage.get_engine().begin() as conn:
        conn.execute(text("DELETE FROM metrics WHERE metric_name = 'null_rate'"))
    response = http.post("/api/inference/predict", json=config["bindings"][0])
    assert response.status_code == 422
    assert response.json["error"]["code"] == "insufficient_history"


def test_anonymous_explanation_cache_and_wrong_json_types(ui):
    app, http, _ = ui
    stamp = datetime.now(UTC).isoformat()
    storage.save_explanation(
        "orders", "row_count", stamp, "secret", "fix", 0.8, project_id=SOURCE["project_id"]
    )
    assert (
        app.test_client()
        .post("/api/explain", json={"table": "orders", "metric": "row_count", "ts": stamp})
        .status_code
        == 401
    )
    for body in [[1], {"table": 5, "metric": "row_count", "ts": stamp}]:
        assert http.post("/api/explain", json=body).status_code == 400


def test_cache_has_full_source_scope_and_utc_equivalent_keys(ui):
    stamp = datetime.now(UTC).isoformat()
    scope = {"project_id": SOURCE["project_id"], "connection_id": "a", "schema_name": "public"}
    storage.save_explanation("orders", "row_count", stamp, "A", "fix", 0.8, **scope)
    assert (
        storage.get_cached_explanation("orders", "row_count", stamp, **scope)["explanation"] == "A"
    )
    assert (
        storage.get_cached_explanation(
            "orders", "row_count", stamp, **{**scope, "connection_id": "b"}
        )
        is None
    )
    assert (
        storage.get_cached_explanation(
            "orders", "row_count", stamp, **{**scope, "schema_name": "other"}
        )
        is None
    )
    assert (
        storage.get_cached_explanation("orders", "row_count", stamp, ttl_hours=-1, **scope) is None
    )
    assert canonical_ts(stamp) == canonical_ts(
        datetime.fromisoformat(stamp).astimezone().isoformat()
    )
    assert (
        storage.get_cached_explanation(
            "orders", "row_count", datetime.fromisoformat(stamp).astimezone().isoformat(), **scope
        )["explanation"]
        == "A"
    )


def test_scoped_llm_context_and_telegram_ambiguity(ui, monkeypatch):
    from app import llm

    _, http, config = ui
    source_storage.save_snapshot(SOURCE, [{"name": "only_source_a", "type": "int"}])
    storage.save_schema_snapshot("orders", [{"name": "wrong_project_snapshot", "type": "int"}])
    stamp = source_storage.metrics(SOURCE, "row_count")[-1]["ts"]
    prompts = []

    def nim(prompt):
        prompts.append(prompt)
        return json.dumps({"explanation": "source A", "suggested_fix": "check", "confidence": 0.8})

    monkeypatch.setattr(storage.settings, "NIM_API_KEY", "synthetic")
    monkeypatch.setattr(llm, "_call_nim", nim)
    # Direct explain_anomaly is also the path used by Telegram notifications.
    assert (
        llm.explain_anomaly("orders", "row_count", stamp, SOURCE["project_id"])["explanation"]
        == "source A"
    )
    assert "only_source_a" in prompts[0]
    assert "wrong_project_snapshot" not in prompts[0]
    body = {"source": SOURCE, "table": "orders", "metric": "row_count", "ts": stamp}
    assert http.post("/api/explain", json=body).status_code == 200
    assert http.post("/api/explain", json=body).json["explanation"] == "source A"
    assert len(prompts) == 2
    storage.save_metrics(
        [{"ts": stamp, "table_name": "orders", "metric_name": "row_count", "value": 999}],
        SOURCE["project_id"],
        connection_id="b",
        schema_name="public",
    )
    storage.create_connection(
        connection_id="b", project_id=SOURCE["project_id"], name="B", dsn_encrypted=b"unused"
    )
    ambiguous = llm.explain_anomaly("orders", "row_count", stamp, SOURCE["project_id"])
    assert ambiguous["confidence"] == 0.3
    assert len(prompts) == 2
    assert (
        http.post(
            "/api/explain", json={key: value for key, value in body.items() if key != "source"}
        ).status_code
        == 409
    )


def test_background_scoring_uses_the_same_http_service(ui):
    app, _, _ = ui
    adapter.score_project(SOURCE["project_id"], client=app.config["INFERENCE_HTTP_CLIENT"])
    with storage.get_engine().connect() as conn:
        assert (
            conn.execute(
                text("SELECT COUNT(*) FROM anomaly_scores WHERE model_version <> 'legacy-unknown'")
            ).scalar()
            == 1
        )


def test_nightly_job_does_not_retrain_the_served_legacy_model(ui, monkeypatch):
    from collectors import scheduler

    calls = []
    monkeypatch.setattr(scheduler, "_iter_retrain_projects", lambda: [SOURCE["project_id"]])
    monkeypatch.setattr(adapter, "train_candidates", lambda pid: calls.append(("candidate", pid)))
    monkeypatch.setattr(adapter, "score_project", lambda pid: calls.append(("score", pid)))

    def forbidden(*args, **kwargs):
        pytest.fail("served source must not retrain/score the legacy mutable model")

    monkeypatch.setattr("ml.anomaly_detector.retrain_all", forbidden)
    monkeypatch.setattr("ml.anomaly_detector.score_table", forbidden)
    scheduler.retrain_anomaly_detectors()
    assert calls == [("candidate", SOURCE["project_id"]), ("score", SOURCE["project_id"])]


def test_candidate_training_failure_does_not_disable_published_scoring(ui, monkeypatch):
    from collectors import scheduler

    calls = []
    monkeypatch.setattr(scheduler, "_iter_retrain_projects", lambda: [SOURCE["project_id"]])

    def rejected(pid):
        raise ValueError("not enough training history")

    monkeypatch.setattr(adapter, "train_candidates", rejected)
    monkeypatch.setattr(adapter, "score_project", lambda pid: calls.append(pid))
    scheduler.retrain_anomaly_detectors()
    assert calls == [SOURCE["project_id"]]


@pytest.fixture
def two_sources(ui, monkeypatch, bundles):
    """First binding is independent of the existing published source."""
    app, _, config = ui
    other = {**SOURCE, "connection_id": "connection-b"}
    storage.create_connection(
        connection_id=other["connection_id"],
        project_id=other["project_id"],
        name="B",
        dsn_encrypted=b"unused",
    )
    for metric in ("row_count", "null_rate"):
        storage.save_metrics(
            [
                {**row, "table_name": "orders", "metric_name": metric}
                for row in source_storage.metrics(SOURCE, metric)
            ],
            other["project_id"],
            connection_id=other["connection_id"],
            schema_name="public",
        )
    now = datetime.now(UTC) - timedelta(days=1)
    rc = [{"ts": (now + timedelta(minutes=i)).isoformat(), "value": 100 + i} for i in range(205)]
    nr = [{"ts": row["ts"], "value": 0.02} for row in rc]
    candidate = registry.train_candidate(
        bundles[0],
        source=other,
        row_counts=rc,
        null_rates=nr,
        git_sha="a" * 40,
        parameters={"n_estimators": 3},
    )
    first = {**config["bindings"][0], "source": other, "model_version": candidate["model_version"]}
    config["bindings"].insert(0, first)
    monkeypatch.setattr(adapter, "configuration", lambda: config)
    app.config["INFERENCE_HTTP_CLIENT"] = TestClient(
        create_app(
            {
                "registry_path": str(bundles[0]),
                "demo": True,
                "principals": [{"token": TOKEN, "sources": [SOURCE, other]}],
            }
        )
    )
    return app, config, first


def test_source_failure_does_not_block_other_published_model(two_sources, caplog):
    app, config, first = two_sources
    # A real ASGI 503 for an absent artifact, followed by a source with a real model.
    first["model_version"] = "missing"
    before = copy.deepcopy(config)
    adapter.score_project(SOURCE["project_id"], client=app.config["INFERENCE_HTTP_CLIENT"])
    healthy = config["bindings"][1]
    assert len(source_storage.scores(SOURCE, healthy["model_version"], "1.0.0")) == 1
    assert source_storage.versions(first["source"]) == []
    assert config == before  # A failure must never select a different serving model.
    assert "connection-b" in caplog.text
    assert "model_unavailable" in caplog.text
    assert "project-a" in caplog.text and "public" in caplog.text and "orders" in caplog.text


@pytest.mark.parametrize("failure", ["metrics", "features"])
def test_source_history_failure_is_isolated(two_sources, monkeypatch, caplog, failure):
    from sqlalchemy.exc import OperationalError

    app, config, first = two_sources
    original = source_storage.metrics

    def broken(source, metric, *args, **kwargs):
        if source == first["source"]:
            if failure == "metrics":
                raise OperationalError("SELECT", {}, Exception("unavailable"))
            return [
                {"ts": "2026-10-01T00:00:00Z", "value": "bad"},
                {"ts": "2026-10-01T00:15:00Z", "value": 100},
            ]
        return original(source, metric, *args, **kwargs)

    monkeypatch.setattr(source_storage, "metrics", broken)
    adapter.score_project(SOURCE["project_id"], client=app.config["INFERENCE_HTTP_CLIENT"])
    assert source_storage.versions(first["source"]) == []
    assert len(source_storage.scores(SOURCE, config["bindings"][1]["model_version"], "1.0.0")) == 1
    assert "connection-b" in caplog.text


def test_two_healthy_connections_are_isolated_and_idempotent(two_sources):
    app, config, _ = two_sources
    for _ in range(2):
        adapter.score_project(SOURCE["project_id"], client=app.config["INFERENCE_HTTP_CLIENT"])
    for entry in config["bindings"]:
        rows = source_storage.scores(entry["source"], entry["model_version"], "1.0.0")
        assert len(rows) == 1
        assert rows[0]["source"] == entry["source"]
    with storage.get_engine().connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM anomaly_scores")).scalar() == 2


def test_source_outage_keeps_existing_versions(two_sources):
    app, config, first = two_sources
    adapter.score_project(SOURCE["project_id"], client=app.config["INFERENCE_HTTP_CLIENT"])
    previous = [
        source_storage.scores(e["source"], e["model_version"], "1.0.0") for e in config["bindings"]
    ]
    first["model_version"] = "missing"
    adapter.score_project(SOURCE["project_id"], client=app.config["INFERENCE_HTTP_CLIENT"])
    for entry, rows in zip(config["bindings"], previous, strict=True):
        assert source_storage.scores(entry["source"], rows[0]["model_version"], "1.0.0") == rows
    assert source_storage.scores(first["source"], "missing", "1.0.0") == []


@pytest.mark.parametrize("error", [RuntimeError("bug"), MemoryError("out of memory")])
def test_system_errors_are_not_suppressed(two_sources, monkeypatch, error):
    app, _, _ = two_sources
    attempts = []

    def broken(source, *args, **kwargs):
        attempts.append(source["connection_id"])
        raise type(error)(str(error))

    monkeypatch.setattr(source_storage, "metrics", broken)
    if isinstance(error, MemoryError):
        # Resource exhaustion stops the job at once instead of trying more sources.
        with pytest.raises(MemoryError, match=str(error)):
            adapter.score_project(SOURCE["project_id"], client=app.config["INFERENCE_HTTP_CLIENT"])
        assert attempts == ["connection-b"]
        return
    with pytest.raises(ExceptionGroup) as info:
        adapter.score_project(SOURCE["project_id"], client=app.config["INFERENCE_HTTP_CLIENT"])
    # Every source is attempted; the failures are then reported together.
    assert attempts == ["connection-b", "connection-a"]
    assert [(type(e), str(e)) for e in info.value.exceptions] == [(RuntimeError, "bug")] * 2
    assert "connection-b" in info.value.exceptions[0].__notes__[0]
    assert "connection-a" in info.value.exceptions[1].__notes__[0]


def test_unexpected_prediction_value_error_is_not_suppressed(two_sources, monkeypatch):
    app, _, _ = two_sources
    calls = []

    def broken(source, *args, **kwargs):
        calls.append(source["connection_id"])
        raise ValueError("unexpected prediction bug")

    monkeypatch.setattr(adapter, "predict", broken)
    with pytest.raises(ExceptionGroup) as info:
        adapter.score_project(SOURCE["project_id"], client=app.config["INFERENCE_HTTP_CLIENT"])
    assert calls == ["connection-b", "connection-a"]
    assert [(type(e), str(e)) for e in info.value.exceptions] == [
        (ValueError, "unexpected prediction bug")
    ] * 2


def duplicate_observation(source, metric):
    """Different textual timestamps can represent one canonical observation."""
    stamp = source_storage.metrics(source, metric)[-1]["ts"]
    equivalent = datetime.fromisoformat(stamp.replace("Z", "+00:00")).strftime(
        "%Y-%m-%dT%H:%M:%S.000000+00:00"
    )
    with storage.get_engine().begin() as conn:
        conn.execute(
            text(
                "INSERT INTO metrics (project_id, connection_id, schema_name, table_name, "
                "metric_name, ts, value) VALUES (:project_id, :connection_id, :schema_name, "
                ":table_name, :metric, :ts, 999)"
            ),
            {**source, "metric": metric, "ts": equivalent},
        )
    return stamp


@pytest.mark.parametrize("metric", ["row_count", "null_rate"])
def test_duplicate_metrics_returns_conflict(ui, metric):
    app, http, config = ui
    app.config["PROPAGATE_EXCEPTIONS"] = False
    duplicate_observation(SOURCE, metric)
    response = http.get("/api/inference/metrics", query_string={**SOURCE, "metric": metric})
    assert response.status_code == 409
    assert response.json == {"error": {"code": "ambiguous_source", "message": "ambiguous_source"}}
    response = http.post("/api/inference/predict", json=config["bindings"][0])
    assert response.status_code == 409
    assert response.json["error"]["code"] == "ambiguous_source"


@pytest.mark.parametrize("metric", ["row_count", "null_rate"])
def test_explanation_conflict_precedes_cache_and_llm(ui, monkeypatch, metric):
    from app import api, llm

    app, http, _ = ui
    app.config["PROPAGATE_EXCEPTIONS"] = False
    stamp = duplicate_observation(SOURCE, metric)

    def forbidden(*args, **kwargs):
        pytest.fail("ambiguous history must not read/write cache or call LLM")

    monkeypatch.setattr(api, "get_cached_explanation", forbidden)
    monkeypatch.setattr(api, "save_explanation", forbidden)
    monkeypatch.setattr(llm, "_call_nim", forbidden)
    monkeypatch.setattr(llm, "explain_anomaly", forbidden)
    response = http.post(
        "/api/explain",
        json={
            "source": SOURCE,
            "table": "orders",
            "metric": "row_count",
            "ts": stamp,
        },
    )
    assert response.status_code == 409
    assert response.json == {"error": "ambiguous_source"}
    with storage.get_engine().connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM llm_explanations")).scalar() == 0


@pytest.mark.parametrize("route", ["metrics", "explain"])
def test_history_authorization_precedes_conflict(ui, monkeypatch, route):
    app, http, _ = ui
    stamp = duplicate_observation(SOURCE, "row_count")

    def forbidden(*args, **kwargs):
        pytest.fail("unauthorized request must not read protected history")

    monkeypatch.setattr(source_storage, "metrics", forbidden)
    foreign_sources = [
        {**SOURCE, "project_id": "foreign-project", "connection_id": "foreign-project-connection"},
        {**SOURCE, "connection_id": "foreign-connection"},
        {**SOURCE, "schema_name": "foreign-schema"},
    ]
    for client, source, expected in [
        (app.test_client(), SOURCE, 401),
        *[(http, s, 403) for s in foreign_sources],
    ]:
        if route == "metrics":
            response = client.get("/api/inference/metrics", query_string=source)
        else:
            response = client.post(
                "/api/explain",
                json={
                    "source": source,
                    "table": "orders",
                    "metric": "row_count",
                    "ts": stamp,
                },
            )
        assert response.status_code == expected
        assert "ambiguous_source" not in response.text
        assert "999" not in response.text


def test_conflict_does_not_cross_connection_or_project(two_sources, ui, monkeypatch):
    from app import llm

    _, http, _ = ui
    _, _, first = two_sources
    duplicate_observation(first["source"], "row_count")
    # Same connection/schema/table strings in another project also stay isolated.
    foreign = {**SOURCE, "project_id": "foreign-project"}
    storage.save_metrics(
        [
            {"ts": row["ts"], "value": 999, "table_name": "orders", "metric_name": "row_count"}
            for row in source_storage.metrics(SOURCE, "row_count")
        ],
        foreign["project_id"],
        connection_id=foreign["connection_id"],
        schema_name="public",
    )
    duplicate_observation(foreign, "row_count")
    response = http.get("/api/inference/metrics", query_string=SOURCE)
    assert response.status_code == 200
    assert [r["value"] for r in response.json] == [100, 110]
    # The GET contract uses the session's project, never a caller's query project.
    assert http.get("/api/inference/metrics", query_string=foreign).json == response.json
    monkeypatch.setattr(
        llm,
        "explain_anomaly",
        lambda *a, **kw: {
            "explanation": "healthy",
            "suggested_fix": "check",
            "confidence": 0.8,
        },
    )
    response = http.post(
        "/api/explain",
        json={
            "source": SOURCE,
            "table": "orders",
            "metric": "row_count",
            "ts": response.json[-1]["ts"],
        },
    )
    assert response.status_code == 200
    assert response.json["explanation"] == "healthy"
    assert http.get("/api/inference/metrics", query_string=first["source"]).status_code == 409


@pytest.mark.parametrize("route", ["metrics", "explain", "predict"])
def test_unknown_value_error_is_not_an_ambiguity(ui, monkeypatch, route):
    app, http, config = ui

    def broken(*args, **kwargs):
        raise ValueError("unexpected internal failure")

    monkeypatch.setattr(source_storage, "metrics", broken)
    app.config["PROPAGATE_EXCEPTIONS"] = False
    if route == "metrics":
        response = http.get("/api/inference/metrics", query_string=SOURCE)
    elif route == "predict":
        response = http.post("/api/inference/predict", json=config["bindings"][0])
    else:
        response = http.post(
            "/api/explain",
            json={
                "source": SOURCE,
                "table": "orders",
                "metric": "row_count",
                "ts": datetime.now(UTC).isoformat(),
            },
        )
    assert response.status_code == 500
    assert "ambiguous_source" not in response.text


def add_history(source, minutes_back, row_counts):
    """Aligned row_count/null_rate points `minutes_back` minutes before now."""
    base = datetime.now(UTC).replace(second=0, microsecond=0)
    for metric, values in [("row_count", row_counts), ("null_rate", [0.02] * len(row_counts))]:
        storage.save_metrics(
            [
                {
                    "ts": (base - timedelta(minutes=minutes)).isoformat(),
                    "table_name": source["table_name"],
                    "metric_name": metric,
                    "value": value,
                }
                for minutes, value in zip(minutes_back, values, strict=True)
            ],
            source["project_id"],
            connection_id=source["connection_id"],
            schema_name=source["schema_name"],
        )


def set_value(source, metric, ts, value):
    with storage.get_engine().begin() as conn:
        conn.execute(
            text(
                "UPDATE metrics SET value = :value WHERE project_id = :project_id "
                "AND connection_id = :connection_id AND schema_name = :schema_name "
                "AND table_name = :table_name AND metric_name = :metric AND ts = :ts"
            ),
            {**source, "metric": metric, "ts": ts, "value": value},
        )


def stored_ts(source, metric="row_count"):
    """Raw stored timestamps (needed to update a row in place)."""
    with storage.get_engine().connect() as conn:
        return [
            row[0]
            for row in conn.execute(
                text(
                    "SELECT ts FROM metrics WHERE project_id = :project_id "
                    "AND connection_id = :connection_id AND metric_name = :metric ORDER BY ts"
                ),
                {**source, "metric": metric},
            )
        ]


def break_allowlist(source):
    with storage.get_engine().begin() as conn:
        conn.execute(
            text("UPDATE connections SET iceberg_namespace_allowlist = 'not-json' WHERE id = :id"),
            {"id": source["connection_id"]},
        )


class RecordingClient:
    """Real ASGI client; optionally answers one observation with a fixed status."""

    def __init__(self, inner, reject=None, status=422):
        self.inner, self.reject, self.status, self.sent = inner, reject, status, []

    def post(self, url, json=None, **kwargs):
        self.sent.append((json["source"]["connection_id"], json["observed_at"]))
        if json["observed_at"] == self.reject:
            return httpx.Response(self.status, json={"error": {"code": "rejected"}})
        return self.inner.post(url, json=json, **kwargs)


def test_malformed_source_configuration_is_isolated_and_reported(two_sources, caplog):
    app, config, first = two_sources
    break_allowlist(first["source"])
    with pytest.raises(ExceptionGroup) as info:
        adapter.score_project(SOURCE["project_id"], client=app.config["INFERENCE_HTTP_CLIENT"])
    healthy = config["bindings"][1]
    assert len(source_storage.scores(SOURCE, healthy["model_version"], "1.0.0")) == 1
    assert source_storage.versions(first["source"]) == []
    (error,) = info.value.exceptions
    assert isinstance(error, json.JSONDecodeError)
    assert "connection-b" in error.__notes__[0] and "orders" in error.__notes__[0]
    # Reported once, by the caller that receives the group, not also here.
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_memory_error_still_reports_earlier_unexpected_failures(two_sources, monkeypatch, caplog):
    app, _, first = two_sources

    def broken(source, *args, **kwargs):
        if source == first["source"]:
            raise RuntimeError("bug")
        raise MemoryError("out of memory")

    monkeypatch.setattr(source_storage, "metrics", broken)
    with pytest.raises(MemoryError):
        adapter.score_project(SOURCE["project_id"], client=app.config["INFERENCE_HTTP_CLIENT"])
    (record,) = [r for r in caplog.records if r.levelno >= logging.ERROR]
    group = record.exc_info[1]
    assert isinstance(group, ExceptionGroup)
    assert [(type(e), str(e)) for e in group.exceptions] == [(RuntimeError, "bug")]
    assert "connection-b" in group.exceptions[0].__notes__[0]


@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
def test_non_finite_history_never_reaches_service(two_sources, caplog, value):
    app, config, first = two_sources
    caplog.set_level(logging.WARNING)
    set_value(first["source"], "row_count", stored_ts(first["source"])[-1], value)
    assert source_storage.metrics(first["source"], "row_count")[-1]["value"] == value
    client = RecordingClient(app.config["INFERENCE_HTTP_CLIENT"])
    adapter.score_project(SOURCE["project_id"], client=client)  # nothing unexpected
    assert [sent[0] for sent in client.sent] == ["connection-a"]
    assert source_storage.versions(first["source"]) == []
    healthy = config["bindings"][1]
    assert len(source_storage.scores(SOURCE, healthy["model_version"], "1.0.0")) == 1
    assert "invalid_history" in caplog.text and "connection-b" in caplog.text


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_non_finite_history_returns_invalid_history(ui, monkeypatch, value):
    app, http, config = ui
    app.config["PROPAGATE_EXCEPTIONS"] = False
    client = RecordingClient(app.config["INFERENCE_HTTP_CLIENT"])
    app.config["INFERENCE_HTTP_CLIENT"] = client
    if value == float("inf"):
        set_value(SOURCE, "null_rate", stored_ts(SOURCE, "null_rate")[-1], value)
    else:  # SQLite cannot store NaN; PostgreSQL DOUBLE PRECISION can.
        original = source_storage.metrics

        def with_nan(source, metric, *args, **kwargs):
            rows = original(source, metric, *args, **kwargs)
            return [*rows[:-1], {**rows[-1], "value": value}] if metric == "null_rate" else rows

        monkeypatch.setattr(source_storage, "metrics", with_nan)
    response = http.post("/api/inference/predict", json=config["bindings"][0])
    assert response.status_code == 422
    assert response.json == {"error": {"code": "invalid_history", "message": "invalid_history"}}
    assert client.sent == []
    with storage.get_engine().connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM anomaly_scores")).scalar() == 0


def test_real_duplicate_in_one_source_does_not_block_others(two_sources, caplog):
    app, config, first = two_sources
    duplicate_observation(first["source"], "null_rate")
    adapter.score_project(SOURCE["project_id"], client=app.config["INFERENCE_HTTP_CLIENT"])
    healthy = config["bindings"][1]
    assert len(source_storage.scores(SOURCE, healthy["model_version"], "1.0.0")) == 1
    assert source_storage.versions(first["source"]) == []
    assert "ambiguous_source" in caplog.text and "connection-b" in caplog.text


def test_unexpected_error_is_reported_after_other_sources(two_sources, monkeypatch):
    app, config, first = two_sources
    original = source_storage.insert_prediction

    def broken(result):
        if result["source"] == first["source"]:
            raise RuntimeError("storage bug")
        return original(result)

    monkeypatch.setattr(source_storage, "insert_prediction", broken)
    with pytest.raises(ExceptionGroup) as info:
        adapter.score_project(SOURCE["project_id"], client=app.config["INFERENCE_HTTP_CLIENT"])
    (error,) = info.value.exceptions
    assert (type(error), str(error)) == (RuntimeError, "storage bug")
    assert "connection-b" in error.__notes__[0]
    healthy = config["bindings"][1]
    assert len(source_storage.scores(SOURCE, healthy["model_version"], "1.0.0")) == 1


@pytest.fixture
def long_history(ui):
    """SOURCE gets five aligned points, i.e. four scorable observations."""
    app, _, config = ui
    add_history(SOURCE, [75, 60, 45], [70, 80, 90])
    stamps = [row["ts"] for row in source_storage.metrics(SOURCE, "row_count")]
    assert len(stamps) == 5
    return app, config, stamps


def saved(entry):
    rows = source_storage.scores(entry["source"], entry["model_version"], "1.0.0")
    assert all(
        (r["source"], r["model_version"], r["feature_schema_version"])
        == (entry["source"], entry["model_version"], "1.0.0")
        for r in rows
    )
    return [canonical_ts(r["ts"]) for r in rows]


def test_rejected_middle_observation_does_not_block_later_ones(long_history, caplog):
    app, config, stamps = long_history
    caplog.set_level(logging.WARNING)
    entry = config["bindings"][0]
    client = RecordingClient(app.config["INFERENCE_HTTP_CLIENT"], reject=stamps[2])
    for _ in range(2):  # repeated runs stay idempotent around the rejected point
        adapter.score_project(SOURCE["project_id"], client=client)
    assert saved(entry) == [stamps[1], stamps[3], stamps[4]]
    assert source_storage.versions(SOURCE) == [
        {"model_version": entry["model_version"], "feature_schema_version": "1.0.0"}
    ]
    assert stamps[2] in caplog.text and "invalid_input" in caplog.text


def test_non_finite_middle_point_skips_only_affected_observations(long_history):
    app, config, stamps = long_history
    set_value(SOURCE, "row_count", stored_ts(SOURCE)[2], float("inf"))
    client = RecordingClient(app.config["INFERENCE_HTTP_CLIENT"])
    adapter.score_project(SOURCE["project_id"], client=client)
    # The point and its successor carry a non-finite value or delta.
    assert saved(config["bindings"][0]) == [stamps[1], stamps[4]]
    assert [sent[1] for sent in client.sent] == [stamps[1], stamps[4]]


@pytest.mark.parametrize("metric", ["row_count", "null_rate"])
def test_text_middle_point_skips_only_affected_observations(long_history, metric):
    app, config, stamps = long_history
    set_value(SOURCE, metric, stored_ts(SOURCE, metric)[2], "not-a-number")
    client = RecordingClient(app.config["INFERENCE_HTTP_CLIENT"])
    adapter.score_project(SOURCE["project_id"], client=client)
    assert saved(config["bindings"][0]) == [stamps[1], stamps[4]]
    assert [sent[1] for sent in client.sent] == [stamps[1], stamps[4]]
    # An older corrupt point must not prevent a valid latest UI prediction.
    result = adapter.predict(SOURCE, config["bindings"][0]["model_version"], "1.0.0", client=client)
    assert result["observed_at"] == stamps[4]


@pytest.mark.parametrize("value", ["not-a-number", float("inf"), float("-inf")])
def test_metrics_rejects_invalid_stored_numbers_with_strict_json(ui, value):
    _, http, _ = ui
    set_value(SOURCE, "row_count", stored_ts(SOURCE)[-1], value)
    response = http.get("/api/inference/metrics", query_string=SOURCE)
    assert response.status_code == 422
    assert response.json == {"error": {"code": "invalid_history", "message": "invalid_history"}}
    json.dumps(response.json, allow_nan=False)
    assert "not-a-number" not in response.text and "Infinity" not in response.text


def test_metrics_rejects_postgres_nan_with_strict_json(ui, monkeypatch):
    _, http, _ = ui
    original = source_storage.metrics

    def with_nan(*args, **kwargs):
        rows = original(*args, **kwargs)
        return [*rows[:-1], {**rows[-1], "value": float("nan")}]

    monkeypatch.setattr(source_storage, "metrics", with_nan)
    response = http.get("/api/inference/metrics", query_string=SOURCE)
    assert response.status_code == 422
    json.dumps(response.json, allow_nan=False)


@pytest.mark.parametrize("metric", ["row_count", "null_rate"])
@pytest.mark.parametrize("value", ["not-a-number", float("inf"), float("-inf"), float("nan")])
def test_explanation_rejects_invalid_context_numbers(ui, monkeypatch, metric, value):
    from app import api, llm

    app, http, _ = ui
    app.config["PROPAGATE_EXCEPTIONS"] = False
    monkeypatch.setattr(llm.settings, "NIM_API_KEY", "")
    stamp = stored_ts(SOURCE, metric)[-1]
    if isinstance(value, float) and math.isnan(value):
        # SQLite cannot store NaN in this NOT NULL column; emulate a PG row.
        original = source_storage.metrics

        def with_nan(source, name, *args, **kwargs):
            rows = original(source, name, *args, **kwargs)
            if name == metric:
                rows[-1] = {**rows[-1], "value": value}
            return rows

        monkeypatch.setattr(source_storage, "metrics", with_nan)
    else:
        set_value(SOURCE, metric, stamp, value)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid context must not read/write the cache or call the LLM")

    monkeypatch.setattr(api, "get_cached_explanation", forbidden)
    monkeypatch.setattr(api, "save_explanation", forbidden)
    monkeypatch.setattr(llm, "explain_anomaly", forbidden)
    response = http.post(
        "/api/explain",
        json={"source": SOURCE, "table": "orders", "metric": metric, "ts": stamp},
    )
    assert response.status_code == 422
    assert response.json == {"error": "invalid_history"}
    with storage.get_engine().connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM llm_explanations")).scalar() == 0


def test_source_level_refusal_stops_remaining_observations(long_history):
    app, config, stamps = long_history
    client = RecordingClient(app.config["INFERENCE_HTTP_CLIENT"], reject=stamps[2], status=503)
    adapter.score_project(SOURCE["project_id"], client=client)
    # model_unavailable concerns the whole source, so later points are not retried.
    assert [sent[1] for sent in client.sent] == [stamps[1], stamps[2]]
    assert saved(config["bindings"][0]) == [stamps[1]]


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_version", "models/registry/if-demo"),  # a path instead of an ID
        ("model_version", "legacy-unknown"),
        ("feature_schema_version", "1.0"),
    ],
)
def test_invalid_binding_is_a_source_error(two_sources, caplog, field, value):
    app, config, first = two_sources
    add_history(first["source"], [75, 60, 45], [70, 80, 90])
    first[field] = value
    caplog.set_level(logging.WARNING)
    client = RecordingClient(app.config["INFERENCE_HTTP_CLIENT"])
    for _ in range(2):  # every run reports it once and sends nothing for the source
        adapter.score_project(SOURCE["project_id"], client=client)
    assert {sent[0] for sent in client.sent} == {"connection-a"}
    assert source_storage.versions(first["source"]) == []
    healthy = config["bindings"][1]
    assert len(source_storage.scores(SOURCE, healthy["model_version"], "1.0.0")) == 1
    bad = [r for r in caplog.records if "connection-b" in r.getMessage()]
    assert [r.levelno for r in bad] == [logging.ERROR] * 2
    assert all(r.exc_info[1].code == "invalid_binding" for r in bad)


@pytest.mark.parametrize("entry_point", ["scheduler", "per_project"])
def test_scheduler_logs_aggregated_source_failures(two_sources, monkeypatch, caplog, entry_point):
    from collectors import per_project, scheduler

    app, config, first = two_sources
    break_allowlist(first["source"])
    real = adapter.score_project
    monkeypatch.setattr(
        adapter, "score_project", lambda pid: real(pid, client=app.config["INFERENCE_HTTP_CLIENT"])
    )
    monkeypatch.setattr(adapter, "train_candidates", lambda pid: [])
    caplog.set_level(logging.ERROR)
    if entry_point == "scheduler":
        monkeypatch.setattr(scheduler, "_iter_retrain_projects", lambda: [SOURCE["project_id"]])
        scheduler.retrain_anomaly_detectors()
    else:
        per_project._maybe_notify_anomalies(SOURCE["project_id"], ["orders"])
    # One unexpected failure gives exactly one error record, with its traceback.
    (record,) = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert record.getMessage() == f"[project={SOURCE['project_id']}] published inference failed"
    group = record.exc_info[1]
    assert isinstance(group, ExceptionGroup)
    assert [type(e) for e in group.exceptions] == [json.JSONDecodeError]
    healthy = config["bindings"][1]
    assert len(source_storage.scores(SOURCE, healthy["model_version"], "1.0.0")) == 1


@pytest.mark.parametrize("explicit", [False, True])
def test_pr39_explain_without_inference_bindings(ui, monkeypatch, explicit):
    from app import llm

    _, http, _ = ui
    monkeypatch.delenv("INFERENCE_ADAPTER_CONFIG")
    monkeypatch.setattr(llm.settings, "NIM_API_KEY", "")
    stamp = source_storage.metrics(SOURCE, "row_count")[-1]["ts"]
    body = {"table": "orders", "metric": "row_count", "ts": stamp}
    if explicit:
        body["source"] = SOURCE
    response = http.post("/api/explain", json=body)
    assert response.status_code == 200, response.text
    with storage.get_engine().connect() as conn:
        cached = conn.execute(text("SELECT project_id, connection_id FROM llm_explanations")).one()
    assert tuple(cached) == (SOURCE["project_id"], SOURCE["connection_id"])


def test_pr39_stale_connection_does_not_own_new_history(ui, monkeypatch):
    from app import llm

    _, http, _ = ui
    monkeypatch.setattr(llm.settings, "NIM_API_KEY", "")
    stamp = source_storage.metrics(SOURCE, "row_count")[-1]["ts"]
    storage.save_metrics(
        [{"table_name": "orders", "metric_name": "row_count", "ts": stamp, "value": 999}],
        SOURCE["project_id"],
        connection_id="deleted-connection",
        schema_name="public",
    )
    assert (
        source_storage.context_source(SOURCE["project_id"], "orders", stamp, "row_count") == SOURCE
    )
    body = {"table": "orders", "metric": "row_count", "ts": stamp}
    assert http.post("/api/explain", json=body).status_code == 200
    body["source"] = {**SOURCE, "connection_id": "deleted-connection"}
    assert http.post("/api/explain", json=body).status_code == 403


@pytest.mark.parametrize("metric", ["row_count", "null_rate"])
def test_pr39_old_duplicate_only_quarantines_affected_pairs(long_history, metric):
    app, config, stamps = long_history
    old = datetime.fromisoformat(stamps[1]).isoformat(timespec="seconds")
    storage.save_metrics(
        [{"ts": old, "table_name": "orders", "metric_name": metric, "value": 999}],
        SOURCE["project_id"],
        connection_id=SOURCE["connection_id"],
        schema_name="public",
    )
    client = RecordingClient(app.config["INFERENCE_HTTP_CLIENT"])
    adapter.score_project(SOURCE["project_id"], client=client)
    assert saved(config["bindings"][0]) == stamps[3:]
    assert [stamp for _, stamp in client.sent] == stamps[3:]


@pytest.mark.parametrize("route", ["predict", "metrics", "explain"])
def test_pr39_invalid_timestamp_is_controlled(ui, route):
    app, http, config = ui
    app.config["PROPAGATE_EXCEPTIONS"] = False
    stamp = stored_ts(SOURCE)[-1]
    with storage.get_engine().begin() as conn:
        conn.execute(text("UPDATE metrics SET ts='broken-timestamp' WHERE ts=:ts"), {"ts": stamp})
    if route == "predict":
        response = http.post("/api/inference/predict", json=config["bindings"][0])
    elif route == "metrics":
        response = http.get("/api/inference/metrics", query_string=SOURCE)
    else:
        response = http.post(
            "/api/explain",
            json={"source": SOURCE, "table": "orders", "metric": "row_count", "ts": stamp},
        )
    assert response.status_code == 422, response.text
    error = response.json["error"]
    assert (error["code"] if isinstance(error, dict) else error) == "invalid_history"


def test_pr39_background_skips_malformed_timestamp(long_history):
    app, config, stamps = long_history
    with storage.get_engine().begin() as conn:
        conn.execute(
            text("UPDATE metrics SET ts='broken-timestamp' WHERE ts=:ts"),
            {"ts": stored_ts(SOURCE)[0]},
        )
    adapter.score_project(SOURCE["project_id"], client=app.config["INFERENCE_HTTP_CLIENT"])
    assert saved(config["bindings"][0]) == stamps[2:]


def test_pr39_incremental_scoring_and_notifications(long_history, monkeypatch):
    from app.notifications import telegram

    app, config, stamps = long_history
    notifications = []
    monkeypatch.setattr(
        telegram, "load_project_telegram_config", lambda pid: ("synthetic", "chat", 1)
    )
    monkeypatch.setattr(
        telegram, "notify_anomaly", lambda *args, **kwargs: notifications.append(args)
    )

    class Anomalies(RecordingClient):
        def post(self, url, json=None, **kwargs):
            self.sent.append((json["source"]["connection_id"], json["observed_at"]))
            return httpx.Response(
                200,
                json={
                    **{
                        key: json[key]
                        for key in (
                            "source",
                            "model_version",
                            "feature_schema_version",
                            "observed_at",
                        )
                    },
                    "score": -0.5,
                    "threshold": 0.0,
                    "is_anomaly": True,
                    "demo": True,
                },
            )

    client = Anomalies(None)
    entry = config["bindings"][0]
    adapter.score_project(SOURCE["project_id"], client=client)
    # Every observation is scored; backfill notifies only the current one.
    assert len(client.sent) == len(stamps) - 1
    assert [args[2] for args in notifications] == [stamps[-1]]
    adapter.score_project(SOURCE["project_id"], client=client)
    assert len(client.sent) == len(stamps) - 1
    assert len(notifications) == 1
    assert (
        source_storage.insert_prediction(
            {
                **source_storage.scores(SOURCE, entry["model_version"], "1.0.0")[-1],
                "observed_at": stamps[-1],
            }
        )
        is False
    )
    # Main screens see exactly the currently configured pair, with legacy separate.
    storage.save_anomaly_scores(
        [{"ts": stamps[-1], "table_name": "orders", "score": -9, "is_anomaly": 1}],
        SOURCE["project_id"],
    )
    active = storage.active_anomaly_scores(SOURCE["project_id"])
    assert len(active) == len(stamps) - 1
    assert all(r["model_version"] == entry["model_version"] for r in active)
    aggregate = storage.build_history_aggregate(SOURCE["project_id"])
    assert sum(aggregate["anomalies_by_ts"].values()) == len(stamps) - 1
    assert all(count == 1 for count in aggregate["anomalies_by_ts"].values())
    assert len(aggregate["timestamps"]) == len(stamps)
    # A new model must backfill independently; rollback must reuse prior results.
    filename = Path(os.environ["INFERENCE_ADAPTER_CONFIG"])
    config["bindings"][0]["model_version"] = "new-version"
    filename.write_text(json.dumps(config))
    adapter.score_project(SOURCE["project_id"], client=client)
    assert len(client.sent) == 2 * (len(stamps) - 1)
    assert {r["model_version"] for r in storage.active_anomaly_scores(SOURCE["project_id"])} == {
        "new-version"
    }


@pytest.mark.parametrize("route", ["/v1/predict", "/v1/model"])
@pytest.mark.parametrize(
    "raw,status",
    [("[" * 2000 + "]" * 2000, 400), ('{"x":"' + "a" * 65536 + '"}', 413)],
    ids=["too-deep", "too-large"],
)
def test_pr39_request_resource_limits(client, route, raw, status):
    response = client.post(
        route, content=raw, headers={**HEADERS, "Content-Type": "application/json"}
    )
    assert response.status_code == status
    assert set(response.json()) == {"error"}


@pytest.mark.parametrize(
    "value",
    ["2026-10-30 12:31:00Z", "20261030T123100Z", "2026-10-30T12:31:00", "2026-10-30T25:00:00Z"],
)
def test_pr39_strict_rfc3339(value):
    from jsonschema import ValidationError

    with pytest.raises(ValidationError):
        validator("Timestamp").validate(value)


def test_pr39_viewer_cannot_persist_predictions(ui):
    _, http, config = ui
    storage.create_user("viewer", "viewer@example.test", "unused")
    storage.add_project_member(SOURCE["project_id"], "viewer", "viewer")
    with http.session_transaction() as session:
        session["_user_id"] = "viewer"
    assert http.post("/api/inference/predict", json=config["bindings"][0]).status_code == 403
    assert source_storage.versions(SOURCE) == []


def test_pr39_legacy_llm_uses_typed_time_and_score(monkeypatch):
    from app import llm

    stamp = "2026-10-30T12:31:31+00:00"
    monkeypatch.setattr(llm, "get_schema_snapshot", lambda *a, **kw: [])
    monkeypatch.setattr(llm, "get_changepoints", lambda *a, **kw: [])
    monkeypatch.setattr(
        llm,
        "get_metrics",
        lambda *a, **kw: [
            {"ts": "2026-10-30T12:30:31.000000+00:00", "value": 100},
            {"ts": "2026-10-30T12:31:31.000000+00:00", "value": 7},
            {"ts": "2026-10-30T12:32:31+00:00", "value": 999},
        ],
    )
    monkeypatch.setattr(
        llm,
        "get_anomaly_scores",
        lambda *a, **kw: [
            {"ts": "2026-10-30T15:31:31.000000+03:00", "score": -0.1234, "is_anomaly": True}
        ],
    )
    prompt = llm._build_prompt("orders", "row_count", stamp)
    assert "12:31 UTC: 7" in prompt and "-0.1234" in prompt
    assert "999" not in prompt
    assert "с 100 до 7" in llm._rule_based_explain("orders", "row_count", stamp)["explanation"]


def test_pr39_training_one_source_does_not_block_second(two_sources, tmp_path, monkeypatch):
    _, config, first = two_sources
    config["candidate_registry_path"] = str(tmp_path / "candidates")
    Path(os.environ["INFERENCE_ADAPTER_CONFIG"]).write_text(json.dumps(config))
    calls = []

    def train(root, *, source, **kwargs):
        calls.append(source)
        if source == first["source"]:
            raise registry.RegistryError("insufficient data")
        return {"source": source, "model_version": "candidate"}

    monkeypatch.setattr(registry, "train_candidate", train)
    assert adapter.train_candidates(SOURCE["project_id"]) == [
        {"source": SOURCE, "model_version": "candidate"}
    ]
    assert calls == [first["source"], SOURCE]


def test_pr39_delete_project_cleans_scoped_data_only(ui):
    _, _, config = ui
    stamp = source_storage.metrics(SOURCE, "row_count")[-1]["ts"]
    source_storage.save_snapshot(SOURCE, [])
    storage.save_explanation(
        "orders",
        "row_count",
        stamp,
        "test",
        "fix",
        0.5,
        project_id=SOURCE["project_id"],
        connection_id=SOURCE["connection_id"],
        schema_name="public",
    )
    storage.save_explanation("orders", "row_count", stamp, "other", "fix", 0.5, project_id="other")
    storage.save_anomaly_scores(
        [{"ts": stamp, "table_name": "orders", "score": -1, "is_anomaly": True}],
        SOURCE["project_id"],
    )
    assert storage.delete_project("wrong-owner", SOURCE["project_id"]) is False
    assert storage.delete_project("u", SOURCE["project_id"]) is True
    with storage.get_engine().connect() as conn:
        from sqlalchemy import inspect

        for table in inspect(conn).get_table_names():
            columns = {c["name"] for c in inspect(conn).get_columns(table)}
            if "project_id" in columns:
                assert (
                    conn.execute(
                        text(f"SELECT COUNT(*) FROM {table} WHERE project_id=:pid"),
                        {"pid": SOURCE["project_id"]},
                    ).scalar()
                    == 0
                ), table
        assert (
            conn.execute(
                text("SELECT explanation FROM llm_explanations WHERE project_id='other'")
            ).scalar()
            == "other"
        )


@pytest.mark.parametrize(
    "raw", ["{broken", "[]", '{"bindings":null}', '{"bindings":[],"demo":"false"}']
)
def test_pr39_invalid_adapter_configuration_is_controlled(ui, raw, caplog):
    _, http, _ = ui
    Path(os.environ["INFERENCE_ADAPTER_CONFIG"]).write_text(raw)
    response = http.get("/api/inference/sources")
    assert response.status_code == 503
    assert response.json["error"]["code"] == "invalid_configuration"
    assert raw not in response.text
    assert "Invalid inference adapter configuration" in caplog.text


def test_pr39_bad_binding_in_another_project_is_isolated(ui):
    _, http, config = ui
    config["bindings"].insert(0, {"source": {"project_id": "other-project"}})
    Path(os.environ["INFERENCE_ADAPTER_CONFIG"]).write_text(json.dumps(config))
    assert http.get("/api/inference/sources").status_code == 200
    with pytest.raises(adapter.AdapterError) as error:
        adapter.bindings("other-project")
    assert error.value.code == "invalid_configuration"


def test_pr39_bad_configuration_does_not_abort_other_project_jobs(ui, monkeypatch):
    from collectors import scheduler
    from ml import anomaly_detector

    calls = []
    monkeypatch.setattr(scheduler, "_iter_retrain_projects", lambda: ["bad", "good"])
    monkeypatch.setattr(storage, "list_metric_tables", lambda pid: ["orders"])
    original = adapter.bindings

    def bindings(pid):
        if pid == "bad":
            raise adapter.AdapterError(503, "invalid_configuration")
        return original(pid)

    monkeypatch.setattr(adapter, "bindings", bindings)
    monkeypatch.setattr(
        anomaly_detector, "retrain_all", lambda **kwargs: calls.append(kwargs["project_id"]) or {}
    )
    monkeypatch.setattr(anomaly_detector, "score_table", lambda *a, **kw: [])
    scheduler.retrain_anomaly_detectors()
    assert calls == ["good"]


def test_pr39_delete_is_transactional(ui):
    source_storage.save_snapshot(SOURCE, [])
    with storage.get_engine().begin() as conn:
        conn.execute(
            text(
                "CREATE TRIGGER fail_delete BEFORE DELETE ON source_schema_snapshots "
                "BEGIN SELECT RAISE(ABORT, 'synthetic failure'); END"
            )
        )
    from sqlalchemy.exc import IntegrityError

    with pytest.raises(IntegrityError):
        storage.delete_project("u", SOURCE["project_id"])
    assert storage.get_project_by_id("u", SOURCE["project_id"])
    assert len(source_storage.metrics(SOURCE, "row_count")) == 2
    assert source_storage.snapshot(SOURCE) == []


def test_pr39_actual_training_continues_after_insufficient_source(
    two_sources, tmp_path, monkeypatch
):
    _, config, first = two_sources
    config["candidate_registry_path"] = str(tmp_path / "candidates")
    Path(os.environ["INFERENCE_ADAPTER_CONFIG"]).write_text(json.dumps(config))
    monkeypatch.setenv("INFERENCE_TRAINING_GIT_SHA", "a" * 40)
    add_history(SOURCE, list(range(15 * 206, 30, -15)), list(range(204)))
    candidates = adapter.train_candidates(SOURCE["project_id"])
    assert len(candidates) == 1
    assert candidates[0]["source"] == SOURCE
    assert candidates[0]["training"]["n_points"] >= 200
    assert not (tmp_path / "candidates" / "active").exists()


def test_pr39_service_configuration_and_framework_errors(client):
    with pytest.raises(ValueError, match="registry_path"):
        create_app({"principals": []})
    for method, path, status, code in [
        ("get", "/missing", 404, "not_found"),
        ("get", "/v1/predict", 405, "method_not_allowed"),
    ]:
        response = getattr(client, method)(path)
        assert response.status_code == status
        assert response.json() == {"error": {"code": code, "message": code}}


@pytest.mark.parametrize("age_days", [1, 2, 15])
def test_pr39_duplicates_on_older_dates_do_not_block_new_scores(ui, age_days):
    app, _, config = ui
    add_history(SOURCE, [age_days * 1440 + 15, age_days * 1440, 60], [70, 80, 90])
    stamps = [r["ts"] for r in source_storage.metrics(SOURCE, "row_count")]
    storage.save_metrics(
        [{"ts": stamps[1], "table_name": "orders", "metric_name": "row_count", "value": 999}],
        SOURCE["project_id"],
        connection_id=SOURCE["connection_id"],
        schema_name="public",
    )
    client = RecordingClient(app.config["INFERENCE_HTTP_CLIENT"])
    adapter.score_project(SOURCE["project_id"], client=client)
    assert saved(config["bindings"][0]) == stamps[3:]
    assert [ts for _, ts in client.sent] == stamps[3:]


@pytest.mark.parametrize("route", ["/dashboard/inference", "/api/anomalies/orders"])
def test_pr39_configuration_failure_outside_inference_blueprint(ui, route):
    app, http, _ = ui
    app.config["PROPAGATE_EXCEPTIONS"] = False
    Path(os.environ["INFERENCE_ADAPTER_CONFIG"]).write_text(
        "{broken configuration with private text"
    )
    response = http.get(route)
    assert response.status_code == 503
    assert response.json == {"error": "invalid_configuration"}
    assert "private text" not in response.text


@pytest.mark.parametrize("same_version", [True, False])
def test_pr39_concurrent_model_loads_are_shared_per_key(client, bundles, monkeypatch, same_version):
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    import inference.service as service

    original = service.load_version
    cold = []
    start = threading.Barrier(2)
    independent = threading.Barrier(2)

    def load(*args, **kwargs):
        if kwargs.get("cached") is None:
            cold.append(args[1])
            if same_version:
                time.sleep(0.1)  # expose two simultaneous cold loads of one artifact
            else:
                independent.wait(timeout=5)  # different keys must not share a load lock
        return original(*args, **kwargs)

    monkeypatch.setattr(service, "load_version", load)

    def request(index):
        start.wait(timeout=5)
        return client.post("/v1/predict", json=payload(bundles, version=index), headers=HEADERS)

    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(request, [0, 0 if same_version else 1]))
    assert [r.status_code for r in responses] == [200, 200]
    assert len(cold) == (1 if same_version else 2)


# --- Independent review of e8e9f03: Н-1..Н-10 ------------------------------


def legacy_point(project_id=SOURCE["project_id"], table="orders", age=timedelta(days=3), value=70):
    """A pre-update observation: seconds precision and no attributed owner."""
    stamp = (datetime.now(UTC) - age).replace(microsecond=0).isoformat()
    for metric, number in [("row_count", value), ("null_rate", 0.5)]:
        storage.save_metrics(
            [{"ts": stamp, "table_name": table, "metric_name": metric, "value": number}],
            project_id,
        )
    return stamp


def capture_prompts(monkeypatch):
    from app import llm

    prompts = []

    def nim(prompt):
        prompts.append(prompt)
        return json.dumps({"explanation": "llm", "suggested_fix": "check", "confidence": 0.8})

    monkeypatch.setattr(storage.settings, "NIM_API_KEY", "synthetic")
    monkeypatch.setattr(llm, "_call_nim", nim)
    return prompts


def explain_body(stamp, **extra):
    return {"table": "orders", "metric": "row_count", "ts": stamp, **extra}


def score_line(stamp):
    return f"Isolation Forest score at {stamp[:16].replace('T', ' ')} UTC: "


@pytest.mark.parametrize("bindings", [True, False], ids=["bindings", "no-bindings"])
def test_review_old_anomaly_is_explained_after_update(ui, monkeypatch, bindings):
    _, http, _ = ui
    prompts = capture_prompts(monkeypatch)
    if not bindings:
        monkeypatch.delenv("INFERENCE_ADAPTER_CONFIG")
    stamp = legacy_point()
    storage.save_anomaly_scores(
        [{"ts": stamp, "table_name": "orders", "score": -0.4321, "is_anomaly": 1}],
        SOURCE["project_id"],
    )
    response = http.post("/api/explain", json=explain_body(stamp))
    assert response.status_code == 200, response.text
    assert score_line(stamp) + "-0.4321" in prompts[0]
    with storage.get_engine().connect() as conn:
        # Neither the history nor its explanation acquires an unconfirmed owner.
        assert conn.execute(text("SELECT DISTINCT connection_id FROM llm_explanations")).scalar() == (
            "legacy-unknown"
        )
        assert (
            conn.execute(
                text(
                    "SELECT COUNT(*) FROM metrics WHERE ts = :ts AND connection_id = 'legacy-unknown'"
                ),
                {"ts": stamp},
            ).scalar()
            == 2
        )
    assert source_storage.context_source(SOURCE["project_id"], "orders", stamp, "row_count") is None


def test_review_new_anomaly_in_mixed_history_keeps_its_source(ui):
    _, http, _ = ui
    legacy_point()
    stamp = source_storage.metrics(SOURCE, "row_count")[-1]["ts"]
    assert (
        source_storage.context_source(SOURCE["project_id"], "orders", stamp, "row_count") == SOURCE
    )
    assert http.post("/api/explain", json=explain_body(stamp)).status_code == 200
    with storage.get_engine().connect() as conn:
        assert conn.execute(text("SELECT connection_id FROM llm_explanations")).scalar() == (
            SOURCE["connection_id"]
        )


def test_review_old_anomaly_is_not_ambiguous_with_same_named_sources(two_sources, ui, monkeypatch):
    _, http, _ = ui
    prompts = capture_prompts(monkeypatch)
    stamp = legacy_point(value=71)
    assert http.post("/api/explain", json=explain_body(stamp)).status_code == 200
    assert ": 71" in prompts[0]
    assert ": 100" not in prompts[0] and ": 110" not in prompts[0]


def test_review_same_named_sources_at_one_time_still_need_a_choice(two_sources, ui):
    _, http, _ = ui
    stamp = source_storage.metrics(SOURCE, "row_count")[-1]["ts"]
    response = http.post("/api/explain", json=explain_body(stamp))
    assert response.status_code == 409
    assert response.json == {"error": "source_required"}


def test_review_forbidden_observation_stays_forbidden(ui, monkeypatch):
    from app import llm

    _, http, _ = ui
    monkeypatch.setattr(llm.settings, "NIM_API_KEY", "")
    stamp = (datetime.now(UTC) - timedelta(days=2)).isoformat()
    storage.save_metrics(
        [{"ts": stamp, "table_name": "orders", "metric_name": "row_count", "value": 999}],
        SOURCE["project_id"],
        connection_id="deleted-connection",
        schema_name="public",
    )
    response = http.post("/api/explain", json=explain_body(stamp))
    assert response.status_code == 403
    assert response.json == {"error": "source_forbidden"}


def test_review_legacy_context_excludes_forbidden_rows_at_same_time(ui, monkeypatch):
    _, http, _ = ui
    prompts = capture_prompts(monkeypatch)
    stamp = legacy_point(value=71)
    storage.save_metrics(
        [{"ts": stamp, "table_name": "orders", "metric_name": "row_count", "value": 999}],
        SOURCE["project_id"],
        connection_id="deleted-connection",
        schema_name="public",
    )
    assert http.post("/api/explain", json=explain_body(stamp)).status_code == 200
    assert ": 71" in prompts[0]
    assert "999" not in prompts[0]


def test_review_unknown_time_is_not_reported_as_forbidden(ui, monkeypatch):
    from app import llm

    _, http, _ = ui
    monkeypatch.setattr(llm.settings, "NIM_API_KEY", "")
    missing = (datetime.now(UTC) - timedelta(days=9)).isoformat()
    assert http.post("/api/explain", json=explain_body(missing)).status_code == 404
    with storage.get_engine().connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM llm_explanations")).scalar() == 0


def test_review_scores_keep_old_anomaly_after_metrics_retention(ui):
    _, http, _ = ui
    stamp = (datetime.now(UTC) - timedelta(days=3)).replace(microsecond=0).isoformat()
    storage.save_anomaly_scores(
        [{"ts": stamp, "table_name": "orders", "score": -0.4321, "is_anomaly": 1}],
        SOURCE["project_id"],
    )
    assert http.post("/api/explain", json=explain_body(stamp)).status_code == 200


def test_review_foreign_project_history_is_invisible(ui):
    _, http, _ = ui
    storage.create_project("project-b", "u", "B", "b")
    stamp = legacy_point(project_id="project-b", table="invoices")
    assert http.post("/api/explain", json={**explain_body(stamp), "table": "invoices"}).status_code == 404
    # The caller is not a member of this project even when the row is legacy.
    storage.create_user("other", "other@example.test", "unused")
    storage.create_project("project-c", "other", "C", "c")
    stamp = legacy_point(project_id="project-c", table="invoices")
    assert http.post("/api/explain", json={**explain_body(stamp), "table": "invoices"}).status_code == 404


def test_review_explicit_source_outside_default_window_is_found(ui, monkeypatch):
    from app import llm

    _, http, _ = ui
    monkeypatch.setattr(llm.settings, "NIM_API_KEY", "")
    stamp = (datetime.now(UTC) - timedelta(days=45)).isoformat()
    other = {**SOURCE, "table_name": "archive"}
    for metric in ("row_count", "null_rate"):
        storage.save_metrics(
            [{"ts": stamp, "table_name": "archive", "metric_name": metric, "value": 1}],
            SOURCE["project_id"],
            connection_id=SOURCE["connection_id"],
            schema_name="public",
        )
    body = {"source": other, "table": "archive", "metric": "row_count", "ts": stamp}
    assert http.post("/api/explain", json=body).status_code == 200


def changepoint(stamp, before=100.0, after=500.0):
    return {
        "ts": stamp,
        "table_name": "orders",
        "metric_name": "row_count",
        "score": 3.0,
        "value_before": before,
        "value_after": after,
    }


def test_review_legacy_score_and_changepoint_reach_the_prompt(ui, monkeypatch):
    """Standard path: collector history with an owner, legacy detector results."""
    from app import llm

    _, http, _ = ui
    prompts = capture_prompts(monkeypatch)
    monkeypatch.delenv("INFERENCE_ADAPTER_CONFIG")
    stamp = source_storage.metrics(SOURCE, "row_count")[-1]["ts"]
    earlier = (datetime.fromisoformat(stamp) - timedelta(days=1)).isoformat()
    storage.save_anomaly_scores(
        [{"ts": stamp, "table_name": "orders", "score": -0.4321, "is_anomaly": 1}],
        SOURCE["project_id"],
    )
    storage.save_changepoints([changepoint(earlier)], project_id=SOURCE["project_id"])

    assert http.post("/api/explain", json=explain_body(stamp)).status_code == 200
    # The notification path enters without an HTTP request or a cached result.
    llm.explain_anomaly("orders", "row_count", stamp, SOURCE["project_id"])

    assert len(prompts) == 2
    for prompt in prompts:
        assert score_line(stamp) + "-0.4321" in prompt
        assert "row_count: 100.00 → 500.00" in prompt


def fixed_scores(score=-0.5, anomalous=lambda body: True, sent=None):
    """HTTP boundary answering with a chosen score, keeping the wire identity."""

    class Client:
        def __init__(self):
            self.sent = [] if sent is None else sent

        def post(self, url, json=None, **kwargs):
            self.sent.append((json["source"]["connection_id"], json["observed_at"]))
            identity = ("source", "model_version", "feature_schema_version", "observed_at")
            value = score(json) if callable(score) else score
            return httpx.Response(
                200,
                json={
                    **{key: json[key] for key in identity},
                    "score": value,
                    "threshold": 0.0,
                    "is_anomaly": anomalous(json),
                    "demo": True,
                },
            )

    return Client()


def test_review_active_model_score_reaches_the_prompt(ui, monkeypatch):
    _, http, config = ui
    prompts = capture_prompts(monkeypatch)
    adapter.score_project(SOURCE["project_id"], client=fixed_scores(-0.1234))
    stamp = source_storage.metrics(SOURCE, "row_count")[-1]["ts"]
    # Same observation scored by another version and the legacy detector.
    source_storage.save_prediction(
        {
            "source": SOURCE,
            "model_version": "other-version",
            "feature_schema_version": "1.0.0",
            "observed_at": stamp,
            "score": -0.9999,
            "is_anomaly": True,
            "demo": True,
        }
    )
    storage.save_anomaly_scores(
        [{"ts": stamp, "table_name": "orders", "score": -0.8888, "is_anomaly": 1}],
        SOURCE["project_id"],
    )
    assert http.post("/api/explain", json=explain_body(stamp, source=SOURCE)).status_code == 200
    assert score_line(stamp) + "-0.1234" in prompts[0]
    assert "-0.9999" not in prompts[0] and "-0.8888" not in prompts[0]


def test_review_source_context_never_uses_other_connection(two_sources, ui, monkeypatch):
    _, http, _ = ui
    app, config, first = two_sources
    prompts = capture_prompts(monkeypatch)
    other = first["source"]
    scores = {SOURCE["connection_id"]: -0.1111, other["connection_id"]: -0.2222}
    client = fixed_scores(lambda body: scores[body["source"]["connection_id"]])
    adapter.score_project(SOURCE["project_id"], client=client)
    stamp = source_storage.metrics(SOURCE, "row_count")[-1]["ts"]
    storage.save_anomaly_scores(
        [{"ts": stamp, "table_name": "orders", "score": -0.8888, "is_anomaly": 1}],
        SOURCE["project_id"],
    )
    storage.save_changepoints(
        [changepoint((datetime.fromisoformat(stamp) - timedelta(days=1)).isoformat())],
        project_id=SOURCE["project_id"],
    )
    for source, expected in [(SOURCE, "-0.1111"), (other, "-0.2222")]:
        body = explain_body(stamp, source=source)
        assert http.post("/api/explain", json=body).status_code == 200
        assert score_line(stamp) + expected in prompts[-1]
        # Change-points have no source key: two owners make them unattributable.
        assert "500.00" not in prompts[-1]
        assert "-0.8888" not in prompts[-1]


def test_review_legacy_score_needs_one_attributed_owner(ui, monkeypatch):
    _, http, _ = ui
    prompts = capture_prompts(monkeypatch)
    monkeypatch.delenv("INFERENCE_ADAPTER_CONFIG")
    storage.create_connection(
        connection_id="connection-b",
        project_id=SOURCE["project_id"],
        name="B",
        dsn_encrypted=b"unused",
    )
    add_history({**SOURCE, "connection_id": "connection-b"}, [7], [999])
    stamp = source_storage.metrics(SOURCE, "row_count")[-1]["ts"]
    storage.save_anomaly_scores(
        [{"ts": stamp, "table_name": "orders", "score": -0.4321, "is_anomaly": 1}],
        SOURCE["project_id"],
    )
    storage.save_changepoints(
        [changepoint((datetime.fromisoformat(stamp) - timedelta(days=1)).isoformat())],
        project_id=SOURCE["project_id"],
    )
    assert http.post("/api/explain", json=explain_body(stamp)).status_code == 200
    assert score_line(stamp) + "not available" in prompts[0]
    assert "500.00" not in prompts[0]


def test_review_missing_score_stays_not_available(ui, monkeypatch):
    _, http, _ = ui
    prompts = capture_prompts(monkeypatch)
    stamp = source_storage.metrics(SOURCE, "row_count")[-1]["ts"]
    assert http.post("/api/explain", json=explain_body(stamp, source=SOURCE)).status_code == 200
    assert score_line(stamp) + "not available" in prompts[0]


def test_review_future_changepoint_is_not_context(ui, monkeypatch):
    _, http, _ = ui
    prompts = capture_prompts(monkeypatch)
    monkeypatch.delenv("INFERENCE_ADAPTER_CONFIG")
    stamp = source_storage.metrics(SOURCE, "row_count")[0]["ts"]
    later = (datetime.fromisoformat(stamp) + timedelta(minutes=5)).isoformat()
    storage.save_changepoints([changepoint(later)], project_id=SOURCE["project_id"])
    assert http.post("/api/explain", json=explain_body(stamp)).status_code == 200
    assert "500.00" not in prompts[0]


@pytest.fixture
def alerts(monkeypatch):
    """Real notify_anomaly and throttle; only the Telegram transport is replaced."""
    from app.notifications import telegram

    sent = []
    monkeypatch.setattr(
        telegram, "load_project_telegram_config", lambda pid: ("synthetic", "chat", 30)
    )
    monkeypatch.setattr(
        telegram, "send_message", lambda body, **kwargs: sent.append(body) or (True, None)
    )
    return sent


def clear_throttle():
    with storage.get_engine().begin() as conn:
        conn.execute(text("DELETE FROM telegram_throttle"))


def alert_time(body):
    return re.search(r"Время: (.+) UTC", body).group(1)


def minute(stamp):
    return stamp[:16].replace("T", " ")


def replace_history(source, minutes_back):
    with storage.get_engine().begin() as conn:
        conn.execute(text("DELETE FROM metrics"))
    add_history(source, minutes_back, [100 + i for i in range(len(minutes_back))])
    return [row["ts"] for row in source_storage.metrics(source, "row_count")]


def tick(project_id=SOURCE["project_id"], client=None):
    client = client or fixed_scores()
    adapter.score_project(project_id, client=client)
    return client


def test_review_first_backfill_alerts_only_the_current_observation(ui, alerts):
    stamps = replace_history(SOURCE, [15 * i for i in range(20, 0, -1)])
    client = tick()
    assert len(client.sent) == len(stamps) - 1
    assert [alert_time(body) for body in alerts] == [minute(stamps[-1])]


def test_review_backfill_of_stale_history_is_silent(ui, alerts):
    stamps = replace_history(SOURCE, [24 * 60 * i for i in range(20, 0, -1)])
    client = tick()
    assert len(client.sent) == len(stamps) - 1
    assert alerts == []


def test_review_new_anomaly_after_backfill_is_delivered(ui, alerts):
    replace_history(SOURCE, [24 * 60 * i for i in range(20, 0, -1)])
    tick()
    add_history(SOURCE, [1], [999])
    tick()
    stamp = source_storage.metrics(SOURCE, "row_count")[-1]["ts"]
    assert [alert_time(body) for body in alerts] == [minute(stamp)]


def test_review_one_new_normal_result_is_silent(ui, alerts):
    stamps = replace_history(SOURCE, [60, 45, 30])
    tick(client=fixed_scores(anomalous=lambda body: body["observed_at"] != stamps[-1]))
    assert alerts == []


def test_review_several_new_results_alert_the_newest(ui, alerts):
    replace_history(SOURCE, [24 * 60 * i for i in range(5, 0, -1)])
    tick()
    add_history(SOURCE, [50, 40, 30], [1, 2, 3])
    tick()
    newest = source_storage.metrics(SOURCE, "row_count")[-1]["ts"]
    assert [alert_time(body) for body in alerts] == [minute(newest)]
    tick()
    assert len(alerts) == 1


def test_review_newest_anomalous_result_is_selected_by_canonical_time(ui, alerts):
    replace_history(SOURCE, [24 * 60 * i for i in range(5, 0, -1)])
    tick()
    add_history(SOURCE, [50, 40, 30], [1, 2, 3])
    rows = source_storage.metrics(SOURCE, "row_count")
    anomaly = rows[-2]["ts"]
    tick(client=fixed_scores(anomalous=lambda body: body["observed_at"] != rows[-1]["ts"]))
    assert [alert_time(body) for body in alerts] == [minute(anomaly)]


def test_review_version_switch_and_rollback_alert_only_fresh_events(ui, alerts):
    _, _, config = ui
    stamps = replace_history(SOURCE, [15 * i for i in range(20, 0, -1)])
    tick()
    assert [alert_time(body) for body in alerts] == [minute(stamps[-1])]
    clear_throttle()

    original = config["bindings"][0]["model_version"]
    filename = Path(os.environ["INFERENCE_ADAPTER_CONFIG"])

    def switch(version):
        config["bindings"][0]["model_version"] = version
        filename.write_text(json.dumps(config))

    switch("next-version")
    tick()  # Full recalculation with no new observation is not an event.
    assert len(alerts) == 1
    add_history(SOURCE, [1], [999])
    tick()
    fresh = source_storage.metrics(SOURCE, "row_count")[-1]["ts"]
    assert [alert_time(body) for body in alerts[1:]] == [minute(fresh)]
    clear_throttle()

    switch(original)
    tick()  # Rollback recalculates what the other version already reported.
    assert len(alerts) == 2


def test_review_stale_backfill_does_not_occupy_the_throttle_window(ui, alerts):
    from app.metrics_storage import is_throttled

    replace_history(SOURCE, [24 * 60 * i for i in range(20, 0, -1)])
    tick()
    assert not is_throttled(SOURCE["project_id"], "orders", "anomaly", throttle_minutes=30)
    with storage.get_engine().connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM telegram_throttle")).scalar() == 0


def test_review_second_event_inside_the_throttle_window_is_suppressed(ui, alerts):
    replace_history(SOURCE, [24 * 60 * i for i in range(5, 0, -1)])
    tick()
    add_history(SOURCE, [10], [1])
    tick()
    add_history(SOURCE, [5], [2])
    tick()
    assert len(alerts) == 1


def test_review_claimed_event_survives_source_failure_later_in_the_tick(ui, alerts):
    replace_history(SOURCE, [24 * 60 * i for i in range(5, 0, -1)])
    tick()
    add_history(SOURCE, [20, 10], [1, 2])
    rows = source_storage.metrics(SOURCE, "row_count")
    good = fixed_scores()

    class FailsOnLast:
        def post(self, url, json=None, **kwargs):
            if json["observed_at"] == rows[-1]["ts"]:
                return httpx.Response(503, json={"error": {"code": "down"}})
            return good.post(url, json=json, **kwargs)

    tick(client=FailsOnLast())
    assert [alert_time(body) for body in alerts] == [minute(rows[-2]["ts"])]


def run_concurrently(calls):
    import threading

    barrier = threading.Barrier(len(calls))
    results = [None] * len(calls)
    failures = []

    def worker(index, call):
        barrier.wait(timeout=10)
        try:
            results[index] = call()
        except BaseException as exc:  # Reported to the test below.
            failures.append(exc)

    threads = [
        threading.Thread(target=worker, args=(index, call)) for index, call in enumerate(calls)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert failures == []
    return results


def test_review_concurrent_ticks_alert_once(ui, alerts):
    replace_history(SOURCE, [24 * 60 * i for i in range(5, 0, -1)])
    tick()
    add_history(SOURCE, [20, 10], [1, 2])
    run_concurrently([tick, tick, tick, tick])
    assert len(alerts) == 1
    newest = source_storage.metrics(SOURCE, "row_count")[-1]["ts"]
    assert alert_time(alerts[0]) == minute(newest)


def test_review_worker_with_an_older_claim_yields_to_the_newest_anomaly(ui, alerts):
    stamps = replace_history(SOURCE, [20, 15, 10])
    entry = ui[2]["bindings"][0]
    older, newest = (
        {
            "source": SOURCE,
            "model_version": entry["model_version"],
            "feature_schema_version": "1.0.0",
            "observed_at": stamp,
            "score": -0.5,
            "is_anomaly": True,
            "demo": True,
        }
        for stamp in stamps[-2:]
    )
    assert source_storage.insert_prediction(older) and source_storage.insert_prediction(newest)
    adapter._alert_new_event(older, stamps[0])
    assert alerts == []
    adapter._alert_new_event(newest, stamps[0])
    assert [alert_time(body) for body in alerts] == [minute(stamps[-1])]


def test_review_atomic_claim_has_one_winner(ui):
    stamp = source_storage.metrics(SOURCE, "row_count")[-1]["ts"]
    result = {
        "source": SOURCE,
        "model_version": "claimed-version",
        "feature_schema_version": "1.0.0",
        "observed_at": stamp,
        "score": -0.5,
        "is_anomaly": True,
        "demo": True,
    }
    results = run_concurrently([lambda: source_storage.insert_prediction(result)] * 8)
    assert sorted(results) == [False] * 7 + [True]
    assert len(source_storage.scores(SOURCE, "claimed-version", "1.0.0")) == 1
    # A later contender must also lose and leave the stored result untouched.
    assert source_storage.insert_prediction({**result, "score": -0.9}) is False
    assert source_storage.scores(SOURCE, "claimed-version", "1.0.0")[0]["score"] == -0.5


@pytest.fixture
def same_named(two_sources, alerts):
    """Two connections of one project serve `orders`; both report one anomaly."""
    app, config, first = two_sources
    other = first["source"]
    for source in (SOURCE, other):
        with storage.get_engine().begin() as conn:
            conn.execute(text("DELETE FROM metrics WHERE connection_id = :id"), {"id": source["connection_id"]})
    add_history(SOURCE, [60, 30, 10], [100, 110, 500])
    add_history(other, [55, 25, 5], [200, 210, 900])
    return app, config, other


def test_review_same_named_connections_alert_independently(same_named, alerts, monkeypatch):
    prompts = capture_prompts(monkeypatch)
    monkeypatch.setenv("FF_LLM_NOTIFICATIONS", "1")
    _, config, other = same_named
    scores = {SOURCE["connection_id"]: -0.1111, other["connection_id"]: -0.2222}
    client = tick(client=fixed_scores(lambda body: scores[body["source"]["connection_id"]]))
    assert len(client.sent) == 4

    assert len(source_storage.scores(SOURCE, config["bindings"][1]["model_version"], "1.0.0")) == 2
    assert len(source_storage.scores(other, config["bindings"][0]["model_version"], "1.0.0")) == 2

    assert len(alerts) == 2
    by_connection = {
        next(
            key
            for key in (SOURCE["connection_id"], other["connection_id"])
            if key in body.split("\n\n")[0]
        ): body
        for body in alerts
    }
    assert set(by_connection) == set(scores)
    assert "Score: -0.1111" in by_connection[SOURCE["connection_id"]]
    assert "Score: -0.2222" in by_connection[other["connection_id"]]
    assert alerts[0].split("\n\n")[0] != alerts[1].split("\n\n")[0]

    assert len(prompts) == 2
    joined = {re.search(r"Isolation Forest score at .*: (\S+)", p).group(1): p for p in prompts}
    assert set(joined) == {"-0.1111", "-0.2222"}
    assert ": 500" in joined["-0.1111"] and ": 900" not in joined["-0.1111"]
    assert ": 900" in joined["-0.2222"] and ": 500" not in joined["-0.2222"]

    tick()
    assert len(alerts) == 2
    assert len(prompts) == 2


def test_review_throttle_is_scoped_by_full_source(ui, alerts):
    from app.notifications.telegram import notify_anomaly

    stamp = datetime.now(UTC).isoformat()
    config = {"bot_token": "x", "chat_id": "c", "throttle_minutes": 30}
    sources = [
        SOURCE,
        {**SOURCE, "connection_id": "connection-b"},
        {**SOURCE, "schema_name": "sales"},
    ]
    for source in sources:
        notify_anomaly(SOURCE["project_id"], "orders", stamp, -0.5, source=source, **config)
        notify_anomaly(SOURCE["project_id"], "orders", stamp, -0.5, source=source, **config)
    assert len(alerts) == 3
    notify_anomaly(
        SOURCE["project_id"], "orders", stamp, -0.5, metric="null_rate", source=SOURCE, **config
    )
    assert len(alerts) == 4


def test_review_legacy_notification_is_unchanged(ui, alerts, monkeypatch):
    from app.notifications.telegram import notify_anomaly

    stamp = datetime.now(UTC).isoformat()
    config = {"bot_token": "x", "chat_id": "c", "throttle_minutes": 30}
    notify_anomaly(SOURCE["project_id"], "orders", stamp, -0.5, **config)
    notify_anomaly(SOURCE["project_id"], "orders", stamp, -0.5, **config)
    assert len(alerts) == 1
    assert "Подключение" not in alerts[0]
    with storage.get_engine().connect() as conn:
        assert conn.execute(text("SELECT event_key FROM telegram_throttle")).scalar() == "anomaly"


def test_review_notification_passes_full_source_to_explanation(two_sources, alerts, monkeypatch):
    from app.notifications import telegram

    seen = []

    def explain(table, metric, ts, project_id="legacy", *, source=None):
        seen.append(source)
        return {"explanation": "x", "suggested_fix": "y", "confidence": 0.9}

    monkeypatch.setattr(telegram, "explain_anomaly", explain)
    monkeypatch.setattr(telegram, "is_enabled", lambda name: True)
    _, _, first = two_sources
    tick()
    assert sorted(source["connection_id"] for source in seen) == sorted(
        [SOURCE["connection_id"], first["source"]["connection_id"]]
    )


def test_review_quality_gate_compares_times_not_strings(ui, monkeypatch):
    from collectors.per_project import _latest_row_count_at, _passes_alert_quality_gate

    monkeypatch.setattr(storage.settings, "ANOMALY_NOTIFY_MIN_SCORE_MAGNITUDE", 0.01)
    monkeypatch.setattr(storage.settings, "ANOMALY_NOTIFY_MIN_DELTA_RATIO", 0.2)
    with storage.get_engine().begin() as conn:
        conn.execute(text("DELETE FROM metrics"))
    add_history(SOURCE, [40, 30, 20, 10], [100, 100, 100, 101])
    stored = stored_ts(SOURCE)[-1]
    assert stored.endswith(".000000+00:00")
    for form in (stored, stored.replace(".000000", ""), stored.replace("+00:00", "Z")):
        assert _latest_row_count_at("orders", SOURCE["project_id"], form) == 101
        assert not _passes_alert_quality_gate(
            "orders", {"ts": form, "score": -0.5}, SOURCE["project_id"]
        )
    assert _latest_row_count_at("orders", SOURCE["project_id"], "2001-01-01T00:00:00+00:00") is None


@pytest.mark.parametrize("value", [["x"], {"a": 1}, 5, 1.5, True, None], ids=str)
@pytest.mark.parametrize("field", ["model_version", "feature_schema_version"])
@pytest.mark.parametrize(
    "route",
    [
        "/api/anomalies/orders",
        "/dashboard/inference",
        "/api/inference/sources",
        "/dashboard/",
        "/dashboard/history",
    ],
)
def test_review_invalid_version_types_are_controlled(ui, field, value, route):
    app, http, config = ui
    app.config["PROPAGATE_EXCEPTIONS"] = False
    config["bindings"][0][field] = value
    Path(os.environ["INFERENCE_ADAPTER_CONFIG"]).write_text(json.dumps(config))
    response = http.get(route)
    assert response.status_code in {404, 503}, response.text
    if response.status_code == 503:
        error = response.json["error"]
        assert (error["code"] if isinstance(error, dict) else error) == "invalid_configuration"


def test_review_invalid_version_type_stays_inside_its_project(ui):
    _, http, config = ui
    config["bindings"].insert(
        0,
        {
            "source": {**SOURCE, "project_id": "other-project"},
            "model_version": ["x"],
            "feature_schema_version": {"a": 1},
        },
    )
    Path(os.environ["INFERENCE_ADAPTER_CONFIG"]).write_text(json.dumps(config))
    assert http.get("/api/inference/sources").status_code == 200
    with pytest.raises(adapter.AdapterError) as error:
        adapter.bindings("other-project")
    assert error.value.code == "invalid_configuration"


@pytest.mark.parametrize(
    "raw,status,code",
    [
        ("[" * 60000, 400, "invalid_json"),
        ('{"a":' * 100000 + "1" + "}" * 100000, 413, "request_too_large"),
        ('{"table":"' + "x" * 70000 + '"}', 413, "request_too_large"),
        ("[" * 40 + "]" * 40, 400, "invalid_json"),
    ],
    ids=["deep-open", "deep-oversized", "large", "just-too-deep"],
)
@pytest.mark.parametrize("route", ["/api/explain", "/api/inference/predict"])
def test_review_flask_json_limits(ui, route, raw, status, code):
    app, http, _ = ui
    app.config["PROPAGATE_EXCEPTIONS"] = False
    response = http.post(route, data=raw, content_type="application/json")
    assert response.status_code == status, response.text
    error = response.json["error"]
    assert (error["code"] if isinstance(error, dict) else error) == code


def test_review_flask_json_limits_keep_authentication_first(ui):
    app, _, _ = ui
    anonymous = app.test_client()
    raw = "[" * 100000
    for route in ("/api/explain", "/api/inference/predict"):
        assert anonymous.post(route, data=raw, content_type="application/json").status_code == 401


def test_review_flask_json_limits_keep_valid_nested_requests(ui):
    _, http, config = ui
    response = http.post("/api/inference/predict", json=config["bindings"][0])
    assert response.status_code == 200, response.text
    assert http.post("/api/explain", data="{broken", content_type="application/json").status_code == 400
    assert http.post("/api/inference/predict", data="{broken", content_type="application/json").status_code == 422
    assert http.post("/api/explain", data='{"a": 1}', content_type="text/plain").status_code == 400


def test_review_manual_score_of_the_newest_observation_alerts_once(ui, alerts, monkeypatch):
    app, http, config = ui
    app.config["INFERENCE_HTTP_CLIENT"] = fixed_scores()
    entry = config["bindings"][0]
    assert http.post("/api/inference/predict", json=entry).status_code == 200
    stamp = source_storage.metrics(SOURCE, "row_count")[-1]["ts"]
    assert [alert_time(body) for body in alerts] == [minute(stamp)]
    clear_throttle()
    assert http.post("/api/inference/predict", json=entry).status_code == 200
    tick()
    assert len(alerts) == 1


def test_review_manual_score_of_an_old_observation_is_silent(ui, alerts):
    app, http, config = ui
    app.config["INFERENCE_HTTP_CLIENT"] = fixed_scores()
    replace_history(SOURCE, [24 * 60 * i for i in range(3, 0, -1)])
    assert http.post("/api/inference/predict", json=config["bindings"][0]).status_code == 200
    assert alerts == []


def test_review_manual_score_of_inactive_version_is_silent(ui, alerts):
    app, http, config = ui
    app.config["INFERENCE_HTTP_CLIENT"] = fixed_scores()
    entry = {**config["bindings"][0], "model_version": "inactive-version"}
    assert http.post("/api/inference/predict", json=entry).status_code == 200
    assert alerts == []


def test_review_manual_recalculation_with_active_version_is_silent(ui, alerts):
    app, http, config = ui
    app.config["INFERENCE_HTTP_CLIENT"] = fixed_scores()
    entry = config["bindings"][0]
    tick(client=fixed_scores(anomalous=lambda body: False))
    with storage.get_engine().begin() as conn:
        conn.execute(text("DELETE FROM anomaly_scores WHERE ts = (SELECT MAX(ts) FROM anomaly_scores)"))
        conn.execute(
            text(
                "INSERT INTO anomaly_scores (project_id, connection_id, schema_name, table_name, "
                "ts, model_version, feature_schema_version, score, is_anomaly, is_demo) VALUES "
                "(:project_id, :connection_id, :schema_name, :table_name, :ts, 'other', '1.0.0', "
                "0.1, 0, 1)"
            ),
            {**SOURCE, "ts": source_storage.metrics(SOURCE, "row_count")[-1]["ts"]},
        )
    assert http.post("/api/inference/predict", json=entry).status_code == 200
    assert alerts == []


def foreign_binding(ui):
    """A complete, active and allowlisted source of a project the user cannot use."""
    _, _, config = ui
    storage.create_user("stranger", "stranger@example.test", "unused")
    storage.create_project("project-z", "stranger", "Z", "z")
    storage.create_connection(
        connection_id="connection-z", project_id="project-z", name="Z", dsn_encrypted=b"unused"
    )
    foreign = {**SOURCE, "project_id": "project-z", "connection_id": "connection-z"}
    add_history(foreign, [30, 15], [1000, 2000])
    entry = {**config["bindings"][0], "source": foreign}
    config["bindings"].append(entry)
    Path(os.environ["INFERENCE_ADAPTER_CONFIG"]).write_text(json.dumps(config))
    return entry


def test_review_project_scope_blocks_a_valid_foreign_binding(ui):
    app, http, _ = ui
    app.config["INFERENCE_HTTP_CLIENT"] = fixed_scores(-0.7777)
    entry = foreign_binding(ui)
    foreign = entry["source"]
    stamp = source_storage.metrics(foreign, "row_count")[-1]["ts"]

    assert http.post("/api/inference/predict", json=entry).status_code == 403
    response = http.post("/api/explain", json=explain_body(stamp, source=foreign))
    assert response.status_code == 403
    assert "1000" not in response.text and "2000" not in response.text
    with storage.get_engine().connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM anomaly_scores")).scalar() == 0
        assert conn.execute(text("SELECT COUNT(*) FROM llm_explanations")).scalar() == 0


def test_review_project_query_parameter_cannot_select_foreign_data(ui):
    _, http, _ = ui
    entry = foreign_binding(ui)
    foreign = entry["source"]
    response = http.get("/api/inference/metrics", query_string=foreign)
    # The session project replaces the parameter: connection-z is not in project-a.
    assert response.status_code == 403
    assert "1000" not in response.text


def test_review_mixed_serving_of_one_table_is_reported(ui, caplog):
    storage.create_connection(
        connection_id="connection-b",
        project_id=SOURCE["project_id"],
        name="B",
        dsn_encrypted=b"unused",
    )
    other = {**SOURCE, "connection_id": "connection-b"}
    add_history(other, [20, 10], [1, 2])
    assert adapter.unbound_sources(SOURCE["project_id"]) == [other]
    caplog.set_level(logging.WARNING)
    tick()
    assert "connection-b" in caplog.text and "unsupported" in caplog.text


def test_review_full_binding_of_one_table_is_not_reported(two_sources, caplog):
    assert adapter.unbound_sources(SOURCE["project_id"]) == []
    caplog.set_level(logging.WARNING)
    tick()
    assert "unsupported" not in caplog.text


@pytest.fixture
def collect(ui, monkeypatch, alerts):
    """A real collector tick; only the monitored database and HTTP model are simulated."""
    from unittest import mock

    from cryptography.fernet import Fernet

    import app.db as db_mod
    from app import crypto
    from collectors import metrics_collector, per_project

    monkeypatch.setenv("FERNET_KEY", Fernet.generate_key().decode())
    crypto.reset_for_tests()
    with storage.get_engine().begin() as conn:
        conn.execute(
            text("UPDATE connections SET dsn_encrypted = :dsn"),
            {"dsn": crypto.encrypt_dsn("postgresql://u:p@unreachable:5432/d")},
        )
    scoring = {"client": fixed_scores(-0.4321)}
    original = adapter.score_project
    monkeypatch.setattr(
        adapter, "score_project", lambda pid: original(pid, client=scoring["client"])
    )

    def run(connection_id, row_count=500, null_rate=0.02, age=timedelta(0)):
        def rows(table, ts=None):
            stamp = ts - age
            return [
                {"ts": stamp, "table_name": table, "metric_name": "row_count", "value": row_count},
                {"ts": stamp, "table_name": table, "metric_name": "null_rate", "value": null_rate},
            ]

        with (
            mock.patch.object(metrics_collector.MetricsCollector, "collect", side_effect=rows),
            mock.patch.object(
                db_mod.PostgresAdapter,
                "list_tables",
                autospec=True,
                side_effect=lambda self, schema: [{"table_name": "orders", "schema": schema}],
            ),
            mock.patch("collectors.schema_collector.collect_table_schema", return_value=None),
        ):
            per_project.collect_for_connection(SOURCE["project_id"], connection_id)
        return source_storage.metrics(
            {**SOURCE, "connection_id": connection_id}, "row_count"
        )[-1]["ts"]

    run.scoring = scoring
    return run


def legacy_detector(monkeypatch, score=-0.4321):
    """Published-model-free install: the persisted detector reports the newest tick."""
    rows = []

    def score_table(table, window_days=14, project_id="legacy"):
        stamp = source_storage.metrics(SOURCE, "row_count")[-1]["ts"]
        rows.append(stamp)
        return [{"ts": stamp, "score": score, "is_anomaly": 1}]

    monkeypatch.setattr("ml.anomaly_detector.score_table", score_table)
    return rows


def test_review_scenario_a_old_anomaly_after_update(ui, collect, monkeypatch):
    _, http, _ = ui
    prompts = capture_prompts(monkeypatch)
    monkeypatch.delenv("INFERENCE_ADAPTER_CONFIG")
    legacy_detector(monkeypatch)
    stamp = legacy_point(value=71)
    storage.save_anomaly_scores(
        [{"ts": stamp, "table_name": "orders", "score": -0.2468, "is_anomaly": 1}],
        SOURCE["project_id"],
    )
    collect(SOURCE["connection_id"])
    listed = http.get("/api/anomalies/orders?range=7d").json
    old = next(row for row in listed if row["ts"].startswith(stamp[:19]))
    response = http.post("/api/explain", json=explain_body(old["ts"]))
    assert response.status_code == 200, response.text
    assert score_line(stamp) + "-0.2468" in prompts[-1]
    assert ": 71" in prompts[-1] and ": 500" not in prompts[-1]


def test_review_scenario_b_collection_score_explanation_alert(ui, collect, alerts, monkeypatch):
    _, http, config = ui
    prompts = capture_prompts(monkeypatch)
    monkeypatch.setenv("FF_LLM_NOTIFICATIONS", "1")
    storage.save_changepoints(
        [changepoint((datetime.now(UTC) - timedelta(days=1)).isoformat())],
        project_id=SOURCE["project_id"],
    )
    stamp = collect(SOURCE["connection_id"])

    # Stored under the active pair and full source, and sent exactly once.
    entry = config["bindings"][0]
    stored = source_storage.scores(SOURCE, entry["model_version"], "1.0.0")
    assert stored[-1]["ts"] == stamp and stored[-1]["score"] == -0.4321
    assert [alert_time(body) for body in alerts] == [minute(stamp)]
    assert "Подключение: A (connection-a)" in alerts[0]
    assert "Score: -0.4321" in alerts[0]
    # The notification itself carried the real context.
    assert score_line(stamp) + "-0.4321" in prompts[-1]
    assert "row_count: 100.00 → 500.00" in prompts[-1]

    listed = http.get("/api/anomalies/orders?range=7d").json
    row = next(item for item in listed if item["ts"].startswith(stamp[:19]))
    response = http.post("/api/explain", json=explain_body(row["ts"], source=row["source"]))
    assert response.status_code == 200, response.text
    assert score_line(stamp) + "-0.4321" in prompts[-1]


def test_review_scenario_c_backfill_then_fresh_event(ui, collect, alerts):
    replace_history(SOURCE, [24 * 60 * i for i in range(29, 0, -1)])
    fresh = collect(SOURCE["connection_id"], age=timedelta(minutes=5))
    # The tick recalculated a month of history and reported only its own point.
    assert len(collect.scoring["client"].sent) == 29
    assert [alert_time(body) for body in alerts] == [minute(fresh)]
    clear_throttle()
    fresher = collect(SOURCE["connection_id"], row_count=900)
    assert [alert_time(body) for body in alerts[1:]] == [minute(fresher)]


def test_review_scenario_d_two_sources_in_collection(two_sources, ui, collect, alerts):
    _, config, first = two_sources
    other = first["source"]
    collect(SOURCE["connection_id"])
    collect(other["connection_id"])
    assert len(alerts) == 2
    assert {SOURCE["connection_id"], other["connection_id"]} == {
        re.search(r"Подключение: \S+ \((.+)\)", body).group(1) for body in alerts
    }
    for entry in config["bindings"]:
        assert source_storage.scores(entry["source"], entry["model_version"], "1.0.0")
    before = len(collect.scoring["client"].sent)
    adapter.score_project(SOURCE["project_id"])
    assert len(collect.scoring["client"].sent) == before
    assert len(alerts) == 2


def test_review_scenario_f_standard_install_keeps_legacy_alerts(ui, collect, alerts, monkeypatch):
    _, http, _ = ui
    prompts = capture_prompts(monkeypatch)
    monkeypatch.setenv("FF_LLM_NOTIFICATIONS", "1")
    monkeypatch.delenv("INFERENCE_ADAPTER_CONFIG")
    legacy_detector(monkeypatch)
    stamp = collect(SOURCE["connection_id"])
    assert collect.scoring["client"].sent == []
    assert len(alerts) == 1
    assert "Подключение" not in alerts[0]
    assert score_line(stamp) + "-0.4321" in prompts[-1]
    with storage.get_engine().connect() as conn:
        assert conn.execute(text("SELECT event_key FROM telegram_throttle")).scalar() == "anomaly"
    assert storage.get_anomaly_scores("orders", SOURCE["project_id"])[-1]["score"] == -0.4321
    assert http.get("/dashboard/history").status_code == 200
