"""Course #4: existing HTTP responses and the proposed inference boundary.

The future /v1/predict endpoint is NOT implemented or mocked here. Its examples
and input schema are checked; actual HTTP/auth/model integration belongs to #12.
"""

import copy
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator, FormatChecker, ValidationError
from openapi_spec_validator import validate
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

from app import metrics_storage as storage
from app.app import create_app
from ml import anomaly_detector as detector
from ml import forecast as forecast_module

CONTRACTS = Path(__file__).resolve().parents[1] / "docs" / "course" / "api"
LEGACY = yaml.safe_load((CONTRACTS / "legacy.openapi.yaml").read_text(encoding="utf-8"))
INFERENCE = yaml.safe_load((CONTRACTS / "inference.openapi.yaml").read_text(encoding="utf-8"))
ANOMALIES = "/api/anomalies/{table_name}"
FORECAST = "/api/forecast/{table_name}"
PREDICT = "/v1/predict"


def _escape(value):
    return str(value).replace("~", "~0").replace("/", "~1")


def _validator(spec, *path):
    """Resolve internal OpenAPI refs using a registry with no network retriever."""
    uri = "urn:dbmonitor:contract"
    resource = Resource.from_contents(spec, default_specification=DRAFT202012)
    registry = Registry().with_resource(uri, resource)
    pointer = "/".join(_escape(part) for part in path)
    return Draft202012Validator(
        {"$ref": f"{uri}#/{pointer}"}, registry=registry, format_checker=FormatChecker()
    )


def _check_http(response, path):
    status = str(response.status_code)
    responses = LEGACY["paths"][path]["get"]["responses"]
    declared = responses[status]
    if "$ref" in declared:
        declared = LEGACY["components"]["responses"][declared["$ref"].split("/")[-1]]
    assert response.mimetype in declared["content"]
    if response.is_json:
        body = response.get_json()
        json.dumps(body, allow_nan=False)
        _validator(
            LEGACY,
            "paths",
            path,
            "get",
            "responses",
            status,
            "content",
            "application/json",
            "schema",
        ).validate(body)
    return response.get_json() if response.is_json else response.get_data(as_text=True)


def _request_example():
    return copy.deepcopy(
        INFERENCE["paths"][PREDICT]["post"]["requestBody"]["content"]["application/json"]["example"]
    )


def _request_validator():
    return _validator(INFERENCE, "components", "schemas", "PredictionRequest")


@pytest.mark.parametrize("spec", [LEGACY, INFERENCE], ids=["legacy", "inference-proposal"])
def test_openapi_document_is_valid(spec):
    validate(spec)


@pytest.mark.parametrize("spec", [LEGACY, INFERENCE], ids=["legacy", "inference-proposal"])
def test_all_documented_payload_examples_match_their_schema(spec):
    for path, item in spec["paths"].items():
        for method, operation in item.items():
            containers = [
                ("responses", status, response)
                for status, response in operation["responses"].items()
            ]
            if "requestBody" in operation:
                containers.append(("requestBody", None, operation["requestBody"]))
            for kind, status, container in containers:
                for media, content in container.get("content", {}).items():
                    examples = [v["value"] for v in content.get("examples", {}).values()]
                    if "example" in content:
                        examples.append(content["example"])
                    pointer = ["paths", path, method, kind]
                    if status is not None:
                        pointer.append(status)
                    validator = _validator(spec, *pointer, "content", media, "schema")
                    for value in examples:
                        json.dumps(value, allow_nan=False)
                        validator.validate(value)


@pytest.fixture
def contract_app(tmp_path, monkeypatch):
    monkeypatch.setattr(storage.settings, "MONITOR_DB_URL", f"sqlite:///{tmp_path / 'contract.db'}")
    monkeypatch.setattr(storage.settings, "DATABASE_URL", "sqlite://")
    monkeypatch.setattr(storage.settings, "ADMIN_EMAIL", "")
    monkeypatch.setattr(storage.settings, "SENTRY_DSN", "")
    monkeypatch.setattr(storage.settings, "NIM_API_KEY", "")
    monkeypatch.setattr(storage.settings, "TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setattr(storage, "_engine", None)
    monkeypatch.setattr(storage, "_initialized", False)
    monkeypatch.setattr(detector, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(forecast_module, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(forecast_module, "_HAS_PROPHET", False)
    monkeypatch.setenv("FF_FORECAST", "true")
    app = create_app({"TESTING": True, "LOGIN_DISABLED": False})
    yield app
    if storage._engine is not None:
        storage._engine.dispose()


def test_anomalies_with_empty_history_returns_empty_array(contract_app):
    with contract_app.test_client() as client:
        assert _check_http(client.get("/api/anomalies/missing"), ANOMALIES) == []


def test_stored_anomaly_does_not_require_model_or_trigger_training(contract_app, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Reading persisted scores must not train a model")

    monkeypatch.setattr(detector, "train", forbidden)
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    storage.save_anomaly_scores(
        [{"table_name": "orders", "ts": stamp, "score": -0.12, "is_anomaly": 1}]
    )
    with contract_app.test_client() as client:
        body = _check_http(client.get("/api/anomalies/orders"), ANOMALIES)
    assert body == [{"ts": stamp, "score": -0.12, "is_anomaly": 1}]


def test_optional_feature_breakdown_matches_contract(contract_app, monkeypatch):
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    storage.save_anomaly_scores(
        [{"table_name": "orders", "ts": stamp, "score": -0.12, "is_anomaly": 1}]
    )
    values = dict.fromkeys(detector.FEATURE_NAMES, 0.0)
    details = {"values": values, "z_scores": values, "top_feature": "row_count"}
    monkeypatch.setattr(detector, "feature_breakdown", lambda *args, **kwargs: {stamp: details})
    with contract_app.test_client() as client:
        body = _check_http(client.get("/api/anomalies/orders"), ANOMALIES)
    assert body[0]["features"] == details


@pytest.mark.parametrize("query", ["range=bad", "range="])
def test_invalid_anomaly_range_matches_contract(contract_app, query):
    with contract_app.test_client() as client:
        response = client.get(f"/api/anomalies/orders?{query}")
        assert response.status_code == 400
        _check_http(response, ANOMALIES)


@pytest.mark.parametrize("query", ["metric=null_rate", "horizon=2d"])
def test_invalid_forecast_parameters_match_contract(contract_app, query):
    with contract_app.test_client() as client:
        response = client.get(f"/api/forecast/orders?{query}")
        assert response.status_code == 400
        _check_http(response, FORECAST)


def test_forecast_empty_history_matches_422_contract(contract_app):
    with contract_app.test_client() as client:
        response = client.get("/api/forecast/missing")
        assert response.status_code == 422
        assert _check_http(response, FORECAST)["error"] == "insufficient_data"


def test_forecast_disabled_returns_html_even_for_bad_parameters(contract_app, monkeypatch):
    monkeypatch.setenv("FF_FORECAST", "false")
    with contract_app.test_client() as client:
        response = client.get("/api/forecast/orders?metric=bad")
        assert response.status_code == 404
        _check_http(response, FORECAST)


@pytest.mark.parametrize(
    ("url", "path", "module", "function"),
    [
        ("/api/anomalies/orders", ANOMALIES, "app.api", "get_anomaly_scores"),
        ("/api/forecast/orders", FORECAST, "ml.forecast", "forecast"),
    ],
)
def test_unhandled_runtime_error_returns_html_500(
    contract_app, monkeypatch, url, path, module, function
):
    import importlib

    def unavailable(*args, **kwargs):
        raise RuntimeError("Synthetic storage/model failure")

    monkeypatch.setattr(importlib.import_module(module), function, unavailable)
    contract_app.config["PROPAGATE_EXCEPTIONS"] = False
    with contract_app.test_client() as client:
        response = client.get(url)
        assert response.status_code == 500
        _check_http(response, path)


def test_forecast_real_linear_model_matches_response_contract(contract_app):
    start = datetime.now(UTC) - timedelta(hours=9)
    storage.save_metrics(
        [
            {
                "ts": (start + timedelta(hours=i)).isoformat(timespec="seconds"),
                "table_name": "orders",
                "metric_name": "row_count",
                "value": 100 + 10 * i,
            }
            for i in range(10)
        ],
        "legacy",
    )
    with contract_app.test_client() as client:
        response = client.get("/api/forecast/orders?horizon=1d")
        assert response.status_code == 200
        points = _check_http(response, FORECAST)
    assert len(points) == 24
    assert all(p["yhat_lower"] <= p["yhat"] <= p["yhat_upper"] for p in points)


def test_current_session_selects_project_query_parameter_does_not(contract_app):
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    clients = []
    for name, score in [("a", 0.1), ("b", 0.2)]:
        storage.create_user(name, f"{name}@example.test", "unused-hash")
        storage.create_project(name, name, name, name)
        storage.save_anomaly_scores(
            [{"table_name": "orders", "ts": stamp, "score": score, "is_anomaly": 0}], name
        )
        client = contract_app.test_client()
        with client.session_transaction() as session:
            session["_user_id"] = name
            session["_fresh"] = True
            session["current_project_id"] = name
        clients.append(client)
    for client, expected in zip(clients, [0.1, 0.2], strict=True):
        body = _check_http(client.get("/api/anomalies/orders?project_id=b"), ANOMALIES)
        assert body[0]["score"] == expected
    with contract_app.test_client() as anonymous:
        assert _check_http(anonymous.get("/api/anomalies/orders?project_id=a"), ANOMALIES) == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("row_count", "110"),
        ("row_count", True),
        ("row_count", -1),
        ("null_rate", 1.01),
        ("null_rate", None),
        ("d_null_rate", -1.01),
    ],
)
def test_proposed_contract_rejects_invalid_feature_types_and_ranges(field, value):
    request = _request_example()
    request["features"][field] = value
    with pytest.raises(ValidationError):
        _request_validator().validate(request)


@pytest.mark.parametrize(
    "field", ["source", "model_version", "previous_observed_at", "feature_schema_version"]
)
def test_proposed_contract_rejects_missing_identity_or_version(field):
    request = _request_example()
    del request[field]
    with pytest.raises(ValidationError):
        _request_validator().validate(request)


@pytest.mark.parametrize("field", ["connection_id", "schema_name"])
def test_proposed_contract_requires_unambiguous_source(field):
    request = _request_example()
    del request["source"][field]
    with pytest.raises(ValidationError):
        _request_validator().validate(request)


@pytest.mark.parametrize("value", ["2026-10-25T12:00:00", "not-a-date"])
def test_proposed_contract_rejects_timestamp_without_timezone(value):
    request = _request_example()
    request["observed_at"] = value
    with pytest.raises(ValidationError):
        _request_validator().validate(request)


def test_proposed_contract_rejects_array_and_extra_features():
    request = _request_example()
    with pytest.raises(ValidationError):
        _request_validator().validate([request])
    request["features"]["unexpected"] = 0.0
    with pytest.raises(ValidationError):
        _request_validator().validate(request)


def test_proposed_contract_rejects_legacy_unknown_as_published_model():
    request = _request_example()
    request["model_version"] = "legacy-unknown"
    with pytest.raises(ValidationError):
        _request_validator().validate(request)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_feature_cannot_be_serialized_as_contract_json(value):
    # JSON Schema operates on parsed Python values, not the HTTP JSON parser.
    # The future server must separately enforce this wire-format rule in #12.
    request = _request_example()
    request["features"]["row_count"] = value
    with pytest.raises(ValueError):
        json.dumps(request, allow_nan=False)


def test_renamed_score_breaks_contract_but_additive_response_field_does_not():
    operation = INFERENCE["paths"][PREDICT]["post"]
    response = copy.deepcopy(
        operation["responses"]["200"]["content"]["application/json"]["example"]
    )
    validator = _validator(INFERENCE, "components", "schemas", "PredictionResponse")
    response["processing_ms"] = 2.5  # Backward-compatible optional response field.
    validator.validate(response)
    response["anomaly_score"] = response.pop("score")  # Breaking change.
    with pytest.raises(ValidationError, match="score"):
        validator.validate(response)


def test_proposed_features_follow_existing_alignment_and_delta_semantics(monkeypatch):
    # A gap and an unmatched row_count must not become a zero-filled point
    # or a time-normalised delta. This exercises the actual legacy loader.
    rc = [
        {"ts": "2026-10-25T11:00:00+00:00", "value": 100.0},
        {"ts": "2026-10-25T11:45:00+00:00", "value": 105.0},
        {"ts": "2026-10-25T12:00:00+00:00", "value": 110.0},
    ]
    nr = [
        {"ts": "2026-10-25T11:00:00+00:00", "value": 0.02},
        {"ts": "2026-10-25T12:00:00+00:00", "value": 0.03},
    ]
    monkeypatch.setattr(
        detector, "get_metrics", lambda table, metric, *a, **kw: rc if metric == "row_count" else nr
    )
    timestamps, matrix = detector._load_features("orders", timedelta(days=1), "project-a")
    request = _request_example()
    request["previous_observed_at"] = rc[0]["ts"]
    _request_validator().validate(request)
    assert detector.FEATURE_NAMES == tuple(request["features"])
    assert len(timestamps) == 1
    assert matrix[0] == pytest.approx(
        [request["features"][name] for name in detector.FEATURE_NAMES]
    )


@pytest.mark.parametrize("aligned", [True, False], ids=["only-current", "no-common-timestamp"])
def test_missing_previous_aligned_point_is_not_zero_filled(monkeypatch, aligned):
    def load(table, metric, *args, **kwargs):
        stamp = "2026-10-25T12:00:00+00:00"
        if not aligned and metric == "null_rate":
            stamp = "2026-10-25T11:45:00+00:00"
        return [{"ts": stamp, "value": 100.0 if metric == "row_count" else 0.02}]

    monkeypatch.setattr(detector, "get_metrics", load)
    with pytest.raises(detector.InsufficientDataError):
        detector._load_features("orders", timedelta(days=1), "project-a")
