"""Versioned UI API. The user's current project is authoritative."""

import json

from flask import Blueprint, current_app, g, jsonify, request
from flask_login import current_user
from jsonschema import ValidationError

from app import inference_adapter as adapter
from app import source_storage
from ml.inference_contract import MAX_JSON_BYTES, FiniteValidator, json_too_deep, validator

bp = Blueprint("inference_api", __name__, url_prefix="/api/inference")


def json_body():
    """Request JSON within the service limits; ``None`` when it is not valid JSON."""
    if not request.is_json:
        return None
    if (request.content_length or 0) > MAX_JSON_BYTES:
        raise adapter.AdapterError(413, "request_too_large")
    raw = bytearray()
    while chunk := request.stream.read(MAX_JSON_BYTES + 1 - len(raw)):
        raw.extend(chunk)
        if len(raw) > MAX_JSON_BYTES:
            raise adapter.AdapterError(413, "request_too_large")
    if json_too_deep(raw):
        raise adapter.AdapterError(400, "invalid_json")
    try:
        return json.loads(raw)
    except (ValueError, RecursionError):
        return None


def project():
    if not current_user.is_authenticated:
        raise adapter.AdapterError(401, "authentication_required")
    current = getattr(g, "current_project", None)
    if current is None:
        raise adapter.AdapterError(403, "source_forbidden")
    return current["id"]


@bp.errorhandler(adapter.AdapterError)
def error(exc):
    return jsonify({"error": {"code": exc.code, "message": exc.code}}), exc.status


@bp.errorhandler(source_storage.AmbiguousSourceError)
def ambiguous_source(exc):
    return error(adapter.AdapterError(409, "ambiguous_source"))


@bp.errorhandler(source_storage.InvalidHistoryError)
def invalid_history(exc):
    return error(adapter.AdapterError(422, "invalid_history"))


def authorized_source(value, *, serving=True):
    pid = project()
    try:
        validator("Source").validate(value)
    except ValidationError:
        raise adapter.AdapterError(422, "invalid_input") from None
    if value["project_id"] != pid:
        raise adapter.AdapterError(403, "source_forbidden")
    if serving:
        adapter.check_source(value)
    else:
        try:
            source_storage.check_access(value)
        except ValueError:
            raise adapter.AdapterError(403, "source_forbidden") from None
    return value


@bp.get("/sources")
def sources():
    result = []
    for entry in adapter.bindings(project()):
        try:
            adapter.check_source(entry["source"])
        except adapter.AdapterError:
            continue
        result.append({**entry, "versions": source_storage.versions(entry["source"])})
    return jsonify(result)


@bp.post("/predict")
def predict():
    from app.metrics_storage import get_member_role

    pid = project()
    if get_member_role(pid, current_user.id) not in {"owner", "editor"}:
        raise adapter.AdapterError(403, "source_forbidden")
    body = json_body()
    try:
        validator("ModelRequest").validate(body)
    except ValidationError:
        raise adapter.AdapterError(422, "invalid_input") from None
    source = authorized_source(body["source"])
    return jsonify(
        adapter.predict(
            source,
            body["model_version"],
            body["feature_schema_version"],
            client=current_app.config.get("INFERENCE_HTTP_CLIENT"),
        )
    )


@bp.get("/scores")
def scores():
    source = authorized_source(
        {
            "project_id": project(),
            "connection_id": request.args.get("connection_id"),
            "schema_name": request.args.get("schema_name"),
            "table_name": request.args.get("table_name"),
        }
    )
    model, schema = request.args.get("model_version"), request.args.get("feature_schema_version")
    if not model or not schema:
        raise adapter.AdapterError(422, "version_required")
    return jsonify(source_storage.scores(source, model, schema))


@bp.get("/metrics")
def metrics():
    source = authorized_source(
        {
            "project_id": project(),
            "connection_id": request.args.get("connection_id"),
            "schema_name": request.args.get("schema_name"),
            "table_name": request.args.get("table_name"),
        }
    )
    metric = request.args.get("metric", "row_count")
    if metric not in {"row_count", "null_rate"}:
        raise adapter.AdapterError(422, "invalid_input")
    rows = source_storage.metrics(source, metric)
    if any(not FiniteValidator.TYPE_CHECKER.is_type(row["value"], "number") for row in rows):
        raise adapter.AdapterError(422, "invalid_history")
    return jsonify(rows)
