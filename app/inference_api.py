"""Versioned UI API. The user's current project is authoritative."""

from flask import Blueprint, current_app, g, jsonify, request
from flask_login import current_user
from jsonschema import ValidationError

from app import inference_adapter as adapter
from app import source_storage
from ml.inference_contract import validator

bp = Blueprint("inference_api", __name__, url_prefix="/api/inference")


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


def authorized_source(value):
    pid = project()
    try:
        validator("Source").validate(value)
    except ValidationError:
        raise adapter.AdapterError(422, "invalid_input") from None
    if value["project_id"] != pid:
        raise adapter.AdapterError(403, "source_forbidden")
    adapter.check_source(value)
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
    project()
    body = request.get_json(silent=True)
    try:
        validator("ModelRequest").validate(body)
    except ValidationError:
        raise adapter.AdapterError(422, "invalid_input") from None
    source = authorized_source(body["source"])
    try:
        return jsonify(
            adapter.predict(
                source,
                body["model_version"],
                body["feature_schema_version"],
                client=current_app.config.get("INFERENCE_HTTP_CLIENT"),
            )
        )
    except ValueError as exc:
        if isinstance(exc, adapter.AdapterError):
            raise
        raise adapter.AdapterError(409, "ambiguous_source") from None


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
    return jsonify(source_storage.metrics(source, metric))
