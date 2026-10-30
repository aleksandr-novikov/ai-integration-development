"""Authenticated Flask/background caller. History never enters the ASGI service."""

import json
import logging
import math
import os
from pathlib import Path

import httpx
from jsonschema import ValidationError
from sqlalchemy.exc import OperationalError
from sqlalchemy.exc import TimeoutError as PoolTimeoutError

from app import source_storage
from ml.features import FEATURE_NAMES, build_rows
from ml.inference_contract import canonical_ts, validator

logger = logging.getLogger(__name__)


class AdapterError(ValueError):
    def __init__(self, status, code):
        self.status, self.code = status, code


# Expected failures of one source: invalid history, service refusals and
# transient storage errors. They are logged and never stop other sources.
SOURCE_ERRORS = (AdapterError, source_storage.AmbiguousSourceError, OperationalError, PoolTimeoutError)
# Refusals tied to one observation; later observations stay scorable.
OBSERVATION_ERRORS = frozenset({"invalid_history", "insufficient_history", "invalid_input"})


def configuration():
    filename = os.environ.get("INFERENCE_ADAPTER_CONFIG")
    if not filename:
        return {"bindings": []}
    return json.loads(Path(filename).read_text(encoding="utf-8"))


def bindings(project_id):
    return [
        entry
        for entry in configuration()["bindings"]
        if entry["source"]["project_id"] == project_id
    ]


def is_served(project_id, table):
    return any(entry["source"]["table_name"] == table for entry in bindings(project_id))


def check_source(source):
    """Configuration is an explicit allowlist; membership is checked by Flask."""
    from app.metrics_storage import get_connection

    if not any(entry["source"] == source for entry in bindings(source["project_id"])):
        raise AdapterError(403, "source_forbidden")
    conn = get_connection(source["project_id"], source["connection_id"])
    if not conn or not conn["is_active"]:
        raise AdapterError(403, "source_forbidden")
    namespaces = {conn.get("iceberg_namespace") or conn["schema_name"]}
    raw = conn.get("iceberg_namespace_allowlist")
    if raw:
        namespaces.update(json.loads(raw))
    if source["schema_name"] not in namespaces:
        raise AdapterError(403, "source_forbidden")


def predict(source, model_version, feature_schema_version, *, client=None, observed_at=None):
    check_source(source)
    rc = source_storage.metrics(source, "row_count")
    nr = source_storage.metrics(source, "null_rate")
    if observed_at is not None:
        stamp = canonical_ts(observed_at)
        rc = [row for row in rc if row["ts"] <= stamp]
        nr = [row for row in nr if row["ts"] <= stamp]
    try:
        times, rows = build_rows(rc, nr)
    except ValueError as exc:
        raise AdapterError(422, "invalid_history") from exc
    if not rows:
        raise AdapterError(422, "insufficient_history")
    if observed_at is not None and times[-1] != canonical_ts(observed_at):
        raise AdapterError(422, "insufficient_history")
    features = dict(zip(FEATURE_NAMES, rows[-1], strict=True))
    # JSON has no NaN/Infinity: such history is invalid, not a service failure.
    if not all(math.isfinite(value) for value in features.values()):
        raise AdapterError(422, "invalid_history")
    aligned = sorted({r["ts"] for r in rc} & {r["ts"] for r in nr})
    payload = {
        "source": source,
        "model_version": model_version,
        "feature_schema_version": feature_schema_version,
        "observed_at": times[-1],
        "previous_observed_at": aligned[-2],
        "features": features,
    }
    config = configuration()
    token = os.environ.get(config.get("token_env", "INFERENCE_ADAPTER_TOKEN"), "")
    if not token:
        raise AdapterError(503, "inference_unavailable")
    url = config.get("url", "http://127.0.0.1:8000").rstrip("/")

    def call(http):
        return http.post(
            url + "/v1/predict", json=payload, headers={"Authorization": f"Bearer {token}"}
        )

    try:
        if client is None:
            with httpx.Client(timeout=10, follow_redirects=False, trust_env=False) as http:
                response = call(http)
        else:
            response = call(client)
    except httpx.HTTPError:
        raise AdapterError(503, "inference_unavailable") from None
    if response.status_code != 200:
        allowed = {
            400: "invalid_json",
            401: "authentication_required",
            403: "source_forbidden",
            409: "feature_schema_mismatch",
            415: "unsupported_media_type",
            422: "invalid_input",
            503: "model_unavailable",
        }
        raise AdapterError(
            response.status_code if response.status_code in allowed else 503,
            allowed.get(response.status_code, "inference_unavailable"),
        )
    try:
        result = response.json()
        validator("PredictionResponse").validate(result)
        if result.get("demo") is not config.get("demo", False):
            raise ValueError("inference demo mode mismatch")
        for field in ("source", "model_version", "feature_schema_version", "observed_at"):
            if result[field] != payload[field]:
                raise ValueError("inference identity mismatch")
    except (ValueError, KeyError, ValidationError):
        raise AdapterError(503, "invalid_inference_response") from None
    source_storage.save_prediction(result)
    return result


def check_binding(model_version, feature_schema_version):
    """Configured versions concern the whole source: check them before any request.

    The service answers a malformed version with the same 422 invalid_input as a
    rejected observation, so this must not be left to the per-observation path.
    """
    try:
        validator("ModelVersion").validate(model_version)
        validator("FeatureSchemaVersion").validate(feature_schema_version)
    except ValidationError:
        raise AdapterError(422, "invalid_binding") from None


def score_source(entry, *, client=None):
    """Score every stored observation of one configured source."""
    source = entry["source"]
    check_binding(entry["model_version"], entry["feature_schema_version"])
    rc = source_storage.metrics(source, "row_count")
    nr = source_storage.metrics(source, "null_rate")
    try:
        times, _ = build_rows(rc, nr)
    except ValueError as exc:
        raise AdapterError(422, "invalid_history") from exc
    for stamp in times:
        try:
            predict(
                source,
                entry["model_version"],
                entry["feature_schema_version"],
                observed_at=stamp,
                client=client,
            )
        except AdapterError as exc:
            if exc.code not in OBSERVATION_ERRORS:
                raise
            logger.warning(
                "Published inference skipped observed_at=%s for source=%s: %s",
                stamp,
                source,
                exc.code,
            )


def score_project(project_id, *, client=None):
    """Periodic scoring uses the same published version and HTTP boundary as UI.

    Every source is attempted. Expected source failures are logged; unexpected
    ones are annotated with the source and raised together as an ExceptionGroup,
    which the calling job logs once with every traceback.
    """
    message = f"published inference failed for project {project_id}"
    unexpected = []
    for entry in bindings(project_id):
        source = entry["source"]
        try:
            score_source(entry, client=client)
        except SOURCE_ERRORS:
            logger.exception("Published inference failed for source=%s", source)
        except MemoryError:
            if unexpected:  # the group is never raised, so report it here
                logger.error(message, exc_info=ExceptionGroup(message, unexpected))
            raise
        except Exception as exc:
            exc.add_note(f"source={source}")
            unexpected.append(exc)
    if unexpected:
        raise ExceptionGroup(message, unexpected)


def train_candidates(project_id):
    """Optional nightly candidates never change configured serving versions."""
    from ml.model_registry import train_candidate

    config = configuration()
    root = config.get("candidate_registry_path")
    if not root:
        return []
    git_sha = os.environ.get("INFERENCE_TRAINING_GIT_SHA", "")
    return [
        train_candidate(
            root,
            source=entry["source"],
            row_counts=source_storage.metrics(entry["source"], "row_count"),
            null_rates=source_storage.metrics(entry["source"], "null_rate"),
            git_sha=git_sha,
        )
        for entry in bindings(project_id)
    ]
