"""Authenticated Flask/background caller. History never enters the ASGI service."""

import itertools
import json
import logging
import math
import os
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
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
SOURCE_ERRORS = (
    AdapterError,
    source_storage.AmbiguousSourceError,
    source_storage.InvalidHistoryError,
    OperationalError,
    PoolTimeoutError,
)
# Refusals tied to one observation; later observations stay scorable.
OBSERVATION_ERRORS = frozenset({"invalid_history", "insufficient_history", "invalid_input"})
VERSION_FIELDS = ("model_version", "feature_schema_version")
# Notifications concern current events. After an outage unreported observations
# stay relevant for the reach of the legacy per-tick scoring window.
MAX_EVENT_AGE = timedelta(days=1)
# Without any stored result only a just collected observation is an event.
MIN_FRESHNESS = timedelta(minutes=30)
FRESH_COLLECTIONS = 2


def configuration():
    filename = os.environ.get("INFERENCE_ADAPTER_CONFIG")
    if not filename:
        return {"bindings": []}
    try:
        config = json.loads(Path(filename).read_text(encoding="utf-8"))
        if not isinstance(config, dict) or not isinstance(config.get("bindings"), list):
            raise ValueError("bindings must be an array")
        if type(config.get("demo", False)) is not bool:
            raise ValueError("demo must be boolean")
        for entry in config["bindings"]:
            if not isinstance(entry, dict):
                raise ValueError("binding must be an object")
            source = entry.get("source")
            if not isinstance(source, dict) or not isinstance(source.get("project_id"), str):
                raise ValueError("binding must identify its project")
        for key in ("url", "token_env", "candidate_registry_path"):
            if key in config and (not isinstance(config[key], str) or not config[key]):
                raise ValueError("invalid configuration field")
        return config
    except (OSError, ValueError, KeyError, TypeError):
        logger.error("Invalid inference adapter configuration")
        raise AdapterError(503, "invalid_configuration") from None


def bindings(project_id):
    entries = [
        entry
        for entry in configuration()["bindings"]
        if entry["source"]["project_id"] == project_id
    ]
    try:
        for entry in entries:
            source_storage.source_params(entry["source"])
            # The contract refines the values per source; a wrong type has no
            # usable meaning anywhere, from SQL parameters to UI labels.
            if not all(isinstance(entry.get(key), str) for key in VERSION_FIELDS):
                raise ValueError("binding versions must be strings")
        identities = [tuple(sorted(entry["source"].items())) for entry in entries]
        if len(set(identities)) != len(identities):
            raise ValueError("duplicate source binding")
    except (ValueError, KeyError, TypeError):
        logger.error("Invalid inference bindings for project=%s", project_id)
        raise AdapterError(503, "invalid_configuration") from None
    return entries


def is_served(project_id, table):
    return any(entry["source"]["table_name"] == table for entry in bindings(project_id))


def unbound_sources(project_id):
    """Sources sharing a served table name without their own binding.

    A binding moves its table name out of the project-level legacy detector, so
    these sources are neither scored nor alerted until they are bound as well.
    """
    entries = bindings(project_id)
    bound = [entry["source"] for entry in entries]
    tables = sorted({source["table_name"] for source in bound})
    return [
        source
        for table in tables
        for source in source_storage.recent_sources(project_id, table)
        if source not in bound
    ]


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


def predict(
    source,
    model_version,
    feature_schema_version,
    *,
    client=None,
    observed_at=None,
    _history=None,
    _config=None,
):
    """Score one observation. Only the interactive form reads and stores history;
    with ``_history`` the caller owns storing the result and its notification.
    """
    interactive = _history is None
    if interactive:
        check_source(source)
        rc = source_storage.metrics(source, "row_count")
        nr = source_storage.metrics(source, "null_rate")
    else:
        rc, nr = _history
    if observed_at is not None:
        stamp = canonical_ts(observed_at)
        rc = [row for row in rc if row["ts"] <= stamp]
        nr = [row for row in nr if row["ts"] <= stamp]
    aligned = sorted({r["ts"] for r in rc} & {r["ts"] for r in nr})
    # Only this pair contributes to the prediction. An older malformed value
    # must not prevent later healthy pairs from being scored.
    pair = set(aligned[-2:])
    try:
        times, rows = build_rows(
            [row for row in rc if row["ts"] in pair],
            [row for row in nr if row["ts"] in pair],
        )
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
    payload = {
        "source": source,
        "model_version": model_version,
        "feature_schema_version": feature_schema_version,
        "observed_at": times[-1],
        "previous_observed_at": aligned[-2],
        "features": features,
    }
    config = configuration() if _config is None else _config
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
    if interactive:
        floor = _new_event_floor(source, aligned)
        if source_storage.insert_prediction(result) and any(
            entry["source"] == source
            and all(entry[key] == result[key] for key in VERSION_FIELDS)
            for entry in bindings(source["project_id"])
        ):
            _alert_new_event(result, floor)
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


def _new_event_floor(source, times):
    """Observations after this instant are new events for notifications.

    Backfill, a version switch or a rollback only recalculate observations that
    the high-water mark of stored results, kept for every version, already covers.
    With no result at all the newest collected observation alone can be an event.
    Both limits rely on stored data, so a restart does not change them.
    """
    from app.metrics_storage import get_connection

    now = datetime.now(UTC)
    processed = source_storage.latest_scored_at(source)
    if processed is not None:
        return max(processed, canonical_ts((now - MAX_EVENT_AGE).isoformat()))
    connection = get_connection(source["project_id"], source["connection_id"]) or {}
    collection = timedelta(minutes=connection.get("interval_minutes", 0))
    freshness = max(MIN_FRESHNESS, FRESH_COLLECTIONS * collection)
    return max(times[-2], canonical_ts((now - freshness).isoformat()))


def _alert_new_event(result, floor):
    """Notify about a claimed result; never raises into scoring.

    Concurrent workers claim different observations, so each checks that its
    candidate is still the newest stored anomaly: one current event, one message.
    """
    if result is None or not result["is_anomaly"] or result["observed_at"] <= floor:
        return
    source = result["source"]
    try:
        newest = source_storage.latest_anomaly_at(
            source, result["model_version"], result["feature_schema_version"]
        )
        if newest == result["observed_at"]:
            _notify_prediction(result)
    except Exception:
        logger.exception("Published anomaly notification failed for source=%s", source)


def _notify_prediction(result):
    from app.notifications.telegram import load_project_telegram_config, notify_anomaly

    source = result["source"]
    cfg = load_project_telegram_config(source["project_id"])
    if cfg is None:
        return
    token, chat, throttle = cfg
    notify_anomaly(
        source["project_id"],
        source["table_name"],
        result["observed_at"],
        result["score"],
        bot_token=token,
        chat_id=chat,
        throttle_minutes=throttle,
        source=source,
    )


def score_source(entry, *, client=None):
    """Read history once and score only missing versioned observations.

    Every result is stored, and only the newest claimed anomaly can notify.
    """
    source = entry["source"]
    check_binding(entry["model_version"], entry["feature_schema_version"])
    check_source(source)
    config = configuration()
    rc = {row["ts"]: row for row in source_storage.metrics(source, "row_count", quarantine=True)}
    nr = {row["ts"]: row for row in source_storage.metrics(source, "null_rate", quarantine=True)}
    times = sorted(rc.keys() & nr.keys())
    existing = {
        row["ts"]
        for row in source_storage.scores(
            source, entry["model_version"], entry["feature_schema_version"]
        )
    }
    missing = [pair for pair in itertools.pairwise(times) if pair[1] not in existing]
    if not missing:
        return
    floor = _new_event_floor(source, times)
    candidate = None
    try:
        for previous, stamp in missing:
            try:
                result = predict(
                    source,
                    entry["model_version"],
                    entry["feature_schema_version"],
                    observed_at=stamp,
                    client=client,
                    _config=config,
                    _history=([rc[previous], rc[stamp]], [nr[previous], nr[stamp]]),
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
                continue
            if source_storage.insert_prediction(result) and result["is_anomaly"]:
                candidate = result
    finally:
        # A later source failure must not cost the event already claimed here.
        _alert_new_event(candidate, floor)


def _warn_unbound_sources(project_id):
    try:
        unbound = unbound_sources(project_id)
    except (OperationalError, PoolTimeoutError):
        logger.exception("Mixed serving check failed for project=%s", project_id)
        return
    if unbound:
        logger.warning(
            "unsupported mixed serving: sources without a binding share a served table "
            "and receive no scores or alerts: %s",
            unbound,
        )


def score_project(project_id, *, client=None):
    """Periodic scoring uses the same published version and HTTP boundary as UI.

    Every source is attempted. Expected source failures are logged; unexpected
    ones are annotated with the source and raised together as an ExceptionGroup,
    which the calling job logs once with every traceback.
    """
    message = f"published inference failed for project {project_id}"
    unexpected = []
    entries = bindings(project_id)
    _warn_unbound_sources(project_id)
    manager = (
        nullcontext(client)
        if client is not None
        else httpx.Client(timeout=10, follow_redirects=False, trust_env=False)
    )
    with manager as http:
        for entry in entries:
            source = entry["source"]
            try:
                score_source(entry, client=http)
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
    results = []
    unexpected = []
    for entry in bindings(project_id):
        source = entry["source"]
        try:
            check_source(source)
            results.append(
                train_candidate(
                    root,
                    source=source,
                    row_counts=source_storage.metrics(source, "row_count"),
                    null_rates=source_storage.metrics(source, "null_rate"),
                    git_sha=git_sha,
                )
            )
        except (ValueError, OperationalError, PoolTimeoutError):
            logger.exception("Candidate training failed for source=%s", source)
        except MemoryError:
            raise
        except Exception as exc:
            exc.add_note(f"source={source}")
            unexpected.append(exc)
    if unexpected:
        raise ExceptionGroup("candidate training failed", unexpected)
    return results
