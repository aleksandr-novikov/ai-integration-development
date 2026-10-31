"""Full source identity at the storage boundary, without guessing legacy owners."""

import json
import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy import text

from ml.inference_contract import canonical_ts
from ml.model_registry import validate_source

logger = logging.getLogger(__name__)

UNKNOWN = "legacy-unknown"
# Legacy scores and change-points are keyed by project and table. This is the
# training history of the legacy detector, i.e. the data such a result can mix.
LEGACY_HISTORY = timedelta(days=60)
RECENT_SOURCE = timedelta(days=7)
SOURCE_KEY = ["project_id", "connection_id", "schema_name", "table_name"]
SCORE_KEY = [*SOURCE_KEY, "ts", "model_version", "feature_schema_version"]
CACHE_KEY = [*SOURCE_KEY, "metric", "ts"]


class AmbiguousSourceError(ValueError):
    """Several stored observations resolve to the same source/metric/time."""


class InvalidHistoryError(ValueError):
    """Stored history contains an invalid timestamp."""


class ObservationNotFoundError(ValueError):
    """Attributed history exists, but nothing was observed at the requested time."""


def source_params(source):
    return validate_source(source)


def check_access(source):
    """Authorize history by its live connection, independently of ML bindings."""
    from app.metrics_storage import get_connection

    source = source_params(source)
    connection = get_connection(source["project_id"], source["connection_id"])
    if not connection or not connection["is_active"]:
        raise ValueError("source is unavailable")
    namespaces = {connection.get("iceberg_namespace") or connection["schema_name"]}
    if connection.get("iceberg_namespace_allowlist"):
        namespaces.update(json.loads(connection["iceberg_namespace_allowlist"]))
    if source["schema_name"] not in namespaces:
        raise ValueError("source namespace is unavailable")
    return source


def _source_of(project_id, table, identity):
    connection_id, schema_name = identity
    return {
        "project_id": project_id,
        "connection_id": connection_id,
        "schema_name": schema_name,
        "table_name": table,
    }


def _target(ts):
    from app.metrics_storage import stored_instant

    return stored_instant(canonical_ts(ts))


def _observers(conn, project_id, table, target, metric=None):
    """(connection, schema) pairs with a stored observation at the instant."""
    from app.metrics_storage import instant_range, stored_instant

    low, high = instant_range(target)
    params = {"pid": project_id, "table": table, "low": low, "high": high}
    metric_filter = ""
    if metric is not None:
        metric_filter = "AND metric_name = :metric"
        params["metric"] = metric
    rows = conn.execute(
        text(
            "SELECT connection_id, schema_name, ts FROM metrics "
            f"WHERE project_id = :pid AND table_name = :table {metric_filter} "
            "AND ts >= :low AND ts <= :high"
        ),
        params,
    ).fetchall()
    observers = set()
    for connection_id, schema_name, stamp in rows:
        instant = stored_instant(stamp)
        if instant is None and connection_id != UNKNOWN:
            raise InvalidHistoryError("invalid_history")
        if instant == target:
            observers.add((connection_id, schema_name))
    return observers


def _legacy_score_exists(conn, project_id, table, target):
    from app.metrics_storage import instant_range, stored_instant

    low, high = instant_range(target)
    rows = conn.execute(
        text(
            "SELECT ts FROM anomaly_scores WHERE project_id = :pid AND table_name = :table "
            "AND connection_id = 'legacy-unknown' AND schema_name = 'legacy-unknown' "
            "AND model_version = 'legacy-unknown' "
            "AND feature_schema_version = 'legacy-unknown' AND ts >= :low AND ts <= :high"
        ),
        {"pid": project_id, "table": table, "low": low, "high": high},
    ).fetchall()
    return any(stored_instant(row[0]) == target for row in rows)


def context_source(project_id, table, ts, metric=None):
    """Resolve the source of the observation at ``ts`` inside ``project_id``.

    Returns the live attributed source, or ``None`` for the project-level legacy
    context of a row without an owner: history is never reassigned, so a deleted
    connection's rows or old unattributed rows do not move to a live source.
    Several live sources need an explicit choice; a time observed only by a
    source the project cannot use is forbidden.
    """
    from app.metrics_storage import get_engine

    target = _target(ts)
    with get_engine().connect() as conn:
        observers = _observers(conn, project_id, table, target, metric)
    attributed = [item for item in observers if item[0] != UNKNOWN]
    allowed = []
    for identity in attributed:
        source = _source_of(project_id, table, identity)
        try:
            check_access(source)
        except ValueError:
            continue
        allowed.append(source)
    if len(allowed) > 1:
        raise AmbiguousSourceError("ambiguous source")
    if allowed:
        return allowed[0]
    if len(attributed) < len(observers):
        return None
    if attributed:
        raise ValueError("source is unavailable")
    with get_engine().connect() as conn:
        if _legacy_score_exists(conn, project_id, table, target):
            return None
        has_owner = conn.execute(
            text(
                "SELECT 1 FROM metrics WHERE project_id = :pid AND table_name = :table "
                "AND connection_id <> 'legacy-unknown' LIMIT 1"
            ),
            {"pid": project_id, "table": table},
        ).first()
    if has_owner:
        raise ObservationNotFoundError("observation not found")
    return None


def owns_project_results(source, ts):
    """True when project/table-level ML results at ``ts`` describe only ``source``.

    Such results were computed from every connection feeding the table name. They
    are attributed only when ``source`` alone observed the instant and no other
    attributed source fed that name during the legacy training history.
    """
    from app.metrics_storage import get_engine, instant_range

    source = source_params(source)
    identity = (source["connection_id"], source["schema_name"])
    target = _target(ts)
    low = instant_range(target, before=LEGACY_HISTORY)[0]
    high = instant_range(target)[1]
    with get_engine().connect() as conn:
        if _observers(conn, source["project_id"], source["table_name"], target) != {identity}:
            return False
        feeders = conn.execute(
            text(
                "SELECT DISTINCT connection_id, schema_name FROM metrics "
                "WHERE project_id = :pid AND table_name = :table AND metric_name = 'row_count' "
                "AND ts >= :low AND ts <= :high AND connection_id <> 'legacy-unknown'"
            ),
            {
                "pid": source["project_id"],
                "table": source["table_name"],
                "low": low,
                "high": high,
            },
        ).fetchall()
    return {tuple(row) for row in feeders} == {identity}


def recent_sources(project_id, table):
    """Live attributed sources that recently reported the table name."""
    from app.metrics_storage import get_engine

    with get_engine().connect() as conn:
        rows = conn.execute(
            text(
                "SELECT DISTINCT connection_id, schema_name FROM metrics "
                "WHERE project_id = :pid AND table_name = :table AND metric_name = 'row_count' "
                "AND ts >= :since AND connection_id <> 'legacy-unknown' "
                "ORDER BY connection_id, schema_name"
            ),
            {
                "pid": project_id,
                "table": table,
                "since": (datetime.now(UTC) - RECENT_SOURCE).isoformat(),
            },
        ).fetchall()
    sources = []
    for row in rows:
        source = _source_of(project_id, table, tuple(row))
        try:
            check_access(source)
        except ValueError:
            continue
        sources.append(source)
    return sources


def where_source():
    return " AND ".join(f"{name} = :{name}" for name in SOURCE_KEY)


def metrics(source, metric, window=timedelta(days=30), *, quarantine=False):
    from app.metrics_storage import get_engine

    params = {
        **source_params(source),
        "metric": metric,
        "since": (datetime.now(UTC) - window).isoformat(),
    }
    with get_engine().connect() as conn:
        rows = conn.execute(
            text(
                f"SELECT ts, value FROM metrics WHERE {where_source()} "
                "AND metric_name = :metric AND ts >= :since ORDER BY ts"
            ),
            params,
        ).fetchall()
    result = {}
    for ts, value in rows:
        try:
            stamp = canonical_ts(ts.isoformat() if isinstance(ts, datetime) else ts)
        except (ValueError, TypeError, AttributeError) as exc:
            if not quarantine:
                raise InvalidHistoryError("invalid_history") from exc
            logger.warning("Invalid history timestamp for source=%s metric=%s", source, metric)
            continue
        if stamp in result:
            if not quarantine:
                raise AmbiguousSourceError("ambiguous_source: duplicate observation")
            logger.warning(
                "ambiguous_source: duplicate history for source=%s metric=%s ts=%s",
                source,
                metric,
                stamp,
            )
            # Keep the timestamp as a barrier: neither this point nor its successor
            # may acquire a delta built from an arbitrarily chosen duplicate.
            value = None
        result[stamp] = {"ts": stamp, "value": value}
    return [result[stamp] for stamp in sorted(result)]


def snapshot(source):
    from app.metrics_storage import get_engine

    with get_engine().connect() as conn:
        row = conn.execute(
            text(f"SELECT columns FROM source_schema_snapshots WHERE {where_source()}"),
            source_params(source),
        ).fetchone()
    return json.loads(row[0]) if row else None


def save_snapshot(source, columns):
    from app.metrics_storage import _upsert_sql, get_engine

    sql = _upsert_sql(
        "source_schema_snapshots", [*SOURCE_KEY, "columns", "captured_at"], SOURCE_KEY
    )
    with get_engine().begin() as conn:
        conn.execute(
            text(sql),
            {
                **source_params(source),
                "columns": json.dumps(columns),
                "captured_at": datetime.now(UTC).isoformat(),
            },
        )


def save_prediction(result):
    from app.metrics_storage import save_anomaly_scores

    source = source_params(result["source"])
    return save_anomaly_scores(
        [
            {
                **result,
                **source,
                "ts": canonical_ts(result["observed_at"]),
            }
        ],
        project_id=source["project_id"],
    )


def insert_prediction(result):
    """Claim a new result atomically; retries/concurrent collectors never notify twice."""
    from app.metrics_storage import get_engine

    params = {
        **source_params(result["source"]),
        "ts": canonical_ts(result["observed_at"]),
        "model_version": result["model_version"],
        "feature_schema_version": result["feature_schema_version"],
        "score": result["score"],
        "is_anomaly": int(result["is_anomaly"]),
        "is_demo": int(result.get("demo", False)),
    }
    columns = [*SCORE_KEY, "score", "is_anomaly", "is_demo"]
    with get_engine().begin() as conn:
        inserted = conn.execute(
            text(
                f"INSERT INTO anomaly_scores ({', '.join(columns)}) "
                f"VALUES ({', '.join(':' + c for c in columns)}) "
                f"ON CONFLICT ({', '.join(SCORE_KEY)}) DO NOTHING"
            ),
            params,
        )
        return inserted.rowcount == 1


def scores(source, model_version, feature_schema_version, window=timedelta(days=30)):
    from app.metrics_storage import get_engine

    params = {
        **source_params(source),
        "model_version": model_version,
        "feature_schema_version": feature_schema_version,
        "since": (datetime.now(UTC) - window).isoformat(),
    }
    with get_engine().connect() as conn:
        rows = conn.execute(
            text(
                f"SELECT ts, score, is_anomaly, is_demo FROM anomaly_scores WHERE {where_source()} "
                "AND model_version = :model_version AND feature_schema_version = :feature_schema_version "
                "AND ts >= :since ORDER BY ts"
            ),
            params,
        ).fetchall()
    return [
        {
            "ts": canonical_ts(ts.isoformat() if isinstance(ts, datetime) else ts),
            "score": score,
            "is_anomaly": bool(flag),
            "source": dict(source),
            "model_version": model_version,
            "feature_schema_version": feature_schema_version,
            "demo": bool(demo),
        }
        for ts, score, flag, demo in rows
    ]


def _latest_score_ts(source, condition="", params=None):
    from app.metrics_storage import get_engine

    with get_engine().connect() as conn:
        stamp = conn.execute(
            text(f"SELECT MAX(ts) FROM anomaly_scores WHERE {where_source()} {condition}"),
            {**source_params(source), **(params or {})},
        ).scalar()
    if stamp is None:
        return None
    return canonical_ts(stamp.isoformat() if isinstance(stamp, datetime) else stamp)


def latest_scored_at(source):
    """High-water mark of processed observations, across every model version."""
    return _latest_score_ts(source)


def latest_anomaly_at(source, model_version, feature_schema_version):
    return _latest_score_ts(
        source,
        "AND model_version = :model_version "
        "AND feature_schema_version = :feature_schema_version AND is_anomaly = 1",
        {"model_version": model_version, "feature_schema_version": feature_schema_version},
    )


def versions(source):
    from app.metrics_storage import get_engine

    with get_engine().connect() as conn:
        rows = conn.execute(
            text(
                "SELECT DISTINCT model_version, feature_schema_version FROM anomaly_scores "
                f"WHERE {where_source()} ORDER BY model_version, feature_schema_version"
            ),
            source_params(source),
        ).fetchall()
    return [{"model_version": row[0], "feature_schema_version": row[1]} for row in rows]
