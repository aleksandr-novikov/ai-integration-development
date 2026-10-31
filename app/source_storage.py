"""Full source identity at the storage boundary, without guessing legacy owners."""

import json
import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy import text

from ml.inference_contract import canonical_ts
from ml.model_registry import validate_source

logger = logging.getLogger(__name__)

UNKNOWN = "legacy-unknown"
SOURCE_KEY = ["project_id", "connection_id", "schema_name", "table_name"]
SCORE_KEY = [*SOURCE_KEY, "ts", "model_version", "feature_schema_version"]
CACHE_KEY = [*SOURCE_KEY, "metric", "ts"]


class AmbiguousSourceError(ValueError):
    """Several stored observations resolve to the same source/metric/time."""


class InvalidHistoryError(ValueError):
    """Stored history contains an invalid timestamp."""


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


def context_source(project_id, table, ts=None, metric=None):
    """Resolve live, explicitly attributed history at the requested observation.

    A deleted connection's history is never reassigned to its replacement.
    """
    from app.metrics_storage import get_engine

    with get_engine().connect() as conn:
        rows = conn.execute(
            text(
                "SELECT connection_id, schema_name, ts, metric_name FROM metrics "
                "WHERE project_id = :pid AND table_name = :table "
                "AND connection_id <> 'legacy-unknown'"
            ),
            {"pid": project_id, "table": table},
        ).fetchall()
    candidates = {}
    access = {}
    target = canonical_ts(ts) if ts is not None else None
    for connection_id, schema_name, stamp, name in rows:
        source = {
            "project_id": project_id,
            "connection_id": connection_id,
            "schema_name": schema_name,
            "table_name": table,
        }
        identity = (connection_id, schema_name)
        if identity not in access:
            try:
                check_access(source)
                access[identity] = True
            except ValueError:
                access[identity] = False
        if not access[identity]:
            continue
        if target is not None:
            try:
                normalized = canonical_ts(
                    stamp.isoformat() if isinstance(stamp, datetime) else stamp
                )
            except (ValueError, TypeError, AttributeError) as exc:
                raise InvalidHistoryError("invalid_history") from exc
            if normalized != target or (metric is not None and metric != name):
                continue
        candidates[(connection_id, schema_name)] = source
    if len(candidates) > 1:
        raise AmbiguousSourceError("ambiguous source")
    if candidates:
        return next(iter(candidates.values()))
    if rows:
        raise ValueError("source is unavailable")
    return None


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
