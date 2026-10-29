"""Full source identity at the storage boundary, without guessing legacy owners."""

import json
from datetime import UTC, datetime, timedelta

from sqlalchemy import text

from ml.inference_contract import canonical_ts
from ml.model_registry import validate_source

UNKNOWN = "legacy-unknown"
SOURCE_KEY = ["project_id", "connection_id", "schema_name", "table_name"]
SCORE_KEY = [*SOURCE_KEY, "ts", "model_version", "feature_schema_version"]
CACHE_KEY = [*SOURCE_KEY, "metric", "ts"]


class AmbiguousSourceError(ValueError):
    """Several stored observations resolve to the same source/metric/time."""


def source_params(source):
    return validate_source(source)


def context_source(project_id, table):
    """Resolve only explicitly attributed rows for trusted Telegram calls.

    Several sources never collapse into an arbitrary table-level context.
    Legacy rows stay in their own project context and are not backfilled.
    """
    from app.metrics_storage import get_connection, get_engine

    with get_engine().connect() as conn:
        rows = conn.execute(
            text(
                "SELECT DISTINCT connection_id, schema_name FROM metrics "
                "WHERE project_id = :pid AND table_name = :table "
                "AND connection_id <> 'legacy-unknown'"
            ),
            {"pid": project_id, "table": table},
        ).fetchall()
    if not rows:
        return None
    if len(rows) != 1:
        raise ValueError("ambiguous source")
    connection_id, schema_name = rows[0]
    connection = get_connection(project_id, connection_id)
    if not connection or not connection["is_active"]:
        raise ValueError("source is unavailable")
    namespaces = {connection.get("iceberg_namespace") or connection["schema_name"]}
    if connection.get("iceberg_namespace_allowlist"):
        namespaces.update(json.loads(connection["iceberg_namespace_allowlist"]))
    if schema_name not in namespaces:
        raise ValueError("source namespace is unavailable")
    return {
        "project_id": project_id,
        "connection_id": connection_id,
        "schema_name": schema_name,
        "table_name": table,
    }


def where_source():
    return " AND ".join(f"{name} = :{name}" for name in SOURCE_KEY)


def metrics(source, metric, window=timedelta(days=30)):
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
    result = []
    seen = set()
    for ts, value in rows:
        stamp = canonical_ts(ts.isoformat() if isinstance(ts, datetime) else ts)
        if stamp in seen:
            raise AmbiguousSourceError("ambiguous_source: duplicate observation")
        seen.add(stamp)
        result.append({"ts": stamp, "value": value})
    return result


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
            "ts": ts.isoformat() if isinstance(ts, datetime) else ts,
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
