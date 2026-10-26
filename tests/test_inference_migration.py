"""Same data-preserving migration assertions on SQLite and PostgreSQL."""

from datetime import UTC, datetime

from sqlalchemy import create_engine, text

from app import metrics_storage as storage
from app import source_storage


def exercise_migration(engine, *, postgres=False):
    timestamp_type = "TIMESTAMPTZ" if postgres else "TEXT"
    float_type = "DOUBLE PRECISION" if postgres else "REAL"
    stamp = datetime.now(UTC).isoformat()
    with engine.begin() as conn:
        conn.execute(
            text(f"""CREATE TABLE anomaly_scores (
            project_id TEXT NOT NULL, ts {timestamp_type} NOT NULL,
            table_name TEXT NOT NULL, score {float_type} NOT NULL, is_anomaly INTEGER NOT NULL,
            PRIMARY KEY (project_id, ts, table_name))""")
        )
        conn.execute(
            text("INSERT INTO anomaly_scores VALUES ('project-a', :ts, 'orders', -0.1, 1)"),
            {"ts": stamp},
        )
        conn.execute(
            text(f"""CREATE TABLE llm_explanations (
            table_name TEXT NOT NULL, metric TEXT NOT NULL, ts {timestamp_type} NOT NULL,
            explanation TEXT NOT NULL, suggested_fix TEXT, confidence REAL,
            created_at {timestamp_type} NOT NULL, PRIMARY KEY (table_name, metric, ts))""")
        )
        conn.execute(
            text(
                "INSERT INTO llm_explanations VALUES ('orders', 'row_count', :ts, 'unowned secret', 'fix', 0.8, :ts)"
            ),
            {"ts": stamp},
        )
    storage._apply_schema(engine)
    storage._apply_schema(engine)
    with engine.connect() as conn:
        old = conn.execute(
            text(
                "SELECT model_version, feature_schema_version, connection_id, schema_name, score FROM anomaly_scores"
            )
        ).fetchone()
        assert tuple(old[:4]) == ("legacy-unknown",) * 4
        assert old[4] == -0.1
        assert (
            conn.execute(text("SELECT project_id FROM llm_explanations")).scalar()
            == "legacy-unowned"
        )
    assert storage.get_cached_explanation("orders", "row_count", stamp) is None
    source = {
        "project_id": "project-a",
        "connection_id": "a",
        "schema_name": "public",
        "table_name": "orders",
    }
    base = {
        "source": source,
        "observed_at": stamp,
        "model_version": "if-version-a",
        "feature_schema_version": "1.0.0",
        "score": -0.2,
        "is_anomaly": True,
    }
    source_storage.save_prediction(base)
    source_storage.save_prediction(base)
    source_storage.save_prediction({**base, "model_version": "if-version-b", "score": -0.3})
    source_storage.save_prediction({**base, "feature_schema_version": "2.0.0"})
    source_storage.save_prediction({**base, "source": {**source, "connection_id": "b"}})
    source_storage.save_prediction({**base, "source": {**source, "schema_name": "other"}})
    storage._apply_schema(engine)
    with engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM anomaly_scores")).scalar() == 6
    assert len(source_storage.scores(source, "if-version-a", "1.0.0")) == 1
    scope = {"project_id": "project-a", "connection_id": "a", "schema_name": "public"}
    storage.save_explanation("orders", "row_count", stamp, "A", "fix", 0.8, **scope)
    assert (
        storage.get_cached_explanation("orders", "row_count", stamp, **scope)["explanation"] == "A"
    )
    assert (
        storage.get_cached_explanation(
            "orders", "row_count", stamp, **{**scope, "project_id": "other"}
        )
        is None
    )
    assert (
        storage.get_cached_explanation("orders", "row_count", stamp, ttl_hours=-1, **scope) is None
    )


def test_existing_sqlite_is_migrated_without_loss(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'old.db'}"
    engine = create_engine(url)
    monkeypatch.setattr(storage.settings, "MONITOR_DB_URL", url)
    monkeypatch.setattr(storage, "_engine", engine)
    monkeypatch.setattr(storage, "_initialized", True)
    try:
        exercise_migration(engine)
    finally:
        engine.dispose()
