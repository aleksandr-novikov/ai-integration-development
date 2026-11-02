"""Overview and explanation source reads on a large temporary SQLite history.

Compare two checkouts on identical data; run from the root of each one:

    DATABASE_URL=sqlite:// PYTHONPATH=. python scripts/benchmark_history_queries.py
"""

import argparse
import json
import statistics
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

PROJECT = "benchmark-project"
TICKS = 56_940  # 4 connections x 2 metrics: 455 520 rows of one table name.
SCORED_TABLES, SCORED_TICKS = 25, 5_760  # 144 000 scores, five minutes apart.


def seed(storage, now):
    storage.create_user("benchmark", "benchmark@example.test", "unused")
    storage.create_project(PROJECT, "benchmark", "Benchmark", "benchmark")
    for index in range(4):
        storage.create_connection(
            connection_id=f"connection-{index}",
            project_id=PROJECT,
            name=f"C{index}",
            dsn_encrypted=b"unused",
        )
    rows = [
        (
            PROJECT,
            f"connection-{tick % 4}",
            "public",
            (now - timedelta(seconds=15 * tick)).isoformat(timespec="microseconds"),
            "orders",
            metric,
            1.0,
        )
        for tick in range(TICKS * 4)
        for metric in ("row_count", "null_rate")
    ]
    scores = [
        (
            PROJECT,
            (now - timedelta(minutes=5 * tick)).isoformat(timespec="seconds"),
            f"table_{table}",
            -0.1,
            tick % 2,
        )
        for table in range(SCORED_TABLES)
        for tick in range(SCORED_TICKS)
    ]
    engine = storage.get_engine()
    with engine.begin() as conn:
        raw = conn.connection
        raw.executemany(
            "INSERT INTO metrics (project_id, connection_id, schema_name, ts, table_name, "
            "metric_name, value) VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        raw.executemany(
            "INSERT INTO anomaly_scores (project_id, ts, table_name, score, is_anomaly) "
            "VALUES (?, ?, ?, ?, ?)",
            scores,
        )
    return rows[len(rows) // 2][3], len(rows), len(scores)


def measure(call, repeats):
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        samples.append(time.perf_counter() - started)
    return round(statistics.median(samples), 4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        from app import dashboard, source_storage
        from app import metrics_storage as storage

        storage.settings.MONITOR_DB_URL = f"sqlite:///{Path(directory) / 'history.db'}"
        stamp, metric_rows, score_rows = seed(storage, datetime.now(UTC))
        connections = storage.list_connections_for_project(PROJECT)
        result = {
            "metric_rows": metric_rows,
            "score_rows": score_rows,
            "seconds": {
                "project_status": measure(
                    lambda: dashboard._build_project_status(PROJECT, connections, True),
                    args.repeats,
                ),
                "ml_last_runs": measure(lambda: dashboard._ml_last_runs(PROJECT), args.repeats),
                "history_anomalies_by_ts": measure(
                    lambda: storage._fetch_anomalies_by_ts(PROJECT), args.repeats
                ),
                "context_source": measure(
                    lambda: source_storage.context_source(PROJECT, "orders", stamp, "row_count"),
                    args.repeats,
                ),
            },
        }
        storage.get_engine().dispose()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
